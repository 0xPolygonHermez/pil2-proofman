use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanError, ProofmanResult, SetupCtx};
use proofman_witness::WitnessComponent;
use proofman_fields::PrimeField64;

use crate::pil_helpers::{Blake3Trace, Blake3TraceRow, Blake3TraceRowOps, Blake3TraceRowPacked, PACKED_INFO};

use super::{
    blake3_constants::{
        CLOCKS, CLOCKS_PER_ROUND, COLS_PER_LANE, G_INDICES, NUM_G_PER_ROUND, RANGE_SIZE, ROUNDS, SHARED_COLS, SIGMA,
        TABLE_SIZE,
    },
    blake3_helpers::{limbs16, random_blake3_input, range_row, table_row, xor_rotr_split},
};

pub struct Blake3Air {
    /// CLOCKS-row cycles that fit in the trace. Each cycle holds one Blake3 per lane.
    num_available_cycles: usize,
    instance_ids: RwLock<Vec<usize>>,
}

impl Blake3Air {
    pub fn new<F: PrimeField64>() -> Arc<Self> {
        let num_rows = Blake3Trace::<Blake3TraceRow<F>>::NUM_ROWS;
        let num_non_usable_rows = num_rows % CLOCKS;
        let num_available_cycles = if num_non_usable_rows == 0 {
            num_rows / CLOCKS
        } else {
            // Subtract 1 because we can't fit a complete cycle in the remaining rows
            (num_rows - num_non_usable_rows) / CLOCKS - 1
        };

        Arc::new(Self { num_available_cycles, instance_ids: RwLock::new(Vec::new()) })
    }

    /// Recover the PIL's `LANES` from the proving key's cm1 width, so the PIL stays the
    /// single source of truth for how many Blake3 lanes share a row.
    fn num_lanes<F: PrimeField64>(sctx: &SetupCtx<F>) -> ProofmanResult<usize> {
        let setup = sctx.get_setup(Blake3Trace::<F>::AIRGROUP_ID, Blake3Trace::<F>::AIR_ID)?;
        let lanes = Self::lanes_from_cm1_width(setup.stark_info.map_sections_n["cm1"] as usize, "proving key")?;

        // The generated trace row is sized for the LANES of whatever pilout the pil-helpers
        // were last generated from. A stale traces.rs would otherwise only surface as an
        // out-of-bounds panic inside a setter.
        if let Some(trace_lanes) = Self::trace_lanes::<F>()? {
            if trace_lanes != lanes {
                return Err(ProofmanError::InvalidSetup(format!(
                    "Blake3 trace was generated for {trace_lanes} lane(s) but the proving key has {lanes}; \
                     regenerate the pil-helpers from the current pilout"
                )));
            }
        }

        Ok(lanes)
    }

    /// Lanes the generated `Blake3TraceRow` was built for, read back from the pilout-derived
    /// column list. `None` when this air is not in `PACKED_INFO` (nothing to cross-check).
    fn trace_lanes<F: PrimeField64>() -> ProofmanResult<Option<usize>> {
        let Some((_, _, packed)) = PACKED_INFO.iter().find(|(airgroup_id, air_id, _)| {
            *airgroup_id == Blake3Trace::<F>::AIRGROUP_ID && *air_id == Blake3Trace::<F>::AIR_ID
        }) else {
            return Ok(None);
        };

        Self::lanes_from_cm1_width(packed.unpack_info.len(), "generated trace").map(Some)
    }

    fn lanes_from_cm1_width(n_cols: usize, source: &str) -> ProofmanResult<usize> {
        if n_cols < SHARED_COLS + COLS_PER_LANE || !(n_cols - SHARED_COLS).is_multiple_of(COLS_PER_LANE) {
            return Err(ProofmanError::InvalidSetup(format!(
                "Blake3 cm1 width {n_cols} from the {source} is not {SHARED_COLS} + a multiple of \
                 {COLS_PER_LANE}; it does not match this witness computation"
            )));
        }

        Ok((n_cols - SHARED_COLS) / COLS_PER_LANE)
    }

    /// Fill one Blake3 invocation into `lane` of `rows` and accumulate the lookup multiplicities
    #[allow(clippy::needless_range_loop)]
    fn process_trace<F: PrimeField64, R: Blake3TraceRowOps<F>>(
        rows: &mut [R],
        lane: usize,
        state: &[u32; 16],
        message: &[u32; 16],
        table_counts: &mut [u64],
        range_counts: &mut [u64],
    ) {
        let mut v = *state;

        for r in 0..ROUNDS {
            for g in 0..NUM_G_PER_ROUND {
                let row = &mut rows[r * CLOCKS_PER_ROUND + g];
                let [ia, ib, ic, id] = G_INDICES[g];
                let x = message[SIGMA[r][2 * g]];
                let y = message[SIGMA[r][2 * g + 1]];
                let (va, vb, vc, vd) = (v[ia], v[ib], v[ic], v[id]);

                // ── BLAKE3 G function ──
                let a1 = va.wrapping_add(vb).wrapping_add(x);
                let d1 = (vd ^ a1).rotate_right(16);
                let c1 = vc.wrapping_add(d1);
                let b1 = (vb ^ c1).rotate_right(12);
                let a2 = a1.wrapping_add(b1).wrapping_add(y);
                let d2 = (d1 ^ a2).rotate_right(8);
                let c2 = c1.wrapping_add(d2);
                let b2 = (b1 ^ c2).rotate_right(7);

                // ── 16-bit limbs of the inputs ──
                let (va_l, vc_l, x_l, y_l) = (limbs16(va), limbs16(vc), limbs16(x), limbs16(y));
                for i in 0..2 {
                    row.set_va(lane, i, va_l[i]);
                    row.set_vc(lane, i, vc_l[i]);
                    row.set_x(lane, i, x_l[i]);
                    row.set_y(lane, i, y_l[i]);
                }

                // ── byte limbs of the inputs and intermediates ──
                let vb_b = vb.to_le_bytes();
                let vd_b = vd.to_le_bytes();
                let a1_b = a1.to_le_bytes();
                let d1_b = d1.to_le_bytes();
                let c1_b = c1.to_le_bytes();
                let b1_b = b1.to_le_bytes();
                let a2_b = a2.to_le_bytes();
                let d2_b = d2.to_le_bytes();
                let c2_b = c2.to_le_bytes();
                let z = b1 ^ c2; // vb'' = rotl1(rotr8(z))
                let z_b = z.to_le_bytes();

                for i in 0..4 {
                    row.set_vb(lane, i, vb_b[i]);
                    row.set_vd(lane, i, vd_b[i]);
                    row.set_va_prime(lane, i, a1_b[i]);
                    row.set_vd_prime(lane, i, d1_b[i]);
                    row.set_vc_prime(lane, i, c1_b[i]);
                    row.set_va_prime_prime(lane, i, a2_b[i]);
                    row.set_vd_prime_prime(lane, i, d2_b[i]);
                    row.set_vc_prime_prime(lane, i, c2_b[i]);

                    // ── ROTR-by-12 split pieces and ROTR-by-7 XOR bytes ──
                    let (s0, s1) = xor_rotr_split(vb_b[i], c1_b[i], 12);
                    row.set_vb_prime_s(lane, i, 0, s0);
                    row.set_vb_prime_s(lane, i, 1, s1);
                    row.set_vb_pp_xor(lane, i, z_b[i]);

                    // ── XOR-rotate table multiplicities (the table is shared by all lanes) ──
                    table_counts[table_row(vd_b[i], a1_b[i], 0)] += 1; // (vd ^ a')  >>> 16
                    table_counts[table_row(vb_b[i], c1_b[i], 12)] += 1; // (vb ^ c')  >>> 12
                    table_counts[table_row(d1_b[i], a2_b[i], 0)] += 1; // (d' ^ a'') >>> 8
                    table_counts[table_row(b1_b[i], c2_b[i], 0)] += 1; // z = b' ^ c'' ((b' ^ c'') >>> 7 is z rotated via the carry bit)
                }
                // top bit of rotr8(z) is bit 7 of z's byte 0
                row.set_vb_pp_t(lane, (z >> 7) & 1 == 1);

                // ── 16-bit range checks ──
                for w in [va_l, vc_l, x_l, y_l] {
                    range_counts[range_row(w[0])] += 1;
                    range_counts[range_row(w[1])] += 1;
                }

                // advance the working state
                v[ia] = a2;
                v[ib] = b2;
                v[ic] = c2;
                v[id] = d2;
            }
        }
    }

    /// Fill the whole trace and wrap it as an air instance. Generic over the row type so the
    /// packed and unpacked layouts share one filler; every write goes through `Blake3TraceRowOps`.
    fn compute_witness_inner<F: PrimeField64, R: Blake3TraceRowOps<F>>(
        &self,
        lanes: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<AirInstance<F>> {
        let num_cycles = self.num_available_cycles;
        let num_blake3s = num_cycles * lanes;

        let mut trace = Blake3Trace::<R>::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let num_rows = trace.num_rows();

        let num_rows_used = num_cycles * CLOCKS;

        tracing::debug!(
            "··· Creating BLAKE3 instance with {} inputs ({} cycles x {} lanes) [{} / {} rows used {:.2}%]",
            num_blake3s,
            num_cycles,
            lanes,
            num_rows_used,
            num_rows,
            num_rows_used as f64 / num_rows as f64 * 100.0
        );

        // Local multiplicity accumulators for the tables
        let mut table_counts = vec![0u64; TABLE_SIZE];
        let mut range_counts = vec![0u64; RANGE_SIZE];

        // 1] Fill one CLOCKS-row cycle per lane-group of Blake3s and count their lookups.
        for cycle in 0..num_cycles {
            let base = cycle * CLOCKS;
            for lane in 0..lanes {
                let (state, message) = random_blake3_input((cycle * lanes + lane) as u64);
                Self::process_trace::<F, R>(
                    &mut trace.buffer[base..base + CLOCKS],
                    lane,
                    &state,
                    &message,
                    &mut table_counts,
                    &mut range_counts,
                );
            }
        }

        // Padding
        let num_padding_rows = num_rows - num_rows_used;

        // Perform the padding table checks. Each padding row does, per lane:
        //      · 12 - xor_rotr_check(a: 0, b: 0, rot: 0,  c0: 0, c1: 0)
        //      ·  4 - xor_rotr_check(a: 0, b: 0, rot: 12, c0: 0, c1: 0)
        let padding_lookups = num_padding_rows * lanes;
        table_counts[table_row(0, 0, 0)] += (padding_lookups * 12) as u64;
        table_counts[table_row(0, 0, 12)] += (padding_lookups * 4) as u64;

        // Perform the padding range checks: 8 zero limbs per lane
        let count_zeros = padding_lookups * 8;
        range_counts[range_row(0)] += count_zeros as u64;

        // Write the multiplicity columns
        for (t, &m) in table_counts.iter().enumerate() {
            if m != 0 {
                trace.buffer[t].set_mul_table(m);
            }
        }
        for (t, &m) in range_counts.iter().enumerate() {
            if m != 0 {
                trace.buffer[t].set_mul_range(m);
            }
        }

        Ok(AirInstance::new_from_trace(FromTrace::new(&mut trace)))
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Blake3Air {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(Blake3Trace::<F>::AIRGROUP_ID, Blake3Trace::<F>::AIR_ID)?;
        *self.instance_ids.write().unwrap() = vec![global_id];
        global_ids.write().unwrap().push(global_id);
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage != 1 {
            return Ok(());
        }

        // LANES independent Blake3s share every row, so a trace of `num_cycles` cycles
        // holds `num_cycles * lanes` permutations.
        let lanes = Self::num_lanes(&sctx)?;

        // Same filler either way -- only the row's storage layout differs.
        let air_instance = if pctx.is_packed(Blake3Trace::<F>::AIRGROUP_ID, Blake3Trace::<F>::AIR_ID) {
            self.compute_witness_inner::<F, Blake3TraceRowPacked<F>>(lanes, buffer_pool)?
        } else {
            self.compute_witness_inner::<F, Blake3TraceRow<F>>(lanes, buffer_pool)?
        };

        pctx.add_air_instance(air_instance, instance_ids[0]);
        Ok(())
    }
}
