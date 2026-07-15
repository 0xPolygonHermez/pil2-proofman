use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;
use fields::PrimeField64;

use crate::pil_helpers::{Blake3Trace, Blake3TraceRow};

use super::{
    blake3_constants::{CLOCKS, CLOCKS_PER_ROUND, G_INDICES, NUM_G_PER_ROUND, RANGE_SIZE, ROUNDS, SIGMA, TABLE_SIZE},
    blake3_helpers::{limbs16, random_blake3_input, range_row, table_row, xor_rotr_full, xor_rotr_split},
};

pub struct Blake3Air {
    num_available_blake3s: usize,
    instance_ids: RwLock<Vec<usize>>,
}

impl Blake3Air {
    pub fn new<F: PrimeField64>() -> Arc<Self> {
        let num_rows = Blake3Trace::<Blake3TraceRow<F>>::NUM_ROWS;
        let num_non_usable_rows = num_rows % CLOCKS;
        let num_available_blake3s = if num_non_usable_rows == 0 {
            num_rows / CLOCKS
        } else {
            // Subtract 1 because we can't fit a complete cycle in the remaining rows
            (num_rows - num_non_usable_rows) / CLOCKS - 1
        };

        Arc::new(Self { num_available_blake3s, instance_ids: RwLock::new(Vec::new()) })
    }

    /// Fill one Blake3 invocation and accumulate the lookup multiplicities
    #[allow(clippy::needless_range_loop)]
    fn process_trace<F: PrimeField64>(
        rows: &mut [Blake3TraceRow<F>],
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

                // ── inputs ──
                row.set_all_va(&limbs16(va));
                row.set_all_vb(&vb.to_le_bytes());
                row.set_all_vc(&limbs16(vc));
                row.set_all_vd(&vd.to_le_bytes());
                row.set_all_x(&limbs16(x));
                row.set_all_y(&limbs16(y));

                // ── intermediates ──
                row.set_all_va_prime(&a1.to_le_bytes());
                row.set_all_vd_prime(&d1.to_le_bytes());
                row.set_all_vc_prime(&c1.to_le_bytes());
                row.set_all_va_prime_prime(&a2.to_le_bytes());
                row.set_all_vd_prime_prime(&d2.to_le_bytes());
                row.set_all_vc_prime_prime(&c2.to_le_bytes());

                // ── ROTR-by-12 split pieces and ROTR-by-7 lane-positioned values ──
                let vb_b = vb.to_le_bytes();
                let c1_b = c1.to_le_bytes();
                let b1_b = b1.to_le_bytes();
                let c2_b = c2.to_le_bytes();
                let mut vb_prime_s = [[0u8; 2]; 4];
                let mut vb_prime_prime_s = [0u32; 4];
                for i in 0..4 {
                    let (s0, s1) = xor_rotr_split(vb_b[i], c1_b[i], 12);
                    vb_prime_s[i] = [s0, s1];
                    vb_prime_prime_s[i] = xor_rotr_full(b1_b[i], c2_b[i], i, 7);
                }
                row.set_all_vb_prime_s(&vb_prime_s);
                row.set_all_vb_prime_prime_s(&vb_prime_prime_s);

                // ── lookup multiplicities ──

                // 16-bit range checks
                for w in [va, vc, x, y] {
                    let [lo, hi] = limbs16(w);
                    range_counts[range_row(lo)] += 1;
                    range_counts[range_row(hi)] += 1;
                }

                // XOR-rotate table
                let vd_b = vd.to_le_bytes();
                let a1_b = a1.to_le_bytes();
                let a2_b = a2.to_le_bytes();
                let d1_b = d1.to_le_bytes();
                for i in 0..4 {
                    table_counts[table_row(0, vd_b[i], a1_b[i], 0)] += 1; // (vd ^ a')  >>> 16
                    table_counts[table_row(0, vb_b[i], c1_b[i], 12)] += 1; // (vb ^ c')  >>> 12
                    table_counts[table_row(0, d1_b[i], a2_b[i], 0)] += 1; // (d' ^ a'') >>> 8
                    table_counts[table_row(i, b1_b[i], c2_b[i], 7)] += 1; // (b' ^ c'') >>> 7
                }

                // advance the working state
                v[ia] = a2;
                v[ib] = b2;
                v[ic] = c2;
                v[id] = d2;
            }
        }
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
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage != 1 {
            return Ok(());
        }

        let num_blake3s: usize = self.num_available_blake3s;
        let num_available_blake3s = self.num_available_blake3s;

        let mut trace = Blake3Trace::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let num_rows = trace.num_rows();

        // Check that we can fit all the BLAKE3 inputs in the trace
        let num_rows_needed = num_blake3s * CLOCKS;
        let num_rows_covered = if num_blake3s < num_available_blake3s {
            num_rows_needed
        } else if num_blake3s == num_available_blake3s {
            num_rows
        } else {
            panic!(
                "Exceeded available BLAKE3 inputs: requested {}, but only {} are available.",
                num_blake3s, num_available_blake3s
            );
        };

        tracing::debug!(
            "··· Creating BLAKE3 instance with {} inputs (of {} available) [{} / {} rows filled {:.2}%]",
            num_blake3s,
            num_available_blake3s,
            num_rows_covered,
            num_rows,
            num_rows_covered as f64 / num_rows as f64 * 100.0
        );

        // Local multiplicity accumulators for the tables
        let mut table_counts = vec![0u64; TABLE_SIZE];
        let mut range_counts = vec![0u64; RANGE_SIZE];

        // 1] Fill one CLOCKS-row cycle per Blake3 and count its lookups.
        for k in 0..num_blake3s {
            let base = k * CLOCKS;
            let (state, message) = random_blake3_input(k as u64);
            Self::process_trace::<F>(
                &mut trace.buffer[base..base + CLOCKS],
                &state,
                &message,
                &mut table_counts,
                &mut range_counts,
            );
        }

        // Padding
        let num_padding_rows = num_rows - num_rows_needed;

        // Perform the padding table checks. Each padding row does:
        //      · 8 - xor_rotr_check(offset: 0, a: 0, b: 0, rot: 0,  c0: 0, c1: 0)
        //      · 4 - xor_rotr_check(offset: 0, a: 0, b: 0, rot: 12, c0: 0, c1: 0)
        //      · 1 - xor_rotr_check(offset: i, a: 0, b: 0, rot: 7,  c0: 0, c1: 0) for each lane i
        table_counts[table_row(0, 0, 0, 0)] += (num_padding_rows * 8) as u64;
        table_counts[table_row(0, 0, 0, 12)] += (num_padding_rows * 4) as u64;
        for i in 0..4 {
            table_counts[table_row(i, 0, 0, 7)] += num_padding_rows as u64;
        }

        // Perform the padding range checks: 8 zero limbs
        let count_zeros = num_padding_rows * 8;
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

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        let instance_id = instance_ids[0];
        pctx.add_air_instance(air_instance, instance_id);
        Ok(())
    }
}
