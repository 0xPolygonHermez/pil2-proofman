use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;
use fields::PrimeField64;

use crate::pil_helpers::{Blake2bTrace, Blake2bTraceRow};

use super::{
    blake2b_constants::{CLOCKS, CLOCKS_PER_ROUND, G_INDICES, NUM_G_PER_ROUND, RANGE_SIZE, ROUNDS, SIGMA, TABLE_SIZE},
    blake2b_helpers::{limbs16, random_blake2b_input, range_row, table_row},
};

pub struct Blake2bAir {
    num_available_blake2bs: usize,
    instance_ids: RwLock<Vec<usize>>,
}

impl Blake2bAir {
    pub fn new<F: PrimeField64>() -> Arc<Self> {
        let num_rows = Blake2bTrace::<Blake2bTraceRow<F>>::NUM_ROWS;
        let num_non_usable_rows = num_rows % CLOCKS;
        let num_available_blake2bs = if num_non_usable_rows == 0 {
            num_rows / CLOCKS
        } else {
            // Subtract 1 because we can't fit a complete cycle in the remaining rows
            (num_rows - num_non_usable_rows) / CLOCKS - 1
        };

        Arc::new(Self { num_available_blake2bs, instance_ids: RwLock::new(Vec::new()) })
    }

    /// Fill one Blake2b invocation and accumulate the lookup multiplicities
    #[allow(clippy::needless_range_loop)]
    fn process_trace<F: PrimeField64>(
        rows: &mut [Blake2bTraceRow<F>],
        state: &[u64; 16],
        message: &[u64; 16],
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

                // ── BLAKE2b G function ──
                let a1 = va.wrapping_add(vb).wrapping_add(x);
                let d1 = (vd ^ a1).rotate_right(32);
                let c1 = vc.wrapping_add(d1);
                let b1 = (vb ^ c1).rotate_right(24);
                let a2 = a1.wrapping_add(b1).wrapping_add(y);
                let d2 = (d1 ^ a2).rotate_right(16);
                let c2 = c1.wrapping_add(d2);
                let b2 = (b1 ^ c2).rotate_right(63);

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
                row.set_all_vb_prime(&b1.to_le_bytes());
                row.set_all_va_prime_prime(&a2.to_le_bytes());
                row.set_all_vd_prime_prime(&d2.to_le_bytes());
                row.set_all_vc_prime_prime(&c2.to_le_bytes());

                // ── ROTR-by-63 as a doubling: z = vb' ^ vc'' plus the two limb top bits ──
                let z = b1 ^ c2;
                row.set_all_vb_pp_xor(&z.to_le_bytes());
                row.set_all_vb_pp_t(&[(z >> 31) & 1 == 1, (z >> 63) & 1 == 1]);

                // ── lookup multiplicities ──

                // 16-bit range checks
                for w in [va, vc, x, y] {
                    for limb in limbs16(w) {
                        range_counts[range_row(limb)] += 1;
                    }
                }

                // XOR table
                let vb_b = vb.to_le_bytes();
                let vd_b = vd.to_le_bytes();
                let a1_b = a1.to_le_bytes();
                let c1_b = c1.to_le_bytes();
                let a2_b = a2.to_le_bytes();
                let d1_b = d1.to_le_bytes();
                let b1_b = b1.to_le_bytes();
                let c2_b = c2.to_le_bytes();
                for i in 0..8 {
                    table_counts[table_row(vd_b[i], a1_b[i])] += 1; // (vd ^ a')  >>> 32
                    table_counts[table_row(vb_b[i], c1_b[i])] += 1; // (vb ^ c')  >>> 24
                    table_counts[table_row(d1_b[i], a2_b[i])] += 1; // (d' ^ a'') >>> 16
                    table_counts[table_row(b1_b[i], c2_b[i])] += 1; // (b' ^ c'') >>> 63
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

impl<F: PrimeField64> WitnessComponent<F> for Blake2bAir {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(Blake2bTrace::<F>::AIRGROUP_ID, Blake2bTrace::<F>::AIR_ID)?;
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

        let num_blake2bs: usize = self.num_available_blake2bs;
        let num_available_blake2bs = self.num_available_blake2bs;

        let mut trace = Blake2bTrace::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let num_rows = trace.num_rows();

        // Check that we can fit all the BLAKE2b inputs in the trace
        let num_rows_needed = num_blake2bs * CLOCKS;
        let num_rows_covered = if num_blake2bs < num_available_blake2bs {
            num_rows_needed
        } else if num_blake2bs == num_available_blake2bs {
            num_rows
        } else {
            panic!(
                "Exceeded available BLAKE2b inputs: requested {}, but only {} are available.",
                num_blake2bs, num_available_blake2bs
            );
        };

        tracing::debug!(
            "··· Creating BLAKE2b instance with {} inputs (of {} available) [{} / {} rows filled {:.2}%]",
            num_blake2bs,
            num_available_blake2bs,
            num_rows_covered,
            num_rows,
            num_rows_covered as f64 / num_rows as f64 * 100.0
        );

        // Local multiplicity accumulators for the tables
        let mut table_counts = vec![0u64; TABLE_SIZE];
        let mut range_counts = vec![0u64; RANGE_SIZE];

        // 1] Fill one CLOCKS-row cycle per Blake2b and count its lookups.
        for k in 0..num_blake2bs {
            let base = k * CLOCKS;
            let (state, message) = random_blake2b_input(k as u64);
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
        //      · 32 - xor_check(a: 0, b: 0, c: 0)
        table_counts[table_row(0, 0)] += (num_padding_rows * 32) as u64;

        // Perform the padding range checks: 16 zero limbs
        let count_zeros = num_padding_rows * 16;
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
