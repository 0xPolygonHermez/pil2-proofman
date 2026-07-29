use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;
use fields::PrimeField64;

use crate::pil_helpers::{Sha2Trace, Sha2TraceRow};

use super::{
    sha2_constants::{CLOCKS, CLOCKS_LOAD_INPUT, CLOCKS_LOAD_STATE, NUM_STEPS, RANGE_SIZE, RC},
    sha2_helpers::{big_sigma0, big_sigma1, bits32, ch, maj, random_sha2_input, range_row, small_sigma0, small_sigma1},
};

pub struct Sha2Air {
    num_available_sha2s: usize,
    instance_ids: RwLock<Vec<usize>>,
}

impl Sha2Air {
    pub fn new<F: PrimeField64>() -> Arc<Self> {
        let num_rows = Sha2Trace::<Sha2TraceRow<F>>::NUM_ROWS;
        let num_non_usable_rows = num_rows % CLOCKS;
        let num_available_sha2s = if num_non_usable_rows == 0 {
            num_rows / CLOCKS
        } else {
            // Subtract 1 because we can't fit a complete cycle in the remaining rows
            (num_rows - num_non_usable_rows) / CLOCKS - 1
        };

        Arc::new(Self { num_available_sha2s, instance_ids: RwLock::new(Vec::new()) })
    }

    /// Fill one SHA2-256 invocation and accumulate the lookup multiplicities
    #[allow(clippy::needless_range_loop)]
    fn process_trace<F: PrimeField64>(
        rows: &mut [Sha2TraceRow<F>],
        state: &[u32; 8],
        input: &[u32; 16],
        range_counts: &mut [u64],
    ) {
        // ── LOAD STATE: rows 0..4 hold [d,c,b,a] in s0 and [h,g,f,e] in s1 ──
        for i in 0..CLOCKS_LOAD_STATE {
            let row = &mut rows[i];
            row.set_all_s0(&bits32(state[3 - i]));
            row.set_all_s1(&bits32(state[7 - i]));
            range_counts[range_row(0, 0, 0)] += 1;
        }

        // ── LOAD INPUT & MIXING: rows 4..68, one mixing step per row ──
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        let mut w = [0u32; NUM_STEPS];
        w[..16].copy_from_slice(input);

        for i in 0..NUM_STEPS {
            let row = &mut rows[CLOCKS_LOAD_STATE + i];

            // Message schedule: rows 4..20 load the input; the rest extend it
            let w_carry = if i < CLOCKS_LOAD_INPUT {
                0u8
            } else {
                let w_full =
                    w[i - 16] as u64 + small_sigma0(w[i - 15]) as u64 + w[i - 7] as u64 + small_sigma1(w[i - 2]) as u64;
                w[i] = w_full as u32;
                (w_full >> 32) as u8
            };

            // Mixer
            let t1 = h as u64 + big_sigma1(e) as u64 + ch(e, f, g) as u64 + RC[i] as u64 + w[i] as u64;
            let t2 = big_sigma0(a) as u64 + maj(a, b, c) as u64;
            let new_a_full = t1 + t2;
            let new_e_full = d as u64 + t1;
            let s0_carry = (new_a_full >> 32) as u8;
            let s1_carry = (new_e_full >> 32) as u8;

            row.set_all_s0(&bits32(new_a_full as u32));
            row.set_all_s1(&bits32(new_e_full as u32));
            row.set_all_w(&bits32(w[i]));
            row.set_new_s0_carry_bits(s0_carry);
            row.set_new_s1_carry_bits(s1_carry);
            row.set_new_w_carry_bits(w_carry);
            range_counts[range_row(s0_carry, s1_carry, w_carry)] += 1;

            // advance the working state
            h = g;
            g = f;
            f = e;
            e = new_e_full as u32;
            d = c;
            c = b;
            b = a;
            a = new_a_full as u32;
        }

        // ── WRITE STATE: rows 68..72 hold state + [d,c,b,a] and state + [h,g,f,e] ──
        let out_s0 = [d, c, b, a];
        let out_s1 = [h, g, f, e];
        for i in 0..4 {
            let row = &mut rows[CLOCKS_LOAD_STATE + NUM_STEPS + i];

            let s0_full = state[3 - i] as u64 + out_s0[i] as u64;
            let s1_full = state[7 - i] as u64 + out_s1[i] as u64;
            let s0_carry = (s0_full >> 32) as u8;
            let s1_carry = (s1_full >> 32) as u8;

            row.set_all_s0(&bits32(s0_full as u32));
            row.set_all_s1(&bits32(s1_full as u32));
            row.set_new_s0_carry_bits(s0_carry);
            row.set_new_s1_carry_bits(s1_carry);
            range_counts[range_row(s0_carry, s1_carry, 0)] += 1;
        }
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Sha2Air {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(Sha2Trace::<F>::AIRGROUP_ID, Sha2Trace::<F>::AIR_ID)?;
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

        let num_sha2s: usize = self.num_available_sha2s;
        let num_available_sha2s = self.num_available_sha2s;

        let mut trace = Sha2Trace::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let num_rows = trace.num_rows();

        // Check that we can fit all the SHA2 inputs in the trace
        let num_rows_needed = num_sha2s * CLOCKS;
        let num_rows_covered = if num_sha2s < num_available_sha2s {
            num_rows_needed
        } else if num_sha2s == num_available_sha2s {
            num_rows
        } else {
            panic!(
                "Exceeded available SHA2 inputs: requested {}, but only {} are available.",
                num_sha2s, num_available_sha2s
            );
        };

        tracing::debug!(
            "··· Creating SHA2 instance with {} inputs (of {} available) [{} / {} rows filled {:.2}%]",
            num_sha2s,
            num_available_sha2s,
            num_rows_covered,
            num_rows,
            num_rows_covered as f64 / num_rows as f64 * 100.0
        );

        // Local multiplicity accumulator for the range checker
        let mut range_counts = vec![0u64; RANGE_SIZE];

        // 1] Fill one CLOCKS-row cycle per SHA2 and count its lookups.
        for k in 0..num_sha2s {
            let base = k * CLOCKS;
            let (state, input) = random_sha2_input(k as u64);
            Self::process_trace::<F>(&mut trace.buffer[base..base + CLOCKS], &state, &input, &mut range_counts);
        }

        // Padding
        // Unlike the Blake AIRs, the all-zero row does not satisfy the SHA2 constraints on
        // clocked cycles: the round constant k is a fixed column, so every complete cycle must
        // contain a valid computation. We fill the remaining cycles with the zero-input SHA2.
        let num_padding_blocks = num_available_sha2s - num_sha2s;
        if num_padding_blocks > 0 {
            let base = num_sha2s * CLOCKS;
            let mut zero_counts = vec![0u64; RANGE_SIZE];
            Self::process_trace::<F>(&mut trace.buffer[base..base + CLOCKS], &[0u32; 8], &[0u32; 16], &mut zero_counts);

            // Replicate the zero-input cycle across the remaining padding cycles
            let (head, tail) = trace.buffer.split_at_mut(base + CLOCKS);
            let zero_block = &head[base..base + CLOCKS];
            for k in 1..num_padding_blocks {
                tail[(k - 1) * CLOCKS..k * CLOCKS].copy_from_slice(zero_block);
            }

            for (t, &m) in zero_counts.iter().enumerate() {
                range_counts[t] += m * num_padding_blocks as u64;
            }
        }

        // The trailing rows where no clock fires range-check the all-zero carry triple
        let num_trailing_rows = num_rows - num_available_sha2s * CLOCKS;
        range_counts[range_row(0, 0, 0)] += num_trailing_rows as u64;

        // Write the multiplicity column
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
