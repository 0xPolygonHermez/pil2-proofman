use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use witness::WitnessComponent;
use pil_std_lib::Std;
use fields::PrimeField64;
use rayon::prelude::*;

use crate::{
    keccakf_constants::*,
    keccakf_helpers::{keccak_f_round, keccakf_bit_pos, keccakf_state_flatten, keccakf_state_from_linear},
    KeccakfTrace, KeccakfTable, KeccakfTraceRow, KeccakfTraceRowOps,
};

/// Witness component that proves a batch of *independent* Keccak-f[1600]
/// invocations over random inputs.
pub struct Keccakf<F: PrimeField64> {
    num_available_keccakfs: usize,
    std_lib: Arc<Std<F>>,
    instance_ids: RwLock<Vec<usize>>,
}

impl<F: PrimeField64> Keccakf<F> {
    pub fn new(std_lib: Arc<Std<F>>) -> Arc<Self> {
        let num_rows = KeccakfTrace::<KeccakfTraceRow<F>>::NUM_ROWS;
        let num_non_usable_rows = num_rows % CLOCKS;
        let num_available_keccakfs = if num_non_usable_rows == 0 {
            num_rows / CLOCKS
        } else {
            // Can't fit a complete cycle in the remaining rows.
            (num_rows - num_non_usable_rows) / CLOCKS - 1
        };

        Arc::new(Self { num_available_keccakfs, std_lib, instance_ids: RwLock::new(Vec::new()) })
    }
}

/// Deterministic, fast PRNG (SplitMix64) so runs are reproducible without
/// pulling in the `rand` crate.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a pseudo-random initial state for the keccakf at index `seed`.
fn random_state(seed: u64) -> [u64; 25] {
    let mut s = seed.wrapping_add(0x0123_4567_89AB_CDEF);
    let mut state = [0u64; 25];
    for word in state.iter_mut() {
        *word = splitmix64(&mut s);
    }
    state
}

/// Fill a single 25-row cycle from one initial state
#[allow(clippy::needless_range_loop)]
fn process_trace<F: PrimeField64, R: KeccakfTraceRowOps<F>>(rows: &mut [R], input: &[u64; 25]) {
    for i in 0..CLOCKS {
        rows[i].set_in_use(true);
    }

    // Row 0: the (random) input state.
    let mut state = keccakf_state_from_linear(input);
    let state_flat = keccakf_state_flatten(&state);

    let mut state_bits = [false; WIDTH];
    for (i, &val) in state_flat.iter().enumerate() {
        state_bits[i] = (val & 1) != 0;
    }
    rows[0].set_all_state(&state_bits);

    let mut accs = [0u32; NUM_CHUNKS];
    for r in 0..ROUNDS {
        // Apply the round in the unreduced expression domain.
        keccak_f_round(&mut state, r);
        let state_flat = keccakf_state_flatten(&state);

        // Pack the unreduced expressions into base-`BASE` accumulators.
        for i in 0..NUM_CHUNKS {
            let offset = i * TABLE_MAX_CHUNKS;
            let num_bits = std::cmp::min(TABLE_MAX_CHUNKS, WIDTH - offset);
            let mut acc = 0u32;
            for j in 0..num_bits {
                acc += (state_flat[offset + j] as u32) * POWS_BASE[j];
            }
            accs[i] = acc;
        }
        rows[r].set_all_chunk_acc(&accs);

        // Reduce modulo 2 and record the bits for the next row.
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    state[x][y][z] %= 2;
                    state_bits[keccakf_bit_pos(x, y, z)] = state[x][y][z] == 1;
                }
            }
        }
        rows[r + 1].set_all_state(&state_bits);
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Keccakf<F> {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(
            KeccakfTrace::<KeccakfTraceRow<F>>::AIRGROUP_ID,
            KeccakfTrace::<KeccakfTraceRow<F>>::AIR_ID,
        )?;
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

        let instance_id = instance_ids[0];
        let mut trace = KeccakfTrace::<KeccakfTraceRow<F>>::new_from_vec_zeroes(buffer_pool.take_buffer())?;

        // Fill every available slot with a random Keccak-f.
        let num_keccakfs = self.num_available_keccakfs;
        tracing::debug!("··· Generating {} random Keccak-f's", num_keccakfs);

        // 1] One 25-row cycle per keccakf, filled in parallel.
        let mut rows = &mut trace.buffer[..];
        let mut cycles: Vec<&mut [KeccakfTraceRow<F>]> = Vec::with_capacity(num_keccakfs);
        for _ in 0..num_keccakfs {
            let (head, tail) = rows.split_at_mut(CLOCKS);
            cycles.push(head);
            rows = tail;
        }
        cycles.par_iter_mut().enumerate().for_each(|(idx, cycle)| {
            let input = random_state(idx as u64);
            process_trace::<F, KeccakfTraceRow<F>>(cycle, &input);
        });

        // 2] Update the lookup-table multiplicities. Count locally first, then
        // push each touched row once to avoid contention.
        let table_id = self.std_lib.get_virtual_table_id(KECCAKF_TABLE_ID)?;
        let mut table = vec![0u32; TABLE_SIZE as usize];
        for k in 0..num_keccakfs {
            let base = k * CLOCKS;
            for r in 0..ROUNDS {
                for &acc in trace.buffer[base + r].get_all_chunk_acc().iter() {
                    let table_row = KeccakfTable::calculate_table_row(acc);
                    table[table_row as usize] += 1;
                }
            }
        }
        table.into_par_iter().enumerate().for_each(|(row, value)| {
            if value > 0 {
                self.std_lib.inc_virtual_row(table_id, row as u32, value);
            }
        });

        let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
        pctx.add_air_instance(air_instance, instance_id);
        Ok(())
    }
}
