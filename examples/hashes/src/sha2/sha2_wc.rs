//! Witness computation for the `Sha2` AIR (`pil/sha2.pil`).
//!
//! 72 rows per compression: rows 0..4 load the state (`s0 = d, c, b, a` and `s1 = h, g, f, e`, one
//! word per row from the bottom up), rows 4..68 run the 64 steps of the message schedule and the
//! mixer (rows 4..20 also load the message words into `w`), and rows 68..72 hold the output words
//! `state + final` in the order of the load rows.
//!
//! Every bit of `s0`, `s1` and `w` is stored as `1 - 2b` (+1 for a 0 bit, -1 for a 1 bit), next to
//! a packed shadow (`s0_pk`, `s1_pk`, `w_pk`) that holds the same word as a number.
//!
//! The AIR's three round equations hold on *every* row. Their carries are bits only where the
//! equation is meaningful (`IS_ROUND` for a/e, `IS_MIXING` for w); elsewhere the low carry cell
//! (`ca0`, `ce0`, `cw0`) is the field element that absorbs the difference. So the carries are
//! filled in a second pass, uniformly for all N rows, from the packed shadows of the first:
//!
//!     carry = (round terms of the previous rows - packed(this row)) / 2^32      (in the field)
//!
//! which is the true small carry on round rows and the absorbing value on the others (load, write
//! and leftover rows, and the first rows whose previous rows wrap around to the end of the trace).

use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use proofman_witness::WitnessComponent;
use proofman_fields::PrimeField64;

use crate::pil_helpers::{Sha2Trace, Sha2TraceRow, Sha2TraceRowOps, Sha2TraceRowPacked};

use super::{
    sha2_constants::{CLOCKS, CLOCKS_LOAD_INPUT, CLOCKS_LOAD_STATE, CLOCKS_WRITE_STATE, NUM_STEPS, RC},
    sha2_helpers::{big_sigma0, big_sigma1, bits32, ch, maj, random_sha2_input, small_sigma0, small_sigma1},
};

/// First row of a compression where the a/e round equations are enforced (`IS_ROUND`)
const FIRST_ROUND_ROW: usize = CLOCKS_LOAD_STATE;

/// First row of a compression where the w round equation is enforced (`IS_MIXING`)
const FIRST_MIXING_ROW: usize = CLOCKS_LOAD_STATE + CLOCKS_LOAD_INPUT;

/// First row of a compression that writes an output word (`IS_WRITE_STATE`)
const FIRST_WRITE_ROW: usize = CLOCKS - CLOCKS_WRITE_STATE;

/// The carry cells of one row
#[derive(Clone, Copy, Default)]
struct Carries {
    ca0: u64,
    ce0: u64,
    cw0: u64,
    ca1: bool,
    ce1: bool,
    ce2: bool,
    cw1: bool,
}

pub struct Sha2Air {
    num_available_sha2s: usize,
    instance_ids: RwLock<Vec<usize>>,
}

impl Sha2Air {
    pub fn new<F: PrimeField64>() -> Arc<Self> {
        let num_available_sha2s = Sha2Trace::<Sha2TraceRow<F>>::NUM_ROWS / CLOCKS;

        Arc::new(Self { num_available_sha2s, instance_ids: RwLock::new(Vec::new()) })
    }

    /// Fill the bit columns and their packed shadows for one SHA2-256 invocation. The carries of
    /// the round equations are filled afterwards by [`Self::carries`], over the whole trace.
    #[allow(clippy::needless_range_loop)]
    fn process_trace<F: PrimeField64, R: Sha2TraceRowOps<F>>(rows: &mut [R], state: &[u32; 8], input: &[u32; 16]) {
        // Every bit cell not written below (the w cells of the load and write rows) is a 0 bit
        for row in rows.iter_mut() {
            blank_row::<F, R>(row);
        }

        // ── LOAD STATE: rows 0..4 hold [d,c,b,a] in s0 and [h,g,f,e] in s1 ──
        for i in 0..CLOCKS_LOAD_STATE {
            set_s0::<F, R>(&mut rows[i], state[3 - i]);
            set_s1::<F, R>(&mut rows[i], state[7 - i]);
        }

        // ── LOAD INPUT & MIXING: rows 4..68, one step per row ──
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        let mut w = [0u32; NUM_STEPS];
        w[..CLOCKS_LOAD_INPUT].copy_from_slice(input);

        for i in 0..NUM_STEPS {
            let row = &mut rows[FIRST_ROUND_ROW + i];

            // Message schedule: rows 4..20 load the input, the rest extend it
            if i >= CLOCKS_LOAD_INPUT {
                w[i] = small_sigma1(w[i - 2])
                    .wrapping_add(w[i - 7])
                    .wrapping_add(small_sigma0(w[i - 15]))
                    .wrapping_add(w[i - 16]);
            }

            // Mixer
            let t1 = h.wrapping_add(big_sigma1(e)).wrapping_add(ch(e, f, g)).wrapping_add(RC[i]).wrapping_add(w[i]);
            let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
            let new_a = t1.wrapping_add(t2);
            let new_e = d.wrapping_add(t1);

            set_s0::<F, R>(row, new_a);
            set_s1::<F, R>(row, new_e);
            set_w::<F, R>(row, w[i]);

            // advance the working state
            h = g;
            g = f;
            f = e;
            e = new_e;
            d = c;
            c = b;
            b = a;
            a = new_a;
        }

        // ── WRITE STATE: rows 68..72 hold state + [d,c,b,a] and state + [h,g,f,e] ──
        let final_s0 = [d, c, b, a];
        let final_s1 = [h, g, f, e];
        for i in 0..CLOCKS_WRITE_STATE {
            let row = &mut rows[FIRST_WRITE_ROW + i];
            set_s0::<F, R>(row, state[3 - i].wrapping_add(final_s0[i]));
            set_s1::<F, R>(row, state[7 - i].wrapping_add(final_s1[i]));
        }
    }

    /// The carry cells of row `r`, read off the packed shadows (previous rows wrap around).
    ///
    /// Each round equation reads `packed + carry * 2^32 == terms`, so `carry = (terms - packed) /
    /// 2^32` in the field. Where the equation is enforced (`IS_ROUND` for a/e, `IS_MIXING` for w)
    /// that is a small integer, split into the bit cells. Elsewhere only the write rows put
    /// something in the higher bits -- their one-bit output carries, in `ca1` / `ce1` -- and the
    /// low cell takes the rest.
    fn carries<F: PrimeField64, R: Sha2TraceRowOps<F>>(rows: &[R], r: usize, num_ops: usize, inv_p2_32: F) -> Carries {
        let n = rows.len();
        let back = |k: usize| &rows[(r + n - k) % n];
        let cur = &rows[r];
        let (s0_pk, s1_pk, w_pk) = (cur.get_s0_pk(), cur.get_s1_pk(), cur.get_w_pk());
        let (a, b, c, d) = (back(1).get_s0_pk(), back(2).get_s0_pk(), back(3).get_s0_pk(), back(4).get_s0_pk());
        let (e, f, g, h) = (back(1).get_s1_pk(), back(2).get_s1_pk(), back(3).get_s1_pk(), back(4).get_s1_pk());
        let (w2, w7, w15, w16) = (back(2).get_w_pk(), back(7).get_w_pk(), back(15).get_w_pk(), back(16).get_w_pk());

        // Which of the three equations the AIR enforces here. The fixed selectors are 0 on the
        // leftover rows past the last whole compression, so nothing is enforced there.
        let (block, clk) = (r / CLOCKS, r % CLOCKS);
        let clocked = block < num_ops;
        let is_round = clocked && (FIRST_ROUND_ROW..FIRST_WRITE_ROW).contains(&clk);
        let is_mixing = clocked && (FIRST_MIXING_ROW..FIRST_WRITE_ROW).contains(&clk);
        let is_write_state = clocked && clk >= FIRST_WRITE_ROW;
        let k = if is_round { RC[clk - FIRST_ROUND_ROW] } else { 0 };

        // new e = d + h + Σ₁(e) + ch(e,f,g) + k + w                    (six terms < 2^32)
        let new_e = d as u64 + h as u64 + big_sigma1(e) as u64 + ch(e, f, g) as u64 + k as u64 + w_pk as u64;
        // new a = new e - d + Σ₀(a) + maj(a,b,c)      (in (-2^32, 3 * 2^32), carry offset by one)
        let new_a = s1_pk as i64 - d as i64 + big_sigma0(a) as i64 + maj(a, b, c) as i64;
        // new w = σ₁(w[i-2]) + w[i-7] + σ₀(w[i-15]) + w[i-16]         (four terms < 2^32)
        let new_w = small_sigma1(w2) as u64 + w7 as u64 + small_sigma0(w15) as u64 + w16 as u64;

        let ce_total = F::from_i64(new_e as i64 - s1_pk as i64) * inv_p2_32;
        let ca_total = F::from_i64(new_a - s0_pk as i64) * inv_p2_32 + F::ONE;
        let cw_total = F::from_i64(new_w as i64 - w_pk as i64) * inv_p2_32;

        // The higher carry bits: the split of a small carry on the rows that enforce the equation,
        // the output carry of `state + final` on the write rows, and 0 everywhere else.
        let mut out = Carries::default();
        if is_round {
            let ce = small_carry(ce_total, 3, "ce");
            (out.ce1, out.ce2) = ((ce >> 1) & 1 == 1, (ce >> 2) & 1 == 1);
            out.ca1 = (small_carry(ca_total, 2, "ca") >> 1) & 1 == 1;
        } else if is_write_state {
            let state_s0 = back(CLOCKS - CLOCKS_WRITE_STATE).get_s0_pk();
            let state_s1 = back(CLOCKS - CLOCKS_WRITE_STATE).get_s1_pk();
            out.ca1 = state_s0 as u64 + d as u64 >= 1 << 32;
            out.ce1 = state_s1 as u64 + h as u64 >= 1 << 32;
        }
        if is_mixing {
            out.cw1 = (small_carry(cw_total, 2, "cw") >> 1) & 1 == 1;
        }

        // The low cell takes whatever the higher bits leave over: the low carry bit where the
        // equation is enforced, the absorbing field element everywhere else.
        out.ca0 = (ca_total - F::from_bool(out.ca1).double()).as_canonical_u64();
        out.ce0 =
            (ce_total - F::from_bool(out.ce1).double() - F::from_bool(out.ce2).double().double()).as_canonical_u64();
        out.cw0 = (cw_total - F::from_bool(out.cw1).double()).as_canonical_u64();
        out
    }

    /// Fill the whole trace and wrap it as an air instance. Generic over the row type so the
    /// packed and unpacked layouts share one filler; every write goes through `Sha2TraceRowOps`.
    fn compute_witness_inner<F: PrimeField64, R: Sha2TraceRowOps<F>>(
        &self,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<AirInstance<F>> {
        // One count today: every available slot is filled. A requested count, when there is one,
        // would make the padding below reachable.
        let num_available_sha2s = self.num_available_sha2s;

        let mut trace = Sha2Trace::<R>::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let num_rows = trace.num_rows();

        // Check that we can fit all the SHA2 inputs in the trace
        let num_rows_needed = num_available_sha2s * CLOCKS;

        tracing::debug!(
            "··· Creating SHA2 instance with {} inputs [{} rows, {:.2}% filled]",
            num_available_sha2s,
            num_rows,
            num_rows_needed as f64 / num_rows as f64 * 100.0
        );

        // 1] Fill one CLOCKS-row cycle per SHA2.
        //
        // Every complete cycle is filled here, so there are no padding cycles. If a requested
        // count is ever plumbed in, they cannot be left zero: unlike the Blake AIRs the all-zero
        // row does not satisfy the SHA2 constraints on clocked cycles (the round constant is a
        // fixed column and the bits are ±1), so each spare cycle needs a real SHA2 -- the
        // compression of the zero state and input -- written into it.
        for k in 0..num_available_sha2s {
            let base = k * CLOCKS;
            let (state, input) = random_sha2_input(k as u64);
            Self::process_trace::<F, R>(&mut trace.buffer[base..base + CLOCKS], &state, &input);
        }

        // 2] The leftover rows past the last whole cycle: no selector fires there, so 0 bits (+1)
        // and the absorbing carries of step 3 are all they need.
        for row in trace.buffer[num_rows_needed..].iter_mut() {
            blank_row::<F, R>(row);
        }

        // 3] Fill the carries of the three round equations, on every row of the trace.
        let inv_p2_32 = F::from_u64(1u64 << 32).inverse();
        for r in 0..num_rows {
            let c = Self::carries::<F, R>(&trace.buffer[..], r, num_available_sha2s, inv_p2_32);
            let row = &mut trace.buffer[r];
            row.set_ca0(c.ca0);
            row.set_ce0(c.ce0);
            row.set_cw0(c.cw0);
            row.set_ca1(c.ca1);
            row.set_ce1(c.ce1);
            row.set_ce2(c.ce2);
            row.set_cw1(c.cw1);
        }

        Ok(AirInstance::new_from_trace(FromTrace::new(&mut trace)))
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

        // Same filler either way -- only the row's storage layout differs.
        let air_instance = if pctx.is_packed(Sha2Trace::<F>::AIRGROUP_ID, Sha2Trace::<F>::AIR_ID) {
            self.compute_witness_inner::<F, Sha2TraceRowPacked<F>>(buffer_pool)?
        } else {
            self.compute_witness_inner::<F, Sha2TraceRow<F>>(buffer_pool)?
        };

        pctx.add_air_instance(air_instance, instance_ids[0]);
        Ok(())
    }
}

/// The bits of `x` as +1 (bit 0) / -1 (bit 1), low bit first, as canonical field values
#[inline]
fn pm_bits<F: PrimeField64>(x: u32) -> [u64; 32] {
    let neg_one = F::NEG_ONE.as_canonical_u64();
    bits32(x).map(|bit| if bit { neg_one } else { 1 })
}

/// Sets every bit cell of the row to +1 (a 0 bit) and every packed shadow to the word 0
#[inline]
fn blank_row<F: PrimeField64, R: Sha2TraceRowOps<F>>(row: &mut R) {
    set_s0::<F, R>(row, 0);
    set_s1::<F, R>(row, 0);
    set_w::<F, R>(row, 0);
}

/// Writes a word into the `s0` bits and its packed shadow, which must never drift apart
#[inline]
fn set_s0<F: PrimeField64, R: Sha2TraceRowOps<F>>(row: &mut R, x: u32) {
    row.set_all_s0(&pm_bits::<F>(x));
    row.set_s0_pk(x);
}

/// Writes a word into the `s1` bits and its packed shadow
#[inline]
fn set_s1<F: PrimeField64, R: Sha2TraceRowOps<F>>(row: &mut R, x: u32) {
    row.set_all_s1(&pm_bits::<F>(x));
    row.set_s1_pk(x);
}

/// Writes a word into the `w` bits and its packed shadow
#[inline]
fn set_w<F: PrimeField64, R: Sha2TraceRowOps<F>>(row: &mut R, x: u32) {
    row.set_all_w(&pm_bits::<F>(x));
    row.set_w_pk(x);
}

/// A carry the AIR range-checks as `bits` bits, as a small integer
#[inline]
fn small_carry<F: PrimeField64>(total: F, bits: u32, what: &str) -> u64 {
    let v = total.as_canonical_u64();
    assert!(v < 1 << bits, "sha2: {what} carry {v} does not fit in {bits} bits");
    v
}
