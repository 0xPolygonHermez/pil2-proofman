//! The zerocheck PIOP: prove that every `EveryRow` constraint vanishes on the
//! whole hypercube via one sumcheck of
//!
//! `G(X) = eq(r, X) · Σ_t α^t · C_t(X)`,
//!
//! where `r` and `α` are transcript challenges sampled after the trace
//! commitments. The sumcheck terminates in evaluation claims on the committed
//! columns at the challenge point.

use crate::eq::{eq_evals, rotate_table};
use crate::evaluator::{constraint_value, eval_instrs, LeafSource};
use crate::hypercube::{fold_mle, to_ext_vec, Ext};
use crate::ir::{AirIr, Boundary};
use crate::sumcheck::SumcheckOracle;
use fields::Goldilocks;

/// Global column order used everywhere (commitments, claims matrix, batching):
/// stage-1 witness columns, stage-2 witness columns, …, then const columns.
pub fn global_col(ir: &AirIr, stage: u8, col: u32) -> usize {
    let mut base = 0usize;
    for s in 0..(stage as usize - 1) {
        base += ir.cols_per_stage[s] as usize;
    }
    base + col as usize
}

pub fn global_const_col(ir: &AirIr, col: u32) -> usize {
    ir.total_witness_cols() + col as usize
}

pub fn global_custom_col(ir: &AirIr, commit: u8, col: u32) -> usize {
    let mut base = ir.total_witness_cols() + ir.n_const_cols as usize;
    for c in 0..commit as usize {
        base += ir.custom_commits[c].n_cols as usize;
    }
    base + col as usize
}

/// The kernels of the batched opening, in canonical order: one rotation kernel
/// per opening offset (offset 0 degenerates to `eq(·, λ)`), then one
/// Boolean-point kernel per hypercube corner referenced by boundary constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSpec {
    /// `K(y) = eq((y − s) mod 2^n, λ)` — discharges `w̃^{→s}(λ)` claims.
    Rot(i32),
    /// `K(y) = eq(y, bits(row))` — discharges `w(row)` corner claims.
    Point(u64),
}

pub fn build_kernels(ir: &AirIr) -> Vec<KernelSpec> {
    let n_rows = 1u64 << ir.n_bits;
    let mut kernels: Vec<KernelSpec> = ir.opening_offsets.iter().map(|&s| KernelSpec::Rot(s)).collect();
    if ir.constraints.iter().any(|c| c.boundary == Boundary::FirstRow) {
        kernels.push(KernelSpec::Point(0));
    }
    if ir.constraints.iter().any(|c| c.boundary == Boundary::LastRow) {
        kernels.push(KernelSpec::Point(n_rows - 1));
    }
    kernels
}

/// Index of the kernel discharging offset-`s` claims.
pub fn kernel_index_of_offset(ir: &AirIr, offset: i32) -> usize {
    ir.offset_index(offset).expect("offset not in opening_offsets")
}

/// Index of the corner kernel for a boundary.
pub fn kernel_index_of_boundary(ir: &AirIr, kernels: &[KernelSpec], boundary: Boundary) -> usize {
    let n_rows = 1u64 << ir.n_bits;
    let row = match boundary {
        Boundary::FirstRow => 0,
        Boundary::LastRow => n_rows - 1,
        Boundary::EveryRow => unreachable!("EveryRow has no corner kernel"),
    };
    kernels.iter().position(|k| *k == KernelSpec::Point(row)).expect("corner kernel missing")
}

/// Batching weights: `alphas[t] = α^k` for the k-th `EveryRow` constraint,
/// zero for boundary constraints (they are checked at corners instead).
pub fn constraint_weights(ir: &AirIr, alpha: Ext) -> Vec<Ext> {
    let mut w = Vec::with_capacity(ir.constraints.len());
    let mut cur = Ext::ONE;
    for c in &ir.constraints {
        if c.boundary == Boundary::EveryRow {
            w.push(cur);
            cur *= alpha;
        } else {
            w.push(Ext::ZERO);
        }
    }
    w
}

/// Prover-side sumcheck oracle for `G(X) = eq(r,X) · Σ_t α^t C_t(X)`.
///
/// Keeps one extension-field table per `(column, offset)` leaf.
/// All tables fold together each round.
pub struct ZerocheckOracle<'a> {
    ir: &'a AirIr,
    tables: Vec<Vec<Ext>>,
    /// `[stage-1][col][offset_idx]` → index into `tables`.
    wit_index: Vec<Vec<Vec<usize>>>,
    /// `[col][offset_idx]` → index into `tables`.
    const_index: Vec<Vec<usize>>,
    /// `[commit][col][offset_idx]` → index into `tables`.
    custom_index: Vec<Vec<Vec<usize>>>,
    eq_table: Vec<Ext>,
    publics: Vec<Ext>,
    challenges: Vec<Ext>,
    air_values: Vec<Ext>,
    airgroup_values: Vec<Ext>,
    weights: Vec<Ext>,
    rounds_left: usize,
}

struct TablePoint<'o> {
    ir: &'o AirIr,
    wit_index: &'o [Vec<Vec<usize>>],
    const_index: &'o [Vec<usize>],
    custom_index: &'o [Vec<Vec<usize>>],
    vals: &'o [Ext],
    publics: &'o [Ext],
    challenges: &'o [Ext],
    air_values: &'o [Ext],
    airgroup_values: &'o [Ext],
}

impl LeafSource for TablePoint<'_> {
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Ext {
        let o = self.ir.offset_index(row_offset).expect("unknown offset");
        self.vals[self.wit_index[stage as usize - 1][col as usize][o]]
    }
    fn constant(&self, col: u32, row_offset: i32) -> Ext {
        let o = self.ir.offset_index(row_offset).expect("unknown offset");
        self.vals[self.const_index[col as usize][o]]
    }
    fn public(&self, idx: u32) -> Ext {
        self.publics[idx as usize]
    }
    fn challenge(&self, idx: u32) -> Ext {
        self.challenges[idx as usize]
    }
    fn air_value(&self, idx: u32) -> Ext {
        self.air_values[idx as usize]
    }
    fn airgroup_value(&self, idx: u32) -> Ext {
        self.airgroup_values[idx as usize]
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Ext {
        let o = self.ir.offset_index(row_offset).expect("unknown offset");
        self.vals[self.custom_index[commit as usize][col as usize][o]]
    }
}

impl<'a> ZerocheckOracle<'a> {
    /// `witness[stage-1]`/`consts`: column-major base-field columns.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ir: &'a AirIr,
        witness: &[Vec<Vec<Goldilocks>>],
        consts: &[Vec<Goldilocks>],
        customs: &[Vec<Vec<Goldilocks>>],
        publics: &[Goldilocks],
        challenges: &[Ext],
        air_values: &[Ext],
        airgroup_values: &[Ext],
        r: &[Ext],
        alpha: Ext,
    ) -> Self {
        let _n_rows = 1usize << ir.n_bits;
        let mut tables = Vec::new();

        let mut make_tables = |col: &[Goldilocks]| -> Vec<usize> {
            ir.opening_offsets
                .iter()
                .map(|&s| {
                    let table: Vec<Ext> = if s == 0 { to_ext_vec(col) } else { to_ext_vec(&rotate_shifted(col, s)) };
                    tables.push(table);
                    tables.len() - 1
                })
                .collect()
        };

        let mut wit_index = Vec::with_capacity(witness.len());
        for stage_cols in witness {
            wit_index.push(stage_cols.iter().map(|c| make_tables(c)).collect::<Vec<_>>());
        }
        let const_index: Vec<Vec<usize>> = consts.iter().map(|c| make_tables(c)).collect();
        let custom_index: Vec<Vec<Vec<usize>>> =
            customs.iter().map(|commit| commit.iter().map(|c| make_tables(c)).collect()).collect();

        Self {
            ir,
            tables,
            wit_index,
            const_index,
            custom_index,
            eq_table: eq_evals(r),
            publics: to_ext_vec(publics),
            challenges: challenges.to_vec(),
            air_values: air_values.to_vec(),
            airgroup_values: airgroup_values.to_vec(),
            weights: constraint_weights(ir, alpha),
            rounds_left: ir.n_bits as usize,
        }
    }
}

/// The shifted column as a table: `out[i] = col[(i + s) mod n]`.
fn rotate_shifted(col: &[Goldilocks], s: i32) -> Vec<Goldilocks> {
    // rotate_table computes out[y] = table[(y − s) mod n], so negate.
    rotate_table(col, -(s as i64))
}

impl SumcheckOracle for ZerocheckOracle<'_> {
    fn num_rounds(&self) -> usize {
        self.rounds_left
    }

    fn round_degree(&self) -> usize {
        self.ir.max_constraint_degree as usize + 1
    }

    fn round_evals(&self) -> Vec<Ext> {
        let n_evals = self.round_degree() + 1;
        let half = self.eq_table.len() / 2;
        let n_tables = self.tables.len();

        let mut g = vec![Ext::ZERO; n_evals];
        let mut vals = vec![Ext::ZERO; n_tables];
        let mut diffs = vec![Ext::ZERO; n_tables];
        let mut temps: Vec<Ext> = Vec::new();

        for i in 0..half {
            for (k, t) in self.tables.iter().enumerate() {
                vals[k] = t[2 * i];
                diffs[k] = t[2 * i + 1] - t[2 * i];
            }
            let eq0 = self.eq_table[2 * i];
            let eq_diff = self.eq_table[2 * i + 1] - eq0;

            let mut eq_x = eq0;
            for (x, gx) in g.iter_mut().enumerate() {
                if x > 0 {
                    for (v, d) in vals.iter_mut().zip(diffs.iter()) {
                        *v += *d;
                    }
                    eq_x += eq_diff;
                }
                let src = TablePoint {
                    ir: self.ir,
                    wit_index: &self.wit_index,
                    const_index: &self.const_index,
                    custom_index: &self.custom_index,
                    vals: &vals,
                    publics: &self.publics,
                    challenges: &self.challenges,
                    air_values: &self.air_values,
                    airgroup_values: &self.airgroup_values,
                };
                eval_instrs(self.ir, &src, &mut temps);
                let mut c = Ext::ZERO;
                for (t, w) in self.weights.iter().enumerate() {
                    if !w.is_zero() {
                        c += *w * constraint_value(self.ir, &src, &temps, t);
                    }
                }
                *gx += eq_x * c;
            }
        }
        g
    }

    fn bind(&mut self, r: Ext) {
        for t in self.tables.iter_mut() {
            fold_mle(t, r);
        }
        fold_mle(&mut self.eq_table, r);
        self.rounds_left -= 1;
    }
}

/// Verifier-side leaf source: reads the claimed openings matrix
/// (`claims[global_col][kernel]`), mapping a column's row offset to its
/// rotation kernel. Used for the final `G(λ)` consistency check.
pub struct ClaimsAtPoint<'a> {
    pub ir: &'a AirIr,
    pub claims: &'a [Vec<Ext>],
    pub publics: &'a [Ext],
    pub challenges: &'a [Ext],
    pub air_values: &'a [Ext],
    pub airgroup_values: &'a [Ext],
}

impl LeafSource for ClaimsAtPoint<'_> {
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Ext {
        self.claims[global_col(self.ir, stage, col)][kernel_index_of_offset(self.ir, row_offset)]
    }
    fn constant(&self, col: u32, row_offset: i32) -> Ext {
        self.claims[global_const_col(self.ir, col)][kernel_index_of_offset(self.ir, row_offset)]
    }
    fn public(&self, idx: u32) -> Ext {
        self.publics[idx as usize]
    }
    fn challenge(&self, idx: u32) -> Ext {
        self.challenges[idx as usize]
    }
    fn air_value(&self, idx: u32) -> Ext {
        self.air_values[idx as usize]
    }
    fn airgroup_value(&self, idx: u32) -> Ext {
        self.airgroup_values[idx as usize]
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Ext {
        self.claims[global_custom_col(self.ir, commit, col)][kernel_index_of_offset(self.ir, row_offset)]
    }
}

/// Verifier-side leaf source for boundary (corner) checks: every column
/// operand reads the corner kernel's claim. Boundary constraints must not
/// reference shifted columns (enforced at IR-compilation time).
pub struct ClaimsAtCorner<'a> {
    pub ir: &'a AirIr,
    pub claims: &'a [Vec<Ext>],
    pub publics: &'a [Ext],
    pub challenges: &'a [Ext],
    pub air_values: &'a [Ext],
    pub airgroup_values: &'a [Ext],
    pub kernel: usize,
}

impl LeafSource for ClaimsAtCorner<'_> {
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Ext {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        self.claims[global_col(self.ir, stage, col)][self.kernel]
    }
    fn constant(&self, col: u32, row_offset: i32) -> Ext {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        self.claims[global_const_col(self.ir, col)][self.kernel]
    }
    fn public(&self, idx: u32) -> Ext {
        self.publics[idx as usize]
    }
    fn challenge(&self, idx: u32) -> Ext {
        self.challenges[idx as usize]
    }
    fn air_value(&self, idx: u32) -> Ext {
        self.air_values[idx as usize]
    }
    fn airgroup_value(&self, idx: u32) -> Ext {
        self.airgroup_values[idx as usize]
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Ext {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        self.claims[global_custom_col(self.ir, commit, col)][self.kernel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::test_air::{fib_ir, fib_trace};
    use crate::hypercube::mle_eval;
    use crate::sumcheck::verify_sumcheck_round;
    use crate::basefold::MlParams;
    use fields::{Field, PrimeField64};
    use rand::{rng, RngExt};

    fn random_ext() -> Ext {
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    /// Zerocheck sumcheck on a valid Fibonacci trace: total sum must be 0,
    /// every round must verify, and the final claim must equal
    /// eq(r,λ)·Σ α^t C_t evaluated from the true column MLEs at λ.
    #[test]
    fn zerocheck_roundtrip_on_fib() {
        let n_bits = 4;
        let ir = fib_ir(n_bits, MlParams::default());
        let (witness, consts, publics) = fib_trace(n_bits);

        let r: Vec<Ext> = (0..n_bits).map(|_| random_ext()).collect();
        let alpha = random_ext();
        let mut oracle = ZerocheckOracle::new(&ir, &witness, &consts, &[], &publics, &[], &[], &[], &r, alpha);

        let mut claim = Ext::ZERO;
        let mut lambda = Vec::new();
        for round in 0..n_bits as usize {
            let evals = oracle.round_evals();
            let ch = random_ext();
            claim = verify_sumcheck_round(claim, &evals, ch, round).expect("zerocheck round");
            oracle.bind(ch);
            lambda.push(ch);
        }

        // Reconstruct G(λ) from the true MLEs.
        let n_rows = 1usize << n_bits;
        let col_a = &witness[0][0];
        let col_b = &witness[0][1];
        let shift = |c: &Vec<Goldilocks>| -> Vec<Goldilocks> { (0..n_rows).map(|i| c[(i + 1) % n_rows]).collect() };
        let ev = |c: &[Goldilocks]| mle_eval(&to_ext_vec(c), &lambda);
        let a_l = ev(col_a);
        let b_l = ev(col_b);
        let a_n = ev(&shift(col_a));
        let b_n = ev(&shift(col_b));
        let nl = ev(&consts[0]);
        let c1 = nl * (b_n - (a_l + b_l));
        let c2 = nl * (a_n - b_l);
        let expected = crate::eq::eq_eval(&r, &lambda) * (c1 + alpha * c2);
        assert_eq!(claim, expected);
    }

    /// A corrupted trace must make the zerocheck total nonzero.
    #[test]
    fn zerocheck_detects_invalid_trace() {
        let n_bits = 4;
        let ir = fib_ir(n_bits, MlParams::default());
        let (mut witness, consts, publics) = fib_trace(n_bits);
        witness[0][1][3] += Goldilocks::ONE;

        let r: Vec<Ext> = (0..n_bits).map(|_| random_ext()).collect();
        let alpha = random_ext();
        let oracle = ZerocheckOracle::new(&ir, &witness, &consts, &[], &publics, &[], &[], &[], &r, alpha);
        let evals = oracle.round_evals();
        // g(0) + g(1) = total sum over the hypercube ≠ 0 w.h.p.
        assert_ne!(evals[0] + evals[1], Ext::ZERO);
    }
}
