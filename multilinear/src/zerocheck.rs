//! The zerocheck PIOP: prove that every `EveryRow` constraint vanishes on the
//! whole hypercube via one sumcheck of
//!
//! `G(X) = eq(r, X) · Σ_t α^t · C_t(X)`,
//!
//! where `r` and `α` are transcript challenges sampled after the trace
//! commitments. The sumcheck terminates in evaluation claims on the committed
//! columns at the challenge point.

use crate::eq::{eq_evals, rotate_table};
use crate::evaluator::{constraint_value, eval_instrs, LeafSource, Val};
use crate::hypercube::{fold_mle, Ext};
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
    /// One table per `(column, offset)` leaf. Base-field until the first `bind`,
    /// then extension-field — so the dominant first round runs over `Goldilocks`.
    tables: Tables,
    /// `[stage-1][col][offset_idx]` → index into `tables`.
    wit_index: Vec<Vec<Vec<usize>>>,
    /// `[col][offset_idx]` → index into `tables`.
    const_index: Vec<Vec<usize>>,
    /// `[commit][col][offset_idx]` → index into `tables`.
    custom_index: Vec<Vec<Vec<usize>>>,
    eq_suffix: Vec<Ext>,
    publics: Vec<Goldilocks>,
    challenges: Vec<Ext>,
    air_values: Vec<Ext>,
    airgroup_values: Vec<Ext>,
    weights: Vec<Ext>,
    rounds_left: usize,
}

/// The `(column, offset)` leaf tables, base-field before the first `bind` and
/// extension-field after (folding by an extension challenge promotes them).
enum Tables {
    Base(Vec<Vec<Goldilocks>>),
    Ext(Vec<Vec<Ext>>),
}

impl Tables {
    #[inline]
    fn n_tables(&self) -> usize {
        match self {
            Tables::Base(t) => t.len(),
            Tables::Ext(t) => t.len(),
        }
    }
    /// Leaf value of table `k` at hypercube index `i`.
    #[inline]
    fn val(&self, k: usize, i: usize) -> Val {
        match self {
            Tables::Base(t) => Val::B(t[k][i]),
            Tables::Ext(t) => Val::E(t[k][i]),
        }
    }
}

struct TablePoint<'o> {
    ir: &'o AirIr,
    wit_index: &'o [Vec<Vec<usize>>],
    const_index: &'o [Vec<usize>],
    custom_index: &'o [Vec<Vec<usize>>],
    vals: &'o [Val],
    publics: &'o [Goldilocks],
    challenges: &'o [Ext],
    air_values: &'o [Ext],
    airgroup_values: &'o [Ext],
}

impl LeafSource for TablePoint<'_> {
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Val {
        let o = self.ir.offset_index(row_offset).expect("unknown offset");
        self.vals[self.wit_index[stage as usize - 1][col as usize][o]]
    }
    fn constant(&self, col: u32, row_offset: i32) -> Val {
        let o = self.ir.offset_index(row_offset).expect("unknown offset");
        self.vals[self.const_index[col as usize][o]]
    }
    fn public(&self, idx: u32) -> Val {
        Val::B(self.publics[idx as usize])
    }
    fn challenge(&self, idx: u32) -> Val {
        Val::E(self.challenges[idx as usize])
    }
    fn air_value(&self, idx: u32) -> Val {
        Val::E(self.air_values[idx as usize])
    }
    fn airgroup_value(&self, idx: u32) -> Val {
        Val::E(self.airgroup_values[idx as usize])
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Val {
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
        // Base-field tables: the first round runs entirely over `Goldilocks`.
        let mut tables: Vec<Vec<Goldilocks>> = Vec::new();

        let mut make_tables = |col: &[Goldilocks]| -> Vec<usize> {
            ir.opening_offsets
                .iter()
                .map(|&s| {
                    let table = if s == 0 { col.to_vec() } else { rotate_shifted(col, s) };
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
            tables: Tables::Base(tables),
            wit_index,
            const_index,
            custom_index,
            eq_suffix: eq_evals(&r[1..]),
            publics: publics.to_vec(),
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
        // The eq(.,.) polynomial does not count
        self.ir.max_constraint_degree as usize
    }

    fn round_evals(&self) -> Vec<Ext> {
        let n_evals = self.round_degree() + 1;
        let half = self.eq_suffix.len();
        let n_tables = self.tables.n_tables();

        let mut g = vec![Ext::ZERO; n_evals];
        let mut vals = vec![Val::zero(); n_tables];
        let mut diffs = vec![Val::zero(); n_tables];
        let mut temps: Vec<Val> = Vec::new();

        for j in 0..half {
            for k in 0..n_tables {
                let v0 = self.tables.val(k, 2 * j);
                vals[k] = v0;
                diffs[k] = self.tables.val(k, 2 * j + 1) - v0;
            }

            let w = self.eq_suffix[j];
            for (x, gx) in g.iter_mut().enumerate() {
                if x > 0 {
                    for (v, d) in vals.iter_mut().zip(diffs.iter()) {
                        *v = *v + *d;
                    }
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
                let mut c = Val::zero();
                for (t, wt) in self.weights.iter().enumerate() {
                    if !wt.is_zero() {
                        c = c + Val::E(*wt) * constraint_value(self.ir, &src, &temps, t);
                    }
                }
                *gx += (Val::E(w) * c).to_ext();
            }
        }
        g
    }

    fn bind(&mut self, r: Ext) {
        match &mut self.tables {
            // First bind: folding base tables by an extension challenge yields
            // extension tables (`e0 + r·(e1−e0)`), so subsequent rounds run in `Ext`.
            Tables::Base(base) => {
                let ext: Vec<Vec<Ext>> = base
                    .iter()
                    .map(|t| {
                        let half = t.len() / 2;
                        (0..half).map(|i| Ext::from_base(t[2 * i]) + r * (t[2 * i + 1] - t[2 * i])).collect()
                    })
                    .collect();
                self.tables = Tables::Ext(ext);
            }
            Tables::Ext(ext) => {
                for t in ext.iter_mut() {
                    fold_mle(t, r);
                }
            }
        }
        // The suffix eq shrinks by one variable per round
        let half = self.eq_suffix.len() / 2;
        for j in 0..half {
            self.eq_suffix[j] = self.eq_suffix[2 * j] + self.eq_suffix[2 * j + 1];
        }
        self.eq_suffix.truncate(half);
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
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Val {
        Val::E(self.claims[global_col(self.ir, stage, col)][kernel_index_of_offset(self.ir, row_offset)])
    }
    fn constant(&self, col: u32, row_offset: i32) -> Val {
        Val::E(self.claims[global_const_col(self.ir, col)][kernel_index_of_offset(self.ir, row_offset)])
    }
    fn public(&self, idx: u32) -> Val {
        Val::E(self.publics[idx as usize])
    }
    fn challenge(&self, idx: u32) -> Val {
        Val::E(self.challenges[idx as usize])
    }
    fn air_value(&self, idx: u32) -> Val {
        Val::E(self.air_values[idx as usize])
    }
    fn airgroup_value(&self, idx: u32) -> Val {
        Val::E(self.airgroup_values[idx as usize])
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Val {
        Val::E(self.claims[global_custom_col(self.ir, commit, col)][kernel_index_of_offset(self.ir, row_offset)])
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
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Val {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        Val::E(self.claims[global_col(self.ir, stage, col)][self.kernel])
    }
    fn constant(&self, col: u32, row_offset: i32) -> Val {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        Val::E(self.claims[global_const_col(self.ir, col)][self.kernel])
    }
    fn public(&self, idx: u32) -> Val {
        Val::E(self.publics[idx as usize])
    }
    fn challenge(&self, idx: u32) -> Val {
        Val::E(self.challenges[idx as usize])
    }
    fn air_value(&self, idx: u32) -> Val {
        Val::E(self.air_values[idx as usize])
    }
    fn airgroup_value(&self, idx: u32) -> Val {
        Val::E(self.airgroup_values[idx as usize])
    }
    fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Val {
        assert_eq!(row_offset, 0, "boundary constraints must not reference shifted columns");
        Val::E(self.claims[global_custom_col(self.ir, commit, col)][self.kernel])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::test_air::{fib_ir, fib_trace};
    use crate::hypercube::{mle_eval, to_ext_vec};
    use crate::sumcheck::interpolate_at;
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

        // Gruen inter-round check: (1−r_k)·g'(0) + r_k·g'(1) = prev.
        let mut claim = Ext::ZERO;
        let mut lambda = Vec::new();
        for round in 0..n_bits as usize {
            let evals = oracle.round_evals();
            let ch = random_ext();
            let rk = r[round];
            assert_eq!((Ext::ONE - rk) * evals[0] + rk * evals[1], claim, "round {round}");
            claim = interpolate_at(&evals, ch);
            oracle.bind(ch);
            lambda.push(ch);
        }

        // Reconstruct the batched constraint value at λ from the true MLEs. With
        // Gruen the eq(r,λ) factor is absorbed into the per-round checks, so the
        // final claim equals the batched value directly (no eq factor).
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
        let expected = c1 + alpha * c2;
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
        // The round-0 weighted combination equals Σ_X eq(r,X)·C(X) = C̃(r), which
        // is nonzero w.h.p. when the trace violates a constraint.
        assert_ne!((Ext::ONE - r[0]) * evals[0] + r[0] * evals[1], Ext::ZERO);
    }
}
