//! The zerocheck PIOP: Given prismalinear polynomials w_1,...,w_k ∈ F[Z,X₁,...,Xₘ],
//! and constraints C_1,...,C_t, prove that
//!
//! `C_t((w_j^{(s)}(y))_{j,s}) = 0` for all `t` and `y` in the hyperprism `D_m`,
//!
//! This is proven using a sumcheck over `D^m` of the polynomial
//!
//! `G(X) = ∑_{y ∈ D_m} eq(X,y)·Σ_t α^{t-1} C_t((w_j^{(s)}(y))_{j,s})`,
//!
//! where `α` and `r` are random challenges.

use crate::eq::{d_subgroup, eq_evals, lagrange_d, rotate_table};
use crate::evaluator::{constraint_value, eval_instrs, LeafSource, Val};
use crate::hypercube::{fold_mle, Ext};
use crate::ir::{AirIr, Boundary};
use crate::sumcheck::{lagrange_eval, SumcheckOracle};
use fields::{Goldilocks, PrimeField64};

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
    wtn_index: Vec<Vec<Vec<usize>>>,
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
    skip_bits: usize,
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
    wtn_index: &'o [Vec<Vec<usize>>],
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
        self.vals[self.wtn_index[stage as usize - 1][col as usize][o]]
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
        skip_bits: usize,
    ) -> Self {
        // Tables over which constraints are defined
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

        let mut wtn_index = Vec::with_capacity(witness.len());
        for stage_cols in witness {
            wtn_index.push(stage_cols.iter().map(|c| make_tables(c)).collect::<Vec<_>>());
        }
        let const_index: Vec<Vec<usize>> = consts.iter().map(|c| make_tables(c)).collect();
        let custom_index: Vec<Vec<Vec<usize>>> =
            customs.iter().map(|commit| commit.iter().map(|c| make_tables(c)).collect()).collect();

        // Compute eq(x, r_{l+1..m}):
        //      - skip_bits = 0: the first round binds one variable, so drop r₁ → eq over r_{2..m}
        //      - skip_bits = l: round 0 sums over the whole suffix weighted by eq(x, r_{l+1..m}), so keep the full suffix eq.
        let eq_suffix = if skip_bits == 0 { eq_evals(&r[1..]) } else { eq_evals(&r[skip_bits..]) };

        Self {
            ir,
            tables: Tables::Base(tables),
            wtn_index,
            const_index,
            custom_index,
            eq_suffix,
            publics: publics.to_vec(),
            challenges: challenges.to_vec(),
            air_values: air_values.to_vec(),
            airgroup_values: airgroup_values.to_vec(),
            weights: constraint_weights(ir, alpha),
            rounds_left: ir.n_bits as usize,
            skip_bits,
        }
    }

    /// Univariate-skip round 0: the polynomial `v(X) = Σ_y C̃(X,y)·eq(y, r_{l+1..m})` of
    /// degree `d·(2^l − 1)`, as its evaluations at `Z = 0, 1, …, deg`.
    pub fn skip_round_evals(&self) -> Vec<Ext> {
        let l = self.skip_bits;
        debug_assert!(l > 0);
        let np = 1usize << l; // |D|
        let d = self.ir.max_constraint_degree as usize;
        let n_pts = d * (np - 1) + 1; // deg(v) + 1
        let nx = self.eq_suffix.len(); // 2^{m-l}
        let base = match &self.tables {
            Tables::Base(t) => t,
            Tables::Ext(_) => panic!("skip_round_evals must run before any bind"),
        };
        let n_tables = base.len();

        let mut temps: Vec<Val> = Vec::new();
        let mut vals = vec![Val::zero(); n_tables];
        // Given the per-leaf values `vals` filled for a fixed Z, accumulate
        // `Σ_x C̃(Z,x)·eq(x, r_X)` over the suffix.
        macro_rules! v_at {
            ($fill_x:expr) => {{
                let mut vk = Ext::ZERO;
                for x in 0..nx {
                    $fill_x(x, &mut vals);
                    let src = self.table_point(&vals);
                    eval_instrs(self.ir, &src, &mut temps);
                    let mut c = Val::zero();
                    for (t, w) in self.weights.iter().enumerate() {
                        if !w.is_zero() {
                            c = c + Val::E(*w) * constraint_value(self.ir, &src, &temps, t);
                        }
                    }
                    vk += self.eq_suffix[x] * c.to_ext();
                }
                vk
            }};
        }

        // Compute v(Z) = Σ_x C̃(Z,x)·eq(x, r_{l+1..m}) for each Z = 0..deg(v).
        let mut node_vals = Vec::with_capacity(n_pts);

        // (a) D-points ω^0..ω^{2^l−1}: Boolean rows → base-field eval_instrs.
        let d_pts = d_subgroup(l);
        for p_prime in 0..np {
            node_vals.push(v_at!(|x: usize, vals: &mut [Val]| {
                for (t, table) in base.iter().enumerate() {
                    vals[t] = Val::B(table[p_prime + x * np]);
                }
            }));
        }

        // (b) Extra nodes in a coset of D (disjoint from D): extension eval_instrs
        //     through each column's D-lift at Z.
        let extra_nodes = coset_nodes(n_pts - np);
        for &z in &extra_nodes {
            let lag = lagrange_d(l, z);
            node_vals.push(v_at!(|x: usize, vals: &mut [Val]| {
                for (t, table) in base.iter().enumerate() {
                    let mut acc = Ext::ZERO;
                    for (p, &lp) in lag.iter().enumerate() {
                        acc += lp * table[p + x * np];
                    }
                    vals[t] = Val::E(acc);
                }
            }));
        }

        // Resample onto Z = 0..deg (the format the verifier interpolates).
        let mut nodes: Vec<Ext> = d_pts.iter().map(|&dp| Ext::from_base(dp)).collect();
        nodes.extend_from_slice(&extra_nodes);
        (0..n_pts).map(|k| lagrange_eval(&nodes, &node_vals, Ext::from_base(Goldilocks::from_u64(k as u64)))).collect()
    }

    /// Bind the univariate-skip block to `γ`: collapse the first `l` variables of
    /// every leaf table via its Lagrange lift over `D` at `γ`, leaving extension
    /// tables over the `m−l` suffix variables; the remaining rounds are ordinary
    /// rounds over `r_{l+1..m}`.
    pub fn skip_bind(&mut self, gamma: Ext) {
        let l = self.skip_bits;
        let np = 1usize << l;
        let lag = lagrange_d(l, gamma);
        let base = match &self.tables {
            Tables::Base(t) => t,
            Tables::Ext(_) => panic!("skip_bind must run before any other bind"),
        };
        let ext: Vec<Vec<Ext>> = base
            .iter()
            .map(|table| {
                let nx = table.len() / np;
                (0..nx)
                    .map(|x| {
                        let mut acc = Ext::ZERO;
                        for (p, &lp) in lag.iter().enumerate() {
                            acc += lp * table[p + x * np];
                        }
                        acc
                    })
                    .collect()
            })
            .collect();
        self.tables = Tables::Ext(ext);
        // eq_suffix = eq(·, r_{l..m}); fold once → eq(·, r_{l+1..m}) for the first
        // suffix Gruen round (valid since (1−r)+r = 1, independent of γ).
        let half = self.eq_suffix.len() / 2;
        for j in 0..half {
            self.eq_suffix[j] = self.eq_suffix[2 * j] + self.eq_suffix[2 * j + 1];
        }
        self.eq_suffix.truncate(half);
        self.rounds_left = self.ir.n_bits as usize - l;
    }

    fn table_point<'s>(&'s self, vals: &'s [Val]) -> TablePoint<'s> {
        TablePoint {
            ir: self.ir,
            wtn_index: &self.wtn_index,
            const_index: &self.const_index,
            custom_index: &self.custom_index,
            vals,
            publics: &self.publics,
            challenges: &self.challenges,
            air_values: &self.air_values,
            airgroup_values: &self.airgroup_values,
        }
    }
}

/// The shifted column as a table: `out[i] = col[(i + s) mod n]`.
fn rotate_shifted(col: &[Goldilocks], s: i32) -> Vec<Goldilocks> {
    // rotate_table computes out[y] = table[(y − s) mod n], so negate.
    rotate_table(col, -(s as i64))
}

/// `n` distinct extra interpolation nodes for the univariate-skip round
/// polynomial: a coset `SHIFT·{ω^0, ω^1, …}` of a subgroup large enough to hold
/// them.
fn coset_nodes(n: usize) -> Vec<Ext> {
    if n == 0 {
        return Vec::new();
    }
    let bits = (n as u64).next_power_of_two().trailing_zeros() as usize;
    let omega = Goldilocks::new(Goldilocks::W[bits]);
    let mut cur = Goldilocks::new(Goldilocks::SHIFT);
    (0..n)
        .map(|_| {
            let z = Ext::from_base(cur);
            cur *= omega;
            z
        })
        .collect()
}

impl SumcheckOracle for ZerocheckOracle<'_> {
    fn num_rounds(&self) -> usize {
        self.rounds_left
    }

    fn round_degree(&self) -> usize {
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
                    wtn_index: &self.wtn_index,
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
    use crate::pcs::MlParams;
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
        let mut oracle = ZerocheckOracle::new(&ir, &witness, &consts, &[], &publics, &[], &[], &[], &r, alpha, 0);

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
        let oracle = ZerocheckOracle::new(&ir, &witness, &consts, &[], &publics, &[], &[], &[], &r, alpha, 0);
        let evals = oracle.round_evals();
        // The round-0 weighted combination equals Σ_X eq(r,X)·C(X) = C̃(r), which
        // is nonzero w.h.p. when the trace violates a constraint.
        assert_ne!((Ext::ONE - r[0]) * evals[0] + r[0] * evals[1], Ext::ZERO);
    }

    /// Univariate-skip zerocheck at the oracle level: the skip round-0 poly `v`
    /// (degree `d·(2^l−1)`) must pass the skip check `Σ_p eq(p,r_P)·v(φ(p)) = 0`
    /// on a valid trace, and after binding `γ` the `m−l` Gruen suffix rounds must
    /// terminate at the batched constraint value at the skip point `(γ, λ_X)` —
    /// reconstructed independently via the (validated) `skip_kernel_table`.
    #[test]
    fn zerocheck_skip_roundtrip_on_fib() {
        use crate::eq::{d_subgroup, eq_eval, skip_kernel_table};
        use crate::hypercube::boolean_point;

        let m = 5usize;
        for l in 1..=3usize {
            let ir = fib_ir(m as u32, MlParams::default());
            let (witness, consts, publics) = fib_trace(m as u32);
            let d = ir.max_constraint_degree as usize;

            let r: Vec<Ext> = (0..m).map(|_| random_ext()).collect();
            let alpha = random_ext();
            let mut oracle = ZerocheckOracle::new(&ir, &witness, &consts, &[], &publics, &[], &[], &[], &r, alpha, l);

            // Skip round + check.
            let v = oracle.skip_round_evals();
            assert_eq!(v.len(), d * ((1 << l) - 1) + 1, "l={l}");
            let dsub = d_subgroup(l);
            let mut skip_sum = Ext::ZERO;
            for (p, &dp) in dsub.iter().enumerate() {
                skip_sum += eq_eval(&boolean_point(p as u64, l), &r[..l]) * interpolate_at(&v, Ext::from_base(dp));
            }
            assert_eq!(skip_sum, Ext::ZERO, "skip check l={l}");

            let gamma = random_ext();
            let mut claim = interpolate_at(&v, gamma);
            oracle.skip_bind(gamma);

            // Gruen suffix rounds over r_X = r[l..].
            let mut lambda_x = Vec::new();
            for round in 0..(m - l) {
                let evals = oracle.round_evals();
                let ch = random_ext();
                let rk = r[l + round];
                assert_eq!((Ext::ONE - rk) * evals[0] + rk * evals[1], claim, "suffix round {round} l={l}");
                claim = interpolate_at(&evals, ch);
                oracle.bind(ch);
                lambda_x.push(ch);
            }

            // Final: batched constraint at (γ, λ_X) via the skip kernels (offset 0
            // and offset 1 = rotate by 1), matching the fib constraints.
            let k0 = skip_kernel_table(l, gamma, &lambda_x);
            let k1 = crate::eq::rotate_table(&k0, 1);
            let ev = |col: &Vec<Goldilocks>, kern: &[Ext]| -> Ext {
                col.iter().zip(kern.iter()).map(|(&w, &k)| k * w).sum()
            };
            let (a_l, b_l) = (ev(&witness[0][0], &k0), ev(&witness[0][1], &k0));
            let (a_n, b_n) = (ev(&witness[0][0], &k1), ev(&witness[0][1], &k1));
            let nl = ev(&consts[0], &k0);
            let expected = nl * (b_n - (a_l + b_l)) + alpha * (nl * (a_n - b_l));
            assert_eq!(claim, expected, "final claim l={l}");
        }
    }
}
