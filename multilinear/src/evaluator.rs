//! Interpreter for the [`AirIr`](crate::ir::AirIr) instruction list, generic
//! over where the leaf values come from:
//!
//! - the **prover** evaluates over (folded) column tables at a point of the
//!   partially-bound hypercube (see the zerocheck round oracle);
//! - the **verifier** evaluates once, on the claimed column openings at the
//!   final sumcheck point.

use crate::error::MlError;
use crate::hypercube::Ext;
use crate::ir::{AirIr, Op, Operand, SrcKind};
use core::ops::{Add, Mul, Neg, Sub};
use fields::{Field, Goldilocks};

/// A leaf/temporary value kept **lazily in the base field** until it must be
/// promoted to the extension. A `base × base` multiply is one Goldilocks mult
/// vs. twelve for `Ext × Ext`, and a mixed `Ext × base` is three — so evaluating
/// constraints over the raw (base-field) trace, as the zerocheck's first round
/// does, is far cheaper. Promotion to `Ext` happens transparently the moment a
/// challenge or an already-bound value enters the expression.
#[derive(Clone, Copy, Debug)]
pub enum Val {
    B(Goldilocks),
    E(Ext),
}

impl Val {
    #[inline]
    pub fn from_base(x: Goldilocks) -> Self {
        Val::B(x)
    }
    #[inline]
    pub fn from_ext(x: Ext) -> Self {
        Val::E(x)
    }
    #[inline]
    pub fn to_ext(self) -> Ext {
        match self {
            Val::B(x) => Ext::from_base(x),
            Val::E(x) => x,
        }
    }
    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            Val::B(x) => x.is_zero(),
            Val::E(x) => x.is_zero(),
        }
    }
    #[inline]
    pub fn zero() -> Self {
        Val::B(Goldilocks::ZERO)
    }
}

impl Add for Val {
    type Output = Val;
    #[inline]
    fn add(self, o: Val) -> Val {
        match (self, o) {
            (Val::B(a), Val::B(b)) => Val::B(a + b),
            (Val::E(e), Val::B(b)) | (Val::B(b), Val::E(e)) => Val::E(e + b),
            (Val::E(a), Val::E(b)) => Val::E(a + b),
        }
    }
}

impl Sub for Val {
    type Output = Val;
    #[inline]
    fn sub(self, o: Val) -> Val {
        match (self, o) {
            (Val::B(a), Val::B(b)) => Val::B(a - b),
            (Val::E(a), Val::B(b)) => Val::E(a - b),
            (Val::B(a), Val::E(b)) => Val::E(Ext::from_base(a) - b),
            (Val::E(a), Val::E(b)) => Val::E(a - b),
        }
    }
}

impl Mul for Val {
    type Output = Val;
    #[inline]
    fn mul(self, o: Val) -> Val {
        match (self, o) {
            (Val::B(a), Val::B(b)) => Val::B(a * b),
            (Val::E(e), Val::B(b)) | (Val::B(b), Val::E(e)) => Val::E(e * b),
            (Val::E(a), Val::E(b)) => Val::E(a * b),
        }
    }
}

impl Neg for Val {
    type Output = Val;
    #[inline]
    fn neg(self) -> Val {
        match self {
            Val::B(x) => Val::B(-x),
            Val::E(x) => Val::E(-x),
        }
    }
}

/// Supplies leaf values for one evaluation point. Witness/const leaves are
/// addressed by *base* column slot; extension-valued columns are reassembled
/// by the evaluator from their `dim` consecutive base slots. Base-field leaves
/// return [`Val::B`] so base-only subexpressions stay in the base field.
pub trait LeafSource {
    /// Value of witness base column `col` of `stage` (1-based), shifted by `row_offset`.
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Val;
    /// Value of fixed column `col`, shifted by `row_offset`.
    fn constant(&self, col: u32, row_offset: i32) -> Val;
    fn public(&self, idx: u32) -> Val;
    fn challenge(&self, idx: u32) -> Val;
    fn air_value(&self, _idx: u32) -> Val {
        unimplemented!("air values not available in this context")
    }
    /// Value of column `col` of custom commit `commit`, shifted by `row_offset`.
    fn custom(&self, _commit: u8, _col: u32, _row_offset: i32) -> Val {
        unimplemented!("custom commits not available in this context")
    }
    fn airgroup_value(&self, _idx: u32) -> Val {
        unimplemented!("airgroup values not available in this context")
    }
}

/// The extension-basis element `u` (coordinates `[0, 1, 0]`): an ext-valued
/// column with coordinate base columns `(c0, c1, c2)` has value
/// `c0 + c1·u + c2·u²`. MLE is linear over the extension, so reassembling
/// coordinate MLEs commutes with folding/evaluation.
#[inline]
fn ext_u() -> Ext {
    Ext::from_array(&[Goldilocks::ZERO, Goldilocks::ONE, Goldilocks::ZERO])
}

fn assemble_dim<S: LeafSource>(src: &S, stage: u8, base: u32, row_offset: i32, dim: u8) -> Val {
    if dim == 1 {
        return src.witness(stage, base, row_offset);
    }
    let u = ext_u();
    let mut acc = Val::zero();
    let mut basis = Ext::ONE;
    for k in 0..dim as u32 {
        acc = acc + src.witness(stage, base + k, row_offset) * Val::E(basis);
        basis *= u;
    }
    acc
}

/// Value of an arbitrary operand after [`eval_instrs`] has run (leaves are
/// read from `src`, temps from the scratch buffer).
pub fn operand_eval<S: LeafSource>(ir: &AirIr, src: &S, temps: &[Val], op: &Operand) -> Val {
    operand_value(ir, src, temps, op)
}

fn operand_value<S: LeafSource>(ir: &AirIr, src: &S, temps: &[Val], op: &Operand) -> Val {
    match op.kind {
        SrcKind::Witness { stage } => assemble_dim(src, stage, op.idx, op.row_offset, op.dim),
        SrcKind::Const => src.constant(op.idx, op.row_offset),
        SrcKind::Custom { commit } => src.custom(commit, op.idx, op.row_offset),
        SrcKind::Public => src.public(op.idx),
        SrcKind::Challenge => src.challenge(op.idx),
        SrcKind::AirValue => src.air_value(op.idx),
        SrcKind::AirGroupValue => src.airgroup_value(op.idx),
        SrcKind::Number => Val::B(Goldilocks::new(ir.numbers[op.idx as usize])),
        SrcKind::Temp => temps[op.idx as usize],
    }
}

/// Run the instruction list once. `temps` is a scratch buffer of length
/// `ir.n_temps` (reused across calls to avoid allocation).
pub fn eval_instrs<S: LeafSource>(ir: &AirIr, src: &S, temps: &mut Vec<Val>) {
    temps.resize(ir.n_temps as usize, Val::zero());
    for instr in &ir.instrs {
        let a = operand_value(ir, src, temps, &instr.a);
        let v = match instr.op {
            Op::Add => a + operand_value(ir, src, temps, &instr.b),
            Op::Sub => a - operand_value(ir, src, temps, &instr.b),
            Op::Mul => a * operand_value(ir, src, temps, &instr.b),
            Op::Neg => -a,
        };
        temps[instr.dst as usize] = v;
    }
}

/// Value of constraint `c_idx` after [`eval_instrs`] has run.
pub fn constraint_value<S: LeafSource>(ir: &AirIr, src: &S, temps: &[Val], c_idx: usize) -> Val {
    operand_value(ir, src, temps, &ir.constraints[c_idx].root)
}

/// Evaluate only the instructions in the dependency cone of constraint
/// `c_idx` (instructions have `dst == index`, so the cone is found by a
/// reverse DFS over temp references). Used for boundary constraints, whose
/// leaf source is only defined for the leaves the constraint actually reads.
pub fn eval_constraint_cone<S: LeafSource>(ir: &AirIr, src: &S, temps: &mut Vec<Val>, c_idx: usize) -> Val {
    eval_operand_cone(ir, src, temps, &ir.constraints[c_idx].root)
}

/// Evaluate only the dependency cone of an arbitrary root operand. Used for
/// scalar bus terms, whose leaf source is only defined for scalar leaves.
pub fn eval_operand_cone<S: LeafSource>(ir: &AirIr, src: &S, temps: &mut Vec<Val>, root: &Operand) -> Val {
    debug_assert!(ir.instrs.iter().enumerate().all(|(i, ins)| ins.dst as usize == i));
    temps.resize(ir.n_temps as usize, Val::zero());

    let mut needed = vec![false; ir.instrs.len()];
    let mut stack: Vec<usize> = Vec::new();
    if root.kind == SrcKind::Temp {
        stack.push(root.idx as usize);
    }
    while let Some(i) = stack.pop() {
        if needed[i] {
            continue;
        }
        needed[i] = true;
        let instr = &ir.instrs[i];
        for op in [&instr.a, &instr.b] {
            if op.kind == SrcKind::Temp {
                stack.push(op.idx as usize);
            }
        }
    }

    for (i, instr) in ir.instrs.iter().enumerate() {
        if !needed[i] {
            continue;
        }
        let a = operand_value(ir, src, temps, &instr.a);
        let v = match instr.op {
            Op::Add => a + operand_value(ir, src, temps, &instr.b),
            Op::Sub => a - operand_value(ir, src, temps, &instr.b),
            Op::Mul => a * operand_value(ir, src, temps, &instr.b),
            Op::Neg => -a,
        };
        temps[instr.dst as usize] = v;
    }
    operand_value(ir, src, temps, root)
}

/// Leaf source reading a base-field trace at one row (columns wrap cyclically
/// for shifted reads). Used by [`check_constraints_on_trace`] and by the bus
/// input-layer builder in `logup_gkr.rs`.
pub struct RowSource<'a> {
    pub witness: &'a [Vec<Vec<Goldilocks>>],
    pub consts: &'a [Vec<Goldilocks>],
    pub customs: &'a [Vec<Vec<Goldilocks>>],
    pub publics: &'a [Goldilocks],
    pub challenges: &'a [Ext],
    pub air_values: &'a [Ext],
    pub airgroup_values: &'a [Ext],
    pub row: usize,
    pub n_rows: usize,
}

impl LeafSource for RowSource<'_> {
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Val {
        let r = (self.row as i64 + row_offset as i64).rem_euclid(self.n_rows as i64) as usize;
        Val::B(self.witness[(stage - 1) as usize][col as usize][r])
    }
    fn constant(&self, col: u32, row_offset: i32) -> Val {
        let r = (self.row as i64 + row_offset as i64).rem_euclid(self.n_rows as i64) as usize;
        Val::B(self.consts[col as usize][r])
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
        let r = (self.row as i64 + row_offset as i64).rem_euclid(self.n_rows as i64) as usize;
        Val::B(self.customs[commit as usize][col as usize][r])
    }
}

/// Debug helper mirroring the C++ `verify_constraints` mode: evaluate every
/// constraint at every row of a base-field trace and report offending rows.
/// `witness[stage−1]` are the stage's columns, `consts` the fixed columns; all
/// column-major (`col[row]`).
#[allow(clippy::too_many_arguments)]
pub fn check_constraints_on_trace(
    ir: &AirIr,
    witness: &[Vec<Vec<Goldilocks>>],
    consts: &[Vec<Goldilocks>],
    customs: &[Vec<Vec<Goldilocks>>],
    publics: &[Goldilocks],
    challenges: &[Ext],
    air_values: &[Ext],
    airgroup_values: &[Ext],
) -> Result<(), MlError> {
    let n_rows = 1usize << ir.n_bits;

    let mut temps = Vec::new();
    for (c_idx, c) in ir.constraints.iter().enumerate() {
        let rows: Box<dyn Iterator<Item = usize>> = match c.boundary {
            crate::ir::Boundary::EveryRow => Box::new(0..n_rows),
            crate::ir::Boundary::FirstRow => Box::new(core::iter::once(0)),
            crate::ir::Boundary::LastRow => Box::new(core::iter::once(n_rows - 1)),
        };
        for row in rows {
            let src =
                RowSource { witness, consts, customs, publics, challenges, air_values, airgroup_values, row, n_rows };
            eval_instrs(ir, &src, &mut temps);
            let v = constraint_value(ir, &src, &temps, c_idx).to_ext();
            if !v.is_zero() {
                return Err(MlError::Constraint(format!("row {row}, value {v}"), c_idx));
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "testutil"))]
pub mod test_air {
    //! A hand-built Fibonacci AIR used across the crate's tests:
    //! columns `a`, `b` (stage 1); constraints
    //!   EveryRow: b' − (a + b) = 0, a' − b = 0   (cyclic wrap on the last row
    //!   is avoided by gating with the fixed column `not_last`),
    //!   FirstRow: a − pub0 = 0, b − pub1 = 0.
    use super::*;
    use crate::pcs::MlParams;
    use crate::ir::{Boundary, ConstraintIr, IrBuilder};
    use fields::Field;

    pub fn fib_ir(n_bits: u32, params: MlParams) -> AirIr {
        let mut b = IrBuilder::default();
        let a_cur = b.witness(1, 0, 0);
        let b_cur = b.witness(1, 1, 0);
        let a_next = b.witness(1, 0, 1);
        let b_next = b.witness(1, 1, 1);
        let not_last = b.constant(0, 0);

        // not_last · (b' − (a + b))
        let sum = b.add(a_cur, b_cur);
        let d1 = b.sub(b_next, sum);
        let c1 = b.mul(not_last, d1);
        // not_last · (a' − b)
        let d2 = b.sub(a_next, b_cur);
        let c2 = b.mul(not_last, d2);
        // a − pub0, b − pub1 (first row)
        let p0 = b.public(0);
        let p1 = b.public(1);
        let c3 = b.sub(a_cur, p0);
        let c4 = b.sub(b_cur, p1);

        AirIr {
            name: "FibTest".into(),
            airgroup_id: 0,
            air_id: 0,
            n_bits,
            cols_per_stage: vec![2],
            n_const_cols: 1,
            custom_commits: vec![],
            n_publics: 2,
            challenge_stages: vec![],
            airvalue_stages: vec![],
            airgroupvalue_stages: vec![],
            numbers: b.numbers.clone(),
            n_temps: b.n_temps(),
            instrs: b.instrs,
            constraints: vec![
                ConstraintIr { boundary: Boundary::EveryRow, root: c1, degree: 2 },
                ConstraintIr { boundary: Boundary::EveryRow, root: c2, degree: 2 },
                ConstraintIr { boundary: Boundary::FirstRow, root: c3, degree: 1 },
                ConstraintIr { boundary: Boundary::FirstRow, root: c4, degree: 1 },
            ],
            max_constraint_degree: 2,
            opening_offsets: vec![0, 1],
            bus: None,
            params,
        }
    }

    /// Build a valid Fibonacci trace: returns (witness stage-1 columns, const columns, publics).
    #[allow(clippy::type_complexity)]
    pub fn fib_trace(n_bits: u32) -> (Vec<Vec<Vec<Goldilocks>>>, Vec<Vec<Goldilocks>>, Vec<Goldilocks>) {
        let n = 1usize << n_bits;
        let mut a = vec![Goldilocks::ONE; n];
        let mut bcol = vec![Goldilocks::TWO; n];
        for i in 1..n {
            a[i] = bcol[i - 1];
            bcol[i] = a[i - 1] + bcol[i - 1];
        }
        let mut not_last = vec![Goldilocks::ONE; n];
        not_last[n - 1] = Goldilocks::ZERO;
        let publics = vec![Goldilocks::ONE, Goldilocks::TWO];
        (vec![vec![a, bcol]], vec![not_last], publics)
    }

    /// A hand-built lookup AIR proven with the LogUp-GKR bus (no committed
    /// running-sum columns): stage-1 columns `a` (looked-up values), `t`
    /// (table) and `mul` (multiplicities); bus terms
    ///   assume: `−1 / (a + γ)`,   prove: `mul / (t + γ)`,
    /// with `γ` the stage-2 challenge and the instance's net contribution
    /// carried in airgroup value 0. `with_scalar_term` adds the direct
    /// fraction `pub0 / (γ + pub1)` at the instance level.
    pub fn lookup_ir(n_bits: u32, params: MlParams, with_scalar_term: bool) -> AirIr {
        use crate::ir::{BusIr, BusTerm};

        let mut b = IrBuilder::default();
        let a = b.witness(1, 0, 0);
        let t = b.witness(1, 1, 0);
        let mul = b.witness(1, 2, 0);
        let gamma = b.challenge(0);

        let one = b.number(1);
        let neg_one = b.neg(one);
        let den_a = b.add(a, gamma);
        let den_t = b.add(t, gamma);

        let scalar_terms = if with_scalar_term {
            let p0 = b.public(0);
            let p1 = b.public(1);
            let den = b.add(gamma, p1);
            vec![BusTerm { num: p0, den, degree: 0 }]
        } else {
            vec![]
        };

        AirIr {
            name: "LookupTest".into(),
            airgroup_id: 0,
            air_id: 1,
            n_bits,
            cols_per_stage: vec![3],
            n_const_cols: 1,
            custom_commits: vec![],
            n_publics: if with_scalar_term { 2 } else { 0 },
            challenge_stages: vec![2],
            airvalue_stages: vec![],
            airgroupvalue_stages: vec![2],
            numbers: b.numbers.clone(),
            n_temps: b.n_temps(),
            instrs: b.instrs,
            constraints: vec![],
            max_constraint_degree: 1,
            opening_offsets: vec![0],
            bus: Some(BusIr {
                terms: vec![
                    BusTerm { num: neg_one, den: den_a, degree: 1 },
                    BusTerm { num: mul, den: den_t, degree: 1 },
                ],
                scalar_terms,
                result_airgroupvalue: Some(0),
                max_term_degree: 1,
            }),
            params,
        }
    }

    /// A balanced lookup trace for [`lookup_ir`]: `t = 0..n`, `a` drawn from
    /// `t`, `mul[v] = #{i : a[i] = v}` — so the bus fraction sum is zero.
    #[allow(clippy::type_complexity)]
    pub fn lookup_trace(n_bits: u32) -> (Vec<Vec<Vec<Goldilocks>>>, Vec<Vec<Goldilocks>>) {
        use fields::PrimeField64;
        let n = 1usize << n_bits;
        let t: Vec<Goldilocks> = (0..n as u64).map(Goldilocks::from_u64).collect();
        let looked: Vec<usize> = (0..n).map(|i| (i * i + 3 * i) % n).collect();
        let a: Vec<Goldilocks> = looked.iter().map(|&v| Goldilocks::from_u64(v as u64)).collect();
        let mut counts = vec![0u64; n];
        for &v in &looked {
            counts[v] += 1;
        }
        let mul: Vec<Goldilocks> = counts.into_iter().map(Goldilocks::from_u64).collect();
        (vec![vec![a, t, mul]], vec![vec![Goldilocks::ZERO; n]])
    }
}

#[cfg(test)]
mod tests {
    use super::test_air::{fib_ir, fib_trace};
    use super::*;
    use crate::pcs::MlParams;
    use fields::Field;

    #[test]
    fn fib_trace_satisfies_ir() {
        let ir = fib_ir(4, MlParams::default());
        let (witness, consts, publics) = fib_trace(4);
        check_constraints_on_trace(&ir, &witness, &consts, &[], &publics, &[], &[], &[]).expect("valid trace");
    }

    #[test]
    fn corrupted_trace_fails_check() {
        let ir = fib_ir(4, MlParams::default());
        let (mut witness, consts, publics) = fib_trace(4);
        witness[0][0][5] += Goldilocks::ONE;
        assert!(check_constraints_on_trace(&ir, &witness, &consts, &[], &publics, &[], &[], &[]).is_err());
    }

    #[test]
    fn ir_bincode_roundtrip() {
        let ir = fib_ir(4, MlParams::default());
        let bytes = bincode::serde::encode_to_vec(&ir, bincode::config::standard()).unwrap();
        let (back, _): (AirIr, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(back.name, ir.name);
        assert_eq!(back.instrs.len(), ir.instrs.len());
        assert_eq!(back.constraints.len(), ir.constraints.len());
    }
}
