//! Interpreter for the [`AirIr`](crate::ir::AirIr) instruction list, generic
//! over where the leaf values come from:
//!
//! - the **prover** evaluates over (folded) column tables at a point of the
//!   partially-bound hypercube (see the zerocheck round oracle);
//! - the **verifier** evaluates once, on the claimed column openings at the
//!   final sumcheck point.

use crate::error::MlError;
use crate::hypercube::{ext_from_base, Ext};
use crate::ir::{AirIr, Op, Operand, SrcKind};
use fields::Goldilocks;

/// Supplies leaf values for one evaluation point.
pub trait LeafSource {
    /// Value of witness column `col` of `stage` (1-based), shifted by `row_offset`.
    fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Ext;
    /// Value of fixed column `col`, shifted by `row_offset`.
    fn constant(&self, col: u32, row_offset: i32) -> Ext;
    fn public(&self, idx: u32) -> Ext;
    fn challenge(&self, idx: u32) -> Ext;
}

fn operand_value<S: LeafSource>(ir: &AirIr, src: &S, temps: &[Ext], op: &Operand) -> Ext {
    match op.kind {
        SrcKind::Witness { stage } => src.witness(stage, op.idx, op.row_offset),
        SrcKind::Const => src.constant(op.idx, op.row_offset),
        SrcKind::Public => src.public(op.idx),
        SrcKind::Challenge => src.challenge(op.idx),
        SrcKind::Number => ext_from_base(Goldilocks::new(ir.numbers[op.idx as usize])),
        SrcKind::Temp => temps[op.idx as usize],
    }
}

/// Run the instruction list once. `temps` is a scratch buffer of length
/// `ir.n_temps` (reused across calls to avoid allocation).
pub fn eval_instrs<S: LeafSource>(ir: &AirIr, src: &S, temps: &mut Vec<Ext>) {
    temps.resize(ir.n_temps as usize, Ext::zero());
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
pub fn constraint_value<S: LeafSource>(ir: &AirIr, src: &S, temps: &[Ext], c_idx: usize) -> Ext {
    operand_value(ir, src, temps, &ir.constraints[c_idx].root)
}

/// Evaluate only the instructions in the dependency cone of constraint
/// `c_idx` (instructions have `dst == index`, so the cone is found by a
/// reverse DFS over temp references). Used for boundary constraints, whose
/// leaf source is only defined for the leaves the constraint actually reads.
pub fn eval_constraint_cone<S: LeafSource>(ir: &AirIr, src: &S, temps: &mut Vec<Ext>, c_idx: usize) -> Ext {
    debug_assert!(ir.instrs.iter().enumerate().all(|(i, ins)| ins.dst as usize == i));
    temps.resize(ir.n_temps as usize, Ext::zero());

    let root = &ir.constraints[c_idx].root;
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

/// Debug helper mirroring the C++ `verify_constraints` mode: evaluate every
/// constraint at every row of a base-field trace and report offending rows.
/// `witness[stage−1]` are the stage's columns, `consts` the fixed columns; all
/// column-major (`col[row]`).
pub fn check_constraints_on_trace(
    ir: &AirIr,
    witness: &[Vec<Vec<Goldilocks>>],
    consts: &[Vec<Goldilocks>],
    publics: &[Goldilocks],
    challenges: &[Ext],
) -> Result<(), MlError> {
    let n_rows = 1usize << ir.n_bits;

    struct RowSource<'a> {
        witness: &'a [Vec<Vec<Goldilocks>>],
        consts: &'a [Vec<Goldilocks>],
        publics: &'a [Goldilocks],
        challenges: &'a [Ext],
        row: usize,
        n_rows: usize,
    }
    impl LeafSource for RowSource<'_> {
        fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Ext {
            let r = (self.row as i64 + row_offset as i64).rem_euclid(self.n_rows as i64) as usize;
            ext_from_base(self.witness[(stage - 1) as usize][col as usize][r])
        }
        fn constant(&self, col: u32, row_offset: i32) -> Ext {
            let r = (self.row as i64 + row_offset as i64).rem_euclid(self.n_rows as i64) as usize;
            ext_from_base(self.consts[col as usize][r])
        }
        fn public(&self, idx: u32) -> Ext {
            ext_from_base(self.publics[idx as usize])
        }
        fn challenge(&self, idx: u32) -> Ext {
            self.challenges[idx as usize]
        }
    }

    let mut temps = Vec::new();
    for (c_idx, c) in ir.constraints.iter().enumerate() {
        let rows: Box<dyn Iterator<Item = usize>> = match c.boundary {
            crate::ir::Boundary::EveryRow => Box::new(0..n_rows),
            crate::ir::Boundary::FirstRow => Box::new(core::iter::once(0)),
            crate::ir::Boundary::LastRow => Box::new(core::iter::once(n_rows - 1)),
        };
        for row in rows {
            let src = RowSource { witness, consts, publics, challenges, row, n_rows };
            eval_instrs(ir, &src, &mut temps);
            let v = constraint_value(ir, &src, &temps, c_idx);
            if !v.is_zero() {
                return Err(MlError::Constraint(format!("row {row}, value {v}"), c_idx));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_air {
    //! A hand-built Fibonacci AIR used across the crate's tests:
    //! columns `a`, `b` (stage 1); constraints
    //!   EveryRow: b' − (a + b) = 0, a' − b = 0   (cyclic wrap on the last row
    //!   is avoided by gating with the fixed column `not_last`),
    //!   FirstRow: a − pub0 = 0, b − pub1 = 0.
    use super::*;
    use crate::basefold::MlParams;
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
            n_publics: 2,
            challenge_stages: vec![],
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
}

#[cfg(test)]
mod tests {
    use super::test_air::{fib_ir, fib_trace};
    use super::*;
    use crate::basefold::MlParams;
    use fields::Field;

    #[test]
    fn fib_trace_satisfies_ir() {
        let ir = fib_ir(4, MlParams::default());
        let (witness, consts, publics) = fib_trace(4);
        check_constraints_on_trace(&ir, &witness, &consts, &publics, &[]).expect("valid trace");
    }

    #[test]
    fn corrupted_trace_fails_check() {
        let ir = fib_ir(4, MlParams::default());
        let (mut witness, consts, publics) = fib_trace(4);
        witness[0][0][5] += Goldilocks::ONE;
        assert!(check_constraints_on_trace(&ir, &witness, &consts, &publics, &[]).is_err());
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
