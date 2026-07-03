//! Compilation of the pilout constraint expressions into the multilinear
//! prover's IR (`<AIR>.mlinfo.bin`, a bincode-serialized
//! [`proofman_multilinear::AirIr`]).
//!
//! The setup expression arena is lowered into a flat instruction list; arena
//! references (`op == "exp"`) are memoized so shared subexpressions become
//! shared temps. AIRs using features the multilinear prover does not support
//! yet (air values, proof values, custom commits, `everyFrame` constraints,
//! shifted columns inside boundary constraints) are rejected with a
//! descriptive error — the setup then simply skips the mlinfo artifact.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};
use fields::{Goldilocks, PrimeField64};
use proofman_multilinear::{AirIr, Boundary, ConstraintIr, Instr, MlParams, Op, Operand, SrcKind};

use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::SetupResult;

struct Compiler<'a> {
    exprs: &'a [Expression],
    instrs: Vec<Instr>,
    numbers: Vec<u64>,
    /// Memoized lowering of arena entries: arena id → (operand, degree, offsets used).
    memo: HashMap<usize, (Operand, u32, BTreeSet<i32>)>,
    offsets: BTreeSet<i32>,
    uses_challenges: bool,
}

impl Compiler<'_> {
    fn number(&mut self, value: &str) -> Result<Operand> {
        let v = value
            .parse::<u128>()
            .map(|v| (v % Goldilocks::ORDER_U64 as u128) as u64)
            .map_err(|_| anyhow!("cannot parse number '{value}'"))?;
        let idx = self.numbers.iter().position(|&n| n == v).unwrap_or_else(|| {
            self.numbers.push(v);
            self.numbers.len() - 1
        });
        Ok(Operand { kind: SrcKind::Number, idx: idx as u32, row_offset: 0 })
    }

    fn emit(&mut self, op: Op, a: Operand, b: Operand) -> Operand {
        let dst = self.instrs.len() as u32;
        self.instrs.push(Instr { op, dst, a, b });
        Operand { kind: SrcKind::Temp, idx: dst, row_offset: 0 }
    }

    /// Lower an arena entry (memoized).
    fn lower_id(&mut self, id: usize) -> Result<(Operand, u32, BTreeSet<i32>)> {
        if let Some(hit) = self.memo.get(&id) {
            return Ok(hit.clone());
        }
        let result = self.lower_expr(&self.exprs[id])?;
        self.memo.insert(id, result.clone());
        Ok(result)
    }

    fn lower_child(&mut self, child: &ExprChild) -> Result<(Operand, u32, BTreeSet<i32>)> {
        match child {
            ExprChild::Id(id) => self.lower_id(*id),
            ExprChild::Inline(expr) => self.lower_expr(expr),
        }
    }

    /// Lower one expression node. Returns (operand, degree, row offsets in its cone).
    fn lower_expr(&mut self, e: &Expression) -> Result<(Operand, u32, BTreeSet<i32>)> {
        let leaf_offset = |e: &Expression| e.row_offset.unwrap_or(0) as i32;
        match e.op.as_str() {
            "cm" => {
                let stage = e.stage as u8;
                if stage != 1 {
                    bail!("witness column of stage {stage} (multi-stage is milestone 2)");
                }
                let s = leaf_offset(e);
                self.offsets.insert(s);
                let col = e.stage_id.ok_or_else(|| anyhow!("cm operand without stage_id"))? as u32;
                Ok((Operand { kind: SrcKind::Witness { stage }, idx: col, row_offset: s }, 1, BTreeSet::from([s])))
            }
            "const" => {
                let s = leaf_offset(e);
                self.offsets.insert(s);
                let col = e.id.ok_or_else(|| anyhow!("const operand without id"))? as u32;
                Ok((Operand { kind: SrcKind::Const, idx: col, row_offset: s }, 1, BTreeSet::from([s])))
            }
            "public" => {
                let idx = e.id.ok_or_else(|| anyhow!("public operand without id"))? as u32;
                Ok((Operand { kind: SrcKind::Public, idx, row_offset: 0 }, 0, BTreeSet::new()))
            }
            "challenge" => {
                let idx = e.id.ok_or_else(|| anyhow!("challenge operand without id"))? as u32;
                self.uses_challenges = true;
                Ok((Operand { kind: SrcKind::Challenge, idx, row_offset: 0 }, 0, BTreeSet::new()))
            }
            "number" => {
                let value = e.value.as_deref().ok_or_else(|| anyhow!("number operand without value"))?;
                Ok((self.number(value)?, 0, BTreeSet::new()))
            }
            "exp" => self.lower_id(e.id.ok_or_else(|| anyhow!("exp operand without id"))?),
            "add" | "sub" | "mul" => {
                if e.values.len() != 2 {
                    bail!("binary op '{}' with {} children", e.op, e.values.len());
                }
                let (a, da, mut offs_a) = self.lower_child(&e.values[0])?;
                let (b, db, offs_b) = self.lower_child(&e.values[1])?;
                offs_a.extend(offs_b);
                let (op, deg) = match e.op.as_str() {
                    "add" => (Op::Add, da.max(db)),
                    "sub" => (Op::Sub, da.max(db)),
                    _ => (Op::Mul, da + db),
                };
                Ok((self.emit(op, a, b), deg, offs_a))
            }
            "neg" => {
                if e.values.len() != 1 {
                    bail!("neg with {} children", e.values.len());
                }
                let (a, d, offs) = self.lower_child(&e.values[0])?;
                Ok((self.emit(Op::Neg, a, a), d, offs))
            }
            other => bail!("unsupported operand '{other}' (multilinear milestone 2)"),
        }
    }
}

/// Compile a `SetupResult` into the multilinear prover's `AirIr`.
pub fn build_air_ir(setup: &SetupResult, n_bits: u32, params: MlParams) -> Result<AirIr> {
    let n_stages = setup.n_stages;
    let mut cols_per_stage = Vec::with_capacity(n_stages);
    for stage in 1..=n_stages {
        let width = *setup
            .map_sections_n
            .get(&format!("cm{stage}"))
            .ok_or_else(|| anyhow!("missing cm{stage} section width"))?;
        cols_per_stage.push(width as u32);
    }

    let mut compiler = Compiler {
        exprs: &setup.expressions,
        instrs: Vec::new(),
        numbers: Vec::new(),
        memo: HashMap::new(),
        offsets: BTreeSet::from([0]),
        uses_challenges: false,
    };

    let mut constraints = Vec::with_capacity(setup.constraints.len());
    let mut max_degree = 1u32;
    for (idx, con) in setup.constraints.iter().enumerate() {
        let boundary = match con.boundary.as_str() {
            "everyRow" => Boundary::EveryRow,
            "firstRow" => Boundary::FirstRow,
            "lastRow" => Boundary::LastRow,
            other => bail!("constraint {idx}: unsupported boundary '{other}'"),
        };
        let (root, degree, offsets) = compiler.lower_id(con.e)?;
        if boundary != Boundary::EveryRow && offsets.iter().any(|&s| s != 0) {
            bail!("constraint {idx}: boundary constraints must not reference shifted columns");
        }
        if boundary == Boundary::EveryRow {
            max_degree = max_degree.max(degree);
        }
        constraints.push(ConstraintIr { boundary, root, degree });
    }

    // Protocol-level challenge symbols exist in every pilout (the univariate
    // pipeline declares them); only surface them if a constraint actually
    // reads a challenge — that is what gates multi-stage (milestone 2) support.
    let challenge_stages: Vec<u8> = if compiler.uses_challenges {
        setup.challenges_map.iter().map(|c| c.stage.unwrap_or(0) as u8).collect()
    } else {
        Vec::new()
    };

    Ok(AirIr {
        name: setup.name.clone(),
        airgroup_id: setup.airgroup_id as u32,
        air_id: setup.air_id as u32,
        n_bits,
        cols_per_stage,
        n_const_cols: setup.n_constants as u32,
        n_publics: setup.n_publics as u32,
        challenge_stages,
        numbers: compiler.numbers,
        n_temps: compiler.instrs.len() as u32,
        instrs: compiler.instrs,
        constraints,
        max_constraint_degree: max_degree,
        opening_offsets: compiler.offsets.into_iter().collect(),
        params,
    })
}
