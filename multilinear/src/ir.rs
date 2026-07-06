//! The AIR intermediate representation: a flat, evaluator-friendly compilation
//! of the PIL constraint expression DAG.
//!
//! The setup pipeline lowers the pilout expression arena into a postorder
//! instruction list (shared subexpressions become shared temps, so CSE comes
//! for free from the arena's `Expression{idx}` references). The same IR is
//! consumed by the prover (evaluating constraints over folded column tables in
//! the extension field) and the verifier (one evaluation on the claimed
//! openings at the final sumcheck point).

use crate::basefold::MlParams;
use crate::error::MlError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SrcKind {
    /// Committed witness column of a stage (1-based stage as in pilout).
    /// `idx` is the *base-slot* offset within the stage; extension-valued
    /// columns (`dim == 3`) occupy `dim` consecutive base slots whose
    /// coordinates the evaluator reassembles.
    Witness { stage: u8 },
    /// Fixed (constant) column.
    Const,
    /// Public input.
    Public,
    /// Transcript challenge (global index into `AirIr::challenge_stages`).
    Challenge,
    /// Air value (per-instance scalar, prover message); global index.
    AirValue,
    /// Airgroup value (per-instance scalar entering global constraints); global index.
    AirGroupValue,
    /// Constant from the `numbers` pool.
    Number,
    /// Result of a previous instruction.
    Temp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operand {
    pub kind: SrcKind,
    pub idx: u32,
    /// Row offset for column operands (`x'` = +1, `x'(2)` = +2, …), 0 otherwise.
    pub row_offset: i32,
    /// Number of base-field slots this operand spans (1 for base-field
    /// columns, 3 for extension-valued stage ≥ 2 columns).
    pub dim: u8,
}

impl Operand {
    pub fn temp(idx: u32) -> Self {
        Self { kind: SrcKind::Temp, idx, row_offset: 0, dim: 1 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Instr {
    pub op: Op,
    pub dst: u32,
    pub a: Operand,
    /// Ignored for `Neg`.
    pub b: Operand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    /// Must vanish at every row (proven by the zerocheck).
    EveryRow,
    /// Must vanish at row 0 (reduces to corner evaluation claims).
    FirstRow,
    /// Must vanish at row 2^n − 1.
    LastRow,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConstraintIr {
    pub boundary: Boundary,
    pub root: Operand,
    /// Total degree of the constraint in the committed columns.
    pub degree: u32,
}

/// Everything the multilinear prover/verifier needs to know about one AIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirIr {
    pub name: String,
    pub airgroup_id: u32,
    pub air_id: u32,
    /// log2 of the number of rows.
    pub n_bits: u32,
    /// Committed witness *base* columns per stage (index 0 = stage 1);
    /// an extension-valued column counts as 3.
    pub cols_per_stage: Vec<u32>,
    pub n_const_cols: u32,
    pub n_publics: u32,
    /// Stage of each challenge, in global challenge order. Only challenges
    /// with `stage <= n_stages` are ever derived/used — later stages belong
    /// to the univariate protocol (quotient/evals/FRI batching).
    pub challenge_stages: Vec<u8>,
    /// Stage of each air value, in global order.
    pub airvalue_stages: Vec<u8>,
    /// Stage of each airgroup value, in global order.
    pub airgroupvalue_stages: Vec<u8>,
    /// Pool of literal constants.
    pub numbers: Vec<u64>,
    pub instrs: Vec<Instr>,
    pub n_temps: u32,
    pub constraints: Vec<ConstraintIr>,
    pub max_constraint_degree: u32,
    /// Distinct row offsets referenced by any constraint (always contains 0).
    pub opening_offsets: Vec<i32>,
    pub params: MlParams,
}

impl AirIr {
    pub fn n_stages(&self) -> usize {
        self.cols_per_stage.len()
    }

    pub fn total_witness_cols(&self) -> usize {
        self.cols_per_stage.iter().map(|&c| c as usize).sum()
    }

    /// Total number of committed columns (witness + const).
    pub fn total_cols(&self) -> usize {
        self.total_witness_cols() + self.n_const_cols as usize
    }

    /// Index of `offset` in `opening_offsets`.
    pub fn offset_index(&self, offset: i32) -> Option<usize> {
        self.opening_offsets.iter().position(|&s| s == offset)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), MlError> {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("serializing AirIr: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| MlError::Io(format!("writing {}: {e}", path.display())))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, MlError> {
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("reading {}: {e}", path.display())))?;
        let (ir, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("decoding AirIr: {e}")))?;
        Ok(ir)
    }
}

/// Convenience builder used by tests and the setup compiler.
#[derive(Default)]
pub struct IrBuilder {
    pub instrs: Vec<Instr>,
    pub numbers: Vec<u64>,
    n_temps: u32,
}

impl IrBuilder {
    pub fn number(&mut self, v: u64) -> Operand {
        let idx = self.numbers.iter().position(|&n| n == v).unwrap_or_else(|| {
            self.numbers.push(v);
            self.numbers.len() - 1
        });
        Operand { kind: SrcKind::Number, idx: idx as u32, row_offset: 0, dim: 1 }
    }

    pub fn witness(&self, stage: u8, col: u32, row_offset: i32) -> Operand {
        Operand { kind: SrcKind::Witness { stage }, idx: col, row_offset, dim: 1 }
    }

    /// Extension-valued witness column occupying `base_slot .. base_slot + 3`.
    pub fn witness_ext(&self, stage: u8, base_slot: u32, row_offset: i32) -> Operand {
        Operand { kind: SrcKind::Witness { stage }, idx: base_slot, row_offset, dim: 3 }
    }

    pub fn constant(&self, col: u32, row_offset: i32) -> Operand {
        Operand { kind: SrcKind::Const, idx: col, row_offset, dim: 1 }
    }

    pub fn public(&self, idx: u32) -> Operand {
        Operand { kind: SrcKind::Public, idx, row_offset: 0, dim: 1 }
    }

    pub fn challenge(&self, idx: u32) -> Operand {
        Operand { kind: SrcKind::Challenge, idx, row_offset: 0, dim: 3 }
    }

    pub fn air_value(&self, idx: u32, dim: u8) -> Operand {
        Operand { kind: SrcKind::AirValue, idx, row_offset: 0, dim }
    }

    pub fn airgroup_value(&self, idx: u32, dim: u8) -> Operand {
        Operand { kind: SrcKind::AirGroupValue, idx, row_offset: 0, dim }
    }

    pub fn op(&mut self, op: Op, a: Operand, b: Operand) -> Operand {
        let dst = self.n_temps;
        self.n_temps += 1;
        self.instrs.push(Instr { op, dst, a, b });
        Operand::temp(dst)
    }

    pub fn add(&mut self, a: Operand, b: Operand) -> Operand {
        self.op(Op::Add, a, b)
    }
    pub fn sub(&mut self, a: Operand, b: Operand) -> Operand {
        self.op(Op::Sub, a, b)
    }
    pub fn mul(&mut self, a: Operand, b: Operand) -> Operand {
        self.op(Op::Mul, a, b)
    }
    pub fn neg(&mut self, a: Operand) -> Operand {
        self.op(Op::Neg, a, a)
    }

    pub fn n_temps(&self) -> u32 {
        self.n_temps
    }
}
