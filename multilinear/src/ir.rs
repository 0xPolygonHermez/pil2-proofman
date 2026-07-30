//! The AIR intermediate representation: a flat, evaluator-friendly compilation
//! of the PIL constraint expression DAG.
//!
//! The setup pipeline lowers the pilout expression arena into a postorder
//! instruction list (shared subexpressions become shared temps, so CSE comes
//! for free from the arena's `Expression{idx}` references). The same IR is
//! consumed by the prover (evaluating constraints over folded column tables in
//! the extension field) and the verifier (one evaluation on the claimed
//! openings at the final sumcheck point).

use crate::pcs::MlParams;
use crate::error::MlError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SrcKind {
    /// Committed witness column of a stage (1-based stage as in pilout).
    /// `idx` is the *base-slot* offset within the stage; extension-valued
    /// columns (`dim == 3`) occupy `dim` consecutive base slots whose
    /// coordinates the evaluator reassembles.
    Witness { stage: u8 },
    /// Fixed (constant) column.
    Const,
    /// Column of a custom commit (fixed data committed separately, e.g. a ROM;
    /// stage 0 only). `idx` is the base slot within the commit.
    Custom { commit: u8 },
    /// Public input.
    Public,
    /// Transcript challenge (global index into `AirIr::challenge_stages`).
    Challenge,
    /// Air value (per-instance scalar, prover message); global index.
    AirValue,
    /// Airgroup value (per-instance scalar entering global constraints); global index.
    AirGroupValue,
    /// Proof value (proof-level scalar, prover message shared by every
    /// instance; stage-1 values enter the global challenge); global index.
    ProofValue,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlCustomCommit {
    pub name: String,
    /// Stage-0 base columns of this commit.
    pub n_cols: u32,
}

/// One fraction `num/den` contributed to the LogUp bus by every row (or, for
/// scalar terms, once per instance). `den` includes the bus challenge `+γ`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BusTerm {
    pub num: Operand,
    pub den: Operand,
    /// Max total degree of `num`/`den` in the committed columns.
    pub degree: u32,
}

/// The LogUp sum bus of one AIR, proven with the GKR fractional sumcheck
/// instead of committed running-sum columns (see `logup_gkr.rs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusIr {
    /// Per-row fraction terms; the input layer of the fraction tree has
    /// `terms.len().next_power_of_two() · 2^n_bits` entries (padded with the
    /// neutral fraction `0/1`), term index in the low variables.
    pub terms: Vec<BusTerm>,
    /// Instance-level "direct" fractions: expressions of scalars only
    /// (publics, challenges, air/airgroup values — never columns). They are
    /// added to `p_out/q_out` when forming the bus result.
    pub scalar_terms: Vec<BusTerm>,
    /// Global index of the airgroup value carrying the instance's net bus
    /// contribution (`gsum_result`); `None` in single-instance mode, where the
    /// contribution must be zero.
    pub result_airgroupvalue: Option<u32>,
    /// Max `BusTerm::degree` over `terms` (round-degree bound of the
    /// input-layer reduction sumcheck).
    pub max_term_degree: u32,
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
    /// Custom (fixed) commitments, e.g. ROMs: committed separately from the
    /// const columns, with their own Merkle roots carried in the proof.
    pub custom_commits: Vec<MlCustomCommit>,
    pub n_publics: u32,
    /// Stage of each challenge, in global challenge order. Only challenges
    /// with `stage <= n_stages` are ever derived/used — later stages belong
    /// to the univariate protocol (quotient/evals/FRI batching).
    pub challenge_stages: Vec<u8>,
    /// Stage of each air value, in global order.
    pub airvalue_stages: Vec<u8>,
    /// Stage of each airgroup value, in global order.
    pub airgroupvalue_stages: Vec<u8>,
    /// Stage of each proof value, in global order.
    pub proofvalue_stages: Vec<u8>,
    /// Pool of literal constants.
    pub numbers: Vec<u64>,
    pub instrs: Vec<Instr>,
    pub n_temps: u32,
    pub constraints: Vec<ConstraintIr>,
    pub max_constraint_degree: u32,
    /// Distinct row offsets referenced by any constraint or bus term (always
    /// contains 0).
    pub opening_offsets: Vec<i32>,
    /// The LogUp sum bus, when this AIR was compiled in GKR mode.
    pub bus: Option<BusIr>,
    pub params: MlParams,
}

impl AirIr {
    pub fn n_stages(&self) -> usize {
        self.cols_per_stage.len()
    }

    pub fn total_witness_cols(&self) -> usize {
        self.cols_per_stage.iter().map(|&c| c as usize).sum()
    }

    /// Total number of committed columns (witness + const + custom).
    pub fn total_cols(&self) -> usize {
        self.total_witness_cols()
            + self.n_const_cols as usize
            + self.custom_commits.iter().map(|c| c.n_cols as usize).sum::<usize>()
    }

    /// Index of `offset` in `opening_offsets`.
    pub fn offset_index(&self, offset: i32) -> Option<usize> {
        self.opening_offsets.iter().position(|&s| s == offset)
    }

    /// Format tag prefixed to `.mlinfo.bin`. Bump whenever `AirIr`'s bincode
    /// layout changes so stale proving-key artifacts fail with a clear
    /// message instead of a decode error.
    pub const MLINFO_MAGIC: &'static [u8; 8] = b"MLINFO05";

    pub fn save(&self, path: &std::path::Path) -> Result<(), MlError> {
        let mut bytes = Self::MLINFO_MAGIC.to_vec();
        bincode::serde::encode_into_std_write(self, &mut bytes, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("serializing AirIr: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| MlError::Io(format!("writing {}: {e}", path.display())))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, MlError> {
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("reading {}: {e}", path.display())))?;
        let Some(payload) = bytes.strip_prefix(Self::MLINFO_MAGIC.as_slice()) else {
            return Err(MlError::Io(format!(
                "{}: stale or foreign .mlinfo.bin (expected format {}) — re-run proofman-setup",
                path.display(),
                String::from_utf8_lossy(Self::MLINFO_MAGIC),
            )));
        };
        let (ir, _) = bincode::serde::decode_from_slice(payload, bincode::config::standard())
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

    pub fn custom(&self, commit: u8, col: u32, row_offset: i32) -> Operand {
        Operand { kind: SrcKind::Custom { commit }, idx: col, row_offset, dim: 1 }
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

    pub fn proof_value(&self, idx: u32, dim: u8) -> Operand {
        Operand { kind: SrcKind::ProofValue, idx, row_offset: 0, dim }
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
