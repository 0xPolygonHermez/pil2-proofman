//! `<air>.bin`: the prover bytecode over `Fr` (spec §4.2.5, A.6). It is the code the prover runs
//! to compute the intermediate polynomials and `Q` (§4.4), and the code `pilfflonk check` runs to
//! say which constraint fails on which row. [`write_air_bin`] writes it from what
//! `pil_info::run(…, &PilInfoCfg::bn254(), …)` returns, and [`Bytecode::read`] reads it back.
//!
//! # Format, revision 1
//!
//! The container is the STARK's `"chps"` binfile, written with `pil-info`'s `BinFileWriter`. The
//! content is pilfflonk's own. The STARK's layout of the operands (`io/parser_args.rs` in
//! `pil2-stark-setup`) is the ABI of the STARK prover's buffers, and this file does not copy it. An
//! operand here says what it is and which one, and the interpreter finds the value in its own
//! buffers. Every integer is little-endian.
//!
//! ```text
//! "chps"      4 bytes
//! version     u32     0x7066_0001: "pf" in the high half, revision 1 in the low half
//! nSections   u32     4
//! 4 × { id u32, size u64, payload }, in the order 1, 2, 3, 4
//! ```
//!
//! The version is pilfflonk's own and much larger than the STARK's (1), so the STARK's reader
//! (`BinFileUtils::openExisting(…, "chps", 1)`) refuses this file. rapidsnark's `BinFile` only
//! checks the version against a maximum and does not expose it, so section 1 repeats the version
//! for pilfflonk's reader to check for equality.
//!
//! **Section 1, header.**
//!
//! ```text
//! version     u32     0x7066_0001, the container's
//! n8          u32     32: the bytes of an element
//! r           32 bytes, the modulus of Fr, little-endian
//! ```
//!
//! **Section 2, expressions.** Every expression the passes generate code for (`expressionsCode`):
//! the intermediate polynomials, `Q`, and the expressions the hints refer to.
//!
//! ```text
//! nExpressions u32
//! nOps         u32    op records in the section
//! nConstants   u32
//! maxTemps     u32    the largest nTemps in the section
//! nExpressions × {
//!   expId      u32    the expression, as expressionsinfo numbers it; the prover looks code up by it
//!   stage      u32
//!   destType   u32    1 (cm): an intermediate polynomial, its values over H are column destId
//!                     15 (q): Q, its values over the extended coset are what the prover commits
//!                     2 (tmp): a plain value, such as an expression a hint refers to
//!   destId     u32    the cmPolsMap index when destType is cm, 0 otherwise
//!   nTemps     u32    temporaries the code uses: slots 0 … nTemps − 1
//!   result     u32    the slot holding the value once the last op has run
//!   opsOffset  u32    the entry's first op record; each entry's follow the previous entry's
//!   nOps       u32    at least 1
//!   line       string the PIL it comes from, UTF-8, NUL-terminated
//! }
//! nOps × op record
//! nConstants × 32 bytes, each a canonical Fr (< r), little-endian
//! ```
//!
//! **Section 3, constraints**, for debugging (`pilfflonk check`): every constraint, the ones the
//! intermediate polynomials add included. The value of a constraint's code at a row is its
//! numerator, which is 0 on every row the constraint holds on.
//!
//! ```text
//! nConstraints u32
//! nOps, nConstants, maxTemps   u32, as in section 2
//! nConstraints × {
//!   stage      u32
//!   firstRow   u32    the rows it holds on: firstRow ≤ i < lastRow. everyRow: 0 and N; firstRow:
//!   lastRow    u32    0 and 1; lastRow: N − 1 and N; everyFrame: offsetMin and N − offsetMax
//!   imPol      u32    1 for the constraint that defines an intermediate polynomial, 0 otherwise
//!   nTemps, result, opsOffset, nOps   u32, as in section 2
//!   line       string
//! }
//! nOps × op record
//! nConstants × 32 bytes
//! ```
//!
//! **Section 4, hints.** `nHints u32`, 0 in this revision, whose reader refuses any other value.
//! Fase 1 has no prover hints: the setup refuses them, and the witness and debug hints are not the
//! prover's (§4.2.1). Fase 2 will define the hint records after `nHints`.
//!
//! **Op record**, 8 words of u32 (32 bytes):
//!
//! ```text
//! opcode  dest  aKind aIndex aOffset  bKind bIndex bOffset
//! ```
//!
//! - `opcode`: 0 add (`a + b`), 1 sub (`a − b`) and 2 mul (`a · b`), the STARK's codes; 4 copy
//!   (`a`), with `b` written as three zeros and not read. Every value is an `Fr`, so neither the
//!   STARK's dimension combinations (its `ops` array) nor its operand swap (`sub_swap`, code 3)
//!   has anything to choose, and neither is in the file. The STARK has no way to write a copy: it
//!   gives it code 0 and a single operand, a record shorter than the others.
//! - `dest`: the slot the op writes. The temporaries are allocated by `pil-info`
//!   (`io/temporaries.rs`, which the STARK uses too): temporaries whose lifetimes do not overlap
//!   share a slot, and an op may write the slot it reads. `dest` can therefore be the slot of `a`
//!   or `b`, and the interpreter must allow it.
//! - Words, not the STARK's 16-bit arguments: indices of 2^16 and more fit.
//!
//! **Operand**, `(kind, index, offset)`. The kind is the value of the C++ `opType`
//! (`pil2-stark/src/starkpil/stark_info.hpp`), which `pil-info`'s `OpType` mirrors. The maps are
//! those of `<air>.pilfflonkinfo.json` and `pilout.globalInfo.json`.
//!
//! | kind | operand | index | offset |
//! |---|---|---|---|
//! | 0 `const` | fixed column | `constPolsMap` id: the column of `<air>.const` | row offset, as i32 |
//! | 1 `cm` | committed column | `cmPolsMap` id | row offset, as i32 |
//! | 2 `tmp` | temporary | slot, `< nTemps` | 0 |
//! | 3 `public` | public input | `publicsMap` id | 0 |
//! | 4 `airgroupvalue` | airgroup value | `airgroupValuesMap` id | 0 |
//! | 5 `challenge` | challenge | `challengesMap` id | 0 |
//! | 6 `number` | constant | its index among the section's constants | 0 |
//! | 8 `airvalue` | air value | `airValuesMap` id | 0 |
//! | 9 `proofvalue` | proof value | `proofValuesMap` id | 0 |
//! | 12 `Zi` | zerofier term of a boundary | `boundaries` index | 0 |
//! | 13 `eval` | evaluation | `evMap` index | 0 |
//!
//! No other kind is an operand. pilfflonk has no custom commits (P5) and no FRI (`xDivXSubXi`,
//! `f`), the passes emit no `x`, and `q` is only a `destType`.
//!
//! **Domains.** An expression whose `destType` is `q` runs over the extended coset `g·H'` (§4.4),
//! point by point. Every other expression, and every constraint, runs over `H`, row by row. On a
//! domain of `M = 2^e·N` points (`e = 0` on `H`), a column at offset `o` is read at point
//! `(i + 2^e·o) mod M`. For boundary 0 (`everyRow`), `Zi` is `1/Z_H(X)`. For any other boundary
//! `D` it is `Z_H(X)/Z_D(X)`, with `Z_D` as A.1 defines it: for `lastRow` that is `X − ω^(N−1)`,
//! not the STARK prover's `X − ω^N` (Annex F.8). `X` is the point of the domain. `Q`'s code is
//! `(Horner fold of the constraints) · Zi(everyRow)`, so it already divides by `Z_H` (A.1).
//! `eval` belongs to code evaluated at `ξ`, as the `qVerifier` is (in its JSON). The code of this
//! file is the prover's and has no `eval`, but the format has room for code of that form, for the
//! interpreter's verifier mode (M17); there `Zi` is taken at `ξ`.
//!
//! # From `pil-info`'s code to the file
//!
//! - **Dimension 1.** Every operand and destination has `dim` 1, or the encoder refuses the code.
//!   The result of `PilInfoCfg::goldilocks` does not fit.
//! - **The result.** The last op's destination (the intermediate polynomial's `cm`, `q`, or a
//!   `tmp`) becomes a new temporary, `result`, as the STARK encoder does. What the value is for is
//!   `destType`/`destId`. Every other destination must be a `tmp`.
//! - **Temporaries**: `pil_info::io::temporaries::get_id_maps`, with extension dimension 1.
//! - **Constants** are encoded from `CodeRef.value`, a decimal string, into 32 bytes. They must be
//!   canonical (`< r`), with no reduction. The STARK's `CodeType.value` is a `u64`, so constants do
//!   not go through it. A section keeps each distinct constant once, in the order the code first
//!   uses it.
//!
//! The encoding is deterministic: the same `PilInfoResult` gives the same bytes.
//!
//! # Reading it (M17)
//!
//! The interpreter needs no `ParserArgs`. It keeps `maxTemps` slots of one block of points per
//! thread, and it can turn the constants into Montgomery form once, at load time. It runs the ops of an
//! entry in order: it loads `a` and `b` by kind, applies the opcode and writes `dest`. After the
//! last op the value is in slot `result`. Loading by kind:
//! - `const` and `cm`: the column's values on the current domain, at the shifted point.
//! - `tmp`: the slot.
//! - `number`: the constant.
//! - `public`, `challenge`, `airvalue`, `airgroupvalue` and `proofvalue`: a scalar, the same for
//!   every point.
//! - `Zi`: its helper vector over the domain.
//! - `eval`: the evaluation.
//!
//! The interpreter resolves each `(kind, index)` with its own tables, built from
//! `pilfflonkinfo.json` (for example, from a `cmPolsMap` id to the buffer of that column's stage),
//! and not with offsets written by the setup. In the prover, the expressions are looked up by
//! `expId`. The intermediate polynomials are those of `destType` cm (`cmPolsMap` has them with
//! `imPol`), and `Q` is the one of `destType` q (the `cExpId` of `pilfflonkinfo.json`).
//!
//! `cmPolsMap` also has the STARK's pieces of the quotient, `Q0 … Q{qDeg−1}` at stage
//! `nStages + 1`. No operand refers to them: `Q`'s code computes the whole `Q`, not its pieces.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use pil_info::io::bin_file_writer::BinFileWriter;
use pil_info::io::temporaries::get_id_maps;
use pil_info::pil::gen_code::{ConstraintCodeEntry, ExpressionCodeEntry};
use pil_info::types::code::{CodeOperation, CodeType, OpType};
use pil_info::types::output::{CodeEntry, CodeRef};
use pil_info::{FieldCfg, PilInfoResult};
use proofman_pilfflonk::field::{FrBytes, FIELD_BYTES};

/// The container's type, the STARK's.
pub const BIN_FILE_TYPE: &str = "chps";

/// pilfflonk's version of the container: `"pf"` in the high half, the revision in the low half.
pub const BIN_VERSION: u32 = 0x7066_0001;

pub const HEADER_SECTION: u32 = 1;
pub const EXPRESSIONS_SECTION: u32 = 2;
pub const CONSTRAINTS_SECTION: u32 = 3;
pub const HINTS_SECTION: u32 = 4;
pub const N_SECTIONS: u32 = 4;

/// The sections, in the order the file has them.
const SECTIONS: [u32; N_SECTIONS as usize] = [HEADER_SECTION, EXPRESSIONS_SECTION, CONSTRAINTS_SECTION, HINTS_SECTION];

/// Words of an op record.
pub const OP_WORDS: usize = 8;

/// The errors of the bytecode (spec §5.4: `thiserror`, following `common/src/error_manager.rs`).
#[derive(Debug, thiserror::Error)]
pub enum BytecodeError {
    /// `pil-info`'s code has something this format cannot hold.
    #[error("Cannot encode {0}")]
    Encode(String),

    /// The file is not a revision-1 pilfflonk bytecode, or it is inconsistent.
    #[error("Invalid bytecode: {0}")]
    Format(String),

    /// The file cannot be read.
    #[error("IO error on {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The file cannot be written.
    #[error("Cannot write {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
}

pub type BytecodeResult<T> = Result<T, BytecodeError>;

fn encode_error<T>(what: impl fmt::Display) -> BytecodeResult<T> {
    Err(BytecodeError::Encode(what.to_string()))
}

fn format_error<T>(what: impl fmt::Display) -> BytecodeResult<T> {
    Err(BytecodeError::Format(what.to_string()))
}

// ---------------------------------------------------------------------------------------------
// The content of the file
// ---------------------------------------------------------------------------------------------

/// The content of `<air>.bin`. Section 4 (hints) is empty in this revision and has no field.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bytecode {
    /// Section 2.
    pub expressions: Vec<ExpressionBin>,
    /// Section 3.
    pub constraints: Vec<ConstraintBin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpressionBin {
    pub exp_id: u32,
    pub stage: u32,
    pub dest: ExpressionDest,
    pub line: String,
    pub code: Code,
}

/// What an expression's value is for: `destType` and `destId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionDest {
    /// An intermediate polynomial: its values over `H` are this `cmPolsMap` column.
    ImPol { cm_id: u32 },
    /// `Q`, over the extended coset.
    Quotient,
    /// A plain value.
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintBin {
    pub stage: u32,
    /// The rows it holds on: `first_row ≤ i < last_row`.
    pub first_row: u32,
    pub last_row: u32,
    /// It defines an intermediate polynomial.
    pub im_pol: bool,
    pub line: String,
    pub code: Code,
}

/// A code block after temporary allocation: its ops, the slots they use and the slot of its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    pub ops: Vec<Op>,
    pub n_temps: u32,
    pub result: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    Copy,
}

impl Opcode {
    fn code(self) -> u32 {
        match self {
            Opcode::Add => 0,
            Opcode::Sub => 1,
            Opcode::Mul => 2,
            Opcode::Copy => 4,
        }
    }

    fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Opcode::Add),
            1 => Some(Opcode::Sub),
            2 => Some(Opcode::Mul),
            4 => Some(Opcode::Copy),
            _ => None,
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "add" => Some(Opcode::Add),
            "sub" => Some(Opcode::Sub),
            "mul" => Some(Opcode::Mul),
            "copy" => Some(Opcode::Copy),
            _ => None,
        }
    }

    /// Its name in `pil-info`'s code.
    pub fn name(self) -> &'static str {
        match self {
            Opcode::Add => "add",
            Opcode::Sub => "sub",
            Opcode::Mul => "mul",
            Opcode::Copy => "copy",
        }
    }
}

/// `dest = a <opcode> b`, or `dest = a` for a copy: `b` is `None` exactly for a copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub opcode: Opcode,
    pub dest: u32,
    pub a: Operand,
    pub b: Option<Operand>,
}

/// An operand: the rows of the table in the module's documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Const { id: u32, offset: i32 },
    Cm { id: u32, offset: i32 },
    Tmp(u32),
    Public(u32),
    AirgroupValue(u32),
    Challenge(u32),
    Number(FrBytes),
    AirValue(u32),
    ProofValue(u32),
    Zi(u32),
    Eval(u32),
}

/// The C++ `opType` of each kind, which `pil-info`'s `OpType` mirrors in the same order.
fn op_type_code(op_type: OpType) -> u32 {
    match op_type {
        OpType::Const => 0,
        OpType::Cm => 1,
        OpType::Tmp => 2,
        OpType::Public => 3,
        OpType::Airgroupvalue => 4,
        OpType::Challenge => 5,
        OpType::Number => 6,
        OpType::StringVal => 7,
        OpType::Airvalue => 8,
        OpType::Proofvalue => 9,
        OpType::Custom => 10,
        OpType::X => 11,
        OpType::Zi => 12,
        OpType::Eval => 13,
        OpType::XDivXSubXi => 14,
        OpType::Q => 15,
        OpType::F => 16,
    }
}

fn op_type_from_code(code: u32) -> Option<OpType> {
    const ALL: [OpType; 17] = [
        OpType::Const,
        OpType::Cm,
        OpType::Tmp,
        OpType::Public,
        OpType::Airgroupvalue,
        OpType::Challenge,
        OpType::Number,
        OpType::StringVal,
        OpType::Airvalue,
        OpType::Proofvalue,
        OpType::Custom,
        OpType::X,
        OpType::Zi,
        OpType::Eval,
        OpType::XDivXSubXi,
        OpType::Q,
        OpType::F,
    ];
    ALL.into_iter().find(|&t| op_type_code(t) == code)
}

impl Operand {
    /// Its kind, as `pil-info` names it.
    pub fn op_type(&self) -> OpType {
        match self {
            Operand::Const { .. } => OpType::Const,
            Operand::Cm { .. } => OpType::Cm,
            Operand::Tmp(_) => OpType::Tmp,
            Operand::Public(_) => OpType::Public,
            Operand::AirgroupValue(_) => OpType::Airgroupvalue,
            Operand::Challenge(_) => OpType::Challenge,
            Operand::Number(_) => OpType::Number,
            Operand::AirValue(_) => OpType::Airvalue,
            Operand::ProofValue(_) => OpType::Proofvalue,
            Operand::Zi(_) => OpType::Zi,
            Operand::Eval(_) => OpType::Eval,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// From pil-info's code
// ---------------------------------------------------------------------------------------------

/// Write `<air>.bin` at `path` for the result of `pil_info::run` with `PilInfoCfg::bn254()`.
pub fn write_air_bin(result: &PilInfoResult, path: &Path) -> BytecodeResult<()> {
    Bytecode::from_pil_info(result)?.write(path)
}

fn to_u32<T: Copy + fmt::Display + TryInto<u32>>(value: T, what: impl fmt::Display) -> BytecodeResult<u32> {
    match value.try_into() {
        Ok(v) => Ok(v),
        Err(_) => encode_error(format!("{what}: {value} does not fit in 32 bits")),
    }
}

impl Bytecode {
    /// The bytecode of the result of `pil_info::run` with `PilInfoCfg::bn254()`.
    ///
    /// The hints the passes collected are not encoded (section 4 is empty in Fase 1): the setup
    /// refuses the prover hints and ignores the others (§4.2.1).
    pub fn from_pil_info(result: &PilInfoResult) -> BytecodeResult<Self> {
        if result.fri_exp_id.is_some() {
            return encode_error("a result with a FRI polynomial: pilfflonk opens with SHPLONK (PilInfoCfg::bn254())");
        }
        let info = &result.pil_code.expressions_info;
        let expressions =
            info.expressions_code.iter().map(|e| expression_bin(e, result)).collect::<BytecodeResult<Vec<_>>>()?;
        let n = 1u64 << result.setup.pil_power;
        let constraints =
            info.constraints.iter().enumerate().map(|(i, c)| constraint_bin(i, c, n)).collect::<BytecodeResult<_>>()?;
        Ok(Bytecode { expressions, constraints })
    }
}

fn expression_bin(entry: &ExpressionCodeEntry, result: &PilInfoResult) -> BytecodeResult<ExpressionBin> {
    let context = format!("expression {}", entry.exp_id);
    let last_dest = entry.code.last().map(|c| c.dest.ref_type.as_str());
    let dest = if entry.exp_id == result.c_exp_id {
        if last_dest != Some("q") {
            return encode_error(format!("{context}: the constraint polynomial's code does not end writing q"));
        }
        ExpressionDest::Quotient
    } else if let Some(d) = &entry.dest {
        let is_im_pol = result.setup.cm_pols_map.get(d.id).is_some_and(|p| p.im_pol);
        if d.op != "cm" || !is_im_pol {
            return encode_error(format!(
                "{context}: its destination ({} {}) is not an intermediate polynomial of cmPolsMap",
                d.op, d.id
            ));
        }
        ExpressionDest::ImPol { cm_id: to_u32(d.id, &context)? }
    } else {
        ExpressionDest::Value
    };
    if dest != ExpressionDest::Quotient && last_dest == Some("q") {
        return encode_error(format!("{context}: only the constraint polynomial ({}) writes q", result.c_exp_id));
    }
    if let Some(last) = entry.code.last().filter(|c| c.dest.ref_type == "cm") {
        if !matches!(dest, ExpressionDest::ImPol { cm_id } if cm_id as usize == last.dest.id) {
            return encode_error(format!(
                "{context}: its code writes cm {}, and its destination is {dest:?}",
                last.dest.id
            ));
        }
    }
    Ok(ExpressionBin {
        exp_id: to_u32(entry.exp_id, &context)?,
        stage: to_u32(entry.stage, &context)?,
        dest,
        line: entry.line.clone(),
        code: lower(&entry.code, &context)?,
    })
}

fn constraint_bin(index: usize, entry: &ConstraintCodeEntry, n: u64) -> BytecodeResult<ConstraintBin> {
    let context = format!("constraint {index}");
    let (first_row, last_row) = match (entry.boundary.as_str(), entry.offset_min, entry.offset_max) {
        ("everyRow", _, _) => (0, n),
        ("firstRow", _, _) => (0, 1),
        ("lastRow", _, _) => (n - 1, n),
        ("everyFrame", Some(min), Some(max)) if u64::from(min) + u64::from(max) <= n => {
            (u64::from(min), n - u64::from(max))
        }
        (boundary, min, max) => {
            return encode_error(format!(
                "{context}: boundary {boundary} (offsets {min:?}, {max:?}) on {n} rows is none of \
                 everyRow, firstRow, lastRow and everyFrame within the trace"
            ))
        }
    };
    Ok(ConstraintBin {
        stage: to_u32(entry.stage, &context)?,
        first_row: to_u32(first_row, format!("{context}: first row"))?,
        last_row: to_u32(last_row, format!("{context}: last row"))?,
        im_pol: entry.im_pol != 0,
        line: entry.line.clone().unwrap_or_default(),
        code: lower(&entry.code, &context)?,
    })
}

impl Code {
    /// Lower a block of `pil-info`'s code (dimension 1) into the format: its last destination
    /// becomes the result, and its temporaries are allocated.
    pub fn from_entries(code: &[CodeEntry]) -> BytecodeResult<Self> {
        lower(code, "the code")
    }
}

/// An operand before temporary allocation: a `Tmp` holds `pil-info`'s id, not a slot.
type RawOp = (Opcode, usize, Operand, Option<Operand>);

fn lower(code: &[CodeEntry], context: &str) -> BytecodeResult<Code> {
    let Some(last) = code.len().checked_sub(1) else {
        return encode_error(format!("{context}: it has no ops"));
    };

    // The last op writes a new temporary, the result.
    let max_tmp = code
        .iter()
        .flat_map(|c| std::iter::once(&c.dest).chain(&c.src))
        .filter(|r| r.ref_type == "tmp")
        .map(|r| r.id)
        .max();
    let result_tmp = max_tmp.map_or(0, |id| id + 1);

    let mut raw: Vec<RawOp> = Vec::with_capacity(code.len());
    for (i, entry) in code.iter().enumerate() {
        let context = format!("{context}, op {i}");
        let Some(opcode) = Opcode::from_name(&entry.op) else {
            return encode_error(format!("{context}: unknown operation {}", entry.op));
        };
        let arity = if opcode == Opcode::Copy { 1 } else { 2 };
        if entry.src.len() != arity {
            return encode_error(format!("{context}: {} with {} operands", entry.op, entry.src.len()));
        }
        check_dim(&entry.dest, &context)?;
        let dest = if i == last {
            if !matches!(entry.dest.ref_type.as_str(), "tmp" | "cm" | "q") {
                return encode_error(format!("{context}: the result is written to a {}", entry.dest.ref_type));
            }
            result_tmp
        } else if entry.dest.ref_type == "tmp" {
            entry.dest.id
        } else {
            return encode_error(format!("{context}: an op before the last writes a {}", entry.dest.ref_type));
        };
        let a = operand(&entry.src[0], &context)?;
        let b = entry.src.get(1).map(|s| operand(s, &context)).transpose()?;
        raw.push((opcode, dest, a, b));
    }

    // pil-info's allocation, the STARK's too. Its input is `CodeOperation`, of which it reads the
    // kind, the id and the dim of each operand: the constants, which `CodeType` would truncate to
    // a u64, are not part of it.
    let slot_of = |t: &Operand| match *t {
        Operand::Tmp(id) => id as usize,
        _ => 0,
    };
    let code_type = |op_type: OpType, id: usize| CodeType { op_type, id: id as u64, dim: 1, ..Default::default() };
    let operation = |(opcode, dest, a, b): &RawOp| CodeOperation {
        op: opcode.name().to_string(),
        dest: code_type(OpType::Tmp, *dest),
        src: std::iter::once(a).chain(b).map(|s| code_type(s.op_type(), slot_of(s))).collect(),
    };
    let operations: Vec<CodeOperation> = raw.iter().map(operation).collect();
    let max_id = result_tmp + 1;
    let mut slots = vec![-1i64; max_id];
    let mut unused = vec![-1i64; max_id];
    let (n_temps, n_ext_temps) = get_id_maps(max_id, &mut slots, &mut unused, &operations, 1);
    if n_ext_temps != 0 {
        return encode_error(format!("{context}: temporaries of the extension field"));
    }

    let slot = |id: usize| -> BytecodeResult<u32> {
        match u32::try_from(slots[id]) {
            Ok(s) => Ok(s),
            Err(_) => encode_error(format!("{context}: temporary {id} has no slot")),
        }
    };
    let allocate = |t: Operand| -> BytecodeResult<Operand> {
        match t {
            Operand::Tmp(id) => Ok(Operand::Tmp(slot(id as usize)?)),
            other => Ok(other),
        }
    };
    let ops = raw
        .into_iter()
        .map(|(opcode, dest, a, b)| {
            Ok(Op { opcode, dest: slot(dest)?, a: allocate(a)?, b: b.map(allocate).transpose()? })
        })
        .collect::<BytecodeResult<Vec<_>>>()?;
    let code = Code { ops, n_temps: to_u32(n_temps, context)?, result: slot(result_tmp)? };
    // What the reader checks, so that a file the encoder writes is one it reads.
    code.check().or_else(|why| encode_error(format!("{context}: {why}")))?;
    Ok(code)
}

fn check_dim(r: &CodeRef, context: &str) -> BytecodeResult<()> {
    if r.dim == 1 {
        Ok(())
    } else {
        encode_error(format!(
            "{context}: a {} of dimension {}; over BN254 every value has dimension 1 (PilInfoCfg::bn254())",
            r.ref_type, r.dim
        ))
    }
}

fn operand(r: &CodeRef, context: &str) -> BytecodeResult<Operand> {
    check_dim(r, context)?;
    let id = || to_u32(r.id, format!("{context}: {} id", r.ref_type));
    let offset = || match i32::try_from(r.prime.unwrap_or(0)) {
        Ok(o) => Ok(o),
        Err(_) => encode_error(format!("{context}: row offset {:?} does not fit in 32 bits", r.prime)),
    };
    let op_type = match OpType::parse(&r.ref_type) {
        Ok(t) => t,
        Err(_) => return encode_error(format!("{context}: unknown operand {}", r.ref_type)),
    };
    Ok(match op_type {
        OpType::Const => Operand::Const { id: id()?, offset: offset()? },
        OpType::Cm => Operand::Cm { id: id()?, offset: offset()? },
        OpType::Tmp => Operand::Tmp(id()?),
        OpType::Public => Operand::Public(id()?),
        OpType::Airgroupvalue => Operand::AirgroupValue(id()?),
        OpType::Challenge => Operand::Challenge(id()?),
        OpType::Number => {
            let value = r.value.as_deref().unwrap_or_default();
            match FrBytes::from_decimal(value) {
                Ok(v) => Operand::Number(v),
                Err(_) => return encode_error(format!("{context}: number {value:?} is not a canonical Fr (below r)")),
            }
        }
        OpType::Airvalue => Operand::AirValue(id()?),
        OpType::Proofvalue => Operand::ProofValue(id()?),
        OpType::Zi => match r.boundary_id {
            Some(b) => Operand::Zi(to_u32(b, format!("{context}: boundary"))?),
            None => return encode_error(format!("{context}: a Zi without a boundary")),
        },
        OpType::Eval => Operand::Eval(id()?),
        OpType::StringVal | OpType::Custom | OpType::X | OpType::XDivXSubXi | OpType::Q | OpType::F => {
            return encode_error(format!("{context}: a {} is not an operand of the pilfflonk bytecode", r.ref_type))
        }
    })
}

impl Code {
    /// What the reader requires of a block, beyond the ranges of its indices: every op has the
    /// operands its opcode takes, every slot is below `n_temps` and written before it is read,
    /// and the result is written. `Err` says why not.
    fn check(&self) -> Result<(), String> {
        if self.ops.is_empty() {
            return Err("a code block with no ops".to_string());
        }
        // Each slot holds a temporary some op writes.
        if self.n_temps as usize > self.ops.len() {
            return Err(format!("{} slots for {} ops", self.n_temps, self.ops.len()));
        }
        let mut written = vec![false; self.n_temps as usize];
        let read = |t: &Operand, written: &[bool]| match *t {
            Operand::Tmp(s) if !written.get(s as usize).copied().unwrap_or(false) => {
                Err(format!("slot {s} is read before it is written, or is not below nTemps"))
            }
            _ => Ok(()),
        };
        for (i, op) in self.ops.iter().enumerate() {
            if (op.opcode == Opcode::Copy) != op.b.is_none() {
                return Err(format!("op {i}: {} with the wrong number of operands", op.opcode.name()));
            }
            read(&op.a, &written).map_err(|why| format!("op {i}: {why}"))?;
            if let Some(b) = &op.b {
                read(b, &written).map_err(|why| format!("op {i}: {why}"))?;
            }
            match written.get_mut(op.dest as usize) {
                Some(w) => *w = true,
                None => return Err(format!("op {i}: slot {} is not below nTemps", op.dest)),
            }
        }
        if !written.get(self.result as usize).copied().unwrap_or(false) {
            return Err(format!("the result, slot {}, is not written", self.result));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(0);
}

/// A section's constants: each distinct value once, in the order the code first uses it.
#[derive(Default)]
struct Constants {
    values: Vec<FrBytes>,
    index: HashMap<FrBytes, u32>,
}

impl Constants {
    fn index_of(&mut self, value: FrBytes) -> BytecodeResult<u32> {
        if let Some(&i) = self.index.get(&value) {
            return Ok(i);
        }
        let i = to_u32(self.values.len(), "the constants of a section")?;
        self.values.push(value);
        self.index.insert(value, i);
        Ok(i)
    }
}

/// `(kind, index, offset)`.
fn operand_words(t: &Operand, constants: &mut Constants) -> BytecodeResult<[u32; 3]> {
    let kind = op_type_code(t.op_type());
    Ok(match *t {
        Operand::Const { id, offset } | Operand::Cm { id, offset } => [kind, id, offset as u32],
        Operand::Number(value) => [kind, constants.index_of(value)?, 0],
        Operand::Tmp(i)
        | Operand::Public(i)
        | Operand::AirgroupValue(i)
        | Operand::Challenge(i)
        | Operand::AirValue(i)
        | Operand::ProofValue(i)
        | Operand::Zi(i)
        | Operand::Eval(i) => [kind, i, 0],
    })
}

/// A code table, sections 2 and 3: the counts, the entries (each written by `entry` after its own
/// fields and before the code's), the op records and the constants.
fn code_table<E>(
    entries: &[E],
    code_of: impl Fn(&E) -> &Code,
    line_of: impl Fn(&E) -> &str,
    fields: impl Fn(&E, &mut Vec<u8>),
) -> BytecodeResult<Vec<u8>> {
    let mut constants = Constants::default();
    let mut records: Vec<u8> = Vec::new();
    let mut headers: Vec<u8> = Vec::new();
    let mut n_ops: u32 = 0;
    let mut max_temps: u32 = 0;
    for entry in entries {
        let code = code_of(entry);
        code.check().or_else(encode_error)?;
        let entry_ops = to_u32(code.ops.len(), "the ops of an entry")?;
        fields(entry, &mut headers);
        put_u32(&mut headers, code.n_temps);
        put_u32(&mut headers, code.result);
        put_u32(&mut headers, n_ops);
        put_u32(&mut headers, entry_ops);
        put_string(&mut headers, line_of(entry));
        for op in &code.ops {
            put_u32(&mut records, op.opcode.code());
            put_u32(&mut records, op.dest);
            let a = operand_words(&op.a, &mut constants)?;
            let b = op.b.as_ref().map(|b| operand_words(b, &mut constants)).transpose()?.unwrap_or([0; 3]);
            for w in a.into_iter().chain(b) {
                put_u32(&mut records, w);
            }
        }
        n_ops = match n_ops.checked_add(entry_ops) {
            Some(n) => n,
            None => return encode_error("more than 2^32 ops in a section"),
        };
        max_temps = max_temps.max(code.n_temps);
    }
    let mut out = Vec::with_capacity(16 + headers.len() + records.len() + FIELD_BYTES * constants.values.len());
    put_u32(&mut out, to_u32(entries.len(), "the entries of a section")?);
    put_u32(&mut out, n_ops);
    put_u32(&mut out, to_u32(constants.values.len(), "the constants of a section")?);
    put_u32(&mut out, max_temps);
    out.extend_from_slice(&headers);
    out.extend_from_slice(&records);
    for value in &constants.values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// `r`, little-endian: the modulus of the field `PilInfoCfg::bn254()` runs the passes over.
fn r_le() -> [u8; FIELD_BYTES] {
    let mut bytes = [0u8; FIELD_BYTES];
    let digits = FieldCfg::bn254().modulus().to_bytes_le();
    bytes[..digits.len()].copy_from_slice(&digits);
    bytes
}

fn dest_words(dest: ExpressionDest) -> (u32, u32) {
    match dest {
        ExpressionDest::ImPol { cm_id } => (op_type_code(OpType::Cm), cm_id),
        ExpressionDest::Quotient => (op_type_code(OpType::Q), 0),
        ExpressionDest::Value => (op_type_code(OpType::Tmp), 0),
    }
}

impl Bytecode {
    /// The payloads of the four sections, in order.
    fn sections(&self) -> BytecodeResult<[Vec<u8>; N_SECTIONS as usize]> {
        let mut header = Vec::with_capacity(8 + FIELD_BYTES);
        put_u32(&mut header, BIN_VERSION);
        put_u32(&mut header, FIELD_BYTES as u32);
        header.extend_from_slice(&r_le());

        let expressions = code_table(
            &self.expressions,
            |e| &e.code,
            |e| &e.line,
            |e, out| {
                let (dest_type, dest_id) = dest_words(e.dest);
                for w in [e.exp_id, e.stage, dest_type, dest_id] {
                    put_u32(out, w);
                }
            },
        )?;
        let constraints = code_table(
            &self.constraints,
            |c| &c.code,
            |c| &c.line,
            |c, out| {
                for w in [c.stage, c.first_row, c.last_row, u32::from(c.im_pol)] {
                    put_u32(out, w);
                }
            },
        )?;
        if let Some((i, c)) = self.constraints.iter().enumerate().find(|(_, c)| c.first_row > c.last_row) {
            return encode_error(format!("constraint {i}: rows {}..{}", c.first_row, c.last_row));
        }
        let mut hints = Vec::with_capacity(4);
        put_u32(&mut hints, 0);
        Ok([header, expressions, constraints, hints])
    }

    /// Write the file at `path`.
    pub fn write(&self, path: &Path) -> BytecodeResult<()> {
        let sections = self.sections()?;
        let write_error = |source: anyhow::Error| BytecodeError::Write { path: path.to_path_buf(), source };
        let Some(path_str) = path.to_str() else {
            return Err(write_error(anyhow::anyhow!("the path is not UTF-8")));
        };
        let mut writer = BinFileWriter::new(path_str, BIN_FILE_TYPE, BIN_VERSION, N_SECTIONS).map_err(write_error)?;
        for (id, payload) in SECTIONS.into_iter().zip(&sections) {
            writer.start_write_section(id).map_err(write_error)?;
            writer.write_bytes(payload).map_err(write_error)?;
            writer.end_write_section().map_err(write_error)?;
        }
        writer.close().map_err(write_error)
    }
}

// ---------------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Self { bytes, pos: 0, what }
    }

    fn take(&mut self, n: usize) -> BytecodeResult<&'a [u8]> {
        match self.pos.checked_add(n).filter(|&end| end <= self.bytes.len()) {
            Some(end) => {
                let bytes = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(bytes)
            }
            None => format_error(format!("{} ends before its content does", self.what)),
        }
    }

    fn u32(&mut self) -> BytecodeResult<u32> {
        let mut word = [0u8; 4];
        word.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(word))
    }

    fn u64(&mut self) -> BytecodeResult<u64> {
        let mut word = [0u8; 8];
        word.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(word))
    }

    fn fr(&mut self) -> BytecodeResult<[u8; FIELD_BYTES]> {
        let mut bytes = [0u8; FIELD_BYTES];
        bytes.copy_from_slice(self.take(FIELD_BYTES)?);
        Ok(bytes)
    }

    fn string(&mut self) -> BytecodeResult<String> {
        let rest = &self.bytes[self.pos..];
        let Some(len) = rest.iter().position(|&b| b == 0) else {
            return format_error(format!("a string of {} has no end", self.what));
        };
        let s = match std::str::from_utf8(&rest[..len]) {
            Ok(s) => s.to_string(),
            Err(_) => return format_error(format!("a string of {} is not UTF-8", self.what)),
        };
        self.pos += len + 1;
        Ok(s)
    }

    fn finish(&self) -> BytecodeResult<()> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            format_error(format!("{} has {} bytes after its content", self.what, self.bytes.len() - self.pos))
        }
    }
}

/// The fields of an entry of a code table that are not its code's.
struct EntryHeader<F> {
    fields: F,
    n_temps: u32,
    result: u32,
    ops_offset: u32,
    n_ops: u32,
    line: String,
}

/// Read a code table whose entries have `N_FIELDS` words of their own.
fn read_code_table<const N_FIELDS: usize>(
    bytes: &[u8],
    what: &'static str,
) -> BytecodeResult<Vec<([u32; N_FIELDS], String, Code)>> {
    let mut r = Reader::new(bytes, what);
    let n_entries = r.u32()?;
    let n_ops = r.u32()?;
    let n_constants = r.u32()?;
    let max_temps = r.u32()?;

    let mut headers: Vec<EntryHeader<[u32; N_FIELDS]>> = Vec::new();
    let mut expected_offset: u64 = 0;
    for i in 0..n_entries {
        let mut fields = [0u32; N_FIELDS];
        for f in fields.iter_mut() {
            *f = r.u32()?;
        }
        let header = EntryHeader {
            fields,
            n_temps: r.u32()?,
            result: r.u32()?,
            ops_offset: r.u32()?,
            n_ops: r.u32()?,
            line: r.string()?,
        };
        if u64::from(header.ops_offset) != expected_offset {
            return format_error(format!(
                "{what}, entry {i}: its ops start at {}, not {expected_offset}",
                header.ops_offset
            ));
        }
        expected_offset += u64::from(header.n_ops);
        headers.push(header);
    }
    if expected_offset != u64::from(n_ops) {
        return format_error(format!("{what}: the entries have {expected_offset} ops, and nOps is {n_ops}"));
    }
    if headers.iter().map(|h| h.n_temps).max().unwrap_or(0) != max_temps {
        return format_error(format!("{what}: maxTemps is {max_temps}, which is not the largest nTemps"));
    }

    let record_bytes = (n_ops as usize).checked_mul(4 * OP_WORDS);
    let records = match record_bytes {
        Some(n) => r.take(n)?,
        None => return format_error(format!("{what}: too many ops")),
    };
    let mut constants = Vec::new();
    for i in 0..n_constants {
        match FrBytes::from_le_bytes(r.fr()?) {
            Ok(v) => constants.push(v),
            Err(_) => return format_error(format!("{what}: constant {i} is not below r")),
        }
    }
    r.finish()?;

    let mut words = records.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    let mut used = vec![false; constants.len()];
    let mut entries = Vec::with_capacity(headers.len());
    for (i, h) in headers.into_iter().enumerate() {
        let mut ops = Vec::with_capacity(h.n_ops as usize);
        for j in 0..h.n_ops {
            let mut record = [0u32; OP_WORDS];
            for w in record.iter_mut() {
                *w = words.next().unwrap_or_default();
            }
            let context = format!("{what}, entry {i}, op {j}");
            let Some(opcode) = Opcode::from_code(record[0]) else {
                return format_error(format!("{context}: unknown opcode {}", record[0]));
            };
            let a = read_operand(&record[2..5], &constants, &mut used, &context)?;
            let b = if opcode == Opcode::Copy {
                if record[5..] != [0, 0, 0] {
                    return format_error(format!("{context}: a copy's second operand is not three zeros"));
                }
                None
            } else {
                Some(read_operand(&record[5..], &constants, &mut used, &context)?)
            };
            ops.push(Op { opcode, dest: record[1], a, b });
        }
        let code = Code { ops, n_temps: h.n_temps, result: h.result };
        code.check().or_else(|why| format_error(format!("{what}, entry {i}: {why}")))?;
        entries.push((h.fields, h.line, code));
    }
    if let Some(i) = used.iter().position(|u| !u) {
        return format_error(format!("{what}: constant {i} is not used"));
    }
    Ok(entries)
}

fn read_operand(words: &[u32], constants: &[FrBytes], used: &mut [bool], context: &str) -> BytecodeResult<Operand> {
    let (kind, index, offset) = (words[0], words[1], words[2]);
    let Some(op_type) = op_type_from_code(kind) else {
        return format_error(format!("{context}: kind {kind} is not an operand"));
    };
    let scalar = |t: Operand| {
        if offset == 0 {
            Ok(t)
        } else {
            format_error(format!("{context}: a {} with offset {offset}", op_type.to_str()))
        }
    };
    match op_type {
        OpType::Const => Ok(Operand::Const { id: index, offset: offset as i32 }),
        OpType::Cm => Ok(Operand::Cm { id: index, offset: offset as i32 }),
        OpType::Tmp => scalar(Operand::Tmp(index)),
        OpType::Public => scalar(Operand::Public(index)),
        OpType::Airgroupvalue => scalar(Operand::AirgroupValue(index)),
        OpType::Challenge => scalar(Operand::Challenge(index)),
        OpType::Number => {
            let Some(value) = constants.get(index as usize) else {
                return format_error(format!("{context}: constant {index}, of {}", constants.len()));
            };
            used[index as usize] = true;
            scalar(Operand::Number(*value))
        }
        OpType::Airvalue => scalar(Operand::AirValue(index)),
        OpType::Proofvalue => scalar(Operand::ProofValue(index)),
        OpType::Zi => scalar(Operand::Zi(index)),
        OpType::Eval => scalar(Operand::Eval(index)),
        OpType::StringVal | OpType::Custom | OpType::X | OpType::XDivXSubXi | OpType::Q | OpType::F => {
            format_error(format!("{context}: kind {kind} ({}) is not an operand", op_type.to_str()))
        }
    }
}

impl Bytecode {
    /// Read the file at `path`.
    pub fn read(path: &Path) -> BytecodeResult<Self> {
        let bytes = std::fs::read(path).map_err(|source| BytecodeError::Io { path: path.to_path_buf(), source })?;
        Self::from_bytes(&bytes)
    }

    /// Read a file's bytes.
    pub fn from_bytes(bytes: &[u8]) -> BytecodeResult<Self> {
        let mut r = Reader::new(bytes, "the file");
        if r.take(4)? != BIN_FILE_TYPE.as_bytes() {
            return format_error(format!("the file is not a {BIN_FILE_TYPE:?} binfile"));
        }
        let version = r.u32()?;
        if version != BIN_VERSION {
            return format_error(format!("version {version:#x}, and pilfflonk's is {BIN_VERSION:#x}"));
        }
        if r.u32()? != N_SECTIONS {
            return format_error(format!("the file does not have {N_SECTIONS} sections"));
        }
        let mut sections: Vec<&[u8]> = Vec::with_capacity(SECTIONS.len());
        for id in SECTIONS {
            let section_id = r.u32()?;
            if section_id != id {
                return format_error(format!("section {section_id} where section {id} goes"));
            }
            let size = r.u64()?;
            let payload = match usize::try_from(size) {
                Ok(size) => r.take(size)?,
                Err(_) => return format_error(format!("section {id} is too large")),
            };
            sections.push(payload);
        }
        r.finish()?;
        let [header, expressions, constraints, hints] = sections[..] else {
            return format_error("the sections");
        };

        let mut header = Reader::new(header, "the header");
        let header_version = header.u32()?;
        let n8 = header.u32()?;
        let modulus = header.fr()?;
        header.finish()?;
        if header_version != BIN_VERSION || n8 != FIELD_BYTES as u32 || modulus != r_le() {
            return format_error(format!(
                "the header (version {header_version:#x}, n8 {n8}) is not that of a revision-1 bytecode over BN254"
            ));
        }

        let expressions = read_code_table::<4>(expressions, "the expressions")?
            .into_iter()
            .enumerate()
            .map(|(i, ([exp_id, stage, dest_type, dest_id], line, code))| {
                let dest = match (dest_type, dest_id) {
                    (t, id) if t == op_type_code(OpType::Cm) => ExpressionDest::ImPol { cm_id: id },
                    (t, 0) if t == op_type_code(OpType::Q) => ExpressionDest::Quotient,
                    (t, 0) if t == op_type_code(OpType::Tmp) => ExpressionDest::Value,
                    _ => return format_error(format!("expression {i}: destination ({dest_type}, {dest_id})")),
                };
                Ok(ExpressionBin { exp_id, stage, dest, line, code })
            })
            .collect::<BytecodeResult<Vec<_>>>()?;
        let constraints = read_code_table::<4>(constraints, "the constraints")?
            .into_iter()
            .enumerate()
            .map(|(i, ([stage, first_row, last_row, im_pol], line, code))| {
                if first_row > last_row || im_pol > 1 {
                    return format_error(format!("constraint {i}: rows {first_row}..{last_row}, imPol {im_pol}"));
                }
                Ok(ConstraintBin { stage, first_row, last_row, im_pol: im_pol == 1, line, code })
            })
            .collect::<BytecodeResult<Vec<_>>>()?;

        let mut hints = Reader::new(hints, "the hints");
        let n_hints = hints.u32()?;
        hints.finish()?;
        if n_hints != 0 {
            return format_error(format!("{n_hints} hints: revision 1 has none"));
        }
        Ok(Bytecode { expressions, constraints })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_kinds_are_the_cpp_op_types() {
        // pil2-stark/src/starkpil/stark_info.hpp: typedef enum { const_ = 0, cm = 1, … } opType.
        let cpp = [
            ("const", 0),
            ("cm", 1),
            ("tmp", 2),
            ("public", 3),
            ("airgroupvalue", 4),
            ("challenge", 5),
            ("number", 6),
            ("string", 7),
            ("airvalue", 8),
            ("proofvalue", 9),
            ("custom", 10),
            ("x", 11),
            ("Zi", 12),
            ("eval", 13),
            ("xDivXSubXi", 14),
            ("q", 15),
            ("f", 16),
        ];
        for (name, code) in cpp {
            let op_type = OpType::parse(name).unwrap();
            assert_eq!(op_type_code(op_type), code, "{name}");
            assert_eq!(op_type_from_code(code), Some(op_type), "{name}");
        }
        assert_eq!(op_type_from_code(17), None);
    }

    #[test]
    fn opcodes_are_the_starks_and_copy() {
        for (opcode, code) in [(Opcode::Add, 0), (Opcode::Sub, 1), (Opcode::Mul, 2), (Opcode::Copy, 4)] {
            assert_eq!(opcode.code(), code);
            assert_eq!(Opcode::from_code(code), Some(opcode));
            assert_eq!(Opcode::from_name(opcode.name()), Some(opcode));
        }
        // The STARK's sub_swap has nothing to swap here.
        assert_eq!(Opcode::from_code(3), None);
    }

    #[test]
    fn the_header_modulus_is_r() {
        let r = num_bigint::BigUint::from_bytes_le(&r_le());
        assert_eq!(r.to_string(), proofman_pilfflonk::field::BN254_R);
    }
}
