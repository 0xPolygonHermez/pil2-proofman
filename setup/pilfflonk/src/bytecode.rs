//! `<air>.bin`: the prover bytecode over `Fr` (spec §4.2.5, A.6). It is the code the prover runs
//! to compute the intermediate polynomials and `Q` (§4.4), the prover hints that compute the
//! columns of stage 2 and above (plans M30, M31), and the code `pilfflonk check` runs to say which
//! constraint fails on which row. [`write_air_bin`] writes it from what
//! `pil_info::run(…, &PilInfoCfg::bn254(), …)` returns, and [`Bytecode::read`] reads it back.
//!
//! # Format, revision 3
//!
//! The file follows the STARK's prover `.bin` field by field, with every value of dimension 1.
//! That is the file that `setup/pil2-stark/src/io/bin_file.rs` writes, with the ops and args of
//! `io/parser_args.rs`, that `pil2-stark/src/starkpil/expressions/expressions_bin.cpp` reads and
//! that `expressions_pack.hpp` runs. It departs from the STARK's only where BN254 and dimension 1
//! force it to:
//!
//! - **The dimension fields go:** `destDim`, `nTemp3` and `maxTmp3`, and with them the
//!   temporaries of the extension field; in the hints, the `dim` of an expression.
//! - **The args are u32:** the STARK's u16 truncates an index above 65535 without a word.
//! - **The numbers are 32-byte canonical `Fr`, little-endian:** the STARK's u64 cannot hold them,
//!   in the code and in the hints.
//! - **Section 1 starts with a prefix:** pilfflonk's version, `n8`, `r` and `nStages`.
//! - **A copy is written as `add(a, 0)`.** The STARK writes a copy as an add without its second
//!   operand, three args short, and its interpreter, which reads 8 args per op, cannot run that.
//!
//! Revision 2 had no hints: its section 3 was `nHints = 0`. Revision 3 writes them (plan M30).
//!
//! Every integer is little-endian. The container is the STARK's `"chps"` binfile, written with
//! `pil-info`'s `BinFileWriter`, with the STARK's three sections:
//!
//! ```text
//! "chps"      4 bytes
//! version     u32     0x7066_0003: "pf" in the high half, revision 3 in the low half
//! nSections   u32     3
//! 3 × { id u32, size u64, payload }, in the order 1, 2, 3
//! ```
//!
//! The version is pilfflonk's own and much larger than the STARK's (1), so the STARK's reader
//! (`BinFileUtils::openExisting(…, "chps", 1)`) refuses this file. rapidsnark's `BinFile` only
//! checks the version against a maximum and does not expose it, so section 1 repeats it for
//! pilfflonk's reader to check for equality: with that check it refuses a STARK `.bin`.
//!
//! **Section 1, expressions.** Every expression the passes generate code for (`expressionsCode`):
//! the intermediate polynomials, `Q`, and the expressions the hints refer to. As in the STARK, the
//! prover finds the code of an intermediate polynomial by the `expId` of its `cmPolsMap` entry,
//! and that of `Q` by `cExpId` (`pilfflonkinfo.json`).
//!
//! ```text
//! version      u32    0x7066_0003, the container's                    ┐
//! n8           u32    32, the bytes of an element                     │ pilfflonk's prefix
//! r            32 bytes, the modulus of Fr                            │
//! nStages      u32    the AIR's: the operand types depend on it       ┘
//! maxTmp       u32    the largest nTemp of sections 1 and 2
//! maxArgs      u32    the largest nArgs of sections 1 and 2
//! maxOps       u32    the largest nOps of sections 1 and 2
//! nOps         u32    in the section
//! nArgs        u32    in the section: 8 per op
//! nNumbers     u32
//! nExpressions u32
//! nExpressions × {
//!   expId      u32    each once: the prover looks code up by it, as the STARK's does
//!   destId     u32    the temporary the value is in after the last op: that op's dest
//!   stage      u32
//!   nTemp      u32    the temporaries the code uses, 0 … nTemp − 1
//!   nOps       u32    at least 1
//!   opsOffset  u32    each entry's ops and args follow the previous entry's
//!   nArgs      u32
//!   argsOffset u32
//!   line       string the PIL it comes from, UTF-8, NUL-terminated
//! }
//! ops          nOps × u8
//! args         nArgs × u32
//! numbers      nNumbers × 32 bytes, each a canonical Fr (< r), little-endian
//! ```
//!
//! **Section 2, constraints**, for debugging (`pilfflonk check`): every constraint, including the
//! ones the intermediate polynomials add. The value of a constraint's code at a row is its
//! numerator, which is 0 on every row the constraint holds on. Its maxima are in section 1.
//!
//! ```text
//! nOps, nArgs, nNumbers, nConstraints   u32
//! nConstraints × {
//!   stage      u32
//!   destId     u32
//!   firstRow   u32    the rows it holds on, firstRow ≤ i < lastRow: everyRow 0 and N,
//!   lastRow    u32    firstRow 0 and 1, lastRow N − 1 and N, everyFrame offsetMin and N − offsetMax
//!   nTemp, nOps, opsOffset, nArgs, argsOffset   u32
//!   imPol      u32    1 for the constraint that defines an intermediate polynomial, 0 otherwise
//!   line       string
//! }
//! ops, args, numbers, as in section 1
//! ```
//!
//! **Section 3, hints.** The prover hints the setup supports, `im_col`, `gprod_col` and `gsum_col`
//! (`crate::validate::SUPPORTED_PROVER_HINTS`), in the pilout's order, as `pil-info` processes
//! them (`addHintsInfo`) and the STARK's `write_hints_section` writes them. The witness and debug
//! hints are not the prover's, and the setup ignores them (§4.2.1): they are not written. The
//! prover looks the hints up by name, so `im_col` (plan M31) needed no new revision: a revision-3
//! reader that did not compute it refused a key with it, and the setup did not write one.
//!
//! ```text
//! nHints u32
//! nHints × {
//!   name       string
//!   nFields    u32
//!   nFields × {
//!     name     string
//!     nValues  u32    one, or the elements of an array field, each with its position
//!     nValues × {
//!       op     string one of the STARK's: cm const tmp number string public challenge
//!                     airvalue airgroupvalue proofvalue (no custom, P5)
//!       number: 32 bytes, a canonical Fr, little-endian; string: string; the others: id u32
//!       rowOffsetIndex u32   cm and const only: the index of its row offset in openingPoints
//!       nPos   u32, pos u32 × nPos   its position in the field's array; none for a single value
//!     }
//!   }
//! }
//! ```
//!
//! The `id` is the STARK's: the `cmPolsMap` index of a cm, the `constPolsMap` index of a const,
//! the `expId` of a tmp (an expression of section 1, which the reader requires), and the index in
//! its map of the rest. The STARK's `dim` of a tmp and `commitId` of a custom column are not
//! written: every value has dimension 1, and there are no custom commits.
//!
//! **Ops.** One byte per op: in the STARK, the index of its combination of dimensions. Here it is
//! always 0, which is `dim1 = dim1 ∘ dim1`.
//!
//! **Args**, 8 per op: `opType dest aType aArg1 aArg2 bType bArg1 bArg2`.
//!
//! - `opType`: the STARK's codes. 0 is add (`a + b`), 1 is sub (`a − b`), 2 is mul (`a · b`) and
//!   3 is sub_swap (`b − a`).
//! - `dest`: the temporary the op writes. The temporaries are allocated by `pil-info`
//!   (`io/temporaries.rs`), as the STARK's are: temporaries whose lifetimes do not overlap share
//!   one, and an op may write the one it reads.
//! - **The sources, in the STARK's order.** Every kind has a rank: const, cm and Zi 0; tmp 1;
//!   public 2; number 3; airvalue 4; proofvalue 5; airgroupvalue 9; challenge 11; eval 12. The
//!   encoder puts first the source of the lower rank, and a sub whose sources it swaps becomes a
//!   sub_swap. `rank(a) ≤ rank(b)` therefore always holds.
//!
//! **Operands**, `(type, arg1, arg2)`. The type is the index of the STARK's buffer. It depends on
//! `nStages`, and pilfflonk has no custom commits (P5), so the tmp buffer is `bs = nStages + 4`.
//! Where the STARK multiplies an index by 3, the dimension, here it is the index itself.
//!
//! | type | operand | arg1 | arg2 |
//! |---|---|---|---|
//! | 0 | fixed column | its column in `<air>.const` (`constPolsMap` id) | `openingPoints` index |
//! | 1 … nStages + 1 | committed column of that stage | its `stagePos` in `cmPolsMap` | `openingPoints` index |
//! | nStages + 2 | `Zi` of a boundary | 1 + its index in `boundaries` | 0 |
//! | bs | tmp | the temporary | 0 |
//! | bs + 2 | public | `publicsMap` id | 0 |
//! | bs + 3 | number | its index in the section's numbers | 0 |
//! | bs + 4 | air value | `airValuesMap` id | 0 |
//! | bs + 5 | proof value | `proofValuesMap` id | 0 |
//! | bs + 6 | airgroup value | `airgroupValuesMap` id | 0 |
//! | bs + 7 | challenge | `challengesMap` id | 0 |
//! | bs + 8 | evaluation | `evMap` id | 0 |
//!
//! Some values of the STARK's are never written here:
//! - `bs + 1`, the tmp3 buffer (dimension 3);
//! - `nStages + 3`, `xDivXSubXi` (FRI);
//! - the custom commits;
//! - `x`, `nStages + 2` with arg1 0 (PIL1).
//!
//! Because the types keep the STARK's values, the STARK's test for a source that is the same for
//! every point still works: `type > bs + 1`. The maps are those of `<air>.pilfflonkinfo.json` and
//! `pilout.globalInfo.json`, which keep `pil-info`'s. Only the code `Q` runs has `Zi` operands. An
//! evaluation belongs to code evaluated at `ξ`, as the `qVerifier` is, in its JSON. This file holds
//! the prover's code and has none, but the format can carry that code for the interpreter's verifier
//! mode (M17).
//!
//! **Semantics.**
//!
//! - **Domains.** As in the STARK, `Q`'s code (`cExpId`) runs over the extended coset `g·H'` (§4.4),
//!   point by point. Every other expression, and every constraint, runs over `H`, row by row.
//! - **Row offsets.** On a domain of `M = 2^e·N` points (`e = 0` on `H`), a column at the opening
//!   point `o = openingPoints[arg2]` is read at point `(i + 2^e·o) mod M`.
//! - **Zerofiers.** `Zi` of boundary 0 (`everyRow`) is `1/Z_H(X)`. For any other boundary `D` it is
//!   `Z_H(X)/Z_D(X)`, with `Z_D` as A.1 defines it: for `lastRow` that is `X − ω^(N−1)`, not the
//!   STARK prover's `X − ω^N` (Annex F.8). `X` is the point of the domain.
//! - **Q.** `Q`'s code is `(Horner fold of the constraints) · Zi(everyRow)`, so it already divides
//!   by `Z_H` (A.1).
//!
//! # From `pil-info`'s code to the file
//!
//! As `prepare_expressions_bin` and `get_parser_args` do for the STARK:
//!
//! - **The last destination.** The last op of an intermediate polynomial's code or of `Q`'s writes a
//!   new temporary, the one numbered `tmpUsed`, which is the `destId`. In every other code the last
//!   destination must be a tmp already, and every other destination must be a tmp.
//! - **Temporaries.** They are allocated with `pil_info::io::temporaries::get_id_maps`, with
//!   extension dimension 1.
//! - **Numbers.** They are encoded from `CodeRef.value`, a decimal string, into 32 bytes. They must
//!   be canonical (`< r`) and are not reduced. The STARK's `CodeType.value` is a `u64`, so the
//!   numbers do not go through it. Each section keeps each distinct number once, in the order its
//!   code first uses it.
//! - **Refusals.** Every operand has `dim` 1, or the encoder refuses the code, so a
//!   `PilInfoCfg::goldilocks` result does not fit.
//! - **Determinism.** The same `PilInfoResult` gives the same bytes.
//!
//! # Reading it (M17)
//!
//! It is `ExpressionsBin::loadExpressionsBin` without the dimension fields:
//! - read and check the prefix;
//! - read args as u32;
//! - read the numbers as 32-byte elements, converted to Montgomery form once, at load time;
//! - read the hints as the STARK's, with 32-byte numbers and no `dim`; the prover looks them up by
//!   name (`getHintIdsByName`) and checks their operands against the pilfflonkinfo (M30).
//!
//! The interpreter is `expressions_pack.hpp`'s over `Fr`, with the case of op 0 only: 8 args per
//! op and the same buffer types. As the STARK's allocation does, the temporaries let an op's `dest`
//! be one of its sources, and the interpreter must allow it.
//!
//! `cmPolsMap` also has the pieces of the quotient, at stage `nStages + 1`. No operand refers to
//! them: `Q`'s code computes the whole `Q`.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use pil_info::io::bin_file_writer::BinFileWriter;
use pil_info::BinFileError;
use pil_info::io::temporaries::get_id_maps;
use pil_info::pil::gen_code::{ConstraintCodeEntry, ExpressionCodeEntry, ProcessedHint, ProcessedHintField};
use pil_info::types::code::{CodeOperation, CodeType, OpType};
use pil_info::types::output::{CodeEntry, CodeRef};
use pil_info::{FieldCfg, PilInfoResult};
use proofman_pilfflonk::field::{FrBytes, FIELD_BYTES};

use crate::validate::{SUPPORTED_PROVER_HINTS, WITNESS_AND_DEBUG_HINTS};

/// The container's type, the STARK's.
pub const BIN_FILE_TYPE: &str = "chps";

/// pilfflonk's version of the container: `"pf"` in the high half, the revision in the low half.
pub const BIN_VERSION: u32 = 0x7066_0003;

/// The STARK's sections (`CHELPERS_*_SECTION`).
pub const EXPRESSIONS_SECTION: u32 = 1;
pub const CONSTRAINTS_SECTION: u32 = 2;
pub const HINTS_SECTION: u32 = 3;
pub const N_SECTIONS: u32 = 3;

/// The sections, in the order the file has them.
const SECTIONS: [u32; N_SECTIONS as usize] = [EXPRESSIONS_SECTION, CONSTRAINTS_SECTION, HINTS_SECTION];

/// Args of an op, as in the STARK.
pub const ARGS_PER_OP: usize = 8;

/// The errors of the bytecode (spec §5.4: `thiserror`, following `common/src/error_manager.rs`).
#[derive(Debug, thiserror::Error)]
pub enum BytecodeError {
    /// `pil-info`'s code has something this format cannot hold.
    #[error("Cannot encode {0}")]
    Encode(String),

    /// The file is not a revision-3 pilfflonk bytecode, or it is inconsistent.
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
        source: BinFileError,
    },
}

pub type BytecodeResult<T> = Result<T, BytecodeError>;

fn encode_error<T>(what: impl fmt::Display) -> BytecodeResult<T> {
    Err(BytecodeError::Encode(what.to_string()))
}

fn format_error<T>(what: impl fmt::Display) -> BytecodeResult<T> {
    Err(BytecodeError::Format(what.to_string()))
}

fn to_u32<T: Copy + fmt::Display + TryInto<u32>>(value: T, what: impl fmt::Display) -> BytecodeResult<u32> {
    match value.try_into() {
        Ok(v) => Ok(v),
        Err(_) => encode_error(format!("{what}: {value} does not fit in 32 bits")),
    }
}

// ---------------------------------------------------------------------------------------------
// The content of the file
// ---------------------------------------------------------------------------------------------

/// The content of `<air>.bin`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bytecode {
    /// The AIR's number of stages, on which the operand types depend.
    pub n_stages: u32,
    /// Section 1.
    pub expressions: Vec<ExpressionBin>,
    /// Section 2.
    pub constraints: Vec<ConstraintBin>,
    /// Section 3.
    pub hints: Vec<HintBin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpressionBin {
    pub exp_id: u32,
    pub stage: u32,
    pub line: String,
    pub code: Code,
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

/// A hint of section 3: the STARK's `Hint`, of named fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintBin {
    pub name: String,
    pub fields: Vec<HintFieldBin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintFieldBin {
    pub name: String,
    /// One value, or the elements of an array, each with its position.
    pub values: Vec<HintValueBin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintValueBin {
    pub operand: HintOperand,
    /// Its position in the field's array: empty for a field of one value.
    pub pos: Vec<u32>,
}

/// A value of a hint field: the STARK's `HintFieldValue`, with the ids of the STARK (the
/// `cmPolsMap` index of a committed column, not its `stagePos`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintOperand {
    /// A committed column, `cmPolsMap[id]`, at `openingPoints[opening]`.
    Cm {
        id: u32,
        opening: u32,
    },
    /// A fixed column, `constPolsMap[id]`, at `openingPoints[opening]`.
    Const {
        id: u32,
        opening: u32,
    },
    /// The expression of section 1 of this `expId`.
    Tmp(u32),
    Number(FrBytes),
    String(String),
    Public(u32),
    Challenge(u32),
    AirValue(u32),
    AirgroupValue(u32),
    ProofValue(u32),
}

impl HintOperand {
    /// The STARK's name of its kind (`opType2string`).
    pub fn op(&self) -> &'static str {
        match self {
            HintOperand::Cm { .. } => "cm",
            HintOperand::Const { .. } => "const",
            HintOperand::Tmp(_) => "tmp",
            HintOperand::Number(_) => "number",
            HintOperand::String(_) => "string",
            HintOperand::Public(_) => "public",
            HintOperand::Challenge(_) => "challenge",
            HintOperand::AirValue(_) => "airvalue",
            HintOperand::AirgroupValue(_) => "airgroupvalue",
            HintOperand::ProofValue(_) => "proofvalue",
        }
    }
}

/// A code block after temporary allocation: its ops, the temporaries they use and the one the
/// value ends in, the last op's `dest`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    pub ops: Vec<Op>,
    pub n_temp: u32,
    pub dest_id: u32,
}

/// The STARK's operation codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    /// `b − a`.
    SubSwap,
}

impl Opcode {
    fn code(self) -> u32 {
        match self {
            Opcode::Add => 0,
            Opcode::Sub => 1,
            Opcode::Mul => 2,
            Opcode::SubSwap => 3,
        }
    }

    fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Opcode::Add),
            1 => Some(Opcode::Sub),
            2 => Some(Opcode::Mul),
            3 => Some(Opcode::SubSwap),
            _ => None,
        }
    }
}

/// `dest = a <opcode> b`, with `rank(a) ≤ rank(b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub opcode: Opcode,
    pub dest: u32,
    pub a: Operand,
    pub b: Operand,
}

/// An operand: the rows of the table in the module's documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// A fixed column, the `id`-th of `<air>.const`, at `openingPoints[opening]`.
    Const {
        id: u32,
        opening: u32,
    },
    /// A committed column of `stage`, at its `stagePos`, at `openingPoints[opening]`.
    Cm {
        stage: u32,
        stage_pos: u32,
        opening: u32,
    },
    /// The zerofier term of `boundaries[boundary]`.
    Zi {
        boundary: u32,
    },
    Tmp(u32),
    Public(u32),
    Number(FrBytes),
    AirValue(u32),
    ProofValue(u32),
    AirgroupValue(u32),
    Challenge(u32),
    Eval(u32),
}

impl Operand {
    /// The STARK's rank of the operand's kind (`operations_map_value` of `io/parser_args.rs`, for
    /// dimension 1): the source of the lower rank goes first.
    pub fn rank(&self) -> u32 {
        match self {
            Operand::Const { .. } | Operand::Cm { .. } | Operand::Zi { .. } => 0,
            Operand::Tmp(_) => 1,
            Operand::Public(_) => 2,
            Operand::Number(_) => 3,
            Operand::AirValue(_) => 4,
            Operand::ProofValue(_) => 5,
            Operand::AirgroupValue(_) => 9,
            Operand::Challenge(_) => 11,
            Operand::Eval(_) => 12,
        }
    }
}

/// The STARK's buffer types for an AIR of `n_stages` stages and no custom commits.
#[derive(Clone, Copy)]
struct Types {
    n_stages: u32,
}

/// The most stages an AIR can have for its types, up to `bs + 8`, to fit in a u32.
const MAX_N_STAGES: u32 = u32::MAX - 12;

impl Types {
    fn new(n_stages: u32) -> Option<Self> {
        (n_stages <= MAX_N_STAGES).then_some(Self { n_stages })
    }

    fn zi(self) -> u32 {
        self.n_stages + 2
    }

    /// `bs`: the tmp buffer, after the stages, the zerofiers, `xDivXSubXi` and no custom commit.
    fn tmp(self) -> u32 {
        self.n_stages + 4
    }

    /// `(type, arg1, arg2)`, the number already an index. `None` for a `Zi` of the last boundary
    /// a u32 can count, whose arg1 would not fit.
    fn words(self, t: &Operand, number: u32) -> Option<[u32; 3]> {
        let bs = self.tmp();
        Some(match *t {
            Operand::Const { id, opening } => [0, id, opening],
            Operand::Cm { stage, stage_pos, opening } => [stage, stage_pos, opening],
            Operand::Zi { boundary } => [self.zi(), boundary.checked_add(1)?, 0],
            Operand::Tmp(slot) => [bs, slot, 0],
            Operand::Public(id) => [bs + 2, id, 0],
            Operand::Number(_) => [bs + 3, number, 0],
            Operand::AirValue(id) => [bs + 4, id, 0],
            Operand::ProofValue(id) => [bs + 5, id, 0],
            Operand::AirgroupValue(id) => [bs + 6, id, 0],
            Operand::Challenge(id) => [bs + 7, id, 0],
            Operand::Eval(id) => [bs + 8, id, 0],
        })
    }

    fn operand(self, [ty, arg1, arg2]: [u32; 3], numbers: &[FrBytes], context: &str) -> BytecodeResult<Operand> {
        let bs = self.tmp();
        let scalar = |t: Operand| {
            if arg2 == 0 {
                Ok(t)
            } else {
                format_error(format!("{context}: an operand of type {ty} with arg2 {arg2}"))
            }
        };
        match ty {
            0 => Ok(Operand::Const { id: arg1, opening: arg2 }),
            s if s <= self.n_stages + 1 => Ok(Operand::Cm { stage: s, stage_pos: arg1, opening: arg2 }),
            t if t == self.zi() && arg1 >= 1 => scalar(Operand::Zi { boundary: arg1 - 1 }),
            t if t == bs => scalar(Operand::Tmp(arg1)),
            t if t == bs + 2 => scalar(Operand::Public(arg1)),
            t if t == bs + 3 => match numbers.get(arg1 as usize) {
                Some(value) => scalar(Operand::Number(*value)),
                None => format_error(format!("{context}: number {arg1}, of {}", numbers.len())),
            },
            t if t == bs + 4 => scalar(Operand::AirValue(arg1)),
            t if t == bs + 5 => scalar(Operand::ProofValue(arg1)),
            t if t == bs + 6 => scalar(Operand::AirgroupValue(arg1)),
            t if t == bs + 7 => scalar(Operand::Challenge(arg1)),
            t if t == bs + 8 => scalar(Operand::Eval(arg1)),
            _ => format_error(format!(
                "{context}: ({ty}, {arg1}, {arg2}) is no operand of an AIR of {} stages",
                self.n_stages
            )),
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

/// What encoding an operand needs to know of the AIR: its stages, where each committed column is,
/// and its opening points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeContext {
    pub n_stages: u32,
    /// `(stage, stagePos)` of each `cmPolsMap` entry.
    pub cm_pols: Vec<(u32, u32)>,
    pub opening_points: Vec<i64>,
}

impl CodeContext {
    pub fn from_pil_info(result: &PilInfoResult) -> BytecodeResult<Self> {
        let setup = &result.setup;
        let cm_pols = setup
            .cm_pols_map
            .iter()
            .enumerate()
            .map(|(i, p)| match (p.stage, p.stage_pos) {
                (Some(stage), Some(stage_pos)) => {
                    Ok((to_u32(stage, format!("cmPolsMap[{i}]"))?, to_u32(stage_pos, format!("cmPolsMap[{i}]"))?))
                }
                _ => encode_error(format!("cmPolsMap[{i}] ({}) has no stage or no stagePos", p.name)),
            })
            .collect::<BytecodeResult<_>>()?;
        Ok(Self { n_stages: to_u32(setup.n_stages, "nStages")?, cm_pols, opening_points: setup.opening_points.clone() })
    }
}

impl Bytecode {
    /// The bytecode of the result of `pil_info::run` with `PilInfoCfg::bn254()`.
    ///
    /// Of the hints the passes collected, section 3 has the prover hints the setup supports; the
    /// witness and debug hints are not the prover's, and the setup ignores them (§4.2.1). Any other
    /// hint is refused: it is none the prover computes.
    pub fn from_pil_info(result: &PilInfoResult) -> BytecodeResult<Self> {
        if result.fri_exp_id.is_some() {
            return encode_error("a result with a FRI polynomial: pilfflonk opens with SHPLONK (PilInfoCfg::bn254())");
        }
        let context = CodeContext::from_pil_info(result)?;
        let info = &result.pil_code.expressions_info;
        let expressions =
            info.expressions_code.iter().map(|e| expression_bin(e, result, &context)).collect::<BytecodeResult<_>>()?;
        let n = 1u64 << result.setup.pil_power;
        if info.constraints.len() != result.setup.constraints.len() {
            return encode_error(format!(
                "{} constraint code blocks for {} constraints",
                info.constraints.len(),
                result.setup.constraints.len()
            ));
        }
        let constraints = info
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| constraint_bin(i, c, result, n, &context))
            .collect::<BytecodeResult<_>>()?;
        let mut hints = Vec::new();
        for hint in &info.hints_info {
            let name = hint.name.as_str();
            if SUPPORTED_PROVER_HINTS.contains(&name) {
                hints.push(hint_bin(hint, result, &context)?);
            } else if !WITNESS_AND_DEBUG_HINTS.contains(&name) {
                return encode_error(format!("hint `{name}`: it is not one the prover computes"));
            }
        }
        Ok(Bytecode { n_stages: context.n_stages, expressions, constraints, hints })
    }
}

/// A prover hint of the passes, `pil-info`'s `addHintsInfo` of it, as section 3 holds it.
fn hint_bin(hint: &ProcessedHint, result: &PilInfoResult, context: &CodeContext) -> BytecodeResult<HintBin> {
    let fields = hint
        .fields
        .iter()
        .map(|field| {
            let what = format!("hint `{}`, field `{}`", hint.name, field.name);
            let values = field
                .values
                .iter()
                .map(|v| {
                    let pos = v.pos.iter().map(|&p| to_u32(p, &what)).collect::<BytecodeResult<_>>()?;
                    Ok(HintValueBin { operand: hint_operand(v, result, context, &what)?, pos })
                })
                .collect::<BytecodeResult<_>>()?;
            Ok(HintFieldBin { name: field.name.clone(), values })
        })
        .collect::<BytecodeResult<_>>()?;
    Ok(HintBin { name: hint.name.clone(), fields })
}

/// A value of a hint field, as the STARK's `write_hints_section` writes it: a column with the index
/// of its offset in `openingPoints`, an expression by its `expId`, a number canonical.
fn hint_operand(
    v: &ProcessedHintField,
    result: &PilInfoResult,
    context: &CodeContext,
    what: &str,
) -> BytecodeResult<HintOperand> {
    if matches!(v.op.as_str(), "cm" | "const" | "tmp") && v.dim != Some(1) {
        return encode_error(format!(
            "{what}: a {} of dimension {:?}; over BN254 every value has dimension 1",
            v.op, v.dim
        ));
    }
    let id = || match v.id {
        Some(id) => to_u32(id, what),
        None => encode_error(format!("{what}: a {} without its id", v.op)),
    };
    // A column's offset must be an opening point: the prover reads it there.
    let opening = |n: usize, map: &str| -> BytecodeResult<(u32, u32)> {
        let id = id()?;
        if id as usize >= n {
            return encode_error(format!("{what}: {} {id} is not in {map}", v.op));
        }
        match v.row_offset_index {
            Some(i) if i >= 0 && (i as usize) < context.opening_points.len() => Ok((id, to_u32(i as usize, what)?)),
            _ => encode_error(format!(
                "{what}: {} {id} at row offset {:?}, which is not an opening point ({:?})",
                v.op, v.row_offset, context.opening_points
            )),
        }
    };
    Ok(match v.op.as_str() {
        "cm" => {
            let (id, opening) = opening(context.cm_pols.len(), "cmPolsMap")?;
            HintOperand::Cm { id, opening }
        }
        "const" => {
            let (id, opening) = opening(result.setup.const_pols_map.len(), "constPolsMap")?;
            HintOperand::Const { id, opening }
        }
        "tmp" => HintOperand::Tmp(id()?),
        "number" => {
            let value = v.value.as_deref().unwrap_or_default();
            match FrBytes::from_decimal(value) {
                Ok(n) => HintOperand::Number(n),
                Err(_) => return encode_error(format!("{what}: number {value:?} is not a canonical Fr (below r)")),
            }
        }
        "string" => HintOperand::String(v.value.clone().unwrap_or_default()),
        "public" => HintOperand::Public(id()?),
        "challenge" => HintOperand::Challenge(id()?),
        "airvalue" => HintOperand::AirValue(id()?),
        "airgroupvalue" => HintOperand::AirgroupValue(id()?),
        "proofvalue" => HintOperand::ProofValue(id()?),
        op => return encode_error(format!("{what}: a {op} is not a value of a pilfflonk hint")),
    })
}

fn expression_bin(
    entry: &ExpressionCodeEntry,
    result: &PilInfoResult,
    context: &CodeContext,
) -> BytecodeResult<ExpressionBin> {
    let what = format!("expression {}", entry.exp_id);
    // The STARK's `is_special`: Q and the intermediate polynomials, whose value goes to a new
    // temporary.
    let special = entry.exp_id == result.c_exp_id
        || result.setup.cm_pols_map.iter().any(|p| p.im_pol && p.exp_id == Some(entry.exp_id));
    let redirect = special.then_some(entry.tmp_used);
    Ok(ExpressionBin {
        exp_id: to_u32(entry.exp_id, &what)?,
        stage: to_u32(entry.stage, &what)?,
        line: entry.line.clone(),
        code: lower(&entry.code, context, redirect, &what)?,
    })
}

fn constraint_bin(
    index: usize,
    entry: &ConstraintCodeEntry,
    result: &PilInfoResult,
    n: u64,
    context: &CodeContext,
) -> BytecodeResult<ConstraintBin> {
    let what = format!("constraint {index}");
    let im_pol_code;
    let code = if entry.code.is_empty() {
        im_pol_code = im_pol_copy(index, result, &what)?;
        &im_pol_code
    } else {
        &entry.code
    };
    let (first_row, last_row) = match (entry.boundary.as_str(), entry.offset_min, entry.offset_max) {
        ("everyRow", _, _) => (0, n),
        ("firstRow", _, _) => (0, 1),
        ("lastRow", _, _) => (n - 1, n),
        ("everyFrame", Some(min), Some(max)) if u64::from(min) + u64::from(max) <= n => {
            (u64::from(min), n - u64::from(max))
        }
        (boundary, min, max) => {
            return encode_error(format!(
                "{what}: boundary {boundary} (offsets {min:?}, {max:?}) on {n} rows is none of \
                 everyRow, firstRow, lastRow and everyFrame within the trace"
            ))
        }
    };
    Ok(ConstraintBin {
        stage: to_u32(entry.stage, &what)?,
        first_row: to_u32(first_row, format!("{what}: first row"))?,
        last_row: to_u32(last_row, format!("{what}: last row"))?,
        im_pol: entry.im_pol != 0,
        line: entry.line.clone().unwrap_or_default(),
        code: lower(code, context, None, &what)?,
    })
}

/// The code of constraint `index` when its whole expression is an intermediate polynomial: a
/// copy of the im pol's column at the row. `pil-info` leaves that code empty, since it marks the
/// im pols' expressions as computed before it generates the constraints' (`gen_code.rs`,
/// `generate_constraints_debug_code`). It happens to a constraint not on `everyRow` whose
/// expression the search promotes, because its `Zi` adds 1 to its degree (A.1): `x·x − p` on
/// `firstRow` with `--max-constraint-degree 2` (plan M24).
fn im_pol_copy(index: usize, result: &PilInfoResult, what: &str) -> BytecodeResult<Vec<CodeEntry>> {
    let setup = &result.setup;
    let e = match setup.constraints.get(index) {
        Some(c) => c.e,
        None => return encode_error(format!("{what}: it is not a constraint of the AIR")),
    };
    let Some(cm) = setup.cm_pols_map.iter().position(|p| p.im_pol && p.exp_id == Some(e)) else {
        return encode_error(format!("{what}: it has no ops, and its expression {e} is no intermediate polynomial"));
    };
    let reference = |ref_type: &str, id: usize| CodeRef {
        ref_type: ref_type.to_string(),
        id,
        dim: 1,
        prime: Some(0),
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
        opening: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    };
    Ok(vec![CodeEntry { op: "copy".to_string(), dest: reference("tmp", 0), src: vec![reference("cm", cm)] }])
}

impl Code {
    /// Lower a block of `pil-info`'s code (dimension 1) into the format. With `redirect`, the last
    /// op writes a new temporary with that id, as the STARK does for `Q` and the intermediate
    /// polynomials (their last destination may be a `q` or a `cm`); without it, its destination
    /// must be a tmp.
    pub fn from_entries(code: &[CodeEntry], context: &CodeContext, redirect: Option<usize>) -> BytecodeResult<Self> {
        lower(code, context, redirect, "the code")
    }
}

/// An op before temporary allocation: a `Tmp` holds `pil-info`'s id, not the allocated one.
type RawOp = (Opcode, u32, Operand, Operand);

/// `a <op> b` in the STARK's order of sources: the higher rank second, a sub becoming a sub_swap.
fn ordered(opcode: Opcode, a: Operand, b: Operand) -> (Opcode, Operand, Operand) {
    if a.rank() > b.rank() {
        let opcode = if opcode == Opcode::Sub { Opcode::SubSwap } else { opcode };
        (opcode, b, a)
    } else {
        (opcode, a, b)
    }
}

fn lower(code: &[CodeEntry], context: &CodeContext, redirect: Option<usize>, what: &str) -> BytecodeResult<Code> {
    let Some(last) = code.len().checked_sub(1) else {
        return encode_error(format!("{what}: it has no ops"));
    };
    if let Some(id) = redirect {
        let taken =
            code.iter().flat_map(|c| std::iter::once(&c.dest).chain(&c.src)).any(|r| r.ref_type == "tmp" && r.id >= id);
        if taken {
            return encode_error(format!("{what}: a temporary numbered tmpUsed ({id}) or above"));
        }
    }

    let mut raw: Vec<RawOp> = Vec::with_capacity(code.len());
    for (i, entry) in code.iter().enumerate() {
        let what = format!("{what}, op {i}");
        check_dim(&entry.dest, &what)?;
        let dest = match (i == last, redirect) {
            (true, Some(id)) => to_u32(id, &what)?,
            _ if entry.dest.ref_type == "tmp" => to_u32(entry.dest.id, &what)?,
            _ => return encode_error(format!("{what}: it writes a {}, not a tmp", entry.dest.ref_type)),
        };
        let operands = entry.src.iter().map(|s| operand(s, context, &what)).collect::<BytecodeResult<Vec<_>>>()?;
        let (opcode, a, b) = match (entry.op.as_str(), operands.as_slice()) {
            ("add", [a, b]) => ordered(Opcode::Add, *a, *b),
            ("sub", [a, b]) => ordered(Opcode::Sub, *a, *b),
            ("mul", [a, b]) => ordered(Opcode::Mul, *a, *b),
            // The STARK's add without a second operand, completed with 0.
            ("copy", [a]) => ordered(Opcode::Add, *a, Operand::Number(FrBytes::ZERO)),
            (op, operands) => return encode_error(format!("{what}: {op} with {} operands", operands.len())),
        };
        raw.push((opcode, dest, a, b));
    }

    // pil-info's allocation, the STARK's too. Its input is `CodeOperation`, of which it reads the
    // kind, the id and the dim of each operand: the numbers, which `CodeType` would truncate to a
    // u64, are not part of it.
    let code_type = |t: &Operand| match *t {
        Operand::Tmp(id) => CodeType { op_type: OpType::Tmp, id: u64::from(id), dim: 1, ..Default::default() },
        _ => CodeType { op_type: OpType::Number, dim: 1, ..Default::default() },
    };
    let operations: Vec<CodeOperation> = raw
        .iter()
        .map(|(_, dest, a, b)| CodeOperation {
            op: String::new(),
            dest: code_type(&Operand::Tmp(*dest)),
            src: vec![code_type(a), code_type(b)],
        })
        .collect();
    let max_id = raw
        .iter()
        .flat_map(|(_, dest, a, b)| {
            let tmp = |t: &Operand| if let Operand::Tmp(id) = *t { Some(id as usize) } else { None };
            [Some(*dest as usize), tmp(a), tmp(b)]
        })
        .flatten()
        .max()
        .map_or(0, |id| id + 1);
    let mut slots = vec![-1i64; max_id];
    let mut unused = vec![-1i64; max_id];
    let (n_temp, n_temp_ext) = match get_id_maps(max_id, &mut slots, &mut unused, &operations, 1) {
        Ok(counts) => counts,
        Err(e) => return encode_error(format!("{what}: {e}")),
    };
    if n_temp_ext != 0 {
        return encode_error(format!("{what}: temporaries of the extension field"));
    }

    let slot = |id: u32| -> BytecodeResult<u32> {
        match u32::try_from(slots[id as usize]) {
            Ok(s) => Ok(s),
            Err(_) => encode_error(format!("{what}: temporary {id} has no slot")),
        }
    };
    let allocate = |t: Operand| -> BytecodeResult<Operand> {
        match t {
            Operand::Tmp(id) => Ok(Operand::Tmp(slot(id)?)),
            other => Ok(other),
        }
    };
    let ops = raw
        .into_iter()
        .map(|(opcode, dest, a, b)| Ok(Op { opcode, dest: slot(dest)?, a: allocate(a)?, b: allocate(b)? }))
        .collect::<BytecodeResult<Vec<Op>>>()?;
    let dest_id = ops[last].dest;
    let code = Code { ops, n_temp: to_u32(n_temp, what)?, dest_id };
    // What the reader checks, so that a file the encoder writes is one it reads.
    code.check().or_else(|why| encode_error(format!("{what}: {why}")))?;
    Ok(code)
}

fn check_dim(r: &CodeRef, what: &str) -> BytecodeResult<()> {
    if r.dim == 1 {
        Ok(())
    } else {
        encode_error(format!(
            "{what}: a {} of dimension {}; over BN254 every value has dimension 1 (PilInfoCfg::bn254())",
            r.ref_type, r.dim
        ))
    }
}

fn operand(r: &CodeRef, context: &CodeContext, what: &str) -> BytecodeResult<Operand> {
    check_dim(r, what)?;
    let id = || to_u32(r.id, format!("{what}: {} id", r.ref_type));
    let opening = || {
        let prime = r.prime.unwrap_or(0);
        match context.opening_points.iter().position(|&p| p == prime) {
            Some(i) => to_u32(i, what),
            None => encode_error(format!(
                "{what}: {} {} at {prime}, which is not an opening point ({:?})",
                r.ref_type, r.id, context.opening_points
            )),
        }
    };
    let op_type = match OpType::parse(&r.ref_type) {
        Ok(t) => t,
        Err(_) => return encode_error(format!("{what}: unknown operand {}", r.ref_type)),
    };
    Ok(match op_type {
        OpType::Const => Operand::Const { id: id()?, opening: opening()? },
        OpType::Cm => match context.cm_pols.get(r.id) {
            Some(&(stage, stage_pos)) if (1..=context.n_stages + 1).contains(&stage) => {
                Operand::Cm { stage, stage_pos, opening: opening()? }
            }
            Some((stage, _)) => return encode_error(format!("{what}: cm {} is of stage {stage}", r.id)),
            None => return encode_error(format!("{what}: cm {} is not in cmPolsMap", r.id)),
        },
        OpType::Tmp => Operand::Tmp(id()?),
        OpType::Public => Operand::Public(id()?),
        OpType::Airgroupvalue => Operand::AirgroupValue(id()?),
        OpType::Challenge => Operand::Challenge(id()?),
        OpType::Number => {
            let value = r.value.as_deref().unwrap_or_default();
            match FrBytes::from_decimal(value) {
                Ok(v) => Operand::Number(v),
                Err(_) => return encode_error(format!("{what}: number {value:?} is not a canonical Fr (below r)")),
            }
        }
        OpType::Airvalue => Operand::AirValue(id()?),
        OpType::Proofvalue => Operand::ProofValue(id()?),
        OpType::Zi => match r.boundary_id {
            Some(b) => Operand::Zi { boundary: to_u32(b, format!("{what}: boundary"))? },
            None => return encode_error(format!("{what}: a Zi without a boundary")),
        },
        OpType::Eval => Operand::Eval(id()?),
        OpType::StringVal | OpType::Custom | OpType::X | OpType::XDivXSubXi | OpType::Q | OpType::F => {
            return encode_error(format!("{what}: a {} is not an operand of the pilfflonk bytecode", r.ref_type))
        }
    })
}

impl Code {
    /// What the reader requires of a block, beyond the ranges of its indices: no more temporaries
    /// than ops, every temporary below `n_temp` and written before it is read, the sources in the
    /// STARK's order, and `dest_id` the last op's `dest`. `Err` says why not.
    fn check(&self) -> Result<(), String> {
        let Some(last) = self.ops.last() else {
            return Err("a code block with no ops".to_string());
        };
        // Each temporary holds a value some op writes.
        if self.n_temp as usize > self.ops.len() {
            return Err(format!("{} temporaries for {} ops", self.n_temp, self.ops.len()));
        }
        if self.dest_id != last.dest {
            return Err(format!("destId {} is not the last op's dest, {}", self.dest_id, last.dest));
        }
        let mut written = vec![false; self.n_temp as usize];
        let read = |t: &Operand, written: &[bool]| match *t {
            Operand::Tmp(s) if !written.get(s as usize).copied().unwrap_or(false) => {
                Err(format!("temporary {s} is read before it is written, or is not below nTemp"))
            }
            _ => Ok(()),
        };
        for (i, op) in self.ops.iter().enumerate() {
            if op.a.rank() > op.b.rank() {
                return Err(format!("op {i}: its sources are not in the STARK's order"));
            }
            read(&op.a, &written).map_err(|why| format!("op {i}: {why}"))?;
            read(&op.b, &written).map_err(|why| format!("op {i}: {why}"))?;
            match written.get_mut(op.dest as usize) {
                Some(w) => *w = true,
                None => return Err(format!("op {i}: temporary {} is not below nTemp", op.dest)),
            }
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

/// `put_string`, but refusing a string with a NUL, which would read back as a shorter one.
fn put_hint_string(out: &mut Vec<u8>, s: &str, what: &str) -> BytecodeResult<()> {
    if s.as_bytes().contains(&0) {
        return encode_error(format!("{what}: a string with a NUL, {s:?}"));
    }
    put_string(out, s);
    Ok(())
}

/// A section's numbers: each distinct value once, in the order the code first uses it.
#[derive(Default)]
struct Numbers {
    values: Vec<FrBytes>,
    index: HashMap<FrBytes, u32>,
}

impl Numbers {
    fn index_of(&mut self, t: &Operand) -> BytecodeResult<u32> {
        let Operand::Number(value) = *t else {
            return Ok(0);
        };
        if let Some(&i) = self.index.get(&value) {
            return Ok(i);
        }
        let i = to_u32(self.values.len(), "the numbers of a section")?;
        self.values.push(value);
        self.index.insert(value, i);
        Ok(i)
    }
}

/// A section's code, as the STARK's `write_expressions_section` and `write_constraints_section`
/// lay it out, each entry's fields written by `fields` around its code's.
struct CodeTable {
    headers: Vec<u8>,
    ops: Vec<u8>,
    args: Vec<u8>,
    numbers: Numbers,
    n_ops: u32,
    n_args: u32,
    max_temp: u32,
    max_args: u32,
    max_ops: u32,
}

impl CodeTable {
    fn new<'a, F: FnOnce(&mut Vec<u8>, [u32; 5])>(
        types: Types,
        entries: impl Iterator<Item = (&'a Code, F)>,
    ) -> BytecodeResult<Self> {
        let mut table = CodeTable {
            headers: Vec::new(),
            ops: Vec::new(),
            args: Vec::new(),
            numbers: Numbers::default(),
            n_ops: 0,
            n_args: 0,
            max_temp: 0,
            max_args: 0,
            max_ops: 0,
        };
        for (code, fields) in entries {
            code.check().or_else(encode_error)?;
            let n_ops = to_u32(code.ops.len(), "the ops of an entry")?;
            let n_args = match n_ops.checked_mul(ARGS_PER_OP as u32) {
                Some(n) => n,
                None => return encode_error("more than 2^32 args in an entry"),
            };
            fields(&mut table.headers, [code.n_temp, n_ops, table.n_ops, n_args, table.n_args]);
            for op in &code.ops {
                table.ops.push(0);
                let mut operand = |t: &Operand| -> BytecodeResult<[u32; 3]> {
                    let words = types.words(t, table.numbers.index_of(t)?);
                    // A stage out of 1 … nStages + 1, for one, would read as another operand.
                    match words {
                        Some(w) if types.operand(w, &table.numbers.values, "").ok().as_ref() == Some(t) => Ok(w),
                        _ => encode_error(format!("{t:?} in an AIR of {} stages", types.n_stages)),
                    }
                };
                let (a, b) = (operand(&op.a)?, operand(&op.b)?);
                for w in [op.opcode.code(), op.dest].into_iter().chain(a).chain(b) {
                    put_u32(&mut table.args, w);
                }
            }
            table.n_ops = match table.n_ops.checked_add(n_ops) {
                Some(n) => n,
                None => return encode_error("more than 2^32 ops in a section"),
            };
            table.n_args = match table.n_args.checked_add(n_args) {
                Some(n) => n,
                None => return encode_error("more than 2^32 args in a section"),
            };
            table.max_temp = table.max_temp.max(code.n_temp);
            table.max_args = table.max_args.max(n_args);
            table.max_ops = table.max_ops.max(n_ops);
        }
        Ok(table)
    }

    /// The headers, then the ops, the args and the numbers.
    fn write_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.headers);
        out.extend_from_slice(&self.ops);
        out.extend_from_slice(&self.args);
        for value in &self.numbers.values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// An `expId` that two expressions have, if any.
fn repeated_exp_id(expressions: &[ExpressionBin]) -> Option<u32> {
    let mut seen = std::collections::HashSet::new();
    expressions.iter().map(|e| e.exp_id).find(|&id| !seen.insert(id))
}

/// `r`, little-endian: the modulus of the field `PilInfoCfg::bn254()` runs the passes over.
fn r_le() -> [u8; FIELD_BYTES] {
    let mut bytes = [0u8; FIELD_BYTES];
    let digits = FieldCfg::bn254().modulus().to_bytes_le();
    bytes[..digits.len()].copy_from_slice(&digits);
    bytes
}

impl Bytecode {
    /// The payloads of the three sections, in order.
    fn sections(&self) -> BytecodeResult<[Vec<u8>; N_SECTIONS as usize]> {
        let Some(types) = Types::new(self.n_stages) else {
            return encode_error(format!("{} stages", self.n_stages));
        };
        if let Some((i, c)) = self.constraints.iter().enumerate().find(|(_, c)| c.first_row > c.last_row) {
            return encode_error(format!("constraint {i}: rows {}..{}", c.first_row, c.last_row));
        }
        if let Some(exp_id) = repeated_exp_id(&self.expressions) {
            return encode_error(format!("expression {exp_id} is there twice"));
        }

        // Per expression: expId, destId, stage, nTemp, nOps, opsOffset, nArgs, argsOffset, line.
        let expressions = CodeTable::new(
            types,
            self.expressions.iter().map(|e| {
                let fields = move |out: &mut Vec<u8>, [n_temp, n_ops, ops_offset, n_args, args_offset]: [u32; 5]| {
                    for w in [e.exp_id, e.code.dest_id, e.stage, n_temp, n_ops, ops_offset, n_args, args_offset] {
                        put_u32(out, w);
                    }
                    put_string(out, &e.line);
                };
                (&e.code, fields)
            }),
        )?;
        // Per constraint: stage, destId, firstRow, lastRow, nTemp, nOps, opsOffset, nArgs,
        // argsOffset, imPol, line.
        let constraints = CodeTable::new(
            types,
            self.constraints.iter().map(|c| {
                let fields = move |out: &mut Vec<u8>, [n_temp, n_ops, ops_offset, n_args, args_offset]: [u32; 5]| {
                    for w in [
                        c.stage,
                        c.code.dest_id,
                        c.first_row,
                        c.last_row,
                        n_temp,
                        n_ops,
                        ops_offset,
                        n_args,
                        args_offset,
                    ] {
                        put_u32(out, w);
                    }
                    put_u32(out, u32::from(c.im_pol));
                    put_string(out, &c.line);
                };
                (&c.code, fields)
            }),
        )?;

        let mut section1 = Vec::new();
        for w in [BIN_VERSION, FIELD_BYTES as u32] {
            put_u32(&mut section1, w);
        }
        section1.extend_from_slice(&r_le());
        put_u32(&mut section1, self.n_stages);
        // The STARK's maxima cover both sections.
        let max_temp = expressions.max_temp.max(constraints.max_temp);
        let max_args = expressions.max_args.max(constraints.max_args);
        let max_ops = expressions.max_ops.max(constraints.max_ops);
        let n_numbers = to_u32(expressions.numbers.values.len(), "the numbers of section 1")?;
        let n_expressions = to_u32(self.expressions.len(), "the expressions")?;
        for w in [max_temp, max_args, max_ops, expressions.n_ops, expressions.n_args, n_numbers, n_expressions] {
            put_u32(&mut section1, w);
        }
        expressions.write_body(&mut section1);

        let mut section2 = Vec::new();
        let n_numbers = to_u32(constraints.numbers.values.len(), "the numbers of section 2")?;
        let n_constraints = to_u32(self.constraints.len(), "the constraints")?;
        for w in [constraints.n_ops, constraints.n_args, n_numbers, n_constraints] {
            put_u32(&mut section2, w);
        }
        constraints.write_body(&mut section2);

        let mut section3 = Vec::new();
        self.write_hints(&mut section3)?;
        Ok([section1, section2, section3])
    }

    /// Section 3, as the STARK's `write_hints_section` lays it out (see [the module](self)).
    fn write_hints(&self, out: &mut Vec<u8>) -> BytecodeResult<()> {
        let count = |n: usize, what: &str| to_u32(n, what);
        put_u32(out, count(self.hints.len(), "the hints")?);
        for hint in &self.hints {
            put_hint_string(out, &hint.name, "a hint's name")?;
            put_u32(out, count(hint.fields.len(), "the fields of a hint")?);
            for field in &hint.fields {
                let what = format!("hint `{}`, field `{}`", hint.name, field.name);
                put_hint_string(out, &field.name, &what)?;
                put_u32(out, count(field.values.len(), "the values of a hint field")?);
                for value in &field.values {
                    let t = &value.operand;
                    if let HintOperand::Tmp(exp_id) = *t {
                        if !self.expressions.iter().any(|e| e.exp_id == exp_id) {
                            return encode_error(format!("{what}: expression {exp_id}, which section 1 does not have"));
                        }
                    }
                    put_string(out, t.op());
                    match t {
                        HintOperand::Number(n) => out.extend_from_slice(&n.to_le_bytes()),
                        HintOperand::String(text) => put_hint_string(out, text, &what)?,
                        HintOperand::Cm { id, opening } | HintOperand::Const { id, opening } => {
                            put_u32(out, *id);
                            put_u32(out, *opening);
                        }
                        HintOperand::Tmp(id)
                        | HintOperand::Public(id)
                        | HintOperand::Challenge(id)
                        | HintOperand::AirValue(id)
                        | HintOperand::AirgroupValue(id)
                        | HintOperand::ProofValue(id) => put_u32(out, *id),
                    }
                    put_u32(out, count(value.pos.len(), "the positions of a hint value")?);
                    for &p in &value.pos {
                        put_u32(out, p);
                    }
                }
            }
        }
        Ok(())
    }

    /// Write the file at `path`.
    pub fn write(&self, path: &Path) -> BytecodeResult<()> {
        let sections = self.sections()?;
        let write_error = |source: BinFileError| BytecodeError::Write { path: path.to_path_buf(), source };
        let Some(path_str) = path.to_str() else {
            let not_utf8 = std::io::Error::new(std::io::ErrorKind::InvalidInput, "the path is not UTF-8");
            return Err(write_error(not_utf8.into()));
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

    fn words<const N: usize>(&mut self) -> BytecodeResult<[u32; N]> {
        let mut words = [0u32; N];
        for w in words.iter_mut() {
            *w = self.u32()?;
        }
        Ok(words)
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

/// An entry of a code table as the file has it: its own fields, its code's counts and offsets
/// (`nTemp`, `nOps`, `opsOffset`, `nArgs`, `argsOffset`) and its line.
struct EntryHeader<const N: usize> {
    fields: [u32; N],
    code: [u32; 5],
    dest_id: u32,
    line: String,
}

/// The maxima a section 1 states, which cover both code tables.
#[derive(Default)]
struct Maxima {
    temp: u32,
    args: u32,
    ops: u32,
}

/// The entries of a code table whose ops, args and numbers `r` is at, and their codes.
fn read_code<const N: usize>(
    r: &mut Reader,
    types: Types,
    headers: Vec<EntryHeader<N>>,
    [n_ops, n_args, n_numbers]: [u32; 3],
    maxima: &mut Maxima,
    what: &str,
) -> BytecodeResult<Vec<([u32; N], String, Code)>> {
    let (mut ops_offset, mut args_offset) = (0u64, 0u64);
    for (i, h) in headers.iter().enumerate() {
        let [n_temp, entry_ops, entry_ops_offset, entry_args, entry_args_offset] = h.code;
        if u64::from(entry_ops_offset) != ops_offset || u64::from(entry_args_offset) != args_offset {
            return format_error(format!("{what}, entry {i}: its ops or args do not follow the previous entry's"));
        }
        if u64::from(entry_args) != u64::from(entry_ops) * ARGS_PER_OP as u64 {
            return format_error(format!("{what}, entry {i}: {entry_args} args for {entry_ops} ops"));
        }
        ops_offset += u64::from(entry_ops);
        args_offset += u64::from(entry_args);
        maxima.temp = maxima.temp.max(n_temp);
        maxima.args = maxima.args.max(entry_args);
        maxima.ops = maxima.ops.max(entry_ops);
    }
    if ops_offset != u64::from(n_ops) || args_offset != u64::from(n_args) {
        return format_error(format!(
            "{what}: the entries have {ops_offset} ops and {args_offset} args, not {n_ops} and {n_args}"
        ));
    }

    let ops = r.take(n_ops as usize)?;
    if let Some(i) = ops.iter().position(|&o| o != 0) {
        return format_error(format!("{what}: op {i} is of dimensions {}, and every value has dimension 1", ops[i]));
    }
    let args = match (n_args as usize).checked_mul(4) {
        Some(n) => r.take(n)?,
        None => return format_error(format!("{what}: too many args")),
    };
    let mut numbers = Vec::new();
    for i in 0..n_numbers {
        match FrBytes::from_le_bytes(r.fr()?) {
            Ok(v) => numbers.push(v),
            Err(_) => return format_error(format!("{what}: number {i} is not below r")),
        }
    }
    r.finish()?;

    let mut words = args.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    let mut used = vec![false; numbers.len()];
    let mut entries = Vec::with_capacity(headers.len());
    for (i, h) in headers.into_iter().enumerate() {
        let [n_temp, entry_ops, ..] = h.code;
        let mut ops = Vec::with_capacity(entry_ops as usize);
        for j in 0..entry_ops {
            let mut record = [0u32; ARGS_PER_OP];
            for w in record.iter_mut() {
                *w = words.next().unwrap_or_default();
            }
            let context = format!("{what}, entry {i}, op {j}");
            let Some(opcode) = Opcode::from_code(record[0]) else {
                return format_error(format!("{context}: unknown opType {}", record[0]));
            };
            let mut source = |words: &[u32]| -> BytecodeResult<Operand> {
                let t = types.operand([words[0], words[1], words[2]], &numbers, &context)?;
                if words[0] == types.tmp() + 3 {
                    used[words[1] as usize] = true;
                }
                Ok(t)
            };
            let a = source(&record[2..5])?;
            let b = source(&record[5..8])?;
            ops.push(Op { opcode, dest: record[1], a, b });
        }
        let code = Code { ops, n_temp, dest_id: h.dest_id };
        code.check().or_else(|why| format_error(format!("{what}, entry {i}: {why}")))?;
        entries.push((h.fields, h.line, code));
    }
    if let Some(i) = used.iter().position(|u| !u) {
        return format_error(format!("{what}: number {i} is not used"));
    }
    Ok(entries)
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
        let [expressions, constraints, hints] = sections[..] else {
            return format_error("the sections");
        };

        // Section 1: the prefix, the counts and the entries.
        let mut r1 = Reader::new(expressions, "the expressions");
        let [prefix_version, n8] = r1.words()?;
        let modulus = r1.fr()?;
        if prefix_version != BIN_VERSION || n8 != FIELD_BYTES as u32 || modulus != r_le() {
            return format_error(format!(
                "section 1 starts with version {prefix_version:#x} and n8 {n8}, not those of a revision-3 \
                 bytecode over BN254"
            ));
        }
        let [n_stages] = r1.words()?;
        let Some(types) = Types::new(n_stages) else {
            return format_error(format!("{n_stages} stages"));
        };
        let [max_temp, max_args, max_ops, n_ops, n_args, n_numbers, n_expressions] = r1.words()?;
        let mut headers = Vec::new();
        for _ in 0..n_expressions {
            let [exp_id, dest_id, stage] = r1.words()?;
            headers.push(EntryHeader { fields: [exp_id, stage], code: r1.words()?, dest_id, line: r1.string()? });
        }
        let mut maxima = Maxima::default();
        let expressions: Vec<ExpressionBin> =
            read_code(&mut r1, types, headers, [n_ops, n_args, n_numbers], &mut maxima, "the expressions")?
                .into_iter()
                .map(|([exp_id, stage], line, code)| ExpressionBin { exp_id, stage, line, code })
                .collect();
        // The prover looks code up by expId, as the STARK's does.
        if let Some(exp_id) = repeated_exp_id(&expressions) {
            return format_error(format!("expression {exp_id} is there twice"));
        }

        // Section 2.
        let mut r2 = Reader::new(constraints, "the constraints");
        let [n_ops, n_args, n_numbers, n_constraints] = r2.words()?;
        let mut headers = Vec::new();
        for _ in 0..n_constraints {
            let [stage, dest_id, first_row, last_row] = r2.words()?;
            let code = r2.words()?;
            let [im_pol] = r2.words()?;
            headers.push(EntryHeader {
                fields: [stage, first_row, last_row, im_pol],
                code,
                dest_id,
                line: r2.string()?,
            });
        }
        let constraints =
            read_code(&mut r2, types, headers, [n_ops, n_args, n_numbers], &mut maxima, "the constraints")?
                .into_iter()
                .enumerate()
                .map(|(i, ([stage, first_row, last_row, im_pol], line, code))| {
                    if first_row > last_row || im_pol > 1 {
                        return format_error(format!("constraint {i}: rows {first_row}..{last_row}, imPol {im_pol}"));
                    }
                    Ok(ConstraintBin { stage, first_row, last_row, im_pol: im_pol == 1, line, code })
                })
                .collect::<BytecodeResult<Vec<_>>>()?;
        if (max_temp, max_args, max_ops) != (maxima.temp, maxima.args, maxima.ops) {
            return format_error(format!(
                "maxTmp, maxArgs and maxOps are {max_temp}, {max_args} and {max_ops}, and the code's are {}, {} and {}",
                maxima.temp, maxima.args, maxima.ops
            ));
        }

        // Section 3.
        let mut r3 = Reader::new(hints, "the hints");
        let hints = read_hints(&mut r3, &expressions)?;
        r3.finish()?;
        Ok(Bytecode { n_stages, expressions, constraints, hints })
    }
}

/// Section 3 (see [the module](self)): each expression a hint refers to must be one of section 1.
fn read_hints(r: &mut Reader, expressions: &[ExpressionBin]) -> BytecodeResult<Vec<HintBin>> {
    let [n_hints] = r.words()?;
    let mut hints = Vec::new();
    for h in 0..n_hints {
        let name = r.string()?;
        let [n_fields] = r.words()?;
        let mut fields = Vec::new();
        for _ in 0..n_fields {
            let field = r.string()?;
            let what = format!("hint {h} (`{name}`), field `{field}`");
            let [n_values] = r.words()?;
            let mut values = Vec::new();
            for _ in 0..n_values {
                let op = r.string()?;
                let operand = match op.as_str() {
                    "cm" | "const" => {
                        let [id, opening] = r.words()?;
                        if op == "cm" {
                            HintOperand::Cm { id, opening }
                        } else {
                            HintOperand::Const { id, opening }
                        }
                    }
                    "tmp" => {
                        let [exp_id] = r.words()?;
                        if !expressions.iter().any(|e| e.exp_id == exp_id) {
                            return format_error(format!("{what}: expression {exp_id}, which section 1 does not have"));
                        }
                        HintOperand::Tmp(exp_id)
                    }
                    "number" => match FrBytes::from_le_bytes(r.fr()?) {
                        Ok(n) => HintOperand::Number(n),
                        Err(_) => return format_error(format!("{what}: a number not below r")),
                    },
                    "string" => HintOperand::String(r.string()?),
                    "public" | "challenge" | "airvalue" | "airgroupvalue" | "proofvalue" => {
                        let [id] = r.words()?;
                        match op.as_str() {
                            "public" => HintOperand::Public(id),
                            "challenge" => HintOperand::Challenge(id),
                            "airvalue" => HintOperand::AirValue(id),
                            "airgroupvalue" => HintOperand::AirgroupValue(id),
                            _ => HintOperand::ProofValue(id),
                        }
                    }
                    _ => return format_error(format!("{what}: unknown value kind {op:?}")),
                };
                let [n_pos] = r.words()?;
                let mut pos = Vec::new();
                for _ in 0..n_pos {
                    pos.push(r.u32()?);
                }
                values.push(HintValueBin { operand, pos });
            }
            fields.push(HintFieldBin { name: field, values });
        }
        hints.push(HintBin { name, fields });
    }
    Ok(hints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcodes_are_the_starks() {
        for (opcode, code) in [(Opcode::Add, 0), (Opcode::Sub, 1), (Opcode::Mul, 2), (Opcode::SubSwap, 3)] {
            assert_eq!(opcode.code(), code);
            assert_eq!(Opcode::from_code(code), Some(opcode));
        }
        assert_eq!(Opcode::from_code(4), None);
    }

    /// `operations_map_value` of `setup/pil2-stark/src/io/parser_args.rs`, for the keys of
    /// dimension 1.
    #[test]
    fn ranks_are_the_starks() {
        let one = FrBytes::from_u64(1);
        let ranks = [
            (Operand::Const { id: 0, opening: 0 }, 0),               // "const"
            (Operand::Cm { stage: 1, stage_pos: 0, opening: 0 }, 0), // "commit1"
            (Operand::Zi { boundary: 0 }, 0),                        // "Zi"
            (Operand::Tmp(0), 1),                                    // "tmp1"
            (Operand::Public(0), 2),                                 // "public"
            (Operand::Number(one), 3),                               // "number"
            (Operand::AirValue(0), 4),                               // "airvalue1"
            (Operand::ProofValue(0), 5),                             // "proofvalue1"
            (Operand::AirgroupValue(0), 9),                          // "airgroupvalue"
            (Operand::Challenge(0), 11),                             // "challenge"
            (Operand::Eval(0), 12),                                  // "eval"
        ];
        for (t, rank) in ranks {
            assert_eq!(t.rank(), rank, "{t:?}");
        }
    }

    /// The buffer types of `push_args`, for an AIR of 2 stages and no custom commits
    /// (`buffer_size = 1 + 2 + 3 = 6`), and back.
    #[test]
    fn types_are_the_starks_buffers() {
        let types = Types::new(2).unwrap();
        let wide = FrBytes::from_u64(7);
        let cases = [
            (Operand::Const { id: 5, opening: 1 }, [0, 5, 1]),
            (Operand::Cm { stage: 1, stage_pos: 4, opening: 2 }, [1, 4, 2]),
            (Operand::Cm { stage: 3, stage_pos: 0, opening: 0 }, [3, 0, 0]),
            (Operand::Zi { boundary: 0 }, [4, 1, 0]),
            (Operand::Tmp(9), [6, 9, 0]),
            (Operand::Public(1), [8, 1, 0]),
            (Operand::Number(wide), [9, 0, 0]),
            (Operand::AirValue(2), [10, 2, 0]),
            (Operand::ProofValue(3), [11, 3, 0]),
            (Operand::AirgroupValue(4), [12, 4, 0]),
            (Operand::Challenge(5), [13, 5, 0]),
            (Operand::Eval(6), [14, 6, 0]),
        ];
        for (t, words) in cases {
            assert_eq!(types.words(&t, 0), Some(words), "{t:?}");
            assert_eq!(types.operand(words, &[wide], "").unwrap(), t);
        }
        // x, xDivXSubXi, tmp3 and beyond evals are not operands here.
        for words in [[4, 0, 0], [5, 0, 0], [7, 0, 0], [15, 0, 0]] {
            assert!(types.operand(words, &[wide], "").is_err(), "{words:?}");
        }
    }

    #[test]
    fn the_prefix_modulus_is_r() {
        let r = num_bigint::BigUint::from_bytes_le(&r_le());
        assert_eq!(r.to_string(), proofman_pilfflonk::field::BN254_R);
    }
}
