//! The `.exec` file: how a recursion air's trace is gathered out of its circom witness. plonk2pil
//! writes one beside each air it builds (`write_exec_file`, setup/stark-recurser/plonk2pil), and
//! this module is where its format is defined for Rust: the constants, how each field writes a
//! coefficient, where each part lies, and [`ExecFile`], a reader of either version. The STARK
//! prover loads its own with [`load_exec_file`](crate::load_exec_file), which reads version 2 only.
//! `exec_layout.hpp`, in pil2-stark/src/starkpil/recursion_trace/ and its copy in setup/circom/,
//! mirrors the constants for the C++ readers.
//!
//! Every word is a u64, little-endian on disk:
//! - the header: `EXEC_MAGIC | version`, `n_adds`, `map_rows`, `map_cols`, and in version 3
//!   `coef_words` and `n_vars`;
//! - `n_adds` additions, `(sl, sr, coef_l, coef_r)`: a word per wire and `coef_words` per
//!   coefficient;
//! - the map: `map_rows * map_cols` u32 entries, the signal of each cell or 0 for none, row-major,
//!   two to a word (low half first), padded to a whole word with a zero half;
//! - the gate bands: [`GATE_BAND_FORMAT_VERSION`], the band count, a per-air aux word, then
//!   `(row, kind, payload)` per band.
//!
//! A band of the STARK's kinds is rows whose interior its trace expander rebuilds; the pilfflonk
//! wrap's kind, [`RANGE_CHECK_BAND_KIND`], is a row the map gathers whole, which the wrap's witness
//! counts into a column the map does not have.
//!
//! The two versions differ in the width of a coefficient, and in what the header records:
//! - version 2, [`EXEC_FORMAT_VERSION`], is Goldilocks': one word per coefficient, implied. Every
//!   STARK recursion air carries one, and it is the only version the STARK prover reads.
//! - version 3, [`EXEC_FORMAT_VERSION_WIDE`], records the width in the header, in whole words, and
//!   writes a coefficient's canonical value least significant word first. plonk2pil writes it over
//!   BN128, four words (the original pil-fflonk's 32-byte `Fr`), for the pilfflonk wrap. Its header
//!   also records `n_vars`, the r1cs's wire count, from which the additions' wires are numbered, so
//!   that a reader refuses the witness of another compile of the circuit rather than gathering
//!   other wires; version 2 leaves it to the witness's length, as `getCommitedPols` does.
//!
//! The map and the bands do not depend on the field, and are laid out alike in both.
//!
//! [`ExecFile::committed_pols`] applies a file to a circom witness, over either field: the
//! semantics of the STARK's `getCommitedPols` (pil2-stark/src/starkpil/recursion_trace/exec_file.hpp),
//! which the STARK prover runs in C++ over Goldilocks. The pilfflonk wrap runs it over BN128.

use std::path::Path;

use proofman_fields::{Bn128, Field, Goldilocks, PrimeField64, QuotientMap};
use rayon::prelude::*;

use crate::{ProofmanError, ProofmanResult};

/// "PXEC" in the high half of the first word, tagging the layout. The pre-magic layout opened with
/// `n_adds`, a small count, so no older file can be mistaken for one carrying it.
pub const EXEC_MAGIC: u64 = 0x5058_4543_0000_0000;

/// The half of the first word the magic is in. The version is in the other.
pub const EXEC_MAGIC_MASK: u64 = 0xFFFF_FFFF_0000_0000;

/// The layout over Goldilocks, a word per coefficient. Bump on any change to the header, the map or
/// their order. Mirrored by `exec_layout::EXEC_FORMAT_VERSION`.
pub const EXEC_FORMAT_VERSION: u64 = 2;

/// The layout with the coefficient width in the header, for a field wider than a word: BN128, the
/// pilfflonk wrap's. Mirrored by `exec_layout::EXEC_FORMAT_VERSION_WIDE`, which the C++ readers
/// only refuse.
pub const EXEC_FORMAT_VERSION_WIDE: u64 = 3;

/// Words of a version 2 header: `magic|version`, `n_adds`, `map_rows`, `map_cols`.
pub const EXEC_HEADER_WORDS: usize = 4;

/// Words of a version 3 header: version 2's, then `coef_words` and `n_vars`.
pub const EXEC_WIDE_HEADER_WORDS: usize = 6;

/// Layout version of the gate-band section. Mirrored by `GATE_BAND_FORMAT_VERSION` in
/// pil2-stark/src/starkpil/recursion_trace/gate_bands/gate_bands.hpp, which reads it.
pub const GATE_BAND_FORMAT_VERSION: u64 = 2;

/// Words ahead of the bands in their section: the version, the count and the aux word.
pub const GATE_BAND_HEADER_WORDS: usize = 3;

/// Words of one band: `row`, `kind`, `payload`.
pub const GATE_BAND_WORDS: usize = 3;

/// The band kind of a range-check row of the pilfflonk wrap (plonk2pil's
/// `GateBandKind::PoseidonBn128WrapRangeCheck`): a use of circom's `Num2Bytes(nBits)`, its `in` at
/// column 0 and its `⌈nBits/16⌉` chunks from column 1 ([`RANGE_CHECK_CHUNK_COLS`]), the other
/// chunk cells empty. `payload` is its number of chunks, and the section's aux word the stage-1
/// column of the multiplicity, `RANGE_MUL`, which no map entry fills.
///
/// It has no interior to rebuild: the map gathers the whole row. The band says which rows look
/// their chunk cells up in the AIR's table, every one of them, so that the wrap's witness can count
/// how many times each value of the table is looked up.
///
/// A new kind and not a new layout, so [`GATE_BAND_FORMAT_VERSION`] stays 2: the section's words
/// are as they were, and a reader of the section that does not know a kind refuses its band, as
/// `is_known_kind` does in pil2-stark's gate_bands.hpp, and the wrap's witness does. Only the wrap
/// writes it, in exec files of version 3, which the C++ readers refuse whole.
pub const RANGE_CHECK_BAND_KIND: u64 = 12;

/// The chunk cells of a range-check row, `a[1..=5]`: up to 5 chunks, `nBits ≤ 80`.
pub const RANGE_CHECK_CHUNK_COLS: std::ops::Range<usize> = 1..6;

/// Bits of a chunk: the range checks' table is `[0, 2^16)`.
pub const RANGE_CHECK_CHUNK_BITS: u32 = 16;

/// The gate band of a range-check row of the blake3 BN128 wrap's AIR: [`BLAKE3_WRAP_RANGE_CHECK_SLOTS`]
/// `Num2Bytes` uses of `payload` chunks, use `s` on columns `6s..6s+5` as a range-check row of the
/// PoseidonBN128 wrap holds one on `0..5`. Its chunk cells are looked up in the 16-bit table of the
/// AIR's blake3 lanes, whose multiplicity the wrap's witness counts them into.
pub const BLAKE3_WRAP_RANGE_CHECK_BAND_KIND: u64 = 13;

/// The `Num2Bytes` uses of a range-check row of the blake3 BN128 wrap.
pub const BLAKE3_WRAP_RANGE_CHECK_SLOTS: usize = 3;

/// The gate bands of the blake3 BN128 wrap's blocks, of [`BLAKE3_WRAP_BLOCK_ROWS`] rows from the
/// band's row: a `Blake3Node`, a `Blake3Compress` chunk and parent, `payload` their flags. The
/// wrap's witness rebuilds every column of the block but the band's from its input cells.
pub const BLAKE3_WRAP_NODE_BAND_KIND: u64 = 14;
pub const BLAKE3_WRAP_CHUNK_BAND_KIND: u64 = 15;
pub const BLAKE3_WRAP_PARENT_BAND_KIND: u64 = 16;

/// Rows of a block of the blake3 BN128 wrap: 56 G steps, then 8 of feedforward.
pub const BLAKE3_WRAP_BLOCK_ROWS: usize = 64;

/// Cells of a range-check use of the blake3 BN128 wrap: `in` and its chunks.
pub const BLAKE3_WRAP_RANGE_CHECK_CELLS: usize = 1 + RANGE_CHECK_CHUNK_COLS.end - RANGE_CHECK_CHUNK_COLS.start;

/// BLAKE3's IV and message schedule, as the blake3 BN128 wrap's blocks use them.
pub const BLAKE3_IV: [u32; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
pub const BLAKE3_SIGMA: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// The stage-1 columns of the blake3 BN128 wrap's AIR (plonk2pil's `pil/blake3_bn128/wrap.pil`), in
/// its order; pil2-stark's `pilfflonk_wrap_exec.cu` has a copy.
pub mod blake3_wrap_cols {
    pub const A: usize = 0;
    pub const ST: usize = 18;
    pub const VA: usize = 34;
    pub const VC: usize = 36;
    pub const VB: usize = 37;
    pub const VD: usize = 41;
    pub const X: usize = 45;
    pub const Y: usize = 47;
    pub const VA_P: usize = 49;
    pub const VD_P: usize = 53;
    pub const VC_P: usize = 57;
    pub const VB_P_S: usize = 61;
    pub const VA_PP: usize = 69;
    pub const VD_PP: usize = 73;
    pub const VC_PP: usize = 77;
    pub const VB_PP_XOR: usize = 81;
    pub const VB_PP_T: usize = 85;
    pub const D_INV: usize = 86;
    pub const MUL_TABLE: usize = 87;
    pub const MUL_RANGE: usize = 88;
    pub const N_COLS: usize = 89;
}

/// A field an exec file's coefficients are elements of, and how it writes one.
pub trait ExecField: Copy {
    /// The field's name, for the errors that refuse a file over another.
    const NAME: &'static str;

    /// The layout version an exec over this field is written in.
    const EXEC_VERSION: u64;

    /// Words per coefficient.
    const COEF_WORDS: usize;

    /// Writes the canonical value into `out`, [`COEF_WORDS`](Self::COEF_WORDS) long, least
    /// significant word first.
    fn write_exec_words(&self, out: &mut [u64]);

    /// The element whose canonical value `words` holds, least significant word first. `None` if the
    /// value is not below the prime or `words` is not [`COEF_WORDS`](Self::COEF_WORDS) long.
    fn read_exec_words(words: &[u64]) -> Option<Self>;
}

impl ExecField for Goldilocks {
    const NAME: &'static str = "Goldilocks";
    const EXEC_VERSION: u64 = EXEC_FORMAT_VERSION;
    const COEF_WORDS: usize = 1;

    fn write_exec_words(&self, out: &mut [u64]) {
        out.copy_from_slice(&[self.as_canonical_u64()]);
    }

    fn read_exec_words(words: &[u64]) -> Option<Self> {
        match words {
            [word] => Self::from_canonical_checked(*word),
            _ => None,
        }
    }
}

impl ExecField for Bn128 {
    const NAME: &'static str = "BN128";
    const EXEC_VERSION: u64 = EXEC_FORMAT_VERSION_WIDE;
    const COEF_WORDS: usize = 4;

    fn write_exec_words(&self, out: &mut [u64]) {
        let mut words = [0u64; 4];
        for (word, chunk) in words.iter_mut().zip(self.to_le_bytes().chunks_exact(8)) {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            *word = u64::from_le_bytes(bytes);
        }
        out.copy_from_slice(&words);
    }

    fn read_exec_words(words: &[u64]) -> Option<Self> {
        let words: &[u64; 4] = words.try_into().ok()?;
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        Self::from_le_bytes(bytes)
    }
}

/// What an exec file's header says: its version, the width of a coefficient, in version 3 the
/// r1cs's wire count, and the extents of what follows, and from them where each part lies.
///
/// The offsets are plain arithmetic, total on any layout this module hands out: [`ExecLayout::new`]
/// describes something already in memory, and [`ExecFile`] checks a file's header before it builds
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecLayout {
    version: u64,
    coef_words: usize,
    /// `Some` exactly in version 3, which records it.
    n_vars: Option<usize>,
    n_adds: usize,
    map_rows: usize,
    map_cols: usize,
}

impl ExecLayout {
    /// The layout of an exec over `F` of an r1cs of `n_vars` wires, with `n_adds` additions and a
    /// `map_rows x map_cols` map. Only version 3 records `n_vars`: over Goldilocks it is dropped.
    pub fn new<F: ExecField>(n_vars: usize, n_adds: usize, map_rows: usize, map_cols: usize) -> Self {
        // Checked when `F` is compiled in: a version 2 coefficient is a word by definition.
        const {
            assert!(
                (F::EXEC_VERSION == EXEC_FORMAT_VERSION && F::COEF_WORDS == 1)
                    || (F::EXEC_VERSION == EXEC_FORMAT_VERSION_WIDE && F::COEF_WORDS > 0),
                "an ExecField's version and coefficient width disagree"
            )
        };
        let n_vars = (F::EXEC_VERSION == EXEC_FORMAT_VERSION_WIDE).then_some(n_vars);
        Self { version: F::EXEC_VERSION, coef_words: F::COEF_WORDS, n_vars, n_adds, map_rows, map_cols }
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Words per coefficient: recorded in a version 3 header, 1 in version 2.
    pub fn coef_words(&self) -> usize {
        self.coef_words
    }

    /// The r1cs's wire count, the length of the circom witness the file applies to and the wire of
    /// its first addition: recorded in a version 3 header, `None` in version 2, which takes it from
    /// the witness.
    pub fn n_vars(&self) -> Option<usize> {
        self.n_vars
    }

    pub fn n_adds(&self) -> usize {
        self.n_adds
    }

    /// Rows of the map's live extent.
    pub fn map_rows(&self) -> usize {
        self.map_rows
    }

    /// Columns of the map's live extent.
    pub fn map_cols(&self) -> usize {
        self.map_cols
    }

    /// Words of the header, where the additions start.
    pub fn header_words(&self) -> usize {
        if self.version == EXEC_FORMAT_VERSION {
            EXEC_HEADER_WORDS
        } else {
            EXEC_WIDE_HEADER_WORDS
        }
    }

    /// Words of one addition: two wires and two coefficients.
    pub fn addition_words(&self) -> usize {
        2 + 2 * self.coef_words
    }

    /// First word of the map.
    pub fn map_at(&self) -> usize {
        self.header_words() + self.n_adds * self.addition_words()
    }

    /// First word of the gate-band section: past the map, its entries two to a word.
    pub fn bands_at(&self) -> usize {
        self.map_at() + (self.map_rows * self.map_cols).div_ceil(2)
    }

    /// Writes the header into `out[..header_words()]`.
    pub fn write_header(&self, out: &mut [u64]) {
        out[0] = EXEC_MAGIC | self.version;
        out[1] = self.n_adds as u64;
        out[2] = self.map_rows as u64;
        out[3] = self.map_cols as u64;
        if self.version != EXEC_FORMAT_VERSION {
            out[4] = self.coef_words as u64;
        }
        if let Some(n_vars) = self.n_vars {
            out[5] = n_vars as u64;
        }
    }

    /// The layout `words` opens with, if its header is one this module reads and everything up to
    /// the gate-band section's own header fits in `words`. The error completes "exec file ...".
    fn parse(words: &[u64]) -> Result<Self, String> {
        let first = words.first().copied().ok_or("is empty, with no header")?;
        if first & EXEC_MAGIC_MASK != EXEC_MAGIC {
            return Err("does not open with an exec header; it predates the current layout".into());
        }
        let version = first & !EXEC_MAGIC_MASK;
        let header_words = match version {
            EXEC_FORMAT_VERSION => EXEC_HEADER_WORDS,
            EXEC_FORMAT_VERSION_WIDE => EXEC_WIDE_HEADER_WORDS,
            _ => {
                return Err(format!(
                    "is format version {version}, but this build reads versions {EXEC_FORMAT_VERSION} and \
                     {EXEC_FORMAT_VERSION_WIDE}"
                ))
            }
        };
        let header = words.get(..header_words).ok_or_else(|| {
            format!("is {} words, too short for its version {version} header of {header_words}", words.len())
        })?;
        let (coef_words, n_vars) =
            if version == EXEC_FORMAT_VERSION { (1, None) } else { (header[4], Some(header[5])) };
        if coef_words == 0 {
            return Err("records coefficients of 0 words".into());
        }
        let size = |what: &str, value: u64| {
            usize::try_from(value).map_err(|_| format!("records {what} {value}, more than this machine addresses"))
        };
        let layout = Self {
            version,
            coef_words: size("a coefficient width of", coef_words)?,
            n_vars: n_vars.map(|n_vars| size("an r1cs wire count of", n_vars)).transpose()?,
            n_adds: size("an addition count of", header[1])?,
            map_rows: size("a map height of", header[2])?,
            map_cols: size("a map width of", header[3])?,
        };

        // Checked here, once, so that the offsets are total on what this returns.
        let section_end = (|| {
            let addition_words = layout.coef_words.checked_mul(2)?.checked_add(2)?;
            let additions = layout.n_adds.checked_mul(addition_words)?;
            let map = layout.map_rows.checked_mul(layout.map_cols)?.div_ceil(2);
            header_words.checked_add(additions)?.checked_add(map)?.checked_add(GATE_BAND_HEADER_WORDS)
        })();
        match section_end {
            Some(end) if end <= words.len() => Ok(layout),
            _ => Err(format!(
                "is {} words, too short for the {} additions with {coef_words}-word coefficients, the {} x {} \
                 map and the gate-band section its header claims",
                words.len(),
                layout.n_adds,
                layout.map_rows,
                layout.map_cols
            )),
        }
    }
}

/// An addition: the wire it introduces is `coeffs[0]*wires[0] + coeffs[1]*wires[1]`. The `i`-th
/// addition of a file is wire `n_vars + i`, where `n_vars` is the r1cs's wire count and the circom
/// witness's length, which a version 3 header records. They apply in order: one may read a wire an
/// earlier one introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecAddition<F> {
    /// `[sl, sr]`.
    pub wires: [u32; 2],
    /// `[coef_l, coef_r]`.
    pub coeffs: [F; 2],
}

/// A gate band, as written: rows whose interior a trace expander rebuilds from the boundary rather
/// than the map gathering it, or a range-check row of the pilfflonk wrap
/// ([`RANGE_CHECK_BAND_KIND`]). `kind` is plonk2pil's `GateBandKind` discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecGateBand {
    pub row: u64,
    pub kind: u64,
    /// A per-block constant the expander cannot read off the trace: BLAKE3's `flags`, a range
    /// check's number of chunks, 0 otherwise.
    pub payload: u64,
}

/// An exec file over `F`, read whole: the header, the additions, the map and the gate bands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecFile<F> {
    pub layout: ExecLayout,
    pub additions: Vec<ExecAddition<F>>,
    /// The signal of each cell of the map's live extent, row-major, `map_rows * map_cols` of them;
    /// 0 for a cell that gathers none. See [`ExecFile::map_entry`].
    pub map: Vec<u32>,
    /// The band section's per-air word: BLAKE3 packs its LANES and band width in it, Poseidon
    /// writes 0, and the pilfflonk wrap the stage-1 column of its range checks' multiplicity, if it
    /// has range-check rows (0 if not).
    pub band_aux: u64,
    pub bands: Vec<ExecGateBand>,
}

impl<F: ExecField> ExecFile<F> {
    /// Reads an exec file from its words, as plonk2pil returns them.
    pub fn from_words(words: &[u64]) -> ProofmanResult<Self> {
        Self::parse(words).map_err(|e| ProofmanError::InvalidSetup(format!("exec buffer {e}")))
    }

    /// Reads the exec file at `path`.
    pub fn read(path: impl AsRef<Path>) -> ProofmanResult<Self> {
        let path = path.as_ref();
        let invalid = |e: String| ProofmanError::InvalidSetup(format!("exec file {} {e}", path.display()));
        let bytes = std::fs::read(path).map_err(|e| invalid(format!("cannot be read: {e}")))?;
        if bytes.len() % 8 != 0 {
            return Err(invalid(format!("is {} bytes, not a multiple of 8", bytes.len())));
        }
        let words: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| {
                let mut word = [0u8; 8];
                word.copy_from_slice(chunk);
                u64::from_le_bytes(word)
            })
            .collect();
        Self::parse(&words).map_err(invalid)
    }

    /// The signal the map gathers into `row`, `col` of the trace. 0, no signal, outside the live
    /// extent: what the trace holds there before any gate band is expanded.
    pub fn map_entry(&self, row: usize, col: usize) -> u32 {
        if row < self.layout.map_rows && col < self.layout.map_cols {
            self.map[row * self.layout.map_cols + col]
        } else {
            0
        }
    }

    /// The file `words` holds, or why it is not one over `F`. The error completes "exec file ...".
    fn parse(words: &[u64]) -> Result<Self, String> {
        let layout = ExecLayout::parse(words)?;
        // Both: a version 3 header records any width, and one of a word is still not Goldilocks'.
        if (layout.version, layout.coef_words) != (F::EXEC_VERSION, F::COEF_WORDS) {
            return Err(format!(
                "is format version {} with {}-word coefficients, not version {} with the {}-word ones of {}",
                layout.version,
                layout.coef_words,
                F::EXEC_VERSION,
                F::COEF_WORDS,
                F::NAME
            ));
        }

        let additions = words[layout.header_words()..layout.map_at()]
            .chunks_exact(layout.addition_words())
            .enumerate()
            .map(|(i, add)| {
                let wire = |k: usize| {
                    u32::try_from(add[k])
                        .map_err(|_| format!("has addition {i} reading wire {}, which does not fit 32 bits", add[k]))
                };
                let coeff = |k: usize| {
                    let at = 2 + k * F::COEF_WORDS;
                    F::read_exec_words(&add[at..at + F::COEF_WORDS]).ok_or_else(|| {
                        format!("has addition {i} with a coefficient that is not an element of {}", F::NAME)
                    })
                };
                Ok(ExecAddition { wires: [wire(0)?, wire(1)?], coeffs: [coeff(0)?, coeff(1)?] })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let map_at = layout.map_at();
        let entries = layout.map_rows * layout.map_cols;
        let map = (0..entries).map(|entry| (words[map_at + entry / 2] >> (32 * (entry % 2))) as u32).collect();
        // An odd count leaves the high half of the last word unused, and the writer leaves it zero.
        if entries % 2 == 1 {
            let padding = words[map_at + entries / 2] >> 32;
            if padding != 0 {
                return Err(format!("has padding {padding:#x}, not 0, in the unused high half of its map's last word"));
            }
        }

        // `ExecLayout::parse` saw the section's header inside `words`.
        let section = &words[layout.bands_at()..];
        let (version, count, band_aux) = (section[0], section[1], section[2]);
        if version != GATE_BAND_FORMAT_VERSION {
            return Err(format!(
                "has a gate-band section of format version {version}, but this build reads version \
                 {GATE_BAND_FORMAT_VERSION}"
            ));
        }
        let body = &section[GATE_BAND_HEADER_WORDS..];
        if count.checked_mul(GATE_BAND_WORDS as u64) != Some(body.len() as u64) {
            return Err(format!(
                "has {} words past its gate-band header, not the {count} bands of {GATE_BAND_WORDS} words it claims",
                body.len()
            ));
        }
        let bands = body
            .chunks_exact(GATE_BAND_WORDS)
            .map(|band| ExecGateBand { row: band[0], kind: band[1], payload: band[2] })
            .collect();

        Ok(Self { layout, additions, map, band_aux, bands })
    }
}

/// What an exec file gathers out of a circom witness ([`ExecFile::committed_pols`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedPols<F> {
    /// Wires `1 ..= n_publics` of the witness.
    pub publics: Vec<F>,
    /// The `n_rows x n_cols` trace, row after row.
    pub trace: Vec<F>,
}

impl<F: ExecField + Field> ExecFile<F> {
    /// The trace's cells, `n_rows x n_cols`, for a circom witness of `n_witness` wires and its
    /// `n_publics` publics, or why [`committed_pols`](Self::committed_pols) refuses them.
    pub fn check_witness(
        &self,
        n_witness: usize,
        n_publics: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> ProofmanResult<usize> {
        let invalid = |e: String| Err(ProofmanError::InvalidSetup(format!("exec: {e}")));
        if let Some(n_vars) = self.layout.n_vars.filter(|&n_vars| n_vars != n_witness) {
            return invalid(format!(
                "the circom witness has {n_witness} wires, and the r1cs the exec was written for has {n_vars}: they \
                 are not of the same compile of the circuit"
            ));
        }
        if n_publics >= n_witness {
            return invalid(format!(
                "the circom witness has {n_witness} wires, too few for wire 0 and {n_publics} publics"
            ));
        }
        let (map_rows, map_cols) = (self.layout.map_rows, self.layout.map_cols);
        if map_rows > n_rows || map_cols > n_cols {
            return invalid(format!("the map is {map_rows} x {map_cols}, larger than the trace's {n_rows} x {n_cols}"));
        }
        let Some(len) = n_rows.checked_mul(n_cols) else {
            return invalid(format!("a trace of {n_rows} x {n_cols} does not fit in memory"));
        };
        let n_wires = n_witness + self.additions.len();
        if let Some(entry) = self.map.iter().position(|&wire| wire as usize >= n_wires) {
            return invalid(format!(
                "the map's cell at row {}, column {} is wire {}, and there are {n_wires}: {n_witness} of the \
                 circom witness and {} additions",
                entry / map_cols,
                entry % map_cols,
                self.map[entry],
                self.additions.len()
            ));
        }
        Ok(len)
    }

    /// The publics and the trace this file gathers out of `witness`, the circom witness of the
    /// circuit it was made for, with the semantics of the STARK's `getCommitedPols`
    /// (pil2-stark/src/starkpil/recursion_trace/exec_file.hpp), over any field:
    /// - the `n_publics` publics are wires `1 ..= n_publics`, after wire 0, the constant one;
    /// - the additions run in order, the `i`-th introducing wire `witness.len() + i`, which is
    ///   `w[sl]·coef_l + w[sr]·coef_r`: one may read a wire an earlier one introduced;
    /// - the cell at `row`, `col` of the `n_rows x n_cols` trace is the wire its map entry names,
    ///   and zero for the entry 0 and outside the map's live extent.
    ///
    /// The wires are witness indices, the r1cs's, as plonk2pil writes them, and `witness` holds a
    /// value per witness index, as circom's `getWitness` writes it. They are not circom's signal
    /// indices: `prepareSignalMap` (setup/circom/main.cpp) folds those in for the STARK's fused
    /// `getWitnessTrace` only, which reads circom's signal values.
    ///
    /// The gate bands are not expanded, as `getCommitedPols` does not expand them: the cells they
    /// fill come out zero. Where `getCommitedPols` clamps or trusts, this refuses: a witness of
    /// another length than the r1cs's `n_vars` of a version 3 header (of another compile of the
    /// circuit, whose additions would sit at other wires), a map wider or taller than the trace, an
    /// addition or a map entry reading a wire not defined before it, and a witness without its
    /// publics.
    pub fn committed_pols(
        &self,
        mut witness: Vec<F>,
        n_publics: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> ProofmanResult<CommittedPols<F>> {
        let n_witness = witness.len();
        let len = self.check_witness(n_witness, n_publics, n_rows, n_cols)?;
        let (map_rows, map_cols) = (self.layout.map_rows, self.layout.map_cols);
        let invalid = |e: String| Err(ProofmanError::InvalidSetup(format!("exec: {e}")));

        let publics = witness[1..=n_publics].to_vec();

        // In order: an addition may read a wire an earlier one introduced.
        witness.reserve_exact(self.additions.len());
        for (i, addition) in self.additions.iter().enumerate() {
            let wire = n_witness + i;
            if let Some(&read) = addition.wires.iter().find(|&&read| read as usize >= wire) {
                return invalid(format!(
                    "addition {i}, wire {wire}, reads wire {read}, which is not defined before it"
                ));
            }
            let [l, r] = addition.wires.map(|read| witness[read as usize]);
            witness.push(l * addition.coeffs[0] + r * addition.coeffs[1]);
        }

        let mut trace = vec![F::ZERO; len];
        if map_cols > 0 {
            trace[..map_rows * n_cols].par_chunks_exact_mut(n_cols).enumerate().for_each(|(row, cells)| {
                let entries = &self.map[row * map_cols..(row + 1) * map_cols];
                for (cell, &wire) in cells.iter_mut().zip(entries) {
                    if wire != 0 {
                        *cell = witness[wire as usize];
                    }
                }
            });
        }
        Ok(CommittedPols { publics, trace })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `r`, BN128's prime, least significant word first.
    const R: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

    /// A coefficient is its canonical value, least significant word first, in the field's width. A
    /// value at or above the prime, or in another width, is not one.
    #[test]
    fn a_coefficient_is_its_canonical_value_in_words() {
        let mut words = [0u64; 4];
        Bn128::NEG_ONE.write_exec_words(&mut words);
        assert_eq!(words, [R[0] - 1, R[1], R[2], R[3]]);
        assert_eq!(Bn128::read_exec_words(&words), Some(Bn128::NEG_ONE));
        assert_eq!(Bn128::read_exec_words(&R), None, "r itself");
        assert_eq!(Bn128::read_exec_words(&words[..3]), None);

        let mut word = [0u64];
        Goldilocks::NEG_ONE.write_exec_words(&mut word);
        assert_eq!(word, [0xFFFF_FFFF_0000_0000]);
        assert_eq!(Goldilocks::read_exec_words(&word), Some(Goldilocks::NEG_ONE));
        assert_eq!(Goldilocks::read_exec_words(&[0xFFFF_FFFF_0000_0001]), None, "p itself");
        assert_eq!(Goldilocks::read_exec_words(&[1, 0]), None);
    }

    /// A version 3 exec over BN128, word by word: an r1cs of 9 wires, one addition, a 1 x 1 map,
    /// whose one entry leaves the high half of its word for padding, and no bands.
    fn one_addition() -> Vec<u64> {
        let mut exec = vec![EXEC_MAGIC | EXEC_FORMAT_VERSION_WIDE, 1, 1, 1, 4, 9];
        exec.extend([3, 5]); // the wires
        exec.extend([7, 0, 0, 0]); // coef_l = 7
        exec.extend([R[0] - 1, R[1], R[2], R[3]]); // coef_r = -1
        exec.push(9); // the map's one entry, the addition, wire n_vars + 0; padding 0
        exec.extend([GATE_BAND_FORMAT_VERSION, 0, 0]); // no bands, aux 0
        exec
    }

    #[test]
    fn a_version_3_exec_reads_word_by_word() {
        let file = ExecFile::<Bn128>::from_words(&one_addition()).unwrap();
        assert_eq!(file.layout, ExecLayout::new::<Bn128>(9, 1, 1, 1));
        assert_eq!(file.layout.n_vars(), Some(9));
        assert_eq!(file.additions, [ExecAddition { wires: [3, 5], coeffs: [Bn128::from_int(7u64), Bn128::NEG_ONE] }]);
        assert_eq!((file.map_entry(0, 0), file.map_entry(1, 0), file.map_entry(0, 1)), (9, 0, 0));
        assert_eq!((file.band_aux, file.bands.len()), (0, 0));

        let mut header = [0u64; EXEC_WIDE_HEADER_WORDS];
        file.layout.write_header(&mut header);
        assert_eq!(header[..], one_addition()[..EXEC_WIDE_HEADER_WORDS]);
    }

    /// Version 2 records neither the coefficient width nor `n_vars`: a layout over Goldilocks drops
    /// the latter, and its header is the four words it was.
    #[test]
    fn a_version_2_header_records_no_wire_count() {
        let layout = ExecLayout::new::<Goldilocks>(9, 1, 1, 1);
        assert_eq!((layout.version(), layout.coef_words(), layout.n_vars()), (EXEC_FORMAT_VERSION, 1, None));
        assert_eq!(layout.header_words(), EXEC_HEADER_WORDS);
        let mut header = [u64::MAX; EXEC_WIDE_HEADER_WORDS];
        layout.write_header(&mut header);
        assert_eq!(header, [EXEC_MAGIC | EXEC_FORMAT_VERSION, 1, 1, 1, u64::MAX, u64::MAX]);
        let words = [&header[..EXEC_HEADER_WORDS], &[3, 5, 7, 11, 9, GATE_BAND_FORMAT_VERSION, 0, 0]].concat();
        assert_eq!(ExecFile::<Goldilocks>::from_words(&words).unwrap().layout, layout);
    }

    /// An edit that corrupts [`one_addition`].
    type Corruption<'a> = &'a dyn Fn(&mut Vec<u64>);

    /// Every corruption is refused saying what is wrong, and none reads past the buffer.
    #[test]
    fn a_corrupt_exec_is_refused_saying_why() {
        let r = R;
        let cases: [(&str, Corruption); 15] = [
            ("is empty", &|e| e.clear()),
            ("does not open with an exec header", &|e| e[0] = 1),
            ("is format version 4, but this build reads versions 2 and 3", &|e| e[0] = EXEC_MAGIC | 4),
            ("too short for its version 3 header of 6", &|e| e.truncate(5)),
            ("records coefficients of 0 words", &|e| e[4] = 0),
            ("is format version 2 with 1-word coefficients, not version 3 with the 4-word ones of BN128", &|e| {
                e[0] = EXEC_MAGIC | EXEC_FORMAT_VERSION
            }),
            ("is format version 3 with 1-word coefficients, not version 3 with the 4-word ones of BN128", &|e| {
                e[4] = 1
            }),
            ("too short for the 1000 additions", &|e| e[1] = 1000),
            ("too short for the 18446744073709551615 additions", &|e| e[1] = u64::MAX),
            ("reading wire 4294967296, which does not fit 32 bits", &|e| e[6] = 1 << 32),
            ("with a coefficient that is not an element of BN128", &|e| e[12..16].copy_from_slice(&r)),
            ("has padding 0xdeadbeef, not 0, in the unused high half of its map's last word", &|e| {
                e[16] |= 0xdead_beef << 32
            }),
            ("gate-band section of format version 3", &|e| e[17] = 3),
            ("has 0 words past its gate-band header, not the 1 bands", &|e| e[18] = 1),
            ("has 1 words past its gate-band header, not the 0 bands", &|e| e.push(0)),
        ];
        for (why, corrupt) in cases {
            let mut exec = one_addition();
            corrupt(&mut exec);
            let err = ExecFile::<Bn128>::from_words(&exec).unwrap_err().to_string();
            assert!(err.contains(why), "expected \"{why}\", got: {err}");
        }
    }

    /// A file is read only in the version of its field: a version 3 header of one-word coefficients
    /// is not Goldilocks', which writes version 2, though each of its coefficients is a word.
    #[test]
    fn an_exec_is_refused_in_a_field_whose_version_it_is_not() {
        let wide =
            [EXEC_MAGIC | EXEC_FORMAT_VERSION_WIDE, 1, 1, 1, 1, 3, 3, 5, 7, 11, 3, GATE_BAND_FORMAT_VERSION, 0, 0];
        let err = ExecFile::<Goldilocks>::from_words(&wide).unwrap_err().to_string();
        assert!(
            err.contains(
                "is format version 3 with 1-word coefficients, not version 2 with the 1-word ones of Goldilocks"
            ),
            "{err}"
        );
    }

    /// A file over `F` of an r1cs of `n_vars` wires with no bands, built as a reader returns one.
    fn exec<F: ExecField>(
        n_vars: usize,
        additions: Vec<ExecAddition<F>>,
        map_rows: usize,
        map_cols: usize,
        map: Vec<u32>,
    ) -> ExecFile<F> {
        assert_eq!(map.len(), map_rows * map_cols);
        let layout = ExecLayout::new::<F>(n_vars, additions.len(), map_rows, map_cols);
        ExecFile { layout, additions, map, band_aux: 0, bands: vec![] }
    }

    /// The witness `[1, 5, 7, 11, 13]`, two publics, and a file whose second addition reads the
    /// first and whose 2 x 3 map has an empty cell, applied to a 4 x 4 trace.
    fn gathered<F: ExecField + Field + QuotientMap<u64>>() -> (CommittedPols<F>, CommittedPols<F>) {
        let f = |v: u64| F::from_int(v);
        let witness = vec![f(1), f(5), f(7), f(11), f(13)];
        let additions = vec![
            // Wire 5: 2·5 + 3·7 = 31.
            ExecAddition { wires: [1, 2], coeffs: [f(2), f(3)] },
            // Wire 6: −1·31 + 1·11 = −20, from the wire the addition before introduced.
            ExecAddition { wires: [5, 3], coeffs: [F::NEG_ONE, F::ONE] },
        ];
        let file = exec(witness.len(), additions, 2, 3, vec![1, 5, 0, 6, 4, 3]);
        let got = file.committed_pols(witness, 2, 4, 4).unwrap();

        let (zero, minus_20) = (F::ZERO, -f(20));
        #[rustfmt::skip]
        let trace = vec![
            f(5),     f(31), zero,  zero,
            minus_20, f(13), f(11), zero,
            zero,     zero,  zero,  zero,
            zero,     zero,  zero,  zero,
        ];
        (got, CommittedPols { publics: vec![f(5), f(7)], trace })
    }

    /// `getCommitedPols`'s semantics, over both fields: the publics after wire 0, the additions in
    /// order, the cells the map names, and zeros for its empty cells and outside its extent.
    #[test]
    fn an_exec_gathers_the_trace_as_get_commited_pols_does() {
        let (got, want) = gathered::<Goldilocks>();
        assert_eq!(got, want);
        let (got, want) = gathered::<Bn128>();
        assert_eq!(got, want);
        // −20 is r − 20 over BN128: the coefficient −1 is four words wide.
        assert_eq!(
            got.trace[4].to_string(),
            "21888242871839275222246405745257275088548364400416034343698204186575808495597"
        );
    }

    /// What `getCommitedPols` clamps or trusts is refused, saying why.
    #[test]
    fn an_exec_that_does_not_fit_its_witness_or_trace_is_refused() {
        let f = |v: u64| Bn128::from_int(v);
        let witness = vec![f(1), f(5), f(7)];
        let add = |wires| ExecAddition { wires, coeffs: [Bn128::ONE, Bn128::ONE] };
        let cases: [(&str, ExecFile<Bn128>, usize, usize, usize); 8] = [
            // The witness of a compile of the circuit with one wire fewer or more than the exec's.
            (
                "the circom witness has 3 wires, and the r1cs the exec was written for has 2",
                exec(2, vec![], 0, 0, vec![]),
                0,
                1,
                1,
            ),
            (
                "the circom witness has 3 wires, and the r1cs the exec was written for has 4",
                exec(4, vec![], 0, 0, vec![]),
                0,
                1,
                1,
            ),
            ("too few for wire 0 and 3 publics", exec(3, vec![], 0, 0, vec![]), 3, 1, 1),
            ("the map is 2 x 1, larger than the trace's 1 x 1", exec(3, vec![], 2, 1, vec![1, 2]), 0, 1, 1),
            ("the map is 1 x 2, larger than the trace's 2 x 1", exec(3, vec![], 1, 2, vec![1, 2]), 0, 2, 1),
            ("row 1, column 0 is wire 4, and there are 4", exec(3, vec![add([1, 2])], 2, 1, vec![3, 4]), 0, 2, 1),
            ("addition 0, wire 3, reads wire 3", exec(3, vec![add([1, 3])], 0, 0, vec![]), 0, 1, 1),
            ("addition 1, wire 4, reads wire 5", exec(3, vec![add([1, 2]), add([5, 0])], 0, 0, vec![]), 0, 1, 1),
        ];
        for (why, file, n_publics, n_rows, n_cols) in cases {
            let err = file.committed_pols(witness.clone(), n_publics, n_rows, n_cols).unwrap_err().to_string();
            assert!(err.contains(why), "expected \"{why}\", got: {err}");
        }
        // An empty map gathers nothing: a trace of zeros.
        let pols = exec::<Bn128>(3, vec![], 0, 0, vec![]).committed_pols(witness, 2, 2, 3).unwrap();
        assert_eq!((pols.publics, pols.trace), (vec![f(5), f(7)], vec![Bn128::ZERO; 6]));
    }

    /// The witness of another compile, one wire longer, would have put the addition at wire 4,
    /// which nothing reads, and gathered the witness's own wire 3 in its place: refused over BN128,
    /// whose header records `n_vars`. Version 2 does not, and numbers the additions from whatever
    /// witness it is given, as `getCommitedPols` does.
    #[test]
    fn the_witness_of_another_compile_is_refused_over_bn128() {
        fn file<F: ExecField + Field>() -> ExecFile<F> {
            // Wire 3 = w1 + w2, of a 3-wire r1cs; the map reads it and wire 2.
            exec(3, vec![ExecAddition { wires: [1, 2], coeffs: [F::ONE, F::ONE] }], 1, 2, vec![3, 2])
        }
        let bn = |v: u64| Bn128::from_int(v);
        let pols = file::<Bn128>().committed_pols(vec![bn(1), bn(5), bn(7)], 0, 1, 2).unwrap();
        assert_eq!(pols.trace, [bn(12), bn(7)]);
        let err = file::<Bn128>().committed_pols(vec![bn(1), bn(5), bn(7), bn(100)], 0, 1, 2).unwrap_err();
        assert!(err.to_string().contains("has 4 wires, and the r1cs the exec was written for has 3"), "{err}");

        let gl = |v: u64| Goldilocks::from_int(v);
        let pols = file::<Goldilocks>().committed_pols(vec![gl(1), gl(5), gl(7), gl(100)], 0, 1, 2).unwrap();
        assert_eq!(pols.trace, [gl(100), gl(7)], "version 2 trusts the witness's length");
    }
}
