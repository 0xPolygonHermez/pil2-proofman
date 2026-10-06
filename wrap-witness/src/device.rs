//! The blake3 wrap's stage-1 witness as its parts, for the GPU to build
//! ([`proofman_pilfflonk::prove_exec`], pil2-stark's pilfflonk_wrap_exec.hpp): the circom witness,
//! the exec's additions and map, and the blocks' inputs, which the host reads out of the witness
//! through the map and checks (every output the block computes, every Goldilocks input canonical),
//! with the range checks' counts. The device computes the additions and gathers the map as the PLONK
//! GPU prover does, and expands the blocks.

use proofman_common::exec_format::blake3_wrap_cols::A;
use proofman_common::exec_format::{
    ExecFile, BLAKE3_IV, BLAKE3_WRAP_BLOCK_ROWS, BLAKE3_WRAP_NODE_BAND_KIND, BLAKE3_WRAP_PARENT_BAND_KIND,
    BLAKE3_WRAP_RANGE_CHECK_CELLS, BLAKE3_WRAP_RANGE_CHECK_SLOTS, RANGE_CHECK_CHUNK_BITS,
};
use proofman_fields::{Bn128, Field, Goldilocks, PrimeField64, QuotientMap};
use proofman_pilfflonk::{ExecStatic, ExecWitness, FrBytes, WrapBlock};
use rayon::prelude::*;

use crate::blake3::{compress, feedforward, node_word, Blake3Bands};
use crate::error::{WrapWitnessError, WrapWitnessResult};
use crate::trace::{addition_levels, VALUE_BYTES};

/// What does not change from a proof to the next: the additions as arrays, and the map column by
/// column, its empty cells on the zero past the additions.
#[derive(Debug)]
pub(crate) struct ExecParts {
    n_vars: usize,
    add_wire1: Vec<u32>,
    add_wire2: Vec<u32>,
    add_coef1: Vec<u8>,
    add_coef2: Vec<u8>,
    add_level: Vec<u8>,
    n_levels: u64,
    map: Vec<u32>,
    map_cols: u64,
}

/// What a proof adds to [`ExecParts`]: the circom witness, the blocks and the range checks' counts.
#[derive(Debug)]
pub(crate) struct ProofParts {
    wires: Vec<u8>,
    blocks: Vec<WrapBlock>,
    range_counts: Vec<u32>,
}

impl ExecParts {
    pub(crate) fn of(exec: &ExecFile<Bn128>) -> WrapWitnessResult<Self> {
        let mismatch = |e: String| WrapWitnessError::Mismatch(format!("the exec {e}"));
        let n_vars = exec.layout.n_vars().ok_or_else(|| mismatch("records no wire count".into()))?;
        let (level, n_levels) = addition_levels(exec, n_vars)?;
        // The device's additions kernel counts levels in a u8.
        if n_levels > 255 {
            return Err(mismatch(format!("has additions of {n_levels} levels, more than the device's 255")));
        }
        let n_adds = exec.additions.len();
        let (rows, cols) = (exec.layout.map_rows(), exec.layout.map_cols());
        let empty = u32::try_from(n_vars + n_adds).map_err(|_| mismatch("has more wires than 2^32".into()))?;
        // The device reads what the map names unchecked.
        if let Some(&w) = exec.map.iter().find(|&&w| w >= empty) {
            return Err(mismatch(format!("maps wire {w}, past its {n_vars} wires and {n_adds} additions")));
        }
        let map = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| (r, c)))
            .map(|(r, c)| match exec.map[r * cols + c] {
                0 => empty,
                w => w,
            })
            .collect();
        Ok(Self {
            n_vars,
            add_wire1: exec.additions.iter().map(|a| a.wires[0]).collect(),
            add_wire2: exec.additions.iter().map(|a| a.wires[1]).collect(),
            add_coef1: exec.additions.iter().flat_map(|a| a.coeffs[0].to_le_bytes()).collect(),
            add_coef2: exec.additions.iter().flat_map(|a| a.coeffs[1].to_le_bytes()).collect(),
            add_level: level.iter().map(|&l| l as u8).collect(),
            n_levels: u64::from(n_levels),
            map,
            map_cols: cols as u64,
        })
    }

    /// The parts every proof shares, which the device takes once, with the key.
    pub(crate) fn exec_static(&self) -> ExecStatic<'_> {
        ExecStatic {
            n_wires: self.n_vars as u64,
            add_wire1: &self.add_wire1,
            add_wire2: &self.add_wire2,
            add_coef1: &self.add_coef1,
            add_coef2: &self.add_coef2,
            add_level: &self.add_level,
            n_levels: self.n_levels,
            map: &self.map,
            map_cols: self.map_cols,
        }
    }

    /// The parts of a proof, `proof`, as the device takes them.
    pub(crate) fn exec_witness<'a>(proof: &'a ProofParts) -> ExecWitness<'a> {
        ExecWitness { wires: &proof.wires, blocks: &proof.blocks, range_counts: &proof.range_counts }
    }

    /// The parts of the witness of `wires`, the circom witness's bytes (each below `r`), whose
    /// bands are `bands`; and its `n_publics` publics. Refuses what [`blocks_of`] refuses.
    pub(crate) fn witness(
        &self,
        exec: &ExecFile<Bn128>,
        bands: &Blake3Bands,
        wires: Vec<u8>,
        n_publics: usize,
    ) -> WrapWitnessResult<(Vec<FrBytes>, ProofParts)> {
        if n_publics >= wires.len() / VALUE_BYTES {
            return mismatch(format!("the circom witness has too few wires for {n_publics} publics"));
        }
        let publics = (1..=n_publics)
            .map(|w| FrBytes::from_le_bytes(wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES].try_into().expect("32")))
            .collect::<Result<_, _>>()
            .map_err(|_| WrapWitnessError::Mismatch("a public is not below r".into()))?;
        let (blocks, range_counts) = blocks_of(exec, bands, &wires)?;
        Ok((publics, ProofParts { wires, blocks, range_counts }))
    }
}

/// The blocks of `bands` as their inputs, read out of `wires`, the circom witness's bytes, through
/// `exec`'s map, and the counts of the 16-bit table its range-check rows' chunk cells look up.
/// Refuses a witness of another compile of the circuit, a block input cell that is not a word or not
/// canonical, an output the block does not compute, and a range-check chunk not below 2^16.
pub(crate) fn blocks_of(
    exec: &ExecFile<Bn128>,
    bands: &Blake3Bands,
    wires: &[u8],
) -> WrapWitnessResult<(Vec<WrapBlock>, Vec<u32>)> {
    let n_witness = wires.len() / VALUE_BYTES;
    if exec.layout.n_vars().is_some_and(|n| n != n_witness) || exec.layout.map_cols() <= A + 12 {
        return mismatch(format!(
            "the circom witness has {n_witness} wires, and the r1cs the exec was written for has {:?}",
            exec.layout.n_vars()
        ));
    }
    // The cell of the trace at `row`, `col`: the map's wire, a circom one or none.
    let cell = |row: usize, col: usize| -> WrapWitnessResult<u64> {
        // 0 is no wire: the empty cell is 0, not wire 0, the constant one.
        let w = exec.map_entry(row, col) as usize;
        if w == 0 {
            return Ok(0);
        }
        if w >= n_witness {
            return mismatch(format!("the cell at row {row}, column {col} is addition {}, not a wire", w - n_witness));
        }
        let bytes = &wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES];
        if bytes[8..].iter().any(|&b| b != 0) {
            return mismatch(format!("the cell at row {row}, column {col} is not a 64-bit word"));
        }
        Ok(u64::from_le_bytes(bytes[..8].try_into().expect("8")))
    };
    let u32_cell = |row: usize, col: usize| -> WrapWitnessResult<u32> {
        let v = cell(row, col)?;
        u32::try_from(v).or_else(|_| mismatch(format!("the cell at row {row}, column {col} is {v}, not a u32")))
    };

    let blocks = bands
        .blocks
        .par_iter()
        .enumerate()
        .map(|(b, &(kind, flags))| {
            let base = b * BLAKE3_WRAP_BLOCK_ROWS;
            let (node, parent) = (kind == BLAKE3_WRAP_NODE_BAND_KIND, kind == BLAKE3_WRAP_PARENT_BAND_KIND);
            let kind = if node {
                0
            } else if parent {
                2
            } else {
                1
            };
            let mut blk = WrapBlock { kind, flags: flags as u32, cv: BLAKE3_IV, ..Default::default() };
            if node {
                (blk.block_len, blk.counter_lo) = (64, 0);
            } else {
                (blk.block_len, blk.counter_lo) = (u32_cell(base, A + 11)?, u32_cell(base, A + 12)?);
            }
            if parent {
                for t in 0..8 {
                    blk.m[2 * t] = u32_cell(base + t, A)?;
                    blk.m[2 * t + 1] = u32_cell(base + t, A + 1)?;
                }
            } else {
                let key = if node { cell(base, A + 2)? } else { 0 };
                if !node {
                    for j in 0..8 {
                        blk.cv[j] = u32_cell(base, A + 3 + j)?;
                    }
                }
                for t in 0..8 {
                    let w = cell(base + t, A + usize::from(key == 1))?;
                    if w >= Goldilocks::ORDER_U64 {
                        return mismatch(format!(
                            "the Goldilocks word {w} of the blake3 block at row {base} is not canonical"
                        ));
                    }
                    blk.m[2 * t] = w as u32;
                    blk.m[2 * t + 1] = (w >> 32) as u32;
                    // The canonical split's 1/(hi - (2^32 - 1)), at lo = 0 a don't-care.
                    if blk.m[2 * t] != 0 {
                        let d = Bn128::from_int(u64::from(blk.m[2 * t + 1])) - Bn128::from_int(0xFFFF_FFFFu64);
                        blk.dinv[t] = d.try_inverse().expect("canonical: hi < 2^32 - 1 or lo = 0").to_le_bytes();
                    }
                }
            }
            let v = compress(&blk.cv, &blk.m, blk.block_len, blk.counter_lo, blk.flags);
            for k in 0..if node { 4 } else { 8 } {
                let (c0, c1) = feedforward(&v, &blk.cv, k);
                let row = base + 56 + k;
                let expected = if node {
                    let (packed, over) = node_word(c0, c1);
                    blk.over[k] = u32::from(over);
                    // The digest's canonicity: 1/C0 when over, else 1/(C1 - (2^32 - 1)), free at C0 = 0.
                    let d = if over {
                        Some(Bn128::from_int(u64::from(c0)))
                    } else if c0 != 0 {
                        Some(Bn128::from_int(u64::from(c1)) - Bn128::from_int(0xFFFF_FFFFu64))
                    } else {
                        None
                    };
                    if let Some(d) = d {
                        blk.dinv_ff[k] = d.try_inverse().expect("canonical: C0 != 0, or C1 < 2^32 - 1").to_le_bytes();
                    }
                    vec![(0, packed)]
                } else {
                    vec![(0, u64::from(c0)), (1, u64::from(c1))]
                };
                for (col, want) in expected {
                    let got = cell(row, A + col)?;
                    if got != want {
                        return mismatch(format!(
                            "the output of the blake3 block at row {base} (row {row}, column {col}) is {got}, and \
                             the block computes {want}: the circuit's witness does not satisfy its blake3 gate"
                        ));
                    }
                }
            }
            Ok(blk)
        })
        .collect::<WrapWitnessResult<Vec<_>>>()?;

    // The range-check rows' chunk cells, every one, the zeros past a use's chunks too.
    let mut range_counts = vec![0u32; 1 << RANGE_CHECK_CHUNK_BITS];
    for &row in &bands.range_rows {
        for slot in 0..BLAKE3_WRAP_RANGE_CHECK_SLOTS {
            for k in 1..BLAKE3_WRAP_RANGE_CHECK_CELLS {
                let col = BLAKE3_WRAP_RANGE_CHECK_CELLS * slot + k;
                match cell(row, col)? {
                    v if v < 1 << RANGE_CHECK_CHUNK_BITS => range_counts[v as usize] += 1,
                    v => {
                        return mismatch(format!(
                            "the chunk of the range check at row {row}, column {col}, is {v}, not below \
                             2^{RANGE_CHECK_CHUNK_BITS}: the circuit's witness does not satisfy its Num2Bytes"
                        ))
                    }
                }
            }
        }
    }
    Ok((blocks, range_counts))
}

fn mismatch<T>(e: String) -> WrapWitnessResult<T> {
    Err(WrapWitnessError::Mismatch(e))
}
