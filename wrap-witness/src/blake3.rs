//! The blocks of the blake3 BN128 wrap's AIR (plonk2pil's `pil/blake3_bn128/wrap.pil`): every column
//! of a block but the band's, which the map does not gather, expanded from the block's inputs
//! ([`blocks_of`](crate::device::blocks_of) reads and checks them), and the two multiplicities of
//! blake3's tables, its range-check rows' chunk cells counted into the 16-bit one.

use proofman_common::exec_format::blake3_wrap_cols::*;
use proofman_common::exec_format::{
    ExecFile, BLAKE3_IV, BLAKE3_SIGMA, BLAKE3_WRAP_BLOCK_ROWS, BLAKE3_WRAP_NODE_BAND_KIND,
    BLAKE3_WRAP_PARENT_BAND_KIND, BLAKE3_WRAP_RANGE_CHECK_BAND_KIND, RANGE_CHECK_CHUNK_BITS,
};
use proofman_fields::{Bn128, Goldilocks, PrimeField64};
use proofman_pilfflonk::WrapBlock;
use rayon::prelude::*;

use crate::error::{WrapWitnessError, WrapWitnessResult};
use crate::trace::{put_word, VALUE_BYTES};

const TABLE_ROWS: usize = 1 << 17;
const RANGE_ROWS: usize = 1 << RANGE_CHECK_CHUNK_BITS;

/// (va, vb, vc, vd) of G g.
const IDX: [[usize; 4]; 8] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// The blake3 bands of an exec: its blocks, end to end from row 0, and its range-check rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Blake3Bands {
    /// `(kind, flags)` of block `i`.
    pub(crate) blocks: Vec<(u64, u64)>,
    pub(crate) range_rows: Vec<usize>,
}

impl Blake3Bands {
    /// The blake3 bands of `exec`, `None` if it has no block, or why they are not the wrap's: a
    /// band of another kind, or blocks that are not end to end from row 0. The error completes
    /// "the exec ...".
    pub(crate) fn of(exec: &ExecFile<Bn128>) -> Result<Option<Self>, String> {
        let is_block = |k: u64| (BLAKE3_WRAP_NODE_BAND_KIND..=BLAKE3_WRAP_PARENT_BAND_KIND).contains(&k);
        if !exec.bands.iter().any(|b| is_block(b.kind)) {
            return Ok(None);
        }
        let mut blocks = Vec::new();
        let mut range_rows = Vec::new();
        for b in &exec.bands {
            match b.kind {
                k if is_block(k) => {
                    if b.row != (BLAKE3_WRAP_BLOCK_ROWS * blocks.len()) as u64 {
                        return Err(format!("has a blake3 block at row {}, not end to end from row 0", b.row));
                    }
                    blocks.push((k, b.payload));
                }
                BLAKE3_WRAP_RANGE_CHECK_BAND_KIND => range_rows.push(b.row as usize),
                kind => return Err(format!("has a gate band of kind {kind} at row {} beside blake3 blocks", b.row)),
            }
        }
        Ok(Some(Self { blocks, range_rows }))
    }

    /// Fills `trace`, `n_rows x n_cols` row after row, whose band the map has gathered: every block's
    /// other columns, expanded from `blocks`, and the tables' multiplicities, the range-check rows'
    /// `range_counts` included. Refuses an AIR that is not the wrap's.
    pub(crate) fn fill(
        &self,
        trace: &mut [u8],
        n_rows: usize,
        n_cols: usize,
        blocks: &[WrapBlock],
        range_counts: &[u32],
    ) -> WrapWitnessResult<()> {
        let mismatch = |e: String| Err(WrapWitnessError::Mismatch(e));
        if n_cols != N_COLS {
            return mismatch(format!("the AIR has {n_cols} stage-1 columns, and the blake3 wrap's has {N_COLS}"));
        }
        if n_rows < TABLE_ROWS || BLAKE3_WRAP_BLOCK_ROWS * blocks.len() > n_rows {
            return mismatch(format!("the AIR has {n_rows} rows, too few for {} blake3 blocks", blocks.len()));
        }
        if let Some(&row) = self.range_rows.iter().find(|&&row| row >= n_rows) {
            return mismatch(format!("the exec has a range check at row {row}, and the AIR has {n_rows} rows"));
        }
        let block_cells = BLAKE3_WRAP_BLOCK_ROWS * N_COLS * VALUE_BYTES;
        let (table, range) = trace[..blocks.len() * block_cells]
            .par_chunks_mut(block_cells)
            .zip(blocks)
            .with_min_len(64)
            .fold(
                || (vec![0u64; TABLE_ROWS], vec![0u64; RANGE_ROWS]),
                |(mut table, mut range), (cells, blk)| {
                    expand(blk, cells, &mut table, &mut range);
                    (table, range)
                },
            )
            .reduce(
                || (vec![0u64; TABLE_ROWS], vec![0u64; RANGE_ROWS]),
                |(mut ta, mut ra), (tb, rb)| {
                    ta.iter_mut().zip(tb).for_each(|(a, b)| *a += b);
                    ra.iter_mut().zip(rb).for_each(|(a, b)| *a += b);
                    (ta, ra)
                },
            );
        for (row, count) in table.into_iter().enumerate() {
            put_word(trace, row * n_cols + MUL_TABLE, count);
        }
        for (row, (count, &host)) in range.into_iter().zip(range_counts).enumerate() {
            put_word(trace, row * n_cols + MUL_RANGE, count + u64::from(host));
        }
        Ok(())
    }
}

/// The columns of block `blk` into `cells`, its 64 rows, and its lookups counted into `table` and
/// `range`, as pil2-stark's `pilfflonk_wrap_exec.cu` does on the device.
fn expand(blk: &WrapBlock, cells: &mut [u8], table: &mut [u64], range: &mut [u64]) {
    if blk.kind != 2 {
        for (t, dinv) in blk.dinv.iter().enumerate() {
            let at = (t * N_COLS + D_INV) * VALUE_BYTES;
            cells[at..at + VALUE_BYTES].copy_from_slice(dinv);
        }
    }
    if blk.kind == 0 {
        for (k, dinv) in blk.dinv_ff.iter().enumerate() {
            let at = ((56 + k) * N_COLS + D_INV) * VALUE_BYTES;
            cells[at..at + VALUE_BYTES].copy_from_slice(dinv);
        }
    }
    let mut put = |r: usize, c: usize, v: u64| put_word(cells, r * N_COLS + c, v);
    let mut xor = |a: u32, b: u32, rot: u8| {
        for i in 0..4 {
            table[table_row((a >> (8 * i)) as u8, (b >> (8 * i)) as u8, rot)] += 1;
        }
    };
    let mut v = initial_state(&blk.cv, blk.block_len, blk.counter_lo, blk.flags);
    for t in 0..56 {
        let (r, g) = (t / 8, t % 8);
        for (j, &w) in v.iter().enumerate() {
            put(t, ST + j, u64::from(w));
        }
        let [ia, ib, ic, id] = IDX[g];
        let (va, vb, vc, vd) = (v[ia], v[ib], v[ic], v[id]);
        let (x, y) = (blk.m[BLAKE3_SIGMA[r][2 * g]], blk.m[BLAKE3_SIGMA[r][2 * g + 1]]);
        let [a1, d1, c1, b1, a2, d2, c2, z] = g_step(va, vb, vc, vd, x, y);
        for (col, w) in [(VA, va), (X, x), (Y, y)] {
            for h in 0..2 {
                let limb = u64::from(w >> (16 * h) & 0xffff);
                put(t, col + h, limb);
                range[limb as usize] += 1;
            }
        }
        put(t, VC, u64::from(vc));
        for (col, w) in [
            (VB, vb),
            (VD, vd),
            (VA_P, a1),
            (VD_P, d1),
            (VC_P, c1),
            (VA_PP, a2),
            (VD_PP, d2),
            (VC_PP, c2),
            (VB_PP_XOR, z),
        ] {
            for i in 0..4 {
                put(t, col + i, u64::from(w >> (8 * i) & 0xff));
            }
        }
        for i in 0..4 {
            let (s0, s1) = table_out((vb >> (8 * i)) as u8, (c1 >> (8 * i)) as u8, 12);
            put(t, VB_P_S + 2 * i, u64::from(s0));
            put(t, VB_P_S + 2 * i + 1, u64::from(s1));
        }
        put(t, VB_PP_T, u64::from(z >> 7 & 1));
        xor(vd, a1, 0);
        xor(vb, c1, 12);
        xor(d1, a2, 0);
        xor(b1, c2, 0);
        (v[ia], v[ib], v[ic], v[id]) = (a2, z.rotate_right(7), c2, d2);
    }
    for t in 56..BLAKE3_WRAP_BLOCK_ROWS {
        for (j, &w) in v.iter().enumerate() {
            put(t, ST + j, u64::from(w));
        }
    }
    // The feedforward: row 56 + k, out[2k] and out[2k + 1].
    for k in 0..if blk.kind == 0 { 4 } else { 8 } {
        let t = 56 + k;
        let (a0, a1) = (v[2 * k], v[2 * k + 1]);
        let (b0, b1) = if k < 4 { (v[2 * k + 8], v[2 * k + 9]) } else { (blk.cv[2 * k - 8], blk.cv[2 * k - 7]) };
        for (col, w) in [(VB, a0), (VD, b0), (VA_P, a0 ^ b0), (VD_P, a1), (VC_P, b1), (VA_PP, a1 ^ b1)] {
            for i in 0..4 {
                put(t, col + i, u64::from(w >> (8 * i) & 0xff));
            }
        }
        xor(a0, b0, 0);
        xor(a1, b1, 0);
        if blk.kind == 0 {
            put(t, A + 1, u64::from(blk.over[k]));
        }
    }
}

fn initial_state(cv: &[u32; 8], block_len: u32, counter_lo: u32, flags: u32) -> [u32; 16] {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(cv);
    v[8..12].copy_from_slice(&BLAKE3_IV[..4]);
    (v[12], v[14], v[15]) = (counter_lo, block_len, flags);
    v
}

/// G's intermediate words (a', d', c', b', a'', d'', c'', b'' before its rotation by 7).
fn g_step(va: u32, vb: u32, vc: u32, vd: u32, x: u32, y: u32) -> [u32; 8] {
    let a1 = va.wrapping_add(vb).wrapping_add(x);
    let d1 = (vd ^ a1).rotate_right(16);
    let c1 = vc.wrapping_add(d1);
    let b1 = (vb ^ c1).rotate_right(12);
    let a2 = a1.wrapping_add(b1).wrapping_add(y);
    let d2 = (d1 ^ a2).rotate_right(8);
    let c2 = c1.wrapping_add(d2);
    [a1, d1, c1, b1, a2, d2, c2, b1 ^ c2]
}

/// BLAKE3's compression of the message `m` under the chaining value `cv`: the final state, before
/// the feedforward.
pub(crate) fn compress(cv: &[u32; 8], m: &[u32; 16], block_len: u32, counter_lo: u32, flags: u32) -> [u32; 16] {
    let mut v = initial_state(cv, block_len, counter_lo, flags);
    for r in 0..7 {
        for (g, &[ia, ib, ic, id]) in IDX.iter().enumerate() {
            let (x, y) = (m[BLAKE3_SIGMA[r][2 * g]], m[BLAKE3_SIGMA[r][2 * g + 1]]);
            let [_, _, _, _, a2, d2, c2, z] = g_step(v[ia], v[ib], v[ic], v[id], x, y);
            (v[ia], v[ib], v[ic], v[id]) = (a2, z.rotate_right(7), c2, d2);
        }
    }
    v
}

/// Output words `2k` and `2k + 1` of the final state `v` under the chaining value `cv`.
pub(crate) fn feedforward(v: &[u32; 16], cv: &[u32; 8], k: usize) -> (u32, u32) {
    let partner = |h: usize| if k < 4 { v[2 * k + h + 8] } else { cv[2 * k - 8 + h] };
    (v[2 * k] ^ partner(0), v[2 * k + 1] ^ partner(1))
}

/// A Node's output word of the u32 pair (lo, hi): canonical, and whether it was reduced.
pub(crate) fn node_word(lo: u32, hi: u32) -> (u64, bool) {
    let packed = u64::from(lo) | u64::from(hi) << 32;
    let over = packed >= Goldilocks::ORDER_U64;
    (if over { packed - Goldilocks::ORDER_U64 } else { packed }, over)
}

/// Row of the XOR/ROTR table for (a, b, rot), as blake3Tables lays it out: A fastest, then B, then
/// ROTATION.
fn table_row(a: u8, b: u8, rot: u8) -> usize {
    (if rot == 12 { 1 << 16 } else { 0 }) | (b as usize) << 8 | a as usize
}

/// The table's two outputs for (a, b, rot), as blake3Tables computes them.
fn table_out(a: u8, b: u8, rot: u32) -> (u8, u8) {
    let c = u32::from(a ^ b).rotate_right(rot);
    let shift = (32 - rot) % 32 / 8 % 4;
    ((c >> (8 * shift)) as u8, (c >> (8 * ((shift + 1) % 4))) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's outputs follow blake3Tables' formula, rot 12 included.
    #[test]
    fn table_out_is_the_tables() {
        for (a, b) in [(0x11u8, 0x00u8), (0xff, 0x0f), (0x80, 0x01)] {
            assert_eq!(table_out(a, b, 0), (a ^ b, 0));
            let c = u32::from(a ^ b).rotate_right(12);
            assert_eq!(table_out(a, b, 12), ((c >> 16) as u8, (c >> 24) as u8));
        }
    }
}
