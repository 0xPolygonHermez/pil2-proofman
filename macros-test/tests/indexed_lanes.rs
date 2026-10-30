// An indexed row with lanes, unpacked as unpack_indexed_row.hpp does, must give the full row.
#![allow(clippy::int_plus_one)]
use proofman_common::trace::{IndexedFill, TraceRow};
use proofman_fields::{Goldilocks, PrimeField64};
use proofman_macros::{indexed_trace_row, trace_row};

// A column of the air itself, then the steps.
trace_row!(MxRow<F> {
    seg: u8,
    steps: [lane { a: [u16; 2], pc: u32, imm: [ubit(20); 2], flag: bit, st: ubit(38), r: ubit(53) }; 3],
});

indexed_trace_row!(MxRow<F> {
    seg: u8,
    steps: [lane { a: [u16; 2], pc: u32 @instr, imm: [ubit(20); 2] @instr, flag: bit, st: ubit(38), r: ubit(53) }; 3],
});

type F = Goldilocks;
type Ix = MxRowPackedIndexed<F>;
const LANES: usize = 3;

const _: () = assert!(<Ix as IndexedFill>::IS_INDEXED);
// Header + seg (104 bits) pad to word 2, each lane's 124 runtime bits to 2 words. The full row's
// 196-bit lanes would pad by 60 bits, too many, so the full row stays unaligned.
const _: () = assert!(Ix::STEPS_LANE_WORDS.0 == 2 && Ix::STEPS_LANE_WORDS.1 == 2);
const _: () = assert!(Ix::LANES == LANES);

/// `indexedReadBits`: LSB first, a value may straddle two words.
fn read_bits(words: &[u64], pos: &mut u64, bits: u64) -> u64 {
    let (w, off) = ((*pos / 64) as usize, *pos % 64);
    let low = words[w] >> off;
    let val = if off + bits <= 64 { low } else { low | (words[w + 1] << (64 - off)) };
    *pos += bits;
    if bits == 64 {
        val
    } else {
        val & ((1 << bits) - 1)
    }
}

/// The full packed row, column by column.
fn unpack_full(words: &[u64]) -> Vec<u64> {
    let mut pos = 0;
    MxRowPacked::<F>::UNPACK_INFO.iter().map(|&b| read_bits(words, &mut pos, b)).collect()
}

/// `unpackIndexedRow`: runtime columns from past the header, then each lane's `@instr` columns
/// from the entry its index selects.
fn unpack_indexed(row: &[u64], table: &[Vec<u64>]) -> Vec<u64> {
    let (info, source, lane) = (Ix::UNPACK_INFO, Ix::COL_SOURCE, Ix::COL_LANE);
    let mut out = vec![0; info.len()];
    let mut rc = LANES as u64 * Ix::INDEX_BITS;
    for c in 0..info.len() {
        if source[c] == 0 {
            out[c] = read_bits(row, &mut rc, info[c]);
        }
    }
    for l in 0..LANES {
        let index = read_bits(row, &mut (l as u64 * Ix::INDEX_BITS), Ix::INDEX_BITS) as usize;
        let mut tc = 0;
        for c in 0..info.len() {
            if source[c] == 1 && lane[c] as usize == l {
                out[c] = read_bits(&table[index], &mut tc, info[c]);
            }
        }
    }
    out
}

/// What lane `l` of a row holds: its instruction is `instr`, the rest is per step.
fn fill_lane<R: MxRowOps<F>>(row: &mut R, l: usize, instr: u64) {
    let step = 7 * l as u64 + 1;
    let mut lane = row.steps(l);
    lane.set_a(0, (0xA000 + step) as u16);
    lane.set_a(1, (0xB000 + step) as u16);
    lane.set_pc((0x1000_0000 + instr) as u32);
    lane.set_imm(0, (0xF_0000 | instr) as u32);
    lane.set_imm(1, (0xE_0000 | instr) as u32);
    lane.set_flag(step.is_multiple_of(2));
    lane.set_st((1 << 37) | (step << 3));
    lane.set_r((1 << 52) | step);
}

fn words<R: TraceRow>(row: &R) -> Vec<u64> {
    unsafe { std::slice::from_raw_parts(row as *const R as *const u64, R::ROW_SIZE).to_vec() }
}

#[test]
fn an_indexed_row_unpacks_to_the_full_row() {
    // Three instructions in the table; the lanes run 2, 0, 2.
    let instrs = [0x11u64, 0x22, 0x33];
    let lane_instr = [2usize, 0, 2];

    // Reused buffers: every bit the setters leave, padding included, must not matter.
    let mut full = MxRowPacked::<F> { packed: [u64::MAX; MxRowPacked::<F>::PACKED_WORDS], ..Default::default() };
    let mut compact = Ix { packed: [u64::MAX; Ix::PACKED_WORDS], ..Default::default() };
    full.set_seg(0x5A);
    compact.set_seg(0x5A);
    for (l, &i) in lane_instr.iter().enumerate() {
        fill_lane(&mut full, l, instrs[i]);
        fill_lane(&mut compact, l, instrs[i]);
        compact.set_row_index(l, i as u32);
    }
    let table: Vec<Vec<u64>> = instrs
        .iter()
        .map(|&instr| {
            // The writer of a row fills an entry too: the lane it names is the entry's own.
            let mut entry = MxRowInstrTable::<F>::default();
            fill_lane(&mut entry, 0, instr);
            words(&entry)
        })
        .collect();

    // The descriptor proofman gets is the row's own.
    let info = Ix::packed_info();
    assert_eq!(
        (info.num_packed_words, info.unpack_info.as_slice()),
        (Ix::PACKED_WORDS as u64, Ix::UNPACK_INFO.as_slice())
    );
    assert_eq!(
        (info.col_source.as_slice(), info.col_lane.as_slice()),
        (Ix::COL_SOURCE.as_slice(), Ix::COL_LANE.as_slice())
    );
    assert_eq!((info.words_per_entry, info.lanes), (MxRowInstrTable::<F>::PACKED_WORDS as u64, LANES as u64));

    let want = unpack_full(&words(&full));
    assert_eq!(unpack_indexed(&words(&compact), &table), want);
    // And the full row is in order: lane 1's pc is column 1 + 8 + 2.
    assert_eq!(want[11], 0x1000_0000 + 0x11);
}
