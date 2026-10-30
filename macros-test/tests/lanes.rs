// A row with lanes must lay its columns out as a row declaring them all in order.
#![allow(clippy::int_plus_one)]
use proofman_common::trace::TraceRow;
use proofman_fields::{Goldilocks, PrimeField64};
use proofman_macros::trace_row;

// `hi` is 121 bits a lane, padded to 128; `lo`, 7 bits, is too narrow to pad and stays unaligned.
trace_row!(LRow<F> {
    solo: ubit(5),
    bin: lane { flag: bit, a: ubit(40) },
    hi: [lane { flag: bit, a: ubit(40), v: [u16; 2], w: [[u8; 3]; 2] }; 3],
    lo: [lane { b: ubit(7) }; 2],
    tail: u8,
});

// The same columns, in order.
trace_row!(FlatRow<F> {
    solo: ubit(5),
    bin_flag: bit, bin_a: ubit(40),
    flag0: bit, a0: ubit(40), v0: [u16; 2], w0: [[u8; 3]; 2],
    flag1: bit, a1: ubit(40), v1: [u16; 2], w1: [[u8; 3]; 2],
    flag2: bit, a2: ubit(40), v2: [u16; 2], w2: [[u8; 3]; 2],
    b0: ubit(7), b1: ubit(7),
    tail: u8,
});

type F = Goldilocks;

const _: () = assert!(<LRow<F> as TraceRow>::ROW_SIZE == <FlatRow<F> as TraceRow>::ROW_SIZE);
const _: () = assert!(LRowPacked::<F>::UNPACK_INFO.len() == FlatRowPacked::<F>::UNPACK_INFO.len());
const _: () = assert!(LRowHiView::<'static, LRowPacked<F>>::LANES == 3);

fn value(l: usize, k: u64) -> u64 {
    (l as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k + 1))
}

fn fill_hi<L: LRowHiOps<F>>(lane: &mut L, l: usize) {
    lane.set_flag(l % 2 == 1);
    lane.set_a(value(l, 0) & ((1 << 40) - 1));
    for i in 0..2 {
        lane.set_v(i, value(l, 1 + i as u64) as u16);
        for j in 0..3 {
            lane.set_w(i, j, value(l, 3 + (i * 3 + j) as u64) as u8);
        }
    }
}

fn fill<R: LRowOps<F>>() -> R {
    let mut row = R::default();
    fill_into(&mut row);
    row
}

fn fill_into<R: LRowOps<F>>(row: &mut R) {
    row.set_solo(17);
    let mut bin = row.bin();
    bin.set_flag(true);
    bin.set_a(value(7, 0) & ((1 << 40) - 1));
    for l in 0..3 {
        fill_hi(&mut row.hi(l), l);
    }
    for l in 0..2 {
        row.lo(l).set_b(100 + l as u8);
    }
    row.set_tail(0xAB);
}

macro_rules! fill_flat_hi {
    ($row:expr, $l:expr, $flag:ident, $a:ident, $v:ident, $w:ident) => {{
        let l = $l;
        $row.$flag(l % 2 == 1);
        $row.$a(value(l, 0) & ((1 << 40) - 1));
        for i in 0..2 {
            $row.$v(i, value(l, 1 + i as u64) as u16);
            for j in 0..3 {
                $row.$w(i, j, value(l, 3 + (i * 3 + j) as u64) as u8);
            }
        }
    }};
}

fn fill_flat<R: FlatRowOps<F>>() -> R {
    let mut row = R::default();
    row.set_solo(17);
    row.set_bin_flag(true);
    row.set_bin_a(value(7, 0) & ((1 << 40) - 1));
    fill_flat_hi!(row, 0, set_flag0, set_a0, set_v0, set_w0);
    fill_flat_hi!(row, 1, set_flag1, set_a1, set_v1, set_w1);
    fill_flat_hi!(row, 2, set_flag2, set_a2, set_v2, set_w2);
    row.set_b0(100);
    row.set_b1(101);
    row.set_tail(0xAB);
    row
}

fn words<R: TraceRow, T: Copy>(row: &R) -> Vec<T> {
    // repr(C) rows of `ROW_SIZE` 64-bit words (an F, or a packed u64).
    unsafe { std::slice::from_raw_parts(row as *const R as *const T, R::ROW_SIZE).to_vec() }
}

/// The prover's unpack: each column's `UNPACK_INFO` bits in turn, LSB first.
fn unpack(words: &[u64], widths: &[u64]) -> Vec<u64> {
    let mut pos = 0u64;
    widths
        .iter()
        .map(|&bits| {
            let (w, off) = ((pos / 64) as usize, pos % 64);
            let low = words[w] >> off;
            let val = if off + bits <= 64 { low } else { low | (words[w + 1] << (64 - off)) };
            pos += bits;
            if bits == 64 {
                val
            } else {
                val & ((1 << bits) - 1)
            }
        })
        .collect()
}

#[test]
fn packed_lanes_unpack_to_the_columns_in_order() {
    let laned: LRowPacked<F> = fill();
    let flat: FlatRowPacked<F> = fill_flat();
    let columns = unpack(&words(&flat), &FlatRowPacked::<F>::UNPACK_INFO);
    assert_eq!(unpack(&words(&laned), &LRowPacked::<F>::UNPACK_INFO), columns);
    assert_eq!(LRowPacked::<F>::UNPACK_INFO.iter().sum::<u64>() as usize, LRowPacked::<F>::PACKED_BITS);
}

// A reused buffer: the setters must zero the padding the prover reads with each column.
#[test]
fn setters_zero_the_padding_of_a_dirty_row() {
    let mut row = LRowPacked::<F> { packed: [u64::MAX; LRowPacked::<F>::PACKED_WORDS], ..Default::default() };
    fill_into(&mut row);
    let flat: FlatRowPacked<F> = fill_flat();
    let columns = unpack(&words(&flat), &FlatRowPacked::<F>::UNPACK_INFO);
    assert_eq!(unpack(&words(&row), &LRowPacked::<F>::UNPACK_INFO), columns);
}

#[test]
fn each_lane_owns_its_words() {
    for (l, (first, per_lane)) in [LRowPacked::<F>::HI_LANE_WORDS; 3].into_iter().enumerate() {
        let mut row = LRowPacked::<F>::default();
        let mut lane = row.hi(l);
        lane.set_flag(true);
        lane.set_a((1 << 40) - 1);
        for i in 0..2 {
            lane.set_v(i, u16::MAX);
            for j in 0..3 {
                lane.set_w(i, j, u8::MAX);
            }
        }
        let own = first + l * per_lane..first + (l + 1) * per_lane;
        for (w, word) in words::<_, u64>(&row).into_iter().enumerate() {
            assert_eq!(word != 0, own.contains(&w), "hi[{l}] wrote word {w}, its words are {own:?}");
        }
    }
}

#[test]
fn unpacked_lanes_are_laid_out_in_order() {
    let laned: LRow<F> = fill();
    let flat: FlatRow<F> = fill_flat();
    assert_eq!(words::<_, F>(&laned), words::<_, F>(&flat));
}

#[test]
fn a_lane_reads_back_what_the_row_wrote() {
    let mut row: LRowPacked<F> = fill();
    assert!(row.bin().get_flag());
    for l in 0..3 {
        let lane = row.hi(l);
        assert_eq!(lane.get_flag(), l % 2 == 1);
        assert_eq!(lane.get_a(), value(l, 0) & ((1 << 40) - 1));
        assert_eq!(lane.get_v(1), value(l, 2) as u16);
        assert_eq!(lane.get_w(1, 2), value(l, 8) as u8);
        assert_eq!(row.get_hi_w(l, 1, 2), value(l, 8) as u8);
    }
    assert_eq!(row.get_all_lo_b(), [100, 101]);
}

#[test]
fn whole_arrays_span_every_lane() {
    let mut row = LRowPacked::<F>::default();
    let w = [[1, 2, 3], [4, 5, 6]];
    row.set_all_hi_v(&[[7, 8], [9, 10], [11, 12]]);
    assert_eq!(row.hi(1).get_v(1), 10);
    row.set_all_hi_flag(&[true, false, true]);
    assert_eq!(row.get_all_hi_flag(), [true, false, true]);

    // The unpacked lane is a row of its own.
    let mut unpacked = LRow::<F>::default();
    unpacked.hi[1].set_all_w(&w);
    assert_eq!(unpacked.get_hi_w(1, 1, 0), 4);
    assert_eq!(unpacked.hi[1].w[1][0], F::from_u64(4));
    unpacked.bin.set_a(9);
    assert_eq!(unpacked.get_bin_a(), 9);
}

// Without `bits(n)` every column is an `F`: unpacked only.
trace_row!(GRow<F> {
    x: F,
    ys: [lane { y: [F; 2] }; 2],
});

// 34 bits of padding before a 2 x 128-bit group would cost a word: left unaligned (286 bits).
trace_row!(CostRow<F> {
    a: ubit(30),
    g: [lane { x: u64, y: u64 }; 2],
});
const _: () = assert!(CostRowPacked::<F>::PACKED_WORDS == 5);

// A single lane's scalar `F` column, reached through the row trait.
trace_row!(GsRow<F> {
    x: F,
    bin: lane { a: F, b: [F; 2] },
});

#[test]
fn a_single_lanes_f_column_goes_through_the_trait() {
    let mut row = GsRow::<F>::default();
    GsRowOps::set_bin_a(&mut row, F::from_u64(9));
    assert_eq!(GsRowOps::get_bin_a(&row), F::from_u64(9));
    assert_eq!(row.bin.a, F::from_u64(9));
}

#[test]
fn a_row_of_f_columns_has_its_lanes_side_by_side() {
    assert_eq!(<GRow<F> as TraceRow>::ROW_SIZE, 5);
    let mut row = GRow::<F>::default();
    row.ys[1].y[1] = F::from_u64(7);
    assert_eq!(words::<_, F>(&row)[4], F::from_u64(7));
}

#[test]
fn steps_fill_a_row_lane_by_lane() {
    let mut rows = [LRowPacked::<F>::default(); 2];
    for step in 0..6 {
        LRowHiView::step(&mut rows, step).set_a(step as u64);
    }
    assert_eq!(rows[0].get_all_hi_a(), [0, 1, 2]);
    assert_eq!(rows[1].get_all_hi_a(), [3, 4, 5]);
}
