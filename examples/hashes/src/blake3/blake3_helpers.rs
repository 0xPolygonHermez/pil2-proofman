use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Pseudo-random Blake3 input generator
pub(crate) fn random_blake3_input(seed: u64) -> ([u32; 16], [u32; 16]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = core::array::from_fn(|_| rng.random());
    let message = core::array::from_fn(|_| rng.random());
    (state, message)
}

/// Split a 32-bit word into two little-endian 16-bit limbs [lo, hi]
#[inline]
pub(crate) fn limbs16(w: u32) -> [u16; 2] {
    [(w & 0xffff) as u16, (w >> 16) as u16]
}

/// Row index of a range-checker
#[inline]
pub(crate) fn range_row(v: u16) -> usize {
    v as usize
}

/// Row index of an XOR-rotate table tuple (offset, a, b, rot), rot in {0, 12, 7}
#[inline]
pub(crate) fn table_row(offset: usize, a: u8, b: u8, rot: u32) -> usize {
    debug_assert!(offset < 4);
    let rot_block = match rot {
        0 => 0,
        12 => 1,
        7 => 2,
        _ => panic!("rotation {rot} is not in the table (expected 0, 12 or 7)"),
    };
    rot_block * (1 << 18) + offset * (1 << 16) + (b as usize) * 256 + a as usize
}

/// Split the XOR-rotate output into its two limb pieces
pub(crate) fn xor_rotr_split(a: u8, b: u8, rot: u32) -> (u8, u8) {
    let byte = (a ^ b) as u32;
    let c = byte.rotate_right(rot);

    let s = (32 - rot) % 32; // normalized bit shift
    let l = (s / 8) % 4;
    let lp1 = (l + 1) % 4;

    let c0 = ((c >> (8 * l)) & 0xff) as u8;
    let c1 = ((c >> (8 * lp1)) & 0xff) as u8;
    (c0, c1)
}

/// Full 32-bit lane-positioned XOR-rotate output: rotr((a ^ b) << 8·offset, rot)
#[inline]
pub(crate) fn xor_rotr_full(a: u8, b: u8, offset: usize, rot: u32) -> u32 {
    (((a ^ b) as u32) << (8 * offset)).rotate_right(rot)
}
