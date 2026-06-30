/// Deterministic pseudo-random Blake3 input generator
pub(crate) fn random_blake3_input(seed: u64) -> ([u32; 16], [u32; 16]) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678_9ABC_DEF0);
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    };
    let state = core::array::from_fn(|_| next());
    let message = core::array::from_fn(|_| next());
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

/// Row index of an XOR-rotate table tuple (a, b, rot), rot in {0, 12, 7}
#[inline]
pub(crate) fn table_row(a: u8, b: u8, rot: u32) -> usize {
    let rot_block = match rot {
        0 => 0,
        12 => 1,
        7 => 2,
        _ => panic!("rotation {rot} is not in the table (expected 0, 12 or 7)"),
    };
    rot_block * (1 << 16) + (b as usize) * 256 + a as usize
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
