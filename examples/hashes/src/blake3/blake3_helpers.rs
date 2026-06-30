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

/// Row index of a range-checker tuple (a, b)
#[inline]
pub(crate) fn range_row(a: u8, b: u8) -> usize {
    (b as usize) * 256 + a as usize
}

/// Row index of an XOR-rotate table tuple (offset, a, b, rot), rot in {0, 12, 7}
#[inline]
pub(crate) fn table_row(offset: usize, a: u8, b: u8, rot: u32) -> usize {
    let rot_block = match rot {
        0 => 0,
        12 => 1,
        7 => 2,
        _ => panic!("rotation {rot} is not in the table (expected 0, 12 or 7)"),
    };
    rot_block * (1 << 18) + offset * (1 << 16) + (b as usize) * 256 + a as usize
}

/// Split the XOR-rotate operation into two bytes, given the offset of the byte in the 32-bit word
pub(crate) fn xor_rotr_split(offset: usize, a: u8, b: u8, rot: u32) -> (u8, u8) {
    // Position the byte correctly and compute the rotation
    let byte = (a ^ b) as u32;
    let byte_pos = byte << (8 * offset as u32);
    let c = byte_pos.rotate_right(rot);

    let s = (8 * offset as i32 - rot as i32).rem_euclid(32) as u32; // normalized bit shift
    let l = (s / 8) % 4;
    let lp1 = (l + 1) % 4;

    let c0 = ((c >> (8 * l)) & 0xff) as u8;
    let c1 = ((c >> (8 * lp1)) & 0xff) as u8;
    (c0, c1)
}
