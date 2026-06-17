/// State representation as 5x5x64 values. Each value fits in a `u8`
/// (max 144 during a round).
pub type KeccakState = [[[u8; 64]; 5]; 5];

/// Round constants of Keccak-f.
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets for the ρ step (in ρ/π traversal order).
const RHO: [usize; 24] = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

/// Lane permutation targets for the π step (in ρ/π traversal order).
const PI: [(usize, usize); 24] = [
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 3),
    (3, 3),
    (3, 0),
    (0, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 4),
    (4, 0),
    (0, 3),
    (3, 4),
    (4, 3),
    (3, 2),
    (2, 2),
    (2, 0),
    (0, 4),
    (4, 2),
    (2, 4),
    (4, 1),
    (1, 1),
    (1, 0),
];

const fn bits_from_u64(value: u64) -> [bool; 64] {
    let mut bits = [false; 64];
    let mut i = 0;
    while i < 64 {
        bits[i] = (value >> i) & 1 == 1;
        i += 1;
    }
    bits
}

const RC_BITS: [[bool; 64]; 24] = {
    let mut bits = [[false; 64]; 24];
    let mut i = 0;
    while i < 24 {
        bits[i] = bits_from_u64(RC[i]);
        i += 1;
    }
    bits
};

/// Flat bit position of `(x, y, z)` in the linear `[bit; 1600]` state.
/// Matches `bit_pos` in `pil/keccakf.pil`: `(y * 5 + x) * 64 + z`.
pub const fn keccakf_bit_pos(x: usize, y: usize, z: usize) -> usize {
    64 * x + 320 * y + z
}

/// Convert from the linear `[u64; 25]` lane representation to 5x5x64 bits.
#[allow(clippy::needless_range_loop)]
pub fn keccakf_state_from_linear(linear: &[u64; 25]) -> KeccakState {
    let mut state = [[[0u8; 64]; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            let word = linear[x + y * 5];
            for z in 0..64 {
                state[x][y][z] = ((word >> z) & 1) as u8;
            }
        }
    }
    state
}

/// Flatten the 5x5x64 state into the linear `[u8; 1600]` layout used by the trace.
#[allow(clippy::needless_range_loop)]
pub fn keccakf_state_flatten(state: &KeccakState) -> [u8; WIDTH] {
    let mut linear = [0u8; WIDTH];
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                linear[keccakf_bit_pos(x, y, z)] = state[x][y][z];
            }
        }
    }
    linear
}

use crate::keccakf_constants::WIDTH;

/// Apply a single Keccak-f round in the unreduced expression domain.
/// The input `state` must hold reduced bits (0/1); the output holds unreduced
/// values (0..=144) that the caller packs into `chunk_acc` and then reduces.
#[allow(clippy::needless_range_loop)]
pub fn keccak_f_round(state: &mut KeccakState, round: usize) {
    // θ (Theta) — column parity and mixing
    let mut parity = [[0u8; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            parity[x][z] = state[x][0][z] + state[x][1][z] + state[x][2][z] + state[x][3][z] + state[x][4][z];
        }
    }

    let mut d = [[0u8; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            d[x][z] = parity[(x + 4) % 5][z] + parity[(x + 1) % 5][(z + 63) % 64];
        }
    }

    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                state[x][y][z] += d[x][z];
            }
        }
    }

    // ρ (Rho) rotation + π (Pi) lane permutation
    let mut last = state[1][0];
    for t in 0..24 {
        let (x, y) = PI[t];
        let tmp = state[x][y];
        let shift = 64 - RHO[t];
        for z in 0..64 {
            state[x][y][z] = last[(z + shift) & 63];
        }
        last = tmp;
    }

    // χ (Chi) nonlinear step
    let mut plane = [[0u8; 64]; 5];
    for y in 0..5 {
        for x in 0..5 {
            for z in 0..64 {
                plane[x][z] = state[x][y][z];
            }
        }
        for x in 0..5 {
            for z in 0..64 {
                state[x][y][z] += (1 + plane[(x + 1) % 5][z]) * plane[(x + 2) % 5][z];
            }
        }
    }

    // ι (Iota) — add round constant
    for z in 0..64 {
        state[0][0][z] += RC_BITS[round][z] as u8;
    }
}
