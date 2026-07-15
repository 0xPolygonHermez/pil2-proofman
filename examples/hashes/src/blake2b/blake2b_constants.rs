/// Rows per BLAKE2b mixing function G
pub const CLOCKS_PER_G: usize = 1;

/// Number of G functions per round
pub const NUM_G_PER_ROUND: usize = 8;

/// Number of BLAKE2b rounds
/// As per the "Too Much Crypto (https://eprint.iacr.org/2019/1492.pdf)" paper
pub const ROUNDS: usize = 8;

/// Rows per BLAKE2b round
pub const CLOCKS_PER_ROUND: usize = CLOCKS_PER_G * NUM_G_PER_ROUND;

/// Rows per BLAKE2b invocation
pub const CLOCKS: usize = CLOCKS_PER_ROUND * ROUNDS;

/// State indices (a, b, c, d) for each of the 8 G invocations of a round
pub const G_INDICES: [[usize; 4]; NUM_G_PER_ROUND] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// Message word permutation schedule
pub const SIGMA: [[usize; 16]; ROUNDS] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
];

/// Range checker size
pub const RANGE_SIZE: usize = 1 << 16;

/// XOR table size
pub const TABLE_SIZE: usize = 1 << 16;
