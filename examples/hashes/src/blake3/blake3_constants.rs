/// Rows per BLAKE3 quarter-round function G
pub const CLOCKS_PER_G: usize = 1;

/// Number of G functions per round
pub const NUM_G_PER_ROUND: usize = 8;

/// Number of BLAKE3 rounds
pub const ROUNDS: usize = 7;

/// Rows per BLAKE3 round
pub const CLOCKS_PER_ROUND: usize = CLOCKS_PER_G * NUM_G_PER_ROUND;

/// Rows per BLAKE3 invocation
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
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Range checker size
pub const RANGE_SIZE: usize = 1 << 16;

/// XOR-rotate table size (3 rotation blocks of 4 offsets × 2^16 input pairs)
pub const TABLE_SIZE: usize = (1 << 19) + (1 << 18);
