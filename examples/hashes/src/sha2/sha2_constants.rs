/// SHA2-256s sharing every cell: a cell holds the same bit of each lane, Σ_k bit_k · 2^(SLOT_BITS·k)
pub const LANES: usize = 5;

/// Slot spacing inside a cell
pub const SLOT_BITS: usize = 12;

/// Width of the chunks the 32-bit additions are checked on
pub const CHUNK_BITS: usize = 8;

/// Chunks per 32-bit word
pub const NCHUNK: usize = 4;

/// Rows dedicated to loading the state
pub const CLOCKS_LOAD_STATE: usize = 4;

/// Rounds that load the input message
pub const CLOCKS_LOAD_INPUT: usize = 16;

/// Rounds, one per row
pub const NUM_ROUNDS: usize = 64;

/// Rows per SHA2-256 cycle (every lane at once)
pub const CLOCKS: usize = CLOCKS_LOAD_STATE + NUM_ROUNDS;

/// Table sizes, as circuits/sha2.pil declares them
pub const P3_ROWS: usize = 1 << 21;
pub const CHR_ROWS: usize = 1 << 20;
pub const RANGE_ROWS: usize = 1 << 20;
pub const CARRY_ROWS: usize = 16807;

/// Round constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
pub const RC: [u32; NUM_ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];
