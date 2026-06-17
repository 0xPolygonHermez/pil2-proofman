/// Number of state bits (5 x 5 lanes x 64 bits).
pub const WIDTH: usize = 1600;

/// Number of Keccak-f rounds.
pub const ROUNDS: usize = 24;

/// Rows per Keccak-f invocation: 1 input row + one row per round.
pub const CLOCKS: usize = 1 + ROUNDS;

/// The maximum value any (unreduced) expression can reach during a round.
pub const MAX_EXPR_VALUE: u32 = 144;

/// Base used to pack unreduced round expressions into a single field value.
const BASE: u32 = MAX_EXPR_VALUE + 1;

/// Number of state bits packed into a single lookup-table chunk.
pub const TABLE_MAX_CHUNKS: usize = calculate_chunk_size() as usize;

/// Number of chunks needed to cover the whole state.
pub const NUM_CHUNKS: usize = WIDTH.div_ceil(TABLE_MAX_CHUNKS);

/// Number of rows in the Keccak-f lookup table.
pub const TABLE_SIZE: u32 = BASE.pow(TABLE_MAX_CHUNKS as u32);

/// Lookup table id, must match `KECCAKF_TABLE_ID` in `pil/keccakf_table.pil`.
pub const KECCAKF_TABLE_ID: usize = 5000;

/// Powers of `BASE`, used to weight each bit within a chunk.
pub const POWS_BASE: [u32; TABLE_MAX_CHUNKS] = {
    let mut pow = [1u32; TABLE_MAX_CHUNKS];
    let mut i = 1;
    while i < TABLE_MAX_CHUNKS {
        pow[i] = pow[i - 1] * BASE;
        i += 1;
    }
    pow
};

/// Largest chunk count whose value still fits below `P2_23`.
const fn calculate_chunk_size() -> u32 {
    /// 2^23, the bound used to size each lookup chunk.
    const P2_23: u64 = 1 << 23;

    let mut chunks = 1u32;
    while (BASE as u64).pow(chunks + 1) < P2_23 {
        chunks += 1;
    }
    chunks
}
