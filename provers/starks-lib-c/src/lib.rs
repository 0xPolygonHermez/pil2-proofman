mod ffi_goldilocks;
mod ffi_starks;
pub use ffi_goldilocks::*;
pub use ffi_starks::*;

#[doc(hidden)]
pub const GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY: u64 = 4;
