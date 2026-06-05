mod ffi_goldilocks;
mod ffi_starks;
pub use ffi_goldilocks::*;
pub use ffi_starks::*;

#[doc(hidden)]
#[cfg(feature = "stark-poseidon1")]
pub const GOLDILOCKS_MERKLE_TREE_ARITY: u64 = 4;
#[doc(hidden)]
#[cfg(not(feature = "stark-poseidon1"))]
pub const GOLDILOCKS_MERKLE_TREE_ARITY: u64 = 4;
