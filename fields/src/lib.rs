#![no_std]

extern crate alloc;

mod goldilocks;
mod integers;
mod goldilocks_quintic_extension;
mod field;
mod poseidon2;
mod poseidon1;
mod hash;
// Guest-only in production; `test` keeps its equivalence tests against the `blake3` crate running
// on the host, which is the only thing holding the two implementations identical.
#[cfg(any(all(target_os = "zkvm", target_vendor = "zisk"), test))]
mod blake3_core;
mod blake3_transcript;
mod sha256_transcript;
mod merkle;
mod transcript;
mod transcript_api;
mod fri;
mod extended_field;
mod poseidon2_constants;
mod poseidon1_constants;
mod utils;

pub use goldilocks::*;
pub use integers::*;
pub use goldilocks_quintic_extension::*;
pub use field::*;
pub use poseidon2::*;
pub use poseidon1::*;
pub use hash::*;
pub use blake3_transcript::*;
pub use sha256_transcript::*;
pub use merkle::*;
pub use transcript::*;
pub use transcript_api::*;
pub use fri::*;
pub use extended_field::*;
pub use poseidon2_constants::*;
pub use poseidon1_constants::*;
pub use utils::*;
