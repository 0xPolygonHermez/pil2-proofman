#![cfg_attr(not(feature = "std"), no_std)]

//! STARK verification primitives: `stark_verify`, the `VerifierInfo` it reads,
//! and the proof types.
//!
//! No per-family verifier is committed here. The recursion aggregator binds the
//! application's publics into the q_verify expression, so a verifier generated
//! for one application rejects proofs another's correct prover produced. proofman
//! verifies from the proving key (`<base>.starkinfo.json` + `<base>.verifier.bin` +
//! `<base>.verkey.json`, see `proofman::verify_proof`); consumers that need a
//! compiled `no_std` verifier commit their own, generated from their own key by
//! `proofman-setup setup -r`, against the items this crate exports.

extern crate alloc;

#[macro_use]
mod log;

mod proof;
mod verifier;

pub use proof::*;
pub use verifier::*;
