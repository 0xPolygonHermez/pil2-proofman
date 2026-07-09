//! Multilinear STARK primitives: Basefold polynomial commitment + sumcheck PIOP.
//!
//! This crate implements the protocol described in `docs/multilinear-pcs.md` and
//! `docs/multilinear-STARK.md`:
//!
//! - Trace columns with `2^n` rows are viewed as multilinear polynomials over the
//!   Boolean hypercube `{0,1}^n` (variable 1 = least-significant bit of the row index).
//! - Columns are committed via **Basefold**: the raw column values are the
//!   *coefficients* of a univariate polynomial of degree `< 2^n`, Reed–Solomon
//!   encoded on a coset of size `2^(n + log_blowup)` and Merkle-committed.
//! - Constraints are proven with a **zerocheck** (sumcheck of `eq(r,·) · C_α`),
//!   which terminates in column evaluation claims at a random point.
//! - Evaluation claims (including virtual shifted-column claims, via rotation
//!   kernels) are discharged with a batched Basefold opening: a degree-2 sumcheck
//!   run in lockstep with the FRI folding cascade, sharing challenges.
//!
//! Everything is pure Rust on top of the `fields` crate (Goldilocks + cubic
//! extension, Poseidon2 Merkle trees and Fiat–Shamir transcript).

mod encoding;
mod eq;
mod error;
mod evaluator;
mod hypercube;
mod ir;
mod merkle;
mod pcs;
mod prover;
mod sumcheck;
mod transcript;
mod verifier;
mod zerocheck;

pub use encoding::*;
pub use eq::*;
pub use error::*;
pub use evaluator::*;
pub use hypercube::*;
pub use ir::*;
pub use merkle::*;
pub use pcs::*;
pub use prover::*;
pub use sumcheck::*;
pub use transcript::*;
pub use verifier::*;
pub use zerocheck::*;
