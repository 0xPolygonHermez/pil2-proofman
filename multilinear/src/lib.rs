//! Multilinear STARK primitives.
//!
//! This crate implements the following protocol:
//!
//! - Trace columns with `2^n` rows are viewed as multilinear polynomials over the
//!   Boolean hypercube `{0,1}^n`.
//! - Columns are committed via its univariate vision: the raw column values are the
//!   *coefficients* of a univariate polynomial of degree `< 2^n`, Reed–Solomon
//!   encoded on a coset of size `2^(n + log_blowup)` and Merkle-committed.
//! - Constraints are proven with a **zerocheck**, which terminates in column
//!   evaluation claims at a random point.
//! - Evaluation claims (including virtual shifted-column claims, via rotation
//!   kernels) are discharged with a batched PCS opening.

mod encoding;
mod eq;
mod error;
mod hypercube;
mod par;
mod sumcheck;
mod transcript;

pub use encoding::*;
pub use eq::*;
pub use error::*;
pub use hypercube::*;
pub use sumcheck::*;
pub use transcript::*;