//! Witness library for the plain-Fibonacci example proven with the
//! multilinear (Basefold + sumcheck) prover:
//!
//! ```text
//! proofman-cli prove --multilinear --witness-lib target/debug/libfibonacci_multilinear.so \
//!     --proving-key examples/fibonacci-multilinear/build/provingKey \
//!     --public-inputs examples/fibonacci-multilinear/src/inputs.json \
//!     --output-dir examples/fibonacci-multilinear/build/proofs
//! proofman-cli verify-multilinear \
//!     --proof examples/fibonacci-multilinear/build/proofs/FibonacciML_0.mlproof.bin \
//!     --proving-key examples/fibonacci-multilinear/build/provingKey
//! ```

mod fibonacci;
mod fibonacci_lib;
mod pil_helpers;

pub use fibonacci::*;
pub use fibonacci_lib::*;
pub use pil_helpers::*;
