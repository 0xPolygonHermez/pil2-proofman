//! The pilfflonk witness library of the Fibonacci fixture, `../fibonacci.pil` (plan M38b, D4): it
//! computes the witness over BN254's `Fr`, in the rows `pil_helpers` generates for the fixture's
//! BN254 pilout, and exports it with `pilfflonk_witness_library!`.
//!
//! `src/pil_helpers` is generated, and versioned: a BN254 pilout needs a compiler that honours
//! `prime` (`PIL2C_EXEC`), so it cannot be regenerated on every build, as the STARK's test libraries
//! do in their `build.rs`. `tests/fibonacci_lib.rs` checks it is what pil-helpers writes. From the
//! repository root:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo run --bin proofman-setup -- compile-pil \
//!     -p pilfflonk/tests/fixtures/fibonacci/fibonacci.pil -I ./pil2-components/lib/std/pil \
//!     -P pilfflonk/tests/fixtures/fibonacci/bn254.json -o <out>/fibonacci.pilout
//! cargo run --bin proofman-cli pil-helpers --pilout <out>/fibonacci.pilout \
//!     --path pilfflonk/tests/fixtures/fibonacci/rs/src -o
//! ```
//!
//! The pilout's name, and so the name of `FibonacciPublics`, is the stem of its file.

mod fibonacci_lib;
#[allow(unused_imports)]
mod pil_helpers;

pub use fibonacci_lib::*;
pub use pil_helpers::*;
