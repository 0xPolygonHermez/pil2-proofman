//! The pilfflonk witness library (pilfflonk/docs/README.md#witness) of the Connection fixture,
//! `../connection.pil`: it computes the witness over BN254's `Fr`, in the rows `pil_helpers`
//! generates for the fixture's BN254 pilout, and exports it with `pilfflonk_witness_library!`, as
//! the Fibonacci's (`pilfflonk/tests/fixtures/fibonacci/rs`).
//!
//! One library for both buses: the pilouts of `connection_sum.pil` and `connection_prod.pil` have the
//! same stage-1 columns, and so the same rows, and pil-helpers writes the same `src/pil_helpers` for
//! both but for `PILOUT_HASH`. It is generated from the sum bus's, and versioned, as the
//! Fibonacci's: `tests/connection_lib.rs` checks it is what pil-helpers writes. From the repository
//! root:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo run --bin proofman-setup -- compile-pil \
//!     -p pilfflonk/tests/fixtures/connection/connection_sum.pil -I ./pil2-components/lib/std/pil \
//!     --field bn254 -o <out>/connection.pilout
//! cargo run --bin proofman-cli pil-helpers --pilout <out>/connection.pilout \
//!     --path pilfflonk/tests/fixtures/connection/rs/src -o
//! ```
//!
//! The pilout's name is the stem of its file, `connection`. The program has no publics, and so no
//! `ConnectionPublics`.

mod connection_lib;
#[allow(unused_imports)]
mod pil_helpers;

pub use connection_lib::*;
pub use pil_helpers::*;
