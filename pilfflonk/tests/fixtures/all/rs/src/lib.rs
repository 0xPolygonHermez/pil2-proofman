//! The pilfflonk witness library of the fixture `all`, `../all.pil` (plan M38c, D4): pil-fflonk's
//! example `all`, the Fibonacci, Connection, Permutation and Plookup state machines in one AIR. It
//! computes the witness over BN254's `Fr`, in the rows `pil_helpers` generates for the fixture's
//! BN254 pilout, and exports it with `pilfflonk_witness_library!`, as the Fibonacci's
//! (`pilfflonk/tests/fixtures/fibonacci/rs`, plan M38b).
//!
//! One library for both buses: the pilouts of `all_sum.pil` and `all_prod.pil` have the same
//! stage-1 columns and publics, and so the same rows, and pil-helpers writes the same
//! `src/pil_helpers` for both but for `PILOUT_HASH`. It is generated from the sum bus's, and
//! versioned, as the Fibonacci's: `tests/all_lib.rs` checks it is what pil-helpers writes. From the
//! repository root:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo run --bin proofman-setup -- compile-pil \
//!     -p pilfflonk/tests/fixtures/all/all_sum.pil -I ./pil2-components/lib/std/pil \
//!     -P pilfflonk/tests/fixtures/fibonacci/bn254.json -o <out>/all.pilout
//! cargo run --bin proofman-cli pil-helpers --pilout <out>/all.pilout \
//!     --path pilfflonk/tests/fixtures/all/rs/src -o
//! ```
//!
//! The pilout's name, and so the name of `AllPublics`, is the stem of its file, `all`.

mod all_lib;
#[allow(unused_imports)]
mod pil_helpers;

pub use all_lib::*;
pub use pil_helpers::*;
