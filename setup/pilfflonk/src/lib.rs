//! Setup for the pilfflonk backend: validates a BN254 pilout, groups its polynomials for fflonk and
//! writes the bytecode, the keys and the `provingKey/` directory.
//!
//! The command is [`command::run_setup_pilfflonk`], which `proofman-setup setup-pilfflonk` calls
//! (`pil2-stark-setup` hosts it; this crate does not depend on that one:
//! pilfflonk/docs/README.md#code-map). Its steps (pilfflonk/docs/README.md#setup-pilfflonk):
//!
//! | Step | Module |
//! |---|---|
//! | reading and validating the pilout (pilfflonk/docs/README.md#what-the-setup-refuses) | [`validate`] |
//! | the fixed columns and `<air>.const` | [`fixed`] |
//! | `pilout.globalInfo.json` | [`global_info`] |
//! | the symbolic passes over BN254 (pilfflonk/docs/protocol.md#degree-search) | [`passes`] |
//! | the committed polynomials, their bounds, `nBitsExt` and the layout | [`layout`] |
//! | the grouping of the committed polynomials in `f_i` (pilfflonk/docs/protocol.md#grouping-rules) | [`grouping`] |
//! | `<air>.pilfflonkinfo.json` from the passes' result | [`air_info`] |
//! | `pilfflonk.srs.bin`, `<air>.verkey.json` and `[τ]₂` | [`keys`] |
//! | the vkey's digest (pilfflonk/docs/formats.md#digest) | [`digest`] |
//! | `<air>.bin` | [`bytecode`] |
//! | `pilfflonk.verifier.sol`, with `--solidity` (pilfflonk/docs/verifier.md#solidity-verifier) | [`solidity`] |
//!
//! With the feature `test-ptau`, the module `test_ptau` writes a ptau with `τ = 1`, for tests only.

pub mod air_info;
pub mod bytecode;
pub mod command;
pub mod digest;
pub mod error;
pub mod fixed;
pub mod global_info;
pub mod grouping;
pub mod keys;
pub mod layout;
pub mod passes;
pub mod solidity;
pub mod validate;

#[cfg(feature = "test-ptau")]
pub mod test_ptau;

pub use command::{run_setup_pilfflonk, SetupPilfflonkOptions};
pub use error::SetupError;
