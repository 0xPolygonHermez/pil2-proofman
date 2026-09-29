//! Setup for the pilfflonk backend: validates a BN254 pilout, groups its polynomials for fflonk and
//! writes the bytecode, the keys and the `provingKey/` directory.
//!
//! The command is [`command::run_setup_pilfflonk`], which `proofman-setup setup-pilfflonk` calls
//! (`pil2-stark-setup` hosts it; this crate does not depend on that one, spec §5.2). Its steps:
//!
//! | Step | Module |
//! |---|---|
//! | reading and validating the pilout (spec §4.2.1) | [`validate`] |
//! | the fixed columns and `<air>.const` | [`fixed`] |
//! | `pilout.globalInfo.json` | [`global_info`] |
//! | the symbolic passes over BN254 (§4.2.2, §4.2.3) | [`passes`] |
//! | the committed polynomials, their bounds, `nBitsExt` and the layout (§4.2.4, A.1–A.3) | [`layout`] |
//! | `<air>.pilfflonkinfo.json` from the passes' result | [`air_info`] |
//! | `pilfflonk.srs.bin`, `<air>.verkey.json` and `[τ]₂` | [`keys`] |
//! | the vkey's digest (A.6) | [`digest`] |
//! | `<air>.bin` | [`bytecode`] |
//!
//! With the feature `test-ptau`, the module `test_ptau` writes a ptau with `τ = 1`, for tests only.

pub mod air_info;
pub mod bytecode;
pub mod command;
pub mod digest;
pub mod error;
pub mod fixed;
pub mod global_info;
pub mod keys;
pub mod layout;
pub mod passes;
pub mod validate;

#[cfg(feature = "test-ptau")]
pub mod test_ptau;

pub use command::{run_setup_pilfflonk, SetupPilfflonkOptions};
pub use error::SetupError;
