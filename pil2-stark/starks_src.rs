//! Source locator for the vendored pil2-stark prover.
//!
//! `proofman-starks-lib-c` build-depends on this crate and compiles the
//! C/C++/CUDA sources that ship alongside it (this crate's own directory,
//! including the `blst` and `sppark` submodules). This crate carries no Rust
//! logic beyond pointing at that source tree.

use std::path::PathBuf;

/// Absolute path to the pil2-stark source tree (this crate's directory).
///
/// In local development this is the workspace path; for a published crate it is
/// the read-only registry checkout, with the submodule sources bundled in.
pub fn source_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
