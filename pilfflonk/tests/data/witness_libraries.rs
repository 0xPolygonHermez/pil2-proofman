//! The witness libraries of the tests, as Cargo builds them (pilfflonk/docs/README.md#witness): the
//! pilfflonk ones of the fixtures, `pilfflonk/tests/fixtures/<fixture>/rs` (`pilfflonk_fibonacci`,
//! `pilfflonk_connection` and `pilfflonk_all`), and a STARK one, `examples/fibonacci-square`'s
//! (`fibonacci_square`), which the pilfflonk loader refuses. Cargo builds each for the tests of its
//! own crate, and for those of a crate that has it as a dev-dependency, beside their binaries.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/witness_libraries.rs"] mod witness_libraries;`.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::path::PathBuf;

/// The dynamic library `name` that Cargo built for this test, beside its binary
/// (`target/<profile>/deps`): its crate's, or a dependency's.
pub fn built_library(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let path = exe.parent().unwrap().join(format!("{DLL_PREFIX}{name}{DLL_SUFFIX}"));
    assert!(path.is_file(), "{} is not built", path.display());
    path
}
