//! The witness libraries of the tests, as Cargo builds them (pilfflonk/docs/README.md#witness): the
//! pilfflonk ones of the fixtures, `pilfflonk/tests/fixtures/<fixture>/rs` (`pilfflonk_fibonacci`,
//! `pilfflonk_connection` and `pilfflonk_all`), and a STARK one, `examples/fibonacci-square`'s
//! (`fibonacci_square`), which the pilfflonk loader refuses. Cargo builds each for the tests of its
//! own crate, and for those of a crate that has it as a dev-dependency.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/witness_libraries.rs"] mod witness_libraries;`.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::path::PathBuf;

/// The dynamic library `name` that Cargo built for this test: its crate's, or a dependency's.
///
/// With Cargo's classic layout it is beside the test's binary (`target/<profile>/deps`). With the
/// build-dir layout of newer Cargo, each unit has a directory of its own
/// (`target/<profile>/build/<package>/<hash>/out`), and the dylib is only in the one of its unit and
/// uplifted to `target/<profile>`: hence the binary's directory and then its ancestors.
pub fn built_library(name: &str) -> PathBuf {
    let file = format!("{DLL_PREFIX}{name}{DLL_SUFFIX}");
    let exe = std::env::current_exe().unwrap();
    let dirs = exe.parent().unwrap().ancestors();
    dirs.map(|dir| dir.join(&file)).find(|path| path.is_file()).unwrap_or_else(|| {
        panic!("{file} is not built: not beside {} nor in any of its ancestors", exe.display())
    })
}
