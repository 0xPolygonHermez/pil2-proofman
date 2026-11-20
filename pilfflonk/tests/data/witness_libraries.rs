//! The witness libraries of the tests, as Cargo builds them (pilfflonk/docs/README.md#witness): the
//! pilfflonk ones of the fixtures, `pilfflonk/tests/fixtures/<fixture>/rs` (`pilfflonk_fibonacci`,
//! `pilfflonk_connection` and `pilfflonk_all`), and a STARK one, `examples/fibonacci-square`'s
//! (`fibonacci_square`), which the pilfflonk loader refuses. Cargo builds each for the tests of its
//! own crate, and for those of a crate that has it as a dev-dependency.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/witness_libraries.rs"] mod witness_libraries;`.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::fs;
use std::path::{Path, PathBuf};

/// The dynamic library `name` that Cargo built for this test: its crate's, or a dependency's.
///
/// With Cargo's classic layout it is beside the test's binary (`target/<profile>/deps`). With the
/// build-dir layout of newer Cargo, each unit has a directory of its own
/// (`target/<profile>/build/<package>/<hash>/out`): a `dylib` is uplifted to `target/<profile>`, an
/// ancestor of the binary's directory, but a dependency's `cdylib` is only in its unit's, hence the
/// last look, at the newest of the `build/*/*/out` of an ancestor.
pub fn built_library(name: &str) -> PathBuf {
    let file = format!("{DLL_PREFIX}{name}{DLL_SUFFIX}");
    let exe = std::env::current_exe().unwrap();
    let dirs = || exe.parent().unwrap().ancestors();
    dirs()
        .map(|dir| dir.join(&file))
        .find(|path| path.is_file())
        .or_else(|| dirs().find_map(|dir| newest_unit_output(&dir.join("build"), &file)))
        .unwrap_or_else(|| panic!("{file} is not built: not beside {} nor in its ancestors", exe.display()))
}

/// The newest `build/<package>/<hash>/out/<file>`, of Cargo's build-dir layout.
fn newest_unit_output(build: &Path, file: &str) -> Option<PathBuf> {
    let subdirs = |dir: &Path| fs::read_dir(dir).into_iter().flatten().flatten().map(|entry| entry.path());
    subdirs(build)
        .flat_map(|package| subdirs(&package).collect::<Vec<_>>())
        .map(|unit| unit.join("out").join(file))
        .filter_map(|path| Some((fs::metadata(&path).ok()?.modified().ok()?, path)))
        .max()
        .map(|(_, path)| path)
}
