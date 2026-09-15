//! Runs the C++ unit tests as part of `cargo test`.
//!
//! The prover itself is C++ under `cpp/`, built by its own Makefile. Without
//! this the 51 gtest cases there are invisible to `cargo test` and to CI, so a
//! C++ regression would land green.
//!
//! This shells out rather than linking the library: `cpp/` exports no C API
//! yet, so there is nothing for a `build.rs` to bind, and linking it into the
//! Rust crate would put a C++ toolchain in the way of every `cargo test` for
//! code that is pure Rust. When the FFI surface arrives it belongs in its own
//! `*-lib-c` crate, the way `proofman-starks-lib-c` wraps pil2-stark.
//!
//! Skipped when the toolchain is missing, so a Rust-only checkout still tests
//! clean -- but never skipped silently: the reason is printed.

use std::path::PathBuf;
use std::process::Command;

fn cpp_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cpp")
}

/// The first tool the C++ build needs that is not installed, if any.
fn missing_tool() -> Option<String> {
    for (tool, arg) in [("make", "--version"), ("g++", "--version"), ("nasm", "-v")] {
        let ok = Command::new(tool).arg(arg).output().map(|o| o.status.success()).unwrap_or(false);
        if !ok {
            return Some(tool.to_string());
        }
    }
    None
}

#[test]
fn cpp_unit_tests_pass() {
    if let Some(missing) = missing_tool() {
        eprintln!("skipping the C++ tests: {missing} is not available");
        return;
    }

    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).to_string();
    let output = Command::new("make")
        .args(["-j", &jobs, "runtests"])
        .current_dir(cpp_dir())
        .output()
        .expect("failed to run make");

    // gtest reports failures on stdout, make's own errors on stderr; show both
    // or a failure here is untraceable.
    if !output.status.success() {
        panic!(
            "the C++ tests failed ({:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PASSED"), "make succeeded but gtest reported no passes:\n{stdout}");
}

/// The static library the future FFI crate will link must actually build.
#[test]
fn the_static_library_builds() {
    if let Some(missing) = missing_tool() {
        eprintln!("skipping the C++ library build: {missing} is not available");
        return;
    }

    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).to_string();
    let status =
        Command::new("make").args(["-j", &jobs, "lib"]).current_dir(cpp_dir()).status().expect("failed to run make");
    assert!(status.success(), "`make lib` failed");

    assert!(cpp_dir().join("lib/libpilfflonk.a").exists(), "libpilfflonk.a was not produced");
}
