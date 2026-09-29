//! The prover's wrappers on what they refuse before any proof: a `provingKey/` that is not there or
//! not a pilfflonk one, and a path the C side cannot take. The prover itself is tested in C++
//! (`make -C pil2-stark pilfflonk_test`, `pilfflonk_prover_test.cpp`) and end to end in
//! `cli/tests/pilfflonk_prove.rs`.

use std::ffi::OsStr;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use proofman_starks_lib_c::{PilFflonkErrorKind, PilFflonkProverCtx};

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_prover_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_missing_proving_key_is_an_io_error() {
    let dir = scratch("missing");
    let err = PilFflonkProverCtx::load(&dir.join("nothing")).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Io, "{err}");
    assert!(err.message.contains("pilfflonk_ctx_new: globalInfo: cannot open"), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_stark_proving_key_is_a_format_error() {
    let dir = scratch("stark");
    fs::write(dir.join("pilout.globalInfo.json"), r#"{"name": "x", "backend": "stark"}"#).unwrap();
    let err = PilFflonkProverCtx::load(&dir).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::Format, "{err}");
    assert!(err.message.contains("backend: must be \"pilfflonk\""), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_path_with_a_nul_byte_is_refused_before_the_call() {
    let err = PilFflonkProverCtx::load(Path::new(OsStr::from_bytes(b"a\0b"))).unwrap_err();
    assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
    assert!(err.message.contains("pilfflonk_ctx_new") && err.message.contains("NUL"), "{err}");
}
