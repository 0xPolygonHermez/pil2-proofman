use std::fs;
use std::path::PathBuf;

use common::{compile, setup};

const SPECIAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/.."); // .../pil2-components/test/special

/// Sorted list of `*.pil` files directly inside `dir` (non-recursive).
fn pils_in(dir: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "pil").unwrap_or(false))
            .collect(),
        Err(e) => panic!("cannot read {dir}: {e}"),
    };
    v.sort();
    v
}

/// Per-PIL build dir under `<crate>/build/<sub>/<stem>`.
fn build_dir(sub: &str, stem: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/build")).join(sub).join(stem)
}

#[test]
fn special_pils_compile_and_setup() {
    let mut failures = Vec::new();
    for pil in pils_in(SPECIAL) {
        let stem = pil.file_stem().unwrap().to_string_lossy().into_owned();
        let build = build_dir("ok", &stem);
        match compile(&pil.to_string_lossy(), &build) {
            Ok(pilout) => {
                if let Err(e) = setup(&pilout, &build) {
                    failures.push(format!("{stem}: setup failed: {e}"));
                }
            }
            Err(e) => failures.push(format!("{stem}: compile failed: {e}")),
        }
    }
    assert!(failures.is_empty(), "special pils that should compile+setup failed:\n{}", failures.join("\n"));
}

#[test]
fn error_pils_fail_to_compile() {
    let dir = format!("{SPECIAL}/errors");
    let pils = pils_in(&dir);
    assert!(!pils.is_empty(), "no error fixtures found in {dir}");

    let mut wrong = Vec::new();
    for pil in pils {
        let stem = pil.file_stem().unwrap().to_string_lossy().into_owned();
        let build = build_dir("errors", &stem);
        if compile(&pil.to_string_lossy(), &build).is_ok() {
            wrong.push(stem);
        }
    }
    assert!(wrong.is_empty(), "error fixtures that compiled but must fail:\n{}", wrong.join("\n"));
}
