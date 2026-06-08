use std::fs;
use std::path::{Path, PathBuf};

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

/// Sorted list of `*.pil` files anywhere under `dir`.
fn pils_in_recursive(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else { return };
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().map(|x| x == "pil").unwrap_or(false) {
                out.push(path);
            }
        }
    }
    let mut v = Vec::new();
    walk(dir, &mut v);
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
    let errors_root = PathBuf::from(format!("{SPECIAL}/errors"));
    let pils = pils_in_recursive(&errors_root);
    assert!(!pils.is_empty(), "no error fixtures found in {}", errors_root.display());

    let mut wrong = Vec::new();
    for pil in pils {
        let rel = pil.strip_prefix(&errors_root).unwrap().with_extension("");
        let label = rel.to_string_lossy().into_owned();
        let build = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/build/errors")).join(&rel);
        if compile(&pil.to_string_lossy(), &build).is_ok() {
            wrong.push(label);
        }
    }
    assert!(wrong.is_empty(), "error fixtures that compiled but must fail:\n{}", wrong.join("\n"));
}
