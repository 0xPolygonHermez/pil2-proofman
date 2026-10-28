//! `proofman-setup setup-snark` through the binary, with its defaults, on the build dir of a blake3
//! key whose vadcop_final verkey is not a root: it fails where the verkey is read, which tells
//! whether it ran setup-recursivef's step first. Each run is a process of its own, as the hash
//! family the starks library takes can be set only once in a process.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh directory for one test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("setup_snark_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// In `dir`, the build dir of a blake3 key "fibo" whose vadcop_final's files are there but its
/// verkey is no root (an object), with a recursivef if `recursivef`.
fn blake3_build_dir(dir: &Path, recursivef: bool) {
    let proving_key = dir.join("provingKey");
    let vadcop_final = proving_key.join("fibo/vadcop_final");
    fs::create_dir_all(&vadcop_final).unwrap();
    fs::write(proving_key.join("pilout.globalInfo.json"), r#"{"name": "fibo", "hash": "blake3"}"#).unwrap();
    for file in ["verkey", "starkinfo", "verifierinfo"] {
        fs::write(vadcop_final.join(format!("vadcop_final.{file}.json")), "{}").unwrap();
    }
    if recursivef {
        let recursivef = dir.join("provingKeySnark/recursivef");
        fs::create_dir_all(&recursivef).unwrap();
        fs::write(recursivef.join("recursivef.starkinfo.json"), "{}").unwrap();
    }
}

/// `proofman-setup setup-snark -b <dir>` and `args`, and what it wrote: the logs and the error.
fn setup_snark(dir: &Path, args: &[&str]) -> (Output, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_proofman-setup"))
        .args(["setup-snark", "-b", dir.to_str().unwrap()])
        .args(args)
        .output()
        .expect("proofman-setup runs");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out, text)
}

/// Without `--final-snark`, a blake3 key is wrapped in pilfflonk, so setup-snark takes it and runs
/// setup-recursivef's step on a key without a recursivef (here it fails on the verkey), and reuses
/// the recursivef of one that has it (and fails on the verkey itself). PLONK and FFLONK are refused
/// before the step.
#[test]
fn by_default_a_blake3_key_gets_its_recursivef_set_up_unless_it_has_one() {
    let (step, verkey) = ("setup-recursivef's step on the blake3 key", "Failed to parse vadcop_final.verkey.json");

    let missing = TestDir::new("without_recursivef");
    blake3_build_dir(&missing.0, false);
    let (out, text) = setup_snark(&missing.0, &[]);
    assert!(!out.status.success(), "{text}");
    assert!(text.contains("has no recursivef") && text.contains(step) && text.contains(verkey), "{text}");
    for final_snark in ["fflonk", "plonk"] {
        let (out, text) = setup_snark(&missing.0, &["--final-snark", final_snark]);
        assert!(!out.status.success(), "{final_snark}: {text}");
        assert!(text.contains("with --final-snark pilfflonk only") && !text.contains(step), "{final_snark}: {text}");
    }

    let set_up = TestDir::new("with_recursivef");
    blake3_build_dir(&set_up.0, true);
    let (out, text) = setup_snark(&set_up.0, &[]);
    assert!(!out.status.success(), "{text}");
    assert!(text.contains("reusing the blake3 key's recursivef") && text.contains(verkey), "{text}");
    assert!(!text.contains(step), "{text}");
}
