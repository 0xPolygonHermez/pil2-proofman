//! The pilfflonk key of a plonk2pil wrap, which the tests of the wrap (`tests/poseidon_bn128_wrap.rs`)
//! and of its witness (`wrap-witness/tests`) set up alike: plonk2pil's PIL compiled with
//! `PIL2C_EXEC` over BN128, and set up with plonk2pil's fixed columns and the family's knobs
//! (`wrap::EXTRA_MULS`, and `wrap::MAX_CONSTRAINT_DEGREE` in the PIL). Included with `#[path]` from
//! outside this crate, so it reads nothing of `common` and is told where the repository is.

// Each test crate that includes it uses some of it.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pil2_stark_recurser::plonk2pil::setups::poseidon_bn128::wrap::EXTRA_MULS;
use pil2_stark_recurser::plonk2pil::PlonkResult;
use pilfflonk_setup::command::{DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::{run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupPilfflonkOptions};
use proofman_fields::Bn128;
use proofman_pilfflonk::FrBytes;

#[path = "../../../../pilfflonk/tests/data/ptau.rs"]
mod ptau;

/// Compiles the PIL of `res` in `dir` and sets it up there: the key's directory, `provingKey/`.
/// `repo_root` is the repository's root, where plonk2pil's PIL and the std are.
pub fn set_up_key(repo_root: &Path, dir: &Path, res: &PlonkResult<Bn128>) -> PathBuf {
    // More powers than the layout's largest degree, 13·N + 12 for L1 at the family's knobs.
    set_up_key_with(repo_root, dir, res, 14 << res.n_bits, EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE)
}

/// [`set_up_key`] with an official ptau of at least `n_g1` powers, at least the layout's largest
/// degree, and the family's `extra_muls` and `max_constraint_degree` (the blake3 wrap's are not the
/// poseidon one's).
pub fn set_up_key_with(
    repo_root: &Path,
    dir: &Path,
    res: &PlonkResult<Bn128>,
    n_g1: usize,
    extra_muls: u64,
    max_constraint_degree: u64,
) -> PathBuf {
    let pilout = compile_pil(repo_root, dir, &res.pil_str);
    let setup = SetupPilfflonkOptions {
        airout_path: pilout,
        build_dir: dir.join("build"),
        powers_of_tau: ptau::official(n_g1),
        max_constraint_degree,
        extra_muls,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    };
    let external = res
        .fixed_pols
        .iter()
        .map(|p| ExternalFixedColumn {
            name: p.name.clone(),
            index: p.index,
            values: p.values.iter().map(|&v| FrBytes::from(v)).collect(),
        })
        .collect();
    run_setup_pilfflonk_with_external_fixed(&setup, external).unwrap_or_else(|e| panic!("{e:#}"));
    setup.build_dir.join(PROVING_KEY_DIR)
}

/// Compiles `pil` in `dir` with `PIL2C_EXEC` over BN128 (`--field bn128`), with plonk2pil's PIL and the std
/// on the include path, as the pilfflonk fixtures are compiled: the pilout.
pub fn compile_pil(repo_root: &Path, dir: &Path, pil: &str) -> PathBuf {
    let (source, pilout) = (dir.join("wrap.pil"), dir.join("wrap.pilout"));
    fs::write(&source, pil).unwrap();
    let includes = ["setup/stark-recurser/plonk2pil/pil", "pil2-components/lib/std/pil"];
    let includes: Vec<String> = includes.iter().map(|p| repo_root.join(p).display().to_string()).collect();
    let out = Command::new(std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC names a pil2com that has `--field`"))
        .arg(&source)
        .arg("-I")
        .arg(includes.join(","))
        .arg("--field")
        .arg("bn128")
        .arg("-o")
        .arg(&pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pil2com: {log}");
    pilout
}
