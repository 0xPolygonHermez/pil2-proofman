//! The pilfflonk key of a plonk2pil wrap, which the tests of the wrap (`tests/poseidon_bn254_wrap.rs`)
//! and of its witness (`wrap-witness/tests`) set up alike: plonk2pil's PIL compiled with
//! `PIL2C_EXEC` over BN254, and set up with plonk2pil's fixed columns and the family's knobs
//! (`wrap::EXTRA_MULS`, and `wrap::MAX_CONSTRAINT_DEGREE` in the PIL). Included with `#[path]` from
//! outside this crate, so it reads nothing of `common` and is told where the repository is.

// Each test crate that includes it uses some of it.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pil2_stark_recurser::plonk2pil::setups::poseidon_bn254::wrap::EXTRA_MULS;
use pil2_stark_recurser::plonk2pil::PlonkResult;
use pilfflonk_setup::command::{DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau, write_tau_one_ptau};
use pilfflonk_setup::{run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupPilfflonkOptions};
use proofman_fields::Bn254;
use proofman_pilfflonk::FrBytes;

/// The powers of tau of a key (`pilfflonk_setup::test_ptau`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ptau {
    /// `τ = 1`: written at once, and enough to `pilfflonk check` a witness, which reads no
    /// commitment; but anyone can open anything with it, so a proof of it means nothing.
    TauOne,
    /// The full-width `τ` of the C++ helper: a proof verifies only if it is sound. Its powers take
    /// a scalar multiplication each over `num-bigint`, minutes for the 2^20 of a 2^16-row wrap in a
    /// debug build, so it is written once into the target's temporary directory and kept there.
    FixedTau,
}

/// Compiles the PIL of `res` in `dir` and sets it up there: the key's directory, `provingKey/`.
/// `repo_root` is the repository's root, where plonk2pil's PIL and the std are.
pub fn set_up_key(repo_root: &Path, dir: &Path, res: &PlonkResult<Bn254>, ptau: Ptau) -> PathBuf {
    let pilout = compile_pil(repo_root, dir, &res.pil_str);
    // More powers than the layout's largest degree, 13·N + 12 for L1 at the family's knobs.
    let n_g1 = 14 << res.n_bits;
    let powers_of_tau = match ptau {
        Ptau::TauOne => {
            let path = dir.join("tau_one.ptau");
            write_tau_one_ptau(&path, n_g1).unwrap();
            path
        }
        Ptau::FixedTau => fixed_tau_ptau(n_g1),
    };
    let setup = SetupPilfflonkOptions {
        airout_path: pilout,
        build_dir: dir.join("build"),
        powers_of_tau,
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: EXTRA_MULS,
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

/// Compiles `pil` in `dir` with `PIL2C_EXEC` over BN254 (`--field bn254`), with plonk2pil's PIL and the std
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
        .arg("bn254")
        .arg("-o")
        .arg(&pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pil2com: {log}");
    pilout
}

/// The [`Ptau::FixedTau`] ptau of `n_g1` powers: written once into the target's temporary
/// directory, through a file of its own that is renamed when whole, and read from there after.
fn fixed_tau_ptau(n_g1: usize) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_fixed_tau_{n_g1}.ptau"));
    if !path.is_file() {
        let partial = path.with_extension(format!("{}.partial", std::process::id()));
        write_fixed_tau_ptau(&partial, n_g1, &test_tau()).unwrap();
        fs::rename(&partial, &path).unwrap();
    }
    path
}
