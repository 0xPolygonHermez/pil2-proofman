//! Setup-snark command: orchestrate the final SNARK setup from vadcop_final artifacts.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::snark_setup::{gen_snark_setup, FinalSnark, SnarkSetupConfig};
use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};

/// Options for the setup-snark subcommand.
pub struct SetupSnarkOptions {
    /// Build directory (must contain provingKey/).
    pub build_dir: String,
    /// Powers-of-tau (.ptau) file path.
    pub powers_of_tau: Option<String>,
    /// SNARK type: "fflonk" (default) or "plonk".
    pub final_snark: String,
    /// Optional path to publics hash info JSON.
    pub publics_info: Option<String>,
    /// Only generate the recursivef step; skip the final SNARK.
    pub only_recursive_final: bool,
}

/// Run the setup-snark pipeline.
pub fn run_setup_snark(opts: &SetupSnarkOptions) -> Result<()> {
    let final_snark = match opts.final_snark.as_str() {
        "fflonk" => FinalSnark::Fflonk,
        "plonk" => FinalSnark::Plonk,
        // Not offered until its setup builds the pilfflonk proving key: today it would stop at the
        // final circuit.
        "pilfflonk" => bail!("--final-snark pilfflonk is not available yet: its setup is not complete"),
        other => bail!("unknown --final-snark {other:?}: expected fflonk or plonk"),
    };
    setup_snark(opts, final_snark)
}

/// [`run_setup_snark`] for `final_snark`, which stands for the name in `opts.final_snark`. It takes
/// pilfflonk as well: that is how the tests reach its setup while the command does not offer it.
fn setup_snark(opts: &SetupSnarkOptions, final_snark: FinalSnark) -> Result<()> {
    let build_dir = &opts.build_dir;

    // Read globalInfo to get the name.
    let global_info_path = PathBuf::from(build_dir).join("provingKey").join("pilout.globalInfo.json");

    if !global_info_path.exists() {
        bail!("Global info file not found: {:?}. Run the regular setup first.", global_info_path);
    }

    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)?;
    let name = global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string();
    let hash = global_info
        .get("hash")
        .and_then(|v| v.as_str())
        .with_context(|| format!("'hash' missing from {:?}; re-run the regular setup", global_info_path))?
        .to_string();
    if !proofman_common::hash_family::is_known_family(&hash) {
        bail!(
            "unknown hash family {:?} in {:?}; known: {:?}",
            hash,
            global_info_path,
            proofman_common::hash_family::FAMILIES
        );
    }
    // Refuse before building anything: the recursivef verifier and its circom templates are
    // poseidon-only, so a blake3 key would produce a snark stage that cannot verify.
    if !proofman_common::hash_family::supports_snark(&hash) {
        bail!(
            "the {hash} proving key at {:?} has no SNARK stage: the BN128 wrap is only built for \
             the poseidon families. Re-run the regular setup with --hash Poseidon1 or Poseidon2 \
             if you need a snark proof.",
            global_info_path
        );
    }

    proofman_starks_lib_c::set_hash_family_c(&hash);

    tracing::info!("setup-snark: name='{}', hash='{}', build_dir='{}'", name, hash, build_dir);

    // Read vadcop_final artifacts.
    let vadcop_dir = PathBuf::from(build_dir).join("provingKey").join(&name).join("vadcop_final");

    let const_root_path = vadcop_dir.join("vadcop_final.verkey.json");
    let starkinfo_path = vadcop_dir.join("vadcop_final.starkinfo.json");
    let verifier_info_path = vadcop_dir.join("vadcop_final.verifierinfo.json");

    for p in [&const_root_path, &starkinfo_path, &verifier_info_path] {
        if !p.exists() {
            bail!("Required file not found: {:?}. Make sure you have run the regular setup first.", p);
        }
    }

    let const_root_json: Value = serde_json::from_str(&fs::read_to_string(&const_root_path)?)?;
    let const_root: [u64; 4] =
        parse_const_root(&const_root_json).context("Failed to parse vadcop_final.verkey.json")?;

    let stark_info: Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
    let verifier_info: Value = serde_json::from_str(&fs::read_to_string(&verifier_info_path)?)?;

    // Read optional publics info.
    let publics_info: Option<Value> = if let Some(ref pi_path) = opts.publics_info {
        let content =
            fs::read_to_string(pi_path).with_context(|| format!("Failed to read publics info: {}", pi_path))?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Resolve tool paths (same logic as recursive_setup).
    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let final_snark_circom_helpers_dir = resolve_path_env("FINAL_SNARK_CIRCOM_HELPERS_DIR", "setup/final_snark_circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");
    let circom_exec = resolve_circom_exec(&circom_helpers_dir);

    // BN128 and circomlib paths.
    let circuits_bn128_path =
        resolve_path_env("CIRCUITS_BN128_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.bn128");
    let circomlib_path =
        crate::proving_key::recursive::ensure_node_module_subpath("CIRCOMLIB_PATH", "circomlib", "circuits");

    // Create provingKeySnark directory.
    let snark_dir = PathBuf::from(build_dir).join("provingKeySnark");
    fs::create_dir_all(&snark_dir)?;

    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    let snark_config = SnarkSetupConfig {
        build_dir,
        name: &name,
        hash: &hash,
        circom_exec: &circom_exec,
        circuits_gl_path: &circuits_gl_path,
        circuits_bn128_path: &circuits_bn128_path,
        circomlib_path: &circomlib_path,
        recurser_circuits_path: &recurser_circuits_path,
        std_pil_path: &std_pil_path,
        recurser_pil_path: &recurser_pil_path,
        circom_helpers_dir: &circom_helpers_dir,
        final_snark_circom_helpers_dir: &final_snark_circom_helpers_dir,
        powers_of_tau: opts.powers_of_tau.as_deref(),
        final_snark,
        publics_info,
        only_recursive_final: opts.only_recursive_final,
    };

    gen_snark_setup(&snark_config, &witness_tracker, &const_root, &stark_info, &verifier_info)
        .context("Final SNARK setup failed")?;

    tracing::info!("setup-snark completed successfully");
    Ok(())
}

/// Parse a vadcop_final.verkey.json array ([[u64;4]]) into [u64;4].
fn parse_const_root(json: &Value) -> Result<[u64; 4]> {
    let arr = json.as_array().ok_or_else(|| anyhow::anyhow!("verkey.json is not an array"))?;
    if arr.len() < 4 {
        bail!("verkey.json has {} elements, expected 4", arr.len());
    }
    let parse_one = |v: &Value, idx: usize| -> Result<u64> {
        v.as_u64()
            .or_else(|| v.as_str()?.parse::<u64>().ok())
            .ok_or_else(|| anyhow::anyhow!("verkey.json element {} is not a valid u64: {}", idx, v))
    };
    Ok([parse_one(&arr[0], 0)?, parse_one(&arr[1], 1)?, parse_one(&arr[2], 2)?, parse_one(&arr[3], 3)?])
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pil2_stark_recurser::plonk2pil::r1cs_types::read_r1cs_from_bytes;
    use proofman_fields::Bn254;

    use super::*;

    fn options(build_dir: &str, final_snark: &str) -> SetupSnarkOptions {
        SetupSnarkOptions {
            build_dir: build_dir.to_string(),
            powers_of_tau: None,
            final_snark: final_snark.to_string(),
            publics_info: None,
            only_recursive_final: false,
        }
    }

    /// The command takes PLONK and FFLONK only, and refuses anything else before it reads the build
    /// dir.
    #[test]
    fn refuses_other_final_snarks_up_front() {
        let missing = "/nonexistent/setup-snark/build";
        let err = run_setup_snark(&options(missing, "pilfflonk")).unwrap_err().to_string();
        assert!(err.contains("pilfflonk is not available yet"), "{err}");
        let err = run_setup_snark(&options(missing, "groth16")).unwrap_err().to_string();
        assert!(err.contains("unknown --final-snark \"groth16\""), "{err}");
        // One it takes gets as far as the build dir.
        let err = run_setup_snark(&options(missing, "plonk")).unwrap_err().to_string();
        assert!(err.contains("Global info file not found"), "{err}");
    }

    /// setup-snark for pilfflonk on the `vadcop_final` of a real program: the recursivef with custom
    /// trees and `lastLevelVerification` 0, the final circuit with custom templates, which the
    /// committed circom compiles over BN254, and the circuit's witness library.
    ///
    /// `SETUP_SNARK_BUILD_DIR` names the build dir of a recursive setup (`proofman-setup setup -r`
    /// with a poseidon family; fibonacci-square's takes some 11 minutes), and the test writes the
    /// outputs of setup-snark into it, as `setup-snark -b` does. `SETUP_SNARK_PUBLICS_INFO`, if set,
    /// is the `--publics-info`. That takes some 2 minutes and 5 GB:
    ///
    /// ```text
    /// SETUP_SNARK_BUILD_DIR=<dir> SETUP_SNARK_PUBLICS_INFO=$PWD/examples/fibonacci-square/src/publics_info.json \
    ///     cargo test --release -p pil2-stark-setup --features proofman-starks-lib-c/cpu-only \
    ///     --lib pilfflonk_final_circuit -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "a recursivef and a final circuit of millions of constraints, for SETUP_SNARK_BUILD_DIR"]
    fn pilfflonk_final_circuit() {
        let Ok(build_dir) = std::env::var("SETUP_SNARK_BUILD_DIR") else {
            eprintln!("skipped: SETUP_SNARK_BUILD_DIR does not name a recursive setup's build dir");
            return;
        };
        let opts = SetupSnarkOptions {
            publics_info: std::env::var("SETUP_SNARK_PUBLICS_INFO").ok(),
            ..options(&build_dir, "pilfflonk")
        };
        setup_snark(&opts, FinalSnark::Pilfflonk).expect("setup-snark for pilfflonk");

        let dir = Path::new(&build_dir);
        let starkinfo: Value = serde_json::from_str(
            &fs::read_to_string(dir.join("provingKeySnark/recursivef/recursivef.starkinfo.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(starkinfo["starkStruct"]["merkleTreeCustom"], true);
        assert_eq!(starkinfo["starkStruct"]["lastLevelVerification"], 0);

        let circuit = fs::read_to_string(dir.join("circom/final.circom")).unwrap();
        let head: Vec<&str> = circuit.lines().take(2).collect();
        assert_eq!(head, ["pragma circom 2.1.0;", "pragma custom_templates;"]);

        let final_dir = dir.join("provingKeySnark/final");
        let library = if cfg!(target_os = "macos") { "final.dylib" } else { "final.so" };
        for file in ["final.dat", library] {
            assert!(final_dir.join(file).is_file(), "{file} missing from {}", final_dir.display());
        }
        assert!(!final_dir.join("final.zkey").exists(), "a rapidsnark zkey for pilfflonk");

        // read_r1cs_from_bytes refuses an r1cs that is not over BN254. Its custom gates are
        // PoseidonT(5), for the hashes, and a Num2Bytes(nBits) per width up to 80 bits, for the
        // range checks of the Goldilocks arithmetic.
        let r1cs = read_r1cs_from_bytes::<Bn254>(&fs::read(dir.join("build/final.r1cs")).unwrap()).unwrap();
        let widths = Bn254::from_decimal("1").unwrap()..=Bn254::from_decimal("80").unwrap();
        for gate in &r1cs.custom_gates {
            match gate.template_name.as_str() {
                "PoseidonT" => assert_eq!(gate.parameters, [Bn254::from_decimal("5").unwrap()]),
                "Num2Bytes" => assert!(
                    matches!(gate.parameters[..], [n_bits] if widths.contains(&n_bits)),
                    "Num2Bytes{:?}",
                    gate.parameters
                ),
                other => panic!("a custom gate {other}"),
            }
        }
        let uses = |name: &str| {
            r1cs.custom_gates_uses.iter().filter(|u| r1cs.custom_gates[u.id as usize].template_name == name).count()
        };
        let (hashes, range_checks) = (uses("PoseidonT"), uses("Num2Bytes"));
        assert!(hashes > 0 && range_checks > 0, "{hashes} PoseidonT(5) uses, {range_checks} Num2Bytes uses");
        eprintln!(
            "final circuit: {} r1cs constraints, {hashes} PoseidonT(5) uses, {range_checks} Num2Bytes uses",
            r1cs.header.n_constraints
        );
    }
}
