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
    /// SNARK type: "fflonk" (default), "plonk" or "pilfflonk".
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
        "pilfflonk" => FinalSnark::Pilfflonk,
        other => bail!("unknown --final-snark {other:?}: expected fflonk, plonk or pilfflonk"),
    };
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
    use pilfflonk_setup::command::PROVING_KEY_DIR;
    use pilfflonk_setup::layout::max_degree;
    use pilfflonk_setup::solidity::VERIFIER_SOL_FILE;
    use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
    use proofman_fields::Bn128;
    use proofman_pilfflonk::global_info::{GLOBAL_CONSTRAINTS_FILE, GLOBAL_INFO_FILE};
    use proofman_pilfflonk::{AirFile, CalldataLayout, JsonFile, PilfflonkGlobalInfo, Vkey};

    use super::*;
    use crate::proving_key::snark_setup::tests::{assert_exec_gathers_the_air_columns, assert_set_up_at_the_family_knobs};

    fn options(build_dir: &str, final_snark: &str) -> SetupSnarkOptions {
        SetupSnarkOptions {
            build_dir: build_dir.to_string(),
            powers_of_tau: None,
            final_snark: final_snark.to_string(),
            publics_info: None,
            only_recursive_final: false,
        }
    }

    /// The command takes FFLONK, PLONK and pilfflonk, and refuses anything else before it reads the
    /// build dir.
    #[test]
    fn takes_the_three_final_snarks_and_refuses_others_up_front() {
        let missing = "/nonexistent/setup-snark/build";
        let err = run_setup_snark(&options(missing, "groth16")).unwrap_err().to_string();
        assert!(err.contains("unknown --final-snark \"groth16\": expected fflonk, plonk or pilfflonk"), "{err}");
        // One it takes gets as far as the build dir.
        for final_snark in ["fflonk", "plonk", "pilfflonk"] {
            let err = run_setup_snark(&options(missing, final_snark)).unwrap_err().to_string();
            assert!(err.contains("Global info file not found"), "{final_snark}: {err}");
        }
    }

    /// The powers `[τ^i]₁` of the test ptau [`pilfflonk_final_circuit`] writes: fibonacci-square's
    /// final circuit is an AIR of 2^19 rows, whose layout needs `13·2^19 + 12` with range checks,
    /// and this is `14·N`, as plonk2pil's wrap tests size theirs.
    const TEST_PTAU_POWERS: usize = 14 << 19;

    /// setup-snark for pilfflonk on the `vadcop_final` of a real program: the recursivef with custom
    /// trees and `lastLevelVerification` 0, the final circuit with custom templates, which the
    /// committed circom compiles over BN128, the circuit's witness library, and the pilfflonk key of
    /// the AIR plonk2pil makes of it, every file of `provingKeySnark/final/` as `gen_pilfflonk_key`
    /// lays it out.
    ///
    /// `SETUP_SNARK_BUILD_DIR` names the build dir of a recursive setup (`proofman-setup setup -r`
    /// with a poseidon family; fibonacci-square's takes some 11 minutes), and the test writes the
    /// outputs of setup-snark into it, as `setup-snark -b` does. `SETUP_SNARK_PUBLICS_INFO`, if set,
    /// is the `--publics-info`. `SETUP_SNARK_POWERS_OF_TAU` is the `--powers-of-tau`: if there is no
    /// file there, the test writes a ptau of [`TEST_PTAU_POWERS`] powers there, with the fixed τ of
    /// the tests (`pilfflonk_setup::test_ptau`), never to be used for a real key. The PIL of the AIR
    /// is compiled over BN128 with `PIL2C_EXEC`, which must have `--field`
    /// (pilfflonk/docs/README.md#compile-pil). With fibonacci-square and the Hermez ptau of 2^24 on
    /// 32 threads, that takes some 4 minutes and 3.5 GB, most of it the build of the final
    /// circuit's witness library; the test ptau holds 0.47 GB on disk:
    ///
    /// ```text
    /// SETUP_SNARK_BUILD_DIR=<dir> SETUP_SNARK_PUBLICS_INFO=$PWD/examples/fibonacci-square/src/publics_info.json \
    ///     SETUP_SNARK_POWERS_OF_TAU=<ptau> PIL2C_EXEC=<pil2-compiler>/src/pil.js \
    ///     cargo test --release -p pil2-stark-setup --features proofman-starks-lib-c/cpu-only \
    ///     --lib pilfflonk_final_circuit -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "a recursivef and a final circuit of 2^19 rows, for SETUP_SNARK_BUILD_DIR"]
    fn pilfflonk_final_circuit() {
        let Ok(build_dir) = std::env::var("SETUP_SNARK_BUILD_DIR") else {
            eprintln!("skipped: SETUP_SNARK_BUILD_DIR does not name a recursive setup's build dir");
            return;
        };
        let ptau = std::env::var("SETUP_SNARK_POWERS_OF_TAU").expect("SETUP_SNARK_POWERS_OF_TAU names a ptau");
        if !Path::new(&ptau).exists() {
            eprintln!("writing a test ptau of {TEST_PTAU_POWERS} powers at {ptau}");
            write_fixed_tau_ptau(Path::new(&ptau), TEST_PTAU_POWERS, &test_tau()).unwrap();
        }
        let opts = SetupSnarkOptions {
            powers_of_tau: Some(ptau),
            publics_info: std::env::var("SETUP_SNARK_PUBLICS_INFO").ok(),
            ..options(&build_dir, "pilfflonk")
        };
        run_setup_snark(&opts).unwrap_or_else(|e| panic!("setup-snark for pilfflonk: {e:#}"));

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

        // read_r1cs_from_bytes refuses an r1cs that is not over BN128. Its custom gates are
        // PoseidonT(5), for the hashes, and a Num2Bytes(nBits) per width up to 80 bits, for the
        // range checks of the Goldilocks arithmetic.
        let r1cs = read_r1cs_from_bytes::<Bn128>(&fs::read(dir.join("build/final.r1cs")).unwrap()).unwrap();
        let widths = Bn128::from_decimal("1").unwrap()..=Bn128::from_decimal("80").unwrap();
        for gate in &r1cs.custom_gates {
            match gate.template_name.as_str() {
                "PoseidonT" => assert_eq!(gate.parameters, [Bn128::from_decimal("5").unwrap()]),
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

        // Every file of provingKeySnark/final/, and the key's under provingKey/.
        let final_dir = dir.join("provingKeySnark/final");
        let proving_key = final_dir.join(PROVING_KEY_DIR);
        let global_info = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
        assert_eq!(global_info.name, "final", "the pilout's name, build/final.pilout's");
        let name: Value =
            serde_json::from_str(&fs::read_to_string(dir.join("provingKey/pilout.globalInfo.json")).unwrap()).unwrap();
        let name = name["name"].as_str().unwrap();
        let contract = format!("{}{}Verifier.sol", name[..1].to_uppercase(), &name[1..]);
        let library = if cfg!(target_os = "macos") { "final.dylib" } else { "final.so" };
        let mut files: Vec<PathBuf> = ["final.dat", library, "final.exec", &contract, &format!("I{contract}")]
            .iter()
            .map(|file| final_dir.join(file))
            .collect();
        files.extend([GLOBAL_INFO_FILE, GLOBAL_CONSTRAINTS_FILE].map(|file| proving_key.join(file)));
        let verifier = global_info.backend_dir(&proving_key).join(VERIFIER_SOL_FILE);
        files.extend([global_info.srs_path(&proving_key), global_info.vkey_path(&proving_key), verifier.clone()]);
        for air_file in [
            AirFile::Const,
            AirFile::PilfflonkInfo,
            AirFile::ExpressionsInfo,
            AirFile::VerifierInfo,
            AirFile::Bin,
            AirFile::Verkey,
        ] {
            files.push(global_info.air_file(&proving_key, 0, 0, air_file).unwrap());
        }
        for file in &files {
            assert!(file.is_file(), "{} missing", file.display());
        }
        assert!(!final_dir.join("final.zkey").exists(), "a rapidsnark zkey for pilfflonk");

        // At the wrap family's knobs, the layout of an AIR with range checks.
        assert_set_up_at_the_family_knobs(&global_info);
        let vkey = Vkey::read(&global_info.vkey_path(&proving_key)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(vkey.n_public, 1, "the final circuit's one public, its publics hash");
        let degree = max_degree(&vkey.layout);
        assert_eq!(degree, 13 * (1 << vkey.power) + 12, "the largest degree of layout L1 with range checks");
        assert_exec_gathers_the_air_columns(&final_dir.join("final.exec"), &proving_key, range_checks);

        // The project's verifier extends the key's, from where it is, with a proof of its calldata.
        let words = CalldataLayout::of(&vkey).words();
        let project = fs::read_to_string(final_dir.join(&contract)).unwrap();
        let source = verifier.strip_prefix(&final_dir).unwrap().display().to_string();
        for line in [
            format!("import {{PilfflonkVerifier}} from \"./{source}\";"),
            format!("bytes32[{words}] memory proofDecoded = abi.decode(proofBytes, (bytes32[{words}]));"),
        ] {
            assert!(project.contains(&line), "{line} missing from {contract}");
        }
        eprintln!(
            "pilfflonk key: 2^{} rows, {} f, {degree} powers [τ^i]₁, {words} calldata words",
            vkey.power,
            vkey.layout.0.len()
        );
    }
}
