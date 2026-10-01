//! Final SNARK setup: recursivef (GL→BN128 bridge) + final (fflonk/plonk/pilfflonk) steps.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use proofman_starks_lib_c::{generate_fflonk_zkey_c, generate_plonk_zkey_c, get_plonk_circuit_stats_c};

use crate::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use crate::io::recurser::{gen_circom, pil2circom, GenCircomInput, GenCircomOptions, Pil2CircomOptions};
use pil2_stark_recurser::stark2circom::templates::{gen_solidity, gen_iverifier, SnarkVerifier};
use crate::proving_key::{bctree, recursive::compile_pil};
use crate::io::fixed_cols;
use crate::output::witness_gen::WitnessTracker;
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil2_stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use pil2_stark_recurser::plonk2pil::{self, PlonkResult};
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::solidity::VERIFIER_SOL_FILE;
use pilfflonk_setup::{run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupPilfflonkOptions};
use proofman_common::hash_family::BN254_WRAP_FAMILY;
use proofman_fields::Bn254;
use proofman_pilfflonk::{CalldataLayout, FrBytes, JsonFile, PilfflonkGlobalInfo, Vkey, BN254_R};
use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

/// The protocol that proves the final circuit, the verifier of the recursivef. It decides how the
/// recursivef commits to its trees, and with that the final circuit
/// ([`FinalSnark::recursivef_settings`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalSnark {
    /// rapidsnark's FFLONK, over the final circuit's r1cs.
    Fflonk,
    /// rapidsnark's PLONK, over the final circuit's r1cs.
    Plonk,
    /// pilfflonk, over the AIR plonk2pil makes of the final circuit with custom gates
    /// ([`gen_pilfflonk_key`]).
    Pilfflonk,
}

impl FinalSnark {
    /// The recursivef's stark struct settings, which decide the final circuit as well: stark2circom
    /// writes the recursivef's verifier, and the final circuit that includes it, with circom custom
    /// templates exactly when the recursivef's trees are custom.
    ///
    /// Every protocol gets arity-4 Poseidon trees over BN128, a blowup of 6 and 19 bits of grinding:
    /// the JS's settings (`generateFinalSnarkSetup.js`), which grind 17. They differ in how the final
    /// circuit checks the trees:
    /// - PLONK and FFLONK: with circomlib's Poseidon, and Merkle paths that stop 2 levels short of
    ///   the root, at a published 16-node level (`lastLevelVerification` 2, which the JS never
    ///   implemented for BN128);
    /// - pilfflonk: with custom templates, one `PoseidonT(5)` custom gate per hash for plonk2pil to
    ///   lay out, and whole Merkle paths (`lastLevelVerification` 0, as the JS):
    ///   `circuits.bn128/custom/` has no last-level templates.
    fn recursivef_settings(self) -> StarkSettings {
        let (merkle_tree_custom, last_level_verification) = match self {
            FinalSnark::Fflonk | FinalSnark::Plonk => (false, 2),
            FinalSnark::Pilfflonk => (true, 0),
        };
        StarkSettings {
            verification_hash_type: Some("BN128".to_string()),
            blowup_factor: Some(6),
            merkle_tree_arity: Some(4),
            merkle_tree_custom: Some(merkle_tree_custom),
            last_level_verification: Some(last_level_verification),
            pow_bits: Some(19),
            ..Default::default()
        }
    }
}

/// The protocol's name, as `--final-snark` spells it, and snarkjs for PLONK and FFLONK.
impl fmt::Display for FinalSnark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FinalSnark::Fflonk => "fflonk",
            FinalSnark::Plonk => "plonk",
            FinalSnark::Pilfflonk => "pilfflonk",
        })
    }
}

/// Configuration for the final SNARK setup.
pub struct SnarkSetupConfig<'a> {
    /// Build directory (must contain provingKey/{name}/vadcop_final/).
    pub build_dir: &'a str,
    /// Circuit name (from globalInfo.name).
    pub name: &'a str,
    /// Hash family (from globalInfo.hash, e.g. "Poseidon1"/"Poseidon2"). The
    /// recursivef plonk2pil and circom verifier must use the same family the
    /// proving key was set up with — never a hardcoded default.
    pub hash: &'a str,
    /// Tool paths (same sources as recursive setup).
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    /// BN128 circuit library path (node_modules/stark-recurser/src/pil2circom/circuits.bn128).
    pub circuits_bn128_path: &'a str,
    /// Circomlib circuits path (node_modules/circomlib/circuits).
    pub circomlib_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
    /// Directory containing BN128 fr.cpp/fr.asm and Makefile for the `final` SNARK witness library.
    /// Corresponds to `final_snark_circom/`
    pub final_snark_circom_helpers_dir: &'a str,
    /// Powers-of-tau (.ptau) file for the final SNARK's setup (required for PLONK, FFLONK and
    /// pilfflonk if !only_recursive_final).
    pub powers_of_tau: Option<&'a str>,
    /// The protocol that proves the final circuit.
    pub final_snark: FinalSnark,
    /// Optional publics hash info JSON value.
    pub publics_info: Option<Value>,
    /// When true, only run the recursivef step and stop before the final SNARK.
    pub only_recursive_final: bool,
}

/// Run the final SNARK setup pipeline.
///
/// Phase 1 (recursivef): GL → BN128 bridge circuit.
/// Phase 2 (final):      BN128 R1CS → SNARK zkey + Solidity verifiers (PLONK and FFLONK), or the
///                       pilfflonk key of its AIR + Solidity verifiers (pilfflonk).
pub fn gen_snark_setup(
    config: &SnarkSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
    const_root: &[u64; 4],
    stark_info: &Value,
    verifier_info: &Value,
) -> Result<()> {
    let build_dir = PathBuf::from(config.build_dir);
    let circom_dir = build_dir.join("circom");
    let build_path = build_dir.join("build");
    let pil_dir = build_dir.join("pil");
    let snark_dir = build_dir.join("provingKeySnark");

    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_path)?;
    fs::create_dir_all(&pil_dir)?;
    fs::create_dir_all(&snark_dir)?;

    // Write vadcop_final.verkey.json (copy of the incoming constRoot).
    let const_root_json: Vec<u64> = const_root.to_vec();
    fs::write(snark_dir.join("vadcop_final.verkey.json"), serde_json::to_string_pretty(&const_root_json)?)?;

    // ── Phase 1: recursivef ───────────────────────────────────────────────────
    let recursivef_dir = snark_dir.join("recursivef");
    fs::create_dir_all(&recursivef_dir)?;

    let const_root_str: [String; 4] =
        [const_root[0].to_string(), const_root[1].to_string(), const_root[2].to_string(), const_root[3].to_string()];

    // pil2circom: generate vadcop_final.verifier.circom
    let verifier_name_rf = "vadcop_final.verifier.circom";
    let pil2circom_opts = Pil2CircomOptions {
        skip_main: true,
        verkey_input: true,
        enable_input: false,
        input_challenges: false,
        hash: config.hash.to_string(),
    };
    let verifier_circom_rf = pil2circom(&const_root_str, stark_info, verifier_info, &pil2circom_opts)
        .context("pil2circom failed for recursivef")?;
    fs::write(circom_dir.join(verifier_name_rf), &verifier_circom_rf)?;

    // gen_circom: generate recursivef.circom using the recursivef.circom.ejs template.
    // basic_vk = [[constRoot]] (one airgroup, one air with the vadcop_final constRoot)
    // recursivef wraps a single proof: the template never reads `agg_arity`, and 0 is
    // rejected outright by `gen_recursive2`.
    let gen_opts_rf = GenCircomOptions {
        airgroup_id: None,
        has_compressor: false,
        has_recursion: false,
        is_final: false,
        agg_arity: 0,
    };
    let rf_basic_vk: Vec<Vec<Vec<String>>> = vec![vec![const_root_str.to_vec()]];
    let gen_input_rf = GenCircomInput {
        template_name: "src/recursion/templates/recursivef.circom.ejs",
        stark_infos: std::slice::from_ref(stark_info),
        vadcop_info: &Value::Null,
        verifier_filenames: &[verifier_name_rf.to_string()],
        basic_verification_keys: &rf_basic_vk,
        agg_verification_keys: &[],
        publics: &[],
        options: &gen_opts_rf,
    };
    let circom_rf = gen_circom(&gen_input_rf).context("gen_circom failed for recursivef")?;
    let circom_rf_path = circom_dir.join("recursivef.circom");
    fs::write(&circom_rf_path, &circom_rf)?;

    // Compile recursivef with GL circuits (prime goldilocks).
    tracing::info!("Compiling recursivef...");
    let compile_rf = std::process::Command::new(config.circom_exec)
        .args([
            "--O2",
            "--r1cs",
            "--prime",
            "goldilocks",
            "--c",
            "--verbose",
            "-l",
            config.recurser_circuits_path,
            "-l",
            config.circuits_gl_path,
        ])
        .arg(circom_rf_path.to_str().unwrap())
        .arg("-o")
        .arg(build_path.to_str().unwrap())
        .output()
        .context("Failed to execute circom for recursivef")?;
    if !compile_rf.status.success() {
        bail!("Circom compilation failed for recursivef: {}", String::from_utf8_lossy(&compile_rf.stderr));
    }

    // Copy .dat file from build to provingKeySnark/recursivef/.
    let dat_src_rf = build_path.join("recursivef_cpp").join("recursivef.dat");
    if dat_src_rf.exists() {
        fs::copy(&dat_src_rf, recursivef_dir.join("recursivef.dat"))?;
    }

    // Generate witness library (background).
    witness_tracker.run_witness_library_generation(
        config.build_dir,
        recursivef_dir.to_str().unwrap_or(""),
        "recursivef",
        "recursivef",
        config.circom_helpers_dir,
    );

    // plonk2pil → PIL → compile PIL.
    let r1cs_rf = build_path.join("recursivef.r1cs");
    let r1cs_data_rf =
        fs::read(&r1cs_rf).with_context(|| format!("Failed to read recursivef.r1cs: {}", r1cs_rf.display()))?;
    let plonk_opts_rf = PlonkOptions {
        airgroup_name: Some("Recursivef".to_string()),
        max_constraint_degree: None,
        hash_id: config.hash.to_string(),
        merge_copies: true,
        // blake3 chooses LANES in its own setup; None takes the air's default of 4.
        blake3_lanes: None,
        min_n_bits: None,
    };
    let _span = tracing::info_span!("stage", t = "recursivef").entered();
    let plonk_rf = plonk2pil::plonk2pil(&r1cs_data_rf, "aggregation", &plonk_opts_rf)
        .context("plonk2pil failed for recursivef")?;

    // Write fixed pols binary.
    let fixed_bin_rf = build_path.join("recursivef.fixed.bin");
    let fixed_info_rf: Vec<(String, Vec<u32>, Vec<u64>)> =
        plonk_rf.fixed_pols.iter().map(|fp| (fp.name.clone(), vec![fp.index as u32], fp.values.clone())).collect();
    fixed_cols::write_fixed_pols_bin(
        fixed_bin_rf.to_str().unwrap(),
        &plonk_rf.airgroup_name,
        &plonk_rf.air_name,
        1u64 << plonk_rf.n_bits,
        &fixed_info_rf,
    )?;

    let pil_rf = pil_dir.join("recursivef.pil");
    fs::write(&pil_rf, &plonk_rf.pil_str)?;

    write_exec(&recursivef_dir.join("recursivef.exec"), &plonk_rf.exec)?;

    let pilout_rf = build_path.join("recursivef.pilout");
    compile_pil(pil_rf.to_str().unwrap(), pilout_rf.to_str().unwrap(), config.std_pil_path, config.recurser_pil_path)?;

    // pil_info with BN128 stark struct.
    let proxy_rf = PilOutProxy::new(pilout_rf.to_str().unwrap_or(""))
        .map_err(|e| anyhow::anyhow!("Failed to load recursivef pilout: {}", e))?;
    let pilout_inner = &proxy_rf.pilout;
    if pilout_inner.air_groups.is_empty() || pilout_inner.air_groups[0].airs.is_empty() {
        bail!("recursivef pilout has no AIR groups");
    }
    let air_rf = &pilout_inner.air_groups[0].airs[0];
    let n_bits_rf = {
        let nr = air_rf.num_rows.unwrap_or(0) as usize;
        if nr > 0 {
            (nr as f64).log2() as usize
        } else {
            plonk_rf.n_bits
        }
    };

    let stark_struct_rf = generate_stark_struct(&config.final_snark.recursivef_settings(), n_bits_rf, config.hash);

    let pil_result_rf = crate::pil::info::pil_info(pilout_inner, 0, 0, &stark_struct_rf, &Default::default())?;

    // Build starkinfo output.
    let opening_points_rf = crate::output::stark_info::collect_opening_points(&pil_result_rf.setup);
    let field_size = crate::types::security::goldilocks_safe_extension_field_size();
    let ev_map_len_rf = pil_result_rf.pil_code.ev_map.len();
    let log_folding_factors_rf = crate::output::stark_info::compute_log_folding_factors(&stark_struct_rf);
    let regime = crate::types::security::regimes::DecodingRegime::Jbr;
    let fri_config_rf = crate::types::security::pcs::FriConfig {
        field_size,
        trace_length: 1u32 << stark_struct_rf.n_bits,
        rate: 1.0 / (1u64 << (stark_struct_rf.n_bits_ext - stark_struct_rf.n_bits)) as f64,
        batch_size: ev_map_len_rf.max(1) as u64,
        batching: crate::types::security::pcs::Batching::Powers,
        log_folding_factors: log_folding_factors_rf,
        max_grinding_bits_query: stark_struct_rf.pow_bits as u64,
        use_max_grinding_bits_query: true,
        tree_arity: stark_struct_rf.merkle_tree_arity as u64,
        hash_size_bits: 256,
        target_security_bits: 128,
        regime,
    };
    let fri_rf = crate::types::security::pcs::Fri::new(fri_config_rf);

    let starkinfo_rf = crate::output::stark_info::build_starkinfo_output(
        &pil_result_rf.setup,
        &stark_struct_rf,
        &pil_result_rf.pil_code,
        &opening_points_rf,
        &fri_rf,
        0,
        0,
        "Recursivef",
        pil_result_rf.c_exp_id,
        pil_result_rf.fri_exp_id,
        pil_result_rf.q_deg,
    );
    let starkinfo_rf_json = crate::output::json::to_json_string(&starkinfo_rf)?;
    let starkinfo_rf_path = recursivef_dir.join("recursivef.starkinfo.json");
    fs::write(&starkinfo_rf_path, &starkinfo_rf_json)?;

    let verifier_info_rf = &pil_result_rf.pil_code.verifier_info;
    let expressions_info_rf = &pil_result_rf.pil_code.expressions_info;

    fs::write(
        recursivef_dir.join("recursivef.verifierinfo.json"),
        crate::output::json::to_json_string(verifier_info_rf)?,
    )?;
    fs::write(
        recursivef_dir.join("recursivef.expressionsinfo.json"),
        crate::output::json::to_json_string(expressions_info_rf)?,
    )?;

    // Write const file.
    let const_rf = recursivef_dir.join("recursivef.const");
    {
        let plonk_values = fixed_cols::reorder_plonk_pols_for_pilout(&plonk_rf.fixed_pols, &pilout_inner.symbols, 0, 0);
        fixed_cols::write_const_file(const_rf.to_str().unwrap(), air_rf, &plonk_values)?;
    }

    // Compute const tree (bctree) for recursivef.
    tracing::info!("Computing constant tree for recursivef...");
    let verkey_rf_path = recursivef_dir.join("recursivef.verkey.json");
    let rf_const_root = bctree::compute_const_tree(
        const_rf.to_str().unwrap(),
        starkinfo_rf_path.to_str().unwrap(),
        verkey_rf_path.to_str().unwrap(),
    )?;
    let mut verkey_bin_rf = Vec::with_capacity(32);
    for &v in rf_const_root.iter() {
        verkey_bin_rf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(recursivef_dir.join("recursivef.verkey.bin"), &verkey_bin_rf)?;

    // Write bin files.
    let si_val_rf: Value = serde_json::from_str(&starkinfo_rf_json)?;
    let si_loaded_rf = crate::types::stark_info::StarkInfo::from_json(&si_val_rf)?;
    let ei_rf = crate::types::stark_info::ExpressionsInfo::try_from(expressions_info_rf)?;
    crate::io::bin_file::write_expressions_bin_file(
        recursivef_dir.join("recursivef.bin").to_str().unwrap(),
        &si_loaded_rf,
        &ei_rf,
    )?;
    let vi_rf_loaded = crate::types::stark_info::VerifierInfo::try_from(verifier_info_rf)?;
    crate::io::bin_file::write_verifier_expressions_bin_file(
        recursivef_dir.join("recursivef.verifier.bin").to_str().unwrap(),
        &si_loaded_rf,
        &vi_rf_loaded,
    )?;

    if config.only_recursive_final {
        tracing::info!("only_recursive_final=true: skipping final SNARK setup");
        witness_tracker.await_all()?;
        return Ok(());
    }

    // ── Phase 2: final SNARK ──────────────────────────────────────────────────
    let final_dir = snark_dir.join("final");
    fs::create_dir_all(&final_dir)?;

    let rf_const_root_json: Value = serde_json::from_str(
        &fs::read_to_string(&verkey_rf_path)
            .with_context(|| format!("Failed to read recursivef.verkey.json: {}", verkey_rf_path.display()))?,
    )?;
    // The verkey.json format depends on the hash type:
    //   GL      → [u64, u64, u64, u64]  (4-element JSON array)
    //   BN128   → "<decimal_string>"    (single BN128 field element as JSON string)
    // Either way we store as [String; 4], putting the scalar in [0] for BN128.
    let rf_const_root_str: [String; 4] = {
        if let Some(arr) = rf_const_root_json.as_array() {
            // GL case
            if arr.len() < 4 {
                bail!("recursivef verkey has fewer than 4 elements");
            }
            [
                arr[0]
                    .as_u64()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| arr[0].to_string().trim_matches('"').to_string()),
                arr[1]
                    .as_u64()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| arr[1].to_string().trim_matches('"').to_string()),
                arr[2]
                    .as_u64()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| arr[2].to_string().trim_matches('"').to_string()),
                arr[3]
                    .as_u64()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| arr[3].to_string().trim_matches('"').to_string()),
            ]
        } else if let Some(s) = rf_const_root_json.as_str() {
            // BN128 case: single scalar; store in slot 0, zeros in the rest
            [s.to_string(), "0".into(), "0".into(), "0".into()]
        } else {
            bail!("recursivef verkey.json has unexpected format: {}", rf_const_root_json);
        }
    };

    let starkinfo_rf_val: Value = serde_json::from_str(&starkinfo_rf_json)?;
    let verifierinfo_json_path = recursivef_dir.join("recursivef.verifierinfo.json");
    let verifierinfo_rf_val: Value =
        serde_json::from_str(&fs::read_to_string(&verifierinfo_json_path).with_context(|| {
            format!("Failed to read recursivef.verifierinfo.json: {}", verifierinfo_json_path.display())
        })?)?;

    // pil2circom: generate recursivef.verifier.circom (verkeyInput=false for final).
    let verifier_name_final = "recursivef.verifier.circom";
    let pil2circom_opts_final = Pil2CircomOptions {
        skip_main: true,
        verkey_input: false,
        enable_input: false,
        input_challenges: false,
        hash: config.hash.to_string(),
    };
    let verifier_circom_final =
        pil2circom(&rf_const_root_str, &starkinfo_rf_val, &verifierinfo_rf_val, &pil2circom_opts_final)
            .context("pil2circom failed for final")?;
    fs::write(circom_dir.join(verifier_name_final), &verifier_circom_final)?;

    // gen_circom: generate final.circom using final.circom.ejs template.
    let publics_vec: Vec<Value> =
        if let Some(ref pi) = config.publics_info { vec![pi.clone()] } else { vec![Value::Null] };
    // The snark final circuit wraps a single recursivef proof: the template never reads
    // `agg_arity`, and 0 is rejected outright by `gen_recursive2`.
    let gen_opts_final = GenCircomOptions {
        airgroup_id: None,
        has_compressor: false,
        has_recursion: false,
        is_final: true,
        agg_arity: 0,
    };
    let gen_input_final = GenCircomInput {
        template_name: "src/recursion/templates/final.circom.ejs",
        stark_infos: std::slice::from_ref(&starkinfo_rf_val),
        vadcop_info: &Value::Null,
        verifier_filenames: &[verifier_name_final.to_string()],
        basic_verification_keys: &[],
        agg_verification_keys: &[],
        publics: &publics_vec,
        options: &gen_opts_final,
    };
    let circom_final = gen_circom(&gen_input_final).context("gen_circom failed for final")?;
    let circom_final_path = circom_dir.join("final.circom");
    fs::write(&circom_final_path, &circom_final)?;

    // Compile final with BN128 circuits.
    tracing::info!("Compiling final...");
    let compile_final = std::process::Command::new(config.circom_exec)
        .args([
            "--O1",
            "--r1cs",
            "--inspect",
            "--wasm",
            "--c",
            "--verbose",
            "-l",
            config.recurser_circuits_path,
            "-l",
            config.circuits_bn128_path,
            "-l",
            config.circomlib_path,
        ])
        .arg(circom_final_path.to_str().unwrap())
        .arg("-o")
        .arg(build_path.to_str().unwrap())
        .output()
        .context("Failed to execute circom for final")?;
    if !compile_final.status.success() {
        bail!("Circom compilation failed for final: {}", String::from_utf8_lossy(&compile_final.stderr));
    }

    // Copy .dat file.
    let dat_src_final = build_path.join("final_cpp").join("final.dat");
    if dat_src_final.exists() {
        fs::copy(&dat_src_final, final_dir.join("final.dat"))?;
    }

    let r1cs_final = build_path.join("final.r1cs");
    if !r1cs_final.exists() {
        bail!("final.r1cs not found at {}: circom compilation may have failed", r1cs_final.display());
    }

    match config.final_snark {
        FinalSnark::Fflonk | FinalSnark::Plonk => {
            gen_rapidsnark_key(config, witness_tracker, &r1cs_final, &final_dir, const_root)?
        }
        FinalSnark::Pilfflonk => {
            gen_pilfflonk_key(config, witness_tracker, &build_path, &pil_dir, &final_dir, const_root)?
        }
    }

    // Write publics_info.json if provided.
    if let Some(ref pi) = config.publics_info {
        fs::write(snark_dir.join("publics_info.json"), serde_json::to_string_pretty(pi)?)?;
    }

    tracing::info!("Final SNARK setup complete");
    Ok(())
}

/// The PLONK or FFLONK key of the final circuit: rapidsnark's zkey of its r1cs, snarkjs's
/// verification key and Solidity verifier, the project's Solidity verifier around that one, and
/// the circuit's witness library.
fn gen_rapidsnark_key(
    config: &SnarkSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
    r1cs_final: &Path,
    final_dir: &Path,
    const_root: &[u64; 4],
) -> Result<()> {
    let fflonk = match config.final_snark {
        FinalSnark::Fflonk => true,
        FinalSnark::Plonk => false,
        FinalSnark::Pilfflonk => bail!("the pilfflonk final circuit has no rapidsnark key"),
    };

    if let Some((n_constraints, n_additions)) = get_plonk_circuit_stats_c(r1cs_final.to_str().unwrap()) {
        let circuit_power = std::cmp::max(3, 64 - (n_constraints + 1).leading_zeros() as u64);
        tracing::info!(
            "Final circuit: {} plonk constraints, {} plonk additions (circuit power {}, domain size {})",
            n_constraints,
            n_additions,
            circuit_power,
            1u64 << circuit_power
        );
    }

    // Validate inputs for the zkey setup before launching parallel work.
    let powers_of_tau = required_powers_of_tau(config)?;
    let zkey_final = final_dir.join("final.zkey");

    // Launch witness library generation (make) in background, then run the
    // zkey FFI setup concurrently on this thread — both only need the circom
    // output and produce independent artifacts.
    run_final_witness_library_generation(config, witness_tracker, final_dir);

    tracing::info!("Running {} setup via FFI (parallel with make)...", config.final_snark);
    let ret = if fflonk {
        generate_fflonk_zkey_c(r1cs_final.to_str().unwrap(), powers_of_tau, zkey_final.to_str().unwrap())
    } else {
        generate_plonk_zkey_c(r1cs_final.to_str().unwrap(), powers_of_tau, zkey_final.to_str().unwrap())
    };
    if ret != 0 {
        bail!("{} setup FFI call failed with return code {}", config.final_snark, ret);
    }

    // Now wait for make to finish before proceeding.
    witness_tracker.await_all()?;

    // Export verification key (snarkjs.zKey.exportVerificationKey) via Node.js.
    tracing::info!("Exporting verification key...");
    run_snarkjs_export_vk(zkey_final.to_str().unwrap(), final_dir.join("final.verkey.json").to_str().unwrap())?;

    // Export Solidity verifier (snarkjs.zKey.exportSolidityVerifier) via Node.js.
    tracing::info!("Exporting Solidity verifier...");
    let snark_verifier_sol = if fflonk { "FflonkVerifier.sol" } else { "PlonkVerifier.sol" };
    run_snarkjs_export_solidity(
        zkey_final.to_str().unwrap(),
        final_dir.join(snark_verifier_sol).to_str().unwrap(),
        &config.final_snark.to_string(),
    )?;

    let verifier = if fflonk { SnarkVerifier::Fflonk } else { SnarkVerifier::Plonk };
    write_project_verifier(config, final_dir, const_root, &verifier)
}

/// The pilfflonk key of the final circuit, in `final_dir`, `provingKeySnark/final/`:
///
/// ```text
/// provingKeySnark/final/
/// ├── final.so, final.dat    the circuit's witness calculator, as for PLONK and FFLONK
/// ├── final.exec             plonk2pil's BN254 exec (version 3): the AIR's stage-1 columns out of
/// │                          the circuit's witness
/// ├── provingKey/            what `setup-pilfflonk -b provingKeySnark/final --solidity` writes
/// │   │                      (pilfflonk/docs/formats.md#provingkey)
/// │   ├── pilout.globalInfo.json, pilout.globalConstraints.json
/// │   └── final/             the pilout's name
/// │       ├── pilfflonk/     pilfflonk.srs.bin, pilfflonk.vkey.json, pilfflonk.verifier.sol
/// │       └── Wrap/airs/Wrap/air/
/// │                          Wrap.{const, pilfflonkinfo.json, expressionsinfo.json,
/// │                                verifierinfo.json, bin, verkey.json}
/// ├── <Name>Verifier.sol     the project's verifier, which imports pilfflonk.verifier.sol from
/// │                          provingKey/ and extends its PilfflonkVerifier
/// └── I<Name>Verifier.sol    its interface, as for PLONK and FFLONK
/// ```
///
/// `final.so`, `final.dat` and `final.exec` are the files `pilfflonk-wrap-witness` computes the
/// wrap's witness from, as `WrapArtifacts::with_stem` names them for the stem
/// `provingKeySnark/final/final`. The AIR's PIL and pilout stay beside the recursivef's, in
/// `pil/final.pil` and `build/final.pilout` ([`set_up_wrap_air`]). The witness library is built
/// while the key is, and `--powers-of-tau` is required, as for PLONK and FFLONK; the setup refuses
/// one with fewer powers than the layout needs.
fn gen_pilfflonk_key(
    config: &SnarkSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
    build_path: &Path,
    pil_dir: &Path,
    final_dir: &Path,
    const_root: &[u64; 4],
) -> Result<()> {
    let powers_of_tau = required_powers_of_tau(config)?;
    run_final_witness_library_generation(config, witness_tracker, final_dir);

    let files = WrapAirFiles {
        pil: pil_dir.join("final.pil"),
        pil_config: build_path.join("final.bn254.json"),
        pilout: build_path.join("final.pilout"),
        exec: final_dir.join("final.exec"),
    };
    let includes = [config.recurser_pil_path.to_string(), config.std_pil_path.to_string()];
    let key = set_up_wrap_air(&build_path.join("final.r1cs"), &files, &includes, Path::new(powers_of_tau), final_dir)?;
    let verifier = key.snark_verifier()?;

    witness_tracker.await_all()?;
    write_project_verifier(config, final_dir, const_root, &verifier)
}

/// The files of the pilfflonk wrap of a circuit ([`set_up_wrap_air`]).
struct WrapAirFiles {
    /// plonk2pil's PIL of the AIR.
    pil: PathBuf,
    /// The pil2com configuration that compiles it over BN254, `{"prime": r}`.
    pil_config: PathBuf,
    /// The PIL's pilout. Its stem is the pilout's name, which names the key's directory in
    /// `provingKey/`.
    pilout: PathBuf,
    /// plonk2pil's exec of the AIR, over BN254.
    exec: PathBuf,
}

/// The pilfflonk key [`set_up_wrap_air`] writes: its globalInfo, which names its files, and its
/// vkey.
struct WrapKey {
    global_info: PilfflonkGlobalInfo,
    vkey: Vkey,
}

impl WrapKey {
    /// The key's Solidity verifier, as the project's verifier beside `provingKey/` extends it:
    /// `pilfflonk.verifier.sol` by its path from there, and the words of the proof its `verifyProof`
    /// takes (pilfflonk/docs/formats.md#calldata). The project's verifier hands it one public, the
    /// final circuit's publics hash, and a vkey of any other number is refused.
    fn snark_verifier(&self) -> Result<SnarkVerifier> {
        ensure!(
            self.vkey.n_public == 1,
            "the pilfflonk vkey has {} publics, and the final circuit has one, its publics hash",
            self.vkey.n_public
        );
        let source = self.global_info.backend_dir(Path::new(PROVING_KEY_DIR)).join(VERIFIER_SOL_FILE);
        Ok(SnarkVerifier::Pilfflonk {
            source: format!("./{}", source.display()),
            words: CalldataLayout::of(&self.vkey).words(),
        })
    }
}

/// Sets up the AIR plonk2pil makes of the circuit `r1cs`, over BN254, for pilfflonk, writing
/// `files` and `provingKey/` under `key_dir`:
/// 1. plonk2pil lays out the r1cs in the final SNARK wrap's family ([`BN254_WRAP_FAMILY`],
///    PoseidonBN254 in layout L1), and its PIL and exec are written;
/// 2. pil2com compiles the PIL over BN254, with `includes` (plonk2pil's PIL and the std): `-P` with
///    a `prime` of r, which only a pil2com that honours `prime` does (`PIL2C_EXEC`,
///    pilfflonk/docs/README.md#compile-pil). One that ignores it compiles over Goldilocks, and the
///    setup refuses the pilout, saying so;
/// 3. setup-pilfflonk sets the pilout up with plonk2pil's fixed columns, which the pilout declares
///    `#pragma fixed_external`, and writes its Solidity verifier. It refuses a ptau with fewer
///    powers `[τ^i]₁` than the layout's largest degree, `12·N + 11` in L1, saying how many it holds
///    and how many it needs, before it writes any file of the key.
fn set_up_wrap_air(
    r1cs: &Path,
    files: &WrapAirFiles,
    includes: &[String],
    powers_of_tau: &Path,
    key_dir: &Path,
) -> Result<WrapKey> {
    tracing::info!("plonk2pil: the {BN254_WRAP_FAMILY} wrap of {}...", r1cs.display());
    let r1cs_data = fs::read(r1cs).with_context(|| format!("Failed to read {}", r1cs.display()))?;
    let options = PlonkOptions { hash_id: BN254_WRAP_FAMILY.into(), ..Default::default() };
    let PlonkResult::<Bn254> { exec, pil_str, fixed_pols, .. } = plonk2pil::plonk2pil(&r1cs_data, "wrap", &options)
        .with_context(|| format!("plonk2pil failed for {}", r1cs.display()))?;
    drop(r1cs_data);
    fs::write(&files.pil, pil_str)?;
    write_exec(&files.exec, &exec)?;
    drop(exec);

    tracing::info!("Compiling {} over BN254...", files.pil.display());
    fs::write(&files.pil_config, serde_json::json!({ "prime": BN254_R }).to_string())?;
    run_compile_pil(&CompilePilOptions {
        pil_path: files.pil.to_string_lossy().into_owned(),
        output_path: files.pilout.to_string_lossy().into_owned(),
        include_paths: includes.to_vec(),
        fixed_dir: None,
        config: Some(files.pil_config.to_string_lossy().into_owned()),
        fixed_to_file: false,
        no_proto_fixed_data: false,
    })?;

    tracing::info!("Running the pilfflonk setup of {}...", files.pilout.display());
    let setup = SetupPilfflonkOptions {
        airout_path: files.pilout.clone(),
        build_dir: key_dir.to_path_buf(),
        powers_of_tau: powers_of_tau.to_path_buf(),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: true,
    };
    let external = fixed_pols
        .into_iter()
        .map(|p| ExternalFixedColumn {
            name: p.name,
            index: p.index,
            values: p.values.into_iter().map(FrBytes::from).collect(),
        })
        .collect();
    run_setup_pilfflonk_with_external_fixed(&setup, external)?;

    let proving_key = key_dir.join(PROVING_KEY_DIR);
    let global_info = PilfflonkGlobalInfo::from_proving_key(&proving_key)?;
    let vkey = Vkey::read(&global_info.vkey_path(&proving_key))?;
    Ok(WrapKey { global_info, vkey })
}

/// The `--powers-of-tau` of the final SNARK's setup, which every protocol requires, if there is a
/// file there.
fn required_powers_of_tau<'a>(config: &SnarkSetupConfig<'a>) -> Result<&'a str> {
    let powers_of_tau =
        config.powers_of_tau.ok_or_else(|| anyhow::anyhow!("--powers-of-tau is required for final SNARK setup"))?;
    if !Path::new(powers_of_tau).exists() {
        bail!("powers-of-tau file not found: {}", powers_of_tau);
    }
    Ok(powers_of_tau)
}

/// Writes plonk2pil's exec buffer at `path`, its words little-endian
/// (`proofman_common::exec_format`).
fn write_exec(path: &Path, words: &[u64]) -> Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(path, bytes).with_context(|| format!("Failed to write {}", path.display()))
}

/// The project's Solidity verifier, `<Name>Verifier.sol`, which extends `verifier`, and its
/// interface, `I<Name>Verifier.sol`, in `final_dir` (pure Rust — no Node.js required).
fn write_project_verifier(
    config: &SnarkSetupConfig<'_>,
    final_dir: &Path,
    const_root: &[u64; 4],
    verifier: &SnarkVerifier,
) -> Result<()> {
    tracing::info!("Generating {} Solidity verifier...", config.name);
    let publics_ref = config.publics_info.as_ref();
    let camel = {
        let mut c = config.name.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().to_string() + c.as_str(),
        }
    };
    let sol = gen_solidity(config.name, const_root, publics_ref, verifier);
    let isol = gen_iverifier(config.name, publics_ref);
    fs::write(final_dir.join(format!("{camel}Verifier.sol")), sol)?;
    fs::write(final_dir.join(format!("I{camel}Verifier.sol")), isol)?;
    Ok(())
}

/// Launch the build of the final circuit's witness library, `final.so`, in the background. The
/// circuit is BN128-based and requires fr.cpp/fr.asm: it takes the dedicated final_snark_circom
/// helpers dir, not the goldilocks circom one.
fn run_final_witness_library_generation(
    config: &SnarkSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
    final_dir: &Path,
) {
    witness_tracker.run_witness_library_generation(
        config.build_dir,
        final_dir.to_str().unwrap_or(""),
        "final",
        "final",
        config.final_snark_circom_helpers_dir,
    );
}

/// Find the snarkjs package root, checking (in order):
///   1. `SNARKJS_PATH` environment variable
///   2. `node_modules/snarkjs` relative to cwd
///   3. Walk up from the executable's location to find `node_modules/snarkjs`
///
/// Install via: `npm install`  (reads package.json in the setup crate)
fn resolve_snarkjs_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SNARKJS_PATH") {
        let pb = PathBuf::from(&p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    let local = PathBuf::from("node_modules/snarkjs");
    if local.is_dir() {
        return local.canonicalize().ok();
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("node_modules/snarkjs");
            if candidate.is_dir() {
                return candidate.canonicalize().ok();
            }
            dir = d.parent();
        }
    }
    None
}

/// [`resolve_snarkjs_root`], but self-bootstrapping: when snarkjs is missing,
/// install the Node deps (see [`crate::proving_key::node_deps`]) and look again.
fn ensure_snarkjs_root() -> Option<PathBuf> {
    if let Some(root) = resolve_snarkjs_root() {
        return Some(root);
    }
    let root = crate::proving_key::node_deps::ensure_node_deps("snarkjs")?;
    root.join("node_modules/snarkjs").canonicalize().ok()
}

/// Make a path absolute against the current working directory. Required before
/// passing paths into the inline node scripts below — those run with cwd set to
/// the snarkjs package's parent dir (so `require('snarkjs')` resolves), which
/// is generally NOT the cwd from which cargo-zisk was invoked, so any relative
/// build_dir like `build2/...` would be looked up in the wrong place.
fn absolutize(p: &str) -> Result<String> {
    let pb = std::path::PathBuf::from(p);
    let abs = if pb.is_absolute() { pb } else { std::env::current_dir()?.join(pb) };
    Ok(abs.to_string_lossy().into_owned())
}

/// Export snarkjs verification key by spawning a small Node.js inline script.
fn run_snarkjs_export_vk(zkey_path: &str, output_path: &str) -> Result<()> {
    let snarkjs_root = ensure_snarkjs_root()
        .context("Cannot find snarkjs and automatic `npm install` did not produce it. Install Node.js/npm")?;
    let cwd = snarkjs_root.parent().unwrap_or(&snarkjs_root).to_path_buf();
    let zkey_abs = absolutize(zkey_path)?;
    let out_abs = absolutize(output_path)?;
    let script = format!(
        r#"
const snarkjs = require('snarkjs');
const fs = require('fs');
(async () => {{
    const vk = await snarkjs.zKey.exportVerificationKey({zkey:?});
    fs.writeFileSync({out:?}, JSON.stringify(vk));
}})().then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"#,
        zkey = zkey_abs,
        out = out_abs,
    );
    run_node_inline(&script, "snarkjs exportVerificationKey", &cwd)
}

/// Export snarkjs Solidity verifier by spawning a small Node.js inline script.
fn run_snarkjs_export_solidity(zkey_path: &str, output_path: &str, snark_type: &str) -> Result<()> {
    let snarkjs_root = ensure_snarkjs_root()
        .context("Cannot find snarkjs and automatic `npm install` did not produce it. Install Node.js/npm")?;
    let cwd = snarkjs_root.parent().unwrap_or(&snarkjs_root).to_path_buf();
    let zkey_abs = absolutize(zkey_path)?;
    let out_abs = absolutize(output_path)?;
    let template_key = snark_type;
    let script = format!(
        r#"
const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');
(async () => {{
    // require.resolve('snarkjs') → .../snarkjs/build/main.cjs; go up one level
    // past 'build/' to reach the package root where templates/ lives.
    // Neither './templates/...' nor './package.json' are in the exports map so
    // require.resolve shortcuts are unavailable.
    const snarkjsRoot = path.resolve(path.dirname(require.resolve('snarkjs')), '..');
    const tmplPath = path.join(snarkjsRoot, 'templates', 'verifier_{snark_type}.sol.ejs');
    const tmpl = {{ {template_key}: fs.readFileSync(tmplPath, 'utf8') }};
    const sol = await snarkjs.zKey.exportSolidityVerifier({zkey:?}, tmpl);
    fs.writeFileSync({out:?}, sol);
}})().then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"#,
        snark_type = snark_type,
        template_key = template_key,
        zkey = zkey_abs,
        out = out_abs,
    );
    run_node_inline(&script, "snarkjs exportSolidityVerifier", &cwd)
}

/// Run a Node.js inline script (`node -e "..."`), inheriting stdio.
/// `cwd` is the working directory for the node process — must be a directory
/// that contains a `node_modules/snarkjs` (or its parent) so that
/// `require('snarkjs')` resolves correctly.
fn run_node_inline(script: &str, context: &str, cwd: &std::path::Path) -> Result<()> {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(script)
        .current_dir(cwd)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .output()
        .with_context(|| format!("Failed to spawn node for {}", context))?;
    if !out.status.success() {
        bail!("{} failed (exit {})", context, out.status.code().unwrap_or(-1));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::process::Command;

    use pilfflonk_setup::layout::max_degree;
    use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau, write_tau_one_ptau};
    use proofman_common::exec_format::{ExecFile, EXEC_FORMAT_VERSION_WIDE};
    use proofman_pilfflonk::{AirFile, PilfflonkInfo, WitnessShape};

    use super::*;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The exec at `exec` is over BN254 (version 3), has no gate band, which the wrap's witness
    /// refuses, and gathers exactly the stage-1 columns of the AIR of the key at `proving_key`, in
    /// its rows.
    pub(crate) fn assert_exec_gathers_the_air_columns(exec: &Path, proving_key: &Path) {
        let exec = ExecFile::<Bn254>::read(exec).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(exec.layout.version(), EXEC_FORMAT_VERSION_WIDE);
        assert_eq!(exec.layout.coef_words(), 4);
        assert!(exec.bands.is_empty(), "{} gate bands", exec.bands.len());
        let global_info = PilfflonkGlobalInfo::from_proving_key(proving_key).unwrap();
        let info_path = global_info.air_file(proving_key, 0, 0, AirFile::PilfflonkInfo).unwrap();
        let info = PilfflonkInfo::read(&info_path).unwrap();
        let shape = WitnessShape::from_proving_key(&global_info, &[&info]).unwrap();
        let air = shape.airs()[0];
        assert_eq!(exec.layout.map_cols(), air.n_cols, "the exec's columns and the AIR's stage-1 ones");
        assert!(exec.layout.map_rows() <= 1 << air.n_bits, "{} rows of 2^{}", exec.layout.map_rows(), air.n_bits);
    }

    /// M49's end-to-end circuit, `setup/stark-recurser/tests/fixtures/bn254/wrap.circom`: a
    /// `PoseidonT(5)` use among PLONK gates, and one public. Its r1cs, compiled for BN254 into
    /// `dir` with the committed circom.
    fn small_wrap_r1cs(dir: &Path) -> PathBuf {
        let root = repo_root();
        let circom = root.join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" });
        let out = Command::new(&circom)
            .args(["--O1", "--r1cs", "--prime", "bn128", "-l"])
            .arg(root.join("setup/stark-recurser/stark2circom/circom_verifier/circuits.bn128"))
            .arg(root.join("setup/stark-recurser/tests/fixtures/bn254/wrap.circom"))
            .arg("-o")
            .arg(dir)
            .output()
            .unwrap_or_else(|e| panic!("run {}: {e}", circom.display()));
        assert!(out.status.success(), "circom failed:\n{}", String::from_utf8_lossy(&out.stderr));
        dir.join("wrap.r1cs")
    }

    /// The pilfflonk wrap of a small circuit: a ptau with fewer powers than the layout's largest
    /// degree, `12·N + 11` in L1, is refused, saying how many it holds and how many it needs; with
    /// enough, the key is set up, with its vkey, its Solidity verifier and the exec of its AIR.
    ///
    /// Needs `PIL2C_EXEC`, a pil2com that honours `prime` (pilfflonk/docs/README.md#compile-pil);
    /// without it the test says so and passes.
    #[test]
    fn the_wrap_air_refuses_a_ptau_too_small_and_is_set_up_with_enough() {
        if std::env::var_os("PIL2C_EXEC").is_none() {
            eprintln!("skipped: PIL2C_EXEC does not name a pil2com that honours `prime`");
            return;
        }
        let dir = std::env::temp_dir().join(format!("snark_setup_wrap_air_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let r1cs = small_wrap_r1cs(&dir);
        let files = WrapAirFiles {
            pil: dir.join("wrap.pil"),
            pil_config: dir.join("bn254.json"),
            pilout: dir.join("wrap.pilout"),
            exec: dir.join("wrap.exec"),
        };
        let root = repo_root();
        let includes = ["setup/stark-recurser/plonk2pil/pil", "pil2-components/lib/std/pil"]
            .map(|include| root.join(include).to_string_lossy().into_owned());

        let small = dir.join("small.ptau");
        write_tau_one_ptau(&small, 64).unwrap();
        let refused = set_up_wrap_air(&r1cs, &files, &includes, &small, &dir.join("refused")).err().expect("refused");
        let refused = format!("{refused:#}");
        assert!(refused.contains("holds 64 powers [τ^i]₁, fewer than the "), "{refused}");

        let ptau = dir.join("fixed_tau.ptau");
        write_fixed_tau_ptau(&ptau, 8192, &test_tau()).unwrap();
        let key_dir = dir.join("key");
        let key = set_up_wrap_air(&r1cs, &files, &includes, &ptau, &key_dir).unwrap_or_else(|e| panic!("{e:#}"));
        let needed = max_degree(&key.vkey.layout);
        assert_eq!(needed, 12 * (1 << key.vkey.power) + 11, "the largest degree of layout L1");
        assert!(refused.contains(&format!("fewer than the {needed} requested")), "{refused}");

        assert_exec_gathers_the_air_columns(&files.exec, &key_dir.join(PROVING_KEY_DIR));
        assert_eq!(key.vkey.n_public, 1);
        let source = "./provingKey/wrap/pilfflonk/pilfflonk.verifier.sol";
        let words = CalldataLayout::of(&key.vkey).words();
        assert_eq!(key.snark_verifier().unwrap(), SnarkVerifier::Pilfflonk { source: source.into(), words });
        assert!(key_dir.join(source).is_file(), "{source} missing from {}", key_dir.display());
        fs::remove_dir_all(&dir).unwrap();
    }
}
