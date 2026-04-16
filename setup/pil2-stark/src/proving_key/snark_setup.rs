//! Final SNARK setup: recursivef (GL→BN128 bridge) + final (fflonk/plonk) steps.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use proofman_starks_lib_c::{generate_fflonk_zkey_c, generate_plonk_zkey_c};

use crate::io::recurser::{gen_circom, pil2circom, GenCircomInput, GenCircomOptions, Pil2CircomOptions};
use crate::proving_key::{bctree, recursive::compile_pil};
use crate::io::fixed_cols;
use crate::output::witness_gen::WitnessTracker;
use pilout::pilout_proxy::PilOutProxy;
use stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use stark_recurser::plonk2pil;
use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

/// Configuration for the final SNARK setup.
pub struct SnarkSetupConfig<'a> {
    /// Build directory (must contain provingKey/{name}/vadcop_final/).
    pub build_dir: &'a str,
    /// Circuit name (from globalInfo.name).
    pub name: &'a str,
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
    /// Powers-of-tau (.ptau) file for snarkjs final setup (required if !only_recursive_final).
    pub powers_of_tau: Option<&'a str>,
    /// "fflonk" or "plonk".
    pub final_snark: &'a str,
    /// Optional publics hash info JSON value.
    pub publics_info: Option<Value>,
    /// When true, only run the recursivef step and stop before the final SNARK.
    pub only_recursive_final: bool,
}

/// Run the final SNARK setup pipeline.
///
/// Phase 1 (recursivef): GL → BN128 bridge circuit.
/// Phase 2 (final):      BN128 R1CS → SNARK zkey + Solidity verifiers.
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
    let pil2circom_opts =
        Pil2CircomOptions { skip_main: true, verkey_input: true, enable_input: false, input_challenges: false };
    let verifier_circom_rf = pil2circom(&const_root_str, stark_info, verifier_info, &pil2circom_opts)
        .context("pil2circom failed for recursivef")?;
    fs::write(circom_dir.join(verifier_name_rf), &verifier_circom_rf)?;

    // gen_circom: generate recursivef.circom using the recursivef.circom.ejs template.
    // basic_vk = [[constRoot]] (one airgroup, one air with the vadcop_final constRoot)
    let gen_opts_rf =
        GenCircomOptions { airgroup_id: None, has_compressor: false, has_recursion: false, is_final: false };
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
            "--O1",
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
    let plonk_opts_rf = PlonkOptions { airgroup_name: Some("Recursivef".to_string()), max_constraint_degree: None };
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

    // Write exec buffer.
    let exec_rf = recursivef_dir.join("recursivef.exec");
    let exec_bytes_rf: Vec<u8> = plonk_rf.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&exec_rf, &exec_bytes_rf)?;

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

    // BN128 stark struct settings (matches JS: blowupFactor=6, powBits=17, arity=4).
    let bn128_settings = StarkSettings {
        verification_hash_type: Some("BN128".to_string()),
        blowup_factor: Some(6),
        merkle_tree_arity: Some(4),
        merkle_tree_custom: Some(false),
        last_level_verification: Some(0),
        pow_bits: Some(17),
        ..Default::default()
    };
    let stark_struct_rf = generate_stark_struct(&bn128_settings, n_bits_rf);

    let pil_result_rf = crate::pil::info::pil_info(pilout_inner, 0, 0, &stark_struct_rf, &Default::default());

    // Build starkinfo output.
    let opening_points_rf = crate::output::stark_info::collect_opening_points(&pil_result_rf.setup);
    let field_size = crate::types::security::goldilocks_cube_field_size();
    let ev_map_len_rf = pil_result_rf.pil_code.ev_map.len();
    let folding_factors_rf = crate::output::stark_info::compute_folding_factors(&stark_struct_rf);
    let fri_params_rf = crate::types::security::FRISecurityParams {
        field_size,
        dimension: 1u64 << stark_struct_rf.n_bits,
        rate: 1.0 / (1u64 << (stark_struct_rf.n_bits_ext - stark_struct_rf.n_bits)) as f64,
        n_opening_points: opening_points_rf.len() as u64,
        n_functions: ev_map_len_rf.max(1) as u64,
        folding_factors: folding_factors_rf,
        max_grinding_bits: stark_struct_rf.pow_bits as u64,
        use_max_grinding_bits: true,
        tree_arity: stark_struct_rf.merkle_tree_arity as u64,
        target_security_bits: 128,
    };
    let fri_security_rf = crate::types::security::get_optimal_fri_query_params("JBR", &fri_params_rf);

    let starkinfo_rf = crate::output::stark_info::build_starkinfo_output(
        &pil_result_rf.setup,
        &stark_struct_rf,
        &pil_result_rf.pil_code,
        &opening_points_rf,
        &fri_security_rf,
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
    );
    let mut verkey_bin_rf = Vec::with_capacity(32);
    for &v in rf_const_root.iter() {
        verkey_bin_rf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(recursivef_dir.join("recursivef.verkey.bin"), &verkey_bin_rf)?;

    // Write bin files.
    let si_val_rf: Value = serde_json::from_str(&starkinfo_rf_json)?;
    let si_loaded_rf = crate::types::stark_info::StarkInfo::from_json(&si_val_rf)?;
    let ei_rf = crate::types::stark_info::ExpressionsInfo::from(expressions_info_rf);
    crate::io::bin_file::write_expressions_bin_file(
        recursivef_dir.join("recursivef.bin").to_str().unwrap(),
        &si_loaded_rf,
        &ei_rf,
    )?;
    let vi_rf_loaded = crate::types::stark_info::VerifierInfo::from(verifier_info_rf);
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

    let rf_const_root_json: Value = serde_json::from_str(&fs::read_to_string(&verkey_rf_path)?)?;
    let rf_const_root_str: [String; 4] = {
        let arr = rf_const_root_json.as_array().ok_or_else(|| anyhow::anyhow!("recursivef verkey is not an array"))?;
        if arr.len() < 4 {
            bail!("recursivef verkey has fewer than 4 elements");
        }
        [
            arr[0].to_string().trim_matches('"').to_string(),
            arr[1].to_string().trim_matches('"').to_string(),
            arr[2].to_string().trim_matches('"').to_string(),
            arr[3].to_string().trim_matches('"').to_string(),
        ]
    };

    let starkinfo_rf_val: Value = serde_json::from_str(&starkinfo_rf_json)?;
    let verifierinfo_rf_val: Value =
        serde_json::from_str(&fs::read_to_string(recursivef_dir.join("recursivef.verifierinfo.json"))?)?;

    // pil2circom: generate recursivef.verifier.circom (verkeyInput=false for final).
    let verifier_name_final = "recursivef.verifier.circom";
    let pil2circom_opts_final =
        Pil2CircomOptions { skip_main: true, verkey_input: false, enable_input: false, input_challenges: false };
    let verifier_circom_final =
        pil2circom(&rf_const_root_str, &starkinfo_rf_val, &verifierinfo_rf_val, &pil2circom_opts_final)
            .context("pil2circom failed for final")?;
    fs::write(circom_dir.join(verifier_name_final), &verifier_circom_final)?;

    // gen_circom: generate final.circom using final.circom.ejs template.
    let publics_vec: Vec<Value> =
        if let Some(ref pi) = config.publics_info { vec![pi.clone()] } else { vec![Value::Null] };
    let gen_opts_final =
        GenCircomOptions { airgroup_id: None, has_compressor: false, has_recursion: false, is_final: true };
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

    // Run witness library generation and wait for it (final step must complete fully).
    witness_tracker.run_witness_library_generation(
        config.build_dir,
        final_dir.to_str().unwrap_or(""),
        "final",
        "final",
        config.circom_helpers_dir,
    );
    witness_tracker.await_all()?;

    // Run snarkjs fflonk/plonk setup.
    let powers_of_tau =
        config.powers_of_tau.ok_or_else(|| anyhow::anyhow!("--powers-of-tau is required for final SNARK setup"))?;

    let r1cs_final = build_path.join("final.r1cs");
    let zkey_final = final_dir.join("final.zkey");
    tracing::info!("Running {} setup via FFI...", config.final_snark);
    let ret = if config.final_snark == "fflonk" {
        generate_fflonk_zkey_c(r1cs_final.to_str().unwrap(), powers_of_tau, zkey_final.to_str().unwrap())
    } else {
        generate_plonk_zkey_c(r1cs_final.to_str().unwrap(), powers_of_tau, zkey_final.to_str().unwrap())
    };
    if ret != 0 {
        bail!("{} setup FFI call failed with return code {}", config.final_snark, ret);
    }

    // Export verification key (snarkjs.zKey.exportVerificationKey) via Node.js.
    tracing::info!("Exporting verification key...");
    run_snarkjs_export_vk(zkey_final.to_str().unwrap(), final_dir.join("final.verkey.json").to_str().unwrap())?;

    // Export Solidity verifier (snarkjs.zKey.exportSolidityVerifier) via Node.js.
    tracing::info!("Exporting Solidity verifier...");
    let snark_verifier_sol = if config.final_snark == "fflonk" { "FflonkVerifier.sol" } else { "PlonkVerifier.sol" };
    run_snarkjs_export_solidity(
        zkey_final.to_str().unwrap(),
        final_dir.join(snark_verifier_sol).to_str().unwrap(),
        config.final_snark,
    )?;

    // Generate project-specific Solidity verifier via genSolidity.
    tracing::info!("Generating {} Solidity verifier...", config.name);
    let publics_hash_for_solidity = config.publics_info.clone();
    run_gen_solidity(
        config.name,
        const_root,
        publics_hash_for_solidity.as_ref(),
        config.final_snark == "fflonk",
        final_dir.to_str().unwrap(),
    )?;

    // Write publics_info.json if provided.
    if let Some(ref pi) = config.publics_info {
        fs::write(snark_dir.join("publics_info.json"), serde_json::to_string_pretty(pi)?)?;
    }

    tracing::info!("Final SNARK setup complete");
    Ok(())
}

/// Export snarkjs verification key by spawning a small Node.js inline script.
fn run_snarkjs_export_vk(zkey_path: &str, output_path: &str) -> Result<()> {
    let script = format!(
        r#"
const snarkjs = require('snarkjs');
const fs = require('fs');
(async () => {{
    const vk = await snarkjs.zKey.exportVerificationKey({zkey:?});
    fs.writeFileSync({out:?}, JSON.stringify(vk));
}})().then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"#,
        zkey = zkey_path,
        out = output_path,
    );
    run_node_inline(&script, "snarkjs exportVerificationKey")
}

/// Export snarkjs Solidity verifier by spawning a small Node.js inline script.
fn run_snarkjs_export_solidity(zkey_path: &str, output_path: &str, snark_type: &str) -> Result<()> {
    let template_key = snark_type;
    let script = format!(
        r#"
const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');
(async () => {{
    const tmplPath = require.resolve(`snarkjs/templates/verifier_{snark_type}.sol.ejs`);
    const tmpl = {{ {template_key}: fs.readFileSync(tmplPath, 'utf8') }};
    const sol = await snarkjs.zKey.exportSolidityVerifier({zkey:?}, tmpl);
    fs.writeFileSync({out:?}, sol);
}})().then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"#,
        snark_type = snark_type,
        template_key = template_key,
        zkey = zkey_path,
        out = output_path,
    );
    run_node_inline(&script, "snarkjs exportSolidityVerifier")
}

/// Call `genSolidity` from stark-recurser to produce the project-specific Solidity verifier.
fn run_gen_solidity(
    name: &str,
    const_root: &[u64; 4],
    publics_hash: Option<&Value>,
    is_fflonk: bool,
    output_dir: &str,
) -> Result<()> {
    let publics_hash_json = match publics_hash {
        Some(v) => serde_json::to_string(v)?,
        None => "undefined".to_string(),
    };
    let const_root_json = serde_json::to_string(const_root)?;
    let script = format!(
        r#"
const {{ genSolidity }} = require('stark-recurser/src/gensolidity.js');
const fs = require('fs');
const path = require('path');
(async () => {{
    const name = {name:?};
    const constRoot = {const_root_json};
    const publicsHash = {publics_hash_json};
    const isFflonk = {is_fflonk};
    const {{ solidityVerifier, solidityVerifierInterface }} = await genSolidity(name, constRoot, publicsHash, isFflonk);
    const outDir = {output_dir:?};
    const camelName = name.charAt(0).toUpperCase() + name.slice(1);
    fs.writeFileSync(path.join(outDir, camelName + 'Verifier.sol'), solidityVerifier);
    fs.writeFileSync(path.join(outDir, 'I' + camelName + 'Verifier.sol'), solidityVerifierInterface);
}})().then(() => process.exit(0)).catch(e => {{ console.error(e); process.exit(1); }});
"#,
        name = name,
        const_root_json = const_root_json,
        publics_hash_json = publics_hash_json,
        is_fflonk = is_fflonk,
        output_dir = output_dir,
    );
    run_node_inline(&script, "genSolidity")
}

/// Run a Node.js inline script (`node -e "..."`), inheriting stdio.
fn run_node_inline(script: &str, context: &str) -> Result<()> {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(script)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .output()
        .with_context(|| format!("Failed to spawn node for {}", context))?;
    if !out.status.success() {
        bail!("{} failed (exit {})", context, out.status.code().unwrap_or(-1));
    }
    Ok(())
}
