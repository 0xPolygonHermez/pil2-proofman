//! Port of `generateFinalSetup.js`: generate the vadcop_final circuit,
//! compile it, and produce the proving key artifacts.
//!
//! The final setup:
//! 1. Reads all recursive2 starkInfo/verifierInfo/vks for each airgroup
//! 2. Generates the final circom via gencircom with the "final" template
//! 3. Compiles circom to R1CS + C++
//! 4. Converts R1CS to PIL
//! 5. Compiles PIL
//! 6. Runs starkSetup with specific final settings
//! 7. Computes constant tree and writes all artifacts
//! 8. Writes verifier.rs for the final circuit

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::io::recurser::{gen_circom, pil2circom, GenCircomInput, GenCircomOptions, Pil2CircomOptions};
use pilout::pilout_proxy::PilOutProxy;
use stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use stark_recurser::plonk2pil::{self, PlonkResult};

use crate::proving_key::bctree;
use crate::io::fixed_cols;
use crate::proving_key::recursive::compile_pil;
use crate::output::witness_gen::WitnessTracker;

/// Configuration for the final setup.
pub struct FinalSetupConfig<'a> {
    pub build_dir: &'a str,
    pub hash: &'a str,
    pub global_info: &'a Value,
    pub global_constraints: &'a Value,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
}

/// Result of the final setup.
pub struct FinalSetupResult {
    pub stark_info: Value,
    pub verifier_info: Value,
    pub const_root: [u64; 4],
}

/// Run the final setup.
///
/// Ports `genFinalSetup()` from `generateFinalSetup.js`.
/// Parse a JSON value that may be either a string or a u64 number into a String.
/// Used when reading vks.json fields that are stored as u64 integers.
fn json_val_to_str(v: &Value) -> String {
    v.as_str().map(|s| s.to_string()).or_else(|| v.as_u64().map(|n| n.to_string())).unwrap_or_else(|| "0".to_string())
}

pub fn gen_final_setup(config: &FinalSetupConfig<'_>, witness_tracker: &WitnessTracker) -> Result<FinalSetupResult> {
    let build_dir = PathBuf::from(config.build_dir);
    let global_name = config.global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout");

    let _agg_types = config.global_info.get("aggTypes").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);

    let air_groups: Vec<String> = config
        .global_info
        .get("air_groups")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or("unnamed").to_string()).collect())
        .unwrap_or_default();

    let mut stark_infos = Vec::new();
    let mut verifier_infos = Vec::new();
    let mut agg_keys_recursive2 = Vec::new();
    let mut basic_keys_recursive1 = Vec::new();
    let mut verifier_names = Vec::new();

    // Read recursive2 artifacts for each airgroup
    for ag_name in air_groups.iter() {
        let r2_dir = build_dir.join("provingKey").join(global_name).join(ag_name).join("recursive2");

        let si_path = r2_dir.join("recursive2.starkinfo.json");
        let vi_path = r2_dir.join("recursive2.verifierinfo.json");
        let vks_path = r2_dir.join("recursive2.vks.json");

        if si_path.exists() && vi_path.exists() && vks_path.exists() {
            let si: Value = serde_json::from_str(&fs::read_to_string(&si_path)?)?;
            let vi: Value = serde_json::from_str(&fs::read_to_string(&vi_path)?)?;
            let vks: Value = serde_json::from_str(&fs::read_to_string(&vks_path)?)?;

            stark_infos.push(si);
            verifier_infos.push(vi);

            if let Some(root) = vks.get("rootCRecursive2") {
                agg_keys_recursive2.push(
                    root.as_array().map(|a| a.iter().map(json_val_to_str).collect::<Vec<_>>()).unwrap_or_default(),
                );
            } else {
                agg_keys_recursive2.push(vec![]);
            }

            if let Some(keys) = vks.get("rootCRecursives1") {
                // rootCRecursives1 = [[v0,v1,v2,v3], [v0,...], ...] — flat 2-D
                basic_keys_recursive1.push(
                    keys.as_array()
                        .map(|airs| {
                            airs.iter()
                                .map(|air_vk| {
                                    air_vk
                                        .as_array()
                                        .map(|vals| vals.iter().map(json_val_to_str).collect::<Vec<_>>())
                                        .unwrap_or_default()
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default(),
                );
            } else {
                basic_keys_recursive1.push(vec![]);
            }
        } else {
            bail!(
                "Recursive2 artifacts not found for airgroup '{}'. \
                 Run recursive setup first.",
                ag_name
            );
        }

        verifier_names.push(format!("{}_recursive2.verifier.circom", ag_name));
    }

    let files_dir = build_dir.join("provingKey").join(global_name).join("vadcop_final");
    fs::create_dir_all(&files_dir)?;

    let circom_dir = build_dir.join("circom");
    let build_path = build_dir.join("build");
    let pil_dir = build_dir.join("pil");
    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_path)?;
    fs::create_dir_all(&pil_dir)?;

    // Generate verifier circom for each airgroup's recursive2 output.
    // This must use the recursive2 starkinfo (not reuse the verifier from the
    // recursive2 step, which was generated from the recursive1 starkinfo).
    for (i, (si, vi)) in stark_infos.iter().zip(verifier_infos.iter()).enumerate() {
        let const_root = if i < agg_keys_recursive2.len() && agg_keys_recursive2[i].len() == 4 {
            [
                agg_keys_recursive2[i][0].clone(),
                agg_keys_recursive2[i][1].clone(),
                agg_keys_recursive2[i][2].clone(),
                agg_keys_recursive2[i][3].clone(),
            ]
        } else {
            ["0".into(), "0".into(), "0".into(), "0".into()]
        };

        let pil2circom_opts = Pil2CircomOptions {
            skip_main: true,
            verkey_input: true,
            enable_input: true,
            hash: config.hash.to_string(),
            ..Default::default()
        };

        let verifier_circom = pil2circom(&const_root, si, vi, &pil2circom_opts)
            .context("pil2circom failed generating recursive2 verifier for final setup")?;

        let verifier_path = circom_dir.join(&verifier_names[i]);
        fs::write(&verifier_path, &verifier_circom)?;
    }

    // Build global info with constraints for the template
    let mut final_global_info = config.global_info.clone();
    if let Some(constraints) = config.global_constraints.get("constraints") {
        final_global_info.as_object_mut().map(|obj| obj.insert("globalConstraints".to_string(), constraints.clone()));
    }

    // Generate final circom
    let gen_circom_opts = GenCircomOptions { is_final: true, ..Default::default() };

    let gen_input = GenCircomInput {
        template_name: "src/vadcop/templates/final.circom.ejs",
        stark_infos: &stark_infos,
        vadcop_info: &final_global_info,
        verifier_filenames: &verifier_names,
        basic_verification_keys: &basic_keys_recursive1,
        agg_verification_keys: &agg_keys_recursive2,
        publics: &[],
        options: &gen_circom_opts,
    };

    let final_circom = gen_circom(&gen_input).context("gen_circom failed in final setup")?;

    let final_circom_path = circom_dir.join("vadcop_final.circom");
    fs::write(&final_circom_path, &final_circom)?;

    // Compile circom
    tracing::info!("Compiling vadcop_final...");
    let compile_output = std::process::Command::new(config.circom_exec)
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
        .arg(final_circom_path.to_str().unwrap())
        .arg("-o")
        .arg(build_path.to_str().unwrap())
        .output()
        .context("Failed to execute circom for final setup")?;

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr);
        bail!("Circom compilation failed for vadcop_final: {}", stderr);
    }

    // Copy .dat file
    tracing::info!("Copying circom files...");
    let dat_src = build_path.join("vadcop_final_cpp").join("vadcop_final.dat");
    let dat_dst = files_dir.join("vadcop_final.dat");
    if dat_src.exists() {
        fs::copy(&dat_src, &dat_dst)?;
    }

    // Generate witness library
    witness_tracker.run_witness_library_generation(
        config.build_dir,
        files_dir.to_str().unwrap_or(""),
        "vadcop_final",
        "vadcop_final",
        config.circom_helpers_dir,
    );

    // plonk2pil
    let r1cs_path = build_path.join("vadcop_final.r1cs");
    let r1cs_data = fs::read(&r1cs_path).with_context(|| format!("Failed to read R1CS: {}", r1cs_path.display()))?;

    let plonk_opts = PlonkOptions {
        airgroup_name: Some("FinalVadcop".to_string()),
        max_constraint_degree: None,
        hash_id: config.hash.to_string(),
        merge_copies: true,
    };
    let plonk_result: PlonkResult =
        plonk2pil::plonk2pil(&r1cs_data, "aggregation", &plonk_opts).context("plonk2pil failed in final setup")?;

    // Write fixed pols binary
    let fixed_bin_path = build_path.join("vadcop_final.fixed.bin");
    let fixed_info: Vec<(String, Vec<u32>, Vec<u64>)> =
        plonk_result.fixed_pols.iter().map(|fp| (fp.name.clone(), vec![fp.index as u32], fp.values.clone())).collect();
    fixed_cols::write_fixed_pols_bin(
        fixed_bin_path.to_str().unwrap(),
        &plonk_result.airgroup_name,
        &plonk_result.air_name,
        1u64 << plonk_result.n_bits,
        &fixed_info,
    )?;

    // Write PIL
    let pil_path = pil_dir.join("vadcop_final.pil");
    fs::write(&pil_path, &plonk_result.pil_str)?;

    // Compile PIL
    let pilout_path = build_path.join("vadcop_final.pilout");
    compile_pil(
        pil_path.to_str().unwrap(),
        pilout_path.to_str().unwrap(),
        config.std_pil_path,
        config.recurser_pil_path,
    )?;

    // Write exec
    let exec_path = files_dir.join("vadcop_final.exec");
    let exec_bytes: Vec<u8> = plonk_result.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&exec_path, &exec_bytes)?;

    // Const file writing is deferred until after pil_info, which determines
    // the true nConstants (may be larger than plonk2pil's fixedPols count).
    let const_path = files_dir.join("vadcop_final.const");
    let plonk_n_rows = 1usize << plonk_result.n_bits;
    let plonk_n_fixed = plonk_result.fixed_pols.len();

    // Final stark struct settings
    let final_settings = crate::types::stark_struct::StarkSettings {
        blowup_factor: Some(4),
        folding_factor: Some(4),
        pow_bits: Some(22),
        last_level_verification: Some(2),
        ..Default::default()
    };
    let final_stark_struct = crate::types::stark_struct::generate_stark_struct(&final_settings, plonk_result.n_bits);

    // Run real starkSetup via pil_info on the compiled vadcop_final pilout
    let starkinfo_path = files_dir.join("vadcop_final.starkinfo.json");

    let pilout_file_str = pilout_path.to_str().unwrap_or("");
    if !Path::new(pilout_file_str).exists() {
        bail!("vadcop_final pilout not found at {}", pilout_file_str);
    }

    let proxy = PilOutProxy::new(pilout_file_str).map_err(|e| anyhow::anyhow!("Failed to load pilout: {}", e))?;
    let pilout = &proxy.pilout;
    if pilout.air_groups.is_empty() || pilout.air_groups[0].airs.is_empty() {
        bail!("vadcop_final pilout has no AIR groups");
    }

    let pil_info_result = crate::pil::info::pil_info(pilout, 0, 0, &final_stark_struct, &Default::default());

    // Build JSON representations using the same helpers as the non-recursive path
    let opening_points = crate::output::stark_info::collect_opening_points(&pil_info_result.setup);
    let ev_map_len = pil_info_result.pil_code.ev_map.len();
    let fri = crate::output::stark_info::build_fri(&final_stark_struct, ev_map_len.max(1) as u64);

    let starkinfo_output = crate::output::stark_info::build_starkinfo_output(
        &pil_info_result.setup,
        &final_stark_struct,
        &pil_info_result.pil_code,
        &opening_points,
        &fri,
        0,
        0,
        "vadcop_final",
        pil_info_result.c_exp_id,
        pil_info_result.fri_exp_id,
        pil_info_result.q_deg,
    );
    let verifier_info_ref = &pil_info_result.pil_code.verifier_info;
    let expressions_info_ref = &pil_info_result.pil_code.expressions_info;

    // Write all JSON files
    fs::write(&starkinfo_path, crate::output::json::to_json_string(&starkinfo_output)?)?;
    fs::write(
        files_dir.join("vadcop_final.expressionsinfo.json"),
        crate::output::json::to_json_string(expressions_info_ref)?,
    )?;
    fs::write(
        files_dir.join("vadcop_final.verifierinfo.json"),
        crate::output::json::to_json_string(verifier_info_ref)?,
    )?;

    // Write binary files — convert in-memory structs directly (no disk round-trip)
    {
        let si_val: serde_json::Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
        let stark_info_loaded = crate::types::stark_info::StarkInfo::from_json(&si_val)?;

        let expressions_loaded = crate::types::stark_info::ExpressionsInfo::from(expressions_info_ref);
        crate::io::bin_file::write_expressions_bin_file(
            files_dir.join("vadcop_final.bin").to_str().unwrap(),
            &stark_info_loaded,
            &expressions_loaded,
        )?;

        let verifier_loaded = crate::types::stark_info::VerifierInfo::from(verifier_info_ref);
        crate::io::bin_file::write_verifier_expressions_bin_file(
            files_dir.join("vadcop_final.verifier.bin").to_str().unwrap(),
            &stark_info_loaded,
            &verifier_loaded,
        )?;

        // Write verifier Rust file
        crate::output::verifier::write_verifier_rust_file(
            files_dir.join("vadcop_final.verifier.rs").to_str().unwrap(),
            &stark_info_loaded,
            &verifier_loaded,
            true,
            config.hash,
        )?;

        // Write const file: pilout inline values are the selector polynomials;
        // plonk2pil polynomials fill the columns with empty pilout values.
        {
            let air = &pilout.air_groups[0].airs[0];
            let plonk_values =
                fixed_cols::reorder_plonk_pols_for_pilout(&plonk_result.fixed_pols, &pilout.symbols, 0, 0);
            fixed_cols::write_const_file(const_path.to_str().unwrap(), air, &plonk_values)?;
            tracing::info!(
                "Wrote vadcop_final const file: {} cols ({} from plonk + {} from pilout), {} rows",
                air.fixed_cols.len(),
                plonk_n_fixed,
                air.fixed_cols.len().saturating_sub(plonk_n_fixed),
                plonk_n_rows
            );
        }
    }

    // Compute constant tree
    tracing::info!("Computing Constant Tree for vadcop_final...");
    let verkey_json_path = files_dir.join("vadcop_final.verkey.json");
    let const_root = if const_path.exists() {
        let root = bctree::compute_const_tree(
            const_path.to_str().unwrap(),
            starkinfo_path.to_str().unwrap(),
            verkey_json_path.to_str().unwrap(),
        );

        let mut verkey_bin = Vec::with_capacity(32);
        for &val in root.iter() {
            verkey_bin.extend_from_slice(&val.to_le_bytes());
        }
        fs::write(files_dir.join("vadcop_final.verkey.bin"), &verkey_bin)?;

        root
    } else {
        tracing::warn!("Skipping const tree for vadcop_final: const file not found");
        [0u64; 4]
    };

    // Write verifier.rs
    tracing::info!("Final setup verifier.rs generation pending full starkSetup integration");

    // Wait for witness library
    witness_tracker.await_all()?;

    let result_stark_info = serde_json::to_value(&starkinfo_output)?;
    let result_verifier_info = serde_json::to_value(verifier_info_ref).unwrap_or(serde_json::Value::Null);

    Ok(FinalSetupResult { stark_info: result_stark_info, verifier_info: result_verifier_info, const_root })
}
