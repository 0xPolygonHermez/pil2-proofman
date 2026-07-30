//! Port of `genRecursiveSetupTest` from `generateRecursiveSetup.js`.
//!
//! Unlike the production `gen_recursive_setup`, this function takes a raw circom
//! file as input (no pil2circom/gencircom steps) and is intended for the
//! `test-recursive` CI example and the `setup-recursive-test` subcommand.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use pilout::pilout_proxy::PilOutProxy;
use stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use stark_recurser::plonk2pil;

use crate::io::fixed_cols;
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::bctree;
use crate::proving_key::recursive::compile_pil;
use crate::commands::recursive_setup::resolve_path_env;
use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

/// Run the recursive test setup from a user-provided circom file.
///
/// Ports `genRecursiveSetupTest()` from `generateRecursiveSetup.js`.
///
/// Unlike the production `gen_recursive_setup`, this function:
/// - Takes a raw circom file directly (no pil2circom + gencircom steps).
/// - Always uses the name "Compressor" for all output files.
/// - Always uses `{blowupFactor: 3, lastLevelVerification: 1}` for the starkstruct.
/// - Writes globalInfo and globalConstraints at the end.
///
/// # Arguments
/// * `build_dir` - Build output directory (e.g. `./examples/test-recursive/build`).
/// * `circom_path` - Path to the circom source file.
/// * `circom_name` - Circuit name (e.g. "test").
/// * `std_pil_path` - Standard PIL library path (for compiling the generated PIL).
/// * `setup_type` - One of "compressor", "aggregation".
/// * `circom_exec` - Path to the circom binary.
/// * `circuits_gl_path` - Path to circuits.gl (for circom -l).
/// * `recurser_circuits_path` - Path to vadcop helpers/circuits (for circom -l).
/// * `recurser_pil_path` - Path to circom2pil/pil (for pil2com include).
/// * `circom_helpers_dir` - Directory with the witness generation Makefile + helpers.
/// * `witness_tracker` - Tracker for background witness library builds.
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_test_setup(
    build_dir: &str,
    circom_path: &str,
    circom_name: &str,
    setup_type: &str,
    hash: &str,
    circom_exec: &str,
    circuits_gl_path: &str,
    recurser_circuits_path: &str,
    recurser_pil_path: &str,
    circom_helpers_dir: &str,
    witness_tracker: &WitnessTracker,
) -> Result<()> {
    if !["compressor", "aggregation"].contains(&setup_type) {
        bail!("Invalid setup type '{}'. Must be one of: compressor, aggregation", setup_type);
    }

    // JS nameFile is always "Compressor" regardless of the setup type.
    const NAME_FILE: &str = "Compressor";

    let build_path = PathBuf::from(build_dir);
    // JS filesDir: path.join(buildDir, "provingKey", "build", nameFile, "airs", nameFile, "air")
    let files_dir =
        build_path.join("provingKey").join("build").join(NAME_FILE).join("airs").join(NAME_FILE).join("air");
    let circom_dir = build_path.join("circom");
    let build_inner = build_path.join("build");
    let pil_dir = build_path.join("pil");
    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_inner)?;
    fs::create_dir_all(&pil_dir)?;
    fs::create_dir_all(&files_dir)?;

    // -------------------------------------------------------------------------
    // Step 1: Compile the user-provided circom file directly.
    // (Unlike gen_recursive_setup, we skip pil2circom + gencircom.)
    // -------------------------------------------------------------------------
    tracing::info!("Compiling {}...", circom_name);
    let compile_status = std::process::Command::new(circom_exec)
        .args([
            "--O2",
            "--r1cs",
            "--prime",
            "goldilocks",
            "--c",
            "--verbose",
            "-l",
            recurser_circuits_path,
            "-l",
            circuits_gl_path,
        ])
        .arg(circom_path)
        .arg("-o")
        .arg(&build_inner)
        .output()
        .context("Failed to execute circom compiler")?;
    if !compile_status.status.success() {
        let stderr = String::from_utf8_lossy(&compile_status.stderr);
        bail!("Circom compilation failed for {}: {}", circom_name, stderr);
    }

    // -------------------------------------------------------------------------
    // Step 2: Copy .dat file to filesDir.
    // -------------------------------------------------------------------------
    let dat_src = build_inner.join(format!("{}_cpp", circom_name)).join(format!("{}.dat", circom_name));
    let dat_dst = files_dir.join(format!("{}.dat", NAME_FILE));
    if dat_src.exists() {
        tracing::info!("Copying circom files...");
        fs::copy(&dat_src, &dat_dst)?;
    }

    // -------------------------------------------------------------------------
    // Step 3: Witness library generation (background thread).
    // -------------------------------------------------------------------------
    witness_tracker.run_witness_library_generation(
        build_dir,
        files_dir.to_str().unwrap_or(""),
        circom_name,
        NAME_FILE,
        circom_helpers_dir,
    );

    // -------------------------------------------------------------------------
    // Step 4: plonk2pil — convert R1CS to PIL.
    // Use airgroup_name = "Compressor" (deterministic, avoids random hex suffix).
    // -------------------------------------------------------------------------
    let max_constraint_degree = if setup_type == "compressor" { Some(5) } else { None };
    let plonk_opts = PlonkOptions {
        airgroup_name: Some(NAME_FILE.to_string()),
        max_constraint_degree,
        hash_id: hash.to_string(),
        merge_copies: true,
    };
    let r1cs_path = build_inner.join(format!("{}.r1cs", circom_name));
    let r1cs_data =
        fs::read(&r1cs_path).with_context(|| format!("Failed to read R1CS file: {}", r1cs_path.display()))?;
    let plonk_result = plonk2pil::plonk2pil(&r1cs_data, setup_type, &plonk_opts)
        .context("plonk2pil failed in recursive test setup")?;

    // -------------------------------------------------------------------------
    // Step 5: Write fixed polynomials binary (intermediate).
    // -------------------------------------------------------------------------
    let fixed_bin_path = build_inner.join(format!("{}.fixed.bin", NAME_FILE));
    let fixed_info: Vec<(String, Vec<u32>, Vec<u64>)> =
        plonk_result.fixed_pols.iter().map(|fp| (fp.name.clone(), vec![fp.index as u32], fp.values.clone())).collect();
    fixed_cols::write_fixed_pols_bin(
        fixed_bin_path.to_str().unwrap(),
        &plonk_result.airgroup_name,
        &plonk_result.air_name,
        1u64 << plonk_result.n_bits,
        &fixed_info,
    )?;

    // -------------------------------------------------------------------------
    // Step 6: Write PIL source.
    // -------------------------------------------------------------------------
    let pil_path = pil_dir.join(format!("{}.pil", NAME_FILE));
    fs::write(&pil_path, &plonk_result.pil_str)?;

    // -------------------------------------------------------------------------
    // Step 7: Write exec buffer.
    // -------------------------------------------------------------------------
    let exec_path = files_dir.join(format!("{}.exec", NAME_FILE));
    let exec_bytes: Vec<u8> = plonk_result.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&exec_path, &exec_bytes)?;

    // -------------------------------------------------------------------------
    // Step 8: Compile PIL via pil2com (npm package).
    // -------------------------------------------------------------------------
    let pilout_path = build_inner.join(format!("{}.pilout", NAME_FILE));
    let std_pil_path = resolve_path_env("STD_PIL_PATH", "pil2-components/lib/std/pil");
    compile_pil(pil_path.to_str().unwrap(), pilout_path.to_str().unwrap(), &std_pil_path, recurser_pil_path)?;

    // -------------------------------------------------------------------------
    // Step 9: Load compiled pilout and run pil_info.
    // -------------------------------------------------------------------------
    let pilout_str = pilout_path.to_str().unwrap_or("");
    if !Path::new(pilout_str).exists() {
        bail!("Pilout not found at {}", pilout_str);
    }
    let proxy = PilOutProxy::new(pilout_str).map_err(|e| anyhow::anyhow!("Failed to load pilout: {}", e))?;
    let pilout = &proxy.pilout;
    if pilout.air_groups.is_empty() || pilout.air_groups[0].airs.is_empty() {
        bail!("Compiled pilout has no AIR groups: {}", pilout_str);
    }
    let air = &pilout.air_groups[0].airs[0];
    let num_rows = air.num_rows.unwrap_or(0) as usize;
    let n_bits_air = if num_rows > 0 { (num_rows as f64).log2() as usize } else { plonk_result.n_bits };

    // JS genRecursiveSetupTest always uses {blowupFactor: 3, lastLevelVerification: 1}
    // regardless of the setup type (unlike gen_recursive_setup which varies by template).
    let settings = StarkSettings { blowup_factor: Some(3), last_level_verification: Some(1), ..Default::default() };
    let stark_struct = generate_stark_struct(&settings, n_bits_air);

    let pil_info_result = crate::pil::info::pil_info(pilout, 0, 0, &stark_struct, &Default::default());

    // -------------------------------------------------------------------------
    // Step 10: Build and write starkinfo JSON.
    // -------------------------------------------------------------------------
    let opening_points = crate::output::stark_info::collect_opening_points(&pil_info_result.setup);
    let ev_map_len = pil_info_result.pil_code.ev_map.len();
    let fri = crate::output::stark_info::build_fri(&stark_struct, ev_map_len.max(1) as u64);

    let starkinfo_output = crate::output::stark_info::build_starkinfo_output(
        &pil_info_result.setup,
        &stark_struct,
        &pil_info_result.pil_code,
        &opening_points,
        &fri,
        0,
        0,
        NAME_FILE,
        pil_info_result.c_exp_id,
        pil_info_result.fri_exp_id,
        pil_info_result.q_deg,
    );

    let starkinfo_path = files_dir.join(format!("{}.starkinfo.json", NAME_FILE));
    fs::write(&starkinfo_path, crate::output::json::to_json_string(&starkinfo_output)?)?;

    // -------------------------------------------------------------------------
    // Step 11: Write verifierinfo and expressionsinfo JSON.
    // -------------------------------------------------------------------------
    let vi_ref = &pil_info_result.pil_code.verifier_info;
    let ei_ref = &pil_info_result.pil_code.expressions_info;
    fs::write(
        files_dir.join(format!("{}.verifierinfo.json", NAME_FILE)),
        crate::output::json::to_json_string(vi_ref)?,
    )?;
    fs::write(
        files_dir.join(format!("{}.expressionsinfo.json", NAME_FILE)),
        crate::output::json::to_json_string(ei_ref)?,
    )?;

    // -------------------------------------------------------------------------
    // Step 12: Write const file.
    // -------------------------------------------------------------------------
    let const_path = files_dir.join(format!("{}.const", NAME_FILE));
    let plonk_values = fixed_cols::reorder_plonk_pols_for_pilout(&plonk_result.fixed_pols, &pilout.symbols, 0, 0);
    fixed_cols::write_const_file(const_path.to_str().unwrap(), air, &plonk_values)?;

    // -------------------------------------------------------------------------
    // Step 13: Compute constant tree → verkey.json → verkey.bin.
    // -------------------------------------------------------------------------
    let verkey_json_path = files_dir.join(format!("{}.verkey.json", NAME_FILE));
    let const_root = bctree::compute_const_tree(
        const_path.to_str().unwrap(),
        starkinfo_path.to_str().unwrap(),
        verkey_json_path.to_str().unwrap(),
    );

    let mut verkey_bin = Vec::with_capacity(32);
    for &val in const_root.iter() {
        verkey_bin.extend_from_slice(&val.to_le_bytes());
    }
    fs::write(files_dir.join(format!("{}.verkey.bin", NAME_FILE)), &verkey_bin)?;

    // -------------------------------------------------------------------------
    // Step 14: Write .bin and .verifier.bin (always written in the test setup).
    // -------------------------------------------------------------------------
    let si_val: serde_json::Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
    let stark_info_loaded = crate::types::stark_info::StarkInfo::from_json(&si_val)?;

    let expressions_loaded = crate::types::stark_info::ExpressionsInfo::from(ei_ref);
    crate::io::bin_file::write_expressions_bin_file(
        files_dir.join(format!("{}.bin", NAME_FILE)).to_str().unwrap(),
        &stark_info_loaded,
        &expressions_loaded,
    )?;

    let verifier_loaded = crate::types::stark_info::VerifierInfo::from(vi_ref);
    crate::io::bin_file::write_verifier_expressions_bin_file(
        files_dir.join(format!("{}.verifier.bin", NAME_FILE)).to_str().unwrap(),
        &stark_info_loaded,
        &verifier_loaded,
    )?;

    // -------------------------------------------------------------------------
    // Step 15: Wait for witness library builds to complete.
    // -------------------------------------------------------------------------
    witness_tracker.await_all()?;

    // -------------------------------------------------------------------------
    // Step 16: Write globalInfo.json, globalConstraints.json, and
    //          globalConstraints.bin (uses the compiled pilout).
    //
    // JS equivalent: setAiroutInfo(airout, "EcMasFp5") after overriding
    //   airout.name = "build", airout.airGroups[0].name = "Compressor",
    //   air.name = "Compressor".
    // Since we passed airgroup_name = Some("Compressor") to plonk2pil, the
    // compiled pilout already has "Compressor" as the airgroup/air names.
    // We pass pilout_name = "build" to match JS airout.name = "build".
    // -------------------------------------------------------------------------
    let empty_settings = crate::types::stark_struct::StarkStructsConfig::default();
    crate::output::global_info::write_global_info(pilout, "build", build_dir, &empty_settings, hash)?;

    println!("files Generated Correctly");
    Ok(())
}
