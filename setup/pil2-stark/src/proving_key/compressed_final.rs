//! Port of `generateCompressedFinalSetup.js`: generate the vadcop_final_compressed
//! circuit, compile it, and produce the proving key artifacts.
//!
//! The compressed final setup:
//! 1. Generates a verifier circom for vadcop_final using pil2circom
//! 2. Generates the compressed final circom via gencircom
//! 3. Compiles circom to R1CS + C++
//! 4. Converts R1CS to PIL (plonk2pil with "aggregation")
//! 5. Compiles PIL
//! 6. Runs starkSetup with specific compressed final settings
//! 7. Computes constant tree
//! 8. Writes verifier.rs for the compressed final circuit

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use pil2_pilout::pilout_proxy::PilOutProxy;
use pil2_stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use pil2_stark_recurser::plonk2pil::{self, PlonkResult};

use crate::proving_key::bctree;
use crate::io::fixed_cols;
use pil2_stark_recurser::stark2circom::{
    gen_circom_circuit, gen_stark_verifier, CircomGenOptions, GenCircomCircuitInput, StarkVerifierOptions,
};
use crate::proving_key::recursive::compile_pil;
use proofman_common::hash_family::HashFamily;
use crate::output::witness_gen::WitnessTracker;

/// Configuration for the compressed final setup.
pub struct CompressedFinalConfig<'a> {
    pub build_dir: &'a str,
    pub hash: &'a str,
    pub name: &'a str,
    pub const_root: &'a [String; 4],
    pub verification_keys: &'a [Vec<Vec<String>>],
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
    /// Low-degree test of the compressed proof; `None` is the family's (`compressed_final_stir`).
    pub low_degree_test: Option<crate::types::stark_struct::LowDegreeTestKind>,
}

/// Run the compressed final setup.
///
/// Ports `genCompressedFinalSetup()` from `generateCompressedFinalSetup.js`.
pub fn gen_compressed_final_setup(config: &CompressedFinalConfig<'_>, witness_tracker: &WitnessTracker) -> Result<()> {
    let family: HashFamily = config.hash.parse().map_err(anyhow::Error::msg)?;
    let template = "vadcop_final_compressed";
    let verifier_name = "vadcop_final_stark.verifier.circom";
    let build_dir = PathBuf::from(config.build_dir);

    let files_dir = build_dir.join("provingKey").join(config.name).join(template);
    fs::create_dir_all(&files_dir)?;

    let circom_dir = build_dir.join("circom");
    let build_path = build_dir.join("build");
    let pil_dir = build_dir.join("pil");
    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_path)?;
    fs::create_dir_all(&pil_dir)?;

    // Generate verifier circom via Rust gen_stark_verifier
    {
        let rust_opts = StarkVerifierOptions {
            hash: config.hash.to_string(),
            skip_main: true,
            verkey_input: false,
            enable_input: false,
            input_challenges: false,
            fri_queries_batch_size: None,
        };
        let circom_src =
            gen_stark_verifier(Some(config.const_root), config.stark_info, config.verifier_info, &rust_opts)
                .context("gen_stark_verifier failed in compressed final setup")?;
        fs::write(circom_dir.join(verifier_name), &circom_src).context("Failed to write verifier circom")?;
    }

    // Generate compressed final circom via Rust gen_circom_circuit
    let verifier_filenames = [verifier_name.to_string()];
    let circom_out = circom_dir.join(format!("{}.circom", template));
    {
        // Build per-airgroup, per-air VKs as String vecs matching GenCircomCircuitInput format.
        let basic_vk: Vec<Vec<Vec<String>>> = config.verification_keys.to_vec();

        // Aggregates nothing: the template never reads `agg_arity`, and `gen_recursive2`
        // rejects 0 outright, so it cannot pass for one.
        let rust_opts = CircomGenOptions {
            airgroup_id: None,
            has_compressor: false,
            has_recursion: false,
            is_final: false,
            agg_arity: 0,
            batch_size: 1,
        };
        let rust_input = GenCircomCircuitInput {
            template_name: "src/vadcop/templates/final_compressed.circom.ejs",
            stark_infos: std::slice::from_ref(config.stark_info),
            vadcop_info: &serde_json::Value::Null,
            verifier_filenames: &verifier_filenames,
            basic_vk: &basic_vk,
            agg_vk: &[],
            publics: &[],
            options: &rust_opts,
        };
        let circom_src = gen_circom_circuit(&rust_input).context("gen_circom_circuit failed for final_compressed")?;
        fs::write(&circom_out, &circom_src).context("Failed to write final_compressed circom")?;
    }

    // Compile circom
    tracing::info!("Compiling {}...", template);
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
        .arg(circom_out.to_str().unwrap())
        .arg("-o")
        .arg(build_path.to_str().unwrap())
        .output()
        .context("Failed to execute circom for compressed final setup")?;

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr);
        bail!("Circom compilation failed for {}: {}", template, stderr);
    }

    // Copy .dat file
    tracing::info!("Copying circom files...");
    let dat_src = build_path.join(format!("{}_cpp", template)).join(format!("{}.dat", template));
    let dat_dst = files_dir.join(format!("{}.dat", template));
    if dat_src.exists() {
        fs::copy(&dat_src, &dat_dst)?;
    }

    // Generate witness library
    witness_tracker.run_witness_library_generation(
        config.build_dir,
        files_dir.to_str().unwrap_or(""),
        template,
        template,
        config.circom_helpers_dir,
    );

    // plonk2pil
    let r1cs_path = build_path.join(format!("{}.r1cs", template));
    let r1cs_data = fs::read(&r1cs_path).with_context(|| format!("Failed to read R1CS: {}", r1cs_path.display()))?;

    let plonk_opts = PlonkOptions {
        airgroup_name: Some("VadcopFinalCompressed".to_string()),
        // blake3 follows the blowup, as vadcop_final does; poseidon keeps the packer default.
        max_constraint_degree: family.is_blake3().then(|| {
            proofman_common::hash_family::max_constraint_degree_for_blowup(
                proofman_common::hash_family::compressed_final_blowup_factor(config.hash),
            )
        }),
        hash_id: config.hash.to_string(),
        merge_copies: true,
        blake3_lanes: proofman_common::hash_family::compressed_final_blake3_lanes(config.hash),
        min_n_bits: None,
    };
    let _span = tracing::info_span!("stage", t = "vadcop_final_compressed").entered();
    let plonk_result: PlonkResult = plonk2pil::plonk2pil(&r1cs_data, "aggregation", &plonk_opts)
        .context("plonk2pil failed in compressed final setup")?;

    // Write fixed pols binary
    let fixed_bin_path = build_path.join(format!("{}.fixed.bin", template));
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
    let pil_path = pil_dir.join(format!("{}.pil", template));
    fs::write(&pil_path, &plonk_result.pil_str)?;

    // Compile PIL
    let pilout_path = build_path.join(format!("{}.pilout", template));
    compile_pil(
        pil_path.to_str().unwrap(),
        pilout_path.to_str().unwrap(),
        config.std_pil_path,
        config.recurser_pil_path,
    )?;

    // Write exec
    let exec_path = files_dir.join(format!("{}.exec", template));
    let exec_bytes: Vec<u8> = plonk_result.exec.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&exec_path, &exec_bytes)?;

    // Const file writing is deferred until after pil_info, which determines
    // the true nConstants (may be larger than plonk2pil's fixedPols count).
    let const_path = files_dir.join(format!("{}.const", template));
    let plonk_n_rows = 1usize << plonk_result.n_bits;
    let plonk_n_fixed = plonk_result.fixed_pols.len();

    // Compressed final stark struct settings: no wrap reads this proof, so its geometry is a proof-size knob.
    use crate::types::stark_struct::LowDegreeTestKind;
    let low_degree_test =
        config.low_degree_test.unwrap_or(if proofman_common::hash_family::compressed_final_stir(config.hash) {
            LowDegreeTestKind::Stir
        } else {
            LowDegreeTestKind::Fri
        });
    let multiproof = proofman_common::hash_family::compressed_final_multiproof(config.hash);
    // A pinned schedule applies only at the air size and blowup it was measured at.
    let here = (plonk_result.n_bits, proofman_common::hash_family::compressed_final_blowup_factor(config.hash));
    let pinned = |measured: (usize, usize)| {
        if measured != here {
            tracing::warn!(
                "compressed final at (2^n, blowup) {here:?}: schedule pinned for {measured:?}, deriving one"
            );
        }
        measured == here
    };
    let stir_schedule = (low_degree_test == LowDegreeTestKind::Stir)
        .then(|| proofman_common::hash_family::compressed_final_stir_schedule(config.hash))
        .flatten()
        .filter(|(measured, ..)| pinned(*measured));
    let fri_folds = (low_degree_test == LowDegreeTestKind::Fri)
        .then(|| proofman_common::hash_family::compressed_final_fri_folds(config.hash))
        .flatten()
        .filter(|(measured, _)| pinned(*measured));
    let compressed_settings = crate::types::stark_struct::StarkSettings {
        initial_blowup_factor: Some(proofman_common::hash_family::compressed_final_blowup_factor(config.hash)),
        initial_folding_factor: Some(if family.is_blake3() { 4 } else { 3 }),
        grinding_bits: Some(proofman_common::hash_family::compressed_final_grinding_bits(config.hash)),
        merkle_tree_arity: (!family.is_blake3()).then_some(2),
        last_level_verification: if multiproof {
            Some(0)
        } else {
            proofman_common::hash_family::compressed_final_last_level_verification(config.hash)
        },
        // final_degree is a log degree here; the helper gives the terminal FRI domain.
        final_degree: Some(
            proofman_common::hash_family::compressed_final_fri_terminal_degree(config.hash)
                - proofman_common::hash_family::compressed_final_blowup_factor(config.hash),
        ),
        low_degree_test: Some(low_degree_test),
        folding_factors: stir_schedule
            .as_ref()
            .map(|(_, folds, _)| folds.clone())
            .or(fri_folds.map(|(_, folds)| folds)),
        grinding_bits_queries: stir_schedule.as_ref().map(|(_, _, grinding)| grinding.clone()),
        ..Default::default()
    };
    let compressed_stark_struct =
        crate::types::stark_struct::generate_stark_struct(&compressed_settings, plonk_result.n_bits, config.hash);
    let (mta, ta) = (compressed_stark_struct.merkle_tree_arity, compressed_stark_struct.transcript_arity);
    if multiproof && !proofman_common::hash_family::has_multiproof_codec(config.hash, mta as u64, ta as u64) {
        bail!("no {} multiproof codec at tree arity {mta}, transcript arity {ta}", config.hash);
    }

    // Run real starkSetup via pil_info on the compiled compressed final pilout
    let starkinfo_path = files_dir.join(format!("{}.starkinfo.json", template));

    let pilout_file_str = pilout_path.to_str().unwrap_or("");
    if !Path::new(pilout_file_str).exists() {
        bail!("Compressed final pilout not found at {}", pilout_file_str);
    }

    let proxy = PilOutProxy::new(pilout_file_str).map_err(|e| anyhow::anyhow!("Failed to load pilout: {}", e))?;
    let pilout = &proxy.pilout;
    if pilout.air_groups.is_empty() || pilout.air_groups[0].airs.is_empty() {
        bail!("Compressed final pilout has no AIR groups");
    }

    let key = crate::pil::stark_setup::KeyIdentity {
        name: Some("compressed_final".into()),
        multiproof,
        ..Default::default()
    };
    let pil_info_result =
        crate::pil::stark_setup::stark_setup(pilout, 0, 0, &compressed_stark_struct, &Default::default(), &key)?;

    let starkinfo_output = pil_info_result.starkinfo;
    tracing::info!(
        "Circuit '{}' low-degree test: {}",
        template,
        starkinfo_output.stark_struct.low_degree_test.describe()
    );
    let verifier_info_ref = &pil_info_result.pil_code.verifier_info;
    let expressions_info_ref = &pil_info_result.pil_code.expressions_info;

    fs::write(&starkinfo_path, crate::output::json::to_json_string(&starkinfo_output)?)?;
    fs::write(
        files_dir.join(format!("{}.verifierinfo.json", template)),
        crate::output::json::to_json_string(verifier_info_ref)?,
    )?;
    fs::write(
        files_dir.join(format!("{}.expressionsinfo.json", template)),
        crate::output::json::to_json_string(expressions_info_ref)?,
    )?;

    // Write binary files — convert in-memory structs directly (no disk round-trip)
    {
        let si_val: serde_json::Value = serde_json::from_str(&fs::read_to_string(&starkinfo_path)?)?;
        let si_loaded = crate::types::stark_info::StarkInfo::from_json(&si_val)?;

        let expr_loaded = crate::types::stark_info::ExpressionsInfo::try_from(expressions_info_ref)?;
        crate::io::bin_file::write_expressions_bin_file(
            files_dir.join(format!("{}.bin", template)).to_str().unwrap(),
            &si_loaded,
            &expr_loaded,
        )?;

        // Write const file: pilout inline values are the selector polynomials;
        // plonk2pil polynomials fill the columns with empty pilout values.
        {
            let air = &pilout.air_groups[0].airs[0];
            let plonk_values =
                fixed_cols::reorder_plonk_pols_for_pilout(&plonk_result.fixed_pols, &pilout.symbols, 0, 0);
            fixed_cols::write_const_file(const_path.to_str().unwrap(), air, &plonk_values)?;
            tracing::info!(
                "Wrote {} const file: {} cols ({} from plonk + {} from pilout), {} rows",
                template,
                air.fixed_cols.len(),
                plonk_n_fixed,
                air.fixed_cols.len().saturating_sub(plonk_n_fixed),
                plonk_n_rows
            );
        }
    }

    // Compute constant tree
    tracing::info!("Computing Constant Tree for {}...", template);
    let verkey_json_path = files_dir.join(format!("{}.verkey.json", template));
    crate::proving_key::util::require_const_file(template, &const_path)?;
    let root = bctree::compute_const_tree(
        const_path.to_str().unwrap(),
        starkinfo_path.to_str().unwrap(),
        verkey_json_path.to_str().unwrap(),
    )?;
    let mut verkey_bin = Vec::with_capacity(32);
    for &val in root.iter() {
        verkey_bin.extend_from_slice(&val.to_le_bytes());
    }
    fs::write(files_dir.join(format!("{}.verkey.bin", template)), &verkey_bin)?;

    // Write verifier.rs from computed starkinfo and verifierinfo
    {
        let si_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(files_dir.join(format!("{}.starkinfo.json", template)))?)?;
        let si_loaded = crate::types::stark_info::StarkInfo::from_json(&si_val)?;
        let ver_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(files_dir.join(format!("{}.verifierinfo.json", template)))?)?;
        let ver_loaded = crate::types::stark_info::VerifierInfo::from_json(&ver_val)?;
        crate::output::verifier::write_verifier_rust_file(
            files_dir.join(format!("{}.verifier.rs", template)).to_str().unwrap(),
            &si_loaded,
            &ver_loaded,
            true,
            config.hash,
        )?;
    }

    // Wait for witness library
    witness_tracker.await_all()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Deriving Poseidon's blowup and grinding from the family table reproduces what it shipped with.
    #[test]
    fn the_derived_settings_match_the_shipped_geometry() {
        for fam in ["Poseidon1", "Poseidon2"] {
            assert_eq!(proofman_common::hash_family::compressed_final_blowup_factor(fam), 4, "{fam} blowup");
            assert_eq!(proofman_common::hash_family::compressed_final_grinding_bits(fam), 22, "{fam} grinding");
        }
    }
}

#[cfg(test)]
mod missing_const_tests {
    /// The const-tree block bails on a missing const file instead of warning and returning a zero root.
    #[test]
    fn missing_const_file_is_an_error() {
        let p = std::path::Path::new("/nonexistent/compressed_final.const");
        assert!(crate::proving_key::util::require_const_file("compressed_final", p).is_err());
    }
}
