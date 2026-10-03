//! Build and write the global proving-key files:
//!   - `pilout.globalInfo.json`
//!   - `pilout.globalConstraints.json`
//!   - `pilout.globalConstraints.bin`
//!
//! Also provides `write_bin_files_native` for the test suite (reads
//! expressionsinfo/verifierinfo JSON from disk and writes binary files).

use std::fs;
use std::path::Path;

use anyhow::Result;
use pil2_pilout::pilout as pb;
use serde_json::json;

use crate::types::stark_struct::StarkStructsConfig;
use pil_info::output::global_constraints::build_global_constraints_json;
use pil_info::output::global_info::{build_global_proof_values_map, build_global_publics_map};

/// Build the `globalInfo` JSON value in memory (does not write to disk).
pub(crate) fn build_global_info_json(
    pilout: &pb::PilOut,
    pilout_name: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
    agg_arity: usize,
    has_compressed_final: bool,
    setup_version: Option<&str>,
) -> serde_json::Value {
    let mut airs = Vec::new();
    let mut air_groups = Vec::new();
    let mut agg_types = Vec::new();

    for airgroup in &pilout.air_groups {
        let ag_name = airgroup.name.clone().unwrap_or_else(|| "unnamed".to_string());
        air_groups.push(ag_name.clone());

        let agv: Vec<serde_json::Value> =
            airgroup.air_group_values.iter().map(|v| json!({"aggType": v.agg_type, "stage": v.stage})).collect();
        agg_types.push(agv);

        let mut air_list = Vec::new();
        for air in &airgroup.airs {
            let a_name = air.name.clone().unwrap_or_else(|| "unnamed".to_string());
            let has_compressor = settings_map.has_compressor(&ag_name, &a_name);
            let mut entry = json!({
                "name": a_name,
                "num_rows": air.num_rows.unwrap_or(0),
            });
            if has_compressor {
                entry.as_object_mut().unwrap().insert("hasCompressor".to_string(), json!(true));
            }
            air_list.push(entry);
        }
        airs.push(serde_json::Value::Array(air_list));
    }

    let num_challenges: Vec<u32> =
        if pilout.num_challenges.is_empty() { vec![0] } else { pilout.num_challenges.clone() };

    let proof_values_map = build_global_proof_values_map(&pilout.symbols);
    let publics_map = build_global_publics_map(&pilout.symbols);

    let transcript_arity: u64 = proofman_common::hash_family::transcript_arity(hash);

    let mut global_info = json!({
        "name": pilout_name,
        "airs": airs,
        "air_groups": air_groups,
        "aggTypes": agg_types,
        "curve": "None",
        "latticeSize": 368,
        "transcriptArity": transcript_arity,
        "aggregationArity": agg_arity,
        // Whether this key carries the vadcop_final_compressed stage. Written so a consumer can
        // tell a key built without it from a key that is missing files: the loader must not try to
        // read a starkinfo for a stage that was never generated.
        "hasCompressedFinal": has_compressed_final,
        "nPublics": pilout.num_public_values,
        "numChallenges": num_challenges,
        "numProofValues": pilout.num_proof_values,
        "proofValuesMap": proof_values_map,
        "publicsMap": publics_map,
        "hash": hash,
    });
    // Consumer-assigned label (e.g. ZisK's); proofman only records it, the consumer checks it.
    if let Some(v) = setup_version {
        global_info.as_object_mut().unwrap().insert("setupVersion".to_string(), json!(v));
    }
    global_info
}

/// Write only `pilout.globalInfo.json` into `<build_dir>/provingKey/`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_global_info_json(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
    agg_arity: usize,
    has_compressed_final: bool,
    setup_version: Option<&str>,
) -> Result<()> {
    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;
    let global_info =
        build_global_info_json(pilout, pilout_name, settings_map, hash, agg_arity, has_compressed_final, setup_version);
    let global_info_str = crate::output::json::to_json_string(&global_info)?;
    fs::write(proving_key_dir.join("pilout.globalInfo.json"), &global_info_str)?;
    Ok(())
}

/// Write `pilout.globalConstraints.json` and `pilout.globalConstraints.bin`
/// into `<build_dir>/provingKey/`. Does not write `globalInfo.json`.
pub(crate) fn write_global_constraints(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
) -> Result<()> {
    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;

    // Never written out: only proofValuesMap/aggTypes are read below and the JSON is dropped,
    // so neither the hash nor the arity reaches a file or a reader.
    let global_info = build_global_info_json(
        pilout,
        pilout_name,
        settings_map,
        proofman_common::hash_family::DEFAULT_HASH_ID,
        proofman_common::global_info::fallback_aggregation_arity(),
        // Same reason as the hash and the arity above: this JSON is dropped, so the value is
        // never read back and any of the two would do.
        true,
        None,
    );

    let global_constraints = build_global_constraints_json(pilout, &pil_info::FieldCfg::goldilocks())?;
    let gc_str = crate::output::json::to_json_string(&global_constraints)?;
    fs::write(proving_key_dir.join("pilout.globalConstraints.json"), &gc_str)?;

    {
        use crate::output::global_constraints::write_global_constraints_bin_file;
        use crate::io::parser_args::{GlobalInfo as ParserGlobalInfo, ProofValueEntry};
        use crate::types::stark_info::GlobalConstraintsInfo;

        let proof_values_map_json =
            global_info.get("proofValuesMap").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let pvm: Vec<ProofValueEntry> = proof_values_map_json
            .iter()
            .map(|entry| ProofValueEntry { stage: entry.get("stage").and_then(|s| s.as_u64()).unwrap_or(1) })
            .collect();

        let agg_types_json = global_info.get("aggTypes").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let agg_types: Vec<Vec<u64>> = agg_types_json
            .iter()
            .map(|ag| {
                ag.as_array()
                    .map(|arr| arr.iter().map(|v| v.get("aggType").and_then(|a| a.as_u64()).unwrap_or(0)).collect())
                    .unwrap_or_default()
            })
            .collect();

        let gi = ParserGlobalInfo { proof_values_map: pvm, agg_types };
        let gci = GlobalConstraintsInfo::from_json(&global_constraints)?;
        let bin_path = proving_key_dir.join("pilout.globalConstraints.bin");
        write_global_constraints_bin_file(&gi, &gci, bin_path.to_str().unwrap_or(""))?;
    }

    Ok(())
}

/// Write all three output files (convenience wrapper, used by non-recursive path and tests).
pub(crate) fn write_global_info(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
    agg_arity: usize,
    has_compressed_final: bool,
) -> Result<()> {
    write_global_constraints(pilout, pilout_name, build_dir, settings_map)?;
    write_global_info_json(pilout, pilout_name, build_dir, settings_map, hash, agg_arity, has_compressed_final, None)?;
    tracing::info!("Global info and constraints written");
    Ok(())
}

/// Read expressionsinfo and verifierinfo JSON files from disk and write binary outputs.
///
/// Used by the golden-reference test suite. In production the binary files are
/// written directly from in-memory structs via `write_bin_files_from_pil_code`.
#[cfg(test)]
pub(crate) fn write_bin_files_native(
    starkinfo_path: &Path,
    expressionsinfo_path: &Path,
    verifierinfo_path: &Path,
    bin_output: &Path,
    verifier_bin_output: &Path,
) -> Result<()> {
    use anyhow::Context;
    use crate::types::stark_info::{ExpressionsInfo, StarkInfo, VerifierInfo};

    let si_data =
        fs::read_to_string(starkinfo_path).with_context(|| format!("Cannot read starkinfo: {:?}", starkinfo_path))?;
    let si_json: serde_json::Value = serde_json::from_str(&si_data)?;
    let stark_info = StarkInfo::from_json(&si_json)?;

    let ei_data = fs::read_to_string(expressionsinfo_path)
        .with_context(|| format!("Cannot read expressionsinfo: {:?}", expressionsinfo_path))?;
    let ei_json: serde_json::Value = serde_json::from_str(&ei_data)?;
    let ei = ExpressionsInfo::from_json(&ei_json)?;

    let vi_data = fs::read_to_string(verifierinfo_path)
        .with_context(|| format!("Cannot read verifierinfo: {:?}", verifierinfo_path))?;
    let vi_json: serde_json::Value = serde_json::from_str(&vi_data)?;
    let vi = VerifierInfo::from_json(&vi_json)?;

    crate::io::bin_file::write_expressions_bin_file(bin_output.to_str().unwrap_or(""), &stark_info, &ei)?;
    crate::io::bin_file::write_verifier_expressions_bin_file(
        verifier_bin_output.to_str().unwrap_or(""),
        &stark_info,
        &vi,
    )?;

    Ok(())
}

#[cfg(test)]
mod agg_arity_tests {
    use super::*;

    #[test]
    fn the_builder_emits_the_aggregation_arity() {
        let pilout = pb::PilOut::default();
        let settings = StarkStructsConfig::default();
        for arity in [2usize, 3] {
            let v = super::build_global_info_json(&pilout, "t", &settings, "Poseidon2", arity, true, None);
            assert_eq!(v["aggregationArity"], serde_json::json!(arity));
        }
    }

    #[test]
    fn the_builder_emits_the_setup_version_only_when_given() {
        let pilout = pb::PilOut::default();
        let settings = StarkStructsConfig::default();
        let v = super::build_global_info_json(&pilout, "t", &settings, "Poseidon2", 3, true, None);
        assert!(v.get("setupVersion").is_none());
        let v = super::build_global_info_json(&pilout, "t", &settings, "Poseidon2", 3, true, Some("1.3.1"));
        assert_eq!(v["setupVersion"], serde_json::json!("1.3.1"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn test_global_info_has_compressor() {
        let pilout_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../pil/zisk.pilout");
        if !std::path::Path::new(pilout_path).exists() {
            eprintln!("Skipping test_global_info_has_compressor: pilout not found");
            return;
        }

        let pilout_data = std::fs::read(pilout_path).unwrap();
        let pilout = pb::PilOut::decode(pilout_data.as_slice()).unwrap();
        let pilout_name = pilout.name.clone().unwrap_or_else(|| "pilout".to_string());

        let settings_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../state-machines/starkstructs.json");
        let settings_map: StarkStructsConfig = if std::path::Path::new(settings_path).exists() {
            let data = std::fs::read_to_string(settings_path).unwrap();
            StarkStructsConfig::from_json_str(&data).unwrap()
        } else {
            StarkStructsConfig::default()
        };

        let build_dir = "/tmp/r39_test_global_info";
        let _ = std::fs::remove_dir_all(build_dir);
        write_global_info(&pilout, &pilout_name, build_dir, &settings_map, "Poseidon2", 3, true).unwrap();

        let gi_str = std::fs::read_to_string(format!("{}/provingKey/pilout.globalInfo.json", build_dir)).unwrap();
        let gi: serde_json::Value = serde_json::from_str(&gi_str).unwrap();

        let airs = gi.get("airs").unwrap().as_array().unwrap();
        assert!(!airs.is_empty());
        let first_group = airs[0].as_array().unwrap();
        let mut found_has_compressor = false;
        for air in first_group {
            let name = air.get("name").unwrap().as_str().unwrap();
            if name == "Keccakf" || name == "Sha256f" || name == "ArithEq" || name == "ArithEq384" {
                assert!(air.get("hasCompressor").is_some());
                assert!(air.get("hasCompressor").unwrap().as_bool().unwrap());
                found_has_compressor = true;
            }
            if name == "Main" || name == "Mem" {
                assert!(air.get("hasCompressor").is_none());
            }
        }
        assert!(found_has_compressor);

        let gc_str =
            std::fs::read_to_string(format!("{}/provingKey/pilout.globalConstraints.json", build_dir)).unwrap();
        let gc: serde_json::Value = serde_json::from_str(&gc_str).unwrap();
        let constraints = gc.get("constraints").unwrap().as_array().unwrap();
        assert!(!constraints.is_empty());
        let c0 = &constraints[0];
        assert!(c0.get("tmpUsed").is_some());
        assert!(c0.get("code").is_some());
        assert_eq!(c0.get("boundary").unwrap().as_str().unwrap(), "finalProof");
        assert!(!gc.get("hints").unwrap().as_array().unwrap().is_empty());
        assert!(std::path::Path::new(&format!("{}/provingKey/pilout.globalConstraints.bin", build_dir)).exists());

        let _ = std::fs::remove_dir_all(build_dir);
    }

    #[test]
    fn test_bin_file_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Dma/air");
        let si = dir.join("Dma.starkinfo.json");
        let golden_bin = dir.join("Dma.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Dma files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_bin_regression_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Dma.bin");
        let out_vbin = tmp.join("Dma.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Dma.expressionsinfo.json"),
            &dir.join("Dma.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Dma.bin size mismatch");
        assert_eq!(golden, actual, "Dma.bin content mismatch");
        assert_eq!(
            std::fs::read(dir.join("Dma.verifier.bin")).unwrap(),
            std::fs::read(&out_vbin).unwrap(),
            "Dma.verifier.bin mismatch"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_binary_bin_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Binary/air");
        let si = dir.join("Binary.starkinfo.json");
        let golden_bin = dir.join("Binary.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Binary files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_binary_bin_regression_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Binary.bin");
        let out_vbin = tmp.join("Binary.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Binary.expressionsinfo.json"),
            &dir.join("Binary.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Binary.bin size mismatch");
        if golden != actual {
            let pos = golden.iter().zip(actual.iter()).position(|(a, b)| a != b).unwrap_or(0);
            panic!("Binary.bin mismatch at byte {} (golden={:#x} actual={:#x})", pos, golden[pos], actual[pos]);
        }
        assert_eq!(std::fs::read(dir.join("Binary.verifier.bin")).unwrap(), std::fs::read(&out_vbin).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_arith_bin_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Arith/air");
        let si = dir.join("Arith.starkinfo.json");
        let golden_bin = dir.join("Arith.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Arith files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_arith_bin_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Arith.bin");
        let out_vbin = tmp.join("Arith.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Arith.expressionsinfo.json"),
            &dir.join("Arith.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden, actual, "Arith.bin mismatch");
        assert_eq!(std::fs::read(dir.join("Arith.verifier.bin")).unwrap(), std::fs::read(&out_vbin).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
