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
use pil_info::core::dag::build::build_global_dag;
use pil_info::core::global::global_from_core;
use pil_info::core::pilout_core::PiloutCore;
use pil_info::output::global_constraints::global_constraints_json;
use pil_info::output::global_info::global_values_maps;
use serde_json::json;
use proofman_common::hash_family::HashFamily;

use crate::types::stark_struct::StarkStructsConfig;

/// The `globalInfo` JSON value of the pilout's header (`PiloutCore::header_of`); does not write to disk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn global_info_json(
    core: &PiloutCore,
    pilout_name: &str,
    settings_map: &StarkStructsConfig,
    hash: HashFamily,
    agg_arity: usize,
    has_compressed_final: bool,
    setup_version: Option<&str>,
    // (airgroup, air) -> recursive1 batch size; by index, since air names repeat across airgroups.
    r1_batch_sizes: &std::collections::HashMap<(usize, usize), usize>,
) -> Result<serde_json::Value> {
    let (proof_values_map, publics_map) = global_values_maps(core)?;
    let unnamed = |n: &Option<String>| n.clone().unwrap_or_else(|| "unnamed".to_string());
    let (mut airs, mut air_groups, mut agg_types) = (Vec::new(), Vec::new(), Vec::new());
    for (ag_idx, airgroup) in core.air_groups.iter().enumerate() {
        let ag_name = unnamed(&airgroup.name);
        let agv = airgroup.air_group_values.iter().map(|v| json!({"aggType": v.agg_type as i32, "stage": v.stage}));
        agg_types.push(agv.collect::<Vec<_>>());
        let mut air_list = Vec::new();
        for (air_idx, air) in airgroup.airs.iter().enumerate() {
            let a_name = unnamed(&air.name);
            let mut entry = json!({ "name": a_name, "num_rows": air.num_rows });
            if settings_map.has_compressor(&ag_name, &a_name) {
                entry["hasCompressor"] = json!(true);
            }
            if let Some(&k) = r1_batch_sizes.get(&(ag_idx, air_idx)).filter(|&&k| k > 1) {
                entry["r1BatchSize"] = json!(k);
            }
            air_list.push(entry);
        }
        airs.push(air_list);
        air_groups.push(ag_name);
    }
    let num_challenges = if core.num_challenges.is_empty() { vec![0] } else { core.num_challenges.clone() };

    let mut global_info = json!({
        "name": pilout_name,
        "airs": airs,
        "air_groups": air_groups,
        "aggTypes": agg_types,
        "curve": "None",
        "latticeSize": 368,
        "transcriptArity": proofman_common::hash_family::transcript_arity(hash),
        "aggregationArity": agg_arity,
        // Whether this key carries the vadcop_final_compressed stage. Written so a consumer can
        // tell a key built without it from a key that is missing files: the loader must not try to
        // read a starkinfo for a stage that was never generated.
        "hasCompressedFinal": has_compressed_final,
        "nPublics": core.num_public_values,
        "numChallenges": num_challenges,
        "numProofValues": core.num_proof_values,
        "proofValuesMap": proof_values_map,
        "publicsMap": publics_map,
        "hash": hash,
    });
    // The global constraints, for every verifier of them (vadcop_final, verifyGlobalConstraints); absent if none.
    let gs = global_from_core(core, &pil_info::FieldCfg::goldilocks())?;
    if let Some(dag) = build_global_dag(&gs).map_err(|e| anyhow::anyhow!("global constraints: {e:?}"))? {
        global_info["globalConstraintDag"] = serde_json::to_value(&dag)?;
    }
    // Consumer-assigned label (e.g. ZisK's); proofman only records it, the consumer checks it.
    if let Some(v) = setup_version {
        global_info["setupVersion"] = json!(v);
    }
    Ok(global_info)
}

/// Write only `pilout.globalInfo.json` into `<build_dir>/provingKey/`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_global_info_json(
    core: &PiloutCore,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: HashFamily,
    agg_arity: usize,
    has_compressed_final: bool,
    setup_version: Option<&str>,
    // (airgroup, air) -> recursive1 batch size; by index, since air names repeat across airgroups.
    r1_batch_sizes: &std::collections::HashMap<(usize, usize), usize>,
) -> Result<()> {
    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;
    let global_info = global_info_json(
        core,
        pilout_name,
        settings_map,
        hash,
        agg_arity,
        has_compressed_final,
        setup_version,
        r1_batch_sizes,
    )?;
    let global_info_str = crate::output::json::to_json_string(&global_info)?;
    fs::write(proving_key_dir.join("pilout.globalInfo.json"), &global_info_str)?;
    Ok(())
}

/// Write `pilout.globalConstraints.json` and `pilout.globalConstraints.bin`
/// into `<build_dir>/provingKey/`. Does not write `globalInfo.json`.
pub(crate) fn write_global_constraints(core: &PiloutCore, build_dir: &str) -> Result<()> {
    use crate::io::parser_args::{GlobalInfo as ParserGlobalInfo, ProofValueEntry};
    use crate::output::global_constraints::write_global_constraints_bin_file;
    use crate::types::stark_info::GlobalConstraintsInfo;

    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;
    let gs = global_from_core(core, &pil_info::FieldCfg::goldilocks())?;
    let global_constraints = global_constraints_json(&gs)?;
    let gc_str = crate::output::json::to_json_string(&global_constraints)?;
    fs::write(proving_key_dir.join("pilout.globalConstraints.json"), &gc_str)?;

    let gi = ParserGlobalInfo {
        proof_values_map: gs.proof_value_stages.iter().map(|&s| ProofValueEntry { stage: u64::from(s) }).collect(),
        agg_types: gs.airgroup_values.iter().map(|g| g.iter().map(|v| v.agg_type as u64).collect()).collect(),
    };
    let gci = GlobalConstraintsInfo::from_json(&global_constraints)?;
    let bin_path = proving_key_dir.join("pilout.globalConstraints.bin");
    write_global_constraints_bin_file(&gi, &gci, bin_path.to_str().unwrap_or(""))?;
    Ok(())
}

/// Write all three output files (convenience wrapper, used by non-recursive path and tests).
pub(crate) fn write_global_info(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: HashFamily,
    agg_arity: usize,
    has_compressed_final: bool,
) -> Result<()> {
    let core = PiloutCore::header_of(pilout)?;
    write_global_constraints(&core, build_dir)?;
    write_global_info_json(
        &core,
        pilout_name,
        build_dir,
        settings_map,
        hash,
        agg_arity,
        has_compressed_final,
        None,
        &Default::default(),
    )?;
    tracing::info!("Global info and constraints written");
    Ok(())
}

/// Read the starkinfo and expressionsinfo JSON files from disk and write the `.bin`.
///
/// Used by the golden-reference test suite. In production the binary files are
/// written directly from in-memory structs via `write_bin_files_from_pil_code`.
#[cfg(test)]
pub(crate) fn write_bin_files_native(
    starkinfo_path: &Path,
    expressionsinfo_path: &Path,
    bin_output: &Path,
) -> Result<()> {
    use anyhow::Context;
    use crate::types::stark_info::{ExpressionsInfo, StarkInfo};

    let si_data =
        fs::read_to_string(starkinfo_path).with_context(|| format!("Cannot read starkinfo: {:?}", starkinfo_path))?;
    let si_json: serde_json::Value = serde_json::from_str(&si_data)?;
    let stark_info = StarkInfo::from_json(&si_json)?;

    let ei_data = fs::read_to_string(expressionsinfo_path)
        .with_context(|| format!("Cannot read expressionsinfo: {:?}", expressionsinfo_path))?;
    let ei_json: serde_json::Value = serde_json::from_str(&ei_data)?;
    let ei = ExpressionsInfo::from_json(&ei_json)?;

    crate::io::bin_file::write_expressions_bin_file(bin_output.to_str().unwrap_or(""), &stark_info, &ei)
}

#[cfg(test)]
mod r1_batch_size_tests {
    use super::*;

    fn pilout() -> pb::PilOut {
        pb::PilOut {
            air_groups: vec![pb::AirGroup {
                name: Some("g".into()),
                airs: vec![
                    pb::Air { name: Some("big".into()), num_rows: Some(1024), ..Default::default() },
                    pb::Air { name: Some("small".into()), num_rows: Some(256), ..Default::default() },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn airs(v: &serde_json::Value) -> Vec<serde_json::Value> {
        v["airs"][0].as_array().unwrap().clone()
    }

    /// Only batched airs get the field; its absence reads as 1.
    #[test]
    fn only_a_batched_air_gets_the_field() {
        let mut sizes = std::collections::HashMap::new();
        sizes.insert((0usize, 1usize), 5usize);
        sizes.insert((0usize, 0usize), 1usize);
        let v = global_info_json(
            &PiloutCore::header_of(&pilout()).unwrap(),
            "t",
            &StarkStructsConfig::default(),
            HashFamily::Blake3,
            2,
            false,
            None,
            &sizes,
        )
        .unwrap();
        let a = airs(&v);
        assert_eq!(a[0]["name"], "big");
        assert!(a[0].get("r1BatchSize").is_none(), "k=1 must not be written");
        assert_eq!(a[1]["r1BatchSize"], 5);
    }

    /// The field has to survive the trip, since the prover sizes the witness from it.
    #[test]
    fn the_reader_takes_it_back_off_the_json() {
        let air: proofman_common::global_info::GlobalInfoAir =
            serde_json::from_value(serde_json::json!({"name": "a", "num_rows": 8, "r1BatchSize": 3})).unwrap();
        assert_eq!(air.r1_batch_size, Some(3));
        let plain: proofman_common::global_info::GlobalInfoAir =
            serde_json::from_value(serde_json::json!({"name": "a", "num_rows": 8})).unwrap();
        assert_eq!(plain.r1_batch_size, None);
    }
}

#[cfg(test)]
mod agg_arity_tests {
    use super::*;

    #[test]
    fn the_builder_emits_the_aggregation_arity() {
        let pilout = PiloutCore::header_of(&pb::PilOut::default()).unwrap();
        let settings = StarkStructsConfig::default();
        for arity in proofman_common::global_info::VALID_AGGREGATION_ARITIES {
            let v = super::global_info_json(
                &pilout,
                "t",
                &settings,
                HashFamily::Poseidon2,
                arity,
                true,
                None,
                &Default::default(),
            )
            .unwrap();
            assert_eq!(v["aggregationArity"], serde_json::json!(arity));
        }
    }

    #[test]
    fn the_builder_emits_the_setup_version_only_when_given() {
        let pilout = PiloutCore::header_of(&pb::PilOut::default()).unwrap();
        let settings = StarkStructsConfig::default();
        let v =
            super::global_info_json(&pilout, "t", &settings, HashFamily::Poseidon2, 3, true, None, &Default::default())
                .unwrap();
        assert!(v.get("setupVersion").is_none());
        let v = super::global_info_json(
            &pilout,
            "t",
            &settings,
            HashFamily::Poseidon2,
            3,
            true,
            Some("1.4.0"),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(v["setupVersion"], serde_json::json!("1.4.0"));
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
        write_global_info(&pilout, &pilout_name, build_dir, &settings_map, HashFamily::Poseidon2, 3, true).unwrap();

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
        write_bin_files_native(&si, &dir.join("Dma.expressionsinfo.json"), &out_bin)
            .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Dma.bin size mismatch");
        assert_eq!(golden, actual, "Dma.bin content mismatch");
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
        write_bin_files_native(&si, &dir.join("Binary.expressionsinfo.json"), &out_bin)
            .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Binary.bin size mismatch");
        if golden != actual {
            let pos = golden.iter().zip(actual.iter()).position(|(a, b)| a != b).unwrap_or(0);
            panic!("Binary.bin mismatch at byte {} (golden={:#x} actual={:#x})", pos, golden[pos], actual[pos]);
        }
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
        write_bin_files_native(&si, &dir.join("Arith.expressionsinfo.json"), &out_bin)
            .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden, actual, "Arith.bin mismatch");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
