//! `pilout.globalInfo.json` of a pilfflonk `provingKey/` (pilfflonk/docs/formats.md#globalinfo),
//! and where the other files of the `provingKey/` go (pilfflonk/docs/formats.md#provingkey).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{invalid, PilfflonkResult};
use crate::json::JsonFile;
use crate::pilfflonk_info::NameStageEntry;
use crate::tag::{Backend, Field, Modulus, Transcript};

/// The file name of the globalInfo, at the root of the `provingKey/`.
pub const GLOBAL_INFO_FILE: &str = "pilout.globalInfo.json";

/// The file name of the global constraints, at the root of the `provingKey/`: `pil-info` writes
/// it, in the STARK's format (pilfflonk/docs/formats.md#globalconstraints).
pub const GLOBAL_CONSTRAINTS_FILE: &str = "pilout.globalConstraints.json";

/// The directory, under `<provingKey>/<name>/`, of the backend's global files (the convention of
/// `common::GlobalInfo::get_setup_path`).
pub const BACKEND_DIR: &str = "pilfflonk";

/// The file name of the SRS, in the backend directory.
pub const SRS_FILE: &str = "pilfflonk.srs.bin";

/// The file name of the vkey, in the backend directory.
pub const VKEY_FILE: &str = "pilfflonk.vkey.json";

/// The version of the formats this crate reads and writes (pilfflonk/docs/formats.md):
/// `formatVersion` of the globalInfo and of the vkey.
pub const FORMAT_VERSION: u64 = 1;

/// The largest `nBits` there is: `r - 1 = 2^28 · odd`, so no domain of roots of unity, and so no
/// trace, has more than `2^28` points (pilfflonk/docs/protocol.md#notation).
pub const MAX_NBITS: u64 = 28;

/// `pilout.globalInfo.json`: the part of the STARK's schema that does not depend on the backend,
/// and the pilfflonk fields (pilfflonk/docs/formats.md#globalinfo). It has no `hash`, `curve`,
/// `transcriptArity`, `aggregationArity`, `latticeSize` nor `hasCompressedFinal`, so
/// `common::GlobalInfo` refuses it (pilfflonk/docs/formats.md#provingkey): the STARK tools cannot
/// load it by mistake, and this is the type the pilfflonk runtime reads it with.
///
/// The common fields keep the STARK's names and meaning (`setup/pil2-stark/src/output/
/// global_info.rs`); the pilfflonk ones go where the STARK has its own.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilfflonkGlobalInfo {
    pub name: String,
    /// The AIRs of each airgroup, in the pilout's order.
    pub airs: Vec<Vec<GlobalInfoAir>>,
    pub air_groups: Vec<String>,
    /// The airgroup values of each airgroup.
    #[serde(rename = "aggTypes")]
    pub agg_types: Vec<Vec<AggType>>,
    pub backend: Backend,
    #[serde(rename = "formatVersion")]
    pub format_version: u64,
    pub field: Field,
    pub modulus: Modulus,
    pub transcript: Transcript,
    #[serde(rename = "setupParams")]
    pub setup_params: SetupParams,
    #[serde(rename = "nPublics")]
    pub n_publics: u64,
    /// The challenges of each stage, as the pilout counts them.
    #[serde(rename = "numChallenges")]
    pub num_challenges: Vec<u64>,
    #[serde(rename = "numProofValues")]
    pub num_proof_values: Vec<u64>,
    #[serde(rename = "proofValuesMap")]
    pub proof_values_map: Vec<NameStageEntry>,
    #[serde(rename = "publicsMap")]
    pub publics_map: Vec<NameStageEntry>,
}

/// An AIR in the globalInfo.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalInfoAir {
    pub name: String,
    /// `N = 2^nBits`.
    pub num_rows: u64,
}

/// An airgroup value in the globalInfo: how it aggregates, and its stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggType {
    #[serde(rename = "aggType")]
    pub agg_type: u64,
    pub stage: u64,
}

/// The parameters `setup-pilfflonk` ran with (pilfflonk/docs/README.md#setup-pilfflonk), which fix
/// the layout and the degrees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SetupParams {
    /// `--max-constraint-degree` (pilfflonk/docs/protocol.md#degree-search).
    pub max_constraint_degree: u64,
    /// `--extra-muls` (pilfflonk/docs/protocol.md#grouping-rules, rule 3).
    pub extra_muls: u64,
    /// `--max-q-degree`: 0 does not split `Q` (pilfflonk/docs/protocol.md#q-pieces).
    pub max_q_degree: u64,
    /// `false` with `--no-packing`, which forces `k = 1` (pilfflonk/docs/protocol.md#unpacked-layout).
    pub packing: bool,
}

/// A file of an AIR, in `<provingKey>/<name>/<airgroup>/airs/<air>/air/`
/// (pilfflonk/docs/formats.md#provingkey).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AirFile {
    /// `<air>.const`: the fixed columns.
    Const,
    /// `<air>.pilfflonkinfo.json`.
    PilfflonkInfo,
    /// `<air>.expressionsinfo.json`.
    ExpressionsInfo,
    /// `<air>.verifierinfo.json`.
    VerifierInfo,
    /// `<air>.bin`: the prover's bytecode.
    Bin,
    /// `<air>.verkey.json`: the commitments of the fixed `f_i`.
    Verkey,
}

impl AirFile {
    pub fn extension(self) -> &'static str {
        match self {
            AirFile::Const => "const",
            AirFile::PilfflonkInfo => "pilfflonkinfo.json",
            AirFile::ExpressionsInfo => "expressionsinfo.json",
            AirFile::VerifierInfo => "verifierinfo.json",
            AirFile::Bin => "bin",
            AirFile::Verkey => "verkey.json",
        }
    }
}

impl PilfflonkGlobalInfo {
    /// Reads `<proving_key>/pilout.globalInfo.json`.
    pub fn from_proving_key(proving_key: &Path) -> PilfflonkResult<Self> {
        Self::read(&proving_key.join(GLOBAL_INFO_FILE))
    }

    /// `<proving_key>/<name>/pilfflonk/`.
    pub fn backend_dir(&self, proving_key: &Path) -> PathBuf {
        proving_key.join(&self.name).join(BACKEND_DIR)
    }

    pub fn srs_path(&self, proving_key: &Path) -> PathBuf {
        self.backend_dir(proving_key).join(SRS_FILE)
    }

    pub fn vkey_path(&self, proving_key: &Path) -> PathBuf {
        self.backend_dir(proving_key).join(VKEY_FILE)
    }

    pub fn air(&self, airgroup_id: u64, air_id: u64) -> PilfflonkResult<&GlobalInfoAir> {
        let air = usize::try_from(airgroup_id)
            .ok()
            .and_then(|ag| self.airs.get(ag))
            .and_then(|airs| usize::try_from(air_id).ok().and_then(|a| airs.get(a)));
        match air {
            Some(air) => Ok(air),
            None => invalid!("the globalInfo has no air {air_id} in airgroup {airgroup_id}"),
        }
    }

    /// `<proving_key>/<name>/<airgroup>/airs/<air>/air/`.
    pub fn air_dir(&self, proving_key: &Path, airgroup_id: u64, air_id: u64) -> PilfflonkResult<PathBuf> {
        let air = self.air(airgroup_id, air_id)?;
        let Some(airgroup) = usize::try_from(airgroup_id).ok().and_then(|ag| self.air_groups.get(ag)) else {
            return invalid!("the globalInfo has no airgroup {airgroup_id}");
        };
        Ok(proving_key.join(&self.name).join(airgroup).join("airs").join(&air.name).join("air"))
    }

    /// `<air_dir>/<air>.<extension>`.
    pub fn air_file(
        &self,
        proving_key: &Path,
        airgroup_id: u64,
        air_id: u64,
        file: AirFile,
    ) -> PilfflonkResult<PathBuf> {
        let air = self.air(airgroup_id, air_id)?;
        let name = format!("{}.{}", air.name, file.extension());
        Ok(self.air_dir(proving_key, airgroup_id, air_id)?.join(name))
    }
}

impl JsonFile for PilfflonkGlobalInfo {
    fn validate(&self) -> PilfflonkResult<()> {
        if self.format_version != FORMAT_VERSION {
            return invalid!(
                "formatVersion {} is not supported; this build reads {FORMAT_VERSION}",
                self.format_version
            );
        }
        if self.airs.len() != self.air_groups.len() || self.agg_types.len() != self.air_groups.len() {
            return invalid!(
                "airs ({}), air_groups ({}) and aggTypes ({}) must have an entry per airgroup",
                self.airs.len(),
                self.air_groups.len(),
                self.agg_types.len()
            );
        }
        for (airgroup, airs) in self.air_groups.iter().zip(&self.airs) {
            for air in airs {
                let n_bits_ok = air.num_rows.is_power_of_two() && air.num_rows.trailing_zeros() as u64 <= MAX_NBITS;
                if !n_bits_ok {
                    return invalid!(
                        "air {airgroup}/{} has {} rows, which is not a power of two of at most 2^{MAX_NBITS}",
                        air.name,
                        air.num_rows
                    );
                }
            }
        }
        if self.n_publics != self.publics_map.len() as u64 {
            return invalid!("nPublics is {} but publicsMap has {} entries", self.n_publics, self.publics_map.len());
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::PilfflonkError;

    pub(crate) fn sample() -> PilfflonkGlobalInfo {
        PilfflonkGlobalInfo {
            name: "build".into(),
            airs: vec![vec![GlobalInfoAir { name: "Sample".into(), num_rows: 16 }]],
            air_groups: vec!["Sample".into()],
            agg_types: vec![vec![AggType { agg_type: 0, stage: 2 }]],
            backend: Backend,
            format_version: FORMAT_VERSION,
            field: Field,
            modulus: Modulus,
            transcript: Transcript,
            setup_params: SetupParams { max_constraint_degree: 9, extra_muls: 2, max_q_degree: 0, packing: true },
            n_publics: 2,
            num_challenges: vec![0, 2],
            num_proof_values: vec![1],
            proof_values_map: vec![NameStageEntry { name: "pv".into(), stage: 1, lengths: vec![] }],
            publics_map: vec![
                NameStageEntry { name: "in".into(), stage: 1, lengths: vec![] },
                NameStageEntry { name: "out".into(), stage: 1, lengths: vec![] },
            ],
        }
    }

    /// The keys of the top-level object of a file this crate wrote, in the order they are written.
    pub(crate) fn top_level_keys(text: &str) -> Vec<String> {
        text.lines()
            .filter_map(|line| line.strip_prefix(" \""))
            .filter_map(|rest| rest.split_once("\":").map(|(key, _)| key.to_string()))
            .collect()
    }

    #[test]
    fn it_writes_the_fields_of_a6_in_a_fixed_order() {
        let text = sample().to_json_string().unwrap();
        let expected = [
            "name",
            "airs",
            "air_groups",
            "aggTypes",
            "backend",
            "formatVersion",
            "field",
            "modulus",
            "transcript",
            "setupParams",
            "nPublics",
            "numChallenges",
            "numProofValues",
            "proofValuesMap",
            "publicsMap",
        ];
        assert_eq!(top_level_keys(&text), expected);
        for absent in ["hash", "curve", "transcriptArity", "aggregationArity", "latticeSize", "hasCompressedFinal"] {
            assert!(!text.contains(&format!("\"{absent}\"")), "{absent} must not be written");
        }
        assert!(text.contains(&format!("\"modulus\": \"{}\"", crate::field::BN254_R)));
        assert!(text.contains("\"backend\": \"pilfflonk\"") && text.contains("\"transcript\": \"keccak256\""));
    }

    #[test]
    fn it_round_trips_byte_for_byte() {
        let text = sample().to_json_string().unwrap();
        assert_eq!(sample().to_json_string().unwrap(), text);
        let back = PilfflonkGlobalInfo::from_json_str(&text).unwrap();
        assert_eq!(back, sample());
        assert_eq!(back.to_json_string().unwrap(), text);
    }

    #[test]
    fn it_refuses_other_backends_and_versions() {
        let text = sample().to_json_string().unwrap();
        for (from, to) in [
            ("\"backend\": \"pilfflonk\"", "\"backend\": \"stark\""),
            ("\"field\": \"bn254\"", "\"field\": \"goldilocks\""),
            ("\"transcript\": \"keccak256\"", "\"transcript\": \"poseidon2\""),
            ("\"formatVersion\": 1", "\"formatVersion\": 2"),
            (crate::field::BN254_R, crate::field::BN254_Q),
            ("\"nPublics\": 2", "\"nPublics\": 3"),
            ("\"num_rows\": 16", "\"num_rows\": 12"),
            ("\"num_rows\": 16", "\"num_rows\": 536870912"),
            ("\"name\": \"build\"", "\"name\": \"build\", \"hash\": \"Poseidon2\""),
        ] {
            assert!(text.contains(from), "{from}");
            let bad = text.replacen(from, to, 1);
            assert!(PilfflonkGlobalInfo::from_json_str(&bad).is_err(), "{to} must be refused");
        }
    }

    #[test]
    fn it_refuses_a_stark_global_info() {
        let stark = r#"{"name": "t", "airs": [[{"name": "A", "num_rows": 16}]], "air_groups": ["G"],
            "aggTypes": [[]], "curve": "None", "latticeSize": 368, "transcriptArity": 4,
            "aggregationArity": 3, "hasCompressedFinal": true, "nPublics": 0, "numChallenges": [0],
            "numProofValues": [0], "proofValuesMap": [], "publicsMap": [], "hash": "Poseidon2"}"#;
        assert!(matches!(PilfflonkGlobalInfo::from_json_str(stark), Err(PilfflonkError::Json(_))));
    }

    #[test]
    fn it_places_the_files_as_the_stark_setup_does() {
        let gi = sample();
        let pk = Path::new("/pk");
        assert_eq!(gi.vkey_path(pk), Path::new("/pk/build/pilfflonk/pilfflonk.vkey.json"));
        assert_eq!(gi.srs_path(pk), Path::new("/pk/build/pilfflonk/pilfflonk.srs.bin"));
        assert_eq!(
            gi.air_file(pk, 0, 0, AirFile::PilfflonkInfo).unwrap(),
            Path::new("/pk/build/Sample/airs/Sample/air/Sample.pilfflonkinfo.json")
        );
        assert_eq!(
            gi.air_file(pk, 0, 0, AirFile::Const).unwrap(),
            Path::new("/pk/build/Sample/airs/Sample/air/Sample.const")
        );
        assert!(gi.air_dir(pk, 0, 1).is_err() && gi.air_dir(pk, 1, 0).is_err());
    }
}
