//! Where the wrap's witness comes from: the final circuit's files ([`WrapArtifacts`]), and, for the
//! dynamic library, the JSON of `-i` that names them and the zkin ([`WrapInputs`]).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{WrapWitnessError, WrapWitnessResult};

/// The final circuit's files the wrap's witness is computed from, which setup-snark leaves in
/// `provingKeySnark/final/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapArtifacts {
    /// The circuit's witness calculator, `final.so` (`final.dylib` on macOS), which setup-snark
    /// builds from the circuit's circom C++ and `setup/final_snark_circom/`.
    pub witness_calculator: PathBuf,
    /// The calculator's `final.dat`, which circom writes beside the C++.
    pub dat: PathBuf,
    /// The BN254 `.exec` plonk2pil writes for the circuit: its additions, and the map of the AIR's
    /// stage-1 columns.
    pub exec: PathBuf,
}

impl WrapArtifacts {
    /// `<stem>.so` (`.dylib` on macOS), `<stem>.dat` and `<stem>.exec`, as
    /// `proofman::generate_witness_final_snark` names the first two: setup-snark's stem is
    /// `provingKeySnark/final/final`.
    pub fn with_stem(stem: &Path) -> Self {
        let file = |extension: &str| {
            let mut name = stem.as_os_str().to_os_string();
            name.push(extension);
            PathBuf::from(name)
        };
        let library = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
        Self { witness_calculator: file(library), dat: file(".dat"), exec: file(".exec") }
    }
}

/// The JSON file the dynamic library reads from `-i` (`--public-inputs`): the zkin, and the files of
/// [`WrapArtifacts`].
///
/// ```json
/// {
///  "zkin": "recursivef.zkin.json",
///  "witnessCalculator": "provingKeySnark/final/final.so",
///  "dat": "provingKeySnark/final/final.dat",
///  "exec": "provingKeySnark/final/final.exec"
/// }
/// ```
///
/// Every field is required and none other is taken. A relative path is relative to the directory of
/// the JSON file, not to the working directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapInputs {
    /// The zkin: the final circuit's inputs, as a JSON object.
    pub zkin: PathBuf,
    pub artifacts: WrapArtifacts,
}

/// The JSON of [`WrapInputs`].
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WrapInputsJson {
    zkin: PathBuf,
    witness_calculator: PathBuf,
    dat: PathBuf,
    exec: PathBuf,
}

impl WrapInputs {
    /// Reads the JSON file at `path` (see the type).
    pub fn read(path: &Path) -> WrapWitnessResult<Self> {
        let text = fs::read(path).map_err(|source| WrapWitnessError::Io { path: path.to_path_buf(), source })?;
        let json: WrapInputsJson = serde_json::from_slice(&text)
            .map_err(|source| WrapWitnessError::Json { path: path.to_path_buf(), source })?;
        let dir = path.parent().unwrap_or(Path::new(""));
        Ok(Self {
            zkin: dir.join(json.zkin),
            artifacts: WrapArtifacts {
                witness_calculator: dir.join(json.witness_calculator),
                dat: dir.join(json.dat),
                exec: dir.join(json.exec),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_of_a_stem_are_its_library_dat_and_exec() {
        let artifacts = WrapArtifacts::with_stem(Path::new("provingKeySnark/final/final"));
        let library = if cfg!(target_os = "macos") { "final.dylib" } else { "final.so" };
        assert_eq!(artifacts.witness_calculator, Path::new("provingKeySnark/final").join(library));
        assert_eq!(artifacts.dat, Path::new("provingKeySnark/final/final.dat"));
        assert_eq!(artifacts.exec, Path::new("provingKeySnark/final/final.exec"));
    }

    /// Relative paths are the JSON file's directory's, absolute ones are kept; a field missing or
    /// one too many is refused, with the file.
    #[test]
    fn the_inputs_name_their_files_from_their_own_directory() {
        let dir = std::env::temp_dir().join(format!("pilfflonk_wrap_witness_inputs_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("inputs.json");
        fs::write(
            &path,
            r#"{"zkin": "/proofs/zkin.json", "witnessCalculator": "final/final.so", "dat": "final/final.dat",
                "exec": "../final.exec"}"#,
        )
        .unwrap();
        let inputs = WrapInputs::read(&path).unwrap();
        assert_eq!(inputs.zkin, Path::new("/proofs/zkin.json"));
        assert_eq!(inputs.artifacts.witness_calculator, dir.join("final/final.so"));
        assert_eq!(inputs.artifacts.dat, dir.join("final/final.dat"));
        assert_eq!(inputs.artifacts.exec, dir.join("../final.exec"));

        for text in [
            r#"{"zkin": "z", "witnessCalculator": "f.so", "dat": "f.dat"}"#,
            r#"{"zkin": "z", "witnessCalculator": "f.so", "dat": "f.dat", "exec": "f.exec", "n": 1}"#,
            "zkin.json",
        ] {
            fs::write(&path, text).unwrap();
            match WrapInputs::read(&path) {
                Err(WrapWitnessError::Json { path: at, .. }) => assert_eq!(at, path),
                other => panic!("{text}: {other:?}"),
            }
        }
        assert!(matches!(WrapInputs::read(&dir.join("none.json")), Err(WrapWitnessError::Io { .. })));
        fs::remove_dir_all(&dir).unwrap();
    }
}
