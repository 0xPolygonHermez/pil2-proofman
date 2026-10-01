//! The wrap's witness: the final circuit's circom witness, computed by its witness calculator from a
//! zkin, and the stage-1 columns of the wrap's AIR its exec gathers out of it.

use std::ffi::c_void;
use std::path::Path;

use proofman_common::exec_format::ExecFile;
use proofman_common::final_witness::{FinalWitnessLibrary, FINAL_WITNESS_VALUE_BYTES};
use proofman_fields::Bn254;
use proofman_pilfflonk::{AirInstanceRef, AirShape, FrBytes, InstanceWitness, Stage1Witness, Witness, WitnessShape};
use proofman_util::{timer_start_info, timer_stop_and_log_info};
use rayon::prelude::*;

use crate::artifacts::WrapArtifacts;
use crate::error::{WrapWitnessError, WrapWitnessResult};
use crate::zkin::Zkin;

/// The final circuit's witness calculator and exec, loaded: what computes the witness of the wrap's
/// AIR from a zkin, as many times as asked.
#[derive(Debug)]
pub struct WrapWitness {
    artifacts: WrapArtifacts,
    calculator: FinalWitnessLibrary,
    exec: ExecFile<Bn254>,
}

impl WrapWitness {
    /// Loads the files of `artifacts`. Refuses a file that is not there, an exec that is not one
    /// over BN254 or that has gate bands (only the STARK's trace expander rebuilds those), and a
    /// witness calculator that cannot be loaded.
    pub fn load(artifacts: &WrapArtifacts) -> WrapWitnessResult<Self> {
        for (what, path) in [
            ("witness calculator", &artifacts.witness_calculator),
            ("witness calculator's dat", &artifacts.dat),
            ("exec", &artifacts.exec),
        ] {
            if !path.is_file() {
                return Err(WrapWitnessError::Missing { what, path: path.clone() });
            }
        }
        let exec = ExecFile::<Bn254>::read(&artifacts.exec).map_err(WrapWitnessError::Exec)?;
        if !exec.bands.is_empty() {
            return Err(WrapWitnessError::Mismatch(format!(
                "the exec {} has {} gate bands, which only the STARK's trace expander rebuilds",
                artifacts.exec.display(),
                exec.bands.len()
            )));
        }
        let calculator =
            FinalWitnessLibrary::load(&artifacts.witness_calculator, &artifacts.dat).map_err(|source| {
                WrapWitnessError::Calculator { path: artifacts.witness_calculator.clone(), zkin: None, source }
            })?;
        Ok(Self { artifacts: artifacts.clone(), calculator, exec })
    }

    pub fn artifacts(&self) -> &WrapArtifacts {
        &self.artifacts
    }

    pub fn exec(&self) -> &ExecFile<Bn254> {
        &self.exec
    }

    /// The witness of the wrap's AIR, of shape `shape`, from the zkin in the file `zkin`: what the
    /// dynamic library returns for `pilfflonk prove -w`.
    pub fn witness(&self, shape: &WitnessShape, zkin: &Path) -> WrapWitnessResult<Witness> {
        wrap_air(shape)?;
        witness_from_circom(&self.exec, self.circom_witness(zkin)?, shape)
    }

    /// [`witness`](Self::witness), from a zkin in memory, as the PLONK and FFLONK wraps hand the
    /// recursivef proof to their final circuit.
    ///
    /// # Safety
    ///
    /// As [`FinalWitnessLibrary::witness`]: `zkin` must point to a live `nlohmann::json` of the
    /// nlohmann/json the witness calculator was compiled against, which nothing else uses during
    /// the call.
    pub unsafe fn witness_from_json(&self, shape: &WitnessShape, zkin: *mut c_void) -> WrapWitnessResult<Witness> {
        wrap_air(shape)?;
        witness_from_circom(&self.exec, self.circom_witness_from_json(zkin)?, shape)
    }

    /// The final circuit's witness for the zkin in the file `zkin`: a value per witness index, wire
    /// 0 the constant one (see [`Zkin::read`] for what is refused of the file).
    pub fn circom_witness(&self, zkin: &Path) -> WrapWitnessResult<Vec<Bn254>> {
        let json = Zkin::read(zkin)?;
        // SAFETY: `json` is a nlohmann::json of the calculator's nlohmann/json (src/zkin.cpp), alive
        // and only the calculator's until it returns.
        unsafe { self.compute(json.as_ptr(), Some(zkin)) }
    }

    /// [`circom_witness`](Self::circom_witness), from a zkin in memory.
    ///
    /// # Safety
    ///
    /// As [`witness_from_json`](Self::witness_from_json).
    pub unsafe fn circom_witness_from_json(&self, zkin: *mut c_void) -> WrapWitnessResult<Vec<Bn254>> {
        self.compute(zkin, None)
    }

    /// The circuit's witness for the zkin `json`, read from the file `zkin` if it was.
    ///
    /// # Safety
    ///
    /// As [`witness_from_json`](Self::witness_from_json).
    unsafe fn compute(&self, json: *mut c_void, zkin: Option<&Path>) -> WrapWitnessResult<Vec<Bn254>> {
        let path = &self.artifacts.witness_calculator;
        timer_start_info!(PILFFLONK_WRAP_CIRCOM_WITNESS);
        let bytes = self.calculator.witness(json).map_err(|source| WrapWitnessError::Calculator {
            path: path.clone(),
            zkin: zkin.map(Path::to_path_buf),
            source,
        })?;
        let witness = bytes
            .par_chunks_exact(FINAL_WITNESS_VALUE_BYTES)
            .enumerate()
            .map(|(wire, value)| {
                let mut canonical = [0u8; FINAL_WITNESS_VALUE_BYTES];
                canonical.copy_from_slice(value);
                Bn254::from_le_bytes(canonical).ok_or(wire)
            })
            .collect::<Result<Vec<_>, usize>>()
            .map_err(|wire| {
                WrapWitnessError::Mismatch(format!(
                    "the witness calculator {} wrote wire {wire} of the circuit's witness not below r",
                    path.display()
                ))
            })?;
        timer_stop_and_log_info!(PILFFLONK_WRAP_CIRCOM_WITNESS);
        Ok(witness)
    }
}

/// The witness of the wrap's AIR, of shape `shape`, from `circom`, the final circuit's witness: the
/// stage-1 columns and the publics `exec` gathers out of it ([`ExecFile::committed_pols`]), with no
/// air values and no proof values.
///
/// The shape is the key's ([`ProvingKey::witness_shape`](proofman_pilfflonk::ProvingKey::witness_shape)):
/// it must have one AIR, with no stage-1 air values and no stage-1 proof values, as the wrap's
/// has; its publics are wires `1 ..= nPublics` of `circom`; and its trace must hold the exec's map.
pub fn witness_from_circom(
    exec: &ExecFile<Bn254>,
    circom: Vec<Bn254>,
    shape: &WitnessShape,
) -> WrapWitnessResult<Witness> {
    let air = wrap_air(shape)?;
    timer_start_info!(PILFFLONK_WRAP_EXEC);
    let n_rows = 1usize << air.n_bits;
    let pols = exec.committed_pols(circom, shape.n_publics(), n_rows, air.n_cols).map_err(WrapWitnessError::Exec)?;
    let stage1 = Stage1Witness::from_rows(n_rows, air.n_cols, &pols.trace, vec![])?;
    timer_stop_and_log_info!(PILFFLONK_WRAP_EXEC);
    Ok(Witness {
        instances: vec![InstanceWitness {
            air: AirInstanceRef { airgroup_id: air.airgroup_id, air_id: air.air_id },
            stage1,
        }],
        publics: pols.publics.into_iter().map(FrBytes::from).collect(),
        proof_values: vec![],
    })
}

/// The AIR of the wrap's key: its only one, with no stage-1 air values, in a key with no stage-1
/// proof values. `WitnessShape::new` keeps its rows within `2^28`.
fn wrap_air(shape: &WitnessShape) -> WrapWitnessResult<&AirShape> {
    let mismatch = |e: String| Err(WrapWitnessError::Mismatch(e));
    let [air] = shape.airs() else {
        return mismatch(format!("the key has {} AIRs, and the wrap's witness is of one", shape.airs().len()));
    };
    if air.n_air_values != 0 {
        return mismatch(format!(
            "air {}/{} has {} stage-1 air values, and the wrap's witness has none",
            air.airgroup_id, air.air_id, air.n_air_values
        ));
    }
    if shape.n_proof_values() != 0 {
        return mismatch(format!(
            "the key has {} stage-1 proof values, and the wrap's witness has none",
            shape.n_proof_values()
        ));
    }
    Ok(air)
}
