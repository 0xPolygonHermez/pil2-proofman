//! The witness the prover takes from outside (spec §4.3): the stage-1 columns and the stage-1 air
//! values of each instance, the publics and the stage-1 proof values. The prover reads it through
//! [`WitnessSource`]; stages 2 and later, and the im pols, are the prover's to compute.
//!
//! In phases 1 and 2 it comes from a directory (plan N9), read by [`FileWitnessSource`] and
//! written by [`Witness::write`], which the fixture generators call. Phase 3 adds a source that
//! computes it over `Fr` (D4).
//!
//! # The witness directory (version 1)
//!
//! A directory that holds exactly these files, and nothing else:
//!
//! | File | Content |
//! |---|---|
//! | `instances.json` | the instances, in canonical order, and their stage-1 air values |
//! | `instance_<ag>_<a>_<t>.bin` | the stage-1 columns of one instance |
//! | `publics.json` | the publics |
//! | `proof_values.json` | the stage-1 proof values |
//!
//! - **`instances.json`**: a JSON array with one object per instance, at least one:
//!   `{"airgroupId": ag, "airId": a, "airValues": ["v", …]}`, with no other field.
//!   - **Order**: canonical (spec glossary), non-decreasing in `(airgroupId, airId)`. The
//!     instances of an AIR are consecutive, and the `t`-th of them (from 0) is instance `t` of
//!     that AIR.
//!   - **`airValues`**: the air values of stage 1 of the instance, in the order their entries of
//!     stage 1 have in the AIR's `airValuesMap`. Empty in v1 (D2).
//! - **`instance_<ag>_<a>_<t>.bin`**: one per instance, named by its `airgroupId`, `airId` and `t`
//!   in decimal (`instance_0_0_0.bin`). Raw bytes, with no header: the `N = 2^nBits` rows of the
//!   instance, row after row, each the values of its `C` stage-1 columns in order, each value 32
//!   bytes little-endian in canonical form (`< r`). The value of column `c` at row `i` is at byte
//!   `(i·C + c)·32`, and the file has exactly `N·C·32` bytes.
//!   - **The columns**: column `c` is the pilout's `WitnessCol { stage: 1, colIdx: c }`, and `C`
//!     is the AIR's `stageWidths[0]`. In the pilfflonkinfo they are the entries of stage 1 of
//!     `cmPolsMap` that are not im pols, and column `c` is the one with `stageId` `c`. The im pols
//!     of stage 1 are not in the file.
//! - **`publics.json`**: the `nPublics` publics, in the order of the globalInfo's `publicsMap`,
//!   as [`Publics`] writes them.
//! - **`proof_values.json`**: the proof values of stage 1, in the order their entries of stage 1
//!   have in the globalInfo's `proofValuesMap`. Empty in v1 (D2).
//!
//! Every value in the JSON files is a decimal string, in the one spelling `crate::field` reads (no
//! sign, spaces or leading zeros), below `r`. The JSON files have the layout of every
//! [`JsonFile`] (`JSON.stringify(value, null, 1)`).
//!
//! **Checks.** A directory is checked against the [`WitnessShape`] of the AIRs it is for: from the
//! `provingKey/` ([`WitnessShape::from_proving_key`]) for the prover, or from the pilout in tests.
//! [`FileWitnessSource::open`] refuses a directory with a file too many or one missing, an
//! instance of an AIR the shape does not have, instances out of canonical order, a file of the
//! wrong size, and counts of air values, publics or proof values other than the shape's. Each
//! `.bin` is read, and each of its values checked to be below `r`, when the prover asks for it
//! ([`WitnessSource::stage1`]). [`Witness::write`] makes the same checks before it writes, so it
//! only writes directories `open` accepts.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::{FrBytes, FIELD_BYTES};
use crate::global_info::{PilfflonkGlobalInfo, MAX_NBITS};
use crate::json::JsonFile;
use crate::pilfflonk_info::PilfflonkInfo;
use crate::proof::{Publics, PUBLICS_FILE};

/// The file name of the instances.
pub const INSTANCES_FILE: &str = "instances.json";

/// The file name of the stage-1 proof values.
pub const PROOF_VALUES_FILE: &str = "proof_values.json";

/// Where the witness of the prover comes from (spec §5.3).
///
/// The sketch of §5.3, with the crate's errors: a source validates what it holds, and
/// [`Stage1Witness`] carries the trace as the bytes `pilfflonk_instance_new` takes, with the
/// stage-1 air values of the instance.
pub trait WitnessSource {
    /// The instances, in canonical order.
    fn instances(&self) -> Vec<AirInstanceRef>;

    /// The stage-1 columns and air values of instance `instance`, its index in `instances()`.
    fn stage1(&self, instance: usize) -> PilfflonkResult<Stage1Witness>;

    /// The publics, in the order of the globalInfo's `publicsMap`.
    fn publics(&self) -> PilfflonkResult<Vec<FrBytes>>;

    /// The stage-1 proof values, in the order of the globalInfo's `proofValuesMap`.
    fn proof_values(&self) -> PilfflonkResult<Vec<FrBytes>>;
}

/// The AIR an instance is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AirInstanceRef {
    pub airgroup_id: u64,
    pub air_id: u64,
}

/// The stage-1 witness of an instance: its `n_rows × n_cols` columns, as the bytes of its `.bin`
/// (see the module), and its stage-1 air values. Every value is below `r`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Witness {
    n_rows: usize,
    n_cols: usize,
    trace: Vec<u8>,
    air_values: Vec<FrBytes>,
}

fn trace_len(n_rows: usize, n_cols: usize) -> PilfflonkResult<usize> {
    match n_rows.checked_mul(n_cols).and_then(|n| n.checked_mul(FIELD_BYTES)) {
        Some(len) => Ok(len),
        None => invalid!("a trace of {n_rows} rows and {n_cols} columns does not fit in memory"),
    }
}

impl Stage1Witness {
    /// From the bytes of a trace, row after row: refuses a length other than `n_rows·n_cols·32`
    /// and a value that is not below `r`.
    pub fn new(n_rows: usize, n_cols: usize, trace: Vec<u8>, air_values: Vec<FrBytes>) -> PilfflonkResult<Self> {
        let len = trace_len(n_rows, n_cols)?;
        if trace.len() != len {
            return invalid!(
                "a trace of {n_rows} rows and {n_cols} columns has {len} bytes, and this one has {}",
                trace.len()
            );
        }
        for (i, chunk) in trace.chunks_exact(FIELD_BYTES).enumerate() {
            let mut bytes = [0u8; FIELD_BYTES];
            bytes.copy_from_slice(chunk);
            if FrBytes::from_le_bytes(bytes).is_err() {
                return invalid!("the value of row {}, column {} is not below r", i / n_cols, i % n_cols);
            }
        }
        Ok(Self { n_rows, n_cols, trace, air_values })
    }

    /// From its columns, each of `n_rows` values.
    pub fn from_columns(n_rows: usize, columns: &[Vec<FrBytes>], air_values: Vec<FrBytes>) -> PilfflonkResult<Self> {
        if let Some(c) = columns.iter().position(|column| column.len() != n_rows) {
            return invalid!("column {c} has {} values, not one per row ({n_rows})", columns[c].len());
        }
        let n_cols = columns.len();
        let mut trace = Vec::with_capacity(trace_len(n_rows, n_cols)?);
        for row in 0..n_rows {
            for column in columns {
                trace.extend_from_slice(&column[row].to_le_bytes());
            }
        }
        Ok(Self { n_rows, n_cols, trace, air_values })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn offset(&self, row: usize, col: usize) -> Option<usize> {
        (row < self.n_rows && col < self.n_cols).then(|| (row * self.n_cols + col) * FIELD_BYTES)
    }

    /// The value of column `col` at row `row`.
    pub fn get(&self, row: usize, col: usize) -> Option<FrBytes> {
        let at = self.offset(row, col)?;
        let mut bytes = [0u8; FIELD_BYTES];
        bytes.copy_from_slice(&self.trace[at..at + FIELD_BYTES]);
        // Every value was checked on the way in.
        FrBytes::from_le_bytes(bytes).ok()
    }

    pub fn set(&mut self, row: usize, col: usize, value: FrBytes) -> PilfflonkResult<()> {
        let Some(at) = self.offset(row, col) else {
            return invalid!(
                "a trace of {} rows and {} columns has no row {row}, column {col}",
                self.n_rows,
                self.n_cols
            );
        };
        self.trace[at..at + FIELD_BYTES].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Column `col`, row by row.
    pub fn column(&self, col: usize) -> Option<Vec<FrBytes>> {
        (col < self.n_cols).then(|| (0..self.n_rows).filter_map(|row| self.get(row, col)).collect())
    }

    /// The trace as its `.bin` has it: what `pilfflonk_instance_new` takes as `stage1`.
    pub fn trace_bytes(&self) -> &[u8] {
        &self.trace
    }

    pub fn air_values(&self) -> &[FrBytes] {
        &self.air_values
    }
}

/// What a witness must look like for a set of AIRs: the size of the stage-1 trace and the number
/// of stage-1 air values of each AIR, and the numbers of publics and of stage-1 proof values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessShape {
    airs: Vec<AirShape>,
    n_publics: usize,
    n_proof_values: usize,
}

/// The shape of the witness of an instance of an AIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AirShape {
    pub airgroup_id: u64,
    pub air_id: u64,
    /// `N = 2^n_bits` rows.
    pub n_bits: u64,
    /// The stage-1 columns (see the module).
    pub n_cols: usize,
    /// The stage-1 air values.
    pub n_air_values: usize,
}

impl AirShape {
    /// `2^n_bits`, for a shape `WitnessShape::new` accepted.
    fn n_rows(&self) -> usize {
        1 << self.n_bits
    }

    fn air(&self) -> AirInstanceRef {
        AirInstanceRef { airgroup_id: self.airgroup_id, air_id: self.air_id }
    }
}

impl WitnessShape {
    /// Refuses an AIR twice and an AIR of more than `2^28` rows.
    pub fn new(airs: Vec<AirShape>, n_publics: usize, n_proof_values: usize) -> PilfflonkResult<Self> {
        let mut seen = BTreeSet::new();
        for air in &airs {
            if !seen.insert(air.air()) {
                return invalid!("air {}/{} is in the shape twice", air.airgroup_id, air.air_id);
            }
            if air.n_bits > MAX_NBITS {
                return invalid!(
                    "air {}/{} has 2^{} rows, above 2^{MAX_NBITS}",
                    air.airgroup_id,
                    air.air_id,
                    air.n_bits
                );
            }
            trace_len(air.n_rows(), air.n_cols)?;
        }
        Ok(Self { airs, n_publics, n_proof_values })
    }

    /// The shape of the witness of a `provingKey/`, for the AIRs of `infos`, each the
    /// pilfflonkinfo of an AIR of `global_info` as read (and so validated).
    pub fn from_proving_key(global_info: &PilfflonkGlobalInfo, infos: &[&PilfflonkInfo]) -> PilfflonkResult<Self> {
        let mut airs = Vec::with_capacity(infos.len());
        for info in infos {
            let air = global_info.air(info.airgroup_id, info.air_id)?;
            if air.name != info.name || Some(air.num_rows) != 1u64.checked_shl(info.n_bits as u32) {
                return invalid!(
                    "air {}/{} is {} of {} rows in the globalInfo, and its pilfflonkinfo is {} of 2^{}",
                    info.airgroup_id,
                    info.air_id,
                    air.name,
                    air.num_rows,
                    info.name,
                    info.n_bits
                );
            }
            // The columns of the file are the stage-1 ones the setup did not add: they must come
            // first in the stage, in the pilout's order, which is what `stageId` numbers.
            let stage_ids: Vec<u64> =
                info.cm_pols_map.iter().filter(|p| p.stage == 1 && !p.im_pol).map(|p| p.stage_id).collect();
            if stage_ids.iter().enumerate().any(|(c, id)| *id != c as u64) {
                return invalid!(
                    "the stage-1 columns of {} that are not im pols have stageIds {stage_ids:?}, not 0 … {}",
                    info.name,
                    stage_ids.len().saturating_sub(1)
                );
            }
            airs.push(AirShape {
                airgroup_id: info.airgroup_id,
                air_id: info.air_id,
                n_bits: info.n_bits,
                n_cols: stage_ids.len(),
                n_air_values: info.air_values_map.iter().filter(|v| v.stage == 1).count(),
            });
        }
        let n_publics = usize::try_from(global_info.n_publics).unwrap_or(usize::MAX);
        let n_proof_values = global_info.proof_values_map.iter().filter(|v| v.stage == 1).count();
        Self::new(airs, n_publics, n_proof_values)
    }

    pub fn airs(&self) -> &[AirShape] {
        &self.airs
    }

    pub fn n_publics(&self) -> usize {
        self.n_publics
    }

    pub fn n_proof_values(&self) -> usize {
        self.n_proof_values
    }

    /// The shape of the instances of an AIR.
    pub fn air(&self, air: AirInstanceRef) -> PilfflonkResult<&AirShape> {
        match self.airs.iter().find(|shape| shape.air() == air) {
            Some(shape) => Ok(shape),
            None => invalid!(
                "the witness has an instance of air {}/{}, which is not one of its AIRs",
                air.airgroup_id,
                air.air_id
            ),
        }
    }

    /// Checks the instances and their air values, and returns the file name of each instance.
    fn check_instances(&self, instances: &[(AirInstanceRef, usize)]) -> PilfflonkResult<Vec<String>> {
        if instances.is_empty() {
            return invalid!("a witness has one instance at least");
        }
        let mut names = Vec::with_capacity(instances.len());
        let mut previous: Option<AirInstanceRef> = None;
        let mut index_in_air = 0;
        for (i, &(air, n_air_values)) in instances.iter().enumerate() {
            match previous {
                Some(last) if last == air => index_in_air += 1,
                Some(last) if last > air => {
                    return invalid!(
                        "the instances are not in canonical order: instance {i} is of air {}/{}, after one of air {}/{}",
                        air.airgroup_id,
                        air.air_id,
                        last.airgroup_id,
                        last.air_id
                    );
                }
                _ => index_in_air = 0,
            }
            previous = Some(air);
            let shape = self.air(air)?;
            if n_air_values != shape.n_air_values {
                return invalid!(
                    "instance {i} has {n_air_values} air values, and its air {}/{} has {} of stage 1",
                    air.airgroup_id,
                    air.air_id,
                    shape.n_air_values
                );
            }
            names.push(instance_file_name(air, index_in_air));
        }
        Ok(names)
    }

    fn check_counts(&self, n_publics: usize, n_proof_values: usize) -> PilfflonkResult<()> {
        if n_publics != self.n_publics {
            return invalid!("the witness has {n_publics} publics, and nPublics is {}", self.n_publics);
        }
        if n_proof_values != self.n_proof_values {
            return invalid!(
                "the witness has {n_proof_values} proof values, and there are {} of stage 1",
                self.n_proof_values
            );
        }
        Ok(())
    }

    /// Checks a whole witness: what `FileWitnessSource::open` checks of a directory, and the size
    /// of every trace.
    pub fn check(&self, witness: &Witness) -> PilfflonkResult<()> {
        self.check_witness(witness).map(|_| ())
    }

    /// `check`, returning the file name of each instance.
    fn check_witness(&self, witness: &Witness) -> PilfflonkResult<Vec<String>> {
        let instances: Vec<_> = witness.instances.iter().map(|i| (i.air, i.stage1.air_values.len())).collect();
        let names = self.check_instances(&instances)?;
        for (i, instance) in witness.instances.iter().enumerate() {
            let shape = self.air(instance.air)?;
            if (instance.stage1.n_rows, instance.stage1.n_cols) != (shape.n_rows(), shape.n_cols) {
                return invalid!(
                    "instance {i} has {} rows and {} columns, and its air {}/{} has {} and {}",
                    instance.stage1.n_rows,
                    instance.stage1.n_cols,
                    shape.airgroup_id,
                    shape.air_id,
                    shape.n_rows(),
                    shape.n_cols
                );
            }
        }
        self.check_counts(witness.publics.len(), witness.proof_values.len())?;
        Ok(names)
    }
}

/// The file name of instance `index_in_air` of AIR `air`.
pub fn instance_file_name(air: AirInstanceRef, index_in_air: usize) -> String {
    format!("instance_{}_{}_{index_in_air}.bin", air.airgroup_id, air.air_id)
}

/// An entry of `instances.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InstanceEntry {
    pub airgroup_id: u64,
    pub air_id: u64,
    pub air_values: Vec<FrBytes>,
}

impl InstanceEntry {
    pub fn air(&self) -> AirInstanceRef {
        AirInstanceRef { airgroup_id: self.airgroup_id, air_id: self.air_id }
    }
}

/// `instances.json` (see the module). Its order and air values are checked against a shape by
/// `FileWitnessSource::open`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Instances(pub Vec<InstanceEntry>);

impl JsonFile for Instances {
    fn validate(&self) -> PilfflonkResult<()> {
        Ok(())
    }
}

/// `proof_values.json`: the stage-1 proof values, as decimal strings (see the module).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProofValues(pub Vec<FrBytes>);

impl JsonFile for ProofValues {
    fn validate(&self) -> PilfflonkResult<()> {
        Ok(())
    }
}

/// A witness in memory: what a generator writes to a directory, or a source reads into memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    /// In canonical order.
    pub instances: Vec<InstanceWitness>,
    pub publics: Vec<FrBytes>,
    pub proof_values: Vec<FrBytes>,
}

/// The witness of one instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceWitness {
    pub air: AirInstanceRef,
    pub stage1: Stage1Witness,
}

impl Witness {
    /// Everything a source holds.
    pub fn from_source(source: &impl WitnessSource) -> PilfflonkResult<Self> {
        let instances = source
            .instances()
            .into_iter()
            .enumerate()
            .map(|(i, air)| Ok(InstanceWitness { air, stage1: source.stage1(i)? }))
            .collect::<PilfflonkResult<_>>()?;
        Ok(Self { instances, publics: source.publics()?, proof_values: source.proof_values()? })
    }

    /// Writes the witness directory `dir` (see the module) after checking the witness against
    /// `shape`. `dir` is created if it does not exist, and must be empty if it does.
    pub fn write(&self, dir: &Path, shape: &WitnessShape) -> PilfflonkResult<()> {
        let names = shape.check_witness(self)?;
        fs::create_dir_all(dir).map_err(|source| io_error(dir, source))?;
        if fs::read_dir(dir).map_err(|source| io_error(dir, source))?.next().is_some() {
            return Err(PilfflonkError::InvalidFormat(
                "a witness directory holds its own files only, and this one is not empty".to_string(),
            )
            .in_file(dir));
        }
        let entries = self
            .instances
            .iter()
            .map(|i| InstanceEntry {
                airgroup_id: i.air.airgroup_id,
                air_id: i.air.air_id,
                air_values: i.stage1.air_values.clone(),
            })
            .collect();
        Instances(entries).write(&dir.join(INSTANCES_FILE))?;
        for (instance, name) in self.instances.iter().zip(names) {
            let path = dir.join(name);
            fs::write(&path, &instance.stage1.trace).map_err(|source| io_error(&path, source))?;
        }
        Publics(self.publics.clone()).write(&dir.join(PUBLICS_FILE))?;
        ProofValues(self.proof_values.clone()).write(&dir.join(PROOF_VALUES_FILE))
    }
}

impl WitnessSource for Witness {
    fn instances(&self) -> Vec<AirInstanceRef> {
        self.instances.iter().map(|i| i.air).collect()
    }

    fn stage1(&self, instance: usize) -> PilfflonkResult<Stage1Witness> {
        match self.instances.get(instance) {
            Some(i) => Ok(i.stage1.clone()),
            None => invalid!("the witness has {} instances, and no instance {instance}", self.instances.len()),
        }
    }

    fn publics(&self) -> PilfflonkResult<Vec<FrBytes>> {
        Ok(self.publics.clone())
    }

    fn proof_values(&self) -> PilfflonkResult<Vec<FrBytes>> {
        Ok(self.proof_values.clone())
    }
}

fn io_error(path: &Path, source: std::io::Error) -> PilfflonkError {
    PilfflonkError::Io { path: path.to_path_buf(), source }
}

/// A witness directory (see the module), checked against a shape when it is opened. The traces
/// are read, and their values checked, one instance at a time, by `stage1`.
#[derive(Clone, Debug)]
pub struct FileWitnessSource {
    dir: PathBuf,
    instances: Vec<FileInstance>,
    publics: Vec<FrBytes>,
    proof_values: Vec<FrBytes>,
}

#[derive(Clone, Debug)]
struct FileInstance {
    entry: InstanceEntry,
    path: PathBuf,
    n_rows: usize,
    n_cols: usize,
}

impl FileWitnessSource {
    /// Opens `dir` and checks it against `shape`: everything but the values of the traces.
    pub fn open(dir: &Path, shape: &WitnessShape) -> PilfflonkResult<Self> {
        let instances = Instances::read(&dir.join(INSTANCES_FILE))?.0;
        let checked: Vec<_> = instances.iter().map(|e| (e.air(), e.air_values.len())).collect();
        let names = shape.check_instances(&checked).map_err(|e| e.in_file(&dir.join(INSTANCES_FILE)))?;
        let publics = Publics::read(&dir.join(PUBLICS_FILE))?.0;
        let proof_values = ProofValues::read(&dir.join(PROOF_VALUES_FILE))?.0;
        shape.check_counts(publics.len(), proof_values.len()).map_err(|e| e.in_file(dir))?;

        let mut expected: BTreeSet<String> =
            [INSTANCES_FILE, PUBLICS_FILE, PROOF_VALUES_FILE].into_iter().map(str::to_string).collect();
        expected.extend(names.iter().cloned());
        for entry in fs::read_dir(dir).map_err(|source| io_error(dir, source))? {
            let entry = entry.map_err(|source| io_error(dir, source))?;
            let name = entry.file_name();
            if !name.to_str().is_some_and(|name| expected.contains(name)) {
                return Err(PilfflonkError::InvalidFormat(format!(
                    "{name:?} is not a file of this witness directory: it holds {expected:?} and nothing else"
                ))
                .in_file(dir));
            }
        }

        let mut files = Vec::with_capacity(instances.len());
        for (entry, name) in instances.into_iter().zip(names) {
            let path = dir.join(name);
            let shape = shape.air(entry.air())?;
            let (n_rows, n_cols) = (shape.n_rows(), shape.n_cols);
            let len = trace_len(n_rows, n_cols)?;
            let metadata = fs::metadata(&path).map_err(|source| io_error(&path, source))?;
            check_trace_file(&metadata, len).map_err(|e| e.in_file(&path))?;
            files.push(FileInstance { entry, path, n_rows, n_cols });
        }
        Ok(Self { dir: dir.to_path_buf(), instances: files, publics, proof_values })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

fn check_trace_file(metadata: &fs::Metadata, len: usize) -> PilfflonkResult<()> {
    if !metadata.is_file() {
        return invalid!("the trace of an instance must be a file");
    }
    if metadata.len() != len as u64 {
        return invalid!("the trace has {} bytes, and one of this AIR has {len}", metadata.len());
    }
    Ok(())
}

impl WitnessSource for FileWitnessSource {
    fn instances(&self) -> Vec<AirInstanceRef> {
        self.instances.iter().map(|i| i.entry.air()).collect()
    }

    fn stage1(&self, instance: usize) -> PilfflonkResult<Stage1Witness> {
        let Some(file) = self.instances.get(instance) else {
            return invalid!("the witness has {} instances, and no instance {instance}", self.instances.len());
        };
        let trace = fs::read(&file.path).map_err(|source| io_error(&file.path, source))?;
        Stage1Witness::new(file.n_rows, file.n_cols, trace, file.entry.air_values.clone())
            .map_err(|e| e.in_file(&file.path))
    }

    fn publics(&self) -> PilfflonkResult<Vec<FrBytes>> {
        Ok(self.publics.clone())
    }

    fn proof_values(&self) -> PilfflonkResult<Vec<FrBytes>> {
        Ok(self.proof_values.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_row_after_row_little_endian() {
        let a = vec![FrBytes::from_u64(1), FrBytes::from_u64(2)];
        let b = vec![FrBytes::from_u64(0x0304), FrBytes::from_u64(5)];
        let w = Stage1Witness::from_columns(2, &[a.clone(), b.clone()], vec![]).unwrap();
        let bytes = w.trace_bytes();
        assert_eq!(bytes.len(), 2 * 2 * 32);
        // Row 0: a[0], b[0]; row 1: a[1], b[1].
        assert_eq!((bytes[0], bytes[32], bytes[33], bytes[64], bytes[96]), (1, 4, 3, 2, 5));
        assert_eq!(w.column(0).unwrap(), a);
        assert_eq!(w.column(1).unwrap(), b);
        assert_eq!(w.get(1, 1), Some(FrBytes::from_u64(5)));
        assert_eq!(w.get(2, 0), None);
        assert_eq!(w.get(0, 2), None);
        assert_eq!(Stage1Witness::new(2, 2, bytes.to_vec(), vec![]).unwrap(), w);
    }

    #[test]
    fn a_trace_has_its_size_and_values_below_r() {
        assert!(Stage1Witness::new(2, 2, vec![0; 127], vec![]).is_err());
        let mut bytes = vec![0; 128];
        bytes[96..].copy_from_slice(&[0xff; 32]);
        let err = Stage1Witness::new(2, 2, bytes, vec![]).unwrap_err();
        assert!(err.to_string().contains("row 1, column 1"), "{err}");
        assert!(Stage1Witness::from_columns(2, &[vec![FrBytes::ZERO]], vec![]).is_err());

        let mut w = Stage1Witness::from_columns(1, &[vec![FrBytes::ZERO]], vec![]).unwrap();
        w.set(0, 0, FrBytes::from_u64(7)).unwrap();
        assert_eq!(w.get(0, 0), Some(FrBytes::from_u64(7)));
        assert!(w.set(1, 0, FrBytes::ZERO).is_err());
    }

    #[test]
    fn instance_files_are_named_by_air_and_index() {
        assert_eq!(instance_file_name(AirInstanceRef { airgroup_id: 0, air_id: 0 }, 0), "instance_0_0_0.bin");
        assert_eq!(instance_file_name(AirInstanceRef { airgroup_id: 2, air_id: 10 }, 3), "instance_2_10_3.bin");
    }
}
