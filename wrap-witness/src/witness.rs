//! The wrap's witness: the final circuit's circom witness, computed by its witness calculator from a
//! zkin, and the stage-1 columns of the wrap's AIR its exec gathers out of it, with the multiplicity
//! of its range checks counted from the rows the exec's range-check bands name.

use std::ffi::c_void;
use std::path::Path;

use proofman_common::exec_format::{ExecFile, RANGE_CHECK_BAND_KIND, RANGE_CHECK_CHUNK_BITS, RANGE_CHECK_CHUNK_COLS};
use proofman_common::final_witness::{FinalWitnessLibrary, FINAL_WITNESS_VALUE_BYTES};
use proofman_fields::Bn128;
use proofman_pilfflonk::{
    AirInstanceRef, AirShape, ExecStatic, ExecWitness, FrBytes, InstanceWitness, Stage1Witness, Witness, WitnessShape,
};
use proofman_util::{timer_start_debug, timer_stop_and_log_debug};
use rayon::prelude::*;

use crate::artifacts::WrapArtifacts;
use crate::blake3::Blake3Bands;
use crate::device::{blocks_of, ExecParts, ProofParts};
use crate::trace::{put_word, show, trace_of, word, VALUE_BYTES};
use crate::error::{WrapWitnessError, WrapWitnessResult};
use crate::zkin::Zkin;

/// The final circuit's witness calculator and exec, loaded: what computes the witness of the wrap's
/// AIR from a zkin, as many times as asked.
#[derive(Debug)]
pub struct WrapWitness {
    artifacts: WrapArtifacts,
    calculator: FinalWitnessLibrary,
    exec: ExecFile<Bn128>,
    bands: Option<Bands>,
    /// The exec's parts for the device, of a blake3 wrap.
    device: Option<ExecParts>,
}

/// The stage-1 witness of a proof of the blake3 wrap as its parts, for the GPU to build
/// ([`WrapWitness::device_witness`], [`proofman_pilfflonk::prove_exec`]).
#[derive(Debug)]
pub struct DeviceWitness {
    pub air: AirInstanceRef,
    pub publics: Vec<FrBytes>,
    proof: ProofParts,
}

impl DeviceWitness {
    /// The parts, as `prove_exec` takes them.
    pub fn exec(&self) -> ExecWitness<'_> {
        ExecParts::exec_witness(&self.proof)
    }
}

impl WrapWitness {
    /// Loads the files of `artifacts`. Refuses a file that is not there, an exec that is not one
    /// over BN128 or whose gate bands are not the wrap's range checks, and a witness calculator
    /// that cannot be loaded.
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
        let exec = ExecFile::<Bn128>::read(&artifacts.exec).map_err(WrapWitnessError::Exec)?;
        let bands = Bands::of(&exec)
            .map_err(|e| WrapWitnessError::Mismatch(format!("the exec {} {e}", artifacts.exec.display())))?;
        let calculator =
            FinalWitnessLibrary::load(&artifacts.witness_calculator, &artifacts.dat).map_err(|source| {
                WrapWitnessError::Calculator { path: artifacts.witness_calculator.clone(), zkin: None, source }
            })?;
        // With the key's load, rather than in the first proof.
        let device = match bands {
            Some(Bands::Blake3(_)) => Some(ExecParts::of(&exec)?),
            _ => None,
        };
        Ok(Self { artifacts: artifacts.clone(), calculator, exec, bands, device })
    }

    pub fn artifacts(&self) -> &WrapArtifacts {
        &self.artifacts
    }

    pub fn exec(&self) -> &ExecFile<Bn128> {
        &self.exec
    }

    /// The parts of every proof's witness on the device, of a blake3 wrap: what the key takes once
    /// (`ProvingKey::set_exec`).
    pub fn exec_static(&self) -> Option<ExecStatic<'_>> {
        self.device.as_ref().map(ExecParts::exec_static)
    }

    /// The witness of the wrap's AIR, of shape `shape`, from the zkin in the file `zkin`: what the
    /// dynamic library returns for `pilfflonk prove -w`.
    pub fn witness(&self, shape: &WitnessShape, zkin: &Path) -> WrapWitnessResult<Witness> {
        wrap_air(shape)?;
        let json = Zkin::read(zkin)?;
        // SAFETY: `json` is a nlohmann::json of the calculator's nlohmann/json (src/zkin.cpp), alive
        // and only the calculator's until it returns.
        let wires = unsafe { self.compute_bytes(json.as_ptr(), Some(zkin))? };
        witness_from_bytes(&self.exec, self.bands.as_ref(), wires, shape)
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
        witness_from_bytes(&self.exec, self.bands.as_ref(), self.compute_bytes(zkin, None)?, shape)
    }

    /// [`witness`](Self::witness), from the zkin `zkin`, a JSON object
    /// ([`gl_proof_zkin`](crate::gl_proof_zkin)'s).
    pub fn witness_from_zkin(&self, shape: &WitnessShape, zkin: &serde_json::Value) -> WrapWitnessResult<Witness> {
        wrap_air(shape)?;
        let json = Zkin::of_value(zkin)?;
        // SAFETY: `json` is a nlohmann::json of the calculator's nlohmann/json, alive and only the
        // calculator's until it returns.
        let wires = unsafe { self.compute_bytes(json.as_ptr(), None)? };
        witness_from_bytes(&self.exec, self.bands.as_ref(), wires, shape)
    }

    /// [`witness_from_zkin`](Self::witness_from_zkin) as its parts, for the GPU to build: of the
    /// blake3 wrap only, whose exec has blake3 blocks.
    pub fn device_witness(&self, shape: &WitnessShape, zkin: &serde_json::Value) -> WrapWitnessResult<DeviceWitness> {
        let air = wrap_air(shape)?;
        let (Some(Bands::Blake3(bands)), Some(parts)) = (&self.bands, &self.device) else {
            return Err(WrapWitnessError::Mismatch("the exec has no blake3 blocks for the device to build".into()));
        };
        let json = Zkin::of_value(zkin)?;
        // SAFETY: `json` is a nlohmann::json of the calculator's nlohmann/json, alive and only the
        // calculator's until it returns.
        let wires = unsafe { self.compute_bytes(json.as_ptr(), None)? };
        timer_start_debug!(PILFFLONK_WRAP_DEVICE_PARTS);
        let (publics, proof) = parts.witness(&self.exec, bands, wires, shape.n_publics())?;
        timer_stop_and_log_debug!(PILFFLONK_WRAP_DEVICE_PARTS);
        Ok(DeviceWitness { air: AirInstanceRef { airgroup_id: air.airgroup_id, air_id: air.air_id }, publics, proof })
    }

    /// The final circuit's witness for the zkin in the file `zkin`: a value per witness index, wire
    /// 0 the constant one (see [`Zkin::read`] for what is refused of the file).
    pub fn circom_witness(&self, zkin: &Path) -> WrapWitnessResult<Vec<Bn128>> {
        let json = Zkin::read(zkin)?;
        // SAFETY: `json` is a nlohmann::json of the calculator's nlohmann/json (src/zkin.cpp), alive
        // and only the calculator's until it returns.
        unsafe { self.compute(json.as_ptr(), Some(zkin)) }
    }

    /// The circuit's witness for the zkin `json`, read from the file `zkin` if it was.
    ///
    /// # Safety
    ///
    /// As [`witness_from_json`](Self::witness_from_json).
    unsafe fn compute(&self, json: *mut c_void, zkin: Option<&Path>) -> WrapWitnessResult<Vec<Bn128>> {
        let path = &self.artifacts.witness_calculator;
        let bytes = self.compute_bytes(json, zkin)?;
        bytes
            .par_chunks_exact(FINAL_WITNESS_VALUE_BYTES)
            .enumerate()
            .map(|(wire, value)| {
                let mut canonical = [0u8; FINAL_WITNESS_VALUE_BYTES];
                canonical.copy_from_slice(value);
                Bn128::from_le_bytes(canonical).ok_or(wire)
            })
            .collect::<Result<Vec<_>, usize>>()
            .map_err(|wire| not_below_r(path, wire))
    }

    /// The circuit's witness for the zkin `json`, as the calculator writes it: each value 32 bytes,
    /// canonical, little endian, each checked below `r`.
    ///
    /// # Safety
    ///
    /// As [`witness_from_json`](Self::witness_from_json).
    unsafe fn compute_bytes(&self, json: *mut c_void, zkin: Option<&Path>) -> WrapWitnessResult<Vec<u8>> {
        let path = &self.artifacts.witness_calculator;
        timer_start_debug!(PILFFLONK_WRAP_CIRCOM_WITNESS);
        let bytes = self.calculator.witness(json).map_err(|source| WrapWitnessError::Calculator {
            path: path.clone(),
            zkin: zkin.map(Path::to_path_buf),
            source,
        })?;
        if let Some(wire) = bytes.par_chunks_exact(FINAL_WITNESS_VALUE_BYTES).position_any(|value| {
            let mut canonical = [0u8; FINAL_WITNESS_VALUE_BYTES];
            canonical.copy_from_slice(value);
            FrBytes::from_le_bytes(canonical).is_err()
        }) {
            return Err(not_below_r(path, wire));
        }
        timer_stop_and_log_debug!(PILFFLONK_WRAP_CIRCOM_WITNESS);
        Ok(bytes)
    }
}

/// The witness of the wrap's AIR, of shape `shape`, from `circom`, the final circuit's witness: the
/// stage-1 columns and the publics `exec` gathers out of it ([`ExecFile::committed_pols`]), the
/// multiplicity of its range checks if it has any, and no air values and no proof values.
///
/// The shape is the key's ([`ProvingKey::witness_shape`](proofman_pilfflonk::ProvingKey::witness_shape)):
/// it must have one AIR, with no stage-1 air values and no stage-1 proof values, as the wrap's
/// has; its publics are wires `1 ..= nPublics` of `circom`; and its trace must hold the exec's map
/// and the multiplicity's column.
pub fn witness_from_circom(
    exec: &ExecFile<Bn128>,
    circom: Vec<Bn128>,
    shape: &WitnessShape,
) -> WrapWitnessResult<Witness> {
    let wires = circom.par_iter().flat_map_iter(|v| v.to_le_bytes()).collect();
    let bands = Bands::of(exec).map_err(|e| WrapWitnessError::Mismatch(format!("the exec {e}")))?;
    witness_from_bytes(exec, bands.as_ref(), wires, shape)
}

/// [`witness_from_circom`], of the circom witness's bytes, each value below `r`, and `exec`'s bands.
fn witness_from_bytes(
    exec: &ExecFile<Bn128>,
    bands: Option<&Bands>,
    wires: Vec<u8>,
    shape: &WitnessShape,
) -> WrapWitnessResult<Witness> {
    let air = wrap_air(shape)?;
    timer_start_debug!(PILFFLONK_WRAP_EXEC);
    let n_rows = 1usize << air.n_bits;
    let blocks = match bands {
        Some(Bands::Blake3(blake3)) => Some(blocks_of(exec, blake3, &wires)?),
        _ => None,
    };
    let (publics, mut trace) = trace_of(exec, wires, shape.n_publics(), n_rows, air.n_cols)?;
    match (bands, blocks) {
        (Some(Bands::RangeChecks(range_checks)), _) => range_checks.count(&mut trace, n_rows, air.n_cols)?,
        (Some(Bands::Blake3(blake3)), Some((blocks, range))) => {
            blake3.fill(&mut trace, n_rows, air.n_cols, &blocks, &range)?
        }
        _ => {}
    }
    let stage1 = Stage1Witness::new(n_rows, air.n_cols, trace, vec![])?;
    timer_stop_and_log_debug!(PILFFLONK_WRAP_EXEC);
    Ok(Witness {
        instances: vec![InstanceWitness {
            air: AirInstanceRef { airgroup_id: air.airgroup_id, air_id: air.air_id },
            stage1,
        }],
        publics,
        proof_values: vec![],
    })
}

/// A wire of the circuit's witness that its calculator wrote not below `r`.
fn not_below_r(path: &Path, wire: usize) -> WrapWitnessError {
    WrapWitnessError::Mismatch(format!(
        "the witness calculator {} wrote wire {wire} of the circuit's witness not below r",
        path.display()
    ))
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

/// The gate bands of the wrap's AIR: the PoseidonBN128 wrap's range checks, or the blake3 BN128
/// wrap's blocks and range checks.
#[derive(Debug)]
enum Bands {
    RangeChecks(RangeChecks),
    Blake3(Blake3Bands),
}

impl Bands {
    /// The bands of `exec`, `None` if it has none, or why they are not a wrap's. The error completes
    /// "the exec ...".
    fn of(exec: &ExecFile<Bn128>) -> Result<Option<Self>, String> {
        if let Some(blake3) = Blake3Bands::of(exec)? {
            return Ok(Some(Self::Blake3(blake3)));
        }
        Ok(RangeChecks::of(exec)?.map(Self::RangeChecks))
    }
}

/// The range checks of the wrap's AIR, as its exec describes them: a range-check band of the exec
/// ([`RANGE_CHECK_BAND_KIND`]) for each, whose row the map gathers whole, `in` and its chunks, and
/// the stage-1 column of their table's multiplicity, `RANGE_MUL`, the band section's aux word.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RangeChecks {
    rows: Vec<usize>,
    multiplicity_column: usize,
}

impl RangeChecks {
    /// The range checks of `exec`, `None` if it has no gate band, or why its bands are not the
    /// wrap's: a band of another kind, which only the STARK's trace expander rebuilds, a range check
    /// of more chunks than a row holds or of none, or a multiplicity column that the map fills. The
    /// error completes "the exec ...".
    fn of(exec: &ExecFile<Bn128>) -> Result<Option<Self>, String> {
        if exec.bands.is_empty() {
            return Ok(None);
        }
        let max_chunks = RANGE_CHECK_CHUNK_COLS.len() as u64;
        let mut rows = Vec::with_capacity(exec.bands.len());
        for band in &exec.bands {
            if band.kind != RANGE_CHECK_BAND_KIND {
                return Err(format!(
                    "has a gate band of kind {} at row {}, which only the STARK's trace expander rebuilds",
                    band.kind, band.row
                ));
            }
            if !(1..=max_chunks).contains(&band.payload) {
                return Err(format!(
                    "has a range check at row {} of {} chunks, and a range-check row holds 1 to {max_chunks}",
                    band.row, band.payload
                ));
            }
            let row = usize::try_from(band.row)
                .map_err(|_| format!("has a range check at row {}, past what this machine addresses", band.row))?;
            rows.push(row);
        }
        let multiplicity_column = usize::try_from(exec.band_aux).unwrap_or(usize::MAX);
        if multiplicity_column < exec.layout.map_cols() {
            return Err(format!(
                "counts its range checks into stage-1 column {multiplicity_column}, which its map fills: its map has {} \
                 columns",
                exec.layout.map_cols()
            ));
        }
        Ok(Some(Self { rows, multiplicity_column }))
    }

    /// Writes the multiplicity into its column of `trace`, `n_rows x n_cols`, row after row, which
    /// holds what the map gathers: at row `v < 2^16`, how many chunk cells of the range-check rows
    /// hold `v`. Every chunk cell of a row counts, those past its chunks too (zero), as the AIR looks
    /// them all up. Refuses an AIR of fewer rows than the table's `2^16` or without the
    /// multiplicity's column, a range-check row past its rows, and a chunk of `2^16` or more, which
    /// no row of the table holds: the circuit's witness does not satisfy its `Num2Bytes`.
    fn count(&self, trace: &mut [u8], n_rows: usize, n_cols: usize) -> WrapWitnessResult<()> {
        let mismatch = |e: String| Err(WrapWitnessError::Mismatch(e));
        let table_rows = 1usize << RANGE_CHECK_CHUNK_BITS;
        if n_rows < table_rows {
            return mismatch(format!(
                "the AIR has {n_rows} rows, fewer than the 2^{RANGE_CHECK_CHUNK_BITS} of its range table"
            ));
        }
        if self.multiplicity_column >= n_cols {
            return mismatch(format!(
                "the exec counts its range checks into stage-1 column {}, and the AIR has {n_cols}",
                self.multiplicity_column
            ));
        }
        debug_assert_eq!(trace.len(), n_rows * n_cols * VALUE_BYTES);
        let mut counts = vec![0u64; table_rows];
        for &row in &self.rows {
            if row >= n_rows {
                return mismatch(format!("the exec has a range check at row {row}, and the AIR has {n_rows} rows"));
            }
            for col in RANGE_CHECK_CHUNK_COLS {
                let Some(chunk) = word(trace, row * n_cols + col).filter(|&v| v < 1 << RANGE_CHECK_CHUNK_BITS) else {
                    let cell = show(trace, row * n_cols + col);
                    return mismatch(format!(
                        "the chunk of the range check at row {row}, column {col}, is {cell}, not below \
                         2^{RANGE_CHECK_CHUNK_BITS}: the circuit's witness does not satisfy its Num2Bytes"
                    ));
                };
                counts[chunk as usize] += 1;
            }
        }
        for (row, count) in counts.into_iter().enumerate() {
            put_word(trace, row * n_cols + self.multiplicity_column, count);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use proofman_common::exec_format::{ExecGateBand, ExecLayout};
    use proofman_fields::{Field, PrimeField, QuotientMap};

    use super::*;

    /// [`RangeChecks::count`] on a trace of values.
    fn count(checks: &RangeChecks, trace: &mut [Bn128], n_rows: usize, n_cols: usize) -> WrapWitnessResult<()> {
        let mut bytes: Vec<u8> = trace.iter().flat_map(|v| v.to_le_bytes()).collect();
        let counted = checks.count(&mut bytes, n_rows, n_cols);
        for (v, b) in trace.iter_mut().zip(bytes.chunks_exact(VALUE_BYTES)) {
            *v = Bn128::from_le_bytes(b.try_into().unwrap()).unwrap();
        }
        counted
    }

    /// The wrap's columns: the 9 wires and `RANGE_MUL`.
    const N_COLS: usize = 10;

    /// The rows of the table, the fewest an AIR with range checks has.
    const N_ROWS: usize = 1 << RANGE_CHECK_CHUNK_BITS;

    /// An exec over a map of 9 columns with these range-check bands, `(row, chunks)`, counting into
    /// `column`.
    fn exec(bands: &[(u64, u64)], column: u64) -> ExecFile<Bn128> {
        let bands = bands.iter().map(|&(row, payload)| ExecGateBand { row, kind: RANGE_CHECK_BAND_KIND, payload });
        ExecFile {
            layout: ExecLayout::new::<Bn128>(1, 0, 1, 9),
            additions: vec![],
            map: vec![0; 9],
            band_aux: column,
            bands: bands.collect(),
        }
    }

    /// THE NAIVE REFERENCE: how many chunk cells, `a[1..=5]` of each row of `rows`, hold each value.
    fn naive_counts(trace: &[Bn128], rows: &[usize]) -> HashMap<u64, u64> {
        let mut counts = HashMap::new();
        for &row in rows {
            for col in 1..=5 {
                let value = trace[row * N_COLS + col].as_canonical_biguint();
                *counts.entry(u64::try_from(&value).unwrap()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Range-check rows of chunks from a fixed seed, all of [0, 2^16) and the top values, with the
    /// cells past a row's chunks zero, as the map gathers them.
    fn trace_of(rows: &[(usize, usize)]) -> Vec<Bn128> {
        let mut trace = vec![Bn128::ZERO; N_ROWS * N_COLS];
        let mut state = 0x4d35_3661_u64;
        for &(row, n_chunks) in rows {
            for k in 0..n_chunks {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let chunk = if k == 0 && row % 3 == 0 { 0xffff } else { state >> 48 };
                trace[row * N_COLS + 1 + k] = Bn128::from_int(chunk);
            }
        }
        trace
    }

    /// The multiplicity is the naive count, at the row of each value and 0 past them, and the map's
    /// columns are untouched.
    #[test]
    fn the_multiplicity_is_the_count_of_every_chunk_cell() {
        let rows: Vec<(usize, usize)> = (0..300).map(|i| (100 + 2 * i, 1 + i % 5)).collect();
        let bands: Vec<(u64, u64)> = rows.iter().map(|&(row, n)| (row as u64, n as u64)).collect();
        let checks = RangeChecks::of(&exec(&bands, 9)).unwrap().expect("range checks");
        let mut trace = trace_of(&rows);
        let gathered = trace.clone();
        count(&checks, &mut trace, N_ROWS, N_COLS).unwrap();

        let rows: Vec<usize> = rows.iter().map(|&(row, _)| row).collect();
        let counts = naive_counts(&gathered, &rows);
        assert!(counts[&0] > 0 && counts[&0xffff] > 0, "the zeros past the chunks and the top value count");
        assert_eq!(counts.values().sum::<u64>(), 5 * rows.len() as u64);
        for row in 0..N_ROWS {
            let expected = counts.get(&(row as u64)).copied().unwrap_or(0);
            assert_eq!(trace[row * N_COLS + 9], Bn128::from_int(expected), "RANGE_MUL at row {row}");
            assert_eq!(trace[row * N_COLS..row * N_COLS + 9], gathered[row * N_COLS..row * N_COLS + 9], "row {row}");
        }

        // Past the table, at 2^17 rows: the same counts, and zeros after them.
        let mut tall = vec![Bn128::ZERO; 2 * N_ROWS * N_COLS];
        tall[..N_ROWS * N_COLS].copy_from_slice(&gathered);
        count(&checks, &mut tall, 2 * N_ROWS, N_COLS).unwrap();
        assert_eq!(tall[..N_ROWS * N_COLS], trace[..]);
        assert!(tall[N_ROWS * N_COLS..].iter().all(|v| *v == Bn128::ZERO));
    }

    #[test]
    fn an_exec_with_no_band_has_no_range_check() {
        assert_eq!(RangeChecks::of(&exec(&[], 0)), Ok(None));
    }

    /// A band of the STARK, a range check of no chunks or of more than a row holds, and a
    /// multiplicity in the map's columns are refused, saying which.
    #[test]
    fn bands_that_are_not_the_wraps_range_checks_are_refused() {
        let mut stark = exec(&[(4, 1)], 9);
        stark.bands[0].kind = 1;
        let cases = [
            (stark, "has a gate band of kind 1 at row 4, which only the STARK's trace expander rebuilds"),
            (exec(&[(4, 0)], 9), "has a range check at row 4 of 0 chunks, and a range-check row holds 1 to 5"),
            (exec(&[(4, 6)], 9), "of 6 chunks"),
            (exec(&[(4, 5)], 8), "counts its range checks into stage-1 column 8, which its map fills"),
        ];
        for (file, why) in cases {
            let err = RangeChecks::of(&file).unwrap_err();
            assert!(err.contains(why), "expected \"{why}\", got: {err}");
        }
    }

    /// An AIR without the table's rows or the multiplicity's column, a range check past its rows,
    /// and a chunk outside the table are refused, saying which.
    #[test]
    fn what_the_count_cannot_fit_is_refused() {
        let checks = |bands: &[(u64, u64)], column| RangeChecks::of(&exec(bands, column)).unwrap().unwrap();
        let mut trace = trace_of(&[(3, 2)]);
        let err = |checks: RangeChecks, trace: &mut [Bn128], n_rows, n_cols| {
            count(&checks, trace, n_rows, n_cols).unwrap_err().to_string()
        };
        let mut short = vec![Bn128::ZERO; (N_ROWS / 2) * N_COLS];
        assert!(err(checks(&[(3, 2)], 9), &mut short, N_ROWS / 2, N_COLS)
            .contains("fewer than the 2^16 of its range table"));
        assert!(err(checks(&[(3, 2)], 10), &mut trace, N_ROWS, N_COLS)
            .contains("into stage-1 column 10, and the AIR has 10"));
        let past = N_ROWS as u64;
        assert!(err(checks(&[(past, 2)], 9), &mut trace, N_ROWS, N_COLS).contains("a range check at row 65536"));
        trace[3 * N_COLS + 2] = Bn128::from_int(1u64 << 16);
        let outside = err(checks(&[(3, 2)], 9), &mut trace, N_ROWS, N_COLS);
        assert!(outside.contains("row 3, column 2, is 65536, not below 2^16"), "{outside}");
    }
}
