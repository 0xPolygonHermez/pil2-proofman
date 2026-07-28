//! Helpers for the multilinear (Basefold + sumcheck) proving path.
//!
//! The multilinear prover is pure Rust: it consumes the witness trace computed
//! by the normal witness pipeline, the raw `.const` file and the `.mlinfo.bin`
//! artifact (the compiled [`AirIr`]) produced by `proofman-setup`, and never
//! crosses the C++ FFI. Entered via `ProofMan::generate_proof` with
//! `ProofOptions::proof_system == ProofSystem::Multilinear`.
//!
//! When present, the prover also loads a `.mlconst.bin` artifact — the prebuilt
//! Basefold commitment of the fixed columns, produced by `proofman-setup`
//! alongside `.mlinfo.bin`. Reusing it skips re-encoding and re-hashing the
//! const tree on every proof (and every instance of the same AIR). See
//! [`MlSetupCache`]. Absent, the commitment is built once per AIR from the
//! loaded columns, so older proving keys still work.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fields::{Field, Goldilocks, PrimeField64};
use proofman_common::{load_const_pols, ProofmanError, ProofmanResult, Setup};
use proofman_multilinear::{AirIr, MlPcs, Pcs, PcsCommitment};

/// Cache of loaded `.mlinfo.bin` artifacts, keyed by (airgroup_id, air_id).
#[derive(Default)]
pub struct AirIrCache {
    cache: Mutex<HashMap<(usize, usize), Arc<AirIr>>>,
}

impl AirIrCache {
    pub fn get<F: PrimeField64>(&self, setup: &Setup<F>) -> ProofmanResult<Arc<AirIr>> {
        let key = (setup.airgroup_id, setup.air_id);
        if let Some(ir) = self.cache.lock().unwrap().get(&key) {
            return Ok(ir.clone());
        }
        let path = setup.setup_path.with_extension("mlinfo.bin");
        if !path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "{} not found — the proving key was generated without multilinear support \
                 (re-run proofman-setup) or this AIR is not supported by the multilinear prover yet",
                path.display()
            )));
        }
        let ir = Arc::new(
            AirIr::load(&path)
                .map_err(|e| ProofmanError::InvalidParameters(format!("loading {}: {e}", path.display())))?,
        );
        self.cache.lock().unwrap().insert(key, ir.clone());
        Ok(ir)
    }
}

/// Per-AIR fixed-column data, loaded once per run and shared by every
/// instance: the row-major buffer in the FFI element type (consumed by
/// `init_fixed` / the C++ stage-1 expression evaluator) and the column-major
/// Goldilocks copy the multilinear prover consumes — both derived from a
/// single read of the `.const` file.
pub struct ConstData<F> {
    pub row_major: Arc<Vec<F>>,
    pub columns: Arc<Vec<Vec<Goldilocks>>>,
}

/// Per-AIR custom-commit (e.g. ROM) fixed data: raw columns plus their PCS
/// commitments, computed once per AIR and reused by every instance.
pub struct CustomData {
    pub columns: Vec<Vec<Vec<Goldilocks>>>,
    pub commitments: Vec<PcsCommitment>,
}

/// All per-AIR artifacts reusable across instances and passes of one
/// multilinear proving run: compiled IRs (`.mlinfo.bin`), fixed columns in
/// both layouts, the fixed-column commitment (`.mlconst.bin` or built once),
/// and custom-commit columns + commitments. All getters are thread-safe;
/// entries are immutable once inserted.
type AirCache<T> = Mutex<HashMap<(usize, usize), Arc<T>>>;

pub struct MlSetupCache<F> {
    irs: AirIrCache,
    const_data: AirCache<ConstData<F>>,
    const_matrices: AirCache<PcsCommitment>,
    customs: AirCache<CustomData>,
}

impl<F> Default for MlSetupCache<F> {
    fn default() -> Self {
        Self {
            irs: AirIrCache::default(),
            const_data: Mutex::new(HashMap::new()),
            const_matrices: Mutex::new(HashMap::new()),
            customs: Mutex::new(HashMap::new()),
        }
    }
}

impl<F: PrimeField64> MlSetupCache<F> {
    /// The compiled [`AirIr`] of an AIR (`.mlinfo.bin`), loaded once per run.
    pub fn ir(&self, setup: &Setup<F>) -> ProofmanResult<Arc<AirIr>> {
        self.irs.get(setup)
    }

    /// Fixed columns of an AIR, read from the `.const` file once per run.
    pub fn const_data(&self, setup: &Setup<F>, ir: &AirIr) -> ProofmanResult<Arc<ConstData<F>>> {
        let key = (setup.airgroup_id, setup.air_id);
        if let Some(d) = self.const_data.lock().unwrap().get(&key) {
            return Ok(d.clone());
        }
        let n_rows = 1usize << ir.n_bits;
        let n_cols = ir.n_const_cols as usize;
        let mut row_major = vec![F::ZERO; setup.const_pols_size];
        if setup.const_pols_size > 0 {
            load_const_pols(setup, &mut row_major);
        }
        if row_major.len() < n_cols * n_rows {
            return Err(ProofmanError::InvalidParameters(format!(
                "const buffer for {}: {} elements < {n_cols} cols × {n_rows} rows",
                ir.name,
                row_major.len()
            )));
        }
        let mut columns = vec![vec![Goldilocks::ZERO; n_rows]; n_cols];
        for row in 0..n_rows {
            for (c, col) in columns.iter_mut().enumerate() {
                col[row] = Goldilocks::new(row_major[row * n_cols + c].as_canonical_u64());
            }
        }
        let data = Arc::new(ConstData { row_major: Arc::new(row_major), columns: Arc::new(columns) });
        self.const_data.lock().unwrap().insert(key, data.clone());
        Ok(data)
    }

    /// The fixed-column commitment of an AIR: the prebuilt `.mlconst.bin`
    /// artifact when present, otherwise built once per run from the loaded
    /// columns (older proving keys).
    pub fn const_matrix(&self, setup: &Setup<F>, ir: &AirIr) -> ProofmanResult<Arc<PcsCommitment>> {
        let key = (setup.airgroup_id, setup.air_id);
        if let Some(m) = self.const_matrices.lock().unwrap().get(&key) {
            return Ok(m.clone());
        }
        let path = setup.setup_path.with_extension("mlconst.bin");
        let matrix = if path.exists() {
            Pcs::load_commitment(&path).map_err(|e| {
                ProofmanError::InvalidParameters(format!("loading fixed-column commitment {}: {e}", path.display()))
            })?
        } else {
            let data = self.const_data(setup, ir)?;
            let refs: Vec<&[Goldilocks]> = data.columns.iter().map(|c| c.as_slice()).collect();
            Pcs::commit(&refs, &ir.params)
        };
        let matrix = Arc::new(matrix);
        self.const_matrices.lock().unwrap().insert(key, matrix.clone());
        Ok(matrix)
    }

    /// Custom-commit columns and their commitments, loaded/committed once per
    /// AIR. `paths` are the registered fixed-buffer files, in
    /// `ir.custom_commits` order.
    pub fn custom_data(&self, key: (usize, usize), ir: &AirIr, paths: &[PathBuf]) -> ProofmanResult<Arc<CustomData>> {
        if let Some(d) = self.customs.lock().unwrap().get(&key) {
            return Ok(d.clone());
        }
        let n_rows = 1usize << ir.n_bits;
        let mut columns = Vec::with_capacity(ir.custom_commits.len());
        for (cc, path) in ir.custom_commits.iter().zip(paths.iter()) {
            columns.push(load_custom_columns(path, cc.n_cols as usize, n_rows)?);
        }
        let commitments = columns
            .iter()
            .map(|cols| {
                let refs: Vec<&[Goldilocks]> = cols.iter().map(|c| c.as_slice()).collect();
                Pcs::commit(&refs, &ir.params)
            })
            .collect();
        let data = Arc::new(CustomData { columns, commitments });
        self.customs.lock().unwrap().insert(key, data.clone());
        Ok(data)
    }
}

/// Split a row-major trace buffer into base-field columns (columns transpose
/// in parallel).
pub fn trace_to_columns<F: PrimeField64 + Sync>(trace: &[F], num_rows: usize, n_cols: usize) -> Vec<Vec<Goldilocks>> {
    use rayon::prelude::*;
    assert!(
        trace.len() >= num_rows * n_cols,
        "trace buffer too small: {} < {num_rows} rows x {n_cols} cols",
        trace.len()
    );
    (0..n_cols)
        .into_par_iter()
        .map(|c| (0..num_rows).map(|row| Goldilocks::new(trace[row * n_cols + c].as_canonical_u64())).collect())
        .collect()
}

/// Load the raw (CPU) `.const` file: headerless little-endian u64s, row-major.
pub fn load_const_columns(path: &Path, n_cols: usize, n_rows: usize) -> ProofmanResult<Vec<Vec<Goldilocks>>> {
    if n_cols == 0 {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(path)
        .map_err(|e| ProofmanError::InvalidParameters(format!("reading {}: {e}", path.display())))?;
    if bytes.len() != n_cols * n_rows * 8 {
        return Err(ProofmanError::InvalidParameters(format!(
            "{}: expected {} bytes ({n_cols} cols × {n_rows} rows), found {}",
            path.display(),
            n_cols * n_rows * 8,
            bytes.len()
        )));
    }
    let mut cols = vec![vec![Goldilocks::ZERO; n_rows]; n_cols];
    for row in 0..n_rows {
        for (c, col) in cols.iter_mut().enumerate() {
            let off = (row * n_cols + c) * 8;
            col[row] = Goldilocks::new(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }
    }
    Ok(cols)
}

/// Load the stage-0 columns of a custom commit from its registered buffer
/// file. The file layout (written by `gen-custom-commits-fixed` /
/// `write_custom_commit_trace`) is `[univariate merkle root: 4 words]
/// [N rows][extended rows][merkle tree]`; only the `n_rows × n_cols`
/// little-endian words after the root are the raw columns (row-major).
pub fn load_custom_columns(path: &Path, n_cols: usize, n_rows: usize) -> ProofmanResult<Vec<Vec<Goldilocks>>> {
    const ROOT_WORDS: usize = 4;
    if n_cols == 0 {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(path)
        .map_err(|e| ProofmanError::InvalidParameters(format!("reading {}: {e}", path.display())))?;
    let needed = (ROOT_WORDS + n_cols * n_rows) * 8;
    if bytes.len() < needed {
        return Err(ProofmanError::InvalidParameters(format!(
            "{}: expected at least {needed} bytes (root + {n_cols} cols × {n_rows} rows), found {}",
            path.display(),
            bytes.len()
        )));
    }
    let mut cols = vec![vec![Goldilocks::ZERO; n_rows]; n_cols];
    for row in 0..n_rows {
        for (c, col) in cols.iter_mut().enumerate() {
            let off = (ROOT_WORDS + row * n_cols + c) * 8;
            col[row] = Goldilocks::new(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }
    }
    Ok(cols)
}

/// Convert a slice of generic field elements to Goldilocks.
pub fn to_goldilocks<F: PrimeField64>(vals: &[F]) -> Vec<Goldilocks> {
    vals.iter().map(|v| Goldilocks::new(v.as_canonical_u64())).collect()
}

/// Reassemble a flat value buffer into `count` extension elements.
/// With `stride3`, every value occupies 3 slots (the airgroup-value layout);
/// otherwise the buffer must be exactly `3 * count` as well.
pub fn values_to_ext<F: PrimeField64>(
    vals: &[F],
    count: usize,
    _stride3: bool,
) -> ProofmanResult<Vec<proofman_multilinear::Ext>> {
    if vals.len() < 3 * count {
        return Err(ProofmanError::InvalidParameters(format!(
            "value buffer too small: {} < {}",
            vals.len(),
            3 * count
        )));
    }
    Ok((0..count)
        .map(|k| {
            proofman_multilinear::Ext::from_array(&[
                Goldilocks::new(vals[3 * k].as_canonical_u64()),
                Goldilocks::new(vals[3 * k + 1].as_canonical_u64()),
                Goldilocks::new(vals[3 * k + 2].as_canonical_u64()),
            ])
        })
        .collect())
}

/// Expand pctx's flat challenge buffer into the `Ext` vector `prove_air` and
/// `verify_air` consume: stage ≥ 2 challenges become Ext values, the rest
/// (univariate-only stages) stay zero. Shared by the prover and the proof-set
/// verifier so the two expansions cannot drift.
pub fn ml_challenges<F: PrimeField64>(
    pctx_challenges: &[F],
    challenge_stages: &[u8],
) -> Vec<proofman_multilinear::Ext> {
    let zero = proofman_multilinear::Ext::from_array(&[Goldilocks::ZERO; 3]);
    challenge_stages
        .iter()
        .enumerate()
        .map(|(id, &st)| {
            let base = 3 * id;
            if st >= 2 && base + 3 <= pctx_challenges.len() {
                proofman_multilinear::Ext::from_array(&[
                    Goldilocks::new(pctx_challenges[base].as_canonical_u64()),
                    Goldilocks::new(pctx_challenges[base + 1].as_canonical_u64()),
                    Goldilocks::new(pctx_challenges[base + 2].as_canonical_u64()),
                ])
            } else {
                zero
            }
        })
        .collect()
}

/// Reassemble air values, whose buffer layout depends on the value's stage:
/// stage-1 values occupy one slot, stage ≥ 2 values three.
pub fn ext_values_by_stage<F: PrimeField64>(
    vals: &[F],
    stages: &[u8],
) -> ProofmanResult<Vec<proofman_multilinear::Ext>> {
    let mut out = Vec::with_capacity(stages.len());
    let mut off = 0usize;
    for &st in stages {
        let dim = if st <= 1 { 1 } else { 3 };
        if vals.len() < off + dim {
            return Err(ProofmanError::InvalidParameters(format!(
                "air value buffer too small: {} < {}",
                vals.len(),
                off + dim
            )));
        }
        let mut coords = [Goldilocks::ZERO; 3];
        for (k, c) in coords.iter_mut().enumerate().take(dim) {
            *c = Goldilocks::new(vals[off + k].as_canonical_u64());
        }
        out.push(proofman_multilinear::Ext::from_array(&coords));
        off += dim;
    }
    Ok(out)
}
