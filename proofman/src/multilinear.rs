//! Helpers for the multilinear (Basefold + sumcheck) proving path.
//!
//! The multilinear prover is pure Rust: it consumes the witness trace computed
//! by the normal witness pipeline, the raw `.const` file and the `.mlinfo.bin`
//! artifact (the compiled [`AirIr`]) produced by `proofman-setup`, and never
//! crosses the C++ FFI. See `ProofMan::generate_multilinear_proof`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use fields::{Field, Goldilocks, PrimeField64};
use proofman_common::{ProofmanError, ProofmanResult, Setup};
use proofman_multilinear::AirIr;

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

/// Split a row-major trace buffer into base-field columns.
pub fn trace_to_columns<F: PrimeField64>(trace: &[F], num_rows: usize, n_cols: usize) -> Vec<Vec<Goldilocks>> {
    assert!(
        trace.len() >= num_rows * n_cols,
        "trace buffer too small: {} < {num_rows} rows x {n_cols} cols",
        trace.len()
    );
    let mut cols = vec![Vec::with_capacity(num_rows); n_cols];
    for row in 0..num_rows {
        for (c, col) in cols.iter_mut().enumerate() {
            col.push(Goldilocks::new(trace[row * n_cols + c].as_canonical_u64()));
        }
    }
    cols
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
