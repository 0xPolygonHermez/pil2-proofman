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
    debug_assert!(trace.len() >= num_rows * n_cols);
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

/// Convert a slice of generic field elements to Goldilocks.
pub fn to_goldilocks<F: PrimeField64>(vals: &[F]) -> Vec<Goldilocks> {
    vals.iter().map(|v| Goldilocks::new(v.as_canonical_u64())).collect()
}
