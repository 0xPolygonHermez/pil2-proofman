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

use proofman_fields::{Field, Goldilocks, PrimeField64};
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
    // 1.5 packed layout: root (4 words), words_per_row, one bit width per column, then each row's
    // columns concatenated LSB-first (a column may straddle two words) -- pack_columns.hpp.
    const ROOT_WORDS: usize = 4;
    if n_cols == 0 {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(path)
        .map_err(|e| ProofmanError::InvalidParameters(format!("reading {}: {e}", path.display())))?;
    let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    let words_per_row = word(ROOT_WORDS) as usize;
    let needed = (ROOT_WORDS + 1 + n_cols + n_rows * words_per_row) * 8;
    if words_per_row == 0 || words_per_row > n_cols || bytes.len() != needed {
        return Err(ProofmanError::InvalidParameters(format!(
            "{}: {} bytes is not a packed custom commit of {n_cols} cols x {n_rows} rows",
            path.display(),
            bytes.len()
        )));
    }
    let widths: Vec<u64> = (0..n_cols).map(|c| word(ROOT_WORDS + 1 + c)).collect();
    let base = ROOT_WORDS + 1 + n_cols;
    let mut cols = vec![vec![Goldilocks::ZERO; n_rows]; n_cols];
    for row in 0..n_rows {
        let row0 = base + row * words_per_row;
        let (mut idx, mut w, mut off) = (0usize, word(row0), 0u64);
        for (c, col) in cols.iter_mut().enumerate() {
            let nbits = widths[c];
            let left = 64 - off;
            let val = if nbits <= left {
                let mask = if nbits == 64 { !0u64 } else { (1u64 << nbits) - 1 };
                let v = (w >> off) & mask;
                off += nbits;
                if off == 64 && idx + 1 < words_per_row {
                    idx += 1;
                    w = word(row0 + idx);
                    off = 0;
                }
                v
            } else {
                let low = w >> off;
                idx += 1;
                w = word(row0 + idx);
                let high = w & ((1u64 << (nbits - left)) - 1);
                off = nbits - left;
                (high << left) | low
            };
            col[row] = Goldilocks::new(val);
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

/// One multilinear proving run riding the verify-constraints witness flow: every instance gets its
/// witness (stage 1, `<==` columns, stage 2) exactly as verify-constraints computes it and is
/// captured as an [`MlJob`]; the jobs are proven once every witness is done, since publics are only
/// final then (and witness code holds them while it waits for trace buffers). The stage challenges
/// are that flow's fixed dummy, not derived from the stage-1 roots: fine for timing and for checking
/// that proofs verify, not a sound aggregation.
pub struct MlRun<F> {
    pub cache: MlSetupCache<F>,
    pub verify_proofs: bool,
    /// The prover's own rayon pool: the global one runs the witness tasks, which block on trace
    /// buffers.
    pub pool: rayon::ThreadPool,
    pub jobs: Mutex<Vec<MlJob>>,
    /// (air name, n_bits, prove ms, verify ms, proof bytes) per proven instance.
    pub stats: Mutex<Vec<(String, u32, f64, f64, usize)>>,
}

/// One instance's prover inputs, owned, captured before its trace buffer is recycled.
pub struct MlJob {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_instance_id: usize,
    pub ir: Arc<AirIr>,
    pub witness: Vec<Vec<Vec<Goldilocks>>>,
    pub air_values: Vec<proofman_multilinear::Ext>,
    pub airgroup_values: Vec<proofman_multilinear::Ext>,
}

impl<F> MlRun<F> {
    pub fn new(verify_proofs: bool) -> Self {
        let pool = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("ml-prover-{i}")).build().unwrap();
        Self {
            cache: MlSetupCache::default(),
            verify_proofs,
            pool,
            jobs: Mutex::new(Vec::new()),
            stats: Mutex::new(Vec::new()),
        }
    }
}

/// Capture one instance whose verify-constraints witness is in place. Reads only per-instance state.
pub fn ml_capture_instance<F: PrimeField64>(
    pctx: &proofman_common::ProofCtx<F>,
    setup: &Setup<F>,
    instance_id: usize,
    run: &MlRun<F>,
) -> ProofmanResult<()> {
    let (airgroup_id, air_id) = (setup.airgroup_id, setup.air_id);
    let ir = run.cache.ir(setup)?;
    let n_rows = 1usize << ir.n_bits;
    if ir.n_stages() > 1 {
        return Err(ProofmanError::InvalidParameters(format!(
            "{}: {} witness stages; this port proves stage-1-only (LogUp-GKR) airs",
            ir.name,
            ir.n_stages()
        )));
    }
    let (witness, air_values) = {
        let air_instance = pctx.air_instances[instance_id].read().unwrap();
        if air_instance.num_rows != n_rows || ir.cols_per_stage[0] as usize != air_instance.n_cols_trace {
            return Err(ProofmanError::InvalidParameters(format!(
                "{}: trace is {} rows x {} cols, mlinfo expects {n_rows} x {:?}",
                ir.name, air_instance.num_rows, air_instance.n_cols_trace, ir.cols_per_stage
            )));
        }
        let cols = trace_to_columns(&air_instance.trace, n_rows, air_instance.n_cols_trace);
        (vec![cols], ext_values_by_stage(&air_instance.airvalues, &ir.airvalue_stages)?)
    };
    let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;
    // 1.5 packs airgroup values like air values: one slot at stage 1, three after.
    let airgroup_values = ext_values_by_stage(
        &pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id)?,
        &ir.airgroupvalue_stages,
    )?;
    run.jobs.lock().unwrap().push(MlJob { airgroup_id, air_id, air_instance_id, ir, witness, air_values, airgroup_values });
    Ok(())
}

/// Prove (and with `-y` verify) every captured instance, with the final publics and proof values.
pub fn ml_prove_jobs<F: PrimeField64>(
    pctx: &proofman_common::ProofCtx<F>,
    sctx: &proofman_common::SetupCtx<F>,
    run: &MlRun<F>,
) -> ProofmanResult<()> {
    let publics = to_goldilocks(&pctx.get_publics());
    let all_proof_values = pctx.get_proof_values();
    let mut jobs = std::mem::take(&mut *run.jobs.lock().unwrap());
    jobs.sort_by_key(|j| (j.airgroup_id, j.air_id, j.air_instance_id));
    for job in jobs {
        let setup = sctx.get_setup(job.airgroup_id, job.air_id)?;
        let ir = &job.ir;
        let consts = run.cache.const_data(setup, ir)?.columns.clone();
        let const_matrix = if ir.n_const_cols == 0 { None } else { Some(run.cache.const_matrix(setup, ir)?) };
        let custom_paths: Vec<PathBuf> = ir
            .custom_commits
            .iter()
            .map(|cc| pctx.get_custom_commits_fixed_buffer(&cc.name, true))
            .collect::<ProofmanResult<_>>()?;
        let custom = run.cache.custom_data((job.airgroup_id, job.air_id), ir, &custom_paths)?;
        let proof_values = ext_values_by_stage(&all_proof_values, &ir.proofvalue_stages)?;
        let challenges = ml_challenges(&pctx.get_challenges(), &ir.challenge_stages);

        let t = std::time::Instant::now();
        let proof = proofman_multilinear::prove_air(
            ir,
            &job.witness,
            &consts,
            const_matrix.as_deref(),
            &custom.columns,
            Some(&custom.commitments),
            &publics,
            &challenges,
            &job.air_values,
            &job.airgroup_values,
            &proof_values,
        )
        .map_err(|e| ProofmanError::InvalidProof(format!("{}: multilinear prover failed: {e}", ir.name)))?;
        let prove_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut verify_ms = 0.0;
        if run.verify_proofs {
            let t = std::time::Instant::now();
            proofman_multilinear::verify_air(ir, &proof, &publics, None, Some(&challenges), Some(&proof_values))
                .map_err(|e| ProofmanError::InvalidProof(format!("{}: multilinear proof rejected: {e}", ir.name)))?;
            verify_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).map(|v| v.len()).unwrap_or(0);
        tracing::info!(
            "    ML {} #{} (2^{}): prove {prove_ms:.1} ms{}, proof {proof_bytes} B",
            ir.name,
            job.air_instance_id,
            ir.n_bits,
            if run.verify_proofs { format!(", verified in {verify_ms:.1} ms") } else { String::new() }
        );
        run.stats.lock().unwrap().push((ir.name.clone(), ir.n_bits, prove_ms, verify_ms, proof_bytes));
    }
    Ok(())
}
