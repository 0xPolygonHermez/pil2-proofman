//! Port of `generateRecursiveSetup.js`: generate recursive verifier circuits,
//! compile them, and produce the proving key artifacts for recursive1,
//! recursive2, and compressor templates.
//!
//! The function drives:
//! 1. Circom verifier generation (pil2circom)
//! 2. Recursive circom generation (gencircom)
//! 3. Circom compilation to R1CS + C++
//! 4. R1CS-to-PIL conversion (plonk2pil)
//! 5. PIL compilation (pil2com JS compiler)
//! 6. starkSetup (pil_info)
//! 7. Constant tree computation (bctree)
//! 8. Binary file generation

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use proofman_common::hash_family::HashFamily;

/// Blowup buys constraint degree: an air whose constraints reach degree d needs
/// `2^blowup + 1 >= d`. The poseidon aggregator is built at degree 8 and so needs 3; blake3's is
/// built at degree 5 (`PlonkOptions::max_constraint_degree`), which 2 already covers. Blowup
/// beyond what the degree needs is paid for in prover time and memory and buys nothing.
///
/// Scoped by family rather than lowered for everyone because the recursive grinding bits pin a
/// query count that the committed poseidon native verifiers and circom fixtures already encode.
///
/// blake3's compressor takes the recursion's 2 as well. Measured on the test-recursive circuit, 2^20/
/// LANES=3 at blowup 1 against the 2^19/LANES=4 the compressor now picks at blowup 2: degree 3
/// inflated stage2 (252 columns against 114) and doubled the queries (209 against 104), so blowup 2
/// proved 33% faster in 31% less GPU memory, with half the proof its recursive1 verifies.
pub fn recursive_blowup(template: RecursiveTemplate, hash: HashFamily) -> usize {
    if hash.is_blake3() || template == RecursiveTemplate::Compressor {
        2
    } else {
        3
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BatchFit {
    pub k: usize,
    pub per_proof: u64,
    pub capacity: u64,
    /// Lanes the capacity was computed at, since it scales with them.
    pub lanes: usize,
    pub n_queries: u64,
}

/// Proofs of `inner_si`'s shape a `2^n_bits` aggregator fits, by hash count; the build loop corrects a band-bound air.
fn recursion_batch_fit(hash: HashFamily, n_bits: usize, lanes: usize, inner_si: &Value) -> Option<BatchFit> {
    if !hash.is_blake3() {
        return None;
    }
    let geom = crate::verifier_hashes::geometry_from_starkinfo(inner_si).ok()?;
    let per_proof = crate::verifier_hashes::verifier_hashes(&geom, hash.as_str()).total();
    if per_proof == 0 {
        return None;
    }
    let capacity = pil2_stark_recurser::plonk2pil::setups::blake3::blake3_capacity(1usize << n_bits, lanes) as u64;
    let k = capacity / per_proof;
    Some(BatchFit { k: (k as usize).max(1), per_proof, capacity, lanes, n_queries: geom.n_queries })
}

/// Merkle levels a proof carries outright, per TEMPLATE.
///
/// This value is paid by whoever VERIFIES the proof, not by the air that emits it: a level kept is a
/// compression the verifier skips, bought with `arity^llv - 1` SelectValue gates per opening (see
/// `hash_family::recursive_last_level_verification`). So the compressor's llv sizes the recursive1
/// above it, and recursive1's own llv sizes recursive2.
///
/// The compressor takes 6, one above blake3's 5: each level trades a recursive1 compression per
/// opening for twice the SelectValue gates.
pub fn recursive_last_level_verification(template: RecursiveTemplate, hash: HashFamily) -> Option<usize> {
    match template {
        RecursiveTemplate::Compressor if hash.is_blake3() => Some(6),
        _ => proofman_common::hash_family::recursive_last_level_verification(hash.as_str()),
    }
}

/// The compressor grinds 27 so two Keccakf compressor proofs fit one recursive1 (at 25: 2 x 4728 of 9361 blocks).
pub fn recursive_grinding_bits(template: RecursiveTemplate, hash: HashFamily) -> usize {
    match template {
        RecursiveTemplate::Compressor if hash.is_blake3() => 27,
        _ => proofman_common::hash_family::recursive_grinding_bits(hash.as_str()),
    }
}

/// Whether a recursive1 must stop and ask for a compressor, from the circuit's OWN size.
///
/// Its own size and not the pinned one: `--recursive-n-bits` says how big the recursion runs, so
/// reading the threshold off the pinned size would make every air look too big.
///
/// Kept separate from the floor check in `gen_recursive_setup` so their precedence is testable. It
/// is load-bearing: an air over the threshold is exactly what a compressor fixes, and letting the
/// floor bail first turns that into a fatal "cannot be reconciled".
///
/// Against the recursion's domain, so a larger `--recursive-n-bits` asks for fewer compressors.
fn needs_compressor(
    template: RecursiveTemplate,
    has_compressor: bool,
    n_bits_natural: usize,
    domain_bits: usize,
) -> bool {
    template == RecursiveTemplate::Recursive1 && !has_compressor && n_bits_natural > domain_bits
}

/// Sentinel error returned when recursive1 detects that a compressor is required.
///
/// This is returned instead of a generic error so the caller can distinguish
/// "compressor needed" from a real failure and auto-retry.
#[derive(Debug)]
pub struct NeedsCompressorError {
    /// The n_bits value from plonk2pil that exceeded the 17-bit threshold.
    pub n_bits: usize,
}

impl std::fmt::Display for NeedsCompressorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recursive1 circuit exceeds 17-bit threshold (n_bits={}); a compressor is needed", self.n_bits)
    }
}

impl std::error::Error for NeedsCompressorError {}

use serde_json::Value;

use pil2_stark_recurser::stark2circom::CircomGenOptions;

use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::kinds::{self, RecursiveCfg};
use crate::proving_key::pipeline::{self, CircuitInput, PipelineCtx, Step, Tools, VerifierInput};

/// Which recursive template to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursiveTemplate {
    Compressor,
    Recursive1,
    Recursive2,
}

/// The stark settings of a recursion circuit — the configuration the whole recursion tree is
/// built from. There is no per-circuit config file: the defaults below are per template and
/// family, and `user` — the starkstructs `"recursion"` entry, see
/// `StarkStructsConfig::recursion_settings` — overlays the low-degree-test knobs on top of them
/// (`lowDegreeTest`, `initialFoldingFactor`, `grindingBits`, `grindingBitsQueries`,
/// `finalDegree`). An explicit `starkStruct` override only ever raises the query count, see
/// `gen_recursive_setup`.
///
/// FRI is the default low-degree test of compressor, recursive1 and recursive2, for every
/// family: the STIR schedules tried so far put blake3's recursive2 past the 2^19 every recursion
/// circuit shares (the quotient machinery outweighs the queries it saves), so STIR is opt-in
/// through `lowDegreeTest: "STIR"` in the recursion entry until a parameter set fits. The
/// vadcop-final layers are FRI regardless: their proofs are checked by the committed native Rust
/// verifiers, which are FRI builds.
pub fn recursive_stark_settings(
    template: RecursiveTemplate,
    hash: HashFamily,
    user: &crate::types::stark_struct::StarkSettings,
) -> crate::types::stark_struct::StarkSettings {
    let low_degree_test = user.low_degree_test.unwrap_or(crate::types::stark_struct::LowDegreeTestKind::Fri);
    let blowup = recursive_blowup(template, hash);
    // finalDegree is the final polynomial's log-degree bound, for both tests.
    //
    // FRI: the family's `fri_terminal_degree` is the log2 of the last committed DOMAIN -- the
    // `finalPol` the verifier evaluates at every query, which is what the constant was sized
    // on -- so the degree bound is that minus the (constant) log rate.
    //
    // STIR: the constant is taken as the degree bound d_M itself. Every quotient round needs
    // |Gᵢ| = tᵢ₋₁ + 1 < dᵢ, and the last one is the tight spot: with d_M = 2^5 the last fold
    // (≥ 1 bit) leaves d_{M−1} ≥ 64 against |G| ≈ 25–35 at these rates, at every trace size a
    // recursion circuit takes — a smaller bound breaks whenever the schedule needs a shortened
    // last fold (e.g. a compressor at 2^19). 32 coefficients in the clear cost nothing.
    let default_final_degree = || {
        let terminal = proofman_common::hash_family::fri_terminal_degree(hash.as_str());
        match low_degree_test {
            crate::types::stark_struct::LowDegreeTestKind::Fri => terminal.saturating_sub(blowup),
            crate::types::stark_struct::LowDegreeTestKind::Stir => terminal,
        }
    };
    crate::types::stark_struct::StarkSettings {
        low_degree_test: Some(low_degree_test),
        initial_blowup_factor: Some(blowup),
        initial_folding_factor: Some(user.initial_folding_factor.unwrap_or(match low_degree_test {
            crate::types::stark_struct::LowDegreeTestKind::Fri => 3,
            crate::types::stark_struct::LowDegreeTestKind::Stir => 4,
        })),
        final_degree: Some(user.final_degree.unwrap_or_else(default_final_degree)),
        // The grinding budget; under STIR every round grinds it unless grindingBitsQueries
        // sets a per-round one. The solver derives every query count from it.
        grinding_bits: Some(user.grinding_bits.unwrap_or_else(|| recursive_grinding_bits(template, hash))),
        // STIR only: a per-round budget overriding the seed (generate_stark_struct rejects it for FRI).
        grinding_bits_queries: user.grinding_bits_queries.clone(),
        folding_factors: user.folding_factors.clone(),
        last_level_verification: recursive_last_level_verification(template, hash),
        ..Default::default()
    }
}

impl RecursiveTemplate {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Compressor => "compressor",
            Self::Recursive1 => "recursive1",
            Self::Recursive2 => "recursive2",
        }
    }

    pub fn ejs_template(&self) -> &'static str {
        match self {
            Self::Compressor => "src/vadcop/templates/compressor.circom.ejs",
            Self::Recursive1 => "src/vadcop/templates/recursive1.circom.ejs",
            Self::Recursive2 => "src/vadcop/templates/recursive2.circom.ejs",
        }
    }
}

/// Configuration for the recursive setup.
pub struct RecursiveSetupConfig<'a> {
    pub build_dir: &'a str,
    pub template: RecursiveTemplate,
    pub airgroup_name: &'a str,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_name: &'a str,
    pub global_info: &'a Value,
    pub const_root: &'a [String; 4],
    pub verification_keys: &'a [Vec<Vec<String>>],
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,
    pub has_compressor: bool,
    pub hash: &'a str,
    /// The user's knobs on the recursion tree's own low-degree test (the starkstructs
    /// `"recursion"` entry), overlaid on `recursive_stark_settings`' defaults.
    pub recursion_settings: &'a crate::types::stark_struct::StarkSettings,
    /// Number of proofs the `recursive2` circuit aggregates. Ignored by the
    /// compressor and recursive1 templates.
    pub agg_arity: usize,
    /// Pin this air to 2^N rows (see the --recursive-n-bits flag).
    pub recursive_n_bits: Option<usize>,

    /// Optional pre-computed pil_info result to reuse instead of running starkSetup again.
    /// When provided, the pil_info computation (starkInfo / verifierInfo / expressionsInfo)
    /// is skipped and these values are used directly — this mirrors the JS behaviour where
    /// `setupAggregation_` is passed in for the 2nd+ airs and for recursive2.
    ///
    /// Format: `(stark_info, verifier_info, expressions_info)` as JSON Values.
    pub existing_pil_info: Option<(Value, Value, Value)>,

    // Tool paths
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub std_pil_path: &'a str,
    pub recurser_pil_path: &'a str,
    pub circom_helpers_dir: &'a str,
}

/// Result of a recursive setup step.
pub struct RecursiveSetupResult {
    /// The 4-element constant root.
    pub const_root: [u64; 4],
    /// The starkInfo JSON for this recursive circuit.
    pub stark_info: Option<Value>,
    /// The verifierInfo JSON for this recursive circuit.
    pub verifier_info: Option<Value>,
    /// The expressionsInfo JSON for this recursive circuit.
    pub expressions_info: Option<Value>,
    /// n_bits from plonk2pil (log2 of circuit rows).  Exposed so the caller
    /// can validate size without re-reading the starkInfo JSON.
    pub n_bits: usize,
    /// Proofs this circuit verifies; reaches globalInfo.json from here for the prover.
    pub batch_size: usize,
}

/// Run the recursive setup for a single air/template combination (`genRecursiveSetup()`).
pub fn gen_recursive_setup(
    config: &RecursiveSetupConfig<'_>,
    witness_tracker: &WitnessTracker,
) -> Result<RecursiveSetupResult> {
    let family: HashFamily = config.hash.parse().map_err(anyhow::Error::msg)?;
    let template = config.template;
    let template_str = template.as_str();

    // Nested under the `air` (or `airgroup`) span, so a line says which pass it belongs to.
    let _span = tracing::info_span!("stage", t = %template_str).entered();

    let (verifier_name, name_filename, files_dir) = resolve_names_and_paths(config);
    let mut ctx = PipelineCtx::new(config.build_dir.into(), files_dir, template_str, family);
    ctx.circuit = name_filename;
    ctx.label = match template {
        RecursiveTemplate::Recursive2 => format!("Airgroup '{}' recursive2", config.airgroup_name),
        _ => format!("Air '{}' {}", config.air_name, template_str),
    };
    ctx.tools = Tools {
        circom_exec: config.circom_exec.into(),
        libs: vec![config.recurser_circuits_path.into(), config.circuits_gl_path.into()],
        std_pil_path: config.std_pil_path.into(),
        recurser_pil_path: config.recurser_pil_path.into(),
        circom_helpers_dir: config.circom_helpers_dir.into(),
    };
    ctx.witness = Some(witness_tracker.clone());
    let const_root = if config.const_root.iter().all(|s| s.is_empty()) {
        ["0".to_string(), "0".to_string(), "0".to_string(), "0".to_string()]
    } else {
        config.const_root.clone()
    };
    ctx.verifiers = vec![VerifierInput {
        file: verifier_name,
        const_root,
        stark_info: config.stark_info.clone(),
        verifier_info: config.verifier_info.clone(),
    }];
    ctx.circuit_input = Some(CircuitInput {
        stark_infos: vec![config.stark_info.clone()],
        vadcop_info: config.global_info.clone(),
        basic_vk: config.verification_keys.to_vec(),
        agg_vk: vec![],
        publics: vec![],
        opts: CircomGenOptions {
            airgroup_id: Some(config.airgroup_id),
            has_compressor: config.has_compressor,
            has_recursion: false,
            is_final: false,
            agg_arity: config.agg_arity,
            batch_size: 1,
        },
    });
    ctx.settings = recursive_stark_settings(template, family, config.recursion_settings);
    // recursive2 keeps the recursive1 name: "{airgroup}_{air}_recursive1".
    let key_template = if template == RecursiveTemplate::Recursive2 { "recursive1" } else { template_str };
    ctx.key = crate::pil::stark_setup::KeyIdentity {
        name: Some(format!("{}_{}_{}", config.airgroup_name, config.air_name, key_template)),
        ids: Some((config.airgroup_id, config.air_id)),
        ..Default::default()
    };
    ctx.reuse_pil_info = config.existing_pil_info.clone();

    let cfg = RecursiveCfg {
        template,
        family,
        reuse: config.existing_pil_info.is_some(),
        has_compressor: config.has_compressor,
        multi_air: multi_air(config.global_info),
        domain_bits: config
            .recursive_n_bits
            .unwrap_or(proofman_common::hash_family::recursive_bits_threshold(config.hash)),
        agg_arity: config.agg_arity,
        airgroup_name: config.airgroup_name.to_string(),
        air_name: config.air_name.to_string(),
    };
    pipeline::run(&kinds::steps_for_recursive(&cfg), &mut ctx)?;

    let plonk = ctx.plonk.take().context("no plonk2pil result")?;
    let (si, vi, ei) = ctx.pil_info.take().context("no starkinfo")?;
    Ok(RecursiveSetupResult {
        const_root: ctx.const_root.context("no const root")?,
        stark_info: Some(si),
        verifier_info: Some(vi),
        expressions_info: Some(ei),
        n_bits: plonk.n_bits,
        batch_size: ctx.batch_size,
    })
}

/// Logs the batch fit before compiling, so a run that bails still shows it; sets recursive1's batch
/// and fails a recursive2 that cannot host its own arity.
pub(crate) fn fit_batch(cfg: &RecursiveCfg, ctx: &mut PipelineCtx) -> Result<()> {
    let template = cfg.template;
    let domain_bits = cfg.domain_bits;
    let inner = &ctx.verifiers.first().context("recursive circuit without its inner proof")?.stark_info;
    // Budgets are affine in k, so the largest k that fits also fills the domain.
    let lanes = pil2_stark_recurser::plonk2pil::setups::blake3::DEFAULT_LANES;
    let fit = recursion_batch_fit(cfg.family, domain_bits, lanes, inner);
    if let Some(fit) = fit {
        let head = format!(
            "Batch fit ({}): one verification of this shape costs {} compressions at {} queries, \
             {:.1}% of the {} a 2^{} aggregator hosts at LANES {}",
            template.as_str(),
            fit.per_proof,
            fit.n_queries,
            100.0 * fit.per_proof as f64 / fit.capacity as f64,
            fit.capacity,
            domain_bits,
            fit.lanes,
        );
        let packed = |n: usize| 100.0 * (n as u64 * fit.per_proof) as f64 / fit.capacity as f64;
        if template == RecursiveTemplate::Recursive2 {
            tracing::info!(
                "{head} -> aggregates {} of them ({:.1}% packed); {} would fit",
                cfg.agg_arity,
                packed(cfg.agg_arity),
                fit.k
            );
            // recursive2's fit is its self-fit: fail with the cause, not just a size.
            if fit.k < cfg.agg_arity {
                bail!(
                    "recursive2 at LANES {} fits {} proof(s) of its own shape in 2^{} but aggregates {}: \
                     one verification costs {} of the {} compressions available. RAISE the aggregator's LANES \
                     (capacity grows faster than the widened trace costs), lower --agg-arity, or \
                     raise --recursive-n-bits.",
                    fit.lanes,
                    fit.k,
                    domain_bits,
                    cfg.agg_arity,
                    fit.per_proof,
                    fit.capacity
                );
            }
        } else {
            tracing::info!("{head} -> will verify {} ({:.1}% packed)", fit.k, packed(fit.k));
        }
    }
    if template == RecursiveTemplate::Recursive1 {
        ctx.batch_size =
            fit.map_or(1, |f| f.k).min(pil2_stark_recurser::stark2circom::circuit_templates::MAX_RECURSIVE1_BATCH);
    }
    Ok(())
}

/// recursive1 whose band binds before its hashing does: `rerun` the prologue at a smaller batch.
pub(crate) fn retry_batch(cfg: &RecursiveCfg, ctx: &mut PipelineCtx, rerun: &[Step]) -> Result<()> {
    loop {
        let (r, batch) = (ctx.plonk()?, ctx.batch_size);
        if r.n_bits_natural <= cfg.domain_bits || batch == 1 {
            break;
        }
        // `n_used` is affine in k, so scale, not decrement.
        let scaled = (batch as u64).saturating_mul(1u64 << cfg.domain_bits) / (r.n_used as u64).max(1);
        let next = (scaled as usize).clamp(1, batch - 1);
        tracing::info!(
            "Recursive1 for air '{}' at k={} packs to 2^{} (n_used={}), past 2^{}; retrying at k={}",
            cfg.air_name,
            batch,
            r.n_bits_natural,
            r.n_used,
            cfg.domain_bits,
            next
        );
        ctx.batch_size = next;
        pipeline::run(rerun, ctx)?;
    }
    if ctx.batch_size > 1 {
        tracing::info!("Recursive1 for air '{}' verifies {} proofs per circuit", cfg.air_name, ctx.batch_size);
    }
    Ok(())
}

/// Too big even alone: ask for a compressor, before the floor check makes it fatal.
pub(crate) fn check_compressor(cfg: &RecursiveCfg, ctx: &mut PipelineCtx) -> Result<()> {
    let p = ctx.plonk()?;
    if needs_compressor(cfg.template, cfg.has_compressor, p.n_bits_natural, cfg.domain_bits) {
        tracing::warn!(
            "Recursive1 for air '{}' has n_bits={} > {} — compressor needed",
            cfg.air_name,
            p.n_bits_natural,
            cfg.domain_bits
        );
        return Err(anyhow::Error::new(NeedsCompressorError { n_bits: p.n_bits }));
    }
    // Unreconcilable: a compressor is already applied, the batch is 1, and a floor only pads up.
    if cfg.template != RecursiveTemplate::Compressor && p.n_bits > cfg.domain_bits {
        let why = if cfg.has_compressor {
            "this air already has a compressor and is still too big, so the compressor is not \
             the answer here: either raise the pin, or take rows out of the air."
        } else {
            "the recursion runs at that size (--recursive-n-bits, or the family threshold). \
             Either raise it, or take rows out of this air -- for blake3 the gate rows placed \
             outside the blake3 bands are the slack, since a band leaves the plonk columns free \
             on all but its first LANES rows."
        };
        bail!("{} compiles to 2^{} rows but must be 2^{}: {}", ctx.circuit, p.n_bits, cfg.domain_bits, why);
    }
    Ok(())
}

/// recursive2's `vks.json`: the recursive1 roots it accepts and its own.
pub(crate) fn write_vks_json(_cfg: &RecursiveCfg, ctx: &mut PipelineCtx) -> Result<()> {
    let r1_vks = ctx.circuit_input.as_ref().and_then(|c| c.basic_vk.first().cloned()).unwrap_or_default();
    let root = ctx.const_root.context("vks.json before the const tree")?;
    let num = |v: &String| v.parse::<u64>().map(Value::from).unwrap_or_else(|_| Value::String(v.clone()));
    let vks = serde_json::json!({
        "rootCRecursives1": r1_vks.iter().map(|vk| vk.iter().map(num).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "rootCRecursive2": root.iter().map(|&v| Value::from(v)).collect::<Vec<_>>(),
    });
    fs::write(ctx.file("vks.json"), serde_json::to_string_pretty(&vks)?)?;
    Ok(())
}

/// (verifier circom file, circuit name, files dir) of `config.template`.
fn resolve_names_and_paths(config: &RecursiveSetupConfig<'_>) -> (String, String, PathBuf) {
    let template = config.template;
    let airgroup_dir = PathBuf::from(config.build_dir)
        .join("provingKey")
        .join(get_global_name(config.global_info))
        .join(config.airgroup_name);
    match template {
        RecursiveTemplate::Compressor | RecursiveTemplate::Recursive1 => {
            let verifier = if template == RecursiveTemplate::Recursive1 && config.has_compressor {
                format!("{}_compressor.verifier.circom", config.air_name)
            } else {
                format!("{}.verifier.circom", config.air_name)
            };
            let files_dir = airgroup_dir.join("airs").join(config.air_name).join(template.as_str());
            (verifier, format!("{}_{}", config.air_name, template.as_str()), files_dir)
        }
        RecursiveTemplate::Recursive2 => (
            format!("{}_recursive2.verifier.circom", config.airgroup_name),
            format!("{}_{}", config.airgroup_name, template.as_str()),
            airgroup_dir.join(template.as_str()),
        ),
    }
}

/// More than one airgroup, or more than one air in the first.
fn multi_air(global_info: &Value) -> bool {
    let n_airgroups = global_info.get("air_groups").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(1);
    let n_airs_first = global_info
        .get("airs")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(1);
    n_airgroups > 1 || n_airs_first > 1
}

/// Extract the global name from vadcopInfo.
fn get_global_name(global_info: &Value) -> String {
    global_info.get("name").and_then(|v| v.as_str()).unwrap_or("pilout").to_string()
}

/// Compile PIL source by spawning `pil2com` (JS compiler) as a subprocess.
/// Install pil2com: `npm install` (reads package.json in the setup crate).
///
/// Thin wrapper around [`crate::commands::compile_pil::run_compile_pil`] so
/// that the recursive setup pipeline and the public CLI command share a
/// single implementation.
pub fn compile_pil(pil_path: &str, output_path: &str, std_pil_path: &str, recurser_pil_path: &str) -> Result<()> {
    use crate::commands::compile_pil::{run_compile_pil, CompilePilOptions};
    let opts = CompilePilOptions {
        pil_path: pil_path.to_string(),
        output_path: output_path.to_string(),
        include_paths: vec![std_pil_path.to_string(), recurser_pil_path.to_string()],
        ..Default::default()
    };
    run_compile_pil(&opts)
}

/// Locate an already-installed `pil2com`, checking (in order):
///   1. `PIL2C_EXEC` environment variable
///   2. The setup crate's own dir (compile-time baked) — covers consumers that
///      use proofman as a path/git dependency from another workspace
///   3. `node_modules/.bin/pil2com` relative to cwd
///   4. Walk up from the executable's location to find `node_modules/.bin/pil2com`
///   5. Global install: `pil2com` on PATH
///
/// Pure lookup — installs nothing. [`ensure_pil2com_exec`] is the entry point
/// that bootstraps when every step here misses.
fn resolve_pil2com_exec() -> Option<String> {
    if let Ok(path) = std::env::var("PIL2C_EXEC") {
        if Path::new(&path).is_file() {
            return Some(path);
        }
    }
    // npm install in the setup crate dir (this crate sits at <root>/setup/pil2-stark)
    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");
    let baked = Path::new(CRATE_ROOT).join("node_modules/.bin/pil2com");
    if baked.is_file() {
        if let Ok(abs) = baked.canonicalize() {
            return abs.to_str().map(|s| s.to_string());
        }
    }
    // Local npm install (preferred)
    let local_npm = Path::new("node_modules/.bin/pil2com");
    if local_npm.is_file() {
        if let Ok(abs) = local_npm.canonicalize() {
            return abs.to_str().map(|s| s.to_string());
        }
    }
    // Walk up from the executable's location
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("node_modules/.bin/pil2com");
            if candidate.is_file() {
                if let Ok(abs) = candidate.canonicalize() {
                    return abs.to_str().map(|s| s.to_string());
                }
            }
            dir = d.parent();
        }
    }
    // Global install on PATH
    which::which("pil2com").ok().map(|p| p.to_string_lossy().into_owned())
}

/// [`resolve_pil2com_exec`], but self-bootstrapping: when pil2com is missing,
/// install the Node deps (see [`crate::proving_key::node_deps`]) and use the
/// root it reports. Setup owns making its own tooling available — callers
/// (SDK, worker, CLI, tests) shouldn't each carry an npm bootstrap.
///
/// Bootstrapping through `node_deps` rather than npm-installing this crate's
/// directory directly is what makes this work for a published consumer: a
/// registry checkout is read-only, and `node_deps` falls back to a user-cache
/// root seeded from the embedded `package.json`. The snarkjs/circomlib
/// bootstrap in [`ensure_node_module_subpath`] goes through the same helper.
///
/// The result is memoized: a recursive setup calls this once per air (and again
/// per compressor attempt), and neither the PATH scan nor a failed bootstrap
/// gets cheaper by being repeated.
pub(crate) fn ensure_pil2com_exec() -> Option<String> {
    static PIL2COM_EXEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PIL2COM_EXEC.get_or_init(|| resolve_pil2com_exec().or_else(|| bootstrapped_node_path(".bin/pil2com"))).clone()
}

/// Install the Node deps if needed and return the absolute path to `probe`
/// (a path relative to `node_modules`), or `None` if the bootstrap failed.
fn bootstrapped_node_path(probe: &str) -> Option<String> {
    let root = crate::proving_key::node_deps::ensure_node_deps(probe)?;
    let path = root.join("node_modules").join(probe);
    path.canonicalize().ok().map(|p| p.to_string_lossy().into_owned())
}

/// Resolve a path inside `node_modules/<package>/<sub_path>`, with an env-var override.
/// Search order:
///   1. Env var override
///   2. `<setup-crate-dir>/node_modules/<package>/<sub_path>` (compile-time baked)
///   3. CWD-relative `node_modules/<package>/<sub_path>`
///   4. Walk up from the running executable
fn find_node_module_subpath(env_var: &str, package: &str, sub_path: &str) -> Option<String> {
    if let Ok(v) = std::env::var(env_var) {
        if !v.is_empty() {
            return Some(v);
        }
    }
    // Compile-time path to the setup crate dir (where node_modules lives).
    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");
    let baked = std::path::Path::new(CRATE_ROOT).join("node_modules").join(package).join(sub_path);
    if baked.is_dir() {
        if let Ok(abs) = baked.canonicalize() {
            return Some(abs.to_string_lossy().into_owned());
        }
    }
    let rel = PathBuf::from("node_modules").join(package).join(sub_path);
    if rel.is_dir() {
        if let Ok(abs) = rel.canonicalize() {
            return Some(abs.to_string_lossy().into_owned());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            let candidate = d.join("node_modules").join(package).join(sub_path);
            if candidate.is_dir() {
                if let Ok(abs) = candidate.canonicalize() {
                    return Some(abs.to_string_lossy().into_owned());
                }
            }
            dir = d.parent();
        }
    }
    None
}

/// [`find_node_module_subpath`], but self-bootstrapping: when the package is
/// missing, install the Node deps (see [`crate::proving_key::node_deps`]) and
/// look again. Falls back to the literal relative path so callers keep getting
/// a usable-for-error-messages string even when bootstrap is impossible.
pub(crate) fn ensure_node_module_subpath(env_var: &str, package: &str, sub_path: &str) -> String {
    if let Some(p) = find_node_module_subpath(env_var, package, sub_path) {
        return p;
    }
    let probe = format!("{package}/{sub_path}");
    bootstrapped_node_path(&probe).unwrap_or_else(|| format!("node_modules/{probe}"))
}

#[cfg(test)]
mod blowup_tests {
    use super::*;

    /// A recursive1 over the threshold must ask for a compressor, and that decision has to be
    /// reached BEFORE the "too big for its shared setup" bail -- ZisK's Keccakf packs to 2^21
    /// against a 2^19 airgroup floor, and with the floor checked first it died with
    /// "compiles to 2^21 rows but must be 2^19" and never got the compressor that fixes it.
    #[test]
    fn over_the_threshold_asks_for_a_compressor_before_the_floor_complains() {
        // Keccakf's real numbers: natural 2^21, blake3 threshold 19.
        assert!(
            super::needs_compressor(RecursiveTemplate::Recursive1, false, 21, 19),
            "an air 2 bits over the threshold must ask for a compressor"
        );
        // Already compressed and still too big: nothing left to try, so the floor bail is correct.
        assert!(!super::needs_compressor(RecursiveTemplate::Recursive1, true, 21, 19));
        // recursive2 cannot have a compressor, so it never asks for one.
        for t in [RecursiveTemplate::Recursive2, RecursiveTemplate::Compressor] {
            assert!(!super::needs_compressor(t, false, 21, 19), "{t:?} must not ask");
        }
        // Against the domain: a larger pin needs no compressor.
        assert!(!super::needs_compressor(RecursiveTemplate::Recursive1, false, 21, 21));
        // At or below the domain the air fits the shared recursion and needs nothing.
        for n in 0..=19 {
            assert!(!super::needs_compressor(RecursiveTemplate::Recursive1, false, n, 19), "n_bits {n}");
        }
        // Poseidon's threshold is lower, so the same air crosses it earlier.
        assert!(super::needs_compressor(RecursiveTemplate::Recursive1, false, 18, 17));
        assert!(!super::needs_compressor(RecursiveTemplate::Recursive1, false, 17, 17));
    }

    /// The compressor keeps one more Merkle level than the recursion pair, and only the compressor.
    #[test]
    fn only_the_compressor_keeps_the_extra_merkle_level() {
        assert_eq!(recursive_last_level_verification(RecursiveTemplate::Compressor, HashFamily::Blake3), Some(6));
        // The recursion pair keeps the family value; only the compressor departs from it.
        for t in [RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            assert_eq!(
                recursive_last_level_verification(t, HashFamily::Blake3),
                Some(5),
                "{t:?} takes the family value"
            );
        }
        for t in [RecursiveTemplate::Compressor, RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            assert_eq!(recursive_last_level_verification(t, HashFamily::Poseidon1), None, "{t:?}");
            assert_eq!(recursive_last_level_verification(t, HashFamily::Poseidon2), None, "{t:?}");
        }
    }

    /// Degree 5 needs 2^2 + 1; degree 8 needs 2^3 + 1. Raising blake3 back to 3 would cost prover
    /// time and memory for degree the air never uses, and lowering poseidon would break the query
    /// count the recursive grinding bits pin into the committed verifiers.
    #[test]
    fn blowup_follows_the_family_constraint_degree() {
        for t in [RecursiveTemplate::Recursive1, RecursiveTemplate::Recursive2] {
            assert_eq!(recursive_blowup(t, HashFamily::Blake3), 2, "blake3 {t:?} is built at degree 5");
        }
        assert_eq!(
            recursive_blowup(RecursiveTemplate::Compressor, HashFamily::Blake3),
            2,
            "measured: see recursive_blowup"
        );
        for h in [HashFamily::Poseidon1, HashFamily::Poseidon2] {
            assert_eq!(recursive_blowup(RecursiveTemplate::Compressor, h), 2);
            assert_eq!(recursive_blowup(RecursiveTemplate::Recursive1, h), 3, "{h:?} aggregator is degree 8");
            assert_eq!(recursive_blowup(RecursiveTemplate::Recursive2, h), 3, "{h:?} aggregator is degree 8");
        }
    }
}

#[cfg(test)]
mod batch_fit_tests {
    use super::*;

    /// One committed stage, no custom commits, one FRI step.
    fn si(n_queries: u64) -> Value {
        serde_json::json!({
            "nStages": 1, "nConstants": 1, "nPublics": 0,
            "mapSectionsN": { "cm1": 1, "cm2": 1 },
            "evMap": [{}], "challengesMap": [{"stage": 2}], "airValuesMap": [],
            "starkStruct": {
                "nBits": 8, "nBitsExt": 9, "merkleTreeArity": 2, "transcriptArity": 2,
                "merkleTreeCustom": false, "verificationHashType": "GL",
                "lastLevelVerification": 4, "hashCommits": true,
                "foldingFactors": [4], "logDegrees": [8, 4], "logDomainSizes": [9, 5],
                "numQueries": n_queries, "grindingBitsQueries": 24
            }
        })
    }

    const L: usize = pil2_stark_recurser::plonk2pil::setups::blake3::DEFAULT_LANES;

    /// The recurser's own, so a block-geometry change moves test and code together.
    fn capacity(n_bits: usize) -> u64 {
        pil2_stark_recurser::plonk2pil::setups::blake3::blake3_capacity(1usize << n_bits, L) as u64
    }

    #[test]
    fn k_is_the_hashing_budget_divided_by_one_verification() {
        let fit = recursion_batch_fit(HashFamily::Blake3, 19, L, &si(40)).unwrap();
        assert_eq!(fit.capacity, capacity(19));
        assert_eq!(fit.n_queries, 40);
        // Floors: a partial slot is a slot that does not fit.
        assert_eq!(fit.k as u64, fit.capacity / fit.per_proof);
        assert!(fit.k as u64 * fit.per_proof <= fit.capacity, "k proofs must fit the budget");
        assert!((fit.k as u64 + 1) * fit.per_proof > fit.capacity, "k must be the LARGEST count that fits");
    }

    /// A circuit that does not fit at all is `NeedsCompressorError`'s problem, not this one's.
    #[test]
    fn k_is_at_least_one_even_when_nothing_fits() {
        let huge = recursion_batch_fit(HashFamily::Blake3, 12, L, &si(600)).unwrap();
        assert_eq!(huge.k, 1);
    }

    /// The poseidon aggregators are laid out differently and need their own budget.
    #[test]
    fn no_estimate_for_a_family_without_a_block_model() {
        assert!(recursion_batch_fit(HashFamily::Poseidon1, 17, L, &si(40)).is_none());
        assert!(recursion_batch_fit(HashFamily::Poseidon2, 17, L, &si(40)).is_none());
    }
}

#[cfg(test)]
mod tests {
    use crate::types::stark_struct::{LowDegreeTestKind, StarkSettings};
    use proofman_common::hash_family::HashFamily;

    #[test]
    fn recursion_settings_keep_explicit_stir_folds() {
        let user = StarkSettings {
            low_degree_test: Some(LowDegreeTestKind::Stir),
            folding_factors: Some(vec![4, 3, 3]),
            ..Default::default()
        };
        let s = super::recursive_stark_settings(super::RecursiveTemplate::Recursive2, HashFamily::Blake3, &user);
        assert_eq!(s.folding_factors, Some(vec![4, 3, 3]));
    }
}
