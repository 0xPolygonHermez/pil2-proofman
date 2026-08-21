//! Per-AIR expression -> straight-line CUDA kernel codegen.
//!
//! Two kernel families are emitted into each AIR's self-contained
//! `<base>.exps.so` (placed next to that AIR's `.bin`):
//! * the Q (cExp) kernel -- register-bounded, chunk size autotuned to zero
//!   register spill;
//! * one small trace-domain kernel per other covered expression (hint fields,
//!   im columns, ...), dispatched by expId via `exps_expr_covered` /
//!   `exps_launch_expr` (see `emit_exprs_tu`).
//!
//! The prover `dlopen`s the library by convention and falls back to the
//! bytecode interpreter for anything absent: missing `.so`, missing symbol,
//! or an uncovered expression.
//!
//! Two entry points:
//! * [`generate_air`]  — one AIR dir -> its `.exps.so`.
//! * [`generate_all`]  — a provingKey dir -> every AIR's `.exps.so`.

mod autotune;
mod check;
mod emit;
mod field;
mod ir;
mod model;
mod opt;
mod toolchain;

use anyhow::{Context, Result};
use ir::{plan_chunks, UnhandledOperand};
use model::{ExpressionsInfo, StarkInfo};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use toolchain::Toolchain;

const DEFAULT_CAP: usize = 40000; // skip an AIR whose Q has more ops than this
const DEFAULT_CHUNK: usize = 512; // fixed ops/chunk when autotuning is off
const SLOTS_CAP: u64 = 1000; // skip an AIR whose cross-chunk cut exceeds this

/// Codegen configuration (the CLI defaults: cap 60000, autotune on, arch auto, optimizer on).
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Skip an AIR whose Q has more than this many ops (-> interpreter).
    pub cap: usize,
    /// Fixed ops/chunk for every AIR; `None` turns the no-spill autotuner ON.
    pub chunk: Option<usize>,
    /// CUDA arch spec: `auto` (default), `major`, or a list like `89,120`.
    pub archspec: String,
    /// pil2-stark source root; `None` uses the vendored `proofman-starks-src` tree.
    pub stark_src: Option<PathBuf>,
    /// Retain the generated `.cu`/`.o` here; `None` uses a temp dir removed on exit.
    pub keep_dir: Option<PathBuf>,
    /// Emit the `.cu` sources only — skip compiling/linking the `.so` (so the
    /// provingKey is untouched). Requires `keep_dir`. Used for inspecting the
    /// generated sources.
    pub dry_run: bool,
    /// Run the Q IR optimizer (CSE, Horner→powers, scheduling; see `opt.rs`).
    /// `false` emits the expression exactly as the setup wrote it (A/B baseline).
    pub optimize: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            cap: DEFAULT_CAP,
            chunk: None,
            archspec: "auto".into(),
            stark_src: None,
            keep_dir: None,
            dry_run: false,
            optimize: true,
        }
    }
}

/// One AIR whose kernel was generated.
#[derive(Debug, Clone)]
pub struct GeneratedAir {
    pub name: String,
    pub base: String,
    pub sym: String,
    pub nbits: u64,
    pub cexp: i64,
    pub n_ops: usize,
    pub slots: u64,
}

/// Outcome of a codegen run.
#[derive(Debug, Default)]
pub struct GenSummary {
    pub generated: Vec<GeneratedAir>,
    pub skipped: Vec<(String, String)>,
    pub placed: usize,
    pub max_scratch_bytes: u64,
}

/// A discovered, codegen-eligible AIR (unique per `sym`).
struct Candidate {
    stark_info: StarkInfo,
    expr_info: ExpressionsInfo,
    sym: String,
    nbits: u64,
    cexp: i64,
    name: String,
    n_ops: usize,
    base: String,
}

/// One `.so` destination (every eligible AIR, not deduped by sym).
struct Placement {
    name: String,
    base: String,
    sym: String,
    air_dir: PathBuf,
}

/// Recursively collect `*.starkinfo.json` paths under `root`, sorted.
fn find_starkinfos(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with(".starkinfo.json")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The proof phase a circuit belongs to, derived from the AIR dir's leaf
/// component. Part of the kernel identity so circuits from different phases
/// never collide on `(airgroupId, airId, nBits, cExpId)`.
fn proof_phase(air_dir: &Path) -> String {
    let comp = air_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    match comp {
        "air" => "basic",
        "compressor" => "compressor",
        "recursive1" | "recursive2" => "recursive",
        c if c.starts_with("vadcop_final") => "final",
        other => other, // unknown layout: keep the raw dir name so it still disambiguates
    }
    .to_string()
}

/// `sym` — an AIR's kernel identity string `<phase>_a<airgroupId>_<airId>_b<nBits>_e<cExpId>`
/// (phase ∈ basic|compressor|recursive|final). Dedup key, file stem, and C/CUDA
/// symbol stem all at once.
fn make_sym(si: &StarkInfo, phase: &str) -> String {
    format!("{phase}_a{}_{}_b{}_e{}", si.airgroup_id, si.air_id, si.stark_struct.n_bits, si.c_exp_id)
}

/// Parse one starkinfo + sibling expressionsinfo into a Candidate, or return a
/// skip reason (string), or `None` if the file pair isn't a codegen target.
fn load_candidate(
    stark_info_path: &Path,
    root: &Path,
    cap: usize,
) -> Result<Option<Candidate>, Option<(String, String)>> {
    let air_dir = stark_info_path.parent().unwrap();
    let fname = stark_info_path.file_name().unwrap().to_string_lossy();
    let base = fname.strip_suffix(".starkinfo.json").unwrap().to_string();
    let expr_info_path = air_dir.join(format!("{base}.expressionsinfo.json"));
    if !expr_info_path.exists() {
        return Err(None);
    }
    let name = air_dir.strip_prefix(root).unwrap_or(air_dir).to_string_lossy().to_string();

    let stark_info: StarkInfo = match std::fs::read(stark_info_path).ok().and_then(|b| serde_json::from_slice(&b).ok())
    {
        Some(si) => si,
        None => return Err(None), // unparseable / not a full AIR starkinfo
    };
    let expr_info: ExpressionsInfo =
        match std::fs::read(&expr_info_path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(ei) => ei,
            None => return Err(None),
        };

    let cexp = stark_info.c_exp_id;
    let Some(code) = expr_info.expressions_code.iter().find(|e| e.exp_id == cexp) else {
        return Err(None); // cExpId not in expressionsCode
    };
    let n_ops = code.code.len();
    let nbits = stark_info.stark_struct.n_bits;
    if n_ops > cap {
        return Err(Some((name, format!("{n_ops} ops > CAP"))));
    }
    let sym = make_sym(&stark_info, &proof_phase(air_dir));
    Ok(Some(Candidate { stark_info, expr_info, sym, nbits, cexp, name, n_ops, base }))
}

/// Generate every AIR's `.exps.so` under `proving_key`. Returns a summary of
/// what was generated, skipped, and placed.
pub fn generate_all(proving_key: &Path, cfg: &GenConfig) -> Result<GenSummary> {
    if cfg.dry_run && cfg.keep_dir.is_none() {
        anyhow::bail!("dry_run requires keep_dir (nowhere to write the .cu otherwise)");
    }
    let work = WorkDir::new(cfg.keep_dir.clone())?;
    std::fs::write(work.path().join("gen_common.cuh"), emit::COMMON_CUH)?;
    let tc = Toolchain::new(cfg.stark_src.clone(), &cfg.archspec, work.path())?;
    eprintln!("[exps-codegen] generating kernels for {} (archs: {})", proving_key.display(), tc.arch_summary());

    // Phase 1: discovery — unique candidates (by sym) + every placement.
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut placements: Vec<Placement> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for si_path in find_starkinfos(proving_key) {
        match load_candidate(&si_path, proving_key, cfg.cap) {
            Ok(Some(c)) => {
                placements.push(Placement {
                    name: c.name.clone(),
                    base: c.base.clone(),
                    sym: c.sym.clone(),
                    air_dir: si_path.parent().unwrap().to_path_buf(),
                });
                if seen.insert(c.sym.clone()) {
                    candidates.push(c);
                }
            }
            Ok(None) => {}
            Err(Some(skip)) => skipped.push(skip),
            Err(None) => {}
        }
    }

    let mut summary = run_pipeline(&tc, work.path(), &candidates, &placements, cfg)?;
    summary.skipped.extend(skipped);
    summary.skipped.sort();
    print_summary(&summary, cfg);
    Ok(summary)
}

/// Generate the `.exps.so` for a single AIR directory (the dir containing its
/// `*.starkinfo.json` + `*.expressionsinfo.json`). Returns the `.so` path, or
/// an error if the AIR was skipped (unhandled operand, over CAP, or spills).
pub fn generate_air(air_dir: &Path, cfg: &GenConfig) -> Result<PathBuf> {
    let si_path = find_starkinfos(air_dir)
        .into_iter()
        .find(|p| p.parent() == Some(air_dir))
        .with_context(|| format!("no *.starkinfo.json in {}", air_dir.display()))?;

    let work = WorkDir::new(cfg.keep_dir.clone())?;
    std::fs::write(work.path().join("gen_common.cuh"), emit::COMMON_CUH)?;
    let tc = Toolchain::new(cfg.stark_src.clone(), &cfg.archspec, work.path())?;

    let root = air_dir;
    let candidate = match load_candidate(&si_path, root, cfg.cap) {
        Ok(Some(c)) => c,
        Ok(None) => anyhow::bail!("{} is not a codegen target", air_dir.display()),
        Err(Some((_, why))) => anyhow::bail!("skipped: {why}"),
        Err(None) => anyhow::bail!("{} missing/invalid expressionsinfo", air_dir.display()),
    };
    let placement = Placement {
        name: candidate.name.clone(),
        base: candidate.base.clone(),
        sym: candidate.sym.clone(),
        air_dir: air_dir.to_path_buf(),
    };
    let dest = air_dir.join(format!("{}.exps.so", candidate.base));

    let summary =
        run_pipeline(&tc, work.path(), std::slice::from_ref(&candidate), std::slice::from_ref(&placement), cfg)?;
    if summary.placed == 1 {
        Ok(dest)
    } else {
        let why = summary.skipped.first().map(|(_, w)| w.clone()).unwrap_or_else(|| "unknown".into());
        anyhow::bail!("{}: not generated ({why})", candidate.name)
    }
}

/// Optimize the Q IR and prove the result equivalent to the original on random
/// inputs. A mismatch is a generator bug: the AIR is skipped loudly rather than
/// emitted (the prover falls back to the interpreter for it).
fn optimize_checked(ir: ir::Ir, c: &Candidate, cfg: &GenConfig) -> std::result::Result<ir::Ir, String> {
    if !cfg.optimize {
        return Ok(ir);
    }
    let (opt, st) = opt::optimize(&ir);
    if let Err(e) = check::equivalent(&ir, &opt, 3) {
        eprintln!("[exps-codegen] {}: OPTIMIZED IR MISMATCH — {e}; skipping", c.name);
        return Err("optimizer self-check failed".to_string());
    }
    // Decline the optimized IR when CSE inflates live pressure without a large
    // arithmetic win. Measured (aggregation mode, 66-tx block): recursive2 with
    // live 40->223 and cost 84% ran +48% slower, the compressors (58->185, 77%)
    // +21%, vadcop_final (40->223, 75%) +34%; Keccakf (453->349, 76%) ran -35%
    // and ArithEq (55->92, 39%) -22%. Slots, not ops, decide for those shapes.
    // Absolute floor: a jump from 5 to 11 live values is free (Dma64AlignedMemCpy
    // measured -14%); the regressions all sat at 185-223 live, i.e. far past what
    // a chunk can hold in registers, where every extra value is a scratch slot.
    let live_ratio = st.max_live_after as f64 / st.max_live_before.max(1) as f64;
    let cost_ratio = st.cost_after as f64 / st.cost_before.max(1) as f64;
    let decline = !std::env::var("EXPS_NO_DECLINE").is_ok_and(|v| v != "0")
        && live_ratio > 2.0
        && st.max_live_after > 64
        && cost_ratio > 0.5;
    if decline {
        eprintln!(
            "[exps-codegen] {}: optimizer DECLINED (max live {} -> {}, cost {:.1}%): keeping the setup's expression",
            c.name,
            st.max_live_before,
            st.max_live_after,
            100.0 * cost_ratio
        );
        return Ok(ir);
    }
    // cross-chunk slots at the dry-run chunk size (or the default), before/after
    let chunk = cfg.chunk.unwrap_or(DEFAULT_CHUNK);
    let slots = |x: &ir::Ir| ir::plan_chunks(x, chunk, &c.sym).map(|p| (p.n_chunks, p.total_slots)).unwrap_or((0, 0));
    let (nc0, s0) = slots(&ir);
    let (nc1, s1) = slots(&opt);
    eprintln!(
        "[exps-codegen] {}: ops {} -> {} ({:.1}%), cost {} -> {} ({:.1}%), horner terms {}, max live {} -> {}, remat {}, hoisted {} ops -> {} words, inner {}{}, ltd {} terms/{} atoms, chunks {} -> {}, slots {} -> {}",
        c.name,
        st.ops_before,
        st.ops_after,
        100.0 * st.ops_after as f64 / st.ops_before.max(1) as f64,
        st.cost_before,
        st.cost_after,
        100.0 * st.cost_after as f64 / st.cost_before.max(1) as f64,
        st.horner_terms,
        st.max_live_before,
        st.max_live_after,
        st.remat,
        st.tab_ops,
        st.tab_words,
        st.inner_chains,
        if st.inner_selected { " (fold)" } else { " (no fold)" },
        st.ltd_terms,
        st.ltd_atoms,
        nc0,
        nc1,
        s0,
        s1
    );
    Ok(opt)
}

fn run_pipeline(
    tc: &Toolchain,
    work: &Path,
    candidates: &[Candidate],
    placements: &[Placement],
    cfg: &GenConfig,
) -> Result<GenSummary> {
    let autotune = cfg.chunk.is_none();

    // Build IR for every candidate (catches unhandled operands here). Each entry
    // is (candidate, Ok(ir) | Err(skip-reason)).
    let built: Vec<(&Candidate, std::result::Result<ir::Ir, String>)> = candidates
        .par_iter()
        .map(|c| {
            let r = match ir::build_ir(&c.stark_info, &c.expr_info) {
                Ok(ir) => optimize_checked(ir, c, cfg),
                Err(e) if e.downcast_ref::<UnhandledOperand>().is_some() => Err("unhandled operand".to_string()),
                Err(e) => Err(format!("build_ir error: {e}")),
            };
            (c, r)
        })
        .collect();

    // Phase 2: autotune the no-spill chunk size per AIR (parallel). Maps sym -> Some(chunk) | None(spills).
    let chunk_map: std::collections::HashMap<String, Option<usize>> = if autotune {
        built
            .par_iter()
            .filter_map(|(c, r)| {
                r.as_ref()
                    .ok()
                    .map(|ir| autotune::tune_chunk(tc, ir, &c.sym, c.n_ops, work).map(|ck| (c.sym.clone(), ck)))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?
    } else {
        Default::default()
    };

    // Phase 3: emit the .cu sources; record per-sym slot counts.
    let mut slots_by_sym: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut generated: Vec<GeneratedAir> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut max_scratch: u64 = 0;
    for (c, r) in &built {
        let ir = match r {
            Ok(ir) => ir,
            Err(why) => {
                skipped.push((c.name.clone(), why.clone()));
                continue;
            }
        };
        let chunk = if autotune {
            match chunk_map.get(&c.sym).copied().flatten() {
                Some(ck) => ck,
                None => {
                    skipped.push((c.name.clone(), format!("{} ops: still spills at CHUNK_MIN", c.n_ops)));
                    continue;
                }
            }
        } else {
            cfg.chunk.unwrap_or(DEFAULT_CHUNK)
        };
        let plan = match plan_chunks(ir, chunk, &c.sym) {
            Ok(p) => p,
            Err(why) => {
                skipped.push((c.name.clone(), format!("chunk plan failed: {why}")));
                continue;
            }
        };
        if plan.total_slots > SLOTS_CAP {
            skipped.push((c.name.clone(), format!("slots {} > SLOTS_CAP (wide cut)", plan.total_slots)));
            continue;
        }
        for (fname, text) in emit::emit_air(ir, &plan, &c.sym) {
            std::fs::write(work.join(&fname), text)?;
        }
        // Generic (non-Q) expression kernels: cover every trace-domain
        // expression the interpreter might be asked for (hint fields, im
        // columns, ...). Small straight-line kernels only; anything odd
        // (zerofier use, non-canonical output shape, oversized) is skipped
        // and stays on the interpreter.
        {
            const EXPR_CAP: usize = 512;
            let mut items: Vec<(i64, ir::Ir, u64)> = Vec::new();
            for ec in &c.expr_info.expressions_code {
                if ec.exp_id == c.cexp || ec.code.is_empty() || ec.code.len() > EXPR_CAP {
                    continue;
                }
                let Ok(eir) = ir::build_ir_expr(&c.stark_info, &c.expr_info, ec.exp_id, false) else {
                    continue;
                };
                if eir.uses_zi() {
                    continue;
                }
                let Some(od) = eir.out_dim() else { continue };
                if od != 1 && od != 3 {
                    continue;
                }
                // Optimize the standalone expression too (inner-chain fold with the
                // powers kept in registers, CSE, scheduling; no hoisting: no table).
                // Equivalence-gated exactly like Q; on mismatch the original is kept.
                let eir = if cfg.optimize && std::env::var("EXPS_X_OPT").ok().is_none_or(|v| v != "0") {
                    let (mut o, _st) = opt::optimize_opts(&eir, false, true);
                    if check::equivalent(&eir, &o, 3).is_ok() {
                        o.pow_in_regs = true;
                        o
                    } else {
                        eprintln!(
                            "[exps-codegen] {} x{}: OPTIMIZED IR MISMATCH; keeping the setup's expression",
                            c.name, ec.exp_id
                        );
                        eir
                    }
                } else {
                    eir
                };
                items.push((ec.exp_id, eir, od));
            }
            if !items.is_empty() {
                std::fs::write(work.join(format!("gen_{}_cexprs.cu", c.sym)), emit::emit_exprs_tu(&c.sym, &items))?;
            }
        }
        let n_ext = 1u64 << c.stark_info.stark_struct.n_bits_ext;
        max_scratch = max_scratch.max(plan.total_slots * n_ext);
        slots_by_sym.insert(c.sym.clone(), plan.total_slots);
        generated.push(GeneratedAir {
            name: c.name.clone(),
            base: c.base.clone(),
            sym: c.sym.clone(),
            nbits: c.nbits,
            cexp: c.cexp,
            n_ops: c.n_ops,
            slots: plan.total_slots,
        });
    }

    // Phase 3b: compile every emitted .cu that does not already have its .o
    // (parallel). The autotuner leaves the winning Q objects in `work`; the
    // generic-expression TUs (and, without autotune, the Q TUs) are compiled
    // here so the link step below is uniformly object-based.
    {
        let mut jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
        for sym in slots_by_sym.keys() {
            for cu in collect_artifacts(work, sym, "cu") {
                let obj = cu.with_extension("o");
                if !obj.exists() {
                    jobs.push((cu, obj));
                }
            }
        }
        // Plain scoped-thread fan-out (NOT rayon: a worker blocked on a child
        // nvcc inside the global pool can deadlock against the pipeline's
        // outer parallel bridges).
        let par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        for batch in jobs.chunks(par) {
            let errs: Vec<String> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|(cu, obj)| {
                        scope.spawn(move || -> Option<String> {
                            match tc.compile_tu(cu, obj, Some(work)) {
                                Ok((true, _)) => None,
                                Ok((false, log)) => Some(format!("nvcc failed for {}: {log}", cu.display())),
                                Err(e) => Some(format!("nvcc spawn failed for {}: {e}", cu.display())),
                            }
                        })
                    })
                    .collect();
                handles.into_iter().filter_map(|h| h.join().ok().flatten()).collect()
            });
            if let Some(e) = errs.into_iter().next() {
                anyhow::bail!(e);
            }
        }
    }

    // Phase 4: link one self-contained .so per placement whose sym was generated (parallel).
    let placed: Vec<&Placement> = placements.iter().filter(|p| slots_by_sym.contains_key(&p.sym)).collect();
    write_gen_log(work, &placed, &slots_by_sym)?;
    if !cfg.dry_run {
        placed.par_iter().try_for_each(|p| -> Result<()> {
            let dest = p.air_dir.join(format!("{}.exps.so", p.base));
            link_one(tc, work, &p.sym, &dest)
        })?;
    }

    Ok(GenSummary { placed: placed.len(), generated, skipped, max_scratch_bytes: max_scratch * 8 })
}

/// Link (or compile+link) one AIR's objects/sources into `dest`.
fn link_one(tc: &Toolchain, work: &Path, sym: &str, dest: &Path) -> Result<()> {
    let objs = collect_artifacts(work, sym, "o");
    if !objs.is_empty() {
        tc.link_objs(&objs, dest)
    } else {
        let cus = collect_artifacts(work, sym, "cu");
        tc.compile_link_cus(&cus, dest)
    }
}

/// `gen_<sym>.<ext>` + `gen_<sym>_c*.<ext>` present in `work`.
fn collect_artifacts(work: &Path, sym: &str, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let main = work.join(format!("gen_{sym}.{ext}"));
    if main.exists() {
        out.push(main);
    }
    let prefix = format!("gen_{sym}_c");
    if let Ok(entries) = std::fs::read_dir(work) {
        let mut chunks: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with(&prefix) && s.ends_with(&format!(".{ext}")))
            })
            .collect();
        chunks.sort();
        out.extend(chunks);
    }
    out
}

/// gen.log: one TAB-separated line per placement, written into the work dir as
/// an inspection aid (useful with `--keep-dir`); nothing reads it back.
fn write_gen_log(
    work: &Path,
    placed: &[&Placement],
    slots_by_sym: &std::collections::HashMap<String, u64>,
) -> Result<()> {
    let mut log = String::new();
    for p in placed {
        log.push_str(&format!("{}\t{}\t{}\t{}\n", p.name, p.base, p.sym, slots_by_sym[&p.sym]));
    }
    std::fs::write(work.join("gen.log"), log)?;
    Ok(())
}

fn print_summary(s: &GenSummary, cfg: &GenConfig) {
    let chunk_info = if let Some(chunk) = cfg.chunk { format!(", chunk={}", chunk) } else { String::new() };
    eprintln!(
        "generated {} kernels -> {} per-AIR .exps.so (CAP={}{}, max scratch {:.0}MB):",
        s.generated.len(),
        s.placed,
        cfg.cap,
        chunk_info,
        s.max_scratch_bytes as f64 / 1e6
    );
    for g in &s.generated {
        let chunked = if g.slots > 0 { format!("CHUNKED slots={}", g.slots) } else { "single".into() };
        eprintln!("  {:40} {}.exps.so  nBits={} cExp={} ops={} {}", g.name, g.base, g.nbits, g.cexp, g.n_ops, chunked);
    }
    for (name, why) in &s.skipped {
        eprintln!("  SKIP {name:38} {why}");
    }
}

/// A work dir that is either user-provided (kept) or a unique temp dir removed on drop.
struct WorkDir {
    path: PathBuf,
    temp: bool,
}

impl WorkDir {
    fn new(keep_dir: Option<PathBuf>) -> Result<Self> {
        match keep_dir {
            Some(p) => {
                std::fs::create_dir_all(&p)?;
                eprintln!("[exps-codegen] keeping generated code in {}", p.display());
                Ok(WorkDir { path: p, temp: false })
            }
            None => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static SEQ: AtomicU64 = AtomicU64::new(0);
                let seq = SEQ.fetch_add(1, Ordering::Relaxed);
                // Fresh directory, never a pre-existing path: `create_dir` fails if
                // something (a leftover or a planted symlink) is already there.
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0);
                let p = std::env::temp_dir().join(format!("genexps_{}_{}_{}", std::process::id(), seq, nanos));
                std::fs::create_dir(&p).with_context(|| format!("creating work dir {}", p.display()))?;
                Ok(WorkDir { path: p, temp: true })
            }
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        if self.temp {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
