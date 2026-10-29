//! No-spill autotuner: for one AIR, find the largest chunk size that compiles
//! with zero register spill (STACK=0 per `cuobjdump -res-usage`), bisecting down
//! from the start size.

use anyhow::Context;
use crate::emit::{chunked_program, emit_air, emit_exprs_chunked_tu, ChunkedExpr, ChunkedPick, ExprKey, SLOTS_CAP};
use crate::ir::{plan_chunks, plan_chunks_ops_pinned, ChunkPlan, Ir};
use crate::toolchain::Toolchain;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

// No-spill bisection bounds.
const CHUNK_MAX: usize = 512; // start chunk
const CHUNK_MIN: usize = 64; // give-up floor
const BIG_OPS: usize = 20000; // ops threshold for "large expressions"
const BIG_START_CHUNK: usize = 250; // autotuner start chunk for AIRs with > BIG_OPS ops

/// Start chunk for the recursion phases, whatever their op count.
///
/// This tuner takes the LARGEST chunk that does not spill, and largest is not fastest: past some size
/// the register pressure costs more than the launches it saves, and the recursion AIRs sit past that
/// point at CHUNK_MAX. Scoped to those phases rather than lowered globally because the optimum is
/// per-AIR: the recursion AIRs are the same handful of shapes on every proof, while a basic AIR is
/// whatever the project makes it -- opt.rs records Main preferring the opposite of Binary and Rom on
/// the neighbouring fold decision. `--exps-chunk N` pins it per run.
const RECURSION_START_CHUNK: usize = 250;

/// Whether an AIR's phase (the `sym` prefix, see `proof_phase`) is one of the recursion ones.
fn is_recursion_phase(sym: &str) -> bool {
    ["compressor_", "recursive_", "final_"].iter().any(|p| sym.starts_with(p))
}

/// `cuobjdump -res-usage` lines: a function header and its resource line.
static RE_FUNCTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Function\s+(\S+?):\s*$").unwrap());
static RE_STACK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"STACK:(\d+)").unwrap());
/// The once-per-launch table kernel (`*_pow`, one thread runs the hoisted-invariants program
/// straight-line) and its slices, by mangled name: `..._powPK11StepsParams...` / `..._pow_s3PK...`.
static RE_POW: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_pow(?:_s\d+)?(?:PK|P\d|\(|:|\s|$)").unwrap());

/// Per-function STACK (local memory bytes per thread) of an object or library, the max across
/// its per-arch entries (the autotuners gate on every arch they compile for). Keys are the
/// mangled names as `cuobjdump -res-usage` prints them.
pub(crate) fn resource_usage(obj: &Path) -> anyhow::Result<HashMap<String, i64>> {
    let out = std::process::Command::new(crate::toolchain::cuda_bin("cuobjdump"))
        .arg("-res-usage")
        .arg(obj)
        .output()
        .with_context(|| format!("running cuobjdump on {}", obj.display()))?;
    // A failed run must not pass as "no kernels, no local memory": the tuners and the audit
    // trust this map.
    if !out.status.success() {
        anyhow::bail!(
            "cuobjdump -res-usage failed on {} ({}): {}",
            obj.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut usage: HashMap<String, i64> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        if let Some(c) = RE_FUNCTION.captures(line) {
            cur = Some(c[1].to_string());
            continue;
        }
        if let (Some(f), Some(c)) = (&cur, RE_STACK.captures(line)) {
            let st: i64 = c[1].parse().unwrap_or(0);
            let e = usage.entry(f.clone()).or_insert(0);
            *e = (*e).max(st);
        }
    }
    Ok(usage)
}

/// Max STACK (spill) bytes across an object's per-row kernels. 0 = no spill; -1 = cuobjdump
/// could not be run (the caller bails with actionable guidance). The table kernel is left out:
/// it may spill freely, and gating on it made the Q autotuner reject Main at every chunk size
/// ("still spills at CHUNK_MIN").
pub(crate) fn max_stack(obj: &Path) -> i64 {
    match resource_usage(obj) {
        Ok(usage) => usage.iter().filter(|(f, _)| !RE_POW.is_match(f)).map(|(_, &st)| st).max().unwrap_or(0),
        Err(_) => -1,
    }
}

/// The smallest Q chunk probed: an expression shorter than `CHUNK_MIN` ops is probed once, whole.
pub(crate) fn q_chunk_floor(n_ops: usize) -> usize {
    CHUNK_MIN.min(n_ops.max(1))
}

/// Whether a bisection bracket (clean `lo`, spilling `hi`) is still worth probing: stop when it
/// is within ~1/8 of `lo`, the slot/grid payoff below that is noise.
fn bracket_open(lo: usize, hi: usize) -> bool {
    hi - lo > (lo / 8).max(8)
}

/// Returns the largest chunk size (halving from the start size) that compiles
/// with STACK=0, or None if it still spills at CHUNK_MIN (or fails to compile).
/// On success the winning objects are copied into `out_dir` so the final link
/// step reuses them without recompiling.
pub fn tune_chunk(tc: &Toolchain, ir: &Ir, sym: &str, n_ops: usize, out_dir: &Path) -> anyhow::Result<Option<usize>> {
    // Probe dir lives under out_dir (keyed by the unique sym) so parallel AIRs
    // never collide and /tmp churn can't wipe it mid-run.
    let probe = out_dir.join(format!(".probe_{sym}"));
    std::fs::create_dir_all(&probe)?;
    std::fs::write(probe.join("gen_common.cuh"), crate::emit::COMMON_CUH)?;

    let result = tune_inner(tc, ir, sym, n_ops, &probe, out_dir);
    let _ = std::fs::remove_dir_all(&probe);
    result
}

/// One probe: `Some(max STACK over the chunk TUs)` with their objects, or `None` when nvcc
/// rejected a TU (reported; treated as spilling by the callers).
type Probe = (Option<i64>, Vec<PathBuf>);

fn tune_inner(
    tc: &Toolchain,
    ir: &Ir,
    sym: &str,
    n_ops: usize,
    probe: &Path,
    out_dir: &Path,
) -> anyhow::Result<Option<usize>> {
    let start = if is_recursion_phase(sym) {
        RECURSION_START_CHUNK
    } else if n_ops > BIG_OPS {
        BIG_START_CHUNK
    } else {
        CHUNK_MAX
    };
    // Every probe compiles into a subdir of its own and keeps its objects, so the winner is
    // staged from them: re-emitting and recompiling it after the bisection was one whole extra
    // round on the critical path of every refined AIR.
    let compile_at = |chunk: usize| -> anyhow::Result<(Probe, usize)> {
        let dir = probe.join(chunk.to_string());
        std::fs::create_dir_all(&dir)?;
        let plan = plan_chunks(ir, chunk, sym)?;
        let files = emit_air(ir, &plan, sym);
        let objs: Vec<PathBuf> =
            files.iter().map(|(fname, _)| dir.join(fname.strip_suffix(".cu").unwrap().to_string() + ".o")).collect();
        for (fname, text) in &files {
            std::fs::write(dir.join(fname), text)?;
        }
        let results: Vec<(bool, String)> = files
            .par_iter()
            .zip(&objs)
            .map(|((fname, _), obj)| tc.compile_tu(&dir.join(fname), obj, Some(probe)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if let Some((_, err)) = results.iter().find(|(ok, _)| !ok) {
            let last = err.trim().lines().last().unwrap_or("");
            eprintln!("  [tune] {sym} chunk={chunk} COMPILE FAILED: {last}");
            return Ok(((None, objs), files.len()));
        }
        let stacks: Vec<i64> = objs.iter().map(|o| max_stack(o)).collect();
        if stacks.iter().any(|&s| s < 0) {
            anyhow::bail!(
                "cuobjdump could not read register usage for {sym} (not found or failed to run). \
                 Install CUDA binutils so the no-spill autotuner can verify STACK=0, \
                 which the generator needs to prove every kernel free of local memory."
            );
        }
        Ok(((Some(stacks.into_iter().max().unwrap_or(0)), objs), files.len()))
    };

    let mut chunk = n_ops.min(start);
    let mut last_spill: Option<usize> = None;
    while chunk >= q_chunk_floor(n_ops) {
        let ((st, objs), n_files) = compile_at(chunk)?;
        let Some(st) = st else { return Ok(None) };
        eprintln!("  [tune] {sym} chunk={chunk} ({n_files} TUs) -> STACK={st}");
        // Zero, nothing else: any local memory, even the 8 bytes a single-kernel Q once
        // kept (VirtualTableZisk1, whose split costs +30% of a sub-millisecond kernel), makes
        // the driver grow the context's local-memory pool, and the post-link audit would
        // reject the library.
        if st == 0 {
            // Halving overshoots: a spill of a few bytes at one size costs the whole
            // next size down, and with it double the cross-chunk slots. Bisect upward
            // between this clean size and the last spilling one; keep the largest clean.
            let (best_chunk, best_objs) = refine_up(sym, &compile_at, chunk, last_spill, objs)?;
            for obj in &best_objs {
                let dest = out_dir.join(obj.file_name().unwrap());
                std::fs::copy(obj, &dest)
                    .with_context(|| format!("staging tuned object {} -> {}", obj.display(), dest.display()))?;
            }
            return Ok(Some(best_chunk));
        }
        last_spill = Some(chunk);
        chunk /= 2;
    }
    Ok(None)
}

/// Bisection between a clean chunk `lo` and a spilling `hi` (None = `lo` was the
/// first probe, nothing above it to try). Returns the largest clean chunk found
/// and its compiled objects.
fn refine_up(
    sym: &str,
    compile_at: &dyn Fn(usize) -> anyhow::Result<(Probe, usize)>,
    lo: usize,
    hi: Option<usize>,
    lo_objs: Vec<PathBuf>,
) -> anyhow::Result<(usize, Vec<PathBuf>)> {
    let Some(mut hi) = hi else {
        return Ok((lo, lo_objs)); // first probe was clean: nothing above it to try
    };
    let (mut lo, mut lo_objs) = (lo, lo_objs);
    while bracket_open(lo, hi) {
        let mid = (lo + hi) / 2;
        let ((st, objs), _) = compile_at(mid)?;
        match st {
            Some(0) => {
                eprintln!("  [tune] {sym} chunk={mid} (refine) -> STACK=0");
                lo = mid;
                lo_objs = objs;
            }
            Some(st) => {
                eprintln!("  [tune] {sym} chunk={mid} (refine) -> STACK={st}");
                hi = mid;
            }
            None => {
                eprintln!("  [tune] {sym} chunk={mid} (refine) -> compile failed, treated as spilling");
                hi = mid;
            }
        }
    }
    Ok((lo, lo_objs))
}

/// Of `candidates` (the generic kernels a part TU holds), those whose straight-line kernel
/// `gen_<sym>_<tag>` keeps local memory in `usage`.
pub(crate) fn spilling_exprs(usage: &HashMap<String, i64>, sym: &str, candidates: &[ExprKey]) -> Vec<ExprKey> {
    let re = Regex::new(&format!(r"gen_{}_(xp?\d+(?:_\d+)?)PK", regex::escape(sym))).unwrap();
    let stack_by_tag = max_stack_by_tag(usage, &re);
    candidates.iter().copied().filter(|k| stack_by_tag.get(&k.tag()).is_some_and(|&st| st > 0)).collect()
}

/// Max STACK per kernel tag (the first capture of `re` on the mangled name), over `usage`.
fn max_stack_by_tag(usage: &HashMap<String, i64>, re: &Regex) -> HashMap<String, i64> {
    let mut out: HashMap<String, i64> = HashMap::new();
    for (f, &st) in usage {
        if let Some(c) = re.captures(f) {
            let e = out.entry(c[1].to_string()).or_insert(0);
            *e = (*e).max(st);
        }
    }
    out
}

// Chunk bounds of the generic kernels: smaller programs than Q, so the start is lower, and the
// floor keeps a 1024-op kernel under 32 launches per wave.
const GENERIC_CHUNK_START: usize = 256;
const GENERIC_CHUNK_MIN: usize = 32;

/// Bisection state of one chunked generic kernel: `lo` the largest chunk seen clean, `hi` the
/// smallest seen spilling, `cur` the size the next round compiles. Finished with `lo` = None
/// means dropped (spills down to the floor, or a cut wider than `SLOTS_CAP`).
struct Trial {
    lo: Option<usize>,
    hi: Option<usize>,
    cur: usize,
    done: bool,
}

impl Trial {
    fn new(cur: usize) -> Self {
        Trial { lo: None, hi: None, cur, done: false }
    }
    fn dropped(&self) -> bool {
        self.done && self.lo.is_none()
    }
    /// Record the probe at `cur` and pick the next size: halve while nothing is clean, then
    /// bisect the bracket (the Q tuner's `refine_up` rule).
    fn step(&mut self, spilled: bool) {
        if spilled {
            self.hi = Some(self.cur);
        } else {
            self.lo = Some(self.cur);
        }
        match (self.lo, self.hi) {
            (None, Some(hi)) => {
                if hi / 2 < GENERIC_CHUNK_MIN {
                    self.done = true;
                } else {
                    self.cur = hi / 2;
                }
            }
            (Some(_), None) => self.done = true, // first probe clean: nothing above it to try
            (Some(lo), Some(hi)) => {
                if bracket_open(lo, hi) {
                    self.cur = (lo + hi) / 2;
                } else {
                    self.done = true;
                }
            }
            (None, None) => unreachable!("a probe records lo or hi"),
        }
    }
}

/// Re-emit the generic kernels in `keys` chunked and tune each one's chunk size until every
/// chunk kernel shows STACK=0 (all archs): halving from `GENERIC_CHUNK_START`, then bisecting up
/// between the largest clean and the smallest spilling size. A kernel still spilling at
/// `GENERIC_CHUNK_MIN`, or whose cut is wider than `SLOTS_CAP`, is dropped (left to the interpreter). Each round compiles one TU, `gen_<sym>_cexprs_sp.cu` in `work`,
/// holding every surviving kernel -- the finished ones at their clean size -- so the last
/// object compiled is the one linked. Returns the chunked kernels and the dropped keys.
pub fn tune_exprs(
    tc: &Toolchain,
    work: &Path,
    sym: &str,
    xitems: &[(i64, Ir, u64)],
    keys: &[ExprKey],
) -> anyhow::Result<(Vec<ChunkedPick>, Vec<ExprKey>)> {
    let ir_by_id: HashMap<i64, &Ir> = xitems.iter().map(|(id, ir, _)| (*id, ir)).collect();
    let mut items: Vec<ChunkedExpr> = Vec::new();
    let mut excluded: Vec<ExprKey> = Vec::new();
    for &key in keys {
        let ce = match key {
            ExprKey::Single(id) => ir_by_id.get(&id).map(|ir| chunked_program(key, ir, None)),
            ExprKey::Pair(n, d) => {
                ir_by_id.get(&n).zip(ir_by_id.get(&d)).map(|(num, den)| chunked_program(key, num, Some(den)))
            }
        };
        match ce {
            Some(ce) => items.push(ce),
            None => {
                eprintln!("  [tune-generic] {sym} {}: its IR is not among the x items; excluded", key.tag());
                excluded.push(key);
            }
        }
    }
    let cu = work.join(format!("gen_{sym}_cexprs_sp.cu"));
    let obj = cu.with_extension("o");
    let mut trials: Vec<Trial> =
        items.iter().map(|ce| Trial::new(GENERIC_CHUNK_START.min(ce.ir.instrs.len()))).collect();
    let plan_of = |ce: &ChunkedExpr, size: usize| {
        plan_chunks_ops_pinned(&ce.ir.instrs, size, &format!("{sym} {}", ce.key.tag()), &ce.pinned())
    };
    // The plans of a round: every kernel at its current size (the finished ones at their clean
    // size, so the last object compiled holds the final set), None for a dropped one.
    let plans_of = |trials: &[Trial]| -> anyhow::Result<Vec<Option<ChunkPlan>>> {
        items
            .iter()
            .zip(trials)
            .map(|(ce, t)| match if t.done { t.lo } else { Some(t.cur) } {
                Some(size) => plan_of(ce, size).map(Some),
                None => Ok(None),
            })
            .collect()
    };
    // Compile the round's kernels; per kernel its max STACK over its chunk kernels.
    let re_chunk = Regex::new(&format!(r"gen_{}_(xp?\d+(?:_\d+)?)_c\d+PK", regex::escape(sym))).unwrap();
    let compile_at = |plans: &[Option<ChunkPlan>]| -> anyhow::Result<Vec<Option<i64>>> {
        let active: Vec<(&ChunkedExpr, &ChunkPlan)> =
            items.iter().zip(plans).filter_map(|(ce, p)| p.as_ref().map(|p| (ce, p))).collect();
        if active.is_empty() {
            return Ok(vec![None; items.len()]);
        }
        std::fs::write(&cu, emit_exprs_chunked_tu(sym, &active))?;
        let (ok, log) = tc.compile_tu(&cu, &obj, Some(work))?;
        if !ok {
            anyhow::bail!("nvcc failed for {}: {log}", cu.display());
        }
        let stack_of = max_stack_by_tag(&resource_usage(&obj)?, &re_chunk);
        Ok(items
            .iter()
            .zip(plans)
            .map(|(ce, p)| {
                let plan = p.as_ref()?;
                let tag = ce.key.tag();
                let st = stack_of.get(&tag).copied().unwrap_or(0);
                eprintln!(
                    "  [tune-generic] {sym} {tag} ops={} chunk={} ({} chunks, {} slots) -> STACK={st}",
                    ce.ir.instrs.len(),
                    plan.bounds.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0),
                    plan.n_chunks,
                    plan.total_slots
                );
                Some(st)
            })
            .collect())
    };
    let mut last_plans: Vec<Option<ChunkPlan>> = Vec::new();
    let mut last_was_final = false;
    while trials.iter().any(|t| !t.done) {
        let mut plans = plans_of(&trials)?;
        // A cut wider than SLOTS_CAP (Q's bound too) is dropped before spending a compile on it.
        for ((ce, t), plan) in items.iter().zip(trials.iter_mut()).zip(plans.iter_mut()) {
            if !t.done && plan.as_ref().is_some_and(|p| p.total_slots > SLOTS_CAP) {
                eprintln!(
                    "  [tune-generic] {sym} {}: cut wider than {SLOTS_CAP} slots; left to the interpreter",
                    ce.key.tag()
                );
                t.done = true;
                *plan = None;
            }
        }
        let stacks = compile_at(&plans)?;
        last_was_final = true;
        for (t, st) in trials.iter_mut().zip(&stacks) {
            if let (false, Some(st)) = (t.done, st) {
                t.step(*st > 0);
                last_was_final &= *st == 0;
            }
        }
        last_plans = plans;
    }
    // The last round compiled every finished kernel at its clean size unless one of the probes
    // in it spilled (its kernel then sits at a spilling size): re-emit the final set once.
    let plans = if last_was_final {
        last_plans
    } else {
        let plans = plans_of(&trials)?;
        compile_at(&plans)?;
        plans
    };
    let mut picks: Vec<ChunkedPick> = Vec::new();
    for ((ce, t), plan) in items.iter().zip(&trials).zip(plans) {
        match (t.lo, plan) {
            (Some(lo), Some(plan)) => picks.push((ce.key, plan.total_slots, lo)),
            _ => {
                if t.dropped() && t.hi.is_some() {
                    eprintln!(
                        "  [tune-generic] {sym} {}: still spills at chunk {}; left to the interpreter",
                        ce.key.tag(),
                        GENERIC_CHUNK_MIN
                    );
                }
                excluded.push(ce.key);
            }
        }
    }
    if picks.is_empty() {
        let _ = std::fs::remove_file(&cu);
        let _ = std::fs::remove_file(&obj);
    }
    Ok((picks, excluded))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bracket walk of one chunked generic kernel: halve while spilling, then bisect between
    /// the largest clean and the smallest spilling size until the gap is within ~1/8 of the clean
    /// one; a first clean probe ends it; spilling all the way down to the floor drops it.
    #[test]
    fn trial_halves_then_bisects_upward() {
        let mut t = Trial::new(256);
        t.step(true); // 256 spills
        assert_eq!((t.lo, t.hi, t.cur), (None, Some(256), 128));
        t.step(false); // 128 clean -> bisect 128..256
        assert_eq!((t.lo, t.hi, t.cur), (Some(128), Some(256), 192));
        t.step(false); // 192 clean -> 192..256
        assert_eq!((t.lo, t.cur), (Some(192), 224));
        t.step(true); // 224 spills -> 192..224, gap 32 > 24 -> 208
        assert_eq!((t.hi, t.cur), (Some(224), 208));
        t.step(true); // 208 spills -> 192..208, gap 16 <= 24 -> done at 192
        assert!(t.done && !t.dropped());
        assert_eq!(t.lo, Some(192));

        let mut first_clean = Trial::new(256);
        first_clean.step(false);
        assert!(first_clean.done && first_clean.lo == Some(256));

        let mut sink = Trial::new(64);
        sink.step(true); // 64 spills -> 32
        assert_eq!(sink.cur, 32);
        sink.step(true); // 32 spills -> below the floor: dropped
        assert!(sink.dropped());
    }

    /// A Q shorter than the floor used to be skipped unprobed ("still spills at CHUNK_MIN").
    #[test]
    fn a_short_q_is_probed_whole() {
        assert_eq!(q_chunk_floor(19), 19);
        assert_eq!(q_chunk_floor(5000), CHUNK_MIN);
    }

    /// The spill readout finds a candidate by its straight-line kernel's mangled name (singles,
    /// pairs) and is not fooled by a chunk kernel of the chunked TU (`_c<k>`) or by a prefix.
    #[test]
    fn spilling_exprs_matches_straight_line_kernel_names() {
        let sym = "basic_a0_36_b21_e22924";
        let mut usage: HashMap<String, i64> = HashMap::new();
        usage.insert("_Z33gen_basic_a0_36_b21_e22924_x18389PK11StepsParamsP6gl64_tmmmmmmmmmmj".into(), 256);
        usage.insert("_Z37gen_basic_a0_36_b21_e22924_xp22004_18389PK11StepsParamsP6gl64_tmmmmmm".into(), 256);
        usage.insert("_Z33gen_basic_a0_36_b21_e22924_x17736PK11StepsParamsP6gl64_tmmmmmmmmmmj".into(), 0);
        usage.insert("_Z36gen_basic_a0_36_b21_e22924_x1773_c0PK11StepsParamsP6gl64_tmmmmmmmmmmj".into(), 512);
        usage.insert("_Z31gen_basic_a0_36_b21_e22924_powPK11StepsParamsP6gl64_t".into(), 264);
        let candidates =
            [ExprKey::Single(18389), ExprKey::Single(17736), ExprKey::Single(1773), ExprKey::Pair(22004, 18389)];
        let got = spilling_exprs(&usage, sym, &candidates);
        assert_eq!(got, vec![ExprKey::Single(18389), ExprKey::Pair(22004, 18389)]);
    }
    /// The recursion phases take the smaller start chunk; a basic AIR keeps the old bounds. Pinned
    /// because the discriminator is the `sym` prefix, which `make_sym` builds from `proof_phase` --
    /// rename a phase there and this silently stops firing.
    #[test]
    fn only_the_recursion_phases_get_the_smaller_start() {
        for sym in ["compressor_a0_0_b19_e12056", "recursive_a0_1_b21_e900", "final_a0_0_b18_e400"] {
            assert!(super::is_recursion_phase(sym), "{sym} should be a recursion phase");
        }
        for sym in ["basic_a0_2_b19_e4634", "basic_a0_0_b19_e12056", "other_a0_0_b16_e10"] {
            assert!(!super::is_recursion_phase(sym), "{sym} should not be");
        }
    }
}
