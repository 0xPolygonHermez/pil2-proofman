//! No-spill autotuner: for one AIR, find the largest chunk size that compiles
//! with zero register spill (STACK=0 per `cuobjdump -res-usage`), bisecting down
//! from the start size.

use anyhow::Context;
use crate::emit::emit_air;
use crate::ir::{plan_chunks, Ir};
use crate::toolchain::Toolchain;
use rayon::prelude::*;
use regex::Regex;
use std::path::{Path, PathBuf};

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

/// Max STACK (spill) bytes across an object's per-arch entries. 0 = no spill;
/// -1 = cuobjdump could not be run (the caller bails with actionable guidance).
fn max_stack(obj: &Path) -> i64 {
    let out =
        match std::process::Command::new(crate::toolchain::cuda_bin("cuobjdump")).arg("-res-usage").arg(obj).output() {
            Ok(o) => o,
            Err(_) => return -1,
        };
    let text = String::from_utf8_lossy(&out.stdout);
    // Per function: the once-per-launch table kernel (`*_pow`, one thread runs
    // the hoisted-invariants program straight-line) may spill freely; its
    // stack must not gate the per-row chunk kernels -- it made the autotuner
    // reject Main at every chunk size ("still spills at CHUNK_MIN").
    // names are mangled: `..._powPK11StepsParams...` (table kernel) vs
    // `..._c12PK11StepsParams...` (chunk kernels) / `..._kernelPK...`
    let re_fn = Regex::new(r"Function\s+(\S+?):\s*$").unwrap();
    let re_pow = Regex::new(r"_pow(?:PK|P\d|\(|:|\s|$)").unwrap();
    let re_st = Regex::new(r"STACK:(\d+)").unwrap();
    let mut best = 0i64;
    let mut skip = false;
    for line in text.lines() {
        if let Some(c) = re_fn.captures(line) {
            skip = re_pow.is_match(&c[1]);
            continue;
        }
        if skip {
            continue;
        }
        if let Some(c) = re_st.captures(line) {
            if let Ok(v) = c[1].parse::<i64>() {
                best = best.max(v);
            }
        }
    }
    best
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
    let mut chunk = n_ops.min(start);
    let mut last_spill: Option<usize> = None;
    while chunk >= CHUNK_MIN {
        let plan = plan_chunks(ir, chunk, sym)?;
        let files = emit_air(ir, &plan, sym);

        // write + compile every TU (parallel); each .o sits next to its .cu in probe_dir.
        let objs: Vec<PathBuf> =
            files.iter().map(|(fname, _)| probe.join(fname.strip_suffix(".cu").unwrap().to_string() + ".o")).collect();
        for (fname, text) in &files {
            std::fs::write(probe.join(fname), text)?;
        }
        let results: Vec<(bool, String)> = files
            .par_iter()
            .zip(&objs)
            .map(|((fname, _), obj)| tc.compile_tu(&probe.join(fname), obj, Some(probe)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        if let Some((_, err)) = results.iter().find(|(ok, _)| !ok) {
            let last = err.trim().lines().last().unwrap_or("");
            eprintln!("  [tune] {sym} chunk={chunk} COMPILE FAILED: {last}");
            return Ok(None);
        }

        let stacks: Vec<i64> = objs.iter().map(|o| max_stack(o)).collect();
        if stacks.iter().any(|&s| s < 0) {
            anyhow::bail!(
                "cuobjdump could not read register usage for {sym} (not found or failed to run). \
                 Install CUDA binutils so the no-spill autotuner can verify STACK=0, \
                 or pass --chunk <N> to skip autotuning."
            );
        }
        let st = stacks.into_iter().max().unwrap_or(0);
        eprintln!("  [tune] {sym} chunk={chunk} ({} TUs) -> STACK={st}", files.len());
        // A single-kernel AIR (chunk covers every op) may keep a few bytes of
        // local memory: splitting it in two costs a launch plus cross-chunk
        // slots, which measured worse (VirtualTableZisk1: 8-byte spill at 280
        // ops -> 2 chunks -> +30% kernel time). Chunked kernels stay at zero.
        const SINGLE_SPILL_TOL: i64 = 32;
        if st == 0 || (chunk >= n_ops && st <= SINGLE_SPILL_TOL) {
            // Halving overshoots: a spill of a few bytes at one size costs the whole
            // next size down, and with it double the cross-chunk slots. Bisect upward
            // between this clean size and the last spilling one; keep the largest clean.
            let (best_chunk, best_objs) = refine_up(tc, ir, sym, probe, chunk, last_spill, objs)?;
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
/// and its compiled objects (the probe dir is shared, so objects are re-emitted
/// for the winner when a larger probe overwrote them).
fn refine_up(
    tc: &Toolchain,
    ir: &Ir,
    sym: &str,
    probe: &Path,
    lo: usize,
    hi: Option<usize>,
    lo_objs: Vec<PathBuf>,
) -> anyhow::Result<(usize, Vec<PathBuf>)> {
    let compile_at = |chunk: usize| -> anyhow::Result<(Option<i64>, Vec<PathBuf>)> {
        let plan = plan_chunks(ir, chunk, sym)?;
        let files = emit_air(ir, &plan, sym);
        let objs: Vec<PathBuf> =
            files.iter().map(|(fname, _)| probe.join(fname.strip_suffix(".cu").unwrap().to_string() + ".o")).collect();
        for (fname, text) in &files {
            std::fs::write(probe.join(fname), text)?;
        }
        let results: Vec<(bool, String)> = files
            .par_iter()
            .zip(&objs)
            .map(|((fname, _), obj)| tc.compile_tu(&probe.join(fname), obj, Some(probe)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if results.iter().any(|(ok, _)| !ok) {
            return Ok((None, objs));
        }
        let stacks: Vec<i64> = objs.iter().map(|o| max_stack(o)).collect();
        if stacks.iter().any(|&s| s < 0) {
            return Ok((None, objs));
        }
        Ok((Some(stacks.into_iter().max().unwrap_or(0)), objs))
    };
    let Some(mut hi) = hi else {
        return Ok((lo, lo_objs)); // first probe was clean: nothing above it to try
    };
    let mut lo = lo;
    // stop when the bracket is within ~1/8 of lo: the slot/grid payoff below that is noise
    while hi - lo > (lo / 8).max(8) {
        let mid = (lo + hi) / 2;
        let (st, _) = compile_at(mid)?;
        match st {
            Some(0) => {
                eprintln!("  [tune] {sym} chunk={mid} (refine) -> STACK=0");
                lo = mid;
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
    // re-emit the winner so the returned objects match `lo` (probes overwrote them)
    let (_, objs) = compile_at(lo)?;
    Ok((lo, objs))
}

#[cfg(test)]
mod tests {
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
