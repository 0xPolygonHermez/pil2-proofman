//! No-spill autotuner: for one AIR, find the largest chunk size that compiles
//! with zero register spill (STACK=0 per `cuobjdump -res-usage`), bisecting down
//! from the start size.

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

/// Max STACK (spill) bytes across an object's per-arch entries. 0 = no spill;
/// -1 = cuobjdump could not be read (treated as "not zero" -> keep shrinking).
fn max_stack(obj: &Path) -> i64 {
    let out = match std::process::Command::new("cuobjdump").arg("-res-usage").arg(obj).output() {
        Ok(o) => o,
        Err(_) => return -1,
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let re = Regex::new(r"STACK:(\d+)").unwrap();
    re.captures_iter(&text).filter_map(|c| c[1].parse::<i64>().ok()).max().unwrap_or(0)
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

fn tune_inner(tc: &Toolchain, ir: &Ir, sym: &str, n_ops: usize, probe: &Path, out_dir: &Path) -> anyhow::Result<Option<usize>> {
    let mut chunk = n_ops.min(if n_ops > BIG_OPS { BIG_START_CHUNK } else { CHUNK_MAX });
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

        let st = objs.iter().map(|o| max_stack(o)).max().unwrap_or(0);
        eprintln!("  [tune] {sym} chunk={chunk} ({} TUs) -> STACK={st}", files.len());
        if st == 0 {
            for obj in &objs {
                let _ = std::fs::copy(obj, out_dir.join(obj.file_name().unwrap()));
            }
            return Ok(Some(chunk));
        }
        chunk /= 2;
    }
    Ok(None)
}
