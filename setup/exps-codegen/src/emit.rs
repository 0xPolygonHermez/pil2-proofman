//! CUDA source emission: the straight-line / chunked kernel text for one AIR's
//! Q expression, the shared `gen_common.cuh` header, and the fixed C-ABI
//! exports. The template whitespace is deliberate — the emitted `.cu` is what
//! nvcc compiles, so treat these strings as code, not free-form text.

use crate::ir::{ChunkPlan, Instr, Ir, Operand};
use std::collections::HashSet;

/// CUDA threads/block of the generated kernels (single source of truth, baked into each launcher).
pub const GEN_BLK: u64 = 256;
/// Chunk kernels bundled per .cu — amortizes the gen_common.cuh header parse across the batch.
pub const CHUNKS_PER_TU: usize = 8;

/// The header every generated TU `#include`s.
pub const COMMON_CUH: &str = r#"#pragma once
#include "goldilocks_tooling.cuh"
#include "steps.hpp"
#include "goldilocks_trace_layout.cuh"
#include <cstdint>
struct g3 { gl64_t a,b,c; };
__device__ __forceinline__ g3 cg_mul33(g3 x, g3 y){
  gl64_t A=(x.a+x.b)*(y.a+y.b), B=(x.a+x.c)*(y.a+y.c), C=(x.b+x.c)*(y.b+y.c);
  gl64_t D=x.a*y.a, E=x.b*y.b, F=x.c*y.c, G=D-E; g3 r; r.a=(C+G)-F; r.b=(((A+C)-E)-E)-D; r.c=B-G; return r; }
__device__ __forceinline__ g3 cg_mul31(g3 x, gl64_t s){ g3 r; r.a=x.a*s; r.b=x.b*s; r.c=x.c*s; return r; }
__device__ __forceinline__ g3 cg_mul13(gl64_t s, g3 y){ g3 r; r.a=y.a*s; r.b=y.b*s; r.c=y.c*s; return r; }
__device__ __forceinline__ g3 cg_add33(g3 x, g3 y){ g3 r; r.a=x.a+y.a; r.b=x.b+y.b; r.c=x.c+y.c; return r; }
__device__ __forceinline__ g3 cg_add31(g3 x, gl64_t s){ g3 r; r.a=x.a+s; r.b=x.b; r.c=x.c; return r; }
__device__ __forceinline__ g3 cg_add13(gl64_t s, g3 y){ g3 r; r.a=y.a+s; r.b=y.b; r.c=y.c; return r; }
__device__ __forceinline__ g3 cg_sub33(g3 x, g3 y){ g3 r; r.a=x.a-y.a; r.b=x.b-y.b; r.c=x.c-y.c; return r; }
__device__ __forceinline__ g3 cg_sub31(g3 x, gl64_t s){ g3 r; r.a=x.a-s; r.b=x.b; r.c=x.c; return r; }
__device__ __forceinline__ g3 cg_sub13(gl64_t s, g3 y){ g3 r; r.a=s-y.a; r.b=-y.b; r.c=-y.c; return r; }
"#;

/// FORCE_TILED_LAYOUT=1 mirrors the `-DFORCE_TILED_LAYOUT` C++ build; the two MUST match, or the
/// kernel reads committed sections in a layout the LDE/Merkle didn't write. See [`cm_layout`].
fn force_tiled() -> bool {
    std::env::var("FORCE_TILED_LAYOUT").map(|v| v == "1").unwrap_or(false)
}

/// Committed-section layout the generated kernel reads, mirroring `resolveLayout(nBits,nCols)` in
/// goldilocks_trace_layout.cuh. Keyed on small-domain nBits (not extended), like expressions_gpu.cu.
fn cm_layout(n_bits: u64, n_cols: u64) -> &'static str {
    if force_tiled() || (n_bits <= 17 && n_cols > 500) {
        "Layout::ColMajorTiled"
    } else {
        "Layout::ColMajor"
    }
}

/// Storage layout of the fixed (const) section: always ColMajorTiled, matching
/// `fixedLayout()` and the const-tree build.
const CONST_LAYOUT: &str = "Layout::ColMajorTiled";

fn rowexpr(stride: i64) -> String {
    if stride == 0 {
        "row".to_string()
    } else {
        format!("((row+({stride}ll))&MASK)")
    }
}

/// Lines that materialize a non-tmp operand into the local `name`.
fn load_lines(opnd: &Operand, name: &str, ir: &Ir) -> Vec<String> {
    match opnd {
        Operand::Num(v) => vec![format!("  gl64_t {name}(uint64_t({v}ull));")],
        Operand::Zi => vec![format!("  gl64_t {name} = aux[off_zi + row];")],
        Operand::Pub { id } => vec![format!("  gl64_t {name} = pub[{id}];")],
        Operand::Ch { base } => {
            let i = *base;
            vec![format!("  g3 {name}; {name}.a=ch[{i}]; {name}.b=ch[{}]; {name}.c=ch[{}];", i + 1, i + 2)]
        }
        Operand::Av { pos, dim } | Operand::Agv { pos, dim } => {
            let arr = if matches!(opnd, Operand::Av { .. }) { "av" } else { "agv" };
            let i = *pos;
            if *dim == 1 {
                vec![format!("  gl64_t {name} = {arr}[{i}];")]
            } else {
                vec![format!("  g3 {name}; {name}.a={arr}[{i}]; {name}.b={arr}[{}]; {name}.c={arr}[{}];", i + 1, i + 2)]
            }
        }
        Operand::Const { id, stride } => {
            // const sections are stored fixedLayout() (ColMajorTiled), like the const-tree build.
            vec![format!(
                "  gl64_t {name} = cst[OFF({},{id},NExt,{},{CONST_LAYOUT})];",
                rowexpr(*stride),
                ir.n_constants
            )]
        }
        Operand::Cm { stage, pos, dim, stride } => {
            let row = rowexpr(*stride);
            let n_cols = ir.ncols[stage];
            // committed section layout = resolveLayout(small nBits, sectionNCols), matching the
            // commit/LDE writer and the built-in evaluator (expressions_gpu.cu).
            let lyt = cm_layout(ir.n_bits, n_cols);
            if *dim == 1 {
                vec![format!("  gl64_t {name} = aux[off_cm{stage} + OFF({row},{pos},NExt,{n_cols},{lyt})];")]
            } else {
                vec![format!(
                    "  g3 {name}; {name}.a=aux[off_cm{stage}+OFF({row},{pos},NExt,{n_cols},{lyt})]; {name}.b=aux[off_cm{stage}+OFF({row},{},NExt,{n_cols},{lyt})]; {name}.c=aux[off_cm{stage}+OFF({row},{},NExt,{n_cols},{lyt})];",
                    pos + 1,
                    pos + 2
                )]
            }
        }
        Operand::Tmp { .. } => unreachable!("tmp operands are not loaded"),
    }
}

/// Emit lines for one op; tmp operands -> t{id} (must already exist). Returns
/// (lines, is_out). The caller adds tmp dsts to `declared` afterward.
fn emit_op(instr: &Instr, ir: &Ir, declared: &HashSet<u64>) -> (Vec<String>, bool) {
    let mut lines = Vec::new();
    let a_val = match instr.a.as_tmp() {
        Some((id, _)) => format!("t{id}"),
        None => {
            lines.extend(load_lines(&instr.a, &format!("a{}", instr.idx), ir));
            format!("a{}", instr.idx)
        }
    };
    let b_val = match instr.b.as_tmp() {
        Some((id, _)) => format!("t{id}"),
        None => {
            lines.extend(load_lines(&instr.b, &format!("b{}", instr.idx), ir));
            format!("b{}", instr.idx)
        }
    };
    let a_dim = instr.a.dim();
    let b_dim = instr.b.dim();
    let dst_dim = instr.ddim;
    let is_out = !instr.dst_is_tmp;
    let dst = if is_out { "qq".to_string() } else { format!("t{}", instr.dst_id.unwrap()) };
    let decl = if !is_out && instr.dst_id.is_some_and(|id| declared.contains(&id)) {
        ""
    } else if dst_dim == 1 {
        "gl64_t "
    } else {
        "g3 "
    };
    if dst_dim == 1 {
        let op_symbol = match instr.op.as_str() {
            "add" => "+",
            "sub" => "-",
            "mul" => "*",
            other => panic!("unexpected op {other}"),
        };
        lines.push(format!("  {decl}{dst} = {a_val} {op_symbol} {b_val};"));
    } else {
        lines.push(format!("  {decl}{dst} = cg_{}{a_dim}{b_dim}({a_val},{b_val});", instr.op));
    }
    (lines, is_out)
}

/// The final write of `qq` into the q buffer (out_dim 3 vs base-field padded to 3).
fn store_qq(out_dim: u64) -> &'static str {
    // q (the cmQ output) has 3 cols, so resolveLayout(nBits,3) is normally ColMajor
    // (nCols <= 500) — matches how the cmQ commit/Merkle reads it back. Under
    // FORCE_TILED_LAYOUT that decision flips with everything else (see cm_layout).
    match (force_tiled(), out_dim == 3) {
        (false, true) =>
            "    q[OFF(row,0,NExt,3,Layout::ColMajor)]=qq.a; q[OFF(row,1,NExt,3,Layout::ColMajor)]=qq.b; q[OFF(row,2,NExt,3,Layout::ColMajor)]=qq.c;",
        (false, false) =>
            "    q[OFF(row,0,NExt,3,Layout::ColMajor)]=qq; q[OFF(row,1,NExt,3,Layout::ColMajor)]=gl64_t(uint64_t(0)); q[OFF(row,2,NExt,3,Layout::ColMajor)]=gl64_t(uint64_t(0));",
        (true, true) =>
            "    q[OFF(row,0,NExt,3,Layout::ColMajorTiled)]=qq.a; q[OFF(row,1,NExt,3,Layout::ColMajorTiled)]=qq.b; q[OFF(row,2,NExt,3,Layout::ColMajorTiled)]=qq.c;",
        (true, false) =>
            "    q[OFF(row,0,NExt,3,Layout::ColMajorTiled)]=qq; q[OFF(row,1,NExt,3,Layout::ColMajorTiled)]=gl64_t(uint64_t(0)); q[OFF(row,2,NExt,3,Layout::ColMajorTiled)]=gl64_t(uint64_t(0));",
    }
}

/// The fixed C-ABI the loader dlsym's from each `.exps.so`.
fn c_abi_exports(sym: &str, n_slots: u64) -> String {
    format!(
        r#"extern "C" void exps_launch(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
    launch_gen_{sym}(d_params, q, scratch, scratchElems, NExt, off_cm1, off_cm2, off_cm3, off_zi, stream);
}}
extern "C" unsigned long long exps_min_scratch() {{ return {n_slots} * {GEN_BLK}ull; }}"#
    )
}

/// Small-expression path: kernel + launcher + C-ABI exports in ONE self-contained TU.
fn single_kernel_tu(sym: &str, kernel: &str, launcher_body: &str, n_slots: u64) -> String {
    format!(
        r#"// AUTO-GENERATED Q kernel for {sym} (single kernel, no scratch)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{kernel}
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
{launcher_body}
}}
#undef OFF
{}
"#,
        c_abi_exports(sym, n_slots)
    )
}

/// A batch of chunk kernels in one TU, each followed by a C-ABI host wrapper performing its launch.
fn chunk_tu(sym: &str, lo: usize, hi: usize, kernels: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (offset, kernel) in kernels[lo..hi].iter().enumerate() {
        let i = lo + offset;
        parts.push(kernel.clone());
        parts.push(format!(
            r#"extern "C" void run_{sym}_c{i}(uint64_t grid, uint64_t blk, cudaStream_t stream, StepsParams* d_params,
    gl64_t* q, gl64_t* scratch, uint64_t NExt, uint64_t base,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  gen_{sym}_c{i}<<<grid,blk,0,stream>>>(d_params,q,scratch,NExt,base,off_cm1,off_cm2,off_cm3,off_zi);
}}"#
        ));
    }
    format!(
        r#"// AUTO-GENERATED Q chunk kernels {lo}..{} for {sym}
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{}
#undef OFF
"#,
        hi - 1,
        parts.join("\n")
    )
}

/// The launcher TU: cross-TU `run_*` decls + the adaptive-grid wave loop + C-ABI.
fn launcher_tu(sym: &str, n_chunks: usize, total_slots: u64) -> String {
    let decls: Vec<String> = (0..n_chunks)
        .map(|i| {
            format!(
                "extern \"C\" void run_{sym}_c{i}(uint64_t, uint64_t, cudaStream_t, StepsParams*, gl64_t*, gl64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);"
            )
        })
        .collect();
    let calls: Vec<String> = (0..n_chunks)
        .map(|i| {
            format!("    run_{sym}_c{i}(grid, BLK, stream, d_params, q, scratch, NExt, base, off_cm1, off_cm2, off_cm3, off_zi);")
        })
        .collect();
    format!(
        r#"// AUTO-GENERATED Q launcher for {sym} (cross-boundary temps={total_slots}, {n_chunks} chunks)
#include "gen_common.cuh"
{}
// adaptive grid: shrink so total_slots*grid*BLK <= scratchElems (per-wave scratch fits the tmp region);
// each chunk kernel computes WAVE=gridDim*blockDim at runtime, so any grid is correct.
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
  const uint64_t BLK = {GEN_BLK}ull;
  uint64_t grid = {total_slots}ull ? (scratchElems / ({total_slots}ull*BLK)) : 512ull;
  if (grid > 512ull) grid = 512ull;
  if (grid < 1ull) grid = 1ull;
  const uint64_t WAVE = grid * BLK;
  for (uint64_t base=0; base<NExt; base+=WAVE) {{
{}
  }}
}}
{}
"#,
        decls.join("\n"),
        calls.join("\n"),
        c_abi_exports(sym, total_slots)
    )
}

/// Emit the per-AIR TU source files. Returns `(filename, contents)` pairs:
/// one self-contained TU for a single kernel, or a launcher TU + N chunk TUs
/// when chunked. `plan.total_slots` is the cross-chunk cut width.
pub fn emit_air(ir: &Ir, plan: &ChunkPlan, sym: &str) -> Vec<(String, String)> {
    if plan.n_chunks <= 1 {
        // single straight-line kernel (small expression)
        let mut body: Vec<String> = Vec::new();
        let mut declared: HashSet<u64> = HashSet::new();
        for instr in &ir.instrs {
            let (op_lines, is_out) = emit_op(instr, ir, &declared);
            body.extend(op_lines);
            if !is_out {
                declared.insert(instr.dst_id.unwrap());
            }
        }
        let kernel = format!(
            r#"__global__ void gen_{sym}_kernel(const StepsParams* __restrict__ P, gl64_t* __restrict__ q,
    uint64_t NExt, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<NExt; row+=gridDim.x*blockDim.x) {{
{}
{}
  }}
}}"#,
            body.join("\n"),
            store_qq(plan.out_dim)
        );
        let launcher_body = format!(
            "  (void)scratch; (void)scratchElems; gen_{sym}_kernel<<<512,256,0,stream>>>(d_params,q,NExt,off_cm1,off_cm2,off_cm3,off_zi);"
        );
        return vec![(format!("gen_{sym}.cu"), single_kernel_tu(sym, &kernel, &launcher_body, 0))];
    }

    // chunked (tiled): one register-bounded kernel per chunk.
    let mut kernels: Vec<String> = Vec::with_capacity(plan.n_chunks);
    for chunk_idx in 0..plan.n_chunks {
        let lo_op = chunk_idx * plan.chunk;
        let hi_op = ((chunk_idx + 1) * plan.chunk).min(ir.instrs.len());
        let chunk_ops = &ir.instrs[lo_op..hi_op];

        let mut used_temps: HashSet<u64> = HashSet::new();
        for instr in chunk_ops {
            for opnd in [&instr.a, &instr.b] {
                if let Some((tid, _)) = opnd.as_tmp() {
                    used_temps.insert(tid);
                }
            }
        }
        let mut live_in: Vec<u64> = used_temps
            .iter()
            .copied()
            .filter(|t| plan.cut_temps.contains(t) && plan.chunk_of(plan.def_idx[t]) < chunk_idx)
            .collect();
        live_in.sort_unstable();
        let mut live_out: Vec<u64> =
            plan.cut_temps.iter().copied().filter(|t| plan.chunk_of(plan.def_idx[t]) == chunk_idx).collect();
        live_out.sort_unstable();

        let mut declared: HashSet<u64> = HashSet::new();
        let mut lines: Vec<String> = Vec::new();
        for &t in &live_in {
            let slot_base = plan.slot_index(t);
            if plan.dim_of[&t] == 1 {
                lines.push(format!("  gl64_t t{t} = scratch[{slot_base}ull*WAVE + lo_];"));
            } else {
                lines.push(format!(
                    "  g3 t{t}; t{t}.a=scratch[{slot_base}ull*WAVE+lo_]; t{t}.b=scratch[{}ull*WAVE+lo_]; t{t}.c=scratch[{}ull*WAVE+lo_];",
                    slot_base + 1,
                    slot_base + 2
                ));
            }
            declared.insert(t);
        }
        for instr in chunk_ops {
            let (op_lines, is_out) = emit_op(instr, ir, &declared);
            lines.extend(op_lines);
            if !is_out {
                declared.insert(instr.dst_id.unwrap());
            }
        }
        for &t in &live_out {
            let slot_base = plan.slot_index(t);
            if plan.dim_of[&t] == 1 {
                lines.push(format!("  scratch[{slot_base}ull*WAVE + lo_] = t{t};"));
            } else {
                lines.push(format!(
                    "  scratch[{slot_base}ull*WAVE+lo_]=t{t}.a; scratch[{}ull*WAVE+lo_]=t{t}.b; scratch[{}ull*WAVE+lo_]=t{t}.c;",
                    slot_base + 1,
                    slot_base + 2
                ));
            }
        }
        if chunk_idx == plan.n_chunks - 1 {
            lines.push(store_qq(plan.out_dim).to_string());
        }
        kernels.push(format!(
            r#"__global__ void gen_{sym}_c{chunk_idx}(const StepsParams* __restrict__ P, gl64_t* __restrict__ q, gl64_t* __restrict__ scratch,
    uint64_t NExt, uint64_t tileBase, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1; const uint64_t WAVE = (uint64_t)gridDim.x*blockDim.x;
  const uint64_t lo_ = blockIdx.x*blockDim.x + threadIdx.x; const uint64_t row = tileBase + lo_;
  if (row >= NExt) return;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
{}
}}"#,
            lines.join("\n")
        ));
    }

    let mut files: Vec<(String, String)> =
        vec![(format!("gen_{sym}.cu"), launcher_tu(sym, plan.n_chunks, plan.total_slots))];
    let mut tu = 0usize;
    let mut lo = 0usize;
    while lo < plan.n_chunks {
        let hi = (lo + CHUNKS_PER_TU).min(plan.n_chunks);
        files.push((format!("gen_{sym}_c{tu}.cu"), chunk_tu(sym, lo, hi, &kernels)));
        tu += 1;
        lo += CHUNKS_PER_TU;
    }
    files
}
