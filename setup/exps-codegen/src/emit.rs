//! CUDA source emission: the straight-line / chunked kernel text for one AIR's
//! Q expression, one trace-domain kernel per other covered expression
//! (hint fields, im columns, ... -- `emit_exprs_tu`, dispatched by expId via
//! `exps_expr_covered` / `exps_launch_expr`), the shared `gen_common.cuh`
//! header, and the fixed C-ABI
//! exports. The template whitespace is deliberate — the emitted `.cu` is what
//! nvcc compiles, so treat these strings as code, not free-form text.

use crate::ir::{ChunkPlan, Instr, Ir, Operand};
use std::collections::{HashMap, HashSet};

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
#include <cstdio>
#include <cstdlib>
// cg_* = the codegen's Goldilocks cubic-extension helpers (g3 = one Fp3 element).
// Prefixed to avoid collisions with the prover headers this TU also includes;
// digit suffixes are operand dims (mul33 = 3x3, mul31 = 3x1, inv3 = cubic inverse).
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
static __device__ __noinline__ g3 cg_inv3(g3 v){
  gl64_t aa=v.a*v.a, ac=v.a*v.c, ba=v.b*v.a, bb=v.b*v.b, bc=v.b*v.c, cc=v.c*v.c;
  gl64_t aaa=aa*v.a, aac=aa*v.c, abc=ba*v.c, abb=ba*v.b, acc=ac*v.c, bbb=bb*v.b, bcc=bc*v.c, ccc=cc*v.c;
  gl64_t t = abc+abc+abc+abb-aaa-aac-aac-acc-bbb+bcc-ccc;
  gl64_t tinv = t.reciprocal();
  g3 r; r.a=(bc+bb-aa-ac-ac-cc)*tinv; r.b=(ba-cc)*tinv; r.c=(ac+cc-bb)*tinv; return r; }
__device__ __forceinline__ bool g3_is_zero(g3 v){ return v.a.is_zero() && v.b.is_zero() && v.c.is_zero(); }
__device__ __forceinline__ void exps_expr_store(gl64_t* dest, uint64_t row, uint64_t destDomain,
    uint64_t stagePos, uint64_t stageCols, uint64_t destDim, uint32_t destExpr,
    uint64_t mode, uint64_t scalar, g3 v, uint64_t vdim)
{
  uint64_t idx0, idx1 = 0, idx2 = 0;
  if (destExpr) {
    idx0 = row*destDim; idx1 = idx0+1; idx2 = idx0+2;
  } else {
    const Layout lyt = resolveLayout(63 - __clzll(destDomain), stageCols);
    idx0 = getBufferOffset(row, stagePos+0, destDomain, stageCols, lyt);
    if (destDim > 1) {
      idx1 = getBufferOffset(row, stagePos+1, destDomain, stageCols, lyt);
      idx2 = getBufferOffset(row, stagePos+2, destDomain, stageCols, lyt);
    }
  }
  if (mode == 0) {                       // WRITE (pad to destDim)
    if (vdim == 1) { v.b = gl64_t(uint64_t(0)); v.c = gl64_t(uint64_t(0)); }
    dest[idx0] = v.a;
    if (destDim > 1) { dest[idx1] = v.b; dest[idx2] = v.c; }
    return;
  }
  g3 inv;
  if (vdim == 1) { inv.a = v.a.reciprocal(); inv.b = gl64_t(uint64_t(0)); inv.c = gl64_t(uint64_t(0)); }
  else          { inv = cg_inv3(v); }
  if (mode == 1) {                       // MUL_INV: dest = dest * v^-1
    if (destDim == 1) { dest[idx0] = dest[idx0] * inv.a; return; }
    g3 cur; cur.a = dest[idx0]; cur.b = dest[idx1]; cur.c = dest[idx2];
    g3 r = (vdim == 1) ? cg_mul31(cur, inv.a) : cg_mul33(cur, inv);
    dest[idx0] = r.a; dest[idx1] = r.b; dest[idx2] = r.c;
    return;
  }
  // WRITE_INV: dest = scalar * v^-1
  gl64_t sc(scalar);
  if (destDim == 1) { dest[idx0] = sc * inv.a; return; }
  g3 r = cg_mul31(inv, sc);
  dest[idx0] = r.a; dest[idx1] = r.b; dest[idx2] = r.c;
}
"#;

/// Committed-section layout the generated kernel reads, mirroring `resolveLayout(nBits,nCols)` in
/// goldilocks_trace_layout.cuh: always ColMajor since the in-house ColMajor NTT engine serves every
/// shape. A layout change there requires regenerating the exps kernels with a matching change here.
fn cm_layout(_n_bits: u64, _n_cols: u64) -> &'static str {
    "Layout::ColMajor"
}

/// Storage layout of the fixed (const) section: ColMajor, matching `fixedLayout()` and the
/// const-tree build.
const CONST_LAYOUT: &str = "Layout::ColMajor";

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
        Operand::Pow { base, j } if ir.pow_in_regs => {
            vec![format!("  g3 {name} = pwreg_{base}_{j};")]
        }
        Operand::Pow { base, j } => {
            let i = ir.pow_offset(*base) + 3 * *j;
            vec![format!("  g3 {name}; {name}.a=pw[{i}]; {name}.b=pw[{}]; {name}.c=pw[{}];", i + 1, i + 2)]
        }
        Operand::Tab { idx, dim } => {
            let i = ir.pow_words() + *idx;
            if *dim == 1 {
                vec![format!("  gl64_t {name} = pw[{i}];")]
            } else {
                vec![format!("  g3 {name}; {name}.a=pw[{i}]; {name}.b=pw[{}]; {name}.c=pw[{}];", i + 1, i + 2)]
            }
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
        Operand::Custom { off, pos, dim, ncols, stride } => {
            // Custom-commit sections are fixed/preprocessed data stored
            // fixedLayout() (ColMajor) inside pCustomCommitsFixed, like const.
            let row = rowexpr(*stride);
            if *dim == 1 {
                vec![format!("  gl64_t {name} = ccf[{off}ull + OFF({row},{pos},NExt,{ncols},{CONST_LAYOUT})];")]
            } else {
                vec![format!(
                    "  g3 {name}; {name}.a=ccf[{off}ull+OFF({row},{pos},NExt,{ncols},{CONST_LAYOUT})]; {name}.b=ccf[{off}ull+OFF({row},{},NExt,{ncols},{CONST_LAYOUT})]; {name}.c=ccf[{off}ull+OFF({row},{},NExt,{ncols},{CONST_LAYOUT})];",
                    pos + 1,
                    pos + 2
                )]
            }
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
    // q (the cmQ output) is ColMajor like everything else (resolveLayout) -- matches how the cmQ
    // commit/Merkle reads it back.
    if out_dim == 3 {
        "    q[OFF(row,0,NExt,3,Layout::ColMajor)]=qq.a; q[OFF(row,1,NExt,3,Layout::ColMajor)]=qq.b; q[OFF(row,2,NExt,3,Layout::ColMajor)]=qq.c;"
    } else {
        "    q[OFF(row,0,NExt,3,Layout::ColMajor)]=qq; q[OFF(row,1,NExt,3,Layout::ColMajor)]=gl64_t(uint64_t(0)); q[OFF(row,2,NExt,3,Layout::ColMajor)]=gl64_t(uint64_t(0));"
    }
}

/// Host-side check at the top of every launcher. `exps_min_scratch()` is the
/// contract with the loader: the per-launch table plus (chunked kernels) one
/// wave of cross-chunk temps. Below it the table carve-out in the prologue
/// underflows and the kernels would read/write past the scratch buffer, so
/// fail loudly instead.
fn scratch_guard(sym: &str, table_words: u64, total_slots: u64) -> String {
    let need = if total_slots > 0 {
        format!("{table_words}ull + {total_slots}ull*{GEN_BLK}ull")
    } else {
        format!("{table_words}ull")
    };
    format!(
        r#"  if (scratchElems < {need}) {{
    fprintf(stderr, "[exps] {sym}: scratch too small (%llu < %llu elements)\n",
        (unsigned long long)scratchElems, (unsigned long long)({need}));
    abort();
  }}"#
    )
}

/// The fixed C-ABI the loader dlsym's from each `.exps.so`. `exps_min_scratch`
/// covers one wave of cross-chunk temps plus the per-launch table (powers +
/// hoisted invariants).
fn c_abi_exports(sym: &str, n_slots: u64, ir: &Ir) -> String {
    format!(
        r#"extern "C" void exps_launch(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
    launch_gen_{sym}(d_params, q, scratch, scratchElems, NExt, off_cm1, off_cm2, off_cm3, off_zi, stream);
}}
extern "C" unsigned long long exps_min_scratch() {{ return {n_slots} * {GEN_BLK}ull + {}ull; }}"#,
        ir.table_words()
    )
}

/// Straight-line body of the hoisted-invariants program for thread 0 of the
/// table kernel: the ops of `ir.tab` followed by the exports into `pw[]` after
/// the powers. Reuses the chunk emitter, so operand loads and field ops are
/// the exact same code the per-row kernels would have run.
fn tab_body(ir: &Ir) -> String {
    let mut lines: Vec<String> = Vec::new();
    let declared: HashSet<u64> = HashSet::new();
    for instr in &ir.tab {
        let (l, _) = emit_op(instr, ir, &declared);
        lines.extend(l.into_iter().map(|x| format!("  {x}")));
    }
    let base = ir.pow_words();
    for &(tmp, idx, dim) in &ir.tab_out {
        let w = base + idx;
        if dim == 1 {
            lines.push(format!("    pw[{w}] = t{tmp};"));
        } else {
            lines.push(format!("    pw[{w}] = t{tmp}.a; pw[{}] = t{tmp}.b; pw[{}] = t{tmp}.c;", w + 1, w + 2));
        }
    }
    lines.join("\n")
}

/// Kernel filling the challenge-powers table `pw[3*j..3*j+3] = ch[base..]^j`
/// for j in 0..n, plus the launcher prologue that carves the table off the
/// head of scratch and runs it. No kernel (and a prologue that just aliases
/// `pw` to scratch) when the IR has neither powers nor a table.
/// One block: thread t starts at `v^t` and strides by `v^blockDim`.
fn pow_kernel(sym: &str, ir: &Ir) -> (String, String) {
    if ir.pow.is_empty() && ir.tab.is_empty() {
        return (String::new(), "  const gl64_t* pw = scratch;".to_string());
    }
    // One fill loop per challenge region, laid out end to end in `pow_offset` order.
    let regions: String = ir
        .pow
        .iter()
        .map(|&(base, n)| {
            let off = ir.pow_offset(base);
            format!(
                r#"  {{
    g3 v; v.a=ch[{base}]; v.b=ch[{}]; v.c=ch[{}];
    g3 cur=one, b=v; uint64_t e=threadIdx.x;
    while (e) {{ if (e&1) cur=cg_mul33(cur,b); b=cg_mul33(b,b); e>>=1; }}
    g3 step=one; b=v; e=blockDim.x;
    while (e) {{ if (e&1) step=cg_mul33(step,b); b=cg_mul33(b,b); e>>=1; }}
    for (uint64_t k=threadIdx.x; k<{n}ull; k+=blockDim.x) {{
      pw[{off}+3*k]=cur.a; pw[{off}+3*k+1]=cur.b; pw[{off}+3*k+2]=cur.c; cur=cg_mul33(cur,step);
    }}
  }}"#,
                base + 1,
                base + 2
            )
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        );
    let tab = tab_body(ir);
    let kernel = format!(
        r#"__global__ void gen_{sym}_pow(const StepsParams* __restrict__ P, gl64_t* __restrict__ pw) {{
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; [[maybe_unused]] const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  g3 one; one.a=gl64_t(uint64_t(1)); one.b=gl64_t(uint64_t(0)); one.c=gl64_t(uint64_t(0));
{regions}
  __syncthreads();
  // row-invariant program: once per launch, after the powers it may read
  if (threadIdx.x == 0) {{
{tab}
  }}
}}"#
    );
    let prologue = format!(
        r#"  const gl64_t* pw = scratch; scratch += {pe}ull; scratchElems -= {pe}ull;
  gen_{sym}_pow<<<1,256,0,stream>>>(d_params, (gl64_t*)pw);"#,
        pe = ir.table_words()
    );
    (kernel, prologue)
}

/// Small-expression path: kernel + launcher + C-ABI exports in ONE self-contained TU.
fn single_kernel_tu(sym: &str, kernel: &str, launcher_body: &str, ir: &Ir) -> String {
    let (pk, prologue) = pow_kernel(sym, ir);
    format!(
        r#"// AUTO-GENERATED Q kernel for {sym} (single kernel, no scratch)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{pk}
{kernel}
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
{guard}
{prologue}
{launcher_body}
}}
#undef OFF
{}
"#,
        c_abi_exports(sym, 0, ir),
        guard = scratch_guard(sym, ir.table_words(), 0),
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
    gl64_t* q, gl64_t* scratch, const gl64_t* pw, uint64_t NExt, uint64_t base,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  gen_{sym}_c{i}<<<grid,blk,0,stream>>>(d_params,q,scratch,pw,NExt,base,off_cm1,off_cm2,off_cm3,off_zi);
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

/// The Q launcher TU (`exps_launch`): cross-TU `run_*` decls + the adaptive-grid wave loop + C-ABI.
fn launcher_tu(sym: &str, n_chunks: usize, total_slots: u64, ir: &Ir) -> String {
    let decls: Vec<String> = (0..n_chunks)
        .map(|i| {
            format!(
                "extern \"C\" void run_{sym}_c{i}(uint64_t, uint64_t, cudaStream_t, StepsParams*, gl64_t*, gl64_t*, const gl64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);"
            )
        })
        .collect();
    let calls: Vec<String> = (0..n_chunks)
        .map(|i| {
            format!("    run_{sym}_c{i}(grid, BLK, stream, d_params, q, scratch, pw, NExt, base, off_cm1, off_cm2, off_cm3, off_zi);")
        })
        .collect();
    let (pk, prologue) = pow_kernel(sym, ir);
    format!(
        r#"// AUTO-GENERATED Q launcher for {sym} (cross-boundary temps={total_slots}, {n_chunks} chunks)
#include "gen_common.cuh"
{}
{pk}
// adaptive grid: shrink so total_slots*grid*BLK <= scratchElems (per-wave scratch fits the tmp region);
// each chunk kernel computes WAVE=gridDim*blockDim at runtime, so any grid is correct.
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
  const uint64_t BLK = {GEN_BLK}ull;
{guard}
{prologue}
  uint64_t grid = {total_slots}ull ? (scratchElems / ({total_slots}ull*BLK)) : 512ull;
  if (grid > 512ull) grid = 512ull;
  const uint64_t WAVE = grid * BLK;
  for (uint64_t base=0; base<NExt; base+=WAVE) {{
{}
  }}
}}
{}
"#,
        decls.join("\n"),
        calls.join("\n"),
        c_abi_exports(sym, total_slots, ir),
        guard = scratch_guard(sym, ir.table_words(), total_slots),
    )
}

/// Emit the per-AIR TU source files. Returns `(filename, contents)` pairs:
/// one self-contained TU for a single kernel, or a Q launcher TU + N chunk TUs
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
            r#"__global__ void gen_{sym}_kernel(const StepsParams* __restrict__ P, gl64_t* __restrict__ q, [[maybe_unused]] const gl64_t* __restrict__ pw,
    uint64_t NExt, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ ccf=(const gl64_t*)P->pCustomCommitsFixed;
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<NExt; row+=gridDim.x*blockDim.x) {{
{}
{}
  }}
}}"#,
            body.join("\n"),
            store_qq(plan.out_dim)
        );
        let launcher_body = format!(
            "  (void)scratchElems; gen_{sym}_kernel<<<512,256,0,stream>>>(d_params,q,pw,NExt,off_cm1,off_cm2,off_cm3,off_zi);"
        );
        return vec![(format!("gen_{sym}.cu"), single_kernel_tu(sym, &kernel, &launcher_body, ir))];
    }

    // chunked (tiled): one register-bounded kernel per chunk.
    let mut kernels: Vec<String> = Vec::with_capacity(plan.n_chunks);
    for chunk_idx in 0..plan.n_chunks {
        let (lo_op, hi_op) = plan.range(chunk_idx);
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
    [[maybe_unused]] const gl64_t* __restrict__ pw,
    uint64_t NExt, uint64_t tileBase, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1; const uint64_t WAVE = (uint64_t)gridDim.x*blockDim.x;
  const uint64_t lo_ = blockIdx.x*blockDim.x + threadIdx.x; const uint64_t row = tileBase + lo_;
  if (row >= NExt) return;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ ccf=(const gl64_t*)P->pCustomCommitsFixed;
{}
}}"#,
            lines.join("\n"),
        ));
    }

    let mut files: Vec<(String, String)> =
        vec![(format!("gen_{sym}.cu"), launcher_tu(sym, plan.n_chunks, plan.total_slots, ir))];
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

// ---------------------------------------------------------------------------
// Generic (non-Q) expression kernels: one small straight-line kernel per
// covered expression id, evaluated over the TRACE domain, plus the C-ABI
// per-expId dispatch (`exps_expr_covered` / `exps_launch_expr`) the loader
// dlsym's. Store semantics are mode-parameterized (write / mul-inverse /
// scalar-times-inverse) so two-parameter hint dests (num x den^-1) fuse the
// combine into the second kernel's store — no scratch, no pair kernels.
// ---------------------------------------------------------------------------

/// Numerator of an `im_col` hint: an expression (by id, present in `items`)
/// or a literal.
#[derive(Clone, Copy, Debug)]
pub enum ImNum {
    Expr(i64),
    Scalar(u64),
    /// A base-field committed column read directly: `cm` id (for the prover's
    /// hint match), its stage, position and the stage's column count.
    Col {
        cm_id: u64,
        stage: u64,
        pos: u64,
        stage_cols: u64,
    },
}

/// One `im_col` hint the batched kernels cover: `cm[cm_id] = num / den` per
/// row, the column living in stage `stage` at `stage_pos` (cubic).
#[derive(Clone, Debug)]
pub struct ImColHint {
    pub cm_id: u64,
    pub stage: u64,
    pub stage_pos: u64,
    pub stage_cols: u64,
    pub num: ImNum,
    pub den: i64,
}

/// A batch of im_col hints as emitted: the hints, and optionally one merged IR
/// that computes every numerator/denominator with shared subtrees evaluated
/// once (`roots[i]` = tmp of hint i's denominator, `num_roots[i]` = tmp of its
/// expression numerator, when it has one). Without `merged` each expression is
/// evaluated by its own lambda.
#[derive(Clone, Debug)]
pub struct ImColBatch {
    pub hints: Vec<ImColHint>,
    pub merged: Option<MergedBatch>,
}

#[derive(Clone, Debug)]
pub struct MergedBatch {
    pub ir: crate::ir::Ir,
    pub den_roots: Vec<u64>,
    pub num_roots: Vec<Option<u64>>,
}

/// Register-resident challenge powers `pwreg_<base>_<j>` for every (base, n)
/// in `pows` (deduped by base, max n): repeated `cg_mul33` from the challenge,
/// the same arithmetic as the Q table, so folded chains stay exact.
fn pwreg_lines(pows: &[(u64, u64)]) -> String {
    let mut need: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for &(base, n) in pows {
        let e = need.entry(base).or_insert(0);
        *e = (*e).max(n);
    }
    let mut lines: Vec<String> = Vec::new();
    for (&base, &n) in &need {
        if n < 2 {
            continue;
        }
        lines.push(format!(
            "  [[maybe_unused]] g3 pwreg_{base}_1; pwreg_{base}_1.a=ch[{base}]; pwreg_{base}_1.b=ch[{}]; pwreg_{base}_1.c=ch[{}];",
            base + 1,
            base + 2
        ));
        for j in 2..n {
            lines.push(format!(
                "  [[maybe_unused]] g3 pwreg_{base}_{j} = cg_mul33(pwreg_{base}_{}, pwreg_{base}_1);",
                j - 1
            ));
        }
    }
    if lines.is_empty() {
        String::new()
    } else {
        lines.join("\n") + "\n"
    }
}

/// Common kernel prologue: the trace / constant / challenge pointers every
/// expression body reads.
const EXPR_PROLOGUE: &str = r#"  const uint64_t NExt = N; const uint64_t MASK = N-1;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ ccf=(const gl64_t*)P->pCustomCommitsFixed;"#;

/// `auto <name> = [&](uint64_t row) -> <g3|gl64_t> { ... }` evaluating `ir` at
/// one row. Temps are local to the lambda, so several can share a kernel.
fn expr_lambda(name: &str, ir: &crate::ir::Ir, out_dim: u64) -> String {
    let mut body: Vec<String> = Vec::new();
    let mut declared: HashSet<u64> = HashSet::new();
    for instr in &ir.instrs {
        let (op_lines, is_out) = emit_op(instr, ir, &declared);
        body.extend(op_lines);
        if !is_out {
            declared.insert(instr.dst_id.unwrap());
        }
    }
    // The result lives in the last instruction's destination: `qq` for an
    // explicit output store, `t<id>` when the expression ends in a tmp.
    let last = ir.instrs.last().unwrap();
    let result = if last.dst_is_tmp { format!("t{}", last.dst_id.unwrap()) } else { "qq".to_string() };
    let vt = if out_dim == 3 { "g3" } else { "gl64_t" };
    format!("  auto {name} = [&](uint64_t row) -> {vt} {{\n{}\n    return {result};\n  }};", body.join("\n"))
}

/// Emit the `gen_<sym>_cexprs.cu` TU covering `items` = (expId, ir, out_dim),
/// plus one fused kernel per `im_col` batch (see `imcol_kernel`).
pub fn emit_exprs_tu(sym: &str, items: &[(i64, crate::ir::Ir, u64)], batches: &[ImColBatch]) -> String {
    let mut kernels: Vec<String> = Vec::new();
    let mut cases_launch: Vec<String> = Vec::new();
    let mut cases_covered: Vec<String> = Vec::new();
    for (exp_id, ir, out_dim) in items {
        // Value type of this expression and how the plain store receives it.
        let (vt, vdim) = if *out_dim == 3 { ("g3", 3) } else { ("gl64_t", 1) };
        let to_g3 = if *out_dim == 3 {
            "g3 v_=ev_;".to_string()
        } else {
            "g3 v_; v_.a=ev_; v_.b=gl64_t(uint64_t(0)); v_.c=gl64_t(uint64_t(0));".to_string()
        };
        let store = format!("    {{ {to_g3} exps_expr_store(dest,row,destDomain,stagePos,stageCols,destDim,destExpr,mode,scalar,v_,{vdim}); }}");
        // Challenge powers for this kernel, computed once per thread into registers
        // (the kernel has no scratch table; a thread serves many rows, so the few
        // cubic multiplies amortize).
        let pwregs = if ir.pow_in_regs { pwreg_lines(&ir.pow) } else { String::new() };
        let lambda = expr_lambda("eval_", ir, *out_dim);
        let sig = "(const StepsParams* __restrict__ P, gl64_t* __restrict__ dest,\n    uint64_t N, uint64_t destDomain, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3,\n    uint64_t mode, uint64_t scalar, uint64_t stagePos, uint64_t stageCols, uint64_t destDim, uint32_t destExpr)";
        kernels.push(format!(
            r#"__global__ void gen_{sym}_x{exp_id}{sig} {{
{EXPR_PROLOGUE}
{pwregs}{lambda}
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<N; row+=(uint64_t)gridDim.x*blockDim.x) {{
    {vt} ev_ = eval_(row);
{store}
  }}
}}"#
        ));
        cases_launch.push(format!("    case {exp_id}ull: gen_{sym}_x{exp_id}<<<grid,{GEN_BLK},0,stream>>>(P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,mode,scalar,stagePos,stageCols,destDim,destExpr); return 1;"));
        cases_covered.push(format!("    case {exp_id}ull: return 1;"));
    }
    let by_id: HashMap<i64, (&crate::ir::Ir, u64)> = items.iter().map(|(id, ir, od)| (*id, (ir, *od))).collect();
    let mut imb_launch: Vec<String> = Vec::new();
    let mut imb_info: Vec<String> = Vec::new();
    for (k, batch) in batches.iter().enumerate() {
        kernels.push(imcol_kernel(sym, k, batch, &by_id));
        imb_launch.push(format!(
            "    case {k}ull: gen_{sym}_imb{k}<<<grid,{GEN_BLK},0,stream>>>(P,N,off_cm1,off_cm2,off_cm3); return 1;"
        ));
        let mut words: Vec<String> = Vec::new();
        for h in &batch.hints {
            // numerator word: expId; ~0 = literal (value in the next word); ~0-1 = raw
            // column (its cm id in the next word)
            let (nid, sc) = match h.num {
                ImNum::Expr(id) => (id as u64, 0u64),
                ImNum::Scalar(v) => (u64::MAX, v),
                ImNum::Col { cm_id, .. } => (u64::MAX - 1, cm_id),
            };
            words.push(format!("{nid}ull,{sc}ull,{}ull,{}ull", h.den, h.cm_id));
        }
        imb_info.push(format!(
            "    case {k}ull: {{ static const unsigned long long w[] = {{{}}}; n = {}; src = w; break; }}",
            words.join(","),
            batch.hints.len()
        ));
    }
    format!(
        r#"// AUTO-GENERATED generic expression kernels for {sym} ({} expressions, {} im_col batches)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{}
extern "C" int exps_expr_covered(unsigned long long expId) {{
  switch (expId) {{
{}
    default: return 0;
  }}
}}
extern "C" int exps_launch_expr(unsigned long long expId, StepsParams* P, gl64_t* dest,
    unsigned long long N, unsigned long long destDomain,
    unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3,
    unsigned long long mode, unsigned long long scalar,
    unsigned long long stagePos, unsigned long long stageCols,
    unsigned long long destDim, unsigned int destExpr, cudaStream_t stream) {{
  uint64_t grid = (N + {GEN_BLK}ull - 1) / {GEN_BLK}ull;
  if (grid > 512ull) grid = 512ull;
  if (grid < 1ull) grid = 1ull;
  switch (expId) {{
{}
    default: return 0;
  }}
}}
// im_col batches: `exps_imcol_batches()` fused kernels, each covering the hints
// `exps_imcol_batch_info(k, out, cap)` lists as 4 words per hint
// (numerator expId, or ~0 for a literal / ~0-1 for a raw column; literal value
// or column cm id; denominator expId; destination cm id). The prover launches a batch only when every hint it
// lists is an im_col hint of the instance.
extern "C" unsigned long long exps_imcol_batches() {{ return {}ull; }}
extern "C" unsigned long long exps_imcol_batch_info(unsigned long long k, unsigned long long* out, unsigned long long cap) {{
  const unsigned long long* src = nullptr; unsigned long long n = 0;
  switch (k) {{
{}
    default: return 0;
  }}
  for (unsigned long long i = 0; i < 4ull*n && i < cap; ++i) out[i] = src[i];
  return n;
}}
extern "C" int exps_launch_imcol_batch(unsigned long long k, StepsParams* P, unsigned long long N,
    unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3, cudaStream_t stream) {{
  uint64_t grid = (N + {GEN_BLK}ull - 1) / {GEN_BLK}ull;
  if (grid > 512ull) grid = 512ull;
  if (grid < 1ull) grid = 1ull;
  switch (k) {{
{}
    default: return 0;
  }}
}}
#undef OFF
"#,
        items.len(),
        batches.len(),
        kernels.join("\n"),
        cases_covered.join("\n"),
        cases_launch.join("\n"),
        batches.len(),
        imb_info.join("\n"),
        imb_launch.join("\n")
    )
}

/// One fused kernel for a batch of `im_col` hints: per row every numerator and
/// denominator is evaluated (the column loads and challenge powers are shared
/// across the hints), the denominators are inverted with one cubic inverse via
/// Montgomery's trick, and each column gets `num * den^-1`. A zero denominator
/// in the row falls back to per-hint inverses, so the result matches the
/// unbatched path exactly (including 0^-1 = 0).
fn imcol_kernel(sym: &str, k: usize, batch: &ImColBatch, by_id: &HashMap<i64, (&crate::ir::Ir, u64)>) -> String {
    let m = batch.hints.len();
    let mut pows: Vec<(u64, u64)> = Vec::new();
    let mut lambdas: Vec<String> = Vec::new();
    let mut evals: Vec<String> = Vec::new();
    let mut stores: Vec<String> = Vec::new();
    // Merged form: one straight-line body evaluating every expression of the batch
    // once (shared subtrees deduplicated); numerators/denominators are its tmps.
    let merged_body: Option<String> = batch.merged.as_ref().map(|mb| {
        pows.extend(mb.ir.pow.iter().copied());
        let mut body: Vec<String> = Vec::new();
        let mut declared: HashSet<u64> = HashSet::new();
        for instr in &mb.ir.instrs {
            let (op_lines, is_out) = emit_op(instr, &mb.ir, &declared);
            body.extend(op_lines);
            if !is_out {
                declared.insert(instr.dst_id.unwrap());
            }
        }
        body.join("\n")
    });
    for (i, h) in batch.hints.iter().enumerate() {
        let num_val: String;
        if let Some(mb) = &batch.merged {
            evals.push(format!("      d_[{i}] = t{}; z_ |= g3_is_zero(d_[{i}]);", mb.den_roots[i]));
            num_val = mb.num_roots[i].map(|r| format!("t{r}")).unwrap_or_default();
        } else {
            let &(dir, _) = &by_id[&h.den];
            pows.extend(dir.pow.iter().copied());
            lambdas.push(expr_lambda(&format!("den{i}_"), dir, 3));
            evals.push(format!("      d_[{i}] = den{i}_(row); z_ |= g3_is_zero(d_[{i}]);"));
            num_val = format!("num{i}_(row)");
        }
        let prod = match h.num {
            ImNum::Expr(id) => {
                let &(nir, nod) = &by_id[&id];
                if batch.merged.is_none() {
                    pows.extend(nir.pow.iter().copied());
                    lambdas.push(expr_lambda(&format!("num{i}_"), nir, nod));
                }
                if nod == 3 {
                    format!("cg_mul33({num_val}, inv_[{i}])")
                } else {
                    format!("cg_mul31(inv_[{i}], {num_val})")
                }
            }
            ImNum::Scalar(v) => format!("cg_mul31(inv_[{i}], gl64_t(uint64_t({v}ull)))"),
            ImNum::Col { stage, pos, stage_cols, .. } => format!(
                "cg_mul31(inv_[{i}], aux[off_cm{stage} + OFF(row,{pos},N,{stage_cols},{})])",
                cm_layout(0, stage_cols)
            ),
        };
        stores.push(format!(
            "      {{ g3 r_ = {prod}; gl64_t* o_ = auxw + off_cm{}; o_[OFF(row,{p},N,{c},{lyt})]=r_.a; o_[OFF(row,{p}+1,N,{c},{lyt})]=r_.b; o_[OFF(row,{p}+2,N,{c},{lyt})]=r_.c; }}",
            h.stage,
            p = h.stage_pos,
            c = h.stage_cols,
            lyt = cm_layout(0, h.stage_cols)
        ));
    }
    let pwregs = pwreg_lines(&pows);
    format!(
        r#"__global__ void gen_{sym}_imb{k}(const StepsParams* __restrict__ P, uint64_t N, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3) {{
{EXPR_PROLOGUE}
  gl64_t* auxw = (gl64_t*)P->aux_trace;
{pwregs}{}
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<N; row+=(uint64_t)gridDim.x*blockDim.x) {{
    // inv_ doubles as the prefix-product array: p_[i] is dead once inv_[i] is
    // formed (the backward pass only reads p_[i-1]), so both live in inv_.
    g3 d_[{m}]; g3 inv_[{m}]; bool z_ = false;
{merged}
{}
    if (z_) {{
      #pragma unroll
      for (int i_ = 0; i_ < {m}; ++i_) inv_[i_] = cg_inv3(d_[i_]);
    }} else {{
      inv_[0] = d_[0];
      #pragma unroll
      for (int i_ = 1; i_ < {m}; ++i_) inv_[i_] = cg_mul33(inv_[i_-1], d_[i_]);
      g3 t_ = cg_inv3(inv_[{m}-1]);
      #pragma unroll
      for (int i_ = {m}-1; i_ > 0; --i_) {{ g3 v_ = cg_mul33(t_, inv_[i_-1]); t_ = cg_mul33(t_, d_[i_]); inv_[i_] = v_; }}
      inv_[0] = t_;
    }}
{}
  }}
}}"#,
        lambdas.join("\n"),
        evals.join("\n"),
        stores.join("\n"),
        merged = merged_body.unwrap_or_default()
    )
}
