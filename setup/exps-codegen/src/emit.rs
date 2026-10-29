//! CUDA source emission: the straight-line / chunked kernel text for one AIR's
//! Q expression, one trace-domain kernel per other covered expression
//! (hint fields, im columns, ... -- `emit_exprs_tu`, dispatched by expId via
//! `exps_generic_covered` / `exps_generic_launch`), the shared `gen_common.cuh`
//! header, and the fixed C-ABI
//! exports. The template whitespace is deliberate — the emitted `.cu` is what
//! nvcc compiles, so treat these strings as code, not free-form text.

use crate::ir::{plan_chunks_ops, ChunkPlan, Instr, Ir, Operand};
use std::collections::{BTreeMap, HashMap, HashSet};

/// CUDA threads/block of the generated kernels (single source of truth, baked into each launcher).
pub const GEN_BLK: u64 = 256;
/// Widest cross-chunk cut a chunked kernel may have, in scratch words per row: wider cuts cost
/// more scratch traffic than they save, and the launcher's one wave would outgrow the scratch.
pub const SLOTS_CAP: u64 = 1000;
/// Grid cap for the trace-domain expression launchers (singles AND pairs — same work class, same
/// prologue-vs-rows tradeoff, one policy). Empirical, not a hardware bound: challenge powers are
/// rebuilt per THREAD, so a bigger grid trades rows per thread against redundant prologue work.
/// Per-air tuning belongs in autotune.rs.
pub const EXPR_GRID_CAP: u64 = 2048;
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
__device__ __forceinline__ void exps_expr_store_pair(gl64_t* dest, uint64_t row, uint64_t destDomain,
    uint64_t stagePos, uint64_t stageCols, uint64_t destDim, uint32_t destExpr,
    g3 num, uint64_t ndim, g3 den, uint64_t ddim)
{
  uint64_t idx0, idx1 = 0, idx2 = 0;
  if (destExpr) { idx0=row*destDim; idx1=idx0+1; idx2=idx0+2; }
  else {
    const Layout lyt=resolveLayout(63-__clzll(destDomain),stageCols);
    idx0=getBufferOffset(row,stagePos,destDomain,stageCols,lyt);
    if (destDim>1) { idx1=getBufferOffset(row,stagePos+1,destDomain,stageCols,lyt); idx2=getBufferOffset(row,stagePos+2,destDomain,stageCols,lyt); }
  }
  if (ndim==1) { num.b=gl64_t(uint64_t(0)); num.c=gl64_t(uint64_t(0)); }
  g3 inv;
  if (ddim==1) { inv.a=den.a.reciprocal(); inv.b=gl64_t(uint64_t(0)); inv.c=gl64_t(uint64_t(0)); }
  else inv=cg_inv3(den);
  if (destDim==1) { dest[idx0]=num.a*inv.a; return; }
  g3 r=(ddim==1)?cg_mul31(num,inv.a):cg_mul33(num,inv);
  dest[idx0]=r.a; dest[idx1]=r.b; dest[idx2]=r.c;
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

/// The operand pointers every generated kernel opens with; `cst` is the const section the kernel
/// reads: the extended one (Q, over the extended domain) or the small one (trace-domain kernels).
fn operand_ptrs(cst_field: &str) -> String {
    format!(
        "  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->{cst_field};
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ ccf=(const gl64_t*)P->pCustomCommitsFixed;"
    )
}
/// Prologue of a trace-domain kernel: `NExt`/`MASK` over the trace rows, then the pointers.
fn trace_prologue() -> String {
    format!("  const uint64_t NExt = N; const uint64_t MASK = N-1;\n{}", operand_ptrs("pConstPolsAddress"))
}

fn rowexpr(stride: i64) -> String {
    if stride == 0 {
        "row".to_string()
    } else {
        format!("((row+({stride}ll))&MASK)")
    }
}

/// Boundary `b`'s zerofier at this row: one period per boundary, at its offset in the zi region.
fn zi_load(name: &str, b: u64, ir: &Ir) -> Vec<String> {
    let (off, mask) = ir.zi.tables[b as usize];
    vec![format!(
        "  gl64_t {name}; {{ constexpr uint64_t zo{b} = {off}ull, zm{b} = {mask}ull; {name} = aux[off_zi + zo{b} + (row & zm{b})]; }}"
    )]
}

/// Lines that materialize a non-tmp operand into the local `name`.
fn load_lines(opnd: &Operand, name: &str, ir: &Ir) -> Vec<String> {
    match opnd {
        Operand::Num(v) => vec![format!("  gl64_t {name}(uint64_t({v}ull));")],
        Operand::Zi => zi_load(name, 0, ir),
        Operand::Zb { boundary } => zi_load(name, *boundary, ir),
        Operand::Pub { id } => vec![format!("  gl64_t {name} = pub[{id}];")],
        Operand::Ch { base, .. } => {
            let i = *base;
            vec![format!("  g3 {name}; {name}.a=ch[{i}]; {name}.b=ch[{}]; {name}.c=ch[{}];", i + 1, i + 2)]
        }
        Operand::Pow { base, j, .. } if ir.pow_in_regs => {
            vec![format!("  g3 {name} = pwreg_{base}_{j};")]
        }
        Operand::Pow { base, j, .. } => {
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
    // NOTE: do NOT emit `x * uint32_t(c)` for small constants. sppark's
    // gl64_t::mul(uint32_t) PTX is miscompiled by ptxas -O3 (CUDA 13.0, sm_120):
    // in large fused kernels its 32-bit carry-chain rewrite corrupts the result
    // for some inputs (repro: scratchpad micro.cu, correct at -Xptxas -O0). The
    // full 64-bit gl64_t multiply below is unaffected.
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
        "    q[OFF(jq,0,NQ,3,Layout::ColMajor)]=qq.a; q[OFF(jq,1,NQ,3,Layout::ColMajor)]=qq.b; q[OFF(jq,2,NQ,3,Layout::ColMajor)]=qq.c;"
    } else {
        "    q[OFF(jq,0,NQ,3,Layout::ColMajor)]=qq; q[OFF(jq,1,NQ,3,Layout::ColMajor)]=gl64_t(uint64_t(0)); q[OFF(jq,2,NQ,3,Layout::ColMajor)]=gl64_t(uint64_t(0));"
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
    launch_gen_{sym}(d_params, q, scratch, scratchElems, NExt, off_cm1, off_cm2, off_cm3, off_zi, 0u, stream);
}}
// Q at every 2^qs-th row of the extended domain only, written compactly (NExt >> qs rows): enough
// whenever qDeg * N <= NExt >> qs, and the caller's iNTT must then run over NExt >> qs.
extern "C" void exps_launch_step(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs, cudaStream_t stream) {{
    launch_gen_{sym}(d_params, q, scratch, scratchElems, NExt, off_cm1, off_cm2, off_cm3, off_zi, qs, stream);
}}
extern "C" unsigned long long exps_min_scratch() {{ return {}ull; }}
// The boundaries the zi reads above were generated for; the loader refuses a library whose hash differs.
extern "C" uint64_t boundary_layout_hash() {{ return {}ull; }}"#,
        q_min_scratch(ir, sym, n_slots),
        ir.zi.hash
    )
}

/// Ops per slice of the table program. cicc's cost is superlinear in the size of one straight-line
/// function: Main b22/e10704's 4206 table ops in one function took 166 s of cicc (+76 s ptxas)
/// while a chunk TU of 8 kernels x ~500 ops takes 5.6 s. So the program is cut the way the Q program
/// is (`plan_chunks_ops` on the tab ops): one kernel per slice, temps crossing a cut parked in table
/// words after the exports, and each slice in its own TU so they compile in parallel.
const TAB_SLICE: usize = 512;

/// Slice plan of the table program; `None` when it fits one slice.
fn tab_plan(ir: &Ir, sym: &str) -> Option<ChunkPlan> {
    if ir.tab.len() <= TAB_SLICE {
        return None;
    }
    match plan_chunks_ops(&ir.tab, TAB_SLICE, &format!("{sym} table program")) {
        Ok(p) if p.n_chunks > 1 => Some(p),
        Ok(_) => None,
        Err(e) => {
            eprintln!("[exps-codegen] {sym}: table program not sliced ({e}); compiling it whole");
            None
        }
    }
}

/// Words carved off the head of scratch per launch: the powers, the exports, and the table
/// program's cross-slice temps.
pub fn pow_region_words(ir: &Ir, sym: &str) -> u64 {
    ir.table_words() + tab_plan(ir, sym).map_or(0, |p| p.total_slots)
}

/// Straight-line body of one slice of the hoisted-invariants program (thread 0 of a table
/// kernel): loads of the temps entering the slice (`live_in`: tmp, dim, table word), the ops with
/// their exports into `pw[]`, then the temps leaving it (`live_out`). Reuses the chunk emitter, so
/// operand loads and field ops are the exact same code the per-row kernels would have run.
fn tab_body(ir: &Ir, ops: &[Instr], live_in: &[(u64, u64, u64)], live_out: &[(u64, u64, u64)], last: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    let base = ir.pow_words();
    let mut declared: HashSet<u64> = HashSet::new();
    for &(t, dim, w) in live_in {
        if dim == 1 {
            lines.push(format!("    gl64_t t{t} = pw[{w}];"));
        } else {
            lines.push(format!("    g3 t{t}; t{t}.a = pw[{w}]; t{t}.b = pw[{}]; t{t}.c = pw[{}];", w + 1, w + 2));
        }
        declared.insert(t);
    }
    // Exported temps are stored the moment they are produced, not in a block at the end: with
    // thousands of tab ops on one thread, deferring every export kept every exported value live
    // to the end of the program and ptxas spilled them (Main b22/e10704: 16 KB of stack per
    // thread, which the driver reserves for every thread the device can hold -- ~5.6 GB on an
    // RTX 5090 -- so the first real launch failed for want of VRAM and Q fell to the interpreter).
    let mut exports: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
    for &(tmp, idx, dim) in &ir.tab_out {
        exports.entry(tmp).or_default().push((base + idx, dim));
    }
    let store = |tmp: u64, w: u64, dim: u64| -> String {
        if dim == 1 {
            format!("    pw[{w}] = t{tmp};")
        } else {
            format!("    pw[{w}] = t{tmp}.a; pw[{}] = t{tmp}.b; pw[{}] = t{tmp}.c;", w + 1, w + 2)
        }
    };
    for instr in ops {
        let (l, _) = emit_op(instr, ir, &declared);
        lines.extend(l.into_iter().map(|x| format!("  {x}")));
        if let Some(tmp) = instr.dst_id.filter(|_| instr.dst_is_tmp) {
            declared.insert(tmp);
            if let Some(outs) = exports.get(&tmp) {
                for &(w, dim) in outs {
                    lines.push(store(tmp, w, dim));
                }
            }
        }
    }
    if last {
        // Exports whose temp is not produced by any tab op (defensive: keeps the old behaviour for them).
        let produced: HashSet<u64> = ir.tab.iter().filter_map(|i| i.dst_id.filter(|_| i.dst_is_tmp)).collect();
        for &(tmp, idx, dim) in &ir.tab_out {
            if !produced.contains(&tmp) {
                lines.push(store(tmp, base + idx, dim));
            }
        }
    }
    for &(t, dim, w) in live_out {
        lines.push(store(t, w, dim));
    }
    lines.join("\n")
}

/// The per-launch table TUs: `gen_<sym>_pow.cu` holds the kernel filling the challenge-powers
/// table `pw[3*j..3*j+3] = ch[base..]^j` for j in 0..n (one block: thread t starts at `v^t` and
/// strides by `v^blockDim`) followed by the first slice of the hoisted-invariants program on
/// thread 0, plus the C-ABI host wrapper `run_<sym>_pow` the launcher calls, which also runs the
/// remaining slices (`gen_<sym>_pow_s<i>.cu`, one thread each) in order. Empty when the IR has
/// neither powers nor a table.
///
/// Their own TUs, compiled once per AIR, because the text does not depend on the chunk size while
/// for a hoisting-heavy AIR it is by far the slowest to compile (Main b22/e10704: 245 s in one
/// function, against ~11 s for a chunk TU). Inside the launcher TU it was rebuilt at every
/// autotune probe and serialized the whole run.
pub fn emit_pow_tu(sym: &str, ir: &Ir) -> Vec<(String, String)> {
    if ir.pow.is_empty() && ir.tab.is_empty() {
        return Vec::new();
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
        .join("\n");
    let ptrs = r#"  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  [[maybe_unused]] const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; [[maybe_unused]] const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;"#;

    // Slice bodies: one (the whole program) without a plan, else one per plan chunk with the
    // cross-slice temps parked right after the exports.
    let plan = tab_plan(ir, sym);
    let n_slices = plan.as_ref().map_or(1, |p| p.n_chunks);
    let x0 = ir.table_words();
    let bodies: Vec<String> = (0..n_slices)
        .map(|i| match &plan {
            None => tab_body(ir, &ir.tab, &[], &[], true),
            Some(p) => {
                let (lo, hi) = p.range(i);
                let ops = &ir.tab[lo..hi];
                let used: HashSet<u64> =
                    ops.iter().flat_map(|ins| [&ins.a, &ins.b]).filter_map(|o| o.as_tmp().map(|(t, _)| t)).collect();
                let word = |t: u64| (t, p.dim_of[&t], x0 + p.slot_index(t));
                let mut live_in: Vec<(u64, u64, u64)> = used
                    .iter()
                    .copied()
                    .filter(|t| p.cut_temps.contains(t) && p.chunk_of(p.def_idx[t]) < i)
                    .map(word)
                    .collect();
                live_in.sort_unstable();
                let mut live_out: Vec<(u64, u64, u64)> =
                    p.cut_temps.iter().copied().filter(|t| p.chunk_of(p.def_idx[t]) == i).map(word).collect();
                live_out.sort_unstable();
                tab_body(ir, ops, &live_in, &live_out, i + 1 == n_slices)
            }
        })
        .collect();

    let head = |what: &str| {
        format!(
            r#"// AUTO-GENERATED {what} for {sym}
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))"#
        )
    };
    let mut files: Vec<(String, String)> = Vec::with_capacity(n_slices);
    let decls: String = (1..n_slices)
        .map(|i| {
            format!("extern \"C\" void run_{sym}_pow_s{i}(cudaStream_t stream, StepsParams* d_params, gl64_t* pw);\n")
        })
        .collect();
    let calls: String = (1..n_slices).map(|i| format!("\n  run_{sym}_pow_s{i}(stream, d_params, pw);")).collect();
    files.push((
        format!("gen_{sym}_pow.cu"),
        format!(
            r#"{}
{decls}__global__ void gen_{sym}_pow(const StepsParams* __restrict__ P, gl64_t* __restrict__ pw) {{
{ptrs}
  g3 one; one.a=gl64_t(uint64_t(1)); one.b=gl64_t(uint64_t(0)); one.c=gl64_t(uint64_t(0));
{regions}
  __syncthreads();
  // row-invariant program (slice 0 of {n_slices}): once per launch, after the powers it may read
  if (threadIdx.x == 0) {{
{}
  }}
}}
extern "C" void run_{sym}_pow(cudaStream_t stream, StepsParams* d_params, gl64_t* pw) {{
  gen_{sym}_pow<<<1,256,0,stream>>>(d_params, pw);{calls}
}}
#undef OFF
"#,
            head("per-launch table kernel"),
            bodies[0],
        ),
    ));
    for (i, body) in bodies.iter().enumerate().skip(1) {
        files.push((
            format!("gen_{sym}_pow_s{i}.cu"),
            format!(
                r#"{}
__global__ void gen_{sym}_pow_s{i}(const StepsParams* __restrict__ P, gl64_t* __restrict__ pw) {{
{ptrs}
  // row-invariant program, slice {i} of {n_slices}: one thread, after slice {}
{body}
}}
extern "C" void run_{sym}_pow_s{i}(cudaStream_t stream, StepsParams* d_params, gl64_t* pw) {{
  gen_{sym}_pow_s{i}<<<1,1,0,stream>>>(d_params, pw);
}}
#undef OFF
"#,
                head(&format!("table program slice {i}")),
                i - 1,
            ),
        ));
    }
    files
}

/// Launcher-side glue for the table kernel: the file-scope extern decl of its host wrapper and
/// the prologue that carves the table region off the head of scratch and runs it -- or, when the
/// IR has no table, an empty decl and a prologue that just aliases `pw` to scratch.
fn pow_glue(sym: &str, ir: &Ir) -> (String, String) {
    if ir.pow.is_empty() && ir.tab.is_empty() {
        return (String::new(), "  const gl64_t* pw = scratch;".to_string());
    }
    let decl = format!("extern \"C\" void run_{sym}_pow(cudaStream_t stream, StepsParams* d_params, gl64_t* pw);");
    let prologue = format!(
        r#"  const gl64_t* pw = scratch; scratch += {pe}ull; scratchElems -= {pe}ull;
  run_{sym}_pow(stream, d_params, (gl64_t*)pw);"#,
        pe = pow_region_words(ir, sym)
    );
    (decl, prologue)
}

/// Small-expression path: kernel + launcher + C-ABI exports in ONE self-contained TU.
fn single_kernel_tu(sym: &str, kernel: &str, launcher_body: &str, ir: &Ir) -> String {
    let (pow_decl, prologue) = pow_glue(sym, ir);
    format!(
        r#"// AUTO-GENERATED Q kernel for {sym} (single kernel, no scratch)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{pow_decl}
{kernel}
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs, cudaStream_t stream) {{
{guard}
{prologue}
{launcher_body}
}}
#undef OFF
{}
"#,
        c_abi_exports(sym, 0, ir),
        guard = scratch_guard(sym, pow_region_words(ir, sym), 0),
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
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs) {{
  gen_{sym}_c{i}<<<grid,blk,0,stream>>>(d_params,q,scratch,pw,NExt,base,off_cm1,off_cm2,off_cm3,off_zi,qs);
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
                "extern \"C\" void run_{sym}_c{i}(uint64_t, uint64_t, cudaStream_t, StepsParams*, gl64_t*, gl64_t*, const gl64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t);"
            )
        })
        .collect();
    let calls: Vec<String> = (0..n_chunks)
        .map(|i| {
            format!("    run_{sym}_c{i}(grid, BLK, stream, d_params, q, scratch, pw, NExt, base, off_cm1, off_cm2, off_cm3, off_zi, qs);")
        })
        .collect();
    let (pow_decl, prologue) = pow_glue(sym, ir);
    format!(
        r#"// AUTO-GENERATED Q launcher for {sym} (cross-boundary temps={total_slots}, {n_chunks} chunks)
#include "gen_common.cuh"
{}
{pow_decl}
// adaptive grid: shrink so total_slots*grid*BLK <= scratchElems (per-wave scratch fits the tmp region);
// each chunk kernel computes WAVE=gridDim*blockDim at runtime, so any grid is correct.
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs, cudaStream_t stream) {{
  const uint64_t BLK = {GEN_BLK}ull;
{guard}
{prologue}
  // Grid sets how many WAVES the domain is split into, and each wave waits for the previous one
  // (shared scratch). Occupancy saturates well below the scratch limit, so take what the scratch
  // allows but never more blocks than there is work for: extra waves only add serialization.
  const uint64_t NQ = NExt >> qs;
{}
  for (uint64_t base=0; base<NQ; base+=WAVE) {{
{}
  }}
}}
{}
"#,
        decls.join("\n"),
        wave_grid(total_slots, "NQ"),
        calls.join("\n"),
        c_abi_exports(sym, total_slots, ir),
        guard = scratch_guard(sym, pow_region_words(ir, sym), total_slots),
    )
}

/// Body lines of chunk `chunk_idx` of a chunked program: loads of the cut temps entering it
/// (`scratch[slot*WAVE + lo_]`), the ops, then the stores of the cut temps it produces. Shared by
/// the Q chunk kernels and the chunked non-Q kernels, so both park temps the same way.
/// `epilogue_uses` are temps the caller's code after the body reads (a pair kernel's final store
/// consumes results produced in earlier chunks): loaded like any other live-in when cut.
fn chunk_body(ir: &Ir, plan: &ChunkPlan, chunk_idx: usize, epilogue_uses: &[u64]) -> Vec<String> {
    let (lo_op, hi_op) = plan.range(chunk_idx);
    let chunk_ops = &ir.instrs[lo_op..hi_op];

    let mut used_temps: HashSet<u64> = epilogue_uses.iter().copied().collect();
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
    lines
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
    uint64_t NExt, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs) {{
  const uint64_t MASK = NExt-1; const uint64_t NQ = NExt >> qs;
{Q_PTRS}
  for (uint64_t jq=blockIdx.x*blockDim.x+threadIdx.x; jq<NQ; jq+=gridDim.x*blockDim.x) {{
  const uint64_t row = jq << qs;
{}
{}
  }}
}}"#,
            body.join("\n"),
            store_qq(plan.out_dim),
            Q_PTRS = operand_ptrs("pConstPolsExtendedTreeAddress")
        );
        let launcher_body = format!(
            "  (void)scratchElems; gen_{sym}_kernel<<<512,256,0,stream>>>(d_params,q,pw,NExt,off_cm1,off_cm2,off_cm3,off_zi,qs);"
        );
        return vec![(format!("gen_{sym}.cu"), single_kernel_tu(sym, &kernel, &launcher_body, ir))];
    }

    // chunked (tiled): one register-bounded kernel per chunk.
    let mut kernels: Vec<String> = Vec::with_capacity(plan.n_chunks);
    for chunk_idx in 0..plan.n_chunks {
        let mut lines = chunk_body(ir, plan, chunk_idx, &[]);
        if chunk_idx == plan.n_chunks - 1 {
            lines.push(store_qq(plan.out_dim).to_string());
        }
        kernels.push(format!(
            r#"__global__ void gen_{sym}_c{chunk_idx}(const StepsParams* __restrict__ P, gl64_t* __restrict__ q, gl64_t* __restrict__ scratch,
    [[maybe_unused]] const gl64_t* __restrict__ pw,
    uint64_t NExt, uint64_t tileBase, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, uint32_t qs) {{
  const uint64_t MASK = NExt-1; const uint64_t WAVE = (uint64_t)gridDim.x*blockDim.x; const uint64_t NQ = NExt >> qs;
  const uint64_t lo_ = blockIdx.x*blockDim.x + threadIdx.x; const uint64_t jq = tileBase + lo_;
  if (jq >= NQ) return;
  const uint64_t row = jq << qs;
{Q_PTRS}
{}
}}"#,
            lines.join("\n"),
            Q_PTRS = operand_ptrs("pConstPolsExtendedTreeAddress")
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
// per-expId dispatch (`exps_generic_covered` / `exps_generic_launch`) the loader
// dlsym's. Store semantics are mode-parameterized (write / mul-inverse /
// scalar-times-inverse) so two-parameter hint dests (num x den^-1) fuse the
// combine into the second kernel's store — no scratch, no pair kernels.
// ---------------------------------------------------------------------------

/// Kernels per generic-expression TU. One TU per AIR made Main's 833 tiny kernels a single 95 s
/// nvcc job on the critical path (after the last autotune); parts compile in parallel during
/// phase 2. Not smaller: every TU pays ~4.4 s of header (64 per TU doubled the cexprs CPU time,
/// 1454 -> 2719 core-seconds, once the run was CPU-bound).
pub const EXPRS_PER_TU: usize = 128;

/// Programs above this many ops are never tried straight-line: one function that size compiles
/// slowly (cicc is superlinear in function size) and all but certainly spills. The autotuned path
/// chunks them directly (`lib::build_exprs_tus`, every generation path).
pub const GENERIC_STRAIGHT_MAX: usize = 1024;

/// A chunked generic kernel as tuned: (key, scratch words per row, ops per chunk).
pub type ChunkedPick = (ExprKey, u64, usize);

/// Identity of a generic-expression kernel: one expression, or a hint pair `num x den^-1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprKey {
    Single(i64),
    Pair(i64, i64),
}

impl ExprKey {
    /// Kernel-name suffix: `x<id>` / `xp<num>_<den>`; the kernel is `gen_<sym>_<tag>`.
    pub fn tag(&self) -> String {
        match self {
            ExprKey::Single(id) => format!("x{id}"),
            ExprKey::Pair(n, d) => format!("xp{n}_{d}"),
        }
    }
}

/// Elements of scratch one wave of the Q launcher needs (`exps_min_scratch`): the cross-chunk
/// slots of one wave plus the per-launch table.
pub fn q_min_scratch(ir: &Ir, sym: &str, total_slots: u64) -> u64 {
    total_slots * GEN_BLK + pow_region_words(ir, sym)
}

/// The keys among `items` / `pairs` whose program exceeds `GENERIC_STRAIGHT_MAX` ops (a pair counts both).
pub fn oversized_keys(items: &[(i64, Ir, u64)], pairs: &[(i64, i64)]) -> Vec<ExprKey> {
    let ops: HashMap<i64, usize> = items.iter().map(|(id, ir, _)| (*id, ir.instrs.len())).collect();
    let mut big: Vec<ExprKey> = items
        .iter()
        .filter(|(_, ir, _)| ir.instrs.len() > GENERIC_STRAIGHT_MAX)
        .map(|(id, _, _)| ExprKey::Single(*id))
        .collect();
    big.extend(
        pairs
            .iter()
            .filter(|(n, d)| ops.get(n).unwrap_or(&0) + ops.get(d).unwrap_or(&0) > GENERIC_STRAIGHT_MAX)
            .map(|&(n, d)| ExprKey::Pair(n, d)),
    );
    big
}

/// Index of the part (`EXPRS_PER_TU` items each) holding expression `exp_id`'s kernel; a pair
/// kernel lives in its numerator's part.
pub fn exprs_part_of(items: &[(i64, Ir, u64)], exp_id: i64) -> Option<usize> {
    items.iter().position(|(id, _, _)| *id == exp_id).map(|i| i / EXPRS_PER_TU)
}

/// Part `k` of the generic expressions (`gen_<sym>_cexprs_p<k>.cu`): its `EXPRS_PER_TU` items'
/// kernels and the pair kernels whose numerator is among them, with `_p<k>`-suffixed dispatch
/// functions over them. `drop` leaves those kernels out (the ones re-emitted chunked, or
/// excluded): the part keeps its index and the other kernels keep their part, so only the parts
/// that lost a kernel change.
pub fn emit_exprs_part(
    sym: &str,
    items: &[(i64, Ir, u64)],
    pairs: &[(i64, i64)],
    drop: &[ExprKey],
    k: usize,
) -> (String, String) {
    let part = &items[k * EXPRS_PER_TU..((k + 1) * EXPRS_PER_TU).min(items.len())];
    (format!("gen_{sym}_cexprs_p{k}.cu"), exprs_part(sym, part, items, pairs, drop, k))
}

/// Every part, see [`emit_exprs_part`].
pub fn emit_exprs_parts(
    sym: &str,
    items: &[(i64, Ir, u64)],
    pairs: &[(i64, i64)],
    drop: &[ExprKey],
) -> Vec<(String, String)> {
    (0..items.len().div_ceil(EXPRS_PER_TU)).map(|k| emit_exprs_part(sym, items, pairs, drop, k)).collect()
}

/// A generic-expression kernel re-emitted as register-bounded chunks (its straight-line kernel
/// spilled, or was never tried). `ir` is the whole program -- for a pair the numerator followed by
/// the denominator, temps renumbered apart -- ending in temps, and `results` the temps the final
/// store consumes (one per expression, with its dim).
pub struct ChunkedExpr {
    pub key: ExprKey,
    pub ir: Ir,
    pub results: Vec<(u64, u64)>,
}

impl ChunkedExpr {
    /// The result temps, pinned live to the end of the program for the chunk plan.
    pub fn pinned(&self) -> Vec<u64> {
        self.results.iter().map(|&(t, _)| t).collect()
    }
}

/// Largest tmp id used or defined in `ir`, plus one (0 for a tmp-free program).
fn tmp_bound(ir: &Ir) -> u64 {
    ir.instrs
        .iter()
        .flat_map(|i| [i.a.as_tmp().map(|t| t.0), i.b.as_tmp().map(|t| t.0), i.dst_id.filter(|_| i.dst_is_tmp)])
        .flatten()
        .max()
        .map_or(0, |m| m + 1)
}

/// Append `src` to `instrs` with its temps shifted by `off`. A trailing non-tmp store (the
/// explicit output of the setup's program) becomes a fresh tmp so every result is a temp the
/// chunked store can consume; returns that result temp and its dim.
fn append_program(src: &Ir, off: u64, instrs: &mut Vec<Instr>, fresh: &mut u64) -> (u64, u64) {
    let shift = |o: &Operand| match o {
        Operand::Tmp { id, dim } => Operand::Tmp { id: id + off, dim: *dim },
        other => other.clone(),
    };
    let n = src.instrs.len();
    let mut result = (0u64, 0u64);
    for (k, i) in src.instrs.iter().enumerate() {
        let last = k + 1 == n;
        let (dst_is_tmp, dst_id) = if i.dst_is_tmp {
            (true, i.dst_id.map(|d| d + off))
        } else if last {
            let id = *fresh;
            *fresh += 1;
            (true, Some(id))
        } else {
            (false, i.dst_id)
        };
        if last {
            result = (dst_id.expect("result tmp"), i.ddim);
        }
        let idx = instrs.len();
        instrs.push(Instr { op: i.op.clone(), a: shift(&i.a), b: shift(&i.b), dst_is_tmp, dst_id, ddim: i.ddim, idx });
    }
    result
}

/// The challenge-power regions of two programs merged: per challenge base the larger exponent
/// count, sorted by base (the order `Ir::pow` keeps).
fn merge_pows(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut powers: BTreeMap<u64, u64> = BTreeMap::new();
    for &(base, n) in a.iter().chain(b.iter()) {
        powers.entry(base).and_modify(|old| *old = (*old).max(n)).or_insert(n);
    }
    powers.into_iter().collect()
}

/// The chunked program of one generic kernel: `num` alone, or `num` then `den` with the
/// denominator's temps renumbered above the numerator's, powers merged, no table (register
/// powers, as the straight-line kernels).
pub fn chunked_program(key: ExprKey, num: &Ir, den: Option<&Ir>) -> ChunkedExpr {
    let off_den = tmp_bound(num);
    let mut fresh = off_den + den.map_or(0, tmp_bound);
    let mut instrs: Vec<Instr> = Vec::with_capacity(num.instrs.len() + den.map_or(0, |d| d.instrs.len()));
    let mut results = vec![append_program(num, 0, &mut instrs, &mut fresh)];
    if let Some(d) = den {
        results.push(append_program(d, off_den, &mut instrs, &mut fresh));
    }
    let ir = Ir {
        instrs,
        ncols: num.ncols.clone(),
        n_constants: num.n_constants,
        n_bits: num.n_bits,
        pow: merge_pows(&num.pow, den.map_or(&[][..], |d| &d.pow)),
        tab: Vec::new(),
        tab_out: Vec::new(),
        tab_words: 0,
        pow_in_regs: true,
        field: num.field.clone(),
        zi: num.zi.clone(),
    };
    ChunkedExpr { key, ir, results }
}

/// `g3 <name>` from the value `expr` of dim `dim` (a base-field value is padded to the cubic one).
fn result_to_g3(name: &str, expr: &str, dim: u64) -> String {
    if dim == 3 {
        format!("g3 {name}={expr};")
    } else {
        format!("g3 {name}; {name}.a={expr}; {name}.b=gl64_t(uint64_t(0)); {name}.c=gl64_t(uint64_t(0));")
    }
}

/// The store of one row's value `expr` (dim `dim`) through the mode-parameterized
/// `exps_expr_store`, the same text for the straight-line and the chunked kernels.
fn store_single(expr: &str, dim: u64) -> String {
    format!(
        "  {{ {} exps_expr_store(dest,row,destDomain,stagePos,stageCols,destDim,destExpr,mode,scalar,v_,{dim}); }}",
        result_to_g3("v_", expr, dim)
    )
}

/// The store of one row's `num x den^-1` through `exps_expr_store_pair`.
fn store_pair(num: &str, num_dim: u64, den: &str, den_dim: u64) -> String {
    format!(
        "  {{ {} {} exps_expr_store_pair(dest,row,destDomain,stagePos,stageCols,destDim,destExpr,num_,{num_dim},den_,{den_dim}); }}",
        result_to_g3("num_", num, num_dim),
        result_to_g3("den_", den, den_dim)
    )
}

/// The C-ABI parameter lists of the generic-expression launchers, each written once. The parts'
/// own launchers (`exps_launch_expr_p<k>`, straight-line kernels: `GENERIC_LAUNCH_SIG` single,
/// `GENERIC_PAIR_LAUNCH_SIG` pair) take no scratch; the exported ones (`exps_generic_launch` = expId +
/// `GENERIC_SCRATCH_SIG`, `exps_generic_launch_pair` = `GENERIC_PAIR_SCRATCH_SIG`, the loader's `ExprLaunchFn` /
/// `ExprPairLaunchFn` in expressions_codegen.cuh) carry the expression scratch (offset inside
/// aux_trace, elements). `GENERIC_SCRATCH_SIG` is also the chunked launcher's (`run_<sym>_<tag>`, single and
/// pair alike).
const GENERIC_LAUNCH_SIG: &str = "unsigned long long expId, StepsParams* P, gl64_t* dest, unsigned long long N, unsigned long long destDomain, unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3, unsigned long long mode, unsigned long long scalar, unsigned long long stagePos, unsigned long long stageCols, unsigned long long destDim, unsigned int destExpr, cudaStream_t stream";
const GENERIC_PAIR_LAUNCH_SIG: &str = "unsigned long long numId, unsigned long long denId, StepsParams* P, gl64_t* dest, unsigned long long N, unsigned long long destDomain, unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3, unsigned long long stagePos, unsigned long long stageCols, unsigned long long destDim, unsigned int destExpr, cudaStream_t stream";
const GENERIC_SCRATCH_SIG: &str = "StepsParams* P, gl64_t* dest, unsigned long long N, unsigned long long destDomain, unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3, unsigned long long mode, unsigned long long scalar, unsigned long long stagePos, unsigned long long stageCols, unsigned long long destDim, unsigned int destExpr, unsigned long long scratchOff, unsigned long long scratchElems, cudaStream_t stream";
const GENERIC_PAIR_SCRATCH_SIG: &str = "unsigned long long numId, unsigned long long denId, StepsParams* P, gl64_t* dest, unsigned long long N, unsigned long long destDomain, unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3, unsigned long long stagePos, unsigned long long stageCols, unsigned long long destDim, unsigned int destExpr, unsigned long long scratchOff, unsigned long long scratchElems, cudaStream_t stream";

/// Parameters of a straight-line generic kernel (single); the pair kernel has no mode/scalar.
const GENERIC_KERNEL_SIG: &str = "(const StepsParams* __restrict__ P, gl64_t* __restrict__ dest,\n    uint64_t N, uint64_t destDomain, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3,\n    uint64_t mode, uint64_t scalar, uint64_t stagePos, uint64_t stageCols, uint64_t destDim, uint32_t destExpr";
const GENERIC_PAIR_KERNEL_SIG: &str = "(const StepsParams* __restrict__ P, gl64_t* __restrict__ dest, uint64_t N, uint64_t destDomain, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t stagePos, uint64_t stageCols, uint64_t destDim, uint32_t destExpr)";

/// The adaptive wave grid of a scratch-bounded launcher: as many blocks as the scratch holds
/// cross-chunk slots for (`slots x BLK` words per block), never more than the rows need, at
/// least one. Each wave then covers `WAVE = grid x BLK` rows and waits for the previous one.
fn wave_grid(slots: u64, rows: &str) -> String {
    format!(
        r#"  const uint64_t need = ({rows} + BLK - 1) / BLK;
  uint64_t grid = {slots}ull ? scratchElems / ({slots}ull*BLK) : need;
  if (grid > need) grid = need;
  if (grid < 1ull) grid = 1ull;
  const uint64_t WAVE = grid * BLK;"#
    )
}

/// The chunked generic kernels of one AIR in one TU (`gen_<sym>_cexprs_sp.cu`): per item its
/// chunk kernels `gen_<sym>_<tag>_c<k>` (Q-style: cut temps parked in scratch at
/// `slot*WAVE + lo_`, one row per thread per wave) and the host launcher `run_<sym>_<tag>`
/// (adaptive grid from the scratch size, wave loop, 0 when the scratch cannot hold one wave).
/// Scratch is `aux_trace + scratchOff`, computed on the device so the loader passes offsets only.
pub fn emit_exprs_chunked_tu(sym: &str, items: &[(&ChunkedExpr, &ChunkPlan)]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (ce, plan) in items {
        let tag = ce.key.tag();
        let pw = pwregs_lines(&ce.ir);
        let store = match ce.key {
            ExprKey::Single(_) => {
                let (t, dim) = ce.results[0];
                store_single(&format!("t{t}"), dim)
            }
            ExprKey::Pair(_, _) => {
                let (tn, dn) = ce.results[0];
                let (td, dd) = ce.results[1];
                store_pair(&format!("t{tn}"), dn, &format!("t{td}"), dd)
            }
        };
        let pinned = ce.pinned();
        for k in 0..plan.n_chunks {
            let last = k + 1 == plan.n_chunks;
            let mut lines = chunk_body(&ce.ir, plan, k, if last { &pinned } else { &[] });
            if last {
                lines.push(store.clone());
            }
            parts.push(format!(
                r#"__global__ void gen_{sym}_{tag}_c{k}{GENERIC_KERNEL_SIG},
    uint64_t scratchOff, uint64_t tileBase) {{
  const uint64_t WAVE = (uint64_t)gridDim.x*blockDim.x;
  const uint64_t lo_ = blockIdx.x*blockDim.x + threadIdx.x; const uint64_t row = tileBase + lo_;
  if (row >= N) return;
{TRACE_PROLOGUE}
  gl64_t* scratch = (gl64_t*)P->aux_trace + scratchOff;
{pw}{}
}}"#,
                lines.join("\n"),
                TRACE_PROLOGUE = trace_prologue()
            ));
        }
        let calls: Vec<String> = (0..plan.n_chunks)
            .map(|k| format!("    gen_{sym}_{tag}_c{k}<<<grid,BLK,0,stream>>>(P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,mode,scalar,stagePos,stageCols,destDim,destExpr,scratchOff,base);"))
            .collect();
        parts.push(format!(
            r#"// {tag}: {} chunks, {slots} cross-chunk scratch words per row
extern "C" int run_{sym}_{tag}({GENERIC_SCRATCH_SIG}) {{
  const uint64_t BLK = {GEN_BLK}ull;
  if (scratchElems < {slots}ull*BLK) return 0;
{}
  for (uint64_t base = 0; base < N; base += WAVE) {{
{}
  }}
  return 1;
}}"#,
            plan.n_chunks,
            wave_grid(plan.total_slots, "N"),
            calls.join("\n"),
            slots = plan.total_slots
        ));
    }
    format!(
        r#"// AUTO-GENERATED chunked generic expression kernels for {sym} ({} kernels)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{}
#undef OFF
"#,
        items.len(),
        parts.join("\n")
    )
}

/// The C-ABI dispatch TU (`gen_<sym>_cexprs.cu`) over `n_parts` kernel parts and the `chunked`
/// kernels of `gen_<sym>_cexprs_sp.cu`: `exps_generic_covered` / `exps_generic_launch` / `exps_generic_launch_pair`
/// (the loader's `ExprCoveredFn` / `ExprLaunchFn` / `ExprPairLaunchFn`), carrying the expression
/// scratch (offset inside aux_trace, elements). A chunked kernel is routed to its launcher, which
/// returns 0 when the scratch cannot hold one wave; everything else chains the parts, whose
/// straight-line kernels need no scratch. A chunked or excluded expression is in no part
/// (`emit_exprs_part`'s `drop`), so an uncovered one reports 0 and the prover's interpreter runs.
pub fn emit_exprs_dispatch(sym: &str, n_parts: usize, chunked: &[ChunkedPick]) -> String {
    let mut decls: String = (0..n_parts)
        .map(|k| {
            format!(
                r#"extern "C" int exps_expr_covered_p{k}(unsigned long long);
extern "C" int exps_launch_expr_p{k}({GENERIC_LAUNCH_SIG});
extern "C" int exps_launch_expr_pair_p{k}({GENERIC_PAIR_LAUNCH_SIG});
"#
            )
        })
        .collect();
    let chain = |call: &dyn Fn(usize) -> String| -> String {
        (0..n_parts).map(|k| format!("  if ({}) return 1;\n", call(k))).collect()
    };
    let run = |tag: &str, mode: &str| {
        format!("run_{sym}_{tag}(P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,{mode},stagePos,stageCols,destDim,destExpr,scratchOff,scratchElems,stream)")
    };
    // The chunked kernels first (a pair launcher gets mode/scalar 0, unused), then the parts.
    let (mut covered, mut launch, mut pair) = (String::new(), String::new(), String::new());
    for &(key, _, _) in chunked {
        let tag = key.tag();
        decls.push_str(&format!("extern \"C\" int run_{sym}_{tag}({GENERIC_SCRATCH_SIG});\n"));
        match key {
            ExprKey::Single(id) => {
                covered.push_str(&format!("  if (expId == {id}ull) return 1;\n"));
                launch.push_str(&format!("  if (expId == {id}ull) return {};\n", run(&tag, "mode,scalar")));
            }
            ExprKey::Pair(n, d) => {
                pair.push_str(&format!(
                    "  if (numId == {n}ull && denId == {d}ull) return {};\n",
                    run(&tag, "0ull,0ull")
                ));
            }
        }
    }
    covered.push_str(&chain(&|k| format!("exps_expr_covered_p{k}(expId)")));
    launch.push_str(&chain(&|k| {
        format!("exps_launch_expr_p{k}(expId,P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,mode,scalar,stagePos,stageCols,destDim,destExpr,stream)")
    }));
    pair.push_str(&chain(&|k| {
        format!("exps_launch_expr_pair_p{k}(numId,denId,P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,stagePos,stageCols,destDim,destExpr,stream)")
    }));
    format!(
        r#"// AUTO-GENERATED generic expression dispatch for {sym} ({n_parts} parts, {} chunked)
#include "gen_common.cuh"
{decls}extern "C" int exps_generic_covered(unsigned long long expId) {{
{covered}  return 0;
}}
extern "C" int exps_generic_launch(unsigned long long expId, {GENERIC_SCRATCH_SIG}) {{
{launch}  (void)scratchOff; (void)scratchElems;
  return 0;
}}
extern "C" int exps_generic_launch_pair({GENERIC_PAIR_SCRATCH_SIG}) {{
{pair}  (void)scratchOff; (void)scratchElems;
  return 0;
}}
"#,
        chunked.len()
    )
}

/// Challenge powers `pwreg_<base>_<j>` for `pow` regions, computed once per thread in a kernel
/// prologue by repeated cg_mul33 from the challenge, the same arithmetic as the Q table so the
/// folded chains stay exact. Empty when no region needs a power above the first.
fn pwregs_lines_for(pow: &[(u64, u64)]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for &(base, n) in pow {
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

/// The register-resident powers of a `pow_in_regs` kernel (see [`pwregs_lines_for`]).
fn pwregs_lines(ir: &Ir) -> String {
    if ir.pow_in_regs {
        pwregs_lines_for(&ir.pow)
    } else {
        String::new()
    }
}

/// One generic-expression part TU: the kernels of `part_items` (+ the pair kernels whose
/// numerator is among them; `all_items` resolves the denominators) except the `drop`ped ones,
/// and the `_p<part>`-suffixed dispatch functions over them.
fn exprs_part(
    sym: &str,
    part_items: &[(i64, Ir, u64)],
    all_items: &[(i64, Ir, u64)],
    pairs: &[(i64, i64)],
    drop: &[ExprKey],
    part: usize,
) -> String {
    let sfx = format!("_p{part}");
    let items: Vec<&(i64, Ir, u64)> =
        part_items.iter().filter(|(id, _, _)| !drop.contains(&ExprKey::Single(*id))).collect();
    // The per-row evaluation of a program as a lambda: the result lives in the last
    // instruction's destination, `qq` for an explicit output store, `t<id>` for a tmp.
    let eval_lambda = |ir: &Ir, name: &str, vt: &str| -> String {
        let mut body: Vec<String> = Vec::new();
        let mut declared: HashSet<u64> = HashSet::new();
        for instr in &ir.instrs {
            let (op_lines, is_out) = emit_op(instr, ir, &declared);
            body.extend(op_lines);
            if !is_out {
                declared.insert(instr.dst_id.unwrap());
            }
        }
        let last = ir.instrs.last().unwrap();
        let result = if last.dst_is_tmp { format!("t{}", last.dst_id.unwrap()) } else { "qq".to_string() };
        format!("  auto {name} = [&](uint64_t row) -> {vt} {{\n{}\n    return {result};\n  }};", body.join("\n"))
    };
    let vtype = |dim: u64| if dim == 3 { "g3" } else { "gl64_t" };
    let mut kernels: Vec<String> = Vec::new();
    let mut cases_launch: Vec<String> = Vec::new();
    let mut cases_covered: Vec<String> = Vec::new();
    for (exp_id, ir, out_dim) in &items {
        let tag = ExprKey::Single(*exp_id).tag();
        let vt = vtype(*out_dim);
        kernels.push(format!(
            r#"__global__ void gen_{sym}_{tag}{GENERIC_KERNEL_SIG}) {{
{TRACE_PROLOGUE}
{}{}
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<N; row+=(uint64_t)gridDim.x*blockDim.x) {{
    {vt} ev_ = eval_(row);
  {}
  }}
}}"#,
            pwregs_lines(ir),
            eval_lambda(ir, "eval_", vt),
            store_single("ev_", *out_dim),
            TRACE_PROLOGUE = trace_prologue()
        ));
        cases_launch.push(format!("    case {exp_id}ull: gen_{sym}_{tag}<<<grid,{GEN_BLK},0,stream>>>(P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,mode,scalar,stagePos,stageCols,destDim,destExpr); return 1;"));
        cases_covered.push(format!("    case {exp_id}ull: return 1;"));
    }
    let item_by_id: HashMap<i64, (&Ir, u64)> = all_items.iter().map(|(id, ir, dim)| (*id, (ir, *dim))).collect();
    let mut pair_kernels = Vec::new();
    let mut pair_cases = Vec::new();
    for &(num_id, den_id) in pairs {
        if drop.contains(&ExprKey::Pair(num_id, den_id)) || !part_items.iter().any(|(id, _, _)| *id == num_id) {
            continue; // dropped, or emitted by the part holding the numerator
        }
        let (Some(&(num_ir, num_dim)), Some(&(den_ir, den_dim))) = (item_by_id.get(&num_id), item_by_id.get(&den_id))
        else {
            continue;
        };
        let tag = ExprKey::Pair(num_id, den_id).tag();
        pair_kernels.push(format!(
            r#"__global__ void gen_{sym}_{tag}{GENERIC_PAIR_KERNEL_SIG} {{
{TRACE_PROLOGUE}
{}{}
{}
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<N; row+=(uint64_t)gridDim.x*blockDim.x) {{
  {}
  }}
}}"#,
            pwregs_lines_for(&merge_pows(&num_ir.pow, &den_ir.pow)),
            eval_lambda(num_ir, "eval_num_", vtype(num_dim)),
            eval_lambda(den_ir, "eval_den_", vtype(den_dim)),
            store_pair("eval_num_(row)", num_dim, "eval_den_(row)", den_dim),
            TRACE_PROLOGUE = trace_prologue()
        ));
        pair_cases.push(format!("  if (numId=={num_id}ull && denId=={den_id}ull) {{ gen_{sym}_{tag}<<<grid,{GEN_BLK},0,stream>>>(P,dest,N,destDomain,off_cm1,off_cm2,off_cm3,stagePos,stageCols,destDim,destExpr); return 1; }}"));
    }
    format!(
        r#"// AUTO-GENERATED generic expression kernels for {sym} ({} expressions)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc,lyt) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc),(lyt))
{}
{}
extern "C" int exps_expr_covered{sfx}(unsigned long long expId) {{
  switch (expId) {{
{}
    default: return 0;
  }}
}}
extern "C" int exps_launch_expr{sfx}({GENERIC_LAUNCH_SIG}) {{
  // Cap rationale: see EXPR_GRID_CAP in emit.rs (shared with the pair launcher below).
  uint64_t grid = (N + {GEN_BLK}ull - 1) / {GEN_BLK}ull;
  if (grid > {EXPR_GRID_CAP}ull) grid = {EXPR_GRID_CAP}ull;
  if (grid < 1ull) grid = 1ull;
  switch (expId) {{
{}
    default: return 0;
  }}
}}
extern "C" int exps_launch_expr_pair{sfx}({GENERIC_PAIR_LAUNCH_SIG}) {{
  uint64_t grid=(N+{GEN_BLK}ull-1)/{GEN_BLK}ull; if(grid>{EXPR_GRID_CAP}ull)grid={EXPR_GRID_CAP}ull; if(grid<1ull)grid=1ull;
{}
  return 0;
}}
#undef OFF
"#,
        items.len(),
        kernels.join("\n"),
        pair_kernels.join("\n"),
        cases_covered.join("\n"),
        cases_launch.join("\n"),
        pair_cases.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;

    fn ir(instrs: Vec<Instr>, pow: Vec<(u64, u64)>) -> Ir {
        Ir {
            instrs,
            ncols: HashMap::new(),
            n_constants: 0,
            n_bits: 4,
            pow,
            tab: Vec::new(),
            tab_out: Vec::new(),
            tab_words: 0,
            pow_in_regs: true,
            field: Field::Goldilocks,
            zi: Default::default(),
        }
    }
    fn op(idx: usize, a: Operand, b: Operand, dst: Option<u64>) -> Instr {
        Instr { op: "add".into(), a, b, dst_is_tmp: dst.is_some(), dst_id: dst, ddim: 1, idx }
    }
    fn tmp(id: u64) -> Operand {
        Operand::Tmp { id, dim: 1 }
    }

    /// A pair program is the numerator then the denominator with the denominator's temps moved
    /// above the numerator's, a trailing output store turned into a fresh temp, op indices
    /// renumbered, and the powers merged by max exponent.
    #[test]
    fn chunked_program_merges_a_pair() {
        // num: t0 = n1 + n2; t1 = t0 + n3            (ends in a tmp)
        let num = ir(
            vec![op(0, Operand::Num(1), Operand::Num(2), Some(0)), op(1, tmp(0), Operand::Num(3), Some(1))],
            vec![(0, 3)],
        );
        // den: t0 = n4 + n5; out = t0 + n6            (ends in an output store)
        let den = ir(
            vec![op(0, Operand::Num(4), Operand::Num(5), Some(0)), op(1, tmp(0), Operand::Num(6), None)],
            vec![(0, 5), (3, 2)],
        );
        let ce = chunked_program(ExprKey::Pair(7, 8), &num, Some(&den));
        let ids: Vec<Option<u64>> = ce.ir.instrs.iter().map(|i| i.dst_id).collect();
        assert_eq!(ids, vec![Some(0), Some(1), Some(2), Some(3)]);
        assert!(ce.ir.instrs.iter().all(|i| i.dst_is_tmp));
        assert_eq!(ce.ir.instrs[3].a, tmp(2), "den's t0 reads as t2 after the shift");
        assert_eq!(ce.ir.instrs.iter().map(|i| i.idx).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(ce.results, vec![(1, 1), (3, 1)]);
        assert_eq!(ce.pinned(), vec![1, 3]);
        assert_eq!(ce.ir.pow, vec![(0, 5), (3, 2)]);
        assert_eq!(ce.key.tag(), "xp7_8");
    }

    /// Pinning keeps a result produced early alive across every later cut, so the last chunk
    /// loads it from scratch.
    #[test]
    fn pinned_result_crosses_the_cuts() {
        let num = ir(vec![op(0, Operand::Num(1), Operand::Num(2), Some(0))], vec![]);
        let den = ir(
            vec![
                op(0, Operand::Num(4), Operand::Num(5), Some(0)),
                op(1, tmp(0), Operand::Num(6), Some(1)),
                op(2, tmp(1), Operand::Num(6), Some(2)),
            ],
            vec![],
        );
        let ce = chunked_program(ExprKey::Pair(1, 2), &num, Some(&den));
        let plan = crate::ir::plan_chunks_ops_pinned(&ce.ir.instrs, 2, "t", &ce.pinned()).unwrap();
        assert!(plan.n_chunks >= 2);
        assert!(plan.cut_temps.contains(&0), "num result t0 must be a cut temp");
        let last = plan.n_chunks - 1;
        let body = chunk_body(&ce.ir, &plan, last, &ce.pinned()).join("\n");
        assert!(body.contains("gl64_t t0 = scratch["), "last chunk loads the pinned numerator: {body}");
    }
}
