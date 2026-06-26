#!/usr/bin/env python3
"""
Generalized Q codegen with AUTOMATIC register-bounded chunking.

For each AIR's Q (cExp) expression: if it is small it becomes one straight-line
kernel; if it is large it is split into register-bounded chunks (the chunk size
is auto-tuned per AIR to compile with no register spill). Temps that cross a chunk boundary
(the DAG "cut") are materialized to a global scratch buffer (interval-colored to
keep it small); everything else stays in registers. Chunks run in sequence on
the stream.

AIRS_DIR is a provingKey dir, globbed recursively. Each AIR's kernel is
compiled into its own <base>.exps.so next to that AIR's .bin and loaded by path
at runtime.
"""
import json, sys, os, glob, subprocess, tempfile, re, shutil, argparse, atexit

# ---- internal constants (not CLI flags; edit here to tune) ----
GEN_BLK       = 256   # CUDA threads/block of the generated kernels. Single source of truth: baked into each
                      # launcher AND exported via exps_min_scratch() so the loader never hardcodes the block size.
DEFAULT_CHUNK = 512   # fixed ops/chunk used when the autotuner is off and --chunk wasn't given
# No-spill autotuner bisection bounds: start at min(n, CHUNK_MAX), halve until cuobjdump reports STACK=0; if it
# still spills at CHUNK_MIN, skip the AIR (-> interpreter, never a silent per-thread local-memory OOM).
CHUNK_MAX     = 512   # start chunk
CHUNK_MIN     = 64    # give-up floor
SLOTS_CAP     = 1000  # skip an AIR whose cross-chunk cut (slots) exceeds this
# For large expressions, with > BIG_OPS ops, the autotuner starts at a smaller chunk size (BIG_START_CHUNK) because we observe that generally spill with CHUNK_MAX
BIG_OPS         = 20000  # ops threshold for "large expressions"
BIG_START_CHUNK = 250    # autotuner start chunk for AIRs with > BIG_OPS ops

#Chunks per TU granulairty at compile time
CHUNKS_PER_TU   = 8

# All configuration is via CLI flags (no environment variables) — run with -h for the list.
_ap = argparse.ArgumentParser(description="Per-AIR Q-expression -> straight-line CUDA kernel codegen.")
_ap.add_argument("airs_dir", help="provingKey build dir (globbed for *.starkinfo.json + *.expressionsinfo.json)")
_ap.add_argument("outdir", nargs="?", default=None, help="output dir for gen_*.cu / gen_common.cuh / gen.log; "
                 "omit => a temp dir deleted on exit")
_ap.add_argument("--cap", type=int, default=40000, help="skip an AIR whose Q has more ops than this")
_ap.add_argument("--nvcc-cmd", default="", help="nvcc compile command (flags only); turns ON the no-spill "
                 "autotuner (per-AIR chunk compile-verified to zero register spill). Empty => autotuner off.")
_ap.add_argument("--chunk", type=int, default=None, help="fixed ops/chunk for every AIR; setting it forces the "
                 f"autotuner OFF (manual chunking).")
_args = _ap.parse_args()

AIRS_DIR = _args.airs_dir
# No outdir => generate into a temp dir removed on exit. Pass an explicit outdir to retain
# the .cu/.o/gen.log.
OUTDIR   = _args.outdir if _args.outdir is not None else tempfile.mkdtemp(prefix="genexps_")
if _args.outdir is None: atexit.register(shutil.rmtree, OUTDIR, ignore_errors=True)
CAP      = _args.cap
NVCC_CMD = _args.nvcc_cmd
# Autotuner runs iff an nvcc command is given AND no explicit --chunk (a fixed --chunk means "manual chunking").
AUTOTUNE    = (_args.chunk is None) and bool(NVCC_CMD)
CHUNK_FIXED = _args.chunk if _args.chunk is not None else DEFAULT_CHUNK

def _max_stack(obj):
    try:
        out = subprocess.run(["cuobjdump", "-res-usage", obj], capture_output=True, text=True).stdout
    except Exception:
        return -1
    vals = [int(m) for m in re.findall(r"STACK:(\d+)", out)]
    return max(vals) if vals else 0

def _compile_one(cuf, obj, incdir):
    r = subprocess.run(NVCC_CMD.split() + [cuf, "-o", obj, f"-I{incdir}"], capture_output=True, text=True)
    return r.returncode, r.stderr

# Returns the largest chunk (halving from the start size) that compiles with STACK=0, or None if it
# still spills at CHUNK_MIN.
def tune_chunk(stark_info, expr_info, sym, n, common_cuh, cc_pool):
    probe = tempfile.mkdtemp(prefix="genqtune_")
    open(os.path.join(probe, "gen_common.cuh"), "w").write(common_cuh)
    try:
        chunk = min(n, BIG_START_CHUNK if n > BIG_OPS else CHUNK_MAX)
        while chunk >= CHUNK_MIN:
            files, slots = emit_air(stark_info, expr_info, sym, force_chunk=chunk)
            futs = []
            for fn, txt in files:
                cuf = os.path.join(probe, fn); open(cuf, "w").write(txt)
                obj = os.path.join(probe, fn[:-3] + ".o")
                futs.append((obj, cc_pool.submit(_compile_one, cuf, obj, probe)))
            results = [(obj, fut.result()) for obj, fut in futs]   # wait for every TU to compile
            fail = next((err for obj, (rc, err) in results if rc != 0), None)
            if fail is not None:
                print(f"  [tune] {sym} chunk={chunk} COMPILE FAILED: {fail.strip().splitlines()[-1:]}")
                return None
            st = max((_max_stack(obj) for obj, _ in results), default=0)
            print(f"  [tune] {sym} chunk={chunk} ({len(files)} TUs) -> STACK={st}")
            if st == 0:
                # keep every winning .o (launcher + chunks); build_exps.sh links them, no recompile.
                for obj, _ in results:
                    try: shutil.copy(obj, os.path.join(OUTDIR, os.path.basename(obj)))
                    except Exception: pass
                return chunk
            chunk //= 2
        return None
    finally:
        shutil.rmtree(probe, ignore_errors=True)

# ---------- operand resolution ----------
def build_ir(stark_info, expr_info):
    opening = stark_info["openingPoints"]; nBits = stark_info["starkStruct"]["nBits"]
    blowup = 1 << (stark_info["starkStruct"]["nBitsExt"] - nBits)
    nConstants = stark_info["nConstants"]; map_sections_n = stark_info["mapSectionsN"]; cm_pols_map = stark_info["cmPolsMap"]
    air_values_map = stark_info["airValuesMap"]; airgroup_values_map = stark_info.get("airgroupValuesMap", [])
    ncols = {stage: map_sections_n.get("cm" + str(stage), 0) for stage in range(1, stark_info["nStages"] + 2)}
    code = {e["expId"]: e for e in expr_info["expressionsCode"]}[stark_info["cExpId"]]["code"]

    def stride(prime): return opening[opening.index(prime)] * blowup
    def av_pos(idx): return sum(3 if air_values_map[j].get("stage", 1) != 1 else 1 for j in range(idx))
    def agv_pos(idx): return sum(3 if airgroup_values_map[j].get("stage", 1) != 1 else 1 for j in range(idx))

    def operand(src):
        op_type = src["type"]; dim = src.get("dim", 1)
        if op_type == "tmp": return ("tmp", src["id"], dim)
        if op_type == "number": return ("num", int(src["value"]), 1)
        if op_type == "cm":
            cm = cm_pols_map[src["id"]]; return ("cm", (cm["stage"], cm["stagePos"], cm["dim"], stride(src.get("prime", 0))), cm["dim"])
        if op_type == "const": return ("const", (src["id"], stride(src.get("prime", 0))), 1)
        if op_type == "Zi": return ("zi", src.get("boundaryId", 0), 1)
        if op_type == "challenge": return ("ch", 3 * src["id"], 3)
        if op_type == "airvalue": return ("av", av_pos(src["id"]), dim)
        if op_type == "airgroupvalue": return ("agv", agv_pos(src["id"]), dim)
        if op_type == "public": return ("pub", src["id"], 1)
        raise NotImplementedError(op_type)

    ir = [{"op": step["op"], "a": operand(step["src"][0]), "b": operand(step["src"][1]),
           "dst_kind": step["dest"]["type"], "dst_id": step["dest"].get("id"), "ddim": step["dest"]["dim"]} for step in code]
    return ir, ncols, nConstants

def rowexpr(stride): return "row" if stride == 0 else f"((row+({stride}ll))&MASK)"
def load_lines(opnd, name, ncols, nConstants):
    kind = opnd[0]
    if kind == "num": return [f"  gl64_t {name}(uint64_t({opnd[1]}ull));"], 1
    if kind == "zi": return [f"  gl64_t {name} = aux[off_zi + row];"], 1
    if kind == "pub": return [f"  gl64_t {name} = pub[{opnd[1]}];"], 1
    if kind == "ch":
        i = opnd[1]; return [f"  g3 {name}; {name}.a=ch[{i}]; {name}.b=ch[{i+1}]; {name}.c=ch[{i+2}];"], 3
    if kind in ("av", "agv"):
        arr = kind; i = opnd[1]
        if opnd[2] == 1: return [f"  gl64_t {name} = {arr}[{i}];"], 1
        return [f"  g3 {name}; {name}.a={arr}[{i}]; {name}.b={arr}[{i+1}]; {name}.c={arr}[{i+2}];"], 3
    if kind == "const":
        const_id, stride = opnd[1]; return [f"  gl64_t {name} = cst[OFF({rowexpr(stride)},{const_id},NExt,{nConstants})];"], 1
    if kind == "cm":
        stage, pos, dim, stride = opnd[1]; row = rowexpr(stride); n_cols = ncols[stage]
        if dim == 1: return [f"  gl64_t {name} = aux[off_cm{stage} + OFF({row},{pos},NExt,{n_cols})];"], 1
        return [f"  g3 {name}; {name}.a=aux[off_cm{stage}+OFF({row},{pos},NExt,{n_cols})]; {name}.b=aux[off_cm{stage}+OFF({row},{pos+1},NExt,{n_cols})]; {name}.c=aux[off_cm{stage}+OFF({row},{pos+2},NExt,{n_cols})];"], 3
    raise NotImplementedError(kind)

def emit_op(instr, ncols, nConstants, declared):
    """Emit lines for one op; tmp operands -> t{id} (must already exist)."""
    lines = []
    op_a, op_b = instr["a"], instr["b"]
    if op_a[0] == "tmp": a_val, a_dim = f"t{op_a[1]}", op_a[2]
    else: loaded, a_dim = load_lines(op_a, f"a{instr['_i']}", ncols, nConstants); lines += loaded; a_val = f"a{instr['_i']}"
    if op_b[0] == "tmp": b_val, b_dim = f"t{op_b[1]}", op_b[2]
    else: loaded, b_dim = load_lines(op_b, f"b{instr['_i']}", ncols, nConstants); lines += loaded; b_val = f"b{instr['_i']}"
    dst_dim = instr["ddim"]; is_out = instr["dst_kind"] != "tmp"
    dst = "qq" if is_out else f"t{instr['dst_id']}"
    decl = "" if (not is_out and instr['dst_id'] in declared) else ("gl64_t " if dst_dim == 1 else "g3 ")
    if dst_dim == 1:
        op_symbol = {'add': '+', 'sub': '-', 'mul': '*'}[instr['op']]
        lines.append(f"  {decl}{dst} = {a_val} {op_symbol} {b_val};")
    else:
        lines.append(f"  {decl}{dst} = cg_{instr['op']}{a_dim}{b_dim}({a_val},{b_val});")
    return lines, dst_dim, is_out

# ---------- per-AIR emission (with chunking) ----------
def emit_air(stark_info, expr_info, sym, force_chunk=None):
    ir, ncols, nConstants = build_ir(stark_info, expr_info)
    for i, instr in enumerate(ir): instr["_i"] = i
    n_ops = len(ir)

    # liveness on tmp ids: def_idx[t] = op index where temp t is defined, last_use[t] = last op using it
    def_idx, dim_of, last_use = {}, {}, {}
    for i, instr in enumerate(ir):
        if instr["dst_kind"] == "tmp":
            def_idx[instr["dst_id"]] = i; dim_of[instr["dst_id"]] = instr["ddim"]
        for opnd in (instr["a"], instr["b"]):
            if opnd[0] == "tmp":
                last_use[opnd[1]] = i; dim_of[opnd[1]] = opnd[2]

    if force_chunk is not None:
        chunk = force_chunk    # autotuner: compile-verified no-spill size
    else:
        chunk = CHUNK_FIXED    # manual: one fixed chunk for every AIR (autotuner off)
    chunk = max(1, min(chunk, n_ops))
    n_chunks = (n_ops + chunk - 1) // chunk
    def chunk_of(op_idx): return op_idx // chunk

    out_dim = 3
    for instr in ir:
        if instr["dst_kind"] != "tmp": out_dim = instr["ddim"]

    if n_chunks <= 1:
        # single straight-line kernel (small expression)
        body = []
        declared = set()
        for instr in ir:
            op_lines, dst_dim, is_out = emit_op(instr, ncols, nConstants, declared)
            body += op_lines
            if not is_out: declared.add(instr["dst_id"])
        store = ("    q[OFF(row,0,NExt,3)]=qq.a; q[OFF(row,1,NExt,3)]=qq.b; q[OFF(row,2,NExt,3)]=qq.c;"
                 if out_dim == 3 else
                 "    q[OFF(row,0,NExt,3)]=qq; q[OFF(row,1,NExt,3)]=gl64_t(uint64_t(0)); q[OFF(row,2,NExt,3)]=gl64_t(uint64_t(0));")
        kernels = [f"""__global__ void gen_{sym}_kernel(const StepsParams* __restrict__ P, gl64_t* __restrict__ q,
    uint64_t NExt, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
  for (uint64_t row=blockIdx.x*blockDim.x+threadIdx.x; row<NExt; row+=gridDim.x*blockDim.x) {{
{chr(10).join(body)}
{store}
  }}
}}"""]
        launcher_body = f"  (void)scratch; (void)scratchElems; gen_{sym}_kernel<<<512,256,0,stream>>>(d_params,q,NExt,off_cm1,off_cm2,off_cm3,off_zi);"
        # single kernel -> one self-contained TU (nothing to parallelize)
        return [(f"gen_{sym}.cu", single_kernel_tu(sym, kernels, launcher_body, 0))], 0

    # ---- chunked (tiled): K register-bounded kernels. Each launch processes one WAVE of rows
    #      (= grid*block). Temps crossing chunk boundaries are materialized to a small per-wave
    #      scratch (cutWidth*WAVE), reused across waves -> fits the existing tmp buffer. ----
    cut_temps = set(t for t in def_idx if t in last_use and chunk_of(last_use[t]) > chunk_of(def_idx[t]))
    def color(temps, width):
        slot_of = {}; active = []; next_base = 0; free_bases = []
        for t in sorted(temps, key=lambda t: chunk_of(def_idx[t])):
            def_chunk, end_chunk = chunk_of(def_idx[t]), chunk_of(last_use[t])
            for (end, base) in [entry for entry in active if entry[0] < def_chunk]: free_bases.append(base)
            active = [entry for entry in active if entry[0] >= def_chunk]
            base = free_bases.pop() if free_bases else (next_base, next_base := next_base + width)[0]
            slot_of[t] = base; active.append((end_chunk, base))
        return slot_of, next_base
    slot1, n_slots1 = color([t for t in cut_temps if dim_of[t] == 1], 1)
    slot3, n_slots3 = color([t for t in cut_temps if dim_of[t] == 3], 3)
    base3 = n_slots1; total_slots = n_slots1 + n_slots3
    def slot_index(t): return slot1[t] if dim_of[t] == 1 else base3 + slot3[t]

    kernels = []
    for chunk_idx in range(n_chunks):
        lo_op, hi_op = chunk_idx * chunk, min((chunk_idx + 1) * chunk, n_ops)
        chunk_ops = ir[lo_op:hi_op]
        used_temps = set(opnd[1] for instr in chunk_ops for opnd in (instr["a"], instr["b"]) if opnd[0] == "tmp")
        live_in = sorted(t for t in used_temps if t in cut_temps and chunk_of(def_idx[t]) < chunk_idx)
        live_out = sorted(t for t in cut_temps if chunk_of(def_idx[t]) == chunk_idx)
        declared = set()
        lines = []
        for t in live_in:
            slot_base = slot_index(t)
            if dim_of[t] == 1: lines.append(f"  gl64_t t{t} = scratch[{slot_base}ull*WAVE + lo_];")
            else: lines.append(f"  g3 t{t}; t{t}.a=scratch[{slot_base}ull*WAVE+lo_]; t{t}.b=scratch[{slot_base+1}ull*WAVE+lo_]; t{t}.c=scratch[{slot_base+2}ull*WAVE+lo_];")
            declared.add(t)
        for instr in chunk_ops:
            op_lines, dst_dim, is_out = emit_op(instr, ncols, nConstants, declared)
            lines += op_lines
            if not is_out: declared.add(instr["dst_id"])
        for t in live_out:
            slot_base = slot_index(t)
            if dim_of[t] == 1: lines.append(f"  scratch[{slot_base}ull*WAVE + lo_] = t{t};")
            else: lines.append(f"  scratch[{slot_base}ull*WAVE+lo_]=t{t}.a; scratch[{slot_base+1}ull*WAVE+lo_]=t{t}.b; scratch[{slot_base+2}ull*WAVE+lo_]=t{t}.c;")
        if chunk_idx == n_chunks - 1:
            lines.append("    q[OFF(row,0,NExt,3)]=qq.a; q[OFF(row,1,NExt,3)]=qq.b; q[OFF(row,2,NExt,3)]=qq.c;" if out_dim == 3
                     else "    q[OFF(row,0,NExt,3)]=qq; q[OFF(row,1,NExt,3)]=gl64_t(uint64_t(0)); q[OFF(row,2,NExt,3)]=gl64_t(uint64_t(0));")
        kernels.append(f"""__global__ void gen_{sym}_c{chunk_idx}(const StepsParams* __restrict__ P, gl64_t* __restrict__ q, gl64_t* __restrict__ scratch,
    uint64_t NExt, uint64_t tileBase, uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  const uint64_t MASK = NExt-1; const uint64_t WAVE = (uint64_t)gridDim.x*blockDim.x;
  const uint64_t lo_ = blockIdx.x*blockDim.x + threadIdx.x; const uint64_t row = tileBase + lo_;
  if (row >= NExt) return;
  const gl64_t* __restrict__ aux=(const gl64_t*)P->aux_trace; const gl64_t* __restrict__ cst=(const gl64_t*)P->pConstPolsExtendedTreeAddress;
  const gl64_t* __restrict__ ch=(const gl64_t*)P->challenges; const gl64_t* __restrict__ av=(const gl64_t*)P->airValues;
  const gl64_t* __restrict__ agv=(const gl64_t*)P->airgroupValues; const gl64_t* __restrict__ pub=(const gl64_t*)P->publicInputs;
{chr(10).join(lines)}
}}""")
    # Split into one TU per chunk kernel + a thin launcher TU, so the (often 100+) chunk kernels
    # compile in PARALLEL (separate .o) instead of one giant serial nvcc. The kernel bodies are
    # byte-identical to the monolithic form, so the computed result is unchanged.
    files = [(f"gen_{sym}.cu", launcher_tu(sym, n_chunks, total_slots))]
    for tu, lo in enumerate(range(0, n_chunks, CHUNKS_PER_TU)):
        idxs = range(lo, min(lo + CHUNKS_PER_TU, n_chunks))
        files.append((f"gen_{sym}_c{tu}.cu", chunk_tu(sym, idxs, kernels)))
    return files, total_slots

# The fixed C-ABI the loader dlsym's from each AIR's .exps.so (sibling of the AIR's .bin): exps_launch
# delegates to this AIR's launch_gen_<sym>; exps_min_scratch reports the single-block (grid=1) scratch
# need = slots*block (the loader bails to the interpreter if the tmp/destVals region won't fit it;
# 0 => unchunked, no scratch). Shared by the single-kernel TU and the chunked launcher TU.
def c_abi_exports(sym, n_slots):
    return f"""extern "C" void exps_launch(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
    launch_gen_{sym}(d_params, q, scratch, scratchElems, NExt, off_cm1, off_cm2, off_cm3, off_zi, stream);
}}
extern "C" unsigned long long exps_min_scratch() {{ return {n_slots} * {GEN_BLK}ull; }}"""

# Small-expression path: kernel + launcher + C-ABI exports in ONE self-contained TU (no chunking, no scratch).
def single_kernel_tu(sym, kernels, launcher_body, n_slots):
    body = "\n\n".join(kernels)
    return f"""// AUTO-GENERATED Q kernel for {sym} (single kernel, no scratch)
#include "gen_common.cuh"
#define OFF(r,c,nr,nc) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc))
{body}
void launch_gen_{sym}(StepsParams* d_params, gl64_t* q, gl64_t* scratch, uint64_t scratchElems, uint64_t NExt,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi, cudaStream_t stream) {{
{launcher_body}
}}
#undef OFF
{c_abi_exports(sym, n_slots)}
"""

# ---------- chunked emission: batches of chunk kernels per TU + a launcher TU (parallel-compilable) --
def chunk_tu(sym, idxs, kernels):
    # a batch of chunk kernels in ONE TU (amortizes the ~4s gen_common.cuh header parse over the batch),
    # each followed by a C-ABI host wrapper that performs its <<<>>> launch so the launcher TU can call
    # it across TUs without -rdc / device linking.
    parts = []
    for i in idxs:
        parts.append(kernels[i])
        parts.append(f"""extern "C" void run_{sym}_c{i}(uint64_t grid, uint64_t blk, cudaStream_t stream, StepsParams* d_params,
    gl64_t* q, gl64_t* scratch, uint64_t NExt, uint64_t base,
    uint64_t off_cm1, uint64_t off_cm2, uint64_t off_cm3, uint64_t off_zi) {{
  gen_{sym}_c{i}<<<grid,blk,0,stream>>>(d_params,q,scratch,NExt,base,off_cm1,off_cm2,off_cm3,off_zi);
}}""")
    return f"""// AUTO-GENERATED Q chunk kernels {idxs.start}..{idxs.stop-1} for {sym}
#include "gen_common.cuh"
#define OFF(r,c,nr,nc) getBufferOffset((uint64_t)(r),(uint64_t)(c),(uint64_t)(nr),(uint64_t)(nc))
{chr(10).join(parts)}
#undef OFF
"""

def launcher_tu(sym, n_chunks, total_slots):
    decls = "\n".join(f'extern "C" void run_{sym}_c{i}(uint64_t, uint64_t, cudaStream_t, StepsParams*, '
                      f'gl64_t*, gl64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);'
                      for i in range(n_chunks))
    calls = "\n".join(f"    run_{sym}_c{i}(grid, BLK, stream, d_params, q, scratch, NExt, base, "
                      f"off_cm1, off_cm2, off_cm3, off_zi);" for i in range(n_chunks))
    return f"""// AUTO-GENERATED Q launcher for {sym} (cross-boundary temps={total_slots}, {n_chunks} chunks)
#include "gen_common.cuh"
{decls}
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
{calls}
  }}
}}
{c_abi_exports(sym, total_slots)}
"""

# ---------- driver ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    common = """#pragma once
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
"""
    open(os.path.join(OUTDIR, "gen_common.cuh"), "w").write(common)

    generated, skipped = [], []
    slots_by_sym = {}   # sym -> slots, for the syms whose .cu was generated OK
    max_scratch = 0

    # ---- Phase 1: discovery -----------------------------------------------------------------------
    # Walk the provingKey tree once (every *.starkinfo.json + its sibling *.expressionsinfo.json) and
    # build TWO lists with intentionally different lengths:
    #   * candidates -- the UNIQUE kernels to actually generate, one per distinct `sym` (= circuit).
    #   * placements -- EVERY codegen-eligible AIR dir, one entry each, even when several share a sym.
    # Why they differ: one circuit can live in several AIR dirs (the shared compressors; recursive1
    # reuses recursive2's setup). We want to *generate* each kernel once but still drop a `.so` into
    # *every* dir that uses it, so each AIR loads its own sibling `.so`. So we dedup candidates by
    # `sym` (via `seen`) while leaving placements un-deduped. An AIR is dropped here -- i.e. it stays
    # on the bytecode interpreter -- if it lacks an expressionsinfo file, has unparseable JSON, has no
    # cExpId, or its Q has more than CAP ops. Phases 2-3 below consume `candidates`; `placements`
    # is only used at the end to write gen.log (one line per .so destination).
    candidates = []   # unique kernels: (stark_info, expr_info, sym, nBits, c_exp_id, name, n_ops, base)
    placements = []   # every codegen-eligible AIR: (name, base, sym)
    seen = set()      # syms already queued in `candidates` (dedup key);
    for stark_info_path in sorted(glob.glob(os.path.join(AIRS_DIR, "**/*.starkinfo.json"), recursive=True)):
        air_dir = os.path.dirname(stark_info_path)
        base = os.path.basename(stark_info_path).replace(".starkinfo.json", "")
        expr_info_path = os.path.join(air_dir, base + ".expressionsinfo.json")
        if not os.path.exists(expr_info_path): continue
        try: stark_info = json.load(open(stark_info_path)); expr_info = json.load(open(expr_info_path))
        except Exception: continue
        if "cExpId" not in stark_info or "expressionsCode" not in expr_info: continue
        c_exp_id = stark_info["cExpId"]
        if c_exp_id not in {e["expId"] for e in expr_info["expressionsCode"]}: continue
        n_ops = len({e["expId"]: e for e in expr_info["expressionsCode"]}[c_exp_id]["code"])
        nBits = stark_info["starkStruct"]["nBits"]
        name = os.path.relpath(air_dir, AIRS_DIR)
        if n_ops > CAP: skipped.append((name, f"{n_ops} ops > CAP")); continue
        sym = f"a{stark_info['airgroupId']}_{stark_info['airId']}_b{nBits}_e{c_exp_id}"
        placements.append((name, base, sym))
        if sym in seen: continue              # already queued for generation; this dir reuses its .cu
        seen.add(sym)
        candidates.append((stark_info, expr_info, sym, nBits, c_exp_id, name, n_ops, base))

    # ---- Phase 2: auto-tune the no-spill chunk size, per AIR, in parallel -------------------------
    # Output: chunk_map[sym] -> how to chunk that kernel in Phase 3. Three possible values:
    #   * an int      -- the largest chunk size (#ops) that compiled with zero register spill,
    #   * None        -- still spills even at CHUNK_MIN (give up; Phase 3 drops it -> interpreter),
    #   * "UNHANDLED" -- tune_chunk hit an operand type the generator doesn't support.
    # Each candidate is probed by tune_chunk (emit per-chunk TUs -> nvcc -> cuobjdump STACK, bisecting
    # the chunk down until STACK=0; huge AIRs start at BIG_START_CHUNK). Parallelism is via the GIL-free
    # `nvcc` SUBPROCESSes, on TWO pools: the orchestration pool runs one _tune per AIR, and each _tune
    # submits its TUs' compiles to the shared, nproc-bounded cc_pool -- so every chunk of every AIR
    # compiles concurrently with total nvcc <= nproc (no oversubscription; separate pools => no deadlock).
    # If AUTOTUNE is off this whole block is skipped: chunk_map stays empty and Phase 3 emits every
    # kernel at the fixed CHUNK_FIXED size instead (a guess, not spill-verified -- only CAP guards it).
    chunk_map = {}
    if AUTOTUNE:
        from concurrent.futures import ThreadPoolExecutor
        nproc = os.cpu_count() or 4
        # Two pools. cc_pool (bounded at nproc) runs ALL the nvcc compiles across every AIR's chunk TUs,
        # so total concurrent nvcc <= nproc (no oversubscription). The orch threads just drive each AIR's
        # bisection and block on cc_pool futures -- a SEPARATE pool, so the nested submits can't deadlock.
        cc_pool = ThreadPoolExecutor(max_workers=nproc)
        def _tune(cand):
            stark_info, expr_info, sym, nBits, c_exp_id, name, n_ops, base = cand
            try: return sym, tune_chunk(stark_info, expr_info, sym, n_ops, common, cc_pool)
            except NotImplementedError: return sym, "UNHANDLED"
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as orch:
                for sym, ck in orch.map(_tune, candidates): chunk_map[sym] = ck
        finally:
            cc_pool.shutdown(wait=True)

    # ---- Phase 3: emit the .cu source and write it to disk ----------------------------------------
    # For each candidate, pick the chunk size, then emit_air returns (files, slots): `files` is the list
    # of (name, text) TUs to write (1 TU for a single kernel; a launcher TU + N chunk TUs when chunked),
    # `slots` is the cross-chunk cut width (0 = fit as one kernel, no scratch).
    #   AUTOTUNE on  -> chunk_map[sym]: an int is the no-spill size (emit with force_chunk=it);
    #                   None / "UNHANDLED" mean Phase 2 gave up -> skip (record why; stays on interp).
    #   AUTOTUNE off -> emit_air with no force_chunk (the fixed CHUNK_FIXED); an unsupported operand
    #                   raises NotImplementedError, caught here as a skip.
    # Each emitted kernel produces: the gen_<sym>.cu file, slots_by_sym[sym] (-> gen.log in Phase 4),
    # a `generated` summary row, and updates max_scratch (informational, for the printout).
    for (stark_info, expr_info, sym, nBits, c_exp_id, name, n_ops, base) in candidates:
        try:
            if AUTOTUNE:
                ck = chunk_map.get(sym)
                if ck == "UNHANDLED": skipped.append((name, "unhandled operand")); continue
                if ck is None: skipped.append((name, f"{n_ops} ops: still spills at CHUNK_MIN")); continue
                files, slots = emit_air(stark_info, expr_info, sym, force_chunk=ck)
            else:
                files, slots = emit_air(stark_info, expr_info, sym)
        except NotImplementedError as ex:
            skipped.append((name, f"unhandled operand {ex}")); continue
        if slots > SLOTS_CAP: skipped.append((name, f"slots {slots} > SLOTS_CAP (wide cut)")); continue
        for fn, txt in files: open(os.path.join(OUTDIR, fn), "w").write(txt)
        NExt = 1 << stark_info["starkStruct"]["nBitsExt"]
        max_scratch = max(max_scratch, slots * NExt)
        slots_by_sym[sym] = slots
        generated.append((stark_info["airgroupId"], stark_info["airId"], nBits, c_exp_id, sym, name, n_ops, slots, base))

    # ---- Phase 4: write gen.log + print the summary -----------------------------------------------
    # gen.log is the manifest build_exps.sh reads: one TAB-separated line per buildable AIR PLACEMENT
    # = "<reldir>\t<base>\t<sym>\t<slots>". `placed` is the Phase 1 split in reverse: it filters
    # placements down to syms that actually generated (dropping anything Phase 3 skipped -> those AIRs
    # stay on the interpreter), then re-expands each deduped kernel back across every placement, so a
    # duplicate-sym AIR gets its own line. build_exps.sh links each into <provingKey>/<reldir>/
    # <base>.exps.so (next to that AIR's .bin) -- hence the "N kernels -> M .exps.so" header (M >= N,
    # M-N = duplicate placements). The rest is just the human-readable summary: generated, then skipped.
    placed = [(name, base, sym) for (name, base, sym) in placements if sym in slots_by_sym]
    log = "".join(f"{name}\t{base}\t{sym}\t{slots_by_sym[sym]}\n" for (name, base, sym) in placed)
    open(os.path.join(OUTDIR, "gen.log"), "w").write(log)
    chunk_info = "" if AUTOTUNE else f", chunk={CHUNK_FIXED}"
    print(f"generated {len(generated)} kernels -> {len(placed)} per-AIR .exps.so (CAP={CAP}{chunk_info}, max scratch {max_scratch*8/1e6:.0f}MB):")
    for airgroup_id, air_id, n_bits, c_exp_id, sym, name, n_ops, slots, base in generated:
        print(f"  {name:40s} {base}.exps.so  nBits={n_bits} cExp={c_exp_id} ops={n_ops} {'CHUNKED slots='+str(slots) if slots else 'single'}")
    for name, why in skipped: print(f"  SKIP {name:38s} {why}")

main()
