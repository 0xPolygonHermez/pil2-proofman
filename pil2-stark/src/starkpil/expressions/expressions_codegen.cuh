#pragma once

// Per-AIR loader for generated expression kernels (cubin-in-setup).
//
// The setup step compiles each AIR's generated expression kernel(s) into its own self-contained
// shared library placed in <air-dir>/<basename>.exps.so

#include "steps.hpp"
#include "setup_ctx.hpp"
#include <dlfcn.h>
#include <sys/stat.h>
#include <string>
#include <cstdlib>
#include <cstdio>

// Q launcher: must match the signature emitted by the codegen
// (extern "C" exps_launch in setup/exps-codegen/src/emit.rs).
typedef void (*ExpsQLaunchFn)(StepsParams*, gl64_t*, gl64_t*, uint64_t, uint64_t,
                              uint64_t, uint64_t, uint64_t, uint64_t, cudaStream_t);
// Same, at every 2^qs-th extended row only (exps_launch_step; absent from older .so).
typedef void (*ExpsQLaunchStepFn)(StepsParams*, gl64_t*, gl64_t*, uint64_t, uint64_t,
                                  uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, cudaStream_t);

// Generic (non-Q) expression kernels 
// Store modes — must match exps_expr_store in the emitted gen_common.cuh.
enum ExprMode : uint64_t {
    EXPR_MODE_WRITE = 0,      // dest  = expr
    EXPR_MODE_MUL_INV = 1,    // dest *= expr^{-1}
    EXPR_MODE_WRITE_INV = 2,  // dest  = scalar * expr^{-1}
};
typedef int (*ExprCoveredFn)(unsigned long long expId);
// The generic (non-Q) expression launchers, `exps_generic_launch` / `exps_generic_launch_pair`: the
// straight-line kernels and the chunked ones (register-bounded, cut temps parked in the expression
// scratch given as element offset inside aux_trace + size, so no kernel in the .so keeps local
// memory). Return 0 when nothing was launched (scratch cannot hold one wave); the caller then
// falls back to the interpreter.
typedef int (*ExprLaunchFn)(unsigned long long expId, StepsParams*, gl64_t* dest,
                                unsigned long long N, unsigned long long destDomain,
                                unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3,
                                unsigned long long mode, unsigned long long scalar,
                                unsigned long long stagePos, unsigned long long stageCols,
                                unsigned long long destDim, unsigned int destExpr,
                                unsigned long long scratchOff, unsigned long long scratchElems, cudaStream_t);
typedef int (*ExprPairLaunchFn)(unsigned long long, unsigned long long, StepsParams*, gl64_t*, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned int, unsigned long long, unsigned long long, cudaStream_t);
// One AIR's resolved generated kernels. Two families (see expressions_gpu.cuh):
//   lib        : dlopen handle for the AIR's .exps.so (kept open for the AIR's lifetime; dlclose'd in dtor)
//   qLaunch    : the Q launcher (`exps_launch`) -- chunked kernels over the extended domain
//   qMinScratch: scratch elements one Q wave needs (`exps_min_scratch`; 0 if unchunked)
//   exprCovered/exprLaunch/exprPairLaunch: per-expId trace-domain kernels for the non-Q
//                expressions (`exps_generic_covered` / `exps_generic_launch` / `exps_generic_launch_pair`; optional)
struct ExpsKernel {
    void* lib = nullptr;
    ExpsQLaunchFn qLaunch = nullptr;
    ExpsQLaunchStepFn qLaunchStep = nullptr;
    uint64_t qMinScratch = 0;
    // Generic expression kernels (`exps_generic_*`): null when the library has none, or when the
    // memory map does not prove the expression scratch dead in stage 2 (expsCheckScratchLayout);
    // the prover's interpreter then evaluates them.
    ExprCoveredFn exprCovered = nullptr;
    ExprLaunchFn exprLaunch = nullptr;
    ExprPairLaunchFn exprPairLaunch = nullptr;
    // The expression scratch, `[scratchOff, scratchOff + scratchElems)` of aux_trace: tmp1 up to
    // the end of the buffer, the region Q uses too.
    uint64_t scratchOff = 0;
    uint64_t scratchElems = 0;
};

// Prove, from the memory map alone, that the expression scratch `[tmp1, mem_exps_end)` is
// disjoint from everything the stage-2 expressions read or write inside aux_trace: the small cm
// sections (stages 1..nStages), the values at the head of the buffer, and the q zone where the
// gsum/gprod hints keep their row-major temporaries (`q .. q + 3N`) and accumulation helpers --
// the scratch begins past q + 3NExt + zi, so the whole q+zi zone must end before it. What else
// is live then (constants, custom fixed, the raw trace) lives in other buffers. Returns false,
// with the reason logged, when a layout breaks an assumption; the caller then keeps the
// scratch-using kernels off for that air.
inline bool expsCheckScratchLayout(SetupCtx& sc) {
    auto& si = sc.starkInfo;
    auto off = [&](const char* name, bool ext) -> int64_t {
        auto it = si.mapOffsets.find(std::make_pair(std::string(name), ext));
        return it == si.mapOffsets.end() ? -1 : (int64_t)it->second;
    };
    const uint64_t N = 1ull << si.starkStruct.nBits;
    const uint64_t NExt = 1ull << si.starkStruct.nBitsExt;
    const int64_t tmp1 = off("tmp1", false), end = off("mem_exps_end", false), q = off("q", true), zi = off("zi", true);
    auto fail = [&](const std::string& why) {
        fprintf(stderr, "[EXPS] %s: expression scratch layout check failed (%s); chunked expression kernels stay on the interpreter\n", sc.expressionsBin.binFile.c_str(), why.c_str());
        return false;
    };
    if (tmp1 < 0 || end < 0 || q < 0) return fail("tmp1/mem_exps_end/q missing from the map");
    if (end < tmp1 || (uint64_t)end > si.mapTotalN) return fail("scratch not within tmp1 .. mapTotalN");
    const uint64_t qZoneEnd = zi >= 0 ? (uint64_t)zi + si.ziSize : (uint64_t)q + 3 * NExt;
    if (qZoneEnd > (uint64_t)tmp1) return fail("q/zi zone reaches into the scratch");
    if ((uint64_t)q + 3 * N > (uint64_t)tmp1) return fail("hint temporaries at q reach into the scratch");
    for (uint64_t stage = 1; stage <= si.nStages; ++stage) {
        std::string sec = "cm" + std::to_string(stage);
        int64_t o = off(sec.c_str(), false);
        if (o < 0) continue;
        if ((uint64_t)o + N * si.mapSectionsN[sec] > (uint64_t)tmp1 && (uint64_t)o < (uint64_t)end) return fail(sec + " section overlaps the scratch");
    }
    for (const char* sec : {"airvalues", "airgroupvalues", "publics", "challenge", "evals", "proofvalues"}) {
        int64_t o = off(sec, false);
        if (o >= 0 && (uint64_t)o >= (uint64_t)tmp1 && (uint64_t)o < (uint64_t)end) return fail(std::string(sec) + " inside the scratch");
    }
    return true;
}

// The library sits next to the AIR's .bin with the .exps.so suffix: swap a trailing ".bin" -> ".exps.so".
inline std::string expsSoPath(const std::string& binFile) {
    const std::string ext = ".bin";
    if (binFile.size() < ext.size() || binFile.compare(binFile.size()-ext.size(), ext.size(), ext) != 0)
        return "";
    return binFile.substr(0, binFile.size()-ext.size()) + ".exps.so";
}

// Scratch memory available from tmp1 to the end of the region dead during the quotient
inline uint64_t expsScratchAvail(SetupCtx& sc) {
    uint64_t o1 = sc.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    auto end = sc.starkInfo.mapOffsets.find(std::make_pair("mem_exps_end", false));
    if (end != sc.starkInfo.mapOffsets.end()) return end->second - o1;
    uint64_t od = sc.starkInfo.mapOffsets[std::make_pair("destVals", false)];
    return (od - o1) + 2ull * 3 * sc.starkInfo.nrowsPack * sc.starkInfo.maxNBlocks;
}

// Empty if the library's zerofier reads (baked from the boundaries at gen-exps time) match this starkinfo.
inline std::string expsBoundaryLayoutError(void* lib, const StarkInfo& si) {
    auto fn = (uint64_t (*)())dlsym(lib, "boundary_layout_hash");
    if (fn != nullptr && fn() == boundaryLayoutHash(si.boundaries, si.starkStruct.nBits, si.starkStruct.nBitsExt)) return "";
    return "stale .exps.so: boundary layout changed, rerun gen-exps";
}

// Called once per ExpressionsGPU (in its constructor). Resolve this AIR's expression kernel.
inline ExpsKernel expsOpenForAir(SetupCtx& sc) {
    ExpsKernel r;
    std::string path = expsSoPath(sc.expressionsBin.binFile);
    if (path.empty()) return r;
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return r;
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "[EXPS] dlopen failed: %s\n", dlerror()); return r; }
    auto qLaunch = (ExpsQLaunchFn)dlsym(h, "exps_launch");
    auto minf = (unsigned long long (*)())dlsym(h, "exps_min_scratch");
    if (!qLaunch || !minf) { fprintf(stderr, "[EXPS] missing symbols in %s\n", path.c_str()); dlclose(h); return r; }
    // An .exps.so reads zi with the offsets and masks of the boundaries it was generated from.
    std::string layoutError = expsBoundaryLayoutError(h, sc.starkInfo);
    if (!layoutError.empty()) {
        fprintf(stderr, "[EXPS] %s: %s\n", path.c_str(), layoutError.c_str());
        exitProcess();
    }
    r.lib = h; r.qLaunch = qLaunch; r.qMinScratch = (uint64_t)minf();
    r.qLaunchStep = (ExpsQLaunchStepFn)dlsym(h, "exps_launch_step");
    r.scratchOff = sc.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    r.scratchElems = expsScratchAvail(sc);
    r.exprCovered = (ExprCoveredFn)dlsym(h, "exps_generic_covered");
    r.exprLaunch = (ExprLaunchFn)dlsym(h, "exps_generic_launch");
    r.exprPairLaunch = (ExprPairLaunchFn)dlsym(h, "exps_generic_launch_pair");
    const bool complete = r.exprCovered != nullptr && r.exprLaunch != nullptr && r.exprPairLaunch != nullptr;
    if (!complete && (r.exprCovered != nullptr || r.exprLaunch != nullptr || r.exprPairLaunch != nullptr)) {
        fprintf(stderr, "[EXPS] %s: incomplete exps_generic_* symbols; its generic expression kernels are ignored\n", path.c_str());
    }
    // The generic kernels also need a scratch the map proves dead in stage 2; each chunked
    // launcher then checks its own one-wave fit at launch and returns 0 (-> interpreter), as Q's.
    if (!complete || !expsCheckScratchLayout(sc)) {
        r.exprCovered = nullptr; r.exprLaunch = nullptr; r.exprPairLaunch = nullptr;
    }
    return r;
}

inline void expsClose(void* lib) { if (lib) dlclose(lib); }

// Warm an AIR's generated kernels at load: a no-op Q launch (NExt=0) plus one 1-row expr
// launch force the lazy module load NOW, while VRAM is free (no kernel keeps local memory any
// more, the generator audits that; module code still loads lazily).
// Done lazily mid-proof they can fail SILENTLY (the .so has its own static CUDA runtime)
// and the proof is invalid. Returns false when skipped (VRAM guard).
inline bool expsWarmup(SetupCtx& sc, const ExpsKernel& ek) {
    if (ek.qLaunch == nullptr) return true;  // no generated kernels at all
    // Leave room for the per-air setup allocations that follow (witness_compact ~1GB).
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess || freeB < (1ull << 30)) return false;
    uint64_t words = (ek.qMinScratch < 1024 ? 1024 : ek.qMinScratch) + 4096;
    gl64_t *dummy = nullptr;
    if (cudaMalloc(&dummy, words * sizeof(uint64_t)) != cudaSuccess) return false;
    CHECKCUDAERR(cudaMemset(dummy, 0, words * sizeof(uint64_t)));
    // Every params pointer targets the zeroed dummy; results are garbage and discarded.
    StepsParams warm = {};
    warm.trace = (Goldilocks::Element *)dummy;
    warm.aux_trace = (Goldilocks::Element *)dummy;
    warm.publicInputs = (Goldilocks::Element *)dummy;
    warm.proofValues = (Goldilocks::Element *)dummy;
    warm.challenges = (Goldilocks::Element *)dummy;
    warm.airgroupValues = (Goldilocks::Element *)dummy;
    warm.airValues = (Goldilocks::Element *)dummy;
    warm.evals = (Goldilocks::Element *)dummy;
    warm.xDivXSub = (Goldilocks::Element *)dummy;
    warm.pConstPolsAddress = (Goldilocks::Element *)dummy;
    warm.pConstPolsExtendedTreeAddress = (Goldilocks::Element *)dummy;
    warm.pCustomCommitsFixed = (Goldilocks::Element *)dummy;
    StepsParams *d_warm = nullptr;
    bool warmed = false;
    if (cudaMalloc(&d_warm, sizeof(StepsParams)) == cudaSuccess) {
        CHECKCUDAERR(cudaMemcpy(d_warm, &warm, sizeof(StepsParams), cudaMemcpyHostToDevice));
        ek.qLaunch(d_warm, dummy, dummy, words, /*NExt=*/0, 0, 0, 0, 0, (cudaStream_t)0);
        // One 1-row expr launch (destExpr=1 -> dummy[0]) loads the expr kernels' module too.
        // Its scratch is the dummy itself; a chunked launcher returns 0 if that is short.
        if (ek.exprCovered != nullptr) {
            for (const auto& kv : sc.expressionsBin.expressionsInfo) {
                if (!ek.exprCovered(kv.first)) continue;
                ek.exprLaunch(kv.first, d_warm, dummy, /*N=*/1, /*destDomain=*/1, 0, 0, 0, EXPR_MODE_WRITE, 0,
                              /*stagePos=*/0, /*stageCols=*/1, /*destDim=*/1, /*destExpr=*/1, /*scratchOff=*/0, words, (cudaStream_t)0);
                break;
            }
        }
        cudaStreamSynchronize((cudaStream_t)0);
        cudaFree(d_warm);
        warmed = true;
    }
    cudaFree(dummy);
    return warmed;
}

// Launch the AIR's pre-resolved Q kernel. Returns false (caller falls back to the interpreter) if
// the kernel's one-wave scratch requirement doesn't fit the tmp1 .. mem_exps_end region.
// `stepFn`/`qStepLg` ask for q at every 2^qStepLg-th extended row only; `*qStepLgUsed` reports
// what the launch did (0 without exps_launch_step), and the caller's computeQ must use it.
inline bool tryLaunchExpsQ(SetupCtx& sc, ExpsQLaunchFn fn, uint64_t minScratch,
                          StepsParams* d_params, gl64_t* d_q, cudaStream_t stream,
                          bool* launchVerified = nullptr, ExpsQLaunchStepFn stepFn = nullptr,
                          uint32_t qStepLg = 0, uint32_t* qStepLgUsed = nullptr) {
    uint64_t scratchAvail = expsScratchAvail(sc);
    if (minScratch > scratchAvail) return false;
    uint64_t NExt = 1ull << sc.starkInfo.starkStruct.nBitsExt;
    uint64_t oq = sc.starkInfo.mapOffsets[std::make_pair("q", true)];
    uint64_t otmp = sc.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    gl64_t*  os = d_q - oq + otmp;  // scratch offset == aux_trace (d_q - oq) + offsetTmp1 (otmp)
    uint64_t o1 = sc.starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t o2 = sc.starkInfo.mapOffsets[std::make_pair("cm2", true)];
    uint64_t o3 = sc.starkInfo.mapOffsets[std::make_pair("cm3", true)];
    uint64_t oz = sc.starkInfo.mapOffsets[std::make_pair("zi",  true)];
    // First-call sentinel: pre-zero pw[0]; a successful launch always writes it nonzero
    // (v^0 or a tab value), a silently failed one leaves our 0. Checked once per air.
    const bool verify = launchVerified != nullptr && !*launchVerified && minScratch > 0;
    if (verify) {
        CHECKCUDAERR(cudaMemsetAsync(os, 0, sizeof(uint64_t), stream));
    }
    const uint32_t qs = stepFn != nullptr ? qStepLg : 0;
    if (qs > 0) stepFn(d_params, d_q, os, scratchAvail, NExt, o1, o2, o3, oz, qs, stream);
    else fn(d_params, d_q, os, scratchAvail, NExt, o1, o2, o3, oz, stream);
    if (qStepLgUsed != nullptr) *qStepLgUsed = qs;
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "[exps] Q kernel launch failed (%s); falling back to the interpreter\n",
                cudaGetErrorString(launchErr));
        return false;
    }
    if (verify) {
        // On this stream: the prover streams are blocking, so a plain cudaMemcpy would also
        // wait for every other stream's queued proof.
        uint64_t pw0 = 0;
        CHECKCUDAERR(cudaMemcpyAsync(&pw0, os, sizeof(pw0), cudaMemcpyDeviceToHost, stream));
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        if (pw0 == 0) {
            fprintf(stderr, "[exps] Q kernel launch did not execute (pw[0] still 0, likely lazy "
                            "module-load OOM); falling back to the interpreter\n");
            return false;
        }
        *launchVerified = true;
    }
    return true;
}
