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

// Generic (non-Q) expression kernels 
// Store modes — must match exps_expr_store in the emitted gen_common.cuh.
enum ExprMode : uint64_t {
    EXPR_MODE_WRITE = 0,      // dest  = expr
    EXPR_MODE_MUL_INV = 1,    // dest *= expr^{-1}
    EXPR_MODE_WRITE_INV = 2,  // dest  = scalar * expr^{-1}
};
typedef int (*ExprCoveredFn)(unsigned long long expId);
typedef int (*ExprLaunchFn)(unsigned long long expId, StepsParams*, gl64_t* dest,
                                unsigned long long N, unsigned long long destDomain,
                                unsigned long long off_cm1, unsigned long long off_cm2, unsigned long long off_cm3,
                                unsigned long long mode, unsigned long long scalar,
                                unsigned long long stagePos, unsigned long long stageCols,
                                unsigned long long destDim, unsigned int destExpr, cudaStream_t);
typedef int (*ExprPairLaunchFn)(unsigned long long, unsigned long long, StepsParams*, gl64_t*, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned int, cudaStream_t);
// One AIR's resolved generated kernels. Two families (see expressions_gpu.cuh):
//   lib        : dlopen handle for the AIR's .exps.so (kept open for the AIR's lifetime; dlclose'd in dtor)
//   qLaunch    : the Q launcher (`exps_launch`) -- chunked kernels over the extended domain
//   qMinScratch: scratch elements one Q wave needs (`exps_min_scratch`; 0 if unchunked)
//   exprCovered/exprLaunch: per-expId trace-domain kernels for the non-Q expressions
//                (`exps_expr_covered` / `exps_launch_expr`; optional -- older .so lack them)
struct ExpsKernel {
    void* lib = nullptr;
    ExpsQLaunchFn qLaunch = nullptr;
    uint64_t qMinScratch = 0;
    ExprCoveredFn exprCovered = nullptr;
    ExprLaunchFn exprLaunch = nullptr;
    ExprPairLaunchFn exprPairLaunch = nullptr;
};

// The library sits next to the AIR's .bin with the .exps.so suffix: swap a trailing ".bin" -> ".exps.so".
inline std::string expsSoPath(const std::string& binFile) {
    const std::string ext = ".bin";
    if (binFile.size() < ext.size() || binFile.compare(binFile.size()-ext.size(), ext.size(), ext) != 0)
        return "";
    return binFile.substr(0, binFile.size()-ext.size()) + ".exps.so";
}

// Scratch memory available in the aux_trace tmp...destVals region
inline uint64_t expsScratchAvail(SetupCtx& sc) {
    uint64_t o1 = sc.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    uint64_t od = sc.starkInfo.mapOffsets[std::make_pair("destVals", false)];
    return (od - o1) + 2ull * 3 * sc.starkInfo.nrowsPack * sc.starkInfo.maxNBlocks;
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
    r.lib = h; r.qLaunch = qLaunch; r.qMinScratch = (uint64_t)minf();
    r.exprCovered = (ExprCoveredFn)dlsym(h, "exps_expr_covered");
    r.exprLaunch = (ExprLaunchFn)dlsym(h, "exps_launch_expr");
    r.exprPairLaunch = (ExprPairLaunchFn)dlsym(h, "exps_launch_expr_pair");
    if ((r.exprCovered == nullptr) != (r.exprLaunch == nullptr)) { r.exprCovered = nullptr; r.exprLaunch = nullptr; }
    return r;
}

inline void expsClose(void* lib) { if (lib) dlclose(lib); }

// Warm an AIR's generated kernels at load: a no-op Q launch (NExt=0) plus one 1-row expr
// launch force the lazy module load + local-mem pool allocations NOW, while VRAM is free.
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
        // One 1-row expr launch claims the expr kernels' pool too (destExpr=1 -> dummy[0]).
        if (ek.exprCovered != nullptr && ek.exprLaunch != nullptr) {
            for (const auto& kv : sc.expressionsBin.expressionsInfo) {
                if (ek.exprCovered(kv.first)) {
                    ek.exprLaunch(kv.first, d_warm, dummy, /*N=*/1, /*destDomain=*/1,
                                  0, 0, 0, EXPR_MODE_WRITE, 0, /*stagePos=*/0, /*stageCols=*/1,
                                  /*destDim=*/1, /*destExpr=*/1, (cudaStream_t)0);
                    break;
                }
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
// the kernel's single-block scratch requirement doesn't fit the tmp...destVals region.
inline bool tryLaunchExpsQ(SetupCtx& sc, ExpsQLaunchFn fn, uint64_t minScratch,
                          StepsParams* d_params, gl64_t* d_q, cudaStream_t stream,
                          bool* launchVerified = nullptr) {
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
    fn(d_params, d_q, os, scratchAvail, NExt, o1, o2, o3, oz, stream);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "[exps] Q kernel launch failed (%s); falling back to the interpreter\n",
                cudaGetErrorString(launchErr));
        return false;
    }
    if (verify) {
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        uint64_t pw0 = 0;
        CHECKCUDAERR(cudaMemcpy(&pw0, os, sizeof(pw0), cudaMemcpyDeviceToHost));
        if (pw0 == 0) {
            fprintf(stderr, "[exps] Q kernel launch did not execute (pw[0] still 0, likely lazy "
                            "module-load OOM); falling back to the interpreter\n");
            return false;
        }
        *launchVerified = true;
    }
    return true;
}
