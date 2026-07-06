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

// Must match the launcher signature emitted by the codegen
// (extern "C" exps_launch in setup/exps-codegen/src/emit.rs).
typedef void (*ExpsKernelFn)(StepsParams*, gl64_t*, gl64_t*, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t, uint64_t, cudaStream_t);
// One AIR's resolved expression kernel:
//   lib       : dlopen handle for the AIR's .exps.so (kept open for the AIR's lifetime; dlclose'd in dtor)
//   fn        : the .so's C-ABI host launcher (exps_launch) that issues this AIR's kernel launch(es)
//   minScratch: scratch elements a single-block launch needs (the .so reports it via exps_min_scratch;
//               0 if unchunked).
struct ExpsKernel {
    void* lib = nullptr;
    ExpsKernelFn fn = nullptr;
    uint64_t minScratch = 0;
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
    auto launch = (ExpsKernelFn)dlsym(h, "exps_launch");
    auto minf = (unsigned long long (*)())dlsym(h, "exps_min_scratch");
    if (!launch || !minf) { fprintf(stderr, "[EXPS] missing symbols in %s\n", path.c_str()); dlclose(h); return r; }
    r.lib = h; r.fn = launch; r.minScratch = (uint64_t)minf();
    return r;
}

inline void expsClose(void* lib) { if (lib) dlclose(lib); }

// Launch the AIR's pre-resolved kernel. Returns false (caller falls back to the interpreter) if the
// kernel's single-block scratch requirement doesn't fit the tmp...destVals region.
inline bool tryLaunchExps(SetupCtx& sc, ExpsKernelFn fn, uint64_t minScratch,
                          StepsParams* d_params, gl64_t* d_q, cudaStream_t stream) {
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
    fn(d_params, d_q, os, scratchAvail, NExt, o1, o2, o3, oz, stream);
    return true;
}
