#ifndef PROOFMAN_SUMCHECK_CUH
#define PROOFMAN_SUMCHECK_CUH

// Diagnostic per-stage device-buffer checksum, gated at runtime by PROOFMAN_SUMCHECK=1.
// Hashes a device buffer and prints inst/airgroup/air/stage/cksum so two runs can be
// diffed stage-by-stage to localize a divergence. No-op (one branch) when the env var
// is unset, so probe calls can stay compiled in.
//
// Usage: set the context once at the top of an entry point, then probe with the macro:
//   proofman_sumcheck_set_context(instanceId, airgroupId, airId);
//   PROOFMAN_SUMCHECK("stage_label", d_ptr, n_u64, stream);
// Cross-TU probes (e.g. starks_gpu.cu, reached under genProof) inherit the context set
// by their caller -- it is thread-local.

#include <cstdint>
#include <cuda_runtime.h>

// Per-thread probe context, so call sites deep in the pipeline need not thread the ids
// through. Set by whichever entry point owns the current proof (see set_context).
extern thread_local uint64_t g_sumcheck_inst;
extern thread_local uint64_t g_sumcheck_airgroup;
extern thread_local uint64_t g_sumcheck_air;

inline void proofman_sumcheck_set_context(uint64_t inst, uint64_t airgroup, uint64_t air) {
    g_sumcheck_inst = inst;
    g_sumcheck_airgroup = airgroup;
    g_sumcheck_air = air;
}

// `stage` is a printf-style format string; any trailing args fill it in. The label is only
// formatted when the probe is enabled (the gate is checked before vsnprintf), so a disabled
// probe costs nothing beyond evaluating the args. Format-checked by the compiler.
void proofman_sumcheck_impl(const void *d_ptr, uint64_t n_u64, cudaStream_t stream,
                            uint64_t instanceId, uint64_t airgroupId, uint64_t airId,
                            const char *stage, ...) __attribute__((format(printf, 7, 8)));

// Terse probe: pulls inst/airgroup/air from the thread-local context. Usage:
//   PROOFMAN_SUMCHECK("cm1_before", ptr, n, stream);            // literal label
//   PROOFMAN_SUMCHECK("lde_cm%u", ptr, n, stream, (unsigned)step);  // formatted label
#define PROOFMAN_SUMCHECK(stage, d_ptr, n_u64, stream, ...) \
    proofman_sumcheck_impl((d_ptr), (n_u64), (stream), \
        g_sumcheck_inst, g_sumcheck_airgroup, g_sumcheck_air, (stage), ##__VA_ARGS__)

#endif // PROOFMAN_SUMCHECK_CUH
