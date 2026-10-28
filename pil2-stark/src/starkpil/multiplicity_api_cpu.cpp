// C entry points for the prover-side multiplicities on the CPU backend.
//
// The GPU copy is multiplicity_api.cu; this file is empty in the GPU library. Same entry points on
// both so the Rust side stays backend-blind.
#ifndef __USE_CUDA__

#include <cstdint>
#include <vector>
#include <algorithm>
#include "multiplicity.hpp"
#include "multiplicity_decoders.hpp"
#include "multiplicity_cpu.hpp"
#include "zklog.hpp"

using namespace std;

extern "C" {

// No device to export from: accepted and ignored so the caller need not know the backend.
void mul_set_device_export(uint64_t enabled) {
    (void)enabled;
}

// Never: this backend has no device accumulator, so every air's trace is built on the host.
uint64_t mul_air_device_owned(uint64_t airKey) {
    (void)airKey;
    return 0;
}

// Called once every instance has counted (ProofMan waits on mul_commit_count).
uint64_t mul_sync_commits(uint64_t expectedCommits) {
    if (mulDecoders().empty()) return MUL_SYNC_OK;
    if (const MulSyncStatus st = mul_check_commits(expectedCommits); st != MUL_SYNC_OK) return st;
    return mulCpuOobTotal().load(std::memory_order_relaxed) != 0 ? MUL_SYNC_OOB : MUL_SYNC_OK;
}

// After mul_sync_commits.
void mul_fold(uint64_t airKey, uint64_t *hostAcc) {
    if (mulDecoders().empty() || hostAcc == nullptr) return;
    mul_cpu_fold(airKey, hostAcc);
}

void mul_alloc(void *d_buffers_) {
    (void)d_buffers_;
    mul_cpu_alloc();
    mul_log_coverage();
}

void mul_reset() {
    mul_reset_commits();
    mul_cpu_reset();
}

} // extern "C"

#endif // __USE_CUDA__
