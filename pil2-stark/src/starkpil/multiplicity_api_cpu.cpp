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
uint64_t mul_air_device_owned(uint64_t airId) {
    (void)airId;
    return 0;
}

// The GPU backend's ordering point. A no-op here (the CPU scatter is inline), kept for a uniform
// call sequence.
void mul_sync_commits(uint64_t expectedCommits) {
    if (mulDecoders().empty()) return;
    if (!mul_await_commits(expectedCommits)) exitProcess();
}

// No commit barrier: the CPU scatter runs inline, so counts are in when the commit returns.
// `expectedCommits` is ignored.
void mul_fold(uint64_t airId, uint64_t *hostAcc, uint64_t expectedCommits) {
    if (mulDecoders().empty() || hostAcc == nullptr) return;
    // The scatter runs on the instance workers, which the caller does not join before building the
    // table trace, so wait for every instance to have counted before reading the accumulator.
    if (!mul_await_commits(expectedCommits)) exitProcess();
    mul_cpu_fold(airId, hostAcc);
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
