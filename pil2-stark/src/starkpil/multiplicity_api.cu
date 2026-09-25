// C entry points for the GPU-multiplicities path. Separate from starks_api.cu, whose include
// chain defines non-inline GPU symbols that would be duplicated at link time.
#include <cstdint>
#include "multiplicity.hpp"
#include "multiplicity_decoders.hpp"
#include "multiplicity.cuh"
#include "multiplicity_cpu.hpp"
#include "zklog.hpp"
#include "goldilocks_tooling.cuh"
#include <vector>
#include <algorithm>

using namespace std;

void stream_commit_warmup_gpu(void *d_buffers_);   // starks_api.cu

extern "C" {

// Off unless no cross-rank reduction is needed: the device accumulator holds only this rank's share.
void mul_set_device_export(uint64_t enabled) {
    mulDeviceExportEnabled() = (enabled != 0);
}

// True when the device produces this air's whole cm1, so the host must not build its trace.
uint64_t mul_air_device_owned(uint64_t airId) {
    return mul_air_fully_owned(airId) ? 1 : 0;
}

// Ordering point for the device export: once this returns every instance has launched its
// scatter, so the table's own commit sees a complete accumulator.
void mul_sync_commits(uint64_t expectedCommits) {
    if (mulDecoders().empty()) return;
    if (!mul_await_commits(expectedCommits)) exitProcess();
    mul_oob_report();
}

// Fold the prover-owned spans into the caller's accumulator, once per proof, after
// mul_sync_commits. `hostAcc` is not retained.
void mul_fold(uint64_t airId, uint64_t *hostAcc) {
    if (mulDecoders().empty() || hostAcc == nullptr) return;
    // One of the two accumulators is always empty, so folding both keeps the caller backend-blind.
    mul_fold_air(airId, hostAcc);
    mul_cpu_fold(airId, hostAcc);
}

// Allocate device mirrors for airs hosting a migrated table. Idempotent; no-op without decoders.
void mul_alloc(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    std::vector<int> gpuIds(d_buffers->n_gpus);
    for (uint32_t g = 0; g < d_buffers->n_gpus; ++g) gpuIds[g] = (int)d_buffers->my_gpu_ids[g];
    mul_alloc_devices(gpuIds.data(), (int)gpuIds.size());
    for (int id : gpuIds) { mul_alloc_oob(id); mul_alloc_maps(id); }
    if (gpuIds.size() > 1 && mulDeviceExportEnabled()) mul_alloc_peers(gpuIds);

    // Coverage, then GPU memory per device (it competes with that device's prover arena).
    mul_log_coverage();
    for (int id : gpuIds) {
        const uint64_t bytes = mul_gpu_resident_bytes(id);
        if (bytes != 0)
            zklog.info("Multiplicity: " + to_string(bytes / (1024 * 1024))
                       + " MB GPU-resident on gpu " + to_string(id));
    }
    // After the accumulators: the warm-up builds the scatter programs that point into them.
    stream_commit_warmup_gpu(d_buffers_);
}

void mul_reset() {
    mul_reset_all();
    mul_cpu_reset();
}

} // extern "C"
