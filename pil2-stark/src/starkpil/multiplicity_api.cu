// C entry points for the GPU-multiplicities path. Separate from starks_api.cu, whose include
// chain defines non-inline GPU symbols that would be duplicated at link time.
#include <cstdint>
#include "multiplicity.hpp"
#include "multiplicity_decoders.hpp"
#include "multiplicity.cuh"
#include "multiplicity_kernel.cuh"
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
uint64_t mul_air_device_owned(uint64_t airKey) {
    return mul_air_fully_owned(airKey) ? 1 : 0;
}

// Ordering point for the device export: once this returns MUL_SYNC_OK every instance has launched
// its scatter, so the table's own commit sees a complete accumulator.
uint64_t mul_sync_commits(uint64_t expectedCommits) {
    if (mulDecoders().empty()) return MUL_SYNC_OK;
    if (const MulSyncStatus st = mul_check_commits(expectedCommits); st != MUL_SYNC_OK) return st;
    const uint64_t oob = mul_oob_report() + mulCpuOobTotal().load(std::memory_order_relaxed);
    return oob != 0 ? MUL_SYNC_OOB : MUL_SYNC_OK;
}

// Fold the prover-owned spans into the caller's accumulator, once per proof, after
// mul_sync_commits. `hostAcc` is not retained.
void mul_fold(uint64_t airKey, uint64_t *hostAcc) {
    if (mulDecoders().empty() || hostAcc == nullptr) return;
    // One of the two accumulators is always empty, so folding both keeps the caller backend-blind.
    mul_fold_air(airKey, hostAcc);
    mul_cpu_fold(airKey, hostAcc);
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

    // Coverage, then GPU memory per device (it competes with that device's prover arena). Once,
    // though every reset gets here.
    static std::once_flag logged;
    std::call_once(logged, [&] {
        mul_log_coverage();
        for (int id : gpuIds) {
            const uint64_t bytes = mul_gpu_resident_bytes(id);
            if (bytes != 0)
                zklog.info("Multiplicity: " + to_string(bytes / (1024 * 1024))
                           + " MB GPU-resident on gpu " + to_string(id));
        }
    });
    mul_alloc_fold_staging();
    // The prover streams' events (the verify path scatters there) and every plan's device copy.
    if (!mulDecoders().empty()) {
        int prev = 0;
        CHECKCUDAERR(cudaGetDevice(&prev));
        for (uint32_t i = 0; i < d_buffers->n_total_streams; ++i)
            mul_warm_stream_event((int)d_buffers->streamsData[i].gpuId, d_buffers->streamsData[i].stream);
        // Basic setups only: mulPlanFor caches by (airgroup, air), which a recursive setup shares.
        for (auto& air : d_buffers->air_instances) {
            auto basic = air.second.find("basic");
            if (basic == air.second.end()) continue;
            for (uint32_t gl = 0; gl < basic->second.size() && gl < d_buffers->n_gpus; ++gl) {
                AirInstanceInfo* aii = basic->second[gl];
                if (aii == nullptr || aii->setupCtx == nullptr) continue;
                // The transpose writes one word per column: a packed table air cannot be exported.
                if (aii->is_packed && mul_air_fully_owned(mulAirKey(air.first.first, air.first.second))) {
                    zklog.error("multiplicity: device-owned table air " + to_string(air.first.first) + "/"
                                + to_string(air.first.second) + " is packed");
                    exitProcess();
                }
                const MulPlan& p = mulPlanFor(*aii->setupCtx, air.first.first, air.first.second);
                if (p.jobs.empty()) continue;
                const int gpu = (int)d_buffers->my_gpu_ids[gl];
                CHECKCUDAERR(cudaSetDevice(gpu));
                mulPlanDevice(p, air.first.first, air.first.second, gpu);
            }
        }
        CHECKCUDAERR(cudaSetDevice(prev));
    }
    // After the accumulators: the warm-up builds the scatter programs that point into them.
    stream_commit_warmup_gpu(d_buffers_);
}

void mul_reset() {
    mul_reset_all();
    mul_cpu_reset();
}

} // extern "C"
