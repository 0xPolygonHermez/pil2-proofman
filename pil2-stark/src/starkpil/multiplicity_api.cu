// C entry points for the GPU-multiplicities path. Its own translation unit rather than part of
// starks_api.cu, whose include chain defines non-inline GPU symbols that would be duplicated at
// link time. The Makefile picks it up via `find ./src/starkpil -name "*.cu"`.
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

extern "C" {

void mul_register_range_tables(const uint64_t *tableIds, const int64_t *biases, uint64_t n) {
    mul_register_range_tables_impl(tableIds, biases, n);
}

// Geometry only, registered once per run by the host binary.
void register_mul_vt(uint64_t airgroupId, uint64_t airId, uint64_t numRows, uint64_t numCols,
                     const uint64_t *tableIds, const uint64_t *accBases, uint64_t nTables) {
    mul_register_vt(airgroupId, airId, numRows, numCols, tableIds, accBases, nTables);
    mul_materialize_decoders();
}

// Tables whose multiplicities the prover owns, so Std stops counting them.
uint64_t mul_migrated_tables(uint64_t *out, uint64_t cap) {
    return mul_migrated_tables_impl(out, cap);
}

// Fold the prover-owned spans into the caller's accumulator, once per proof. `hostAcc` is scoped to
// this call and never retained.
void mul_fold(uint64_t airId, uint64_t *hostAcc, uint64_t expectedCommits) {
    // Nothing prover-owned means no scatter ever ran, so there is nothing to fold.
    if (mulDecoders().empty() || hostAcc == nullptr) return;

    // Device mirrors exist only on a GPU run; a CPU run of this same library scatters into the host
    // accumulator instead, and must still reach the fold below.
    // Every instance must have counted, or its scatter is still queued and the counts are short.
    if (!mul_await_commits(expectedCommits)) exitProcess();
    if (!mulAccs().empty()) {
        mul_oob_report();
        mul_fold_air(airId, hostAcc);
    }

    // One of the two accumulators is always empty, so folding both keeps the caller backend-blind.
    mul_cpu_fold(airId, hostAcc);
}

// Allocate device mirrors for airs hosting a migrated table. Idempotent; a no-op while every
// table is still Rust-owned, so it costs nothing until the first decoder is registered.
void mul_alloc(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    std::vector<int> gpuIds(d_buffers->n_gpus);
    for (uint32_t g = 0; g < d_buffers->n_gpus; ++g) gpuIds[g] = (int)d_buffers->my_gpu_ids[g];
    mul_alloc_devices(gpuIds.data(), (int)gpuIds.size());
    for (int id : gpuIds) mul_alloc_oob(id);

    // Inventory: what is left to migrate, with the height each decoder would have to cover.
    for (const auto &L : mulVtLayouts()) {
        size_t have = 0;
        std::string todo;
        for (const auto &kv : L.accBase) {
            if (mulDecoderFor(kv.first) != nullptr) ++have;
            else todo += " " + to_string(kv.first) + "(h=" + to_string(L.tableHeight(kv.first)) + ")";
        }
        zklog.info("Multiplicity coverage air " + to_string(L.airId) + ": " + to_string(have)
                   + "/" + to_string(L.accBase.size()) + " tables prover-owned; remaining:" + todo);
    }
}

// Once per proof, at the same point the std resets its host accumulator.
// Count one instance on the host, for the paths that never commit -- verify-constraints
// calculates the witness and evaluates the constraints without merkelizing, so the commit hook
// in commit_witness_gpu does not fire there.
uint64_t mul_air_has_lookups(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr) return 0;
    const MulPlan &p = mulPlanFor(*(SetupCtx *)pSetupCtx_, airgroupId, airId);
    return (p.jobs.empty() && p.fallback.empty()) ? 0 : 1;
}

void mul_scatter(void *pSetupCtx_, void *params_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr || params_ == nullptr) return;
    mul_cpu_alloc();
    mul_scatter_cpu(*(SetupCtx *)pSetupCtx_, *(StepsParams *)params_, airgroupId, airId);
    mul_note_commit();
}

void mul_reset() {
    mul_reset_all();
    mul_cpu_reset();
}

} // extern "C"
