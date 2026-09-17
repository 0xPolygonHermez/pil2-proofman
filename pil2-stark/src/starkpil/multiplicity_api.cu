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
// A fitted tuple->row map for one table, derived and verified by the caller from that table's own
// fixed columns.
void mul_register_table_decode(uint64_t tableId, const uint64_t *coef, uint64_t nCoef,
                               uint64_t konst) {
    mul_register_table_decode_impl(tableId, coef, nCoef, konst);
}

// An exact-match key->row map for one table: [key, row] pairs, open addressed.
void mul_register_table_digits(uint64_t tableId, const uint64_t *tab, uint64_t n,
                               const uint32_t *cols, uint64_t nCols) {
    mul_register_table_digits_impl(tableId, tab, n, cols, nCols);
}

void mul_register_table_map(uint64_t tableId, const uint64_t *kv, uint64_t n, uint64_t slots,
                            uint64_t nKey) {
    mul_register_table_map_impl(tableId, kv, n, slots, nKey);
}

// A per-block base plus uniform strides for one table: row = base[idx(selector)] + sum stride*key.
void mul_register_table_indexed_base(uint64_t tableId, const uint64_t *sel, uint64_t nSel,
                                     const uint64_t *base, uint64_t nBase,
                                     const uint64_t *stride, uint64_t nStride) {
    mul_register_table_indexed_base_impl(tableId, sel, nSel, base, nBase, stride, nStride);
}

uint64_t mul_migrated_tables(uint64_t *out, uint64_t cap) {
    return mul_migrated_tables_impl(out, cap);
}

// Fold the prover-owned spans into the caller's accumulator, once per proof. `hostAcc` is scoped to
// this call and never retained.
// True when the device can produce this air's whole cm1 by itself, so the host must not build a
// trace for it. False on a CPU-only build, where there is no device accumulator to export from.
// Off until the caller says the counts never need a cross-rank reduction: with several ranks the
// host adds every rank's share into one accumulator, and this device one holds only ours.
uint64_t mul_air_has_owned(uint64_t airId) {
    return mul_air_has_owned_tables(airId) ? 1 : 0;
}

void mul_set_device_export(uint64_t enabled) {
#ifdef __USE_CUDA__
    mulDeviceExportEnabled() = (enabled != 0);
#else
    (void)enabled;
#endif
}

uint64_t mul_air_device_owned(uint64_t airId) {
#ifdef __USE_CUDA__
    return mul_air_fully_owned(airId) ? 1 : 0;
#else
    (void)airId;
    return 0;
#endif
}

// Ordering point for the device export: every instance has launched its scatter once this returns,
// which is what lets the table's own commit transpose a complete accumulator. The fold used to be
// where this happened, and the wait is all that is still needed once nothing comes back to the host.
void mul_sync_commits(uint64_t expectedCommits) {
    if (mulDecoders().empty()) return;
    if (!mul_await_commits(expectedCommits)) exitProcess();
#ifdef __USE_CUDA__
    mul_phase_report();
    mul_oob_report();
#endif
}

void mul_fold(uint64_t airId, uint64_t *hostAcc, uint64_t expectedCommits) {
    // Nothing prover-owned means no scatter ever ran, so there is nothing to fold.
    if (mulDecoders().empty() || hostAcc == nullptr) return;

    // Device mirrors exist only on a GPU run; a CPU run of this same library scatters into the host
    // accumulator instead, and must still reach the fold below.
    // Every instance must have counted, or its scatter is still queued and the counts are short.
    if (!mul_await_commits(expectedCommits)) exitProcess();
    if (!mulAccs().empty()) {
        mul_phase_report();
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
    for (int id : gpuIds) { mul_alloc_oob(id); mul_alloc_maps(id); mul_alloc_digits(id); mul_alloc_indexed_base(id); }

    // The two things an operator needs from this phase: how much of the proving key the prover
    // claimed (one aggregate line, shared with the CPU-only backend), and how much GPU memory that
    // costs, per device -- the number with real consequence, since it competes directly with the
    // prover's own arena on that device and does not sum meaningfully across devices.
    mul_log_coverage();
    for (int id : gpuIds) {
        const uint64_t bytes = mul_gpu_resident_bytes(id);
        if (bytes != 0)
            zklog.info("Multiplicity: " + to_string(bytes / (1024 * 1024))
                       + " MB GPU-resident on gpu " + to_string(id));
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
