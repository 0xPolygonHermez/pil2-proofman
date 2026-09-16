// C entry points for the prover-side multiplicities on the CPU backend.
//
// The GPU backend's copy lives in multiplicity_api.cu. Both are compiled into the GPU library, so
// this file is empty there and the two never collide; the CPU library compiles only this one.
// Keeping the CPU path behind the SAME entry points is what lets the Rust side stay backend-blind:
// `mul_fold_c` folds a device mirror or a host accumulator depending only on which library is
// linked, and a table is prover-owned either way.
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

// Table geometry for the range tables, registered once per run before any decoder is materialized.
void mul_register_range_tables(const uint64_t *tableIds, const int64_t *biases, uint64_t n) {
    mul_register_range_tables_impl(tableIds, biases, n);
}

void register_mul_vt(uint64_t airgroupId, uint64_t airId, uint64_t numRows, uint64_t numCols,
                     const uint64_t *tableIds, const uint64_t *accBases, uint64_t nTables) {
    mul_register_vt(airgroupId, airId, numRows, numCols, tableIds, accBases, nTables);
    mul_materialize_decoders();
}

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

// No commit barrier: the CPU scatter runs inline on the committing thread, so by the time an
// instance's commit returns its counts are already in. `expectedCommits` is accepted and ignored so
// the Rust side needs no backend-specific call.
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

// Count one instance, for the paths that never commit: verify-constraints calculates the witness
// and evaluates the constraints without ever merkelizing, so the commit hook does not fire there.
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
    mul_reset_commits();
    mul_cpu_reset();
}

} // extern "C"

#endif // __USE_CUDA__
