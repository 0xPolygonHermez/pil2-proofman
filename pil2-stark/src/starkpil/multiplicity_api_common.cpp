// C entry points for prover-side multiplicities that are identical on both backends; the
// backend-specific half is in multiplicity_api.cu / multiplicity_api_cpu.cpp.
// No CUDA in the include chain: compiled into both libstarks.a and libstarksgpu.a.

#include <cstdint>
#include "multiplicity.hpp"
#include "multiplicity_decoders.hpp"
#include "multiplicity_plan.hpp"
#include "multiplicity_cpu.hpp"
#include "zklog.hpp"

extern "C" {

// --- table registration: geometry and decode rules, before any proof starts ---

void mul_register_range_tables(const uint64_t *tableIds, const int64_t *biases, uint64_t n) {
    mul_register_range_tables_impl(tableIds, biases, n);
}

void mul_register_table_decode(uint64_t tableId, const uint64_t *coef, uint64_t nCoef,
                               uint64_t konst) {
    mul_register_table_decode_impl(tableId, coef, nCoef, konst);
}

void mul_register_table_map(uint64_t tableId, const uint64_t *kv, uint64_t n, uint64_t slots,
                            uint64_t nKey) {
    mul_register_table_map_impl(tableId, kv, n, slots, nKey);
}

void mul_register_table_digits(uint64_t tableId, const uint64_t *tab, uint64_t n,
                               const uint32_t *cols, uint64_t nCols) {
    mul_register_table_digits_impl(tableId, tab, n, cols, nCols);
}


// --- queries the caller uses to decide what the host still has to build ---

uint64_t mul_migrated_tables(uint64_t *out, uint64_t cap) {
    return mul_migrated_tables_impl(out, cap);
}

// Whether the prover counts any table of this air.
uint64_t mul_air_has_owned(uint64_t airId) {
    return mul_air_has_owned_tables(airId) ? 1 : 0;
}

uint64_t mul_air_has_lookups(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr) return 0;
    const MulPlan &p = mulPlanFor(*(SetupCtx *)pSetupCtx_, airgroupId, airId);
    return (p.jobs.empty() && p.fallback.empty()) ? 0 : 1;
}

// The host scatter: a GPU build uses it for anything the device path declined.
void mul_scatter(void *pSetupCtx_, void *params_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr || params_ == nullptr) return;
    mul_cpu_alloc();
    mul_scatter_cpu(*(SetupCtx *)pSetupCtx_, *(StepsParams *)params_, airgroupId, airId);
    mul_note_commit();
}


// Record a virtual table's layout, then materialise any decoders that are now complete.
void register_mul_vt(uint64_t airgroupId, uint64_t airId, uint64_t numRows, uint64_t numCols,
                     const uint64_t *tableIds, const uint64_t *accBases, uint64_t nTables) {
    mul_register_vt(airgroupId, airId, numRows, numCols, tableIds, accBases, nTables);
    mul_materialize_decoders();
}
} // extern "C"
