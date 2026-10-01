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

void mul_register_table_map(uint64_t tableId, const uint64_t *kv, uint64_t n, uint64_t slots,
                            uint64_t nKey) {
    mul_register_table_map_impl(tableId, kv, n, slots, nKey);
}

// --- queries the caller uses to decide what the host still has to build ---

uint64_t mul_migrated_tables(uint64_t *out, uint64_t cap) {
    return mul_migrated_tables_impl(out, cap);
}

// Whether the prover counts any table of this air.
uint64_t mul_air_has_owned(uint64_t airKey) {
    return mul_air_has_owned_tables(airKey) ? 1 : 0;
}

// The host scatter: a GPU build uses it for anything the device path declined. `auxReady`: stage 2
// and the im-pols are in the aux trace.
void mul_scatter(void *pSetupCtx_, void *params_, uint64_t airgroupId, uint64_t airId, uint64_t auxReady) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr || params_ == nullptr) return;
    mul_cpu_alloc();
    mul_scatter_cpu(*(SetupCtx *)pSetupCtx_, *(StepsParams *)params_, airgroupId, airId, auxReady != 0);
    mul_note_commit();
}

// Instances counted so far this proof.
uint64_t mul_commit_count() { return mulCommits().load(std::memory_order_acquire); }

// Whether the air looks up any prover-owned table. A table air must not: it commits after the fold.
uint64_t mul_air_has_jobs(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr) return 0;
    return mulPlanFor(*(SetupCtx *)pSetupCtx_, airgroupId, airId).jobs.empty() ? 0 : 1;
}

// Whether such a lookup reads a stage-2 or im-pol value, which no commit can count.
uint64_t mul_air_reads_aux(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty() || pSetupCtx_ == nullptr) return 0;
    return (mulPlanFor(*(SetupCtx *)pSetupCtx_, airgroupId, airId).srcMask & (1u << MUL_SRC_AUX)) ? 1 : 0;
}

// --- a table air's own rows, evaluated from its fixed columns for the host's row-map fit ---

static const HintField* mulHintField(const Hint& h, const char* name) {
    for (const auto& f : h.fields) if (f.name == name) return &f;
    return nullptr;
}

// Its proves-side gsum_debug_data hints, in hint order.
static std::vector<const Hint*> mulProvesHints(SetupCtx& s) {
    std::vector<const Hint*> out;
    const uint64_t n = s.expressionsBin.getNumberHintIdsByName("gsum_debug_data");
    std::vector<uint64_t> ids(n);
    if (n != 0) s.expressionsBin.getHintIdsByName(ids.data(), "gsum_debug_data");
    for (uint64_t id : ids) {
        const Hint& h = s.expressionsBin.hints[id];
        const HintField* ty = mulHintField(h, "type_piop");
        if (ty != nullptr && !ty->values.empty() && ty->values[0].operand == opType::number
            && ty->values[0].value == MUL_PIOP_PROVES)
            out.push_back(&h);
    }
    return out;
}

// Per proves-side lookup: its multiplicity's stage-1 column (-1 if not a plain one) and tuple length.
// Returns how many there are.
uint64_t mul_proves_hints(void *pSetupCtx_, int64_t *cols, uint64_t *lens, uint64_t cap) {
    SetupCtx& s = *(SetupCtx *)pSetupCtx_;
    const auto hints = mulProvesHints(s);
    for (uint64_t k = 0; k < hints.size() && k < cap; ++k) {
        const HintField* m = mulHintField(*hints[k], "num_reps");
        const HintField* ex = mulHintField(*hints[k], "expressions");
        int64_t col = -1;
        if (m != nullptr && m->values.size() == 1 && m->values[0].operand == opType::cm) {
            const HintFieldValue& v = m->values[0];
            const PolMap& p = s.starkInfo.cmPolsMap[v.id];
            if (p.stage == 1 && p.dim == 1 && v.rowOffsetIndex < s.starkInfo.openingPoints.size()
                && s.starkInfo.openingPoints[v.rowOffsetIndex] == 0)
                col = (int64_t)p.stagePos;
        }
        cols[k] = col;
        lens[k] = ex == nullptr ? 0 : ex->values.size();
    }
    return hints.size();
}

// Rows [r0, r1) of proves-side lookup k: bus ids, and the leading `len` tuple elements row-major.
// `constPols` is the air's .const (`constLen` words). 0 when a field does not compile or reads more
// than fixed columns.
uint64_t mul_eval_proves_hint(void *pSetupCtx_, uint64_t k, const uint64_t *constPols, uint64_t constLen,
                              uint64_t r0, uint64_t r1, uint64_t *bus, uint64_t *tuple, uint64_t len) {
    SetupCtx& s = *(SetupCtx *)pSetupCtx_;
    const StarkInfo& si = s.starkInfo;
    const uint64_t nRows = 1ULL << si.starkStruct.nBits;
    const auto hints = mulProvesHints(s);
    if (k >= hints.size() || r1 > nRows || r0 > r1 || constLen < nRows * si.nConstants) return 0;
    const HintField* b = mulHintField(*hints[k], "busid");
    const HintField* ex = mulHintField(*hints[k], "expressions");
    if (b == nullptr || b->values.size() != 1 || ex == nullptr || ex->values.size() < len) return 0;

    const uint64_t bufferCommitSize = 1 + si.nStages + 3 + si.customCommits.size();
    std::vector<MulProgram> progs(1 + len);
    for (size_t e = 0; e < progs.size(); ++e) {
        const HintFieldValue& v = e == 0 ? b->values[0] : ex->values[e - 1];
        if (!mulCompileField(s, v, bufferCommitSize, progs[e])) return 0;
        for (const auto& in : progs[e].insns)
            for (const MulOperandDev* o : {&in.a, &in.b})
                if (o->kind == MUL_OPND_COL && o->term.src != MUL_SRC_CONST) return 0;
    }
    const uint64_t* bases[MUL_SRC_N] = {constPols};
    auto run = [&](const MulProgram& p, uint64_t r) {
        return mulEvalProgramCPU(p.insns.data(), (uint32_t)p.insns.size(), bases, r, nRows - 1);
    };
    for (uint64_t r = r0; r < r1; ++r) {
        bus[r - r0] = run(progs[0], r);
        for (uint64_t e = 0; e < len; ++e) tuple[(r - r0) * len + e] = run(progs[e + 1], r);
    }
    return 1;
}

// Record a virtual table's layout, then materialise any decoders that are now complete.
void register_mul_vt(uint64_t airgroupId, uint64_t airId, uint64_t numRows, uint64_t numCols,
                     const uint64_t *tableIds, const uint64_t *accBases, uint64_t nTables) {
    mul_register_vt(airgroupId, airId, numRows, numCols, tableIds, accBases, nTables);
    mul_materialize_decoders();
}
} // extern "C"
