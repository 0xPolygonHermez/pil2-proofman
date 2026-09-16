#ifndef MULTIPLICITY_CPU_HPP
#define MULTIPLICITY_CPU_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <omp.h>
#include "multiplicity_plan.hpp"
#include "multiplicity.hpp"
#include "steps.hpp"

// The CPU half of the prover-side scatter.
//
// It exists so that owning a table is a property of the PROOF, not of the backend. The GPU path
// stops the state machines counting these tables; without a CPU path, a CPU run (or
// verify-constraints, which is CPU-only) would have nobody counting them and the lookup argument
// would not close. Same jobs, same decode, same accumulator layout as the kernel -- only the loop
// differs.

// One accumulator per air, sized like the virtual table it mirrors. Counters are u64 and updated
// atomically: instances commit from several threads, and unlike the GPU there is no per-device
// mirror to fold, so they share one array.
inline std::map<uint64_t, std::vector<std::atomic<uint64_t>>>& mulCpuAccs() {
    static std::map<uint64_t, std::vector<std::atomic<uint64_t>>> m;
    return m;
}
inline std::mutex& mulCpuAccsMutex() { static std::mutex m; return m; }

// Allocate an accumulator for every air hosting a prover-owned table. Idempotent, and a no-op
// while nothing is owned.
inline void mul_cpu_alloc() {
    std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
    for (const auto& L : mulVtLayouts()) {
        bool hosts = false;
        for (const auto& d : mulDecoders())
            if (L.accBase.count(d.table_id)) { hosts = true; break; }
        if (!hosts || mulCpuAccs().count(L.airId)) continue;
        mulCpuAccs().emplace(L.airId, std::vector<std::atomic<uint64_t>>(L.nCounters));
        // Per air -- host memory, not the GPU arena the coordinator's "operator acts on" line is
        // about, and it scales with the proving key. Trace only.
        zklog.trace("Multiplicity accumulator (CPU): " + std::to_string(L.nCounters * 8 / (1 << 20))
                   + " MB for air " + std::to_string(L.airId));
    }
}

inline void mul_cpu_reset() {
    std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
    for (auto& kv : mulCpuAccs())
        for (auto& c : kv.second) c.store(0, std::memory_order_relaxed);
}

// Move an air's counts into the std's host accumulator, then clear them. Mirrors mul_fold_air.
inline void mul_cpu_fold(uint64_t airId, uint64_t* hostAcc) {
    std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
    auto it = mulCpuAccs().find(airId);
    if (it == mulCpuAccs().end() || hostAcc == nullptr) return;
    for (const auto& d : mulDecoders()) {
        bool owned = false;
        for (const auto& L : mulVtLayouts())
            if (L.airId == airId && L.accBase.count(d.table_id)) { owned = true; break; }
        if (!owned) continue;
        for (uint64_t i = 0; i < d.n_rows; ++i)
            hostAcc[d.acc_base + i] += it->second[d.acc_base + i].load(std::memory_order_relaxed);
    }
}

inline uint64_t mulEvalTermsCPU(const MulTermDev* t_, uint32_t n, uint64_t konst,
                                const uint64_t* const* bases, uint64_t row, uint64_t rowMask) {
    uint64_t acc = konst;
    for (uint32_t i = 0; i < n; ++i) {
        const MulTermDev& t = t_[i];
        // Row-major: this backend stores a section as `row * nCols + col` (expressions_pack.hpp).
        const uint64_t off = MUL_SRC_IS_UNIFORM(t.src)
                           ? t.sectionOffset
                           : t.sectionOffset + ((row + (uint64_t)t.rowStride) & rowMask) * t.nCols
                             + t.col;
        const uint64_t v = mulCanonHD(bases[t.src][off]);
        acc = mulAddFE(acc, t.coef == 1 ? v : mulMulFE(v, t.coef));
    }
    return acc;
}

inline uint64_t mulEvalFormCPU(const MulFormDev& f, const uint64_t* const* bases,
                               uint64_t row, uint64_t rowMask) {
    const uint64_t a = mulEvalTermsCPU(f.t, f.n, f.konst, bases, row, rowMask);
    if (!f.hasProduct) return a;
    return mulMulFE(a, mulEvalTermsCPU(f.t2, f.n2, f.konst2, bases, row, rowMask));
}

// The lookups the extractor could not reduce to a closed form. The GPU commit path runs these
// through the expression interpreter; this is the same thing on the host, so a CPU run counts
// exactly what a GPU run does instead of coming up short by whatever failed to reduce.
//
// The fields are materialised into plain buffers and scattered afterwards rather than scattered
// from inside the evaluator: the CPU evaluator has no scatter destination, and a hint that reaches
// here is rare enough that one pass over a column costs nothing worth saving.
// Defined in multiplicity_fallback_cpu.cpp: pulling the expression evaluator's headers in here
// would close an include cycle through const_pols.hpp, and this is the only place that needs them.
void mul_scatter_fallback_cpu(SetupCtx& setupCtx, StepsParams& params, const MulPlan& plan,
                              uint64_t hostAirId);

// Count one instance's lookups into its air's accumulator. Called from the CPU commit path once the
// witness is filled, which is the same point the GPU launches its scatter.
// A sum of products: the single-product evaluator, summed. Mirrors mulEvalPoly on the device.
inline uint64_t mulEvalPolyCPU(const MulPolyDev& q, const uint64_t* const* bases,
                               uint64_t row, uint64_t rowMask) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < q.nProd; ++i)
        acc = mulAddFE(acc, mulEvalFormCPU(q.p[i], bases, row, rowMask));
    return acc;
}

inline void mul_scatter_cpu(SetupCtx& setupCtx, StepsParams& params, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty()) return;
    const MulPlan& plan = mulPlanFor(setupCtx, airgroupId, airId);
    if (plan.jobs.empty() && plan.fallback.empty()) return;

    const uint64_t* bases[MUL_SRC_N] = {
        (const uint64_t*)params.pConstPolsAddress, (const uint64_t*)params.trace,
        (const uint64_t*)params.aux_trace,         (const uint64_t*)params.publicInputs,
        (const uint64_t*)params.airValues,         (const uint64_t*)params.proofValues,
        (const uint64_t*)params.airgroupValues,    (const uint64_t*)params.pCustomCommitsFixed };
    const uint64_t rowMask = (1ULL << setupCtx.starkInfo.starkStruct.nBits) - 1;

    if (!plan.fallback.empty() && !plan.jobs.empty())
        mul_scatter_fallback_cpu(setupCtx, params, plan, plan.jobs[0].hostAirId);
    else if (!plan.fallback.empty())
        mul_scatter_fallback_cpu(setupCtx, params, plan, mulCpuAccs().empty() ? 0 : mulCpuAccs().begin()->first);

    for (const auto& j : plan.jobs) {
        // Per job, not per call: a lookup's counters live in the air that hosts its table, and one
        // instance can feed tables in different airs.
        std::vector<std::atomic<uint64_t>>* acc = nullptr;
        {
            std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
            auto it = mulCpuAccs().find(j.hostAirId);
            if (it == mulCpuAccs().end()) continue;
            acc = &it->second;
        }
        // Resolved fresh here, once per job, rather than cached in the job at plan-build time:
        // mulDecoders() keeps growing from other airs' registration and this plan may have been
        // built before that finished. By the time a job is actually scattered (proving time),
        // registration is done, so this lookup is safe and cheap relative to the per-row loop below.
        const MulDecoder* dec = j.hasIndexedBase ? mulDecoderFor(j.tableId) : nullptr;
        std::atomic<uint64_t> oob{0}, firstBadIdx{0};
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < (int64_t)j.rows; ++row) {
            uint64_t sel = 1;
            if (!j.selConstOne) {
                sel = mulEvalPolyCPU(j.sel, bases, row, rowMask);
                if (sel == 0) continue;
            }
            if (j.hasBus && mulEvalPolyCPU(j.bus, bases, row, rowMask) != (uint64_t)j.tableId) continue;
            uint64_t key[MUL_MAX_TUPLE];
            if (j.mapSlots == 0 && j.digitCols == 0) {
                key[0] = mulAddFE(mulEvalPolyCPU(j.value, bases, row, rowMask), j.biasFE);
            } else {
                for (uint32_t c = 0; c < j.nKey; ++c) key[c] = mulEvalPolyCPU(j.key[c], bases, row, rowMask);
            }
            uint64_t idx;
            // Same resolve the kernel uses, so the two backends cannot disagree about a row.
            if (!mulResolveRow(key, j.nKey, j.mapSlots, j.mapKV, idx, j.digitCols, j.digitTab, dec) || idx >= j.nTableRows) {
                if (oob.fetch_add(1, std::memory_order_relaxed) == 0)
                    firstBadIdx.store(key[0], std::memory_order_relaxed);
                continue;
            }
            (*acc)[j.accBase + idx].fetch_add(sel, std::memory_order_relaxed);
        }
        // A correct decode never lands outside the table: the lookup constrains the value. Loud
        // rather than silent, because a dropped decode shows up only as a proof that will not close.
        if (oob.load() != 0) {
            zklog.error("multiplicity: " + std::to_string(oob.load()) + " decodes outside table "
                        + std::to_string(j.tableId) + " from air " + std::to_string(airgroupId) + "/"
                        + std::to_string(airId) + " (first " + std::to_string(firstBadIdx.load())
                        + " of " + std::to_string(j.nTableRows) + ") -- that decoder is wrong");
        }
    }
}

#endif
