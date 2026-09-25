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

// The CPU half of the prover-side scatter, so table ownership holds on every backend (CPU runs
// and verify-constraints included). Same jobs, decode and accumulator layout as the kernel.

// One accumulator per air, sized like its virtual table. Atomic u64: instances commit from
// several threads into one shared array.
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

// Whatever did not compile still goes through the expression interpreter.
void mul_scatter_fallback_cpu(SetupCtx& setupCtx, StepsParams& params, const MulPlan& plan);

// Host copy of mulEvalProgram (multiplicity_eval.cuh); must match it. Only addressing differs:
// sections are row-major here (`row * nCols + col`), column-major on the device.
inline uint64_t mulTermValueCPU(const MulTermDev& t, const uint64_t* const* bases,
                                uint64_t row, uint64_t rowMask) {
    if (MUL_SRC_IS_UNIFORM(t.src)) return mulCanonHD(bases[t.src][t.sectionOffset]);
    // MUL_SRC_PACKED / MUL_SRC_HINTCOL are slot-commit sources; nothing reaches this backend.
    const uint64_t r = (row + (uint64_t)t.rowStride) & rowMask;
    return mulCanonHD(bases[t.src][t.sectionOffset + r * t.nCols + t.col]);
}

inline uint64_t mulOperandValCPU(const MulOperandDev& o, const uint64_t* const* bases,
                                 uint64_t row, uint64_t rowMask, const uint64_t* tmp) {
    if (o.kind == MUL_OPND_TEMP)  return tmp[o.tmp];
    if (o.kind == MUL_OPND_CONST) return o.konst;
    return mulTermValueCPU(o.term, bases, row, rowMask);
}

inline uint64_t mulEvalProgramCPU(const MulInsnDev* prog, uint32_t n, const uint64_t* const* bases,
                                  uint64_t row, uint64_t rowMask) {
    uint64_t tmp[MUL_PROG_MAX_TEMP];
    uint64_t last = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const MulInsnDev& in = prog[k];
        const uint64_t a = mulOperandValCPU(in.a, bases, row, rowMask, tmp);
        const uint64_t b = mulOperandValCPU(in.b, bases, row, rowMask, tmp);
        uint64_t r;
        switch (in.op) {
            case 0:  r = mulAddFE(a, b); break;
            case 1:  r = mulSubFEHD(a, b); break;
            case 2:  r = mulMulFE(a, b); break;
            default: r = mulSubFEHD(b, a); break;   // rsub
        }
        if (k + 1 == n) { last = r; break; }
        tmp[in.dst] = r;
    }
    return last;
}

inline void mul_scatter_cpu(SetupCtx& setupCtx, StepsParams& params, uint64_t airgroupId, uint64_t airId) {
    if (mulDecoders().empty()) return;
    const MulPlan& plan = mulPlanFor(setupCtx, airgroupId, airId);
    if (plan.jobs.empty() && plan.fallback.empty()) return;

    const uint64_t* bases[MUL_SRC_N] = {
        (const uint64_t*)params.pConstPolsAddress, (const uint64_t*)params.trace,
        (const uint64_t*)params.aux_trace,         (const uint64_t*)params.publicInputs,
        (const uint64_t*)params.airValues,         (const uint64_t*)params.proofValues,
        (const uint64_t*)params.airgroupValues,    (const uint64_t*)params.pCustomCommitsFixed,
        nullptr, nullptr };   // slot-only sources; the CPU scatter never sees one
    const uint64_t rowMask = (1ULL << setupCtx.starkInfo.starkStruct.nBits) - 1;

    mul_scatter_fallback_cpu(setupCtx, params, plan);

    // Every field is bytecode; the offsets in a job index this one buffer.
    const MulInsnDev* prog = plan.prog.data();

    for (const auto& j : plan.jobs) {
        // Per job: counters live in the air hosting the table.
        std::vector<std::atomic<uint64_t>>* acc = nullptr;
        {
            std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
            auto it = mulCpuAccs().find(j.hostAirId);
            if (it == mulCpuAccs().end()) continue;
            acc = &it->second;
        }
        std::atomic<uint64_t> oob{0}, firstBadIdx{0};
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < (int64_t)j.rows; ++row) {
            uint64_t sel = 1;
            if (!j.selConstOne) {
                sel = mulEvalProgramCPU(prog + j.selProgOff, j.selProgLen, bases, row, rowMask);
                if (sel == 0) continue;
            }
            if (j.hasBus && mulEvalProgramCPU(prog + j.busProgOff, j.busProgLen, bases, row, rowMask)
                            != (uint64_t)j.tableId) continue;
            uint64_t key[MUL_MAX_TUPLE];
            if (j.mapSlots == 0 && j.digitCols == 0) {
                key[0] = mulAddFE(mulEvalProgramCPU(prog + j.valProgOff, j.valProgLen, bases, row, rowMask),
                                  j.biasFE);
            } else {
                for (uint32_t c = 0; c < j.nKey; ++c)
                    key[c] = mulEvalProgramCPU(prog + j.keyProgOff[c], j.keyProgLen[c], bases, row, rowMask);
            }
            uint64_t idx;
            // Same resolve as the kernel.
            if (!mulResolveRow(key, j.nKey, j.mapSlots, j.mapKV, idx, j.digitCols, j.digitTab) || idx >= j.nTableRows) {
                if (oob.fetch_add(1, std::memory_order_relaxed) == 0)
                    firstBadIdx.store(key[0], std::memory_order_relaxed);
                continue;
            }
            (*acc)[j.accBase + idx].fetch_add(sel, std::memory_order_relaxed);
        }
        // A correct decode never lands outside the table; report it loudly.

        if (oob.load() != 0) {
            zklog.error("multiplicity: " + std::to_string(oob.load()) + " decodes outside table "
                        + std::to_string(j.tableId) + " from air " + std::to_string(airgroupId) + "/"
                        + std::to_string(airId) + " (first " + std::to_string(firstBadIdx.load())
                        + " of " + std::to_string(j.nTableRows) + ") -- that decoder is wrong");
        }
    }
}

#endif
