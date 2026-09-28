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
        if (!mulLayoutHostsMigrated(L) || mulCpuAccs().count(L.airKey)) continue;
        mulCpuAccs().emplace(L.airKey, std::vector<std::atomic<uint64_t>>(L.nCounters));
        zklog.trace("Multiplicity accumulator (CPU): " + std::to_string(L.nCounters * 8 / (1 << 20))
                   + " MB for air " + mulAirName(L.airKey));
    }
}

// Bad decodes this proof over every CPU scatter, for mul_sync_commits.
inline std::atomic<uint64_t>& mulCpuOobTotal() { static std::atomic<uint64_t> n{0}; return n; }

inline void mul_cpu_reset() {
    mulCpuOobTotal().store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
    for (auto& kv : mulCpuAccs())
        for (auto& c : kv.second) c.store(0, std::memory_order_relaxed);
}

// Add an air's counts into the std's host accumulator. Mirrors mul_fold_air.
inline void mul_cpu_fold(uint64_t airKey, uint64_t* hostAcc) {
    std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
    auto it = mulCpuAccs().find(airKey);
    if (it == mulCpuAccs().end() || hostAcc == nullptr) return;
    for (const auto& d : mulDecoders()) {
        if (d.hostAirKey != airKey) continue;
        for (uint64_t i = 0; i < d.n_rows; ++i)
            hostAcc[d.acc_base + i] += it->second[d.acc_base + i].load(std::memory_order_relaxed);
    }
}

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

// Before stage 2 (`!auxReady`) the aux trace holds another air's data.
inline void mul_scatter_cpu(SetupCtx& setupCtx, StepsParams& params, uint64_t airgroupId, uint64_t airId,
                            bool auxReady) {
    if (mulDecoders().empty()) return;
    const MulPlan& plan = mulPlanFor(setupCtx, airgroupId, airId);
    if (plan.jobs.empty()) return;
    if (!auxReady && (plan.srcMask & (1u << MUL_SRC_AUX)) != 0) {
        zklog.error("multiplicity: air " + std::to_string(airgroupId) + "/" + std::to_string(airId)
                    + " looks up a stage-2 or im-pol value, which this commit does not have yet");
        exitProcess();
    }

    const uint64_t* bases[MUL_SRC_N] = {
        (const uint64_t*)params.pConstPolsAddress, (const uint64_t*)params.trace,
        (const uint64_t*)params.aux_trace,         (const uint64_t*)params.publicInputs,
        (const uint64_t*)params.airValues,         (const uint64_t*)params.proofValues,
        (const uint64_t*)params.airgroupValues,    (const uint64_t*)params.pCustomCommitsFixed,
        nullptr, nullptr };   // slot-only sources; the CPU scatter never sees one
    const uint64_t rowMask = (1ULL << setupCtx.starkInfo.starkStruct.nBits) - 1;

    // Every field is bytecode; the offsets in a job index this one buffer.
    const MulInsnDev* prog = plan.prog.data();

    for (const auto& j : plan.jobs) {
        // Per job: counters live in the air hosting the table.
        std::vector<std::atomic<uint64_t>>* acc = nullptr;
        {
            std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
            auto it = mulCpuAccs().find(j.hostAirKey);
            if (it == mulCpuAccs().end()) continue;
            acc = &it->second;
        }
        std::atomic<uint64_t> oob{0}, firstBadIdx{0}, badSel{0}, firstBadSel{0};
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < (int64_t)j.rows; ++row) {
            uint64_t sel = 1;
            if (!j.selConstOne) {
                sel = mulEvalProgramCPU(prog + j.selProgOff, j.selProgLen, bases, row, rowMask);
                if (sel == 0) continue;
            }
            if (j.hasBus && mulEvalProgramCPU(prog + j.busProgOff, j.busProgLen, bases, row, rowMask)
                            != (uint64_t)j.tableId) continue;
            // Added as an integer: a "negative" field selector would wrap mod 2^64, not p.
            if (sel >= MUL_SEL_MAX) {
                if (badSel.fetch_add(1, std::memory_order_relaxed) == 0)
                    firstBadSel.store(sel, std::memory_order_relaxed);
                continue;
            }
            // Same resolve as the kernel.
            uint64_t idx = 0, first = 0;
            bool ok = true;
            if (j.mapSlots == 0) {
                idx = first = mulAddFE(mulEvalProgramCPU(prog + j.valProgOff, j.valProgLen, bases, row, rowMask),
                                       j.biasFE);
            } else {
                const uint32_t* refs = plan.keyRefs.data() + j.keyRefOff;
                uint64_t kw[MUL_MAP_MAX_WORDS] = {};
                for (uint32_t c = 0; c < j.nKey && ok; ++c) {
                    const uint64_t v = mulEvalProgramCPU(prog + refs[2 * c], refs[2 * c + 1], bases, row, rowMask);
                    if (c == 0) first = v;
                    ok = mulMapPack(j.mapKV, c, v, kw);
                }
                ok = ok && mulMapFind(j.mapKV, kw, idx);
            }
            if (!ok || idx >= j.nTableRows) {
                if (oob.fetch_add(1, std::memory_order_relaxed) == 0)
                    firstBadIdx.store(first, std::memory_order_relaxed);
                continue;
            }
            (*acc)[j.accBase + idx].fetch_add(sel, std::memory_order_relaxed);
        }
        // A correct decode never lands outside the table; report it loudly.
        mulCpuOobTotal().fetch_add(oob.load() + badSel.load(), std::memory_order_relaxed);
        if (oob.load() != 0)
            zklog.error("multiplicity: " + std::to_string(oob.load()) + " decodes outside table "
                        + std::to_string(j.tableId) + " from air " + std::to_string(airgroupId) + "/"
                        + std::to_string(airId) + " (first " + std::to_string(firstBadIdx.load())
                        + " of " + std::to_string(j.nTableRows) + ")");
        if (badSel.load() != 0)
            zklog.error("multiplicity: " + std::to_string(badSel.load()) + " selectors >= 2^32 in lookups "
                        "into table " + std::to_string(j.tableId) + " from air " + std::to_string(airgroupId)
                        + "/" + std::to_string(airId) + " (first " + std::to_string(firstBadSel.load()) + ")");
    }
}

#endif
