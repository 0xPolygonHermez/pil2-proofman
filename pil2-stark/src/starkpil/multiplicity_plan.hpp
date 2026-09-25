#ifndef MULTIPLICITY_PLAN_HPP
#define MULTIPLICITY_PLAN_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include "multiplicity_extract.hpp"
#include "multiplicity_job.hpp"
#include "multiplicity.hpp"
#include "setup_ctx.hpp"

// Per-air kernel plan, derived from setup alone and shared by every instance of the air.
struct MulPlan {
    std::vector<MulJobDev>      jobs;      // lookups the kernel handles
    // All compiled programs, concatenated; jobs hold offsets into this.
    std::vector<MulInsnDev>     prog;
    // Bit per MulSrc read by the jobs: a slot commit has no aux trace or custom commits on device.
    uint32_t                    srcMask = 0;
    uint64_t                    maxRows = 0;       // tallest job: the row-stationary grid height
};

inline bool mulPlanStreamable(const MulPlan& p) {
    // Const pols and packed rows are resident; publics and value pools are staged per slot
    // (MulStreamCtx::dVals, capped by PINNED_AUX_VALUES_MAX). Aux and custom commits stay out.
    const uint32_t resident = (1u << MUL_SRC_CONST)   | (1u << MUL_SRC_TRACE)
                            | (1u << MUL_SRC_PUBLIC)  | (1u << MUL_SRC_AIRVALUE)
                            | (1u << MUL_SRC_PROOFVALUE) | (1u << MUL_SRC_AIRGROUPVALUE);
    // Shifted cm1 reads wrap on rowMask over the whole domain, so no reach bound is needed.
    return (p.srcMask & ~resident) == 0;
}

inline std::string mulSrcName(uint32_t s) {
    switch (s) {
        case MUL_SRC_CONST:         return "const";
        case MUL_SRC_TRACE:         return "cm1";
        case MUL_SRC_AUX:           return "aux";
        case MUL_SRC_PUBLIC:        return "public";
        case MUL_SRC_AIRVALUE:      return "airvalue";
        case MUL_SRC_PROOFVALUE:    return "proofvalue";
        case MUL_SRC_AIRGROUPVALUE: return "airgroupvalue";
        case MUL_SRC_CUSTOM:        return "custom";
        default:                    return "?";
    }
}

inline std::string mulSrcMaskNames(uint32_t mask) {
    std::string out;
    for (uint32_t s = 0; s < MUL_SRC_N; ++s)
        if (mask & (1u << s)) out += (out.empty() ? "" : ",") + mulSrcName(s);
    return out.empty() ? "none" : out;
}

// Walk `gsum_debug_data` once per air and turn every range-check lookup it feeds into a job. A
// lookup into a prover-owned table that does not compile is fatal: nothing else counts it.
inline MulPlan mulBuildPlan(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId) {
    MulPlan plan;
    std::map<std::string, uint32_t> progCache;
    const uint64_t n = setupCtx.expressionsBin.getNumberHintIdsByName("gsum_debug_data");
    if (n == 0) return plan;
    std::vector<uint64_t> hints(n);
    setupCtx.expressionsBin.getHintIdsByName(hints.data(), "gsum_debug_data");

    const uint64_t nRows = 1ULL << setupCtx.starkInfo.starkStruct.nBits;
    const uint64_t bufferCommitSize = 1 + setupCtx.starkInfo.nStages + 3
                                    + setupCtx.starkInfo.customCommits.size();
    auto fatal = [&](uint32_t tableId, const std::string& why) {
        zklog.error("multiplicity: air " + std::to_string(airgroupId) + "/" + std::to_string(airId)
                    + " looks up prover-owned table " + std::to_string(tableId) + ", but " + why
                    + " -- the prover cannot count it");
        exitProcess();
    };

    for (uint64_t i = 0; i < n; ++i) {
        const Hint& hint = setupCtx.expressionsBin.hints[hints[i]];
        auto fld = [&](const char* nm) -> const HintField* {
            for (const auto& f : hint.fields) if (f.name == nm) return &f;
            return nullptr;
        };
        auto isNum = [](const HintField* f) {
            return f != nullptr && !f->values.empty() && f->values[0].operand == opType::number;
        };
        const HintField *fEx = fld("expressions"), *fOp = fld("opids"), *fTy = fld("type_piop");
        const HintField *fDx = fld("deg_expr"), *fDs = fld("deg_sel");
        if (!isNum(fOp) || !isNum(fTy) || !isNum(fDx) || !isNum(fDs)) continue;
        if (fEx == nullptr || fEx->values.empty()) continue;
        if (fTy->values[0].value != MUL_PIOP_ASSUMES) continue;

        // Degree zero in tuple and selector: counted once per instance, not per row. Read from
        // the hint, since an expression over airvalues is degree 0 too.
        const uint64_t rows = (fDx->values[0].value == 0 && fDs->values[0].value == 0) ? 1ULL : nRows;
        const HintField* fSel = fld("num_reps");
        const bool selConst1 = isNum(fSel) && fSel->values[0].value == 1;
        // Several opids, or a computed busid, means the bus is chosen per row.
        const bool dynBus = fOp->values.size() > 1 || (fld("busid") != nullptr && !isNum(fld("busid")));

        // Append a program to `plan.prog`, deduplicated by instruction bytes (call sites get
        // fresh expression ids). Identical programs must share an offset so the kernel can
        // evaluate a shared selector once for a run of jobs.
        auto place = [&](const MulProgram& pg, uint32_t& off, uint32_t& len) {
            const std::string body((const char*)pg.insns.data(), pg.insns.size() * sizeof(MulInsnDev));
            len = (uint32_t)pg.insns.size();
            auto seen = progCache.find(body);
            if (seen != progCache.end()) { off = seen->second; return; }
            off = (uint32_t)plan.prog.size();
            plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
            progCache[body] = off;
        };
        // `why` is set when the field does not compile.
        struct Field { uint32_t off = 0, len = 0; std::string why; };
        auto compile = [&](const HintFieldValue* v) {
            Field r;
            MulProgram pg;
            if (v == nullptr) r.why = "missing";
            else if (!mulCompileField(setupCtx, *v, bufferCommitSize, pg))
                r.why = mulProgFailReason() ? mulProgFailReason() : "unknown";
            else place(pg, r.off, r.len);
            return r;
        };
        auto first = [](const HintField* f) { return f == nullptr || f->values.empty() ? nullptr : &f->values[0]; };

        // Compiled once per hint, not per (hint, table). A mapped table ignores `value`, but sel and
        // bus ARE the multiplicity.
        const Field val = compile(&fEx->values[0]);
        const Field sel = selConst1 ? Field{} : compile(first(fSel));
        const Field bus = dynBus ? compile(first(fld("busid"))) : Field{};

        for (const auto& dec : mulDecoders()) {
            bool feeds = false;
            for (const auto& ov : fOp->values)
                if (ov.operand == opType::number && ov.value == dec.table_id) { feeds = true; break; }
            if (!feeds) continue;
            // A K-element tuple needs an exact map; a 1-element tuple is a range check and uses
            // `bias`.
            if (fEx->values.size() != 1 && dec.mapSlots == 0)
                fatal(dec.table_id, "it sends a " + std::to_string(fEx->values.size())
                                    + "-element tuple and the table has no fitted row map");
            if (!sel.why.empty()) fatal(dec.table_id, "its selector does not compile (" + sel.why + ")");
            if (!bus.why.empty()) fatal(dec.table_id, "its bus id does not compile (" + bus.why + ")");
            if (!val.why.empty() && dec.mapSlots == 0)
                fatal(dec.table_id, "its value does not compile (" + val.why + ")");

            MulJobDev job{};
            // Counters live in the air hosting the table, not the air being committed.
            job.hostAirId   = dec.hostAirId;
            job.accBase     = dec.acc_base;
            job.nTableRows  = dec.n_rows;
            job.biasFE      = mulBiasFE(dec.bias);
            job.mapSlots    = dec.mapSlots;
            job.mapKV       = dec.mapKV;
            job.nKey        = dec.nKey;
            // An exact map looks the tuple up verbatim: one program per key column.
            if (dec.mapSlots != 0) {
                if (dec.nKey == 0 || dec.nKey > fEx->values.size()) {
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id) + " maps on "
                                + std::to_string(dec.nKey) + " columns but the lookup supplies "
                                + std::to_string(fEx->values.size()));
                    exitProcess();
                }
                for (uint32_t c = 0; c < dec.nKey; ++c) {
                    const Field key = compile(&fEx->values[c]);
                    if (!key.why.empty())
                        fatal(dec.table_id, "key column " + std::to_string(c) + " does not compile ("
                                            + key.why + ")");
                    job.keyProgOff[c] = key.off;
                    job.keyProgLen[c] = key.len;
                }
            }
            job.valProgOff  = val.off;
            job.valProgLen  = val.len;
            job.selProgOff  = sel.off;
            job.selProgLen  = sel.len;
            job.busProgOff  = bus.off;
            job.busProgLen  = bus.len;
            job.rows        = rows;
            job.tableId     = dec.table_id;
            job.selConstOne = selConst1 ? 1u : 0u;
            job.hasBus      = dynBus ? 1u : 0u;
            plan.jobs.push_back(job);
        }
    }
    for (const auto& in : plan.prog)
        for (const MulOperandDev* o : {&in.a, &in.b})
            if (o->kind == MUL_OPND_COL) plan.srcMask |= 1u << o->term.src;

    // Group jobs sharing a selector so the kernel's carry-across hits; order is otherwise free.
    std::stable_sort(plan.jobs.begin(), plan.jobs.end(), [](const MulJobDev& a, const MulJobDev& b) {
        return a.selProgOff < b.selProgOff;
    });
    for (const auto& j : plan.jobs) if (j.rows > plan.maxRows) plan.maxRows = j.rows;
    if (!plan.jobs.empty())
        zklog.trace("Multiplicity plan: reads=" + mulSrcMaskNames(plan.srcMask)
                   + " streamable=" + std::to_string((int)mulPlanStreamable(plan))
                   + " insns=" + std::to_string(plan.prog.size()));
    return plan;
}

// One plan per air, built on first use and kept for the process lifetime.
inline MulPlan& mulPlanFor(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId) {
    static std::map<std::pair<uint64_t,uint64_t>, MulPlan> plans;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(airgroupId, airId);
    auto it = plans.find(key);
    if (it == plans.end()) {
        it = plans.emplace(key, mulBuildPlan(setupCtx, airgroupId, airId)).first;
        if (!it->second.jobs.empty())
            zklog.trace("Multiplicity: air " + std::to_string(airgroupId) + "/" + std::to_string(airId)
                       + " -> " + std::to_string(it->second.jobs.size()) + " kernel jobs");
    }
    return it->second;
}

#endif
