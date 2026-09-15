#ifndef MULTIPLICITY_PLAN_HPP
#define MULTIPLICITY_PLAN_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <mutex>
#include <set>
#include "multiplicity_extract.hpp"
#include "multiplicity_job.hpp"
#include "multiplicity.hpp"
#include "setup_ctx.hpp"

// What the dedicated kernel needs for one air, derived from setup alone: no trace, no challenges,
// nothing that changes between instances. Built once per air and reused by every instance of it.
// A lookup the extractor could not reduce, resolved so the interpreter path needs no hint walk.
struct MulFallbackJob {
    uint64_t   hintId = 0;
    MulDecoder dec {};
    uint64_t   rows = 0;
    uint8_t    selConstOne = 0;
    uint8_t    hasBus = 0;
};

struct MulPlan {
    std::vector<MulJobDev>      jobs;      // lookups the kernel handles
    std::vector<MulFallbackJob> fallback;  // the rest, for the interpreter
    // The distinct stage-1 columns the jobs read, sorted. The streaming commit never materialises
    // cm1 -- it unpacks a few columns at a time and LDEs them in place -- so serving that path
    // means gathering exactly these columns for a tile of rows. Empty when nothing reads cm1.
    std::vector<uint32_t>       cols1;
    bool                        packable = false;  // every term is cm1, a const pol, or a uniform
    // Bit per MulSrc actually read by the jobs. A slot commit has only the const pols and the
    // tile it unpacks -- no aux trace, so no publics/value pools/custom commits live on device
    // there. `packable` alone does not say that (it only rules out later stages), so the streaming
    // path gates on this instead.
    uint32_t                    srcMask = 0;
    uint64_t                    maxRows = 0;       // tallest job: the row-stationary grid height
};

inline bool mulPlanStreamable(const MulPlan& p) {
    const uint32_t resident = (1u << MUL_SRC_CONST) | (1u << MUL_SRC_TRACE);
    return p.packable && (p.srcMask & ~resident) == 0;
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

// A term's address, resolved: every prover buffer is flat column-major (goldilocks_trace_layout.cuh),
// so a column reference is one element offset plus the row.
inline bool mulTermToDev(SetupCtx& setupCtx, const MulLinTerm& t, uint64_t domainSize, MulTermDev& out) {
    (void)domainSize;
    const auto& si = setupCtx.starkInfo;
    const uint32_t base = (uint32_t)(1 + si.nStages + 3 + si.customCommits.size());
    out = MulTermDev{};
    out.coef = t.coef;

    if (mulIsUniformType(t.type, base)) {       // publics and the three value pools: no row term
        out.src = t.type == base + 2 ? MUL_SRC_PUBLIC
                : t.type == base + 4 ? MUL_SRC_AIRVALUE
                : t.type == base + 5 ? MUL_SRC_PROOFVALUE
                                     : MUL_SRC_AIRGROUPVALUE;
        out.sectionOffset = t.argIdx;           // a flat index into the pool
        return true;
    }

    const uint32_t customBase = (uint32_t)(si.nStages + 4);
    if (t.type >= customBase && t.type < customBase + si.customCommits.size()) {
        const uint64_t idx = t.type - customBase;
        const std::string sec = si.customCommits[idx].name + "0";
        auto it = si.mapOffsets.find(std::make_pair(sec, false));
        auto in = si.mapSectionsN.find(sec);
        if (it == si.mapOffsets.end() || in == si.mapSectionsN.end()) return false;
        if (t.rowOff >= si.openingPoints.size()) return false;
        out.rowStride = si.verify ? 0 : (int32_t)si.openingPoints[t.rowOff];
        out.src = MUL_SRC_CUSTOM;
        out.sectionOffset = it->second;
        out.col = t.argIdx;
        out.nCols = (uint32_t)in->second;
        return true;
    }
    if (t.rowOff >= si.openingPoints.size()) return false;
    out.rowStride = si.verify ? 0 : (int32_t)si.openingPoints[t.rowOff];
    out.col = t.argIdx;
    if (t.type == 0) {                          // const pols: their own buffer, no offset
        out.src = MUL_SRC_CONST;
        out.nCols = (uint32_t)si.nConstants;
    } else if (t.type == 1) {                   // stage 1: the trace buffer, no offset
        out.src = MUL_SRC_TRACE;
        out.nCols = (uint32_t)si.mapSectionsN.at("cm1");
    } else {                                    // later stages: a section of aux_trace
        const std::string sec = "cm" + std::to_string(t.type);
        auto it = si.mapOffsets.find(std::make_pair(sec, false));
        auto in = si.mapSectionsN.find(sec);
        if (it == si.mapOffsets.end() || in == si.mapSectionsN.end()) return false;
        out.src = MUL_SRC_AUX;
        out.sectionOffset = it->second;
        out.nCols = (uint32_t)in->second;
    }
    return true;
}

inline bool mulFormToDev(SetupCtx& setupCtx, const MulLinForm& f, uint64_t domainSize, MulFormDev& out) {
    if (!f.ok) return false;
    out = MulFormDev{};
    out.konst = f.konst;
    out.n = f.n;
    for (uint8_t i = 0; i < f.n; ++i)
        if (!mulTermToDev(setupCtx, f.t[i], domainSize, out.t[i])) return false;
    out.hasProduct = f.hasProduct ? 1u : 0u;
    out.konst2 = f.konst2;
    out.n2 = f.n2;
    for (uint8_t i = 0; i < f.n2; ++i)
        if (!mulTermToDev(setupCtx, f.t2[i], domainSize, out.t2[i])) return false;
    return true;
}

// Walk `gsum_debug_data` once per air and turn every range-check lookup it feeds into a job. A
// lookup that does not reduce to a linear form goes to `fallback`: the extractor may give up,
// never guess.
inline MulPlan mulBuildPlan(SetupCtx& setupCtx) {
    MulPlan plan;
    const uint64_t n = setupCtx.expressionsBin.getNumberHintIdsByName("gsum_debug_data");
    if (n == 0) return plan;
    std::vector<uint64_t> hints(n);
    setupCtx.expressionsBin.getHintIdsByName(hints.data(), "gsum_debug_data");

    const uint64_t nRows = 1ULL << setupCtx.starkInfo.starkStruct.nBits;
    const uint64_t bufferCommitSize = 1 + setupCtx.starkInfo.nStages + 3
                                    + setupCtx.starkInfo.customCommits.size();

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

        // Degree zero in both tuple and selector means the term does not vary along the trace: it
        // contributes once per instance, not once per row. Read from the hint rather than guessed
        // from the operand -- an expression over airvalues is degree 0 too.
        const uint64_t rows = (fDx->values[0].value == 0 && fDs->values[0].value == 0) ? 1ULL : nRows;
        const HintField* fSel = fld("num_reps");
        const bool selConst1 = isNum(fSel) && fSel->values[0].value == 1;
        // Several opids, or a computed busid, means the bus is chosen per row.
        const bool dynBus = fOp->values.size() > 1 || (fld("busid") != nullptr && !isNum(fld("busid")));

        // The three forms do not depend on which table the hint feeds, so extract them ONCE:
        // deciding per (hint, table) would let one failed table re-count the ones that succeeded.
        MulFormDev value{}, selF{}, busF{};
        const char* whichField = "value";
        const HintField* failed = nullptr;
        auto extract = [&](const HintField* f, MulFormDev& out) {
            failed = f;
            return f != nullptr && !f->values.empty()
                && mulFormToDev(setupCtx, mulExtractField(setupCtx, f->values[0], bufferCommitSize), nRows, out);
        };
        bool ok = extract(fEx, value);
        if (ok && !selConst1) { whichField = "sel"; ok = extract(fSel, selF); }
        if (ok && dynBus)     { whichField = "bus"; ok = extract(fld("busid"), busF); }
        // The fallback is the interpreter, which only the GPU commit path has -- so a hint that
        // lands here is uncounted anywhere else. Say why, because closing the gap means extending
        // the extractor rather than discovering the shortfall in a proof that will not close.
        if (!ok)
            zklog.warning(std::string("multiplicity: a lookup falls back to the interpreter -- field ")
                          + whichField + ", operand kind "
                          + std::to_string(failed && !failed->values.empty()
                                           ? (int)failed->values[0].operand : -1)
                          + ", expression " + std::to_string(failed && !failed->values.empty()
                                                             ? (long long)failed->values[0].id : -1)
                          + ", reason " + (mulWalkFailReason() ? mulWalkFailReason() : "operand kind not modelled")
                          + ", rows " + std::to_string(rows));

        for (const auto& dec : mulDecoders()) {
            bool feeds = false;
            for (const auto& ov : fOp->values)
                if (ov.operand == opType::number && ov.value == dec.table_id) { feeds = true; break; }
            if (!feeds) continue;
            // A K-element tuple needs a fitted map; a 1-element one is the range-check case and
            // uses `bias`. A tuple with no fitted map is nobody's to count here.
            if (fEx->values.size() != 1 && dec.nCoef == 0) {
                static std::set<uint32_t> warned;
                if (warned.insert(dec.table_id).second)
                    zklog.info("multiplicity: table " + std::to_string(dec.table_id) + " has a "
                               + std::to_string(fEx->values.size()) + "-element tuple and no fitted "
                               "row map; left to the std to count");
                continue;
            }
            // The fit is over the TABLE's columns; a lookup may supply fewer, which is fine as long
            // as every column it omits was found irrelevant (coefficient zero). A range check is
            // exactly this: one value against a two-column group whose second column is not a key.
            uint8_t nFold = dec.nCoef;
            if (dec.nCoef != 0) {
                if (fEx->values.size() > dec.nCoef) {
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id) + " fitted over "
                                + std::to_string((int)dec.nCoef) + " columns but a lookup supplies "
                                + std::to_string(fEx->values.size()) + " -- the fit is not for this table");
                    exitProcess();
                }
                nFold = (uint8_t)fEx->values.size();
                for (uint8_t e = nFold; e < dec.nCoef; ++e) {
                    if (dec.coef[e] == 0) continue;
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id) + " needs column "
                                + std::to_string((int)e) + " to address a row, but the lookup supplies "
                                "only " + std::to_string((int)nFold) + " elements");
                    exitProcess();
                }
            }
            if (!ok) {
                plan.fallback.push_back({hints[i], dec, rows,
                                         (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                continue;
            }

            // The counters live in the air that HOSTS the table, which is not the air being
            // committed: one instance can feed tables belonging to different airs.
            uint64_t hostAirId = UINT64_MAX;
            for (const auto& L : mulVtLayouts())
                if (L.accBase.count(dec.table_id)) { hostAirId = L.airId; break; }
            if (hostAirId == UINT64_MAX) continue;

            // Fold a fitted tuple map into the single linear form the kernel already evaluates:
            // sum(coef[j] * element_j) + konst is itself linear, so no new kernel, no new job
            // shape, and the CPU mirror follows for free. A coefficient of zero marks a column the
            // fit found irrelevant (an output, not part of the key) and costs nothing to skip.
            MulFormDev valueForTable = value;
            if (dec.nCoef != 0) {
                MulLinForm folded = mulLinConst(dec.konst);
                bool foldOk = true;
                for (uint8_t e = 0; e < nFold && foldOk; ++e) {
                    if (dec.coef[e] == 0) continue;
                    MulLinForm el = mulExtractField(setupCtx, fEx->values[e], bufferCommitSize);
                    MulLinForm scaled{};
                    foldOk = el.ok && mulLinScale(scaled, el, dec.coef[e])
                          && mulLinAdd(folded, folded, scaled, false);
                }
                if (!foldOk || !mulFormToDev(setupCtx, folded, nRows, valueForTable)) {
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
            }

            MulJobDev job{};
            job.hostAirId   = hostAirId;
            job.value       = valueForTable;
            job.sel         = selF;
            job.bus         = busF;
            job.accBase     = dec.acc_base;
            job.nTableRows  = dec.n_rows;
            job.biasFE      = dec.nCoef != 0 ? 0ULL : mulBiasFE(dec.bias);
            job.keyMin      = dec.keyMin;
            job.indexLen    = dec.indexLen;
            job.index       = dec.index;
            job.baseIn      = dec.baseIn;
            job.baseOut     = dec.baseOut;
            job.nDigits     = dec.nDigits;
            for (uint32_t d = 0; d < MUL_MAX_DIGIT_BASE; ++d) job.digitMap[d] = dec.digitMap[d];
            job.rows        = rows;
            job.tableId     = dec.table_id;
            job.selConstOne = selConst1 ? 1u : 0u;
            job.hasBus      = dynBus ? 1u : 0u;
            plan.jobs.push_back(job);
        }
    }
    // Columns the jobs read from stage 1, and whether this air could be served from the packed
    // trace alone (nothing reads a later stage, which is true by construction at commit time).
    {
        std::set<uint32_t> cols;
        bool ok = true;
        for (const auto& j : plan.jobs)
            for (const MulFormDev* f : {&j.value, &j.sel, &j.bus})
                {
                    for (uint32_t k = 0; k < f->n; ++k) {
                        plan.srcMask |= 1u << f->t[k].src;
                        if (f->t[k].src == MUL_SRC_TRACE) cols.insert(f->t[k].col);
                        else if (f->t[k].src == MUL_SRC_AUX) ok = false;
                    }
                    for (uint32_t k = 0; k < f->n2; ++k) {
                        plan.srcMask |= 1u << f->t2[k].src;
                        if (f->t2[k].src == MUL_SRC_TRACE) cols.insert(f->t2[k].col);
                        else if (f->t2[k].src == MUL_SRC_AUX) ok = false;
                    }
                }
        plan.cols1.assign(cols.begin(), cols.end());
        plan.packable = ok;
        for (const auto& j : plan.jobs) if (j.rows > plan.maxRows) plan.maxRows = j.rows;
        uint32_t shifted = 0;
        for (const auto& j : plan.jobs)
            for (const MulFormDev* f : {&j.value, &j.sel, &j.bus})
                {
                    for (uint32_t k = 0; k < f->n; ++k)
                        if (f->t[k].src == MUL_SRC_TRACE && f->t[k].rowStride != 0) ++shifted;
                    for (uint32_t k = 0; k < f->n2; ++k)
                        if (f->t2[k].src == MUL_SRC_TRACE && f->t2[k].rowStride != 0) ++shifted;
                }
        if (!plan.jobs.empty())
            zklog.info("Multiplicity plan: cm1 cols=" + std::to_string(plan.cols1.size())
                       + " packable=" + std::to_string((int)plan.packable)
                       + " shiftedTerms=" + std::to_string(shifted)
                       + " reads=" + mulSrcMaskNames(plan.srcMask)
                       + " streamable=" + std::to_string((int)mulPlanStreamable(plan)));
    }
    return plan;
}

// One plan per air, and one device copy per (air, gpu). Both are setup-derived, so they outlive
// every instance and are built on first use.
inline MulPlan& mulPlanFor(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId) {
    static std::map<std::pair<uint64_t,uint64_t>, MulPlan> plans;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(airgroupId, airId);
    auto it = plans.find(key);
    if (it == plans.end()) {
        it = plans.emplace(key, mulBuildPlan(setupCtx)).first;
        if (!it->second.jobs.empty() || !it->second.fallback.empty())
            zklog.info("Multiplicity: air " + std::to_string(airgroupId) + "/" + std::to_string(airId)
                       + " -> " + std::to_string(it->second.jobs.size()) + " kernel jobs, "
                       + std::to_string(it->second.fallback.size()) + " on the interpreter");
    }
    return it->second;
}

#endif
