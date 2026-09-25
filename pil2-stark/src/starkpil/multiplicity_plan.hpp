#ifndef MULTIPLICITY_PLAN_HPP
#define MULTIPLICITY_PLAN_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <mutex>
#include <set>
#include <array>
#include <algorithm>
#include "multiplicity_extract.hpp"
#include "multiplicity_job.hpp"
#include "multiplicity_job.hpp"
#include "multiplicity.hpp"
#include "setup_ctx.hpp"

// A lookup that did not compile, resolved so the interpreter path needs no hint walk.
struct MulFallbackJob {
    uint64_t   hintId = 0;
    MulDecoder dec {};
    uint64_t   rows = 0;
    uint8_t    selConstOne = 0;
    uint8_t    hasBus = 0;
};

// Per-air kernel plan, derived from setup alone and shared by every instance of the air.
struct MulPlan {
    std::vector<MulJobDev>      jobs;      // lookups the kernel handles
    // All compiled programs, concatenated; jobs hold offsets into this.
    std::vector<MulInsnDev>     prog;
    std::vector<MulFallbackJob> fallback;  // the rest, for the interpreter
    uint32_t                    nCols1 = 0;   // distinct stage-1 columns the jobs read
    bool                        packable = false;  // every term is cm1, a const pol, or a uniform
    // Bit per MulSrc read by the jobs. The streaming path gates on this, not on `packable`,
    // because a slot commit has no aux trace or custom commits on device.
    uint32_t                    srcMask = 0;
    uint64_t                    maxRows = 0;       // tallest job: the row-stationary grid height
    // How far a cm1 reference reaches off its own row, in rows. A tile must carry this many
    // extra rows on each side or those reads fall outside it; 0 for most airs, 4 for zisk
    // Keccakf, whose shifts hide inside the compiled programs.
    uint32_t                    traceHalo = 0;
};

// A tile carries this many extra rows on each side. Generous next to the 4 rows zisk's widest
// air reaches, and still negligible against MUL_STREAM_TILE_ROWS.
#define MUL_TILE_MAX_HALO 64u

inline bool mulPlanStreamable(const MulPlan& p) {
    // Const pols and the tile are resident by construction; publics and the three value pools are
    // staged into the hook's own per-slot buffer (see MulStreamCtx::hostVals), which is cheap
    // because PINNED_AUX_VALUES_MAX caps all four at 512 KB. Aux and the custom commits are
    // trace-sized and stay out.
    const uint32_t resident = (1u << MUL_SRC_CONST)   | (1u << MUL_SRC_TRACE)
                            | (1u << MUL_SRC_PUBLIC)  | (1u << MUL_SRC_AIRVALUE)
                            | (1u << MUL_SRC_PROOFVALUE) | (1u << MUL_SRC_AIRGROUPVALUE);
    // Compiled programs, exact maps and digit rules used to be refused here because the tile
    // had its own cut-down evaluator. It no longer has one -- a tile is the same kernel over a
    // different window -- and a `'`-shifted CM1 reference is served by giving that window a halo,
    // so the only thing left to refuse is a reach the halo would not cover.
    return p.packable && (p.srcMask & ~resident) == 0 && p.traceHalo <= MUL_TILE_MAX_HALO;
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
// lookup that does not compile goes to `fallback`: the compiler may give up, never guess.
inline MulPlan mulBuildPlan(SetupCtx& setupCtx) {
    MulPlan plan;
    std::map<std::string, uint32_t> progCache;
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

        // Degree zero in tuple and selector: counted once per instance, not per row. Read from
        // the hint, since an expression over airvalues is degree 0 too.
        const uint64_t rows = (fDx->values[0].value == 0 && fDs->values[0].value == 0) ? 1ULL : nRows;
        const HintField* fSel = fld("num_reps");
        const bool selConst1 = isNum(fSel) && fSel->values[0].value == 1;
        // Several opids, or a computed busid, means the bus is chosen per row.
        const bool dynBus = fOp->values.size() > 1 || (fld("busid") != nullptr && !isNum(fld("busid")));

        // Compile the three forms once per hint, not per (hint, table), so one failed table
        // cannot re-count the others. value/sel/bus get separate `ok`s: a mapped table ignores
        // `value`, but sel and bus ARE the multiplicity.
        const char* whichField = "value";
        const HintFieldValue* failed = nullptr;
        // Append a program to `plan.prog`, deduplicated by instruction bytes (call sites get
        // fresh expression ids). Identical programs must share an offset so the kernel can
        // evaluate a shared selector once for a run of jobs.
        auto place = [&](const MulProgram& pg, uint32_t& off, uint32_t& len) {
            const std::string body((const char*)pg.insns.data(), pg.insns.size() * sizeof(MulInsnDev));
            auto seen = progCache.find(body);
            if (seen != progCache.end()) {
                off = seen->second;
                len = (uint32_t)pg.insns.size();
                return true;
            }
            off = (uint32_t)plan.prog.size();
            len = (uint32_t)pg.insns.size();
            plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
            progCache[body] = off;
            return true;
        };

        auto compile = [&](const HintField* f, uint32_t& off, uint32_t& len) {
            off = len = 0;
            if (f == nullptr || f->values.empty()) return false;
            MulProgram pg;
            if (!mulCompileField(setupCtx, f->values[0], bufferCommitSize, nRows, pg)) {
                failed = &f->values[0];
                return false;
            }
            return place(pg, off, len);
        };

        uint32_t valPOff = 0, valPLen = 0, selPOff = 0, selPLen = 0, busPOff = 0, busPLen = 0;
        const bool okValue = compile(fEx, valPOff, valPLen);
        bool okSel = true, okBus = true;
        if (okValue && !selConst1) { whichField = "sel"; okSel = compile(fSel, selPOff, selPLen); }
        if (okValue && okSel && dynBus) { whichField = "bus"; okBus = compile(fld("busid"), busPOff, busPLen); }
        const bool ok = okValue && okSel && okBus;
        // Only the GPU commit path has the interpreter, so say why a hint lands there.
        if (!ok)
            zklog.warning(std::string("multiplicity: a lookup falls back to the interpreter -- field ")
                          + whichField
                          + ", operand kind " + std::to_string(failed ? (int)failed->operand : -1)
                          + ", expression " + std::to_string(failed ? (long long)failed->id : -1)
                          + ", rows " + std::to_string(rows)
                          + ", reason " + (mulProgFailReason() ? mulProgFailReason() : "unknown"));

        for (const auto& dec : mulDecoders()) {
            bool feeds = false;
            for (const auto& ov : fOp->values)
                if (ov.operand == opType::number && ov.value == dec.table_id) { feeds = true; break; }
            if (!feeds) continue;
            // A K-element tuple needs a row map (affine fit, exact map or digit rule); a
            // 1-element tuple is a range check and uses `bias`.
            if (fEx->values.size() != 1 && dec.nCoef == 0 && dec.mapSlots == 0
                && dec.digitCols == 0) {
                static std::set<uint32_t> warned;
                if (warned.insert(dec.table_id).second)
                    zklog.trace("multiplicity: table " + std::to_string(dec.table_id) + " has a "
                               + std::to_string(fEx->values.size()) + "-element tuple and no fitted "
                               "row map; left to the std to count");
                continue;
            }
            // The fit is over the table's columns; a lookup may supply fewer if every omitted
            // column has coefficient zero.
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
            // A mapped table does not need `value` (and the interpreter has no map support);
            // sel and bus it always needs.
            if (!okSel || !okBus || (!okValue && dec.mapSlots == 0)) {
                plan.fallback.push_back({hints[i], dec, rows,
                                         (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                continue;
            }

            // Counters live in the air hosting the table, not the air being committed.
            uint64_t hostAirId = UINT64_MAX;
            for (const auto& L : mulVtLayouts())
                if (L.accBase.count(dec.table_id)) { hostAirId = L.airId; break; }
            if (hostAirId == UINT64_MAX) continue;

            // A fitted table's row is `sum(coef * element) + konst`; that fold replaces the
            // field's value program.
            uint32_t valProgOffForTable = valPOff, valProgLenForTable = valPLen;
            // coef == [1] is the range-check case: reuse the value program with konst as bias.
            uint64_t biasForTable = dec.nCoef != 0 ? 0ULL : mulBiasFE(dec.bias);
            if (dec.nCoef == 1 && dec.coef[0] == 1) {
                biasForTable = dec.konst;
            } else if (dec.nCoef != 0) {
                MulProgram foldPg;
                if (!mulCompileFold(setupCtx, fEx->values, dec.coef, nFold, dec.konst,
                                    bufferCommitSize, nRows, foldPg)
                    || !place(foldPg, valProgOffForTable, valProgLenForTable)) {
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
            }

            MulJobDev job{};
            job.hostAirId   = hostAirId;
            job.accBase     = dec.acc_base;
            job.nTableRows  = dec.n_rows;
            job.biasFE      = biasForTable;
            job.mapSlots    = dec.mapSlots;
            job.mapKV       = dec.mapKV;
            job.digitTab    = dec.digitTab;
            job.digitCols   = dec.digitCols;
            job.nKey        = dec.nKey;
            // An exact map looks the tuple up verbatim: one program per key column.
            auto keyProg = [&](uint32_t slot, uint32_t src) {
                if (src >= fEx->values.size()) {
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                + " keys on column " + std::to_string(src)
                                + " but the lookup supplies " + std::to_string(fEx->values.size()));
                    exitProcess();
                }
                MulProgram pg;
                return mulCompileField(setupCtx, fEx->values[src], bufferCommitSize, nRows, pg)
                    && place(pg, job.keyProgOff[slot], job.keyProgLen[slot]);
            };
            if (dec.digitCols != 0) {
                // Only the columns the rule reads, in its own order.
                job.nKey = dec.digitCols;
                bool keyOk = true;
                for (uint32_t c = 0; c < dec.digitCols && keyOk; ++c) keyOk = keyProg(c, dec.digitCol[c]);
                if (!keyOk) {
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
            } else if (dec.mapSlots != 0) {
                if (dec.nKey == 0 || dec.nKey > fEx->values.size()) {
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id) + " maps on "
                                + std::to_string(dec.nKey) + " columns but the lookup supplies "
                                + std::to_string(fEx->values.size()));
                    exitProcess();
                }
                bool keyOk = true;
                uint32_t badCol = 0;
                for (uint32_t c = 0; c < dec.nKey && keyOk; ++c) { badCol = c; keyOk = keyProg(c, c); }
                if (!keyOk) {
                    static std::set<uint32_t> warned;
                    if (warned.insert(dec.table_id).second)
                        zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                    + " is mapped but key column " + std::to_string(badCol) + " of "
                                    + std::to_string(dec.nKey) + " does not compile (reason: "
                                    + (mulProgFailReason() ? mulProgFailReason() : "unknown")
                                    + ") -- routed to the interpreter");
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
            }
            job.valProgOff  = valProgOffForTable;
            job.valProgLen  = valProgLenForTable;
            job.selProgOff  = selPOff;
            job.selProgLen  = selPLen;
            job.busProgOff  = busPOff;
            job.busProgLen  = busPLen;
            job.rows        = rows;
            job.tableId     = dec.table_id;
            job.selConstOne = selConst1 ? 1u : 0u;
            job.hasBus      = dynBus ? 1u : 0u;
            plan.jobs.push_back(job);
        }
    }
    // Stage-1 columns read, and whether the packed trace alone serves this air.
    {
        std::set<uint32_t> cols;
        bool ok = true;
        // Sources must come from the compiled operands, or mulPlanStreamable sees none.
        for (const auto& in : plan.prog)
            for (const MulOperandDev* o : {&in.a, &in.b}) {
                if (o->kind != MUL_OPND_COL) continue;
                plan.srcMask |= 1u << o->term.src;
                if (o->term.src == MUL_SRC_TRACE) cols.insert(o->term.col);
                else if (o->term.src == MUL_SRC_AUX) ok = false;
            }

        // Group jobs sharing a selector so the kernel's carry-across hits; order is otherwise free.
        std::stable_sort(plan.jobs.begin(), plan.jobs.end(), [](const MulJobDev& a, const MulJobDev& b) {
            return a.selProgOff < b.selProgOff;
        });
        plan.nCols1 = (uint32_t)cols.size();
        plan.packable = ok;
        for (const auto& j : plan.jobs) if (j.rows > plan.maxRows) plan.maxRows = j.rows;
        // The halo. Every field is a program, so its operands are the only place a cm1 reference
        // can hide -- scanning anything less is what once made Keccakf look shift-free while 1720
        // of its operands were shifted.
        for (const auto& in : plan.prog)
            for (const MulOperandDev* o : {&in.a, &in.b})
                if (o->kind == MUL_OPND_COL && o->term.src == MUL_SRC_TRACE)
                    plan.traceHalo = std::max(plan.traceHalo, (uint32_t)std::abs((int)o->term.rowStride));

        if (!plan.jobs.empty())
            zklog.trace("Multiplicity plan: cm1 cols=" + std::to_string(plan.nCols1)
                       + " packable=" + std::to_string((int)plan.packable)
                       + " reads=" + mulSrcMaskNames(plan.srcMask)
                       + " streamable=" + std::to_string((int)mulPlanStreamable(plan))
                       + " insns=" + std::to_string(plan.prog.size()));
    }
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
        it = plans.emplace(key, mulBuildPlan(setupCtx)).first;
        if (!it->second.jobs.empty() || !it->second.fallback.empty())
            zklog.trace("Multiplicity: air " + std::to_string(airgroupId) + "/" + std::to_string(airId)
                       + " -> " + std::to_string(it->second.jobs.size()) + " kernel jobs, "
                       + std::to_string(it->second.fallback.size()) + " on the interpreter");
    }
    return it->second;
}

#endif
