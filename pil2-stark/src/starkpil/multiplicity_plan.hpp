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
#include "multiplicity_program.hpp"
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

// Per-air kernel plan, derived from setup alone and shared by every instance of the air.
struct MulPlan {
    std::vector<MulJobDev>      jobs;      // lookups the kernel handles
    // Instructions for the fields no closed form expresses, concatenated; jobs hold offsets into
    // this. One buffer per air so the device needs a single allocation.
    std::vector<MulInsnDev>     prog;
    std::vector<MulFallbackJob> fallback;  // the rest, for the interpreter
    // The distinct stage-1 columns the jobs read, sorted. The streaming commit never materialises
    // cm1 -- it unpacks a few columns at a time and LDEs them in place -- so serving that path
    // means gathering exactly these columns for a tile of rows. Empty when nothing reads cm1.
    std::vector<uint32_t>       cols1;
    bool                        packable = false;  // every term is cm1, a const pol, or a uniform
    // Bit per MulSrc read by the jobs. The streaming path gates on this, not on `packable`,
    // because a slot commit has no aux trace or custom commits on device.
    uint32_t                    srcMask = 0;
    uint64_t                    maxRows = 0;       // tallest job: the row-stationary grid height
};

inline bool mulPlanStreamable(const MulPlan& p) {
    const uint32_t resident = (1u << MUL_SRC_CONST) | (1u << MUL_SRC_TRACE);
    // The tile kernel only knows the affine row map: it computes `value + bias` directly and never
    // resolves a tuple. A compiled program, an exact map or a digit rule would all fall through it
    // and count the wrong row -- or nothing. Refusing the streaming path here keeps that a routing
    // decision rather than a silent shortfall.
    for (const auto& j : p.jobs)
        if (j.mapSlots != 0 || j.digitCols != 0 || j.hasIndexedBase) return false;
    return p.packable && (p.srcMask & ~resident) == 0 && p.prog.empty();
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


inline bool mulFormToDev(SetupCtx& setupCtx, const MulLinForm& f, uint64_t domainSize, MulFormDev& out) {
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

// A sum of products lowers to an array of the single-product form the kernel already evaluates.
inline bool mulPolyToDev(SetupCtx& setupCtx, const MulPoly& q, uint64_t domainSize, MulPolyDev& out) {
    if (!q.ok || q.nProd == 0 || q.nProd > MUL_MAX_PRODUCTS) return false;
    out = MulPolyDev{};
    out.nProd = q.nProd;
    for (uint8_t i = 0; i < q.nProd; ++i)
        if (!mulFormToDev(setupCtx, q.p[i], domainSize, out.p[i])) return false;
    return true;
}

// Walk `gsum_debug_data` once per air and turn every range-check lookup it feeds into a job. A
// lookup that does not reduce to a linear form goes to `fallback`: the extractor may give up,
// never guess.
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

        // The three forms do not depend on which table the hint feeds, so extract them ONCE:
        // deciding per (hint, table) would let one failed table re-count the ones that succeeded.
        MulPolyDev value{}, selF{}, busF{};
        const char* whichField = "value";
        const HintField* failed = nullptr;
        auto extract = [&](const HintField* f, MulPolyDev& out) {
            failed = f;
            return f != nullptr && !f->values.empty()
                && mulPolyToDev(setupCtx, mulExtractField(setupCtx, f->values[0], bufferCommitSize), nRows, out);
        };
        // Kept apart on purpose. A mapped table evaluates its key columns instead of the folded
        // `value`, so a value the extractor cannot reduce costs it nothing -- but a selector or bus
        // id it cannot reduce is the multiplicity itself, and proceeding would count with a
        // default-constructed form. One `ok` for all three hid that: raising MUL_MAX_PRODUCTS moved
        // the failures from `value` to `sel` and the counts went wrong with no diagnostic.
        // Second rung: compile the expression as the PIL wrote it. Appends to `plan.prog` and
        // hands back the (offset, length) the job carries. Only what this cannot address at all
        // -- dim3 temporaries, challenges -- reaches the interpreter.
        // Memoised by the expression it compiles: the lookups of one air share selectors heavily
        // (Keccakf's 22 all read the same clock predicate), and an identical program must land at
        // the same offset or the kernel cannot tell that two jobs evaluate the same thing.
        // Deduplicated by the instructions themselves, not by the expression they came from.
        // `lookup_assumes` gets a fresh expression id at every call site, so Keccakf's 22 xor5
        // lookups -- which all pass the SAME `rounds_active` selector -- compile to 22 byte-identical
        // 121-instruction programs under distinct ids. Keying on content collapses them to one
        // offset, which is what lets the kernel evaluate the selector once for the whole run of
        // jobs instead of 22 times a row.
        auto compile = [&](const HintField* f, uint32_t& off, uint32_t& len) {
            off = len = 0;
            if (f == nullptr || f->values.empty()) return false;
            MulProgram pg;
            if (!mulCompileField(setupCtx, f->values[0], bufferCommitSize, nRows, pg)) return false;

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

        uint32_t valPOff = 0, valPLen = 0, selPOff = 0, selPLen = 0, busPOff = 0, busPLen = 0;
        bool okValue = extract(fEx, value);
        if (!okValue) okValue = compile(fEx, valPOff, valPLen);
        bool okSel = true, okBus = true;
        if (!selConst1) {
            whichField = "sel";
            okSel = extract(fSel, selF);
            if (!okSel) okSel = compile(fSel, selPOff, selPLen);
        }
        if (okSel && dynBus) {
            whichField = "bus";
            okBus = extract(fld("busid"), busF);
            if (!okBus) okBus = compile(fld("busid"), busPOff, busPLen);
        }
        const bool ok = okValue && okSel && okBus;
        if (ok) whichField = "value";
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
                          + ", rows " + std::to_string(rows)
                          + ", nOps " + std::to_string(mulWalkNOps()) + ", nTemp " + std::to_string(mulWalkNTemp())
                          + ", program rung also failed: "
                          + (mulProgFailReason() ? mulProgFailReason() : "unknown"));

        for (const auto& dec : mulDecoders()) {
            bool feeds = false;
            for (const auto& ov : fOp->values)
                if (ov.operand == opType::number && ov.value == dec.table_id) { feeds = true; break; }
            if (!feeds) continue;
            // A K-element tuple needs a row map (affine fit, exact map or digit rule); a
            // 1-element tuple is a range check and uses `bias`.
            if (fEx->values.size() != 1 && dec.nCoef == 0 && dec.mapSlots == 0 && dec.digitCols == 0
                && dec.nSel == 0) {
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
            // A mapped table evaluates its key columns instead of the folded `value`, so a value
            // the extractor cannot reduce costs it nothing -- that is why table 331, which reached
            // the fallback this way, contributed zero. The selector and the bus id are a different
            // matter: they ARE the multiplicity, and a job built with an unreduced one counts with
            // a default-constructed form. Sharing a single `ok` across all three hid that.
            if (!okSel || !okBus || (!okValue && dec.mapSlots == 0 && dec.nSel == 0)) {
                plan.fallback.push_back({hints[i], dec, rows,
                                         (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                continue;
            }

            // Counters live in the air hosting the table, not the air being committed.
            uint64_t hostAirId = UINT64_MAX;
            for (const auto& L : mulVtLayouts())
                if (L.accBase.count(dec.table_id)) { hostAirId = L.airId; break; }
            if (hostAirId == UINT64_MAX) continue;

            // Fold a fitted tuple map into the single linear form the kernel already evaluates:
            // sum(coef[j] * element_j) + konst is itself linear, so no new kernel, no new job
            // shape, and the CPU mirror follows for free. A coefficient of zero marks a column the
            // fit found irrelevant (an output, not part of the key) and costs nothing to skip.
            MulPolyDev valueForTable = value;
            uint32_t valProgOffForTable = valPOff, valProgLenForTable = valPLen;
            if (dec.nCoef != 0) {
                MulPoly folded = mulPolyOf(mulLinConst(dec.konst));
                bool foldOk = true;
                for (uint8_t e = 0; e < nFold && foldOk; ++e) {
                    if (dec.coef[e] == 0) continue;
                    MulPoly el = mulExtractField(setupCtx, fEx->values[e], bufferCommitSize);
                    MulPoly scaled{};
                    foldOk = el.ok && mulPolyScale(scaled, el, dec.coef[e])
                          && mulPolyAdd(folded, folded, scaled, false);
                }
                if (!foldOk || !mulPolyToDev(setupCtx, folded, nRows, valueForTable)) {
                    // A fold over several elements has no single expression to compile; a
                    // one-element one is just that element, so the program rung still applies.
                    uint32_t fOff = 0, fLen = 0;
                    if (nFold != 1 || !compile(fEx, fOff, fLen)) {
                        plan.fallback.push_back({hints[i], dec, rows,
                                                 (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                        continue;
                    }
                    valProgOffForTable = fOff;
                    valProgLenForTable = fLen;
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
            job.mapSlots    = dec.mapSlots;
            job.mapKV       = dec.mapKV;
            job.digitTab    = dec.digitTab;
            job.digitCols   = dec.digitCols;
            job.nKey        = dec.nKey;
            // An exact map looks the lookup's own tuple up verbatim, so the job carries one form
            // per key column rather than the single folded value the affine path uses.
            if (dec.digitCols != 0) {
                // Only the columns the rule reads, in its own order.
                job.nKey = dec.digitCols;
                bool keyOk = true;
                for (uint32_t c = 0; c < dec.digitCols && keyOk; ++c) {
                    const uint32_t src = dec.digitCol[c];
                    if (src >= fEx->values.size()) {
                        zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                    + " separable over column " + std::to_string(src)
                                    + " but the lookup supplies " + std::to_string(fEx->values.size()));
                        exitProcess();
                    }
                    keyOk = mulPolyToDev(setupCtx,
                                         mulExtractField(setupCtx, fEx->values[src], bufferCommitSize),
                                         nRows, job.key[c]);
                    if (!keyOk) {
                        MulProgram pg;
                        if (mulCompileField(setupCtx, fEx->values[src], bufferCommitSize, nRows, pg)) {
                            job.keyProgOff[c] = (uint32_t)plan.prog.size();
                            job.keyProgLen[c] = (uint32_t)pg.insns.size();
                            plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
                            keyOk = true;
                        }
                    }
                }
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
                for (uint32_t c = 0; c < dec.nKey && keyOk; ++c) {
                    badCol = c;
                    keyOk = mulPolyToDev(setupCtx,
                                         mulExtractField(setupCtx, fEx->values[c], bufferCommitSize),
                                         nRows, job.key[c]);
                    if (!keyOk) {
                        MulProgram pg;
                        if (mulCompileField(setupCtx, fEx->values[c], bufferCommitSize, nRows, pg)) {
                            job.keyProgOff[c] = (uint32_t)plan.prog.size();
                            job.keyProgLen[c] = (uint32_t)pg.insns.size();
                            plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
                            keyOk = true;
                        }
                    }
                }
                if (!keyOk) {
                    // The closed-form extractor cannot reduce this key, so evaluate it with the
                    // general interpreter instead. Slower, but it needs no closed form at all and
                    // never needs extending again -- which is the point of having this rung.
                    static std::set<uint32_t> warned;
                    if (warned.insert(dec.table_id).second)
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                + " is mapped but key column " + std::to_string(badCol) + " of "
                                + std::to_string(dec.nKey) + " does not reduce to a linear form"
                                + " (reason: " + (mulWalkFailReason() ? mulWalkFailReason()
                                                                     : "operand kind not modelled")
                                + ") -- routed to the interpreter");
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
            } else if (dec.nSel != 0) {
                // The selector and stride column indices are defined over the lookup's FULL tuple
                // (see vt_rule.rs), unlike the digit rule's pruned-to-what-it-reads columns -- so
                // every tuple element the lookup supplies must land in job.key at its own index.
                if (fEx->values.size() > MUL_MAX_TUPLE) {
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                + " indexed-base tuple has " + std::to_string(fEx->values.size())
                                + " elements, more than the tuple holds");
                    exitProcess();
                }
                job.nKey = (uint32_t)fEx->values.size();
                bool keyOk = true;
                uint32_t badCol = 0;
                for (uint32_t c = 0; c < job.nKey && keyOk; ++c) {
                    badCol = c;
                    keyOk = mulPolyToDev(setupCtx,
                                         mulExtractField(setupCtx, fEx->values[c], bufferCommitSize),
                                         nRows, job.key[c]);
                    if (!keyOk) {
                        MulProgram pg;
                        if (mulCompileField(setupCtx, fEx->values[c], bufferCommitSize, nRows, pg)) {
                            job.keyProgOff[c] = (uint32_t)plan.prog.size();
                            job.keyProgLen[c] = (uint32_t)pg.insns.size();
                            plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
                            keyOk = true;
                        }
                    }
                }
                if (!keyOk) {
                    static std::set<uint32_t> warned;
                    if (warned.insert(dec.table_id).second)
                    zklog.error("multiplicity: table " + std::to_string(dec.table_id)
                                + " is indexed-base but tuple column " + std::to_string(badCol)
                                + " of " + std::to_string(job.nKey) + " does not reduce to a linear"
                                " form (reason: " + (mulWalkFailReason() ? mulWalkFailReason()
                                                                        : "operand kind not modelled")
                                + ") -- routed to the interpreter");
                    plan.fallback.push_back({hints[i], dec, rows,
                                             (uint8_t)(selConst1 ? 1 : 0), (uint8_t)(dynBus ? 1 : 0)});
                    continue;
                }
                // A table_id flag, not a cached pointer into mulDecoders(): that vector keeps
                // growing from OTHER airs' registration calls and reallocates without warning, and
                // this plan can be built before every air has registered. `dec` itself is resolved
                // later, at use (mulPlanDevice for the GPU, mulDecoderFor(tableId) for the CPU).
                job.hasIndexedBase = 1;
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
        for (const auto& j : plan.jobs)
            for (const MulPolyDev* q : {&j.value, &j.sel, &j.bus})
              for (const MulFormDev* f = q->p; f < q->p + q->nProd; ++f)
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
        // Group jobs sharing a selector so the kernel's carry-across hits; order is otherwise free.
        std::stable_sort(plan.jobs.begin(), plan.jobs.end(), [](const MulJobDev& a, const MulJobDev& b) {
            return a.selProgOff < b.selProgOff;
        });
        if (!plan.prog.empty()) {
            std::set<uint32_t> selProgs;
            size_t withProg = 0;
            for (const auto& j : plan.jobs) {
                if (j.selProgLen) selProgs.insert(j.selProgOff);
                if (j.selProgLen || j.valProgLen || j.busProgLen) ++withProg;
            }
        }
        plan.cols1.assign(cols.begin(), cols.end());
        plan.packable = ok;
        for (const auto& j : plan.jobs) if (j.rows > plan.maxRows) plan.maxRows = j.rows;
        uint32_t shifted = 0;
        for (const auto& j : plan.jobs)
            for (const MulPolyDev* q : {&j.value, &j.sel, &j.bus})
              for (const MulFormDev* f = q->p; f < q->p + q->nProd; ++f)
                {
                    for (uint32_t k = 0; k < f->n; ++k)
                        if (f->t[k].src == MUL_SRC_TRACE && f->t[k].rowStride != 0) ++shifted;
                    for (uint32_t k = 0; k < f->n2; ++k)
                        if (f->t2[k].src == MUL_SRC_TRACE && f->t2[k].rowStride != 0) ++shifted;
                }
        if (!plan.jobs.empty())
            zklog.trace("Multiplicity plan: cm1 cols=" + std::to_string(plan.cols1.size())
                       + " packable=" + std::to_string((int)plan.packable)
                       + " shiftedTerms=" + std::to_string(shifted)
                       + " reads=" + mulSrcMaskNames(plan.srcMask)
                       + " streamable=" + std::to_string((int)mulPlanStreamable(plan)));
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
