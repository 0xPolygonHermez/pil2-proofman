#ifndef WITNESS_HINTS_SLOT_HPP
#define WITNESS_HINTS_SLOT_HPP

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "multiplicity_job.hpp"
#include "multiplicity_extract.hpp"
#include "setup_ctx.hpp"

// Evaluating an air's witness_calc hints inside a streaming-commit slot.
//
// A hint is a stage-1 column the PROVER computes. The legacy commit unpacks cm1 and runs
// calculateWitnessExpr_gpu before the LDE; a slot never materialises cm1, so hints are evaluated
// from what a slot holds:
//
//   * cm1      -- the WHOLE packed witness is resident, so a `'`-shifted read is another row.
//   * const    -- the unpacked const scratch the multiplicity hook allocates.
//   * airvalue -- the pinned value window the multiplicity hook stages.
//
// Hints reading other hints' columns are evaluated in declaration order into a side buffer
// (nDest * N); only backward references are accepted.
//
// This header is the host half: compile each hint once per air and rewrite its operands to those
// sources. An air whose hints it cannot express keeps `ok == false` and must not take the slot.

struct SlotHintOp {
    uint32_t progOff = 0, progLen = 0;
    uint32_t destCol = 0;    // the stage-1 column it writes
    uint32_t destSlot = 0;   // where that column lives in the side buffer
};

struct SlotHintPlan {
    std::vector<MulInsnDev> prog;      // every hint's bytecode, concatenated
    std::vector<SlotHintOp> ops;       // in DECLARATION order; the schedule is that order
    std::vector<uint32_t>   destCols;  // sorted; the chunk loop patches these after each unpack
    uint64_t                nRows = 0;
    bool                    ok = false;
    std::string             why;       // why not, when !ok
};


// Compile this air's witness_calc hints into a schedule a slot can run. `widths` is the packed
// column layout (PackedInfo::unpack_info) and `wordsPerRow` its row stride; unpacked airs cannot
// use this. `ok` in the result says whether every hint was expressible.
inline SlotHintPlan slotHintBuildPlan(SetupCtx& setupCtx, const std::vector<uint64_t>& widths,
                                      uint64_t wordsPerRow) {
    SlotHintPlan plan;
    plan.nRows = 1ULL << setupCtx.starkInfo.starkStruct.nBits;
    const uint64_t nh = setupCtx.expressionsBin.getNumberHintIdsByName("witness_calc");
    if (nh == 0) { plan.ok = true; return plan; }          // nothing to do is a valid plan
    if (widths.empty() || wordsPerRow == 0) { plan.why = "air is not packed"; return plan; }

    std::vector<uint64_t> bitOf(widths.size(), 0);
    uint64_t bit = 0;
    for (size_t c = 0; c < widths.size(); ++c) { bitOf[c] = bit; bit += widths[c]; }
    if (bit == 0 || bit > wordsPerRow * 64) { plan.why = "packed layout does not fit the row"; return plan; }

    std::vector<uint64_t> ids(nh);
    setupCtx.expressionsBin.getHintIdsByName(ids.data(), "witness_calc");
    const uint64_t bcs = 1 + setupCtx.starkInfo.nStages + 3 + setupCtx.starkInfo.customCommits.size();

    // Pass 1: the stage-1 columns these hints write, and where each lands in the side buffer.
    std::map<uint32_t, uint32_t> slotOf;
    for (uint64_t i = 0; i < nh; ++i)
        for (auto& f : setupCtx.expressionsBin.hints[ids[i]].fields) {
            if (f.name != "reference" || f.values.empty()) continue;
            if (f.values[0].operand != opType::cm) { plan.why = "hint writes something other than a column"; return plan; }
            const auto& pm = setupCtx.starkInfo.cmPolsMap[f.values[0].id];
            if (pm.stage != 1) { plan.why = "hint writes a later stage"; return plan; }
            for (uint64_t k = 0; k < pm.dim; ++k) {
                const uint32_t col = (uint32_t)(pm.stagePos + k);
                if (!slotOf.count(col)) { const uint32_t n = (uint32_t)slotOf.size(); slotOf[col] = n; }
            }
        }

    // Pass 2: compile each hint in declaration order, rewriting operands to what a slot has. A hint
    // column is accepted only once an EARLIER hint wrote it; forward references are refused.
    std::set<uint32_t> written;
    for (uint64_t i = 0; i < nh; ++i) {
        Hint& h = setupCtx.expressionsBin.hints[ids[i]];
        const HintField *fe = nullptr, *fr = nullptr;
        for (auto& f : h.fields) { if (f.name == "expression") fe = &f; if (f.name == "reference") fr = &f; }
        if (fe == nullptr || fe->values.empty() || fr == nullptr || fr->values.empty()) {
            plan.why = "hint has no expression or no destination"; return plan;
        }
        MulProgram pg;
        if (!mulCompileField(setupCtx, fe->values[0], bcs, plan.nRows, pg)) {
            plan.why = std::string("expression did not compile: ") + (mulProgFailReason() ? mulProgFailReason() : "?");
            return plan;
        }
        for (auto& in : pg.insns)
            for (MulOperandDev* o : {&in.a, &in.b}) {
                if (o->kind != MUL_OPND_COL) continue;
                MulTermDev& t = o->term;
                if (MUL_SRC_IS_UNIFORM(t.src) || t.src == MUL_SRC_CONST) continue;   // served as-is
                if (t.src != MUL_SRC_TRACE) { plan.why = "operand reads a stage a slot does not have"; return plan; }
                auto sit = slotOf.find(t.col);
                if (sit != slotOf.end()) {
                    if (!written.count(t.col)) { plan.why = "hint reads a column a LATER hint writes"; return plan; }
                    t.src = MUL_SRC_HINTCOL;
                    t.sectionOffset = sit->second;
                    continue;
                }
                if (t.col >= widths.size() || widths[t.col] == 0 || widths[t.col] >= 64) {
                    plan.why = "cm1 operand has no usable packed width"; return plan;
                }
                t.src = MUL_SRC_PACKED;
                t.sectionOffset = bitOf[t.col];
                t.nCols = (uint32_t)widths[t.col];
            }

        const auto& pm = setupCtx.starkInfo.cmPolsMap[fr->values[0].id];
        if (pm.dim != 1) { plan.why = "hint destination is not a single column"; return plan; }
        SlotHintOp op;
        op.progOff = (uint32_t)plan.prog.size();
        op.progLen = (uint32_t)pg.insns.size();
        op.destCol = (uint32_t)pm.stagePos;
        op.destSlot = slotOf[op.destCol];
        plan.prog.insert(plan.prog.end(), pg.insns.begin(), pg.insns.end());
        plan.ops.push_back(op);
        written.insert(op.destCol);
    }

    for (auto& kv : slotOf) plan.destCols.push_back(kv.first);
    std::sort(plan.destCols.begin(), plan.destCols.end());
    plan.ok = true;
    return plan;
}

// One plan per air; setup-derived, so it outlives every instance.
inline SlotHintPlan& slotHintPlanFor(SetupCtx& setupCtx, uint64_t airgroupId, uint64_t airId,
                                     const std::vector<uint64_t>& widths, uint64_t wordsPerRow) {
    static std::map<std::pair<uint64_t,uint64_t>, SlotHintPlan> plans;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(airgroupId, airId);
    auto it = plans.find(key);
    if (it == plans.end())
        it = plans.emplace(key, slotHintBuildPlan(setupCtx, widths, wordsPerRow)).first;
    return it->second;
}

#endif
