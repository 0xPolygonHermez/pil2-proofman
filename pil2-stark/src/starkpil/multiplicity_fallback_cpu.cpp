// The interpreter fallback for the host scatter.
//
// A lookup the extractor cannot reduce to a closed form is counted here instead, so a CPU run
// counts exactly what a GPU run does rather than coming up short by whatever failed to reduce.
// Its own translation unit because the expression evaluator's headers cannot be included from
// multiplicity_cpu.hpp without closing an include cycle through const_pols.hpp.
#include "multiplicity_cpu.hpp"
#include "expressions_pack.hpp"
#include "hints.hpp"

// Insurance for the single-value affine path below: a decoder with any tuple-resolved shape (map,
// digit rule, or indexed-base) must never reach it -- that path folds straight to one column and
// would silently count a wrong row. This is the one remaining host call site for the raw decode,
// so route it through here rather than `mul_decode` directly, matching the mulClaimShape
// convention of refusing loudly instead of miscounting quietly.
static uint64_t mul_decode_checked(const MulDecoder& d, uint64_t value) {
    if (d.mapSlots != 0 || d.digitCols != 0 || d.nSel != 0) {
        zklog.error("multiplicity: table " + std::to_string(d.table_id) + " has a map/digit/"
                    "indexed-base row map but reached the single-value affine decode on the "
                    "interpreter fallback -- a caller forgot to route it through mulResolveRow");
        exitProcess();
    }
    return mul_decode(d, value);
}

void mul_scatter_fallback_cpu(SetupCtx& setupCtx, StepsParams& params,
                                     const MulPlan& plan, uint64_t hostAirId) {
    if (plan.fallback.empty()) return;
    std::vector<std::atomic<uint64_t>>* acc = nullptr;
    {
        std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
        auto it = mulCpuAccs().find(hostAirId);
        if (it == mulCpuAccs().end()) return;
        acc = &it->second;
    }

    ProverHelpers helpers;
    ExpressionsPack ctx(setupCtx, &helpers);
    HintFieldOptions opts;

    for (const MulFallbackJob& fb : plan.fallback) {
        const uint64_t rows = fb.rows;

        // A tuple-resolved table (map, digit rule, or indexed-base) needs every column
        // mulResolveRow reads, not just the single folded value the affine/bias path uses --
        // mirrors the GPU interpreter fallback's tupleForm branch (expressions_gpu.cu) so both
        // backends resolve the same key the same way instead of disagreeing by construction.
        const bool tupleForm = fb.dec.mapSlots != 0 || fb.dec.digitCols != 0 || fb.dec.nSel != 0;
        const uint32_t nVal = fb.dec.digitCols != 0 ? fb.dec.digitCols
                                                     : (tupleForm ? fb.dec.nKey : 1u);
        if (nVal == 0 || nVal > MUL_MAX_TUPLE) {
            zklog.error("multiplicity: table " + std::to_string(fb.dec.table_id)
                        + " interpreter fallback needs " + std::to_string(nVal)
                        + " tuple columns on the CPU, outside [1," + std::to_string(MUL_MAX_TUPLE)
                        + "]");
            exitProcess();
        }

        // One evaluation per tuple column, each into its own buffer -- `valueIdx` (the hint field's
        // own index) picks which element of the lookup's tuple that column is; a range check's
        // single-element tuple is just the nVal==1 case of the same loop. `skipRedundantOne` off: a
        // tuple element that is the literal 1 must still be produced, not elided.
        std::vector<std::vector<Goldilocks::Element>> value(nVal);
        std::vector<uint64_t> vStride(nVal, 1);
        for (uint32_t c = 0; c < nVal; ++c) {
            value[c].resize(rows * FIELD_EXTENSION);
            Dest dValue(value[c].data(), rows, 0);
            addHintFieldAt(setupCtx, params, fb.hintId, dValue, "expressions", c, opts, false);
            ctx.calculateExpressions(params, dValue, rows, false, false);
            // A dim-1 expression is written one element per row, not FIELD_EXTENSION apart.
            // Reading it with the wide stride samples every third row and silently loses two
            // thirds of the count.
            vStride[c] = dValue.dim;
        }

        std::vector<Goldilocks::Element> sel, bus;
        uint64_t sStride = 1, bStride = 1;

        if (!fb.selConstOne) {
            sel.resize(rows * FIELD_EXTENSION);
            Dest d(sel.data(), rows, 0);
            addHintField(setupCtx, params, fb.hintId, d, "num_reps", opts);
            ctx.calculateExpressions(params, d, rows, false, false);
            sStride = d.dim;
        }
        if (fb.hasBus) {
            bus.resize(rows * FIELD_EXTENSION);
            Dest d(bus.data(), rows, 0);
            addHintField(setupCtx, params, fb.hintId, d, "busid", opts);
            ctx.calculateExpressions(params, d, rows, false, false);
            bStride = d.dim;
        }

        std::atomic<uint64_t> oob{0};
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < (int64_t)rows; ++row) {
            const uint64_t m = fb.selConstOne
                             ? 1ULL
                             : mulCanonHD(Goldilocks::toU64(sel[row * sStride]));
            if (m == 0) continue;
            if (fb.hasBus
                && mulCanonHD(Goldilocks::toU64(bus[row * bStride])) != (uint64_t)fb.dec.table_id)
                continue;

            uint64_t idx;
            if (tupleForm) {
                // Same resolve the GPU interpreter fallback uses (expressions_gpu.cu's tupleForm
                // branch), so the two backends cannot disagree about a row.
                uint64_t key[MUL_MAX_TUPLE];
                for (uint32_t c = 0; c < nVal; ++c)
                    key[c] = mulCanonHD(Goldilocks::toU64(value[c][row * vStride[c]]));
                if (!mulResolveRow(key, nVal, fb.dec.mapSlots, fb.dec.mapKV, idx,
                                   fb.dec.digitCols, fb.dec.digitTab, &fb.dec)) {
                    oob.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
            } else {
                idx = mul_decode_checked(fb.dec, Goldilocks::toU64(value[0][row * vStride[0]]));
            }
            if (idx >= fb.dec.n_rows) { oob.fetch_add(1, std::memory_order_relaxed); continue; }
            (*acc)[fb.dec.acc_base + idx].fetch_add(m, std::memory_order_relaxed);
        }
        if (oob.load())
            zklog.error("multiplicity: " + std::to_string(oob.load()) + " decodes outside table "
                        + std::to_string(fb.dec.table_id) + " on the interpreter fallback");
    }
}

