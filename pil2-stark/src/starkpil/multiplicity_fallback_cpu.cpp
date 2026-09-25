// The interpreter fallback for the host scatter: a lookup that does not compile is counted here,
// so CPU and GPU runs count the same. Separate TU to avoid an include cycle through const_pols.hpp.
#include "multiplicity_cpu.hpp"
#include "expressions_pack.hpp"
#include "hints.hpp"

// Guard for the single-value affine path: a tuple-resolved decoder (map, digit rule, indexed-base)
// must never reach it, or it would count a wrong row. Refuse loudly, as mulClaimShape does.
static uint64_t mul_decode_checked(const MulDecoder& d, uint64_t value) {
    if (d.mapSlots != 0 || d.digitCols != 0) {
        zklog.error("multiplicity: table " + std::to_string(d.table_id) + " has a map/digit/"
                    "row map but reached the single-value affine decode on the "
                    "interpreter fallback -- a caller forgot to route it through mulResolveRow");
        exitProcess();
    }
    return mul_decode(d, value);
}

void mul_scatter_fallback_cpu(SetupCtx& setupCtx, StepsParams& params, const MulPlan& plan) {
    if (plan.fallback.empty()) return;
    ProverHelpers helpers;
    ExpressionsPack ctx(setupCtx, &helpers);
    HintFieldOptions opts;

    for (const MulFallbackJob& fb : plan.fallback) {
        const uint64_t rows = fb.rows;
        // Per job, like the kernel jobs: the counters live in the air hosting this job's table.
        std::vector<std::atomic<uint64_t>>* acc = nullptr;
        {
            uint64_t hostAirId = UINT64_MAX;
            for (const auto& L : mulVtLayouts())
                if (L.accBase.count(fb.dec.table_id)) { hostAirId = L.airId; break; }
            std::lock_guard<std::mutex> lk(mulCpuAccsMutex());
            auto it = mulCpuAccs().find(hostAirId);
            if (it == mulCpuAccs().end()) continue;
            acc = &it->second;
        }

        // A tuple-resolved table needs every column mulResolveRow reads, not just the folded value;
        // mirrors the GPU fallback's tupleForm branch (expressions_gpu.cu).
        const bool tupleForm = fb.dec.mapSlots != 0 || fb.dec.digitCols != 0;
        const uint32_t nVal = fb.dec.digitCols != 0 ? fb.dec.digitCols
                                                     : (tupleForm ? fb.dec.nKey : 1u);
        if (nVal == 0 || nVal > MUL_MAX_TUPLE) {
            zklog.error("multiplicity: table " + std::to_string(fb.dec.table_id)
                        + " interpreter fallback needs " + std::to_string(nVal)
                        + " tuple columns on the CPU, outside [1," + std::to_string(MUL_MAX_TUPLE)
                        + "]");
            exitProcess();
        }

        // One evaluation per tuple column; `valueIdx` selects the tuple element. `skipRedundantOne` off:
        // a literal-1 element must still be produced.
        std::vector<std::vector<Goldilocks::Element>> value(nVal);
        std::vector<uint64_t> vStride(nVal, 1);
        for (uint32_t c = 0; c < nVal; ++c) {
            value[c].resize(rows * FIELD_EXTENSION);
            Dest dValue(value[c].data(), rows, 0);
            addHintFieldAt(setupCtx, params, fb.hintId, dValue, "expressions", c, opts, false);
            ctx.calculateExpressions(params, dValue, rows, false, false);
            // A dim-1 expression is one element per row, not FIELD_EXTENSION apart; the wide stride would
            // drop two thirds of the count.
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
                                   fb.dec.digitCols, fb.dec.digitTab)) {
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

