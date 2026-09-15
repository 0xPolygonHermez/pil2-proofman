// The interpreter fallback for the host scatter.
//
// A lookup the extractor cannot reduce to a closed form is counted here instead, so a CPU run
// counts exactly what a GPU run does rather than coming up short by whatever failed to reduce.
// Its own translation unit because the expression evaluator's headers cannot be included from
// multiplicity_cpu.hpp without closing an include cycle through const_pols.hpp.
#include "multiplicity_cpu.hpp"
#include "expressions_pack.hpp"
#include "hints.hpp"

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
        std::vector<Goldilocks::Element> value(rows * FIELD_EXTENSION);
        std::vector<Goldilocks::Element> sel, bus;
        uint64_t sStride = 1, bStride = 1;

        // One evaluation per field, each into its own buffer. `skipRedundantOne` off: a tuple that
        // is the literal 1 must still be produced, not elided.
        Dest dValue(value.data(), rows, 0);
        addHintFieldAt(setupCtx, params, fb.hintId, dValue, "expressions", 0, opts, false);
        ctx.calculateExpressions(params, dValue, rows, false, false);
        // A dim-1 expression is written one element per row, not FIELD_EXTENSION apart. Reading it
        // with the wide stride samples every third row and silently loses two thirds of the count.
        const uint64_t vStride = dValue.dim;

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
            const uint64_t idx = mul_decode(fb.dec, Goldilocks::toU64(value[row * vStride]));
            if (idx >= fb.dec.n_rows) { oob.fetch_add(1, std::memory_order_relaxed); continue; }
            (*acc)[fb.dec.acc_base + idx].fetch_add(m, std::memory_order_relaxed);
        }
        if (oob.load())
            zklog.error("multiplicity: " + std::to_string(oob.load()) + " decodes outside table "
                        + std::to_string(fb.dec.table_id) + " on the interpreter fallback");
    }
}

