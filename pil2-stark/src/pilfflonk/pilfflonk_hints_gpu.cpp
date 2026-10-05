// The hint columns and im pols of the stages on a key on the GPU (pilfflonk_hints_gpu.hpp): compiled
// into the GPU library only (the Makefile's %_gpu.cpp rule), with g++. It calls the kernels of
// pilfflonk_hints.cu and the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu) through their C
// linkage.
#include "pilfflonk_hints_gpu.hpp"

#include <algorithm>
#include <cstring>

#include "pilfflonk_expressions_gpu.hpp"
#include "pilfflonk_hints_kernels.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "timer.hpp"

namespace PilFflonk {

namespace {

constexpr uint64_t NO_ROW = UINT64_MAX;

uint64_t aligned(uint64_t bytes) { return (bytes + 255) & ~uint64_t(255); }

FrElement *elements(uint8_t *base, uint64_t offset) { return reinterpret_cast<FrElement *>(base + offset); }

// The columns of stage s on H in the arena, column p at evaluations(s) + p·N.
FrElement *evaluations(const GpuAirKey &air, uint64_t stage) {
    return elements(air.gpuKey().arena(), air.arena().evaluations[stage]);
}

} // namespace

StageScratch stageScratch(const AirKey &air, uint64_t stage) {
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    std::vector<uint64_t> fixed;
    auto readFixed = [&](const std::vector<ColumnRead> &reads) {
        for (const ColumnRead &c : reads) {
            if (c.type == 0) {
                fixed.push_back(c.index);
            }
        }
    };
    bool hinted = false;
    for (const StdHint &hint : air.stdHints()) {
        if (hint.stage != stage) {
            continue;
        }
        hinted = true;
        for (const HintInput *in : {&hint.numerator, &hint.denominator}) {
            if (in->kind == HintInput::Kind::Column) {
                readFixed({in->column});
            } else if (in->kind == HintInput::Kind::Expression) {
                readFixed(columnsRead(air.bin(), in->expId));
            }
        }
    }
    for (const PolMapEntry &p : info.cmPolsMap) {
        if (p.imPol && p.stage == stage) {
            readFixed(columnsRead(air.bin(), p.expId));
        }
    }
    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

    StageScratch s;
    uint64_t end = 0;
    auto take = [&](uint64_t bytes) {
        const uint64_t start = end;
        end = aligned(end + bytes);
        return start;
    };
    s.fixedValues = take(fixed.size() * N * sizeof(FrElement));
    s.denominator = take(hinted ? N * sizeof(FrElement) : 0);
    s.work = take(hinted ? pilfflonk_gpu_prefix_scan_work_elements(N) * sizeof(FrElement) : 0);
    s.zeroRow = take(hinted ? sizeof(uint64_t) : 0);
    s.bytes = end;
    s.fixed = std::move(fixed);
    return s;
}

uint64_t stageScratchBytes(const AirKey &air) {
    uint64_t bytes = 0;
    for (uint64_t s = 1; s <= air.info().nStages; ++s) {
        bytes = std::max(bytes, stageScratch(air, s).bytes);
    }
    return bytes;
}

void computeStageColumns(const GpuAirKey &air, uint64_t stage, const ProverValues &scalars) {
    const ProofCall call(air.gpuKey());
    const AirKey &key = air.airKey();
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    const StageScratch scratch = stageScratch(key, stage);
    uint8_t *base = air.gpuKey().arena() + air.arena().work;
    Staging &staging = air.gpuKey().staging();

    TimerStartExpr(PILFFLONK_HINT_COLUMNS, stage);
    // The values the code reads: the fixed columns the stage reads, on H from their coefficients,
    // the columns of the stages before it, and those of this one, which the hints and the im pols
    // compute in an order where each reads only what is computed before it (AirKey).
    ProverValues values = scalars;
    values.columns.assign(stage + 1, {});
    values.columns[0].assign(info.nConstants, nullptr);
    FrElement *fixed = elements(base, scratch.fixedValues);
    for (uint64_t k = 0; k < scratch.fixed.size(); ++k) {
        FrElement *column = fixed + k * N;
        gpu_plonk_memcpy_d2d(column, air.fixedCoefficients() + scratch.fixed[k] * N, N * sizeof(FrElement));
        transformOnDevice(column, info.nBits, false);
        values.columns[0][scratch.fixed[k]] = column;
    }
    for (uint64_t s = 1; s <= stage; ++s) {
        for (uint64_t p = 0; p < key.cmIds()[s].size(); ++p) {
            values.columns[s].push_back(evaluations(air, s) + p * N);
        }
    }
    const ExpressionsDomainGpu trace = ExpressionsDomainGpu::trace(info.nBits);
    const ExpressionsGpu &interpreter = air.expressions();
    // An operand on every row of H, as addHintField reads it: a column at row i + offset,
    // cyclically, a number, or an expression, evaluated into `buffer`.
    auto operand = [&](const HintInput &in, FrElement *buffer) {
        HintOperand op{};
        switch (in.kind) {
        case HintInput::Kind::Expression:
            interpreter.calculateExpression(in.expId, trace, values, buffer);
            op.values = buffer;
            break;
        case HintInput::Kind::Column:
            op.values = values.columns[in.column.type][in.column.index];
            op.shift = hintRowShift(in, N);
            break;
        case HintInput::Kind::Number:
            static_assert(sizeof(op.number) == sizeof(FrElement), "a number is one element");
            std::memcpy(op.number, &in.number, sizeof(op.number));
            break;
        }
        return op;
    };
    uint64_t *zeroRow = reinterpret_cast<uint64_t *>(base + scratch.zeroRow);
    FrElement *work = elements(base, scratch.work);
    for (const StdHint &hint : key.stdHints()) {
        if (hint.stage != stage) {
            continue;
        }
        FrElement *dest = evaluations(air, stage) + hint.stagePos * N;
        const HintOperand numerator = operand(hint.numerator, dest);
        const HintOperand denominator = operand(hint.denominator, elements(base, scratch.denominator));
        pilfflonk_gpu_hint_quotient(dest, N, &numerator, &denominator, zeroRow);
        uint64_t row = NO_ROW;
        staging.toHost(&row, zeroRow, sizeof(row));
        if (row != NO_ROW) {
            throw zeroDenominatorError(key, hint, row);
        }
        if (hint.kind == StdHint::Kind::Prod) {
            gpu_plonk_prefix_scan_multiply(dest, N, work);
        } else if (hint.kind == StdHint::Kind::Sum) {
            pilfflonk_gpu_prefix_scan_add(dest, N, work);
        }
    }
    gpu_plonk_cuda_device_sync();
    TimerStopAndLogExpr(PILFFLONK_HINT_COLUMNS, stage);

    TimerStartExpr(PILFFLONK_IM_POLS, stage);
    for (uint64_t id : imPolOrder(key, stage)) {
        const PolMapEntry &p = info.cmPolsMap[id];
        interpreter.calculateExpression(p.expId, trace, values, evaluations(air, stage) + p.stagePos * N);
    }
    gpu_plonk_cuda_device_sync();
    TimerStopAndLogExpr(PILFFLONK_IM_POLS, stage);
}

void stageColumnToHost(const GpuAirKey &air, uint64_t stage, uint64_t stagePos, FrElement *out) {
    const ProofCall call(air.gpuKey());
    const uint64_t N = air.airKey().n();
    air.gpuKey().staging().toHost(out, evaluations(air, stage) + stagePos * N, N * sizeof(FrElement));
}

} // namespace PilFflonk
