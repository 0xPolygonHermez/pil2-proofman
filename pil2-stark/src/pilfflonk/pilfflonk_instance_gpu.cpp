// The device side of an Instance (pilfflonk_instance_gpu.hpp): compiled into the GPU library only
// (the Makefile's %_gpu.cpp rule), with g++.
#include "pilfflonk_instance_gpu.hpp"

#include <stdexcept>
#include <string>

#include "pilfflonk_expressions_gpu.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_lde_gpu.hpp"
#include "pilfflonk_proving_key.hpp"
#include "timer.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares them.
extern "C" void gpu_plonk_memcpy_d2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_cuda_device_sync();

namespace PilFflonk {

namespace {

FrElement *elements(uint8_t *arena, uint64_t offset) { return reinterpret_cast<FrElement *>(arena + offset); }

} // namespace

InstanceGpu::InstanceGpu(const GpuAirKey &_air, const uint8_t *stage1, FrElement *stageOne)
    : air(_air), lease(_air.gpuKey(), "Instance") {
    const CopyLog copies(&air.gpuKey(), "INSTANCE");
    const AirKey &key = air.airKey();
    const uint64_t N = key.n(), C = key.witnessColumns().size();
    Staging &staging = air.gpuKey().staging();
    uint8_t *arena = air.gpuKey().arena();
    FrElement *raw = elements(arena, air.arena().work);
    FrElement *columns = elements(arena, air.arena().evaluations[1]);
    staging.toDevice(raw, stage1, N * C * sizeof(FrElement));
    pilfflonk_gpu_transpose_witness(columns, raw, air.witnessPositions(), N, C);
    for (const auto &[first, count] : air.witnessColumns()) {
        staging.toHost(stageOne + first * N, columns + first * N, count * N * sizeof(FrElement));
    }
}

std::vector<G1Point> InstanceGpu::commitStage(uint64_t stage, const FrElement *columns, const FrElement *factors,
                                              std::vector<std::unique_ptr<Poly>> &polys) {
    const CopyLog copies(&air.gpuKey(), "STAGE_" + std::to_string(stage));
    const AirKey &key = air.airKey();
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    const GpuKey &gpu = air.gpuKey();
    const ArenaLayout &layout = air.arena();
    Staging &staging = gpu.staging();
    uint8_t *arena = gpu.arena();
    FrElement *evaluations = elements(arena, layout.evaluations[stage]);
    FrElement *slots = elements(arena, layout.polys);
    FrElement *deviceFactors = elements(arena, layout.factors);
    uint64_t *counts = reinterpret_cast<uint64_t *>(arena + layout.counts);

    for (const auto &[first, count] : air.hostColumns(stage)) {
        staging.toDevice(evaluations + first * N, columns + first * N, count * N * sizeof(FrElement));
    }
    uint64_t nFactors = 0;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        if (info.layout[f].stage == stage) {
            nFactors += info.layout[f].k * key.blindLength(f);
        }
    }
    staging.toDevice(deviceFactors, factors, nFactors * sizeof(FrElement));

    std::vector<G1Point> commitments;
    uint64_t drawn = 0, counted = 0;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage != stage) {
            continue;
        }
        const uint64_t b = key.blindLength(f), length = N + b;
        TimerStartExpr(PILFFLONK_INTT, f);
        for (uint64_t j = 0; j < entry.k; ++j) {
            FrElement *slot = slots + layout.slot[f] + j * length;
            const uint64_t stagePos = info.cmPolsMap[entry.pols[j].id].stagePos;
            gpu_plonk_memcpy_d2d(slot, evaluations + stagePos * N, N * sizeof(FrElement));
            pilfflonk_gpu_memset_zero(slot + N, b * sizeof(FrElement));
            transformOnDevice(slot, info.nBits, true);
        }
        pilfflonk_gpu_blind(slots, air.offsets(f), entry.k, N, deviceFactors + drawn, b);
        pilfflonk_gpu_count_coefficients(counts + counted, slots, air.offsets(f), entry.k, length);
        TimerStopAndLogExpr(PILFFLONK_INTT, f);
        TimerStartExpr(PILFFLONK_COMMIT, f);
        commitments.push_back(gpu.commit(slots, air.offsets(f), entry.k, length, entry.degree,
                                         elements(arena, layout.work)));
        TimerStopAndLogExpr(PILFFLONK_COMMIT, f);
        drawn += entry.k * b;
        counted += entry.k;
    }

    // The host's copies, over the mirror, which has the arena's slots.
    std::vector<uint64_t> found(counted);
    staging.toHost(found.data(), counts, counted * sizeof(uint64_t));
    FrElement *mirror = gpu.mirror();
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage == stage) {
            const uint64_t elementsOfF = entry.k * (N + key.blindLength(f));
            staging.toRegisteredHost(mirror + layout.slot[f], slots + layout.slot[f], elementsOfF * sizeof(FrElement));
        }
    }
    staging.wait();
    counted = 0;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage != stage) {
            continue;
        }
        const uint64_t length = N + key.blindLength(f);
        for (uint64_t j = 0; j < entry.k; ++j) {
            const uint64_t id = entry.pols[j].id;
            polys[id] = mirrorPolynomial(mirror + layout.slot[f] + j * length, length, found[counted++],
                                         key.name() + ": the column " + info.cmPolsMap[id].name);
        }
    }
    return commitments;
}

void InstanceGpu::requireQParts(uint64_t partBits, const char *function) const {
    const AirKey &key = air.airKey();
    const uint64_t needed = qPhaseBytes(key, air.arena(), partBits), held = air.gpuKey().arenaSize();
    if (needed > held) {
        throw std::invalid_argument(std::string(function) + ": not enough GPU memory for Q in parts of 2^" +
                                    std::to_string(partBits) + " points: it needs " + std::to_string(needed) +
                                    " bytes of the key's arena of device memory, and the arena has " +
                                    std::to_string(held) + " (the default parts, of 2^" +
                                    std::to_string(key.info().nBits) + " points, need " +
                                    std::to_string(qPhaseBytes(key, air.arena(), key.info().nBits)) +
                                    "; pilfflonk/docs/performance.md#selection-memory-and-errors)");
    }
}

uint64_t InstanceGpu::computeQ(uint64_t partBits, const QValues &valuesOn) {
    requireQParts(partBits, "InstanceGpu::computeQ");
    const AirKey &key = air.airKey();
    const PilfflonkInfo &info = key.info();
    const ArenaLayout &layout = air.arena();
    uint8_t *arena = air.gpuKey().arena();
    const uint64_t S = uint64_t(1) << partBits, M = key.lde().extendedSize(), nParts = M / S;
    const QPartLayout part = qPartLayout(key, partBits);
    void *zerofiers = arena + layout.qPart + part.zerofiers;
    FrElement *columns = elements(arena, layout.qPart + part.columns);
    FrElement *q = elements(arena, layout.q);
    const LdeGpu lde(air, partBits, columns);
    const ProverValues values = valuesOn(columns, S);

    // Each phase is waited for, so that its timer has its time.
    for (uint64_t p = 0; p < nParts; ++p) {
        TimerStartExpr(PILFFLONK_Q_EXTEND, p);
        lde.extendPart(p);
        gpu_plonk_cuda_device_sync();
        TimerStopAndLogExpr(PILFFLONK_Q_EXTEND, p);
        TimerStartExpr(PILFFLONK_Q_DOMAIN, p);
        const ExpressionsDomainGpu domain = ExpressionsDomainGpu::cosetPart(
            info.nBits, key.degrees().nBitsExt, partBits, p, info.boundaries, zerofiers);
        gpu_plonk_cuda_device_sync();
        TimerStopAndLogExpr(PILFFLONK_Q_DOMAIN, p);
        TimerStartExpr(PILFFLONK_Q_EVALUATE, p);
        air.expressions().calculateExpression(info.cExpId, domain, values, q + p, nParts);
        gpu_plonk_cuda_device_sync();
        TimerStopAndLogExpr(PILFFLONK_Q_EVALUATE, p);
    }
    TimerStart(PILFFLONK_Q_INTERPOLATE);
    lde.interpolate();
    gpu_plonk_cuda_device_sync();
    TimerStopAndLog(PILFFLONK_Q_INTERPOLATE);

    uint64_t *count = reinterpret_cast<uint64_t *>(arena + layout.qCounts);
    pilfflonk_gpu_count_coefficients(count, q, air.qPieceOffsets(), 1, M);
    uint64_t coefficients = 0;
    air.gpuKey().staging().toHost(&coefficients, count, sizeof(coefficients));
    return coefficients;
}

std::vector<G1Point> InstanceGpu::commitQ(const FrElement *factors) {
    const AirKey &key = air.airKey();
    const PilfflonkInfo &info = key.info();
    const AirDegrees &d = key.degrees();
    const GpuKey &gpu = air.gpuKey();
    const ArenaLayout &layout = air.arena();
    Staging &staging = gpu.staging();
    uint8_t *arena = gpu.arena();
    const uint64_t m = key.nQPieces(), slot = layout.qPieceElements;
    const FrElement *q = elements(arena, layout.q);
    FrElement *pieces = elements(arena, layout.qPieces);
    uint64_t *counts = reinterpret_cast<uint64_t *>(arena + layout.qCounts) + 1;

    // Unsplit, the one piece is Q itself, at its coefficients, unblinded.
    if (m > 1) {
        pilfflonk_gpu_memset_zero(pieces, m * slot * sizeof(FrElement));
        for (uint64_t i = 0; i < m; ++i) {
            const QPieceRange range = qPieceRange(d, i);
            gpu_plonk_memcpy_d2d(pieces + i * slot, q + range.start, range.length * sizeof(FrElement));
        }
        FrElement *deviceFactors = elements(arena, layout.qFactors);
        staging.toDevice(deviceFactors, factors, 2 * (m - 1) * sizeof(FrElement));
        pilfflonk_gpu_blind_q_boundaries(pieces, slot, d.qStride, m, deviceFactors);
    }
    pilfflonk_gpu_count_coefficients(counts, pieces, air.qPieceOffsets(), m, slot);

    std::vector<G1Point> commitments;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage == info.qStage()) {
            commitments.push_back(
                gpu.commit(pieces, air.offsets(f), entry.k, slot, entry.degree, elements(arena, layout.shplonk)));
        }
    }
    qPieceCounts.assign(m, 0);
    staging.toHost(qPieceCounts.data(), counts, m * sizeof(uint64_t));
    return commitments;
}

uint64_t InstanceGpu::qPieceDegree(uint64_t i) const {
    const uint64_t count = qPieceCounts.at(i);
    return count == 0 ? 0 : count - 1;
}

std::vector<std::unique_ptr<Poly>> InstanceGpu::qPiecesToHost(FrElement *coefs) const {
    const AirKey &key = air.airKey();
    const std::vector<uint64_t> &bounds = key.degrees().qPieceCoefficients;
    const ArenaLayout &layout = air.arena();
    const FrElement *pieces = elements(air.gpuKey().arena(), layout.qPieces);
    Staging &staging = air.gpuKey().staging();
    std::vector<std::unique_ptr<Poly>> polys;
    uint64_t at = 0;
    for (uint64_t i = 0; i < bounds.size(); ++i) {
        staging.toHost(coefs + at, pieces + i * layout.qPieceElements, bounds[i] * sizeof(FrElement));
        polys.push_back(
            mirrorPolynomial(coefs + at, bounds[i], qPieceCounts.at(i), key.name() + ": piece Q" + std::to_string(i)));
        at += bounds[i];
    }
    return polys;
}

} // namespace PilFflonk
