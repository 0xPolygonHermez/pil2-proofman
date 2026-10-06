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

namespace PilFflonk {

namespace {

FrElement *elements(uint8_t *arena, uint64_t offset) { return reinterpret_cast<FrElement *>(arena + offset); }

// Whether the timers' lines are printed: they log at trace level (timer.hpp), as gpu_timer.cuh's
// events are recorded only then.
bool timersLogged() {
    return CPlusPlusLogging::Logger::getInstance(CPlusPlusLogging::LOG_TYPE::CONSOLE)->getLogLevel() >=
           CPlusPlusLogging::LOG_LEVEL_TRACE;
}

} // namespace

InstanceGpu::InstanceGpu(const GpuAirKey &_air, const uint8_t *stage1)
    : air(_air), lease(_air.gpuKey(), "Instance") {
    const ProofCall call(air.gpuKey());
    const CopyLog copies(&air.gpuKey(), "INSTANCE");
    const AirKey &key = air.airKey();
    const uint64_t N = key.n(), C = key.witnessColumns().size();
    Staging &staging = air.gpuKey().staging();
    uint8_t *arena = air.gpuKey().arena();
    FrElement *raw = elements(arena, air.arena().work);
    FrElement *columns = elements(arena, air.arena().evaluations[1]);
    staging.toDevice(raw, stage1, N * C * sizeof(FrElement));
    pilfflonk_gpu_transpose_witness(columns, raw, air.witnessPositions(), N, C);
    polyCounts.assign(key.info().cmPolsMap.size(), 0);
    committed.assign(key.info().cmPolsMap.size(), false);
}

InstanceGpu::InstanceGpu(const GpuAirKey &_air, const pilfflonk_exec_witness &exec)
    : air(_air), lease(_air.gpuKey(), "Instance") {
    const ProofCall call(air.gpuKey());
    const CopyLog copies(&air.gpuKey(), "INSTANCE");
    const AirKey &key = air.airKey();
    const uint64_t N = key.n(), C = key.witnessColumns().size();
    constexpr uint64_t WRAP_COLS = 89, TABLE_ROWS = 1ull << 17, RANGE_ROWS = 1ull << 16;
    const GpuAirKey::Exec *parts = air.exec();
    if (parts == nullptr) {
        throw std::invalid_argument("InstanceGpu: the key has no exec parts (pilfflonk_ctx_set_exec)");
    }
    if (C != WRAP_COLS || N < TABLE_ROWS || exec.n_blocks * 64 > N || exec.n_wires != parts->nWires) {
        throw std::invalid_argument("InstanceGpu: the exec witness is not one of the blake3 wrap's AIR (" +
                                    std::to_string(C) + " columns, 2^" + std::to_string(key.info().nBits) +
                                    " rows, " + std::to_string(parts->nWires) + " wires)");
    }
    Staging &staging = air.gpuKey().staging();
    uint8_t *arena = air.gpuKey().arena();
    FrElement *columns = elements(arena, air.arena().evaluations[1]);

    // What the proof adds, end to end at `work`, each at a multiple of 256 bytes.
    uint64_t at = air.arena().work;
    const uint64_t end = at + air.arena().workElements * sizeof(FrElement);
    auto take = [&](uint64_t bytes) {
        uint8_t *p = arena + at;
        at += (bytes + 255) / 256 * 256;
        return p;
    };
    const uint64_t fr = sizeof(FrElement);
    // The wires, then the additions and a zero, the empty cells' wire.
    uint8_t *wires = take(exec.n_wires * fr);
    uint8_t *internal = take((parts->nAdds + 1) * fr);
    uint8_t *blocks = take(exec.n_blocks * sizeof(pilfflonk_wrap_block));
    uint8_t *counts = take((TABLE_ROWS + RANGE_ROWS) * sizeof(uint32_t));
    uint8_t *hostRange = take(RANGE_ROWS * sizeof(uint32_t));
    if (at > end) {
        throw std::invalid_argument("InstanceGpu: the exec witness needs " + std::to_string(at - air.arena().work) +
                                    " bytes of scratch, and the arena's work has " +
                                    std::to_string(end - air.arena().work));
    }
    staging.toDevice(wires, exec.wires, exec.n_wires * fr);
    staging.toDevice(blocks, exec.blocks, exec.n_blocks * sizeof(pilfflonk_wrap_block));
    staging.toDevice(hostRange, exec.range_counts, RANGE_ROWS * sizeof(uint32_t));

    pilfflonk_gpu_memset_zero(internal + parts->nAdds * fr, fr);
    if (parts->nAdds > 0) {
        gpu_plonk_calculate_additions(internal, wires, parts->id1, parts->id2, parts->f1, parts->f2, parts->levels,
                                      static_cast<uint8_t>(parts->nLevels - 1), static_cast<uint32_t>(parts->nAdds),
                                      static_cast<uint32_t>(parts->nWires));
    }
    // The gathered columns zero-padded by the gather itself; the others zeroed here.
    const std::vector<uint64_t> &positions = key.witnessColumns();
    for (uint64_t c = parts->mapRows > 0 ? parts->mapCols : 0; c < C; ++c) {
        pilfflonk_gpu_memset_zero(columns + positions[c] * N, N * fr);
    }
    for (uint64_t c = 0; parts->mapRows > 0 && c < parts->mapCols; ++c) {
        gpu_plonk_gather_witness(columns + positions[c] * N, parts->map + c * parts->mapRows * sizeof(uint32_t), wires,
                                 internal, static_cast<uint32_t>(parts->nWires), parts->mapRows, N);
    }
    pilfflonk_gpu_memset_zero(counts, (TABLE_ROWS + RANGE_ROWS) * sizeof(uint32_t));
    uint32_t *tableCounts = reinterpret_cast<uint32_t *>(counts);
    pilfflonk_gpu_wrap_blake3(columns, air.witnessPositions(), N, blocks, exec.n_blocks, tableCounts,
                              tableCounts + TABLE_ROWS, reinterpret_cast<const uint32_t *>(hostRange));
    polyCounts.assign(key.info().cmPolsMap.size(), 0);
    committed.assign(key.info().cmPolsMap.size(), false);
}

std::vector<G1Point> InstanceGpu::commitStage(uint64_t stage, const FrElement *factors) {
    const ProofCall call(air.gpuKey());
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

    // Their counts, which the opening's components and the copies on demand read: the polynomials
    // stay in the arena.
    std::vector<uint64_t> found(counted);
    staging.toHost(found.data(), counts, counted * sizeof(uint64_t));
    counted = 0;
    for (const LayoutEntry &entry : info.layout) {
        if (entry.stage != stage) {
            continue;
        }
        for (uint64_t j = 0; j < entry.k; ++j) {
            polyCounts[entry.pols[j].id] = found[counted++];
            committed[entry.pols[j].id] = true;
        }
    }
    return commitments;
}

uint64_t InstanceGpu::polynomialCount(uint64_t cmId) const {
    if (cmId >= committed.size() || !committed[cmId]) {
        throw std::invalid_argument("InstanceGpu: the polynomial of cm " + std::to_string(cmId) +
                                    " is not committed");
    }
    return polyCounts[cmId];
}

std::unique_ptr<Poly> InstanceGpu::polynomialToHost(uint64_t cmId, FrElement *coefs) const {
    const ProofCall call(air.gpuKey());
    const uint64_t count = polynomialCount(cmId);
    const AirKey &key = air.airKey();
    const LayoutPosition &at = key.cmPosition(cmId);
    const uint64_t length = key.n() + key.blindLength(at.f);
    const FrElement *slot = elements(air.gpuKey().arena(), air.arena().polys) + air.arena().slot[at.f] + at.j * length;
    air.gpuKey().staging().toHost(coefs, slot, length * sizeof(FrElement));
    return mirrorPolynomial(coefs, length, count, key.name() + ": the column " + key.info().cmPolsMap[cmId].name);
}

void InstanceGpu::requireQParts(uint64_t partBits, const char *function) const {
    const ProofCall call(air.gpuKey());
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
    const ProofCall call(air.gpuKey());
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

    // Each phase is waited for when the timers are printed, so that each has its time; the default
    // stream orders the phases, and the count's copy below waits for them, either way.
    const bool timed = timersLogged();
    auto phaseDone = [timed] {
        if (timed) {
            gpu_plonk_cuda_device_sync();
        }
    };
    for (uint64_t p = 0; p < nParts; ++p) {
        TimerStartExpr(PILFFLONK_Q_EXTEND, p);
        lde.extendPart(p);
        phaseDone();
        TimerStopAndLogExpr(PILFFLONK_Q_EXTEND, p);
        TimerStartExpr(PILFFLONK_Q_DOMAIN, p);
        const ExpressionsDomainGpu domain = ExpressionsDomainGpu::cosetPart(
            info.nBits, key.degrees().nBitsExt, partBits, p, info.boundaries, zerofiers);
        phaseDone();
        TimerStopAndLogExpr(PILFFLONK_Q_DOMAIN, p);
        TimerStartExpr(PILFFLONK_Q_EVALUATE, p);
        air.expressions().calculateExpression(info.cExpId, domain, values, q + p, nParts);
        phaseDone();
        TimerStopAndLogExpr(PILFFLONK_Q_EVALUATE, p);
    }
    TimerStart(PILFFLONK_Q_INTERPOLATE);
    lde.interpolate();
    phaseDone();
    TimerStopAndLog(PILFFLONK_Q_INTERPOLATE);

    uint64_t *count = reinterpret_cast<uint64_t *>(arena + layout.qCounts);
    pilfflonk_gpu_count_coefficients(count, q, air.qPieceOffsets(), 1, M);
    uint64_t coefficients = 0;
    air.gpuKey().staging().toHost(&coefficients, count, sizeof(coefficients));
    return coefficients;
}

std::vector<G1Point> InstanceGpu::commitQ(const FrElement *factors) {
    const ProofCall call(air.gpuKey());
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
    const ProofCall call(air.gpuKey());
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
