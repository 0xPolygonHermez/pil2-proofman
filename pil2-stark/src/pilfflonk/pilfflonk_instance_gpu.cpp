// The device side of an Instance (pilfflonk_instance_gpu.hpp): compiled into the GPU library only
// (the Makefile's %_gpu.cpp rule), with g++.
#include "pilfflonk_instance_gpu.hpp"

#include <string>

#include "pilfflonk_kernels.hpp"
#include "pilfflonk_proving_key.hpp"
#include "timer.hpp"

// The PLONK GPU prover's helper (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares it.
extern "C" void gpu_plonk_memcpy_d2d(void *dst, const void *src, size_t bytes);

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

} // namespace PilFflonk
