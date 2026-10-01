// The LDE on the device (pilfflonk_lde_gpu.hpp): compiled into the GPU library only (the Makefile's
// %_gpu.cpp rule), with g++. It calls the kernels of pilfflonk_lde.cu and the PLONK GPU prover's
// helper (rapidsnark/plonk_prover.cu) through their C linkage.
#include "pilfflonk_lde_gpu.hpp"

#include <algorithm>

#include "pilfflonk_lde_kernels.hpp"
#include "pilfflonk_proving_key.hpp"

// The PLONK GPU prover's helper (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares it.
extern "C" void gpu_plonk_precompute_omega_tables_async(void *dBases, void *dTid, const void *omega4xPtr,
                                                        uint32_t blockSize, uint32_t numBlocks, void *stream);

namespace PilFflonk {

namespace {

// The block of the tables of powers: blocks[k] = b^(256·k), and the 256 powers b^t.
constexpr uint64_t TABLE_BLOCK = 256;

FrElement *elements(uint8_t *arena, uint64_t offset) { return reinterpret_cast<FrElement *>(arena + offset); }

// The tables of the powers of `base` (pilfflonk_lde_kernels.hpp) for every index below n,
// 1 <= n <= N', in `tables` (ldeTableElements(lde) elements): the blocks at its start, and the
// powers after the blocks of every index below N'.
struct PowerTables {
    const FrElement *blocks;
    const FrElement *powers;
};

PowerTables powersOf(const Lde &lde, const FrElement &base, uint64_t n, FrElement *tables) {
    FrElement *blocks = tables, *powers = tables + lde.extendedSize() / TABLE_BLOCK + 1;
    const uint64_t nBlocks = (n + TABLE_BLOCK - 1) / TABLE_BLOCK;
    gpu_plonk_precompute_omega_tables_async(blocks, powers, &base, static_cast<uint32_t>(TABLE_BLOCK),
                                            static_cast<uint32_t>(nBlocks), nullptr);
    return PowerTables{blocks, powers};
}

} // namespace

uint64_t ldeTableElements(const Lde &lde) { return lde.extendedSize() / TABLE_BLOCK + 1 + TABLE_BLOCK; }

void extendCosetPartOnDevice(const Lde &lde, const FrElement *const *coefs, const uint64_t *nCoefs, uint64_t nCols,
                             uint64_t partBits, uint64_t part, FrElement *evals, FrElement *tables) {
    if (nCols == 0) {
        return;
    }
    const uint64_t S = uint64_t(1) << partBits;
    const FrElement shift = lde.partShift(part);
    const FrElement shiftS = power(shift, S);
    // The fold reads the powers c^r for r below S and below the most coefficients of a column.
    const uint64_t longest = *std::max_element(nCoefs, nCoefs + nCols);
    const PowerTables t = powersOf(lde, shift, std::min(S, longest), tables);
    const std::vector<const void *> sources(coefs, coefs + nCols);
    pilfflonk_gpu_fold_by_powers(evals, S, sources.data(), nCoefs, nCols, t.blocks, t.powers, &shiftS);
    for (uint64_t c = 0; c < nCols; ++c) {
        transformOnDevice(evals + c * S, partBits, false);
    }
}

void interpolateCosetOnDevice(const Lde &lde, FrElement *values, FrElement *tables) {
    const uint64_t M = lde.extendedSize();
    transformOnDevice(values, static_cast<uint64_t>(__builtin_ctzll(M)), true);
    const PowerTables t = powersOf(lde, lde.shiftInverse(), M, tables);
    pilfflonk_gpu_mul_by_powers(values, M, t.blocks, t.powers);
}

LdeGpu::LdeGpu(const GpuAirKey &_air, uint64_t _partBits, FrElement *_columns)
    : air(_air), partBits(_partBits), columns(_columns) {
    const AirKey &key = air.airKey();
    const uint64_t N = key.n();
    const FrElement *committed = elements(air.gpuKey().arena(), air.arena().polys);
    for (const ColumnRead &c : key.qReads()) {
        if (c.type == 0) {
            sources.push_back(air.fixedCoefficients() + c.index * N);
            lengths.push_back(N);
            continue;
        }
        const LayoutPosition &at = key.cmPosition(key.cmIds()[c.type][c.index]);
        const uint64_t length = N + key.blindLength(at.f);
        sources.push_back(committed + air.arena().slot[at.f] + at.j * length);
        lengths.push_back(length);
    }
}

void LdeGpu::extendPart(uint64_t part) const {
    extendCosetPartOnDevice(air.airKey().lde(), sources.data(), lengths.data(), sources.size(), partBits, part,
                            columns, elements(air.gpuKey().arena(), air.arena().qTables));
}

void LdeGpu::interpolate() const {
    uint8_t *arena = air.gpuKey().arena();
    interpolateCosetOnDevice(air.airKey().lde(), elements(arena, air.arena().q), elements(arena, air.arena().qTables));
}

} // namespace PilFflonk
