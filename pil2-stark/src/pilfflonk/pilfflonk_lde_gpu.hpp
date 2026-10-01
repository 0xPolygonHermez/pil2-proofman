#ifndef PILFFLONK_LDE_GPU_HPP
#define PILFFLONK_LDE_GPU_HPP

// The LDE on the device: Lde's coset transforms (pilfflonk_lde.hpp) on data in device memory, and
// Q's on a key on the GPU (LdeGpu). Only a library built with the GPU (__USE_CUDA__) has it, in
// pilfflonk_lde_gpu.cpp; it calls the kernels of pilfflonk_lde.cu, sppark's NTT (transformOnDevice)
// and the PLONK GPU prover's tables of powers (gpu_plonk_precompute_omega_tables_async), on the
// device's legacy default stream, as pilfflonk_key_gpu.hpp says.
//
// The results are the CPU's Lde's bit for bit: the same fold, scaling and transforms, each value the
// same field element (pilfflonk_lde.cu), and every constant (the shifts and their powers) from the
// Lde itself.

#include <cstdint>
#include <vector>

#include "pilfflonk_key_gpu.hpp"
#include "pilfflonk_lde.hpp"

namespace PilFflonk {

// The scratch, in elements, that the functions below take for the tables of the powers of a shift
// (pilfflonk_lde_kernels.hpp) on an Lde of N' points: (N' >> 8) + 1 blocks and 256 powers.
uint64_t ldeTableElements(const Lde &lde);

// Lde::extendCosetPart of nCols columns on the device: column t, the nCoefs[t] coefficients at
// coefs[t], into its S = 2^partBits evaluations on part `part` of the coset, at evals + t·S. Each
// column has its own number of coefficients, 1 <= nCoefs[t] <= N'; nBits <= partBits <= nBitsExt
// and part < N'/S, as extendCosetPart takes them. coefs and nCoefs are host arrays, coefs of
// device pointers; evals (nCols·S elements) overlaps no column, and tables holds
// ldeTableElements(lde) elements of scratch. Its work is ordered as a kernel's
// (pilfflonk_kernels.hpp): what runs after it on the default stream, or a Staging copy, sees the
// evaluations.
void extendCosetPartOnDevice(const Lde &lde, const FrElement *const *coefs, const uint64_t *nCoefs, uint64_t nCols,
                             uint64_t partBits, uint64_t part, FrElement *evals, FrElement *tables);

// Lde::interpolateCoset of one column on the device, in place: the N' evaluations on g·H' at
// `values` into its N' coefficients. tables holds ldeTableElements(lde) elements of scratch. Ordered
// as extendCosetPartOnDevice.
void interpolateCosetOnDevice(const Lde &lde, FrElement *values, FrElement *tables);

// Q's LDE on a key on the GPU (Instance::commitQ), for the instance that holds the key's arena
// (GpuKey::Lease) with every stage committed: each column Q's code reads (AirKey::qReads), from its
// polynomial where the device keeps it (the fixed columns' coefficients of the GpuAirKey, the
// committed polynomials in the arena), extended there to each part of the coset in turn, and Q
// interpolated there, in the arena's Q phase (ArenaLayout::q). Until the interpreter runs on the
// device, it reads the columns and writes Q's values in the key's host buffer for Q (GpuKey::qHost):
// each part's columns are copied to it, and Q's values go to the device and back for their
// interpolation. Nothing else crosses: no coefficient goes to the device.
class LdeGpu {
public:
    // Q in parts of 2^partBits points (Instance::setQPartBits), nBits <= partBits <= nBitsExt. The
    // key's host buffer for Q grows to N' + |qReads|·2^partBits elements if it holds fewer.
    LdeGpu(const GpuAirKey &air, uint64_t partBits);

    // Q's N' values, in the key's host buffer for Q, and after them the S values on a part of each
    // column Q reads, column r of qReads at partColumns() + r·S.
    FrElement *values() const { return hostValues; }
    FrElement *partColumns() const { return hostColumns; }

    // Part `part` of the coset (part < N'/S): every column extended to it on the device
    // (extendCosetPartOnDevice), as many at a time as the arena's Q phase holds, and copied to
    // partColumns(). Returns once they are there.
    void extendPart(uint64_t part) const;

    // Q's N' values, in values(), into its coefficients there: copied to the device, interpolated
    // (interpolateCosetOnDevice) and copied back.
    void interpolate() const;

private:
    const GpuAirKey &air;
    uint64_t partBits;
    std::vector<const FrElement *> sources; // by column of qReads, its polynomial on the device
    std::vector<uint64_t> lengths;          // and its coefficients, as many as its host copy has
    FrElement *hostValues = nullptr;
    FrElement *hostColumns = nullptr;
};

} // namespace PilFflonk

#endif
