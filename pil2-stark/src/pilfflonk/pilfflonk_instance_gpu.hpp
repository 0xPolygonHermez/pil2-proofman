#ifndef PILFFLONK_INSTANCE_GPU_HPP
#define PILFFLONK_INSTANCE_GPU_HPP

// The device side of an Instance on a key on the GPU (pilfflonk_key_gpu.hpp): its witness and the
// commitments of its stages, on the device, in the GpuKey's arena. Only a library built with the GPU
// (__USE_CUDA__) has it, in pilfflonk_instance_gpu.cpp.

#include <cstdint>
#include <memory>
#include <vector>

#include "pilfflonk_key_gpu.hpp"

namespace PilFflonk {

// What the device does of an Instance (pilfflonk_prover.hpp) while the hints, the im pols, Q and the
// opening run on the host: each commits what it gets from the host and copies back what the host
// needs. The data stay in the arena of the key, which the instance holds from its construction
// (GpuKey::Lease) to its end, and their copies on the host are the instance's columns of stage 1 and
// the GpuKey's mirror of the committed polynomials. Everything is the CPU's bit for bit: the same
// INTTs and MSMs, the same field operations in the same order on each element, and the blinding
// factors the host drew.
class InstanceGpu {
public:
    // The device side of an instance of `air`: waits for the key's arena (GpuKey::Lease), copies to
    // it stage1, the witness as Instance takes it (N rows of C canonical scalars), transposes it there
    // into the columns of stage 1 in Montgomery form, and copies those to stageOne, the instance's
    // columns of stage 1 on the host (column p at p·N). Throws std::invalid_argument, before it
    // waits, if this thread holds the arena for another instance.
    InstanceGpu(const GpuAirKey &air, const uint8_t *stage1, FrElement *stageOne);
    InstanceGpu(const InstanceGpu &) = delete;
    InstanceGpu &operator=(const InstanceGpu &) = delete;

    // Instance::commitF of `stage` on the device, given what the host computed: copies to the device
    // the stage's columns on H the host computed (GpuAirKey::hostColumns) from `columns` (column p at
    // p·N), and `factors`, the blinding factors of the stage's f as Instance::drawBlinding drew them;
    // then for each f of the stage, in the order of the layout: the INTT of each column into its slot
    // in the arena, its blinding, the count of its coefficients, and the commitment of f
    // (GpuKey::commit). Then the polynomials go to the key's mirror on the host, and polys (by
    // cmPolsMap index) gets the polynomials over them. The commitments, in the order of the layout.
    std::vector<G1Point> commitStage(uint64_t stage, const FrElement *columns, const FrElement *factors,
                                     std::vector<std::unique_ptr<Poly>> &polys);

private:
    const GpuAirKey &air;
    GpuKey::Lease lease;
};

} // namespace PilFflonk

#endif
