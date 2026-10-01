#ifndef PILFFLONK_INSTANCE_GPU_HPP
#define PILFFLONK_INSTANCE_GPU_HPP

// The device side of an Instance on a key on the GPU (pilfflonk_key_gpu.hpp): its witness, the
// commitments of its stages, and Q and its commitments, on the device, in the GpuKey's arena. Only a
// library built with the GPU (__USE_CUDA__) has it, in pilfflonk_instance_gpu.cpp.

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "pilfflonk_expressions.hpp"
#include "pilfflonk_key_gpu.hpp"

namespace PilFflonk {

// What the device does of an Instance (pilfflonk_prover.hpp) while the hints and the im pols run on
// the host: the stages commit what they get from the host and copy back what the host needs, and Q
// runs there whole, from the committed polynomials, with the scalars and the blinding factors the
// host gives it. The data stay in the arena of the key, which the instance holds from its
// construction (GpuKey::Lease) to its end, and where the opening reads them (OpeningGpu); their copies
// on the host are the instance's columns of stage 1 and the GpuKey's mirror of the committed
// polynomials, and Q's pieces only when they are asked for (qPiecesToHost). Everything is the CPU's
// bit for bit: the same INTTs and MSMs, the same field operations in the same order on each element,
// and the blinding factors the host drew.
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

    // What Q's code reads on a part of S points of the coset: column r of AirKey::qReads at
    // columns + r·S, on the device, and the instance's scalars (Instance::commitQ's ProverValues).
    using QValues = std::function<ProverValues(const FrElement *columns, uint64_t S)>;

    // Throws std::invalid_argument, naming `function`, if the key's arena (GpuKey::arenaSize) cannot
    // hold Q's phase in parts of 2^partBits points (qPhaseBytes), saying how many bytes it needs and
    // the arena has (decisions D1 and D5: no fallback, and no other part size than asked). The
    // default parts, of 2^nBits points, it always holds (arenaLayout).
    void requireQParts(uint64_t partBits, const char *function) const;

    // Instance::commitQ's Q, once every stage is committed (pilfflonk/docs/protocol.md#q-in-parts):
    // requireQParts first, before any work on the device; then on each part of the coset of
    // 2^partBits points (Instance::setQPartBits), each column of qReads extended to the part (LdeGpu),
    // Zi of the part
    // (ExpressionsDomainGpu::cosetPart) and Q's code (cExpId) evaluated there by the key's interpreter
    // (GpuAirKey::expressions) with values(columns, S), point i of part p into point p + nParts·i of
    // Q's values at ArenaLayout::q; then Q interpolated there. Returns 1 + the degree of Q's highest
    // coefficient not zero, 0 if Q is zero: commitQ checks it against Q's bound. Throws as
    // requireQParts, and what Expressions::calculateExpression throws for Q's code.
    uint64_t computeQ(uint64_t partBits, const QValues &values);

    // Instance::commitQ's pieces of Q and their commitments, from the coefficients computeQ left:
    // piece i (qPieceRange) at ArenaLayout::qPieces, zero above its own, and the blinding of the
    // boundaries between them with `factors`, two per boundary as Instance::drawQBlinding drew them
    // (pilfflonk_gpu_blind_q_boundaries); their degrees (qPieceDegree); and each f of Q's stage, its
    // pieces packed, committed (GpuKey::commit), in the order of the layout. The pieces stay on the
    // device for the opening (OpeningGpu::ofInstance).
    std::vector<G1Point> commitQ(const FrElement *factors);

    // The degree of piece i of Q that commitQ found (0 if the piece is zero).
    uint64_t qPieceDegree(uint64_t i) const;

    // Q's pieces after commitQ, copied to `coefs` on the host (the Σ AirDegrees::qPieceCoefficients
    // elements of their bounds, piece i after those of the pieces before it), and polynomials over
    // them (mirrorPolynomial): for tests and diagnostics (Instance::qPiece). Throws
    // std::logic_error if a copy's degree is not the device's.
    std::vector<std::unique_ptr<Poly>> qPiecesToHost(FrElement *coefs) const;

private:
    const GpuAirKey &air;
    GpuKey::Lease lease;
    std::vector<uint64_t> qPieceCounts; // by piece: 1 + its degree, 0 if it is zero
};

} // namespace PilFflonk

#endif
