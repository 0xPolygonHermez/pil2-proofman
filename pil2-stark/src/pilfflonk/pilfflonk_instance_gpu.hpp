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
#include "pilfflonk_wrap_exec.hpp"

namespace PilFflonk {

// What the device does of an Instance (pilfflonk_prover.hpp): its witness goes up once, its stages'
// columns (computeStageColumns, pilfflonk_hints_gpu.hpp) and commitments, and Q whole, run there from
// the committed polynomials, with the scalars and the blinding factors the host gives it. The data
// stay in the arena of the key, which the instance holds from its construction (GpuKey::Lease) to its
// end, and where the opening reads them (OpeningGpu); nothing of them comes back to the host but the
// counts of the polynomials' coefficients, and copies on demand (polynomialToHost, qPiecesToHost,
// stageColumnToHost) for tests and diagnostics. Everything is the CPU's bit for bit: the same INTTs
// and MSMs, the same field operations in the same order on each element, and the blinding factors
// the host drew. Each of its calls but the counts' holds a ProofCall (pilfflonk_key_gpu.hpp): it runs
// on the GPU's device, and its thread is the one that holds the key's arena.
class InstanceGpu {
public:
    // The device side of an instance of `air`: waits for the key's arena (GpuKey::Lease), copies to
    // it stage1, the witness as Instance takes it (N rows of C canonical scalars), and transposes it
    // there into the columns of stage 1 in Montgomery form (ArenaLayout::evaluations). Throws
    // std::invalid_argument, before it waits, if this thread holds the arena for another instance.
    InstanceGpu(const GpuAirKey &air, const uint8_t *stage1);
    // As the one above, of the wrap's witness as its parts (pilfflonk_wrap_exec.hpp), which the
    // device builds into the columns of stage 1, its scratch in the arena's `work`. Throws
    // std::invalid_argument if the AIR is not the wrap's or `work` cannot hold the parts.
    InstanceGpu(const GpuAirKey &air, const pilfflonk_exec_witness &exec);
    InstanceGpu(const InstanceGpu &) = delete;
    InstanceGpu &operator=(const InstanceGpu &) = delete;

    // Instance::commitF of `stage` on the device, whose columns computeStageColumns has computed:
    // copies to the device `factors`, the blinding factors of the stage's f as Instance::drawBlinding
    // drew them; then for each f of the stage, in the order of the layout: the INTT of each column into
    // its slot in the arena, its blinding, the count of its coefficients, and the commitment of f
    // (GpuKey::commit). The counts come back (polynomialCount). The commitments, in the order of the
    // layout.
    std::vector<G1Point> commitStage(uint64_t stage, const FrElement *factors);

    // Of the committed polynomial of column cmId (cmPolsMap), once its stage is committed: 1 + its
    // degree, 0 if it is zero, as the device counted it; and its N + blindLength(f) coefficients
    // copied to `coefs` on the host, a polynomial over them (mirrorPolynomial), for tests and
    // diagnostics (Instance::polynomial). Throw std::invalid_argument if its stage is not committed.
    uint64_t polynomialCount(uint64_t cmId) const;
    std::unique_ptr<Poly> polynomialToHost(uint64_t cmId, FrElement *coefs) const;

    // What Q's code reads on a part of S points of the coset: column r of AirKey::qReads at
    // columns + r·S, on the device, and the instance's scalars (Instance::commitQ's ProverValues).
    using QValues = std::function<ProverValues(const FrElement *columns, uint64_t S)>;

    // Throws std::invalid_argument, naming `function`, if the key's arena (GpuKey::arenaSize) cannot
    // hold Q's phase in parts of 2^partBits points (qPhaseBytes), saying how many bytes it needs and
    // the arena has: no fallback, and no other part size than asked
    // (pilfflonk/docs/performance.md#rules-of-the-device-path). The default parts, of 2^nBits
    // points, it always holds (arenaLayout).
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
    std::vector<uint64_t> polyCounts;   // by cmPolsMap index, once its stage is committed: 1 + its degree
    std::vector<bool> committed;        // by cmPolsMap index: its polynomial is in the arena
    std::vector<uint64_t> qPieceCounts; // by piece: 1 + its degree, 0 if it is zero
};

} // namespace PilFflonk

#endif
