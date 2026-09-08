#ifndef PIL2_FFLONK_FR_EXPRESSIONS_HPP
#define PIL2_FFLONK_FR_EXPRESSIONS_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <alt_bn128.hpp>

#include "fr_pack.hpp"

// Operand addressing for the BN254 expression evaluator -- the counterpart of
// ExpressionsPack::load in pil2-stark/src/starkpil/expressions.
//
// The bytecode names an operand with a type tag and up to two indices:
//
//     type     = args[i_args]       which buffer
//     stagePos = args[i_args + 1]   which column within it
//     offset   = args[i_args + 2]   index into nextStrides, the row offset
//                                   used for next-row access (x')
//
// Buffers are row-major interleaved, so an operand resolves to
// `buffer[(row + o) * nCols + stagePos]`, or the same with the row index taken
// modulo the domain when the expression wraps.
//
// Two of Goldilocks' operand types cannot occur here:
//
//   * `xi`   (type == nStages + 3) is the FRI evaluation point and is dim-3
//     only -- the Goldilocks loader calls exit(-1) when asked for it at dim 1.
//     SHPLONK has no FRI, and its challenges are plain Fr.
//   * `tmp3` (type == bufferCommitsSize + 1) is the extension-field temporary.
//     Every BN254 temporary is dim 1.
//
// Both are rejected explicitly rather than left to fall through, so a bytecode
// that somehow names one fails loudly instead of reading the wrong buffer.

namespace PilFflonk {

/// The buffers an expression can read, over BN254.
///
/// Mirrors pil2-stark's StepsParams, minus the members that only exist for FRI
/// (xDivXSub) or the extension field.
struct StepsParams {
    FrEl *trace = nullptr;         ///< stage 1 committed polynomials
    FrEl *aux_trace = nullptr;     ///< later stages and temporaries
    FrEl *constPols = nullptr;     ///< constant polynomials, base domain
    FrEl *constPolsExt = nullptr;  ///< constant polynomials, extended domain
    FrEl *publicInputs = nullptr;
    FrEl *challenges = nullptr;
    FrEl *airValues = nullptr;
    FrEl *numbers = nullptr;       ///< literals from the bytecode
};

/// Layout of the buffers an expression reads: how many columns each stage has,
/// and where each stage's buffer starts.
struct ExpressionsLayout {
    uint64_t nStages = 0;
    uint64_t nConstants = 0;
    /// Columns per operand type, indexed by type tag.
    std::vector<uint64_t> mapSectionsN;
    /// Offset of each stage's buffer within aux_trace, indexed by type tag.
    std::vector<uint64_t> mapOffsets;
    /// Row offsets the bytecode can name, e.g. {0, 1} for current and next row.
    std::vector<int64_t> nextStrides;

    /// One past the last committed-buffer type tag. Temporaries start here.
    uint64_t bufferCommitsSize() const { return 1 + nStages + 3; }
    /// Type tag of the dim-1 temporary buffer.
    uint64_t tmpType() const { return bufferCommitsSize(); }
    /// Type tag of the dim-3 temporary, which BN254 never uses.
    uint64_t tmp3Type() const { return bufferCommitsSize() + 1; }
    /// Type tag of the FRI evaluation point, which BN254 never uses.
    uint64_t xiType() const { return nStages + 3; }
};

/// Resolve one operand into `value`, returning a pointer to `nrowsPack`
/// contiguous elements.
///
/// Returns a pointer into an existing buffer when the operand is already
/// contiguous (temporaries, numbers) and fills `value` otherwise, matching the
/// Goldilocks loader so callers need not know which happened.
inline const FrEl *load(const ExpressionsLayout &layout, const StepsParams &params, FrEl *value,
                        FrEl *const *tmpBuffers, const uint16_t *args, uint64_t i_args, uint64_t row,
                        uint64_t nrowsPack, uint64_t domainSize, bool domainExtended, bool isCyclic) {
    const uint64_t type = args[i_args];

    if (type == layout.xiType()) {
        throw std::runtime_error(
            "PilFflonk::load: operand type xi is FRI-specific and dim-3 only; it cannot appear in a BN254 AIR");
    }
    if (type == layout.tmp3Type()) {
        throw std::runtime_error(
            "PilFflonk::load: operand type tmp3 is the extension temporary; every BN254 temporary is dim 1");
    }

    // Constant polynomials.
    if (type == 0) {
        const FrEl *constPols = domainExtended ? params.constPolsExt : params.constPols;
        if (constPols == nullptr) throw std::runtime_error("PilFflonk::load: constant polynomials not bound");

        const uint64_t stagePos = args[i_args + 1];
        const int64_t o = layout.nextStrides.at(args[i_args + 2]);
        const uint64_t nCols = layout.nConstants;

        if (isCyclic) {
            for (uint64_t j = 0; j < nrowsPack; ++j) {
                const uint64_t l = (row + j + o) % domainSize;
                value[j] = constPols[l * nCols + stagePos];
            }
        } else {
            const uint64_t base = (row + o) * nCols + stagePos;
            for (uint64_t j = 0; j < nrowsPack; ++j) value[j] = constPols[base + j * nCols];
        }
        return value;
    }

    // Committed polynomials, by stage. Stage 1 lives in `trace`; later stages
    // are packed into `aux_trace` at their mapped offsets.
    if (type <= layout.nStages + 1) {
        const uint64_t stagePos = args[i_args + 1];
        const int64_t o = layout.nextStrides.at(args[i_args + 2]);
        const uint64_t nCols = layout.mapSectionsN.at(type);
        const bool inTrace = (type == 1 && !domainExtended);

        const FrEl *buffer = inTrace ? params.trace : params.aux_trace;
        if (buffer == nullptr) throw std::runtime_error("PilFflonk::load: trace buffer not bound");
        const uint64_t offset = inTrace ? 0 : layout.mapOffsets.at(type);

        if (isCyclic) {
            for (uint64_t j = 0; j < nrowsPack; ++j) {
                const uint64_t l = (row + j + o) % domainSize;
                value[j] = buffer[offset + l * nCols + stagePos];
            }
        } else {
            const uint64_t base = offset + (row + o) * nCols + stagePos;
            for (uint64_t j = 0; j < nrowsPack; ++j) value[j] = buffer[base + j * nCols];
        }
        return value;
    }

    // Dim-1 temporaries: already contiguous, one block per index.
    if (type == layout.tmpType()) {
        if (tmpBuffers == nullptr || tmpBuffers[0] == nullptr) {
            throw std::runtime_error("PilFflonk::load: temporary buffer not bound");
        }
        return &tmpBuffers[0][args[i_args + 1] * nrowsPack];
    }

    // Literals: a single value the caller broadcasts.
    if (params.numbers == nullptr) {
        throw std::runtime_error("PilFflonk::load: unhandled operand type " + std::to_string(type));
    }
    return &params.numbers[args[i_args + 1]];
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_EXPRESSIONS_HPP
