#ifndef PIL2_FFLONK_FR_EVALUATOR_HPP
#define PIL2_FFLONK_FR_EVALUATOR_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fr_expressions.hpp"
#include "fr_pack.hpp"

// The BN254 expression interpreter -- the counterpart of
// ExpressionsPack::calculateExpressions in pil2-stark.
//
// The bytecode is a flat sequence of operations. Each one occupies eight
// uint16 words:
//
//     args[i + 0]        field opcode: 0 add, 1 sub, 2 mul, 3 reverse sub
//     args[i + 1]        destination temporary index
//     args[i + 2 .. 4]   operand A: (type, position, stride index)
//     args[i + 5 .. 7]   operand B: same
//
// An operand is a constant when its type tag lies past the temporary buffers;
// the loader returns a single value for those and op_pack broadcasts it.
//
// The result of the final operation is the expression's value; every earlier
// operation writes to the temporary named by args[i + 1].
//
// Goldilocks encodes the operand dimensions in a parallel `ops` array (0 for
// dim1-dim1, 1 for dim3-dim1, 2 for dim3-dim3) and steps i_args by a different
// amount per case. Over BN254 every operand is dim 1, so only case 0 occurs and
// the stride is always eight. A non-zero op code means the bytecode was built
// for a different field, which is checked rather than assumed.

namespace PilFflonk {

/// Words per operation for the dim1-dim1 case, the only one BN254 uses.
constexpr uint64_t ARGS_PER_OP = 8;

/// Where one expression's bytecode lives and how long it is.
struct ParserParams {
    uint64_t nOps = 0;
    uint64_t opsOffset = 0;
    uint64_t nArgs = 0;
    uint64_t argsOffset = 0;
};

/// The two flat arrays the bytecode is split across.
struct ParserArgs {
    const uint8_t *ops = nullptr;
    const uint16_t *args = nullptr;
};

/// Evaluate one expression for `nrowsPack` consecutive rows starting at `row`.
///
/// `tmp` must hold `maxTmp * nrowsPack` elements; `out` receives `nrowsPack`
/// results. Returns the number of argument words consumed, which the caller can
/// check against `ParserParams::nArgs`.
inline uint64_t evaluateExpression(const ExpressionsLayout &layout, const StepsParams &params,
                                   const ParserParams &pp, const ParserArgs &pa, FrEl *tmp, FrEl *out,
                                   uint64_t row, uint64_t nrowsPack, uint64_t domainSize, bool domainExtended,
                                   bool isCyclic) {
    if (pa.args == nullptr || pa.ops == nullptr) {
        throw std::runtime_error("PilFflonk::evaluateExpression: bytecode not bound");
    }
    if (pp.nOps == 0) {
        throw std::runtime_error("PilFflonk::evaluateExpression: expression has no operations");
    }

    const uint16_t *args = &pa.args[pp.argsOffset];
    const uint8_t *ops = &pa.ops[pp.opsOffset];

    // Scratch for the two operands when they are not already contiguous.
    std::vector<FrEl> valueA(nrowsPack), valueB(nrowsPack);

    FrEl *tmpBuffers[1] = {tmp};
    const uint64_t constantThreshold = layout.bufferCommitsSize() + 1;

    uint64_t i_args = 0;
    for (uint64_t k = 0; k < pp.nOps; ++k) {
        if (ops[k] != 0) {
            throw std::runtime_error(
                "PilFflonk::evaluateExpression: operation " + std::to_string(k) + " has dimension case " +
                std::to_string((uint64_t)ops[k]) +
                "; BN254 expressions are dim 1 throughout, so this bytecode was built for another field");
        }

        const FrEl *a = load(layout, params, valueA.data(), tmpBuffers, args, i_args + 2, row, nrowsPack, domainSize,
                             domainExtended, isCyclic);
        const FrEl *b = load(layout, params, valueB.data(), tmpBuffers, args, i_args + 5, row, nrowsPack, domainSize,
                             domainExtended, isCyclic);

        const bool isConstantA = args[i_args + 2] > constantThreshold;
        const bool isConstantB = args[i_args + 5] > constantThreshold;

        // The last operation yields the expression's value; the rest feed
        // temporaries.
        FrEl *res = (k == pp.nOps - 1) ? out : &tmp[args[i_args + 1] * nrowsPack];

        op_pack(AltBn128::Engine::engine, nrowsPack, args[i_args], res, a, isConstantA, b, isConstantB);

        i_args += ARGS_PER_OP;
    }

    if (pp.nArgs != 0 && i_args != pp.nArgs) {
        throw std::runtime_error("PilFflonk::evaluateExpression: consumed " + std::to_string(i_args) +
                                 " argument words but the expression declares " + std::to_string(pp.nArgs));
    }

    return i_args;
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_EVALUATOR_HPP
