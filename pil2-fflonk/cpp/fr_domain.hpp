#ifndef PIL2_FFLONK_FR_DOMAIN_HPP
#define PIL2_FFLONK_FR_DOMAIN_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fr_dest.hpp"
#include "fr_evaluator.hpp"

// The domain-level driver: evaluate an expression over every row.
//
// This is the outer loop of pil2-stark's ExpressionsPack::calculateExpressions.
// It walks the domain in blocks of NROWS_PACK, evaluates the expression for
// each block and stores the results, which is where the blocking factor
// actually pays: temporaries are reused across the whole block and the
// destination is written with one strided pass.
//
// Two details matter for correctness at the edges:
//
//   * The last block is short whenever the domain is not a multiple of the
//     block size, so the tail is evaluated with a smaller nrowsPack rather
//     than reading past the trace.
//   * A block that straddles the domain boundary can only be evaluated
//     cyclically. The Goldilocks driver decides this per block; the same rule
//     applies here, since a next-row access on the final row must wrap.

namespace PilFflonk {

/// Evaluate `pp` across `domainSize` rows, writing results through `dest`.
///
/// `maxTmp` is the number of temporaries the expression uses; the scratch
/// buffer is sized from it. `nrowsPack` is the block size, clamped to the
/// domain.
inline void calculateExpression(const ExpressionsLayout &layout, const StepsParams &params, const ParserParams &pp,
                                const ParserArgs &pa, const Dest &dest, uint64_t domainSize, uint64_t maxTmp,
                                bool domainExtended, uint64_t nrowsPack = NROWS_PACK) {
    if (domainSize == 0) return;
    if (nrowsPack == 0) throw std::runtime_error("PilFflonk::calculateExpression: block size must be non-zero");

    nrowsPack = std::min(nrowsPack, domainSize);

    // The largest row offset the expression can name decides which blocks need
    // cyclic handling: any block whose last row plus that offset leaves the
    // domain must wrap.
    int64_t maxStride = 0;
    for (int64_t s : layout.nextStrides) maxStride = std::max(maxStride, s);

    std::vector<FrEl> tmp((maxTmp + 1) * nrowsPack);
    std::vector<FrEl> out(nrowsPack);

    for (uint64_t row = 0; row < domainSize; row += nrowsPack) {
        // Short final block rather than reading past the trace.
        const uint64_t rows = std::min(nrowsPack, domainSize - row);

        // Only the blocks that actually run off the end pay for the modulo.
        const bool isCyclic = (row + rows + (uint64_t)maxStride) > domainSize;

        evaluateExpression(layout, params, pp, pa, tmp.data(), out.data(), row, rows, domainSize, domainExtended,
                           isCyclic);

        storeValues(dest, out.data(), row, rows, /*isConstant=*/false);
    }
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_DOMAIN_HPP
