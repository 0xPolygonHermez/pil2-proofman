#ifndef PIL2_FFLONK_FR_DEST_HPP
#define PIL2_FFLONK_FR_DEST_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fr_pack.hpp"

// Result writeback for the BN254 evaluator -- the counterpart of
// storePolynomial, multiplyPolynomials and getInversePolinomial in
// pil2-stark's ExpressionsPack.
//
// A Dest names where an expression's values land and how they are combined
// first. Three things can happen between the interpreter producing a block of
// values and those values reaching the destination buffer:
//
//   1. inversion, when the expression is used as a denominator;
//   2. multiplication, when the destination combines two expressions;
//   3. the strided store itself.
//
// All three collapse considerably at dim 1: the Goldilocks versions branch on
// dim and carry a FIELD_EXTENSION-wide scratch buffer, none of which applies
// over BN254.

namespace PilFflonk {

/// Where an expression's values are written.
struct Dest {
    FrEl *dest = nullptr;
    /// Stride between consecutive rows in `dest`. Zero means tightly packed,
    /// matching the Goldilocks convention of substituting the dimension.
    uint64_t offset = 0;
    /// Number of rows the destination spans.
    uint64_t domainSize = 0;

    uint64_t stride() const { return offset != 0 ? offset : 1; }
};

/// Invert a block of values in place.
///
/// `batch` uses a single inversion for the whole block (Montgomery's trick, as
/// the Goldilocks path does); otherwise each value is inverted individually.
/// Batch inversion is not defined when any value is zero, so that is rejected
/// rather than silently producing a wrong result.
inline void invertValues(AltBn128::Engine &E, FrEl *vals, uint64_t n, bool batch) {
    if (n == 0) return;
    if (vals == nullptr) throw std::runtime_error("PilFflonk::invertValues: null buffer");

    for (uint64_t i = 0; i < n; ++i) {
        if (E.fr.isZero(vals[i])) {
            throw std::runtime_error("PilFflonk::invertValues: value at index " + std::to_string(i) +
                                     " is zero and cannot be inverted");
        }
    }

    if (!batch) {
        for (uint64_t i = 0; i < n; ++i) E.fr.inv(vals[i], vals[i]);
        return;
    }

    // Montgomery batch inversion: prefix products, one inversion, then unwind.
    std::vector<FrEl> prefix(n);
    prefix[0] = vals[0];
    for (uint64_t i = 1; i < n; ++i) prefix[i] = E.fr.mul(prefix[i - 1], vals[i]);

    FrEl acc;
    E.fr.inv(acc, prefix[n - 1]);

    for (uint64_t i = n; i-- > 1;) {
        const FrEl inv_i = E.fr.mul(acc, prefix[i - 1]);
        acc = E.fr.mul(acc, vals[i]);
        vals[i] = inv_i;
    }
    vals[0] = acc;
}

/// Combine two expressions into one destination by multiplication.
///
/// `a` and `b` are blocks of `nrowsPack` values; either may be a single value
/// broadcast across the block. The product lands in `a`.
inline void multiplyValues(AltBn128::Engine &E, FrEl *a, bool constA, const FrEl *b, bool constB,
                           uint64_t nrowsPack) {
    op_pack(E, nrowsPack, (uint64_t)FrOp::mul, a, a, constA, b, constB);
}

/// Write a block of values into the destination at `row`, honouring its stride.
///
/// When `isConstant`, the expression evaluated to a single value and every row
/// receives it.
inline void storeValues(const Dest &dest, const FrEl *vals, uint64_t row, uint64_t nrowsPack, bool isConstant) {
    if (dest.dest == nullptr) throw std::runtime_error("PilFflonk::storeValues: destination not bound");

    const uint64_t stride = dest.stride();
    if (dest.domainSize != 0 && row + nrowsPack > dest.domainSize) {
        throw std::runtime_error("PilFflonk::storeValues: writing rows " + std::to_string(row) + ".." +
                                 std::to_string(row + nrowsPack) + " past domain size " +
                                 std::to_string(dest.domainSize));
    }

    for (uint64_t j = 0; j < nrowsPack; ++j) {
        dest.dest[(row + j) * stride] = isConstant ? vals[0] : vals[j];
    }
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_DEST_HPP
