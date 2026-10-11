#ifndef PIL2_FFLONK_FR_HINTS_HPP
#define PIL2_FFLONK_FR_HINTS_HPP

#include <cstdint>
#include <stdexcept>

#include "fr_pack.hpp"

// Accumulation primitives for the BN254 evaluator -- the counterpart of
// accOperation in pil2-stark/src/starkpil/hints.cu.
//
// This is where pil2's arguments are actually computed. pil1 hardcoded plookup
// h1/h2 and a bespoke grand-product Z; pil2 replaced both with hint-driven
// accumulations over a column, so `std_prod` (permutations, connections) and
// `std_sum` (lookups via logup) are the same scan with a different operator.
//
// The scan is inclusive and sequential: vals[i] becomes vals[0] op ... op
// vals[i], leaving the total in the last entry. The grand-product argument
// depends on exactly that -- the running value at each row is the partial
// product, and the final entry is the product over the whole column.
//
// Goldilocks carries a dim parameter to select between the base field and the
// cubic extension. BN254 has no extension, so there is no dim here.

namespace PilFflonk {

/// In-place inclusive prefix scan over `n` field elements.
///
/// `add == true` accumulates with addition (logup / GSum), `false` with
/// multiplication (grand product / GProd).
inline void accOperation(AltBn128::Engine &E, FrEl *vals, uint64_t n, bool add) {
    if (n == 0) return;
    if (vals == nullptr) throw std::runtime_error("PilFflonk::accOperation: null buffer");

    if (add) {
        for (uint64_t i = 1; i < n; ++i) vals[i] = E.fr.add(vals[i], vals[i - 1]);
    } else {
        for (uint64_t i = 1; i < n; ++i) vals[i] = E.fr.mul(vals[i], vals[i - 1]);
    }
}

/// The accumulated total, i.e. the last entry after a scan.
///
/// For a grand product this is the value the argument must match against its
/// counterpart column; a mismatch is what makes an invalid permutation fail.
inline FrEl accTotal(const FrEl *vals, uint64_t n) {
    if (n == 0) throw std::runtime_error("PilFflonk::accTotal: empty buffer has no total");
    return vals[n - 1];
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_HINTS_HPP
