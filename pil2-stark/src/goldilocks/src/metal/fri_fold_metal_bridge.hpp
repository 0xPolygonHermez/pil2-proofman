#ifndef PIL2_FRI_FOLD_METAL_BRIDGE_HPP
#define PIL2_FRI_FOLD_METAL_BRIDGE_HPP

#include "../platform.hpp"

#if PIL2_HAS_METAL

#include "metal_c_api.h"
#include "../goldilocks_base_field.hpp"
#include "metal_context.hpp"
#include <cstdint>
#include <vector>

namespace pil2 { namespace metal {

// Thin adapter from FRI<Goldilocks::Element>::fold's per-step work to
// pil2::metal::fri_fold_w8_metal. Pre-computes the four scalars and the
// 8-th roots table the kernel needs, then dispatches.
//
// Contract: `pol` has `nX * pol2N * FIELD_EXTENSION` u64s on input. On
// return, the first `pol2N * FIELD_EXTENSION` u64s hold the folded
// polynomial (exactly matches the CPU semantic). Caller is responsible
// for calling this only when step != 0 and nX == 8.
inline void fri_fold_w8_via_metal(Goldilocks::Element* pol,
                                  const Goldilocks::Element* challenge,
                                  uint64_t nBitsExt,
                                  uint64_t prevBits,
                                  uint64_t currentBits) {
    static_assert(sizeof(Goldilocks::Element) == sizeof(uint64_t),
                  "Goldilocks::Element must be layout-compatible with uint64_t");

    const uint64_t pol2N = 1ULL << currentBits;
    constexpr uint64_t nX = 8;

    // polShiftInv with (nBitsExt - prevBits) squarings applied when
    // step != 0 (matches FRI::fold lines 36-43). Caller guarantees
    // step != 0 at this entry.
    Goldilocks::Element polShiftInv = Goldilocks::inv(Goldilocks::shift());
    for (uint64_t j = 0; j < nBitsExt - prevBits; ++j) {
        polShiftInv = polShiftInv * polShiftInv;
    }

    const uint64_t polShiftInv_u = Goldilocks::toU64(polShiftInv);
    const uint64_t wi_u          = Goldilocks::toU64(Goldilocks::inv(Goldilocks::w(prevBits)));
    const uint64_t inv8_u        = Goldilocks::toU64(Goldilocks::inv(Goldilocks::fromU64(nX)));

    // 8-th roots of unity (w = Goldilocks::w(3))
    std::vector<uint64_t> roots8(nX);
    const Goldilocks::Element w = Goldilocks::w(3);
    Goldilocks::Element acc = Goldilocks::one();
    for (uint64_t k = 0; k < nX; ++k) {
        roots8[k] = Goldilocks::toU64(acc);
        acc = acc * w;
    }

    // challenge is 3 consecutive Goldilocks::Element — layout-compat with u64[3].
    const uint64_t* challenge_u = reinterpret_cast<const uint64_t*>(challenge);

    fri_fold_w8_metal(get_context(),
                      reinterpret_cast<uint64_t*>(pol),
                      challenge_u,
                      pol2N,
                      polShiftInv_u, wi_u, inv8_u,
                      roots8.data());
}

}} // namespace pil2::metal

#endif // PIL2_HAS_METAL
#endif // PIL2_FRI_FOLD_METAL_BRIDGE_HPP
