#ifndef NTT_GOLDILOCKS_NEON_HPP
#define NTT_GOLDILOCKS_NEON_HPP

#include "platform.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_neon.hpp"

#if PIL2_HAS_NEON

#include <arm_neon.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// NEON NTT butterfly primitive.
//
// The scalar inner loop in ntt_goldilocks.cpp does, for each twiddle w and
// each column k in [0, ncols):
//
//     t = w * a[offset1 + k]
//     u = a[offset2 + k]
//     a[offset2 + k] = u + t
//     a[offset1 + k] = u - t
//
// On NEON we can fold two columns into a single uint64x2_t since elements
// k and k+1 are adjacent in memory. Twiddle is constant across columns so
// it broadcasts via splat. Scalar tail handles odd ncols.
//
// strideA / offsetA from the caller are pre-baked into offset1 / offset2 by
// the caller — this primitive operates on a flat element array indexed by
// offset1+k / offset2+k.
//
// Bit-exact with the scalar path: gl_mul / gl_add / gl_sub all match
// Goldilocks::mul / add / sub bit-for-bit (including non-canonical results
// in [p, 2p) where the scalar path tolerates them). The existing
// test_ntt_cpu.cpp suite (NTT, INTT roundtrip, LDE_correctness) is the gate.
// ----------------------------------------------------------------------------
inline void ntt_neon_butterfly(Goldilocks::Element* a,
                               uint64_t offset1, uint64_t offset2,
                               Goldilocks::Element w, uint64_t ncols)
{
    namespace N = Goldilocks_neon;
    uint64x2_t w_vec = N::splat(w.fe);

    uint64_t k = 0;
    // Two columns per iter — adjacent in memory, twiddle broadcast.
    for (; k + 1 < ncols; k += 2) {
        uint64x2_t t = N::gl_mul(N::load(&a[offset1 + k]), w_vec);
        uint64x2_t u = N::load(&a[offset2 + k]);
        N::store(&a[offset2 + k], N::gl_add(u, t));
        N::store(&a[offset1 + k], N::gl_sub(u, t));
    }
    // Scalar tail for odd ncols.
    for (; k < ncols; ++k) {
        Goldilocks::Element t = w * a[offset1 + k];
        Goldilocks::Element u = a[offset2 + k];
        Goldilocks::add(a[offset2 + k], t, u);
        Goldilocks::sub(a[offset1 + k], u, t);
    }
}

// Per-column scalar-multiply: dst[k] = src[k] * scalar, for k in [0, ncols).
// Used by NTT_iters' "any phase" branch (twiddle-mul) and the INTT
// last-phase scaling (powTwoInv broadcast). Same NEON pattern as the
// butterfly — broadcast scalar, two columns per iter, scalar tail.
inline void ntt_neon_scale(Goldilocks::Element* dst, uint64_t dst_off,
                           const Goldilocks::Element* src, uint64_t src_off,
                           Goldilocks::Element scalar, uint64_t ncols)
{
    namespace N = Goldilocks_neon;
    uint64x2_t s_vec = N::splat(scalar.fe);

    uint64_t k = 0;
    for (; k + 1 < ncols; k += 2) {
        uint64x2_t v = N::load(&src[src_off + k]);
        N::store(&dst[dst_off + k], N::gl_mul(v, s_vec));
    }
    for (; k < ncols; ++k) {
        Goldilocks::mul(dst[dst_off + k], src[src_off + k], scalar);
    }
}

#endif  // PIL2_HAS_NEON
#endif  // NTT_GOLDILOCKS_NEON_HPP
