#ifndef PIL2_NTT_METAL_BRIDGE_HPP
#define PIL2_NTT_METAL_BRIDGE_HPP

#include "../platform.hpp"

#if PIL2_HAS_METAL

#include "metal_c_api.h"
#include "../goldilocks_base_field.hpp"
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace pil2 { namespace metal {

namespace detail {
// Roots-table cache. Keyed by log2(size) — the table content is fully
// determined by Goldilocks::w(log2_size), so a single cached copy per
// size is reusable across every LDE / NTT / INTT bridge call. Prior to
// caching, each call re-ran an N-element gl_mul chain on the host,
// costing ~50ms at N=2^23 — material when a single inner proof makes
// 15+ Metal NTT-family calls at that size.
//
// Thread safety: std::unordered_map is guarded by roots_cache_mutex().
// Reads and writes go through the same lock — short critical section.
// The returned pointer is stable for the process lifetime since the
// cache owns its vectors by value in an unordered_map (element
// addresses remain stable across inserts as long as we never remove).
inline std::mutex& roots_cache_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<uint32_t, std::vector<uint64_t>>& roots_cache() {
    static std::unordered_map<uint32_t, std::vector<uint64_t>> c;
    return c;
}

// Returns a borrowed pointer to roots[0..size) with roots[k] = w^k for
// w = Goldilocks::w(log2(size)). First call per size pays the build;
// later calls are O(1) lookup.
inline const uint64_t* build_roots_cached(uint64_t size) {
    uint32_t log2_size = 0;
    for (uint64_t v = size; v > 1; v >>= 1) ++log2_size;

    std::lock_guard<std::mutex> lock(roots_cache_mutex());
    auto& cache = roots_cache();
    auto it = cache.find(log2_size);
    if (it != cache.end()) {
        return it->second.data();
    }
    std::vector<uint64_t> roots(size);
    const Goldilocks::Element w = Goldilocks::w(log2_size);
    Goldilocks::Element acc = Goldilocks::one();
    for (uint64_t k = 0; k < size; ++k) {
        roots[k] = Goldilocks::toU64(acc);
        acc = acc * w;
    }
    auto [ins, _] = cache.emplace(log2_size, std::move(roots));
    return ins->second.data();
}
} // namespace detail

// Thin adapter from NTT_Goldilocks::LDE(output, input, N_Extended, N, ncols)
// to pil2_metal_lde(...). Builds the roots and coset-scale tables on the fly
// using exactly the same math NTT_Goldilocks uses internally, so the Metal
// output is bit-exact with the CPU path.
//
// Roots: roots[k] = w^k where w = Goldilocks::w(log2(N_Extended)). Length
// N_Extended (a power of 2), stored as canonical u64.
//
// Coset scale: r_[i] = shift^i * inv(N). Length N. This matches
// NTT_Goldilocks::computeR, which stores r_[i] = r[i] * powTwoInv[domainPow]
// = shift^i * inv(N).
//
// Layout note: Goldilocks::Element is `struct { uint64_t fe; }`. It is
// layout-compatible with uint64_t (same size, same alignment, single
// member), so reinterpret_cast to uint64_t* is well-defined for arrays.
// The static_assert below catches any future layout change.
inline void lde_via_metal(Goldilocks::Element* output,
                          const Goldilocks::Element* input,
                          uint64_t N_Extended,
                          uint64_t N,
                          uint64_t ncols) {
    static_assert(sizeof(Goldilocks::Element) == sizeof(uint64_t),
                  "Goldilocks::Element must be layout-compatible with uint64_t");

    const uint64_t* roots = detail::build_roots_cached(N_Extended);

    std::vector<uint64_t> r_table(N);
    const Goldilocks::Element shift = Goldilocks::shift();
    const Goldilocks::Element inv_N = Goldilocks::inv(Goldilocks::fromU64(N));
    Goldilocks::Element racc = inv_N;            // r_[0] = inv(N)
    r_table[0] = Goldilocks::toU64(racc);
    for (uint64_t i = 1; i < N; ++i) {
        racc = racc * shift;                     // r_[i] = shift^i * inv(N)
        r_table[i] = Goldilocks::toU64(racc);
    }

    pil2_metal_lde(reinterpret_cast<uint64_t*>(output),
                   reinterpret_cast<const uint64_t*>(input),
                   N_Extended, N, ncols,
                   roots, N_Extended,
                   r_table.data());
}

// In-place forward NTT. Matches NTT_Goldilocks::NTT(dst, src, size, ncols)
// when dst == src. Used by Starks::computeQ after the coset scale step.
inline void ntt_via_metal(Goldilocks::Element* data,
                          uint64_t size,
                          uint64_t ncols) {
    static_assert(sizeof(Goldilocks::Element) == sizeof(uint64_t),
                  "Goldilocks::Element must be layout-compatible with uint64_t");
    const uint64_t* roots = detail::build_roots_cached(size);
    pil2_metal_ntt_forward(reinterpret_cast<uint64_t*>(data),
                           size, ncols, roots, size);
}

// In-place inverse NTT. Matches NTT_Goldilocks::INTT(dst, src, size, ncols)
// when dst == src. Used by Starks::computeQ before coset scaling.
inline void intt_via_metal(Goldilocks::Element* data,
                           uint64_t size,
                           uint64_t ncols) {
    static_assert(sizeof(Goldilocks::Element) == sizeof(uint64_t),
                  "Goldilocks::Element must be layout-compatible with uint64_t");
    const uint64_t* roots = detail::build_roots_cached(size);
    const uint64_t inv_n = Goldilocks::toU64(Goldilocks::inv(Goldilocks::fromU64(size)));
    pil2_metal_ntt_inverse(reinterpret_cast<uint64_t*>(data),
                           size, ncols, roots, size, inv_n);
}

}} // namespace pil2::metal

#endif // PIL2_HAS_METAL
#endif // PIL2_NTT_METAL_BRIDGE_HPP
