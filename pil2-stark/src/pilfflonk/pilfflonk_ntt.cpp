#include "pilfflonk_ntt.hpp"

#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// (u, v) <- (u + t, u − t), t = w·v; inverse: (u − t, u + t), w = −ω^−j's twiddle (see twiddleOf).
template <bool Inverse> inline void butterfly(FrElement &u, FrElement &v, const FrElement &w) {
    FrElement t;
    Fr_rawMMul(t.v, w.v, v.v);
    if (Inverse) {
        frAdd(v, u, t);
        frSub(u, u, t);
    } else {
        frSub(v, u, t);
        frAdd(u, u, t);
    }
}

// (u, v) <- (u + v, u − v): the twiddle ω^0.
inline void butterflyOne(FrElement &u, FrElement &v) {
    FrElement t = v;
    frSub(v, u, t);
    frAdd(u, u, t);
}

// The twiddle of j > 0 at a stage of half size m: ω_{2m}^j, or for the inverse ω_{2m}^(m−j) =
// −ω_{2m}^−j, whose sign butterfly<true> takes.
template <bool Inverse> inline const FrElement &twiddleOf(const FrElement *tw, uint64_t m, uint64_t j) {
    return Inverse ? tw[2 * m - j] : tw[m + j];
}

// Stages within blocks of 2^BLOCK_BITS elements (64 KB) in cache, and the rest on groups of GROUP
// contiguous columns of the blocks.
constexpr uint64_t BLOCK_BITS = 11;
constexpr uint64_t GROUP = 8;

template <bool Inverse> void blockStages(FrElement *a, uint64_t L, const FrElement *tw) {
    const uint64_t B = uint64_t(1) << L;
    for (uint64_t s = 1; s <= L; ++s) {
        const uint64_t m = uint64_t(1) << (s - 1);
        for (uint64_t base = 0; base < B; base += 2 * m) {
            FrElement *x = a + base, *y = a + base + m;
            butterflyOne(x[0], y[0]);
            for (uint64_t j = 1; j < m; ++j) {
                butterfly<Inverse>(x[j], y[j], twiddleOf<Inverse>(tw, m, j));
            }
        }
    }
}

// Stages L + 1 … k of the GROUP columns j0 … of the 2^(k−L) blocks of 2^L, gathered in tmp.
template <bool Inverse>
void groupStages(FrElement *a, uint64_t k, uint64_t L, uint64_t j0, const FrElement *tw, FrElement *tmp) {
    const uint64_t B = uint64_t(1) << L, nBlocks = uint64_t(1) << (k - L);
    const uint64_t J = std::min(GROUP, B);
    for (uint64_t t = 0; t < nBlocks; ++t) {
        std::copy(a + t * B + j0, a + t * B + j0 + J, tmp + t * J);
    }
    for (uint64_t s = L + 1; s <= k; ++s) {
        const uint64_t m = uint64_t(1) << (s - 1), mb = m >> L;
        for (uint64_t tb = 0; tb < nBlocks; tb += 2 * mb) {
            for (uint64_t u = 0; u < mb; ++u) {
                FrElement *x = tmp + (tb + u) * J, *y = tmp + (tb + u + mb) * J;
                const uint64_t e = j0 + B * u;
                uint64_t d = 0;
                if (e == 0) {
                    butterflyOne(x[0], y[0]);
                    d = 1;
                }
                for (; d < J; ++d) {
                    butterfly<Inverse>(x[d], y[d], twiddleOf<Inverse>(tw, m, e + d));
                }
            }
        }
    }
    for (uint64_t t = 0; t < nBlocks; ++t) {
        std::copy(tmp + t * J, tmp + t * J + J, a + t * B + j0);
    }
}

template <bool Inverse> void transform(FrElement *a, uint64_t k, const FrElement *tw, bool parallel) {
    const uint64_t L = std::min(k, BLOCK_BITS);
    const uint64_t B = uint64_t(1) << L, nBlocks = uint64_t(1) << (k - L);
#pragma omp parallel for schedule(static) if (parallel && nBlocks > 1)
    for (uint64_t b = 0; b < nBlocks; ++b) {
        blockStages<Inverse>(a + b * B, L, tw);
    }
    if (k == L) {
        return;
    }
    const uint64_t J = std::min(GROUP, B);
#pragma omp parallel if (parallel)
    {
        std::vector<FrElement> tmp(nBlocks * J);
#pragma omp for schedule(static)
        for (uint64_t j0 = 0; j0 < B; j0 += J) {
            groupStages<Inverse>(a, k, L, j0, tw, tmp.data());
        }
    }
}

} // namespace

Ntt::Ntt(uint64_t maxBits) : bits(maxBits) {
    if (maxBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("Ntt: 2^" + std::to_string(maxBits) + " points exceed the 2-adicity of Fr");
    }
    const uint64_t size = uint64_t(1) << maxBits;
    twiddles.reset(new FrElement[std::max<uint64_t>(size, 2)]);
    if (maxBits == 0) {
        return;
    }
    // The top stage, m = size/2, from its root; each stage below takes every other twiddle above it.
    const uint64_t M = size / 2;
    FrElement *top = twiddles.get() + M;
    const FrElement root = rootOfUnity(maxBits);
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (M + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(M, omp_get_thread_num() * chunk), end = std::min(M, begin + chunk);
        if (begin < end) {
            FrElement w = power(root, begin);
            for (uint64_t j = begin; j < end; ++j) {
                top[j] = w;
                Fr_rawMMul(w.v, w.v, root.v);
            }
        }
    }
    for (uint64_t m = M / 2; m >= 1; m /= 2) {
#pragma omp parallel for schedule(static) if (m > 4096)
        for (uint64_t j = 0; j < m; ++j) {
            twiddles[m + j] = twiddles[2 * m + 2 * j];
        }
    }
}

void Ntt::fromBitReversed(FrElement *a, uint64_t k, bool inverse, bool parallel) const {
    if (k > bits) {
        throw std::invalid_argument("Ntt: 2^" + std::to_string(k) + " points, and it has twiddles for 2^" +
                                    std::to_string(bits));
    }
    if (inverse) {
        transform<true>(a, k, twiddles.get(), parallel);
    } else {
        transform<false>(a, k, twiddles.get(), parallel);
    }
}

void Ntt::bitReverse(FrElement *a, uint64_t k, bool parallel) {
    const uint64_t n = uint64_t(1) << k;
#pragma omp parallel for schedule(static) if (parallel)
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t r = reverse(i, k);
        if (i < r) {
            std::swap(a[i], a[r]);
        }
    }
}

} // namespace PilFflonk
