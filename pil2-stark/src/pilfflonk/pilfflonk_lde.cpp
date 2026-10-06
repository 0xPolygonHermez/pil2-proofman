#include "pilfflonk_lde.hpp"

#include <omp.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"

namespace PilFflonk {

namespace {

using Engine = Lde::Engine;

constexpr InvalidArgument invalid("Lde::");

// Throws unless `buffers` holds nCols >= 1 pointers, none of them null.
template <typename Pointer>
void checkBuffers(const char *function, const char *name, const Pointer *buffers, uint64_t nCols) {
    if (nCols == 0) {
        throw invalid(function, "no columns");
    }
    if (buffers == nullptr) {
        throw invalid(function, std::string(name) + " is null");
    }
    for (uint64_t c = 0; c < nCols; ++c) {
        if (buffers[c] == nullptr) {
            throw invalid(function, std::string(name) + "[" + std::to_string(c) + "] is null");
        }
    }
}

// Runs body(c, parallel) for every column c < nCols, and rethrows the first exception it throws:
// one column after another on the whole team (parallel) when there are fewer columns than OpenMP
// threads, and one column per thread otherwise, taken as threads free up (the cores need not be
// equally fast). omp_set_num_threads(1) inside the region sets the ICV of that thread's implicit
// task only, so the OpenMP regions of the helpers below get one thread each. Every column gets the
// same field operations either way, so the results are the same bit for bit.
template <typename Body>
void forEachColumn(uint64_t nCols, const Body &body) {
    if (nCols < static_cast<uint64_t>(omp_get_max_threads())) {
        for (uint64_t c = 0; c < nCols; ++c) {
            body(c, true);
        }
        return;
    }

    // An exception must not leave an OpenMP region.
    std::exception_ptr failure;
#pragma omp parallel
    {
        omp_set_num_threads(1);
#pragma omp for schedule(dynamic, 1)
        for (uint64_t c = 0; c < nCols; ++c) {
            try {
                body(c, false);
            } catch (...) {
#pragma omp critical(pilfflonk_lde_failure)
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

// dst[j] = src[j] · first · base^j for j < n; dst may be src. Each thread starts its chunk from
// first·base^begin, so the powers are the same whatever the team.
void mulByPowers(FrElement *dst, const FrElement *src, uint64_t n, const FrElement &base, const FrElement &first) {
    Engine::Fr &fr = Engine::engine.fr;
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (n + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(n, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(n, begin + chunk);
        if (begin < end) {
            FrElement factor;
            fr.mul(factor, first, power(base, begin));
            for (uint64_t j = begin; j < end; ++j) {
                fr.mul(dst[j], src[j], factor);
                fr.mul(factor, factor, base);
            }
        }
    }
}

// dst[d] = src[BR(d)]·factor for the 2^k elements of src (dst != src).
void reverseCopy(FrElement *dst, const FrElement *src, uint64_t k, const FrElement *factor, bool parallel) {
    const uint64_t n = uint64_t(1) << k;
#pragma omp parallel for schedule(static) if (parallel)
    for (uint64_t d = 0; d < n; ++d) {
        const FrElement &v = src[Ntt::reverse(d, k)];
        if (factor != nullptr) {
            Fr_rawMMul(dst[d].v, v.v, factor->v);
        } else {
            dst[d] = v;
        }
    }
}

// powers[j] = base^j for j < n.
std::unique_ptr<FrElement[]> powersOf(const FrElement &base, uint64_t n) {
    std::unique_ptr<FrElement[]> powers(new FrElement[n]);
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (n + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(n, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(n, begin + chunk);
        if (begin < end) {
            FrElement factor = power(base, begin);
            for (uint64_t j = begin; j < end; ++j) {
                powers[j] = factor;
                Fr_rawMMul(factor.v, factor.v, base.v);
            }
        }
    }
    return powers;
}

// k for n = 2^k.
uint64_t bitsOf(uint64_t n) { return static_cast<uint64_t>(__builtin_ctzll(n)); }

// dst[r] = Σ_t src[r + t·s]·base^(r + t·s) for r < s, over the terms with r + t·s < n: the n
// coefficients in src scaled by the powers of base and folded modulo s, which may be below or above
// n (dst[r] = 0 for n <= r < s). dst may be src: dst[r] is written after src[r] is read, and every
// other term it reads is at s or beyond, where nothing is written. Each thread starts its chunk from
// base^begin, as mulByPowers.
void foldByPowers(FrElement *dst, const FrElement *src, uint64_t n, uint64_t s, const FrElement &base) {
    Engine::Fr &fr = Engine::engine.fr;
    const FrElement baseS = power(base, s);
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (s + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(s, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(s, begin + chunk);
        if (begin < end) {
            FrElement factor = power(base, begin);
            for (uint64_t r = begin; r < end; ++r) {
                if (r >= n) {
                    dst[r] = fr.zero();
                    continue;
                }
                // src[r]·base^r, and then each term of r + t·s, base^(r + t·s) a factor base^s apart.
                FrElement acc, f = factor;
                fr.mul(acc, src[r], f);
                for (uint64_t j = r + s; j < n; j += s) {
                    FrElement term;
                    fr.mul(f, f, baseS);
                    fr.mul(term, src[j], f);
                    fr.add(acc, acc, term);
                }
                dst[r] = acc;
                fr.mul(factor, factor, base);
            }
        }
    }
}

} // namespace

Lde::Lde(uint64_t _nBits, uint64_t _nBitsExt) {
    if (_nBitsExt > MAX_NBITS_EXT) {
        throw invalid("Lde", "nBitsExt = " + std::to_string(_nBitsExt) + " exceeds " + std::to_string(MAX_NBITS_EXT) +
                                 ", the 2-adicity of the BN128 scalar field");
    }
    if (_nBits > _nBitsExt) {
        throw invalid("Lde",
                      "nBits = " + std::to_string(_nBits) + " exceeds nBitsExt = " + std::to_string(_nBitsExt));
    }
#ifdef __USE_ASSEMBLY__
    N = uint64_t(1) << _nBits;
    NExtended = uint64_t(1) << _nBitsExt;
    Engine::Fr &fr = Engine::engine.fr;
    fr.fromUI(shift, COSET_SHIFT);
    fr.inv(shiftInv, shift);
    extendedRoot = rootOfUnity(_nBitsExt);
    fr.fromUI(nInv, N);
    fr.inv(nInv, nInv);
    fr.fromUI(nExtendedInv, NExtended);
    fr.inv(nExtendedInv, nExtendedInv);
#else
    throw std::runtime_error("the LDE needs ffiasm's assembly backend, not built on this platform");
#endif
}

const Ntt &Lde::transforms() const {
    std::call_once(nttBuilt, [this] { ntt_ = std::make_unique<Ntt>(bitsOf(NExtended)); });
    return *ntt_;
}

std::vector<std::unique_ptr<Lde::Poly>> Lde::intt(FrElement *const *evals, FrElement *const *coefs, uint64_t nCols,
                                                  uint64_t blindLength) const {
    checkBuffers("intt", "evals", evals, nCols);
    checkBuffers("intt", "coefs", coefs, nCols);
    for (uint64_t c = 0; c < nCols; ++c) {
        if (evals[c] == coefs[c]) {
            const std::string column = "[" + std::to_string(c) + "]";
            throw invalid("intt", "evals" + column + " is coefs" + column +
                                      ": the coefficients are written before the evaluations are read");
        }
    }
    if (blindLength > NExtended - N) {
        throw invalid("intt", "N + blindLength = " + std::to_string(N) + " + " + std::to_string(blindLength) +
                                  " coefficients exceed the " + std::to_string(NExtended) + " of the extended domain");
    }

    const Ntt &transform = transforms();
    const uint64_t k = bitsOf(N);
    std::vector<std::unique_ptr<Poly>> polys(nCols);
    forEachColumn(nCols, [&](uint64_t c, bool parallel) {
        // The 1/N of the inverse in the bit-reversed copy; the blinding's room cleared.
        reverseCopy(coefs[c], evals[c], k, &nInv, parallel);
        std::fill(coefs[c] + N, coefs[c] + N + blindLength, Engine::engine.fr.zero());
        transform.fromBitReversed(coefs[c], k, true, parallel);
        polys[c].reset(Poly::fromReservedBuffer(Engine::engine, coefs[c], N + blindLength));
    });
    return polys;
}

void Lde::ntt(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols) const {
    checkBuffers("ntt", "coefs", coefs, nCols);
    checkBuffers("ntt", "evals", evals, nCols);

    const Ntt &transform = transforms();
    const uint64_t k = bitsOf(N);
    forEachColumn(nCols, [&](uint64_t c, bool parallel) {
        if (evals[c] != coefs[c]) {
            reverseCopy(evals[c], coefs[c], k, nullptr, parallel);
        } else {
            Ntt::bitReverse(evals[c], k, parallel);
        }
        transform.fromBitReversed(evals[c], k, false, parallel);
    });
}

void Lde::extendCoset(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs) const {
    checkBuffers("extendCoset", "coefs", coefs, nCols);
    checkBuffers("extendCoset", "evals", evals, nCols);
    if (nCoefs == 0) {
        throw invalid("extendCoset", "no coefficients");
    }
    if (nCoefs > NExtended) {
        throw invalid("extendCoset", std::to_string(nCoefs) + " coefficients exceed the " + std::to_string(NExtended) +
                                         " of the extended domain");
    }

    const std::vector<uint64_t> counts(nCols, nCoefs);
    extendPart(coefs, evals, nCols, counts.data(), bitsOf(NExtended), 0);
}

void Lde::extendCosetPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs,
                          uint64_t partBits, uint64_t part) const {
    checkBuffers("extendCosetPart", "coefs", coefs, nCols);
    const std::vector<uint64_t> counts(nCols, nCoefs);
    extendCosetPart(coefs, evals, nCols, counts.data(), partBits, part);
}

void Lde::extendCosetPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols,
                          const uint64_t *nCoefs, uint64_t partBits, uint64_t part) const {
    checkBuffers("extendCosetPart", "coefs", coefs, nCols);
    checkBuffers("extendCosetPart", "evals", evals, nCols);
    for (uint64_t c = 0; c < nCols; ++c) {
        if (nCoefs[c] == 0) {
            throw invalid("extendCosetPart", "no coefficients");
        }
        if (nCoefs[c] > NExtended) {
            throw invalid("extendCosetPart", std::to_string(nCoefs[c]) + " coefficients exceed the " +
                                                 std::to_string(NExtended) + " of the extended domain");
        }
    }
    if (partBits < bitsOf(N) || partBits > bitsOf(NExtended)) {
        throw invalid("extendCosetPart", "a part of 2^" + std::to_string(partBits) +
                                             " points, and the parts have from " + std::to_string(N) + " to " +
                                             std::to_string(NExtended));
    }
    if (part >= NExtended >> partBits) {
        throw invalid("extendCosetPart", "part " + std::to_string(part) + " of the " +
                                             std::to_string(NExtended >> partBits) + " of 2^" +
                                             std::to_string(partBits) + " points");
    }
    extendPart(coefs, evals, nCols, nCoefs, partBits, part);
}

FrElement Lde::partShift(uint64_t part) const {
    FrElement c = shift;
    if (part > 0) {
        Engine::engine.fr.mul(c, shift, power(extendedRoot, part));
    }
    return c;
}

void Lde::extendPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols,
                     const uint64_t *counts, uint64_t partBits, uint64_t part) const {
    const uint64_t S = uint64_t(1) << partBits;
    // Part 0's shift is g, extendCoset's scaling.
    const FrElement c = partShift(part);
    const Ntt &transform = transforms();
    // c^j for j < S, for every column; the terms folded from j >= S by Horner in c^S.
    const std::unique_ptr<FrElement[]> powers = powersOf(c, S);
    const FrElement cS = power(c, S);
    Engine::Fr &fr = Engine::engine.fr;
    forEachColumn(nCols, [&](uint64_t col, bool parallel) {
        FrElement *dst = evals[col];
        const FrElement *src = coefs[col];
        const uint64_t nCoefs = counts[col];
        if (dst == src) {
            foldByPowers(dst, src, nCoefs, S, c);
            Ntt::bitReverse(dst, partBits, parallel);
        } else {
            // dst[d] = Σ_t src[r + t·S]·c^(r + t·S), r = BR(d): the fold, bit-reversed.
#pragma omp parallel for schedule(static) if (parallel)
            for (uint64_t d = 0; d < S; ++d) {
                const uint64_t r = Ntt::reverse(d, partBits);
                if (r >= nCoefs) {
                    dst[d] = fr.zero();
                    continue;
                }
                FrElement acc = src[r];
                if (r + S < nCoefs) {
                    uint64_t top = r + ((nCoefs - 1 - r) / S) * S;
                    acc = src[top];
                    for (; top > r; top -= S) {
                        fr.mul(acc, acc, cS);
                        fr.add(acc, acc, src[top - S]);
                    }
                }
                Fr_rawMMul(dst[d].v, acc.v, powers[r].v);
            }
        }
        transform.fromBitReversed(dst, partBits, false, parallel);
    });
}

void Lde::interpolateCoset(const FrElement *const *evals, FrElement *const *coefs, uint64_t nCols) const {
    checkBuffers("interpolateCoset", "evals", evals, nCols);
    checkBuffers("interpolateCoset", "coefs", coefs, nCols);

    const Ntt &transform = transforms();
    const uint64_t k = bitsOf(NExtended);
    forEachColumn(nCols, [&](uint64_t c, bool parallel) {
        if (coefs[c] != evals[c]) {
            reverseCopy(coefs[c], evals[c], k, nullptr, parallel);
        } else {
            Ntt::bitReverse(coefs[c], k, parallel);
        }
        transform.fromBitReversed(coefs[c], k, true, parallel);
        mulByPowers(coefs[c], coefs[c], NExtended, shiftInv, nExtendedInv);
    });
}

} // namespace PilFflonk
