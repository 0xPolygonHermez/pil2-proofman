#include "pilfflonk_lde.hpp"

#include <omp.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

#include "pilfflonk_gpu.hpp"
#include "thread_utils.hpp"

namespace PilFflonk {

namespace {

using Engine = Lde::Engine;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string("Lde::") + function + ": " + message);
}

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

// Runs body(c) for every column c < nCols, and rethrows the first exception it throws.
//
// ffiasm's FFT parallelises every call on its own, with bare `omp parallel for` loops: its
// nThreads argument is stored but never used. So each FFT runs on whatever team the calling
// thread's ICVs give, and there are two regimes:
// - Fewer columns than threads: one column after another, each FFT on the whole team. Splitting
//   the team across the columns would leave most threads idle with a handful of large columns.
// - Otherwise, one column per thread. omp_set_num_threads(1) inside the region sets the ICV of
//   that thread's implicit task only, so its FFTs' own regions get one thread each: nothing is
//   oversubscribed even with nested parallelism on (OMP_MAX_ACTIVE_LEVELS > 1), and the caller's
//   setting is untouched once the region ends.
// Every column gets the same field operations either way, so the results are the same bit for bit.
template <typename Body>
void forEachColumn(uint64_t nCols, const Body &body) {
    if (nCols < static_cast<uint64_t>(omp_get_max_threads())) {
        for (uint64_t c = 0; c < nCols; ++c) {
            body(c);
        }
        return;
    }

    // An exception must not leave an OpenMP region.
    std::exception_ptr failure;
#pragma omp parallel
    {
        omp_set_num_threads(1);
#pragma omp for schedule(static)
        for (uint64_t c = 0; c < nCols; ++c) {
            try {
                body(c);
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

// dst[j] = src[j] · base^j for j < n; dst may be src. Each thread starts its chunk from
// base^begin, so the powers are the same whatever the team.
void mulByPowers(FrElement *dst, const FrElement *src, uint64_t n, const FrElement &base) {
    Engine::Fr &fr = Engine::engine.fr;
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (n + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(n, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(n, begin + chunk);
        if (begin < end) {
            FrElement factor = power(base, begin);
            for (uint64_t j = begin; j < end; ++j) {
                fr.mul(dst[j], src[j], factor);
                fr.mul(factor, factor, base);
            }
        }
    }
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

FrElement power(const FrElement &base, uint64_t exponent) {
    uint8_t littleEndian[sizeof(exponent)];
    for (size_t i = 0; i < sizeof(exponent); ++i) {
        littleEndian[i] = static_cast<uint8_t>(exponent >> (8 * i));
    }
    FrElement result;
    Engine::engine.fr.exp(result, base, littleEndian, sizeof(littleEndian));
    return result;
}

bool batchInverse(FrElement *out, const FrElement *values, uint64_t n) {
    Engine::Fr &fr = Engine::engine.fr;
    bool zero = false;
#pragma omp parallel reduction(|| : zero)
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (n + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(n, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(n, begin + chunk);
        if (begin < end) {
            // out[i] = values[begin] · … · values[i − 1]
            FrElement acc = fr.one();
            for (uint64_t i = begin; i < end; ++i) {
                out[i] = acc;
                fr.mul(acc, acc, values[i]);
            }
            if (fr.isZero(acc)) {
                zero = true;
            } else {
                FrElement inv;
                fr.inv(inv, acc);
                for (uint64_t i = end; i-- > begin;) {
                    FrElement t;
                    fr.mul(t, inv, out[i]);
                    fr.mul(inv, inv, values[i]);
                    out[i] = t;
                }
            }
        }
    }
    return !zero;
}

Lde::Lde(uint64_t _nBits, uint64_t _nBitsExt, const Gpu *gpu) : device(gpu) {
    if (_nBitsExt > MAX_NBITS_EXT) {
        throw invalid("Lde", "nBitsExt = " + std::to_string(_nBitsExt) + " exceeds " + std::to_string(MAX_NBITS_EXT) +
                                 ", the 2-adicity of the BN254 scalar field");
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
    fft = std::make_unique<FFT<Engine::Fr>>(NExtended);
#else
    throw std::runtime_error("the LDE needs ffiasm's assembly backend, not built on this platform");
#endif
}

std::vector<std::unique_ptr<Lde::Poly>> Lde::intt(FrElement *const *evals, FrElement *const *coefs, uint64_t nCols,
                                                  uint64_t blindLength) const {
    checkBuffers("intt", "evals", evals, nCols);
    checkBuffers("intt", "coefs", coefs, nCols);
    for (uint64_t c = 0; c < nCols; ++c) {
        if (evals[c] == coefs[c]) {
            const std::string column = "[" + std::to_string(c) + "]";
            throw invalid("intt", "evals" + column + " is coefs" + column +
                                      ": fromEvaluations clears coefs before it reads evals");
        }
    }
    if (blindLength > NExtended - N) {
        throw invalid("intt", "N + blindLength = " + std::to_string(N) + " + " + std::to_string(blindLength) +
                                  " coefficients exceed the " + std::to_string(NExtended) + " of the extended domain");
    }

    std::vector<std::unique_ptr<Poly>> polys(nCols);
#ifdef __USE_CUDA__
    if (device != nullptr) {
        // Poly::fromEvaluations with its inverse FFT on the GPU: the polynomial over coefs[c], cleared
        // to its N + blindLength coefficients, the first N of them the INTT of evals[c], its degree fixed.
        for (uint64_t c = 0; c < nCols; ++c) {
            polys[c].reset(new Poly(Engine::engine, coefs[c], N, blindLength));
            device->intt(evals[c], coefs[c], bitsOf(N));
            polys[c]->fixDegree();
        }
        return polys;
    }
#endif
    forEachColumn(nCols, [&](uint64_t c) {
        polys[c].reset(Poly::fromEvaluations(Engine::engine, fft.get(), evals[c], coefs[c], N, blindLength));
    });
    return polys;
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

    extendPart(coefs, evals, nCols, nCoefs, bitsOf(NExtended), 0);
}

void Lde::extendCosetPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs,
                          uint64_t partBits, uint64_t part) const {
    checkBuffers("extendCosetPart", "coefs", coefs, nCols);
    checkBuffers("extendCosetPart", "evals", evals, nCols);
    if (nCoefs == 0) {
        throw invalid("extendCosetPart", "no coefficients");
    }
    if (nCoefs > NExtended) {
        throw invalid("extendCosetPart", std::to_string(nCoefs) + " coefficients exceed the " +
                                             std::to_string(NExtended) + " of the extended domain");
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
        Engine::engine.fr.mul(c, shift, power(fft->root(static_cast<uint32_t>(bitsOf(NExtended)), 1), part));
    }
    return c;
}

void Lde::extendPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs,
                     uint64_t partBits, uint64_t part) const {
    const uint64_t S = uint64_t(1) << partBits;
    // Part 0's shift is g, extendCoset's scaling.
    const FrElement c = partShift(part);
#ifdef __USE_CUDA__
    if (device != nullptr) {
        for (uint64_t col = 0; col < nCols; ++col) {
            foldByPowers(evals[col], coefs[col], nCoefs, S, c);
            device->ntt(evals[col], evals[col], partBits);
        }
        return;
    }
#endif
    forEachColumn(nCols, [&](uint64_t col) {
        foldByPowers(evals[col], coefs[col], nCoefs, S, c);
        fft->fft(evals[col], S);
    });
}

void Lde::interpolateCoset(const FrElement *const *evals, FrElement *const *coefs, uint64_t nCols) const {
    checkBuffers("interpolateCoset", "evals", evals, nCols);
    checkBuffers("interpolateCoset", "coefs", coefs, nCols);

#ifdef __USE_CUDA__
    if (device != nullptr) {
        for (uint64_t c = 0; c < nCols; ++c) {
            device->intt(evals[c], coefs[c], bitsOf(NExtended));
            mulByPowers(coefs[c], coefs[c], NExtended, shiftInv);
        }
        return;
    }
#endif
    forEachColumn(nCols, [&](uint64_t c) {
        if (coefs[c] != evals[c]) {
            ThreadUtils::parcpy(coefs[c], evals[c], NExtended * sizeof(FrElement), omp_get_max_threads());
        }
        fft->ifft(coefs[c], NExtended);
        mulByPowers(coefs[c], coefs[c], NExtended, shiftInv);
    });
}

} // namespace PilFflonk
