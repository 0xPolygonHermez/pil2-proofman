#include "pilfflonk_lde.hpp"

#include <omp.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

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

Lde::Lde(uint64_t _nBits, uint64_t _nBitsExt) {
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

    forEachColumn(nCols, [&](uint64_t c) {
        mulByPowers(evals[c], coefs[c], nCoefs, shift);
        if (nCoefs < NExtended) {
            ThreadUtils::parset(evals[c] + nCoefs, 0, (NExtended - nCoefs) * sizeof(FrElement), omp_get_max_threads());
        }
        fft->fft(evals[c], NExtended);
    });
}

void Lde::interpolateCoset(const FrElement *const *evals, FrElement *const *coefs, uint64_t nCols) const {
    checkBuffers("interpolateCoset", "evals", evals, nCols);
    checkBuffers("interpolateCoset", "coefs", coefs, nCols);

    forEachColumn(nCols, [&](uint64_t c) {
        if (coefs[c] != evals[c]) {
            ThreadUtils::parcpy(coefs[c], evals[c], NExtended * sizeof(FrElement), omp_get_max_threads());
        }
        fft->ifft(coefs[c], NExtended);
        mulByPowers(coefs[c], coefs[c], NExtended, shiftInv);
    });
}

} // namespace PilFflonk
