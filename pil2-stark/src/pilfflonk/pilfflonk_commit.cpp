#include "pilfflonk_commit.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpolynomial.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string(function) + ": " + message);
}

} // namespace

uint64_t packedBufferLength(uint64_t k, uint64_t n) {
    if (k == 0 || n == 0) {
        throw invalid("packedBufferLength",
                      "k = " + std::to_string(k) + " polynomials of n = " + std::to_string(n) + " coefficients");
    }
    constexpr uint64_t MAX_LENGTH = uint64_t(1) << 63;
    if (n > MAX_LENGTH / k) {
        throw invalid("packedBufferLength", "k·n = " + std::to_string(k) + "·" + std::to_string(n) + " exceeds 2^63");
    }
    uint64_t length = 1;
    while (length < k * n) {
        length <<= 1;
    }
    return length;
}

uint64_t pack(Poly *const *polys, uint64_t k, FrElement *packed, uint64_t bufferLength) {
    if (k == 0) {
        throw invalid("pack", "k = 0: no polynomials");
    }
    if (k > static_cast<uint64_t>(INT_MAX)) {
        throw invalid("pack", "k = " + std::to_string(k) + " exceeds INT_MAX, the most CPolynomial counts");
    }
    if (polys == nullptr) {
        throw invalid("pack", "polys is null");
    }
    uint64_t maxLength = 0;
    for (uint64_t j = 0; j < k; ++j) {
        const std::string name = "polys[" + std::to_string(j) + "]";
        if (polys[j] == nullptr) {
            throw invalid("pack", name + " is null");
        }
        if (polys[j]->getLength() == 0) {
            throw invalid("pack", name + " has no coefficients");
        }
        maxLength = std::max(maxLength, polys[j]->getLength());
    }
    const uint64_t needed = packedBufferLength(k, maxLength);
    if (packed == nullptr) {
        throw invalid("pack", "packed is null");
    }
    if (bufferLength < needed) {
        throw invalid("pack", "a buffer of " + std::to_string(bufferLength) + " elements is shorter than the " +
                                  std::to_string(needed) + " that k = " + std::to_string(k) +
                                  " polynomials of up to " + std::to_string(maxLength) + " coefficients need");
    }

    CPolynomial<Engine> cpolynomial(Engine::engine, static_cast<int>(k));
    for (uint64_t j = 0; j < k; ++j) {
        cpolynomial.addPolynomial(static_cast<int>(j), polys[j]);
    }
    const uint64_t maxDegree = cpolynomial.getDegree();
    if (maxDegree < 2) {
        // getPolynomial sizes f as 2^(floor(log2(maxDegree - 1)) + 1), undefined below 2: log2(0)
        // is -inf, and maxDegree - 1 wraps around at 0. f has then at most two coefficients, from a
        // p_0 of degree at most 1 (k = 1) or from two constants (k = 2).
        for (uint64_t m = 0; m <= maxDegree; ++m) {
            packed[m] = polys[m % k]->coef[m / k];
        }
        return maxDegree + 1;
    }
    // getPolynomial writes f to `packed` and returns a polynomial over it, which does not own it.
    // Its length is the smallest power of two not below maxDegree: one short when maxDegree is a
    // power of two, whose coefficient is then in `packed` but beyond that polynomial's length and
    // its getDegree(). Hence the count returned is CPolynomial's degree bound, not that degree.
    const std::unique_ptr<Poly> wrapper(cpolynomial.getPolynomial(packed));
    return maxDegree + 1;
}

G1Point commitFixed(const Srs &srs, const Lde &lde, FrElement *const *evals, uint64_t k) {
    const uint64_t N = lde.domainSize();
    if (k == 0) {
        throw invalid("commitFixed", "k = 0: f packs no columns");
    }
    if (k > srs.nG1() / N) {
        throw invalid("commitFixed", "f's k·N = " + std::to_string(k) + "·" + std::to_string(N) +
                                         " coefficients exceed the " + std::to_string(srs.nG1()) +
                                         " powers [τ^i]₁ of the SRS");
    }

    // Poly::fromEvaluations clears each column before it writes it.
    std::unique_ptr<FrElement[]> coefs(new FrElement[k * N]);
    std::vector<FrElement *> columns(k);
    for (uint64_t j = 0; j < k; ++j) {
        columns[j] = coefs.get() + j * N;
    }
    const std::vector<std::unique_ptr<Poly>> polys = lde.intt(evals, columns.data(), k);
    std::vector<Poly *> packing(k);
    for (uint64_t j = 0; j < k; ++j) {
        packing[j] = polys[j].get();
    }

    const uint64_t length = packedBufferLength(k, N);
    std::unique_ptr<FrElement[]> packed(new FrElement[length]);
    const uint64_t nCoefs = pack(packing.data(), k, packed.get(), length);
    return srs.commit(packed.get(), nCoefs);
}

} // namespace PilFflonk
