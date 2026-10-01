#include "pilfflonk_commit.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

    // CPolynomial's degree bound, max_j(k·deg p_j + j), which is above deg f if the top
    // coefficients are zero.
    std::vector<uint64_t> degrees(k);
    uint64_t maxDegree = 0;
    for (uint64_t j = 0; j < k; ++j) {
        degrees[j] = polys[j]->getDegree();
        maxDegree = std::max(maxDegree, degrees[j] * k + j);
    }
    // f row by row, a row of k coefficients being a coefficient of every p_j: what
    // CPolynomial::getPolynomial writes, in one pass of f's coefficients only (it clears a
    // power-of-two prefix of the buffer first, and scans it for the degree after).
    const uint64_t n = maxDegree + 1;
    const FrElement zero = Engine::engine.fr.zero();
#pragma omp parallel for
    for (uint64_t row = 0; row < (n + k - 1) / k; ++row) {
        const uint64_t end = std::min(k, n - row * k);
        for (uint64_t j = 0; j < end; ++j) {
            packed[row * k + j] = row <= degrees[j] ? polys[j]->coef[row] : zero;
        }
    }
    return n;
}

G1Point commitPacked(const Srs &srs, Poly *const *polys, uint64_t k) {
    if (polys == nullptr) {
        throw invalid("commitPacked", "polys is null");
    }
    uint64_t maxLength = 0;
    for (uint64_t j = 0; j < k; ++j) {
        if (polys[j] == nullptr) {
            throw invalid("commitPacked", "polys[" + std::to_string(j) + "] is null");
        }
        maxLength = std::max(maxLength, polys[j]->getLength());
    }
    // Throws if k or every length is 0, before anything is allocated.
    const uint64_t length = packedBufferLength(k, maxLength);
    std::unique_ptr<FrElement[]> packed(new FrElement[length]);
    const uint64_t nCoefs = pack(polys, k, packed.get(), length);
    return srs.commit(packed.get(), nCoefs);
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
    return commitPacked(srs, packing.data(), k);
}

} // namespace PilFflonk
