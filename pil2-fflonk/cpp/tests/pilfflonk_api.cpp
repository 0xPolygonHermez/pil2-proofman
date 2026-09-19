// The C surface Rust drives the prover through.
//
// These tests use the generator alone rather than a proving key, so the
// arithmetic is checkable by hand: with powers of tau standing in as an
// arbitrary point sequence, an MSM with one-hot scalars must return the
// corresponding point, and summing must be additive.
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include <alt_bn128.hpp>

#include "pilfflonk_api.hpp"

using Engine = AltBn128::Engine;
static Engine &E = Engine::engine;

namespace {

// A stand-in "powers of tau": g, 2g, 3g, ... Any point sequence will do; the
// identities under test do not depend on it being a real SRS.
//
// The sequence starts at 1, not 0: a leading 0*g would be the point at
// infinity, and selecting it tells you nothing about indexing.
std::vector<Engine::G1PointAffine> points(uint64_t n) {
    std::vector<Engine::G1PointAffine> out(n);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t k = i + 1;
        Engine::G1Point p;
        E.g1.mulByScalar(p, E.g1.one(), (uint8_t *)&k, sizeof(k));
        E.g1.copy(out[i], p);
    }
    return out;
}

// Scalars are passed in the key's representation, so a plain integer has to be
// converted in before being handed over.
std::vector<Engine::FrElement> montgomery(const std::vector<uint64_t> &values) {
    std::vector<Engine::FrElement> out(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        out[i] = E.fr.set(values[i]);
    }
    return out;
}

Engine::G1PointAffine msm(const std::vector<Engine::G1PointAffine> &ptau, const std::vector<uint64_t> &scalars) {
    auto coeffs = montgomery(scalars);
    Engine::G1PointAffine out;
    EXPECT_EQ(0, pilfflonk_msm((const uint8_t *)ptau.data(), (const uint8_t *)coeffs.data(), scalars.size(),
                               (uint8_t *)&out));
    return out;
}

bool same(const Engine::G1PointAffine &a, const Engine::G1PointAffine &b) {
    return std::memcmp(&a, &b, PILFFLONK_G1_AFFINE_BYTES) == 0;
}

} // namespace

// A one-hot scalar vector selects exactly one point, which pins both the
// indexing and the Montgomery conversion at the boundary.
TEST(PILFFLONK_API, oneHotSelectsItsPoint) {
    auto ptau = points(8);

    for (uint64_t j = 0; j < 8; j++) {
        std::vector<uint64_t> scalars(8, 0);
        scalars[j] = 1;
        EXPECT_TRUE(same(msm(ptau, scalars), ptau[j])) << "index " << j;
    }
}

// Scalars are read as field elements, not as raw limbs: passing the stored
// Montgomery form straight to the MSM would select a different multiple.
TEST(PILFFLONK_API, scalarsAreFieldElementsNotRawLimbs) {
    auto ptau = points(4);
    std::vector<uint64_t> scalars = {3, 0, 0, 0};

    Engine::G1Point want;
    uint64_t three = 3;
    E.g1.mulByScalar(want, E.g1.one(), (uint8_t *)&three, sizeof(three));
    Engine::G1PointAffine wantAffine;
    E.g1.copy(wantAffine, want);

    // ptau[0] is the generator, so 3 * ptau[0] is 3g.
    EXPECT_TRUE(same(msm(ptau, scalars), wantAffine));
}

TEST(PILFFLONK_API, isAdditiveAcrossTerms) {
    auto ptau = points(4);

    auto a = msm(ptau, {1, 0, 0, 0});
    auto b = msm(ptau, {0, 1, 0, 0});
    auto both = msm(ptau, {1, 1, 0, 0});

    Engine::G1Point sum;
    E.g1.add(sum, a, b);
    Engine::G1PointAffine sumAffine;
    E.g1.copy(sumAffine, sum);

    EXPECT_TRUE(same(both, sumAffine));
}

// All-zero scalars, and an empty input, both give the point at infinity --
// written as all zeroes so a caller can recognise it without curve arithmetic.
TEST(PILFFLONK_API, theEmptySumIsInfinity) {
    auto ptau = points(4);
    uint8_t zeros[PILFFLONK_G1_AFFINE_BYTES] = {};

    auto none = msm(ptau, {0, 0, 0, 0});
    EXPECT_EQ(0, std::memcmp(&none, zeros, PILFFLONK_G1_AFFINE_BYTES));

    uint8_t out[PILFFLONK_G1_AFFINE_BYTES];
    std::memset(out, 0xAA, sizeof(out));
    EXPECT_EQ(0, pilfflonk_msm(nullptr, nullptr, 0, out));
    EXPECT_EQ(0, std::memcmp(out, zeros, PILFFLONK_G1_AFFINE_BYTES));
}

// A null buffer is reported, not dereferenced, and the message survives for
// the caller to read.
TEST(PILFFLONK_API, reportsNullInputsInsteadOfCrashing) {
    uint8_t out[PILFFLONK_G1_AFFINE_BYTES];
    auto ptau = points(2);

    EXPECT_NE(0, pilfflonk_msm(nullptr, (const uint8_t *)ptau.data(), 2, out));
    ASSERT_NE(nullptr, pilfflonk_last_error());
    EXPECT_NE(nullptr, std::strstr(pilfflonk_last_error(), "null"));

    EXPECT_NE(0, pilfflonk_msm((const uint8_t *)ptau.data(), (const uint8_t *)ptau.data(), 2, nullptr));
}

// The error slot is cleared by a successful call, so a stale message cannot be
// mistaken for a fresh failure.
TEST(PILFFLONK_API, successClearsTheLastError) {
    uint8_t out[PILFFLONK_G1_AFFINE_BYTES];
    EXPECT_NE(0, pilfflonk_msm(nullptr, nullptr, 4, out));
    EXPECT_NE(nullptr, pilfflonk_last_error());

    auto ptau = points(2);
    msm(ptau, {1, 0});
    EXPECT_EQ(nullptr, pilfflonk_last_error());
}
