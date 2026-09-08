// CPolynomial<Engine> tests.
//
// CPolynomial interleaves n polynomials into one "combined" polynomial, which
// is how fflonk (and pil-fflonk's SHPLONK) pack several committed polynomials
// behind a single commitment. It is live code -- fflonk_prover and fflonk_setup
// both use it -- but had no coverage.
#include <gtest/gtest.h>
#include <iostream>

#include <alt_bn128.hpp>
#include "polynomial/cpolynomial.hpp"

static AltBn128::Engine &E = AltBn128::Engine::engine;

// Polynomial of the given degree with coefficients 1..degree+1.
static Polynomial<AltBn128::Engine>* polOfDegree(u_int64_t degree, u_int64_t base) {
    auto pol = new Polynomial<AltBn128::Engine>(E, degree + 1);
    for (u_int64_t i = 0; i <= degree; i++) {
        pol->setCoef(i, E.fr.set(base + i + 1));
    }
    return pol;
}

// getDegree() reports the degree of the *combined* polynomial: a component at
// slot i with degree d occupies index d * n + i.
TEST(CPOLYNOMIAL, getDegree) {
    CPolynomial<AltBn128::Engine> cpol(E, 2);

    auto p0 = polOfDegree(3, 0);   // occupies indices 0,2,4,6
    auto p1 = polOfDegree(2, 10);  // occupies indices 1,3,5

    cpol.addPolynomial(0, p0);
    cpol.addPolynomial(1, p1);

    // max(3 * 2 + 0, 2 * 2 + 1) = 6
    ASSERT_EQ(cpol.getDegree(), 6);
}

TEST(CPOLYNOMIAL, addPolynomialRejectsOutOfRange) {
    CPolynomial<AltBn128::Engine> cpol(E, 2);
    auto p = polOfDegree(1, 0);

    EXPECT_ANY_THROW(cpol.addPolynomial(2, p));
    EXPECT_NO_THROW(cpol.addPolynomial(1, p));
}

// The combined polynomial must interleave its components: component j's
// coefficient i lands at index i * n + j.
TEST(CPOLYNOMIAL, interleavesComponents) {
    const int n = 2;
    CPolynomial<AltBn128::Engine> cpol(E, n);

    auto p0 = polOfDegree(3, 0);
    auto p1 = polOfDegree(3, 10);

    cpol.addPolynomial(0, p0);
    cpol.addPolynomial(1, p1);

    AltBn128::FrElement buffer[64];
    auto combined = cpol.getPolynomial(buffer);

    for (u_int64_t i = 0; i <= 3; i++) {
        ASSERT_TRUE(E.fr.eq(combined->getCoef(i * n + 0), p0->getCoef(i)));
        ASSERT_TRUE(E.fr.eq(combined->getCoef(i * n + 1), p1->getCoef(i)));
    }
}

// Regression: the combined polynomial must be long enough to hold its own
// highest coefficient. With n = 2 and a degree-4 component at slot 0 the
// combined degree is exactly 8, so the buffer must span 9 coefficients.
// Sizing it as a power of two from log2(maxDegree - 1) yields 8 here, one
// short, which leaves the top coefficient outside the recorded length and
// makes the subsequent fixDegree() under-report the degree.
TEST(CPOLYNOMIAL, sizesCombinedBufferForPowerOfTwoDegree) {
    const int n = 2;
    CPolynomial<AltBn128::Engine> cpol(E, n);

    auto p0 = polOfDegree(4, 0);   // 4 * 2 + 0 = 8  <- combined degree
    auto p1 = polOfDegree(3, 10);  // 3 * 2 + 1 = 7

    cpol.addPolynomial(0, p0);
    cpol.addPolynomial(1, p1);

    ASSERT_EQ(cpol.getDegree(), 8);

    AltBn128::FrElement buffer[64];
    auto combined = cpol.getPolynomial(buffer);

    ASSERT_GE(combined->getLength(), 9u);
    ASSERT_EQ(combined->getDegree(), 8u);
    ASSERT_TRUE(E.fr.eq(combined->getCoef(8), p0->getCoef(4)));
}
