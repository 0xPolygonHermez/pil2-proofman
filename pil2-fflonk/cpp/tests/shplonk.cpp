// SHPLONK opening tests.
//
// The point of pil-fflonk's SHPLONK over rapidsnark's fflonk_prover is that the
// opening structure is data driven: fflonk hardcodes three opening sets
// (computeLiS0/S1/S2), while this reads the set of combined polynomials f_i,
// each with its own polynomial count and opening points, from the setup. These
// tests cover that generalisation directly.
#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "shplonk.hpp"

static AltBn128::Engine &E = AltBn128::Engine::engine;

// calculateOpeningPoints / calculateRoots are protected. Deriving to expose
// them is the standard way to test protected members -- no macro tricks, and
// shplonk.hpp stays untouched.
class ProverAccess : public ShPlonk::ShPlonkProver {
public:
    ProverAccess(AltBn128::Engine &e, PilFflonk::ShPlonkSetup *s) : ShPlonk::ShPlonkProver(e, s) {}

    using ShPlonk::ShPlonkProver::calculateOpeningPoints;
    using ShPlonk::ShPlonkProver::calculateRoots;
    using ShPlonk::ShPlonkProver::openingPoints;
    using ShPlonk::ShPlonkProver::rootsMap;
    using ShPlonk::ShPlonkProver::challengeXiSeed;
};

// Build an f_i holding nPols polynomials, opened at the given points.
static PilFflonk::ShPlonkPol* makeF(uint32_t index, uint32_t nPols,
                                    const std::vector<uint32_t> &points) {
    auto f = new PilFflonk::ShPlonkPol();
    f->index = index;
    f->degree = 0;
    f->nPols = nPols;
    f->pols = new std::string[nPols];
    for (uint32_t i = 0; i < nPols; i++) {
        f->pols[i] = "p" + std::to_string(index) + "_" + std::to_string(i);
    }
    f->nOpeningPoints = points.size();
    f->openingPoints = new uint32_t[points.size()];
    for (size_t i = 0; i < points.size(); i++) {
        f->openingPoints[i] = points[i];
    }
    f->nStages = 0;
    f->stages = nullptr;
    return f;
}

// A primitive n-th root of unity in Fr, for n a power of two.
static AltBn128::FrElement rootOfUnity(uint32_t n) {
    // Fr has a 2^28 subgroup; repeatedly square the generator of it down to n.
    AltBn128::FrElement w;
    E.fr.fromString(w, "19103219067921713944291392827692070036145651957329286315305642004821462161904");
    uint32_t order = 1 << 28;
    while (order > n) {
        w = E.fr.mul(w, w);
        order >>= 1;
    }
    return w;
}

// openingPoints is the deduplicated union of every f_i's opening points, in
// order of first appearance.
TEST(SHPLONK, openingPointsAreDedupedUnion) {
    PilFflonk::ShPlonkSetup setup;
    setup.power = 4;
    setup.powerW = 2;

    setup.f[0] = makeF(0, 2, {0, 1});
    setup.f[1] = makeF(1, 2, {1, 2});   // 1 already seen, 2 is new
    setup.f[2] = makeF(2, 2, {0});      // both already seen

    ProverAccess prover(E, &setup);
    prover.calculateOpeningPoints();

    ASSERT_EQ(prover.openingPoints.size(), 3u);
    EXPECT_EQ(prover.openingPoints[0], 0u);
    EXPECT_EQ(prover.openingPoints[1], 1u);
    EXPECT_EQ(prover.openingPoints[2], 2u);
}

// A single f_i opened at one point still yields one entry -- the degenerate
// case fflonk's hardcoded three-set form cannot express.
TEST(SHPLONK, singleOpeningPoint) {
    PilFflonk::ShPlonkSetup setup;
    setup.power = 4;
    setup.powerW = 1;

    setup.f[0] = makeF(0, 1, {0});

    ProverAccess prover(E, &setup);
    prover.calculateOpeningPoints();

    ASSERT_EQ(prover.openingPoints.size(), 1u);
    EXPECT_EQ(prover.openingPoints[0], 0u);
}

// For each f_i and each opening point, calculateRoots emits nPols roots that
// are the nPols-th roots of a common value: raising each to the nPols power
// must give the same result. That holds however many polynomials and opening
// points the setup declares, which is the property the generalisation needs.
TEST(SHPLONK, rootsAreNthRootsOfACommonValue) {
    const uint32_t nPols = 2;

    PilFflonk::ShPlonkSetup setup;
    setup.power = 4;
    setup.powerW = nPols;

    setup.omegas["w" + std::to_string(nPols)] = rootOfUnity(nPols);
    setup.f[0] = makeF(0, nPols, {0});

    ProverAccess prover(E, &setup);
    prover.challengeXiSeed = E.fr.set(7);
    prover.calculateOpeningPoints();
    prover.calculateRoots();

    auto roots = prover.rootsMap["f0"];
    ASSERT_NE(roots, nullptr);

    AltBn128::FrElement expected = roots[0];
    for (uint32_t p = 1; p < nPols; p++) {
        expected = E.fr.mul(expected, roots[0]);
    }

    for (uint32_t j = 0; j < nPols; j++) {
        AltBn128::FrElement powed = roots[j];
        for (uint32_t p = 1; p < nPols; p++) {
            powed = E.fr.mul(powed, roots[j]);
        }
        ASSERT_TRUE(E.fr.eq(powed, expected)) << "root " << j << " is not an nPols-th root";
    }

    // The roots must be distinct, or the opening set collapses.
    for (uint32_t j = 1; j < nPols; j++) {
        ASSERT_FALSE(E.fr.eq(roots[j], roots[0])) << "root " << j << " duplicates root 0";
    }
}
