// The CPU's MSM (pilfflonk_msm) against ffiasm: the sum of its scalar multiplications for small n, and
// its Pippenger for large n. Above all on what random scalars do not reach: repeated scalars, the
// same base twice (a doubling in the affine buckets), a base and its negation (a bucket that
// cancels), points at infinity, and 0, 1 and r − 1.

#include "pilfflonk_test.hpp"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_msm.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;

Engine &E = Engine::engine;

// A scalar as the MSM takes it: four canonical little-endian limbs.
struct Scalar {
    uint64_t limbs[4] = {};
};

Scalar fromLimb(uint64_t low) {
    Scalar s;
    s.limbs[0] = low;
    return s;
}

// r − 1, the largest canonical scalar.
Scalar rMinusOne() {
    const Bytes32 r(R_HEX);
    Scalar s;
    std::memcpy(s.limbs, r.bytes, 32);
    s.limbs[0] -= 1; // r is odd
    return s;
}

struct Rng {
    std::mt19937_64 gen;
    explicit Rng(uint64_t seed) : gen(seed) {}

    // Below 2^253 < r.
    Scalar scalar() {
        Scalar s;
        for (uint64_t &l : s.limbs) {
            l = gen();
        }
        s.limbs[3] &= 0x1fffffffffffffffull;
        return s;
    }

    G1PointAffine point() {
        G1Point p = g1Times(fromUI(gen() | 1));
        G1PointAffine a;
        E.g1.copy(a, p);
        return a;
    }
};

// Σ scalars[i]·bases[i], one scalar multiplication at a time.
G1Point naive(std::vector<G1PointAffine> bases, std::vector<Scalar> scalars) {
    G1Point sum;
    E.g1.copy(sum, E.g1.zero());
    for (size_t i = 0; i < bases.size(); ++i) {
        G1Point base, term;
        E.g1.copy(base, bases[i]);
        E.g1.mulByScalar(term, base, reinterpret_cast<uint8_t *>(scalars[i].limbs), 32);
        E.g1.add(sum, sum, term);
    }
    return sum;
}

// ffiasm's Pippenger, for the sizes the naive sum is too slow for.
G1Point ffiasm(std::vector<G1PointAffine> bases, std::vector<Scalar> scalars) {
    G1Point result;
    E.g1.multiMulByScalar(result, bases.data(), reinterpret_cast<uint8_t *>(scalars.data()), 32,
                          static_cast<unsigned int>(bases.size()));
    return result;
}

G1Point msm(const std::vector<G1PointAffine> &bases, const std::vector<Scalar> &scalars) {
    return PilFflonk::msm(bases.data(), &scalars[0].limbs[0], bases.size());
}

void check(const std::vector<G1PointAffine> &bases, const std::vector<Scalar> &scalars, const char *what) {
    const G1Point want = bases.size() <= 4096 ? naive(bases, scalars) : ffiasm(bases, scalars);
    if (!samePoint(msm(bases, scalars), want)) {
        std::fprintf(stderr, "pilfflonk_test: the MSM of %zu points (%s) is not ffiasm's\n", bases.size(), what);
        assert(false);
    }
}

std::vector<G1PointAffine> distinctPoints(Rng &rng, size_t n) {
    std::vector<G1PointAffine> bases(n);
    for (G1PointAffine &b : bases) {
        b = rng.point();
    }
    return bases;
}

void testRandom() {
    Rng rng(1);
    for (size_t n : {0, 1, 2, 3, 255, 256, 257, 1000, 4096}) {
        std::vector<Scalar> scalars(n);
        for (Scalar &s : scalars) {
            s = rng.scalar();
        }
        check(distinctPoints(rng, n), scalars, "random scalars");
    }
}

void testRepeatedScalars() {
    Rng rng(2);
    for (size_t n : {300, 4096, 1 << 16}) {
        const std::vector<G1PointAffine> bases = distinctPoints(rng, n);
        const Scalar s = rng.scalar();
        check(bases, std::vector<Scalar>(n, s), "one scalar repeated");
        std::vector<Scalar> few(n);
        for (size_t i = 0; i < n; ++i) {
            few[i] = fromLimb(i % 3);
        }
        check(bases, few, "scalars 0, 1 and 2");
        std::vector<Scalar> l1(n);
        l1[n / 2] = fromLimb(1);
        check(bases, l1, "an L1 column: one 1");
        check(bases, std::vector<Scalar>(n), "all zero");
        std::vector<Scalar> extremes(n);
        for (size_t i = 0; i < n; ++i) {
            extremes[i] = i % 2 ? rMinusOne() : fromLimb(1);
        }
        check(bases, extremes, "1 and r - 1");
    }
}

void testAffineEdgeCases() {
    Rng rng(3);
    for (size_t n : {512, 4096}) {
        std::vector<Scalar> scalars(n);
        for (Scalar &s : scalars) {
            s = rng.scalar();
        }
        // The same base throughout: buckets add a point to itself.
        const std::vector<G1PointAffine> same(n, rng.point());
        check(same, scalars, "one base repeated");
        check(same, std::vector<Scalar>(n, scalars[0]), "one base and one scalar repeated");

        // P and −P with the same scalar: each pair cancels, the sum is the point at infinity.
        std::vector<G1PointAffine> pairs(n);
        std::vector<Scalar> paired(n);
        for (size_t i = 0; i + 1 < n; i += 2) {
            pairs[i] = rng.point();
            E.g1.neg(pairs[i + 1], pairs[i]);
            paired[i] = paired[i + 1] = rng.scalar();
        }
        check(pairs, paired, "a base and its negation");
        G1Point cancelled = msm(pairs, paired);
        assert(E.g1.isZero(cancelled));

        // Points at infinity among the bases.
        std::vector<G1PointAffine> withInfinity = distinctPoints(rng, n);
        for (size_t i = 0; i < n; i += 7) {
            E.g1.copy(withInfinity[i], E.g1.zero());
        }
        check(withInfinity, scalars, "points at infinity");
    }
}

void testRefusesNonCanonicalScalars() {
    Rng rng(4);
    const std::vector<G1PointAffine> bases = distinctPoints(rng, 300);
    std::vector<Scalar> scalars(300, fromLimb(1));
    scalars[123].limbs[3] = 1ull << 63;
    bool refused = false;
    try {
        msm(bases, scalars);
    } catch (const std::invalid_argument &) {
        refused = true;
    }
    assert(refused);
}

} // namespace

void runMsmTests() {
    testRandom();
    testRepeatedScalars();
    testAffineEdgeCases();
    testRefusesNonCanonicalScalars();
    std::printf("pilfflonk_test: the CPU's MSM is ffiasm's, repeated scalars and bases, negations and infinity\n");
}

} // namespace PilFflonkTest
