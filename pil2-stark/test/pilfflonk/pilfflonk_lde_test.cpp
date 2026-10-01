// Tests for PilFflonk::Lde: the coset LDE must equal Horner's rule at g·ω'^i, the round trips must
// be exact, a batch of columns must give the single-column results bit for bit, and every refused
// argument must come back as an exception, not an abort.
#include "pilfflonk_test.hpp"

#include <gmp.h>
#include <omp.h>

#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "alt_bn128.hpp"
#include "evaluations.hpp"
#include "fft.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_lde.hpp"
#include "polynomial.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_key_gpu.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu).
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void *dst, const void *src, size_t bytes);
#endif

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using PilFflonk::Lde;
using Column = std::vector<FrElement>;

Engine &E = Engine::engine;

bool equal(const FrElement &a, const FrElement &b) {
    return E.fr.eq(a, b);
}

// Bit for bit, Montgomery limbs and all. (memcmp must not see the null data() of an empty vector.)
bool identical(const FrElement *a, const FrElement *b, uint64_t n) {
    return n == 0 || std::memcmp(a, b, n * sizeof(FrElement)) == 0;
}

bool allZero(const FrElement *a, uint64_t n) {
    const Column zeros(n);
    return identical(a, zeros.data(), n);
}

// Elements below 2^253 < r from a fixed seed. Any value below r is the Montgomery form of some
// element, so the limbs are used as they are.
class Random {
public:
    explicit Random(uint64_t seed) : generator(seed) {}

    FrElement element() {
        FrElement e;
        for (uint64_t &limb : e.v) {
            limb = generator();
        }
        e.v[3] >>= 3;
        return e;
    }

    Column column(uint64_t n) {
        Column c(n);
        for (FrElement &e : c) {
            e = element();
        }
        return c;
    }

    uint64_t below(uint64_t n) { return generator() % n; }

private:
    std::mt19937_64 generator;
};

FrElement fromUI(uint64_t value) {
    FrElement e;
    E.fr.fromUI(e, value);
    return e;
}

FrElement power(const FrElement &base, const mpz_t exponent) {
    uint8_t littleEndian[32] = {};
    assert(mpz_sizeinbase(exponent, 256) <= sizeof(littleEndian));
    mpz_export(littleEndian, nullptr, -1, 1, -1, 0, exponent);
    FrElement result;
    E.fr.exp(result, base, littleEndian, sizeof(littleEndian));
    return result;
}

FrElement power(const FrElement &base, uint64_t exponent) {
    mpz_t e;
    mpz_init_set_ui(e, exponent);
    const FrElement result = power(base, e);
    mpz_clear(e);
    return result;
}

// (r - 1) / 2^k, into an initialised `out`.
void rMinusOneOver2ToThe(uint64_t k, mpz_t out) {
    const int parsed = mpz_set_str(out, R_HEX, 16);
    assert(parsed == 0);
    mpz_sub_ui(out, out, 1);
    mpz_fdiv_q_2exp(out, out, k);
}

// ω_{2^k} = 5^((r-1)/2^k), computed from r: independent of ffiasm's FFT tables.
FrElement rootOfUnity(uint64_t k) {
    mpz_t e;
    mpz_init(e);
    rMinusOneOver2ToThe(k, e);
    const FrElement w = power(fromUI(5), e);
    mpz_clear(e);
    return w;
}

// scale·ω_{2^k}^i for each i: points of H (scale 1) or of the coset g·H' (scale 5).
Column points(uint64_t k, uint64_t scale, const std::vector<uint64_t> &indices) {
    const FrElement w = rootOfUnity(k);
    const FrElement s = fromUI(scale);
    Column xs;
    for (uint64_t i : indices) {
        xs.push_back(E.fr.mul(s, power(w, i)));
    }
    return xs;
}

std::vector<uint64_t> allIndices(uint64_t n) {
    std::vector<uint64_t> indices(n);
    for (uint64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    return indices;
}

// The polynomial with these coefficients at each x, by Horner's rule: rapidsnark's
// Polynomial::evaluate.
Column horner(const FrElement *coefs, uint64_t nCoefs, const Column &xs) {
    Polynomial<Engine> poly(E, nCoefs);
    std::memcpy(poly.coef, coefs, nCoefs * sizeof(FrElement));
    poly.fixDegree();
    Column values(xs.size());
#pragma omp parallel for
    for (size_t i = 0; i < xs.size(); ++i) {
        values[i] = poly.evaluate(xs[i]);
    }
    return values;
}

// Single-column calls, on the caller's vectors.
std::unique_ptr<Lde::Poly> intt(const Lde &lde, Column &evals, Column &coefs, uint64_t blindLength) {
    FrElement *in = evals.data();
    FrElement *out = coefs.data();
    std::vector<std::unique_ptr<Lde::Poly>> polys = lde.intt(&in, &out, 1, blindLength);
    assert(polys.size() == 1);
    return std::move(polys[0]);
}

void extendCoset(const Lde &lde, const FrElement *coefs, FrElement *evals, uint64_t nCoefs) {
    lde.extendCoset(&coefs, &evals, 1, nCoefs);
}

void interpolateCoset(const Lde &lde, const FrElement *evals, FrElement *coefs) {
    lde.interpolateCoset(&evals, &coefs, 1);
}

// Pins g = 5 (pilfflonk/docs/protocol.md#extended-coset) and ffiasm's roots
// (pilfflonk/docs/protocol.md#notation).
void testCosetShift() {
    assert(PilFflonk::COSET_SHIFT == 5);
    assert(PilFflonk::MAX_NBITS_EXT == 28);

    mpz_t half; // (r - 1) / 2: Euler's criterion
    mpz_init(half);
    rMinusOneOver2ToThe(1, half);
    // 2 and 3 are squares, and so is 4: 5 is the smallest non-residue.
    assert(equal(power(fromUI(2), half), E.fr.one()));
    assert(equal(power(fromUI(3), half), E.fr.one()));
    assert(equal(power(fromUI(5), half), E.fr.negOne()));
    mpz_clear(half);

    // 2-adicity 28: ω_{2^28} is a primitive 2^28-th root of unity.
    FrElement w = rootOfUnity(28);
    for (int i = 0; i < 27; ++i) {
        E.fr.square(w, w);
    }
    assert(equal(w, E.fr.negOne()));

    // 5^(2^28) != 1: 5 is in no subgroup of order 2^k with k <= 28, so no coset g·H' meets H'.
    FrElement g = fromUI(5);
    for (int i = 0; i < 28; ++i) {
        E.fr.square(g, g);
    }
    assert(!equal(g, E.fr.one()));

    // ffiasm's roots are the powers of 5^((r-1)/2^k): 5 is its nqr. (root(0, 1) would read past its
    // table.)
    FFT<Engine::Fr> fft(uint64_t(1) << 10);
    for (uint32_t k = 1; k <= 10; ++k) {
        assert(equal(fft.root(k, 1), rootOfUnity(k)));
    }
}

// The INTT against Horner's rule on H: the coefficients evaluate back to the evaluations.
void testInttMatchesHorner() {
    Random random(1);
    for (uint64_t nBits = 0; nBits <= 8; ++nBits) {
        const Lde lde(nBits, nBits + 1);
        const uint64_t N = lde.domainSize();
        Column evals = random.column(N);
        const Column original = evals;
        Column coefs(N);
        const std::unique_ptr<Lde::Poly> poly = intt(lde, evals, coefs, 0);
        assert(poly->coef == coefs.data() && poly->getLength() == N);
        assert(identical(evals.data(), original.data(), N));
        const Column values = horner(coefs.data(), N, points(nBits, 1, allIndices(N)));
        assert(identical(values.data(), evals.data(), N));
    }
}

// nCoefs random coefficients extended to g·H' must equal Horner's rule at g·ω'^i, i in `indices`.
void checkExtendAgainstHorner(const Lde &lde, uint64_t nBitsExt, uint64_t nCoefs, const std::vector<uint64_t> &indices,
                              Random &random) {
    const Column coefs = random.column(nCoefs);
    Column evals(lde.extendedSize());
    extendCoset(lde, coefs.data(), evals.data(), nCoefs);
    const Column expected = horner(coefs.data(), nCoefs, points(nBitsExt, 5, indices));
    for (size_t k = 0; k < indices.size(); ++k) {
        assert(equal(evals[indices[k]], expected[k]));
    }
}

void testExtendMatchesHorner() {
    Random random(2);
    // Every point.
    for (uint64_t nBits = 0; nBits <= 6; ++nBits) {
        for (uint64_t nBitsExt = nBits; nBitsExt <= nBits + 3; ++nBitsExt) {
            const Lde lde(nBits, nBitsExt);
            const uint64_t N = lde.domainSize();
            const uint64_t NExt = lde.extendedSize();
            for (uint64_t nCoefs : {uint64_t(1), N, (N + NExt) / 2, NExt}) {
                checkExtendAgainstHorner(lde, nBitsExt, nCoefs, allIndices(NExt), random);
            }
        }
    }
    // Sampled points.
    for (uint64_t nBits : {8, 12, 16}) {
        for (uint64_t nBitsExt = nBits + 1; nBitsExt <= nBits + 3; ++nBitsExt) {
            const Lde lde(nBits, nBitsExt);
            const uint64_t N = lde.domainSize();
            const uint64_t NExt = lde.extendedSize();
            std::vector<uint64_t> indices = {0, 1, 2, N, NExt / 2, NExt - 1};
            for (int k = 0; k < 8; ++k) {
                indices.push_back(random.below(NExt));
            }
            for (uint64_t nCoefs : {N + 3, NExt}) {
                checkExtendAgainstHorner(lde, nBitsExt, nCoefs, indices, random);
            }
        }
    }
}

void testRoundTrips() {
    Random random(3);
    FFT<Engine::Fr> referenceFft(uint64_t(1) << 16);
    for (uint64_t nBits = 0; nBits <= 16; ++nBits) {
        for (uint64_t nBitsExt = nBits; nBitsExt <= nBits + 3; ++nBitsExt) {
            const Lde lde(nBits, nBitsExt);
            const uint64_t N = lde.domainSize();
            const uint64_t NExt = lde.extendedSize();

            // H: evaluations → coefficients → evaluations, the way back by rapidsnark's Evaluations.
            Column evals = random.column(N);
            Column coefs(N);
            const std::unique_ptr<Lde::Poly> poly = intt(lde, evals, coefs, 0);
            const Evaluations<Engine> back(E, &referenceFft, *poly, N);
            assert(identical(back.eval, evals.data(), N));

            // g·H': coefficients → evaluations → coefficients, zero-padded to N'.
            const uint64_t nCoefs = 1 + random.below(NExt);
            const Column shortCoefs = random.column(nCoefs);
            Column coset(NExt);
            Column recovered(NExt, random.element());
            extendCoset(lde, shortCoefs.data(), coset.data(), nCoefs);
            interpolateCoset(lde, coset.data(), recovered.data());
            assert(identical(recovered.data(), shortCoefs.data(), nCoefs));
            assert(allZero(recovered.data() + nCoefs, NExt - nCoefs));

            // The same in place.
            Column buffer = shortCoefs;
            buffer.resize(NExt, random.element()); // beyond nCoefs: must be ignored
            extendCoset(lde, buffer.data(), buffer.data(), nCoefs);
            assert(identical(buffer.data(), coset.data(), NExt));
            interpolateCoset(lde, buffer.data(), buffer.data());
            assert(identical(buffer.data(), recovered.data(), NExt));

            // g·H': evaluations → coefficients → evaluations.
            const Column cosetEvals = random.column(NExt);
            Column cosetCoefs(NExt);
            Column again(NExt);
            interpolateCoset(lde, cosetEvals.data(), cosetCoefs.data());
            extendCoset(lde, cosetCoefs.data(), again.data(), NExt);
            assert(identical(again.data(), cosetEvals.data(), NExt));
        }
    }
}

void testInttKeepsRoomForBlinding() {
    Random random(4);
    for (uint64_t nBits : {0, 1, 2, 5, 10, 16}) {
        const Lde lde(nBits, nBits + 3);
        const uint64_t N = lde.domainSize();
        const uint64_t NExt = lde.extendedSize();
        Column evals = random.column(N);
        Column plain(N);
        const std::unique_ptr<Lde::Poly> plainPoly = intt(lde, evals, plain, 0);

        for (uint64_t blindLength : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(7), NExt - N}) {
            Column coefs(N + blindLength, random.element()); // stale data: the tail must come out zeroed
            const std::unique_ptr<Lde::Poly> poly = intt(lde, evals, coefs, blindLength);
            assert(poly->coef == coefs.data() && poly->getLength() == N + blindLength);
            assert(poly->getDegree() == plainPoly->getDegree());
            assert(identical(coefs.data(), plain.data(), N));
            assert(allZero(coefs.data() + N, blindLength));

            // The room is where blindCoefficients adds (X^N - 1)·b(X), which vanishes on H.
            if (nBits <= 5) {
                Column b = random.column(blindLength);
                poly->blindCoefficients(b.data(), blindLength);
                const Column values = horner(coefs.data(), N + blindLength, points(nBits, 1, allIndices(N)));
                assert(identical(values.data(), evals.data(), N));
            }
        }
    }
}

// Everything a column goes through: INTT with blinding room, LDE, and back from the coset.
struct Results {
    std::vector<Column> coefs;
    std::vector<uint64_t> degrees;
    std::vector<Column> coset;
    std::vector<Column> back;
};

Results runColumns(const Lde &lde, std::vector<Column> evals, uint64_t blindLength, bool batched) {
    const uint64_t nCols = evals.size();
    const uint64_t nCoefs = lde.domainSize() + blindLength;
    const uint64_t NExt = lde.extendedSize();
    Results results{std::vector<Column>(nCols, Column(nCoefs)), std::vector<uint64_t>(nCols),
                    std::vector<Column>(nCols, Column(NExt)), std::vector<Column>(nCols, Column(NExt))};
    std::vector<FrElement *> evalPtrs, coefPtrs, cosetPtrs, backPtrs;
    for (uint64_t c = 0; c < nCols; ++c) {
        evalPtrs.push_back(evals[c].data());
        coefPtrs.push_back(results.coefs[c].data());
        cosetPtrs.push_back(results.coset[c].data());
        backPtrs.push_back(results.back[c].data());
    }

    const uint64_t step = batched ? nCols : 1;
    for (uint64_t first = 0; first < nCols; first += step) {
        const std::vector<std::unique_ptr<Lde::Poly>> polys =
            lde.intt(&evalPtrs[first], &coefPtrs[first], step, blindLength);
        assert(polys.size() == step);
        for (uint64_t c = 0; c < step; ++c) {
            assert(polys[c] != nullptr);
            assert(polys[c]->coef == coefPtrs[first + c] && polys[c]->getLength() == nCoefs);
            results.degrees[first + c] = polys[c]->getDegree();
        }
        lde.extendCoset(&coefPtrs[first], &cosetPtrs[first], step, nCoefs);
        lde.interpolateCoset(&cosetPtrs[first], &backPtrs[first], step);
    }
    return results;
}

void expectIdentical(const Results &a, const Results &b) {
    assert(a.degrees == b.degrees);
    for (size_t c = 0; c < a.coefs.size(); ++c) {
        assert(identical(a.coefs[c].data(), b.coefs[c].data(), a.coefs[c].size()));
        assert(identical(a.coset[c].data(), b.coset[c].data(), a.coset[c].size()));
        assert(identical(a.back[c].data(), b.back[c].data(), a.back[c].size()));
    }
}

// A batch against one call per column, with fewer columns than threads (one after another) and
// with at least as many (one per thread), down to a single thread. The single-column calls run on
// one thread: at the full thread count the other tests cover them, and here they would only make
// the run slow on a loaded machine, where every region of the whole team waits for its slowest
// thread.
void testBatchMatchesSingleColumns() {
    Random random(5);
    const int maxThreads = omp_get_max_threads();
    struct Case {
        uint64_t nBits, nBitsExt;
        std::vector<uint64_t> nCols;
    };
    std::vector<uint64_t> oneTo17;
    for (uint64_t n = 1; n <= 17; ++n) {
        oneTo17.push_back(n);
    }
    const std::vector<Case> cases = {
        {1, 2, oneTo17},
        {4, 6, oneTo17},
        {9, 12, {1, 2, 3, 5, 8, 16, 17}},
        {16, 17, {1, 4, 17}},
        {4, 6, {static_cast<uint64_t>(maxThreads) + 1}}, // one per thread at the full thread count
    };
    for (const Case &test : cases) {
        const Lde lde(test.nBits, test.nBitsExt);
        for (uint64_t nCols : test.nCols) {
            std::vector<Column> evals;
            for (uint64_t c = 0; c < nCols; ++c) {
                evals.push_back(random.column(lde.domainSize()));
            }
            const uint64_t blindLength = 2;
            omp_set_num_threads(1);
            const Results single = runColumns(lde, evals, blindLength, false);
            for (int threads : {maxThreads, 4, 1}) {
                omp_set_num_threads(threads);
                const Results batch = runColumns(lde, evals, blindLength, true);
                // One column per thread leaves the caller's thread count as it was.
                assert(omp_get_max_threads() == threads);
                omp_set_num_threads(maxThreads);
                expectIdentical(single, batch);
            }
        }
    }
}

// The coset in parts (pilfflonk/docs/protocol.md#q-in-parts): for every size S = 2^partBits from N
// to N' and every part p, extendCosetPart gives evaluation p + (N'/S)·i of extendCoset as its i-th,
// bit for bit, for polynomials of fewer, as many and more coefficients than a part has (their
// coefficients fold into it), in a batch and one column at a time, and in place.
void testExtendCosetParts() {
    Random random(7);
    struct Case {
        uint64_t nBits, nBitsExt;
    };
    for (const Case &test : std::vector<Case>{{0, 3}, {2, 2}, {3, 5}, {4, 7}, {9, 11}}) {
        const Lde lde(test.nBits, test.nBitsExt);
        const uint64_t N = lde.domainSize(), NExt = lde.extendedSize();
        std::vector<uint64_t> lengths = {1, N, NExt};
        if (N + 3 <= NExt) {
            lengths.push_back(N + 3);
        }
        if (NExt > 2) {
            lengths.push_back(NExt - 1);
        }
        for (uint64_t nCoefs : lengths) {
            const std::vector<Column> coefs = {random.column(nCoefs), random.column(nCoefs)};
            std::vector<Column> whole(2, Column(NExt));
            for (uint64_t c = 0; c < 2; ++c) {
                extendCoset(lde, coefs[c].data(), whole[c].data(), nCoefs);
            }
            for (uint64_t partBits = test.nBits; partBits <= test.nBitsExt; ++partBits) {
                const uint64_t S = uint64_t(1) << partBits, nParts = NExt / S;
                for (uint64_t part = 0; part < nParts; ++part) {
                    Column expected(S);
                    for (uint64_t i = 0; i < S; ++i) {
                        expected[i] = whole[0][part + nParts * i];
                    }
                    // A batch of two columns, the first against the whole coset.
                    std::vector<Column> out(2, Column(S));
                    const FrElement *in[2] = {coefs[0].data(), coefs[1].data()};
                    FrElement *dst[2] = {out[0].data(), out[1].data()};
                    lde.extendCosetPart(in, dst, 2, nCoefs, partBits, part);
                    assert(identical(out[0].data(), expected.data(), S));
                    for (uint64_t i = 0; i < S; ++i) {
                        assert(identical(&out[1][i], &whole[1][part + nParts * i], 1));
                    }
                    // In place, in a buffer of max(nCoefs, S) elements.
                    Column buffer = coefs[0];
                    buffer.resize(std::max(nCoefs, S));
                    FrElement *self = buffer.data();
                    lde.extendCosetPart(&self, &self, 1, nCoefs, partBits, part);
                    assert(identical(buffer.data(), expected.data(), S));
                }
            }
        }
    }
}

template <typename Call>
void expectInvalid(Call call, const char *message) {
    try {
        call();
    } catch (const std::invalid_argument &e) {
        assert(std::strstr(e.what(), message) != nullptr);
        return;
    }
    assert(!"expected std::invalid_argument");
}

void testRefusedArguments() {
    expectInvalid([] { Lde(29, 29); }, "nBitsExt = 29 exceeds 28");
    expectInvalid([] { Lde(0, 29); }, "nBitsExt = 29 exceeds 28");
    expectInvalid([] { Lde(64, 64); }, "nBitsExt = 64 exceeds 28");
    expectInvalid([] { Lde(0, UINT64_MAX); }, "exceeds 28");
    expectInvalid([] { Lde(5, 4); }, "nBits = 5 exceeds nBitsExt = 4");

    Random random(6);
    const Lde lde(3, 5);
    const uint64_t N = lde.domainSize();
    const uint64_t NExt = lde.extendedSize();
    Column a = random.column(NExt), b = random.column(NExt), c = random.column(NExt);
    const Column aBefore = a, bBefore = b, cBefore = c;
    FrElement *pa = a.data();
    FrElement *pb = b.data();
    FrElement *pair[2] = {pa, pb};
    FrElement *secondInPlace[2] = {c.data(), pb};
    FrElement *withNull[2] = {pa, nullptr};
    const FrElement *constA = pa;
    const FrElement *constPair[2] = {pa, pb};
    const FrElement *constWithNull[2] = {pb, nullptr};
    FrElement *outputs[2] = {pb, pa};

    // intt
    expectInvalid([&] { lde.intt(&pa, &pb, 0); }, "Lde::intt: no columns");
    expectInvalid([&] { lde.intt(nullptr, &pb, 1); }, "Lde::intt: evals is null");
    expectInvalid([&] { lde.intt(&pa, nullptr, 1); }, "Lde::intt: coefs is null");
    expectInvalid([&] { lde.intt(withNull, pair, 2); }, "Lde::intt: evals[1] is null");
    expectInvalid([&] { lde.intt(pair, withNull, 2); }, "Lde::intt: coefs[1] is null");
    expectInvalid([&] { lde.intt(&pa, &pa, 1); }, "Lde::intt: evals[0] is coefs[0]");
    expectInvalid([&] { lde.intt(pair, secondInPlace, 2); }, "Lde::intt: evals[1] is coefs[1]");
    expectInvalid([&] { lde.intt(&pa, &pb, 1, NExt - N + 1); }, "N + blindLength = 8 + 25 coefficients exceed the 32");
    expectInvalid([&] { lde.intt(&pa, &pb, 1, UINT64_MAX); }, "exceed the 32");

    // extendCoset
    expectInvalid([&] { lde.extendCoset(&constA, &pb, 0, 1); }, "Lde::extendCoset: no columns");
    expectInvalid([&] { lde.extendCoset(nullptr, &pb, 1, 1); }, "Lde::extendCoset: coefs is null");
    expectInvalid([&] { lde.extendCoset(&constA, nullptr, 1, 1); }, "Lde::extendCoset: evals is null");
    expectInvalid([&] { lde.extendCoset(constWithNull, outputs, 2, 1); }, "Lde::extendCoset: coefs[1] is null");
    expectInvalid([&] { lde.extendCoset(constPair, withNull, 2, 1); }, "Lde::extendCoset: evals[1] is null");
    expectInvalid([&] { lde.extendCoset(&constA, &pb, 1, 0); }, "Lde::extendCoset: no coefficients");
    expectInvalid([&] { lde.extendCoset(&constA, &pb, 1, NExt + 1); }, "33 coefficients exceed the 32");
    expectInvalid([&] { lde.extendCoset(&constA, &pb, 1, UINT64_MAX); }, "coefficients exceed the 32");

    // extendCosetPart
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 0, 1, 3, 0); }, "Lde::extendCosetPart: no columns");
    expectInvalid([&] { lde.extendCosetPart(nullptr, &pb, 1, 1, 3, 0); }, "Lde::extendCosetPart: coefs is null");
    expectInvalid([&] { lde.extendCosetPart(&constA, nullptr, 1, 1, 3, 0); }, "Lde::extendCosetPart: evals is null");
    expectInvalid([&] { lde.extendCosetPart(constPair, withNull, 2, 1, 3, 0); },
                  "Lde::extendCosetPart: evals[1] is null");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, 0, 3, 0); }, "Lde::extendCosetPart: no coefficients");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, NExt + 1, 3, 0); }, "33 coefficients exceed the 32");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, 1, 2, 0); },
                  "a part of 2^2 points, and the parts have from 8 to 32");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, 1, 6, 0); }, "a part of 2^6 points");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, 1, 3, 4); }, "part 4 of the 4 of 2^3 points");
    expectInvalid([&] { lde.extendCosetPart(&constA, &pb, 1, 1, 5, 1); }, "part 1 of the 1 of 2^5 points");

    // interpolateCoset
    expectInvalid([&] { lde.interpolateCoset(&constA, &pb, 0); }, "Lde::interpolateCoset: no columns");
    expectInvalid([&] { lde.interpolateCoset(nullptr, &pb, 1); }, "Lde::interpolateCoset: evals is null");
    expectInvalid([&] { lde.interpolateCoset(&constA, nullptr, 1); }, "Lde::interpolateCoset: coefs is null");
    expectInvalid([&] { lde.interpolateCoset(constWithNull, outputs, 2); }, "Lde::interpolateCoset: evals[1] is null");
    expectInvalid([&] { lde.interpolateCoset(constPair, withNull, 2); }, "Lde::interpolateCoset: coefs[1] is null");

    // A refused call writes nothing, not even to the columns it could have processed.
    assert(identical(a.data(), aBefore.data(), NExt));
    assert(identical(b.data(), bBefore.data(), NExt));
    assert(identical(c.data(), cBefore.data(), NExt));
}

// The GPU (pilfflonk/docs/performance.md#why-the-proof-is-the-same), where there is one: sppark's NTT
// and INTT on the device (transformOnDevice) are ffiasm's fft and ifft, bit for bit, Montgomery limbs
// and all, at every size from one point to 2^20 (the coset transforms around them on the device are
// pilfflonk_gpu_test.cpp's). And what a Gpu refuses.
void testTheGpuIsTheCpu() {
#ifdef __USE_CUDA__
    if (!gpuUnderTest("the transforms on the GPU")) {
        return;
    }
    using PilFflonk::Gpu;
    Random random(8);
    const uint64_t maxBits = 20;
    FFT<Engine::Fr> fft(uint64_t(1) << maxBits);
    const PilFflonk::DeviceBuffer data((uint64_t(1) << maxBits) * sizeof(FrElement));
    for (uint64_t bits = 0; bits <= maxBits; ++bits) {
        const uint64_t n = uint64_t(1) << bits;
        const Column values = random.column(n);
        for (bool inverse : {false, true}) {
            Column expected = values;
            if (inverse) {
                fft.ifft(expected.data(), n);
            } else {
                fft.fft(expected.data(), n);
            }
            gpu_plonk_memcpy_h2d(data.data(), values.data(), n * sizeof(FrElement));
            PilFflonk::transformOnDevice(data.data(), bits, inverse);
            Column out(n);
            gpu_plonk_memcpy_d2h(out.data(), data.data(), n * sizeof(FrElement));
            assert(identical(out.data(), expected.data(), n));
        }
    }

    // What a Gpu refuses.
    Engine::G1PointAffine generator = E.g1.oneAffine();
    expectInvalid([&] { Gpu(nullptr, 1); }, "Gpu::Gpu: points is null");
    expectInvalid([&] { Gpu(&generator, 0); }, "Gpu::Gpu: no points");
#else
    // A library built without the GPU: nothing to compare (pilfflonk_gpu_test.cpp tests the refusal).
    assert(!PilFflonk::gpuAvailable());
#endif
}

} // namespace

void runLdeTests() {
    testCosetShift();
    testInttMatchesHorner();
    testExtendMatchesHorner();
    testRoundTrips();
    testInttKeepsRoomForBlinding();
    testBatchMatchesSingleColumns();
    testExtendCosetParts();
    testRefusedArguments();
    testTheGpuIsTheCpu();
}

} // namespace PilFflonkTest
