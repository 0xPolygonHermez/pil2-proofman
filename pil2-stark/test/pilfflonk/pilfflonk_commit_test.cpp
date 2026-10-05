// Tests for the KZG commitment, the fflonk packing and the fixed commitments, with a test SRS of
// known τ: commit(p) must be [p(τ)]₁; pack() must interleave as f(X) = Σ_j p_j(X^k)·X^j, so that
// its commitment is that of the interleaved coefficients and [f(τ)]₁ with f(τ) = Σ_j p_j(τ^k)·τ^j;
// and pilfflonk_commit_fixed must equal INTT + packing + commitment done by hand.
#include "pilfflonk_test.hpp"

#include <climits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "alt_bn128.hpp"
#include "fft.hpp"
#include "pilfflonk_api.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_test_ptau.hpp"
#include "polynomial.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_key_gpu.hpp"
#endif

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;
using Poly = Polynomial<Engine>;
using Column = std::vector<FrElement>;
using PilFflonk::Lde;
using PilFflonk::Srs;

Engine &E = Engine::engine;

// The powers of the test SRS: enough for k·N = 12·64.
constexpr uint64_t N_G1 = 1024;

const std::vector<uint64_t> PACKINGS = {1, 2, 3, 4, 6, 12};

// Elements below 2^253 < r from a fixed seed, used as Montgomery limbs as they are.
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

private:
    std::mt19937_64 generator;
};

// The SRS every test commits with: the first N_G1 powers of a test ptau.
const Srs &testSrs() {
    static const Srs srs = [] {
        TestDir dir;
        const std::string ptau = dir.file("commit.ptau");
        writeTestPtau(ptau, N_G1);
        return Srs::fromPtau(ptau, N_G1);
    }();
    return srs;
}

bool identical(const FrElement *a, const FrElement *b, uint64_t n) {
    return n == 0 || std::memcmp(a, b, n * sizeof(FrElement)) == 0;
}

bool allZero(const FrElement *a, uint64_t n) {
    const Column zeros(n);
    return identical(a, zeros.data(), n);
}

// A polynomial owning a copy of `coefs`, with its degree fixed.
std::unique_ptr<Poly> polynomial(const Column &coefs) {
    assert(!coefs.empty());
    std::unique_ptr<Poly> p(new Poly(E, coefs.size()));
    std::memcpy(p->coef, coefs.data(), coefs.size() * sizeof(FrElement));
    p->fixDegree();
    return p;
}

std::vector<Poly *> pointers(const std::vector<std::unique_ptr<Poly>> &polys) {
    std::vector<Poly *> raw;
    for (const std::unique_ptr<Poly> &p : polys) {
        raw.push_back(p.get());
    }
    return raw;
}

// f(τ) = Σ_j p_j(τ^k)·τ^j, by Horner's rule on each p_j (rapidsnark's Polynomial::evaluate).
FrElement packedAt(const std::vector<Poly *> &polys, const FrElement &tau) {
    const uint64_t k = polys.size();
    const FrElement tauK = power(tau, k);
    FrElement sum = E.fr.zero();
    FrElement tauJ = E.fr.one();
    for (uint64_t j = 0; j < k; ++j) {
        E.fr.add(sum, sum, E.fr.mul(polys[j]->evaluate(tauK), tauJ));
        E.fr.mul(tauJ, tauJ, tau);
    }
    return sum;
}

// Written out here: coefficient i·k + j of f is coefficient i of p_j.
Column interleave(const std::vector<Poly *> &polys) {
    const uint64_t k = polys.size();
    uint64_t maxLength = 0;
    for (const Poly *p : polys) {
        maxLength = std::max(maxLength, p->getLength());
    }
    Column f(k * maxLength);
    for (uint64_t j = 0; j < k; ++j) {
        for (uint64_t i = 0; i < polys[j]->getLength(); ++i) {
            f[i * k + j] = polys[j]->coef[i];
        }
    }
    return f;
}

// pack() into a buffer of stale data: what it returns must not depend on it.
Column packed(const std::vector<Poly *> &polys, uint64_t &nCoefs, Random &random) {
    uint64_t maxLength = 0;
    for (const Poly *p : polys) {
        maxLength = std::max(maxLength, p->getLength());
    }
    const uint64_t length = PilFflonk::packedBufferLength(polys.size(), maxLength);
    Column buffer = random.column(length);
    nCoefs = PilFflonk::pack(polys.data(), polys.size(), buffer.data(), length);
    assert(nCoefs <= length);
    return buffer;
}

template <typename Call>
void expectInvalid(Call call, const char *message) {
    try {
        call();
    } catch (const std::invalid_argument &e) {
        if (std::strstr(e.what(), message) == nullptr) {
            std::fprintf(stderr, "unexpected message: %s\n", e.what());
            assert(!"the exception does not say what was expected");
        }
        return;
    }
    assert(!"expected std::invalid_argument");
}

void testCommitIsTheEvaluationAtTau() {
    Random random(1);
    const Srs &srs = testSrs();
    const FrElement tau = testTau();
    for (uint64_t n : {1, 2, 3, 5, 64, 257, 1023, 1024}) {
        const Column coefs = random.column(n);
        assert(samePoint(srs.commit(coefs.data(), n), g1Times(polynomial(coefs)->evaluate(tau))));
    }

    // The scalars are the canonical values, not Montgomery limbs: 1 commits to G, X to [τ]₁.
    const Column one = {E.fr.one()};
    assert(samePoint(srs.commit(one.data(), 1), E.g1.oneAffine()));
    const Column x = {E.fr.zero(), E.fr.one()};
    assert(samePoint(srs.commit(x.data(), 2), srs.g1(1)));
    assert(samePoint(g1Times(tau), srs.g1(1)));
    // No coefficients, or only zeros: the point at infinity.
    G1Point none = srs.commit(nullptr, 0);
    assert(E.g1.isZero(none));
    const Column zeros(5);
    G1Point zero = srs.commit(zeros.data(), zeros.size());
    assert(E.g1.isZero(zero));

    const Column tooLong = random.column(N_G1 + 1);
    expectInvalid([&] { srs.commit(tooLong.data(), N_G1 + 1); },
                  "Srs::commit: 1025 coefficients exceed the 1024 powers [τ^i]₁ of the SRS");
    expectInvalid([&] { srs.commit(nullptr, 1); }, "Srs::commit: coefs is null");
}

void testPackedBufferLength() {
    assert(PilFflonk::packedBufferLength(1, 1) == 1);
    assert(PilFflonk::packedBufferLength(1, 16) == 16);
    assert(PilFflonk::packedBufferLength(3, 5) == 16);
    assert(PilFflonk::packedBufferLength(3, 16) == 64);
    assert(PilFflonk::packedBufferLength(12, 64) == 1024);
    assert(PilFflonk::packedBufferLength(1, uint64_t(1) << 63) == uint64_t(1) << 63);
    expectInvalid([] { PilFflonk::packedBufferLength(0, 1); }, "k = 0 polynomials");
    expectInvalid([] { PilFflonk::packedBufferLength(1, 0); }, "n = 0 coefficients");
    expectInvalid([] { PilFflonk::packedBufferLength(3, uint64_t(1) << 62); }, "exceeds 2^63");
    expectInvalid([] { PilFflonk::packedBufferLength(UINT64_MAX, UINT64_MAX); }, "exceeds 2^63");
}

// Every k of PACKINGS, with p_j of assorted degrees: full, one short, low, constant, zero.
void testPackInterleaves() {
    Random random(2);
    const Srs &srs = testSrs();
    const FrElement tau = testTau();
    const uint64_t n = 64;
    for (uint64_t k : PACKINGS) {
        std::vector<std::unique_ptr<Poly>> owned;
        uint64_t expectedCoefs = 0;
        for (uint64_t j = 0; j < k; ++j) {
            const uint64_t degrees[] = {n - 1, n - 2, 7, 0, 0};
            Column coefs(n);
            if (j % 5 != 4) {
                for (uint64_t i = 0; i <= degrees[j % 5]; ++i) {
                    coefs[i] = random.element();
                }
            }
            owned.push_back(polynomial(coefs));
            assert(owned.back()->getDegree() == degrees[j % 5]);
            expectedCoefs = std::max(expectedCoefs, k * degrees[j % 5] + j + 1);
        }
        const std::vector<Poly *> polys = pointers(owned);

        uint64_t nCoefs = 0;
        const Column f = packed(polys, nCoefs, random);
        const Column hand = interleave(polys);
        assert(nCoefs == expectedCoefs);
        assert(identical(f.data(), hand.data(), nCoefs));
        assert(allZero(hand.data() + nCoefs, hand.size() - nCoefs));

        const G1Point commitment = srs.commit(f.data(), nCoefs);
        assert(samePoint(commitment, srs.commit(hand.data(), hand.size())));
        assert(samePoint(commitment, g1Times(packedAt(polys, tau))));
    }
}

// f of degree a power of two: CPolynomial's polynomial over the buffer is one coefficient short
// then, and pack() must count that coefficient all the same.
void testPackDegreeAPowerOfTwo() {
    Random random(3);
    const Srs &srs = testSrs();
    const FrElement tau = testTau();
    struct Case {
        std::vector<uint64_t> lengths; // the coefficients of each p_j, the last one nonzero
        uint64_t nCoefs;
    };
    const std::vector<Case> cases = {
        {{5, 4}, 9},        // 2·4 + 0 = 8
        {{5}, 5},           // k = 1, degree 4
        {{17}, 17},         // k = 1, degree 16
        {{3, 1, 1, 1}, 9},  // 4·2 + 0 = 8
        {{1, 6, 1}, 17},    // 3·5 + 1 = 16
        {{1, 1, 1}, 3},     // 3·0 + 2 = 2
    };
    for (const Case &test : cases) {
        std::vector<std::unique_ptr<Poly>> owned;
        for (uint64_t length : test.lengths) {
            owned.push_back(polynomial(random.column(length)));
        }
        const std::vector<Poly *> polys = pointers(owned);
        uint64_t nCoefs = 0;
        const Column f = packed(polys, nCoefs, random);
        const Column hand = interleave(polys);
        assert(nCoefs == test.nCoefs);
        assert(identical(f.data(), hand.data(), nCoefs));
        assert(samePoint(srs.commit(f.data(), nCoefs), g1Times(packedAt(polys, tau))));
    }
}

// Degree bounds below 2, where CPolynomial is undefined and pack() writes f itself.
void testPackDegenerate() {
    Random random(4);
    const Srs &srs = testSrs();
    const FrElement tau = testTau();
    const FrElement a = random.element(), b = random.element();
    struct Case {
        std::vector<Column> coefs;
        Column f;
    };
    const std::vector<Case> cases = {
        {{{a, E.fr.zero(), E.fr.zero()}}, {a}},                  // k = 1, a constant
        {{{a, b, E.fr.zero(), E.fr.zero()}}, {a, b}},            // k = 1, linear
        {{Column(4)}, {E.fr.zero()}},                            // k = 1, zero
        {{{a, E.fr.zero()}, {b}}, {a, b}},                       // k = 2, two constants
        {{Column(8), Column(8)}, {E.fr.zero(), E.fr.zero()}},    // k = 2, two zeros
        {{{a}, {E.fr.zero(), b}}, {a, E.fr.zero(), E.fr.zero(), b}}, // k = 2, degree bound 3: CPolynomial
    };
    for (const Case &test : cases) {
        std::vector<std::unique_ptr<Poly>> owned;
        for (const Column &coefs : test.coefs) {
            owned.push_back(polynomial(coefs));
        }
        const std::vector<Poly *> polys = pointers(owned);
        uint64_t nCoefs = 0;
        const Column f = packed(polys, nCoefs, random);
        assert(nCoefs == test.f.size());
        assert(identical(f.data(), test.f.data(), nCoefs));
        assert(samePoint(srs.commit(f.data(), nCoefs), g1Times(packedAt(polys, tau))));
    }
}

void testPackRefusesArguments() {
    Random random(5);
    std::unique_ptr<Poly> p = polynomial(random.column(5));
    Poly empty(E, 0);
    Poly *one[1] = {p.get()};
    Poly *three[3] = {p.get(), p.get(), p.get()};
    Poly *withNull[2] = {p.get(), nullptr};
    Poly *withEmpty[2] = {p.get(), &empty};
    Column buffer = random.column(16);
    const Column before = buffer;

    expectInvalid([&] { PilFflonk::pack(one, 0, buffer.data(), 16); }, "pack: k = 0: no polynomials");
    expectInvalid([&] { PilFflonk::pack(one, uint64_t(INT_MAX) + 1, buffer.data(), 16); }, "exceeds INT_MAX");
    expectInvalid([&] { PilFflonk::pack(nullptr, 1, buffer.data(), 16); }, "pack: polys is null");
    expectInvalid([&] { PilFflonk::pack(withNull, 2, buffer.data(), 16); }, "pack: polys[1] is null");
    expectInvalid([&] { PilFflonk::pack(withEmpty, 2, buffer.data(), 16); }, "pack: polys[1] has no coefficients");
    expectInvalid([&] { PilFflonk::pack(one, 1, nullptr, 16); }, "pack: packed is null");
    expectInvalid([&] { PilFflonk::pack(three, 3, buffer.data(), 15); },
                  "a buffer of 15 elements is shorter than the 16 that k = 3 polynomials of up to 5 coefficients need");
    assert(identical(buffer.data(), before.data(), buffer.size()));
    assert(PilFflonk::pack(three, 3, buffer.data(), 16) == 15);
}

// The evaluations on H, in natural order, of the polynomial with these coefficients.
Column evaluationsOf(const Column &coefs, uint64_t nBits) {
    const uint64_t N = uint64_t(1) << nBits;
    FFT<Engine::Fr> fft(std::max<uint64_t>(N, 2));
    const FrElement w = nBits == 0 ? E.fr.one() : fft.root(nBits, 1);
    Column evals(N);
    const std::unique_ptr<Poly> p = polynomial(coefs);
    FrElement x = E.fr.one();
    for (uint64_t i = 0; i < N; ++i) {
        evals[i] = p->evaluate(x);
        E.fr.mul(x, x, w);
    }
    return evals;
}

// What pilfflonk_commit_fixed does, one step at a time: Lde's INTT, the interleaving written out
// here, and the MSM. Also [f(τ)]₁ from the INTT's polynomials.
struct ByHand {
    G1Point commitment;
    G1Point atTau;
};

ByHand commitByHand(const Lde &lde, std::vector<Column> columns) {
    const Srs &srs = testSrs();
    const uint64_t N = lde.domainSize();
    std::vector<Column> coefs(columns.size(), Column(N));
    std::vector<FrElement *> in, out;
    for (size_t j = 0; j < columns.size(); ++j) {
        in.push_back(columns[j].data());
        out.push_back(coefs[j].data());
    }
    const std::vector<std::unique_ptr<Poly>> polys = lde.intt(in.data(), out.data(), columns.size());
    const Column f = interleave(pointers(polys));
    return {srs.commit(f.data(), f.size()), g1Times(packedAt(pointers(polys), testTau()))};
}

G1Point commitFixed(const Lde &lde, std::vector<Column> columns) {
    std::vector<FrElement *> in;
    for (Column &c : columns) {
        in.push_back(c.data());
    }
    const std::vector<Column> before = columns;
    const G1Point commitment = PilFflonk::commitFixed(testSrs(), lde, in.data(), columns.size());
    for (size_t j = 0; j < columns.size(); ++j) {
        assert(identical(columns[j].data(), before[j].data(), columns[j].size()));
    }
    return commitment;
}

void testCommitFixed() {
    Random random(6);
    for (uint64_t nBits : {0, 1, 2, 4, 6}) {
        const Lde lde(nBits, nBits);
        const uint64_t N = lde.domainSize();
        for (uint64_t k : PACKINGS) {
            std::vector<Column> columns;
            for (uint64_t j = 0; j < k; ++j) {
                columns.push_back(random.column(N));
            }
            const G1Point commitment = commitFixed(lde, columns);
            const ByHand expected = commitByHand(lde, columns);
            assert(samePoint(commitment, expected.commitment));
            assert(samePoint(commitment, expected.atTau));
        }
    }
}

// Columns whose f has a degree bound that is a power of two, or below 2, or is zero.
void testCommitFixedSpecialColumns() {
    Random random(7);
    const Srs &srs = testSrs();
    const FrElement tau = testTau();
    const Lde lde(4, 4);
    Column x4(5);
    x4[4] = E.fr.one();
    const Column evalsX4 = evaluationsOf(x4, 4);
    const FrElement a = random.element(), b = random.element(), c = random.element();
    const Column constantA(16, a), constantB(16, b), zero(16);
    const Column linear = evaluationsOf({a, b}, 4);

    // X^4: degree 4, and 8 when packed with a zero column.
    assert(samePoint(commitFixed(lde, {evalsX4}), srs.g1(4)));
    assert(samePoint(commitFixed(lde, {evalsX4, zero}), srs.g1(8)));
    assert(samePoint(commitFixed(lde, {zero, evalsX4}), srs.g1(9)));
    // Constants and a linear column: CPolynomial's degree bound is below 2.
    assert(samePoint(commitFixed(lde, {constantA}), g1Times(a)));
    assert(samePoint(commitFixed(lde, {constantA, constantB}), g1Times(E.fr.add(a, E.fr.mul(b, tau)))));
    assert(samePoint(commitFixed(lde, {linear}), g1Times(E.fr.add(a, E.fr.mul(b, tau)))));
    // And past it: a + c·X + b·X^2 + ... for k = 3.
    const Column constantC(16, c);
    const FrElement expected = E.fr.add(E.fr.add(a, E.fr.mul(c, tau)), E.fr.mul(b, E.fr.mul(tau, tau)));
    assert(samePoint(commitFixed(lde, {constantA, constantC, constantB}), g1Times(expected)));
    // Nothing: the point at infinity.
    G1Point none = commitFixed(lde, {zero, zero, zero});
    assert(E.g1.isZero(none));
}

void testCommitFixedRefusesArguments() {
    Random random(8);
    const Srs &srs = testSrs();
    const Lde lde(4, 4);
    Column column = random.column(16);
    FrElement *columns[13];
    for (FrElement *&c : columns) {
        c = column.data();
    }
    FrElement *withNull[2] = {column.data(), nullptr};
    expectInvalid([&] { PilFflonk::commitFixed(srs, lde, columns, 0); }, "commitFixed: k = 0: f packs no columns");
    // The SRS holds 1024 powers: 16 columns of 64 fit, 17 do not (refused before evals is read).
    const Lde big(6, 6);
    expectInvalid([&] { PilFflonk::commitFixed(srs, big, columns, 17); },
                  "commitFixed: f's k·N = 17·64 coefficients exceed the 1024 powers [τ^i]₁ of the SRS");
    expectInvalid([&] { PilFflonk::commitFixed(srs, lde, nullptr, 1); }, "Lde::intt: evals is null");
    expectInvalid([&] { PilFflonk::commitFixed(srs, lde, withNull, 2); }, "Lde::intt: evals[1] is null");
}

// The C API: canonical little-endian scalars in, x‖y out.
void encodeColumns(const std::vector<Column> &columns, std::vector<uint8_t> &bytes) {
    bytes.clear();
    for (const Column &column : columns) {
        for (const FrElement &e : column) {
            FrElement canonical;
            E.fr.fromMontgomery(canonical, e);
            const uint8_t *limbs = reinterpret_cast<const uint8_t *>(canonical.v);
            bytes.insert(bytes.end(), limbs, limbs + 32);
        }
    }
}

G1PointAffine decodeG1(const uint8_t bytes[64]) {
    G1PointAffine p;
    E.f1.fromRprLE(p.x, bytes, 32);
    E.f1.fromRprLE(p.y, bytes + 32, 32);
    return p;
}

void expectStatus(int status, int expected, const char *text) {
    if (status != expected || pilfflonk_last_status() != expected ||
        std::strstr(pilfflonk_last_error(), text) == nullptr) {
        std::fprintf(stderr, "status %d (last %d), expected %d: %s\n", status, pilfflonk_last_status(), expected,
                     pilfflonk_last_error());
        assert(!"unexpected status");
    }
}

void testApiCommitFixed() {
    Random random(9);
    TestDir dir;
    const std::string ptau = dir.file("api.ptau");
    const std::string srsPath = dir.file("api.srs.bin");
    writeTestPtau(ptau, N_G1);
    assert(pilfflonk_srs_from_ptau(ptau.c_str(), 256, srsPath.c_str()) == PILFFLONK_OK);
    void *srs = pilfflonk_srs_load(srsPath.c_str());
    assert(srs != nullptr);

    std::vector<uint8_t> bytes;
    uint8_t out[64];
    for (uint64_t nBits : {0, 2, 4}) {
        const Lde lde(nBits, nBits);
        for (uint64_t k : PACKINGS) {
            std::vector<Column> columns;
            for (uint64_t j = 0; j < k; ++j) {
                columns.push_back(random.column(lde.domainSize()));
            }
            encodeColumns(columns, bytes);
            assert(pilfflonk_commit_fixed(srs, nBits, k, bytes.data(), out) == PILFFLONK_OK);
            assert(pilfflonk_last_status() == PILFFLONK_OK && pilfflonk_last_error()[0] == '\0');
            assert(samePoint(commitByHand(lde, columns).commitment, decodeG1(out)));
        }
    }

    // The point at infinity is (0, 0).
    const std::vector<Column> zeros(2, Column(16));
    encodeColumns(zeros, bytes);
    std::memset(out, 0xff, sizeof(out));
    assert(pilfflonk_commit_fixed(srs, 4, 2, bytes.data(), out) == PILFFLONK_OK);
    const uint8_t infinity[64] = {};
    assert(std::memcmp(out, infinity, sizeof(out)) == 0);
    // The generator: a constant column of ones, whose f is 1.
    encodeColumns({Column(8, E.fr.one())}, bytes);
    assert(pilfflonk_commit_fixed(srs, 3, 1, bytes.data(), out) == PILFFLONK_OK);
    const Bytes32 one("0000000000000000000000000000000000000000000000000000000000000001");
    const Bytes32 two("0000000000000000000000000000000000000000000000000000000000000002");
    assert(std::memcmp(out, one.bytes, 32) == 0 && std::memcmp(out + 32, two.bytes, 32) == 0);

    // Refusals leave out_g1 as it was.
    encodeColumns({random.column(16), random.column(16), random.column(16)}, bytes);
    uint8_t untouched[64];
    std::memset(out, 0xab, sizeof(out));
    std::memcpy(untouched, out, sizeof(out));
    expectStatus(pilfflonk_commit_fixed(nullptr, 4, 3, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "pilfflonk_commit_fixed: srs is NULL");
    expectStatus(pilfflonk_commit_fixed(srs, 4, 3, nullptr, out), PILFFLONK_ERR_INVALID_ARGUMENT, "evals is NULL");
    expectStatus(pilfflonk_commit_fixed(srs, 4, 3, bytes.data(), nullptr), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "out_g1 is NULL");
    expectStatus(pilfflonk_commit_fixed(srs, 4, 0, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "k = 0: f packs no columns");
    expectStatus(pilfflonk_commit_fixed(srs, 29, 1, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "n_bits = 29 exceeds 28");
    expectStatus(pilfflonk_commit_fixed(srs, UINT64_MAX, 1, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "exceeds 28");
    // The SRS holds 256 powers: 16 columns of 16 fit, 17 do not; nor does k = 2^64 - 1.
    expectStatus(pilfflonk_commit_fixed(srs, 4, 17, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "f's k·N = 17·16 coefficients exceed the 256 powers [τ^i]₁ of the SRS");
    expectStatus(pilfflonk_commit_fixed(srs, 4, UINT64_MAX, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "coefficients exceed the 256 powers");
    expectStatus(pilfflonk_commit_fixed(srs, 9, 1, bytes.data(), out), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "f's k·N = 1·512 coefficients exceed the 256 powers");
    // Scalar 37 is row 5 of column 2.
    const Bytes32 r(R_HEX);
    std::memcpy(bytes.data() + 37 * 32, r.bytes, 32);
    expectStatus(pilfflonk_commit_fixed(srs, 4, 3, bytes.data(), out), PILFFLONK_ERR_NON_CANONICAL,
                 "scalar 37 (column 2, row 5) is not below r");
    std::memset(bytes.data() + 3 * 32, 0xff, 32);
    expectStatus(pilfflonk_commit_fixed(srs, 4, 3, bytes.data(), out), PILFFLONK_ERR_NON_CANONICAL,
                 "scalar 3 (column 0, row 3) is not below r");
    assert(std::memcmp(out, untouched, sizeof(out)) == 0);

    pilfflonk_srs_free(srs);
}

// The GPU (pilfflonk/docs/performance.md#why-the-proof-is-the-same), where there is one: the MSM of a
// key on the GPU (GpuKey::commit, on its copy of the SRS's powers) gives ffiasm's commitments, in
// affine coordinates byte for byte, for lengths around sppark's warp of 32 points and its window for
// small MSMs (192 points), up to the whole SRS, and for scalars of every size (0, 1, r − 1); zeros
// commit to the point at infinity on both. And repeated scalars, which make sppark's Pippenger slow
// and which the shift of GpuKey::commit spreads out: all equal, as L1's coefficients; every other
// one; −ρ, whose shifted scalars are all zero, so that the GPU's MSM of them is the point at infinity
// and the commitment is the shift's negated; and −ρ + 1, all equal once shifted. Each at lengths the
// shift sums separately, and again at one it has summed.
void testTheGpuCommitsAsTheCpu() {
#ifdef __USE_CUDA__
    if (!gpuUnderTest("commitments on the GPU")) {
        return;
    }
    const Srs &cpu = testSrs();
    PilFflonk::GpuKey key(cpu);
    assert(key.nPowers() == N_G1 && key.holds(cpu));
    const PilFflonk::DeviceBuffer coefs(N_G1 * sizeof(FrElement)), work(N_G1 * sizeof(FrElement)),
        zero(sizeof(uint64_t));
    const uint64_t offset = 0;
    gpu_plonk_memcpy_h2d(zero.data(), &offset, sizeof(offset));

    auto affine = [](G1Point p) {
        G1PointAffine a;
        E.g1.copy(a, p);
        return std::vector<uint8_t>(reinterpret_cast<uint8_t *>(&a), reinterpret_cast<uint8_t *>(&a) + sizeof(a));
    };
    auto onGpu = [&](const Column &c, uint64_t n) {
        gpu_plonk_memcpy_h2d(coefs.data(), c.data(), n * sizeof(FrElement));
        key.addShiftSum(n, work.data());
        return key.commit(coefs.data(), reinterpret_cast<const uint64_t *>(zero.data()), 1, n, n, work.data());
    };
    auto sameCommitment = [&](const Column &c, uint64_t n) {
        return affine(cpu.commit(c.data(), n)) == affine(onGpu(c, n));
    };

    Random random(9);
    for (uint64_t n : {1, 2, 3, 31, 32, 33, 63, 64, 65, 191, 192, 193, 257, 511, 1000, 1023, 1024}) {
        assert(sameCommitment(random.column(n), n));
    }
    const FrElement minusOne = E.fr.negOne();
    for (const Column &special : {Column(64, minusOne), Column(64, E.fr.one()), Column{E.fr.zero(), minusOne}}) {
        assert(sameCommitment(special, special.size()));
    }
    const Column zeros(100);
    G1Point zero100 = onGpu(zeros, zeros.size());
    assert(E.g1.isZero(zero100));
    assert(sameCommitment(zeros, zeros.size()));

    Column rho(N_G1);
    rho[0] = PilFflonk::msmShiftRatio();
    for (uint64_t i = 1; i < N_G1; ++i) {
        E.fr.mul(rho[i], rho[i - 1], rho[0]);
    }
    FrElement inverseN;
    E.fr.inv(inverseN, E.fr.set(64));
    Column allEqual(N_G1, inverseN), everyOther = random.column(N_G1), minusRho(N_G1), minusRhoPlusOne(N_G1);
    for (uint64_t i = 0; i < N_G1; ++i) {
        if (i % 2 == 0) {
            everyOther[i] = inverseN;
        }
        E.fr.neg(minusRho[i], rho[i]);
        E.fr.add(minusRhoPlusOne[i], minusRho[i], E.fr.one());
    }
    for (const Column *repeated : {&allEqual, &everyOther, &minusRho, &minusRhoPlusOne}) {
        for (uint64_t n : {uint64_t(1), uint64_t(33), uint64_t(193), uint64_t(500), N_G1, uint64_t(33)}) {
            assert(sameCommitment(*repeated, n));
        }
    }
    G1Point minusShift = onGpu(minusRho, N_G1);
    assert(!E.g1.isZero(minusShift));
#else
    // A library built without the GPU: nothing to compare (pilfflonk_gpu_test.cpp tests the refusal).
    assert(!PilFflonk::gpuAvailable());
#endif
}

// The shift of the GPU's MSMs (GpuKey::commit), which needs no GPU: h's order is not a power of two,
// so ρ_i = h^(i+1) is no root of unity of a domain.
void testTheGpuShift() {
#ifdef __USE_CUDA__
    assert(!E.fr.eq(PilFflonk::power(PilFflonk::msmShiftRatio(), uint64_t(1) << 28), E.fr.one()));
#endif
}

} // namespace

void runCommitTests() {
    testCommitIsTheEvaluationAtTau();
    testPackedBufferLength();
    testPackInterleaves();
    testPackDegreeAPowerOfTwo();
    testPackDegenerate();
    testPackRefusesArguments();
    testCommitFixed();
    testCommitFixedSpecialColumns();
    testCommitFixedRefusesArguments();
    testApiCommitFixed();
    testTheGpuCommitsAsTheCpu();
    testTheGpuShift();
}

} // namespace PilFflonkTest
