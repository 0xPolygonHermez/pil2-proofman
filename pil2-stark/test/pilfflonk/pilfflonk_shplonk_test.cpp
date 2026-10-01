// Tests for PilFflonk::ShplonkProver, with a test SRS of known τ and no pairing:
// - the roots are the x with x^k = ξ·ω_N^s (negative s too), derived here independently, and the
//   evaluations are p_j(ξ·ω_N^s);
// - r_i agrees with f_i on T_i, each f_i - r_i is divisible by Z_{T_i}, and W and W' are what the
//   prover's formulas (pilfflonk/docs/protocol.md#pairing-check) give when computed literally here
//   (rapidsnark's Euclidean division, the zerofiers as polynomials), with the expected degrees;
// - the verifier's identity F - E - J + y·W' = τ·W', both on the values at τ and on the G1
//   commitments, with α_S and y replayed from the transcript as a verifier would
//   (pilfflonk/docs/protocol.md#transcript);
// - tampering an evaluation, W, W' or a commitment breaks it; bad arguments throw; the same
//   inputs give the same proof on any number of threads.
//
// With PILFFLONK_SHPLONK_FIXTURES=<dir> in the environment, every opening checked here is also
// written to <dir>/shplonk_<name>.json for the JS verifier (pilfflonk/docs/README.md#tests); the
// format is at writeFixture().
#include "pilfflonk_test.hpp"

#include <gmp.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "alt_bn128.hpp"
#include "fft.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_shplonk_prover.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_test_ptau.hpp"
#include "pilfflonk_transcript.hpp"
#include "polynomial.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;
using G2PointAffine = Engine::G2PointAffine;
using Poly = Polynomial<Engine>;
using Column = std::vector<FrElement>;
using PilFflonk::ShplonkOpening;
using PilFflonk::ShplonkPolynomial;
using PilFflonk::ShplonkProof;
using PilFflonk::ShplonkProver;
using PilFflonk::Srs;
using PilFflonk::Transcript;
using Evaluations = ShplonkProver::Evaluations;

Engine &E = Engine::engine;

// The powers of the test SRS: enough for the longest f, 12·(16 + 4 + 1) coefficients.
constexpr uint64_t N_G1 = 1024;

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

private:
    std::mt19937_64 generator;
};

// The SRS every test commits with: the first N_G1 powers of a test ptau.
const Srs &testSrs() {
    static const Srs srs = [] {
        TestDir dir;
        const std::string ptau = dir.file("shplonk.ptau");
        writeTestPtau(ptau, N_G1);
        return Srs::fromPtau(ptau, N_G1);
    }();
    return srs;
}

bool equal(const FrElement &a, const FrElement &b) {
    return E.fr.eq(a, b);
}

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

// base^s, the inverse of base^|s| for s < 0.
FrElement signedPower(const FrElement &base, int64_t s) {
    if (s >= 0) {
        return power(base, static_cast<uint64_t>(s));
    }
    FrElement inverse;
    E.fr.inv(inverse, power(base, static_cast<uint64_t>(-s)));
    return inverse;
}

FrElement inverse(const FrElement &a) {
    FrElement r;
    E.fr.inv(r, a);
    return r;
}

// 5^((r-1)/n), from r written out in the test: independent of the prover's, which starts from
// ffiasm's r - 1.
FrElement rootOfUnity(uint64_t n) {
    mpz_t e;
    mpz_init(e);
    const int parsed = mpz_set_str(e, R_HEX, 16);
    assert(parsed == 0);
    mpz_sub_ui(e, e, 1);
    assert(mpz_divisible_ui_p(e, n));
    mpz_divexact_ui(e, e, n);
    const FrElement w = power(fromUI(5), e);
    mpz_clear(e);
    return w;
}

uint64_t lcm(const std::vector<uint64_t> &ks) {
    uint64_t l = 1;
    for (uint64_t k : ks) {
        l = std::lcm(l, k);
    }
    return l;
}

// Scalar multiplication x·p through ffiasm's mulByScalar on the canonical value of x.
G1Point times(G1Point p, const FrElement &x) {
    FrElement canonical;
    E.fr.fromMontgomery(canonical, x);
    G1Point r;
    E.g1.mulByScalar(r, p, reinterpret_cast<uint8_t *>(canonical.v), sizeof(canonical.v));
    return r;
}

G1Point g1Times(const FrElement &x) {
    G1Point g;
    E.g1.copy(g, E.g1.oneAffine());
    return times(g, x);
}

G1Point add(G1Point a, G1Point b) {
    G1Point r;
    E.g1.add(r, a, b);
    return r;
}

G1Point sub(G1Point a, G1Point b) {
    G1Point r;
    E.g1.sub(r, a, b);
    return r;
}

bool samePoint(G1Point a, G1Point b) {
    return E.g1.eq(a, b);
}

// A polynomial owning `length` coefficients, the first of them `coefs`, with its degree fixed.
std::unique_ptr<Poly> polynomial(const FrElement *coefs, uint64_t n, uint64_t length) {
    assert(n <= length && length > 0);
    std::unique_ptr<Poly> p(new Poly(E, length));
    for (uint64_t i = 0; i < n; ++i) {
        p->coef[i] = coefs[i];
    }
    p->fixDegree();
    return p;
}

std::unique_ptr<Poly> copyOf(const Poly &p, uint64_t length) {
    assert(p.getDegree() < length);
    return polynomial(p.coef, p.getDegree() + 1, length);
}

bool samePolynomial(const Poly &a, const Poly &b) {
    if (a.getDegree() != b.getDegree()) {
        return false;
    }
    for (uint64_t i = 0; i <= a.getDegree(); ++i) {
        if (!equal(a.coef[i], b.coef[i])) {
            return false;
        }
    }
    return true;
}

// Euclidean division by rapidsnark's divBy: a := a / b, and whether the remainder is zero.
bool dividesExactly(Poly &a, Poly &b) {
    const std::unique_ptr<Poly> remainder(a.divBy(b));
    return remainder->getDegree() == 0 && E.fr.isZero(remainder->coef[0]);
}

template <typename Exception, typename Call>
void expectThrows(Call call, const char *message) {
    try {
        call();
    } catch (const Exception &e) {
        if (std::strstr(e.what(), message) == nullptr) {
            std::fprintf(stderr, "unexpected message: %s\n", e.what());
            assert(!"the exception does not say what was expected");
        }
        return;
    }
    assert(!"expected an exception");
}

// One f of an opening, as the verifier knows it.
struct Shape {
    uint64_t k;
    std::vector<int64_t> offsets;
};

struct Case {
    std::string name;
    uint64_t nBits;
    std::vector<Shape> shapes;
    // Every component of f_i for i in `constants` has a single coefficient: f_i then has fewer
    // coefficients than roots when |T_i| > k, so that f_i = r_i and it adds nothing to W.
    std::vector<uint64_t> constants = {};
    // f_i for i in `lowDegree` has p_0 of degree 1 and the other components constant: f_i has degree
    // k, and with one offset f_i − r_i too, which divideExactly divides by X^k − ξ·ω^s in the range
    // k <= deg < 2k − 1 where rapidsnark's divByMonic writes below its buffer.
    std::vector<uint64_t> lowDegree = {};
};

// What the verifier gets from the proof and the key, and the transcript it replays, the protocol's
// in miniature (pilfflonk/docs/protocol.md#transcript): absorb `seed` (standing in for the digest),
// then every [f_i] in order; squeeze xiSeed; absorb the evaluations, f by f, offset-major, p_0 first
// within an offset; then the opening: squeeze α_S, absorb [W], squeeze y.
struct Opened {
    uint64_t nBits;
    uint64_t powerW;
    std::vector<Shape> shapes;
    FrElement seed;
    std::vector<G1Point> commitments;
    Evaluations evaluations;
    G1Point w;
    G1Point wp;
};

struct Challenges {
    FrElement xiSeed;
    FrElement alpha;
    FrElement y;
};

std::vector<FrElement> flatten(const Evaluations &evaluations) {
    std::vector<FrElement> flat;
    for (const Column &e : evaluations) {
        flat.insert(flat.end(), e.begin(), e.end());
    }
    return flat;
}

// The transcript up to xiSeed.
Transcript transcriptToXiSeed(const FrElement &seed, const std::vector<G1Point> &commitments) {
    Transcript t;
    t.absorb(std::vector<FrElement>{seed});
    t.absorb(commitments);
    return t;
}

Challenges replay(const Opened &o) {
    Challenges c;
    Transcript t = transcriptToXiSeed(o.seed, o.commitments);
    c.xiSeed = t.squeeze();
    t.absorb(flatten(o.evaluations));
    c.alpha = t.squeeze();
    t.absorb(std::vector<G1Point>{o.w});
    c.y = t.squeeze();
    return c;
}

// T_i (pilfflonk/docs/protocol.md#roots), offset-major: x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j.
std::vector<Column> rootsOf(uint64_t nBits, uint64_t powerW, const FrElement &xiSeed,
                            const std::vector<Shape> &shapes) {
    const uint64_t N = uint64_t(1) << nBits;
    std::vector<Column> roots;
    for (const Shape &f : shapes) {
        const FrElement wk = rootOfUnity(f.k);
        const FrElement omega = rootOfUnity(f.k * N);
        const FrElement seed = power(xiSeed, powerW / f.k);
        Column t;
        for (int64_t s : f.offsets) {
            for (uint64_t j = 0; j < f.k; ++j) {
                t.push_back(E.fr.mul(E.fr.mul(seed, signedPower(omega, s)), power(wk, j)));
            }
        }
        roots.push_back(t);
    }
    return roots;
}

// Z_{T_i}(y) = Π_{x in T_i} (y - x).
FrElement zerofierAt(const Column &roots, const FrElement &y) {
    FrElement z = E.fr.one();
    for (const FrElement &x : roots) {
        E.fr.mul(z, z, E.fr.sub(y, x));
    }
    return z;
}

// r_i(y) as the verifier computes it (pilfflonk/docs/protocol.md#pairing-check):
// f_i(x) = Σ_j p_j(ξ·ω^s)·x^j on every root x, then the Lagrange basis of T_i at y.
FrElement interpolantAt(const Column &roots, const Column &evaluations, uint64_t k, const FrElement &y) {
    const uint64_t n = roots.size();
    Column values(n);
    for (uint64_t m = 0; m < n; ++m) {
        const uint64_t offset = m / k;
        FrElement value = E.fr.zero();
        FrElement xPower = E.fr.one();
        for (uint64_t j = 0; j < k; ++j) {
            E.fr.add(value, value, E.fr.mul(evaluations[offset * k + j], xPower));
            E.fr.mul(xPower, xPower, roots[m]);
        }
        values[m] = value;
    }
    FrElement result = E.fr.zero();
    for (uint64_t m = 0; m < n; ++m) {
        FrElement numerator = E.fr.one();
        FrElement denominator = E.fr.one();
        for (uint64_t l = 0; l < n; ++l) {
            if (l != m) {
                E.fr.mul(numerator, numerator, E.fr.sub(y, roots[l]));
                E.fr.mul(denominator, denominator, E.fr.sub(roots[m], roots[l]));
            }
        }
        E.fr.add(result, result, E.fr.mul(values[m], E.fr.mul(numerator, inverse(denominator))));
    }
    return result;
}

// The scalars of the pairing check (pilfflonk/docs/protocol.md#pairing-check): q_0 = Z_{T_0}(y),
// q_i = α^i·Z_{T_0}(y)/Z_{T_i}(y), and e = r_0(y) + Σ_{i>=1} q_i·r_i(y).
struct Quotients {
    Column q;
    FrElement e;
};

Quotients quotientsOf(const std::vector<Column> &roots, const Evaluations &evaluations,
                      const std::vector<Shape> &shapes, const FrElement &alpha, const FrElement &y) {
    Quotients result;
    const FrElement z0 = zerofierAt(roots[0], y);
    result.q.push_back(z0);
    result.e = interpolantAt(roots[0], evaluations[0], shapes[0].k, y);
    FrElement alphaPower = alpha;
    for (uint64_t i = 1; i < roots.size(); ++i) {
        const FrElement qi = E.fr.mul(alphaPower, E.fr.mul(z0, inverse(zerofierAt(roots[i], y))));
        result.q.push_back(qi);
        E.fr.add(result.e, result.e, E.fr.mul(qi, interpolantAt(roots[i], evaluations[i], shapes[i].k, y)));
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    return result;
}

// F - E - J + y·W' = τ·W' on the values at τ: fTau[i] = f_i(τ), wTau = W(τ), wpTau = W'(τ).
bool scalarIdentityHolds(const Opened &o, const Challenges &c, const Column &fTau, const FrElement &wTau,
                         const FrElement &wpTau) {
    const std::vector<Column> roots = rootsOf(o.nBits, o.powerW, c.xiSeed, o.shapes);
    const Quotients q = quotientsOf(roots, o.evaluations, o.shapes, c.alpha, c.y);
    FrElement F = fTau[0];
    for (uint64_t i = 1; i < fTau.size(); ++i) {
        E.fr.add(F, F, E.fr.mul(q.q[i], fTau[i]));
    }
    const FrElement J = E.fr.mul(q.q[0], wTau);
    const FrElement lhs = E.fr.add(E.fr.sub(E.fr.sub(F, q.e), J), E.fr.mul(c.y, wpTau));
    return equal(lhs, E.fr.mul(testTau(), wpTau));
}

// The same identity in G1, e(F - E - J + y·[W'], [1]₂) = e([W'], [τ]₂) with the pairing replaced
// by the known τ: F - E - J + y·[W'] = τ·[W'].
bool identityHolds(const Opened &o, const Challenges &c) {
    const std::vector<Column> roots = rootsOf(o.nBits, o.powerW, c.xiSeed, o.shapes);
    const Quotients q = quotientsOf(roots, o.evaluations, o.shapes, c.alpha, c.y);
    G1Point F = o.commitments[0];
    for (uint64_t i = 1; i < o.commitments.size(); ++i) {
        F = add(F, times(o.commitments[i], q.q[i]));
    }
    const G1Point J = times(o.w, q.q[0]);
    const G1Point lhs = add(sub(sub(F, g1Times(q.e)), J), times(o.wp, c.y));
    return samePoint(lhs, times(o.wp, testTau()));
}

// What a verifier does: replay the transcript, then check the identity.
bool verifies(const Opened &o) {
    return identityHolds(o, replay(o));
}

// Everything a checked opening leaves, for the negative tests and the fixtures.
struct Result {
    Opened opened;
    Challenges challenges;
    Column fTau;
    FrElement wTau;
    FrElement wpTau;
    std::vector<Column> points;
    FrElement xi;
};

// Components of f with k and O: k columns of up to N + |O| + 1 coefficients (a blinded column's
// bound, pilfflonk/docs/protocol.md#degrees), of assorted degrees: full, half, full minus one, and
// one all zero when k > 2.
std::vector<std::unique_ptr<Poly>> componentsOf(const Shape &f, uint64_t N, bool constant, bool lowDegree,
                                                Random &random) {
    const uint64_t length = N + f.offsets.size() + 1;
    std::vector<std::unique_ptr<Poly>> components;
    if (lowDegree) {
        for (uint64_t j = 0; j < f.k; ++j) {
            std::unique_ptr<Poly> p(new Poly(E, j == 0 ? 2 : 1));
            for (uint64_t i = 0; i < p->getLength(); ++i) {
                p->coef[i] = random.element();
            }
            p->fixDegree();
            components.push_back(std::move(p));
        }
        return components;
    }
    for (uint64_t j = 0; j < f.k; ++j) {
        std::unique_ptr<Poly> p(new Poly(E, constant ? 1 : length));
        const uint64_t degrees[] = {length - 1, length / 2, length - 2, length - 1};
        const bool zero = f.k > 2 && j == 2;
        if (!zero) {
            for (uint64_t i = 0; i <= (constant ? 0 : degrees[j % 4]); ++i) {
                p->coef[i] = random.element();
            }
        }
        p->fixDegree();
        components.push_back(std::move(p));
    }
    return components;
}

// f packed by pack(), independently of the prover.
std::unique_ptr<Poly> packedOf(const std::vector<std::unique_ptr<Poly>> &components, uint64_t minLength) {
    std::vector<Poly *> raw;
    uint64_t maxLength = 0;
    for (const std::unique_ptr<Poly> &p : components) {
        raw.push_back(p.get());
        maxLength = std::max(maxLength, p->getLength());
    }
    const uint64_t bufferLength = PilFflonk::packedBufferLength(raw.size(), maxLength);
    Column buffer(bufferLength);
    const uint64_t n = PilFflonk::pack(raw.data(), raw.size(), buffer.data(), bufferLength);
    return polynomial(buffer.data(), n, std::max(n, minLength));
}

// Builds the case's opening from `seed`, proves it and checks every step and the identity.
Result prove(const Case &c, uint64_t seed) {
    Random random(seed);
    const Srs &srs = testSrs();
    const uint64_t N = uint64_t(1) << c.nBits;
    const uint64_t nF = c.shapes.size();

    std::vector<uint64_t> ks;
    std::vector<std::vector<std::unique_ptr<Poly>>> components;
    std::vector<std::unique_ptr<Poly>> fs;
    Result result;
    Opened &o = result.opened;
    o.nBits = c.nBits;
    o.shapes = c.shapes;
    for (uint64_t i = 0; i < nF; ++i) {
        const Shape &f = c.shapes[i];
        ks.push_back(f.k);
        const bool constant = std::find(c.constants.begin(), c.constants.end(), i) != c.constants.end();
        const bool lowDegree = std::find(c.lowDegree.begin(), c.lowDegree.end(), i) != c.lowDegree.end();
        components.push_back(componentsOf(f, N, constant, lowDegree, random));
        fs.push_back(packedOf(components.back(), f.k * f.offsets.size()));
        o.commitments.push_back(srs.commit(fs.back()->coef, fs.back()->getDegree() + 1));
        // [f_i] = f_i(τ)·G, and one the transcript can absorb
        // (pilfflonk/docs/protocol.md#transcript).
        result.fTau.push_back(fs.back()->evaluate(testTau()));
        assert(samePoint(o.commitments.back(), g1Times(result.fTau.back())));
        uint8_t bytes[PilFflonk::G1_BYTES];
        G1Point decoded;
        PilFflonk::encodeG1(o.commitments.back(), bytes);
        assert(PilFflonk::decodeG1(bytes, decoded) == PilFflonk::AbsorbError::None);
    }
    o.powerW = lcm(ks);
    o.seed = random.element();

    Transcript transcript = transcriptToXiSeed(o.seed, o.commitments);
    const FrElement xiSeed = transcript.squeeze();
    ShplonkOpening opening;
    opening.nBits = c.nBits;
    opening.powerW = o.powerW;
    opening.xiSeed = xiSeed;
    for (uint64_t i = 0; i < nF; ++i) {
        ShplonkPolynomial f;
        for (const std::unique_ptr<Poly> &p : components[i]) {
            f.components.push_back(p.get());
        }
        f.offsets = c.shapes[i].offsets;
        opening.polynomials.push_back(f);
    }
    const ShplonkProver prover(opening);
    assert(prover.size() == nF);

    // ξ, the points ξ·ω_N^s with ω_N the generator of H that ffiasm's FFT uses, and the roots.
    const FrElement xi = power(xiSeed, o.powerW);
    assert(equal(prover.xi(), xi));
    result.xi = xi;
    FFT<Engine::Fr> fft(N);
    const FrElement omegaN = fft.root(c.nBits, 1);
    assert(equal(omegaN, rootOfUnity(N)));
    const std::vector<Column> roots = rootsOf(c.nBits, o.powerW, xiSeed, c.shapes);
    for (uint64_t i = 0; i < nF; ++i) {
        const Shape &f = c.shapes[i];
        assert(prover.k(i) == f.k);
        assert(prover.nCoefs(i) >= fs[i]->getDegree() + 1);
        assert(prover.points(i).size() == f.offsets.size());
        assert(prover.roots(i).size() == f.k * f.offsets.size());
        Column points;
        for (uint64_t m = 0; m < f.offsets.size(); ++m) {
            const FrElement point = E.fr.mul(xi, signedPower(omegaN, f.offsets[m]));
            assert(equal(prover.points(i)[m], point));
            points.push_back(point);
            for (uint64_t j = 0; j < f.k; ++j) {
                const FrElement &x = prover.roots(i)[m * f.k + j];
                assert(equal(x, roots[i][m * f.k + j]));
                assert(equal(power(x, f.k), point));
            }
        }
        result.points.push_back(points);
        // No root twice in T_i: offsets apart give different x^k, and w_k has order k.
        for (uint64_t a = 0; a < roots[i].size(); ++a) {
            for (uint64_t b = a + 1; b < roots[i].size(); ++b) {
                assert(!equal(roots[i][a], roots[i][b]));
            }
        }
        // Z_{T_i}(X) = Π_s (X^k - ξ·ω_N^s), which the prover divides by one factor at a time.
        const FrElement at = random.element();
        FrElement product = E.fr.one();
        for (const FrElement &point : points) {
            E.fr.mul(product, product, E.fr.sub(power(at, f.k), point));
        }
        assert(equal(zerofierAt(roots[i], at), product));
    }

    // The evaluations are p_j(ξ·ω_N^s), by Horner's rule.
    o.evaluations = prover.evaluations();
    assert(o.evaluations.size() == nF);
    for (uint64_t i = 0; i < nF; ++i) {
        const uint64_t k = c.shapes[i].k;
        assert(o.evaluations[i].size() == k * c.shapes[i].offsets.size());
        for (uint64_t m = 0; m < c.shapes[i].offsets.size(); ++m) {
            for (uint64_t j = 0; j < k; ++j) {
                assert(equal(o.evaluations[i][m * k + j], components[i][j]->evaluate(result.points[i][m])));
            }
        }
    }

    // r_i = f_i on T_i, of degree below |T_i|; f_i - r_i divisible by Z_{T_i}.
    const ShplonkProver::Interpolants r = prover.interpolants();
    assert(r.size() == nF);
    std::vector<std::unique_ptr<Poly>> quotients;
    for (uint64_t i = 0; i < nF; ++i) {
        const uint64_t nRoots = roots[i].size();
        assert(r[i]->getLength() <= nRoots && r[i]->getDegree() < nRoots);
        for (const FrElement &x : roots[i]) {
            assert(equal(r[i]->evaluate(x), fs[i]->evaluate(x)));
        }
        std::unique_ptr<Poly> term = copyOf(*fs[i], fs[i]->getLength());
        term->sub(*r[i]);
        Column t = roots[i];
        const std::unique_ptr<Poly> zerofier(Poly::zerofierPolynomial(t.data(), static_cast<uint32_t>(nRoots)));
        assert(zerofier->getDegree() == nRoots);
        assert(dividesExactly(*term, *zerofier));
        quotients.push_back(std::move(term));
    }

    // The opening.
    transcript.absorb(flatten(o.evaluations));
    const ShplonkProof proof = prover.open(srs, transcript);
    o.w = proof.w;
    o.wp = proof.wp;
    result.challenges = replay(o);
    assert(equal(result.challenges.xiSeed, xiSeed));
    assert(equal(result.challenges.alpha, proof.alpha));
    assert(equal(result.challenges.y, proof.y));
    const FrElement &alpha = proof.alpha;
    const FrElement &y = proof.y;

    // W = Σ_i α^i·(f_i - r_i)/Z_{T_i}, of degree max_i (deg f_i - |T_i|).
    const std::unique_ptr<Poly> W = prover.quotientW(r, alpha);
    uint64_t maxLength = 1;
    uint64_t wDegree = 0;
    uint64_t maxDegree = 0;
    for (uint64_t i = 0; i < nF; ++i) {
        maxLength = std::max({maxLength, fs[i]->getLength(), roots[i].size()});
        maxDegree = std::max(maxDegree, fs[i]->getDegree());
        if (fs[i]->getDegree() >= roots[i].size()) {
            wDegree = std::max(wDegree, fs[i]->getDegree() - roots[i].size());
        }
    }
    std::unique_ptr<Poly> expectedW(new Poly(E, maxLength));
    FrElement alphaPower = E.fr.one();
    for (uint64_t i = 0; i < nF; ++i) {
        std::unique_ptr<Poly> term = copyOf(*quotients[i], maxLength);
        term->mulScalar(alphaPower);
        expectedW->add(*term);
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    assert(samePolynomial(*W, *expectedW));
    assert(W->getDegree() == wDegree);
    assert(samePoint(proof.w, srs.commit(W->coef, W->getDegree() + 1)));
    result.wTau = W->evaluate(testTau());
    assert(samePoint(proof.w, g1Times(result.wTau)));

    // W' = L/(Z_{T∖T_0}(y)·(X - y)) with L = Σ_i α^i·Z_{T∖T_i}(y)·(f_i - r_i(y)) - Z_T(y)·W, the
    // zerofiers evaluated from rapidsnark's zerofier polynomials, Z_T with repetitions and
    // Z_{T∖T_i}(y) as the product of the others: no inverses but the last.
    const std::unique_ptr<Poly> Wp = prover.quotientWp(r, alpha, y, *W);
    Column zi;
    for (uint64_t i = 0; i < nF; ++i) {
        Column t = roots[i];
        const std::unique_ptr<Poly> zerofier(Poly::zerofierPolynomial(t.data(), static_cast<uint32_t>(t.size())));
        zi.push_back(zerofier->evaluate(y));
        assert(equal(zi.back(), zerofierAt(roots[i], y)));
    }
    std::unique_ptr<Poly> L = copyOf(*W, maxLength);
    FrElement minusZT = E.fr.one();
    for (const FrElement &z : zi) {
        E.fr.mul(minusZT, minusZT, z);
    }
    minusZT = E.fr.neg(minusZT);
    L->mulScalar(minusZT);
    alphaPower = E.fr.one();
    for (uint64_t i = 0; i < nF; ++i) {
        FrElement factor = alphaPower;
        for (uint64_t j = 0; j < nF; ++j) {
            if (j != i) {
                E.fr.mul(factor, factor, zi[j]);
            }
        }
        std::unique_ptr<Poly> term = copyOf(*fs[i], maxLength);
        FrElement ry = r[i]->evaluate(y);
        term->subScalar(ry);
        term->mulScalar(factor);
        L->add(*term);
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    FrElement zTMinusT0 = E.fr.one();
    for (uint64_t i = 1; i < nF; ++i) {
        E.fr.mul(zTMinusT0, zTMinusT0, zi[i]);
    }
    FrElement scale = inverse(zTMinusT0);
    L->mulScalar(scale);
    const FrElement xMinusY[2] = {E.fr.neg(y), E.fr.one()};
    const std::unique_ptr<Poly> divisor = polynomial(xMinusY, 2, 2);
    assert(dividesExactly(*L, *divisor));
    assert(samePolynomial(*Wp, *L));
    assert(Wp->getDegree() == (maxDegree > 0 ? maxDegree - 1 : 0));
    assert(samePoint(proof.wp, srs.commit(Wp->coef, Wp->getDegree() + 1)));
    result.wpTau = Wp->evaluate(testTau());
    assert(samePoint(proof.wp, g1Times(result.wpTau)));

    // The identity, on the values at τ and on the commitments, as the verifier replays it.
    assert(scalarIdentityHolds(o, result.challenges, result.fTau, result.wTau, result.wpTau));
    assert(identityHolds(o, result.challenges));
    assert(verifies(o));
    return result;
}

std::string decimal(const FrElement &e) {
    return E.fr.toString(e, 10);
}

nlohmann::json g1Json(G1Point p) {
    G1PointAffine a;
    E.g1.copy(a, p);
    return nlohmann::json::array({E.f1.toString(a.x, 10), E.f1.toString(a.y, 10)});
}

// An explicit array of arrays: nlohmann reads {{"a", "b"}, {"c", "d"}} as an object.
nlohmann::json g2Json(G2PointAffine p) {
    const nlohmann::json x = nlohmann::json::array({E.f1.toString(p.x.a, 10), E.f1.toString(p.x.b, 10)});
    const nlohmann::json y = nlohmann::json::array({E.f1.toString(p.y.a, 10), E.f1.toString(p.y.b, 10)});
    return nlohmann::json::array({x, y});
}

nlohmann::json scalarsJson(const Column &scalars) {
    nlohmann::json values = nlohmann::json::array();
    for (const FrElement &e : scalars) {
        values.push_back(decimal(e));
    }
    return values;
}

// Writes the opening to <dir>/shplonk_<name>.json, every number a decimal string, points affine:
//
// {
//   "name": "<name>", "curve": "bn128", "protocol": "pilfflonk-shplonk",
//   "nBits": n,                     N = 2^n
//   "powerW": w,                    the lcm of every k
//   "f": [                          in the global order: f_0 first
//     { "k": k, "offsets": [s, ...] (signed integers),
//       "commitment": ["x", "y"],   [f_i]₁
//       "points": ["ξ·ω_N^s", ...], one per offset (derived; for debugging)
//       "evaluations": [["p_0(ξ·ω_N^s)", ..., "p_{k-1}(ξ·ω_N^s)"], ...] one row per offset }, ...
//   ],
//   "xiSeed": "...", "xi": "...",   ξ = xiSeed^powerW
//   "alpha": "...", "y": "...",     α_S and y
//   "W": ["x", "y"], "Wp": ["x", "y"],
//   "X2": { "one": [[x.c0, x.c1], [y.c0, y.c1]], "tau": [...] },   [1]₂ and [τ]₂, c0 + c1·u
//   "tau": "...",                   the test SRS's τ: never known for a real one
//   "transcript": [                 every operation, in order, from a new Keccak256Transcript
//     {"op": "absorb", "kind": "fr", "values": ["..."]},                      the seed
//     {"op": "absorb", "kind": "g1", "values": [["x", "y"], ...]},           every [f_i]
//     {"op": "squeeze", "name": "xiSeed", "value": "..."},
//     {"op": "absorb", "kind": "fr", "values": [...]},   the evaluations: f by f, offset-major
//     {"op": "squeeze", "name": "alpha", "value": "..."},
//     {"op": "absorb", "kind": "g1", "values": [["x", "y"]]},                [W]₁
//     {"op": "squeeze", "name": "y", "value": "..."}
//   ]
// }
//
// A verifier accepts it iff e(F - E - J + y·[W'], [1]₂) = e([W'], [τ]₂)
// (pilfflonk/docs/protocol.md#pairing-check), with the roots x^k = ξ·ω_N^s of each f,
// f_i(x) = Σ_j p_j(ξ·ω_N^s)·x^j, and Z_T with repetitions.
void writeFixture(const std::string &dir, const std::string &name, const Result &result) {
    const Opened &o = result.opened;
    const Challenges &c = result.challenges;
    const Srs &srs = testSrs();
    nlohmann::json fixture;
    fixture["name"] = name;
    fixture["curve"] = "bn128";
    fixture["protocol"] = "pilfflonk-shplonk";
    fixture["nBits"] = o.nBits;
    fixture["powerW"] = o.powerW;
    nlohmann::json fs = nlohmann::json::array();
    for (uint64_t i = 0; i < o.shapes.size(); ++i) {
        const Shape &shape = o.shapes[i];
        nlohmann::json f;
        f["k"] = shape.k;
        f["offsets"] = shape.offsets;
        f["commitment"] = g1Json(o.commitments[i]);
        f["points"] = scalarsJson(result.points[i]);
        nlohmann::json evaluations = nlohmann::json::array();
        for (uint64_t m = 0; m < shape.offsets.size(); ++m) {
            evaluations.push_back(scalarsJson(Column(o.evaluations[i].begin() + m * shape.k,
                                                     o.evaluations[i].begin() + (m + 1) * shape.k)));
        }
        f["evaluations"] = evaluations;
        fs.push_back(f);
    }
    fixture["f"] = fs;
    fixture["xiSeed"] = decimal(c.xiSeed);
    fixture["xi"] = decimal(result.xi);
    fixture["alpha"] = decimal(c.alpha);
    fixture["y"] = decimal(c.y);
    fixture["W"] = g1Json(o.w);
    fixture["Wp"] = g1Json(o.wp);
    fixture["X2"] = {{"one", g2Json(srs.g2(0))}, {"tau", g2Json(srs.g2(1))}};
    fixture["tau"] = decimal(testTau());

    nlohmann::json commitments = nlohmann::json::array();
    for (const G1Point &p : o.commitments) {
        commitments.push_back(g1Json(p));
    }
    fixture["transcript"] = {
        {{"op", "absorb"}, {"kind", "fr"}, {"values", scalarsJson({o.seed})}},
        {{"op", "absorb"}, {"kind", "g1"}, {"values", commitments}},
        {{"op", "squeeze"}, {"name", "xiSeed"}, {"value", decimal(c.xiSeed)}},
        {{"op", "absorb"}, {"kind", "fr"}, {"values", scalarsJson(flatten(o.evaluations))}},
        {{"op", "squeeze"}, {"name", "alpha"}, {"value", decimal(c.alpha)}},
        {{"op", "absorb"}, {"kind", "g1"}, {"values", nlohmann::json::array({g1Json(o.w)})}},
        {{"op", "squeeze"}, {"name", "y"}, {"value", decimal(c.y)}},
    };

    const std::string path = dir + "/shplonk_" + name + ".json";
    std::ofstream out(path);
    out << fixture.dump(2) << "\n";
    out.close();
    assert(out.good());
    std::printf("pilfflonk_test: wrote %s\n", path.c_str());
}

const std::vector<uint64_t> KS = {1, 2, 3, 4, 6, 12};

std::vector<Shape> everyK(const std::vector<int64_t> &offsets) {
    std::vector<Shape> shapes;
    for (uint64_t k : KS) {
        shapes.push_back({k, offsets});
    }
    return shapes;
}

std::vector<Case> cases() {
    return {
        {"single", 4, {{1, {0}}}},
        {"everyk_0", 4, everyK({0})},
        {"everyk_01", 4, everyK({0, 1})},
        {"everyk_m1012", 4, everyK({-1, 0, 1, 2})},
        // Roots shared between f_i: Z_T counts them once per f_i.
        {"repeated",
         3,
         {{4, {0, 1}},
          {4, {0}},
          {3, {-1, 0, 1, 2}},
          {3, {-1, 0, 1, 2}},
          {1, {0}},
          {1, {0, 1}},
          {12, {0, 1}},
          {2, {-1, 2}},
          {6, {1}}}},
        // f_1 has fewer coefficients than roots: f_1 = r_1, so it adds nothing to W.
        {"short", 4, {{3, {0, 1}}, {2, {-1, 0, 1, 2}}, {1, {0, 1}}}, {1}},
        // N = 2: ω_N = -1, and s = -1 is the same row as s = 1.
        {"tiny", 1, {{2, {-1, 0}}, {1, {0}}, {4, {-1}}}},
        // The fixtures the JS tests need at the least: {0}, {0,1} and {-1,0,1,2} with k in {1, 3, 4}.
        {"k134_0", 4, {{1, {0}}, {3, {0}}, {4, {0}}}},
        {"k134_01", 4, {{1, {0, 1}}, {3, {0, 1}}, {4, {0, 1}}, {3, {0}}}},
        {"k134_m1012", 4, {{1, {-1, 0, 1, 2}}, {3, {-1, 0, 1, 2}}, {4, {-1, 0, 1, 2}}}},
        // f_i of degree k with one offset (see Case::lowDegree), for every k >= 2.
        {"lowdegree", 4, {{2, {0}}, {3, {0}}, {4, {0}}, {6, {0}}, {12, {1}}, {1, {0, 1}}}, {}, {0, 1, 2, 3, 4}},
    };
}

void testOpenings() {
    const char *dir = std::getenv("PILFFLONK_SHPLONK_FIXTURES");
    const std::vector<Case> all = cases();
    for (uint64_t c = 0; c < all.size(); ++c) {
        const Result result = prove(all[c], 1000 + c);
        if (dir != nullptr && dir[0] != '\0') {
            writeFixture(dir, all[c].name, result);
        }
    }
}

// Changing any evaluation, [W]₁, [W']₁ or [f_i]₁ breaks the identity: with the challenges the
// transcript gives for the changed proof, and with the original ones.
void testTamperingBreaksTheIdentity() {
    const Result result = prove(cases()[4], 2000);
    const Opened &o = result.opened;
    assert(verifies(o));

    const auto rejected = [&](const Opened &tampered) {
        return !verifies(tampered) && !identityHolds(tampered, result.challenges);
    };
    const G1Point g = g1Times(E.fr.one());
    for (uint64_t i = 0; i < o.evaluations.size(); ++i) {
        for (uint64_t m = 0; m < o.evaluations[i].size(); ++m) {
            Opened tampered = o;
            E.fr.add(tampered.evaluations[i][m], tampered.evaluations[i][m], E.fr.one());
            assert(rejected(tampered));
        }
        Opened tampered = o;
        tampered.commitments[i] = add(tampered.commitments[i], g);
        assert(rejected(tampered));
    }
    Opened tampered = o;
    tampered.w = add(tampered.w, g);
    assert(rejected(tampered));
    tampered = o;
    tampered.wp = add(tampered.wp, g);
    assert(rejected(tampered));

    // And on the values at τ.
    const Challenges &c = result.challenges;
    assert(scalarIdentityHolds(o, c, result.fTau, result.wTau, result.wpTau));
    const FrElement one = E.fr.one();
    assert(!scalarIdentityHolds(o, c, result.fTau, E.fr.add(result.wTau, one), result.wpTau));
    assert(!scalarIdentityHolds(o, c, result.fTau, result.wTau, E.fr.add(result.wpTau, one)));
    for (uint64_t i = 0; i < result.fTau.size(); ++i) {
        Column fTau = result.fTau;
        E.fr.add(fTau[i], fTau[i], one);
        assert(!scalarIdentityHolds(o, c, fTau, result.wTau, result.wpTau));
    }
    Opened wrongEvaluation = o;
    E.fr.add(wrongEvaluation.evaluations[2][5], wrongEvaluation.evaluations[2][5], one);
    assert(!scalarIdentityHolds(wrongEvaluation, c, result.fTau, result.wTau, result.wpTau));
}

// The same inputs give the same proof, bit for bit, whatever the number of threads.
void testDeterminism() {
    const std::vector<Case> all = cases();
    const Case &c = all[3];
    const Result first = prove(c, 3000);
    const Result second = prove(c, 3000);
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const Result serial = prove(c, 3000);
    omp_set_num_threads(threads);
    for (const Result *other : {&second, &serial}) {
        assert(samePoint(first.opened.w, other->opened.w));
        assert(samePoint(first.opened.wp, other->opened.wp));
        assert(equal(first.challenges.alpha, other->challenges.alpha));
        assert(equal(first.challenges.y, other->challenges.y));
    }
}

// A small valid opening to break one argument at a time: f_0 with k = 3 at {0, 1}, f_1 with k = 1
// at {-1}, N = 16.
struct SmallOpening {
    std::vector<std::unique_ptr<Poly>> owned;
    ShplonkOpening opening;

    SmallOpening() {
        Random random(4000);
        opening.nBits = 4;
        opening.powerW = 3;
        opening.xiSeed = random.element();
        const std::vector<Shape> shapes = {{3, {0, 1}}, {1, {-1}}};
        for (const Shape &shape : shapes) {
            ShplonkPolynomial f;
            for (uint64_t j = 0; j < shape.k; ++j) {
                std::unique_ptr<Poly> p(new Poly(E, 19));
                for (uint64_t i = 0; i < 19; ++i) {
                    p->coef[i] = random.element();
                }
                p->fixDegree();
                f.components.push_back(p.get());
                owned.push_back(std::move(p));
            }
            f.offsets = shape.offsets;
            opening.polynomials.push_back(f);
        }
    }
};

template <typename Change>
void expectRefused(Change change, const char *message) {
    SmallOpening fixture;
    change(fixture.opening, fixture.owned);
    expectThrows<std::invalid_argument>([&] { ShplonkProver prover(fixture.opening); }, message);
}

void testRefusesArguments() {
    using Owned = std::vector<std::unique_ptr<Poly>>;
    {
        SmallOpening fixture;
        const ShplonkProver prover(fixture.opening);
        assert(prover.size() == 2);
    }
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials.clear(); }, "ShplonkProver: no polynomials to open");
    expectRefused([](ShplonkOpening &o, Owned &) { o.nBits = 29; }, "nBits = 29 exceeds 28");
    expectRefused([](ShplonkOpening &o, Owned &) { o.nBits = UINT64_MAX; }, "exceeds 28");
    expectRefused([](ShplonkOpening &o, Owned &) { o.xiSeed = E.fr.zero(); }, "xiSeed is zero");
    expectRefused([](ShplonkOpening &o, Owned &) { o.powerW = 6; }, "powerW = 6 is not 3, the lcm of every k");
    expectRefused([](ShplonkOpening &o, Owned &) { o.powerW = 0; }, "powerW = 0 is not 3");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].components.clear(); }, "f_1 has no components");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[0].components[2] = nullptr; },
                  "f_0: component 2 is null");
    expectRefused(
        [](ShplonkOpening &o, Owned &owned) {
            owned.emplace_back(new Poly(E, 0));
            o.polynomials[1].components[0] = owned.back().get();
        },
        "f_1: component 0 has no coefficients");
    // k must divide r - 1 = 2^28·3^2·13·29·983·…: 5, 7 and 9·3 = 27 do not.
    for (uint64_t k : {5, 7, 27}) {
        expectRefused(
            [k](ShplonkOpening &o, Owned &) {
                o.polynomials[0].components.resize(k, o.polynomials[0].components[0]);
                o.powerW = k;
            },
            ("f_0: k = " + std::to_string(k) + " does not divide r - 1").c_str());
    }
    // kN within 2^28: k = 4 at N = 2^27 goes beyond it; k = 3 does not (it is odd).
    expectRefused(
        [](ShplonkOpening &o, Owned &) {
            o.nBits = 27;
            o.polynomials[0].components.resize(4, o.polynomials[0].components[0]);
            o.powerW = 4;
        },
        "f_0: kN = 4·2^27 goes beyond the 2-adicity 2^28 of r - 1");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].offsets.clear(); }, "f_1 has no offsets");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[0].offsets = {1, 1}; },
                  "f_0: two offsets are the same row modulo N = 16");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[0].offsets = {-1, 15}; },
                  "f_0: two offsets are the same row modulo N = 16");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].offsets = {16}; },
                  "f_1: offset 16 is not below N = 16 in absolute value");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].offsets = {-16}; }, "offset -16 is not below");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].offsets = {INT64_MIN}; },
                  "offset -9223372036854775808 is not below");
    expectRefused([](ShplonkOpening &o, Owned &) { o.polynomials[1].offsets = {INT64_MAX}; },
                  "offset 9223372036854775807 is not below");

    // open(): an f longer than the SRS, or an empty transcript, before the transcript is touched.
    {
        SmallOpening fixture;
        std::unique_ptr<Poly> tooLong(new Poly(E, N_G1 + 1));
        for (uint64_t i = 0; i <= N_G1; ++i) {
            tooLong->coef[i] = E.fr.one();
        }
        tooLong->fixDegree();
        fixture.opening.polynomials[1].components[0] = tooLong.get();
        const ShplonkProver prover(fixture.opening);
        Transcript t;
        t.absorb(std::vector<FrElement>{E.fr.one()});
        Transcript twin = t;
        expectThrows<std::invalid_argument>([&] { prover.open(testSrs(), t); },
                                            "ShplonkProver::open: f_1's 1025 coefficients exceed the 1024 powers");
        assert(equal(t.squeeze(), twin.squeeze()));
    }
    {
        SmallOpening fixture;
        const ShplonkProver prover(fixture.opening);
        Transcript empty;
        expectThrows<std::invalid_argument>([&] { prover.open(testSrs(), empty); },
                                            "ShplonkProver::open: the transcript is empty");
        assert(empty.empty());
    }
}

// The failures after the transcript has moved on: [W]₁ at infinity (every f_i = r_i, so W = 0),
// and the exact divisions refusing an interpolant that is not one.
void testFailuresAfterSqueezing() {
    {
        Random random(5000);
        std::vector<std::unique_ptr<Poly>> owned;
        ShplonkOpening opening;
        opening.nBits = 4;
        opening.powerW = 1;
        opening.xiSeed = random.element();
        for (uint64_t i = 0; i < 2; ++i) {
            // Two coefficients, two roots: f = r.
            std::unique_ptr<Poly> p(new Poly(E, 2));
            p->coef[0] = random.element();
            p->coef[1] = random.element();
            p->fixDegree();
            opening.polynomials.push_back({{p.get()}, {0, 1}});
            owned.push_back(std::move(p));
        }
        const ShplonkProver prover(opening);
        Transcript t;
        t.absorb(std::vector<FrElement>{E.fr.one()});
        expectThrows<std::runtime_error>([&] { prover.open(testSrs(), t); },
                                         "ShplonkProver::open: [W]₁ is the point at infinity");
    }
    {
        SmallOpening fixture;
        const ShplonkProver prover(fixture.opening);
        Random random(5001);
        const FrElement alpha = random.element();
        const FrElement y = random.element();
        ShplonkProver::Interpolants r = prover.interpolants();
        const std::unique_ptr<Poly> W = prover.quotientW(r, alpha);
        prover.quotientWp(r, alpha, y, *W);
        // r_0 off by one: f_0 - r_0 is not divisible by Z_{T_0}, and L(y) is not 0.
        FrElement one = E.fr.one();
        r[0]->addScalar(one);
        expectThrows<std::logic_error>([&] { prover.quotientW(r, alpha); },
                                       "ShplonkProver: f_0 - r_i is not divisible");
        expectThrows<std::logic_error>([&] { prover.quotientWp(r, alpha, y, *W); },
                                       "ShplonkProver: L is not divisible");
        ShplonkProver::Interpolants tooFew = prover.interpolants();
        tooFew.pop_back();
        expectThrows<std::invalid_argument>([&] { prover.quotientW(tooFew, alpha); }, "1 interpolants for 2");
        expectThrows<std::invalid_argument>([&] { prover.quotientWp(tooFew, alpha, y, *W); }, "1 interpolants for 2");
    }
}

// divideExactly against the definition: for X^m − β with m in {1, 2, 3, 4, 6}, a = q·(X^m − β) of
// every degree d up to 3m is divided back to q, whatever the length of its buffer; a + ρ, ρ of degree
// below m not zero, is refused. In particular the degrees m <= d < 2m − 1, where rapidsnark's
// divByMonic writes below its buffer (under ASan: heap-buffer-overflow), are divided here.
void testDivideExactly() {
    Random random(6001);
    for (uint64_t m : {1, 2, 3, 4, 6}) {
        const FrElement beta = random.element();
        for (uint64_t d = 0; d <= 3 * m; ++d) {
            for (uint64_t extra : {0, 3}) {
                // q of degree d − m (none if d < m: a is then 0, the only multiple of that degree).
                const uint64_t qLength = d >= m ? d - m + 1 : 0;
                Column q(qLength);
                for (FrElement &c : q) {
                    c = random.element();
                }
                if (qLength > 0 && E.fr.isZero(q.back())) {
                    q.back() = E.fr.one();
                }
                // a = q·X^m − β·q, in room for ρ below (m coefficients at least).
                std::unique_ptr<Poly> a(new Poly(E, std::max(d + 1, m) + extra));
                for (uint64_t j = 0; j < qLength; ++j) {
                    E.fr.add(a->coef[j + m], a->coef[j + m], q[j]);
                    E.fr.sub(a->coef[j], a->coef[j], E.fr.mul(beta, q[j]));
                }
                a->fixDegree();
                assert(qLength == 0 || a->getDegree() == d);

                std::unique_ptr<Poly> refused = copyOf(*a, a->getLength());
                PilFflonk::divideExactly(*a, m, beta, "a");
                for (uint64_t j = 0; j < a->getLength(); ++j) {
                    assert(equal(a->coef[j], j < qLength ? q[j] : E.fr.zero()));
                }
                assert(a->getDegree() == (qLength > 0 ? qLength - 1 : 0));

                // Not divisible: the remainder ρ is checked, whatever the degree.
                FrElement rho = random.element();
                E.fr.add(refused->coef[m - 1], refused->coef[m - 1], rho);
                refused->fixDegree();
                expectThrows<std::logic_error>([&] { PilFflonk::divideExactly(*refused, m, beta, "a + ρ"); },
                                               "ShplonkProver: a + ρ is not divisible");
            }
        }
    }
    std::unique_ptr<Poly> a(new Poly(E, 2));
    expectThrows<std::invalid_argument>([&] { PilFflonk::divideExactly(*a, 0, E.fr.one(), "a"); },
                                        "X^0 - β is not monic");
}

// divideExactly across the blocks of rapidsnark's divByMonicInPlace, whose division
// pilfflonk_polynomial_test.cpp checks: for m in {1, 3}, a = q·(X^m − β) with q of two blocks and
// one coefficient is divided back to q, in a buffer it does not own, which it keeps, on 1 and 32
// threads; a + ρ·X^(m−1) and a + X^(d−1) are refused, naming what was divided.
void testDivideExactlyAcrossBlocks() {
    const int threads = omp_get_max_threads();
    Random random(6002);
    for (uint64_t m : {1, 3}) {
        Column q(2 * Poly::divByMonicInPlaceBlockLength(m) + 1);
        for (FrElement &c : q) {
            c = random.element();
        }
        q.back() = E.fr.one();
        const FrElement beta = random.element();
        assert(!E.fr.isZero(beta));
        // a = q·X^m − β·q
        std::unique_ptr<Poly> a(new Poly(E, q.size() + m));
        for (uint64_t j = 0; j < q.size(); ++j) {
            E.fr.add(a->coef[j + m], a->coef[j + m], q[j]);
            E.fr.sub(a->coef[j], a->coef[j], E.fr.mul(beta, q[j]));
        }
        a->fixDegree();
        const uint64_t d = a->getDegree();
        assert(d == q.size() - 1 + m);

        for (int t : {1, 32}) {
            omp_set_num_threads(t);
            Column reserved(a->getLength());
            Poly borrowed(E, reserved.data(), reserved.size());
            std::copy(a->coef, a->coef + a->getLength(), borrowed.coef);
            borrowed.fixDegree();
            PilFflonk::divideExactly(borrowed, m, beta, "a");
            assert(borrowed.coef == reserved.data() && borrowed.getDegree() == q.size() - 1);
            for (uint64_t j = 0; j < reserved.size(); ++j) {
                assert(equal(reserved[j], j < q.size() ? q[j] : E.fr.zero()));
            }

            // The remainder: ρ at X^(m−1), and β^((d − 1 − r)/m)·X^r, r = (d − 1) mod m, from the top.
            std::unique_ptr<Poly> low = copyOf(*a, a->getLength());
            E.fr.add(low->coef[m - 1], low->coef[m - 1], random.element());
            low->fixDegree();
            expectThrows<std::logic_error>([&] { PilFflonk::divideExactly(*low, m, beta, "a + ρ"); },
                                           "ShplonkProver: a + ρ is not divisible");
            std::unique_ptr<Poly> high = copyOf(*a, a->getLength());
            E.fr.add(high->coef[d - 1], high->coef[d - 1], E.fr.one());
            high->fixDegree();
            expectThrows<std::logic_error>([&] { PilFflonk::divideExactly(*high, m, beta, "a + X^(d−1)"); },
                                           "ShplonkProver: a + X^(d−1) is not divisible");
        }
        omp_set_num_threads(threads);
    }
}

} // namespace

void runShplonkTests() {
    testDivideExactly();
    testDivideExactlyAcrossBlocks();
    testOpenings();
    testTamperingBreaksTheIdentity();
    testDeterminism();
    testRefusesArguments();
    testFailuresAfterSqueezing();
}

} // namespace PilFflonkTest
