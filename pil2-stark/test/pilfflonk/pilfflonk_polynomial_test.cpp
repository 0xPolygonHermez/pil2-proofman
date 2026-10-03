// Tests for the methods of rapidsnark's Polynomial that pilfflonk's SHPLONK prover and the final
// wrap's FflonkProver call, and that used to leak (pilfflonk/docs/README.md#rapidsnark-and-ffiasm):
// - each result (its length, its degree and its coefficients) is pinned by a Keccak-256 to what the
//   code gave before the leaks were fixed, quirks included: add() and byXSubValue() grow the buffer
//   without updating the length, and a polynomial on a reserved buffer moves to one of its own;
// - what is well defined is also checked for what it is: quotients, products, interpolants and
//   zerofiers, and the contents left in a reserved buffer;
// - the interpolations and zerofiers have the FflonkProver's sizes: R0, R1 and R2 interpolate 8, 4
//   and 6 points, and ZT and ZTS2 vanish on 18 and 10.
// Under LeakSanitizer they also show that these calls free everything they allocate, on owned and
// on reserved buffers alike.
//
// The methods added for pilfflonk (pilfflonk/docs/performance.md#the-shplonk-division) are checked
// against what they replace: Polynomial::divByMonicInPlace against divByMonic and the serial
// division, Polynomial::fromReservedBuffer against the constructor on a reserved buffer, and
// CPolynomial::getCoefficients against the packing written out and against getPolynomial.
#include "pilfflonk_test.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
#include "cpolynomial.hpp"
#include "pilfflonk_api.hpp"
#include "polynomial.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using Poly = Polynomial<Engine>;
using Column = std::vector<FrElement>;

Engine &E = Engine::engine;

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

private:
    std::mt19937_64 generator;
};

bool equal(const FrElement &a, const FrElement &b) {
    return E.fr.eq(a, b);
}

bool same(const FrElement *a, const Column &b) {
    for (uint64_t i = 0; i < b.size(); ++i) {
        if (!equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

// Whether coef[0..n) holds `expected` followed by zeros.
bool holds(const Poly &p, const Column &expected, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        if (!equal(p.coef[i], i < expected.size() ? expected[i] : E.fr.zero())) {
            return false;
        }
    }
    return true;
}

// The polynomial with these coefficients, as a column of `length`.
Column padded(const Column &coefs, uint64_t length) {
    Column c(length, E.fr.zero());
    std::copy(coefs.begin(), coefs.end(), c.begin());
    return c;
}

Column scaled(const Column &a, const FrElement &s) {
    Column r(a.size());
    for (uint64_t i = 0; i < a.size(); ++i) {
        r[i] = E.fr.mul(a[i], s);
    }
    return r;
}

// Copies `coefs` into p, which must be long enough, and fixes its degree.
void load(Poly &p, const Column &coefs) {
    assert(coefs.size() <= p.getLength());
    std::copy(coefs.begin(), coefs.end(), p.coef);
    p.fixDegree();
}

// Keccak-256 of the length and the degree, as u64 little-endian, then of the n coefficients from
// coef[0], each as 32 canonical little-endian bytes. n can exceed the length, for the buffers that
// add() and byXSubValue() grow without updating it.
std::string digest(const Poly &p, uint64_t n) {
    std::vector<uint8_t> bytes(16 + 32 * n, 0);
    const uint64_t header[2] = {p.getLength(), p.getDegree()};
    for (int w = 0; w < 2; ++w) {
        for (int b = 0; b < 8; ++b) {
            bytes[8 * w + b] = static_cast<uint8_t>(header[w] >> (8 * b));
        }
    }
    for (uint64_t i = 0; i < n; ++i) {
        E.fr.toRprLE(p.coef[i], bytes.data() + 16 + 32 * i, 32);
    }
    uint8_t out[32];
    const int status = pilfflonk_keccak256(bytes.data(), bytes.size(), out);
    assert(status == PILFFLONK_OK);
    static const char *const hex = "0123456789abcdef";
    std::string s;
    for (uint8_t byte : out) {
        s += hex[byte >> 4];
        s += hex[byte & 15];
    }
    return s;
}

int mismatches = 0;

// Every mismatch is printed, with the digest found, before runPolynomialTests() fails.
void expectDigest(const std::string &what, const Poly &p, uint64_t n, const char *expected) {
    const std::string found = digest(p, n);
    if (found != expected) {
        std::fprintf(stderr, "polynomial test: %s: digest %s, expected %s\n", what.c_str(), found.c_str(), expected);
        ++mismatches;
    }
}

// divByMonic(m, β) on a = q·(X^m - β) + r, deg r < m: the quotient q, on an owned buffer and on a
// reserved one, which it leaves as it was. m = 1 is the SHPLONK prover's W' = L/(X - y); the others
// are its f_i - r_i divided by X^k - ξ·ω^s. (4, 3) and (6, 5) have deg a = 2m - 1, the least it
// takes (it writes below its buffer otherwise), for which only its first loop runs.
void testDivByMonic() {
    struct Case {
        uint32_t m;
        uint64_t qDegree;
        const char *expected;
    };
    const Case cases[] = {
        {1, 10, "c7f40f1256a7bc449f8da02e2f991e721ff2c579d26d07214c8ce029924d45af"},
        {3, 9, "e38141fb9b2bf896671b31d862e834377633edde21ae997ab20209d3652cabe4"},
        {4, 3, "9911e0a71318bc2970171c9b6586f9e111ce8c9860cf139b94957cb2daf3b19c"},
        {6, 5, "84230496e9ee137a6af12063c08fb89a0767fd3b8d589600aa9d1ca3852ecb19"},
    };
    const uint64_t length = 16;
    Random random(0x9d1);
    for (const Case &c : cases) {
        const Column q = random.column(c.qDegree + 1);
        const Column r = random.column(c.m);
        const FrElement beta = random.element();
        Column a = padded(r, length);
        for (uint64_t j = 0; j <= c.qDegree; ++j) {
            a[j + c.m] = E.fr.add(a[j + c.m], q[j]);
            a[j] = E.fr.sub(a[j], E.fr.mul(beta, q[j]));
        }
        const std::string name = "divByMonic(" + std::to_string(c.m) + ")";

        Poly owned(E, length);
        load(owned, a);
        assert(owned.getDegree() == c.qDegree + c.m);
        owned.divByMonic(c.m, beta);
        assert(owned.getLength() == length && owned.getDegree() == c.qDegree && holds(owned, q, length));
        expectDigest(name + " owned", owned, length, c.expected);

        Column reserved(length);
        Poly borrowed(E, reserved.data(), length);
        load(borrowed, a);
        borrowed.divByMonic(c.m, beta);
        assert(borrowed.getLength() == length && borrowed.getDegree() == c.qDegree && holds(borrowed, q, length));
        assert(same(reserved.data(), a));
        expectDigest(name + " reserved", borrowed, length, c.expected);
    }
}

// byXSubValue(x): p·(X - x). On a reserved buffer it leaves -x·p there (it scales p in place) and
// moves to a buffer of its own. When the top coefficient is not zero the buffer grows by one, but
// the length and the degree stay: the product is all there, past the length.
void testByXSubValue() {
    Random random(0xb75);
    const Column xs = random.column(3);
    FrElement x1 = xs[1];
    FrElement x2 = xs[2];
    const Column linear = {E.fr.neg(xs[0]), E.fr.one()};
    // (X - x0)(X - x1)
    const Column quadratic = {E.fr.mul(xs[0], xs[1]), E.fr.neg(E.fr.add(xs[0], xs[1])), E.fr.one()};
    const char *const cubic = "027fc3f4bfe89323cb3bdfe2730a52bc44d9ec8144281fab9d9b6da6e5b34408";
    const char *const grown = "1d1cf187b229ba2b513c86736427e57b2d364bc99edb65faff9d5aea8ad98b41";

    Poly owned(E, 4);
    load(owned, linear);
    owned.byXSubValue(x1);
    assert(owned.getLength() == 4 && owned.getDegree() == 2 && holds(owned, quadratic, 4));
    owned.byXSubValue(x2);
    assert(owned.getLength() == 4 && owned.getDegree() == 3 && equal(owned.coef[3], E.fr.one()));
    for (const FrElement &x : xs) {
        assert(E.fr.isZero(owned.evaluate(x)));
    }
    expectDigest("byXSubValue owned", owned, 4, cubic);

    Column reserved(4);
    Poly borrowed(E, reserved.data(), 4);
    load(borrowed, linear);
    borrowed.byXSubValue(x1);
    assert(same(reserved.data(), padded(scaled(linear, E.fr.neg(x1)), 4)));
    borrowed.byXSubValue(x2);
    assert(same(reserved.data(), padded(scaled(linear, E.fr.neg(x1)), 4)));
    expectDigest("byXSubValue reserved", borrowed, 4, cubic);

    Poly full(E, 2);
    load(full, linear);
    full.byXSubValue(x1);
    assert(full.getLength() == 2 && full.getDegree() == 1 && holds(full, quadratic, 3));
    expectDigest("byXSubValue grown owned", full, 3, grown);

    Column reservedFull(2);
    Poly borrowedFull(E, reservedFull.data(), 2);
    load(borrowedFull, linear);
    borrowedFull.byXSubValue(x1);
    assert(borrowedFull.getLength() == 2 && borrowedFull.getDegree() == 1 && holds(borrowedFull, quadratic, 3));
    assert(same(reservedFull.data(), scaled(linear, E.fr.neg(x1))));
    expectDigest("byXSubValue grown reserved", borrowedFull, 3, grown);
}

// add(b) and addBlinding(b, s), a + b and a + s·b: in place when b is not longer; otherwise into a
// buffer of b's length, keeping a's length (and so a degree below a's length). A reserved buffer
// is then left as it was.
void testAdd() {
    Random random(0xadd);
    const Column a = random.column(3);
    const Column b = random.column(5);
    FrElement s = random.element();
    Column sum(5), blinded(5);
    for (uint64_t i = 0; i < 5; ++i) {
        const FrElement ai = i < 3 ? a[i] : E.fr.zero();
        sum[i] = E.fr.add(ai, b[i]);
        blinded[i] = E.fr.add(ai, E.fr.mul(s, b[i]));
    }
    Poly pb(E, 5);
    load(pb, b);

    Poly inPlace(E, 5);
    load(inPlace, a);
    inPlace.add(pb);
    assert(inPlace.getLength() == 5 && inPlace.getDegree() == 4 && holds(inPlace, sum, 5));
    expectDigest("add in place", inPlace, 5,
                 "423376dca8c902cd6e80f29b506745c9186e1a34215434c24a0df2efb43f75e2");

    Poly owned(E, 3);
    load(owned, a);
    owned.add(pb);
    assert(owned.getLength() == 3 && owned.getDegree() == 2 && holds(owned, sum, 5));
    expectDigest("add grown owned", owned, 5,
                 "a7180635a20ee8a9b7a4228201a0c0fcca92c5f3d66d7d73fb8ace6475f4521a");

    Column reserved(3);
    Poly borrowed(E, reserved.data(), 3);
    load(borrowed, a);
    borrowed.add(pb);
    assert(borrowed.getLength() == 3 && borrowed.getDegree() == 2 && holds(borrowed, sum, 5));
    assert(same(reserved.data(), a));
    expectDigest("add grown reserved", borrowed, 5,
                 "a7180635a20ee8a9b7a4228201a0c0fcca92c5f3d66d7d73fb8ace6475f4521a");

    Poly blindedInPlace(E, 5);
    load(blindedInPlace, a);
    blindedInPlace.addBlinding(pb, s);
    assert(blindedInPlace.getDegree() == 4 && holds(blindedInPlace, blinded, 5));
    expectDigest("addBlinding in place", blindedInPlace, 5,
                 "0d2da86669a5459b891eb01aa55d1c4331cb8fb338c023706d876e3c7be35d8b");

    Column reservedBlinded(3);
    Poly borrowedBlinded(E, reservedBlinded.data(), 3);
    load(borrowedBlinded, a);
    borrowedBlinded.addBlinding(pb, s);
    assert(borrowedBlinded.getLength() == 3 && borrowedBlinded.getDegree() == 2 &&
           holds(borrowedBlinded, blinded, 5));
    assert(same(reservedBlinded.data(), a));
    expectDigest("addBlinding grown reserved", borrowedBlinded, 5,
                 "0a1d89e0b7f47a4a5fdbd9b4abd566e7e4dbb23ae28e9822b8053fc390864900");
}

// divBy(b), Euclidean division: a := q and the remainder r returned, for a = q·b + r. The shplonk
// tests check (f_i - r_i)·Z_{T∖T_i} with it. On a reserved buffer the remainder stays there, in a
// polynomial that does not free it, and a moves to a buffer it owns (before the leak fix, the
// remainder freed the reserved buffer and a's new one leaked).
void testDivBy() {
    Random random(0xd1b);
    const Column b = random.column(4);
    const Column q = random.column(6);
    const Column r = random.column(3);
    const uint64_t length = 12;
    Column a = padded(r, length);
    for (uint64_t i = 0; i < q.size(); ++i) {
        for (uint64_t j = 0; j < b.size(); ++j) {
            a[i + j] = E.fr.add(a[i + j], E.fr.mul(q[i], b[j]));
        }
    }
    const char *const quotient = "843246d64a57d5089d85367862f17d1c3e676ccb91be6a005e357a044dbf1e0d";
    const char *const remainder = "4f7741e6d3d0e146a844519328a240cbc6d2257676a47b94f7c0d2d289737bde";
    Poly pb(E, 4);
    load(pb, b);

    Poly owned(E, length);
    load(owned, a);
    const std::unique_ptr<Poly> ownedRemainder(owned.divBy(pb));
    assert(owned.getLength() == length && owned.getDegree() == 5 && holds(owned, q, length));
    assert(ownedRemainder->getLength() == length && ownedRemainder->getDegree() == 2 &&
           holds(*ownedRemainder, r, length));
    expectDigest("divBy quotient owned", owned, length, quotient);
    expectDigest("divBy remainder owned", *ownedRemainder, length, remainder);

    Column reserved(length);
    Poly borrowed(E, reserved.data(), length);
    load(borrowed, a);
    const std::unique_ptr<Poly> borrowedRemainder(borrowed.divBy(pb));
    assert(borrowedRemainder->coef == reserved.data());
    assert(borrowed.getLength() == length && borrowed.getDegree() == 5 && holds(borrowed, q, length));
    assert(borrowedRemainder->getLength() == length && borrowedRemainder->getDegree() == 2 &&
           holds(*borrowedRemainder, r, length));
    expectDigest("divBy quotient reserved", borrowed, length, quotient);
    expectDigest("divBy remainder reserved", *borrowedRemainder, length, remainder);
}

// lagrangePolynomialInterpolation over n points: the polynomial of degree < n through them, in a
// buffer of n coefficients. 4, 6 and 8 are the FflonkProver's R1, R2 and R0; 12 is a SHPLONK
// opening at two points of a 6-component f.
void testLagrangeInterpolation() {
    struct Case {
        uint32_t n;
        const char *expected;
    };
    const Case cases[] = {
        {2, "ab7d02d00ecebbe31f818feb6b77fc151de33eb7818c2038f1e924736d0ef4f8"},
        {3, "dd2fb304a682b87a3516b0ab4b4bb00a0a136a9c52ee10ef59027c87c12a00f9"},
        {4, "664a247118a8a4a61424c5381f5c7010632f63e11ea92d349c6424724692259d"},
        {6, "3cf9a20487e155a8f88de80a6a2149c0b88fa36d0ce063d002e2afbad9c754a1"},
        {8, "4c21f173a4f2d04207bdbb2a77eef27d8f6359117138f8145ff79fba76ccebb7"},
        {12, "485a9178a602ce3509653daecc81e631df43b357b6f71dd3d57f118879e2dbb5"},
    };
    Random random(0x1a9);
    for (const Case &c : cases) {
        Column xs = random.column(c.n);
        Column ys = random.column(c.n);
        const std::unique_ptr<Poly> p(Poly::lagrangePolynomialInterpolation(xs.data(), ys.data(), c.n));
        assert(p->getLength() == c.n && p->getDegree() < c.n);
        for (uint32_t i = 0; i < c.n; ++i) {
            assert(equal(p->evaluate(xs[i]), ys[i]));
        }
        expectDigest("lagrangePolynomialInterpolation(" + std::to_string(c.n) + ")", *p, c.n, c.expected);
    }
}

// zerofierPolynomial over n points: the monic polynomial of degree n vanishing on them, in a buffer
// of n + 1 coefficients. 10 and 18 are the FflonkProver's ZTS2 and ZT.
void testZerofier() {
    struct Case {
        uint32_t n;
        const char *expected;
    };
    const Case cases[] = {
        {1, "af3cddf8a4b0c3d401d33c0391c4bf2d30082d2fd97083ab708b0db1f208b3e0"},
        {2, "c0bab863b575c9bf1615ad0e922b4e6c2c141950aea7c313cc18ddebbfb94372"},
        {10, "e2ca758d19f76867d9800f32e280a931869d9e32feae151c8854625026ec5d21"},
        {18, "05c7d63f6cd1c9fd87b950e85977e7e0ed91982afb3b01bcf3d4541ef0237682"},
    };
    Random random(0x2e0);
    for (const Case &c : cases) {
        Column xs = random.column(c.n);
        const std::unique_ptr<Poly> p(Poly::zerofierPolynomial(xs.data(), c.n));
        assert(p->getLength() == c.n + 1 && p->getDegree() == c.n && equal(p->coef[c.n], E.fr.one()));
        for (const FrElement &x : xs) {
            assert(E.fr.isZero(p->evaluate(x)));
        }
        expectDigest("zerofierPolynomial(" + std::to_string(c.n) + ")", *p, c.n + 1, c.expected);
    }
}

// A polynomial owning a copy of p's coefficients, of p's length, with its degree fixed.
std::unique_ptr<Poly> copyOf(const Poly &p) {
    std::unique_ptr<Poly> copy(new Poly(E, p.getLength()));
    std::copy(p.coef, p.coef + p.getLength(), copy->coef);
    copy->fixDegree();
    return copy;
}

// a = q·(X^m − β) in `extra` more coefficients than it has.
std::unique_ptr<Poly> multiple(const Column &q, uint64_t m, const FrElement &beta, uint64_t extra) {
    std::unique_ptr<Poly> a(new Poly(E, q.size() + m + extra));
    for (uint64_t j = 0; j < q.size(); ++j) {
        E.fr.add(a->coef[j + m], a->coef[j + m], q[j]);
        E.fr.sub(a->coef[j], a->coef[j], E.fr.mul(beta, q[j]));
    }
    a->fixDegree();
    return a;
}

bool sameCoefficients(const Poly &a, const Poly &b) {
    if (a.getLength() != b.getLength() || a.getDegree() != b.getDegree()) {
        return false;
    }
    for (uint64_t j = 0; j < a.getLength(); ++j) {
        if (!equal(a.coef[j], b.coef[j])) {
            return false;
        }
    }
    return true;
}

template <typename Call>
void expectRuntimeError(Call call, const char *message) {
    try {
        call();
    } catch (const std::runtime_error &e) {
        if (std::strstr(e.what(), message) == nullptr) {
            std::fprintf(stderr, "unexpected message: %s\n", e.what());
            assert(!"the exception does not say what was expected");
        }
        return;
    }
    assert(!"expected std::runtime_error");
}

// The division by X^m − β, deg a >= m, serially: divByMonic, or for m <= deg a < 2m − 1, where it
// writes below its buffer, q_j = a_{j+m} by hand; then the remainder a_j + β·q_j (j < m). a is
// its quotient; returns whether the remainder is zero.
bool serialDivByMonic(Poly &a, uint64_t m, const FrElement &beta) {
    const uint64_t d = a.getDegree();
    assert(d >= m);
    const Column low(a.coef, a.coef + m);
    if (d >= 2 * m - 1) {
        a.divByMonic(static_cast<uint32_t>(m), beta);
    } else {
        for (uint64_t j = 0; j <= d; ++j) {
            a.coef[j] = j <= d - m ? a.coef[j + m] : E.fr.zero();
        }
        a.fixDegree();
    }
    for (uint64_t j = 0; j < m; ++j) {
        if (!E.fr.isZero(E.fr.add(low[j], E.fr.mul(beta, a.coef[j])))) {
            return false;
        }
    }
    return true;
}

// divByMonicInPlace's blocked scan against the serial division (serialDivByMonic) and against
// divByMonic, coefficient by coefficient, the length and the degree too, on 1, 2 and 32 threads.
// For m in {1, 2, 3, 5, 9}, a = q·(X^m − β) with q of 1, m, 2m − 1 and 2m coefficients, of a block,
// one more and one less, two and three blocks around their boundaries, and 2^20 + 3; q random, all
// ones, or zero but its top and bottom; β random, 0, 1 or −1. Each a is divided back to q, in its
// own buffer, with a zero remainder; a + ρ·X^j, for a j below m and one in the top block, has the
// serial quotient and a remainder that is not zero (unless β = 0 for the top one).
void testDivByMonicInPlace() {
    const int threads = omp_get_max_threads();
    Random random(6002);
    const FrElement betas[] = {random.element(), E.fr.zero(), E.fr.one(), E.fr.negOne()};
    for (uint64_t m : {1, 2, 3, 5, 9}) {
        const uint64_t block = Poly::divByMonicInPlaceBlockLength(m);
        assert(block % m == 0 && block >= 4096 - m && block <= 4096);
        const uint64_t lengths[] = {1,         m,         2 * m - 1, 2 * m,     block - 1, block,     block + 1,
                                    2 * block - 1, 2 * block, 2 * block + 1, 3 * block + m, (uint64_t(1) << 20) + 3};
        for (uint64_t qLength : lengths) {
            const bool large = qLength > 4 * block;
            for (int kind = 0; kind < (large ? 1 : 3); ++kind) {
                Column q(qLength, E.fr.zero());
                for (uint64_t j = 0; j < qLength; ++j) {
                    if (kind == 0) {
                        q[j] = random.element();
                    } else if (kind == 1) {
                        q[j] = E.fr.one();
                    }
                }
                q.front() = kind == 2 ? E.fr.one() : q.front();
                q.back() = E.fr.isZero(q.back()) || kind == 2 ? E.fr.one() : q.back();
                for (const FrElement &beta : betas) {
                    if (large && &beta != &betas[0]) {
                        continue;
                    }
                    const std::unique_ptr<Poly> a = multiple(q, m, beta, 2);
                    assert(a->getDegree() == qLength - 1 + m);

                    std::unique_ptr<Poly> serial = copyOf(*a);
                    assert(serialDivByMonic(*serial, m, beta));
                    assert(serial->getDegree() == qLength - 1);
                    for (uint64_t j = 0; j < serial->getLength(); ++j) {
                        assert(equal(serial->coef[j], j < qLength ? q[j] : E.fr.zero()));
                    }
                    if (a->getDegree() >= 2 * m - 1) {
                        std::unique_ptr<Poly> byMonic = copyOf(*a);
                        byMonic->divByMonic(static_cast<uint32_t>(m), beta);
                        assert(sameCoefficients(*byMonic, *serial));
                    }

                    // Not divisible: ρ·X^j, j < m, adds ρ to the remainder; at j in the top block,
                    // β^((j − j mod m)/m)·ρ·X^(j mod m) (or nothing if β = 0), through every carry.
                    std::unique_ptr<Poly> low = copyOf(*a);
                    E.fr.add(low->coef[m - 1], low->coef[m - 1], random.element());
                    low->fixDegree();
                    std::unique_ptr<Poly> lowSerial = copyOf(*low);
                    assert(!serialDivByMonic(*lowSerial, m, beta));
                    std::unique_ptr<Poly> high = copyOf(*a);
                    E.fr.add(high->coef[a->getDegree() - 1], high->coef[a->getDegree() - 1], E.fr.one());
                    high->fixDegree();
                    const bool highDivisible = E.fr.isZero(beta) && a->getDegree() - 1 >= m;
                    std::unique_ptr<Poly> highSerial = copyOf(*high);
                    assert(serialDivByMonic(*highSerial, m, beta) == highDivisible);

                    for (int t : {1, 2, 32}) {
                        omp_set_num_threads(t);
                        std::unique_ptr<Poly> parallel = copyOf(*a);
                        const FrElement *buffer = parallel->coef;
                        assert(parallel->divByMonicInPlace(m, beta));
                        assert(parallel->coef == buffer && sameCoefficients(*parallel, *serial));

                        std::unique_ptr<Poly> refused = copyOf(*low);
                        assert(!refused->divByMonicInPlace(m, beta));
                        assert(sameCoefficients(*refused, *lowSerial));
                        std::unique_ptr<Poly> top = copyOf(*high);
                        assert(top->divByMonicInPlace(m, beta) == highDivisible);
                        assert(sameCoefficients(*top, *highSerial));
                    }
                    omp_set_num_threads(threads);
                }
            }
        }
    }

    // In place, in a buffer it does not own, which it keeps: divByMonic moves to one of its own.
    const uint64_t m = 3;
    Column q(2 * Poly::divByMonicInPlaceBlockLength(m) + 7);
    for (FrElement &c : q) {
        c = random.element();
    }
    q.back() = E.fr.one();
    const FrElement beta = random.element();
    const std::unique_ptr<Poly> a = multiple(q, m, beta, 0);
    Column reserved(a->getLength());
    Poly borrowed(E, reserved.data(), reserved.size());
    std::copy(a->coef, a->coef + a->getLength(), borrowed.coef);
    borrowed.fixDegree();
    assert(borrowed.divByMonicInPlace(m, beta));
    assert(borrowed.coef == reserved.data() && borrowed.getLength() == reserved.size() &&
           borrowed.getDegree() == q.size() - 1);
    for (uint64_t j = 0; j < reserved.size(); ++j) {
        assert(equal(reserved[j], j < q.size() ? q[j] : E.fr.zero()));
    }

    // A divisor of degree 0 or above the polynomial's.
    expectRuntimeError([&] { borrowed.divByMonicInPlace(0, beta); }, "X^0 - beta is not of degree at least 1");
    expectRuntimeError([] { Poly::divByMonicInPlaceBlockLength(0); }, "X^0 - beta is not of degree at least 1");
    expectRuntimeError([&] { borrowed.divByMonicInPlace(borrowed.getDegree() + 1, beta); },
                       "X^m - beta is of a degree above the polynomial's");
    Poly constant(E, 4);
    constant.coef[0] = beta;
    constant.fixDegree();
    expectRuntimeError([&] { constant.divByMonicInPlace(1, beta); },
                       "X^m - beta is of a degree above the polynomial's");
}

// fromReservedBuffer(buffer, n): a polynomial over the n coefficients the buffer holds, which it
// neither clears nor copies (the constructor on a reserved buffer clears them), with its degree
// fixed, and which leaves the buffer to its owner; what is beyond n is not touched. Then the
// SHPLONK prover's use of it, divByMonicInPlace on it, in the same buffer.
void testFromReservedBuffer() {
    Random random(0xf2b);
    const FrElement beta = random.element();
    const Column q = random.column(9);
    Column a(16, E.fr.zero());
    for (uint64_t j = 0; j < q.size(); ++j) {
        a[j + 3] = E.fr.add(a[j + 3], q[j]);
        a[j] = E.fr.sub(a[j], E.fr.mul(beta, q[j]));
    }
    a[12] = random.element();
    a[15] = random.element();
    // a's first 12 coefficients, the multiple q·(X^3 − β) and a zero, then two stale ones.
    Column buffer = a;
    {
        const std::unique_ptr<Poly> p(Poly::fromReservedBuffer(E, buffer.data(), 12));
        assert(p->coef == buffer.data() && p->getLength() == 12 && p->getDegree() == 11);
        assert(same(buffer.data(), a));
        assert(p->divByMonicInPlace(3, beta));
        assert(p->coef == buffer.data() && p->getDegree() == 8 && holds(*p, q, 12));
    }
    assert(same(buffer.data() + 12, Column(a.begin() + 12, a.end())));

    // Zeros at the top: the degree is fixed below them; none at all: degree 0, nothing read.
    const std::unique_ptr<Poly> zeros(Poly::fromReservedBuffer(E, buffer.data() + 9, 3));
    assert(zeros->getLength() == 3 && zeros->getDegree() == 0);
    const std::unique_ptr<Poly> empty(Poly::fromReservedBuffer(E, buffer.data(), 0));
    assert(empty->getLength() == 0 && empty->getDegree() == 0 && empty->coef == buffer.data());
}

// The packing of CPolynomial written out (pilfflonk/docs/protocol.md#layout): coefficient c·k + i
// of f is coefficient c of p_i up to its degree, zero above it and where no p_i was added, in
// 1 + max_i(k·deg p_i + i) coefficients, over the positions with a polynomial.
Column interleaved(const std::vector<const Poly *> &polys) {
    const uint64_t k = polys.size();
    uint64_t bound = 0;
    for (uint64_t i = 0; i < k; ++i) {
        if (polys[i] != nullptr) {
            bound = std::max(bound, k * polys[i]->getDegree() + i);
        }
    }
    Column f(bound + 1, E.fr.zero());
    for (uint64_t i = 0; i < k; ++i) {
        if (polys[i] != nullptr) {
            for (uint64_t c = 0; c <= polys[i]->getDegree(); ++c) {
                f[c * k + i] = polys[i]->coef[c];
            }
        }
    }
    return f;
}

// CPolynomial::getCoefficients against the packing written out (interleaved: the coefficients and
// the count pilfflonk's pack() returns) and, where it is defined (a degree bound of 2 or more),
// against getPolynomial's coefficients, on 1, 2 and 32 threads. k = 1 and 12, constants, zeros,
// mixed degrees, a degree bound a power of two (getPolynomial's polynomial is one coefficient short
// then) and above deg f (zero top polynomials), a position with no polynomial, and f of 12,286
// coefficients. It writes into a buffer of stale data, and nothing beyond the count.
void testGetCoefficients() {
    const int threads = omp_get_max_threads();
    Random random(0xc0f);
    // The degree of each p_i, or -1 for none; -2 for the zero polynomial.
    const std::vector<std::vector<int64_t>> cases = {
        {0}, {1}, {4}, {16}, {1000}, {-2},
        {0, 0}, {3, 0}, {0, 3}, {-2, -2}, {1, -1},
        {0, -2, -2}, {5, -1, 2}, {4095, 4094, 100},
        {7, 2, 0, 5},
        {63, 62, 7, 0, -2, 63, 62, 7, 0, -2, 1, -1},
    };
    for (const std::vector<int64_t> &degrees : cases) {
        const uint64_t k = degrees.size();
        std::vector<std::unique_ptr<Poly>> owned;
        std::vector<const Poly *> polys;
        CPolynomial<Engine> cpolynomial(E, static_cast<int>(k));
        for (uint64_t i = 0; i < k; ++i) {
            if (degrees[i] == -1) {
                polys.push_back(nullptr);
                continue;
            }
            // Two coefficients more than the degree, which are zero.
            const uint64_t degree = degrees[i] < 0 ? 0 : static_cast<uint64_t>(degrees[i]);
            owned.emplace_back(new Poly(E, degree + 3));
            Poly &p = *owned.back();
            if (degrees[i] >= 0) {
                for (uint64_t c = 0; c <= degree; ++c) {
                    p.coef[c] = random.element();
                }
                p.coef[degree] = E.fr.isZero(p.coef[degree]) ? E.fr.one() : p.coef[degree];
            }
            p.fixDegree();
            assert(p.getDegree() == degree);
            polys.push_back(&p);
            cpolynomial.addPolynomial(static_cast<int>(i), &p);
        }
        const Column expected = interleaved(polys);
        const uint64_t bound = expected.size() - 1;
        assert(cpolynomial.getDegree() == bound);

        const Column stale = random.column(expected.size() + 5);
        for (int t : {1, 2, 32}) {
            omp_set_num_threads(t);
            Column buffer = stale;
            assert(cpolynomial.getCoefficients(buffer.data()) == expected.size());
            assert(same(buffer.data(), expected));
            for (uint64_t j = expected.size(); j < buffer.size(); ++j) {
                assert(equal(buffer[j], stale[j]));
            }
        }
        omp_set_num_threads(threads);

        // getPolynomial clears 2^(floor(log2(bound − 1)) + 1) <= 2·bound elements, then writes up to
        // coefficient `bound`.
        if (bound >= 2) {
            Column reserved = random.column(2 * bound + 2);
            const std::unique_ptr<Poly> f(cpolynomial.getPolynomial(reserved.data()));
            assert(f->coef == reserved.data());
            assert(same(reserved.data(), expected));
        }
    }
}

} // namespace

void runPolynomialTests() {
    testDivByMonic();
    testByXSubValue();
    testAdd();
    testDivBy();
    testLagrangeInterpolation();
    testZerofier();
    testDivByMonicInPlace();
    testFromReservedBuffer();
    testGetCoefficients();
    assert(mismatches == 0);
}

} // namespace PilFflonkTest
