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
#include "pilfflonk_test.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
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

} // namespace

void runPolynomialTests() {
    testDivByMonic();
    testByXSubValue();
    testAdd();
    testDivBy();
    testLagrangeInterpolation();
    testZerofier();
    assert(mismatches == 0);
}

} // namespace PilFflonkTest
