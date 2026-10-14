// Tests for rapidsnark's Polynomial over BN128's scalar field, for the functions that grow, divide
// or swap a polynomial's buffer:
// - each result is checked for what it is: quotients and remainders, products, interpolants and
//   zerofiers, and what is left in a reserved buffer;
// - each result (its length, its degree and its coefficients) is also pinned by a Keccak-256 digest
//   to what the code returned when these tests were written, quirks included: add() and
//   byXSubValue() grow the buffer without updating the length, and a polynomial on a reserved
//   buffer moves to a buffer of its own when it grows or is divided;
// - the interpolations and zerofiers have the FflonkProver's sizes: R0, R1 and R2 interpolate 8, 4
//   and 6 points, and ZT and ZTS2 vanish on 18 and 10.
// Under AddressSanitizer they also show that these functions free what they allocate, and nothing
// else, on owned and reserved buffers alike.
#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
#include "keccak_wrapper.hpp"
#include "polynomial.hpp"

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

// Whether a[0, b.size()) is b.
bool same(const FrElement *a, const Column &b) {
    for (uint64_t i = 0; i < b.size(); ++i) {
        if (!equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

// Whether coef[0, n) holds `expected` followed by zeros.
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

// The coefficients of the product of the polynomials with coefficients a and b.
Column product(const Column &a, const Column &b) {
    Column c(a.size() + b.size() - 1, E.fr.zero());
    for (uint64_t i = 0; i < a.size(); ++i) {
        for (uint64_t j = 0; j < b.size(); ++j) {
            c[i + j] = E.fr.add(c[i + j], E.fr.mul(a[i], b[j]));
        }
    }
    return c;
}

// q·(X^m - β) + r in `length` coefficients, deg r < m.
Column multiple(const Column &q, uint64_t m, const FrElement &beta, const Column &r, uint64_t length) {
    Column a = padded(r, length);
    for (uint64_t j = 0; j < q.size(); ++j) {
        a[j + m] = E.fr.add(a[j + m], q[j]);
        a[j] = E.fr.sub(a[j], E.fr.mul(beta, q[j]));
    }
    return a;
}

// Copies `coefs` into p, which must be long enough, and fixes its degree.
void load(Poly &p, const Column &coefs) {
    ASSERT_LE(coefs.size(), p.getLength());
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
    keccak(bytes.data(), bytes.size(), out, sizeof(out));
    static const char *const hex = "0123456789abcdef";
    std::string s;
    for (uint8_t byte : out) {
        s += hex[byte >> 4];
        s += hex[byte & 15];
    }
    return s;
}

// divByMonic(m, β) on a = q·(X^m - β) + r, deg r < m: the quotient q, on an owned buffer and on a
// reserved one, which it leaves as it was. (4, 3) and (6, 5) have deg a = 2m - 1, the least it
// takes (it writes below its buffer otherwise), for which only its first loop runs.
TEST(PolynomialTest, div_by_monic_leaves_the_quotient) {
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
        SCOPED_TRACE("m = " + std::to_string(c.m));
        const Column q = random.column(c.qDegree + 1);
        const Column r = random.column(c.m);
        const FrElement beta = random.element();
        const Column a = multiple(q, c.m, beta, r, length);

        Poly owned(E, length);
        load(owned, a);
        ASSERT_EQ(owned.getDegree(), c.qDegree + c.m);
        owned.divByMonic(c.m, beta);
        EXPECT_EQ(owned.getLength(), length);
        EXPECT_EQ(owned.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(owned, q, length));
        EXPECT_EQ(digest(owned, length), c.expected);

        Column reserved(length);
        Poly borrowed(E, reserved.data(), length);
        load(borrowed, a);
        borrowed.divByMonic(c.m, beta);
        EXPECT_EQ(borrowed.getLength(), length);
        EXPECT_EQ(borrowed.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(borrowed, q, length));
        EXPECT_TRUE(same(reserved.data(), a));
        EXPECT_EQ(digest(borrowed, length), c.expected);
    }
}

// byXSubValue(x): p·(X - x). On a reserved buffer it leaves -x·p there (it scales p in place) and
// moves to a buffer of its own. When the top coefficient is not zero the buffer grows by one, but
// the length and the degree stay: the product is all there, past the length.
TEST(PolynomialTest, by_x_sub_value_multiplies_by_x_minus_value) {
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
    EXPECT_EQ(owned.getLength(), 4u);
    EXPECT_EQ(owned.getDegree(), 2u);
    EXPECT_TRUE(holds(owned, quadratic, 4));
    owned.byXSubValue(x2);
    EXPECT_EQ(owned.getLength(), 4u);
    EXPECT_EQ(owned.getDegree(), 3u);
    EXPECT_TRUE(equal(owned.coef[3], E.fr.one()));
    for (const FrElement &x : xs) {
        EXPECT_TRUE(E.fr.isZero(owned.evaluate(x)));
    }
    EXPECT_EQ(digest(owned, 4), cubic);

    Column reserved(4);
    Poly borrowed(E, reserved.data(), 4);
    load(borrowed, linear);
    borrowed.byXSubValue(x1);
    EXPECT_TRUE(same(reserved.data(), padded(scaled(linear, E.fr.neg(x1)), 4)));
    borrowed.byXSubValue(x2);
    EXPECT_TRUE(same(reserved.data(), padded(scaled(linear, E.fr.neg(x1)), 4)));
    EXPECT_EQ(digest(borrowed, 4), cubic);

    Poly full(E, 2);
    load(full, linear);
    full.byXSubValue(x1);
    EXPECT_EQ(full.getLength(), 2u);
    EXPECT_EQ(full.getDegree(), 1u);
    EXPECT_TRUE(holds(full, quadratic, 3));
    EXPECT_EQ(digest(full, 3), grown);

    Column reservedFull(2);
    Poly borrowedFull(E, reservedFull.data(), 2);
    load(borrowedFull, linear);
    borrowedFull.byXSubValue(x1);
    EXPECT_EQ(borrowedFull.getLength(), 2u);
    EXPECT_EQ(borrowedFull.getDegree(), 1u);
    EXPECT_TRUE(holds(borrowedFull, quadratic, 3));
    EXPECT_TRUE(same(reservedFull.data(), scaled(linear, E.fr.neg(x1))));
    EXPECT_EQ(digest(borrowedFull, 3), grown);
}

// byXNSubValue(n, v): p·(X^n + v), for it adds v·p (byXSubValue adds -x·p) to X^n·p. On a
// reserved buffer it leaves v·p there and moves to a buffer of its own. When the product does not
// fit, the buffer grows by n, but the length stays and the degree is found below it, as with
// byXSubValue. The expected values are the products written out: until it detached the buffer it
// takes, it freed that buffer and read it back.
TEST(PolynomialTest, by_xn_sub_value_multiplies_by_xn_plus_value) {
    Random random(0xb7e);
    const Column p = random.column(3);
    FrElement v1 = random.element();
    FrElement v2 = random.element();
    // p·(X^2 + v1) and p·(X^2 + v1)·(X + v2)
    const Column first = product(p, {v1, E.fr.zero(), E.fr.one()});
    const Column second = product(first, {v2, E.fr.one()});

    Poly owned(E, 8);
    load(owned, p);
    owned.byXNSubValue(2, v1);
    EXPECT_EQ(owned.getLength(), 8u);
    EXPECT_EQ(owned.getDegree(), 4u);
    EXPECT_TRUE(holds(owned, first, 8));
    owned.byXNSubValue(1, v2);
    EXPECT_EQ(owned.getLength(), 8u);
    EXPECT_EQ(owned.getDegree(), 5u);
    EXPECT_TRUE(holds(owned, second, 8));

    Column reserved(8);
    Poly borrowed(E, reserved.data(), 8);
    load(borrowed, p);
    borrowed.byXNSubValue(2, v1);
    EXPECT_NE(borrowed.coef, reserved.data());
    EXPECT_EQ(borrowed.getDegree(), 4u);
    EXPECT_TRUE(holds(borrowed, first, 8));
    EXPECT_TRUE(same(reserved.data(), padded(scaled(p, v1), 8)));
    borrowed.byXNSubValue(1, v2);
    EXPECT_EQ(borrowed.getLength(), 8u);
    EXPECT_EQ(borrowed.getDegree(), 5u);
    EXPECT_TRUE(holds(borrowed, second, 8));
    EXPECT_TRUE(same(reserved.data(), padded(scaled(p, v1), 8)));

    Poly full(E, 3);
    load(full, p);
    full.byXNSubValue(2, v1);
    EXPECT_EQ(full.getLength(), 3u);
    EXPECT_EQ(full.getDegree(), 2u);
    EXPECT_TRUE(holds(full, first, 5));

    Column reservedFull(3);
    Poly borrowedFull(E, reservedFull.data(), 3);
    load(borrowedFull, p);
    borrowedFull.byXNSubValue(2, v1);
    EXPECT_EQ(borrowedFull.getLength(), 3u);
    EXPECT_EQ(borrowedFull.getDegree(), 2u);
    EXPECT_TRUE(holds(borrowedFull, first, 5));
    EXPECT_TRUE(same(reservedFull.data(), scaled(p, v1)));
}

// add(b) and addBlinding(b, s), a + b and a + s·b: in place when b is not longer; otherwise into a
// buffer of b's length, keeping a's length (and so a degree below a's length). A reserved buffer
// is then left as it was.
TEST(PolynomialTest, add_and_add_blinding_grow_into_a_new_buffer) {
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
    EXPECT_EQ(inPlace.getLength(), 5u);
    EXPECT_EQ(inPlace.getDegree(), 4u);
    EXPECT_TRUE(holds(inPlace, sum, 5));
    EXPECT_EQ(digest(inPlace, 5), "423376dca8c902cd6e80f29b506745c9186e1a34215434c24a0df2efb43f75e2");

    Poly owned(E, 3);
    load(owned, a);
    owned.add(pb);
    EXPECT_EQ(owned.getLength(), 3u);
    EXPECT_EQ(owned.getDegree(), 2u);
    EXPECT_TRUE(holds(owned, sum, 5));
    EXPECT_EQ(digest(owned, 5), "a7180635a20ee8a9b7a4228201a0c0fcca92c5f3d66d7d73fb8ace6475f4521a");

    Column reserved(3);
    Poly borrowed(E, reserved.data(), 3);
    load(borrowed, a);
    borrowed.add(pb);
    EXPECT_EQ(borrowed.getLength(), 3u);
    EXPECT_EQ(borrowed.getDegree(), 2u);
    EXPECT_TRUE(holds(borrowed, sum, 5));
    EXPECT_TRUE(same(reserved.data(), a));
    EXPECT_EQ(digest(borrowed, 5), "a7180635a20ee8a9b7a4228201a0c0fcca92c5f3d66d7d73fb8ace6475f4521a");

    Poly blindedInPlace(E, 5);
    load(blindedInPlace, a);
    blindedInPlace.addBlinding(pb, s);
    EXPECT_EQ(blindedInPlace.getDegree(), 4u);
    EXPECT_TRUE(holds(blindedInPlace, blinded, 5));
    EXPECT_EQ(digest(blindedInPlace, 5), "0d2da86669a5459b891eb01aa55d1c4331cb8fb338c023706d876e3c7be35d8b");

    Poly blindedOwned(E, 3);
    load(blindedOwned, a);
    blindedOwned.addBlinding(pb, s);
    EXPECT_EQ(blindedOwned.getLength(), 3u);
    EXPECT_EQ(blindedOwned.getDegree(), 2u);
    EXPECT_TRUE(holds(blindedOwned, blinded, 5));
    EXPECT_EQ(digest(blindedOwned, 5), "0a1d89e0b7f47a4a5fdbd9b4abd566e7e4dbb23ae28e9822b8053fc390864900");

    Column reservedBlinded(3);
    Poly borrowedBlinded(E, reservedBlinded.data(), 3);
    load(borrowedBlinded, a);
    borrowedBlinded.addBlinding(pb, s);
    EXPECT_EQ(borrowedBlinded.getLength(), 3u);
    EXPECT_EQ(borrowedBlinded.getDegree(), 2u);
    EXPECT_TRUE(holds(borrowedBlinded, blinded, 5));
    EXPECT_TRUE(same(reservedBlinded.data(), a));
    EXPECT_EQ(digest(borrowedBlinded, 5), "0a1d89e0b7f47a4a5fdbd9b4abd566e7e4dbb23ae28e9822b8053fc390864900");
}

// divBy(b), Euclidean division: a := q and the remainder r returned, for a = q·b + r. On a
// reserved buffer the remainder stays there, in a polynomial that does not free it, and a moves to
// a buffer of its own.
TEST(PolynomialTest, div_by_returns_the_remainder) {
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
    EXPECT_EQ(owned.getLength(), length);
    EXPECT_EQ(owned.getDegree(), 5u);
    EXPECT_TRUE(holds(owned, q, length));
    EXPECT_EQ(ownedRemainder->getLength(), length);
    EXPECT_EQ(ownedRemainder->getDegree(), 2u);
    EXPECT_TRUE(holds(*ownedRemainder, r, length));
    EXPECT_EQ(digest(owned, length), quotient);
    EXPECT_EQ(digest(*ownedRemainder, length), remainder);

    Column reserved(length);
    Poly borrowed(E, reserved.data(), length);
    load(borrowed, a);
    const std::unique_ptr<Poly> borrowedRemainder(borrowed.divBy(pb));
    EXPECT_EQ(borrowedRemainder->coef, reserved.data());
    EXPECT_EQ(borrowed.getLength(), length);
    EXPECT_EQ(borrowed.getDegree(), 5u);
    EXPECT_TRUE(holds(borrowed, q, length));
    EXPECT_EQ(borrowedRemainder->getLength(), length);
    EXPECT_EQ(borrowedRemainder->getDegree(), 2u);
    EXPECT_TRUE(holds(*borrowedRemainder, r, length));
    EXPECT_EQ(digest(borrowed, length), quotient);
    EXPECT_EQ(digest(*borrowedRemainder, length), remainder);
}

// divByVanishing(m, β) on a = q·(X^m - β) + r, deg r < m: a := q and r returned. On a reserved
// buffer the remainder stays there, as with divBy.
TEST(PolynomialTest, div_by_vanishing_returns_the_remainder) {
    struct Case {
        uint32_t m;
        uint64_t qDegree;
        const char *quotient;
        const char *remainder;
    };
    const Case cases[] = {
        {1, 10, "c05dfbac6a0058fd57eb52f935ff12e71323f900977223ac3cd4188894a35240",
         "469c438ad8b762738a9ca1d22fd895a1da4e4ac14629d399be0f268b7af4202f"},
        {3, 9, "159af10da91fd6029e71b3b16bff90462341e88e7a13aca26d86bf81f8af0fd0",
         "5b1c9d8c3c1de662dc83df9de9142eb348e5688b5f366bea237c0f56bffeb1aa"},
        {4, 3, "b75d3e3131311798934ea62b1b43b0228bbf1e585add2cf7c46f040bc64f5138",
         "df1f1ab18d2c776453c92651d1e90aa5a873c4c40786100c3137c3980c07b49d"},
        {6, 5, "a4592abcbc6936832e6cdb9593edca9f5f92e5d62e83f9a67263f87aaa84ba80",
         "186147eed07da30b983f34bddcc5eaf802f7ae9d4cd20aae96cd431dbf7ab435"},
    };
    const uint64_t length = 16;
    Random random(0xd7a);
    for (const Case &c : cases) {
        SCOPED_TRACE("m = " + std::to_string(c.m));
        const Column q = random.column(c.qDegree + 1);
        const Column r = random.column(c.m);
        const FrElement beta = random.element();
        const Column a = multiple(q, c.m, beta, r, length);

        Poly owned(E, length);
        load(owned, a);
        const std::unique_ptr<Poly> ownedRemainder(owned.divByVanishing(c.m, beta));
        EXPECT_EQ(owned.getLength(), length);
        EXPECT_EQ(owned.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(owned, q, length));
        EXPECT_EQ(ownedRemainder->getLength(), length);
        EXPECT_EQ(ownedRemainder->getDegree(), c.m - 1);
        EXPECT_TRUE(holds(*ownedRemainder, r, length));
        EXPECT_EQ(digest(owned, length), c.quotient);
        EXPECT_EQ(digest(*ownedRemainder, length), c.remainder);

        Column reserved(length);
        Poly borrowed(E, reserved.data(), length);
        load(borrowed, a);
        const std::unique_ptr<Poly> borrowedRemainder(borrowed.divByVanishing(c.m, beta));
        EXPECT_EQ(borrowedRemainder->coef, reserved.data());
        EXPECT_EQ(borrowed.getLength(), length);
        EXPECT_EQ(borrowed.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(borrowed, q, length));
        EXPECT_EQ(borrowedRemainder->getLength(), length);
        EXPECT_EQ(borrowedRemainder->getDegree(), c.m - 1);
        EXPECT_TRUE(holds(*borrowedRemainder, r, length));
        EXPECT_EQ(digest(borrowed, length), c.quotient);
        EXPECT_EQ(digest(*borrowedRemainder, length), c.remainder);
    }
}

// fastDivByVanishing(buffer, m, β) on a = q·(X^m - β) + r, deg r < m: a := q, with a copied to
// the buffer. On one thread: on more, the steps that every thread runs race with the last one (no
// barrier separates them), and the quotient is sometimes wrong. It reads coef[deg a + 1], so the
// degree is below length - 1.
TEST(PolynomialTest, fast_div_by_vanishing_leaves_the_quotient) {
    struct Case {
        uint32_t m;
        uint64_t qDegree;
        const char *expected;
    };
    const Case cases[] = {
        {1, 40, "6d316374364e3020177bb3b9e0ea692777dde9bb89750f695ef2e015348af362"},
        {3, 30, "ac3497881b6eff62cd5a988e1424c2f5025f6b6c3f2e6ee6bb2c8a64a900571b"},
        {4, 50, "8910c3043ce5b546f8cbef8482ce0fe603f6c09384b32ee1a0aa1c4e1886d3c4"},
    };
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const uint64_t length = 64;
    Random random(0xfd7);
    for (const Case &c : cases) {
        SCOPED_TRACE("m = " + std::to_string(c.m));
        const Column q = random.column(c.qDegree + 1);
        const Column r = random.column(c.m);
        const FrElement beta = random.element();
        const Column a = multiple(q, c.m, beta, r, length);

        Poly p(E, length);
        load(p, a);
        Column buffer(length);
        p.fastDivByVanishing(buffer.data(), c.m, beta);
        EXPECT_EQ(p.getLength(), length);
        EXPECT_EQ(p.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(p, q, length));
        EXPECT_TRUE(same(buffer.data(), a));
        EXPECT_EQ(digest(p, length), c.expected);
    }
    omp_set_num_threads(threads);
}

// lagrangePolynomialInterpolation over n points: the polynomial of degree < n through them, in a
// buffer of n coefficients. 4, 6 and 8 are the FflonkProver's R1, R2 and R0.
TEST(PolynomialTest, lagrange_interpolation_goes_through_the_points) {
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
        SCOPED_TRACE("n = " + std::to_string(c.n));
        Column xs = random.column(c.n);
        Column ys = random.column(c.n);
        const std::unique_ptr<Poly> p(Poly::lagrangePolynomialInterpolation(xs.data(), ys.data(), c.n));
        EXPECT_EQ(p->getLength(), c.n);
        EXPECT_LT(p->getDegree(), c.n);
        for (uint32_t i = 0; i < c.n; ++i) {
            EXPECT_TRUE(equal(p->evaluate(xs[i]), ys[i]));
        }
        EXPECT_EQ(digest(*p, c.n), c.expected);
    }
}

// zerofierPolynomial over n points: the monic polynomial of degree n vanishing on them, in a buffer
// of n + 1 coefficients. 10 and 18 are the FflonkProver's ZTS2 and ZT.
TEST(PolynomialTest, zerofier_vanishes_on_the_points) {
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
        SCOPED_TRACE("n = " + std::to_string(c.n));
        Column xs = random.column(c.n);
        const std::unique_ptr<Poly> p(Poly::zerofierPolynomial(xs.data(), c.n));
        EXPECT_EQ(p->getLength(), c.n + 1);
        EXPECT_EQ(p->getDegree(), c.n);
        EXPECT_TRUE(equal(p->coef[c.n], E.fr.one()));
        for (const FrElement &x : xs) {
            EXPECT_TRUE(E.fr.isZero(p->evaluate(x)));
        }
        EXPECT_EQ(digest(*p, c.n + 1), c.expected);
    }
}

} // namespace
