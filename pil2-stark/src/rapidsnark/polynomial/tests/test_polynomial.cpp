// Tests for rapidsnark's Polynomial over BN128's scalar field, for the functions that grow, divide
// or swap a polynomial's buffer:
// - each result is checked for what it is: quotients and remainders, products, interpolants and
//   zerofiers, and what is left in a reserved buffer;
// - each result (its length, its degree and its coefficients) is also pinned by a Keccak-256 digest
//   to what the code returned when these tests were written, but for the length and the degree of a
//   polynomial that grows, which did not follow its buffer then; quirks included: a polynomial on a
//   reserved buffer moves to a buffer of its own when it grows or divBy or divByVanishing divide it;
// - the interpolations and zerofiers have the FflonkProver's sizes: R0, R1 and R2 interpolate 8, 4
//   and 6 points, and ZT and ZTS2 vanish on 18 and 10.
// divByMonic and CPolynomial::getPolynomial are also checked against the code they replaced, kept
// here as it was.
// Under AddressSanitizer they also show that these functions free what they allocate, and nothing
// else, on owned and reserved buffers alike.
#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
#include "cpolynomial.hpp"
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
// coef[0], each as 32 canonical little-endian bytes.
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

// divByMonic(m, β) on a = q·(X^m - β) + r, deg r < m: the quotient q, in place, on an owned buffer
// and on a reserved one, and false, for r is not zero. The digests are those of the quotient that
// divByMonic left in a buffer of its own before it divided in place. (4, 3) and (6, 5) have
// deg a = 2m - 1, the least that one took.
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
        EXPECT_FALSE(owned.divByMonic(c.m, beta));
        EXPECT_EQ(owned.getLength(), length);
        EXPECT_EQ(owned.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(owned, q, length));
        EXPECT_EQ(digest(owned, length), c.expected);

        Column reserved(length);
        Poly borrowed(E, reserved.data(), length);
        load(borrowed, a);
        EXPECT_FALSE(borrowed.divByMonic(c.m, beta));
        EXPECT_EQ(borrowed.coef, reserved.data());
        EXPECT_EQ(borrowed.getLength(), length);
        EXPECT_EQ(borrowed.getDegree(), c.qDegree);
        EXPECT_TRUE(holds(borrowed, q, length));
        EXPECT_EQ(digest(borrowed, length), c.expected);
    }
}

// The OpenMP threads, for the scope of the object.
class Threads {
public:
    explicit Threads(int n) : saved(omp_get_max_threads()) {
        omp_set_num_threads(n);
    }

    ~Threads() {
        omp_set_num_threads(saved);
    }

private:
    const int saved;
};

// The degree that fixDegree finds in these coefficients.
uint64_t degreeOf(const Column &coefs) {
    uint64_t degree = coefs.size() - 1;
    while (degree != 0 && E.fr.isZero(coefs[degree])) {
        --degree;
    }
    return degree;
}

// divByMonic before it divided in place, the reference for the quotient: its loops as they were,
// over the coefficients of a polynomial of degree d >= 2m - 1 (it wrote below its buffer
// otherwise), into a new column of the same length.
Column divByMonicBefore(const Column &coef, uint64_t d, uint32_t m, FrElement beta) {
    Column polResult(coef.size(), E.fr.zero());
    Column bArr(m);

    for (uint32_t i = 0; i < m; i++) {
        polResult[(d - i) - m] = coef[d - i];
        bArr[i] = coef[d - i];
    }

    uint32_t nThreads = m;
    #pragma omp parallel for
    for(uint32_t k = 0; k<nThreads;k++) {
        for (int i = d - 2 * m - k; i >= 0; i = i - nThreads) {
            if(i<0) break;
            uint32_t idx = k;
            bArr[idx] = E.fr.add(coef[i + m], E.fr.mul(bArr[idx], beta));
            polResult[i] = bArr[idx];
        }
    }

    return polResult;
}

// divByMonic against the one it replaced (divByMonicBefore), coefficient by coefficient over the
// whole buffer, and the length and the degree, wherever that one was defined (deg a >= 2m - 1);
// below, against the quotient written out, q_j = a_{j+m}. Whether it is exact against the remainder
// a_j + β·q_j (j < m) written out. For m = 1 to 9, a = q·(X^m - β) + ρ with q of 1, m, 2m - 1 and
// 2m coefficients, of a block, one more and one less, two and three blocks around their
// boundaries, and 2^18 + 3; q random, all ones, or zero but its top and bottom; β random, 0, 1 or
// -1; ρ zero, or ρ·X^j for a j below m (not exact) or in the top block (not exact unless β = 0),
// which runs through every carry. On 1 thread (a single block), and on 2 and 32 threads when q
// spans more than one block; and on a reserved buffer, which it keeps.
TEST(PolynomialTest, div_by_monic_is_the_division_it_replaces) {
    Random random(6002);
    const FrElement betas[] = {random.element(), E.fr.zero(), E.fr.one(), E.fr.negOne()};
    for (uint64_t m = 1; m <= 9; ++m) {
        const uint64_t block = Poly::divByMonicBlockLength(m);
        ASSERT_EQ(block % m, 0u);
        ASSERT_GE(block, 4096 - m);
        ASSERT_LE(block, 4096u);
        const uint64_t lengths[] = {1,         m,         2 * m - 1, 2 * m,     block - 1, block,     block + 1,
                                    2 * block - 1, 2 * block, 2 * block + 1, 3 * block + m, (uint64_t(1) << 18) + 3};
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
                for (int b = 0; b < (large ? 1 : 4); ++b) {
                    const FrElement &beta = betas[b];
                    const uint64_t d = qLength - 1 + m;
                    for (int rho = 0; rho < 3; ++rho) {
                        SCOPED_TRACE("m = " + std::to_string(m) + ", q of " + std::to_string(qLength) +
                                     " coefficients, kind " + std::to_string(kind) + ", beta " +
                                     std::to_string(b) + ", rho " + std::to_string(rho));
                        Column a = multiple(q, m, beta, {}, d + 3);
                        if (rho == 1) {
                            a[m - 1] = E.fr.add(a[m - 1], random.element());
                        } else if (rho == 2) {
                            a[d - 1] = E.fr.add(a[d - 1], E.fr.one());
                        }

                        Column quotient(a.size(), E.fr.zero());
                        if (d >= 2 * m - 1) {
                            quotient = divByMonicBefore(a, d, static_cast<uint32_t>(m), beta);
                        } else {
                            for (uint64_t j = 0; j <= d - m; ++j) {
                                quotient[j] = a[j + m];
                            }
                        }
                        bool exact = true;
                        for (uint64_t j = 0; j < m; ++j) {
                            exact = exact && E.fr.isZero(E.fr.add(a[j], E.fr.mul(beta, quotient[j])));
                        }
                        if (rho == 0) {
                            ASSERT_TRUE(exact);
                            ASSERT_TRUE(same(quotient.data(), padded(q, a.size())));
                        } else if (rho == 1) {
                            ASSERT_FALSE(exact);
                        } else {
                            ASSERT_EQ(exact, E.fr.isZero(beta) && d - 1 >= m);
                        }

                        for (int t : {1, 2, 32}) {
                            if (t != 32 && qLength <= block) {
                                continue;
                            }
                            SCOPED_TRACE(std::to_string(t) + " threads");
                            const Threads threads(t);
                            Poly owned(E, a.size());
                            load(owned, a);
                            ASSERT_EQ(owned.getDegree(), d);
                            const FrElement *buffer = owned.coef;
                            ASSERT_EQ(owned.divByMonic(m, beta), exact);
                            ASSERT_EQ(owned.coef, buffer);
                            ASSERT_EQ(owned.getLength(), a.size());
                            ASSERT_EQ(owned.getDegree(), degreeOf(quotient));
                            ASSERT_TRUE(same(owned.coef, quotient));
                        }

                        Column reserved(a.size());
                        Poly borrowed(E, reserved.data(), reserved.size());
                        load(borrowed, a);
                        ASSERT_EQ(borrowed.divByMonic(m, beta), exact);
                        ASSERT_EQ(borrowed.coef, reserved.data());
                        ASSERT_EQ(borrowed.getLength(), a.size());
                        ASSERT_EQ(borrowed.getDegree(), degreeOf(quotient));
                        ASSERT_TRUE(same(reserved.data(), quotient));
                    }
                }
            }
        }
    }
}

// The message of the std::runtime_error that call throws, or "" if it throws none.
template <typename Call>
std::string runtimeError(Call call) {
    try {
        call();
    } catch (const std::runtime_error &e) {
        return e.what();
    }
    return "";
}

// A divisor of degree 0 or above the polynomial's.
TEST(PolynomialTest, div_by_monic_refuses_a_divisor_of_degree_0_or_too_high) {
    Random random(0xdeb);
    const FrElement beta = random.element();
    Column coefs = random.column(8);
    coefs.back() = E.fr.one();
    Poly a(E, coefs.size());
    load(a, coefs);
    ASSERT_EQ(a.getDegree(), 7u);
    const std::string degree0 = "divByMonic: X^0 - beta is not of degree at least 1";
    const std::string tooHigh = "divByMonic: X^m - beta is of a degree above the polynomial's";
    EXPECT_EQ(runtimeError([&] { a.divByMonic(0, beta); }), degree0);
    EXPECT_EQ(runtimeError([] { Poly::divByMonicBlockLength(0); }), degree0);
    EXPECT_EQ(runtimeError([&] { a.divByMonic(8, beta); }), tooHigh);
    EXPECT_TRUE(same(a.coef, coefs));
    Poly constant(E, 4);
    constant.coef[0] = beta;
    constant.fixDegree();
    EXPECT_EQ(runtimeError([&] { constant.divByMonic(1, beta); }), tooHigh);
}

// byXSubValue(x): p·(X - x). On a reserved buffer it leaves -x·p there (it scales p in place) and
// moves to a buffer of its own. When the top coefficient is not zero the buffer and the length grow
// by one, and the next product grows from there.
TEST(PolynomialTest, by_x_sub_value_multiplies_by_x_minus_value) {
    Random random(0xb75);
    const Column xs = random.column(3);
    FrElement x1 = xs[1];
    FrElement x2 = xs[2];
    const Column linear = {E.fr.neg(xs[0]), E.fr.one()};
    // (X - x0)(X - x1)
    const Column quadratic = {E.fr.mul(xs[0], xs[1]), E.fr.neg(E.fr.add(xs[0], xs[1])), E.fr.one()};
    const char *const cubic = "027fc3f4bfe89323cb3bdfe2730a52bc44d9ec8144281fab9d9b6da6e5b34408";
    const char *const grown = "bedfdbc13193caaa248bab75907a6de9b9c9aedf2a95fa83373a022a7c6eefbf";

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
    EXPECT_EQ(full.getLength(), 3u);
    EXPECT_EQ(full.getDegree(), 2u);
    EXPECT_TRUE(holds(full, quadratic, 3));
    EXPECT_EQ(digest(full, 3), grown);
    // The cubic again, grown from two coefficients.
    full.byXSubValue(x2);
    EXPECT_EQ(full.getLength(), 4u);
    EXPECT_EQ(full.getDegree(), 3u);
    EXPECT_EQ(digest(full, 4), cubic);

    Column reservedFull(2);
    Poly borrowedFull(E, reservedFull.data(), 2);
    load(borrowedFull, linear);
    borrowedFull.byXSubValue(x1);
    EXPECT_EQ(borrowedFull.getLength(), 3u);
    EXPECT_EQ(borrowedFull.getDegree(), 2u);
    EXPECT_TRUE(holds(borrowedFull, quadratic, 3));
    EXPECT_TRUE(same(reservedFull.data(), scaled(linear, E.fr.neg(x1))));
    EXPECT_EQ(digest(borrowedFull, 3), grown);
}

// byXNSubValue(n, v): p·(X^n + v), for it adds v·p (byXSubValue adds -x·p) to X^n·p, as snarkjs's
// byXNSubValue does and says (polynomial.js: "Multiply current polynomial by the polynomial
// (X^n + value)"), whatever the name. On a reserved buffer it leaves v·p there and moves to a
// buffer of its own. When the product does not fit, the buffer and the length grow by n, and the
// next product grows from there. The expected values are the products written out: until it
// detached the buffer it takes, it freed that buffer and read it back.
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
    EXPECT_EQ(full.getLength(), 5u);
    EXPECT_EQ(full.getDegree(), 4u);
    EXPECT_TRUE(holds(full, first, 5));
    full.byXNSubValue(1, v2);
    EXPECT_EQ(full.getLength(), 6u);
    EXPECT_EQ(full.getDegree(), 5u);
    EXPECT_TRUE(holds(full, second, 6));

    Column reservedFull(3);
    Poly borrowedFull(E, reservedFull.data(), 3);
    load(borrowedFull, p);
    borrowedFull.byXNSubValue(2, v1);
    EXPECT_EQ(borrowedFull.getLength(), 5u);
    EXPECT_EQ(borrowedFull.getDegree(), 4u);
    EXPECT_TRUE(holds(borrowedFull, first, 5));
    EXPECT_TRUE(same(reservedFull.data(), scaled(p, v1)));
}

// add(b) and addBlinding(b, s), a + b and a + s·b: in place when b is not longer; otherwise into a
// buffer of b's length, which becomes its length: the sum is then the one computed in place, digest
// included. A reserved buffer is then left as it was.
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
    EXPECT_EQ(owned.getLength(), 5u);
    EXPECT_EQ(owned.getDegree(), 4u);
    EXPECT_TRUE(holds(owned, sum, 5));
    EXPECT_EQ(digest(owned, 5), "423376dca8c902cd6e80f29b506745c9186e1a34215434c24a0df2efb43f75e2");

    Column reserved(3);
    Poly borrowed(E, reserved.data(), 3);
    load(borrowed, a);
    borrowed.add(pb);
    EXPECT_EQ(borrowed.getLength(), 5u);
    EXPECT_EQ(borrowed.getDegree(), 4u);
    EXPECT_TRUE(holds(borrowed, sum, 5));
    EXPECT_TRUE(same(reserved.data(), a));
    EXPECT_EQ(digest(borrowed, 5), "423376dca8c902cd6e80f29b506745c9186e1a34215434c24a0df2efb43f75e2");

    Poly blindedInPlace(E, 5);
    load(blindedInPlace, a);
    blindedInPlace.addBlinding(pb, s);
    EXPECT_EQ(blindedInPlace.getDegree(), 4u);
    EXPECT_TRUE(holds(blindedInPlace, blinded, 5));
    EXPECT_EQ(digest(blindedInPlace, 5), "0d2da86669a5459b891eb01aa55d1c4331cb8fb338c023706d876e3c7be35d8b");

    Poly blindedOwned(E, 3);
    load(blindedOwned, a);
    blindedOwned.addBlinding(pb, s);
    EXPECT_EQ(blindedOwned.getLength(), 5u);
    EXPECT_EQ(blindedOwned.getDegree(), 4u);
    EXPECT_TRUE(holds(blindedOwned, blinded, 5));
    EXPECT_EQ(digest(blindedOwned, 5), "0d2da86669a5459b891eb01aa55d1c4331cb8fb338c023706d876e3c7be35d8b");

    Column reservedBlinded(3);
    Poly borrowedBlinded(E, reservedBlinded.data(), 3);
    load(borrowedBlinded, a);
    borrowedBlinded.addBlinding(pb, s);
    EXPECT_EQ(borrowedBlinded.getLength(), 5u);
    EXPECT_EQ(borrowedBlinded.getDegree(), 4u);
    EXPECT_TRUE(holds(borrowedBlinded, blinded, 5));
    EXPECT_TRUE(same(reservedBlinded.data(), a));
    EXPECT_EQ(digest(borrowedBlinded, 5), "0d2da86669a5459b891eb01aa55d1c4331cb8fb338c023706d876e3c7be35d8b");
}

// sub(b) and subBlinding(b, s), a - b and a - s·b: in place when b is not longer; otherwise into a
// buffer of b's length, which becomes its length, as add does (they wrote past a's buffer): the
// difference is then the one computed in place. A reserved buffer is then left as it was.
TEST(PolynomialTest, sub_and_sub_blinding_grow_into_a_new_buffer) {
    Random random(0x5b0);
    const Column a = random.column(3);
    const Column b = random.column(5);
    FrElement s = random.element();
    Column difference(5), blinded(5);
    for (uint64_t i = 0; i < 5; ++i) {
        const FrElement ai = i < 3 ? a[i] : E.fr.zero();
        difference[i] = E.fr.sub(ai, b[i]);
        blinded[i] = E.fr.sub(ai, E.fr.mul(s, b[i]));
    }
    Poly pb(E, 5);
    load(pb, b);

    for (int blinding = 0; blinding < 2; ++blinding) {
        SCOPED_TRACE(blinding ? "subBlinding" : "sub");
        const Column &expected = blinding ? blinded : difference;
        auto subtract = [&](Poly &p) {
            if (blinding) {
                p.subBlinding(pb, s);
            } else {
                p.sub(pb);
            }
        };

        Poly inPlace(E, 5);
        load(inPlace, a);
        subtract(inPlace);
        EXPECT_EQ(inPlace.getLength(), 5u);
        EXPECT_EQ(inPlace.getDegree(), 4u);
        EXPECT_TRUE(holds(inPlace, expected, 5));

        Poly owned(E, 3);
        load(owned, a);
        subtract(owned);
        EXPECT_EQ(owned.getLength(), 5u);
        EXPECT_EQ(owned.getDegree(), 4u);
        EXPECT_TRUE(holds(owned, expected, 5));
        EXPECT_EQ(digest(owned, 5), digest(inPlace, 5));

        Column reserved(3);
        Poly borrowed(E, reserved.data(), 3);
        load(borrowed, a);
        subtract(borrowed);
        EXPECT_NE(borrowed.coef, reserved.data());
        EXPECT_EQ(borrowed.getLength(), 5u);
        EXPECT_EQ(borrowed.getDegree(), 4u);
        EXPECT_TRUE(holds(borrowed, expected, 5));
        EXPECT_TRUE(same(reserved.data(), a));
    }
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
// the buffer, on one thread (the tests below run it on more).
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

// fastDivByVanishing(buffer, m, β) on a = q·(X^m - β) + r, a of `length` coefficients and degree
// qDegree + m: on `threads` threads, `runs` times, the quotient q, which divByVanishing gives too,
// and a copied to the buffer. Returns in how many runs the quotient was not q.
int fastDivByVanishingMisses(uint32_t m, uint64_t length, uint64_t qDegree, int threads, int runs, Random &random) {
    const Column q = random.column(qDegree + 1);
    const Column r = random.column(m);
    const FrElement beta = random.element();
    const Column a = multiple(q, m, beta, r, length);

    Poly reference(E, length);
    load(reference, a);
    const std::unique_ptr<Poly> remainder(reference.divByVanishing(m, beta));
    EXPECT_TRUE(holds(reference, q, length));

    const Threads scope(threads);
    int misses = 0;
    for (int run = 0; run < runs; ++run) {
        Poly p(E, length);
        load(p, a);
        Column buffer(length);
        p.fastDivByVanishing(buffer.data(), m, beta);
        const bool right = p.getLength() == length && p.getDegree() == qDegree && holds(p, q, length) &&
                           same(buffer.data(), a);
        misses += right ? 0 : 1;
    }
    return misses;
}

// On 1, 2, 4 and 32 threads, 50 times each, chunks of many rows per thread: the quotient every
// time. Every thread computed the rows below the chunks, and read the chunks' lowest rows, while
// others were already correcting them, and the quotient was wrong about once in 8 runs on 4
// threads.
TEST(PolynomialTest, fast_div_by_vanishing_is_the_quotient_on_any_threads) {
    struct Case {
        uint32_t m;
        uint64_t length;
        uint64_t qDegree;
    };
    const Case cases[] = {{1, 1024, 1000}, {3, 2048, 2000}, {4, 4096, 4000}, {8, 1 << 14, 16000}};
    Random random(0xfd8);
    for (const Case &c : cases) {
        for (int t : {1, 2, 4, 32}) {
            EXPECT_EQ(fastDivByVanishingMisses(c.m, c.length, c.qDegree, t, 50, random), 0)
                << "m = " << c.m << ", length " << c.length << ", " << t << " threads";
        }
    }
}

// A degree of length - 1: it fixed the quotient's degree from coef[deg + 1], past the buffer.
TEST(PolynomialTest, fast_div_by_vanishing_reads_nothing_past_the_buffer) {
    Random random(0xfd9);
    EXPECT_EQ(fastDivByVanishingMisses(1, 1024, 1022, 1, 1, random), 0);
    EXPECT_EQ(fastDivByVanishingMisses(4, 4096, 4091, 1, 1, random), 0);
}

// Fewer rows of m than threads, or less than a row: fewer chunks, or none (it wrote far past the
// buffer, sizing a chunk to no row).
TEST(PolynomialTest, fast_div_by_vanishing_has_fewer_rows_than_threads) {
    struct Case {
        uint32_t m;
        uint64_t length;
        uint64_t qDegree;
    };
    const Case cases[] = {{1, 2, 0}, {3, 4, 0}, {4, 7, 2}, {3, 40, 30}, {4, 64, 59}, {2, 20, 17}};
    Random random(0xfda);
    for (const Case &c : cases) {
        for (int t : {1, 2, 4, 32}) {
            EXPECT_EQ(fastDivByVanishingMisses(c.m, c.length, c.qDegree, t, 10, random), 0)
                << "m = " << c.m << ", length " << c.length << ", " << t << " threads";
        }
    }
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

// fromReservedBuffer(buffer, n): a polynomial over the n coefficients the buffer holds, which it
// neither clears nor copies (the constructor on a reserved buffer clears them), with its degree
// fixed, and which leaves the buffer to its owner; what is beyond n is not touched. Then
// divByMonic on it, in the same buffer.
TEST(PolynomialTest, from_reserved_buffer_takes_the_coefficients_in_it) {
    Random random(0xf2b);
    const FrElement beta = random.element();
    const Column q = random.column(9);
    // q·(X^3 - β) in the first 12 coefficients, then stale ones at 12 and 15.
    Column a = multiple(q, 3, beta, {}, 16);
    a[12] = random.element();
    a[15] = random.element();
    Column buffer = a;
    {
        const std::unique_ptr<Poly> p(Poly::fromReservedBuffer(E, buffer.data(), 12));
        EXPECT_EQ(p->coef, buffer.data());
        EXPECT_EQ(p->getLength(), 12u);
        EXPECT_EQ(p->getDegree(), 11u);
        EXPECT_TRUE(same(buffer.data(), a));
        EXPECT_TRUE(p->divByMonic(3, beta));
        EXPECT_EQ(p->coef, buffer.data());
        EXPECT_EQ(p->getDegree(), 8u);
        EXPECT_TRUE(holds(*p, q, 12));
    }
    EXPECT_TRUE(same(buffer.data() + 12, Column(a.begin() + 12, a.end())));

    // Zeros at the top: the degree is fixed below them; none at all: degree 0, nothing read.
    const std::unique_ptr<Poly> zeros(Poly::fromReservedBuffer(E, buffer.data() + 9, 3));
    EXPECT_EQ(zeros->getLength(), 3u);
    EXPECT_EQ(zeros->getDegree(), 0u);
    const std::unique_ptr<Poly> empty(Poly::fromReservedBuffer(E, buffer.data(), 0));
    EXPECT_EQ(empty->getLength(), 0u);
    EXPECT_EQ(empty->getDegree(), 0u);
    EXPECT_EQ(empty->coef, buffer.data());
}

// Polynomials of these degrees, -1 for no polynomial and -2 for the zero polynomial, each in two
// coefficients more than its degree, which are zero, added to a CPolynomial at their positions.
struct Packing {
    std::vector<std::unique_ptr<Poly>> owned;
    std::vector<Poly *> polys;
    std::unique_ptr<CPolynomial<Engine>> cpolynomial;

    Packing(const std::vector<int64_t> &degrees, Random &random)
        : cpolynomial(new CPolynomial<Engine>(E, static_cast<int>(degrees.size()))) {
        for (uint64_t i = 0; i < degrees.size(); ++i) {
            if (degrees[i] == -1) {
                polys.push_back(nullptr);
                continue;
            }
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
            polys.push_back(&p);
            cpolynomial->addPolynomial(static_cast<int>(i), &p);
        }
    }
};

// The packing of CPolynomial written out: coefficient c·k + i of f is coefficient c of p_i up to
// its degree, zero above it and where no p_i was added, in 1 + max_i(k·deg p_i + i) coefficients,
// over the positions with a polynomial.
Column interleaved(const std::vector<Poly *> &polys) {
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
// their count), on 1, 2 and 32 threads. k = 1 and 12, constants, zeros, mixed degrees, a degree
// bound a power of two and above deg f (zero top polynomials), a position with no polynomial, and
// f of 12,286 coefficients. It writes into a buffer of stale data, and nothing beyond the count.
TEST(CPolynomialTest, get_coefficients_is_the_packing) {
    Random random(0xc0f);
    const std::vector<std::vector<int64_t>> cases = {
        {0}, {1}, {4}, {16}, {1000}, {-2},
        {0, 0}, {3, 0}, {0, 3}, {-2, -2}, {1, -1},
        {0, -2, -2}, {5, -1, 2}, {4095, 4094, 100},
        {7, 2, 0, 5},
        {63, 62, 7, 0, -2, 63, 62, 7, 0, -2, 1, -1},
    };
    for (const std::vector<int64_t> &degrees : cases) {
        std::string name = "degrees";
        for (int64_t degree : degrees) {
            name += " " + std::to_string(degree);
        }
        SCOPED_TRACE(name);
        const Packing packing(degrees, random);
        const Column expected = interleaved(packing.polys);
        ASSERT_EQ(packing.cpolynomial->getDegree(), expected.size() - 1);

        const Column stale = random.column(expected.size() + 5);
        for (int t : {1, 2, 32}) {
            SCOPED_TRACE(std::to_string(t) + " threads");
            const Threads threads(t);
            Column buffer = stale;
            EXPECT_EQ(packing.cpolynomial->getCoefficients(buffer.data()), expected.size());
            EXPECT_TRUE(same(buffer.data(), expected));
            EXPECT_TRUE(same(buffer.data() + expected.size(), Column(stale.begin() + expected.size(), stale.end())));
        }
    }
}

// CPolynomial::getPolynomial before it packed in one pass, the reference: its body as it was, over
// the polynomials and the degree bound (its getDegree()) of a CPolynomial.
Poly *getPolynomialBefore(const std::vector<Poly *> &polynomials, uint64_t degreeBound, FrElement *reservedBuffer) {
    const int n = static_cast<int>(polynomials.size());
    std::vector<u_int64_t> degrees(n);

    for (int i = 0; i < n; i++) {
        degrees[i] = polynomials[i] == NULL ? 0 : polynomials[i]->getDegree();
    }

    u_int64_t maxDegree = degreeBound;
    u_int64_t lengthBuffer = std::pow(2, ((u_int64_t)log2(maxDegree - 1)) + 1);
    Polynomial<Engine> *polynomial = new Polynomial<Engine>(E, reservedBuffer, lengthBuffer);

    #pragma omp parallel for
    for (u_int64_t i = 0; i < maxDegree+1; i++) {
        for (int j = 0; j < n; j++) {
            if (polynomials[j] != NULL) {
                if (i <= degrees[j]) polynomial->coef[i * n + j] = polynomials[j]->coef[i];
            }
        }
    }

    polynomial->fixDegree();

    return polynomial;
}

// getPolynomial against getPolynomialBefore on a CPolynomial of polynomials of these degrees (see
// Packing), on the same stale buffer, on 1, 2 and 32 threads: the buffer it returns, the length,
// the degree, and the whole buffer, before, over and past the length.
void expectGetPolynomialAsBefore(const std::vector<int64_t> &degrees, Random &random) {
    const Packing packing(degrees, random);
    const uint64_t bound = packing.cpolynomial->getDegree();
    ASSERT_GE(bound, 2u);
    // The length is below 2·bound.
    const Column stale = random.column(2 * bound + 8);
    Column before = stale;
    const std::unique_ptr<Poly> reference(getPolynomialBefore(packing.polys, bound, before.data()));
    ASSERT_LE(reference->getLength(), 2 * bound);
    for (int t : {1, 2, 32}) {
        SCOPED_TRACE(std::to_string(t) + " threads");
        const Threads threads(t);
        Column after = stale;
        const std::unique_ptr<Poly> f(packing.cpolynomial->getPolynomial(after.data()));
        ASSERT_EQ(f->coef, after.data());
        ASSERT_EQ(f->getLength(), reference->getLength());
        ASSERT_EQ(f->getDegree(), reference->getDegree());
        ASSERT_TRUE(same(after.data(), before));
    }
}

// getPolynomial against the one it replaced. FFLONK's shapes, for a domain of D = 8, 64 and 1024
// and at their top degrees and below: C0 (setup), 8 polynomials of degree below D; C1, a, b and c
// of degree below D and T0 below 2D - 2; C2, z below D + 3, T1 below D + 2 and T2 below 3D. Then
// k = 1 to 5 with random degrees, positions with no polynomial, zero polynomials and constants,
// and degree bounds powers of two (the top coefficient is then past the length) and one more.
TEST(CPolynomialTest, get_polynomial_is_the_packing_it_replaces) {
    Random random(0x9e7);
    for (int64_t D : {8, 64, 1024}) {
        const std::vector<std::vector<int64_t>> shapes = {
            std::vector<int64_t>(8, D - 1),
            {D - 1, D - 2, D - 1, D - 3, D / 2, D - 1, D - 1, D - 2},
            {D - 1, D - 1, D - 1, 2 * D - 3},
            {D - 2, D - 1, D - 3, D},
            {D + 2, D + 1, 3 * D - 1},
            {D + 1, D, 3 * D - 3},
        };
        for (const std::vector<int64_t> &degrees : shapes) {
            std::string name = "D = " + std::to_string(D) + ", degrees";
            for (int64_t degree : degrees) {
                name += " " + std::to_string(degree);
            }
            SCOPED_TRACE(name);
            expectGetPolynomialAsBefore(degrees, random);
            if (HasFatalFailure()) {
                return;
            }
        }
    }

    // Degree bounds 2, a power of two (16) and one more, k·d + i.
    const std::vector<std::vector<int64_t>> edges = {
        {2}, {16}, {17}, {1, 0}, {8, -1}, {8, 0}, {-2, 8}, {5, -1, 0}, {5, 0, -2}, {4, 3, 3, 3},
    };
    for (const std::vector<int64_t> &degrees : edges) {
        std::string name = "degrees";
        for (int64_t degree : degrees) {
            name += " " + std::to_string(degree);
        }
        SCOPED_TRACE(name);
        expectGetPolynomialAsBefore(degrees, random);
        if (HasFatalFailure()) {
            return;
        }
    }

    for (int trial = 0; trial < 200; ++trial) {
        const uint64_t k = 1 + trial % 5;
        std::vector<int64_t> degrees(k);
        bool some = false;
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t r = random.element().v[0] % 10;
            degrees[i] = r == 0 ? -1 : r == 1 ? -2 : static_cast<int64_t>(random.element().v[0] % 300);
            some = some || degrees[i] >= 0;
        }
        degrees[k - 1] = some ? degrees[k - 1] : 3;
        uint64_t bound = 0;
        for (uint64_t i = 0; i < k; ++i) {
            if (degrees[i] != -1) {
                bound = std::max<uint64_t>(bound, k * std::max<int64_t>(degrees[i], 0) + i);
            }
        }
        if (bound < 2) {
            continue;
        }
        std::string name = "trial " + std::to_string(trial) + ", degrees";
        for (int64_t degree : degrees) {
            name += " " + std::to_string(degree);
        }
        SCOPED_TRACE(name);
        expectGetPolynomialAsBefore(degrees, random);
        if (HasFatalFailure()) {
            return;
        }
    }
}

} // namespace
