#include "pilfflonk_shplonk_prover.hpp"

#include <gmp.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "pilfflonk_lde.hpp"
#include "thread_utils.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// mpz_divisible_ui_p and mpz_divexact_ui take the divisor as an unsigned long.
static_assert(sizeof(unsigned long) == sizeof(uint64_t), "a 64-bit unsigned long");

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string("ShplonkProver") + function + ": " + message);
}

std::string name(uint64_t i) {
    return "f_" + std::to_string(i);
}

// An mpz_t that clears itself.
struct Mpz {
    mpz_t v;
    Mpz() { mpz_init(v); }
    ~Mpz() { mpz_clear(v); }
    Mpz(const Mpz &) = delete;
    Mpz &operator=(const Mpz &) = delete;
};

// r - 1, the order of the multiplicative group of Fr.
void rMinusOne(Mpz &out) {
    Engine &E = Engine::engine;
    E.fr.toMpz(out.v, E.fr.negOne());
}

// base^e by ffiasm's square-and-multiply, which reads e as little-endian bytes.
FrElement power(const FrElement &base, const Mpz &e) {
    uint8_t bytes[32] = {};
    if (mpz_sizeinbase(e.v, 256) > sizeof(bytes)) {
        throw std::logic_error("ShplonkProver: an exponent above 2^256");
    }
    mpz_export(bytes, nullptr, -1, 1, -1, 0, e.v);
    FrElement result;
    Engine::engine.fr.exp(result, base, bytes, sizeof(bytes));
    return result;
}

FrElement power(const FrElement &base, uint64_t e) {
    uint8_t bytes[sizeof(e)];
    for (size_t i = 0; i < sizeof(e); ++i) {
        bytes[i] = static_cast<uint8_t>(e >> (8 * i));
    }
    FrElement result;
    Engine::engine.fr.exp(result, base, bytes, sizeof(bytes));
    return result;
}

bool dividesRMinusOne(uint64_t n) {
    Mpz m;
    rMinusOne(m);
    return mpz_divisible_ui_p(m.v, n) != 0;
}

// 5^((r-1)/n), a primitive n-th root of unity, for n dividing r - 1: w_k for n = k and ω_{kN} for
// n = kN (spec A.2.5). 5 is the smallest quadratic non-residue, the generator ffiasm's FFT and
// ffjavascript raise, so ω_N is the generator of H they use.
FrElement rootOfUnity(uint64_t n) {
    Mpz e;
    rMinusOne(e);
    mpz_divexact_ui(e.v, e.v, n);
    FrElement five;
    Engine::engine.fr.fromUI(five, 5);
    return power(five, e);
}

// s mod n in [0, n), for |s| < n: the exponent of ω_n^s, since ω_n^n = 1.
uint64_t exponent(int64_t s, uint64_t n) {
    return s >= 0 ? static_cast<uint64_t>(s) : n - static_cast<uint64_t>(-s);
}

// 1/a_i for every a_i, none zero, with a single field inversion: Montgomery's trick, as
// rapidsnark's FflonkProver::batchInverse does it.
std::vector<FrElement> batchInverse(const std::vector<FrElement> &a) {
    Engine &E = Engine::engine;
    const size_t n = a.size();
    std::vector<FrElement> inverses(n);
    if (n == 0) {
        return inverses;
    }
    // products[i] = a_0·…·a_i
    std::vector<FrElement> products(n);
    products[0] = a[0];
    for (size_t i = 1; i < n; ++i) {
        E.fr.mul(products[i], products[i - 1], a[i]);
    }
    // inverse = 1/(a_0·…·a_i) going down.
    FrElement inverse;
    E.fr.inv(inverse, products[n - 1]);
    for (size_t i = n - 1; i > 0; --i) {
        E.fr.mul(inverses[i], inverse, products[i - 1]);
        E.fr.mul(inverse, inverse, a[i]);
    }
    inverses[0] = inverse;
    return inverses;
}

bool isZero(const Poly &p) {
    return p.getDegree() == 0 && Engine::engine.fr.isZero(p.coef[0]);
}

// a := a / (X^m - β), which must be exact: throws std::logic_error, naming `what`, otherwise.
// rapidsnark's divByMonic computes only the quotient, so the remainder a_j + β·q_j (j < m) is
// checked here. It also needs deg a >= m (it writes below its buffer otherwise) and a that owns its
// buffer (it swaps it for one it allocates, which a borrowed one would leak), and a's degree up to
// date.
void divideExactly(Poly &a, uint64_t m, const FrElement &beta, const std::string &what) {
    Engine &E = Engine::engine;
    if (a.getDegree() < m) {
        // The quotient is 0 and the remainder a itself.
        if (!isZero(a)) {
            throw std::logic_error("ShplonkProver: " + what + " is not divisible");
        }
        return;
    }
    const std::vector<FrElement> low(a.coef, a.coef + m);
    a.divByMonic(static_cast<uint32_t>(m), beta);
    for (uint64_t j = 0; j < m; ++j) {
        FrElement remainder;
        E.fr.mul(remainder, beta, a.coef[j]);
        E.fr.add(remainder, remainder, low[j]);
        if (!E.fr.isZero(remainder)) {
            throw std::logic_error("ShplonkProver: " + what + " is not divisible");
        }
    }
}

} // namespace

ShplonkProver::ShplonkProver(ShplonkOpening opening) {
#ifndef __USE_ASSEMBLY__
    throw std::runtime_error("ShplonkProver needs ffiasm's assembly backend, not built on this platform");
#endif
    Engine &E = Engine::engine;
    const char *function = "";
    const uint64_t nBits = opening.nBits;

    // Every check first: nothing is computed for an opening that is refused.
    if (opening.polynomials.empty()) {
        throw invalid(function, "no polynomials to open");
    }
    if (nBits > MAX_NBITS_EXT) {
        throw invalid(function, "nBits = " + std::to_string(nBits) + " exceeds " + std::to_string(MAX_NBITS_EXT) +
                                    ", the 2-adicity of r - 1");
    }
    if (E.fr.isZero(opening.xiSeed)) {
        throw invalid(function, "xiSeed is zero");
    }
    const uint64_t N = uint64_t(1) << nBits;
    uint64_t lcm = 1;
    for (uint64_t i = 0; i < opening.polynomials.size(); ++i) {
        const ShplonkPolynomial &f = opening.polynomials[i];
        const uint64_t k = f.components.size();
        if (k == 0) {
            throw invalid(function, name(i) + " has no components");
        }
        uint64_t maxLength = 0;
        for (uint64_t j = 0; j < k; ++j) {
            if (f.components[j] == nullptr) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) + " is null");
            }
            if (f.components[j]->getLength() == 0) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) + " has no coefficients");
            }
            maxLength = std::max(maxLength, f.components[j]->getLength());
        }
        if (k > static_cast<uint64_t>(INT_MAX) / maxLength) {
            throw invalid(function, name(i) + ": k = " + std::to_string(k) + " components of up to " +
                                        std::to_string(maxLength) + " coefficients exceed INT_MAX coefficients");
        }
        if (!dividesRMinusOne(k)) {
            throw invalid(function, name(i) + ": k = " + std::to_string(k) + " does not divide r - 1");
        }
        if (__builtin_ctzll(k) + nBits > MAX_NBITS_EXT) {
            throw invalid(function, name(i) + ": kN = " + std::to_string(k) + "·2^" + std::to_string(nBits) +
                                        " goes beyond the 2-adicity 2^" + std::to_string(MAX_NBITS_EXT) +
                                        " of r - 1");
        }
        if (f.offsets.empty()) {
            throw invalid(function, name(i) + " has no offsets");
        }
        std::vector<uint64_t> rows;
        for (int64_t s : f.offsets) {
            // Compared as unsigned: -s of INT64_MIN would overflow.
            if ((s >= 0 ? static_cast<uint64_t>(s) : uint64_t(0) - static_cast<uint64_t>(s)) >= N) {
                throw invalid(function, name(i) + ": offset " + std::to_string(s) + " is not below N = " +
                                            std::to_string(N) + " in absolute value");
            }
            rows.push_back(exponent(s, N));
        }
        std::sort(rows.begin(), rows.end());
        if (std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
            throw invalid(function, name(i) + ": two offsets are the same row modulo N = " + std::to_string(N));
        }
        if (k > static_cast<uint64_t>(INT_MAX) / f.offsets.size()) {
            throw invalid(function, name(i) + ": k·|O| = " + std::to_string(k) + "·" +
                                        std::to_string(f.offsets.size()) + " roots exceed INT_MAX");
        }
        const uint64_t gcd = std::gcd(lcm, k);
        if (lcm / gcd > UINT64_MAX / k) {
            throw invalid(function, "the lcm of every k exceeds 2^64");
        }
        lcm = lcm / gcd * k;
    }
    if (opening.powerW != lcm) {
        throw invalid(function, "powerW = " + std::to_string(opening.powerW) + " is not " + std::to_string(lcm) +
                                    ", the lcm of every k");
    }

    // The roots, spec A.2.5.
    challengeXi = power(opening.xiSeed, opening.powerW);
    const FrElement omegaN = rootOfUnity(N);
    fs.reserve(opening.polynomials.size());
    for (ShplonkPolynomial &f : opening.polynomials) {
        Entry entry;
        const uint64_t k = f.components.size();
        const FrElement wk = rootOfUnity(k);
        const FrElement omegaKN = rootOfUnity(k * N);
        const FrElement seed = power(opening.xiSeed, opening.powerW / k);
        for (uint64_t j = 0; j < k; ++j) {
            entry.maxComponentLength = std::max(entry.maxComponentLength, f.components[j]->getLength());
            // pack()'s count, CPolynomial's degree bound.
            entry.nCoefs = std::max(entry.nCoefs, k * f.components[j]->getDegree() + j + 1);
        }
        for (int64_t s : f.offsets) {
            // ξ·ω_N^s, and x_0 = xiSeed^(powerW/k)·ω_{kN}^s: x_0^k = ξ·ω_N^s.
            entry.points.push_back(E.fr.mul(challengeXi, power(omegaN, exponent(s, N))));
            FrElement x = E.fr.mul(seed, power(omegaKN, exponent(s, k * N)));
            for (uint64_t j = 0; j < k; ++j) {
                entry.roots.push_back(x);
                E.fr.mul(x, x, wk);
            }
        }
        entry.components = std::move(f.components);
        fs.push_back(std::move(entry));
    }

    // The evaluations, rapidsnark's fastEvaluate on each p_j.
    evals.resize(fs.size());
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        const uint64_t k = f.components.size();
        evals[i].resize(f.points.size() * k);
        for (uint64_t m = 0; m < f.points.size(); ++m) {
            for (uint64_t j = 0; j < k; ++j) {
                evals[i][m * k + j] = f.components[j]->fastEvaluate(f.points[m]);
            }
        }
    }
}

uint64_t ShplonkProver::scratchLength() const {
    uint64_t length = 0;
    for (const Entry &f : fs) {
        length = std::max(length, packedBufferLength(f.components.size(), f.maxComponentLength));
    }
    return length;
}

uint64_t ShplonkProver::workLength() const {
    uint64_t length = 0;
    for (const Entry &f : fs) {
        length = std::max({length, f.nCoefs, static_cast<uint64_t>(f.roots.size())});
    }
    return length;
}

std::unique_ptr<Poly> ShplonkProver::packed(uint64_t i, uint64_t length, FrElement *scratch) const {
    const Entry &f = fs[i];
    const uint64_t k = f.components.size();
    const uint64_t n = pack(f.components.data(), k, scratch, packedBufferLength(k, f.maxComponentLength));
    if (n != f.nCoefs) {
        throw std::logic_error("ShplonkProver: the components of " + name(i) + " changed after it was built");
    }
    std::unique_ptr<Poly> p(new Poly(Engine::engine, length));
    ThreadUtils::parcpy(p->coef, scratch, n * sizeof(FrElement), omp_get_max_threads());
    p->fixDegree();
    return p;
}

ShplonkProver::Interpolants ShplonkProver::interpolants() const {
    Engine &E = Engine::engine;
    Interpolants r;
    r.reserve(fs.size());
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        const uint64_t k = f.components.size();
        const uint64_t nRoots = f.roots.size();
        // f_i(x) = Σ_j p_j(ξ·ω_N^s)·x^j on each root x of offset s, by Horner's rule.
        std::vector<FrElement> xs = f.roots;
        std::vector<FrElement> ys(nRoots);
        for (uint64_t m = 0; m < f.points.size(); ++m) {
            for (uint64_t j = 0; j < k; ++j) {
                const FrElement &x = xs[m * k + j];
                FrElement value = E.fr.zero();
                for (uint64_t l = k; l > 0; --l) {
                    E.fr.mul(value, value, x);
                    E.fr.add(value, value, evals[i][m * k + l - 1]);
                }
                ys[m * k + j] = value;
            }
        }
        std::unique_ptr<Poly> ri;
        if (nRoots == 1) {
            // lagrangePolynomialInterpolation dereferences a null polynomial for a single point.
            ri.reset(new Poly(E, 1));
            ri->coef[0] = ys[0];
            ri->fixDegree();
        } else {
            ri.reset(Poly::lagrangePolynomialInterpolation(xs.data(), ys.data(), static_cast<uint32_t>(nRoots)));
        }
        r.push_back(std::move(ri));
    }
    return r;
}

std::unique_ptr<Poly> ShplonkProver::quotientW(const Interpolants &r, const FrElement &alpha) const {
    Engine &E = Engine::engine;
    if (r.size() != fs.size()) {
        throw invalid("::quotientW", std::to_string(r.size()) + " interpolants for " + std::to_string(fs.size()) +
                                         " polynomials");
    }
    // W's coefficients: max_i (nCoefs(i) - |T_i|), each (f_i - r_i)/Z_{T_i} having that many.
    uint64_t wLength = 1;
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const uint64_t nRoots = fs[i].roots.size();
        if (r[i] == nullptr || r[i]->getLength() > nRoots) {
            throw invalid("::quotientW", "r[" + std::to_string(i) + "] is not an interpolant of " + name(i));
        }
        if (fs[i].nCoefs > nRoots) {
            wLength = std::max(wLength, fs[i].nCoefs - nRoots);
        }
    }

    const std::unique_ptr<FrElement[]> scratch(new FrElement[scratchLength()]);
    // Every term fits in it, so add() never has to grow W (which it does without updating W's
    // length, and leaking a borrowed buffer).
    std::unique_ptr<Poly> W(new Poly(E, workLength()));
    FrElement alphaPower = E.fr.one();
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        const uint64_t k = f.components.size();
        // At least |T_i| coefficients: sub() writes as many as the longer operand has.
        std::unique_ptr<Poly> term = packed(i, std::max<uint64_t>(f.nCoefs, f.roots.size()), scratch.get());
        term->sub(*r[i]);
        // Z_{T_i}(X) = Π_{s in O_i} (X^k - ξ·ω_N^s): the roots of offset s are those of X^k - ξ·ω_N^s.
        for (const FrElement &point : f.points) {
            divideExactly(*term, k, point, name(i) + " - r_i");
        }
        term->mulScalar(alphaPower);
        W->add(*term);
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    if (W->getDegree() >= wLength) {
        throw std::logic_error("ShplonkProver: W has degree " + std::to_string(W->getDegree()) + ", not below " +
                               std::to_string(wLength));
    }
    return W;
}

std::unique_ptr<Poly> ShplonkProver::quotientWp(const Interpolants &r, const FrElement &alpha, const FrElement &y,
                                                const Poly &W) const {
    Engine &E = Engine::engine;
    if (r.size() != fs.size()) {
        throw invalid("::quotientWp", std::to_string(r.size()) + " interpolants for " + std::to_string(fs.size()) +
                                          " polynomials");
    }
    for (uint64_t i = 0; i < fs.size(); ++i) {
        if (r[i] == nullptr) {
            throw invalid("::quotientWp", "r[" + std::to_string(i) + "] is null");
        }
    }
    const uint64_t length = workLength();
    if (W.getDegree() >= length) {
        throw invalid("::quotientWp", "W has degree " + std::to_string(W.getDegree()) + ": it is not quotientW's");
    }

    // Z_{T_i}(y), and the scalars of L/Z_{T∖T_0}(y) (spec A.5's q_i):
    //     α^i·Z_{T∖T_i}(y)/Z_{T∖T_0}(y) for f_i - r_i(y), and Z_T(y)/Z_{T∖T_0}(y) for W,
    // with Z_T(y) = Π_i Z_{T_i}(y), repetitions included, and Z_{T∖T_i}(y) = Z_T(y)/Z_{T_i}(y).
    std::vector<FrElement> zi(fs.size());
    for (uint64_t i = 0; i < fs.size(); ++i) {
        FrElement z = E.fr.one();
        for (const FrElement &x : fs[i].roots) {
            E.fr.mul(z, z, E.fr.sub(y, x));
        }
        if (E.fr.isZero(z)) {
            throw std::runtime_error("ShplonkProver: y is a root of " + name(i));
        }
        zi[i] = z;
    }
    const std::vector<FrElement> ziInverse = batchInverse(zi);
    FrElement zT = E.fr.one();
    for (const FrElement &z : zi) {
        E.fr.mul(zT, zT, z);
    }
    // 1/Z_{T∖T_0}(y) = Π_{i>=1} 1/Z_{T_i}(y)
    FrElement scale = E.fr.one();
    for (uint64_t i = 1; i < fs.size(); ++i) {
        E.fr.mul(scale, scale, ziInverse[i]);
    }

    // L starts as -(Z_T(y)/Z_{T∖T_0}(y))·W, as long as every term, so that add() never grows it.
    const FrElement wScale = E.fr.neg(E.fr.mul(zT, scale));
    std::unique_ptr<Poly> L(new Poly(E, length));
    const uint64_t wCoefs = W.getDegree() + 1;
#pragma omp parallel for
    for (uint64_t m = 0; m < wCoefs; ++m) {
        E.fr.mul(L->coef[m], wScale, W.coef[m]);
    }
    L->fixDegree();

    const std::unique_ptr<FrElement[]> scratch(new FrElement[scratchLength()]);
    uint64_t maxCoefs = 1;
    FrElement alphaPower = E.fr.one();
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        maxCoefs = std::max(maxCoefs, f.nCoefs);
        std::unique_ptr<Poly> term = packed(i, f.nCoefs, scratch.get());
        FrElement ry = r[i]->evaluate(y);
        term->subScalar(ry);
        FrElement factor = E.fr.mul(E.fr.mul(alphaPower, zT), E.fr.mul(ziInverse[i], scale));
        term->mulScalar(factor);
        L->add(*term);
        E.fr.mul(alphaPower, alphaPower, alpha);
    }

    // W' = L/(Z_{T∖T_0}(y)·(X - y)): divByMonic(1, y) in place of pil-fflonk's divByXSubValue.
    divideExactly(*L, 1, y, "L");
    // deg L <= max_i deg f_i, so W' has at most max_i nCoefs(i) - 1 coefficients.
    const uint64_t wpLength = std::max<uint64_t>(1, maxCoefs - 1);
    if (L->getDegree() >= wpLength) {
        throw std::logic_error("ShplonkProver: W' has degree " + std::to_string(L->getDegree()) + ", not below " +
                               std::to_string(wpLength));
    }
    return L;
}

ShplonkProof ShplonkProver::open(const Srs &srs, Transcript &transcript) const {
    if (transcript.empty()) {
        throw invalid("::open", "the transcript is empty: it must hold what the proof absorbed before the opening");
    }
    if (!transcript.intact()) {
        throw invalid("::open", "an earlier failure left the transcript incomplete");
    }
    for (uint64_t i = 0; i < fs.size(); ++i) {
        if (fs[i].nCoefs > srs.nG1()) {
            throw invalid("::open", name(i) + "'s " + std::to_string(fs[i].nCoefs) + " coefficients exceed the " +
                                        std::to_string(srs.nG1()) + " powers [τ^i]₁ of the SRS");
        }
    }

    // Independent of the challenges: the transcript is untouched if it fails.
    const Interpolants r = interpolants();

    ShplonkProof proof;
    proof.alpha = transcript.squeeze();
    std::unique_ptr<Poly> W = quotientW(r, proof.alpha);
    proof.w = srs.commit(W->coef, W->getDegree() + 1);

    uint8_t bytes[G1_BYTES];
    encodeG1(proof.w, bytes);
    G1Point decoded;
    switch (decodeG1(bytes, decoded)) {
    case AbsorbError::None:
        break;
    case AbsorbError::Infinity:
        throw std::runtime_error("ShplonkProver::open: [W]₁ is the point at infinity, which the transcript does "
                                 "not absorb (spec A.4)");
    case AbsorbError::ShortCoordinate:
        throw std::runtime_error("ShplonkProver::open: [W]₁ has a coordinate below 2^192, which the transcript "
                                 "does not absorb (spec A.4)");
    default:
        throw std::logic_error("ShplonkProver::open: [W]₁ is not a valid point");
    }
    transcript.absorb(std::vector<G1Point>{proof.w});

    proof.y = transcript.squeeze();
    const std::unique_ptr<Poly> Wp = quotientWp(r, proof.alpha, proof.y, *W);
    W.reset();
    proof.wp = srs.commit(Wp->coef, Wp->getDegree() + 1);
    return proof;
}

} // namespace PilFflonk
