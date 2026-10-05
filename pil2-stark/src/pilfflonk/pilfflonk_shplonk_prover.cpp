#include "pilfflonk_shplonk_prover.hpp"

#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_lde.hpp"
#include "timer.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

constexpr InvalidArgument invalid("ShplonkProver");

std::string name(uint64_t i) {
    return "f_" + std::to_string(i);
}

// What quotientW and quotientWp divide, as their errors name it: f_i - r_i, and L.
std::string remainder(uint64_t i) {
    return name(i) + " - r_i";
}
constexpr const char *L_DIVIDEND = "L";

std::logic_error notDivisible(const std::string &what) {
    return std::logic_error("ShplonkProver: " + what + " is not divisible");
}

std::logic_error degreeNotBelow(const char *what, uint64_t degree, uint64_t bound) {
    return std::logic_error(std::string("ShplonkProver: ") + what + " has degree " + std::to_string(degree) +
                            ", not below " + std::to_string(bound));
}

// s mod n in [0, n), for |s| < n: the exponent of ω_n^s, since ω_n^n = 1.
uint64_t exponent(int64_t s, uint64_t n) {
    return s >= 0 ? static_cast<uint64_t>(s) : n - static_cast<uint64_t>(-s);
}

// out[j·stride] += s·in[j] for j < n.
void addScaled(FrElement *out, uint64_t stride, const FrElement *in, uint64_t n, const FrElement &s) {
    Engine &E = Engine::engine;
#pragma omp parallel for
    for (uint64_t j = 0; j < n; ++j) {
        FrElement term;
        E.fr.mul(term, s, in[j]);
        E.fr.add(out[j * stride], out[j * stride], term);
    }
}

std::logic_error componentsChanged(uint64_t i) {
    return std::logic_error("ShplonkProver: the components of " + name(i) + " changed after it was built");
}

// Throws std::invalid_argument, naming `function`, if a component of `prover` is not on the host,
// whose evaluations and quotients read its coefficients.
void requireOnHost(const ShplonkProver &prover, const char *function) {
    for (uint64_t i = 0; i < prover.size(); ++i) {
        for (uint64_t j = 0; j < prover.k(i); ++j) {
            if (prover.components(i)[j].host() == nullptr) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) +
                                            " is not on the host, whose evaluations and quotients read it");
            }
        }
    }
}

// The host's evaluations: rapidsnark's fastEvaluate on each p_j at each point of its f.
ShplonkProver::Evaluations hostEvaluations(const ShplonkProver &prover) {
    requireOnHost(prover, "");
    ShplonkProver::Evaluations evals(prover.size());
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const uint64_t k = prover.k(i);
        const std::vector<FrElement> &points = prover.points(i);
        evals[i].resize(points.size() * k);
        for (uint64_t m = 0; m < points.size(); ++m) {
            for (uint64_t j = 0; j < k; ++j) {
                evals[i][m * k + j] = prover.components(i)[j].host()->fastEvaluate(points[m]);
            }
        }
    }
    return evals;
}

// The host's quotients: quotientW, quotientWp and srs.commit, W' in place (Srs::commitInPlace), as
// nothing reads it after its commitment; W is read by W' after its own.
class HostQuotients final : public ShplonkQuotients {
public:
    explicit HostQuotients(const Srs &_srs) : srs(_srs) {}

    void computeW(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha) override {
        W = prover.quotientW(r, alpha);
    }
    G1Point commitW() override { return srs.commit(W->coef, W->getDegree() + 1); }
    void computeWp(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha,
                   const FrElement &y) override {
        Wp = prover.quotientWp(r, alpha, y, *W);
        W.reset();
    }
    G1Point commitWp() override {
        const G1Point commitment = srs.commitInPlace(Wp->coef, Wp->getDegree() + 1);
        Wp.reset();
        return commitment;
    }

private:
    const Srs &srs;
    std::unique_ptr<Poly> W;
    std::unique_ptr<Poly> Wp;
};

} // namespace

ShplonkComponent ShplonkComponent::elsewhere(uint64_t length, uint64_t degree) {
    ShplonkComponent c;
    c.isElsewhere = true;
    c.nCoefficients = length;
    c.topDegree = degree;
    return c;
}

void divideExactly(Poly &a, uint64_t m, const FrElement &beta, const std::string &what) {
    if (m == 0) {
        throw std::invalid_argument("divideExactly: X^0 - β is not monic of degree at least 1");
    }
    if (a.getDegree() < m) {
        // The quotient is 0 and the remainder a itself.
        if (a.getDegree() != 0 || !Engine::engine.fr.isZero(a.coef[0])) {
            throw notDivisible(what);
        }
        return;
    }
    // rapidsnark's parallel division (pilfflonk/docs/performance.md#the-shplonk-division): the
    // quotient in place, and whether the remainder a_j + β·q_j (j < m) is zero.
    if (!a.divByMonicInPlace(m, beta)) {
        throw notDivisible(what);
    }
}

ShplonkProver::ShplonkProver(ShplonkOpening opening) : ShplonkProver(std::move(opening), hostEvaluations) {}

ShplonkProver::ShplonkProver(ShplonkOpening opening, const Evaluator &evaluate) {
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
            const ShplonkComponent &c = f.components[j];
            if (!c.given()) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) + " is null");
            }
            if (c.length() == 0) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) + " has no coefficients");
            }
            if (c.degree() >= c.length()) {
                throw invalid(function, name(i) + ": component " + std::to_string(j) + " has degree " +
                                            std::to_string(c.degree()) + " and " + std::to_string(c.length()) +
                                            " coefficients");
            }
            maxLength = std::max(maxLength, c.length());
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

    // The roots (pilfflonk/docs/protocol.md#roots).
    challengeXi = power(opening.xiSeed, opening.powerW);
    const FrElement omegaN = rootOfUnityOfOrder(N);
    fs.reserve(opening.polynomials.size());
    for (ShplonkPolynomial &f : opening.polynomials) {
        Entry entry;
        const uint64_t k = f.components.size();
        const FrElement wk = rootOfUnityOfOrder(k);
        const FrElement omegaKN = rootOfUnityOfOrder(k * N);
        const FrElement seed = power(opening.xiSeed, opening.powerW / k);
        for (uint64_t j = 0; j < k; ++j) {
            entry.maxComponentLength = std::max(entry.maxComponentLength, f.components[j].length());
            // pack()'s count, CPolynomial's degree bound.
            entry.nCoefs = std::max(entry.nCoefs, k * f.components[j].degree() + j + 1);
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

    evals = evaluate(*this);
    bool shaped = evals.size() == fs.size();
    for (uint64_t i = 0; shaped && i < fs.size(); ++i) {
        shaped = evals[i].size() == fs[i].points.size() * fs[i].components.size();
    }
    if (!shaped) {
        throw std::logic_error("ShplonkProver: the evaluator did not give one evaluation per component and point");
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

uint64_t ShplonkProver::wLength() const {
    // Each (f_i - r_i)/Z_{T_i} has nCoefs(i) - |T_i| coefficients.
    uint64_t length = 1;
    for (const Entry &f : fs) {
        if (f.nCoefs > f.roots.size()) {
            length = std::max<uint64_t>(length, f.nCoefs - f.roots.size());
        }
    }
    return length;
}

uint64_t ShplonkProver::wpLength() const {
    // deg L <= max_i deg f_i, so W' has at most max_i nCoefs(i) - 1 coefficients.
    uint64_t maxCoefs = 1;
    for (const Entry &f : fs) {
        maxCoefs = std::max(maxCoefs, f.nCoefs);
    }
    return std::max<uint64_t>(1, maxCoefs - 1);
}

void ShplonkProver::checkW(uint64_t degree) const {
    if (degree >= wLength()) {
        throw degreeNotBelow("W", degree, wLength());
    }
}

void ShplonkProver::checkWp(uint64_t degree) const {
    if (degree >= wpLength()) {
        throw degreeNotBelow("W'", degree, wpLength());
    }
}

std::logic_error ShplonkProver::remainderNotDivisible(uint64_t i) {
    return notDivisible(remainder(i));
}

std::logic_error ShplonkProver::lNotDivisible() {
    return notDivisible(L_DIVIDEND);
}

ShplonkProver::LScalars ShplonkProver::lScalars(const FrElement &alpha, const FrElement &y) const {
    Engine &E = Engine::engine;
    // Z_{T_i}(y), and the scalars of L/Z_{T∖T_0}(y), the q_i
    // (pilfflonk/docs/protocol.md#pairing-check):
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
    std::vector<FrElement> ziInverse(zi.size());
    if (!batchInverse(ziInverse.data(), zi.data(), zi.size())) {
        throw std::logic_error("ShplonkProver: batchInverse found a zero Z_{T_i}(y), which is refused above");
    }
    FrElement zT = E.fr.one();
    for (const FrElement &z : zi) {
        E.fr.mul(zT, zT, z);
    }
    // 1/Z_{T∖T_0}(y) = Π_{i>=1} 1/Z_{T_i}(y)
    FrElement scale = E.fr.one();
    for (uint64_t i = 1; i < fs.size(); ++i) {
        E.fr.mul(scale, scale, ziInverse[i]);
    }

    LScalars scalars;
    scalars.w = E.fr.neg(E.fr.mul(zT, scale));
    scalars.f.resize(fs.size());
    FrElement alphaPower = E.fr.one();
    for (uint64_t i = 0; i < fs.size(); ++i) {
        scalars.f[i] = E.fr.mul(E.fr.mul(alphaPower, zT), E.fr.mul(ziInverse[i], scale));
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    return scalars;
}

uint64_t ShplonkProver::packed(uint64_t i, FrElement *out) const {
    const Entry &f = fs[i];
    const uint64_t k = f.components.size();
    std::vector<Poly *> polys(k);
    for (uint64_t j = 0; j < k; ++j) {
        polys[j] = f.components[j].host();
    }
    const uint64_t n = pack(polys.data(), k, out, packedBufferLength(k, f.maxComponentLength));
    if (n != f.nCoefs) {
        throw componentsChanged(i);
    }
    return n;
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
    for (uint64_t i = 0; i < fs.size(); ++i) {
        if (r[i] == nullptr || r[i]->getLength() > fs[i].roots.size()) {
            throw invalid("::quotientW", "r[" + std::to_string(i) + "] is not an interpolant of " + name(i));
        }
    }
    requireOnHost(*this, "::quotientW");

    // W, and one buffer for every f_i - r_i in turn, packed into it (pack() needs scratchLength()
    // elements) and divided in place as a polynomial over it, which does not clear it: no
    // coefficients are allocated, cleared or copied per f_i.
    const uint64_t length = workLength();
    std::unique_ptr<Poly> W(new Poly(E, length));
    const std::unique_ptr<FrElement[]> buffer(new FrElement[std::max(scratchLength(), length)]);
    FrElement alphaPower = E.fr.one();
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        const uint64_t k = f.components.size();
        // f_i - r_i in max(nCoefs(i), |T_i|) coefficients, as r_i has up to |T_i| (checked above).
        const uint64_t nCoefs = packed(i, buffer.get());
        const uint64_t n = std::max<uint64_t>(nCoefs, f.roots.size());
        std::fill(buffer.get() + nCoefs, buffer.get() + n, E.fr.zero());
        for (uint64_t j = 0; j < r[i]->getLength(); ++j) {
            E.fr.sub(buffer[j], buffer[j], r[i]->coef[j]);
        }
        const std::unique_ptr<Poly> term(Poly::fromReservedBuffer(E, buffer.get(), n));
        // Z_{T_i}(X) = Π_{s in O_i} (X^k - ξ·ω_N^s): the roots of offset s are those of X^k - ξ·ω_N^s.
        for (const FrElement &point : f.points) {
            divideExactly(*term, k, point, remainder(i));
        }
        addScaled(W->coef, 1, term->coef, term->getDegree() + 1, alphaPower);
        E.fr.mul(alphaPower, alphaPower, alpha);
    }
    W->fixDegree();
    checkW(W->getDegree());
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
    requireOnHost(*this, "::quotientWp");

    const LScalars scalars = lScalars(alpha, y);

    // L starts as -(Z_T(y)/Z_{T∖T_0}(y))·W, as long as every term.
    std::unique_ptr<Poly> L(new Poly(E, length));
    const uint64_t wCoefs = W.getDegree() + 1;
#pragma omp parallel for
    for (uint64_t m = 0; m < wCoefs; ++m) {
        E.fr.mul(L->coef[m], scalars.w, W.coef[m]);
    }

    // Each f_i − r_i(y) added to L: its components strided, coefficient c of p_j at c·k + j as pack()
    // puts it, as on the device (OpeningGpu::computeWp), with no f_i packed, and r_i(y) at 0.
    for (uint64_t i = 0; i < fs.size(); ++i) {
        const Entry &f = fs[i];
        const uint64_t k = f.components.size();
        uint64_t nCoefs = 0;
        for (uint64_t j = 0; j < k; ++j) {
            nCoefs = std::max(nCoefs, k * f.components[j].host()->getDegree() + j + 1);
        }
        if (nCoefs != f.nCoefs) {
            throw componentsChanged(i);
        }
        for (uint64_t j = 0; j < k; ++j) {
            const Poly &p = *f.components[j].host();
            addScaled(L->coef + j, k, p.coef, p.getDegree() + 1, scalars.f[i]);
        }
        E.fr.sub(L->coef[0], L->coef[0], E.fr.mul(scalars.f[i], r[i]->evaluate(y)));
    }
    L->fixDegree();

    // W' = L/(Z_{T∖T_0}(y)·(X - y)): divideExactly(1, y) in place of pil-fflonk's divByXSubValue.
    divideExactly(*L, 1, y, L_DIVIDEND);
    checkWp(L->getDegree());
    return L;
}

ShplonkProof ShplonkProver::open(const Srs &srs, Transcript &transcript) const {
    requireOnHost(*this, "::open");
    HostQuotients quotients(srs);
    return open(srs, transcript, quotients);
}

ShplonkProof ShplonkProver::open(const Srs &srs, Transcript &transcript, ShplonkQuotients &quotients) const {
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
    TimerStart(PILFFLONK_SHPLONK_W);
    quotients.computeW(*this, r, proof.alpha);
    TimerStopAndLog(PILFFLONK_SHPLONK_W);
    TimerStart(PILFFLONK_SHPLONK_COMMIT_W);
    proof.w = quotients.commitW();
    TimerStopAndLog(PILFFLONK_SHPLONK_COMMIT_W);

    uint8_t bytes[G1_BYTES];
    encodeG1(proof.w, bytes);
    G1Point decoded;
    switch (decodeG1(bytes, decoded)) {
    case AbsorbError::None:
        break;
    case AbsorbError::Infinity:
        throw std::runtime_error("ShplonkProver::open: [W]₁ is the point at infinity, which the transcript does "
                                 "not absorb (pilfflonk/docs/protocol.md#transcript)");
    case AbsorbError::ShortCoordinate:
        throw std::runtime_error("ShplonkProver::open: [W]₁ has a coordinate below 2^192, which the transcript "
                                 "does not absorb (pilfflonk/docs/protocol.md#transcript)");
    default:
        throw std::logic_error("ShplonkProver::open: [W]₁ is not a valid point");
    }
    transcript.absorb(std::vector<G1Point>{proof.w});

    proof.y = transcript.squeeze();
    TimerStart(PILFFLONK_SHPLONK_WP);
    quotients.computeWp(*this, r, proof.alpha, proof.y);
    TimerStopAndLog(PILFFLONK_SHPLONK_WP);
    TimerStart(PILFFLONK_SHPLONK_COMMIT_WP);
    proof.wp = quotients.commitWp();
    TimerStopAndLog(PILFFLONK_SHPLONK_COMMIT_WP);
    return proof;
}

std::vector<FrElement> verifierDenominators(const ShplonkProver &prover, const FrElement &y) {
    Engine &E = Engine::engine;
    std::vector<FrElement> denominators;
    for (uint64_t i = 1; i < prover.size(); ++i) {
        FrElement z = E.fr.one();
        for (const FrElement &x : prover.roots(i)) {
            E.fr.mul(z, z, E.fr.sub(y, x));
        }
        denominators.push_back(z);
    }
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const std::vector<FrElement> &T = prover.roots(i);
        for (uint64_t m = 0; m < T.size(); ++m) {
            FrElement den = E.fr.sub(y, T[m]);
            for (uint64_t l = 0; l < T.size(); ++l) {
                if (l != m) {
                    E.fr.mul(den, den, E.fr.sub(T[m], T[l]));
                }
            }
            denominators.push_back(den);
        }
    }
    return denominators;
}

FrElement verifierInverse(const ShplonkProver &prover, const FrElement &y) {
    Engine &E = Engine::engine;
    FrElement product = E.fr.one();
    for (const FrElement &d : verifierDenominators(prover, y)) {
        E.fr.mul(product, product, d);
    }
    if (E.fr.isZero(product)) {
        throw std::runtime_error("verifierInverse: a denominator of the verifier is zero: y is a root of some f_i");
    }
    FrElement inverse;
    E.fr.inv(inverse, product);
    return inverse;
}

} // namespace PilFflonk
