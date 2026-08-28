#ifndef STIR_MATH_HPP
#define STIR_MATH_HPP

// The polynomial arithmetic of STIR (Arnon, Chiesa, Fenzi, Yogev — ePrint 2024/390),
// in the notation of the paper. Nothing here knows about Merkle trees or the
// transcript: that is `stir.hpp`. Everything is over F = Goldilocks^3, stored as
// FIELD_EXTENSION consecutive base-field limbs per element, like the rest of the prover.
//
// Notation (Section 4 / Construction 5.2):
//
//   L = shift·⟨ω_n⟩          a smooth coset of size 2^n; index g ↔ the point shift·ω_n^g.
//   L^k = { x^k : x ∈ L }    the image of L under x ↦ x^k, k = 2^logK. It is the coset
//                            shift^k·⟨ω_{n−logK}⟩, and the k preimages of its g-th point are
//                            the points of L at indices g + j·2^{n−logK}, j = 0..k−1.
//   Fold(f, k, r)(x)         for x ∈ L^k: p̂_x(r), where p̂_x is the degree < k interpolant
//                            of f on the k preimages of x (Definition 4.?). If f = f̂|_L with
//                            deg f̂ < d, then Fold(f, k, r) = (Σ_j r^j f̂_j)|_{L^k} where
//                            f̂(X) = Σ_j X^j f̂_j(X^k), so it has degree < d/k.
//   Quotient(f, S, Ans, Fill)(x) = (f(x) − Âns(x)) / ∏_{a∈S}(x − a)   for x ∉ S,
//                                 = Fill(x)                             for x ∈ S,
//                            where Âns is the degree < |S| interpolant of Ans (Definition 4.?).
//   DegCor(d, r, f, d')(x)   = f(x) · Σ_{j=0}^{d−d'} (r·x)^j, raising a degree < d' claim to
//                            a degree < d one (Definition 4.?).

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "ntt_goldilocks.hpp"

namespace stir
{

using FE = Goldilocks::Element;   // base field element
using E3 = Goldilocks3::Element;  // extension field element, FE[3]

// A smooth coset L = shift·⟨ω_n⟩, |L| = 2^logSize.
struct Domain
{
    FE shift;
    uint64_t logSize;

    uint64_t size() const { return uint64_t(1) << logSize; }

    // The g-th point of L, shift·ω_n^g.
    FE point(uint64_t g) const { return Goldilocks::mul(shift, Goldilocks::exp(Goldilocks::w(logSize), g)); }

    // L^k for k = 2^logK: the coset shift^k·⟨ω_{n−logK}⟩.
    Domain power(uint64_t logK) const
    {
        assert(logK <= logSize);
        return Domain{Goldilocks::exp(shift, uint64_t(1) << logK), logSize - logK};
    }

    // Whether the extension element x lies in L (only base-field elements can).
    bool contains(const E3 &x) const
    {
        if (!Goldilocks::isZero(x[1]) || !Goldilocks::isZero(x[2])) return false;
        // x ∈ shift·⟨ω_n⟩  ⇔  (x / shift)^{2^n} = 1
        FE y = Goldilocks::mul(x[0], Goldilocks::inv(shift));
        return Goldilocks::isOne(Goldilocks::exp(y, size()));
    }

    // Whether the two cosets are disjoint. shift_A·⟨ω_a⟩ and shift_B·⟨ω_b⟩ intersect iff
    // shift_A / shift_B lies in the larger of the two subgroups, ⟨ω_max(a,b)⟩.
    static bool disjoint(const Domain &A, const Domain &B)
    {
        FE ratio = Goldilocks::mul(A.shift, Goldilocks::inv(B.shift));
        uint64_t logMax = std::max(A.logSize, B.logSize);
        return !Goldilocks::isOne(Goldilocks::exp(ratio, uint64_t(1) << logMax));
    }
};

// Embed a base-field element into F.
inline void embed(E3 &out, const FE &x)
{
    out[0] = x;
    out[1] = Goldilocks::zero();
    out[2] = Goldilocks::zero();
}

inline bool equal(const E3 &a, const E3 &b)
{
    return Goldilocks::toU64(a[0]) == Goldilocks::toU64(b[0]) && Goldilocks::toU64(a[1]) == Goldilocks::toU64(b[1]) &&
           Goldilocks::toU64(a[2]) == Goldilocks::toU64(b[2]);
}

inline bool isZero(const E3 &a)
{
    return Goldilocks::isZero(a[0]) && Goldilocks::isZero(a[1]) && Goldilocks::isZero(a[2]);
}

// Horner evaluation of Σ_j coeffs[j] X^j at x ∈ F.
inline void evalPoly(E3 &out, const FE *coeffs, uint64_t nCoeffs, const E3 &x)
{
    Goldilocks3::zero(out);
    for (int64_t j = int64_t(nCoeffs) - 1; j >= 0; j--)
    {
        E3 tmp;
        Goldilocks3::mul(tmp, out, x);
        Goldilocks3::add(out, tmp, (const E3 &)coeffs[j * FIELD_EXTENSION]);
    }
}

// Σ_{j=0}^{e} y^j, in closed form (1 − y^{e+1}) / (1 − y) unless y = 1.
inline void geometricSum(E3 &out, const E3 &y, uint64_t e)
{
    E3 one;
    Goldilocks3::one(one);
    if (equal(y, one))
    {
        Goldilocks3::mul(out, one, e + 1);
        return;
    }
    E3 yPow, num, den;
    Goldilocks3::pow(yPow, const_cast<E3 &>(y), e + 1);
    Goldilocks3::sub(num, one, yPow);
    Goldilocks3::sub(den, one, y);
    Goldilocks3::inv(den, den);
    Goldilocks3::mul(out, num, den);
}

// Montgomery's trick: invert n extension elements with one inversion.
inline void batchInverse(FE *values, uint64_t n)
{
    if (n == 0) return;
    std::vector<FE> prefix(n * FIELD_EXTENSION);
    Goldilocks3::copy((E3 &)prefix[0], (const E3 &)values[0]);
    for (uint64_t i = 1; i < n; i++)
    {
        Goldilocks3::mul((E3 &)prefix[i * FIELD_EXTENSION], (const E3 &)prefix[(i - 1) * FIELD_EXTENSION], (const E3 &)values[i * FIELD_EXTENSION]);
    }
    E3 inv;
    Goldilocks3::inv(inv, (const E3 &)prefix[(n - 1) * FIELD_EXTENSION]);
    for (int64_t i = int64_t(n) - 1; i > 0; i--)
    {
        E3 tmp;
        Goldilocks3::mul(tmp, inv, (const E3 &)prefix[(i - 1) * FIELD_EXTENSION]);   // 1 / values[i]
        Goldilocks3::mul(inv, inv, (const E3 &)values[i * FIELD_EXTENSION]);         // 1 / (values[0]···values[i−1])
        Goldilocks3::copy((E3 &)values[i * FIELD_EXTENSION], tmp);
    }
    Goldilocks3::copy((E3 &)values[0], inv);
}

// ---------------------------------------------------------------------------------------------
// Evaluations on a coset  ↔  coefficients
// ---------------------------------------------------------------------------------------------

// From the evaluations of a polynomial f̂ of degree < |L| on L = shift·⟨ω_n⟩ to its
// coefficients. The INTT interpolates on ⟨ω_n⟩: it returns q with q(ω^g) = f̂(shift·ω^g),
// i.e. q(X) = f̂(shift·X), so f̂'s j-th coefficient is q_j·shift^{−j}.
inline void evaluationsToCoefficients(FE *coeffs, const FE *evals, const Domain &L)
{
    uint64_t N = L.size();
    NTT_Goldilocks ntt(N);
    ntt.INTT(coeffs, const_cast<FE *>(evals), N, FIELD_EXTENSION);
    FE shiftInv = Goldilocks::inv(L.shift);
    FE s = Goldilocks::one();
    for (uint64_t j = 0; j < N; j++)
    {
        Goldilocks3::mul((E3 &)coeffs[j * FIELD_EXTENSION], (const E3 &)coeffs[j * FIELD_EXTENSION], s);
        s = Goldilocks::mul(s, shiftInv);
    }
}

// From nCoeffs ≤ |L| coefficients to the evaluations on L = shift·⟨ω_n⟩: zero-pad, absorb the
// coset shift into the coefficients (f̂(shift·X) has coefficients f̂_j·shift^j) and NTT.
inline void coefficientsToEvaluations(FE *evals, const FE *coeffs, uint64_t nCoeffs, const Domain &L)
{
    uint64_t N = L.size();
    assert(nCoeffs <= N);
    std::vector<FE> scaled(N * FIELD_EXTENSION, Goldilocks::zero());
    FE s = Goldilocks::one();
    for (uint64_t j = 0; j < nCoeffs; j++)
    {
        Goldilocks3::mul((E3 &)scaled[j * FIELD_EXTENSION], (const E3 &)coeffs[j * FIELD_EXTENSION], s);
        s = Goldilocks::mul(s, L.shift);
    }
    NTT_Goldilocks ntt(N);
    ntt.NTT(evals, scaled.data(), N, FIELD_EXTENSION);
}

// Number of leading coefficients (from the top) that are zero, i.e. N − (degree + 1).
// Used by the tests and debug assertions to check degree bounds.
inline uint64_t numTrailingZeroCoefficients(const FE *coeffs, uint64_t N)
{
    uint64_t z = 0;
    for (int64_t j = int64_t(N) - 1; j >= 0 && isZero((const E3 &)coeffs[j * FIELD_EXTENSION]); j--) z++;
    return z;
}

// ---------------------------------------------------------------------------------------------
// Fold
// ---------------------------------------------------------------------------------------------

// Fold(f, k, r) on L^k, from the evaluations f of f̂ on L (Definition 4.8).
//
// For the g-th point x of L^k, the k preimages are the points of L at indices g + j·2^{n−logK},
// which form the coset c·⟨ω_k⟩ with c = shift·ω_n^g. Interpolating f there is an INTT on
// ⟨ω_k⟩ followed by the coset correction q_j ↦ q_j·c^{−j}; Fold(f, k, r)(x) = p̂_x(r) is then a
// Horner evaluation at r.
//
// `out` receives 2^{n−logK} extension elements, indexed like L^k.
//
// `foldCoset` is the single-point version — the verifier's case, since it only ever holds one
// coset at a time (the k values of one Merkle leaf). `coset` lists them in leaf order j = 0..k−1,
// i.e. `coset[j]` is f at the L-index `m + j·|L^k|`, which is exactly how `commit` lays out a leaf.
// `ntt` must have been built for size k; `scratch` is grown as needed and may be reused.
inline void foldCoset(E3 &out, const FE *coset, const Domain &L, uint64_t logK, uint64_t m, const E3 &r, NTT_Goldilocks &ntt, std::vector<FE> &scratch)
{
    uint64_t k = uint64_t(1) << logK;
    if (scratch.size() < k * FIELD_EXTENSION) scratch.resize(k * FIELD_EXTENSION);
    std::memcpy(scratch.data(), coset, k * FIELD_EXTENSION * sizeof(FE));
    ntt.INTT(scratch.data(), scratch.data(), k, FIELD_EXTENSION);

    // p̂_x(X) = q(X / c) with c = shift·ω_n^m: coefficient j gets c^{−j}.
    FE cInv = Goldilocks::inv(Goldilocks::mul(L.shift, Goldilocks::exp(Goldilocks::w(L.logSize), m)));
    FE s = Goldilocks::one();
    for (uint64_t j = 0; j < k; j++)
    {
        Goldilocks3::mul((E3 &)scratch[j * FIELD_EXTENSION], (const E3 &)scratch[j * FIELD_EXTENSION], s);
        s = Goldilocks::mul(s, cInv);
    }
    evalPoly(out, scratch.data(), k, r);
}

inline void fold(FE *out, const FE *f, const Domain &L, uint64_t logK, const E3 &r)
{
    assert(logK >= 1 && logK <= L.logSize);
    uint64_t k = uint64_t(1) << logK;
    uint64_t M = L.size() >> logK;   // |L^k|

#pragma omp parallel
    {
        NTT_Goldilocks ntt(k, 1);
        std::vector<FE> coset(k * FIELD_EXTENSION), scratch(k * FIELD_EXTENSION);
#pragma omp for
        for (uint64_t m = 0; m < M; m++)
        {
            for (uint64_t j = 0; j < k; j++)
            {
                std::memcpy(&coset[j * FIELD_EXTENSION], &f[(m + j * M) * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(FE));
            }
            foldCoset((E3 &)out[m * FIELD_EXTENSION], coset.data(), L, logK, m, r, ntt, scratch);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Quotient and degree correction
// ---------------------------------------------------------------------------------------------

// The coefficients of Âns, the unique polynomial of degree < |S| with Âns(a) = Ans(a) for all
// a ∈ S (Newton interpolation, O(|S|²)). `points`/`values` hold |S| extension elements each.
inline void interpolate(std::vector<FE> &coeffs, const FE *points, const FE *values, uint64_t n)
{
    coeffs.assign(n * FIELD_EXTENSION, Goldilocks::zero());
    if (n == 0) return;

    // Newton divided differences: c[j] = Ans[a_0, …, a_j].
    std::vector<FE> c(values, values + n * FIELD_EXTENSION);
    for (uint64_t level = 1; level < n; level++)
    {
        for (uint64_t j = n - 1; j >= level; j--)
        {
            E3 num, den;
            Goldilocks3::sub(num, (const E3 &)c[j * FIELD_EXTENSION], (const E3 &)c[(j - 1) * FIELD_EXTENSION]);
            Goldilocks3::sub(den, (const E3 &)points[j * FIELD_EXTENSION], (const E3 &)points[(j - level) * FIELD_EXTENSION]);
            assert(!isZero(den) && "interpolation points must be distinct");
            Goldilocks3::inv(den, den);
            Goldilocks3::mul((E3 &)c[j * FIELD_EXTENSION], num, den);
        }
    }

    // Newton form → monomial coefficients, Horner-style from the top:
    //   Âns = c_0 + (X − a_0)(c_1 + (X − a_1)(c_2 + …))
    // Starting from P = c_{n−1}, repeat P ← P·(X − a_j) + c_j for j = n−2 down to 0.
    Goldilocks3::copy((E3 &)coeffs[0], (const E3 &)c[(n - 1) * FIELD_EXTENSION]);
    uint64_t deg = 0;   // degree of P
    for (int64_t j = int64_t(n) - 2; j >= 0; j--)
    {
        const E3 &a = (const E3 &)points[j * FIELD_EXTENSION];
        // P·(X − a): descending m so coeffs[m] is still the old value when it is moved up.
        for (int64_t m = int64_t(deg); m >= 0; m--)
        {
            E3 &lo = (E3 &)coeffs[m * FIELD_EXTENSION];
            E3 &hi = (E3 &)coeffs[(m + 1) * FIELD_EXTENSION];
            Goldilocks3::add(hi, hi, lo);   // X·P
            E3 t, zero;
            Goldilocks3::zero(zero);
            Goldilocks3::mul(t, lo, a);
            Goldilocks3::sub(lo, zero, t);  // lo = −a·P_m
        }
        Goldilocks3::add((E3 &)coeffs[0], (const E3 &)coeffs[0], (const E3 &)c[j * FIELD_EXTENSION]);
        deg++;
    }
}

// (G, Ans, r_comb) of one iteration: everything step 2(e) of Construction 5.2 needs in order to
// turn a value of the committed g_i into the corresponding value of the virtual
//
//   f_i = DegCor(d_i, r_comb, Quotient(g_i, G_i, Ans_i, Fill_i), d_i − |G_i|)
//
// `apply` does it at one point — the verifier's case, one opened value at a time — and
// `quotientAndDegreeCorrect` over a whole domain, the prover's case. Both go through
// `applyWithInverseDenominator`, so there is exactly one implementation of the formula.
struct QuotientContext
{
    std::vector<FE> points;      // G, as |G| extension elements
    std::vector<FE> values;      // Ans on G
    std::vector<FE> ansCoeffs;   // Âns, degree < |G|
    E3 rComb;

    uint64_t size() const { return points.size() / FIELD_EXTENSION; }

    void reset(const E3 &rComb_)
    {
        points.clear();
        values.clear();
        ansCoeffs.clear();
        Goldilocks3::copy(rComb, rComb_);
    }

    // G is a set: the shift queries are sampled with replacement, so a repeated point is dropped.
    // Returns whether the point was new. Prover and verifier must apply the same rule.
    bool add(const E3 &pt, const E3 &val)
    {
        for (uint64_t m = 0; m < size(); m++)
        {
            if (equal((const E3 &)points[m * FIELD_EXTENSION], pt)) return false;
        }
        points.insert(points.end(), pt, pt + FIELD_EXTENSION);
        values.insert(values.end(), val, val + FIELD_EXTENSION);
        return true;
    }

    void build() { interpolate(ansCoeffs, points.data(), values.data(), size()); }

    //   f_i(x) = ( (g_i(x) − Âns(x)) · denomInv ) · Σ_{j=0}^{|G|} (r_comb·x)^j,
    // where denomInv = 1 / ∏_{a∈G}(x − a). The exponent range is d_i − (d_i − |G|) = |G|.
    void applyWithInverseDenominator(E3 &out, const E3 &gx, const E3 &x, const E3 &denomInv) const
    {
        E3 ans, q, geo, rx;
        evalPoly(ans, ansCoeffs.data(), size(), x);
        Goldilocks3::sub(q, gx, ans);
        Goldilocks3::mul(q, q, denomInv);
        Goldilocks3::mul(rx, rComb, x);
        geometricSum(geo, rx, size());
        Goldilocks3::mul(out, q, geo);
    }

    // ∏_{a∈G}(x − a). Non-zero because G ∩ L = ∅ for every domain L this is used on — the
    // out-of-domain samples are drawn from F \ L_i and the shift queries live in L_{i−1}^{k_{i−1}},
    // which `Parameters::validate` checks to be disjoint from L_i (Remark 5.3).
    void denominator(E3 &out, const E3 &x) const
    {
        Goldilocks3::one(out);
        for (uint64_t a = 0; a < size(); a++)
        {
            E3 diff;
            Goldilocks3::sub(diff, x, (const E3 &)points[a * FIELD_EXTENSION]);
            assert(!isZero(diff) && "G ∩ L must be empty (Remark 5.3: Fill is never used)");
            Goldilocks3::mul(out, out, diff);
        }
    }

    void apply(E3 &out, const E3 &gx, const E3 &x) const
    {
        E3 denomInv;
        denominator(denomInv, x);
        Goldilocks3::inv(denomInv, denomInv);
        applyWithInverseDenominator(out, gx, x, denomInv);
    }
};

// f_i on the whole of L, from the evaluations `g` of g_i on L. Same formula as `apply`, with the
// denominators inverted in one batch.
inline void quotientAndDegreeCorrect(FE *fOut, const FE *g, const Domain &L, const QuotientContext &ctx)
{
    uint64_t N = L.size();

    std::vector<FE> denom(N * FIELD_EXTENSION);
#pragma omp parallel for
    for (uint64_t m = 0; m < N; m++)
    {
        E3 x;
        embed(x, L.point(m));
        ctx.denominator((E3 &)denom[m * FIELD_EXTENSION], x);
    }
    batchInverse(denom.data(), N);

#pragma omp parallel for
    for (uint64_t m = 0; m < N; m++)
    {
        E3 x;
        embed(x, L.point(m));
        ctx.applyWithInverseDenominator((E3 &)fOut[m * FIELD_EXTENSION], (const E3 &)g[m * FIELD_EXTENSION], x, (const E3 &)denom[m * FIELD_EXTENSION]);
    }
}

// ---------------------------------------------------------------------------------------------
// The prover's view of one STIR execution: the sequence f_0, g_1, f_1, …, g_M = p̂.
// ---------------------------------------------------------------------------------------------

// Parameters of Construction 5.2, as the stark info records them (see `StirStruct`):
//   M iterations; k_i = 2^logFoldingFactors[i]; d_i = 2^logDegrees[i] (i = 0..M);
//   L_i = shift·⟨ω⟩ of size 2^logDomainSizes[i] (i = 0..M), |L_{i+1}| = |L_i| / 2.
struct Parameters
{
    FE shift;
    std::vector<uint64_t> logFoldingFactors;   // k_i, length M
    std::vector<uint64_t> logDegrees;          // d_i, length M+1
    std::vector<uint64_t> logDomainSizes;      // |L_i|, length M+1

    uint64_t M() const { return logFoldingFactors.size(); }
    Domain L(uint64_t i) const { return Domain{shift, logDomainSizes[i]}; }
    uint64_t k(uint64_t i) const { return uint64_t(1) << logFoldingFactors[i]; }
    uint64_t d(uint64_t i) const { return uint64_t(1) << logDegrees[i]; }

    void validate() const
    {
        uint64_t m = M();
        assert(m >= 1);
        assert(logDegrees.size() == m + 1 && logDomainSizes.size() == m + 1);
        for (uint64_t i = 0; i < m; i++)
        {
            assert(logFoldingFactors[i] >= 1);
            assert(logDegrees[i + 1] + logFoldingFactors[i] == logDegrees[i] && "d_{i+1} = d_i / k_i");
            assert(logDomainSizes[i + 1] + 1 == logDomainSizes[i] && "|L_{i+1}| = |L_i| / 2");
            assert(logDegrees[i] < logDomainSizes[i] && "rate < 1");
            // deg ĝ_{i+1} < d_{i+1} must fit the fold domain L_i^{k_i}
            assert(logDegrees[i + 1] <= logDomainSizes[i] - logFoldingFactors[i]);
            // Remark 5.3: with L_i^{k_i} ∩ L_{i+1} = ∅ the Fill_{i+1} oracle and its verifier check
            // disappear. Here L_i^{k_i} = shift^{k_i}·⟨…⟩ and L_{i+1} = shift·⟨…⟩ meet iff shift^{k_i−1}
            // is a 2-adic root of unity, which for shift = 7 (a generator of F*, of order
            // 2^32·(2^32−1)) needs 2^32−1 | k_i−1 — never for any usable k_i. Checked, not assumed.
            assert(Domain::disjoint(L(i).power(logFoldingFactors[i]), L(i + 1)) && "Remark 5.3 requires L_i^{k_i} ∩ L_{i+1} = ∅");
        }
        assert(logDegrees[m] <= logDomainSizes[m]);
    }
};

// The prover's per-iteration state. `Prover` does the arithmetic of one iteration at a time;
// the protocol driver (`stir.hpp`) interleaves it with the commitments and the transcript.
class Prover
{
public:
    Prover(const Parameters &params, const FE *f0) : p(params), i(0)
    {
        p.validate();
        f.assign(f0, f0 + p.L(0).size() * FIELD_EXTENSION);
    }

    // Current iteration index i: `f` holds f_i on L_i.
    uint64_t iteration() const { return i; }
    const std::vector<FE> &f_i() const { return f; }

    // Step 2(a) of iteration i+1 (or the final step when i+1 = M): given k_i, r^fold_i, compute
    //   g_{i+1} := Fold(f_i, k_i, r^fold_i)
    // as evaluations on L_i^{k_i}, and from them the coefficients ĝ_{i+1} (degree < d_{i+1}).
    // Then evaluate g_{i+1} on L_{i+1}, which is what gets committed.
    void fold(const E3 &rFold)
    {
        assert(i < p.M());
        Domain Li = p.L(i);
        Domain LiK = Li.power(p.logFoldingFactors[i]);

        foldOnLiK.assign(LiK.size() * FIELD_EXTENSION, Goldilocks::zero());
        stir::fold(foldOnLiK.data(), f.data(), Li, p.logFoldingFactors[i], rFold);

        // ĝ_{i+1}: degree < d_{i+1} ≤ |L_i^{k_i}|, so interpolating on L_i^{k_i} is exact.
        gHat.assign(LiK.size() * FIELD_EXTENSION, Goldilocks::zero());
        evaluationsToCoefficients(gHat.data(), foldOnLiK.data(), LiK);
        assert(numTrailingZeroCoefficients(gHat.data(), LiK.size()) >= LiK.size() - p.d(i + 1) && "Fold(f_i) must have degree < d_{i+1}");
        gHat.resize(p.d(i + 1) * FIELD_EXTENSION);

        // g_{i+1} on L_{i+1} (the committed oracle). Not needed after the last fold.
        if (i + 1 < p.M())
        {
            g.assign(p.L(i + 1).size() * FIELD_EXTENSION, Goldilocks::zero());
            coefficientsToEvaluations(g.data(), gHat.data(), p.d(i + 1), p.L(i + 1));
        }
    }

    // Fold(f_i, k_i, r^fold_i) on L_i^{k_i}, indexed like L_i^{k_i} (valid after `fold`). The verifier
    // recomputes these values at the shift queries from the opened cosets of f_i, so they are
    // Ans_{i+1}(r_shift^{i+1,j}).
    const std::vector<FE> &foldValues() const { return foldOnLiK; }

    // Evaluations of g_{i+1} on L_{i+1} (valid after `fold`, i+1 < M).
    const std::vector<FE> &g_next() const { return g; }

    // Coefficients of ĝ_{i+1}, d_{i+1} extension elements (valid after `fold`). After the last
    // fold this is p, the final polynomial sent in the clear.
    const std::vector<FE> &gHat_next() const { return gHat; }

    // Step 2(c): β := ĝ_{i+1}(r_out) for an out-of-domain sample r_out ∈ F \ L_{i+1}.
    void outOfDomainAnswer(E3 &beta, const E3 &rOut) const
    {
        assert(!p.L(i + 1).contains(rOut) && "r_out must be sampled from F \\ L_{i+1}");
        evalPoly(beta, gHat.data(), p.d(i + 1), rOut);
    }

    // Step 2(e): move to iteration i+1 by defining
    //   f_{i+1} := DegCor(d_{i+1}, r_comb, Quotient(g_{i+1}, G, Ans, Fill), d_{i+1} − |G|)
    // on L_{i+1}, where G = {r_out^{·}} ∪ {r_shift^{·}} and Ans the corresponding answers
    // (β's, and Fold(f_i, k_i, r^fold_i)(r_shift) = foldValues()[index of r_shift]).
    //
    // `shiftIndices` index L_i^{k_i}; duplicates (sampling with replacement) are removed, G being a
    // set. Points of G are returned in `pointsOut`/`valuesOut` in the order used, so the caller
    // can record them.
    void degreeCorrect(const std::vector<E3> &rOut, const std::vector<E3> &beta, const std::vector<uint64_t> &shiftIndices, const E3 &rComb)
    {
        assert(i + 1 < p.M());
        assert(rOut.size() == beta.size());
        Domain LiK = p.L(i).power(p.logFoldingFactors[i]);

        QuotientContext ctx;
        ctx.reset(rComb);
        for (uint64_t m = 0; m < rOut.size(); m++) ctx.add(rOut[m], beta[m]);
        for (uint64_t idx : shiftIndices)
        {
            assert(idx < LiK.size());
            E3 pt;
            embed(pt, LiK.point(idx));
            ctx.add(pt, (const E3 &)foldOnLiK[idx * FIELD_EXTENSION]);
        }
        assert(ctx.size() < p.d(i + 1) && "|G| must be below d_{i+1} for the quotient to have positive degree bound");
        ctx.build();

        Domain Lnext = p.L(i + 1);
        f.assign(Lnext.size() * FIELD_EXTENSION, Goldilocks::zero());
        quotientAndDegreeCorrect(f.data(), g.data(), Lnext, ctx);
        i++;
    }

    const Parameters &params() const { return p; }

private:
    Parameters p;
    uint64_t i;
    std::vector<FE> f;          // f_i on L_i
    std::vector<FE> foldOnLiK;  // Fold(f_i, k_i, r^fold_i) on L_i^{k_i}
    std::vector<FE> gHat;       // coefficients of ĝ_{i+1}
    std::vector<FE> g;          // g_{i+1} on L_{i+1}
};

} // namespace stir

#endif
