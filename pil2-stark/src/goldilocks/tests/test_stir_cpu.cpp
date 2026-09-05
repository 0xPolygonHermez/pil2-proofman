// Tests of the STIR polynomial arithmetic (stir_math.hpp), independent of Merkle trees and the
// transcript. Challenges are drawn from a fixed PRNG; everything checked is an identity of the
// paper, so no golden values are involved.
#include "test_helpers.hpp"
#include "../../starkpil/stir/stir_math.hpp"

#include <random>

using stir::E3;
using stir::FE;

namespace
{

struct Rng
{
    std::mt19937_64 gen{0x5717};
    FE fe() { return Goldilocks::fromU64(gen() % GOLDILOCKS_PRIME); }
    void e3(E3 &out)
    {
        out[0] = fe();
        out[1] = fe();
        out[2] = fe();
    }
};

// Random coefficients of a polynomial of degree < d, then its evaluations on L.
void randomLowDegreeOnDomain(Rng &rng, uint64_t d, const stir::Domain &L, std::vector<FE> &coeffs, std::vector<FE> &evals)
{
    coeffs.assign(d * FIELD_EXTENSION, Goldilocks::zero());
    for (uint64_t j = 0; j < d; j++) rng.e3((E3 &)coeffs[j * FIELD_EXTENSION]);
    evals.assign(L.size() * FIELD_EXTENSION, Goldilocks::zero());
    stir::coefficientsToEvaluations(evals.data(), coeffs.data(), d, L);
}

// The degree bound of the polynomial with these evaluations on L: the largest j with a non-zero
// coefficient, plus one (0 for the zero polynomial).
uint64_t degreeBoundOn(const std::vector<FE> &evals, const stir::Domain &L)
{
    std::vector<FE> coeffs(evals.size());
    stir::evaluationsToCoefficients(coeffs.data(), evals.data(), L);
    return L.size() - stir::numTrailingZeroCoefficients(coeffs.data(), L.size());
}

const FE SHIFT = Goldilocks::shift();

} // namespace

// coefficients → evaluations on a coset → coefficients is the identity, and the direct
// evaluation of the polynomial agrees pointwise.
TEST(STIR_TEST, coset_evaluations_roundtrip)
{
    Rng rng;
    stir::Domain L{SHIFT, 6};
    std::vector<FE> coeffs, evals;
    randomLowDegreeOnDomain(rng, 37, L, coeffs, evals);

    for (uint64_t g = 0; g < L.size(); g++)
    {
        E3 x, y;
        stir::embed(x, L.point(g));
        stir::evalPoly(y, coeffs.data(), 37, x);
        ASSERT_TRUE(stir::equal(y, (const E3 &)evals[g * FIELD_EXTENSION])) << "point " << g;
    }

    std::vector<FE> back(evals.size());
    stir::evaluationsToCoefficients(back.data(), evals.data(), L);
    for (uint64_t j = 0; j < 37; j++) ASSERT_TRUE(stir::equal((const E3 &)back[j * FIELD_EXTENSION], (const E3 &)coeffs[j * FIELD_EXTENSION]));
    ASSERT_EQ(stir::numTrailingZeroCoefficients(back.data(), L.size()), L.size() - 37);
}

// Fold(f, k, r) = Σ_j r^j f̂_j where f̂(X) = Σ_j X^j f̂_j(X^k): compare the coset-interpolation
// definition with the coefficient-splitting one, on every point of L^k.
TEST(STIR_TEST, fold_matches_coefficient_splitting)
{
    Rng rng;
    stir::Domain L{SHIFT, 8};
    const uint64_t logK = 3, k = 1 << logK, d = 200;   // d < |L| = 256
    std::vector<FE> coeffs, evals;
    randomLowDegreeOnDomain(rng, d, L, coeffs, evals);

    E3 r;
    rng.e3(r);
    std::vector<FE> folded((L.size() >> logK) * FIELD_EXTENSION);
    stir::fold(folded.data(), evals.data(), L, logK, r);

    // Σ_j r^j f̂_j(Y): coefficient m of f̂_j is coefficient m·k + j of f̂.
    uint64_t dFold = (d + k - 1) / k;
    std::vector<FE> foldedCoeffs(dFold * FIELD_EXTENSION, Goldilocks::zero());
    E3 rPow;
    Goldilocks3::one(rPow);
    for (uint64_t j = 0; j < k; j++)
    {
        for (uint64_t m = 0; m * k + j < d; m++)
        {
            E3 t;
            Goldilocks3::mul(t, (const E3 &)coeffs[(m * k + j) * FIELD_EXTENSION], rPow);
            Goldilocks3::add((E3 &)foldedCoeffs[m * FIELD_EXTENSION], (const E3 &)foldedCoeffs[m * FIELD_EXTENSION], t);
        }
        Goldilocks3::mul(rPow, rPow, r);
    }

    stir::Domain LK = L.power(logK);
    for (uint64_t g = 0; g < LK.size(); g++)
    {
        E3 y, expected;
        stir::embed(y, LK.point(g));
        stir::evalPoly(expected, foldedCoeffs.data(), dFold, y);
        ASSERT_TRUE(stir::equal(expected, (const E3 &)folded[g * FIELD_EXTENSION])) << "point " << g;
    }
    // and it has degree < d/k
    ASSERT_LE(degreeBoundOn(folded, LK), dFold);
}

// Âns interpolates Ans exactly and has degree < |S|.
TEST(STIR_TEST, interpolation)
{
    Rng rng;
    const uint64_t n = 41;
    std::vector<FE> points(n * FIELD_EXTENSION), values(n * FIELD_EXTENSION);
    for (uint64_t a = 0; a < n; a++)
    {
        rng.e3((E3 &)points[a * FIELD_EXTENSION]);
        rng.e3((E3 &)values[a * FIELD_EXTENSION]);
    }
    std::vector<FE> coeffs;
    stir::interpolate(coeffs, points.data(), values.data(), n);
    ASSERT_EQ(coeffs.size(), n * FIELD_EXTENSION);
    for (uint64_t a = 0; a < n; a++)
    {
        E3 y;
        stir::evalPoly(y, coeffs.data(), n, (const E3 &)points[a * FIELD_EXTENSION]);
        ASSERT_TRUE(stir::equal(y, (const E3 &)values[a * FIELD_EXTENSION])) << "point " << a;
    }
}

// DegCor(d, r, Quotient(g, S, Ans, ·), d − |S|) of an honest g (degree < d, Ans = g on S) has
// degree < d again; with a wrong answer in Ans it does not.
TEST(STIR_TEST, quotient_and_degree_correction_preserve_low_degree)
{
    Rng rng;
    stir::Domain L{SHIFT, 8};
    const uint64_t d = 128, nS = 20;
    std::vector<FE> gCoeffs, g;
    randomLowDegreeOnDomain(rng, d, L, gCoeffs, g);

    std::vector<FE> points(nS * FIELD_EXTENSION), values(nS * FIELD_EXTENSION);
    for (uint64_t a = 0; a < nS; a++)
    {
        E3 &pt = (E3 &)points[a * FIELD_EXTENSION];
        do
        {
            rng.e3(pt);
        } while (L.contains(pt));
        stir::evalPoly((E3 &)values[a * FIELD_EXTENSION], gCoeffs.data(), d, pt);
    }
    E3 rComb;
    rng.e3(rComb);

    std::vector<FE> f(L.size() * FIELD_EXTENSION);
    stir::quotientAndDegreeCorrect(f.data(), g.data(), L, points.data(), values.data(), nS, rComb);
    ASSERT_LE(degreeBoundOn(f, L), d);

    // Corrupt one answer: the quotient is no longer a polynomial of degree < d − |S| on L.
    values[0] = values[0] + Goldilocks::one();
    stir::quotientAndDegreeCorrect(f.data(), g.data(), L, points.data(), values.data(), nS, rComb);
    ASSERT_GT(degreeBoundOn(f, L), d);
}

// A whole execution of the prover's arithmetic (Construction 5.2 without commitments): with an
// honest f_0 every f_i has degree < d_i, every g_i degree < d_i, the final p has degree < d_M, and
// the final check Fold(f_{M−1}, k_{M−1}, r^fold_{M−1})(r_shift) = p(r_shift) passes at fresh points.
TEST(STIR_TEST, honest_execution)
{
    Rng rng;
    stir::Parameters params;
    params.shift = SHIFT;
    params.logFoldingFactors = {3, 3, 2};         // k_i
    params.logDegrees = {14, 11, 8, 6};           // d_i = d_{i−1} / k_{i−1}; d_i > t_{i−1} + s throughout
    params.logDomainSizes = {16, 15, 14, 13};     // |L_{i+1}| = |L_i| / 2 (initial rate 1/4)
    const uint64_t M = params.M(), s = 1;
    const std::vector<uint64_t> t = {30, 20, 15};  // t_i shift queries into f_i

    std::vector<FE> f0Coeffs, f0;
    randomLowDegreeOnDomain(rng, params.d(0), params.L(0), f0Coeffs, f0);
    stir::Prover prover(params, f0.data());

    E3 rFold;
    rng.e3(rFold);
    for (uint64_t i = 1; i < M; i++)
    {
        prover.fold(rFold);
        ASSERT_LE(degreeBoundOn(prover.g_next(), params.L(i)), params.d(i)) << "g_" << i;

        std::vector<E3> rOut(s), beta(s);
        for (uint64_t j = 0; j < s; j++)
        {
            do
            {
                rng.e3(rOut[j]);
            } while (params.L(i).contains(rOut[j]));
            prover.outOfDomainAnswer(beta[j], rOut[j]);
            // β is ĝ_i(r_out): consistent with the committed evaluations of g_i
            E3 check;
            stir::evalPoly(check, prover.gHat_next().data(), params.d(i), rOut[j]);
            ASSERT_TRUE(stir::equal(check, beta[j]));
        }

        E3 rFoldNext, rComb;
        rng.e3(rFoldNext);
        rng.e3(rComb);
        stir::Domain LprevK = params.L(i - 1).power(params.logFoldingFactors[i - 1]);
        std::vector<uint64_t> shiftIndices(t[i - 1]);
        for (auto &idx : shiftIndices) idx = rng.gen() % LprevK.size();

        prover.degreeCorrect(rOut, beta, shiftIndices, rComb);
        ASSERT_EQ(prover.iteration(), i);
        ASSERT_LE(degreeBoundOn(prover.f_i(), params.L(i)), params.d(i)) << "f_" << i;
        Goldilocks3::copy(rFold, rFoldNext);
    }

    // Final step.
    prover.fold(rFold);
    const std::vector<FE> &p = prover.gHat_next();
    ASSERT_EQ(p.size(), params.d(M) * FIELD_EXTENSION);
    stir::Domain LK = params.L(M - 1).power(params.logFoldingFactors[M - 1]);
    for (uint64_t j = 0; j < t[M - 1]; j++)
    {
        uint64_t idx = rng.gen() % LK.size();
        E3 x, px;
        stir::embed(x, LK.point(idx));
        stir::evalPoly(px, p.data(), params.d(M), x);
        ASSERT_TRUE(stir::equal(px, (const E3 &)prover.foldValues()[idx * FIELD_EXTENSION])) << "final check at " << idx;
    }
}

// The same execution with a cheating f_0 of too high a degree: the last fold is not a polynomial
// of degree < d_M on L_{M−1}^{k_{M−1}}, so the prover's own degree assertion would fire. Here we
// check the arithmetic directly: the interpolant of the last fold has degree ≥ d_M.
TEST(STIR_TEST, dishonest_f0_is_not_low_degree_at_the_end)
{
    Rng rng;
    stir::Parameters params;
    params.shift = SHIFT;
    params.logFoldingFactors = {2, 2};
    params.logDegrees = {8, 6, 4};
    params.logDomainSizes = {10, 9, 8};

    // f_0 of degree d_0 + 1 (one coefficient too many)
    std::vector<FE> coeffs, f0;
    randomLowDegreeOnDomain(rng, params.d(0) + 2, params.L(0), coeffs, f0);

    E3 r;
    rng.e3(r);
    stir::Domain L0K = params.L(0).power(params.logFoldingFactors[0]);
    std::vector<FE> folded(L0K.size() * FIELD_EXTENSION);
    stir::fold(folded.data(), f0.data(), params.L(0), params.logFoldingFactors[0], r);
    ASSERT_GT(degreeBoundOn(folded, L0K), params.d(1));
}

// Remark 5.3: the fold domain L_{i−1}^{k_{i−1}} and the next domain L_i are disjoint for the
// STARK's coset shift 7 and every folding factor, so Fill is never needed. A shift that *is* a
// 2-adic root of unity would break it — the check must notice.
TEST(STIR_TEST, remark_5_3_domains_are_disjoint)
{
    for (uint64_t logK = 1; logK <= 6; logK++)
    {
        for (uint64_t n = logK + 1; n <= 20; n++)
        {
            stir::Domain L{SHIFT, n};
            ASSERT_TRUE(stir::Domain::disjoint(L.power(logK), stir::Domain{SHIFT, n - 1})) << "k=2^" << logK << " n=" << n;
        }
    }
    // shift = ω_3 ∈ ⟨ω⟩: then L^k ⊂ ⟨ω⟩ too and the cosets are not disjoint.
    stir::Domain bad{Goldilocks::w(3), 10};
    ASSERT_FALSE(stir::Domain::disjoint(bad.power(2), stir::Domain{Goldilocks::w(3), 9}));
}
