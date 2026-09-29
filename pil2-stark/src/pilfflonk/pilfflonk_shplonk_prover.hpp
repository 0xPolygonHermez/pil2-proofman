#ifndef PILFFLONK_SHPLONK_PROVER_HPP
#define PILFFLONK_SHPLONK_PROVER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonk {

// One polynomial f of a SHPLONK opening (spec A.5): f(X) = Σ_{j<k} p_j(X^k)·X^j, packed as pack()
// does, opened at ξ·ω_N^s for each offset s in O.
struct ShplonkPolynomial {
    // p_j = components[j] for j < k = components.size(). Not owned: they must outlive the
    // ShplonkProver built on them, unchanged. Each one's degree must be up to date
    // (Poly::fixDegree), as pack() requires.
    std::vector<Poly *> components;
    // O, signed: s opens f at ξ·ω_N^s, the row s rows after ξ's (before it if s < 0). Distinct
    // modulo N, with |s| < N. The order is kept: it is the order of the roots and evaluations.
    std::vector<int64_t> offsets;
};

// What a SHPLONK opening proves: every f_i of the proof, in the global order of spec A.5 (which the
// caller decides; f_0 is the first), and the challenge ξ they are opened at.
struct ShplonkOpening {
    // N = 2^nBits, the size of the trace domain H = <ω_N>.
    uint64_t nBits = 0;
    // The lcm of the k of every f_i (spec A.2.5): ξ = xiSeed^powerW.
    uint64_t powerW = 0;
    FrElement xiSeed = {};
    std::vector<ShplonkPolynomial> polynomials;
};

// What the opening adds to the proof (spec A.4 step 5 and A.5), besides the evaluations.
struct ShplonkProof {
    FrElement alpha = {}; // α_S
    G1Point w = {};       // [W]₁
    FrElement y = {};
    G1Point wp = {}; // [W']₁
};

// The prover side of SHPLONK (spec A.5, §4.4 step 5): pil-fflonk's ShPlonkProver orchestration
// (shplonk.cpp) for an arbitrary ordered list of packed f_i, each with its own k and signed
// offsets, on rapidsnark's Polynomial and M6's pack() and Srs::commit.
//
// The roots (spec A.2.5, derived, not stored): w_k = 5^((r-1)/k), and for offset s the k roots of
// f are x_j = xiSeed^(powerW/k) · ω_{kN}^s · w_k^j, j < k: the x with x^k = ξ·ω_N^s, where
// ω_{kN} = 5^((r-1)/(kN)) (so ω_{kN}^k = ω_N, the generator of H that ffiasm's FFT uses). T_i
// is the union of the roots of f_i's offsets, offset-major: T_i[m·k + j] is x_j of offset O_i[m].
//
// The opening (pil-fflonk's convention, shplonk.cpp:83-101, 263-270): r_i interpolates f_i on
// T_i, and with α = α_S,
//     W(X)  = Σ_i α^i·(f_i(X) - r_i(X)) / Z_{T_i}(X),
//     L(X)  = Σ_i α^i·Z_{T∖T_i}(y)·(f_i(X) - r_i(y)) - Z_T(y)·W(X),
//     W'(X) = L(X) / (Z_{T∖T_0}(y)·(X - y)),
// where Z_T = Π_i Z_{T_i}, repetitions included (a root shared by two f_i counts twice), and
// Z_{T∖T_i} = Z_T / Z_{T_i}. So that the verifier can check e(F - E - J + y·[W'], [1]₂) = e([W'], [τ]₂).
//
// Not safe to use from several threads at once; the Srs may be shared.
class ShplonkProver {
public:
    using Evaluations = std::vector<std::vector<FrElement>>;
    using Interpolants = std::vector<std::unique_ptr<Poly>>;

    // Checks the opening, derives the roots and computes the evaluations. Throws
    // std::invalid_argument, before any work, if there are no polynomials; if nBits exceeds 28;
    // if powerW is not the lcm of every k; if xiSeed is zero; if an f_i has no components, a null
    // one or one of no coefficients; if its k does not divide r - 1 or kN goes beyond the 2-adicity
    // 2^28 of r - 1 (v₂(k) + nBits > 28); if its offsets are none, not distinct modulo N, or some
    // |s| >= N; or if k times the most coefficients of a p_j, or k·|O_i|, exceeds INT_MAX (the
    // most rapidsnark's Polynomial counts in its division and interpolation). Throws
    // std::runtime_error where ffiasm has no assembly backend.
    explicit ShplonkProver(ShplonkOpening opening);

    uint64_t size() const { return fs.size(); }
    uint64_t k(uint64_t i) const { return fs[i].components.size(); }
    // f_i's coefficients as pack() writes them: 1 + max_j(k·deg p_j + j).
    uint64_t nCoefs(uint64_t i) const { return fs[i].nCoefs; }
    // ξ = xiSeed^powerW.
    const FrElement &xi() const { return challengeXi; }
    // ξ·ω_N^s for each s in O_i, in O_i's order.
    const std::vector<FrElement> &points(uint64_t i) const { return fs[i].points; }
    // T_i, offset-major (see above).
    const std::vector<FrElement> &roots(uint64_t i) const { return fs[i].roots; }

    // What the proof carries for f_i (spec A.5): evaluations()[i][m·k + j] = p_j(ξ·ω_N^s) for
    // s = O_i[m], from which the verifier rebuilds f_i(x) = Σ_j p_j(ξ·ω_N^s)·x^j on the roots of
    // s. Offset-major, as the roots. The prover absorbs them before the opening (spec A.4 step 4).
    //
    // The proof's `inv` (D7) is not computed here: see verifierInverse below.
    const Evaluations &evaluations() const { return evals; }

    // The opening, spec A.4 step 5: α_S = squeeze(); W and [W]₁; absorb [W]₁; y = squeeze(); W'
    // and [W']₁. The transcript must hold everything absorbed before the opening (the digest,
    // commitments and evaluations; the caller decides what).
    //
    // Throws std::invalid_argument, before the transcript is touched, if it is empty or an earlier
    // failure left it incomplete, or if an f_i has more coefficients than the srs.nG1() powers
    // [τ^i]₁. After that the transcript has moved on, and on any failure the proof must be
    // abandoned: std::runtime_error if [W]₁ is a point the transcript cannot absorb (the point at
    // infinity, or a coordinate below 2^192, spec A.4) or y is a root of some f_i, each of
    // negligible probability; std::logic_error if a division that must be exact is not (a bug).
    ShplonkProof open(const Srs &srs, Transcript &transcript) const;

    // The steps of open(), for tests.
    //
    // r_i for each f_i: the polynomial of degree below |T_i| through (x, f_i(x)) for x in T_i, with
    // f_i(x) = Σ_j p_j(ξ·ω_N^s)·x^j from evaluations() (the verifier's formula, A.5).
    Interpolants interpolants() const;
    // W for α, as above, in max_i max(nCoefs(i), |T_i|) coefficients. Throws
    // std::invalid_argument if r is not one interpolant per f_i of at most |T_i| coefficients,
    // std::logic_error if an f_i - r_i is not divisible by Z_{T_i}.
    std::unique_ptr<Poly> quotientW(const Interpolants &r, const FrElement &alpha) const;
    // W' for α, y and the W that quotientW(r, alpha) returned. Throws std::invalid_argument if r is
    // not one interpolant per f_i or W does not fit in as many coefficients as quotientW gives it,
    // std::runtime_error if y is in some T_i, std::logic_error if L is not divisible by X - y.
    std::unique_ptr<Poly> quotientWp(const Interpolants &r, const FrElement &alpha, const FrElement &y,
                                     const Poly &W) const;

private:
    struct Entry {
        std::vector<Poly *> components;
        uint64_t maxComponentLength = 0;
        uint64_t nCoefs = 0;
        std::vector<FrElement> points;
        std::vector<FrElement> roots;
    };

    // f_i as an owning polynomial of `length` >= nCoefs(i) coefficients, packed through `scratch`,
    // which holds at least scratchLength() elements.
    std::unique_ptr<Poly> packed(uint64_t i, uint64_t length, FrElement *scratch) const;
    // The buffer pack() needs for the longest f_i.
    uint64_t scratchLength() const;
    // The coefficients W, L and every f_i - r_i fit in: max_i max(nCoefs(i), |T_i|).
    uint64_t workLength() const;

    FrElement challengeXi = {};
    std::vector<Entry> fs;
    Evaluations evals;
};

// a := a / (X^m − β), m >= 1, which must be exact: throws std::logic_error, naming `what`, if it is
// not, leaving `a` unspecified. a's degree must be up to date (Poly::fixDegree) and a must own its
// buffer, which may be replaced. rapidsnark's divByMonic computes only the quotient, so the
// remainder a_j + β·q_j (j < m) is checked here, and it writes below its buffer unless
// deg a >= 2m − 1: of lower degrees, the quotient is computed here. Throws std::invalid_argument
// for m = 0. Used by ShplonkProver; public for its tests.
void divideExactly(Poly &a, uint64_t m, const FrElement &beta, const std::string &what);

// The denominators the verifier inverts in its SHPLONK check at y (pilfflonk/js/src/shplonk.js),
// in this order, for the n f_i of `prover`:
//   1. Z_{T_i}(y) = Π_{x ∈ T_i} (y − x) for i = 1 … n − 1: those of q_i = α^i·Z_{T_0}(y)/Z_{T_i}(y)
//      (computeQuotients);
//   2. for each f_i, i = 0 … n − 1, and each root x_m of T_i in its order (offset-major, as
//      roots(i)): (y − x_m)·Π_{l≠m} (x_m − x_l), that of the Lagrange basis
//      L_m(y) = Z_{T_i}(y)/((y − x_m)·Π_{l≠m} (x_m − x_l)) of r_i(y) (computeR).
// None is zero if y is in no T_i, which ShplonkProver::open checks.
std::vector<FrElement> verifierDenominators(const ShplonkProver &prover, const FrElement &y);

// The proof's `inv` (spec A.6, D7): the inverse of the product of verifierDenominators(prover, y),
// as snarkjs' fflonk prover computes its own (fflonk_prove.js, getMontgomeryBatchedInverse) for its
// Solidity verifier, which checks inv·Π = 1 and recovers every inverse from it with Montgomery's
// trick. Whether the proof keeps it, and with which denominators, is pending a decision (plan §8):
// this function is all there is of it. The JS verifier ignores it, as snarkjs' does. Throws
// std::runtime_error if a denominator is zero.
FrElement verifierInverse(const ShplonkProver &prover, const FrElement &y);

} // namespace PilFflonk

#endif
