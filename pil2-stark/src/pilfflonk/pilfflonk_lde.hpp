#ifndef PILFFLONK_LDE_HPP
#define PILFFLONK_LDE_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "alt_bn128.hpp"
#include "fft.hpp"
#include "polynomial.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;

// The 2-adicity of the BN254 scalar field, r - 1 = 2^28 · odd: no domain of roots of unity, and so
// no FFT, has more than 2^28 points (spec A.1).
constexpr uint64_t MAX_NBITS_EXT = 28;

// The shift g of the extended coset g·H' on which Q is evaluated (spec §4.4, "Coset"): 5, the
// smallest quadratic non-residue mod r. It is the `nqr` that ffiasm's FFT finds and raises to get
// its roots of unity, and g^(2^28) != 1, so g lies in no subgroup of order 2^k: g·H' meets
// neither H' nor H, and Z_H does not vanish on it. Internal to the prover; the verifier never
// sees it.
constexpr unsigned int COSET_SHIFT = 5;

// Moves BN254 columns between evaluations on the trace domain H (N = 2^nBits points), their
// coefficients, and evaluations on the extended coset g·H' (N' = 2^nBitsExt points, N <= N').
// It has no NTT of its own: the INTT is rapidsnark's Polynomial::fromEvaluations, and the coset
// transforms are ffiasm's FFT, which has no coset API, around a scaling by the powers of g.
//
// - The roots are ffiasm's, ω_k = 5^((r-1)/k) (spec A.2): H = <ω_N> and H' = <ω_N'>.
// - Coefficients go in increasing degree. Evaluations go in natural order: the i-th is at ω_N^i on
//   H, or at g·ω_N'^i on the coset.
// - Elements are in ffiasm's Montgomery form.
//
// Each function takes nCols >= 1 columns, one buffer per column, and checks all its arguments
// before it writes anything: a bad one throws std::invalid_argument and leaves every buffer as it
// was. Unless a function says otherwise, the output buffers are distinct and overlap no input.
// Only running out of memory can fail after that, and it leaves the outputs unspecified.
//
// Columns run one after another when there are fewer of them than OpenMP threads, each on the
// whole team, and one per thread otherwise (details in pilfflonk_lde.cpp). The results are the
// same bit for bit either way. The const functions may run concurrently with each other.
class Lde {
public:
    using Engine = AltBn128::Engine;
    using Poly = Polynomial<Engine>;

    // Throws std::invalid_argument unless nBits <= nBitsExt <= MAX_NBITS_EXT, and
    // std::runtime_error where ffiasm has no assembly backend. Builds ffiasm's table of the
    // N' roots of unity, which takes 32·N' bytes.
    Lde(uint64_t _nBits, uint64_t _nBitsExt);

    uint64_t domainSize() const { return N; }
    uint64_t extendedSize() const { return NExtended; }

    // INTT: the N evaluations on H of column c, in evals[c], into its N coefficients, followed by
    // blindLength zero coefficients kept for the blinding (spec A.3, Poly::blindCoefficients).
    // This is Poly::fromEvaluations with reserved buffers: coefs[c] holds N + blindLength
    // elements, which must be at most N'. The polynomials wrap coefs[c] and do not own it.
    // evals[c] is only read; it is not const because fromEvaluations does not take it as const.
    // Not in place: fromEvaluations clears coefs[c] before it reads evals[c].
    std::vector<std::unique_ptr<Poly>> intt(FrElement *const *evals, FrElement *const *coefs, uint64_t nCols,
                                            uint64_t blindLength = 0) const;

    // LDE: column c, the polynomial with the nCoefs coefficients in coefs[c] (1 <= nCoefs <= N'),
    // into its N' evaluations on g·H', in evals[c]. Coefficient j is scaled by g^j, and the rest up
    // to N' set to zero, before ffiasm's FFT. evals[c] may be coefs[c] itself (in place), if that
    // buffer holds N' elements.
    void extendCoset(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs) const;

    // The inverse of extendCoset: the N' evaluations on g·H' of column c, in evals[c], into its N'
    // coefficients, in coefs[c]: ffiasm's inverse FFT, then coefficient j scaled by g^-j.
    // coefs[c] may be evals[c] itself (in place).
    void interpolateCoset(const FrElement *const *evals, FrElement *const *coefs, uint64_t nCols) const;

private:
    uint64_t N;
    uint64_t NExtended;
    FrElement shift;
    FrElement shiftInv;
    // Sized for N', it also serves N. Behind a pointer because FFT::fft and FFT::ifft are not
    // const, although they only read the FFT's tables.
    std::unique_ptr<FFT<Engine::Fr>> fft;
};

} // namespace PilFflonk

#endif
