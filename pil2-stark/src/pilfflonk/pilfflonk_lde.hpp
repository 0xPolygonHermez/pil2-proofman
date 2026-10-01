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
// no FFT, has more than 2^28 points (pilfflonk/docs/protocol.md#notation).
constexpr uint64_t MAX_NBITS_EXT = 28;

// The shift g of the extended coset g·H' on which Q is evaluated
// (pilfflonk/docs/protocol.md#extended-coset): 5, the smallest quadratic non-residue mod r. It is
// the `nqr` that ffiasm's FFT finds and raises to get its roots of unity, and g^(2^28) != 1, so g
// lies in no subgroup of order 2^k: g·H' meets neither H' nor H, and Z_H does not vanish on it.
// Internal to the prover; the verifier never sees it.
constexpr unsigned int COSET_SHIFT = 5;

class Gpu; // pilfflonk_gpu.hpp

// base^exponent, in Montgomery form, by ffiasm's square-and-multiply.
FrElement power(const FrElement &base, uint64_t exponent);

// out[i] = 1/values[i] for i < n, with one inversion per thread (Montgomery's trick on each
// thread's chunk, which keeps its prefix products in out: out and values must not overlap). Returns
// false, leaving out unspecified, if some value is 0. Allocates nothing: nothing in its parallel
// region can throw.
bool batchInverse(FrElement *out, const FrElement *values, uint64_t n);

// Moves BN254 columns between evaluations on the trace domain H (N = 2^nBits points), their
// coefficients, and evaluations on the extended coset g·H' (N' = 2^nBitsExt points, N <= N').
// It has no NTT of its own: the INTT is rapidsnark's Polynomial::fromEvaluations, and the coset
// transforms are ffiasm's FFT, which has no coset API, around a scaling by the powers of g.
//
// - The roots are ffiasm's, ω_k = 5^((r-1)/k) (pilfflonk/docs/protocol.md#notation): H = <ω_N>
//   and H' = <ω_N'>.
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
//
// With a Gpu (pilfflonk/docs/performance.md#what-runs-on-the-gpu), every FFT and inverse FFT runs
// on it instead (Gpu::ntt, Gpu::intt), one column after another, and the scalings by the powers of
// the shift stay here, around them, on the whole team: the results are the same bit for bit.
class Lde {
public:
    using Engine = AltBn128::Engine;
    using Poly = Polynomial<Engine>;

    // Throws std::invalid_argument unless nBits <= nBitsExt <= MAX_NBITS_EXT, and
    // std::runtime_error where ffiasm has no assembly backend. Builds ffiasm's table of the
    // N' roots of unity, which takes 32·N' bytes. The transforms run on `gpu` if it is not null; it
    // must outlive the Lde, and only a library built with the GPU takes one.
    Lde(uint64_t _nBits, uint64_t _nBitsExt, const Gpu *gpu = nullptr);

    uint64_t domainSize() const { return N; }
    uint64_t extendedSize() const { return NExtended; }
    const Gpu *gpu() const { return device; }

    // INTT: the N evaluations on H of column c, in evals[c], into its N coefficients, followed by
    // blindLength zero coefficients kept for the blinding (pilfflonk/docs/protocol.md#blinding,
    // Poly::blindCoefficients). This is Poly::fromEvaluations with reserved buffers: coefs[c] holds
    // N + blindLength elements, which must be at most N'. The polynomials wrap coefs[c] and do not
    // own it.
    // evals[c] is only read; it is not const because fromEvaluations does not take it as const.
    // Not in place: fromEvaluations clears coefs[c] before it reads evals[c].
    std::vector<std::unique_ptr<Poly>> intt(FrElement *const *evals, FrElement *const *coefs, uint64_t nCols,
                                            uint64_t blindLength = 0) const;

    // LDE: column c, the polynomial with the nCoefs coefficients in coefs[c] (1 <= nCoefs <= N'),
    // into its N' evaluations on g·H', in evals[c]. Coefficient j is scaled by g^j, and the rest up
    // to N' set to zero, before ffiasm's FFT. evals[c] may be coefs[c] itself (in place), if that
    // buffer holds N' elements.
    void extendCoset(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs) const;

    // The LDE on one part of g·H' (pilfflonk/docs/protocol.md#q-in-parts): column c, the polynomial
    // with the nCoefs coefficients in coefs[c] (1 <= nCoefs <= N'), into its S = 2^partBits
    // evaluations on part `part` of the coset, in evals[c] (S elements), for
    // nBits <= partBits <= nBitsExt and part < N'/S. Part p is the S points g·ω_N'^(p + (N'/S)·i),
    // i < S: evaluation i of part p is evaluation p + (N'/S)·i of extendCoset, the same bit for bit,
    // and the N'/S parts are the whole coset; S = N (one coset of H) is the least memory, and
    // S = N' is extendCoset itself. The points of part p are c·ω_S^i for its shift c = g·ω_N'^p, so
    // coefficient j is scaled by c^j and folded into j mod S before ffiasm's FFT of S points.
    // evals[c] may be coefs[c] itself (in place), if that buffer holds max(nCoefs, S) elements.
    void extendCosetPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs,
                         uint64_t partBits, uint64_t part) const;

    // The inverse of extendCoset: the N' evaluations on g·H' of column c, in evals[c], into its N'
    // coefficients, in coefs[c]: ffiasm's inverse FFT, then coefficient j scaled by g^-j.
    // coefs[c] may be evals[c] itself (in place).
    void interpolateCoset(const FrElement *const *evals, FrElement *const *coefs, uint64_t nCols) const;

    // The shift c = g·ω_N'^part of part `part` of the coset, whatever the size of the parts
    // (extendCosetPart): part 0's is g itself. For part < N'.
    FrElement partShift(uint64_t part) const;
    // g^-1, whose powers interpolateCoset scales by.
    const FrElement &shiftInverse() const { return shiftInv; }

private:
    // extendCosetPart once its arguments are checked.
    void extendPart(const FrElement *const *coefs, FrElement *const *evals, uint64_t nCols, uint64_t nCoefs,
                    uint64_t partBits, uint64_t part) const;

    uint64_t N;
    uint64_t NExtended;
    FrElement shift;
    FrElement shiftInv;
    // Sized for N', it also serves N. Behind a pointer because FFT::fft and FFT::ifft are not
    // const, although they only read the FFT's tables.
    std::unique_ptr<FFT<Engine::Fr>> fft;
    const Gpu *device;
};

} // namespace PilFflonk

#endif
