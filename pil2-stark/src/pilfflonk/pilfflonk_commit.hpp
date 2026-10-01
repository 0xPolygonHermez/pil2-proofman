#ifndef PILFFLONK_COMMIT_HPP
#define PILFFLONK_COMMIT_HPP

#include <cstdint>

#include "alt_bn128.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_srs.hpp"
#include "polynomial.hpp"

namespace PilFflonk {

using Poly = Polynomial<AltBn128::Engine>;

// The length of the buffer pack() needs for k polynomials of at most n coefficients each: k·n
// rounded up to a power of two, the buffer rapidsnark's CPolynomial::getPolynomial clears, which
// pack() keeps asking for. Throws std::invalid_argument if k·n is 0 or above 2^63.
uint64_t packedBufferLength(uint64_t k, uint64_t n);

// The fflonk packing f(X) = Σ_{j<k} p_j(X^k)·X^j of p_j = polys[j]
// (pilfflonk/docs/protocol.md#layout), as rapidsnark's CPolynomial packs: coefficient i·k + j of f
// is coefficient i of p_j, and zero for i > deg p_j. Each p_j's degree must be up to date
// (Poly::fixDegree, which every constructor from evaluations calls): coefficients are copied up to
// it only.
//
// Writes f's coefficients to packed[0, n), in parallel, and returns n = 1 + max_j(k·deg p_j + j),
// the degree bound CPolynomial computes (above deg f if the top coefficients are zero); the rest of
// the buffer is unspecified. bufferLength must be at least packedBufferLength(k, max_j length(p_j)).
//
// Throws std::invalid_argument, before writing anything, if k is 0 or above INT_MAX (CPolynomial's
// bound, which counts in an int), if polys or a p_j is null or has no coefficients, or if the
// buffer is short.
uint64_t pack(Poly *const *polys, uint64_t k, FrElement *packed, uint64_t bufferLength);

// The commitment [f(τ)]₁ of f(X) = Σ_{j<k} p_j(X^k)·X^j, p_j = polys[j]: pack() into a buffer of
// its own, then srs.commit(). Throws std::invalid_argument as pack() does, if polys or a p_j is
// null, and as Srs::commit if f has more coefficients than the SRS has powers.
G1Point commitPacked(const Srs &srs, Poly *const *polys, uint64_t k);

// The commitment [f(τ)]₁ of a fixed f (pilfflonk/docs/README.md#setup-pilfflonk): evals[j] holds
// the N evaluations on H, in natural order and Montgomery form, of column j, the p_j of
// f(X) = Σ_{j<k} p_j(X^k)·X^j. Its coefficients come from lde.intt with no room for blinding
// (constants get none, pilfflonk/docs/protocol.md#blinding); then pack() and srs.commit().
// N = lde.domainSize(). evals is only read, and lde's extended domain is not used.
//
// Throws std::invalid_argument, before any work, if k is 0, if f's k·N coefficients exceed the
// srs.nG1() powers, or if evals or a column is null.
G1Point commitFixed(const Srs &srs, const Lde &lde, FrElement *const *evals, uint64_t k);

} // namespace PilFflonk

#endif
