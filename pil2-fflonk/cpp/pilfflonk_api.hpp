#ifndef PIL2_FFLONK_API_HPP
#define PIL2_FFLONK_API_HPP

#include <stdint.h>

// The C surface Rust drives the prover through.
//
// Rust owns setup and orchestration -- it already reads the proving key
// byte-exactly -- while the field and curve arithmetic stays in C++ over
// ffiasm. This header is the seam.
//
// Buffers cross the boundary in **ffiasm's own representation**, not a
// canonical one: scalars are `RawFr::Element` (four little-endian 64-bit limbs,
// Montgomery form) and points are `G1PointAffine` (two `RawFq::Element`, also
// Montgomery). That is exactly how the proving key stores them, so Rust passes
// the key's sections straight through without interpreting them. Converting to
// a canonical form at the boundary would mean Rust reimplementing Montgomery
// reduction for two different primes, for no benefit.
//
// Every entry point returns 0 on success and non-zero on failure, and never
// throws across the boundary -- an exception reaching Rust is undefined
// behaviour.

#ifdef __cplusplus
extern "C" {
#endif

// Sizes of the representations above, in bytes.
#define PILFFLONK_FR_BYTES 32
#define PILFFLONK_G1_AFFINE_BYTES 64

// Multi-scalar multiplication: `out = sum_j coeffs[j] * ptau[j]`.
//
// `coeffs` is `n` scalars and `ptau` is `n` affine points, both in the
// representation described above. `out` receives one affine point
// (PILFFLONK_G1_AFFINE_BYTES); the point at infinity is written as all zeroes.
//
// This is the operation every commitment is built from: committing to a
// polynomial is its coefficients against the key's powers of tau.
int pilfflonk_msm(const uint8_t *ptau, const uint8_t *coeffs, uint64_t n, uint8_t *out);

// Interpolate columns of evaluations into coefficients, in place of `out`.
//
// `src` is `size` rows of `ncols` values each, row-major -- value `c` of row
// `r` at `r * ncols + c`, which is how the trace is stored. Every column is
// interpolated over the same domain of `size` points, and `out` receives the
// coefficients in the same layout.
//
// `size` must be a power of two: the domain is the `size`-th roots of unity.
//
// This is how a committed stage gets from the trace to something committable,
// since a commitment is over coefficients.
int pilfflonk_intt(const uint8_t *src, uint64_t size, uint64_t ncols, uint8_t *out);

// Convert an affine point to canonical big-endian coordinates.
//
// `point` is PILFFLONK_G1_AFFINE_BYTES in the key's representation; `out`
// receives 2 * PILFFLONK_FR_BYTES, x then y, each big-endian and fully
// reduced. The point at infinity becomes all zeroes.
//
// This is the form a proof records and the form the Fiat-Shamir transcript
// hashes, so it is how a commitment leaves the library.
int pilfflonk_g1_to_bytes_be(const uint8_t *point, uint8_t *out);

// Evaluate a polynomial at a point.
//
// `coeffs` is `n` coefficients in the key's representation, ascending degree.
// `x` and `out` are PILFFLONK_FR_BYTES each, canonical big-endian -- the form a
// proof records, so a caller need not know the key's representation to ask for
// an opening.
//
// This is what produces the claimed openings a proof carries.
int pilfflonk_eval(const uint8_t *coeffs, uint64_t n, const uint8_t *x, uint8_t *out);

// The error text for the last failed call on this thread, or NULL if the last
// call succeeded. Owned by the library and valid until the next call.
const char *pilfflonk_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // PIL2_FFLONK_API_HPP
