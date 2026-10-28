#ifndef PILFFLONK_FR_HPP
#define PILFFLONK_FR_HPP

#include <cstddef>
#include <cstdint>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;

// A BN128 scalar crosses the C API as 32 bytes: a canonical (< r) little-endian integer.
constexpr size_t FR_BYTES = 32;

// A coordinate of a G1 point crosses the C API as 32 bytes: a canonical (< q) little-endian integer.
constexpr size_t FQ_BYTES = 32;

// r, the BN128 scalar field modulus, little-endian: 0x30644e72…43e1f593f0000001.
inline constexpr uint8_t FR_MODULUS_LE[FR_BYTES] = {0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9,
                                                    0x79, 0x48, 0xe8, 0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45,
                                                    0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e, 0x64, 0x30};

// The 2-adicity of the BN128 scalar field, r - 1 = 2^28 · odd: no domain of roots of unity, and so
// no FFT, has more than 2^28 points (pilfflonk/docs/protocol.md#notation).
constexpr uint64_t MAX_NBITS_EXT = 28;

// Whether `bytes`, read as a 256-bit little-endian integer, is below the BN128 scalar modulus r.
// Throws std::runtime_error where ffiasm has no assembly backend (its modulus is then a mock).
bool isCanonicalFr(const uint8_t bytes[FR_BYTES]);

// Whether `bytes`, read as a 256-bit little-endian integer, is below the BN128 base field modulus q.
// Throws std::runtime_error where ffiasm has no assembly backend (its modulus is then a mock).
bool isCanonicalFq(const uint8_t bytes[FQ_BYTES]);

// The index of the first of the n scalars of FR_BYTES bytes at `bytes` that is not below r, or n if
// every one is. Checks them in parallel. Throws std::runtime_error, before it reads any, where
// ffiasm has no assembly backend.
uint64_t firstNonCanonicalFr(const uint8_t *bytes, uint64_t n);

// base^exponent, in Montgomery form, by ffiasm's square-and-multiply.
FrElement power(const FrElement &base, uint64_t exponent);

// out[i] = 1/values[i] for i < n, with one inversion per thread (Montgomery's trick on each
// thread's chunk, which keeps its prefix products in out: out and values must not overlap). Returns
// false, leaving out unspecified, if some value is 0. Allocates nothing: nothing in its parallel
// region can throw.
bool batchInverse(FrElement *out, const FrElement *values, uint64_t n);

// Whether n divides r − 1, the order of the multiplicative group of Fr: whether Fr has a primitive
// n-th root of unity.
bool dividesRMinusOne(uint64_t n);

// 5^((r − 1)/n), a primitive n-th root of unity, for n dividing r − 1: w_k for n = k and ω_{kN} for
// n = kN (pilfflonk/docs/protocol.md#roots). 5 is the smallest quadratic non-residue, the generator
// ffiasm's FFT and ffjavascript raise, so ω_N is the generator of H they use. Throws
// std::invalid_argument unless n divides r − 1.
FrElement rootOfUnityOfOrder(uint64_t n);

// ω_{2^nBits}: ffiasm's root of unity of order 2^nBits, 5^((r − 1)/2^nBits)
// (pilfflonk/docs/protocol.md#notation). Throws std::invalid_argument for nBits > 28.
FrElement rootOfUnity(uint64_t nBits);

} // namespace PilFflonk

#endif
