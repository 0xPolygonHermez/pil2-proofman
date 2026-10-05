#ifndef PILFFLONK_FR_HPP
#define PILFFLONK_FR_HPP

#include <cstddef>
#include <cstdint>

namespace PilFflonk {

// A BN128 scalar crosses the C API as 32 bytes: a canonical (< r) little-endian integer.
constexpr size_t FR_BYTES = 32;

// A coordinate of a G1 point crosses the C API as 32 bytes: a canonical (< q) little-endian integer.
constexpr size_t FQ_BYTES = 32;

// r, the BN128 scalar field modulus, little-endian: 0x30644e72…43e1f593f0000001.
inline constexpr uint8_t FR_MODULUS_LE[FR_BYTES] = {0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9,
                                                    0x79, 0x48, 0xe8, 0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45,
                                                    0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e, 0x64, 0x30};

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

} // namespace PilFflonk

#endif
