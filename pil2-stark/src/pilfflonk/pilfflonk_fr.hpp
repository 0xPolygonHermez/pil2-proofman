#ifndef PILFFLONK_FR_HPP
#define PILFFLONK_FR_HPP

#include <cstddef>
#include <cstdint>

namespace PilFflonk {

// A BN254 scalar crosses the C API as 32 bytes: a canonical (< r) little-endian integer.
constexpr size_t FR_BYTES = 32;

// A coordinate of a G1 point crosses the C API as 32 bytes: a canonical (< q) little-endian integer.
constexpr size_t FQ_BYTES = 32;

// Whether `bytes`, read as a 256-bit little-endian integer, is below the BN254 scalar modulus r.
// Throws std::runtime_error where ffiasm has no assembly backend (its modulus is then a mock).
bool isCanonicalFr(const uint8_t bytes[FR_BYTES]);

// Whether `bytes`, read as a 256-bit little-endian integer, is below the BN254 base field modulus q.
// Throws std::runtime_error where ffiasm has no assembly backend (its modulus is then a mock).
bool isCanonicalFq(const uint8_t bytes[FQ_BYTES]);

} // namespace PilFflonk

#endif
