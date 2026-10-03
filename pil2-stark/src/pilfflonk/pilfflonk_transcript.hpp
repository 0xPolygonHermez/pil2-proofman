#ifndef PILFFLONK_TRANSCRIPT_HPP
#define PILFFLONK_TRANSCRIPT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "alt_bn128.hpp"
#include "keccak_256_transcript.hpp"
#include "pilfflonk_fr.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;
using G1Point = AltBn128::Engine::G1Point;

// A G1 point crosses the C API affine, as x‖y: two coordinates of FQ_BYTES each.
constexpr size_t G1_BYTES = 2 * FQ_BYTES;

// Why bytes from the C API cannot be absorbed into the transcript.
enum class AbsorbError {
    None,
    NonCanonical,    // a scalar not below r, or a coordinate not below q
    Infinity,        // (0, 0): ffiasm's affine point at infinity, which Keccak256Transcript does not hash
    NotOnCurve,      // y^2 != x^3 + 3
    ShortCoordinate, // a coordinate below 2^192, which ffiasm does not write as 32 big-endian bytes
};

// Decodes a canonical little-endian scalar (Montgomery form in `out`); `out` is unspecified on error.
AbsorbError decodeFr(const uint8_t bytes[FR_BYTES], FrElement &out);

// Writes `element` as a canonical little-endian scalar.
void encodeFr(const FrElement &element, uint8_t out[FR_BYTES]);

// The Montgomery form of a scalar the caller knows to be canonical (firstNonCanonicalFr): its limbs
// as they are, then ffiasm's toMontgomery. Unlike decodeFr it does not go through GMP, so that whole
// columns can be decoded, in parallel.
FrElement fromCanonicalFr(const uint8_t bytes[FR_BYTES]);

// Decodes an affine point x‖y of canonical little-endian coordinates, one the transcript can absorb
// (pilfflonk/docs/protocol.md#transcript); `out` is unspecified on error. BN254's G1 has cofactor
// 1, so a point on the curve is in the r-torsion group.
AbsorbError decodeG1(const uint8_t bytes[G1_BYTES], G1Point &out);

// Writes `point` affine, as x‖y of canonical little-endian coordinates; the point at infinity as
// (0, 0), ffiasm's affine form of it. decodeG1 of the result tells whether the transcript can
// absorb the point.
void encodeG1(const G1Point &point, uint8_t out[G1_BYTES]);

// The Fiat-Shamir transcript of a proof (pilfflonk/docs/protocol.md#transcript): rapidsnark's
// Keccak256Transcript, unmodified, driven as FflonkProver drives it. Absorbing only appends
// elements; squeeze hashes them. Not safe to use from several threads at once.
class Transcript {
public:
    // Throws std::runtime_error where ffiasm has no assembly backend.
    Transcript();

    // Whether nothing has been absorbed yet. Only a new transcript is empty: squeeze seeds it.
    bool empty() const { return pendingBytes == 0; }

    // False once an absorb or a squeeze has thrown midway (out of memory): the transcript may then
    // hold only part of what the caller asked for, and every later challenge would be wrong.
    bool intact() const { return isIntact; }

    // Whether `n` more scalars (or points) keep the buffer that Keccak256Transcript::getChallenge
    // hashes within the `int` it sizes it with.
    bool fitsScalars(uint64_t n) const;
    bool fitsPoints(uint64_t n) const;

    void absorb(const std::vector<FrElement> &scalars);
    void absorb(const std::vector<G1Point> &points);

    // The challenge keccak256(buffer) mod r over everything absorbed since the previous squeeze;
    // then reset() + addScalar(challenge), as FflonkProver does between rounds. The transcript must
    // not be empty: getChallenge() would declare a zero-length array.
    FrElement squeeze();

private:
    Keccak256Transcript<AltBn128::Engine> transcript;
    // The size of the buffer getChallenge() would declare now, computed as it does.
    uint64_t pendingBytes = 0;
    bool isIntact = true;
};

} // namespace PilFflonk

#endif
