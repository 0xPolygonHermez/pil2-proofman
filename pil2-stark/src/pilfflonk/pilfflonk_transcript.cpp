#include "pilfflonk_transcript.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace PilFflonk {

namespace {

// What Keccak256Transcript::getChallenge reserves in its buffer for each element: fr.bytes() per
// scalar and g1.F.bytes() * 2 * 3 per point. It sizes the buffer with an `int`.
constexpr uint64_t SCALAR_BUFFER_BYTES = FR_BYTES;
constexpr uint64_t POINT_BUFFER_BYTES = FQ_BYTES * 2 * 3;
constexpr uint64_t MAX_BUFFER_BYTES = std::numeric_limits<int>::max();

// Whether a canonical little-endian coordinate is below 2^192. RawFq::toRprBE writes only the
// significant 64-bit words of a value, left-aligned, so it writes such a coordinate as a larger
// number instead of as 32 big-endian bytes. Zero would come out right (the buffer starts zeroed),
// but no point on the curve has a zero coordinate: 3 is not a square mod q, and the group has odd
// order, so it has no point of order 2.
bool isShortCoordinate(const uint8_t bytes[FQ_BYTES]) {
    for (size_t i = FQ_BYTES - sizeof(uint64_t); i < FQ_BYTES; ++i) {
        if (bytes[i] != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

AbsorbError decodeFr(const uint8_t bytes[FR_BYTES], FrElement &out) {
    if (!isCanonicalFr(bytes)) {
        return AbsorbError::NonCanonical;
    }
    AltBn128::Engine::engine.fr.fromRprLE(out, bytes, FR_BYTES);
    return AbsorbError::None;
}

FrElement fromCanonicalFr(const uint8_t bytes[FR_BYTES]) {
    FrElement canonical;
    for (int limb = 0; limb < RawFr::N64; ++limb) {
        uint64_t value = 0;
        for (int byte = 7; byte >= 0; --byte) {
            value = (value << 8) | bytes[limb * 8 + byte];
        }
        canonical.v[limb] = value;
    }
    FrElement out;
    AltBn128::Engine::engine.fr.toMontgomery(out, canonical);
    return out;
}

void encodeFr(const FrElement &element, uint8_t out[FR_BYTES]) {
    // toRprLE writes only the significant bytes.
    std::memset(out, 0, FR_BYTES);
    AltBn128::Engine::engine.fr.toRprLE(element, out, FR_BYTES);
}

AbsorbError decodeG1(const uint8_t bytes[G1_BYTES], G1Point &out) {
    const uint8_t *xBytes = bytes;
    const uint8_t *yBytes = bytes + FQ_BYTES;
    if (!isCanonicalFq(xBytes) || !isCanonicalFq(yBytes)) {
        return AbsorbError::NonCanonical;
    }

    AltBn128::Engine &E = AltBn128::Engine::engine;
    AltBn128::Engine::G1PointAffine point;
    E.f1.fromRprLE(point.x, xBytes, FQ_BYTES);
    E.f1.fromRprLE(point.y, yBytes, FQ_BYTES);
    if (E.g1.isZero(point)) {
        return AbsorbError::Infinity;
    }

    AltBn128::Engine::F1Element y2, x2, x3, rhs;
    E.f1.square(y2, point.y);
    E.f1.square(x2, point.x);
    E.f1.mul(x3, x2, point.x);
    E.f1.add(rhs, x3, E.g1.b());
    if (!E.f1.eq(y2, rhs)) {
        return AbsorbError::NotOnCurve;
    }

    if (isShortCoordinate(xBytes) || isShortCoordinate(yBytes)) {
        return AbsorbError::ShortCoordinate;
    }

    E.g1.copy(out, point);
    return AbsorbError::None;
}

void encodeG1(const G1Point &point, uint8_t out[G1_BYTES]) {
    AltBn128::Engine &E = AltBn128::Engine::engine;
    // ffiasm's curve functions take their points as non-const.
    G1Point projective = point;
    AltBn128::Engine::G1PointAffine affine;
    E.g1.copy(affine, projective);
    // toRprLE writes only the significant bytes.
    std::memset(out, 0, G1_BYTES);
    E.f1.toRprLE(affine.x, out, FQ_BYTES);
    E.f1.toRprLE(affine.y, out + FQ_BYTES, FQ_BYTES);
}

Transcript::Transcript() : transcript(AltBn128::Engine::engine) {
#ifndef __USE_ASSEMBLY__
    throw std::runtime_error("the transcript needs ffiasm's assembly backend, not built on this platform");
#endif
}

bool Transcript::fitsScalars(uint64_t n) const {
    return n <= (MAX_BUFFER_BYTES - pendingBytes) / SCALAR_BUFFER_BYTES;
}

bool Transcript::fitsPoints(uint64_t n) const {
    return n <= (MAX_BUFFER_BYTES - pendingBytes) / POINT_BUFFER_BYTES;
}

void Transcript::absorb(const std::vector<FrElement> &scalars) {
    if (!isIntact) {
        throw std::logic_error("absorb on a transcript that an earlier failure left incomplete");
    }
    if (!fitsScalars(scalars.size())) {
        throw std::length_error("absorb beyond the size of Keccak256Transcript's buffer");
    }
    // An exception midway through would leave only some of the scalars absorbed.
    isIntact = false;
    for (const FrElement &scalar : scalars) {
        transcript.addScalar(scalar);
    }
    pendingBytes += scalars.size() * SCALAR_BUFFER_BYTES;
    isIntact = true;
}

void Transcript::absorb(const std::vector<G1Point> &points) {
    if (!isIntact) {
        throw std::logic_error("absorb on a transcript that an earlier failure left incomplete");
    }
    if (!fitsPoints(points.size())) {
        throw std::length_error("absorb beyond the size of Keccak256Transcript's buffer");
    }
    // An exception midway through would leave only some of the points absorbed.
    isIntact = false;
    for (const G1Point &point : points) {
        transcript.addPolCommitment(point);
    }
    pendingBytes += points.size() * POINT_BUFFER_BYTES;
    isIntact = true;
}

FrElement Transcript::squeeze() {
    if (!isIntact) {
        throw std::logic_error("squeeze on a transcript that an earlier failure left incomplete");
    }
    if (empty()) {
        throw std::logic_error("squeeze on an empty transcript");
    }
    FrElement challenge = transcript.getChallenge();
    // An exception between reset() and addScalar() would lose the seed of the next round.
    isIntact = false;
    transcript.reset();
    transcript.addScalar(challenge);
    pendingBytes = SCALAR_BUFFER_BYTES;
    isIntact = true;
    return challenge;
}

} // namespace PilFflonk
