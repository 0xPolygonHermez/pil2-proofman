#include "pilfflonk_fr.hpp"

#include <stdexcept>

#include "fq.hpp"
#include "fr.hpp"

namespace PilFflonk {

static_assert(FR_BYTES == RawFr::N64 * sizeof(uint64_t), "a scalar must fill ffiasm's raw Fr limbs exactly");
static_assert(FQ_BYTES == RawFq::N64 * sizeof(uint64_t), "a coordinate must fill ffiasm's raw Fq limbs exactly");
static_assert(RawFr::N64 == RawFq::N64, "both BN128 fields are compared with the same limb loop");

#ifdef __USE_ASSEMBLY__
namespace {

// Whether the little-endian integer `bytes` is below `modulus`, given in ffiasm's raw form:
// little-endian 64-bit limbs. Compares from the most significant limb.
bool isBelow(const uint8_t *bytes, const uint64_t *modulus) {
    for (int limb = RawFr::N64 - 1; limb >= 0; --limb) {
        uint64_t value = 0;
        for (int byte = 7; byte >= 0; --byte) {
            value = (value << 8) | bytes[limb * 8 + byte];
        }
        if (value != modulus[limb]) {
            return value < modulus[limb];
        }
    }
    return false;
}

} // namespace
#endif

bool isCanonicalFr(const uint8_t bytes[FR_BYTES]) {
#ifdef __USE_ASSEMBLY__
    return isBelow(bytes, Fr_rawq);
#else
    throw std::runtime_error("the BN128 scalar field needs ffiasm's assembly backend, not built on this platform");
#endif
}

bool isCanonicalFq(const uint8_t bytes[FQ_BYTES]) {
#ifdef __USE_ASSEMBLY__
    return isBelow(bytes, Fq_rawq);
#else
    throw std::runtime_error("the BN128 base field needs ffiasm's assembly backend, not built on this platform");
#endif
}

uint64_t firstNonCanonicalFr(const uint8_t *bytes, uint64_t n) {
#ifdef __USE_ASSEMBLY__
    uint64_t first = n;
#pragma omp parallel for reduction(min : first)
    for (uint64_t i = 0; i < n; ++i) {
        if (!isBelow(bytes + i * FR_BYTES, Fr_rawq)) {
            first = i < first ? i : first;
        }
    }
    return first;
#else
    (void)bytes;
    (void)n;
    throw std::runtime_error("the BN128 scalar field needs ffiasm's assembly backend, not built on this platform");
#endif
}

} // namespace PilFflonk
