#include "pilfflonk_fr.hpp"

#include <gmp.h>
#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "fq.hpp"
#include "fr.hpp"

namespace PilFflonk {

static_assert(FR_BYTES == RawFr::N64 * sizeof(uint64_t), "a scalar must fill ffiasm's raw Fr limbs exactly");
static_assert(FQ_BYTES == RawFq::N64 * sizeof(uint64_t), "a coordinate must fill ffiasm's raw Fq limbs exactly");
static_assert(RawFr::N64 == RawFq::N64, "both BN128 fields are compared with the same limb loop");

namespace {

using Engine = AltBn128::Engine;

// mpz_divisible_ui_p and mpz_divexact_ui take the divisor as an unsigned long.
static_assert(sizeof(unsigned long) == sizeof(uint64_t), "a 64-bit unsigned long");

// An mpz_t that clears itself.
struct Mpz {
    mpz_t v;
    Mpz() { mpz_init(v); }
    ~Mpz() { mpz_clear(v); }
    Mpz(const Mpz &) = delete;
    Mpz &operator=(const Mpz &) = delete;
};

// r - 1, the order of the multiplicative group of Fr.
void rMinusOne(Mpz &out) {
    Engine &E = Engine::engine;
    E.fr.toMpz(out.v, E.fr.negOne());
}

// base^e by ffiasm's square-and-multiply, which reads e as little-endian bytes.
FrElement power(const FrElement &base, const Mpz &e) {
    uint8_t bytes[FR_BYTES] = {};
    if (mpz_sizeinbase(e.v, 256) > sizeof(bytes)) {
        throw std::logic_error("power: an exponent above 2^256");
    }
    mpz_export(bytes, nullptr, -1, 1, -1, 0, e.v);
    FrElement result;
    Engine::engine.fr.exp(result, base, bytes, sizeof(bytes));
    return result;
}

#ifdef __USE_ASSEMBLY__

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
#endif

} // namespace

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

FrElement power(const FrElement &base, uint64_t exponent) {
    uint8_t littleEndian[sizeof(exponent)];
    for (size_t i = 0; i < sizeof(exponent); ++i) {
        littleEndian[i] = static_cast<uint8_t>(exponent >> (8 * i));
    }
    FrElement result;
    Engine::engine.fr.exp(result, base, littleEndian, sizeof(littleEndian));
    return result;
}

bool batchInverse(FrElement *out, const FrElement *values, uint64_t n) {
    Engine::Fr &fr = Engine::engine.fr;
    bool zero = false;
#pragma omp parallel reduction(|| : zero)
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (n + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(n, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(n, begin + chunk);
        if (begin < end) {
            // out[i] = values[begin] · … · values[i − 1]
            FrElement acc = fr.one();
            for (uint64_t i = begin; i < end; ++i) {
                out[i] = acc;
                fr.mul(acc, acc, values[i]);
            }
            if (fr.isZero(acc)) {
                zero = true;
            } else {
                FrElement inv;
                fr.inv(inv, acc);
                for (uint64_t i = end; i-- > begin;) {
                    FrElement t;
                    fr.mul(t, inv, out[i]);
                    fr.mul(inv, inv, values[i]);
                    out[i] = t;
                }
            }
        }
    }
    return !zero;
}

bool dividesRMinusOne(uint64_t n) {
    Mpz m;
    rMinusOne(m);
    return n != 0 && mpz_divisible_ui_p(m.v, n) != 0;
}

FrElement rootOfUnityOfOrder(uint64_t n) {
    if (!dividesRMinusOne(n)) {
        throw std::invalid_argument("rootOfUnityOfOrder: n = " + std::to_string(n) + " does not divide r - 1");
    }
    Mpz e;
    rMinusOne(e);
    mpz_divexact_ui(e.v, e.v, n);
    FrElement five;
    Engine::engine.fr.fromUI(five, 5);
    return power(five, e);
}

FrElement rootOfUnity(uint64_t nBits) {
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("rootOfUnity: there is no root of unity of order 2^" + std::to_string(nBits) +
                                    ": r - 1 = 2^28 · odd");
    }
    return rootOfUnityOfOrder(uint64_t(1) << nBits);
}

} // namespace PilFflonk
