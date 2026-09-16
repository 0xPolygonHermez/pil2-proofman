#include "pilfflonk_api.hpp"

#include <cstring>
#include <gmp.h>
#include <exception>
#include <string>
#include <vector>

#include <alt_bn128.hpp>
#include <ntt_bn128.hpp>

namespace {

using Engine = AltBn128::Engine;
using FrEl = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;

// Per-thread so concurrent callers do not overwrite each other's message.
thread_local std::string lastError;

// Every entry point funnels through this: C++ exceptions must not unwind into
// Rust, so they are caught here and turned into a return code.
template <typename Body>
int guard(Body body) {
    try {
        lastError.clear();
        body();
        return 0;
    } catch (const std::exception &e) {
        lastError = e.what();
        return 1;
    } catch (...) {
        lastError = "unknown error";
        return 2;
    }
}

// Canonical big-endian, fixed width. mpz_export writes left-aligned and
// variable-width, so a short value has to be shifted down into the field --
// otherwise the same number would encode two ways and the transcript, which
// hashes this form, would disagree with the prover.
void writeBE(uint8_t *out, const mpz_t v) {
    std::memset(out, 0, PILFFLONK_FR_BYTES);
    size_t count = 0;
    mpz_export(out, &count, 1, 1, 1, 0, v);
    if (count > 0 && count < PILFFLONK_FR_BYTES) {
        std::memmove(out + (PILFFLONK_FR_BYTES - count), out, count);
        std::memset(out, 0, PILFFLONK_FR_BYTES - count);
    }
}

} // namespace

extern "C" const char *pilfflonk_last_error(void) {
    return lastError.empty() ? nullptr : lastError.c_str();
}

extern "C" int pilfflonk_eval(const uint8_t *coeffs, uint64_t n, const uint8_t *x, uint8_t *out) {
    return guard([&] {
        if (x == nullptr || out == nullptr) throw std::runtime_error("pilfflonk_eval: null buffer");
        if (n > 0 && coeffs == nullptr) throw std::runtime_error("pilfflonk_eval: null coefficients");

        auto &E = Engine::engine;

        // The point arrives canonical; the coefficients are already in the
        // key's representation, so only the point needs converting.
        mpz_t xm;
        mpz_init(xm);
        mpz_import(xm, PILFFLONK_FR_BYTES, 1, 1, 1, 0, x);
        FrEl point;
        E.fr.fromMpz(point, xm);
        mpz_clear(xm);

        // Horner, descending: acc = acc * x + c[i].
        FrEl acc = E.fr.zero();
        for (uint64_t i = n; i > 0; i--) {
            FrEl c;
            std::memcpy(&c, coeffs + (i - 1) * sizeof(FrEl), sizeof(FrEl));
            E.fr.mul(acc, acc, point);
            E.fr.add(acc, acc, c);
        }

        mpz_t r;
        mpz_init(r);
        E.fr.toMpz(r, acc);
        writeBE(out, r);
        mpz_clear(r);
    });
}

extern "C" int pilfflonk_g1_to_bytes_be(const uint8_t *point, uint8_t *out) {
    return guard([&] {
        if (point == nullptr || out == nullptr) throw std::runtime_error("pilfflonk_g1_to_bytes_be: null buffer");

        auto &E = Engine::engine;

        G1PointAffine p;
        std::memcpy(&p, point, PILFFLONK_G1_AFFINE_BYTES);

        if (E.g1.isZero(p)) {
            std::memset(out, 0, 2 * PILFFLONK_FR_BYTES);
            return;
        }

        // toMpz gives the canonical integer; the key holds Montgomery form, so
        // writing the stored limbs directly would emit a different number.
        mpz_t x, y;
        mpz_init(x);
        mpz_init(y);
        E.f1.toMpz(x, p.x);
        E.f1.toMpz(y, p.y);

        writeBE(out, x);
        writeBE(out + PILFFLONK_FR_BYTES, y);

        mpz_clear(x);
        mpz_clear(y);
    });
}

extern "C" int pilfflonk_intt(const uint8_t *src, uint64_t size, uint64_t ncols, uint8_t *out) {
    return guard([&] {
        if (src == nullptr || out == nullptr) throw std::runtime_error("pilfflonk_intt: null buffer");
        if (ncols == 0) throw std::runtime_error("pilfflonk_intt: no columns");
        if (size == 0 || (size & (size - 1)) != 0) {
            throw std::runtime_error("pilfflonk_intt: size " + std::to_string(size) + " is not a power of two");
        }

        // The transform reads and writes the same buffer, so the caller's input
        // is copied rather than cast away its constness.
        auto &E = Engine::engine;

        std::vector<FrEl> work(size * ncols);
        std::memcpy(work.data(), src, size * ncols * sizeof(FrEl));

        // Constructing the transform precomputes the roots for this domain, so
        // it is scoped to the call rather than shared -- a cache keyed by size
        // would be worth it only once a prover runs many transforms.
        NTT_AltBn128 ntt(E, size);
        ntt.INTT(work.data(), work.data(), size, ncols);

        std::memcpy(out, work.data(), size * ncols * sizeof(FrEl));
    });
}

extern "C" int pilfflonk_msm(const uint8_t *ptau, const uint8_t *coeffs, uint64_t n, uint8_t *out) {
    return guard([&] {
        if (out == nullptr) throw std::runtime_error("pilfflonk_msm: out is null");
        if (n == 0) {
            // The empty sum is the point at infinity, which this
            // representation writes as all zeroes.
            std::memset(out, 0, PILFFLONK_G1_AFFINE_BYTES);
            return;
        }
        if (ptau == nullptr || coeffs == nullptr) throw std::runtime_error("pilfflonk_msm: null input buffer");

        auto &E = Engine::engine;

        // multiMulByScalar consumes scalars in canonical form, while the
        // proving key stores them in Montgomery form -- committing the stored
        // values directly would silently compute the wrong point. The copy is
        // also what keeps the caller's buffer const.
        std::vector<FrEl> scalars(n);
        std::memcpy(scalars.data(), coeffs, n * sizeof(FrEl));
        for (uint64_t i = 0; i < n; i++) {
            E.fr.fromMontgomery(scalars[i], scalars[i]);
        }

        // nx/x describe how ffiasm splits the work across threads; one span of
        // the whole range asks it to choose for itself.
        uint64_t lengths[1] = {n};
        G1Point result;
        E.g1.multiMulByScalar(
            result, (G1PointAffine *)ptau, (uint8_t *)scalars.data(), sizeof(FrEl), n, 1, lengths);

        G1PointAffine affine;
        E.g1.copy(affine, result);
        std::memcpy(out, &affine, PILFFLONK_G1_AFFINE_BYTES);
    });
}
