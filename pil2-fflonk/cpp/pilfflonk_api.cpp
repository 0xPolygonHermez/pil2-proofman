#include "pilfflonk_api.hpp"

#include <cstring>
#include <gmp.h>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <alt_bn128.hpp>
#include <ntt_bn128.hpp>
#include <polynomial/cpolynomial.hpp>
#include <polynomial/polynomial.hpp>
#include <keccak_256_transcript.hpp>
#include <msm/msm_bn128.hpp>

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

/// Read a canonical big-endian scalar into the engine's representation.
FrEl readFr(const uint8_t *in) {
    mpz_t v;
    mpz_init(v);
    mpz_import(v, PILFFLONK_FR_BYTES, 1, 1, 1, 0, in);
    FrEl e;
    Engine::engine.fr.fromMpz(e, v);
    mpz_clear(v);
    return e;
}

} // namespace

extern "C" const char *pilfflonk_last_error(void) {
    return lastError.empty() ? nullptr : lastError.c_str();
}

extern "C" void *pilfflonk_transcript_new(void) {
    try {
        lastError.clear();
        return new Keccak256Transcript<Engine>(Engine::engine);
    } catch (const std::exception &e) {
        lastError = e.what();
        return nullptr;
    } catch (...) {
        lastError = "unknown error";
        return nullptr;
    }
}

extern "C" void pilfflonk_transcript_free(void *handle) {
    delete static_cast<Keccak256Transcript<Engine> *>(handle);
}

extern "C" int pilfflonk_transcript_reset(void *handle) {
    return guard([&] {
        if (handle == nullptr) throw std::runtime_error("pilfflonk_transcript_reset: null handle");
        static_cast<Keccak256Transcript<Engine> *>(handle)->reset();
    });
}

extern "C" int pilfflonk_transcript_add_scalar(void *handle, const uint8_t *value) {
    return guard([&] {
        if (handle == nullptr || value == nullptr) throw std::runtime_error("pilfflonk_transcript_add_scalar: null");
        static_cast<Keccak256Transcript<Engine> *>(handle)->addScalar(readFr(value));
    });
}

extern "C" int pilfflonk_transcript_add_commitment(void *handle, const uint8_t *xy) {
    return guard([&] {
        if (handle == nullptr || xy == nullptr) throw std::runtime_error("pilfflonk_transcript_add_commitment: null");

        auto &E = Engine::engine;

        // The transcript takes a projective point; a proof records affine
        // coordinates, so z is one.
        mpz_t v;
        mpz_init(v);
        G1Point p;
        mpz_import(v, PILFFLONK_FR_BYTES, 1, 1, 1, 0, xy);
        E.f1.fromMpz(p.x, v);
        mpz_import(v, PILFFLONK_FR_BYTES, 1, 1, 1, 0, xy + PILFFLONK_FR_BYTES);
        E.f1.fromMpz(p.y, v);
        mpz_clear(v);
        E.f1.copy(p.zz, E.f1.one());
        E.f1.copy(p.zzz, E.f1.one());

        static_cast<Keccak256Transcript<Engine> *>(handle)->addPolCommitment(p);
    });
}

extern "C" int pilfflonk_transcript_challenge(void *handle, uint8_t *out) {
    return guard([&] {
        if (handle == nullptr || out == nullptr) throw std::runtime_error("pilfflonk_transcript_challenge: null");

        FrEl c = static_cast<Keccak256Transcript<Engine> *>(handle)->getChallenge();
        mpz_t v;
        mpz_init(v);
        Engine::engine.fr.toMpz(v, c);
        writeBE(out, v);
        mpz_clear(v);
    });
}

extern "C" int pilfflonk_combine(const uint8_t *stage, uint64_t stage_len, uint64_t stage_cols,
                                 const uint64_t *col_ids, const uint64_t *col_lens, uint64_t n, uint8_t *out,
                                 uint64_t out_cap, uint64_t *out_len) {
    return guard([&] {
        if (stage == nullptr || col_ids == nullptr || col_lens == nullptr || out == nullptr || out_len == nullptr) {
            throw std::runtime_error("pilfflonk_combine: null buffer");
        }
        if (n == 0) throw std::runtime_error("pilfflonk_combine: no columns to pack");
        if (stage_cols == 0) throw std::runtime_error("pilfflonk_combine: the stage has no columns");

        auto &E = Engine::engine;

        // One buffer per slot, read out of the stage with its stride. Held for
        // the lifetime of the call: CPolynomial keeps pointers, it does not copy.
        std::vector<std::vector<FrEl>> slots(n);
        std::vector<std::unique_ptr<Polynomial<Engine>>> polys(n);

        CPolynomial<Engine> combined(E, (int)n);
        for (uint64_t j = 0; j < n; j++) {
            if (col_ids[j] >= stage_cols) {
                throw std::runtime_error("pilfflonk_combine: column " + std::to_string(col_ids[j]) +
                                         " is outside a stage " + std::to_string(stage_cols) + " columns wide");
            }

            // A column that runs past the stage means the caller's idea of the
            // degree disagrees with the key's -- report it rather than reading
            // out of bounds.
            if (col_lens[j] > 0) {
                uint64_t last = col_ids[j] + stage_cols * (col_lens[j] - 1);
                if (last >= stage_len) {
                    throw std::runtime_error("pilfflonk_combine: coefficient " + std::to_string(col_lens[j] - 1) +
                                             " of column " + std::to_string(col_ids[j]) + " runs past the stage (" +
                                             std::to_string(stage_len) + " coefficients)");
                }
            }

            slots[j].resize(col_lens[j]);

            // Construct first: the constructor zeroes the buffer it is handed,
            // so filling it beforehand would be undone. Then read the column
            // out of the stage with its stride, and let fixDegree find the top.
            polys[j] = std::make_unique<Polynomial<Engine>>(E, slots[j].data(), col_lens[j]);
            for (uint64_t i = 0; i < col_lens[j]; i++) {
                std::memcpy(&polys[j]->coef[i], stage + (col_ids[j] + stage_cols * i) * sizeof(FrEl), sizeof(FrEl));
            }
            polys[j]->fixDegree();

            combined.addPolynomial((int)j, polys[j].get());
        }

        std::vector<FrEl> buffer(combined.getDegree() + 1);
        std::unique_ptr<Polynomial<Engine>> result(combined.getPolynomial(buffer.data()));

        uint64_t len = result->getDegree() + 1;
        if (len > out_cap) {
            throw std::runtime_error("pilfflonk_combine: the combined polynomial needs " + std::to_string(len) +
                                     " coefficients but only " + std::to_string(out_cap) + " were provided");
        }

        std::memcpy(out, buffer.data(), len * sizeof(FrEl));
        *out_len = len;
    });
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

        // Polynomial::fastEvaluate rather than a second Horner loop here. The
        // constructor zeroes the buffer it wraps, so the coefficients go in
        // after it, and fixDegree finds the real top.
        std::vector<FrEl> buffer(n == 0 ? 1 : n);
        Polynomial<Engine> poly(E, buffer.data(), n == 0 ? 1 : n);
        if (n > 0) {
            std::memcpy(poly.coef, coeffs, n * sizeof(FrEl));
            poly.fixDegree();
        }
        FrEl acc = poly.fastEvaluate(point);

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

        // Through the shared helper rather than calling multiMulByScalar here,
        // so the CPU MSM has one definition.
        G1Point result = MsmBn128::msmHost(
            E, (G1PointAffine *)ptau, (uint8_t *)scalars.data(), sizeof(FrEl), (unsigned int)n);

        G1PointAffine affine;
        E.g1.copy(affine, result);
        std::memcpy(out, &affine, PILFFLONK_G1_AFFINE_BYTES);
    });
}
