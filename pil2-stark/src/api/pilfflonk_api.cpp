#include "pilfflonk_api.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "pilfflonk_commit.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_transcript.hpp"

namespace {

// Per thread, like errno: concurrent callers never see each other's errors. A fixed buffer, so
// that recording a failure cannot itself fail.
thread_local char lastError[512];
thread_local int lastStatus = PILFFLONK_OK;

// Called by every entry point first: the last error is always that of the latest call.
void clearLastError() noexcept {
    lastError[0] = '\0';
    lastStatus = PILFFLONK_OK;
}

__attribute__((format(printf, 3, 4))) int fail(int status, const char *function, const char *format, ...) noexcept {
    lastStatus = status;
    const int prefix = std::snprintf(lastError, sizeof(lastError), "%s: ", function);
    if (prefix >= 0 && static_cast<size_t>(prefix) < sizeof(lastError)) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(lastError + prefix, sizeof(lastError) - prefix, format, args);
        va_end(args);
    }
    return status;
}

// Records the exception being handled as a failure: an argument the library refused, a file that
// cannot be used, or else an internal failure. Call only from a catch block.
int failWithCurrentException(const char *function) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return fail(PILFFLONK_ERR_INTERNAL, function, "out of memory");
    } catch (const std::invalid_argument &e) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "%s", e.what());
    } catch (const PilFflonk::IoError &e) {
        return fail(PILFFLONK_ERR_IO, function, "%s", e.what());
    } catch (const PilFflonk::FormatError &e) {
        return fail(PILFFLONK_ERR_FORMAT, function, "%s", e.what());
    } catch (const std::exception &e) {
        return fail(PILFFLONK_ERR_INTERNAL, function, "%s", e.what());
    } catch (...) {
        return fail(PILFFLONK_ERR_INTERNAL, function, "unknown exception");
    }
}

// Runs the body of an int-returning entry point: clears the previous error and turns any
// exception into a status code, so that none crosses the C ABI.
template <typename Body>
int guard(const char *function, Body &&body) noexcept {
    clearLastError();
    try {
        return body();
    } catch (...) {
        return failWithCurrentException(function);
    }
}

// Runs the body of an entry point that creates an object: as guard(), but a failure is NULL. A
// body that refuses its arguments records why with fail() and returns nullptr itself.
template <typename Body>
void *guardNew(const char *function, Body &&body) noexcept {
    clearLastError();
    try {
        return body();
    } catch (...) {
        failWithCurrentException(function);
        return nullptr;
    }
}

// Decodes `n` elements of `size` bytes each from `data`, stopping at the first one `decode` refuses.
template <typename Element, typename Decode>
PilFflonk::AbsorbError decodeAll(const uint8_t *data, uint64_t n, size_t size, Decode decode,
                                 std::vector<Element> &elements, uint64_t &refused) {
    elements.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        const PilFflonk::AbsorbError error = decode(data + i * size, elements[i]);
        if (error != PilFflonk::AbsorbError::None) {
            refused = i;
            return error;
        }
    }
    return PilFflonk::AbsorbError::None;
}

int failAbsorb(const char *function, PilFflonk::AbsorbError error, uint64_t element) noexcept {
    switch (error) {
    case PilFflonk::AbsorbError::NonCanonical:
        return fail(PILFFLONK_ERR_NON_CANONICAL, function,
                    "element %" PRIu64 " is not canonical (a scalar not below r, or a coordinate not below q)", element);
    case PilFflonk::AbsorbError::Infinity:
        return fail(PILFFLONK_ERR_INVALID_POINT, function,
                    "element %" PRIu64 " is (0, 0), the point at infinity, which Keccak256Transcript does not hash",
                    element);
    case PilFflonk::AbsorbError::NotOnCurve:
        return fail(PILFFLONK_ERR_INVALID_POINT, function, "element %" PRIu64 " is not on the curve y^2 = x^3 + 3",
                    element);
    case PilFflonk::AbsorbError::ShortCoordinate:
        return fail(PILFFLONK_ERR_INVALID_POINT, function,
                    "element %" PRIu64 " has a coordinate below 2^192, which Keccak256Transcript does not hash as "
                    "32 big-endian bytes",
                    element);
    case PilFflonk::AbsorbError::None:
        break;
    }
    return fail(PILFFLONK_ERR_INTERNAL, function, "element %" PRIu64 " refused for no reason", element);
}

// The index of the first of the n scalars at `bytes` that is not below r, or n if all are.
// isCanonicalFr throws only where ffiasm has no assembly backend: call this after something that
// throws there first, as no exception may leave the parallel region.
uint64_t firstNonCanonicalFr(const uint8_t *bytes, uint64_t n) {
    uint64_t first = n;
#pragma omp parallel for reduction(min : first)
    for (uint64_t i = 0; i < n; ++i) {
        if (!PilFflonk::isCanonicalFr(bytes + i * PilFflonk::FR_BYTES)) {
            first = std::min(first, i);
        }
    }
    return first;
}

// n canonical little-endian scalars into Montgomery form, in parallel: the limbs as they are, then
// ffiasm's toMontgomery (decodeFr goes through GMP, too slow for whole columns).
void decodeCanonicalFr(const uint8_t *bytes, uint64_t n, PilFflonk::FrElement *out) {
    AltBn128::Engine &E = AltBn128::Engine::engine;
#pragma omp parallel for
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t *scalar = bytes + i * PilFflonk::FR_BYTES;
        PilFflonk::FrElement canonical;
        for (int limb = 0; limb < RawFr::N64; ++limb) {
            uint64_t value = 0;
            for (int byte = 7; byte >= 0; --byte) {
                value = (value << 8) | scalar[limb * 8 + byte];
            }
            canonical.v[limb] = value;
        }
        E.fr.toMontgomery(out[i], canonical);
    }
}

// Writes `point` as the C API passes G1 points: affine x‖y, canonical little-endian coordinates;
// the point at infinity as (0, 0), ffiasm's affine form of it.
void encodeG1(PilFflonk::G1Point &point, uint8_t out[PilFflonk::G1_BYTES]) {
    AltBn128::Engine &E = AltBn128::Engine::engine;
    PilFflonk::G1PointAffine affine;
    E.g1.copy(affine, point);
    // toRprLE writes only the significant bytes.
    std::memset(out, 0, PilFflonk::G1_BYTES);
    E.f1.toRprLE(affine.x, out, PilFflonk::FQ_BYTES);
    E.f1.toRprLE(affine.y, out + PilFflonk::FQ_BYTES, PilFflonk::FQ_BYTES);
}

} // namespace

const char *pilfflonk_last_error(void) {
    return lastError;
}

int pilfflonk_last_status(void) {
    return lastStatus;
}

int pilfflonk_fr_check_canonical(const uint8_t scalar[32]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (scalar == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "scalar is NULL");
        }
        if (!PilFflonk::isCanonicalFr(scalar)) {
            return fail(PILFFLONK_ERR_NON_CANONICAL, function, "scalar is not below the BN254 scalar modulus r");
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

void *pilfflonk_transcript_new(void) {
    return guardNew(__func__, [] { return static_cast<void *>(new PilFflonk::Transcript()); });
}

void pilfflonk_transcript_free(void *transcript) {
    clearLastError();
    delete static_cast<PilFflonk::Transcript *>(transcript);
}

int pilfflonk_transcript_absorb(void *transcript, const uint8_t *data, uint64_t n, uint32_t kind) {
    const char *function = __func__;
    return guard(function, [&] {
        if (transcript == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "transcript is NULL");
        }
        if (kind != PILFFLONK_TRANSCRIPT_FR && kind != PILFFLONK_TRANSCRIPT_G1) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "kind %" PRIu32 " is neither PILFFLONK_TRANSCRIPT_FR nor PILFFLONK_TRANSCRIPT_G1", kind);
        }
        if (data == nullptr && n != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "data is NULL");
        }
        PilFflonk::Transcript &t = *static_cast<PilFflonk::Transcript *>(transcript);
        if (!t.intact()) {
            return fail(PILFFLONK_ERR_INTERNAL, function, "an earlier failure left the transcript incomplete");
        }
        const bool scalars = kind == PILFFLONK_TRANSCRIPT_FR;
        if (!(scalars ? t.fitsScalars(n) : t.fitsPoints(n))) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "%" PRIu64 " more elements exceed what Keccak256Transcript can hash at once", n);
        }

        uint64_t refused = 0;
        if (scalars) {
            std::vector<PilFflonk::FrElement> elements;
            const PilFflonk::AbsorbError error =
                decodeAll(data, n, PilFflonk::FR_BYTES, PilFflonk::decodeFr, elements, refused);
            if (error != PilFflonk::AbsorbError::None) {
                return failAbsorb(function, error, refused);
            }
            t.absorb(elements);
        } else {
            std::vector<PilFflonk::G1Point> elements;
            const PilFflonk::AbsorbError error =
                decodeAll(data, n, PilFflonk::G1_BYTES, PilFflonk::decodeG1, elements, refused);
            if (error != PilFflonk::AbsorbError::None) {
                return failAbsorb(function, error, refused);
            }
            t.absorb(elements);
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_transcript_squeeze(void *transcript, uint8_t out[32]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (transcript == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "transcript is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        PilFflonk::Transcript &t = *static_cast<PilFflonk::Transcript *>(transcript);
        if (!t.intact()) {
            return fail(PILFFLONK_ERR_INTERNAL, function, "an earlier failure left the transcript incomplete");
        }
        if (t.empty()) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "nothing has been absorbed yet");
        }
        PilFflonk::encodeFr(t.squeeze(), out);
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_srs_from_ptau(const char *ptau_path, uint64_t n_g1, const char *srs_path) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ptau_path == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ptau_path is NULL");
        }
        if (srs_path == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "srs_path is NULL");
        }
        PilFflonk::Srs::fromPtau(ptau_path, n_g1).save(srs_path);
        return static_cast<int>(PILFFLONK_OK);
    });
}

void *pilfflonk_srs_load(const char *srs_path) {
    const char *function = __func__;
    return guardNew(function, [&]() -> void * {
        if (srs_path == nullptr) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "srs_path is NULL");
            return nullptr;
        }
        return new PilFflonk::Srs(PilFflonk::Srs::load(srs_path));
    });
}

void pilfflonk_srs_free(void *srs) {
    clearLastError();
    delete static_cast<PilFflonk::Srs *>(srs);
}

int pilfflonk_commit_fixed(const void *srs, uint64_t n_bits, uint64_t k, const uint8_t *evals, uint8_t out_g1[64]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (srs == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "srs is NULL");
        }
        if (evals == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "evals is NULL");
        }
        if (out_g1 == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_g1 is NULL");
        }
        if (k == 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "k = 0: f packs no columns");
        }
        if (n_bits > PilFflonk::MAX_NBITS_EXT) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "n_bits = %" PRIu64 " exceeds %" PRIu64 ", the 2-adicity of the BN254 scalar field", n_bits,
                        PilFflonk::MAX_NBITS_EXT);
        }
        const PilFflonk::Srs &s = *static_cast<const PilFflonk::Srs *>(srs);
        const uint64_t N = uint64_t(1) << n_bits;
        // Also bounds k·N, and so the size of evals, by the SRS's size.
        if (k > s.nG1() / N) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "f's k·N = %" PRIu64 "·%" PRIu64 " coefficients exceed the %" PRIu64
                        " powers [τ^i]₁ of the SRS",
                        k, N, s.nG1());
        }
        // Throws where ffiasm has no assembly backend, before firstNonCanonicalFr would.
        const PilFflonk::Lde lde(n_bits, n_bits);

        const uint64_t n = k * N;
        const uint64_t refused = firstNonCanonicalFr(evals, n);
        if (refused < n) {
            return fail(PILFFLONK_ERR_NON_CANONICAL, function,
                        "scalar %" PRIu64 " (column %" PRIu64 ", row %" PRIu64 ") is not below r", refused,
                        refused / N, refused % N);
        }
        std::unique_ptr<PilFflonk::FrElement[]> elements(new PilFflonk::FrElement[n]);
        decodeCanonicalFr(evals, n, elements.get());
        std::vector<PilFflonk::FrElement *> columns(k);
        for (uint64_t j = 0; j < k; ++j) {
            columns[j] = elements.get() + j * N;
        }

        PilFflonk::G1Point commitment = PilFflonk::commitFixed(s, lde, columns.data(), k);
        encodeG1(commitment, out_g1);
        return static_cast<int>(PILFFLONK_OK);
    });
}
