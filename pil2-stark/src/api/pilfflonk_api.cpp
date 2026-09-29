#include "pilfflonk_api.hpp"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <new>
#include <vector>

#include "pilfflonk_fr.hpp"
#include "pilfflonk_transcript.hpp"

namespace {

// Per thread, like errno: concurrent callers never see each other's errors. A fixed buffer, so
// that recording a failure cannot itself fail.
thread_local char lastError[512];

__attribute__((format(printf, 3, 4))) int fail(int status, const char *function, const char *format, ...) noexcept {
    const int prefix = std::snprintf(lastError, sizeof(lastError), "%s: ", function);
    if (prefix >= 0 && static_cast<size_t>(prefix) < sizeof(lastError)) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(lastError + prefix, sizeof(lastError) - prefix, format, args);
        va_end(args);
    }
    return status;
}

// Records the exception being handled as an internal failure. Call only from a catch block.
int failWithCurrentException(const char *function) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return fail(PILFFLONK_ERR_INTERNAL, function, "out of memory");
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
    lastError[0] = '\0';
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
    lastError[0] = '\0';
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

} // namespace

const char *pilfflonk_last_error(void) {
    return lastError;
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
    lastError[0] = '\0';
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
