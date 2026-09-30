#include "pilfflonk_api.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "keccak_wrapper.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_rng.hpp"
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
    } catch (const PilFflonk::UnsatisfiedError &e) {
        return fail(PILFFLONK_ERR_UNSATISFIED, function, "%s", e.what());
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

// n canonical little-endian scalars into Montgomery form, in parallel (fromCanonicalFr): the caller
// checks them first (firstNonCanonicalFr).
void decodeCanonicalFr(const uint8_t *bytes, uint64_t n, PilFflonk::FrElement *out) {
#pragma omp parallel for
    for (uint64_t i = 0; i < n; ++i) {
        out[i] = PilFflonk::fromCanonicalFr(bytes + i * PilFflonk::FR_BYTES);
    }
}

// The n scalars at `bytes` (canonical little-endian, 32 bytes each) into `out`, or false with the
// index of the first one not below r in `refused`.
bool decodeScalars(const uint8_t *bytes, uint64_t n, std::vector<PilFflonk::FrElement> &out, uint64_t &refused) {
    out.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        if (PilFflonk::decodeFr(bytes + i * PilFflonk::FR_BYTES, out[i]) != PilFflonk::AbsorbError::None) {
            refused = i;
            return false;
        }
    }
    return true;
}

// Writes the n points to `out`, 64 bytes each (encodeG1).
void encodePoints(const std::vector<PilFflonk::G1Point> &points, uint8_t *out) {
    for (size_t i = 0; i < points.size(); ++i) {
        PilFflonk::encodeG1(points[i], out + i * PilFflonk::G1_BYTES);
    }
}

// Writes an affine G2 point of the SRS (Montgomery form) as x.c0‖x.c1‖y.c0‖y.c1, each coordinate
// canonical little-endian.
void encodeG2(const PilFflonk::G2PointAffine &point, uint8_t out[PilFflonk::SRS_G2_BYTES]) {
    AltBn128::Engine &E = AltBn128::Engine::engine;
    // toRprLE writes only the significant bytes.
    std::memset(out, 0, PilFflonk::SRS_G2_BYTES);
    E.f1.toRprLE(point.x.a, out, PilFflonk::FQ_BYTES);
    E.f1.toRprLE(point.x.b, out + PilFflonk::FQ_BYTES, PilFflonk::FQ_BYTES);
    E.f1.toRprLE(point.y.a, out + 2 * PilFflonk::FQ_BYTES, PilFflonk::FQ_BYTES);
    E.f1.toRprLE(point.y.b, out + 3 * PilFflonk::FQ_BYTES, PilFflonk::FQ_BYTES);
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

int pilfflonk_keccak256(const uint8_t *data, uint64_t len, uint8_t out[32]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        if (data == nullptr && len != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "data is NULL");
        }
        // keccak() takes the size as an int64_t.
        if (len > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "len = %" PRIu64 " exceeds 2^63 - 1", len);
        }
        // keccak() only reads its input, although it takes it as non-const; with len = 0 it reads
        // nothing, so any pointer will do.
        uint8_t nothing = 0;
        void *input = data == nullptr ? &nothing : const_cast<uint8_t *>(data);
        if (keccak(input, static_cast<int64_t>(len), out, 32) != 32) {
            return fail(PILFFLONK_ERR_INTERNAL, function, "keccak_wrapper refused a 32-byte output");
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

namespace {

// pilfflonk_srs_g2 and pilfflonk_ctx_srs_g2, once the SRS is found.
int srsG2(const char *function, const PilFflonk::Srs &srs, uint64_t i, uint8_t *out_g2) {
    if (out_g2 == nullptr) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_g2 is NULL");
    }
    if (i >= PilFflonk::Srs::N_G2) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                    "i = %" PRIu64 ": an SRS holds [1]₂ (i = 0) and [τ]₂ (i = 1) only", i);
    }
    encodeG2(srs.g2(i), out_g2);
    return static_cast<int>(PILFFLONK_OK);
}

} // namespace

int pilfflonk_srs_g2(const void *srs, uint64_t i, uint8_t out_g2[128]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (srs == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "srs is NULL");
        }
        return srsG2(function, *static_cast<const PilFflonk::Srs *>(srs), i, out_g2);
    });
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
        const PilFflonk::Lde lde(n_bits, n_bits);

        const uint64_t n = k * N;
        const uint64_t refused = PilFflonk::firstNonCanonicalFr(evals, n);
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
        PilFflonk::encodeG1(commitment, out_g1);
        return static_cast<int>(PILFFLONK_OK);
    });
}

// -------------------------------------------------------------------------------------------------
// The prover
// -------------------------------------------------------------------------------------------------

void *pilfflonk_ctx_new(const char *proving_key_dir) {
    const char *function = __func__;
    return guardNew(function, [&]() -> void * {
        if (proving_key_dir == nullptr) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "proving_key_dir is NULL");
            return nullptr;
        }
        return PilFflonk::ProvingKey::load(proving_key_dir).release();
    });
}

void pilfflonk_ctx_free(void *ctx) {
    clearLastError();
    delete static_cast<PilFflonk::ProvingKey *>(ctx);
}

int pilfflonk_ctx_n_bits_ext(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint64_t *out) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ctx == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        *out = static_cast<const PilFflonk::ProvingKey *>(ctx)->air(airgroup_id, air_id).degrees().nBitsExt;
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_ctx_srs_g2(const void *ctx, uint64_t i, uint8_t out_g2[128]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ctx == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
        }
        return srsG2(function, static_cast<const PilFflonk::ProvingKey *>(ctx)->srs(), i, out_g2);
    });
}

int pilfflonk_ctx_fixed_commitments(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint8_t *out_g1,
                                    uint64_t n) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ctx == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
        }
        if (out_g1 == nullptr && n != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_g1 is NULL");
        }
        const PilFflonk::ProvingKey &pk = *static_cast<const PilFflonk::ProvingKey *>(ctx);
        const PilFflonk::AirKey &air = pk.air(airgroup_id, air_id);
        if (n != air.nFixedF()) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "n = %" PRIu64 ", and %s has %" PRIu64 " fixed f", n,
                        air.name().c_str(), air.nFixedF());
        }
        encodePoints(air.fixedCommitments(pk.srs()), out_g1);
        return static_cast<int>(PILFFLONK_OK);
    });
}

void *pilfflonk_instance_new(const void *ctx, uint64_t airgroup_id, uint64_t air_id, const uint8_t *stage1,
                             uint64_t stage1_len, const uint8_t *air_values, uint64_t n_air_values,
                             const uint8_t *publics, uint64_t n_publics, const uint8_t *proof_values,
                             uint64_t n_proof_values, const uint8_t *insecure_blinding_seed) {
    const char *function = __func__;
    return guardNew(function, [&]() -> void * {
        if (ctx == nullptr) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
            return nullptr;
        }
        const struct {
            const uint8_t *data;
            uint64_t n;
            const char *name;
        } arrays[] = {{stage1, stage1_len, "stage1"},
                      {air_values, n_air_values, "air_values"},
                      {publics, n_publics, "publics"},
                      {proof_values, n_proof_values, "proof_values"}};
        for (const auto &array : arrays) {
            if (array.data == nullptr && array.n != 0) {
                fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "%s is NULL", array.name);
                return nullptr;
            }
        }
        const PilFflonk::ProvingKey &pk = *static_cast<const PilFflonk::ProvingKey *>(ctx);
        const PilFflonk::AirKey &air = pk.air(airgroup_id, air_id);

        // The witness's scalars are checked here, to report them as not canonical; the rest of its
        // shape, by the Instance.
        const uint64_t nCols = air.witnessColumns().size();
        if (stage1_len % PilFflonk::FR_BYTES != 0) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "stage1_len = %" PRIu64 " is not a whole number of scalars",
                 stage1_len);
            return nullptr;
        }
        const uint64_t nScalars = stage1_len / PilFflonk::FR_BYTES;
        const uint64_t refusedCell = PilFflonk::firstNonCanonicalFr(stage1, nScalars);
        if (refusedCell < nScalars) {
            fail(PILFFLONK_ERR_NON_CANONICAL, function,
                 "the stage-1 witness at row %" PRIu64 ", column %" PRIu64 " is not below r",
                 nCols == 0 ? 0 : refusedCell / nCols, nCols == 0 ? 0 : refusedCell % nCols);
            return nullptr;
        }
        std::vector<PilFflonk::FrElement> values[3];
        const uint8_t *bytes[3] = {air_values, publics, proof_values};
        const uint64_t counts[3] = {n_air_values, n_publics, n_proof_values};
        const char *names[3] = {"air_values", "publics", "proof_values"};
        for (int i = 0; i < 3; ++i) {
            uint64_t refused = 0;
            if (!decodeScalars(bytes[i], counts[i], values[i], refused)) {
                fail(PILFFLONK_ERR_NON_CANONICAL, function, "%s[%" PRIu64 "] is not below r", names[i], refused);
                return nullptr;
            }
        }
        std::unique_ptr<PilFflonk::BlindingSource> blinding =
            insecure_blinding_seed == nullptr ? std::make_unique<PilFflonk::BlindingRng>()
                                              : std::make_unique<PilFflonk::BlindingRng>(insecure_blinding_seed);
        return new PilFflonk::Instance(pk, airgroup_id, air_id, stage1, stage1_len, std::move(values[0]),
                                       std::move(values[1]), std::move(values[2]), std::move(blinding));
    });
}

void pilfflonk_instance_free(void *instance) {
    clearLastError();
    delete static_cast<PilFflonk::Instance *>(instance);
}

namespace {

// commit_stage and commit_q: their arguments, the call, and the commitments written out.
template <typename Commit>
int commitCall(const char *function, void *instance, uint64_t stage, const uint8_t *challenges, uint64_t n_challenges,
               uint8_t *out_g1, uint64_t n_out, Commit commit) {
    if (instance == nullptr) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instance is NULL");
    }
    if (challenges == nullptr && n_challenges != 0) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "challenges is NULL");
    }
    if (out_g1 == nullptr && n_out != 0) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_g1 is NULL");
    }
    PilFflonk::Instance &inst = *static_cast<PilFflonk::Instance *>(instance);
    const uint64_t expected = inst.nCommitments(stage);
    if (n_out != expected) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                    "n_out = %" PRIu64 ", and stage %" PRIu64 " has %" PRIu64 " f to commit", n_out, stage, expected);
    }
    std::vector<PilFflonk::FrElement> values;
    uint64_t refused = 0;
    if (!decodeScalars(challenges, n_challenges, values, refused)) {
        return fail(PILFFLONK_ERR_NON_CANONICAL, function, "challenges[%" PRIu64 "] is not below r", refused);
    }
    encodePoints(commit(inst, values), out_g1);
    return static_cast<int>(PILFFLONK_OK);
}

} // namespace

int pilfflonk_commit_stage(void *instance, uint32_t stage, const uint8_t *challenges, uint64_t n_challenges,
                           uint8_t *out_g1, uint64_t n_out) {
    const char *function = __func__;
    return guard(function, [&] {
        return commitCall(function, instance, stage, challenges, n_challenges, out_g1, n_out,
                          [&](PilFflonk::Instance &inst, const std::vector<PilFflonk::FrElement> &values) {
                              return inst.commitStage(stage, values);
                          });
    });
}

int pilfflonk_commit_q(void *instance, const uint8_t *challenges, uint64_t n_challenges, uint8_t *out_g1,
                       uint64_t n_out) {
    const char *function = __func__;
    return guard(function, [&] {
        const uint64_t qStage =
            instance == nullptr ? 0 : static_cast<PilFflonk::Instance *>(instance)->air().info().qStage();
        return commitCall(function, instance, qStage, challenges, n_challenges, out_g1, n_out,
                          [&](PilFflonk::Instance &inst, const std::vector<PilFflonk::FrElement> &values) {
                              return inst.commitQ(values);
                          });
    });
}

int pilfflonk_instance_set_q_part_bits(void *instance, uint64_t part_bits) {
    const char *function = __func__;
    return guard(function, [&] {
        if (instance == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instance is NULL");
        }
        static_cast<PilFflonk::Instance *>(instance)->setQPartBits(part_bits);
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_instance_column(const void *instance, uint32_t stage, uint64_t stage_pos, uint8_t *out, uint64_t n) {
    const char *function = __func__;
    return guard(function, [&] {
        if (instance == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instance is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        const PilFflonk::Instance &inst = *static_cast<const PilFflonk::Instance *>(instance);
        const uint64_t N = inst.air().n();
        if (n != N) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "n = %" PRIu64 ", and %s has %" PRIu64 " rows", n,
                        inst.air().name().c_str(), N);
        }
        const PilFflonk::FrElement *values = inst.column(stage, stage_pos);
        for (uint64_t i = 0; i < N; ++i) {
            PilFflonk::encodeFr(values[i], out + i * PilFflonk::FR_BYTES);
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

void *pilfflonk_opening_new(const void *const *instances, uint64_t n_instances, const uint8_t xi_seed[32]) {
    const char *function = __func__;
    return guardNew(function, [&]() -> void * {
        if (instances == nullptr) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instances is NULL");
            return nullptr;
        }
        if (xi_seed == nullptr) {
            fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "xi_seed is NULL");
            return nullptr;
        }
        PilFflonk::FrElement seed;
        if (PilFflonk::decodeFr(xi_seed, seed) != PilFflonk::AbsorbError::None) {
            fail(PILFFLONK_ERR_NON_CANONICAL, function, "xi_seed is not below r");
            return nullptr;
        }
        std::vector<const PilFflonk::Instance *> list(n_instances);
        for (uint64_t i = 0; i < n_instances; ++i) {
            list[i] = static_cast<const PilFflonk::Instance *>(instances[i]);
        }
        return new PilFflonk::Opening(list, seed);
    });
}

void pilfflonk_opening_free(void *opening) {
    clearLastError();
    delete static_cast<PilFflonk::Opening *>(opening);
}

uint64_t pilfflonk_opening_n_evaluations(const void *opening) {
    clearLastError();
    if (opening == nullptr) {
        fail(PILFFLONK_ERR_INVALID_ARGUMENT, __func__, "opening is NULL");
        return 0;
    }
    return static_cast<const PilFflonk::Opening *>(opening)->evaluations().size();
}

int pilfflonk_opening_evaluations(const void *opening, uint64_t n, uint8_t *out) {
    const char *function = __func__;
    return guard(function, [&] {
        if (opening == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "opening is NULL");
        }
        if (out == nullptr && n != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        const std::vector<PilFflonk::FrElement> &evaluations =
            static_cast<const PilFflonk::Opening *>(opening)->evaluations();
        if (n != evaluations.size()) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "n = %" PRIu64 ", and the proof has %zu evaluations",
                        n, evaluations.size());
        }
        for (uint64_t i = 0; i < n; ++i) {
            PilFflonk::encodeFr(evaluations[i], out + i * PilFflonk::FR_BYTES);
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_opening_q(const void *opening, uint64_t instance, uint8_t out[32]) {
    const char *function = __func__;
    return guard(function, [&] {
        if (opening == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "opening is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        PilFflonk::encodeFr(static_cast<const PilFflonk::Opening *>(opening)->q(instance), out);
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_opening_open(const void *opening, void *transcript, uint8_t out_w[64], uint8_t out_wp[64],
                           uint8_t out_inv[32], uint8_t out_inv_zh[32]) {
    const char *function = __func__;
    return guard(function, [&] {
        const struct {
            const void *pointer;
            const char *name;
        } arguments[] = {{opening, "opening"},     {transcript, "transcript"}, {out_w, "out_w"},
                         {out_wp, "out_wp"},       {out_inv, "out_inv"},       {out_inv_zh, "out_inv_zh"}};
        for (const auto &argument : arguments) {
            if (argument.pointer == nullptr) {
                return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "%s is NULL", argument.name);
            }
        }
        const PilFflonk::Opening::Proof proof = static_cast<const PilFflonk::Opening *>(opening)->open(
            *static_cast<PilFflonk::Transcript *>(transcript));
        PilFflonk::encodeG1(proof.shplonk.w, out_w);
        PilFflonk::encodeG1(proof.shplonk.wp, out_wp);
        PilFflonk::encodeFr(proof.inv, out_inv);
        PilFflonk::encodeFr(proof.invZh, out_inv_zh);
        return static_cast<int>(PILFFLONK_OK);
    });
}

// -------------------------------------------------------------------------------------------------
// The check
// -------------------------------------------------------------------------------------------------

namespace {

// Constraint `index` of section 2 of the .bin of an AIR of ctx (not NULL). Throws
// std::invalid_argument if there is no such AIR or constraint.
const PilFflonk::ParserParams &constraintOf(const void *ctx, uint64_t airgroupId, uint64_t airId, uint64_t index) {
    const PilFflonk::AirKey &air = static_cast<const PilFflonk::ProvingKey *>(ctx)->air(airgroupId, airId);
    const std::vector<PilFflonk::ParserParams> &constraints = air.bin().constraintsInfoDebug;
    if (index >= constraints.size()) {
        throw std::invalid_argument(air.name() + " has no constraint " + std::to_string(index) + ", of " +
                                    std::to_string(constraints.size()));
    }
    return constraints[index];
}

} // namespace

int pilfflonk_ctx_n_constraints(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint64_t *out) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ctx == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        const PilFflonk::AirKey &air = static_cast<const PilFflonk::ProvingKey *>(ctx)->air(airgroup_id, air_id);
        *out = air.bin().constraintsInfoDebug.size();
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_ctx_constraint(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint64_t index,
                             uint64_t *stage, uint64_t *first_row, uint64_t *last_row, uint32_t *im_pol,
                             uint64_t *line_len) {
    const char *function = __func__;
    return guard(function, [&] {
        const struct {
            const void *pointer;
            const char *name;
        } arguments[] = {{ctx, "ctx"},           {stage, "stage"},   {first_row, "first_row"},
                         {last_row, "last_row"}, {im_pol, "im_pol"}, {line_len, "line_len"}};
        for (const auto &argument : arguments) {
            if (argument.pointer == nullptr) {
                return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "%s is NULL", argument.name);
            }
        }
        const PilFflonk::ParserParams &constraint = constraintOf(ctx, airgroup_id, air_id, index);
        *stage = constraint.stage;
        *first_row = constraint.firstRow;
        *last_row = constraint.lastRow;
        *im_pol = constraint.imPol ? 1 : 0;
        *line_len = constraint.line.size();
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_ctx_constraint_line(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint64_t index,
                                  uint8_t *out, uint64_t n) {
    const char *function = __func__;
    return guard(function, [&] {
        if (ctx == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "ctx is NULL");
        }
        if (out == nullptr && n != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        const std::string &line = constraintOf(ctx, airgroup_id, air_id, index).line;
        if (n != line.size()) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "n = %" PRIu64 ", and the line of constraint %" PRIu64 " has %zu bytes", n, index,
                        line.size());
        }
        if (n != 0) {
            std::copy(line.begin(), line.end(), out);
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

namespace {

// Decodes the challenges pilfflonk_check and pilfflonk_check_column take: PILFFLONK_OK, or the
// failure, said.
int checkChallenges(const char *function, const uint8_t *challenges, uint64_t n_challenges,
                    std::vector<PilFflonk::FrElement> &values) {
    if (challenges == nullptr && n_challenges != 0) {
        return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "challenges is NULL");
    }
    uint64_t refused = 0;
    if (!decodeScalars(challenges, n_challenges, values, refused)) {
        return fail(PILFFLONK_ERR_NON_CANONICAL, function, "challenges[%" PRIu64 "] is not below r", refused);
    }
    return static_cast<int>(PILFFLONK_OK);
}

} // namespace

int pilfflonk_check(void *instance, const uint8_t *challenges, uint64_t n_challenges, uint64_t max_rows,
                    uint64_t n_constraints, uint64_t *out_n_failed, uint64_t *out_rows, uint8_t *out_values) {
    const char *function = __func__;
    return guard(function, [&] {
        if (instance == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instance is NULL");
        }
        std::vector<PilFflonk::FrElement> values;
        const int decoded = checkChallenges(function, challenges, n_challenges, values);
        if (decoded != PILFFLONK_OK) {
            return decoded;
        }
        if (out_n_failed == nullptr && n_constraints != 0) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_n_failed is NULL");
        }
        // Also bounds every offset into out_rows and out_values below.
        const uint64_t fitting = std::numeric_limits<uint64_t>::max() / PilFflonk::FR_BYTES;
        if (n_constraints != 0 && max_rows > fitting / n_constraints) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function,
                        "n_constraints·max_rows = %" PRIu64 "·%" PRIu64 " scalars exceed 2^64 bytes", n_constraints,
                        max_rows);
        }
        const bool entries = n_constraints != 0 && max_rows != 0;
        if (entries && out_rows == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_rows is NULL");
        }
        if (entries && out_values == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out_values is NULL");
        }
        PilFflonk::Instance &inst = *static_cast<PilFflonk::Instance *>(instance);
        const uint64_t expected = inst.air().bin().constraintsInfoDebug.size();
        if (n_constraints != expected) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "n_constraints = %" PRIu64 ", and %s has %" PRIu64,
                        n_constraints, inst.air().name().c_str(), expected);
        }

        const std::vector<PilFflonk::ConstraintCheck> checks = inst.check(max_rows, values);
        for (uint64_t c = 0; c < n_constraints; ++c) {
            out_n_failed[c] = checks[c].nFailed;
            if (!entries) {
                continue;
            }
            const std::vector<PilFflonk::FailedRow> &rows = checks[c].rows;
            uint64_t *rowsOut = out_rows + c * max_rows;
            uint8_t *valuesOut = out_values + c * max_rows * PilFflonk::FR_BYTES;
            for (uint64_t j = 0; j < rows.size(); ++j) {
                rowsOut[j] = rows[j].row;
                PilFflonk::encodeFr(rows[j].value, valuesOut + j * PilFflonk::FR_BYTES);
            }
            std::fill(rowsOut + rows.size(), rowsOut + max_rows, uint64_t(0));
            std::fill(valuesOut + rows.size() * PilFflonk::FR_BYTES, valuesOut + max_rows * PilFflonk::FR_BYTES,
                      uint8_t(0));
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}

int pilfflonk_check_column(void *instance, const uint8_t *challenges, uint64_t n_challenges, uint32_t stage,
                           uint64_t stage_pos, uint8_t *out, uint64_t n) {
    const char *function = __func__;
    return guard(function, [&] {
        if (instance == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "instance is NULL");
        }
        if (out == nullptr) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "out is NULL");
        }
        std::vector<PilFflonk::FrElement> values;
        const int decoded = checkChallenges(function, challenges, n_challenges, values);
        if (decoded != PILFFLONK_OK) {
            return decoded;
        }
        PilFflonk::Instance &inst = *static_cast<PilFflonk::Instance *>(instance);
        const uint64_t N = inst.air().n();
        if (n != N) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "n = %" PRIu64 ", and %s has %" PRIu64 " rows", n,
                        inst.air().name().c_str(), N);
        }
        if (stage == 0 || stage > inst.air().info().nStages || stage_pos >= inst.air().cmIds()[stage].size()) {
            return fail(PILFFLONK_ERR_INVALID_ARGUMENT, function, "%s has no column of stage %" PRIu32
                        " at stagePos %" PRIu64, inst.air().name().c_str(), stage, stage_pos);
        }
        const std::vector<std::vector<PilFflonk::FrElement>> columns = inst.checkColumns(values);
        const PilFflonk::FrElement *column = columns[stage].data() + stage_pos * N;
        for (uint64_t i = 0; i < N; ++i) {
            PilFflonk::encodeFr(column[i], out + i * PilFflonk::FR_BYTES);
        }
        return static_cast<int>(PILFFLONK_OK);
    });
}
