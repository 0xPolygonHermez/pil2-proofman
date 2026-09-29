#ifndef PILFFLONK_API_HPP
#define PILFFLONK_API_HPP

#include <stdint.h>

// C API of the pilfflonk BN254 backend. The Rust side declares it by hand in
// provers/starks-lib-c/bindings_pilfflonk.rs: keep both in sync.
//
// Conventions:
// - A scalar is 32 bytes, canonical (< r) little-endian.
// - A G1 point is 64 bytes, affine x‖y, each coordinate 32 bytes canonical (< q) little-endian.
// - Objects are opaque handles, released with their _free function.
// - Functions that create an object return NULL on failure; every other function returns one of
//   the status codes below. No function exits or aborts the process.
// - After a failure, pilfflonk_last_error() describes it.

#ifdef __cplusplus
extern "C" {
#endif

    enum pilfflonk_status {
        PILFFLONK_OK = 0,
        PILFFLONK_ERR_INVALID_ARGUMENT = 1, // a null pointer, an out-of-range size or kind, or a call
                                            // the object's state does not allow
        PILFFLONK_ERR_NON_CANONICAL = 2,    // a scalar not below r, or a coordinate not below q
        PILFFLONK_ERR_INTERNAL = 3,         // an unexpected failure inside the library
        PILFFLONK_ERR_INVALID_POINT = 4,    // a G1 point off the curve, or one the transcript cannot hash
    };

    // What pilfflonk_transcript_absorb reads.
    enum pilfflonk_transcript_kind {
        PILFFLONK_TRANSCRIPT_FR = 0, // scalars, 32 bytes each
        PILFFLONK_TRANSCRIPT_G1 = 1, // G1 points, 64 bytes each
    };

    // Why the most recent other pilfflonk_* call on the calling thread failed; empty if it
    // succeeded. Never NULL; the next pilfflonk_* call on the same thread overwrites the text.
    const char *pilfflonk_last_error(void);

    // PILFFLONK_OK if `scalar` (little-endian) is below the BN254 scalar modulus r,
    // PILFFLONK_ERR_NON_CANONICAL otherwise.
    int pilfflonk_fr_check_canonical(const uint8_t scalar[32]);

    // The Fiat-Shamir transcript of a proof (spec A.4): rapidsnark's Keccak256Transcript, driven as
    // the existing FFLONK prover drives it. Absorbing only appends elements; a squeeze hashes all
    // the elements absorbed since the previous one. Not safe to use from several threads at once.
    // A call that fails with PILFFLONK_ERR_INTERNAL may have been cut short (out of memory); the
    // transcript then refuses every later absorb and squeeze, with the same status.

    // A new, empty transcript, or NULL on failure. Release it with pilfflonk_transcript_free.
    void *pilfflonk_transcript_new(void);

    // Releases a transcript. NULL is a no-op.
    void pilfflonk_transcript_free(void *transcript);

    // Appends `n` elements of `kind`, read from `data`: n * 32 bytes of scalars, or n * 64 bytes of
    // G1 points. All or nothing: if any element is refused, none is absorbed. With n = 0 it does
    // nothing, and `data` may be NULL.
    //
    // A point must be on the curve. It cannot be the point at infinity, which has no x‖y encoding;
    // (0, 0), ffiasm's affine form of it, is refused with its own message because
    // Keccak256Transcript does not hash it (it clears the start of its buffer instead). Nor can a
    // coordinate be below 2^192: ffiasm writes such a coordinate as a larger number instead of as
    // 32 big-endian bytes, so the challenge would not be the A.4 one. Neither case happens in a
    // proof: every point absorbed is blinded, and a random coordinate is that small with
    // probability about 2^-62. Both are PILFFLONK_ERR_INVALID_POINT.
    //
    // PILFFLONK_ERR_INVALID_ARGUMENT also if the elements absorbed since the last squeeze would
    // exceed the 2^31 - 1 bytes that Keccak256Transcript can hash at once (32 per scalar, 192 per
    // point, as it counts them); it is then checked before `data` is read.
    int pilfflonk_transcript_absorb(void *transcript, const uint8_t *data, uint64_t n, uint32_t kind);

    // Writes to `out` the challenge keccak256(buffer) mod r, as a scalar, where the buffer holds
    // the elements absorbed since the previous squeeze in A.4's big-endian encoding. Then restarts
    // the transcript seeded with that challenge: Keccak256Transcript's reset() + addScalar().
    // PILFFLONK_ERR_INVALID_ARGUMENT on a transcript to which nothing has been absorbed yet, like
    // snarkjs's Keccak256Transcript.
    int pilfflonk_transcript_squeeze(void *transcript, uint8_t out[32]);

#ifdef __cplusplus
}
#endif

#endif
