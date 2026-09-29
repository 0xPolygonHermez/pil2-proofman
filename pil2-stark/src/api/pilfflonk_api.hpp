#ifndef PILFFLONK_API_HPP
#define PILFFLONK_API_HPP

#include <stdint.h>

// C API of the pilfflonk BN254 backend. The Rust side declares it by hand in
// provers/starks-lib-c/bindings_pilfflonk.rs: keep both in sync.
//
// Conventions:
// - A scalar is 32 bytes, canonical (< r) little-endian.
// - A G1 point is 64 bytes, affine x‖y, each coordinate 32 bytes canonical (< q) little-endian.
//   The point at infinity has no affine coordinates: a function that writes it writes (0, 0), as
//   ffiasm and Ethereum's precompiles (EIP-196) do.
// - Objects are opaque handles, released with their _free function.
// - Functions that create an object return NULL on failure; every other function returns one of
//   the status codes below. No function exits or aborts the process.
// - After a failure, pilfflonk_last_error() describes it and pilfflonk_last_status() returns its
//   status, which is how a function that returns NULL tells why.

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
        PILFFLONK_ERR_IO = 5,               // a file cannot be opened, read or written
        PILFFLONK_ERR_FORMAT = 6,           // a file is not in the format expected: the wrong type, a
                                            // section missing or of the wrong size, a value out of range
    };

    // What pilfflonk_transcript_absorb reads.
    enum pilfflonk_transcript_kind {
        PILFFLONK_TRANSCRIPT_FR = 0, // scalars, 32 bytes each
        PILFFLONK_TRANSCRIPT_G1 = 1, // G1 points, 64 bytes each
    };

    // Why the most recent other pilfflonk_* call on the calling thread failed; empty if it
    // succeeded. Never NULL; the next pilfflonk_* call on the same thread overwrites the text.
    const char *pilfflonk_last_error(void);

    // The status of the most recent other pilfflonk_* call on the calling thread: PILFFLONK_OK if it
    // succeeded. The next pilfflonk_* call on the same thread overwrites it.
    int pilfflonk_last_status(void);

    // PILFFLONK_OK if `scalar` (little-endian) is below the BN254 scalar modulus r,
    // PILFFLONK_ERR_NON_CANONICAL otherwise.
    int pilfflonk_fr_check_canonical(const uint8_t scalar[32]);

    // Writes to `out` the Keccak-256 hash of the `len` bytes at `data`: Keccak with its original
    // padding (0x01), as Ethereum and snarkjs use it, not SHA3-256. It is rapidsnark's
    // keccak_wrapper, the hash Keccak256Transcript uses (spec A.4). The setup hashes the vkey's
    // digest preimage with it (spec A.6). With len = 0, `data` may be NULL.
    // PILFFLONK_ERR_INVALID_ARGUMENT if `out` is NULL, if `data` is NULL and len is not 0, or if
    // len is above 2^63 - 1.
    int pilfflonk_keccak256(const uint8_t *data, uint64_t len, uint8_t out[32]);

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

    // The structured reference string (spec §4.2.5, "SRS"): the powers [τ^i]₁ for i < n_g1, and [1]₂
    // and [τ]₂, of a snarkjs powers-of-tau file. The setup extracts them into pilfflonk.srs.bin
    // (spec A.6; the format is in src/pilfflonk/pilfflonk_srs.hpp) and the prover loads that file.
    // Every point is checked as it is read: coordinates below q, on the curve, and [1]₁ and [1]₂
    // the generators.

    // Reads the first n_g1 powers [τ^i]₁, and [1]₂ and [τ]₂, of the ptau at ptau_path -- only
    // those points, from sections 1 to 3 (section 12 is not needed, spec §4.2.1) -- and writes them
    // to srs_path as pilfflonk.srs.bin, replacing any file there. It writes srs_path + ".tmp" and
    // renames it into place, so srs_path is never left half written.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a path is NULL, if n_g1 is 0 or above 2^32 - 1, or if the
    // ptau has fewer than n_g1 powers [τ^i]₁; PILFFLONK_ERR_IO if a file cannot be opened, read or
    // written; PILFFLONK_ERR_FORMAT if the ptau is not a BN254 one, a section is missing or cut
    // short, or a point is not valid.
    int pilfflonk_srs_from_ptau(const char *ptau_path, uint64_t n_g1, const char *srs_path);

    // Loads the pilfflonk.srs.bin at srs_path, or returns NULL. pilfflonk_last_status() then says
    // why: PILFFLONK_ERR_INVALID_ARGUMENT if srs_path is NULL, PILFFLONK_ERR_IO if the file cannot
    // be opened or read, PILFFLONK_ERR_FORMAT if it is not such a file (a header field or a section
    // size that does not match, or a point that is not valid). Release it with pilfflonk_srs_free.
    // An SRS is immutable: the calls that use it may run concurrently.
    void *pilfflonk_srs_load(const char *srs_path);

    // Releases an SRS. NULL is a no-op.
    void pilfflonk_srs_free(void *srs);

    // Writes to out_g2 the power [τ^i]₂ of the SRS, for i = 0 ([1]₂) or i = 1 ([τ]₂): the points
    // of the verifier's pairing (spec A.5); the vkey's X_2 is [τ]₂ (A.6). The point is affine, x‖y,
    // each coordinate an Fq2 element c0 + c1·u written c0‖c1, and each of the four Fq values 32
    // bytes canonical (< q) little-endian: x.c0‖x.c1‖y.c0‖y.c1, the order of the vkey's
    // X_2 = [[x.c0, x.c1], [y.c0, y.c1]].
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL or i is neither 0 nor 1.
    int pilfflonk_srs_g2(const void *srs, uint64_t i, uint8_t out_g2[128]);

    // Writes to out_g1 the KZG commitment [f(τ)]₁ of a fixed f (spec §4.2.5, "Compromisos fixos"):
    // f(X) = Σ_{j<k} p_j(X^k)·X^j, where p_j is the polynomial of degree < N = 2^n_bits whose
    // evaluations on H are column j. `evals` holds the k columns one after another, each the N
    // scalars p_j(ω^i) for i = 0 … N-1, where ω = 5^((r-1)/N) (spec A.2). The columns are
    // interpolated with no blinding (constants get none, A.3), packed, and committed with the SRS.
    // An f that is zero commits to the point at infinity, written as (0, 0), which
    // pilfflonk_transcript_absorb refuses.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL, if k is 0, if n_bits exceeds 28, or if
    // f's k·N coefficients exceed the SRS's powers [τ^i]₁; PILFFLONK_ERR_NON_CANONICAL if a scalar
    // is not below r (checked before anything is computed).
    // This is spec §5.3's pilfflonk_commit_fixed in memory, one f per call: the form that reads the
    // pilfflonkinfo and the .const file waits on their formats (plan M12, M15).
    int pilfflonk_commit_fixed(const void *srs, uint64_t n_bits, uint64_t k, const uint8_t *evals,
                               uint8_t out_g1[64]);

#ifdef __cplusplus
}
#endif

#endif
