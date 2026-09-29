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
        PILFFLONK_ERR_UNSATISFIED = 7,      // the witness does not satisfy the AIR's constraints: its
                                            // constraint polynomial Q is not of its degree (spec A.1)
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

    // ---------------------------------------------------------------------------------------------
    // The prover (spec §4.4). The orchestrator (proofman_pilfflonk::prover) drives it in the order of
    // the transcript of spec A.4, which it owns and absorbs into:
    //
    //   ctx = pilfflonk_ctx_new(provingKey/)                      step 1: the proving key
    //   inst = pilfflonk_instance_new(ctx, …, stage-1 witness, publics, …, seed)
    //   pilfflonk_commit_stage(inst, s, challenges of s, …)       step 2, for s = 1 … nStages
    //   pilfflonk_commit_q(inst, [std_vc], …)                     step 3
    //   op = pilfflonk_opening_new([inst], xiSeed)                step 4: the evaluations
    //   pilfflonk_opening_evaluations(op, …)
    //   pilfflonk_opening_open(op, transcript, W, W', inv, invZh) step 5: squeezes α_S, absorbs W, squeezes y
    //
    // A ctx outlives its instances, and instances their openings. A ctx is immutable: instances of
    // several proofs may share one, from several threads. An instance or an opening is not safe to use
    // from several threads at once.
    // ---------------------------------------------------------------------------------------------

    // The proving key at proving_key_dir, the provingKey/ setup-pilfflonk writes (spec §4.2.6):
    // pilout.globalInfo.json, <name>/pilfflonk/pilfflonk.srs.bin and, for every AIR,
    // <air>.pilfflonkinfo.json, <air>.bin and <air>.const. The fixed columns are interpolated as it
    // is loaded, and everything else the prover derives (the degrees and the extended domain of A.1,
    // the blinding of each column, A.3) is derived. The vkey is not read: its digest, which the
    // transcript absorbs, is the orchestrator's. Returns NULL on failure: PILFFLONK_ERR_INVALID_ARGUMENT
    // if proving_key_dir is NULL, PILFFLONK_ERR_IO if a file cannot be read, PILFFLONK_ERR_FORMAT if one
    // is not what it should be or they do not agree (an AIR that is not the globalInfo's, a layout
    // needing more powers than the SRS holds, a split Q, which this prover does not support yet).
    void *pilfflonk_ctx_new(const char *proving_key_dir);

    // Releases a ctx, which no instance may still use. NULL is a no-op.
    void pilfflonk_ctx_free(void *ctx);

    // Writes to out the nBitsExt of air air_id of airgroup airgroup_id (spec A.1), as the prover
    // derives it, for the orchestrator to check against its own. PILFFLONK_ERR_INVALID_ARGUMENT if a
    // pointer is NULL or there is no such AIR.
    int pilfflonk_ctx_n_bits_ext(const void *ctx, uint64_t airgroup_id, uint64_t air_id, uint64_t *out);

    // An instance of air air_id of airgroup airgroup_id, or NULL.
    // - stage1 holds its stage-1 witness as the witness directory's instance file does (spec A.6):
    //   the N rows one after another, each the values of the C stage-1 columns (the witness columns of
    //   stage 1, by stageId; not the im pols, which the prover computes), stage1_len = N·C·32 bytes.
    // - air_values: the n_air_values values of the stage-1 entries of the AIR's airValuesMap, in
    //   their order; publics, the globalInfo's nPublics; proof_values, those of the stage-1 entries of
    //   the globalInfo's proofValuesMap. Each array may be NULL if its count is 0.
    // - insecure_blinding_seed: NULL for a real proof, blinded with the OS's randomness. Otherwise 32
    //   bytes that fix the blinding (decision D6: tests and CI only): the same seed gives the same
    //   proof, and whoever knows it can remove the blinding, so the proof is not zero-knowledge.
    // PILFFLONK_ERR_INVALID_ARGUMENT if ctx is NULL, if there is no such AIR, if a count or stage1_len
    // is not the one expected, or an array is NULL with a count other than 0;
    // PILFFLONK_ERR_NON_CANONICAL if a scalar is not below r (for stage1, naming its row and column).
    void *pilfflonk_instance_new(const void *ctx, uint64_t airgroup_id, uint64_t air_id, const uint8_t *stage1,
                                 uint64_t stage1_len, const uint8_t *air_values, uint64_t n_air_values,
                                 const uint8_t *publics, uint64_t n_publics, const uint8_t *proof_values,
                                 uint64_t n_proof_values, const uint8_t *insecure_blinding_seed);

    // Releases an instance, which no opening may still use. NULL is a no-op.
    void pilfflonk_instance_free(void *instance);

    // Commits stage `stage` of the instance (spec §4.4 step 2): its columns (the witness's for stage
    // 1), its intermediate polynomials (with the AIR's bytecode on H), and for each f of the stage in
    // the layout, its columns interpolated, blinded as spec A.3 says (p' = p + (X^N − 1)·b, b of
    // |O_f| + 1 coefficients), packed and committed. `challenges` holds the n_challenges challenges of
    // the stage by stageId (none for stage 1); out_g1 receives the n_out commitments of the stage's f,
    // in the order of the layout, n_out being their number.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL with a count other than 0, if stage is not the
    // next stage to commit (1, 2, … nStages in turn), if a count is not the stage's, or for a stage >= 2,
    // whose columns come from prover hints this prover does not compute yet (plan M30);
    // PILFFLONK_ERR_NON_CANONICAL if a challenge is not below r.
    int pilfflonk_commit_stage(void *instance, uint32_t stage, const uint8_t *challenges, uint64_t n_challenges,
                               uint8_t *out_g1, uint64_t n_out);

    // Commits Q (spec §4.4 step 3), once every stage is: the columns it reads extended to the coset,
    // Q (the AIR's cExpId) there, and its coefficients, committed unblinded (spec A.3). `challenges`
    // holds the challenges of stage nStages + 1, std_vc; out_g1 receives the n_out commitments of Q's
    // f (1: Q not split). PILFFLONK_ERR_UNSATISFIED if the witness does not satisfy the AIR's
    // constraints: Q has a coefficient not zero beyond its bound of spec A.1. Otherwise as
    // pilfflonk_commit_stage, and PILFFLONK_ERR_INVALID_ARGUMENT if a stage is not committed yet or Q
    // is committed already.
    int pilfflonk_commit_q(void *instance, const uint8_t *challenges, uint64_t n_challenges, uint8_t *out_g1,
                           uint64_t n_out);

    // The opening of a proof (spec §4.4 steps 4 and 5), or NULL: every f of the instances, in the
    // global order of spec A.5, evaluated at ξ = xi_seed^powerW (powerW the lcm of every k). The
    // n_instances instances, of one ctx and in canonical order, must all have Q committed, and stay
    // alive and unchanged while the opening is.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL, if there are no instances, if they are not as
    // said, or if xi_seed is 0; PILFFLONK_ERR_NON_CANONICAL if xi_seed is not below r;
    // PILFFLONK_ERR_INTERNAL if ξ is in H, where Z_H vanishes (probability N/r).
    void *pilfflonk_opening_new(const void *const *instances, uint64_t n_instances, const uint8_t xi_seed[32]);

    // Releases an opening. NULL is a no-op.
    void pilfflonk_opening_free(void *opening);

    // The number of evaluations of the proof, the n that pilfflonk_opening_evaluations writes. 0 if
    // opening is NULL (and pilfflonk_last_status() says so).
    uint64_t pilfflonk_opening_n_evaluations(const void *opening);

    // Writes to out the n evaluations of the proof, in the order of spec A.4 step 4 and of the proof
    // (A.6): for each AIR with an instance its fixed columns, then for each instance its other
    // columns, each in the order of the AIR's evMap, whose entry (type, id, prime) is its column at
    // ξ·ω^prime. The orchestrator absorbs them before pilfflonk_opening_open.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL or n is not their number.
    int pilfflonk_opening_evaluations(const void *opening, uint64_t n, uint8_t *out);

    // Writes to out Q(ξ) of instance `instance` of the opening: the value the verifier computes from
    // the evaluations (spec A.1), for tests and diagnostics; it is not part of the proof.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL or there is no such instance.
    int pilfflonk_opening_q(const void *opening, uint64_t instance, uint8_t out[32]);

    // SHPLONK (spec A.4 step 5, A.5): squeezes α_S from the transcript, which must hold everything
    // absorbed before (the evaluations last), writes [W]₁ to out_w and absorbs it, squeezes y, and
    // writes [W']₁ to out_wp; then the proof's inv to out_inv (the inverse of the product of the
    // denominators the verifier inverts in the SHPLONK check, in the order verifierInverse lists them
    // in src/pilfflonk/pilfflonk_shplonk_prover.hpp) and 1/Z_H(ξ) to out_inv_zh.
    // PILFFLONK_ERR_INVALID_ARGUMENT if a pointer is NULL or the transcript is empty or incomplete
    // (before it is touched). After that, on failure the proof must be abandoned: PILFFLONK_ERR_INTERNAL
    // if [W]₁ is a point the transcript cannot absorb or y is a root of some f (probability about
    // 2^-60 and deg/r).
    int pilfflonk_opening_open(const void *opening, void *transcript, uint8_t out_w[64], uint8_t out_wp[64],
                               uint8_t out_inv[32], uint8_t out_inv_zh[32]);

#ifdef __cplusplus
}
#endif

#endif
