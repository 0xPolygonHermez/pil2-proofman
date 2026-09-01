// TranscriptGL, the one piece of the hash plumbing with no coverage until now.
//
// blake3 and sha256 are checked against the RUST transcripts (Blake3Transcript /
// Sha256Transcript), which are themselves checked against an independent Python
// FIPS/BLAKE3 reference -- so this pins C++ and Rust to each other through a third
// party. Prover and verifier disagreeing here is a proof that fails to verify with
// nothing pointing at the transcript, which is exactly how blake3's first attempt went.
//
// Poseidon1/Poseidon2 are CHARACTERIZATION: the values are whatever the code produced
// when they were captured. They exist so a refactor of this class cannot move them.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "transcriptGL.hpp"
#include "hash_family.hpp"
#include "transcript_vectors.hpp"

static const uint64_t GL_P = 0xFFFFFFFF00000001ULL;

static Goldilocks::Element w(uint64_t i)
{
    return Goldilocks::fromU64((i * 0x9E3779B97F4A7C15ULL) % GL_P);
}

// A sequence with the shapes that matter: a squeeze straight after an absorb, a refill
// mid-stream, an absorb that must invalidate cached output, getState, and query indices.
static std::vector<uint64_t> runScript(uint64_t arity)
{
    TranscriptGL t(arity, false);
    std::vector<uint64_t> out;

    auto put = [&](uint64_t a, uint64_t b) {
        std::vector<Goldilocks::Element> v;
        for (uint64_t i = a; i < b; ++i) v.push_back(w(i));
        t.put(v.data(), v.size());
    };
    // getField returns the RAW .fe. A sponge permutation's output is not guaranteed canonical, so
    // reducing it would change poseidon's challenges -- and would NOT show up here unless a value
    // happens to land in [p, 2^64), which this script's values do not. Hence the check below.
    auto field = [&]() {
        uint64_t c[3];
        t.getField(c);
        for (int i = 0; i < 3; ++i) out.push_back(c[i]);
    };

    put(0, 4);
    for (int i = 0; i < 3; ++i) field();
    put(4, 20);
    for (int i = 0; i < 2; ++i) field();
    put(20, 21);
    field();

    Goldilocks::Element st[4];
    t.getState(st);
    for (int i = 0; i < 4; ++i) out.push_back(Goldilocks::toU64(st[i]));

    uint64_t perms[8];
    t.getPermutations(perms, 8, 12);
    for (int i = 0; i < 8; ++i) out.push_back(perms[i]);

    return out;
}

// `define_hash_family` refuses to change the family at runtime, which is the right guard for
// production and the wrong one for a test binary covering four families. Writing the global
// directly is the only way to exercise them all in one process.
static void forceFamily(HashFamily f) { g_hash_family = f; }

static void expectSequence(HashFamily f, uint64_t arity, const uint64_t *want, size_t n, const char *name)
{
    forceFamily(f);
    const std::vector<uint64_t> got = runScript(arity);
    ASSERT_EQ(got.size(), n) << name;
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(got[i], want[i]) << name << ": word " << i << " differs";
}

TEST(TranscriptGL, Sha256MatchesTheRustTranscript)
{
    expectSequence(HashFamily::Sha256, 2, TRX_EXPECT_SHA256, 30, "sha256");
}

TEST(TranscriptGL, Blake3MatchesTheRustTranscript)
{
    expectSequence(HashFamily::Blake3, 2, TRX_EXPECT_BLAKE3, 30, "blake3");
}

TEST(TranscriptGL, Poseidon1SequenceIsUnchanged)
{
    expectSequence(HashFamily::Poseidon1, 4, TRX_EXPECT_POSEIDON1, 30, "Poseidon1");
}

TEST(TranscriptGL, Poseidon2SequenceIsUnchanged)
{
    expectSequence(HashFamily::Poseidon2, 4, TRX_EXPECT_POSEIDON2, 30, "Poseidon2");
}

/// blake3 and sha256 pack canonical words, so every challenge they emit must be < p. Poseidon's
/// need not be, which is exactly why getField must not reduce -- see the note in runScript.
TEST(TranscriptGL, TheStreamingFamiliesEmitCanonicalWords)
{
    const uint64_t p = GL_P;
    for (HashFamily f : {HashFamily::Blake3, HashFamily::Sha256})
    {
        forceFamily(f);
        for (uint64_t v : runScript(2)) EXPECT_LT(v, p);
    }
}

/// The families must not agree: identical challenges would mean the family never reached the
/// transcript, which is the failure mode the `if/else` dispatch chains invite.
TEST(TranscriptGL, EveryFamilyProducesADistinctSequence)
{
    forceFamily(HashFamily::Sha256);
    const auto sha = runScript(2);
    forceFamily(HashFamily::Blake3);
    const auto b3 = runScript(2);
    forceFamily(HashFamily::Poseidon2);
    const auto p2 = runScript(4);
    EXPECT_NE(sha, b3);
    EXPECT_NE(sha, p2);
    EXPECT_NE(b3, p2);
}
