// Validates sha256_core.hpp against an INDEPENDENT FIPS 180-4 implementation: the
// vectors come from a pure-Python SHA-256 written from the spec, which separately
// asserts itself against hashlib. Inputs include words >= p (and p) so the
// canonicalization is exercised rather than assumed.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "sha256_core.hpp"
#include "sha256_core_vectors.hpp"

using namespace sha256core;
namespace V = sha256_vectors;

static void expectDigest(const uint64_t got[4], const uint64_t want[4], const char *what, uint32_t n)
{
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(got[i], want[i]) << what << " mismatch at word " << i << " (n = " << n << ")";
}

// ===========================================================================
// The three constructions, each against the independent reference.
// ===========================================================================

TEST(Sha256Core, LeafHashMatchesIndependentReference)
{
    for (int c = 0; c < V::leaf_case_count; ++c)
    {
        const V::LeafCase &k = V::leaf_cases[c];
        uint64_t out[4];
        hash_le64(k.in, k.n, out);
        expectDigest(out, k.out, "hash_le64", k.n);
    }
}

TEST(Sha256Core, NodeHashMatchesIndependentReference)
{
    for (int c = 0; c < V::node_case_count; ++c)
    {
        const V::NodeCase &k = V::node_cases[c];
        uint64_t out[4];
        node_hash(k.in, k.n, out);
        expectDigest(out, k.out, "node_hash", k.n);
    }
}

TEST(Sha256Core, GrindHashMatchesIndependentReference)
{
    for (int c = 0; c < V::grind_case_count; ++c)
    {
        uint64_t out[4];
        grind_hash(V::grind_cases[c].in, out);
        expectDigest(out, V::grind_cases[c].out, "grind_hash", 8);
    }
}

TEST(Sha256Core, TranscriptSqueezeMatchesIndependentReference)
{
    for (int c = 0; c < V::trx_case_count; ++c)
    {
        const V::TrxCase &k = V::trx_cases[c];
        Hasher h;
        h.init();
        h.absorb(k.in, k.n);
        uint64_t out[4];
        h.squeeze(k.counter, out);
        expectDigest(out, k.out, "Hasher::squeeze", k.n);
    }
}

// ===========================================================================
// Properties the surrounding code depends on, which vectors alone do not pin.
// ===========================================================================

/// The node hash must be a DIFFERENT function from the leaf hash at the same
/// length. blake3 fails this by design and leans on the fixed tree shape instead.
TEST(Sha256Core, NodeHashIsDomainSeparatedFromLeafHash)
{
    const uint64_t *in = V::node_cases[0].in;   // 8 words = one arity-2 node
    uint64_t leaf[4], node[4];
    hash_le64(in, 8, leaf);
    node_hash(in, 8, node);
    bool identical = true;
    for (int i = 0; i < 4; ++i) identical &= (leaf[i] == node[i]);
    EXPECT_FALSE(identical) << "node_hash must not coincide with hash_le64 at the node width";
}

/// Streaming Hasher vs one-shot hash_le64: two code paths, so this catches a
/// buffering bug the vectors would not.
TEST(Sha256Core, TranscriptDigestEqualsLeafHashOfTheWholeStream)
{
    for (uint32_t n : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 64u, 130u})
    {
        std::vector<uint64_t> in(n ? n : 1);
        for (uint32_t i = 0; i < n; ++i) in[i] = 0x9E3779B97F4A7C15ULL * (i + 1);

        Hasher h;
        h.init();
        h.absorb(in.data(), n);
        uint64_t streamed[4], oneshot[4];
        h.digest(streamed);
        hash_le64(in.data(), n, oneshot);
        expectDigest(streamed, oneshot, "Hasher::digest vs hash_le64", n);
    }
}

/// _add1 feeds one element at a time; the split must not matter.
TEST(Sha256Core, TranscriptIsIndependentOfAbsorbChunking)
{
    const uint32_t n = 37;
    std::vector<uint64_t> in(n);
    for (uint32_t i = 0; i < n; ++i) in[i] = 0xDEADBEEF00000000ULL + i;

    uint64_t bulk[4];
    {
        Hasher h; h.init(); h.absorb(in.data(), n); h.digest(bulk);
    }
    for (uint32_t chunk : {1u, 2u, 3u, 5u, 8u, 9u, 16u})
    {
        Hasher h; h.init();
        for (uint32_t i = 0; i < n; i += chunk)
            h.absorb(in.data() + i, (i + chunk <= n) ? chunk : (n - i));
        uint64_t got[4];
        h.digest(got);
        expectDigest(got, bulk, "chunked absorb", chunk);
    }
}

/// A transcript keeps absorbing after every challenge, so both must be const.
TEST(Sha256Core, SqueezeDoesNotConsumeTheState)
{
    const uint64_t in[5] = {1, 2, 3, 4, 5};
    Hasher h; h.init(); h.absorb(in, 5);

    uint64_t a[4], b[4];
    h.squeeze(0, a);
    h.squeeze(0, b);
    expectDigest(b, a, "repeated squeeze(0)", 5);

    // Different counters must give different material.
    uint64_t c1[4];
    h.squeeze(1, c1);
    bool same = true;
    for (int i = 0; i < 4; ++i) same &= (c1[i] == a[i]);
    EXPECT_FALSE(same) << "squeeze(0) and squeeze(1) must differ";

    // Absorbing after squeezing must still track the full stream.
    const uint64_t more[2] = {6, 7};
    h.absorb(more, 2);
    uint64_t after[4], expect[4];
    h.digest(after);
    const uint64_t all[7] = {1, 2, 3, 4, 5, 6, 7};
    hash_le64(all, 7, expect);
    expectDigest(after, expect, "digest after squeeze-then-absorb", 7);
}

/// The verifier re-hashes reduced values, so +p must be indistinguishable.
TEST(Sha256Core, AllConstructionsHashCanonicalValues)
{
    uint64_t raw[8], plus_p[8];
    for (int i = 0; i < 8; ++i)
    {
        raw[i] = (uint64_t)(i + 1) * 0x1000003ULL;
        plus_p[i] = raw[i] + GL_P;          // same field element, different u64
        ASSERT_LT(raw[i], GL_P);
        ASSERT_GE(plus_p[i], GL_P);
    }

    uint64_t a[4], b[4];
    hash_le64(raw, 8, a);  hash_le64(plus_p, 8, b);
    expectDigest(b, a, "hash_le64 canonicalization", 8);

    node_hash(raw, 8, a);  node_hash(plus_p, 8, b);
    expectDigest(b, a, "node_hash canonicalization", 8);

    grind_hash(raw, a);    grind_hash(plus_p, b);
    expectDigest(b, a, "grind_hash canonicalization", 8);

    Hasher h1, h2;
    h1.init(); h1.absorb(raw, 8);     h1.digest(a);
    h2.init(); h2.absorb(plus_p, 8);  h2.digest(b);
    expectDigest(b, a, "Hasher canonicalization", 8);
}

/// Outputs go straight to Goldilocks::fromU64, so they must be canonical.
TEST(Sha256Core, OutputsAreCanonical)
{
    for (int c = 0; c < V::leaf_case_count; ++c)
    {
        uint64_t out[4];
        hash_le64(V::leaf_cases[c].in, V::leaf_cases[c].n, out);
        for (int i = 0; i < 4; ++i)
            ASSERT_LT(out[i], GL_P) << "leaf digest word " << i << " not canonical";
    }
    uint64_t out[4];
    node_hash(V::node_cases[0].in, 8, out);
    for (int i = 0; i < 4; ++i) ASSERT_LT(out[i], GL_P);
    grind_hash(V::grind_cases[0].in, out);
    for (int i = 0; i < 4; ++i) ASSERT_LT(out[i], GL_P);
}
