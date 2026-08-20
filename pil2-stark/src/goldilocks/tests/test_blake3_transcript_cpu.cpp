#include "test_helpers.hpp"

#include "../src/blake3_core.hpp"

// ---------------------------------------------------------------------------
// blake3core::Hasher — the incremental hasher backing the Fiat-Shamir
// transcript. Absorbs whole Goldilocks words and produces challenge material
// from BLAKE3's XOF, so the transcript hash is genuinely
// blake3(canonical-LE byte stream).
//
// Golden vectors below are the first 64 bytes of the BLAKE3 XOF over the
// canonical-LE stream of words {i*7+3 : i < n}, read as 8 little-endian u64s
// and reduced by to_canonical. Produced with the reference implementation:
//     b3sum --no-names --raw --length 64 <stream>
// They do NOT come from this codebase, so these tests anchor the construction
// to real BLAKE3 rather than to our own output.
// ---------------------------------------------------------------------------

namespace {

struct GoldenXof
{
    uint32_t n;
    uint64_t out[8];
};

const GoldenXof kGolden[] = {
    {0, {12007152915317330863ULL, 5317022963504857248ULL, 13191819210669804443ULL, 7075753032064146124ULL,
         7778449616772730848ULL, 5778175280161008255ULL, 9713341381474225459ULL, 4189482813856917916ULL}},
    {1, {7175107962703238627ULL, 10266939839236670080ULL, 4991015526828484636ULL, 8987319093039342568ULL,
         1351182096268293405ULL, 7939036329090870593ULL, 164747584250988928ULL, 4740858698442839413ULL}},
    {8, {4730537481712038625ULL, 11017037621828616572ULL, 8170974584552723967ULL, 7517513685320260256ULL,
         6657018348538791ULL, 1565292996946479837ULL, 7314032597147226577ULL, 13026651682203648936ULL}},
    {9, {2424636365142760339ULL, 15165381830123158802ULL, 9487485792073438855ULL, 5920058426812994410ULL,
         16462720151111991777ULL, 7237086037464224556ULL, 14801379881922525855ULL, 18396241790501459263ULL}},
    {128, {17371188378716342344ULL, 16531850910111656179ULL, 1014328584800827036ULL, 3601941703461256790ULL,
           1751875036402092858ULL, 4808593708865557967ULL, 15152517445808520735ULL, 2997300546274182758ULL}},
    {129, {5720223199198177089ULL, 1616328176700693306ULL, 11607354963061503359ULL, 124068739580767596ULL,
           10168400764208780594ULL, 2177400346631034771ULL, 6479388027346566143ULL, 3331708523207561586ULL}},
    {300, {8904515693256777727ULL, 8781243969420736812ULL, 5279824308682382935ULL, 17909270760646641756ULL,
           7942935639058460579ULL, 1402413059402118237ULL, 14886447596766557147ULL, 5094709958704476073ULL}},
};

std::vector<uint64_t> makeStream(uint32_t n)
{
    std::vector<uint64_t> v(n);
    for (uint32_t i = 0; i < n; ++i) v[i] = (uint64_t)i * 7 + 3;
    return v;
}

}  // namespace

// The XOF's first four words ARE the BLAKE3 digest, which hash_le64 already
// computes and which is itself b3sum-verified. Cheap oracle at every length.
TEST(Blake3Transcript, FinalizeMatchesHashLe64)
{
    for (uint32_t n : {0u, 1u, 7u, 8u, 9u, 127u, 128u, 129u, 200u, 256u, 257u, 300u, 641u})
    {
        std::vector<uint64_t> in = makeStream(n);

        blake3core::Hasher h;
        h.init();
        h.absorb(in.data(), n);
        uint64_t xof[8];
        h.finalize_xof(0, xof);

        uint64_t expect[4];
        blake3core::hash_le64(in.data(), n, expect);
        for (int i = 0; i < 4; ++i) ASSERT_EQ(xof[i], expect[i]) << "n=" << n << " word=" << i;
    }
}

// Full 64-byte XOF against the reference implementation.
TEST(Blake3Transcript, FinalizeMatchesReferenceBlake3)
{
    for (const auto &g : kGolden)
    {
        std::vector<uint64_t> in = makeStream(g.n);

        blake3core::Hasher h;
        h.init();
        h.absorb(in.data(), g.n);
        uint64_t xof[8];
        h.finalize_xof(0, xof);

        for (int i = 0; i < 8; ++i) ASSERT_EQ(xof[i], g.out[i]) << "n=" << g.n << " word=" << i;
    }
}

// Absorbing in arbitrary pieces must equal absorbing in one go. A transcript
// calls put() with whatever sizes the protocol happens to use.
TEST(Blake3Transcript, AbsorbIsChunkingInvariant)
{
    std::vector<uint64_t> in = makeStream(300);

    uint64_t one[8];
    {
        blake3core::Hasher h;
        h.init();
        h.absorb(in.data(), 300);
        h.finalize_xof(0, one);
    }

    for (uint32_t split : {1u, 3u, 7u, 8u, 9u, 63u, 128u, 129u})
    {
        blake3core::Hasher h;
        h.init();
        uint32_t off = 0;
        while (off < 300)
        {
            uint32_t take = split;
            if (off + take > 300) take = 300 - off;
            h.absorb(in.data() + off, take);
            off += take;
        }
        uint64_t got[8];
        h.finalize_xof(0, got);
        for (int i = 0; i < 8; ++i) ASSERT_EQ(got[i], one[i]) << "split=" << split << " word=" << i;
    }
}

// finalize_xof roots a copy, so squeezing mid-transcript must not disturb the
// absorb chain. This is the property that makes Fiat-Shamir possible at all.
TEST(Blake3Transcript, FinalizeDoesNotDisturbTheChain)
{
    std::vector<uint64_t> in = makeStream(300);

    blake3core::Hasher h;
    h.init();
    h.absorb(in.data(), 150);
    uint64_t mid[8];
    h.finalize_xof(0, mid);
    h.absorb(in.data() + 150, 150);
    uint64_t got[8];
    h.finalize_xof(0, got);

    blake3core::Hasher clean;
    clean.init();
    clean.absorb(in.data(), 300);
    uint64_t want[8];
    clean.finalize_xof(0, want);

    for (int i = 0; i < 8; ++i) ASSERT_EQ(got[i], want[i]) << "word=" << i;
}

// The root node's counter field carries the output-block index, so ob=1 must
// yield the next 64 bytes of the stream rather than repeating ob=0.
TEST(Blake3Transcript, XofOutputBlockCounterAdvances)
{
    std::vector<uint64_t> in = makeStream(9);
    blake3core::Hasher h;
    h.init();
    h.absorb(in.data(), 9);

    uint64_t b0[8], b1[8];
    h.finalize_xof(0, b0);
    h.finalize_xof(1, b1);

    bool differs = false;
    for (int i = 0; i < 8; ++i)
        if (b0[i] != b1[i]) differs = true;
    ASSERT_TRUE(differs);
}
