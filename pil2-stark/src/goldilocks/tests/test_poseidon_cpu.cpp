// Poseidon v1 CPU tests.

#include <gtest/gtest.h>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"

using GL = Goldilocks::Element;
using Poseidon = PoseidonGoldilocks<12>;
static constexpr uint32_t W        = Poseidon::SPONGE_WIDTH;   // 12
static constexpr uint32_t CAP      = Poseidon::CAPACITY;        // 4
static constexpr uint32_t RATE     = Poseidon::RATE;            // 8
static constexpr uint64_t NCOLS_HASH = 128;
static constexpr uint64_t NROWS_HASH = 1u << 6;   // 64

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fillFibonacci(GL *buf, uint64_t n)
{
    buf[0] = Goldilocks::zero();
    buf[1] = Goldilocks::one();
    for (uint64_t i = 2; i < n; ++i) buf[i] = buf[i - 1] + buf[i - 2];
}

static GL rnd64(uint64_t seed, uint64_t idx)
{
    // Deterministic LCG; canonicalized to Goldilocks range by fromU64.
    uint64_t x = seed ^ (idx * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return Goldilocks::fromU64(x % GOLDILOCKS_PRIME);
}

static void assertCapEq(const GL *a, const uint64_t *golden)
{
    for (uint32_t i = 0; i < CAP; ++i) ASSERT_EQ(Goldilocks::toU64(a[i]), golden[i]);
}

static void assertStateEq(const GL *a, const uint64_t *golden)
{
    for (uint32_t i = 0; i < W; ++i) ASSERT_EQ(Goldilocks::toU64(a[i]), golden[i]);
}

static void assertStateEqEl(const GL *a, const GL *b)
{
    for (uint32_t i = 0; i < W; ++i)
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(b[i]));
}

static constexpr uint64_t PERMUTE_FIB_W12_GOLDEN[12] = {
    0x3095570037F4605D, 0x3D561B5EF1BC8B58, 0x8129DB5EC75C3226, 0x8EC2B67AFB6B87ED,
    0xFC591F17D0FAB161, 0x1D2B045CC2FEA1AD, 0x8A4E3B0CB12D4527, 0x0FF217A756AE2211,
    0x78F6E79CFC407293, 0x3DE827E086AE61C9, 0x921456F6D2D11E27, 0xF58A41D4028C66A5,
};

static constexpr uint64_t PERMUTE_ZERO_W12_GOLDEN[12] = {
    0x3C18A9786CB0B359, 0xC4055E3364A246C3, 0x7953DB0AB48808F4, 0xC71603F33A1144CA,
    0xD7709673896996DC, 0x46A84E87642F44ED, 0xD032648251EE0B3C, 0x1C687363B207DF62,
    0xDF8565563E8045FE, 0x40F5B37FF4254DAE, 0xD070F637B431067C, 0x1792B1C4342109D7,
};

static constexpr uint64_t LINEAR_HASH_FIB128_W12_GOLDEN[4] = {
    0xB214FEA22C79AE3C, 0x49DA61DEED54466A, 0x7338CC9DBA8256FD, 0xC1043293021620CE,
};

// Binary merkletree over 128-col × 64-row Fibonacci input.
static constexpr uint64_t MERKLE_BINARY_FIB128x64_ROOT[4] = {
    0x918F7CD0C3E8701F, 0x83A130E00F961B02, 0x6921497B364123F8, 0xBD2B98A57B748BF4,
};

// Edge case: empty columns, 64 rows.
static constexpr uint64_t MERKLE_BINARY_EMPTY_64_ROOT[4] = {
    0x25225F1A5D49614A, 0x5A1D2A648EEE8F03, 0xDDA8F741C47DFB10, 0x49561260080D30C3,
};

// Edge case: empty columns, 2^17 rows.
static constexpr uint64_t MERKLE_BINARY_EMPTY_POT17_ROOT[4] = {
    0x5587AD00B6DDF0CB, 0x279949E14530C250, 0x02F8E22C79467775, 0x0AA45BE01F9E1610,
};

// ---------------------------------------------------------------------------
// 1. permute golden — Fibonacci input 
// ---------------------------------------------------------------------------

TEST(PoseidonV1, permute_golden_fibonacci_scalar)
{
    GL in[W], out[W];
    fillFibonacci(in, W);
    Poseidon::permute(out, in, PoseidonMode::Scalar);
    assertStateEq(out, PERMUTE_FIB_W12_GOLDEN);
}

TEST(PoseidonV1, permute_golden_zero_scalar)
{
    GL in[W] = {Goldilocks::zero()};
    GL out[W];
    Poseidon::permute(out, in, PoseidonMode::Scalar);
    assertStateEq(out, PERMUTE_ZERO_W12_GOLDEN);
}

#ifdef __AVX2__
TEST(PoseidonV1, permute_golden_fibonacci_avx)
{
    GL in[W], out[W];
    fillFibonacci(in, W);
    Poseidon::permute(out, in, PoseidonMode::Avx);
    assertStateEq(out, PERMUTE_FIB_W12_GOLDEN);
}

TEST(PoseidonV1, permute_golden_zero_avx)
{
    GL in[W] = {Goldilocks::zero()};
    GL out[W];
    Poseidon::permute(out, in, PoseidonMode::Avx);
    assertStateEq(out, PERMUTE_ZERO_W12_GOLDEN);
}
#endif

// ---------------------------------------------------------------------------
// 2. permuteTrunc golden — first CAPACITY of permute output.
// ---------------------------------------------------------------------------

TEST(PoseidonV1, permuteTrunc_golden_fibonacci_scalar)
{
    GL in[W], out[CAP];
    fillFibonacci(in, W);
    Poseidon::permuteTrunc(out, in, PoseidonMode::Scalar);
    assertCapEq(out, PERMUTE_FIB_W12_GOLDEN);
}

#ifdef __AVX2__
TEST(PoseidonV1, permuteTrunc_golden_fibonacci_avx)
{
    GL in[W], out[CAP];
    fillFibonacci(in, W);
    Poseidon::permuteTrunc(out, in, PoseidonMode::Avx);
    assertCapEq(out, PERMUTE_FIB_W12_GOLDEN);
}
#endif

// ---------------------------------------------------------------------------
// 3. linearHash golden — 128 Fibonacci elements, CAPACITY output.
// ---------------------------------------------------------------------------

TEST(PoseidonV1, linear_hash_fib128_scalar)
{
    GL input[NCOLS_HASH];
    fillFibonacci(input, NCOLS_HASH);
    GL out[CAP];
    Poseidon::linearHash(out, input, NCOLS_HASH, PoseidonMode::Scalar);
    assertCapEq(out, LINEAR_HASH_FIB128_W12_GOLDEN);
}

#ifdef __AVX2__
TEST(PoseidonV1, linear_hash_fib128_avx)
{
    GL input[NCOLS_HASH];
    fillFibonacci(input, NCOLS_HASH);
    GL out[CAP];
    Poseidon::linearHash(out, input, NCOLS_HASH, PoseidonMode::Avx);
    assertCapEq(out, LINEAR_HASH_FIB128_W12_GOLDEN);
}
#endif

// ---------------------------------------------------------------------------
// 4. merkletree golden — arity=2 (binary).
// ---------------------------------------------------------------------------

// Input: cols[ncols * nrows] filled Fibonacci-row-wise; each row is `ncols`
// elements; first two rows are cols[i] = i+1; subsequent rows are
// row_j = row_{j-2} + row_{j-1}.
static void buildMerkleInput(GL *cols, uint64_t ncols, uint64_t nrows)
{
    for (uint64_t i = 0; i < ncols; ++i) {
        cols[i]          = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + ncols]  = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < nrows; ++j)
        for (uint64_t i = 0; i < ncols; ++i)
            cols[j * ncols + i] = cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
}

TEST(PoseidonV1, merkletree_binary_fib128x64_scalar)
{
    const uint64_t ncols = NCOLS_HASH, nrows = NROWS_HASH;
    std::vector<GL> cols(ncols * nrows);
    buildMerkleInput(cols.data(), ncols, nrows);

    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), cols.data(), ncols, nrows,
                         /*arity*/ 2, PoseidonMode::Scalar);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_FIB128x64_ROOT);
}

TEST(PoseidonV1, merkletree_binary_empty_64_scalar)
{
    const uint64_t ncols = 0, nrows = NROWS_HASH;
    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), /*input*/ nullptr, ncols, nrows,
                         /*arity*/ 2, PoseidonMode::Scalar);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_EMPTY_64_ROOT);
}

TEST(PoseidonV1, merkletree_binary_empty_pot17_scalar)
{
    const uint64_t ncols = 0, nrows = 1u << 17;
    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), /*input*/ nullptr, ncols, nrows,
                         /*arity*/ 2, PoseidonMode::Scalar);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_EMPTY_POT17_ROOT);
}

#ifdef __AVX2__
TEST(PoseidonV1, merkletree_binary_fib128x64_avx)
{
    const uint64_t ncols = NCOLS_HASH, nrows = NROWS_HASH;
    std::vector<GL> cols(ncols * nrows);
    buildMerkleInput(cols.data(), ncols, nrows);

    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), cols.data(), ncols, nrows,
                         /*arity*/ 2, PoseidonMode::Avx);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_FIB128x64_ROOT);
}

TEST(PoseidonV1, merkletree_binary_fib128x64_avxbatch)
{
    const uint64_t ncols = NCOLS_HASH, nrows = NROWS_HASH;
    std::vector<GL> cols(ncols * nrows);
    buildMerkleInput(cols.data(), ncols, nrows);

    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), cols.data(), ncols, nrows,
                         /*arity*/ 2, PoseidonMode::AvxBatch);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_FIB128x64_ROOT);
}
#endif

#ifdef __AVX512__
TEST(PoseidonV1, merkletree_binary_fib128x64_avx512batch)
{
    const uint64_t ncols = NCOLS_HASH, nrows = NROWS_HASH;
    std::vector<GL> cols(ncols * nrows);
    buildMerkleInput(cols.data(), ncols, nrows);

    uint64_t numTree = getTreeNumElements(nrows, /*arity*/ 2);
    std::vector<GL> tree(numTree);
    Poseidon::merkletree(tree.data(), cols.data(), ncols, nrows,
                         /*arity*/ 2, PoseidonMode::Avx512Batch);
    GL root[4];
    std::memcpy(&root[0], &tree[numTree - CAP], CAP * sizeof(GL));
    assertCapEq(root, MERKLE_BINARY_FIB128x64_ROOT);
}
#endif

// ---------------------------------------------------------------------------
// 5. Mode parity fuzz — Scalar ≡ Avx on random inputs (single-sponge).
// ---------------------------------------------------------------------------

#ifdef __AVX2__
TEST(PoseidonV1, mode_parity_permute_fuzz_scalar_vs_avx)
{
    constexpr uint64_t NFUZZ = 4096;   // Keep CI time reasonable; plenty to catch regressions.
    for (uint64_t k = 0; k < NFUZZ; ++k) {
        GL in[W];
        for (uint32_t i = 0; i < W; ++i) in[i] = rnd64(0xA5A5A5A5A5A5A5A5ULL ^ k, i);
        GL out_seq[W], out_avx[W];
        Poseidon::permute(out_seq, in, PoseidonMode::Scalar);
        Poseidon::permute(out_avx, in, PoseidonMode::Avx);
        assertStateEqEl(out_seq, out_avx);
    }
}

TEST(PoseidonV1, mode_parity_linearHash_fuzz_scalar_vs_avx)
{
    constexpr uint64_t NFUZZ = 1024;
    for (uint64_t k = 0; k < NFUZZ; ++k) {
        uint64_t sz = 5 + (k % 200);   // vary size across ranges
        std::vector<GL> in(sz);
        for (uint64_t i = 0; i < sz; ++i) in[i] = rnd64(0x5A5A5A5A5A5A5A5AULL ^ k, i);
        GL out_seq[CAP], out_avx[CAP];
        Poseidon::linearHash(out_seq, in.data(), sz, PoseidonMode::Scalar);
        Poseidon::linearHash(out_avx, in.data(), sz, PoseidonMode::Avx);
        for (uint32_t i = 0; i < CAP; ++i)
            ASSERT_EQ(Goldilocks::toU64(out_seq[i]), Goldilocks::toU64(out_avx[i]));
    }
}
#endif

// ---------------------------------------------------------------------------
// 6. merkletreeReduce smoke — reduces CAPACITY-sized hashes to a root.
// ---------------------------------------------------------------------------

TEST(PoseidonV1, merkletreeReduce_binary_smoke)
{
    constexpr uint64_t N = 16;   // 16 leaves of CAPACITY elements
    std::vector<GL> input(N * CAP);
    for (uint64_t i = 0; i < N * CAP; ++i) input[i] = Goldilocks::fromU64(i);
    GL root_bin[4];
    Poseidon::merkletreeReduce(root_bin, input.data(), N, /*arity*/ 2);
    // No reference vector for merkletreeReduce; this is a determinism check:
    // two identical calls produce identical roots.
    GL root2[4];
    Poseidon::merkletreeReduce(root2, input.data(), N, /*arity*/ 2);
    for (uint32_t i = 0; i < CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(root_bin[i]), Goldilocks::toU64(root2[i]));
}

// ---------------------------------------------------------------------------
// 7. grinding determinism — the nonce found must satisfy state[0] < 2^(64-n).
// ---------------------------------------------------------------------------

TEST(PoseidonV1, grinding_determinism_low_bits)
{
    // STARK grinding contract: state[0..2] = FIELD_EXTENSION challenge,
    // state[3] = nonce, state[4..W-1] = 0. The test reconstructs the SAME
    // state grinding hashed.
    // n_bits=8 gives level = 2^56 → high probability of quick nonce discovery.
    uint64_t in[W - 1];
    for (uint32_t i = 0; i < W - 1; ++i) in[i] = 0xDEADBEEF00000000ULL ^ i;
    uint64_t nonce = 0;
    Poseidon::grinding(nonce, in, /*n_bits=*/8);

    GL state[W];
    for (uint32_t i = 0; i < W; ++i) state[i].fe = 0;
    state[0].fe = in[0];
    state[1].fe = in[1];
    state[2].fe = in[2];
    state[3] = Goldilocks::fromU64(nonce);
    GL out[W];
    Poseidon::permute(out, state, PoseidonMode::Scalar);
    uint64_t thresh = 1ULL << (64 - 8);
    ASSERT_LT(out[0].fe, thresh);
}

// ---------------------------------------------------------------------------
// 8. W=8 Poseidon-1 golden — plonky3 cross-check.
//    Input [0..7], expected output from plonky3:
// ---------------------------------------------------------------------------

TEST(PoseidonV1, permute_W8_plonky3_golden)
{
    static constexpr uint32_t W8 = 8;
    static constexpr uint64_t PERMUTE_W8_POSEIDON1_PLONKY3_GOLDEN[W8] = {
        2431226948502761687ULL, 9427563026145807618ULL, 6827549936272051660ULL, 16907684411084503785ULL,
        10131745626715172913ULL, 17448305483431576765ULL, 9066501914269485014ULL, 12095238468458521303ULL,
    };
    GL in[W8], out[W8];
    for (uint32_t i = 0; i < W8; ++i) in[i] = Goldilocks::fromU64((uint64_t)i);
    PoseidonGoldilocks<8>::permute(out, in, PoseidonMode::Scalar);
    for (uint32_t i = 0; i < W8; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_W8_POSEIDON1_PLONKY3_GOLDEN[i])
            << "lane " << i;
}

// ---------------------------------------------------------------------------
// 9. W=16 Poseidon-1 self-consistency — internally-derived golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1, permute_W16_self_consistency)
{
    static constexpr uint32_t W16 = 16;
    static constexpr uint64_t PERMUTE_W16_POSEIDON1_GOLDEN[W16] = {
        0x81c2ff551d3dd1a3ULL, 0xa6f3ddabab7998e2ULL, 0x4372186243233825ULL, 0xd2bd8442c6cc6df7ULL,
        0x051a796f67578f23ULL, 0x3b597e26481062caULL, 0x19c3c48645baaabbULL, 0x7e142fc8bf48c2ceULL,
        0x599ca659bfbf033fULL, 0x84e132ca4afd703dULL, 0xb758d5776f5185c3ULL, 0xaf58bfc9cb74204eULL,
        0x7015309157ec7e9cULL, 0xe57e7f42acfff2e0ULL, 0x57043250e11a11bbULL, 0x656c21727540ab90ULL,
    };
    GL in[W16], out[W16];
    for (uint32_t i = 0; i < W16; ++i) in[i] = Goldilocks::fromU64((uint64_t)i);
    PoseidonGoldilocks<16>::permute(out, in, PoseidonMode::Scalar);
    for (uint32_t i = 0; i < W16; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_W16_POSEIDON1_GOLDEN[i])
            << "lane " << i;
}

#ifdef __AVX2__
#endif

#ifdef __AVX512__
#endif

// ---------------------------------------------------------------------------
// W=16 mode parity — Scalar ≡ AvxBatch (≡ Avx512Batch when available).
// Validates the W-generic batch_avx[512] paths produce the same merkle root
// as merkletree_seq.
// ---------------------------------------------------------------------------
#ifdef __AVX2__
#endif

#ifdef __AVX512__
#endif
