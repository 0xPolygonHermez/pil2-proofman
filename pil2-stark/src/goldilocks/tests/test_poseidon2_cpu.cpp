#include "test_helpers.hpp"

using namespace GoldilocksTestData;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template<uint32_t W>
static void fillSequential(Goldilocks::Element *buf, uint64_t start, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i)
        buf[i] = Goldilocks::fromU64(start + i);
}

static void assertCapacityEq(const Goldilocks::Element *a, const uint64_t *golden)
{
    for (int i = 0; i < HASH_SIZE; ++i)
        ASSERT_EQ(Goldilocks::toU64(a[i]), golden[i]);
}

template<uint32_t N>
static void assertArrayEq(const Goldilocks::Element *a, const uint64_t *golden)
{
    for (uint32_t i = 0; i < N; ++i)
        ASSERT_EQ(Goldilocks::toU64(a[i]), golden[i]);
}

template<uint32_t N>
static void assertArrayEqElem(const Goldilocks::Element *a, const Goldilocks::Element *b)
{
    for (uint32_t i = 0; i < N; ++i)
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(b[i]));
}

// ---------------------------------------------------------------------------
// 1. permute golden
// ---------------------------------------------------------------------------

TEST(Poseidon2, permute_golden_W4)
{
    Goldilocks::Element in[4], out[4];
    fillSequential<4>(in, 0, 4);
    Poseidon2Goldilocks<4>::permute(out, in, Poseidon2Mode::Scalar);
    assertArrayEq<4>(out, PERMUTE_W4_GOLDEN);
}

TEST(Poseidon2, permute_golden_W8)
{
    Goldilocks::Element in[8], out[8];
    fillSequential<8>(in, 0, 8);
    Poseidon2Goldilocks<8>::permute(out, in, Poseidon2Mode::Scalar);
    assertArrayEq<8>(out, PERMUTE_W8_GOLDEN);
}

TEST(Poseidon2, permute_golden_W12)
{
    Goldilocks::Element in[12], out[12];
    fillSequential<12>(in, 0, 12);
    Poseidon2Goldilocks<12>::permute(out, in, Poseidon2Mode::Scalar);
    assertArrayEq<12>(out, PERMUTE_W12_GOLDEN);
}

TEST(Poseidon2, permute_golden_W16)
{
    Goldilocks::Element in[16], out[16];
    fillSequential<16>(in, 0, 16);
    Poseidon2Goldilocks<16>::permute(out, in, Poseidon2Mode::Scalar);
    assertArrayEq<16>(out, PERMUTE_W16_GOLDEN);
}

// ---------------------------------------------------------------------------
// 2. permuteTrunc golden
// ---------------------------------------------------------------------------

TEST(Poseidon2, permuteTrunc_golden_W4)
{
    Goldilocks::Element in[4], out[4];
    fillSequential<4>(in, 0, 4);
    Poseidon2Goldilocks<4>::permuteTrunc(out, in, Poseidon2Mode::Scalar);
    assertCapacityEq(out, PERMUTE_TRUNC_W4_GOLDEN);
}

TEST(Poseidon2, permuteTrunc_golden_W8)
{
    Goldilocks::Element in[8], out[4];
    fillSequential<8>(in, 0, 8);
    Poseidon2Goldilocks<8>::permuteTrunc(out, in, Poseidon2Mode::Scalar);
    assertCapacityEq(out, PERMUTE_TRUNC_W8_GOLDEN);
}

TEST(Poseidon2, permuteTrunc_golden_W12)
{
    Goldilocks::Element in[12], out[4];
    fillSequential<12>(in, 0, 12);
    Poseidon2Goldilocks<12>::permuteTrunc(out, in, Poseidon2Mode::Scalar);
    assertCapacityEq(out, PERMUTE_TRUNC_W12_GOLDEN);
}

TEST(Poseidon2, permuteTrunc_golden_W16)
{
    Goldilocks::Element in[16], out[4];
    fillSequential<16>(in, 0, 16);
    Poseidon2Goldilocks<16>::permuteTrunc(out, in, Poseidon2Mode::Scalar);
    assertCapacityEq(out, PERMUTE_TRUNC_W16_GOLDEN);
}

// ---------------------------------------------------------------------------
// 3. permute mode equivalence
// ---------------------------------------------------------------------------

#ifdef __AVX2__
template<uint32_t W>
static void permuteModeEquiv()
{
    Goldilocks::Element in[W];
    for (uint32_t i = 0; i < W; ++i)
        in[i] = Goldilocks::fromU64(i * 31 + 7);

    Goldilocks::Element out_scalar[W], out_avx[W], out_auto[W];
    Poseidon2Goldilocks<W>::permute(out_scalar, in, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<W>::permute(out_avx,    in, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<W>::permute(out_auto,   in, Poseidon2Mode::Auto);

    for (uint32_t i = 0; i < W; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]))
            << "Scalar!=Avx at W=" << W << " i=" << i;
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]))
            << "Scalar!=Auto at W=" << W << " i=" << i;
    }
}

TEST(Poseidon2, permute_mode_equiv_W4)  { permuteModeEquiv<4>();  }
TEST(Poseidon2, permute_mode_equiv_W8)  { permuteModeEquiv<8>();  }
TEST(Poseidon2, permute_mode_equiv_W12) { permuteModeEquiv<12>(); }
TEST(Poseidon2, permute_mode_equiv_W16) { permuteModeEquiv<16>(); }
#endif

// ---------------------------------------------------------------------------
// 4. permuteTrunc mode equivalence
// ---------------------------------------------------------------------------

#ifdef __AVX2__
template<uint32_t W>
static void permuteTruncModeEquiv()
{
    constexpr uint32_t C = Poseidon2Goldilocks<W>::CAPACITY;
    Goldilocks::Element in[W];
    for (uint32_t i = 0; i < W; ++i)
        in[i] = Goldilocks::fromU64(i * 37 + 1);

    Goldilocks::Element out_scalar[C], out_avx[C], out_auto[C];
    Poseidon2Goldilocks<W>::permuteTrunc(out_scalar, in, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<W>::permuteTrunc(out_avx,    in, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<W>::permuteTrunc(out_auto,   in, Poseidon2Mode::Auto);

    for (uint32_t i = 0; i < C; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]))
            << "Scalar!=Avx at W=" << W << " i=" << i;
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]))
            << "Scalar!=Auto at W=" << W << " i=" << i;
    }
}

TEST(Poseidon2, permuteTrunc_mode_equiv_W4)  { permuteTruncModeEquiv<4>();  }
TEST(Poseidon2, permuteTrunc_mode_equiv_W8)  { permuteTruncModeEquiv<8>();  }
TEST(Poseidon2, permuteTrunc_mode_equiv_W12) { permuteTruncModeEquiv<12>(); }
TEST(Poseidon2, permuteTrunc_mode_equiv_W16) { permuteTruncModeEquiv<16>(); }
#endif

// ---------------------------------------------------------------------------
// 5. linearHash golden
// ---------------------------------------------------------------------------

template<uint32_t W>
static void linearHashGolden(const uint64_t *golden)
{
    constexpr uint64_t size = 20;
    Goldilocks::Element in[size];
    fillSequential<W>(in, 1, size);

    Goldilocks::Element out[HASH_SIZE];
    Poseidon2Goldilocks<W>::linearHash(out, in, size, Poseidon2Mode::Scalar);
    assertCapacityEq(out, golden);
}

TEST(Poseidon2, linearHash_golden_W8)  { linearHashGolden<8>(LINEAR_HASH_W8_GOLDEN);   }
TEST(Poseidon2, linearHash_golden_W12) { linearHashGolden<12>(LINEAR_HASH_W12_GOLDEN); }
TEST(Poseidon2, linearHash_golden_W16) { linearHashGolden<16>(LINEAR_HASH_W16_GOLDEN); }

// ---------------------------------------------------------------------------
// 6. linearHash mode equivalence
// ---------------------------------------------------------------------------

#ifdef __AVX2__
template<uint32_t W>
static void linearHashModeEquiv()
{
    constexpr uint64_t size = 100;
    std::vector<Goldilocks::Element> in(size);
    for (uint64_t i = 0; i < size; ++i)
        in[i] = Goldilocks::fromU64(i * 13 + 5);

    Goldilocks::Element out_scalar[HASH_SIZE], out_avx[HASH_SIZE], out_auto[HASH_SIZE];
    Poseidon2Goldilocks<W>::linearHash(out_scalar, in.data(), size, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<W>::linearHash(out_avx,    in.data(), size, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<W>::linearHash(out_auto,   in.data(), size, Poseidon2Mode::Auto);

    for (int i = 0; i < HASH_SIZE; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]))
            << "Scalar!=Avx at W=" << W << " i=" << i;
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]))
            << "Scalar!=Auto at W=" << W << " i=" << i;
    }
}

TEST(Poseidon2, linearHash_mode_equiv_W8)  { linearHashModeEquiv<8>();  }
TEST(Poseidon2, linearHash_mode_equiv_W12) { linearHashModeEquiv<12>(); }
TEST(Poseidon2, linearHash_mode_equiv_W16) { linearHashModeEquiv<16>(); }
#endif

// ---------------------------------------------------------------------------
// 7. merkletree mode equivalence
// ---------------------------------------------------------------------------

#ifdef __AVX2__
template<uint32_t W>
static void merkletreeModeEquivalence(uint64_t arity, uint64_t nrows, uint64_t ncols)
{
    uint64_t numElems = getTreeNumElements(nrows, arity);
    std::vector<Goldilocks::Element> input(nrows * ncols);
    for (uint64_t i = 0; i < nrows * ncols; ++i)
        input[i] = Goldilocks::fromU64(i * 1000003ULL + 1);

    auto rootOf = [&](Poseidon2Mode m, Goldilocks::Element out[4]) {
        std::vector<Goldilocks::Element> tree(numElems);
        Poseidon2Goldilocks<W>::merkletree(tree.data(), input.data(), ncols, nrows, arity, m);
        std::memcpy(out, &tree[numElems - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));
    };

    Goldilocks::Element r_scalar[4];
    rootOf(Poseidon2Mode::Scalar, r_scalar);

    Goldilocks::Element r_avx[4];
    rootOf(Poseidon2Mode::Avx, r_avx);

    Goldilocks::Element r_avxbatch[4];
    rootOf(Poseidon2Mode::AvxBatch, r_avxbatch);

    Goldilocks::Element r_auto[4];
    rootOf(Poseidon2Mode::Auto, r_auto);

    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avx[i]))
            << "Scalar!=Avx at W=" << W << " arity=" << arity << " ncols=" << ncols;
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avxbatch[i]))
            << "Scalar!=AvxBatch at W=" << W << " arity=" << arity << " ncols=" << ncols;
        ASSERT_EQ(Goldilocks::toU64(r_avxbatch[i]), Goldilocks::toU64(r_auto[i]))
            << "Auto!=AvxBatch at W=" << W << " arity=" << arity << " ncols=" << ncols;
    }

#ifdef __AVX512__
    Goldilocks::Element r_avx512batch[4];
    rootOf(Poseidon2Mode::Avx512Batch, r_avx512batch);
    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avx512batch[i]))
            << "Scalar!=Avx512Batch at W=" << W << " arity=" << arity << " ncols=" << ncols;
#endif
}

TEST(Poseidon2, merkletree_mode_equiv)
{
    constexpr uint64_t nrows = 256;
    constexpr uint64_t colSizes[] = { 1, 8, 64 };
    for (uint64_t ncols : colSizes) {
        merkletreeModeEquivalence<8>(2, nrows, ncols);
        merkletreeModeEquivalence<12>(3, nrows, ncols);
        merkletreeModeEquivalence<16>(4, nrows, ncols);
    }
}
#endif

// ---------------------------------------------------------------------------
// 8. merkletreeReduce golden
// ---------------------------------------------------------------------------

TEST(Poseidon2, merkletreeReduce_golden_W12_ar3)
{
    Goldilocks::Element input[12];
    fillSequential<12>(input, 1, 12);

    Goldilocks::Element root[4];
    Poseidon2Goldilocks<12>::merkletreeReduce(root, input, 3, 3);
    assertCapacityEq(root, MERKLETREE_REDUCE_W12_AR3_GOLDEN);
}

// ---------------------------------------------------------------------------
// 9. merkletreeReduce edge n=1
// ---------------------------------------------------------------------------

TEST(Poseidon2, merkletreeReduce_single_digest)
{
    Goldilocks::Element input[4];
    for (int i = 0; i < 4; ++i)
        input[i] = Goldilocks::fromU64(0xABCD0000ULL + i);

    Goldilocks::Element root[4];
    Poseidon2Goldilocks<12>::merkletreeReduce(root, input, 1, 3);

    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(Goldilocks::toU64(root[i]), Goldilocks::toU64(input[i]));
}

// ---------------------------------------------------------------------------
// 10. grinding
// ---------------------------------------------------------------------------

TEST(Poseidon2, grinding_cpu)
{
    using G = Poseidon2GoldilocksGrinding;
    constexpr uint8_t n_bits = 8;
    constexpr uint32_t W = G::SPONGE_WIDTH;
    uint64_t in[3] = {0x1234567890abcdef, 0xfedcba0987654321, 0x0123456789abcdef};
    uint64_t result_index = UINT64_MAX;

    G::grinding(result_index, in, n_bits);
    ASSERT_NE(result_index, UINT64_MAX);

    // STARK grinding contract: state[0..2] = challenge, state[3] = nonce,
    // state[4..W-1] = 0.
    Goldilocks::Element x[W] = {};
    x[0] = Goldilocks::fromU64(in[0]);
    x[1] = Goldilocks::fromU64(in[1]);
    x[2] = Goldilocks::fromU64(in[2]);
    x[3] = Goldilocks::fromU64(result_index);
    Goldilocks::Element result[W];
    G::permute(result, x, Poseidon2Mode::Scalar);
    uint64_t level = (1ULL << (64 - n_bits));
    ASSERT_LT(Goldilocks::toU64(result[0]), level);
}

// ---------------------------------------------------------------------------
// 11. Edge: permuteTrunc all-zero input
// ---------------------------------------------------------------------------

template<uint32_t W>
static void permuteTruncAllZero()
{
    constexpr uint32_t C = Poseidon2Goldilocks<W>::CAPACITY;
    Goldilocks::Element in[W];
    memset(in, 0, sizeof(in));

    Goldilocks::Element out[C];
    Poseidon2Goldilocks<W>::permuteTrunc(out, in, Poseidon2Mode::Scalar);

    bool allZero = true;
    for (uint32_t i = 0; i < C; ++i) {
        if (Goldilocks::toU64(out[i]) != 0) {
            allZero = false;
            break;
        }
    }
    ASSERT_FALSE(allZero) << "permuteTrunc(zeros) should not produce all-zero output (W=" << W << ")";
}

TEST(Poseidon2, permuteTrunc_allzero_W4)  { permuteTruncAllZero<4>();  }
TEST(Poseidon2, permuteTrunc_allzero_W8)  { permuteTruncAllZero<8>();  }
TEST(Poseidon2, permuteTrunc_allzero_W12) { permuteTruncAllZero<12>(); }
TEST(Poseidon2, permuteTrunc_allzero_W16) { permuteTruncAllZero<16>(); }

// ---------------------------------------------------------------------------
// 12. Edge: linearHash size=0
// ---------------------------------------------------------------------------

template<uint32_t W>
static void linearHashSizeZero()
{
    Goldilocks::Element out[HASH_SIZE];
    memset(out, 0xFF, sizeof(out));
    Goldilocks::Element dummy;
    Poseidon2Goldilocks<W>::linearHash(out, &dummy, 0, Poseidon2Mode::Scalar);

    for (int i = 0; i < HASH_SIZE; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), 0ULL) << "linearHash(size=0) must return zeros (W=" << W << " i=" << i << ")";
}

TEST(Poseidon2, linearHash_size0_W8)  { linearHashSizeZero<8>();  }
TEST(Poseidon2, linearHash_size0_W12) { linearHashSizeZero<12>(); }
TEST(Poseidon2, linearHash_size0_W16) { linearHashSizeZero<16>(); }

// ---------------------------------------------------------------------------
// 13. Edge: linearHash size=1 Scalar==Avx
// ---------------------------------------------------------------------------

#ifdef __AVX2__
template<uint32_t W>
static void linearHashSize1Equiv()
{
    Goldilocks::Element in[1] = { Goldilocks::fromU64(42) };
    Goldilocks::Element out_scalar[HASH_SIZE], out_avx[HASH_SIZE];
    Poseidon2Goldilocks<W>::linearHash(out_scalar, in, 1, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<W>::linearHash(out_avx,    in, 1, Poseidon2Mode::Avx);

    for (int i = 0; i < HASH_SIZE; ++i)
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]))
            << "linearHash size=1 Scalar!=Avx at W=" << W << " i=" << i;
}

TEST(Poseidon2, linearHash_size1_equiv_W8)  { linearHashSize1Equiv<8>();  }
TEST(Poseidon2, linearHash_size1_equiv_W12) { linearHashSize1Equiv<12>(); }
TEST(Poseidon2, linearHash_size1_equiv_W16) { linearHashSize1Equiv<16>(); }
#endif

// ---------------------------------------------------------------------------
// 14. Edge: merkletree nrows=1 root matches linearHash
// ---------------------------------------------------------------------------

template<uint32_t W>
static void merkletreeNrows1(uint64_t arity, uint64_t ncols)
{
    std::vector<Goldilocks::Element> input(ncols);
    for (uint64_t i = 0; i < ncols; ++i)
        input[i] = Goldilocks::fromU64(i * 7 + 3);

    Goldilocks::Element lh_out[HASH_SIZE];
    Poseidon2Goldilocks<W>::linearHash(lh_out, input.data(), ncols, Poseidon2Mode::Scalar);

    uint64_t numElems = getTreeNumElements(1, arity);
    std::vector<Goldilocks::Element> tree(numElems);
    Poseidon2Goldilocks<W>::merkletree(tree.data(), input.data(), ncols, 1, arity, Poseidon2Mode::Scalar);

    Goldilocks::Element root[HASH_SIZE];
    std::memcpy(root, &tree[numElems - HASH_SIZE], HASH_SIZE * sizeof(Goldilocks::Element));

    for (int i = 0; i < HASH_SIZE; ++i)
        ASSERT_EQ(Goldilocks::toU64(root[i]), Goldilocks::toU64(lh_out[i]))
            << "merkletree(nrows=1) root != linearHash at W=" << W << " ncols=" << ncols << " i=" << i;
}

TEST(Poseidon2, merkletree_nrows1_W8)  { merkletreeNrows1<8>(2, 8);  }
TEST(Poseidon2, merkletree_nrows1_W12) { merkletreeNrows1<12>(3, 12); }
TEST(Poseidon2, merkletree_nrows1_W16) { merkletreeNrows1<16>(4, 16); }

// ---------------------------------------------------------------------------
// 15. Edge: grinding n_bits=1
// ---------------------------------------------------------------------------

TEST(Poseidon2, grinding_nbits1)
{
    using G = Poseidon2GoldilocksGrinding;
    constexpr uint32_t W = G::SPONGE_WIDTH;
    uint64_t in[3] = {1, 2, 3};
    uint64_t nonce = UINT64_MAX;

    G::grinding(nonce, in, 1);
    ASSERT_NE(nonce, UINT64_MAX);

    Goldilocks::Element x[W] = {};
    x[0] = Goldilocks::fromU64(in[0]);
    x[1] = Goldilocks::fromU64(in[1]);
    x[2] = Goldilocks::fromU64(in[2]);
    x[3] = Goldilocks::fromU64(nonce);
    Goldilocks::Element result[W];
    G::permute(result, x, Poseidon2Mode::Scalar);
    uint64_t level = (1ULL << 63);
    ASSERT_LT(Goldilocks::toU64(result[0]), level);
}
