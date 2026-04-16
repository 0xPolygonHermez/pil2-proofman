#include "test_helpers.hpp"

TEST(GOLDILOCKS_TEST, poseidon2_seq)
{
    Goldilocks::Element x4[Poseidon2Goldilocks<4>::SPONGE_WIDTH];
    Goldilocks::Element result[Poseidon2Goldilocks<4>::CAPACITY];
    for (uint64_t i = 0; i < Poseidon2Goldilocks<4>::SPONGE_WIDTH; i++)
    {
        x4[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<4>::hash(result, x4, Poseidon2Mode::Scalar);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0x758085b0af0a16aa);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0x85141acc29c479de);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0x50127371e2b77ae5);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xefee3a8033630029);

    Goldilocks::Element x8[Poseidon2Goldilocks<8>::SPONGE_WIDTH];
    for (uint64_t i = 0; i < Poseidon2Goldilocks<8>::SPONGE_WIDTH; i++)
    {
        x8[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<8>::hash(result, x8, Poseidon2Mode::Scalar);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0xc5fb1cfe0b4697bb);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0x4a4a32ff849af473);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0xd2fd266077f8efba);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xf4ad9b74e833916d);

    Goldilocks::Element x12[Poseidon2Goldilocks<12>::SPONGE_WIDTH];
    for (uint64_t i = 0; i < Poseidon2Goldilocks<12>::SPONGE_WIDTH; i++)
    {
        x12[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<12>::hash(result, x12, Poseidon2Mode::Scalar);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X1EAEF96BDF1C0C1 );
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X1F0D2CC525B2540C);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X6282C1DFE1E0358D);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0XE780D721F698E1E6);

    Goldilocks::Element x16[Poseidon2Goldilocks<16>::SPONGE_WIDTH];
    for (uint64_t i = 0; i < Poseidon2Goldilocks<16>::SPONGE_WIDTH; i++)
    {
        x16[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<16>::hash(result, x16, Poseidon2Mode::Scalar);
    
    ASSERT_EQ(Goldilocks::toU64(result[0]), 0x85c54702470d9756);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0xaa53c7a7d52d9898);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0x285128096efb0dd7);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xf3fde5edd3050ac8);
    

}
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, poseidon2_avx)
{

    Goldilocks::Element x4[Poseidon2Goldilocks<4>::SPONGE_WIDTH];
    Goldilocks::Element result[Poseidon2Goldilocks<4>::CAPACITY];

    for (uint64_t i = 0; i < Poseidon2Goldilocks<4>::SPONGE_WIDTH; i++)
    {
        x4[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<4>::hash(result, x4, Poseidon2Mode::Avx);
    ASSERT_EQ(Goldilocks::toU64(result[0]), 0x758085b0af0a16aa );
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0x85141acc29c479de);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0x50127371e2b77ae5);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xefee3a8033630029);

    Goldilocks::Element x8[Poseidon2Goldilocks<8>::SPONGE_WIDTH];

    for (uint64_t i = 0; i < Poseidon2Goldilocks<8>::SPONGE_WIDTH; i++)
    {
        x8[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<8>::hash(result, x8, Poseidon2Mode::Avx);
    ASSERT_EQ(Goldilocks::toU64(result[0]), 0xc5fb1cfe0b4697bb);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0x4a4a32ff849af473);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0xd2fd266077f8efba);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xf4ad9b74e833916d);

    Goldilocks::Element x12[Poseidon2Goldilocks<12>::SPONGE_WIDTH];

    for (uint64_t i = 0; i < Poseidon2Goldilocks<12>::SPONGE_WIDTH; i++)
    {
        x12[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<12>::hash(result, x12, Poseidon2Mode::Avx);
    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X1EAEF96BDF1C0C1 );
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X1F0D2CC525B2540C);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X6282C1DFE1E0358D);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0XE780D721F698E1E6);

    Goldilocks::Element x16[Poseidon2Goldilocks<16>::SPONGE_WIDTH];
    for (uint64_t i = 0; i < Poseidon2Goldilocks<16>::SPONGE_WIDTH; i++)
    {
        x16[i] = Goldilocks::fromU64(i);
    }

    Poseidon2Goldilocks<16>::hash(result, x16, Poseidon2Mode::Avx);
    ASSERT_EQ(Goldilocks::toU64(result[0]), 0x85c54702470d9756);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0xaa53c7a7d52d9898);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0x285128096efb0dd7);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0xf3fde5edd3050ac8);
    
}
#endif


TEST(GOLDILOCKS_TEST, grinding_cpu)
{
    constexpr uint8_t n_bits = 8;
    uint64_t in[3] = {0x1234567890abcdef, 0xfedcba0987654321, 0x0123456789abcdef};
    uint64_t result_index = UINT64_MAX;

    // Call CPU grinding function
    Poseidon2GoldilocksGrinding::grinding(result_index, in, n_bits);

    // Verify we found a valid nonce
    ASSERT_NE(result_index, UINT64_MAX);

    // Verify the hash at result_index satisfies the grinding requirement
    uint64_t level = (1ULL << (64 - n_bits));
    
    // Compute the hash with the found nonce
    Goldilocks::Element x[4] = {in[0], in[1], in[2], result_index};
    Goldilocks::Element result[4];
    Poseidon2GoldilocksGrinding::hashFullResult(result, &x[0], Poseidon2Mode::Scalar);
    
    // Check that result[0] < level
    ASSERT_LT(Goldilocks::toU64(result[0]), level);
}

// Verify LDE(output, input, NExt, N, ncols):
//   input  = evaluations of polynomial p at plain N-th roots {omega_N^j}
//   output = evaluations of p at the extended coset {shift * omega_NExt^j}
//   (the coset shift is introduced internally by INTT(extend=true))
//
// Reference: Horner evaluation of p at each coset point — fully independent of
// the NTT implementation, uses only field arithmetic.

#ifdef __AVX2__

// ---------------------------------------------------------------------------
// Mode-equivalence tests: every compiled-in mode must produce bit-identical
// output for the same input. Private primitives are covered implicitly —
// each mode routes through a distinct backend.
// ---------------------------------------------------------------------------

TEST(GOLDILOCKS_TEST, mode_hashFullResult_equivalence)
{
    constexpr uint32_t W = Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    Goldilocks::Element in[W];
    for (uint32_t i = 0; i < W; ++i) in[i] = Goldilocks::fromU64(i * 31 + 7);

    Goldilocks::Element out_scalar[W], out_avx[W], out_auto[W];
    Poseidon2Goldilocks<16>::hashFullResult(out_scalar, in, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<16>::hashFullResult(out_avx,    in, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<16>::hashFullResult(out_auto,   in, Poseidon2Mode::Auto);

    for (uint32_t i = 0; i < W; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]));
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]));
    }
}

TEST(GOLDILOCKS_TEST, mode_hash_equivalence)
{
    constexpr uint32_t W = Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    constexpr uint32_t C = Poseidon2Goldilocks<16>::CAPACITY;
    Goldilocks::Element in[W];
    for (uint32_t i = 0; i < W; ++i) in[i] = Goldilocks::fromU64(i * 37 + 1);

    Goldilocks::Element out_scalar[C], out_avx[C], out_auto[C];
    Poseidon2Goldilocks<16>::hash(out_scalar, in, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<16>::hash(out_avx,    in, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<16>::hash(out_auto,   in, Poseidon2Mode::Auto);

    for (uint32_t i = 0; i < C; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]));
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]));
    }
}

TEST(GOLDILOCKS_TEST, mode_linearHash_equivalence)
{
    constexpr uint64_t size = 100;
    std::vector<Goldilocks::Element> in(size);
    for (uint64_t i = 0; i < size; ++i) in[i] = Goldilocks::fromU64(i * 13 + 5);

    Goldilocks::Element out_scalar[HASH_SIZE], out_avx[HASH_SIZE], out_auto[HASH_SIZE];
    Poseidon2Goldilocks<16>::linearHash(out_scalar, in.data(), size, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<16>::linearHash(out_avx,    in.data(), size, Poseidon2Mode::Avx);
    Poseidon2Goldilocks<16>::linearHash(out_auto,   in.data(), size, Poseidon2Mode::Auto);

    for (int i = 0; i < HASH_SIZE; ++i) {
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_avx[i]));
        ASSERT_EQ(Goldilocks::toU64(out_scalar[i]), Goldilocks::toU64(out_auto[i]));
    }
}

// For merkletree, assert that every compiled-in mode produces the same root.
template<uint32_t W>
static void merkletreeModeEquivalence(uint64_t arity, uint64_t nrows, uint64_t ncols)
{
    uint64_t numElems = MerklehashGoldilocks::getTreeNumElements(nrows, arity);
    std::vector<Goldilocks::Element> input(nrows * ncols);
    for (uint64_t i = 0; i < nrows * ncols; ++i)
        input[i] = Goldilocks::fromU64(i * 1000003ULL + 1);

    auto rootOf = [&](Poseidon2Mode m, Goldilocks::Element out[4]) {
        std::vector<Goldilocks::Element> tree(numElems);
        Poseidon2Goldilocks<W>::merkletree(tree.data(), input.data(), ncols, nrows, arity,
                                           /*nThreads=*/0, /*dim=*/1, m);
        MerklehashGoldilocks::root((Goldilocks::Element*)out, tree.data(), numElems);
    };

    Goldilocks::Element r_scalar[4];
    rootOf(Poseidon2Mode::Scalar, r_scalar);

    Goldilocks::Element r_avx[4];        rootOf(Poseidon2Mode::Avx,        r_avx);
    Goldilocks::Element r_avxbatch[4];   rootOf(Poseidon2Mode::AvxBatch,   r_avxbatch);
    Goldilocks::Element r_auto[4];       rootOf(Poseidon2Mode::Auto,       r_auto);

    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avx[i]))
            << "Scalar≠Avx at W=" << W << " arity=" << arity;
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avxbatch[i]))
            << "Scalar≠AvxBatch at W=" << W << " arity=" << arity;
        // Auto on AVX2 host resolves to AvxBatch.
        ASSERT_EQ(Goldilocks::toU64(r_avxbatch[i]), Goldilocks::toU64(r_auto[i]))
            << "Auto≠AvxBatch at W=" << W << " arity=" << arity;
    }

#ifdef __AVX512__
    Goldilocks::Element r_avx512batch[4];
    rootOf(Poseidon2Mode::Avx512Batch, r_avx512batch);
    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(Goldilocks::toU64(r_scalar[i]), Goldilocks::toU64(r_avx512batch[i]))
            << "Scalar≠Avx512Batch at W=" << W << " arity=" << arity;
#endif
}

TEST(GOLDILOCKS_TEST, mode_merkletree_equivalence)
{
    const uint64_t nrows = 256;
    const uint64_t colSizes[] = { 1, 8, 64 };
    for (uint64_t ncols : colSizes) {
        merkletreeModeEquivalence<8> (2, nrows, ncols);
        merkletreeModeEquivalence<12>(3, nrows, ncols);
        merkletreeModeEquivalence<16>(4, nrows, ncols);
    }
}
#endif // __AVX2__

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Build commands AVX:

// g++:
//  g++ tests/tests.cpp src/*.{cpp,hpp} -lgtest -lgmp -lomp -o test -g  -Wall -pthread -fopenmp -mavx2 -L$(find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
//  Intel:
//  icpx tests/tests.cpp src/*.{cpp,hpp} -o test -lgtest -lgmp  -pthread -fopenmp -mavx2

// Build commands AVX512:

