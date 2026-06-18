// Poseidon v1 GPU tests: golden vectors (shared with test_poseidon_cpu.cpp)
// plus a CPU-vs-GPU fuzz.
//
// Each test calls PoseidonGoldilocksGPU<12>::initConstants at the top; the
// __constant__ upload is a one-shot guarded by a static flag in the .cu, so
// repeated calls are cheap.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/poseidon_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.cuh"   // Layout enum lives here
#include "../src/goldilocks_tooling.hpp"
#include "../src/goldilocks_tooling.cuh"

using GL = Goldilocks::Element;
using PosCPU = PoseidonGoldilocks<12>;
using PosGPU = PoseidonGoldilocksGPU<12>;

static constexpr uint32_t W        = PosGPU::SPONGE_WIDTH;   // 12
static constexpr uint32_t CAP      = PosGPU::CAPACITY;        // 4
static constexpr uint64_t NCOLS_HASH = 128;
static constexpr uint64_t NROWS_HASH = 1u << 6;              // 64

// ---------------------------------------------------------------------------
// Helpers (mirror test_poseidon_cpu.cpp)
// ---------------------------------------------------------------------------

static void fillFibonacci(GL *buf, uint64_t n)
{
    buf[0] = Goldilocks::zero();
    buf[1] = Goldilocks::one();
    for (uint64_t i = 2; i < n; ++i) buf[i] = buf[i - 1] + buf[i - 2];
}

static GL rnd64(uint64_t seed, uint64_t idx)
{
    uint64_t x = seed ^ (idx * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return Goldilocks::fromU64(x % GOLDILOCKS_PRIME);
}

static void buildMerkleInput(GL *cols, uint64_t ncols, uint64_t nrows)
{
    for (uint64_t i = 0; i < ncols; ++i) {
        cols[i]         = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + ncols] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < nrows; ++j)
        for (uint64_t i = 0; i < ncols; ++i)
            cols[j * ncols + i] = cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
}

// ---------------------------------------------------------------------------
// Golden vectors (shared with test_poseidon_cpu.cpp).
// ---------------------------------------------------------------------------

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

static constexpr uint64_t MERKLE_BINARY_FIB128x64_ROOT[4] = {
    0x918F7CD0C3E8701F, 0x83A130E00F961B02, 0x6921497B364123F8, 0xBD2B98A57B748BF4,
};

// ---------------------------------------------------------------------------
// 1. permute(Fibonacci) — golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permute_fibonacci_w12_golden)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    GL in[W], out[W];
    fillFibonacci(in, W);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, W * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PosGPU::permute((uint64_t *)d_out, (uint64_t *)d_in);
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaMemcpy(out, d_out, W * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < W; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_FIB_W12_GOLDEN[i])
            << "permute(Fibonacci) mismatch at i=" << i;

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// 2. permute(zero input) — golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permute_zero_w12_golden)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    GL in[W] = {Goldilocks::zero()};
    GL out[W];

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, W * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PosGPU::permute((uint64_t *)d_out, (uint64_t *)d_in);
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaMemcpy(out, d_out, W * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < W; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_ZERO_W12_GOLDEN[i])
            << "permute(zero) mismatch at i=" << i;

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// 3. permuteTrunc(Fibonacci) — first CAPACITY of permute(Fibonacci).
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permuteTrunc_fibonacci_w12_golden)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    GL in[W], out[CAP];
    fillFibonacci(in, W);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, CAP * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, W * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PosGPU::permuteTrunc((uint64_t *)d_out, (uint64_t *)d_in);
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaMemcpy(out, d_out, CAP * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_FIB_W12_GOLDEN[i])
            << "permuteTrunc(Fibonacci) mismatch at i=" << i;

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// 4. linearHash(128 Fibonacci elements) — golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, linear_hash_fib128_w12_golden)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    std::vector<GL> input(NCOLS_HASH);
    fillFibonacci(input.data(), NCOLS_HASH);

    // Single row: nRows=1, nCols=128.
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  NCOLS_HASH * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, CAP        * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, input.data(), NCOLS_HASH * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PosGPU::linearHash((uint64_t *)d_out, (uint64_t *)d_in,
                       NCOLS_HASH, /*num_rows=*/1, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    GL out[CAP];
    CHECKCUDAERR(cudaMemcpy(out, d_out, CAP * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), LINEAR_HASH_FIB128_W12_GOLDEN[i])
            << "linearHash(Fibonacci 128) mismatch at i=" << i;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 5. merkletree(arity=2) on 128x64 Fibonacci — root golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, merkletree_binary_fib128x64_golden)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    constexpr uint32_t arity = 2;
    const uint64_t num_cols = NCOLS_HASH;
    const uint64_t num_rows = NROWS_HASH;
    const uint64_t tree_elems = getTreeNumElements(num_rows, arity);

    std::vector<GL> h_input(num_cols * num_rows);
    buildMerkleInput(h_input.data(), num_cols, num_rows);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_input;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_input, num_cols * num_rows * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_elems * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_input, h_input.data(),
                            num_cols * num_rows * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PosGPU::merkletree(arity, (uint64_t *)d_tree, (uint64_t *)d_input,
                       num_cols, num_rows, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    GL h_root[CAP];
    CHECKCUDAERR(cudaMemcpy(h_root, d_tree + (tree_elems - CAP),
                            CAP * sizeof(GL), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(h_root[i]), MERKLE_BINARY_FIB128x64_ROOT[i])
            << "merkletree binary root mismatch at i=" << i;

    cudaFree(d_input);
    cudaFree(d_tree);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 6. linearHash (Tiles layout) parity vs CPU linear_hash_seq.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, linear_hash_tiled_parity)
{
    constexpr uint64_t nRows = 256;
    constexpr uint64_t nCols = 12;

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    std::vector<GL> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; ++i)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    std::vector<GL> h_cpu(nRows * CAP);
    for (uint64_t i = 0; i < nRows; ++i)
        PosCPU::linearHash(h_cpu.data() + i * CAP,
                           h_trace.data() + i * nCols, nCols,
                           PoseidonMode::Scalar);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_flat, *d_tiled, *d_gpu;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat,  nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tiled, nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_gpu,   nRows * CAP   * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_flat, h_trace.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToTiled(nRows, nCols, d_flat, d_tiled, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    PosGPU::linearHash((uint64_t *)d_gpu, (uint64_t *)d_tiled, nCols, nRows,
                       Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<GL> h_gpu(nRows * CAP);
    CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_gpu, nRows * CAP * sizeof(GL), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < nRows * CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(h_gpu[i]), Goldilocks::toU64(h_cpu[i]))
            << "linearHash(Tiles) mismatch at i=" << i;

    cudaFree(d_flat);
    cudaFree(d_tiled);
    cudaFree(d_gpu);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 7. merkletreeReduce parity vs CPU merkletreeReduce.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, merkletree_reduce_parity)
{
    // Only arity == 2 (binary merkle) is supported; both merkletree and
    // merkletreeReduce assert it. Tests use 2.
    constexpr uint32_t arity = 2;
    constexpr uint64_t nElems = 10;

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    std::vector<GL> h_input(nElems * CAP);
    for (uint64_t i = 0; i < nElems * CAP; ++i)
        h_input[i] = Goldilocks::fromU64(i + 1);

    GL h_cpu_root[CAP];
    PosCPU::merkletreeReduce(h_cpu_root, h_input.data(), nElems, arity);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t *d_input, *d_root;
    CHECKCUDAERR(cudaMalloc((void **)&d_input, nElems * CAP * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_root,  CAP          * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_input, h_input.data(),
                            nElems * CAP * sizeof(uint64_t), cudaMemcpyHostToDevice));

    PosGPU::merkletreeReduce(d_root, d_input, nElems, arity, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    GL h_gpu_root[CAP];
    CHECKCUDAERR(cudaMemcpy(h_gpu_root, d_root, CAP * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAP; ++i)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "merkletreeReduce parity mismatch at i=" << i;

    cudaFree(d_input);
    cudaFree(d_root);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 8. Fuzz: GPU permute vs CPU Scalar permute on 256 random W-tuples.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permute_fuzz_vs_cpu_scalar)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PosGPU::initConstants(&gpu_id, 1);

    constexpr uint32_t N_TRIALS = 256;

    std::vector<GL> h_in (N_TRIALS * W);
    std::vector<GL> h_gpu(W);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W * sizeof(gl64_t)));

    for (uint32_t t = 0; t < N_TRIALS; ++t)
    {
        GL inp[W], cpu_out[W];
        for (uint32_t j = 0; j < W; ++j)
            inp[j] = rnd64(0xDEADBEEFCAFEBABEULL ^ (t * 1315423911u), j);

        PosCPU::permute(cpu_out, inp, PoseidonMode::Scalar);

        CHECKCUDAERR(cudaMemcpy(d_in, inp, W * sizeof(gl64_t), cudaMemcpyHostToDevice));
        PosGPU::permute((uint64_t *)d_out, (uint64_t *)d_in);
        CHECKCUDAERR(cudaDeviceSynchronize());
        CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_out, W * sizeof(gl64_t), cudaMemcpyDeviceToHost));

        for (uint32_t j = 0; j < W; ++j)
            ASSERT_EQ(Goldilocks::toU64(h_gpu[j]), Goldilocks::toU64(cpu_out[j]))
                << "fuzz permute mismatch at trial=" << t << " j=" << j;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// W=8 Poseidon-1 golden — plonky3 cross-check.
//    Input [0..7], expected output from plonky3:
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permute_W8_plonky3_golden)
{
    static constexpr uint32_t W8 = 8;
    static constexpr uint64_t PERMUTE_W8_POSEIDON1_PLONKY3_GOLDEN[W8] = {
        2431226948502761687ULL, 9427563026145807618ULL, 6827549936272051660ULL, 16907684411084503785ULL,
        10131745626715172913ULL, 17448305483431576765ULL, 9066501914269485014ULL, 12095238468458521303ULL,
    };

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W8>::initConstants(&gpu_id, 1);

    GL in[W8], out[W8];
    for (uint32_t i = 0; i < W8; ++i) in[i] = Goldilocks::fromU64((uint64_t)i);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W8 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W8 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, W8 * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PoseidonGoldilocksGPU<W8>::permute((uint64_t *)d_out, (uint64_t *)d_in);
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaMemcpy(out, d_out, W8 * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < W8; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_W8_POSEIDON1_PLONKY3_GOLDEN[i])
            << "lane " << i;

    cudaFree(d_in);
    cudaFree(d_out);
}

// W=8 vs CPU parity fuzz.
TEST(PoseidonV1Gpu, permute_W8_parity_vs_cpu_scalar)
{
    static constexpr uint32_t W8 = 8;

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W8>::initConstants(&gpu_id, 1);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W8 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W8 * sizeof(gl64_t)));
    std::vector<GL> h_gpu(W8);

    const uint32_t N_TRIALS = 32;
    for (uint32_t t = 0; t < N_TRIALS; ++t)
    {
        GL inp[W8], cpu_out[W8];
        for (uint32_t j = 0; j < W8; ++j)
            inp[j] = rnd64(0xB6E91709FC5A6731ULL ^ (t * 1597334677u), j);

        PoseidonGoldilocks<W8>::permute(cpu_out, inp, PoseidonMode::Scalar);

        CHECKCUDAERR(cudaMemcpy(d_in, inp, W8 * sizeof(gl64_t), cudaMemcpyHostToDevice));
        PoseidonGoldilocksGPU<W8>::permute((uint64_t *)d_out, (uint64_t *)d_in);
        CHECKCUDAERR(cudaDeviceSynchronize());
        CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_out, W8 * sizeof(gl64_t), cudaMemcpyDeviceToHost));

        for (uint32_t j = 0; j < W8; ++j)
            ASSERT_EQ(Goldilocks::toU64(h_gpu[j]), Goldilocks::toU64(cpu_out[j]))
                << "W=8 fuzz parity mismatch at trial=" << t << " j=" << j;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// W=16 Poseidon-1 self-consistency on GPU — internally-derived golden.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, permute_W16_self_consistency)
{
    static constexpr uint32_t W16 = 16;
    static constexpr uint64_t PERMUTE_W16_POSEIDON1_GOLDEN[W16] = {
        0x81c2ff551d3dd1a3ULL, 0xa6f3ddabab7998e2ULL, 0x4372186243233825ULL, 0xd2bd8442c6cc6df7ULL,
        0x051a796f67578f23ULL, 0x3b597e26481062caULL, 0x19c3c48645baaabbULL, 0x7e142fc8bf48c2ceULL,
        0x599ca659bfbf033fULL, 0x84e132ca4afd703dULL, 0xb758d5776f5185c3ULL, 0xaf58bfc9cb74204eULL,
        0x7015309157ec7e9cULL, 0xe57e7f42acfff2e0ULL, 0x57043250e11a11bbULL, 0x656c21727540ab90ULL,
    };

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W16>::initConstants(&gpu_id, 1);

    GL in[W16], out[W16];
    for (uint32_t i = 0; i < W16; ++i) in[i] = Goldilocks::fromU64((uint64_t)i);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W16 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W16 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, W16 * sizeof(gl64_t), cudaMemcpyHostToDevice));

    PoseidonGoldilocksGPU<W16>::permute((uint64_t *)d_out, (uint64_t *)d_in);
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaMemcpy(out, d_out, W16 * sizeof(gl64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < W16; ++i)
        ASSERT_EQ(Goldilocks::toU64(out[i]), PERMUTE_W16_POSEIDON1_GOLDEN[i])
            << "lane " << i;

    cudaFree(d_in);
    cudaFree(d_out);
}

// W=16 GPU/CPU parity fuzz.
TEST(PoseidonV1Gpu, permute_W16_parity_vs_cpu_scalar)
{
    static constexpr uint32_t W16 = 16;

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W16>::initConstants(&gpu_id, 1);

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  W16 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, W16 * sizeof(gl64_t)));
    std::vector<GL> h_gpu(W16);

    const uint32_t N_TRIALS = 32;
    for (uint32_t t = 0; t < N_TRIALS; ++t)
    {
        GL inp[W16], cpu_out[W16];
        for (uint32_t j = 0; j < W16; ++j)
            inp[j] = rnd64(0xC34F6A1DE9B25718ULL ^ (t * 2654435761u), j);

        PoseidonGoldilocks<W16>::permute(cpu_out, inp, PoseidonMode::Scalar);

        CHECKCUDAERR(cudaMemcpy(d_in, inp, W16 * sizeof(gl64_t), cudaMemcpyHostToDevice));
        PoseidonGoldilocksGPU<W16>::permute((uint64_t *)d_out, (uint64_t *)d_in);
        CHECKCUDAERR(cudaDeviceSynchronize());
        CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_out, W16 * sizeof(gl64_t), cudaMemcpyDeviceToHost));

        for (uint32_t j = 0; j < W16; ++j)
            ASSERT_EQ(Goldilocks::toU64(h_gpu[j]), Goldilocks::toU64(cpu_out[j]))
                << "W=16 fuzz parity mismatch at trial=" << t << " j=" << j;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// GPU grinding → CPU re-verify (mirrors the prove+verify split where the
// prover grinds on GPU and the verifier permutes on CPU). The bug shows up
// here if `poseidon1PermuteReg<8>` running inside the grinding kernel
// disagrees with `Poseidon::permute(_, _, Scalar)`.
// ---------------------------------------------------------------------------

TEST(PoseidonV1Gpu, grinding_W8_gpu_cpu_reverify)
{
    using G = PoseidonGoldilocksGPUGrinding;          // PoseidonGoldilocksGPU<8>
    constexpr uint32_t W = G::SPONGE_WIDTH;

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    G::initConstants(&gpu_id, 1);

    Goldilocks::Element in[3];
    for (int i = 0; i < 3; ++i) in[i] = Goldilocks::fromU64((uint64_t)(i * 7 + 1));

    gl64_t *d_in, *d_nonce, *d_nonceBlock;
    CHECKCUDAERR(cudaMalloc((void **)&d_in, 3 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonce, sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock, NONCES_LAUNCH_GRID_SIZE * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_in, in, 3 * sizeof(gl64_t), cudaMemcpyHostToDevice));

    uint32_t n_bits = 8;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    G::grinding((uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);
    cudaStreamSynchronize(stream);

    uint64_t nonce_idx = 0;
    CHECKCUDAERR(cudaMemcpy(&nonce_idx, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    ASSERT_NE(nonce_idx, UINT64_MAX) << "GPU grinding did not find a nonce";

    // CPU re-verify: rebuild the same input shape (challenge | nonce | 0..0)
    // and permute on CPU using the path the C++ verifier uses.
    Goldilocks::Element x[W] = {};
    x[0] = in[0]; x[1] = in[1]; x[2] = in[2];
    x[3] = Goldilocks::fromU64(nonce_idx);
    Goldilocks::Element result[W];
    PoseidonGoldilocks<W>::permute(result, x, PoseidonMode::Scalar);

    uint64_t threshold = 1ULL << (64 - n_bits);
    EXPECT_LT(Goldilocks::toU64(result[0]), threshold)
        << "GPU-found nonce " << nonce_idx
        << " does NOT satisfy CPU permute check — algorithm divergence between "
           "poseidon1PermuteReg<8> (GPU) and permute_seq<8> (CPU).";

    cudaFree(d_in);
    cudaFree(d_nonce);
    cudaFree(d_nonceBlock);
    cudaStreamDestroy(stream);
}
