#include <gtest/gtest.h>
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/goldilocks_tooling.cuh"
#include "../src/ntt_goldilocks.cuh"

TEST(GOLDILOCKS_TEST, poseidon2)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    Goldilocks::Element in[16], out[16];
    for (int i = 0; i < 16; i++)
    {
        in[i] = Goldilocks::fromU64(i);
    }

    gl64_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, 16 * sizeof(gl64_t));
    cudaMemcpy(d_in, in, 16 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_out, 16 * sizeof(gl64_t));
   
    Poseidon2GoldilocksGPU<4>::permuteTrunc((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 4 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0x758085b0af0a16aa));   
    ASSERT_EQ(out[1].fe, uint64_t(0x85141acc29c479de));
    ASSERT_EQ(out[2].fe, uint64_t(0x50127371e2b77ae5));
    ASSERT_EQ(out[3].fe, uint64_t(0xefee3a8033630029));

    Poseidon2GoldilocksGPU<8>::permute((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 8 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0xc5fb1cfe0b4697bb));   
    ASSERT_EQ(out[1].fe, uint64_t(0x4a4a32ff849af473));
    ASSERT_EQ(out[2].fe, uint64_t(0xd2fd266077f8efba));
    ASSERT_EQ(out[3].fe, uint64_t(0xf4ad9b74e833916d));
    ASSERT_EQ(out[4].fe, uint64_t(0xe6648eb0acc11463));
    ASSERT_EQ(out[5].fe, uint64_t(0x8d5529a930d75194));
    ASSERT_EQ(out[6].fe, uint64_t(0xe8c993aa10da6c90));
    ASSERT_EQ(out[7].fe, uint64_t(0xa73104a95b68031c));

    Poseidon2GoldilocksGPU<12>::permute((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 12 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0x01eaef96bdf1c0c1));   
    ASSERT_EQ(out[1].fe, uint64_t(0x1f0d2cc525b2540c));
    ASSERT_EQ(out[2].fe, uint64_t(0x6282c1dfe1e0358d));
    ASSERT_EQ(out[3].fe, uint64_t(0xe780d721f698e1e6));
    ASSERT_EQ(out[4].fe, uint64_t(0x280c0b6f753d833b));
    ASSERT_EQ(out[5].fe, uint64_t(0x1b942dd5023156ab));
    ASSERT_EQ(out[6].fe, uint64_t(0x43f0df3fcccb8398));
    ASSERT_EQ(out[7].fe, uint64_t(0xe8e8190585489025));
    ASSERT_EQ(out[8].fe, uint64_t(0x56bdbf72f77ada22));
    ASSERT_EQ(out[9].fe, uint64_t(0x7911c32bf9dcd705));
    ASSERT_EQ(out[10].fe, uint64_t(0xec467926508fbe67));
    ASSERT_EQ(out[11].fe, uint64_t(0x6a50450ddf85a6ed));

    Poseidon2GoldilocksGPU<16>::permute((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 16 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe,uint64_t(0x85c54702470d9756));
    ASSERT_EQ(out[1].fe,uint64_t(0xaa53c7a7d52d9898));
    ASSERT_EQ(out[2].fe,uint64_t(0x285128096efb0dd7));
    ASSERT_EQ(out[3].fe,uint64_t(0xf3fde5edd3050ac8));
    ASSERT_EQ(out[4].fe,uint64_t(0xc7b65efd040df908));
    ASSERT_EQ(out[5].fe,uint64_t(0x4be3f6c467f57ae9));
    ASSERT_EQ(out[6].fe,uint64_t(0x274e9a67b41754fb));
    ASSERT_EQ(out[7].fe,uint64_t(0x0f7d39cd5de94dac));
    ASSERT_EQ(out[8].fe,uint64_t(0xd0224b9794d0b78c));
    ASSERT_EQ(out[9].fe,uint64_t(0x372f6139570042e1));
    ASSERT_EQ(out[10].fe,uint64_t(0xce6e8a93dc4ec26c));
    ASSERT_EQ(out[11].fe,uint64_t(0xace65e30a4daf7af));
    ASSERT_EQ(out[12].fe,uint64_t(0x016f2824cc1ba3db));
    ASSERT_EQ(out[13].fe,uint64_t(0x2e8f3af37c434dec));
    ASSERT_EQ(out[14].fe,uint64_t(0xc80831bb6e09da01));
    ASSERT_EQ(out[15].fe,uint64_t(0x3a7d670bf1a86ee8));

    cudaFree(d_in);
    cudaFree(d_out);

}

TEST(GOLDILOCKS_TEST, grinding)
{
    using G = Poseidon2GoldilocksGPUGrinding;
    constexpr uint32_t W = G::SPONGE_WIDTH;
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    G::initConstants(&gpu_id, 1);

    // STARK grinding contract: 3 FIELD_EXTENSION challenge elements + nonce,
    // zero-padded to SPONGE_WIDTH. The kernel only reads the first 3 elements
    // of `in`, but we allocate the full sponge for clarity.
    Goldilocks::Element in[3];
    for (int i = 0; i < 3; i++)
        in[i] = Goldilocks::fromU64(i * 7);

    gl64_t *d_in, *d_out, *d_nonceBlock;
    cudaMalloc((void **)&d_in, 3 * sizeof(gl64_t));
    cudaMemcpy(d_in, in, 3 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_out, sizeof(gl64_t));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock, NONCES_LAUNCH_GRID_SIZE * sizeof(gl64_t)));

    uint32_t n_bits = 8;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    G::grinding((uint64_t *)d_out, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);

    uint64_t result_index;
    cudaMemcpy(&result_index, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    ASSERT_NE(result_index, UINT64_MAX);

    // STARK grinding contract: test_in[0..2] = challenge, test_in[3] = nonce,
    // test_in[4..W-1] = 0.
    Goldilocks::Element test_in[W] = {};
    test_in[0] = in[0];
    test_in[1] = in[1];
    test_in[2] = in[2];
    test_in[3] = Goldilocks::fromU64(result_index);

    gl64_t *d_test_in, *d_hash_out;
    cudaMalloc((void **)&d_test_in, W * sizeof(gl64_t));
    cudaMemcpy(d_test_in, test_in, W * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_hash_out, W * sizeof(gl64_t));

    G::permute((uint64_t *)d_hash_out, (uint64_t *)d_test_in);
    cudaStreamSynchronize(stream);

    Goldilocks::Element hash_result[W];
    cudaMemcpy(hash_result, d_hash_out, W * sizeof(gl64_t), cudaMemcpyDeviceToHost);

    uint64_t level = 1ULL << (64 - n_bits);
    ASSERT_LT(hash_result[0].fe, level) << "Hash does not satisfy grinding requirement";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_test_in);
    cudaFree(d_hash_out);
    cudaFree(d_nonceBlock);
    cudaStreamDestroy(stream);
}

TEST(GOLDILOCKS_TEST, poseidon2_gpu_linear_hash)
{
    constexpr uint64_t nRows = 256;
    constexpr uint64_t nCols = 12;
    constexpr uint32_t CAPACITY = Poseidon2GoldilocksGPU<12>::CAPACITY;

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference: one linear_hash_seq call per row
    std::vector<Goldilocks::Element> h_cpu_out(nRows * CAPACITY);
    for (uint64_t i = 0; i < nRows; i++)
        Poseidon2Goldilocks<12>::linearHash(
            h_cpu_out.data() + i * CAPACITY,
            h_trace.data() + i * nCols,
            nCols,
            Poseidon2Mode::Scalar);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_tiled, *d_gpu_out;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,    nRows * nCols    * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tiled,   nRows * nCols    * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_gpu_out, nRows * CAPACITY * sizeof(gl64_t)));

    CHECKCUDAERR(cudaMemcpy(d_flat, h_trace.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToTiled(nRows, nCols, d_flat, d_tiled, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    Poseidon2GoldilocksGPU<12>::linearHash(
        (uint64_t*)d_gpu_out, (uint64_t*)d_tiled, nCols, nRows, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu_out(nRows * CAPACITY);
    CHECKCUDAERR(cudaMemcpy(h_gpu_out.data(), d_gpu_out, nRows * CAPACITY * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < nRows * CAPACITY; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_out[i]), Goldilocks::toU64(h_cpu_out[i]))
            << "linearHash mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_tiled));
    CHECKCUDAERR(cudaFree(d_gpu_out));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// merkletree(Layout::Tiles): GPU root (block-tiled input) must match CPU merkletree_seq root.
// prepare_blocks_trace converts flat row-major → block-tiled before calling the GPU kernel.
TEST(GOLDILOCKS_TEST, poseidon2_gpu_merkletree_coalescedblocks)
{
    constexpr uint64_t nRows = 256;
    constexpr uint64_t nCols = 12;
    constexpr uint32_t arity = 3;
    uint64_t tree_size = getTreeNumElements(nRows, arity);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_trace.data(), nCols, nRows, arity, Poseidon2Mode::Scalar);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_tiled;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,  nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tiled, nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    CHECKCUDAERR(cudaMemcpy(d_flat, h_trace.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToTiled(nRows, nCols, d_flat, d_tiled, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    Poseidon2GoldilocksGPU<12>::merkletree(
        arity, (uint64_t*)d_tree, (uint64_t*)d_tiled, nCols, nRows, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    const Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "merkletree(Layout::Tiles) root mismatch at element " << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_tiled));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// merkletree: GPU root (flat row-major input) must match CPU merkletree_seq root.
TEST(GOLDILOCKS_TEST, poseidon2_gpu_merkletree_coalesced)
{
    constexpr uint64_t nRows = 256;
    constexpr uint64_t nCols = 12;
    constexpr uint32_t arity = 3;
    uint64_t tree_size = getTreeNumElements(nRows, arity);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_trace.data(), nCols, nRows, arity, Poseidon2Mode::Scalar);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void**)&d_trace, nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    CHECKCUDAERR(cudaMemcpy(d_trace, h_trace.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));

    Poseidon2GoldilocksGPU<12>::merkletree(
        arity, (uint64_t*)d_tree, (uint64_t*)d_trace, nCols, nRows, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    const Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "merkletree root mismatch at element " << i;

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// linearHash (flat): GPU output must match CPU linear_hash_seq per row.
TEST(GOLDILOCKS_TEST, poseidon2_gpu_linear_hash_flat)
{
    constexpr uint64_t nRows = 256;
    constexpr uint64_t nCols = 24;
    constexpr uint32_t CAPACITY = Poseidon2GoldilocksGPU<12>::CAPACITY;

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference: one linear_hash_seq call per row
    std::vector<Goldilocks::Element> h_cpu_out(nRows * CAPACITY);
    for (uint64_t i = 0; i < nRows; i++)
        Poseidon2Goldilocks<12>::linearHash(
            h_cpu_out.data() + i * CAPACITY,
            h_trace.data() + i * nCols,
            nCols,
            Poseidon2Mode::Scalar);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_flat, *d_gpu_out;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,    nRows * nCols    * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_gpu_out, nRows * CAPACITY * sizeof(gl64_t)));

    CHECKCUDAERR(cudaMemcpy(d_flat, h_trace.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));

    Poseidon2GoldilocksGPU<12>::linearHash(
        (uint64_t*)d_gpu_out, (uint64_t*)d_flat, nCols, nRows, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu_out(nRows * CAPACITY);
    CHECKCUDAERR(cudaMemcpy(h_gpu_out.data(), d_gpu_out, nRows * CAPACITY * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < nRows * CAPACITY; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_out[i]), Goldilocks::toU64(h_cpu_out[i]))
            << "linearHash flat mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_gpu_out));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// merkletreeReduce: GPU root over N pre-hashed digests must match CPU merkletreeReduce.
TEST(GOLDILOCKS_TEST, poseidon2_gpu_merkletree_reduce)
{
    constexpr uint32_t CAPACITY = Poseidon2GoldilocksGPU<12>::CAPACITY;
    constexpr uint64_t arity    = 3;
    constexpr uint64_t nElems   = 10;

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    std::vector<Goldilocks::Element> h_input(nElems * CAPACITY);
    for (uint64_t i = 0; i < nElems * CAPACITY; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    Goldilocks::Element h_cpu_root[CAPACITY];
    Poseidon2Goldilocks<12>::merkletreeReduce(h_cpu_root, h_input.data(), nElems, arity);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t *d_input, *d_root;
    CHECKCUDAERR(cudaMalloc((void**)&d_input, nElems * CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_root,  CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_input, h_input.data(), nElems * CAPACITY * sizeof(uint64_t), cudaMemcpyHostToDevice));

    Poseidon2GoldilocksGPU<12>::merkletreeReduce(d_root, d_input, nElems, arity, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    Goldilocks::Element h_gpu_root[CAPACITY];
    CHECKCUDAERR(cudaMemcpy(h_gpu_root, d_root, CAPACITY * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAPACITY; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "merkletreeReduce root mismatch at element " << i;

    CHECKCUDAERR(cudaFree(d_input));
    CHECKCUDAERR(cudaFree(d_root));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// merkletreeReduce edge case: n=1 must pass the single digest through unchanged.
TEST(GOLDILOCKS_TEST, poseidon2_gpu_merkletree_reduce_single)
{
    constexpr uint32_t CAPACITY = Poseidon2GoldilocksGPU<12>::CAPACITY;
    constexpr uint64_t arity    = 3;

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    Goldilocks::Element h_input[CAPACITY];
    for (uint32_t i = 0; i < CAPACITY; i++)
        h_input[i] = Goldilocks::fromU64(0xABCD0000ULL + i);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t *d_input, *d_root;
    CHECKCUDAERR(cudaMalloc((void**)&d_input, CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_root,  CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_input, h_input, CAPACITY * sizeof(uint64_t), cudaMemcpyHostToDevice));

    Poseidon2GoldilocksGPU<12>::merkletreeReduce(d_root, d_input, 1, arity, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    Goldilocks::Element h_gpu_root[CAPACITY];
    CHECKCUDAERR(cudaMemcpy(h_gpu_root, d_root, CAPACITY * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < CAPACITY; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_input[i]))
            << "merkletreeReduce(n=1) must return input unchanged at i=" << i;

    CHECKCUDAERR(cudaFree(d_input));
    CHECKCUDAERR(cudaFree(d_root));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}
