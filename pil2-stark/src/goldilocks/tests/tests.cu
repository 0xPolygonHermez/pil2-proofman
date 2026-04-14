#include <gtest/gtest.h>
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/ntt_goldilocks.cuh"
#include "../src/gl64_tooling.cuh"
#include "../src/merklehash_goldilocks.hpp"
#include "../src/goldilocks_cubic_extension.hpp"


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
   
    Poseidon2GoldilocksGPU<4>::hash((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 4 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0x758085b0af0a16aa));   
    ASSERT_EQ(out[1].fe, uint64_t(0x85141acc29c479de));
    ASSERT_EQ(out[2].fe, uint64_t(0x50127371e2b77ae5));
    ASSERT_EQ(out[3].fe, uint64_t(0xefee3a8033630029));

    Poseidon2GoldilocksGPU<8>::hash((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 8 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0xc5fb1cfe0b4697bb));   
    ASSERT_EQ(out[1].fe, uint64_t(0x4a4a32ff849af473));
    ASSERT_EQ(out[2].fe, uint64_t(0xd2fd266077f8efba));
    ASSERT_EQ(out[3].fe, uint64_t(0xf4ad9b74e833916d));
    ASSERT_EQ(out[4].fe, uint64_t(0xe6648eb0acc11463));
    ASSERT_EQ(out[5].fe, uint64_t(0x8d5529a930d75194));
    ASSERT_EQ(out[6].fe, uint64_t(0xe8c993aa10da6c90));
    ASSERT_EQ(out[7].fe, uint64_t(0xa73104a95b68031c));

    Poseidon2GoldilocksGPU<12>::hash((uint64_t *)d_out, (uint64_t *)d_in);
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

    Poseidon2GoldilocksGPU<16>::hash((uint64_t *)d_out, (uint64_t *)d_in);
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
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPUGrinding::initConstants(&gpu_id, 1);

    // Input data for grinding (4 elements for SPONGE_WIDTH=4)
    Goldilocks::Element in[4];
    for (int i = 0; i < 3; i++)
    {
        in[i] = Goldilocks::fromU64(i * 7); 
    }

    gl64_t *d_in, *d_out, *d_nonceBlock;
    cudaMalloc((void **)&d_in, 4 * sizeof(gl64_t));
    cudaMemcpy(d_in, in, 4 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_out, sizeof(gl64_t));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock, NONCES_LAUNCH_GRID_SIZE * sizeof(gl64_t)));

    uint32_t n_bits = 8; // Looking for hash with 8 leading zero bits
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Poseidon2GoldilocksGPUGrinding::grinding((uint64_t *)d_out, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);
    
    uint64_t result_index;
    cudaMemcpy(&result_index, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Verify the result is not UINT64_MAX (meaning a valid nonce was found)
    ASSERT_NE(result_index, UINT64_MAX);
    
    // Verify the hash at this index actually satisfies the grinding requirement
    Goldilocks::Element test_in[4];
    for (int i = 0; i < 3; i++)
    {
        test_in[i] = in[i];
    }
    test_in[3] = Goldilocks::fromU64(result_index);
    
    gl64_t *d_test_in, *d_hash_out;
    cudaMalloc((void **)&d_test_in, 4 * sizeof(gl64_t));
    cudaMemcpy(d_test_in, test_in, 4 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_hash_out, 4 * sizeof(gl64_t));
    
    Poseidon2GoldilocksGPU<4>::hash((uint64_t *)d_hash_out, (uint64_t *)d_test_in);
    cudaStreamSynchronize(stream);
    
    Goldilocks::Element hash_result[4];
    cudaMemcpy(hash_result, d_hash_out, 4 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    
    // Check that the first element of the hash satisfies the grinding requirement
    uint64_t level = 1ULL << (64 - n_bits);
    ASSERT_LT(hash_result[0].fe, level) << "Hash does not satisfy grinding requirement";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_test_in);
    cudaFree(d_hash_out);
    cudaFree(d_nonceBlock);
    cudaStreamDestroy(stream);
}

// Round-trip test: CPU NTT → GPU INTT must recover the original input.
TEST(GOLDILOCKS_TEST, ntt_gpu_intt_roundtrip)
{
    constexpr uint64_t n_bits = 20;
    constexpr uint64_t domain_size = 1ULL << n_bits;
    constexpr uint64_t nCols = 1;

    // Build Fibonacci input
    std::vector<Goldilocks::Element> h_original(domain_size);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < domain_size; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    // CPU NTT: time domain → frequency domain
    std::vector<Goldilocks::Element> h_ntt(h_original);
    NTT_Goldilocks cpu_ntt(domain_size);
    cpu_ntt.NTT(h_ntt.data(), h_ntt.data(), domain_size);

    // Upload frequency-domain data to GPU
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(20, 1, &gpu_id);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_ntt.data(), domain_size * sizeof(gl64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    // GPU INTT: frequency domain → time domain
    gpu_ntt.INTT(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Download result and compare with original
    std::vector<Goldilocks::Element> h_result(domain_size);
    CHECKCUDAERR(cudaMemcpy(h_result.data(), d_data, domain_size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < domain_size; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i])) << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Round-trip test: GPU NTT → GPU INTT must recover the original input.
TEST(GOLDILOCKS_TEST, ntt_gpu_ntt_intt_roundtrip)
{
    constexpr uint64_t n_bits = 20;
    constexpr uint64_t domain_size = 1ULL << n_bits;
    constexpr uint64_t nCols = 1;

    // Build Fibonacci input
    std::vector<Goldilocks::Element> h_original(domain_size);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < domain_size; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(n_bits, 1, &gpu_id);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_original.data(), domain_size * sizeof(gl64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    // GPU NTT then GPU INTT should recover original
    gpu_ntt.NTT(d_data, n_bits, nCols, stream);
    gpu_ntt.INTT(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_result(domain_size);
    CHECKCUDAERR(cudaMemcpy(h_result.data(), d_data, domain_size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < domain_size; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i])) << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Round-trip test: CPU NTT → GPU INTT must recover the original input, nCols=TILE_WIDTH.
TEST(GOLDILOCKS_TEST, ntt_gpu_intt_multicol)
{
    constexpr uint64_t n_bits  = 16;
    constexpr uint64_t nRows   = 1ULL << n_bits;
    constexpr uint64_t nCols   = TILE_WIDTH;
    constexpr uint64_t nElems  = nRows * nCols;

    // Build Fibonacci input in row-major order
    std::vector<Goldilocks::Element> h_original(nElems);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < nElems; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    // CPU NTT over all columns at once (row-major interleaved layout)
    std::vector<Goldilocks::Element> h_ntt(h_original);
    NTT_Goldilocks cpu_ntt(nRows);
    cpu_ntt.NTT(h_ntt.data(), h_ntt.data(), nRows, nCols);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(16, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    // Upload flat NTT data; prepare_blocks_trace converts it to block-tiled layout
    gl64_t *d_flat, *d_tiled;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,  nElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tiled, nElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_flat, h_ntt.data(), nElems * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToTiled(nRows, nCols, d_flat, d_tiled, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // GPU INTT in-place on block-tiled data
    gpu_ntt.INTT(d_tiled, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Download block-tiled result and un-tile back to row-major
    std::vector<Goldilocks::Element> h_tiled(nElems);
    CHECKCUDAERR(cudaMemcpy(h_tiled.data(), d_tiled, nElems * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    std::vector<Goldilocks::Element> h_result(nElems);
    for (uint64_t row = 0; row < nRows; row++) {
        for (uint64_t col = 0; col < nCols; col++) {
            uint64_t blockY      = col / TILE_WIDTH;
            uint64_t blockX      = row / TILE_HEIGHT;
            uint64_t nCols_block = std::min((uint64_t)TILE_WIDTH, nCols - TILE_WIDTH * blockY);
            uint64_t tiled_idx   = blockY * (uint64_t)TILE_WIDTH * nRows
                                 + blockX * nCols_block * TILE_HEIGHT
                                 + (col % TILE_WIDTH) * TILE_HEIGHT
                                 + (row % TILE_HEIGHT);
            h_result[row * nCols + col] = h_tiled[tiled_idx];
        }
    }

    for (uint64_t i = 0; i < nElems; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i]))
            << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_tiled));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Correctness test: GPU LDE must match CPU LDE element-by-element.
TEST(GOLDILOCKS_TEST, ntt_gpu_lde)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14; // 4× blowup
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 1;

    std::vector<Goldilocks::Element> h_input(nRows);
    for (uint64_t i = 0; i < nRows; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu(nRows_ext);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu.data(), h_input.data(), nRows_ext, nRows, nCols);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void**)&d_src, nRows     * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_dst, nRows_ext * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_src, h_input.data(), nRows * sizeof(gl64_t), cudaMemcpyHostToDevice));

    gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu(nRows_ext);
    CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_dst, nRows_ext * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < nRows_ext; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu[i]), Goldilocks::toU64(h_cpu[i]))
            << "LDE mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// LDE + buildMerkleTreeGPU(Layout::Tiles) root must match CPU merkletree_seq built from the same extended polynomial.
TEST(GOLDILOCKS_TEST, ntt_gpu_lde_merkletree)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14;
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 1;
    constexpr uint32_t arity      = 3;

    std::vector<Goldilocks::Element> h_input(nRows);
    for (uint64_t i = 0; i < nRows; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(nRows_ext, arity);

    gl64_t *d_src, *d_lde_mt;
    Goldilocks::Element *d_tree_gpu;
    CHECKCUDAERR(cudaMalloc((void**)&d_src,      nRows     * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_lde_mt,   nRows_ext * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree_gpu, tree_size * sizeof(Goldilocks::Element)));

    // GPU: LDE then merkle tree
    CHECKCUDAERR(cudaMemcpy(d_src, h_input.data(), nRows * sizeof(gl64_t), cudaMemcpyHostToDevice));
    gpu_ntt.LDE(d_lde_mt, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    buildMerkleTreeGPU(arity, (uint64_t*)d_tree_gpu, (uint64_t*)d_lde_mt, nCols, nRows_ext, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // CPU reference: LDE then merkletree_seq
    std::vector<Goldilocks::Element> h_cpu_lde(nRows_ext);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu_lde.data(), h_input.data(), nRows_ext, nRows, nCols);

    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_cpu_lde.data(), nCols, nRows_ext, arity, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);

    // Download GPU root and compare with CPU root element by element
    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree_gpu + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "Merkle root mismatch at element " << i;

    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_lde_mt));
    CHECKCUDAERR(cudaFree(d_tree_gpu));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// LDE + buildMerkleTreeGPU(Layout::Tiles) with multiple columns: block-tiled input via fromRowMajorToTiled.
// GPU root must match CPU LDE + merkletree_seq (both row-major).
TEST(GOLDILOCKS_TEST, ntt_gpu_lde_merkletree_multicol)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14;
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 12;
    constexpr uint32_t arity      = 3;

    // Row-major input: h_input[row * nCols + col]
    std::vector<Goldilocks::Element> h_input(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(nRows_ext, arity);

    gl64_t *d_flat, *d_tiled, *d_lde;
    Goldilocks::Element *d_tree_gpu;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,     nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tiled,    nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_lde,      nRows_ext * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree_gpu, tree_size * sizeof(Goldilocks::Element)));

    // GPU: flat → block-tiled → LDE + merkle tree
    CHECKCUDAERR(cudaMemcpy(d_flat, h_input.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToTiled(nRows, nCols, d_flat, d_tiled, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    gpu_ntt.LDE(d_lde, 0, d_tiled, 0, n_bits, n_bits_ext, nCols, timer, stream);
    buildMerkleTreeGPU(arity, (uint64_t*)d_tree_gpu, (uint64_t*)d_lde, nCols, nRows_ext, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // CPU reference: LDE (row-major) then merkletree_seq (row-major)
    std::vector<Goldilocks::Element> h_cpu_lde(nRows_ext * nCols);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu_lde.data(), h_input.data(), nRows_ext, nRows, nCols);

    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_cpu_lde.data(), nCols, nRows_ext, arity, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);

    // Download GPU root and compare with CPU root
    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree_gpu + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "Merkle root mismatch at element " << i << " (nCols=" << nCols << ")";

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_tiled));
    CHECKCUDAERR(cudaFree(d_lde));
    CHECKCUDAERR(cudaFree(d_tree_gpu));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// linearHash(Layout::Tiles): GPU output (block-tiled input) must match CPU linear_hash_seq per row.
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
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(nRows, arity);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_trace.data(), nCols, nRows, arity, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);

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
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(nRows, arity);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);

    std::vector<Goldilocks::Element> h_trace(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_trace[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_trace.data(), nCols, nRows, arity, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);

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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
