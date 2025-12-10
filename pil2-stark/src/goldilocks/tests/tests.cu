#include <gtest/gtest.h>
#include "../src/poseidon2_goldilocks.cuh"


TEST(GOLDILOCKS_TEST, poseidon2)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);

    Goldilocks::Element in[16], out[16];
    for (int i = 0; i < 16; i++)
    {
        in[i] = Goldilocks::fromU64(i);
    }

    gl64_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, 16 * sizeof(gl64_t));
    cudaMemcpy(d_in, in, 16 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_out, 16 * sizeof(gl64_t));
   
    Poseidon2GoldilocksGPU<4>::hashFullResult((uint64_t *)d_out, (uint64_t *)d_in);
    cudaMemcpy(out, d_out, 4 * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(out[0].fe, uint64_t(0x758085b0af0a16aa));   
    ASSERT_EQ(out[1].fe, uint64_t(0x85141acc29c479de));
    ASSERT_EQ(out[2].fe, uint64_t(0x50127371e2b77ae5));
    ASSERT_EQ(out[3].fe, uint64_t(0xefee3a8033630029));

    Poseidon2GoldilocksGPU<12>::hashFullResult((uint64_t *)d_out, (uint64_t *)d_in);
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

    Poseidon2GoldilocksGPU<16>::hashFullResult((uint64_t *)d_out, (uint64_t *)d_in);
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
    Poseidon2GoldilocksGPUGrinding::initPoseidon2GPUConstants(&gpu_id, 1);

    // Input data for grinding (4 elements for SPONGE_WIDTH=4)
    Goldilocks::Element in[4];
    for (int i = 0; i < 3; i++)
    {
        in[i] = Goldilocks::fromU64(i * 7); 
    }

    gl64_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, 4 * sizeof(gl64_t));
    cudaMemcpy(d_in, in, 4 * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_out, sizeof(gl64_t));

    uint32_t n_bits = 8; // Looking for hash with 8 leading zero bits
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Poseidon2GoldilocksGPUGrinding::grinding((uint64_t *)d_out, (uint64_t *)d_in, n_bits, stream);
    
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
    
    Poseidon2GoldilocksGPU<4>::hashFullResult((uint64_t *)d_hash_out, (uint64_t *)d_test_in);
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
    cudaStreamDestroy(stream);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
