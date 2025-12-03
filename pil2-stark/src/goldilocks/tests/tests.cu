#include <gtest/gtest.h>
#include "../src/poseidon2_goldilocks.cuh"

TEST(GOLDILOCKS_TEST, poseidon2)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU::init_gpu_const_2(&gpu_id, 1);

    Goldilocks::Element a[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        a[i] = Goldilocks::fromU64(i);
    }

    gl64_t *d_a;
    cudaMalloc((void **)&d_a, SPONGE_WIDTH * sizeof(gl64_t));
    cudaMemcpy(d_a, a, SPONGE_WIDTH * sizeof(gl64_t), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    hash_full_result_2<<<1, 1, SPONGE_WIDTH * sizeof(gl64_t), stream>>>((uint64_t *)d_a, (uint64_t *)d_a);
    cudaStreamSynchronize(stream);
    cudaMemcpy(a, d_a, SPONGE_WIDTH * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaStreamDestroy(stream);
#if SPONGE_WIDTH == 4
    std::cout << "POSEIDON2 TESTS FOR SPONGE WIDTH 4" << std::endl;
    ASSERT_EQ(a[0].fe, uint64_t(0x758085b0af0a16aa));   
    ASSERT_EQ(a[1].fe, uint64_t(0x85141acc29c479de));
    ASSERT_EQ(a[2].fe, uint64_t(0x50127371e2b77ae5));
    ASSERT_EQ(a[3].fe, uint64_t(0xefee3a8033630029));

#elif SPONGE_WIDTH == 12
    std::cout << "POSEIDON2 TESTS FOR SPONGE WIDTH 12" << std::endl;
    ASSERT_EQ(a[0].fe, uint64_t(0x01eaef96bdf1c0c1));   
    ASSERT_EQ(a[1].fe, uint64_t(0x1f0d2cc525b2540c));
    ASSERT_EQ(a[2].fe, uint64_t(0x6282c1dfe1e0358d));
    ASSERT_EQ(a[3].fe, uint64_t(0xe780d721f698e1e6));
    ASSERT_EQ(a[4].fe, uint64_t(0x280c0b6f753d833b));
    ASSERT_EQ(a[5].fe, uint64_t(0x1b942dd5023156ab));
    ASSERT_EQ(a[6].fe, uint64_t(0x43f0df3fcccb8398));
    ASSERT_EQ(a[7].fe, uint64_t(0xe8e8190585489025));
    ASSERT_EQ(a[8].fe, uint64_t(0x56bdbf72f77ada22));
    ASSERT_EQ(a[9].fe, uint64_t(0x7911c32bf9dcd705));
    ASSERT_EQ(a[10].fe, uint64_t(0xec467926508fbe67));
    ASSERT_EQ(a[11].fe, uint64_t(0x6a50450ddf85a6ed));

#elif SPONGE_WIDTH == 16
    std::cout << "POSEIDON2 TESTS FOR SPONGE WIDTH 16" << std::endl;
    ASSERT_EQ(a[0].fe,uint64_t(0x85c54702470d9756));
    ASSERT_EQ(a[1].fe,uint64_t(0xaa53c7a7d52d9898));
    ASSERT_EQ(a[2].fe,uint64_t(0x285128096efb0dd7));
    ASSERT_EQ(a[3].fe,uint64_t(0xf3fde5edd3050ac8));
    ASSERT_EQ(a[4].fe,uint64_t(0xc7b65efd040df908));
    ASSERT_EQ(a[5].fe,uint64_t(0x4be3f6c467f57ae9));
    ASSERT_EQ(a[6].fe,uint64_t(0x274e9a67b41754fb));
    ASSERT_EQ(a[7].fe,uint64_t(0x0f7d39cd5de94dac));
    ASSERT_EQ(a[8].fe,uint64_t(0xd0224b9794d0b78c));
    ASSERT_EQ(a[9].fe,uint64_t(0x372f6139570042e1));
    ASSERT_EQ(a[10].fe,uint64_t(0xce6e8a93dc4ec26c));
    ASSERT_EQ(a[11].fe,uint64_t(0xace65e30a4daf7af));
    ASSERT_EQ(a[12].fe,uint64_t(0x016f2824cc1ba3db));
    ASSERT_EQ(a[13].fe,uint64_t(0x2e8f3af37c434dec));
    ASSERT_EQ(a[14].fe,uint64_t(0xc80831bb6e09da01));
    ASSERT_EQ(a[15].fe,uint64_t(0x3a7d670bf1a86ee8));
#else
#error "Unsupported SPONGE_WIDTH for Poseidon2 tests"
#endif
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
