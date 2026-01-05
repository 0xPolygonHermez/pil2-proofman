
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <bn128.cuh>

__global__ void kernel_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_add_one_one(int* ok)
{
    BN128GPUScalarField::Element a;
    BN128GPUScalarField::Element b;
    BN128GPUScalarField::Element r;

    a.v[0] = 1;
    b.v[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a.v[i] = 0;
        b.v[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUScalarField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r.v[i] != expected) {
            same = false;
        }
    }
    *ok = same;
}
#endif

TEST(BN128, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_add_one_one<<<1,1>>>(d_ok);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ok);

    EXPECT_EQ(h_ok, 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
