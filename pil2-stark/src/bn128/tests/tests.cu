
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "poseidon2_bn128.cuh"

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

// Forward declarations for GPU kernels
__global__ void init_state_kernel(BN128GPUScalarField::Element* state, int t);
__global__ void from_montgomery_kernel(BN128GPUScalarField::Element* state, int t);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
// GPU kernel to initialize state values: state[i] = i (in Montgomery form)
__global__ void init_state_kernel(BN128GPUScalarField::Element* state, int t) {
    for (int i = 0; i < t; i++) {
        // Initialize to i
        state[i].v[0] = i;
        for (int j = 1; j < 8; j++) {
            state[i].v[j] = 0;
        }
        state[i].v.to(); // Convert to Montgomery form
    }
}

// GPU kernel to convert state from Montgomery form and copy to result buffer
__global__ void from_montgomery_kernel(BN128GPUScalarField::Element* state, int t) {
    for (int i = 0; i < t; i++) {
        state[i].v.from();
    }
}

#endif

TEST(BN128_POSEIDON2_TEST, hash_gpu_t2) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 2;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1); // Initialize GPU constants
    
    // Initialize state: state[i] = i (in Montgomery form)
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Run hash kernel
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    // Convert from Montgomery form
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Copy result to host
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    // Convert to hex string for comparison
    char hex0[65], hex1[65]; //64 hex chars + 1 null
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0].v[7], h_state[0].v[6], h_state[0].v[5], h_state[0].v[4],
             h_state[0].v[3], h_state[0].v[2], h_state[0].v[1], h_state[0].v[0]);
    snprintf(hex1, sizeof(hex1), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[1].v[7], h_state[1].v[6], h_state[1].v[5], h_state[1].v[4],
             h_state[1].v[3], h_state[1].v[2], h_state[1].v[1], h_state[1].v[0]);
    delete[] h_state;
    EXPECT_STREQ(hex0, "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
    EXPECT_STREQ(hex1, "0d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");

    t = 16;    
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    
    // Initialize state: state[i] = i (in Montgomery form)
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Run hash kernel
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    // Convert from Montgomery form
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Copy result to host
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    // Convert to hex string for comparison
    char hex15[65]; //64 hex chars + 1 null
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0].v[7], h_state[0].v[6], h_state[0].v[5], h_state[0].v[4],
             h_state[0].v[3], h_state[0].v[2], h_state[0].v[1], h_state[0].v[0]);
    snprintf(hex15, sizeof(hex15), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[15].v[7], h_state[15].v[6], h_state[15].v[5], h_state[15].v[4],
             h_state[15].v[3], h_state[15].v[2], h_state[15].v[1], h_state[15].v[0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
    EXPECT_STREQ(hex15, "0e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
