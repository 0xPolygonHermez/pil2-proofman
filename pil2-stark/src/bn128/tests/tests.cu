
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "fq.cuh"
#include "poseidon2_bn128.cuh"
#include "msm_bn128.cuh"
#include "point.cuh"

__global__ void kernel_fr_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_fr_add_one_one(int* ok)
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

TEST(BN128_FR, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_fr_add_one_one<<<1,1>>>(d_ok);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ok);

    EXPECT_EQ(h_ok, 1);
}

// =====================
// Fq (Base Field) Tests
// =====================
__global__ void kernel_fq_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_fq_add_one_one(int* ok)
{
    BN128GPUBaseField::Element a;
    BN128GPUBaseField::Element b;
    BN128GPUBaseField::Element r;

    a.v[0] = 1;
    b.v[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a.v[i] = 0;
        b.v[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUBaseField::add(r, a, b);

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

TEST(BN128_FQ, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_fq_add_one_one<<<1,1>>>(d_ok);
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
    
    // Use pointer cast to access the raw uint32_t values 
    char hex0[65], hex1[65]; //64 hex chars + 1 null
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&h_state[1].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex1, sizeof(hex1), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p1[7], p1[6], p1[5], p1[4], p1[3], p1[2], p1[1], p1[0]);
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
    
    // Use pointer cast to access the raw uint32_t values 
    char hex15[65]; //64 hex chars + 1 null
    p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p15 = reinterpret_cast<const uint32_t*>(&h_state[15].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex15, sizeof(hex15), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p15[7], p15[6], p15[5], p15[4], p15[3], p15[2], p15[1], p15[0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
    EXPECT_STREQ(hex15, "0e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
}

// =====================
// MSM (Multi-Scalar Multiplication) GPU Test
// =====================
#ifndef __CUDA_ARCH__

TEST(BN128_MSM, simple_msm) {
    // BN254 generator point G1 = (1, 2) in Montgomery form
    // These are the Montgomery representations for the BN254 base field
    static const uint64_t G1_X_MONT[4] = {
        0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
        0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
    };  // Montgomery(1) mod p
    static const uint64_t G1_Y_MONT[4] = {
        0xa6ba871b8b1e1b3aULL, 0x14f1d651eb8e167bULL,
        0xccdd46def0f28c58ULL, 0x1c14ef83340fbe5eULL
    };  // Montgomery(2) mod p
    
    // Create 4 copies of G1 generator point
    const size_t npoints = 4;
    PointAffineGPU* h_points = new PointAffineGPU[npoints];
    BN128GPUScalarField::Element* h_scalars = new BN128GPUScalarField::Element[npoints];
    
    // Initialize all points as the generator
    for (size_t i = 0; i < npoints; i++) {
        memcpy(&h_points[i].x, G1_X_MONT, sizeof(G1_X_MONT));
        memcpy(&h_points[i].y, G1_Y_MONT, sizeof(G1_Y_MONT));
    }
    
    // Initialize scalars: [1, 2, 3, 4] - NOT in Montgomery form
    // MSM result should be: 1*G + 2*G + 3*G + 4*G = 10*G
    for (size_t i = 0; i < npoints; i++) {
        uint64_t scalar_val[4] = {i + 1, 0, 0, 0};
        memcpy(&h_scalars[i], scalar_val, sizeof(scalar_val));
    }
    
    // Perform MSM on GPU
    PointJacobianGPU result;
    memset(&result, 0, sizeof(result));
    
    MSM_BN128_GPU::msm(result, h_points, h_scalars, npoints, false);
    
    // Verify result is not point at infinity (Z != 0)
    // Cast to raw bytes to check
    const uint64_t* pz = reinterpret_cast<const uint64_t*>(&result.Z);
    bool z_nonzero = (pz[0] | pz[1] | pz[2] | pz[3]) != 0;
    
    EXPECT_TRUE(z_nonzero) << "Result is point at infinity (Z=0)";
    
    // Print result using raw access
    const uint32_t* px = reinterpret_cast<const uint32_t*>(&result.X);
    const uint32_t* py = reinterpret_cast<const uint32_t*>(&result.Y);
    const uint32_t* pz32 = reinterpret_cast<const uint32_t*>(&result.Z);
    
    printf("MSM test completed. Result (10*G1) in Jacobian form:\n");
    printf("  X = %08x%08x%08x%08x%08x%08x%08x%08x\n", 
           px[7], px[6], px[5], px[4], px[3], px[2], px[1], px[0]);
    printf("  Y = %08x%08x%08x%08x%08x%08x%08x%08x\n",
           py[7], py[6], py[5], py[4], py[3], py[2], py[1], py[0]);
    printf("  Z = %08x%08x%08x%08x%08x%08x%08x%08x\n",
           pz32[7], pz32[6], pz32[5], pz32[4], pz32[3], pz32[2], pz32[1], pz32[0]);
    
    delete[] h_points;
    delete[] h_scalars;
}

#endif // !__CUDA_ARCH__

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
