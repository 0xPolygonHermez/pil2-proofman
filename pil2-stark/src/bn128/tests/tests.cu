
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "fq.cuh"
#include "poseidon2_bn128.cuh"
#include "msm_bn128.cuh"
#include "point.cuh"
#include "alt_bn128.hpp"

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

TEST(BN128_MSM, msm) {
    // Use CPU curve for computing expected result
    AltBn128::G1PointAffine& G = AltBn128::G1.oneAffine();
    
    // Create points: [G, 2G, 4G, 8G]
    // With large 253-bit scalars (same as CPU test)
    // MSM result: s0*G + s1*(2G) + s2*(4G) + s3*(8G) = (s0 + 2*s1 + 4*s2 + 8*s3)*G
    const size_t npoints = 4;
    const size_t scalarSize = 32;  // 256-bit scalars
    PointAffineGPU* h_points = new PointAffineGPU[npoints];
    BN128GPUScalarField::Element* h_scalars = new BN128GPUScalarField::Element[npoints];
    
    // Compute points: G, 2G, 4G, 8G using CPU
    AltBn128::G1Point P;
    AltBn128::G1PointAffine P_affine;
    AltBn128::G1.copy(P, G);  // P = G
    
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::G1.copy(P_affine, P);
        memcpy(&h_points[i].x, &P_affine.x, sizeof(AltBn128::F1Element));
        memcpy(&h_points[i].y, &P_affine.y, sizeof(AltBn128::F1Element));
        AltBn128::G1.dbl(P, P);
    }
    
    // Large 253-bit scalars (same as CPU multiexp test)
    const char* scalarStrs[4] = {
        "5708990770823839524233143877797980545530985996",
        "8563486156235759286349715816696970818296478975",
        "9234567890123456789012345678901234567890123456",
        "10876543210987654321098765432109876543210987654"
    };
    
    // Parse scalars and convert to little-endian format for GPU
    AltBn128::FrElement rawScalars[4];
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::Fr.fromString(rawScalars[i], scalarStrs[i], 10);
        // Fr stores in Montgomery form internally, but we need raw LE for GPU
        // Convert to big-endian bytes, then reverse to little-endian
        uint8_t beBytes[scalarSize];
        AltBn128::Fr.toRprBE(rawScalars[i], beBytes, scalarSize);
        // Reverse to little-endian (GPU MSM expects LE)
        uint8_t leBytes[scalarSize];
        for (size_t j = 0; j < scalarSize; j++) {
            leBytes[j] = beBytes[scalarSize - 1 - j];
        }
        memcpy(&h_scalars[i], leBytes, scalarSize);
    }
    
    // ========== GPU MSM ==========
    PointJacobianGPU gpu_result;
    memset(&gpu_result, 0, sizeof(gpu_result));
    
    MSM_BN128_GPU::msm(gpu_result, h_points, h_scalars, npoints, false);
    
    // ========== CPU verification ==========
    // Compute combined scalar: s0 + 2*s1 + 4*s2 + 8*s3
    AltBn128::FrElement combinedScalar;
    AltBn128::Fr.fromUI(combinedScalar, 0);
    
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::FrElement powerOfTwo;
        AltBn128::Fr.fromUI(powerOfTwo, 1ULL << i);
        AltBn128::FrElement term;
        AltBn128::Fr.mul(term, rawScalars[i], powerOfTwo);
        AltBn128::Fr.add(combinedScalar, combinedScalar, term);
    }
    
    // Convert combined scalar to little-endian bytes
    uint8_t combinedBE[scalarSize];
    AltBn128::Fr.toRprBE(combinedScalar, combinedBE, scalarSize);
    uint8_t combinedLE[scalarSize];
    for (size_t j = 0; j < scalarSize; j++) {
        combinedLE[j] = combinedBE[scalarSize - 1 - j];
    }
    
    // Compute expected: (s0 + 2*s1 + 4*s2 + 8*s3) * G
    AltBn128::G1Point cpu_result;
    AltBn128::G1.mulByScalar(cpu_result, G, combinedLE, scalarSize);
    
    // Convert CPU result to affine
    AltBn128::G1PointAffine cpu_affine;
    AltBn128::G1.copy(cpu_affine, cpu_result);
    
    // Convert GPU result (Jacobian) to affine using CPU field operations
    // GPU Jacobian: affine_x = X/Z^2, affine_y = Y/Z^3
    
    AltBn128::F1Element gpu_X, gpu_Y, gpu_Z;
    memcpy(&gpu_X, &gpu_result.X, sizeof(AltBn128::F1Element));
    memcpy(&gpu_Y, &gpu_result.Y, sizeof(AltBn128::F1Element));
    memcpy(&gpu_Z, &gpu_result.Z, sizeof(AltBn128::F1Element));
    
    // Compute Z^2 and Z^3
    AltBn128::F1Element z2, z3, z_inv, z2_inv, z3_inv;
    AltBn128::F1.square(z2, gpu_Z);
    AltBn128::F1.mul(z3, z2, gpu_Z);
    
    // Compute inverses
    AltBn128::F1.inv(z2_inv, z2);
    AltBn128::F1.inv(z3_inv, z3);
    
    // Compute affine coordinates
    AltBn128::F1Element gpu_affine_x, gpu_affine_y;
    AltBn128::F1.mul(gpu_affine_x, gpu_X, z2_inv);
    AltBn128::F1.mul(gpu_affine_y, gpu_Y, z3_inv);
    
    // Compare with CPU affine result
    bool x_eq = AltBn128::F1.eq(gpu_affine_x, cpu_affine.x);
    bool y_eq = AltBn128::F1.eq(gpu_affine_y, cpu_affine.y);
    
    EXPECT_TRUE(x_eq) << "GPU X coordinate does not match CPU";
    EXPECT_TRUE(y_eq) << "GPU Y coordinate does not match CPU";
    
    if (!x_eq || !y_eq) {
        // Print both results for debugging
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(&gpu_affine_x);
        const uint32_t* gy = reinterpret_cast<const uint32_t*>(&gpu_affine_y);
        const uint32_t* cx = reinterpret_cast<const uint32_t*>(&cpu_affine.x);
        const uint32_t* cy = reinterpret_cast<const uint32_t*>(&cpu_affine.y);
        
        printf("GPU affine X = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               gx[7], gx[6], gx[5], gx[4], gx[3], gx[2], gx[1], gx[0]);
        printf("CPU affine X = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               cx[7], cx[6], cx[5], cx[4], cx[3], cx[2], cx[1], cx[0]);
        printf("GPU affine Y = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               gy[7], gy[6], gy[5], gy[4], gy[3], gy[2], gy[1], gy[0]);
        printf("CPU affine Y = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               cy[7], cy[6], cy[5], cy[4], cy[3], cy[2], cy[1], cy[0]);
    }
    
    delete[] h_points;
    delete[] h_scalars;
}

#endif // !__CUDA_ARCH__

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
