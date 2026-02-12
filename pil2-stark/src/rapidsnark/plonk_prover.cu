
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

#include "../bn128/src/ffigpu/fr.cuh"
#include "cuda_utils.cuh"

using Fr = BN128GPUScalarField;
using Element = Fr::Element;

// ============================================================================
// computeZ GPU kernel
// ============================================================================

// domainSize is always a power of 2 (2^zkeyPower), and threadsPerBlock=256=2^8,
// so every warp is fully occupied. This is required because Fr::reciprocal()
// uses warp-level shuffle instructions (__shfl_xor_sync) internally.
__global__ void computeZRatiosKernel(
    Element* __restrict__ ratioOut,
    const Element* __restrict__ buffA,
    const Element* __restrict__ buffB,
    const Element* __restrict__ buffC,
    const Element* __restrict__ sigma1,
    const Element* __restrict__ sigma2,
    const Element* __restrict__ sigma3,
    uint32_t sigmaStride,
    Element beta,
    Element gamma,
    Element k1,
    Element k2,
    Element omega,
    uint64_t domainSize)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domainSize) return;

    // omega_i = omega^i
    Element omega_i = Fr::pow(omega, (uint32_t)i);
    Element betaw = Fr::mul(beta, omega_i);

    // num = (a + beta*omega^i + gamma)(b + k1*beta*omega^i + gamma)(c + k2*beta*omega^i + gamma)
    Element num1 = Fr::add(Fr::add(buffA[i], betaw), gamma);
    Element num2 = Fr::add(Fr::add(buffB[i], Fr::mul(k1, betaw)), gamma);
    Element num3 = Fr::add(Fr::add(buffC[i], Fr::mul(k2, betaw)), gamma);
    Element num = Fr::mul(num1, Fr::mul(num2, num3));

    // den = (a + beta*sigma1[i] + gamma)(b + beta*sigma2[i] + gamma)(c + beta*sigma3[i] + gamma)
    Element den1 = Fr::add(Fr::add(buffA[i], Fr::mul(beta, sigma1[i * sigmaStride])), gamma);
    Element den2 = Fr::add(Fr::add(buffB[i], Fr::mul(beta, sigma2[i * sigmaStride])), gamma);
    Element den3 = Fr::add(Fr::add(buffC[i], Fr::mul(beta, sigma3[i * sigmaStride])), gamma);
    Element den = Fr::mul(den1, Fr::mul(den2, den3));

    // ratio[i] = num * den^{-1}
    ratioOut[i] = Fr::mul(num, Fr::reciprocal(den));
}

extern "C" void compute_z_ratios_gpu(
    void* ratioOut,
    const void* buffA,
    const void* buffB,
    const void* buffC,
    const void* dStaticEvals,
    const void* betaPtr,
    const void* gammaPtr,
    const void* k1Ptr,
    const void* k2Ptr,
    const void* omegaPtr,
    uint64_t domainSize)
{
    const Element* hBuffA = (const Element*)buffA;
    const Element* hBuffB = (const Element*)buffB;
    const Element* hBuffC = (const Element*)buffC;

    Element beta  = *(const Element*)betaPtr;
    Element gamma = *(const Element*)gammaPtr;
    Element k1    = *(const Element*)k1Ptr;
    Element k2    = *(const Element*)k2Ptr;
    Element omega = *(const Element*)omegaPtr;

    size_t arrayBytes = domainSize * sizeof(Element);
    uint64_t fullN = 4 * domainSize;

    // Sigma evals are already on GPU in d_staticEvalsBuffer: S1, S2, S3 (each fullN elements)
    const Element* dStaticBase = (const Element*)dStaticEvals;
    const Element* dS1 = dStaticBase + 0 * fullN;
    const Element* dS2 = dStaticBase + 1 * fullN;
    const Element* dS3 = dStaticBase + 2 * fullN;

    // Allocate device memory for A, B, C and output
    Element *dRatioOut, *dBuffA, *dBuffB, *dBuffC;

    CHECKCUDAERR(cudaMalloc(&dRatioOut, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffA, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffB, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffC, arrayBytes));

    // Copy inputs H2D
    CHECKCUDAERR(cudaMemcpy(dBuffA, hBuffA, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dBuffB, hBuffB, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dBuffC, hBuffC, arrayBytes, cudaMemcpyHostToDevice));

    // Launch kernel — sigma arrays use stride 4 (full eval size = 4 * domainSize)
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)((domainSize + threadsPerBlock - 1) / threadsPerBlock);

    computeZRatiosKernel<<<blocks, threadsPerBlock>>>(
        dRatioOut, dBuffA, dBuffB, dBuffC,
        dS1, dS2, dS3, 4,
        beta, gamma, k1, k2, omega,
        domainSize);

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    // Copy result D2H
    CHECKCUDAERR(cudaMemcpy(ratioOut, dRatioOut, arrayBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dRatioOut);
    cudaFree(dBuffA);
    cudaFree(dBuffB);
    cudaFree(dBuffC);
}

// ============================================================================
// computePI GPU kernel — precompute PI(X) = -sum_j L_j(X) * publicA[j]
// ============================================================================

// pi[i] -= lagrange[i] * publicVal   (one public input at a time)
__global__ void computePIAccumulateKernel(
    Element* __restrict__ pi,
    const Element* __restrict__ lagrange,  // fullN elements (one L_j slice)
    Element publicVal,
    uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pi[i] = Fr::sub(pi[i], Fr::mul(lagrange[i], publicVal));
}

extern "C" void compute_pi_gpu(
    void* piOut,
    const void* evalLagrange,
    const void* publicA,
    uint64_t fullN, uint32_t nPublic)
{
    size_t fullBytes = fullN * sizeof(Element);
    const Element* hLagrange = (const Element*)evalLagrange;
    const Element* hPublicA  = (const Element*)publicA;

    // Use a non-blocking stream so GPU work doesn't serialize with the default stream
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Allocate device: PI accumulator + one lagrange slice
    Element *dPI, *dLag;
    CHECKCUDAERR(cudaMalloc(&dPI, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dLag, fullBytes));

    // Zero the PI accumulator
    CHECKCUDAERR(cudaMemsetAsync(dPI, 0, fullBytes, stream));

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(fullN / threadsPerBlock);

    for (uint32_t j = 0; j < nPublic; j++) {
        // Upload lagrange slice j
        CHECKCUDAERR(cudaMemcpyAsync(dLag, hLagrange + j * fullN, fullBytes, cudaMemcpyHostToDevice, stream));

        computePIAccumulateKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            dPI, dLag, hPublicA[j], fullN);

        CHECKCUDAERR(cudaGetLastError());
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaMemcpy(piOut, dPI, fullBytes, cudaMemcpyDeviceToHost));

    cudaFree(dPI);
    cudaFree(dLag);
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ============================================================================
// computeT GPU kernel
// ============================================================================

__device__ __forceinline__
void mulz_mul2(Element& r, Element& rz,
               const Element& a, const Element& b,
               const Element& ap, const Element& bp,
               const Element* Z1, uint32_t p)
{
    Element a_b   = Fr::mul(a, b);
    Element a_bp  = Fr::mul(a, bp);
    Element ap_b  = Fr::mul(ap, b);
    Element ap_bp = Fr::mul(ap, bp);

    r = a_b;

    Element a0 = Fr::add(a_bp, ap_b);
    rz = Fr::add(a0, Fr::mul(Z1[p], ap_bp));
}

__device__ __forceinline__
void mulz_mul4(Element& r, Element& rz,
               const Element& a, const Element& b,
               const Element& c, const Element& d,
               const Element& ap, const Element& bp,
               const Element& cp, const Element& dp,
               const Element* Z1, const Element* Z2, const Element* Z3,
               uint32_t p)
{
    Element a_b   = Fr::mul(a, b);
    Element a_bp  = Fr::mul(a, bp);
    Element ap_b  = Fr::mul(ap, b);
    Element ap_bp = Fr::mul(ap, bp);

    Element c_d   = Fr::mul(c, d);
    Element c_dp  = Fr::mul(c, dp);
    Element cp_d  = Fr::mul(cp, d);
    Element cp_dp = Fr::mul(cp, dp);

    r = Fr::mul(a_b, c_d);

    // a0: all single-derivative terms
    Element a0 = Fr::mul(ap_b, c_d);
    a0 = Fr::add(a0, Fr::mul(a_bp, c_d));
    a0 = Fr::add(a0, Fr::mul(a_b, cp_d));
    a0 = Fr::add(a0, Fr::mul(a_b, c_dp));

    // a1: all two-derivative terms
    Element a1 = Fr::mul(ap_bp, c_d);
    a1 = Fr::add(a1, Fr::mul(ap_b, cp_d));
    a1 = Fr::add(a1, Fr::mul(ap_b, c_dp));
    a1 = Fr::add(a1, Fr::mul(a_bp, cp_d));
    a1 = Fr::add(a1, Fr::mul(a_bp, c_dp));
    a1 = Fr::add(a1, Fr::mul(a_b, cp_dp));

    // a2: all three-derivative terms
    Element a2 = Fr::mul(a_bp, cp_dp);
    a2 = Fr::add(a2, Fr::mul(ap_b, cp_dp));
    a2 = Fr::add(a2, Fr::mul(ap_bp, c_dp));
    a2 = Fr::add(a2, Fr::mul(ap_bp, cp_d));

    // a3: all four derivatives
    Element a3 = Fr::mul(ap_bp, cp_dp);

    rz = a0;
    rz = Fr::add(rz, Fr::mul(Z1[p], a1));
    rz = Fr::add(rz, Fr::mul(Z2[p], a2));
    rz = Fr::add(rz, Fr::mul(Z3[p], a3));
}

struct ComputeTConst {
    Element bf[10];       // blindingFactors[0..9]
    Element Z1[4], Z2[4], Z3[4];
    Element beta, gamma, alpha, alpha2, k1, k2;
    Element omega_4x, omega1;
    uint64_t fullN;
};

// tOut is pre-loaded with PI(X); kernel reads pi from tOut[i] before overwriting with result.
// tzOut is pre-loaded with L_0(X); kernel reads lagrange from tzOut[i] before overwriting.
__global__ void computeTEvaluationsKernel(
    Element* __restrict__ tOut,              // IN: PI(X), OUT: T(X)
    Element* __restrict__ tzOut,             // IN: L_0(X), OUT: Tz(X)
    const Element* __restrict__ evalA,
    const Element* __restrict__ evalB,
    const Element* __restrict__ evalC,
    const Element* __restrict__ evalZ,       // fullN + 4 elements (for zW wrap-around)
    const Element* __restrict__ evalQM,
    const Element* __restrict__ evalQL,
    const Element* __restrict__ evalQR,
    const Element* __restrict__ evalQO,
    const Element* __restrict__ evalQC,
    const Element* __restrict__ evalS1,
    const Element* __restrict__ evalS2,
    const Element* __restrict__ evalS3,
    ComputeTConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.fullN) return;

    // Read PI and L_0 from output buffers before overwriting
    Element pi = tOut[i];
    Element lagrange_eval = tzOut[i];

    // Compute roots of unity
    Element omega   = Fr::pow(c.omega_4x, (uint32_t)i);
    Element omega2  = Fr::square(omega);
    Element omegaW  = Fr::mul(omega, c.omega1);
    Element omegaW2 = Fr::square(omegaW);

    // Load evaluations
    Element a  = evalA[i];
    Element b  = evalB[i];
    Element cc = evalC[i];
    Element z  = evalZ[i];
    Element zW = evalZ[i + 4];  // wrap-around handled by extra 4 elements

    Element qm = evalQM[i];
    Element ql = evalQL[i];
    Element qr = evalQR[i];
    Element qo = evalQO[i];
    Element qc = evalQC[i];
    Element s1 = evalS1[i];
    Element s2 = evalS2[i];
    Element s3 = evalS3[i];

    // Blinding derivatives
    Element ap = Fr::add(c.bf[2], Fr::mul(c.bf[1], omega));
    Element bp = Fr::add(c.bf[4], Fr::mul(c.bf[3], omega));
    Element cp = Fr::add(c.bf[6], Fr::mul(c.bf[5], omega));

    Element zp  = Fr::add(Fr::add(Fr::mul(c.bf[7], omega2), Fr::mul(c.bf[8], omega)), c.bf[9]);
    Element zWp = Fr::add(Fr::add(Fr::mul(c.bf[7], omegaW2), Fr::mul(c.bf[8], omegaW)), c.bf[9]);

    // e1: gate constraint
    // e1 = a*b*qm + a*ql + b*qr + c*qo + pi + qc
    Element e1, e1z;
    mulz_mul2(e1, e1z, a, b, ap, bp, c.Z1, (uint32_t)(i % 4));
    e1 = Fr::mul(e1, qm);
    e1z = Fr::mul(e1z, qm);

    e1 = Fr::add(e1, Fr::mul(a, ql));
    e1z = Fr::add(e1z, Fr::mul(ap, ql));

    e1 = Fr::add(e1, Fr::mul(b, qr));
    e1z = Fr::add(e1z, Fr::mul(bp, qr));

    e1 = Fr::add(e1, Fr::mul(cc, qo));
    e1z = Fr::add(e1z, Fr::mul(cp, qo));

    e1 = Fr::add(e1, pi);
    e1 = Fr::add(e1, qc);

    // e2: permutation numerator
    // alpha * (a + beta*omega + gamma)(b + beta*k1*omega + gamma)(c + beta*k2*omega + gamma) * z
    Element betaw = Fr::mul(c.beta, omega);
    Element e2a = Fr::add(Fr::add(a, betaw), c.gamma);
    Element e2b = Fr::add(Fr::add(b, Fr::mul(betaw, c.k1)), c.gamma);
    Element e2c = Fr::add(Fr::add(cc, Fr::mul(betaw, c.k2)), c.gamma);

    Element e2, e2z;
    mulz_mul4(e2, e2z, e2a, e2b, e2c, z, ap, bp, cp, zp, c.Z1, c.Z2, c.Z3, (uint32_t)(i % 4));
    e2 = Fr::mul(e2, c.alpha);
    e2z = Fr::mul(e2z, c.alpha);

    // e3: permutation denominator
    // alpha * (a + beta*s1 + gamma)(b + beta*s2 + gamma)(c + beta*s3 + gamma) * zW
    Element e3a = Fr::add(Fr::add(a, Fr::mul(c.beta, s1)), c.gamma);
    Element e3b = Fr::add(Fr::add(b, Fr::mul(c.beta, s2)), c.gamma);
    Element e3c = Fr::add(Fr::add(cc, Fr::mul(c.beta, s3)), c.gamma);

    Element e3, e3z;
    mulz_mul4(e3, e3z, e3a, e3b, e3c, zW, ap, bp, cp, zWp, c.Z1, c.Z2, c.Z3, (uint32_t)(i % 4));
    e3 = Fr::mul(e3, c.alpha);
    e3z = Fr::mul(e3z, c.alpha);

    // e4: L1 constraint
    // alpha2 * (z - 1) * L1
    Element e4 = Fr::sub(z, Fr::one());
    e4 = Fr::mul(e4, lagrange_eval);
    e4 = Fr::mul(e4, c.alpha2);

    Element e4z = Fr::mul(zp, lagrange_eval);
    e4z = Fr::mul(e4z, c.alpha2);

    // T = e1 + e2 - e3 + e4
    Element t  = Fr::add(Fr::sub(Fr::add(e1, e2), e3), e4);
    Element tz = Fr::add(Fr::sub(Fr::add(e1z, e2z), e3z), e4z);

    tOut[i] = t;
    tzOut[i] = tz;
}

extern "C" void compute_t_evaluations_gpu(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB,
    const void* evalC, const void* evalZ,
    const void* dStaticEvals,    // GPU pointer — 8 arrays already uploaded (S1,S2,S3,QL,QR,QM,QO,QC)
    const void* evalLagrange0,   // fullN elements — L_0(X) for e4
    const void* piPrecomp,       // fullN elements — precomputed PI(X)
    const void* blindFactors,
    const void* betaPtr, const void* gammaPtr,
    const void* alphaPtr, const void* alpha2Ptr,
    const void* k1Ptr, const void* k2Ptr,
    const void* omega4xPtr, const void* omega1Ptr,
    const void* Z1Ptr, const void* Z2Ptr, const void* Z3Ptr,
    uint64_t domainSize)
{
    uint64_t fullN = 4 * domainSize;
    size_t fullBytes = fullN * sizeof(Element);

    // Prepare constants struct
    ComputeTConst consts;
    memcpy(consts.bf, blindFactors, 10 * sizeof(Element));
    memcpy(consts.Z1, Z1Ptr, 4 * sizeof(Element));
    memcpy(consts.Z2, Z2Ptr, 4 * sizeof(Element));
    memcpy(consts.Z3, Z3Ptr, 4 * sizeof(Element));
    consts.beta     = *(const Element*)betaPtr;
    consts.gamma    = *(const Element*)gammaPtr;
    consts.alpha    = *(const Element*)alphaPtr;
    consts.alpha2   = *(const Element*)alpha2Ptr;
    consts.k1       = *(const Element*)k1Ptr;
    consts.k2       = *(const Element*)k2Ptr;
    consts.omega_4x = *(const Element*)omega4xPtr;
    consts.omega1   = *(const Element*)omega1Ptr;
    consts.fullN    = fullN;

    const Element* hEvalZ = (const Element*)evalZ;

    // Static evals are already on GPU: S1,S2,S3,QL,QR,QM,QO,QC (each fullN elements)
    const Element* dStaticBase = (const Element*)dStaticEvals;
    const Element* dS1 = dStaticBase + 0 * fullN;
    const Element* dS2 = dStaticBase + 1 * fullN;
    const Element* dS3 = dStaticBase + 2 * fullN;
    const Element* dQL = dStaticBase + 3 * fullN;
    const Element* dQR = dStaticBase + 4 * fullN;
    const Element* dQM = dStaticBase + 5 * fullN;
    const Element* dQO = dStaticBase + 6 * fullN;
    const Element* dQC = dStaticBase + 7 * fullN;

    // Allocate device arrays: 2 output + 4 dynamic input (A, B, C, Z)
    Element *dT, *dTz;
    Element *dA, *dB, *dC, *dZ;

    CHECKCUDAERR(cudaMalloc(&dT, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dTz, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dA, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dB, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dC, fullBytes));
    CHECKCUDAERR(cudaMalloc(&dZ, (fullN + 4) * sizeof(Element)));

    // Pre-load PI(X) into dT and L_0(X) into dTz (kernel reads before overwriting)
    CHECKCUDAERR(cudaMemcpy(dT, piPrecomp, fullBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dTz, evalLagrange0, fullBytes, cudaMemcpyHostToDevice));

    // Copy dynamic eval arrays H2D
    CHECKCUDAERR(cudaMemcpy(dA, evalA, fullBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dB, evalB, fullBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dC, evalC, fullBytes, cudaMemcpyHostToDevice));

    // Z: fullN elements + 4 extra wrapping around to the start
    CHECKCUDAERR(cudaMemcpy(dZ, hEvalZ, fullBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dZ + fullN, hEvalZ, 4 * sizeof(Element), cudaMemcpyHostToDevice));

    // Launch kernel
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(fullN / threadsPerBlock);

    computeTEvaluationsKernel<<<blocks, threadsPerBlock>>>(
        dT, dTz,
        dA, dB, dC, dZ,
        dQM, dQL, dQR, dQO, dQC,
        dS1, dS2, dS3,
        consts);

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    // Copy results D2H
    CHECKCUDAERR(cudaMemcpy(tOut, dT, fullBytes, cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(tzOut, dTz, fullBytes, cudaMemcpyDeviceToHost));

    // Free only dynamic device memory (static evals are managed externally)
    cudaFree(dT); cudaFree(dTz);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dZ);
}

// ============================================================================
// Async GPU transfer for static evaluation arrays
// ============================================================================

extern "C" void alloc_static_eval_buffers_gpu(
    void** dBuffer,
    uint64_t gpuBytes)
{
    CHECKCUDAERR(cudaMalloc(dBuffer, gpuBytes));
}

extern "C" void free_static_eval_buffers_gpu(void* dBuffer)
{
    if (dBuffer) cudaFree(dBuffer);
}

extern "C" void alloc_pinned_buffer_gpu(void** pinnedBuffer, size_t pinnedSize)
{
    CHECKCUDAERR(cudaMallocHost(pinnedBuffer, pinnedSize));
}

extern "C" void free_pinned_buffer_gpu(void* pinnedBuffer)
{
    if (pinnedBuffer) cudaFreeHost(pinnedBuffer);
}

typedef void (*FileReadFn)(void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx);

extern "C" void start_static_eval_transfer_gpu(
    FileReadFn readFn, void* readCtx,
    void* dBuffer, void* pinnedBuffer, size_t pinnedSize,
    const uint32_t* sectionIds, const uint64_t* byteOffsets, const uint64_t* byteSizes, int numArrays)
{
    size_t halfSize = pinnedSize / 2;
    uint8_t* buf[2] = { (uint8_t*)pinnedBuffer, (uint8_t*)pinnedBuffer + halfSize };
    uint8_t* gpuDst = (uint8_t*)dBuffer;
    int cur = 0;
    bool first = true;
    int chunk_idx = 0;

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int arr = 0; arr < numArrays; arr++) {
        uint64_t remaining = byteSizes[arr];
        uint64_t off = byteOffsets[arr];
        while (remaining > 0) {
            size_t chunk = (remaining < (uint64_t)halfSize) ? (size_t)remaining : halfSize;

            readFn(buf[cur], sectionIds[arr], off, chunk, readCtx);

            if (!first) CHECKCUDAERR(cudaStreamSynchronize(stream));
            CHECKCUDAERR(cudaMemcpyAsync(gpuDst, buf[cur], chunk, cudaMemcpyHostToDevice, stream));

            gpuDst += chunk;
            off += chunk;
            remaining -= chunk;
            cur ^= 1;
            first = false;
            chunk_idx++;
        }
    }
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}
