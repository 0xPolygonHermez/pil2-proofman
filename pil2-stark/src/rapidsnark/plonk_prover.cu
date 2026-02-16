
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

typedef void (*FileReadFn)(void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx);

// pi[i] -= lagrange[i] * publicVal   (one public input at a time)
__global__ void computePIAccumulateKernel(
    Element* __restrict__ pi,
    const Element* __restrict__ lagrange,  // NExt elements (one L_j slice)
    Element publicVal,
    uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pi[i] = Fr::sub(pi[i], Fr::mul(lagrange[i], publicVal));
}

extern "C" void gpu_plonk_compute_pi(
    void* piOut,
    FileReadFn readFn, void* readCtx,
    uint32_t lagrangeSectionId,
    uint64_t lagrangeBaseOffset,
    uint64_t lagrangeStride,
    const void* publicA,
    uint64_t NExt, uint32_t nPublic,
    void* dPI, void* dLag,
    void* pinnedBuf, size_t pinnedSize)
{
    size_t fullBytes = NExt * sizeof(Element);
    const Element* hPublicA = (const Element*)publicA;

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Zero the PI accumulator
    CHECKCUDAERR(cudaMemsetAsync(dPI, 0, fullBytes, stream));

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    for (uint32_t j = 0; j < nPublic; j++) {
        // Transfer L_j from file to GPU (chunked if slice > pinnedSize)
        uint64_t remaining = fullBytes;
        uint64_t fileOffset = lagrangeBaseOffset + j * lagrangeStride;
        uint8_t* dDst = (uint8_t*)dLag;

        while (remaining > 0) {
            size_t chunk = (remaining < pinnedSize) ? (size_t)remaining : pinnedSize;
            CHECKCUDAERR(cudaStreamSynchronize(stream));
            readFn(pinnedBuf, lagrangeSectionId, fileOffset, chunk, readCtx);
            CHECKCUDAERR(cudaMemcpyAsync(dDst, pinnedBuf, chunk, cudaMemcpyHostToDevice, stream));
            dDst += chunk;
            fileOffset += chunk;
            remaining -= chunk;
        }

        computePIAccumulateKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            (Element*)dPI, (Element*)dLag, hPublicA[j], NExt);

        CHECKCUDAERR(cudaGetLastError());
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // D2H to CPU only if caller needs it; when piOut == NULL, PI stays on GPU in dPI
    if (piOut) {
        CHECKCUDAERR(cudaMemcpy(piOut, dPI, fullBytes, cudaMemcpyDeviceToHost));
    }

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
    uint64_t NExt;
};

// tOut is pre-loaded with PI(X); kernel reads pi from tOut[i] before overwriting with result.
// tzOut is pre-loaded with L_0(X); kernel reads lagrange from tzOut[i] before overwriting.
__global__ void computeTEvaluationsKernel(
    Element* __restrict__ tOut,              // IN: PI(X), OUT: T(X)
    Element* __restrict__ tzOut,             // IN: L_0(X), OUT: Tz(X)
    const Element* __restrict__ evalA,
    const Element* __restrict__ evalB,
    const Element* __restrict__ evalC,
    const Element* __restrict__ evalZ,       // NExt + 4 elements (for zW wrap-around)
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
    if (i >= c.NExt) return;

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

extern "C" void gpu_plonk_compute_t_evaluations(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB,
    const void* evalC, const void* evalZ,
    const void* dStaticEvals,    // GPU pointer — 9 arrays: S1,S2,S3,L0,QL,QR,QM,QO,QC
    const void* piPrecomp,       // NExt elements — precomputed PI(X)
    const void* blindFactors,
    const void* betaPtr, const void* gammaPtr,
    const void* alphaPtr, const void* alpha2Ptr,
    const void* k1Ptr, const void* k2Ptr,
    const void* omega4xPtr, const void* omega1Ptr,
    const void* Z1Ptr, const void* Z2Ptr, const void* Z3Ptr,
    uint64_t N,
    void* dScratch0, void* dScratch1,
    bool evalsOnGPU,
    const void* dEvalA, const void* dEvalB, const void* dEvalC,
    const void* dEvalZ,
    bool piOnGPU,
    bool tResultOnGPU)
{
    uint64_t NExt = 4 * N;
    size_t fullBytes = NExt * sizeof(Element);

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
    consts.NExt    = NExt;

    const Element* hEvalZ = (const Element*)evalZ;

    // Static evals on GPU: S1(0), S2(1), S3(2), L0(3), QL(4), QR(5), QM(6), QO(7), QC(8)
    const Element* dStaticBase = (const Element*)dStaticEvals;
    const Element* dS1 = dStaticBase + 0 * NExt;
    const Element* dS2 = dStaticBase + 1 * NExt;
    const Element* dS3 = dStaticBase + 2 * NExt;
    const Element* dL0 = dStaticBase + 3 * NExt;
    const Element* dQL = dStaticBase + 4 * NExt;
    const Element* dQR = dStaticBase + 5 * NExt;
    const Element* dQM = dStaticBase + 6 * NExt;
    const Element* dQO = dStaticBase + 7 * NExt;
    const Element* dQC = dStaticBase + 8 * NExt;

    // Reuse pre-allocated GPU buffers for T and Tz outputs
    Element *dT  = (Element*)dScratch0;
    Element *dTz = (Element*)dScratch1;
    Element *dA, *dB, *dC, *dZ;

    if (evalsOnGPU) {
        // A, B, C already on GPU — use device pointers directly
        dA = (Element*)dEvalA;
        dB = (Element*)dEvalB;
        dC = (Element*)dEvalC;
    } else {
        // Allocate and H2D for A, B, C
        CHECKCUDAERR(cudaMalloc(&dA, fullBytes));
        CHECKCUDAERR(cudaMalloc(&dB, fullBytes));
        CHECKCUDAERR(cudaMalloc(&dC, fullBytes));
        CHECKCUDAERR(cudaMemcpy(dA, evalA, fullBytes, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dB, evalB, fullBytes, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dC, evalC, fullBytes, cudaMemcpyHostToDevice));
    }

    if (dEvalZ) {
        // Z already on GPU with wrap-around set
        dZ = (Element*)dEvalZ;
    } else {
        // Allocate and H2D for Z (NExt + 4 elements for wrap-around)
        CHECKCUDAERR(cudaMalloc(&dZ, (NExt + 4) * sizeof(Element)));
        CHECKCUDAERR(cudaMemcpy(dZ, hEvalZ, fullBytes, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dZ + NExt, hEvalZ, 4 * sizeof(Element), cudaMemcpyHostToDevice));
    }

    // Pre-load PI(X) into dT and L_0(X) into dTz
    if (piOnGPU) {
        // PI already in dT (= d_piBuffer) from async gpu_plonk_compute_pi — skip H2D
    } else {
        CHECKCUDAERR(cudaMemcpy(dT, piPrecomp, fullBytes, cudaMemcpyHostToDevice));
    }
    CHECKCUDAERR(cudaMemcpy(dTz, dL0, fullBytes, cudaMemcpyDeviceToDevice));

    // Launch kernel
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    computeTEvaluationsKernel<<<blocks, threadsPerBlock>>>(
        dT, dTz,
        dA, dB, dC, dZ,
        dQM, dQL, dQR, dQO, dQC,
        dS1, dS2, dS3,
        consts);

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    // Copy results D2H (skip if caller will IFFT on GPU and D2H later)
    if (!tResultOnGPU) {
        CHECKCUDAERR(cudaMemcpy(tOut, dT, fullBytes, cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(tzOut, dTz, fullBytes, cudaMemcpyDeviceToHost));
    }

    // Free only dynamically allocated device memory
    if (!evalsOnGPU) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    if (!dEvalZ) {
        cudaFree(dZ);
    }
}

// ============================================================================
// Async GPU transfer for static evaluation arrays
// ============================================================================

extern "C" void gpu_plonk_memcpy_h2d(void* dst, const void* src, size_t bytes)
{
    CHECKCUDAERR(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

extern "C" void gpu_plonk_memcpy_d2h(void* dst, const void* src, size_t bytes)
{
    CHECKCUDAERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void gpu_plonk_memcpy_d2d(void* dst, const void* src, size_t bytes)
{
    CHECKCUDAERR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

extern "C" void gpu_plonk_cuda_malloc(
    void** dBuffer,
    uint64_t gpuBytes)
{
    CHECKCUDAERR(cudaMalloc(dBuffer, gpuBytes));
}

extern "C" void gpu_plonk_free_static_eval_buffers(void* dBuffer)
{
    if (dBuffer) cudaFree(dBuffer);
}

extern "C" void gpu_plonk_cuda_malloc_pinned_buffer(void** pinnedBuffer, size_t pinnedSize)
{
    CHECKCUDAERR(cudaMallocHost(pinnedBuffer, pinnedSize));
}

extern "C" void gpu_plonk_free_pinned_buffer(void* pinnedBuffer)
{
    if (pinnedBuffer) cudaFreeHost(pinnedBuffer);
}

extern "C" void gpu_plonk_cuda_device_sync()
{
    cudaDeviceSynchronize();
}

extern "C" void* gpu_plonk_create_cuda_stream_nonblocking()
{
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return (void*)stream;
}

extern "C" void gpu_plonk_destroy_cuda_stream(void* stream)
{
    CHECKCUDAERR(cudaStreamDestroy((cudaStream_t)stream));
}

extern "C" void gpu_plonk_sync_cuda_stream(void* stream)
{
    CHECKCUDAERR(cudaStreamSynchronize((cudaStream_t)stream));
}

extern "C" void gpu_plonk_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream)
{
    CHECKCUDAERR(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

extern "C" void gpu_plonk_pin_host_memory(void* ptr, size_t bytes)
{
    CHECKCUDAERR(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault));
}

extern "C" void gpu_plonk_unpin_host_memory(void* ptr)
{
    CHECKCUDAERR(cudaHostUnregister(ptr));
}

extern "C" void gpu_plonk_start_static_eval_transfer(
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

// CPU-to-GPU transfer using double-buffered pinned staging (same pattern as
// gpu_plonk_start_static_eval_transfer but reads from host memory instead of file).
extern "C" void gpu_plonk_start_cpu_to_gpu_transfer(
    void** dDsts, const void** hostSrcs, const size_t* sizes, int numArrays,
    void* pinnedBuffer, size_t pinnedSize)
{
    size_t halfSize = pinnedSize / 2;
    uint8_t* buf[2] = { (uint8_t*)pinnedBuffer, (uint8_t*)pinnedBuffer + halfSize };
    int cur = 0;
    bool first = true;

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int arr = 0; arr < numArrays; arr++) {
        uint64_t remaining = sizes[arr];
        uint64_t srcOff = 0;
        uint8_t* gpuDst = (uint8_t*)dDsts[arr];
        const uint8_t* hostSrc = (const uint8_t*)hostSrcs[arr];

        while (remaining > 0) {
            size_t chunk = (remaining < (uint64_t)halfSize) ? (size_t)remaining : halfSize;

            memcpy(buf[cur], hostSrc + srcOff, chunk);

            if (!first) CHECKCUDAERR(cudaStreamSynchronize(stream));
            CHECKCUDAERR(cudaMemcpyAsync(gpuDst, buf[cur], chunk, cudaMemcpyHostToDevice, stream));

            gpuDst += chunk;
            srcOff += chunk;
            remaining -= chunk;
            cur ^= 1;
            first = false;
        }
    }
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ============================================================================
// Incremental computeT kernels — process one wire at a time
// ============================================================================

// Zero elements [startElem, endElem) on GPU device buffer
__global__ void zeroPadKernel(Element* __restrict__ buf, uint64_t startElem, uint64_t count)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    buf[startElem + i] = Fr::zero();
}

extern "C" void gpu_plonk_zero_pad(void* buf, uint64_t startElem, uint64_t endElem)
{
    uint64_t count = endElem - startElem;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)((count + threadsPerBlock - 1) / threadsPerBlock);
    zeroPadKernel<<<blocks, threadsPerBlock>>>((Element*)buf, startElem, count);
    CHECKCUDAERR(cudaGetLastError());
    // Must sync before NTT which runs on its own stream (non-blocking)
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// Constants struct for incremental gate kernels
struct IncrGateConst {
    Element bf[10];       // blindingFactors[0..9] — raw values (not Montgomery)
    Element omega_4x;     // primitive root of the 4x extended domain
    uint64_t NExt;
};

// Gate A kernel: T = a*QL + PI,  Tz = ap*QL
// PI is pre-loaded in tOut; this kernel OVERWRITES tOut with the result.
// ap(i) = bf[2] + bf[1]*omega^i  (derivative of a's blind polynomial)
__global__ void kernelGateA(
    Element* __restrict__ tOut,              // IN: PI(X), OUT: T accumulator
    Element* __restrict__ tzOut,             // OUT: Tz accumulator (first write)
    const Element* __restrict__ evalA,       // evalsA (4N elements)
    const Element* __restrict__ evalQL,      // QL evaluations (4N elements)
    const Element* __restrict__ omegaBases,  // precomputed omega^(blockIdx*256)
    const Element* __restrict__ omegaTid,    // precomputed omega^0..omega^255
    IncrGateConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.NExt) return;

    Element a  = evalA[i];
    Element ql = evalQL[i];
    Element pi = tOut[i];    // read PI before overwriting

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element ap = Fr::add(c.bf[2], Fr::mul(c.bf[1], omega));

    tOut[i]  = Fr::add(Fr::mul(a, ql), pi);    // T = a*QL + PI
    tzOut[i] = Fr::mul(ap, ql);                 // Tz = ap*QL  (first write)
}

// Gate B kernel: T += b*QR,  Tz += bp*QR
// bp(i) = bf[4] + bf[3]*omega^i
__global__ void kernelGateB(
    Element* __restrict__ tOut,
    Element* __restrict__ tzOut,
    const Element* __restrict__ evalB,
    const Element* __restrict__ evalQR,
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    IncrGateConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.NExt) return;

    Element b  = evalB[i];
    Element qr = evalQR[i];

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element bp = Fr::add(c.bf[4], Fr::mul(c.bf[3], omega));

    tOut[i]  = Fr::add(tOut[i], Fr::mul(b, qr));
    tzOut[i] = Fr::add(tzOut[i], Fr::mul(bp, qr));
}

// Gate C kernel: T += c*QO + QC,  Tz += cp*QO
// cp(i) = bf[6] + bf[5]*omega^i
__global__ void kernelGateC(
    Element* __restrict__ tOut,
    Element* __restrict__ tzOut,
    const Element* __restrict__ evalC,
    const Element* __restrict__ evalQO,
    const Element* __restrict__ evalQC,
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    IncrGateConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.NExt) return;

    Element cc = evalC[i];
    Element qo = evalQO[i];
    Element qc = evalQC[i];

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element cp = Fr::add(c.bf[6], Fr::mul(c.bf[5], omega));

    tOut[i]  = Fr::add(Fr::add(tOut[i], Fr::mul(cc, qo)), qc);
    tzOut[i] = Fr::add(tzOut[i], Fr::mul(cp, qo));
}

// QM + Permutation kernel: computes a*b*QM + e2 - e3 + e4 and all derivatives
// evalsA/B/C are read from the overwritten Q slots (QL/QR/QO positions).
// evalsZ is in a separate temp buffer.
struct QMPermConst {
    Element bf[10];
    Element Z1[4], Z2[4], Z3[4];
    Element beta, gamma, alpha, alpha2, k1, k2;
    Element omega_4x, omega1;
    uint64_t NExt;
};

__global__ void kernelQMPermutation(
    Element* __restrict__ tOut,
    Element* __restrict__ tzOut,
    const Element* __restrict__ evalA,       // stored in QL slot after gate_A
    const Element* __restrict__ evalB,       // stored in QR slot after gate_B
    const Element* __restrict__ evalC,       // stored in QO slot after gate_C
    const Element* __restrict__ evalZ,       // 4N+4 elements (wrap-around)
    const Element* __restrict__ evalQM,
    const Element* __restrict__ evalS1,
    const Element* __restrict__ evalS2,
    const Element* __restrict__ evalS3,
    const Element* __restrict__ evalL0,      // L_0 evaluations
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    QMPermConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.NExt) return;

    Element a  = evalA[i];
    Element b  = evalB[i];
    Element cc = evalC[i];
    Element z  = evalZ[i];
    Element zW = evalZ[i + 4];    // wrap-around

    Element qm = evalQM[i];
    Element s1 = evalS1[i];
    Element s2 = evalS2[i];
    Element s3 = evalS3[i];
    Element lagrange = evalL0[i];

    // Compute roots of unity (precomputed base + per-thread table)
    Element omega   = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element omega2  = Fr::square(omega);
    Element omegaW  = Fr::mul(omega, c.omega1);
    Element omegaW2 = Fr::square(omegaW);

    // Blinding derivatives
    Element ap = Fr::add(c.bf[2], Fr::mul(c.bf[1], omega));
    Element bp = Fr::add(c.bf[4], Fr::mul(c.bf[3], omega));
    Element cp = Fr::add(c.bf[6], Fr::mul(c.bf[5], omega));
    Element zp  = Fr::add(Fr::add(Fr::mul(c.bf[7], omega2), Fr::mul(c.bf[8], omega)), c.bf[9]);
    Element zWp = Fr::add(Fr::add(Fr::mul(c.bf[7], omegaW2), Fr::mul(c.bf[8], omegaW)), c.bf[9]);

    // a*b*QM term
    Element abqm, abqmz;
    mulz_mul2(abqm, abqmz, a, b, ap, bp, c.Z1, (uint32_t)(i % 4));
    abqm  = Fr::mul(abqm, qm);
    abqmz = Fr::mul(abqmz, qm);

    // e2: permutation numerator
    Element betaw = Fr::mul(c.beta, omega);
    Element e2a = Fr::add(Fr::add(a, betaw), c.gamma);
    Element e2b = Fr::add(Fr::add(b, Fr::mul(betaw, c.k1)), c.gamma);
    Element e2c = Fr::add(Fr::add(cc, Fr::mul(betaw, c.k2)), c.gamma);

    Element e2, e2z;
    mulz_mul4(e2, e2z, e2a, e2b, e2c, z, ap, bp, cp, zp, c.Z1, c.Z2, c.Z3, (uint32_t)(i % 4));
    e2  = Fr::mul(e2, c.alpha);
    e2z = Fr::mul(e2z, c.alpha);

    // e3: permutation denominator
    Element e3a = Fr::add(Fr::add(a, Fr::mul(c.beta, s1)), c.gamma);
    Element e3b = Fr::add(Fr::add(b, Fr::mul(c.beta, s2)), c.gamma);
    Element e3c = Fr::add(Fr::add(cc, Fr::mul(c.beta, s3)), c.gamma);

    Element e3, e3z;
    mulz_mul4(e3, e3z, e3a, e3b, e3c, zW, ap, bp, cp, zWp, c.Z1, c.Z2, c.Z3, (uint32_t)(i % 4));
    e3  = Fr::mul(e3, c.alpha);
    e3z = Fr::mul(e3z, c.alpha);

    // e4: L1 constraint  alpha2 * (z - 1) * L0
    Element e4 = Fr::mul(Fr::mul(Fr::sub(z, Fr::one()), lagrange), c.alpha2);
    Element e4z = Fr::mul(Fr::mul(zp, lagrange), c.alpha2);

    // Accumulate: T += a*b*QM + e2 - e3 + e4
    Element contrib  = Fr::add(Fr::sub(Fr::add(abqm, e2), e3), e4);
    Element contribz = Fr::add(Fr::sub(Fr::add(abqmz, e2z), e3z), e4z);

    tOut[i]  = Fr::add(tOut[i], contrib);
    tzOut[i] = Fr::add(tzOut[i], contribz);
}

// C wrapper for incremental gate kernels
extern "C" void gpu_plonk_compute_gate_a(
    void* tOut, void* tzOut,
    const void* evalA, const void* dStaticEvals,
    const void* blindFactors, const void* omega4xPtr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;

    IncrGateConst c;
    memcpy(c.bf, blindFactors, 10 * sizeof(Element));
    c.omega_4x = *(const Element*)omega4xPtr;
    c.NExt = NExt;

    // QL is at offset 4*NExt in static evals buffer (slot 4)
    const Element* dQL = (const Element*)dStaticEvals + 4 * NExt;

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateA<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalA, dQL,
        (const Element*)omegaBases, (const Element*)omegaTid, c);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_gate_b(
    void* tOut, void* tzOut,
    const void* evalB, const void* dStaticEvals,
    const void* blindFactors, const void* omega4xPtr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;

    IncrGateConst c;
    memcpy(c.bf, blindFactors, 10 * sizeof(Element));
    c.omega_4x = *(const Element*)omega4xPtr;
    c.NExt = NExt;

    // QR is at offset 5*NExt in static evals buffer (slot 5)
    const Element* dQR = (const Element*)dStaticEvals + 5 * NExt;

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateB<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalB, dQR,
        (const Element*)omegaBases, (const Element*)omegaTid, c);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_gate_c(
    void* tOut, void* tzOut,
    const void* evalC, const void* dStaticEvals,
    const void* blindFactors, const void* omega4xPtr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;

    IncrGateConst c;
    memcpy(c.bf, blindFactors, 10 * sizeof(Element));
    c.omega_4x = *(const Element*)omega4xPtr;
    c.NExt = NExt;

    // QO at offset 7*NExt, QC at offset 8*NExt in static evals buffer
    const Element* dQO = (const Element*)dStaticEvals + 7 * NExt;
    const Element* dQC = (const Element*)dStaticEvals + 8 * NExt;

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateC<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalC, dQO, dQC,
        (const Element*)omegaBases, (const Element*)omegaTid, c);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_qm_permutation(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB, const void* evalC,
    const void* evalZ,
    const void* dStaticEvals,
    const void* blindFactors,
    const void* betaPtr, const void* gammaPtr,
    const void* alphaPtr, const void* alpha2Ptr,
    const void* k1Ptr, const void* k2Ptr,
    const void* omega4xPtr, const void* omega1Ptr,
    const void* Z1Ptr, const void* Z2Ptr, const void* Z3Ptr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;

    QMPermConst c;
    memcpy(c.bf, blindFactors, 10 * sizeof(Element));
    memcpy(c.Z1, Z1Ptr, 4 * sizeof(Element));
    memcpy(c.Z2, Z2Ptr, 4 * sizeof(Element));
    memcpy(c.Z3, Z3Ptr, 4 * sizeof(Element));
    c.beta     = *(const Element*)betaPtr;
    c.gamma    = *(const Element*)gammaPtr;
    c.alpha    = *(const Element*)alphaPtr;
    c.alpha2   = *(const Element*)alpha2Ptr;
    c.k1       = *(const Element*)k1Ptr;
    c.k2       = *(const Element*)k2Ptr;
    c.omega_4x = *(const Element*)omega4xPtr;
    c.omega1   = *(const Element*)omega1Ptr;
    c.NExt    = NExt;

    // Static eval layout: S1(0), S2(1), S3(2), L0(3), QL(4), QR(5), QM(6), QO(7), QC(8)
    const Element* dStaticBase = (const Element*)dStaticEvals;
    const Element* dS1 = dStaticBase + 0 * NExt;
    const Element* dS2 = dStaticBase + 1 * NExt;
    const Element* dS3 = dStaticBase + 2 * NExt;
    const Element* dL0 = dStaticBase + 3 * NExt;
    const Element* dQM = dStaticBase + 6 * NExt;

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelQMPermutation<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalA, (const Element*)evalB, (const Element*)evalC,
        (const Element*)evalZ,
        dQM, dS1, dS2, dS3, dL0,
        (const Element*)omegaBases, (const Element*)omegaTid,
        c);
    CHECKCUDAERR(cudaGetLastError());
    // Must sync before INTT which runs on its own stream
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// Precompute omega^(i * blockSize) for all block indices
__global__ void kernelPrecomputeOmegaBases(
    Element* __restrict__ bases,
    Element omega,
    uint32_t blockSize,
    uint32_t numBlocks)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBlocks) return;
    bases[i] = Fr::pow(omega, (uint32_t)(i * blockSize));
}

// Precompute omega^0 .. omega^255
__global__ void kernelPrecomputeOmegaTid(
    Element* __restrict__ table,
    Element omega)
{
    uint32_t i = threadIdx.x;
    table[i] = Fr::pow(omega, i);
}

extern "C" void gpu_plonk_precompute_omega_tables_async(
    void* dBases, void* dTid, const void* omega4xPtr,
    uint32_t blockSize, uint32_t numBlocks, void* stream)
{
    Element omega = *(const Element*)omega4xPtr;
    cudaStream_t s = (cudaStream_t)stream;
    uint32_t t = 256;
    uint32_t b = (numBlocks + t - 1) / t;
    kernelPrecomputeOmegaBases<<<b, t, 0, s>>>((Element*)dBases, omega, blockSize, numBlocks);
    CHECKCUDAERR(cudaGetLastError());
    kernelPrecomputeOmegaTid<<<1, 256, 0, s>>>((Element*)dTid, omega);
    CHECKCUDAERR(cudaGetLastError());
}

// Block-level inclusive scan: 256 threads, 4 elements/thread = 1024 elems/block
__global__ void mulScanBlockKernel(Element* data, Element* blockTotals, uint64_t N)
{
    __shared__ Element sdata[256];
    uint32_t tid = threadIdx.x;
    uint64_t blockStart = (uint64_t)blockIdx.x * 1024;

    // Phase 1: Each thread loads 4 elements and does local inclusive scan
    Element local[4];
    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        local[k] = (gi < N) ? data[gi] : Fr::one();
    }
    local[1] = Fr::mul(local[0], local[1]);
    local[2] = Fr::mul(local[1], local[2]);
    local[3] = Fr::mul(local[2], local[3]);

    // Store per-thread aggregate in shared memory
    sdata[tid] = local[3];
    __syncthreads();

    // Phase 2: Hillis-Steele inclusive scan on 256 aggregates
    for (uint32_t stride = 1; stride < 256; stride <<= 1) {
        Element val = (tid >= stride) ? Fr::mul(sdata[tid - stride], sdata[tid]) : sdata[tid];
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }

    // Save block total
    if (tid == 255 && blockTotals != nullptr) {
        blockTotals[blockIdx.x] = sdata[255];
    }

    // Phase 3: Compute exclusive prefix for this thread from scanned aggregates
    Element threadPrefix = (tid > 0) ? sdata[tid - 1] : Fr::one();

    // Phase 4: Apply prefix to each local element and write back
    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        if (gi < N) {
            data[gi] = Fr::mul(threadPrefix, local[k]);
        }
    }
}

// Propagate: multiply each block's elements by the scanned block prefix
__global__ void mulScanPropagateKernel(Element* data, const Element* blockPrefixes, uint64_t N)
{
    uint64_t blockStart = (uint64_t)blockIdx.x * 1024;
    uint32_t tid = threadIdx.x;
    Element prefix = blockPrefixes[blockIdx.x];
    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        if (gi < N) {
            data[gi] = Fr::mul(prefix, data[gi]);
        }
    }
}

// Rotate left by 1: dst[0] = src[N-1], dst[i] = src[i-1] for i > 0
__global__ void rotateLeftKernel(Element* dst, const Element* src, uint64_t N)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[(i == 0) ? N - 1 : i - 1];
}

// Recursive scan — operates in-place on dData
static void mulScanRecursive(Element* dData, uint64_t N, Element* dWork)
{
    if (N <= 1) return;
    uint32_t numBlocks = (uint32_t)((N + 1023) / 1024);

    if (numBlocks == 1) {
        mulScanBlockKernel<<<1, 256>>>(dData, nullptr, N);
        CHECKCUDAERR(cudaGetLastError());
        return;
    }

    mulScanBlockKernel<<<numBlocks, 256>>>(dData, dWork, N);
    CHECKCUDAERR(cudaGetLastError());

    Element* nextWork = dWork + numBlocks;
    mulScanRecursive(dWork, numBlocks, nextWork);

    mulScanPropagateKernel<<<numBlocks - 1, 256>>>(dData + 1024, dWork, N - 1024);
    CHECKCUDAERR(cudaGetLastError());
    
}

extern "C" void gpu_plonk_prefix_scan_multiply(void* dData, uint64_t N, void* dWork)
{
    mulScanRecursive((Element*)dData, N, (Element*)dWork);
}

extern "C" void gpu_plonk_rotate_left(void* dst, const void* src, uint64_t N)
{
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((N + threads - 1) / threads);
    rotateLeftKernel<<<blocks, threads>>>((Element*)dst, (const Element*)src, N);
    CHECKCUDAERR(cudaGetLastError());
}

// ============================================================================
// GPU Parallel Prefix Scan — Affine (for divByZerofier)
// ============================================================================

struct AffinePair {
    Element a;
    Element b;
};

__device__ __forceinline__ AffinePair affineCompose(const AffinePair& f1, const AffinePair& f2) {
    AffinePair r;
    r.a = Fr::mul(f2.a, f1.a);
    r.b = Fr::add(Fr::mul(f2.a, f1.b), f2.b);
    return r;
}

__device__ __forceinline__ AffinePair affineIdentity() {
    AffinePair r;
    r.a = Fr::one();
    r.b = Fr::zero();
    return r;
}

__global__ void affineScanBlockKernel(AffinePair* pairs, AffinePair* blockTotals, uint64_t N)
{
    __shared__ AffinePair sdata[256];
    uint32_t tid = threadIdx.x;
    uint64_t blockStart = (uint64_t)blockIdx.x * 1024;

    AffinePair local[4];
    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        local[k] = (gi < N) ? pairs[gi] : affineIdentity();
    }
    local[1] = affineCompose(local[0], local[1]);
    local[2] = affineCompose(local[1], local[2]);
    local[3] = affineCompose(local[2], local[3]);

    sdata[tid] = local[3];
    __syncthreads();

    for (uint32_t stride = 1; stride < 256; stride <<= 1) {
        AffinePair val = (tid >= stride) ? affineCompose(sdata[tid - stride], sdata[tid]) : sdata[tid];
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }

    if (tid == 255 && blockTotals != nullptr) {
        blockTotals[blockIdx.x] = sdata[255];
    }

    AffinePair threadPrefix = (tid > 0) ? sdata[tid - 1] : affineIdentity();

    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        if (gi < N) {
            pairs[gi] = affineCompose(threadPrefix, local[k]);
        }
    }
}

__global__ void affineScanPropagateKernel(AffinePair* pairs, const AffinePair* blockPrefixes, uint64_t N)
{
    uint64_t blockStart = (uint64_t)blockIdx.x * 1024;
    uint32_t tid = threadIdx.x;
    AffinePair prefix = blockPrefixes[blockIdx.x];
    for (int k = 0; k < 4; k++) {
        uint64_t gi = blockStart + tid * 4 + k;
        if (gi < N) {
            pairs[gi] = affineCompose(prefix, pairs[gi]);
        }
    }
}

static void affineScanRecursive(AffinePair* dPairs, uint64_t N, AffinePair* dWork)
{
    if (N <= 1) return;
    uint32_t numBlocks = (uint32_t)((N + 1023) / 1024);

    if (numBlocks == 1) {
        affineScanBlockKernel<<<1, 256>>>(dPairs, nullptr, N);
        CHECKCUDAERR(cudaGetLastError());
        return;
    }

    affineScanBlockKernel<<<numBlocks, 256>>>(dPairs, dWork, N);
    CHECKCUDAERR(cudaGetLastError());

    AffinePair* nextWork = dWork + numBlocks;
    affineScanRecursive(dWork, numBlocks, nextWork);

    if (numBlocks > 1) {
        affineScanPropagateKernel<<<numBlocks - 1, 256>>>(dPairs + 1024, dWork, N - 1024);
        CHECKCUDAERR(cudaGetLastError());
    }
}

__global__ void buildAffinePairsKernel(AffinePair* pairs, const Element* coefs, Element alpha, uint64_t numPairs)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPairs) return;
    AffinePair p;
    p.a = alpha;
    p.b = Fr::mul(Fr::sub(Fr::zero(), alpha), coefs[i + 1]);
    pairs[i] = p;
}

__global__ void applyAffineScanKernel(Element* coefs, const AffinePair* scannedPairs, Element y0, uint64_t numPairs)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPairs) return;
    coefs[i + 1] = Fr::add(Fr::mul(scannedPairs[i].a, y0), scannedPairs[i].b);
}

extern "C" void gpu_plonk_compute_div_zerofier(
    void* dCoefs, uint64_t length,
    const void* alphaPtr, const void* y0Ptr,
    void* dPairWork)
{
    Element alpha = *(const Element*)alphaPtr;
    Element y0    = *(const Element*)y0Ptr;
    uint64_t numPairs = length - 1;
    AffinePair* dPairs = (AffinePair*)dPairWork;

    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((numPairs + threads - 1) / threads);
    buildAffinePairsKernel<<<blocks, threads>>>(dPairs, (Element*)dCoefs, alpha, numPairs);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    AffinePair* dRecursiveWork = dPairs + numPairs;
    affineScanRecursive(dPairs, numPairs, dRecursiveWork);
    CHECKCUDAERR(cudaDeviceSynchronize());

    applyAffineScanKernel<<<blocks, threads>>>((Element*)dCoefs, dPairs, y0, numPairs);
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaMemcpy(dCoefs, &y0, sizeof(Element), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaDeviceSynchronize());
}

// ============================================================================
// GPU divZh + T+Tz addition (combined kernel)
// ============================================================================
// Each thread handles one column j across all 4 chunks.
// divZh on T only (negate chunk 0, sequential subtraction), then add Tz to result.
// This matches the original CPU order: divZh(T) + Tz
__global__ void divZhAddKernel(Element* T, const Element* Tz, uint64_t N)
{
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // Load T chunks (without Tz)
    Element t0 = T[j];
    Element t1 = T[N + j];
    Element t2 = T[2*N + j];
    Element t3 = T[3*N + j];

    // divZh on T only: negate first chunk, then sequential subtraction
    Element c0 = Fr::sub(Fr::zero(), t0);
    Element c1 = Fr::sub(c0, t1);
    Element c2 = Fr::sub(c1, t2);
    Element c3 = Fr::sub(c2, t3);

    // Add Tz to the divZh result
    T[j]       = Fr::add(c0, Tz[j]);
    T[N + j]   = Fr::add(c1, Tz[N + j]);
    T[2*N + j] = Fr::add(c2, Tz[2*N + j]);
    T[3*N + j] = Fr::add(c3, Tz[3*N + j]);
}

extern "C" void gpu_plonk_divzh_add(void* dT, const void* dTz, uint64_t N)
{
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((N + threads - 1) / threads);
    divZhAddKernel<<<blocks, threads>>>((Element*)dT, (const Element*)dTz, N);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// ============================================================================
// GPU T split + blinding kernel
// ============================================================================
// Splits combined T polynomial (3N+6 coefficients) into T1, T2, T3 with blinding:
//   T1[0..N-1] = T[0..N-1], T1[N] = bf10                  (N+1 elements)
//   T2[0..N-1] = T[N..2N-1], T2[0] -= bf10, T2[N] = bf11  (N+1 elements)
//   T3[0..N+5] = T[2N..3N+5], T3[0] -= bf11               (N+6 elements)
__global__ void splitTBlindingKernel(
    Element* T1, Element* T2, Element* T3,
    const Element* Tcombined, Element bf10, Element bf11, uint64_t N)
{
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j > N + 5) return;

    // T1: N+1 elements
    if (j <= N) {
        T1[j] = (j < N) ? Tcombined[j] : bf10;
    }

    // T2: N+1 elements
    if (j <= N) {
        Element val = (j < N) ? Tcombined[N + j] : bf11;
        T2[j] = (j == 0) ? Fr::sub(val, bf10) : val;
    }

    // T3: N+6 elements (j can be up to N+5)
    {
        Element val = Tcombined[2*N + j];
        T3[j] = (j == 0) ? Fr::sub(val, bf11) : val;
    }
}

extern "C" void gpu_plonk_split_t_blinding(
    void* dT1, void* dT2, void* dT3,
    const void* dTcombined,
    const void* bf10Ptr, const void* bf11Ptr, uint64_t N)
{
    Element bf10 = *(const Element*)bf10Ptr;
    Element bf11 = *(const Element*)bf11Ptr;
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((N + 6 + threads - 1) / threads);
    splitTBlindingKernel<<<blocks, threads>>>(
        (Element*)dT1, (Element*)dT2, (Element*)dT3,
        (const Element*)dTcombined, bf10, bf11, N);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// Fused gather+z_ratios — reads witness+maps from GPU, uses precomputed omega tables
__global__ void kernelGatherZRatios(
    Element* __restrict__ ratioOut,
    const uint32_t* __restrict__ mapA,
    const uint32_t* __restrict__ mapB,
    const uint32_t* __restrict__ mapC,
    const Element* __restrict__ witness,
    const Element* __restrict__ intWitness,
    uint32_t nDirect,
    uint64_t nConstraints,
    const Element* __restrict__ sigma1,
    const Element* __restrict__ sigma2,
    const Element* __restrict__ sigma3,
    uint32_t sigmaStride,
    Element beta,
    Element gamma,
    Element k1,
    Element k2,
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    uint64_t N)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Element a, b, c;
    if (i < nConstraints) {
        // Inline gather + Montgomery for wire A
        uint32_t idxA = mapA[i];
        a = (idxA < nDirect) ? witness[idxA] : intWitness[idxA - nDirect];
        Fr::toMontgomery(a);

        // Inline gather + Montgomery for wire B
        uint32_t idxB = mapB[i];
        b = (idxB < nDirect) ? witness[idxB] : intWitness[idxB - nDirect];
        Fr::toMontgomery(b);

        // Inline gather + Montgomery for wire C
        uint32_t idxC = mapC[i];
        c = (idxC < nDirect) ? witness[idxC] : intWitness[idxC - nDirect];
        Fr::toMontgomery(c);
    } else {
        // Zero-pad region
        a = Fr::zero();
        b = Fr::zero();
        c = Fr::zero();
    }

    // omega^i from precomputed tables: omega^(blockIdx*256) * omega^(threadIdx)
    Element omega_i = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element betaw = Fr::mul(beta, omega_i);
    
    // Z(X) := numArr / denArr
    // numArr := (a + beta·ω + gamma)(b + beta·ω·k1 + gamma)(c + beta·ω·k2 + gamma)
    Element num1 = Fr::add(Fr::add(a, betaw), gamma);
    Element num2 = Fr::add(Fr::add(b, Fr::mul(k1, betaw)), gamma);
    Element num3 = Fr::add(Fr::add(c, Fr::mul(k2, betaw)), gamma);
    Element num = Fr::mul(num1, Fr::mul(num2, num3));

    // denArr := (a + beta·sigma1 + gamma)(b + beta·sigma2 + gamma)(c + beta·sigma3 + gamma)
    Element den1 = Fr::add(Fr::add(a, Fr::mul(beta, sigma1[i * sigmaStride])), gamma);
    Element den2 = Fr::add(Fr::add(b, Fr::mul(beta, sigma2[i * sigmaStride])), gamma);
    Element den3 = Fr::add(Fr::add(c, Fr::mul(beta, sigma3[i * sigmaStride])), gamma);
    Element den = Fr::mul(den1, Fr::mul(den2, den3));

    ratioOut[i] = Fr::mul(num, Fr::reciprocal(den));
}

extern "C" void gpu_plonk_compute_z_ratios_gather(
    void* ratioOut,
    const void* mapA, const void* mapB, const void* mapC,
    const void* witness, const void* intWitness,
    uint32_t nDirect, uint64_t nConstraints,
    const void* dStaticEvals,
    const void* betaPtr, const void* gammaPtr,
    const void* k1Ptr, const void* k2Ptr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    Element beta  = *(const Element*)betaPtr;
    Element gamma = *(const Element*)gammaPtr;
    Element k1    = *(const Element*)k1Ptr;
    Element k2    = *(const Element*)k2Ptr;

    uint64_t NExt = 4 * N;
    const Element* dStaticBase = (const Element*)dStaticEvals;
    const Element* dS1 = dStaticBase + 0 * NExt;
    const Element* dS2 = dStaticBase + 1 * NExt;
    const Element* dS3 = dStaticBase + 2 * NExt;

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)((N + threadsPerBlock - 1) / threadsPerBlock);

    kernelGatherZRatios<<<blocks, threadsPerBlock>>>(
        (Element*)ratioOut,
        (const uint32_t*)mapA, (const uint32_t*)mapB, (const uint32_t*)mapC,
        (const Element*)witness, (const Element*)intWitness,
        nDirect, nConstraints,
        dS1, dS2, dS3, 4,
        beta, gamma, k1, k2,
        (const Element*)omegaBases, (const Element*)omegaTid,
        N);

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// ============================================================================
// GPU computeR+Wxi kernel — replaces fused CPU loop
// ============================================================================
// Computes Wxi polynomial coefficients directly on GPU from 15 input polynomials.
// All inputs are GPU-resident: A/B/C/Z at dPolCoef, T1/T2/T3 at dT slots,
// QM/QL/QR/QO/QC/S1/S2/S3 uploaded to repurposed d_piBuffer/d_lagBuffer.

struct RWxiConst {
    Element coef_ab, eval_a, eval_b, eval_c;
    Element e2_plus_e4, e3_beta;
    Element v1, v2, v3, v4, v5;
    Element neg_zh, xin, xin2;
    Element r0, wxi_offset;
    // Blinding correction: blindCoefficients modifies CPU poly coefs but GPU dPolCoef has unblinded IFFT data.
    // blindDelta[j] = v1*bfA[j] + v2*bfB[j] + v3*bfC[j] + e2_plus_e4*bfZ[j]
    // Subtracted at i=j (j=0,1,2), added at i=N+j.
    Element blindDelta[3];
    uint64_t N;
};

__global__ void computeRWxiKernel(
    Element* __restrict__ wxi,
    const Element* __restrict__ polA, const Element* __restrict__ polB,
    const Element* __restrict__ polC, const Element* __restrict__ polZ,
    const Element* __restrict__ polQM, const Element* __restrict__ polQL,
    const Element* __restrict__ polQR, const Element* __restrict__ polQO,
    const Element* __restrict__ polQC,
    const Element* __restrict__ polS1, const Element* __restrict__ polS2,
    const Element* __restrict__ polS3,
    const Element* __restrict__ polT1, const Element* __restrict__ polT2,
    const Element* __restrict__ polT3,
    RWxiConst c)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N + 6) return;

    Element val = Fr::zero();

    // Phase 1: zkey selectors + sigmas (i < N)
    if (i < c.N) {
        val = Fr::mul(polQM[i], c.coef_ab);
        val = Fr::add(val, Fr::mul(polQL[i], c.eval_a));
        val = Fr::add(val, Fr::mul(polQR[i], c.eval_b));
        val = Fr::add(val, Fr::mul(polQO[i], c.eval_c));
        val = Fr::add(val, polQC[i]);
        val = Fr::sub(val, Fr::mul(polS3[i], c.e3_beta));
        val = Fr::add(val, Fr::mul(polS1[i], c.v4));
        val = Fr::add(val, Fr::mul(polS2[i], c.v5));
    }

    // Phase 2: Z polynomial (i < N — GPU has unblinded IFFT data)
    if (i < c.N) {
        val = Fr::add(val, Fr::mul(polZ[i], c.e2_plus_e4));
    }

    // Phase 3: Wire polynomials (i < N — GPU has unblinded IFFT data)
    if (i < c.N) {
        val = Fr::add(val, Fr::mul(polA[i], c.v1));
        val = Fr::add(val, Fr::mul(polB[i], c.v2));
        val = Fr::add(val, Fr::mul(polC[i], c.v3));
    }

    // Phase 3b: Blinding corrections for A/B/C/Z
    // CPU blindCoefficients subtracts bf at low indices and adds bf at high indices.
    // blindDelta[j] = combined correction = v1*bfA[j] + v2*bfB[j] + v3*bfC[j] + e2_plus_e4*bfZ[j]
    if (i < 3) {
        val = Fr::sub(val, c.blindDelta[i]);
    }
    if (i >= c.N && i < c.N + 3) {
        val = Fr::add(val, c.blindDelta[i - c.N]);
    }

    // Phase 4: Quotient polynomial combination
    // T3 has N+6 elements, T1/T2 have N+1 elements each
    Element tval = Fr::mul(polT3[i], c.xin2);
    if (i <= c.N) {
        tval = Fr::add(tval, polT1[i]);
        tval = Fr::add(tval, Fr::mul(polT2[i], c.xin));
    }
    val = Fr::add(val, Fr::mul(tval, c.neg_zh));

    // Phase 5: Scalar adjustments at index 0
    if (i == 0) {
        val = Fr::add(val, c.r0);
        val = Fr::sub(val, c.wxi_offset);
    }

    wxi[i] = val;
}

extern "C" void gpu_plonk_compute_r_wxi(
    void* wxi,
    const void* polA, const void* polB, const void* polC, const void* polZ,
    const void* polQM, const void* polQL, const void* polQR, const void* polQO,
    const void* polQC,
    const void* polS1, const void* polS2, const void* polS3,
    const void* polT1, const void* polT2, const void* polT3,
    const void* constants, uint64_t N)
{
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((N + 6 + threads - 1) / threads);
    RWxiConst c = *(const RWxiConst*)constants;
    computeRWxiKernel<<<blocks, threads>>>(
        (Element*)wxi,
        (const Element*)polA, (const Element*)polB, (const Element*)polC, (const Element*)polZ,
        (const Element*)polQM, (const Element*)polQL, (const Element*)polQR, (const Element*)polQO,
        (const Element*)polQC,
        (const Element*)polS1, (const Element*)polS2, (const Element*)polS3,
        (const Element*)polT1, (const Element*)polT2, (const Element*)polT3,
        c);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
}

__global__ void kernelGatherWitness(
    Element* __restrict__ evalOut,
    const uint32_t* __restrict__ mapBuffer,
    const Element* __restrict__ witness,
    const Element* __restrict__ intWitness,
    uint32_t nDirect,
    uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t idx = mapBuffer[i];
    Element val;
    if (idx < nDirect) {
        val = witness[idx];
    } else {
        val = intWitness[idx - nDirect];
    }
    // Convert normal form → Montgomery form
    Fr::toMontgomery(val);
    evalOut[i] = val;
}

extern "C" void gpu_plonk_gather_witness(
    void* evalOut, const void* mapBuffer,
    const void* witness, const void* intWitness,
    uint32_t nDirect, uint64_t nConstraints, uint64_t N)
{
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((nConstraints + threads - 1) / threads);
    kernelGatherWitness<<<blocks, threads>>>(
        (Element*)evalOut,
        (const uint32_t*)mapBuffer,
        (const Element*)witness,
        (const Element*)intWitness,
        nDirect, nConstraints);
    CHECKCUDAERR(cudaGetLastError());

    // Zero-pad evalOut[nConstraints..N) for IFFT
    if (nConstraints < N) {
        size_t padBytes = (N - nConstraints) * sizeof(Element);
        CHECKCUDAERR(cudaMemset((Element*)evalOut + nConstraints, 0, padBytes));
    }
    // This is necessary becasue sppark's IFFT is lanuched into specific CUDA stream, and we need to ensure the gather+pad completes before IFFT starts.
    CHECKCUDAERR(cudaDeviceSynchronize());
}

// ============================================================================
// Phase 10: GPU polynomial evaluation at a single point
// ============================================================================
// Evaluates P(x) = sum_{i=0}^{N-1} coef[i] * x^i using parallel monomial evaluation.
// Phase 1: Each thread computes coef[i]*x^i, block-level reduction → per-block partial sum.
// Phase 2: Iterative reduction of block partial sums → single result.

__global__ void kernelPolyEval(
    Element* __restrict__ blockResults,
    const Element* __restrict__ coefs,
    Element point,
    uint64_t N)
{
    __shared__ Element sdata[256];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        Element xi = Fr::pow(point, (uint32_t)i);
        sdata[threadIdx.x] = Fr::mul(coefs[i], xi);
    } else {
        sdata[threadIdx.x] = Fr::zero();
    }
    __syncthreads();

    // Block-level tree reduction
    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = Fr::add(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        blockResults[blockIdx.x] = sdata[0];
    }
}

// Reduction kernel: sums data[0..count) → data[0..numBlocks)
__global__ void kernelReduceSum(
    Element* __restrict__ data,
    uint64_t count)
{
    __shared__ Element sdata[256];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < count) ? data[i] : Fr::zero();
    __syncthreads();

    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = Fr::add(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

// Evaluate polynomial on GPU, return result to host.
// hostResult: HOST pointer for single FrElement output
// coefs: DEVICE pointer to N coefficient FrElements (Montgomery form)
// pointPtr: HOST pointer to single FrElement (evaluation point, Montgomery form)
// N: number of coefficients
// dWork: device scratch buffer, must hold >= ceil(N/256) FrElements
extern "C" void gpu_plonk_poly_eval_to_host(
    void* hostResult,
    const void* coefs,
    const void* pointPtr,
    uint64_t N,
    void* dWork)
{
    Element point = *(const Element*)pointPtr;
    uint32_t threads = 256;

    // Phase 1: N elements → numBlocks partial sums
    uint32_t numBlocks = (uint32_t)((N + threads - 1) / threads);
    Element* work = (Element*)dWork;

    kernelPolyEval<<<numBlocks, threads>>>(
        work, (const Element*)coefs, point, N);
    CHECKCUDAERR(cudaGetLastError());

    // Phase 2+: Iterative reduction until 1 element remains
    uint64_t count = numBlocks;
    while (count > 1) {
        uint32_t nb = (uint32_t)((count + threads - 1) / threads);
        kernelReduceSum<<<nb, threads>>>(work, count);
        CHECKCUDAERR(cudaGetLastError());
        count = nb;
    }

    // D2H single result element (32 bytes)
    CHECKCUDAERR(cudaMemcpy(hostResult, work, sizeof(Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaDeviceSynchronize());
}
