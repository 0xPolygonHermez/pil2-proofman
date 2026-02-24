
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
    const Element* __restrict__ lagrange,
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

// Simplified PI computation for nPublic=1 when L1 evaluations are already on GPU (in dLag).
extern "C" void gpu_plonk_compute_pi_single(
    void* dPI, const void* dLag,
    const void* publicVal, uint64_t NExt)
{
    size_t fullBytes = NExt * sizeof(Element);

    CHECKCUDAERR(cudaMemset(dPI, 0, fullBytes));

    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    computePIAccumulateKernel<<<blocks, threadsPerBlock>>>(
        (Element*)dPI, (const Element*)dLag, *((const Element*)publicVal), NExt);

    CHECKCUDAERR(cudaGetLastError());
}



__device__ __forceinline__
void mulz_mul2_scalar(Element& r, Element& rz,
               const Element& a, const Element& b,
               const Element& ap, const Element& bp,
               const Element& z1p)
{
    Element a_b   = Fr::mul(a, b);
    Element a_bp  = Fr::mul(a, bp);
    Element ap_b  = Fr::mul(ap, b);
    Element ap_bp = Fr::mul(ap, bp);

    r = a_b;

    Element a0 = Fr::add(a_bp, ap_b);
    rz = Fr::add(a0, Fr::mul(z1p, ap_bp));
}

__device__ __forceinline__
void mulz_mul4_scalar(Element& r, Element& rz,
               const Element& a, const Element& b,
               const Element& c, const Element& d,
               const Element& ap, const Element& bp,
               const Element& cp, const Element& dp,
               const Element& z1p, const Element& z2p, const Element& z3p)
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
    rz = Fr::add(rz, Fr::mul(z1p, a1));
    rz = Fr::add(rz, Fr::mul(z2p, a2));
    rz = Fr::add(rz, Fr::mul(z3p, a3));
}


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

extern "C" void gpu_plonk_cuda_free(void* dBuffer)
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
}

extern "C" void gpu_plonk_zero_pad_async(void* buf, uint64_t startElem, uint64_t endElem, void* stream)
{
    uint64_t count = endElem - startElem;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)((count + threadsPerBlock - 1) / threadsPerBlock);
    zeroPadKernel<<<blocks, threadsPerBlock, 0, (cudaStream_t)stream>>>((Element*)buf, startElem, count);
    CHECKCUDAERR(cudaGetLastError());
}

// Gate A kernel: T = a*QL + PI,  Tz = ap*QL
// PI is pre-loaded in tOut; this kernel OVERWRITES tOut with the result.
// ap(i) = bf[2] + bf[1]*omega^i  (derivative of a's blind polynomial)
__global__ void kernelGateA(
    Element* __restrict__ tOut,              // IN: PI(X), OUT: T accumulator
    Element* __restrict__ tzOut,             // OUT: Tz accumulator (first write)
    const Element* __restrict__ evalA,
    const Element* __restrict__ evalQL,
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    const Element* __restrict__ d_blindings, uint64_t NExt)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NExt) return;

    Element a  = evalA[i];
    Element ql = evalQL[i];
    Element pi = tOut[i];    // read PI before overwriting

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element ap = Fr::add(d_blindings[2], Fr::mul(d_blindings[1], omega));

    tOut[i]  = Fr::add(Fr::mul(a, ql), pi);     // T = a*QL + PI
    tzOut[i] = Fr::mul(ap, ql);                 // Tz = ap*QL
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
    const Element* __restrict__ d_blindings, uint64_t NExt)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NExt) return;

    Element b  = evalB[i];
    Element qr = evalQR[i];

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element bp = Fr::add(d_blindings[4], Fr::mul(d_blindings[3], omega));

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
    const Element* __restrict__ d_blindings, uint64_t NExt)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NExt) return;

    Element cc = evalC[i];
    Element qo = evalQO[i];
    Element qc = evalQC[i];

    Element omega = Fr::mul(omegaBases[blockIdx.x], omegaTid[threadIdx.x]);
    Element cp = Fr::add(d_blindings[6], Fr::mul(d_blindings[5], omega));

    tOut[i]  = Fr::add(Fr::add(tOut[i], Fr::mul(cc, qo)), qc);
    tzOut[i] = Fr::add(tzOut[i], Fr::mul(cp, qo));
}

// QM + Permutation + L1 kernel: computes a*b*QM + (e2 - e3)*alpha + (z-1)*L1*alpha^2
// and all blinding derivatives. Accumulates into T/Tz buffers from gate kernels.
// Uses shared memory for blindings, zvals, and omegaTid to reduce register pressure.
__global__ void kernelQMPermL1(
    Element* __restrict__ tOut,
    Element* __restrict__ tzOut,
    const Element* __restrict__ evalA,
    const Element* __restrict__ evalB,
    const Element* __restrict__ evalC,
    const Element* __restrict__ evalZ,       // 4N+4 elements (wrap-around)
    const Element* __restrict__ evalQM,
    const Element* __restrict__ evalS1,
    const Element* __restrict__ evalS2,
    const Element* __restrict__ evalS3,
    const Element* __restrict__ evalL1,      // L_1 evaluations
    const Element* __restrict__ omegaBases,
    const Element* __restrict__ omegaTid,
    const Element* __restrict__ d_blindings,
    const Element* __restrict__ d_zvals, 
    Element beta, Element gamma, Element alpha, Element alpha2,
    Element k1, Element k2, Element omega1,
    uint64_t NExt)
{
    __shared__ __align__(alignof(Element)) unsigned char smem_raw[(12 + 12 + 256) * sizeof(Element)];
    Element* smem = reinterpret_cast<Element*>(smem_raw);
    Element* sBlindings = smem;
    Element* sZvals     = sBlindings + 12;
    Element* sOmegaTid  = sZvals + 12;

    // Cooperative load: first 12 threads load blindings, next 12 load zvals
    if (threadIdx.x < 12) {
        sBlindings[threadIdx.x] = d_blindings[threadIdx.x];
    }
    if (threadIdx.x >= 12 && threadIdx.x < 24) {
        sZvals[threadIdx.x - 12] = d_zvals[threadIdx.x - 12];
    }
    // All 256 threads load omegaTid
    sOmegaTid[threadIdx.x] = omegaTid[threadIdx.x];
    __syncthreads();

    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NExt) return;

    // Select Z values for this thread's position (i % 4)
    uint32_t p = (uint32_t)(i & 3);
    Element z1p = sZvals[p];       // Z1[p]
    Element z2p = sZvals[4 + p];   // Z2[p]
    Element z3p = sZvals[8 + p];   // Z3[p]

    Element a  = evalA[i];
    Element b  = evalB[i];
    Element cc = evalC[i];
    Element z  = evalZ[i];
    Element zW = evalZ[i + 4];    // wrap-around

    Element qm = evalQM[i];
    Element s1 = evalS1[i];
    Element s2 = evalS2[i];
    Element s3 = evalS3[i];
    Element lagrange = evalL1[i];

    // Compute roots of unity (precomputed base + per-thread table from shared mem)
    Element omega   = Fr::mul(omegaBases[blockIdx.x], sOmegaTid[threadIdx.x]);
    Element omega2  = Fr::square(omega);
    Element omegaW  = Fr::mul(omega, omega1);
    Element omegaW2 = Fr::square(omegaW);

    // Blinding derivatives (read from shared memory)
    Element ap = Fr::add(sBlindings[2], Fr::mul(sBlindings[1], omega));
    Element bp = Fr::add(sBlindings[4], Fr::mul(sBlindings[3], omega));
    Element cp = Fr::add(sBlindings[6], Fr::mul(sBlindings[5], omega));
    Element zp  = Fr::add(Fr::add(Fr::mul(sBlindings[7], omega2), Fr::mul(sBlindings[8], omega)), sBlindings[9]);
    Element zWp = Fr::add(Fr::add(Fr::mul(sBlindings[7], omegaW2), Fr::mul(sBlindings[8], omegaW)), sBlindings[9]);

    // a*b*QM term
    Element abqm, abqmz;
    mulz_mul2_scalar(abqm, abqmz, a, b, ap, bp, z1p);
    abqm  = Fr::mul(abqm, qm);
    abqmz = Fr::mul(abqmz, qm);

    // e2: permutation numerator
    Element betaw = Fr::mul(beta, omega);
    Element e2a = Fr::add(Fr::add(a, betaw), gamma);
    Element e2b = Fr::add(Fr::add(b, Fr::mul(betaw, k1)), gamma);
    Element e2c = Fr::add(Fr::add(cc, Fr::mul(betaw, k2)), gamma);

    Element e2, e2z;
    mulz_mul4_scalar(e2, e2z, e2a, e2b, e2c, z, ap, bp, cp, zp, z1p, z2p, z3p);
    e2  = Fr::mul(e2, alpha);
    e2z = Fr::mul(e2z, alpha);

    // e3: permutation denominator
    Element e3a = Fr::add(Fr::add(a, Fr::mul(beta, s1)), gamma);
    Element e3b = Fr::add(Fr::add(b, Fr::mul(beta, s2)), gamma);
    Element e3c = Fr::add(Fr::add(cc, Fr::mul(beta, s3)), gamma);

    Element e3, e3z;
    mulz_mul4_scalar(e3, e3z, e3a, e3b, e3c, zW, ap, bp, cp, zWp, z1p, z2p, z3p);
    e3  = Fr::mul(e3, alpha);
    e3z = Fr::mul(e3z, alpha);

    // e4: L1 constraint  alpha2 * (z - 1) * L1
    Element e4 = Fr::mul(Fr::mul(Fr::sub(z, Fr::one()), lagrange), alpha2);
    Element e4z = Fr::mul(Fr::mul(zp, lagrange), alpha2);

    // Accumulate: T += a*b*QM + e2 - e3 + e4
    Element contrib  = Fr::add(Fr::sub(Fr::add(abqm, e2), e3), e4);
    Element contribz = Fr::add(Fr::sub(Fr::add(abqmz, e2z), e3z), e4z);

    tOut[i]  = Fr::add(tOut[i], contrib);
    tzOut[i] = Fr::add(tzOut[i], contribz);
}

extern "C" void gpu_plonk_compute_gate_a(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalQL,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateA<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalA, (const Element*)evalQL,
        (const Element*)omegaBases, (const Element*)omegaTid,
        (const Element*)d_blindings, NExt);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_gate_b(
    void* tOut, void* tzOut,
    const void* evalB, const void* evalQR,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateB<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalB, (const Element*)evalQR,
        (const Element*)omegaBases, (const Element*)omegaTid,
        (const Element*)d_blindings, NExt);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_gate_c(
    void* tOut, void* tzOut,
    const void* evalC, const void* evalQO, const void* evalQC,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelGateC<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalC, (const Element*)evalQO, (const Element*)evalQC,
        (const Element*)omegaBases, (const Element*)omegaTid,
        (const Element*)d_blindings, NExt);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void gpu_plonk_compute_qm_perm_l1(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB, const void* evalC,
    const void* evalZ,
    const void* evalQM, const void* evalS1, const void* evalS2, const void* evalS3, const void* evalL1,
    const void* d_blindings, const void* d_zvals,
    const void* betaPtr, const void* gammaPtr,
    const void* alphaPtr, const void* alpha2Ptr,
    const void* k1Ptr, const void* k2Ptr,
    const void* omega1Ptr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid)
{
    uint64_t NExt = 4 * N;
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)(NExt / threadsPerBlock);

    kernelQMPermL1<<<blocks, threadsPerBlock>>>(
        (Element*)tOut, (Element*)tzOut,
        (const Element*)evalA, (const Element*)evalB, (const Element*)evalC,
        (const Element*)evalZ,
        (const Element*)evalQM, (const Element*)evalS1, (const Element*)evalS2,
        (const Element*)evalS3, (const Element*)evalL1,
        (const Element*)omegaBases, (const Element*)omegaTid,
        (const Element*)d_blindings,
        (const Element*)d_zvals,
        *(const Element*)betaPtr, *(const Element*)gammaPtr,
        *(const Element*)alphaPtr, *(const Element*)alpha2Ptr,
        *(const Element*)k1Ptr, *(const Element*)k2Ptr,
        *(const Element*)omega1Ptr,
        NExt);
    CHECKCUDAERR(cudaGetLastError());
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
    __shared__ __align__(alignof(Element)) unsigned char sdata_raw[256 * sizeof(Element)];
    Element* sdata = reinterpret_cast<Element*>(sdata_raw);
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
    __shared__ __align__(alignof(AffinePair)) unsigned char sdata_raw[256 * sizeof(AffinePair)];
    AffinePair* sdata = reinterpret_cast<AffinePair*>(sdata_raw);
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

__global__ void buildAffinePairsKernel(AffinePair* pairs, const Element* coefs, Element invBeta, uint64_t numPairs)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPairs) return;
    AffinePair p;
    p.a = invBeta;
    p.b = Fr::mul(Fr::sub(Fr::zero(), invBeta), coefs[i + 1]);
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
    const void* invBetaPtr, const void* y0Ptr,
    void* dPairWork)
{
    Element invBeta = *(const Element*)invBetaPtr;
    Element y0    = *(const Element*)y0Ptr;
    uint64_t numPairs = length - 1;
    AffinePair* dPairs = (AffinePair*)dPairWork;

    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((numPairs + threads - 1) / threads);
    buildAffinePairsKernel<<<blocks, threads>>>(dPairs, (Element*)dCoefs, invBeta, numPairs);
    CHECKCUDAERR(cudaGetLastError());

    AffinePair* dRecursiveWork = dPairs + numPairs;
    affineScanRecursive(dPairs, numPairs, dRecursiveWork);

    applyAffineScanKernel<<<blocks, threads>>>((Element*)dCoefs, dPairs, y0, numPairs);
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaMemcpy(dCoefs, &y0, sizeof(Element), cudaMemcpyHostToDevice));
}


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
}

// Splits combined T polynomial (3N+6 coefficients) into T1, T2, T3 with blinding:
//   T1[0..N-1] = T[0..N-1], T1[N] = bf[10]                    (N+1 elements)
//   T2[0..N-1] = T[N..2N-1], T2[0] -= bf[10], T2[N] = bf[11]  (N+1 elements)
//   T3[0..N+5] = T[2N..3N+5], T3[0] -= bf[11]                 (N+6 elements)
__global__ void splitTBlindingKernel(
    Element* T1, Element* T2, Element* T3,
    const Element* Tcombined, const Element* __restrict__ d_blindings, uint64_t N)
{
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j > N + 5) return;

    Element bf10 = d_blindings[10];
    Element bf11 = d_blindings[11];

    if (j <= N) {
        T1[j] = (j < N) ? Tcombined[j] : bf10;
    }

    if (j <= N) {
        Element val = (j < N) ? Tcombined[N + j] : bf11;
        T2[j] = (j == 0) ? Fr::sub(val, bf10) : val;
    }

    
    Element val = Tcombined[2*N + j];
    T3[j] = (j == 0) ? Fr::sub(val, bf11) : val;
    
}

extern "C" void gpu_plonk_split_t_blinding(
    void* dT1, void* dT2, void* dT3,
    const void* dTcombined, const void* d_blindings, uint64_t N)
{
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((N + 6 + threads - 1) / threads);
    splitTBlindingKernel<<<blocks, threads>>>(
        (Element*)dT1, (Element*)dT2, (Element*)dT3,
        (const Element*)dTcombined, (const Element*)d_blindings, N);
    CHECKCUDAERR(cudaGetLastError());
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
}


// Computes Wxi polynomial coefficients directly on GPU from 15 input polynomials.
struct RWxiConst {
    Element coef_ab, eval_a, eval_b, eval_c;
    Element e2_plus_e4, e3_beta;
    Element v1, v2, v3, v4, v5;
    Element neg_zh, xin, xin2;
    Element r0, wxi_offset;
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

    
    if (i < c.N) {
        val = Fr::mul(polQM[i], c.coef_ab);
        val = Fr::add(val, Fr::mul(polQL[i], c.eval_a));
        val = Fr::add(val, Fr::mul(polQR[i], c.eval_b));
        val = Fr::add(val, Fr::mul(polQO[i], c.eval_c));
        val = Fr::add(val, polQC[i]);
        val = Fr::sub(val, Fr::mul(polS3[i], c.e3_beta));
        val = Fr::add(val, Fr::mul(polS1[i], c.v4));
        val = Fr::add(val, Fr::mul(polS2[i], c.v5));
        val = Fr::add(val, Fr::mul(polZ[i], c.e2_plus_e4));
        val = Fr::add(val, Fr::mul(polA[i], c.v1));
        val = Fr::add(val, Fr::mul(polB[i], c.v2));
        val = Fr::add(val, Fr::mul(polC[i], c.v3));
    }

    // Phase 3b: Blinding corrections for A/B/C/Z
    if (i < 3) {
        val = Fr::sub(val, c.blindDelta[i]);
    }
    if (i >= c.N && i < c.N + 3) {
        val = Fr::add(val, c.blindDelta[i - c.N]);
    }

    // Quotient polynomial combination
    // T3 has N+6 elements, T1/T2 have N+1 elements each
    Element tval = Fr::mul(polT3[i], c.xin2);
    if (i <= c.N) {
        tval = Fr::add(tval, polT1[i]);
        tval = Fr::add(tval, Fr::mul(polT2[i], c.xin));
    }
    val = Fr::add(val, Fr::mul(tval, c.neg_zh));

    // Scalar adjustments at index 0
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
}


// Evaluates P(x) = sum_{i=0}^{N-1} coef[i] * x^i using parallel monomial evaluation.
__global__ void kernelPolyEval(
    Element* __restrict__ blockResults,
    const Element* __restrict__ coefs,
    Element point,
    uint64_t N)
{
    __shared__ __align__(alignof(Element)) unsigned char sdata_raw[256 * sizeof(Element)];
    Element* sdata = reinterpret_cast<Element*>(sdata_raw);
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

// Reduction kernel: sums dataIn[0..count) → dataOut[0..numBlocks)
// dataIn and dataOut must not overlap.
__global__ void kernelReduceSum(
    const Element* __restrict__ dataIn,
    Element* __restrict__ dataOut,
    uint64_t count)
{
    __shared__ __align__(alignof(Element)) unsigned char sdata_raw[256 * sizeof(Element)];
    Element* sdata = reinterpret_cast<Element*>(sdata_raw);
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < count) ? dataIn[i] : Fr::zero();
    __syncthreads();

    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = Fr::add(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dataOut[blockIdx.x] = sdata[0];
    }
}

// Evaluate polynomial on GPU, return result to host.
// dWork: device scratch buffer, must hold >= sum_{k=1..L} ceil(N/256^k) FrElements (L = ceil(log256(N)))
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

    // Phase 2+: Iterative reduction with non-overlapping read/write regions
    uint64_t count = numBlocks;
    Element* readPtr = work;
    Element* writePtr = work + count;
    while (count > 1) {
        uint32_t nb = (uint32_t)((count + threads - 1) / threads);
        kernelReduceSum<<<nb, threads>>>(readPtr, writePtr, count);
        CHECKCUDAERR(cudaGetLastError());
        readPtr = writePtr;
        writePtr += nb;
        count = nb;
    }

    CHECKCUDAERR(cudaMemcpy(hostResult, readPtr, sizeof(Element), cudaMemcpyDeviceToHost));
}

// Calculating additions (parallel per level)
__global__ void kernelCalculateAdditions(
    Element* __restrict__ buffInternalWitness,
    const Element* __restrict__ buffWitness,
    const uint32_t* __restrict__ signalId1Array,
    const uint32_t* __restrict__ signalId2Array,
    const Element* __restrict__ factor1Array,
    const Element* __restrict__ factor2Array,
    const uint8_t* __restrict__ levels,
    uint8_t currentLevel,
    uint32_t nAdditions,
    uint32_t nDirect)  
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nAdditions) return;
    if (levels[i] != currentLevel) return;

    uint32_t signalId1 = signalId1Array[i];
    uint32_t signalId2 = signalId2Array[i];
    Element factor1 = factor1Array[i];
    Element factor2 = factor2Array[i];

    Element w1 = (signalId1 < nDirect)
        ? buffWitness[signalId1]
        : buffInternalWitness[signalId1 - nDirect];

    Element w2 = (signalId2 < nDirect)
        ? buffWitness[signalId2]
        : buffInternalWitness[signalId2 - nDirect];

    // Compute: result = factor1 * w1 + factor2 * w2
    w1 = Fr::mul(factor1, w1);
    w2 = Fr::mul(factor2, w2);
    buffInternalWitness[i] = Fr::add(w1, w2);
}

extern "C" void gpu_plonk_calculate_additions(
    void* d_buffInternalWitness,
    const void* d_buffWitness,
    const void* d_addSignalId1,
    const void* d_addSignalId2,
    const void* d_addFactor1,
    const void* d_addFactor2,
    const void* d_additionLevels,
    uint8_t maxLevel,
    uint32_t nAdditions,
    uint32_t nDirect)
{
    uint32_t threads = 256;
    uint32_t blocks = (nAdditions + threads - 1) / threads;

    for (uint8_t level = 0; level <= maxLevel; level++) {
        kernelCalculateAdditions<<<blocks, threads>>>(
            (Element*)d_buffInternalWitness,
            (const Element*)d_buffWitness,
            (const uint32_t*)d_addSignalId1,
            (const uint32_t*)d_addSignalId2,
            (const Element*)d_addFactor1,
            (const Element*)d_addFactor2,
            (const uint8_t*)d_additionLevels,
            level,
            nAdditions,
            nDirect
        );
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaDeviceSynchronize());
}

