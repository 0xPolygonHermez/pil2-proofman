
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
    Element den1 = Fr::add(Fr::add(buffA[i], Fr::mul(beta, sigma1[i])), gamma);
    Element den2 = Fr::add(Fr::add(buffB[i], Fr::mul(beta, sigma2[i])), gamma);
    Element den3 = Fr::add(Fr::add(buffC[i], Fr::mul(beta, sigma3[i])), gamma);
    Element den = Fr::mul(den1, Fr::mul(den2, den3));

    // ratio[i] = num * den^{-1}
    ratioOut[i] = Fr::mul(num, Fr::reciprocal(den));
}

extern "C" void compute_z_ratios_gpu(
    void* ratioOut,          
    const void* buffA,       
    const void* buffB,       
    const void* buffC,       
    const void* sigma1Eval,  
    const void* sigma2Eval,  
    const void* sigma3Eval,  
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
    const Element* hSigma1 = (const Element*)sigma1Eval;
    const Element* hSigma2 = (const Element*)sigma2Eval;
    const Element* hSigma3 = (const Element*)sigma3Eval;

    Element beta  = *(const Element*)betaPtr;
    Element gamma = *(const Element*)gammaPtr;
    Element k1    = *(const Element*)k1Ptr;
    Element k2    = *(const Element*)k2Ptr;
    Element omega = *(const Element*)omegaPtr;

    size_t arrayBytes = domainSize * sizeof(Element);

    // Extract sigma values at stride 4 into compact arrays
    Element* hSigma1Compact = (Element*)malloc(arrayBytes);
    Element* hSigma2Compact = (Element*)malloc(arrayBytes);
    Element* hSigma3Compact = (Element*)malloc(arrayBytes);
    for (uint64_t i = 0; i < domainSize; i++) {
        hSigma1Compact[i] = hSigma1[i * 4];
        hSigma2Compact[i] = hSigma2[i * 4];
        hSigma3Compact[i] = hSigma3[i * 4];
    }

    // Allocate device memory
    Element *dRatioOut, *dBuffA, *dBuffB, *dBuffC;
    Element *dSigma1, *dSigma2, *dSigma3;

    CHECKCUDAERR(cudaMalloc(&dRatioOut, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffA, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffB, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dBuffC, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dSigma1, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dSigma2, arrayBytes));
    CHECKCUDAERR(cudaMalloc(&dSigma3, arrayBytes));

    // Copy inputs H2D
    CHECKCUDAERR(cudaMemcpy(dBuffA, hBuffA, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dBuffB, hBuffB, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dBuffC, hBuffC, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dSigma1, hSigma1Compact, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dSigma2, hSigma2Compact, arrayBytes, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dSigma3, hSigma3Compact, arrayBytes, cudaMemcpyHostToDevice));

    free(hSigma1Compact);
    free(hSigma2Compact);
    free(hSigma3Compact);

    // Launch kernel
    uint32_t threadsPerBlock = 256;
    uint32_t blocks = (uint32_t)((domainSize + threadsPerBlock - 1) / threadsPerBlock);

    computeZRatiosKernel<<<blocks, threadsPerBlock>>>(
        dRatioOut, dBuffA, dBuffB, dBuffC,
        dSigma1, dSigma2, dSigma3,
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
    cudaFree(dSigma1);
    cudaFree(dSigma2);
    cudaFree(dSigma3);
}
