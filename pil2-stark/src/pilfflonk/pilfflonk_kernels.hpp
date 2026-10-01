#ifndef PILFFLONK_KERNELS_HPP
#define PILFFLONK_KERNELS_HPP

#include <cstddef>
#include <cstdint>

// The kernels of pilfflonk's GPU path (pilfflonk_kernels.cu), and the CUDA runtime calls its host
// side needs besides the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), with C linkage as
// those are: the files that call them are compiled with g++, without the CUDA headers.
//
// Every pointer but a host one named so is device memory. An element is a BN254 scalar of 32
// bytes, in Montgomery form unless said otherwise, as BN128GPUScalarField (bn128/src/ffigpu/fr.cuh)
// and ffiasm keep it; a polynomial is its coefficients in increasing degree. Indices are 64-bit.
// The kernels run on the legacy default stream, in order with each other and with the PLONK
// helpers', and return before they finish. Nothing is launched for an empty range. A CUDA failure
// aborts the process (CHECKCUDAERR), as in the PLONK GPU prover.
extern "C" {

// The witness of nRows rows of nCols canonical scalars, row after row (an Instance's stage1), into
// its columns: columns[positions[c]·nRows + row] is raw[row·nCols + c] in Montgomery form.
void pilfflonk_gpu_transpose_witness(void *columns, const void *raw, const uint64_t *positions, uint64_t nRows,
                                     uint64_t nCols);

// The fflonk packing f(X) = Σ_{j<k} p_j(X^k)·X^j of k polynomials of `length` coefficients each,
// p_j at base + offsets[j] (offsets in elements), shifted for the MSM:
// out[i] = f_i + h^(i+1) for i < n, f_i = p_{i mod k}[⌊i/k⌋] if ⌊i/k⌋ < length and 0 otherwise.
// hBlocks[b] = h^(256·b) for b <= n/256 and hPowers[t] = h^t for t < 256 are the tables of
// gpu_plonk_precompute_omega_tables_async with base h and block size 256. With base null, out[i] is
// h^(i+1) alone, and offsets, k and length are not read.
void pilfflonk_gpu_pack_shift(void *out, uint64_t n, const void *base, const uint64_t *offsets, uint64_t k,
                              uint64_t length, const void *hBlocks, const void *hPowers);

// The blinding p'(X) = p(X) + (X^n − 1)·b(X) of nPolys polynomials, p_t at base + offsets[t] with
// n + nFactors coefficients, and b's nFactors coefficients at factors + t·nFactors: for i < nFactors
// in order, p_t[n + i] += b_i and then p_t[i] −= b_i, as rapidsnark's Polynomial::blindCoefficients.
void pilfflonk_gpu_blind(void *base, const uint64_t *offsets, uint64_t nPolys, uint64_t n, const void *factors,
                         uint64_t nFactors);

// counts[t] = 1 + the index of the highest coefficient of p_t that is not zero, or 0 if every one
// is, for the nPolys polynomials of `length` coefficients, p_t at base + offsets[t]: the degree
// rapidsnark's Polynomial::fixDegree finds is max(counts[t], 1) − 1.
void pilfflonk_gpu_count_coefficients(uint64_t *counts, const void *base, const uint64_t *offsets, uint64_t nPolys,
                                      uint64_t length);

// The device's free and total memory, in bytes (cudaMemGetInfo), and its number of multiprocessors.
void pilfflonk_gpu_memory(uint64_t *freeBytes, uint64_t *totalBytes);
uint32_t pilfflonk_gpu_multiprocessors();

// Events without timing, to order a non-blocking stream with the default one (a null stream).
void *pilfflonk_gpu_event_create();
void pilfflonk_gpu_event_destroy(void *event);
void pilfflonk_gpu_event_record(void *event, void *stream);
void pilfflonk_gpu_event_sync(void *event);
void pilfflonk_gpu_stream_wait_event(void *stream, void *event);

// An asynchronous copy to the host on `stream`, into host memory that is pinned or registered.
void pilfflonk_gpu_memcpy_d2h_async(void *hostDst, const void *src, size_t bytes, void *stream);

// Zeros `bytes` bytes at dst, on the default stream.
void pilfflonk_gpu_memset_zero(void *dst, size_t bytes);

} // extern "C"

#endif
