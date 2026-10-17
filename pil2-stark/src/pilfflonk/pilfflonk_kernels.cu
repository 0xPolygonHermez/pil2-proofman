// The kernels of pilfflonk's GPU path (pilfflonk_kernels.hpp), in the GPU library only. They are
// elementwise, over BN128 scalars in sppark's Montgomery arithmetic (BN128GPUScalarField), which
// keeps every result fully reduced, as ffiasm does: the same field operation gives the same bytes
// on either.
#include "pilfflonk_cuda.cuh"
#include "pilfflonk_kernels.hpp"

namespace {

// The most polynomials a launch of count_coefficients covers, one per row of blocks.
constexpr uint64_t MAX_ROWS = 65535;

__global__ void transposeWitness(Element *columns, const Element *raw, const uint64_t *positions, uint64_t nRows,
                                 uint64_t nCols) {
    // Row by row within a column, so that the writes are consecutive.
    for (uint64_t t = firstIndex(); t < nRows * nCols; t += stride()) {
        const uint64_t c = t / nRows, row = t % nRows;
        Element value = raw[row * nCols + c];
        Fr::toMontgomery(value);
        columns[positions[c] * nRows + row] = value;
    }
}

__global__ void packShift(Element *out, uint64_t n, const Element *base, const uint64_t *offsets, uint64_t k,
                          uint64_t length, const Element *hBlocks, const Element *hPowers) {
    for (uint64_t i = firstIndex(); i < n; i += stride()) {
        const uint64_t e = i + 1;
        Element value = Fr::mul(hBlocks[e >> 8], hPowers[e & 255]);
        if (base != nullptr) {
            const uint64_t c = i / k;
            if (c < length) {
                value = Fr::add(base[offsets[i % k] + c], value);
            }
        }
        out[i] = value;
    }
}

// One thread per polynomial: the factors of each are few (a polynomial's |O_f| + 1), and the CPU's
// order, which matters when n < nFactors, is kept.
__global__ void blind(Element *base, const uint64_t *offsets, uint64_t nPolys, uint64_t n, const Element *factors,
                      uint64_t nFactors) {
    for (uint64_t t = firstIndex(); t < nPolys; t += stride()) {
        Element *p = base + offsets[t];
        const Element *b = factors + t * nFactors;
        for (uint64_t i = 0; i < nFactors; ++i) {
            p[n + i] = Fr::add(p[n + i], b[i]);
            p[i] = Fr::sub(p[i], b[i]);
        }
    }
}

// One thread per boundary: two coefficients of the piece below it and two of the one above, which
// no other boundary touches (qStride >= 2).
__global__ void blindQBoundaries(Element *base, uint64_t slot, uint64_t qStride, uint64_t nBoundaries,
                                 const Element *factors) {
    for (uint64_t t = firstIndex(); t < nBoundaries; t += stride()) {
        Element *below = base + t * slot + qStride, *above = base + (t + 1) * slot;
        for (uint64_t i = 0; i < 2; ++i) {
            below[i] = factors[2 * t + i];
            above[i] = Fr::sub(above[i], factors[2 * t + i]);
        }
    }
}

// Each row of blocks one polynomial; each thread the highest non-zero index of its coefficients, the
// largest of a warp's then folded into counts with atomicMax. Every thread of a warp reaches the
// shuffle.
__global__ void countCoefficients(unsigned long long *counts, const Element *base, const uint64_t *offsets,
                                  uint64_t nPolys, uint64_t length) {
    for (uint64_t t = blockIdx.y; t < nPolys; t += gridDim.y) {
        const Element *p = base + offsets[t];
        unsigned long long top = 0;
        for (uint64_t i = firstIndex(); i < length; i += stride()) {
            if (!isZero(p[i])) {
                top = i + 1;
            }
        }
        for (int lane = 16; lane > 0; lane >>= 1) {
            top = max(top, __shfl_down_sync(0xffffffffu, top, lane));
        }
        if ((threadIdx.x & 31) == 0 && top != 0) {
            atomicMax(&counts[t], top);
        }
    }
}

} // namespace

extern "C" void pilfflonk_gpu_transpose_witness(void *columns, const void *raw, const uint64_t *positions,
                                                uint64_t nRows, uint64_t nCols) {
    if (nRows == 0 || nCols == 0) {
        return;
    }
    transposeWitness<<<blocksFor(nRows * nCols), THREADS>>>(static_cast<Element *>(columns),
                                                             static_cast<const Element *>(raw), positions, nRows,
                                                             nCols);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_pack_shift(void *out, uint64_t n, const void *base, const uint64_t *offsets, uint64_t k,
                                         uint64_t length, const void *hBlocks, const void *hPowers) {
    if (n == 0) {
        return;
    }
    packShift<<<blocksFor(n), THREADS>>>(static_cast<Element *>(out), n, static_cast<const Element *>(base), offsets, k,
                                         length, static_cast<const Element *>(hBlocks),
                                         static_cast<const Element *>(hPowers));
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_blind(void *base, const uint64_t *offsets, uint64_t nPolys, uint64_t n,
                                    const void *factors, uint64_t nFactors) {
    if (nPolys == 0 || nFactors == 0) {
        return;
    }
    blind<<<blocksFor(nPolys), THREADS>>>(static_cast<Element *>(base), offsets, nPolys, n,
                                          static_cast<const Element *>(factors), nFactors);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_blind_q_boundaries(void *base, uint64_t slot, uint64_t qStride, uint64_t nPieces,
                                                 const void *factors) {
    if (nPieces < 2) {
        return;
    }
    blindQBoundaries<<<blocksFor(nPieces - 1), THREADS>>>(static_cast<Element *>(base), slot, qStride, nPieces - 1,
                                                          static_cast<const Element *>(factors));
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_count_coefficients(uint64_t *counts, const void *base, const uint64_t *offsets,
                                                 uint64_t nPolys, uint64_t length) {
    if (nPolys == 0) {
        return;
    }
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "atomicMax counts in 64 bits");
    CHECKCUDAERR(cudaMemsetAsync(counts, 0, nPolys * sizeof(uint64_t), 0));
    if (length == 0) {
        return;
    }
    // A row of blocks per polynomial, up to MAX_ROWS rows, and up to 2^16 blocks in all.
    const uint64_t rows = std::min(nPolys, MAX_ROWS);
    const uint64_t perRow = std::max<uint64_t>(1, std::min<uint64_t>(blocksFor(length), (uint64_t(1) << 16) / rows));
    const dim3 grid(static_cast<uint32_t>(perRow), static_cast<uint32_t>(rows));
    countCoefficients<<<grid, THREADS>>>(reinterpret_cast<unsigned long long *>(counts),
                                         static_cast<const Element *>(base), offsets, nPolys, length);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_memory(uint64_t *freeBytes, uint64_t *totalBytes) {
    size_t free = 0, total = 0;
    CHECKCUDAERR(cudaMemGetInfo(&free, &total));
    *freeBytes = free;
    *totalBytes = total;
}

extern "C" uint32_t pilfflonk_gpu_multiprocessors() {
    int device = 0, count = 0;
    CHECKCUDAERR(cudaGetDevice(&device));
    CHECKCUDAERR(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return static_cast<uint32_t>(count);
}

extern "C" int pilfflonk_gpu_current_device() {
    int device = 0;
    CHECKCUDAERR(cudaGetDevice(&device));
    return device;
}

extern "C" void *pilfflonk_gpu_event_create() {
    cudaEvent_t event;
    CHECKCUDAERR(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

extern "C" void pilfflonk_gpu_event_destroy(void *event) {
    if (event != nullptr) {
        CHECKCUDAERR(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
    }
}

extern "C" void pilfflonk_gpu_event_record(void *event, void *stream) {
    CHECKCUDAERR(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
}

extern "C" void pilfflonk_gpu_event_sync(void *event) {
    CHECKCUDAERR(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
}

extern "C" void pilfflonk_gpu_stream_wait_event(void *stream, void *event) {
    CHECKCUDAERR(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0));
}

extern "C" void pilfflonk_gpu_memcpy_d2h_async(void *hostDst, const void *src, size_t bytes, void *stream) {
    CHECKCUDAERR(cudaMemcpyAsync(hostDst, src, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
}

extern "C" void pilfflonk_gpu_memset_zero(void *dst, size_t bytes) {
    if (bytes != 0) {
        CHECKCUDAERR(cudaMemsetAsync(dst, 0, bytes, 0));
    }
}
