#ifndef GATE_BANDS_BLAKE3_GPU_CUH
#define GATE_BANDS_BLAKE3_GPU_CUH

// Device back-end for the BLAKE3 band kinds.
//
// The arithmetic is the SAME code the CPU expander runs: gate_bands_blake3.hpp is one templated
// implementation that both backends instantiate, so the permutation cannot drift between them.
// Only the multiplicity sink differs. A CPU thread owns a private 1.5 MB Multiplicities and
// reduces once at the end; a thread here is a single band among thousands, so it adds atomically.
//
// That sharing is a luxury of BLAKE3's tiny constants and matching shape -- see the note in
// gate_bands_poseidon_gpu.cuh for why Poseidon cannot do the same.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"
#include "gate_bands.hpp"
#include "gate_bands_blake3.hpp"

namespace gate_bands {
namespace blake3_gpu {

// Counters go into a dense scratch buffer, NOT straight into the trace's two multiplicity columns.
// In the trace those columns are strided by nCols, so every atomic would take a cache line of its
// own and share it with witness cells other threads are storing to. Dense, the counters are a
// contiguous 1.5 MB that stays L2-resident, neighbouring counters share a line, and nothing
// false-shares with the trace. The scatter kernel moves them into the columns at the end.
//
// Layout: the table's TABLE_SIZE counters, then the range checker's RANGE_SIZE.
struct AtomicSink {
    uint64_t *mul;  // dense, TABLE_SIZE + RANGE_SIZE words
    __device__ void table(uint64_t i) {
        if (i < gate_bands::blake3::TABLE_SIZE) atomicAdd((unsigned long long *)&mul[i], 1ull);
    }
    __device__ void range(uint64_t i) {
        if (i < gate_bands::blake3::RANGE_SIZE)
            atomicAdd((unsigned long long *)&mul[gate_bands::blake3::TABLE_SIZE + i], 1ull);
    }
};

// SCRATCH_WORDS below is what the caller allocates, via gateBandScratchWordsGPU.

// Writes the dense counters into the trace's two multiplicity columns: one pass, and the only place
// the strided column layout is paid for. Rows past nRows cannot hold a counter, so their counts would
// be lost; the host checks that nRows covers the table, and the guards here keep a short air from
// writing out of bounds.
__global__ void scatterBlake3MultiplicitiesKernel(uint64_t *trace, uint64_t nCols, uint64_t nRows,
                                                  uint64_t lanes, uint64_t band, const uint64_t *mul) {
    const gate_bands::blake3::Layout L = gate_bands::blake3::layout(lanes, band);
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < gate_bands::blake3::TABLE_SIZE && i < nRows) {
        trace[i * nCols + L.mul_table] = mul[i];
    }
    if (i < gate_bands::blake3::RANGE_SIZE && i < nRows) {
        trace[i * nCols + L.mul_range] = mul[gate_bands::blake3::TABLE_SIZE + i];
    }
}

// One thread per (band, lane), the block's bands in lockstep. Each clock's interior cells (one
// contiguous column range, all written) go to a shared tile, then out as coalesced row stores; direct
// per-thread stores were nCols apart and store-bound. Boundary columns and lookups go straight out.
constexpr uint32_t EXPAND_THREADS = 64;

// Interior columns of a row: va .. vb_pp_t over every lane.
__host__ __device__ __forceinline__ uint64_t interior_cols(const gate_bands::blake3::Layout &L, uint64_t lanes) {
    return L.vb_pp_t + lanes - L.va;
}

__global__ void __launch_bounds__(EXPAND_THREADS) expandBlake3BandsKernel(uint64_t *trace, uint64_t nCols,
                                        const uint64_t *bands, uint64_t nBands, uint64_t lanes,
                                        uint64_t band, uint64_t *mul) {
    namespace b3 = gate_bands::blake3;
    extern __shared__ uint16_t expandTile[];   // [bandsPerBlock][width]
    const b3::Layout L = b3::layout(lanes, band);
    const uint64_t col0 = L.va, width = interior_cols(L, lanes);
    const uint32_t perBlock = EXPAND_THREADS / (uint32_t)lanes;
    const uint32_t lb = threadIdx.x / (uint32_t)lanes, lane = threadIdx.x % (uint32_t)lanes;
    const uint64_t b0 = (uint64_t)blockIdx.x * perBlock, b = b0 + lb;
    // Also validated host-side in load_device_setup; an unknown kind must never be filled
    // with a guessed permutation. A non-BLAKE3 kind here would mean the host launched the wrong
    // kernel for this setup, so skip it rather than reconstruct it with the wrong permutation.
    auto valid = [&](uint64_t bb) { return bb < nBands && is_known_kind(bands[bb * 3 + 1]) && is_blake3(bands[bb * 3 + 1]); };
    const bool mine = lb < perBlock && valid(b);
    uint64_t row = 0;
    b3::Kind k = b3::Kind::Parent;
    b3::BlockInputs in{};
    uint32_t v[16];
    if (mine) {
        row = bands[b * 3];
        const uint64_t kind = bands[b * 3 + 1];
        k = kind == GB_BLAKE3_NODE ? b3::Kind::Node : kind == GB_BLAKE3_COMPRESS_CHUNK ? b3::Kind::Chunk : b3::Kind::Parent;
        in = b3::read_boundary(trace, nCols, row + lane, k, bands[b * 3 + 2]);
        in.initial_state(v);
    }
    AtomicSink sink{mul};
    // Interior cells are bytes, bits or 16-bit limbs.
    uint16_t *tile = expandTile + (uint64_t)lb * width;
    auto put = [&](uint64_t, uint64_t col, uint64_t x) { tile[col - col0] = (uint16_t)x; };
    const uint32_t warp = threadIdx.x / 32, wl = threadIdx.x % 32, nWarps = EXPAND_THREADS / 32;
    // Unrolled: r and g must be compile-time, or the state v[16] is indexed dynamically and spills.
    #pragma unroll
    for (int r = 0; r < b3::ROUNDS; r++) {
        #pragma unroll
        for (int g = 0; g < b3::G_PER_ROUND; g++) {
            if (mine) b3::expand_clock(put, row, lane, L, in, sink, v, r, g);
            __syncthreads();
            const uint64_t c = (uint64_t)(r * b3::G_PER_ROUND + g);
            for (uint32_t i = warp; i < perBlock; i += nWarps) {
                if (!valid(b0 + i)) continue;
                uint64_t *dst = trace + (bands[(b0 + i) * 3] + c) * nCols + col0;
                const uint16_t *src = expandTile + (uint64_t)i * width;
                for (uint64_t x = wl; x < width; x += 32) dst[x] = src[x];
            }
            __syncthreads();
        }
    }
    // Reads back round 6's z cells, stored by other threads; the last barrier makes them visible.
    if (mine) b3::expand_boundary_columns(trace, nCols, row, lane, L, k, in, v, sink);
}

inline size_t expandShmBytes(uint64_t lanes, uint64_t band) {
    const gate_bands::blake3::Layout L = gate_bands::blake3::layout(lanes, band);
    return (EXPAND_THREADS / lanes) * interior_cols(L, lanes) * sizeof(uint16_t);
}
// Dense counters, memset ahead of the fill on the same stream. `write_multiplicities` is the host's
// equivalent of the scatter kernel below.
constexpr uint64_t SCRATCH_WORDS = blake3::TABLE_SIZE + blake3::RANGE_SIZE;

// `d_scratch` MUST belong to this stream alone: the three launches below are ordered only within
// `stream`, so a shared buffer lets another stream's memset land inside this one's fill.
inline void launch(uint64_t *d_trace, uint64_t nCols, uint64_t nRows, const uint64_t *d_bands,
                   uint64_t nBands, uint64_t aux, uint64_t *d_scratch, cudaStream_t stream) {
    // LANES low, the a[]/S[] band width high. Both are setup parameters the kernel cannot recover;
    // load_device_setup rejects a zero in either half, so they are trusted here.
    const uint64_t lanes = aux & 0xFFFFFFFFull;
    const uint64_t band = aux >> 32;
    const int tpb = 128;

    // fprintf rather than zklog: zklog.error followed by exitProcess is what made a const-file size
    // mismatch exit 255 with no message anywhere -- not stdout, not stderr, not a file -- and these
    // two conditions are setup bugs whose whole value is being readable.
    if (d_scratch == nullptr) {
        fprintf(stderr, "gate bands: a BLAKE3 air reached the expander with no multiplicity scratch; "
                        "the caller allocates it per STREAM on that stream's first BLAKE3 commit, so "
                        "the two disagree and every lookup count would be dropped\n");
        fflush(stderr);
        exit(-1);
    }
    // The table has to fit the trace, or its tail rows have counts nothing can carry.
    if (nRows < blake3::TABLE_SIZE) {
        fprintf(stderr, "gate bands: nRows %llu cannot hold the %llu-row BLAKE3 table\n",
                (unsigned long long)nRows, (unsigned long long)blake3::TABLE_SIZE);
        fflush(stderr);
        exit(-1);
    }

    // Stream-ordered ahead of the expansion, so every atomicAdd lands on a zeroed counter. Dense,
    // so this is one contiguous memset rather than a strided kernel over trace columns.
    CHECKCUDAERR(cudaMemsetAsync(d_scratch, 0, SCRATCH_WORDS * sizeof(uint64_t), stream));

    if (lanes == 0 || lanes > EXPAND_THREADS) {
        fprintf(stderr, "gate bands: LANES %llu does not fit the BLAKE3 expander's %u-thread block\n",
                (unsigned long long)lanes, EXPAND_THREADS);
        fflush(stderr);
        exit(-1);
    }
    const uint64_t perBlock = EXPAND_THREADS / lanes;
    expandBlake3BandsKernel<<<(unsigned)((nBands + perBlock - 1) / perBlock), EXPAND_THREADS, expandShmBytes(lanes, band), stream>>>(
        d_trace, nCols, d_bands, nBands, lanes, band, d_scratch);
    CHECKCUDAERR(cudaGetLastError());

    const uint64_t nScatter = blake3::TABLE_SIZE;
    scatterBlake3MultiplicitiesKernel<<<(unsigned)((nScatter + tpb - 1) / tpb), tpb, 0, stream>>>(
        d_trace, nCols, nRows, lanes, band, d_scratch);
    CHECKCUDAERR(cudaGetLastError());
}

}  // namespace blake3_gpu
}  // namespace gate_bands

#endif
