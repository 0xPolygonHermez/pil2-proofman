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

__device__ inline void expandBlake3Lane(uint64_t *trace, uint64_t nCols, uint64_t row,
                                        uint64_t lanes, uint64_t band, uint64_t lane, uint64_t kind,
                                        uint64_t flags, uint64_t *mul) {
    namespace b3 = gate_bands::blake3;
    const b3::Kind k = kind == GB_BLAKE3_NODE           ? b3::Kind::Node
                     : kind == GB_BLAKE3_COMPRESS_CHUNK ? b3::Kind::Chunk
                                                        : b3::Kind::Parent;
    AtomicSink sink{mul};
    b3::expand_one_lane(trace, nCols, row, lanes, band, lane, k, flags, sink);
}

// One thread per (band, lane), not per band: a band's lanes are independent, so per-band would leave
// each thread running all `lanes` of them in sequence and the grid `lanes` times smaller.
//
// The lane is the FAST index deliberately. Lane `l`'s columns are `base + l * per + i`, so threads
// 4b..4b+3 write neighbouring columns of one row and their stores coalesce; band-major would put
// consecutive threads a whole block of rows apart, one memory transaction per thread per store.
__global__ void expandBlake3BandsKernel(uint64_t *trace, uint64_t nCols,
                                        const uint64_t *bands, uint64_t nBands, uint64_t lanes,
                                        uint64_t band, uint64_t *mul) {
    const uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nBands * lanes) return;
    const uint64_t b = t / lanes, lane = t % lanes;
    const uint64_t row = bands[b * 3], kind = bands[b * 3 + 1];
    // Also validated host-side in load_device_setup; an unknown kind must never be filled
    // with a guessed permutation. A non-BLAKE3 kind here would mean the host launched the wrong
    // kernel for this setup, so skip it rather than reconstruct it with the wrong permutation.
    if (!is_known_kind(kind) || !is_blake3(kind)) return;
    expandBlake3Lane(trace, nCols, row, lanes, band, lane, kind, bands[b * 3 + 2], mul);
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

    // One thread per (band, lane); see the kernel.
    const uint64_t nThreads = nBands * lanes;
    expandBlake3BandsKernel<<<(unsigned)((nThreads + tpb - 1) / tpb), tpb, 0, stream>>>(
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
