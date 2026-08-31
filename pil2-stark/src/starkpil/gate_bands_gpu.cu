// GPU expander: the same reconstruction as gate_bands_cpu.hpp, on device.
//
// One thread per band, launched after the trace copy so the interiors are never written on the
// host. Reads the band's boundary cells out of the device trace, runs its family's width-16
// permutation, and scatters the snapshots into the interior.
//
// A third implementation of the same permutations, so all three must agree exactly:
// GateBands.Poseidon*SnapshotsMatchTheGate pins the host copies against the gates, and
// GateBandsGPU.MatchesTheHostExpander pins this file against those, cell for cell.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "poseidon_goldilocks_constants.hpp"
#include "poseidon2_goldilocks_constants.hpp"
#include "gate_bands.hpp"
#include "gate_bands_blake3.hpp"

namespace {

// Band geometry and the kind predicates; the row layouts below index straight into them.
using namespace gate_bands;

constexpr int W = 16;
constexpr int HALF_F = 4;
constexpr int N_PART = 22;
constexpr int S_STRIDE = 2 * W - 1;

__device__ __constant__ uint64_t GB_C16[150];
__device__ __constant__ uint64_t GB_S16[682];
__device__ __constant__ uint64_t GB_M16[256];
__device__ __constant__ uint64_t GB_P16[256];
__device__ __constant__ uint64_t GB_P2_C16[150];
__device__ __constant__ uint64_t GB_P2_D16[16];

__device__ __forceinline__ void gb_pow7(gl64_t &x) {
    gl64_t x2 = x * x, x3 = x * x2, x4 = x2 * x2;
    x = x3 * x4;
}

__device__ __forceinline__ void gb_add(gl64_t *st, const gl64_t *C) {
#pragma unroll
    for (int i = 0; i < W; i++) st[i] = st[i] + C[i];
}

__device__ __forceinline__ void gb_pow7add(gl64_t *st, const gl64_t *C) {
#pragma unroll
    for (int i = 0; i < W; i++) { gb_pow7(st[i]); st[i] = st[i] + C[i]; }
}

// state <- mat^T * state, matching mvp_state_pos1 (column-major dot per output).
__device__ __forceinline__ void gb_mvp(gl64_t *st, const gl64_t *mat) {
    gl64_t old[W];
#pragma unroll
    for (int i = 0; i < W; i++) old[i] = st[i];
#pragma unroll
    for (int i = 0; i < W; i++) {
        gl64_t acc = gl64_t(uint64_t(0));
#pragma unroll
        for (int j = 0; j < W; j++) acc = acc + old[j] * mat[j * W + i];
        st[i] = acc;
    }
}

__device__ __forceinline__ gl64_t gb_dot(const gl64_t *st, const gl64_t *S) {
    gl64_t acc = gl64_t(uint64_t(0));
#pragma unroll
    for (int i = 0; i < W; i++) acc = acc + st[i] * S[i];
    return acc;
}

// Snapshot-taking permutation. Each snapshot goes to `sink` the moment it exists and is never
// kept: `state(group, st)` takes a full 16-word group, `anchor(group, i, v)` one partial-round
// word. Streaming keeps a thread's working set to the 16-word state; buffering all 12 groups
// costs 1.5 KB of local memory per thread and about a third of the kernel's throughput.
template <typename Sink>
__device__ void gb_perm_snapshots(gl64_t *st, Sink &sink) {
    const gl64_t *C = (const gl64_t *)GB_C16;
    const gl64_t *S = (const gl64_t *)GB_S16;
    const gl64_t *M = (const gl64_t *)GB_M16;
    const gl64_t *P = (const gl64_t *)GB_P16;

    sink.state(0, st);                                       // ordered initial state
    gb_add(st, C);
    for (int r = 0; r < HALF_F - 1; r++) { gb_pow7add(st, &C[(r + 1) * W]); gb_mvp(st, M); sink.state(r + 1, st); }
    gb_pow7add(st, &C[HALF_F * W]); gb_mvp(st, P); sink.state(4, st);

    for (int r = 0; r < N_PART; r++) {
        // Partial-round anchors: rounds 0..10 form group 5, rounds 11..21 group 7.
        sink.anchor(r <= 10 ? 5 : 7, r <= 10 ? r : r - 11, (uint64_t)st[0]);
        gb_pow7(st[0]);
        st[0] = st[0] + C[(HALF_F + 1) * W + r];
        const gl64_t *Sr = &S[S_STRIDE * r];
        gl64_t s0 = gb_dot(st, Sr);
        const gl64_t *Sw = Sr + (W - 1);
        gl64_t a = st[0];
#pragma unroll
        for (int t = 1; t < W; t++) st[t] = st[t] + a * Sw[t];
        st[0] = s0;
        if (r == 10) sink.state(6, st);        // midstate; unused by both row layouts
        if (r == 21) sink.state(8, st);
    }
    for (int r = 0; r < HALF_F - 1; r++) {
        gb_pow7add(st, &C[(HALF_F + 1) * W + N_PART + r * W]); gb_mvp(st, M); sink.state(9 + r, st);
    }
    // The last round is not snapshotted -- its result is `out`, a boundary cell the map already
    // placed -- but it is still run, so `st` ends up holding the permutation output.
#pragma unroll
    for (int i = 0; i < W; i++) gb_pow7(st[i]);
    gb_mvp(st, M);
}

// ── Poseidon2 ────────────────────────────────────────────────────────────────
// External matmul: four M4 blocks, then add the block-wise column sums.
__device__ __forceinline__ void p2_matmul_external(gl64_t *x) {
#pragma unroll
    for (int b = 0; b < 4; b++) {
        gl64_t *s = x + 4 * b;
        gl64_t t0 = s[0] + s[1], t1 = s[2] + s[3];
        gl64_t t2 = s[1] + s[1] + t1, t3 = s[3] + s[3] + t0;
        gl64_t t4 = t1 + t1; t4 = t4 + t4 + t3;
        gl64_t t5 = t0 + t0; t5 = t5 + t5 + t2;
        s[0] = t3 + t5; s[1] = t5; s[2] = t2 + t4; s[3] = t4;
    }
    gl64_t sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) sum[i] = x[i] + x[4 + i] + x[8 + i] + x[12 + i];
#pragma unroll
    for (int b = 0; b < 4; b++)
        for (int i = 0; i < 4; i++) x[4 * b + i] = x[4 * b + i] + sum[i];
}

template <typename Sink>
__device__ void gb_perm_snapshots_p2(gl64_t *st, Sink &sink) {
    const gl64_t *RC = (const gl64_t *)GB_P2_C16;
    const gl64_t *D  = (const gl64_t *)GB_P2_D16;

    p2_matmul_external(st);
    sink.state(0, st);
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int i = 0; i < W; i++) { st[i] = st[i] + RC[W * r + i]; gb_pow7(st[i]); }
        p2_matmul_external(st);
        sink.state(r + 1, st);                               // groups 1..4
    }
    for (int r = 0; r < 22; r++) {
        sink.anchor(r <= 10 ? 5 : 7, r <= 10 ? r : r - 11, (uint64_t)st[0]);
        st[0] = st[0] + RC[4 * W + r];
        gb_pow7(st[0]);
        gl64_t sum = gl64_t(uint64_t(0));
#pragma unroll
        for (int i = 0; i < W; i++) sum = sum + st[i];
#pragma unroll
        for (int i = 0; i < W; i++) st[i] = st[i] * D[i] + sum;
        if (r == 10) sink.state(6, st);
        if (r == 21) sink.state(8, st);
    }
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int i = 0; i < W; i++) { st[i] = st[i] + RC[4 * W + 22 + r * W + i]; gb_pow7(st[i]); }
        p2_matmul_external(st);
        if (r < 3) sink.state(9 + r, st);                    // groups 9..11
    }
}

// Scatters snapshots into a band's interior as they arrive. The two geometries differ only in
// which row and column each group lands on, so `agg` is all that distinguishes them.
struct BandSink {
    uint64_t *trace;
    uint64_t nCols;
    uint64_t row;
    bool agg;

    // Row offset and column for a full 16-word group, or col < 0 when the layout drops it.
    __device__ __forceinline__ void groupSlot(int group, int &rowOff, int &col) const {
        col = -1;
        if (agg) {
            // Two chain slots per row, so groups pair up; group 4 sits alone on chain 1 of
            // the anchor row.
            switch (group) {
                case 0:  rowOff = 0; col = AGG_COL_P1; return;
                case 1:  rowOff = 0; col = AGG_COL_P2; return;
                case 2:  rowOff = 1; col = AGG_COL_P1; return;
                case 3:  rowOff = 1; col = AGG_COL_P2; return;
                case 4:  rowOff = AGG_ANCHOR_ROW; col = AGG_COL_P1; return;
                case 8:  rowOff = 3; col = AGG_COL_P1; return;
                case 9:  rowOff = 3; col = AGG_COL_P2; return;
                case 10: rowOff = 4; col = AGG_COL_P1; return;
                case 11: rowOff = 4; col = AGG_COL_P2; return;
                default: return;                              // 5, 7 are anchors; 6 is dropped
            }
        }
        // One chain slot, groups running down the rows; pos_row_im is the inverse.
        switch (group) {
            case 0: case 1: case 2: case 3: case 4: rowOff = group;     col = POS_COL_P; return;
            case 8: case 9: case 10: case 11:       rowOff = group - 2; col = POS_COL_P; return;
            default: return;
        }
    }

    __device__ __forceinline__ void state(int group, const gl64_t *st) {
        int rowOff, col;
        groupSlot(group, rowOff, col);
        if (col < 0) return;
        uint64_t *dst = trace + (row + rowOff) * nCols + col;
#pragma unroll
        for (int i = 0; i < W; i++) dst[i] = (uint64_t)st[i];
    }

    // Anchors run end to end across the anchor row: group 5's eleven words then group 7's, the
    // first sixteen filling a chain slot and the last six spilling to the overflow columns --
    // a[18..23] for a compressor band, a[9..14] for an aggregation one. Both are free of plonk
    // gates, and the aggregation tail stops short of the key bit at a[15].
    __device__ __forceinline__ void anchor(int group, int i, uint64_t v) {
        const int idx = (group == 5) ? i : 11 + i;
        const int chainCol = agg ? AGG_COL_P2 : POS_COL_P;
        const int overflow = agg ? AGG_ANCHOR_OVERFLOW_COL : POS_ANCHOR_OVERFLOW_COL;
        const uint64_t r = row + (agg ? AGG_ANCHOR_ROW : POS_ANCHOR_ROW);
        trace[r * nCols + (idx < 16 ? chainCol + idx : overflow + (idx - 16))] = v;
    }
};


// ─── Compact witness widening ────────────────────────────────────────────────
//
// `getCommitedPols` can only place what the circom witness carries, so it fills the exec map's
// columns and leaves the rest for a gate-band expander to rebuild. Those columns therefore need never
// cross PCIe: the host hands over a compact `N x mapCols` buffer, copied in one contiguous run, and
// this widens it on device.
//
// The copy has to be contiguous. Going straight into the strided columns with a 2D copy is slower
// than shipping the full width, because the rows are only mapCols*8 bytes -- too small for the DMA
// engine. The caller zeroes the destination first: the expander does not write every row of the
// columns it owns (padding rows past the last band, the multiplicity columns past the table), and
// those cells still enter the commitment, so they must be a function of the witness rather than
// whatever the buffer last held.
__global__ void widenCompactWitnessKernel(uint64_t *trace, uint64_t nCols, uint64_t nRows,
                                          const uint64_t *compact, uint64_t mapCols) {
    const uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nRows * mapCols) return;
    const uint64_t row = t / mapCols, col = t - row * mapCols;
    trace[row * nCols + col] = compact[t];
}


// ─── BLAKE3 ──────────────────────────────────────────────────────────────────
//
// The arithmetic is the SAME code the CPU expander runs: gate_bands_blake3.hpp is one templated
// implementation that both backends instantiate, so the permutation cannot drift between them.
// Only the multiplicity sink differs. A CPU thread owns a private 1.5 MB Multiplicities and
// reduces once at the end; a thread here is a single band among thousands, so it adds straight
// into the trace's own mul columns. The counts ARE the field elements -- Goldilocks::fromU64 is
// the identity -- so no conversion pass follows, and at 2^17 table entries contention stays low.
//
// The counters must start at zero, which the cudaMemsetAsync in expandGateBandsGPU guarantees; the
// host path gets the same from a freshly constructed Multiplicities.
// Counters go into a dense scratch buffer, NOT straight into the trace's two multiplicity columns.
// In the trace those columns are strided by nCols, so every atomic would take a cache line of its
// own and share it with witness cells other threads are storing to. Dense, the counters are a
// contiguous 1.5 MB that stays L2-resident, neighbouring counters share a line, and nothing
// false-shares with the trace. scatterBlake3MultiplicitiesKernel moves them into the columns at the
// end.
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

// The dense buffer is sized by AirInstanceInfo::blake3MulWords(), which hardcodes the two sizes
// rather than including this header. Keep them in step.
static_assert(gate_bands::blake3::TABLE_SIZE == (1ull << 17), "blake3MulWords assumes TABLE_SIZE 2^17");
static_assert(gate_bands::blake3::RANGE_SIZE == (1ull << 16), "blake3MulWords assumes RANGE_SIZE 2^16");

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

// A kernel per family rather than one kernel with a branch: ptxas allocates registers for the union
// of whatever a kernel can reach, so merged, each path pays for the other's live state and spills.
// Nothing is lost by splitting -- a setup is one hash family, so a band list never mixes the two,
// and `anyBlake3` already tells the host which one to launch.

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

// No nRows: first_bad_band enforces the row bound host-side, once.
__global__ void expandPoseidonBandsKernel(uint64_t *trace, uint64_t nCols,
                                          const uint64_t *bands, uint64_t nBands, uint64_t lanes) {
    const uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nBands) return;
    const uint64_t row = bands[t * 3], kind = bands[t * 3 + 1];
    if (!is_known_kind(kind) || is_blake3(kind)) return;
    const bool p1 = is_poseidon1(kind);
    const bool agg = is_aggregation(kind);

    uint64_t *base = trace + row * nCols;
    gl64_t st[W];
    if (is_compression(kind)) {
        // Key bits sit where the setup put them: a[16] and a[17] of the compressor's first
        // row, a[15] of the two middle rows for aggregation.
        const uint64_t key = agg
            ? ((trace[(row + 1) * nCols + 15] & 1) | ((trace[(row + 2) * nCols + 15] & 1) << 1))
            : ((base[16] & 1) | ((base[17] & 1) << 1));
        // in[0..4] goes to 4-group slot `key`, the others shifting down.
        int src = 1;
        for (int g = 0; g < 4; g++) {
            const int from = (g == (int)key) ? 0 : src++;
#pragma unroll
            for (int i = 0; i < 4; i++) st[g * 4 + i] = gl64_t(base[from * 4 + i]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < W; i++) st[i] = gl64_t(base[i]);
    }

    BandSink sink{trace, nCols, row, agg};
    if (p1) gb_perm_snapshots(st, sink);
    else    gb_perm_snapshots_p2(st, sink);
}

}  // namespace

// Uploads the Poseidon constant tables to the current device. They are __constant__, so this is
// per-device state: the caller selects the device first. Idempotent, so repeating it per air is
// harmless, but it must not race work already reading the tables on that device.
extern "C" void uploadGateBandConstantsGPU() {
    // Same tables the CPU gate uses, so the two cannot drift.
    using PC = PoseidonGoldilocksConstants::Poseidon1Tables<16>;
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_C16, PC::C, 150 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_S16, PC::S, 682 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_M16, PC::M_flat, 256 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_P16, PC::P_flat, 256 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_P2_C16, Poseidon2GoldilocksConstants::C16, 150 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyToSymbol(GB_P2_D16, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t)));
}

// Expands every band in place on the device trace. The band list is already device-resident,
// uploaded with the air's setup, so this is launch-only -- no allocation, no sync -- and sits
// stream-ordered behind the trace copy.
//
// `d_blake3Mul` MUST belong to this stream alone: the three launches below are ordered only within
// `stream_`, so a shared buffer lets another stream's memset land inside this one's fill.
// `lanes` is the band section's aux word: a setup parameter, so it travels rather than being
// derived from nCols, which cannot tell the lane layouts apart. `anyBlake3` is decided host-side
// at setup and gates the one extra pass this needs.
// Widens a compact `N x mapCols` device buffer into the trace's first `mapCols` columns. The
// destination must already be zeroed; see the kernel.
extern "C" void widenCompactWitnessGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                       const uint64_t *d_compact, uint64_t mapCols, void *stream_) {
    if (mapCols == 0 || nRows == 0) return;
    const int tpb = 256;
    const uint64_t n = nRows * mapCols;
    widenCompactWitnessKernel<<<(unsigned)((n + tpb - 1) / tpb), tpb, 0, (cudaStream_t)stream_>>>(
        d_trace, nCols, nRows, d_compact, mapCols);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void expandGateBandsGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                   const uint64_t *d_bands, uint64_t nBands, uint64_t aux,
                                   bool anyBlake3, uint64_t *d_blake3Mul, void *stream_) {
    if (nBands == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;
    const int tpb = 128;
    // The band section's aux word, unpacked exactly as the CPU driver does it: LANES low, the
    // a[]/S[] band width high. Poseidon leaves it 0 and reads neither.
    const uint64_t lanes = aux & 0xFFFFFFFFull;
    const uint64_t band = aux >> 32;
    if (!anyBlake3) {
        expandPoseidonBandsKernel<<<(unsigned)((nBands + tpb - 1) / tpb), tpb, 0, stream>>>(
            d_trace, nCols, d_bands, nBands, lanes);
        CHECKCUDAERR(cudaGetLastError());
        return;
    }

    // A BLAKE3 air needs the scratch; set_gate_bands allocates it whenever anyBlake3 is set, so a
    // null here means the two disagree and the counters would land nowhere.
    // fprintf rather than zklog: zklog.error followed by exitProcess is what made a const-file
    // size mismatch exit 255 with no message anywhere -- not stdout, not stderr, not a file -- and
    // these two conditions are setup bugs whose whole value is being readable.
    if (d_blake3Mul == nullptr) {
        fprintf(stderr, "expandGateBandsGPU: a BLAKE3 air reached the expander with no multiplicity "
                        "scratch; the caller allocates it per STREAM on that stream's first BLAKE3 "
                        "commit, so the two disagree and every lookup count would be dropped\n");
        fflush(stderr);
        exit(-1);
    }
    // The table has to fit the trace, or its tail rows have counts nothing can carry.
    if (nRows < gate_bands::blake3::TABLE_SIZE) {
        fprintf(stderr, "expandGateBandsGPU: nRows %llu cannot hold the %llu-row BLAKE3 table\n",
                (unsigned long long)nRows, (unsigned long long)gate_bands::blake3::TABLE_SIZE);
        fflush(stderr);
        exit(-1);
    }

    const uint64_t mulWords = gate_bands::blake3::TABLE_SIZE + gate_bands::blake3::RANGE_SIZE;
    // Stream-ordered ahead of the expansion, so every atomicAdd lands on a zeroed counter. Dense,
    // so this is one contiguous memset rather than a strided kernel over trace columns.
    CHECKCUDAERR(cudaMemsetAsync(d_blake3Mul, 0, mulWords * sizeof(uint64_t), stream));

    // One thread per (band, lane); see the kernel.
    const uint64_t nThreads = nBands * lanes;
    expandBlake3BandsKernel<<<(unsigned)((nThreads + tpb - 1) / tpb), tpb, 0, stream>>>(
        d_trace, nCols, d_bands, nBands, lanes, band, d_blake3Mul);
    CHECKCUDAERR(cudaGetLastError());

    const uint64_t nScatter = gate_bands::blake3::TABLE_SIZE;
    scatterBlake3MultiplicitiesKernel<<<(unsigned)((nScatter + tpb - 1) / tpb), tpb, 0, stream>>>(
        d_trace, nCols, nRows, lanes, band, d_blake3Mul);
    CHECKCUDAERR(cudaGetLastError());
}
