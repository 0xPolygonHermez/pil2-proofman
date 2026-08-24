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
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "poseidon_goldilocks_constants.hpp"
#include "poseidon2_goldilocks_constants.hpp"
#include "gate_bands.hpp"

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

__global__ void expandBandsKernel(uint64_t *trace, uint64_t nCols, const uint64_t *bands, uint64_t nBands) {
    const uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (t >= nBands) return;
    const uint64_t row = bands[t * 2], kind = bands[t * 2 + 1];
    // Also validated host-side in load_device_setup; an unknown kind must never be filled
    // with a guessed permutation.
    if (!is_known_kind(kind)) return;
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
extern "C" void expandGateBandsGPU(uint64_t *d_trace, uint64_t nCols,
                                   const uint64_t *d_bands, uint64_t nBands, void *stream_) {
    if (nBands == 0) return;
    const int tpb = 128;
    expandBandsKernel<<<(unsigned)((nBands + tpb - 1) / tpb), tpb, 0, (cudaStream_t)stream_>>>(
        d_trace, nCols, d_bands, nBands);
    CHECKCUDAERR(cudaGetLastError());
}
