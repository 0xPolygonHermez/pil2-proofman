// The device side of the blake3 BN128 wrap's stage-1 witness (pilfflonk_wrap_exec.hpp), in the GPU
// library only: the blocks' expansion, as wrap-witness/src/blake3.rs does on the host, with the
// recursion's blake3 permutation (gate_bands_blake3.hpp). The additions and the map's gather are the
// PLONK GPU prover's (rapidsnark/plonk_prover.cu).
#include "pilfflonk_cuda.cuh"
#include "pilfflonk_wrap_exec.hpp"

#define B3_NO_GOLDILOCKS
#include "../starkpil/recursion_trace/gate_bands/gate_bands_blake3.hpp"

namespace {

namespace b3 = gate_bands::blake3;

// The stage-1 columns of the wrap's AIR, as common/src/exec_format.rs's blake3_wrap_cols.
constexpr uint64_t A = 0, ST = 18, VA = 34, VC = 36, VB = 37, VD = 41, X = 45, Y = 47, VA_P = 49, VD_P = 53,
                   VC_P = 57, VB_P_S = 61, VA_PP = 69, VD_PP = 73, VC_PP = 77, VB_PP_XOR = 81, VB_PP_T = 85,
                   D_INV = 86, MUL_TABLE = 87, MUL_RANGE = 88;
constexpr uint64_t BLOCK_ROWS = 64;
constexpr uint64_t TABLE_ROWS = 1ull << 17, RANGE_ROWS = 1ull << 16;

__device__ __forceinline__ Element small(uint64_t v) {
    Element e = Fr::zero();
    e[0] = static_cast<uint32_t>(v);
    e[1] = static_cast<uint32_t>(v >> 32);
    Fr::toMontgomery(e);
    return e;
}

struct Cells {
    Element *columns;
    const uint64_t *positions;
    uint64_t n;
    uint64_t base;
    __device__ __forceinline__ void put(uint64_t row, uint64_t col, uint64_t v) const {
        columns[positions[col] * n + base + row] = small(v);
    }
    __device__ __forceinline__ void bytes(uint64_t row, uint64_t col, uint32_t w) const {
        for (int i = 0; i < 4; ++i) put(row, col + i, (w >> (8 * i)) & 0xff);
    }
};

__global__ void toMontgomery(Element *data, uint64_t n) {
    for (uint64_t i = firstIndex(); i < n; i += stride()) {
        Fr::toMontgomery(data[i]);
    }
}

// One thread a block: its 56 G rows and its feedforward, as wrap-witness/src/blake3.rs's Block::fill.
__global__ void blake3Blocks(Element *columns, const uint64_t *positions, uint64_t n,
                             const pilfflonk_wrap_block *blocks, uint64_t nBlocks, uint32_t *tableCounts,
                             uint32_t *rangeCounts) {
    for (uint64_t i = firstIndex(); i < nBlocks; i += stride()) {
        const pilfflonk_wrap_block &blk = blocks[i];
        const Cells c{columns, positions, n, i * BLOCK_ROWS};
        const bool node = blk.kind == 0, parent = blk.kind == 2;
        auto xor4 = [&](uint32_t a, uint32_t b, uint8_t rot) {
            for (int k = 0; k < 4; ++k) {
                atomicAdd(&tableCounts[b3::table_row(a >> (8 * k), b >> (8 * k), rot)], 1u);
            }
        };

        uint32_t v[16];
        for (int j = 0; j < 8; ++j) v[j] = blk.cv[j];
        for (int j = 0; j < 4; ++j) v[8 + j] = b3::iv(j);
        v[12] = blk.counter_lo;
        v[13] = 0;
        v[14] = blk.block_len;
        v[15] = blk.flags;

        // d_inv, as the host inverted it: sppark's reciprocal needs every lane of a warp, which a
        // thread a block cannot give it.
        auto putInverse = [&](int t, const uint8_t *bytes) {
            Element d;
            for (int l = 0; l < 8; ++l) {
                const uint8_t *b = bytes + 4 * l;
                d[l] = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
            }
            Fr::toMontgomery(d);
            columns[positions[D_INV] * n + c.base + t] = d;
        };
        if (!parent) {
            for (int t = 0; t < 8; ++t) putInverse(t, blk.dinv[t]);
        }
        if (node) {
            for (int k = 0; k < 4; ++k) putInverse(56 + k, blk.dinv_ff[k]);
        }

        for (int t = 0; t < 56; ++t) {
            const int r = t / 8, g = t % 8;
            for (int j = 0; j < 16; ++j) c.put(t, ST + j, v[j]);
            const int ia = b3::g_idx(g, 0), ib = b3::g_idx(g, 1), ic = b3::g_idx(g, 2), id = b3::g_idx(g, 3);
            const uint32_t x = blk.m[b3::sigma(r, 2 * g)], y = blk.m[b3::sigma(r, 2 * g + 1)];
            const b3::GTrace s = b3::g_step(v[ia], v[ib], v[ic], v[id], x, y);
            const uint32_t limbs[3] = {s.va, x, y};
            const uint64_t limbCols[3] = {VA, X, Y};
            for (int l = 0; l < 3; ++l) {
                for (int h = 0; h < 2; ++h) {
                    const uint32_t limb = (limbs[l] >> (16 * h)) & 0xffff;
                    c.put(t, limbCols[l] + h, limb);
                    atomicAdd(&rangeCounts[limb], 1u);
                }
            }
            c.put(t, VC, v[ic]);
            c.bytes(t, VB, s.vb);
            c.bytes(t, VD, s.vd);
            c.bytes(t, VA_P, s.a1);
            c.bytes(t, VD_P, s.d1);
            c.bytes(t, VC_P, s.c1);
            c.bytes(t, VA_PP, s.a2);
            c.bytes(t, VD_PP, s.d2);
            c.bytes(t, VC_PP, s.c2);
            c.bytes(t, VB_PP_XOR, s.z);
            for (int k = 0; k < 4; ++k) {
                uint8_t s0, s1;
                b3::table_out((s.vb >> (8 * k)) & 0xff, (s.c1 >> (8 * k)) & 0xff, 12, s0, s1);
                c.put(t, VB_P_S + 2 * k, s0);
                c.put(t, VB_P_S + 2 * k + 1, s1);
            }
            c.put(t, VB_PP_T, (s.z >> 7) & 1);
            xor4(s.vd, s.a1, 0);
            xor4(s.vb, s.c1, 12);
            xor4(s.d1, s.a2, 0);
            xor4(s.b1, s.c2, 0);
            v[ia] = s.a2;
            v[ib] = s.b2;
            v[ic] = s.c2;
            v[id] = s.d2;
        }
        for (int t = 56; t < 64; ++t) {
            for (int j = 0; j < 16; ++j) c.put(t, ST + j, v[j]);
        }
        for (int k = 0; k < (node ? 4 : 8); ++k) {
            const int t = 56 + k;
            const uint32_t a0 = v[2 * k], a1 = v[2 * k + 1];
            const uint32_t b0 = k < 4 ? v[2 * k + 8] : blk.cv[2 * k - 8];
            const uint32_t b1 = k < 4 ? v[2 * k + 9] : blk.cv[2 * k - 7];
            c.bytes(t, VB, a0);
            c.bytes(t, VD, b0);
            c.bytes(t, VA_P, a0 ^ b0);
            c.bytes(t, VD_P, a1);
            c.bytes(t, VC_P, b1);
            c.bytes(t, VA_PP, a1 ^ b1);
            xor4(a0, b0, 0);
            xor4(a1, b1, 0);
            if (node) {
                c.put(t, A + 1, blk.over[k]);
            }
        }
    }
}

__global__ void multiplicities(Element *columns, const uint64_t *positions, uint64_t n, const uint32_t *tableCounts,
                               const uint32_t *rangeCounts, const uint32_t *hostRange) {
    for (uint64_t row = firstIndex(); row < TABLE_ROWS; row += stride()) {
        columns[positions[MUL_TABLE] * n + row] = small(tableCounts[row]);
        if (row < RANGE_ROWS) {
            columns[positions[MUL_RANGE] * n + row] = small(uint64_t(rangeCounts[row]) + hostRange[row]);
        }
    }
}

} // namespace

extern "C" {

void pilfflonk_gpu_to_montgomery(void *data, uint64_t n) {
    if (n == 0) return;
    toMontgomery<<<blocksFor(n), THREADS>>>(static_cast<Element *>(data), n);
    CHECKCUDAERR(cudaGetLastError());
}

void pilfflonk_gpu_wrap_blake3(void *columns, const uint64_t *positions, uint64_t nRows, const void *blocks,
                               uint64_t nBlocks, uint32_t *tableCounts, uint32_t *rangeCounts,
                               const uint32_t *hostRange) {
    if (nBlocks > 0) {
        // A thread holds a block's whole state: small blocks, many of them.
        const uint32_t threads = 64, grid = static_cast<uint32_t>((nBlocks + threads - 1) / threads);
        blake3Blocks<<<grid, threads>>>(static_cast<Element *>(columns), positions, nRows,
                                                 static_cast<const pilfflonk_wrap_block *>(blocks), nBlocks,
                                                 tableCounts, rangeCounts);
        CHECKCUDAERR(cudaGetLastError());
    }
    multiplicities<<<blocksFor(TABLE_ROWS), THREADS>>>(static_cast<Element *>(columns), positions, nRows, tableCounts,
                                                       rangeCounts, hostRange);
    CHECKCUDAERR(cudaGetLastError());
}

}
