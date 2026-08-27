#ifndef GATE_BANDS_BLAKE3_HPP
#define GATE_BANDS_BLAKE3_HPP

// Rebuilding a BLAKE3 block's interior from its boundary.
//
// The block is 56 rows (7 rounds x 8 G) hosting LANES permutations in parallel column groups. The
// setup places only two rows per lane -- inputs at clock `lane`, outputs at clock 56-LANES+lane --
// and the 59 columns per lane in between are a pure function of them, so they are recomputed here.
//
// Unlike the poseidon expanders this one also owes MULTIPLICITIES: BLAKE3 is the first recursion
// family whose air uses lookup tables, and `mul_table` / `mul_range` are witness columns nobody
// else fills. A wrong count does not fail per-air verification -- every air's constraints,
// including the grand-sum recursion, are satisfied by whatever values result. Only the GLOBAL
// constraint catches it. See spec 5.6.
//
// See docs/superpowers/specs/2026-08-25-blake3-recursion-air-design.md and, for the column order
// this must agree with, the generated pil_helpers for the aggregator air.

#include <cstdint>
#include <cstring>
#include <vector>
#include "goldilocks_base_field.hpp"

// The pure arithmetic below runs on both backends: the CPU expander calls it directly, the CUDA
// kernel calls the same functions on device. Only the trace writing and the multiplicity
// accumulation differ -- see gate_bands_gpu.cu, where the counts go through atomicAdd because a
// thread there is one band, not one worker with its own buffer.
#ifdef __CUDACC__
#define B3_HD __host__ __device__
#else
#define B3_HD
#endif

namespace gate_bands {
namespace blake3 {

constexpr int CLOCKS = 56;
constexpr int ROUNDS = 7;
constexpr int G_PER_ROUND = 8;
constexpr int BAND_COLS = 18;
constexpr int COLS_PER_LANE = 59;

constexpr uint64_t TABLE_SIZE = 1ull << 17;  // (a:8, b:8, rot:1)
constexpr uint64_t RANGE_SIZE = 1ull << 16;

// The constant tables are reached through accessors rather than declared at namespace scope. A
// namespace-scope constant is host-only, and marking it __constant__ makes it device-only -- the
// compiler warns that reading it from the other side is wrong, and warning is all it does. A
// function-local constexpr inside a host+device function is materialised correctly on both.
B3_HD inline uint32_t iv(int i) {
    constexpr uint32_t V[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
    return V[i];
}

B3_HD inline int sigma(int r, int i) {
    constexpr int S[ROUNDS][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};
    return S[r][i];
}

// (va, vb, vc, vd) indices into the 16-word state, per G within a round: four column mixes then
// four diagonal ones.
B3_HD inline int g_idx(int g, int i) {
    constexpr int G[G_PER_ROUND][4] = {
    {0, 4,  8, 12}, {1, 5,  9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15},
    {0, 5, 10, 15}, {1, 6, 11, 12}, {2, 7,  8, 13}, {3, 4,  9, 14},
};
    return G[g][i];
}

// Column offsets, as a function of LANES. Every array is LANE-MAJOR -- va[lane][limb], not
// va[limb][lane] -- because the air declares them that way and the generated trace follows.
struct Layout {
    uint64_t va, vb, vd, x, y;
    uint64_t va_p, vd_p, vc_p, vb_p_s;
    uint64_t va_pp, vd_pp, vc_pp, vb_pp_xor, vb_pp_t;
    uint64_t dinv, vbTopHi, outBytes, mul_table, mul_range;
};

B3_HD inline Layout layout(uint64_t lanes) {
    Layout l{};
    uint64_t o = BAND_COLS;
    auto take = [&](uint64_t per_lane) { const uint64_t at = o; o += per_lane * lanes; return at; };
    // Declaration order in blake3/aggregator.pil, which is what fixes the trace layout. The
    // BOUNDARY columns come first: the air has to declare them before it calls blake3Lanes, because
    // that function binds them and PIL2 needs an argument declared before it is passed.
    l.dinv = take(1); l.vbTopHi = take(3); l.outBytes = take(4);
    l.va = take(2);   l.vb = take(4);   l.vd = take(4);
    l.x  = take(2);   l.y  = take(2);
    l.va_p = take(4); l.vd_p = take(4); l.vc_p = take(4); l.vb_p_s = take(8);
    l.va_pp = take(4); l.vd_pp = take(4); l.vc_pp = take(4);
    l.vb_pp_xor = take(4); l.vb_pp_t = take(1);
    l.mul_table = o;  l.mul_range = o + 1;
    return l;
}

B3_HD inline uint64_t stage1_cols(uint64_t lanes) { return BAND_COLS + COLS_PER_LANE * lanes + 2; }

// Row of the XOR/ROTR table for (a, b, rot), matching blake3Tables in circuits/blake3.pil:
// A cycles fastest, then B, then ROTATION.
B3_HD inline uint64_t table_row(uint8_t a, uint8_t b, uint8_t rot) {
    return (rot == 12 ? (1ull << 16) : 0ull) | ((uint64_t)b << 8) | (uint64_t)a;
}

// Accumulators the whole air shares: one set per proof, not per band.
struct Multiplicities {
    std::vector<uint64_t> table;
    std::vector<uint64_t> range;
    Multiplicities() : table(TABLE_SIZE, 0), range(RANGE_SIZE, 0) {}
};

// The expander below is ONE implementation that both backends instantiate, so the permutation
// cannot drift between them. Two things differ, and only these two:
//
//   * the cell type -- Goldilocks::Element on the host, a bare uint64_t on the device. Element
//     wraps a u64 and fromU64 is the identity, so the stored bits are the same either way.
//   * where a lookup count goes. The host thread owns a private Multiplicities and reduces once
//     (HostSink); a device thread is one band among many and has to add atomically.
B3_HD inline void store(Goldilocks::Element &c, uint64_t v) { c.fe = v; }
B3_HD inline void store(uint64_t &c, uint64_t v) { c = v; }
B3_HD inline uint64_t load(const Goldilocks::Element &c) { return c.fe; }
B3_HD inline uint64_t load(const uint64_t &c) { return c; }

struct HostSink {
    Multiplicities &m;
    void table(uint64_t i) { m.table[i]++; }
    void range(uint64_t i) { m.range[i]++; }
};

// ─── Reference permutation ───────────────────────────────────────────────────
// Written independently of setup/circom/blake3_gate.cpp rather than shared with it: two
// implementations that can be differenced is the point, and this one has to expose every
// intermediate the trace holds, which the witness gate does not.

B3_HD inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << ((32 - n) & 31)); }

// One G evaluation, publishing every intermediate the air commits to.
struct GTrace {
    uint32_t va, vb, vd;           // inputs the air commits to; vc is an expression there
    uint32_t a1, d1, c1, b1;       // after the first half
    uint32_t a2, d2, c2, b2;       // after the second
    uint32_t z;                    // vb' ^ vc'', whose rotl1 is vb''
};

B3_HD inline GTrace g_step(uint32_t va, uint32_t vb, uint32_t vc, uint32_t vd, uint32_t x, uint32_t y) {
    GTrace t{};
    t.va = va; t.vb = vb; t.vd = vd;
    t.a1 = va + vb + x;
    t.d1 = rotr32(vd ^ t.a1, 16);
    t.c1 = vc + t.d1;
    t.b1 = rotr32(vb ^ t.c1, 12);
    t.a2 = t.a1 + t.b1 + y;
    t.d2 = rotr32(t.d1 ^ t.a2, 8);
    t.c2 = t.c1 + t.d2;
    t.z  = t.b1 ^ t.c2;
    t.b2 = rotr32(t.z, 7);
    return t;
}

// The XOR/ROTR table's two outputs for (a, b, rot), matching blake3Tables.
B3_HD inline void table_out(uint8_t a, uint8_t b, uint8_t rot, uint8_t &c0, uint8_t &c1) {
    const uint32_t byte = (uint32_t)(a ^ b);
    const uint32_t c = rot == 0 ? byte : ((byte >> rot) | (byte << ((32 - rot) % 32)));
    const int shift = ((32 - rot) % 32) / 8 % 4;
    c0 = (uint8_t)((c >> (8 * shift)) & 0xFF);
    c1 = (uint8_t)((c >> (8 * ((shift + 1) % 4))) & 0xFF);
}

// ─── Block expansion ─────────────────────────────────────────────────────────

enum class Kind { Node, Chunk, Parent };

// What a block feeds the permutation. The initial state is derived, not stored: BLAKE3 fixes
// st[0..8] = cv, st[8..12] = IV[0..4], st[12] = counterLo, st[13] = counterHi, st[14] = blockLen,
// st[15] = flags, and counterHi is identically zero across this design.
struct BlockInputs {
    uint32_t cv[8];
    uint32_t block[16];
    uint32_t blockLen, counterLo, flags;

    B3_HD void initial_state(uint32_t st[16]) const {
        for (int i = 0; i < 8; i++) st[i] = cv[i];
        for (int i = 0; i < 4; i++) st[8 + i] = iv(i);
        st[12] = counterLo;
        st[13] = 0;
        st[14] = blockLen;
        st[15] = flags;
    }
};

// Rebuild one lane's whole 56-row column group, and count every lookup it makes. Returns the final
// state, which the caller needs for the feedforward.
template <class T, class Sink>
B3_HD inline void expand_lane(T *trace, uint64_t nCols, uint64_t base, uint64_t lane,
                        const Layout &L, const BlockInputs &in, Sink &mul,
                        uint32_t finalState[16]) {
    auto put = [&](uint64_t r, uint64_t c, uint64_t v) { store(trace[r * nCols + c], v); };

    uint32_t v[16];
    in.initial_state(v);

    for (int r = 0; r < ROUNDS; r++) {
        for (int g = 0; g < G_PER_ROUND; g++) {
            const uint64_t row = base + (uint64_t)(r * G_PER_ROUND + g);
            const int ia = g_idx(g, 0), ib = g_idx(g, 1), ic = g_idx(g, 2), id = g_idx(g, 3);
            const uint32_t x = in.block[sigma(r, 2 * g)];
            const uint32_t y = in.block[sigma(r, 2 * g + 1)];
            const GTrace t = g_step(v[ia], v[ib], v[ic], v[id], x, y);

            auto lim = [&](uint64_t colBase, int per, int idx) { return colBase + lane * per + idx; };
            for (int i = 0; i < 2; i++) {
                put(row, lim(L.va, 2, i), (t.va >> (16 * i)) & 0xFFFF);
                put(row, lim(L.x,  2, i), (x    >> (16 * i)) & 0xFFFF);
                put(row, lim(L.y,  2, i), (y    >> (16 * i)) & 0xFFFF);
                // va, x and y: every limb, every row. vc is not a column -- the air builds it
                // from vc'' and the IV.
                mul.range((t.va >> (16 * i)) & 0xFFFF);
                mul.range((x    >> (16 * i)) & 0xFFFF);
                mul.range((y    >> (16 * i)) & 0xFFFF);
            }
            for (int i = 0; i < 4; i++) {
                const uint8_t vb_b = (t.vb >> (8 * i)) & 0xFF, vd_b = (t.vd >> (8 * i)) & 0xFF;
                const uint8_t a1_b = (t.a1 >> (8 * i)) & 0xFF, d1_b = (t.d1 >> (8 * i)) & 0xFF;
                const uint8_t c1_b = (t.c1 >> (8 * i)) & 0xFF, a2_b = (t.a2 >> (8 * i)) & 0xFF;
                const uint8_t d2_b = (t.d2 >> (8 * i)) & 0xFF, c2_b = (t.c2 >> (8 * i)) & 0xFF;
                const uint8_t b1_b = (t.b1 >> (8 * i)) & 0xFF, z_b = (t.z >> (8 * i)) & 0xFF;
                put(row, lim(L.vb, 4, i), vb_b);
                put(row, lim(L.vd, 4, i), vd_b);
                put(row, lim(L.va_p, 4, i), a1_b);
                put(row, lim(L.vd_p, 4, i), d1_b);
                put(row, lim(L.vc_p, 4, i), c1_b);
                put(row, lim(L.va_pp, 4, i), a2_b);
                put(row, lim(L.vd_pp, 4, i), d2_b);
                put(row, lim(L.vc_pp, 4, i), c2_b);
                put(row, lim(L.vb_pp_xor, 4, i), z_b);

                uint8_t s0, s1;
                table_out(vb_b, c1_b, 12, s0, s1);
                put(row, L.vb_p_s + lane * 8 + i * 2 + 0, s0);
                put(row, L.vb_p_s + lane * 8 + i * 2 + 1, s1);

                // the four XOR groups the permutation looks up, per byte
                mul.table(table_row(vd_b, a1_b, 0));    // vd' = (vd ^ va') >>> 16
                mul.table(table_row(vb_b, c1_b, 12));   // vb' = (vb ^ vc') >>> 12
                mul.table(table_row(d1_b, a2_b, 0));    // vd'' = (vd' ^ va'')
                mul.table(table_row(b1_b, c2_b, 0));    // z = vb' ^ vc''
            }
            put(row, L.vb_pp_t + lane, (t.z >> 7) & 1);

            v[ia] = t.a2; v[ib] = t.b2; v[ic] = t.c2; v[id] = t.d2;
        }
    }
    std::memcpy(finalState, v, sizeof(v));
}

// ─── Whole-block expansion ───────────────────────────────────────────────────

// Modular inverse in Goldilocks by Fermat, or 0 for zero -- the canonicity gadget is
// `sel * (lo * (d*dinv - 1)) === 0`, so at lo = 0 any dinv satisfies it.
B3_HD inline uint64_t inv_p(uint64_t a) {
    constexpr uint64_t P = 0xFFFFFFFF00000001ull;
    a %= P;
    if (a == 0) return 0;
    auto mul = [](uint64_t x, uint64_t y) {
        return (uint64_t)((unsigned __int128)x * y % (unsigned __int128)0xFFFFFFFF00000001ull);
    };
    uint64_t r = 1, b = a, e = P - 2;
    while (e) { if (e & 1) r = mul(r, b); b = mul(b, b); e >>= 1; }
    return r;
}

// Build a block's inputs from the boundary cells the map already placed.
//
// The three kinds differ only here. Node and Compress-parent take the IV as their chaining value;
// only Node applies the Merkle path bit, which swaps the block's two halves. `flags` never comes
// off the trace -- the air holds it in a fixed column -- so it arrives in the band's payload.
template <class T>
B3_HD inline BlockInputs read_boundary(const T *trace, uint64_t nCols, uint64_t inRow,
                                 Kind kind, uint64_t flags) {
    auto cell = [&](int j) { return load(trace[inRow * nCols + j]); };
    BlockInputs in{};
    in.flags = (uint32_t)flags;

    if (kind == Kind::Parent) {
        for (int i = 0; i < 8; i++) in.cv[i] = iv(i);
        for (int i = 0; i < 16; i++) in.block[i] = (uint32_t)cell(i);
        in.blockLen = (uint32_t)cell(16);
        in.counterLo = (uint32_t)cell(17);
        return in;
    }

    // Node and chunk both split Goldilocks words into (lo, hi) halves.
    uint64_t words[8];
    if (kind == Kind::Node) {
        for (int i = 0; i < 8; i++) in.cv[i] = iv(i);
        const uint64_t key = cell(8);
        for (int i = 0; i < 4; i++) {
            words[i]     = key ? cell(4 + i) : cell(i);
            words[4 + i] = key ? cell(i)     : cell(4 + i);
        }
        in.blockLen = 64;
        in.counterLo = 0;
    } else {
        for (int i = 0; i < 8; i++) in.cv[i] = (uint32_t)cell(i);
        for (int i = 0; i < 8; i++) words[i] = cell(8 + i);
        in.blockLen = (uint32_t)cell(16);
        in.counterLo = (uint32_t)cell(17);
    }
    for (int i = 0; i < 8; i++) {
        in.block[2 * i]     = (uint32_t)(words[i] & 0xFFFFFFFFull);
        in.block[2 * i + 1] = (uint32_t)(words[i] >> 32);
    }
    return in;
}

// Everything a block owes beyond the permutation itself: the canonicity witness, the cv bytes, and
// the output feedforward with its lookups.
template <class T, class Sink>
B3_HD inline void expand_boundary_columns(T *trace, uint64_t nCols, uint64_t base,
                                    uint64_t lane, const Layout &L, Kind kind,
                                    const BlockInputs &in, const uint32_t fs[16],
                                    Sink &mul) {
    auto put = [&](uint64_t r, uint64_t c, uint64_t v) { store(trace[r * nCols + c], v); };
    constexpr uint64_t P = 0xFFFFFFFF00000001ull;

    // dinv, clocks 0..7, for the two kinds that split Goldilocks words. `hi` at clock c is
    // block[2c+1]; d = hi - (2^32 - 1).
    if (kind != Kind::Parent) {
        for (int c = 0; c < 8; c++) {
            const uint64_t lo = in.block[2 * c], hi = in.block[2 * c + 1];
            const uint64_t d = (hi + P - 0xFFFFFFFFull) % P;
            put(base + (uint64_t)c, L.dinv + lane, lo == 0 ? 0 : inv_p(d));
        }
    }

    // cv[0..4]'s bytes at clocks 0..3, aliased onto the feedforward group -- live at clocks 40..55,
    // so the two never collide. Written for EVERY kind, Node included: the feedforward below reads
    // `in.cv` for out[8..16] whatever the kind, and blake3FeedforwardAll looks those bytes up out of
    // these cells. A Node's clocks 0..3 of this group are free, which is what makes the write safe.
    for (int c = 0; c < 4; c++) {
        for (int b = 0; b < 4; b++) {
            put(base + (uint64_t)c, L.outBytes + lane * 4 + b, (in.cv[c] >> (8 * b)) & 0xFF);
        }
    }

    // The output feedforward, one word per clock over 40..55.
    //   out[i]   = st_final[i] ^ st_final[i+8]      i = 0..8
    //   out[8+i] = st_final[8+i] ^ cv[i]            i = 0..8
    for (int i = 0; i < 16; i++) {
        const uint32_t lo = fs[i];
        const uint32_t hi = i < 8 ? fs[i + 8] : in.cv[i - 8];
        const uint32_t out = lo ^ hi;
        const uint64_t row = base + (uint64_t)(CLOCKS - 16 + i);
        for (int b = 0; b < 4; b++) {
            const uint8_t la = (lo >> (8 * b)) & 0xFF, hb = (hi >> (8 * b)) & 0xFF;
            put(row, L.outBytes + lane * 4 + b, (out >> (8 * b)) & 0xFF);
            mul.table(table_row(la, hb, 0));
        }
    }

    // vb'' top bits, clocks 52..55: bits 7 of z[1..4]. z[0]'s bit is vb_pp_t, which the permutation
    // already writes on every row.
    for (int c = CLOCKS - 4; c < CLOCKS; c++) {
        const uint64_t row = base + (uint64_t)c;
        const uint64_t zcol = L.vb_pp_xor + lane * 4;
        for (int j = 1; j < 4; j++) {
            const uint64_t zb = load(trace[row * nCols + zcol + j]);
            put(row, L.vbTopHi + lane * 3 + (j - 1), (zb >> 7) & 1);
        }
    }
}

// One lane of one block: its interior, its boundary columns, and the lookups they make. A lane
// touches nothing another lane does -- its own boundary row, its own 16-word state, and column
// group `lane` of every shared array -- so this is the finest grain the reconstruction admits, and
// the GPU launches one thread per (band, lane) at it. What it cannot be split further into is the
// 56 clocks: the permutation state carries from one to the next.
template <class T, class Sink>
B3_HD inline void expand_one_lane(T *trace, uint64_t nCols, uint64_t row, uint64_t lanes,
                            uint64_t lane, Kind kind, uint64_t flags, Sink &mul) {
    const Layout L = layout(lanes);
    const BlockInputs in = read_boundary(trace, nCols, row + lane, kind, flags);
    uint32_t fs[16];
    expand_lane(trace, nCols, row, lane, L, in, mul, fs);
    expand_boundary_columns(trace, nCols, row, lane, L, kind, in, fs, mul);
}

// Expand one BLAKE3 block: every lane's interior, its boundary columns, and the lookups they make.
template <class T, class Sink>
B3_HD inline void expand_block(T *trace, uint64_t nCols, uint64_t row, uint64_t lanes,
                         Kind kind, uint64_t flags, Sink &mul) {
    for (uint64_t lane = 0; lane < lanes; lane++) {
        expand_one_lane(trace, nCols, row, lanes, lane, kind, flags, mul);
    }
}

// Write the accumulated counts into the trace's last two stage-1 columns, which is where
// blake3Tables puts mul_table and mul_range -- it is called after blake3Lanes, so they are last.
inline void write_multiplicities(Goldilocks::Element *trace, uint64_t nCols, uint64_t nRows,
                                 uint64_t lanes, const Multiplicities &mul) {
    const Layout L = layout(lanes);
    for (uint64_t i = 0; i < TABLE_SIZE && i < nRows; i++) {
        trace[i * nCols + L.mul_table] = Goldilocks::fromU64(mul.table[i]);
    }
    for (uint64_t i = 0; i < RANGE_SIZE && i < nRows; i++) {
        trace[i * nCols + L.mul_range] = Goldilocks::fromU64(mul.range[i]);
    }
}

}  // namespace blake3
}  // namespace gate_bands

#endif


