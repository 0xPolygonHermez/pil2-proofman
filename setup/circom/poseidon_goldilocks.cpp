#ifndef POSEIDON_GOLDILOCKS_WITNESS
#define POSEIDON_GOLDILOCKS_WITNESS

// im[11][16] layout produced by the Poseidon1 width-16 permutation. Used by
// both Poseidon1_16 (sponge / linear hash) and CustPoseidon1_16 — only the output extraction differs at the end.
//
//   im[0]     : R0 — the ordered initial state (before any round constant)
//   im[1..4]  : full state after each of 4 first full rounds (M, M, M, P)
//   im[5]     : pre-pow7 state[0] for partial rounds  0..10 (11 cells) + 5 zero pad
//   im[6]     : full state after partial round 10
//   im[7]     : pre-pow7 state[0] for partial rounds 11..21 (11 cells) + 5 zero pad
//   im[8]     : full state after partial round 21
//   im[9..11] : full state after first 3 last full rounds

#include <vector>
#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#ifdef __AVX2__
    #include <immintrin.h>
#endif

namespace {

constexpr int WIDTH_POS1 = 16;
constexpr int N_AVX_VECS_POS1 = WIDTH_POS1 / 4;       // 4 AVX vectors at width 16
constexpr int HALF_FULL_ROUNDS_POS1 = 4;
constexpr int N_PARTIAL_ROUNDS_POS1 = 22;
constexpr int S_STRIDE_POS1 = 2 * WIDTH_POS1 - 1;     // 31 entries per partial round

inline void pow7_pos1(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
}

inline void add_state_pos1(Goldilocks::Element *state, const Goldilocks::Element *C)
{
    for (int i = 0; i < WIDTH_POS1; i++) state[i] = state[i] + C[i];
}

inline void pow7add_state_pos1(Goldilocks::Element *state, const Goldilocks::Element *C)
{
    for (int i = 0; i < WIDTH_POS1; i++) {
        pow7_pos1(state[i]);
        state[i] = state[i] + C[i];
    }
}

inline void pow7_all_pos1(Goldilocks::Element *state)
{
    for (int i = 0; i < WIDTH_POS1; i++) pow7_pos1(state[i]);
}

inline Goldilocks::Element dot_state_pos1(const Goldilocks::Element *state, const Goldilocks::Element *C)
{
    Goldilocks::Element s = state[0] * C[0];
    for (int i = 1; i < WIDTH_POS1; i++) s = s + state[i] * C[i];
    return s;
}

inline void mvp_state_pos1(Goldilocks::Element *state,
                           const Goldilocks::Element mat[WIDTH_POS1][WIDTH_POS1])
{
    Goldilocks::Element old[WIDTH_POS1];
    for (int i = 0; i < WIDTH_POS1; i++) old[i] = state[i];
    for (int i = 0; i < WIDTH_POS1; i++) {
        state[i] = mat[0][i] * old[0];
        for (int j = 1; j < WIDTH_POS1; j++)
            state[i] = state[i] + mat[j][i] * old[j];
    }
}

inline void snapshot_pos1(uint64_t *im, uint64_t &index, const Goldilocks::Element *state)
{
    for (int i = 0; i < WIDTH_POS1; i++) im[index++] = Goldilocks::toU64(state[i]);
}

void Poseidon1_perm_seq(Goldilocks::Element *state, uint64_t *im)
{
    using PC = PoseidonGoldilocksConstants::Poseidon1Tables<WIDTH_POS1>;
    uint64_t index = 0;

    // im[0] = R0 (the ordered initial state, before any round constant). Emitted
    // so the trace mapping can wire R0 from a real circom signal — matches the
    // PIL input-ordering check R0 === permuted(input, key) and Poseidon2's im[0].
    snapshot_pos1(im, index, state);

    add_state_pos1(state, &PC::C[0]);

    for (int r = 0; r < HALF_FULL_ROUNDS_POS1 - 1; r++) {
        pow7add_state_pos1(state, &PC::C[(r + 1) * WIDTH_POS1]);
        mvp_state_pos1(state, PC::M);
        snapshot_pos1(im, index, state);
    }
    pow7add_state_pos1(state, &PC::C[HALF_FULL_ROUNDS_POS1 * WIDTH_POS1]);
    mvp_state_pos1(state, PC::P);
    snapshot_pos1(im, index, state);

    for (int r = 0; r < N_PARTIAL_ROUNDS_POS1; r++) {
        im[index++] = Goldilocks::toU64(state[0]);

        pow7_pos1(state[0]);
        state[0] = state[0] + PC::C[(HALF_FULL_ROUNDS_POS1 + 1) * WIDTH_POS1 + r];

        const Goldilocks::Element *Sr = &PC::S[S_STRIDE_POS1 * r];
        Goldilocks::Element s0 = dot_state_pos1(state, Sr);
        const Goldilocks::Element *Sr_w = Sr + (WIDTH_POS1 - 1);
        for (int t = 1; t < WIDTH_POS1; t++) {
            state[t] = state[t] + state[0] * Sr_w[t];
        }
        state[0] = s0;

        if (r == 10 || r == 21) {
            for (int i = 11; i < WIDTH_POS1; i++) im[index++] = 0;
            snapshot_pos1(im, index, state);
        }
    }

    for (int r = 0; r < HALF_FULL_ROUNDS_POS1 - 1; r++) {
        pow7add_state_pos1(state,
            &PC::C[(HALF_FULL_ROUNDS_POS1 + 1) * WIDTH_POS1 + N_PARTIAL_ROUNDS_POS1
                   + r * WIDTH_POS1]);
        mvp_state_pos1(state, PC::M);
        snapshot_pos1(im, index, state);
    }
    pow7_all_pos1(state);
    mvp_state_pos1(state, PC::M);
}

#ifdef __AVX2__

inline void pow7_avx_pos1(__m256i st[N_AVX_VECS_POS1])
{
    __m256i pw2[N_AVX_VECS_POS1], pw3[N_AVX_VECS_POS1], pw4[N_AVX_VECS_POS1];
    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::square_avx(pw2[i], st[i]);
    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::square_avx(pw4[i], pw2[i]);
    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::mult_avx(pw3[i], pw2[i], st[i]);
    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::mult_avx(st[i], pw3[i], pw4[i]);
}

inline void add_avx_pos1(__m256i st[N_AVX_VECS_POS1], const Goldilocks::Element C[WIDTH_POS1])
{
    __m256i c;
    for (int i = 0; i < N_AVX_VECS_POS1; i++) {
        Goldilocks::load_avx(c, &C[4 * i]);
        Goldilocks::add_avx(st[i], st[i], c);
    }
}

inline void add_avx_small_pos1(__m256i st[N_AVX_VECS_POS1], const Goldilocks::Element C_small[WIDTH_POS1])
{
    __m256i c;
    for (int i = 0; i < N_AVX_VECS_POS1; i++) {
        Goldilocks::load_avx(c, &C_small[4 * i]);
        Goldilocks::add_avx_b_small(st[i], st[i], c);
    }
}

// state[i] = sum_{j=0..15} mat[j][i] * old[j]
// Broadcast each old[j], multiply by the j-th row of mat (loaded as 4 AVX
// vectors), accumulate into the output. Row-major access — no transpose.
inline void mmult_avx_pos1(__m256i st[N_AVX_VECS_POS1],
                           const Goldilocks::Element mat[WIDTH_POS1][WIDTH_POS1])
{
    Goldilocks::Element old[WIDTH_POS1];
    for (int i = 0; i < N_AVX_VECS_POS1; i++) {
        Goldilocks::store_avx(&old[4 * i], st[i]);
    }

    __m256i acc[N_AVX_VECS_POS1];
    __m256i row;

    // Initialize with j = 0
    __m256i s = _mm256_set1_epi64x(old[0].fe);
    for (int k = 0; k < N_AVX_VECS_POS1; k++) {
        Goldilocks::load_avx(row, &mat[0][4 * k]);
        Goldilocks::mult_avx(acc[k], s, row);
    }

    // Accumulate j = 1..15
    for (int j = 1; j < WIDTH_POS1; j++) {
        s = _mm256_set1_epi64x(old[j].fe);
        for (int k = 0; k < N_AVX_VECS_POS1; k++) {
            Goldilocks::load_avx(row, &mat[j][4 * k]);
            __m256i prod;
            Goldilocks::mult_avx(prod, s, row);
            Goldilocks::add_avx(acc[k], acc[k], prod);
        }
    }

    for (int k = 0; k < N_AVX_VECS_POS1; k++) st[k] = acc[k];
}

// Returns sum_{i=0..15} st[i] * b[i] as a scalar Goldilocks element.
inline Goldilocks::Element dot_avx_pos1(const __m256i st[N_AVX_VECS_POS1],
                                       const Goldilocks::Element b[WIDTH_POS1])
{
    __m256i bv, c[N_AVX_VECS_POS1];
    for (int i = 0; i < N_AVX_VECS_POS1; i++) {
        Goldilocks::load_avx(bv, &b[4 * i]);
        Goldilocks::mult_avx(c[i], st[i], bv);
    }
    __m256i c01, c23, c_total;
    Goldilocks::add_avx(c01, c[0], c[1]);
    Goldilocks::add_avx(c23, c[2], c[3]);
    Goldilocks::add_avx(c_total, c01, c23);

    Goldilocks::Element lanes[4];
    Goldilocks::store_avx(lanes, c_total);
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}

inline void snapshot_avx_pos1(uint64_t *im, uint64_t &index, Goldilocks::Element *scratch,
                              const __m256i st[N_AVX_VECS_POS1])
{
    for (int i = 0; i < N_AVX_VECS_POS1; i++) {
        Goldilocks::store_avx(&scratch[4 * i], st[i]);
    }
    for (int i = 0; i < WIDTH_POS1; i++) im[index++] = Goldilocks::toU64(scratch[i]);
}

void Poseidon1_perm_avx(Goldilocks::Element *state, uint64_t *im)
{
    using PC = PoseidonGoldilocksConstants::Poseidon1Tables<WIDTH_POS1>;
    uint64_t index = 0;
    Goldilocks::Element scratch[WIDTH_POS1];

    // im[0] = R0 (the ordered initial state, before any round constant). Emitted
    // so the trace mapping can wire R0 from a real circom signal — matches the
    // PIL input-ordering check R0 === permuted(input, key) and Poseidon2's im[0].
    for (int i = 0; i < WIDTH_POS1; i++) im[index++] = Goldilocks::toU64(state[i]);

    __m256i st[N_AVX_VECS_POS1];
    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::load_avx(st[i], &state[4 * i]);

    add_avx_small_pos1(st, &PC::C[0]);

    // First (HALF_FULL_ROUNDS - 1) full rounds with M-matrix.
    for (int r = 0; r < HALF_FULL_ROUNDS_POS1 - 1; r++) {
        pow7_avx_pos1(st);
        add_avx_small_pos1(st, &PC::C[(r + 1) * WIDTH_POS1]);
        mmult_avx_pos1(st, PC::M);
        snapshot_avx_pos1(im, index, scratch, st);
    }
    // Transition full round: post-ARC then P-matrix instead of M.
    pow7_avx_pos1(st);
    add_avx_pos1(st, &PC::C[HALF_FULL_ROUNDS_POS1 * WIDTH_POS1]);
    mmult_avx_pos1(st, PC::P);
    snapshot_avx_pos1(im, index, scratch, st);

    // Partial rounds: state[0] lives as a scalar (state0_) across iters; AVX
    // st[0..3] carries state[1..15]. Lane 0 of st[0] is junk — zeroed via mask
    // before dot_avx_pos1 and overwritten on exit.
    Goldilocks::store_avx(&scratch[0], st[0]);
    Goldilocks::Element state0_ = scratch[0];
    const __m256i lane0_mask = _mm256_set_epi64x(
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0);

    for (int r = 0; r < N_PARTIAL_ROUNDS_POS1; r++) {
        im[index++] = Goldilocks::toU64(state0_);

        Goldilocks::Element state0 = state0_;
        pow7_pos1(state0);
        state0 = state0 + PC::C[(HALF_FULL_ROUNDS_POS1 + 1) * WIDTH_POS1 + r];

        const Goldilocks::Element *Sr = &PC::S[S_STRIDE_POS1 * r];
        state0_ = state0 * Sr[0];
        st[0] = _mm256_and_si256(st[0], lane0_mask);
        state0_ = state0_ + dot_avx_pos1(st, Sr);

        // state[t] += state0 * Sr_w[t] for t in 1..15.
        // Sr_w[0] aligns with lane 0 of st[0] (= state[0] slot, masked); the
        // post-update junk written there is overwritten before the next dot.
        const Goldilocks::Element *Sr_w = &PC::S[S_STRIDE_POS1 * r + WIDTH_POS1 - 1];
        __m256i scalar1 = _mm256_set1_epi64x(state0.fe);
        __m256i sv, w;
        for (int i = 0; i < N_AVX_VECS_POS1; i++) {
            Goldilocks::load_avx(sv, &Sr_w[4 * i]);
            Goldilocks::mult_avx(w, scalar1, sv);
            Goldilocks::add_avx(st[i], st[i], w);
        }

        if (r == 10 || r == 21) {
            for (int i = 11; i < WIDTH_POS1; i++) im[index++] = 0;
            for (int i = 0; i < N_AVX_VECS_POS1; i++) {
                Goldilocks::store_avx(&scratch[4 * i], st[i]);
            }
            scratch[0] = state0_;
            for (int i = 0; i < WIDTH_POS1; i++) im[index++] = Goldilocks::toU64(scratch[i]);
        }
    }

    // Reinsert state0_ back into st[0] lane 0 for the final full rounds.
    Goldilocks::store_avx(&scratch[0], st[0]);
    scratch[0] = state0_;
    Goldilocks::load_avx(st[0], &scratch[0]);

    // Last (HALF_FULL_ROUNDS - 1) full rounds with M-matrix and post-ARC.
    for (int r = 0; r < HALF_FULL_ROUNDS_POS1 - 1; r++) {
        pow7_avx_pos1(st);
        add_avx_small_pos1(st,
            &PC::C[(HALF_FULL_ROUNDS_POS1 + 1) * WIDTH_POS1 + N_PARTIAL_ROUNDS_POS1
                   + r * WIDTH_POS1]);
        mmult_avx_pos1(st, PC::M);
        snapshot_avx_pos1(im, index, scratch, st);
    }
    // Final full round: pow7 + M, no post-ARC.
    pow7_avx_pos1(st);
    mmult_avx_pos1(st, PC::M);

    for (int i = 0; i < N_AVX_VECS_POS1; i++) Goldilocks::store_avx(&state[4 * i], st[i]);
}

#endif // __AVX2__

inline void Poseidon1_perm(Goldilocks::Element *state, uint64_t *im)
{
#ifdef __AVX2__
    Poseidon1_perm_avx(state, im);
#else
    Poseidon1_perm_seq(state, im);
#endif
}

} // anonymous namespace

// Sponge / linear-hash entry: applies the Poseidon1 width-16 permutation and
// writes the full permuted state to `out`. The caller treats input as
// `rate || capacity` (typically 12 || 4) per the sponge construction used by
// the recursive verifier's transcript and linear hash.
void Poseidon1_16(uint64_t *im, uint *size_im, uint64_t *out, uint *size_out,
                  uint64_t *in, uint *size_in)
{
    Goldilocks::Element state[WIDTH_POS1];
    for (int i = 0; i < WIDTH_POS1; i++) {
        state[i] = Goldilocks::fromU64(in[i]);
    }
    Poseidon1_perm(state, im);
    for (int i = 0; i < WIDTH_POS1; i++) out[i] = Goldilocks::toU64(state[i]);
}

// Compression entry: applies the Poseidon1 width-16 permutation to the
// key-reordered inputs and writes the PLAIN permuted state to `out`. This matches the circom CustPoseidon1_16 gate, which
// extracts `out[t] <-- st[t]`, and the PIL constraints, which contain no
// `+ in` feedback. Merkle-node compression: 16 input field elements (four
// 4-element child hashes) -> 16-element output, of which the consumer truncates
// the first 4.
void CustPoseidon1_16(uint64_t *im, uint *size_im, uint64_t *out, uint *size_out,
                      uint64_t *im_m, uint *size_im_m,
                      uint64_t *in, uint *size_in, uint64_t *key, uint *size_key)
{
    // One-hot encoding of the key bits, consumed by the AIR's input-ordering constraint as a
    // stored degree-1 value. Derived from the same branch selection used for the permutation
    // below rather than by indexing on key[] arithmetic: this is an extern_c gate, so the
    // circom body's `assert(key[i]*(key[i]-1)==0)` never runs, and a malformed key would make
    // im_m[key[0] + 2*key[1]] an out-of-bounds write. The final branch mirrors the
    // permutation's trailing `else`, so mask and reordering can never disagree.
    int key_idx = 3;
    if (key[0] == 0 && key[1] == 0) {
        key_idx = 0;
    } else if (key[0] == 1 && key[1] == 0) {
        key_idx = 1;
    } else if (key[0] == 0 && key[1] == 1) {
        key_idx = 2;
    }
    for (int i = 0; i < 4; i++) im_m[i] = (i == key_idx) ? 1 : 0;

    // Order the inputs according to the two key bits before permuting — matches
    // the circom CustPoseidon1_16 `initialSt` block and the PIL
    // poseidon1InputOrderGl helper. key picks which 4-element child hash leads.
    Goldilocks::Element state[WIDTH_POS1];
    if (key[0] == 0 && key[1] == 0) {
        for (int i = 0; i < WIDTH_POS1; i++) state[i] = Goldilocks::fromU64(in[i]);
    } else if (key[0] == 1 && key[1] == 0) {
        for (int i = 0; i < 4; i++)  state[i]      = Goldilocks::fromU64(in[4 + i]);
        for (int i = 0; i < 4; i++)  state[4 + i]  = Goldilocks::fromU64(in[i]);
        for (int i = 8; i < 16; i++) state[i]      = Goldilocks::fromU64(in[i]);
    } else if (key[0] == 0 && key[1] == 1) {
        for (int i = 0; i < 4; i++)  state[i]      = Goldilocks::fromU64(in[4 + i]);
        for (int i = 0; i < 4; i++)  state[4 + i]  = Goldilocks::fromU64(in[8 + i]);
        for (int i = 0; i < 4; i++)  state[8 + i]  = Goldilocks::fromU64(in[i]);
        for (int i = 12; i < 16; i++) state[i]     = Goldilocks::fromU64(in[i]);
    } else {
        for (int i = 0; i < 12; i++) state[i]      = Goldilocks::fromU64(in[4 + i]);
        for (int i = 0; i < 4; i++)  state[12 + i] = Goldilocks::fromU64(in[i]);
    }


    Poseidon1_perm(state, im);
    for (int i = 0; i < WIDTH_POS1; i++) out[i] = Goldilocks::toU64(state[i]);
}

#endif
