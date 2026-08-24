#ifndef GATE_BANDS_POSEIDON1_HPP
#define GATE_BANDS_POSEIDON1_HPP

// The Poseidon1 width-16 permutation, taking the round snapshots a band's interior needs.
//
// The witness gate (setup/circom/poseidon_gate.cpp) publishes only its boundary, so the
// snapshots are computed here instead. That makes this a second implementation of one
// function, and the two must agree or the trace is wrong:
// GateBands.Poseidon1SnapshotsMatchTheGate pins it against PoseidonGoldilocks<16>::permute,
// the same call the gate makes, over random states and all four key orderings.
//
// im layout, 12 groups of 16:
//   im[0]     R0 -- the ordered initial state, before any round constant
//   im[1..4]  full state after each of the 4 first full rounds (M, M, M, P)
//   im[5]     pre-pow7 state[0] for partial rounds 0..10 (11 cells) + 5 zero pad
//   im[6]     full state after partial round 10   (unused by the current row layout)
//   im[7]     pre-pow7 state[0] for partial rounds 11..21 (11 cells) + 5 zero pad
//   im[8]     full state after partial round 21
//   im[9..11] full state after the first 3 last full rounds

#include <cstdint>
#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"

namespace gate_bands {
namespace poseidon1 {

constexpr int WIDTH_POS1 = 16;
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

inline void perm_with_snapshots(Goldilocks::Element *state, uint64_t *im)
{
    using PC = PoseidonGoldilocksConstants::Poseidon1Tables<WIDTH_POS1>;
    uint64_t index = 0;

    // im[0] = R0, the ordered initial state before any round constant. The PIL checks
    // R0 === permuted(input, key) against it.
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

// in[0..4] goes to 4-group slot `key`, the others shifting down: the ordering
// CustPoseidon1_16 applies before permuting.
inline void order_by_key(Goldilocks::Element *state, const uint64_t *in, uint64_t key)
{
    int src = 1;
    for (int g = 0; g < 4; g++) {
        const int from = (g == (int)key) ? 0 : src++;
        for (int i = 0; i < 4; i++) state[g * 4 + i] = Goldilocks::fromU64(in[from * 4 + i]);
    }
}

// `im` receives 12 * 16 words; `out` the permuted state, which is what the gate publishes.
inline void snapshots(uint64_t *im, uint64_t *out, const uint64_t *in, uint64_t key, bool compression)
{
    Goldilocks::Element state[WIDTH_POS1];
    if (compression) {
        order_by_key(state, in, key);
    } else {
        for (int i = 0; i < WIDTH_POS1; i++) state[i] = Goldilocks::fromU64(in[i]);
    }
    perm_with_snapshots(state, im);
    for (int i = 0; i < WIDTH_POS1; i++) out[i] = Goldilocks::toU64(state[i]);
}

}  // namespace poseidon1
}  // namespace gate_bands

#endif
