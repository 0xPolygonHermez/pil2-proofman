#ifndef GATE_BANDS_POSEIDON2_HPP
#define GATE_BANDS_POSEIDON2_HPP

// The Poseidon2 width-16 permutation, taking the round snapshots a band's interior needs.
//
// Companion to gate_bands_poseidon1.hpp; see that file for why the expander carries its own
// permutation, and for the im layout, which is the same 12 groups of 16.
// GateBands.Poseidon2SnapshotsMatchTheGate pins this against Poseidon2Goldilocks<16>::permute,
// the same call the gate makes, over random states and all four key orderings.

#include <cstdint>
#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"

namespace gate_bands {
namespace poseidon2 {

constexpr int W = 16;

inline void p2_pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};

inline void p2_add_(Goldilocks::Element &x, const Goldilocks::Element *st)
{
    for (int i = 0; i < W; ++i)
    {
        x = x + st[i];
    }
}
inline void prodp2_add_(Goldilocks::Element *x, const Goldilocks::Element *D, const Goldilocks::Element &sum)
{
    for (int i = 0; i < W; ++i)
    {
        x[i] = x[i]*D[i] + sum;
    }
}

inline void pow7p2_add_(Goldilocks::Element *x, const Goldilocks::Element *C)
{
    Goldilocks::Element x2[W], x3[W], x4[W];
    
    for (int i = 0; i < W; ++i)
    {
        Goldilocks::Element xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

inline void p2_matmul_m4_(Goldilocks::Element *x) {
    Goldilocks::Element t0 = x[0] + x[1];
    Goldilocks::Element t1 = x[2] + x[3];
    Goldilocks::Element t2 = x[1] + x[1] + t1;
    Goldilocks::Element t3 = x[3] + x[3] + t0;
    Goldilocks::Element t1_2 = t1 + t1;
    Goldilocks::Element t0_2 = t0 + t0;
    Goldilocks::Element t4 = t1_2 + t1_2 + t3;
    Goldilocks::Element t5 = t0_2 + t0_2 + t2;
    Goldilocks::Element t6 = t3 + t5;
    Goldilocks::Element t7 = t2 + t4;
    
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

inline void p2_matmul_external_(Goldilocks::Element *x) {
    for (int i = 0; i < W/4; ++i) {
        p2_matmul_m4_(&x[i*4]);
    }
    
    Goldilocks::Element stored[4] = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};

    for(int i = 0; i < 4; ++i) {
        for (int j = 0; j < W/4; ++j) {
            stored[i] = stored[i] + x[j*4 + i];
        }
    }
    
    for (int i = 0; i < W; ++i)
    {
        x[i] = x[i] + stored[i % 4];
    }
}

// Snapshot-taking permutation. `im` receives 12 * 16 words.
inline void perm_with_snapshots(Goldilocks::Element *state, uint64_t *im)
{
    const Goldilocks::Element *RC = Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element *D = Poseidon2GoldilocksConstants::D16;
    uint64_t index = 0;

    p2_matmul_external_(state);
    
    for(uint64_t i = 0; i < W; ++i) {
        im[index++] = Goldilocks::toU64(state[i]);
    }

    for (int r = 0; r < 4; r++)
    {
        pow7p2_add_(state, &(RC[W * r]));
        p2_matmul_external_(state);
        for(uint64_t i = 0; i < W; ++i) {
            im[index++] = Goldilocks::toU64(state[i]);
        }
    }

    for (int r = 0; r < 22; r++)
    {
        im[index++] = Goldilocks::toU64(state[0]);
        state[0] = state[0] + RC[4 * W + r];
        p2_pow7(state[0]);
        Goldilocks::Element sum_ = Goldilocks::zero();
        p2_add_(sum_, state);
        prodp2_add_(state, D, sum_);
        if (r == 10 || r == 21) {
            for (int i = 11; i < W; i++) {
                im[index++] = 0;
            }
            for (int i = 0; i < W; i++) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }

    for (int r = 0; r < 4; r++)
    {
        pow7p2_add_(state, &(RC[4 * W + 22 + r * W]));
        p2_matmul_external_(state);
        if(r < 3) {
            for(uint64_t i = 0; i < W; ++i) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }
}

// in[0..4] goes to 4-group slot `key`, the others shifting down -- the ordering
// CustPoseidon2_16 applies before permuting.
inline void order_by_key(Goldilocks::Element *state, const uint64_t *in, uint64_t key)
{
    int src = 1;
    for (int g = 0; g < 4; g++) {
        const int from = (g == (int)key) ? 0 : src++;
        for (int i = 0; i < 4; i++) state[g * 4 + i] = Goldilocks::fromU64(in[from * 4 + i]);
    }
}

inline void snapshots(uint64_t *im, uint64_t *out, const uint64_t *in, uint64_t key, bool compression)
{
    Goldilocks::Element state[W];
    if (compression) {
        order_by_key(state, in, key);
    } else {
        for (int i = 0; i < W; i++) state[i] = Goldilocks::fromU64(in[i]);
    }
    perm_with_snapshots(state, im);
    for (int i = 0; i < W; i++) out[i] = Goldilocks::toU64(state[i]);
}

}  // namespace poseidon2
}  // namespace gate_bands

#endif
