#ifndef POSEIDON2_GOLDILOCKS_WITNESS
#define POSEIDON2_GOLDILOCKS_WITNESS

// Witness implementation of the Poseidon2 circom custom gates.
//
// The gates publish their boundary only -- inputs and the permuted output. The round
// intermediates a compressor or aggregation AIR needs are recomputed from that boundary when
// the trace is filled (see pil2-stark/src/starkpil/gate_bands.hpp), so nothing here records
// them and this is exactly the permutation the prover runs: Poseidon2Goldilocks<16>, mode
// Auto, which picks the best backend compiled in.

#include <cstdint>

#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

namespace {

constexpr int WIDTH = 16;

// in[0..4] goes to 4-group slot `key`, the others shifting down. Matches the circom
// CustPoseidon2_16 `initialSt` block and poseidonInputOrder in poseidon2.pil.
inline void order_by_key(Goldilocks::Element (&state)[WIDTH], const uint64_t *in, uint64_t key)
{
    int src = 1;
    for (int g = 0; g < 4; g++) {
        const int from = (g == (int)key) ? 0 : src++;
        for (int i = 0; i < 4; i++) state[g * 4 + i] = Goldilocks::fromU64(in[from * 4 + i]);
    }
}

inline void emit(uint64_t *out, const Goldilocks::Element (&res)[WIDTH])
{
    for (int i = 0; i < WIDTH; i++) out[i] = Goldilocks::toU64(res[i]);
}

}  // namespace

void Poseidon2_16(uint64_t *out, uint *size_out, uint64_t *in, uint *size_in)
{
    Goldilocks::Element state[WIDTH], res[WIDTH];
    for (int i = 0; i < WIDTH; i++) state[i] = Goldilocks::fromU64(in[i]);
    Poseidon2Goldilocks<WIDTH>::permute(res, state, Poseidon2Mode::Auto);
    emit(out, res);
}

void CustPoseidon2_16(uint64_t *out, uint *size_out, uint64_t *in, uint *size_in,
                      uint64_t *key, uint *size_key)
{
    Goldilocks::Element state[WIDTH], res[WIDTH];
    order_by_key(state, in, (key[0] & 1) | ((key[1] & 1) << 1));
    Poseidon2Goldilocks<WIDTH>::permute(res, state, Poseidon2Mode::Auto);
    emit(out, res);
}

#endif
