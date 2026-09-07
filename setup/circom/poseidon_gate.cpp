#ifndef POSEIDON_GOLDILOCKS_WITNESS
#define POSEIDON_GOLDILOCKS_WITNESS

// Witness implementation of the Poseidon1 circom custom gates.
//
// The gates publish their boundary only -- inputs and the permuted output. The round
// intermediates a compressor or aggregation AIR needs are recomputed from that boundary when
// the trace is filled (see pil2-stark/src/starkpil/gate_bands.hpp), so nothing here records
// them and this is exactly the permutation the prover runs: PoseidonGoldilocks<16>, mode
// Auto, which picks the best backend compiled in.

#include <cstdint>

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

namespace {

constexpr int WIDTH = 16;

// in[0..4] goes to 4-group slot `key`, the others shifting down. Matches the circom
// CustPoseidon1_16 `initialSt` block and the PIL poseidon1InputOrderGl helper: `key` picks
// which 4-element child hash leads.
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

// Sponge / linear-hash entry: the caller treats input as `rate || capacity` (12 || 4) per the
// sponge construction the recursive verifier's transcript and linear hash use.
void Poseidon1_16(uint64_t *out, uint *size_out, uint64_t *in, uint *size_in)
{
    Goldilocks::Element state[WIDTH], res[WIDTH];
    for (int i = 0; i < WIDTH; i++) state[i] = Goldilocks::fromU64(in[i]);
    PoseidonGoldilocks<WIDTH>::permute(res, state, PoseidonMode::Auto);
    emit(out, res);
}

// Compression entry: orders the two halves by the key bits, then permutes.
void CustPoseidon1_16(uint64_t *out, uint *size_out, uint64_t *in, uint *size_in,
                      uint64_t *key, uint *size_key)
{
    Goldilocks::Element state[WIDTH], res[WIDTH];
    order_by_key(state, in, (key[0] & 1) | ((key[1] & 1) << 1));
    PoseidonGoldilocks<WIDTH>::permute(res, state, PoseidonMode::Auto);
    emit(out, res);
}

#endif
