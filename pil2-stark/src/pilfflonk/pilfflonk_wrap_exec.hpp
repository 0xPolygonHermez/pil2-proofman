#ifndef PILFFLONK_WRAP_EXEC_HPP
#define PILFFLONK_WRAP_EXEC_HPP

// The stage-1 witness of the blake3 BN128 wrap's AIR as its parts, for the device to build
// (pilfflonk_instance_new_exec, InstanceGpu): the final circuit's circom witness, the exec's
// additions and map (plonk2pil's `.exec`), and the blocks of its blake3 compressions, whose columns
// the device expands as the host does (wrap-witness/src/blake3.rs). The additions reuse the PLONK GPU
// prover's kernel (gpu_plonk_calculate_additions), the blocks the recursion's blake3 permutation
// (gate_bands_blake3.hpp). Scalars are 32 bytes, canonical, little endian.

#include <cstdint>

extern "C" {

// A block of the wrap (plonk2pil's setups/blake3_bn128/wrap.rs), at row 64·i for the i-th: its kind
// (0 Node, 1 chunk, 2 parent), flags, chaining value, sixteen message words, blockLen and counterLo,
// a Node's `over` bits, and d_inv on its input rows, canonical (the host inverts them).
struct pilfflonk_wrap_block {
    uint32_t kind;
    uint32_t flags;
    uint32_t cv[8];
    uint32_t m[16];
    uint32_t block_len;
    uint32_t counter_lo;
    uint32_t over[4];
    uint8_t dinv[8][32];
    // A Node's digest rows' d_inv: 1/C0 when over, else 1/(C1 - (2^32 - 1)) (0 where free).
    uint8_t dinv_ff[4][32];
};
// The Rust binding's layout (bindings_pilfflonk.rs).
static_assert(sizeof(struct pilfflonk_wrap_block) == 512, "pilfflonk_wrap_block changed: update its Rust binding");

// What does not change from a proof to the next, uploaded once with the key (pilfflonk_ctx_set_exec):
// addition i is wire n_wires + i = coef1·wire1 + coef2·wire2, of level level[i] (one past the
// deepest addition it reads), n_levels levels, at most 255; the map, map_cols columns of map_rows
// wires each, column after column, wire n_wires + n_adds (a zero past the additions) for a cell it
// leaves empty.
struct pilfflonk_exec_static {
    uint64_t n_wires;
    const uint32_t *add_wire1;
    const uint32_t *add_wire2;
    const uint8_t *add_coef1;
    const uint8_t *add_coef2;
    const uint8_t *add_level;
    uint64_t n_adds;
    uint64_t n_levels;
    const uint32_t *map;
    uint64_t map_rows;
    uint64_t map_cols;
};

// What a proof adds: the circom witness (n_wires scalars, wire 0 the constant one), its blocks, and
// the counts of the 16-bit table's rows the range-check rows look up, 2^16 of them, which the host
// counts and the blocks' add to.
struct pilfflonk_exec_witness {
    const uint8_t *wires;
    uint64_t n_wires;
    const struct pilfflonk_wrap_block *blocks;
    uint64_t n_blocks;
    const uint32_t *range_counts;
};

// The device side, for InstanceGpu: device pointers.

// n canonical scalars at data into Montgomery form, in place.
void pilfflonk_gpu_to_montgomery(void *data, uint64_t n);

// The blocks' columns, their lookups counted into tableCounts (2^17) and rangeCounts (2^16), zero on
// entry; then the multiplicities' columns, rangeCounts plus hostRange.
void pilfflonk_gpu_wrap_blake3(void *columns, const uint64_t *positions, uint64_t nRows, const void *blocks,
                               uint64_t nBlocks, uint32_t *tableCounts, uint32_t *rangeCounts,
                               const uint32_t *hostRange);

// The PLONK GPU prover's gather (rapidsnark/plonk_prover.cu): evalOut[i] = w(map[i]) in Montgomery
// form for i < nConstraints, w as for its additions, normal form.
void gpu_plonk_gather_witness(void *evalOut, const void *mapBuffer, const void *witness, const void *intWitness,
                              uint32_t nDirect, uint64_t nConstraints, uint64_t N);

// The PLONK GPU prover's additions (rapidsnark/plonk_prover.cu): internal[i] = f1[i]·w(id1[i]) +
// f2[i]·w(id2[i]) level by level, w(j) = witness[j] below nDirect and internal[j - nDirect] past it;
// the witness in normal form and the factors in Montgomery form, so the sums are in normal form.
void gpu_plonk_calculate_additions(void *d_buffInternalWitness, const void *d_buffWitness, const void *d_addSignalId1,
                                   const void *d_addSignalId2, const void *d_addFactor1, const void *d_addFactor2,
                                   const void *d_additionLevels, uint8_t maxLevel, uint32_t nAdditions,
                                   uint32_t nDirect);

}

#endif
