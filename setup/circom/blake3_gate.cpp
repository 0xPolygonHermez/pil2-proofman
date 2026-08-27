#ifndef BLAKE3_GOLDILOCKS_CIRCOM
#define BLAKE3_GOLDILOCKS_CIRCOM

// ---------------------------------------------------------------------------
// Witness implementation of the Blake3 circom custom gates.
//
// Companion to circuits.gl/hash/blake3/blake3.circom. The gate bodies there are
// dead code under `extern_c` (the compiler replaces them with a call into this
// file), so these functions are what actually produces the witness.
//
// Two gates. Blake3Node(in[8], key) is the Merkle-node shape -- eight Goldilocks
// words hashed at the IV with the root flags, digest packed back to four words --
// and is the overwhelming majority of compressions in a verifier. Blake3Compress
// is the general one, kept for the chunk-chaining path where the chaining value
// really is a witness rather than the IV. See blake3.circom for why the node
// shape is worth its own gate.
//
// Blake3Compress(in[16], blockLen, counterLo, flags, isParent).
// Two input shapes over one row geometry, selected by `isParent`:
//
//   isParent = 0  chunk block: `in[0..8]` is the chaining value, `in[8..16]` a block of eight
//             full-range Goldilocks words, ordered by `key` and split here.
//   isParent = 1  parent node: `in` IS the sixteen u32 block words, two
//             chaining values side by side -- and the compression's own
//             chaining value is the IV. `key` is a don't-care on this path.
//
// A cv plus a Goldilocks block is exactly 16 cells, the same as a parent block,
// which is why the parent shape needs no block input of its own.
//
// counterHi is not passed: st[13] is identically zero across the design (chunk
// index capped at 2^24 by CV_STACK, XOF output block index 0 at width 8,
// parents at counter 0), so it is hardcoded rather than wired.
//
// Each gate publishes only its output words. A Blake3 AIR wanting the per-G
// round intermediates recomputes them in Rust from `in` rather than reading
// them out of the witness, so there is nothing here to emit beyond the digest
// -- which is why this calls the shared core the prover hashes with instead of
// carrying its own round loop.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "blake3_core.hpp"
#include "goldilocks_base_field.hpp"

namespace {

// Any u64 is below 2p, so one conditional subtraction is the whole reduction.
constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ull;

inline uint64_t canonical(uint64_t x) { return x >= GOLDILOCKS_P ? x - GOLDILOCKS_P : x; }

// Order the two 4-word halves by the key bit, then split each Goldilocks word
// into its (lo, hi) u32 halves -- the sixteen block words.
inline void split_ordered(uint32_t block[16], const uint64_t *lo4, const uint64_t *hi4,
                          uint64_t key)
{
    uint64_t ordered[8];
    for (int i = 0; i < 4; ++i) {
        if (key == 0) {
            ordered[i] = lo4[i];
            ordered[4 + i] = hi4[i];
        } else {
            ordered[i] = hi4[i];
            ordered[4 + i] = lo4[i];
        }
    }
    // Split the canonical representative: x = lo + 2^32*hi alone admits a second
    // solution whenever the alternative lands on x + p, and the two hash to
    // different digests. Witness values arrive canonical; reduce anyway so a
    // non-canonical one cannot pick the other split.
    for (int i = 0; i < 8; ++i) {
        const uint64_t x = canonical(ordered[i]);
        block[2 * i] = (uint32_t)x;
        block[2 * i + 1] = (uint32_t)(x >> 32);
    }
}

}  // namespace

void Blake3Node(uint64_t *out, uint *size_out,
                uint64_t *in, uint *size_in,
                uint64_t *key, uint *size_key)
{
    uint32_t cv[8];
    uint32_t block[16];
    for (int i = 0; i < 8; ++i) cv[i] = blake3core::b3_iv(i);
    split_ordered(block, in, in + 4, key[0]);

    uint32_t xof[16];
    const uint8_t flags = blake3core::FLAG_CHUNK_START | blake3core::FLAG_CHUNK_END |
                          blake3core::FLAG_ROOT;
    blake3core::compress_xof(cv, block, 64, 0, flags, xof);

    // Pack each (lo, hi) pair back to a Goldilocks digest word.
    for (int i = 0; i < 4; ++i) {
        out[i] = canonical((uint64_t)xof[2 * i] + ((uint64_t)xof[2 * i + 1] << 32));
    }
}

// `flags` and `isParent` are TEMPLATE PARAMETERS, so the fork emits them as leading scalar
// arguments of this one function -- the generated header is named after the bare template, not the
// parameterized instance, so eight (flags, isParent) pairs still share a single implementation.
// They come before the io_signals because the generator pushes `instance.arguments` first.
void Blake3Compress(uint64_t flags, uint64_t isParent,
                    uint64_t *out, uint *size_out,
                    uint64_t *in, uint *size_in,
                    uint64_t *blockLen, uint *size_blockLen,
                    uint64_t *counterLo, uint *size_counterLo)
{
    uint32_t cv[8];
    uint32_t block[16];

    if (isParent != 0) {
        for (int i = 0; i < 8; ++i) {
            cv[i] = blake3core::b3_iv(i);
            block[i] = (uint32_t)in[i];
            block[8 + i] = (uint32_t)in[8 + i];
        }
    } else {
        for (int i = 0; i < 8; ++i) cv[i] = (uint32_t)in[i];

        // No key on this gate: every caller drives the Merkle path bit through Blake3Node
        // instead, so the halves are never swapped here.
        split_ordered(block, in + 8, in + 12, 0);
    }

    uint32_t xof[16];
    blake3core::compress_xof(cv, block, (uint8_t)blockLen[0],
                             (uint64_t)counterLo[0], (uint8_t)flags, xof);
    for (int i = 0; i < 16; ++i) out[i] = xof[i];
}

#endif
