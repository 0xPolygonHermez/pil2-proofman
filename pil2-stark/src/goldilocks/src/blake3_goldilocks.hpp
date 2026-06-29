#ifndef BLAKE3_GOLDILOCKS_HPP
#define BLAKE3_GOLDILOCKS_HPP

// ---------------------------------------------------------------------------
// Blake3Goldilocks -- CPU BLAKE3 hashing over Goldilocks, exposing the subset
// of the Poseidon interface used by MerkleTreeGL / TranscriptGL / grinding.
// Forced arity 2 (binary tree). Built on the shared blake3_core so it is
// bit-identical to the GPU prover.
// ---------------------------------------------------------------------------

#include <cstdint>
#include "goldilocks_base_field.hpp"

#ifndef HASH_SIZE
#define HASH_SIZE 4
#endif

class Blake3Goldilocks
{
public:
    static constexpr uint32_t CAPACITY = 4;

    // Leaf / linear hash: hash `size` field elements into a 4-element digest.
    static void linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);

    // Merkle node (arity 2): hash 8 field elements (two child digests) -> 4.
    static void permuteTrunc(Goldilocks::Element (&output)[CAPACITY], const Goldilocks::Element (&input)[8]);

    // Grinding permutation: 8 -> 8 (64-byte BLAKE3 XOF).
    static void permute(Goldilocks::Element (&output)[8], const Goldilocks::Element (&input)[8]);

    // Transcript permutation of arbitrary width (= 4*arity: 8/12/16): a BLAKE3
    // XOF over `width` words squeezing `width` words. Drop-in for the Poseidon
    // sponge permute used by TranscriptGL.
    static void permuteTranscript(Goldilocks::Element *output, const Goldilocks::Element *input, uint64_t width);

    // Full Merkle tree: leaf hash each row, then arity-ary reduction.
    static void merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                           uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                           int num_threads = 0, uint64_t dim = 1);

    // Reduce num_elements leaf digests up to a single 4-element root.
    static void merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                 uint64_t num_elements, uint64_t arity);

    // Proof-of-work: find a nonce so that permute([challenge,nonce,0..])[0] has
    // n_bits leading zero bits (same contract as Poseidon grinding).
    static void grinding(uint64_t &nonce, const uint64_t *in, const uint32_t n_bits);
};

#endif  // BLAKE3_GOLDILOCKS_HPP
