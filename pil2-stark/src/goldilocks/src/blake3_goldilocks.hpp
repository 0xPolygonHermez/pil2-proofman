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

    static void linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);

    static void permuteTrunc(Goldilocks::Element (&output)[CAPACITY], const Goldilocks::Element (&input)[8]);

    static void permute(Goldilocks::Element (&output)[8], const Goldilocks::Element (&input)[8]);

    static void permuteTranscript(Goldilocks::Element *output, const Goldilocks::Element *input, uint64_t width);

    static void merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                           uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                           int num_threads = 0, uint64_t dim = 1);

    static void merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                 uint64_t num_elements, uint64_t arity);

    static void grinding(uint64_t &nonce, const uint64_t *in, const uint32_t n_bits);
};

#endif  // BLAKE3_GOLDILOCKS_HPP
