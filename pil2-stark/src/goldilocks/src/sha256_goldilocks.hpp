#ifndef SHA256_GOLDILOCKS_HPP
#define SHA256_GOLDILOCKS_HPP

// CPU SHA-256 over Goldilocks: the subset of the Poseidon interface MerkleTreeGL /
// TranscriptGL / grinding use. Forced arity 2, on the shared sha256_core so it is
// bit-identical to the GPU class (pinned by test_sha256_core_gpu.cu).
//
// CALLERS MUST RESPECT: blake3 hashes leaves and nodes with the same function, so
// every call site there spells the node hash `linearHash`. Here they differ --
// leaves are `linearHash` (FIPS), nodes are `nodeHash` (domain-separated). The
// three sites needing nodeHash are MerkleTreeGL::merkelize,
// ::calculateRootFromProof (the VERIFIER's) and ::verifyMerkleRoot.

#include <cstdint>
#include "goldilocks_base_field.hpp"

#ifndef HASH_SIZE
#define HASH_SIZE 4
#endif

class Sha256Goldilocks
{
public:
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t ARITY    = 2;

    /// LEAF: literal FIPS 180-4 over `size` words.
    static void linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);

    /// INTERNAL NODE: `size` must be a positive multiple of 8 (arity * CAPACITY).
    /// Domain-separated from linearHash, and pays no padding block.
    static void nodeHash(Goldilocks::Element *output, const Goldilocks::Element *input, uint64_t size);

    static void permuteTrunc(Goldilocks::Element (&output)[CAPACITY], const Goldilocks::Element (&input)[8]);

    /// Grinding. SHA-256's digest is FOUR words, so output[0..4] carries it and
    /// output[4..8] is cleared, as `Blake3_8::hash` does in Rust. The one caller
    /// (stark_verify.hpp) reads only output[0].
    static void permute(Goldilocks::Element (&output)[8], const Goldilocks::Element (&input)[8]);

    static void merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                           uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                           int num_threads = 0, uint64_t dim = 1);

    static void merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                 uint64_t num_elements, uint64_t arity);

    static void grinding(uint64_t &nonce, const uint64_t *in, const uint32_t n_bits);
};

#endif  // SHA256_GOLDILOCKS_HPP
