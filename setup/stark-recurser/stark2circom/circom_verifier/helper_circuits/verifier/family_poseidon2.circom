pragma circom 2.1.0;
pragma custom_templates;

// The GL verifier of a poseidon2 key, the one include of its generated verifier: its hash, tree
// selector and transcript files and the library, which calls them by family-neutral names.
include "hash/poseidon2/pow.circom";
include "tree/treeselector4.circom";
include "hash/poseidon2/merklehash.circom";
include "hash/poseidon2/sponge.circom";
include "transcript_sponge.circom";
include "stark_verifier.circom";
