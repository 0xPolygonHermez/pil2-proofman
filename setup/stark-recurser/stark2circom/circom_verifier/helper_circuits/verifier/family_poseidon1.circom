pragma circom 2.1.0;
pragma custom_templates;

// The GL verifier of a poseidon1 key, the one include of its generated verifier: its hash, tree
// selector and transcript files and the library, which calls them by family-neutral names.
include "hash/poseidon1/pow.circom";
include "tree/treeselector8.circom";
include "hash/poseidon1/merklehash.circom";
include "hash/poseidon1/sponge.circom";
include "transcript_sponge.circom";
include "stark_verifier.circom";
