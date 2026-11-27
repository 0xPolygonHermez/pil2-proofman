pragma circom 2.1.0;
pragma custom_templates;

// The GL verifier of a blake3 key, the one include of its generated verifier: its hash, tree
// selector and transcript files and the library, which calls them by family-neutral names.
// TreeSelector selects by FRI fold bits, not Merkle arity: blake3 reuses the arity-4 one.
include "hash/blake3/pow.circom";
include "tree/treeselector4.circom";
include "hash/blake3/merklehash.circom";
include "transcript_blake3.circom";
include "stark_verifier.circom";
