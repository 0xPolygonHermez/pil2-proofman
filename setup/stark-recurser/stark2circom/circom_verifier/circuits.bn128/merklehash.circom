pragma circom 2.1.0;

include "linearhash.circom";
include "merkle.circom";
include "utils.circom";

/*
    Given a set of leaf values, their sibling path and their key, calculate the merkle tree root 
    - eSize: Size of the extended field (usually it will be either 3 if we are in Fp³ or 1)
    - elementsInLinear: Each leave of the merkle tree is made by this number of values. 
*/
template MerkleHash(eSize, elementsInLinear, nLinears, arity) {
    var nBits = log2(nLinears);
    var logArity = log2(arity);
    var nLevels = (nBits - 1)\logArity +1;
    signal input values[elementsInLinear][eSize];
    signal input siblings[nLevels][arity]; // Sibling path to calculate the merkle root given a set of values. 
    signal input {binary} key[nBits]; // Defines either each element of the sibling path is the left or right one
    signal output root; // Root of the merkle tree

    // Each leaf in the merkle tree might be composed by multiple values. Therefore, the first step is to 
    // reduce all those values into a single one by hashing all of them
    signal linearHash <== LinearHash(elementsInLinear, eSize, arity)(values);

    // Calculate the merkle root 
    root <== Merkle(nBits, arity)(linearHash, siblings ,key);
}


/*
    Given a set of leaf values, their sibling path, their key, their merkle root and a boolean, check that the merkle tree root matches with the one sent as input
    - eSize: Size of the extended field (usually it will be either 3 if we are in Fp³ or 1)
    - elementsInLinear: Each leave of the merkle tree is made by this number of values. 
    - nLinears: Number of leaves of the merkle tree
*/
template parallel VerifyMerkleHash(eSize, elementsInLinear, nLinears, arity) {
    var nLeaves = log2(arity);
    var nBits = log2(nLinears);
    assert(1 << nBits == nLinears);
    var nLevels = (nBits - 1)\nLeaves +1;
    signal input values[elementsInLinear][eSize];
    signal input siblings[nLevels][arity]; // Sibling path to calculate the merkle root given a set of values.
    signal input {binary} key[nBits]; // Defines either each element of the sibling path is the left or right one
    signal input root; // Root of the merkle tree
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not

    // Calculate the merkle root
    signal merkleRoot <== MerkleHash(eSize, elementsInLinear, nLinears, arity)(values, siblings, key);

    // If enable is set to 1, check that the merkleRoot being calculated matches with the one sent as input
    enable * (merkleRoot - root) === 0;
}

// ── Last-level verification ──────────────────────────────────────────────────
// The prover truncates every query's sibling path nLastLevels levels before the
// root and instead publishes the whole tree level of arity**nLastLevels nodes
// once per tree. Each query checks its truncated path against the published
// level (VerifyMerkleHashUntilLevel*); the level itself is hashed up to the
// committed root once (VerifyMerkleRoot).

// Select values[key] (little-endian key bits) from the first 2**nBits entries
// of values (n >= 2**nBits). Fully constrained binary mux tree.
template SelectLastLevel(nBits, n) {
    signal input values[n];
    signal input {binary} key[nBits];
    signal output out;

    var size = 1 << nBits;
    if (nBits == 0) {
        out <== values[0];
    } else {
        signal im[size - 1];
        var levelN = size \ 2;
        var o = 0;
        var lo = 0;
        for (var i = 0; i < nBits; i++) {
            for (var j = 0; j < levelN; j++) {
                if (i == 0) {
                    im[o + j] <== key[i] * (values[2*j + 1] - values[2*j]) + values[2*j];
                } else {
                    im[o + j] <== key[i] * (im[lo + 2*j + 1] - im[lo + 2*j]) + im[lo + 2*j];
                }
            }
            lo = o;
            o = o + levelN;
            levelN = levelN \ 2;
        }
        out <== im[size - 2];
    }
}

// Verify a Merkle path that stops nLastLevels levels before the root: the
// calculated intermediate node must equal the corresponding entry of the
// published last level. The full key is passed; the first
// (nLevels - nLastLevels)*log2(arity) bits drive the truncated path and the
// remaining bits select within last_levels.
template VerifyMerkleHashUntilLevel(eSize, elementsInLinear, nLinears, arity, nLastLevels) {
    var nBits = log2(nLinears);
    var logArity = log2(arity);
    var nLevels = (nBits - 1)\logArity + 1;
    var nLevelsTrunc = nLevels - nLastLevels;
    var truncBits = nLevelsTrunc * logArity;
    var remBits = nBits - truncBits;

    signal input values[elementsInLinear][eSize];
    signal input siblings[nLevelsTrunc][arity];
    signal input {binary} key[nBits];
    signal input last_levels[arity**nLastLevels];
    signal input {binary} enable;

    signal linearHash <== LinearHash(elementsInLinear, eSize, arity)(values);

    signal {binary} keyTrunc[truncBits];
    for (var i = 0; i < truncBits; i++) {
        keyTrunc[i] <== key[i];
    }
    signal calculatedVal <== Merkle(truncBits, arity)(linearHash, siblings, keyTrunc);

    signal {binary} keyLast[remBits];
    for (var i = 0; i < remBits; i++) {
        keyLast[i] <== key[truncBits + i];
    }
    signal expectedVal <== SelectLastLevel(remBits, arity**nLastLevels)(last_levels, keyLast);

    enable * (calculatedVal - expectedVal) === 0;
}

// Degenerate case: the whole tree fits inside the published last levels
// (nLevels <= nLastLevels). The leaf hash is checked directly against the level.
template VerifyMerkleHashUntilLevelEmpty(eSize, elementsInLinear, nLinears, arity, nLastLevels) {
    var nBits = log2(nLinears);

    signal input values[elementsInLinear][eSize];
    signal input {binary} key[nBits];
    signal input last_levels[arity**nLastLevels];
    signal input {binary} enable;

    signal calculatedVal <== LinearHash(elementsInLinear, eSize, arity)(values);
    signal expectedVal <== SelectLastLevel(nBits, arity**nLastLevels)(last_levels, key);

    enable * (calculatedVal - expectedVal) === 0;
}

// Hash a published level (num_nodes_level real nodes, zero-padded to
// arity**nLastLevels) up to the root, replicating MerkleTreeBN128's padding:
// parents = ceil(n / arity), missing children are zeros.
template CalculateLastLevelsRoot(nLastLevels, arity, num_nodes_level) {
    signal input values[arity**nLastLevels];
    signal output root;

    if (nLastLevels == 0) {
        root <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity - 1)) \ arity;
        component hashes[next_n];
        component mNext = CalculateLastLevelsRoot(nLastLevels - 1, arity, next_n);

        for (var j = 0; j < next_n; j++) {
            hashes[j] = Poseidon(arity);
            for (var a = 0; a < arity; a++) {
                hashes[j].inputs[a] <== values[arity*j + a];
            }
            mNext.values[j] <== hashes[j].out;
        }
        for (var k = next_n; k < arity**(nLastLevels - 1); k++) {
            mNext.values[k] <== 0;
        }
        root <== mNext.root;
    }
}

template VerifyMerkleRoot(nLastLevels, arity, height) {
    signal input values[arity**nLastLevels];
    signal input root;
    signal input {binary} enable;

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLastLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }

    signal calculatedRoot <== CalculateLastLevelsRoot(nLastLevels, arity, num_nodes_level)(values);

    enable * (calculatedRoot - root) === 0;
}

