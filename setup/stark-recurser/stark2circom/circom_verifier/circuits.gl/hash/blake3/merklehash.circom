pragma circom 2.1.0;
pragma custom_templates;

include "linearhash.circom";
include "merkle.circom";
include "utils.circom";
include "selectval.circom";

/*
    Blake3 merkle-hash verification. Mirrors hash/poseidon2/merklehash.circom
    template for template so the generated verifier can swap families, with the
    arity-4 sponge node hash replaced by the arity-2 Blake3 node hash.

    - eSize: Size of the extended field (usually 3 for Fp^3, or 1)
    - elementsInLinear: number of values making up one leaf
    - arity: forced to 2 for blake3
*/
template MerkleHash(eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);

    signal input values[elementsInLinear][eSize]; // Values contained in a leaf
    signal input siblings[nLevels][(arity - 1) * 4];
    signal input {binary} key[nLevels][nBits];
    signal output root[4];

    // Reduce every value in the leaf to a single digest, then walk the path.
    signal linearHash[4] <== LinearHash(elementsInLinear, arity, eSize)(values);

    root <== Merkle(arity, nLevels)(linearHash, siblings, key);
}

template VerifyMerkleHash(eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);
    signal input values[elementsInLinear][eSize];
    signal input siblings[nLevels][(arity - 1) * 4];
    signal input {binary} key[nLevels][nBits];
    signal input root[4];
    signal input {binary} enable;

    signal merkleRoot[4] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values, siblings, key);

    enable * (merkleRoot[0] - root[0]) === 0;
    enable * (merkleRoot[1] - root[1]) === 0;
    enable * (merkleRoot[2] - root[2]) === 0;
    enable * (merkleRoot[3] - root[3]) === 0;
}

template VerifyMerkleHashBatch(queries, eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);
    signal input values[queries][elementsInLinear][eSize];
    signal input siblings[queries][nLevels][(arity - 1) * 4];
    signal input {binary} key[queries][nLevels][nBits];
    signal input root[4];
    signal input {binary} enable;

    signal merkleRoot[queries][4];

    for (var i = 0; i < queries; i++) {
        merkleRoot[i] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values[i], siblings[i], key[i]);
        enable * (merkleRoot[i][0] - root[0]) === 0;
        enable * (merkleRoot[i][1] - root[1]) === 0;
        enable * (merkleRoot[i][2] - root[2]) === 0;
        enable * (merkleRoot[i][3] - root[3]) === 0;
    }
}

template VerifyMerkleHashUntilLevel(eSize, elementsInLinear, arity, nLevels, nLastLevels, height) {
    var nBits = log2(arity);
    signal input values[elementsInLinear][eSize];
    signal input siblings[nLevels][(arity - 1) * 4];
    signal input {binary} key[nLevels + nLastLevels][nBits];
    signal input last_mt_levels[arity**nLastLevels][4];
    signal input {binary} enable;

    signal {binary} keys_merkle[nLevels][nBits];
    for (var i = 0; i < nLevels; i++) {
        keys_merkle[i] <== key[i];
    }
    signal calculatedVal[4] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values, siblings, keys_merkle);

    signal last_levels_keys[nLastLevels][nBits];
    for (var i = 0; i < nLastLevels; i++) {
        for (var j = 0; j < nBits; j++) {
            last_levels_keys[i][j] <== key[nLevels + i][j];
        }
    }

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLastLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }

    signal expectedVal[4] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys);

    enable * (calculatedVal[0] - expectedVal[0]) === 0;
    enable * (calculatedVal[1] - expectedVal[1]) === 0;
    enable * (calculatedVal[2] - expectedVal[2]) === 0;
    enable * (calculatedVal[3] - expectedVal[3]) === 0;
}

template VerifyMerkleHashUntilLevelBatch(queries, eSize, elementsInLinear, arity, nLevels, nLastLevels, height) {
    var nBits = log2(arity);
    signal input values[queries][elementsInLinear][eSize];
    signal input siblings[queries][nLevels][(arity - 1) * 4];
    signal input {binary} key[queries][nLevels + nLastLevels][nBits];
    signal input last_mt_levels[arity**nLastLevels][4];
    signal input {binary} enable;

    signal {binary} keys_merkle[queries][nLevels][nBits];
    signal {binary} last_levels_keys[queries][nLastLevels][nBits];
    signal calculatedVal[queries][4];
    signal expectedVal[queries][4];

    for (var q = 0; q < queries; q++) {
        for (var i = 0; i < nLevels; i++) {
            for (var j = 0; j < nBits; j++) {
                keys_merkle[q][i][j] <== key[q][i][j];
            }
        }
        calculatedVal[q] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values[q], siblings[q], keys_merkle[q]);

        for (var i = 0; i < nLastLevels; i++) {
            for (var j = 0; j < nBits; j++) {
                last_levels_keys[q][i][j] <== key[q][nLevels + i][j];
            }
        }

        var num_nodes_level = height;
        while (num_nodes_level > arity ** nLastLevels) {
            num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
        }

        expectedVal[q] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys[q]);

        enable * (calculatedVal[q][0] - expectedVal[q][0]) === 0;
        enable * (calculatedVal[q][1] - expectedVal[q][1]) === 0;
        enable * (calculatedVal[q][2] - expectedVal[q][2]) === 0;
        enable * (calculatedVal[q][3] - expectedVal[q][3]) === 0;
    }
}

template VerifyMerkleHashUntilLevelEmpty(eSize, elementsInLinear, arity, nLastLevels, height) {
    var nBits = log2(arity);
    signal input values[elementsInLinear][eSize];
    signal input {binary} key[nLastLevels][nBits];
    signal input last_mt_levels[arity**nLastLevels][4];
    signal input {binary} enable;

    signal calculatedVal[4] <== LinearHash(elementsInLinear, arity, eSize)(values);

    signal last_levels_keys[nLastLevels][nBits];
    for (var i = 0; i < nLastLevels; i++) {
        for (var j = 0; j < nBits; j++) {
            last_levels_keys[i][j] <== key[i][j];
        }
    }

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLastLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }

    signal expectedVal[4] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys);

    enable * (calculatedVal[0] - expectedVal[0]) === 0;
    enable * (calculatedVal[1] - expectedVal[1]) === 0;
    enable * (calculatedVal[2] - expectedVal[2]) === 0;
    enable * (calculatedVal[3] - expectedVal[3]) === 0;
}

template VerifyMerkleRoot(nLevels, arity, height) {
    signal input mt_values[arity**nLevels][4];
    signal input root[4];
    signal input {binary} enable;

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }

    signal calculatedRoot[4] <== CalculateLevelMT(nLevels, arity, num_nodes_level)(mt_values);

    enable * (calculatedRoot[0] - root[0]) === 0;
    enable * (calculatedRoot[1] - root[1]) === 0;
    enable * (calculatedRoot[2] - root[2]) === 0;
    enable * (calculatedRoot[3] - root[3]) === 0;
}

/*
    Reduce a level of the merkle tree up to its root. Blake3 nodes absorb both
    children in one compression, so unlike the Poseidon version there is no
    capacity slot to route the last child into.
*/
template CalculateLevelMT(nLevels, arity, num_nodes_level) {
    assert(arity == 2);

    signal input values[arity**nLevels][4];
    signal output root[4];

    if (nLevels == 0) {
        root <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity - 1)) \ arity;
        component hashes[next_n];

        component mNext = CalculateLevelMT(nLevels - 1, arity, next_n);

        for (var j = 0; j < next_n; j++) {
            hashes[j] = Blake3Hash8();
            for (var a = 0; a < arity; a++) {
                for (var k = 0; k < 4; k++) {
                    hashes[j].in[4 * a + k] <== values[arity * j + a][k];
                }
            }
            // Tree reduction has no path key, so the identity ordering.
            hashes[j].key <== 0;
            mNext.values[j] <== hashes[j].out;
        }

        for (var k = next_n; k < arity**(nLevels - 1); k++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[k][t] <== 0;
            }
        }
        root <== mNext.root;
    }
}
