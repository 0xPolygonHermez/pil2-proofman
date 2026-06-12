pragma circom 2.1.0;
pragma custom_templates;

include "linearhash.circom";
include "merkle.circom";
include "utils.circom";
include "selectval.circom";

/*
    Given a leaf value and its sibling path, calculate the merkle tree root 
    - eSize: Size of the extended field (usually it will be either 3 if we are in Fp³ or 1)
    - elementsInLinear: Each leave of the merkle tree is made by this number of values. 
    - nLinears: Number of leaves of the merkle tree
*/
template MerkleHash(eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);

    signal input values[elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input siblings[nLevels][(arity - 1) * 4]; // Sibling path to calculate the merkle root given a set of values
    signal input {binary} key[nLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal output root[4]; // Root of the merkle tree

    // Each leaf in the merkle tree might be composed by multiple values. Therefore, the first step is to 
    // reduce all those values into a single one by hashing all of them
    signal linearHash[4] <== LinearHash(elementsInLinear, arity, eSize)(values);

    // Calculate the merkle root 
    root <== Merkle(arity, nLevels)(linearHash, siblings, key);
}


template VerifyMerkleHash(eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);
    signal input values[elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input siblings[nLevels][(arity - 1) * 4]; // Sibling path to calculate the merkle root given a set of values
    signal input {binary} key[nLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal input root[4]; // Root of the merkle tree
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not

    signal merkleRoot[4] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values, siblings, key);
   
    // If enable is set to 1, check that the merkleRoot being calculated matches with the one sent as input
    enable * (merkleRoot[0] - root[0]) === 0;
    enable * (merkleRoot[1] - root[1]) === 0;
    enable * (merkleRoot[2] - root[2]) === 0;
    enable * (merkleRoot[3] - root[3]) === 0;
}

template VerifyMerkleHashBatch(queries, eSize, elementsInLinear, arity, nLevels) {
    var nBits = log2(arity);
    signal input values[queries][elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input siblings[queries][nLevels][(arity - 1) * 4]; // Sibling path to calculate the merkle root given a set of values
    signal input {binary} key[queries][nLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal input root[4]; // Root of the merkle tree
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not

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
    signal input values[elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input siblings[nLevels][(arity - 1) * 4]; // Sibling path to calculate the merkle root given a set of values
    signal input {binary} key[nLevels + nLastLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal input last_mt_levels[arity**nLastLevels][4]; // The last two levels of the merkle tree, used to optimize the verification process
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not

    signal {binary} keys_merkle[nLevels][nBits];
    for (var i=0; i<nLevels; i++) {
        keys_merkle[i] <== key[i];
    }
    signal calculatedVal[4] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values, siblings, keys_merkle);
    
    signal last_levels_keys[nLastLevels][nBits];
    for (var i=0; i<nLastLevels; i++) {
        for (var j=0; j<nBits; j++) {
            last_levels_keys[i][j] <== key[nLevels + i][j];
        }
    }

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLastLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }


    signal expectedVal[4] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys);

    // If enable is set to 1, check that the expectedRoot being calculated matches with the one sent as input
    enable * (calculatedVal[0] - expectedVal[0]) === 0;
    enable * (calculatedVal[1] - expectedVal[1]) === 0;
    enable * (calculatedVal[2] - expectedVal[2]) === 0;
    enable * (calculatedVal[3] - expectedVal[3]) === 0;
}

template VerifyMerkleHashUntilLevelBatch(queries, eSize, elementsInLinear, arity, nLevels, nLastLevels, height) {
    var nBits = log2(arity);
    signal input values[queries][elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input siblings[queries][nLevels][(arity - 1) * 4]; // Sibling path to calculate the merkle root given a set of values
    signal input {binary} key[queries][nLevels + nLastLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal input last_mt_levels[arity**nLastLevels][4]; // The last two levels of the merkle tree, used to optimize the verification process
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not

    signal {binary} keys_merkle[queries][nLevels][nBits];
    signal {binary} last_levels_keys[queries][nLastLevels][nBits];
    signal calculatedVal[queries][4];
    signal expectedVal[queries][4];
    for (var q=0; q<queries; q++) {
        for (var i=0; i<nLevels; i++) {
            for (var j=0; j<nBits; j++) {
                keys_merkle[q][i][j] <== key[q][i][j];
            }
        }
        calculatedVal[q] <== MerkleHash(eSize, elementsInLinear, arity, nLevels)(values[q], siblings[q], keys_merkle[q]);
        
        
        for (var i=0; i<nLastLevels; i++) {
            for (var j=0; j<nBits; j++) {
                last_levels_keys[q][i][j] <== key[q][nLevels + i][j];
            }
        }

        var num_nodes_level = height;
        while (num_nodes_level > arity ** nLastLevels) {
            num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
        }


        expectedVal[q] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys[q]);

        // If enable is set to 1, check that the expectedRoot being calculated matches with the one sent as input
        enable * (calculatedVal[q][0] - expectedVal[q][0]) === 0;
        enable * (calculatedVal[q][1] - expectedVal[q][1]) === 0;
        enable * (calculatedVal[q][2] - expectedVal[q][2]) === 0;
        enable * (calculatedVal[q][3] - expectedVal[q][3]) === 0;
    }
}

template VerifyMerkleHashUntilLevelEmpty(eSize, elementsInLinear, arity, nLastLevels, height) {
    var nBits = log2(arity);
    signal input values[elementsInLinear][eSize]; // Values that are contained in a leaf
    signal input {binary} key[nLastLevels][nBits]; // Defines either each element of the sibling path is the left or right one
    signal input last_mt_levels[arity**nLastLevels][4]; // The last two levels of the merkle tree, used to optimize the verification process
    signal input {binary} enable; // Boolean that determines either we want to check that roots matches or not
    
    signal calculatedVal[4] <== LinearHash(elementsInLinear, arity, eSize)(values);

    signal last_levels_keys[nLastLevels][nBits];
    for (var i=0; i<nLastLevels; i++) {
        for (var j=0; j<nBits; j++) {
            last_levels_keys[i][j] <== key[i][j];
        }
    }

    var num_nodes_level = height;
    while (num_nodes_level > arity ** nLastLevels) {
        num_nodes_level = (num_nodes_level + (arity - 1)) \ arity;
    }


    signal expectedVal[4] <== SelectValue(arity, nLastLevels, num_nodes_level)(last_mt_levels, last_levels_keys);

    // If enable is set to 1, check that the expectedRoot being calculated matches with the one sent as input
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

    // If enable is set to 1, check that the merkleRoot being calculated matches with the one sent as input
    enable * (calculatedRoot[0] - root[0]) === 0;
    enable * (calculatedRoot[1] - root[1]) === 0;
    enable * (calculatedRoot[2] - root[2]) === 0;
    enable * (calculatedRoot[3] - root[3]) === 0;
}

template CalculateLevelMT(nLevels, arity, num_nodes_level) {
    signal input values[arity**nLevels][4];
    signal output root[4];

    if (nLevels == 0) {
        root <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity -  1)) \ arity;
        component hashes[next_n];

        component mNext = CalculateLevelMT(nLevels - 1, arity, next_n);
       
        for (var j = 0; j < next_n; j++) {
            hashes[j] = Poseidon2(4, 4);
            for (var a = 0; a < arity; a++) {
                for (var k = 0; k < 4; k++) {  
                    if (a == 3) {
                        hashes[j].capacity[k] <== values[arity * j + a][k];
                    } else {
                        hashes[j].in[4*a + k] <== values[arity * j + a][k];
                    }
                }
            }
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