pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2.circom";

/*
    Given a value and its sibling path (with its corresponding vector determining if the element is the left or right one), 
    calculate its merkle root.
*/
template Merkle(arity, nLevels) {
    var nBits = log2(arity);

    signal input value[4]; // Leaf value
    signal input siblings[nLevels][(arity - 1) * 4]; // Sibling values
    signal input {binary} key[nLevels][nBits];
    signal output root[4];

    component hash[nLevels];

    for (var i=0; i<nLevels; i++) {
        // Hash the corresponding value with the corresponding sibling path value, which 
        // are 4 GL elements each, using Goldilocks Poseidon. Returns a 4 GL element output.
        // Therefore, a 2 inputs Poseidon is being performed.
        // The key that determines which element is the left one and which one the right one is also
        // sent to the custom gate
        hash[i] = CustPoseidon2(arity, 4);
        for (var k = 0; k<4; k++) {
            if (i>0) {
                hash[i].in[k] <== hash[i-1].out[k];
            } else {
                hash[i].in[k] <== value[k];
            }
        }
        for (var k=0; k<(arity - 1) * 4; k++) {
            hash[i].in[k+4] <== siblings[i][k];
        }

        hash[i].key <== key[i];
    }

    root <== hash[nLevels-1].out;
}

