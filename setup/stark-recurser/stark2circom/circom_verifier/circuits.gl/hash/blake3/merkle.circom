pragma circom 2.1.0;
pragma custom_templates;

include "blake3.circom";
include "utils.circom";

/*
    Given a value and its sibling path (with the key bits saying whether the
    node is the left or right child at each level), calculate the merkle root.

    Blake3 forces arity 2, so each level has one sibling and one key bit. The
    ordering happens inside the gate, exactly as CustPoseidon2_16 does it: the
    AIR folds the key into its block binding as a masked combination, which
    costs a degree bump instead of the ~8 plonk rows per level that circom
    muxes would spend.
*/
template Merkle(arity, nLevels) {
    assert(arity == 2);
    var nBits = log2(arity);

    signal input value[4];                          // Leaf value
    signal input siblings[nLevels][(arity - 1) * 4]; // Sibling values
    signal input {binary} key[nLevels][nBits];
    signal output root[4];

    component hash[nLevels];

    for (var i = 0; i < nLevels; i++) {
        // Hash the corresponding value with the corresponding sibling path value, which
        // are 4 GL elements each, using the arity-2 Blake3 node hash. Returns a 4 GL
        // element output.
        // The key that determines which element is the left one and which one the right
        // one is also sent to the custom gate, rather than muxed here.
        hash[i] = Blake3Hash8();
        for (var k = 0; k < 4; k++) {
            if (i > 0) {
                hash[i].in[k] <== hash[i-1].out[k];
            } else {
                hash[i].in[k] <== value[k];
            }
        }
        for (var k = 0; k < (arity - 1) * 4; k++) {
            hash[i].in[k+4] <== siblings[i][k];
        }

        hash[i].key <== key[i][0];
    }

    root <== hash[nLevels-1].out;
}
