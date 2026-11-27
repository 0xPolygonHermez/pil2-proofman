pragma circom 2.1.0;
pragma custom_templates;

include "linearhash.circom";
include "merkle.circom";
include "../merklehash.circom";

// The Poseidon2 node hash of ../merklehash.circom: three children in the rate, the fourth in the capacity.
template MerkleNode(arity) {
    signal input in[arity][4];
    signal output out[4];

    component h = Poseidon2(4, 4);
    for (var a = 0; a < arity; a++) {
        for (var k = 0; k < 4; k++) {
            if (a == 3) {
                h.capacity[k] <== in[a][k];
            } else {
                h.in[4 * a + k] <== in[a][k];
            }
        }
    }
    out <== h.out;
}
