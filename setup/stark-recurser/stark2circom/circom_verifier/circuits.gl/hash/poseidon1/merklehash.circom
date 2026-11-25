pragma circom 2.1.0;
pragma custom_templates;

include "linearhash.circom";
include "merkle.circom";
include "../merklehash.circom";

// The Poseidon1 node hash of ../merklehash.circom: CustPoseidon with key [0, 0] is the sponge-mode
// Poseidon1_16 of the PIL CustPoseidon1_16 gate.
template MerkleNode(arity) {
    signal input in[arity][4];
    signal output out[4];

    component h = CustPoseidon(4, 4);
    for (var a = 0; a < arity; a++) {
        for (var k = 0; k < 4; k++) {
            h.in[4 * a + k] <== in[a][k];
        }
    }
    h.key[0] <== 0;
    h.key[1] <== 0;
    out <== h.out;
}
