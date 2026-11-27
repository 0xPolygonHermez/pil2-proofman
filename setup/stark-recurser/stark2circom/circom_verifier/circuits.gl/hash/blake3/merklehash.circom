pragma circom 2.1.0;
pragma custom_templates;

include "linearhash.circom";
include "merkle.circom";
include "../merklehash.circom";

// The Blake3 node hash of ../merklehash.circom: both children in one compression, identity ordering.
template MerkleNode(arity) {
    assert(arity == 2);
    signal input in[arity][4];
    signal output out[4];

    component h = Blake3Hash8();
    for (var a = 0; a < arity; a++) {
        for (var k = 0; k < 4; k++) {
            h.in[4 * a + k] <== in[a][k];
        }
    }
    h.key <== 0;
    out <== h.out;
}
