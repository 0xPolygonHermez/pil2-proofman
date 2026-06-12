pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2.circom";

template LinearHash(nInputs, arity, eSize) {

    signal input in[nInputs][eSize];
    signal output out[4];

    var nHashes = (nInputs*eSize - 1)\((arity - 1)*4) +1;

    component hash[nHashes];

    var curInput=0;
    var curC=0;

    for (var i=0; i<nHashes; i++) {
        hash[i] = Poseidon2(arity, 4);
        for (var k=0; k<(arity - 1) * 4; k++) {
            if (curInput<nInputs) {
                hash[i].in[k] <== in[curInput][curC];
                curC++;
                if (curC == eSize) {
                    curC =0;
                    curInput += 1;
                }
            } else {
                hash[i].in[k] <== 0;
            }
        }
        for (var k=0; k<4; k++) {
            if (i>0) {
                hash[i].capacity[k] <== hash[i-1].out[k];
            } else {
                hash[i].capacity[k] <== 0;
            }
        }
    }

    for (var k=0; k<4; k++) {
        out[k] <== hash[nHashes-1].out[k];
    }
}



