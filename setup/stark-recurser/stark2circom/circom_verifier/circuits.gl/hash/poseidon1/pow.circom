pragma circom 2.1.0;
pragma custom_templates;

include "bitify.circom";
include "poseidon8.circom";

template VerifyPoW(powBits) {
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;

    signal hashInput[8];
    hashInput[0] <== challengeFRIQueries[0];
    hashInput[1] <== challengeFRIQueries[1];
    hashInput[2] <== challengeFRIQueries[2];
    hashInput[3] <== nonce;
    hashInput[4] <== 0;
    hashInput[5] <== 0;
    hashInput[6] <== 0;
    hashInput[7] <== 0;

    signal hashOutput[8] <== Poseidon1_8()(hashInput);

    signal bits[64] <== Num2Bits_strict()(hashOutput[0]);
    for (var i = 63; i >= 64 - powBits; i--) {
        enable * bits[i] === 0;
    }
 
    for (var i = 64 - powBits - 1; i >= 0; i--) {
        _ <== bits[i];
    }
}