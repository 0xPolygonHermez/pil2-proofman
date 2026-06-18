pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2_8.circom";
include "bitify.circom";

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

    signal hashOutput[4] <== Poseidon2_8(4)(hashInput);

    signal bits[64] <== Num2Bits_strict()(hashOutput[0]);
    for (var i = 63; i >= 64 - powBits; i--) {
        enable * bits[i] === 0;
    }

    for (var i = 64 - powBits - 1; i >= 0; i--) {
        _ <== bits[i];
    }
}