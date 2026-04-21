pragma circom 2.1.0;
pragma custom_templates;

include "poseidon.circom";
include "bitifyT.circom";

template VerifyPoW(powBits) {
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;

    signal hash[1] <== PoseidonEx(4, 1)([challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce], 0);


    signal {binary} n2b[254] <== Num2Bits_strictT()(hash[0]);

    signal {binary} bits[64];
    for (var j=0; j<64; j++) {
        bits[j] <== n2b[j];
    }

    for (var i = 63; i >= 64 - powBits; i--) {
        enable * bits[i] === 0;
    }

    for (var i = 64 - powBits - 1; i >= 0; i--) {
        _ <== bits[i];
    }
}