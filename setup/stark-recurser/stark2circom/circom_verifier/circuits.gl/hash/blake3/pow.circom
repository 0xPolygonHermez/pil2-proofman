pragma circom 2.1.0;
pragma custom_templates;

include "blake3.circom";
include "bitify.circom";

/*
    Verify the grinding nonce.

    Native contract (Blake3Goldilocks::grinding): state = [c0, c1, c2, nonce,
    0, 0, 0, 0], one compression at the IV with the root flags, and the nonce is
    accepted when
    to_canonical(out[0]) < 2^(64 - powBits).

    The comparison must be done on the *packed Goldilocks* output rather than on
    the gate's raw u32 pair. The native check reduces mod p first, and when the
    unreduced value is >= p its raw high word is 2^32 - 1 while the canonical
    high word is small -- so reading the raw pair would reject nonces the prover
    legitimately found (about 2^-32 of them). Num2Bits_strict decomposes the
    canonical element and is exact; it also matches what the Poseidon pow does.
    It runs once per proof, so its ~64 constraints are not worth a custom gate.
*/
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

    signal hashOutput[4] <== Blake3Hash8()(hashInput, 0);
    for (var i = 1; i < 4; i++) {
        _ <== hashOutput[i];
    }

    signal bits[64] <== Num2Bits_strict()(hashOutput[0]);
    for (var i = 63; i >= 64 - powBits; i--) {
        enable * bits[i] === 0;
    }

    for (var i = 64 - powBits - 1; i >= 0; i--) {
        _ <== bits[i];
    }
}
