pragma circom 2.1.0;

// The plain field family's hooks of gl.circom and fft.circom: Num2BitsT range checks.
include "bitifyT.circom";
include "lessthangl.circom";
include "fft.circom";

template GLRangeBits(n) {
    signal input in;

    _ <== Num2BitsT(n)(in);
}

// The FRI fold's interpolation (verifier/stark_verifier_bn128.circom): unreduced, as EvalPol reduces.
template FFTFold(nBits) {
    signal input {maxNum} in[1 << nBits][3];
    signal output {maxNum} out[1 << nBits][3];

    out <== FFTLazy(nBits, 1)(in);
}
