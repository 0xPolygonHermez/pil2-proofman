pragma circom 2.1.0;
pragma custom_templates;

// The custom field family's hooks of gl.circom and fft.circom: Num2Bytes range checks.
include "rangecheck.circom";
include "lessthangl.circom";
include "../fft.circom";

template GLRangeBits(n) {
    signal input in;

    RangeCheck(n)(in);
}

// The FRI fold's interpolation (verifier/stark_verifier_bn128.circom): reduced, for the custom EvalPol.
template FFTFold(nBits) {
    signal input {maxNum} in[1 << nBits][3];
    signal output {maxNum} out[1 << nBits][3];

    out <== FFT(nBits, 1)(in);
}
