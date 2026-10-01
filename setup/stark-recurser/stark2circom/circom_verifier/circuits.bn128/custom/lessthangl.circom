pragma circom 2.1.0;
pragma custom_templates;

include "rangecheck.circom";

// The templates of ../lessthangl.circom, with Num2Bytes gates for its range checks. A circuit
// includes one file or the other and never both, since they define the same templates (circom's
// T2008): the verifier with custom templates, and the final circuit around it, include this one.

// Given an integer a, checks whether a < GL: a < 2^64 and a + 2^64 - GL < 2^64, which together are
// a < GL exactly, in whole 16-bit chunks.
template LessThanGoldilocks() {
    var p = 0xFFFFFFFF00000001;
    signal input in;
    signal output {maxNum} out;

    _ <== Num2Bytes(64)(in);
    _ <== Num2Bytes(64)(in + (1<<64) - p);

    out.maxNum = p - 1;
    out <== in;
}

template LessThan64Bits() {
    signal input in;
    signal output {maxNum} out;

    _ <== Num2Bytes(64)(in);

    out.maxNum = 0xFFFFFFFFFFFFFFFF;
    out <== in;
}
