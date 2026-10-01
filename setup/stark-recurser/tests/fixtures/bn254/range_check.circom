pragma circom 2.1.0;
pragma custom_templates;

include "custom/lessthangl.circom";

// The circuit of tests/range_check_bn254.rs: a LessThanGoldilocks, and a RangeCheck of each width
// that matters to Num2Bytes, whose chunks are 16 bits and whose gates take 80 at most:
// - 1, 16 and 17: a partial chunk, a whole one, and one bit into the next;
// - 64 and 65: a Goldilocks element, and one bit more;
// - 80 and 81: the most one gate takes, and the least that needs two;
// - 154 and 160: the widest quotient of the final circuit, and the widest RangeCheck.
template RangeChecks() {
    signal input in[9];
    signal input gl;
    signal output out;

    RangeCheck(1)(in[0]);
    RangeCheck(16)(in[1]);
    RangeCheck(17)(in[2]);
    RangeCheck(64)(in[3]);
    RangeCheck(65)(in[4]);
    RangeCheck(80)(in[5]);
    RangeCheck(81)(in[6]);
    RangeCheck(154)(in[7]);
    RangeCheck(160)(in[8]);
    out <== LessThanGoldilocks()(gl);
}

component main = RangeChecks();
