pragma circom 2.1.0;
pragma custom_templates;

include "custom/poseidon.circom";
include "custom/rangecheck.circom";

// The range-check test of tests/poseidon_bn254_wrap.rs: uses of the custom gate Num2Bytes of whole
// chunks (16, 64 and 80 bits) and of a partial top chunk (3, 17 and 70 bits), among PLONK gates
// and a PoseidonT(5) use, with a public.
//
// - The uses of `x` and `y` are isolated: those inputs and their chunks are read by nothing else,
//   so their range-check rows have no copy constraint, and a test that changes their cells breaks
//   only the range check's own constraints.
// - The others are connected: their inputs are computed by PLONK gates, and their chunks read by
//   more of them, so the connection binds their rows to the rest of the circuit.
//
// The outputs of Num2Bytes go into signals of no tag, so the fixture compiles whether the library's
// template tags them or not.
template Num2BytesUses() {
    signal input x;
    signal input y;
    signal input a;
    signal input b;
    signal input c;
    signal input s[4];
    signal output out;

    _ <== Num2Bytes(64)(x);
    _ <== Num2Bytes(70)(y);

    signal a64[4] <== Num2Bytes(64)(a);
    signal b80[5] <== Num2Bytes(80)(b);
    signal c3[1] <== Num2Bytes(3)(c);

    component p = CustomPoseidon(4);
    p.initialState <== c;
    p.in <== s;

    // in < 2^16 and < 2^17: a chunk of a, and the sum of two more.
    signal d16[1] <== Num2Bytes(16)(a64[1]);
    signal e <== a64[2] + a64[3];
    signal e17[2] <== Num2Bytes(17)(e);
    // A second use of the 64-bit gate.
    signal ab <== a64[0] * b80[4];
    signal ab64[4] <== Num2Bytes(64)(ab);

    signal acc[6];
    acc[0] <== p.out[0] * d16[0];
    acc[1] <== acc[0] * e17[1] + b80[0];
    acc[2] <== acc[1] * ab64[3] + b80[1];
    acc[3] <== acc[2] * a64[3] + b80[2];
    acc[4] <== acc[3] * e17[0] + b80[3];
    acc[5] <== acc[4] * ab64[0] + ab64[1] + ab64[2];
    out <== acc[5] + p.out[1];
}

component main = Num2BytesUses();
