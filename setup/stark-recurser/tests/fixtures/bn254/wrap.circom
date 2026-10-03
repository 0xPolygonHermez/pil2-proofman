pragma circom 2.1.0;
pragma custom_templates;

include "custom/poseidon.circom";

// The end-to-end test of tests/poseidon_bn254_wrap.rs: a PoseidonT(5) use among multiplications,
// additions and copies, as the final circuit has them, more than its band's rows hold, and one
// public.
template Wrap() {
    signal input a;
    signal input b;
    signal input x[4];
    signal output out;

    signal ab <== a * b;
    signal s <== ab + 3 * a + 7;
    signal sq <== s * s;

    component p = CustomPoseidon(4);
    p.initialState <== a + b;
    p.in[0] <== x[0] * sq;
    p.in[1] <== x[1] + sq;
    p.in[2] <== x[2];
    p.in[3] <== x[3] * x[3];

    // Multiplications and sums enough for PLONK rows past the band.
    signal acc[120];
    acc[0] <== p.out[0] * p.out[1];
    for (var i = 1; i < 120; i++) {
        if (i % 2 == 1) {
            acc[i] <== acc[i - 1] * x[i % 4];
        } else {
            acc[i] <== acc[i - 1] + 3 * x[i % 4] + i;
        }
    }
    out <== acc[119] + p.out[2] - 5 * p.out[4];
}

component main = Wrap();
