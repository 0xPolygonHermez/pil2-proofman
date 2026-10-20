pragma circom 2.1.0;
pragma custom_templates;

include "custom/poseidon.circom";

// The publics test of tests/poseidon_bn128_wrap.rs: 12 outputs and 2 public inputs, 14 publics,
// more than the 9 of a public row, around a PoseidonT(5) use.
template Publics() {
    signal input s;
    signal input k;
    signal input x[4];
    signal output out[12];

    component p = CustomPoseidon(4);
    p.initialState <== s;
    p.in <== x;
    for (var i = 0; i < 5; i++) {
        out[i] <== p.out[i] * k;
    }
    for (var i = 5; i < 12; i++) {
        out[i] <== out[i - 5] + out[i - 4] * x[i % 4];
    }
}

component main {public [s, k]} = Publics();
