pragma circom 2.1.0;
pragma custom_templates;

include "custom/poseidon.circom";

// The gate test of tests/poseidon_bn128_wrap.rs: n uses of PoseidonT(5) in a chain, each
// permutation's first output the next one's initial state, and the last one's the public. Nothing
// else, so the AIR's rows are the bands, a few copies and the public's.
template Chain(n) {
    signal input initialState;
    signal input in[n][4];
    signal output out;

    component p[n];
    for (var i = 0; i < n; i++) {
        p[i] = CustomPoseidon(4);
        if (i == 0) {
            p[i].initialState <== initialState;
        } else {
            p[i].initialState <== p[i - 1].out[0];
        }
        p[i].in <== in[i];
    }
    out <== p[n - 1].out[0];
}

component main = Chain(3);
