pragma circom 2.1.0;
pragma custom_templates;

include "poseidon.circom";
include "verifier/transcript_words_gl.circom";

// The transcript sponge's permutation under the name verifier/transcript_sponge.circom calls.
function sponge_rate() { return 12; }
function sponge_capacity() { return 4; }

template SpongePermutation() {
    signal input in[12];
    signal input capacity[4];
    signal output out[16];

    out <== Poseidon(16)(in, capacity);
}

// One step of the vadcop lattice chain: the permutation over a whole 16-word state.
function chain_w() { return 16; }

template ChainPermutation() {
    signal input in[16];
    signal output out[16];

    out <== SpongePermutation()([in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11]], [in[12], in[13], in[14], in[15]]);
}
