pragma circom 2.1.0;

include "poseidon.circom";
include "verifier/transcript_words_bn128.circom";

// The transcript sponge's permutation under the name verifier/transcript_sponge.circom calls: circomlib's `PoseidonEx(16, 17)`.
function sponge_rate() { return 16; }
function sponge_capacity() { return 1; }

template SpongePermutation() {
    signal input in[16];
    signal input capacity[1];
    signal output out[16 + 1];

    out <== PoseidonEx(16, 17)(in, capacity[0]);
}
