pragma circom 2.1.0;
pragma custom_templates;

include "poseidon.circom";
include "verifier/transcript_words_bn128.circom";

// The transcript sponge's permutation under the name verifier/transcript_sponge.circom calls: `CustomPoseidon(rate)`,
// rate = the Merkle arity; the generated verifier defines sponge_rate() / sponge_capacity() (1).
template SpongePermutation() {
    signal input in[sponge_rate()];
    signal input capacity[1];
    signal output out[sponge_rate() + 1];

    out <== CustomPoseidon(sponge_rate())(in, capacity[0]);
}
