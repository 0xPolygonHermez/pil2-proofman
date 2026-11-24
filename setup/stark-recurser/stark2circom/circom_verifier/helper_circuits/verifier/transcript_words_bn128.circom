pragma circom 2.1.0;

include "bn1togl3.circom";

// Transcript words that are BN128 elements: one per root, digest and challenge, 253 query bits a word.
function tr_digestW() { return 1; }
function tr_challengeW() { return 1; }
function tr_wordBits() { return 253; }

template TranscriptChallenge() {
    signal input in[1];
    signal output out[3];

    out <== BN1toGL3()(in[0]);
}
