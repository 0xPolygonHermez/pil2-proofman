pragma circom 2.1.0;

// Transcript words that are Goldilocks words: 4-word roots and digests, 3-word challenges, 63 query bits a word.
function tr_digestW() { return 4; }
function tr_challengeW() { return 3; }
function tr_wordBits() { return 63; }

template TranscriptChallenge() {
    signal input in[3];
    signal output out[3];

    out <== in;
}
