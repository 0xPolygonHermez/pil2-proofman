pragma circom 2.1.0;
pragma custom_templates;

include "elliptic_curve.circom";
include "verifier/geometry.circom";

/*
    The vadcop transcripts on the family's TranscriptEngine and ChainPermutation (the verifier's
    include picks the family): a proof's stage-1 contribution, the global challenge, the global
    constraints' challenges.
*/

// Absorbs rootC, root1, circuitType and the stage-1 airvalues (AV[j] = stage of airvalue j); returns the
// first `nOut` squeezed words. Airvalues of later stages are not absorbed.
template Stage1Transcript(circuitType, nAirValues, AV, nOut) {
    signal input rootC[4];
    signal input root1[4];
    signal input airValues[geo_len1(nAirValues)][3];
    signal output out[nOut];

    var n1 = 0;
    for (var j = 0; j < nAirValues; j++) { if (AV[j] == 1) { n1++; } }
    var ops[4] = [0, 9 + n1, 1, nOut];
    component tr = TranscriptEngine(9 + n1, nOut, 2, ops);
    for (var i = 0; i < 4; i++) {
        tr.in[i] <== rootC[i];
        tr.in[4 + i] <== root1[i];
    }
    tr.in[8] <== circuitType;
    var k = 9;
    for (var j = 0; j < nAirValues; j++) {
        if (AV[j] == 1) {
            tr.in[k] <== airValues[j][0];
            k++;
        } else {
            _ <== airValues[j];
        }
    }
    if (nAirValues == 0) { _ <== airValues; }
    out <== tr.out;
}

// The lattice contribution: chain_w() squeezed words, then ChainPermutation round after round.
template CalculateStage1Hash(circuitType, nAirValues, AV, latticeSize) {
    signal input rootC[4];
    signal input root1[4];
    signal input airValues[geo_len1(nAirValues)][3];
    signal output values[latticeSize];

    var w = chain_w();
    assert(latticeSize % w == 0);
    signal seed[w] <== Stage1Transcript(circuitType, nAirValues, AV, w)(rootC, root1, airValues);
    for (var i = 0; i < w; i++) { values[i] <== seed[i]; }
    component ch[geo_len1(latticeSize \ w - 1)];
    for (var r = 0; r + 1 < latticeSize \ w; r++) {
        ch[r] = ChainPermutation();
        for (var j = 0; j < w; j++) { ch[r].in[j] <== values[r * w + j]; }
        for (var j = 0; j < w; j++) { values[(r + 1) * w + j] <== ch[r].out[j]; }
    }
}

// The curve contribution: ten squeezed words as a point (elliptic_curve.circom's HashToCurve).
template CalculateStage1HashEC(circuitType, nAirValues, AV, A, B, Z, C1, C2) {
    signal input rootC[4];
    signal input root1[4];
    signal input airValues[geo_len1(nAirValues)][3];
    signal output P[2][5];

    signal xy[10] <== Stage1Transcript(circuitType, nAirValues, AV, 10)(rootC, root1, airValues);
    signal x[5] <== [xy[0], xy[1], xy[2], xy[3], xy[4]];
    signal y[5] <== [xy[5], xy[6], xy[7], xy[8], xy[9]];
    P <== HashToCurve(A, B, Z, C1, C2)(x, y);
}

// globalChallenge == the transcript of the publics, the stage-1 proof values (PV[j] = stage) and every
// airgroup's stage-1 contribution (W words each).
template VerifyGlobalChallenges(nPublics, nProofValues, PV, nAirgroups, W) {
    signal input publics[geo_len1(nPublics)];
    signal input proofValues[geo_len1(nProofValues)][3];
    signal input stage1Hash[nAirgroups][W];
    signal input globalChallenge[3];

    var n1 = 0;
    for (var j = 0; j < nProofValues; j++) { if (PV[j] == 1) { n1++; } }
    var nIn = nPublics + n1 + nAirgroups * W;
    var ops[4] = [0, nIn, 1, 3];
    component tr = TranscriptEngine(nIn, 3, 2, ops);
    var k = 0;
    for (var i = 0; i < nPublics; i++) { tr.in[k] <== publics[i]; k++; }
    for (var j = 0; j < nProofValues; j++) {
        if (PV[j] == 1) {
            tr.in[k] <== proofValues[j][0];
            k++;
        } else {
            _ <== proofValues[j][0];
        }
    }
    for (var a = 0; a < nAirgroups; a++) {
        for (var i = 0; i < W; i++) { tr.in[k] <== stage1Hash[a][i]; k++; }
    }
    if (nPublics == 0) { _ <== publics; }
    if (nProofValues == 0) { _ <== proofValues; }
    globalChallenge === tr.out;
}

// The global constraints' n challenges, squeezed from a transcript over globalChallenge.
template GlobalChallenges(n) {
    signal input globalChallenge[3];
    signal output challenges[n][3];

    var ops[4] = [0, 3, 1, 3 * n];
    component tr = TranscriptEngine(3, 3 * n, 2, ops);
    tr.in <== globalChallenge;
    for (var i = 0; i < n; i++) { challenges[i] <== [tr.out[3 * i], tr.out[3 * i + 1], tr.out[3 * i + 2]]; }
}
