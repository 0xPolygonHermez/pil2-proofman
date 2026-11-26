pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/circuit_type.circom";
include "vadcop/fold.circom";
include "vadcop/values.circom";
include "vadcop/verify_child.circom";

// The compressor and recursive1: verify basic proofs (or compressor proofs) of one air and mint their
// folded vadcop values.

/*
    The compressor (withRootCAgg = 0) or a recursive1 over one basic proof: the proof verified, its vadcop
    values minted. A recursive1 also takes rootCAgg, unused here: it is public so that a recursive1's
    publics are laid out as a recursive2's, which verifies both.
*/
template Recursive1(withRootCAgg) {
    var airgroupId = vadcop_airgroupId();
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[vadcop_nPublics()];
    signal input proofValues[vadcop_nProofValues()][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4 * withRootCAgg];
    signal input proof[key_proofSize(0)];

    component verifier = StarkVerifierGL(0);
    verifier.proof <== proof;
    for (var i = 0; i < key_nPublics(0); i++) { verifier.publics[i] <== publics[i]; }
    for (var i = 0; i < key_nProofValues(0); i++) { verifier.proofvalues[i] <== proofValues[i]; }
    verifier.globalChallenge <== globalChallenge;

    component values = BasicValues(0, circuitTypeOf(airgroupId, vadcop_airId() + 1), nAgg, 0);
    values.rootC <== verifier.rootC;
    values.proof <== proof;
    (sv_circuitType, sv_aggregatedProofs, sv_aggregationTypes, sv_airgroupvalues, sv_stage1Hash) <== SplitVadcopValues(nAgg, W)(values.vadcopValues);
    if (withRootCAgg == 1) { _ <== rootCAgg; }
}

// A recursive1 over one compressor proof: its vadcop values arrive public with it.
template Recursive1Compressed() {
    var airgroupId = vadcop_airgroupId();
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();

    signal input sv_circuitType;
    signal input sv_aggregatedProofs;
    signal input sv_aggregationTypes[nAgg];
    signal input sv_airgroupvalues[nAgg][3];
    signal input sv_stage1Hash[W];
    signal input publics[vadcop_nPublics()];
    signal input proofValues[vadcop_nProofValues()][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal input proof[key_proofSize(0)];

    // The compressor minted the airgroup's types; this proof re-exposes exactly those.
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(airgroupId);
    for (var i = 0; i < nAgg; i++) { sv_aggregationTypes[i] === AGG[i]; }

    component child = VerifyChild(0, airgroupId, -1, 0);
    child.vadcopValues <== JoinVadcopValues(nAgg, W)(sv_circuitType, sv_aggregatedProofs, sv_aggregationTypes, sv_airgroupvalues, sv_stage1Hash);
    child.proof <== proof;
    child.publics <== publics;
    child.proofValues <== proofValues;
    child.globalChallenge <== globalChallenge;
    child.rootCAgg <== rootCAgg;
}

/*
    A recursive1 over n proofs of one air: basic proofs, empty when slotNull says so, or (compressed)
    compressor proofs, each after its vadcop values, empty when their circuitType is 0.
*/
template Recursive1Batched(n, compressed) {
    var airgroupId = vadcop_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, vadcop_airId() + 1);
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();
    var S = vadcopValuesSize(nAgg, W);
    var P = key_proofSize(0);

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[vadcop_nPublics()];
    signal input proofValues[vadcop_nProofValues()][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    // Per slot: [its vadcop values (compressed)] and its proof.
    signal input slots[n][S * compressed + P];
    // 1 = empty slot, dropped from the fold and the count.
    signal input slotNull[n * (1 - compressed)];

    component fold = FoldSlots(n, nAgg, W, 1);
    component child[n * compressed];
    component verifier[n * (1 - compressed)];
    component values[n * (1 - compressed)];
    for (var s = 0; s < n; s++) {
        if (compressed == 1) {
            child[s] = parallel VerifyChild(0, airgroupId, -1, 1);
            for (var i = 0; i < S; i++) { child[s].vadcopValues[i] <== slots[s][i]; }
            for (var i = 0; i < P; i++) { child[s].proof[i] <== slots[s][S + i]; }
            child[s].publics <== publics;
            child[s].proofValues <== proofValues;
            child[s].globalChallenge <== globalChallenge;
            child[s].rootCAgg <== rootCAgg;
            for (var i = 0; i < S; i++) { fold.slots[s][i] <== slots[s][i]; }
            fold.isNullIn[s] <== child[s].isNull[0];
        } else {
            slotNull[s] * (slotNull[s] - 1) === 0;
            verifier[s] = parallel StarkVerifierGL(0);
            verifier[s].proof <== slots[s];
            for (var i = 0; i < key_nPublics(0); i++) { verifier[s].publics[i] <== publics[i]; }
            for (var i = 0; i < key_nProofValues(0); i++) { verifier[s].proofvalues[i] <== proofValues[i]; }
            verifier[s].globalChallenge <== globalChallenge;
            verifier[s].enable[0] <== 1 - slotNull[s];
            values[s] = BasicValues(0, circuitType, nAgg, 1);
            values[s].rootC <== verifier[s].rootC;
            values[s].proof <== slots[s];
            values[s].isNull[0] <== slotNull[s];
            fold.slots[s] <== values[s].vadcopValues;
            fold.isNullIn[s] <== slotNull[s];
        }
    }
    if (compressed == 1) { _ <== slotNull; }

    sv_aggregatedProofs <== fold.aggregatedProofs;
    sv_aggregationTypes <== fold.aggregationTypes;
    sv_airgroupvalues <== fold.airgroupvalues;
    sv_stage1Hash <== fold.stage1Hash;

    // An all-empty batch would prove nothing, and vadcop_final drops the count that shows it.
    signal {binary} noProofs <== IsZero()(sv_aggregatedProofs);
    noProofs === 0;
    // The air's own type: the aggregated one would make recursive2 select rootCAgg.
    sv_circuitType <== circuitType;
}
