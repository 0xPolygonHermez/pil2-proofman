pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/circuit_type.circom";
include "iszero.circom";
include "vadcop/select_vk.circom";
include "vadcop/fold.circom";
include "vadcop/values.circom";

// recursive2: verifies n proofs of one airgroup (recursive1 or recursive2) and folds their vadcop values.

// A recursive2 over n proofs of one airgroup (recursive1 or recursive2 proofs, keyed by circuitType).
template Recursive2(n) {
    var airgroupId = vadcop_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, 0);
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();
    var nPub = vadcop_nPublics();
    var nPV = vadcop_nProofValues();
    var S = vadcopValuesSize(nAgg, W);
    var P = key_proofSize(0);
    var multiAir = vadcop_multiAir();
    var nBasics = vadcop_nBasics(airgroupId);
    var rootCBasics[nBasics][4] = vadcop_basics(airgroupId);
    var nPubV = key_nPublics(0);

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal input slots[n][S + P];

    signal vadcopValues[n][S];
    for (var s = 0; s < n; s++) { for (var i = 0; i < S; i++) { vadcopValues[s][i] <== slots[s][i]; } }

    component vs[n];
    component pubs[n];
    signal {binary} isNull[n * multiAir];
    for (var s = 0; s < n; s++) {
        vs[s] = parallel StarkVerifierGL(0);
        for (var i = 0; i < P; i++) { vs[s].proof[i] <== slots[s][S + i]; }
        for (var i = 0; i < key_nProofValues(0); i++) { vs[s].proofvalues[i] <== proofValues[i]; }
        pubs[s] = VadcopPublics(airgroupId, nAgg, W, nPub, nPV, nPubV, nPubV - 4);
        pubs[s].vadcopValues <== vadcopValues[s];
        pubs[s].publics <== publics;
        pubs[s].proofValues <== proofValues;
        pubs[s].globalChallenge <== globalChallenge;
        pubs[s].rootCAgg <== rootCAgg;
        vs[s].publics <== pubs[s].out;
        if (multiAir == 1) {
            isNull[s] <== IsZero()(vadcopValues[s][0]);
            vs[s].enable[0] <== 1 - isNull[s];
            vs[s].rootCIn <== SelectVerificationKeyNull(nBasics, airgroupId)(vadcopValues[s][0], rootCBasics, rootCAgg);
        } else {
            vs[s].rootCIn <== SelectVerificationKey(nBasics, airgroupId)(vadcopValues[s][0], rootCBasics, rootCAgg);
        }
    }

    sv_circuitType <== circuitType;
    component out = FoldOutputs(n, nAgg, W, multiAir);
    out.slots <== vadcopValues;
    out.isNull <== isNull;
    sv_aggregatedProofs <== out.vadcopValues[0];
    for (var i = 0; i < nAgg; i++) {
        sv_aggregationTypes[i] <== out.vadcopValues[1 + i];
        for (var e = 0; e < 3; e++) { sv_airgroupvalues[i][e] <== out.vadcopValues[1 + nAgg + 3 * i + e]; }
    }
    for (var i = 0; i < W; i++) { sv_stage1Hash[i] <== out.vadcopValues[1 + 4 * nAgg + i]; }
}
