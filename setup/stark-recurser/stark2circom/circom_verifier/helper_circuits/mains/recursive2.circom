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
    var airgroupId = vd_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, 0);
    var nAgg = vd_nAgg(airgroupId);
    var W = vd_W();
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();
    var S = sv_size(nAgg, W);
    var P = svd_proofSize(0);
    var multiAir = vd_multiAir();
    var nBasics = vd_nBasics(airgroupId);
    var rootCBasics[nBasics][4] = vd_basics(airgroupId);
    var nPubV = svd_nPublics(0);

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

    signal sv[n][S];
    for (var s = 0; s < n; s++) { for (var i = 0; i < S; i++) { sv[s][i] <== slots[s][i]; } }

    component vs[n];
    component pubs[n];
    signal {binary} isNull[n * multiAir];
    for (var s = 0; s < n; s++) {
        vs[s] = parallel StarkVerifierGL(0);
        for (var i = 0; i < P; i++) { vs[s].proof[i] <== slots[s][S + i]; }
        for (var i = 0; i < svd_nProofValues(0); i++) { vs[s].proofvalues[i] <== proofValues[i]; }
        pubs[s] = VadcopPublics(airgroupId, nAgg, W, nPub, nPV, nPubV, nPubV - 4);
        pubs[s].sv <== sv[s];
        pubs[s].publics <== publics;
        pubs[s].proofValues <== proofValues;
        pubs[s].globalChallenge <== globalChallenge;
        pubs[s].rootCAgg <== rootCAgg;
        vs[s].publics <== pubs[s].out;
        if (multiAir == 1) {
            isNull[s] <== IsZero()(sv[s][0]);
            vs[s].enable[0] <== 1 - isNull[s];
            vs[s].rootCIn <== SelectVerificationKeyNull(nBasics, airgroupId)(sv[s][0], rootCBasics, rootCAgg);
        } else {
            vs[s].rootCIn <== SelectVerificationKey(nBasics, airgroupId)(sv[s][0], rootCBasics, rootCAgg);
        }
    }

    sv_circuitType <== circuitType;
    component out = FoldOutputs(n, nAgg, W, multiAir);
    out.slots <== sv;
    out.isNull <== isNull;
    sv_aggregatedProofs <== out.sv[0];
    for (var i = 0; i < nAgg; i++) {
        sv_aggregationTypes[i] <== out.sv[1 + i];
        for (var e = 0; e < 3; e++) { sv_airgroupvalues[i][e] <== out.sv[1 + nAgg + 3 * i + e]; }
    }
    for (var i = 0; i < W; i++) { sv_stage1Hash[i] <== out.sv[1 + 4 * nAgg + i]; }
}
