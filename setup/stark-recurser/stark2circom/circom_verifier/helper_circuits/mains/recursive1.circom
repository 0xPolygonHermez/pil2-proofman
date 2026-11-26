pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/circuit_type.circom";
include "iszero.circom";
include "vadcop/fold.circom";
include "vadcop/values.circom";

// recursive1: verifies basic proofs (or compressor proofs) of one air and mints their folded vadcop values.

// A recursive1 over one basic proof: verified, its vadcop values minted.
template Recursive1() {
    var airgroupId = vd_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, vd_airId() + 1);
    var nAgg = vd_nAgg(airgroupId);
    var W = vd_W();
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal input proof[svd_proofSize(0)];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    for (var i = 0; i < svd_nPublics(0); i++) { sV.publics[i] <== publics[i]; }
    for (var i = 0; i < svd_nProofValues(0); i++) { sV.proofvalues[i] <== proofValues[i]; }
    sV.globalChallenge <== globalChallenge;

    component v = BasicValues(0, circuitType, nAgg, 0);
    v.rootC <== sV.rootC;
    v.proof <== proof;
    sv_circuitType <== v.circuitTypeOut;
    sv_aggregatedProofs <== v.aggregatedProofs;
    sv_aggregationTypes <== v.aggregationTypes;
    sv_airgroupvalues <== v.airgroupvalues;
    sv_stage1Hash <== v.stage1Hash;
    _ <== rootCAgg;
}

// A recursive1 over one compressor proof: its vadcop values arrive public with it.
template Recursive1Compressed() {
    var nAgg = vd_nAgg(vd_airgroupId());
    var W = vd_W();
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();
    var S = sv_size(nAgg, W);

    signal input sv_circuitType;
    signal input sv_aggregatedProofs;
    signal input sv_aggregationTypes[nAgg];
    signal input sv_airgroupvalues[nAgg][3];
    signal input sv_stage1Hash[W];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal input proof[svd_proofSize(0)];

    // The compressor minted the airgroup's types; this proof re-exposes exactly those.
    var AGG[geo_len1(nAgg)] = vd_AGG(vd_airgroupId());
    for (var i = 0; i < nAgg; i++) { sv_aggregationTypes[i] === AGG[i]; }

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    for (var i = 0; i < svd_nProofValues(0); i++) { sV.proofvalues[i] <== proofValues[i]; }
    component p = VadcopPublics(vd_airgroupId(), nAgg, W, nPub, nPV, svd_nPublics(0), -1);
    p.sv[0] <== sv_circuitType;
    p.sv[1] <== sv_aggregatedProofs;
    for (var i = 0; i < nAgg; i++) {
        p.sv[2 + i] <== sv_aggregationTypes[i];
        for (var e = 0; e < 3; e++) { p.sv[2 + nAgg + 3 * i + e] <== sv_airgroupvalues[i][e]; }
    }
    for (var i = 0; i < W; i++) { p.sv[2 + 4 * nAgg + i] <== sv_stage1Hash[i]; }
    p.publics <== publics;
    p.proofValues <== proofValues;
    p.globalChallenge <== globalChallenge;
    p.rootCAgg <== rootCAgg;
    sV.publics <== p.out;
}

// A recursive1 over k proofs of one air (basic, or compressor proofs when compressed); a basic slot
// is null when its flag says so, a compressor slot when its circuitType is 0.
template Recursive1Batched(k, compressed) {
    var airgroupId = vd_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, vd_airId() + 1);
    var nAgg = vd_nAgg(airgroupId);
    var W = vd_W();
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();
    var S = sv_size(nAgg, W);
    var P = svd_proofSize(0);

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    // Per slot: [its vadcop values (compressed)] and its proof.
    signal input slots[k][S * compressed + P];
    // 1 = empty slot, dropped from the fold and the count.
    signal input slotNull[k * (1 - compressed)];

    signal sv[k][S];
    signal {binary} isNull[k];
    component vs[k];
    component vals[k];
    component pubs[k];
    for (var s = 0; s < k; s++) {
        vs[s] = parallel StarkVerifierGL(0);
        for (var i = 0; i < P; i++) { vs[s].proof[i] <== slots[s][S * compressed + i]; }
        for (var i = 0; i < svd_nProofValues(0); i++) { vs[s].proofvalues[i] <== proofValues[i]; }
        if (compressed == 1) {
            for (var i = 0; i < S; i++) { sv[s][i] <== slots[s][i]; }
            pubs[s] = VadcopPublics(vd_airgroupId(), nAgg, W, nPub, nPV, svd_nPublics(0), -1);
            pubs[s].sv <== sv[s];
            pubs[s].publics <== publics;
            pubs[s].proofValues <== proofValues;
            pubs[s].globalChallenge <== globalChallenge;
            pubs[s].rootCAgg <== rootCAgg;
            vs[s].publics <== pubs[s].out;
            isNull[s] <== IsZero()(slots[s][0]);
            vs[s].enable[0] <== 1 - isNull[s];
        } else {
            slotNull[s] * (slotNull[s] - 1) === 0;
            for (var i = 0; i < svd_nPublics(0); i++) { vs[s].publics[i] <== publics[i]; }
            vs[s].enable[0] <== 1 - slotNull[s];
            vs[s].globalChallenge <== globalChallenge;
            isNull[s] <== slotNull[s];
            vals[s] = BasicValues(0, circuitType, nAgg, 1);
            vals[s].rootC <== vs[s].rootC;
            for (var i = 0; i < P; i++) { vals[s].proof[i] <== slots[s][i]; }
            vals[s].isNull[0] <== isNull[s];
            sv[s][0] <== vals[s].circuitTypeOut;
            sv[s][1] <== vals[s].aggregatedProofs;
            for (var i = 0; i < nAgg; i++) {
                sv[s][2 + i] <== vals[s].aggregationTypes[i];
                sv[s][2 + nAgg + 3 * i] <== vals[s].airgroupvalues[i][0];
                sv[s][2 + nAgg + 3 * i + 1] <== vals[s].airgroupvalues[i][1];
                sv[s][2 + nAgg + 3 * i + 2] <== vals[s].airgroupvalues[i][2];
            }
            for (var i = 0; i < W; i++) { sv[s][2 + 4 * nAgg + i] <== vals[s].stage1Hash[i]; }
        }
    }
    if (compressed == 1) { _ <== slotNull; }

    component out = FoldOutputs(k, nAgg, W, 1);
    out.slots <== sv;
    out.isNull <== isNull;
    sv_aggregatedProofs <== out.sv[0];
    for (var i = 0; i < nAgg; i++) {
        sv_aggregationTypes[i] <== out.sv[1 + i];
        for (var e = 0; e < 3; e++) { sv_airgroupvalues[i][e] <== out.sv[1 + nAgg + 3 * i + e]; }
    }
    for (var i = 0; i < W; i++) { sv_stage1Hash[i] <== out.sv[1 + 4 * nAgg + i]; }

    // An all-empty batch would prove nothing, and vadcop_final drops the count that shows it.
    signal {binary} noProofs <== IsZero()(sv_aggregatedProofs);
    noProofs === 0;
    // The air's own type: the aggregated one would make recursive2 select rootCAgg.
    sv_circuitType <== circuitType;
}
