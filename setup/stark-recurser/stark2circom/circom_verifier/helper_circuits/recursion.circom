pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "circuit_type.circom";
include "select_vk.circom";
include "agg_values.circom";
include "acc_points.circom";
include "elliptic_curve.circom";
include "vadcop_hashes.circom";

/*
    The recursion circuits (compressor, recursive1 plain / over a compressor / batched, recursive2,
    final_compressed, recursivef; vadcop_final in final_vadcop.circom) around StarkVerifierGL(k). The
    generated main file holds only data: the keys' svd_*(k) and ConstraintRootsAt(k), the vadcop data
    vd_*() (circuit_templates/vadcop_inputs.rs) and the `component main` line. Inputs keep the order the witness
    fill copies the zkin into: a proof is one flat array, a slot (a proof and the vadcop values before
    it) one row.

    A proof's vadcop values ("sv"): circuitType, aggregatedProofs, aggregationTypes[nAgg],
    airgroupvalues[nAgg][3], stage1Hash[W] (W = latticeSize, or 10: an EC point's two Fp5 limbs).
*/

function sv_size(nAgg, W) { return 2 + 4 * nAgg + W; }

// The stage-1 contribution of a proof of key k: its rootC, root1 and air values, as a lattice or EC point.
template Stage1Hash(k, circuitType) {
    var nAV = svd_nAirValues(k);
    var W = vd_W();

    signal input rootC[4];
    signal input root1[4];
    signal input airvalues[geo_len1(nAV)][3];
    signal output out[W];

    if (vd_ec() == 0) {
        out <== CalculateStage1Hash(circuitType, nAV, svd_AV(k), W)(rootC, root1, airvalues);
    } else {
        signal P[2][5] <== CalculateStage1HashEC(circuitType, nAV, svd_AV(k), vd_ecA(), vd_ecB(), vd_ecZ(), vd_ecC1(), vd_ecC2())(rootC, root1, airvalues);
        for (var i = 0; i < 10; i++) { out[i] <== P[i \ 5][i % 5]; }
    }
}

// The vadcop values a verified basic proof of key k mints (a null slot reports circuitType 0).
template BasicValues(k, circuitType, nAgg, nullable) {
    var W = vd_W();
    var nAV = svd_nAirValues(k);
    var offAV = svd_offAV(k);
    var offAGV = svd_offAGV(k);
    var OFF_ROOT[svd_nStages(k) + 1] = svd_OFF_ROOT(k);
    var AGG[geo_len1(nAgg)] = vd_AGG(vd_airgroupId());

    signal input rootC[4];
    signal input proof[svd_proofSize(k)];
    signal input isNull[nullable];
    signal output circuitTypeOut;
    signal output aggregatedProofs;
    signal output aggregationTypes[nAgg];
    signal output airgroupvalues[nAgg][3];
    signal output stage1Hash[W];

    if (nullable == 1) { circuitTypeOut <== (1 - isNull[0]) * circuitType; } else { circuitTypeOut <== circuitType; }
    // Stays 1 for an empty slot too: AggregateProofsNull multiplies it out.
    aggregatedProofs <== 1;
    for (var i = 0; i < nAgg; i++) {
        aggregationTypes[i] <== AGG[i];
        for (var e = 0; e < 3; e++) { airgroupvalues[i][e] <== proof[offAGV + 3 * i + e]; }
    }
    component h = Stage1Hash(k, circuitType);
    h.rootC <== rootC;
    for (var i = 0; i < 4; i++) { h.root1[i] <== proof[OFF_ROOT[0] + i]; }
    if (nAV > 0) {
        for (var i = 0; i < nAV; i++) { for (var e = 0; e < 3; e++) { h.airvalues[i][e] <== proof[offAV + 3 * i + e]; } }
    } else {
        h.airvalues <== [[0, 0, 0]];
    }
    stage1Hash <== h.out;
}

// The publics of a proof that carries vadcop values: sv, the run's publics and proof values, the global
// challenge, then (rootCAt >= 0) the aggregation key.
template VadcopPublics(nAgg, W, nPub, nPV, nOut, rootCAt) {
    signal input sv[sv_size(nAgg, W)];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal output out[nOut];

    var n = sv_size(nAgg, W);
    for (var i = 0; i < n; i++) { out[i] <== sv[i]; }
    for (var i = 0; i < nPub; i++) { out[n + i] <== publics[i]; }
    for (var i = 0; i < nPV; i++) { for (var e = 0; e < 3; e++) { out[n + nPub + 3 * i + e] <== proofValues[i][e]; } }
    for (var e = 0; e < 3; e++) { out[n + nPub + 3 * nPV + e] <== globalChallenge[e]; }
    if (rootCAt >= 0) {
        for (var i = 0; i < 4; i++) { out[rootCAt + i] <== rootCAgg[i]; }
    } else {
        _ <== rootCAgg;
    }
}

// Two stage-1 hashes combined (null-aware: a null side contributes nothing).
template FoldHash(W, nullAware) {
    signal input a[W];
    signal input b[W];
    signal input isNull[2 * nullAware];
    signal output out[W];

    signal {binary} nb[2 * nullAware];
    nb <== isNull;

    if (vd_ec() == 0 && nullAware == 1) {
        out <== AggregateValuesNull(W)(a, b, nb[0], nb[1]);
    } else if (vd_ec() == 0) {
        out <== AggregateValues(W)(a, b);
    } else {
        signal pa[2][5];
        signal pb[2][5];
        signal p[2][5];
        for (var i = 0; i < 10; i++) { pa[i \ 5][i % 5] <== a[i]; pb[i \ 5][i % 5] <== b[i]; }
        if (nullAware == 1) { p <== AccumulatePointsNull()(pa, pb, nb[0], nb[1]); } else { p <== AccumulatePoints()(pa, pb); }
        for (var i = 0; i < 10; i++) { out[i] <== p[i \ 5][i % 5]; }
    }
}

// n slots' values folded left to right; nullAware drops the slots flagged null.
template FoldSlots(n, nAgg, W, nullAware) {
    signal input {binary} aggTypes[nAgg];
    signal input aggregatedProofs[n];
    signal input airgroupvalues[n][nAgg][3];
    signal input stage1Hash[n][W];
    signal input isNullIn[n * nullAware];
    signal output outAggregatedProofs;
    signal output outAirgroupvalues[nAgg][3];
    signal output outStage1Hash[W];

    signal {binary} isNull[n * nullAware];
    isNull <== isNullIn;
    // prefNull[j - 2]: the first j slots are all null.
    signal {binary} prefNull[geo_len1(n - 2)];
    if (nullAware == 1) {
        for (var j = 2; j < n; j++) {
            var s = 0;
            for (var i = 0; i < j; i++) { s += isNull[i]; }
            prefNull[j - 2] <== IsZero()(j - s);
        }
    }
    if (nullAware == 0 || n <= 2) { prefNull[0] <== 0; }

    // acc[j - 1]: the first j + 1 slots folded.
    signal accAGV[geo_len1(n - 2)][nAgg][3];
    for (var i = 0; i < nAgg; i++) {
        for (var j = 1; j < n; j++) {
            if (nullAware == 1) {
                if (j == 1 && j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValuesNull()(airgroupvalues[0][i], airgroupvalues[1][i], aggTypes[i], isNull[0], isNull[1]);
                } else if (j == 1) {
                    accAGV[0][i] <== AggregateAirgroupValuesNull()(airgroupvalues[0][i], airgroupvalues[1][i], aggTypes[i], isNull[0], isNull[1]);
                } else if (j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValuesNull()(accAGV[j - 2][i], airgroupvalues[j][i], aggTypes[i], prefNull[j - 2], isNull[j]);
                } else {
                    accAGV[j - 1][i] <== AggregateAirgroupValuesNull()(accAGV[j - 2][i], airgroupvalues[j][i], aggTypes[i], prefNull[j - 2], isNull[j]);
                }
            } else {
                if (j == 1 && j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValues()(airgroupvalues[0][i], airgroupvalues[1][i], aggTypes[i]);
                } else if (j == 1) {
                    accAGV[0][i] <== AggregateAirgroupValues()(airgroupvalues[0][i], airgroupvalues[1][i], aggTypes[i]);
                } else if (j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValues()(accAGV[j - 2][i], airgroupvalues[j][i], aggTypes[i]);
                } else {
                    accAGV[j - 1][i] <== AggregateAirgroupValues()(accAGV[j - 2][i], airgroupvalues[j][i], aggTypes[i]);
                }
            }
        }
    }
    if (nAgg == 0 || n <= 2) { for (var i = 0; i < nAgg; i++) { accAGV[0][i] <== [0, 0, 0]; } }

    if (nullAware == 1) {
        outAggregatedProofs <== AggregateProofsNull(n)(aggregatedProofs, isNull);
    } else {
        outAggregatedProofs <== AggregateProofs(n)(aggregatedProofs);
    }

    signal accHash[geo_len1(n - 2)][W];
    component fold[n - 1];
    for (var j = 1; j < n; j++) {
        fold[j - 1] = FoldHash(W, nullAware);
        if (j == 1) { fold[j - 1].a <== stage1Hash[0]; } else { fold[j - 1].a <== accHash[j - 2]; }
        fold[j - 1].b <== stage1Hash[j];
        if (nullAware == 1) {
            if (j == 1) { fold[j - 1].isNull <== [isNull[0], isNull[1]]; } else { fold[j - 1].isNull <== [prefNull[j - 2], isNull[j]]; }
        }
        if (j == n - 1) { outStage1Hash <== fold[j - 1].out; } else { accHash[j - 1] <== fold[j - 1].out; }
    }
    if (n <= 2) { for (var i = 0; i < W; i++) { accHash[0][i] <== 0; } }
}

// aggregationTypes from slot 0, every slot required to agree (gated on its circuitType when gated).
template AggTypesConsistency(n, nAgg, W, gated) {
    signal input slots[n][sv_size(nAgg, W)];
    signal output aggregationTypes[nAgg];

    for (var i = 0; i < nAgg; i++) {
        aggregationTypes[i] <== slots[0][2 + i];
        for (var s = 1; s < n; s++) {
            if (gated == 1) {
                slots[s][0] * (slots[0][2 + i] - slots[s][2 + i]) === 0;
            } else {
                slots[0][2 + i] === slots[s][2 + i];
            }
        }
    }
}

// The fold's outputs: the folded values, the types checked binary.
template FoldOutputs(n, nAgg, W, nullAware) {
    signal input aggregationTypes[nAgg];
    signal input slots[n][sv_size(nAgg, W)];
    signal input isNull[n * nullAware];
    signal output sv[sv_size(nAgg, W) - 1];

    signal {binary} aggTypes[nAgg];
    for (var i = 0; i < nAgg; i++) {
        aggregationTypes[i] * (aggregationTypes[i] - 1) === 0;
        aggTypes[i] <== aggregationTypes[i];
    }
    component f = FoldSlots(n, nAgg, W, nullAware);
    f.aggTypes <== aggTypes;
    for (var s = 0; s < n; s++) {
        f.aggregatedProofs[s] <== slots[s][1];
        for (var i = 0; i < nAgg; i++) { for (var e = 0; e < 3; e++) { f.airgroupvalues[s][i][e] <== slots[s][2 + nAgg + 3 * i + e]; } }
        for (var i = 0; i < W; i++) { f.stage1Hash[s][i] <== slots[s][2 + 4 * nAgg + i]; }
    }
    f.isNullIn <== isNull;
    // sv without circuitType: aggregatedProofs, aggregationTypes, airgroupvalues, stage1Hash.
    sv[0] <== f.outAggregatedProofs;
    for (var i = 0; i < nAgg; i++) {
        sv[1 + i] <== aggregationTypes[i];
        for (var e = 0; e < 3; e++) { sv[1 + nAgg + 3 * i + e] <== f.outAirgroupvalues[i][e]; }
    }
    for (var i = 0; i < W; i++) { sv[1 + 4 * nAgg + i] <== f.outStage1Hash[i]; }
}

// The compressor of a basic air: its proof verified, its vadcop values minted.
template Compressor() {
    var airgroupId = vd_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, vd_airId() + 1);
    var nAgg = vd_nAgg(airgroupId);
    var W = vd_W();
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();

    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input proof[svd_proofSize(0)];
    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];

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
}

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

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    for (var i = 0; i < svd_nProofValues(0); i++) { sV.proofvalues[i] <== proofValues[i]; }
    component p = VadcopPublics(nAgg, W, nPub, nPV, svd_nPublics(0), -1);
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
            pubs[s] = VadcopPublics(nAgg, W, nPub, nPV, svd_nPublics(0), -1);
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

    signal aggregationTypes[nAgg] <== AggTypesConsistency(k, nAgg, W, compressed)(sv);
    component out = FoldOutputs(k, nAgg, W, 1);
    out.aggregationTypes <== aggregationTypes;
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
    signal aggregationTypes[nAgg] <== AggTypesConsistency(n, nAgg, W, 0)(sv);

    component vs[n];
    component pubs[n];
    signal {binary} isNull[n * multiAir];
    for (var s = 0; s < n; s++) {
        vs[s] = parallel StarkVerifierGL(0);
        for (var i = 0; i < P; i++) { vs[s].proof[i] <== slots[s][S + i]; }
        for (var i = 0; i < svd_nProofValues(0); i++) { vs[s].proofvalues[i] <== proofValues[i]; }
        pubs[s] = VadcopPublics(nAgg, W, nPub, nPV, nPubV, nPubV - 4);
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
    out.aggregationTypes <== aggregationTypes;
    out.slots <== sv;
    out.isNull <== isNull;
    sv_aggregatedProofs <== out.sv[0];
    for (var i = 0; i < nAgg; i++) {
        sv_aggregationTypes[i] <== out.sv[1 + i];
        for (var e = 0; e < 3; e++) { sv_airgroupvalues[i][e] <== out.sv[1 + nAgg + 3 * i + e]; }
    }
    for (var i = 0; i < W; i++) { sv_stage1Hash[i] <== out.sv[1 + 4 * nAgg + i]; }
}

// The compressed vadcop_final: the incoming proof's publics are [is_vadcop_final_proof | vk(4) | inputs];
// this proof re-exposes the flag-free rest.
template FinalCompressed(nPub, nReal) {
    signal input publics[nPub];
    signal output publics_stripped[nReal];
    signal input proof[svd_proofSize(0)];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    if (nPub > 0) { sV.publics <== publics; }
    for (var i = 0; i < nReal; i++) { publics_stripped[i] <== publics[i + 1]; }
}

// A recursivef (Goldilocks): its verifier with rootC and the publics public.
template Recursivef() {
    signal input rootC[4];
    signal input publics[svd_nPublics(0)];
    signal input proof[svd_proofSize(0)];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    sV.publics <== publics;
    sV.rootCIn <== rootC;
}
