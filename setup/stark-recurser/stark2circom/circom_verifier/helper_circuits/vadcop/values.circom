pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/hashes.circom";

// The vadcop values ("sv") of a proof and what a verified child proof exposes of them.
//
//   sv = circuitType, aggregatedProofs, aggregationTypes[nAgg], airgroupvalues[nAgg][3], stage1Hash[W]
//   (W = the lattice hash size, or 10: an EC point's two Fp5 limbs).

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

// The publics of a proof of airgroup `airgroupId` that carries vadcop values: sv, the run's publics and
// proof values, the global challenge, then (rootCAt >= 0) the aggregation key. Its aggregationTypes are
// the airgroup's own (vd_AGG), not sv's: a proof only verifies if it was made with them, which binds
// them even when no slot is verified (all null).
template VadcopPublics(airgroupId, nAgg, W, nPub, nPV, nOut, rootCAt) {
    signal input sv[sv_size(nAgg, W)];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal output out[nOut];

    var n = sv_size(nAgg, W);
    var AGG[geo_len1(nAgg)] = vd_AGG(airgroupId);
    for (var i = 0; i < n; i++) {
        if (i >= 2 && i < 2 + nAgg) {
            out[i] <== AGG[i - 2];
            _ <== sv[i];
        } else {
            out[i] <== sv[i];
        }
    }
    for (var i = 0; i < nPub; i++) { out[n + i] <== publics[i]; }
    for (var i = 0; i < nPV; i++) { for (var e = 0; e < 3; e++) { out[n + nPub + 3 * i + e] <== proofValues[i][e]; } }
    for (var e = 0; e < 3; e++) { out[n + nPub + 3 * nPV + e] <== globalChallenge[e]; }
    if (rootCAt >= 0) {
        for (var i = 0; i < 4; i++) { out[rootCAt + i] <== rootCAgg[i]; }
    } else {
        _ <== rootCAgg;
    }
}
