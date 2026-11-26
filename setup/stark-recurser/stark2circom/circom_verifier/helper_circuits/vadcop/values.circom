pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/hashes.circom";

// The vadcop values ("vadcopValues") of a proof and what a verified child proof exposes of them.
//
//   vadcopValues = circuitType, aggregatedProofs, aggregationTypes[nAgg], airgroupvalues[nAgg][3], stage1Hash[W]
//   (W = the lattice hash size, or 10: an EC point's two Fp5 limbs).

function vadcopValuesSize(nAgg, W) { return 2 + 4 * nAgg + W; }
// Where airgroupvalues and stage1Hash start in vadcopValues.
function airgroupValuesOffset(nAgg) { return 2 + nAgg; }
function stage1HashOffset(nAgg) { return 2 + 4 * nAgg; }

// vadcopValues into its fields, and back.
template SplitVadcopValues(nAgg, W) {
    signal input vadcopValues[vadcopValuesSize(nAgg, W)];
    signal output circuitType;
    signal output aggregatedProofs;
    signal output aggregationTypes[nAgg];
    signal output airgroupvalues[nAgg][3];
    signal output stage1Hash[W];

    circuitType <== vadcopValues[0];
    aggregatedProofs <== vadcopValues[1];
    for (var i = 0; i < nAgg; i++) {
        aggregationTypes[i] <== vadcopValues[2 + i];
        for (var e = 0; e < 3; e++) { airgroupvalues[i][e] <== vadcopValues[airgroupValuesOffset(nAgg) + 3 * i + e]; }
    }
    for (var i = 0; i < W; i++) { stage1Hash[i] <== vadcopValues[stage1HashOffset(nAgg) + i]; }
}

template JoinVadcopValues(nAgg, W) {
    signal input circuitType;
    signal input aggregatedProofs;
    signal input aggregationTypes[nAgg];
    signal input airgroupvalues[nAgg][3];
    signal input stage1Hash[W];
    signal output vadcopValues[vadcopValuesSize(nAgg, W)];

    vadcopValues[0] <== circuitType;
    vadcopValues[1] <== aggregatedProofs;
    for (var i = 0; i < nAgg; i++) {
        vadcopValues[2 + i] <== aggregationTypes[i];
        for (var e = 0; e < 3; e++) { vadcopValues[airgroupValuesOffset(nAgg) + 3 * i + e] <== airgroupvalues[i][e]; }
    }
    for (var i = 0; i < W; i++) { vadcopValues[stage1HashOffset(nAgg) + i] <== stage1Hash[i]; }
}

// The stage-1 contribution of a proof of key k: its rootC, root1 and air values, as a lattice or EC point.
template Stage1Hash(k, circuitType) {
    var nAV = key_nAirValues(k);
    var W = vadcop_W();

    signal input rootC[4];
    signal input root1[4];
    signal input airvalues[nonEmpty(nAV)][3];
    signal output out[W];

    if (vadcop_ec() == 0) {
        out <== CalculateStage1Hash(circuitType, nAV, key_AIR_VALUE_STAGE(k), W)(rootC, root1, airvalues);
    } else {
        signal P[2][5] <== CalculateStage1HashEC(circuitType, nAV, key_AIR_VALUE_STAGE(k), vadcop_ecA(), vadcop_ecB(), vadcop_ecZ(), vadcop_ecC1(), vadcop_ecC2())(rootC, root1, airvalues);
        for (var i = 0; i < 10; i++) { out[i] <== P[i \ 5][i % 5]; }
    }
}

// The vadcop values a verified basic proof of key k mints (a null slot reports circuitType 0).
template BasicValues(k, circuitType, nAgg, nullable) {
    var W = vadcop_W();
    var nAV = key_nAirValues(k);
    var offAirValues = key_offAirValues(k);
    var offAirgroupValues = key_offAirgroupValues(k);
    var OFF_ROOT[key_nStages(k) + 1] = key_OFF_ROOT(k);
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(vadcop_airgroupId());

    signal input rootC[4];
    signal input proof[key_proofSize(k)];
    signal input isNull[nullable];
    signal output vadcopValues[vadcopValuesSize(nAgg, W)];

    if (nullable == 1) { vadcopValues[0] <== (1 - isNull[0]) * circuitType; } else { vadcopValues[0] <== circuitType; }
    // Stays 1 for an empty slot too: AggregateProofsNull multiplies it out.
    vadcopValues[1] <== 1;
    for (var i = 0; i < nAgg; i++) {
        vadcopValues[2 + i] <== AGG[i];
        for (var e = 0; e < 3; e++) { vadcopValues[airgroupValuesOffset(nAgg) + 3 * i + e] <== proof[offAirgroupValues + 3 * i + e]; }
    }
    component h = Stage1Hash(k, circuitType);
    h.rootC <== rootC;
    for (var i = 0; i < 4; i++) { h.root1[i] <== proof[OFF_ROOT[0] + i]; }
    if (nAV > 0) {
        for (var i = 0; i < nAV; i++) { for (var e = 0; e < 3; e++) { h.airvalues[i][e] <== proof[offAirValues + 3 * i + e]; } }
    } else {
        h.airvalues <== [[0, 0, 0]];
    }
    for (var i = 0; i < W; i++) { vadcopValues[stage1HashOffset(nAgg) + i] <== h.out[i]; }
}

// The publics of a proof of airgroup `airgroupId` that carries vadcop values: vadcopValues, the run's publics and
// proof values, the global challenge, then (rootCAt >= 0) the aggregation key. Its aggregationTypes are
// the airgroup's own (vadcop_AGG), not vadcopValues's: a proof only verifies if it was made with them, which binds
// them even when no slot is verified (all null).
template VadcopPublics(airgroupId, nAgg, W, nPub, nPV, nOut, rootCAt) {
    signal input vadcopValues[vadcopValuesSize(nAgg, W)];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal output out[nOut];

    var n = vadcopValuesSize(nAgg, W);
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(airgroupId);
    for (var i = 0; i < n; i++) {
        if (i >= 2 && i < 2 + nAgg) {
            out[i] <== AGG[i - 2];
            _ <== vadcopValues[i];
        } else {
            out[i] <== vadcopValues[i];
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
