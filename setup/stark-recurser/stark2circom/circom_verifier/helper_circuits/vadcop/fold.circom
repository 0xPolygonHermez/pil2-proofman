pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/acc_points.circom";
include "vadcop/agg_values.circom";
include "iszero.circom";
include "vadcop/values.circom";

// n children's vadcop values folded into one: airgroup values by each airgroup's own aggregation, stage-1
// hashes by the lattice sum or the EC sum, proof counts added; a null child contributes nothing.

// Two stage-1 hashes combined (null-aware: a null side contributes nothing).
template FoldHash(W, nullAware) {
    signal input a[W];
    signal input b[W];
    signal input isNull[2 * nullAware];
    signal output out[W];

    signal {binary} nb[2 * nullAware];
    nb <== isNull;

    if (vadcop_ec() == 0 && nullAware == 1) {
        out <== AggregateValuesNull(W)(a, b, nb[0], nb[1]);
    } else if (vadcop_ec() == 0) {
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

/*
    n children's vadcop values (slots) folded left to right into the parent's, circuitType aside; nullAware
    drops the slots flagged null. The parent's aggregation types are the airgroup's own.
*/
template FoldSlots(n, nAgg, W, nullAware) {
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(vadcop_airgroupId());
    signal input slots[n][vadcopValuesSize(nAgg, W)];
    signal input isNullIn[n * nullAware];
    signal output aggregatedProofs;
    signal output aggregationTypes[nAgg];
    signal output airgroupvalues[nAgg][3];
    signal output stage1Hash[W];

    component slot[n];
    signal slotAggregatedProofs[n];
    for (var s = 0; s < n; s++) {
        slot[s] = SplitVadcopValues(nAgg, W);
        slot[s].vadcopValues <== slots[s];
        slotAggregatedProofs[s] <== slot[s].aggregatedProofs;
    }
    signal {binary} isNull[n * nullAware];
    isNull <== isNullIn;
    // allNullBefore[j - 1]: slots 0..j-1 are all null, the left side of fold j.
    signal {binary} allNullBefore[(n - 1) * nullAware];
    if (nullAware == 1) {
        allNullBefore[0] <== isNull[0];
        for (var j = 2; j < n; j++) {
            var nNull = 0;
            for (var i = 0; i < j; i++) { nNull += isNull[i]; }
            allNullBefore[j - 1] <== IsZero()(j - nNull);
        }
    }

    // acc*[j]: slots 0..j folded.
    signal accAirgroupValues[n][nAgg][3];
    signal accStage1Hash[n][W];
    if (nAgg > 0) { accAirgroupValues[0] <== slot[0].airgroupvalues; }
    accStage1Hash[0] <== slot[0].stage1Hash;
    component foldHash[n - 1];
    for (var j = 1; j < n; j++) {
        for (var i = 0; i < nAgg; i++) {
            if (nullAware == 1) {
                accAirgroupValues[j][i] <== AggregateAirgroupValuesNull(AGG[i])(accAirgroupValues[j - 1][i], slot[j].airgroupvalues[i], allNullBefore[j - 1], isNull[j]);
            } else {
                accAirgroupValues[j][i] <== AggregateAirgroupValues(AGG[i])(accAirgroupValues[j - 1][i], slot[j].airgroupvalues[i]);
            }
        }
        foldHash[j - 1] = FoldHash(W, nullAware);
        foldHash[j - 1].a <== accStage1Hash[j - 1];
        foldHash[j - 1].b <== slot[j].stage1Hash;
        if (nullAware == 1) { foldHash[j - 1].isNull <== [allNullBefore[j - 1], isNull[j]]; }
        accStage1Hash[j] <== foldHash[j - 1].out;
    }

    if (nullAware == 1) {
        aggregatedProofs <== AggregateProofsNull(n)(slotAggregatedProofs, isNull);
    } else {
        aggregatedProofs <== AggregateProofs(n)(slotAggregatedProofs);
    }
    for (var i = 0; i < nAgg; i++) { aggregationTypes[i] <== AGG[i]; }
    airgroupvalues <== accAirgroupValues[n - 1];
    stage1Hash <== accStage1Hash[n - 1];
}
