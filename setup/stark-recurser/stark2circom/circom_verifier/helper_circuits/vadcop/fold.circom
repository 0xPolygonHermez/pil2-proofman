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

// n slots' values folded left to right; nullAware drops the slots flagged null.
template FoldSlots(n, nAgg, W, nullAware) {
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(vadcop_airgroupId());
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
    signal {binary} prefNull[nonEmpty(n - 2)];
    if (nullAware == 1) {
        for (var j = 2; j < n; j++) {
            var s = 0;
            for (var i = 0; i < j; i++) { s += isNull[i]; }
            prefNull[j - 2] <== IsZero()(j - s);
        }
    }
    if (nullAware == 0 || n <= 2) { prefNull[0] <== 0; }

    // acc[j - 1]: the first j + 1 slots folded.
    signal accAGV[nonEmpty(n - 2)][nAgg][3];
    for (var i = 0; i < nAgg; i++) {
        for (var j = 1; j < n; j++) {
            if (nullAware == 1) {
                if (j == 1 && j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValuesNull(AGG[i])(airgroupvalues[0][i], airgroupvalues[1][i], isNull[0], isNull[1]);
                } else if (j == 1) {
                    accAGV[0][i] <== AggregateAirgroupValuesNull(AGG[i])(airgroupvalues[0][i], airgroupvalues[1][i], isNull[0], isNull[1]);
                } else if (j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValuesNull(AGG[i])(accAGV[j - 2][i], airgroupvalues[j][i], prefNull[j - 2], isNull[j]);
                } else {
                    accAGV[j - 1][i] <== AggregateAirgroupValuesNull(AGG[i])(accAGV[j - 2][i], airgroupvalues[j][i], prefNull[j - 2], isNull[j]);
                }
            } else {
                if (j == 1 && j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValues(AGG[i])(airgroupvalues[0][i], airgroupvalues[1][i]);
                } else if (j == 1) {
                    accAGV[0][i] <== AggregateAirgroupValues(AGG[i])(airgroupvalues[0][i], airgroupvalues[1][i]);
                } else if (j == n - 1) {
                    outAirgroupvalues[i] <== AggregateAirgroupValues(AGG[i])(accAGV[j - 2][i], airgroupvalues[j][i]);
                } else {
                    accAGV[j - 1][i] <== AggregateAirgroupValues(AGG[i])(accAGV[j - 2][i], airgroupvalues[j][i]);
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

    signal accHash[nonEmpty(n - 2)][W];
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

// The fold's outputs: the folded values, and the airgroup's aggregation types.
template FoldOutputs(n, nAgg, W, nullAware) {
    var AGG[nonEmpty(nAgg)] = vadcop_AGG(vadcop_airgroupId());
    signal input slots[n][vadcopValuesSize(nAgg, W)];
    signal input isNull[n * nullAware];
    signal output vadcopValues[vadcopValuesSize(nAgg, W) - 1];

    component f = FoldSlots(n, nAgg, W, nullAware);
    for (var s = 0; s < n; s++) {
        f.aggregatedProofs[s] <== slots[s][1];
        for (var i = 0; i < nAgg; i++) { for (var e = 0; e < 3; e++) { f.airgroupvalues[s][i][e] <== slots[s][2 + nAgg + 3 * i + e]; } }
        for (var i = 0; i < W; i++) { f.stage1Hash[s][i] <== slots[s][2 + 4 * nAgg + i]; }
    }
    f.isNullIn <== isNull;
    // vadcopValues without circuitType: aggregatedProofs, aggregationTypes, airgroupvalues, stage1Hash.
    vadcopValues[0] <== f.outAggregatedProofs;
    for (var i = 0; i < nAgg; i++) {
        vadcopValues[1 + i] <== AGG[i];
        for (var e = 0; e < 3; e++) { vadcopValues[1 + nAgg + 3 * i + e] <== f.outAirgroupvalues[i][e]; }
    }
    for (var i = 0; i < W; i++) { vadcopValues[1 + 4 * nAgg + i] <== f.outStage1Hash[i]; }
}
