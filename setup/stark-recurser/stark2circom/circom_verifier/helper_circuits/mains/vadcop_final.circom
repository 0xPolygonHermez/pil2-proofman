pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "vadcop/values.circom";
include "vadcop/agg_values.circom";
include "vadcop/verify_child.circom";
include "vadcop/hashes.circom";

/*
    vadcop_final (the conventions of vadcop/values.circom); the generated main adds GlobalConstraints,
    vadcop_nGlobalChallenges() and the global data vadcop_nPVMap() / vadcop_PV() (proofValuesMap's size and stages).
*/
// The vadcop_final of nAg airgroups: one recursive2 proof each (key a), the global challenge and the
// global constraints.
template VadcopFinal(nAg) {
    var nPub = vadcop_nPublics();
    var nPV = vadcop_nProofValues();
    var W = vadcop_W();
    var multiAir = vadcop_multiAir();
    // A null proof leaves its airgroup's values unconstrained, so the global checks must see what
    // an airgroup with no instances contributes; a single airgroup can never be null.
    var maskNull = multiAir == 1 && nAg > 1 ? 1 : 0;
    var maxAgg = 1;
    var T = 0;
    var OFF[nAg];
    for (var a = 0; a < nAg; a++) {
        OFF[a] = T;
        T += vadcopValuesSize(vadcop_nAgg(a), W) + key_proofSize(a);
        if (vadcop_nAgg(a) > maxAgg) { maxAgg = vadcop_nAgg(a); }
    }

    signal output is_vadcop_final_proof <== 1;
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    // Per airgroup: its recursive2's vadcop values, then its proof.
    signal input airgroups[T];

    // An airgroup's recursive2: its rootCAgg is a constant here, and the child's last four publics.
    component child[nAg];
    signal {binary} isNull[nAg];
    signal nAggregatedProofs[nAg];
    signal stage1Hash[nAg][W];
    signal agv[nAg][maxAgg][3];
    for (var a = 0; a < nAg; a++) {
        var nAgg = vadcop_nAgg(a);
        var S = vadcopValuesSize(nAgg, W);
        child[a] = VerifyChild(a, a, key_nPublics(a) - 4, multiAir);
        for (var i = 0; i < S; i++) { child[a].vadcopValues[i] <== airgroups[OFF[a] + i]; }
        for (var i = 0; i < key_proofSize(a); i++) { child[a].proof[i] <== airgroups[OFF[a] + S + i]; }
        child[a].publics <== publics;
        child[a].proofValues <== proofValues;
        child[a].globalChallenge <== globalChallenge;
        child[a].rootCAgg <== vadcop_rootCAgg(a);
        if (multiAir == 1) { isNull[a] <== child[a].isNull[0]; } else { isNull[a] <== IsZero()(airgroups[OFF[a]]); }

        nAggregatedProofs[a] <== airgroups[OFF[a] + 1];
        for (var i = 0; i < W; i++) {
            if (maskNull == 1) {
                stage1Hash[a][i] <== (1 - isNull[a]) * airgroups[OFF[a] + stage1HashOffset(nAgg) + i];
            } else {
                stage1Hash[a][i] <== airgroups[OFF[a] + stage1HashOffset(nAgg) + i];
            }
        }
        var AGG[nonEmpty(nAgg)] = vadcop_AGG(a);
        for (var i = 0; i < maxAgg; i++) {
            var o = OFF[a] + airgroupValuesOffset(nAgg) + 3 * i;
            if (i >= nAgg) {
                agv[a][i] <== [0, 0, 0];
            } else if (maskNull == 1) {
                agv[a][i] <== NullifyAirgroupValue(AGG[i])([airgroups[o], airgroups[o + 1], airgroups[o + 2]], isNull[a]);
            } else {
                agv[a][i] <== [airgroups[o], airgroups[o + 1], airgroups[o + 2]];
            }
        }
    }

    // The global challenge is the transcript of the publics, stage-1 proof values and contributions.
    component verifyChallenges = VerifyGlobalChallenges(nPub, vadcop_nPVMap(), vadcop_PV(), nAg, W);
    if (nPub > 0) { verifyChallenges.publics <== publics; } else { verifyChallenges.publics <== [0]; }
    if (vadcop_nPVMap() > 0) { verifyChallenges.proofValues <== proofValues; } else { verifyChallenges.proofValues <== [[0, 0, 0]]; }
    verifyChallenges.stage1Hash <== stage1Hash;
    verifyChallenges.globalChallenge <== globalChallenge;

    // The global constraints (generated GlobalConstraints) over the padded values and stage-2 challenges.
    var nCh = vadcop_nGlobalChallenges();
    signal challenges[nCh][3];
    if (nCh > 0) { challenges <== GlobalChallenges(nCh)(globalChallenge); }
    component globalConstraints = GlobalConstraints(nAg, maxAgg, nPub, nPV, nCh);
    globalConstraints.airgroupvalues <== agv;
    globalConstraints.publics <== publics;
    globalConstraints.proofValues <== proofValues;
    globalConstraints.challenges <== challenges;

    if (multiAir == 1) {
        signal totalAggregatedProofs <== AggregateProofsNull(nAg)(nAggregatedProofs, isNull);
        signal noAggregatedProofs <== IsZero()(totalAggregatedProofs);
        noAggregatedProofs === 0;
    } else {
        _ <== AggregateProofsNull(nAg)(nAggregatedProofs, isNull);
    }
}

