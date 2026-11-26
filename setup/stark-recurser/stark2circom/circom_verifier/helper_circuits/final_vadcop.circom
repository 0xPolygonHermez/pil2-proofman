pragma circom 2.1.0;
pragma custom_templates;

include "recursion.circom";

/*
    vadcop_final (recursion.circom's conventions); the generated main adds GlobalConstraints,
    vd_nGlobalChallenges() and the global data vd_nPVMap() / vd_PV() (proofValuesMap's size and stages).
*/
// The vadcop_final of nAg airgroups: one recursive2 proof each (key a), the global challenge and the
// global constraints.
template VadcopFinal(nAg) {
    var nPub = vd_nPublics();
    var nPV = vd_nProofValues();
    var W = vd_W();
    var multiAir = vd_multiAir();
    // A null proof leaves its airgroup's values unconstrained, so the global checks must see what
    // an airgroup with no instances contributes; a single airgroup can never be null.
    var maskNull = multiAir == 1 && nAg > 1 ? 1 : 0;
    var maxAgg = 1;
    var T = 0;
    var OFF[nAg];
    for (var a = 0; a < nAg; a++) {
        OFF[a] = T;
        T += sv_size(vd_nAgg(a), W) + svd_proofSize(a);
        if (vd_nAgg(a) > maxAgg) { maxAgg = vd_nAgg(a); }
    }

    signal output is_vadcop_final_proof <== 1;
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    // Per airgroup: its recursive2's vadcop values, then its proof.
    signal input airgroups[T];

    component vs[nAg];
    component pubs[nAg];
    signal {binary} isNull[nAg];
    signal nAggregatedProofs[nAg];
    signal stage1Hash[nAg][W];
    signal agv[nAg][maxAgg][3];
    for (var a = 0; a < nAg; a++) {
        var nAgg = vd_nAgg(a);
        var S = sv_size(nAgg, W);
        var nPubV = svd_nPublics(a);
        var nBasics = vd_nBasics(a);
        var rootCBasics[nBasics][4] = vd_basics(a);
        var rootCAgg[4] = vd_rootCAgg(a);
        vs[a] = StarkVerifierGL(a);
        for (var i = 0; i < svd_proofSize(a); i++) { vs[a].proof[i] <== airgroups[OFF[a] + S + i]; }
        for (var i = 0; i < svd_nProofValues(a); i++) { vs[a].proofvalues[i] <== proofValues[i]; }
        pubs[a] = VadcopPublics(a, nAgg, W, nPub, nPV, nPubV, nPubV - 4);
        for (var i = 0; i < S; i++) { pubs[a].sv[i] <== airgroups[OFF[a] + i]; }
        pubs[a].publics <== publics;
        pubs[a].proofValues <== proofValues;
        pubs[a].globalChallenge <== globalChallenge;
        pubs[a].rootCAgg <== rootCAgg;
        vs[a].publics <== pubs[a].out;
        if (multiAir == 1) {
            isNull[a] <== IsZero()(airgroups[OFF[a]]);
            vs[a].enable[0] <== 1 - isNull[a];
            vs[a].rootCIn <== SelectVerificationKeyNull(nBasics, a)(airgroups[OFF[a]], rootCBasics, rootCAgg);
        } else {
            isNull[a] <== IsZero()(airgroups[OFF[a]]);
            vs[a].rootCIn <== SelectVerificationKey(nBasics, a)(airgroups[OFF[a]], rootCBasics, rootCAgg);
        }
        nAggregatedProofs[a] <== airgroups[OFF[a] + 1];
        for (var i = 0; i < W; i++) {
            if (maskNull == 1) {
                stage1Hash[a][i] <== (1 - isNull[a]) * airgroups[OFF[a] + 2 + 4 * nAgg + i];
            } else {
                stage1Hash[a][i] <== airgroups[OFF[a] + 2 + 4 * nAgg + i];
            }
        }
        var AGG[geo_len1(nAgg)] = vd_AGG(a);
        for (var i = 0; i < maxAgg; i++) {
            if (i >= nAgg) {
                agv[a][i] <== [0, 0, 0];
            } else if (maskNull == 1) {
                var o = OFF[a] + 2 + nAgg + 3 * i;
                agv[a][i] <== NullifyAirgroupValue(AGG[i])([airgroups[o], airgroups[o + 1], airgroups[o + 2]], isNull[a]);
            } else {
                for (var e = 0; e < 3; e++) { agv[a][i][e] <== airgroups[OFF[a] + 2 + nAgg + 3 * i + e]; }
            }
        }
    }

    // The global challenge is the transcript of the publics, stage-1 proof values and contributions.
    component verifyChallenges = VerifyGlobalChallenges(nPub, vd_nPVMap(), vd_PV(), nAg, W);
    if (nPub > 0) { verifyChallenges.publics <== publics; } else { verifyChallenges.publics <== [0]; }
    if (vd_nPVMap() > 0) { verifyChallenges.proofValues <== proofValues; } else { verifyChallenges.proofValues <== [[0, 0, 0]]; }
    verifyChallenges.stage1Hash <== stage1Hash;
    verifyChallenges.globalChallenge <== globalChallenge;

    // The global constraints (generated GlobalConstraints) over the padded values and stage-2 challenges.
    var nCh = vd_nGlobalChallenges();
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

