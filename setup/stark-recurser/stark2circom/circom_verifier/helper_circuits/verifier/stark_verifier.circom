pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "proof_shape.circom";
include "transcript.circom";
include "deep.circom";
include "fri.circom";
include "stir.circom";

/*
    The GL STARK verifier (FRI or STIR) of the compilation's key k, `StarkVerifierGL(k)` (below): its data
    is the key's `keyData<sfx>_*` functions (lib_params.rs) read through the compilation's `key_*(k)`, its
    constraints the key's `ConstraintRoots<sfx>` through `ConstraintRootsAt(k)`. It wires:

        StarkVerifierTranscript   every challenge and the FRI queries (transcript.circom)
        VerifyEvaluations         the roots fold to Q(xi) (below)
        DeepConstants             the query-independent DEEP terms (deep.circom)
        QueryBatchGL              per query the Merkle openings (merkle.circom), DEEP, FRI folds
        StirVerifyRounds          STIR only: the rounds after the L_0 queries (stir.circom)
*/

/*
    The constraints at xi fold to the quotient: C = sum_i vc^(n-1-i) * root_i (Horner, roots in key
    order: the circom copy of core::fold::fold_roots), C * Zi = Q(xi), Zi = 1 / (xi^N - 1), and
    Q(xi) = sum_i xi^(i N) Q_i(xi) over its qDeg pieces.
*/
template VerifyEvaluations(nBits, nRoots, firstRootDim, qDeg, qPolEv, nEvals) {
    var nR = nRoots;
    var qEv = qPolEv;

    signal input xi[3];
    signal input vc[3];
    signal input roots[nR][3];
    signal input evals[nEvals][3];
    signal input {binary} enable;

    // zMul[i] = xi^(2^(i+1)); Zh = 1 / (xi^N - 1)
    signal zMul[nBits][3];
    for (var i = 0; i < nBits; i++) {
        if (i == 0) {
            zMul[i] <== CMul()(xi, xi);
        } else {
            zMul[i] <== CMul()(zMul[i - 1], zMul[i - 1]);
        }
    }
    signal Z[3] <== [zMul[nBits - 1][0] - 1, zMul[nBits - 1][1], zMul[nBits - 1][2]];
    signal Zh[3] <== CInv()(Z);

    // A base-field first root stays in the base field until its first product with vc.
    signal foldMul[nonEmpty(nR - 1)][3];
    signal foldAcc[nonEmpty(nR - 1)][3];
    var acc[3];
    for (var e = 0; e < 3; e++) { acc[e] = roots[0][e]; }
    var accDim = firstRootDim;
    for (var i = 1; i < nR; i++) {
        if (accDim == 1) {
            foldMul[i - 1] <== [vc[0] * acc[0], vc[1] * acc[0], vc[2] * acc[0]];
        } else {
            foldMul[i - 1] <== CMul()(vc, acc);
        }
        // VERIFY-CONSTRAINTS: a base-field root arrives as [c, 0, 0]; its zero limbs must fold away under --O1.
        foldAcc[i - 1] <== [foldMul[i - 1][0] + roots[i][0], foldMul[i - 1][1] + roots[i][1], foldMul[i - 1][2] + roots[i][2]];
        for (var e = 0; e < 3; e++) { acc[e] = foldAcc[i - 1][e]; }
        accDim = 3;
    }
    signal cZ[3];
    if (accDim == 1) {
        cZ <== [acc[0] * Zh[0], acc[0] * Zh[1], acc[0] * Zh[2]];
    } else {
        cZ <== CMul()(acc, Zh);
    }

    signal xAcc[qDeg][3];
    signal qStep[nonEmpty(qDeg - 1)][3];
    signal qAcc[qDeg][3];
    for (var i = 0; i < qDeg; i++) {
        if (i == 0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[qEv];
        } else {
            xAcc[i] <== CMul()(xAcc[i - 1], zMul[nBits - 1]);
            qStep[i - 1] <== CMul()(xAcc[i], evals[qEv + i]);
            qAcc[i][0] <== qAcc[i - 1][0] + qStep[i - 1][0];
            qAcc[i][1] <== qAcc[i - 1][1] + qStep[i - 1][1];
            qAcc[i][2] <== qAcc[i - 1][2] + qStep[i - 1][2];
        }
    }

    enable * (cZ[0] - qAcc[qDeg - 1][0]) === 0;
    enable * (cZ[1] - qAcc[qDeg - 1][1]) === 0;
    enable * (cZ[2] - qAcc[qDeg - 1][2]) === 0;
}

/*
    One STARK proof of key k (the compilation's key_*(k) data, lib_params.rs), verified from its flat proof
    (Proofs::proof2pointer order; offsets in the data). The constraints come from the key's
    ConstraintRoots through the compilation's ConstraintRootsAt(k).
*/
template StarkVerifierGL(k) {
    var nBits = key_nBits(k);
    var nBitsExt = key_nBitsExt(k);
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nQueries = key_nQueries(k);
    var batchSize = key_batchSize(k);
    var nEvals = key_nEvals(k);
    var nPublics = key_nPublics(k);
    var nAirValues = key_nAirValues(k);
    var nAirgroupValues = key_nAirgroupValues(k);
    var nProofValues = key_nProofValues(k);
    var nStages = key_nStages(k);
    var nChallenges = key_nChallenges(k);
    var circuitType = key_circuitType(k);
    var nRoots = key_nRoots(k);
    var nSteps = key_nSteps(k);
    var stir = key_stir(k);
    var stirM = key_stirM(k);
    var nTrees = key_nTrees(k);
    var nOpenings = key_nOpenings(k);
    var nPols = key_nPols(k);
    var verkeyInput = key_verkeyInput(k);
    var enableInput = key_enableInput(k);
    var offAirgroupValues = key_offAirgroupValues(k);
    var offAirValues = key_offAirValues(k);
    var offEvals = key_offEvals(k);
    var offFinal = key_offFinal(k);
    var offNonce = key_offNonce(k);
    var offBetas = key_offBetas(k);
    var offNonces = key_offNonces(k);
    var NF = key_nFolds(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var STIR_NUM_QUERIES[nonEmpty(stirM)] = key_STIR_NUM_QUERIES(k);
    var ROOT_C[4] = key_ROOT_C(k);
    var OFF_ROOT[nStages + 1] = key_OFF_ROOT(k);
    var OFF_L0_VALUES[nTrees] = key_OFF_L0_VALUES(k);
    var OFF_L0_SIBLINGS[nTrees] = key_OFF_L0_SIBLINGS(k);
    var OFF_L0_LAST_LEVELS[nTrees] = key_OFF_L0_LAST_LEVELS(k);
    var TREE_ROOT[nTrees] = key_TREE_ROOT(k);
    var TREE_PUB[4 * nTrees] = key_TREE_PUB(k);
    var OFF_FOLD_ROOT[nonEmpty(NF)] = key_OFF_FOLD_ROOT(k);
    var OFF_FOLD_VALUES[nonEmpty(NF)] = key_OFF_FOLD_VALUES(k);
    var OFF_FOLD_SIBLINGS[nonEmpty(NF)] = key_OFF_FOLD_SIBLINGS(k);
    var OFF_FOLD_LAST_LEVELS[nonEmpty(NF)] = key_OFF_FOLD_LAST_LEVELS(k);
    var OFF_ANS[nonEmpty(stirM - 1)] = key_OFF_ANS(k);
    var OFF_SHAKE[nonEmpty(stirM - 1)] = key_OFF_SHAKE(k);

    var L0 = l0PathLength(arity, lastLevels, STEPS);
    var wordsPerLevel = siblingWords(arity);
    var lastSize = clearLevelNodes(arity, lastLevels);
    var F = finalPolSize(stir, nSteps, STEPS, key_stirFinalBits(k));
    var inCh = circuitType > 0 ? 1 : 0;
    // Fold trees opened at the L_0 queries; STIR's only one there is T_0.
    var nq = stir == 1 ? 1 : NF;

    signal input proof[key_proofSize(k)];
    signal input publics[nPublics];
    signal input proofvalues[nProofValues][3];
    signal input globalChallenge[3 * inCh];
    signal input rootCIn[4 * verkeyInput];
    signal input enable[enableInput];
    signal output rootC[4];

    if (verkeyInput == 1) { rootC <== rootCIn; } else { rootC <== ROOT_C; }
    signal {binary} enabled;
    if (enableInput == 1) {
        enable[0] * (enable[0] - 1) === 0;
        enabled <== enable[0];
    } else {
        enabled <== 1;
    }

    // The proof's small parts; the queries' data goes from the proof straight into the batches.
    signal evals[nEvals][3];
    for (var i = 0; i < nEvals; i++) { for (var e = 0; e < 3; e++) { evals[i][e] <== proof[offEvals + 3 * i + e]; } }
    signal airvalues[nAirValues][3];
    for (var i = 0; i < nAirValues; i++) { for (var e = 0; e < 3; e++) { airvalues[i][e] <== proof[offAirValues + 3 * i + e]; } }
    signal airgroupvalues[nAirgroupValues][3];
    for (var i = 0; i < nAirgroupValues; i++) { for (var e = 0; e < 3; e++) { airgroupvalues[i][e] <== proof[offAirgroupValues + 3 * i + e]; } }
    signal finalPol[F][3];
    for (var i = 0; i < F; i++) { for (var e = 0; e < 3; e++) { finalPol[i][e] <== proof[offFinal + 3 * i + e]; } }
    // Each L_0 tree's root (a stage root, rootC, or a custom commit's publics) and levels sent in clear.
    signal treeRoot[nTrees][4];
    for (var t = 0; t < nTrees; t++) {
        for (var i = 0; i < 4; i++) {
            if (TREE_ROOT[t] >= 0) {
                treeRoot[t][i] <== proof[TREE_ROOT[t] + i];
            } else if (TREE_ROOT[t] == -1) {
                treeRoot[t][i] <== rootC[i];
            } else {
                treeRoot[t][i] <== publics[TREE_PUB[4 * t + i]];
            }
        }
    }
    signal treeLast[nTrees][lastSize][4];
    signal foldRoot[NF][4];
    signal foldLast[NF][lastSize][4];
    for (var t = 0; t < nTrees; t++) {
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < 4; i++) {
                if (lastLevels > 0) { treeLast[t][j][i] <== proof[OFF_L0_LAST_LEVELS[t] + 4 * j + i]; } else { treeLast[t][j][i] <== 0; }
            }
        }
    }
    for (var f = 0; f < NF; f++) {
        for (var i = 0; i < 4; i++) { foldRoot[f][i] <== proof[OFF_FOLD_ROOT[f] + i]; }
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < 4; i++) {
                if (lastLevels > 0) { foldLast[f][j][i] <== proof[OFF_FOLD_LAST_LEVELS[f] + 4 * j + i]; } else { foldLast[f][j][i] <== 0; }
            }
        }
    }

    // STIR's Ans/shake hints, flat, once for the transcript and the rounds.
    signal stirHints[nonEmpty(stir_nHints(stir, stirM, STIR_NUM_QUERIES))][3];
    if (stir == 1) {
        for (var r = 0; r < stirM - 1; r++) {
            var h0 = stir_hintOff(STIR_NUM_QUERIES, r);
            for (var m = 0; m <= STIR_NUM_QUERIES[r]; m++) {
                for (var e = 0; e < 3; e++) {
                    stirHints[h0 + m][e] <== proof[OFF_ANS[r] + 3 * m + e];
                    stirHints[h0 + STIR_NUM_QUERIES[r] + 1 + m][e] <== proof[OFF_SHAKE[r] + 3 * m + e];
                }
            }
        }
        if (stirM == 1) { stirHints <== [[0, 0, 0]]; }
    } else {
        stirHints <== [[0, 0, 0]];
    }

    component transcript = StarkVerifierTranscript(nStages, nPublics, nAirValues, nAirgroupValues, nEvals, nChallenges, nSteps, STEPS, stir, stirM, key_stirFinalBits(k), STIR_NUM_QUERIES, key_hashCommits(k), circuitType, nQueries, key_powBits(k), key_CH(k), key_AIR_VALUE_STAGE(k), key_AIRGROUP_VALUE_STAGE(k));
    if (inCh == 1) { transcript.globalChallenge <== globalChallenge; } else { transcript.globalChallenge <== [0, 0, 0]; }
    if (nPublics > 0) { transcript.publics <== publics; } else { transcript.publics <== [0]; }
    transcript.rootC <== rootC;
    for (var s = 0; s <= nStages; s++) { for (var i = 0; i < 4; i++) { transcript.stageRoots[s][i] <== proof[OFF_ROOT[s] + i]; } }
    if (nAirValues > 0) { transcript.airvalues <== airvalues; } else { transcript.airvalues <== [[0, 0, 0]]; }
    if (nAirgroupValues > 0) { transcript.airgroupvalues <== airgroupvalues; } else { transcript.airgroupvalues <== [[0, 0, 0]]; }
    transcript.evals <== evals;
    if (NF > 0) { transcript.foldRoots <== foldRoot; } else { transcript.foldRoots <== [[0, 0, 0, 0]]; }
    if (offBetas >= 0) {
        for (var i = 0; i < stirM - 1; i++) { for (var e = 0; e < 3; e++) { transcript.betas[i][e] <== proof[offBetas + 3 * i + e]; } }
    } else {
        transcript.betas <== [[0, 0, 0]];
    }
    transcript.hints <== stirHints;
    transcript.finalPol <== finalPol;
    if (stir == 1) {
        transcript.nonce <== proof[offNonces];
    } else if (offNonce >= 0) {
        transcript.nonce <== proof[offNonce];
    } else {
        transcript.nonce <== 0;
    }
    transcript.enable <== enabled;

    component cr = ConstraintRootsAt(k);
    cr.evals <== evals;
    cr.challenges <== transcript.challenges;
    if (nPublics > 0) { cr.publics <== publics; } else { cr.publics <== [0]; }
    if (nProofValues > 0) { cr.proofvalues <== proofvalues; } else { cr.proofvalues <== [[0, 0, 0]]; }
    if (nAirValues > 0) { cr.airvalues <== airvalues; } else { cr.airvalues <== [[0, 0, 0]]; }
    if (nAirgroupValues > 0) { cr.airgroupvalues <== airgroupvalues; } else { cr.airgroupvalues <== [[0, 0, 0]]; }
    signal roots[nRoots][3] <== cr.roots;

    // challenges end with vc, xi, vf1, vf2.
    VerifyEvaluations(nBits, nRoots, key_firstRootDim(k), key_qDeg(k), key_qPolEv(k), nEvals)(transcript.challenges[nChallenges - 3], transcript.challenges[nChallenges - 4], roots, evals, enabled);

    component deepConsts = DeepConstants(nOpenings, nPols, nEvals, key_OPENING_NUM_EVALS(k), key_EVAL_COLUMN(k));
    deepConsts.vf[0] <== transcript.challenges[nChallenges - 2];
    deepConsts.vf[1] <== transcript.challenges[nChallenges - 1];
    deepConsts.evals <== evals;

    // The L_0 queries in batches of batchSize, the rest in one tail batch.
    var nFull = nQueries \ batchSize;
    var nTail = nQueries - nFull * batchSize;
    component batch[nonEmpty(nFull)];
    component tail[1];
    // Per batch: its per-proof inputs, then its queries' data, so it starts as soon as it is wired.
    var W[8];
    var C[8];
    var M[8];
    for (var j = 0; j < 8; j++) {
        W[j] = geo_slotW(L0_TREE_START, nTrees, j);
        C[j] = geo_slotC(STEPS, nq, j);
        M[j] = geo_slotM(arity, lastLevels, STEPS, nq, j);
    }
    for (var b = 0; b < nFull; b++) {
        batch[b] = QueryBatchGL(k, batchSize);
        var n = batchSize;
        var q0 = b * batchSize;
        batch[b].xi <== transcript.challenges[nChallenges - 3];
        batch[b].friConsts <== deepConsts.friConsts;
        batch[b].last <== treeLast;
        batch[b].root <== treeRoot;
        for (var f = 0; f < nq; f++) {
            batch[b].flast[f] <== foldLast[f];
            batch[b].froot[f] <== foldRoot[f];
            if (stir == 0) { batch[b].rFold[f] <== transcript.ldt[f]; }
        }
        if (stir == 0) { batch[b].finalPol <== finalPol; }
        batch[b].enable <== enabled;
        for (var q = 0; q < n; q++) {
            var g = q0 + q;
            batch[b].queriesL0[q] <== transcript.queriesL0[g];
            for (var t = 0; t < nTrees; t++) { for (var l = 0; l < L0; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].sibs[t][q][l][i] <== proof[OFF_L0_SIBLINGS[t] + (g * L0 + l) * wordsPerLevel + i]; } } }
            for (var i = 0; i < W[0]; i++) { batch[b].v0[q][i][0] <== proof[OFF_L0_VALUES[0] + g * W[0] + i]; }
            for (var i = 0; i < W[1]; i++) { batch[b].v1[q][i][0] <== proof[OFF_L0_VALUES[1] + g * W[1] + i]; }
            for (var i = 0; i < W[2]; i++) { batch[b].v2[q][i][0] <== proof[OFF_L0_VALUES[2] + g * W[2] + i]; }
            for (var i = 0; i < W[3]; i++) { batch[b].v3[q][i][0] <== proof[OFF_L0_VALUES[3] + g * W[3] + i]; }
            for (var i = 0; i < W[4]; i++) { batch[b].v4[q][i][0] <== proof[OFF_L0_VALUES[4] + g * W[4] + i]; }
            for (var i = 0; i < W[5]; i++) { batch[b].v5[q][i][0] <== proof[OFF_L0_VALUES[5] + g * W[5] + i]; }
            for (var i = 0; i < W[6]; i++) { batch[b].v6[q][i][0] <== proof[OFF_L0_VALUES[6] + g * W[6] + i]; }
            for (var i = 0; i < W[7]; i++) { batch[b].v7[q][i][0] <== proof[OFF_L0_VALUES[7] + g * W[7] + i]; }
            for (var c = 0; c < C[0]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv0[q][c][e] <== proof[OFF_FOLD_VALUES[0] + (g * C[0] + c) * 3 + e]; } }
            for (var l = 0; l < M[0]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs0[q][l][i] <== proof[OFF_FOLD_SIBLINGS[0] + (g * M[0] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[1]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv1[q][c][e] <== proof[OFF_FOLD_VALUES[1] + (g * C[1] + c) * 3 + e]; } }
            for (var l = 0; l < M[1]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs1[q][l][i] <== proof[OFF_FOLD_SIBLINGS[1] + (g * M[1] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[2]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv2[q][c][e] <== proof[OFF_FOLD_VALUES[2] + (g * C[2] + c) * 3 + e]; } }
            for (var l = 0; l < M[2]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs2[q][l][i] <== proof[OFF_FOLD_SIBLINGS[2] + (g * M[2] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[3]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv3[q][c][e] <== proof[OFF_FOLD_VALUES[3] + (g * C[3] + c) * 3 + e]; } }
            for (var l = 0; l < M[3]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs3[q][l][i] <== proof[OFF_FOLD_SIBLINGS[3] + (g * M[3] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[4]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv4[q][c][e] <== proof[OFF_FOLD_VALUES[4] + (g * C[4] + c) * 3 + e]; } }
            for (var l = 0; l < M[4]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs4[q][l][i] <== proof[OFF_FOLD_SIBLINGS[4] + (g * M[4] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[5]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv5[q][c][e] <== proof[OFF_FOLD_VALUES[5] + (g * C[5] + c) * 3 + e]; } }
            for (var l = 0; l < M[5]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs5[q][l][i] <== proof[OFF_FOLD_SIBLINGS[5] + (g * M[5] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[6]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv6[q][c][e] <== proof[OFF_FOLD_VALUES[6] + (g * C[6] + c) * 3 + e]; } }
            for (var l = 0; l < M[6]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs6[q][l][i] <== proof[OFF_FOLD_SIBLINGS[6] + (g * M[6] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[7]; c++) { for (var e = 0; e < 3; e++) { batch[b].fv7[q][c][e] <== proof[OFF_FOLD_VALUES[7] + (g * C[7] + c) * 3 + e]; } }
            for (var l = 0; l < M[7]; l++) { for (var i = 0; i < wordsPerLevel; i++) { batch[b].fs7[q][l][i] <== proof[OFF_FOLD_SIBLINGS[7] + (g * M[7] + l) * wordsPerLevel + i]; } }
        }
    }
    if (nTail > 0) {
        tail[0] = QueryBatchGL(k, nTail);
        var n = nTail;
        var q0 = nFull * batchSize;
        tail[0].xi <== transcript.challenges[nChallenges - 3];
        tail[0].friConsts <== deepConsts.friConsts;
        tail[0].last <== treeLast;
        tail[0].root <== treeRoot;
        for (var f = 0; f < nq; f++) {
            tail[0].flast[f] <== foldLast[f];
            tail[0].froot[f] <== foldRoot[f];
            if (stir == 0) { tail[0].rFold[f] <== transcript.ldt[f]; }
        }
        if (stir == 0) { tail[0].finalPol <== finalPol; }
        tail[0].enable <== enabled;
        for (var q = 0; q < n; q++) {
            var g = q0 + q;
            tail[0].queriesL0[q] <== transcript.queriesL0[g];
            for (var t = 0; t < nTrees; t++) { for (var l = 0; l < L0; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].sibs[t][q][l][i] <== proof[OFF_L0_SIBLINGS[t] + (g * L0 + l) * wordsPerLevel + i]; } } }
            for (var i = 0; i < W[0]; i++) { tail[0].v0[q][i][0] <== proof[OFF_L0_VALUES[0] + g * W[0] + i]; }
            for (var i = 0; i < W[1]; i++) { tail[0].v1[q][i][0] <== proof[OFF_L0_VALUES[1] + g * W[1] + i]; }
            for (var i = 0; i < W[2]; i++) { tail[0].v2[q][i][0] <== proof[OFF_L0_VALUES[2] + g * W[2] + i]; }
            for (var i = 0; i < W[3]; i++) { tail[0].v3[q][i][0] <== proof[OFF_L0_VALUES[3] + g * W[3] + i]; }
            for (var i = 0; i < W[4]; i++) { tail[0].v4[q][i][0] <== proof[OFF_L0_VALUES[4] + g * W[4] + i]; }
            for (var i = 0; i < W[5]; i++) { tail[0].v5[q][i][0] <== proof[OFF_L0_VALUES[5] + g * W[5] + i]; }
            for (var i = 0; i < W[6]; i++) { tail[0].v6[q][i][0] <== proof[OFF_L0_VALUES[6] + g * W[6] + i]; }
            for (var i = 0; i < W[7]; i++) { tail[0].v7[q][i][0] <== proof[OFF_L0_VALUES[7] + g * W[7] + i]; }
            for (var c = 0; c < C[0]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv0[q][c][e] <== proof[OFF_FOLD_VALUES[0] + (g * C[0] + c) * 3 + e]; } }
            for (var l = 0; l < M[0]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs0[q][l][i] <== proof[OFF_FOLD_SIBLINGS[0] + (g * M[0] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[1]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv1[q][c][e] <== proof[OFF_FOLD_VALUES[1] + (g * C[1] + c) * 3 + e]; } }
            for (var l = 0; l < M[1]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs1[q][l][i] <== proof[OFF_FOLD_SIBLINGS[1] + (g * M[1] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[2]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv2[q][c][e] <== proof[OFF_FOLD_VALUES[2] + (g * C[2] + c) * 3 + e]; } }
            for (var l = 0; l < M[2]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs2[q][l][i] <== proof[OFF_FOLD_SIBLINGS[2] + (g * M[2] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[3]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv3[q][c][e] <== proof[OFF_FOLD_VALUES[3] + (g * C[3] + c) * 3 + e]; } }
            for (var l = 0; l < M[3]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs3[q][l][i] <== proof[OFF_FOLD_SIBLINGS[3] + (g * M[3] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[4]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv4[q][c][e] <== proof[OFF_FOLD_VALUES[4] + (g * C[4] + c) * 3 + e]; } }
            for (var l = 0; l < M[4]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs4[q][l][i] <== proof[OFF_FOLD_SIBLINGS[4] + (g * M[4] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[5]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv5[q][c][e] <== proof[OFF_FOLD_VALUES[5] + (g * C[5] + c) * 3 + e]; } }
            for (var l = 0; l < M[5]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs5[q][l][i] <== proof[OFF_FOLD_SIBLINGS[5] + (g * M[5] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[6]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv6[q][c][e] <== proof[OFF_FOLD_VALUES[6] + (g * C[6] + c) * 3 + e]; } }
            for (var l = 0; l < M[6]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs6[q][l][i] <== proof[OFF_FOLD_SIBLINGS[6] + (g * M[6] + l) * wordsPerLevel + i]; } }
            for (var c = 0; c < C[7]; c++) { for (var e = 0; e < 3; e++) { tail[0].fv7[q][c][e] <== proof[OFF_FOLD_VALUES[7] + (g * C[7] + c) * 3 + e]; } }
            for (var l = 0; l < M[7]; l++) { for (var i = 0; i < wordsPerLevel; i++) { tail[0].fs7[q][l][i] <== proof[OFF_FOLD_SIBLINGS[7] + (g * M[7] + l) * wordsPerLevel + i]; } }
        }
    }

    // Once per proof: the levels sent in clear hash to the committed roots, finalPol's degree.
    if (lastLevels > 0) {
        for (var t = 0; t < nTrees; t++) { VerifyMerkleRoot(lastLevels, arity, 1 << nBitsExt)(treeLast[t], treeRoot[t], enabled); }
        for (var f = 0; f < nq; f++) { VerifyMerkleRoot(lastLevels, arity, 1 << STEPS[f + 1])(foldLast[f], foldRoot[f], enabled); }
    }
    if (stir == 1) {
        var STIR_LOG_DOMAIN_SIZES[stirM + 1] = key_STIR_LOG_DOMAIN_SIZES(k);
        var STIR_FOLDING_FACTORS[stirM] = key_STIR_FOLDING_FACTORS(k);
        signal stirVals[stir_valsOff(STIR_NUM_QUERIES, STIR_FOLDING_FACTORS, stirM)][3];
        signal stirSibs[nonEmpty(stir_sibOff(arity, lastLevels, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, stirM))][wordsPerLevel];
        for (var r = 0; r < stirM; r++) {
            var vo = stir_valsOff(STIR_NUM_QUERIES, STIR_FOLDING_FACTORS, r);
            for (var q = 0; q < STIR_NUM_QUERIES[r]; q++) {
                for (var c = 0; c < (1 << STIR_FOLDING_FACTORS[r]); c++) {
                    for (var e = 0; e < 3; e++) { stirVals[vo + (q << STIR_FOLDING_FACTORS[r]) + c][e] <== proof[OFF_FOLD_VALUES[r] + (q << STIR_FOLDING_FACTORS[r]) * 3 + c * 3 + e]; }
                }
            }
            if (r > 0 && OFF_FOLD_SIBLINGS[r] >= 0) {
                var ml = merklePathLength(arity, lastLevels, STIR_LOG_DOMAIN_SIZES[r] - STIR_FOLDING_FACTORS[r]);
                var so = stir_sibOff(arity, lastLevels, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, r);
                for (var q = 0; q < STIR_NUM_QUERIES[r]; q++) {
                    for (var j = 0; j < ml; j++) {
                        for (var i = 0; i < wordsPerLevel; i++) { stirSibs[so + q * ml + j][i] <== proof[OFF_FOLD_SIBLINGS[r] + (q * ml + j) * wordsPerLevel + i]; }
                    }
                }
            }
        }
        if (stir_sibOff(arity, lastLevels, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, stirM) == 0) { for (var i = 0; i < wordsPerLevel; i++) { stirSibs[0][i] <== 0; } }

        component rounds = StirVerifyRounds(arity, lastLevels, stirM, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, key_STIR_GRINDING_BITS(k), key_stirFinalBits(k));
        rounds.queriesL0 <== transcript.queriesL0;
        rounds.ldt <== transcript.ldt;
        for (var r = 0; r < stirM; r++) { rounds.nonces[r] <== proof[offNonces + r]; }
        rounds.vals <== stirVals;
        rounds.siblings <== stirSibs;
        for (var r = 0; r < stirM; r++) {
            if (lastLevels > 0) { rounds.last[r] <== foldLast[r]; } else { rounds.last[r] <== [[0, 0, 0, 0]]; }
            rounds.roots[r] <== foldRoot[r];
        }
        rounds.hints <== stirHints;
        if (offBetas >= 0) {
            for (var i = 0; i < stirM - 1; i++) { for (var e = 0; e < 3; e++) { rounds.betas[i][e] <== proof[offBetas + 3 * i + e]; } }
        } else {
            rounds.betas <== [[0, 0, 0]];
        }
        rounds.finalPol <== finalPol;
        rounds.enable <== enabled;
    } else {
        VerifyFinalPol(STEPS[nSteps - 1], finalPolDegreeBound(nBits, nBitsExt, nSteps, STEPS))(finalPol, enabled);
    }
}

/*
    One batch of queries: per L_0 tree and per fold tree (STIR: T_0) a component array of a single
    template instance (circom resolves an array mixing instances through a map at run time), DEEP and
    (FRI) every folding step. Slots v<j>/fv<j>/fs<j> past the key's trees and steps are empty.
*/
template parallel QueryBatchGL(k, n) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nBitsExt = key_nBitsExt(k);
    var nSteps = key_nSteps(k);
    var stir = key_stir(k);
    var nTrees = key_nTrees(k);
    var nOpenings = key_nOpenings(k);
    var nPols = key_nPols(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var L0 = l0PathLength(arity, lastLevels, STEPS);
    var wordsPerLevel = siblingWords(arity);
    var lastSize = clearLevelNodes(arity, lastLevels);
    var fri = 1 - stir;
    var nq = stir == 1 ? 1 : nSteps - 1;
    assert(nTrees <= 8 && nq <= 8);
    var zeroSib[1][wordsPerLevel];
    for (var i = 0; i < wordsPerLevel; i++) { zeroSib[0][i] = 0; }

    signal input {binary} queriesL0[n][STEPS[0]];
    signal input xi[3];
    signal input friConsts[2 + nPols + 2 * nOpenings][3];
    signal input sibs[nTrees][n][L0][wordsPerLevel];
    signal input last[nTrees][lastSize][4];
    signal input root[nTrees][4];
    signal input flast[nq][lastSize][4];
    signal input froot[nq][4];
    signal input rFold[nq * fri][3];
    signal input finalPol[fri * lastDomainSize(nSteps, STEPS)][3];
    signal input {binary} enable;

    signal input v0[n][geo_slotW(L0_TREE_START, nTrees, 0)][1];
    signal input v1[n][geo_slotW(L0_TREE_START, nTrees, 1)][1];
    signal input v2[n][geo_slotW(L0_TREE_START, nTrees, 2)][1];
    signal input v3[n][geo_slotW(L0_TREE_START, nTrees, 3)][1];
    signal input v4[n][geo_slotW(L0_TREE_START, nTrees, 4)][1];
    signal input v5[n][geo_slotW(L0_TREE_START, nTrees, 5)][1];
    signal input v6[n][geo_slotW(L0_TREE_START, nTrees, 6)][1];
    signal input v7[n][geo_slotW(L0_TREE_START, nTrees, 7)][1];
    signal input fv0[n][geo_slotC(STEPS, nq, 0)][3];
    signal input fv1[n][geo_slotC(STEPS, nq, 1)][3];
    signal input fv2[n][geo_slotC(STEPS, nq, 2)][3];
    signal input fv3[n][geo_slotC(STEPS, nq, 3)][3];
    signal input fv4[n][geo_slotC(STEPS, nq, 4)][3];
    signal input fv5[n][geo_slotC(STEPS, nq, 5)][3];
    signal input fv6[n][geo_slotC(STEPS, nq, 6)][3];
    signal input fv7[n][geo_slotC(STEPS, nq, 7)][3];
    signal input fs0[n][geo_slotM(arity, lastLevels, STEPS, nq, 0)][wordsPerLevel];
    signal input fs1[n][geo_slotM(arity, lastLevels, STEPS, nq, 1)][wordsPerLevel];
    signal input fs2[n][geo_slotM(arity, lastLevels, STEPS, nq, 2)][wordsPerLevel];
    signal input fs3[n][geo_slotM(arity, lastLevels, STEPS, nq, 3)][wordsPerLevel];
    signal input fs4[n][geo_slotM(arity, lastLevels, STEPS, nq, 4)][wordsPerLevel];
    signal input fs5[n][geo_slotM(arity, lastLevels, STEPS, nq, 5)][wordsPerLevel];
    signal input fs6[n][geo_slotM(arity, lastLevels, STEPS, nq, 6)][wordsPerLevel];
    signal input fs7[n][geo_slotM(arity, lastLevels, STEPS, nq, 7)][wordsPerLevel];

    component keys0[n];
    component deep[n];
    for (var q = 0; q < n; q++) {
        // One key split per query, shared by the L_0 trees.
        keys0[q] = QueryKeys(arity, L0 + lastLevels, STEPS[0], STEPS[0]);
        keys0[q].queries <== queriesL0[q];
        deep[q] = DeepQuery(nOpenings, nPols, key_nColGroups(k), key_nSums(k), STEPS[0], nextFoldDomainBits(STEPS, nSteps, 0), L0_TREE_START[nTrees], key_OPENING_ROOT(k), key_SUMS(k), key_COLS(k));
        deep[q].queriesL0 <== queriesL0[q];
        deep[q].xi <== xi;
        deep[q].friConsts <== friConsts;
        if (nq > 0) { deep[q].s1_vals <== fv0[q]; } else { deep[q].s1_vals <== finalPol; }
        deep[q].enable <== enable;

        if (nTrees > 0) {
            VerifyTreeOpening(1, L0_TREE_START[0 + 1] - L0_TREE_START[0], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v0[q], sibs[0][q], last[0], root[0], enable);
            for (var i = 0; i < L0_TREE_START[0 + 1] - L0_TREE_START[0]; i++) { deep[q].s0_vals[L0_TREE_START[0] + i] <== v0[q][i][0]; }
        }
        if (nTrees > 1) {
            VerifyTreeOpening(1, L0_TREE_START[1 + 1] - L0_TREE_START[1], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v1[q], sibs[1][q], last[1], root[1], enable);
            for (var i = 0; i < L0_TREE_START[1 + 1] - L0_TREE_START[1]; i++) { deep[q].s0_vals[L0_TREE_START[1] + i] <== v1[q][i][0]; }
        }
        if (nTrees > 2) {
            VerifyTreeOpening(1, L0_TREE_START[2 + 1] - L0_TREE_START[2], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v2[q], sibs[2][q], last[2], root[2], enable);
            for (var i = 0; i < L0_TREE_START[2 + 1] - L0_TREE_START[2]; i++) { deep[q].s0_vals[L0_TREE_START[2] + i] <== v2[q][i][0]; }
        }
        if (nTrees > 3) {
            VerifyTreeOpening(1, L0_TREE_START[3 + 1] - L0_TREE_START[3], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v3[q], sibs[3][q], last[3], root[3], enable);
            for (var i = 0; i < L0_TREE_START[3 + 1] - L0_TREE_START[3]; i++) { deep[q].s0_vals[L0_TREE_START[3] + i] <== v3[q][i][0]; }
        }
        if (nTrees > 4) {
            VerifyTreeOpening(1, L0_TREE_START[4 + 1] - L0_TREE_START[4], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v4[q], sibs[4][q], last[4], root[4], enable);
            for (var i = 0; i < L0_TREE_START[4 + 1] - L0_TREE_START[4]; i++) { deep[q].s0_vals[L0_TREE_START[4] + i] <== v4[q][i][0]; }
        }
        if (nTrees > 5) {
            VerifyTreeOpening(1, L0_TREE_START[5 + 1] - L0_TREE_START[5], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v5[q], sibs[5][q], last[5], root[5], enable);
            for (var i = 0; i < L0_TREE_START[5 + 1] - L0_TREE_START[5]; i++) { deep[q].s0_vals[L0_TREE_START[5] + i] <== v5[q][i][0]; }
        }
        if (nTrees > 6) {
            VerifyTreeOpening(1, L0_TREE_START[6 + 1] - L0_TREE_START[6], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v6[q], sibs[6][q], last[6], root[6], enable);
            for (var i = 0; i < L0_TREE_START[6 + 1] - L0_TREE_START[6]; i++) { deep[q].s0_vals[L0_TREE_START[6] + i] <== v6[q][i][0]; }
        }
        if (nTrees > 7) {
            VerifyTreeOpening(1, L0_TREE_START[7 + 1] - L0_TREE_START[7], arity, lastLevels, L0, nBitsExt)(keys0[q].keys, v7[q], sibs[7][q], last[7], root[7], enable);
            for (var i = 0; i < L0_TREE_START[7 + 1] - L0_TREE_START[7]; i++) { deep[q].s0_vals[L0_TREE_START[7] + i] <== v7[q][i][0]; }
        }
        if (nq > 0) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 1);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 1), arity, lastLevels, ml, STEPS[1])(QueryKeys(arity, ml + lastLevels, STEPS[1], STEPS[0])(queriesL0[q]), fv0[q], fs0[q], flast[0], froot[0], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 1), arity, lastLevels, 0, STEPS[1])(QueryKeys(arity, lastLevels, STEPS[1], STEPS[0])(queriesL0[q]), fv0[q], zeroSib, flast[0], froot[0], enable);
            }
            if (fri == 1 && nq > 1) {
                VerifyFRIFold(STEPS[0], STEPS[1], nextFoldDomainBits(STEPS, nSteps, 1), foldCosetShiftInv(nBitsExt, STEPS, 1), STEPS[0])(queriesL0[q], rFold[0], fv0[q], fv1[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[0], STEPS[1], 0, foldCosetShiftInv(nBitsExt, STEPS, 1), STEPS[0])(queriesL0[q], rFold[0], fv0[q], finalPol, enable);
            }
        }
        if (nq > 1) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 2);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 2), arity, lastLevels, ml, STEPS[2])(QueryKeys(arity, ml + lastLevels, STEPS[2], STEPS[0])(queriesL0[q]), fv1[q], fs1[q], flast[1], froot[1], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 2), arity, lastLevels, 0, STEPS[2])(QueryKeys(arity, lastLevels, STEPS[2], STEPS[0])(queriesL0[q]), fv1[q], zeroSib, flast[1], froot[1], enable);
            }
            if (fri == 1 && nq > 2) {
                VerifyFRIFold(STEPS[1], STEPS[2], nextFoldDomainBits(STEPS, nSteps, 2), foldCosetShiftInv(nBitsExt, STEPS, 2), STEPS[0])(queriesL0[q], rFold[1], fv1[q], fv2[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[1], STEPS[2], 0, foldCosetShiftInv(nBitsExt, STEPS, 2), STEPS[0])(queriesL0[q], rFold[1], fv1[q], finalPol, enable);
            }
        }
        if (nq > 2) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 3);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 3), arity, lastLevels, ml, STEPS[3])(QueryKeys(arity, ml + lastLevels, STEPS[3], STEPS[0])(queriesL0[q]), fv2[q], fs2[q], flast[2], froot[2], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 3), arity, lastLevels, 0, STEPS[3])(QueryKeys(arity, lastLevels, STEPS[3], STEPS[0])(queriesL0[q]), fv2[q], zeroSib, flast[2], froot[2], enable);
            }
            if (fri == 1 && nq > 3) {
                VerifyFRIFold(STEPS[2], STEPS[3], nextFoldDomainBits(STEPS, nSteps, 3), foldCosetShiftInv(nBitsExt, STEPS, 3), STEPS[0])(queriesL0[q], rFold[2], fv2[q], fv3[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[2], STEPS[3], 0, foldCosetShiftInv(nBitsExt, STEPS, 3), STEPS[0])(queriesL0[q], rFold[2], fv2[q], finalPol, enable);
            }
        }
        if (nq > 3) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 4);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 4), arity, lastLevels, ml, STEPS[4])(QueryKeys(arity, ml + lastLevels, STEPS[4], STEPS[0])(queriesL0[q]), fv3[q], fs3[q], flast[3], froot[3], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 4), arity, lastLevels, 0, STEPS[4])(QueryKeys(arity, lastLevels, STEPS[4], STEPS[0])(queriesL0[q]), fv3[q], zeroSib, flast[3], froot[3], enable);
            }
            if (fri == 1 && nq > 4) {
                VerifyFRIFold(STEPS[3], STEPS[4], nextFoldDomainBits(STEPS, nSteps, 4), foldCosetShiftInv(nBitsExt, STEPS, 4), STEPS[0])(queriesL0[q], rFold[3], fv3[q], fv4[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[3], STEPS[4], 0, foldCosetShiftInv(nBitsExt, STEPS, 4), STEPS[0])(queriesL0[q], rFold[3], fv3[q], finalPol, enable);
            }
        }
        if (nq > 4) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 5);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 5), arity, lastLevels, ml, STEPS[5])(QueryKeys(arity, ml + lastLevels, STEPS[5], STEPS[0])(queriesL0[q]), fv4[q], fs4[q], flast[4], froot[4], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 5), arity, lastLevels, 0, STEPS[5])(QueryKeys(arity, lastLevels, STEPS[5], STEPS[0])(queriesL0[q]), fv4[q], zeroSib, flast[4], froot[4], enable);
            }
            if (fri == 1 && nq > 5) {
                VerifyFRIFold(STEPS[4], STEPS[5], nextFoldDomainBits(STEPS, nSteps, 5), foldCosetShiftInv(nBitsExt, STEPS, 5), STEPS[0])(queriesL0[q], rFold[4], fv4[q], fv5[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[4], STEPS[5], 0, foldCosetShiftInv(nBitsExt, STEPS, 5), STEPS[0])(queriesL0[q], rFold[4], fv4[q], finalPol, enable);
            }
        }
        if (nq > 5) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 6);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 6), arity, lastLevels, ml, STEPS[6])(QueryKeys(arity, ml + lastLevels, STEPS[6], STEPS[0])(queriesL0[q]), fv5[q], fs5[q], flast[5], froot[5], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 6), arity, lastLevels, 0, STEPS[6])(QueryKeys(arity, lastLevels, STEPS[6], STEPS[0])(queriesL0[q]), fv5[q], zeroSib, flast[5], froot[5], enable);
            }
            if (fri == 1 && nq > 6) {
                VerifyFRIFold(STEPS[5], STEPS[6], nextFoldDomainBits(STEPS, nSteps, 6), foldCosetShiftInv(nBitsExt, STEPS, 6), STEPS[0])(queriesL0[q], rFold[5], fv5[q], fv6[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[5], STEPS[6], 0, foldCosetShiftInv(nBitsExt, STEPS, 6), STEPS[0])(queriesL0[q], rFold[5], fv5[q], finalPol, enable);
            }
        }
        if (nq > 6) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 7);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 7), arity, lastLevels, ml, STEPS[7])(QueryKeys(arity, ml + lastLevels, STEPS[7], STEPS[0])(queriesL0[q]), fv6[q], fs6[q], flast[6], froot[6], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 7), arity, lastLevels, 0, STEPS[7])(QueryKeys(arity, lastLevels, STEPS[7], STEPS[0])(queriesL0[q]), fv6[q], zeroSib, flast[6], froot[6], enable);
            }
            if (fri == 1 && nq > 7) {
                VerifyFRIFold(STEPS[6], STEPS[7], nextFoldDomainBits(STEPS, nSteps, 7), foldCosetShiftInv(nBitsExt, STEPS, 7), STEPS[0])(queriesL0[q], rFold[6], fv6[q], fv7[q], enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[6], STEPS[7], 0, foldCosetShiftInv(nBitsExt, STEPS, 7), STEPS[0])(queriesL0[q], rFold[6], fv6[q], finalPol, enable);
            }
        }
        if (nq > 7) {
            var ml = foldPathLength(arity, lastLevels, STEPS, 8);
            if (ml > 0) {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 8), arity, lastLevels, ml, STEPS[8])(QueryKeys(arity, ml + lastLevels, STEPS[8], STEPS[0])(queriesL0[q]), fv7[q], fs7[q], flast[7], froot[7], enable);
            } else {
                VerifyTreeOpening(3, foldCosetSize(STEPS, 8), arity, lastLevels, 0, STEPS[8])(QueryKeys(arity, lastLevels, STEPS[8], STEPS[0])(queriesL0[q]), fv7[q], zeroSib, flast[7], froot[7], enable);
            }
            if (fri == 1 && nq > 8) {
                VerifyFRIFold(STEPS[7], STEPS[8], nextFoldDomainBits(STEPS, nSteps, 8), foldCosetShiftInv(nBitsExt, STEPS, 8), STEPS[0])(queriesL0[q], rFold[7], fv7[q], finalPol, enable);
            } else if (fri == 1) {
                VerifyFRIFold(STEPS[7], STEPS[8], 0, foldCosetShiftInv(nBitsExt, STEPS, 8), STEPS[0])(queriesL0[q], rFold[7], fv7[q], finalPol, enable);
            }
        }
    }
}
