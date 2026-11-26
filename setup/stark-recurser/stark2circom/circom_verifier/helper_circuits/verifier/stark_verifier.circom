pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "proof_shape.circom";
include "proof_rows.circom";
include "transcript.circom";
include "deep.circom";
include "fri.circom";
include "stir.circom";

/*
    The GL STARK verifier (FRI or STIR) of the compilation's key k, `StarkVerifierGL(k)` (below): its data
    is the main's `key_*(k)` (lib_params.rs), its constraints the key's `ConstraintRoots<sfx>` through
    `ConstraintRootsAt(k)`. It wires:

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
    var circuitType = key_circuitType(k);
    var nRoots = key_nRoots(k);
    var nSteps = key_nSteps(k);
    var stir = key_stir(k);
    var stirM = key_stirM(k);
    var nTrees = key_nTrees(k);
    var verkeyInput = key_verkeyInput(k);
    var enableInput = key_enableInput(k);
    var offNonce = key_offNonce(k);
    var offBetas = key_offBetas(k);
    var offNonces = key_offNonces(k);
    var NF = key_nFolds(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var STIR_NUM_QUERIES[nonEmpty(stirM)] = key_STIR_NUM_QUERIES(k);
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
    var nq = nQueryFoldTrees(nSteps);

    signal input proof[key_proofSize(k)];
    signal input publics[nPublics];
    signal input proofvalues[nProofValues][3];
    signal input globalChallenge[3 * inCh];
    signal input rootCIn[4 * verkeyInput];
    signal input enable[enableInput];
    signal output rootC[4];

    if (verkeyInput == 1) { rootC <== rootCIn; } else { rootC <== key_ROOT_C(k); }
    signal {binary} enabled;
    if (enableInput == 1) {
        enable[0] * (enable[0] - 1) === 0;
        enabled <== enable[0];
    } else {
        enabled <== 1;
    }

    // The proof's small parts.
    var P = key_proofSize(k);
    signal evals[nEvals][3] <== ProofRows(P, key_offEvals(k), nEvals, 3)(proof);
    signal airvalues[nAirValues][3] <== ProofRows(P, key_offAirValues(k), nAirValues, 3)(proof);
    signal airgroupvalues[nAirgroupValues][3] <== ProofRows(P, key_offAirgroupValues(k), nAirgroupValues, 3)(proof);
    signal finalPol[F][3] <== ProofRows(P, key_offFinal(k), F, 3)(proof);
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

    // Per L_0 query: every L_0 tree's row, flat at L0_TREE_START, and path; every fold tree's opened coset
    // and path, flat at foldCosetStart / foldPathStart. The proof holds each tree's data query after query.
    var nFoldCosetRows = foldCosetStart(STEPS, nq);
    var nFoldPathLevels = foldPathStart(arity, lastLevels, STEPS, nq);
    signal l0Values[nQueries][L0_TREE_START[nTrees]];
    signal l0Siblings[nQueries][nTrees][L0][wordsPerLevel];
    signal foldCosets[nQueries][nFoldCosetRows][3];
    signal foldSiblings[nQueries][nFoldPathLevels][wordsPerLevel];
    for (var g = 0; g < nQueries; g++) {
        for (var t = 0; t < nTrees; t++) {
            var width = L0_TREE_START[t + 1] - L0_TREE_START[t];
            for (var i = 0; i < width; i++) { l0Values[g][L0_TREE_START[t] + i] <== proof[OFF_L0_VALUES[t] + g * width + i]; }
            for (var l = 0; l < L0; l++) { for (var i = 0; i < wordsPerLevel; i++) { l0Siblings[g][t][l][i] <== proof[OFF_L0_SIBLINGS[t] + (g * L0 + l) * wordsPerLevel + i]; } }
        }
        for (var f = 0; f < nq; f++) {
            var cosetSize = foldCosetSize(STEPS, f + 1);
            var pathLevels = foldPathLength(arity, lastLevels, STEPS, f + 1);
            var cosetStart = foldCosetStart(STEPS, f);
            var pathStart = foldPathStart(arity, lastLevels, STEPS, f);
            for (var c = 0; c < cosetSize; c++) { for (var e = 0; e < 3; e++) { foldCosets[g][cosetStart + c][e] <== proof[OFF_FOLD_VALUES[f] + (g * cosetSize + c) * 3 + e]; } }
            for (var l = 0; l < pathLevels; l++) { for (var i = 0; i < wordsPerLevel; i++) { foldSiblings[g][pathStart + l][i] <== proof[OFF_FOLD_SIBLINGS[f] + (g * pathLevels + l) * wordsPerLevel + i]; } }
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

    component transcript = StarkVerifierTranscript(nStages, nPublics, nAirValues, nAirgroupValues, nEvals, key_nChallenges(k), nSteps, STEPS, stir, stirM, key_stirFinalBits(k), STIR_NUM_QUERIES, key_hashCommits(k), circuitType, nQueries, key_powBits(k), key_CH(k), key_AIR_VALUE_STAGE(k), key_AIRGROUP_VALUE_STAGE(k));
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

    VerifyEvaluations(nBits, nRoots, key_firstRootDim(k), key_qDeg(k), key_qPolEv(k), nEvals)(transcript.xi, transcript.vc, roots, evals, enabled);

    component deepConsts = DeepConstants(key_nOpenings(k), key_nPols(k), nEvals, key_OPENING_NUM_EVALS(k), key_EVAL_COLUMN(k));
    deepConsts.vf <== transcript.vf;
    deepConsts.evals <== evals;

    // The L_0 queries in batches of batchSize, the rest in one tail batch: its own array, as circom
    // resolves an array mixing template instances through a map at run time. Per batch its per-proof
    // inputs, then its queries' data, so it starts as soon as it is wired.
    var nFull = nQueries \ batchSize;
    var nTail = nQueries - nFull * batchSize;
    component batch[nonEmpty(nFull)];
    component tail[1];
    for (var b = 0; b < nFull; b++) {
        batch[b] = QueryBatchGL(k, batchSize);
        batch[b].xi <== transcript.xi;
        batch[b].friConsts <== deepConsts.friConsts;
        batch[b].treeLast <== treeLast;
        batch[b].treeRoot <== treeRoot;
        for (var f = 0; f < nq; f++) {
            batch[b].foldLast[f] <== foldLast[f];
            batch[b].foldRoot[f] <== foldRoot[f];
            if (stir == 0) { batch[b].rFold[f] <== transcript.ldt[ldt_rFold(stir, f)]; }
        }
        if (stir == 0) { batch[b].finalPol <== finalPol; }
        batch[b].enable <== enabled;
        for (var q = 0; q < batchSize; q++) {
            var g = b * batchSize + q;
            batch[b].queriesL0[q] <== transcript.queriesL0[g];
            batch[b].l0Values[q] <== l0Values[g];
            batch[b].l0Siblings[q] <== l0Siblings[g];
            if (nFoldCosetRows > 0) { batch[b].foldCosets[q] <== foldCosets[g]; }
            if (nFoldPathLevels > 0) { batch[b].foldSiblings[q] <== foldSiblings[g]; }
        }
    }
    if (nTail > 0) {
        tail[0] = QueryBatchGL(k, nTail);
        tail[0].xi <== transcript.xi;
        tail[0].friConsts <== deepConsts.friConsts;
        tail[0].treeLast <== treeLast;
        tail[0].treeRoot <== treeRoot;
        for (var f = 0; f < nq; f++) {
            tail[0].foldLast[f] <== foldLast[f];
            tail[0].foldRoot[f] <== foldRoot[f];
            if (stir == 0) { tail[0].rFold[f] <== transcript.ldt[ldt_rFold(stir, f)]; }
        }
        if (stir == 0) { tail[0].finalPol <== finalPol; }
        tail[0].enable <== enabled;
        for (var q = 0; q < nTail; q++) {
            var g = nFull * batchSize + q;
            tail[0].queriesL0[q] <== transcript.queriesL0[g];
            tail[0].l0Values[q] <== l0Values[g];
            tail[0].l0Siblings[q] <== l0Siblings[g];
            if (nFoldCosetRows > 0) { tail[0].foldCosets[q] <== foldCosets[g]; }
            if (nFoldPathLevels > 0) { tail[0].foldSiblings[q] <== foldSiblings[g]; }
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
    One batch of n L_0 queries, the unit of parallel witness computation. Per query: every L_0 tree's
    row against its tree (QueryL0TreeOpening), DEEP, and every fold step (QueryFoldStep; STIR: T_0's
    opening only). A query's data comes flat: its L_0 rows at L0_TREE_START, its fold-tree cosets and
    paths at foldCosetStart / foldPathStart. The per-tree and per-step arrays mix template instances, which
    circom resolves through a map at run time: each takes the query's data whole, one store per input.
*/
template parallel QueryBatchGL(k, n) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
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
    var nq = nQueryFoldTrees(nSteps);

    signal input {binary} queriesL0[n][STEPS[0]];
    signal input xi[3];
    signal input friConsts[2 + nPols + 2 * nOpenings][3];
    signal input treeLast[nTrees][lastSize][4];
    signal input treeRoot[nTrees][4];
    signal input foldLast[nq][lastSize][4];
    signal input foldRoot[nq][4];
    signal input rFold[nq * fri][3];
    signal input finalPol[fri * lastDomainSize(nSteps, STEPS)][3];
    signal input {binary} enable;

    signal input l0Values[n][L0_TREE_START[nTrees]];
    signal input l0Siblings[n][nTrees][L0][wordsPerLevel];
    signal input foldCosets[n][foldCosetStart(STEPS, nq)][3];
    signal input foldSiblings[n][foldPathStart(arity, lastLevels, STEPS, nq)][wordsPerLevel];

    component l0Keys[n];
    component l0Opening[n][nTrees];
    component deep[n];
    component foldStep[n][nonEmpty(nq)];
    for (var q = 0; q < n; q++) {
        // One key split per query, shared by the L_0 trees.
        l0Keys[q] = QueryKeys(arity, L0 + lastLevels, STEPS[0], STEPS[0]);
        l0Keys[q].queries <== queriesL0[q];
        for (var t = 0; t < nTrees; t++) {
            l0Opening[q][t] = QueryL0TreeOpening(k, t);
            l0Opening[q][t].keys <== l0Keys[q].keys;
            l0Opening[q][t].l0Values <== l0Values[q];
            l0Opening[q][t].siblings <== l0Siblings[q][t];
            l0Opening[q][t].last <== treeLast[t];
            l0Opening[q][t].root <== treeRoot[t];
            l0Opening[q][t].enable <== enable;
        }

        deep[q] = DeepQuery(nOpenings, nPols, key_nColGroups(k), key_nSums(k), STEPS[0], nextFoldDomainBits(STEPS, nSteps, 0), L0_TREE_START[nTrees], key_OPENING_ROOT(k), key_SUMS(k), key_COLS(k));
        deep[q].queriesL0 <== queriesL0[q];
        deep[q].xi <== xi;
        deep[q].friConsts <== friConsts;
        deep[q].s0_vals <== l0Values[q];
        // f_0 lands on fold tree 0's coset, or on finalPol when nothing folds.
        if (nq > 0) {
            for (var c = 0; c < foldCosetSize(STEPS, 1); c++) { deep[q].s1_vals[c] <== foldCosets[q][c]; }
        } else {
            deep[q].s1_vals <== finalPol;
        }
        deep[q].enable <== enable;

        for (var f = 0; f < nq; f++) {
            foldStep[q][f] = QueryFoldStep(k, f + 1);
            foldStep[q][f].queriesL0 <== queriesL0[q];
            foldStep[q][f].foldCosets <== foldCosets[q];
            if (foldPathStart(arity, lastLevels, STEPS, nq) > 0) { foldStep[q][f].foldSiblings <== foldSiblings[q]; }
            foldStep[q][f].last <== foldLast[f];
            foldStep[q][f].root <== foldRoot[f];
            if (fri == 1) { foldStep[q][f].rFold[0] <== rFold[f]; }
            if (fri == 1 && f == nq - 1) { foldStep[q][f].finalPol <== finalPol; }
            foldStep[q][f].enable <== enable;
        }
    }
}

// L_0 tree t's row at one query against the tree: the row is t's columns of the query's L_0 values.
template QueryL0TreeOpening(k, t) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nSteps = key_nSteps(k);
    var nTrees = key_nTrees(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var L0 = l0PathLength(arity, lastLevels, STEPS);
    var width = L0_TREE_START[t + 1] - L0_TREE_START[t];

    signal input {binary} keys[L0 + lastLevels][log2Arity(arity)];
    signal input l0Values[L0_TREE_START[nTrees]];
    signal input siblings[L0][siblingWords(arity)];
    signal input last[clearLevelNodes(arity, lastLevels)][4];
    signal input root[4];
    signal input {binary} enable;

    signal row[width][1];
    for (var i = 0; i < width; i++) { row[i][0] <== l0Values[L0_TREE_START[t] + i]; }
    VerifyTreeOpening(1, width, arity, lastLevels, L0, key_nBitsExt(k))(keys, row, siblings, last, root, enable);
}

/*
    Fold step s >= 1 at one query: fold tree s - 1's coset (f_{s-1} around the query) against the tree
    and, FRI only, that coset folded with rFold equals its member in the next fold tree's coset, or in
    finalPol after the last step. Reads its cosets out of the query's flat fold data.
*/
template QueryFoldStep(k, s) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nSteps = key_nSteps(k);
    var STEPS[nSteps] = key_STEPS(k);
    var nq = nQueryFoldTrees(nSteps);
    var wordsPerLevel = siblingWords(arity);
    var fri = 1 - key_stir(k);
    var cosetSize = foldCosetSize(STEPS, s);
    var pathLevels = foldPathLength(arity, lastLevels, STEPS, s);
    var nextBits = nextFoldDomainBits(STEPS, nSteps, s);
    var foldsIntoFinalPol = fri == 1 && s == nq ? 1 : 0;

    signal input {binary} queriesL0[STEPS[0]];
    signal input foldCosets[foldCosetStart(STEPS, nq)][3];
    signal input foldSiblings[foldPathStart(arity, lastLevels, STEPS, nq)][wordsPerLevel];
    signal input last[clearLevelNodes(arity, lastLevels)][4];
    signal input root[4];
    signal input rFold[fri][3];
    signal input finalPol[foldsIntoFinalPol * lastDomainSize(nSteps, STEPS)][3];
    signal input {binary} enable;

    signal coset[cosetSize][3];
    for (var c = 0; c < cosetSize; c++) { coset[c] <== foldCosets[foldCosetStart(STEPS, s - 1) + c]; }
    signal {binary} keys[pathLevels + lastLevels][log2Arity(arity)] <== QueryKeys(arity, pathLevels + lastLevels, STEPS[s], STEPS[0])(queriesL0);
    if (pathLevels > 0) {
        signal siblings[pathLevels][wordsPerLevel];
        for (var l = 0; l < pathLevels; l++) { siblings[l] <== foldSiblings[foldPathStart(arity, lastLevels, STEPS, s - 1) + l]; }
        VerifyTreeOpening(3, cosetSize, arity, lastLevels, pathLevels, STEPS[s])(keys, coset, siblings, last, root, enable);
    } else {
        // The whole path is sent in clear: VerifyTreeOpening's Empty variant ignores its one siblings row.
        var noSiblings[1][wordsPerLevel];
        for (var i = 0; i < wordsPerLevel; i++) { noSiblings[0][i] = 0; }
        VerifyTreeOpening(3, cosetSize, arity, lastLevels, 0, STEPS[s])(keys, coset, noSiblings, last, root, enable);
    }

    if (fri == 1) {
        signal nextCoset[1 << (STEPS[s] - nextBits)][3];
        if (foldsIntoFinalPol == 1) {
            nextCoset <== finalPol;
        } else {
            for (var c = 0; c < 1 << (STEPS[s] - nextBits); c++) { nextCoset[c] <== foldCosets[foldCosetStart(STEPS, s) + c]; }
        }
        VerifyFRIFold(STEPS[s - 1], STEPS[s], nextBits, foldCosetShiftInv(key_nBitsExt(k), STEPS, s), STEPS[0])(queriesL0, rFold[0], coset, nextCoset, enable);
    }
}
