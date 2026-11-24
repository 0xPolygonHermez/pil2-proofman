pragma circom 2.1.0;
pragma custom_templates;

include "bitify.circom";
include "geometry.circom";

/*
    The STARK's Fiat-Shamir schedule on top of `TranscriptEngine` (transcript_sponge.circom or
    transcript_blake3.circom, whichever the generated verifier includes) and `VerifyPoW`
    (hash/<family>/pow.circom). The engine's family sets the word widths: tr_digestW() per root or
    hash, tr_challengeW() per challenge (`TranscriptChallenge` makes it 3 limbs), tr_wordBits() per
    query word; `TranscriptWordBits` (the field's bitify) splits a word.
*/

// nQueries indices of nBits bits from a fresh transcript over challenge (and nonce), proof of work first.
template QueryBits(nQueries, nBits, powBits, withNonce) {
    signal input challenge[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queries[nQueries][nBits];

    component pow[1];
    if (powBits > 0) {
        pow[0] = VerifyPoW(powBits);
        pow[0].challengeFRIQueries <== challenge;
        pow[0].nonce <== nonce;
        pow[0].enable <== enable;
    }

    var nIn = withNonce == 1 ? 4 : 3;
    var wb = tr_wordBits();
    var totalBits = nQueries * nBits;
    var nFields = (totalBits - 1) \ wb + 1;
    var ops[4] = [0, nIn, 1, nFields];
    component tr = TranscriptEngine(nIn, nFields, 2, ops);
    for (var i = 0; i < 3; i++) { tr.in[i] <== challenge[i]; }
    if (withNonce == 1) { tr.in[3] <== nonce; }

    // wb bits from each word but the last, the remainder from that one.
    signal {binary} n2b[nFields][wb + 1];
    var q = 0;
    var b = 0;
    for (var f = 0; f < nFields; f++) {
        n2b[f] <== TranscriptWordBits(wb + 1)(tr.out[f]);
        var nb = f + 1 == nFields ? totalBits - wb * f : wb;
        for (var j = 0; j < nb; j++) {
            queries[q][b] <== n2b[f][j];
            b++;
            if (b == nBits) {
                b = 0;
                q++;
            }
        }
    }
}

// Absorbs `n` words and returns the digest read from the fresh output.
template TranscriptHash(n) {
    signal input in[n];
    signal output out[tr_digestW()];

    var ops[4] = [0, n, 1, tr_digestW()];
    out <== TranscriptEngine(n, tr_digestW(), 2, ops)(in);
}

/*
    The STARK transcript: every challenge the verifier draws, in the prover's order.

        [rootC, publics (or their hash), root1 | globalChallenge]
        per stage s = 2..nStages: its challenges, [circuitType at s = 2 when inputChallenges],
                                  root_s, its air values, its airgroup values
        vc, root_Q, xi, evals (or their hash), vf1 vf2,
        FRI:  per fold: rFold[s-1] after the root of f_{s-1}; finalPol (or its hash), the query challenge
        STIR: root T_0, rFold[0]; per iteration i >= 1: root T_i, rOut, beta, rFold[i], rComb, the query
              challenge of round i - 1, its Ans and shake hints, rho; finalPol (or its hash), the last
              round's query challenge

    challengesMap is in draw order (lib_params.rs checks it), so the squeezes fill `challenges` in
    order, then `ldt` (geometry.circom's ldt_* name its entries). CH[s] counts stage s's challenges,
    AV / AGV are the air(group) values' stages; circuitType > 0 starts from the global challenge (a
    basic air). queriesL0 are round 0's queries (FRI's only round).
*/
template StarkVerifierTranscript(nStages, nPublics, nAirValues, nAirgroupValues, nEvals, nChallenges, nSteps, STEPS, stir, stirM, stirFinalBits, STIR_T, hashCommits, circuitType, nQueries, powBits, CH, AV, AGV) {
    var nP = nPublics;
    var nAV = nAirValues;
    var nAGV = nAirgroupValues;
    var nEv = nEvals;
    var NF = geo_nFolds(stir, nSteps, stirM);
    var F = geo_finalPolSize(stir, nSteps, STEPS, stirFinalBits);
    var nB = stir == 1 ? NF - 1 : 0;
    var nH = stir_nHints(stir, NF, STIR_T);
    var hc = hashCommits;
    var inCh = circuitType > 0 ? 1 : 0;
    var D = tr_digestW();
    var CW = tr_challengeW();

    signal input globalChallenge[3];
    signal input publics[geo_len1(nP)];
    signal input rootC[D];
    signal input stageRoots[nStages + 1][D];
    signal input airvalues[geo_len1(nAV)][3];
    signal input airgroupvalues[geo_len1(nAGV)][3];
    signal input evals[nEv][3];
    signal input foldRoots[geo_len1(NF)][D];
    signal input betas[geo_len1(nB)][3];
    signal input hints[geo_len1(nH)][3];
    signal input finalPol[F][3];
    signal input nonce;
    signal input {binary} enable;

    signal output challenges[nChallenges][3];
    signal output ldt[ldt_n(stir, NF)][3];
    signal output {binary} queriesL0[nQueries][STEPS[0]];

    // Stage-1 airgroup values use limb 0 only.
    for (var a = 0; a < nAGV; a++) {
        if (AGV[a] == 1) {
            airgroupvalues[a][1] === 0;
            airgroupvalues[a][2] === 0;
        }
    }

    // hashCommits: publics, evals and finalPol enter as the state of their own transcript.
    component publicsHash[1];
    component evalsHash[1];
    component finalPolHash[1];
    if (hc == 1 && inCh == 0 && nP > 0) {
        publicsHash[0] = TranscriptHash(nP);
        for (var i = 0; i < nP; i++) { publicsHash[0].in[i] <== publics[i]; }
    }
    if (hc == 1) {
        evalsHash[0] = TranscriptHash(3 * nEv);
        for (var i = 0; i < nEv; i++) {
            for (var e = 0; e < 3; e++) { evalsHash[0].in[3 * i + e] <== evals[i][e]; }
        }
        finalPolHash[0] = TranscriptHash(3 * F);
        for (var i = 0; i < F; i++) {
            for (var e = 0; e < 3; e++) { finalPolHash[0].in[3 * i + e] <== finalPol[i][e]; }
        }
    }

    // The schedule: absorbed words in `src`, engine ops in `ops`; nSq squeezed triples.
    var MAXIN = 3 * D + nP + D * nStages + 1 + 3 * nAV + 3 * nAGV + D + 3 * nEv + D * NF + 3 * nB + 3 * nH + 3 * F;
    var MAXOPS = 8 + 5 * nStages + 3 * nChallenges + nAV + nAGV + 8 * NF + 8;
    var src[MAXIN];
    var ops[2 * MAXOPS];
    var nIn = 0;
    var nOps = 0;
    var nSq = 0;

    if (inCh == 0) {
        for (var i = 0; i < D; i++) { src[nIn] = rootC[i]; nIn++; }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
        if (nP > 0 && hc == 1) {
            for (var i = 0; i < D; i++) { src[nIn] = publicsHash[0].out[i]; nIn++; }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
        }
        if (nP > 0 && hc == 0) {
            for (var i = 0; i < nP; i++) { src[nIn] = publics[i]; nIn++; }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = nP; nOps++;
        }
        for (var i = 0; i < D; i++) { src[nIn] = stageRoots[0][i]; nIn++; }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
    } else {
        for (var i = 0; i < 3; i++) { src[nIn] = globalChallenge[i]; nIn++; }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3; nOps++;
    }

    for (var s = 2; s <= nStages; s++) {
        for (var j = 0; j < CH[s]; j++) {
            ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
        }
        // A basic air's identity, absorbed once the shared stage-2 challenges are out.
        // VERIFY-CONSTRAINTS: a constant engine input (today a literal hash input) must fold away under --O1.
        if (inCh == 1 && s == 2) {
            src[nIn] = circuitType; nIn++;
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = 1; nOps++;
        }
        for (var i = 0; i < D; i++) { src[nIn] = stageRoots[s - 1][i]; nIn++; }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
        for (var a = 0; a < nAV; a++) {
            if (AV[a] == s) {
                for (var e = 0; e < 3; e++) { src[nIn] = airvalues[a][e]; nIn++; }
                ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3; nOps++;
            }
        }
        // Stage-1 airgroup values (limb 0 only) are absorbed in round 2, wider ones in their own round.
        for (var a = 0; a < nAGV; a++) {
            var st = AGV[a];
            var eff = st > 2 ? st : 2;
            if (eff == s) {
                var w = st == 1 ? 1 : 3;
                for (var e = 0; e < w; e++) { src[nIn] = airgroupvalues[a][e]; nIn++; }
                ops[2 * nOps] = 0; ops[2 * nOps + 1] = w; nOps++;
            }
        }
    }

    // vc, root_Q, xi
    ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
    for (var i = 0; i < D; i++) { src[nIn] = stageRoots[nStages][i]; nIn++; }
    ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
    ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;

    if (hc == 1) {
        for (var i = 0; i < D; i++) { src[nIn] = evalsHash[0].out[i]; nIn++; }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
    } else {
        for (var i = 0; i < nEv; i++) {
            for (var e = 0; e < 3; e++) { src[nIn] = evals[i][e]; nIn++; }
        }
        ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3 * nEv; nOps++;
    }
    // vf1, vf2
    ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
    ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;

    // foldRoots[s] commits f_s; rFold[s] is drawn right after it (STIR: after rOut and beta for s >= 1).
    for (var s = 0; s <= NF; s++) {
        if (s > 0) {
            ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
        }
        // STIR iteration s - 1 >= 1: rComb, the query challenge of round s - 2, its hints, rho.
        if (stir == 1 && s >= 2) {
            ops[2 * nOps] = 1; ops[2 * nOps + 1] = 2 * CW; nOps++; nSq += 2;
            var h0 = stir_hintOff(STIR_T, s - 2);
            var hn = 2 * (STIR_T[s - 2] + 1);
            for (var i = 0; i < hn; i++) {
                for (var e = 0; e < 3; e++) { src[nIn] = hints[h0 + i][e]; nIn++; }
            }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3 * hn; nOps++;
            ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
        }
        if (s < NF) {
            for (var i = 0; i < D; i++) { src[nIn] = foldRoots[s][i]; nIn++; }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
            if (stir == 1 && s >= 1) {
                ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
                for (var e = 0; e < 3; e++) { src[nIn] = betas[s - 1][e]; nIn++; }
                ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3; nOps++;
            }
        } else if (hc == 1) {
            for (var i = 0; i < D; i++) { src[nIn] = finalPolHash[0].out[i]; nIn++; }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = D; nOps++;
        } else {
            for (var i = 0; i < F; i++) {
                for (var e = 0; e < 3; e++) { src[nIn] = finalPol[i][e]; nIn++; }
            }
            ops[2 * nOps] = 0; ops[2 * nOps + 1] = 3 * F; nOps++;
        }
    }
    ops[2 * nOps] = 1; ops[2 * nOps + 1] = CW; nOps++; nSq++;
    assert(nSq == nChallenges + ldt_n(stir, NF));

    component tr = TranscriptEngine(nIn, CW * nSq, nOps, ops);
    for (var i = 0; i < nIn; i++) { tr.in[i] <== src[i]; }

    component ch[nSq];
    for (var k = 0; k < nSq; k++) {
        ch[k] = TranscriptChallenge();
        for (var i = 0; i < CW; i++) { ch[k].in[i] <== tr.out[CW * k + i]; }
        if (k < nChallenges) {
            challenges[k] <== ch[k].out;
        } else {
            ldt[k - nChallenges] <== ch[k].out;
        }
    }

    queriesL0 <== QueryBits(nQueries, STEPS[0], powBits, stir == 1 || powBits > 0 ? 1 : 0)(ldt[ldt_query(stir, NF, 0)], nonce, enable);
}
