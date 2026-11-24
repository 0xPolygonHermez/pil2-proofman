pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "geometry.circom";
include "transcript.circom";
include "deep.circom";
include "fri.circom";
include "stir.circom";

/*
    The GL STARK verifier (FRI or STIR). The generated `StarkVerifier<suffix>` declares the key's data as vars
    named after these templates' parameters and wires, around its own `ConstraintRoots<suffix>`:

        StarkVerifierTranscript   every challenge and the FRI queries (transcript.circom)
        VerifyEvaluations         the roots fold to Q(xi) (below)
        DeepConstants             the query-independent DEEP terms (deep.circom)
        QueryBatch<suffix>        generated: per query the Merkle openings (fri.circom), DEEP, FRI folds
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

    signal input challengeOut[3];
    signal input vc[3];
    signal input roots[nR][3];
    signal input evals[nEvals][3];
    signal input {binary} enable;

    // zMul[i] = xi^(2^(i+1)); Zh = 1 / (xi^N - 1)
    signal zMul[nBits][3];
    for (var i = 0; i < nBits; i++) {
        if (i == 0) {
            zMul[i] <== CMul()(challengeOut, challengeOut);
        } else {
            zMul[i] <== CMul()(zMul[i - 1], zMul[i - 1]);
        }
    }
    signal Z[3] <== [zMul[nBits - 1][0] - 1, zMul[nBits - 1][1], zMul[nBits - 1][2]];
    signal Zh[3] <== CInv()(Z);

    // A base-field first root stays in the base field until its first product with vc.
    signal foldMul[geo_len1(nR - 1)][3];
    signal foldAcc[geo_len1(nR - 1)][3];
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
    signal qStep[geo_len1(qDeg - 1)][3];
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
