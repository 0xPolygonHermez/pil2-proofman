pragma circom 2.1.0;
pragma custom_templates;

include "fft.circom";
include "evalpol.circom";
include "proof_shape.circom";
include "deep.circom";
include "merkle.circom";
include "query_point.circom";

/*
    FRI checks of one query (stark_verifier.circom's QueryFoldStep runs them per fold step).
    `TreeSelector` (tree/treeselector<4|8>.circom) comes from the generated verifier's includes.
*/

// One FRI folding step: the folded value at the query equals the next oracle's opened member.
template VerifyFRIFold(prevStepBits, currStepBits, nextStepBits, e0, nIn) {
    var nextStep = currStepBits - nextStepBits;
    var step = prevStepBits - currStepBits;

    signal input {binary} queriesL0[nIn];
    signal input friChallenge[3];
    signal input s_vals_curr[1 << step][3];
    signal input s_vals_next[1 << nextStep][3];
    signal input {binary} enable;

    // 1 / c, c the offset of the query's coset in the previous domain (its low currStepBits bits pick it).
    signal sxInv <== QueryPoint(currStepBits, prevStepBits, e0, 1, nIn)(queriesL0);

    // Interpolate the opened coset and evaluate it at the challenge over the coset point.
    signal coefs[1 << step][3] <== FFT(step, 3, 1)(s_vals_curr);
    signal evalXprime[3] <== [friChallenge[0] * sxInv, friChallenge[1] * sxInv, friChallenge[2] * sxInv];
    signal evalPol[3] <== EvalPol(1 << step)(coefs, evalXprime);

    signal {binary} keys_lowValues[nextStep];
    for (var i = 0; i < nextStep; i++) { keys_lowValues[i] <== queriesL0[i + nextStepBits]; }
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s_vals_next, keys_lowValues);

    enable * (lowValues[0] - evalPol[0]) === 0;
    enable * (lowValues[1] - evalPol[1]) === 0;
    enable * (lowValues[2] - evalPol[2]) === 0;
}

// finalPol is sent as evaluations: its coefficients above the degree bound must vanish.
template VerifyFinalPol(nLastBits, maxDegSize) {
    signal input finalPol[1 << nLastBits][3];
    signal input {binary} enable;

    signal lastIFFT[1 << nLastBits][3] <== FFT(nLastBits, 3, 1)(finalPol);

    for (var k = maxDegSize; k < 1 << nLastBits; k++) {
        for (var e = 0; e < 3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    for (var k = 0; k < maxDegSize; k++) {
        _ <== lastIFFT[k];
    }
}
