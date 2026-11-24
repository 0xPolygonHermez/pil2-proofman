pragma circom 2.1.0;
pragma custom_templates;

include "fft.circom";
include "evalpol.circom";
include "geometry.circom";
include "deep.circom";
include "merkle.circom";

/*
    FRI checks of one query. `TreeSelector` (tree/treeselector<4|8>.circom) comes from the generated
    verifier's includes.

    The generated `QueryBatch<suffix>` instantiates these per tree and per fold step, one component
    array each: circom resolves the signals of an array mixing template instances through a map at
    run time, so trees and steps (different shapes) never share an array.
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

    signal sx[currStepBits];

    sx[0] <== e0 * (queriesL0[0] * (invroots(prevStepBits) - 1) + 1);
    for (var i = 1; i < currStepBits; i++) {
        sx[i] <== sx[i - 1] * (queriesL0[i] * (invroots(prevStepBits - i) - 1) + 1);
    }

    // Interpolate the opened coset and evaluate it at the challenge over the coset point.
    signal coefs[1 << step][3] <== FFT(step, 3, 1)(s_vals_curr);
    signal evalXprime[3] <== [friChallenge[0] * sx[currStepBits - 1], friChallenge[1] * sx[currStepBits - 1], friChallenge[2] * sx[currStepBits - 1]];
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
