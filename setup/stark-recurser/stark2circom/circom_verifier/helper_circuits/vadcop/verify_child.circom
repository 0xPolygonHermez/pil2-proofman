pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "vadcop/select_vk.circom";
include "vadcop/values.circom";

/*
    A child proof of key k that carries vadcop values (a compressor, recursive1 or recursive2 proof of
    airgroup `airgroupId`), verified against the publics VadcopPublics builds from them; rootCAt as there.
    When nullable, a circuitType of 0 marks an empty slot: isNull, and the verification is disabled. A
    verifier that takes its key as input (key_verkeyInput) gets the one the circuitType selects.
*/
template VerifyChild(k, airgroupId, rootCAt, nullable) {
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();
    var nPub = vadcop_nPublics();
    var nPV = vadcop_nProofValues();

    signal input vadcopValues[vadcopValuesSize(nAgg, W)];
    signal input proof[key_proofSize(k)];
    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal output {binary} isNull[nullable];

    component verifier = StarkVerifierGL(k);
    verifier.proof <== proof;
    for (var i = 0; i < key_nProofValues(k); i++) { verifier.proofvalues[i] <== proofValues[i]; }
    verifier.publics <== VadcopPublics(airgroupId, nAgg, W, nPub, nPV, key_nPublics(k), rootCAt)(vadcopValues, publics, proofValues, globalChallenge, rootCAgg);
    if (nullable == 1) {
        isNull[0] <== IsZero()(vadcopValues[0]);
        verifier.enable[0] <== 1 - isNull[0];
    }
    if (key_verkeyInput(k) == 1) {
        var nBasics = vadcop_nBasics(airgroupId);
        var rootCBasics[nBasics][4] = vadcop_basics(airgroupId);
        verifier.rootCIn <== SelectVerificationKey(nBasics, airgroupId, nullable)(vadcopValues[0], rootCBasics, rootCAgg);
    }
}
