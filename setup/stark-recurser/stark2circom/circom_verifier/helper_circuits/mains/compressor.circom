pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/circuit_type.circom";
include "vadcop/values.circom";

// The compressor of a basic air: its proof verified, its vadcop values minted.

// The compressor of a basic air: its proof verified, its vadcop values minted.
template Compressor() {
    var airgroupId = vadcop_airgroupId();
    var circuitType = circuitTypeOf(airgroupId, vadcop_airId() + 1);
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();
    var nPub = vadcop_nPublics();
    var nPV = vadcop_nProofValues();

    signal input publics[nPub];
    signal input proofValues[nPV][3];
    signal input globalChallenge[3];
    signal input proof[key_proofSize(0)];
    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    for (var i = 0; i < key_nPublics(0); i++) { sV.publics[i] <== publics[i]; }
    for (var i = 0; i < key_nProofValues(0); i++) { sV.proofvalues[i] <== proofValues[i]; }
    sV.globalChallenge <== globalChallenge;

    component v = BasicValues(0, circuitType, nAgg, 0);
    v.rootC <== sV.rootC;
    v.proof <== proof;
    sv_circuitType <== v.circuitTypeOut;
    sv_aggregatedProofs <== v.aggregatedProofs;
    sv_aggregationTypes <== v.aggregationTypes;
    sv_airgroupvalues <== v.airgroupvalues;
    sv_stage1Hash <== v.stage1Hash;
}
