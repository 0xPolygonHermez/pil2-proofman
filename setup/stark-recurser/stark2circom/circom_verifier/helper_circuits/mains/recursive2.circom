pragma circom 2.1.0;
pragma custom_templates;

include "vadcop/circuit_type.circom";
include "vadcop/fold.circom";
include "vadcop/values.circom";
include "vadcop/verify_child.circom";

// recursive2: verifies n proofs of one airgroup (recursive1 or recursive2) and folds their vadcop values.

/*
    A recursive2 over n proofs of one airgroup, each slot a child's vadcop values then its proof. Its
    circuitType selects the child's key (recursive1 or recursive2); with several airs, 0 marks the slot
    empty.
*/
template Recursive2(n) {
    var airgroupId = vadcop_airgroupId();
    var nAgg = vadcop_nAgg(airgroupId);
    var W = vadcop_W();
    var S = vadcopValuesSize(nAgg, W);
    var multiAir = vadcop_multiAir();

    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[nAgg];
    signal output sv_airgroupvalues[nAgg][3];
    signal output sv_stage1Hash[W];
    signal input publics[vadcop_nPublics()];
    signal input proofValues[vadcop_nProofValues()][3];
    signal input globalChallenge[3];
    signal input rootCAgg[4];
    signal input slots[n][S + key_proofSize(0)];

    component fold = FoldSlots(n, nAgg, W, multiAir);
    component child[n];
    for (var s = 0; s < n; s++) {
        // rootCAgg is the child's last four publics.
        child[s] = parallel VerifyChild(0, airgroupId, key_nPublics(0) - 4, multiAir);
        for (var i = 0; i < S; i++) { child[s].vadcopValues[i] <== slots[s][i]; }
        for (var i = 0; i < key_proofSize(0); i++) { child[s].proof[i] <== slots[s][S + i]; }
        child[s].publics <== publics;
        child[s].proofValues <== proofValues;
        child[s].globalChallenge <== globalChallenge;
        child[s].rootCAgg <== rootCAgg;
        for (var i = 0; i < S; i++) { fold.slots[s][i] <== slots[s][i]; }
        if (multiAir == 1) { fold.isNullIn[s] <== child[s].isNull[0]; }
    }

    sv_circuitType <== circuitTypeOf(airgroupId, 0);
    sv_aggregatedProofs <== fold.aggregatedProofs;
    sv_aggregationTypes <== fold.aggregationTypes;
    sv_airgroupvalues <== fold.airgroupvalues;
    sv_stage1Hash <== fold.stage1Hash;
}
