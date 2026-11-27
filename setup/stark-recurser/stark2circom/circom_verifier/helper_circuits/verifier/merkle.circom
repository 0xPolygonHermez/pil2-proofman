pragma circom 2.1.0;
pragma custom_templates;

include "proof_shape.circom";

/*
    Merkle openings of the Goldilocks-word trees (every GL family; blake3's in the BN128 verifier too).
    `VerifyMerkleHash*` and `VerifyMerkleRoot` (hash/<family>/merklehash.circom) come from the
    generated verifier's includes; every family exports the same names.
*/

// The query's bits grouped per Merkle level (log2(arity) each), zero-padded past the low `bits`.
template QueryKeys(arity, nLevels, bits, nIn) {
    var nBA = log2Arity(arity);

    signal input {binary} queries[nIn];
    signal output {binary} keys[nLevels][nBA];

    for (var j = 0; j < nLevels; j++) {
        for (var k = 0; k < nBA; k++) {
            if (k + j * nBA >= bits) {
                keys[j][k] <== 0;
            } else {
                keys[j][k] <== queries[j * nBA + k];
            }
        }
    }
}

// One opened leaf (nVals elements of eSize limbs) of a tree over 2^leafBits leaves, keyed by QueryKeys:
// Merkle path below the levels sent in clear. L_0 rows, FRI cosets, STIR's T_i.
template VerifyTreeOpening(eSize, nVals, arity, lastLevels, ml, leafBits) {
    var LL = lastLevels;

    signal input {binary} keys[ml + LL][log2Arity(arity)];
    signal input values[nVals][eSize];
    signal input siblings[nonEmpty(ml)][siblingWords(arity)];
    signal input last[clearLevelNodes(arity, LL)][4];
    signal input root[4];
    signal input {binary} enable;

    if (LL > 0 && ml == 0) {
        component mtE = VerifyMerkleHashUntilLevelEmpty(eSize, nVals, arity, LL, 1 << leafBits);
        mtE.values <== values;
        mtE.key <== keys;
        mtE.last_levels <== last;
        mtE.enable <== enable;
    } else if (LL > 0) {
        component mtU = VerifyMerkleHashUntilLevel(eSize, nVals, arity, ml, LL, 1 << leafBits);
        mtU.values <== values;
        mtU.siblings <== siblings;
        mtU.key <== keys;
        mtU.last_levels <== last;
        mtU.enable <== enable;
    } else {
        component mtR = VerifyMerkleHash(eSize, nVals, arity, ml + LL);
        mtR.values <== values;
        mtR.siblings <== siblings;
        mtR.key <== keys;
        mtR.root <== root;
        mtR.enable <== enable;
    }
}
