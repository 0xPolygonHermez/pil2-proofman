pragma circom 2.1.0;

include "gl_family.circom";
include "evalpol.circom";
include "treeselector.circom";
include "merklehash.circom";
include "pow.circom";
include "sponge.circom";
include "transcript_sponge.circom";
include "proof_shape.circom";
include "stark_verifier_bn128.circom";

/*
    The BN128 verifier of a BN128 Poseidon tree (circuits.bn128/merklehash.circom, one-element roots),
    the one include of its generated verifier: see bn128_open_blake3.circom.
*/

function bn_rootW() { return 1; }
function bn_sibW(arity) { return arity; }

template BN128Open(eSize, nVals, arity, lastLevels, ml, leafBits, nBits0) {
    signal input {binary} queries[nBits0];
    signal input values[nVals][eSize];
    signal input siblings[nonEmpty(ml)][arity];
    signal input last[clearLevelNodes(arity, lastLevels)][1];
    signal input root[1];
    signal input {binary} enable;

    signal {binary} keys[leafBits];
    for (var i = 0; i < leafBits; i++) { keys[i] <== queries[i]; }
    signal lastLevelsIn[clearLevelNodes(arity, lastLevels)];
    for (var i = 0; i < clearLevelNodes(arity, lastLevels); i++) { lastLevelsIn[i] <== last[i][0]; }
    if (lastLevels > 0 && ml == 0) {
        VerifyMerkleHashUntilLevelEmpty(eSize, nVals, 1 << leafBits, arity, lastLevels)(values, keys, lastLevelsIn, enable);
        _ <== siblings;
        _ <== root;
    } else if (lastLevels > 0) {
        VerifyMerkleHashUntilLevel(eSize, nVals, 1 << leafBits, arity, lastLevels)(values, siblings, keys, lastLevelsIn, enable);
        _ <== root;
    } else {
        VerifyMerkleHash(eSize, nVals, 1 << leafBits, arity)(values, siblings, keys, root[0], enable);
        _ <== lastLevelsIn;
    }
}

template BN128RootCheck(lastLevels, arity, leafBits) {
    signal input last[clearLevelNodes(arity, lastLevels)][1];
    signal input root[1];
    signal input {binary} enable;

    signal values[clearLevelNodes(arity, lastLevels)];
    for (var i = 0; i < clearLevelNodes(arity, lastLevels); i++) { values[i] <== last[i][0]; }
    VerifyMerkleRoot(lastLevels, arity, 1 << leafBits)(values, root[0], enable);
}
