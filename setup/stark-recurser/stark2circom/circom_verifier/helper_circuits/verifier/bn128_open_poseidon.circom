pragma circom 2.1.0;

include "geometry.circom";

/*
    The BN128 verifier's Merkle openings of a BN128 Poseidon tree (circuits.bn128/merklehash.circom,
    one-element roots): see bn128_open_blake3.circom.
*/

function bn_rootW() { return 1; }
function bn_sibW(arity) { return arity; }

template BN128Open(eSize, nVals, arity, lastLevels, ml, leafBits, nBits0) {
    signal input {binary} queries[nBits0];
    signal input values[nVals][eSize];
    signal input siblings[geo_len1(ml)][arity];
    signal input last[geo_lastSize(arity, lastLevels)][1];
    signal input root[1];
    signal input {binary} enable;

    signal {binary} keys[leafBits];
    for (var i = 0; i < leafBits; i++) { keys[i] <== queries[i]; }
    signal lastLevelsIn[geo_lastSize(arity, lastLevels)];
    for (var i = 0; i < geo_lastSize(arity, lastLevels); i++) { lastLevelsIn[i] <== last[i][0]; }
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
    signal input last[geo_lastSize(arity, lastLevels)][1];
    signal input root[1];
    signal input {binary} enable;

    signal values[geo_lastSize(arity, lastLevels)];
    for (var i = 0; i < geo_lastSize(arity, lastLevels); i++) { values[i] <== last[i][0]; }
    VerifyMerkleRoot(lastLevels, arity, 1 << leafBits)(values, root[0], enable);
}
