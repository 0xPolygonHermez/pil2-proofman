pragma circom 2.1.0;
pragma custom_templates;

include "geometry.circom";

/*
    The BN128 verifier's Merkle openings of a custom-gate Poseidon tree (circuits.bn128/custom/
    merklehash.circom), which sends no levels in clear: see bn128_open_blake3.circom.
*/

function bn_rootW() { return 1; }
function bn_sibW(arity) { return arity; }

template BN128Open(eSize, nVals, arity, lastLevels, ml, leafBits, nBits0) {
    signal input {binary} queries[nBits0];
    signal input values[nVals][eSize];
    signal input siblings[geo_len1(ml)][arity];
    signal input last[1][1];
    signal input root[1];
    signal input {binary} enable;

    assert(lastLevels == 0);
    signal {binary} keys[leafBits];
    for (var i = 0; i < leafBits; i++) { keys[i] <== queries[i]; }
    VerifyMerkleHash(eSize, nVals, 1 << leafBits, arity)(values, siblings, keys, root[0], enable);
    _ <== last;
}

template BN128RootCheck(lastLevels, arity, leafBits) {
    signal input last[1][1];
    signal input root[1];
    signal input {binary} enable;

    assert(0);
}
