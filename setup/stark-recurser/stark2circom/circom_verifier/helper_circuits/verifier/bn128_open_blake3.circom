pragma circom 2.1.0;
pragma custom_templates;

include "custom/gl_family.circom";
include "evalpol.circom";
include "bitifyT.circom";
include "treeselector.circom";
include "../../circuits.gl/hash/blake3/merklehash.circom";
include "../../circuits.gl/hash/blake3/pow.circom";
include "transcript_blake3.circom";
include "merkle.circom";
include "stark_verifier_bn128.circom";

/*
    The BN128 verifier of a Goldilocks blake3 tree (roots of four words), the one include of its
    generated verifier: its field, hash and transcript files and the family half of
    stark_verifier_bn128.circom. The Poseidon entries (bn128_open_poseidon.circom,
    bn128_open_custom.circom) export the same templates.
*/

function bn_rootW() { return 4; }
function bn_sibW(arity) { return siblingWords(arity); }

// One opened leaf of nVals elements of eSize limbs, below the levels sent in clear.
template BN128Open(eSize, nVals, arity, lastLevels, ml, leafBits, nBits0) {
    signal input {binary} queries[nBits0];
    signal input values[nVals][eSize];
    signal input siblings[nonEmpty(ml)][siblingWords(arity)];
    signal input last[clearLevelNodes(arity, lastLevels)][4];
    signal input root[4];
    signal input {binary} enable;

    // circuits.bn128's b3_field and selectval, not circuits.gl's: their pick is the include path's.
    assert(b3_field_is_gl() == 0);
    if (lastLevels > 0) {
        VerifyTreeOpening(eSize, nVals, arity, lastLevels, ml, leafBits)(QueryKeys(arity, ml + lastLevels, leafBits, nBits0)(queries), values, siblings, last, [0, 0, 0, 0], enable);
        _ <== root;
    } else {
        VerifyTreeOpening(eSize, nVals, arity, lastLevels, ml, leafBits)(QueryKeys(arity, ml + lastLevels, leafBits, nBits0)(queries), values, siblings, [[0, 0, 0, 0]], root, enable);
        _ <== last;
    }
}

// The levels sent in clear hash to the committed root.
template BN128RootCheck(lastLevels, arity, leafBits) {
    signal input last[clearLevelNodes(arity, lastLevels)][4];
    signal input root[4];
    signal input {binary} enable;

    VerifyMerkleRoot(lastLevels, arity, 1 << leafBits)(last, root, enable);
}
