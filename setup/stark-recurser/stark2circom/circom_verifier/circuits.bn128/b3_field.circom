pragma circom 2.1.0;

/*
    What the blake3 circuits (circuits.gl/hash/blake3/) need of their field, here BN128, where a
    Goldilocks word is a field element below p that nothing reduces: what the Goldilocks field
    does for free is done here. circuits.gl has the Goldilocks one.
*/

include "bitify.circom";

// A u32 pair as a canonical Goldilocks word, for a witness.
function b3_pack_value(lo, hi) {
    return (lo + 4294967296 * hi) % 0xFFFFFFFF00000001;
}

/*
    A u32 pair as a canonical Goldilocks word: lo + 2^32*hi, less p when that is p or more, which
    is exactly when hi = 2^32 - 1 and lo > 0. lo and hi are the gate's u32 outputs, which its PIL
    bounds.
*/
template B3Pack() {
    signal input lo;
    signal input hi;
    signal output out;

    var p = 0xFFFFFFFF00000001;
    signal dHi <== hi - 4294967295;
    signal invHi <-- dHi != 0 ? 1 / dHi : 0;
    signal isMax <== 1 - dHi * invHi;
    isMax * dHi === 0;
    signal invLo <-- lo != 0 ? 1 / lo : 0;
    signal nzLo <== lo * invLo;
    (1 - nzLo) * lo === 0;
    signal over <== isMax * nzLo;
    out <== lo + 4294967296 * hi - over * p;
}

// The 64 bits of a Node digest, for the grinding check: neither non-canonical form fits 64 bits
// with its top powBits clear.
template B3PowBits() {
    signal input in;
    signal output out[64];

    out <== Num2Bits(64)(in);
}

/*
    The node a path's last levels lead to, out of the published level. The Goldilocks verifier
    selects through a custom gate (SelectValueArity2); the BN128 wrap has no such gate, so the
    selection is key*(b - a) + a, one product per word per node per level, which plonk2pil lays
    out as PLONK.
*/
template B3SelectValue(arity, nLastLevels, num_nodes_level) {
    assert(arity == 2);
    signal input values[arity**nLastLevels][4];
    signal input {binary} key[nLastLevels][1];
    signal output selected_value[4];

    if (nLastLevels == 0) {
        selected_value <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity - 1)) \ arity;
        component mNext = B3SelectValue(arity, nLastLevels - 1, next_n);
        for (var j = 0; j < next_n; j++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[j][t] <== key[0][0] * (values[2 * j + 1][t] - values[2 * j][t]) + values[2 * j][t];
            }
        }
        for (var k = next_n; k < arity**(nLastLevels - 1); k++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[k][t] <== 0;
            }
        }
        signal {binary} keyTags[nLastLevels - 1][1];
        for (var b = 0; b < nLastLevels - 1; b++) {
            keyTags[b] <== key[b + 1];
        }
        mNext.key <== keyTags;
        selected_value <== mNext.selected_value;
    }
}
