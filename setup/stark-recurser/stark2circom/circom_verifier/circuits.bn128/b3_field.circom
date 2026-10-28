pragma circom 2.1.0;

/*
    What the blake3 circuits (circuits.gl/hash/blake3/) need of their field, here BN128, where a
    Goldilocks word is a field element below p that nothing reduces: what the Goldilocks field
    does for free is done here. circuits.gl has the Goldilocks one.
*/

include "bitify.circom";

function b3_field_is_gl() {
    return 0;
}

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
