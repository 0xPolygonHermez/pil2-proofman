pragma circom 2.1.0;
pragma custom_templates;

/*
    What the blake3 circuits (hash/blake3/) need of their field, here Goldilocks; circuits.bn128
    has the BN128 verifier's. Each is found by the include path, so the circuits are one set.
*/

include "bitify.circom";

// The circuits are Goldilocks's: a compile-time choice, so its r1cs is the plain one's.
function b3_field_is_gl() {
    return 1;
}

// A u32 pair as a canonical Goldilocks word, for a witness.
function b3_pack_value(lo, hi) {
    return lo + 4294967296 * hi;
}

// A u32 pair as a Goldilocks word: the field reduces lo + 2^32*hi for free.
template B3Pack() {
    signal input lo;
    signal input hi;
    signal output out;

    out <== lo + 4294967296 * hi;
}

// The 64 bits of a canonical Goldilocks element, for the grinding check.
template B3PowBits() {
    signal input in;
    signal output out[64];

    out <== Num2Bits_strict()(in);
}
