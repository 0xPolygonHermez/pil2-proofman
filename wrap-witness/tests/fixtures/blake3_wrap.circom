pragma circom 2.1.0;
pragma custom_templates;

include "hash/blake3/blake3.circom";
include "custom/rangecheck.circom";

// The circuit of tests/device.rs: each block kind of the blake3 BN128 wrap and a range check.
// - `over`: a Node input whose digest word 3 is p or more (input.json), so its `over` bit is 1;
// - `loZero`: one whose digest word 0 has a zero low half, so that word's d_inv is free;
// - a chunk and a parent Compress, and a Num2Bytes of the Nodes' outputs.
template Blake3WrapUses() {
    signal input over[8];
    signal input loZero[8];
    signal input key;
    signal input chunk[16];
    signal input parent[16];
    signal input blockLen;
    signal input counterLo;
    signal output out;

    signal a[4] <== Blake3Node()(over, key);
    signal b[4] <== Blake3Node()(loZero, key);
    signal c[16] <== Blake3Compress(3, 0)(chunk, blockLen, counterLo);
    signal d[16] <== Blake3Compress(4, 1)(parent, blockLen, counterLo);
    signal r[4] <== Num2Bytes(64)(a[3]);

    out <== a[0] + b[0] + c[0] + d[0] + r[0];
}

component main = Blake3WrapUses();
