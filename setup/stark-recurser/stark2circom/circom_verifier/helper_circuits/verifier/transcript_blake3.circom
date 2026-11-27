pragma circom 2.1.0;
pragma custom_templates;

// Relative, so the BN128 verifier (no circuits.gl on its include path) finds it too.
include "../../circuits.gl/hash/blake3/blake3.circom";
include "transcript_words_gl.circom";

/*
    Fiat-Shamir transcript, blake3: a real BLAKE3 chunk chain over the absorbed words, the prover's
    TranscriptGL (blake3) in circom. A squeeze roots a copy of the chain (ROOT is
    terminal, the transcript keeps absorbing) and reads its XOF words in order.

    OPS[2k] is 0 (absorb) or 1 (squeeze), OPS[2k+1] how many words; `in` holds the absorbed words
    and `out` the squeezed ones, each in order.
*/
template TranscriptEngine(nIn, nOut, nOps, OPS) {
    signal input in[nIn];
    signal output out[nOut];

    // Stack depth bound: one chaining value per set bit of the completed-chunk count.
    var depth = 1;
    while ((1 << depth) <= nIn \ 128 + 1) { depth++; }

    // Upper bounds; entries past the last use are never created.
    component blk[nIn \ 8 + nOut + 2];
    component par[nIn \ 128 + 1 + (nOut + 1) * depth];
    component fch[nOut + 1];
    component fpa[nOut + 1];

    var buf[8];
    var nBuf = 0;
    var cv[8];
    for (var i = 0; i < 8; i++) { cv[i] = B3_IV(i); }
    var stack[depth + 1][8];
    var nStack = 0;
    var chunkBlocks = 0;
    var chunkCounter = 0;
    var chunksDone = 0;
    var xof[8];
    var xofValid = 0;
    var xofOffset = 0;
    var ob = 0;
    var nB = 0;
    var nP = 0;
    var nFC = 0;
    var nFP = 0;
    var ii = 0;
    var oo = 0;
    for (var k = 0; k < nOps; k++) {
        for (var j = 0; j < OPS[2 * k + 1]; j++) {
            if (OPS[2 * k] == 0) {
                // A block is emitted only once more input is known to follow.
                if (nBuf == 8) {
                    var fl = 0;
                    if (chunkBlocks == 0) { fl += B3_CHUNK_START(); }
                    if (chunkBlocks == 15) { fl += B3_CHUNK_END(); }
                    blk[nB] = Blake3AbsorbBlock(fl);
                    for (var i = 0; i < 8; i++) {
                        blk[nB].cv[i] <== cv[i];
                        blk[nB].in[i] <== buf[i];
                    }
                    blk[nB].blockLen <== 64;
                    blk[nB].counterLo <== chunkCounter;
                    for (var i = 0; i < 8; i++) { cv[i] = blk[nB].cvOut[i]; }
                    nB++;
                    nBuf = 0;
                    chunkBlocks++;
                    if (chunkBlocks == 16) {
                        // Push the chunk's cv, merging while the completed count is even.
                        var node[8];
                        for (var i = 0; i < 8; i++) { node[i] = cv[i]; }
                        var total = chunksDone + 1;
                        var merging = 1;
                        for (var d = 0; d < depth; d++) {
                            if (merging == 1 && total % 2 == 0) {
                                nStack--;
                                par[nP] = Blake3Parent(B3_PARENT());
                                for (var i = 0; i < 8; i++) {
                                    par[nP].left[i] <== stack[nStack][i];
                                    par[nP].right[i] <== node[i];
                                }
                                for (var i = 0; i < 8; i++) { node[i] = par[nP].cvOut[i]; }
                                nP++;
                                total = total \ 2;
                            } else {
                                merging = 0;
                            }
                        }
                        for (var i = 0; i < 8; i++) { stack[nStack][i] = node[i]; }
                        nStack++;
                        chunksDone++;
                        chunkCounter++;
                        chunkBlocks = 0;
                        for (var i = 0; i < 8; i++) { cv[i] = B3_IV(i); }
                    }
                }
                buf[nBuf] = in[ii];
                ii++;
                nBuf++;
                // The stream changed, so XOF words of the old prefix are stale.
                xofValid = 0;
                xofOffset = 0;
                ob = 0;
            } else {
                if (xofValid == 1 && xofOffset == 8) { ob++; }
                if (xofValid == 0 || xofOffset == 8) {
                    var words[8];
                    for (var i = 0; i < 8; i++) {
                        if (i < nBuf) { words[i] = buf[i]; } else { words[i] = 0; }
                    }
                    var fl = B3_CHUNK_END();
                    if (chunkBlocks == 0) { fl += B3_CHUNK_START(); }
                    if (nStack == 0) {
                        // Single chunk: the held-back block is the root node.
                        fch[nFC] = Blake3FinalizeChunk(fl + B3_ROOT());
                        for (var i = 0; i < 8; i++) {
                            fch[nFC].cv[i] <== cv[i];
                            fch[nFC].in[i] <== words[i];
                        }
                        fch[nFC].blockLen <== 8 * nBuf;
                        fch[nFC].ob <== ob;
                        for (var i = 0; i < 8; i++) { xof[i] = fch[nFC].out[i]; }
                        nFC++;
                    } else {
                        // Multi-chunk: close this chunk, merge a copy of the stack; the last parent is the root.
                        blk[nB] = Blake3AbsorbBlock(fl);
                        for (var i = 0; i < 8; i++) {
                            blk[nB].cv[i] <== cv[i];
                            blk[nB].in[i] <== words[i];
                        }
                        blk[nB].blockLen <== 8 * nBuf;
                        blk[nB].counterLo <== chunkCounter;
                        var node[8];
                        for (var i = 0; i < 8; i++) { node[i] = blk[nB].cvOut[i]; }
                        nB++;
                        for (var s = nStack - 1; s >= 1; s--) {
                            par[nP] = Blake3Parent(B3_PARENT());
                            for (var i = 0; i < 8; i++) {
                                par[nP].left[i] <== stack[s][i];
                                par[nP].right[i] <== node[i];
                            }
                            for (var i = 0; i < 8; i++) { node[i] = par[nP].cvOut[i]; }
                            nP++;
                        }
                        fpa[nFP] = Blake3FinalizeParent();
                        for (var i = 0; i < 8; i++) {
                            fpa[nFP].left[i] <== stack[0][i];
                            fpa[nFP].right[i] <== node[i];
                        }
                        fpa[nFP].ob <== ob;
                        for (var i = 0; i < 8; i++) { xof[i] = fpa[nFP].out[i]; }
                        nFP++;
                    }
                    xofValid = 1;
                    xofOffset = 0;
                }
                out[oo] <== xof[xofOffset];
                xofOffset++;
                oo++;
            }
        }
    }
}

// One step of the vadcop lattice chain: a whole XOF block, the prover's chain width.
function chain_w() { return 8; }

template ChainPermutation() {
    signal input in[8];
    signal output out[8];

    out <== Blake3Permute8()(in);
}
