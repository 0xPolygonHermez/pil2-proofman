pragma circom 2.1.0;
pragma custom_templates;

include "blake3.circom";

/*
    BLAKE3 over a stream of Goldilocks words: the chunk chain, the chunk tree,
    and the XOF squeeze. Computes the same function as blake3core::Hasher
    (pil2-stark/src/goldilocks/src/blake3_core.hpp), which is what
    Blake3Goldilocks, TranscriptGL and TranscriptGL_GPU all use.

    It does not mirror that Hasher's *structure*: the Hasher is incremental, so
    it emulates the tree with a merge-while-even cv stack, whereas `n` is known
    here and the tree is written out directly. Agreement therefore rests on a
    test, not on the shapes matching -- see the note above Blake3Subtree.

    One implementation serves both callers, because they are the same hash:

      * the leaf hash (`LinearHash` below) is Blake3Goldilocks::linearHash, i.e.
        hash_le64 over every word of the leaf
      * the Fiat-Shamir transcript is a genuine BLAKE3 over the absorbed stream
        with challenges read from its XOF

    They differ only in what they read out -- four digest words versus a whole
    64-byte XOF block at index `ob` -- and the first four XOF words *are* the
    digest. Note this is NOT the sponge the Poseidon families use: the rate is 8
    words per compression, not 4, because the state lives in the chaining value
    rather than in block content.

    The transcript's incremental path (interleaved absorb and squeeze) cannot be
    a fixed-`n` template, so it is emitted round by round from
    stark2circom/transcript.rs against the same wrappers in blake3.circom.
*/

/*
    Words of the stream that chunks [c0, c0 + nc) cover: a full 128 each, less
    whatever the last chunk of the stream is short by.
*/
function b3_span(n, c0, nc) {
    var hi = 128 * (c0 + nc);
    if (hi > n) {
        hi = n;
    }
    return hi - 128 * c0;
}

/*
    Block `b` of a chunk holding `nw` words, zero-padded past the chunk's end.
    Matches compress_chunk's `(k < in_block) ? in[..] : 0`.
*/
template Blake3BlockWords(nw, b) {
    signal input in[nw];
    signal output out[8];

    for (var k = 0; k < 8; k++) {
        if (8 * b + k < nw) {
            out[k] <== in[8 * b + k];
        } else {
            out[k] <== 0;
        }
    }
}

/*
    Chaining value of chunk `c` after every block but its last. That last block
    is left to the caller: closing a chunk and rooting one differ only in the
    flags they put on it, so this is the part they share.
*/
template Blake3ChunkHead(nw, c) {
    signal input in[nw];
    signal output cv[8];

    var nHead = (nw + 7) \ 8 - 1;
    component blk[nHead + 1];    // +1: circom needs a non-empty component array

    for (var j = 0; j < nHead; j++) {
        // A head block is never the chunk's last, so never CHUNK_END.
        var hf = 0;
        if (j == 0) {
            hf = B3_CHUNK_START();
        }
        blk[j] = Blake3AbsorbBlock(hf);
        for (var i = 0; i < 8; i++) {
            if (j > 0) {
                blk[j].cv[i] <== blk[j - 1].cvOut[i];
            } else {
                blk[j].cv[i] <== B3_IV(i);
            }
        }
        blk[j].in <== Blake3BlockWords(nw, j)(in);
        blk[j].blockLen <== 64;      // only a chunk's last block can be short
        blk[j].counterLo <== c;
    }

    for (var i = 0; i < 8; i++) {
        if (nHead > 0) {
            cv[i] <== blk[nHead - 1].cvOut[i];
        } else {
            cv[i] <== B3_IV(i);
        }
    }
}

/*
    Chaining value of a complete chunk: its head, then its last block carrying
    CHUNK_END -- and CHUNK_START as well when the chunk is a single block.
*/
template Blake3ChunkCV(nw, c) {
    signal input in[nw];
    signal output cv[8];

    var bl = (nw + 7) \ 8 - 1;
    var fl = B3_CHUNK_END();
    if (bl == 0) {
        fl += B3_CHUNK_START();
    }

    cv <== Blake3AbsorbBlock(fl)(Blake3ChunkHead(nw, c)(in),
                                 Blake3BlockWords(nw, bl)(in),
                                 8 * (nw - 8 * bl), c);
}

/*
    Chaining value of the subtree over chunks [c0, c0 + nc), never ROOT.

    BLAKE3 gives the left child the largest power of two below `nc`. That is the
    same shape blake3core::Hasher's merge-while-the-count-is-even stack builds,
    but `n` is known here, so the tree can be named directly instead of
    emulating the stack.
*/
template Blake3Subtree(n, c0, nc) {
    signal input in[b3_span(n, c0, nc)];
    signal output cv[8];

    if (nc == 1) {
        cv <== Blake3ChunkCV(b3_span(n, c0, 1), c0)(in);
    } else {
        var nl = 1;
        while (2 * nl < nc) {
            nl = 2 * nl;
        }
        // The left child is whole chunks, so its span is exactly 128 * nl and
        // the right child's words start there.
        var wl = 128 * nl;
        var wr = b3_span(n, c0 + nl, nc - nl);

        component l = Blake3Subtree(n, c0, nl);
        for (var i = 0; i < wl; i++) {
            l.in[i] <== in[i];
        }
        component r = Blake3Subtree(n, c0 + nl, nc - nl);
        for (var i = 0; i < wr; i++) {
            r.in[i] <== in[wl + i];
        }

        cv <== Blake3Parent(B3_PARENT())(l.cv, r.cv);
    }
}

/*
    64 bytes of BLAKE3 XOF output, at output block `ob`, over a stream of `n`
    Goldilocks words. Equivalent to
        Hasher h; h.absorb(in, n); h.finalize_xof(ob, out)
    and therefore to `b3sum --length 64` of the canonical-LE stream.

    ROOT goes on the top node only, which is why it is applied here and never
    inside the subtree: a one-chunk stream roots that chunk's last block, and
    anything longer roots the final parent.
*/
template Blake3StreamXof(n, ob) {
    // A zero-length `signal input in[0]` is not meaningful in circom -- the
    // circuit degenerates and every output reads as 0 rather than as BLAKE3 of
    // the empty string. Fail at compile time instead of silently. A leaf always
    // has words and a transcript always absorbs before its first squeeze, so
    // this is unreachable in practice; the CPU and Rust implementations do
    // handle n = 0.
    assert(n > 0);

    signal input in[n];
    signal output out[8];

    var nChunks = ((n + 7) \ 8 + 15) \ 16;

    if (nChunks == 1) {
        // One chunk: its last block is the root node.
        var bl = (n + 7) \ 8 - 1;
        var fl = B3_CHUNK_END() + B3_ROOT();
        if (bl == 0) {
            fl += B3_CHUNK_START();
        }
        out <== Blake3FinalizeChunk(fl)(Blake3ChunkHead(n, 0)(in),
                                        Blake3BlockWords(n, bl)(in),
                                        8 * (n - 8 * bl), ob);
    } else {
        var nl = 1;
        while (2 * nl < nChunks) {
            nl = 2 * nl;
        }
        var wl = 128 * nl;
        var wr = b3_span(n, nl, nChunks - nl);

        component l = Blake3Subtree(n, 0, nl);
        for (var i = 0; i < wl; i++) {
            l.in[i] <== in[i];
        }
        component r = Blake3Subtree(n, nl, nChunks - nl);
        for (var i = 0; i < wr; i++) {
            r.in[i] <== in[wl + i];
        }

        out <== Blake3FinalizeParent()(l.cv, r.cv, ob);
    }
}


/*
    Leaf hash: Blake3Goldilocks::linearHash, which is hash_le64 over every word
    of the leaf. Deliberately NOT shaped like the Poseidon LinearHash, which
    chains a capacity between permutations.

    The first four XOF words are the 32-byte digest, so this is Blake3StreamXof
    at output block 0, truncated. Leaves routinely exceed one 128-word chunk, so
    the chunk tree inside is required for correctness, not an optimization.

    `arity` is accepted for signature compatibility with the Poseidon templates
    and must be 2.
*/
template LinearHash(nInputs, arity, eSize) {
    assert(arity == 2);

    signal input in[nInputs][eSize];
    signal output out[4];

    var n = nInputs * eSize;

    signal flat[n];
    for (var w = 0; w < n; w++) {
        flat[w] <== in[w \ eSize][w % eSize];
    }

    signal xof[8] <== Blake3StreamXof(n, 0)(flat);
    for (var i = 0; i < 4; i++) {
        out[i] <== xof[i];
    }
    for (var i = 4; i < 8; i++) {
        _ <== xof[i];
    }
}
