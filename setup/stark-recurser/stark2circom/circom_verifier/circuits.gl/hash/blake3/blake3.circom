pragma circom 2.1.0;
pragma custom_templates;

include "blake3_core.circom";

/*
    Blake3 custom gate and the primitives built on it.

    Every hash the verifier needs (leaf hash, Merkle node, chunk parent,
    transcript absorb, transcript squeeze, PoW) is one BLAKE3 compression --
    7 rounds of 8 G functions. Circom passes values and does no arithmetic of
    its own.

    Two gates compute it. Blake3Compress is the general one, taking a chaining
    value and either input shape; Blake3Node is the Merkle-node specialisation,
    which is most of a verifier's compressions and spends a third of the rows
    per call. See the note on Blake3Node.

    Blake3Compress takes two input shapes over one 16-cell row geometry,
    selected by `raw`:

      raw = 0   `in[0..8]` is the chaining value (u32), `in[8..16]` a block of
                eight full-range Goldilocks words, which the gate splits into
                the sixteen u32 block words itself, ordering the two 4-word
                halves by `key`.

      raw = 1   `in` IS the sixteen u32 block words -- two chaining values side
                by side -- and the compression's chaining value is the IV.
                `key` is a don't-care here, so the selector multiplies the
                ordering term away and the cell is free.

    The raw shape exists because chaining values cannot be packed into
    Goldilocks and split back: pack(lo, 2^32-1) = lo + p - 1 = lo - 1, which
    re-splits to (lo-1, 0), so a parent's arbitrary 512-bit block does not
    survive the round trip. Reachable only inside a BLAKE3 chunk tree -- leaves
    over 128 words, transcripts past their first chunk -- which is about one
    compression in sixteen. The Merkle tree and PoW go through Blake3Hash8,
    which is raw = 0.

    Both shapes need exactly 16 input cells, so one gate serves both.

    Packing u32 back to Goldilocks needs no gate: to_canonical(lo + 2^32*hi) is
    exactly reduction mod p for any u64 (a u64 is always below 2p), so a digest
    word is the plain linear combination `lo + 2^32*hi`.

    `counterHi` is not wired: chunk indices are capped at 2^24 by the native
    CV_STACK, the XOF output block index is 0 at width 8, and parents use
    counter 0, so st[13] is identically zero across the design and is hardcoded.

    `blockLen`, `counterLo`, `flags` and `raw` are signals driven by circuit
    constants rather than template parameters, because `counterLo` as a
    parameter would spawn one gate id per chunk index.
*/

// ─── Gates ───────────────────────────────────────────────────────────────────

template custom extern_c Blake3Compress() {
    signal input in[16];       // raw=0: cv[0..8] (u32), then the Goldilocks block
                               //        -- the two halves are differently typed
                               // raw=1: the sixteen u32 block words
    signal input blockLen;     // 8 * valid words, -> st[14]
    signal input counterLo;    // chunk index, or the XOF block index at a root
    signal input flags;        //                  -> st[15]
    signal input key;          // raw=0: 0 identity, 1 swaps the halves. raw=1: unused
    signal input raw;          // 0: split `b` from Goldilocks.  1: `a`||`b` is the block
    signal output out[16];     // out[i]=st[i]^st[i+8]; out[8+i]=st[8+i]^cv[i]

    // `key` and `raw` must be boolean, but no assert here can enforce it: this
    // body is dead under extern_c (the compiler replaces it with a call into
    // blake3_goldilocks.cpp), so a witness-time assert on a signal never runs.
    // Booleanity is the AIR's job; the witness side treats any nonzero as 1.

    var cells[16] = in;
    var r[16] = b3_compress_gate(cells, blockLen, counterLo, 0, flags, key, raw);

    for (var i = 0; i < 16; i++) {
        out[i] <-- r[i];
    }
}

/*
    The Merkle-node compression: eight Goldilocks words at the IV with the root
    flags, digest packed back to four words. The overwhelming majority of a
    verifier's compressions; Blake3Compress serves the rest.

    Its own gate because every wire crossing a custom-gate boundary costs a
    linear constraint that --O2 cannot substitute through -- the application
    names signal indices. This shape spends 5 per call where driving
    Blake3Compress spends 17, the difference being eight constant B3_IV cells
    and eight u32 output halves a caller would recombine.

    The AIR binds less here, not more: the chaining value is a constant rather
    than eight input cells, and out[i] ties to its own final-state u32 columns
    as lo + 2^32*hi -- exactly reduction mod p for any u64, so no range check
    of its own.
*/
template custom extern_c Blake3Node() {
    signal input in[8];
    signal input key;
    signal output out[4];

    var iv[16];
    for (var i = 0; i < 8; i++) {
        iv[i]     = B3_IV(i);
        iv[8 + i] = in[i];
    }
    var r[16] = b3_compress_gate(iv, 64, 0, 0,
                                 B3_CHUNK_START() + B3_CHUNK_END() + B3_ROOT(), key, 0);
    for (var i = 0; i < 4; i++) {
        out[i] <-- r[2 * i] + 4294967296 * r[2 * i + 1];
    }
}

// ─── Digest ──────────────────────────────────────────────────────────────────

/*
    hash_le64 over eight Goldilocks words: a whole 64-byte block hashed on its
    own, at the IV, with the root flags set. This is simultaneously
    permuteTrunc, the Merkle node hash and the PoW permutation.

    `key` orders the two halves inside the gate, as CustPoseidon2_16 does (see
    poseidonInputOrder in poseidon2.pil): a masked combination in the block
    binding, one degree bump instead of the ~8 plonk rows per Merkle level that
    circom muxes would spend. Pass 0 where there is no path bit.
*/
template Blake3Hash8() {
    signal input in[8];
    signal input key;
    signal output out[4];

    component c = Blake3Node();
    c.in <== in;
    c.key <== key;
    out <== c.out;
}


/*
    blake3core::permute8: absorb eight Goldilocks words, squeeze the whole 64-byte XOF
    block back. Drop-in for a width-8 sponge permutation, and the same single
    compression as Blake3Hash8 -- that one just truncates to the four digest words.

    Drives the general gate rather than Blake3Node because it needs all eight
    squeezed words, and Blake3Node publishes only the four digest ones.
*/
template Blake3Permute8() {
    signal input in[8];
    signal output out[8];

    component c = Blake3Compress();
    for (var i = 0; i < 8; i++) {
        c.in[i]     <== B3_IV(i);
        c.in[8 + i] <== in[i];
    }
    c.blockLen <== 64;
    c.counterLo <== 0;
    c.flags <== B3_CHUNK_START() + B3_CHUNK_END() + B3_ROOT();
    c.key <== 0;
    c.raw <== 0;

    for (var i = 0; i < 8; i++) {
        out[i] <== c.out[2 * i] + 4294967296 * c.out[2 * i + 1];
    }
}

// ─── Absorption ──────────────────────────────────────────────────────────────

/*
    Absorb one 8-word block into a chaining value. `cv` and `cvOut` are u32
    words and never pass through a Goldilocks packing, because to_canonical is
    lossy and chaining through it would diverge from the native hash.

    Padding words of a short final block are passed as 0 by the caller, matching
    compress_chunk's `(k < in_block) ? in[..] : 0`.
*/
template Blake3AbsorbBlock() {
    signal input cv[8];
    signal input in[8];
    signal input blockLen;
    signal input counterLo;
    signal input flags;
    signal output cvOut[8];

    component c = Blake3Compress();
    for (var i = 0; i < 8; i++) {
        c.in[i]     <== cv[i];
        c.in[8 + i] <== in[i];
    }
    c.blockLen <== blockLen;
    c.counterLo <== counterLo;
    c.flags <== flags;
    c.key <== 0;
    c.raw <== 0;

    for (var i = 0; i < 8; i++) {
        cvOut[i] <== c.out[i];
    }
    for (var i = 8; i < 16; i++) {
        _ <== c.out[i];
    }
}

/*
    parent_cv: merge two chaining values. `flags` is PARENT, or PARENT | ROOT
    for the final merge.
*/
template Blake3Parent() {
    signal input left[8];
    signal input right[8];
    signal input flags;
    signal output cvOut[8];

    component c = Blake3Compress();
    for (var i = 0; i < 8; i++) {
        c.in[i]     <== left[i];
        c.in[8 + i] <== right[i];
    }
    c.blockLen <== 64;
    c.counterLo <== 0;
    c.flags <== flags;
    c.key <== 0;
    c.raw <== 1;

    for (var i = 0; i < 8; i++) {
        cvOut[i] <== c.out[i];
    }
    for (var i = 8; i < 16; i++) {
        _ <== c.out[i];
    }
}

// ─── Finalization ────────────────────────────────────────────────────────────
//
// ROOT is terminal in BLAKE3 while a transcript keeps absorbing after a
// challenge, so a squeeze roots a *copy* of the chain. That is why finalization
// is separate from absorption rather than the same permutation, and why both
// forms return all sixteen xof words (64 bytes) rather than just the digest.

/*
    Root a single-chunk chain: the held-back final block becomes the root node,
    carrying CHUNK_END | ROOT (plus CHUNK_START when it is also the chunk's
    first block). `ob` is the XOF output-block index -- BLAKE3 puts it in the
    root node's counter field, which is why it lands on counterLo here.

    Returns 8 Goldilocks words = 64 bytes of XOF output.
*/
template Blake3FinalizeChunk() {
    signal input cv[8];        // u32-valued chaining value of the open chunk
    signal input in[8];        // the held-back block, zero-padded
    signal input blockLen;     // 8 * valid words in that block
    signal input flags;        // CHUNK_END | ROOT [| CHUNK_START]
    signal input ob;           // XOF output-block index
    signal output out[8];

    component c = Blake3Compress();
    for (var i = 0; i < 8; i++) {
        c.in[i]     <== cv[i];
        c.in[8 + i] <== in[i];
    }
    c.blockLen <== blockLen;
    c.counterLo <== ob;
    c.flags <== flags;
    c.key <== 0;
    c.raw <== 0;

    for (var i = 0; i < 8; i++) {
        out[i] <== c.out[2 * i] + 4294967296 * c.out[2 * i + 1];
    }
}

/*
    Root a multi-chunk chain: the final parent node, carrying PARENT | ROOT.
    Same `ob` convention.
*/
template Blake3FinalizeParent() {
    signal input left[8];      // u32-valued
    signal input right[8];     // u32-valued
    signal input ob;
    signal output out[8];

    component c = Blake3Compress();
    for (var i = 0; i < 8; i++) {
        c.in[i]     <== left[i];
        c.in[8 + i] <== right[i];
    }
    c.blockLen <== 64;
    c.counterLo <== ob;
    c.flags <== B3_PARENT() + B3_ROOT();
    c.key <== 0;
    c.raw <== 1;

    for (var i = 0; i < 8; i++) {
        out[i] <== c.out[2 * i] + 4294967296 * c.out[2 * i + 1];
    }
}
