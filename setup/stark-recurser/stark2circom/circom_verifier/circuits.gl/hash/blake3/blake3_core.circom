pragma circom 2.1.0;

/*
    BLAKE3 core, expressed as circom functions over `var` arithmetic.

    This is the single source of truth for what the Blake3 custom gates compute.
    The `extern_c` gate bodies in blake3.circom call these functions; the
    compiler replaces those bodies with a call into
    setup/circom/blake3_goldilocks.cpp, which computes the same function by
    calling the shared core the prover hashes with.

    Everything here mirrors pil2-stark/src/goldilocks/src/blake3_core.hpp
    operation for operation. Divergence there is a soundness bug: the prover
    hashes with blake3_core.hpp and the verifier must agree bit for bit.
    `spec_and_witness_gate_agree` differences the two.

    These functions cover the compression itself. Its arithmetization -- the
    per-G round intermediates and their byte and 16-bit-limb decompositions --
    lives in examples/hashes/pil/blake3.pil.
*/

// ─── Flags ───────────────────────────────────────────────────────────────────

function B3_CHUNK_START() { return 1; }
function B3_CHUNK_END()   { return 2; }
function B3_PARENT()      { return 4; }
function B3_ROOT()        { return 8; }

// ─── Constants ───────────────────────────────────────────────────────────────

function B3_IV(i) {
    var iv[8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    ];
    return iv[i];
}

// Message schedule, flattened to 7*16 so it is indexable at runtime.
function B3_SIGMA(r, k) {
    var s[112] = [
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
         2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8,
         3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1,
        10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6,
        12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4,
         9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7,
        11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13
    ];
    return s[16 * r + k];
}

// The four state words a G function mixes, for function g of a round.
// g in [0,4) are the column mixes, g in [4,8) the diagonal mixes.
function B3_G_IDX(g, j) {
    var idx[32] = [
        0, 4,  8, 12,
        1, 5,  9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
        0, 5, 10, 15,
        1, 6, 11, 12,
        2, 7,  8, 13,
        3, 4,  9, 14
    ];
    return idx[4 * g + j];
}

// ─── Word helpers ────────────────────────────────────────────────────────────

function b3_rotr32(x, n) {
    return (x >> n) | ((x << (32 - n)) & 0xFFFFFFFF);
}

// ─── Compression ─────────────────────────────────────────────────────────────

/*
    One BLAKE3 compression over a raw sixteen-word u32 block, returning the 16
    xof output words. `b3_compress_gate` below wraps this for both input shapes.

    `out[i]     = st[i] ^ st[i+8]`   for i < 8   (the compress_in_place result)
    `out[8 + i] = st[8+i] ^ cv[i]`   for i < 8   (the extra XOF half)

    Digest users read out[0..8]; the width-8 transcript XOF reads all 16.
*/
function b3_compress(cv, block, blockLen, counterLo, counterHi, flags) {
    var st[16];
    for (var i = 0; i < 8; i++) {
        st[i] = cv[i];
    }
    st[8]  = B3_IV(0);
    st[9]  = B3_IV(1);
    st[10] = B3_IV(2);
    st[11] = B3_IV(3);
    st[12] = counterLo;
    st[13] = counterHi;
    st[14] = blockLen;
    st[15] = flags;

    for (var round = 0; round < 7; round++) {
        for (var g = 0; g < 8; g++) {
            var ia = B3_G_IDX(g, 0);
            var ib = B3_G_IDX(g, 1);
            var ic = B3_G_IDX(g, 2);
            var id = B3_G_IDX(g, 3);

            var va = st[ia];
            var vb = st[ib];
            var vc = st[ic];
            var vd = st[id];
            var x = block[B3_SIGMA(round, 2 * g)];
            var y = block[B3_SIGMA(round, 2 * g + 1)];

            var vaP = (va + vb + x) & 0xFFFFFFFF;
            var vdP = b3_rotr32(vd ^ vaP, 16);
            var vcP = (vc + vdP) & 0xFFFFFFFF;
            var vbP = b3_rotr32(vb ^ vcP, 12);

            var vaPP = (vaP + vbP + y) & 0xFFFFFFFF;
            var vdPP = b3_rotr32(vdP ^ vaPP, 8);
            var vcPP = (vcP + vdPP) & 0xFFFFFFFF;
            var vbPP = b3_rotr32(vbP ^ vcPP, 7);

            st[ia] = vaPP;
            st[ib] = vbPP;
            st[ic] = vcPP;
            st[id] = vdPP;
        }
    }

    // Feed-forward.
    var out[16];
    for (var i = 0; i < 8; i++) {
        out[i] = st[i] ^ st[8 + i];
        out[8 + i] = st[8 + i] ^ cv[i];
    }
    return out;
}

/*
    Split a Goldilocks element into its (lo, hi) u32 halves.

    The canonical representative is what gets split, and that matters for
    soundness rather than being bookkeeping: `x = lo + 2^32*hi` alone admits a
    second solution whenever the alternative lands on x + p, and the two hash to
    different digests. Since p - 1 = (2^32 - 1)*2^32, the non-canonical
    representations are exactly {hi = 2^32 - 1, lo > 0}. A Blake3 AIR has to rule
    those out; the witness side is handed canonical values and just splits them.
*/
function b3_split_word(x) {
    var r[2] = [x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF];
    return r;
}

/*
    One compression, in either of the two shapes the design needs, returning the
    16 xof output words.

    raw = 0 -- the Goldilocks shape. `in[0..8]` is the chaining value (u32 words)
    and `in[8..16]` a block of eight full-range Goldilocks words, ordered by
    `key` and then split into the sixteen u32 block words.

    raw = 1 -- the parent shape. The block is two chaining values side by side,
    already u32, so `in` IS the block. The compression's own chaining value is
    the IV. `key` has no effect here -- a parent has no ordering to choose -- so
    it is a don't-care on this path.

    A cv plus a Goldilocks block is exactly 16 cells, the same as a raw block,
    which is why the parent shape needs no block input of its own.
*/
function b3_compress_gate(in, blockLen, counterLo, counterHi, flags, key, raw) {
    var cv[8];
    var block[16];

    if (raw == 1) {
        for (var i = 0; i < 8; i++) {
            cv[i] = B3_IV(i);
            block[i] = in[i];
            block[8 + i] = in[8 + i];
        }
    } else {
        // Order the two 4-word halves by the key bit before splitting.
        var ordered[8];
        for (var i = 0; i < 4; i++) {
            if (key == 0) {
                ordered[i] = in[8 + i];
                ordered[4 + i] = in[12 + i];
            } else {
                ordered[i] = in[12 + i];
                ordered[4 + i] = in[8 + i];
            }
        }
        for (var i = 0; i < 8; i++) {
            cv[i] = in[i];
            var sp[2] = b3_split_word(ordered[i]);
            block[2 * i] = sp[0];
            block[2 * i + 1] = sp[1];
        }
    }

    return b3_compress(cv, block, blockLen, counterLo, counterHi, flags);
}
