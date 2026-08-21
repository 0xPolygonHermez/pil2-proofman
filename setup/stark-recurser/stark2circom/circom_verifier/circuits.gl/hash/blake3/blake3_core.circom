pragma circom 2.1.0;

/*
    BLAKE3 core, expressed as circom functions over `var` arithmetic.

    This is the single source of truth for what the Blake3 custom gates compute.
    The `extern_c` gate bodies in blake3.circom call these functions; the
    compiler replaces those bodies with a call into
    setup/circom/blake3_goldilocks.cpp, which mirrors them operation for
    operation and cross-checks itself against blake3core.

    Everything here mirrors pil2-stark/src/goldilocks/src/blake3_core.hpp
    operation for operation. Divergence there is a soundness bug: the prover
    hashes with blake3_core.hpp and the verifier must agree bit for bit.

    Limb conventions, matching examples/hashes/pil/blake3.pil:
      * a 32-bit word as 16-bit limbs -> [lo, hi],   word = lo + 2^16*hi
      * a 32-bit word as bytes        -> [b0..b3],   word = sum b_i * 2^(8i)
*/

// ─── Flags ───────────────────────────────────────────────────────────────────

function B3_CHUNK_START() { return 1; }
function B3_CHUNK_END()   { return 2; }
function B3_PARENT()      { return 4; }
function B3_ROOT()        { return 8; }

// ─── im layout ───────────────────────────────────────────────────────────────
//
// Flat rather than rectangular: G rows need 53 cells and the feed-forward
// section needs 160, so a rectangular im would waste ~19% of ~3k real witness
// signals. The future PIL mirrors these offsets.
//
//   [0, 2968)      56 G rows x 53 cells; round r, function g -> row 8r + g
//   [2968, 2984)   st[4..8]  byte limbs, after the last round
//   [2984, 3000)   cv[0..4]  byte limbs
//   [3000, 3064)   out[0..16] byte limbs
//   [3064, 3080)   (dInv, isMax) per Goldilocks input word; zero on the raw path
//
// The feed-forward sections carry only what the G rows do not already hold, in
// the same byte decomposition:
//
//   * rows 52-55 produce the whole final state -- va''/vc''/vd'' are stored as
//     bytes and cover st[0..4] and st[8..16], so only the four vb'' words
//     (st[4..8]) need bytes here. vb'' is an expression over vb''_xor and the
//     carry bit rather than stored bytes, which is why those four are the gap.
//   * cv[4..8] are rows 0-3's vb bytes already; only cv[0..4] need bytes, since
//     those rows store va as 16-bit limbs.
//
// Not deduplicated: each row's input cells duplicate an earlier row's outputs
// (~416 cells). examples/hashes/pil/blake3.pil deliberately keeps those columns
// and ties them with transition constraints rather than consuming rotated
// output expressions, so removing them is an arithmetization choice that
// belongs with the PIL work.
//
// Both gates declare the same im size so plonk2pil sees one row geometry. The
// raw-block path leaves the split section zero and unconstrained, so the round
// constraints are shared and the gates differ only in how the block wires bind.
//
// Blake3Compress takes eight full-range Goldilocks words and splits them into
// the sixteen u32 block words itself, which costs only the 16 canonicity cells
// above: the block words are already limb-decomposed in the G rows, since every
// block word appears once as an `x` or `y` 16-bit limb pair during round 0.
//
// Within a G row:
//   [ 0,  2)  va   16-bit limbs      [16, 20)  va'          bytes
//   [ 2,  6)  vb   bytes             [20, 24)  vd'          bytes
//   [ 6,  8)  vc   16-bit limbs      [24, 28)  vc'          bytes
//   [ 8, 12)  vd   bytes             [28, 36)  vb'_s[4][2]  bytes
//   [12, 14)  x    16-bit limbs      [36, 40)  va''         bytes
//   [14, 16)  y    16-bit limbs      [40, 44)  vd''         bytes
//                                    [44, 48)  vc''         bytes
//                                    [48, 52)  vb''_xor     bytes
//                                    [52, 53)  vb''_t       bit

function B3_ROW_CELLS() { return 53; }
function B3_OFF_ST()    { return 2968; }   // st[4..8] only
function B3_OFF_CV()    { return 2984; }   // cv[0..4] only
function B3_OFF_OUT()   { return 3000; }
function B3_OFF_SPLIT() { return 3064; }
function B3_NIM_RAW()   { return 3064; }
function B3_NIM_GL()    { return 3080; }

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

function b3_limbs16(x) {
    var l[2] = [x & 0xFFFF, (x >> 16) & 0xFFFF];
    return l;
}

function b3_bytes(x) {
    var b[4] = [x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF];
    return b;
}

// ─── Compression ─────────────────────────────────────────────────────────────

/*
    One BLAKE3 compression over a raw sixteen-word u32 block. Returns a flat
    array of B3_NIM_RAW() + 16 values: the im cells followed by the 16 xof
    output words. `b3_compress_gate` below wraps this for both input shapes.

    `out[i]     = st[i] ^ st[i+8]`   for i < 8   (the compress_in_place result)
    `out[8 + i] = st[8+i] ^ cv[i]`   for i < 8   (the extra XOF half)

    Digest users read out[0..8]; the width-8 transcript XOF reads all 16.
*/
function b3_compress(cv, block, blockLen, counterLo, counterHi, flags) {
    var r[3080];

    var st[16];
    for (var i = 0; i < 8; i++) {
        st[i] = cv[i];
        st[8 + i] = 0;
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
            var base = B3_ROW_CELLS() * (8 * round + g);

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

            // Row inputs.
            var vaL[2] = b3_limbs16(va);
            var vbB[4] = b3_bytes(vb);
            var vcL[2] = b3_limbs16(vc);
            var vdB[4] = b3_bytes(vd);
            var xL[2] = b3_limbs16(x);
            var yL[2] = b3_limbs16(y);
            for (var i = 0; i < 2; i++) {
                r[base + 0 + i]  = vaL[i];
                r[base + 6 + i]  = vcL[i];
                r[base + 12 + i] = xL[i];
                r[base + 14 + i] = yL[i];
            }
            for (var i = 0; i < 4; i++) {
                r[base + 2 + i] = vbB[i];
                r[base + 8 + i] = vdB[i];
            }

            // va' = (va + vb + x) mod 2^32
            var vaP = (va + vb + x) & 0xFFFFFFFF;
            var vaPB[4] = b3_bytes(vaP);

            // vd' = (vd ^ va') >>> 16
            var zd = vd ^ vaP;
            var vdP = b3_rotr32(zd, 16);
            var vdPB[4] = b3_bytes(vdP);

            // vc' = (vc + vd') mod 2^32
            var vcP = (vc + vdP) & 0xFFFFFFFF;
            var vcPB[4] = b3_bytes(vcP);

            // vb' = (vb ^ vc') >>> 12
            //
            // The XOR-rotr table splits each operand byte across two result
            // bytes because 12 is not byte-aligned: for operand byte z,
            // c0 = (z & 0xF) << 4 and c1 = z >> 4. The AIR recombines them;
            // here we store both limbs and compute vb' directly.
            var zb = vb ^ vcP;
            var vbP = b3_rotr32(zb, 12);
            var zbB[4] = b3_bytes(zb);
            for (var i = 0; i < 4; i++) {
                r[base + 28 + 2 * i]     = (zbB[i] & 0xF) * 16;
                r[base + 28 + 2 * i + 1] = zbB[i] >> 4;
            }

            // va'' = (va' + vb' + y) mod 2^32
            var vaPP = (vaP + vbP + y) & 0xFFFFFFFF;
            var vaPPB[4] = b3_bytes(vaPP);

            // vd'' = (vd' ^ va'') >>> 8
            var zd2 = vdP ^ vaPP;
            var vdPP = b3_rotr32(zd2, 8);
            var vdPPB[4] = b3_bytes(vdPP);

            // vc'' = (vc' + vd'') mod 2^32
            var vcPP = (vcP + vdPP) & 0xFFFFFFFF;
            var vcPPB[4] = b3_bytes(vcPP);

            // vb'' = (vb' ^ vc'') >>> 7 = rotl1(rotr8(z))
            // Stored as the plain XOR bytes plus the one bit rotl1 carries out.
            var zb2 = vbP ^ vcPP;
            var vbPP = b3_rotr32(zb2, 7);
            var zb2B[4] = b3_bytes(zb2);

            for (var i = 0; i < 4; i++) {
                r[base + 16 + i] = vaPB[i];
                r[base + 20 + i] = vdPB[i];
                r[base + 24 + i] = vcPB[i];
                r[base + 36 + i] = vaPPB[i];
                r[base + 40 + i] = vdPPB[i];
                r[base + 44 + i] = vcPPB[i];
                r[base + 48 + i] = zb2B[i];
            }
            // top bit of rotr8(zb2), i.e. bit 7 of byte 0 of zb2
            r[base + 52] = (zb2B[0] >> 7) & 1;

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

    // st[0..4] and st[8..16] are already rows 52-55's va''/vc''/vd'' bytes; only
    // the vb'' words st[4..8] are missing.
    for (var i = 4; i < 8; i++) {
        var stB[4] = b3_bytes(st[i]);
        for (var b = 0; b < 4; b++) {
            r[B3_OFF_ST() + 4 * (i - 4) + b] = stB[b];
        }
    }
    // cv[4..8] are already rows 0-3's vb bytes; cv[0..4] are stored there as
    // 16-bit limbs, so their bytes are new.
    for (var i = 0; i < 4; i++) {
        var cvB[4] = b3_bytes(cv[i]);
        for (var b = 0; b < 4; b++) {
            r[B3_OFF_CV() + 4 * i + b] = cvB[b];
        }
    }
    for (var i = 0; i < 16; i++) {
        var outB[4] = b3_bytes(out[i]);
        for (var b = 0; b < 4; b++) {
            r[B3_OFF_OUT() + 4 * i + b] = outB[b];
        }
    }

    for (var i = 0; i < 16; i++) {
        r[B3_NIM_RAW() + i] = out[i];
    }

    return r;
}

/*
    Split a canonical Goldilocks element into its (lo, hi) u32 halves plus the
    canonicity witnesses the AIR needs.

    Returns [lo, hi, dInv, isMax].

    Canonicity is a soundness requirement, not bookkeeping: `x = lo + 2^32*hi`
    alone admits a second solution whenever the alternative lands on x + p, and
    the two hash to different digests. Since p - 1 = (2^32 - 1)*2^32, a split is
    canonical iff hi != 2^32 - 1 or lo == 0. `isMax` flags hi == 2^32 - 1 and
    `dInv` witnesses the inverse of (2^32 - 1) - hi when it does not.
*/
function b3_split_word(x) {
    var r[4];

    var lo = x & 0xFFFFFFFF;
    var hi = (x >> 32) & 0xFFFFFFFF;
    r[0] = lo;
    r[1] = hi;

    var d = 0xFFFFFFFF - hi;
    if (d == 0) {
        r[2] = 0;
        r[3] = 1;
    } else {
        r[2] = 1 / d;
        r[3] = 0;
    }

    return r;
}

/*
    One compression, in either of the two shapes the design needs. Returns a
    flat array of B3_NIM_GL() + 16 values: im cells followed by the 16 xof
    output words.

    raw = 0 -- the Goldilocks shape. `a` is the chaining value (u32 words) and
    `b` is a block of eight full-range Goldilocks words, ordered by `key` and
    then split into the sixteen u32 block words, with the canonicity witnesses
    landing at B3_OFF_SPLIT().

    raw = 1 -- the parent shape. The block is two chaining values side by side,
    already u32, so `a` and `b` ARE the block: a is block[0..8], b is
    block[8..16]. The compression's own chaining value is the IV, and the split
    section stays zero because nothing was split. `key` has no effect here --
    a parent has no ordering to choose - so it is a don't-care on this path.

    The two 8-word input arrays are exactly 16 slots, which is why the parent
    shape needs no separate block input and the whole design fits one gate.
*/
function b3_compress_gate(in, blockLen, counterLo, counterHi, flags, key, raw) {
    var cv[8];
    var block[16];
    var wit[16];

    if (raw == 1) {
        for (var i = 0; i < 8; i++) {
            cv[i] = B3_IV(i);
            block[i] = in[i];
            block[8 + i] = in[8 + i];
            wit[2 * i] = 0;
            wit[2 * i + 1] = 0;
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
            var sp[4] = b3_split_word(ordered[i]);
            block[2 * i] = sp[0];
            block[2 * i + 1] = sp[1];
            wit[2 * i] = sp[2];
            wit[2 * i + 1] = sp[3];
        }
    }

    var core[3080] = b3_compress(cv, block, blockLen, counterLo, counterHi, flags);

    var r[3096];
    for (var i = 0; i < B3_NIM_RAW(); i++) {
        r[i] = core[i];
    }
    for (var i = 0; i < 16; i++) {
        r[B3_OFF_SPLIT() + i] = wit[i];
        r[B3_NIM_GL() + i] = core[B3_NIM_RAW() + i];
    }

    return r;
}
