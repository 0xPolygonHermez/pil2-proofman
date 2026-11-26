pragma circom 2.1.0;
pragma custom_templates;

include "sha256/sha256.circom";
include "bitify.circom";

/*
    The final circuit's publics hash: sha256 over the publics (each checked below Goldilocks, with the
    LessThanGoldilocks of the verifier's include) and rootCVadcopFinal, as one field element.
    DEFS[6d..6d+5] per publics_info definition: initialPos, nValues, nChunks, bitsPerChunk, avoidAlias,
    verificationKey. A definition is packed byte-swapped per chunk; a single-chunk verification key
    big-endian, its top bit 0 under avoidAlias.
*/
template GetSha256Inputs(nPublics, nDefs, DEFS) {
    signal input publics[nPublics];
    signal input rootCVadcopFinal[4];
    signal output publicsHash;

    for (var i = 0; i < nPublics; i++) {
        _ <== LessThanGoldilocks()(publics[i]);
    }

    var totalBits = 256;
    var nWords = 0;
    for (var d = 0; d < nDefs; d++) {
        totalBits += DEFS[6 * d + 1] * DEFS[6 * d + 2] * DEFS[6 * d + 3];
        nWords += DEFS[6 * d + 1] * DEFS[6 * d + 2];
        // The byte-swapped form has no slot for a dropped top bit.
        assert(DEFS[6 * d + 4] == 0 || (DEFS[6 * d + 5] == 1 && DEFS[6 * d + 2] == 1));
    }

    component publicsHasher = Sha256(totalBits);
    component words[nWords > 0 ? nWords : 1];

    var off = 0;
    var w = 0;
    for (var d = 0; d < nDefs; d++) {
        var pos = DEFS[6 * d];
        var nValues = DEFS[6 * d + 1];
        var nChunks = DEFS[6 * d + 2];
        var bits = DEFS[6 * d + 3];
        var vk = nChunks == 1 && DEFS[6 * d + 5] == 1;
        var eff = DEFS[6 * d + 4] == 1 ? bits - 1 : bits;
        for (var k = 0; k < nValues; k++) {
            for (var i = 0; i < nChunks; i++) {
                words[w] = Num2Bits(eff);
                words[w].in <== publics[pos + k * nChunks + i];
                for (var j = 0; j < eff; j++) {
                    if (vk) {
                        publicsHasher.in[off + k * bits + bits - 1 - j] <== words[w].out[j];
                    } else {
                        publicsHasher.in[off + (k * nChunks + i) * bits + (j \ 8) * 8 + (7 - j % 8)] <== words[w].out[j];
                    }
                }
                if (eff < bits) {
                    publicsHasher.in[off + k * bits] <== 0;
                }
                w++;
            }
        }
        off += nValues * nChunks * bits;
    }

    component rootCBits[4];
    for (var i = 0; i < 4; i++) {
        rootCBits[i] = Num2Bits(64);
        rootCBits[i].in <== rootCVadcopFinal[i];
        for (var j = 0; j < 64; j++) {
            publicsHasher.in[off + i * 64 + 63 - j] <== rootCBits[i].out[j];
        }
    }

    component b2nPublicsHash = Bits2Num(256);
    for (var i = 0; i < 256; i++) {
        b2nPublicsHash.in[i] <== publicsHasher.out[255 - i];
    }

    publicsHash <== b2nPublicsHash.out;
}
