pragma circom 2.1.0;
pragma custom_templates;

// The 16-bit chunks of in, from the least significant: out[k] holds bits 16k to 16k+15. A custom
// gate: circom only computes it, and its witness refuses an in of more than nBits bits. The gate
// plonk2pil lays out for it constrains every chunk below 2^16 and in to their sum, so what it
// proves is in < 2^(16·ceil(nBits/16)), a whole number of chunks. The signals of a use in the r1cs
// are in, then out.
template custom extern_c Num2Bytes(nBits) {
    assert(nBits <= 80);
    var nBytes = (nBits + 15)\16;
    signal input in;
    signal output out[nBytes];

    var lc1=0;
    var e2=1;

    var b = 0;
    var e = 1;
    for(var i =0; i < nBits; i++) {
        b += ((in >> i) & 1) * e;
        e = e+e;
        if(i%16 == 15) {
            out[i\16] <-- b;
            assert(b < 65536);
            lc1 += b * e2;
            e2 = e2*65536;
            b = 0;
            e = 1;
        }
    }

    if(nBits%16 != 0) {
        out[nBytes - 1] <-- b;
        lc1 += b * e2;
    }

    assert(lc1 == in);
}

// Checks in < 2^(16·ceil(nBits/16)), as Num2Bytes does: with one gate up to 80 bits, and above
// that with two, on the low 80 bits of in and on the rest, which the r1cs adds back up to in.
template RangeCheck(nBits) {
    assert(nBits <= 160);
    // The callers are Goldilocks quotients k, whose k·p + r, with r < 2^64, must stay below the
    // BN128 order (> 2^253) for their equations to hold over the integers.
    assert(16 * ((nBits + 15)\16) + 64 < 253);
    signal input in;

    if (nBits <= 80) {
        _ <== Num2Bytes(nBits)(in);
    } else {
        signal lo <-- in & ((1 << 80) - 1);
        signal hi <-- in >> 80;
        _ <== Num2Bytes(80)(lo);
        _ <== Num2Bytes(nBits - 80)(hi);
        in === lo + hi * (1 << 80);
    }
}
