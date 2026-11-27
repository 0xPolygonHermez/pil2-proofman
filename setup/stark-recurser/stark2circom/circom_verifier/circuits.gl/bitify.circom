pragma circom 2.1.0;

include "iszero.circom";

template LessThan20Bits() {
    signal input in;

    _ <== Num2Bits(20)(in);    
}

template Num2Bits(n) {
    signal input in;
    signal output {binary} out[n];
    var lc1=0;

    var e2=1;
    for (var i = 0; i<n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] -1 ) === 0;
        lc1 += out[i] * e2;
        e2 = e2+e2;
    }

    lc1 === in;
}

template AliasCheck() {
    signal input {binary} in[64];

    // The Goldilocks prime p = 2^64 - 2^32 + 1 is expressed in binary as:
    //  p = 0b1111111111111111111111111111111100000000000000000000000000000001
    // Thus, checking that in < p is equivalent to checking that if all the 
    // 32 most-significant bits are 1, then all the 32 least-significant bits must be 0.
    
    var least_sig_32_sum = 0;
    for (var i = 0; i < 32; i++){
        least_sig_32_sum += in[i];
    }

    var most_sig_32_sum = 0;
    for (var i = 32; i < 64; i++){
        most_sig_32_sum += in[i];
    }
    
    signal all_zero <== IsZero()(least_sig_32_sum);
    signal all_one <== IsZero()(32 - most_sig_32_sum);
    
    // all_one implies all_zero
    all_one * (1 - all_zero) === 0; 
}

template Num2Bits_strict() {
    signal input in;
    signal output {binary} out[64];

    signal n2b[64] <== Num2Bits(64)(in);
    
    AliasCheck()(n2b);
    out <== n2b;
}

template Bits2Num(n) {
    signal input {binary} in[n];
    signal output out;
    var lc1=0;

    var e2 = 1;
    for (var i = 0; i<n; i++) {
        lc1 += in[i] * e2;
        e2 = e2 + e2;
    }

    lc1 ==> out;
}

// A transcript word's bits (verifier/transcript.circom): a Goldilocks word, 64 bits.
template TranscriptWordBits(n) {
    signal input in;
    signal output {binary} out[n];

    assert(n == 64);
    out <== Num2Bits_strict()(in);
}
