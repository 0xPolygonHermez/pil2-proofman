pragma circom 2.1.0;
pragma custom_templates;

include "fft.circom";

/*
    base·ω^index (inverse = 0) or base·ω^-index (inverse = 1): the point a query's index selects in a coset
    of the subgroup of order 2^logDomain (ω = roots(logDomain)), index given LSB-first in its low nBits
    bits. Bit i contributes ω^(±2^i) = roots(logDomain - i)^(±1).
*/
template QueryPoint(nBits, logDomain, base, inverse, nIn) {
    signal input {binary} bits[nIn];
    signal output out;

    signal acc[nBits];
    var r0 = inverse == 1 ? invroots(logDomain) : roots(logDomain);
    acc[0] <== bits[0] * (base * r0 - base) + base;
    for (var i = 1; i < nBits; i++) {
        var ri = inverse == 1 ? invroots(logDomain - i) : roots(logDomain - i);
        acc[i] <== acc[i - 1] * (bits[i] * (ri - 1) + 1);
    }
    out <== acc[nBits - 1];
}
