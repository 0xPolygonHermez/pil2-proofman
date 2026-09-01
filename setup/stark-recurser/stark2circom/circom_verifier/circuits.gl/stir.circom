pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "iszero.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";

/*
    Helpers for the in-circuit STIR verifier (Arnon–Chiesa–Fenzi–Yogev, ePrint 2024/390,
    Construction 5.2), transcribing pil2-stark/src/starkpil/stir/stir_math.hpp.

    Notation, as in the C++:
      L_i   iteration i's evaluation domain, the coset shift·⟨ω_{|L_i|}⟩ (every L_i shares
            the same shift, only the subgroup shrinks);
      k_i   the folding factor, in bits: iteration i folds by 2^{k_i};
      G_i   the quotient set of iteration i: the out-of-domain sample r_out plus the fresh
            (first-occurrence) shift queries, as points of L_{i−1}^{k_{i−1}};
      Âns_i the interpolation of the claimed values on G_i.
*/

// Whether an Fp³ element is zero (all three coordinates).
template StirIsZero3() {
    signal input in[3];
    signal output {binary} out;

    signal {binary} z0 <== IsZero()(in[0]);
    signal {binary} z1 <== IsZero()(in[1]);
    signal {binary} z2 <== IsZero()(in[2]);
    signal {binary} z01 <== z0 * z1;
    out <== z01 * z2;
}

// Whether two indices, given LSB-first in bits, are equal. Used to detect repeated shift
// queries: G is a set, so a repeated point is dropped (prover and verifier alike).
template StirEqualBits(nBits) {
    signal input {binary} a[nBits];
    signal input {binary} b[nBits];
    signal output {binary} out;

    // a XOR b = a + b − 2ab for bits; the indices are equal iff the sum of XORs is zero.
    signal ab[nBits];
    var acc = 0;
    for (var i = 0; i < nBits; i++) {
        ab[i] <== a[i] * b[i];
        acc += a[i] + b[i] - 2 * ab[i];
    }
    signal diff <== acc;
    out <== IsZero()(diff);
}

// Whether a < b, for a, b < 2^nBits. Used to zero the tail of the Âns coefficient hints:
// position m is a live coefficient iff m < |G|.
template StirLessThan(nBits) {
    signal input a;
    signal input b;
    signal output {binary} out;

    signal {binary} bits[nBits + 1] <== Num2Bits(nBits + 1)(a + (1 << nBits) - b);
    out <== 1 - bits[nBits];
}

// base·ω^{index} (inv = 0) or base·ω^{−index} (inv = 1), where ω generates the subgroup of
// order 2^{logDomain} and `index` is given LSB-first in bits. The same accumulation as
// VerifyQuery's xacc: bit i multiplies by ω^{±2^i} = (inv)roots(logDomain − i).
template StirPointAcc(nBits, logDomain, base, inv) {
    signal input {binary} bits[nBits];
    signal output out;

    signal acc[nBits];
    var r0 = inv == 1 ? invroots(logDomain) : roots(logDomain);
    acc[0] <== bits[0] * (base * r0 - base) + base;
    for (var i = 1; i < nBits; i++) {
        var ri = inv == 1 ? invroots(logDomain - i) : roots(logDomain - i);
        acc[i] <== acc[i - 1] * (bits[i] * (ri - 1) + 1);
    }
    out <== acc[nBits - 1];
}

// y^e in Fp³ for a data-dependent exponent e < 2^maxBits, given LSB-first in bits.
// Square-and-multiply with a per-bit select of the square or 1.
template StirPowE3(maxBits) {
    signal input y[3];
    signal input {binary} expBits[maxBits];
    signal output out[3];

    signal sq[maxBits][3];
    signal sel[maxBits][3];
    signal acc[maxBits][3];
    sq[0] <== y;
    for (var i = 1; i < maxBits; i++) {
        sq[i] <== CMul()(sq[i - 1], sq[i - 1]);
    }
    for (var i = 0; i < maxBits; i++) {
        // expBits[i] ? sq[i] : 1
        sel[i][0] <== expBits[i] * (sq[i][0] - 1) + 1;
        sel[i][1] <== expBits[i] * sq[i][1];
        sel[i][2] <== expBits[i] * sq[i][2];
    }
    acc[0] <== sel[0];
    for (var i = 1; i < maxBits; i++) {
        acc[i] <== CMul()(acc[i - 1], sel[i]);
    }
    out <== acc[maxBits - 1];
}

// The degree-correction factor Σ_{j=0}^{e} y^j = (y^{e+1} − 1)/(y − 1), with the y = 1 case
// equal to e + 1 (stir_math.hpp geometricSum). The exponent is data-dependent — e = |G| after
// dropping repeated shift queries — so it arrives as bits of e+1 plus the matching field value.
// Branch-free: when y = 1 the quotient term is 0/adjusted-denominator = 0 and the correction
// term denZero·(e+1) supplies the answer, so no signal is ever inverted at zero.
template StirGeoSum(maxBits) {
    signal input y[3];
    signal input {binary} expBits[maxBits]; // e + 1, LSB-first
    signal input expVal;                    // e + 1 as a field element (callers bind Σ bits·2^i)
    signal output out[3];

    signal pow[3] <== StirPowE3(maxBits)(y, expBits);   // y^{e+1}
    signal den[3] <== [y[0] - 1, y[1], y[2]];
    signal num[3] <== [pow[0] - 1, pow[1], pow[2]];
    signal {binary} denZero <== StirIsZero3()(den);

    // (y − 1) + denZero never vanishes, and when y = 1 the numerator is 0 anyway.
    signal denAdj[3] <== [den[0] + denZero, den[1], den[2]];
    signal dinv[3] <== CInv()(denAdj);
    signal quot[3] <== CMul()(num, dinv);

    out[0] <== quot[0] + denZero * expVal;
    out[1] <== quot[1];
    out[2] <== quot[2];
}

// Fold(f, 2^logK, r) evaluated at one point of L^{2^logK}, from the opened coset of f on L
// (stir_math.hpp foldCoset). The coset of leaf m is {shift·ω^{m + j·2^{logL−logK}}}: an IFFT
// of its k values gives q with p̂_x(X) = q(X/c), c = shift·ω^m, so the fold value is q(r/c).
template StirFoldCoset(logK, logL, shiftInv) {
    signal input vals[1 << logK][3];
    signal input {binary} leafBits[logL - logK];
    signal input rFold[3];
    signal output out[3];

    signal coefs[1 << logK][3] <== FFT(logK, 3, 1)(vals);
    signal cInv <== StirPointAcc(logL - logK, logL, shiftInv, 1)(leafBits);
    signal evalX[3] <== [rFold[0] * cInv, rFold[1] * cInv, rFold[2] * cInv];
    out <== EvalPol(1 << logK)(coefs, evalX);
}
