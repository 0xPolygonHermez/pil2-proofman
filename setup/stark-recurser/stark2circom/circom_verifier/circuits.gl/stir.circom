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

// Which shift queries are the first occurrence of their point. G is a set, so a repeated
// point is dropped (prover and verifier alike): fresh[q] = 1 iff no earlier query hit pts[q].
// The point is injective in the leaf index, so comparing field values is exact, and
// Π_{m<q}(pts[q] − pts[m]) vanishes iff some earlier point equals pts[q]: one constraint per
// pair plus one zero test per query.
template StirFresh(t) {
    signal input pts[t];
    signal output {binary} fresh[t];

    signal prod[t][t];
    signal {binary} seen[t];
    for (var q = 0; q < t; q++) {
        if (q == 0) {
            fresh[0] <== 1;
        } else {
            for (var m = 0; m < q; m++) {
                if (m == 0) {
                    prod[q][0] <== pts[q] - pts[0];
                } else {
                    prod[q][m] <== prod[q][m - 1] * (pts[q] - pts[m]);
                }
            }
            seen[q] <== IsZero()(prod[q][q - 1]);
            fresh[q] <== 1 - seen[q];
        }
    }
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

// y^e in F_p for a data-dependent exponent e < 2^maxBits, given LSB-first in bits.
template StirPowBase(maxBits) {
    signal input y;
    signal input {binary} expBits[maxBits];
    signal output out;

    signal sq[maxBits];
    signal sel[maxBits];
    signal acc[maxBits];
    sq[0] <== y;
    for (var i = 1; i < maxBits; i++) {
        sq[i] <== sq[i - 1] * sq[i - 1];
    }
    for (var i = 0; i < maxBits; i++) {
        sel[i] <== expBits[i] * (sq[i] - 1) + 1;
    }
    acc[0] <== sel[0];
    for (var i = 1; i < maxBits; i++) {
        acc[i] <== acc[i - 1] * sel[i];
    }
    out <== acc[maxBits - 1];
}

// One-hot decoding of an nBits value given LSB-first in bits: ind[v] = 1 iff bits encode v.
// Used to pick the constant ω^{j·v} of a data-dependent exponent v = e mod 2^k.
template StirIndicators(nBits) {
    signal input {binary} bits[nBits];
    signal output {binary} ind[1 << nBits];

    var n = 1 << nBits;
    signal lv[nBits + 1][n];
    lv[0][0] <== 1;
    for (var v = 1; v < n; v++) { lv[0][v] <== 0; }
    for (var i = 0; i < nBits; i++) {
        var w = 1 << i;
        for (var v = 0; v < w; v++) {
            lv[i + 1][v + w] <== lv[i][v] * bits[i];
            lv[i + 1][v] <== lv[i][v] - lv[i + 1][v + w];
        }
        for (var v = 2 * w; v < n; v++) { lv[i + 1][v] <== 0; }
    }
    for (var v = 0; v < n; v++) { ind[v] <== lv[nBits][v]; }
}

// Coefficients (low degree first) of U(X) = Π_m (X − h[m]), monic of degree t. With
// h[m] = fresh[m]·g_m this is the vanishing polynomial of the fresh shift points times X^d,
// d = number of dropped repeats: one product per coefficient per factor, t(t+1)/2 in all.
template StirVanishingShifted(t) {
    signal input h[t];
    signal output coefs[t + 1];

    signal u[t + 1][t + 1];   // u[m] = the coefficients after m factors (degree m)
    signal hu[t][t + 1];      // h[m] · u[m][k]
    u[0][0] <== 1;
    for (var k = 1; k <= t; k++) { u[0][k] <== 0; }
    for (var m = 0; m < t; m++) {
        for (var k = 0; k <= t; k++) {
            if (k <= m) {
                hu[m][k] <== h[m] * u[m][k];
            } else {
                hu[m][k] <== 0;
            }
            if (k == 0) {
                u[m + 1][0] <== - hu[m][0];
            } else {
                u[m + 1][k] <== u[m][k - 1] - hu[m][k];
            }
        }
    }
    coefs <== u[t];
}

// A base-field polynomial P (n coefficients, low degree first) evaluated on the whole coset
// {c·ω^u : u < 2^logK}, ω = roots(logK). Every coset point satisfies X^{2^k} = c^{2^k} =: y, so
// P agrees on the coset with its remainder R(X) = Σ_{j<2^k} R_j X^j, R_j = Σ_m P[2^k·m + j] y^m:
// 2^k Horner passes in y (n multiplications in all), then P(c·ω^u) = Σ_j (R_j c^j) ω^{ju}, a
// size-2^k DFT with constant twiddles. Output u is the member ordering of the leaf.
template StirCosetEval1(n, logK) {
    var K = 1 << logK;
    var nmax = (n + K - 1) \ K;
    signal input coefs[n];
    signal input c;
    signal output vals[K];

    signal cp[logK + 1];
    cp[0] <== c;
    for (var i = 1; i <= logK; i++) { cp[i] <== cp[i - 1] * cp[i - 1]; }

    signal acc[K][nmax];
    for (var j = 0; j < K; j++) {
        for (var m = 0; m < nmax; m++) {
            var idx = K * (nmax - 1 - m) + j;
            if (m == 0) {
                if (idx < n) { acc[j][0] <== coefs[idx]; } else { acc[j][0] <== 0; }
            } else {
                if (idx < n) { acc[j][m] <== acc[j][m - 1] * cp[logK] + coefs[idx]; }
                else { acc[j][m] <== acc[j][m - 1] * cp[logK]; }
            }
        }
    }
    signal cj[K];
    signal sc[K];
    cj[0] <== 1;
    for (var j = 1; j < K; j++) {
        if (j == 1) { cj[1] <== c; } else { cj[j] <== cj[j - 1] * c; }
    }
    for (var j = 0; j < K; j++) { sc[j] <== acc[j][nmax - 1] * cj[j]; }
    for (var u = 0; u < K; u++) {
        var sum = 0;
        for (var j = 0; j < K; j++) { sum += sc[j] * (roots(logK) ** (u * j)); }
        vals[u] <== sum;
    }
}

// StirCosetEval1 for F_p³ coefficients: the Horner passes run in EvalPol (EvPol4 gates) at the
// base point y = c^{2^k}, the scaling and the DFT act coordinate-wise.
template StirCosetEval3(n, logK) {
    var K = 1 << logK;
    var nmax = (n + K - 1) \ K;
    signal input coefs[n][3];
    signal input c;
    signal output vals[K][3];

    signal cp[logK + 1];
    cp[0] <== c;
    for (var i = 1; i <= logK; i++) { cp[i] <== cp[i - 1] * cp[i - 1]; }

    signal sub[K][nmax][3];
    signal res[K][3];
    for (var j = 0; j < K; j++) {
        for (var m = 0; m < nmax; m++) {
            var idx = K * m + j;
            for (var e = 0; e < 3; e++) {
                if (idx < n) { sub[j][m][e] <== coefs[idx][e]; } else { sub[j][m][e] <== 0; }
            }
        }
        res[j] <== EvalPol(nmax)(sub[j], [cp[logK], 0, 0]);
    }
    signal cj[K];
    signal sc[K][3];
    cj[0] <== 1;
    for (var j = 1; j < K; j++) {
        if (j == 1) { cj[1] <== c; } else { cj[j] <== cj[j - 1] * c; }
    }
    for (var j = 0; j < K; j++) {
        for (var e = 0; e < 3; e++) { sc[j][e] <== res[j][e] * cj[j]; }
    }
    for (var u = 0; u < K; u++) {
        for (var e = 0; e < 3; e++) {
            var sum = 0;
            for (var j = 0; j < K; j++) { sum += sc[j][e] * (roots(logK) ** (u * j)); }
            vals[u][e] <== sum;
        }
    }
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
