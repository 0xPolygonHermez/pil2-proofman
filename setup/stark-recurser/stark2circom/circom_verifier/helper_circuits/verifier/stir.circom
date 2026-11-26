pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "iszero.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "proof_shape.circom";
include "transcript.circom";
include "fri.circom";
include "query_point.circom";

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

// A base-field polynomial P evaluated on the whole coset
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

// StirCosetEval1 for F_p³ coefficients: the Horner passes run coordinate-wise at the base point
// y = c^{2^k}, as do the scaling and the DFT.
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

    vals <== FFT(logK, 3, 0)(sc);
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
    signal cInv <== QueryPoint(logL - logK, logL, shiftInv, 1, logL - logK)(leafBits);
    signal evalX[3] <== [rFold[0] * cInv, rFold[1] * cInv, rFold[2] * cInv];
    out <== EvalPol(1 << logK)(coefs, evalX);
}

// Round r's leaf bits (the low log|L_r| - k_r query bits) and the point of L_r^{2^k_r} each query folds to.
template StirShiftPoints(t, logL, logK) {
    var nLeaf = logL - logK;
    signal input {binary} queries[t][logL];
    signal output {binary} leafBits[t][nLeaf];
    signal output gpt[t];

    var shiftPow = domainShift();
    for (var i = 0; i < logK; i++) { shiftPow = shiftPow * shiftPow; }
    for (var q = 0; q < t; q++) {
        for (var i = 0; i < nLeaf; i++) { leafBits[q][i] <== queries[q][i]; }
        gpt[q] <== QueryPoint(nLeaf, nLeaf, shiftPow, 0, nLeaf)(leafBits[q]);
    }
}

// Fold(f, 2^k, rFold) at each query's point, from f on its opened coset.
template StirFoldQueries(t, logL, logK) {
    signal input vals[t][1 << logK][3];
    signal input {binary} leafBits[t][logL - logK];
    signal input rFold[3];
    signal output out[t][3];

    for (var q = 0; q < t; q++) {
        out[q] <== StirFoldCoset(logK, logL, 1 / domainShift())(vals[q], leafBits[q], rFold);
    }
}

/*
    Round r >= 1: opens T_r (holding g_r) and evaluates the virtual f_r on each opened coset,
        f(x) = (g(x) - Ans(x)) * V_G(x)^{-1} * sum_{j=0}^{|G|} (rComb x)^j,
    with round r-1's quotient data (Ans hint, U = V_G / (X - rOut) * X^d, d and |G| + 1 in bits).
    Member j of leaf m sits at x = c w^j, c = shift w^m; Ans and U are evaluated per coset
    (StirCosetEval*), (rComb x)^{|G|+1} = (rComb c)^{|G|+1} w^{j(|G|+1)} once per coset.
*/
template StirVirtualOracle(arity, lastLevels, logL, logK, t, prevT) {
    var K = 1 << logK;
    var nLeaf = logL - logK;
    var nG = prevT + 1;
    var eb = stir_expBits(prevT);
    var ml = merklePathLength(arity, lastLevels, nLeaf);

    signal input {binary} queries[t][logL];
    signal input vals[t][K][3];
    signal input siblings[t][nonEmpty(ml)][siblingWords(arity)];
    signal input last[clearLevelNodes(arity, lastLevels)][4];
    signal input root[4];
    signal input ansPrev[nG][3];
    signal input uPrev[nG];
    signal input {binary} dBits[eb];
    signal input {binary} expBits[eb];
    signal input rComb[3];
    signal input rOut[3];
    signal input {binary} enable;
    signal output fx[t][K][3];

    for (var q = 0; q < t; q++) {
        VerifyTreeOpening(3, K, arity, lastLevels, ml, nLeaf)(QueryKeys(arity, ml + lastLevels, nLeaf, logL)(queries[q]), vals[q], siblings[q], last, root, enable);
    }

    signal {binary} leafBits[t][nLeaf];
    signal cPt[t];
    signal xm[t][K];
    for (var q = 0; q < t; q++) {
        for (var i = 0; i < nLeaf; i++) { leafBits[q][i] <== queries[q][i]; }
        cPt[q] <== QueryPoint(nLeaf, logL, domainShift(), 0, nLeaf)(leafBits[q]);
        // VERIFY-CONSTRAINTS: member offsets as roots(logK) ** j (today GL_W[logK]^j printed by Rust).
        for (var j = 0; j < K; j++) { xm[q][j] <== cPt[q] * (roots(logK) ** j); }
    }

    // Constant twiddles w^{jv} selected by the data-dependent v = d mod 2^k (dropped repeats) and
    // v = (|G| + 1) mod 2^k (degree-correction exponent), from their low k bits.
    signal {binary} dLow[logK];
    signal {binary} eLow[logK];
    for (var i = 0; i < logK; i++) {
        dLow[i] <== dBits[i];
        eLow[i] <== expBits[i];
    }
    signal {binary} indD[K] <== StirIndicators(logK)(dLow);
    signal {binary} indE[K] <== StirIndicators(logK)(eLow);
    signal wD[K];
    signal wE[K];
    for (var j = 0; j < K; j++) {
        var accD = 0;
        var accE = 0;
        for (var v = 0; v < K; v++) {
            accD += indD[v] * (roots(logK) ** (j * v));
            accE += indE[v] * (roots(logK) ** (j * v));
        }
        wD[j] <== accD;
        wE[j] <== accE;
    }

    signal ansAt[t][K][3];
    signal uAt[t][K];
    signal cD[t];
    // (rComb c)^e = rComb^e c^e: the extension power is shared by every query of the round.
    signal rCombE[3] <== StirPowE3(eb)(rComb, expBits);
    signal cE[t];
    signal zPow[t][3];
    signal xD[t][K];
    signal vD[t][K][3];
    signal den[t][K][3];
    signal vP[t][K][3];
    signal num[t][K][3];
    signal numS[t][K][3];
    signal gNum[t][K][3];
    for (var q = 0; q < t; q++) {
        ansAt[q] <== StirCosetEval3(nG, logK)(ansPrev, cPt[q]);
        uAt[q] <== StirCosetEval1(nG, logK)(uPrev, cPt[q]);
        cD[q] <== StirPowBase(eb)(cPt[q], dBits);
        cE[q] <== StirPowBase(eb)(cPt[q], expBits);
        for (var e = 0; e < 3; e++) { zPow[q][e] <== rCombE[e] * cE[q]; }
        for (var j = 0; j < K; j++) {
            // V_G(x) = (x - rOut) U(x) x^{-d}, nonzero as G and L are disjoint; x^d = c^d w^{jd}.
            xD[q][j] <== cD[q] * wD[j];
            vD[q][j][0] <== (xm[q][j] - rOut[0]) * uAt[q][j];
            vD[q][j][1] <== (- rOut[1]) * uAt[q][j];
            vD[q][j][2] <== (- rOut[2]) * uAt[q][j];
            // sum_{i<=|G|} y^i = (y^{|G|+1} - 1) / (y - 1), y = rComb x != 1 (rComb is not in F_p).
            for (var e = 0; e < 3; e++) {
                den[q][j][e] <== rComb[e] * xm[q][j] - (e == 0 ? 1 : 0);
                num[q][j][e] <== zPow[q][e] * wE[j] - (e == 0 ? 1 : 0);
            }
            // One division: f(x) = (g(x) - Ans(x)) x^d num / ((x - rOut) U(x) (y - 1)).
            vP[q][j] <== CMul()(vD[q][j], den[q][j]);
            for (var e = 0; e < 3; e++) { numS[q][j][e] <== num[q][j][e] * xD[q][j]; }
            gNum[q][j] <== CMul()([vals[q][j][0] - ansAt[q][j][0], vals[q][j][1] - ansAt[q][j][1], vals[q][j][2] - ansAt[q][j][2]], numS[q][j]);
            fx[q][j] <== CDiv()(gNum[q][j], vP[q][j]);
        }
    }
}

/*
    Round r's quotient set G_{r+1} = {rOut} + its fresh shift points, and the shake check (stir.hpp
    step 2(f)): Ans interpolates G exactly when Sh(X) = sum_{(a,y) in G} (Ans(X) - y) / (X - a) is a
    polynomial, tested at rho, squeezed after both hints. The zero-padded hint tails are pinned.
    Outputs the data round r + 1 evaluates its virtual oracle with.
*/
template StirQuotient(t) {
    var nG = t + 1;
    var eb = stir_expBits(t);

    signal input gpt[t];
    signal input foldVals[t][3];
    signal input ans[nG][3];
    signal input shake[nG][3];
    signal input rho[3];
    signal input rOut[3];
    signal input beta[3];
    signal input {binary} enable;
    signal output uCoefs[nG];
    signal output {binary} dBits[eb];
    signal output {binary} expBits[eb];

    // A repeated shift query is dropped (sampling is with replacement).
    signal {binary} fresh[t] <== StirFresh(t)(gpt);

    var nGAcc = 1;
    for (var q = 0; q < t; q++) { nGAcc += fresh[q]; }
    signal nGs <== nGAcc;
    expBits <== Num2Bits(eb)(nGs + 1);

    // U(X) = prod_q (X - fresh_q gpt_q) = V_{fresh points}(X) X^d, d = t + 1 - |G| dropped repeats.
    signal hPt[t];
    for (var q = 0; q < t; q++) { hPt[q] <== fresh[q] * gpt[q]; }
    uCoefs <== StirVanishingShifted(t)(hPt);
    signal dDup <== t + 1 - nGs;
    dBits <== Num2Bits(eb)(dDup);
    signal {binary} ansUsed[nG];
    for (var m = 0; m < nG; m++) {
        ansUsed[m] <== StirLessThan(eb)(m, nGs);
        for (var e = 0; e < 3; e++) {
            (1 - ansUsed[m]) * ans[m][e] === 0;
            (1 - ansUsed[m]) * shake[m][e] === 0;
        }
    }

    signal ansAtRho[3] <== EvalPol(nG)(ans, rho);
    signal shakeAtRho[3] <== EvalPol(nG)(shake, rho);
    signal invRhoOut[3] <== CInv()([rho[0] - rOut[0], rho[1] - rOut[1], rho[2] - rOut[2]]);
    signal termOut[3] <== CMul()([ansAtRho[0] - beta[0], ansAtRho[1] - beta[1], ansAtRho[2] - beta[2]], invRhoOut);
    signal invRhoG[t][3];
    signal termG[t][3];
    signal shakeAcc[t + 1][3];
    shakeAcc[0] <== termOut;
    for (var q = 0; q < t; q++) {
        invRhoG[q] <== CInv()([rho[0] - gpt[q], rho[1], rho[2]]);
        termG[q] <== CMul()([ansAtRho[0] - foldVals[q][0], ansAtRho[1] - foldVals[q][1], ansAtRho[2] - foldVals[q][2]], invRhoG[q]);
        for (var e = 0; e < 3; e++) {
            shakeAcc[q + 1][e] <== shakeAcc[q][e] + fresh[q] * termG[q][e];
        }
    }
    for (var e = 0; e < 3; e++) {
        enable * (shakeAtRho[e] - shakeAcc[t][e]) === 0;
    }
}

// The last round: Fold(f, 2^k, rFold) at each query's point equals finalPol (coefficients) there.
template StirFinalCheck(t, F) {
    signal input gpt[t];
    signal input foldVals[t][3];
    signal input finalPol[F][3];
    signal input {binary} enable;

    signal finalAt[t][3];
    for (var q = 0; q < t; q++) {
        finalAt[q] <== EvalPol(F)(finalPol, [gpt[q], 0, 0]);
        for (var e = 0; e < 3; e++) {
            enable * (finalAt[q][e] - foldVals[q][e]) === 0;
        }
    }
}

/*
    STIR after the L_0 queries (fri.circom with friFold = 0 opens L_0 and T_0 and checks DEEP):
    per round r, fold f_r at its queries (f_0 from T_0's cosets, f_r >= 1 the virtual oracle of T_r),
    then round r's quotient data (r < M - 1) or the final check (r = M - 1). Rounds r >= 1 derive
    their own queries; T_r's levels sent in clear hash up to its root.

    Flat inputs (offsets in proof_shape.circom): vals = every round's opened cosets, siblings = rounds
    >= 1's Merkle paths, hints = per round r < M - 1 its Ans then its shake coefficients.
*/
template StirVerifyRounds(arity, lastLevels, stirM, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, STIR_GRINDING_BITS, stirFinalBits) {
    var M = stirM;
    var LL = lastLevels;
    var wordsPerLevel = siblingWords(arity);
    var lastSize = clearLevelNodes(arity, LL);

    signal input {binary} queriesL0[STIR_NUM_QUERIES[0]][STIR_LOG_DOMAIN_SIZES[0]];
    signal input ldt[ldt_n(1, M)][3];
    signal input nonces[M];
    signal input vals[stir_valsOff(STIR_NUM_QUERIES, STIR_FOLDING_FACTORS, M)][3];
    signal input siblings[nonEmpty(stir_sibOff(arity, LL, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, M))][wordsPerLevel];
    signal input last[M][lastSize][4];
    signal input roots[M][4];
    signal input hints[nonEmpty(stir_nHints(1, M, STIR_NUM_QUERIES))][3];
    signal input betas[nonEmpty(M - 1)][3];
    signal input finalPol[1 << stirFinalBits][3];
    signal input {binary} enable;

    component qb[M];
    component pts[M];
    component oracle[M];
    component fold[M];
    component quot[M];
    component fin[1];
    for (var r = 0; r < M; r++) {
        var logL = STIR_LOG_DOMAIN_SIZES[r];
        var logK = STIR_FOLDING_FACTORS[r];
        var K = 1 << logK;
        var t = STIR_NUM_QUERIES[r];
        var vo = stir_valsOff(STIR_NUM_QUERIES, STIR_FOLDING_FACTORS, r);

        if (r > 0) {
            qb[r] = QueryBits(t, logL, STIR_GRINDING_BITS[r], 1);
            qb[r].challenge <== ldt[ldt_query(1, M, r)];
            qb[r].nonce <== nonces[r];
            qb[r].enable <== enable;
        }
        pts[r] = StirShiftPoints(t, logL, logK);
        if (r == 0) {
            pts[r].queries <== queriesL0;
        } else {
            pts[r].queries <== qb[r].queries;
        }

        if (r > 0) {
            var ml = merklePathLength(arity, LL, logL - logK);
            var so = stir_sibOff(arity, LL, STIR_LOG_DOMAIN_SIZES, STIR_FOLDING_FACTORS, STIR_NUM_QUERIES, r);
            var hp = stir_hintOff(STIR_NUM_QUERIES, r - 1);
            oracle[r] = StirVirtualOracle(arity, LL, logL, logK, t, STIR_NUM_QUERIES[r - 1]);
            oracle[r].queries <== qb[r].queries;
            for (var q = 0; q < t; q++) {
                for (var c = 0; c < K; c++) { oracle[r].vals[q][c] <== vals[vo + q * K + c]; }
                for (var j = 0; j < ml; j++) { oracle[r].siblings[q][j] <== siblings[so + q * ml + j]; }
                // VERIFY-CONSTRAINTS: the unused path of an Empty tree is wired to constants.
                if (ml == 0) { for (var k = 0; k < wordsPerLevel; k++) { oracle[r].siblings[q][0][k] <== 0; } }
            }
            oracle[r].last <== last[r];
            oracle[r].root <== roots[r];
            for (var m = 0; m <= STIR_NUM_QUERIES[r - 1]; m++) { oracle[r].ansPrev[m] <== hints[hp + m]; }
            oracle[r].uPrev <== quot[r - 1].uCoefs;
            oracle[r].dBits <== quot[r - 1].dBits;
            oracle[r].expBits <== quot[r - 1].expBits;
            oracle[r].rComb <== ldt[ldt_rComb(r - 1)];
            oracle[r].rOut <== ldt[ldt_rOut(r - 1)];
            oracle[r].enable <== enable;
        }

        fold[r] = StirFoldQueries(t, logL, logK);
        // VERIFY-CONSTRAINTS: round 0 folds here from copies of T_0's cosets (today inside the query batches).
        if (r == 0) {
            for (var q = 0; q < t; q++) {
                for (var c = 0; c < K; c++) { fold[r].vals[q][c] <== vals[vo + q * K + c]; }
            }
        } else {
            fold[r].vals <== oracle[r].fx;
        }
        fold[r].leafBits <== pts[r].leafBits;
        fold[r].rFold <== ldt[ldt_rFold(1, r)];

        if (r < M - 1) {
            var hq = stir_hintOff(STIR_NUM_QUERIES, r);
            quot[r] = StirQuotient(t);
            quot[r].gpt <== pts[r].gpt;
            quot[r].foldVals <== fold[r].out;
            for (var m = 0; m <= t; m++) {
                quot[r].ans[m] <== hints[hq + m];
                quot[r].shake[m] <== hints[hq + t + 1 + m];
            }
            quot[r].rho <== ldt[ldt_rho(r)];
            quot[r].rOut <== ldt[ldt_rOut(r)];
            quot[r].beta <== betas[r];
            quot[r].enable <== enable;
        } else {
            fin[0] = StirFinalCheck(t, 1 << stirFinalBits);
            fin[0].gpt <== pts[r].gpt;
            fin[0].foldVals <== fold[r].out;
            fin[0].finalPol <== finalPol;
            fin[0].enable <== enable;
        }

        // T_0's levels sent in clear are checked with the L_0 trees (fri.circom).
        if (LL > 0 && r > 0) {
            VerifyMerkleRoot(LL, arity, 1 << (logL - logK))(last[r], roots[r], enable);
        }
    }
}
