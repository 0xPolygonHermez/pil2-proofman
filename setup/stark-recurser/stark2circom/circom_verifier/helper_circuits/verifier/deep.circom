pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "fft.circom";
include "proof_shape.circom";
include "query_point.circom";

/*
    The DEEP polynomial f_0, grouped as the native provers and verifiers compute it (deep.rs's plan,
    same operations in the same order):

        f_0(x) = sum_G S_G * (sum_{o in G} A_o) - sum_o A_o * K_o,   A_o = vf1^{e_o} / (x - g^o z)

    o runs over the openings with evaluations (rank r), e_o counts the later ones, G over the column
    groups (columns opened at exactly the same openings), S_G = sum_{c in G} vf2^c * p_c(x) and
    K_o = sum_{c in o} vf2^c * eval(c, o), c being a column's first appearance in evMap.

    friConsts = [vf1, vf2, vf2^0..vf2^{nPols-1}, vf1^{e_o} per opening, K_o per opening]. The grouping
    arrives precomputed (lib_params.rs): OPENING_NUM_EVALS / EVAL_COLUMN per opening, SUMS and COLS per column group.
*/

// The query-independent DEEP terms, once per proof.
template DeepConstants(nOpenings, nPols, nEvals, OPENING_NUM_EVALS, EVAL_COLUMN) {
    var n = nOpenings;

    signal input vf[2][3];
    signal input evals[nEvals][3];
    signal output friConsts[2 + nPols + 2 * n][3];

    // vf2^c, vf1^r: [1, v, v*v, ...], one CMul per power from the square on.
    signal vf2Pow[nPols][3];
    signal vf1Pow[n][3];
    var p2[nPols][3];
    var p1[n][3];
    for (var c = 0; c < nPols; c++) {
        if (c == 1) { for (var e = 0; e < 3; e++) { p2[c][e] = vf[1][e]; } }
        if (c > 1) {
            vf2Pow[c] <== CMul()(p2[c - 1], vf[1]);
            for (var e = 0; e < 3; e++) { p2[c][e] = vf2Pow[c][e]; }
        }
        if (c == 0) { p2[c][0] = 1; }
    }
    for (var r = 0; r < n; r++) {
        if (r == 1) { for (var e = 0; e < 3; e++) { p1[r][e] = vf[0][e]; } }
        if (r > 1) {
            vf1Pow[r] <== CMul()(p1[r - 1], vf[0]);
            for (var e = 0; e < 3; e++) { p1[r][e] = vf1Pow[r][e]; }
        }
        if (r == 0) { p1[r][0] = 1; }
    }

    // K_r: the evaluations of opening r, vf2-folded left to right.
    signal kMul[nEvals][3];
    signal kSum[nEvals][3];
    var kVal[n][3];
    var k0 = 0;
    for (var r = 0; r < n; r++) {
        var acc[3];
        for (var i = 0; i < OPENING_NUM_EVALS[r]; i++) {
            var k = k0 + i;
            var c = EVAL_COLUMN[k];
            var t[3];
            if (c == 0) {
                for (var e = 0; e < 3; e++) { t[e] = evals[k][e]; }
            } else {
                kMul[k] <== CMul()(p2[c], evals[k]);
                for (var e = 0; e < 3; e++) { t[e] = kMul[k][e]; }
            }
            if (i == 0) {
                for (var e = 0; e < 3; e++) { acc[e] = t[e]; }
            } else {
                kSum[k] <== [acc[0] + t[0], acc[1] + t[1], acc[2] + t[2]];
                for (var e = 0; e < 3; e++) { acc[e] = kSum[k][e]; }
            }
        }
        for (var e = 0; e < 3; e++) { kVal[r][e] = acc[e]; }
        k0 += OPENING_NUM_EVALS[r];
    }

    friConsts[0] <== vf[0];
    friConsts[1] <== vf[1];
    for (var c = 0; c < nPols; c++) { friConsts[2 + c] <== p2[c]; }
    for (var r = 0; r < n; r++) {
        friConsts[2 + nPols + r] <== p1[n - 1 - r];
        friConsts[2 + nPols + n + r] <== kVal[r];
    }
}

/*
    f_0 at one query x = shift * w^i of L_0 (i from its LSB-first bits), from that row's opened leaves
    (s0_vals, already checked against their Merkle roots), matched against the member of fold tree
    1's opened coset that x corresponds to (selected by the query bits above nextStepBits).
*/
template DeepQuery(nOpenings, nPols, nColGroups, nSums, bits0, nextBits, s0Width, OPENING_ROOT, SUMS, COLS) {
    var n = nOpenings;
    var nextStep = bits0 - nextBits;

    signal input {binary} queriesL0[bits0];
    signal input xi[3];
    signal input friConsts[2 + nPols + 2 * n][3];
    signal input s0_vals[s0Width];
    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    // The query's point of L_0.
    signal x <== QueryPoint(bits0, bits0, domainShift(), 0, bits0)(queriesL0);

    // 1 / (x - g^o z) per opening with evaluations.
    signal xDivXSubXi[n][3];
    for (var r = 0; r < n; r++) {
        var root = OPENING_ROOT[r];
        xDivXSubXi[r] <== CInv()([x - root * xi[0], - root * xi[1], - root * xi[2]]);
    }

    // A_r = vf1^{e_r} * xDivXSubXi[r] (e = 0 for the last opening).
    signal aMul[n][3];
    var a[n][3];
    for (var r = 0; r < n; r++) {
        if (r == n - 1) {
            for (var e = 0; e < 3; e++) { a[r][e] = xDivXSubXi[r][e]; }
        } else {
            aMul[r] <== CMul()(friConsts[2 + nPols + r], xDivXSubXi[r]);
            for (var e = 0; e < 3; e++) { a[r][e] = aMul[r][e]; }
        }
    }

    // Shared prefix sums of A: SUMS[k] = (source, rank), a source s < n being a[s], else aSum[s - n].
    signal aSum[nonEmpty(nSums)][3];
    for (var k = 0; k < nSums; k++) {
        var src = SUMS[2 * k];
        var rj = SUMS[2 * k + 1];
        var prev[3];
        if (src < n) {
            for (var e = 0; e < 3; e++) { prev[e] = a[src][e]; }
        } else {
            for (var e = 0; e < 3; e++) { prev[e] = aSum[src - n][e]; }
        }
        aSum[k] <== [prev[0] + a[rj][0], prev[1] + a[rj][1], prev[2] + a[rj][2]];
    }

    // S_G * sum A per group: a base-field column 0 alone keeps S_G in the base field.
    // COLS per group: the source of its A sum, its column count, then (column, s0 offset, dim) per column.
    // VERIFY-CONSTRAINTS: leaves read straight from s0_vals (today through MapValues' signal copies).
    signal colMul[nPols][3];
    signal colSum[nPols][3];
    signal term[nColGroups][3];
    signal termSum[nColGroups][3];
    var res[3];
    var p = 0;
    for (var g = 0; g < nColGroups; g++) {
        var src = COLS[p];
        var nCols = COLS[p + 1];
        p += 2;
        var accA[3];
        if (src < n) {
            for (var e = 0; e < 3; e++) { accA[e] = a[src][e]; }
        } else {
            for (var e = 0; e < 3; e++) { accA[e] = aSum[src - n][e]; }
        }
        var s[3];
        var sDim = 3;
        for (var i = 0; i < nCols; i++) {
            var c = COLS[p];
            var off = COLS[p + 1];
            var dim = COLS[p + 2];
            p += 3;
            var t[3];
            var tDim = 3;
            if (c == 0) {
                t[0] = s0_vals[off];
                if (dim == 1) {
                    t[1] = 0;
                    t[2] = 0;
                    tDim = 1;
                } else {
                    t[1] = s0_vals[off + 1];
                    t[2] = s0_vals[off + 2];
                }
            } else {
                if (dim == 1) {
                    colMul[c] <== [friConsts[2 + c][0] * s0_vals[off], friConsts[2 + c][1] * s0_vals[off], friConsts[2 + c][2] * s0_vals[off]];
                } else {
                    colMul[c] <== CMul()(friConsts[2 + c], [s0_vals[off], s0_vals[off + 1], s0_vals[off + 2]]);
                }
                for (var e = 0; e < 3; e++) { t[e] = colMul[c][e]; }
            }
            if (i == 0) {
                for (var e = 0; e < 3; e++) { s[e] = t[e]; }
                sDim = tDim;
            } else {
                colSum[c] <== [s[0] + t[0], s[1] + t[1], s[2] + t[2]];
                for (var e = 0; e < 3; e++) { s[e] = colSum[c][e]; }
                sDim = 3;
            }
        }
        if (sDim == 1) {
            term[g] <== [s[0] * accA[0], s[0] * accA[1], s[0] * accA[2]];
        } else {
            term[g] <== CMul()(s, accA);
        }
        if (g == 0) {
            for (var e = 0; e < 3; e++) { res[e] = term[g][e]; }
        } else {
            termSum[g] <== [res[0] + term[g][0], res[1] + term[g][1], res[2] + term[g][2]];
            for (var e = 0; e < 3; e++) { res[e] = termSum[g][e]; }
        }
    }

    // - sum_r A_r * K_r
    signal kMul[n][3];
    signal kSub[n][3];
    for (var r = 0; r < n; r++) {
        kMul[r] <== CMul()(a[r], friConsts[2 + nPols + n + r]);
        kSub[r] <== [res[0] - kMul[r][0], res[1] - kMul[r][1], res[2] - kMul[r][2]];
        for (var e = 0; e < 3; e++) { res[e] = kSub[r][e]; }
    }

    signal {binary} keysLow[nextStep];
    for (var i = 0; i < nextStep; i++) { keysLow[i] <== queriesL0[i + nextBits]; }
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s1_vals, keysLow);

    enable * (lowValues[0] - res[0]) === 0;
    enable * (lowValues[1] - res[1]) === 0;
    enable * (lowValues[2] - res[2]) === 0;
}
