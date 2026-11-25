pragma circom 2.1.0;
pragma custom_templates;

include "geometry.circom";
include "transcript.circom";

/*
    The BN128 STARK verifier: Goldilocks emulated over BN128 with lazily reduced `{maxNum}` signals
    (circuits.bn128/gl.circom or custom/gl.circom, whichever the generated verifier includes). Every
    reduction (GLNorm / GLCNorm, the GL* quotients) is where the Tera verifier had it, so the circuit
    is the same. The generated `StarkVerifier` declares the key's data as vars and wires:

        StarkVerifierTranscript   shared with the GL verifier (transcript.circom)
        ZhBN128, ConstraintEval (generated, everyRow roots only), QuotientCheckBN128
        DeepConstantsBN128
        VerifySingleQuery (generated): per query the Merkle openings, DeepQueryBN128, VerifyFRIBN128
        VerifyFinalPolBN128
*/

template GLConst(num) {
    signal output {maxNum} out;

    out.maxNum = num;
    out <== num;
}

template GLConst3(num) {
    signal output {maxNum} out[3];

    out.maxNum = num;
    out <== [num, 0, 0];
}

template GLC3() {
    signal input {maxNum} in;
    signal output {maxNum} out[3];

    out.maxNum = in.maxNum;
    out <== [in, 0, 0];
}

// Goldilocks words in canonical range.
template GLVals(n) {
    signal input in[n];
    signal output {maxNum} out[n];

    out.maxNum = 0xFFFFFFFFFFFFFFFF;
    for (var i = 0; i < n; i++) { out[i] <== LessThan64Bits()(in[i]); }
}

template GLVals3(n) {
    signal input in[n][3];
    signal output {maxNum} out[n][3];

    out.maxNum = 0xFFFFFFFFFFFFFFFF;
    for (var i = 0; i < n; i++) {
        for (var e = 0; e < 3; e++) { out[i][e] <== LessThan64Bits()(in[i][e]); }
    }
}

// Goldilocks power and inverse, for constants computed at compile time.
// 1 / shift^(2^(nBitsExt - prevBits)) (and times w[prevBits] for e1): the fold step's coset offsets.
function bn_e0(nBitsExt, prevBits) {
    var p = 0xFFFFFFFF00000001;
    var e = geo_shift();
    for (var i = 0; i < nBitsExt - prevBits; i++) { e = (e * e) % p; }
    return _inv1(e);
}

function bn_e1(nBitsExt, prevBits) {
    var p = 0xFFFFFFFF00000001;
    var e = geo_shift();
    for (var i = 0; i < nBitsExt - prevBits; i++) { e = (e * e) % p; }
    return _inv1((e * roots(prevBits)) % p);
}

// xi^(2^(i+1)) up to xi^N, and Zh = 1 / (xi^N - 1).
template ZhBN128(nBits) {
    var q = 0xFFFFFFFFFFFFFFFF;

    signal input {maxNum} challengeOut[3];
    signal output {maxNum} Zh[3];
    signal output {maxNum} xN[3];

    signal {maxNum} zMul[nBits][3];
    zMul.maxNum = q;
    for (var i = 0; i < nBits; i++) {
        if (i == 0) {
            zMul[i] <== GLCMul()(challengeOut, challengeOut);
        } else {
            zMul[i] <== GLCMul()(zMul[i - 1], zMul[i - 1]);
        }
    }

    signal {maxNum} oneGL[3];
    oneGL.maxNum = 1;
    oneGL <== [1, 0, 0];

    signal {maxNum} Z[3] <== GLCSub()(zMul[nBits - 1], oneGL);
    Zh <== GLCInv()(Z);
    xN <== zMul[nBits - 1];
}

// C(xi) (the generated ConstraintEval, already times its zerofier) equals Q(xi) = sum_i xi^(i N) Q_i(xi).
template QuotientCheckBN128(qDeg, qPolEv, nEvals) {
    var q = 0xFFFFFFFFFFFFFFFF;

    signal input {maxNum} xN[3];
    signal input {maxNum} evalP[3];
    signal input {maxNum} evals[nEvals][3];
    signal input {binary} enable;

    signal {maxNum} xAcc[qDeg][3];
    signal {maxNum} qStep[geo_len1(qDeg - 1)][3];
    signal {maxNum} qAcc[qDeg][3];

    qStep.maxNum = q;
    qAcc.maxNum = qDeg * q;
    xAcc.maxNum = q;

    for (var i = 0; i < qDeg; i++) {
        if (i == 0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[qPolEv];
        } else {
            xAcc[i] <== GLCMul()(xAcc[i - 1], xN);
            qStep[i - 1] <== GLCMul()(xAcc[i], evals[qPolEv + i]);
            qAcc[i] <== GLCAdd()(qAcc[i - 1], qStep[i - 1]);
        }
    }

    signal {maxNum} vals[3] <== GLCSub()(evalP, qAcc[qDeg - 1]);
    signal {maxNum} normC[3] <== GLCNorm()(vals);

    enable * normC[0] === 0;
    enable * normC[1] === 0;
    enable * normC[2] === 0;
}

/*
    The query-independent DEEP terms (deep.circom's DeepConstants, emulated): friConsts =
    [vf1, vf2, vf2^0..vf2^{nPols-1}, vf1^{e_o} per opening, K_o per opening], each reduced.
*/
template DeepConstantsBN128(nGroups, nPols, nEvals, GROUP_LEN, REF_COL) {
    var n = nGroups;
    var q = 0xFFFFFFFFFFFFFFFF;

    signal input {maxNum} vf[2][3];
    signal input {maxNum} evals[nEvals][3];
    signal output {maxNum} friConsts[2 + nPols + 2 * n][3];

    // vf2^c and vf1^r from the square on.
    component p2[nPols];
    component p1[n];
    for (var c = 2; c < nPols; c++) {
        p2[c] = GLCMul();
        if (c == 2) { p2[c].ina <== vf[1]; } else { p2[c].ina <== p2[c - 1].out; }
        p2[c].inb <== vf[1];
    }
    for (var r = 2; r < n; r++) {
        p1[r] = GLCMul();
        if (r == 2) { p1[r].ina <== vf[0]; } else { p1[r].ina <== p1[r - 1].out; }
        p1[r].inb <== vf[0];
    }

    // K_r: the evaluations of opening r, vf2-folded left to right (its products, then its sums).
    component kMul[nEvals];
    component kSum[nEvals];
    var k0 = 0;
    for (var r = 0; r < n; r++) {
        for (var i = 0; i < GROUP_LEN[r]; i++) {
            var k = k0 + i;
            var c = REF_COL[k];
            if (c > 0) {
                kMul[k] = GLCMul();
                if (c == 1) { kMul[k].ina <== vf[1]; } else { kMul[k].ina <== p2[c].out; }
                kMul[k].inb <== evals[k];
            }
        }
        for (var i = 1; i < GROUP_LEN[r]; i++) {
            var k = k0 + i;
            var c = REF_COL[k];
            kSum[k] = GLCAdd();
            if (i > 1) {
                kSum[k].ina <== kSum[k - 1].out;
            } else if (REF_COL[k - 1] == 0) {
                kSum[k].ina <== evals[k - 1];
            } else {
                kSum[k].ina <== kMul[k - 1].out;
            }
            if (c == 0) { kSum[k].inb <== evals[k]; } else { kSum[k].inb <== kMul[k].out; }
        }
        k0 += GROUP_LEN[r];
    }

    // A sum exceeds q and is reduced; a product or an input is already below it.
    friConsts.maxNum = q;
    friConsts[0] <== vf[0];
    friConsts[1] <== vf[1];
    for (var c = 0; c < nPols; c++) {
        if (c == 0) { friConsts[2] <== [1, 0, 0]; }
        if (c == 1) { friConsts[3] <== vf[1]; }
        if (c > 1) { friConsts[2 + c] <== p2[c].out; }
    }
    k0 = 0;
    for (var r = 0; r < n; r++) {
        var e = n - 1 - r;
        if (e == 0) { friConsts[2 + nPols + r] <== [1, 0, 0]; }
        if (e == 1) { friConsts[2 + nPols + r] <== vf[0]; }
        if (e > 1) { friConsts[2 + nPols + r] <== p1[e].out; }
        var kl = k0 + GROUP_LEN[r] - 1;
        if (GROUP_LEN[r] > 1) {
            friConsts[2 + nPols + n + r] <== GLCNorm()(kSum[kl].out);
        } else if (REF_COL[kl] == 0) {
            friConsts[2 + nPols + n + r] <== evals[kl];
        } else {
            friConsts[2 + nPols + n + r] <== kMul[kl].out;
        }
        k0 += GROUP_LEN[r];
    }
}

/*
    f_0 at one query x of L_0 against fold tree 1's opened coset (deep.circom's DeepQuery, emulated):
    x and every x - g^o xi from repeated GLMul as the Tera verifier did, the same grouped sum
    (lib_params.rs: OPEN, SUMS, COLS), a base-field leaf lifted with GLC3 where it meets an extension.
*/
template DeepQueryBN128(nGroups, nPols, nColGroups, nSums, bits0, nextBits, s0Width, nBits, OPEN, SUMS, COLS) {
    var n = nGroups;
    var nextStep = bits0 - nextBits;
    var q = 0xFFFFFFFFFFFFFFFF;

    signal input {binary} queriesL0[bits0];
    signal input {maxNum} challengeOut[3];
    signal input {maxNum} friConsts[2 + nPols + 2 * n][3];
    signal input {maxNum} leaves[s0Width];
    signal input {maxNum} s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    // x = shift * w^i from the query bits.
    signal {maxNum} xacc[bits0 - 1];
    xacc.maxNum = q;
    signal {maxNum} ina;
    signal {maxNum} inb;
    ina.maxNum = q;
    inb.maxNum = q;
    signal {maxNum} inp[geo_len1(bits0 - 2)];
    inp.maxNum = q;
    ina <== queriesL0[0] * (geo_shift() * roots(bits0) - geo_shift()) + geo_shift();
    inb <== queriesL0[1] * (roots(bits0 - 1) - 1) + 1;
    xacc[0] <== GLMul()(ina, inb);
    for (var i = 2; i < bits0; i++) {
        inp[i - 2] <== queriesL0[i] * (roots(bits0 - i) - 1) + 1;
        xacc[i - 1] <== GLMul()(xacc[i - 2], inp[i - 2]);
    }
    signal {maxNum} X[3] <== GLC3()(xacc[bits0 - 2]);

    // g^o by |o| GLMul steps from 1, then 1 / (x - g^o xi), per opening with evaluations (rank r).
    var nW = 0;
    for (var r = 0; r < n; r++) { if (OPEN[r] != 0) { nW += (OPEN[r] < 0 ? -OPEN[r] : OPEN[r]) + 1; } }
    signal {maxNum} root <== GLConst(roots(nBits))();
    signal {maxNum} invroot <== GLConst(invroots(nBits))();
    signal {maxNum} w[geo_len1(nW)];
    w.maxNum = q;
    signal {maxNum} xDivXSubXi[n][3];
    xDivXSubXi.maxNum = q;
    component wC[n];
    component wMul[n];
    component wSub[n];
    var wo = 0;
    for (var r = 0; r < n; r++) {
        var o = OPEN[r];
        if (o != 0) {
            var a = o < 0 ? -o : o;
            w[wo] <== 1;
            for (var j = 0; j < a; j++) {
                if (o > 0) { w[wo + j + 1] <== GLMul()(root, w[wo + j]); } else { w[wo + j + 1] <== GLMul()(invroot, w[wo + j]); }
            }
            wC[r] = GLC3();
            wC[r].in <== w[wo + a];
            wMul[r] = GLCMul();
            wMul[r].ina <== challengeOut;
            wMul[r].inb <== wC[r].out;
            wo += a + 1;
        }
        wSub[r] = GLCSub();
        wSub[r].ina <== X;
        if (o == 0) { wSub[r].inb <== challengeOut; } else { wSub[r].inb <== wMul[r].out; }
        xDivXSubXi[r] <== GLCInv()(wSub[r].out);
    }

    // In deep.rs's op order (as the Tera verifier emitted it): A_r = vf1^{e_r} * xDivXSubXi[r] (e = 0
    // for the last opening); per column group its new prefix sums of A (SUMS), its columns' products,
    // their sum S_G and S_G * sum A; then the groups' sum.
    component aMul[n];
    for (var r = 0; r < n - 1; r++) {
        aMul[r] = GLCMul();
        aMul[r].ina <== friConsts[2 + nPols + r];
        aMul[r].inb <== xDivXSubXi[r];
    }
    // COLS per group: its A source, #columns, (column, leaf, dim)... A column's term is colMul[c] (c > 0),
    // else its leaf: an extension inline, a base one lifted (lift0). The running S_G is a column's term
    // (sKind 0: leaf, 1: lift0, 2: colMul[c]) or colSum[c] (3).
    component aSum[geo_len1(nSums)];
    component colLift[nPols];
    component colMul[nPols];
    component colSum[nPols];
    component lift0[1];
    component term[nColGroups];
    var kNext = 0;
    var p = 0;
    for (var g = 0; g < nColGroups; g++) {
        var src = COLS[p];
        var nCols = COLS[p + 1];
        for (var k = kNext; k <= src - n; k++) {
            var s0 = SUMS[2 * k];
            var rj = SUMS[2 * k + 1];
            aSum[k] = GLCAdd();
            if (s0 >= n) { aSum[k].ina <== aSum[s0 - n].out; } else if (s0 == n - 1) { aSum[k].ina <== xDivXSubXi[s0]; } else { aSum[k].ina <== aMul[s0].out; }
            if (rj == n - 1) { aSum[k].inb <== xDivXSubXi[rj]; } else { aSum[k].inb <== aMul[rj].out; }
            kNext = k + 1;
        }
        for (var i = 0; i < nCols; i++) {
            var c = COLS[p + 2 + 3 * i];
            var off = COLS[p + 3 + 3 * i];
            var dim = COLS[p + 4 + 3 * i];
            if (c > 0) {
                colMul[c] = GLCMul();
                colMul[c].ina <== friConsts[2 + c];
                if (dim == 1) {
                    colLift[c] = GLC3();
                    colLift[c].in <== leaves[off];
                    colMul[c].inb <== colLift[c].out;
                } else {
                    colMul[c].inb <== [leaves[off], leaves[off + 1], leaves[off + 2]];
                }
            } else if (dim == 1) {
                lift0[0] = GLC3();
                lift0[0].in <== leaves[off];
            }
        }
        var sKind = 0;
        var sIdx = 0;
        for (var i = 0; i < nCols; i++) {
            var c = COLS[p + 2 + 3 * i];
            var off = COLS[p + 3 + 3 * i];
            var dim = COLS[p + 4 + 3 * i];
            var tKind = c > 0 ? 2 : (dim == 1 ? 1 : 0);
            var tIdx = c == 0 ? off : c;
            if (i == 0) {
                sKind = tKind;
                sIdx = tIdx;
            } else {
                colSum[c] = GLCAdd();
                if (sKind == 0) { colSum[c].ina <== [leaves[sIdx], leaves[sIdx + 1], leaves[sIdx + 2]]; }
                if (sKind == 1) { colSum[c].ina <== lift0[0].out; }
                if (sKind == 2) { colSum[c].ina <== colMul[sIdx].out; }
                if (sKind == 3) { colSum[c].ina <== colSum[sIdx].out; }
                if (tKind == 0) { colSum[c].inb <== [leaves[tIdx], leaves[tIdx + 1], leaves[tIdx + 2]]; }
                if (tKind == 1) { colSum[c].inb <== lift0[0].out; }
                if (tKind == 2) { colSum[c].inb <== colMul[c].out; }
                sKind = 3;
                sIdx = c;
            }
        }
        p += 2 + 3 * nCols;
        term[g] = GLCMul();
        if (sKind == 0) { term[g].ina <== [leaves[sIdx], leaves[sIdx + 1], leaves[sIdx + 2]]; }
        if (sKind == 1) { term[g].ina <== lift0[0].out; }
        if (sKind == 2) { term[g].ina <== colMul[sIdx].out; }
        if (sKind == 3) { term[g].ina <== colSum[sIdx].out; }
        if (src >= n) { term[g].inb <== aSum[src - n].out; } else if (src == n - 1) { term[g].inb <== xDivXSubXi[src]; } else { term[g].inb <== aMul[src].out; }
    }
    component termSum[nColGroups];
    for (var g = 1; g < nColGroups; g++) {
        termSum[g] = GLCAdd();
        if (g == 1) { termSum[g].ina <== term[0].out; } else { termSum[g].ina <== termSum[g - 1].out; }
        termSum[g].inb <== term[g].out;
    }

    // - sum_r A_r * K_r
    component kMul[n];
    component kSub[n];
    for (var r = 0; r < n; r++) {
        kMul[r] = GLCMul();
        if (r == n - 1) { kMul[r].ina <== xDivXSubXi[r]; } else { kMul[r].ina <== aMul[r].out; }
        kMul[r].inb <== friConsts[2 + nPols + n + r];
        kSub[r] = GLCSub();
        if (r > 0) {
            kSub[r].ina <== kSub[r - 1].out;
        } else if (nColGroups > 1) {
            kSub[r].ina <== termSum[nColGroups - 1].out;
        } else {
            kSub[r].ina <== term[0].out;
        }
        kSub[r].inb <== kMul[r].out;
    }

    // Reduced once on its own (as the Tera verifier did), then against the next oracle's member.
    signal queryVals[3] <== GLCNorm()(kSub[n - 1].out);

    signal {binary} keysLow[nextStep];
    for (var i = 0; i < nextStep; i++) { keysLow[i] <== queriesL0[i + nextBits]; }
    signal {maxNum} lowValues[3] <== TreeSelector(nextStep, 3)(s1_vals, keysLow);

    signal {maxNum} valsDiff[3] <== GLCSub()(kSub[n - 1].out, lowValues);
    signal {maxNum} normDiff[3] <== GLCNorm()(valsDiff);

    enable * normDiff[0] === 0;
    enable * normDiff[1] === 0;
    enable * normDiff[2] === 0;
}

// One FRI folding step (fri.circom's VerifyFRIFold, emulated); FFTFold is the family's (fft.circom).
template VerifyFRIBN128(prevStepBits, currStepBits, nextStepBits, nBitsExt, nIn) {
    var nextStep = currStepBits - nextStepBits;
    var step = prevStepBits - currStepBits;
    var q = 0xFFFFFFFFFFFFFFFF;
    var e0 = bn_e0(nBitsExt, prevStepBits);
    var e1 = bn_e1(nBitsExt, prevStepBits);

    signal input {binary} queriesL0[nIn];
    signal input {maxNum} challengeFRIStep[3];
    signal input {maxNum} s_vals_curr[1 << step][3];
    signal input {maxNum} s_vals_next[1 << nextStep][3];
    signal input {binary} enable;

    signal {maxNum} s_X[3];
    if (currStepBits > 1) {
        signal {maxNum} s_sx[currStepBits - 1];
        s_sx.maxNum = q;
        signal {maxNum} inp[geo_len1(currStepBits - 2)];
        inp.maxNum = q;
        signal {maxNum} ina;
        signal {maxNum} inb;
        ina.maxNum = q;
        inb.maxNum = q;
        ina <== queriesL0[0] * (e1 - e0) + e0;
        inb <== queriesL0[1] * (invroots(prevStepBits - 1) - 1) + 1;
        s_sx[0] <== GLMul()(ina, inb);
        for (var i = 1; i < currStepBits - 1; i++) {
            inp[i - 1] <== queriesL0[i + 1] * (invroots(prevStepBits - (i + 1)) - 1) + 1;
            s_sx[i] <== GLMul()(s_sx[i - 1], inp[i - 1]);
        }
        s_X.maxNum = s_sx.maxNum;
        s_X <== [s_sx[currStepBits - 2], 0, 0];
    } else {
        signal {maxNum} inp1 <== GLConst(e0)();
        signal {maxNum} inp2;
        inp2.maxNum = q;
        inp2 <== queriesL0[0] * (invroots(prevStepBits) - 1) + 1;
        signal {maxNum} mul <== GLMul()(inp1, inp2);
        s_X.maxNum = mul.maxNum;
        s_X <== [mul, 0, 0];
    }

    signal {maxNum} coefs[1 << step][3] <== FFTFold(step)(s_vals_curr);
    signal {maxNum} evalXprime[3] <== GLCMul()(challengeFRIStep, s_X);
    signal {maxNum} evalPol[3] <== EvalPol(1 << step)(coefs, evalXprime);

    signal {binary} s_keys_lowValues[nextStep];
    for (var i = 0; i < nextStep; i++) { s_keys_lowValues[i] <== queriesL0[i + nextStepBits]; }
    signal {maxNum} lowValues[3] <== TreeSelector(nextStep, 3)(s_vals_next, s_keys_lowValues);

    signal {maxNum} vals[3] <== GLCSub()(evalPol, lowValues);
    signal {maxNum} normC[3] <== GLCNorm()(vals);

    enable * normC[0] === 0;
    enable * normC[1] === 0;
    enable * normC[2] === 0;
}

// finalPol is sent as evaluations: its coefficients above the degree bound must vanish.
template VerifyFinalPolBN128(nLastBits, maxDegSize) {
    signal input {maxNum} finalPol[1 << nLastBits][3];
    signal input {binary} enable;

    signal lastIFFT[1 << nLastBits][3] <== FFT(nLastBits, 1)(finalPol);

    for (var k = maxDegSize; k < 1 << nLastBits; k++) {
        for (var e = 0; e < 3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    for (var k = 0; k < maxDegSize; k++) {
        _ <== lastIFFT[k];
    }
}
