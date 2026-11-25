pragma circom 2.1.0;
pragma custom_templates;

include "geometry.circom";
include "transcript.circom";

/*
    The BN128 STARK verifier: Goldilocks emulated over BN128 with lazily reduced `{maxNum}` signals
    (circuits.bn128/gl.circom or custom/gl.circom, whichever the generated verifier includes). Every
    reduction (GLNorm / GLCNorm, the GL* quotients) is where the Tera verifier had it, so the circuit
    is the same. StarkVerifierBN128(k) (below) wires:

        StarkVerifierTranscript   shared with the GL verifier (transcript.circom)
        ZhBN128, ConstraintEval (the key's, everyRow roots only), QuotientCheckBN128
        DeepConstantsBN128
        VerifySingleQueryBN128: per query the Merkle openings (BN128Open, bn128_open_<family>.circom),
                                DeepQueryBN128, VerifyFRIBN128
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

/*
    One BN128-verified STARK proof of the compilation's key k (svd_*(k), lib_params.rs) from its flat
    proof; the family half (bn128_open_<family>.circom, picked by the generated verifier's include) opens
    the trees, the key's ConstraintEval checks the constraints.
*/
template StarkVerifierBN128(k) {
    var q = 0xFFFFFFFFFFFFFFFF;
    var nBits = svd_nBits(k);
    var nBitsExt = svd_nBitsExt(k);
    var arity = svd_arity(k);
    var lastLevels = svd_lastLevels(k);
    var nQueries = svd_nQueries(k);
    var nEvals = svd_nEvals(k);
    var nPublics = svd_nPublics(k);
    var nStages = svd_nStages(k);
    var nChallenges = svd_nChallenges(k);
    var nSteps = svd_nSteps(k);
    var nTrees = svd_nTrees(k);
    var nGroups = svd_nGroups(k);
    var nPols = svd_nPols(k);
    var verkeyInput = svd_verkeyInput(k);
    var enableInput = svd_enableInput(k);
    var offEvals = svd_offEvals(k);
    var offFinal = svd_offFinal(k);
    var offNonce = svd_offNonce(k);
    var STEPS[nSteps] = svd_STEPS(k);
    var S0[nTrees + 1] = svd_S0(k);
    var ROOT_C[4] = svd_ROOT_C(k);
    var OFF_ROOT[nStages + 1] = svd_OFF_ROOT(k);
    var OFF_S0V[nTrees] = svd_OFF_S0V(k);
    var OFF_S0S[nTrees] = svd_OFF_S0S(k);
    var OFF_S0L[nTrees] = svd_OFF_S0L(k);
    var TREE_ROOT[nTrees] = svd_TREE_ROOT(k);
    var NF = nSteps - 1;
    var OFF_FROOT[geo_len1(NF)] = svd_OFF_FROOT(k);
    var OFF_FV[geo_len1(NF)] = svd_OFF_FV(k);
    var OFF_FS[geo_len1(NF)] = svd_OFF_FS(k);
    var OFF_FL[geo_len1(NF)] = svd_OFF_FL(k);
    var rw = bn_rootW();
    var sw = bn_sibW(arity);
    var L0 = geo_levels0(arity, lastLevels, STEPS);
    var lastSize = geo_lastSize(arity, lastLevels);
    var FS = geo_finalSize(nSteps, STEPS);
    var zeroRoot[1][rw];
    for (var i = 0; i < rw; i++) { zeroRoot[0][i] = 0; }

    signal input proof[svd_proofSize(k)];
    signal input publics[nPublics];
    signal input rootCIn[rw * verkeyInput];
    signal input enable[enableInput];
    signal output rootC[rw];

    if (verkeyInput == 1) { rootC <== rootCIn; } else { for (var i = 0; i < rw; i++) { rootC[i] <== ROOT_C[i]; } }
    signal {binary} enabled;
    if (enableInput == 1) {
        enable[0] * (enable[0] - 1) === 0;
        enabled <== enable[0];
    } else {
        enabled <== 1;
    }

    signal evals[nEvals][3];
    for (var i = 0; i < nEvals; i++) { for (var e = 0; e < 3; e++) { evals[i][e] <== proof[offEvals + 3 * i + e]; } }
    signal finalPol[FS][3];
    for (var i = 0; i < FS; i++) { for (var e = 0; e < 3; e++) { finalPol[i][e] <== proof[offFinal + 3 * i + e]; } }
    signal stageRoot[nStages + 1][rw];
    for (var s = 0; s <= nStages; s++) { for (var i = 0; i < rw; i++) { stageRoot[s][i] <== proof[OFF_ROOT[s] + i]; } }
    signal treeRoot[nTrees][rw];
    signal treeLast[nTrees][lastSize][rw];
    for (var t = 0; t < nTrees; t++) {
        for (var i = 0; i < rw; i++) {
            if (TREE_ROOT[t] >= 0) { treeRoot[t][i] <== proof[TREE_ROOT[t] + i]; } else { treeRoot[t][i] <== rootC[i]; }
        }
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < rw; i++) {
                if (lastLevels > 0) { treeLast[t][j][i] <== proof[OFF_S0L[t] + rw * j + i]; } else { treeLast[t][j][i] <== 0; }
            }
        }
    }
    signal foldRoot[NF][rw];
    signal foldLast[NF][lastSize][rw];
    for (var f = 0; f < NF; f++) {
        for (var i = 0; i < rw; i++) { foldRoot[f][i] <== proof[OFF_FROOT[f] + i]; }
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < rw; i++) {
                if (lastLevels > 0) { foldLast[f][j][i] <== proof[OFF_FL[f] + rw * j + i]; } else { foldLast[f][j][i] <== 0; }
            }
        }
    }

    signal {maxNum} publicsGL[nPublics];
    if (nPublics > 0) { publicsGL <== GLVals(nPublics)(publics); }
    signal {maxNum} evalsGL[nEvals][3] <== GLVals3(nEvals)(evals);
    signal {maxNum} finalPolGL[FS][3] <== GLVals3(FS)(finalPol);

    // The GL verifier's transcript over this key's engine; its challenges reduced Goldilocks values.
    component transcript = StarkVerifierTranscript(nStages, nPublics, 0, 0, nEvals, nChallenges, nSteps, STEPS, 0, 0, 0, [0], svd_hashCommits(k), 0, nQueries, svd_powBits(k), svd_CH(k), [0], [0]);
    transcript.globalChallenge <== [0, 0, 0];
    if (nPublics > 0) { transcript.publics <== publicsGL; } else { transcript.publics <== [0]; }
    transcript.rootC <== rootC;
    transcript.stageRoots <== stageRoot;
    transcript.airvalues <== [[0, 0, 0]];
    transcript.airgroupvalues <== [[0, 0, 0]];
    transcript.evals <== evalsGL;
    if (NF > 0) { transcript.foldRoots <== foldRoot; } else { transcript.foldRoots <== zeroRoot; }
    transcript.betas <== [[0, 0, 0]];
    transcript.hints <== [[0, 0, 0]];
    transcript.finalPol <== finalPolGL;
    if (offNonce >= 0) { transcript.nonce <== proof[offNonce]; } else { transcript.nonce <== 0; }
    transcript.enable <== enabled;
    signal {maxNum} challenges[nChallenges][3];
    challenges.maxNum = q;
    challenges <== transcript.challenges;
    signal {maxNum} rFold[geo_len1(NF)][3];
    rFold.maxNum = q;
    if (NF > 0) {
        for (var f = 0; f < NF; f++) { rFold[f] <== transcript.ldt[f]; }
    } else {
        rFold[0] <== [0, 0, 0];
    }

    // The constraints at xi against the quotient; challenges end with vc, xi, vf1, vf2.
    component zh = ZhBN128(nBits);
    zh.challengeOut <== challenges[nChallenges - 3];
    component ce = ConstraintEval();
    ce.challenges <== challenges;
    ce.evals <== evalsGL;
    ce.Zh <== zh.Zh;
    ce.publics <== publicsGL;
    QuotientCheckBN128(svd_qDeg(k), svd_qPolEv(k), nEvals)(zh.xN, ce.evalP, evalsGL, enabled);

    component deepConsts = DeepConstantsBN128(nGroups, nPols, nEvals, svd_GROUP_LEN(k), svd_REF_COL(k));
    deepConsts.vf[0] <== challenges[nChallenges - 2];
    deepConsts.vf[1] <== challenges[nChallenges - 1];
    deepConsts.evals <== evalsGL;

    // Per query; the levels sent in clear once.
    component query[nQueries];
    for (var g = 0; g < nQueries; g++) {
        query[g] = VerifySingleQueryBN128(k);
        query[g].queriesL0 <== transcript.queriesL0[g];
        query[g].challengeOut <== challenges[nChallenges - 3];
        query[g].friConsts <== deepConsts.friConsts;
        query[g].rFold <== rFold;
        query[g].finalPol <== finalPolGL;
        query[g].enable <== enabled;
        query[g].last <== treeLast;
        query[g].root <== treeRoot;
        query[g].flast <== foldLast;
        query[g].froot <== foldRoot;
        for (var t = 0; t < nTrees; t++) { for (var l = 0; l < L0; l++) { for (var i = 0; i < sw; i++) { query[g].sibs[t][l][i] <== proof[OFF_S0S[t] + (g * L0 + l) * sw + i]; } } }

        for (var i = 0; i < geo_slotW(S0, nTrees, 0); i++) { query[g].v0[i] <== proof[OFF_S0V[0] + g * geo_slotW(S0, nTrees, 0) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 1); i++) { query[g].v1[i] <== proof[OFF_S0V[1] + g * geo_slotW(S0, nTrees, 1) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 2); i++) { query[g].v2[i] <== proof[OFF_S0V[2] + g * geo_slotW(S0, nTrees, 2) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 3); i++) { query[g].v3[i] <== proof[OFF_S0V[3] + g * geo_slotW(S0, nTrees, 3) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 4); i++) { query[g].v4[i] <== proof[OFF_S0V[4] + g * geo_slotW(S0, nTrees, 4) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 5); i++) { query[g].v5[i] <== proof[OFF_S0V[5] + g * geo_slotW(S0, nTrees, 5) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 6); i++) { query[g].v6[i] <== proof[OFF_S0V[6] + g * geo_slotW(S0, nTrees, 6) + i]; }
        for (var i = 0; i < geo_slotW(S0, nTrees, 7); i++) { query[g].v7[i] <== proof[OFF_S0V[7] + g * geo_slotW(S0, nTrees, 7) + i]; }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 0); i++) { query[g].fv0[i] <== proof[OFF_FV[0] + g * 3 * geo_slotC(STEPS, NF, 0) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 0); l++) { for (var i = 0; i < sw; i++) { query[g].fs0[l][i] <== proof[OFF_FS[0] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 0) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 1); i++) { query[g].fv1[i] <== proof[OFF_FV[1] + g * 3 * geo_slotC(STEPS, NF, 1) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 1); l++) { for (var i = 0; i < sw; i++) { query[g].fs1[l][i] <== proof[OFF_FS[1] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 1) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 2); i++) { query[g].fv2[i] <== proof[OFF_FV[2] + g * 3 * geo_slotC(STEPS, NF, 2) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 2); l++) { for (var i = 0; i < sw; i++) { query[g].fs2[l][i] <== proof[OFF_FS[2] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 2) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 3); i++) { query[g].fv3[i] <== proof[OFF_FV[3] + g * 3 * geo_slotC(STEPS, NF, 3) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 3); l++) { for (var i = 0; i < sw; i++) { query[g].fs3[l][i] <== proof[OFF_FS[3] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 3) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 4); i++) { query[g].fv4[i] <== proof[OFF_FV[4] + g * 3 * geo_slotC(STEPS, NF, 4) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 4); l++) { for (var i = 0; i < sw; i++) { query[g].fs4[l][i] <== proof[OFF_FS[4] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 4) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 5); i++) { query[g].fv5[i] <== proof[OFF_FV[5] + g * 3 * geo_slotC(STEPS, NF, 5) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 5); l++) { for (var i = 0; i < sw; i++) { query[g].fs5[l][i] <== proof[OFF_FS[5] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 5) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 6); i++) { query[g].fv6[i] <== proof[OFF_FV[6] + g * 3 * geo_slotC(STEPS, NF, 6) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 6); l++) { for (var i = 0; i < sw; i++) { query[g].fs6[l][i] <== proof[OFF_FS[6] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 6) + l) * sw + i]; } }
        for (var i = 0; i < 3 * geo_slotC(STEPS, NF, 7); i++) { query[g].fv7[i] <== proof[OFF_FV[7] + g * 3 * geo_slotC(STEPS, NF, 7) + i]; }
        for (var l = 0; l < geo_slotM(arity, lastLevels, STEPS, NF, 7); l++) { for (var i = 0; i < sw; i++) { query[g].fs7[l][i] <== proof[OFF_FS[7] + (g * geo_slotM(arity, lastLevels, STEPS, NF, 7) + l) * sw + i]; } }
    }
    if (lastLevels > 0) {
        for (var t = 0; t < nTrees; t++) { BN128RootCheck(lastLevels, arity, STEPS[0])(treeLast[t], treeRoot[t], enabled); }
        for (var f = 0; f < NF; f++) { BN128RootCheck(lastLevels, arity, STEPS[f + 1])(foldLast[f], foldRoot[f], enabled); }
    }
    VerifyFinalPolBN128(STEPS[nSteps - 1], geo_maxDegSize(nBits, nBitsExt, nSteps, STEPS))(finalPolGL, enabled);
}

// One query: its L_0 rows and fold-tree cosets against their trees, DEEP, every folding step.
template parallel VerifySingleQueryBN128(k) {
    var q = 0xFFFFFFFFFFFFFFFF;
    var nBits = svd_nBits(k);
    var nBitsExt = svd_nBitsExt(k);
    var arity = svd_arity(k);
    var lastLevels = svd_lastLevels(k);
    var nSteps = svd_nSteps(k);
    var nTrees = svd_nTrees(k);
    var nGroups = svd_nGroups(k);
    var nPols = svd_nPols(k);
    var STEPS[nSteps] = svd_STEPS(k);
    var S0[nTrees + 1] = svd_S0(k);
    var NF = nSteps - 1;
    var rw = bn_rootW();
    var sw = bn_sibW(arity);
    var L0 = geo_levels0(arity, lastLevels, STEPS);
    var lastSize = geo_lastSize(arity, lastLevels);
    assert(nTrees <= 8 && NF <= 8);
    var zeroSib[1][sw];
    for (var i = 0; i < sw; i++) { zeroSib[0][i] = 0; }

    signal input {binary} queriesL0[STEPS[0]];
    signal input {maxNum} challengeOut[3];
    signal input {maxNum} friConsts[2 + nPols + 2 * nGroups][3];
    signal input {maxNum} rFold[geo_len1(NF)][3];
    signal input {maxNum} finalPol[geo_finalSize(nSteps, STEPS)][3];
    signal input {binary} enable;
    signal input sibs[nTrees][L0][sw];
    signal input last[nTrees][lastSize][rw];
    signal input root[nTrees][rw];
    signal input flast[NF][lastSize][rw];
    signal input froot[NF][rw];

    signal input v0[geo_slotW(S0, nTrees, 0)];
    signal input v1[geo_slotW(S0, nTrees, 1)];
    signal input v2[geo_slotW(S0, nTrees, 2)];
    signal input v3[geo_slotW(S0, nTrees, 3)];
    signal input v4[geo_slotW(S0, nTrees, 4)];
    signal input v5[geo_slotW(S0, nTrees, 5)];
    signal input v6[geo_slotW(S0, nTrees, 6)];
    signal input v7[geo_slotW(S0, nTrees, 7)];
    signal input fv0[3 * geo_slotC(STEPS, NF, 0)];
    signal input fv1[3 * geo_slotC(STEPS, NF, 1)];
    signal input fv2[3 * geo_slotC(STEPS, NF, 2)];
    signal input fv3[3 * geo_slotC(STEPS, NF, 3)];
    signal input fv4[3 * geo_slotC(STEPS, NF, 4)];
    signal input fv5[3 * geo_slotC(STEPS, NF, 5)];
    signal input fv6[3 * geo_slotC(STEPS, NF, 6)];
    signal input fv7[3 * geo_slotC(STEPS, NF, 7)];
    signal input fs0[geo_slotM(arity, lastLevels, STEPS, NF, 0)][sw];
    signal input fs1[geo_slotM(arity, lastLevels, STEPS, NF, 1)][sw];
    signal input fs2[geo_slotM(arity, lastLevels, STEPS, NF, 2)][sw];
    signal input fs3[geo_slotM(arity, lastLevels, STEPS, NF, 3)][sw];
    signal input fs4[geo_slotM(arity, lastLevels, STEPS, NF, 4)][sw];
    signal input fs5[geo_slotM(arity, lastLevels, STEPS, NF, 5)][sw];
    signal input fs6[geo_slotM(arity, lastLevels, STEPS, NF, 6)][sw];
    signal input fs7[geo_slotM(arity, lastLevels, STEPS, NF, 7)][sw];

    // The L_0 rows, reduced, flat in tree order (S0); every row reduced before any tree is opened.
    signal {maxNum} leaves[S0[nTrees]];
    leaves.maxNum = q;
    for (var i = 0; i < geo_slotW(S0, nTrees, 0); i++) { leaves[S0[0] + i] <== LessThan64Bits()(v0[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 1); i++) { leaves[S0[1] + i] <== LessThan64Bits()(v1[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 2); i++) { leaves[S0[2] + i] <== LessThan64Bits()(v2[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 3); i++) { leaves[S0[3] + i] <== LessThan64Bits()(v3[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 4); i++) { leaves[S0[4] + i] <== LessThan64Bits()(v4[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 5); i++) { leaves[S0[5] + i] <== LessThan64Bits()(v5[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 6); i++) { leaves[S0[6] + i] <== LessThan64Bits()(v6[i]); }
    for (var i = 0; i < geo_slotW(S0, nTrees, 7); i++) { leaves[S0[7] + i] <== LessThan64Bits()(v7[i]); }
    if (nTrees > 0) {
        signal l0[S0[0 + 1] - S0[0]][1];
        for (var i = 0; i < S0[0 + 1] - S0[0]; i++) { l0[i][0] <== leaves[S0[0] + i]; }
        BN128Open(1, S0[0 + 1] - S0[0], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l0, sibs[0], last[0], root[0], enable);
    }
    if (nTrees > 1) {
        signal l1[S0[1 + 1] - S0[1]][1];
        for (var i = 0; i < S0[1 + 1] - S0[1]; i++) { l1[i][0] <== leaves[S0[1] + i]; }
        BN128Open(1, S0[1 + 1] - S0[1], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l1, sibs[1], last[1], root[1], enable);
    }
    if (nTrees > 2) {
        signal l2[S0[2 + 1] - S0[2]][1];
        for (var i = 0; i < S0[2 + 1] - S0[2]; i++) { l2[i][0] <== leaves[S0[2] + i]; }
        BN128Open(1, S0[2 + 1] - S0[2], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l2, sibs[2], last[2], root[2], enable);
    }
    if (nTrees > 3) {
        signal l3[S0[3 + 1] - S0[3]][1];
        for (var i = 0; i < S0[3 + 1] - S0[3]; i++) { l3[i][0] <== leaves[S0[3] + i]; }
        BN128Open(1, S0[3 + 1] - S0[3], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l3, sibs[3], last[3], root[3], enable);
    }
    if (nTrees > 4) {
        signal l4[S0[4 + 1] - S0[4]][1];
        for (var i = 0; i < S0[4 + 1] - S0[4]; i++) { l4[i][0] <== leaves[S0[4] + i]; }
        BN128Open(1, S0[4 + 1] - S0[4], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l4, sibs[4], last[4], root[4], enable);
    }
    if (nTrees > 5) {
        signal l5[S0[5 + 1] - S0[5]][1];
        for (var i = 0; i < S0[5 + 1] - S0[5]; i++) { l5[i][0] <== leaves[S0[5] + i]; }
        BN128Open(1, S0[5 + 1] - S0[5], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l5, sibs[5], last[5], root[5], enable);
    }
    if (nTrees > 6) {
        signal l6[S0[6 + 1] - S0[6]][1];
        for (var i = 0; i < S0[6 + 1] - S0[6]; i++) { l6[i][0] <== leaves[S0[6] + i]; }
        BN128Open(1, S0[6 + 1] - S0[6], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l6, sibs[6], last[6], root[6], enable);
    }
    if (nTrees > 7) {
        signal l7[S0[7 + 1] - S0[7]][1];
        for (var i = 0; i < S0[7 + 1] - S0[7]; i++) { l7[i][0] <== leaves[S0[7] + i]; }
        BN128Open(1, S0[7 + 1] - S0[7], arity, lastLevels, L0, STEPS[0], STEPS[0])(queriesL0, l7, sibs[7], last[7], root[7], enable);
    }

    // Fold tree f's opened coset, reduced; its Merkle path.
    signal {maxNum} fp0[geo_slotC(STEPS, NF, 0)][3];
    fp0.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 0); c++) { for (var e = 0; e < 3; e++) { fp0[c][e] <== LessThan64Bits()(fv0[c * 3 + e]); } }
    if (NF > 0) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 1);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 1), arity, lastLevels, ml, STEPS[1], STEPS[0])(queriesL0, fp0, fs0, flast[0], froot[0], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 1), arity, lastLevels, 0, STEPS[1], STEPS[0])(queriesL0, fp0, zeroSib, flast[0], froot[0], enable);
        }
    }
    signal {maxNum} fp1[geo_slotC(STEPS, NF, 1)][3];
    fp1.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 1); c++) { for (var e = 0; e < 3; e++) { fp1[c][e] <== LessThan64Bits()(fv1[c * 3 + e]); } }
    if (NF > 1) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 2);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 2), arity, lastLevels, ml, STEPS[2], STEPS[0])(queriesL0, fp1, fs1, flast[1], froot[1], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 2), arity, lastLevels, 0, STEPS[2], STEPS[0])(queriesL0, fp1, zeroSib, flast[1], froot[1], enable);
        }
    }
    signal {maxNum} fp2[geo_slotC(STEPS, NF, 2)][3];
    fp2.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 2); c++) { for (var e = 0; e < 3; e++) { fp2[c][e] <== LessThan64Bits()(fv2[c * 3 + e]); } }
    if (NF > 2) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 3);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 3), arity, lastLevels, ml, STEPS[3], STEPS[0])(queriesL0, fp2, fs2, flast[2], froot[2], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 3), arity, lastLevels, 0, STEPS[3], STEPS[0])(queriesL0, fp2, zeroSib, flast[2], froot[2], enable);
        }
    }
    signal {maxNum} fp3[geo_slotC(STEPS, NF, 3)][3];
    fp3.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 3); c++) { for (var e = 0; e < 3; e++) { fp3[c][e] <== LessThan64Bits()(fv3[c * 3 + e]); } }
    if (NF > 3) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 4);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 4), arity, lastLevels, ml, STEPS[4], STEPS[0])(queriesL0, fp3, fs3, flast[3], froot[3], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 4), arity, lastLevels, 0, STEPS[4], STEPS[0])(queriesL0, fp3, zeroSib, flast[3], froot[3], enable);
        }
    }
    signal {maxNum} fp4[geo_slotC(STEPS, NF, 4)][3];
    fp4.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 4); c++) { for (var e = 0; e < 3; e++) { fp4[c][e] <== LessThan64Bits()(fv4[c * 3 + e]); } }
    if (NF > 4) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 5);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 5), arity, lastLevels, ml, STEPS[5], STEPS[0])(queriesL0, fp4, fs4, flast[4], froot[4], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 5), arity, lastLevels, 0, STEPS[5], STEPS[0])(queriesL0, fp4, zeroSib, flast[4], froot[4], enable);
        }
    }
    signal {maxNum} fp5[geo_slotC(STEPS, NF, 5)][3];
    fp5.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 5); c++) { for (var e = 0; e < 3; e++) { fp5[c][e] <== LessThan64Bits()(fv5[c * 3 + e]); } }
    if (NF > 5) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 6);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 6), arity, lastLevels, ml, STEPS[6], STEPS[0])(queriesL0, fp5, fs5, flast[5], froot[5], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 6), arity, lastLevels, 0, STEPS[6], STEPS[0])(queriesL0, fp5, zeroSib, flast[5], froot[5], enable);
        }
    }
    signal {maxNum} fp6[geo_slotC(STEPS, NF, 6)][3];
    fp6.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 6); c++) { for (var e = 0; e < 3; e++) { fp6[c][e] <== LessThan64Bits()(fv6[c * 3 + e]); } }
    if (NF > 6) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 7);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 7), arity, lastLevels, ml, STEPS[7], STEPS[0])(queriesL0, fp6, fs6, flast[6], froot[6], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 7), arity, lastLevels, 0, STEPS[7], STEPS[0])(queriesL0, fp6, zeroSib, flast[6], froot[6], enable);
        }
    }
    signal {maxNum} fp7[geo_slotC(STEPS, NF, 7)][3];
    fp7.maxNum = q;
    for (var c = 0; c < geo_slotC(STEPS, NF, 7); c++) { for (var e = 0; e < 3; e++) { fp7[c][e] <== LessThan64Bits()(fv7[c * 3 + e]); } }
    if (NF > 7) {
        var ml = geo_foldLevels(arity, lastLevels, STEPS, 8);
        if (ml > 0) {
            BN128Open(3, geo_foldExp(STEPS, 8), arity, lastLevels, ml, STEPS[8], STEPS[0])(queriesL0, fp7, fs7, flast[7], froot[7], enable);
        } else {
            BN128Open(3, geo_foldExp(STEPS, 8), arity, lastLevels, 0, STEPS[8], STEPS[0])(queriesL0, fp7, zeroSib, flast[7], froot[7], enable);
        }
    }

    if (NF > 0) {
        DeepQueryBN128(nGroups, nPols, svd_nColGroups(k), svd_nSums(k), STEPS[0], STEPS[1], S0[nTrees], nBits, svd_OPEN(k), svd_SUMS(k), svd_COLS(k))(queriesL0, challengeOut, friConsts, leaves, fp0, enable);
    } else {
        DeepQueryBN128(nGroups, nPols, svd_nColGroups(k), svd_nSums(k), STEPS[0], 0, S0[nTrees], nBits, svd_OPEN(k), svd_SUMS(k), svd_COLS(k))(queriesL0, challengeOut, friConsts, leaves, finalPol, enable);
    }
    if (NF > 1) {
        VerifyFRIBN128(STEPS[0], STEPS[1], STEPS[2], nBitsExt, STEPS[0])(queriesL0, rFold[0], fp0, fp1, enable);
    } else if (NF == 1) {
        VerifyFRIBN128(STEPS[0], STEPS[1], 0, nBitsExt, STEPS[0])(queriesL0, rFold[0], fp0, finalPol, enable);
    }
    if (NF > 2) {
        VerifyFRIBN128(STEPS[1], STEPS[2], STEPS[3], nBitsExt, STEPS[0])(queriesL0, rFold[1], fp1, fp2, enable);
    } else if (NF == 2) {
        VerifyFRIBN128(STEPS[1], STEPS[2], 0, nBitsExt, STEPS[0])(queriesL0, rFold[1], fp1, finalPol, enable);
    }
    if (NF > 3) {
        VerifyFRIBN128(STEPS[2], STEPS[3], STEPS[4], nBitsExt, STEPS[0])(queriesL0, rFold[2], fp2, fp3, enable);
    } else if (NF == 3) {
        VerifyFRIBN128(STEPS[2], STEPS[3], 0, nBitsExt, STEPS[0])(queriesL0, rFold[2], fp2, finalPol, enable);
    }
    if (NF > 4) {
        VerifyFRIBN128(STEPS[3], STEPS[4], STEPS[5], nBitsExt, STEPS[0])(queriesL0, rFold[3], fp3, fp4, enable);
    } else if (NF == 4) {
        VerifyFRIBN128(STEPS[3], STEPS[4], 0, nBitsExt, STEPS[0])(queriesL0, rFold[3], fp3, finalPol, enable);
    }
    if (NF > 5) {
        VerifyFRIBN128(STEPS[4], STEPS[5], STEPS[6], nBitsExt, STEPS[0])(queriesL0, rFold[4], fp4, fp5, enable);
    } else if (NF == 5) {
        VerifyFRIBN128(STEPS[4], STEPS[5], 0, nBitsExt, STEPS[0])(queriesL0, rFold[4], fp4, finalPol, enable);
    }
    if (NF > 6) {
        VerifyFRIBN128(STEPS[5], STEPS[6], STEPS[7], nBitsExt, STEPS[0])(queriesL0, rFold[5], fp5, fp6, enable);
    } else if (NF == 6) {
        VerifyFRIBN128(STEPS[5], STEPS[6], 0, nBitsExt, STEPS[0])(queriesL0, rFold[5], fp5, finalPol, enable);
    }
    if (NF > 7) {
        VerifyFRIBN128(STEPS[6], STEPS[7], STEPS[8], nBitsExt, STEPS[0])(queriesL0, rFold[6], fp6, fp7, enable);
    } else if (NF == 7) {
        VerifyFRIBN128(STEPS[6], STEPS[7], 0, nBitsExt, STEPS[0])(queriesL0, rFold[6], fp6, finalPol, enable);
    }
    if (NF > 8) {
        VerifyFRIBN128(STEPS[7], STEPS[8], STEPS[9], nBitsExt, STEPS[0])(queriesL0, rFold[7], fp7, finalPol, enable);
    } else if (NF == 8) {
        VerifyFRIBN128(STEPS[7], STEPS[8], 0, nBitsExt, STEPS[0])(queriesL0, rFold[7], fp7, finalPol, enable);
    }
}
