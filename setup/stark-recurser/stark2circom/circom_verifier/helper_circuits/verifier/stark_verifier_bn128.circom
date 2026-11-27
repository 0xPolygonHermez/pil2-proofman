pragma circom 2.1.0;
pragma custom_templates;

include "proof_shape.circom";
include "proof_rows.circom";
include "transcript.circom";

/*
    The BN128 STARK verifier: Goldilocks emulated over BN128 with lazily reduced `{maxNum}` signals
    (circuits.bn128/gl.circom, with the range checks of the family bn128_open_<family>.circom includes). Every
    reduction (GLNorm / GLCNorm, the GL* quotients) is where the former templated verifier had it, so the circuit
    is the same. StarkVerifierBN128(k) (below) wires:

        StarkVerifierTranscript   shared with the GL verifier (transcript.circom)
        ZhBN128, ConstraintEval (the key's, everyRow roots only), QuotientCheckBN128
        DeepConstantsBN128
        VerifySingleQueryBN128: per query the Merkle openings (BN128Open, bn128_open_<family>.circom),
                                DeepQueryBN128, VerifyFRIBN128
        VerifyFinalPolBN128
*/

// A constant's bound is at least 1: the GL ops size their quotients from `maxNum - 1`, which a bound
// of 0 wraps in the BN field (GLSub(x, 0) then lands off x mod GL).
function gl_constBound(num) { return num == 0 ? 1 : num; }

template GLConst(num) {
    signal output {maxNum} out;

    out.maxNum = gl_constBound(num);
    out <== num;
}

template GLConst3(num) {
    signal output {maxNum} out[3];

    out.maxNum = gl_constBound(num);
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

    out.maxNum = max64();
    for (var i = 0; i < n; i++) { out[i] <== LessThan64Bits()(in[i]); }
}

template GLVals3(n) {
    signal input in[n][3];
    signal output {maxNum} out[n][3];

    out.maxNum = max64();
    for (var i = 0; i < n; i++) {
        for (var e = 0; e < 3; e++) { out[i][e] <== LessThan64Bits()(in[i][e]); }
    }
}

// Goldilocks power and inverse, for constants computed at compile time.
function gl_pow(base, e) {
    var p = 0xFFFFFFFF00000001;
    var r = 1;
    for (var i = 0; i < e; i++) { r = (r * base) % p; }
    return r;
}

// 1 / shift^(2^(nBitsExt - prevBits)) (and times w[prevBits] for e1): the fold step's coset offsets.
function bn_e0(nBitsExt, prevBits) {
    var p = 0xFFFFFFFF00000001;
    var e = domainShift();
    for (var i = 0; i < nBitsExt - prevBits; i++) { e = (e * e) % p; }
    return _inv1(e);
}

function bn_e1(nBitsExt, prevBits) {
    var p = 0xFFFFFFFF00000001;
    var e = domainShift();
    for (var i = 0; i < nBitsExt - prevBits; i++) { e = (e * e) % p; }
    return _inv1((e * roots(prevBits)) % p);
}

// xi^(2^(i+1)) up to xi^N, and Zh = 1 / (xi^N - 1).
template ZhBN128(nBits) {

    signal input {maxNum} xi[3];
    signal output {maxNum} Zh[3];
    signal output {maxNum} xN[3];

    signal {maxNum} zMul[nBits][3];
    zMul.maxNum = max64();
    for (var i = 0; i < nBits; i++) {
        if (i == 0) {
            zMul[i] <== GLCMul()(xi, xi);
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

    signal input {maxNum} xN[3];
    signal input {maxNum} evalP[3];
    signal input {maxNum} evals[nEvals][3];
    signal input {binary} enable;

    signal {maxNum} xAcc[qDeg][3];
    signal {maxNum} qStep[nonEmpty(qDeg - 1)][3];
    signal {maxNum} qAcc[qDeg][3];

    qStep.maxNum = max64();
    qAcc.maxNum = qDeg * max64();
    xAcc.maxNum = max64();

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
template DeepConstantsBN128(nOpenings, nPols, nEvals, OPENING_NUM_EVALS, EVAL_COLUMN) {
    var n = nOpenings;

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
        for (var i = 0; i < OPENING_NUM_EVALS[r]; i++) {
            var k = k0 + i;
            var c = EVAL_COLUMN[k];
            if (c > 0) {
                kMul[k] = GLCMul();
                if (c == 1) { kMul[k].ina <== vf[1]; } else { kMul[k].ina <== p2[c].out; }
                kMul[k].inb <== evals[k];
            }
        }
        for (var i = 1; i < OPENING_NUM_EVALS[r]; i++) {
            var k = k0 + i;
            var c = EVAL_COLUMN[k];
            kSum[k] = GLCAdd();
            if (i > 1) {
                kSum[k].ina <== kSum[k - 1].out;
            } else if (EVAL_COLUMN[k - 1] == 0) {
                kSum[k].ina <== evals[k - 1];
            } else {
                kSum[k].ina <== kMul[k - 1].out;
            }
            if (c == 0) { kSum[k].inb <== evals[k]; } else { kSum[k].inb <== kMul[k].out; }
        }
        k0 += OPENING_NUM_EVALS[r];
    }

    // A sum exceeds max64() and is reduced; a product or an input is already below it.
    friConsts.maxNum = max64();
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
        var kl = k0 + OPENING_NUM_EVALS[r] - 1;
        if (OPENING_NUM_EVALS[r] > 1) {
            friConsts[2 + nPols + n + r] <== GLCNorm()(kSum[kl].out);
        } else if (EVAL_COLUMN[kl] == 0) {
            friConsts[2 + nPols + n + r] <== evals[kl];
        } else {
            friConsts[2 + nPols + n + r] <== kMul[kl].out;
        }
        k0 += OPENING_NUM_EVALS[r];
    }
}

/*
    f_0 at one query x of L_0 against fold tree 1's opened coset (deep.circom's DeepQuery, emulated):
    x from repeated GLMul as the former templated verifier did, the same grouped sum
    (lib_params.rs: OPENING_EXPONENT, PREFIX_SUMS, COLUMN_GROUPS), a base-field leaf lifted with GLC3 where it meets an extension.
*/
template DeepQueryBN128(nOpenings, nPols, nColumnGroups, nPrefixSums, bits0, nextBits, s0Width, nBits, OPENING_EXPONENT, PREFIX_SUMS, COLUMN_GROUPS) {
    var n = nOpenings;
    var nextStep = bits0 - nextBits;

    signal input {binary} queriesL0[bits0];
    signal input {maxNum} xi[3];
    signal input {maxNum} friConsts[2 + nPols + 2 * n][3];
    signal input {maxNum} leaves[s0Width];
    signal input {maxNum} s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    // x = shift * w^i from the query bits.
    signal {maxNum} xacc[bits0 - 1];
    xacc.maxNum = max64();
    signal {maxNum} ina;
    signal {maxNum} inb;
    ina.maxNum = max64();
    inb.maxNum = max64();
    signal {maxNum} inp[nonEmpty(bits0 - 2)];
    inp.maxNum = max64();
    ina <== queriesL0[0] * (domainShift() * roots(bits0) - domainShift()) + domainShift();
    inb <== queriesL0[1] * (roots(bits0 - 1) - 1) + 1;
    xacc[0] <== GLMul()(ina, inb);
    for (var i = 2; i < bits0; i++) {
        inp[i - 2] <== queriesL0[i] * (roots(bits0 - i) - 1) + 1;
        xacc[i - 1] <== GLMul()(xacc[i - 2], inp[i - 2]);
    }
    signal {maxNum} X[3] <== GLC3()(xacc[bits0 - 2]);

    // 1 / (x - g^o xi) per opening with evaluations (rank r), g = roots(nBits) and g^o a constant.
    signal {maxNum} xDivXSubXi[n][3];
    xDivXSubXi.maxNum = max64();
    component gPowXi[n];
    component xSubPoint[n];
    for (var r = 0; r < n; r++) {
        var o = OPENING_EXPONENT[r];
        if (o != 0) {
            gPowXi[r] = GLCMul();
            gPowXi[r].ina <== xi;
            gPowXi[r].inb <== GLConst3(gl_pow(o > 0 ? roots(nBits) : invroots(nBits), o > 0 ? o : -o))();
        }
        xSubPoint[r] = GLCSub();
        xSubPoint[r].ina <== X;
        if (o == 0) { xSubPoint[r].inb <== xi; } else { xSubPoint[r].inb <== gPowXi[r].out; }
        xDivXSubXi[r] <== GLCInv()(xSubPoint[r].out);
    }

    // In deep.rs's op order (as the former templated verifier emitted it): A_r = vf1^{e_r} * xDivXSubXi[r] (e = 0
    // for the last opening); per column group its new prefix sums of A (PREFIX_SUMS), its columns' products,
    // their sum S_G and S_G * sum A; then the groups' sum.
    component aMul[n];
    for (var r = 0; r < n - 1; r++) {
        aMul[r] = GLCMul();
        aMul[r].ina <== friConsts[2 + nPols + r];
        aMul[r].inb <== xDivXSubXi[r];
    }
    // COLUMN_GROUPS per group: its A source, #columns, (column, leaf, dim)... A column's term is colMul[c] (c > 0),
    // else its leaf: an extension inline, a base one lifted (lift0). The running S_G is a column's term
    // (sKind 0: leaf, 1: lift0, 2: colMul[c]) or colSum[c] (3).
    component aSum[nonEmpty(nPrefixSums)];
    component colLift[nPols];
    component colMul[nPols];
    component colSum[nPols];
    component lift0[1];
    component term[nColumnGroups];
    var kNext = 0;
    var p = 0;
    for (var g = 0; g < nColumnGroups; g++) {
        var src = COLUMN_GROUPS[p];
        var nCols = COLUMN_GROUPS[p + 1];
        for (var k = kNext; k <= src - n; k++) {
            var s0 = PREFIX_SUMS[2 * k];
            var rj = PREFIX_SUMS[2 * k + 1];
            aSum[k] = GLCAdd();
            if (s0 >= n) { aSum[k].ina <== aSum[s0 - n].out; } else if (s0 == n - 1) { aSum[k].ina <== xDivXSubXi[s0]; } else { aSum[k].ina <== aMul[s0].out; }
            if (rj == n - 1) { aSum[k].inb <== xDivXSubXi[rj]; } else { aSum[k].inb <== aMul[rj].out; }
            kNext = k + 1;
        }
        for (var i = 0; i < nCols; i++) {
            var c = COLUMN_GROUPS[p + 2 + 3 * i];
            var off = COLUMN_GROUPS[p + 3 + 3 * i];
            var dim = COLUMN_GROUPS[p + 4 + 3 * i];
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
            var c = COLUMN_GROUPS[p + 2 + 3 * i];
            var off = COLUMN_GROUPS[p + 3 + 3 * i];
            var dim = COLUMN_GROUPS[p + 4 + 3 * i];
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
    component termSum[nColumnGroups];
    for (var g = 1; g < nColumnGroups; g++) {
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
        } else if (nColumnGroups > 1) {
            kSub[r].ina <== termSum[nColumnGroups - 1].out;
        } else {
            kSub[r].ina <== term[0].out;
        }
        kSub[r].inb <== kMul[r].out;
    }

    // Reduced once on its own (as the former templated verifier did), then against the next oracle's member.
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
        s_sx.maxNum = max64();
        signal {maxNum} inp[nonEmpty(currStepBits - 2)];
        inp.maxNum = max64();
        signal {maxNum} ina;
        signal {maxNum} inb;
        ina.maxNum = max64();
        inb.maxNum = max64();
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
        inp2.maxNum = max64();
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
    One BN128-verified STARK proof of the compilation's key k (key_*(k), lib_params.rs) from its flat
    proof; the family half (bn128_open_<family>.circom, picked by the generated verifier's include) opens
    the trees, the key's ConstraintEval checks the constraints.
*/
template StarkVerifierBN128(k) {
    var nBits = key_nBits(k);
    var nBitsExt = key_nBitsExt(k);
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nQueries = key_nQueries(k);
    var nEvals = key_nEvals(k);
    var nPublics = key_nPublics(k);
    var nStages = key_nStages(k);
    var nChallenges = key_nChallenges(k);
    var nSteps = key_nSteps(k);
    var nTrees = key_nTrees(k);
    var verkeyInput = key_verkeyInput(k);
    var enableInput = key_enableInput(k);
    var offNonce = key_offNonce(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var ROOT_C[4] = key_ROOT_C(k);
    var OFF_STAGE_ROOT[nStages + 1] = key_OFF_STAGE_ROOT(k);
    var OFF_L0_VALUES[nTrees] = key_OFF_L0_VALUES(k);
    var OFF_L0_SIBLINGS[nTrees] = key_OFF_L0_SIBLINGS(k);
    var OFF_L0_LAST_LEVELS[nTrees] = key_OFF_L0_LAST_LEVELS(k);
    var OFF_L0_ROOT[nTrees] = key_OFF_L0_ROOT(k);
    var nFoldTrees = nSteps - 1;
    var OFF_FOLD_ROOT[nonEmpty(nFoldTrees)] = key_OFF_FOLD_ROOT(k);
    var OFF_FOLD_VALUES[nonEmpty(nFoldTrees)] = key_OFF_FOLD_VALUES(k);
    var OFF_FOLD_SIBLINGS[nonEmpty(nFoldTrees)] = key_OFF_FOLD_SIBLINGS(k);
    var OFF_FOLD_LAST_LEVELS[nonEmpty(nFoldTrees)] = key_OFF_FOLD_LAST_LEVELS(k);
    var rw = bn_rootW();
    var sw = bn_sibW(arity);
    var L0 = l0PathLength(arity, lastLevels, STEPS);
    var lastSize = clearLevelNodes(arity, lastLevels);
    var FS = lastDomainSize(nSteps, STEPS);
    var zeroRoot[1][rw];
    for (var i = 0; i < rw; i++) { zeroRoot[0][i] = 0; }

    signal input proof[key_proofSize(k)];
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

    var P = key_proofSize(k);
    signal evals[nEvals][3] <== ProofRows(P, key_offEvals(k), nEvals, 3)(proof);
    signal finalPol[FS][3] <== ProofRows(P, key_offFinal(k), FS, 3)(proof);
    signal stageRoot[nStages + 1][rw];
    for (var s = 0; s <= nStages; s++) { for (var i = 0; i < rw; i++) { stageRoot[s][i] <== proof[OFF_STAGE_ROOT[s] + i]; } }
    signal treeRoot[nTrees][rw];
    signal treeLast[nTrees][lastSize][rw];
    for (var t = 0; t < nTrees; t++) {
        for (var i = 0; i < rw; i++) {
            if (OFF_L0_ROOT[t] >= 0) { treeRoot[t][i] <== proof[OFF_L0_ROOT[t] + i]; } else { treeRoot[t][i] <== rootC[i]; }
        }
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < rw; i++) {
                if (lastLevels > 0) { treeLast[t][j][i] <== proof[OFF_L0_LAST_LEVELS[t] + rw * j + i]; } else { treeLast[t][j][i] <== 0; }
            }
        }
    }
    signal foldRoot[nFoldTrees][rw];
    signal foldLast[nFoldTrees][lastSize][rw];
    for (var f = 0; f < nFoldTrees; f++) {
        for (var i = 0; i < rw; i++) { foldRoot[f][i] <== proof[OFF_FOLD_ROOT[f] + i]; }
        for (var j = 0; j < lastSize; j++) {
            for (var i = 0; i < rw; i++) {
                if (lastLevels > 0) { foldLast[f][j][i] <== proof[OFF_FOLD_LAST_LEVELS[f] + rw * j + i]; } else { foldLast[f][j][i] <== 0; }
            }
        }
    }

    signal {maxNum} publicsGL[nPublics];
    if (nPublics > 0) { publicsGL <== GLVals(nPublics)(publics); }
    signal {maxNum} evalsGL[nEvals][3] <== GLVals3(nEvals)(evals);
    signal {maxNum} finalPolGL[FS][3] <== GLVals3(FS)(finalPol);

    // The GL verifier's transcript over this key's engine; its challenges reduced Goldilocks values.
    component transcript = StarkVerifierTranscript(nStages, nPublics, 0, 0, nEvals, nChallenges, nSteps, STEPS, 0, 0, 0, [0], key_hashCommits(k), 0, nQueries, key_powBits(k), key_CHALLENGES_PER_STAGE(k), [0], [0]);
    transcript.globalChallenge <== [0, 0, 0];
    if (nPublics > 0) { transcript.publics <== publicsGL; } else { transcript.publics <== [0]; }
    transcript.rootC <== rootC;
    transcript.stageRoots <== stageRoot;
    transcript.airvalues <== [[0, 0, 0]];
    transcript.airgroupvalues <== [[0, 0, 0]];
    transcript.evals <== evalsGL;
    if (nFoldTrees > 0) { transcript.foldRoots <== foldRoot; } else { transcript.foldRoots <== zeroRoot; }
    transcript.betas <== [[0, 0, 0]];
    transcript.hints <== [[0, 0, 0]];
    transcript.finalPol <== finalPolGL;
    if (offNonce >= 0) { transcript.nonce <== proof[offNonce]; } else { transcript.nonce <== 0; }
    transcript.enable <== enabled;
    signal {maxNum} challenges[nChallenges][3];
    challenges.maxNum = max64();
    challenges <== transcript.challenges;
    signal {maxNum} xi[3];
    xi.maxNum = max64();
    xi <== transcript.xi;
    signal {maxNum} vf[2][3];
    vf.maxNum = max64();
    vf <== transcript.vf;
    signal {maxNum} rFold[nonEmpty(nFoldTrees)][3];
    rFold.maxNum = max64();
    if (nFoldTrees > 0) {
        for (var f = 0; f < nFoldTrees; f++) { rFold[f] <== transcript.ldt[ldt_rFold(0, f)]; }
    } else {
        rFold[0] <== [0, 0, 0];
    }

    // The constraints at xi against the quotient.
    component zh = ZhBN128(nBits);
    zh.xi <== xi;
    component ce = ConstraintEval();
    ce.challenges <== challenges;
    ce.evals <== evalsGL;
    ce.Zh <== zh.Zh;
    ce.publics <== publicsGL;
    QuotientCheckBN128(key_qDeg(k), key_qPolEv(k), nEvals)(zh.xN, ce.evalP, evalsGL, enabled);

    component deepConsts = DeepConstantsBN128(key_nOpenings(k), key_nPols(k), nEvals, key_OPENING_NUM_EVALS(k), key_EVAL_COLUMN(k));
    deepConsts.vf <== vf;
    deepConsts.evals <== evalsGL;

    // Per query (the proof holds each tree's data query after query); the levels sent in clear once.
    component query[nQueries];
    for (var g = 0; g < nQueries; g++) {
        query[g] = VerifySingleQueryBN128(k);
        query[g].queriesL0 <== transcript.queriesL0[g];
        query[g].xi <== xi;
        query[g].friConsts <== deepConsts.friConsts;
        query[g].rFold <== rFold;
        query[g].finalPol <== finalPolGL;
        query[g].enable <== enabled;
        query[g].treeLast <== treeLast;
        query[g].treeRoot <== treeRoot;
        query[g].foldLast <== foldLast;
        query[g].foldRoot <== foldRoot;
        for (var t = 0; t < nTrees; t++) {
            var width = L0_TREE_START[t + 1] - L0_TREE_START[t];
            for (var i = 0; i < width; i++) { query[g].l0Values[L0_TREE_START[t] + i] <== proof[OFF_L0_VALUES[t] + g * width + i]; }
            for (var l = 0; l < L0; l++) { for (var i = 0; i < sw; i++) { query[g].l0Siblings[t][l][i] <== proof[OFF_L0_SIBLINGS[t] + (g * L0 + l) * sw + i]; } }
        }
        for (var f = 0; f < nFoldTrees; f++) {
            var cosetSize = foldCosetSize(STEPS, f + 1);
            var pathLevels = foldPathLength(arity, lastLevels, STEPS, f + 1);
            var cosetStart = foldCosetStart(STEPS, f);
            var pathStart = foldPathStart(arity, lastLevels, STEPS, f);
            for (var c = 0; c < cosetSize; c++) { for (var e = 0; e < 3; e++) { query[g].foldCosets[cosetStart + c][e] <== proof[OFF_FOLD_VALUES[f] + (g * cosetSize + c) * 3 + e]; } }
            for (var l = 0; l < pathLevels; l++) { for (var i = 0; i < sw; i++) { query[g].foldSiblings[pathStart + l][i] <== proof[OFF_FOLD_SIBLINGS[f] + (g * pathLevels + l) * sw + i]; } }
        }
    }
    if (lastLevels > 0) {
        for (var t = 0; t < nTrees; t++) { BN128RootCheck(lastLevels, arity, STEPS[0])(treeLast[t], treeRoot[t], enabled); }
        for (var f = 0; f < nFoldTrees; f++) { BN128RootCheck(lastLevels, arity, STEPS[f + 1])(foldLast[f], foldRoot[f], enabled); }
    }
    VerifyFinalPolBN128(STEPS[nSteps - 1], finalPolDegreeBound(nBits, nBitsExt, nSteps, STEPS))(finalPolGL, enabled);
}

/*
    One query: every L_0 tree's row and fold tree's coset, reduced, against their trees; DEEP; every
    folding step (QueryFoldStepBN128). Its data comes flat: the L_0 rows at L0_TREE_START, the fold-tree
    cosets and paths at foldCosetStart / foldPathStart.
*/
template parallel VerifySingleQueryBN128(k) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nSteps = key_nSteps(k);
    var nTrees = key_nTrees(k);
    var nOpenings = key_nOpenings(k);
    var nPols = key_nPols(k);
    var STEPS[nSteps] = key_STEPS(k);
    var L0_TREE_START[nTrees + 1] = key_L0_TREE_START(k);
    var nFoldTrees = nSteps - 1;
    var rw = bn_rootW();
    var sw = bn_sibW(arity);
    var L0 = l0PathLength(arity, lastLevels, STEPS);
    var lastSize = clearLevelNodes(arity, lastLevels);
    var nFoldCosetRows = foldCosetStart(STEPS, nFoldTrees);

    signal input {binary} queriesL0[STEPS[0]];
    signal input {maxNum} xi[3];
    signal input {maxNum} friConsts[2 + nPols + 2 * nOpenings][3];
    signal input {maxNum} rFold[nonEmpty(nFoldTrees)][3];
    signal input {maxNum} finalPol[lastDomainSize(nSteps, STEPS)][3];
    signal input {binary} enable;
    signal input treeLast[nTrees][lastSize][rw];
    signal input treeRoot[nTrees][rw];
    signal input foldLast[nFoldTrees][lastSize][rw];
    signal input foldRoot[nFoldTrees][rw];

    signal input l0Values[L0_TREE_START[nTrees]];
    signal input l0Siblings[nTrees][L0][sw];
    signal input foldCosets[nFoldCosetRows][3];
    signal input foldSiblings[foldPathStart(arity, lastLevels, STEPS, nFoldTrees)][sw];

    // Every L_0 row and fold-tree coset reduced before any tree is opened.
    signal {maxNum} leaves[L0_TREE_START[nTrees]];
    leaves.maxNum = max64();
    for (var i = 0; i < L0_TREE_START[nTrees]; i++) { leaves[i] <== LessThan64Bits()(l0Values[i]); }
    signal {maxNum} cosets[nFoldCosetRows][3];
    cosets.maxNum = max64();
    for (var c = 0; c < nFoldCosetRows; c++) { for (var e = 0; e < 3; e++) { cosets[c][e] <== LessThan64Bits()(foldCosets[c][e]); } }

    // The per-tree and per-step arrays mix template instances, resolved through a map at run time.
    component l0Opening[nTrees];
    for (var t = 0; t < nTrees; t++) {
        var width = L0_TREE_START[t + 1] - L0_TREE_START[t];
        l0Opening[t] = BN128Open(1, width, arity, lastLevels, L0, STEPS[0], STEPS[0]);
        l0Opening[t].queries <== queriesL0;
        for (var i = 0; i < width; i++) { l0Opening[t].values[i][0] <== leaves[L0_TREE_START[t] + i]; }
        l0Opening[t].siblings <== l0Siblings[t];
        l0Opening[t].last <== treeLast[t];
        l0Opening[t].root <== treeRoot[t];
        l0Opening[t].enable <== enable;
    }

    // f_0 lands on fold tree 0's coset, or on finalPol when nothing folds.
    if (nFoldTrees > 0) {
        signal {maxNum} firstCoset[foldCosetSize(STEPS, 1)][3];
        firstCoset.maxNum = cosets.maxNum;
        for (var c = 0; c < foldCosetSize(STEPS, 1); c++) { firstCoset[c] <== cosets[c]; }
        DeepQueryBN128(nOpenings, nPols, key_nColumnGroups(k), key_nPrefixSums(k), STEPS[0], STEPS[1], L0_TREE_START[nTrees], key_nBits(k), key_OPENING_EXPONENT(k), key_PREFIX_SUMS(k), key_COLUMN_GROUPS(k))(queriesL0, xi, friConsts, leaves, firstCoset, enable);
    } else {
        DeepQueryBN128(nOpenings, nPols, key_nColumnGroups(k), key_nPrefixSums(k), STEPS[0], 0, L0_TREE_START[nTrees], key_nBits(k), key_OPENING_EXPONENT(k), key_PREFIX_SUMS(k), key_COLUMN_GROUPS(k))(queriesL0, xi, friConsts, leaves, finalPol, enable);
    }

    component foldStep[nonEmpty(nFoldTrees)];
    for (var f = 0; f < nFoldTrees; f++) {
        foldStep[f] = QueryFoldStepBN128(k, f + 1);
        foldStep[f].queriesL0 <== queriesL0;
        foldStep[f].cosets <== cosets;
        if (foldPathStart(arity, lastLevels, STEPS, nFoldTrees) > 0) {
            foldStep[f].foldSiblings <== foldSiblings;
        } else {
            for (var i = 0; i < sw; i++) { foldStep[f].foldSiblings[0][i] <== 0; }
        }
        foldStep[f].last <== foldLast[f];
        foldStep[f].root <== foldRoot[f];
        foldStep[f].rFold <== rFold[f];
        foldStep[f].finalPol <== finalPol;
        foldStep[f].enable <== enable;
    }
}

/*
    Fold step s >= 1 at one query (QueryFoldStep, emulated): fold tree s - 1's reduced coset against the
    tree, and that coset folded with rFold equals its member in the next fold tree's coset, or in finalPol
    after the last step. No input is empty, as for QueryFoldStep: every step gets finalPol, and fold paths
    all sent in clear still pass one row of zeros.
*/
template QueryFoldStepBN128(k, s) {
    var arity = key_arity(k);
    var lastLevels = key_lastLevels(k);
    var nSteps = key_nSteps(k);
    var STEPS[nSteps] = key_STEPS(k);
    var nFoldTrees = nSteps - 1;
    var rw = bn_rootW();
    var sw = bn_sibW(arity);
    var cosetSize = foldCosetSize(STEPS, s);
    var pathLevels = foldPathLength(arity, lastLevels, STEPS, s);
    var nextBits = nextFoldDomainBits(STEPS, nSteps, s);
    var nextCosetSize = 1 << (STEPS[s] - nextBits);
    var foldsIntoFinalPol = s == nFoldTrees ? 1 : 0;

    signal input {binary} queriesL0[STEPS[0]];
    signal input {maxNum} cosets[foldCosetStart(STEPS, nFoldTrees)][3];
    signal input foldSiblings[nonEmpty(foldPathStart(arity, lastLevels, STEPS, nFoldTrees))][sw];
    signal input last[clearLevelNodes(arity, lastLevels)][rw];
    signal input root[rw];
    signal input {maxNum} rFold[3];
    signal input {maxNum} finalPol[lastDomainSize(nSteps, STEPS)][3];
    signal input {binary} enable;

    signal {maxNum} coset[cosetSize][3];
    coset.maxNum = cosets.maxNum;
    for (var c = 0; c < cosetSize; c++) { coset[c] <== cosets[foldCosetStart(STEPS, s - 1) + c]; }
    if (pathLevels > 0) {
        signal siblings[pathLevels][sw];
        for (var l = 0; l < pathLevels; l++) { siblings[l] <== foldSiblings[foldPathStart(arity, lastLevels, STEPS, s - 1) + l]; }
        BN128Open(3, cosetSize, arity, lastLevels, pathLevels, STEPS[s], STEPS[0])(queriesL0, coset, siblings, last, root, enable);
    } else {
        // The whole path is sent in clear: the opening ignores its one siblings row.
        var noSiblings[1][sw];
        for (var i = 0; i < sw; i++) { noSiblings[0][i] = 0; }
        BN128Open(3, cosetSize, arity, lastLevels, 0, STEPS[s], STEPS[0])(queriesL0, coset, noSiblings, last, root, enable);
    }

    signal {maxNum} nextCoset[nextCosetSize][3];
    if (foldsIntoFinalPol == 1) {
        nextCoset.maxNum = finalPol.maxNum;
        nextCoset <== finalPol;
    } else {
        nextCoset.maxNum = cosets.maxNum;
        for (var c = 0; c < nextCosetSize; c++) { nextCoset[c] <== cosets[foldCosetStart(STEPS, s) + c]; }
    }
    VerifyFRIBN128(STEPS[s - 1], STEPS[s], nextBits, key_nBitsExt(k), STEPS[0])(queriesL0, rFold, coset, nextCoset, enable);
}
