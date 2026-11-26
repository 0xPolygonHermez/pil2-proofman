pragma circom 2.1.0;

// FRI/STIR shapes derived from the key's data; the one place they are computed.
// STEPS: FRI logDomainSizes; STIR [log|L_0|, log|L_0| - k_0] (L_0 queries and tree T_0).

// circom rejects zero-length arrays: optional inputs are padded to one entry.
function nonEmpty(n) { return n > 0 ? n : 1; }

// The STARK domain shift.
function domainShift() { return 7; }

function log2Arity(arity) {
    var b = 0;
    while ((1 << b) < arity) { b++; }
    return b;
}

function siblingWords(arity) { return (arity - 1) * 4; }

// ceil(bits / log2(arity)): Merkle levels of a tree over 2^bits leaves.
function merkleHeight(arity, bits) {
    var nba = log2Arity(arity);
    return (bits + nba - 1) \ nba;
}

function clearLevelNodes(arity, lastLevels) { return lastLevels > 0 ? arity ** lastLevels : 1; }

// Merkle path below the levels sent in clear, of a tree over 2^bits leaves (0 selects the Empty variant).
function merklePathLength(arity, lastLevels, bits) {
    var full = merkleHeight(arity, bits);
    return full > lastLevels ? full - lastLevels : 0;
}

// Path length of an L_0 tree below the levels sent in clear.
function l0PathLength(arity, lastLevels, STEPS) { return merkleHeight(arity, STEPS[0]) - lastLevels; }

function lastDomainSize(nSteps, STEPS) { return 1 << STEPS[nSteps - 1]; }

// Coefficients of finalPol below the degree bound.
function finalPolDegreeBound(nBits, nBitsExt, nSteps, STEPS) {
    var blowup = nBitsExt - nBits;
    var fb = STEPS[nSteps - 1];
    return 1 << (fb > blowup ? fb - blowup : 0);
}

// Fold step s >= 1: coset size of tree s (f_{s-1} grouped by 2^{k_{s-1}}).
function foldCosetSize(STEPS, s) { return 1 << (STEPS[s - 1] - STEPS[s]); }

// Fold step s >= 1: Merkle path of tree s below the levels sent in clear.
function foldPathLength(arity, lastLevels, STEPS, s) { return merklePathLength(arity, lastLevels, STEPS[s]); }

// 1 / shift^(2^(nBitsExt - STEPS[s-1])): the coset offset of fold step s's domain.
function foldCosetShiftInv(nBitsExt, STEPS, s) {
    var e = domainShift();
    for (var i = 0; i < nBitsExt - STEPS[s - 1]; i++) { e = e * e; }
    return 1 / e;
}

// Committed oracles after f_0's: FRI's fold trees, STIR's T_0..T_{M-1}.
function nFoldTrees(stir, nSteps, stirM) { return stir == 1 ? stirM : nSteps - 1; }

// finalPol rows in the proof: FRI's evaluations over the last domain, STIR's coefficients.
function finalPolSize(stir, nSteps, STEPS, stirFinalBits) { return stir == 1 ? 1 << stirFinalBits : lastDomainSize(nSteps, STEPS); }

// The low-degree test's squeezes in draw order (transcript output ldt). FRI: rFold[0..nFoldTrees), the query
// challenge. STIR: rFold[0], then per iteration i >= 1: rOut, rFold[i], rComb, query challenge of round
// i - 1, rho; then the last round's query challenge.
function ldt_n(stir, nFoldTrees) { return stir == 1 ? 5 * nFoldTrees - 3 : nFoldTrees + 1; }
function ldt_rFold(stir, i) { return stir == 1 && i > 0 ? 5 * i - 3 : i; }
function ldt_rOut(j) { return 5 * j + 1; }
function ldt_rComb(j) { return 5 * j + 3; }
function ldt_rho(j) { return 5 * j + 5; }
function ldt_query(stir, nFoldTrees, r) {
    if (stir == 0) { return nFoldTrees; }
    return r < nFoldTrees - 1 ? 5 * r + 4 : 5 * nFoldTrees - 4;
}

// STIR round r (L = logDomainSizes, K = foldingFactors, T = numQueries).
// Bits of |G_{r+1}| + 1 <= t_r + 2.
function stir_expBits(t) {
    var b = 0;
    while ((1 << b) <= t + 2) { b++; }
    return b;
}

// Offsets in the flat STIR inputs: opened cosets (every round), Merkle paths (rounds >= 1), Ans/shake hints.
function stir_valsOff(T, K, r) {
    var off = 0;
    for (var j = 0; j < r; j++) { off += T[j] << K[j]; }
    return off;
}

function stir_sibOff(arity, lastLevels, L, K, T, r) {
    var off = 0;
    for (var j = 1; j < r; j++) { off += T[j] * merklePathLength(arity, lastLevels, L[j] - K[j]); }
    return off;
}

function stir_hintOff(T, r) {
    var off = 0;
    for (var j = 0; j < r; j++) { off += 2 * (T[j] + 1); }
    return off;
}

function stir_nHints(stir, M, T) {
    if (stir == 0) { return 0; }
    return stir_hintOff(T, M - 1);
}

// Fold trees opened at the L_0 queries: FRI's every one, STIR's T_0 only (its STEPS has two entries).
function nQueryFoldTrees(nSteps) { return nSteps - 1; }

// A query's fold-tree data is flat in fold-tree order: where fold tree f's (fold step f + 1's) opened
// coset and Merkle path start; at f = nQueryFoldTrees, their totals.
function foldCosetStart(STEPS, f) {
    var start = 0;
    for (var j = 0; j < f; j++) { start += foldCosetSize(STEPS, j + 1); }
    return start;
}
function foldPathStart(arity, lastLevels, STEPS, f) {
    var start = 0;
    for (var j = 0; j < f; j++) { start += foldPathLength(arity, lastLevels, STEPS, j + 1); }
    return start;
}

// The bits of fold step s + 1's domain, 0 past the last.
function nextFoldDomainBits(STEPS, nSteps, s) {
    if (s + 1 < nSteps) { return STEPS[s + 1]; }
    return 0;
}
