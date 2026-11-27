pragma circom 2.1.0;

// nRows rows of `width` words of a flat proof from `offset` (evals, air values, finalPol, ...).
template ProofRows(proofSize, offset, nRows, width) {
    signal input proof[proofSize];
    signal output rows[nRows][width];

    for (var i = 0; i < nRows; i++) {
        for (var e = 0; e < width; e++) { rows[i][e] <== proof[offset + width * i + e]; }
    }
}
