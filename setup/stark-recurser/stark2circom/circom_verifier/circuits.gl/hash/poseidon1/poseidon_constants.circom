pragma circom 2.1.0;

// Placeholder constants for Poseidon1_16 (Hades-Goldilocks, width 16).
// All values are zero — regenerate with the actual round constants and matrix
// entries before producing real proofs.
//
// Function shapes:
//   MDS(in[16])   → out[16]      applies the M matrix to a width-16 state
//   CNST(i)                      C[i] for i in 0..149
//   S(i)                         S[i] for i in 0..681 (22 × 31 sparse matrix entries)
//   M(i, j)                      M[i][j] for i,j in 0..15
//   P(i, j)                      P[i][j] for i,j in 0..15

function MDS(in) {
    var out[16];
    for (var i = 0; i < 16; i++) {
        out[i] = 0;
    }
    return out;
}

function CNST(i) {
    var c[150];
    for (var k = 0; k < 150; k++) { c[k] = 0; }
    return c[i];
}

function S(i) {
    var s[682];
    for (var k = 0; k < 682; k++) { s[k] = 0; }
    return s[i];
}

function M(i, j) {
    var m[16][16];
    for (var a = 0; a < 16; a++) {
        for (var b = 0; b < 16; b++) {
            m[a][b] = 0;
        }
    }
    return m[i][j];
}

function P(i, j) {
    var p[16][16];
    for (var a = 0; a < 16; a++) {
        for (var b = 0; b < 16; b++) {
            p[a][b] = 0;
        }
    }
    return p[i][j];
}

