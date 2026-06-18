pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2_constants.circom";

template Sigma() {
    signal input in;
    signal output out;

    signal in2;
    signal in4;
    signal in6;

    in2 <== in*in;
    in4 <== in2*in2;
    in6 <== in4*in2;

    out <== in6*in;
}

// Internal (partial) mixing layer for width 8: sum all 8, then out[i] = in[i]*DIAG(2,i) + sum
template Mix8() {
    signal input in[8];
    signal output out[8];

    signal sum <== in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];

    for (var i = 0; i < 8; i++) {
        out[i] <== in[i] * MATRIX_DIAGONAL(2, i) + sum;
    }
}

template MatMul_M4() {
    signal input in[4];
    signal output out[4];

    signal t0 <== in[0] + in[1];
    signal t1 <== in[2] + in[3];
    signal t2 <== 2*in[1] + t1;
    signal t3 <== 2*in[3] + t0;
    signal t4 <== 4*t1 + t3;
    signal t5 <== 4*t0 + t2;
    signal t6 <== t3 + t5;
    signal t7 <== t2 + t4;

    out[0] <== t6;
    out[1] <== t5;
    out[2] <== t7;
    out[3] <== t4;
}

// External matrix for width 8: M4 on each 4-element half, then out[j] += stored[j%4]
// where stored[i] = sum of column i across all chunks (post-M4).
// For W=8: stored[i] = half0[i] + half1[i], so:
//   out[i]     = half0[i] + stored[i]     = 2*half0[i] + half1[i]
//   out[4 + i] = half1[i] + stored[i]     = half0[i]   + 2*half1[i]
template MatMul_External8() {
    signal input in[8];
    signal output out[8];

    signal half0[4] <== MatMul_M4()([in[0], in[1], in[2], in[3]]);
    signal half1[4] <== MatMul_M4()([in[4], in[5], in[6], in[7]]);

    for (var i = 0; i < 4; i++) {
        out[i]     <== 2*half0[i] + half1[i];
        out[4 + i] <== half0[i]   + 2*half1[i];
    }
}

template Poseidon2_8(nOuts) {
    signal input in[8];
    signal output out[nOuts];

    signal st0[8] <== MatMul_External8()(in);

    component sigmaF[8][8];
    component matmulF[8];
    component sigmaP[22];
    component mixP[22];

    // First 4 full rounds
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF[r][t] = Sigma();
            if (r == 0) {
                sigmaF[r][t].in <== st0[t] + CONSTANTS(2, 8*r + t);
            } else {
                sigmaF[r][t].in <== matmulF[r-1].out[t] + CONSTANTS(2, 8*r + t);
            }
        }
        matmulF[r] = MatMul_External8();
        matmulF[r].in <== [
            sigmaF[r][0].out, sigmaF[r][1].out, sigmaF[r][2].out, sigmaF[r][3].out,
            sigmaF[r][4].out, sigmaF[r][5].out, sigmaF[r][6].out, sigmaF[r][7].out
        ];
    }

    // 22 partial rounds
    for (var r = 0; r < 22; r++) {
        sigmaP[r] = Sigma();
        mixP[r] = Mix8();
        if (r == 0) {
            sigmaP[r].in <== matmulF[3].out[0] + CONSTANTS(2, 4*8 + r);
            mixP[r].in <== [
                sigmaP[r].out,
                matmulF[3].out[1], matmulF[3].out[2], matmulF[3].out[3],
                matmulF[3].out[4], matmulF[3].out[5], matmulF[3].out[6], matmulF[3].out[7]
            ];
        } else {
            sigmaP[r].in <== mixP[r-1].out[0] + CONSTANTS(2, 4*8 + r);
            mixP[r].in <== [
                sigmaP[r].out,
                mixP[r-1].out[1], mixP[r-1].out[2], mixP[r-1].out[3],
                mixP[r-1].out[4], mixP[r-1].out[5], mixP[r-1].out[6], mixP[r-1].out[7]
            ];
        }
    }

    // Last 4 full rounds
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF[4 + r][t] = Sigma();
            if (r == 0) {
                sigmaF[4 + r][t].in <== mixP[21].out[t] + CONSTANTS(2, 4*8 + 22 + 8*r + t);
            } else {
                sigmaF[4 + r][t].in <== matmulF[4 + r - 1].out[t] + CONSTANTS(2, 4*8 + 22 + 8*r + t);
            }
        }
        matmulF[4 + r] = MatMul_External8();
        matmulF[4 + r].in <== [
            sigmaF[4 + r][0].out, sigmaF[4 + r][1].out, sigmaF[4 + r][2].out, sigmaF[4 + r][3].out,
            sigmaF[4 + r][4].out, sigmaF[4 + r][5].out, sigmaF[4 + r][6].out, sigmaF[4 + r][7].out
        ];
    }

    for (var t = 0; t < nOuts; t++) {
        out[t] <== matmulF[7].out[t];
    }
}
