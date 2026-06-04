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

template Mix() {
    signal input in[4];
    signal output out[4];

    signal sum <== in[0] + in[1] + in[2] + in[3];
    
    for (var i = 0; i < 4; i++) {
        out[i] <== sum + in[i] * MATRIX_DIAGONAL(1, i);
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

template Poseidon2_1(nOuts) {
    signal input in[4];
    signal output out[nOuts];

    signal initial_st[4] <== MatMul_M4()(in);

    component matmul[8];
    component sigmaF[8][4];
    component sigmaP[21];
    component mixP[21];
    

    for(var r = 0; r < 4; r++) {
        for (var t=0; t < 4; t++) {
            sigmaF[r][t] = Sigma();
            if (r == 0) {
                sigmaF[r][t].in <== initial_st[t] + CONSTANTS(1, 4*r + t);
            } else {
                sigmaF[r][t].in <== matmul[r-1].out[t] + CONSTANTS(1, 4*r + t);
            }
        }
        matmul[r] = MatMul_M4();
        matmul[r].in <== [sigmaF[r][0].out, sigmaF[r][1].out, sigmaF[r][2].out, sigmaF[r][3].out];
    }

    for(var r = 0; r < 21; r++) {
        sigmaP[r] = Sigma();
        mixP[r] = Mix();
        if (r == 0) {
            sigmaP[r].in <== matmul[3].out[0] + CONSTANTS(1, 16 + r);
            mixP[r].in <== [sigmaP[r].out, matmul[3].out[1], matmul[3].out[2], matmul[3].out[3]];
        } else {
            sigmaP[r].in <== mixP[r-1].out[0] + CONSTANTS(1, 16 + r);
            mixP[r].in <== [sigmaP[r].out, mixP[r-1].out[1], mixP[r-1].out[2], mixP[r-1].out[3]];
        }
    }

    for(var r = 0; r < 4; r++) {
        for (var t=0; t < 4; t++) {
            sigmaF[4 + r][t] = Sigma();
            if (r == 0) {
                sigmaF[4 + r][t].in <== mixP[20].out[t] + CONSTANTS(1, 16 + 21 + 4*r + t);
            } else {
                sigmaF[4 + r][t].in <== matmul[4 + r - 1].out[t] + CONSTANTS(1, 16 + 21 + 4*r + t);
            }
        }
        matmul[4 + r] = MatMul_M4();
        matmul[4 + r].in <== [sigmaF[4 + r][0].out, sigmaF[4 + r][1].out, sigmaF[4 + r][2].out, sigmaF[4 + r][3].out];
    }

    for (var t=0; t < nOuts; t++) {
        out[t] <== matmul[7].out[t];
    }
   
}