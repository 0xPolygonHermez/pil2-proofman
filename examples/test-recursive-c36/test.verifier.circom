pragma circom 2.1.0;
pragma custom_templates;




include "cmul.circom";
include "cinv.circom";
include "poseidon2.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "treeselector4.circom";
include "merklehash.circom";


/* 
    Calculate FRI Queries
*/
template calculateFRIQueries0() {
    
    signal input challengeFRIQueries[3];
    signal output {binary} queriesFRI[128][23];


    
    signal transcriptHash_friQueries_0[12] <== Poseidon2(12)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],0,0,0,0,0], [0,0,0,0]);
    signal {binary} transcriptN2b_0[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[0]);
    signal {binary} transcriptN2b_1[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[1]);
    signal {binary} transcriptN2b_2[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[2]);
    signal {binary} transcriptN2b_3[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[3]);
    signal {binary} transcriptN2b_4[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[4]);
    signal {binary} transcriptN2b_5[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[5]);
    signal {binary} transcriptN2b_6[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[6]);
    signal {binary} transcriptN2b_7[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[7]);
    signal {binary} transcriptN2b_8[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[8]);
    signal {binary} transcriptN2b_9[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[9]);
    signal {binary} transcriptN2b_10[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[10]);
    signal {binary} transcriptN2b_11[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[11]);
    
    signal transcriptHash_friQueries_1[12] <== Poseidon2(12)([0,0,0,0,0,0,0,0], [transcriptHash_friQueries_0[0],transcriptHash_friQueries_0[1],transcriptHash_friQueries_0[2],transcriptHash_friQueries_0[3]]);
    signal {binary} transcriptN2b_12[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[0]);
    signal {binary} transcriptN2b_13[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[1]);
    signal {binary} transcriptN2b_14[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[2]);
    signal {binary} transcriptN2b_15[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[3]);
    signal {binary} transcriptN2b_16[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[4]);
    signal {binary} transcriptN2b_17[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[5]);
    signal {binary} transcriptN2b_18[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[6]);
    signal {binary} transcriptN2b_19[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[7]);
    signal {binary} transcriptN2b_20[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[8]);
    signal {binary} transcriptN2b_21[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[9]);
    signal {binary} transcriptN2b_22[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[10]);
    signal {binary} transcriptN2b_23[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[11]);
    
    signal transcriptHash_friQueries_2[12] <== Poseidon2(12)([0,0,0,0,0,0,0,0], [transcriptHash_friQueries_1[0],transcriptHash_friQueries_1[1],transcriptHash_friQueries_1[2],transcriptHash_friQueries_1[3]]);
    signal {binary} transcriptN2b_24[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[0]);
    signal {binary} transcriptN2b_25[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[1]);
    signal {binary} transcriptN2b_26[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[2]);
    signal {binary} transcriptN2b_27[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[3]);
    signal {binary} transcriptN2b_28[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[4]);
    signal {binary} transcriptN2b_29[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[5]);
    signal {binary} transcriptN2b_30[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[6]);
    signal {binary} transcriptN2b_31[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[7]);
    signal {binary} transcriptN2b_32[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[8]);
    signal {binary} transcriptN2b_33[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[9]);
    signal {binary} transcriptN2b_34[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[10]);
    signal {binary} transcriptN2b_35[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[11]);
    
    signal transcriptHash_friQueries_3[12] <== Poseidon2(12)([0,0,0,0,0,0,0,0], [transcriptHash_friQueries_2[0],transcriptHash_friQueries_2[1],transcriptHash_friQueries_2[2],transcriptHash_friQueries_2[3]]);
    signal {binary} transcriptN2b_36[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[0]);
    signal {binary} transcriptN2b_37[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[1]);
    signal {binary} transcriptN2b_38[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[2]);
    signal {binary} transcriptN2b_39[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[3]);
    signal {binary} transcriptN2b_40[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[4]);
    signal {binary} transcriptN2b_41[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[5]);
    signal {binary} transcriptN2b_42[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[6]);
    signal {binary} transcriptN2b_43[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[7]);
    signal {binary} transcriptN2b_44[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[8]);
    signal {binary} transcriptN2b_45[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[9]);
    signal {binary} transcriptN2b_46[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[10]);
    for(var i = 11; i < 12; i++){
        _ <== transcriptHash_friQueries_3[i]; // Unused transcript values        
    }

    // From each transcript hash converted to bits, we assign those bits to queriesFRI[q] to define the query positions
    var q = 0; // Query number 
    var b = 0; // Bit number 
    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_0[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_0[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_1[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_1[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_2[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_2[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_3[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_3[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_4[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_4[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_5[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_5[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_6[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_6[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_7[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_7[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_8[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_8[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_9[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_9[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_10[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_10[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_11[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_11[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_12[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_12[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_13[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_14[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_14[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_15[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_15[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_16[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_16[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_17[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_17[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_18[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_18[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_19[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_19[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_20[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_20[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_21[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_21[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_22[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_22[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_23[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_23[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_24[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_24[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_25[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_25[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_26[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_26[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_27[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_27[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_28[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_28[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_29[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_29[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_30[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_30[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_31[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_31[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_32[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_32[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_33[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_33[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_34[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_34[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_35[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_35[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_36[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_36[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_37[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_37[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_38[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_38[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_39[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_39[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_40[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_40[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_41[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_41[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_42[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_42[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_43[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_43[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_44[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_44[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_45[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_45[63]; // Unused last bit

    for(var j = 0; j < 46; j++) {
        queriesFRI[q][b] <== transcriptN2b_46[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    for(var j = 46; j < 64; j++) {
        _ <== transcriptN2b_46[j]; // Unused bits        
    }
}


/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input globalChallenge[3]; 

    signal input airValues[3][3];
    
    signal input root2[4];
                  
    signal input root3[4];
    signal input evals[20][3]; 
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];
    signal input finalPol[32][3];
    
    signal output challengesStage2[2][3];

    signal output challengeQ[3];
    signal output challengeXi[3];
    signal output challengesFRI[2][3];
    signal output challengesFRISteps[7][3];
    signal output {binary} queriesFRI[128][23];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];


    
    signal transcriptHash_0[12] <== Poseidon2(12)([globalChallenge[0],globalChallenge[1],globalChallenge[2],0,0,0,0,0], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for(var i = 6; i < 12; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_1[12] <== Poseidon2(12)([root2[0],root2[1],root2[2],root2[3],airValues[2][0],airValues[2][1],airValues[2][2],0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_2[12] <== Poseidon2(12)([root3[0],root3[1],root3[2],root3[3],0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeXi <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];
    
    signal transcriptHash_evals_0[12] <== Poseidon2(12)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1]], [0,0,0,0]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_1[12] <== Poseidon2(12)([evals[2][2],evals[3][0],evals[3][1],evals[3][2],evals[4][0],evals[4][1],evals[4][2],evals[5][0]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_2[12] <== Poseidon2(12)([evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_3[12] <== Poseidon2(12)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_4[12] <== Poseidon2(12)([evals[10][2],evals[11][0],evals[11][1],evals[11][2],evals[12][0],evals[12][1],evals[12][2],evals[13][0]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_5[12] <== Poseidon2(12)([evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_6[12] <== Poseidon2(12)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1]], [transcriptHash_evals_5[0],transcriptHash_evals_5[1],transcriptHash_evals_5[2],transcriptHash_evals_5[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_7[12] <== Poseidon2(12)([evals[18][2],evals[19][0],evals[19][1],evals[19][2],0,0,0,0], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    evalsHash <== [transcriptHash_evals_7[0], transcriptHash_evals_7[1], transcriptHash_evals_7[2], transcriptHash_evals_7[3]];
    
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_3[12] <== Poseidon2(12)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesFRI[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesFRI[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    challengesFRISteps[0] <== [transcriptHash_3[6], transcriptHash_3[7], transcriptHash_3[8]];
    for(var i = 9; i < 12; i++){
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_4[12] <== Poseidon2(12)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    challengesFRISteps[1] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_5[12] <== Poseidon2(12)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    challengesFRISteps[2] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_6[12] <== Poseidon2(12)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    challengesFRISteps[3] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_7[12] <== Poseidon2(12)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    challengesFRISteps[4] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_8[12] <== Poseidon2(12)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    challengesFRISteps[5] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];
    
    signal transcriptHash_lastPolFRI_0[12] <== Poseidon2(12)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1]], [0,0,0,0]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_1[12] <== Poseidon2(12)([finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2],finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_2[12] <== Poseidon2(12)([finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_3[12] <== Poseidon2(12)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_4[12] <== Poseidon2(12)([finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2],finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_5[12] <== Poseidon2(12)([finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_6[12] <== Poseidon2(12)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_7[12] <== Poseidon2(12)([finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2],finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_8[12] <== Poseidon2(12)([finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_7[0],transcriptHash_lastPolFRI_7[1],transcriptHash_lastPolFRI_7[2],transcriptHash_lastPolFRI_7[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_9[12] <== Poseidon2(12)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1]], [transcriptHash_lastPolFRI_8[0],transcriptHash_lastPolFRI_8[1],transcriptHash_lastPolFRI_8[2],transcriptHash_lastPolFRI_8[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_9[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_10[12] <== Poseidon2(12)([finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2],finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0]], [transcriptHash_lastPolFRI_9[0],transcriptHash_lastPolFRI_9[1],transcriptHash_lastPolFRI_9[2],transcriptHash_lastPolFRI_9[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_lastPolFRI_10[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_11[12] <== Poseidon2(12)([finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_10[0],transcriptHash_lastPolFRI_10[1],transcriptHash_lastPolFRI_10[2],transcriptHash_lastPolFRI_10[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_11[0], transcriptHash_lastPolFRI_11[1], transcriptHash_lastPolFRI_11[2], transcriptHash_lastPolFRI_11[3]];

    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_9[12] <== Poseidon2(12)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    challengesFRISteps[6] <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];

    queriesFRI <== calculateFRIQueries0()(challengesFRISteps[6]);
}

/*
    Verify that FRI polynomials are built properly
*/
template VerifyFRI0(nBitsExt, prevStepBits, currStepBits, nextStepBits, e0) {
    var nextStep = currStepBits - nextStepBits; 
    var step = prevStepBits - currStepBits;

    signal input {binary} queriesFRI[currStepBits];
    signal input friChallenge[3];
    signal input s_vals_curr[1<< step][3];
    signal input s_vals_next[1<< nextStep][3];
    signal input {binary} enable;

    signal sx[currStepBits];
    
    sx[0] <==  e0 *( queriesFRI[0] * (invroots(prevStepBits) -1) + 1);
    for (var i=1; i< currStepBits; i++) {
        sx[i] <== sx[i-1] *  ( queriesFRI[i] * (invroots(prevStepBits -i) -1) +1);
    }
        
    // Perform an IFFT to obtain the coefficients of the polynomial given s_vals and evaluate it 
    signal coefs[1 << step][3] <== FFT(step, 3, 1)(s_vals_curr);
    signal evalXprime[3] <== [friChallenge[0] *  sx[currStepBits - 1], friChallenge[1] * sx[currStepBits - 1], friChallenge[2] *  sx[currStepBits - 1]];
    signal evalPol[3] <== EvalPol(1 << step)(coefs, evalXprime);

    signal {binary} keys_lowValues[nextStep];
    for(var i = 0; i < nextStep; i++) { keys_lowValues[i] <== queriesFRI[i + nextStepBits]; } 
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s_vals_next, keys_lowValues);

    enable * (lowValues[0] - evalPol[0]) === 0;
    enable * (lowValues[1] - evalPol[1]) === 0;
    enable * (lowValues[2] - evalPol[2]) === 0;
}

/* 
    Verify that all committed polynomials are calculated correctly
*/

template VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[20][3];
        signal input publics[8];
        signal input airgroupvalues[1][3];
    signal input airvalues[3][3];
    signal input proofvalues[2][3];
    signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeXi) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[22][3];
    for (var i=0; i< 22 ; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeXi, challengeXi);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[21][0] - 1, zMul[21][1], zMul[21][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_65[3] <== [evals[8][0] - publics[1], evals[8][1], evals[8][2]];
    signal tmp_66[3] <== CMul()(evals[2], tmp_65);
    signal tmp_67[3] <== [evals[12][0] - publics[2], evals[12][1], evals[12][2]];
    signal tmp_68[3] <== CMul()(evals[2], tmp_67);
    signal tmp_69[3] <== [evals[12][0] - publics[3], evals[12][1], evals[12][2]];
    signal tmp_70[3] <== CMul()(evals[3], tmp_69);
    signal tmp_71[3] <== [evals[9][0] - evals[12][0], evals[9][1] - evals[12][1], evals[9][2] - evals[12][2]];
    signal tmp_72[3] <== [1 - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_73[3] <== CMul()(tmp_71, tmp_72);
    signal tmp_74[3] <== [evals[10][0] - evals[13][0], evals[10][1] - evals[13][1], evals[10][2] - evals[13][2]];
    signal tmp_75[3] <== [1 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_76[3] <== CMul()(tmp_74, tmp_75);
    signal tmp_77[3] <== [evals[11][0] - evals[14][0], evals[11][1] - evals[14][1], evals[11][2] - evals[14][2]];
    signal tmp_78[3] <== [1 - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_79[3] <== CMul()(tmp_77, tmp_78);
    signal tmp_80 <== publics[0] * proofvalues[0][0];
    signal tmp_81 <== tmp_80 - proofvalues[1][0];
    signal tmp_82 <== 2 * airvalues[0][0];
    signal tmp_83 <== tmp_82 - airvalues[1][0];
    signal tmp_84[3] <== [evals[1][0] + 1, evals[1][1], evals[1][2]];
    signal tmp_85[3] <== [evals[0][0] - tmp_84[0], evals[0][1] - tmp_84[1], evals[0][2] - tmp_84[2]];
    signal tmp_86[3] <== CMul()(evals[17], evals[18]);
    signal tmp_87[3] <== [1 - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_88[3] <== [0 - tmp_87[0], -tmp_87[1], -tmp_87[2]];
    signal tmp_89[3] <== [tmp_86[0] - tmp_88[0], tmp_86[1] - tmp_88[1], tmp_86[2] - tmp_88[2]];
    signal tmp_90[3] <== [1 - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_91[3] <== CMul()(evals[15], tmp_90);
    signal tmp_92[3] <== [evals[16][0] - tmp_91[0], evals[16][1] - tmp_91[1], evals[16][2] - tmp_91[2]];
    signal tmp_93[3] <== [tmp_92[0] - evals[17][0], tmp_92[1] - evals[17][1], tmp_92[2] - evals[17][2]];
    signal tmp_94[3] <== [airgroupvalues[0][0] - evals[16][0], airgroupvalues[0][1] - evals[16][1], airgroupvalues[0][2] - evals[16][2]];
    signal tmp_95[3] <== CMul()(evals[7], tmp_94);
    signal tmp_96[3] <== CMul()(evals[13], challengesStage2[0]);
    signal tmp_97[3] <== CMul()(evals[8], evals[8]);
    signal tmp_98[3] <== CMul()(evals[12], evals[12]);
    signal tmp_99[3] <== [tmp_97[0] + tmp_98[0], tmp_97[1] + tmp_98[1], tmp_97[2] + tmp_98[2]];
    signal tmp_100[3] <== [tmp_96[0] + tmp_99[0], tmp_96[1] + tmp_99[1], tmp_96[2] + tmp_99[2]];
    signal tmp_101[3] <== CMul()(tmp_100, challengesStage2[0]);
    signal tmp_102[3] <== [tmp_101[0] + 1, tmp_101[1], tmp_101[2]];
    signal tmp_38[3] <== CMul()(challengeQ, tmp_66);
    signal tmp_39[3] <== [tmp_38[0] + tmp_68[0], tmp_38[1] + tmp_68[1], tmp_38[2] + tmp_68[2]];
    signal tmp_40[3] <== CMul()(challengeQ, tmp_39);
    signal tmp_41[3] <== [tmp_40[0] + tmp_70[0], tmp_40[1] + tmp_70[1], tmp_40[2] + tmp_70[2]];
    signal tmp_42[3] <== CMul()(challengeQ, tmp_41);
    signal tmp_43[3] <== [tmp_42[0] + tmp_73[0], tmp_42[1] + tmp_73[1], tmp_42[2] + tmp_73[2]];
    signal tmp_44[3] <== CMul()(challengeQ, tmp_43);
    signal tmp_45[3] <== [tmp_44[0] + tmp_76[0], tmp_44[1] + tmp_76[1], tmp_44[2] + tmp_76[2]];
    signal tmp_46[3] <== CMul()(challengeQ, tmp_45);
    signal tmp_47[3] <== [tmp_46[0] + tmp_79[0], tmp_46[1] + tmp_79[1], tmp_46[2] + tmp_79[2]];
    signal tmp_48[3] <== CMul()(challengeQ, tmp_47);
    signal tmp_49[3] <== [tmp_48[0] + tmp_81, tmp_48[1], tmp_48[2]];
    signal tmp_50[3] <== CMul()(challengeQ, tmp_49);
    signal tmp_51[3] <== [tmp_50[0] + tmp_83, tmp_50[1], tmp_50[2]];
    signal tmp_52[3] <== CMul()(challengeQ, tmp_51);
    signal tmp_53[3] <== [tmp_52[0] + tmp_85[0], tmp_52[1] + tmp_85[1], tmp_52[2] + tmp_85[2]];
    signal tmp_54[3] <== CMul()(challengeQ, tmp_53);
    signal tmp_55[3] <== [tmp_54[0] + tmp_89[0], tmp_54[1] + tmp_89[1], tmp_54[2] + tmp_89[2]];
    signal tmp_56[3] <== CMul()(challengeQ, tmp_55);
    signal tmp_57[3] <== [tmp_56[0] + tmp_93[0], tmp_56[1] + tmp_93[1], tmp_56[2] + tmp_93[2]];
    signal tmp_58[3] <== CMul()(challengeQ, tmp_57);
    signal tmp_59[3] <== [tmp_58[0] + tmp_95[0], tmp_58[1] + tmp_95[1], tmp_58[2] + tmp_95[2]];
    signal tmp_60[3] <== CMul()(challengeQ, tmp_59);
    signal tmp_61[3] <== [tmp_102[0] + challengesStage2[1][0], tmp_102[1] + challengesStage2[1][1], tmp_102[2] + challengesStage2[1][2]];
    signal tmp_62[3] <== [evals[18][0] - tmp_61[0], evals[18][1] - tmp_61[1], evals[18][2] - tmp_61[2]];
    signal tmp_63[3] <== [tmp_60[0] + tmp_62[0], tmp_60[1] + tmp_62[1], tmp_60[2] + tmp_62[2]];
    signal tmp_103[3] <== CMul()(tmp_63, Zh);

    signal xAcc[1][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[0][3]; // Stores the evaluations of Q_i
    signal qAcc[1][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 1; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[19+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[21]);
            qStep[i-1] <== CMul()(xAcc[i], evals[19+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i, which is stored at tmp_103 
    enable * (tmp_103[0] - qAcc[0][0]) === 0;
    enable * (tmp_103[1] - qAcc[0][1]) === 0;
    enable * (tmp_103[2] - qAcc[0][2]) === 0;
}

/*  Calculate FRI polinomial */
template CalculateFRIPolValue0() {
    signal input {binary} queriesFRI[23];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[20][3];
 
    signal input cm1[2];
 
    signal input cm2[9];
    signal input cm3[3];
    signal input consts[2];
    signal input custom_rom_0[2];
    
    signal output queryVals[3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
 
    mapValues.vals1 <== cm1;
 
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;
    mapValues.vals_rom_0 <== custom_rom_0;
    signal xacc[23];
    xacc[0] <== queriesFRI[0]*(7 * roots(23)-7) + 7;
    for (var i=1; i<23; i++) {
        xacc[i] <== xacc[i-1] * ( queriesFRI[i]*(roots(23 - i) - 1) +1);
    }

    signal xDivXSubXi[5][3];

    signal den0inv[3] <== CInv()([xacc[22] - 10420286214021487819 * challengeXi[0], - 10420286214021487819 * challengeXi[1], - 10420286214021487819 * challengeXi[2]]);
    xDivXSubXi[0] <== [xacc[22] * den0inv[0], xacc[22] * den0inv[1],  xacc[22] * den0inv[2]];
    signal den1inv[3] <== CInv()([xacc[22] - 1 * challengeXi[0], - 1 * challengeXi[1], - 1 * challengeXi[2]]);
    xDivXSubXi[1] <== [xacc[22] * den1inv[0], xacc[22] * den1inv[1],  xacc[22] * den1inv[2]];
    signal den2inv[3] <== CInv()([xacc[22] - 8124823329697072476 * challengeXi[0], - 8124823329697072476 * challengeXi[1], - 8124823329697072476 * challengeXi[2]]);
    xDivXSubXi[2] <== [xacc[22] * den2inv[0], xacc[22] * den2inv[1],  xacc[22] * den2inv[2]];
    signal den3inv[3] <== CInv()([xacc[22] - 6553637399136210105 * challengeXi[0], - 6553637399136210105 * challengeXi[1], - 6553637399136210105 * challengeXi[2]]);
    xDivXSubXi[3] <== [xacc[22] * den3inv[0], xacc[22] * den3inv[1],  xacc[22] * den3inv[2]];
    signal den4inv[3] <== CInv()([xacc[22] - 331116024603048646 * challengeXi[0], - 331116024603048646 * challengeXi[1], - 331116024603048646 * challengeXi[2]]);
    xDivXSubXi[4] <== [xacc[22] * den4inv[0], xacc[22] * den4inv[1],  xacc[22] * den4inv[2]];

    signal tmp_0[3] <== [mapValues.custom_rom_0_0 - evals[0][0], -evals[0][1], -evals[0][2]];
    signal tmp_1[3] <== CMul()(tmp_0, challengesFRI[1]);
    signal tmp_2[3] <== [mapValues.custom_rom_0_1 - evals[1][0], -evals[1][1], -evals[1][2]];
    signal tmp_3[3] <== [tmp_1[0] + tmp_2[0], tmp_1[1] + tmp_2[1], tmp_1[2] + tmp_2[2]];
    signal tmp_4[3] <== CMul()(tmp_3, challengesFRI[1]);
    signal tmp_5[3] <== [consts[0] - evals[2][0], -evals[2][1], -evals[2][2]];
    signal tmp_6[3] <== [tmp_4[0] + tmp_5[0], tmp_4[1] + tmp_5[1], tmp_4[2] + tmp_5[2]];
    signal tmp_7[3] <== CMul()(tmp_6, challengesFRI[1]);
    signal tmp_8[3] <== [consts[1] - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_9[3] <== [tmp_7[0] + tmp_8[0], tmp_7[1] + tmp_8[1], tmp_7[2] + tmp_8[2]];
    signal tmp_10[3] <== CMul()(tmp_9, challengesFRI[1]);
    signal tmp_11[3] <== [mapValues.cm1_0 - evals[8][0], -evals[8][1], -evals[8][2]];
    signal tmp_12[3] <== [tmp_10[0] + tmp_11[0], tmp_10[1] + tmp_11[1], tmp_10[2] + tmp_11[2]];
    signal tmp_13[3] <== CMul()(tmp_12, challengesFRI[1]);
    signal tmp_14[3] <== [mapValues.cm1_1 - evals[12][0], -evals[12][1], -evals[12][2]];
    signal tmp_15[3] <== [tmp_13[0] + tmp_14[0], tmp_13[1] + tmp_14[1], tmp_13[2] + tmp_14[2]];
    signal tmp_16[3] <== CMul()(tmp_15, challengesFRI[1]);
    signal tmp_17[3] <== [mapValues.cm2_0[0] - evals[16][0], mapValues.cm2_0[1] - evals[16][1], mapValues.cm2_0[2] - evals[16][2]];
    signal tmp_18[3] <== [tmp_16[0] + tmp_17[0], tmp_16[1] + tmp_17[1], tmp_16[2] + tmp_17[2]];
    signal tmp_19[3] <== CMul()(tmp_18, challengesFRI[1]);
    signal tmp_20[3] <== [mapValues.cm2_1[0] - evals[17][0], mapValues.cm2_1[1] - evals[17][1], mapValues.cm2_1[2] - evals[17][2]];
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_21, challengesFRI[1]);
    signal tmp_23[3] <== [mapValues.cm2_2[0] - evals[18][0], mapValues.cm2_2[1] - evals[18][1], mapValues.cm2_2[2] - evals[18][2]];
    signal tmp_24[3] <== [tmp_22[0] + tmp_23[0], tmp_22[1] + tmp_23[1], tmp_22[2] + tmp_23[2]];
    signal tmp_25[3] <== CMul()(tmp_24, challengesFRI[1]);
    signal tmp_26[3] <== [mapValues.cm3_0[0] - evals[19][0], mapValues.cm3_0[1] - evals[19][1], mapValues.cm3_0[2] - evals[19][2]];
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_27, xDivXSubXi[1]);
    signal tmp_29[3] <== CMul()(challengesFRI[0], tmp_28);
    signal tmp_30[3] <== [consts[0] - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_31[3] <== CMul()(tmp_30, challengesFRI[1]);
    signal tmp_32[3] <== [consts[1] - evals[7][0], -evals[7][1], -evals[7][2]];
    signal tmp_33[3] <== [tmp_31[0] + tmp_32[0], tmp_31[1] + tmp_32[1], tmp_31[2] + tmp_32[2]];
    signal tmp_34[3] <== CMul()(tmp_33, challengesFRI[1]);
    signal tmp_35[3] <== [mapValues.cm1_0 - evals[9][0], -evals[9][1], -evals[9][2]];
    signal tmp_36[3] <== [tmp_34[0] + tmp_35[0], tmp_34[1] + tmp_35[1], tmp_34[2] + tmp_35[2]];
    signal tmp_37[3] <== CMul()(tmp_36, challengesFRI[1]);
    signal tmp_38[3] <== [mapValues.cm1_1 - evals[13][0], -evals[13][1], -evals[13][2]];
    signal tmp_39[3] <== [tmp_37[0] + tmp_38[0], tmp_37[1] + tmp_38[1], tmp_37[2] + tmp_38[2]];
    signal tmp_40[3] <== CMul()(tmp_39, xDivXSubXi[2]);
    signal tmp_41[3] <== [tmp_29[0] + tmp_40[0], tmp_29[1] + tmp_40[1], tmp_29[2] + tmp_40[2]];
    signal tmp_42[3] <== CMul()(challengesFRI[0], tmp_41);
    signal tmp_43[3] <== [consts[0] - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_44[3] <== CMul()(tmp_43, challengesFRI[1]);
    signal tmp_45[3] <== [mapValues.cm1_0 - evals[10][0], -evals[10][1], -evals[10][2]];
    signal tmp_46[3] <== [tmp_44[0] + tmp_45[0], tmp_44[1] + tmp_45[1], tmp_44[2] + tmp_45[2]];
    signal tmp_47[3] <== CMul()(tmp_46, challengesFRI[1]);
    signal tmp_48[3] <== [mapValues.cm1_1 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_49[3] <== [tmp_47[0] + tmp_48[0], tmp_47[1] + tmp_48[1], tmp_47[2] + tmp_48[2]];
    signal tmp_50[3] <== CMul()(tmp_49, xDivXSubXi[3]);
    signal tmp_51[3] <== [tmp_42[0] + tmp_50[0], tmp_42[1] + tmp_50[1], tmp_42[2] + tmp_50[2]];
    signal tmp_52[3] <== CMul()(challengesFRI[0], tmp_51);
    signal tmp_53[3] <== [consts[0] - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_54[3] <== CMul()(tmp_53, challengesFRI[1]);
    signal tmp_55[3] <== [mapValues.cm1_0 - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_56[3] <== [tmp_54[0] + tmp_55[0], tmp_54[1] + tmp_55[1], tmp_54[2] + tmp_55[2]];
    signal tmp_57[3] <== CMul()(tmp_56, xDivXSubXi[4]);
    signal tmp_58[3] <== [tmp_52[0] + tmp_57[0], tmp_52[1] + tmp_57[1], tmp_52[2] + tmp_57[2]];
    signal tmp_59[3] <== CMul()(challengesFRI[0], tmp_58);
    signal tmp_60[3] <== [mapValues.cm2_0[0] - evals[15][0], mapValues.cm2_0[1] - evals[15][1], mapValues.cm2_0[2] - evals[15][2]];
    signal tmp_61[3] <== CMul()(tmp_60, xDivXSubXi[0]);
    signal tmp_63[3] <== [tmp_59[0] + tmp_61[0], tmp_59[1] + tmp_61[1], tmp_59[2] + tmp_61[2]];

    queryVals[0] <== tmp_63[0];
    queryVals[1] <== tmp_63[1];
    queryVals[2] <== tmp_63[2];
}

/* 
    Verify that the initial FRI polynomial, which is the lineal combination of the committed polynomials
    during the STARK phases, is built properly
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesFRI[23];
    signal input queryVals[3];
    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;
    
    signal {binary} s0_keys_lowValues[nextStep];
    for(var i = 0; i < nextStep; i++) {
        s0_keys_lowValues[i] <== queriesFRI[i + nextStepBits];
    }

    for(var i = 0; i < nextStepBits; i++) {
        _ <== queriesFRI[i];
    }
   
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s1_vals, s0_keys_lowValues);

    enable * (lowValues[0] - queryVals[0]) === 0;
    enable * (lowValues[1] - queryVals[1]) === 0;
    enable * (lowValues[2] - queryVals[2]) === 0;
}

// Polynomials can either have dimension 1 (if they are defined in the base field) or dimension 3 (if they are defined in the 
// extended field). In general, all initial polynomials (constants and tr) will have dim 1 and the other ones such as Z (grand product),
// Q (quotient) or h_i (plookup) will have dim 3.
// This function processes the values, which are stored in an array vals[n] and splits them in multiple signals of size 1 (vals_i) 
// or 3 (vals_i[3]) depending on its dimension.
template MapValues0() {
 
    signal input vals1[2];
 
    signal input vals2[9];
    signal input vals3[3];
    signal input vals_rom_0[2];
    signal output cm1_0;
    signal output cm1_1;
    signal output cm2_0[3];
    signal output cm2_1[3];
    signal output cm2_2[3];
    signal output cm3_0[3];
    signal output custom_rom_0_0;
    signal output custom_rom_0_1;

    custom_rom_0_0 <== vals_rom_0[0];
    custom_rom_0_1 <== vals_rom_0[1];

    cm1_0 <== vals1[0];
    cm1_1 <== vals1[1];
    cm2_0 <== [vals2[0],vals2[1] , vals2[2]];
    cm2_1 <== [vals2[3],vals2[4] , vals2[5]];
    cm2_2 <== [vals2[6],vals2[7] , vals2[8]];
    cm3_0 <== [vals3[0],vals3[1] , vals3[2]];
}

template VerifyFinalPol0() {
    ///////
    // Check Degree last pol
    ///////
    signal input finalPol[32][3];
    signal input {binary} enable;
    
    // Calculate the IFFT to get the coefficients of finalPol 
    signal lastIFFT[32][3] <== FFT(5, 3, 1)(finalPol);

    // Check that the degree of the final polynomial is bounded by the degree defined in the last step of the folding
    for (var k= 16; k< 32; k++) {
        for (var e=0; e<3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k= 0; k < 16; k++) {
        _ <== lastIFFT[k];
    }
}

template StarkVerifier0() {
    signal input publics[8]; // publics polynomials
    signal input airgroupvalues[1][3]; // airgroupvalue values
    signal input airvalues[3][3]; // air values
    signal input proofvalues[2][3]; // air values
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials

    signal output rootC[4] <== [6937688178068921927,15008864101305609067,1549094540851172845,7448837168161214622 ]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[20][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
 
    signal input s0_vals1[128][2];
 
    signal input s0_vals2[128][9];
                                       
    signal input s0_vals3[128][3];
    signal input s0_valsC[128][2];

    signal input s0_vals_rom_0[128][2];

    // Merkle proofs for each of the evaluations
 
    signal input s0_siblings1[128][15][8];
 
    signal input s0_siblings2[128][15][8];
 
    signal input s0_siblings3[128][15][8];
    signal input s0_siblingsC[128][15][8];
    signal input s0_siblings_rom_0[128][15][8];
    // Contains the root of the original polynomial and all the intermediate FRI polynomials except for the last step
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];

    // For each intermediate FRI polynomial and the last one, we store at vals the values needed to check the queries.
    // Given a query r,  the verifier needs b points to check it out, being b = 2^u, where u is the difference between two consecutive step
    // and the sibling paths for each query.
    signal input s1_vals[128][48];
    signal input s1_siblings[128][12][8];
    signal input s2_vals[128][48];
    signal input s2_siblings[128][10][8];
    signal input s3_vals[128][48];
    signal input s3_siblings[128][7][8];
    signal input s4_vals[128][24];
    signal input s4_siblings[128][6][8];
    signal input s5_vals[128][24];
    signal input s5_siblings[128][4][8];

    // Evaluations of the final FRI polynomial over a set of points of size bounded its degree
    signal input finalPol[32][3];

    signal {binary} enabled;
    enabled <== 1;

    signal input globalChallenge[3];

    signal queryVals[128][3];

    signal challengesStage2[2][3];

    signal challengeQ[3];
    signal challengeXi[3];
    signal challengesFRI[2][3];


    // challengesFRISteps contains the random value provided by the verifier at each step of the folding so that 
    // the prover can commit the polynomial.
    // Remember that, when folding, the prover does as follows: f0 = g_0 + X*g_1 + ... + (X^b)*g_b and then the 
    // verifier provides a random X so that the prover can commit it. This value is stored here.
    signal challengesFRISteps[7][3];

    // Challenges from which we derive all the queries
    signal {binary} queriesFRI[128][23];


    ///////////
    // Calculate challenges, challengesFRISteps and queriesFRI
    ///////////

 

    (challengesStage2,challengeQ,challengeXi,challengesFRI,challengesFRISteps,queriesFRI) <== Transcript0()(globalChallenge,airvalues,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,s5_root,finalPol);

    ///////////
    // Check constraints polynomial in the evaluation point
    ///////////

 

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, airgroupvalues, airvalues, proofvalues, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputs so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
 
    var s0_vals1_p[128][2][1];
 
    var s0_vals2_p[128][9][1];
 
    var s0_vals3_p[128][3][1];
    var s0_valsC_p[128][2][1];
    var s0_vals_rom_0_p[128][2][1];
    var s0_vals_p[128][1][3]; 
    var s1_vals_p[128][16][3]; 
    var s2_vals_p[128][16][3]; 
    var s3_vals_p[128][16][3]; 
    var s4_vals_p[128][8][3]; 
    var s5_vals_p[128][8][3]; 

    for (var q=0; q<128; q++) {
        // Preprocess vals for the initial FRI polynomial
 
        for (var i = 0; i < 2; i++) {
            s0_vals1_p[q][i][0] = s0_vals1[q][i];
        }
 
        for (var i = 0; i < 9; i++) {
            s0_vals2_p[q][i][0] = s0_vals2[q][i];
        }
 
        for (var i = 0; i < 3; i++) {
            s0_vals3_p[q][i][0] = s0_vals3[q][i];
        }
        for (var i = 0; i < 2; i++) {
            s0_valsC_p[q][i][0] = s0_valsC[q][i];
        }
    for (var i = 0; i < 2; i++) {
        s0_vals_rom_0_p[q][i][0] = s0_vals_rom_0[q][i];
    }

        // Preprocess vals for each folded polynomial
        for(var e=0; e < 3; e++) {
            for(var c=0; c < 16; c++) {
                s1_vals_p[q][c][e] = s1_vals[q][c*3+e];
            }
            for(var c=0; c < 16; c++) {
                s2_vals_p[q][c][e] = s2_vals[q][c*3+e];
            }
            for(var c=0; c < 16; c++) {
                s3_vals_p[q][c][e] = s3_vals[q][c*3+e];
            }
            for(var c=0; c < 8; c++) {
                s4_vals_p[q][c][e] = s4_vals[q][c*3+e];
            }
            for(var c=0; c < 8; c++) {
                s5_vals_p[q][c][e] = s5_vals[q][c*3+e];
            }
        }
    }
    
    ///////////
    // Verify Merkle Roots
    ///////////
    signal {binary} queriesFRITernary[128][15][2];
    signal queriesFRINum[128];
    for(var i = 0; i < 128; i++) {
        queriesFRINum[i] <== Bits2Num(23)(queriesFRI[i]);
        queriesFRITernary[i] <== Num2Ternary(15)(queriesFRINum[i]);
    }

    //Calculate merkle root for s0 vals
 
    for (var q=0; q<128; q++) {
        VerifyMerkleHash(1, 2, 3, 15)(s0_vals1_p[q], s0_siblings1[q], queriesFRITernary[q], root1, enabled);
    }
 
    for (var q=0; q<128; q++) {
        VerifyMerkleHash(1, 9, 3, 15)(s0_vals2_p[q], s0_siblings2[q], queriesFRITernary[q], root2, enabled);
    }

    for (var q=0; q<128; q++) {
        VerifyMerkleHash(1, 3, 3, 15)(s0_vals3_p[q], s0_siblings3[q], queriesFRITernary[q], root3, enabled);
    }

    for (var q=0; q<128; q++) {
        VerifyMerkleHash(1, 2, 3, 15)(s0_valsC_p[q], s0_siblingsC[q], queriesFRITernary[q], rootC, enabled);                                    
    }

    signal root_rom_0[4] <== [publics[4], publics[5], publics[6], publics[7]];
    for (var q=0; q<128; q++) {
        VerifyMerkleHash(1, 2, 3, 15)(s0_vals_rom_0_p[q], s0_siblings_rom_0[q], queriesFRITernary[q], root_rom_0, enabled);                                    
    }

    signal {binary} s1_keys_merkle[128][19];
    signal s1_keys_merkle_num[128];
    signal {binary} s1_keys_merkle_ternary[128][12][2];
    for (var q=0; q<128; q++) {
        // Calculate merkle root for s1 vals
        for(var i = 0; i < 19; i++) { s1_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s1_keys_merkle_num[q] <== Bits2Num(19)(s1_keys_merkle[q]);
        s1_keys_merkle_ternary[q] <== Num2Ternary(12)(s1_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 12)(s1_vals_p[q], s1_siblings[q], s1_keys_merkle_ternary[q], s1_root, enabled);
    }
    signal {binary} s2_keys_merkle[128][15];
    signal s2_keys_merkle_num[128];
    signal {binary} s2_keys_merkle_ternary[128][10][2];
    for (var q=0; q<128; q++) {
        // Calculate merkle root for s2 vals
        for(var i = 0; i < 15; i++) { s2_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s2_keys_merkle_num[q] <== Bits2Num(15)(s2_keys_merkle[q]);
        s2_keys_merkle_ternary[q] <== Num2Ternary(10)(s2_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 10)(s2_vals_p[q], s2_siblings[q], s2_keys_merkle_ternary[q], s2_root, enabled);
    }
    signal {binary} s3_keys_merkle[128][11];
    signal s3_keys_merkle_num[128];
    signal {binary} s3_keys_merkle_ternary[128][7][2];
    for (var q=0; q<128; q++) {
        // Calculate merkle root for s3 vals
        for(var i = 0; i < 11; i++) { s3_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s3_keys_merkle_num[q] <== Bits2Num(11)(s3_keys_merkle[q]);
        s3_keys_merkle_ternary[q] <== Num2Ternary(7)(s3_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 7)(s3_vals_p[q], s3_siblings[q], s3_keys_merkle_ternary[q], s3_root, enabled);
    }
    signal {binary} s4_keys_merkle[128][8];
    signal s4_keys_merkle_num[128];
    signal {binary} s4_keys_merkle_ternary[128][6][2];
    for (var q=0; q<128; q++) {
        // Calculate merkle root for s4 vals
        for(var i = 0; i < 8; i++) { s4_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s4_keys_merkle_num[q] <== Bits2Num(8)(s4_keys_merkle[q]);
        s4_keys_merkle_ternary[q] <== Num2Ternary(6)(s4_keys_merkle_num[q]);
        VerifyMerkleHash(3, 8, 3, 6)(s4_vals_p[q], s4_siblings[q], s4_keys_merkle_ternary[q], s4_root, enabled);
    }
    signal {binary} s5_keys_merkle[128][5];
    signal s5_keys_merkle_num[128];
    signal {binary} s5_keys_merkle_ternary[128][4][2];
    for (var q=0; q<128; q++) {
        // Calculate merkle root for s5 vals
        for(var i = 0; i < 5; i++) { s5_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s5_keys_merkle_num[q] <== Bits2Num(5)(s5_keys_merkle[q]);
        s5_keys_merkle_ternary[q] <== Num2Ternary(4)(s5_keys_merkle_num[q]);
        VerifyMerkleHash(3, 8, 3, 4)(s5_vals_p[q], s5_siblings[q], s5_keys_merkle_ternary[q], s5_root, enabled);
    }
        

    ///////////
    // Calculate FRI Polinomial
    ///////////
    
    for (var q=0; q<128; q++) {
        // Reconstruct FRI polinomial from evaluations
        queryVals[q] <== CalculateFRIPolValue0()(queriesFRI[q], challengeXi, challengesFRI, evals, s0_vals1[q], s0_vals2[q], s0_vals3[q], s0_valsC[q], s0_vals_rom_0[q]);
    }

    ///////////
    // Verify FRI Polinomial
    ///////////
    signal {binary} s1_queriesFRI[128][19];
    signal {binary} s2_queriesFRI[128][15];
    signal {binary} s3_queriesFRI[128][11];
    signal {binary} s4_queriesFRI[128][8];
    signal {binary} s5_queriesFRI[128][5];

    for (var q=0; q<128; q++) {
      
        // Verify that the query is properly constructed. This is done by checking that the linear combination of the set of 
        // polynomials committed during the different rounds evaluated at z matches with the commitment of the FRI polynomial
        VerifyQuery0(23, 19)(queriesFRI[q], queryVals[q], s1_vals_p[q], enabled);

        ///////////
        // Verify FRI construction
        ///////////

        // For each folding level we need to check that the polynomial is properly constructed
        // Remember that if the step between polynomials is b = 2^l, the next polynomial p_(i+1) will have degree deg(p_i) / b

        // Check S1
        for(var i = 0; i < 19; i++) { s1_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 23, 19, 15, 2635249152773512046)(s1_queriesFRI[q], challengesFRISteps[1], s1_vals_p[q], s2_vals_p[q], enabled);

        // Check S2
        for(var i = 0; i < 15; i++) { s2_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 19, 15, 11, 11131999729878195124)(s2_queriesFRI[q], challengesFRISteps[2], s2_vals_p[q], s3_vals_p[q], enabled);

        // Check S3
        for(var i = 0; i < 11; i++) { s3_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 15, 11, 8, 16627473974463641638)(s3_queriesFRI[q], challengesFRISteps[3], s3_vals_p[q], s4_vals_p[q], enabled);

        // Check S4
        for(var i = 0; i < 8; i++) { s4_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 11, 8, 5, 140704680260498080)(s4_queriesFRI[q], challengesFRISteps[4], s4_vals_p[q], s5_vals_p[q], enabled);

        // Check S5
        for(var i = 0; i < 5; i++) { s5_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 8, 5, 0, 10193707927880991676)(s5_queriesFRI[q], challengesFRISteps[5], s5_vals_p[q], finalPol, enabled);
    }

    VerifyFinalPol0()(finalPol, enabled);
}
    
