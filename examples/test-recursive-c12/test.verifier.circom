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
    signal output {binary} queriesFRI[43][20];


    
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
    for(var i = 2; i < 12; i++){
        _ <== transcriptHash_friQueries_1[i]; // Unused transcript values        
    }

    // From each transcript hash converted to bits, we assign those bits to queriesFRI[q] to define the query positions
    var q = 0; // Query number 
    var b = 0; // Bit number 
    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_0[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_0[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_1[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_1[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_2[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_2[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_3[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_3[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_4[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_4[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_5[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_5[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_6[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_6[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_7[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_7[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_8[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_8[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_9[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_9[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_10[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_10[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_11[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_11[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_12[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_12[63]; // Unused last bit

    for(var j = 0; j < 41; j++) {
        queriesFRI[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    for(var j = 41; j < 64; j++) {
        _ <== transcriptN2b_13[j]; // Unused bits        
    }
}


/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input publics[37];
    signal input rootC[4];
    signal input root1[4];

    
    signal input root2[4];
                  
    signal input root3[4];
    signal input evals[136][3]; 
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input finalPol[32][3];
    
    signal output challengesStage2[2][3];

    signal output challengeQ[3];
    signal output challengeXi[3];
    signal output challengesFRI[2][3];
    signal output challengesFRISteps[6][3];
    signal output {binary} queriesFRI[43][20];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];


    
    signal transcriptHash_publics_0[12] <== Poseidon2(12)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7]], [0,0,0,0]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_publics_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_1[12] <== Poseidon2(12)([publics[8],publics[9],publics[10],publics[11],publics[12],publics[13],publics[14],publics[15]], [transcriptHash_publics_0[0],transcriptHash_publics_0[1],transcriptHash_publics_0[2],transcriptHash_publics_0[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_publics_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_2[12] <== Poseidon2(12)([publics[16],publics[17],publics[18],publics[19],publics[20],publics[21],publics[22],publics[23]], [transcriptHash_publics_1[0],transcriptHash_publics_1[1],transcriptHash_publics_1[2],transcriptHash_publics_1[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_publics_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_3[12] <== Poseidon2(12)([publics[24],publics[25],publics[26],publics[27],publics[28],publics[29],publics[30],publics[31]], [transcriptHash_publics_2[0],transcriptHash_publics_2[1],transcriptHash_publics_2[2],transcriptHash_publics_2[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_publics_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_4[12] <== Poseidon2(12)([publics[32],publics[33],publics[34],publics[35],publics[36],0,0,0], [transcriptHash_publics_3[0],transcriptHash_publics_3[1],transcriptHash_publics_3[2],transcriptHash_publics_3[3]]);
    publicsHash <== [transcriptHash_publics_4[0], transcriptHash_publics_4[1], transcriptHash_publics_4[2], transcriptHash_publics_4[3]];

    
    signal transcriptHash_0[12] <== Poseidon2(12)([rootC[0],rootC[1],rootC[2],rootC[3],publicsHash[0],publicsHash[1],publicsHash[2],publicsHash[3]], [0,0,0,0]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_1[12] <== Poseidon2(12)([root1[0],root1[1],root1[2],root1[3],0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengesStage2[0] <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    challengesStage2[1] <== [transcriptHash_1[3], transcriptHash_1[4], transcriptHash_1[5]];
    for(var i = 6; i < 12; i++){
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_2[12] <== Poseidon2(12)([root2[0],root2[1],root2[2],root2[3],0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeQ <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_3[12] <== Poseidon2(12)([root3[0],root3[1],root3[2],root3[3],0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengeXi <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    
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
    
    signal transcriptHash_evals_7[12] <== Poseidon2(12)([evals[18][2],evals[19][0],evals[19][1],evals[19][2],evals[20][0],evals[20][1],evals[20][2],evals[21][0]], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_8[12] <== Poseidon2(12)([evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],evals[23][0],evals[23][1],evals[23][2]], [transcriptHash_evals_7[0],transcriptHash_evals_7[1],transcriptHash_evals_7[2],transcriptHash_evals_7[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_9[12] <== Poseidon2(12)([evals[24][0],evals[24][1],evals[24][2],evals[25][0],evals[25][1],evals[25][2],evals[26][0],evals[26][1]], [transcriptHash_evals_8[0],transcriptHash_evals_8[1],transcriptHash_evals_8[2],transcriptHash_evals_8[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_9[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_10[12] <== Poseidon2(12)([evals[26][2],evals[27][0],evals[27][1],evals[27][2],evals[28][0],evals[28][1],evals[28][2],evals[29][0]], [transcriptHash_evals_9[0],transcriptHash_evals_9[1],transcriptHash_evals_9[2],transcriptHash_evals_9[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_10[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_11[12] <== Poseidon2(12)([evals[29][1],evals[29][2],evals[30][0],evals[30][1],evals[30][2],evals[31][0],evals[31][1],evals[31][2]], [transcriptHash_evals_10[0],transcriptHash_evals_10[1],transcriptHash_evals_10[2],transcriptHash_evals_10[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_11[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_12[12] <== Poseidon2(12)([evals[32][0],evals[32][1],evals[32][2],evals[33][0],evals[33][1],evals[33][2],evals[34][0],evals[34][1]], [transcriptHash_evals_11[0],transcriptHash_evals_11[1],transcriptHash_evals_11[2],transcriptHash_evals_11[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_12[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_13[12] <== Poseidon2(12)([evals[34][2],evals[35][0],evals[35][1],evals[35][2],evals[36][0],evals[36][1],evals[36][2],evals[37][0]], [transcriptHash_evals_12[0],transcriptHash_evals_12[1],transcriptHash_evals_12[2],transcriptHash_evals_12[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_13[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_14[12] <== Poseidon2(12)([evals[37][1],evals[37][2],evals[38][0],evals[38][1],evals[38][2],evals[39][0],evals[39][1],evals[39][2]], [transcriptHash_evals_13[0],transcriptHash_evals_13[1],transcriptHash_evals_13[2],transcriptHash_evals_13[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_14[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_15[12] <== Poseidon2(12)([evals[40][0],evals[40][1],evals[40][2],evals[41][0],evals[41][1],evals[41][2],evals[42][0],evals[42][1]], [transcriptHash_evals_14[0],transcriptHash_evals_14[1],transcriptHash_evals_14[2],transcriptHash_evals_14[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_15[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_16[12] <== Poseidon2(12)([evals[42][2],evals[43][0],evals[43][1],evals[43][2],evals[44][0],evals[44][1],evals[44][2],evals[45][0]], [transcriptHash_evals_15[0],transcriptHash_evals_15[1],transcriptHash_evals_15[2],transcriptHash_evals_15[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_16[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_17[12] <== Poseidon2(12)([evals[45][1],evals[45][2],evals[46][0],evals[46][1],evals[46][2],evals[47][0],evals[47][1],evals[47][2]], [transcriptHash_evals_16[0],transcriptHash_evals_16[1],transcriptHash_evals_16[2],transcriptHash_evals_16[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_17[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_18[12] <== Poseidon2(12)([evals[48][0],evals[48][1],evals[48][2],evals[49][0],evals[49][1],evals[49][2],evals[50][0],evals[50][1]], [transcriptHash_evals_17[0],transcriptHash_evals_17[1],transcriptHash_evals_17[2],transcriptHash_evals_17[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_18[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_19[12] <== Poseidon2(12)([evals[50][2],evals[51][0],evals[51][1],evals[51][2],evals[52][0],evals[52][1],evals[52][2],evals[53][0]], [transcriptHash_evals_18[0],transcriptHash_evals_18[1],transcriptHash_evals_18[2],transcriptHash_evals_18[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_19[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_20[12] <== Poseidon2(12)([evals[53][1],evals[53][2],evals[54][0],evals[54][1],evals[54][2],evals[55][0],evals[55][1],evals[55][2]], [transcriptHash_evals_19[0],transcriptHash_evals_19[1],transcriptHash_evals_19[2],transcriptHash_evals_19[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_20[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_21[12] <== Poseidon2(12)([evals[56][0],evals[56][1],evals[56][2],evals[57][0],evals[57][1],evals[57][2],evals[58][0],evals[58][1]], [transcriptHash_evals_20[0],transcriptHash_evals_20[1],transcriptHash_evals_20[2],transcriptHash_evals_20[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_21[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_22[12] <== Poseidon2(12)([evals[58][2],evals[59][0],evals[59][1],evals[59][2],evals[60][0],evals[60][1],evals[60][2],evals[61][0]], [transcriptHash_evals_21[0],transcriptHash_evals_21[1],transcriptHash_evals_21[2],transcriptHash_evals_21[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_22[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_23[12] <== Poseidon2(12)([evals[61][1],evals[61][2],evals[62][0],evals[62][1],evals[62][2],evals[63][0],evals[63][1],evals[63][2]], [transcriptHash_evals_22[0],transcriptHash_evals_22[1],transcriptHash_evals_22[2],transcriptHash_evals_22[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_23[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_24[12] <== Poseidon2(12)([evals[64][0],evals[64][1],evals[64][2],evals[65][0],evals[65][1],evals[65][2],evals[66][0],evals[66][1]], [transcriptHash_evals_23[0],transcriptHash_evals_23[1],transcriptHash_evals_23[2],transcriptHash_evals_23[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_24[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_25[12] <== Poseidon2(12)([evals[66][2],evals[67][0],evals[67][1],evals[67][2],evals[68][0],evals[68][1],evals[68][2],evals[69][0]], [transcriptHash_evals_24[0],transcriptHash_evals_24[1],transcriptHash_evals_24[2],transcriptHash_evals_24[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_25[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_26[12] <== Poseidon2(12)([evals[69][1],evals[69][2],evals[70][0],evals[70][1],evals[70][2],evals[71][0],evals[71][1],evals[71][2]], [transcriptHash_evals_25[0],transcriptHash_evals_25[1],transcriptHash_evals_25[2],transcriptHash_evals_25[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_26[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_27[12] <== Poseidon2(12)([evals[72][0],evals[72][1],evals[72][2],evals[73][0],evals[73][1],evals[73][2],evals[74][0],evals[74][1]], [transcriptHash_evals_26[0],transcriptHash_evals_26[1],transcriptHash_evals_26[2],transcriptHash_evals_26[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_27[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_28[12] <== Poseidon2(12)([evals[74][2],evals[75][0],evals[75][1],evals[75][2],evals[76][0],evals[76][1],evals[76][2],evals[77][0]], [transcriptHash_evals_27[0],transcriptHash_evals_27[1],transcriptHash_evals_27[2],transcriptHash_evals_27[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_28[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_29[12] <== Poseidon2(12)([evals[77][1],evals[77][2],evals[78][0],evals[78][1],evals[78][2],evals[79][0],evals[79][1],evals[79][2]], [transcriptHash_evals_28[0],transcriptHash_evals_28[1],transcriptHash_evals_28[2],transcriptHash_evals_28[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_29[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_30[12] <== Poseidon2(12)([evals[80][0],evals[80][1],evals[80][2],evals[81][0],evals[81][1],evals[81][2],evals[82][0],evals[82][1]], [transcriptHash_evals_29[0],transcriptHash_evals_29[1],transcriptHash_evals_29[2],transcriptHash_evals_29[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_30[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_31[12] <== Poseidon2(12)([evals[82][2],evals[83][0],evals[83][1],evals[83][2],evals[84][0],evals[84][1],evals[84][2],evals[85][0]], [transcriptHash_evals_30[0],transcriptHash_evals_30[1],transcriptHash_evals_30[2],transcriptHash_evals_30[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_31[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_32[12] <== Poseidon2(12)([evals[85][1],evals[85][2],evals[86][0],evals[86][1],evals[86][2],evals[87][0],evals[87][1],evals[87][2]], [transcriptHash_evals_31[0],transcriptHash_evals_31[1],transcriptHash_evals_31[2],transcriptHash_evals_31[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_32[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_33[12] <== Poseidon2(12)([evals[88][0],evals[88][1],evals[88][2],evals[89][0],evals[89][1],evals[89][2],evals[90][0],evals[90][1]], [transcriptHash_evals_32[0],transcriptHash_evals_32[1],transcriptHash_evals_32[2],transcriptHash_evals_32[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_33[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_34[12] <== Poseidon2(12)([evals[90][2],evals[91][0],evals[91][1],evals[91][2],evals[92][0],evals[92][1],evals[92][2],evals[93][0]], [transcriptHash_evals_33[0],transcriptHash_evals_33[1],transcriptHash_evals_33[2],transcriptHash_evals_33[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_34[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_35[12] <== Poseidon2(12)([evals[93][1],evals[93][2],evals[94][0],evals[94][1],evals[94][2],evals[95][0],evals[95][1],evals[95][2]], [transcriptHash_evals_34[0],transcriptHash_evals_34[1],transcriptHash_evals_34[2],transcriptHash_evals_34[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_35[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_36[12] <== Poseidon2(12)([evals[96][0],evals[96][1],evals[96][2],evals[97][0],evals[97][1],evals[97][2],evals[98][0],evals[98][1]], [transcriptHash_evals_35[0],transcriptHash_evals_35[1],transcriptHash_evals_35[2],transcriptHash_evals_35[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_36[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_37[12] <== Poseidon2(12)([evals[98][2],evals[99][0],evals[99][1],evals[99][2],evals[100][0],evals[100][1],evals[100][2],evals[101][0]], [transcriptHash_evals_36[0],transcriptHash_evals_36[1],transcriptHash_evals_36[2],transcriptHash_evals_36[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_37[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_38[12] <== Poseidon2(12)([evals[101][1],evals[101][2],evals[102][0],evals[102][1],evals[102][2],evals[103][0],evals[103][1],evals[103][2]], [transcriptHash_evals_37[0],transcriptHash_evals_37[1],transcriptHash_evals_37[2],transcriptHash_evals_37[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_38[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_39[12] <== Poseidon2(12)([evals[104][0],evals[104][1],evals[104][2],evals[105][0],evals[105][1],evals[105][2],evals[106][0],evals[106][1]], [transcriptHash_evals_38[0],transcriptHash_evals_38[1],transcriptHash_evals_38[2],transcriptHash_evals_38[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_39[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_40[12] <== Poseidon2(12)([evals[106][2],evals[107][0],evals[107][1],evals[107][2],evals[108][0],evals[108][1],evals[108][2],evals[109][0]], [transcriptHash_evals_39[0],transcriptHash_evals_39[1],transcriptHash_evals_39[2],transcriptHash_evals_39[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_40[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_41[12] <== Poseidon2(12)([evals[109][1],evals[109][2],evals[110][0],evals[110][1],evals[110][2],evals[111][0],evals[111][1],evals[111][2]], [transcriptHash_evals_40[0],transcriptHash_evals_40[1],transcriptHash_evals_40[2],transcriptHash_evals_40[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_41[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_42[12] <== Poseidon2(12)([evals[112][0],evals[112][1],evals[112][2],evals[113][0],evals[113][1],evals[113][2],evals[114][0],evals[114][1]], [transcriptHash_evals_41[0],transcriptHash_evals_41[1],transcriptHash_evals_41[2],transcriptHash_evals_41[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_42[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_43[12] <== Poseidon2(12)([evals[114][2],evals[115][0],evals[115][1],evals[115][2],evals[116][0],evals[116][1],evals[116][2],evals[117][0]], [transcriptHash_evals_42[0],transcriptHash_evals_42[1],transcriptHash_evals_42[2],transcriptHash_evals_42[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_43[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_44[12] <== Poseidon2(12)([evals[117][1],evals[117][2],evals[118][0],evals[118][1],evals[118][2],evals[119][0],evals[119][1],evals[119][2]], [transcriptHash_evals_43[0],transcriptHash_evals_43[1],transcriptHash_evals_43[2],transcriptHash_evals_43[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_44[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_45[12] <== Poseidon2(12)([evals[120][0],evals[120][1],evals[120][2],evals[121][0],evals[121][1],evals[121][2],evals[122][0],evals[122][1]], [transcriptHash_evals_44[0],transcriptHash_evals_44[1],transcriptHash_evals_44[2],transcriptHash_evals_44[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_45[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_46[12] <== Poseidon2(12)([evals[122][2],evals[123][0],evals[123][1],evals[123][2],evals[124][0],evals[124][1],evals[124][2],evals[125][0]], [transcriptHash_evals_45[0],transcriptHash_evals_45[1],transcriptHash_evals_45[2],transcriptHash_evals_45[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_46[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_47[12] <== Poseidon2(12)([evals[125][1],evals[125][2],evals[126][0],evals[126][1],evals[126][2],evals[127][0],evals[127][1],evals[127][2]], [transcriptHash_evals_46[0],transcriptHash_evals_46[1],transcriptHash_evals_46[2],transcriptHash_evals_46[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_47[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_48[12] <== Poseidon2(12)([evals[128][0],evals[128][1],evals[128][2],evals[129][0],evals[129][1],evals[129][2],evals[130][0],evals[130][1]], [transcriptHash_evals_47[0],transcriptHash_evals_47[1],transcriptHash_evals_47[2],transcriptHash_evals_47[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_48[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_49[12] <== Poseidon2(12)([evals[130][2],evals[131][0],evals[131][1],evals[131][2],evals[132][0],evals[132][1],evals[132][2],evals[133][0]], [transcriptHash_evals_48[0],transcriptHash_evals_48[1],transcriptHash_evals_48[2],transcriptHash_evals_48[3]]);
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_evals_49[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_50[12] <== Poseidon2(12)([evals[133][1],evals[133][2],evals[134][0],evals[134][1],evals[134][2],evals[135][0],evals[135][1],evals[135][2]], [transcriptHash_evals_49[0],transcriptHash_evals_49[1],transcriptHash_evals_49[2],transcriptHash_evals_49[3]]);
    evalsHash <== [transcriptHash_evals_50[0], transcriptHash_evals_50[1], transcriptHash_evals_50[2], transcriptHash_evals_50[3]];
    
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_4[12] <== Poseidon2(12)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    challengesFRI[0] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    challengesFRI[1] <== [transcriptHash_4[3], transcriptHash_4[4], transcriptHash_4[5]];
    challengesFRISteps[0] <== [transcriptHash_4[6], transcriptHash_4[7], transcriptHash_4[8]];
    for(var i = 9; i < 12; i++){
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_5[12] <== Poseidon2(12)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    challengesFRISteps[1] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_6[12] <== Poseidon2(12)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    challengesFRISteps[2] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_7[12] <== Poseidon2(12)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    challengesFRISteps[3] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for(var i = 4; i < 12; i++){
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_8[12] <== Poseidon2(12)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    challengesFRISteps[4] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];
    
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
    challengesFRISteps[5] <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];

    queriesFRI <== calculateFRIQueries0()(challengesFRISteps[5]);
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
    signal input evals[136][3];
        signal input publics[37];
        signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeXi) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[17][3];
    for (var i=0; i< 17 ; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeXi, challengeXi);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[16][0] - 1, zMul[16][1], zMul[16][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_2966[3] <== [evals[0][0] + evals[1][0], evals[0][1] + evals[1][1], evals[0][2] + evals[1][2]];
    signal tmp_2967[3] <== [evals[57][0] + tmp_2966[0], evals[57][1] + tmp_2966[1], evals[57][2] + tmp_2966[2]];
    signal tmp_2968[3] <== [tmp_2967[0] + evals[107][0], tmp_2967[1] + evals[107][1], tmp_2967[2] + evals[107][2]];
    signal tmp_2969[3] <== [tmp_2968[0] + evals[51][0], tmp_2968[1] + evals[51][1], tmp_2968[2] + evals[51][2]];
    signal tmp_2970[3] <== [tmp_2969[0] + evals[108][0], tmp_2969[1] + evals[108][1], tmp_2969[2] + evals[108][2]];
    signal tmp_2971[3] <== CMul()(evals[60], evals[61]);
    signal tmp_2972[3] <== CMul()(evals[39], tmp_2971);
    signal tmp_2973[3] <== CMul()(evals[40], evals[60]);
    signal tmp_2974[3] <== [tmp_2972[0] + tmp_2973[0], tmp_2972[1] + tmp_2973[1], tmp_2972[2] + tmp_2973[2]];
    signal tmp_2975[3] <== CMul()(evals[41], evals[61]);
    signal tmp_2976[3] <== [tmp_2974[0] + tmp_2975[0], tmp_2974[1] + tmp_2975[1], tmp_2974[2] + tmp_2975[2]];
    signal tmp_2977[3] <== CMul()(evals[42], evals[62]);
    signal tmp_2978[3] <== [tmp_2976[0] + tmp_2977[0], tmp_2976[1] + tmp_2977[1], tmp_2976[2] + tmp_2977[2]];
    signal tmp_2979[3] <== [tmp_2978[0] + evals[43][0], tmp_2978[1] + evals[43][1], tmp_2978[2] + evals[43][2]];
    signal tmp_2980[3] <== CMul()(tmp_2970, tmp_2979);
    signal tmp_2981[3] <== CMul()(challengeQ, tmp_2980);
    signal tmp_2982[3] <== CMul()(evals[63], evals[64]);
    signal tmp_2983[3] <== CMul()(evals[39], tmp_2982);
    signal tmp_2984[3] <== CMul()(evals[40], evals[63]);
    signal tmp_2985[3] <== [tmp_2983[0] + tmp_2984[0], tmp_2983[1] + tmp_2984[1], tmp_2983[2] + tmp_2984[2]];
    signal tmp_2986[3] <== CMul()(evals[41], evals[64]);
    signal tmp_2987[3] <== [tmp_2985[0] + tmp_2986[0], tmp_2985[1] + tmp_2986[1], tmp_2985[2] + tmp_2986[2]];
    signal tmp_2988[3] <== CMul()(evals[42], evals[65]);
    signal tmp_2989[3] <== [tmp_2987[0] + tmp_2988[0], tmp_2987[1] + tmp_2988[1], tmp_2987[2] + tmp_2988[2]];
    signal tmp_2990[3] <== [tmp_2989[0] + evals[43][0], tmp_2989[1] + evals[43][1], tmp_2989[2] + evals[43][2]];
    signal tmp_2991[3] <== CMul()(tmp_2970, tmp_2990);
    signal tmp_2992[3] <== [tmp_2981[0] + tmp_2991[0], tmp_2981[1] + tmp_2991[1], tmp_2981[2] + tmp_2991[2]];
    signal tmp_2993[3] <== CMul()(challengeQ, tmp_2992);
    signal tmp_2994[3] <== CMul()(evals[66], evals[67]);
    signal tmp_2995[3] <== CMul()(evals[44], tmp_2994);
    signal tmp_2996[3] <== CMul()(evals[45], evals[66]);
    signal tmp_2997[3] <== [tmp_2995[0] + tmp_2996[0], tmp_2995[1] + tmp_2996[1], tmp_2995[2] + tmp_2996[2]];
    signal tmp_2998[3] <== CMul()(evals[46], evals[67]);
    signal tmp_2999[3] <== [tmp_2997[0] + tmp_2998[0], tmp_2997[1] + tmp_2998[1], tmp_2997[2] + tmp_2998[2]];
    signal tmp_3000[3] <== CMul()(evals[47], evals[68]);
    signal tmp_3001[3] <== [tmp_2999[0] + tmp_3000[0], tmp_2999[1] + tmp_3000[1], tmp_2999[2] + tmp_3000[2]];
    signal tmp_3002[3] <== [tmp_3001[0] + evals[48][0], tmp_3001[1] + evals[48][1], tmp_3001[2] + evals[48][2]];
    signal tmp_3003[3] <== CMul()(tmp_2970, tmp_3002);
    signal tmp_3004[3] <== [tmp_2993[0] + tmp_3003[0], tmp_2993[1] + tmp_3003[1], tmp_2993[2] + tmp_3003[2]];
    signal tmp_3005[3] <== CMul()(challengeQ, tmp_3004);
    signal tmp_3006[3] <== CMul()(evals[69], evals[70]);
    signal tmp_3007[3] <== CMul()(evals[44], tmp_3006);
    signal tmp_3008[3] <== CMul()(evals[45], evals[69]);
    signal tmp_3009[3] <== [tmp_3007[0] + tmp_3008[0], tmp_3007[1] + tmp_3008[1], tmp_3007[2] + tmp_3008[2]];
    signal tmp_3010[3] <== CMul()(evals[46], evals[70]);
    signal tmp_3011[3] <== [tmp_3009[0] + tmp_3010[0], tmp_3009[1] + tmp_3010[1], tmp_3009[2] + tmp_3010[2]];
    signal tmp_3012[3] <== CMul()(evals[47], evals[71]);
    signal tmp_3013[3] <== [tmp_3011[0] + tmp_3012[0], tmp_3011[1] + tmp_3012[1], tmp_3011[2] + tmp_3012[2]];
    signal tmp_3014[3] <== [tmp_3013[0] + evals[48][0], tmp_3013[1] + evals[48][1], tmp_3013[2] + evals[48][2]];
    signal tmp_3015[3] <== CMul()(tmp_2970, tmp_3014);
    signal tmp_3016[3] <== [tmp_3005[0] + tmp_3015[0], tmp_3005[1] + tmp_3015[1], tmp_3005[2] + tmp_3015[2]];
    signal tmp_3017[3] <== CMul()(challengeQ, tmp_3016);
    signal tmp_3018[3] <== [evals[57][0] + evals[107][0], evals[57][1] + evals[107][1], evals[57][2] + evals[107][2]];
    signal tmp_3019[3] <== CMul()(evals[72], evals[73]);
    signal tmp_3020[3] <== CMul()(evals[44], tmp_3019);
    signal tmp_3021[3] <== CMul()(evals[45], evals[72]);
    signal tmp_3022[3] <== [tmp_3020[0] + tmp_3021[0], tmp_3020[1] + tmp_3021[1], tmp_3020[2] + tmp_3021[2]];
    signal tmp_3023[3] <== CMul()(evals[46], evals[73]);
    signal tmp_3024[3] <== [tmp_3022[0] + tmp_3023[0], tmp_3022[1] + tmp_3023[1], tmp_3022[2] + tmp_3023[2]];
    signal tmp_3025[3] <== CMul()(evals[47], evals[74]);
    signal tmp_3026[3] <== [tmp_3024[0] + tmp_3025[0], tmp_3024[1] + tmp_3025[1], tmp_3024[2] + tmp_3025[2]];
    signal tmp_3027[3] <== [tmp_3026[0] + evals[48][0], tmp_3026[1] + evals[48][1], tmp_3026[2] + evals[48][2]];
    signal tmp_3028[3] <== CMul()(tmp_3018, tmp_3027);
    signal tmp_3029[3] <== [tmp_3017[0] + tmp_3028[0], tmp_3017[1] + tmp_3028[1], tmp_3017[2] + tmp_3028[2]];
    signal tmp_3030[3] <== CMul()(challengeQ, tmp_3029);
    signal tmp_3031[3] <== [evals[57][0] + evals[107][0], evals[57][1] + evals[107][1], evals[57][2] + evals[107][2]];
    signal tmp_3032[3] <== CMul()(evals[75], evals[76]);
    signal tmp_3033[3] <== CMul()(evals[44], tmp_3032);
    signal tmp_3034[3] <== CMul()(evals[45], evals[75]);
    signal tmp_3035[3] <== [tmp_3033[0] + tmp_3034[0], tmp_3033[1] + tmp_3034[1], tmp_3033[2] + tmp_3034[2]];
    signal tmp_3036[3] <== CMul()(evals[46], evals[76]);
    signal tmp_3037[3] <== [tmp_3035[0] + tmp_3036[0], tmp_3035[1] + tmp_3036[1], tmp_3035[2] + tmp_3036[2]];
    signal tmp_3038[3] <== CMul()(evals[47], evals[77]);
    signal tmp_3039[3] <== [tmp_3037[0] + tmp_3038[0], tmp_3037[1] + tmp_3038[1], tmp_3037[2] + tmp_3038[2]];
    signal tmp_3040[3] <== [tmp_3039[0] + evals[48][0], tmp_3039[1] + evals[48][1], tmp_3039[2] + evals[48][2]];
    signal tmp_3041[3] <== CMul()(tmp_3031, tmp_3040);
    signal tmp_3042[3] <== [tmp_3030[0] + tmp_3041[0], tmp_3030[1] + tmp_3041[1], tmp_3030[2] + tmp_3041[2]];
    signal tmp_3043[3] <== CMul()(challengeQ, tmp_3042);
    signal tmp_3044[3] <== [evals[57][0] + evals[107][0], evals[57][1] + evals[107][1], evals[57][2] + evals[107][2]];
    signal tmp_3045[3] <== [tmp_3044[0] + evals[53][0], tmp_3044[1] + evals[53][1], tmp_3044[2] + evals[53][2]];
    signal tmp_3046[3] <== [tmp_3045[0] + evals[56][0], tmp_3045[1] + evals[56][1], tmp_3045[2] + evals[56][2]];
    signal tmp_3047[3] <== CMul()(evals[78], evals[79]);
    signal tmp_3048[3] <== CMul()(evals[44], tmp_3047);
    signal tmp_3049[3] <== CMul()(evals[45], evals[78]);
    signal tmp_3050[3] <== [tmp_3048[0] + tmp_3049[0], tmp_3048[1] + tmp_3049[1], tmp_3048[2] + tmp_3049[2]];
    signal tmp_3051[3] <== CMul()(evals[46], evals[79]);
    signal tmp_3052[3] <== [tmp_3050[0] + tmp_3051[0], tmp_3050[1] + tmp_3051[1], tmp_3050[2] + tmp_3051[2]];
    signal tmp_3053[3] <== CMul()(evals[47], evals[80]);
    signal tmp_3054[3] <== [tmp_3052[0] + tmp_3053[0], tmp_3052[1] + tmp_3053[1], tmp_3052[2] + tmp_3053[2]];
    signal tmp_3055[3] <== [tmp_3054[0] + evals[48][0], tmp_3054[1] + evals[48][1], tmp_3054[2] + evals[48][2]];
    signal tmp_3056[3] <== CMul()(tmp_3046, tmp_3055);
    signal tmp_3057[3] <== [tmp_3043[0] + tmp_3056[0], tmp_3043[1] + tmp_3056[1], tmp_3043[2] + tmp_3056[2]];
    signal tmp_3058[3] <== CMul()(challengeQ, tmp_3057);
    signal tmp_3059[3] <== [evals[57][0] + evals[107][0], evals[57][1] + evals[107][1], evals[57][2] + evals[107][2]];
    signal tmp_3060[3] <== [tmp_3059[0] + evals[53][0], tmp_3059[1] + evals[53][1], tmp_3059[2] + evals[53][2]];
    signal tmp_3061[3] <== [tmp_3060[0] + evals[56][0], tmp_3060[1] + evals[56][1], tmp_3060[2] + evals[56][2]];
    signal tmp_3062[3] <== CMul()(evals[81], evals[82]);
    signal tmp_3063[3] <== CMul()(evals[44], tmp_3062);
    signal tmp_3064[3] <== CMul()(evals[45], evals[81]);
    signal tmp_3065[3] <== [tmp_3063[0] + tmp_3064[0], tmp_3063[1] + tmp_3064[1], tmp_3063[2] + tmp_3064[2]];
    signal tmp_3066[3] <== CMul()(evals[46], evals[82]);
    signal tmp_3067[3] <== [tmp_3065[0] + tmp_3066[0], tmp_3065[1] + tmp_3066[1], tmp_3065[2] + tmp_3066[2]];
    signal tmp_3068[3] <== CMul()(evals[47], evals[83]);
    signal tmp_3069[3] <== [tmp_3067[0] + tmp_3068[0], tmp_3067[1] + tmp_3068[1], tmp_3067[2] + tmp_3068[2]];
    signal tmp_3070[3] <== [tmp_3069[0] + evals[48][0], tmp_3069[1] + evals[48][1], tmp_3069[2] + evals[48][2]];
    signal tmp_3071[3] <== CMul()(tmp_3061, tmp_3070);
    signal tmp_3072[3] <== [tmp_3058[0] + tmp_3071[0], tmp_3058[1] + tmp_3071[1], tmp_3058[2] + tmp_3071[2]];
    signal tmp_3073[3] <== CMul()(challengeQ, tmp_3072);
    signal tmp_3074[3] <== CMul()(evals[50], evals[134]);
    signal tmp_3075[3] <== [evals[134][0] - 1, evals[134][1], evals[134][2]];
    signal tmp_3076[3] <== CMul()(tmp_3074, tmp_3075);
    signal tmp_3077[3] <== [tmp_3073[0] + tmp_3076[0], tmp_3073[1] + tmp_3076[1], tmp_3073[2] + tmp_3076[2]];
    signal tmp_3078[3] <== CMul()(challengeQ, tmp_3077);
    signal tmp_3079[3] <== CMul()(evals[50], evals[135]);
    signal tmp_3080[3] <== [evals[135][0] - 1, evals[135][1], evals[135][2]];
    signal tmp_3081[3] <== CMul()(tmp_3079, tmp_3080);
    signal tmp_3082[3] <== [tmp_3078[0] + tmp_3081[0], tmp_3078[1] + tmp_3081[1], tmp_3078[2] + tmp_3081[2]];
    signal tmp_3083[3] <== CMul()(challengeQ, tmp_3082);
    signal tmp_3084[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3085[3] <== [1 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_3086[3] <== [1 - evals[135][0], -evals[135][1], -evals[135][2]];
    signal tmp_3087[3] <== CMul()(tmp_3085, tmp_3086);
    signal tmp_3088[3] <== CMul()(tmp_3087, evals[63]);
    signal tmp_3089[3] <== [1 - evals[135][0], -evals[135][1], -evals[135][2]];
    signal tmp_3090[3] <== CMul()(evals[134], tmp_3089);
    signal tmp_3091[3] <== CMul()(tmp_3090, evals[67]);
    signal tmp_3092[3] <== [tmp_3088[0] + tmp_3091[0], tmp_3088[1] + tmp_3091[1], tmp_3088[2] + tmp_3091[2]];
    signal tmp_3093[3] <== [1 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_3094[3] <== CMul()(tmp_3093, evals[135]);
    signal tmp_3095[3] <== CMul()(tmp_3094, evals[67]);
    signal tmp_3096[3] <== [tmp_3092[0] + tmp_3095[0], tmp_3092[1] + tmp_3095[1], tmp_3092[2] + tmp_3095[2]];
    signal tmp_3097[3] <== CMul()(evals[50], tmp_3096);
    signal tmp_3098[3] <== CMul()(evals[49], evals[63]);
    signal tmp_3099[3] <== [tmp_3097[0] + tmp_3098[0], tmp_3097[1] + tmp_3098[1], tmp_3097[2] + tmp_3098[2]];
    signal tmp_3100[3] <== [2 * tmp_3099[0], 2 * tmp_3099[1], 2 * tmp_3099[2]];
    signal tmp_3101[3] <== CMul()(tmp_3087, evals[60]);
    signal tmp_3102[3] <== CMul()(tmp_3090, evals[64]);
    signal tmp_3103[3] <== [tmp_3101[0] + tmp_3102[0], tmp_3101[1] + tmp_3102[1], tmp_3101[2] + tmp_3102[2]];
    signal tmp_3104[3] <== CMul()(tmp_3094, evals[64]);
    signal tmp_3105[3] <== [tmp_3103[0] + tmp_3104[0], tmp_3103[1] + tmp_3104[1], tmp_3103[2] + tmp_3104[2]];
    signal tmp_3106[3] <== CMul()(evals[50], tmp_3105);
    signal tmp_3107[3] <== CMul()(evals[49], evals[60]);
    signal tmp_3108[3] <== [tmp_3106[0] + tmp_3107[0], tmp_3106[1] + tmp_3107[1], tmp_3106[2] + tmp_3107[2]];
    signal tmp_3109[3] <== CMul()(tmp_3087, evals[61]);
    signal tmp_3110[3] <== CMul()(tmp_3090, evals[65]);
    signal tmp_3111[3] <== [tmp_3109[0] + tmp_3110[0], tmp_3109[1] + tmp_3110[1], tmp_3109[2] + tmp_3110[2]];
    signal tmp_3112[3] <== CMul()(tmp_3094, evals[65]);
    signal tmp_3113[3] <== [tmp_3111[0] + tmp_3112[0], tmp_3111[1] + tmp_3112[1], tmp_3111[2] + tmp_3112[2]];
    signal tmp_3114[3] <== CMul()(evals[50], tmp_3113);
    signal tmp_3115[3] <== CMul()(evals[49], evals[61]);
    signal tmp_3116[3] <== [tmp_3114[0] + tmp_3115[0], tmp_3114[1] + tmp_3115[1], tmp_3114[2] + tmp_3115[2]];
    signal tmp_3117[3] <== [tmp_3108[0] + tmp_3116[0], tmp_3108[1] + tmp_3116[1], tmp_3108[2] + tmp_3116[2]];
    signal tmp_3118[3] <== [tmp_3100[0] + tmp_3117[0], tmp_3100[1] + tmp_3117[1], tmp_3100[2] + tmp_3117[2]];
    signal tmp_3119[3] <== [4 * tmp_3117[0], 4 * tmp_3117[1], 4 * tmp_3117[2]];
    signal tmp_3120[3] <== CMul()(tmp_3087, evals[61]);
    signal tmp_3121[3] <== CMul()(tmp_3090, evals[65]);
    signal tmp_3122[3] <== [tmp_3120[0] + tmp_3121[0], tmp_3120[1] + tmp_3121[1], tmp_3120[2] + tmp_3121[2]];
    signal tmp_3123[3] <== CMul()(tmp_3094, evals[65]);
    signal tmp_3124[3] <== [tmp_3122[0] + tmp_3123[0], tmp_3122[1] + tmp_3123[1], tmp_3122[2] + tmp_3123[2]];
    signal tmp_3125[3] <== CMul()(evals[50], tmp_3124);
    signal tmp_3126[3] <== CMul()(evals[49], evals[61]);
    signal tmp_3127[3] <== [tmp_3125[0] + tmp_3126[0], tmp_3125[1] + tmp_3126[1], tmp_3125[2] + tmp_3126[2]];
    signal tmp_3128[3] <== [2 * tmp_3127[0], 2 * tmp_3127[1], 2 * tmp_3127[2]];
    signal tmp_3129[3] <== CMul()(tmp_3087, evals[62]);
    signal tmp_3130[3] <== CMul()(tmp_3090, evals[66]);
    signal tmp_3131[3] <== [tmp_3129[0] + tmp_3130[0], tmp_3129[1] + tmp_3130[1], tmp_3129[2] + tmp_3130[2]];
    signal tmp_3132[3] <== CMul()(tmp_3094, evals[66]);
    signal tmp_3133[3] <== [tmp_3131[0] + tmp_3132[0], tmp_3131[1] + tmp_3132[1], tmp_3131[2] + tmp_3132[2]];
    signal tmp_3134[3] <== CMul()(evals[50], tmp_3133);
    signal tmp_3135[3] <== CMul()(evals[49], evals[62]);
    signal tmp_3136[3] <== [tmp_3134[0] + tmp_3135[0], tmp_3134[1] + tmp_3135[1], tmp_3134[2] + tmp_3135[2]];
    signal tmp_3137[3] <== CMul()(tmp_3087, evals[63]);
    signal tmp_3138[3] <== CMul()(tmp_3090, evals[67]);
    signal tmp_3139[3] <== [tmp_3137[0] + tmp_3138[0], tmp_3137[1] + tmp_3138[1], tmp_3137[2] + tmp_3138[2]];
    signal tmp_3140[3] <== CMul()(tmp_3094, evals[67]);
    signal tmp_3141[3] <== [tmp_3139[0] + tmp_3140[0], tmp_3139[1] + tmp_3140[1], tmp_3139[2] + tmp_3140[2]];
    signal tmp_3142[3] <== CMul()(evals[50], tmp_3141);
    signal tmp_3143[3] <== CMul()(evals[49], evals[63]);
    signal tmp_3144[3] <== [tmp_3142[0] + tmp_3143[0], tmp_3142[1] + tmp_3143[1], tmp_3142[2] + tmp_3143[2]];
    signal tmp_3145[3] <== [tmp_3136[0] + tmp_3144[0], tmp_3136[1] + tmp_3144[1], tmp_3136[2] + tmp_3144[2]];
    signal tmp_3146[3] <== [tmp_3128[0] + tmp_3145[0], tmp_3128[1] + tmp_3145[1], tmp_3128[2] + tmp_3145[2]];
    signal tmp_3147[3] <== [tmp_3119[0] + tmp_3146[0], tmp_3119[1] + tmp_3146[1], tmp_3119[2] + tmp_3146[2]];
    signal tmp_3148[3] <== [tmp_3118[0] + tmp_3147[0], tmp_3118[1] + tmp_3147[1], tmp_3118[2] + tmp_3147[2]];
    signal tmp_3149[3] <== CMul()(tmp_3087, evals[67]);
    signal tmp_3150[3] <== CMul()(tmp_3090, evals[63]);
    signal tmp_3151[3] <== [tmp_3149[0] + tmp_3150[0], tmp_3149[1] + tmp_3150[1], tmp_3149[2] + tmp_3150[2]];
    signal tmp_3152[3] <== CMul()(tmp_3094, evals[71]);
    signal tmp_3153[3] <== [tmp_3151[0] + tmp_3152[0], tmp_3151[1] + tmp_3152[1], tmp_3151[2] + tmp_3152[2]];
    signal tmp_3154[3] <== CMul()(evals[50], tmp_3153);
    signal tmp_3155[3] <== CMul()(evals[49], evals[67]);
    signal tmp_3156[3] <== [tmp_3154[0] + tmp_3155[0], tmp_3154[1] + tmp_3155[1], tmp_3154[2] + tmp_3155[2]];
    signal tmp_3157[3] <== [2 * tmp_3156[0], 2 * tmp_3156[1], 2 * tmp_3156[2]];
    signal tmp_3158[3] <== CMul()(tmp_3087, evals[64]);
    signal tmp_3159[3] <== CMul()(tmp_3090, evals[60]);
    signal tmp_3160[3] <== [tmp_3158[0] + tmp_3159[0], tmp_3158[1] + tmp_3159[1], tmp_3158[2] + tmp_3159[2]];
    signal tmp_3161[3] <== CMul()(tmp_3094, evals[68]);
    signal tmp_3162[3] <== [tmp_3160[0] + tmp_3161[0], tmp_3160[1] + tmp_3161[1], tmp_3160[2] + tmp_3161[2]];
    signal tmp_3163[3] <== CMul()(evals[50], tmp_3162);
    signal tmp_3164[3] <== CMul()(evals[49], evals[64]);
    signal tmp_3165[3] <== [tmp_3163[0] + tmp_3164[0], tmp_3163[1] + tmp_3164[1], tmp_3163[2] + tmp_3164[2]];
    signal tmp_3166[3] <== CMul()(tmp_3087, evals[65]);
    signal tmp_3167[3] <== CMul()(tmp_3090, evals[61]);
    signal tmp_3168[3] <== [tmp_3166[0] + tmp_3167[0], tmp_3166[1] + tmp_3167[1], tmp_3166[2] + tmp_3167[2]];
    signal tmp_3169[3] <== CMul()(tmp_3094, evals[69]);
    signal tmp_3170[3] <== [tmp_3168[0] + tmp_3169[0], tmp_3168[1] + tmp_3169[1], tmp_3168[2] + tmp_3169[2]];
    signal tmp_3171[3] <== CMul()(evals[50], tmp_3170);
    signal tmp_3172[3] <== CMul()(evals[49], evals[65]);
    signal tmp_3173[3] <== [tmp_3171[0] + tmp_3172[0], tmp_3171[1] + tmp_3172[1], tmp_3171[2] + tmp_3172[2]];
    signal tmp_3174[3] <== [tmp_3165[0] + tmp_3173[0], tmp_3165[1] + tmp_3173[1], tmp_3165[2] + tmp_3173[2]];
    signal tmp_3175[3] <== [tmp_3157[0] + tmp_3174[0], tmp_3157[1] + tmp_3174[1], tmp_3157[2] + tmp_3174[2]];
    signal tmp_3176[3] <== [4 * tmp_3174[0], 4 * tmp_3174[1], 4 * tmp_3174[2]];
    signal tmp_3177[3] <== CMul()(tmp_3087, evals[65]);
    signal tmp_3178[3] <== CMul()(tmp_3090, evals[61]);
    signal tmp_3179[3] <== [tmp_3177[0] + tmp_3178[0], tmp_3177[1] + tmp_3178[1], tmp_3177[2] + tmp_3178[2]];
    signal tmp_3180[3] <== CMul()(tmp_3094, evals[69]);
    signal tmp_3181[3] <== [tmp_3179[0] + tmp_3180[0], tmp_3179[1] + tmp_3180[1], tmp_3179[2] + tmp_3180[2]];
    signal tmp_3182[3] <== CMul()(evals[50], tmp_3181);
    signal tmp_3183[3] <== CMul()(evals[49], evals[65]);
    signal tmp_3184[3] <== [tmp_3182[0] + tmp_3183[0], tmp_3182[1] + tmp_3183[1], tmp_3182[2] + tmp_3183[2]];
    signal tmp_3185[3] <== [2 * tmp_3184[0], 2 * tmp_3184[1], 2 * tmp_3184[2]];
    signal tmp_3186[3] <== CMul()(tmp_3087, evals[66]);
    signal tmp_3187[3] <== CMul()(tmp_3090, evals[62]);
    signal tmp_3188[3] <== [tmp_3186[0] + tmp_3187[0], tmp_3186[1] + tmp_3187[1], tmp_3186[2] + tmp_3187[2]];
    signal tmp_3189[3] <== CMul()(tmp_3094, evals[70]);
    signal tmp_3190[3] <== [tmp_3188[0] + tmp_3189[0], tmp_3188[1] + tmp_3189[1], tmp_3188[2] + tmp_3189[2]];
    signal tmp_3191[3] <== CMul()(evals[50], tmp_3190);
    signal tmp_3192[3] <== CMul()(evals[49], evals[66]);
    signal tmp_3193[3] <== [tmp_3191[0] + tmp_3192[0], tmp_3191[1] + tmp_3192[1], tmp_3191[2] + tmp_3192[2]];
    signal tmp_3194[3] <== CMul()(tmp_3087, evals[67]);
    signal tmp_3195[3] <== CMul()(tmp_3090, evals[63]);
    signal tmp_3196[3] <== [tmp_3194[0] + tmp_3195[0], tmp_3194[1] + tmp_3195[1], tmp_3194[2] + tmp_3195[2]];
    signal tmp_3197[3] <== CMul()(tmp_3094, evals[71]);
    signal tmp_3198[3] <== [tmp_3196[0] + tmp_3197[0], tmp_3196[1] + tmp_3197[1], tmp_3196[2] + tmp_3197[2]];
    signal tmp_3199[3] <== CMul()(evals[50], tmp_3198);
    signal tmp_3200[3] <== CMul()(evals[49], evals[67]);
    signal tmp_3201[3] <== [tmp_3199[0] + tmp_3200[0], tmp_3199[1] + tmp_3200[1], tmp_3199[2] + tmp_3200[2]];
    signal tmp_3202[3] <== [tmp_3193[0] + tmp_3201[0], tmp_3193[1] + tmp_3201[1], tmp_3193[2] + tmp_3201[2]];
    signal tmp_3203[3] <== [tmp_3185[0] + tmp_3202[0], tmp_3185[1] + tmp_3202[1], tmp_3185[2] + tmp_3202[2]];
    signal tmp_3204[3] <== [tmp_3176[0] + tmp_3203[0], tmp_3176[1] + tmp_3203[1], tmp_3176[2] + tmp_3203[2]];
    signal tmp_3205[3] <== [tmp_3175[0] + tmp_3204[0], tmp_3175[1] + tmp_3204[1], tmp_3175[2] + tmp_3204[2]];
    signal tmp_3206[3] <== [tmp_3148[0] + tmp_3205[0], tmp_3148[1] + tmp_3205[1], tmp_3148[2] + tmp_3205[2]];
    signal tmp_3207[3] <== CMul()(tmp_3087, evals[71]);
    signal tmp_3208[3] <== CMul()(tmp_3090, evals[71]);
    signal tmp_3209[3] <== [tmp_3207[0] + tmp_3208[0], tmp_3207[1] + tmp_3208[1], tmp_3207[2] + tmp_3208[2]];
    signal tmp_3210[3] <== CMul()(tmp_3094, evals[63]);
    signal tmp_3211[3] <== [tmp_3209[0] + tmp_3210[0], tmp_3209[1] + tmp_3210[1], tmp_3209[2] + tmp_3210[2]];
    signal tmp_3212[3] <== CMul()(evals[50], tmp_3211);
    signal tmp_3213[3] <== CMul()(evals[49], evals[71]);
    signal tmp_3214[3] <== [tmp_3212[0] + tmp_3213[0], tmp_3212[1] + tmp_3213[1], tmp_3212[2] + tmp_3213[2]];
    signal tmp_3215[3] <== [2 * tmp_3214[0], 2 * tmp_3214[1], 2 * tmp_3214[2]];
    signal tmp_3216[3] <== CMul()(tmp_3087, evals[68]);
    signal tmp_3217[3] <== CMul()(tmp_3090, evals[68]);
    signal tmp_3218[3] <== [tmp_3216[0] + tmp_3217[0], tmp_3216[1] + tmp_3217[1], tmp_3216[2] + tmp_3217[2]];
    signal tmp_3219[3] <== CMul()(tmp_3094, evals[60]);
    signal tmp_3220[3] <== [tmp_3218[0] + tmp_3219[0], tmp_3218[1] + tmp_3219[1], tmp_3218[2] + tmp_3219[2]];
    signal tmp_3221[3] <== CMul()(evals[50], tmp_3220);
    signal tmp_3222[3] <== CMul()(evals[49], evals[68]);
    signal tmp_3223[3] <== [tmp_3221[0] + tmp_3222[0], tmp_3221[1] + tmp_3222[1], tmp_3221[2] + tmp_3222[2]];
    signal tmp_3224[3] <== CMul()(tmp_3087, evals[69]);
    signal tmp_3225[3] <== CMul()(tmp_3090, evals[69]);
    signal tmp_3226[3] <== [tmp_3224[0] + tmp_3225[0], tmp_3224[1] + tmp_3225[1], tmp_3224[2] + tmp_3225[2]];
    signal tmp_3227[3] <== CMul()(tmp_3094, evals[61]);
    signal tmp_3228[3] <== [tmp_3226[0] + tmp_3227[0], tmp_3226[1] + tmp_3227[1], tmp_3226[2] + tmp_3227[2]];
    signal tmp_3229[3] <== CMul()(evals[50], tmp_3228);
    signal tmp_3230[3] <== CMul()(evals[49], evals[69]);
    signal tmp_3231[3] <== [tmp_3229[0] + tmp_3230[0], tmp_3229[1] + tmp_3230[1], tmp_3229[2] + tmp_3230[2]];
    signal tmp_3232[3] <== [tmp_3223[0] + tmp_3231[0], tmp_3223[1] + tmp_3231[1], tmp_3223[2] + tmp_3231[2]];
    signal tmp_3233[3] <== [tmp_3215[0] + tmp_3232[0], tmp_3215[1] + tmp_3232[1], tmp_3215[2] + tmp_3232[2]];
    signal tmp_3234[3] <== [4 * tmp_3232[0], 4 * tmp_3232[1], 4 * tmp_3232[2]];
    signal tmp_3235[3] <== CMul()(tmp_3087, evals[69]);
    signal tmp_3236[3] <== CMul()(tmp_3090, evals[69]);
    signal tmp_3237[3] <== [tmp_3235[0] + tmp_3236[0], tmp_3235[1] + tmp_3236[1], tmp_3235[2] + tmp_3236[2]];
    signal tmp_3238[3] <== CMul()(tmp_3094, evals[61]);
    signal tmp_3239[3] <== [tmp_3237[0] + tmp_3238[0], tmp_3237[1] + tmp_3238[1], tmp_3237[2] + tmp_3238[2]];
    signal tmp_3240[3] <== CMul()(evals[50], tmp_3239);
    signal tmp_3241[3] <== CMul()(evals[49], evals[69]);
    signal tmp_3242[3] <== [tmp_3240[0] + tmp_3241[0], tmp_3240[1] + tmp_3241[1], tmp_3240[2] + tmp_3241[2]];
    signal tmp_3243[3] <== [2 * tmp_3242[0], 2 * tmp_3242[1], 2 * tmp_3242[2]];
    signal tmp_3244[3] <== CMul()(tmp_3087, evals[70]);
    signal tmp_3245[3] <== CMul()(tmp_3090, evals[70]);
    signal tmp_3246[3] <== [tmp_3244[0] + tmp_3245[0], tmp_3244[1] + tmp_3245[1], tmp_3244[2] + tmp_3245[2]];
    signal tmp_3247[3] <== CMul()(tmp_3094, evals[62]);
    signal tmp_3248[3] <== [tmp_3246[0] + tmp_3247[0], tmp_3246[1] + tmp_3247[1], tmp_3246[2] + tmp_3247[2]];
    signal tmp_3249[3] <== CMul()(evals[50], tmp_3248);
    signal tmp_3250[3] <== CMul()(evals[49], evals[70]);
    signal tmp_3251[3] <== [tmp_3249[0] + tmp_3250[0], tmp_3249[1] + tmp_3250[1], tmp_3249[2] + tmp_3250[2]];
    signal tmp_3252[3] <== CMul()(tmp_3087, evals[71]);
    signal tmp_3253[3] <== CMul()(tmp_3090, evals[71]);
    signal tmp_3254[3] <== [tmp_3252[0] + tmp_3253[0], tmp_3252[1] + tmp_3253[1], tmp_3252[2] + tmp_3253[2]];
    signal tmp_3255[3] <== CMul()(tmp_3094, evals[63]);
    signal tmp_3256[3] <== [tmp_3254[0] + tmp_3255[0], tmp_3254[1] + tmp_3255[1], tmp_3254[2] + tmp_3255[2]];
    signal tmp_3257[3] <== CMul()(evals[50], tmp_3256);
    signal tmp_3258[3] <== CMul()(evals[49], evals[71]);
    signal tmp_3259[3] <== [tmp_3257[0] + tmp_3258[0], tmp_3257[1] + tmp_3258[1], tmp_3257[2] + tmp_3258[2]];
    signal tmp_3260[3] <== [tmp_3251[0] + tmp_3259[0], tmp_3251[1] + tmp_3259[1], tmp_3251[2] + tmp_3259[2]];
    signal tmp_3261[3] <== [tmp_3243[0] + tmp_3260[0], tmp_3243[1] + tmp_3260[1], tmp_3243[2] + tmp_3260[2]];
    signal tmp_3262[3] <== [tmp_3234[0] + tmp_3261[0], tmp_3234[1] + tmp_3261[1], tmp_3234[2] + tmp_3261[2]];
    signal tmp_3263[3] <== [tmp_3233[0] + tmp_3262[0], tmp_3233[1] + tmp_3262[1], tmp_3233[2] + tmp_3262[2]];
    signal tmp_3264[3] <== [tmp_3206[0] + tmp_3263[0], tmp_3206[1] + tmp_3263[1], tmp_3206[2] + tmp_3263[2]];
    signal tmp_3265[3] <== [tmp_3148[0] + tmp_3264[0], tmp_3148[1] + tmp_3264[1], tmp_3148[2] + tmp_3264[2]];
    signal tmp_3266[3] <== [evals[72][0] - tmp_3265[0], evals[72][1] - tmp_3265[1], evals[72][2] - tmp_3265[2]];
    signal tmp_3267[3] <== CMul()(tmp_3084, tmp_3266);
    signal tmp_3268[3] <== [tmp_3083[0] + tmp_3267[0], tmp_3083[1] + tmp_3267[1], tmp_3083[2] + tmp_3267[2]];
    signal tmp_3269[3] <== CMul()(challengeQ, tmp_3268);
    signal tmp_3270[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3271[3] <== [tmp_3147[0] + tmp_3204[0], tmp_3147[1] + tmp_3204[1], tmp_3147[2] + tmp_3204[2]];
    signal tmp_3272[3] <== [tmp_3271[0] + tmp_3262[0], tmp_3271[1] + tmp_3262[1], tmp_3271[2] + tmp_3262[2]];
    signal tmp_3273[3] <== [tmp_3147[0] + tmp_3272[0], tmp_3147[1] + tmp_3272[1], tmp_3147[2] + tmp_3272[2]];
    signal tmp_3274[3] <== [evals[73][0] - tmp_3273[0], evals[73][1] - tmp_3273[1], evals[73][2] - tmp_3273[2]];
    signal tmp_3275[3] <== CMul()(tmp_3270, tmp_3274);
    signal tmp_3276[3] <== [tmp_3269[0] + tmp_3275[0], tmp_3269[1] + tmp_3275[1], tmp_3269[2] + tmp_3275[2]];
    signal tmp_3277[3] <== CMul()(challengeQ, tmp_3276);
    signal tmp_3278[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3279[3] <== [4 * tmp_3145[0], 4 * tmp_3145[1], 4 * tmp_3145[2]];
    signal tmp_3280[3] <== [tmp_3279[0] + tmp_3118[0], tmp_3279[1] + tmp_3118[1], tmp_3279[2] + tmp_3118[2]];
    signal tmp_3281[3] <== [tmp_3146[0] + tmp_3280[0], tmp_3146[1] + tmp_3280[1], tmp_3146[2] + tmp_3280[2]];
    signal tmp_3282[3] <== [4 * tmp_3202[0], 4 * tmp_3202[1], 4 * tmp_3202[2]];
    signal tmp_3283[3] <== [tmp_3282[0] + tmp_3175[0], tmp_3282[1] + tmp_3175[1], tmp_3282[2] + tmp_3175[2]];
    signal tmp_3284[3] <== [tmp_3203[0] + tmp_3283[0], tmp_3203[1] + tmp_3283[1], tmp_3203[2] + tmp_3283[2]];
    signal tmp_3285[3] <== [tmp_3281[0] + tmp_3284[0], tmp_3281[1] + tmp_3284[1], tmp_3281[2] + tmp_3284[2]];
    signal tmp_3286[3] <== [4 * tmp_3260[0], 4 * tmp_3260[1], 4 * tmp_3260[2]];
    signal tmp_3287[3] <== [tmp_3286[0] + tmp_3233[0], tmp_3286[1] + tmp_3233[1], tmp_3286[2] + tmp_3233[2]];
    signal tmp_3288[3] <== [tmp_3261[0] + tmp_3287[0], tmp_3261[1] + tmp_3287[1], tmp_3261[2] + tmp_3287[2]];
    signal tmp_3289[3] <== [tmp_3285[0] + tmp_3288[0], tmp_3285[1] + tmp_3288[1], tmp_3285[2] + tmp_3288[2]];
    signal tmp_3290[3] <== [tmp_3281[0] + tmp_3289[0], tmp_3281[1] + tmp_3289[1], tmp_3281[2] + tmp_3289[2]];
    signal tmp_3291[3] <== [evals[74][0] - tmp_3290[0], evals[74][1] - tmp_3290[1], evals[74][2] - tmp_3290[2]];
    signal tmp_3292[3] <== CMul()(tmp_3278, tmp_3291);
    signal tmp_3293[3] <== [tmp_3277[0] + tmp_3292[0], tmp_3277[1] + tmp_3292[1], tmp_3277[2] + tmp_3292[2]];
    signal tmp_3294[3] <== CMul()(challengeQ, tmp_3293);
    signal tmp_3295[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3296[3] <== [tmp_3280[0] + tmp_3283[0], tmp_3280[1] + tmp_3283[1], tmp_3280[2] + tmp_3283[2]];
    signal tmp_3297[3] <== [tmp_3296[0] + tmp_3287[0], tmp_3296[1] + tmp_3287[1], tmp_3296[2] + tmp_3287[2]];
    signal tmp_3298[3] <== [tmp_3280[0] + tmp_3297[0], tmp_3280[1] + tmp_3297[1], tmp_3280[2] + tmp_3297[2]];
    signal tmp_3299[3] <== [evals[75][0] - tmp_3298[0], evals[75][1] - tmp_3298[1], evals[75][2] - tmp_3298[2]];
    signal tmp_3300[3] <== CMul()(tmp_3295, tmp_3299);
    signal tmp_3301[3] <== [tmp_3294[0] + tmp_3300[0], tmp_3294[1] + tmp_3300[1], tmp_3294[2] + tmp_3300[2]];
    signal tmp_3302[3] <== CMul()(challengeQ, tmp_3301);
    signal tmp_3303[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3304[3] <== [tmp_3205[0] + tmp_3264[0], tmp_3205[1] + tmp_3264[1], tmp_3205[2] + tmp_3264[2]];
    signal tmp_3305[3] <== [evals[76][0] - tmp_3304[0], evals[76][1] - tmp_3304[1], evals[76][2] - tmp_3304[2]];
    signal tmp_3306[3] <== CMul()(tmp_3303, tmp_3305);
    signal tmp_3307[3] <== [tmp_3302[0] + tmp_3306[0], tmp_3302[1] + tmp_3306[1], tmp_3302[2] + tmp_3306[2]];
    signal tmp_3308[3] <== CMul()(challengeQ, tmp_3307);
    signal tmp_3309[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3310[3] <== [tmp_3204[0] + tmp_3272[0], tmp_3204[1] + tmp_3272[1], tmp_3204[2] + tmp_3272[2]];
    signal tmp_3311[3] <== [evals[77][0] - tmp_3310[0], evals[77][1] - tmp_3310[1], evals[77][2] - tmp_3310[2]];
    signal tmp_3312[3] <== CMul()(tmp_3309, tmp_3311);
    signal tmp_3313[3] <== [tmp_3308[0] + tmp_3312[0], tmp_3308[1] + tmp_3312[1], tmp_3308[2] + tmp_3312[2]];
    signal tmp_3314[3] <== CMul()(challengeQ, tmp_3313);
    signal tmp_3315[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3316[3] <== [tmp_3284[0] + tmp_3289[0], tmp_3284[1] + tmp_3289[1], tmp_3284[2] + tmp_3289[2]];
    signal tmp_3317[3] <== [evals[78][0] - tmp_3316[0], evals[78][1] - tmp_3316[1], evals[78][2] - tmp_3316[2]];
    signal tmp_3318[3] <== CMul()(tmp_3315, tmp_3317);
    signal tmp_3319[3] <== [tmp_3314[0] + tmp_3318[0], tmp_3314[1] + tmp_3318[1], tmp_3314[2] + tmp_3318[2]];
    signal tmp_3320[3] <== CMul()(challengeQ, tmp_3319);
    signal tmp_3321[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3322[3] <== [tmp_3283[0] + tmp_3297[0], tmp_3283[1] + tmp_3297[1], tmp_3283[2] + tmp_3297[2]];
    signal tmp_3323[3] <== [evals[79][0] - tmp_3322[0], evals[79][1] - tmp_3322[1], evals[79][2] - tmp_3322[2]];
    signal tmp_3324[3] <== CMul()(tmp_3321, tmp_3323);
    signal tmp_3325[3] <== [tmp_3320[0] + tmp_3324[0], tmp_3320[1] + tmp_3324[1], tmp_3320[2] + tmp_3324[2]];
    signal tmp_3326[3] <== CMul()(challengeQ, tmp_3325);
    signal tmp_3327[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3328[3] <== [tmp_3263[0] + tmp_3264[0], tmp_3263[1] + tmp_3264[1], tmp_3263[2] + tmp_3264[2]];
    signal tmp_3329[3] <== [evals[80][0] - tmp_3328[0], evals[80][1] - tmp_3328[1], evals[80][2] - tmp_3328[2]];
    signal tmp_3330[3] <== CMul()(tmp_3327, tmp_3329);
    signal tmp_3331[3] <== [tmp_3326[0] + tmp_3330[0], tmp_3326[1] + tmp_3330[1], tmp_3326[2] + tmp_3330[2]];
    signal tmp_3332[3] <== CMul()(challengeQ, tmp_3331);
    signal tmp_3333[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3334[3] <== [tmp_3262[0] + tmp_3272[0], tmp_3262[1] + tmp_3272[1], tmp_3262[2] + tmp_3272[2]];
    signal tmp_3335[3] <== [evals[81][0] - tmp_3334[0], evals[81][1] - tmp_3334[1], evals[81][2] - tmp_3334[2]];
    signal tmp_3336[3] <== CMul()(tmp_3333, tmp_3335);
    signal tmp_3337[3] <== [tmp_3332[0] + tmp_3336[0], tmp_3332[1] + tmp_3336[1], tmp_3332[2] + tmp_3336[2]];
    signal tmp_3338[3] <== CMul()(challengeQ, tmp_3337);
    signal tmp_3339[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3340[3] <== [tmp_3288[0] + tmp_3289[0], tmp_3288[1] + tmp_3289[1], tmp_3288[2] + tmp_3289[2]];
    signal tmp_3341[3] <== [evals[82][0] - tmp_3340[0], evals[82][1] - tmp_3340[1], evals[82][2] - tmp_3340[2]];
    signal tmp_3342[3] <== CMul()(tmp_3339, tmp_3341);
    signal tmp_3343[3] <== [tmp_3338[0] + tmp_3342[0], tmp_3338[1] + tmp_3342[1], tmp_3338[2] + tmp_3342[2]];
    signal tmp_3344[3] <== CMul()(challengeQ, tmp_3343);
    signal tmp_3345[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3346[3] <== [tmp_3287[0] + tmp_3297[0], tmp_3287[1] + tmp_3297[1], tmp_3287[2] + tmp_3297[2]];
    signal tmp_3347[3] <== [evals[83][0] - tmp_3346[0], evals[83][1] - tmp_3346[1], evals[83][2] - tmp_3346[2]];
    signal tmp_3348[3] <== CMul()(tmp_3345, tmp_3347);
    signal tmp_3349[3] <== [tmp_3344[0] + tmp_3348[0], tmp_3344[1] + tmp_3348[1], tmp_3344[2] + tmp_3348[2]];
    signal tmp_3350[3] <== CMul()(challengeQ, tmp_3349);
    signal tmp_3351[3] <== [evals[49][0] + evals[50][0], evals[49][1] + evals[50][1], evals[49][2] + evals[50][2]];
    signal tmp_3352[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3353[3] <== [tmp_3352[0] + evals[108][0], tmp_3352[1] + evals[108][1], tmp_3352[2] + evals[108][2]];
    signal tmp_3354[3] <== [tmp_3353[0] + evals[52][0], tmp_3353[1] + evals[52][1], tmp_3353[2] + evals[52][2]];
    signal tmp_3355[3] <== [tmp_3351[0] * 10625215922958251110, tmp_3351[1] * 10625215922958251110, tmp_3351[2] * 10625215922958251110];
    signal tmp_3356[3] <== [tmp_2966[0] * 4179687232228901671, tmp_2966[1] * 4179687232228901671, tmp_2966[2] * 4179687232228901671];
    signal tmp_3357[3] <== [tmp_3355[0] + tmp_3356[0], tmp_3355[1] + tmp_3356[1], tmp_3355[2] + tmp_3356[2]];
    signal tmp_3358[3] <== [evals[108][0] * 7221072982690633460, evals[108][1] * 7221072982690633460, evals[108][2] * 7221072982690633460];
    signal tmp_3359[3] <== [tmp_3357[0] + tmp_3358[0], tmp_3357[1] + tmp_3358[1], tmp_3357[2] + tmp_3358[2]];
    signal tmp_3360[3] <== [evals[52][0] * 11718666476052241225, evals[52][1] * 11718666476052241225, evals[52][2] * 11718666476052241225];
    signal tmp_3361[3] <== [tmp_3359[0] + tmp_3360[0], tmp_3359[1] + tmp_3360[1], tmp_3359[2] + tmp_3360[2]];
    signal tmp_3362[3] <== [evals[51][0] * 17941136338888340715, evals[51][1] * 17941136338888340715, evals[51][2] * 17941136338888340715];
    signal tmp_3363[3] <== [tmp_3361[0] + tmp_3362[0], tmp_3361[1] + tmp_3362[1], tmp_3361[2] + tmp_3362[2]];
    signal tmp_3364[3] <== [evals[75][0] + tmp_3363[0], evals[75][1] + tmp_3363[1], evals[75][2] + tmp_3363[2]];
    signal tmp_3365[3] <== [evals[75][0] + tmp_3363[0], evals[75][1] + tmp_3363[1], evals[75][2] + tmp_3363[2]];
    signal tmp_3366[3] <== CMul()(tmp_3364, tmp_3365);
    signal tmp_3367[3] <== CMul()(tmp_3366, tmp_3366);
    signal tmp_3368[3] <== CMul()(tmp_3367, tmp_3366);
    signal tmp_3369[3] <== [evals[75][0] + tmp_3363[0], evals[75][1] + tmp_3363[1], evals[75][2] + tmp_3363[2]];
    signal tmp_3370[3] <== CMul()(tmp_3368, tmp_3369);
    signal tmp_3371[3] <== [2 * tmp_3370[0], 2 * tmp_3370[1], 2 * tmp_3370[2]];
    signal tmp_3372[3] <== [tmp_3351[0] * 1431286215153372998, tmp_3351[1] * 1431286215153372998, tmp_3351[2] * 1431286215153372998];
    signal tmp_3373[3] <== [tmp_2966[0] * 3342624911463171251, tmp_2966[1] * 3342624911463171251, tmp_2966[2] * 3342624911463171251];
    signal tmp_3374[3] <== [tmp_3372[0] + tmp_3373[0], tmp_3372[1] + tmp_3373[1], tmp_3372[2] + tmp_3373[2]];
    signal tmp_3375[3] <== [evals[108][0] * 14306783492963476045, evals[108][1] * 14306783492963476045, evals[108][2] * 14306783492963476045];
    signal tmp_3376[3] <== [tmp_3374[0] + tmp_3375[0], tmp_3374[1] + tmp_3375[1], tmp_3374[2] + tmp_3375[2]];
    signal tmp_3377[3] <== [evals[52][0] * 2911694411222711481, evals[52][1] * 2911694411222711481, evals[52][2] * 2911694411222711481];
    signal tmp_3378[3] <== [tmp_3376[0] + tmp_3377[0], tmp_3376[1] + tmp_3377[1], tmp_3376[2] + tmp_3377[2]];
    signal tmp_3379[3] <== [evals[51][0] * 16242299839765162610, evals[51][1] * 16242299839765162610, evals[51][2] * 16242299839765162610];
    signal tmp_3380[3] <== [tmp_3378[0] + tmp_3379[0], tmp_3378[1] + tmp_3379[1], tmp_3378[2] + tmp_3379[2]];
    signal tmp_3381[3] <== [evals[72][0] + tmp_3380[0], evals[72][1] + tmp_3380[1], evals[72][2] + tmp_3380[2]];
    signal tmp_3382[3] <== [evals[72][0] + tmp_3380[0], evals[72][1] + tmp_3380[1], evals[72][2] + tmp_3380[2]];
    signal tmp_3383[3] <== CMul()(tmp_3381, tmp_3382);
    signal tmp_3384[3] <== CMul()(tmp_3383, tmp_3383);
    signal tmp_3385[3] <== CMul()(tmp_3384, tmp_3383);
    signal tmp_3386[3] <== [evals[72][0] + tmp_3380[0], evals[72][1] + tmp_3380[1], evals[72][2] + tmp_3380[2]];
    signal tmp_3387[3] <== CMul()(tmp_3385, tmp_3386);
    signal tmp_3388[3] <== [tmp_3351[0] * 3509349009260703107, tmp_3351[1] * 3509349009260703107, tmp_3351[2] * 3509349009260703107];
    signal tmp_3389[3] <== [tmp_2966[0] * 6781356195391537436, tmp_2966[1] * 6781356195391537436, tmp_2966[2] * 6781356195391537436];
    signal tmp_3390[3] <== [tmp_3388[0] + tmp_3389[0], tmp_3388[1] + tmp_3389[1], tmp_3388[2] + tmp_3389[2]];
    signal tmp_3391[3] <== [evals[108][0] * 12653264875831356889, evals[108][1] * 12653264875831356889, evals[108][2] * 12653264875831356889];
    signal tmp_3392[3] <== [tmp_3390[0] + tmp_3391[0], tmp_3390[1] + tmp_3391[1], tmp_3390[2] + tmp_3391[2]];
    signal tmp_3393[3] <== [evals[52][0] * 6420652251792580406, evals[52][1] * 6420652251792580406, evals[52][2] * 6420652251792580406];
    signal tmp_3394[3] <== [tmp_3392[0] + tmp_3393[0], tmp_3392[1] + tmp_3393[1], tmp_3392[2] + tmp_3393[2]];
    signal tmp_3395[3] <== [evals[51][0] * 12203738590896308135, evals[51][1] * 12203738590896308135, evals[51][2] * 12203738590896308135];
    signal tmp_3396[3] <== [tmp_3394[0] + tmp_3395[0], tmp_3394[1] + tmp_3395[1], tmp_3394[2] + tmp_3395[2]];
    signal tmp_3397[3] <== [evals[73][0] + tmp_3396[0], evals[73][1] + tmp_3396[1], evals[73][2] + tmp_3396[2]];
    signal tmp_3398[3] <== [evals[73][0] + tmp_3396[0], evals[73][1] + tmp_3396[1], evals[73][2] + tmp_3396[2]];
    signal tmp_3399[3] <== CMul()(tmp_3397, tmp_3398);
    signal tmp_3400[3] <== CMul()(tmp_3399, tmp_3399);
    signal tmp_3401[3] <== CMul()(tmp_3400, tmp_3399);
    signal tmp_3402[3] <== [evals[73][0] + tmp_3396[0], evals[73][1] + tmp_3396[1], evals[73][2] + tmp_3396[2]];
    signal tmp_3403[3] <== CMul()(tmp_3401, tmp_3402);
    signal tmp_3404[3] <== [tmp_3387[0] + tmp_3403[0], tmp_3387[1] + tmp_3403[1], tmp_3387[2] + tmp_3403[2]];
    signal tmp_3405[3] <== [tmp_3371[0] + tmp_3404[0], tmp_3371[1] + tmp_3404[1], tmp_3371[2] + tmp_3404[2]];
    signal tmp_3406[3] <== [4 * tmp_3404[0], 4 * tmp_3404[1], 4 * tmp_3404[2]];
    signal tmp_3407[3] <== [evals[73][0] + tmp_3396[0], evals[73][1] + tmp_3396[1], evals[73][2] + tmp_3396[2]];
    signal tmp_3408[3] <== CMul()(tmp_3401, tmp_3407);
    signal tmp_3409[3] <== [2 * tmp_3408[0], 2 * tmp_3408[1], 2 * tmp_3408[2]];
    signal tmp_3410[3] <== [tmp_3351[0] * 2289575380984896342, tmp_3351[1] * 2289575380984896342, tmp_3351[2] * 2289575380984896342];
    signal tmp_3411[3] <== [tmp_2966[0] * 4697929572322733707, tmp_2966[1] * 4697929572322733707, tmp_2966[2] * 4697929572322733707];
    signal tmp_3412[3] <== [tmp_3410[0] + tmp_3411[0], tmp_3410[1] + tmp_3411[1], tmp_3410[2] + tmp_3411[2]];
    signal tmp_3413[3] <== [evals[108][0] * 10887434669785806501, evals[108][1] * 10887434669785806501, evals[108][2] * 10887434669785806501];
    signal tmp_3414[3] <== [tmp_3412[0] + tmp_3413[0], tmp_3412[1] + tmp_3413[1], tmp_3412[2] + tmp_3413[2]];
    signal tmp_3415[3] <== [evals[52][0] * 323544930728360053, evals[52][1] * 323544930728360053, evals[52][2] * 323544930728360053];
    signal tmp_3416[3] <== [tmp_3414[0] + tmp_3415[0], tmp_3414[1] + tmp_3415[1], tmp_3414[2] + tmp_3415[2]];
    signal tmp_3417[3] <== [evals[51][0] * 5395176197344543510, evals[51][1] * 5395176197344543510, evals[51][2] * 5395176197344543510];
    signal tmp_3418[3] <== [tmp_3416[0] + tmp_3417[0], tmp_3416[1] + tmp_3417[1], tmp_3416[2] + tmp_3417[2]];
    signal tmp_3419[3] <== [evals[74][0] + tmp_3418[0], evals[74][1] + tmp_3418[1], evals[74][2] + tmp_3418[2]];
    signal tmp_3420[3] <== [evals[74][0] + tmp_3418[0], evals[74][1] + tmp_3418[1], evals[74][2] + tmp_3418[2]];
    signal tmp_3421[3] <== CMul()(tmp_3419, tmp_3420);
    signal tmp_3422[3] <== CMul()(tmp_3421, tmp_3421);
    signal tmp_3423[3] <== CMul()(tmp_3422, tmp_3421);
    signal tmp_3424[3] <== [evals[74][0] + tmp_3418[0], evals[74][1] + tmp_3418[1], evals[74][2] + tmp_3418[2]];
    signal tmp_3425[3] <== CMul()(tmp_3423, tmp_3424);
    signal tmp_3426[3] <== [evals[75][0] + tmp_3363[0], evals[75][1] + tmp_3363[1], evals[75][2] + tmp_3363[2]];
    signal tmp_3427[3] <== CMul()(tmp_3368, tmp_3426);
    signal tmp_3428[3] <== [tmp_3425[0] + tmp_3427[0], tmp_3425[1] + tmp_3427[1], tmp_3425[2] + tmp_3427[2]];
    signal tmp_3429[3] <== [tmp_3409[0] + tmp_3428[0], tmp_3409[1] + tmp_3428[1], tmp_3409[2] + tmp_3428[2]];
    signal tmp_3430[3] <== [tmp_3406[0] + tmp_3429[0], tmp_3406[1] + tmp_3429[1], tmp_3406[2] + tmp_3429[2]];
    signal tmp_3431[3] <== [tmp_3405[0] + tmp_3430[0], tmp_3405[1] + tmp_3430[1], tmp_3405[2] + tmp_3430[2]];
    signal tmp_3432[3] <== [tmp_3351[0] * 7736066733515538648, tmp_3351[1] * 7736066733515538648, tmp_3351[2] * 7736066733515538648];
    signal tmp_3433[3] <== [tmp_2966[0] * 6306257051437840427, tmp_2966[1] * 6306257051437840427, tmp_2966[2] * 6306257051437840427];
    signal tmp_3434[3] <== [tmp_3432[0] + tmp_3433[0], tmp_3432[1] + tmp_3433[1], tmp_3432[2] + tmp_3433[2]];
    signal tmp_3435[3] <== [evals[108][0] * 17311934738088402529, evals[108][1] * 17311934738088402529, evals[108][2] * 17311934738088402529];
    signal tmp_3436[3] <== [tmp_3434[0] + tmp_3435[0], tmp_3434[1] + tmp_3435[1], tmp_3434[2] + tmp_3435[2]];
    signal tmp_3437[3] <== [evals[52][0] * 3788504801066818367, evals[52][1] * 3788504801066818367, evals[52][2] * 3788504801066818367];
    signal tmp_3438[3] <== [tmp_3436[0] + tmp_3437[0], tmp_3436[1] + tmp_3437[1], tmp_3436[2] + tmp_3437[2]];
    signal tmp_3439[3] <== [evals[51][0] * 10078371877170729592, evals[51][1] * 10078371877170729592, evals[51][2] * 10078371877170729592];
    signal tmp_3440[3] <== [tmp_3438[0] + tmp_3439[0], tmp_3438[1] + tmp_3439[1], tmp_3438[2] + tmp_3439[2]];
    signal tmp_3441[3] <== [evals[79][0] + tmp_3440[0], evals[79][1] + tmp_3440[1], evals[79][2] + tmp_3440[2]];
    signal tmp_3442[3] <== [evals[79][0] + tmp_3440[0], evals[79][1] + tmp_3440[1], evals[79][2] + tmp_3440[2]];
    signal tmp_3443[3] <== CMul()(tmp_3441, tmp_3442);
    signal tmp_3444[3] <== CMul()(tmp_3443, tmp_3443);
    signal tmp_3445[3] <== CMul()(tmp_3444, tmp_3443);
    signal tmp_3446[3] <== [evals[79][0] + tmp_3440[0], evals[79][1] + tmp_3440[1], evals[79][2] + tmp_3440[2]];
    signal tmp_3447[3] <== CMul()(tmp_3445, tmp_3446);
    signal tmp_3448[3] <== [2 * tmp_3447[0], 2 * tmp_3447[1], 2 * tmp_3447[2]];
    signal tmp_3449[3] <== [tmp_3351[0] * 17137022507167291684, tmp_3351[1] * 17137022507167291684, tmp_3351[2] * 17137022507167291684];
    signal tmp_3450[3] <== [tmp_2966[0] * 17841073646522133059, tmp_2966[1] * 17841073646522133059, tmp_2966[2] * 17841073646522133059];
    signal tmp_3451[3] <== [tmp_3449[0] + tmp_3450[0], tmp_3449[1] + tmp_3450[1], tmp_3449[2] + tmp_3450[2]];
    signal tmp_3452[3] <== [evals[108][0] * 9953585853856674407, evals[108][1] * 9953585853856674407, evals[108][2] * 9953585853856674407];
    signal tmp_3453[3] <== [tmp_3451[0] + tmp_3452[0], tmp_3451[1] + tmp_3452[1], tmp_3451[2] + tmp_3452[2]];
    signal tmp_3454[3] <== [evals[52][0] * 2449132068789045592, evals[52][1] * 2449132068789045592, evals[52][2] * 2449132068789045592];
    signal tmp_3455[3] <== [tmp_3453[0] + tmp_3454[0], tmp_3453[1] + tmp_3454[1], tmp_3453[2] + tmp_3454[2]];
    signal tmp_3456[3] <== [evals[51][0] * 7559392505546762987, evals[51][1] * 7559392505546762987, evals[51][2] * 7559392505546762987];
    signal tmp_3457[3] <== [tmp_3455[0] + tmp_3456[0], tmp_3455[1] + tmp_3456[1], tmp_3455[2] + tmp_3456[2]];
    signal tmp_3458[3] <== [evals[76][0] + tmp_3457[0], evals[76][1] + tmp_3457[1], evals[76][2] + tmp_3457[2]];
    signal tmp_3459[3] <== [evals[76][0] + tmp_3457[0], evals[76][1] + tmp_3457[1], evals[76][2] + tmp_3457[2]];
    signal tmp_3460[3] <== CMul()(tmp_3458, tmp_3459);
    signal tmp_3461[3] <== CMul()(tmp_3460, tmp_3460);
    signal tmp_3462[3] <== CMul()(tmp_3461, tmp_3460);
    signal tmp_3463[3] <== [evals[76][0] + tmp_3457[0], evals[76][1] + tmp_3457[1], evals[76][2] + tmp_3457[2]];
    signal tmp_3464[3] <== CMul()(tmp_3462, tmp_3463);
    signal tmp_3465[3] <== [tmp_3351[0] * 17143426961497010024, tmp_3351[1] * 17143426961497010024, tmp_3351[2] * 17143426961497010024];
    signal tmp_3466[3] <== [tmp_2966[0] * 18340176721233187897, tmp_2966[1] * 18340176721233187897, tmp_2966[2] * 18340176721233187897];
    signal tmp_3467[3] <== [tmp_3465[0] + tmp_3466[0], tmp_3465[1] + tmp_3466[1], tmp_3465[2] + tmp_3466[2]];
    signal tmp_3468[3] <== [evals[108][0] * 13497620366078753434, evals[108][1] * 13497620366078753434, evals[108][2] * 13497620366078753434];
    signal tmp_3469[3] <== [tmp_3467[0] + tmp_3468[0], tmp_3467[1] + tmp_3468[1], tmp_3467[2] + tmp_3468[2]];
    signal tmp_3470[3] <== [evals[52][0] * 17993014181992530560, evals[52][1] * 17993014181992530560, evals[52][2] * 17993014181992530560];
    signal tmp_3471[3] <== [tmp_3469[0] + tmp_3470[0], tmp_3469[1] + tmp_3470[1], tmp_3469[2] + tmp_3470[2]];
    signal tmp_3472[3] <== [evals[51][0] * 549633128904721280, evals[51][1] * 549633128904721280, evals[51][2] * 549633128904721280];
    signal tmp_3473[3] <== [tmp_3471[0] + tmp_3472[0], tmp_3471[1] + tmp_3472[1], tmp_3471[2] + tmp_3472[2]];
    signal tmp_3474[3] <== [evals[77][0] + tmp_3473[0], evals[77][1] + tmp_3473[1], evals[77][2] + tmp_3473[2]];
    signal tmp_3475[3] <== [evals[77][0] + tmp_3473[0], evals[77][1] + tmp_3473[1], evals[77][2] + tmp_3473[2]];
    signal tmp_3476[3] <== CMul()(tmp_3474, tmp_3475);
    signal tmp_3477[3] <== CMul()(tmp_3476, tmp_3476);
    signal tmp_3478[3] <== CMul()(tmp_3477, tmp_3476);
    signal tmp_3479[3] <== [evals[77][0] + tmp_3473[0], evals[77][1] + tmp_3473[1], evals[77][2] + tmp_3473[2]];
    signal tmp_3480[3] <== CMul()(tmp_3478, tmp_3479);
    signal tmp_3481[3] <== [tmp_3464[0] + tmp_3480[0], tmp_3464[1] + tmp_3480[1], tmp_3464[2] + tmp_3480[2]];
    signal tmp_3482[3] <== [tmp_3448[0] + tmp_3481[0], tmp_3448[1] + tmp_3481[1], tmp_3448[2] + tmp_3481[2]];
    signal tmp_3483[3] <== [4 * tmp_3481[0], 4 * tmp_3481[1], 4 * tmp_3481[2]];
    signal tmp_3484[3] <== [evals[77][0] + tmp_3473[0], evals[77][1] + tmp_3473[1], evals[77][2] + tmp_3473[2]];
    signal tmp_3485[3] <== CMul()(tmp_3478, tmp_3484);
    signal tmp_3486[3] <== [2 * tmp_3485[0], 2 * tmp_3485[1], 2 * tmp_3485[2]];
    signal tmp_3487[3] <== [tmp_3351[0] * 9589775313463224365, tmp_3351[1] * 9589775313463224365, tmp_3351[2] * 9589775313463224365];
    signal tmp_3488[3] <== [tmp_2966[0] * 13152929999122219197, tmp_2966[1] * 13152929999122219197, tmp_2966[2] * 13152929999122219197];
    signal tmp_3489[3] <== [tmp_3487[0] + tmp_3488[0], tmp_3487[1] + tmp_3488[1], tmp_3487[2] + tmp_3488[2]];
    signal tmp_3490[3] <== [evals[108][0] * 18140292631504202243, evals[108][1] * 18140292631504202243, evals[108][2] * 18140292631504202243];
    signal tmp_3491[3] <== [tmp_3489[0] + tmp_3490[0], tmp_3489[1] + tmp_3490[1], tmp_3489[2] + tmp_3490[2]];
    signal tmp_3492[3] <== [evals[52][0] * 15161788952257357966, evals[52][1] * 15161788952257357966, evals[52][2] * 15161788952257357966];
    signal tmp_3493[3] <== [tmp_3491[0] + tmp_3492[0], tmp_3491[1] + tmp_3492[1], tmp_3491[2] + tmp_3492[2]];
    signal tmp_3494[3] <== [evals[51][0] * 15658455328409267684, evals[51][1] * 15658455328409267684, evals[51][2] * 15658455328409267684];
    signal tmp_3495[3] <== [tmp_3493[0] + tmp_3494[0], tmp_3493[1] + tmp_3494[1], tmp_3493[2] + tmp_3494[2]];
    signal tmp_3496[3] <== [evals[78][0] + tmp_3495[0], evals[78][1] + tmp_3495[1], evals[78][2] + tmp_3495[2]];
    signal tmp_3497[3] <== [evals[78][0] + tmp_3495[0], evals[78][1] + tmp_3495[1], evals[78][2] + tmp_3495[2]];
    signal tmp_3498[3] <== CMul()(tmp_3496, tmp_3497);
    signal tmp_3499[3] <== CMul()(tmp_3498, tmp_3498);
    signal tmp_3500[3] <== CMul()(tmp_3499, tmp_3498);
    signal tmp_3501[3] <== [evals[78][0] + tmp_3495[0], evals[78][1] + tmp_3495[1], evals[78][2] + tmp_3495[2]];
    signal tmp_3502[3] <== CMul()(tmp_3500, tmp_3501);
    signal tmp_3503[3] <== [evals[79][0] + tmp_3440[0], evals[79][1] + tmp_3440[1], evals[79][2] + tmp_3440[2]];
    signal tmp_3504[3] <== CMul()(tmp_3445, tmp_3503);
    signal tmp_3505[3] <== [tmp_3502[0] + tmp_3504[0], tmp_3502[1] + tmp_3504[1], tmp_3502[2] + tmp_3504[2]];
    signal tmp_3506[3] <== [tmp_3486[0] + tmp_3505[0], tmp_3486[1] + tmp_3505[1], tmp_3486[2] + tmp_3505[2]];
    signal tmp_3507[3] <== [tmp_3483[0] + tmp_3506[0], tmp_3483[1] + tmp_3506[1], tmp_3483[2] + tmp_3506[2]];
    signal tmp_3508[3] <== [tmp_3482[0] + tmp_3507[0], tmp_3482[1] + tmp_3507[1], tmp_3482[2] + tmp_3507[2]];
    signal tmp_3509[3] <== [tmp_3431[0] + tmp_3508[0], tmp_3431[1] + tmp_3508[1], tmp_3431[2] + tmp_3508[2]];
    signal tmp_3510[3] <== [tmp_3351[0] * 5332470884919453534, tmp_3351[1] * 5332470884919453534, tmp_3351[2] * 5332470884919453534];
    signal tmp_3511[3] <== [tmp_2966[0] * 18323286026903235604, tmp_2966[1] * 18323286026903235604, tmp_2966[2] * 18323286026903235604];
    signal tmp_3512[3] <== [tmp_3510[0] + tmp_3511[0], tmp_3510[1] + tmp_3511[1], tmp_3510[2] + tmp_3511[2]];
    signal tmp_3513[3] <== [evals[108][0] * 3362219552562939863, evals[108][1] * 3362219552562939863, evals[108][2] * 3362219552562939863];
    signal tmp_3514[3] <== [tmp_3512[0] + tmp_3513[0], tmp_3512[1] + tmp_3513[1], tmp_3512[2] + tmp_3513[2]];
    signal tmp_3515[3] <== [evals[52][0] * 2161980224591127360, evals[52][1] * 2161980224591127360, evals[52][2] * 2161980224591127360];
    signal tmp_3516[3] <== [tmp_3514[0] + tmp_3515[0], tmp_3514[1] + tmp_3515[1], tmp_3514[2] + tmp_3515[2]];
    signal tmp_3517[3] <== [evals[51][0] * 9471330315555975806, evals[51][1] * 9471330315555975806, evals[51][2] * 9471330315555975806];
    signal tmp_3518[3] <== [tmp_3516[0] + tmp_3517[0], tmp_3516[1] + tmp_3517[1], tmp_3516[2] + tmp_3517[2]];
    signal tmp_3519[3] <== [evals[83][0] + tmp_3518[0], evals[83][1] + tmp_3518[1], evals[83][2] + tmp_3518[2]];
    signal tmp_3520[3] <== [evals[83][0] + tmp_3518[0], evals[83][1] + tmp_3518[1], evals[83][2] + tmp_3518[2]];
    signal tmp_3521[3] <== CMul()(tmp_3519, tmp_3520);
    signal tmp_3522[3] <== CMul()(tmp_3521, tmp_3521);
    signal tmp_3523[3] <== CMul()(tmp_3522, tmp_3521);
    signal tmp_3524[3] <== [evals[83][0] + tmp_3518[0], evals[83][1] + tmp_3518[1], evals[83][2] + tmp_3518[2]];
    signal tmp_3525[3] <== CMul()(tmp_3523, tmp_3524);
    signal tmp_3526[3] <== [2 * tmp_3525[0], 2 * tmp_3525[1], 2 * tmp_3525[2]];
    signal tmp_3527[3] <== [tmp_3351[0] * 2217569167061322248, tmp_3351[1] * 2217569167061322248, tmp_3351[2] * 2217569167061322248];
    signal tmp_3528[3] <== [tmp_2966[0] * 4974451914008050921, tmp_2966[1] * 4974451914008050921, tmp_2966[2] * 4974451914008050921];
    signal tmp_3529[3] <== [tmp_3527[0] + tmp_3528[0], tmp_3527[1] + tmp_3528[1], tmp_3527[2] + tmp_3528[2]];
    signal tmp_3530[3] <== [evals[108][0] * 6686302214424395771, evals[108][1] * 6686302214424395771, evals[108][2] * 6686302214424395771];
    signal tmp_3531[3] <== [tmp_3529[0] + tmp_3530[0], tmp_3529[1] + tmp_3530[1], tmp_3529[2] + tmp_3530[2]];
    signal tmp_3532[3] <== [evals[52][0] * 1282111773460545571, evals[52][1] * 1282111773460545571, evals[52][2] * 1282111773460545571];
    signal tmp_3533[3] <== [tmp_3531[0] + tmp_3532[0], tmp_3531[1] + tmp_3532[1], tmp_3531[2] + tmp_3532[2]];
    signal tmp_3534[3] <== [evals[51][0] * 2349868247408080783, evals[51][1] * 2349868247408080783, evals[51][2] * 2349868247408080783];
    signal tmp_3535[3] <== [tmp_3533[0] + tmp_3534[0], tmp_3533[1] + tmp_3534[1], tmp_3533[2] + tmp_3534[2]];
    signal tmp_3536[3] <== [evals[80][0] + tmp_3535[0], evals[80][1] + tmp_3535[1], evals[80][2] + tmp_3535[2]];
    signal tmp_3537[3] <== [evals[80][0] + tmp_3535[0], evals[80][1] + tmp_3535[1], evals[80][2] + tmp_3535[2]];
    signal tmp_3538[3] <== CMul()(tmp_3536, tmp_3537);
    signal tmp_3539[3] <== CMul()(tmp_3538, tmp_3538);
    signal tmp_3540[3] <== CMul()(tmp_3539, tmp_3538);
    signal tmp_3541[3] <== [evals[80][0] + tmp_3535[0], evals[80][1] + tmp_3535[1], evals[80][2] + tmp_3535[2]];
    signal tmp_3542[3] <== CMul()(tmp_3540, tmp_3541);
    signal tmp_3543[3] <== [tmp_3351[0] * 10394930802584583083, tmp_3351[1] * 10394930802584583083, tmp_3351[2] * 10394930802584583083];
    signal tmp_3544[3] <== [tmp_2966[0] * 11258703678970285201, tmp_2966[1] * 11258703678970285201, tmp_2966[2] * 11258703678970285201];
    signal tmp_3545[3] <== [tmp_3543[0] + tmp_3544[0], tmp_3543[1] + tmp_3544[1], tmp_3543[2] + tmp_3544[2]];
    signal tmp_3546[3] <== [evals[108][0] * 11193071888943695519, evals[108][1] * 11193071888943695519, evals[108][2] * 11193071888943695519];
    signal tmp_3547[3] <== [tmp_3545[0] + tmp_3546[0], tmp_3545[1] + tmp_3546[1], tmp_3545[2] + tmp_3546[2]];
    signal tmp_3548[3] <== [evals[52][0] * 8849495164481705550, evals[52][1] * 8849495164481705550, evals[52][2] * 8849495164481705550];
    signal tmp_3549[3] <== [tmp_3547[0] + tmp_3548[0], tmp_3547[1] + tmp_3548[1], tmp_3547[2] + tmp_3548[2]];
    signal tmp_3550[3] <== [evals[51][0] * 13105911261634181239, evals[51][1] * 13105911261634181239, evals[51][2] * 13105911261634181239];
    signal tmp_3551[3] <== [tmp_3549[0] + tmp_3550[0], tmp_3549[1] + tmp_3550[1], tmp_3549[2] + tmp_3550[2]];
    signal tmp_3552[3] <== [evals[81][0] + tmp_3551[0], evals[81][1] + tmp_3551[1], evals[81][2] + tmp_3551[2]];
    signal tmp_3553[3] <== [evals[81][0] + tmp_3551[0], evals[81][1] + tmp_3551[1], evals[81][2] + tmp_3551[2]];
    signal tmp_3554[3] <== CMul()(tmp_3552, tmp_3553);
    signal tmp_3555[3] <== CMul()(tmp_3554, tmp_3554);
    signal tmp_3556[3] <== CMul()(tmp_3555, tmp_3554);
    signal tmp_3557[3] <== [evals[81][0] + tmp_3551[0], evals[81][1] + tmp_3551[1], evals[81][2] + tmp_3551[2]];
    signal tmp_3558[3] <== CMul()(tmp_3556, tmp_3557);
    signal tmp_3559[3] <== [tmp_3542[0] + tmp_3558[0], tmp_3542[1] + tmp_3558[1], tmp_3542[2] + tmp_3558[2]];
    signal tmp_3560[3] <== [tmp_3526[0] + tmp_3559[0], tmp_3526[1] + tmp_3559[1], tmp_3526[2] + tmp_3559[2]];
    signal tmp_3561[3] <== [4 * tmp_3559[0], 4 * tmp_3559[1], 4 * tmp_3559[2]];
    signal tmp_3562[3] <== [evals[81][0] + tmp_3551[0], evals[81][1] + tmp_3551[1], evals[81][2] + tmp_3551[2]];
    signal tmp_3563[3] <== CMul()(tmp_3556, tmp_3562);
    signal tmp_3564[3] <== [2 * tmp_3563[0], 2 * tmp_3563[1], 2 * tmp_3563[2]];
    signal tmp_3565[3] <== [tmp_3351[0] * 4612393375016695705, tmp_3351[1] * 4612393375016695705, tmp_3351[2] * 4612393375016695705];
    signal tmp_3566[3] <== [tmp_2966[0] * 581736081259960204, tmp_2966[1] * 581736081259960204, tmp_2966[2] * 581736081259960204];
    signal tmp_3567[3] <== [tmp_3565[0] + tmp_3566[0], tmp_3565[1] + tmp_3566[1], tmp_3565[2] + tmp_3566[2]];
    signal tmp_3568[3] <== [evals[108][0] * 10233795775801758543, evals[108][1] * 10233795775801758543, evals[108][2] * 10233795775801758543];
    signal tmp_3569[3] <== [tmp_3567[0] + tmp_3568[0], tmp_3567[1] + tmp_3568[1], tmp_3567[2] + tmp_3568[2]];
    signal tmp_3570[3] <== [evals[52][0] * 8380852402060721190, evals[52][1] * 8380852402060721190, evals[52][2] * 8380852402060721190];
    signal tmp_3571[3] <== [tmp_3569[0] + tmp_3570[0], tmp_3569[1] + tmp_3570[1], tmp_3569[2] + tmp_3570[2]];
    signal tmp_3572[3] <== [evals[51][0] * 12868653202234053626, evals[51][1] * 12868653202234053626, evals[51][2] * 12868653202234053626];
    signal tmp_3573[3] <== [tmp_3571[0] + tmp_3572[0], tmp_3571[1] + tmp_3572[1], tmp_3571[2] + tmp_3572[2]];
    signal tmp_3574[3] <== [evals[82][0] + tmp_3573[0], evals[82][1] + tmp_3573[1], evals[82][2] + tmp_3573[2]];
    signal tmp_3575[3] <== [evals[82][0] + tmp_3573[0], evals[82][1] + tmp_3573[1], evals[82][2] + tmp_3573[2]];
    signal tmp_3576[3] <== CMul()(tmp_3574, tmp_3575);
    signal tmp_3577[3] <== CMul()(tmp_3576, tmp_3576);
    signal tmp_3578[3] <== CMul()(tmp_3577, tmp_3576);
    signal tmp_3579[3] <== [evals[82][0] + tmp_3573[0], evals[82][1] + tmp_3573[1], evals[82][2] + tmp_3573[2]];
    signal tmp_3580[3] <== CMul()(tmp_3578, tmp_3579);
    signal tmp_3581[3] <== [evals[83][0] + tmp_3518[0], evals[83][1] + tmp_3518[1], evals[83][2] + tmp_3518[2]];
    signal tmp_3582[3] <== CMul()(tmp_3523, tmp_3581);
    signal tmp_3583[3] <== [tmp_3580[0] + tmp_3582[0], tmp_3580[1] + tmp_3582[1], tmp_3580[2] + tmp_3582[2]];
    signal tmp_3584[3] <== [tmp_3564[0] + tmp_3583[0], tmp_3564[1] + tmp_3583[1], tmp_3564[2] + tmp_3583[2]];
    signal tmp_3585[3] <== [tmp_3561[0] + tmp_3584[0], tmp_3561[1] + tmp_3584[1], tmp_3561[2] + tmp_3584[2]];
    signal tmp_3586[3] <== [tmp_3560[0] + tmp_3585[0], tmp_3560[1] + tmp_3585[1], tmp_3560[2] + tmp_3585[2]];
    signal tmp_3587[3] <== [tmp_3509[0] + tmp_3586[0], tmp_3509[1] + tmp_3586[1], tmp_3509[2] + tmp_3586[2]];
    signal tmp_3588[3] <== [tmp_3431[0] + tmp_3587[0], tmp_3431[1] + tmp_3587[1], tmp_3431[2] + tmp_3587[2]];
    signal tmp_3589[3] <== [evals[84][0] - tmp_3588[0], evals[84][1] - tmp_3588[1], evals[84][2] - tmp_3588[2]];
    signal tmp_3590[3] <== CMul()(tmp_3354, tmp_3589);
    signal tmp_3591[3] <== [tmp_3350[0] + tmp_3590[0], tmp_3350[1] + tmp_3590[1], tmp_3350[2] + tmp_3590[2]];
    signal tmp_3592[3] <== CMul()(challengeQ, tmp_3591);
    signal tmp_3593[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3594[3] <== [tmp_3593[0] + evals[108][0], tmp_3593[1] + evals[108][1], tmp_3593[2] + evals[108][2]];
    signal tmp_3595[3] <== [tmp_3594[0] + evals[52][0], tmp_3594[1] + evals[52][1], tmp_3594[2] + evals[52][2]];
    signal tmp_3596[3] <== [tmp_3430[0] + tmp_3507[0], tmp_3430[1] + tmp_3507[1], tmp_3430[2] + tmp_3507[2]];
    signal tmp_3597[3] <== [tmp_3596[0] + tmp_3585[0], tmp_3596[1] + tmp_3585[1], tmp_3596[2] + tmp_3585[2]];
    signal tmp_3598[3] <== [tmp_3430[0] + tmp_3597[0], tmp_3430[1] + tmp_3597[1], tmp_3430[2] + tmp_3597[2]];
    signal tmp_3599[3] <== [evals[85][0] - tmp_3598[0], evals[85][1] - tmp_3598[1], evals[85][2] - tmp_3598[2]];
    signal tmp_3600[3] <== CMul()(tmp_3595, tmp_3599);
    signal tmp_3601[3] <== [tmp_3592[0] + tmp_3600[0], tmp_3592[1] + tmp_3600[1], tmp_3592[2] + tmp_3600[2]];
    signal tmp_3602[3] <== CMul()(challengeQ, tmp_3601);
    signal tmp_3603[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3604[3] <== [tmp_3603[0] + evals[108][0], tmp_3603[1] + evals[108][1], tmp_3603[2] + evals[108][2]];
    signal tmp_3605[3] <== [tmp_3604[0] + evals[52][0], tmp_3604[1] + evals[52][1], tmp_3604[2] + evals[52][2]];
    signal tmp_3606[3] <== [4 * tmp_3428[0], 4 * tmp_3428[1], 4 * tmp_3428[2]];
    signal tmp_3607[3] <== [tmp_3606[0] + tmp_3405[0], tmp_3606[1] + tmp_3405[1], tmp_3606[2] + tmp_3405[2]];
    signal tmp_3608[3] <== [tmp_3429[0] + tmp_3607[0], tmp_3429[1] + tmp_3607[1], tmp_3429[2] + tmp_3607[2]];
    signal tmp_3609[3] <== [4 * tmp_3505[0], 4 * tmp_3505[1], 4 * tmp_3505[2]];
    signal tmp_3610[3] <== [tmp_3609[0] + tmp_3482[0], tmp_3609[1] + tmp_3482[1], tmp_3609[2] + tmp_3482[2]];
    signal tmp_3611[3] <== [tmp_3506[0] + tmp_3610[0], tmp_3506[1] + tmp_3610[1], tmp_3506[2] + tmp_3610[2]];
    signal tmp_3612[3] <== [tmp_3608[0] + tmp_3611[0], tmp_3608[1] + tmp_3611[1], tmp_3608[2] + tmp_3611[2]];
    signal tmp_3613[3] <== [4 * tmp_3583[0], 4 * tmp_3583[1], 4 * tmp_3583[2]];
    signal tmp_3614[3] <== [tmp_3613[0] + tmp_3560[0], tmp_3613[1] + tmp_3560[1], tmp_3613[2] + tmp_3560[2]];
    signal tmp_3615[3] <== [tmp_3584[0] + tmp_3614[0], tmp_3584[1] + tmp_3614[1], tmp_3584[2] + tmp_3614[2]];
    signal tmp_3616[3] <== [tmp_3612[0] + tmp_3615[0], tmp_3612[1] + tmp_3615[1], tmp_3612[2] + tmp_3615[2]];
    signal tmp_3617[3] <== [tmp_3608[0] + tmp_3616[0], tmp_3608[1] + tmp_3616[1], tmp_3608[2] + tmp_3616[2]];
    signal tmp_3618[3] <== [evals[86][0] - tmp_3617[0], evals[86][1] - tmp_3617[1], evals[86][2] - tmp_3617[2]];
    signal tmp_3619[3] <== CMul()(tmp_3605, tmp_3618);
    signal tmp_3620[3] <== [tmp_3602[0] + tmp_3619[0], tmp_3602[1] + tmp_3619[1], tmp_3602[2] + tmp_3619[2]];
    signal tmp_3621[3] <== CMul()(challengeQ, tmp_3620);
    signal tmp_3622[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3623[3] <== [tmp_3622[0] + evals[108][0], tmp_3622[1] + evals[108][1], tmp_3622[2] + evals[108][2]];
    signal tmp_3624[3] <== [tmp_3623[0] + evals[52][0], tmp_3623[1] + evals[52][1], tmp_3623[2] + evals[52][2]];
    signal tmp_3625[3] <== [tmp_3607[0] + tmp_3610[0], tmp_3607[1] + tmp_3610[1], tmp_3607[2] + tmp_3610[2]];
    signal tmp_3626[3] <== [tmp_3625[0] + tmp_3614[0], tmp_3625[1] + tmp_3614[1], tmp_3625[2] + tmp_3614[2]];
    signal tmp_3627[3] <== [tmp_3607[0] + tmp_3626[0], tmp_3607[1] + tmp_3626[1], tmp_3607[2] + tmp_3626[2]];
    signal tmp_3628[3] <== [evals[87][0] - tmp_3627[0], evals[87][1] - tmp_3627[1], evals[87][2] - tmp_3627[2]];
    signal tmp_3629[3] <== CMul()(tmp_3624, tmp_3628);
    signal tmp_3630[3] <== [tmp_3621[0] + tmp_3629[0], tmp_3621[1] + tmp_3629[1], tmp_3621[2] + tmp_3629[2]];
    signal tmp_3631[3] <== CMul()(challengeQ, tmp_3630);
    signal tmp_3632[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3633[3] <== [tmp_3632[0] + evals[108][0], tmp_3632[1] + evals[108][1], tmp_3632[2] + evals[108][2]];
    signal tmp_3634[3] <== [tmp_3633[0] + evals[52][0], tmp_3633[1] + evals[52][1], tmp_3633[2] + evals[52][2]];
    signal tmp_3635[3] <== [tmp_3508[0] + tmp_3587[0], tmp_3508[1] + tmp_3587[1], tmp_3508[2] + tmp_3587[2]];
    signal tmp_3636[3] <== [evals[88][0] - tmp_3635[0], evals[88][1] - tmp_3635[1], evals[88][2] - tmp_3635[2]];
    signal tmp_3637[3] <== CMul()(tmp_3634, tmp_3636);
    signal tmp_3638[3] <== [tmp_3631[0] + tmp_3637[0], tmp_3631[1] + tmp_3637[1], tmp_3631[2] + tmp_3637[2]];
    signal tmp_3639[3] <== CMul()(challengeQ, tmp_3638);
    signal tmp_3640[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3641[3] <== [tmp_3640[0] + evals[108][0], tmp_3640[1] + evals[108][1], tmp_3640[2] + evals[108][2]];
    signal tmp_3642[3] <== [tmp_3641[0] + evals[52][0], tmp_3641[1] + evals[52][1], tmp_3641[2] + evals[52][2]];
    signal tmp_3643[3] <== [tmp_3507[0] + tmp_3597[0], tmp_3507[1] + tmp_3597[1], tmp_3507[2] + tmp_3597[2]];
    signal tmp_3644[3] <== [evals[89][0] - tmp_3643[0], evals[89][1] - tmp_3643[1], evals[89][2] - tmp_3643[2]];
    signal tmp_3645[3] <== CMul()(tmp_3642, tmp_3644);
    signal tmp_3646[3] <== [tmp_3639[0] + tmp_3645[0], tmp_3639[1] + tmp_3645[1], tmp_3639[2] + tmp_3645[2]];
    signal tmp_3647[3] <== CMul()(challengeQ, tmp_3646);
    signal tmp_3648[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3649[3] <== [tmp_3648[0] + evals[108][0], tmp_3648[1] + evals[108][1], tmp_3648[2] + evals[108][2]];
    signal tmp_3650[3] <== [tmp_3649[0] + evals[52][0], tmp_3649[1] + evals[52][1], tmp_3649[2] + evals[52][2]];
    signal tmp_3651[3] <== [tmp_3611[0] + tmp_3616[0], tmp_3611[1] + tmp_3616[1], tmp_3611[2] + tmp_3616[2]];
    signal tmp_3652[3] <== [evals[90][0] - tmp_3651[0], evals[90][1] - tmp_3651[1], evals[90][2] - tmp_3651[2]];
    signal tmp_3653[3] <== CMul()(tmp_3650, tmp_3652);
    signal tmp_3654[3] <== [tmp_3647[0] + tmp_3653[0], tmp_3647[1] + tmp_3653[1], tmp_3647[2] + tmp_3653[2]];
    signal tmp_3655[3] <== CMul()(challengeQ, tmp_3654);
    signal tmp_3656[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3657[3] <== [tmp_3656[0] + evals[108][0], tmp_3656[1] + evals[108][1], tmp_3656[2] + evals[108][2]];
    signal tmp_3658[3] <== [tmp_3657[0] + evals[52][0], tmp_3657[1] + evals[52][1], tmp_3657[2] + evals[52][2]];
    signal tmp_3659[3] <== [tmp_3610[0] + tmp_3626[0], tmp_3610[1] + tmp_3626[1], tmp_3610[2] + tmp_3626[2]];
    signal tmp_3660[3] <== [evals[91][0] - tmp_3659[0], evals[91][1] - tmp_3659[1], evals[91][2] - tmp_3659[2]];
    signal tmp_3661[3] <== CMul()(tmp_3658, tmp_3660);
    signal tmp_3662[3] <== [tmp_3655[0] + tmp_3661[0], tmp_3655[1] + tmp_3661[1], tmp_3655[2] + tmp_3661[2]];
    signal tmp_3663[3] <== CMul()(challengeQ, tmp_3662);
    signal tmp_3664[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3665[3] <== [tmp_3664[0] + evals[108][0], tmp_3664[1] + evals[108][1], tmp_3664[2] + evals[108][2]];
    signal tmp_3666[3] <== [tmp_3665[0] + evals[52][0], tmp_3665[1] + evals[52][1], tmp_3665[2] + evals[52][2]];
    signal tmp_3667[3] <== [tmp_3586[0] + tmp_3587[0], tmp_3586[1] + tmp_3587[1], tmp_3586[2] + tmp_3587[2]];
    signal tmp_3668[3] <== [evals[92][0] - tmp_3667[0], evals[92][1] - tmp_3667[1], evals[92][2] - tmp_3667[2]];
    signal tmp_3669[3] <== CMul()(tmp_3666, tmp_3668);
    signal tmp_3670[3] <== [tmp_3663[0] + tmp_3669[0], tmp_3663[1] + tmp_3669[1], tmp_3663[2] + tmp_3669[2]];
    signal tmp_3671[3] <== CMul()(challengeQ, tmp_3670);
    signal tmp_3672[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3673[3] <== [tmp_3672[0] + evals[108][0], tmp_3672[1] + evals[108][1], tmp_3672[2] + evals[108][2]];
    signal tmp_3674[3] <== [tmp_3673[0] + evals[52][0], tmp_3673[1] + evals[52][1], tmp_3673[2] + evals[52][2]];
    signal tmp_3675[3] <== [tmp_3585[0] + tmp_3597[0], tmp_3585[1] + tmp_3597[1], tmp_3585[2] + tmp_3597[2]];
    signal tmp_3676[3] <== [evals[93][0] - tmp_3675[0], evals[93][1] - tmp_3675[1], evals[93][2] - tmp_3675[2]];
    signal tmp_3677[3] <== CMul()(tmp_3674, tmp_3676);
    signal tmp_3678[3] <== [tmp_3671[0] + tmp_3677[0], tmp_3671[1] + tmp_3677[1], tmp_3671[2] + tmp_3677[2]];
    signal tmp_3679[3] <== CMul()(challengeQ, tmp_3678);
    signal tmp_3680[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3681[3] <== [tmp_3680[0] + evals[108][0], tmp_3680[1] + evals[108][1], tmp_3680[2] + evals[108][2]];
    signal tmp_3682[3] <== [tmp_3681[0] + evals[52][0], tmp_3681[1] + evals[52][1], tmp_3681[2] + evals[52][2]];
    signal tmp_3683[3] <== [tmp_3615[0] + tmp_3616[0], tmp_3615[1] + tmp_3616[1], tmp_3615[2] + tmp_3616[2]];
    signal tmp_3684[3] <== [evals[94][0] - tmp_3683[0], evals[94][1] - tmp_3683[1], evals[94][2] - tmp_3683[2]];
    signal tmp_3685[3] <== CMul()(tmp_3682, tmp_3684);
    signal tmp_3686[3] <== [tmp_3679[0] + tmp_3685[0], tmp_3679[1] + tmp_3685[1], tmp_3679[2] + tmp_3685[2]];
    signal tmp_3687[3] <== CMul()(challengeQ, tmp_3686);
    signal tmp_3688[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3689[3] <== [tmp_3688[0] + evals[108][0], tmp_3688[1] + evals[108][1], tmp_3688[2] + evals[108][2]];
    signal tmp_3690[3] <== [tmp_3689[0] + evals[52][0], tmp_3689[1] + evals[52][1], tmp_3689[2] + evals[52][2]];
    signal tmp_3691[3] <== [tmp_3614[0] + tmp_3626[0], tmp_3614[1] + tmp_3626[1], tmp_3614[2] + tmp_3626[2]];
    signal tmp_3692[3] <== [evals[95][0] - tmp_3691[0], evals[95][1] - tmp_3691[1], evals[95][2] - tmp_3691[2]];
    signal tmp_3693[3] <== CMul()(tmp_3690, tmp_3692);
    signal tmp_3694[3] <== [tmp_3687[0] + tmp_3693[0], tmp_3687[1] + tmp_3693[1], tmp_3687[2] + tmp_3693[2]];
    signal tmp_3695[3] <== CMul()(challengeQ, tmp_3694);
    signal tmp_3696[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3697[3] <== [tmp_3696[0] + evals[108][0], tmp_3696[1] + evals[108][1], tmp_3696[2] + evals[108][2]];
    signal tmp_3698[3] <== [tmp_3697[0] + evals[52][0], tmp_3697[1] + evals[52][1], tmp_3697[2] + evals[52][2]];
    signal tmp_3699[3] <== CMul()(evals[52], evals[60]);
    signal tmp_3700[3] <== CMul()(tmp_2966, evals[122]);
    signal tmp_3701[3] <== [tmp_3699[0] + tmp_3700[0], tmp_3699[1] + tmp_3700[1], tmp_3699[2] + tmp_3700[2]];
    signal tmp_3702[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_3703[3] <== [tmp_3702[0] - tmp_2966[0], tmp_3702[1] - tmp_2966[1], tmp_3702[2] - tmp_2966[2]];
    signal tmp_3704[3] <== CMul()(tmp_3703, evals[110]);
    signal tmp_3705[3] <== [tmp_3701[0] + tmp_3704[0], tmp_3701[1] + tmp_3704[1], tmp_3701[2] + tmp_3704[2]];
    signal tmp_3706[3] <== [tmp_3351[0] * 7999687124137420323, tmp_3351[1] * 7999687124137420323, tmp_3351[2] * 7999687124137420323];
    signal tmp_3707[3] <== [tmp_2966[0] * 11416990495425192684, tmp_2966[1] * 11416990495425192684, tmp_2966[2] * 11416990495425192684];
    signal tmp_3708[3] <== [tmp_3706[0] + tmp_3707[0], tmp_3706[1] + tmp_3707[1], tmp_3706[2] + tmp_3707[2]];
    signal tmp_3709[3] <== [evals[108][0] * 12517451587026875834, evals[108][1] * 12517451587026875834, evals[108][2] * 12517451587026875834];
    signal tmp_3710[3] <== [tmp_3708[0] + tmp_3709[0], tmp_3708[1] + tmp_3709[1], tmp_3708[2] + tmp_3709[2]];
    signal tmp_3711[3] <== [evals[52][0] * 17513705631114265826, evals[52][1] * 17513705631114265826, evals[52][2] * 17513705631114265826];
    signal tmp_3712[3] <== [tmp_3710[0] + tmp_3711[0], tmp_3710[1] + tmp_3711[1], tmp_3710[2] + tmp_3711[2]];
    signal tmp_3713[3] <== [evals[51][0] * 7619130111929922899, evals[51][1] * 7619130111929922899, evals[51][2] * 7619130111929922899];
    signal tmp_3714[3] <== [tmp_3712[0] + tmp_3713[0], tmp_3712[1] + tmp_3713[1], tmp_3712[2] + tmp_3713[2]];
    signal tmp_3715[3] <== [evals[87][0] + tmp_3714[0], evals[87][1] + tmp_3714[1], evals[87][2] + tmp_3714[2]];
    signal tmp_3716[3] <== [evals[87][0] + tmp_3714[0], evals[87][1] + tmp_3714[1], evals[87][2] + tmp_3714[2]];
    signal tmp_3717[3] <== CMul()(tmp_3715, tmp_3716);
    signal tmp_3718[3] <== CMul()(tmp_3717, tmp_3717);
    signal tmp_3719[3] <== CMul()(tmp_3718, tmp_3717);
    signal tmp_3720[3] <== [evals[87][0] + tmp_3714[0], evals[87][1] + tmp_3714[1], evals[87][2] + tmp_3714[2]];
    signal tmp_3721[3] <== CMul()(tmp_3719, tmp_3720);
    signal tmp_3722[3] <== [2 * tmp_3721[0], 2 * tmp_3721[1], 2 * tmp_3721[2]];
    signal tmp_3723[3] <== [tmp_3351[0] * 8724526834049581439, tmp_3351[1] * 8724526834049581439, tmp_3351[2] * 8724526834049581439];
    signal tmp_3724[3] <== [tmp_2966[0] * 10250026231324330997, tmp_2966[1] * 10250026231324330997, tmp_2966[2] * 10250026231324330997];
    signal tmp_3725[3] <== [tmp_3723[0] + tmp_3724[0], tmp_3723[1] + tmp_3724[1], tmp_3723[2] + tmp_3724[2]];
    signal tmp_3726[3] <== [evals[108][0] * 8595401306696186761, evals[108][1] * 8595401306696186761, evals[108][2] * 8595401306696186761];
    signal tmp_3727[3] <== [tmp_3725[0] + tmp_3726[0], tmp_3725[1] + tmp_3726[1], tmp_3725[2] + tmp_3726[2]];
    signal tmp_3728[3] <== [evals[52][0] * 2440151485689245146, evals[52][1] * 2440151485689245146, evals[52][2] * 2440151485689245146];
    signal tmp_3729[3] <== [tmp_3727[0] + tmp_3728[0], tmp_3727[1] + tmp_3728[1], tmp_3727[2] + tmp_3728[2]];
    signal tmp_3730[3] <== [evals[51][0] * 4580289636625406680, evals[51][1] * 4580289636625406680, evals[51][2] * 4580289636625406680];
    signal tmp_3731[3] <== [tmp_3729[0] + tmp_3730[0], tmp_3729[1] + tmp_3730[1], tmp_3729[2] + tmp_3730[2]];
    signal tmp_3732[3] <== [evals[84][0] + tmp_3731[0], evals[84][1] + tmp_3731[1], evals[84][2] + tmp_3731[2]];
    signal tmp_3733[3] <== [evals[84][0] + tmp_3731[0], evals[84][1] + tmp_3731[1], evals[84][2] + tmp_3731[2]];
    signal tmp_3734[3] <== CMul()(tmp_3732, tmp_3733);
    signal tmp_3735[3] <== CMul()(tmp_3734, tmp_3734);
    signal tmp_3736[3] <== CMul()(tmp_3735, tmp_3734);
    signal tmp_3737[3] <== [evals[84][0] + tmp_3731[0], evals[84][1] + tmp_3731[1], evals[84][2] + tmp_3731[2]];
    signal tmp_3738[3] <== CMul()(tmp_3736, tmp_3737);
    signal tmp_3739[3] <== [tmp_3351[0] * 17673787971454860688, tmp_3351[1] * 17673787971454860688, tmp_3351[2] * 17673787971454860688];
    signal tmp_3740[3] <== [tmp_2966[0] * 13321947507807660157, tmp_2966[1] * 13321947507807660157, tmp_2966[2] * 13321947507807660157];
    signal tmp_3741[3] <== [tmp_3739[0] + tmp_3740[0], tmp_3739[1] + tmp_3740[1], tmp_3739[2] + tmp_3740[2]];
    signal tmp_3742[3] <== [evals[108][0] * 7753411262943026561, evals[108][1] * 7753411262943026561, evals[108][2] * 7753411262943026561];
    signal tmp_3743[3] <== [tmp_3741[0] + tmp_3742[0], tmp_3741[1] + tmp_3742[1], tmp_3741[2] + tmp_3742[2]];
    signal tmp_3744[3] <== [evals[52][0] * 17521895002090134367, evals[52][1] * 17521895002090134367, evals[52][2] * 17521895002090134367];
    signal tmp_3745[3] <== [tmp_3743[0] + tmp_3744[0], tmp_3743[1] + tmp_3744[1], tmp_3743[2] + tmp_3744[2]];
    signal tmp_3746[3] <== [evals[51][0] * 13222733136951421572, evals[51][1] * 13222733136951421572, evals[51][2] * 13222733136951421572];
    signal tmp_3747[3] <== [tmp_3745[0] + tmp_3746[0], tmp_3745[1] + tmp_3746[1], tmp_3745[2] + tmp_3746[2]];
    signal tmp_3748[3] <== [evals[85][0] + tmp_3747[0], evals[85][1] + tmp_3747[1], evals[85][2] + tmp_3747[2]];
    signal tmp_3749[3] <== [evals[85][0] + tmp_3747[0], evals[85][1] + tmp_3747[1], evals[85][2] + tmp_3747[2]];
    signal tmp_3750[3] <== CMul()(tmp_3748, tmp_3749);
    signal tmp_3751[3] <== CMul()(tmp_3750, tmp_3750);
    signal tmp_3752[3] <== CMul()(tmp_3751, tmp_3750);
    signal tmp_3753[3] <== [evals[85][0] + tmp_3747[0], evals[85][1] + tmp_3747[1], evals[85][2] + tmp_3747[2]];
    signal tmp_3754[3] <== CMul()(tmp_3752, tmp_3753);
    signal tmp_3755[3] <== [tmp_3738[0] + tmp_3754[0], tmp_3738[1] + tmp_3754[1], tmp_3738[2] + tmp_3754[2]];
    signal tmp_3756[3] <== [tmp_3722[0] + tmp_3755[0], tmp_3722[1] + tmp_3755[1], tmp_3722[2] + tmp_3755[2]];
    signal tmp_3757[3] <== [4 * tmp_3755[0], 4 * tmp_3755[1], 4 * tmp_3755[2]];
    signal tmp_3758[3] <== [evals[85][0] + tmp_3747[0], evals[85][1] + tmp_3747[1], evals[85][2] + tmp_3747[2]];
    signal tmp_3759[3] <== CMul()(tmp_3752, tmp_3758);
    signal tmp_3760[3] <== [2 * tmp_3759[0], 2 * tmp_3759[1], 2 * tmp_3759[2]];
    signal tmp_3761[3] <== [tmp_3351[0] * 2519987773101056005, tmp_3351[1] * 2519987773101056005, tmp_3351[2] * 2519987773101056005];
    signal tmp_3762[3] <== [tmp_2966[0] * 13020725208899496943, tmp_2966[1] * 13020725208899496943, tmp_2966[2] * 13020725208899496943];
    signal tmp_3763[3] <== [tmp_3761[0] + tmp_3762[0], tmp_3761[1] + tmp_3762[1], tmp_3761[2] + tmp_3762[2]];
    signal tmp_3764[3] <== [evals[108][0] * 12415218859476220947, evals[108][1] * 12415218859476220947, evals[108][2] * 12415218859476220947];
    signal tmp_3765[3] <== [tmp_3763[0] + tmp_3764[0], tmp_3763[1] + tmp_3764[1], tmp_3763[2] + tmp_3764[2]];
    signal tmp_3766[3] <== [evals[52][0] * 13821005335130766955, evals[52][1] * 13821005335130766955, evals[52][2] * 13821005335130766955];
    signal tmp_3767[3] <== [tmp_3765[0] + tmp_3766[0], tmp_3765[1] + tmp_3766[1], tmp_3765[2] + tmp_3766[2]];
    signal tmp_3768[3] <== [evals[51][0] * 4555032575628627551, evals[51][1] * 4555032575628627551, evals[51][2] * 4555032575628627551];
    signal tmp_3769[3] <== [tmp_3767[0] + tmp_3768[0], tmp_3767[1] + tmp_3768[1], tmp_3767[2] + tmp_3768[2]];
    signal tmp_3770[3] <== [evals[86][0] + tmp_3769[0], evals[86][1] + tmp_3769[1], evals[86][2] + tmp_3769[2]];
    signal tmp_3771[3] <== [evals[86][0] + tmp_3769[0], evals[86][1] + tmp_3769[1], evals[86][2] + tmp_3769[2]];
    signal tmp_3772[3] <== CMul()(tmp_3770, tmp_3771);
    signal tmp_3773[3] <== CMul()(tmp_3772, tmp_3772);
    signal tmp_3774[3] <== CMul()(tmp_3773, tmp_3772);
    signal tmp_3775[3] <== [evals[86][0] + tmp_3769[0], evals[86][1] + tmp_3769[1], evals[86][2] + tmp_3769[2]];
    signal tmp_3776[3] <== CMul()(tmp_3774, tmp_3775);
    signal tmp_3777[3] <== [evals[87][0] + tmp_3714[0], evals[87][1] + tmp_3714[1], evals[87][2] + tmp_3714[2]];
    signal tmp_3778[3] <== CMul()(tmp_3719, tmp_3777);
    signal tmp_3779[3] <== [tmp_3776[0] + tmp_3778[0], tmp_3776[1] + tmp_3778[1], tmp_3776[2] + tmp_3778[2]];
    signal tmp_3780[3] <== [tmp_3760[0] + tmp_3779[0], tmp_3760[1] + tmp_3779[1], tmp_3760[2] + tmp_3779[2]];
    signal tmp_3781[3] <== [tmp_3757[0] + tmp_3780[0], tmp_3757[1] + tmp_3780[1], tmp_3757[2] + tmp_3780[2]];
    signal tmp_3782[3] <== [tmp_3756[0] + tmp_3781[0], tmp_3756[1] + tmp_3781[1], tmp_3756[2] + tmp_3781[2]];
    signal tmp_3783[3] <== [tmp_3351[0] * 5665449074466664773, tmp_3351[1] * 5665449074466664773, tmp_3351[2] * 5665449074466664773];
    signal tmp_3784[3] <== [tmp_2966[0] * 10485489452304998145, tmp_2966[1] * 10485489452304998145, tmp_2966[2] * 10485489452304998145];
    signal tmp_3785[3] <== [tmp_3783[0] + tmp_3784[0], tmp_3783[1] + tmp_3784[1], tmp_3783[2] + tmp_3784[2]];
    signal tmp_3786[3] <== [evals[108][0] * 8659969869470208989, evals[108][1] * 8659969869470208989, evals[108][2] * 8659969869470208989];
    signal tmp_3787[3] <== [tmp_3785[0] + tmp_3786[0], tmp_3785[1] + tmp_3786[1], tmp_3785[2] + tmp_3786[2]];
    signal tmp_3788[3] <== [evals[52][0] * 11615940660682589106, evals[52][1] * 11615940660682589106, evals[52][2] * 11615940660682589106];
    signal tmp_3789[3] <== [tmp_3787[0] + tmp_3788[0], tmp_3787[1] + tmp_3788[1], tmp_3787[2] + tmp_3788[2]];
    signal tmp_3790[3] <== [evals[51][0] * 13585630674756818185, evals[51][1] * 13585630674756818185, evals[51][2] * 13585630674756818185];
    signal tmp_3791[3] <== [tmp_3789[0] + tmp_3790[0], tmp_3789[1] + tmp_3790[1], tmp_3789[2] + tmp_3790[2]];
    signal tmp_3792[3] <== [evals[91][0] + tmp_3791[0], evals[91][1] + tmp_3791[1], evals[91][2] + tmp_3791[2]];
    signal tmp_3793[3] <== [evals[91][0] + tmp_3791[0], evals[91][1] + tmp_3791[1], evals[91][2] + tmp_3791[2]];
    signal tmp_3794[3] <== CMul()(tmp_3792, tmp_3793);
    signal tmp_3795[3] <== CMul()(tmp_3794, tmp_3794);
    signal tmp_3796[3] <== CMul()(tmp_3795, tmp_3794);
    signal tmp_3797[3] <== [evals[91][0] + tmp_3791[0], evals[91][1] + tmp_3791[1], evals[91][2] + tmp_3791[2]];
    signal tmp_3798[3] <== CMul()(tmp_3796, tmp_3797);
    signal tmp_3799[3] <== [2 * tmp_3798[0], 2 * tmp_3798[1], 2 * tmp_3798[2]];
    signal tmp_3800[3] <== [tmp_3351[0] * 18312454652563306701, tmp_3351[1] * 18312454652563306701, tmp_3351[2] * 18312454652563306701];
    signal tmp_3801[3] <== [tmp_2966[0] * 7221795794796219413, tmp_2966[1] * 7221795794796219413, tmp_2966[2] * 7221795794796219413];
    signal tmp_3802[3] <== [tmp_3800[0] + tmp_3801[0], tmp_3800[1] + tmp_3801[1], tmp_3800[2] + tmp_3801[2]];
    signal tmp_3803[3] <== [evals[108][0] * 3257008032900598499, evals[108][1] * 3257008032900598499, evals[108][2] * 3257008032900598499];
    signal tmp_3804[3] <== [tmp_3802[0] + tmp_3803[0], tmp_3802[1] + tmp_3803[1], tmp_3802[2] + tmp_3803[2]];
    signal tmp_3805[3] <== [evals[52][0] * 17068447856797239529, evals[52][1] * 17068447856797239529, evals[52][2] * 17068447856797239529];
    signal tmp_3806[3] <== [tmp_3804[0] + tmp_3805[0], tmp_3804[1] + tmp_3805[1], tmp_3804[2] + tmp_3805[2]];
    signal tmp_3807[3] <== [evals[51][0] * 4547848507246491777, evals[51][1] * 4547848507246491777, evals[51][2] * 4547848507246491777];
    signal tmp_3808[3] <== [tmp_3806[0] + tmp_3807[0], tmp_3806[1] + tmp_3807[1], tmp_3806[2] + tmp_3807[2]];
    signal tmp_3809[3] <== [evals[88][0] + tmp_3808[0], evals[88][1] + tmp_3808[1], evals[88][2] + tmp_3808[2]];
    signal tmp_3810[3] <== [evals[88][0] + tmp_3808[0], evals[88][1] + tmp_3808[1], evals[88][2] + tmp_3808[2]];
    signal tmp_3811[3] <== CMul()(tmp_3809, tmp_3810);
    signal tmp_3812[3] <== CMul()(tmp_3811, tmp_3811);
    signal tmp_3813[3] <== CMul()(tmp_3812, tmp_3811);
    signal tmp_3814[3] <== [evals[88][0] + tmp_3808[0], evals[88][1] + tmp_3808[1], evals[88][2] + tmp_3808[2]];
    signal tmp_3815[3] <== CMul()(tmp_3813, tmp_3814);
    signal tmp_3816[3] <== [tmp_3351[0] * 15136091233824155669, tmp_3351[1] * 15136091233824155669, tmp_3351[2] * 15136091233824155669];
    signal tmp_3817[3] <== [tmp_2966[0] * 2607917872900632985, tmp_2966[1] * 2607917872900632985, tmp_2966[2] * 2607917872900632985];
    signal tmp_3818[3] <== [tmp_3816[0] + tmp_3817[0], tmp_3816[1] + tmp_3817[1], tmp_3816[2] + tmp_3817[2]];
    signal tmp_3819[3] <== [evals[108][0] * 2187469039578904770, evals[108][1] * 2187469039578904770, evals[108][2] * 2187469039578904770];
    signal tmp_3820[3] <== [tmp_3818[0] + tmp_3819[0], tmp_3818[1] + tmp_3819[1], tmp_3818[2] + tmp_3819[2]];
    signal tmp_3821[3] <== [evals[52][0] * 17964439003977043993, evals[52][1] * 17964439003977043993, evals[52][2] * 17964439003977043993];
    signal tmp_3822[3] <== [tmp_3820[0] + tmp_3821[0], tmp_3820[1] + tmp_3821[1], tmp_3820[2] + tmp_3821[2]];
    signal tmp_3823[3] <== [evals[51][0] * 5662043532568004632, evals[51][1] * 5662043532568004632, evals[51][2] * 5662043532568004632];
    signal tmp_3824[3] <== [tmp_3822[0] + tmp_3823[0], tmp_3822[1] + tmp_3823[1], tmp_3822[2] + tmp_3823[2]];
    signal tmp_3825[3] <== [evals[89][0] + tmp_3824[0], evals[89][1] + tmp_3824[1], evals[89][2] + tmp_3824[2]];
    signal tmp_3826[3] <== [evals[89][0] + tmp_3824[0], evals[89][1] + tmp_3824[1], evals[89][2] + tmp_3824[2]];
    signal tmp_3827[3] <== CMul()(tmp_3825, tmp_3826);
    signal tmp_3828[3] <== CMul()(tmp_3827, tmp_3827);
    signal tmp_3829[3] <== CMul()(tmp_3828, tmp_3827);
    signal tmp_3830[3] <== [evals[89][0] + tmp_3824[0], evals[89][1] + tmp_3824[1], evals[89][2] + tmp_3824[2]];
    signal tmp_3831[3] <== CMul()(tmp_3829, tmp_3830);
    signal tmp_3832[3] <== [tmp_3815[0] + tmp_3831[0], tmp_3815[1] + tmp_3831[1], tmp_3815[2] + tmp_3831[2]];
    signal tmp_3833[3] <== [tmp_3799[0] + tmp_3832[0], tmp_3799[1] + tmp_3832[1], tmp_3799[2] + tmp_3832[2]];
    signal tmp_3834[3] <== [4 * tmp_3832[0], 4 * tmp_3832[1], 4 * tmp_3832[2]];
    signal tmp_3835[3] <== [evals[89][0] + tmp_3824[0], evals[89][1] + tmp_3824[1], evals[89][2] + tmp_3824[2]];
    signal tmp_3836[3] <== CMul()(tmp_3829, tmp_3835);
    signal tmp_3837[3] <== [2 * tmp_3836[0], 2 * tmp_3836[1], 2 * tmp_3836[2]];
    signal tmp_3838[3] <== [tmp_3351[0] * 1257110570403430003, tmp_3351[1] * 1257110570403430003, tmp_3351[2] * 1257110570403430003];
    signal tmp_3839[3] <== [tmp_2966[0] * 2591896057192169329, tmp_2966[1] * 2591896057192169329, tmp_2966[2] * 2591896057192169329];
    signal tmp_3840[3] <== [tmp_3838[0] + tmp_3839[0], tmp_3838[1] + tmp_3839[1], tmp_3838[2] + tmp_3839[2]];
    signal tmp_3841[3] <== [evals[108][0] * 657675168296710415, evals[108][1] * 657675168296710415, evals[108][2] * 657675168296710415];
    signal tmp_3842[3] <== [tmp_3840[0] + tmp_3841[0], tmp_3840[1] + tmp_3841[1], tmp_3840[2] + tmp_3841[2]];
    signal tmp_3843[3] <== [evals[52][0] * 5685000919538239429, evals[52][1] * 5685000919538239429, evals[52][2] * 5685000919538239429];
    signal tmp_3844[3] <== [tmp_3842[0] + tmp_3843[0], tmp_3842[1] + tmp_3843[1], tmp_3842[2] + tmp_3843[2]];
    signal tmp_3845[3] <== [evals[51][0] * 15723873049665279492, evals[51][1] * 15723873049665279492, evals[51][2] * 15723873049665279492];
    signal tmp_3846[3] <== [tmp_3844[0] + tmp_3845[0], tmp_3844[1] + tmp_3845[1], tmp_3844[2] + tmp_3845[2]];
    signal tmp_3847[3] <== [evals[90][0] + tmp_3846[0], evals[90][1] + tmp_3846[1], evals[90][2] + tmp_3846[2]];
    signal tmp_3848[3] <== [evals[90][0] + tmp_3846[0], evals[90][1] + tmp_3846[1], evals[90][2] + tmp_3846[2]];
    signal tmp_3849[3] <== CMul()(tmp_3847, tmp_3848);
    signal tmp_3850[3] <== CMul()(tmp_3849, tmp_3849);
    signal tmp_3851[3] <== CMul()(tmp_3850, tmp_3849);
    signal tmp_3852[3] <== [evals[90][0] + tmp_3846[0], evals[90][1] + tmp_3846[1], evals[90][2] + tmp_3846[2]];
    signal tmp_3853[3] <== CMul()(tmp_3851, tmp_3852);
    signal tmp_3854[3] <== [evals[91][0] + tmp_3791[0], evals[91][1] + tmp_3791[1], evals[91][2] + tmp_3791[2]];
    signal tmp_3855[3] <== CMul()(tmp_3796, tmp_3854);
    signal tmp_3856[3] <== [tmp_3853[0] + tmp_3855[0], tmp_3853[1] + tmp_3855[1], tmp_3853[2] + tmp_3855[2]];
    signal tmp_3857[3] <== [tmp_3837[0] + tmp_3856[0], tmp_3837[1] + tmp_3856[1], tmp_3837[2] + tmp_3856[2]];
    signal tmp_3858[3] <== [tmp_3834[0] + tmp_3857[0], tmp_3834[1] + tmp_3857[1], tmp_3834[2] + tmp_3857[2]];
    signal tmp_3859[3] <== [tmp_3833[0] + tmp_3858[0], tmp_3833[1] + tmp_3858[1], tmp_3833[2] + tmp_3858[2]];
    signal tmp_3860[3] <== [tmp_3782[0] + tmp_3859[0], tmp_3782[1] + tmp_3859[1], tmp_3782[2] + tmp_3859[2]];
    signal tmp_3861[3] <== [tmp_3351[0] * 2597062441266647183, tmp_3351[1] * 2597062441266647183, tmp_3351[2] * 2597062441266647183];
    signal tmp_3862[3] <== [tmp_2966[0] * 12203738590896308135, tmp_2966[1] * 12203738590896308135, tmp_2966[2] * 12203738590896308135];
    signal tmp_3863[3] <== [tmp_3861[0] + tmp_3862[0], tmp_3861[1] + tmp_3862[1], tmp_3861[2] + tmp_3862[2]];
    signal tmp_3864[3] <== [evals[108][0] * 7880966905416338909, evals[108][1] * 7880966905416338909, evals[108][2] * 7880966905416338909];
    signal tmp_3865[3] <== [tmp_3863[0] + tmp_3864[0], tmp_3863[1] + tmp_3864[1], tmp_3863[2] + tmp_3864[2]];
    signal tmp_3866[3] <== [evals[52][0] * 10821564568873127316, evals[52][1] * 10821564568873127316, evals[52][2] * 10821564568873127316];
    signal tmp_3867[3] <== [tmp_3865[0] + tmp_3866[0], tmp_3865[1] + tmp_3866[1], tmp_3865[2] + tmp_3866[2]];
    signal tmp_3868[3] <== [evals[51][0] * 17850970025369572891, evals[51][1] * 17850970025369572891, evals[51][2] * 17850970025369572891];
    signal tmp_3869[3] <== [tmp_3867[0] + tmp_3868[0], tmp_3867[1] + tmp_3868[1], tmp_3867[2] + tmp_3868[2]];
    signal tmp_3870[3] <== [evals[95][0] + tmp_3869[0], evals[95][1] + tmp_3869[1], evals[95][2] + tmp_3869[2]];
    signal tmp_3871[3] <== [evals[95][0] + tmp_3869[0], evals[95][1] + tmp_3869[1], evals[95][2] + tmp_3869[2]];
    signal tmp_3872[3] <== CMul()(tmp_3870, tmp_3871);
    signal tmp_3873[3] <== CMul()(tmp_3872, tmp_3872);
    signal tmp_3874[3] <== CMul()(tmp_3873, tmp_3872);
    signal tmp_3875[3] <== [evals[95][0] + tmp_3869[0], evals[95][1] + tmp_3869[1], evals[95][2] + tmp_3869[2]];
    signal tmp_3876[3] <== CMul()(tmp_3874, tmp_3875);
    signal tmp_3877[3] <== [2 * tmp_3876[0], 2 * tmp_3876[1], 2 * tmp_3876[2]];
    signal tmp_3878[3] <== [tmp_3351[0] * 16178737609685266571, tmp_3351[1] * 16178737609685266571, tmp_3351[2] * 16178737609685266571];
    signal tmp_3879[3] <== [tmp_2966[0] * 9480186048908910015, tmp_2966[1] * 9480186048908910015, tmp_2966[2] * 9480186048908910015];
    signal tmp_3880[3] <== [tmp_3878[0] + tmp_3879[0], tmp_3878[1] + tmp_3879[1], tmp_3878[2] + tmp_3879[2]];
    signal tmp_3881[3] <== [evals[108][0] * 12526098871288378639, evals[108][1] * 12526098871288378639, evals[108][2] * 12526098871288378639];
    signal tmp_3882[3] <== [tmp_3880[0] + tmp_3881[0], tmp_3880[1] + tmp_3881[1], tmp_3880[2] + tmp_3881[2]];
    signal tmp_3883[3] <== [evals[52][0] * 2522854885180605258, evals[52][1] * 2522854885180605258, evals[52][2] * 2522854885180605258];
    signal tmp_3884[3] <== [tmp_3882[0] + tmp_3883[0], tmp_3882[1] + tmp_3883[1], tmp_3882[2] + tmp_3883[2]];
    signal tmp_3885[3] <== [evals[51][0] * 6990417929677264473, evals[51][1] * 6990417929677264473, evals[51][2] * 6990417929677264473];
    signal tmp_3886[3] <== [tmp_3884[0] + tmp_3885[0], tmp_3884[1] + tmp_3885[1], tmp_3884[2] + tmp_3885[2]];
    signal tmp_3887[3] <== [evals[92][0] + tmp_3886[0], evals[92][1] + tmp_3886[1], evals[92][2] + tmp_3886[2]];
    signal tmp_3888[3] <== [evals[92][0] + tmp_3886[0], evals[92][1] + tmp_3886[1], evals[92][2] + tmp_3886[2]];
    signal tmp_3889[3] <== CMul()(tmp_3887, tmp_3888);
    signal tmp_3890[3] <== CMul()(tmp_3889, tmp_3889);
    signal tmp_3891[3] <== CMul()(tmp_3890, tmp_3889);
    signal tmp_3892[3] <== [evals[92][0] + tmp_3886[0], evals[92][1] + tmp_3886[1], evals[92][2] + tmp_3886[2]];
    signal tmp_3893[3] <== CMul()(tmp_3891, tmp_3892);
    signal tmp_3894[3] <== [tmp_3351[0] * 52855143527893348, tmp_3351[1] * 52855143527893348, tmp_3351[2] * 52855143527893348];
    signal tmp_3895[3] <== [tmp_2966[0] * 2645141845409940474, tmp_2966[1] * 2645141845409940474, tmp_2966[2] * 2645141845409940474];
    signal tmp_3896[3] <== [tmp_3894[0] + tmp_3895[0], tmp_3894[1] + tmp_3895[1], tmp_3894[2] + tmp_3895[2]];
    signal tmp_3897[3] <== [evals[108][0] * 12525853395769009329, evals[108][1] * 12525853395769009329, evals[108][2] * 12525853395769009329];
    signal tmp_3898[3] <== [tmp_3896[0] + tmp_3897[0], tmp_3896[1] + tmp_3897[1], tmp_3896[2] + tmp_3897[2]];
    signal tmp_3899[3] <== [evals[52][0] * 12584118968072796115, evals[52][1] * 12584118968072796115, evals[52][2] * 12584118968072796115];
    signal tmp_3900[3] <== [tmp_3898[0] + tmp_3899[0], tmp_3898[1] + tmp_3899[1], tmp_3898[2] + tmp_3899[2]];
    signal tmp_3901[3] <== [evals[51][0] * 6373257983538884779, evals[51][1] * 6373257983538884779, evals[51][2] * 6373257983538884779];
    signal tmp_3902[3] <== [tmp_3900[0] + tmp_3901[0], tmp_3900[1] + tmp_3901[1], tmp_3900[2] + tmp_3901[2]];
    signal tmp_3903[3] <== [evals[93][0] + tmp_3902[0], evals[93][1] + tmp_3902[1], evals[93][2] + tmp_3902[2]];
    signal tmp_3904[3] <== [evals[93][0] + tmp_3902[0], evals[93][1] + tmp_3902[1], evals[93][2] + tmp_3902[2]];
    signal tmp_3905[3] <== CMul()(tmp_3903, tmp_3904);
    signal tmp_3906[3] <== CMul()(tmp_3905, tmp_3905);
    signal tmp_3907[3] <== CMul()(tmp_3906, tmp_3905);
    signal tmp_3908[3] <== [evals[93][0] + tmp_3902[0], evals[93][1] + tmp_3902[1], evals[93][2] + tmp_3902[2]];
    signal tmp_3909[3] <== CMul()(tmp_3907, tmp_3908);
    signal tmp_3910[3] <== [tmp_3893[0] + tmp_3909[0], tmp_3893[1] + tmp_3909[1], tmp_3893[2] + tmp_3909[2]];
    signal tmp_3911[3] <== [tmp_3877[0] + tmp_3910[0], tmp_3877[1] + tmp_3910[1], tmp_3877[2] + tmp_3910[2]];
    signal tmp_3912[3] <== [4 * tmp_3910[0], 4 * tmp_3910[1], 4 * tmp_3910[2]];
    signal tmp_3913[3] <== [evals[93][0] + tmp_3902[0], evals[93][1] + tmp_3902[1], evals[93][2] + tmp_3902[2]];
    signal tmp_3914[3] <== CMul()(tmp_3907, tmp_3913);
    signal tmp_3915[3] <== [2 * tmp_3914[0], 2 * tmp_3914[1], 2 * tmp_3914[2]];
    signal tmp_3916[3] <== [tmp_3351[0] * 8084454992943870230, tmp_3351[1] * 8084454992943870230, tmp_3351[2] * 8084454992943870230];
    signal tmp_3917[3] <== [tmp_2966[0] * 16242299839765162610, tmp_2966[1] * 16242299839765162610, tmp_2966[2] * 16242299839765162610];
    signal tmp_3918[3] <== [tmp_3916[0] + tmp_3917[0], tmp_3916[1] + tmp_3917[1], tmp_3916[2] + tmp_3917[2]];
    signal tmp_3919[3] <== [evals[108][0] * 15388161689979551704, evals[108][1] * 15388161689979551704, evals[108][2] * 15388161689979551704];
    signal tmp_3920[3] <== [tmp_3918[0] + tmp_3919[0], tmp_3918[1] + tmp_3919[1], tmp_3918[2] + tmp_3919[2]];
    signal tmp_3921[3] <== [evals[52][0] * 17841258728624635591, evals[52][1] * 17841258728624635591, evals[52][2] * 17841258728624635591];
    signal tmp_3922[3] <== [tmp_3920[0] + tmp_3921[0], tmp_3920[1] + tmp_3921[1], tmp_3920[2] + tmp_3921[2]];
    signal tmp_3923[3] <== [evals[51][0] * 1005856792729125863, evals[51][1] * 1005856792729125863, evals[51][2] * 1005856792729125863];
    signal tmp_3924[3] <== [tmp_3922[0] + tmp_3923[0], tmp_3922[1] + tmp_3923[1], tmp_3922[2] + tmp_3923[2]];
    signal tmp_3925[3] <== [evals[94][0] + tmp_3924[0], evals[94][1] + tmp_3924[1], evals[94][2] + tmp_3924[2]];
    signal tmp_3926[3] <== [evals[94][0] + tmp_3924[0], evals[94][1] + tmp_3924[1], evals[94][2] + tmp_3924[2]];
    signal tmp_3927[3] <== CMul()(tmp_3925, tmp_3926);
    signal tmp_3928[3] <== CMul()(tmp_3927, tmp_3927);
    signal tmp_3929[3] <== CMul()(tmp_3928, tmp_3927);
    signal tmp_3930[3] <== [evals[94][0] + tmp_3924[0], evals[94][1] + tmp_3924[1], evals[94][2] + tmp_3924[2]];
    signal tmp_3931[3] <== CMul()(tmp_3929, tmp_3930);
    signal tmp_3932[3] <== [evals[95][0] + tmp_3869[0], evals[95][1] + tmp_3869[1], evals[95][2] + tmp_3869[2]];
    signal tmp_3933[3] <== CMul()(tmp_3874, tmp_3932);
    signal tmp_3934[3] <== [tmp_3931[0] + tmp_3933[0], tmp_3931[1] + tmp_3933[1], tmp_3931[2] + tmp_3933[2]];
    signal tmp_3935[3] <== [tmp_3915[0] + tmp_3934[0], tmp_3915[1] + tmp_3934[1], tmp_3915[2] + tmp_3934[2]];
    signal tmp_3936[3] <== [tmp_3912[0] + tmp_3935[0], tmp_3912[1] + tmp_3935[1], tmp_3912[2] + tmp_3935[2]];
    signal tmp_3937[3] <== [tmp_3911[0] + tmp_3936[0], tmp_3911[1] + tmp_3936[1], tmp_3911[2] + tmp_3936[2]];
    signal tmp_3938[3] <== [tmp_3860[0] + tmp_3937[0], tmp_3860[1] + tmp_3937[1], tmp_3860[2] + tmp_3937[2]];
    signal tmp_3939[3] <== [tmp_3782[0] + tmp_3938[0], tmp_3782[1] + tmp_3938[1], tmp_3782[2] + tmp_3938[2]];
    signal tmp_3940[3] <== [tmp_3705[0] - tmp_3939[0], tmp_3705[1] - tmp_3939[1], tmp_3705[2] - tmp_3939[2]];
    signal tmp_3941[3] <== CMul()(tmp_3698, tmp_3940);
    signal tmp_3942[3] <== [tmp_3695[0] + tmp_3941[0], tmp_3695[1] + tmp_3941[1], tmp_3695[2] + tmp_3941[2]];
    signal tmp_3943[3] <== CMul()(challengeQ, tmp_3942);
    signal tmp_3944[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3945[3] <== [tmp_3944[0] + evals[108][0], tmp_3944[1] + evals[108][1], tmp_3944[2] + evals[108][2]];
    signal tmp_3946[3] <== [tmp_3945[0] + evals[52][0], tmp_3945[1] + evals[52][1], tmp_3945[2] + evals[52][2]];
    signal tmp_3947[3] <== CMul()(evals[52], evals[61]);
    signal tmp_3948[3] <== CMul()(tmp_2966, evals[123]);
    signal tmp_3949[3] <== [tmp_3947[0] + tmp_3948[0], tmp_3947[1] + tmp_3948[1], tmp_3947[2] + tmp_3948[2]];
    signal tmp_3950[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_3951[3] <== [tmp_3950[0] - tmp_2966[0], tmp_3950[1] - tmp_2966[1], tmp_3950[2] - tmp_2966[2]];
    signal tmp_3952[3] <== CMul()(tmp_3951, evals[111]);
    signal tmp_3953[3] <== [tmp_3949[0] + tmp_3952[0], tmp_3949[1] + tmp_3952[1], tmp_3949[2] + tmp_3952[2]];
    signal tmp_3954[3] <== [tmp_3781[0] + tmp_3858[0], tmp_3781[1] + tmp_3858[1], tmp_3781[2] + tmp_3858[2]];
    signal tmp_3955[3] <== [tmp_3954[0] + tmp_3936[0], tmp_3954[1] + tmp_3936[1], tmp_3954[2] + tmp_3936[2]];
    signal tmp_3956[3] <== [tmp_3781[0] + tmp_3955[0], tmp_3781[1] + tmp_3955[1], tmp_3781[2] + tmp_3955[2]];
    signal tmp_3957[3] <== [tmp_3953[0] - tmp_3956[0], tmp_3953[1] - tmp_3956[1], tmp_3953[2] - tmp_3956[2]];
    signal tmp_3958[3] <== CMul()(tmp_3946, tmp_3957);
    signal tmp_3959[3] <== [tmp_3943[0] + tmp_3958[0], tmp_3943[1] + tmp_3958[1], tmp_3943[2] + tmp_3958[2]];
    signal tmp_3960[3] <== CMul()(challengeQ, tmp_3959);
    signal tmp_3961[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3962[3] <== [tmp_3961[0] + evals[108][0], tmp_3961[1] + evals[108][1], tmp_3961[2] + evals[108][2]];
    signal tmp_3963[3] <== [tmp_3962[0] + evals[52][0], tmp_3962[1] + evals[52][1], tmp_3962[2] + evals[52][2]];
    signal tmp_3964[3] <== CMul()(evals[52], evals[62]);
    signal tmp_3965[3] <== CMul()(tmp_2966, evals[124]);
    signal tmp_3966[3] <== [tmp_3964[0] + tmp_3965[0], tmp_3964[1] + tmp_3965[1], tmp_3964[2] + tmp_3965[2]];
    signal tmp_3967[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_3968[3] <== [tmp_3967[0] - tmp_2966[0], tmp_3967[1] - tmp_2966[1], tmp_3967[2] - tmp_2966[2]];
    signal tmp_3969[3] <== CMul()(tmp_3968, evals[112]);
    signal tmp_3970[3] <== [tmp_3966[0] + tmp_3969[0], tmp_3966[1] + tmp_3969[1], tmp_3966[2] + tmp_3969[2]];
    signal tmp_3971[3] <== [4 * tmp_3779[0], 4 * tmp_3779[1], 4 * tmp_3779[2]];
    signal tmp_3972[3] <== [tmp_3971[0] + tmp_3756[0], tmp_3971[1] + tmp_3756[1], tmp_3971[2] + tmp_3756[2]];
    signal tmp_3973[3] <== [tmp_3780[0] + tmp_3972[0], tmp_3780[1] + tmp_3972[1], tmp_3780[2] + tmp_3972[2]];
    signal tmp_3974[3] <== [4 * tmp_3856[0], 4 * tmp_3856[1], 4 * tmp_3856[2]];
    signal tmp_3975[3] <== [tmp_3974[0] + tmp_3833[0], tmp_3974[1] + tmp_3833[1], tmp_3974[2] + tmp_3833[2]];
    signal tmp_3976[3] <== [tmp_3857[0] + tmp_3975[0], tmp_3857[1] + tmp_3975[1], tmp_3857[2] + tmp_3975[2]];
    signal tmp_3977[3] <== [tmp_3973[0] + tmp_3976[0], tmp_3973[1] + tmp_3976[1], tmp_3973[2] + tmp_3976[2]];
    signal tmp_3978[3] <== [4 * tmp_3934[0], 4 * tmp_3934[1], 4 * tmp_3934[2]];
    signal tmp_3979[3] <== [tmp_3978[0] + tmp_3911[0], tmp_3978[1] + tmp_3911[1], tmp_3978[2] + tmp_3911[2]];
    signal tmp_3980[3] <== [tmp_3935[0] + tmp_3979[0], tmp_3935[1] + tmp_3979[1], tmp_3935[2] + tmp_3979[2]];
    signal tmp_3981[3] <== [tmp_3977[0] + tmp_3980[0], tmp_3977[1] + tmp_3980[1], tmp_3977[2] + tmp_3980[2]];
    signal tmp_3982[3] <== [tmp_3973[0] + tmp_3981[0], tmp_3973[1] + tmp_3981[1], tmp_3973[2] + tmp_3981[2]];
    signal tmp_3983[3] <== [tmp_3970[0] - tmp_3982[0], tmp_3970[1] - tmp_3982[1], tmp_3970[2] - tmp_3982[2]];
    signal tmp_3984[3] <== CMul()(tmp_3963, tmp_3983);
    signal tmp_3985[3] <== [tmp_3960[0] + tmp_3984[0], tmp_3960[1] + tmp_3984[1], tmp_3960[2] + tmp_3984[2]];
    signal tmp_3986[3] <== CMul()(challengeQ, tmp_3985);
    signal tmp_3987[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_3988[3] <== [tmp_3987[0] + evals[108][0], tmp_3987[1] + evals[108][1], tmp_3987[2] + evals[108][2]];
    signal tmp_3989[3] <== [tmp_3988[0] + evals[52][0], tmp_3988[1] + evals[52][1], tmp_3988[2] + evals[52][2]];
    signal tmp_3990[3] <== CMul()(evals[52], evals[63]);
    signal tmp_3991[3] <== CMul()(tmp_2966, evals[125]);
    signal tmp_3992[3] <== [tmp_3990[0] + tmp_3991[0], tmp_3990[1] + tmp_3991[1], tmp_3990[2] + tmp_3991[2]];
    signal tmp_3993[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_3994[3] <== [tmp_3993[0] - tmp_2966[0], tmp_3993[1] - tmp_2966[1], tmp_3993[2] - tmp_2966[2]];
    signal tmp_3995[3] <== CMul()(tmp_3994, evals[113]);
    signal tmp_3996[3] <== [tmp_3992[0] + tmp_3995[0], tmp_3992[1] + tmp_3995[1], tmp_3992[2] + tmp_3995[2]];
    signal tmp_3997[3] <== [tmp_3972[0] + tmp_3975[0], tmp_3972[1] + tmp_3975[1], tmp_3972[2] + tmp_3975[2]];
    signal tmp_3998[3] <== [tmp_3997[0] + tmp_3979[0], tmp_3997[1] + tmp_3979[1], tmp_3997[2] + tmp_3979[2]];
    signal tmp_3999[3] <== [tmp_3972[0] + tmp_3998[0], tmp_3972[1] + tmp_3998[1], tmp_3972[2] + tmp_3998[2]];
    signal tmp_4000[3] <== [tmp_3996[0] - tmp_3999[0], tmp_3996[1] - tmp_3999[1], tmp_3996[2] - tmp_3999[2]];
    signal tmp_4001[3] <== CMul()(tmp_3989, tmp_4000);
    signal tmp_4002[3] <== [tmp_3986[0] + tmp_4001[0], tmp_3986[1] + tmp_4001[1], tmp_3986[2] + tmp_4001[2]];
    signal tmp_4003[3] <== CMul()(challengeQ, tmp_4002);
    signal tmp_4004[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4005[3] <== [tmp_4004[0] + evals[108][0], tmp_4004[1] + evals[108][1], tmp_4004[2] + evals[108][2]];
    signal tmp_4006[3] <== [tmp_4005[0] + evals[52][0], tmp_4005[1] + evals[52][1], tmp_4005[2] + evals[52][2]];
    signal tmp_4007[3] <== CMul()(evals[52], evals[64]);
    signal tmp_4008[3] <== CMul()(tmp_2966, evals[126]);
    signal tmp_4009[3] <== [tmp_4007[0] + tmp_4008[0], tmp_4007[1] + tmp_4008[1], tmp_4007[2] + tmp_4008[2]];
    signal tmp_4010[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4011[3] <== [tmp_4010[0] - tmp_2966[0], tmp_4010[1] - tmp_2966[1], tmp_4010[2] - tmp_2966[2]];
    signal tmp_4012[3] <== CMul()(tmp_4011, evals[114]);
    signal tmp_4013[3] <== [tmp_4009[0] + tmp_4012[0], tmp_4009[1] + tmp_4012[1], tmp_4009[2] + tmp_4012[2]];
    signal tmp_4014[3] <== [tmp_3859[0] + tmp_3938[0], tmp_3859[1] + tmp_3938[1], tmp_3859[2] + tmp_3938[2]];
    signal tmp_4015[3] <== [tmp_4013[0] - tmp_4014[0], tmp_4013[1] - tmp_4014[1], tmp_4013[2] - tmp_4014[2]];
    signal tmp_4016[3] <== CMul()(tmp_4006, tmp_4015);
    signal tmp_4017[3] <== [tmp_4003[0] + tmp_4016[0], tmp_4003[1] + tmp_4016[1], tmp_4003[2] + tmp_4016[2]];
    signal tmp_4018[3] <== CMul()(challengeQ, tmp_4017);
    signal tmp_4019[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4020[3] <== [tmp_4019[0] + evals[108][0], tmp_4019[1] + evals[108][1], tmp_4019[2] + evals[108][2]];
    signal tmp_4021[3] <== [tmp_4020[0] + evals[52][0], tmp_4020[1] + evals[52][1], tmp_4020[2] + evals[52][2]];
    signal tmp_4022[3] <== CMul()(evals[52], evals[65]);
    signal tmp_4023[3] <== CMul()(tmp_2966, evals[127]);
    signal tmp_4024[3] <== [tmp_4022[0] + tmp_4023[0], tmp_4022[1] + tmp_4023[1], tmp_4022[2] + tmp_4023[2]];
    signal tmp_4025[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4026[3] <== [tmp_4025[0] - tmp_2966[0], tmp_4025[1] - tmp_2966[1], tmp_4025[2] - tmp_2966[2]];
    signal tmp_4027[3] <== CMul()(tmp_4026, evals[115]);
    signal tmp_4028[3] <== [tmp_4024[0] + tmp_4027[0], tmp_4024[1] + tmp_4027[1], tmp_4024[2] + tmp_4027[2]];
    signal tmp_4029[3] <== [tmp_3858[0] + tmp_3955[0], tmp_3858[1] + tmp_3955[1], tmp_3858[2] + tmp_3955[2]];
    signal tmp_4030[3] <== [tmp_4028[0] - tmp_4029[0], tmp_4028[1] - tmp_4029[1], tmp_4028[2] - tmp_4029[2]];
    signal tmp_4031[3] <== CMul()(tmp_4021, tmp_4030);
    signal tmp_4032[3] <== [tmp_4018[0] + tmp_4031[0], tmp_4018[1] + tmp_4031[1], tmp_4018[2] + tmp_4031[2]];
    signal tmp_4033[3] <== CMul()(challengeQ, tmp_4032);
    signal tmp_4034[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4035[3] <== [tmp_4034[0] + evals[108][0], tmp_4034[1] + evals[108][1], tmp_4034[2] + evals[108][2]];
    signal tmp_4036[3] <== [tmp_4035[0] + evals[52][0], tmp_4035[1] + evals[52][1], tmp_4035[2] + evals[52][2]];
    signal tmp_4037[3] <== CMul()(evals[52], evals[66]);
    signal tmp_4038[3] <== CMul()(tmp_2966, evals[128]);
    signal tmp_4039[3] <== [tmp_4037[0] + tmp_4038[0], tmp_4037[1] + tmp_4038[1], tmp_4037[2] + tmp_4038[2]];
    signal tmp_4040[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4041[3] <== [tmp_4040[0] - tmp_2966[0], tmp_4040[1] - tmp_2966[1], tmp_4040[2] - tmp_2966[2]];
    signal tmp_4042[3] <== CMul()(tmp_4041, evals[116]);
    signal tmp_4043[3] <== [tmp_4039[0] + tmp_4042[0], tmp_4039[1] + tmp_4042[1], tmp_4039[2] + tmp_4042[2]];
    signal tmp_4044[3] <== [tmp_3976[0] + tmp_3981[0], tmp_3976[1] + tmp_3981[1], tmp_3976[2] + tmp_3981[2]];
    signal tmp_4045[3] <== [tmp_4043[0] - tmp_4044[0], tmp_4043[1] - tmp_4044[1], tmp_4043[2] - tmp_4044[2]];
    signal tmp_4046[3] <== CMul()(tmp_4036, tmp_4045);
    signal tmp_4047[3] <== [tmp_4033[0] + tmp_4046[0], tmp_4033[1] + tmp_4046[1], tmp_4033[2] + tmp_4046[2]];
    signal tmp_4048[3] <== CMul()(challengeQ, tmp_4047);
    signal tmp_4049[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4050[3] <== [tmp_4049[0] + evals[108][0], tmp_4049[1] + evals[108][1], tmp_4049[2] + evals[108][2]];
    signal tmp_4051[3] <== [tmp_4050[0] + evals[52][0], tmp_4050[1] + evals[52][1], tmp_4050[2] + evals[52][2]];
    signal tmp_4052[3] <== CMul()(evals[52], evals[67]);
    signal tmp_4053[3] <== CMul()(tmp_2966, evals[129]);
    signal tmp_4054[3] <== [tmp_4052[0] + tmp_4053[0], tmp_4052[1] + tmp_4053[1], tmp_4052[2] + tmp_4053[2]];
    signal tmp_4055[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4056[3] <== [tmp_4055[0] - tmp_2966[0], tmp_4055[1] - tmp_2966[1], tmp_4055[2] - tmp_2966[2]];
    signal tmp_4057[3] <== CMul()(tmp_4056, evals[117]);
    signal tmp_4058[3] <== [tmp_4054[0] + tmp_4057[0], tmp_4054[1] + tmp_4057[1], tmp_4054[2] + tmp_4057[2]];
    signal tmp_4059[3] <== [tmp_3975[0] + tmp_3998[0], tmp_3975[1] + tmp_3998[1], tmp_3975[2] + tmp_3998[2]];
    signal tmp_4060[3] <== [tmp_4058[0] - tmp_4059[0], tmp_4058[1] - tmp_4059[1], tmp_4058[2] - tmp_4059[2]];
    signal tmp_4061[3] <== CMul()(tmp_4051, tmp_4060);
    signal tmp_4062[3] <== [tmp_4048[0] + tmp_4061[0], tmp_4048[1] + tmp_4061[1], tmp_4048[2] + tmp_4061[2]];
    signal tmp_4063[3] <== CMul()(challengeQ, tmp_4062);
    signal tmp_4064[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4065[3] <== [tmp_4064[0] + evals[108][0], tmp_4064[1] + evals[108][1], tmp_4064[2] + evals[108][2]];
    signal tmp_4066[3] <== [tmp_4065[0] + evals[52][0], tmp_4065[1] + evals[52][1], tmp_4065[2] + evals[52][2]];
    signal tmp_4067[3] <== CMul()(evals[52], evals[68]);
    signal tmp_4068[3] <== CMul()(tmp_2966, evals[130]);
    signal tmp_4069[3] <== [tmp_4067[0] + tmp_4068[0], tmp_4067[1] + tmp_4068[1], tmp_4067[2] + tmp_4068[2]];
    signal tmp_4070[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4071[3] <== [tmp_4070[0] - tmp_2966[0], tmp_4070[1] - tmp_2966[1], tmp_4070[2] - tmp_2966[2]];
    signal tmp_4072[3] <== CMul()(tmp_4071, evals[118]);
    signal tmp_4073[3] <== [tmp_4069[0] + tmp_4072[0], tmp_4069[1] + tmp_4072[1], tmp_4069[2] + tmp_4072[2]];
    signal tmp_4074[3] <== [tmp_3937[0] + tmp_3938[0], tmp_3937[1] + tmp_3938[1], tmp_3937[2] + tmp_3938[2]];
    signal tmp_4075[3] <== [tmp_4073[0] - tmp_4074[0], tmp_4073[1] - tmp_4074[1], tmp_4073[2] - tmp_4074[2]];
    signal tmp_4076[3] <== CMul()(tmp_4066, tmp_4075);
    signal tmp_4077[3] <== [tmp_4063[0] + tmp_4076[0], tmp_4063[1] + tmp_4076[1], tmp_4063[2] + tmp_4076[2]];
    signal tmp_4078[3] <== CMul()(challengeQ, tmp_4077);
    signal tmp_4079[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4080[3] <== [tmp_4079[0] + evals[108][0], tmp_4079[1] + evals[108][1], tmp_4079[2] + evals[108][2]];
    signal tmp_4081[3] <== [tmp_4080[0] + evals[52][0], tmp_4080[1] + evals[52][1], tmp_4080[2] + evals[52][2]];
    signal tmp_4082[3] <== CMul()(evals[52], evals[69]);
    signal tmp_4083[3] <== CMul()(tmp_2966, evals[131]);
    signal tmp_4084[3] <== [tmp_4082[0] + tmp_4083[0], tmp_4082[1] + tmp_4083[1], tmp_4082[2] + tmp_4083[2]];
    signal tmp_4085[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4086[3] <== [tmp_4085[0] - tmp_2966[0], tmp_4085[1] - tmp_2966[1], tmp_4085[2] - tmp_2966[2]];
    signal tmp_4087[3] <== CMul()(tmp_4086, evals[119]);
    signal tmp_4088[3] <== [tmp_4084[0] + tmp_4087[0], tmp_4084[1] + tmp_4087[1], tmp_4084[2] + tmp_4087[2]];
    signal tmp_4089[3] <== [tmp_3936[0] + tmp_3955[0], tmp_3936[1] + tmp_3955[1], tmp_3936[2] + tmp_3955[2]];
    signal tmp_4090[3] <== [tmp_4088[0] - tmp_4089[0], tmp_4088[1] - tmp_4089[1], tmp_4088[2] - tmp_4089[2]];
    signal tmp_4091[3] <== CMul()(tmp_4081, tmp_4090);
    signal tmp_4092[3] <== [tmp_4078[0] + tmp_4091[0], tmp_4078[1] + tmp_4091[1], tmp_4078[2] + tmp_4091[2]];
    signal tmp_4093[3] <== CMul()(challengeQ, tmp_4092);
    signal tmp_4094[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4095[3] <== [tmp_4094[0] + evals[108][0], tmp_4094[1] + evals[108][1], tmp_4094[2] + evals[108][2]];
    signal tmp_4096[3] <== [tmp_4095[0] + evals[52][0], tmp_4095[1] + evals[52][1], tmp_4095[2] + evals[52][2]];
    signal tmp_4097[3] <== CMul()(evals[52], evals[70]);
    signal tmp_4098[3] <== CMul()(tmp_2966, evals[132]);
    signal tmp_4099[3] <== [tmp_4097[0] + tmp_4098[0], tmp_4097[1] + tmp_4098[1], tmp_4097[2] + tmp_4098[2]];
    signal tmp_4100[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4101[3] <== [tmp_4100[0] - tmp_2966[0], tmp_4100[1] - tmp_2966[1], tmp_4100[2] - tmp_2966[2]];
    signal tmp_4102[3] <== CMul()(tmp_4101, evals[120]);
    signal tmp_4103[3] <== [tmp_4099[0] + tmp_4102[0], tmp_4099[1] + tmp_4102[1], tmp_4099[2] + tmp_4102[2]];
    signal tmp_4104[3] <== [tmp_3980[0] + tmp_3981[0], tmp_3980[1] + tmp_3981[1], tmp_3980[2] + tmp_3981[2]];
    signal tmp_4105[3] <== [tmp_4103[0] - tmp_4104[0], tmp_4103[1] - tmp_4104[1], tmp_4103[2] - tmp_4104[2]];
    signal tmp_4106[3] <== CMul()(tmp_4096, tmp_4105);
    signal tmp_4107[3] <== [tmp_4093[0] + tmp_4106[0], tmp_4093[1] + tmp_4106[1], tmp_4093[2] + tmp_4106[2]];
    signal tmp_4108[3] <== CMul()(challengeQ, tmp_4107);
    signal tmp_4109[3] <== [tmp_3351[0] + tmp_2966[0], tmp_3351[1] + tmp_2966[1], tmp_3351[2] + tmp_2966[2]];
    signal tmp_4110[3] <== [tmp_4109[0] + evals[108][0], tmp_4109[1] + evals[108][1], tmp_4109[2] + evals[108][2]];
    signal tmp_4111[3] <== [tmp_4110[0] + evals[52][0], tmp_4110[1] + evals[52][1], tmp_4110[2] + evals[52][2]];
    signal tmp_4112[3] <== CMul()(evals[52], evals[71]);
    signal tmp_4113[3] <== CMul()(tmp_2966, evals[133]);
    signal tmp_4114[3] <== [tmp_4112[0] + tmp_4113[0], tmp_4112[1] + tmp_4113[1], tmp_4112[2] + tmp_4113[2]];
    signal tmp_4115[3] <== [1 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_4116[3] <== [tmp_4115[0] - tmp_2966[0], tmp_4115[1] - tmp_2966[1], tmp_4115[2] - tmp_2966[2]];
    signal tmp_4117[3] <== CMul()(tmp_4116, evals[121]);
    signal tmp_4118[3] <== [tmp_4114[0] + tmp_4117[0], tmp_4114[1] + tmp_4117[1], tmp_4114[2] + tmp_4117[2]];
    signal tmp_4119[3] <== [tmp_3979[0] + tmp_3998[0], tmp_3979[1] + tmp_3998[1], tmp_3979[2] + tmp_3998[2]];
    signal tmp_4120[3] <== [tmp_4118[0] - tmp_4119[0], tmp_4118[1] - tmp_4119[1], tmp_4118[2] - tmp_4119[2]];
    signal tmp_4121[3] <== CMul()(tmp_4111, tmp_4120);
    signal tmp_4122[3] <== [tmp_4108[0] + tmp_4121[0], tmp_4108[1] + tmp_4121[1], tmp_4108[2] + tmp_4121[2]];
    signal tmp_4123[3] <== CMul()(challengeQ, tmp_4122);
    signal tmp_4124[3] <== [evals[74][0] - evals[2][0], evals[74][1] - evals[2][1], evals[74][2] - evals[2][2]];
    signal tmp_4125[3] <== CMul()(evals[51], tmp_4124);
    signal tmp_4126[3] <== [tmp_4123[0] + tmp_4125[0], tmp_4123[1] + tmp_4125[1], tmp_4123[2] + tmp_4125[2]];
    signal tmp_4127[3] <== CMul()(challengeQ, tmp_4126);
    signal tmp_4128[3] <== [evals[74][0] + tmp_3418[0], evals[74][1] + tmp_3418[1], evals[74][2] + tmp_3418[2]];
    signal tmp_4129[3] <== CMul()(tmp_3423, tmp_4128);
    signal tmp_4130[3] <== [tmp_4129[0] * 14102670999874605824, tmp_4129[1] * 14102670999874605824, tmp_4129[2] * 14102670999874605824];
    signal tmp_4131[3] <== [tmp_4129[0] + evals[3][0], tmp_4129[1] + evals[3][1], tmp_4129[2] + evals[3][2]];
    signal tmp_4132[3] <== [tmp_4131[0] + evals[4][0], tmp_4131[1] + evals[4][1], tmp_4131[2] + evals[4][2]];
    signal tmp_4133[3] <== [tmp_4132[0] + evals[5][0], tmp_4132[1] + evals[5][1], tmp_4132[2] + evals[5][2]];
    signal tmp_4134[3] <== [tmp_4133[0] + evals[6][0], tmp_4133[1] + evals[6][1], tmp_4133[2] + evals[6][2]];
    signal tmp_4135[3] <== [tmp_4134[0] + evals[7][0], tmp_4134[1] + evals[7][1], tmp_4134[2] + evals[7][2]];
    signal tmp_4136[3] <== [tmp_4135[0] + evals[8][0], tmp_4135[1] + evals[8][1], tmp_4135[2] + evals[8][2]];
    signal tmp_4137[3] <== [tmp_4136[0] + evals[9][0], tmp_4136[1] + evals[9][1], tmp_4136[2] + evals[9][2]];
    signal tmp_4138[3] <== [tmp_4137[0] + evals[10][0], tmp_4137[1] + evals[10][1], tmp_4137[2] + evals[10][2]];
    signal tmp_4139[3] <== [tmp_4138[0] + evals[11][0], tmp_4138[1] + evals[11][1], tmp_4138[2] + evals[11][2]];
    signal tmp_4140[3] <== [tmp_4139[0] + evals[12][0], tmp_4139[1] + evals[12][1], tmp_4139[2] + evals[12][2]];
    signal tmp_4141[3] <== [tmp_4140[0] + evals[13][0], tmp_4140[1] + evals[13][1], tmp_4140[2] + evals[13][2]];
    signal tmp_4142[3] <== [tmp_4130[0] + tmp_4141[0], tmp_4130[1] + tmp_4141[1], tmp_4130[2] + tmp_4141[2]];
    signal tmp_4143[3] <== [evals[75][0] - tmp_4142[0], evals[75][1] - tmp_4142[1], evals[75][2] - tmp_4142[2]];
    signal tmp_4144[3] <== CMul()(evals[51], tmp_4143);
    signal tmp_4145[3] <== [tmp_4127[0] + tmp_4144[0], tmp_4127[1] + tmp_4144[1], tmp_4127[2] + tmp_4144[2]];
    signal tmp_4146[3] <== CMul()(challengeQ, tmp_4145);
    signal tmp_4147[3] <== [evals[75][0] + tmp_3363[0], evals[75][1] + tmp_3363[1], evals[75][2] + tmp_3363[2]];
    signal tmp_4148[3] <== CMul()(tmp_3368, tmp_4147);
    signal tmp_4149[3] <== [tmp_4148[0] * 14102670999874605824, tmp_4148[1] * 14102670999874605824, tmp_4148[2] * 14102670999874605824];
    signal tmp_4150[3] <== [evals[3][0] * 15585654191999307702, evals[3][1] * 15585654191999307702, evals[3][2] * 15585654191999307702];
    signal tmp_4151[3] <== [tmp_4150[0] + tmp_4141[0], tmp_4150[1] + tmp_4141[1], tmp_4150[2] + tmp_4141[2]];
    signal tmp_4152[3] <== [tmp_4148[0] + tmp_4151[0], tmp_4148[1] + tmp_4151[1], tmp_4148[2] + tmp_4151[2]];
    signal tmp_4153[3] <== [evals[4][0] * 940187017142450255, evals[4][1] * 940187017142450255, evals[4][2] * 940187017142450255];
    signal tmp_4154[3] <== [tmp_4153[0] + tmp_4141[0], tmp_4153[1] + tmp_4141[1], tmp_4153[2] + tmp_4141[2]];
    signal tmp_4155[3] <== [tmp_4152[0] + tmp_4154[0], tmp_4152[1] + tmp_4154[1], tmp_4152[2] + tmp_4154[2]];
    signal tmp_4156[3] <== [evals[5][0] * 8747386241522630711, evals[5][1] * 8747386241522630711, evals[5][2] * 8747386241522630711];
    signal tmp_4157[3] <== [tmp_4156[0] + tmp_4141[0], tmp_4156[1] + tmp_4141[1], tmp_4156[2] + tmp_4141[2]];
    signal tmp_4158[3] <== [tmp_4155[0] + tmp_4157[0], tmp_4155[1] + tmp_4157[1], tmp_4155[2] + tmp_4157[2]];
    signal tmp_4159[3] <== [evals[6][0] * 6750641561540124747, evals[6][1] * 6750641561540124747, evals[6][2] * 6750641561540124747];
    signal tmp_4160[3] <== [tmp_4159[0] + tmp_4141[0], tmp_4159[1] + tmp_4141[1], tmp_4159[2] + tmp_4141[2]];
    signal tmp_4161[3] <== [tmp_4158[0] + tmp_4160[0], tmp_4158[1] + tmp_4160[1], tmp_4158[2] + tmp_4160[2]];
    signal tmp_4162[3] <== [evals[7][0] * 7440998025584530007, evals[7][1] * 7440998025584530007, evals[7][2] * 7440998025584530007];
    signal tmp_4163[3] <== [tmp_4162[0] + tmp_4141[0], tmp_4162[1] + tmp_4141[1], tmp_4162[2] + tmp_4141[2]];
    signal tmp_4164[3] <== [tmp_4161[0] + tmp_4163[0], tmp_4161[1] + tmp_4163[1], tmp_4161[2] + tmp_4163[2]];
    signal tmp_4165[3] <== [evals[8][0] * 6136358134615751536, evals[8][1] * 6136358134615751536, evals[8][2] * 6136358134615751536];
    signal tmp_4166[3] <== [tmp_4165[0] + tmp_4141[0], tmp_4165[1] + tmp_4141[1], tmp_4165[2] + tmp_4141[2]];
    signal tmp_4167[3] <== [tmp_4164[0] + tmp_4166[0], tmp_4164[1] + tmp_4166[1], tmp_4164[2] + tmp_4166[2]];
    signal tmp_4168[3] <== [evals[9][0] * 12413576830284969611, evals[9][1] * 12413576830284969611, evals[9][2] * 12413576830284969611];
    signal tmp_4169[3] <== [tmp_4168[0] + tmp_4141[0], tmp_4168[1] + tmp_4141[1], tmp_4168[2] + tmp_4141[2]];
    signal tmp_4170[3] <== [tmp_4167[0] + tmp_4169[0], tmp_4167[1] + tmp_4169[1], tmp_4167[2] + tmp_4169[2]];
    signal tmp_4171[3] <== [evals[10][0] * 11675438539028694709, evals[10][1] * 11675438539028694709, evals[10][2] * 11675438539028694709];
    signal tmp_4172[3] <== [tmp_4171[0] + tmp_4141[0], tmp_4171[1] + tmp_4141[1], tmp_4171[2] + tmp_4141[2]];
    signal tmp_4173[3] <== [tmp_4170[0] + tmp_4172[0], tmp_4170[1] + tmp_4172[1], tmp_4170[2] + tmp_4172[2]];
    signal tmp_4174[3] <== [evals[11][0] * 17580553691069642926, evals[11][1] * 17580553691069642926, evals[11][2] * 17580553691069642926];
    signal tmp_4175[3] <== [tmp_4174[0] + tmp_4141[0], tmp_4174[1] + tmp_4141[1], tmp_4174[2] + tmp_4141[2]];
    signal tmp_4176[3] <== [tmp_4173[0] + tmp_4175[0], tmp_4173[1] + tmp_4175[1], tmp_4173[2] + tmp_4175[2]];
    signal tmp_4177[3] <== [evals[12][0] * 892707462476851331, evals[12][1] * 892707462476851331, evals[12][2] * 892707462476851331];
    signal tmp_4178[3] <== [tmp_4177[0] + tmp_4141[0], tmp_4177[1] + tmp_4141[1], tmp_4177[2] + tmp_4141[2]];
    signal tmp_4179[3] <== [tmp_4176[0] + tmp_4178[0], tmp_4176[1] + tmp_4178[1], tmp_4176[2] + tmp_4178[2]];
    signal tmp_4180[3] <== [evals[13][0] * 15167485180850043744, evals[13][1] * 15167485180850043744, evals[13][2] * 15167485180850043744];
    signal tmp_4181[3] <== [tmp_4180[0] + tmp_4141[0], tmp_4180[1] + tmp_4141[1], tmp_4180[2] + tmp_4141[2]];
    signal tmp_4182[3] <== [tmp_4179[0] + tmp_4181[0], tmp_4179[1] + tmp_4181[1], tmp_4179[2] + tmp_4181[2]];
    signal tmp_4183[3] <== [tmp_4149[0] + tmp_4182[0], tmp_4149[1] + tmp_4182[1], tmp_4149[2] + tmp_4182[2]];
    signal tmp_4184[3] <== [evals[76][0] - tmp_4183[0], evals[76][1] - tmp_4183[1], evals[76][2] - tmp_4183[2]];
    signal tmp_4185[3] <== CMul()(evals[51], tmp_4184);
    signal tmp_4186[3] <== [tmp_4146[0] + tmp_4185[0], tmp_4146[1] + tmp_4185[1], tmp_4146[2] + tmp_4185[2]];
    signal tmp_4187[3] <== CMul()(challengeQ, tmp_4186);
    signal tmp_4188[3] <== [evals[76][0] + tmp_3457[0], evals[76][1] + tmp_3457[1], evals[76][2] + tmp_3457[2]];
    signal tmp_4189[3] <== CMul()(tmp_3462, tmp_4188);
    signal tmp_4190[3] <== [tmp_4189[0] * 14102670999874605824, tmp_4189[1] * 14102670999874605824, tmp_4189[2] * 14102670999874605824];
    signal tmp_4191[3] <== [tmp_4151[0] * 15585654191999307702, tmp_4151[1] * 15585654191999307702, tmp_4151[2] * 15585654191999307702];
    signal tmp_4192[3] <== [tmp_4191[0] + tmp_4182[0], tmp_4191[1] + tmp_4182[1], tmp_4191[2] + tmp_4182[2]];
    signal tmp_4193[3] <== [tmp_4189[0] + tmp_4192[0], tmp_4189[1] + tmp_4192[1], tmp_4189[2] + tmp_4192[2]];
    signal tmp_4194[3] <== [tmp_4154[0] * 940187017142450255, tmp_4154[1] * 940187017142450255, tmp_4154[2] * 940187017142450255];
    signal tmp_4195[3] <== [tmp_4194[0] + tmp_4182[0], tmp_4194[1] + tmp_4182[1], tmp_4194[2] + tmp_4182[2]];
    signal tmp_4196[3] <== [tmp_4193[0] + tmp_4195[0], tmp_4193[1] + tmp_4195[1], tmp_4193[2] + tmp_4195[2]];
    signal tmp_4197[3] <== [tmp_4157[0] * 8747386241522630711, tmp_4157[1] * 8747386241522630711, tmp_4157[2] * 8747386241522630711];
    signal tmp_4198[3] <== [tmp_4197[0] + tmp_4182[0], tmp_4197[1] + tmp_4182[1], tmp_4197[2] + tmp_4182[2]];
    signal tmp_4199[3] <== [tmp_4196[0] + tmp_4198[0], tmp_4196[1] + tmp_4198[1], tmp_4196[2] + tmp_4198[2]];
    signal tmp_4200[3] <== [tmp_4160[0] * 6750641561540124747, tmp_4160[1] * 6750641561540124747, tmp_4160[2] * 6750641561540124747];
    signal tmp_4201[3] <== [tmp_4200[0] + tmp_4182[0], tmp_4200[1] + tmp_4182[1], tmp_4200[2] + tmp_4182[2]];
    signal tmp_4202[3] <== [tmp_4199[0] + tmp_4201[0], tmp_4199[1] + tmp_4201[1], tmp_4199[2] + tmp_4201[2]];
    signal tmp_4203[3] <== [tmp_4163[0] * 7440998025584530007, tmp_4163[1] * 7440998025584530007, tmp_4163[2] * 7440998025584530007];
    signal tmp_4204[3] <== [tmp_4203[0] + tmp_4182[0], tmp_4203[1] + tmp_4182[1], tmp_4203[2] + tmp_4182[2]];
    signal tmp_4205[3] <== [tmp_4202[0] + tmp_4204[0], tmp_4202[1] + tmp_4204[1], tmp_4202[2] + tmp_4204[2]];
    signal tmp_4206[3] <== [tmp_4166[0] * 6136358134615751536, tmp_4166[1] * 6136358134615751536, tmp_4166[2] * 6136358134615751536];
    signal tmp_4207[3] <== [tmp_4206[0] + tmp_4182[0], tmp_4206[1] + tmp_4182[1], tmp_4206[2] + tmp_4182[2]];
    signal tmp_4208[3] <== [tmp_4205[0] + tmp_4207[0], tmp_4205[1] + tmp_4207[1], tmp_4205[2] + tmp_4207[2]];
    signal tmp_4209[3] <== [tmp_4169[0] * 12413576830284969611, tmp_4169[1] * 12413576830284969611, tmp_4169[2] * 12413576830284969611];
    signal tmp_4210[3] <== [tmp_4209[0] + tmp_4182[0], tmp_4209[1] + tmp_4182[1], tmp_4209[2] + tmp_4182[2]];
    signal tmp_4211[3] <== [tmp_4208[0] + tmp_4210[0], tmp_4208[1] + tmp_4210[1], tmp_4208[2] + tmp_4210[2]];
    signal tmp_4212[3] <== [tmp_4172[0] * 11675438539028694709, tmp_4172[1] * 11675438539028694709, tmp_4172[2] * 11675438539028694709];
    signal tmp_4213[3] <== [tmp_4212[0] + tmp_4182[0], tmp_4212[1] + tmp_4182[1], tmp_4212[2] + tmp_4182[2]];
    signal tmp_4214[3] <== [tmp_4211[0] + tmp_4213[0], tmp_4211[1] + tmp_4213[1], tmp_4211[2] + tmp_4213[2]];
    signal tmp_4215[3] <== [tmp_4175[0] * 17580553691069642926, tmp_4175[1] * 17580553691069642926, tmp_4175[2] * 17580553691069642926];
    signal tmp_4216[3] <== [tmp_4215[0] + tmp_4182[0], tmp_4215[1] + tmp_4182[1], tmp_4215[2] + tmp_4182[2]];
    signal tmp_4217[3] <== [tmp_4214[0] + tmp_4216[0], tmp_4214[1] + tmp_4216[1], tmp_4214[2] + tmp_4216[2]];
    signal tmp_4218[3] <== [tmp_4178[0] * 892707462476851331, tmp_4178[1] * 892707462476851331, tmp_4178[2] * 892707462476851331];
    signal tmp_4219[3] <== [tmp_4218[0] + tmp_4182[0], tmp_4218[1] + tmp_4182[1], tmp_4218[2] + tmp_4182[2]];
    signal tmp_4220[3] <== [tmp_4217[0] + tmp_4219[0], tmp_4217[1] + tmp_4219[1], tmp_4217[2] + tmp_4219[2]];
    signal tmp_4221[3] <== [tmp_4181[0] * 15167485180850043744, tmp_4181[1] * 15167485180850043744, tmp_4181[2] * 15167485180850043744];
    signal tmp_4222[3] <== [tmp_4221[0] + tmp_4182[0], tmp_4221[1] + tmp_4182[1], tmp_4221[2] + tmp_4182[2]];
    signal tmp_4223[3] <== [tmp_4220[0] + tmp_4222[0], tmp_4220[1] + tmp_4222[1], tmp_4220[2] + tmp_4222[2]];
    signal tmp_4224[3] <== [tmp_4190[0] + tmp_4223[0], tmp_4190[1] + tmp_4223[1], tmp_4190[2] + tmp_4223[2]];
    signal tmp_4225[3] <== [evals[77][0] - tmp_4224[0], evals[77][1] - tmp_4224[1], evals[77][2] - tmp_4224[2]];
    signal tmp_4226[3] <== CMul()(evals[51], tmp_4225);
    signal tmp_4227[3] <== [tmp_4187[0] + tmp_4226[0], tmp_4187[1] + tmp_4226[1], tmp_4187[2] + tmp_4226[2]];
    signal tmp_4228[3] <== CMul()(challengeQ, tmp_4227);
    signal tmp_4229[3] <== [evals[77][0] + tmp_3473[0], evals[77][1] + tmp_3473[1], evals[77][2] + tmp_3473[2]];
    signal tmp_4230[3] <== CMul()(tmp_3478, tmp_4229);
    signal tmp_4231[3] <== [tmp_4230[0] * 14102670999874605824, tmp_4230[1] * 14102670999874605824, tmp_4230[2] * 14102670999874605824];
    signal tmp_4232[3] <== [tmp_4192[0] * 15585654191999307702, tmp_4192[1] * 15585654191999307702, tmp_4192[2] * 15585654191999307702];
    signal tmp_4233[3] <== [tmp_4232[0] + tmp_4223[0], tmp_4232[1] + tmp_4223[1], tmp_4232[2] + tmp_4223[2]];
    signal tmp_4234[3] <== [tmp_4230[0] + tmp_4233[0], tmp_4230[1] + tmp_4233[1], tmp_4230[2] + tmp_4233[2]];
    signal tmp_4235[3] <== [tmp_4195[0] * 940187017142450255, tmp_4195[1] * 940187017142450255, tmp_4195[2] * 940187017142450255];
    signal tmp_4236[3] <== [tmp_4235[0] + tmp_4223[0], tmp_4235[1] + tmp_4223[1], tmp_4235[2] + tmp_4223[2]];
    signal tmp_4237[3] <== [tmp_4234[0] + tmp_4236[0], tmp_4234[1] + tmp_4236[1], tmp_4234[2] + tmp_4236[2]];
    signal tmp_4238[3] <== [tmp_4198[0] * 8747386241522630711, tmp_4198[1] * 8747386241522630711, tmp_4198[2] * 8747386241522630711];
    signal tmp_4239[3] <== [tmp_4238[0] + tmp_4223[0], tmp_4238[1] + tmp_4223[1], tmp_4238[2] + tmp_4223[2]];
    signal tmp_4240[3] <== [tmp_4237[0] + tmp_4239[0], tmp_4237[1] + tmp_4239[1], tmp_4237[2] + tmp_4239[2]];
    signal tmp_4241[3] <== [tmp_4201[0] * 6750641561540124747, tmp_4201[1] * 6750641561540124747, tmp_4201[2] * 6750641561540124747];
    signal tmp_4242[3] <== [tmp_4241[0] + tmp_4223[0], tmp_4241[1] + tmp_4223[1], tmp_4241[2] + tmp_4223[2]];
    signal tmp_4243[3] <== [tmp_4240[0] + tmp_4242[0], tmp_4240[1] + tmp_4242[1], tmp_4240[2] + tmp_4242[2]];
    signal tmp_4244[3] <== [tmp_4204[0] * 7440998025584530007, tmp_4204[1] * 7440998025584530007, tmp_4204[2] * 7440998025584530007];
    signal tmp_4245[3] <== [tmp_4244[0] + tmp_4223[0], tmp_4244[1] + tmp_4223[1], tmp_4244[2] + tmp_4223[2]];
    signal tmp_4246[3] <== [tmp_4243[0] + tmp_4245[0], tmp_4243[1] + tmp_4245[1], tmp_4243[2] + tmp_4245[2]];
    signal tmp_4247[3] <== [tmp_4207[0] * 6136358134615751536, tmp_4207[1] * 6136358134615751536, tmp_4207[2] * 6136358134615751536];
    signal tmp_4248[3] <== [tmp_4247[0] + tmp_4223[0], tmp_4247[1] + tmp_4223[1], tmp_4247[2] + tmp_4223[2]];
    signal tmp_4249[3] <== [tmp_4246[0] + tmp_4248[0], tmp_4246[1] + tmp_4248[1], tmp_4246[2] + tmp_4248[2]];
    signal tmp_4250[3] <== [tmp_4210[0] * 12413576830284969611, tmp_4210[1] * 12413576830284969611, tmp_4210[2] * 12413576830284969611];
    signal tmp_4251[3] <== [tmp_4250[0] + tmp_4223[0], tmp_4250[1] + tmp_4223[1], tmp_4250[2] + tmp_4223[2]];
    signal tmp_4252[3] <== [tmp_4249[0] + tmp_4251[0], tmp_4249[1] + tmp_4251[1], tmp_4249[2] + tmp_4251[2]];
    signal tmp_4253[3] <== [tmp_4213[0] * 11675438539028694709, tmp_4213[1] * 11675438539028694709, tmp_4213[2] * 11675438539028694709];
    signal tmp_4254[3] <== [tmp_4253[0] + tmp_4223[0], tmp_4253[1] + tmp_4223[1], tmp_4253[2] + tmp_4223[2]];
    signal tmp_4255[3] <== [tmp_4252[0] + tmp_4254[0], tmp_4252[1] + tmp_4254[1], tmp_4252[2] + tmp_4254[2]];
    signal tmp_4256[3] <== [tmp_4216[0] * 17580553691069642926, tmp_4216[1] * 17580553691069642926, tmp_4216[2] * 17580553691069642926];
    signal tmp_4257[3] <== [tmp_4256[0] + tmp_4223[0], tmp_4256[1] + tmp_4223[1], tmp_4256[2] + tmp_4223[2]];
    signal tmp_4258[3] <== [tmp_4255[0] + tmp_4257[0], tmp_4255[1] + tmp_4257[1], tmp_4255[2] + tmp_4257[2]];
    signal tmp_4259[3] <== [tmp_4219[0] * 892707462476851331, tmp_4219[1] * 892707462476851331, tmp_4219[2] * 892707462476851331];
    signal tmp_4260[3] <== [tmp_4259[0] + tmp_4223[0], tmp_4259[1] + tmp_4223[1], tmp_4259[2] + tmp_4223[2]];
    signal tmp_4261[3] <== [tmp_4258[0] + tmp_4260[0], tmp_4258[1] + tmp_4260[1], tmp_4258[2] + tmp_4260[2]];
    signal tmp_4262[3] <== [tmp_4222[0] * 15167485180850043744, tmp_4222[1] * 15167485180850043744, tmp_4222[2] * 15167485180850043744];
    signal tmp_4263[3] <== [tmp_4262[0] + tmp_4223[0], tmp_4262[1] + tmp_4223[1], tmp_4262[2] + tmp_4223[2]];
    signal tmp_4264[3] <== [tmp_4261[0] + tmp_4263[0], tmp_4261[1] + tmp_4263[1], tmp_4261[2] + tmp_4263[2]];
    signal tmp_4265[3] <== [tmp_4231[0] + tmp_4264[0], tmp_4231[1] + tmp_4264[1], tmp_4231[2] + tmp_4264[2]];
    signal tmp_4266[3] <== [evals[78][0] - tmp_4265[0], evals[78][1] - tmp_4265[1], evals[78][2] - tmp_4265[2]];
    signal tmp_4267[3] <== CMul()(evals[51], tmp_4266);
    signal tmp_4268[3] <== [tmp_4228[0] + tmp_4267[0], tmp_4228[1] + tmp_4267[1], tmp_4228[2] + tmp_4267[2]];
    signal tmp_4269[3] <== CMul()(challengeQ, tmp_4268);
    signal tmp_4270[3] <== [evals[78][0] + tmp_3495[0], evals[78][1] + tmp_3495[1], evals[78][2] + tmp_3495[2]];
    signal tmp_4271[3] <== CMul()(tmp_3500, tmp_4270);
    signal tmp_4272[3] <== [tmp_4271[0] * 14102670999874605824, tmp_4271[1] * 14102670999874605824, tmp_4271[2] * 14102670999874605824];
    signal tmp_4273[3] <== [tmp_4233[0] * 15585654191999307702, tmp_4233[1] * 15585654191999307702, tmp_4233[2] * 15585654191999307702];
    signal tmp_4274[3] <== [tmp_4273[0] + tmp_4264[0], tmp_4273[1] + tmp_4264[1], tmp_4273[2] + tmp_4264[2]];
    signal tmp_4275[3] <== [tmp_4271[0] + tmp_4274[0], tmp_4271[1] + tmp_4274[1], tmp_4271[2] + tmp_4274[2]];
    signal tmp_4276[3] <== [tmp_4236[0] * 940187017142450255, tmp_4236[1] * 940187017142450255, tmp_4236[2] * 940187017142450255];
    signal tmp_4277[3] <== [tmp_4276[0] + tmp_4264[0], tmp_4276[1] + tmp_4264[1], tmp_4276[2] + tmp_4264[2]];
    signal tmp_4278[3] <== [tmp_4275[0] + tmp_4277[0], tmp_4275[1] + tmp_4277[1], tmp_4275[2] + tmp_4277[2]];
    signal tmp_4279[3] <== [tmp_4239[0] * 8747386241522630711, tmp_4239[1] * 8747386241522630711, tmp_4239[2] * 8747386241522630711];
    signal tmp_4280[3] <== [tmp_4279[0] + tmp_4264[0], tmp_4279[1] + tmp_4264[1], tmp_4279[2] + tmp_4264[2]];
    signal tmp_4281[3] <== [tmp_4278[0] + tmp_4280[0], tmp_4278[1] + tmp_4280[1], tmp_4278[2] + tmp_4280[2]];
    signal tmp_4282[3] <== [tmp_4242[0] * 6750641561540124747, tmp_4242[1] * 6750641561540124747, tmp_4242[2] * 6750641561540124747];
    signal tmp_4283[3] <== [tmp_4282[0] + tmp_4264[0], tmp_4282[1] + tmp_4264[1], tmp_4282[2] + tmp_4264[2]];
    signal tmp_4284[3] <== [tmp_4281[0] + tmp_4283[0], tmp_4281[1] + tmp_4283[1], tmp_4281[2] + tmp_4283[2]];
    signal tmp_4285[3] <== [tmp_4245[0] * 7440998025584530007, tmp_4245[1] * 7440998025584530007, tmp_4245[2] * 7440998025584530007];
    signal tmp_4286[3] <== [tmp_4285[0] + tmp_4264[0], tmp_4285[1] + tmp_4264[1], tmp_4285[2] + tmp_4264[2]];
    signal tmp_4287[3] <== [tmp_4284[0] + tmp_4286[0], tmp_4284[1] + tmp_4286[1], tmp_4284[2] + tmp_4286[2]];
    signal tmp_4288[3] <== [tmp_4248[0] * 6136358134615751536, tmp_4248[1] * 6136358134615751536, tmp_4248[2] * 6136358134615751536];
    signal tmp_4289[3] <== [tmp_4288[0] + tmp_4264[0], tmp_4288[1] + tmp_4264[1], tmp_4288[2] + tmp_4264[2]];
    signal tmp_4290[3] <== [tmp_4287[0] + tmp_4289[0], tmp_4287[1] + tmp_4289[1], tmp_4287[2] + tmp_4289[2]];
    signal tmp_4291[3] <== [tmp_4251[0] * 12413576830284969611, tmp_4251[1] * 12413576830284969611, tmp_4251[2] * 12413576830284969611];
    signal tmp_4292[3] <== [tmp_4291[0] + tmp_4264[0], tmp_4291[1] + tmp_4264[1], tmp_4291[2] + tmp_4264[2]];
    signal tmp_4293[3] <== [tmp_4290[0] + tmp_4292[0], tmp_4290[1] + tmp_4292[1], tmp_4290[2] + tmp_4292[2]];
    signal tmp_4294[3] <== [tmp_4254[0] * 11675438539028694709, tmp_4254[1] * 11675438539028694709, tmp_4254[2] * 11675438539028694709];
    signal tmp_4295[3] <== [tmp_4294[0] + tmp_4264[0], tmp_4294[1] + tmp_4264[1], tmp_4294[2] + tmp_4264[2]];
    signal tmp_4296[3] <== [tmp_4293[0] + tmp_4295[0], tmp_4293[1] + tmp_4295[1], tmp_4293[2] + tmp_4295[2]];
    signal tmp_4297[3] <== [tmp_4257[0] * 17580553691069642926, tmp_4257[1] * 17580553691069642926, tmp_4257[2] * 17580553691069642926];
    signal tmp_4298[3] <== [tmp_4297[0] + tmp_4264[0], tmp_4297[1] + tmp_4264[1], tmp_4297[2] + tmp_4264[2]];
    signal tmp_4299[3] <== [tmp_4296[0] + tmp_4298[0], tmp_4296[1] + tmp_4298[1], tmp_4296[2] + tmp_4298[2]];
    signal tmp_4300[3] <== [tmp_4260[0] * 892707462476851331, tmp_4260[1] * 892707462476851331, tmp_4260[2] * 892707462476851331];
    signal tmp_4301[3] <== [tmp_4300[0] + tmp_4264[0], tmp_4300[1] + tmp_4264[1], tmp_4300[2] + tmp_4264[2]];
    signal tmp_4302[3] <== [tmp_4299[0] + tmp_4301[0], tmp_4299[1] + tmp_4301[1], tmp_4299[2] + tmp_4301[2]];
    signal tmp_4303[3] <== [tmp_4263[0] * 15167485180850043744, tmp_4263[1] * 15167485180850043744, tmp_4263[2] * 15167485180850043744];
    signal tmp_4304[3] <== [tmp_4303[0] + tmp_4264[0], tmp_4303[1] + tmp_4264[1], tmp_4303[2] + tmp_4264[2]];
    signal tmp_4305[3] <== [tmp_4302[0] + tmp_4304[0], tmp_4302[1] + tmp_4304[1], tmp_4302[2] + tmp_4304[2]];
    signal tmp_4306[3] <== [tmp_4272[0] + tmp_4305[0], tmp_4272[1] + tmp_4305[1], tmp_4272[2] + tmp_4305[2]];
    signal tmp_4307[3] <== [evals[79][0] - tmp_4306[0], evals[79][1] - tmp_4306[1], evals[79][2] - tmp_4306[2]];
    signal tmp_4308[3] <== CMul()(evals[51], tmp_4307);
    signal tmp_4309[3] <== [tmp_4269[0] + tmp_4308[0], tmp_4269[1] + tmp_4308[1], tmp_4269[2] + tmp_4308[2]];
    signal tmp_4310[3] <== CMul()(challengeQ, tmp_4309);
    signal tmp_4311[3] <== [evals[79][0] + tmp_3440[0], evals[79][1] + tmp_3440[1], evals[79][2] + tmp_3440[2]];
    signal tmp_4312[3] <== CMul()(tmp_3445, tmp_4311);
    signal tmp_4313[3] <== [tmp_4312[0] * 14102670999874605824, tmp_4312[1] * 14102670999874605824, tmp_4312[2] * 14102670999874605824];
    signal tmp_4314[3] <== [tmp_4274[0] * 15585654191999307702, tmp_4274[1] * 15585654191999307702, tmp_4274[2] * 15585654191999307702];
    signal tmp_4315[3] <== [tmp_4314[0] + tmp_4305[0], tmp_4314[1] + tmp_4305[1], tmp_4314[2] + tmp_4305[2]];
    signal tmp_4316[3] <== [tmp_4312[0] + tmp_4315[0], tmp_4312[1] + tmp_4315[1], tmp_4312[2] + tmp_4315[2]];
    signal tmp_4317[3] <== [tmp_4277[0] * 940187017142450255, tmp_4277[1] * 940187017142450255, tmp_4277[2] * 940187017142450255];
    signal tmp_4318[3] <== [tmp_4317[0] + tmp_4305[0], tmp_4317[1] + tmp_4305[1], tmp_4317[2] + tmp_4305[2]];
    signal tmp_4319[3] <== [tmp_4316[0] + tmp_4318[0], tmp_4316[1] + tmp_4318[1], tmp_4316[2] + tmp_4318[2]];
    signal tmp_4320[3] <== [tmp_4280[0] * 8747386241522630711, tmp_4280[1] * 8747386241522630711, tmp_4280[2] * 8747386241522630711];
    signal tmp_4321[3] <== [tmp_4320[0] + tmp_4305[0], tmp_4320[1] + tmp_4305[1], tmp_4320[2] + tmp_4305[2]];
    signal tmp_4322[3] <== [tmp_4319[0] + tmp_4321[0], tmp_4319[1] + tmp_4321[1], tmp_4319[2] + tmp_4321[2]];
    signal tmp_4323[3] <== [tmp_4283[0] * 6750641561540124747, tmp_4283[1] * 6750641561540124747, tmp_4283[2] * 6750641561540124747];
    signal tmp_4324[3] <== [tmp_4323[0] + tmp_4305[0], tmp_4323[1] + tmp_4305[1], tmp_4323[2] + tmp_4305[2]];
    signal tmp_4325[3] <== [tmp_4322[0] + tmp_4324[0], tmp_4322[1] + tmp_4324[1], tmp_4322[2] + tmp_4324[2]];
    signal tmp_4326[3] <== [tmp_4286[0] * 7440998025584530007, tmp_4286[1] * 7440998025584530007, tmp_4286[2] * 7440998025584530007];
    signal tmp_4327[3] <== [tmp_4326[0] + tmp_4305[0], tmp_4326[1] + tmp_4305[1], tmp_4326[2] + tmp_4305[2]];
    signal tmp_4328[3] <== [tmp_4325[0] + tmp_4327[0], tmp_4325[1] + tmp_4327[1], tmp_4325[2] + tmp_4327[2]];
    signal tmp_4329[3] <== [tmp_4289[0] * 6136358134615751536, tmp_4289[1] * 6136358134615751536, tmp_4289[2] * 6136358134615751536];
    signal tmp_4330[3] <== [tmp_4329[0] + tmp_4305[0], tmp_4329[1] + tmp_4305[1], tmp_4329[2] + tmp_4305[2]];
    signal tmp_4331[3] <== [tmp_4328[0] + tmp_4330[0], tmp_4328[1] + tmp_4330[1], tmp_4328[2] + tmp_4330[2]];
    signal tmp_4332[3] <== [tmp_4292[0] * 12413576830284969611, tmp_4292[1] * 12413576830284969611, tmp_4292[2] * 12413576830284969611];
    signal tmp_4333[3] <== [tmp_4332[0] + tmp_4305[0], tmp_4332[1] + tmp_4305[1], tmp_4332[2] + tmp_4305[2]];
    signal tmp_4334[3] <== [tmp_4331[0] + tmp_4333[0], tmp_4331[1] + tmp_4333[1], tmp_4331[2] + tmp_4333[2]];
    signal tmp_4335[3] <== [tmp_4295[0] * 11675438539028694709, tmp_4295[1] * 11675438539028694709, tmp_4295[2] * 11675438539028694709];
    signal tmp_4336[3] <== [tmp_4335[0] + tmp_4305[0], tmp_4335[1] + tmp_4305[1], tmp_4335[2] + tmp_4305[2]];
    signal tmp_4337[3] <== [tmp_4334[0] + tmp_4336[0], tmp_4334[1] + tmp_4336[1], tmp_4334[2] + tmp_4336[2]];
    signal tmp_4338[3] <== [tmp_4298[0] * 17580553691069642926, tmp_4298[1] * 17580553691069642926, tmp_4298[2] * 17580553691069642926];
    signal tmp_4339[3] <== [tmp_4338[0] + tmp_4305[0], tmp_4338[1] + tmp_4305[1], tmp_4338[2] + tmp_4305[2]];
    signal tmp_4340[3] <== [tmp_4337[0] + tmp_4339[0], tmp_4337[1] + tmp_4339[1], tmp_4337[2] + tmp_4339[2]];
    signal tmp_4341[3] <== [tmp_4301[0] * 892707462476851331, tmp_4301[1] * 892707462476851331, tmp_4301[2] * 892707462476851331];
    signal tmp_4342[3] <== [tmp_4341[0] + tmp_4305[0], tmp_4341[1] + tmp_4305[1], tmp_4341[2] + tmp_4305[2]];
    signal tmp_4343[3] <== [tmp_4340[0] + tmp_4342[0], tmp_4340[1] + tmp_4342[1], tmp_4340[2] + tmp_4342[2]];
    signal tmp_4344[3] <== [tmp_4304[0] * 15167485180850043744, tmp_4304[1] * 15167485180850043744, tmp_4304[2] * 15167485180850043744];
    signal tmp_4345[3] <== [tmp_4344[0] + tmp_4305[0], tmp_4344[1] + tmp_4305[1], tmp_4344[2] + tmp_4305[2]];
    signal tmp_4346[3] <== [tmp_4343[0] + tmp_4345[0], tmp_4343[1] + tmp_4345[1], tmp_4343[2] + tmp_4345[2]];
    signal tmp_4347[3] <== [tmp_4313[0] + tmp_4346[0], tmp_4313[1] + tmp_4346[1], tmp_4313[2] + tmp_4346[2]];
    signal tmp_4348[3] <== [evals[80][0] - tmp_4347[0], evals[80][1] - tmp_4347[1], evals[80][2] - tmp_4347[2]];
    signal tmp_4349[3] <== CMul()(evals[51], tmp_4348);
    signal tmp_4350[3] <== [tmp_4310[0] + tmp_4349[0], tmp_4310[1] + tmp_4349[1], tmp_4310[2] + tmp_4349[2]];
    signal tmp_4351[3] <== CMul()(challengeQ, tmp_4350);
    signal tmp_4352[3] <== [evals[80][0] + tmp_3535[0], evals[80][1] + tmp_3535[1], evals[80][2] + tmp_3535[2]];
    signal tmp_4353[3] <== CMul()(tmp_3540, tmp_4352);
    signal tmp_4354[3] <== [tmp_4353[0] * 14102670999874605824, tmp_4353[1] * 14102670999874605824, tmp_4353[2] * 14102670999874605824];
    signal tmp_4355[3] <== [tmp_4315[0] * 15585654191999307702, tmp_4315[1] * 15585654191999307702, tmp_4315[2] * 15585654191999307702];
    signal tmp_4356[3] <== [tmp_4355[0] + tmp_4346[0], tmp_4355[1] + tmp_4346[1], tmp_4355[2] + tmp_4346[2]];
    signal tmp_4357[3] <== [tmp_4353[0] + tmp_4356[0], tmp_4353[1] + tmp_4356[1], tmp_4353[2] + tmp_4356[2]];
    signal tmp_4358[3] <== [tmp_4318[0] * 940187017142450255, tmp_4318[1] * 940187017142450255, tmp_4318[2] * 940187017142450255];
    signal tmp_4359[3] <== [tmp_4358[0] + tmp_4346[0], tmp_4358[1] + tmp_4346[1], tmp_4358[2] + tmp_4346[2]];
    signal tmp_4360[3] <== [tmp_4357[0] + tmp_4359[0], tmp_4357[1] + tmp_4359[1], tmp_4357[2] + tmp_4359[2]];
    signal tmp_4361[3] <== [tmp_4321[0] * 8747386241522630711, tmp_4321[1] * 8747386241522630711, tmp_4321[2] * 8747386241522630711];
    signal tmp_4362[3] <== [tmp_4361[0] + tmp_4346[0], tmp_4361[1] + tmp_4346[1], tmp_4361[2] + tmp_4346[2]];
    signal tmp_4363[3] <== [tmp_4360[0] + tmp_4362[0], tmp_4360[1] + tmp_4362[1], tmp_4360[2] + tmp_4362[2]];
    signal tmp_4364[3] <== [tmp_4324[0] * 6750641561540124747, tmp_4324[1] * 6750641561540124747, tmp_4324[2] * 6750641561540124747];
    signal tmp_4365[3] <== [tmp_4364[0] + tmp_4346[0], tmp_4364[1] + tmp_4346[1], tmp_4364[2] + tmp_4346[2]];
    signal tmp_4366[3] <== [tmp_4363[0] + tmp_4365[0], tmp_4363[1] + tmp_4365[1], tmp_4363[2] + tmp_4365[2]];
    signal tmp_4367[3] <== [tmp_4327[0] * 7440998025584530007, tmp_4327[1] * 7440998025584530007, tmp_4327[2] * 7440998025584530007];
    signal tmp_4368[3] <== [tmp_4367[0] + tmp_4346[0], tmp_4367[1] + tmp_4346[1], tmp_4367[2] + tmp_4346[2]];
    signal tmp_4369[3] <== [tmp_4366[0] + tmp_4368[0], tmp_4366[1] + tmp_4368[1], tmp_4366[2] + tmp_4368[2]];
    signal tmp_4370[3] <== [tmp_4330[0] * 6136358134615751536, tmp_4330[1] * 6136358134615751536, tmp_4330[2] * 6136358134615751536];
    signal tmp_4371[3] <== [tmp_4370[0] + tmp_4346[0], tmp_4370[1] + tmp_4346[1], tmp_4370[2] + tmp_4346[2]];
    signal tmp_4372[3] <== [tmp_4369[0] + tmp_4371[0], tmp_4369[1] + tmp_4371[1], tmp_4369[2] + tmp_4371[2]];
    signal tmp_4373[3] <== [tmp_4333[0] * 12413576830284969611, tmp_4333[1] * 12413576830284969611, tmp_4333[2] * 12413576830284969611];
    signal tmp_4374[3] <== [tmp_4373[0] + tmp_4346[0], tmp_4373[1] + tmp_4346[1], tmp_4373[2] + tmp_4346[2]];
    signal tmp_4375[3] <== [tmp_4372[0] + tmp_4374[0], tmp_4372[1] + tmp_4374[1], tmp_4372[2] + tmp_4374[2]];
    signal tmp_4376[3] <== [tmp_4336[0] * 11675438539028694709, tmp_4336[1] * 11675438539028694709, tmp_4336[2] * 11675438539028694709];
    signal tmp_4377[3] <== [tmp_4376[0] + tmp_4346[0], tmp_4376[1] + tmp_4346[1], tmp_4376[2] + tmp_4346[2]];
    signal tmp_4378[3] <== [tmp_4375[0] + tmp_4377[0], tmp_4375[1] + tmp_4377[1], tmp_4375[2] + tmp_4377[2]];
    signal tmp_4379[3] <== [tmp_4339[0] * 17580553691069642926, tmp_4339[1] * 17580553691069642926, tmp_4339[2] * 17580553691069642926];
    signal tmp_4380[3] <== [tmp_4379[0] + tmp_4346[0], tmp_4379[1] + tmp_4346[1], tmp_4379[2] + tmp_4346[2]];
    signal tmp_4381[3] <== [tmp_4378[0] + tmp_4380[0], tmp_4378[1] + tmp_4380[1], tmp_4378[2] + tmp_4380[2]];
    signal tmp_4382[3] <== [tmp_4342[0] * 892707462476851331, tmp_4342[1] * 892707462476851331, tmp_4342[2] * 892707462476851331];
    signal tmp_4383[3] <== [tmp_4382[0] + tmp_4346[0], tmp_4382[1] + tmp_4346[1], tmp_4382[2] + tmp_4346[2]];
    signal tmp_4384[3] <== [tmp_4381[0] + tmp_4383[0], tmp_4381[1] + tmp_4383[1], tmp_4381[2] + tmp_4383[2]];
    signal tmp_4385[3] <== [tmp_4345[0] * 15167485180850043744, tmp_4345[1] * 15167485180850043744, tmp_4345[2] * 15167485180850043744];
    signal tmp_4386[3] <== [tmp_4385[0] + tmp_4346[0], tmp_4385[1] + tmp_4346[1], tmp_4385[2] + tmp_4346[2]];
    signal tmp_4387[3] <== [tmp_4384[0] + tmp_4386[0], tmp_4384[1] + tmp_4386[1], tmp_4384[2] + tmp_4386[2]];
    signal tmp_4388[3] <== [tmp_4354[0] + tmp_4387[0], tmp_4354[1] + tmp_4387[1], tmp_4354[2] + tmp_4387[2]];
    signal tmp_4389[3] <== [evals[81][0] - tmp_4388[0], evals[81][1] - tmp_4388[1], evals[81][2] - tmp_4388[2]];
    signal tmp_4390[3] <== CMul()(evals[51], tmp_4389);
    signal tmp_4391[3] <== [tmp_4351[0] + tmp_4390[0], tmp_4351[1] + tmp_4390[1], tmp_4351[2] + tmp_4390[2]];
    signal tmp_4392[3] <== CMul()(challengeQ, tmp_4391);
    signal tmp_4393[3] <== [evals[81][0] + tmp_3551[0], evals[81][1] + tmp_3551[1], evals[81][2] + tmp_3551[2]];
    signal tmp_4394[3] <== CMul()(tmp_3556, tmp_4393);
    signal tmp_4395[3] <== [tmp_4394[0] * 14102670999874605824, tmp_4394[1] * 14102670999874605824, tmp_4394[2] * 14102670999874605824];
    signal tmp_4396[3] <== [tmp_4356[0] * 15585654191999307702, tmp_4356[1] * 15585654191999307702, tmp_4356[2] * 15585654191999307702];
    signal tmp_4397[3] <== [tmp_4396[0] + tmp_4387[0], tmp_4396[1] + tmp_4387[1], tmp_4396[2] + tmp_4387[2]];
    signal tmp_4398[3] <== [tmp_4394[0] + tmp_4397[0], tmp_4394[1] + tmp_4397[1], tmp_4394[2] + tmp_4397[2]];
    signal tmp_4399[3] <== [tmp_4359[0] * 940187017142450255, tmp_4359[1] * 940187017142450255, tmp_4359[2] * 940187017142450255];
    signal tmp_4400[3] <== [tmp_4399[0] + tmp_4387[0], tmp_4399[1] + tmp_4387[1], tmp_4399[2] + tmp_4387[2]];
    signal tmp_4401[3] <== [tmp_4398[0] + tmp_4400[0], tmp_4398[1] + tmp_4400[1], tmp_4398[2] + tmp_4400[2]];
    signal tmp_4402[3] <== [tmp_4362[0] * 8747386241522630711, tmp_4362[1] * 8747386241522630711, tmp_4362[2] * 8747386241522630711];
    signal tmp_4403[3] <== [tmp_4402[0] + tmp_4387[0], tmp_4402[1] + tmp_4387[1], tmp_4402[2] + tmp_4387[2]];
    signal tmp_4404[3] <== [tmp_4401[0] + tmp_4403[0], tmp_4401[1] + tmp_4403[1], tmp_4401[2] + tmp_4403[2]];
    signal tmp_4405[3] <== [tmp_4365[0] * 6750641561540124747, tmp_4365[1] * 6750641561540124747, tmp_4365[2] * 6750641561540124747];
    signal tmp_4406[3] <== [tmp_4405[0] + tmp_4387[0], tmp_4405[1] + tmp_4387[1], tmp_4405[2] + tmp_4387[2]];
    signal tmp_4407[3] <== [tmp_4404[0] + tmp_4406[0], tmp_4404[1] + tmp_4406[1], tmp_4404[2] + tmp_4406[2]];
    signal tmp_4408[3] <== [tmp_4368[0] * 7440998025584530007, tmp_4368[1] * 7440998025584530007, tmp_4368[2] * 7440998025584530007];
    signal tmp_4409[3] <== [tmp_4408[0] + tmp_4387[0], tmp_4408[1] + tmp_4387[1], tmp_4408[2] + tmp_4387[2]];
    signal tmp_4410[3] <== [tmp_4407[0] + tmp_4409[0], tmp_4407[1] + tmp_4409[1], tmp_4407[2] + tmp_4409[2]];
    signal tmp_4411[3] <== [tmp_4371[0] * 6136358134615751536, tmp_4371[1] * 6136358134615751536, tmp_4371[2] * 6136358134615751536];
    signal tmp_4412[3] <== [tmp_4411[0] + tmp_4387[0], tmp_4411[1] + tmp_4387[1], tmp_4411[2] + tmp_4387[2]];
    signal tmp_4413[3] <== [tmp_4410[0] + tmp_4412[0], tmp_4410[1] + tmp_4412[1], tmp_4410[2] + tmp_4412[2]];
    signal tmp_4414[3] <== [tmp_4374[0] * 12413576830284969611, tmp_4374[1] * 12413576830284969611, tmp_4374[2] * 12413576830284969611];
    signal tmp_4415[3] <== [tmp_4414[0] + tmp_4387[0], tmp_4414[1] + tmp_4387[1], tmp_4414[2] + tmp_4387[2]];
    signal tmp_4416[3] <== [tmp_4413[0] + tmp_4415[0], tmp_4413[1] + tmp_4415[1], tmp_4413[2] + tmp_4415[2]];
    signal tmp_4417[3] <== [tmp_4377[0] * 11675438539028694709, tmp_4377[1] * 11675438539028694709, tmp_4377[2] * 11675438539028694709];
    signal tmp_4418[3] <== [tmp_4417[0] + tmp_4387[0], tmp_4417[1] + tmp_4387[1], tmp_4417[2] + tmp_4387[2]];
    signal tmp_4419[3] <== [tmp_4416[0] + tmp_4418[0], tmp_4416[1] + tmp_4418[1], tmp_4416[2] + tmp_4418[2]];
    signal tmp_4420[3] <== [tmp_4380[0] * 17580553691069642926, tmp_4380[1] * 17580553691069642926, tmp_4380[2] * 17580553691069642926];
    signal tmp_4421[3] <== [tmp_4420[0] + tmp_4387[0], tmp_4420[1] + tmp_4387[1], tmp_4420[2] + tmp_4387[2]];
    signal tmp_4422[3] <== [tmp_4419[0] + tmp_4421[0], tmp_4419[1] + tmp_4421[1], tmp_4419[2] + tmp_4421[2]];
    signal tmp_4423[3] <== [tmp_4383[0] * 892707462476851331, tmp_4383[1] * 892707462476851331, tmp_4383[2] * 892707462476851331];
    signal tmp_4424[3] <== [tmp_4423[0] + tmp_4387[0], tmp_4423[1] + tmp_4387[1], tmp_4423[2] + tmp_4387[2]];
    signal tmp_4425[3] <== [tmp_4422[0] + tmp_4424[0], tmp_4422[1] + tmp_4424[1], tmp_4422[2] + tmp_4424[2]];
    signal tmp_4426[3] <== [tmp_4386[0] * 15167485180850043744, tmp_4386[1] * 15167485180850043744, tmp_4386[2] * 15167485180850043744];
    signal tmp_4427[3] <== [tmp_4426[0] + tmp_4387[0], tmp_4426[1] + tmp_4387[1], tmp_4426[2] + tmp_4387[2]];
    signal tmp_4428[3] <== [tmp_4425[0] + tmp_4427[0], tmp_4425[1] + tmp_4427[1], tmp_4425[2] + tmp_4427[2]];
    signal tmp_4429[3] <== [tmp_4395[0] + tmp_4428[0], tmp_4395[1] + tmp_4428[1], tmp_4395[2] + tmp_4428[2]];
    signal tmp_4430[3] <== [evals[82][0] - tmp_4429[0], evals[82][1] - tmp_4429[1], evals[82][2] - tmp_4429[2]];
    signal tmp_4431[3] <== CMul()(evals[51], tmp_4430);
    signal tmp_4432[3] <== [tmp_4392[0] + tmp_4431[0], tmp_4392[1] + tmp_4431[1], tmp_4392[2] + tmp_4431[2]];
    signal tmp_4433[3] <== CMul()(challengeQ, tmp_4432);
    signal tmp_4434[3] <== [evals[82][0] + tmp_3573[0], evals[82][1] + tmp_3573[1], evals[82][2] + tmp_3573[2]];
    signal tmp_4435[3] <== CMul()(tmp_3578, tmp_4434);
    signal tmp_4436[3] <== [tmp_4435[0] * 14102670999874605824, tmp_4435[1] * 14102670999874605824, tmp_4435[2] * 14102670999874605824];
    signal tmp_4437[3] <== [tmp_4397[0] * 15585654191999307702, tmp_4397[1] * 15585654191999307702, tmp_4397[2] * 15585654191999307702];
    signal tmp_4438[3] <== [tmp_4437[0] + tmp_4428[0], tmp_4437[1] + tmp_4428[1], tmp_4437[2] + tmp_4428[2]];
    signal tmp_4439[3] <== [tmp_4435[0] + tmp_4438[0], tmp_4435[1] + tmp_4438[1], tmp_4435[2] + tmp_4438[2]];
    signal tmp_4440[3] <== [tmp_4400[0] * 940187017142450255, tmp_4400[1] * 940187017142450255, tmp_4400[2] * 940187017142450255];
    signal tmp_4441[3] <== [tmp_4440[0] + tmp_4428[0], tmp_4440[1] + tmp_4428[1], tmp_4440[2] + tmp_4428[2]];
    signal tmp_4442[3] <== [tmp_4439[0] + tmp_4441[0], tmp_4439[1] + tmp_4441[1], tmp_4439[2] + tmp_4441[2]];
    signal tmp_4443[3] <== [tmp_4403[0] * 8747386241522630711, tmp_4403[1] * 8747386241522630711, tmp_4403[2] * 8747386241522630711];
    signal tmp_4444[3] <== [tmp_4443[0] + tmp_4428[0], tmp_4443[1] + tmp_4428[1], tmp_4443[2] + tmp_4428[2]];
    signal tmp_4445[3] <== [tmp_4442[0] + tmp_4444[0], tmp_4442[1] + tmp_4444[1], tmp_4442[2] + tmp_4444[2]];
    signal tmp_4446[3] <== [tmp_4406[0] * 6750641561540124747, tmp_4406[1] * 6750641561540124747, tmp_4406[2] * 6750641561540124747];
    signal tmp_4447[3] <== [tmp_4446[0] + tmp_4428[0], tmp_4446[1] + tmp_4428[1], tmp_4446[2] + tmp_4428[2]];
    signal tmp_4448[3] <== [tmp_4445[0] + tmp_4447[0], tmp_4445[1] + tmp_4447[1], tmp_4445[2] + tmp_4447[2]];
    signal tmp_4449[3] <== [tmp_4409[0] * 7440998025584530007, tmp_4409[1] * 7440998025584530007, tmp_4409[2] * 7440998025584530007];
    signal tmp_4450[3] <== [tmp_4449[0] + tmp_4428[0], tmp_4449[1] + tmp_4428[1], tmp_4449[2] + tmp_4428[2]];
    signal tmp_4451[3] <== [tmp_4448[0] + tmp_4450[0], tmp_4448[1] + tmp_4450[1], tmp_4448[2] + tmp_4450[2]];
    signal tmp_4452[3] <== [tmp_4412[0] * 6136358134615751536, tmp_4412[1] * 6136358134615751536, tmp_4412[2] * 6136358134615751536];
    signal tmp_4453[3] <== [tmp_4452[0] + tmp_4428[0], tmp_4452[1] + tmp_4428[1], tmp_4452[2] + tmp_4428[2]];
    signal tmp_4454[3] <== [tmp_4451[0] + tmp_4453[0], tmp_4451[1] + tmp_4453[1], tmp_4451[2] + tmp_4453[2]];
    signal tmp_4455[3] <== [tmp_4415[0] * 12413576830284969611, tmp_4415[1] * 12413576830284969611, tmp_4415[2] * 12413576830284969611];
    signal tmp_4456[3] <== [tmp_4455[0] + tmp_4428[0], tmp_4455[1] + tmp_4428[1], tmp_4455[2] + tmp_4428[2]];
    signal tmp_4457[3] <== [tmp_4454[0] + tmp_4456[0], tmp_4454[1] + tmp_4456[1], tmp_4454[2] + tmp_4456[2]];
    signal tmp_4458[3] <== [tmp_4418[0] * 11675438539028694709, tmp_4418[1] * 11675438539028694709, tmp_4418[2] * 11675438539028694709];
    signal tmp_4459[3] <== [tmp_4458[0] + tmp_4428[0], tmp_4458[1] + tmp_4428[1], tmp_4458[2] + tmp_4428[2]];
    signal tmp_4460[3] <== [tmp_4457[0] + tmp_4459[0], tmp_4457[1] + tmp_4459[1], tmp_4457[2] + tmp_4459[2]];
    signal tmp_4461[3] <== [tmp_4421[0] * 17580553691069642926, tmp_4421[1] * 17580553691069642926, tmp_4421[2] * 17580553691069642926];
    signal tmp_4462[3] <== [tmp_4461[0] + tmp_4428[0], tmp_4461[1] + tmp_4428[1], tmp_4461[2] + tmp_4428[2]];
    signal tmp_4463[3] <== [tmp_4460[0] + tmp_4462[0], tmp_4460[1] + tmp_4462[1], tmp_4460[2] + tmp_4462[2]];
    signal tmp_4464[3] <== [tmp_4424[0] * 892707462476851331, tmp_4424[1] * 892707462476851331, tmp_4424[2] * 892707462476851331];
    signal tmp_4465[3] <== [tmp_4464[0] + tmp_4428[0], tmp_4464[1] + tmp_4428[1], tmp_4464[2] + tmp_4428[2]];
    signal tmp_4466[3] <== [tmp_4463[0] + tmp_4465[0], tmp_4463[1] + tmp_4465[1], tmp_4463[2] + tmp_4465[2]];
    signal tmp_4467[3] <== [tmp_4427[0] * 15167485180850043744, tmp_4427[1] * 15167485180850043744, tmp_4427[2] * 15167485180850043744];
    signal tmp_4468[3] <== [tmp_4467[0] + tmp_4428[0], tmp_4467[1] + tmp_4428[1], tmp_4467[2] + tmp_4428[2]];
    signal tmp_4469[3] <== [tmp_4466[0] + tmp_4468[0], tmp_4466[1] + tmp_4468[1], tmp_4466[2] + tmp_4468[2]];
    signal tmp_4470[3] <== [tmp_4436[0] + tmp_4469[0], tmp_4436[1] + tmp_4469[1], tmp_4436[2] + tmp_4469[2]];
    signal tmp_4471[3] <== [evals[83][0] - tmp_4470[0], evals[83][1] - tmp_4470[1], evals[83][2] - tmp_4470[2]];
    signal tmp_4472[3] <== CMul()(evals[51], tmp_4471);
    signal tmp_4473[3] <== [tmp_4433[0] + tmp_4472[0], tmp_4433[1] + tmp_4472[1], tmp_4433[2] + tmp_4472[2]];
    signal tmp_4474[3] <== CMul()(challengeQ, tmp_4473);
    signal tmp_4475[3] <== [evals[83][0] + tmp_3518[0], evals[83][1] + tmp_3518[1], evals[83][2] + tmp_3518[2]];
    signal tmp_4476[3] <== CMul()(tmp_3523, tmp_4475);
    signal tmp_4477[3] <== [tmp_4476[0] * 14102670999874605824, tmp_4476[1] * 14102670999874605824, tmp_4476[2] * 14102670999874605824];
    signal tmp_4478[3] <== [tmp_4438[0] * 15585654191999307702, tmp_4438[1] * 15585654191999307702, tmp_4438[2] * 15585654191999307702];
    signal tmp_4479[3] <== [tmp_4478[0] + tmp_4469[0], tmp_4478[1] + tmp_4469[1], tmp_4478[2] + tmp_4469[2]];
    signal tmp_4480[3] <== [tmp_4476[0] + tmp_4479[0], tmp_4476[1] + tmp_4479[1], tmp_4476[2] + tmp_4479[2]];
    signal tmp_4481[3] <== [tmp_4441[0] * 940187017142450255, tmp_4441[1] * 940187017142450255, tmp_4441[2] * 940187017142450255];
    signal tmp_4482[3] <== [tmp_4481[0] + tmp_4469[0], tmp_4481[1] + tmp_4469[1], tmp_4481[2] + tmp_4469[2]];
    signal tmp_4483[3] <== [tmp_4480[0] + tmp_4482[0], tmp_4480[1] + tmp_4482[1], tmp_4480[2] + tmp_4482[2]];
    signal tmp_4484[3] <== [tmp_4444[0] * 8747386241522630711, tmp_4444[1] * 8747386241522630711, tmp_4444[2] * 8747386241522630711];
    signal tmp_4485[3] <== [tmp_4484[0] + tmp_4469[0], tmp_4484[1] + tmp_4469[1], tmp_4484[2] + tmp_4469[2]];
    signal tmp_4486[3] <== [tmp_4483[0] + tmp_4485[0], tmp_4483[1] + tmp_4485[1], tmp_4483[2] + tmp_4485[2]];
    signal tmp_4487[3] <== [tmp_4447[0] * 6750641561540124747, tmp_4447[1] * 6750641561540124747, tmp_4447[2] * 6750641561540124747];
    signal tmp_4488[3] <== [tmp_4487[0] + tmp_4469[0], tmp_4487[1] + tmp_4469[1], tmp_4487[2] + tmp_4469[2]];
    signal tmp_4489[3] <== [tmp_4486[0] + tmp_4488[0], tmp_4486[1] + tmp_4488[1], tmp_4486[2] + tmp_4488[2]];
    signal tmp_4490[3] <== [tmp_4450[0] * 7440998025584530007, tmp_4450[1] * 7440998025584530007, tmp_4450[2] * 7440998025584530007];
    signal tmp_4491[3] <== [tmp_4490[0] + tmp_4469[0], tmp_4490[1] + tmp_4469[1], tmp_4490[2] + tmp_4469[2]];
    signal tmp_4492[3] <== [tmp_4489[0] + tmp_4491[0], tmp_4489[1] + tmp_4491[1], tmp_4489[2] + tmp_4491[2]];
    signal tmp_4493[3] <== [tmp_4453[0] * 6136358134615751536, tmp_4453[1] * 6136358134615751536, tmp_4453[2] * 6136358134615751536];
    signal tmp_4494[3] <== [tmp_4493[0] + tmp_4469[0], tmp_4493[1] + tmp_4469[1], tmp_4493[2] + tmp_4469[2]];
    signal tmp_4495[3] <== [tmp_4492[0] + tmp_4494[0], tmp_4492[1] + tmp_4494[1], tmp_4492[2] + tmp_4494[2]];
    signal tmp_4496[3] <== [tmp_4456[0] * 12413576830284969611, tmp_4456[1] * 12413576830284969611, tmp_4456[2] * 12413576830284969611];
    signal tmp_4497[3] <== [tmp_4496[0] + tmp_4469[0], tmp_4496[1] + tmp_4469[1], tmp_4496[2] + tmp_4469[2]];
    signal tmp_4498[3] <== [tmp_4495[0] + tmp_4497[0], tmp_4495[1] + tmp_4497[1], tmp_4495[2] + tmp_4497[2]];
    signal tmp_4499[3] <== [tmp_4459[0] * 11675438539028694709, tmp_4459[1] * 11675438539028694709, tmp_4459[2] * 11675438539028694709];
    signal tmp_4500[3] <== [tmp_4499[0] + tmp_4469[0], tmp_4499[1] + tmp_4469[1], tmp_4499[2] + tmp_4469[2]];
    signal tmp_4501[3] <== [tmp_4498[0] + tmp_4500[0], tmp_4498[1] + tmp_4500[1], tmp_4498[2] + tmp_4500[2]];
    signal tmp_4502[3] <== [tmp_4462[0] * 17580553691069642926, tmp_4462[1] * 17580553691069642926, tmp_4462[2] * 17580553691069642926];
    signal tmp_4503[3] <== [tmp_4502[0] + tmp_4469[0], tmp_4502[1] + tmp_4469[1], tmp_4502[2] + tmp_4469[2]];
    signal tmp_4504[3] <== [tmp_4501[0] + tmp_4503[0], tmp_4501[1] + tmp_4503[1], tmp_4501[2] + tmp_4503[2]];
    signal tmp_4505[3] <== [tmp_4465[0] * 892707462476851331, tmp_4465[1] * 892707462476851331, tmp_4465[2] * 892707462476851331];
    signal tmp_4506[3] <== [tmp_4505[0] + tmp_4469[0], tmp_4505[1] + tmp_4469[1], tmp_4505[2] + tmp_4469[2]];
    signal tmp_4507[3] <== [tmp_4504[0] + tmp_4506[0], tmp_4504[1] + tmp_4506[1], tmp_4504[2] + tmp_4506[2]];
    signal tmp_4508[3] <== [tmp_4468[0] * 15167485180850043744, tmp_4468[1] * 15167485180850043744, tmp_4468[2] * 15167485180850043744];
    signal tmp_4509[3] <== [tmp_4508[0] + tmp_4469[0], tmp_4508[1] + tmp_4469[1], tmp_4508[2] + tmp_4469[2]];
    signal tmp_4510[3] <== [tmp_4507[0] + tmp_4509[0], tmp_4507[1] + tmp_4509[1], tmp_4507[2] + tmp_4509[2]];
    signal tmp_4511[3] <== [tmp_4477[0] + tmp_4510[0], tmp_4477[1] + tmp_4510[1], tmp_4477[2] + tmp_4510[2]];
    signal tmp_4512[3] <== [evals[84][0] - tmp_4511[0], evals[84][1] - tmp_4511[1], evals[84][2] - tmp_4511[2]];
    signal tmp_4513[3] <== CMul()(evals[51], tmp_4512);
    signal tmp_4514[3] <== [tmp_4474[0] + tmp_4513[0], tmp_4474[1] + tmp_4513[1], tmp_4474[2] + tmp_4513[2]];
    signal tmp_4515[3] <== CMul()(challengeQ, tmp_4514);
    signal tmp_4516[3] <== [evals[84][0] + tmp_3731[0], evals[84][1] + tmp_3731[1], evals[84][2] + tmp_3731[2]];
    signal tmp_4517[3] <== CMul()(tmp_3736, tmp_4516);
    signal tmp_4518[3] <== [tmp_4517[0] * 14102670999874605824, tmp_4517[1] * 14102670999874605824, tmp_4517[2] * 14102670999874605824];
    signal tmp_4519[3] <== [tmp_4479[0] * 15585654191999307702, tmp_4479[1] * 15585654191999307702, tmp_4479[2] * 15585654191999307702];
    signal tmp_4520[3] <== [tmp_4519[0] + tmp_4510[0], tmp_4519[1] + tmp_4510[1], tmp_4519[2] + tmp_4510[2]];
    signal tmp_4521[3] <== [tmp_4517[0] + tmp_4520[0], tmp_4517[1] + tmp_4520[1], tmp_4517[2] + tmp_4520[2]];
    signal tmp_4522[3] <== [tmp_4482[0] * 940187017142450255, tmp_4482[1] * 940187017142450255, tmp_4482[2] * 940187017142450255];
    signal tmp_4523[3] <== [tmp_4522[0] + tmp_4510[0], tmp_4522[1] + tmp_4510[1], tmp_4522[2] + tmp_4510[2]];
    signal tmp_4524[3] <== [tmp_4521[0] + tmp_4523[0], tmp_4521[1] + tmp_4523[1], tmp_4521[2] + tmp_4523[2]];
    signal tmp_4525[3] <== [tmp_4485[0] * 8747386241522630711, tmp_4485[1] * 8747386241522630711, tmp_4485[2] * 8747386241522630711];
    signal tmp_4526[3] <== [tmp_4525[0] + tmp_4510[0], tmp_4525[1] + tmp_4510[1], tmp_4525[2] + tmp_4510[2]];
    signal tmp_4527[3] <== [tmp_4524[0] + tmp_4526[0], tmp_4524[1] + tmp_4526[1], tmp_4524[2] + tmp_4526[2]];
    signal tmp_4528[3] <== [tmp_4488[0] * 6750641561540124747, tmp_4488[1] * 6750641561540124747, tmp_4488[2] * 6750641561540124747];
    signal tmp_4529[3] <== [tmp_4528[0] + tmp_4510[0], tmp_4528[1] + tmp_4510[1], tmp_4528[2] + tmp_4510[2]];
    signal tmp_4530[3] <== [tmp_4527[0] + tmp_4529[0], tmp_4527[1] + tmp_4529[1], tmp_4527[2] + tmp_4529[2]];
    signal tmp_4531[3] <== [tmp_4491[0] * 7440998025584530007, tmp_4491[1] * 7440998025584530007, tmp_4491[2] * 7440998025584530007];
    signal tmp_4532[3] <== [tmp_4531[0] + tmp_4510[0], tmp_4531[1] + tmp_4510[1], tmp_4531[2] + tmp_4510[2]];
    signal tmp_4533[3] <== [tmp_4530[0] + tmp_4532[0], tmp_4530[1] + tmp_4532[1], tmp_4530[2] + tmp_4532[2]];
    signal tmp_4534[3] <== [tmp_4494[0] * 6136358134615751536, tmp_4494[1] * 6136358134615751536, tmp_4494[2] * 6136358134615751536];
    signal tmp_4535[3] <== [tmp_4534[0] + tmp_4510[0], tmp_4534[1] + tmp_4510[1], tmp_4534[2] + tmp_4510[2]];
    signal tmp_4536[3] <== [tmp_4533[0] + tmp_4535[0], tmp_4533[1] + tmp_4535[1], tmp_4533[2] + tmp_4535[2]];
    signal tmp_4537[3] <== [tmp_4497[0] * 12413576830284969611, tmp_4497[1] * 12413576830284969611, tmp_4497[2] * 12413576830284969611];
    signal tmp_4538[3] <== [tmp_4537[0] + tmp_4510[0], tmp_4537[1] + tmp_4510[1], tmp_4537[2] + tmp_4510[2]];
    signal tmp_4539[3] <== [tmp_4536[0] + tmp_4538[0], tmp_4536[1] + tmp_4538[1], tmp_4536[2] + tmp_4538[2]];
    signal tmp_4540[3] <== [tmp_4500[0] * 11675438539028694709, tmp_4500[1] * 11675438539028694709, tmp_4500[2] * 11675438539028694709];
    signal tmp_4541[3] <== [tmp_4540[0] + tmp_4510[0], tmp_4540[1] + tmp_4510[1], tmp_4540[2] + tmp_4510[2]];
    signal tmp_4542[3] <== [tmp_4539[0] + tmp_4541[0], tmp_4539[1] + tmp_4541[1], tmp_4539[2] + tmp_4541[2]];
    signal tmp_4543[3] <== [tmp_4503[0] * 17580553691069642926, tmp_4503[1] * 17580553691069642926, tmp_4503[2] * 17580553691069642926];
    signal tmp_4544[3] <== [tmp_4543[0] + tmp_4510[0], tmp_4543[1] + tmp_4510[1], tmp_4543[2] + tmp_4510[2]];
    signal tmp_4545[3] <== [tmp_4542[0] + tmp_4544[0], tmp_4542[1] + tmp_4544[1], tmp_4542[2] + tmp_4544[2]];
    signal tmp_4546[3] <== [tmp_4506[0] * 892707462476851331, tmp_4506[1] * 892707462476851331, tmp_4506[2] * 892707462476851331];
    signal tmp_4547[3] <== [tmp_4546[0] + tmp_4510[0], tmp_4546[1] + tmp_4510[1], tmp_4546[2] + tmp_4510[2]];
    signal tmp_4548[3] <== [tmp_4545[0] + tmp_4547[0], tmp_4545[1] + tmp_4547[1], tmp_4545[2] + tmp_4547[2]];
    signal tmp_4549[3] <== [tmp_4509[0] * 15167485180850043744, tmp_4509[1] * 15167485180850043744, tmp_4509[2] * 15167485180850043744];
    signal tmp_4550[3] <== [tmp_4549[0] + tmp_4510[0], tmp_4549[1] + tmp_4510[1], tmp_4549[2] + tmp_4510[2]];
    signal tmp_4551[3] <== [tmp_4548[0] + tmp_4550[0], tmp_4548[1] + tmp_4550[1], tmp_4548[2] + tmp_4550[2]];
    signal tmp_4552[3] <== [tmp_4518[0] + tmp_4551[0], tmp_4518[1] + tmp_4551[1], tmp_4518[2] + tmp_4551[2]];
    signal tmp_4553[3] <== [evals[85][0] - tmp_4552[0], evals[85][1] - tmp_4552[1], evals[85][2] - tmp_4552[2]];
    signal tmp_4554[3] <== CMul()(evals[51], tmp_4553);
    signal tmp_4555[3] <== [tmp_4515[0] + tmp_4554[0], tmp_4515[1] + tmp_4554[1], tmp_4515[2] + tmp_4554[2]];
    signal tmp_4556[3] <== CMul()(challengeQ, tmp_4555);
    signal tmp_4557[3] <== [evals[85][0] + tmp_3747[0], evals[85][1] + tmp_3747[1], evals[85][2] + tmp_3747[2]];
    signal tmp_4558[3] <== CMul()(tmp_3752, tmp_4557);
    signal tmp_4559[3] <== [tmp_4558[0] * 14102670999874605824, tmp_4558[1] * 14102670999874605824, tmp_4558[2] * 14102670999874605824];
    signal tmp_4560[3] <== [tmp_4520[0] * 15585654191999307702, tmp_4520[1] * 15585654191999307702, tmp_4520[2] * 15585654191999307702];
    signal tmp_4561[3] <== [tmp_4560[0] + tmp_4551[0], tmp_4560[1] + tmp_4551[1], tmp_4560[2] + tmp_4551[2]];
    signal tmp_4562[3] <== [tmp_4558[0] + tmp_4561[0], tmp_4558[1] + tmp_4561[1], tmp_4558[2] + tmp_4561[2]];
    signal tmp_4563[3] <== [tmp_4523[0] * 940187017142450255, tmp_4523[1] * 940187017142450255, tmp_4523[2] * 940187017142450255];
    signal tmp_4564[3] <== [tmp_4563[0] + tmp_4551[0], tmp_4563[1] + tmp_4551[1], tmp_4563[2] + tmp_4551[2]];
    signal tmp_4565[3] <== [tmp_4562[0] + tmp_4564[0], tmp_4562[1] + tmp_4564[1], tmp_4562[2] + tmp_4564[2]];
    signal tmp_4566[3] <== [tmp_4526[0] * 8747386241522630711, tmp_4526[1] * 8747386241522630711, tmp_4526[2] * 8747386241522630711];
    signal tmp_4567[3] <== [tmp_4566[0] + tmp_4551[0], tmp_4566[1] + tmp_4551[1], tmp_4566[2] + tmp_4551[2]];
    signal tmp_4568[3] <== [tmp_4565[0] + tmp_4567[0], tmp_4565[1] + tmp_4567[1], tmp_4565[2] + tmp_4567[2]];
    signal tmp_4569[3] <== [tmp_4529[0] * 6750641561540124747, tmp_4529[1] * 6750641561540124747, tmp_4529[2] * 6750641561540124747];
    signal tmp_4570[3] <== [tmp_4569[0] + tmp_4551[0], tmp_4569[1] + tmp_4551[1], tmp_4569[2] + tmp_4551[2]];
    signal tmp_4571[3] <== [tmp_4568[0] + tmp_4570[0], tmp_4568[1] + tmp_4570[1], tmp_4568[2] + tmp_4570[2]];
    signal tmp_4572[3] <== [tmp_4532[0] * 7440998025584530007, tmp_4532[1] * 7440998025584530007, tmp_4532[2] * 7440998025584530007];
    signal tmp_4573[3] <== [tmp_4572[0] + tmp_4551[0], tmp_4572[1] + tmp_4551[1], tmp_4572[2] + tmp_4551[2]];
    signal tmp_4574[3] <== [tmp_4571[0] + tmp_4573[0], tmp_4571[1] + tmp_4573[1], tmp_4571[2] + tmp_4573[2]];
    signal tmp_4575[3] <== [tmp_4535[0] * 6136358134615751536, tmp_4535[1] * 6136358134615751536, tmp_4535[2] * 6136358134615751536];
    signal tmp_4576[3] <== [tmp_4575[0] + tmp_4551[0], tmp_4575[1] + tmp_4551[1], tmp_4575[2] + tmp_4551[2]];
    signal tmp_4577[3] <== [tmp_4574[0] + tmp_4576[0], tmp_4574[1] + tmp_4576[1], tmp_4574[2] + tmp_4576[2]];
    signal tmp_4578[3] <== [tmp_4538[0] * 12413576830284969611, tmp_4538[1] * 12413576830284969611, tmp_4538[2] * 12413576830284969611];
    signal tmp_4579[3] <== [tmp_4578[0] + tmp_4551[0], tmp_4578[1] + tmp_4551[1], tmp_4578[2] + tmp_4551[2]];
    signal tmp_4580[3] <== [tmp_4577[0] + tmp_4579[0], tmp_4577[1] + tmp_4579[1], tmp_4577[2] + tmp_4579[2]];
    signal tmp_4581[3] <== [tmp_4541[0] * 11675438539028694709, tmp_4541[1] * 11675438539028694709, tmp_4541[2] * 11675438539028694709];
    signal tmp_4582[3] <== [tmp_4581[0] + tmp_4551[0], tmp_4581[1] + tmp_4551[1], tmp_4581[2] + tmp_4551[2]];
    signal tmp_4583[3] <== [tmp_4580[0] + tmp_4582[0], tmp_4580[1] + tmp_4582[1], tmp_4580[2] + tmp_4582[2]];
    signal tmp_4584[3] <== [tmp_4544[0] * 17580553691069642926, tmp_4544[1] * 17580553691069642926, tmp_4544[2] * 17580553691069642926];
    signal tmp_4585[3] <== [tmp_4584[0] + tmp_4551[0], tmp_4584[1] + tmp_4551[1], tmp_4584[2] + tmp_4551[2]];
    signal tmp_4586[3] <== [tmp_4583[0] + tmp_4585[0], tmp_4583[1] + tmp_4585[1], tmp_4583[2] + tmp_4585[2]];
    signal tmp_4587[3] <== [tmp_4547[0] * 892707462476851331, tmp_4547[1] * 892707462476851331, tmp_4547[2] * 892707462476851331];
    signal tmp_4588[3] <== [tmp_4587[0] + tmp_4551[0], tmp_4587[1] + tmp_4551[1], tmp_4587[2] + tmp_4551[2]];
    signal tmp_4589[3] <== [tmp_4586[0] + tmp_4588[0], tmp_4586[1] + tmp_4588[1], tmp_4586[2] + tmp_4588[2]];
    signal tmp_4590[3] <== [tmp_4550[0] * 15167485180850043744, tmp_4550[1] * 15167485180850043744, tmp_4550[2] * 15167485180850043744];
    signal tmp_4591[3] <== [tmp_4590[0] + tmp_4551[0], tmp_4590[1] + tmp_4551[1], tmp_4590[2] + tmp_4551[2]];
    signal tmp_4592[3] <== [tmp_4589[0] + tmp_4591[0], tmp_4589[1] + tmp_4591[1], tmp_4589[2] + tmp_4591[2]];
    signal tmp_4593[3] <== [tmp_4559[0] + tmp_4592[0], tmp_4559[1] + tmp_4592[1], tmp_4559[2] + tmp_4592[2]];
    signal tmp_4594[3] <== [evals[86][0] - tmp_4593[0], evals[86][1] - tmp_4593[1], evals[86][2] - tmp_4593[2]];
    signal tmp_4595[3] <== CMul()(evals[51], tmp_4594);
    signal tmp_4596[3] <== [tmp_4556[0] + tmp_4595[0], tmp_4556[1] + tmp_4595[1], tmp_4556[2] + tmp_4595[2]];
    signal tmp_4597[3] <== CMul()(challengeQ, tmp_4596);
    signal tmp_4598[3] <== [evals[86][0] + tmp_3769[0], evals[86][1] + tmp_3769[1], evals[86][2] + tmp_3769[2]];
    signal tmp_4599[3] <== CMul()(tmp_3774, tmp_4598);
    signal tmp_4600[3] <== [tmp_4599[0] * 14102670999874605824, tmp_4599[1] * 14102670999874605824, tmp_4599[2] * 14102670999874605824];
    signal tmp_4601[3] <== [tmp_4561[0] * 15585654191999307702, tmp_4561[1] * 15585654191999307702, tmp_4561[2] * 15585654191999307702];
    signal tmp_4602[3] <== [tmp_4601[0] + tmp_4592[0], tmp_4601[1] + tmp_4592[1], tmp_4601[2] + tmp_4592[2]];
    signal tmp_4603[3] <== [tmp_4599[0] + tmp_4602[0], tmp_4599[1] + tmp_4602[1], tmp_4599[2] + tmp_4602[2]];
    signal tmp_4604[3] <== [tmp_4564[0] * 940187017142450255, tmp_4564[1] * 940187017142450255, tmp_4564[2] * 940187017142450255];
    signal tmp_4605[3] <== [tmp_4604[0] + tmp_4592[0], tmp_4604[1] + tmp_4592[1], tmp_4604[2] + tmp_4592[2]];
    signal tmp_4606[3] <== [tmp_4603[0] + tmp_4605[0], tmp_4603[1] + tmp_4605[1], tmp_4603[2] + tmp_4605[2]];
    signal tmp_4607[3] <== [tmp_4567[0] * 8747386241522630711, tmp_4567[1] * 8747386241522630711, tmp_4567[2] * 8747386241522630711];
    signal tmp_4608[3] <== [tmp_4607[0] + tmp_4592[0], tmp_4607[1] + tmp_4592[1], tmp_4607[2] + tmp_4592[2]];
    signal tmp_4609[3] <== [tmp_4606[0] + tmp_4608[0], tmp_4606[1] + tmp_4608[1], tmp_4606[2] + tmp_4608[2]];
    signal tmp_4610[3] <== [tmp_4570[0] * 6750641561540124747, tmp_4570[1] * 6750641561540124747, tmp_4570[2] * 6750641561540124747];
    signal tmp_4611[3] <== [tmp_4610[0] + tmp_4592[0], tmp_4610[1] + tmp_4592[1], tmp_4610[2] + tmp_4592[2]];
    signal tmp_4612[3] <== [tmp_4609[0] + tmp_4611[0], tmp_4609[1] + tmp_4611[1], tmp_4609[2] + tmp_4611[2]];
    signal tmp_4613[3] <== [tmp_4573[0] * 7440998025584530007, tmp_4573[1] * 7440998025584530007, tmp_4573[2] * 7440998025584530007];
    signal tmp_4614[3] <== [tmp_4613[0] + tmp_4592[0], tmp_4613[1] + tmp_4592[1], tmp_4613[2] + tmp_4592[2]];
    signal tmp_4615[3] <== [tmp_4612[0] + tmp_4614[0], tmp_4612[1] + tmp_4614[1], tmp_4612[2] + tmp_4614[2]];
    signal tmp_4616[3] <== [tmp_4576[0] * 6136358134615751536, tmp_4576[1] * 6136358134615751536, tmp_4576[2] * 6136358134615751536];
    signal tmp_4617[3] <== [tmp_4616[0] + tmp_4592[0], tmp_4616[1] + tmp_4592[1], tmp_4616[2] + tmp_4592[2]];
    signal tmp_4618[3] <== [tmp_4615[0] + tmp_4617[0], tmp_4615[1] + tmp_4617[1], tmp_4615[2] + tmp_4617[2]];
    signal tmp_4619[3] <== [tmp_4579[0] * 12413576830284969611, tmp_4579[1] * 12413576830284969611, tmp_4579[2] * 12413576830284969611];
    signal tmp_4620[3] <== [tmp_4619[0] + tmp_4592[0], tmp_4619[1] + tmp_4592[1], tmp_4619[2] + tmp_4592[2]];
    signal tmp_4621[3] <== [tmp_4618[0] + tmp_4620[0], tmp_4618[1] + tmp_4620[1], tmp_4618[2] + tmp_4620[2]];
    signal tmp_4622[3] <== [tmp_4582[0] * 11675438539028694709, tmp_4582[1] * 11675438539028694709, tmp_4582[2] * 11675438539028694709];
    signal tmp_4623[3] <== [tmp_4622[0] + tmp_4592[0], tmp_4622[1] + tmp_4592[1], tmp_4622[2] + tmp_4592[2]];
    signal tmp_4624[3] <== [tmp_4621[0] + tmp_4623[0], tmp_4621[1] + tmp_4623[1], tmp_4621[2] + tmp_4623[2]];
    signal tmp_4625[3] <== [tmp_4585[0] * 17580553691069642926, tmp_4585[1] * 17580553691069642926, tmp_4585[2] * 17580553691069642926];
    signal tmp_4626[3] <== [tmp_4625[0] + tmp_4592[0], tmp_4625[1] + tmp_4592[1], tmp_4625[2] + tmp_4592[2]];
    signal tmp_4627[3] <== [tmp_4624[0] + tmp_4626[0], tmp_4624[1] + tmp_4626[1], tmp_4624[2] + tmp_4626[2]];
    signal tmp_4628[3] <== [tmp_4588[0] * 892707462476851331, tmp_4588[1] * 892707462476851331, tmp_4588[2] * 892707462476851331];
    signal tmp_4629[3] <== [tmp_4628[0] + tmp_4592[0], tmp_4628[1] + tmp_4592[1], tmp_4628[2] + tmp_4592[2]];
    signal tmp_4630[3] <== [tmp_4627[0] + tmp_4629[0], tmp_4627[1] + tmp_4629[1], tmp_4627[2] + tmp_4629[2]];
    signal tmp_4631[3] <== [tmp_4591[0] * 15167485180850043744, tmp_4591[1] * 15167485180850043744, tmp_4591[2] * 15167485180850043744];
    signal tmp_4632[3] <== [tmp_4631[0] + tmp_4592[0], tmp_4631[1] + tmp_4592[1], tmp_4631[2] + tmp_4592[2]];
    signal tmp_4633[3] <== [tmp_4630[0] + tmp_4632[0], tmp_4630[1] + tmp_4632[1], tmp_4630[2] + tmp_4632[2]];
    signal tmp_4634[3] <== [tmp_4600[0] + tmp_4633[0], tmp_4600[1] + tmp_4633[1], tmp_4600[2] + tmp_4633[2]];
    signal tmp_4635[3] <== [evals[87][0] - tmp_4634[0], evals[87][1] - tmp_4634[1], evals[87][2] - tmp_4634[2]];
    signal tmp_4636[3] <== CMul()(evals[51], tmp_4635);
    signal tmp_4637[3] <== [tmp_4597[0] + tmp_4636[0], tmp_4597[1] + tmp_4636[1], tmp_4597[2] + tmp_4636[2]];
    signal tmp_4638[3] <== CMul()(challengeQ, tmp_4637);
    signal tmp_4639[3] <== [evals[87][0] + tmp_3714[0], evals[87][1] + tmp_3714[1], evals[87][2] + tmp_3714[2]];
    signal tmp_4640[3] <== CMul()(tmp_3719, tmp_4639);
    signal tmp_4641[3] <== [tmp_4640[0] * 14102670999874605824, tmp_4640[1] * 14102670999874605824, tmp_4640[2] * 14102670999874605824];
    signal tmp_4642[3] <== [tmp_4602[0] * 15585654191999307702, tmp_4602[1] * 15585654191999307702, tmp_4602[2] * 15585654191999307702];
    signal tmp_4643[3] <== [tmp_4642[0] + tmp_4633[0], tmp_4642[1] + tmp_4633[1], tmp_4642[2] + tmp_4633[2]];
    signal tmp_4644[3] <== [tmp_4640[0] + tmp_4643[0], tmp_4640[1] + tmp_4643[1], tmp_4640[2] + tmp_4643[2]];
    signal tmp_4645[3] <== [tmp_4605[0] * 940187017142450255, tmp_4605[1] * 940187017142450255, tmp_4605[2] * 940187017142450255];
    signal tmp_4646[3] <== [tmp_4645[0] + tmp_4633[0], tmp_4645[1] + tmp_4633[1], tmp_4645[2] + tmp_4633[2]];
    signal tmp_4647[3] <== [tmp_4644[0] + tmp_4646[0], tmp_4644[1] + tmp_4646[1], tmp_4644[2] + tmp_4646[2]];
    signal tmp_4648[3] <== [tmp_4608[0] * 8747386241522630711, tmp_4608[1] * 8747386241522630711, tmp_4608[2] * 8747386241522630711];
    signal tmp_4649[3] <== [tmp_4648[0] + tmp_4633[0], tmp_4648[1] + tmp_4633[1], tmp_4648[2] + tmp_4633[2]];
    signal tmp_4650[3] <== [tmp_4647[0] + tmp_4649[0], tmp_4647[1] + tmp_4649[1], tmp_4647[2] + tmp_4649[2]];
    signal tmp_4651[3] <== [tmp_4611[0] * 6750641561540124747, tmp_4611[1] * 6750641561540124747, tmp_4611[2] * 6750641561540124747];
    signal tmp_4652[3] <== [tmp_4651[0] + tmp_4633[0], tmp_4651[1] + tmp_4633[1], tmp_4651[2] + tmp_4633[2]];
    signal tmp_4653[3] <== [tmp_4650[0] + tmp_4652[0], tmp_4650[1] + tmp_4652[1], tmp_4650[2] + tmp_4652[2]];
    signal tmp_4654[3] <== [tmp_4614[0] * 7440998025584530007, tmp_4614[1] * 7440998025584530007, tmp_4614[2] * 7440998025584530007];
    signal tmp_4655[3] <== [tmp_4654[0] + tmp_4633[0], tmp_4654[1] + tmp_4633[1], tmp_4654[2] + tmp_4633[2]];
    signal tmp_4656[3] <== [tmp_4653[0] + tmp_4655[0], tmp_4653[1] + tmp_4655[1], tmp_4653[2] + tmp_4655[2]];
    signal tmp_4657[3] <== [tmp_4617[0] * 6136358134615751536, tmp_4617[1] * 6136358134615751536, tmp_4617[2] * 6136358134615751536];
    signal tmp_4658[3] <== [tmp_4657[0] + tmp_4633[0], tmp_4657[1] + tmp_4633[1], tmp_4657[2] + tmp_4633[2]];
    signal tmp_4659[3] <== [tmp_4656[0] + tmp_4658[0], tmp_4656[1] + tmp_4658[1], tmp_4656[2] + tmp_4658[2]];
    signal tmp_4660[3] <== [tmp_4620[0] * 12413576830284969611, tmp_4620[1] * 12413576830284969611, tmp_4620[2] * 12413576830284969611];
    signal tmp_4661[3] <== [tmp_4660[0] + tmp_4633[0], tmp_4660[1] + tmp_4633[1], tmp_4660[2] + tmp_4633[2]];
    signal tmp_4662[3] <== [tmp_4659[0] + tmp_4661[0], tmp_4659[1] + tmp_4661[1], tmp_4659[2] + tmp_4661[2]];
    signal tmp_4663[3] <== [tmp_4623[0] * 11675438539028694709, tmp_4623[1] * 11675438539028694709, tmp_4623[2] * 11675438539028694709];
    signal tmp_4664[3] <== [tmp_4663[0] + tmp_4633[0], tmp_4663[1] + tmp_4633[1], tmp_4663[2] + tmp_4633[2]];
    signal tmp_4665[3] <== [tmp_4662[0] + tmp_4664[0], tmp_4662[1] + tmp_4664[1], tmp_4662[2] + tmp_4664[2]];
    signal tmp_4666[3] <== [tmp_4626[0] * 17580553691069642926, tmp_4626[1] * 17580553691069642926, tmp_4626[2] * 17580553691069642926];
    signal tmp_4667[3] <== [tmp_4666[0] + tmp_4633[0], tmp_4666[1] + tmp_4633[1], tmp_4666[2] + tmp_4633[2]];
    signal tmp_4668[3] <== [tmp_4665[0] + tmp_4667[0], tmp_4665[1] + tmp_4667[1], tmp_4665[2] + tmp_4667[2]];
    signal tmp_4669[3] <== [tmp_4629[0] * 892707462476851331, tmp_4629[1] * 892707462476851331, tmp_4629[2] * 892707462476851331];
    signal tmp_4670[3] <== [tmp_4669[0] + tmp_4633[0], tmp_4669[1] + tmp_4633[1], tmp_4669[2] + tmp_4633[2]];
    signal tmp_4671[3] <== [tmp_4668[0] + tmp_4670[0], tmp_4668[1] + tmp_4670[1], tmp_4668[2] + tmp_4670[2]];
    signal tmp_4672[3] <== [tmp_4632[0] * 15167485180850043744, tmp_4632[1] * 15167485180850043744, tmp_4632[2] * 15167485180850043744];
    signal tmp_4673[3] <== [tmp_4672[0] + tmp_4633[0], tmp_4672[1] + tmp_4633[1], tmp_4672[2] + tmp_4633[2]];
    signal tmp_4674[3] <== [tmp_4671[0] + tmp_4673[0], tmp_4671[1] + tmp_4673[1], tmp_4671[2] + tmp_4673[2]];
    signal tmp_4675[3] <== [tmp_4641[0] + tmp_4674[0], tmp_4641[1] + tmp_4674[1], tmp_4641[2] + tmp_4674[2]];
    signal tmp_4676[3] <== [evals[88][0] - tmp_4675[0], evals[88][1] - tmp_4675[1], evals[88][2] - tmp_4675[2]];
    signal tmp_4677[3] <== CMul()(evals[51], tmp_4676);
    signal tmp_4678[3] <== [tmp_4638[0] + tmp_4677[0], tmp_4638[1] + tmp_4677[1], tmp_4638[2] + tmp_4677[2]];
    signal tmp_4679[3] <== CMul()(challengeQ, tmp_4678);
    signal tmp_4680[3] <== [evals[88][0] + tmp_3808[0], evals[88][1] + tmp_3808[1], evals[88][2] + tmp_3808[2]];
    signal tmp_4681[3] <== CMul()(tmp_3813, tmp_4680);
    signal tmp_4682[3] <== [tmp_4681[0] * 14102670999874605824, tmp_4681[1] * 14102670999874605824, tmp_4681[2] * 14102670999874605824];
    signal tmp_4683[3] <== [tmp_4643[0] * 15585654191999307702, tmp_4643[1] * 15585654191999307702, tmp_4643[2] * 15585654191999307702];
    signal tmp_4684[3] <== [tmp_4683[0] + tmp_4674[0], tmp_4683[1] + tmp_4674[1], tmp_4683[2] + tmp_4674[2]];
    signal tmp_4685[3] <== [tmp_4681[0] + tmp_4684[0], tmp_4681[1] + tmp_4684[1], tmp_4681[2] + tmp_4684[2]];
    signal tmp_4686[3] <== [tmp_4646[0] * 940187017142450255, tmp_4646[1] * 940187017142450255, tmp_4646[2] * 940187017142450255];
    signal tmp_4687[3] <== [tmp_4686[0] + tmp_4674[0], tmp_4686[1] + tmp_4674[1], tmp_4686[2] + tmp_4674[2]];
    signal tmp_4688[3] <== [tmp_4685[0] + tmp_4687[0], tmp_4685[1] + tmp_4687[1], tmp_4685[2] + tmp_4687[2]];
    signal tmp_4689[3] <== [tmp_4649[0] * 8747386241522630711, tmp_4649[1] * 8747386241522630711, tmp_4649[2] * 8747386241522630711];
    signal tmp_4690[3] <== [tmp_4689[0] + tmp_4674[0], tmp_4689[1] + tmp_4674[1], tmp_4689[2] + tmp_4674[2]];
    signal tmp_4691[3] <== [tmp_4688[0] + tmp_4690[0], tmp_4688[1] + tmp_4690[1], tmp_4688[2] + tmp_4690[2]];
    signal tmp_4692[3] <== [tmp_4652[0] * 6750641561540124747, tmp_4652[1] * 6750641561540124747, tmp_4652[2] * 6750641561540124747];
    signal tmp_4693[3] <== [tmp_4692[0] + tmp_4674[0], tmp_4692[1] + tmp_4674[1], tmp_4692[2] + tmp_4674[2]];
    signal tmp_4694[3] <== [tmp_4691[0] + tmp_4693[0], tmp_4691[1] + tmp_4693[1], tmp_4691[2] + tmp_4693[2]];
    signal tmp_4695[3] <== [tmp_4655[0] * 7440998025584530007, tmp_4655[1] * 7440998025584530007, tmp_4655[2] * 7440998025584530007];
    signal tmp_4696[3] <== [tmp_4695[0] + tmp_4674[0], tmp_4695[1] + tmp_4674[1], tmp_4695[2] + tmp_4674[2]];
    signal tmp_4697[3] <== [tmp_4694[0] + tmp_4696[0], tmp_4694[1] + tmp_4696[1], tmp_4694[2] + tmp_4696[2]];
    signal tmp_4698[3] <== [tmp_4658[0] * 6136358134615751536, tmp_4658[1] * 6136358134615751536, tmp_4658[2] * 6136358134615751536];
    signal tmp_4699[3] <== [tmp_4698[0] + tmp_4674[0], tmp_4698[1] + tmp_4674[1], tmp_4698[2] + tmp_4674[2]];
    signal tmp_4700[3] <== [tmp_4697[0] + tmp_4699[0], tmp_4697[1] + tmp_4699[1], tmp_4697[2] + tmp_4699[2]];
    signal tmp_4701[3] <== [tmp_4661[0] * 12413576830284969611, tmp_4661[1] * 12413576830284969611, tmp_4661[2] * 12413576830284969611];
    signal tmp_4702[3] <== [tmp_4701[0] + tmp_4674[0], tmp_4701[1] + tmp_4674[1], tmp_4701[2] + tmp_4674[2]];
    signal tmp_4703[3] <== [tmp_4700[0] + tmp_4702[0], tmp_4700[1] + tmp_4702[1], tmp_4700[2] + tmp_4702[2]];
    signal tmp_4704[3] <== [tmp_4664[0] * 11675438539028694709, tmp_4664[1] * 11675438539028694709, tmp_4664[2] * 11675438539028694709];
    signal tmp_4705[3] <== [tmp_4704[0] + tmp_4674[0], tmp_4704[1] + tmp_4674[1], tmp_4704[2] + tmp_4674[2]];
    signal tmp_4706[3] <== [tmp_4703[0] + tmp_4705[0], tmp_4703[1] + tmp_4705[1], tmp_4703[2] + tmp_4705[2]];
    signal tmp_4707[3] <== [tmp_4667[0] * 17580553691069642926, tmp_4667[1] * 17580553691069642926, tmp_4667[2] * 17580553691069642926];
    signal tmp_4708[3] <== [tmp_4707[0] + tmp_4674[0], tmp_4707[1] + tmp_4674[1], tmp_4707[2] + tmp_4674[2]];
    signal tmp_4709[3] <== [tmp_4706[0] + tmp_4708[0], tmp_4706[1] + tmp_4708[1], tmp_4706[2] + tmp_4708[2]];
    signal tmp_4710[3] <== [tmp_4670[0] * 892707462476851331, tmp_4670[1] * 892707462476851331, tmp_4670[2] * 892707462476851331];
    signal tmp_4711[3] <== [tmp_4710[0] + tmp_4674[0], tmp_4710[1] + tmp_4674[1], tmp_4710[2] + tmp_4674[2]];
    signal tmp_4712[3] <== [tmp_4709[0] + tmp_4711[0], tmp_4709[1] + tmp_4711[1], tmp_4709[2] + tmp_4711[2]];
    signal tmp_4713[3] <== [tmp_4673[0] * 15167485180850043744, tmp_4673[1] * 15167485180850043744, tmp_4673[2] * 15167485180850043744];
    signal tmp_4714[3] <== [tmp_4713[0] + tmp_4674[0], tmp_4713[1] + tmp_4674[1], tmp_4713[2] + tmp_4674[2]];
    signal tmp_4715[3] <== [tmp_4712[0] + tmp_4714[0], tmp_4712[1] + tmp_4714[1], tmp_4712[2] + tmp_4714[2]];
    signal tmp_4716[3] <== [tmp_4682[0] + tmp_4715[0], tmp_4682[1] + tmp_4715[1], tmp_4682[2] + tmp_4715[2]];
    signal tmp_4717[3] <== [evals[89][0] - tmp_4716[0], evals[89][1] - tmp_4716[1], evals[89][2] - tmp_4716[2]];
    signal tmp_4718[3] <== CMul()(evals[51], tmp_4717);
    signal tmp_4719[3] <== [tmp_4679[0] + tmp_4718[0], tmp_4679[1] + tmp_4718[1], tmp_4679[2] + tmp_4718[2]];
    signal tmp_4720[3] <== CMul()(challengeQ, tmp_4719);
    signal tmp_4721[3] <== [evals[89][0] + tmp_3824[0], evals[89][1] + tmp_3824[1], evals[89][2] + tmp_3824[2]];
    signal tmp_4722[3] <== CMul()(tmp_3829, tmp_4721);
    signal tmp_4723[3] <== [tmp_4722[0] * 14102670999874605824, tmp_4722[1] * 14102670999874605824, tmp_4722[2] * 14102670999874605824];
    signal tmp_4724[3] <== [tmp_4684[0] * 15585654191999307702, tmp_4684[1] * 15585654191999307702, tmp_4684[2] * 15585654191999307702];
    signal tmp_4725[3] <== [tmp_4724[0] + tmp_4715[0], tmp_4724[1] + tmp_4715[1], tmp_4724[2] + tmp_4715[2]];
    signal tmp_4726[3] <== [tmp_4722[0] + tmp_4725[0], tmp_4722[1] + tmp_4725[1], tmp_4722[2] + tmp_4725[2]];
    signal tmp_4727[3] <== [tmp_4687[0] * 940187017142450255, tmp_4687[1] * 940187017142450255, tmp_4687[2] * 940187017142450255];
    signal tmp_4728[3] <== [tmp_4727[0] + tmp_4715[0], tmp_4727[1] + tmp_4715[1], tmp_4727[2] + tmp_4715[2]];
    signal tmp_4729[3] <== [tmp_4726[0] + tmp_4728[0], tmp_4726[1] + tmp_4728[1], tmp_4726[2] + tmp_4728[2]];
    signal tmp_4730[3] <== [tmp_4690[0] * 8747386241522630711, tmp_4690[1] * 8747386241522630711, tmp_4690[2] * 8747386241522630711];
    signal tmp_4731[3] <== [tmp_4730[0] + tmp_4715[0], tmp_4730[1] + tmp_4715[1], tmp_4730[2] + tmp_4715[2]];
    signal tmp_4732[3] <== [tmp_4729[0] + tmp_4731[0], tmp_4729[1] + tmp_4731[1], tmp_4729[2] + tmp_4731[2]];
    signal tmp_4733[3] <== [tmp_4693[0] * 6750641561540124747, tmp_4693[1] * 6750641561540124747, tmp_4693[2] * 6750641561540124747];
    signal tmp_4734[3] <== [tmp_4733[0] + tmp_4715[0], tmp_4733[1] + tmp_4715[1], tmp_4733[2] + tmp_4715[2]];
    signal tmp_4735[3] <== [tmp_4732[0] + tmp_4734[0], tmp_4732[1] + tmp_4734[1], tmp_4732[2] + tmp_4734[2]];
    signal tmp_4736[3] <== [tmp_4696[0] * 7440998025584530007, tmp_4696[1] * 7440998025584530007, tmp_4696[2] * 7440998025584530007];
    signal tmp_4737[3] <== [tmp_4736[0] + tmp_4715[0], tmp_4736[1] + tmp_4715[1], tmp_4736[2] + tmp_4715[2]];
    signal tmp_4738[3] <== [tmp_4735[0] + tmp_4737[0], tmp_4735[1] + tmp_4737[1], tmp_4735[2] + tmp_4737[2]];
    signal tmp_4739[3] <== [tmp_4699[0] * 6136358134615751536, tmp_4699[1] * 6136358134615751536, tmp_4699[2] * 6136358134615751536];
    signal tmp_4740[3] <== [tmp_4739[0] + tmp_4715[0], tmp_4739[1] + tmp_4715[1], tmp_4739[2] + tmp_4715[2]];
    signal tmp_4741[3] <== [tmp_4738[0] + tmp_4740[0], tmp_4738[1] + tmp_4740[1], tmp_4738[2] + tmp_4740[2]];
    signal tmp_4742[3] <== [tmp_4702[0] * 12413576830284969611, tmp_4702[1] * 12413576830284969611, tmp_4702[2] * 12413576830284969611];
    signal tmp_4743[3] <== [tmp_4742[0] + tmp_4715[0], tmp_4742[1] + tmp_4715[1], tmp_4742[2] + tmp_4715[2]];
    signal tmp_4744[3] <== [tmp_4741[0] + tmp_4743[0], tmp_4741[1] + tmp_4743[1], tmp_4741[2] + tmp_4743[2]];
    signal tmp_4745[3] <== [tmp_4705[0] * 11675438539028694709, tmp_4705[1] * 11675438539028694709, tmp_4705[2] * 11675438539028694709];
    signal tmp_4746[3] <== [tmp_4745[0] + tmp_4715[0], tmp_4745[1] + tmp_4715[1], tmp_4745[2] + tmp_4715[2]];
    signal tmp_4747[3] <== [tmp_4744[0] + tmp_4746[0], tmp_4744[1] + tmp_4746[1], tmp_4744[2] + tmp_4746[2]];
    signal tmp_4748[3] <== [tmp_4708[0] * 17580553691069642926, tmp_4708[1] * 17580553691069642926, tmp_4708[2] * 17580553691069642926];
    signal tmp_4749[3] <== [tmp_4748[0] + tmp_4715[0], tmp_4748[1] + tmp_4715[1], tmp_4748[2] + tmp_4715[2]];
    signal tmp_4750[3] <== [tmp_4747[0] + tmp_4749[0], tmp_4747[1] + tmp_4749[1], tmp_4747[2] + tmp_4749[2]];
    signal tmp_4751[3] <== [tmp_4711[0] * 892707462476851331, tmp_4711[1] * 892707462476851331, tmp_4711[2] * 892707462476851331];
    signal tmp_4752[3] <== [tmp_4751[0] + tmp_4715[0], tmp_4751[1] + tmp_4715[1], tmp_4751[2] + tmp_4715[2]];
    signal tmp_4753[3] <== [tmp_4750[0] + tmp_4752[0], tmp_4750[1] + tmp_4752[1], tmp_4750[2] + tmp_4752[2]];
    signal tmp_4754[3] <== [tmp_4714[0] * 15167485180850043744, tmp_4714[1] * 15167485180850043744, tmp_4714[2] * 15167485180850043744];
    signal tmp_4755[3] <== [tmp_4754[0] + tmp_4715[0], tmp_4754[1] + tmp_4715[1], tmp_4754[2] + tmp_4715[2]];
    signal tmp_4756[3] <== [tmp_4753[0] + tmp_4755[0], tmp_4753[1] + tmp_4755[1], tmp_4753[2] + tmp_4755[2]];
    signal tmp_4757[3] <== [tmp_4723[0] + tmp_4756[0], tmp_4723[1] + tmp_4756[1], tmp_4723[2] + tmp_4756[2]];
    signal tmp_4758[3] <== [evals[90][0] - tmp_4757[0], evals[90][1] - tmp_4757[1], evals[90][2] - tmp_4757[2]];
    signal tmp_4759[3] <== CMul()(evals[51], tmp_4758);
    signal tmp_4760[3] <== [tmp_4720[0] + tmp_4759[0], tmp_4720[1] + tmp_4759[1], tmp_4720[2] + tmp_4759[2]];
    signal tmp_4761[3] <== CMul()(challengeQ, tmp_4760);
    signal tmp_4762[3] <== [evals[90][0] + tmp_3846[0], evals[90][1] + tmp_3846[1], evals[90][2] + tmp_3846[2]];
    signal tmp_4763[3] <== CMul()(tmp_3851, tmp_4762);
    signal tmp_4764[3] <== [tmp_4763[0] * 14102670999874605824, tmp_4763[1] * 14102670999874605824, tmp_4763[2] * 14102670999874605824];
    signal tmp_4765[3] <== [tmp_4725[0] * 15585654191999307702, tmp_4725[1] * 15585654191999307702, tmp_4725[2] * 15585654191999307702];
    signal tmp_4766[3] <== [tmp_4765[0] + tmp_4756[0], tmp_4765[1] + tmp_4756[1], tmp_4765[2] + tmp_4756[2]];
    signal tmp_4767[3] <== [tmp_4763[0] + tmp_4766[0], tmp_4763[1] + tmp_4766[1], tmp_4763[2] + tmp_4766[2]];
    signal tmp_4768[3] <== [tmp_4728[0] * 940187017142450255, tmp_4728[1] * 940187017142450255, tmp_4728[2] * 940187017142450255];
    signal tmp_4769[3] <== [tmp_4768[0] + tmp_4756[0], tmp_4768[1] + tmp_4756[1], tmp_4768[2] + tmp_4756[2]];
    signal tmp_4770[3] <== [tmp_4767[0] + tmp_4769[0], tmp_4767[1] + tmp_4769[1], tmp_4767[2] + tmp_4769[2]];
    signal tmp_4771[3] <== [tmp_4731[0] * 8747386241522630711, tmp_4731[1] * 8747386241522630711, tmp_4731[2] * 8747386241522630711];
    signal tmp_4772[3] <== [tmp_4771[0] + tmp_4756[0], tmp_4771[1] + tmp_4756[1], tmp_4771[2] + tmp_4756[2]];
    signal tmp_4773[3] <== [tmp_4770[0] + tmp_4772[0], tmp_4770[1] + tmp_4772[1], tmp_4770[2] + tmp_4772[2]];
    signal tmp_4774[3] <== [tmp_4734[0] * 6750641561540124747, tmp_4734[1] * 6750641561540124747, tmp_4734[2] * 6750641561540124747];
    signal tmp_4775[3] <== [tmp_4774[0] + tmp_4756[0], tmp_4774[1] + tmp_4756[1], tmp_4774[2] + tmp_4756[2]];
    signal tmp_4776[3] <== [tmp_4773[0] + tmp_4775[0], tmp_4773[1] + tmp_4775[1], tmp_4773[2] + tmp_4775[2]];
    signal tmp_4777[3] <== [tmp_4737[0] * 7440998025584530007, tmp_4737[1] * 7440998025584530007, tmp_4737[2] * 7440998025584530007];
    signal tmp_4778[3] <== [tmp_4777[0] + tmp_4756[0], tmp_4777[1] + tmp_4756[1], tmp_4777[2] + tmp_4756[2]];
    signal tmp_4779[3] <== [tmp_4776[0] + tmp_4778[0], tmp_4776[1] + tmp_4778[1], tmp_4776[2] + tmp_4778[2]];
    signal tmp_4780[3] <== [tmp_4740[0] * 6136358134615751536, tmp_4740[1] * 6136358134615751536, tmp_4740[2] * 6136358134615751536];
    signal tmp_4781[3] <== [tmp_4780[0] + tmp_4756[0], tmp_4780[1] + tmp_4756[1], tmp_4780[2] + tmp_4756[2]];
    signal tmp_4782[3] <== [tmp_4779[0] + tmp_4781[0], tmp_4779[1] + tmp_4781[1], tmp_4779[2] + tmp_4781[2]];
    signal tmp_4783[3] <== [tmp_4743[0] * 12413576830284969611, tmp_4743[1] * 12413576830284969611, tmp_4743[2] * 12413576830284969611];
    signal tmp_4784[3] <== [tmp_4783[0] + tmp_4756[0], tmp_4783[1] + tmp_4756[1], tmp_4783[2] + tmp_4756[2]];
    signal tmp_4785[3] <== [tmp_4782[0] + tmp_4784[0], tmp_4782[1] + tmp_4784[1], tmp_4782[2] + tmp_4784[2]];
    signal tmp_4786[3] <== [tmp_4746[0] * 11675438539028694709, tmp_4746[1] * 11675438539028694709, tmp_4746[2] * 11675438539028694709];
    signal tmp_4787[3] <== [tmp_4786[0] + tmp_4756[0], tmp_4786[1] + tmp_4756[1], tmp_4786[2] + tmp_4756[2]];
    signal tmp_4788[3] <== [tmp_4785[0] + tmp_4787[0], tmp_4785[1] + tmp_4787[1], tmp_4785[2] + tmp_4787[2]];
    signal tmp_4789[3] <== [tmp_4749[0] * 17580553691069642926, tmp_4749[1] * 17580553691069642926, tmp_4749[2] * 17580553691069642926];
    signal tmp_4790[3] <== [tmp_4789[0] + tmp_4756[0], tmp_4789[1] + tmp_4756[1], tmp_4789[2] + tmp_4756[2]];
    signal tmp_4791[3] <== [tmp_4788[0] + tmp_4790[0], tmp_4788[1] + tmp_4790[1], tmp_4788[2] + tmp_4790[2]];
    signal tmp_4792[3] <== [tmp_4752[0] * 892707462476851331, tmp_4752[1] * 892707462476851331, tmp_4752[2] * 892707462476851331];
    signal tmp_4793[3] <== [tmp_4792[0] + tmp_4756[0], tmp_4792[1] + tmp_4756[1], tmp_4792[2] + tmp_4756[2]];
    signal tmp_4794[3] <== [tmp_4791[0] + tmp_4793[0], tmp_4791[1] + tmp_4793[1], tmp_4791[2] + tmp_4793[2]];
    signal tmp_4795[3] <== [tmp_4755[0] * 15167485180850043744, tmp_4755[1] * 15167485180850043744, tmp_4755[2] * 15167485180850043744];
    signal tmp_4796[3] <== [tmp_4795[0] + tmp_4756[0], tmp_4795[1] + tmp_4756[1], tmp_4795[2] + tmp_4756[2]];
    signal tmp_4797[3] <== [tmp_4794[0] + tmp_4796[0], tmp_4794[1] + tmp_4796[1], tmp_4794[2] + tmp_4796[2]];
    signal tmp_4798[3] <== [tmp_4764[0] + tmp_4797[0], tmp_4764[1] + tmp_4797[1], tmp_4764[2] + tmp_4797[2]];
    signal tmp_4799[3] <== [evals[91][0] - tmp_4798[0], evals[91][1] - tmp_4798[1], evals[91][2] - tmp_4798[2]];
    signal tmp_4800[3] <== CMul()(evals[51], tmp_4799);
    signal tmp_4801[3] <== [tmp_4761[0] + tmp_4800[0], tmp_4761[1] + tmp_4800[1], tmp_4761[2] + tmp_4800[2]];
    signal tmp_4802[3] <== CMul()(challengeQ, tmp_4801);
    signal tmp_4803[3] <== [evals[91][0] + tmp_3791[0], evals[91][1] + tmp_3791[1], evals[91][2] + tmp_3791[2]];
    signal tmp_4804[3] <== CMul()(tmp_3796, tmp_4803);
    signal tmp_4805[3] <== [tmp_4804[0] * 14102670999874605824, tmp_4804[1] * 14102670999874605824, tmp_4804[2] * 14102670999874605824];
    signal tmp_4806[3] <== [tmp_4766[0] * 15585654191999307702, tmp_4766[1] * 15585654191999307702, tmp_4766[2] * 15585654191999307702];
    signal tmp_4807[3] <== [tmp_4806[0] + tmp_4797[0], tmp_4806[1] + tmp_4797[1], tmp_4806[2] + tmp_4797[2]];
    signal tmp_4808[3] <== [tmp_4804[0] + tmp_4807[0], tmp_4804[1] + tmp_4807[1], tmp_4804[2] + tmp_4807[2]];
    signal tmp_4809[3] <== [tmp_4769[0] * 940187017142450255, tmp_4769[1] * 940187017142450255, tmp_4769[2] * 940187017142450255];
    signal tmp_4810[3] <== [tmp_4809[0] + tmp_4797[0], tmp_4809[1] + tmp_4797[1], tmp_4809[2] + tmp_4797[2]];
    signal tmp_4811[3] <== [tmp_4808[0] + tmp_4810[0], tmp_4808[1] + tmp_4810[1], tmp_4808[2] + tmp_4810[2]];
    signal tmp_4812[3] <== [tmp_4772[0] * 8747386241522630711, tmp_4772[1] * 8747386241522630711, tmp_4772[2] * 8747386241522630711];
    signal tmp_4813[3] <== [tmp_4812[0] + tmp_4797[0], tmp_4812[1] + tmp_4797[1], tmp_4812[2] + tmp_4797[2]];
    signal tmp_4814[3] <== [tmp_4811[0] + tmp_4813[0], tmp_4811[1] + tmp_4813[1], tmp_4811[2] + tmp_4813[2]];
    signal tmp_4815[3] <== [tmp_4775[0] * 6750641561540124747, tmp_4775[1] * 6750641561540124747, tmp_4775[2] * 6750641561540124747];
    signal tmp_4816[3] <== [tmp_4815[0] + tmp_4797[0], tmp_4815[1] + tmp_4797[1], tmp_4815[2] + tmp_4797[2]];
    signal tmp_4817[3] <== [tmp_4814[0] + tmp_4816[0], tmp_4814[1] + tmp_4816[1], tmp_4814[2] + tmp_4816[2]];
    signal tmp_4818[3] <== [tmp_4778[0] * 7440998025584530007, tmp_4778[1] * 7440998025584530007, tmp_4778[2] * 7440998025584530007];
    signal tmp_4819[3] <== [tmp_4818[0] + tmp_4797[0], tmp_4818[1] + tmp_4797[1], tmp_4818[2] + tmp_4797[2]];
    signal tmp_4820[3] <== [tmp_4817[0] + tmp_4819[0], tmp_4817[1] + tmp_4819[1], tmp_4817[2] + tmp_4819[2]];
    signal tmp_4821[3] <== [tmp_4781[0] * 6136358134615751536, tmp_4781[1] * 6136358134615751536, tmp_4781[2] * 6136358134615751536];
    signal tmp_4822[3] <== [tmp_4821[0] + tmp_4797[0], tmp_4821[1] + tmp_4797[1], tmp_4821[2] + tmp_4797[2]];
    signal tmp_4823[3] <== [tmp_4820[0] + tmp_4822[0], tmp_4820[1] + tmp_4822[1], tmp_4820[2] + tmp_4822[2]];
    signal tmp_4824[3] <== [tmp_4784[0] * 12413576830284969611, tmp_4784[1] * 12413576830284969611, tmp_4784[2] * 12413576830284969611];
    signal tmp_4825[3] <== [tmp_4824[0] + tmp_4797[0], tmp_4824[1] + tmp_4797[1], tmp_4824[2] + tmp_4797[2]];
    signal tmp_4826[3] <== [tmp_4823[0] + tmp_4825[0], tmp_4823[1] + tmp_4825[1], tmp_4823[2] + tmp_4825[2]];
    signal tmp_4827[3] <== [tmp_4787[0] * 11675438539028694709, tmp_4787[1] * 11675438539028694709, tmp_4787[2] * 11675438539028694709];
    signal tmp_4828[3] <== [tmp_4827[0] + tmp_4797[0], tmp_4827[1] + tmp_4797[1], tmp_4827[2] + tmp_4797[2]];
    signal tmp_4829[3] <== [tmp_4826[0] + tmp_4828[0], tmp_4826[1] + tmp_4828[1], tmp_4826[2] + tmp_4828[2]];
    signal tmp_4830[3] <== [tmp_4790[0] * 17580553691069642926, tmp_4790[1] * 17580553691069642926, tmp_4790[2] * 17580553691069642926];
    signal tmp_4831[3] <== [tmp_4830[0] + tmp_4797[0], tmp_4830[1] + tmp_4797[1], tmp_4830[2] + tmp_4797[2]];
    signal tmp_4832[3] <== [tmp_4829[0] + tmp_4831[0], tmp_4829[1] + tmp_4831[1], tmp_4829[2] + tmp_4831[2]];
    signal tmp_4833[3] <== [tmp_4793[0] * 892707462476851331, tmp_4793[1] * 892707462476851331, tmp_4793[2] * 892707462476851331];
    signal tmp_4834[3] <== [tmp_4833[0] + tmp_4797[0], tmp_4833[1] + tmp_4797[1], tmp_4833[2] + tmp_4797[2]];
    signal tmp_4835[3] <== [tmp_4832[0] + tmp_4834[0], tmp_4832[1] + tmp_4834[1], tmp_4832[2] + tmp_4834[2]];
    signal tmp_4836[3] <== [tmp_4796[0] * 15167485180850043744, tmp_4796[1] * 15167485180850043744, tmp_4796[2] * 15167485180850043744];
    signal tmp_4837[3] <== [tmp_4836[0] + tmp_4797[0], tmp_4836[1] + tmp_4797[1], tmp_4836[2] + tmp_4797[2]];
    signal tmp_4838[3] <== [tmp_4835[0] + tmp_4837[0], tmp_4835[1] + tmp_4837[1], tmp_4835[2] + tmp_4837[2]];
    signal tmp_4839[3] <== [tmp_4805[0] + tmp_4838[0], tmp_4805[1] + tmp_4838[1], tmp_4805[2] + tmp_4838[2]];
    signal tmp_4840[3] <== [evals[92][0] - tmp_4839[0], evals[92][1] - tmp_4839[1], evals[92][2] - tmp_4839[2]];
    signal tmp_4841[3] <== CMul()(evals[51], tmp_4840);
    signal tmp_4842[3] <== [tmp_4802[0] + tmp_4841[0], tmp_4802[1] + tmp_4841[1], tmp_4802[2] + tmp_4841[2]];
    signal tmp_4843[3] <== CMul()(challengeQ, tmp_4842);
    signal tmp_4844[3] <== [evals[92][0] + tmp_3886[0], evals[92][1] + tmp_3886[1], evals[92][2] + tmp_3886[2]];
    signal tmp_4845[3] <== CMul()(tmp_3891, tmp_4844);
    signal tmp_4846[3] <== [tmp_4845[0] * 14102670999874605824, tmp_4845[1] * 14102670999874605824, tmp_4845[2] * 14102670999874605824];
    signal tmp_4847[3] <== [tmp_4807[0] * 15585654191999307702, tmp_4807[1] * 15585654191999307702, tmp_4807[2] * 15585654191999307702];
    signal tmp_4848[3] <== [tmp_4847[0] + tmp_4838[0], tmp_4847[1] + tmp_4838[1], tmp_4847[2] + tmp_4838[2]];
    signal tmp_4849[3] <== [tmp_4845[0] + tmp_4848[0], tmp_4845[1] + tmp_4848[1], tmp_4845[2] + tmp_4848[2]];
    signal tmp_4850[3] <== [tmp_4810[0] * 940187017142450255, tmp_4810[1] * 940187017142450255, tmp_4810[2] * 940187017142450255];
    signal tmp_4851[3] <== [tmp_4850[0] + tmp_4838[0], tmp_4850[1] + tmp_4838[1], tmp_4850[2] + tmp_4838[2]];
    signal tmp_4852[3] <== [tmp_4849[0] + tmp_4851[0], tmp_4849[1] + tmp_4851[1], tmp_4849[2] + tmp_4851[2]];
    signal tmp_4853[3] <== [tmp_4813[0] * 8747386241522630711, tmp_4813[1] * 8747386241522630711, tmp_4813[2] * 8747386241522630711];
    signal tmp_4854[3] <== [tmp_4853[0] + tmp_4838[0], tmp_4853[1] + tmp_4838[1], tmp_4853[2] + tmp_4838[2]];
    signal tmp_4855[3] <== [tmp_4852[0] + tmp_4854[0], tmp_4852[1] + tmp_4854[1], tmp_4852[2] + tmp_4854[2]];
    signal tmp_4856[3] <== [tmp_4816[0] * 6750641561540124747, tmp_4816[1] * 6750641561540124747, tmp_4816[2] * 6750641561540124747];
    signal tmp_4857[3] <== [tmp_4856[0] + tmp_4838[0], tmp_4856[1] + tmp_4838[1], tmp_4856[2] + tmp_4838[2]];
    signal tmp_4858[3] <== [tmp_4855[0] + tmp_4857[0], tmp_4855[1] + tmp_4857[1], tmp_4855[2] + tmp_4857[2]];
    signal tmp_4859[3] <== [tmp_4819[0] * 7440998025584530007, tmp_4819[1] * 7440998025584530007, tmp_4819[2] * 7440998025584530007];
    signal tmp_4860[3] <== [tmp_4859[0] + tmp_4838[0], tmp_4859[1] + tmp_4838[1], tmp_4859[2] + tmp_4838[2]];
    signal tmp_4861[3] <== [tmp_4858[0] + tmp_4860[0], tmp_4858[1] + tmp_4860[1], tmp_4858[2] + tmp_4860[2]];
    signal tmp_4862[3] <== [tmp_4822[0] * 6136358134615751536, tmp_4822[1] * 6136358134615751536, tmp_4822[2] * 6136358134615751536];
    signal tmp_4863[3] <== [tmp_4862[0] + tmp_4838[0], tmp_4862[1] + tmp_4838[1], tmp_4862[2] + tmp_4838[2]];
    signal tmp_4864[3] <== [tmp_4861[0] + tmp_4863[0], tmp_4861[1] + tmp_4863[1], tmp_4861[2] + tmp_4863[2]];
    signal tmp_4865[3] <== [tmp_4825[0] * 12413576830284969611, tmp_4825[1] * 12413576830284969611, tmp_4825[2] * 12413576830284969611];
    signal tmp_4866[3] <== [tmp_4865[0] + tmp_4838[0], tmp_4865[1] + tmp_4838[1], tmp_4865[2] + tmp_4838[2]];
    signal tmp_4867[3] <== [tmp_4864[0] + tmp_4866[0], tmp_4864[1] + tmp_4866[1], tmp_4864[2] + tmp_4866[2]];
    signal tmp_4868[3] <== [tmp_4828[0] * 11675438539028694709, tmp_4828[1] * 11675438539028694709, tmp_4828[2] * 11675438539028694709];
    signal tmp_4869[3] <== [tmp_4868[0] + tmp_4838[0], tmp_4868[1] + tmp_4838[1], tmp_4868[2] + tmp_4838[2]];
    signal tmp_4870[3] <== [tmp_4867[0] + tmp_4869[0], tmp_4867[1] + tmp_4869[1], tmp_4867[2] + tmp_4869[2]];
    signal tmp_4871[3] <== [tmp_4831[0] * 17580553691069642926, tmp_4831[1] * 17580553691069642926, tmp_4831[2] * 17580553691069642926];
    signal tmp_4872[3] <== [tmp_4871[0] + tmp_4838[0], tmp_4871[1] + tmp_4838[1], tmp_4871[2] + tmp_4838[2]];
    signal tmp_4873[3] <== [tmp_4870[0] + tmp_4872[0], tmp_4870[1] + tmp_4872[1], tmp_4870[2] + tmp_4872[2]];
    signal tmp_4874[3] <== [tmp_4834[0] * 892707462476851331, tmp_4834[1] * 892707462476851331, tmp_4834[2] * 892707462476851331];
    signal tmp_4875[3] <== [tmp_4874[0] + tmp_4838[0], tmp_4874[1] + tmp_4838[1], tmp_4874[2] + tmp_4838[2]];
    signal tmp_4876[3] <== [tmp_4873[0] + tmp_4875[0], tmp_4873[1] + tmp_4875[1], tmp_4873[2] + tmp_4875[2]];
    signal tmp_4877[3] <== [tmp_4837[0] * 15167485180850043744, tmp_4837[1] * 15167485180850043744, tmp_4837[2] * 15167485180850043744];
    signal tmp_4878[3] <== [tmp_4877[0] + tmp_4838[0], tmp_4877[1] + tmp_4838[1], tmp_4877[2] + tmp_4838[2]];
    signal tmp_4879[3] <== [tmp_4876[0] + tmp_4878[0], tmp_4876[1] + tmp_4878[1], tmp_4876[2] + tmp_4878[2]];
    signal tmp_4880[3] <== [tmp_4846[0] + tmp_4879[0], tmp_4846[1] + tmp_4879[1], tmp_4846[2] + tmp_4879[2]];
    signal tmp_4881[3] <== [evals[93][0] - tmp_4880[0], evals[93][1] - tmp_4880[1], evals[93][2] - tmp_4880[2]];
    signal tmp_4882[3] <== CMul()(evals[51], tmp_4881);
    signal tmp_4883[3] <== [tmp_4843[0] + tmp_4882[0], tmp_4843[1] + tmp_4882[1], tmp_4843[2] + tmp_4882[2]];
    signal tmp_4884[3] <== CMul()(challengeQ, tmp_4883);
    signal tmp_4885[3] <== [evals[93][0] + tmp_3902[0], evals[93][1] + tmp_3902[1], evals[93][2] + tmp_3902[2]];
    signal tmp_4886[3] <== CMul()(tmp_3907, tmp_4885);
    signal tmp_4887[3] <== [tmp_4886[0] * 14102670999874605824, tmp_4886[1] * 14102670999874605824, tmp_4886[2] * 14102670999874605824];
    signal tmp_4888[3] <== [tmp_4848[0] * 15585654191999307702, tmp_4848[1] * 15585654191999307702, tmp_4848[2] * 15585654191999307702];
    signal tmp_4889[3] <== [tmp_4888[0] + tmp_4879[0], tmp_4888[1] + tmp_4879[1], tmp_4888[2] + tmp_4879[2]];
    signal tmp_4890[3] <== [tmp_4886[0] + tmp_4889[0], tmp_4886[1] + tmp_4889[1], tmp_4886[2] + tmp_4889[2]];
    signal tmp_4891[3] <== [tmp_4851[0] * 940187017142450255, tmp_4851[1] * 940187017142450255, tmp_4851[2] * 940187017142450255];
    signal tmp_4892[3] <== [tmp_4891[0] + tmp_4879[0], tmp_4891[1] + tmp_4879[1], tmp_4891[2] + tmp_4879[2]];
    signal tmp_4893[3] <== [tmp_4890[0] + tmp_4892[0], tmp_4890[1] + tmp_4892[1], tmp_4890[2] + tmp_4892[2]];
    signal tmp_4894[3] <== [tmp_4854[0] * 8747386241522630711, tmp_4854[1] * 8747386241522630711, tmp_4854[2] * 8747386241522630711];
    signal tmp_4895[3] <== [tmp_4894[0] + tmp_4879[0], tmp_4894[1] + tmp_4879[1], tmp_4894[2] + tmp_4879[2]];
    signal tmp_4896[3] <== [tmp_4893[0] + tmp_4895[0], tmp_4893[1] + tmp_4895[1], tmp_4893[2] + tmp_4895[2]];
    signal tmp_4897[3] <== [tmp_4857[0] * 6750641561540124747, tmp_4857[1] * 6750641561540124747, tmp_4857[2] * 6750641561540124747];
    signal tmp_4898[3] <== [tmp_4897[0] + tmp_4879[0], tmp_4897[1] + tmp_4879[1], tmp_4897[2] + tmp_4879[2]];
    signal tmp_4899[3] <== [tmp_4896[0] + tmp_4898[0], tmp_4896[1] + tmp_4898[1], tmp_4896[2] + tmp_4898[2]];
    signal tmp_4900[3] <== [tmp_4860[0] * 7440998025584530007, tmp_4860[1] * 7440998025584530007, tmp_4860[2] * 7440998025584530007];
    signal tmp_4901[3] <== [tmp_4900[0] + tmp_4879[0], tmp_4900[1] + tmp_4879[1], tmp_4900[2] + tmp_4879[2]];
    signal tmp_4902[3] <== [tmp_4899[0] + tmp_4901[0], tmp_4899[1] + tmp_4901[1], tmp_4899[2] + tmp_4901[2]];
    signal tmp_4903[3] <== [tmp_4863[0] * 6136358134615751536, tmp_4863[1] * 6136358134615751536, tmp_4863[2] * 6136358134615751536];
    signal tmp_4904[3] <== [tmp_4903[0] + tmp_4879[0], tmp_4903[1] + tmp_4879[1], tmp_4903[2] + tmp_4879[2]];
    signal tmp_4905[3] <== [tmp_4902[0] + tmp_4904[0], tmp_4902[1] + tmp_4904[1], tmp_4902[2] + tmp_4904[2]];
    signal tmp_4906[3] <== [tmp_4866[0] * 12413576830284969611, tmp_4866[1] * 12413576830284969611, tmp_4866[2] * 12413576830284969611];
    signal tmp_4907[3] <== [tmp_4906[0] + tmp_4879[0], tmp_4906[1] + tmp_4879[1], tmp_4906[2] + tmp_4879[2]];
    signal tmp_4908[3] <== [tmp_4905[0] + tmp_4907[0], tmp_4905[1] + tmp_4907[1], tmp_4905[2] + tmp_4907[2]];
    signal tmp_4909[3] <== [tmp_4869[0] * 11675438539028694709, tmp_4869[1] * 11675438539028694709, tmp_4869[2] * 11675438539028694709];
    signal tmp_4910[3] <== [tmp_4909[0] + tmp_4879[0], tmp_4909[1] + tmp_4879[1], tmp_4909[2] + tmp_4879[2]];
    signal tmp_4911[3] <== [tmp_4908[0] + tmp_4910[0], tmp_4908[1] + tmp_4910[1], tmp_4908[2] + tmp_4910[2]];
    signal tmp_4912[3] <== [tmp_4872[0] * 17580553691069642926, tmp_4872[1] * 17580553691069642926, tmp_4872[2] * 17580553691069642926];
    signal tmp_4913[3] <== [tmp_4912[0] + tmp_4879[0], tmp_4912[1] + tmp_4879[1], tmp_4912[2] + tmp_4879[2]];
    signal tmp_4914[3] <== [tmp_4911[0] + tmp_4913[0], tmp_4911[1] + tmp_4913[1], tmp_4911[2] + tmp_4913[2]];
    signal tmp_4915[3] <== [tmp_4875[0] * 892707462476851331, tmp_4875[1] * 892707462476851331, tmp_4875[2] * 892707462476851331];
    signal tmp_4916[3] <== [tmp_4915[0] + tmp_4879[0], tmp_4915[1] + tmp_4879[1], tmp_4915[2] + tmp_4879[2]];
    signal tmp_4917[3] <== [tmp_4914[0] + tmp_4916[0], tmp_4914[1] + tmp_4916[1], tmp_4914[2] + tmp_4916[2]];
    signal tmp_4918[3] <== [tmp_4878[0] * 15167485180850043744, tmp_4878[1] * 15167485180850043744, tmp_4878[2] * 15167485180850043744];
    signal tmp_4919[3] <== [tmp_4918[0] + tmp_4879[0], tmp_4918[1] + tmp_4879[1], tmp_4918[2] + tmp_4879[2]];
    signal tmp_4920[3] <== [tmp_4917[0] + tmp_4919[0], tmp_4917[1] + tmp_4919[1], tmp_4917[2] + tmp_4919[2]];
    signal tmp_4921[3] <== [tmp_4887[0] + tmp_4920[0], tmp_4887[1] + tmp_4920[1], tmp_4887[2] + tmp_4920[2]];
    signal tmp_4922[3] <== [evals[94][0] - tmp_4921[0], evals[94][1] - tmp_4921[1], evals[94][2] - tmp_4921[2]];
    signal tmp_4923[3] <== CMul()(evals[51], tmp_4922);
    signal tmp_4924[3] <== [tmp_4884[0] + tmp_4923[0], tmp_4884[1] + tmp_4923[1], tmp_4884[2] + tmp_4923[2]];
    signal tmp_4925[3] <== CMul()(challengeQ, tmp_4924);
    signal tmp_4926[3] <== [evals[94][0] + tmp_3924[0], evals[94][1] + tmp_3924[1], evals[94][2] + tmp_3924[2]];
    signal tmp_4927[3] <== CMul()(tmp_3929, tmp_4926);
    signal tmp_4928[3] <== [tmp_4927[0] * 14102670999874605824, tmp_4927[1] * 14102670999874605824, tmp_4927[2] * 14102670999874605824];
    signal tmp_4929[3] <== [tmp_4889[0] * 15585654191999307702, tmp_4889[1] * 15585654191999307702, tmp_4889[2] * 15585654191999307702];
    signal tmp_4930[3] <== [tmp_4929[0] + tmp_4920[0], tmp_4929[1] + tmp_4920[1], tmp_4929[2] + tmp_4920[2]];
    signal tmp_4931[3] <== [tmp_4927[0] + tmp_4930[0], tmp_4927[1] + tmp_4930[1], tmp_4927[2] + tmp_4930[2]];
    signal tmp_4932[3] <== [tmp_4892[0] * 940187017142450255, tmp_4892[1] * 940187017142450255, tmp_4892[2] * 940187017142450255];
    signal tmp_4933[3] <== [tmp_4932[0] + tmp_4920[0], tmp_4932[1] + tmp_4920[1], tmp_4932[2] + tmp_4920[2]];
    signal tmp_4934[3] <== [tmp_4931[0] + tmp_4933[0], tmp_4931[1] + tmp_4933[1], tmp_4931[2] + tmp_4933[2]];
    signal tmp_4935[3] <== [tmp_4895[0] * 8747386241522630711, tmp_4895[1] * 8747386241522630711, tmp_4895[2] * 8747386241522630711];
    signal tmp_4936[3] <== [tmp_4935[0] + tmp_4920[0], tmp_4935[1] + tmp_4920[1], tmp_4935[2] + tmp_4920[2]];
    signal tmp_4937[3] <== [tmp_4934[0] + tmp_4936[0], tmp_4934[1] + tmp_4936[1], tmp_4934[2] + tmp_4936[2]];
    signal tmp_4938[3] <== [tmp_4898[0] * 6750641561540124747, tmp_4898[1] * 6750641561540124747, tmp_4898[2] * 6750641561540124747];
    signal tmp_4939[3] <== [tmp_4938[0] + tmp_4920[0], tmp_4938[1] + tmp_4920[1], tmp_4938[2] + tmp_4920[2]];
    signal tmp_4940[3] <== [tmp_4937[0] + tmp_4939[0], tmp_4937[1] + tmp_4939[1], tmp_4937[2] + tmp_4939[2]];
    signal tmp_4941[3] <== [tmp_4901[0] * 7440998025584530007, tmp_4901[1] * 7440998025584530007, tmp_4901[2] * 7440998025584530007];
    signal tmp_4942[3] <== [tmp_4941[0] + tmp_4920[0], tmp_4941[1] + tmp_4920[1], tmp_4941[2] + tmp_4920[2]];
    signal tmp_4943[3] <== [tmp_4940[0] + tmp_4942[0], tmp_4940[1] + tmp_4942[1], tmp_4940[2] + tmp_4942[2]];
    signal tmp_4944[3] <== [tmp_4904[0] * 6136358134615751536, tmp_4904[1] * 6136358134615751536, tmp_4904[2] * 6136358134615751536];
    signal tmp_4945[3] <== [tmp_4944[0] + tmp_4920[0], tmp_4944[1] + tmp_4920[1], tmp_4944[2] + tmp_4920[2]];
    signal tmp_4946[3] <== [tmp_4943[0] + tmp_4945[0], tmp_4943[1] + tmp_4945[1], tmp_4943[2] + tmp_4945[2]];
    signal tmp_4947[3] <== [tmp_4907[0] * 12413576830284969611, tmp_4907[1] * 12413576830284969611, tmp_4907[2] * 12413576830284969611];
    signal tmp_4948[3] <== [tmp_4947[0] + tmp_4920[0], tmp_4947[1] + tmp_4920[1], tmp_4947[2] + tmp_4920[2]];
    signal tmp_4949[3] <== [tmp_4946[0] + tmp_4948[0], tmp_4946[1] + tmp_4948[1], tmp_4946[2] + tmp_4948[2]];
    signal tmp_4950[3] <== [tmp_4910[0] * 11675438539028694709, tmp_4910[1] * 11675438539028694709, tmp_4910[2] * 11675438539028694709];
    signal tmp_4951[3] <== [tmp_4950[0] + tmp_4920[0], tmp_4950[1] + tmp_4920[1], tmp_4950[2] + tmp_4920[2]];
    signal tmp_4952[3] <== [tmp_4949[0] + tmp_4951[0], tmp_4949[1] + tmp_4951[1], tmp_4949[2] + tmp_4951[2]];
    signal tmp_4953[3] <== [tmp_4913[0] * 17580553691069642926, tmp_4913[1] * 17580553691069642926, tmp_4913[2] * 17580553691069642926];
    signal tmp_4954[3] <== [tmp_4953[0] + tmp_4920[0], tmp_4953[1] + tmp_4920[1], tmp_4953[2] + tmp_4920[2]];
    signal tmp_4955[3] <== [tmp_4952[0] + tmp_4954[0], tmp_4952[1] + tmp_4954[1], tmp_4952[2] + tmp_4954[2]];
    signal tmp_4956[3] <== [tmp_4916[0] * 892707462476851331, tmp_4916[1] * 892707462476851331, tmp_4916[2] * 892707462476851331];
    signal tmp_4957[3] <== [tmp_4956[0] + tmp_4920[0], tmp_4956[1] + tmp_4920[1], tmp_4956[2] + tmp_4920[2]];
    signal tmp_4958[3] <== [tmp_4955[0] + tmp_4957[0], tmp_4955[1] + tmp_4957[1], tmp_4955[2] + tmp_4957[2]];
    signal tmp_4959[3] <== [tmp_4919[0] * 15167485180850043744, tmp_4919[1] * 15167485180850043744, tmp_4919[2] * 15167485180850043744];
    signal tmp_4960[3] <== [tmp_4959[0] + tmp_4920[0], tmp_4959[1] + tmp_4920[1], tmp_4959[2] + tmp_4920[2]];
    signal tmp_4961[3] <== [tmp_4958[0] + tmp_4960[0], tmp_4958[1] + tmp_4960[1], tmp_4958[2] + tmp_4960[2]];
    signal tmp_4962[3] <== [tmp_4928[0] + tmp_4961[0], tmp_4928[1] + tmp_4961[1], tmp_4928[2] + tmp_4961[2]];
    signal tmp_4963[3] <== [evals[95][0] - tmp_4962[0], evals[95][1] - tmp_4962[1], evals[95][2] - tmp_4962[2]];
    signal tmp_4964[3] <== CMul()(evals[51], tmp_4963);
    signal tmp_4965[3] <== [tmp_4925[0] + tmp_4964[0], tmp_4925[1] + tmp_4964[1], tmp_4925[2] + tmp_4964[2]];
    signal tmp_4966[3] <== CMul()(challengeQ, tmp_4965);
    signal tmp_4967[3] <== [evals[95][0] + tmp_3869[0], evals[95][1] + tmp_3869[1], evals[95][2] + tmp_3869[2]];
    signal tmp_4968[3] <== CMul()(tmp_3874, tmp_4967);
    signal tmp_4969[3] <== [tmp_4968[0] * 14102670999874605824, tmp_4968[1] * 14102670999874605824, tmp_4968[2] * 14102670999874605824];
    signal tmp_4970[3] <== [tmp_4930[0] * 15585654191999307702, tmp_4930[1] * 15585654191999307702, tmp_4930[2] * 15585654191999307702];
    signal tmp_4971[3] <== [tmp_4970[0] + tmp_4961[0], tmp_4970[1] + tmp_4961[1], tmp_4970[2] + tmp_4961[2]];
    signal tmp_4972[3] <== [tmp_4968[0] + tmp_4971[0], tmp_4968[1] + tmp_4971[1], tmp_4968[2] + tmp_4971[2]];
    signal tmp_4973[3] <== [tmp_4933[0] * 940187017142450255, tmp_4933[1] * 940187017142450255, tmp_4933[2] * 940187017142450255];
    signal tmp_4974[3] <== [tmp_4973[0] + tmp_4961[0], tmp_4973[1] + tmp_4961[1], tmp_4973[2] + tmp_4961[2]];
    signal tmp_4975[3] <== [tmp_4972[0] + tmp_4974[0], tmp_4972[1] + tmp_4974[1], tmp_4972[2] + tmp_4974[2]];
    signal tmp_4976[3] <== [tmp_4936[0] * 8747386241522630711, tmp_4936[1] * 8747386241522630711, tmp_4936[2] * 8747386241522630711];
    signal tmp_4977[3] <== [tmp_4976[0] + tmp_4961[0], tmp_4976[1] + tmp_4961[1], tmp_4976[2] + tmp_4961[2]];
    signal tmp_4978[3] <== [tmp_4975[0] + tmp_4977[0], tmp_4975[1] + tmp_4977[1], tmp_4975[2] + tmp_4977[2]];
    signal tmp_4979[3] <== [tmp_4939[0] * 6750641561540124747, tmp_4939[1] * 6750641561540124747, tmp_4939[2] * 6750641561540124747];
    signal tmp_4980[3] <== [tmp_4979[0] + tmp_4961[0], tmp_4979[1] + tmp_4961[1], tmp_4979[2] + tmp_4961[2]];
    signal tmp_4981[3] <== [tmp_4978[0] + tmp_4980[0], tmp_4978[1] + tmp_4980[1], tmp_4978[2] + tmp_4980[2]];
    signal tmp_4982[3] <== [tmp_4942[0] * 7440998025584530007, tmp_4942[1] * 7440998025584530007, tmp_4942[2] * 7440998025584530007];
    signal tmp_4983[3] <== [tmp_4982[0] + tmp_4961[0], tmp_4982[1] + tmp_4961[1], tmp_4982[2] + tmp_4961[2]];
    signal tmp_4984[3] <== [tmp_4981[0] + tmp_4983[0], tmp_4981[1] + tmp_4983[1], tmp_4981[2] + tmp_4983[2]];
    signal tmp_4985[3] <== [tmp_4945[0] * 6136358134615751536, tmp_4945[1] * 6136358134615751536, tmp_4945[2] * 6136358134615751536];
    signal tmp_4986[3] <== [tmp_4985[0] + tmp_4961[0], tmp_4985[1] + tmp_4961[1], tmp_4985[2] + tmp_4961[2]];
    signal tmp_4987[3] <== [tmp_4984[0] + tmp_4986[0], tmp_4984[1] + tmp_4986[1], tmp_4984[2] + tmp_4986[2]];
    signal tmp_4988[3] <== [tmp_4948[0] * 12413576830284969611, tmp_4948[1] * 12413576830284969611, tmp_4948[2] * 12413576830284969611];
    signal tmp_4989[3] <== [tmp_4988[0] + tmp_4961[0], tmp_4988[1] + tmp_4961[1], tmp_4988[2] + tmp_4961[2]];
    signal tmp_4990[3] <== [tmp_4987[0] + tmp_4989[0], tmp_4987[1] + tmp_4989[1], tmp_4987[2] + tmp_4989[2]];
    signal tmp_4991[3] <== [tmp_4951[0] * 11675438539028694709, tmp_4951[1] * 11675438539028694709, tmp_4951[2] * 11675438539028694709];
    signal tmp_4992[3] <== [tmp_4991[0] + tmp_4961[0], tmp_4991[1] + tmp_4961[1], tmp_4991[2] + tmp_4961[2]];
    signal tmp_4993[3] <== [tmp_4990[0] + tmp_4992[0], tmp_4990[1] + tmp_4992[1], tmp_4990[2] + tmp_4992[2]];
    signal tmp_4994[3] <== [tmp_4954[0] * 17580553691069642926, tmp_4954[1] * 17580553691069642926, tmp_4954[2] * 17580553691069642926];
    signal tmp_4995[3] <== [tmp_4994[0] + tmp_4961[0], tmp_4994[1] + tmp_4961[1], tmp_4994[2] + tmp_4961[2]];
    signal tmp_4996[3] <== [tmp_4993[0] + tmp_4995[0], tmp_4993[1] + tmp_4995[1], tmp_4993[2] + tmp_4995[2]];
    signal tmp_4997[3] <== [tmp_4957[0] * 892707462476851331, tmp_4957[1] * 892707462476851331, tmp_4957[2] * 892707462476851331];
    signal tmp_4998[3] <== [tmp_4997[0] + tmp_4961[0], tmp_4997[1] + tmp_4961[1], tmp_4997[2] + tmp_4961[2]];
    signal tmp_4999[3] <== [tmp_4996[0] + tmp_4998[0], tmp_4996[1] + tmp_4998[1], tmp_4996[2] + tmp_4998[2]];
    signal tmp_5000[3] <== [tmp_4960[0] * 15167485180850043744, tmp_4960[1] * 15167485180850043744, tmp_4960[2] * 15167485180850043744];
    signal tmp_5001[3] <== [tmp_5000[0] + tmp_4961[0], tmp_5000[1] + tmp_4961[1], tmp_5000[2] + tmp_4961[2]];
    signal tmp_5002[3] <== [tmp_4999[0] + tmp_5001[0], tmp_4999[1] + tmp_5001[1], tmp_4999[2] + tmp_5001[2]];
    signal tmp_5003[3] <== [tmp_4969[0] + tmp_5002[0], tmp_4969[1] + tmp_5002[1], tmp_4969[2] + tmp_5002[2]];
    signal tmp_5004[3] <== [evals[110][0] - tmp_5003[0], evals[110][1] - tmp_5003[1], evals[110][2] - tmp_5003[2]];
    signal tmp_5005[3] <== CMul()(evals[51], tmp_5004);
    signal tmp_5006[3] <== [tmp_4966[0] + tmp_5005[0], tmp_4966[1] + tmp_5005[1], tmp_4966[2] + tmp_5005[2]];
    signal tmp_5007[3] <== CMul()(challengeQ, tmp_5006);
    signal tmp_5008[3] <== [tmp_4971[0] * 15585654191999307702, tmp_4971[1] * 15585654191999307702, tmp_4971[2] * 15585654191999307702];
    signal tmp_5009[3] <== [tmp_5008[0] + tmp_5002[0], tmp_5008[1] + tmp_5002[1], tmp_5008[2] + tmp_5002[2]];
    signal tmp_5010[3] <== [evals[111][0] - tmp_5009[0], evals[111][1] - tmp_5009[1], evals[111][2] - tmp_5009[2]];
    signal tmp_5011[3] <== CMul()(evals[51], tmp_5010);
    signal tmp_5012[3] <== [tmp_5007[0] + tmp_5011[0], tmp_5007[1] + tmp_5011[1], tmp_5007[2] + tmp_5011[2]];
    signal tmp_5013[3] <== CMul()(challengeQ, tmp_5012);
    signal tmp_5014[3] <== [tmp_4974[0] * 940187017142450255, tmp_4974[1] * 940187017142450255, tmp_4974[2] * 940187017142450255];
    signal tmp_5015[3] <== [tmp_5014[0] + tmp_5002[0], tmp_5014[1] + tmp_5002[1], tmp_5014[2] + tmp_5002[2]];
    signal tmp_5016[3] <== [evals[112][0] - tmp_5015[0], evals[112][1] - tmp_5015[1], evals[112][2] - tmp_5015[2]];
    signal tmp_5017[3] <== CMul()(evals[51], tmp_5016);
    signal tmp_5018[3] <== [tmp_5013[0] + tmp_5017[0], tmp_5013[1] + tmp_5017[1], tmp_5013[2] + tmp_5017[2]];
    signal tmp_5019[3] <== CMul()(challengeQ, tmp_5018);
    signal tmp_5020[3] <== [tmp_4977[0] * 8747386241522630711, tmp_4977[1] * 8747386241522630711, tmp_4977[2] * 8747386241522630711];
    signal tmp_5021[3] <== [tmp_5020[0] + tmp_5002[0], tmp_5020[1] + tmp_5002[1], tmp_5020[2] + tmp_5002[2]];
    signal tmp_5022[3] <== [evals[113][0] - tmp_5021[0], evals[113][1] - tmp_5021[1], evals[113][2] - tmp_5021[2]];
    signal tmp_5023[3] <== CMul()(evals[51], tmp_5022);
    signal tmp_5024[3] <== [tmp_5019[0] + tmp_5023[0], tmp_5019[1] + tmp_5023[1], tmp_5019[2] + tmp_5023[2]];
    signal tmp_5025[3] <== CMul()(challengeQ, tmp_5024);
    signal tmp_5026[3] <== [tmp_4980[0] * 6750641561540124747, tmp_4980[1] * 6750641561540124747, tmp_4980[2] * 6750641561540124747];
    signal tmp_5027[3] <== [tmp_5026[0] + tmp_5002[0], tmp_5026[1] + tmp_5002[1], tmp_5026[2] + tmp_5002[2]];
    signal tmp_5028[3] <== [evals[114][0] - tmp_5027[0], evals[114][1] - tmp_5027[1], evals[114][2] - tmp_5027[2]];
    signal tmp_5029[3] <== CMul()(evals[51], tmp_5028);
    signal tmp_5030[3] <== [tmp_5025[0] + tmp_5029[0], tmp_5025[1] + tmp_5029[1], tmp_5025[2] + tmp_5029[2]];
    signal tmp_5031[3] <== CMul()(challengeQ, tmp_5030);
    signal tmp_5032[3] <== [tmp_4983[0] * 7440998025584530007, tmp_4983[1] * 7440998025584530007, tmp_4983[2] * 7440998025584530007];
    signal tmp_5033[3] <== [tmp_5032[0] + tmp_5002[0], tmp_5032[1] + tmp_5002[1], tmp_5032[2] + tmp_5002[2]];
    signal tmp_5034[3] <== [evals[115][0] - tmp_5033[0], evals[115][1] - tmp_5033[1], evals[115][2] - tmp_5033[2]];
    signal tmp_5035[3] <== CMul()(evals[51], tmp_5034);
    signal tmp_5036[3] <== [tmp_5031[0] + tmp_5035[0], tmp_5031[1] + tmp_5035[1], tmp_5031[2] + tmp_5035[2]];
    signal tmp_5037[3] <== CMul()(challengeQ, tmp_5036);
    signal tmp_5038[3] <== [tmp_4986[0] * 6136358134615751536, tmp_4986[1] * 6136358134615751536, tmp_4986[2] * 6136358134615751536];
    signal tmp_5039[3] <== [tmp_5038[0] + tmp_5002[0], tmp_5038[1] + tmp_5002[1], tmp_5038[2] + tmp_5002[2]];
    signal tmp_5040[3] <== [evals[116][0] - tmp_5039[0], evals[116][1] - tmp_5039[1], evals[116][2] - tmp_5039[2]];
    signal tmp_5041[3] <== CMul()(evals[51], tmp_5040);
    signal tmp_5042[3] <== [tmp_5037[0] + tmp_5041[0], tmp_5037[1] + tmp_5041[1], tmp_5037[2] + tmp_5041[2]];
    signal tmp_5043[3] <== CMul()(challengeQ, tmp_5042);
    signal tmp_5044[3] <== [tmp_4989[0] * 12413576830284969611, tmp_4989[1] * 12413576830284969611, tmp_4989[2] * 12413576830284969611];
    signal tmp_5045[3] <== [tmp_5044[0] + tmp_5002[0], tmp_5044[1] + tmp_5002[1], tmp_5044[2] + tmp_5002[2]];
    signal tmp_5046[3] <== [evals[117][0] - tmp_5045[0], evals[117][1] - tmp_5045[1], evals[117][2] - tmp_5045[2]];
    signal tmp_5047[3] <== CMul()(evals[51], tmp_5046);
    signal tmp_5048[3] <== [tmp_5043[0] + tmp_5047[0], tmp_5043[1] + tmp_5047[1], tmp_5043[2] + tmp_5047[2]];
    signal tmp_5049[3] <== CMul()(challengeQ, tmp_5048);
    signal tmp_5050[3] <== [tmp_4992[0] * 11675438539028694709, tmp_4992[1] * 11675438539028694709, tmp_4992[2] * 11675438539028694709];
    signal tmp_5051[3] <== [tmp_5050[0] + tmp_5002[0], tmp_5050[1] + tmp_5002[1], tmp_5050[2] + tmp_5002[2]];
    signal tmp_5052[3] <== [evals[118][0] - tmp_5051[0], evals[118][1] - tmp_5051[1], evals[118][2] - tmp_5051[2]];
    signal tmp_5053[3] <== CMul()(evals[51], tmp_5052);
    signal tmp_5054[3] <== [tmp_5049[0] + tmp_5053[0], tmp_5049[1] + tmp_5053[1], tmp_5049[2] + tmp_5053[2]];
    signal tmp_5055[3] <== CMul()(challengeQ, tmp_5054);
    signal tmp_5056[3] <== [tmp_4995[0] * 17580553691069642926, tmp_4995[1] * 17580553691069642926, tmp_4995[2] * 17580553691069642926];
    signal tmp_5057[3] <== [tmp_5056[0] + tmp_5002[0], tmp_5056[1] + tmp_5002[1], tmp_5056[2] + tmp_5002[2]];
    signal tmp_5058[3] <== [evals[119][0] - tmp_5057[0], evals[119][1] - tmp_5057[1], evals[119][2] - tmp_5057[2]];
    signal tmp_5059[3] <== CMul()(evals[51], tmp_5058);
    signal tmp_5060[3] <== [tmp_5055[0] + tmp_5059[0], tmp_5055[1] + tmp_5059[1], tmp_5055[2] + tmp_5059[2]];
    signal tmp_5061[3] <== CMul()(challengeQ, tmp_5060);
    signal tmp_5062[3] <== [tmp_4998[0] * 892707462476851331, tmp_4998[1] * 892707462476851331, tmp_4998[2] * 892707462476851331];
    signal tmp_5063[3] <== [tmp_5062[0] + tmp_5002[0], tmp_5062[1] + tmp_5002[1], tmp_5062[2] + tmp_5002[2]];
    signal tmp_5064[3] <== [evals[120][0] - tmp_5063[0], evals[120][1] - tmp_5063[1], evals[120][2] - tmp_5063[2]];
    signal tmp_5065[3] <== CMul()(evals[51], tmp_5064);
    signal tmp_5066[3] <== [tmp_5061[0] + tmp_5065[0], tmp_5061[1] + tmp_5065[1], tmp_5061[2] + tmp_5065[2]];
    signal tmp_5067[3] <== CMul()(challengeQ, tmp_5066);
    signal tmp_5068[3] <== [tmp_5001[0] * 15167485180850043744, tmp_5001[1] * 15167485180850043744, tmp_5001[2] * 15167485180850043744];
    signal tmp_5069[3] <== [tmp_5068[0] + tmp_5002[0], tmp_5068[1] + tmp_5002[1], tmp_5068[2] + tmp_5002[2]];
    signal tmp_5070[3] <== [evals[121][0] - tmp_5069[0], evals[121][1] - tmp_5069[1], evals[121][2] - tmp_5069[2]];
    signal tmp_5071[3] <== CMul()(evals[51], tmp_5070);
    signal tmp_5072[3] <== [tmp_5067[0] + tmp_5071[0], tmp_5067[1] + tmp_5071[1], tmp_5067[2] + tmp_5071[2]];
    signal tmp_5073[3] <== CMul()(challengeQ, tmp_5072);
    signal tmp_5074[3] <== CMul()(evals[60], evals[63]);
    signal tmp_5075[3] <== CMul()(evals[61], evals[65]);
    signal tmp_5076[3] <== [tmp_5074[0] + tmp_5075[0], tmp_5074[1] + tmp_5075[1], tmp_5074[2] + tmp_5075[2]];
    signal tmp_5077[3] <== CMul()(evals[62], evals[64]);
    signal tmp_5078[3] <== [tmp_5076[0] + tmp_5077[0], tmp_5076[1] + tmp_5077[1], tmp_5076[2] + tmp_5077[2]];
    signal tmp_5079[3] <== [evals[66][0] - tmp_5078[0], evals[66][1] - tmp_5078[1], evals[66][2] - tmp_5078[2]];
    signal tmp_5080[3] <== CMul()(evals[53], tmp_5079);
    signal tmp_5081[3] <== [tmp_5073[0] + tmp_5080[0], tmp_5073[1] + tmp_5080[1], tmp_5073[2] + tmp_5080[2]];
    signal tmp_5082[3] <== CMul()(challengeQ, tmp_5081);
    signal tmp_5083[3] <== CMul()(evals[60], evals[64]);
    signal tmp_5084[3] <== CMul()(evals[61], evals[63]);
    signal tmp_5085[3] <== [tmp_5083[0] + tmp_5084[0], tmp_5083[1] + tmp_5084[1], tmp_5083[2] + tmp_5084[2]];
    signal tmp_5086[3] <== CMul()(evals[61], evals[65]);
    signal tmp_5087[3] <== [tmp_5085[0] + tmp_5086[0], tmp_5085[1] + tmp_5086[1], tmp_5085[2] + tmp_5086[2]];
    signal tmp_5088[3] <== CMul()(evals[62], evals[64]);
    signal tmp_5089[3] <== [tmp_5087[0] + tmp_5088[0], tmp_5087[1] + tmp_5088[1], tmp_5087[2] + tmp_5088[2]];
    signal tmp_5090[3] <== CMul()(evals[62], evals[65]);
    signal tmp_5091[3] <== [tmp_5089[0] + tmp_5090[0], tmp_5089[1] + tmp_5090[1], tmp_5089[2] + tmp_5090[2]];
    signal tmp_5092[3] <== [evals[67][0] - tmp_5091[0], evals[67][1] - tmp_5091[1], evals[67][2] - tmp_5091[2]];
    signal tmp_5093[3] <== CMul()(evals[53], tmp_5092);
    signal tmp_5094[3] <== [tmp_5082[0] + tmp_5093[0], tmp_5082[1] + tmp_5093[1], tmp_5082[2] + tmp_5093[2]];
    signal tmp_5095[3] <== CMul()(challengeQ, tmp_5094);
    signal tmp_5096[3] <== CMul()(evals[60], evals[65]);
    signal tmp_5097[3] <== CMul()(evals[62], evals[65]);
    signal tmp_5098[3] <== [tmp_5096[0] + tmp_5097[0], tmp_5096[1] + tmp_5097[1], tmp_5096[2] + tmp_5097[2]];
    signal tmp_5099[3] <== CMul()(evals[62], evals[63]);
    signal tmp_5100[3] <== [tmp_5098[0] + tmp_5099[0], tmp_5098[1] + tmp_5099[1], tmp_5098[2] + tmp_5099[2]];
    signal tmp_5101[3] <== CMul()(evals[61], evals[64]);
    signal tmp_5102[3] <== [tmp_5100[0] + tmp_5101[0], tmp_5100[1] + tmp_5101[1], tmp_5100[2] + tmp_5101[2]];
    signal tmp_5103[3] <== [evals[68][0] - tmp_5102[0], evals[68][1] - tmp_5102[1], evals[68][2] - tmp_5102[2]];
    signal tmp_5104[3] <== CMul()(evals[53], tmp_5103);
    signal tmp_5105[3] <== [tmp_5095[0] + tmp_5104[0], tmp_5095[1] + tmp_5104[1], tmp_5095[2] + tmp_5104[2]];
    signal tmp_5106[3] <== CMul()(challengeQ, tmp_5105);
    signal tmp_5107[3] <== CMul()(evals[69], evals[72]);
    signal tmp_5108[3] <== CMul()(evals[70], evals[74]);
    signal tmp_5109[3] <== [tmp_5107[0] + tmp_5108[0], tmp_5107[1] + tmp_5108[1], tmp_5107[2] + tmp_5108[2]];
    signal tmp_5110[3] <== CMul()(evals[71], evals[73]);
    signal tmp_5111[3] <== [tmp_5109[0] + tmp_5110[0], tmp_5109[1] + tmp_5110[1], tmp_5109[2] + tmp_5110[2]];
    signal tmp_5112[3] <== [evals[75][0] - tmp_5111[0], evals[75][1] - tmp_5111[1], evals[75][2] - tmp_5111[2]];
    signal tmp_5113[3] <== CMul()(evals[53], tmp_5112);
    signal tmp_5114[3] <== [tmp_5106[0] + tmp_5113[0], tmp_5106[1] + tmp_5113[1], tmp_5106[2] + tmp_5113[2]];
    signal tmp_5115[3] <== CMul()(challengeQ, tmp_5114);
    signal tmp_5116[3] <== CMul()(evals[69], evals[73]);
    signal tmp_5117[3] <== CMul()(evals[70], evals[72]);
    signal tmp_5118[3] <== [tmp_5116[0] + tmp_5117[0], tmp_5116[1] + tmp_5117[1], tmp_5116[2] + tmp_5117[2]];
    signal tmp_5119[3] <== CMul()(evals[70], evals[74]);
    signal tmp_5120[3] <== [tmp_5118[0] + tmp_5119[0], tmp_5118[1] + tmp_5119[1], tmp_5118[2] + tmp_5119[2]];
    signal tmp_5121[3] <== CMul()(evals[71], evals[73]);
    signal tmp_5122[3] <== [tmp_5120[0] + tmp_5121[0], tmp_5120[1] + tmp_5121[1], tmp_5120[2] + tmp_5121[2]];
    signal tmp_5123[3] <== CMul()(evals[71], evals[74]);
    signal tmp_5124[3] <== [tmp_5122[0] + tmp_5123[0], tmp_5122[1] + tmp_5123[1], tmp_5122[2] + tmp_5123[2]];
    signal tmp_5125[3] <== [evals[76][0] - tmp_5124[0], evals[76][1] - tmp_5124[1], evals[76][2] - tmp_5124[2]];
    signal tmp_5126[3] <== CMul()(evals[53], tmp_5125);
    signal tmp_5127[3] <== [tmp_5115[0] + tmp_5126[0], tmp_5115[1] + tmp_5126[1], tmp_5115[2] + tmp_5126[2]];
    signal tmp_5128[3] <== CMul()(challengeQ, tmp_5127);
    signal tmp_5129[3] <== CMul()(evals[69], evals[74]);
    signal tmp_5130[3] <== CMul()(evals[71], evals[74]);
    signal tmp_5131[3] <== [tmp_5129[0] + tmp_5130[0], tmp_5129[1] + tmp_5130[1], tmp_5129[2] + tmp_5130[2]];
    signal tmp_5132[3] <== CMul()(evals[71], evals[72]);
    signal tmp_5133[3] <== [tmp_5131[0] + tmp_5132[0], tmp_5131[1] + tmp_5132[1], tmp_5131[2] + tmp_5132[2]];
    signal tmp_5134[3] <== CMul()(evals[70], evals[73]);
    signal tmp_5135[3] <== [tmp_5133[0] + tmp_5134[0], tmp_5133[1] + tmp_5134[1], tmp_5133[2] + tmp_5134[2]];
    signal tmp_5136[3] <== [evals[77][0] - tmp_5135[0], evals[77][1] - tmp_5135[1], evals[77][2] - tmp_5135[2]];
    signal tmp_5137[3] <== CMul()(evals[53], tmp_5136);
    signal tmp_5138[3] <== [tmp_5128[0] + tmp_5137[0], tmp_5128[1] + tmp_5137[1], tmp_5128[2] + tmp_5137[2]];
    signal tmp_5139[3] <== CMul()(challengeQ, tmp_5138);
    signal tmp_5140[3] <== CMul()(evals[39], evals[60]);
    signal tmp_5141[3] <== CMul()(evals[40], evals[63]);
    signal tmp_5142[3] <== [tmp_5140[0] + tmp_5141[0], tmp_5140[1] + tmp_5141[1], tmp_5140[2] + tmp_5141[2]];
    signal tmp_5143[3] <== CMul()(evals[41], evals[66]);
    signal tmp_5144[3] <== [tmp_5142[0] + tmp_5143[0], tmp_5142[1] + tmp_5143[1], tmp_5142[2] + tmp_5143[2]];
    signal tmp_5145[3] <== CMul()(evals[42], evals[69]);
    signal tmp_5146[3] <== [tmp_5144[0] + tmp_5145[0], tmp_5144[1] + tmp_5145[1], tmp_5144[2] + tmp_5145[2]];
    signal tmp_5147[3] <== CMul()(evals[45], evals[60]);
    signal tmp_5148[3] <== [tmp_5146[0] + tmp_5147[0], tmp_5146[1] + tmp_5147[1], tmp_5146[2] + tmp_5147[2]];
    signal tmp_5149[3] <== CMul()(evals[46], evals[63]);
    signal tmp_5150[3] <== [tmp_5148[0] + tmp_5149[0], tmp_5148[1] + tmp_5149[1], tmp_5148[2] + tmp_5149[2]];
    signal tmp_5151[3] <== [evals[72][0] - tmp_5150[0], evals[72][1] - tmp_5150[1], evals[72][2] - tmp_5150[2]];
    signal tmp_5152[3] <== CMul()(evals[55], tmp_5151);
    signal tmp_5153[3] <== [tmp_5139[0] + tmp_5152[0], tmp_5139[1] + tmp_5152[1], tmp_5139[2] + tmp_5152[2]];
    signal tmp_5154[3] <== CMul()(challengeQ, tmp_5153);
    signal tmp_5155[3] <== CMul()(evals[39], evals[61]);
    signal tmp_5156[3] <== CMul()(evals[40], evals[64]);
    signal tmp_5157[3] <== [tmp_5155[0] + tmp_5156[0], tmp_5155[1] + tmp_5156[1], tmp_5155[2] + tmp_5156[2]];
    signal tmp_5158[3] <== CMul()(evals[41], evals[67]);
    signal tmp_5159[3] <== [tmp_5157[0] + tmp_5158[0], tmp_5157[1] + tmp_5158[1], tmp_5157[2] + tmp_5158[2]];
    signal tmp_5160[3] <== CMul()(evals[42], evals[70]);
    signal tmp_5161[3] <== [tmp_5159[0] + tmp_5160[0], tmp_5159[1] + tmp_5160[1], tmp_5159[2] + tmp_5160[2]];
    signal tmp_5162[3] <== CMul()(evals[45], evals[61]);
    signal tmp_5163[3] <== [tmp_5161[0] + tmp_5162[0], tmp_5161[1] + tmp_5162[1], tmp_5161[2] + tmp_5162[2]];
    signal tmp_5164[3] <== CMul()(evals[46], evals[64]);
    signal tmp_5165[3] <== [tmp_5163[0] + tmp_5164[0], tmp_5163[1] + tmp_5164[1], tmp_5163[2] + tmp_5164[2]];
    signal tmp_5166[3] <== [evals[73][0] - tmp_5165[0], evals[73][1] - tmp_5165[1], evals[73][2] - tmp_5165[2]];
    signal tmp_5167[3] <== CMul()(evals[55], tmp_5166);
    signal tmp_5168[3] <== [tmp_5154[0] + tmp_5167[0], tmp_5154[1] + tmp_5167[1], tmp_5154[2] + tmp_5167[2]];
    signal tmp_5169[3] <== CMul()(challengeQ, tmp_5168);
    signal tmp_5170[3] <== CMul()(evals[39], evals[62]);
    signal tmp_5171[3] <== CMul()(evals[40], evals[65]);
    signal tmp_5172[3] <== [tmp_5170[0] + tmp_5171[0], tmp_5170[1] + tmp_5171[1], tmp_5170[2] + tmp_5171[2]];
    signal tmp_5173[3] <== CMul()(evals[41], evals[68]);
    signal tmp_5174[3] <== [tmp_5172[0] + tmp_5173[0], tmp_5172[1] + tmp_5173[1], tmp_5172[2] + tmp_5173[2]];
    signal tmp_5175[3] <== CMul()(evals[42], evals[71]);
    signal tmp_5176[3] <== [tmp_5174[0] + tmp_5175[0], tmp_5174[1] + tmp_5175[1], tmp_5174[2] + tmp_5175[2]];
    signal tmp_5177[3] <== CMul()(evals[45], evals[62]);
    signal tmp_5178[3] <== [tmp_5176[0] + tmp_5177[0], tmp_5176[1] + tmp_5177[1], tmp_5176[2] + tmp_5177[2]];
    signal tmp_5179[3] <== CMul()(evals[46], evals[65]);
    signal tmp_5180[3] <== [tmp_5178[0] + tmp_5179[0], tmp_5178[1] + tmp_5179[1], tmp_5178[2] + tmp_5179[2]];
    signal tmp_5181[3] <== [evals[74][0] - tmp_5180[0], evals[74][1] - tmp_5180[1], evals[74][2] - tmp_5180[2]];
    signal tmp_5182[3] <== CMul()(evals[55], tmp_5181);
    signal tmp_5183[3] <== [tmp_5169[0] + tmp_5182[0], tmp_5169[1] + tmp_5182[1], tmp_5169[2] + tmp_5182[2]];
    signal tmp_5184[3] <== CMul()(challengeQ, tmp_5183);
    signal tmp_5185[3] <== CMul()(evals[39], evals[60]);
    signal tmp_5186[3] <== CMul()(evals[40], evals[63]);
    signal tmp_5187[3] <== [tmp_5185[0] - tmp_5186[0], tmp_5185[1] - tmp_5186[1], tmp_5185[2] - tmp_5186[2]];
    signal tmp_5188[3] <== CMul()(evals[43], evals[66]);
    signal tmp_5189[3] <== [tmp_5187[0] + tmp_5188[0], tmp_5187[1] + tmp_5188[1], tmp_5187[2] + tmp_5188[2]];
    signal tmp_5190[3] <== CMul()(evals[44], evals[69]);
    signal tmp_5191[3] <== [tmp_5189[0] - tmp_5190[0], tmp_5189[1] - tmp_5190[1], tmp_5189[2] - tmp_5190[2]];
    signal tmp_5192[3] <== CMul()(evals[45], evals[60]);
    signal tmp_5193[3] <== [tmp_5191[0] + tmp_5192[0], tmp_5191[1] + tmp_5192[1], tmp_5191[2] + tmp_5192[2]];
    signal tmp_5194[3] <== CMul()(evals[46], evals[63]);
    signal tmp_5195[3] <== [tmp_5193[0] - tmp_5194[0], tmp_5193[1] - tmp_5194[1], tmp_5193[2] - tmp_5194[2]];
    signal tmp_5196[3] <== [evals[75][0] - tmp_5195[0], evals[75][1] - tmp_5195[1], evals[75][2] - tmp_5195[2]];
    signal tmp_5197[3] <== CMul()(evals[55], tmp_5196);
    signal tmp_5198[3] <== [tmp_5184[0] + tmp_5197[0], tmp_5184[1] + tmp_5197[1], tmp_5184[2] + tmp_5197[2]];
    signal tmp_5199[3] <== CMul()(challengeQ, tmp_5198);
    signal tmp_5200[3] <== CMul()(evals[39], evals[61]);
    signal tmp_5201[3] <== CMul()(evals[40], evals[64]);
    signal tmp_5202[3] <== [tmp_5200[0] - tmp_5201[0], tmp_5200[1] - tmp_5201[1], tmp_5200[2] - tmp_5201[2]];
    signal tmp_5203[3] <== CMul()(evals[43], evals[67]);
    signal tmp_5204[3] <== [tmp_5202[0] + tmp_5203[0], tmp_5202[1] + tmp_5203[1], tmp_5202[2] + tmp_5203[2]];
    signal tmp_5205[3] <== CMul()(evals[44], evals[70]);
    signal tmp_5206[3] <== [tmp_5204[0] - tmp_5205[0], tmp_5204[1] - tmp_5205[1], tmp_5204[2] - tmp_5205[2]];
    signal tmp_5207[3] <== CMul()(evals[45], evals[61]);
    signal tmp_5208[3] <== [tmp_5206[0] + tmp_5207[0], tmp_5206[1] + tmp_5207[1], tmp_5206[2] + tmp_5207[2]];
    signal tmp_5209[3] <== CMul()(evals[46], evals[64]);
    signal tmp_5210[3] <== [tmp_5208[0] - tmp_5209[0], tmp_5208[1] - tmp_5209[1], tmp_5208[2] - tmp_5209[2]];
    signal tmp_5211[3] <== [evals[76][0] - tmp_5210[0], evals[76][1] - tmp_5210[1], evals[76][2] - tmp_5210[2]];
    signal tmp_5212[3] <== CMul()(evals[55], tmp_5211);
    signal tmp_5213[3] <== [tmp_5199[0] + tmp_5212[0], tmp_5199[1] + tmp_5212[1], tmp_5199[2] + tmp_5212[2]];
    signal tmp_5214[3] <== CMul()(challengeQ, tmp_5213);
    signal tmp_5215[3] <== CMul()(evals[39], evals[62]);
    signal tmp_5216[3] <== CMul()(evals[40], evals[65]);
    signal tmp_5217[3] <== [tmp_5215[0] - tmp_5216[0], tmp_5215[1] - tmp_5216[1], tmp_5215[2] - tmp_5216[2]];
    signal tmp_5218[3] <== CMul()(evals[43], evals[68]);
    signal tmp_5219[3] <== [tmp_5217[0] + tmp_5218[0], tmp_5217[1] + tmp_5218[1], tmp_5217[2] + tmp_5218[2]];
    signal tmp_5220[3] <== CMul()(evals[44], evals[71]);
    signal tmp_5221[3] <== [tmp_5219[0] - tmp_5220[0], tmp_5219[1] - tmp_5220[1], tmp_5219[2] - tmp_5220[2]];
    signal tmp_5222[3] <== CMul()(evals[45], evals[62]);
    signal tmp_5223[3] <== [tmp_5221[0] + tmp_5222[0], tmp_5221[1] + tmp_5222[1], tmp_5221[2] + tmp_5222[2]];
    signal tmp_5224[3] <== CMul()(evals[46], evals[65]);
    signal tmp_5225[3] <== [tmp_5223[0] - tmp_5224[0], tmp_5223[1] - tmp_5224[1], tmp_5223[2] - tmp_5224[2]];
    signal tmp_5226[3] <== [evals[77][0] - tmp_5225[0], evals[77][1] - tmp_5225[1], evals[77][2] - tmp_5225[2]];
    signal tmp_5227[3] <== CMul()(evals[55], tmp_5226);
    signal tmp_5228[3] <== [tmp_5214[0] + tmp_5227[0], tmp_5214[1] + tmp_5227[1], tmp_5214[2] + tmp_5227[2]];
    signal tmp_5229[3] <== CMul()(challengeQ, tmp_5228);
    signal tmp_5230[3] <== CMul()(evals[39], evals[60]);
    signal tmp_5231[3] <== CMul()(evals[40], evals[63]);
    signal tmp_5232[3] <== [tmp_5230[0] + tmp_5231[0], tmp_5230[1] + tmp_5231[1], tmp_5230[2] + tmp_5231[2]];
    signal tmp_5233[3] <== CMul()(evals[41], evals[66]);
    signal tmp_5234[3] <== [tmp_5232[0] - tmp_5233[0], tmp_5232[1] - tmp_5233[1], tmp_5232[2] - tmp_5233[2]];
    signal tmp_5235[3] <== CMul()(evals[42], evals[69]);
    signal tmp_5236[3] <== [tmp_5234[0] - tmp_5235[0], tmp_5234[1] - tmp_5235[1], tmp_5234[2] - tmp_5235[2]];
    signal tmp_5237[3] <== CMul()(evals[45], evals[66]);
    signal tmp_5238[3] <== [tmp_5236[0] + tmp_5237[0], tmp_5236[1] + tmp_5237[1], tmp_5236[2] + tmp_5237[2]];
    signal tmp_5239[3] <== CMul()(evals[47], evals[69]);
    signal tmp_5240[3] <== [tmp_5238[0] + tmp_5239[0], tmp_5238[1] + tmp_5239[1], tmp_5238[2] + tmp_5239[2]];
    signal tmp_5241[3] <== [evals[78][0] - tmp_5240[0], evals[78][1] - tmp_5240[1], evals[78][2] - tmp_5240[2]];
    signal tmp_5242[3] <== CMul()(evals[55], tmp_5241);
    signal tmp_5243[3] <== [tmp_5229[0] + tmp_5242[0], tmp_5229[1] + tmp_5242[1], tmp_5229[2] + tmp_5242[2]];
    signal tmp_5244[3] <== CMul()(challengeQ, tmp_5243);
    signal tmp_5245[3] <== CMul()(evals[39], evals[61]);
    signal tmp_5246[3] <== CMul()(evals[40], evals[64]);
    signal tmp_5247[3] <== [tmp_5245[0] + tmp_5246[0], tmp_5245[1] + tmp_5246[1], tmp_5245[2] + tmp_5246[2]];
    signal tmp_5248[3] <== CMul()(evals[41], evals[67]);
    signal tmp_5249[3] <== [tmp_5247[0] - tmp_5248[0], tmp_5247[1] - tmp_5248[1], tmp_5247[2] - tmp_5248[2]];
    signal tmp_5250[3] <== CMul()(evals[42], evals[70]);
    signal tmp_5251[3] <== [tmp_5249[0] - tmp_5250[0], tmp_5249[1] - tmp_5250[1], tmp_5249[2] - tmp_5250[2]];
    signal tmp_5252[3] <== CMul()(evals[45], evals[67]);
    signal tmp_5253[3] <== [tmp_5251[0] + tmp_5252[0], tmp_5251[1] + tmp_5252[1], tmp_5251[2] + tmp_5252[2]];
    signal tmp_5254[3] <== CMul()(evals[47], evals[70]);
    signal tmp_5255[3] <== [tmp_5253[0] + tmp_5254[0], tmp_5253[1] + tmp_5254[1], tmp_5253[2] + tmp_5254[2]];
    signal tmp_5256[3] <== [evals[79][0] - tmp_5255[0], evals[79][1] - tmp_5255[1], evals[79][2] - tmp_5255[2]];
    signal tmp_5257[3] <== CMul()(evals[55], tmp_5256);
    signal tmp_5258[3] <== [tmp_5244[0] + tmp_5257[0], tmp_5244[1] + tmp_5257[1], tmp_5244[2] + tmp_5257[2]];
    signal tmp_5259[3] <== CMul()(challengeQ, tmp_5258);
    signal tmp_5260[3] <== CMul()(evals[39], evals[62]);
    signal tmp_5261[3] <== CMul()(evals[40], evals[65]);
    signal tmp_5262[3] <== [tmp_5260[0] + tmp_5261[0], tmp_5260[1] + tmp_5261[1], tmp_5260[2] + tmp_5261[2]];
    signal tmp_5263[3] <== CMul()(evals[41], evals[68]);
    signal tmp_5264[3] <== [tmp_5262[0] - tmp_5263[0], tmp_5262[1] - tmp_5263[1], tmp_5262[2] - tmp_5263[2]];
    signal tmp_5265[3] <== CMul()(evals[42], evals[71]);
    signal tmp_5266[3] <== [tmp_5264[0] - tmp_5265[0], tmp_5264[1] - tmp_5265[1], tmp_5264[2] - tmp_5265[2]];
    signal tmp_5267[3] <== CMul()(evals[45], evals[68]);
    signal tmp_5268[3] <== [tmp_5266[0] + tmp_5267[0], tmp_5266[1] + tmp_5267[1], tmp_5266[2] + tmp_5267[2]];
    signal tmp_5269[3] <== CMul()(evals[47], evals[71]);
    signal tmp_5270[3] <== [tmp_5268[0] + tmp_5269[0], tmp_5268[1] + tmp_5269[1], tmp_5268[2] + tmp_5269[2]];
    signal tmp_5271[3] <== [evals[80][0] - tmp_5270[0], evals[80][1] - tmp_5270[1], evals[80][2] - tmp_5270[2]];
    signal tmp_5272[3] <== CMul()(evals[55], tmp_5271);
    signal tmp_5273[3] <== [tmp_5259[0] + tmp_5272[0], tmp_5259[1] + tmp_5272[1], tmp_5259[2] + tmp_5272[2]];
    signal tmp_5274[3] <== CMul()(challengeQ, tmp_5273);
    signal tmp_5275[3] <== CMul()(evals[39], evals[60]);
    signal tmp_5276[3] <== CMul()(evals[40], evals[63]);
    signal tmp_5277[3] <== [tmp_5275[0] - tmp_5276[0], tmp_5275[1] - tmp_5276[1], tmp_5275[2] - tmp_5276[2]];
    signal tmp_5278[3] <== CMul()(evals[43], evals[66]);
    signal tmp_5279[3] <== [tmp_5277[0] - tmp_5278[0], tmp_5277[1] - tmp_5278[1], tmp_5277[2] - tmp_5278[2]];
    signal tmp_5280[3] <== CMul()(evals[44], evals[69]);
    signal tmp_5281[3] <== [tmp_5279[0] + tmp_5280[0], tmp_5279[1] + tmp_5280[1], tmp_5279[2] + tmp_5280[2]];
    signal tmp_5282[3] <== CMul()(evals[45], evals[66]);
    signal tmp_5283[3] <== [tmp_5281[0] + tmp_5282[0], tmp_5281[1] + tmp_5282[1], tmp_5281[2] + tmp_5282[2]];
    signal tmp_5284[3] <== CMul()(evals[47], evals[69]);
    signal tmp_5285[3] <== [tmp_5283[0] - tmp_5284[0], tmp_5283[1] - tmp_5284[1], tmp_5283[2] - tmp_5284[2]];
    signal tmp_5286[3] <== [evals[81][0] - tmp_5285[0], evals[81][1] - tmp_5285[1], evals[81][2] - tmp_5285[2]];
    signal tmp_5287[3] <== CMul()(evals[55], tmp_5286);
    signal tmp_5288[3] <== [tmp_5274[0] + tmp_5287[0], tmp_5274[1] + tmp_5287[1], tmp_5274[2] + tmp_5287[2]];
    signal tmp_5289[3] <== CMul()(challengeQ, tmp_5288);
    signal tmp_5290[3] <== CMul()(evals[39], evals[61]);
    signal tmp_5291[3] <== CMul()(evals[40], evals[64]);
    signal tmp_5292[3] <== [tmp_5290[0] - tmp_5291[0], tmp_5290[1] - tmp_5291[1], tmp_5290[2] - tmp_5291[2]];
    signal tmp_5293[3] <== CMul()(evals[43], evals[67]);
    signal tmp_5294[3] <== [tmp_5292[0] - tmp_5293[0], tmp_5292[1] - tmp_5293[1], tmp_5292[2] - tmp_5293[2]];
    signal tmp_5295[3] <== CMul()(evals[44], evals[70]);
    signal tmp_5296[3] <== [tmp_5294[0] + tmp_5295[0], tmp_5294[1] + tmp_5295[1], tmp_5294[2] + tmp_5295[2]];
    signal tmp_5297[3] <== CMul()(evals[45], evals[67]);
    signal tmp_5298[3] <== [tmp_5296[0] + tmp_5297[0], tmp_5296[1] + tmp_5297[1], tmp_5296[2] + tmp_5297[2]];
    signal tmp_5299[3] <== CMul()(evals[47], evals[70]);
    signal tmp_5300[3] <== [tmp_5298[0] - tmp_5299[0], tmp_5298[1] - tmp_5299[1], tmp_5298[2] - tmp_5299[2]];
    signal tmp_5301[3] <== [evals[82][0] - tmp_5300[0], evals[82][1] - tmp_5300[1], evals[82][2] - tmp_5300[2]];
    signal tmp_5302[3] <== CMul()(evals[55], tmp_5301);
    signal tmp_5303[3] <== [tmp_5289[0] + tmp_5302[0], tmp_5289[1] + tmp_5302[1], tmp_5289[2] + tmp_5302[2]];
    signal tmp_5304[3] <== CMul()(challengeQ, tmp_5303);
    signal tmp_5305[3] <== CMul()(evals[39], evals[62]);
    signal tmp_5306[3] <== CMul()(evals[40], evals[65]);
    signal tmp_5307[3] <== [tmp_5305[0] - tmp_5306[0], tmp_5305[1] - tmp_5306[1], tmp_5305[2] - tmp_5306[2]];
    signal tmp_5308[3] <== CMul()(evals[43], evals[68]);
    signal tmp_5309[3] <== [tmp_5307[0] - tmp_5308[0], tmp_5307[1] - tmp_5308[1], tmp_5307[2] - tmp_5308[2]];
    signal tmp_5310[3] <== CMul()(evals[44], evals[71]);
    signal tmp_5311[3] <== [tmp_5309[0] + tmp_5310[0], tmp_5309[1] + tmp_5310[1], tmp_5309[2] + tmp_5310[2]];
    signal tmp_5312[3] <== CMul()(evals[45], evals[68]);
    signal tmp_5313[3] <== [tmp_5311[0] + tmp_5312[0], tmp_5311[1] + tmp_5312[1], tmp_5311[2] + tmp_5312[2]];
    signal tmp_5314[3] <== CMul()(evals[47], evals[71]);
    signal tmp_5315[3] <== [tmp_5313[0] - tmp_5314[0], tmp_5313[1] - tmp_5314[1], tmp_5313[2] - tmp_5314[2]];
    signal tmp_5316[3] <== [evals[83][0] - tmp_5315[0], evals[83][1] - tmp_5315[1], evals[83][2] - tmp_5315[2]];
    signal tmp_5317[3] <== CMul()(evals[55], tmp_5316);
    signal tmp_5318[3] <== [tmp_5304[0] + tmp_5317[0], tmp_5304[1] + tmp_5317[1], tmp_5304[2] + tmp_5317[2]];
    signal tmp_5319[3] <== CMul()(challengeQ, tmp_5318);
    signal tmp_5320[3] <== CMul()(evals[72], evals[75]);
    signal tmp_5321[3] <== CMul()(evals[73], evals[77]);
    signal tmp_5322[3] <== [tmp_5320[0] + tmp_5321[0], tmp_5320[1] + tmp_5321[1], tmp_5320[2] + tmp_5321[2]];
    signal tmp_5323[3] <== CMul()(evals[74], evals[76]);
    signal tmp_5324[3] <== [tmp_5322[0] + tmp_5323[0], tmp_5322[1] + tmp_5323[1], tmp_5322[2] + tmp_5323[2]];
    signal tmp_5325[3] <== [tmp_5324[0] + evals[69][0], tmp_5324[1] + evals[69][1], tmp_5324[2] + evals[69][2]];
    signal tmp_5326[3] <== CMul()(tmp_5325, evals[75]);
    signal tmp_5327[3] <== CMul()(evals[72], evals[76]);
    signal tmp_5328[3] <== CMul()(evals[73], evals[75]);
    signal tmp_5329[3] <== [tmp_5327[0] + tmp_5328[0], tmp_5327[1] + tmp_5328[1], tmp_5327[2] + tmp_5328[2]];
    signal tmp_5330[3] <== CMul()(evals[73], evals[77]);
    signal tmp_5331[3] <== [tmp_5329[0] + tmp_5330[0], tmp_5329[1] + tmp_5330[1], tmp_5329[2] + tmp_5330[2]];
    signal tmp_5332[3] <== CMul()(evals[74], evals[76]);
    signal tmp_5333[3] <== [tmp_5331[0] + tmp_5332[0], tmp_5331[1] + tmp_5332[1], tmp_5331[2] + tmp_5332[2]];
    signal tmp_5334[3] <== CMul()(evals[74], evals[77]);
    signal tmp_5335[3] <== [tmp_5333[0] + tmp_5334[0], tmp_5333[1] + tmp_5334[1], tmp_5333[2] + tmp_5334[2]];
    signal tmp_5336[3] <== [tmp_5335[0] + evals[70][0], tmp_5335[1] + evals[70][1], tmp_5335[2] + evals[70][2]];
    signal tmp_5337[3] <== CMul()(tmp_5336, evals[77]);
    signal tmp_5338[3] <== [tmp_5326[0] + tmp_5337[0], tmp_5326[1] + tmp_5337[1], tmp_5326[2] + tmp_5337[2]];
    signal tmp_5339[3] <== CMul()(evals[72], evals[77]);
    signal tmp_5340[3] <== CMul()(evals[74], evals[77]);
    signal tmp_5341[3] <== [tmp_5339[0] + tmp_5340[0], tmp_5339[1] + tmp_5340[1], tmp_5339[2] + tmp_5340[2]];
    signal tmp_5342[3] <== CMul()(evals[74], evals[75]);
    signal tmp_5343[3] <== [tmp_5341[0] + tmp_5342[0], tmp_5341[1] + tmp_5342[1], tmp_5341[2] + tmp_5342[2]];
    signal tmp_5344[3] <== CMul()(evals[73], evals[76]);
    signal tmp_5345[3] <== [tmp_5343[0] + tmp_5344[0], tmp_5343[1] + tmp_5344[1], tmp_5343[2] + tmp_5344[2]];
    signal tmp_5346[3] <== [tmp_5345[0] + evals[71][0], tmp_5345[1] + evals[71][1], tmp_5345[2] + evals[71][2]];
    signal tmp_5347[3] <== CMul()(tmp_5346, evals[76]);
    signal tmp_5348[3] <== [tmp_5338[0] + tmp_5347[0], tmp_5338[1] + tmp_5347[1], tmp_5338[2] + tmp_5347[2]];
    signal tmp_5349[3] <== [tmp_5348[0] + evals[66][0], tmp_5348[1] + evals[66][1], tmp_5348[2] + evals[66][2]];
    signal tmp_5350[3] <== CMul()(tmp_5349, evals[75]);
    signal tmp_5351[3] <== CMul()(tmp_5325, evals[76]);
    signal tmp_5352[3] <== CMul()(tmp_5336, evals[75]);
    signal tmp_5353[3] <== [tmp_5351[0] + tmp_5352[0], tmp_5351[1] + tmp_5352[1], tmp_5351[2] + tmp_5352[2]];
    signal tmp_5354[3] <== CMul()(tmp_5336, evals[77]);
    signal tmp_5355[3] <== [tmp_5353[0] + tmp_5354[0], tmp_5353[1] + tmp_5354[1], tmp_5353[2] + tmp_5354[2]];
    signal tmp_5356[3] <== CMul()(tmp_5346, evals[76]);
    signal tmp_5357[3] <== [tmp_5355[0] + tmp_5356[0], tmp_5355[1] + tmp_5356[1], tmp_5355[2] + tmp_5356[2]];
    signal tmp_5358[3] <== CMul()(tmp_5346, evals[77]);
    signal tmp_5359[3] <== [tmp_5357[0] + tmp_5358[0], tmp_5357[1] + tmp_5358[1], tmp_5357[2] + tmp_5358[2]];
    signal tmp_5360[3] <== [tmp_5359[0] + evals[67][0], tmp_5359[1] + evals[67][1], tmp_5359[2] + evals[67][2]];
    signal tmp_5361[3] <== CMul()(tmp_5360, evals[77]);
    signal tmp_5362[3] <== [tmp_5350[0] + tmp_5361[0], tmp_5350[1] + tmp_5361[1], tmp_5350[2] + tmp_5361[2]];
    signal tmp_5363[3] <== CMul()(tmp_5325, evals[77]);
    signal tmp_5364[3] <== CMul()(tmp_5346, evals[77]);
    signal tmp_5365[3] <== [tmp_5363[0] + tmp_5364[0], tmp_5363[1] + tmp_5364[1], tmp_5363[2] + tmp_5364[2]];
    signal tmp_5366[3] <== CMul()(tmp_5346, evals[75]);
    signal tmp_5367[3] <== [tmp_5365[0] + tmp_5366[0], tmp_5365[1] + tmp_5366[1], tmp_5365[2] + tmp_5366[2]];
    signal tmp_5368[3] <== CMul()(tmp_5336, evals[76]);
    signal tmp_5369[3] <== [tmp_5367[0] + tmp_5368[0], tmp_5367[1] + tmp_5368[1], tmp_5367[2] + tmp_5368[2]];
    signal tmp_5370[3] <== [tmp_5369[0] + evals[68][0], tmp_5369[1] + evals[68][1], tmp_5369[2] + evals[68][2]];
    signal tmp_5371[3] <== CMul()(tmp_5370, evals[76]);
    signal tmp_5372[3] <== [tmp_5362[0] + tmp_5371[0], tmp_5362[1] + tmp_5371[1], tmp_5362[2] + tmp_5371[2]];
    signal tmp_5373[3] <== [tmp_5372[0] + evals[63][0], tmp_5372[1] + evals[63][1], tmp_5372[2] + evals[63][2]];
    signal tmp_5374[3] <== CMul()(tmp_5373, evals[75]);
    signal tmp_5375[3] <== CMul()(tmp_5349, evals[76]);
    signal tmp_5376[3] <== CMul()(tmp_5360, evals[75]);
    signal tmp_5377[3] <== [tmp_5375[0] + tmp_5376[0], tmp_5375[1] + tmp_5376[1], tmp_5375[2] + tmp_5376[2]];
    signal tmp_5378[3] <== CMul()(tmp_5360, evals[77]);
    signal tmp_5379[3] <== [tmp_5377[0] + tmp_5378[0], tmp_5377[1] + tmp_5378[1], tmp_5377[2] + tmp_5378[2]];
    signal tmp_5380[3] <== CMul()(tmp_5370, evals[76]);
    signal tmp_5381[3] <== [tmp_5379[0] + tmp_5380[0], tmp_5379[1] + tmp_5380[1], tmp_5379[2] + tmp_5380[2]];
    signal tmp_5382[3] <== CMul()(tmp_5370, evals[77]);
    signal tmp_5383[3] <== [tmp_5381[0] + tmp_5382[0], tmp_5381[1] + tmp_5382[1], tmp_5381[2] + tmp_5382[2]];
    signal tmp_5384[3] <== [tmp_5383[0] + evals[64][0], tmp_5383[1] + evals[64][1], tmp_5383[2] + evals[64][2]];
    signal tmp_5385[3] <== CMul()(tmp_5384, evals[77]);
    signal tmp_5386[3] <== [tmp_5374[0] + tmp_5385[0], tmp_5374[1] + tmp_5385[1], tmp_5374[2] + tmp_5385[2]];
    signal tmp_5387[3] <== CMul()(tmp_5349, evals[77]);
    signal tmp_5388[3] <== CMul()(tmp_5370, evals[77]);
    signal tmp_5389[3] <== [tmp_5387[0] + tmp_5388[0], tmp_5387[1] + tmp_5388[1], tmp_5387[2] + tmp_5388[2]];
    signal tmp_5390[3] <== CMul()(tmp_5370, evals[75]);
    signal tmp_5391[3] <== [tmp_5389[0] + tmp_5390[0], tmp_5389[1] + tmp_5390[1], tmp_5389[2] + tmp_5390[2]];
    signal tmp_5392[3] <== CMul()(tmp_5360, evals[76]);
    signal tmp_5393[3] <== [tmp_5391[0] + tmp_5392[0], tmp_5391[1] + tmp_5392[1], tmp_5391[2] + tmp_5392[2]];
    signal tmp_5394[3] <== [tmp_5393[0] + evals[65][0], tmp_5393[1] + evals[65][1], tmp_5393[2] + evals[65][2]];
    signal tmp_5395[3] <== CMul()(tmp_5394, evals[76]);
    signal tmp_5396[3] <== [tmp_5386[0] + tmp_5395[0], tmp_5386[1] + tmp_5395[1], tmp_5386[2] + tmp_5395[2]];
    signal tmp_5397[3] <== [tmp_5396[0] + evals[60][0], tmp_5396[1] + evals[60][1], tmp_5396[2] + evals[60][2]];
    signal tmp_5398[3] <== [tmp_5397[0] - evals[78][0], tmp_5397[1] - evals[78][1], tmp_5397[2] - evals[78][2]];
    signal tmp_5399[3] <== CMul()(evals[54], tmp_5398);
    signal tmp_5400[3] <== [tmp_5319[0] + tmp_5399[0], tmp_5319[1] + tmp_5399[1], tmp_5319[2] + tmp_5399[2]];
    signal tmp_5401[3] <== CMul()(challengeQ, tmp_5400);
    signal tmp_5402[3] <== CMul()(tmp_5373, evals[76]);
    signal tmp_5403[3] <== CMul()(tmp_5384, evals[75]);
    signal tmp_5404[3] <== [tmp_5402[0] + tmp_5403[0], tmp_5402[1] + tmp_5403[1], tmp_5402[2] + tmp_5403[2]];
    signal tmp_5405[3] <== CMul()(tmp_5384, evals[77]);
    signal tmp_5406[3] <== [tmp_5404[0] + tmp_5405[0], tmp_5404[1] + tmp_5405[1], tmp_5404[2] + tmp_5405[2]];
    signal tmp_5407[3] <== CMul()(tmp_5394, evals[76]);
    signal tmp_5408[3] <== [tmp_5406[0] + tmp_5407[0], tmp_5406[1] + tmp_5407[1], tmp_5406[2] + tmp_5407[2]];
    signal tmp_5409[3] <== CMul()(tmp_5394, evals[77]);
    signal tmp_5410[3] <== [tmp_5408[0] + tmp_5409[0], tmp_5408[1] + tmp_5409[1], tmp_5408[2] + tmp_5409[2]];
    signal tmp_5411[3] <== [tmp_5410[0] + evals[61][0], tmp_5410[1] + evals[61][1], tmp_5410[2] + evals[61][2]];
    signal tmp_5412[3] <== [tmp_5411[0] - evals[79][0], tmp_5411[1] - evals[79][1], tmp_5411[2] - evals[79][2]];
    signal tmp_5413[3] <== CMul()(evals[54], tmp_5412);
    signal tmp_5414[3] <== [tmp_5401[0] + tmp_5413[0], tmp_5401[1] + tmp_5413[1], tmp_5401[2] + tmp_5413[2]];
    signal tmp_5415[3] <== CMul()(challengeQ, tmp_5414);
    signal tmp_5416[3] <== CMul()(tmp_5373, evals[77]);
    signal tmp_5417[3] <== CMul()(tmp_5394, evals[77]);
    signal tmp_5418[3] <== [tmp_5416[0] + tmp_5417[0], tmp_5416[1] + tmp_5417[1], tmp_5416[2] + tmp_5417[2]];
    signal tmp_5419[3] <== CMul()(tmp_5394, evals[75]);
    signal tmp_5420[3] <== [tmp_5418[0] + tmp_5419[0], tmp_5418[1] + tmp_5419[1], tmp_5418[2] + tmp_5419[2]];
    signal tmp_5421[3] <== CMul()(tmp_5384, evals[76]);
    signal tmp_5422[3] <== [tmp_5420[0] + tmp_5421[0], tmp_5420[1] + tmp_5421[1], tmp_5420[2] + tmp_5421[2]];
    signal tmp_5423[3] <== [tmp_5422[0] + evals[62][0], tmp_5422[1] + evals[62][1], tmp_5422[2] + evals[62][2]];
    signal tmp_5424[3] <== [tmp_5423[0] - evals[80][0], tmp_5423[1] - evals[80][1], tmp_5423[2] - evals[80][2]];
    signal tmp_5425[3] <== CMul()(evals[54], tmp_5424);
    signal tmp_5426[3] <== [tmp_5415[0] + tmp_5425[0], tmp_5415[1] + tmp_5425[1], tmp_5415[2] + tmp_5425[2]];
    signal tmp_5427[3] <== CMul()(challengeQ, tmp_5426);
    signal tmp_5428[3] <== [1 - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_5429[3] <== [1 - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_5430[3] <== CMul()(tmp_5428, tmp_5429);
    signal tmp_5431[3] <== CMul()(evals[56], tmp_5430);
    signal tmp_5432[3] <== [evals[60][0] - evals[74][0], evals[60][1] - evals[74][1], evals[60][2] - evals[74][2]];
    signal tmp_5433[3] <== CMul()(tmp_5431, tmp_5432);
    signal tmp_5434[3] <== [tmp_5427[0] + tmp_5433[0], tmp_5427[1] + tmp_5433[1], tmp_5427[2] + tmp_5433[2]];
    signal tmp_5435[3] <== CMul()(challengeQ, tmp_5434);
    signal tmp_5436[3] <== CMul()(evals[56], tmp_5430);
    signal tmp_5437[3] <== [evals[61][0] - evals[75][0], evals[61][1] - evals[75][1], evals[61][2] - evals[75][2]];
    signal tmp_5438[3] <== CMul()(tmp_5436, tmp_5437);
    signal tmp_5439[3] <== [tmp_5435[0] + tmp_5438[0], tmp_5435[1] + tmp_5438[1], tmp_5435[2] + tmp_5438[2]];
    signal tmp_5440[3] <== CMul()(challengeQ, tmp_5439);
    signal tmp_5441[3] <== CMul()(evals[56], tmp_5430);
    signal tmp_5442[3] <== [evals[62][0] - evals[76][0], evals[62][1] - evals[76][1], evals[62][2] - evals[76][2]];
    signal tmp_5443[3] <== CMul()(tmp_5441, tmp_5442);
    signal tmp_5444[3] <== [tmp_5440[0] + tmp_5443[0], tmp_5440[1] + tmp_5443[1], tmp_5440[2] + tmp_5443[2]];
    signal tmp_5445[3] <== CMul()(challengeQ, tmp_5444);
    signal tmp_5446[3] <== [1 - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_5447[3] <== CMul()(evals[72], tmp_5446);
    signal tmp_5448[3] <== CMul()(evals[56], tmp_5447);
    signal tmp_5449[3] <== [evals[63][0] - evals[74][0], evals[63][1] - evals[74][1], evals[63][2] - evals[74][2]];
    signal tmp_5450[3] <== CMul()(tmp_5448, tmp_5449);
    signal tmp_5451[3] <== [tmp_5445[0] + tmp_5450[0], tmp_5445[1] + tmp_5450[1], tmp_5445[2] + tmp_5450[2]];
    signal tmp_5452[3] <== CMul()(challengeQ, tmp_5451);
    signal tmp_5453[3] <== CMul()(evals[56], tmp_5447);
    signal tmp_5454[3] <== [evals[64][0] - evals[75][0], evals[64][1] - evals[75][1], evals[64][2] - evals[75][2]];
    signal tmp_5455[3] <== CMul()(tmp_5453, tmp_5454);
    signal tmp_5456[3] <== [tmp_5452[0] + tmp_5455[0], tmp_5452[1] + tmp_5455[1], tmp_5452[2] + tmp_5455[2]];
    signal tmp_5457[3] <== CMul()(challengeQ, tmp_5456);
    signal tmp_5458[3] <== CMul()(evals[56], tmp_5447);
    signal tmp_5459[3] <== [evals[65][0] - evals[76][0], evals[65][1] - evals[76][1], evals[65][2] - evals[76][2]];
    signal tmp_5460[3] <== CMul()(tmp_5458, tmp_5459);
    signal tmp_5461[3] <== [tmp_5457[0] + tmp_5460[0], tmp_5457[1] + tmp_5460[1], tmp_5457[2] + tmp_5460[2]];
    signal tmp_5462[3] <== CMul()(challengeQ, tmp_5461);
    signal tmp_5463[3] <== [1 - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_5464[3] <== CMul()(tmp_5463, evals[73]);
    signal tmp_5465[3] <== CMul()(evals[56], tmp_5464);
    signal tmp_5466[3] <== [evals[66][0] - evals[74][0], evals[66][1] - evals[74][1], evals[66][2] - evals[74][2]];
    signal tmp_5467[3] <== CMul()(tmp_5465, tmp_5466);
    signal tmp_5468[3] <== [tmp_5462[0] + tmp_5467[0], tmp_5462[1] + tmp_5467[1], tmp_5462[2] + tmp_5467[2]];
    signal tmp_5469[3] <== CMul()(challengeQ, tmp_5468);
    signal tmp_5470[3] <== CMul()(evals[56], tmp_5464);
    signal tmp_5471[3] <== [evals[67][0] - evals[75][0], evals[67][1] - evals[75][1], evals[67][2] - evals[75][2]];
    signal tmp_5472[3] <== CMul()(tmp_5470, tmp_5471);
    signal tmp_5473[3] <== [tmp_5469[0] + tmp_5472[0], tmp_5469[1] + tmp_5472[1], tmp_5469[2] + tmp_5472[2]];
    signal tmp_5474[3] <== CMul()(challengeQ, tmp_5473);
    signal tmp_5475[3] <== CMul()(evals[56], tmp_5464);
    signal tmp_5476[3] <== [evals[68][0] - evals[76][0], evals[68][1] - evals[76][1], evals[68][2] - evals[76][2]];
    signal tmp_5477[3] <== CMul()(tmp_5475, tmp_5476);
    signal tmp_5478[3] <== [tmp_5474[0] + tmp_5477[0], tmp_5474[1] + tmp_5477[1], tmp_5474[2] + tmp_5477[2]];
    signal tmp_5479[3] <== CMul()(challengeQ, tmp_5478);
    signal tmp_5480[3] <== CMul()(evals[72], evals[73]);
    signal tmp_5481[3] <== CMul()(evals[56], tmp_5480);
    signal tmp_5482[3] <== [evals[69][0] - evals[74][0], evals[69][1] - evals[74][1], evals[69][2] - evals[74][2]];
    signal tmp_5483[3] <== CMul()(tmp_5481, tmp_5482);
    signal tmp_5484[3] <== [tmp_5479[0] + tmp_5483[0], tmp_5479[1] + tmp_5483[1], tmp_5479[2] + tmp_5483[2]];
    signal tmp_5485[3] <== CMul()(challengeQ, tmp_5484);
    signal tmp_5486[3] <== CMul()(evals[56], tmp_5480);
    signal tmp_5487[3] <== [evals[70][0] - evals[75][0], evals[70][1] - evals[75][1], evals[70][2] - evals[75][2]];
    signal tmp_5488[3] <== CMul()(tmp_5486, tmp_5487);
    signal tmp_5489[3] <== [tmp_5485[0] + tmp_5488[0], tmp_5485[1] + tmp_5488[1], tmp_5485[2] + tmp_5488[2]];
    signal tmp_5490[3] <== CMul()(challengeQ, tmp_5489);
    signal tmp_5491[3] <== CMul()(evals[56], tmp_5480);
    signal tmp_5492[3] <== [evals[71][0] - evals[76][0], evals[71][1] - evals[76][1], evals[71][2] - evals[76][2]];
    signal tmp_5493[3] <== CMul()(tmp_5491, tmp_5492);
    signal tmp_5494[3] <== [tmp_5490[0] + tmp_5493[0], tmp_5490[1] + tmp_5493[1], tmp_5490[2] + tmp_5493[2]];
    signal tmp_5495[3] <== CMul()(challengeQ, tmp_5494);
    signal tmp_5496[3] <== [1 - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_5497[3] <== CMul()(evals[72], tmp_5496);
    signal tmp_5498[3] <== CMul()(evals[56], tmp_5497);
    signal tmp_5499[3] <== [tmp_5495[0] + tmp_5498[0], tmp_5495[1] + tmp_5498[1], tmp_5495[2] + tmp_5498[2]];
    signal tmp_5500[3] <== CMul()(challengeQ, tmp_5499);
    signal tmp_5501[3] <== [1 - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_5502[3] <== CMul()(evals[73], tmp_5501);
    signal tmp_5503[3] <== CMul()(evals[56], tmp_5502);
    signal tmp_5504[3] <== [tmp_5500[0] + tmp_5503[0], tmp_5500[1] + tmp_5503[1], tmp_5500[2] + tmp_5503[2]];
    signal tmp_5505[3] <== CMul()(challengeQ, tmp_5504);
    signal tmp_5506[3] <== CMul()(evals[58], challengesStage2[0]);
    signal tmp_5507[3] <== [tmp_5506[0] + evals[60][0], tmp_5506[1] + evals[60][1], tmp_5506[2] + evals[60][2]];
    signal tmp_5508[3] <== CMul()(tmp_5507, challengesStage2[0]);
    signal tmp_5509[3] <== [tmp_5508[0] + 1, tmp_5508[1], tmp_5508[2]];
    signal tmp_5510[3] <== [tmp_5509[0] + challengesStage2[1][0], tmp_5509[1] + challengesStage2[1][1], tmp_5509[2] + challengesStage2[1][2]];
    signal tmp_5511[3] <== [tmp_5510[0] - 1, tmp_5510[1], tmp_5510[2]];
    signal tmp_5512[3] <== [tmp_5511[0] + 1, tmp_5511[1], tmp_5511[2]];
    signal tmp_5513[3] <== [12275445934081160404 * evals[58][0], 12275445934081160404 * evals[58][1], 12275445934081160404 * evals[58][2]];
    signal tmp_5514[3] <== CMul()(tmp_5513, challengesStage2[0]);
    signal tmp_5515[3] <== [tmp_5514[0] + evals[61][0], tmp_5514[1] + evals[61][1], tmp_5514[2] + evals[61][2]];
    signal tmp_5516[3] <== CMul()(tmp_5515, challengesStage2[0]);
    signal tmp_5517[3] <== [tmp_5516[0] + 1, tmp_5516[1], tmp_5516[2]];
    signal tmp_5518[3] <== [tmp_5517[0] + challengesStage2[1][0], tmp_5517[1] + challengesStage2[1][1], tmp_5517[2] + challengesStage2[1][2]];
    signal tmp_5519[3] <== [tmp_5518[0] - 1, tmp_5518[1], tmp_5518[2]];
    signal tmp_5520[3] <== [tmp_5519[0] + 1, tmp_5519[1], tmp_5519[2]];
    signal tmp_5521[3] <== CMul()(tmp_5512, tmp_5520);
    signal tmp_5522[3] <== [4756475762779100925 * evals[58][0], 4756475762779100925 * evals[58][1], 4756475762779100925 * evals[58][2]];
    signal tmp_5523[3] <== CMul()(tmp_5522, challengesStage2[0]);
    signal tmp_5524[3] <== [tmp_5523[0] + evals[62][0], tmp_5523[1] + evals[62][1], tmp_5523[2] + evals[62][2]];
    signal tmp_5525[3] <== CMul()(tmp_5524, challengesStage2[0]);
    signal tmp_5526[3] <== [tmp_5525[0] + 1, tmp_5525[1], tmp_5525[2]];
    signal tmp_5527[3] <== [tmp_5526[0] + challengesStage2[1][0], tmp_5526[1] + challengesStage2[1][1], tmp_5526[2] + challengesStage2[1][2]];
    signal tmp_5528[3] <== [tmp_5527[0] - 1, tmp_5527[1], tmp_5527[2]];
    signal tmp_5529[3] <== [tmp_5528[0] + 1, tmp_5528[1], tmp_5528[2]];
    signal tmp_5530[3] <== CMul()(tmp_5521, tmp_5529);
    signal tmp_5531[3] <== [1279992132519201448 * evals[58][0], 1279992132519201448 * evals[58][1], 1279992132519201448 * evals[58][2]];
    signal tmp_5532[3] <== CMul()(tmp_5531, challengesStage2[0]);
    signal tmp_5533[3] <== [tmp_5532[0] + evals[63][0], tmp_5532[1] + evals[63][1], tmp_5532[2] + evals[63][2]];
    signal tmp_5534[3] <== CMul()(tmp_5533, challengesStage2[0]);
    signal tmp_5535[3] <== [tmp_5534[0] + 1, tmp_5534[1], tmp_5534[2]];
    signal tmp_5536[3] <== [tmp_5535[0] + challengesStage2[1][0], tmp_5535[1] + challengesStage2[1][1], tmp_5535[2] + challengesStage2[1][2]];
    signal tmp_5537[3] <== [tmp_5536[0] - 1, tmp_5536[1], tmp_5536[2]];
    signal tmp_5538[3] <== [tmp_5537[0] + 1, tmp_5537[1], tmp_5537[2]];
    signal tmp_5539[3] <== CMul()(tmp_5530, tmp_5538);
    signal tmp_5540[3] <== [8312008622371998338 * evals[58][0], 8312008622371998338 * evals[58][1], 8312008622371998338 * evals[58][2]];
    signal tmp_5541[3] <== CMul()(tmp_5540, challengesStage2[0]);
    signal tmp_5542[3] <== [tmp_5541[0] + evals[64][0], tmp_5541[1] + evals[64][1], tmp_5541[2] + evals[64][2]];
    signal tmp_5543[3] <== CMul()(tmp_5542, challengesStage2[0]);
    signal tmp_5544[3] <== [tmp_5543[0] + 1, tmp_5543[1], tmp_5543[2]];
    signal tmp_5545[3] <== [tmp_5544[0] + challengesStage2[1][0], tmp_5544[1] + challengesStage2[1][1], tmp_5544[2] + challengesStage2[1][2]];
    signal tmp_5546[3] <== [tmp_5545[0] - 1, tmp_5545[1], tmp_5545[2]];
    signal tmp_5547[3] <== [tmp_5546[0] + 1, tmp_5546[1], tmp_5546[2]];
    signal tmp_5548[3] <== CMul()(tmp_5539, tmp_5547);
    signal tmp_5549[3] <== [7781028390488215464 * evals[58][0], 7781028390488215464 * evals[58][1], 7781028390488215464 * evals[58][2]];
    signal tmp_5550[3] <== CMul()(tmp_5549, challengesStage2[0]);
    signal tmp_5551[3] <== [tmp_5550[0] + evals[65][0], tmp_5550[1] + evals[65][1], tmp_5550[2] + evals[65][2]];
    signal tmp_5552[3] <== CMul()(tmp_5551, challengesStage2[0]);
    signal tmp_5553[3] <== [tmp_5552[0] + 1, tmp_5552[1], tmp_5552[2]];
    signal tmp_5554[3] <== [tmp_5553[0] + challengesStage2[1][0], tmp_5553[1] + challengesStage2[1][1], tmp_5553[2] + challengesStage2[1][2]];
    signal tmp_5555[3] <== [tmp_5554[0] - 1, tmp_5554[1], tmp_5554[2]];
    signal tmp_5556[3] <== [tmp_5555[0] + 1, tmp_5555[1], tmp_5555[2]];
    signal tmp_5557[3] <== CMul()(tmp_5548, tmp_5556);
    signal tmp_5558[3] <== [11302600489504509467 * evals[58][0], 11302600489504509467 * evals[58][1], 11302600489504509467 * evals[58][2]];
    signal tmp_5559[3] <== CMul()(tmp_5558, challengesStage2[0]);
    signal tmp_5560[3] <== [tmp_5559[0] + evals[66][0], tmp_5559[1] + evals[66][1], tmp_5559[2] + evals[66][2]];
    signal tmp_5561[3] <== CMul()(tmp_5560, challengesStage2[0]);
    signal tmp_5562[3] <== [tmp_5561[0] + 1, tmp_5561[1], tmp_5561[2]];
    signal tmp_5563[3] <== [tmp_5562[0] + challengesStage2[1][0], tmp_5562[1] + challengesStage2[1][1], tmp_5562[2] + challengesStage2[1][2]];
    signal tmp_5564[3] <== [tmp_5563[0] - 1, tmp_5563[1], tmp_5563[2]];
    signal tmp_5565[3] <== [tmp_5564[0] + 1, tmp_5564[1], tmp_5564[2]];
    signal tmp_5566[3] <== CMul()(tmp_5557, tmp_5565);
    signal tmp_5567[3] <== CMul()(evals[97], tmp_5566);
    signal tmp_5568[3] <== CMul()(evals[15], challengesStage2[0]);
    signal tmp_5569[3] <== [tmp_5568[0] + evals[60][0], tmp_5568[1] + evals[60][1], tmp_5568[2] + evals[60][2]];
    signal tmp_5570[3] <== CMul()(tmp_5569, challengesStage2[0]);
    signal tmp_5571[3] <== [tmp_5570[0] + 1, tmp_5570[1], tmp_5570[2]];
    signal tmp_5572[3] <== [tmp_5571[0] + challengesStage2[1][0], tmp_5571[1] + challengesStage2[1][1], tmp_5571[2] + challengesStage2[1][2]];
    signal tmp_5573[3] <== [tmp_5572[0] - 1, tmp_5572[1], tmp_5572[2]];
    signal tmp_5574[3] <== [tmp_5573[0] + 1, tmp_5573[1], tmp_5573[2]];
    signal tmp_5575[3] <== CMul()(evals[16], challengesStage2[0]);
    signal tmp_5576[3] <== [tmp_5575[0] + evals[61][0], tmp_5575[1] + evals[61][1], tmp_5575[2] + evals[61][2]];
    signal tmp_5577[3] <== CMul()(tmp_5576, challengesStage2[0]);
    signal tmp_5578[3] <== [tmp_5577[0] + 1, tmp_5577[1], tmp_5577[2]];
    signal tmp_5579[3] <== [tmp_5578[0] + challengesStage2[1][0], tmp_5578[1] + challengesStage2[1][1], tmp_5578[2] + challengesStage2[1][2]];
    signal tmp_5580[3] <== [tmp_5579[0] - 1, tmp_5579[1], tmp_5579[2]];
    signal tmp_5581[3] <== [tmp_5580[0] + 1, tmp_5580[1], tmp_5580[2]];
    signal tmp_5582[3] <== CMul()(tmp_5574, tmp_5581);
    signal tmp_5583[3] <== CMul()(evals[17], challengesStage2[0]);
    signal tmp_5584[3] <== [tmp_5583[0] + evals[62][0], tmp_5583[1] + evals[62][1], tmp_5583[2] + evals[62][2]];
    signal tmp_5585[3] <== CMul()(tmp_5584, challengesStage2[0]);
    signal tmp_5586[3] <== [tmp_5585[0] + 1, tmp_5585[1], tmp_5585[2]];
    signal tmp_5587[3] <== [tmp_5586[0] + challengesStage2[1][0], tmp_5586[1] + challengesStage2[1][1], tmp_5586[2] + challengesStage2[1][2]];
    signal tmp_5588[3] <== [tmp_5587[0] - 1, tmp_5587[1], tmp_5587[2]];
    signal tmp_5589[3] <== [tmp_5588[0] + 1, tmp_5588[1], tmp_5588[2]];
    signal tmp_5590[3] <== CMul()(tmp_5582, tmp_5589);
    signal tmp_5591[3] <== CMul()(evals[18], challengesStage2[0]);
    signal tmp_5592[3] <== [tmp_5591[0] + evals[63][0], tmp_5591[1] + evals[63][1], tmp_5591[2] + evals[63][2]];
    signal tmp_5593[3] <== CMul()(tmp_5592, challengesStage2[0]);
    signal tmp_5594[3] <== [tmp_5593[0] + 1, tmp_5593[1], tmp_5593[2]];
    signal tmp_5595[3] <== [tmp_5594[0] + challengesStage2[1][0], tmp_5594[1] + challengesStage2[1][1], tmp_5594[2] + challengesStage2[1][2]];
    signal tmp_5596[3] <== [tmp_5595[0] - 1, tmp_5595[1], tmp_5595[2]];
    signal tmp_5597[3] <== [tmp_5596[0] + 1, tmp_5596[1], tmp_5596[2]];
    signal tmp_5598[3] <== CMul()(tmp_5590, tmp_5597);
    signal tmp_5599[3] <== CMul()(evals[19], challengesStage2[0]);
    signal tmp_5600[3] <== [tmp_5599[0] + evals[64][0], tmp_5599[1] + evals[64][1], tmp_5599[2] + evals[64][2]];
    signal tmp_5601[3] <== CMul()(tmp_5600, challengesStage2[0]);
    signal tmp_5602[3] <== [tmp_5601[0] + 1, tmp_5601[1], tmp_5601[2]];
    signal tmp_5603[3] <== [tmp_5602[0] + challengesStage2[1][0], tmp_5602[1] + challengesStage2[1][1], tmp_5602[2] + challengesStage2[1][2]];
    signal tmp_5604[3] <== [tmp_5603[0] - 1, tmp_5603[1], tmp_5603[2]];
    signal tmp_5605[3] <== [tmp_5604[0] + 1, tmp_5604[1], tmp_5604[2]];
    signal tmp_5606[3] <== CMul()(tmp_5598, tmp_5605);
    signal tmp_5607[3] <== CMul()(evals[20], challengesStage2[0]);
    signal tmp_5608[3] <== [tmp_5607[0] + evals[65][0], tmp_5607[1] + evals[65][1], tmp_5607[2] + evals[65][2]];
    signal tmp_5609[3] <== CMul()(tmp_5608, challengesStage2[0]);
    signal tmp_5610[3] <== [tmp_5609[0] + 1, tmp_5609[1], tmp_5609[2]];
    signal tmp_5611[3] <== [tmp_5610[0] + challengesStage2[1][0], tmp_5610[1] + challengesStage2[1][1], tmp_5610[2] + challengesStage2[1][2]];
    signal tmp_5612[3] <== [tmp_5611[0] - 1, tmp_5611[1], tmp_5611[2]];
    signal tmp_5613[3] <== [tmp_5612[0] + 1, tmp_5612[1], tmp_5612[2]];
    signal tmp_5614[3] <== CMul()(tmp_5606, tmp_5613);
    signal tmp_5615[3] <== CMul()(evals[21], challengesStage2[0]);
    signal tmp_5616[3] <== [tmp_5615[0] + evals[66][0], tmp_5615[1] + evals[66][1], tmp_5615[2] + evals[66][2]];
    signal tmp_5617[3] <== CMul()(tmp_5616, challengesStage2[0]);
    signal tmp_5618[3] <== [tmp_5617[0] + 1, tmp_5617[1], tmp_5617[2]];
    signal tmp_5619[3] <== [tmp_5618[0] + challengesStage2[1][0], tmp_5618[1] + challengesStage2[1][1], tmp_5618[2] + challengesStage2[1][2]];
    signal tmp_5620[3] <== [tmp_5619[0] - 1, tmp_5619[1], tmp_5619[2]];
    signal tmp_5621[3] <== [tmp_5620[0] + 1, tmp_5620[1], tmp_5620[2]];
    signal tmp_5622[3] <== CMul()(tmp_5614, tmp_5621);
    signal tmp_5623[3] <== CMul()(evals[22], challengesStage2[0]);
    signal tmp_5624[3] <== [tmp_5623[0] + evals[67][0], tmp_5623[1] + evals[67][1], tmp_5623[2] + evals[67][2]];
    signal tmp_5625[3] <== CMul()(tmp_5624, challengesStage2[0]);
    signal tmp_5626[3] <== [tmp_5625[0] + 1, tmp_5625[1], tmp_5625[2]];
    signal tmp_5627[3] <== [tmp_5626[0] + challengesStage2[1][0], tmp_5626[1] + challengesStage2[1][1], tmp_5626[2] + challengesStage2[1][2]];
    signal tmp_5628[3] <== [tmp_5627[0] - 1, tmp_5627[1], tmp_5627[2]];
    signal tmp_5629[3] <== [tmp_5628[0] + 1, tmp_5628[1], tmp_5628[2]];
    signal tmp_5630[3] <== CMul()(tmp_5622, tmp_5629);
    signal tmp_5631[3] <== [tmp_5567[0] - tmp_5630[0], tmp_5567[1] - tmp_5630[1], tmp_5567[2] - tmp_5630[2]];
    signal tmp_5632[3] <== [tmp_5505[0] + tmp_5631[0], tmp_5505[1] + tmp_5631[1], tmp_5505[2] + tmp_5631[2]];
    signal tmp_5633[3] <== CMul()(challengeQ, tmp_5632);
    signal tmp_5634[3] <== [4549350404001778198 * evals[58][0], 4549350404001778198 * evals[58][1], 4549350404001778198 * evals[58][2]];
    signal tmp_5635[3] <== CMul()(tmp_5634, challengesStage2[0]);
    signal tmp_5636[3] <== [tmp_5635[0] + evals[67][0], tmp_5635[1] + evals[67][1], tmp_5635[2] + evals[67][2]];
    signal tmp_5637[3] <== CMul()(tmp_5636, challengesStage2[0]);
    signal tmp_5638[3] <== [tmp_5637[0] + 1, tmp_5637[1], tmp_5637[2]];
    signal tmp_5639[3] <== [tmp_5638[0] + challengesStage2[1][0], tmp_5638[1] + challengesStage2[1][1], tmp_5638[2] + challengesStage2[1][2]];
    signal tmp_5640[3] <== [tmp_5639[0] - 1, tmp_5639[1], tmp_5639[2]];
    signal tmp_5641[3] <== [tmp_5640[0] + 1, tmp_5640[1], tmp_5640[2]];
    signal tmp_5642[3] <== [3688660304411827445 * evals[58][0], 3688660304411827445 * evals[58][1], 3688660304411827445 * evals[58][2]];
    signal tmp_5643[3] <== CMul()(tmp_5642, challengesStage2[0]);
    signal tmp_5644[3] <== [tmp_5643[0] + evals[68][0], tmp_5643[1] + evals[68][1], tmp_5643[2] + evals[68][2]];
    signal tmp_5645[3] <== CMul()(tmp_5644, challengesStage2[0]);
    signal tmp_5646[3] <== [tmp_5645[0] + 1, tmp_5645[1], tmp_5645[2]];
    signal tmp_5647[3] <== [tmp_5646[0] + challengesStage2[1][0], tmp_5646[1] + challengesStage2[1][1], tmp_5646[2] + challengesStage2[1][2]];
    signal tmp_5648[3] <== [tmp_5647[0] - 1, tmp_5647[1], tmp_5647[2]];
    signal tmp_5649[3] <== [tmp_5648[0] + 1, tmp_5648[1], tmp_5648[2]];
    signal tmp_5650[3] <== CMul()(tmp_5641, tmp_5649);
    signal tmp_5651[3] <== [16725109960945739746 * evals[58][0], 16725109960945739746 * evals[58][1], 16725109960945739746 * evals[58][2]];
    signal tmp_5652[3] <== CMul()(tmp_5651, challengesStage2[0]);
    signal tmp_5653[3] <== [tmp_5652[0] + evals[69][0], tmp_5652[1] + evals[69][1], tmp_5652[2] + evals[69][2]];
    signal tmp_5654[3] <== CMul()(tmp_5653, challengesStage2[0]);
    signal tmp_5655[3] <== [tmp_5654[0] + 1, tmp_5654[1], tmp_5654[2]];
    signal tmp_5656[3] <== [tmp_5655[0] + challengesStage2[1][0], tmp_5655[1] + challengesStage2[1][1], tmp_5655[2] + challengesStage2[1][2]];
    signal tmp_5657[3] <== [tmp_5656[0] - 1, tmp_5656[1], tmp_5656[2]];
    signal tmp_5658[3] <== [tmp_5657[0] + 1, tmp_5657[1], tmp_5657[2]];
    signal tmp_5659[3] <== CMul()(tmp_5650, tmp_5658);
    signal tmp_5660[3] <== [16538725463549498621 * evals[58][0], 16538725463549498621 * evals[58][1], 16538725463549498621 * evals[58][2]];
    signal tmp_5661[3] <== CMul()(tmp_5660, challengesStage2[0]);
    signal tmp_5662[3] <== [tmp_5661[0] + evals[70][0], tmp_5661[1] + evals[70][1], tmp_5661[2] + evals[70][2]];
    signal tmp_5663[3] <== CMul()(tmp_5662, challengesStage2[0]);
    signal tmp_5664[3] <== [tmp_5663[0] + 1, tmp_5663[1], tmp_5663[2]];
    signal tmp_5665[3] <== [tmp_5664[0] + challengesStage2[1][0], tmp_5664[1] + challengesStage2[1][1], tmp_5664[2] + challengesStage2[1][2]];
    signal tmp_5666[3] <== [tmp_5665[0] - 1, tmp_5665[1], tmp_5665[2]];
    signal tmp_5667[3] <== [tmp_5666[0] + 1, tmp_5666[1], tmp_5666[2]];
    signal tmp_5668[3] <== CMul()(tmp_5659, tmp_5667);
    signal tmp_5669[3] <== [12756200801261202346 * evals[58][0], 12756200801261202346 * evals[58][1], 12756200801261202346 * evals[58][2]];
    signal tmp_5670[3] <== CMul()(tmp_5669, challengesStage2[0]);
    signal tmp_5671[3] <== [tmp_5670[0] + evals[71][0], tmp_5670[1] + evals[71][1], tmp_5670[2] + evals[71][2]];
    signal tmp_5672[3] <== CMul()(tmp_5671, challengesStage2[0]);
    signal tmp_5673[3] <== [tmp_5672[0] + 1, tmp_5672[1], tmp_5672[2]];
    signal tmp_5674[3] <== [tmp_5673[0] + challengesStage2[1][0], tmp_5673[1] + challengesStage2[1][1], tmp_5673[2] + challengesStage2[1][2]];
    signal tmp_5675[3] <== [tmp_5674[0] - 1, tmp_5674[1], tmp_5674[2]];
    signal tmp_5676[3] <== [tmp_5675[0] + 1, tmp_5675[1], tmp_5675[2]];
    signal tmp_5677[3] <== CMul()(tmp_5668, tmp_5676);
    signal tmp_5678[3] <== [15099809066790865939 * evals[58][0], 15099809066790865939 * evals[58][1], 15099809066790865939 * evals[58][2]];
    signal tmp_5679[3] <== CMul()(tmp_5678, challengesStage2[0]);
    signal tmp_5680[3] <== [tmp_5679[0] + evals[72][0], tmp_5679[1] + evals[72][1], tmp_5679[2] + evals[72][2]];
    signal tmp_5681[3] <== CMul()(tmp_5680, challengesStage2[0]);
    signal tmp_5682[3] <== [tmp_5681[0] + 1, tmp_5681[1], tmp_5681[2]];
    signal tmp_5683[3] <== [tmp_5682[0] + challengesStage2[1][0], tmp_5682[1] + challengesStage2[1][1], tmp_5682[2] + challengesStage2[1][2]];
    signal tmp_5684[3] <== [tmp_5683[0] - 1, tmp_5683[1], tmp_5683[2]];
    signal tmp_5685[3] <== [tmp_5684[0] + 1, tmp_5684[1], tmp_5684[2]];
    signal tmp_5686[3] <== CMul()(tmp_5677, tmp_5685);
    signal tmp_5687[3] <== [17214954929431464349 * evals[58][0], 17214954929431464349 * evals[58][1], 17214954929431464349 * evals[58][2]];
    signal tmp_5688[3] <== CMul()(tmp_5687, challengesStage2[0]);
    signal tmp_5689[3] <== [tmp_5688[0] + evals[73][0], tmp_5688[1] + evals[73][1], tmp_5688[2] + evals[73][2]];
    signal tmp_5690[3] <== CMul()(tmp_5689, challengesStage2[0]);
    signal tmp_5691[3] <== [tmp_5690[0] + 1, tmp_5690[1], tmp_5690[2]];
    signal tmp_5692[3] <== [tmp_5691[0] + challengesStage2[1][0], tmp_5691[1] + challengesStage2[1][1], tmp_5691[2] + challengesStage2[1][2]];
    signal tmp_5693[3] <== [tmp_5692[0] - 1, tmp_5692[1], tmp_5692[2]];
    signal tmp_5694[3] <== [tmp_5693[0] + 1, tmp_5693[1], tmp_5693[2]];
    signal tmp_5695[3] <== CMul()(tmp_5686, tmp_5694);
    signal tmp_5696[3] <== CMul()(evals[98], tmp_5695);
    signal tmp_5697[3] <== CMul()(evals[23], challengesStage2[0]);
    signal tmp_5698[3] <== [tmp_5697[0] + evals[68][0], tmp_5697[1] + evals[68][1], tmp_5697[2] + evals[68][2]];
    signal tmp_5699[3] <== CMul()(tmp_5698, challengesStage2[0]);
    signal tmp_5700[3] <== [tmp_5699[0] + 1, tmp_5699[1], tmp_5699[2]];
    signal tmp_5701[3] <== [tmp_5700[0] + challengesStage2[1][0], tmp_5700[1] + challengesStage2[1][1], tmp_5700[2] + challengesStage2[1][2]];
    signal tmp_5702[3] <== [tmp_5701[0] - 1, tmp_5701[1], tmp_5701[2]];
    signal tmp_5703[3] <== [tmp_5702[0] + 1, tmp_5702[1], tmp_5702[2]];
    signal tmp_5704[3] <== CMul()(evals[97], tmp_5703);
    signal tmp_5705[3] <== CMul()(evals[24], challengesStage2[0]);
    signal tmp_5706[3] <== [tmp_5705[0] + evals[69][0], tmp_5705[1] + evals[69][1], tmp_5705[2] + evals[69][2]];
    signal tmp_5707[3] <== CMul()(tmp_5706, challengesStage2[0]);
    signal tmp_5708[3] <== [tmp_5707[0] + 1, tmp_5707[1], tmp_5707[2]];
    signal tmp_5709[3] <== [tmp_5708[0] + challengesStage2[1][0], tmp_5708[1] + challengesStage2[1][1], tmp_5708[2] + challengesStage2[1][2]];
    signal tmp_5710[3] <== [tmp_5709[0] - 1, tmp_5709[1], tmp_5709[2]];
    signal tmp_5711[3] <== [tmp_5710[0] + 1, tmp_5710[1], tmp_5710[2]];
    signal tmp_5712[3] <== CMul()(tmp_5704, tmp_5711);
    signal tmp_5713[3] <== CMul()(evals[25], challengesStage2[0]);
    signal tmp_5714[3] <== [tmp_5713[0] + evals[70][0], tmp_5713[1] + evals[70][1], tmp_5713[2] + evals[70][2]];
    signal tmp_5715[3] <== CMul()(tmp_5714, challengesStage2[0]);
    signal tmp_5716[3] <== [tmp_5715[0] + 1, tmp_5715[1], tmp_5715[2]];
    signal tmp_5717[3] <== [tmp_5716[0] + challengesStage2[1][0], tmp_5716[1] + challengesStage2[1][1], tmp_5716[2] + challengesStage2[1][2]];
    signal tmp_5718[3] <== [tmp_5717[0] - 1, tmp_5717[1], tmp_5717[2]];
    signal tmp_5719[3] <== [tmp_5718[0] + 1, tmp_5718[1], tmp_5718[2]];
    signal tmp_5720[3] <== CMul()(tmp_5712, tmp_5719);
    signal tmp_5721[3] <== CMul()(evals[26], challengesStage2[0]);
    signal tmp_5722[3] <== [tmp_5721[0] + evals[71][0], tmp_5721[1] + evals[71][1], tmp_5721[2] + evals[71][2]];
    signal tmp_5723[3] <== CMul()(tmp_5722, challengesStage2[0]);
    signal tmp_5724[3] <== [tmp_5723[0] + 1, tmp_5723[1], tmp_5723[2]];
    signal tmp_5725[3] <== [tmp_5724[0] + challengesStage2[1][0], tmp_5724[1] + challengesStage2[1][1], tmp_5724[2] + challengesStage2[1][2]];
    signal tmp_5726[3] <== [tmp_5725[0] - 1, tmp_5725[1], tmp_5725[2]];
    signal tmp_5727[3] <== [tmp_5726[0] + 1, tmp_5726[1], tmp_5726[2]];
    signal tmp_5728[3] <== CMul()(tmp_5720, tmp_5727);
    signal tmp_5729[3] <== CMul()(evals[27], challengesStage2[0]);
    signal tmp_5730[3] <== [tmp_5729[0] + evals[72][0], tmp_5729[1] + evals[72][1], tmp_5729[2] + evals[72][2]];
    signal tmp_5731[3] <== CMul()(tmp_5730, challengesStage2[0]);
    signal tmp_5732[3] <== [tmp_5731[0] + 1, tmp_5731[1], tmp_5731[2]];
    signal tmp_5733[3] <== [tmp_5732[0] + challengesStage2[1][0], tmp_5732[1] + challengesStage2[1][1], tmp_5732[2] + challengesStage2[1][2]];
    signal tmp_5734[3] <== [tmp_5733[0] - 1, tmp_5733[1], tmp_5733[2]];
    signal tmp_5735[3] <== [tmp_5734[0] + 1, tmp_5734[1], tmp_5734[2]];
    signal tmp_5736[3] <== CMul()(tmp_5728, tmp_5735);
    signal tmp_5737[3] <== CMul()(evals[28], challengesStage2[0]);
    signal tmp_5738[3] <== [tmp_5737[0] + evals[73][0], tmp_5737[1] + evals[73][1], tmp_5737[2] + evals[73][2]];
    signal tmp_5739[3] <== CMul()(tmp_5738, challengesStage2[0]);
    signal tmp_5740[3] <== [tmp_5739[0] + 1, tmp_5739[1], tmp_5739[2]];
    signal tmp_5741[3] <== [tmp_5740[0] + challengesStage2[1][0], tmp_5740[1] + challengesStage2[1][1], tmp_5740[2] + challengesStage2[1][2]];
    signal tmp_5742[3] <== [tmp_5741[0] - 1, tmp_5741[1], tmp_5741[2]];
    signal tmp_5743[3] <== [tmp_5742[0] + 1, tmp_5742[1], tmp_5742[2]];
    signal tmp_5744[3] <== CMul()(tmp_5736, tmp_5743);
    signal tmp_5745[3] <== CMul()(evals[29], challengesStage2[0]);
    signal tmp_5746[3] <== [tmp_5745[0] + evals[74][0], tmp_5745[1] + evals[74][1], tmp_5745[2] + evals[74][2]];
    signal tmp_5747[3] <== CMul()(tmp_5746, challengesStage2[0]);
    signal tmp_5748[3] <== [tmp_5747[0] + 1, tmp_5747[1], tmp_5747[2]];
    signal tmp_5749[3] <== [tmp_5748[0] + challengesStage2[1][0], tmp_5748[1] + challengesStage2[1][1], tmp_5748[2] + challengesStage2[1][2]];
    signal tmp_5750[3] <== [tmp_5749[0] - 1, tmp_5749[1], tmp_5749[2]];
    signal tmp_5751[3] <== [tmp_5750[0] + 1, tmp_5750[1], tmp_5750[2]];
    signal tmp_5752[3] <== CMul()(tmp_5744, tmp_5751);
    signal tmp_5753[3] <== [tmp_5696[0] - tmp_5752[0], tmp_5696[1] - tmp_5752[1], tmp_5696[2] - tmp_5752[2]];
    signal tmp_5754[3] <== [tmp_5633[0] + tmp_5753[0], tmp_5633[1] + tmp_5753[1], tmp_5633[2] + tmp_5753[2]];
    signal tmp_5755[3] <== CMul()(challengeQ, tmp_5754);
    signal tmp_5756[3] <== [11016800570561344835 * evals[58][0], 11016800570561344835 * evals[58][1], 11016800570561344835 * evals[58][2]];
    signal tmp_5757[3] <== CMul()(tmp_5756, challengesStage2[0]);
    signal tmp_5758[3] <== [tmp_5757[0] + evals[74][0], tmp_5757[1] + evals[74][1], tmp_5757[2] + evals[74][2]];
    signal tmp_5759[3] <== CMul()(tmp_5758, challengesStage2[0]);
    signal tmp_5760[3] <== [tmp_5759[0] + 1, tmp_5759[1], tmp_5759[2]];
    signal tmp_5761[3] <== [tmp_5760[0] + challengesStage2[1][0], tmp_5760[1] + challengesStage2[1][1], tmp_5760[2] + challengesStage2[1][2]];
    signal tmp_5762[3] <== [tmp_5761[0] - 1, tmp_5761[1], tmp_5761[2]];
    signal tmp_5763[3] <== [tmp_5762[0] + 1, tmp_5762[1], tmp_5762[2]];
    signal tmp_5764[3] <== [11274872323250451096 * evals[58][0], 11274872323250451096 * evals[58][1], 11274872323250451096 * evals[58][2]];
    signal tmp_5765[3] <== CMul()(tmp_5764, challengesStage2[0]);
    signal tmp_5766[3] <== [tmp_5765[0] + evals[75][0], tmp_5765[1] + evals[75][1], tmp_5765[2] + evals[75][2]];
    signal tmp_5767[3] <== CMul()(tmp_5766, challengesStage2[0]);
    signal tmp_5768[3] <== [tmp_5767[0] + 1, tmp_5767[1], tmp_5767[2]];
    signal tmp_5769[3] <== [tmp_5768[0] + challengesStage2[1][0], tmp_5768[1] + challengesStage2[1][1], tmp_5768[2] + challengesStage2[1][2]];
    signal tmp_5770[3] <== [tmp_5769[0] - 1, tmp_5769[1], tmp_5769[2]];
    signal tmp_5771[3] <== [tmp_5770[0] + 1, tmp_5770[1], tmp_5770[2]];
    signal tmp_5772[3] <== CMul()(tmp_5763, tmp_5771);
    signal tmp_5773[3] <== [6534114114080170934 * evals[58][0], 6534114114080170934 * evals[58][1], 6534114114080170934 * evals[58][2]];
    signal tmp_5774[3] <== CMul()(tmp_5773, challengesStage2[0]);
    signal tmp_5775[3] <== [tmp_5774[0] + evals[76][0], tmp_5774[1] + evals[76][1], tmp_5774[2] + evals[76][2]];
    signal tmp_5776[3] <== CMul()(tmp_5775, challengesStage2[0]);
    signal tmp_5777[3] <== [tmp_5776[0] + 1, tmp_5776[1], tmp_5776[2]];
    signal tmp_5778[3] <== [tmp_5777[0] + challengesStage2[1][0], tmp_5777[1] + challengesStage2[1][1], tmp_5777[2] + challengesStage2[1][2]];
    signal tmp_5779[3] <== [tmp_5778[0] - 1, tmp_5778[1], tmp_5778[2]];
    signal tmp_5780[3] <== [tmp_5779[0] + 1, tmp_5779[1], tmp_5779[2]];
    signal tmp_5781[3] <== CMul()(tmp_5772, tmp_5780);
    signal tmp_5782[3] <== [13047390008333835222 * evals[58][0], 13047390008333835222 * evals[58][1], 13047390008333835222 * evals[58][2]];
    signal tmp_5783[3] <== CMul()(tmp_5782, challengesStage2[0]);
    signal tmp_5784[3] <== [tmp_5783[0] + evals[77][0], tmp_5783[1] + evals[77][1], tmp_5783[2] + evals[77][2]];
    signal tmp_5785[3] <== CMul()(tmp_5784, challengesStage2[0]);
    signal tmp_5786[3] <== [tmp_5785[0] + 1, tmp_5785[1], tmp_5785[2]];
    signal tmp_5787[3] <== [tmp_5786[0] + challengesStage2[1][0], tmp_5786[1] + challengesStage2[1][1], tmp_5786[2] + challengesStage2[1][2]];
    signal tmp_5788[3] <== [tmp_5787[0] - 1, tmp_5787[1], tmp_5787[2]];
    signal tmp_5789[3] <== [tmp_5788[0] + 1, tmp_5788[1], tmp_5788[2]];
    signal tmp_5790[3] <== CMul()(tmp_5781, tmp_5789);
    signal tmp_5791[3] <== [11189528522318044176 * evals[58][0], 11189528522318044176 * evals[58][1], 11189528522318044176 * evals[58][2]];
    signal tmp_5792[3] <== CMul()(tmp_5791, challengesStage2[0]);
    signal tmp_5793[3] <== [tmp_5792[0] + evals[78][0], tmp_5792[1] + evals[78][1], tmp_5792[2] + evals[78][2]];
    signal tmp_5794[3] <== CMul()(tmp_5793, challengesStage2[0]);
    signal tmp_5795[3] <== [tmp_5794[0] + 1, tmp_5794[1], tmp_5794[2]];
    signal tmp_5796[3] <== [tmp_5795[0] + challengesStage2[1][0], tmp_5795[1] + challengesStage2[1][1], tmp_5795[2] + challengesStage2[1][2]];
    signal tmp_5797[3] <== [tmp_5796[0] - 1, tmp_5796[1], tmp_5796[2]];
    signal tmp_5798[3] <== [tmp_5797[0] + 1, tmp_5797[1], tmp_5797[2]];
    signal tmp_5799[3] <== CMul()(tmp_5790, tmp_5798);
    signal tmp_5800[3] <== [3320735505586735876 * evals[58][0], 3320735505586735876 * evals[58][1], 3320735505586735876 * evals[58][2]];
    signal tmp_5801[3] <== CMul()(tmp_5800, challengesStage2[0]);
    signal tmp_5802[3] <== [tmp_5801[0] + evals[79][0], tmp_5801[1] + evals[79][1], tmp_5801[2] + evals[79][2]];
    signal tmp_5803[3] <== CMul()(tmp_5802, challengesStage2[0]);
    signal tmp_5804[3] <== [tmp_5803[0] + 1, tmp_5803[1], tmp_5803[2]];
    signal tmp_5805[3] <== [tmp_5804[0] + challengesStage2[1][0], tmp_5804[1] + challengesStage2[1][1], tmp_5804[2] + challengesStage2[1][2]];
    signal tmp_5806[3] <== [tmp_5805[0] - 1, tmp_5805[1], tmp_5805[2]];
    signal tmp_5807[3] <== [tmp_5806[0] + 1, tmp_5806[1], tmp_5806[2]];
    signal tmp_5808[3] <== CMul()(tmp_5799, tmp_5807);
    signal tmp_5809[3] <== [7240278926970958133 * evals[58][0], 7240278926970958133 * evals[58][1], 7240278926970958133 * evals[58][2]];
    signal tmp_5810[3] <== CMul()(tmp_5809, challengesStage2[0]);
    signal tmp_5811[3] <== [tmp_5810[0] + evals[80][0], tmp_5810[1] + evals[80][1], tmp_5810[2] + evals[80][2]];
    signal tmp_5812[3] <== CMul()(tmp_5811, challengesStage2[0]);
    signal tmp_5813[3] <== [tmp_5812[0] + 1, tmp_5812[1], tmp_5812[2]];
    signal tmp_5814[3] <== [tmp_5813[0] + challengesStage2[1][0], tmp_5813[1] + challengesStage2[1][1], tmp_5813[2] + challengesStage2[1][2]];
    signal tmp_5815[3] <== [tmp_5814[0] - 1, tmp_5814[1], tmp_5814[2]];
    signal tmp_5816[3] <== [tmp_5815[0] + 1, tmp_5815[1], tmp_5815[2]];
    signal tmp_5817[3] <== CMul()(tmp_5808, tmp_5816);
    signal tmp_5818[3] <== CMul()(evals[99], tmp_5817);
    signal tmp_5819[3] <== CMul()(evals[30], challengesStage2[0]);
    signal tmp_5820[3] <== [tmp_5819[0] + evals[75][0], tmp_5819[1] + evals[75][1], tmp_5819[2] + evals[75][2]];
    signal tmp_5821[3] <== CMul()(tmp_5820, challengesStage2[0]);
    signal tmp_5822[3] <== [tmp_5821[0] + 1, tmp_5821[1], tmp_5821[2]];
    signal tmp_5823[3] <== [tmp_5822[0] + challengesStage2[1][0], tmp_5822[1] + challengesStage2[1][1], tmp_5822[2] + challengesStage2[1][2]];
    signal tmp_5824[3] <== [tmp_5823[0] - 1, tmp_5823[1], tmp_5823[2]];
    signal tmp_5825[3] <== [tmp_5824[0] + 1, tmp_5824[1], tmp_5824[2]];
    signal tmp_5826[3] <== CMul()(evals[98], tmp_5825);
    signal tmp_5827[3] <== CMul()(evals[31], challengesStage2[0]);
    signal tmp_5828[3] <== [tmp_5827[0] + evals[76][0], tmp_5827[1] + evals[76][1], tmp_5827[2] + evals[76][2]];
    signal tmp_5829[3] <== CMul()(tmp_5828, challengesStage2[0]);
    signal tmp_5830[3] <== [tmp_5829[0] + 1, tmp_5829[1], tmp_5829[2]];
    signal tmp_5831[3] <== [tmp_5830[0] + challengesStage2[1][0], tmp_5830[1] + challengesStage2[1][1], tmp_5830[2] + challengesStage2[1][2]];
    signal tmp_5832[3] <== [tmp_5831[0] - 1, tmp_5831[1], tmp_5831[2]];
    signal tmp_5833[3] <== [tmp_5832[0] + 1, tmp_5832[1], tmp_5832[2]];
    signal tmp_5834[3] <== CMul()(tmp_5826, tmp_5833);
    signal tmp_5835[3] <== CMul()(evals[32], challengesStage2[0]);
    signal tmp_5836[3] <== [tmp_5835[0] + evals[77][0], tmp_5835[1] + evals[77][1], tmp_5835[2] + evals[77][2]];
    signal tmp_5837[3] <== CMul()(tmp_5836, challengesStage2[0]);
    signal tmp_5838[3] <== [tmp_5837[0] + 1, tmp_5837[1], tmp_5837[2]];
    signal tmp_5839[3] <== [tmp_5838[0] + challengesStage2[1][0], tmp_5838[1] + challengesStage2[1][1], tmp_5838[2] + challengesStage2[1][2]];
    signal tmp_5840[3] <== [tmp_5839[0] - 1, tmp_5839[1], tmp_5839[2]];
    signal tmp_5841[3] <== [tmp_5840[0] + 1, tmp_5840[1], tmp_5840[2]];
    signal tmp_5842[3] <== CMul()(tmp_5834, tmp_5841);
    signal tmp_5843[3] <== CMul()(evals[33], challengesStage2[0]);
    signal tmp_5844[3] <== [tmp_5843[0] + evals[78][0], tmp_5843[1] + evals[78][1], tmp_5843[2] + evals[78][2]];
    signal tmp_5845[3] <== CMul()(tmp_5844, challengesStage2[0]);
    signal tmp_5846[3] <== [tmp_5845[0] + 1, tmp_5845[1], tmp_5845[2]];
    signal tmp_5847[3] <== [tmp_5846[0] + challengesStage2[1][0], tmp_5846[1] + challengesStage2[1][1], tmp_5846[2] + challengesStage2[1][2]];
    signal tmp_5848[3] <== [tmp_5847[0] - 1, tmp_5847[1], tmp_5847[2]];
    signal tmp_5849[3] <== [tmp_5848[0] + 1, tmp_5848[1], tmp_5848[2]];
    signal tmp_5850[3] <== CMul()(tmp_5842, tmp_5849);
    signal tmp_5851[3] <== CMul()(evals[34], challengesStage2[0]);
    signal tmp_5852[3] <== [tmp_5851[0] + evals[79][0], tmp_5851[1] + evals[79][1], tmp_5851[2] + evals[79][2]];
    signal tmp_5853[3] <== CMul()(tmp_5852, challengesStage2[0]);
    signal tmp_5854[3] <== [tmp_5853[0] + 1, tmp_5853[1], tmp_5853[2]];
    signal tmp_5855[3] <== [tmp_5854[0] + challengesStage2[1][0], tmp_5854[1] + challengesStage2[1][1], tmp_5854[2] + challengesStage2[1][2]];
    signal tmp_5856[3] <== [tmp_5855[0] - 1, tmp_5855[1], tmp_5855[2]];
    signal tmp_5857[3] <== [tmp_5856[0] + 1, tmp_5856[1], tmp_5856[2]];
    signal tmp_5858[3] <== CMul()(tmp_5850, tmp_5857);
    signal tmp_5859[3] <== CMul()(evals[35], challengesStage2[0]);
    signal tmp_5860[3] <== [tmp_5859[0] + evals[80][0], tmp_5859[1] + evals[80][1], tmp_5859[2] + evals[80][2]];
    signal tmp_5861[3] <== CMul()(tmp_5860, challengesStage2[0]);
    signal tmp_5862[3] <== [tmp_5861[0] + 1, tmp_5861[1], tmp_5861[2]];
    signal tmp_5863[3] <== [tmp_5862[0] + challengesStage2[1][0], tmp_5862[1] + challengesStage2[1][1], tmp_5862[2] + challengesStage2[1][2]];
    signal tmp_5864[3] <== [tmp_5863[0] - 1, tmp_5863[1], tmp_5863[2]];
    signal tmp_5865[3] <== [tmp_5864[0] + 1, tmp_5864[1], tmp_5864[2]];
    signal tmp_5866[3] <== CMul()(tmp_5858, tmp_5865);
    signal tmp_5867[3] <== CMul()(evals[36], challengesStage2[0]);
    signal tmp_5868[3] <== [tmp_5867[0] + evals[81][0], tmp_5867[1] + evals[81][1], tmp_5867[2] + evals[81][2]];
    signal tmp_5869[3] <== CMul()(tmp_5868, challengesStage2[0]);
    signal tmp_5870[3] <== [tmp_5869[0] + 1, tmp_5869[1], tmp_5869[2]];
    signal tmp_5871[3] <== [tmp_5870[0] + challengesStage2[1][0], tmp_5870[1] + challengesStage2[1][1], tmp_5870[2] + challengesStage2[1][2]];
    signal tmp_5872[3] <== [tmp_5871[0] - 1, tmp_5871[1], tmp_5871[2]];
    signal tmp_5873[3] <== [tmp_5872[0] + 1, tmp_5872[1], tmp_5872[2]];
    signal tmp_5874[3] <== CMul()(tmp_5866, tmp_5873);
    signal tmp_5875[3] <== [tmp_5818[0] - tmp_5874[0], tmp_5818[1] - tmp_5874[1], tmp_5818[2] - tmp_5874[2]];
    signal tmp_5876[3] <== [tmp_5755[0] + tmp_5875[0], tmp_5755[1] + tmp_5875[1], tmp_5755[2] + tmp_5875[2]];
    signal tmp_5877[3] <== CMul()(challengeQ, tmp_5876);
    signal tmp_5878[3] <== [8246665031048405574 * evals[58][0], 8246665031048405574 * evals[58][1], 8246665031048405574 * evals[58][2]];
    signal tmp_5879[3] <== CMul()(tmp_5878, challengesStage2[0]);
    signal tmp_5880[3] <== [tmp_5879[0] + evals[81][0], tmp_5879[1] + evals[81][1], tmp_5879[2] + evals[81][2]];
    signal tmp_5881[3] <== CMul()(tmp_5880, challengesStage2[0]);
    signal tmp_5882[3] <== [tmp_5881[0] + 1, tmp_5881[1], tmp_5881[2]];
    signal tmp_5883[3] <== [tmp_5882[0] + challengesStage2[1][0], tmp_5882[1] + challengesStage2[1][1], tmp_5882[2] + challengesStage2[1][2]];
    signal tmp_5884[3] <== [tmp_5883[0] - 1, tmp_5883[1], tmp_5883[2]];
    signal tmp_5885[3] <== [tmp_5884[0] + 1, tmp_5884[1], tmp_5884[2]];
    signal tmp_5886[3] <== [12693612801792047873 * evals[58][0], 12693612801792047873 * evals[58][1], 12693612801792047873 * evals[58][2]];
    signal tmp_5887[3] <== CMul()(tmp_5886, challengesStage2[0]);
    signal tmp_5888[3] <== [tmp_5887[0] + evals[82][0], tmp_5887[1] + evals[82][1], tmp_5887[2] + evals[82][2]];
    signal tmp_5889[3] <== CMul()(tmp_5888, challengesStage2[0]);
    signal tmp_5890[3] <== [tmp_5889[0] + 1, tmp_5889[1], tmp_5889[2]];
    signal tmp_5891[3] <== [tmp_5890[0] + challengesStage2[1][0], tmp_5890[1] + challengesStage2[1][1], tmp_5890[2] + challengesStage2[1][2]];
    signal tmp_5892[3] <== [tmp_5891[0] - 1, tmp_5891[1], tmp_5891[2]];
    signal tmp_5893[3] <== [tmp_5892[0] + 1, tmp_5892[1], tmp_5892[2]];
    signal tmp_5894[3] <== CMul()(tmp_5885, tmp_5893);
    signal tmp_5895[3] <== [9404062091095256088 * evals[58][0], 9404062091095256088 * evals[58][1], 9404062091095256088 * evals[58][2]];
    signal tmp_5896[3] <== CMul()(tmp_5895, challengesStage2[0]);
    signal tmp_5897[3] <== [tmp_5896[0] + evals[83][0], tmp_5896[1] + evals[83][1], tmp_5896[2] + evals[83][2]];
    signal tmp_5898[3] <== CMul()(tmp_5897, challengesStage2[0]);
    signal tmp_5899[3] <== [tmp_5898[0] + 1, tmp_5898[1], tmp_5898[2]];
    signal tmp_5900[3] <== [tmp_5899[0] + challengesStage2[1][0], tmp_5899[1] + challengesStage2[1][1], tmp_5899[2] + challengesStage2[1][2]];
    signal tmp_5901[3] <== [tmp_5900[0] - 1, tmp_5900[1], tmp_5900[2]];
    signal tmp_5902[3] <== [tmp_5901[0] + 1, tmp_5901[1], tmp_5901[2]];
    signal tmp_5903[3] <== CMul()(tmp_5894, tmp_5902);
    signal tmp_5904[3] <== CMul()(evals[96], tmp_5903);
    signal tmp_5905[3] <== [1 - evals[59][0], -evals[59][1], -evals[59][2]];
    signal tmp_5906[3] <== CMul()(evals[14], tmp_5905);
    signal tmp_5907[3] <== [tmp_5906[0] + evals[59][0], tmp_5906[1] + evals[59][1], tmp_5906[2] + evals[59][2]];
    signal tmp_5908[3] <== CMul()(evals[37], challengesStage2[0]);
    signal tmp_5909[3] <== [tmp_5908[0] + evals[82][0], tmp_5908[1] + evals[82][1], tmp_5908[2] + evals[82][2]];
    signal tmp_5910[3] <== CMul()(tmp_5909, challengesStage2[0]);
    signal tmp_5911[3] <== [tmp_5910[0] + 1, tmp_5910[1], tmp_5910[2]];
    signal tmp_5912[3] <== [tmp_5911[0] + challengesStage2[1][0], tmp_5911[1] + challengesStage2[1][1], tmp_5911[2] + challengesStage2[1][2]];
    signal tmp_5913[3] <== [tmp_5912[0] - 1, tmp_5912[1], tmp_5912[2]];
    signal tmp_5914[3] <== [tmp_5913[0] + 1, tmp_5913[1], tmp_5913[2]];
    signal tmp_5915[3] <== CMul()(evals[99], tmp_5914);
    signal tmp_5916[3] <== CMul()(evals[38], challengesStage2[0]);
    signal tmp_5917[3] <== [tmp_5916[0] + evals[83][0], tmp_5916[1] + evals[83][1], tmp_5916[2] + evals[83][2]];
    signal tmp_5918[3] <== CMul()(tmp_5917, challengesStage2[0]);
    signal tmp_5919[3] <== [tmp_5918[0] + 1, tmp_5918[1], tmp_5918[2]];
    signal tmp_5920[3] <== [tmp_5919[0] + challengesStage2[1][0], tmp_5919[1] + challengesStage2[1][1], tmp_5919[2] + challengesStage2[1][2]];
    signal tmp_5921[3] <== [tmp_5920[0] - 1, tmp_5920[1], tmp_5920[2]];
    signal tmp_5922[3] <== [tmp_5921[0] + 1, tmp_5921[1], tmp_5921[2]];
    signal tmp_5923[3] <== CMul()(tmp_5915, tmp_5922);
    signal tmp_5924[3] <== CMul()(tmp_5907, tmp_5923);
    signal tmp_5925[3] <== [tmp_5904[0] - tmp_5924[0], tmp_5904[1] - tmp_5924[1], tmp_5904[2] - tmp_5924[2]];
    signal tmp_5926[3] <== [tmp_5877[0] + tmp_5925[0], tmp_5877[1] + tmp_5925[1], tmp_5877[2] + tmp_5925[2]];
    signal tmp_5927[3] <== CMul()(challengeQ, tmp_5926);
    signal tmp_5928[3] <== [1 - evals[96][0], -evals[96][1], -evals[96][2]];
    signal tmp_5929[3] <== CMul()(evals[109], tmp_5928);
    signal tmp_2964[3] <== [tmp_5927[0] + tmp_5929[0], tmp_5927[1] + tmp_5929[1], tmp_5927[2] + tmp_5929[2]];
    signal tmp_5930[3] <== CMul()(tmp_2964, Zh);

    signal xAcc[7][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[6][3]; // Stores the evaluations of Q_i
    signal qAcc[7][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 7; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[100+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[16]);
            qStep[i-1] <== CMul()(xAcc[i], evals[100+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i, which is stored at tmp_5930 
    enable * (tmp_5930[0] - qAcc[6][0]) === 0;
    enable * (tmp_5930[1] - qAcc[6][1]) === 0;
    enable * (tmp_5930[2] - qAcc[6][2]) === 0;
}

/*  Calculate FRI polinomial */
template CalculateFRIPolValue0() {
    signal input {binary} queriesFRI[20];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[136][3];
 
    signal input cm1[36];
 
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[45];
    
    signal output queryVals[3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
 
    mapValues.vals1 <== cm1;
 
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;
    signal xacc[20];
    xacc[0] <== queriesFRI[0]*(7 * roots(20)-7) + 7;
    for (var i=1; i<20; i++) {
        xacc[i] <== xacc[i-1] * ( queriesFRI[i]*(roots(20 - i) - 1) +1);
    }

    signal xDivXSubXi[4][3];

    signal den0inv[3] <== CInv()([xacc[19] - 15139302138664925958 * challengeXi[0], - 15139302138664925958 * challengeXi[1], - 15139302138664925958 * challengeXi[2]]);
    xDivXSubXi[0] <== [xacc[19] * den0inv[0], xacc[19] * den0inv[1],  xacc[19] * den0inv[2]];
    signal den1inv[3] <== CInv()([xacc[19] - 1 * challengeXi[0], - 1 * challengeXi[1], - 1 * challengeXi[2]]);
    xDivXSubXi[1] <== [xacc[19] * den1inv[0], xacc[19] * den1inv[1],  xacc[19] * den1inv[2]];
    signal den2inv[3] <== CInv()([xacc[19] - 5718075921287398682 * challengeXi[0], - 5718075921287398682 * challengeXi[1], - 5718075921287398682 * challengeXi[2]]);
    xDivXSubXi[2] <== [xacc[19] * den2inv[0], xacc[19] * den2inv[1],  xacc[19] * den2inv[2]];
    signal den3inv[3] <== CInv()([xacc[19] - 6016448678728204869 * challengeXi[0], - 6016448678728204869 * challengeXi[1], - 6016448678728204869 * challengeXi[2]]);
    xDivXSubXi[3] <== [xacc[19] * den3inv[0], xacc[19] * den3inv[1],  xacc[19] * den3inv[2]];

    signal tmp_0[3] <== [consts[34] - evals[0][0], -evals[0][1], -evals[0][2]];
    signal tmp_1[3] <== CMul()(tmp_0, challengesFRI[1]);
    signal tmp_2[3] <== [consts[35] - evals[1][0], -evals[1][1], -evals[1][2]];
    signal tmp_3[3] <== [tmp_1[0] + tmp_2[0], tmp_1[1] + tmp_2[1], tmp_1[2] + tmp_2[2]];
    signal tmp_4[3] <== CMul()(tmp_3, challengesFRI[1]);
    signal tmp_5[3] <== [mapValues.cm1_24 - evals[2][0], -evals[2][1], -evals[2][2]];
    signal tmp_6[3] <== [tmp_4[0] + tmp_5[0], tmp_4[1] + tmp_5[1], tmp_4[2] + tmp_5[2]];
    signal tmp_7[3] <== CMul()(tmp_6, challengesFRI[1]);
    signal tmp_8[3] <== [mapValues.cm1_25 - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_9[3] <== [tmp_7[0] + tmp_8[0], tmp_7[1] + tmp_8[1], tmp_7[2] + tmp_8[2]];
    signal tmp_10[3] <== CMul()(tmp_9, challengesFRI[1]);
    signal tmp_11[3] <== [mapValues.cm1_26 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_12[3] <== [tmp_10[0] + tmp_11[0], tmp_10[1] + tmp_11[1], tmp_10[2] + tmp_11[2]];
    signal tmp_13[3] <== CMul()(tmp_12, challengesFRI[1]);
    signal tmp_14[3] <== [mapValues.cm1_27 - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_15[3] <== [tmp_13[0] + tmp_14[0], tmp_13[1] + tmp_14[1], tmp_13[2] + tmp_14[2]];
    signal tmp_16[3] <== CMul()(tmp_15, challengesFRI[1]);
    signal tmp_17[3] <== [mapValues.cm1_28 - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_18[3] <== [tmp_16[0] + tmp_17[0], tmp_16[1] + tmp_17[1], tmp_16[2] + tmp_17[2]];
    signal tmp_19[3] <== CMul()(tmp_18, challengesFRI[1]);
    signal tmp_20[3] <== [mapValues.cm1_29 - evals[7][0], -evals[7][1], -evals[7][2]];
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_21, challengesFRI[1]);
    signal tmp_23[3] <== [mapValues.cm1_30 - evals[8][0], -evals[8][1], -evals[8][2]];
    signal tmp_24[3] <== [tmp_22[0] + tmp_23[0], tmp_22[1] + tmp_23[1], tmp_22[2] + tmp_23[2]];
    signal tmp_25[3] <== CMul()(tmp_24, challengesFRI[1]);
    signal tmp_26[3] <== [mapValues.cm1_31 - evals[9][0], -evals[9][1], -evals[9][2]];
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_27, challengesFRI[1]);
    signal tmp_29[3] <== [mapValues.cm1_32 - evals[10][0], -evals[10][1], -evals[10][2]];
    signal tmp_30[3] <== [tmp_28[0] + tmp_29[0], tmp_28[1] + tmp_29[1], tmp_28[2] + tmp_29[2]];
    signal tmp_31[3] <== CMul()(tmp_30, challengesFRI[1]);
    signal tmp_32[3] <== [mapValues.cm1_33 - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_33[3] <== [tmp_31[0] + tmp_32[0], tmp_31[1] + tmp_32[1], tmp_31[2] + tmp_32[2]];
    signal tmp_34[3] <== CMul()(tmp_33, challengesFRI[1]);
    signal tmp_35[3] <== [mapValues.cm1_34 - evals[12][0], -evals[12][1], -evals[12][2]];
    signal tmp_36[3] <== [tmp_34[0] + tmp_35[0], tmp_34[1] + tmp_35[1], tmp_34[2] + tmp_35[2]];
    signal tmp_37[3] <== CMul()(tmp_36, challengesFRI[1]);
    signal tmp_38[3] <== [mapValues.cm1_35 - evals[13][0], -evals[13][1], -evals[13][2]];
    signal tmp_39[3] <== [tmp_37[0] + tmp_38[0], tmp_37[1] + tmp_38[1], tmp_37[2] + tmp_38[2]];
    signal tmp_40[3] <== CMul()(tmp_39, challengesFRI[1]);
    signal tmp_41[3] <== [mapValues.cm2_0[0] - evals[14][0], mapValues.cm2_0[1] - evals[14][1], mapValues.cm2_0[2] - evals[14][2]];
    signal tmp_42[3] <== [tmp_40[0] + tmp_41[0], tmp_40[1] + tmp_41[1], tmp_40[2] + tmp_41[2]];
    signal tmp_43[3] <== CMul()(tmp_42, xDivXSubXi[0]);
    signal tmp_44[3] <== CMul()(challengesFRI[0], tmp_43);
    signal tmp_45[3] <== [consts[0] - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_46[3] <== CMul()(tmp_45, challengesFRI[1]);
    signal tmp_47[3] <== [consts[1] - evals[16][0], -evals[16][1], -evals[16][2]];
    signal tmp_48[3] <== [tmp_46[0] + tmp_47[0], tmp_46[1] + tmp_47[1], tmp_46[2] + tmp_47[2]];
    signal tmp_49[3] <== CMul()(tmp_48, challengesFRI[1]);
    signal tmp_50[3] <== [consts[2] - evals[17][0], -evals[17][1], -evals[17][2]];
    signal tmp_51[3] <== [tmp_49[0] + tmp_50[0], tmp_49[1] + tmp_50[1], tmp_49[2] + tmp_50[2]];
    signal tmp_52[3] <== CMul()(tmp_51, challengesFRI[1]);
    signal tmp_53[3] <== [consts[3] - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_54[3] <== [tmp_52[0] + tmp_53[0], tmp_52[1] + tmp_53[1], tmp_52[2] + tmp_53[2]];
    signal tmp_55[3] <== CMul()(tmp_54, challengesFRI[1]);
    signal tmp_56[3] <== [consts[4] - evals[19][0], -evals[19][1], -evals[19][2]];
    signal tmp_57[3] <== [tmp_55[0] + tmp_56[0], tmp_55[1] + tmp_56[1], tmp_55[2] + tmp_56[2]];
    signal tmp_58[3] <== CMul()(tmp_57, challengesFRI[1]);
    signal tmp_59[3] <== [consts[5] - evals[20][0], -evals[20][1], -evals[20][2]];
    signal tmp_60[3] <== [tmp_58[0] + tmp_59[0], tmp_58[1] + tmp_59[1], tmp_58[2] + tmp_59[2]];
    signal tmp_61[3] <== CMul()(tmp_60, challengesFRI[1]);
    signal tmp_62[3] <== [consts[6] - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_63[3] <== [tmp_61[0] + tmp_62[0], tmp_61[1] + tmp_62[1], tmp_61[2] + tmp_62[2]];
    signal tmp_64[3] <== CMul()(tmp_63, challengesFRI[1]);
    signal tmp_65[3] <== [consts[7] - evals[22][0], -evals[22][1], -evals[22][2]];
    signal tmp_66[3] <== [tmp_64[0] + tmp_65[0], tmp_64[1] + tmp_65[1], tmp_64[2] + tmp_65[2]];
    signal tmp_67[3] <== CMul()(tmp_66, challengesFRI[1]);
    signal tmp_68[3] <== [consts[8] - evals[23][0], -evals[23][1], -evals[23][2]];
    signal tmp_69[3] <== [tmp_67[0] + tmp_68[0], tmp_67[1] + tmp_68[1], tmp_67[2] + tmp_68[2]];
    signal tmp_70[3] <== CMul()(tmp_69, challengesFRI[1]);
    signal tmp_71[3] <== [consts[9] - evals[24][0], -evals[24][1], -evals[24][2]];
    signal tmp_72[3] <== [tmp_70[0] + tmp_71[0], tmp_70[1] + tmp_71[1], tmp_70[2] + tmp_71[2]];
    signal tmp_73[3] <== CMul()(tmp_72, challengesFRI[1]);
    signal tmp_74[3] <== [consts[10] - evals[25][0], -evals[25][1], -evals[25][2]];
    signal tmp_75[3] <== [tmp_73[0] + tmp_74[0], tmp_73[1] + tmp_74[1], tmp_73[2] + tmp_74[2]];
    signal tmp_76[3] <== CMul()(tmp_75, challengesFRI[1]);
    signal tmp_77[3] <== [consts[11] - evals[26][0], -evals[26][1], -evals[26][2]];
    signal tmp_78[3] <== [tmp_76[0] + tmp_77[0], tmp_76[1] + tmp_77[1], tmp_76[2] + tmp_77[2]];
    signal tmp_79[3] <== CMul()(tmp_78, challengesFRI[1]);
    signal tmp_80[3] <== [consts[12] - evals[27][0], -evals[27][1], -evals[27][2]];
    signal tmp_81[3] <== [tmp_79[0] + tmp_80[0], tmp_79[1] + tmp_80[1], tmp_79[2] + tmp_80[2]];
    signal tmp_82[3] <== CMul()(tmp_81, challengesFRI[1]);
    signal tmp_83[3] <== [consts[13] - evals[28][0], -evals[28][1], -evals[28][2]];
    signal tmp_84[3] <== [tmp_82[0] + tmp_83[0], tmp_82[1] + tmp_83[1], tmp_82[2] + tmp_83[2]];
    signal tmp_85[3] <== CMul()(tmp_84, challengesFRI[1]);
    signal tmp_86[3] <== [consts[14] - evals[29][0], -evals[29][1], -evals[29][2]];
    signal tmp_87[3] <== [tmp_85[0] + tmp_86[0], tmp_85[1] + tmp_86[1], tmp_85[2] + tmp_86[2]];
    signal tmp_88[3] <== CMul()(tmp_87, challengesFRI[1]);
    signal tmp_89[3] <== [consts[15] - evals[30][0], -evals[30][1], -evals[30][2]];
    signal tmp_90[3] <== [tmp_88[0] + tmp_89[0], tmp_88[1] + tmp_89[1], tmp_88[2] + tmp_89[2]];
    signal tmp_91[3] <== CMul()(tmp_90, challengesFRI[1]);
    signal tmp_92[3] <== [consts[16] - evals[31][0], -evals[31][1], -evals[31][2]];
    signal tmp_93[3] <== [tmp_91[0] + tmp_92[0], tmp_91[1] + tmp_92[1], tmp_91[2] + tmp_92[2]];
    signal tmp_94[3] <== CMul()(tmp_93, challengesFRI[1]);
    signal tmp_95[3] <== [consts[17] - evals[32][0], -evals[32][1], -evals[32][2]];
    signal tmp_96[3] <== [tmp_94[0] + tmp_95[0], tmp_94[1] + tmp_95[1], tmp_94[2] + tmp_95[2]];
    signal tmp_97[3] <== CMul()(tmp_96, challengesFRI[1]);
    signal tmp_98[3] <== [consts[18] - evals[33][0], -evals[33][1], -evals[33][2]];
    signal tmp_99[3] <== [tmp_97[0] + tmp_98[0], tmp_97[1] + tmp_98[1], tmp_97[2] + tmp_98[2]];
    signal tmp_100[3] <== CMul()(tmp_99, challengesFRI[1]);
    signal tmp_101[3] <== [consts[19] - evals[34][0], -evals[34][1], -evals[34][2]];
    signal tmp_102[3] <== [tmp_100[0] + tmp_101[0], tmp_100[1] + tmp_101[1], tmp_100[2] + tmp_101[2]];
    signal tmp_103[3] <== CMul()(tmp_102, challengesFRI[1]);
    signal tmp_104[3] <== [consts[20] - evals[35][0], -evals[35][1], -evals[35][2]];
    signal tmp_105[3] <== [tmp_103[0] + tmp_104[0], tmp_103[1] + tmp_104[1], tmp_103[2] + tmp_104[2]];
    signal tmp_106[3] <== CMul()(tmp_105, challengesFRI[1]);
    signal tmp_107[3] <== [consts[21] - evals[36][0], -evals[36][1], -evals[36][2]];
    signal tmp_108[3] <== [tmp_106[0] + tmp_107[0], tmp_106[1] + tmp_107[1], tmp_106[2] + tmp_107[2]];
    signal tmp_109[3] <== CMul()(tmp_108, challengesFRI[1]);
    signal tmp_110[3] <== [consts[22] - evals[37][0], -evals[37][1], -evals[37][2]];
    signal tmp_111[3] <== [tmp_109[0] + tmp_110[0], tmp_109[1] + tmp_110[1], tmp_109[2] + tmp_110[2]];
    signal tmp_112[3] <== CMul()(tmp_111, challengesFRI[1]);
    signal tmp_113[3] <== [consts[23] - evals[38][0], -evals[38][1], -evals[38][2]];
    signal tmp_114[3] <== [tmp_112[0] + tmp_113[0], tmp_112[1] + tmp_113[1], tmp_112[2] + tmp_113[2]];
    signal tmp_115[3] <== CMul()(tmp_114, challengesFRI[1]);
    signal tmp_116[3] <== [consts[24] - evals[39][0], -evals[39][1], -evals[39][2]];
    signal tmp_117[3] <== [tmp_115[0] + tmp_116[0], tmp_115[1] + tmp_116[1], tmp_115[2] + tmp_116[2]];
    signal tmp_118[3] <== CMul()(tmp_117, challengesFRI[1]);
    signal tmp_119[3] <== [consts[25] - evals[40][0], -evals[40][1], -evals[40][2]];
    signal tmp_120[3] <== [tmp_118[0] + tmp_119[0], tmp_118[1] + tmp_119[1], tmp_118[2] + tmp_119[2]];
    signal tmp_121[3] <== CMul()(tmp_120, challengesFRI[1]);
    signal tmp_122[3] <== [consts[26] - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_123[3] <== [tmp_121[0] + tmp_122[0], tmp_121[1] + tmp_122[1], tmp_121[2] + tmp_122[2]];
    signal tmp_124[3] <== CMul()(tmp_123, challengesFRI[1]);
    signal tmp_125[3] <== [consts[27] - evals[42][0], -evals[42][1], -evals[42][2]];
    signal tmp_126[3] <== [tmp_124[0] + tmp_125[0], tmp_124[1] + tmp_125[1], tmp_124[2] + tmp_125[2]];
    signal tmp_127[3] <== CMul()(tmp_126, challengesFRI[1]);
    signal tmp_128[3] <== [consts[28] - evals[43][0], -evals[43][1], -evals[43][2]];
    signal tmp_129[3] <== [tmp_127[0] + tmp_128[0], tmp_127[1] + tmp_128[1], tmp_127[2] + tmp_128[2]];
    signal tmp_130[3] <== CMul()(tmp_129, challengesFRI[1]);
    signal tmp_131[3] <== [consts[29] - evals[44][0], -evals[44][1], -evals[44][2]];
    signal tmp_132[3] <== [tmp_130[0] + tmp_131[0], tmp_130[1] + tmp_131[1], tmp_130[2] + tmp_131[2]];
    signal tmp_133[3] <== CMul()(tmp_132, challengesFRI[1]);
    signal tmp_134[3] <== [consts[30] - evals[45][0], -evals[45][1], -evals[45][2]];
    signal tmp_135[3] <== [tmp_133[0] + tmp_134[0], tmp_133[1] + tmp_134[1], tmp_133[2] + tmp_134[2]];
    signal tmp_136[3] <== CMul()(tmp_135, challengesFRI[1]);
    signal tmp_137[3] <== [consts[31] - evals[46][0], -evals[46][1], -evals[46][2]];
    signal tmp_138[3] <== [tmp_136[0] + tmp_137[0], tmp_136[1] + tmp_137[1], tmp_136[2] + tmp_137[2]];
    signal tmp_139[3] <== CMul()(tmp_138, challengesFRI[1]);
    signal tmp_140[3] <== [consts[32] - evals[47][0], -evals[47][1], -evals[47][2]];
    signal tmp_141[3] <== [tmp_139[0] + tmp_140[0], tmp_139[1] + tmp_140[1], tmp_139[2] + tmp_140[2]];
    signal tmp_142[3] <== CMul()(tmp_141, challengesFRI[1]);
    signal tmp_143[3] <== [consts[33] - evals[48][0], -evals[48][1], -evals[48][2]];
    signal tmp_144[3] <== [tmp_142[0] + tmp_143[0], tmp_142[1] + tmp_143[1], tmp_142[2] + tmp_143[2]];
    signal tmp_145[3] <== CMul()(tmp_144, challengesFRI[1]);
    signal tmp_146[3] <== [consts[34] - evals[49][0], -evals[49][1], -evals[49][2]];
    signal tmp_147[3] <== [tmp_145[0] + tmp_146[0], tmp_145[1] + tmp_146[1], tmp_145[2] + tmp_146[2]];
    signal tmp_148[3] <== CMul()(tmp_147, challengesFRI[1]);
    signal tmp_149[3] <== [consts[35] - evals[50][0], -evals[50][1], -evals[50][2]];
    signal tmp_150[3] <== [tmp_148[0] + tmp_149[0], tmp_148[1] + tmp_149[1], tmp_148[2] + tmp_149[2]];
    signal tmp_151[3] <== CMul()(tmp_150, challengesFRI[1]);
    signal tmp_152[3] <== [consts[36] - evals[51][0], -evals[51][1], -evals[51][2]];
    signal tmp_153[3] <== [tmp_151[0] + tmp_152[0], tmp_151[1] + tmp_152[1], tmp_151[2] + tmp_152[2]];
    signal tmp_154[3] <== CMul()(tmp_153, challengesFRI[1]);
    signal tmp_155[3] <== [consts[37] - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_156[3] <== [tmp_154[0] + tmp_155[0], tmp_154[1] + tmp_155[1], tmp_154[2] + tmp_155[2]];
    signal tmp_157[3] <== CMul()(tmp_156, challengesFRI[1]);
    signal tmp_158[3] <== [consts[38] - evals[53][0], -evals[53][1], -evals[53][2]];
    signal tmp_159[3] <== [tmp_157[0] + tmp_158[0], tmp_157[1] + tmp_158[1], tmp_157[2] + tmp_158[2]];
    signal tmp_160[3] <== CMul()(tmp_159, challengesFRI[1]);
    signal tmp_161[3] <== [consts[39] - evals[54][0], -evals[54][1], -evals[54][2]];
    signal tmp_162[3] <== [tmp_160[0] + tmp_161[0], tmp_160[1] + tmp_161[1], tmp_160[2] + tmp_161[2]];
    signal tmp_163[3] <== CMul()(tmp_162, challengesFRI[1]);
    signal tmp_164[3] <== [consts[40] - evals[55][0], -evals[55][1], -evals[55][2]];
    signal tmp_165[3] <== [tmp_163[0] + tmp_164[0], tmp_163[1] + tmp_164[1], tmp_163[2] + tmp_164[2]];
    signal tmp_166[3] <== CMul()(tmp_165, challengesFRI[1]);
    signal tmp_167[3] <== [consts[41] - evals[56][0], -evals[56][1], -evals[56][2]];
    signal tmp_168[3] <== [tmp_166[0] + tmp_167[0], tmp_166[1] + tmp_167[1], tmp_166[2] + tmp_167[2]];
    signal tmp_169[3] <== CMul()(tmp_168, challengesFRI[1]);
    signal tmp_170[3] <== [consts[42] - evals[57][0], -evals[57][1], -evals[57][2]];
    signal tmp_171[3] <== [tmp_169[0] + tmp_170[0], tmp_169[1] + tmp_170[1], tmp_169[2] + tmp_170[2]];
    signal tmp_172[3] <== CMul()(tmp_171, challengesFRI[1]);
    signal tmp_173[3] <== [consts[43] - evals[58][0], -evals[58][1], -evals[58][2]];
    signal tmp_174[3] <== [tmp_172[0] + tmp_173[0], tmp_172[1] + tmp_173[1], tmp_172[2] + tmp_173[2]];
    signal tmp_175[3] <== CMul()(tmp_174, challengesFRI[1]);
    signal tmp_176[3] <== [consts[44] - evals[59][0], -evals[59][1], -evals[59][2]];
    signal tmp_177[3] <== [tmp_175[0] + tmp_176[0], tmp_175[1] + tmp_176[1], tmp_175[2] + tmp_176[2]];
    signal tmp_178[3] <== CMul()(tmp_177, challengesFRI[1]);
    signal tmp_179[3] <== [mapValues.cm1_0 - evals[60][0], -evals[60][1], -evals[60][2]];
    signal tmp_180[3] <== [tmp_178[0] + tmp_179[0], tmp_178[1] + tmp_179[1], tmp_178[2] + tmp_179[2]];
    signal tmp_181[3] <== CMul()(tmp_180, challengesFRI[1]);
    signal tmp_182[3] <== [mapValues.cm1_1 - evals[61][0], -evals[61][1], -evals[61][2]];
    signal tmp_183[3] <== [tmp_181[0] + tmp_182[0], tmp_181[1] + tmp_182[1], tmp_181[2] + tmp_182[2]];
    signal tmp_184[3] <== CMul()(tmp_183, challengesFRI[1]);
    signal tmp_185[3] <== [mapValues.cm1_2 - evals[62][0], -evals[62][1], -evals[62][2]];
    signal tmp_186[3] <== [tmp_184[0] + tmp_185[0], tmp_184[1] + tmp_185[1], tmp_184[2] + tmp_185[2]];
    signal tmp_187[3] <== CMul()(tmp_186, challengesFRI[1]);
    signal tmp_188[3] <== [mapValues.cm1_3 - evals[63][0], -evals[63][1], -evals[63][2]];
    signal tmp_189[3] <== [tmp_187[0] + tmp_188[0], tmp_187[1] + tmp_188[1], tmp_187[2] + tmp_188[2]];
    signal tmp_190[3] <== CMul()(tmp_189, challengesFRI[1]);
    signal tmp_191[3] <== [mapValues.cm1_4 - evals[64][0], -evals[64][1], -evals[64][2]];
    signal tmp_192[3] <== [tmp_190[0] + tmp_191[0], tmp_190[1] + tmp_191[1], tmp_190[2] + tmp_191[2]];
    signal tmp_193[3] <== CMul()(tmp_192, challengesFRI[1]);
    signal tmp_194[3] <== [mapValues.cm1_5 - evals[65][0], -evals[65][1], -evals[65][2]];
    signal tmp_195[3] <== [tmp_193[0] + tmp_194[0], tmp_193[1] + tmp_194[1], tmp_193[2] + tmp_194[2]];
    signal tmp_196[3] <== CMul()(tmp_195, challengesFRI[1]);
    signal tmp_197[3] <== [mapValues.cm1_6 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_198[3] <== [tmp_196[0] + tmp_197[0], tmp_196[1] + tmp_197[1], tmp_196[2] + tmp_197[2]];
    signal tmp_199[3] <== CMul()(tmp_198, challengesFRI[1]);
    signal tmp_200[3] <== [mapValues.cm1_7 - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_201[3] <== [tmp_199[0] + tmp_200[0], tmp_199[1] + tmp_200[1], tmp_199[2] + tmp_200[2]];
    signal tmp_202[3] <== CMul()(tmp_201, challengesFRI[1]);
    signal tmp_203[3] <== [mapValues.cm1_8 - evals[68][0], -evals[68][1], -evals[68][2]];
    signal tmp_204[3] <== [tmp_202[0] + tmp_203[0], tmp_202[1] + tmp_203[1], tmp_202[2] + tmp_203[2]];
    signal tmp_205[3] <== CMul()(tmp_204, challengesFRI[1]);
    signal tmp_206[3] <== [mapValues.cm1_9 - evals[69][0], -evals[69][1], -evals[69][2]];
    signal tmp_207[3] <== [tmp_205[0] + tmp_206[0], tmp_205[1] + tmp_206[1], tmp_205[2] + tmp_206[2]];
    signal tmp_208[3] <== CMul()(tmp_207, challengesFRI[1]);
    signal tmp_209[3] <== [mapValues.cm1_10 - evals[70][0], -evals[70][1], -evals[70][2]];
    signal tmp_210[3] <== [tmp_208[0] + tmp_209[0], tmp_208[1] + tmp_209[1], tmp_208[2] + tmp_209[2]];
    signal tmp_211[3] <== CMul()(tmp_210, challengesFRI[1]);
    signal tmp_212[3] <== [mapValues.cm1_11 - evals[71][0], -evals[71][1], -evals[71][2]];
    signal tmp_213[3] <== [tmp_211[0] + tmp_212[0], tmp_211[1] + tmp_212[1], tmp_211[2] + tmp_212[2]];
    signal tmp_214[3] <== CMul()(tmp_213, challengesFRI[1]);
    signal tmp_215[3] <== [mapValues.cm1_12 - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_216[3] <== [tmp_214[0] + tmp_215[0], tmp_214[1] + tmp_215[1], tmp_214[2] + tmp_215[2]];
    signal tmp_217[3] <== CMul()(tmp_216, challengesFRI[1]);
    signal tmp_218[3] <== [mapValues.cm1_13 - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_219[3] <== [tmp_217[0] + tmp_218[0], tmp_217[1] + tmp_218[1], tmp_217[2] + tmp_218[2]];
    signal tmp_220[3] <== CMul()(tmp_219, challengesFRI[1]);
    signal tmp_221[3] <== [mapValues.cm1_14 - evals[74][0], -evals[74][1], -evals[74][2]];
    signal tmp_222[3] <== [tmp_220[0] + tmp_221[0], tmp_220[1] + tmp_221[1], tmp_220[2] + tmp_221[2]];
    signal tmp_223[3] <== CMul()(tmp_222, challengesFRI[1]);
    signal tmp_224[3] <== [mapValues.cm1_15 - evals[75][0], -evals[75][1], -evals[75][2]];
    signal tmp_225[3] <== [tmp_223[0] + tmp_224[0], tmp_223[1] + tmp_224[1], tmp_223[2] + tmp_224[2]];
    signal tmp_226[3] <== CMul()(tmp_225, challengesFRI[1]);
    signal tmp_227[3] <== [mapValues.cm1_16 - evals[76][0], -evals[76][1], -evals[76][2]];
    signal tmp_228[3] <== [tmp_226[0] + tmp_227[0], tmp_226[1] + tmp_227[1], tmp_226[2] + tmp_227[2]];
    signal tmp_229[3] <== CMul()(tmp_228, challengesFRI[1]);
    signal tmp_230[3] <== [mapValues.cm1_17 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_231[3] <== [tmp_229[0] + tmp_230[0], tmp_229[1] + tmp_230[1], tmp_229[2] + tmp_230[2]];
    signal tmp_232[3] <== CMul()(tmp_231, challengesFRI[1]);
    signal tmp_233[3] <== [mapValues.cm1_18 - evals[78][0], -evals[78][1], -evals[78][2]];
    signal tmp_234[3] <== [tmp_232[0] + tmp_233[0], tmp_232[1] + tmp_233[1], tmp_232[2] + tmp_233[2]];
    signal tmp_235[3] <== CMul()(tmp_234, challengesFRI[1]);
    signal tmp_236[3] <== [mapValues.cm1_19 - evals[79][0], -evals[79][1], -evals[79][2]];
    signal tmp_237[3] <== [tmp_235[0] + tmp_236[0], tmp_235[1] + tmp_236[1], tmp_235[2] + tmp_236[2]];
    signal tmp_238[3] <== CMul()(tmp_237, challengesFRI[1]);
    signal tmp_239[3] <== [mapValues.cm1_20 - evals[80][0], -evals[80][1], -evals[80][2]];
    signal tmp_240[3] <== [tmp_238[0] + tmp_239[0], tmp_238[1] + tmp_239[1], tmp_238[2] + tmp_239[2]];
    signal tmp_241[3] <== CMul()(tmp_240, challengesFRI[1]);
    signal tmp_242[3] <== [mapValues.cm1_21 - evals[81][0], -evals[81][1], -evals[81][2]];
    signal tmp_243[3] <== [tmp_241[0] + tmp_242[0], tmp_241[1] + tmp_242[1], tmp_241[2] + tmp_242[2]];
    signal tmp_244[3] <== CMul()(tmp_243, challengesFRI[1]);
    signal tmp_245[3] <== [mapValues.cm1_22 - evals[82][0], -evals[82][1], -evals[82][2]];
    signal tmp_246[3] <== [tmp_244[0] + tmp_245[0], tmp_244[1] + tmp_245[1], tmp_244[2] + tmp_245[2]];
    signal tmp_247[3] <== CMul()(tmp_246, challengesFRI[1]);
    signal tmp_248[3] <== [mapValues.cm1_23 - evals[83][0], -evals[83][1], -evals[83][2]];
    signal tmp_249[3] <== [tmp_247[0] + tmp_248[0], tmp_247[1] + tmp_248[1], tmp_247[2] + tmp_248[2]];
    signal tmp_250[3] <== CMul()(tmp_249, challengesFRI[1]);
    signal tmp_251[3] <== [mapValues.cm1_24 - evals[84][0], -evals[84][1], -evals[84][2]];
    signal tmp_252[3] <== [tmp_250[0] + tmp_251[0], tmp_250[1] + tmp_251[1], tmp_250[2] + tmp_251[2]];
    signal tmp_253[3] <== CMul()(tmp_252, challengesFRI[1]);
    signal tmp_254[3] <== [mapValues.cm1_25 - evals[85][0], -evals[85][1], -evals[85][2]];
    signal tmp_255[3] <== [tmp_253[0] + tmp_254[0], tmp_253[1] + tmp_254[1], tmp_253[2] + tmp_254[2]];
    signal tmp_256[3] <== CMul()(tmp_255, challengesFRI[1]);
    signal tmp_257[3] <== [mapValues.cm1_26 - evals[86][0], -evals[86][1], -evals[86][2]];
    signal tmp_258[3] <== [tmp_256[0] + tmp_257[0], tmp_256[1] + tmp_257[1], tmp_256[2] + tmp_257[2]];
    signal tmp_259[3] <== CMul()(tmp_258, challengesFRI[1]);
    signal tmp_260[3] <== [mapValues.cm1_27 - evals[87][0], -evals[87][1], -evals[87][2]];
    signal tmp_261[3] <== [tmp_259[0] + tmp_260[0], tmp_259[1] + tmp_260[1], tmp_259[2] + tmp_260[2]];
    signal tmp_262[3] <== CMul()(tmp_261, challengesFRI[1]);
    signal tmp_263[3] <== [mapValues.cm1_28 - evals[88][0], -evals[88][1], -evals[88][2]];
    signal tmp_264[3] <== [tmp_262[0] + tmp_263[0], tmp_262[1] + tmp_263[1], tmp_262[2] + tmp_263[2]];
    signal tmp_265[3] <== CMul()(tmp_264, challengesFRI[1]);
    signal tmp_266[3] <== [mapValues.cm1_29 - evals[89][0], -evals[89][1], -evals[89][2]];
    signal tmp_267[3] <== [tmp_265[0] + tmp_266[0], tmp_265[1] + tmp_266[1], tmp_265[2] + tmp_266[2]];
    signal tmp_268[3] <== CMul()(tmp_267, challengesFRI[1]);
    signal tmp_269[3] <== [mapValues.cm1_30 - evals[90][0], -evals[90][1], -evals[90][2]];
    signal tmp_270[3] <== [tmp_268[0] + tmp_269[0], tmp_268[1] + tmp_269[1], tmp_268[2] + tmp_269[2]];
    signal tmp_271[3] <== CMul()(tmp_270, challengesFRI[1]);
    signal tmp_272[3] <== [mapValues.cm1_31 - evals[91][0], -evals[91][1], -evals[91][2]];
    signal tmp_273[3] <== [tmp_271[0] + tmp_272[0], tmp_271[1] + tmp_272[1], tmp_271[2] + tmp_272[2]];
    signal tmp_274[3] <== CMul()(tmp_273, challengesFRI[1]);
    signal tmp_275[3] <== [mapValues.cm1_32 - evals[92][0], -evals[92][1], -evals[92][2]];
    signal tmp_276[3] <== [tmp_274[0] + tmp_275[0], tmp_274[1] + tmp_275[1], tmp_274[2] + tmp_275[2]];
    signal tmp_277[3] <== CMul()(tmp_276, challengesFRI[1]);
    signal tmp_278[3] <== [mapValues.cm1_33 - evals[93][0], -evals[93][1], -evals[93][2]];
    signal tmp_279[3] <== [tmp_277[0] + tmp_278[0], tmp_277[1] + tmp_278[1], tmp_277[2] + tmp_278[2]];
    signal tmp_280[3] <== CMul()(tmp_279, challengesFRI[1]);
    signal tmp_281[3] <== [mapValues.cm1_34 - evals[94][0], -evals[94][1], -evals[94][2]];
    signal tmp_282[3] <== [tmp_280[0] + tmp_281[0], tmp_280[1] + tmp_281[1], tmp_280[2] + tmp_281[2]];
    signal tmp_283[3] <== CMul()(tmp_282, challengesFRI[1]);
    signal tmp_284[3] <== [mapValues.cm1_35 - evals[95][0], -evals[95][1], -evals[95][2]];
    signal tmp_285[3] <== [tmp_283[0] + tmp_284[0], tmp_283[1] + tmp_284[1], tmp_283[2] + tmp_284[2]];
    signal tmp_286[3] <== CMul()(tmp_285, challengesFRI[1]);
    signal tmp_287[3] <== [mapValues.cm2_0[0] - evals[96][0], mapValues.cm2_0[1] - evals[96][1], mapValues.cm2_0[2] - evals[96][2]];
    signal tmp_288[3] <== [tmp_286[0] + tmp_287[0], tmp_286[1] + tmp_287[1], tmp_286[2] + tmp_287[2]];
    signal tmp_289[3] <== CMul()(tmp_288, challengesFRI[1]);
    signal tmp_290[3] <== [mapValues.cm2_1[0] - evals[97][0], mapValues.cm2_1[1] - evals[97][1], mapValues.cm2_1[2] - evals[97][2]];
    signal tmp_291[3] <== [tmp_289[0] + tmp_290[0], tmp_289[1] + tmp_290[1], tmp_289[2] + tmp_290[2]];
    signal tmp_292[3] <== CMul()(tmp_291, challengesFRI[1]);
    signal tmp_293[3] <== [mapValues.cm2_2[0] - evals[98][0], mapValues.cm2_2[1] - evals[98][1], mapValues.cm2_2[2] - evals[98][2]];
    signal tmp_294[3] <== [tmp_292[0] + tmp_293[0], tmp_292[1] + tmp_293[1], tmp_292[2] + tmp_293[2]];
    signal tmp_295[3] <== CMul()(tmp_294, challengesFRI[1]);
    signal tmp_296[3] <== [mapValues.cm2_3[0] - evals[99][0], mapValues.cm2_3[1] - evals[99][1], mapValues.cm2_3[2] - evals[99][2]];
    signal tmp_297[3] <== [tmp_295[0] + tmp_296[0], tmp_295[1] + tmp_296[1], tmp_295[2] + tmp_296[2]];
    signal tmp_298[3] <== CMul()(tmp_297, challengesFRI[1]);
    signal tmp_299[3] <== [mapValues.cm3_0[0] - evals[100][0], mapValues.cm3_0[1] - evals[100][1], mapValues.cm3_0[2] - evals[100][2]];
    signal tmp_300[3] <== [tmp_298[0] + tmp_299[0], tmp_298[1] + tmp_299[1], tmp_298[2] + tmp_299[2]];
    signal tmp_301[3] <== CMul()(tmp_300, challengesFRI[1]);
    signal tmp_302[3] <== [mapValues.cm3_1[0] - evals[101][0], mapValues.cm3_1[1] - evals[101][1], mapValues.cm3_1[2] - evals[101][2]];
    signal tmp_303[3] <== [tmp_301[0] + tmp_302[0], tmp_301[1] + tmp_302[1], tmp_301[2] + tmp_302[2]];
    signal tmp_304[3] <== CMul()(tmp_303, challengesFRI[1]);
    signal tmp_305[3] <== [mapValues.cm3_2[0] - evals[102][0], mapValues.cm3_2[1] - evals[102][1], mapValues.cm3_2[2] - evals[102][2]];
    signal tmp_306[3] <== [tmp_304[0] + tmp_305[0], tmp_304[1] + tmp_305[1], tmp_304[2] + tmp_305[2]];
    signal tmp_307[3] <== CMul()(tmp_306, challengesFRI[1]);
    signal tmp_308[3] <== [mapValues.cm3_3[0] - evals[103][0], mapValues.cm3_3[1] - evals[103][1], mapValues.cm3_3[2] - evals[103][2]];
    signal tmp_309[3] <== [tmp_307[0] + tmp_308[0], tmp_307[1] + tmp_308[1], tmp_307[2] + tmp_308[2]];
    signal tmp_310[3] <== CMul()(tmp_309, challengesFRI[1]);
    signal tmp_311[3] <== [mapValues.cm3_4[0] - evals[104][0], mapValues.cm3_4[1] - evals[104][1], mapValues.cm3_4[2] - evals[104][2]];
    signal tmp_312[3] <== [tmp_310[0] + tmp_311[0], tmp_310[1] + tmp_311[1], tmp_310[2] + tmp_311[2]];
    signal tmp_313[3] <== CMul()(tmp_312, challengesFRI[1]);
    signal tmp_314[3] <== [mapValues.cm3_5[0] - evals[105][0], mapValues.cm3_5[1] - evals[105][1], mapValues.cm3_5[2] - evals[105][2]];
    signal tmp_315[3] <== [tmp_313[0] + tmp_314[0], tmp_313[1] + tmp_314[1], tmp_313[2] + tmp_314[2]];
    signal tmp_316[3] <== CMul()(tmp_315, challengesFRI[1]);
    signal tmp_317[3] <== [mapValues.cm3_6[0] - evals[106][0], mapValues.cm3_6[1] - evals[106][1], mapValues.cm3_6[2] - evals[106][2]];
    signal tmp_318[3] <== [tmp_316[0] + tmp_317[0], tmp_316[1] + tmp_317[1], tmp_316[2] + tmp_317[2]];
    signal tmp_319[3] <== CMul()(tmp_318, xDivXSubXi[1]);
    signal tmp_320[3] <== [tmp_44[0] + tmp_319[0], tmp_44[1] + tmp_319[1], tmp_44[2] + tmp_319[2]];
    signal tmp_321[3] <== CMul()(challengesFRI[0], tmp_320);
    signal tmp_322[3] <== [consts[36] - evals[107][0], -evals[107][1], -evals[107][2]];
    signal tmp_323[3] <== CMul()(tmp_322, challengesFRI[1]);
    signal tmp_324[3] <== [consts[37] - evals[108][0], -evals[108][1], -evals[108][2]];
    signal tmp_325[3] <== [tmp_323[0] + tmp_324[0], tmp_323[1] + tmp_324[1], tmp_323[2] + tmp_324[2]];
    signal tmp_326[3] <== CMul()(tmp_325, challengesFRI[1]);
    signal tmp_327[3] <== [consts[44] - evals[109][0], -evals[109][1], -evals[109][2]];
    signal tmp_328[3] <== [tmp_326[0] + tmp_327[0], tmp_326[1] + tmp_327[1], tmp_326[2] + tmp_327[2]];
    signal tmp_329[3] <== CMul()(tmp_328, challengesFRI[1]);
    signal tmp_330[3] <== [mapValues.cm1_12 - evals[110][0], -evals[110][1], -evals[110][2]];
    signal tmp_331[3] <== [tmp_329[0] + tmp_330[0], tmp_329[1] + tmp_330[1], tmp_329[2] + tmp_330[2]];
    signal tmp_332[3] <== CMul()(tmp_331, challengesFRI[1]);
    signal tmp_333[3] <== [mapValues.cm1_13 - evals[111][0], -evals[111][1], -evals[111][2]];
    signal tmp_334[3] <== [tmp_332[0] + tmp_333[0], tmp_332[1] + tmp_333[1], tmp_332[2] + tmp_333[2]];
    signal tmp_335[3] <== CMul()(tmp_334, challengesFRI[1]);
    signal tmp_336[3] <== [mapValues.cm1_14 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_337[3] <== [tmp_335[0] + tmp_336[0], tmp_335[1] + tmp_336[1], tmp_335[2] + tmp_336[2]];
    signal tmp_338[3] <== CMul()(tmp_337, challengesFRI[1]);
    signal tmp_339[3] <== [mapValues.cm1_15 - evals[113][0], -evals[113][1], -evals[113][2]];
    signal tmp_340[3] <== [tmp_338[0] + tmp_339[0], tmp_338[1] + tmp_339[1], tmp_338[2] + tmp_339[2]];
    signal tmp_341[3] <== CMul()(tmp_340, challengesFRI[1]);
    signal tmp_342[3] <== [mapValues.cm1_16 - evals[114][0], -evals[114][1], -evals[114][2]];
    signal tmp_343[3] <== [tmp_341[0] + tmp_342[0], tmp_341[1] + tmp_342[1], tmp_341[2] + tmp_342[2]];
    signal tmp_344[3] <== CMul()(tmp_343, challengesFRI[1]);
    signal tmp_345[3] <== [mapValues.cm1_17 - evals[115][0], -evals[115][1], -evals[115][2]];
    signal tmp_346[3] <== [tmp_344[0] + tmp_345[0], tmp_344[1] + tmp_345[1], tmp_344[2] + tmp_345[2]];
    signal tmp_347[3] <== CMul()(tmp_346, challengesFRI[1]);
    signal tmp_348[3] <== [mapValues.cm1_18 - evals[116][0], -evals[116][1], -evals[116][2]];
    signal tmp_349[3] <== [tmp_347[0] + tmp_348[0], tmp_347[1] + tmp_348[1], tmp_347[2] + tmp_348[2]];
    signal tmp_350[3] <== CMul()(tmp_349, challengesFRI[1]);
    signal tmp_351[3] <== [mapValues.cm1_19 - evals[117][0], -evals[117][1], -evals[117][2]];
    signal tmp_352[3] <== [tmp_350[0] + tmp_351[0], tmp_350[1] + tmp_351[1], tmp_350[2] + tmp_351[2]];
    signal tmp_353[3] <== CMul()(tmp_352, challengesFRI[1]);
    signal tmp_354[3] <== [mapValues.cm1_20 - evals[118][0], -evals[118][1], -evals[118][2]];
    signal tmp_355[3] <== [tmp_353[0] + tmp_354[0], tmp_353[1] + tmp_354[1], tmp_353[2] + tmp_354[2]];
    signal tmp_356[3] <== CMul()(tmp_355, challengesFRI[1]);
    signal tmp_357[3] <== [mapValues.cm1_21 - evals[119][0], -evals[119][1], -evals[119][2]];
    signal tmp_358[3] <== [tmp_356[0] + tmp_357[0], tmp_356[1] + tmp_357[1], tmp_356[2] + tmp_357[2]];
    signal tmp_359[3] <== CMul()(tmp_358, challengesFRI[1]);
    signal tmp_360[3] <== [mapValues.cm1_22 - evals[120][0], -evals[120][1], -evals[120][2]];
    signal tmp_361[3] <== [tmp_359[0] + tmp_360[0], tmp_359[1] + tmp_360[1], tmp_359[2] + tmp_360[2]];
    signal tmp_362[3] <== CMul()(tmp_361, challengesFRI[1]);
    signal tmp_363[3] <== [mapValues.cm1_23 - evals[121][0], -evals[121][1], -evals[121][2]];
    signal tmp_364[3] <== [tmp_362[0] + tmp_363[0], tmp_362[1] + tmp_363[1], tmp_362[2] + tmp_363[2]];
    signal tmp_365[3] <== CMul()(tmp_364, challengesFRI[1]);
    signal tmp_366[3] <== [mapValues.cm1_24 - evals[122][0], -evals[122][1], -evals[122][2]];
    signal tmp_367[3] <== [tmp_365[0] + tmp_366[0], tmp_365[1] + tmp_366[1], tmp_365[2] + tmp_366[2]];
    signal tmp_368[3] <== CMul()(tmp_367, challengesFRI[1]);
    signal tmp_369[3] <== [mapValues.cm1_25 - evals[123][0], -evals[123][1], -evals[123][2]];
    signal tmp_370[3] <== [tmp_368[0] + tmp_369[0], tmp_368[1] + tmp_369[1], tmp_368[2] + tmp_369[2]];
    signal tmp_371[3] <== CMul()(tmp_370, challengesFRI[1]);
    signal tmp_372[3] <== [mapValues.cm1_26 - evals[124][0], -evals[124][1], -evals[124][2]];
    signal tmp_373[3] <== [tmp_371[0] + tmp_372[0], tmp_371[1] + tmp_372[1], tmp_371[2] + tmp_372[2]];
    signal tmp_374[3] <== CMul()(tmp_373, challengesFRI[1]);
    signal tmp_375[3] <== [mapValues.cm1_27 - evals[125][0], -evals[125][1], -evals[125][2]];
    signal tmp_376[3] <== [tmp_374[0] + tmp_375[0], tmp_374[1] + tmp_375[1], tmp_374[2] + tmp_375[2]];
    signal tmp_377[3] <== CMul()(tmp_376, challengesFRI[1]);
    signal tmp_378[3] <== [mapValues.cm1_28 - evals[126][0], -evals[126][1], -evals[126][2]];
    signal tmp_379[3] <== [tmp_377[0] + tmp_378[0], tmp_377[1] + tmp_378[1], tmp_377[2] + tmp_378[2]];
    signal tmp_380[3] <== CMul()(tmp_379, challengesFRI[1]);
    signal tmp_381[3] <== [mapValues.cm1_29 - evals[127][0], -evals[127][1], -evals[127][2]];
    signal tmp_382[3] <== [tmp_380[0] + tmp_381[0], tmp_380[1] + tmp_381[1], tmp_380[2] + tmp_381[2]];
    signal tmp_383[3] <== CMul()(tmp_382, challengesFRI[1]);
    signal tmp_384[3] <== [mapValues.cm1_30 - evals[128][0], -evals[128][1], -evals[128][2]];
    signal tmp_385[3] <== [tmp_383[0] + tmp_384[0], tmp_383[1] + tmp_384[1], tmp_383[2] + tmp_384[2]];
    signal tmp_386[3] <== CMul()(tmp_385, challengesFRI[1]);
    signal tmp_387[3] <== [mapValues.cm1_31 - evals[129][0], -evals[129][1], -evals[129][2]];
    signal tmp_388[3] <== [tmp_386[0] + tmp_387[0], tmp_386[1] + tmp_387[1], tmp_386[2] + tmp_387[2]];
    signal tmp_389[3] <== CMul()(tmp_388, challengesFRI[1]);
    signal tmp_390[3] <== [mapValues.cm1_32 - evals[130][0], -evals[130][1], -evals[130][2]];
    signal tmp_391[3] <== [tmp_389[0] + tmp_390[0], tmp_389[1] + tmp_390[1], tmp_389[2] + tmp_390[2]];
    signal tmp_392[3] <== CMul()(tmp_391, challengesFRI[1]);
    signal tmp_393[3] <== [mapValues.cm1_33 - evals[131][0], -evals[131][1], -evals[131][2]];
    signal tmp_394[3] <== [tmp_392[0] + tmp_393[0], tmp_392[1] + tmp_393[1], tmp_392[2] + tmp_393[2]];
    signal tmp_395[3] <== CMul()(tmp_394, challengesFRI[1]);
    signal tmp_396[3] <== [mapValues.cm1_34 - evals[132][0], -evals[132][1], -evals[132][2]];
    signal tmp_397[3] <== [tmp_395[0] + tmp_396[0], tmp_395[1] + tmp_396[1], tmp_395[2] + tmp_396[2]];
    signal tmp_398[3] <== CMul()(tmp_397, challengesFRI[1]);
    signal tmp_399[3] <== [mapValues.cm1_35 - evals[133][0], -evals[133][1], -evals[133][2]];
    signal tmp_400[3] <== [tmp_398[0] + tmp_399[0], tmp_398[1] + tmp_399[1], tmp_398[2] + tmp_399[2]];
    signal tmp_401[3] <== CMul()(tmp_400, xDivXSubXi[2]);
    signal tmp_402[3] <== [tmp_321[0] + tmp_401[0], tmp_321[1] + tmp_401[1], tmp_321[2] + tmp_401[2]];
    signal tmp_403[3] <== CMul()(challengesFRI[0], tmp_402);
    signal tmp_404[3] <== [mapValues.cm1_12 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_405[3] <== CMul()(tmp_404, challengesFRI[1]);
    signal tmp_406[3] <== [mapValues.cm1_13 - evals[135][0], -evals[135][1], -evals[135][2]];
    signal tmp_407[3] <== [tmp_405[0] + tmp_406[0], tmp_405[1] + tmp_406[1], tmp_405[2] + tmp_406[2]];
    signal tmp_408[3] <== CMul()(tmp_407, xDivXSubXi[3]);
    signal tmp_410[3] <== [tmp_403[0] + tmp_408[0], tmp_403[1] + tmp_408[1], tmp_403[2] + tmp_408[2]];

    queryVals[0] <== tmp_410[0];
    queryVals[1] <== tmp_410[1];
    queryVals[2] <== tmp_410[2];
}

/* 
    Verify that the initial FRI polynomial, which is the lineal combination of the committed polynomials
    during the STARK phases, is built properly
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesFRI[20];
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
 
    signal input vals1[36];
 
    signal input vals2[12];
    signal input vals3[21];
    signal output cm1_0;
    signal output cm1_1;
    signal output cm1_2;
    signal output cm1_3;
    signal output cm1_4;
    signal output cm1_5;
    signal output cm1_6;
    signal output cm1_7;
    signal output cm1_8;
    signal output cm1_9;
    signal output cm1_10;
    signal output cm1_11;
    signal output cm1_12;
    signal output cm1_13;
    signal output cm1_14;
    signal output cm1_15;
    signal output cm1_16;
    signal output cm1_17;
    signal output cm1_18;
    signal output cm1_19;
    signal output cm1_20;
    signal output cm1_21;
    signal output cm1_22;
    signal output cm1_23;
    signal output cm1_24;
    signal output cm1_25;
    signal output cm1_26;
    signal output cm1_27;
    signal output cm1_28;
    signal output cm1_29;
    signal output cm1_30;
    signal output cm1_31;
    signal output cm1_32;
    signal output cm1_33;
    signal output cm1_34;
    signal output cm1_35;
    signal output cm2_0[3];
    signal output cm2_1[3];
    signal output cm2_2[3];
    signal output cm2_3[3];
    signal output cm3_0[3];
    signal output cm3_1[3];
    signal output cm3_2[3];
    signal output cm3_3[3];
    signal output cm3_4[3];
    signal output cm3_5[3];
    signal output cm3_6[3];


    cm1_0 <== vals1[0];
    cm1_1 <== vals1[1];
    cm1_2 <== vals1[2];
    cm1_3 <== vals1[3];
    cm1_4 <== vals1[4];
    cm1_5 <== vals1[5];
    cm1_6 <== vals1[6];
    cm1_7 <== vals1[7];
    cm1_8 <== vals1[8];
    cm1_9 <== vals1[9];
    cm1_10 <== vals1[10];
    cm1_11 <== vals1[11];
    cm1_12 <== vals1[12];
    cm1_13 <== vals1[13];
    cm1_14 <== vals1[14];
    cm1_15 <== vals1[15];
    cm1_16 <== vals1[16];
    cm1_17 <== vals1[17];
    cm1_18 <== vals1[18];
    cm1_19 <== vals1[19];
    cm1_20 <== vals1[20];
    cm1_21 <== vals1[21];
    cm1_22 <== vals1[22];
    cm1_23 <== vals1[23];
    cm1_24 <== vals1[24];
    cm1_25 <== vals1[25];
    cm1_26 <== vals1[26];
    cm1_27 <== vals1[27];
    cm1_28 <== vals1[28];
    cm1_29 <== vals1[29];
    cm1_30 <== vals1[30];
    cm1_31 <== vals1[31];
    cm1_32 <== vals1[32];
    cm1_33 <== vals1[33];
    cm1_34 <== vals1[34];
    cm1_35 <== vals1[35];
    cm2_0 <== [vals2[0],vals2[1] , vals2[2]];
    cm2_1 <== [vals2[3],vals2[4] , vals2[5]];
    cm2_2 <== [vals2[6],vals2[7] , vals2[8]];
    cm2_3 <== [vals2[9],vals2[10] , vals2[11]];
    cm3_0 <== [vals3[0],vals3[1] , vals3[2]];
    cm3_1 <== [vals3[3],vals3[4] , vals3[5]];
    cm3_2 <== [vals3[6],vals3[7] , vals3[8]];
    cm3_3 <== [vals3[9],vals3[10] , vals3[11]];
    cm3_4 <== [vals3[12],vals3[13] , vals3[14]];
    cm3_5 <== [vals3[15],vals3[16] , vals3[17]];
    cm3_6 <== [vals3[18],vals3[19] , vals3[20]];
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
    for (var k= 4; k< 32; k++) {
        for (var e=0; e<3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k= 0; k < 4; k++) {
        _ <== lastIFFT[k];
    }
}

template StarkVerifier0() {
    signal input publics[37]; // publics polynomials
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials


    signal input rootC[4]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[136][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
 
    signal input s0_vals1[43][36];
 
    signal input s0_vals2[43][12];
                                       
    signal input s0_vals3[43][21];
    signal input s0_valsC[43][45];


    // Merkle proofs for each of the evaluations
 
    signal input s0_siblings1[43][13][8];
 
    signal input s0_siblings2[43][13][8];
 
    signal input s0_siblings3[43][13][8];
    signal input s0_siblingsC[43][13][8];
    // Contains the root of the original polynomial and all the intermediate FRI polynomials except for the last step
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];

    // For each intermediate FRI polynomial and the last one, we store at vals the values needed to check the queries.
    // Given a query r,  the verifier needs b points to check it out, being b = 2^u, where u is the difference between two consecutive step
    // and the sibling paths for each query.
    signal input s1_vals[43][48];
    signal input s1_siblings[43][11][8];
    signal input s2_vals[43][48];
    signal input s2_siblings[43][8][8];
    signal input s3_vals[43][48];
    signal input s3_siblings[43][6][8];
    signal input s4_vals[43][24];
    signal input s4_siblings[43][4][8];

    // Evaluations of the final FRI polynomial over a set of points of size bounded its degree
    signal input finalPol[32][3];

    signal {binary} enabled;
    signal input enable;
    enable * (enable -1) === 0;
    enabled <== enable;


    signal queryVals[43][3];

    signal challengesStage2[2][3];

    signal challengeQ[3];
    signal challengeXi[3];
    signal challengesFRI[2][3];


    // challengesFRISteps contains the random value provided by the verifier at each step of the folding so that 
    // the prover can commit the polynomial.
    // Remember that, when folding, the prover does as follows: f0 = g_0 + X*g_1 + ... + (X^b)*g_b and then the 
    // verifier provides a random X so that the prover can commit it. This value is stored here.
    signal challengesFRISteps[6][3];

    // Challenges from which we derive all the queries
    signal {binary} queriesFRI[43][20];


    ///////////
    // Calculate challenges, challengesFRISteps and queriesFRI
    ///////////

 

    (challengesStage2,challengeQ,challengeXi,challengesFRI,challengesFRISteps,queriesFRI) <== Transcript0()(publics,rootC,root1,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,finalPol);

    ///////////
    // Check constraints polynomial in the evaluation point
    ///////////

 

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputs so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
 
    var s0_vals1_p[43][36][1];
 
    var s0_vals2_p[43][12][1];
 
    var s0_vals3_p[43][21][1];
    var s0_valsC_p[43][45][1];
    var s0_vals_p[43][1][3]; 
    var s1_vals_p[43][16][3]; 
    var s2_vals_p[43][16][3]; 
    var s3_vals_p[43][16][3]; 
    var s4_vals_p[43][8][3]; 

    for (var q=0; q<43; q++) {
        // Preprocess vals for the initial FRI polynomial
 
        for (var i = 0; i < 36; i++) {
            s0_vals1_p[q][i][0] = s0_vals1[q][i];
        }
 
        for (var i = 0; i < 12; i++) {
            s0_vals2_p[q][i][0] = s0_vals2[q][i];
        }
 
        for (var i = 0; i < 21; i++) {
            s0_vals3_p[q][i][0] = s0_vals3[q][i];
        }
        for (var i = 0; i < 45; i++) {
            s0_valsC_p[q][i][0] = s0_valsC[q][i];
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
        }
    }
    
    ///////////
    // Verify Merkle Roots
    ///////////
    signal {binary} queriesFRITernary[43][13][2];
    signal queriesFRINum[43];
    for(var i = 0; i < 43; i++) {
        queriesFRINum[i] <== Bits2Num(20)(queriesFRI[i]);
        queriesFRITernary[i] <== Num2Ternary(13)(queriesFRINum[i]);
    }

    //Calculate merkle root for s0 vals
 
    for (var q=0; q<43; q++) {
        VerifyMerkleHash(1, 36, 3, 13)(s0_vals1_p[q], s0_siblings1[q], queriesFRITernary[q], root1, enabled);
    }
 
    for (var q=0; q<43; q++) {
        VerifyMerkleHash(1, 12, 3, 13)(s0_vals2_p[q], s0_siblings2[q], queriesFRITernary[q], root2, enabled);
    }

    for (var q=0; q<43; q++) {
        VerifyMerkleHash(1, 21, 3, 13)(s0_vals3_p[q], s0_siblings3[q], queriesFRITernary[q], root3, enabled);
    }

    for (var q=0; q<43; q++) {
        VerifyMerkleHash(1, 45, 3, 13)(s0_valsC_p[q], s0_siblingsC[q], queriesFRITernary[q], rootC, enabled);                                    
    }


    signal {binary} s1_keys_merkle[43][16];
    signal s1_keys_merkle_num[43];
    signal {binary} s1_keys_merkle_ternary[43][11][2];
    for (var q=0; q<43; q++) {
        // Calculate merkle root for s1 vals
        for(var i = 0; i < 16; i++) { s1_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s1_keys_merkle_num[q] <== Bits2Num(16)(s1_keys_merkle[q]);
        s1_keys_merkle_ternary[q] <== Num2Ternary(11)(s1_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 11)(s1_vals_p[q], s1_siblings[q], s1_keys_merkle_ternary[q], s1_root, enabled);
    }
    signal {binary} s2_keys_merkle[43][12];
    signal s2_keys_merkle_num[43];
    signal {binary} s2_keys_merkle_ternary[43][8][2];
    for (var q=0; q<43; q++) {
        // Calculate merkle root for s2 vals
        for(var i = 0; i < 12; i++) { s2_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s2_keys_merkle_num[q] <== Bits2Num(12)(s2_keys_merkle[q]);
        s2_keys_merkle_ternary[q] <== Num2Ternary(8)(s2_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 8)(s2_vals_p[q], s2_siblings[q], s2_keys_merkle_ternary[q], s2_root, enabled);
    }
    signal {binary} s3_keys_merkle[43][8];
    signal s3_keys_merkle_num[43];
    signal {binary} s3_keys_merkle_ternary[43][6][2];
    for (var q=0; q<43; q++) {
        // Calculate merkle root for s3 vals
        for(var i = 0; i < 8; i++) { s3_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s3_keys_merkle_num[q] <== Bits2Num(8)(s3_keys_merkle[q]);
        s3_keys_merkle_ternary[q] <== Num2Ternary(6)(s3_keys_merkle_num[q]);
        VerifyMerkleHash(3, 16, 3, 6)(s3_vals_p[q], s3_siblings[q], s3_keys_merkle_ternary[q], s3_root, enabled);
    }
    signal {binary} s4_keys_merkle[43][5];
    signal s4_keys_merkle_num[43];
    signal {binary} s4_keys_merkle_ternary[43][4][2];
    for (var q=0; q<43; q++) {
        // Calculate merkle root for s4 vals
        for(var i = 0; i < 5; i++) { s4_keys_merkle[q][i] <== queriesFRI[q][i]; }
        s4_keys_merkle_num[q] <== Bits2Num(5)(s4_keys_merkle[q]);
        s4_keys_merkle_ternary[q] <== Num2Ternary(4)(s4_keys_merkle_num[q]);
        VerifyMerkleHash(3, 8, 3, 4)(s4_vals_p[q], s4_siblings[q], s4_keys_merkle_ternary[q], s4_root, enabled);
    }
        

    ///////////
    // Calculate FRI Polinomial
    ///////////
    
    for (var q=0; q<43; q++) {
        // Reconstruct FRI polinomial from evaluations
        queryVals[q] <== CalculateFRIPolValue0()(queriesFRI[q], challengeXi, challengesFRI, evals, s0_vals1[q], s0_vals2[q], s0_vals3[q], s0_valsC[q]);
    }

    ///////////
    // Verify FRI Polinomial
    ///////////
    signal {binary} s1_queriesFRI[43][16];
    signal {binary} s2_queriesFRI[43][12];
    signal {binary} s3_queriesFRI[43][8];
    signal {binary} s4_queriesFRI[43][5];

    for (var q=0; q<43; q++) {
      
        // Verify that the query is properly constructed. This is done by checking that the linear combination of the set of 
        // polynomials committed during the different rounds evaluated at z matches with the commitment of the FRI polynomial
        VerifyQuery0(20, 16)(queriesFRI[q], queryVals[q], s1_vals_p[q], enabled);

        ///////////
        // Verify FRI construction
        ///////////

        // For each folding level we need to check that the polynomial is properly constructed
        // Remember that if the step between polynomials is b = 2^l, the next polynomial p_(i+1) will have degree deg(p_i) / b

        // Check S1
        for(var i = 0; i < 16; i++) { s1_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 20, 16, 12, 2635249152773512046)(s1_queriesFRI[q], challengesFRISteps[1], s1_vals_p[q], s2_vals_p[q], enabled);

        // Check S2
        for(var i = 0; i < 12; i++) { s2_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 16, 12, 8, 11131999729878195124)(s2_queriesFRI[q], challengesFRISteps[2], s2_vals_p[q], s3_vals_p[q], enabled);

        // Check S3
        for(var i = 0; i < 8; i++) { s3_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 12, 8, 5, 16627473974463641638)(s3_queriesFRI[q], challengesFRISteps[3], s3_vals_p[q], s4_vals_p[q], enabled);

        // Check S4
        for(var i = 0; i < 5; i++) { s4_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 8, 5, 0, 140704680260498080)(s4_queriesFRI[q], challengesFRISteps[4], s4_vals_p[q], finalPol, enabled);
    }

    VerifyFinalPol0()(finalPol, enabled);
}
    
