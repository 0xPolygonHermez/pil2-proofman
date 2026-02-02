pragma circom 2.1.0;
pragma custom_templates;




include "cmul.circom";
include "cinv.circom";
include "poseidon2.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "treeselector4.circom";
include "pow.circom";
include "merklehash.circom";


/* 
    Calculate FRI Queries
*/
template calculateFRIQueries0() {
    
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queriesFRI[110][20];

    VerifyPoW(20)(challengeFRIQueries, nonce, enable);

    

    
    signal transcriptHash_friQueries_0[16] <== Poseidon2(4, 16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],nonce,0,0,0,0,0,0,0,0], [0,0,0,0]);
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
    signal {binary} transcriptN2b_12[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[12]);
    signal {binary} transcriptN2b_13[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[13]);
    signal {binary} transcriptN2b_14[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[14]);
    signal {binary} transcriptN2b_15[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[15]);
    
    signal transcriptHash_friQueries_1[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_0[0],transcriptHash_friQueries_0[1],transcriptHash_friQueries_0[2],transcriptHash_friQueries_0[3]]);
    signal {binary} transcriptN2b_16[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[0]);
    signal {binary} transcriptN2b_17[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[1]);
    signal {binary} transcriptN2b_18[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[2]);
    signal {binary} transcriptN2b_19[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[3]);
    signal {binary} transcriptN2b_20[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[4]);
    signal {binary} transcriptN2b_21[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[5]);
    signal {binary} transcriptN2b_22[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[6]);
    signal {binary} transcriptN2b_23[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[7]);
    signal {binary} transcriptN2b_24[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[8]);
    signal {binary} transcriptN2b_25[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[9]);
    signal {binary} transcriptN2b_26[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[10]);
    signal {binary} transcriptN2b_27[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[11]);
    signal {binary} transcriptN2b_28[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[12]);
    signal {binary} transcriptN2b_29[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[13]);
    signal {binary} transcriptN2b_30[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[14]);
    signal {binary} transcriptN2b_31[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[15]);
    
    signal transcriptHash_friQueries_2[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_1[0],transcriptHash_friQueries_1[1],transcriptHash_friQueries_1[2],transcriptHash_friQueries_1[3]]);
    signal {binary} transcriptN2b_32[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[0]);
    signal {binary} transcriptN2b_33[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[1]);
    signal {binary} transcriptN2b_34[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[2]);
    for(var i = 3; i < 16; i++){
        _ <== transcriptHash_friQueries_2[i]; // Unused transcript values        
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

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_13[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_14[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_14[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_15[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_15[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_16[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_16[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_17[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_17[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_18[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_18[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_19[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_19[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_20[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_20[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_21[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_21[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_22[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_22[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_23[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_23[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_24[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_24[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_25[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_25[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_26[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_26[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_27[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_27[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_28[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_28[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_29[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_29[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_30[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_30[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_31[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_31[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_32[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_32[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_33[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_33[63]; // Unused last bit

    for(var j = 0; j < 58; j++) {
        queriesFRI[q][b] <== transcriptN2b_34[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    for(var j = 58; j < 64; j++) {
        _ <== transcriptN2b_34[j]; // Unused bits        
    }
}


/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input publics[391];
    signal input rootC[4];
    signal input root1[4];

    
    signal input root2[4];
                  
    signal input root3[4];
    signal input evals[198][3]; 
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];
    signal input finalPol[32][3];
    signal input nonce;
    signal input {binary} enable;

    signal output challengesStage2[2][3];

    signal output challengeQ[3];
    signal output challengeXi[3];
    signal output challengesFRI[2][3];
    signal output challengesFRISteps[7][3];
    signal output {binary} queriesFRI[110][20];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];


    
    signal transcriptHash_publics_0[16] <== Poseidon2(4, 16)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7],publics[8],publics[9],publics[10],publics[11]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_1[16] <== Poseidon2(4, 16)([publics[12],publics[13],publics[14],publics[15],publics[16],publics[17],publics[18],publics[19],publics[20],publics[21],publics[22],publics[23]], [transcriptHash_publics_0[0],transcriptHash_publics_0[1],transcriptHash_publics_0[2],transcriptHash_publics_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_2[16] <== Poseidon2(4, 16)([publics[24],publics[25],publics[26],publics[27],publics[28],publics[29],publics[30],publics[31],publics[32],publics[33],publics[34],publics[35]], [transcriptHash_publics_1[0],transcriptHash_publics_1[1],transcriptHash_publics_1[2],transcriptHash_publics_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_3[16] <== Poseidon2(4, 16)([publics[36],publics[37],publics[38],publics[39],publics[40],publics[41],publics[42],publics[43],publics[44],publics[45],publics[46],publics[47]], [transcriptHash_publics_2[0],transcriptHash_publics_2[1],transcriptHash_publics_2[2],transcriptHash_publics_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_4[16] <== Poseidon2(4, 16)([publics[48],publics[49],publics[50],publics[51],publics[52],publics[53],publics[54],publics[55],publics[56],publics[57],publics[58],publics[59]], [transcriptHash_publics_3[0],transcriptHash_publics_3[1],transcriptHash_publics_3[2],transcriptHash_publics_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_5[16] <== Poseidon2(4, 16)([publics[60],publics[61],publics[62],publics[63],publics[64],publics[65],publics[66],publics[67],publics[68],publics[69],publics[70],publics[71]], [transcriptHash_publics_4[0],transcriptHash_publics_4[1],transcriptHash_publics_4[2],transcriptHash_publics_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_6[16] <== Poseidon2(4, 16)([publics[72],publics[73],publics[74],publics[75],publics[76],publics[77],publics[78],publics[79],publics[80],publics[81],publics[82],publics[83]], [transcriptHash_publics_5[0],transcriptHash_publics_5[1],transcriptHash_publics_5[2],transcriptHash_publics_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_7[16] <== Poseidon2(4, 16)([publics[84],publics[85],publics[86],publics[87],publics[88],publics[89],publics[90],publics[91],publics[92],publics[93],publics[94],publics[95]], [transcriptHash_publics_6[0],transcriptHash_publics_6[1],transcriptHash_publics_6[2],transcriptHash_publics_6[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_8[16] <== Poseidon2(4, 16)([publics[96],publics[97],publics[98],publics[99],publics[100],publics[101],publics[102],publics[103],publics[104],publics[105],publics[106],publics[107]], [transcriptHash_publics_7[0],transcriptHash_publics_7[1],transcriptHash_publics_7[2],transcriptHash_publics_7[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_9[16] <== Poseidon2(4, 16)([publics[108],publics[109],publics[110],publics[111],publics[112],publics[113],publics[114],publics[115],publics[116],publics[117],publics[118],publics[119]], [transcriptHash_publics_8[0],transcriptHash_publics_8[1],transcriptHash_publics_8[2],transcriptHash_publics_8[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_9[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_10[16] <== Poseidon2(4, 16)([publics[120],publics[121],publics[122],publics[123],publics[124],publics[125],publics[126],publics[127],publics[128],publics[129],publics[130],publics[131]], [transcriptHash_publics_9[0],transcriptHash_publics_9[1],transcriptHash_publics_9[2],transcriptHash_publics_9[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_10[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_11[16] <== Poseidon2(4, 16)([publics[132],publics[133],publics[134],publics[135],publics[136],publics[137],publics[138],publics[139],publics[140],publics[141],publics[142],publics[143]], [transcriptHash_publics_10[0],transcriptHash_publics_10[1],transcriptHash_publics_10[2],transcriptHash_publics_10[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_11[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_12[16] <== Poseidon2(4, 16)([publics[144],publics[145],publics[146],publics[147],publics[148],publics[149],publics[150],publics[151],publics[152],publics[153],publics[154],publics[155]], [transcriptHash_publics_11[0],transcriptHash_publics_11[1],transcriptHash_publics_11[2],transcriptHash_publics_11[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_12[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_13[16] <== Poseidon2(4, 16)([publics[156],publics[157],publics[158],publics[159],publics[160],publics[161],publics[162],publics[163],publics[164],publics[165],publics[166],publics[167]], [transcriptHash_publics_12[0],transcriptHash_publics_12[1],transcriptHash_publics_12[2],transcriptHash_publics_12[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_13[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_14[16] <== Poseidon2(4, 16)([publics[168],publics[169],publics[170],publics[171],publics[172],publics[173],publics[174],publics[175],publics[176],publics[177],publics[178],publics[179]], [transcriptHash_publics_13[0],transcriptHash_publics_13[1],transcriptHash_publics_13[2],transcriptHash_publics_13[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_14[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_15[16] <== Poseidon2(4, 16)([publics[180],publics[181],publics[182],publics[183],publics[184],publics[185],publics[186],publics[187],publics[188],publics[189],publics[190],publics[191]], [transcriptHash_publics_14[0],transcriptHash_publics_14[1],transcriptHash_publics_14[2],transcriptHash_publics_14[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_15[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_16[16] <== Poseidon2(4, 16)([publics[192],publics[193],publics[194],publics[195],publics[196],publics[197],publics[198],publics[199],publics[200],publics[201],publics[202],publics[203]], [transcriptHash_publics_15[0],transcriptHash_publics_15[1],transcriptHash_publics_15[2],transcriptHash_publics_15[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_16[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_17[16] <== Poseidon2(4, 16)([publics[204],publics[205],publics[206],publics[207],publics[208],publics[209],publics[210],publics[211],publics[212],publics[213],publics[214],publics[215]], [transcriptHash_publics_16[0],transcriptHash_publics_16[1],transcriptHash_publics_16[2],transcriptHash_publics_16[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_17[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_18[16] <== Poseidon2(4, 16)([publics[216],publics[217],publics[218],publics[219],publics[220],publics[221],publics[222],publics[223],publics[224],publics[225],publics[226],publics[227]], [transcriptHash_publics_17[0],transcriptHash_publics_17[1],transcriptHash_publics_17[2],transcriptHash_publics_17[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_18[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_19[16] <== Poseidon2(4, 16)([publics[228],publics[229],publics[230],publics[231],publics[232],publics[233],publics[234],publics[235],publics[236],publics[237],publics[238],publics[239]], [transcriptHash_publics_18[0],transcriptHash_publics_18[1],transcriptHash_publics_18[2],transcriptHash_publics_18[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_19[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_20[16] <== Poseidon2(4, 16)([publics[240],publics[241],publics[242],publics[243],publics[244],publics[245],publics[246],publics[247],publics[248],publics[249],publics[250],publics[251]], [transcriptHash_publics_19[0],transcriptHash_publics_19[1],transcriptHash_publics_19[2],transcriptHash_publics_19[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_20[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_21[16] <== Poseidon2(4, 16)([publics[252],publics[253],publics[254],publics[255],publics[256],publics[257],publics[258],publics[259],publics[260],publics[261],publics[262],publics[263]], [transcriptHash_publics_20[0],transcriptHash_publics_20[1],transcriptHash_publics_20[2],transcriptHash_publics_20[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_21[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_22[16] <== Poseidon2(4, 16)([publics[264],publics[265],publics[266],publics[267],publics[268],publics[269],publics[270],publics[271],publics[272],publics[273],publics[274],publics[275]], [transcriptHash_publics_21[0],transcriptHash_publics_21[1],transcriptHash_publics_21[2],transcriptHash_publics_21[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_22[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_23[16] <== Poseidon2(4, 16)([publics[276],publics[277],publics[278],publics[279],publics[280],publics[281],publics[282],publics[283],publics[284],publics[285],publics[286],publics[287]], [transcriptHash_publics_22[0],transcriptHash_publics_22[1],transcriptHash_publics_22[2],transcriptHash_publics_22[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_23[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_24[16] <== Poseidon2(4, 16)([publics[288],publics[289],publics[290],publics[291],publics[292],publics[293],publics[294],publics[295],publics[296],publics[297],publics[298],publics[299]], [transcriptHash_publics_23[0],transcriptHash_publics_23[1],transcriptHash_publics_23[2],transcriptHash_publics_23[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_24[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_25[16] <== Poseidon2(4, 16)([publics[300],publics[301],publics[302],publics[303],publics[304],publics[305],publics[306],publics[307],publics[308],publics[309],publics[310],publics[311]], [transcriptHash_publics_24[0],transcriptHash_publics_24[1],transcriptHash_publics_24[2],transcriptHash_publics_24[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_25[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_26[16] <== Poseidon2(4, 16)([publics[312],publics[313],publics[314],publics[315],publics[316],publics[317],publics[318],publics[319],publics[320],publics[321],publics[322],publics[323]], [transcriptHash_publics_25[0],transcriptHash_publics_25[1],transcriptHash_publics_25[2],transcriptHash_publics_25[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_26[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_27[16] <== Poseidon2(4, 16)([publics[324],publics[325],publics[326],publics[327],publics[328],publics[329],publics[330],publics[331],publics[332],publics[333],publics[334],publics[335]], [transcriptHash_publics_26[0],transcriptHash_publics_26[1],transcriptHash_publics_26[2],transcriptHash_publics_26[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_27[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_28[16] <== Poseidon2(4, 16)([publics[336],publics[337],publics[338],publics[339],publics[340],publics[341],publics[342],publics[343],publics[344],publics[345],publics[346],publics[347]], [transcriptHash_publics_27[0],transcriptHash_publics_27[1],transcriptHash_publics_27[2],transcriptHash_publics_27[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_28[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_29[16] <== Poseidon2(4, 16)([publics[348],publics[349],publics[350],publics[351],publics[352],publics[353],publics[354],publics[355],publics[356],publics[357],publics[358],publics[359]], [transcriptHash_publics_28[0],transcriptHash_publics_28[1],transcriptHash_publics_28[2],transcriptHash_publics_28[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_29[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_30[16] <== Poseidon2(4, 16)([publics[360],publics[361],publics[362],publics[363],publics[364],publics[365],publics[366],publics[367],publics[368],publics[369],publics[370],publics[371]], [transcriptHash_publics_29[0],transcriptHash_publics_29[1],transcriptHash_publics_29[2],transcriptHash_publics_29[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_30[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_31[16] <== Poseidon2(4, 16)([publics[372],publics[373],publics[374],publics[375],publics[376],publics[377],publics[378],publics[379],publics[380],publics[381],publics[382],publics[383]], [transcriptHash_publics_30[0],transcriptHash_publics_30[1],transcriptHash_publics_30[2],transcriptHash_publics_30[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_31[i]; // Unused transcript values 
    }
    
    signal transcriptHash_publics_32[16] <== Poseidon2(4, 16)([publics[384],publics[385],publics[386],publics[387],publics[388],publics[389],publics[390],0,0,0,0,0], [transcriptHash_publics_31[0],transcriptHash_publics_31[1],transcriptHash_publics_31[2],transcriptHash_publics_31[3]]);
    publicsHash <== [transcriptHash_publics_32[0], transcriptHash_publics_32[1], transcriptHash_publics_32[2], transcriptHash_publics_32[3]];

    
    signal transcriptHash_0[16] <== Poseidon2(4, 16)([rootC[0],rootC[1],rootC[2],rootC[3],publicsHash[0],publicsHash[1],publicsHash[2],publicsHash[3],root1[0],root1[1],root1[2],root1[3]], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for(var i = 6; i < 16; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_1[16] <== Poseidon2(4, 16)([root2[0],root2[1],root2[2],root2[3],0,0,0,0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_2[16] <== Poseidon2(4, 16)([root3[0],root3[1],root3[2],root3[3],0,0,0,0,0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeXi <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];
    
    signal transcriptHash_evals_0[16] <== Poseidon2(4, 16)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1],evals[2][2],evals[3][0],evals[3][1],evals[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_1[16] <== Poseidon2(4, 16)([evals[4][0],evals[4][1],evals[4][2],evals[5][0],evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_2[16] <== Poseidon2(4, 16)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1],evals[10][2],evals[11][0],evals[11][1],evals[11][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_3[16] <== Poseidon2(4, 16)([evals[12][0],evals[12][1],evals[12][2],evals[13][0],evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_4[16] <== Poseidon2(4, 16)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1],evals[18][2],evals[19][0],evals[19][1],evals[19][2]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_5[16] <== Poseidon2(4, 16)([evals[20][0],evals[20][1],evals[20][2],evals[21][0],evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],evals[23][0],evals[23][1],evals[23][2]], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_6[16] <== Poseidon2(4, 16)([evals[24][0],evals[24][1],evals[24][2],evals[25][0],evals[25][1],evals[25][2],evals[26][0],evals[26][1],evals[26][2],evals[27][0],evals[27][1],evals[27][2]], [transcriptHash_evals_5[0],transcriptHash_evals_5[1],transcriptHash_evals_5[2],transcriptHash_evals_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_7[16] <== Poseidon2(4, 16)([evals[28][0],evals[28][1],evals[28][2],evals[29][0],evals[29][1],evals[29][2],evals[30][0],evals[30][1],evals[30][2],evals[31][0],evals[31][1],evals[31][2]], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_8[16] <== Poseidon2(4, 16)([evals[32][0],evals[32][1],evals[32][2],evals[33][0],evals[33][1],evals[33][2],evals[34][0],evals[34][1],evals[34][2],evals[35][0],evals[35][1],evals[35][2]], [transcriptHash_evals_7[0],transcriptHash_evals_7[1],transcriptHash_evals_7[2],transcriptHash_evals_7[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_9[16] <== Poseidon2(4, 16)([evals[36][0],evals[36][1],evals[36][2],evals[37][0],evals[37][1],evals[37][2],evals[38][0],evals[38][1],evals[38][2],evals[39][0],evals[39][1],evals[39][2]], [transcriptHash_evals_8[0],transcriptHash_evals_8[1],transcriptHash_evals_8[2],transcriptHash_evals_8[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_9[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_10[16] <== Poseidon2(4, 16)([evals[40][0],evals[40][1],evals[40][2],evals[41][0],evals[41][1],evals[41][2],evals[42][0],evals[42][1],evals[42][2],evals[43][0],evals[43][1],evals[43][2]], [transcriptHash_evals_9[0],transcriptHash_evals_9[1],transcriptHash_evals_9[2],transcriptHash_evals_9[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_10[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_11[16] <== Poseidon2(4, 16)([evals[44][0],evals[44][1],evals[44][2],evals[45][0],evals[45][1],evals[45][2],evals[46][0],evals[46][1],evals[46][2],evals[47][0],evals[47][1],evals[47][2]], [transcriptHash_evals_10[0],transcriptHash_evals_10[1],transcriptHash_evals_10[2],transcriptHash_evals_10[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_11[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_12[16] <== Poseidon2(4, 16)([evals[48][0],evals[48][1],evals[48][2],evals[49][0],evals[49][1],evals[49][2],evals[50][0],evals[50][1],evals[50][2],evals[51][0],evals[51][1],evals[51][2]], [transcriptHash_evals_11[0],transcriptHash_evals_11[1],transcriptHash_evals_11[2],transcriptHash_evals_11[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_12[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_13[16] <== Poseidon2(4, 16)([evals[52][0],evals[52][1],evals[52][2],evals[53][0],evals[53][1],evals[53][2],evals[54][0],evals[54][1],evals[54][2],evals[55][0],evals[55][1],evals[55][2]], [transcriptHash_evals_12[0],transcriptHash_evals_12[1],transcriptHash_evals_12[2],transcriptHash_evals_12[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_13[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_14[16] <== Poseidon2(4, 16)([evals[56][0],evals[56][1],evals[56][2],evals[57][0],evals[57][1],evals[57][2],evals[58][0],evals[58][1],evals[58][2],evals[59][0],evals[59][1],evals[59][2]], [transcriptHash_evals_13[0],transcriptHash_evals_13[1],transcriptHash_evals_13[2],transcriptHash_evals_13[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_14[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_15[16] <== Poseidon2(4, 16)([evals[60][0],evals[60][1],evals[60][2],evals[61][0],evals[61][1],evals[61][2],evals[62][0],evals[62][1],evals[62][2],evals[63][0],evals[63][1],evals[63][2]], [transcriptHash_evals_14[0],transcriptHash_evals_14[1],transcriptHash_evals_14[2],transcriptHash_evals_14[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_15[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_16[16] <== Poseidon2(4, 16)([evals[64][0],evals[64][1],evals[64][2],evals[65][0],evals[65][1],evals[65][2],evals[66][0],evals[66][1],evals[66][2],evals[67][0],evals[67][1],evals[67][2]], [transcriptHash_evals_15[0],transcriptHash_evals_15[1],transcriptHash_evals_15[2],transcriptHash_evals_15[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_16[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_17[16] <== Poseidon2(4, 16)([evals[68][0],evals[68][1],evals[68][2],evals[69][0],evals[69][1],evals[69][2],evals[70][0],evals[70][1],evals[70][2],evals[71][0],evals[71][1],evals[71][2]], [transcriptHash_evals_16[0],transcriptHash_evals_16[1],transcriptHash_evals_16[2],transcriptHash_evals_16[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_17[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_18[16] <== Poseidon2(4, 16)([evals[72][0],evals[72][1],evals[72][2],evals[73][0],evals[73][1],evals[73][2],evals[74][0],evals[74][1],evals[74][2],evals[75][0],evals[75][1],evals[75][2]], [transcriptHash_evals_17[0],transcriptHash_evals_17[1],transcriptHash_evals_17[2],transcriptHash_evals_17[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_18[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_19[16] <== Poseidon2(4, 16)([evals[76][0],evals[76][1],evals[76][2],evals[77][0],evals[77][1],evals[77][2],evals[78][0],evals[78][1],evals[78][2],evals[79][0],evals[79][1],evals[79][2]], [transcriptHash_evals_18[0],transcriptHash_evals_18[1],transcriptHash_evals_18[2],transcriptHash_evals_18[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_19[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_20[16] <== Poseidon2(4, 16)([evals[80][0],evals[80][1],evals[80][2],evals[81][0],evals[81][1],evals[81][2],evals[82][0],evals[82][1],evals[82][2],evals[83][0],evals[83][1],evals[83][2]], [transcriptHash_evals_19[0],transcriptHash_evals_19[1],transcriptHash_evals_19[2],transcriptHash_evals_19[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_20[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_21[16] <== Poseidon2(4, 16)([evals[84][0],evals[84][1],evals[84][2],evals[85][0],evals[85][1],evals[85][2],evals[86][0],evals[86][1],evals[86][2],evals[87][0],evals[87][1],evals[87][2]], [transcriptHash_evals_20[0],transcriptHash_evals_20[1],transcriptHash_evals_20[2],transcriptHash_evals_20[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_21[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_22[16] <== Poseidon2(4, 16)([evals[88][0],evals[88][1],evals[88][2],evals[89][0],evals[89][1],evals[89][2],evals[90][0],evals[90][1],evals[90][2],evals[91][0],evals[91][1],evals[91][2]], [transcriptHash_evals_21[0],transcriptHash_evals_21[1],transcriptHash_evals_21[2],transcriptHash_evals_21[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_22[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_23[16] <== Poseidon2(4, 16)([evals[92][0],evals[92][1],evals[92][2],evals[93][0],evals[93][1],evals[93][2],evals[94][0],evals[94][1],evals[94][2],evals[95][0],evals[95][1],evals[95][2]], [transcriptHash_evals_22[0],transcriptHash_evals_22[1],transcriptHash_evals_22[2],transcriptHash_evals_22[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_23[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_24[16] <== Poseidon2(4, 16)([evals[96][0],evals[96][1],evals[96][2],evals[97][0],evals[97][1],evals[97][2],evals[98][0],evals[98][1],evals[98][2],evals[99][0],evals[99][1],evals[99][2]], [transcriptHash_evals_23[0],transcriptHash_evals_23[1],transcriptHash_evals_23[2],transcriptHash_evals_23[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_24[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_25[16] <== Poseidon2(4, 16)([evals[100][0],evals[100][1],evals[100][2],evals[101][0],evals[101][1],evals[101][2],evals[102][0],evals[102][1],evals[102][2],evals[103][0],evals[103][1],evals[103][2]], [transcriptHash_evals_24[0],transcriptHash_evals_24[1],transcriptHash_evals_24[2],transcriptHash_evals_24[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_25[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_26[16] <== Poseidon2(4, 16)([evals[104][0],evals[104][1],evals[104][2],evals[105][0],evals[105][1],evals[105][2],evals[106][0],evals[106][1],evals[106][2],evals[107][0],evals[107][1],evals[107][2]], [transcriptHash_evals_25[0],transcriptHash_evals_25[1],transcriptHash_evals_25[2],transcriptHash_evals_25[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_26[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_27[16] <== Poseidon2(4, 16)([evals[108][0],evals[108][1],evals[108][2],evals[109][0],evals[109][1],evals[109][2],evals[110][0],evals[110][1],evals[110][2],evals[111][0],evals[111][1],evals[111][2]], [transcriptHash_evals_26[0],transcriptHash_evals_26[1],transcriptHash_evals_26[2],transcriptHash_evals_26[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_27[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_28[16] <== Poseidon2(4, 16)([evals[112][0],evals[112][1],evals[112][2],evals[113][0],evals[113][1],evals[113][2],evals[114][0],evals[114][1],evals[114][2],evals[115][0],evals[115][1],evals[115][2]], [transcriptHash_evals_27[0],transcriptHash_evals_27[1],transcriptHash_evals_27[2],transcriptHash_evals_27[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_28[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_29[16] <== Poseidon2(4, 16)([evals[116][0],evals[116][1],evals[116][2],evals[117][0],evals[117][1],evals[117][2],evals[118][0],evals[118][1],evals[118][2],evals[119][0],evals[119][1],evals[119][2]], [transcriptHash_evals_28[0],transcriptHash_evals_28[1],transcriptHash_evals_28[2],transcriptHash_evals_28[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_29[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_30[16] <== Poseidon2(4, 16)([evals[120][0],evals[120][1],evals[120][2],evals[121][0],evals[121][1],evals[121][2],evals[122][0],evals[122][1],evals[122][2],evals[123][0],evals[123][1],evals[123][2]], [transcriptHash_evals_29[0],transcriptHash_evals_29[1],transcriptHash_evals_29[2],transcriptHash_evals_29[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_30[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_31[16] <== Poseidon2(4, 16)([evals[124][0],evals[124][1],evals[124][2],evals[125][0],evals[125][1],evals[125][2],evals[126][0],evals[126][1],evals[126][2],evals[127][0],evals[127][1],evals[127][2]], [transcriptHash_evals_30[0],transcriptHash_evals_30[1],transcriptHash_evals_30[2],transcriptHash_evals_30[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_31[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_32[16] <== Poseidon2(4, 16)([evals[128][0],evals[128][1],evals[128][2],evals[129][0],evals[129][1],evals[129][2],evals[130][0],evals[130][1],evals[130][2],evals[131][0],evals[131][1],evals[131][2]], [transcriptHash_evals_31[0],transcriptHash_evals_31[1],transcriptHash_evals_31[2],transcriptHash_evals_31[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_32[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_33[16] <== Poseidon2(4, 16)([evals[132][0],evals[132][1],evals[132][2],evals[133][0],evals[133][1],evals[133][2],evals[134][0],evals[134][1],evals[134][2],evals[135][0],evals[135][1],evals[135][2]], [transcriptHash_evals_32[0],transcriptHash_evals_32[1],transcriptHash_evals_32[2],transcriptHash_evals_32[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_33[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_34[16] <== Poseidon2(4, 16)([evals[136][0],evals[136][1],evals[136][2],evals[137][0],evals[137][1],evals[137][2],evals[138][0],evals[138][1],evals[138][2],evals[139][0],evals[139][1],evals[139][2]], [transcriptHash_evals_33[0],transcriptHash_evals_33[1],transcriptHash_evals_33[2],transcriptHash_evals_33[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_34[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_35[16] <== Poseidon2(4, 16)([evals[140][0],evals[140][1],evals[140][2],evals[141][0],evals[141][1],evals[141][2],evals[142][0],evals[142][1],evals[142][2],evals[143][0],evals[143][1],evals[143][2]], [transcriptHash_evals_34[0],transcriptHash_evals_34[1],transcriptHash_evals_34[2],transcriptHash_evals_34[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_35[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_36[16] <== Poseidon2(4, 16)([evals[144][0],evals[144][1],evals[144][2],evals[145][0],evals[145][1],evals[145][2],evals[146][0],evals[146][1],evals[146][2],evals[147][0],evals[147][1],evals[147][2]], [transcriptHash_evals_35[0],transcriptHash_evals_35[1],transcriptHash_evals_35[2],transcriptHash_evals_35[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_36[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_37[16] <== Poseidon2(4, 16)([evals[148][0],evals[148][1],evals[148][2],evals[149][0],evals[149][1],evals[149][2],evals[150][0],evals[150][1],evals[150][2],evals[151][0],evals[151][1],evals[151][2]], [transcriptHash_evals_36[0],transcriptHash_evals_36[1],transcriptHash_evals_36[2],transcriptHash_evals_36[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_37[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_38[16] <== Poseidon2(4, 16)([evals[152][0],evals[152][1],evals[152][2],evals[153][0],evals[153][1],evals[153][2],evals[154][0],evals[154][1],evals[154][2],evals[155][0],evals[155][1],evals[155][2]], [transcriptHash_evals_37[0],transcriptHash_evals_37[1],transcriptHash_evals_37[2],transcriptHash_evals_37[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_38[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_39[16] <== Poseidon2(4, 16)([evals[156][0],evals[156][1],evals[156][2],evals[157][0],evals[157][1],evals[157][2],evals[158][0],evals[158][1],evals[158][2],evals[159][0],evals[159][1],evals[159][2]], [transcriptHash_evals_38[0],transcriptHash_evals_38[1],transcriptHash_evals_38[2],transcriptHash_evals_38[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_39[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_40[16] <== Poseidon2(4, 16)([evals[160][0],evals[160][1],evals[160][2],evals[161][0],evals[161][1],evals[161][2],evals[162][0],evals[162][1],evals[162][2],evals[163][0],evals[163][1],evals[163][2]], [transcriptHash_evals_39[0],transcriptHash_evals_39[1],transcriptHash_evals_39[2],transcriptHash_evals_39[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_40[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_41[16] <== Poseidon2(4, 16)([evals[164][0],evals[164][1],evals[164][2],evals[165][0],evals[165][1],evals[165][2],evals[166][0],evals[166][1],evals[166][2],evals[167][0],evals[167][1],evals[167][2]], [transcriptHash_evals_40[0],transcriptHash_evals_40[1],transcriptHash_evals_40[2],transcriptHash_evals_40[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_41[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_42[16] <== Poseidon2(4, 16)([evals[168][0],evals[168][1],evals[168][2],evals[169][0],evals[169][1],evals[169][2],evals[170][0],evals[170][1],evals[170][2],evals[171][0],evals[171][1],evals[171][2]], [transcriptHash_evals_41[0],transcriptHash_evals_41[1],transcriptHash_evals_41[2],transcriptHash_evals_41[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_42[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_43[16] <== Poseidon2(4, 16)([evals[172][0],evals[172][1],evals[172][2],evals[173][0],evals[173][1],evals[173][2],evals[174][0],evals[174][1],evals[174][2],evals[175][0],evals[175][1],evals[175][2]], [transcriptHash_evals_42[0],transcriptHash_evals_42[1],transcriptHash_evals_42[2],transcriptHash_evals_42[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_43[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_44[16] <== Poseidon2(4, 16)([evals[176][0],evals[176][1],evals[176][2],evals[177][0],evals[177][1],evals[177][2],evals[178][0],evals[178][1],evals[178][2],evals[179][0],evals[179][1],evals[179][2]], [transcriptHash_evals_43[0],transcriptHash_evals_43[1],transcriptHash_evals_43[2],transcriptHash_evals_43[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_44[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_45[16] <== Poseidon2(4, 16)([evals[180][0],evals[180][1],evals[180][2],evals[181][0],evals[181][1],evals[181][2],evals[182][0],evals[182][1],evals[182][2],evals[183][0],evals[183][1],evals[183][2]], [transcriptHash_evals_44[0],transcriptHash_evals_44[1],transcriptHash_evals_44[2],transcriptHash_evals_44[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_45[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_46[16] <== Poseidon2(4, 16)([evals[184][0],evals[184][1],evals[184][2],evals[185][0],evals[185][1],evals[185][2],evals[186][0],evals[186][1],evals[186][2],evals[187][0],evals[187][1],evals[187][2]], [transcriptHash_evals_45[0],transcriptHash_evals_45[1],transcriptHash_evals_45[2],transcriptHash_evals_45[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_46[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_47[16] <== Poseidon2(4, 16)([evals[188][0],evals[188][1],evals[188][2],evals[189][0],evals[189][1],evals[189][2],evals[190][0],evals[190][1],evals[190][2],evals[191][0],evals[191][1],evals[191][2]], [transcriptHash_evals_46[0],transcriptHash_evals_46[1],transcriptHash_evals_46[2],transcriptHash_evals_46[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_47[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_48[16] <== Poseidon2(4, 16)([evals[192][0],evals[192][1],evals[192][2],evals[193][0],evals[193][1],evals[193][2],evals[194][0],evals[194][1],evals[194][2],evals[195][0],evals[195][1],evals[195][2]], [transcriptHash_evals_47[0],transcriptHash_evals_47[1],transcriptHash_evals_47[2],transcriptHash_evals_47[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_48[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_49[16] <== Poseidon2(4, 16)([evals[196][0],evals[196][1],evals[196][2],evals[197][0],evals[197][1],evals[197][2],0,0,0,0,0,0], [transcriptHash_evals_48[0],transcriptHash_evals_48[1],transcriptHash_evals_48[2],transcriptHash_evals_48[3]]);
    evalsHash <== [transcriptHash_evals_49[0], transcriptHash_evals_49[1], transcriptHash_evals_49[2], transcriptHash_evals_49[3]];
    
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_3[16] <== Poseidon2(4, 16)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0,0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesFRI[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesFRI[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    challengesFRISteps[0] <== [transcriptHash_3[6], transcriptHash_3[7], transcriptHash_3[8]];
    for(var i = 9; i < 16; i++){
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_4[16] <== Poseidon2(4, 16)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0,0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    challengesFRISteps[1] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_5[16] <== Poseidon2(4, 16)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0,0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    challengesFRISteps[2] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_6[16] <== Poseidon2(4, 16)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0,0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    challengesFRISteps[3] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_7[16] <== Poseidon2(4, 16)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0,0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    challengesFRISteps[4] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_8[16] <== Poseidon2(4, 16)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0,0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    challengesFRISteps[5] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];
    
    signal transcriptHash_lastPolFRI_0[16] <== Poseidon2(4, 16)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1],finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_1[16] <== Poseidon2(4, 16)([finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0],finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_2[16] <== Poseidon2(4, 16)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1],finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_3[16] <== Poseidon2(4, 16)([finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0],finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_4[16] <== Poseidon2(4, 16)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1],finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_5[16] <== Poseidon2(4, 16)([finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0],finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_6[16] <== Poseidon2(4, 16)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1],finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_7[16] <== Poseidon2(4, 16)([finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0],finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_7[0], transcriptHash_lastPolFRI_7[1], transcriptHash_lastPolFRI_7[2], transcriptHash_lastPolFRI_7[3]];

    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_9[16] <== Poseidon2(4, 16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    challengesFRISteps[6] <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];

    queriesFRI <== calculateFRIQueries0()(challengesFRISteps[6], nonce, enable);
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

template VerifyEvaluationsChunks0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[198][3];
    signal input publics[391];

    signal input Zh[3];


    signal output tmp_3864[3];
 
    signal output tmp_3866[3];
 
    signal output tmp_3978[3];
 
    signal output tmp_4448[3];
 
    signal output tmp_4451[3];
 
    signal output tmp_4452[3];
 
    signal output tmp_4472[3];
 
    signal output tmp_4475[3];
 
    signal output tmp_4476[3];
 
    signal output tmp_4495[3];
 
    signal output tmp_4498[3];
 
    signal output tmp_4499[3];
 
    signal output tmp_4503[3];
 
    signal output tmp_4524[3];
 
    signal output tmp_4527[3];
 
    signal output tmp_4528[3];
 
    signal output tmp_4533[3];
 
    signal output tmp_4534[3];
 
    signal output tmp_4535[3];
 
    signal output tmp_4553[3];
 
    signal output tmp_4556[3];
 
    signal output tmp_4557[3];
 
    signal output tmp_4577[3];
 
    signal output tmp_4580[3];
 
    signal output tmp_4581[3];
 
    signal output tmp_4600[3];
 
    signal output tmp_4603[3];
 
    signal output tmp_4604[3];
 
    signal output tmp_4608[3];
 
    signal output tmp_4629[3];
 
    signal output tmp_4632[3];
 
    signal output tmp_4633[3];
 
    signal output tmp_4638[3];
 
    signal output tmp_4639[3];
 
    signal output tmp_4640[3];
 
    signal output tmp_4641[3];
 
    signal output tmp_4659[3];
 
    signal output tmp_4662[3];
 
    signal output tmp_4663[3];
 
    signal output tmp_4683[3];
 
    signal output tmp_4686[3];
 
    signal output tmp_4687[3];
 
    signal output tmp_4706[3];
 
    signal output tmp_4709[3];
 
    signal output tmp_4710[3];
 
    signal output tmp_4714[3];
 
    signal output tmp_4735[3];
 
    signal output tmp_4738[3];
 
    signal output tmp_4739[3];
 
    signal output tmp_4744[3];
 
    signal output tmp_4745[3];
 
    signal output tmp_4746[3];
 
    signal output tmp_4747[3];
 
    signal output tmp_4765[3];
 
    signal output tmp_4768[3];
 
    signal output tmp_4769[3];
 
    signal output tmp_4789[3];
 
    signal output tmp_4792[3];
 
    signal output tmp_4793[3];
 
    signal output tmp_4812[3];
 
    signal output tmp_4815[3];
 
    signal output tmp_4816[3];
 
    signal output tmp_4820[3];
 
    signal output tmp_4841[3];
 
    signal output tmp_4844[3];
 
    signal output tmp_4845[3];
 
    signal output tmp_4850[3];
 
    signal output tmp_4851[3];
 
    signal output tmp_4852[3];
 
    signal output tmp_4853[3];
 
    signal output tmp_4854[3];
 
    signal output tmp_4859[3];
 
    signal output tmp_4862[3];
 
    tmp_3864 <== [evals[0][0] + evals[1][0], evals[0][1] + evals[1][1], evals[0][2] + evals[1][2]];
    signal tmp_3865[3] <== [evals[83][0] + tmp_3864[0], evals[83][1] + tmp_3864[1], evals[83][2] + tmp_3864[2]];
    tmp_3866 <== [evals[20][0] + evals[21][0], evals[20][1] + evals[21][1], evals[20][2] + evals[21][2]];
    signal tmp_3867[3] <== [tmp_3865[0] + tmp_3866[0], tmp_3865[1] + tmp_3866[1], tmp_3865[2] + tmp_3866[2]];
    signal tmp_3868[3] <== [tmp_3867[0] + evals[196][0], tmp_3867[1] + evals[196][1], tmp_3867[2] + evals[196][2]];
    signal tmp_3869[3] <== [tmp_3868[0] + evals[177][0], tmp_3868[1] + evals[177][1], tmp_3868[2] + evals[177][2]];
    signal tmp_3870[3] <== [tmp_3869[0] + evals[76][0], tmp_3869[1] + evals[76][1], tmp_3869[2] + evals[76][2]];
    signal tmp_3871[3] <== [tmp_3870[0] + evals[2][0], tmp_3870[1] + evals[2][1], tmp_3870[2] + evals[2][2]];
    signal tmp_3872[3] <== [tmp_3871[0] + evals[197][0], tmp_3871[1] + evals[197][1], tmp_3871[2] + evals[197][2]];
    signal tmp_3873[3] <== [tmp_3872[0] + evals[178][0], tmp_3872[1] + evals[178][1], tmp_3872[2] + evals[178][2]];
    signal tmp_3874[3] <== CMul()(evals[86], evals[87]);
    signal tmp_3875[3] <== CMul()(evals[64], tmp_3874);
    signal tmp_3876[3] <== CMul()(evals[65], evals[86]);
    signal tmp_3877[3] <== [tmp_3875[0] + tmp_3876[0], tmp_3875[1] + tmp_3876[1], tmp_3875[2] + tmp_3876[2]];
    signal tmp_3878[3] <== CMul()(evals[66], evals[87]);
    signal tmp_3879[3] <== [tmp_3877[0] + tmp_3878[0], tmp_3877[1] + tmp_3878[1], tmp_3877[2] + tmp_3878[2]];
    signal tmp_3880[3] <== CMul()(evals[67], evals[88]);
    signal tmp_3881[3] <== [tmp_3879[0] + tmp_3880[0], tmp_3879[1] + tmp_3880[1], tmp_3879[2] + tmp_3880[2]];
    signal tmp_3882[3] <== [tmp_3881[0] + evals[68][0], tmp_3881[1] + evals[68][1], tmp_3881[2] + evals[68][2]];
    signal tmp_3883[3] <== CMul()(tmp_3873, tmp_3882);
    signal tmp_3884[3] <== CMul()(challengeQ, tmp_3883);
    signal tmp_3885[3] <== CMul()(evals[89], evals[90]);
    signal tmp_3886[3] <== CMul()(evals[64], tmp_3885);
    signal tmp_3887[3] <== CMul()(evals[65], evals[89]);
    signal tmp_3888[3] <== [tmp_3886[0] + tmp_3887[0], tmp_3886[1] + tmp_3887[1], tmp_3886[2] + tmp_3887[2]];
    signal tmp_3889[3] <== CMul()(evals[66], evals[90]);
    signal tmp_3890[3] <== [tmp_3888[0] + tmp_3889[0], tmp_3888[1] + tmp_3889[1], tmp_3888[2] + tmp_3889[2]];
    signal tmp_3891[3] <== CMul()(evals[67], evals[91]);
    signal tmp_3892[3] <== [tmp_3890[0] + tmp_3891[0], tmp_3890[1] + tmp_3891[1], tmp_3890[2] + tmp_3891[2]];
    signal tmp_3893[3] <== [tmp_3892[0] + evals[68][0], tmp_3892[1] + evals[68][1], tmp_3892[2] + evals[68][2]];
    signal tmp_3894[3] <== CMul()(tmp_3873, tmp_3893);
    signal tmp_3895[3] <== [tmp_3884[0] + tmp_3894[0], tmp_3884[1] + tmp_3894[1], tmp_3884[2] + tmp_3894[2]];
    signal tmp_3896[3] <== CMul()(challengeQ, tmp_3895);
    signal tmp_3897[3] <== CMul()(evals[92], evals[93]);
    signal tmp_3898[3] <== CMul()(evals[64], tmp_3897);
    signal tmp_3899[3] <== CMul()(evals[65], evals[92]);
    signal tmp_3900[3] <== [tmp_3898[0] + tmp_3899[0], tmp_3898[1] + tmp_3899[1], tmp_3898[2] + tmp_3899[2]];
    signal tmp_3901[3] <== CMul()(evals[66], evals[93]);
    signal tmp_3902[3] <== [tmp_3900[0] + tmp_3901[0], tmp_3900[1] + tmp_3901[1], tmp_3900[2] + tmp_3901[2]];
    signal tmp_3903[3] <== CMul()(evals[67], evals[94]);
    signal tmp_3904[3] <== [tmp_3902[0] + tmp_3903[0], tmp_3902[1] + tmp_3903[1], tmp_3902[2] + tmp_3903[2]];
    signal tmp_3905[3] <== [tmp_3904[0] + evals[68][0], tmp_3904[1] + evals[68][1], tmp_3904[2] + evals[68][2]];
    signal tmp_3906[3] <== CMul()(tmp_3873, tmp_3905);
    signal tmp_3907[3] <== [tmp_3896[0] + tmp_3906[0], tmp_3896[1] + tmp_3906[1], tmp_3896[2] + tmp_3906[2]];
    signal tmp_3908[3] <== CMul()(challengeQ, tmp_3907);
    signal tmp_3909[3] <== CMul()(evals[95], evals[96]);
    signal tmp_3910[3] <== CMul()(evals[64], tmp_3909);
    signal tmp_3911[3] <== CMul()(evals[65], evals[95]);
    signal tmp_3912[3] <== [tmp_3910[0] + tmp_3911[0], tmp_3910[1] + tmp_3911[1], tmp_3910[2] + tmp_3911[2]];
    signal tmp_3913[3] <== CMul()(evals[66], evals[96]);
    signal tmp_3914[3] <== [tmp_3912[0] + tmp_3913[0], tmp_3912[1] + tmp_3913[1], tmp_3912[2] + tmp_3913[2]];
    signal tmp_3915[3] <== CMul()(evals[67], evals[97]);
    signal tmp_3916[3] <== [tmp_3914[0] + tmp_3915[0], tmp_3914[1] + tmp_3915[1], tmp_3914[2] + tmp_3915[2]];
    signal tmp_3917[3] <== [tmp_3916[0] + evals[68][0], tmp_3916[1] + evals[68][1], tmp_3916[2] + evals[68][2]];
    signal tmp_3918[3] <== CMul()(tmp_3873, tmp_3917);
    signal tmp_3919[3] <== [tmp_3908[0] + tmp_3918[0], tmp_3908[1] + tmp_3918[1], tmp_3908[2] + tmp_3918[2]];
    signal tmp_3920[3] <== CMul()(challengeQ, tmp_3919);
    signal tmp_3921[3] <== CMul()(evals[98], evals[99]);
    signal tmp_3922[3] <== CMul()(evals[64], tmp_3921);
    signal tmp_3923[3] <== CMul()(evals[65], evals[98]);
    signal tmp_3924[3] <== [tmp_3922[0] + tmp_3923[0], tmp_3922[1] + tmp_3923[1], tmp_3922[2] + tmp_3923[2]];
    signal tmp_3925[3] <== CMul()(evals[66], evals[99]);
    signal tmp_3926[3] <== [tmp_3924[0] + tmp_3925[0], tmp_3924[1] + tmp_3925[1], tmp_3924[2] + tmp_3925[2]];
    signal tmp_3927[3] <== CMul()(evals[67], evals[100]);
    signal tmp_3928[3] <== [tmp_3926[0] + tmp_3927[0], tmp_3926[1] + tmp_3927[1], tmp_3926[2] + tmp_3927[2]];
    signal tmp_3929[3] <== [tmp_3928[0] + evals[68][0], tmp_3928[1] + evals[68][1], tmp_3928[2] + evals[68][2]];
    signal tmp_3930[3] <== CMul()(tmp_3873, tmp_3929);
    signal tmp_3931[3] <== [tmp_3920[0] + tmp_3930[0], tmp_3920[1] + tmp_3930[1], tmp_3920[2] + tmp_3930[2]];
    signal tmp_3932[3] <== CMul()(challengeQ, tmp_3931);
    signal tmp_3933[3] <== CMul()(evals[101], evals[102]);
    signal tmp_3934[3] <== CMul()(evals[64], tmp_3933);
    signal tmp_3935[3] <== CMul()(evals[65], evals[101]);
    signal tmp_3936[3] <== [tmp_3934[0] + tmp_3935[0], tmp_3934[1] + tmp_3935[1], tmp_3934[2] + tmp_3935[2]];
    signal tmp_3937[3] <== CMul()(evals[66], evals[102]);
    signal tmp_3938[3] <== [tmp_3936[0] + tmp_3937[0], tmp_3936[1] + tmp_3937[1], tmp_3936[2] + tmp_3937[2]];
    signal tmp_3939[3] <== CMul()(evals[67], evals[103]);
    signal tmp_3940[3] <== [tmp_3938[0] + tmp_3939[0], tmp_3938[1] + tmp_3939[1], tmp_3938[2] + tmp_3939[2]];
    signal tmp_3941[3] <== [tmp_3940[0] + evals[68][0], tmp_3940[1] + evals[68][1], tmp_3940[2] + evals[68][2]];
    signal tmp_3942[3] <== CMul()(tmp_3873, tmp_3941);
    signal tmp_3943[3] <== [tmp_3932[0] + tmp_3942[0], tmp_3932[1] + tmp_3942[1], tmp_3932[2] + tmp_3942[2]];
    signal tmp_3944[3] <== CMul()(challengeQ, tmp_3943);
    signal tmp_3945[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_3946[3] <== [tmp_3945[0] + evals[77][0], tmp_3945[1] + evals[77][1], tmp_3945[2] + evals[77][2]];
    signal tmp_3947[3] <== CMul()(evals[104], evals[105]);
    signal tmp_3948[3] <== CMul()(evals[69], tmp_3947);
    signal tmp_3949[3] <== CMul()(evals[70], evals[104]);
    signal tmp_3950[3] <== [tmp_3948[0] + tmp_3949[0], tmp_3948[1] + tmp_3949[1], tmp_3948[2] + tmp_3949[2]];
    signal tmp_3951[3] <== CMul()(evals[71], evals[105]);
    signal tmp_3952[3] <== [tmp_3950[0] + tmp_3951[0], tmp_3950[1] + tmp_3951[1], tmp_3950[2] + tmp_3951[2]];
    signal tmp_3953[3] <== CMul()(evals[72], evals[106]);
    signal tmp_3954[3] <== [tmp_3952[0] + tmp_3953[0], tmp_3952[1] + tmp_3953[1], tmp_3952[2] + tmp_3953[2]];
    signal tmp_3955[3] <== [tmp_3954[0] + evals[73][0], tmp_3954[1] + evals[73][1], tmp_3954[2] + evals[73][2]];
    signal tmp_3956[3] <== CMul()(tmp_3946, tmp_3955);
    signal tmp_3957[3] <== [tmp_3944[0] + tmp_3956[0], tmp_3944[1] + tmp_3956[1], tmp_3944[2] + tmp_3956[2]];
    signal tmp_3958[3] <== CMul()(challengeQ, tmp_3957);
    signal tmp_3959[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_3960[3] <== [tmp_3959[0] + evals[77][0], tmp_3959[1] + evals[77][1], tmp_3959[2] + evals[77][2]];
    signal tmp_3961[3] <== [tmp_3960[0] + evals[79][0], tmp_3960[1] + evals[79][1], tmp_3960[2] + evals[79][2]];
    signal tmp_3962[3] <== CMul()(evals[107], evals[108]);
    signal tmp_3963[3] <== CMul()(evals[69], tmp_3962);
    signal tmp_3964[3] <== CMul()(evals[70], evals[107]);
    signal tmp_3965[3] <== [tmp_3963[0] + tmp_3964[0], tmp_3963[1] + tmp_3964[1], tmp_3963[2] + tmp_3964[2]];
    signal tmp_3966[3] <== CMul()(evals[71], evals[108]);
    signal tmp_3967[3] <== [tmp_3965[0] + tmp_3966[0], tmp_3965[1] + tmp_3966[1], tmp_3965[2] + tmp_3966[2]];
    signal tmp_3968[3] <== CMul()(evals[72], evals[109]);
    signal tmp_3969[3] <== [tmp_3967[0] + tmp_3968[0], tmp_3967[1] + tmp_3968[1], tmp_3967[2] + tmp_3968[2]];
    signal tmp_3970[3] <== [tmp_3969[0] + evals[73][0], tmp_3969[1] + evals[73][1], tmp_3969[2] + evals[73][2]];
    signal tmp_3971[3] <== CMul()(tmp_3961, tmp_3970);
    signal tmp_3972[3] <== [tmp_3958[0] + tmp_3971[0], tmp_3958[1] + tmp_3971[1], tmp_3958[2] + tmp_3971[2]];
    signal tmp_3973[3] <== CMul()(challengeQ, tmp_3972);
    signal tmp_3974[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_3975[3] <== [tmp_3974[0] + evals[77][0], tmp_3974[1] + evals[77][1], tmp_3974[2] + evals[77][2]];
    signal tmp_3976[3] <== [tmp_3975[0] + evals[79][0], tmp_3975[1] + evals[79][1], tmp_3975[2] + evals[79][2]];
    signal tmp_3977[3] <== [tmp_3976[0] + evals[82][0], tmp_3976[1] + evals[82][1], tmp_3976[2] + evals[82][2]];
    tmp_3978 <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_3979[3] <== [tmp_3977[0] + tmp_3978[0], tmp_3977[1] + tmp_3978[1], tmp_3977[2] + tmp_3978[2]];
    signal tmp_3980[3] <== CMul()(evals[110], evals[111]);
    signal tmp_3981[3] <== CMul()(evals[69], tmp_3980);
    signal tmp_3982[3] <== CMul()(evals[70], evals[110]);
    signal tmp_3983[3] <== [tmp_3981[0] + tmp_3982[0], tmp_3981[1] + tmp_3982[1], tmp_3981[2] + tmp_3982[2]];
    signal tmp_3984[3] <== CMul()(evals[71], evals[111]);
    signal tmp_3985[3] <== [tmp_3983[0] + tmp_3984[0], tmp_3983[1] + tmp_3984[1], tmp_3983[2] + tmp_3984[2]];
    signal tmp_3986[3] <== CMul()(evals[72], evals[112]);
    signal tmp_3987[3] <== [tmp_3985[0] + tmp_3986[0], tmp_3985[1] + tmp_3986[1], tmp_3985[2] + tmp_3986[2]];
    signal tmp_3988[3] <== [tmp_3987[0] + evals[73][0], tmp_3987[1] + evals[73][1], tmp_3987[2] + evals[73][2]];
    signal tmp_3989[3] <== CMul()(tmp_3979, tmp_3988);
    signal tmp_3990[3] <== [tmp_3973[0] + tmp_3989[0], tmp_3973[1] + tmp_3989[1], tmp_3973[2] + tmp_3989[2]];
    signal tmp_3991[3] <== CMul()(challengeQ, tmp_3990);
    signal tmp_3992[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_3993[3] <== [tmp_3992[0] + evals[77][0], tmp_3992[1] + evals[77][1], tmp_3992[2] + evals[77][2]];
    signal tmp_3994[3] <== [tmp_3993[0] + evals[79][0], tmp_3993[1] + evals[79][1], tmp_3993[2] + evals[79][2]];
    signal tmp_3995[3] <== [tmp_3994[0] + evals[82][0], tmp_3994[1] + evals[82][1], tmp_3994[2] + evals[82][2]];
    signal tmp_3996[3] <== [tmp_3995[0] + tmp_3978[0], tmp_3995[1] + tmp_3978[1], tmp_3995[2] + tmp_3978[2]];
    signal tmp_3997[3] <== CMul()(evals[113], evals[114]);
    signal tmp_3998[3] <== CMul()(evals[69], tmp_3997);
    signal tmp_3999[3] <== CMul()(evals[70], evals[113]);
    signal tmp_4000[3] <== [tmp_3998[0] + tmp_3999[0], tmp_3998[1] + tmp_3999[1], tmp_3998[2] + tmp_3999[2]];
    signal tmp_4001[3] <== CMul()(evals[71], evals[114]);
    signal tmp_4002[3] <== [tmp_4000[0] + tmp_4001[0], tmp_4000[1] + tmp_4001[1], tmp_4000[2] + tmp_4001[2]];
    signal tmp_4003[3] <== CMul()(evals[72], evals[115]);
    signal tmp_4004[3] <== [tmp_4002[0] + tmp_4003[0], tmp_4002[1] + tmp_4003[1], tmp_4002[2] + tmp_4003[2]];
    signal tmp_4005[3] <== [tmp_4004[0] + evals[73][0], tmp_4004[1] + evals[73][1], tmp_4004[2] + evals[73][2]];
    signal tmp_4006[3] <== CMul()(tmp_3996, tmp_4005);
    signal tmp_4007[3] <== [tmp_3991[0] + tmp_4006[0], tmp_3991[1] + tmp_4006[1], tmp_3991[2] + tmp_4006[2]];
    signal tmp_4008[3] <== CMul()(challengeQ, tmp_4007);
    signal tmp_4009[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_4010[3] <== [tmp_4009[0] + evals[77][0], tmp_4009[1] + evals[77][1], tmp_4009[2] + evals[77][2]];
    signal tmp_4011[3] <== [tmp_4010[0] + evals[79][0], tmp_4010[1] + evals[79][1], tmp_4010[2] + evals[79][2]];
    signal tmp_4012[3] <== [tmp_4011[0] + evals[82][0], tmp_4011[1] + evals[82][1], tmp_4011[2] + evals[82][2]];
    signal tmp_4013[3] <== [tmp_4012[0] + tmp_3978[0], tmp_4012[1] + tmp_3978[1], tmp_4012[2] + tmp_3978[2]];
    signal tmp_4014[3] <== CMul()(evals[116], evals[117]);
    signal tmp_4015[3] <== CMul()(evals[69], tmp_4014);
    signal tmp_4016[3] <== CMul()(evals[70], evals[116]);
    signal tmp_4017[3] <== [tmp_4015[0] + tmp_4016[0], tmp_4015[1] + tmp_4016[1], tmp_4015[2] + tmp_4016[2]];
    signal tmp_4018[3] <== CMul()(evals[71], evals[117]);
    signal tmp_4019[3] <== [tmp_4017[0] + tmp_4018[0], tmp_4017[1] + tmp_4018[1], tmp_4017[2] + tmp_4018[2]];
    signal tmp_4020[3] <== CMul()(evals[72], evals[118]);
    signal tmp_4021[3] <== [tmp_4019[0] + tmp_4020[0], tmp_4019[1] + tmp_4020[1], tmp_4019[2] + tmp_4020[2]];
    signal tmp_4022[3] <== [tmp_4021[0] + evals[73][0], tmp_4021[1] + evals[73][1], tmp_4021[2] + evals[73][2]];
    signal tmp_4023[3] <== CMul()(tmp_4013, tmp_4022);
    signal tmp_4024[3] <== [tmp_4008[0] + tmp_4023[0], tmp_4008[1] + tmp_4023[1], tmp_4008[2] + tmp_4023[2]];
    signal tmp_4025[3] <== CMul()(challengeQ, tmp_4024);
    signal tmp_4026[3] <== [tmp_3873[0] + evals[81][0], tmp_3873[1] + evals[81][1], tmp_3873[2] + evals[81][2]];
    signal tmp_4027[3] <== [tmp_4026[0] + evals[77][0], tmp_4026[1] + evals[77][1], tmp_4026[2] + evals[77][2]];
    signal tmp_4028[3] <== [tmp_4027[0] + evals[79][0], tmp_4027[1] + evals[79][1], tmp_4027[2] + evals[79][2]];
    signal tmp_4029[3] <== [tmp_4028[0] + evals[82][0], tmp_4028[1] + evals[82][1], tmp_4028[2] + evals[82][2]];
    signal tmp_4030[3] <== [tmp_4029[0] + tmp_3978[0], tmp_4029[1] + tmp_3978[1], tmp_4029[2] + tmp_3978[2]];
    signal tmp_4031[3] <== CMul()(evals[119], evals[120]);
    signal tmp_4032[3] <== CMul()(evals[69], tmp_4031);
    signal tmp_4033[3] <== CMul()(evals[70], evals[119]);
    signal tmp_4034[3] <== [tmp_4032[0] + tmp_4033[0], tmp_4032[1] + tmp_4033[1], tmp_4032[2] + tmp_4033[2]];
    signal tmp_4035[3] <== CMul()(evals[71], evals[120]);
    signal tmp_4036[3] <== [tmp_4034[0] + tmp_4035[0], tmp_4034[1] + tmp_4035[1], tmp_4034[2] + tmp_4035[2]];
    signal tmp_4037[3] <== CMul()(evals[72], evals[121]);
    signal tmp_4038[3] <== [tmp_4036[0] + tmp_4037[0], tmp_4036[1] + tmp_4037[1], tmp_4036[2] + tmp_4037[2]];
    signal tmp_4039[3] <== [tmp_4038[0] + evals[73][0], tmp_4038[1] + evals[73][1], tmp_4038[2] + evals[73][2]];
    signal tmp_4040[3] <== CMul()(tmp_4030, tmp_4039);
    signal tmp_4041[3] <== [tmp_4025[0] + tmp_4040[0], tmp_4025[1] + tmp_4040[1], tmp_4025[2] + tmp_4040[2]];
    signal tmp_4042[3] <== CMul()(challengeQ, tmp_4041);
    signal tmp_4043[3] <== CMul()(evals[75], evals[102]);
    signal tmp_4044[3] <== [evals[102][0] - 1, evals[102][1], evals[102][2]];
    signal tmp_4045[3] <== CMul()(tmp_4043, tmp_4044);
    signal tmp_4046[3] <== [tmp_4042[0] + tmp_4045[0], tmp_4042[1] + tmp_4045[1], tmp_4042[2] + tmp_4045[2]];
    signal tmp_4047[3] <== CMul()(challengeQ, tmp_4046);
    signal tmp_4048[3] <== CMul()(evals[75], evals[103]);
    signal tmp_4049[3] <== [evals[103][0] - 1, evals[103][1], evals[103][2]];
    signal tmp_4050[3] <== CMul()(tmp_4048, tmp_4049);
    signal tmp_4051[3] <== [tmp_4047[0] + tmp_4050[0], tmp_4047[1] + tmp_4050[1], tmp_4047[2] + tmp_4050[2]];
    signal tmp_4052[3] <== CMul()(challengeQ, tmp_4051);
    signal tmp_4053[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4054[3] <== [1 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_4055[3] <== [1 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_4056[3] <== CMul()(tmp_4054, tmp_4055);
    signal tmp_4057[3] <== CMul()(tmp_4056, evals[89]);
    signal tmp_4058[3] <== [1 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_4059[3] <== CMul()(evals[102], tmp_4058);
    signal tmp_4060[3] <== [1 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_4061[3] <== CMul()(tmp_4060, evals[103]);
    signal tmp_4062[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4063[3] <== CMul()(evals[102], evals[103]);
    signal tmp_4064[3] <== [tmp_4062[0] + tmp_4063[0], tmp_4062[1] + tmp_4063[1], tmp_4062[2] + tmp_4063[2]];
    signal tmp_4065[3] <== CMul()(tmp_4064, evals[93]);
    signal tmp_4066[3] <== [tmp_4057[0] + tmp_4065[0], tmp_4057[1] + tmp_4065[1], tmp_4057[2] + tmp_4065[2]];
    signal tmp_4067[3] <== CMul()(evals[75], tmp_4066);
    signal tmp_4068[3] <== CMul()(evals[74], evals[89]);
    signal tmp_4069[3] <== [tmp_4067[0] + tmp_4068[0], tmp_4067[1] + tmp_4068[1], tmp_4067[2] + tmp_4068[2]];
    signal tmp_4070[3] <== [2 * tmp_4069[0], 2 * tmp_4069[1], 2 * tmp_4069[2]];
    signal tmp_4071[3] <== CMul()(tmp_4056, evals[86]);
    signal tmp_4072[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4073[3] <== [tmp_4072[0] + tmp_4063[0], tmp_4072[1] + tmp_4063[1], tmp_4072[2] + tmp_4063[2]];
    signal tmp_4074[3] <== CMul()(tmp_4073, evals[90]);
    signal tmp_4075[3] <== [tmp_4071[0] + tmp_4074[0], tmp_4071[1] + tmp_4074[1], tmp_4071[2] + tmp_4074[2]];
    signal tmp_4076[3] <== CMul()(evals[75], tmp_4075);
    signal tmp_4077[3] <== CMul()(evals[74], evals[86]);
    signal tmp_4078[3] <== [tmp_4076[0] + tmp_4077[0], tmp_4076[1] + tmp_4077[1], tmp_4076[2] + tmp_4077[2]];
    signal tmp_4079[3] <== CMul()(tmp_4056, evals[87]);
    signal tmp_4080[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4081[3] <== [tmp_4080[0] + tmp_4063[0], tmp_4080[1] + tmp_4063[1], tmp_4080[2] + tmp_4063[2]];
    signal tmp_4082[3] <== CMul()(tmp_4081, evals[91]);
    signal tmp_4083[3] <== [tmp_4079[0] + tmp_4082[0], tmp_4079[1] + tmp_4082[1], tmp_4079[2] + tmp_4082[2]];
    signal tmp_4084[3] <== CMul()(evals[75], tmp_4083);
    signal tmp_4085[3] <== CMul()(evals[74], evals[87]);
    signal tmp_4086[3] <== [tmp_4084[0] + tmp_4085[0], tmp_4084[1] + tmp_4085[1], tmp_4084[2] + tmp_4085[2]];
    signal tmp_4087[3] <== [tmp_4078[0] + tmp_4086[0], tmp_4078[1] + tmp_4086[1], tmp_4078[2] + tmp_4086[2]];
    signal tmp_4088[3] <== [tmp_4070[0] + tmp_4087[0], tmp_4070[1] + tmp_4087[1], tmp_4070[2] + tmp_4087[2]];
    signal tmp_4089[3] <== [4 * tmp_4087[0], 4 * tmp_4087[1], 4 * tmp_4087[2]];
    signal tmp_4090[3] <== CMul()(tmp_4056, evals[87]);
    signal tmp_4091[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4092[3] <== [tmp_4091[0] + tmp_4063[0], tmp_4091[1] + tmp_4063[1], tmp_4091[2] + tmp_4063[2]];
    signal tmp_4093[3] <== CMul()(tmp_4092, evals[91]);
    signal tmp_4094[3] <== [tmp_4090[0] + tmp_4093[0], tmp_4090[1] + tmp_4093[1], tmp_4090[2] + tmp_4093[2]];
    signal tmp_4095[3] <== CMul()(evals[75], tmp_4094);
    signal tmp_4096[3] <== CMul()(evals[74], evals[87]);
    signal tmp_4097[3] <== [tmp_4095[0] + tmp_4096[0], tmp_4095[1] + tmp_4096[1], tmp_4095[2] + tmp_4096[2]];
    signal tmp_4098[3] <== [2 * tmp_4097[0], 2 * tmp_4097[1], 2 * tmp_4097[2]];
    signal tmp_4099[3] <== CMul()(tmp_4056, evals[88]);
    signal tmp_4100[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4101[3] <== [tmp_4100[0] + tmp_4063[0], tmp_4100[1] + tmp_4063[1], tmp_4100[2] + tmp_4063[2]];
    signal tmp_4102[3] <== CMul()(tmp_4101, evals[92]);
    signal tmp_4103[3] <== [tmp_4099[0] + tmp_4102[0], tmp_4099[1] + tmp_4102[1], tmp_4099[2] + tmp_4102[2]];
    signal tmp_4104[3] <== CMul()(evals[75], tmp_4103);
    signal tmp_4105[3] <== CMul()(evals[74], evals[88]);
    signal tmp_4106[3] <== [tmp_4104[0] + tmp_4105[0], tmp_4104[1] + tmp_4105[1], tmp_4104[2] + tmp_4105[2]];
    signal tmp_4107[3] <== CMul()(tmp_4056, evals[89]);
    signal tmp_4108[3] <== [tmp_4059[0] + tmp_4061[0], tmp_4059[1] + tmp_4061[1], tmp_4059[2] + tmp_4061[2]];
    signal tmp_4109[3] <== [tmp_4108[0] + tmp_4063[0], tmp_4108[1] + tmp_4063[1], tmp_4108[2] + tmp_4063[2]];
    signal tmp_4110[3] <== CMul()(tmp_4109, evals[93]);
    signal tmp_4111[3] <== [tmp_4107[0] + tmp_4110[0], tmp_4107[1] + tmp_4110[1], tmp_4107[2] + tmp_4110[2]];
    signal tmp_4112[3] <== CMul()(evals[75], tmp_4111);
    signal tmp_4113[3] <== CMul()(evals[74], evals[89]);
    signal tmp_4114[3] <== [tmp_4112[0] + tmp_4113[0], tmp_4112[1] + tmp_4113[1], tmp_4112[2] + tmp_4113[2]];
    signal tmp_4115[3] <== [tmp_4106[0] + tmp_4114[0], tmp_4106[1] + tmp_4114[1], tmp_4106[2] + tmp_4114[2]];
    signal tmp_4116[3] <== [tmp_4098[0] + tmp_4115[0], tmp_4098[1] + tmp_4115[1], tmp_4098[2] + tmp_4115[2]];
    signal tmp_4117[3] <== [tmp_4089[0] + tmp_4116[0], tmp_4089[1] + tmp_4116[1], tmp_4089[2] + tmp_4116[2]];
    signal tmp_4118[3] <== [tmp_4088[0] + tmp_4117[0], tmp_4088[1] + tmp_4117[1], tmp_4088[2] + tmp_4117[2]];
    signal tmp_4119[3] <== CMul()(tmp_4056, evals[93]);
    signal tmp_4120[3] <== CMul()(tmp_4059, evals[89]);
    signal tmp_4121[3] <== [tmp_4119[0] + tmp_4120[0], tmp_4119[1] + tmp_4120[1], tmp_4119[2] + tmp_4120[2]];
    signal tmp_4122[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4123[3] <== CMul()(tmp_4122, evals[97]);
    signal tmp_4124[3] <== [tmp_4121[0] + tmp_4123[0], tmp_4121[1] + tmp_4123[1], tmp_4121[2] + tmp_4123[2]];
    signal tmp_4125[3] <== CMul()(evals[75], tmp_4124);
    signal tmp_4126[3] <== CMul()(evals[74], evals[93]);
    signal tmp_4127[3] <== [tmp_4125[0] + tmp_4126[0], tmp_4125[1] + tmp_4126[1], tmp_4125[2] + tmp_4126[2]];
    signal tmp_4128[3] <== [2 * tmp_4127[0], 2 * tmp_4127[1], 2 * tmp_4127[2]];
    signal tmp_4129[3] <== CMul()(tmp_4056, evals[90]);
    signal tmp_4130[3] <== CMul()(tmp_4059, evals[86]);
    signal tmp_4131[3] <== [tmp_4129[0] + tmp_4130[0], tmp_4129[1] + tmp_4130[1], tmp_4129[2] + tmp_4130[2]];
    signal tmp_4132[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4133[3] <== CMul()(tmp_4132, evals[94]);
    signal tmp_4134[3] <== [tmp_4131[0] + tmp_4133[0], tmp_4131[1] + tmp_4133[1], tmp_4131[2] + tmp_4133[2]];
    signal tmp_4135[3] <== CMul()(evals[75], tmp_4134);
    signal tmp_4136[3] <== CMul()(evals[74], evals[90]);
    signal tmp_4137[3] <== [tmp_4135[0] + tmp_4136[0], tmp_4135[1] + tmp_4136[1], tmp_4135[2] + tmp_4136[2]];
    signal tmp_4138[3] <== CMul()(tmp_4056, evals[91]);
    signal tmp_4139[3] <== CMul()(tmp_4059, evals[87]);
    signal tmp_4140[3] <== [tmp_4138[0] + tmp_4139[0], tmp_4138[1] + tmp_4139[1], tmp_4138[2] + tmp_4139[2]];
    signal tmp_4141[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4142[3] <== CMul()(tmp_4141, evals[95]);
    signal tmp_4143[3] <== [tmp_4140[0] + tmp_4142[0], tmp_4140[1] + tmp_4142[1], tmp_4140[2] + tmp_4142[2]];
    signal tmp_4144[3] <== CMul()(evals[75], tmp_4143);
    signal tmp_4145[3] <== CMul()(evals[74], evals[91]);
    signal tmp_4146[3] <== [tmp_4144[0] + tmp_4145[0], tmp_4144[1] + tmp_4145[1], tmp_4144[2] + tmp_4145[2]];
    signal tmp_4147[3] <== [tmp_4137[0] + tmp_4146[0], tmp_4137[1] + tmp_4146[1], tmp_4137[2] + tmp_4146[2]];
    signal tmp_4148[3] <== [tmp_4128[0] + tmp_4147[0], tmp_4128[1] + tmp_4147[1], tmp_4128[2] + tmp_4147[2]];
    signal tmp_4149[3] <== [4 * tmp_4147[0], 4 * tmp_4147[1], 4 * tmp_4147[2]];
    signal tmp_4150[3] <== CMul()(tmp_4056, evals[91]);
    signal tmp_4151[3] <== CMul()(tmp_4059, evals[87]);
    signal tmp_4152[3] <== [tmp_4150[0] + tmp_4151[0], tmp_4150[1] + tmp_4151[1], tmp_4150[2] + tmp_4151[2]];
    signal tmp_4153[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4154[3] <== CMul()(tmp_4153, evals[95]);
    signal tmp_4155[3] <== [tmp_4152[0] + tmp_4154[0], tmp_4152[1] + tmp_4154[1], tmp_4152[2] + tmp_4154[2]];
    signal tmp_4156[3] <== CMul()(evals[75], tmp_4155);
    signal tmp_4157[3] <== CMul()(evals[74], evals[91]);
    signal tmp_4158[3] <== [tmp_4156[0] + tmp_4157[0], tmp_4156[1] + tmp_4157[1], tmp_4156[2] + tmp_4157[2]];
    signal tmp_4159[3] <== [2 * tmp_4158[0], 2 * tmp_4158[1], 2 * tmp_4158[2]];
    signal tmp_4160[3] <== CMul()(tmp_4056, evals[92]);
    signal tmp_4161[3] <== CMul()(tmp_4059, evals[88]);
    signal tmp_4162[3] <== [tmp_4160[0] + tmp_4161[0], tmp_4160[1] + tmp_4161[1], tmp_4160[2] + tmp_4161[2]];
    signal tmp_4163[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4164[3] <== CMul()(tmp_4163, evals[96]);
    signal tmp_4165[3] <== [tmp_4162[0] + tmp_4164[0], tmp_4162[1] + tmp_4164[1], tmp_4162[2] + tmp_4164[2]];
    signal tmp_4166[3] <== CMul()(evals[75], tmp_4165);
    signal tmp_4167[3] <== CMul()(evals[74], evals[92]);
    signal tmp_4168[3] <== [tmp_4166[0] + tmp_4167[0], tmp_4166[1] + tmp_4167[1], tmp_4166[2] + tmp_4167[2]];
    signal tmp_4169[3] <== CMul()(tmp_4056, evals[93]);
    signal tmp_4170[3] <== CMul()(tmp_4059, evals[89]);
    signal tmp_4171[3] <== [tmp_4169[0] + tmp_4170[0], tmp_4169[1] + tmp_4170[1], tmp_4169[2] + tmp_4170[2]];
    signal tmp_4172[3] <== [tmp_4061[0] + tmp_4063[0], tmp_4061[1] + tmp_4063[1], tmp_4061[2] + tmp_4063[2]];
    signal tmp_4173[3] <== CMul()(tmp_4172, evals[97]);
    signal tmp_4174[3] <== [tmp_4171[0] + tmp_4173[0], tmp_4171[1] + tmp_4173[1], tmp_4171[2] + tmp_4173[2]];
    signal tmp_4175[3] <== CMul()(evals[75], tmp_4174);
    signal tmp_4176[3] <== CMul()(evals[74], evals[93]);
    signal tmp_4177[3] <== [tmp_4175[0] + tmp_4176[0], tmp_4175[1] + tmp_4176[1], tmp_4175[2] + tmp_4176[2]];
    signal tmp_4178[3] <== [tmp_4168[0] + tmp_4177[0], tmp_4168[1] + tmp_4177[1], tmp_4168[2] + tmp_4177[2]];
    signal tmp_4179[3] <== [tmp_4159[0] + tmp_4178[0], tmp_4159[1] + tmp_4178[1], tmp_4159[2] + tmp_4178[2]];
    signal tmp_4180[3] <== [tmp_4149[0] + tmp_4179[0], tmp_4149[1] + tmp_4179[1], tmp_4149[2] + tmp_4179[2]];
    signal tmp_4181[3] <== [tmp_4148[0] + tmp_4180[0], tmp_4148[1] + tmp_4180[1], tmp_4148[2] + tmp_4180[2]];
    signal tmp_4182[3] <== [tmp_4118[0] + tmp_4181[0], tmp_4118[1] + tmp_4181[1], tmp_4118[2] + tmp_4181[2]];
    signal tmp_4183[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4184[3] <== CMul()(tmp_4183, evals[97]);
    signal tmp_4185[3] <== CMul()(tmp_4061, evals[89]);
    signal tmp_4186[3] <== [tmp_4184[0] + tmp_4185[0], tmp_4184[1] + tmp_4185[1], tmp_4184[2] + tmp_4185[2]];
    signal tmp_4187[3] <== CMul()(tmp_4063, evals[101]);
    signal tmp_4188[3] <== [tmp_4186[0] + tmp_4187[0], tmp_4186[1] + tmp_4187[1], tmp_4186[2] + tmp_4187[2]];
    signal tmp_4189[3] <== CMul()(evals[75], tmp_4188);
    signal tmp_4190[3] <== CMul()(evals[74], evals[97]);
    signal tmp_4191[3] <== [tmp_4189[0] + tmp_4190[0], tmp_4189[1] + tmp_4190[1], tmp_4189[2] + tmp_4190[2]];
    signal tmp_4192[3] <== [2 * tmp_4191[0], 2 * tmp_4191[1], 2 * tmp_4191[2]];
    signal tmp_4193[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4194[3] <== CMul()(tmp_4193, evals[94]);
    signal tmp_4195[3] <== CMul()(tmp_4061, evals[86]);
    signal tmp_4196[3] <== [tmp_4194[0] + tmp_4195[0], tmp_4194[1] + tmp_4195[1], tmp_4194[2] + tmp_4195[2]];
    signal tmp_4197[3] <== CMul()(tmp_4063, evals[98]);
    signal tmp_4198[3] <== [tmp_4196[0] + tmp_4197[0], tmp_4196[1] + tmp_4197[1], tmp_4196[2] + tmp_4197[2]];
    signal tmp_4199[3] <== CMul()(evals[75], tmp_4198);
    signal tmp_4200[3] <== CMul()(evals[74], evals[94]);
    signal tmp_4201[3] <== [tmp_4199[0] + tmp_4200[0], tmp_4199[1] + tmp_4200[1], tmp_4199[2] + tmp_4200[2]];
    signal tmp_4202[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4203[3] <== CMul()(tmp_4202, evals[95]);
    signal tmp_4204[3] <== CMul()(tmp_4061, evals[87]);
    signal tmp_4205[3] <== [tmp_4203[0] + tmp_4204[0], tmp_4203[1] + tmp_4204[1], tmp_4203[2] + tmp_4204[2]];
    signal tmp_4206[3] <== CMul()(tmp_4063, evals[99]);
    signal tmp_4207[3] <== [tmp_4205[0] + tmp_4206[0], tmp_4205[1] + tmp_4206[1], tmp_4205[2] + tmp_4206[2]];
    signal tmp_4208[3] <== CMul()(evals[75], tmp_4207);
    signal tmp_4209[3] <== CMul()(evals[74], evals[95]);
    signal tmp_4210[3] <== [tmp_4208[0] + tmp_4209[0], tmp_4208[1] + tmp_4209[1], tmp_4208[2] + tmp_4209[2]];
    signal tmp_4211[3] <== [tmp_4201[0] + tmp_4210[0], tmp_4201[1] + tmp_4210[1], tmp_4201[2] + tmp_4210[2]];
    signal tmp_4212[3] <== [tmp_4192[0] + tmp_4211[0], tmp_4192[1] + tmp_4211[1], tmp_4192[2] + tmp_4211[2]];
    signal tmp_4213[3] <== [4 * tmp_4211[0], 4 * tmp_4211[1], 4 * tmp_4211[2]];
    signal tmp_4214[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4215[3] <== CMul()(tmp_4214, evals[95]);
    signal tmp_4216[3] <== CMul()(tmp_4061, evals[87]);
    signal tmp_4217[3] <== [tmp_4215[0] + tmp_4216[0], tmp_4215[1] + tmp_4216[1], tmp_4215[2] + tmp_4216[2]];
    signal tmp_4218[3] <== CMul()(tmp_4063, evals[99]);
    signal tmp_4219[3] <== [tmp_4217[0] + tmp_4218[0], tmp_4217[1] + tmp_4218[1], tmp_4217[2] + tmp_4218[2]];
    signal tmp_4220[3] <== CMul()(evals[75], tmp_4219);
    signal tmp_4221[3] <== CMul()(evals[74], evals[95]);
    signal tmp_4222[3] <== [tmp_4220[0] + tmp_4221[0], tmp_4220[1] + tmp_4221[1], tmp_4220[2] + tmp_4221[2]];
    signal tmp_4223[3] <== [2 * tmp_4222[0], 2 * tmp_4222[1], 2 * tmp_4222[2]];
    signal tmp_4224[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4225[3] <== CMul()(tmp_4224, evals[96]);
    signal tmp_4226[3] <== CMul()(tmp_4061, evals[88]);
    signal tmp_4227[3] <== [tmp_4225[0] + tmp_4226[0], tmp_4225[1] + tmp_4226[1], tmp_4225[2] + tmp_4226[2]];
    signal tmp_4228[3] <== CMul()(tmp_4063, evals[100]);
    signal tmp_4229[3] <== [tmp_4227[0] + tmp_4228[0], tmp_4227[1] + tmp_4228[1], tmp_4227[2] + tmp_4228[2]];
    signal tmp_4230[3] <== CMul()(evals[75], tmp_4229);
    signal tmp_4231[3] <== CMul()(evals[74], evals[96]);
    signal tmp_4232[3] <== [tmp_4230[0] + tmp_4231[0], tmp_4230[1] + tmp_4231[1], tmp_4230[2] + tmp_4231[2]];
    signal tmp_4233[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4234[3] <== CMul()(tmp_4233, evals[97]);
    signal tmp_4235[3] <== CMul()(tmp_4061, evals[89]);
    signal tmp_4236[3] <== [tmp_4234[0] + tmp_4235[0], tmp_4234[1] + tmp_4235[1], tmp_4234[2] + tmp_4235[2]];
    signal tmp_4237[3] <== CMul()(tmp_4063, evals[101]);
    signal tmp_4238[3] <== [tmp_4236[0] + tmp_4237[0], tmp_4236[1] + tmp_4237[1], tmp_4236[2] + tmp_4237[2]];
    signal tmp_4239[3] <== CMul()(evals[75], tmp_4238);
    signal tmp_4240[3] <== CMul()(evals[74], evals[97]);
    signal tmp_4241[3] <== [tmp_4239[0] + tmp_4240[0], tmp_4239[1] + tmp_4240[1], tmp_4239[2] + tmp_4240[2]];
    signal tmp_4242[3] <== [tmp_4232[0] + tmp_4241[0], tmp_4232[1] + tmp_4241[1], tmp_4232[2] + tmp_4241[2]];
    signal tmp_4243[3] <== [tmp_4223[0] + tmp_4242[0], tmp_4223[1] + tmp_4242[1], tmp_4223[2] + tmp_4242[2]];
    signal tmp_4244[3] <== [tmp_4213[0] + tmp_4243[0], tmp_4213[1] + tmp_4243[1], tmp_4213[2] + tmp_4243[2]];
    signal tmp_4245[3] <== [tmp_4212[0] + tmp_4244[0], tmp_4212[1] + tmp_4244[1], tmp_4212[2] + tmp_4244[2]];
    signal tmp_4246[3] <== [tmp_4182[0] + tmp_4245[0], tmp_4182[1] + tmp_4245[1], tmp_4182[2] + tmp_4245[2]];
    signal tmp_4247[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4248[3] <== [tmp_4247[0] + tmp_4061[0], tmp_4247[1] + tmp_4061[1], tmp_4247[2] + tmp_4061[2]];
    signal tmp_4249[3] <== CMul()(tmp_4248, evals[101]);
    signal tmp_4250[3] <== CMul()(tmp_4063, evals[89]);
    signal tmp_4251[3] <== [tmp_4249[0] + tmp_4250[0], tmp_4249[1] + tmp_4250[1], tmp_4249[2] + tmp_4250[2]];
    signal tmp_4252[3] <== CMul()(evals[75], tmp_4251);
    signal tmp_4253[3] <== CMul()(evals[74], evals[101]);
    signal tmp_4254[3] <== [tmp_4252[0] + tmp_4253[0], tmp_4252[1] + tmp_4253[1], tmp_4252[2] + tmp_4253[2]];
    signal tmp_4255[3] <== [2 * tmp_4254[0], 2 * tmp_4254[1], 2 * tmp_4254[2]];
    signal tmp_4256[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4257[3] <== [tmp_4256[0] + tmp_4061[0], tmp_4256[1] + tmp_4061[1], tmp_4256[2] + tmp_4061[2]];
    signal tmp_4258[3] <== CMul()(tmp_4257, evals[98]);
    signal tmp_4259[3] <== CMul()(tmp_4063, evals[86]);
    signal tmp_4260[3] <== [tmp_4258[0] + tmp_4259[0], tmp_4258[1] + tmp_4259[1], tmp_4258[2] + tmp_4259[2]];
    signal tmp_4261[3] <== CMul()(evals[75], tmp_4260);
    signal tmp_4262[3] <== CMul()(evals[74], evals[98]);
    signal tmp_4263[3] <== [tmp_4261[0] + tmp_4262[0], tmp_4261[1] + tmp_4262[1], tmp_4261[2] + tmp_4262[2]];
    signal tmp_4264[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4265[3] <== [tmp_4264[0] + tmp_4061[0], tmp_4264[1] + tmp_4061[1], tmp_4264[2] + tmp_4061[2]];
    signal tmp_4266[3] <== CMul()(tmp_4265, evals[99]);
    signal tmp_4267[3] <== CMul()(tmp_4063, evals[87]);
    signal tmp_4268[3] <== [tmp_4266[0] + tmp_4267[0], tmp_4266[1] + tmp_4267[1], tmp_4266[2] + tmp_4267[2]];
    signal tmp_4269[3] <== CMul()(evals[75], tmp_4268);
    signal tmp_4270[3] <== CMul()(evals[74], evals[99]);
    signal tmp_4271[3] <== [tmp_4269[0] + tmp_4270[0], tmp_4269[1] + tmp_4270[1], tmp_4269[2] + tmp_4270[2]];
    signal tmp_4272[3] <== [tmp_4263[0] + tmp_4271[0], tmp_4263[1] + tmp_4271[1], tmp_4263[2] + tmp_4271[2]];
    signal tmp_4273[3] <== [tmp_4255[0] + tmp_4272[0], tmp_4255[1] + tmp_4272[1], tmp_4255[2] + tmp_4272[2]];
    signal tmp_4274[3] <== [4 * tmp_4272[0], 4 * tmp_4272[1], 4 * tmp_4272[2]];
    signal tmp_4275[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4276[3] <== [tmp_4275[0] + tmp_4061[0], tmp_4275[1] + tmp_4061[1], tmp_4275[2] + tmp_4061[2]];
    signal tmp_4277[3] <== CMul()(tmp_4276, evals[99]);
    signal tmp_4278[3] <== CMul()(tmp_4063, evals[87]);
    signal tmp_4279[3] <== [tmp_4277[0] + tmp_4278[0], tmp_4277[1] + tmp_4278[1], tmp_4277[2] + tmp_4278[2]];
    signal tmp_4280[3] <== CMul()(evals[75], tmp_4279);
    signal tmp_4281[3] <== CMul()(evals[74], evals[99]);
    signal tmp_4282[3] <== [tmp_4280[0] + tmp_4281[0], tmp_4280[1] + tmp_4281[1], tmp_4280[2] + tmp_4281[2]];
    signal tmp_4283[3] <== [2 * tmp_4282[0], 2 * tmp_4282[1], 2 * tmp_4282[2]];
    signal tmp_4284[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4285[3] <== [tmp_4284[0] + tmp_4061[0], tmp_4284[1] + tmp_4061[1], tmp_4284[2] + tmp_4061[2]];
    signal tmp_4286[3] <== CMul()(tmp_4285, evals[100]);
    signal tmp_4287[3] <== CMul()(tmp_4063, evals[88]);
    signal tmp_4288[3] <== [tmp_4286[0] + tmp_4287[0], tmp_4286[1] + tmp_4287[1], tmp_4286[2] + tmp_4287[2]];
    signal tmp_4289[3] <== CMul()(evals[75], tmp_4288);
    signal tmp_4290[3] <== CMul()(evals[74], evals[100]);
    signal tmp_4291[3] <== [tmp_4289[0] + tmp_4290[0], tmp_4289[1] + tmp_4290[1], tmp_4289[2] + tmp_4290[2]];
    signal tmp_4292[3] <== [tmp_4056[0] + tmp_4059[0], tmp_4056[1] + tmp_4059[1], tmp_4056[2] + tmp_4059[2]];
    signal tmp_4293[3] <== [tmp_4292[0] + tmp_4061[0], tmp_4292[1] + tmp_4061[1], tmp_4292[2] + tmp_4061[2]];
    signal tmp_4294[3] <== CMul()(tmp_4293, evals[101]);
    signal tmp_4295[3] <== CMul()(tmp_4063, evals[89]);
    signal tmp_4296[3] <== [tmp_4294[0] + tmp_4295[0], tmp_4294[1] + tmp_4295[1], tmp_4294[2] + tmp_4295[2]];
    signal tmp_4297[3] <== CMul()(evals[75], tmp_4296);
    signal tmp_4298[3] <== CMul()(evals[74], evals[101]);
    signal tmp_4299[3] <== [tmp_4297[0] + tmp_4298[0], tmp_4297[1] + tmp_4298[1], tmp_4297[2] + tmp_4298[2]];
    signal tmp_4300[3] <== [tmp_4291[0] + tmp_4299[0], tmp_4291[1] + tmp_4299[1], tmp_4291[2] + tmp_4299[2]];
    signal tmp_4301[3] <== [tmp_4283[0] + tmp_4300[0], tmp_4283[1] + tmp_4300[1], tmp_4283[2] + tmp_4300[2]];
    signal tmp_4302[3] <== [tmp_4274[0] + tmp_4301[0], tmp_4274[1] + tmp_4301[1], tmp_4274[2] + tmp_4301[2]];
    signal tmp_4303[3] <== [tmp_4273[0] + tmp_4302[0], tmp_4273[1] + tmp_4302[1], tmp_4273[2] + tmp_4302[2]];
    signal tmp_4304[3] <== [tmp_4246[0] + tmp_4303[0], tmp_4246[1] + tmp_4303[1], tmp_4246[2] + tmp_4303[2]];
    signal tmp_4305[3] <== [tmp_4118[0] + tmp_4304[0], tmp_4118[1] + tmp_4304[1], tmp_4118[2] + tmp_4304[2]];
    signal tmp_4306[3] <== [evals[122][0] - tmp_4305[0], evals[122][1] - tmp_4305[1], evals[122][2] - tmp_4305[2]];
    signal tmp_4307[3] <== CMul()(tmp_4053, tmp_4306);
    signal tmp_4308[3] <== [tmp_4052[0] + tmp_4307[0], tmp_4052[1] + tmp_4307[1], tmp_4052[2] + tmp_4307[2]];
    signal tmp_4309[3] <== CMul()(challengeQ, tmp_4308);
    signal tmp_4310[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4311[3] <== [tmp_4117[0] + tmp_4180[0], tmp_4117[1] + tmp_4180[1], tmp_4117[2] + tmp_4180[2]];
    signal tmp_4312[3] <== [tmp_4311[0] + tmp_4244[0], tmp_4311[1] + tmp_4244[1], tmp_4311[2] + tmp_4244[2]];
    signal tmp_4313[3] <== [tmp_4312[0] + tmp_4302[0], tmp_4312[1] + tmp_4302[1], tmp_4312[2] + tmp_4302[2]];
    signal tmp_4314[3] <== [tmp_4117[0] + tmp_4313[0], tmp_4117[1] + tmp_4313[1], tmp_4117[2] + tmp_4313[2]];
    signal tmp_4315[3] <== [evals[123][0] - tmp_4314[0], evals[123][1] - tmp_4314[1], evals[123][2] - tmp_4314[2]];
    signal tmp_4316[3] <== CMul()(tmp_4310, tmp_4315);
    signal tmp_4317[3] <== [tmp_4309[0] + tmp_4316[0], tmp_4309[1] + tmp_4316[1], tmp_4309[2] + tmp_4316[2]];
    signal tmp_4318[3] <== CMul()(challengeQ, tmp_4317);
    signal tmp_4319[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4320[3] <== [4 * tmp_4115[0], 4 * tmp_4115[1], 4 * tmp_4115[2]];
    signal tmp_4321[3] <== [tmp_4320[0] + tmp_4088[0], tmp_4320[1] + tmp_4088[1], tmp_4320[2] + tmp_4088[2]];
    signal tmp_4322[3] <== [tmp_4116[0] + tmp_4321[0], tmp_4116[1] + tmp_4321[1], tmp_4116[2] + tmp_4321[2]];
    signal tmp_4323[3] <== [4 * tmp_4178[0], 4 * tmp_4178[1], 4 * tmp_4178[2]];
    signal tmp_4324[3] <== [tmp_4323[0] + tmp_4148[0], tmp_4323[1] + tmp_4148[1], tmp_4323[2] + tmp_4148[2]];
    signal tmp_4325[3] <== [tmp_4179[0] + tmp_4324[0], tmp_4179[1] + tmp_4324[1], tmp_4179[2] + tmp_4324[2]];
    signal tmp_4326[3] <== [tmp_4322[0] + tmp_4325[0], tmp_4322[1] + tmp_4325[1], tmp_4322[2] + tmp_4325[2]];
    signal tmp_4327[3] <== [4 * tmp_4242[0], 4 * tmp_4242[1], 4 * tmp_4242[2]];
    signal tmp_4328[3] <== [tmp_4327[0] + tmp_4212[0], tmp_4327[1] + tmp_4212[1], tmp_4327[2] + tmp_4212[2]];
    signal tmp_4329[3] <== [tmp_4243[0] + tmp_4328[0], tmp_4243[1] + tmp_4328[1], tmp_4243[2] + tmp_4328[2]];
    signal tmp_4330[3] <== [tmp_4326[0] + tmp_4329[0], tmp_4326[1] + tmp_4329[1], tmp_4326[2] + tmp_4329[2]];
    signal tmp_4331[3] <== [4 * tmp_4300[0], 4 * tmp_4300[1], 4 * tmp_4300[2]];
    signal tmp_4332[3] <== [tmp_4331[0] + tmp_4273[0], tmp_4331[1] + tmp_4273[1], tmp_4331[2] + tmp_4273[2]];
    signal tmp_4333[3] <== [tmp_4301[0] + tmp_4332[0], tmp_4301[1] + tmp_4332[1], tmp_4301[2] + tmp_4332[2]];
    signal tmp_4334[3] <== [tmp_4330[0] + tmp_4333[0], tmp_4330[1] + tmp_4333[1], tmp_4330[2] + tmp_4333[2]];
    signal tmp_4335[3] <== [tmp_4322[0] + tmp_4334[0], tmp_4322[1] + tmp_4334[1], tmp_4322[2] + tmp_4334[2]];
    signal tmp_4336[3] <== [evals[124][0] - tmp_4335[0], evals[124][1] - tmp_4335[1], evals[124][2] - tmp_4335[2]];
    signal tmp_4337[3] <== CMul()(tmp_4319, tmp_4336);
    signal tmp_4338[3] <== [tmp_4318[0] + tmp_4337[0], tmp_4318[1] + tmp_4337[1], tmp_4318[2] + tmp_4337[2]];
    signal tmp_4339[3] <== CMul()(challengeQ, tmp_4338);
    signal tmp_4340[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4341[3] <== [tmp_4321[0] + tmp_4324[0], tmp_4321[1] + tmp_4324[1], tmp_4321[2] + tmp_4324[2]];
    signal tmp_4342[3] <== [tmp_4341[0] + tmp_4328[0], tmp_4341[1] + tmp_4328[1], tmp_4341[2] + tmp_4328[2]];
    signal tmp_4343[3] <== [tmp_4342[0] + tmp_4332[0], tmp_4342[1] + tmp_4332[1], tmp_4342[2] + tmp_4332[2]];
    signal tmp_4344[3] <== [tmp_4321[0] + tmp_4343[0], tmp_4321[1] + tmp_4343[1], tmp_4321[2] + tmp_4343[2]];
    signal tmp_4345[3] <== [evals[125][0] - tmp_4344[0], evals[125][1] - tmp_4344[1], evals[125][2] - tmp_4344[2]];
    signal tmp_4346[3] <== CMul()(tmp_4340, tmp_4345);
    signal tmp_4347[3] <== [tmp_4339[0] + tmp_4346[0], tmp_4339[1] + tmp_4346[1], tmp_4339[2] + tmp_4346[2]];
    signal tmp_4348[3] <== CMul()(challengeQ, tmp_4347);
    signal tmp_4349[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4350[3] <== [tmp_4181[0] + tmp_4304[0], tmp_4181[1] + tmp_4304[1], tmp_4181[2] + tmp_4304[2]];
    signal tmp_4351[3] <== [evals[126][0] - tmp_4350[0], evals[126][1] - tmp_4350[1], evals[126][2] - tmp_4350[2]];
    signal tmp_4352[3] <== CMul()(tmp_4349, tmp_4351);
    signal tmp_4353[3] <== [tmp_4348[0] + tmp_4352[0], tmp_4348[1] + tmp_4352[1], tmp_4348[2] + tmp_4352[2]];
    signal tmp_4354[3] <== CMul()(challengeQ, tmp_4353);
    signal tmp_4355[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4356[3] <== [tmp_4180[0] + tmp_4313[0], tmp_4180[1] + tmp_4313[1], tmp_4180[2] + tmp_4313[2]];
    signal tmp_4357[3] <== [evals[127][0] - tmp_4356[0], evals[127][1] - tmp_4356[1], evals[127][2] - tmp_4356[2]];
    signal tmp_4358[3] <== CMul()(tmp_4355, tmp_4357);
    signal tmp_4359[3] <== [tmp_4354[0] + tmp_4358[0], tmp_4354[1] + tmp_4358[1], tmp_4354[2] + tmp_4358[2]];
    signal tmp_4360[3] <== CMul()(challengeQ, tmp_4359);
    signal tmp_4361[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4362[3] <== [tmp_4325[0] + tmp_4334[0], tmp_4325[1] + tmp_4334[1], tmp_4325[2] + tmp_4334[2]];
    signal tmp_4363[3] <== [evals[128][0] - tmp_4362[0], evals[128][1] - tmp_4362[1], evals[128][2] - tmp_4362[2]];
    signal tmp_4364[3] <== CMul()(tmp_4361, tmp_4363);
    signal tmp_4365[3] <== [tmp_4360[0] + tmp_4364[0], tmp_4360[1] + tmp_4364[1], tmp_4360[2] + tmp_4364[2]];
    signal tmp_4366[3] <== CMul()(challengeQ, tmp_4365);
    signal tmp_4367[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4368[3] <== [tmp_4324[0] + tmp_4343[0], tmp_4324[1] + tmp_4343[1], tmp_4324[2] + tmp_4343[2]];
    signal tmp_4369[3] <== [evals[129][0] - tmp_4368[0], evals[129][1] - tmp_4368[1], evals[129][2] - tmp_4368[2]];
    signal tmp_4370[3] <== CMul()(tmp_4367, tmp_4369);
    signal tmp_4371[3] <== [tmp_4366[0] + tmp_4370[0], tmp_4366[1] + tmp_4370[1], tmp_4366[2] + tmp_4370[2]];
    signal tmp_4372[3] <== CMul()(challengeQ, tmp_4371);
    signal tmp_4373[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4374[3] <== [tmp_4245[0] + tmp_4304[0], tmp_4245[1] + tmp_4304[1], tmp_4245[2] + tmp_4304[2]];
    signal tmp_4375[3] <== [evals[130][0] - tmp_4374[0], evals[130][1] - tmp_4374[1], evals[130][2] - tmp_4374[2]];
    signal tmp_4376[3] <== CMul()(tmp_4373, tmp_4375);
    signal tmp_4377[3] <== [tmp_4372[0] + tmp_4376[0], tmp_4372[1] + tmp_4376[1], tmp_4372[2] + tmp_4376[2]];
    signal tmp_4378[3] <== CMul()(challengeQ, tmp_4377);
    signal tmp_4379[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4380[3] <== [tmp_4244[0] + tmp_4313[0], tmp_4244[1] + tmp_4313[1], tmp_4244[2] + tmp_4313[2]];
    signal tmp_4381[3] <== [evals[131][0] - tmp_4380[0], evals[131][1] - tmp_4380[1], evals[131][2] - tmp_4380[2]];
    signal tmp_4382[3] <== CMul()(tmp_4379, tmp_4381);
    signal tmp_4383[3] <== [tmp_4378[0] + tmp_4382[0], tmp_4378[1] + tmp_4382[1], tmp_4378[2] + tmp_4382[2]];
    signal tmp_4384[3] <== CMul()(challengeQ, tmp_4383);
    signal tmp_4385[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4386[3] <== [tmp_4329[0] + tmp_4334[0], tmp_4329[1] + tmp_4334[1], tmp_4329[2] + tmp_4334[2]];
    signal tmp_4387[3] <== [evals[132][0] - tmp_4386[0], evals[132][1] - tmp_4386[1], evals[132][2] - tmp_4386[2]];
    signal tmp_4388[3] <== CMul()(tmp_4385, tmp_4387);
    signal tmp_4389[3] <== [tmp_4384[0] + tmp_4388[0], tmp_4384[1] + tmp_4388[1], tmp_4384[2] + tmp_4388[2]];
    signal tmp_4390[3] <== CMul()(challengeQ, tmp_4389);
    signal tmp_4391[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4392[3] <== [tmp_4328[0] + tmp_4343[0], tmp_4328[1] + tmp_4343[1], tmp_4328[2] + tmp_4343[2]];
    signal tmp_4393[3] <== [evals[133][0] - tmp_4392[0], evals[133][1] - tmp_4392[1], evals[133][2] - tmp_4392[2]];
    signal tmp_4394[3] <== CMul()(tmp_4391, tmp_4393);
    signal tmp_4395[3] <== [tmp_4390[0] + tmp_4394[0], tmp_4390[1] + tmp_4394[1], tmp_4390[2] + tmp_4394[2]];
    signal tmp_4396[3] <== CMul()(challengeQ, tmp_4395);
    signal tmp_4397[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4398[3] <== [tmp_4303[0] + tmp_4304[0], tmp_4303[1] + tmp_4304[1], tmp_4303[2] + tmp_4304[2]];
    signal tmp_4399[3] <== [evals[134][0] - tmp_4398[0], evals[134][1] - tmp_4398[1], evals[134][2] - tmp_4398[2]];
    signal tmp_4400[3] <== CMul()(tmp_4397, tmp_4399);
    signal tmp_4401[3] <== [tmp_4396[0] + tmp_4400[0], tmp_4396[1] + tmp_4400[1], tmp_4396[2] + tmp_4400[2]];
    signal tmp_4402[3] <== CMul()(challengeQ, tmp_4401);
    signal tmp_4403[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4404[3] <== [tmp_4302[0] + tmp_4313[0], tmp_4302[1] + tmp_4313[1], tmp_4302[2] + tmp_4313[2]];
    signal tmp_4405[3] <== [evals[135][0] - tmp_4404[0], evals[135][1] - tmp_4404[1], evals[135][2] - tmp_4404[2]];
    signal tmp_4406[3] <== CMul()(tmp_4403, tmp_4405);
    signal tmp_4407[3] <== [tmp_4402[0] + tmp_4406[0], tmp_4402[1] + tmp_4406[1], tmp_4402[2] + tmp_4406[2]];
    signal tmp_4408[3] <== CMul()(challengeQ, tmp_4407);
    signal tmp_4409[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4410[3] <== [tmp_4333[0] + tmp_4334[0], tmp_4333[1] + tmp_4334[1], tmp_4333[2] + tmp_4334[2]];
    signal tmp_4411[3] <== [evals[136][0] - tmp_4410[0], evals[136][1] - tmp_4410[1], evals[136][2] - tmp_4410[2]];
    signal tmp_4412[3] <== CMul()(tmp_4409, tmp_4411);
    signal tmp_4413[3] <== [tmp_4408[0] + tmp_4412[0], tmp_4408[1] + tmp_4412[1], tmp_4408[2] + tmp_4412[2]];
    signal tmp_4414[3] <== CMul()(challengeQ, tmp_4413);
    signal tmp_4415[3] <== [evals[74][0] + evals[75][0], evals[74][1] + evals[75][1], evals[74][2] + evals[75][2]];
    signal tmp_4416[3] <== [tmp_4332[0] + tmp_4343[0], tmp_4332[1] + tmp_4343[1], tmp_4332[2] + tmp_4343[2]];
    signal tmp_4417[3] <== [evals[137][0] - tmp_4416[0], evals[137][1] - tmp_4416[1], evals[137][2] - tmp_4416[2]];
    signal tmp_4418[3] <== CMul()(tmp_4415, tmp_4417);
    signal tmp_4419[3] <== [tmp_4414[0] + tmp_4418[0], tmp_4414[1] + tmp_4418[1], tmp_4414[2] + tmp_4418[2]];
    signal tmp_4420[3] <== CMul()(challengeQ, tmp_4419);
    signal tmp_4421[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4422[3] <== [tmp_4421[0] + tmp_3866[0], tmp_4421[1] + tmp_3866[1], tmp_4421[2] + tmp_3866[2]];
    signal tmp_4423[3] <== [tmp_4422[0] + evals[196][0], tmp_4422[1] + evals[196][1], tmp_4422[2] + evals[196][2]];
    signal tmp_4424[3] <== [tmp_4423[0] + evals[2][0], tmp_4423[1] + evals[2][1], tmp_4423[2] + evals[2][2]];
    signal tmp_4425[3] <== [tmp_4424[0] + evals[197][0], tmp_4424[1] + evals[197][1], tmp_4424[2] + evals[197][2]];
    signal tmp_4426[3] <== [tmp_4425[0] + evals[178][0], tmp_4425[1] + evals[178][1], tmp_4425[2] + evals[178][2]];
    signal tmp_4427[3] <== [tmp_4426[0] + evals[77][0], tmp_4426[1] + evals[77][1], tmp_4426[2] + evals[77][2]];
    signal tmp_4428[3] <== CMul()(evals[77], evals[86]);
    signal tmp_4429[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4430[3] <== CMul()(tmp_4429, evals[180]);
    signal tmp_4431[3] <== [tmp_4428[0] + tmp_4430[0], tmp_4428[1] + tmp_4430[1], tmp_4428[2] + tmp_4430[2]];
    signal tmp_4432[3] <== [tmp_3978[0] * 5625716564419252202, tmp_3978[1] * 5625716564419252202, tmp_3978[2] * 5625716564419252202];
    signal tmp_4433[3] <== [tmp_3864[0] * 12037145371429260559, tmp_3864[1] * 12037145371429260559, tmp_3864[2] * 12037145371429260559];
    signal tmp_4434[3] <== [tmp_4432[0] + tmp_4433[0], tmp_4432[1] + tmp_4433[1], tmp_4432[2] + tmp_4433[2]];
    signal tmp_4435[3] <== [tmp_3866[0] * 11810209965918630788, tmp_3866[1] * 11810209965918630788, tmp_3866[2] * 11810209965918630788];
    signal tmp_4436[3] <== [tmp_4434[0] + tmp_4435[0], tmp_4434[1] + tmp_4435[1], tmp_4434[2] + tmp_4435[2]];
    signal tmp_4437[3] <== [evals[196][0] * 12343504462597575239, evals[196][1] * 12343504462597575239, evals[196][2] * 12343504462597575239];
    signal tmp_4438[3] <== [tmp_4436[0] + tmp_4437[0], tmp_4436[1] + tmp_4437[1], tmp_4436[2] + tmp_4437[2]];
    signal tmp_4439[3] <== [evals[76][0] * 12399309956582731760, evals[76][1] * 12399309956582731760, evals[76][2] * 12399309956582731760];
    signal tmp_4440[3] <== [tmp_4438[0] + tmp_4439[0], tmp_4438[1] + tmp_4439[1], tmp_4438[2] + tmp_4439[2]];
    signal tmp_4441[3] <== [evals[2][0] * 2197120128337974909, evals[2][1] * 2197120128337974909, evals[2][2] * 2197120128337974909];
    signal tmp_4442[3] <== [tmp_4440[0] + tmp_4441[0], tmp_4440[1] + tmp_4441[1], tmp_4440[2] + tmp_4441[2]];
    signal tmp_4443[3] <== [evals[197][0] * 6108317238076822824, evals[197][1] * 6108317238076822824, evals[197][2] * 6108317238076822824];
    signal tmp_4444[3] <== [tmp_4442[0] + tmp_4443[0], tmp_4442[1] + tmp_4443[1], tmp_4442[2] + tmp_4443[2]];
    signal tmp_4445[3] <== [evals[178][0] * 16343835012250148340, evals[178][1] * 16343835012250148340, evals[178][2] * 16343835012250148340];
    signal tmp_4446[3] <== [tmp_4444[0] + tmp_4445[0], tmp_4444[1] + tmp_4445[1], tmp_4444[2] + tmp_4445[2]];
    signal tmp_4447[3] <== [evals[77][0] * 9076340683484238287, evals[77][1] * 9076340683484238287, evals[77][2] * 9076340683484238287];
    tmp_4448 <== [tmp_4446[0] + tmp_4447[0], tmp_4446[1] + tmp_4447[1], tmp_4446[2] + tmp_4447[2]];
    signal tmp_4449[3] <== [evals[125][0] + tmp_4448[0], evals[125][1] + tmp_4448[1], evals[125][2] + tmp_4448[2]];
    signal tmp_4450[3] <== [evals[125][0] + tmp_4448[0], evals[125][1] + tmp_4448[1], evals[125][2] + tmp_4448[2]];
    tmp_4451 <== CMul()(tmp_4449, tmp_4450);
    tmp_4452 <== CMul()(evals[147], tmp_4451);
    signal tmp_4453[3] <== [evals[125][0] + tmp_4448[0], evals[125][1] + tmp_4448[1], evals[125][2] + tmp_4448[2]];
    signal tmp_4454[3] <== CMul()(tmp_4452, tmp_4453);
    signal tmp_4455[3] <== [2 * tmp_4454[0], 2 * tmp_4454[1], 2 * tmp_4454[2]];
    signal tmp_4456[3] <== [tmp_3978[0] * 1579613653969377219, tmp_3978[1] * 1579613653969377219, tmp_3978[2] * 1579613653969377219];
    signal tmp_4457[3] <== [tmp_3864[0] * 11699769881923825909, tmp_3864[1] * 11699769881923825909, tmp_3864[2] * 11699769881923825909];
    signal tmp_4458[3] <== [tmp_4456[0] + tmp_4457[0], tmp_4456[1] + tmp_4457[1], tmp_4456[2] + tmp_4457[2]];
    signal tmp_4459[3] <== [tmp_3866[0] * 12437259172286625042, tmp_3866[1] * 12437259172286625042, tmp_3866[2] * 12437259172286625042];
    signal tmp_4460[3] <== [tmp_4458[0] + tmp_4459[0], tmp_4458[1] + tmp_4459[1], tmp_4458[2] + tmp_4459[2]];
    signal tmp_4461[3] <== [evals[196][0] * 12355842859741995537, evals[196][1] * 12355842859741995537, evals[196][2] * 12355842859741995537];
    signal tmp_4462[3] <== [tmp_4460[0] + tmp_4461[0], tmp_4460[1] + tmp_4461[1], tmp_4460[2] + tmp_4461[2]];
    signal tmp_4463[3] <== [evals[76][0] * 2949845317987848448, evals[76][1] * 2949845317987848448, evals[76][2] * 2949845317987848448];
    signal tmp_4464[3] <== [tmp_4462[0] + tmp_4463[0], tmp_4462[1] + tmp_4463[1], tmp_4462[2] + tmp_4463[2]];
    signal tmp_4465[3] <== [evals[2][0] * 15766898731627159621, evals[2][1] * 15766898731627159621, evals[2][2] * 15766898731627159621];
    signal tmp_4466[3] <== [tmp_4464[0] + tmp_4465[0], tmp_4464[1] + tmp_4465[1], tmp_4464[2] + tmp_4465[2]];
    signal tmp_4467[3] <== [evals[197][0] * 4158286797471309733, evals[197][1] * 4158286797471309733, evals[197][2] * 4158286797471309733];
    signal tmp_4468[3] <== [tmp_4466[0] + tmp_4467[0], tmp_4466[1] + tmp_4467[1], tmp_4466[2] + tmp_4467[2]];
    signal tmp_4469[3] <== [evals[178][0] * 9738621014989498753, evals[178][1] * 9738621014989498753, evals[178][2] * 9738621014989498753];
    signal tmp_4470[3] <== [tmp_4468[0] + tmp_4469[0], tmp_4468[1] + tmp_4469[1], tmp_4468[2] + tmp_4469[2]];
    signal tmp_4471[3] <== [evals[77][0] * 8041781683128153420, evals[77][1] * 8041781683128153420, evals[77][2] * 8041781683128153420];
    tmp_4472 <== [tmp_4470[0] + tmp_4471[0], tmp_4470[1] + tmp_4471[1], tmp_4470[2] + tmp_4471[2]];
    signal tmp_4473[3] <== [evals[122][0] + tmp_4472[0], evals[122][1] + tmp_4472[1], evals[122][2] + tmp_4472[2]];
    signal tmp_4474[3] <== [evals[122][0] + tmp_4472[0], evals[122][1] + tmp_4472[1], evals[122][2] + tmp_4472[2]];
    tmp_4475 <== CMul()(tmp_4473, tmp_4474);
    tmp_4476 <== CMul()(evals[148], tmp_4475);
    signal tmp_4477[3] <== [evals[122][0] + tmp_4472[0], evals[122][1] + tmp_4472[1], evals[122][2] + tmp_4472[2]];
    signal tmp_4478[3] <== CMul()(tmp_4476, tmp_4477);
    signal tmp_4479[3] <== [tmp_3978[0] * 15509510893087893340, tmp_3978[1] * 15509510893087893340, tmp_3978[2] * 15509510893087893340];
    signal tmp_4480[3] <== [tmp_3864[0] * 4349260652212695338, tmp_3864[1] * 4349260652212695338, tmp_3864[2] * 4349260652212695338];
    signal tmp_4481[3] <== [tmp_4479[0] + tmp_4480[0], tmp_4479[1] + tmp_4480[1], tmp_4479[2] + tmp_4480[2]];
    signal tmp_4482[3] <== [tmp_3866[0] * 3143835545170851304, tmp_3866[1] * 3143835545170851304, tmp_3866[2] * 3143835545170851304];
    signal tmp_4483[3] <== [tmp_4481[0] + tmp_4482[0], tmp_4481[1] + tmp_4482[1], tmp_4481[2] + tmp_4482[2]];
    signal tmp_4484[3] <== [evals[196][0] * 14160921043125059199, evals[196][1] * 14160921043125059199, evals[196][2] * 14160921043125059199];
    signal tmp_4485[3] <== [tmp_4483[0] + tmp_4484[0], tmp_4483[1] + tmp_4484[1], tmp_4483[2] + tmp_4484[2]];
    signal tmp_4486[3] <== [evals[76][0] * 6935557324168256069, evals[76][1] * 6935557324168256069, evals[76][2] * 6935557324168256069];
    signal tmp_4487[3] <== [tmp_4485[0] + tmp_4486[0], tmp_4485[1] + tmp_4486[1], tmp_4485[2] + tmp_4486[2]];
    signal tmp_4488[3] <== [evals[2][0] * 6714370142978148338, evals[2][1] * 6714370142978148338, evals[2][2] * 6714370142978148338];
    signal tmp_4489[3] <== [tmp_4487[0] + tmp_4488[0], tmp_4487[1] + tmp_4488[1], tmp_4487[2] + tmp_4488[2]];
    signal tmp_4490[3] <== [evals[197][0] * 1589482450994236426, evals[197][1] * 1589482450994236426, evals[197][2] * 1589482450994236426];
    signal tmp_4491[3] <== [tmp_4489[0] + tmp_4490[0], tmp_4489[1] + tmp_4490[1], tmp_4489[2] + tmp_4490[2]];
    signal tmp_4492[3] <== [evals[178][0] * 237412858668409460, evals[178][1] * 237412858668409460, evals[178][2] * 237412858668409460];
    signal tmp_4493[3] <== [tmp_4491[0] + tmp_4492[0], tmp_4491[1] + tmp_4492[1], tmp_4491[2] + tmp_4492[2]];
    signal tmp_4494[3] <== [evals[77][0] * 13172746155287932121, evals[77][1] * 13172746155287932121, evals[77][2] * 13172746155287932121];
    tmp_4495 <== [tmp_4493[0] + tmp_4494[0], tmp_4493[1] + tmp_4494[1], tmp_4493[2] + tmp_4494[2]];
    signal tmp_4496[3] <== [evals[123][0] + tmp_4495[0], evals[123][1] + tmp_4495[1], evals[123][2] + tmp_4495[2]];
    signal tmp_4497[3] <== [evals[123][0] + tmp_4495[0], evals[123][1] + tmp_4495[1], evals[123][2] + tmp_4495[2]];
    tmp_4498 <== CMul()(tmp_4496, tmp_4497);
    tmp_4499 <== CMul()(evals[149], tmp_4498);
    signal tmp_4500[3] <== [evals[123][0] + tmp_4495[0], evals[123][1] + tmp_4495[1], evals[123][2] + tmp_4495[2]];
    signal tmp_4501[3] <== CMul()(tmp_4499, tmp_4500);
    signal tmp_4502[3] <== [tmp_4478[0] + tmp_4501[0], tmp_4478[1] + tmp_4501[1], tmp_4478[2] + tmp_4501[2]];
    tmp_4503 <== [tmp_4455[0] + tmp_4502[0], tmp_4455[1] + tmp_4502[1], tmp_4455[2] + tmp_4502[2]];
    signal tmp_4504[3] <== [4 * tmp_4502[0], 4 * tmp_4502[1], 4 * tmp_4502[2]];
    signal tmp_4505[3] <== [evals[123][0] + tmp_4495[0], evals[123][1] + tmp_4495[1], evals[123][2] + tmp_4495[2]];
    signal tmp_4506[3] <== CMul()(tmp_4499, tmp_4505);
    signal tmp_4507[3] <== [2 * tmp_4506[0], 2 * tmp_4506[1], 2 * tmp_4506[2]];
    signal tmp_4508[3] <== [tmp_3978[0] * 10090715174060125222, tmp_3978[1] * 10090715174060125222, tmp_3978[2] * 10090715174060125222];
    signal tmp_4509[3] <== [tmp_3864[0] * 8228348357294662898, tmp_3864[1] * 8228348357294662898, tmp_3864[2] * 8228348357294662898];
    signal tmp_4510[3] <== [tmp_4508[0] + tmp_4509[0], tmp_4508[1] + tmp_4509[1], tmp_4508[2] + tmp_4509[2]];
    signal tmp_4511[3] <== [tmp_3866[0] * 14361575850007152549, tmp_3866[1] * 14361575850007152549, tmp_3866[2] * 14361575850007152549];
    signal tmp_4512[3] <== [tmp_4510[0] + tmp_4511[0], tmp_4510[1] + tmp_4511[1], tmp_4510[2] + tmp_4511[2]];
    signal tmp_4513[3] <== [evals[196][0] * 13682913322779819091, evals[196][1] * 13682913322779819091, evals[196][2] * 13682913322779819091];
    signal tmp_4514[3] <== [tmp_4512[0] + tmp_4513[0], tmp_4512[1] + tmp_4513[1], tmp_4512[2] + tmp_4513[2]];
    signal tmp_4515[3] <== [evals[76][0] * 2055390472619727748, evals[76][1] * 2055390472619727748, evals[76][2] * 2055390472619727748];
    signal tmp_4516[3] <== [tmp_4514[0] + tmp_4515[0], tmp_4514[1] + tmp_4515[1], tmp_4514[2] + tmp_4515[2]];
    signal tmp_4517[3] <== [evals[2][0] * 5906664861467821209, evals[2][1] * 5906664861467821209, evals[2][2] * 5906664861467821209];
    signal tmp_4518[3] <== [tmp_4516[0] + tmp_4517[0], tmp_4516[1] + tmp_4517[1], tmp_4516[2] + tmp_4517[2]];
    signal tmp_4519[3] <== [evals[197][0] * 4836112672767737407, evals[197][1] * 4836112672767737407, evals[197][2] * 4836112672767737407];
    signal tmp_4520[3] <== [tmp_4518[0] + tmp_4519[0], tmp_4518[1] + tmp_4519[1], tmp_4518[2] + tmp_4519[2]];
    signal tmp_4521[3] <== [evals[178][0] * 6405904454922718574, evals[178][1] * 6405904454922718574, evals[178][2] * 6405904454922718574];
    signal tmp_4522[3] <== [tmp_4520[0] + tmp_4521[0], tmp_4520[1] + tmp_4521[1], tmp_4520[2] + tmp_4521[2]];
    signal tmp_4523[3] <== [evals[77][0] * 6611413049825339003, evals[77][1] * 6611413049825339003, evals[77][2] * 6611413049825339003];
    tmp_4524 <== [tmp_4522[0] + tmp_4523[0], tmp_4522[1] + tmp_4523[1], tmp_4522[2] + tmp_4523[2]];
    signal tmp_4525[3] <== [evals[124][0] + tmp_4524[0], evals[124][1] + tmp_4524[1], evals[124][2] + tmp_4524[2]];
    signal tmp_4526[3] <== [evals[124][0] + tmp_4524[0], evals[124][1] + tmp_4524[1], evals[124][2] + tmp_4524[2]];
    tmp_4527 <== CMul()(tmp_4525, tmp_4526);
    tmp_4528 <== CMul()(evals[150], tmp_4527);
    signal tmp_4529[3] <== [evals[124][0] + tmp_4524[0], evals[124][1] + tmp_4524[1], evals[124][2] + tmp_4524[2]];
    signal tmp_4530[3] <== CMul()(tmp_4528, tmp_4529);
    signal tmp_4531[3] <== [evals[125][0] + tmp_4448[0], evals[125][1] + tmp_4448[1], evals[125][2] + tmp_4448[2]];
    signal tmp_4532[3] <== CMul()(tmp_4452, tmp_4531);
    tmp_4533 <== [tmp_4530[0] + tmp_4532[0], tmp_4530[1] + tmp_4532[1], tmp_4530[2] + tmp_4532[2]];
    tmp_4534 <== [tmp_4507[0] + tmp_4533[0], tmp_4507[1] + tmp_4533[1], tmp_4507[2] + tmp_4533[2]];
    tmp_4535 <== [tmp_4504[0] + tmp_4534[0], tmp_4504[1] + tmp_4534[1], tmp_4504[2] + tmp_4534[2]];
    signal tmp_4536[3] <== [tmp_4503[0] + tmp_4535[0], tmp_4503[1] + tmp_4535[1], tmp_4503[2] + tmp_4535[2]];
    signal tmp_4537[3] <== [tmp_3978[0] * 2027625550790675754, tmp_3978[1] * 2027625550790675754, tmp_3978[2] * 2027625550790675754];
    signal tmp_4538[3] <== [tmp_3864[0] * 10157815501749214163, tmp_3864[1] * 10157815501749214163, tmp_3864[2] * 10157815501749214163];
    signal tmp_4539[3] <== [tmp_4537[0] + tmp_4538[0], tmp_4537[1] + tmp_4538[1], tmp_4537[2] + tmp_4538[2]];
    signal tmp_4540[3] <== [tmp_3866[0] * 1778669202480014256, tmp_3866[1] * 1778669202480014256, tmp_3866[2] * 1778669202480014256];
    signal tmp_4541[3] <== [tmp_4539[0] + tmp_4540[0], tmp_4539[1] + tmp_4540[1], tmp_4539[2] + tmp_4540[2]];
    signal tmp_4542[3] <== [evals[196][0] * 690520822232718983, evals[196][1] * 690520822232718983, evals[196][2] * 690520822232718983];
    signal tmp_4543[3] <== [tmp_4541[0] + tmp_4542[0], tmp_4541[1] + tmp_4542[1], tmp_4541[2] + tmp_4542[2]];
    signal tmp_4544[3] <== [evals[76][0] * 14821962908959377092, evals[76][1] * 14821962908959377092, evals[76][2] * 14821962908959377092];
    signal tmp_4545[3] <== [tmp_4543[0] + tmp_4544[0], tmp_4543[1] + tmp_4544[1], tmp_4543[2] + tmp_4544[2]];
    signal tmp_4546[3] <== [evals[2][0] * 12399217337488717232, evals[2][1] * 12399217337488717232, evals[2][2] * 12399217337488717232];
    signal tmp_4547[3] <== [tmp_4545[0] + tmp_4546[0], tmp_4545[1] + tmp_4546[1], tmp_4545[2] + tmp_4546[2]];
    signal tmp_4548[3] <== [evals[197][0] * 6643835433125335951, evals[197][1] * 6643835433125335951, evals[197][2] * 6643835433125335951];
    signal tmp_4549[3] <== [tmp_4547[0] + tmp_4548[0], tmp_4547[1] + tmp_4548[1], tmp_4547[2] + tmp_4548[2]];
    signal tmp_4550[3] <== [evals[178][0] * 16030622638137836774, evals[178][1] * 16030622638137836774, evals[178][2] * 16030622638137836774];
    signal tmp_4551[3] <== [tmp_4549[0] + tmp_4550[0], tmp_4549[1] + tmp_4550[1], tmp_4549[2] + tmp_4550[2]];
    signal tmp_4552[3] <== [evals[77][0] * 16216654076620539925, evals[77][1] * 16216654076620539925, evals[77][2] * 16216654076620539925];
    tmp_4553 <== [tmp_4551[0] + tmp_4552[0], tmp_4551[1] + tmp_4552[1], tmp_4551[2] + tmp_4552[2]];
    signal tmp_4554[3] <== [evals[129][0] + tmp_4553[0], evals[129][1] + tmp_4553[1], evals[129][2] + tmp_4553[2]];
    signal tmp_4555[3] <== [evals[129][0] + tmp_4553[0], evals[129][1] + tmp_4553[1], evals[129][2] + tmp_4553[2]];
    tmp_4556 <== CMul()(tmp_4554, tmp_4555);
    tmp_4557 <== CMul()(evals[151], tmp_4556);
    signal tmp_4558[3] <== [evals[129][0] + tmp_4553[0], evals[129][1] + tmp_4553[1], evals[129][2] + tmp_4553[2]];
    signal tmp_4559[3] <== CMul()(tmp_4557, tmp_4558);
    signal tmp_4560[3] <== [2 * tmp_4559[0], 2 * tmp_4559[1], 2 * tmp_4559[2]];
    signal tmp_4561[3] <== [tmp_3978[0] * 3006005019077469174, tmp_3978[1] * 3006005019077469174, tmp_3978[2] * 3006005019077469174];
    signal tmp_4562[3] <== [tmp_3864[0] * 7743383997471469184, tmp_3864[1] * 7743383997471469184, tmp_3864[2] * 7743383997471469184];
    signal tmp_4563[3] <== [tmp_4561[0] + tmp_4562[0], tmp_4561[1] + tmp_4562[1], tmp_4561[2] + tmp_4562[2]];
    signal tmp_4564[3] <== [tmp_3866[0] * 17667700303515849362, tmp_3866[1] * 17667700303515849362, tmp_3866[2] * 17667700303515849362];
    signal tmp_4565[3] <== [tmp_4563[0] + tmp_4564[0], tmp_4563[1] + tmp_4564[1], tmp_4563[2] + tmp_4564[2]];
    signal tmp_4566[3] <== [evals[196][0] * 14249643564311509087, evals[196][1] * 14249643564311509087, evals[196][2] * 14249643564311509087];
    signal tmp_4567[3] <== [tmp_4565[0] + tmp_4566[0], tmp_4565[1] + tmp_4566[1], tmp_4565[2] + tmp_4566[2]];
    signal tmp_4568[3] <== [evals[76][0] * 3088058410657255303, evals[76][1] * 3088058410657255303, evals[76][2] * 3088058410657255303];
    signal tmp_4569[3] <== [tmp_4567[0] + tmp_4568[0], tmp_4567[1] + tmp_4568[1], tmp_4567[2] + tmp_4568[2]];
    signal tmp_4570[3] <== [evals[2][0] * 9400421283898153864, evals[2][1] * 9400421283898153864, evals[2][2] * 9400421283898153864];
    signal tmp_4571[3] <== [tmp_4569[0] + tmp_4570[0], tmp_4569[1] + tmp_4570[1], tmp_4569[2] + tmp_4570[2]];
    signal tmp_4572[3] <== [evals[197][0] * 11174425919404088598, evals[197][1] * 11174425919404088598, evals[197][2] * 11174425919404088598];
    signal tmp_4573[3] <== [tmp_4571[0] + tmp_4572[0], tmp_4571[1] + tmp_4572[1], tmp_4571[2] + tmp_4572[2]];
    signal tmp_4574[3] <== [evals[178][0] * 9342245617122491936, evals[178][1] * 9342245617122491936, evals[178][2] * 9342245617122491936];
    signal tmp_4575[3] <== [tmp_4573[0] + tmp_4574[0], tmp_4573[1] + tmp_4574[1], tmp_4573[2] + tmp_4574[2]];
    signal tmp_4576[3] <== [evals[77][0] * 12295614633782496176, evals[77][1] * 12295614633782496176, evals[77][2] * 12295614633782496176];
    tmp_4577 <== [tmp_4575[0] + tmp_4576[0], tmp_4575[1] + tmp_4576[1], tmp_4575[2] + tmp_4576[2]];
    signal tmp_4578[3] <== [evals[126][0] + tmp_4577[0], evals[126][1] + tmp_4577[1], evals[126][2] + tmp_4577[2]];
    signal tmp_4579[3] <== [evals[126][0] + tmp_4577[0], evals[126][1] + tmp_4577[1], evals[126][2] + tmp_4577[2]];
    tmp_4580 <== CMul()(tmp_4578, tmp_4579);
    tmp_4581 <== CMul()(evals[152], tmp_4580);
    signal tmp_4582[3] <== [evals[126][0] + tmp_4577[0], evals[126][1] + tmp_4577[1], evals[126][2] + tmp_4577[2]];
    signal tmp_4583[3] <== CMul()(tmp_4581, tmp_4582);
    signal tmp_4584[3] <== [tmp_3978[0] * 18314693207014427912, tmp_3978[1] * 18314693207014427912, tmp_3978[2] * 18314693207014427912];
    signal tmp_4585[3] <== [tmp_3864[0] * 7842006993218978420, tmp_3864[1] * 7842006993218978420, tmp_3864[2] * 7842006993218978420];
    signal tmp_4586[3] <== [tmp_4584[0] + tmp_4585[0], tmp_4584[1] + tmp_4585[1], tmp_4584[2] + tmp_4585[2]];
    signal tmp_4587[3] <== [tmp_3866[0] * 13451932870729425780, tmp_3866[1] * 13451932870729425780, tmp_3866[2] * 13451932870729425780];
    signal tmp_4588[3] <== [tmp_4586[0] + tmp_4587[0], tmp_4586[1] + tmp_4587[1], tmp_4586[2] + tmp_4587[2]];
    signal tmp_4589[3] <== [evals[196][0] * 15849758561085927402, evals[196][1] * 15849758561085927402, evals[196][2] * 15849758561085927402];
    signal tmp_4590[3] <== [tmp_4588[0] + tmp_4589[0], tmp_4588[1] + tmp_4589[1], tmp_4588[2] + tmp_4589[2]];
    signal tmp_4591[3] <== [evals[76][0] * 8790959064194804235, evals[76][1] * 8790959064194804235, evals[76][2] * 8790959064194804235];
    signal tmp_4592[3] <== [tmp_4590[0] + tmp_4591[0], tmp_4590[1] + tmp_4591[1], tmp_4590[2] + tmp_4591[2]];
    signal tmp_4593[3] <== [evals[2][0] * 16740473708727660723, evals[2][1] * 16740473708727660723, evals[2][2] * 16740473708727660723];
    signal tmp_4594[3] <== [tmp_4592[0] + tmp_4593[0], tmp_4592[1] + tmp_4593[1], tmp_4592[2] + tmp_4593[2]];
    signal tmp_4595[3] <== [evals[197][0] * 2755155124184744585, evals[197][1] * 2755155124184744585, evals[197][2] * 2755155124184744585];
    signal tmp_4596[3] <== [tmp_4594[0] + tmp_4595[0], tmp_4594[1] + tmp_4595[1], tmp_4594[2] + tmp_4595[2]];
    signal tmp_4597[3] <== [evals[178][0] * 824922755340087111, evals[178][1] * 824922755340087111, evals[178][2] * 824922755340087111];
    signal tmp_4598[3] <== [tmp_4596[0] + tmp_4597[0], tmp_4596[1] + tmp_4597[1], tmp_4596[2] + tmp_4597[2]];
    signal tmp_4599[3] <== [evals[77][0] * 11414162089917699064, evals[77][1] * 11414162089917699064, evals[77][2] * 11414162089917699064];
    tmp_4600 <== [tmp_4598[0] + tmp_4599[0], tmp_4598[1] + tmp_4599[1], tmp_4598[2] + tmp_4599[2]];
    signal tmp_4601[3] <== [evals[127][0] + tmp_4600[0], evals[127][1] + tmp_4600[1], evals[127][2] + tmp_4600[2]];
    signal tmp_4602[3] <== [evals[127][0] + tmp_4600[0], evals[127][1] + tmp_4600[1], evals[127][2] + tmp_4600[2]];
    tmp_4603 <== CMul()(tmp_4601, tmp_4602);
    tmp_4604 <== CMul()(evals[153], tmp_4603);
    signal tmp_4605[3] <== [evals[127][0] + tmp_4600[0], evals[127][1] + tmp_4600[1], evals[127][2] + tmp_4600[2]];
    signal tmp_4606[3] <== CMul()(tmp_4604, tmp_4605);
    signal tmp_4607[3] <== [tmp_4583[0] + tmp_4606[0], tmp_4583[1] + tmp_4606[1], tmp_4583[2] + tmp_4606[2]];
    tmp_4608 <== [tmp_4560[0] + tmp_4607[0], tmp_4560[1] + tmp_4607[1], tmp_4560[2] + tmp_4607[2]];
    signal tmp_4609[3] <== [4 * tmp_4607[0], 4 * tmp_4607[1], 4 * tmp_4607[2]];
    signal tmp_4610[3] <== [evals[127][0] + tmp_4600[0], evals[127][1] + tmp_4600[1], evals[127][2] + tmp_4600[2]];
    signal tmp_4611[3] <== CMul()(tmp_4604, tmp_4610);
    signal tmp_4612[3] <== [2 * tmp_4611[0], 2 * tmp_4611[1], 2 * tmp_4611[2]];
    signal tmp_4613[3] <== [tmp_3978[0] * 10170571510627764565, tmp_3978[1] * 10170571510627764565, tmp_3978[2] * 10170571510627764565];
    signal tmp_4614[3] <== [tmp_3864[0] * 18369471830829996810, tmp_3864[1] * 18369471830829996810, tmp_3864[2] * 18369471830829996810];
    signal tmp_4615[3] <== [tmp_4613[0] + tmp_4614[0], tmp_4613[1] + tmp_4614[1], tmp_4613[2] + tmp_4614[2]];
    signal tmp_4616[3] <== [tmp_3866[0] * 17496644301504593329, tmp_3866[1] * 17496644301504593329, tmp_3866[2] * 17496644301504593329];
    signal tmp_4617[3] <== [tmp_4615[0] + tmp_4616[0], tmp_4615[1] + tmp_4616[1], tmp_4615[2] + tmp_4616[2]];
    signal tmp_4618[3] <== [evals[196][0] * 12186564686511703262, evals[196][1] * 12186564686511703262, evals[196][2] * 12186564686511703262];
    signal tmp_4619[3] <== [tmp_4617[0] + tmp_4618[0], tmp_4617[1] + tmp_4618[1], tmp_4617[2] + tmp_4618[2]];
    signal tmp_4620[3] <== [evals[76][0] * 4582664136559626757, evals[76][1] * 4582664136559626757, evals[76][2] * 4582664136559626757];
    signal tmp_4621[3] <== [tmp_4619[0] + tmp_4620[0], tmp_4619[1] + tmp_4620[1], tmp_4619[2] + tmp_4620[2]];
    signal tmp_4622[3] <== [evals[2][0] * 7287335615015370599, evals[2][1] * 7287335615015370599, evals[2][2] * 7287335615015370599];
    signal tmp_4623[3] <== [tmp_4621[0] + tmp_4622[0], tmp_4621[1] + tmp_4622[1], tmp_4621[2] + tmp_4622[2]];
    signal tmp_4624[3] <== [evals[197][0] * 13202523778257173956, evals[197][1] * 13202523778257173956, evals[197][2] * 13202523778257173956];
    signal tmp_4625[3] <== [tmp_4623[0] + tmp_4624[0], tmp_4623[1] + tmp_4624[1], tmp_4623[2] + tmp_4624[2]];
    signal tmp_4626[3] <== [evals[178][0] * 5398764608665671277, evals[178][1] * 5398764608665671277, evals[178][2] * 5398764608665671277];
    signal tmp_4627[3] <== [tmp_4625[0] + tmp_4626[0], tmp_4625[1] + tmp_4626[1], tmp_4625[2] + tmp_4626[2]];
    signal tmp_4628[3] <== [evals[77][0] * 13658053783093222498, evals[77][1] * 13658053783093222498, evals[77][2] * 13658053783093222498];
    tmp_4629 <== [tmp_4627[0] + tmp_4628[0], tmp_4627[1] + tmp_4628[1], tmp_4627[2] + tmp_4628[2]];
    signal tmp_4630[3] <== [evals[128][0] + tmp_4629[0], evals[128][1] + tmp_4629[1], evals[128][2] + tmp_4629[2]];
    signal tmp_4631[3] <== [evals[128][0] + tmp_4629[0], evals[128][1] + tmp_4629[1], evals[128][2] + tmp_4629[2]];
    tmp_4632 <== CMul()(tmp_4630, tmp_4631);
    tmp_4633 <== CMul()(evals[154], tmp_4632);
    signal tmp_4634[3] <== [evals[128][0] + tmp_4629[0], evals[128][1] + tmp_4629[1], evals[128][2] + tmp_4629[2]];
    signal tmp_4635[3] <== CMul()(tmp_4633, tmp_4634);
    signal tmp_4636[3] <== [evals[129][0] + tmp_4553[0], evals[129][1] + tmp_4553[1], evals[129][2] + tmp_4553[2]];
    signal tmp_4637[3] <== CMul()(tmp_4557, tmp_4636);
    tmp_4638 <== [tmp_4635[0] + tmp_4637[0], tmp_4635[1] + tmp_4637[1], tmp_4635[2] + tmp_4637[2]];
    tmp_4639 <== [tmp_4612[0] + tmp_4638[0], tmp_4612[1] + tmp_4638[1], tmp_4612[2] + tmp_4638[2]];
    tmp_4640 <== [tmp_4609[0] + tmp_4639[0], tmp_4609[1] + tmp_4639[1], tmp_4609[2] + tmp_4639[2]];
    tmp_4641 <== [tmp_4608[0] + tmp_4640[0], tmp_4608[1] + tmp_4640[1], tmp_4608[2] + tmp_4640[2]];
    signal tmp_4642[3] <== [tmp_4536[0] + tmp_4641[0], tmp_4536[1] + tmp_4641[1], tmp_4536[2] + tmp_4641[2]];
    signal tmp_4643[3] <== [tmp_3978[0] * 3151070045406026687, tmp_3978[1] * 3151070045406026687, tmp_3978[2] * 3151070045406026687];
    signal tmp_4644[3] <== [tmp_3864[0] * 6178857985076893256, tmp_3864[1] * 6178857985076893256, tmp_3864[2] * 6178857985076893256];
    signal tmp_4645[3] <== [tmp_4643[0] + tmp_4644[0], tmp_4643[1] + tmp_4644[1], tmp_4643[2] + tmp_4644[2]];
    signal tmp_4646[3] <== [tmp_3866[0] * 15182093233826554989, tmp_3866[1] * 15182093233826554989, tmp_3866[2] * 15182093233826554989];
    signal tmp_4647[3] <== [tmp_4645[0] + tmp_4646[0], tmp_4645[1] + tmp_4646[1], tmp_4645[2] + tmp_4646[2]];
    signal tmp_4648[3] <== [evals[196][0] * 12083506578988585239, evals[196][1] * 12083506578988585239, evals[196][2] * 12083506578988585239];
    signal tmp_4649[3] <== [tmp_4647[0] + tmp_4648[0], tmp_4647[1] + tmp_4648[1], tmp_4647[2] + tmp_4648[2]];
    signal tmp_4650[3] <== [evals[76][0] * 11973738501993105496, evals[76][1] * 11973738501993105496, evals[76][2] * 11973738501993105496];
    signal tmp_4651[3] <== [tmp_4649[0] + tmp_4650[0], tmp_4649[1] + tmp_4650[1], tmp_4649[2] + tmp_4650[2]];
    signal tmp_4652[3] <== [evals[2][0] * 3483661079725019706, evals[2][1] * 3483661079725019706, evals[2][2] * 3483661079725019706];
    signal tmp_4653[3] <== [tmp_4651[0] + tmp_4652[0], tmp_4651[1] + tmp_4652[1], tmp_4651[2] + tmp_4652[2]];
    signal tmp_4654[3] <== [evals[197][0] * 257624298448087449, evals[197][1] * 257624298448087449, evals[197][2] * 257624298448087449];
    signal tmp_4655[3] <== [tmp_4653[0] + tmp_4654[0], tmp_4653[1] + tmp_4654[1], tmp_4653[2] + tmp_4654[2]];
    signal tmp_4656[3] <== [evals[178][0] * 587812931475326787, evals[178][1] * 587812931475326787, evals[178][2] * 587812931475326787];
    signal tmp_4657[3] <== [tmp_4655[0] + tmp_4656[0], tmp_4655[1] + tmp_4656[1], tmp_4655[2] + tmp_4656[2]];
    signal tmp_4658[3] <== [evals[77][0] * 13499421621575350806, evals[77][1] * 13499421621575350806, evals[77][2] * 13499421621575350806];
    tmp_4659 <== [tmp_4657[0] + tmp_4658[0], tmp_4657[1] + tmp_4658[1], tmp_4657[2] + tmp_4658[2]];
    signal tmp_4660[3] <== [evals[133][0] + tmp_4659[0], evals[133][1] + tmp_4659[1], evals[133][2] + tmp_4659[2]];
    signal tmp_4661[3] <== [evals[133][0] + tmp_4659[0], evals[133][1] + tmp_4659[1], evals[133][2] + tmp_4659[2]];
    tmp_4662 <== CMul()(tmp_4660, tmp_4661);
    tmp_4663 <== CMul()(evals[155], tmp_4662);
    signal tmp_4664[3] <== [evals[133][0] + tmp_4659[0], evals[133][1] + tmp_4659[1], evals[133][2] + tmp_4659[2]];
    signal tmp_4665[3] <== CMul()(tmp_4663, tmp_4664);
    signal tmp_4666[3] <== [2 * tmp_4665[0], 2 * tmp_4665[1], 2 * tmp_4665[2]];
    signal tmp_4667[3] <== [tmp_3978[0] * 3983470257916202094, tmp_3978[1] * 3983470257916202094, tmp_3978[2] * 3983470257916202094];
    signal tmp_4668[3] <== [tmp_3864[0] * 11128005290408113568, tmp_3864[1] * 11128005290408113568, tmp_3864[2] * 11128005290408113568];
    signal tmp_4669[3] <== [tmp_4667[0] + tmp_4668[0], tmp_4667[1] + tmp_4668[1], tmp_4667[2] + tmp_4668[2]];
    signal tmp_4670[3] <== [tmp_3866[0] * 11124488067381312952, tmp_3866[1] * 11124488067381312952, tmp_3866[2] * 11124488067381312952];
    signal tmp_4671[3] <== [tmp_4669[0] + tmp_4670[0], tmp_4669[1] + tmp_4670[1], tmp_4669[2] + tmp_4670[2]];
    signal tmp_4672[3] <== [evals[196][0] * 684200944081655198, evals[196][1] * 684200944081655198, evals[196][2] * 684200944081655198];
    signal tmp_4673[3] <== [tmp_4671[0] + tmp_4672[0], tmp_4671[1] + tmp_4672[1], tmp_4671[2] + tmp_4672[2]];
    signal tmp_4674[3] <== [evals[76][0] * 2002685046256118388, evals[76][1] * 2002685046256118388, evals[76][2] * 2002685046256118388];
    signal tmp_4675[3] <== [tmp_4673[0] + tmp_4674[0], tmp_4673[1] + tmp_4674[1], tmp_4673[2] + tmp_4674[2]];
    signal tmp_4676[3] <== [evals[2][0] * 12655834778591227242, evals[2][1] * 12655834778591227242, evals[2][2] * 12655834778591227242];
    signal tmp_4677[3] <== [tmp_4675[0] + tmp_4676[0], tmp_4675[1] + tmp_4676[1], tmp_4675[2] + tmp_4676[2]];
    signal tmp_4678[3] <== [evals[197][0] * 2748965361441888477, evals[197][1] * 2748965361441888477, evals[197][2] * 2748965361441888477];
    signal tmp_4679[3] <== [tmp_4677[0] + tmp_4678[0], tmp_4677[1] + tmp_4678[1], tmp_4677[2] + tmp_4678[2]];
    signal tmp_4680[3] <== [evals[178][0] * 4921899302518525668, evals[178][1] * 4921899302518525668, evals[178][2] * 4921899302518525668];
    signal tmp_4681[3] <== [tmp_4679[0] + tmp_4680[0], tmp_4679[1] + tmp_4680[1], tmp_4679[2] + tmp_4680[2]];
    signal tmp_4682[3] <== [evals[77][0] * 17782598995085730495, evals[77][1] * 17782598995085730495, evals[77][2] * 17782598995085730495];
    tmp_4683 <== [tmp_4681[0] + tmp_4682[0], tmp_4681[1] + tmp_4682[1], tmp_4681[2] + tmp_4682[2]];
    signal tmp_4684[3] <== [evals[130][0] + tmp_4683[0], evals[130][1] + tmp_4683[1], evals[130][2] + tmp_4683[2]];
    signal tmp_4685[3] <== [evals[130][0] + tmp_4683[0], evals[130][1] + tmp_4683[1], evals[130][2] + tmp_4683[2]];
    tmp_4686 <== CMul()(tmp_4684, tmp_4685);
    tmp_4687 <== CMul()(evals[156], tmp_4686);
    signal tmp_4688[3] <== [evals[130][0] + tmp_4683[0], evals[130][1] + tmp_4683[1], evals[130][2] + tmp_4683[2]];
    signal tmp_4689[3] <== CMul()(tmp_4687, tmp_4688);
    signal tmp_4690[3] <== [tmp_3978[0] * 3423470109396435354, tmp_3978[1] * 3423470109396435354, tmp_3978[2] * 3423470109396435354];
    signal tmp_4691[3] <== [tmp_3864[0] * 12272917402811871099, tmp_3864[1] * 12272917402811871099, tmp_3864[2] * 12272917402811871099];
    signal tmp_4692[3] <== [tmp_4690[0] + tmp_4691[0], tmp_4690[1] + tmp_4691[1], tmp_4690[2] + tmp_4691[2]];
    signal tmp_4693[3] <== [tmp_3866[0] * 14273583775934080755, tmp_3866[1] * 14273583775934080755, tmp_3866[2] * 14273583775934080755];
    signal tmp_4694[3] <== [tmp_4692[0] + tmp_4693[0], tmp_4692[1] + tmp_4693[1], tmp_4692[2] + tmp_4693[2]];
    signal tmp_4695[3] <== [evals[196][0] * 6008958449718781068, evals[196][1] * 6008958449718781068, evals[196][2] * 6008958449718781068];
    signal tmp_4696[3] <== [tmp_4694[0] + tmp_4695[0], tmp_4694[1] + tmp_4695[1], tmp_4694[2] + tmp_4695[2]];
    signal tmp_4697[3] <== [evals[76][0] * 15689593639089632368, evals[76][1] * 15689593639089632368, evals[76][2] * 15689593639089632368];
    signal tmp_4698[3] <== [tmp_4696[0] + tmp_4697[0], tmp_4696[1] + tmp_4697[1], tmp_4696[2] + tmp_4697[2]];
    signal tmp_4699[3] <== [evals[2][0] * 7516917516498792203, evals[2][1] * 7516917516498792203, evals[2][2] * 7516917516498792203];
    signal tmp_4700[3] <== [tmp_4698[0] + tmp_4699[0], tmp_4698[1] + tmp_4699[1], tmp_4698[2] + tmp_4699[2]];
    signal tmp_4701[3] <== [evals[197][0] * 14297205492001640506, evals[197][1] * 14297205492001640506, evals[197][2] * 14297205492001640506];
    signal tmp_4702[3] <== [tmp_4700[0] + tmp_4701[0], tmp_4700[1] + tmp_4701[1], tmp_4700[2] + tmp_4701[2]];
    signal tmp_4703[3] <== [evals[178][0] * 10676667276121395674, evals[178][1] * 10676667276121395674, evals[178][2] * 10676667276121395674];
    signal tmp_4704[3] <== [tmp_4702[0] + tmp_4703[0], tmp_4702[1] + tmp_4703[1], tmp_4702[2] + tmp_4703[2]];
    signal tmp_4705[3] <== [evals[77][0] * 18362903988365127450, evals[77][1] * 18362903988365127450, evals[77][2] * 18362903988365127450];
    tmp_4706 <== [tmp_4704[0] + tmp_4705[0], tmp_4704[1] + tmp_4705[1], tmp_4704[2] + tmp_4705[2]];
    signal tmp_4707[3] <== [evals[131][0] + tmp_4706[0], evals[131][1] + tmp_4706[1], evals[131][2] + tmp_4706[2]];
    signal tmp_4708[3] <== [evals[131][0] + tmp_4706[0], evals[131][1] + tmp_4706[1], evals[131][2] + tmp_4706[2]];
    tmp_4709 <== CMul()(tmp_4707, tmp_4708);
    tmp_4710 <== CMul()(evals[157], tmp_4709);
    signal tmp_4711[3] <== [evals[131][0] + tmp_4706[0], evals[131][1] + tmp_4706[1], evals[131][2] + tmp_4706[2]];
    signal tmp_4712[3] <== CMul()(tmp_4710, tmp_4711);
    signal tmp_4713[3] <== [tmp_4689[0] + tmp_4712[0], tmp_4689[1] + tmp_4712[1], tmp_4689[2] + tmp_4712[2]];
    tmp_4714 <== [tmp_4666[0] + tmp_4713[0], tmp_4666[1] + tmp_4713[1], tmp_4666[2] + tmp_4713[2]];
    signal tmp_4715[3] <== [4 * tmp_4713[0], 4 * tmp_4713[1], 4 * tmp_4713[2]];
    signal tmp_4716[3] <== [evals[131][0] + tmp_4706[0], evals[131][1] + tmp_4706[1], evals[131][2] + tmp_4706[2]];
    signal tmp_4717[3] <== CMul()(tmp_4710, tmp_4716);
    signal tmp_4718[3] <== [2 * tmp_4717[0], 2 * tmp_4717[1], 2 * tmp_4717[2]];
    signal tmp_4719[3] <== [tmp_3978[0] * 3450488264035752368, tmp_3978[1] * 3450488264035752368, tmp_3978[2] * 3450488264035752368];
    signal tmp_4720[3] <== [tmp_3864[0] * 106820976709770637, tmp_3864[1] * 106820976709770637, tmp_3864[2] * 106820976709770637];
    signal tmp_4721[3] <== [tmp_4719[0] + tmp_4720[0], tmp_4719[1] + tmp_4720[1], tmp_4719[2] + tmp_4720[2]];
    signal tmp_4722[3] <== [tmp_3866[0] * 13257379154210125715, tmp_3866[1] * 13257379154210125715, tmp_3866[2] * 13257379154210125715];
    signal tmp_4723[3] <== [tmp_4721[0] + tmp_4722[0], tmp_4721[1] + tmp_4722[1], tmp_4721[2] + tmp_4722[2]];
    signal tmp_4724[3] <== [evals[196][0] * 3098079696536837500, evals[196][1] * 3098079696536837500, evals[196][2] * 3098079696536837500];
    signal tmp_4725[3] <== [tmp_4723[0] + tmp_4724[0], tmp_4723[1] + tmp_4724[1], tmp_4723[2] + tmp_4724[2]];
    signal tmp_4726[3] <== [evals[76][0] * 10512046881800463697, evals[76][1] * 10512046881800463697, evals[76][2] * 10512046881800463697];
    signal tmp_4727[3] <== [tmp_4725[0] + tmp_4726[0], tmp_4725[1] + tmp_4726[1], tmp_4725[2] + tmp_4726[2]];
    signal tmp_4728[3] <== [evals[2][0] * 17291898461659961460, evals[2][1] * 17291898461659961460, evals[2][2] * 17291898461659961460];
    signal tmp_4729[3] <== [tmp_4727[0] + tmp_4728[0], tmp_4727[1] + tmp_4728[1], tmp_4727[2] + tmp_4728[2]];
    signal tmp_4730[3] <== [evals[197][0] * 12595977886586773319, evals[197][1] * 12595977886586773319, evals[197][2] * 12595977886586773319];
    signal tmp_4731[3] <== [tmp_4729[0] + tmp_4730[0], tmp_4729[1] + tmp_4730[1], tmp_4729[2] + tmp_4730[2]];
    signal tmp_4732[3] <== [evals[178][0] * 16304445044992359788, evals[178][1] * 16304445044992359788, evals[178][2] * 16304445044992359788];
    signal tmp_4733[3] <== [tmp_4731[0] + tmp_4732[0], tmp_4731[1] + tmp_4732[1], tmp_4731[2] + tmp_4732[2]];
    signal tmp_4734[3] <== [evals[77][0] * 11596821426909532371, evals[77][1] * 11596821426909532371, evals[77][2] * 11596821426909532371];
    tmp_4735 <== [tmp_4733[0] + tmp_4734[0], tmp_4733[1] + tmp_4734[1], tmp_4733[2] + tmp_4734[2]];
    signal tmp_4736[3] <== [evals[132][0] + tmp_4735[0], evals[132][1] + tmp_4735[1], evals[132][2] + tmp_4735[2]];
    signal tmp_4737[3] <== [evals[132][0] + tmp_4735[0], evals[132][1] + tmp_4735[1], evals[132][2] + tmp_4735[2]];
    tmp_4738 <== CMul()(tmp_4736, tmp_4737);
    tmp_4739 <== CMul()(evals[158], tmp_4738);
    signal tmp_4740[3] <== [evals[132][0] + tmp_4735[0], evals[132][1] + tmp_4735[1], evals[132][2] + tmp_4735[2]];
    signal tmp_4741[3] <== CMul()(tmp_4739, tmp_4740);
    signal tmp_4742[3] <== [evals[133][0] + tmp_4659[0], evals[133][1] + tmp_4659[1], evals[133][2] + tmp_4659[2]];
    signal tmp_4743[3] <== CMul()(tmp_4663, tmp_4742);
    tmp_4744 <== [tmp_4741[0] + tmp_4743[0], tmp_4741[1] + tmp_4743[1], tmp_4741[2] + tmp_4743[2]];
    tmp_4745 <== [tmp_4718[0] + tmp_4744[0], tmp_4718[1] + tmp_4744[1], tmp_4718[2] + tmp_4744[2]];
    tmp_4746 <== [tmp_4715[0] + tmp_4745[0], tmp_4715[1] + tmp_4745[1], tmp_4715[2] + tmp_4745[2]];
    tmp_4747 <== [tmp_4714[0] + tmp_4746[0], tmp_4714[1] + tmp_4746[1], tmp_4714[2] + tmp_4746[2]];
    signal tmp_4748[3] <== [tmp_4642[0] + tmp_4747[0], tmp_4642[1] + tmp_4747[1], tmp_4642[2] + tmp_4747[2]];
    signal tmp_4749[3] <== [tmp_3978[0] * 7780139165529418388, tmp_3978[1] * 7780139165529418388, tmp_3978[2] * 7780139165529418388];
    signal tmp_4750[3] <== [tmp_3864[0] * 10060334973073197601, tmp_3864[1] * 10060334973073197601, tmp_3864[2] * 10060334973073197601];
    signal tmp_4751[3] <== [tmp_4749[0] + tmp_4750[0], tmp_4749[1] + tmp_4750[1], tmp_4749[2] + tmp_4750[2]];
    signal tmp_4752[3] <== [tmp_3866[0] * 3717595218375990594, tmp_3866[1] * 3717595218375990594, tmp_3866[2] * 3717595218375990594];
    signal tmp_4753[3] <== [tmp_4751[0] + tmp_4752[0], tmp_4751[1] + tmp_4752[1], tmp_4751[2] + tmp_4752[2]];
    signal tmp_4754[3] <== [evals[196][0] * 17095617549195818058, evals[196][1] * 17095617549195818058, evals[196][2] * 17095617549195818058];
    signal tmp_4755[3] <== [tmp_4753[0] + tmp_4754[0], tmp_4753[1] + tmp_4754[1], tmp_4753[2] + tmp_4754[2]];
    signal tmp_4756[3] <== [evals[76][0] * 17528314872631229800, evals[76][1] * 17528314872631229800, evals[76][2] * 17528314872631229800];
    signal tmp_4757[3] <== [tmp_4755[0] + tmp_4756[0], tmp_4755[1] + tmp_4756[1], tmp_4755[2] + tmp_4756[2]];
    signal tmp_4758[3] <== [evals[2][0] * 17138720803294251067, evals[2][1] * 17138720803294251067, evals[2][2] * 17138720803294251067];
    signal tmp_4759[3] <== [tmp_4757[0] + tmp_4758[0], tmp_4757[1] + tmp_4758[1], tmp_4757[2] + tmp_4758[2]];
    signal tmp_4760[3] <== [evals[197][0] * 7256201493271212720, evals[197][1] * 7256201493271212720, evals[197][2] * 7256201493271212720];
    signal tmp_4761[3] <== [tmp_4759[0] + tmp_4760[0], tmp_4759[1] + tmp_4760[1], tmp_4759[2] + tmp_4760[2]];
    signal tmp_4762[3] <== [evals[178][0] * 15665352327749112015, evals[178][1] * 15665352327749112015, evals[178][2] * 15665352327749112015];
    signal tmp_4763[3] <== [tmp_4761[0] + tmp_4762[0], tmp_4761[1] + tmp_4762[1], tmp_4761[2] + tmp_4762[2]];
    signal tmp_4764[3] <== [evals[77][0] * 4615393526180652407, evals[77][1] * 4615393526180652407, evals[77][2] * 4615393526180652407];
    tmp_4765 <== [tmp_4763[0] + tmp_4764[0], tmp_4763[1] + tmp_4764[1], tmp_4763[2] + tmp_4764[2]];
    signal tmp_4766[3] <== [evals[137][0] + tmp_4765[0], evals[137][1] + tmp_4765[1], evals[137][2] + tmp_4765[2]];
    signal tmp_4767[3] <== [evals[137][0] + tmp_4765[0], evals[137][1] + tmp_4765[1], evals[137][2] + tmp_4765[2]];
    tmp_4768 <== CMul()(tmp_4766, tmp_4767);
    tmp_4769 <== CMul()(evals[159], tmp_4768);
    signal tmp_4770[3] <== [evals[137][0] + tmp_4765[0], evals[137][1] + tmp_4765[1], evals[137][2] + tmp_4765[2]];
    signal tmp_4771[3] <== CMul()(tmp_4769, tmp_4770);
    signal tmp_4772[3] <== [2 * tmp_4771[0], 2 * tmp_4771[1], 2 * tmp_4771[2]];
    signal tmp_4773[3] <== [tmp_3978[0] * 13462781804006123550, tmp_3978[1] * 13462781804006123550, tmp_3978[2] * 13462781804006123550];
    signal tmp_4774[3] <== [tmp_3864[0] * 14036787039817180193, tmp_3864[1] * 14036787039817180193, tmp_3864[2] * 14036787039817180193];
    signal tmp_4775[3] <== [tmp_4773[0] + tmp_4774[0], tmp_4773[1] + tmp_4774[1], tmp_4773[2] + tmp_4774[2]];
    signal tmp_4776[3] <== [tmp_3866[0] * 5343815959848316783, tmp_3866[1] * 5343815959848316783, tmp_3866[2] * 5343815959848316783];
    signal tmp_4777[3] <== [tmp_4775[0] + tmp_4776[0], tmp_4775[1] + tmp_4776[1], tmp_4775[2] + tmp_4776[2]];
    signal tmp_4778[3] <== [evals[196][0] * 6489163969976034646, evals[196][1] * 6489163969976034646, evals[196][2] * 6489163969976034646];
    signal tmp_4779[3] <== [tmp_4777[0] + tmp_4778[0], tmp_4777[1] + tmp_4778[1], tmp_4777[2] + tmp_4778[2]];
    signal tmp_4780[3] <== [evals[76][0] * 13009070935614847359, evals[76][1] * 13009070935614847359, evals[76][2] * 13009070935614847359];
    signal tmp_4781[3] <== [tmp_4779[0] + tmp_4780[0], tmp_4779[1] + tmp_4780[1], tmp_4779[2] + tmp_4780[2]];
    signal tmp_4782[3] <== [evals[2][0] * 18038566855851505564, evals[2][1] * 18038566855851505564, evals[2][2] * 18038566855851505564];
    signal tmp_4783[3] <== [tmp_4781[0] + tmp_4782[0], tmp_4781[1] + tmp_4782[1], tmp_4781[2] + tmp_4782[2]];
    signal tmp_4784[3] <== [evals[197][0] * 2744326799038764281, evals[197][1] * 2744326799038764281, evals[197][2] * 2744326799038764281];
    signal tmp_4785[3] <== [tmp_4783[0] + tmp_4784[0], tmp_4783[1] + tmp_4784[1], tmp_4783[2] + tmp_4784[2]];
    signal tmp_4786[3] <== [evals[178][0] * 7627897473303434798, evals[178][1] * 7627897473303434798, evals[178][2] * 7627897473303434798];
    signal tmp_4787[3] <== [tmp_4785[0] + tmp_4786[0], tmp_4785[1] + tmp_4786[1], tmp_4785[2] + tmp_4786[2]];
    signal tmp_4788[3] <== [evals[77][0] * 15256312875610823807, evals[77][1] * 15256312875610823807, evals[77][2] * 15256312875610823807];
    tmp_4789 <== [tmp_4787[0] + tmp_4788[0], tmp_4787[1] + tmp_4788[1], tmp_4787[2] + tmp_4788[2]];
    signal tmp_4790[3] <== [evals[134][0] + tmp_4789[0], evals[134][1] + tmp_4789[1], evals[134][2] + tmp_4789[2]];
    signal tmp_4791[3] <== [evals[134][0] + tmp_4789[0], evals[134][1] + tmp_4789[1], evals[134][2] + tmp_4789[2]];
    tmp_4792 <== CMul()(tmp_4790, tmp_4791);
    tmp_4793 <== CMul()(evals[160], tmp_4792);
    signal tmp_4794[3] <== [evals[134][0] + tmp_4789[0], evals[134][1] + tmp_4789[1], evals[134][2] + tmp_4789[2]];
    signal tmp_4795[3] <== CMul()(tmp_4793, tmp_4794);
    signal tmp_4796[3] <== [tmp_3978[0] * 13288575772684627216, tmp_3978[1] * 13288575772684627216, tmp_3978[2] * 13288575772684627216];
    signal tmp_4797[3] <== [tmp_3864[0] * 1659458828486927362, tmp_3864[1] * 1659458828486927362, tmp_3864[2] * 1659458828486927362];
    signal tmp_4798[3] <== [tmp_4796[0] + tmp_4797[0], tmp_4796[1] + tmp_4797[1], tmp_4796[2] + tmp_4797[2]];
    signal tmp_4799[3] <== [tmp_3866[0] * 5696193414571279556, tmp_3866[1] * 5696193414571279556, tmp_3866[2] * 5696193414571279556];
    signal tmp_4800[3] <== [tmp_4798[0] + tmp_4799[0], tmp_4798[1] + tmp_4799[1], tmp_4798[2] + tmp_4799[2]];
    signal tmp_4801[3] <== [evals[196][0] * 1186411428667285248, evals[196][1] * 1186411428667285248, evals[196][2] * 1186411428667285248];
    signal tmp_4802[3] <== [tmp_4800[0] + tmp_4801[0], tmp_4800[1] + tmp_4801[1], tmp_4800[2] + tmp_4801[2]];
    signal tmp_4803[3] <== [evals[76][0] * 12908033183314664096, evals[76][1] * 12908033183314664096, evals[76][2] * 12908033183314664096];
    signal tmp_4804[3] <== [tmp_4802[0] + tmp_4803[0], tmp_4802[1] + tmp_4803[1], tmp_4802[2] + tmp_4803[2]];
    signal tmp_4805[3] <== [evals[2][0] * 6166154478921479976, evals[2][1] * 6166154478921479976, evals[2][2] * 6166154478921479976];
    signal tmp_4806[3] <== [tmp_4804[0] + tmp_4805[0], tmp_4804[1] + tmp_4805[1], tmp_4804[2] + tmp_4805[2]];
    signal tmp_4807[3] <== [evals[197][0] * 9097808034015176417, evals[197][1] * 9097808034015176417, evals[197][2] * 9097808034015176417];
    signal tmp_4808[3] <== [tmp_4806[0] + tmp_4807[0], tmp_4806[1] + tmp_4807[1], tmp_4806[2] + tmp_4807[2]];
    signal tmp_4809[3] <== [evals[178][0] * 2633542348821978290, evals[178][1] * 2633542348821978290, evals[178][2] * 2633542348821978290];
    signal tmp_4810[3] <== [tmp_4808[0] + tmp_4809[0], tmp_4808[1] + tmp_4809[1], tmp_4808[2] + tmp_4809[2]];
    signal tmp_4811[3] <== [evals[77][0] * 1558929959921731273, evals[77][1] * 1558929959921731273, evals[77][2] * 1558929959921731273];
    tmp_4812 <== [tmp_4810[0] + tmp_4811[0], tmp_4810[1] + tmp_4811[1], tmp_4810[2] + tmp_4811[2]];
    signal tmp_4813[3] <== [evals[135][0] + tmp_4812[0], evals[135][1] + tmp_4812[1], evals[135][2] + tmp_4812[2]];
    signal tmp_4814[3] <== [evals[135][0] + tmp_4812[0], evals[135][1] + tmp_4812[1], evals[135][2] + tmp_4812[2]];
    tmp_4815 <== CMul()(tmp_4813, tmp_4814);
    tmp_4816 <== CMul()(evals[161], tmp_4815);
    signal tmp_4817[3] <== [evals[135][0] + tmp_4812[0], evals[135][1] + tmp_4812[1], evals[135][2] + tmp_4812[2]];
    signal tmp_4818[3] <== CMul()(tmp_4816, tmp_4817);
    signal tmp_4819[3] <== [tmp_4795[0] + tmp_4818[0], tmp_4795[1] + tmp_4818[1], tmp_4795[2] + tmp_4818[2]];
    tmp_4820 <== [tmp_4772[0] + tmp_4819[0], tmp_4772[1] + tmp_4819[1], tmp_4772[2] + tmp_4819[2]];
    signal tmp_4821[3] <== [4 * tmp_4819[0], 4 * tmp_4819[1], 4 * tmp_4819[2]];
    signal tmp_4822[3] <== [evals[135][0] + tmp_4812[0], evals[135][1] + tmp_4812[1], evals[135][2] + tmp_4812[2]];
    signal tmp_4823[3] <== CMul()(tmp_4816, tmp_4822);
    signal tmp_4824[3] <== [2 * tmp_4823[0], 2 * tmp_4823[1], 2 * tmp_4823[2]];
    signal tmp_4825[3] <== [tmp_3978[0] * 13745549378090937523, tmp_3978[1] * 13745549378090937523, tmp_3978[2] * 13745549378090937523];
    signal tmp_4826[3] <== [tmp_3864[0] * 15938183887757779615, tmp_3864[1] * 15938183887757779615, tmp_3864[2] * 15938183887757779615];
    signal tmp_4827[3] <== [tmp_4825[0] + tmp_4826[0], tmp_4825[1] + tmp_4826[1], tmp_4825[2] + tmp_4826[2]];
    signal tmp_4828[3] <== [tmp_3866[0] * 6089787229781325813, tmp_3866[1] * 6089787229781325813, tmp_3866[2] * 6089787229781325813];
    signal tmp_4829[3] <== [tmp_4827[0] + tmp_4828[0], tmp_4827[1] + tmp_4828[1], tmp_4827[2] + tmp_4828[2]];
    signal tmp_4830[3] <== [evals[196][0] * 11192474480021157091, evals[196][1] * 11192474480021157091, evals[196][2] * 11192474480021157091];
    signal tmp_4831[3] <== [tmp_4829[0] + tmp_4830[0], tmp_4829[1] + tmp_4830[1], tmp_4829[2] + tmp_4830[2]];
    signal tmp_4832[3] <== [evals[76][0] * 9530849385991780809, evals[76][1] * 9530849385991780809, evals[76][2] * 9530849385991780809];
    signal tmp_4833[3] <== [tmp_4831[0] + tmp_4832[0], tmp_4831[1] + tmp_4832[1], tmp_4831[2] + tmp_4832[2]];
    signal tmp_4834[3] <== [evals[2][0] * 8824538844032254328, evals[2][1] * 8824538844032254328, evals[2][2] * 8824538844032254328];
    signal tmp_4835[3] <== [tmp_4833[0] + tmp_4834[0], tmp_4833[1] + tmp_4834[1], tmp_4833[2] + tmp_4834[2]];
    signal tmp_4836[3] <== [evals[197][0] * 5523099006222423939, evals[197][1] * 5523099006222423939, evals[197][2] * 5523099006222423939];
    signal tmp_4837[3] <== [tmp_4835[0] + tmp_4836[0], tmp_4835[1] + tmp_4836[1], tmp_4835[2] + tmp_4836[2]];
    signal tmp_4838[3] <== [evals[178][0] * 5714662808802647771, evals[178][1] * 5714662808802647771, evals[178][2] * 5714662808802647771];
    signal tmp_4839[3] <== [tmp_4837[0] + tmp_4838[0], tmp_4837[1] + tmp_4838[1], tmp_4837[2] + tmp_4838[2]];
    signal tmp_4840[3] <== [evals[77][0] * 17567284148478398281, evals[77][1] * 17567284148478398281, evals[77][2] * 17567284148478398281];
    tmp_4841 <== [tmp_4839[0] + tmp_4840[0], tmp_4839[1] + tmp_4840[1], tmp_4839[2] + tmp_4840[2]];
    signal tmp_4842[3] <== [evals[136][0] + tmp_4841[0], evals[136][1] + tmp_4841[1], evals[136][2] + tmp_4841[2]];
    signal tmp_4843[3] <== [evals[136][0] + tmp_4841[0], evals[136][1] + tmp_4841[1], evals[136][2] + tmp_4841[2]];
    tmp_4844 <== CMul()(tmp_4842, tmp_4843);
    tmp_4845 <== CMul()(evals[162], tmp_4844);
    signal tmp_4846[3] <== [evals[136][0] + tmp_4841[0], evals[136][1] + tmp_4841[1], evals[136][2] + tmp_4841[2]];
    signal tmp_4847[3] <== CMul()(tmp_4845, tmp_4846);
    signal tmp_4848[3] <== [evals[137][0] + tmp_4765[0], evals[137][1] + tmp_4765[1], evals[137][2] + tmp_4765[2]];
    signal tmp_4849[3] <== CMul()(tmp_4769, tmp_4848);
    tmp_4850 <== [tmp_4847[0] + tmp_4849[0], tmp_4847[1] + tmp_4849[1], tmp_4847[2] + tmp_4849[2]];
    tmp_4851 <== [tmp_4824[0] + tmp_4850[0], tmp_4824[1] + tmp_4850[1], tmp_4824[2] + tmp_4850[2]];
    tmp_4852 <== [tmp_4821[0] + tmp_4851[0], tmp_4821[1] + tmp_4851[1], tmp_4821[2] + tmp_4851[2]];
    tmp_4853 <== [tmp_4820[0] + tmp_4852[0], tmp_4820[1] + tmp_4852[1], tmp_4820[2] + tmp_4852[2]];
    tmp_4854 <== [tmp_4748[0] + tmp_4853[0], tmp_4748[1] + tmp_4853[1], tmp_4748[2] + tmp_4853[2]];
    signal tmp_4855[3] <== [tmp_4536[0] + tmp_4854[0], tmp_4536[1] + tmp_4854[1], tmp_4536[2] + tmp_4854[2]];
    signal tmp_4856[3] <== [tmp_4431[0] - tmp_4855[0], tmp_4431[1] - tmp_4855[1], tmp_4431[2] - tmp_4855[2]];
    signal tmp_4857[3] <== CMul()(tmp_4427, tmp_4856);
    signal tmp_4858[3] <== [tmp_4420[0] + tmp_4857[0], tmp_4420[1] + tmp_4857[1], tmp_4420[2] + tmp_4857[2]];
    tmp_4859 <== CMul()(challengeQ, tmp_4858);
    signal tmp_4860[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4861[3] <== [tmp_4860[0] + tmp_3866[0], tmp_4860[1] + tmp_3866[1], tmp_4860[2] + tmp_3866[2]];
    tmp_4862 <== [tmp_4861[0] + evals[196][0], tmp_4861[1] + evals[196][1], tmp_4861[2] + evals[196][2]];
}

template VerifyEvaluationsChunks1() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[198][3];
    signal input publics[391];

    signal input Zh[3];

    signal input tmp_3864[3];
 
    signal input tmp_3866[3];
 
    signal input tmp_3978[3];
 
    signal input tmp_4448[3];
 
    signal input tmp_4452[3];
 
    signal input tmp_4472[3];
 
    signal input tmp_4476[3];
 
    signal input tmp_4495[3];
 
    signal input tmp_4499[3];
 
    signal input tmp_4503[3];
 
    signal input tmp_4524[3];
 
    signal input tmp_4528[3];
 
    signal input tmp_4533[3];
 
    signal input tmp_4534[3];
 
    signal input tmp_4535[3];
 
    signal input tmp_4553[3];
 
    signal input tmp_4557[3];
 
    signal input tmp_4577[3];
 
    signal input tmp_4581[3];
 
    signal input tmp_4600[3];
 
    signal input tmp_4604[3];
 
    signal input tmp_4608[3];
 
    signal input tmp_4629[3];
 
    signal input tmp_4633[3];
 
    signal input tmp_4638[3];
 
    signal input tmp_4639[3];
 
    signal input tmp_4640[3];
 
    signal input tmp_4641[3];
 
    signal input tmp_4659[3];
 
    signal input tmp_4663[3];
 
    signal input tmp_4683[3];
 
    signal input tmp_4687[3];
 
    signal input tmp_4706[3];
 
    signal input tmp_4710[3];
 
    signal input tmp_4714[3];
 
    signal input tmp_4735[3];
 
    signal input tmp_4739[3];
 
    signal input tmp_4744[3];
 
    signal input tmp_4745[3];
 
    signal input tmp_4746[3];
 
    signal input tmp_4747[3];
 
    signal input tmp_4789[3];
 
    signal input tmp_4793[3];
 
    signal input tmp_4812[3];
 
    signal input tmp_4816[3];
 
    signal input tmp_4820[3];
 
    signal input tmp_4841[3];
 
    signal input tmp_4845[3];
 
    signal input tmp_4850[3];
 
    signal input tmp_4851[3];
 
    signal input tmp_4852[3];
 
    signal input tmp_4853[3];
 
    signal input tmp_4854[3];
 
    signal input tmp_4859[3];
 
    signal input tmp_4862[3];
 

    signal output tmp_5812[3];
 
    signal output tmp_5815[3];
 
    signal output tmp_5818[3];
 
    signal output tmp_5821[3];
 
    signal output tmp_5824[3];
 
    signal output tmp_5827[3];
 
    signal output tmp_5830[3];
 
    signal output tmp_5831[3];
 
    signal output tmp_5836[3];
 
    signal output tmp_5839[3];
 
    signal output tmp_5841[3];
 
    signal output tmp_5844[3];
 
    signal output tmp_5847[3];
 
    signal output tmp_5850[3];
 
    signal output tmp_5853[3];
 
    signal output tmp_5856[3];
 
    signal output tmp_5859[3];
 
    signal output tmp_5860[3];
 
    signal output tmp_5861[3];
 
    signal tmp_4863[3] <== [tmp_4862[0] + evals[2][0], tmp_4862[1] + evals[2][1], tmp_4862[2] + evals[2][2]];
    signal tmp_4864[3] <== [tmp_4863[0] + evals[197][0], tmp_4863[1] + evals[197][1], tmp_4863[2] + evals[197][2]];
    signal tmp_4865[3] <== [tmp_4864[0] + evals[178][0], tmp_4864[1] + evals[178][1], tmp_4864[2] + evals[178][2]];
    signal tmp_4866[3] <== [tmp_4865[0] + evals[77][0], tmp_4865[1] + evals[77][1], tmp_4865[2] + evals[77][2]];
    signal tmp_4867[3] <== CMul()(evals[77], evals[87]);
    signal tmp_4868[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4869[3] <== CMul()(tmp_4868, evals[181]);
    signal tmp_4870[3] <== [tmp_4867[0] + tmp_4869[0], tmp_4867[1] + tmp_4869[1], tmp_4867[2] + tmp_4869[2]];
    signal tmp_4871[3] <== [tmp_4535[0] + tmp_4640[0], tmp_4535[1] + tmp_4640[1], tmp_4535[2] + tmp_4640[2]];
    signal tmp_4872[3] <== [tmp_4871[0] + tmp_4746[0], tmp_4871[1] + tmp_4746[1], tmp_4871[2] + tmp_4746[2]];
    signal tmp_4873[3] <== [tmp_4872[0] + tmp_4852[0], tmp_4872[1] + tmp_4852[1], tmp_4872[2] + tmp_4852[2]];
    signal tmp_4874[3] <== [tmp_4535[0] + tmp_4873[0], tmp_4535[1] + tmp_4873[1], tmp_4535[2] + tmp_4873[2]];
    signal tmp_4875[3] <== [tmp_4870[0] - tmp_4874[0], tmp_4870[1] - tmp_4874[1], tmp_4870[2] - tmp_4874[2]];
    signal tmp_4876[3] <== CMul()(tmp_4866, tmp_4875);
    signal tmp_4877[3] <== [tmp_4859[0] + tmp_4876[0], tmp_4859[1] + tmp_4876[1], tmp_4859[2] + tmp_4876[2]];
    signal tmp_4878[3] <== CMul()(challengeQ, tmp_4877);
    signal tmp_4879[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4880[3] <== [tmp_4879[0] + tmp_3866[0], tmp_4879[1] + tmp_3866[1], tmp_4879[2] + tmp_3866[2]];
    signal tmp_4881[3] <== [tmp_4880[0] + evals[196][0], tmp_4880[1] + evals[196][1], tmp_4880[2] + evals[196][2]];
    signal tmp_4882[3] <== [tmp_4881[0] + evals[2][0], tmp_4881[1] + evals[2][1], tmp_4881[2] + evals[2][2]];
    signal tmp_4883[3] <== [tmp_4882[0] + evals[197][0], tmp_4882[1] + evals[197][1], tmp_4882[2] + evals[197][2]];
    signal tmp_4884[3] <== [tmp_4883[0] + evals[178][0], tmp_4883[1] + evals[178][1], tmp_4883[2] + evals[178][2]];
    signal tmp_4885[3] <== [tmp_4884[0] + evals[77][0], tmp_4884[1] + evals[77][1], tmp_4884[2] + evals[77][2]];
    signal tmp_4886[3] <== CMul()(evals[77], evals[88]);
    signal tmp_4887[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4888[3] <== CMul()(tmp_4887, evals[182]);
    signal tmp_4889[3] <== [tmp_4886[0] + tmp_4888[0], tmp_4886[1] + tmp_4888[1], tmp_4886[2] + tmp_4888[2]];
    signal tmp_4890[3] <== [4 * tmp_4533[0], 4 * tmp_4533[1], 4 * tmp_4533[2]];
    signal tmp_4891[3] <== [tmp_4890[0] + tmp_4503[0], tmp_4890[1] + tmp_4503[1], tmp_4890[2] + tmp_4503[2]];
    signal tmp_4892[3] <== [tmp_4534[0] + tmp_4891[0], tmp_4534[1] + tmp_4891[1], tmp_4534[2] + tmp_4891[2]];
    signal tmp_4893[3] <== [4 * tmp_4638[0], 4 * tmp_4638[1], 4 * tmp_4638[2]];
    signal tmp_4894[3] <== [tmp_4893[0] + tmp_4608[0], tmp_4893[1] + tmp_4608[1], tmp_4893[2] + tmp_4608[2]];
    signal tmp_4895[3] <== [tmp_4639[0] + tmp_4894[0], tmp_4639[1] + tmp_4894[1], tmp_4639[2] + tmp_4894[2]];
    signal tmp_4896[3] <== [tmp_4892[0] + tmp_4895[0], tmp_4892[1] + tmp_4895[1], tmp_4892[2] + tmp_4895[2]];
    signal tmp_4897[3] <== [4 * tmp_4744[0], 4 * tmp_4744[1], 4 * tmp_4744[2]];
    signal tmp_4898[3] <== [tmp_4897[0] + tmp_4714[0], tmp_4897[1] + tmp_4714[1], tmp_4897[2] + tmp_4714[2]];
    signal tmp_4899[3] <== [tmp_4745[0] + tmp_4898[0], tmp_4745[1] + tmp_4898[1], tmp_4745[2] + tmp_4898[2]];
    signal tmp_4900[3] <== [tmp_4896[0] + tmp_4899[0], tmp_4896[1] + tmp_4899[1], tmp_4896[2] + tmp_4899[2]];
    signal tmp_4901[3] <== [4 * tmp_4850[0], 4 * tmp_4850[1], 4 * tmp_4850[2]];
    signal tmp_4902[3] <== [tmp_4901[0] + tmp_4820[0], tmp_4901[1] + tmp_4820[1], tmp_4901[2] + tmp_4820[2]];
    signal tmp_4903[3] <== [tmp_4851[0] + tmp_4902[0], tmp_4851[1] + tmp_4902[1], tmp_4851[2] + tmp_4902[2]];
    signal tmp_4904[3] <== [tmp_4900[0] + tmp_4903[0], tmp_4900[1] + tmp_4903[1], tmp_4900[2] + tmp_4903[2]];
    signal tmp_4905[3] <== [tmp_4892[0] + tmp_4904[0], tmp_4892[1] + tmp_4904[1], tmp_4892[2] + tmp_4904[2]];
    signal tmp_4906[3] <== [tmp_4889[0] - tmp_4905[0], tmp_4889[1] - tmp_4905[1], tmp_4889[2] - tmp_4905[2]];
    signal tmp_4907[3] <== CMul()(tmp_4885, tmp_4906);
    signal tmp_4908[3] <== [tmp_4878[0] + tmp_4907[0], tmp_4878[1] + tmp_4907[1], tmp_4878[2] + tmp_4907[2]];
    signal tmp_4909[3] <== CMul()(challengeQ, tmp_4908);
    signal tmp_4910[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4911[3] <== [tmp_4910[0] + tmp_3866[0], tmp_4910[1] + tmp_3866[1], tmp_4910[2] + tmp_3866[2]];
    signal tmp_4912[3] <== [tmp_4911[0] + evals[196][0], tmp_4911[1] + evals[196][1], tmp_4911[2] + evals[196][2]];
    signal tmp_4913[3] <== [tmp_4912[0] + evals[2][0], tmp_4912[1] + evals[2][1], tmp_4912[2] + evals[2][2]];
    signal tmp_4914[3] <== [tmp_4913[0] + evals[197][0], tmp_4913[1] + evals[197][1], tmp_4913[2] + evals[197][2]];
    signal tmp_4915[3] <== [tmp_4914[0] + evals[178][0], tmp_4914[1] + evals[178][1], tmp_4914[2] + evals[178][2]];
    signal tmp_4916[3] <== [tmp_4915[0] + evals[77][0], tmp_4915[1] + evals[77][1], tmp_4915[2] + evals[77][2]];
    signal tmp_4917[3] <== CMul()(evals[77], evals[89]);
    signal tmp_4918[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4919[3] <== CMul()(tmp_4918, evals[183]);
    signal tmp_4920[3] <== [tmp_4917[0] + tmp_4919[0], tmp_4917[1] + tmp_4919[1], tmp_4917[2] + tmp_4919[2]];
    signal tmp_4921[3] <== [tmp_4891[0] + tmp_4894[0], tmp_4891[1] + tmp_4894[1], tmp_4891[2] + tmp_4894[2]];
    signal tmp_4922[3] <== [tmp_4921[0] + tmp_4898[0], tmp_4921[1] + tmp_4898[1], tmp_4921[2] + tmp_4898[2]];
    signal tmp_4923[3] <== [tmp_4922[0] + tmp_4902[0], tmp_4922[1] + tmp_4902[1], tmp_4922[2] + tmp_4902[2]];
    signal tmp_4924[3] <== [tmp_4891[0] + tmp_4923[0], tmp_4891[1] + tmp_4923[1], tmp_4891[2] + tmp_4923[2]];
    signal tmp_4925[3] <== [tmp_4920[0] - tmp_4924[0], tmp_4920[1] - tmp_4924[1], tmp_4920[2] - tmp_4924[2]];
    signal tmp_4926[3] <== CMul()(tmp_4916, tmp_4925);
    signal tmp_4927[3] <== [tmp_4909[0] + tmp_4926[0], tmp_4909[1] + tmp_4926[1], tmp_4909[2] + tmp_4926[2]];
    signal tmp_4928[3] <== CMul()(challengeQ, tmp_4927);
    signal tmp_4929[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4930[3] <== [tmp_4929[0] + tmp_3866[0], tmp_4929[1] + tmp_3866[1], tmp_4929[2] + tmp_3866[2]];
    signal tmp_4931[3] <== [tmp_4930[0] + evals[196][0], tmp_4930[1] + evals[196][1], tmp_4930[2] + evals[196][2]];
    signal tmp_4932[3] <== [tmp_4931[0] + evals[2][0], tmp_4931[1] + evals[2][1], tmp_4931[2] + evals[2][2]];
    signal tmp_4933[3] <== [tmp_4932[0] + evals[197][0], tmp_4932[1] + evals[197][1], tmp_4932[2] + evals[197][2]];
    signal tmp_4934[3] <== [tmp_4933[0] + evals[178][0], tmp_4933[1] + evals[178][1], tmp_4933[2] + evals[178][2]];
    signal tmp_4935[3] <== [tmp_4934[0] + evals[77][0], tmp_4934[1] + evals[77][1], tmp_4934[2] + evals[77][2]];
    signal tmp_4936[3] <== CMul()(evals[77], evals[90]);
    signal tmp_4937[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4938[3] <== CMul()(tmp_4937, evals[184]);
    signal tmp_4939[3] <== [tmp_4936[0] + tmp_4938[0], tmp_4936[1] + tmp_4938[1], tmp_4936[2] + tmp_4938[2]];
    signal tmp_4940[3] <== [tmp_4641[0] + tmp_4854[0], tmp_4641[1] + tmp_4854[1], tmp_4641[2] + tmp_4854[2]];
    signal tmp_4941[3] <== [tmp_4939[0] - tmp_4940[0], tmp_4939[1] - tmp_4940[1], tmp_4939[2] - tmp_4940[2]];
    signal tmp_4942[3] <== CMul()(tmp_4935, tmp_4941);
    signal tmp_4943[3] <== [tmp_4928[0] + tmp_4942[0], tmp_4928[1] + tmp_4942[1], tmp_4928[2] + tmp_4942[2]];
    signal tmp_4944[3] <== CMul()(challengeQ, tmp_4943);
    signal tmp_4945[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4946[3] <== [tmp_4945[0] + tmp_3866[0], tmp_4945[1] + tmp_3866[1], tmp_4945[2] + tmp_3866[2]];
    signal tmp_4947[3] <== [tmp_4946[0] + evals[196][0], tmp_4946[1] + evals[196][1], tmp_4946[2] + evals[196][2]];
    signal tmp_4948[3] <== [tmp_4947[0] + evals[2][0], tmp_4947[1] + evals[2][1], tmp_4947[2] + evals[2][2]];
    signal tmp_4949[3] <== [tmp_4948[0] + evals[197][0], tmp_4948[1] + evals[197][1], tmp_4948[2] + evals[197][2]];
    signal tmp_4950[3] <== [tmp_4949[0] + evals[178][0], tmp_4949[1] + evals[178][1], tmp_4949[2] + evals[178][2]];
    signal tmp_4951[3] <== [tmp_4950[0] + evals[77][0], tmp_4950[1] + evals[77][1], tmp_4950[2] + evals[77][2]];
    signal tmp_4952[3] <== CMul()(evals[77], evals[91]);
    signal tmp_4953[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4954[3] <== CMul()(tmp_4953, evals[185]);
    signal tmp_4955[3] <== [tmp_4952[0] + tmp_4954[0], tmp_4952[1] + tmp_4954[1], tmp_4952[2] + tmp_4954[2]];
    signal tmp_4956[3] <== [tmp_4640[0] + tmp_4873[0], tmp_4640[1] + tmp_4873[1], tmp_4640[2] + tmp_4873[2]];
    signal tmp_4957[3] <== [tmp_4955[0] - tmp_4956[0], tmp_4955[1] - tmp_4956[1], tmp_4955[2] - tmp_4956[2]];
    signal tmp_4958[3] <== CMul()(tmp_4951, tmp_4957);
    signal tmp_4959[3] <== [tmp_4944[0] + tmp_4958[0], tmp_4944[1] + tmp_4958[1], tmp_4944[2] + tmp_4958[2]];
    signal tmp_4960[3] <== CMul()(challengeQ, tmp_4959);
    signal tmp_4961[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4962[3] <== [tmp_4961[0] + tmp_3866[0], tmp_4961[1] + tmp_3866[1], tmp_4961[2] + tmp_3866[2]];
    signal tmp_4963[3] <== [tmp_4962[0] + evals[196][0], tmp_4962[1] + evals[196][1], tmp_4962[2] + evals[196][2]];
    signal tmp_4964[3] <== [tmp_4963[0] + evals[2][0], tmp_4963[1] + evals[2][1], tmp_4963[2] + evals[2][2]];
    signal tmp_4965[3] <== [tmp_4964[0] + evals[197][0], tmp_4964[1] + evals[197][1], tmp_4964[2] + evals[197][2]];
    signal tmp_4966[3] <== [tmp_4965[0] + evals[178][0], tmp_4965[1] + evals[178][1], tmp_4965[2] + evals[178][2]];
    signal tmp_4967[3] <== [tmp_4966[0] + evals[77][0], tmp_4966[1] + evals[77][1], tmp_4966[2] + evals[77][2]];
    signal tmp_4968[3] <== CMul()(evals[77], evals[92]);
    signal tmp_4969[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4970[3] <== CMul()(tmp_4969, evals[186]);
    signal tmp_4971[3] <== [tmp_4968[0] + tmp_4970[0], tmp_4968[1] + tmp_4970[1], tmp_4968[2] + tmp_4970[2]];
    signal tmp_4972[3] <== [tmp_4895[0] + tmp_4904[0], tmp_4895[1] + tmp_4904[1], tmp_4895[2] + tmp_4904[2]];
    signal tmp_4973[3] <== [tmp_4971[0] - tmp_4972[0], tmp_4971[1] - tmp_4972[1], tmp_4971[2] - tmp_4972[2]];
    signal tmp_4974[3] <== CMul()(tmp_4967, tmp_4973);
    signal tmp_4975[3] <== [tmp_4960[0] + tmp_4974[0], tmp_4960[1] + tmp_4974[1], tmp_4960[2] + tmp_4974[2]];
    signal tmp_4976[3] <== CMul()(challengeQ, tmp_4975);
    signal tmp_4977[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4978[3] <== [tmp_4977[0] + tmp_3866[0], tmp_4977[1] + tmp_3866[1], tmp_4977[2] + tmp_3866[2]];
    signal tmp_4979[3] <== [tmp_4978[0] + evals[196][0], tmp_4978[1] + evals[196][1], tmp_4978[2] + evals[196][2]];
    signal tmp_4980[3] <== [tmp_4979[0] + evals[2][0], tmp_4979[1] + evals[2][1], tmp_4979[2] + evals[2][2]];
    signal tmp_4981[3] <== [tmp_4980[0] + evals[197][0], tmp_4980[1] + evals[197][1], tmp_4980[2] + evals[197][2]];
    signal tmp_4982[3] <== [tmp_4981[0] + evals[178][0], tmp_4981[1] + evals[178][1], tmp_4981[2] + evals[178][2]];
    signal tmp_4983[3] <== [tmp_4982[0] + evals[77][0], tmp_4982[1] + evals[77][1], tmp_4982[2] + evals[77][2]];
    signal tmp_4984[3] <== CMul()(evals[77], evals[93]);
    signal tmp_4985[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_4986[3] <== CMul()(tmp_4985, evals[187]);
    signal tmp_4987[3] <== [tmp_4984[0] + tmp_4986[0], tmp_4984[1] + tmp_4986[1], tmp_4984[2] + tmp_4986[2]];
    signal tmp_4988[3] <== [tmp_4894[0] + tmp_4923[0], tmp_4894[1] + tmp_4923[1], tmp_4894[2] + tmp_4923[2]];
    signal tmp_4989[3] <== [tmp_4987[0] - tmp_4988[0], tmp_4987[1] - tmp_4988[1], tmp_4987[2] - tmp_4988[2]];
    signal tmp_4990[3] <== CMul()(tmp_4983, tmp_4989);
    signal tmp_4991[3] <== [tmp_4976[0] + tmp_4990[0], tmp_4976[1] + tmp_4990[1], tmp_4976[2] + tmp_4990[2]];
    signal tmp_4992[3] <== CMul()(challengeQ, tmp_4991);
    signal tmp_4993[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_4994[3] <== [tmp_4993[0] + tmp_3866[0], tmp_4993[1] + tmp_3866[1], tmp_4993[2] + tmp_3866[2]];
    signal tmp_4995[3] <== [tmp_4994[0] + evals[196][0], tmp_4994[1] + evals[196][1], tmp_4994[2] + evals[196][2]];
    signal tmp_4996[3] <== [tmp_4995[0] + evals[2][0], tmp_4995[1] + evals[2][1], tmp_4995[2] + evals[2][2]];
    signal tmp_4997[3] <== [tmp_4996[0] + evals[197][0], tmp_4996[1] + evals[197][1], tmp_4996[2] + evals[197][2]];
    signal tmp_4998[3] <== [tmp_4997[0] + evals[178][0], tmp_4997[1] + evals[178][1], tmp_4997[2] + evals[178][2]];
    signal tmp_4999[3] <== [tmp_4998[0] + evals[77][0], tmp_4998[1] + evals[77][1], tmp_4998[2] + evals[77][2]];
    signal tmp_5000[3] <== CMul()(evals[77], evals[94]);
    signal tmp_5001[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5002[3] <== CMul()(tmp_5001, evals[188]);
    signal tmp_5003[3] <== [tmp_5000[0] + tmp_5002[0], tmp_5000[1] + tmp_5002[1], tmp_5000[2] + tmp_5002[2]];
    signal tmp_5004[3] <== [tmp_4747[0] + tmp_4854[0], tmp_4747[1] + tmp_4854[1], tmp_4747[2] + tmp_4854[2]];
    signal tmp_5005[3] <== [tmp_5003[0] - tmp_5004[0], tmp_5003[1] - tmp_5004[1], tmp_5003[2] - tmp_5004[2]];
    signal tmp_5006[3] <== CMul()(tmp_4999, tmp_5005);
    signal tmp_5007[3] <== [tmp_4992[0] + tmp_5006[0], tmp_4992[1] + tmp_5006[1], tmp_4992[2] + tmp_5006[2]];
    signal tmp_5008[3] <== CMul()(challengeQ, tmp_5007);
    signal tmp_5009[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5010[3] <== [tmp_5009[0] + tmp_3866[0], tmp_5009[1] + tmp_3866[1], tmp_5009[2] + tmp_3866[2]];
    signal tmp_5011[3] <== [tmp_5010[0] + evals[196][0], tmp_5010[1] + evals[196][1], tmp_5010[2] + evals[196][2]];
    signal tmp_5012[3] <== [tmp_5011[0] + evals[2][0], tmp_5011[1] + evals[2][1], tmp_5011[2] + evals[2][2]];
    signal tmp_5013[3] <== [tmp_5012[0] + evals[197][0], tmp_5012[1] + evals[197][1], tmp_5012[2] + evals[197][2]];
    signal tmp_5014[3] <== [tmp_5013[0] + evals[178][0], tmp_5013[1] + evals[178][1], tmp_5013[2] + evals[178][2]];
    signal tmp_5015[3] <== [tmp_5014[0] + evals[77][0], tmp_5014[1] + evals[77][1], tmp_5014[2] + evals[77][2]];
    signal tmp_5016[3] <== CMul()(evals[77], evals[95]);
    signal tmp_5017[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5018[3] <== CMul()(tmp_5017, evals[189]);
    signal tmp_5019[3] <== [tmp_5016[0] + tmp_5018[0], tmp_5016[1] + tmp_5018[1], tmp_5016[2] + tmp_5018[2]];
    signal tmp_5020[3] <== [tmp_4746[0] + tmp_4873[0], tmp_4746[1] + tmp_4873[1], tmp_4746[2] + tmp_4873[2]];
    signal tmp_5021[3] <== [tmp_5019[0] - tmp_5020[0], tmp_5019[1] - tmp_5020[1], tmp_5019[2] - tmp_5020[2]];
    signal tmp_5022[3] <== CMul()(tmp_5015, tmp_5021);
    signal tmp_5023[3] <== [tmp_5008[0] + tmp_5022[0], tmp_5008[1] + tmp_5022[1], tmp_5008[2] + tmp_5022[2]];
    signal tmp_5024[3] <== CMul()(challengeQ, tmp_5023);
    signal tmp_5025[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5026[3] <== [tmp_5025[0] + tmp_3866[0], tmp_5025[1] + tmp_3866[1], tmp_5025[2] + tmp_3866[2]];
    signal tmp_5027[3] <== [tmp_5026[0] + evals[196][0], tmp_5026[1] + evals[196][1], tmp_5026[2] + evals[196][2]];
    signal tmp_5028[3] <== [tmp_5027[0] + evals[2][0], tmp_5027[1] + evals[2][1], tmp_5027[2] + evals[2][2]];
    signal tmp_5029[3] <== [tmp_5028[0] + evals[197][0], tmp_5028[1] + evals[197][1], tmp_5028[2] + evals[197][2]];
    signal tmp_5030[3] <== [tmp_5029[0] + evals[178][0], tmp_5029[1] + evals[178][1], tmp_5029[2] + evals[178][2]];
    signal tmp_5031[3] <== [tmp_5030[0] + evals[77][0], tmp_5030[1] + evals[77][1], tmp_5030[2] + evals[77][2]];
    signal tmp_5032[3] <== CMul()(evals[77], evals[96]);
    signal tmp_5033[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5034[3] <== CMul()(tmp_5033, evals[190]);
    signal tmp_5035[3] <== [tmp_5032[0] + tmp_5034[0], tmp_5032[1] + tmp_5034[1], tmp_5032[2] + tmp_5034[2]];
    signal tmp_5036[3] <== [tmp_4899[0] + tmp_4904[0], tmp_4899[1] + tmp_4904[1], tmp_4899[2] + tmp_4904[2]];
    signal tmp_5037[3] <== [tmp_5035[0] - tmp_5036[0], tmp_5035[1] - tmp_5036[1], tmp_5035[2] - tmp_5036[2]];
    signal tmp_5038[3] <== CMul()(tmp_5031, tmp_5037);
    signal tmp_5039[3] <== [tmp_5024[0] + tmp_5038[0], tmp_5024[1] + tmp_5038[1], tmp_5024[2] + tmp_5038[2]];
    signal tmp_5040[3] <== CMul()(challengeQ, tmp_5039);
    signal tmp_5041[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5042[3] <== [tmp_5041[0] + tmp_3866[0], tmp_5041[1] + tmp_3866[1], tmp_5041[2] + tmp_3866[2]];
    signal tmp_5043[3] <== [tmp_5042[0] + evals[196][0], tmp_5042[1] + evals[196][1], tmp_5042[2] + evals[196][2]];
    signal tmp_5044[3] <== [tmp_5043[0] + evals[2][0], tmp_5043[1] + evals[2][1], tmp_5043[2] + evals[2][2]];
    signal tmp_5045[3] <== [tmp_5044[0] + evals[197][0], tmp_5044[1] + evals[197][1], tmp_5044[2] + evals[197][2]];
    signal tmp_5046[3] <== [tmp_5045[0] + evals[178][0], tmp_5045[1] + evals[178][1], tmp_5045[2] + evals[178][2]];
    signal tmp_5047[3] <== [tmp_5046[0] + evals[77][0], tmp_5046[1] + evals[77][1], tmp_5046[2] + evals[77][2]];
    signal tmp_5048[3] <== CMul()(evals[77], evals[97]);
    signal tmp_5049[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5050[3] <== CMul()(tmp_5049, evals[191]);
    signal tmp_5051[3] <== [tmp_5048[0] + tmp_5050[0], tmp_5048[1] + tmp_5050[1], tmp_5048[2] + tmp_5050[2]];
    signal tmp_5052[3] <== [tmp_4898[0] + tmp_4923[0], tmp_4898[1] + tmp_4923[1], tmp_4898[2] + tmp_4923[2]];
    signal tmp_5053[3] <== [tmp_5051[0] - tmp_5052[0], tmp_5051[1] - tmp_5052[1], tmp_5051[2] - tmp_5052[2]];
    signal tmp_5054[3] <== CMul()(tmp_5047, tmp_5053);
    signal tmp_5055[3] <== [tmp_5040[0] + tmp_5054[0], tmp_5040[1] + tmp_5054[1], tmp_5040[2] + tmp_5054[2]];
    signal tmp_5056[3] <== CMul()(challengeQ, tmp_5055);
    signal tmp_5057[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5058[3] <== [tmp_5057[0] + tmp_3866[0], tmp_5057[1] + tmp_3866[1], tmp_5057[2] + tmp_3866[2]];
    signal tmp_5059[3] <== [tmp_5058[0] + evals[196][0], tmp_5058[1] + evals[196][1], tmp_5058[2] + evals[196][2]];
    signal tmp_5060[3] <== [tmp_5059[0] + evals[2][0], tmp_5059[1] + evals[2][1], tmp_5059[2] + evals[2][2]];
    signal tmp_5061[3] <== [tmp_5060[0] + evals[197][0], tmp_5060[1] + evals[197][1], tmp_5060[2] + evals[197][2]];
    signal tmp_5062[3] <== [tmp_5061[0] + evals[178][0], tmp_5061[1] + evals[178][1], tmp_5061[2] + evals[178][2]];
    signal tmp_5063[3] <== [tmp_5062[0] + evals[77][0], tmp_5062[1] + evals[77][1], tmp_5062[2] + evals[77][2]];
    signal tmp_5064[3] <== CMul()(evals[77], evals[98]);
    signal tmp_5065[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5066[3] <== CMul()(tmp_5065, evals[192]);
    signal tmp_5067[3] <== [tmp_5064[0] + tmp_5066[0], tmp_5064[1] + tmp_5066[1], tmp_5064[2] + tmp_5066[2]];
    signal tmp_5068[3] <== [tmp_4853[0] + tmp_4854[0], tmp_4853[1] + tmp_4854[1], tmp_4853[2] + tmp_4854[2]];
    signal tmp_5069[3] <== [tmp_5067[0] - tmp_5068[0], tmp_5067[1] - tmp_5068[1], tmp_5067[2] - tmp_5068[2]];
    signal tmp_5070[3] <== CMul()(tmp_5063, tmp_5069);
    signal tmp_5071[3] <== [tmp_5056[0] + tmp_5070[0], tmp_5056[1] + tmp_5070[1], tmp_5056[2] + tmp_5070[2]];
    signal tmp_5072[3] <== CMul()(challengeQ, tmp_5071);
    signal tmp_5073[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5074[3] <== [tmp_5073[0] + tmp_3866[0], tmp_5073[1] + tmp_3866[1], tmp_5073[2] + tmp_3866[2]];
    signal tmp_5075[3] <== [tmp_5074[0] + evals[196][0], tmp_5074[1] + evals[196][1], tmp_5074[2] + evals[196][2]];
    signal tmp_5076[3] <== [tmp_5075[0] + evals[2][0], tmp_5075[1] + evals[2][1], tmp_5075[2] + evals[2][2]];
    signal tmp_5077[3] <== [tmp_5076[0] + evals[197][0], tmp_5076[1] + evals[197][1], tmp_5076[2] + evals[197][2]];
    signal tmp_5078[3] <== [tmp_5077[0] + evals[178][0], tmp_5077[1] + evals[178][1], tmp_5077[2] + evals[178][2]];
    signal tmp_5079[3] <== [tmp_5078[0] + evals[77][0], tmp_5078[1] + evals[77][1], tmp_5078[2] + evals[77][2]];
    signal tmp_5080[3] <== CMul()(evals[77], evals[99]);
    signal tmp_5081[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5082[3] <== CMul()(tmp_5081, evals[193]);
    signal tmp_5083[3] <== [tmp_5080[0] + tmp_5082[0], tmp_5080[1] + tmp_5082[1], tmp_5080[2] + tmp_5082[2]];
    signal tmp_5084[3] <== [tmp_4852[0] + tmp_4873[0], tmp_4852[1] + tmp_4873[1], tmp_4852[2] + tmp_4873[2]];
    signal tmp_5085[3] <== [tmp_5083[0] - tmp_5084[0], tmp_5083[1] - tmp_5084[1], tmp_5083[2] - tmp_5084[2]];
    signal tmp_5086[3] <== CMul()(tmp_5079, tmp_5085);
    signal tmp_5087[3] <== [tmp_5072[0] + tmp_5086[0], tmp_5072[1] + tmp_5086[1], tmp_5072[2] + tmp_5086[2]];
    signal tmp_5088[3] <== CMul()(challengeQ, tmp_5087);
    signal tmp_5089[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5090[3] <== [tmp_5089[0] + tmp_3866[0], tmp_5089[1] + tmp_3866[1], tmp_5089[2] + tmp_3866[2]];
    signal tmp_5091[3] <== [tmp_5090[0] + evals[196][0], tmp_5090[1] + evals[196][1], tmp_5090[2] + evals[196][2]];
    signal tmp_5092[3] <== [tmp_5091[0] + evals[2][0], tmp_5091[1] + evals[2][1], tmp_5091[2] + evals[2][2]];
    signal tmp_5093[3] <== [tmp_5092[0] + evals[197][0], tmp_5092[1] + evals[197][1], tmp_5092[2] + evals[197][2]];
    signal tmp_5094[3] <== [tmp_5093[0] + evals[178][0], tmp_5093[1] + evals[178][1], tmp_5093[2] + evals[178][2]];
    signal tmp_5095[3] <== [tmp_5094[0] + evals[77][0], tmp_5094[1] + evals[77][1], tmp_5094[2] + evals[77][2]];
    signal tmp_5096[3] <== CMul()(evals[77], evals[100]);
    signal tmp_5097[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5098[3] <== CMul()(tmp_5097, evals[194]);
    signal tmp_5099[3] <== [tmp_5096[0] + tmp_5098[0], tmp_5096[1] + tmp_5098[1], tmp_5096[2] + tmp_5098[2]];
    signal tmp_5100[3] <== [tmp_4903[0] + tmp_4904[0], tmp_4903[1] + tmp_4904[1], tmp_4903[2] + tmp_4904[2]];
    signal tmp_5101[3] <== [tmp_5099[0] - tmp_5100[0], tmp_5099[1] - tmp_5100[1], tmp_5099[2] - tmp_5100[2]];
    signal tmp_5102[3] <== CMul()(tmp_5095, tmp_5101);
    signal tmp_5103[3] <== [tmp_5088[0] + tmp_5102[0], tmp_5088[1] + tmp_5102[1], tmp_5088[2] + tmp_5102[2]];
    signal tmp_5104[3] <== CMul()(challengeQ, tmp_5103);
    signal tmp_5105[3] <== [tmp_3978[0] + tmp_3864[0], tmp_3978[1] + tmp_3864[1], tmp_3978[2] + tmp_3864[2]];
    signal tmp_5106[3] <== [tmp_5105[0] + tmp_3866[0], tmp_5105[1] + tmp_3866[1], tmp_5105[2] + tmp_3866[2]];
    signal tmp_5107[3] <== [tmp_5106[0] + evals[196][0], tmp_5106[1] + evals[196][1], tmp_5106[2] + evals[196][2]];
    signal tmp_5108[3] <== [tmp_5107[0] + evals[2][0], tmp_5107[1] + evals[2][1], tmp_5107[2] + evals[2][2]];
    signal tmp_5109[3] <== [tmp_5108[0] + evals[197][0], tmp_5108[1] + evals[197][1], tmp_5108[2] + evals[197][2]];
    signal tmp_5110[3] <== [tmp_5109[0] + evals[178][0], tmp_5109[1] + evals[178][1], tmp_5109[2] + evals[178][2]];
    signal tmp_5111[3] <== [tmp_5110[0] + evals[77][0], tmp_5110[1] + evals[77][1], tmp_5110[2] + evals[77][2]];
    signal tmp_5112[3] <== CMul()(evals[77], evals[101]);
    signal tmp_5113[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_5114[3] <== CMul()(tmp_5113, evals[195]);
    signal tmp_5115[3] <== [tmp_5112[0] + tmp_5114[0], tmp_5112[1] + tmp_5114[1], tmp_5112[2] + tmp_5114[2]];
    signal tmp_5116[3] <== [tmp_4902[0] + tmp_4923[0], tmp_4902[1] + tmp_4923[1], tmp_4902[2] + tmp_4923[2]];
    signal tmp_5117[3] <== [tmp_5115[0] - tmp_5116[0], tmp_5115[1] - tmp_5116[1], tmp_5115[2] - tmp_5116[2]];
    signal tmp_5118[3] <== CMul()(tmp_5111, tmp_5117);
    signal tmp_5119[3] <== [tmp_5104[0] + tmp_5118[0], tmp_5104[1] + tmp_5118[1], tmp_5104[2] + tmp_5118[2]];
    signal tmp_5120[3] <== CMul()(challengeQ, tmp_5119);
    signal tmp_5121[3] <== [evals[122][0] - evals[3][0], evals[122][1] - evals[3][1], evals[122][2] - evals[3][2]];
    signal tmp_5122[3] <== CMul()(evals[76], tmp_5121);
    signal tmp_5123[3] <== [tmp_5120[0] + tmp_5122[0], tmp_5120[1] + tmp_5122[1], tmp_5120[2] + tmp_5122[2]];
    signal tmp_5124[3] <== CMul()(challengeQ, tmp_5123);
    signal tmp_5125[3] <== [evals[122][0] + tmp_4472[0], evals[122][1] + tmp_4472[1], evals[122][2] + tmp_4472[2]];
    signal tmp_5126[3] <== CMul()(tmp_4476, tmp_5125);
    signal tmp_5127[3] <== [tmp_5126[0] * 16040574633112940480, tmp_5126[1] * 16040574633112940480, tmp_5126[2] * 16040574633112940480];
    signal tmp_5128[3] <== [tmp_5126[0] + evals[4][0], tmp_5126[1] + evals[4][1], tmp_5126[2] + evals[4][2]];
    signal tmp_5129[3] <== [tmp_5128[0] + evals[5][0], tmp_5128[1] + evals[5][1], tmp_5128[2] + evals[5][2]];
    signal tmp_5130[3] <== [tmp_5129[0] + evals[6][0], tmp_5129[1] + evals[6][1], tmp_5129[2] + evals[6][2]];
    signal tmp_5131[3] <== [tmp_5130[0] + evals[7][0], tmp_5130[1] + evals[7][1], tmp_5130[2] + evals[7][2]];
    signal tmp_5132[3] <== [tmp_5131[0] + evals[8][0], tmp_5131[1] + evals[8][1], tmp_5131[2] + evals[8][2]];
    signal tmp_5133[3] <== [tmp_5132[0] + evals[9][0], tmp_5132[1] + evals[9][1], tmp_5132[2] + evals[9][2]];
    signal tmp_5134[3] <== [tmp_5133[0] + evals[10][0], tmp_5133[1] + evals[10][1], tmp_5133[2] + evals[10][2]];
    signal tmp_5135[3] <== [tmp_5134[0] + evals[11][0], tmp_5134[1] + evals[11][1], tmp_5134[2] + evals[11][2]];
    signal tmp_5136[3] <== [tmp_5135[0] + evals[12][0], tmp_5135[1] + evals[12][1], tmp_5135[2] + evals[12][2]];
    signal tmp_5137[3] <== [tmp_5136[0] + evals[13][0], tmp_5136[1] + evals[13][1], tmp_5136[2] + evals[13][2]];
    signal tmp_5138[3] <== [tmp_5137[0] + evals[14][0], tmp_5137[1] + evals[14][1], tmp_5137[2] + evals[14][2]];
    signal tmp_5139[3] <== [tmp_5138[0] + evals[15][0], tmp_5138[1] + evals[15][1], tmp_5138[2] + evals[15][2]];
    signal tmp_5140[3] <== [tmp_5139[0] + evals[16][0], tmp_5139[1] + evals[16][1], tmp_5139[2] + evals[16][2]];
    signal tmp_5141[3] <== [tmp_5140[0] + evals[17][0], tmp_5140[1] + evals[17][1], tmp_5140[2] + evals[17][2]];
    signal tmp_5142[3] <== [tmp_5141[0] + evals[18][0], tmp_5141[1] + evals[18][1], tmp_5141[2] + evals[18][2]];
    signal tmp_5143[3] <== [tmp_5127[0] + tmp_5142[0], tmp_5127[1] + tmp_5142[1], tmp_5127[2] + tmp_5142[2]];
    signal tmp_5144[3] <== [evals[123][0] - tmp_5143[0], evals[123][1] - tmp_5143[1], evals[123][2] - tmp_5143[2]];
    signal tmp_5145[3] <== CMul()(evals[76], tmp_5144);
    signal tmp_5146[3] <== [tmp_5124[0] + tmp_5145[0], tmp_5124[1] + tmp_5145[1], tmp_5124[2] + tmp_5145[2]];
    signal tmp_5147[3] <== CMul()(challengeQ, tmp_5146);
    signal tmp_5148[3] <== [evals[123][0] + tmp_4495[0], evals[123][1] + tmp_4495[1], evals[123][2] + tmp_4495[2]];
    signal tmp_5149[3] <== CMul()(tmp_4499, tmp_5148);
    signal tmp_5150[3] <== [tmp_5149[0] * 16040574633112940480, tmp_5149[1] * 16040574633112940480, tmp_5149[2] * 16040574633112940480];
    signal tmp_5151[3] <== [evals[4][0] * 14263299814608977431, evals[4][1] * 14263299814608977431, evals[4][2] * 14263299814608977431];
    signal tmp_5152[3] <== [tmp_5151[0] + tmp_5142[0], tmp_5151[1] + tmp_5142[1], tmp_5151[2] + tmp_5142[2]];
    signal tmp_5153[3] <== [tmp_5149[0] + tmp_5152[0], tmp_5149[1] + tmp_5152[1], tmp_5149[2] + tmp_5152[2]];
    signal tmp_5154[3] <== [evals[5][0] * 770395855193680981, evals[5][1] * 770395855193680981, evals[5][2] * 770395855193680981];
    signal tmp_5155[3] <== [tmp_5154[0] + tmp_5142[0], tmp_5154[1] + tmp_5142[1], tmp_5154[2] + tmp_5142[2]];
    signal tmp_5156[3] <== [tmp_5153[0] + tmp_5155[0], tmp_5153[1] + tmp_5155[1], tmp_5153[2] + tmp_5155[2]];
    signal tmp_5157[3] <== [evals[6][0] * 3459277367440070515, evals[6][1] * 3459277367440070515, evals[6][2] * 3459277367440070515];
    signal tmp_5158[3] <== [tmp_5157[0] + tmp_5142[0], tmp_5157[1] + tmp_5142[1], tmp_5157[2] + tmp_5142[2]];
    signal tmp_5159[3] <== [tmp_5156[0] + tmp_5158[0], tmp_5156[1] + tmp_5158[1], tmp_5156[2] + tmp_5158[2]];
    signal tmp_5160[3] <== [evals[7][0] * 17087697094293314027, evals[7][1] * 17087697094293314027, evals[7][2] * 17087697094293314027];
    signal tmp_5161[3] <== [tmp_5160[0] + tmp_5142[0], tmp_5160[1] + tmp_5142[1], tmp_5160[2] + tmp_5142[2]];
    signal tmp_5162[3] <== [tmp_5159[0] + tmp_5161[0], tmp_5159[1] + tmp_5161[1], tmp_5159[2] + tmp_5161[2]];
    signal tmp_5163[3] <== [evals[8][0] * 6694380135428747348, evals[8][1] * 6694380135428747348, evals[8][2] * 6694380135428747348];
    signal tmp_5164[3] <== [tmp_5163[0] + tmp_5142[0], tmp_5163[1] + tmp_5142[1], tmp_5163[2] + tmp_5142[2]];
    signal tmp_5165[3] <== [tmp_5162[0] + tmp_5164[0], tmp_5162[1] + tmp_5164[1], tmp_5162[2] + tmp_5164[2]];
    signal tmp_5166[3] <== [evals[9][0] * 2034408310088972836, evals[9][1] * 2034408310088972836, evals[9][2] * 2034408310088972836];
    signal tmp_5167[3] <== [tmp_5166[0] + tmp_5142[0], tmp_5166[1] + tmp_5142[1], tmp_5166[2] + tmp_5142[2]];
    signal tmp_5168[3] <== [tmp_5165[0] + tmp_5167[0], tmp_5165[1] + tmp_5167[1], tmp_5165[2] + tmp_5167[2]];
    signal tmp_5169[3] <== [evals[10][0] * 3434575637390274478, evals[10][1] * 3434575637390274478, evals[10][2] * 3434575637390274478];
    signal tmp_5170[3] <== [tmp_5169[0] + tmp_5142[0], tmp_5169[1] + tmp_5142[1], tmp_5169[2] + tmp_5142[2]];
    signal tmp_5171[3] <== [tmp_5168[0] + tmp_5170[0], tmp_5168[1] + tmp_5170[1], tmp_5168[2] + tmp_5170[2]];
    signal tmp_5172[3] <== [evals[11][0] * 6052753985947965968, evals[11][1] * 6052753985947965968, evals[11][2] * 6052753985947965968];
    signal tmp_5173[3] <== [tmp_5172[0] + tmp_5142[0], tmp_5172[1] + tmp_5142[1], tmp_5172[2] + tmp_5142[2]];
    signal tmp_5174[3] <== [tmp_5171[0] + tmp_5173[0], tmp_5171[1] + tmp_5173[1], tmp_5171[2] + tmp_5173[2]];
    signal tmp_5175[3] <== [evals[12][0] * 13608362914817483670, evals[12][1] * 13608362914817483670, evals[12][2] * 13608362914817483670];
    signal tmp_5176[3] <== [tmp_5175[0] + tmp_5142[0], tmp_5175[1] + tmp_5142[1], tmp_5175[2] + tmp_5142[2]];
    signal tmp_5177[3] <== [tmp_5174[0] + tmp_5176[0], tmp_5174[1] + tmp_5176[1], tmp_5174[2] + tmp_5176[2]];
    signal tmp_5178[3] <== [evals[13][0] * 18163707672964630459, evals[13][1] * 18163707672964630459, evals[13][2] * 18163707672964630459];
    signal tmp_5179[3] <== [tmp_5178[0] + tmp_5142[0], tmp_5178[1] + tmp_5142[1], tmp_5178[2] + tmp_5142[2]];
    signal tmp_5180[3] <== [tmp_5177[0] + tmp_5179[0], tmp_5177[1] + tmp_5179[1], tmp_5177[2] + tmp_5179[2]];
    signal tmp_5181[3] <== [evals[14][0] * 14373610220374016704, evals[14][1] * 14373610220374016704, evals[14][2] * 14373610220374016704];
    signal tmp_5182[3] <== [tmp_5181[0] + tmp_5142[0], tmp_5181[1] + tmp_5142[1], tmp_5181[2] + tmp_5142[2]];
    signal tmp_5183[3] <== [tmp_5180[0] + tmp_5182[0], tmp_5180[1] + tmp_5182[1], tmp_5180[2] + tmp_5182[2]];
    signal tmp_5184[3] <== [evals[15][0] * 6226282807566121054, evals[15][1] * 6226282807566121054, evals[15][2] * 6226282807566121054];
    signal tmp_5185[3] <== [tmp_5184[0] + tmp_5142[0], tmp_5184[1] + tmp_5142[1], tmp_5184[2] + tmp_5142[2]];
    signal tmp_5186[3] <== [tmp_5183[0] + tmp_5185[0], tmp_5183[1] + tmp_5185[1], tmp_5183[2] + tmp_5185[2]];
    signal tmp_5187[3] <== [evals[16][0] * 3643354756180461803, evals[16][1] * 3643354756180461803, evals[16][2] * 3643354756180461803];
    signal tmp_5188[3] <== [tmp_5187[0] + tmp_5142[0], tmp_5187[1] + tmp_5142[1], tmp_5187[2] + tmp_5142[2]];
    signal tmp_5189[3] <== [tmp_5186[0] + tmp_5188[0], tmp_5186[1] + tmp_5188[1], tmp_5186[2] + tmp_5188[2]];
    signal tmp_5190[3] <== [evals[17][0] * 13046961313070095543, evals[17][1] * 13046961313070095543, evals[17][2] * 13046961313070095543];
    signal tmp_5191[3] <== [tmp_5190[0] + tmp_5142[0], tmp_5190[1] + tmp_5142[1], tmp_5190[2] + tmp_5142[2]];
    signal tmp_5192[3] <== [tmp_5189[0] + tmp_5191[0], tmp_5189[1] + tmp_5191[1], tmp_5189[2] + tmp_5191[2]];
    signal tmp_5193[3] <== [evals[18][0] * 8594143216561850811, evals[18][1] * 8594143216561850811, evals[18][2] * 8594143216561850811];
    signal tmp_5194[3] <== [tmp_5193[0] + tmp_5142[0], tmp_5193[1] + tmp_5142[1], tmp_5193[2] + tmp_5142[2]];
    signal tmp_5195[3] <== [tmp_5192[0] + tmp_5194[0], tmp_5192[1] + tmp_5194[1], tmp_5192[2] + tmp_5194[2]];
    signal tmp_5196[3] <== [tmp_5150[0] + tmp_5195[0], tmp_5150[1] + tmp_5195[1], tmp_5150[2] + tmp_5195[2]];
    signal tmp_5197[3] <== [evals[124][0] - tmp_5196[0], evals[124][1] - tmp_5196[1], evals[124][2] - tmp_5196[2]];
    signal tmp_5198[3] <== CMul()(evals[76], tmp_5197);
    signal tmp_5199[3] <== [tmp_5147[0] + tmp_5198[0], tmp_5147[1] + tmp_5198[1], tmp_5147[2] + tmp_5198[2]];
    signal tmp_5200[3] <== CMul()(challengeQ, tmp_5199);
    signal tmp_5201[3] <== [evals[124][0] + tmp_4524[0], evals[124][1] + tmp_4524[1], evals[124][2] + tmp_4524[2]];
    signal tmp_5202[3] <== CMul()(tmp_4528, tmp_5201);
    signal tmp_5203[3] <== [tmp_5202[0] * 16040574633112940480, tmp_5202[1] * 16040574633112940480, tmp_5202[2] * 16040574633112940480];
    signal tmp_5204[3] <== [tmp_5152[0] * 14263299814608977431, tmp_5152[1] * 14263299814608977431, tmp_5152[2] * 14263299814608977431];
    signal tmp_5205[3] <== [tmp_5204[0] + tmp_5195[0], tmp_5204[1] + tmp_5195[1], tmp_5204[2] + tmp_5195[2]];
    signal tmp_5206[3] <== [tmp_5202[0] + tmp_5205[0], tmp_5202[1] + tmp_5205[1], tmp_5202[2] + tmp_5205[2]];
    signal tmp_5207[3] <== [tmp_5155[0] * 770395855193680981, tmp_5155[1] * 770395855193680981, tmp_5155[2] * 770395855193680981];
    signal tmp_5208[3] <== [tmp_5207[0] + tmp_5195[0], tmp_5207[1] + tmp_5195[1], tmp_5207[2] + tmp_5195[2]];
    signal tmp_5209[3] <== [tmp_5206[0] + tmp_5208[0], tmp_5206[1] + tmp_5208[1], tmp_5206[2] + tmp_5208[2]];
    signal tmp_5210[3] <== [tmp_5158[0] * 3459277367440070515, tmp_5158[1] * 3459277367440070515, tmp_5158[2] * 3459277367440070515];
    signal tmp_5211[3] <== [tmp_5210[0] + tmp_5195[0], tmp_5210[1] + tmp_5195[1], tmp_5210[2] + tmp_5195[2]];
    signal tmp_5212[3] <== [tmp_5209[0] + tmp_5211[0], tmp_5209[1] + tmp_5211[1], tmp_5209[2] + tmp_5211[2]];
    signal tmp_5213[3] <== [tmp_5161[0] * 17087697094293314027, tmp_5161[1] * 17087697094293314027, tmp_5161[2] * 17087697094293314027];
    signal tmp_5214[3] <== [tmp_5213[0] + tmp_5195[0], tmp_5213[1] + tmp_5195[1], tmp_5213[2] + tmp_5195[2]];
    signal tmp_5215[3] <== [tmp_5212[0] + tmp_5214[0], tmp_5212[1] + tmp_5214[1], tmp_5212[2] + tmp_5214[2]];
    signal tmp_5216[3] <== [tmp_5164[0] * 6694380135428747348, tmp_5164[1] * 6694380135428747348, tmp_5164[2] * 6694380135428747348];
    signal tmp_5217[3] <== [tmp_5216[0] + tmp_5195[0], tmp_5216[1] + tmp_5195[1], tmp_5216[2] + tmp_5195[2]];
    signal tmp_5218[3] <== [tmp_5215[0] + tmp_5217[0], tmp_5215[1] + tmp_5217[1], tmp_5215[2] + tmp_5217[2]];
    signal tmp_5219[3] <== [tmp_5167[0] * 2034408310088972836, tmp_5167[1] * 2034408310088972836, tmp_5167[2] * 2034408310088972836];
    signal tmp_5220[3] <== [tmp_5219[0] + tmp_5195[0], tmp_5219[1] + tmp_5195[1], tmp_5219[2] + tmp_5195[2]];
    signal tmp_5221[3] <== [tmp_5218[0] + tmp_5220[0], tmp_5218[1] + tmp_5220[1], tmp_5218[2] + tmp_5220[2]];
    signal tmp_5222[3] <== [tmp_5170[0] * 3434575637390274478, tmp_5170[1] * 3434575637390274478, tmp_5170[2] * 3434575637390274478];
    signal tmp_5223[3] <== [tmp_5222[0] + tmp_5195[0], tmp_5222[1] + tmp_5195[1], tmp_5222[2] + tmp_5195[2]];
    signal tmp_5224[3] <== [tmp_5221[0] + tmp_5223[0], tmp_5221[1] + tmp_5223[1], tmp_5221[2] + tmp_5223[2]];
    signal tmp_5225[3] <== [tmp_5173[0] * 6052753985947965968, tmp_5173[1] * 6052753985947965968, tmp_5173[2] * 6052753985947965968];
    signal tmp_5226[3] <== [tmp_5225[0] + tmp_5195[0], tmp_5225[1] + tmp_5195[1], tmp_5225[2] + tmp_5195[2]];
    signal tmp_5227[3] <== [tmp_5224[0] + tmp_5226[0], tmp_5224[1] + tmp_5226[1], tmp_5224[2] + tmp_5226[2]];
    signal tmp_5228[3] <== [tmp_5176[0] * 13608362914817483670, tmp_5176[1] * 13608362914817483670, tmp_5176[2] * 13608362914817483670];
    signal tmp_5229[3] <== [tmp_5228[0] + tmp_5195[0], tmp_5228[1] + tmp_5195[1], tmp_5228[2] + tmp_5195[2]];
    signal tmp_5230[3] <== [tmp_5227[0] + tmp_5229[0], tmp_5227[1] + tmp_5229[1], tmp_5227[2] + tmp_5229[2]];
    signal tmp_5231[3] <== [tmp_5179[0] * 18163707672964630459, tmp_5179[1] * 18163707672964630459, tmp_5179[2] * 18163707672964630459];
    signal tmp_5232[3] <== [tmp_5231[0] + tmp_5195[0], tmp_5231[1] + tmp_5195[1], tmp_5231[2] + tmp_5195[2]];
    signal tmp_5233[3] <== [tmp_5230[0] + tmp_5232[0], tmp_5230[1] + tmp_5232[1], tmp_5230[2] + tmp_5232[2]];
    signal tmp_5234[3] <== [tmp_5182[0] * 14373610220374016704, tmp_5182[1] * 14373610220374016704, tmp_5182[2] * 14373610220374016704];
    signal tmp_5235[3] <== [tmp_5234[0] + tmp_5195[0], tmp_5234[1] + tmp_5195[1], tmp_5234[2] + tmp_5195[2]];
    signal tmp_5236[3] <== [tmp_5233[0] + tmp_5235[0], tmp_5233[1] + tmp_5235[1], tmp_5233[2] + tmp_5235[2]];
    signal tmp_5237[3] <== [tmp_5185[0] * 6226282807566121054, tmp_5185[1] * 6226282807566121054, tmp_5185[2] * 6226282807566121054];
    signal tmp_5238[3] <== [tmp_5237[0] + tmp_5195[0], tmp_5237[1] + tmp_5195[1], tmp_5237[2] + tmp_5195[2]];
    signal tmp_5239[3] <== [tmp_5236[0] + tmp_5238[0], tmp_5236[1] + tmp_5238[1], tmp_5236[2] + tmp_5238[2]];
    signal tmp_5240[3] <== [tmp_5188[0] * 3643354756180461803, tmp_5188[1] * 3643354756180461803, tmp_5188[2] * 3643354756180461803];
    signal tmp_5241[3] <== [tmp_5240[0] + tmp_5195[0], tmp_5240[1] + tmp_5195[1], tmp_5240[2] + tmp_5195[2]];
    signal tmp_5242[3] <== [tmp_5239[0] + tmp_5241[0], tmp_5239[1] + tmp_5241[1], tmp_5239[2] + tmp_5241[2]];
    signal tmp_5243[3] <== [tmp_5191[0] * 13046961313070095543, tmp_5191[1] * 13046961313070095543, tmp_5191[2] * 13046961313070095543];
    signal tmp_5244[3] <== [tmp_5243[0] + tmp_5195[0], tmp_5243[1] + tmp_5195[1], tmp_5243[2] + tmp_5195[2]];
    signal tmp_5245[3] <== [tmp_5242[0] + tmp_5244[0], tmp_5242[1] + tmp_5244[1], tmp_5242[2] + tmp_5244[2]];
    signal tmp_5246[3] <== [tmp_5194[0] * 8594143216561850811, tmp_5194[1] * 8594143216561850811, tmp_5194[2] * 8594143216561850811];
    signal tmp_5247[3] <== [tmp_5246[0] + tmp_5195[0], tmp_5246[1] + tmp_5195[1], tmp_5246[2] + tmp_5195[2]];
    signal tmp_5248[3] <== [tmp_5245[0] + tmp_5247[0], tmp_5245[1] + tmp_5247[1], tmp_5245[2] + tmp_5247[2]];
    signal tmp_5249[3] <== [tmp_5203[0] + tmp_5248[0], tmp_5203[1] + tmp_5248[1], tmp_5203[2] + tmp_5248[2]];
    signal tmp_5250[3] <== [evals[125][0] - tmp_5249[0], evals[125][1] - tmp_5249[1], evals[125][2] - tmp_5249[2]];
    signal tmp_5251[3] <== CMul()(evals[76], tmp_5250);
    signal tmp_5252[3] <== [tmp_5200[0] + tmp_5251[0], tmp_5200[1] + tmp_5251[1], tmp_5200[2] + tmp_5251[2]];
    signal tmp_5253[3] <== CMul()(challengeQ, tmp_5252);
    signal tmp_5254[3] <== [evals[125][0] + tmp_4448[0], evals[125][1] + tmp_4448[1], evals[125][2] + tmp_4448[2]];
    signal tmp_5255[3] <== CMul()(tmp_4452, tmp_5254);
    signal tmp_5256[3] <== [tmp_5255[0] * 16040574633112940480, tmp_5255[1] * 16040574633112940480, tmp_5255[2] * 16040574633112940480];
    signal tmp_5257[3] <== [tmp_5205[0] * 14263299814608977431, tmp_5205[1] * 14263299814608977431, tmp_5205[2] * 14263299814608977431];
    signal tmp_5258[3] <== [tmp_5257[0] + tmp_5248[0], tmp_5257[1] + tmp_5248[1], tmp_5257[2] + tmp_5248[2]];
    signal tmp_5259[3] <== [tmp_5255[0] + tmp_5258[0], tmp_5255[1] + tmp_5258[1], tmp_5255[2] + tmp_5258[2]];
    signal tmp_5260[3] <== [tmp_5208[0] * 770395855193680981, tmp_5208[1] * 770395855193680981, tmp_5208[2] * 770395855193680981];
    signal tmp_5261[3] <== [tmp_5260[0] + tmp_5248[0], tmp_5260[1] + tmp_5248[1], tmp_5260[2] + tmp_5248[2]];
    signal tmp_5262[3] <== [tmp_5259[0] + tmp_5261[0], tmp_5259[1] + tmp_5261[1], tmp_5259[2] + tmp_5261[2]];
    signal tmp_5263[3] <== [tmp_5211[0] * 3459277367440070515, tmp_5211[1] * 3459277367440070515, tmp_5211[2] * 3459277367440070515];
    signal tmp_5264[3] <== [tmp_5263[0] + tmp_5248[0], tmp_5263[1] + tmp_5248[1], tmp_5263[2] + tmp_5248[2]];
    signal tmp_5265[3] <== [tmp_5262[0] + tmp_5264[0], tmp_5262[1] + tmp_5264[1], tmp_5262[2] + tmp_5264[2]];
    signal tmp_5266[3] <== [tmp_5214[0] * 17087697094293314027, tmp_5214[1] * 17087697094293314027, tmp_5214[2] * 17087697094293314027];
    signal tmp_5267[3] <== [tmp_5266[0] + tmp_5248[0], tmp_5266[1] + tmp_5248[1], tmp_5266[2] + tmp_5248[2]];
    signal tmp_5268[3] <== [tmp_5265[0] + tmp_5267[0], tmp_5265[1] + tmp_5267[1], tmp_5265[2] + tmp_5267[2]];
    signal tmp_5269[3] <== [tmp_5217[0] * 6694380135428747348, tmp_5217[1] * 6694380135428747348, tmp_5217[2] * 6694380135428747348];
    signal tmp_5270[3] <== [tmp_5269[0] + tmp_5248[0], tmp_5269[1] + tmp_5248[1], tmp_5269[2] + tmp_5248[2]];
    signal tmp_5271[3] <== [tmp_5268[0] + tmp_5270[0], tmp_5268[1] + tmp_5270[1], tmp_5268[2] + tmp_5270[2]];
    signal tmp_5272[3] <== [tmp_5220[0] * 2034408310088972836, tmp_5220[1] * 2034408310088972836, tmp_5220[2] * 2034408310088972836];
    signal tmp_5273[3] <== [tmp_5272[0] + tmp_5248[0], tmp_5272[1] + tmp_5248[1], tmp_5272[2] + tmp_5248[2]];
    signal tmp_5274[3] <== [tmp_5271[0] + tmp_5273[0], tmp_5271[1] + tmp_5273[1], tmp_5271[2] + tmp_5273[2]];
    signal tmp_5275[3] <== [tmp_5223[0] * 3434575637390274478, tmp_5223[1] * 3434575637390274478, tmp_5223[2] * 3434575637390274478];
    signal tmp_5276[3] <== [tmp_5275[0] + tmp_5248[0], tmp_5275[1] + tmp_5248[1], tmp_5275[2] + tmp_5248[2]];
    signal tmp_5277[3] <== [tmp_5274[0] + tmp_5276[0], tmp_5274[1] + tmp_5276[1], tmp_5274[2] + tmp_5276[2]];
    signal tmp_5278[3] <== [tmp_5226[0] * 6052753985947965968, tmp_5226[1] * 6052753985947965968, tmp_5226[2] * 6052753985947965968];
    signal tmp_5279[3] <== [tmp_5278[0] + tmp_5248[0], tmp_5278[1] + tmp_5248[1], tmp_5278[2] + tmp_5248[2]];
    signal tmp_5280[3] <== [tmp_5277[0] + tmp_5279[0], tmp_5277[1] + tmp_5279[1], tmp_5277[2] + tmp_5279[2]];
    signal tmp_5281[3] <== [tmp_5229[0] * 13608362914817483670, tmp_5229[1] * 13608362914817483670, tmp_5229[2] * 13608362914817483670];
    signal tmp_5282[3] <== [tmp_5281[0] + tmp_5248[0], tmp_5281[1] + tmp_5248[1], tmp_5281[2] + tmp_5248[2]];
    signal tmp_5283[3] <== [tmp_5280[0] + tmp_5282[0], tmp_5280[1] + tmp_5282[1], tmp_5280[2] + tmp_5282[2]];
    signal tmp_5284[3] <== [tmp_5232[0] * 18163707672964630459, tmp_5232[1] * 18163707672964630459, tmp_5232[2] * 18163707672964630459];
    signal tmp_5285[3] <== [tmp_5284[0] + tmp_5248[0], tmp_5284[1] + tmp_5248[1], tmp_5284[2] + tmp_5248[2]];
    signal tmp_5286[3] <== [tmp_5283[0] + tmp_5285[0], tmp_5283[1] + tmp_5285[1], tmp_5283[2] + tmp_5285[2]];
    signal tmp_5287[3] <== [tmp_5235[0] * 14373610220374016704, tmp_5235[1] * 14373610220374016704, tmp_5235[2] * 14373610220374016704];
    signal tmp_5288[3] <== [tmp_5287[0] + tmp_5248[0], tmp_5287[1] + tmp_5248[1], tmp_5287[2] + tmp_5248[2]];
    signal tmp_5289[3] <== [tmp_5286[0] + tmp_5288[0], tmp_5286[1] + tmp_5288[1], tmp_5286[2] + tmp_5288[2]];
    signal tmp_5290[3] <== [tmp_5238[0] * 6226282807566121054, tmp_5238[1] * 6226282807566121054, tmp_5238[2] * 6226282807566121054];
    signal tmp_5291[3] <== [tmp_5290[0] + tmp_5248[0], tmp_5290[1] + tmp_5248[1], tmp_5290[2] + tmp_5248[2]];
    signal tmp_5292[3] <== [tmp_5289[0] + tmp_5291[0], tmp_5289[1] + tmp_5291[1], tmp_5289[2] + tmp_5291[2]];
    signal tmp_5293[3] <== [tmp_5241[0] * 3643354756180461803, tmp_5241[1] * 3643354756180461803, tmp_5241[2] * 3643354756180461803];
    signal tmp_5294[3] <== [tmp_5293[0] + tmp_5248[0], tmp_5293[1] + tmp_5248[1], tmp_5293[2] + tmp_5248[2]];
    signal tmp_5295[3] <== [tmp_5292[0] + tmp_5294[0], tmp_5292[1] + tmp_5294[1], tmp_5292[2] + tmp_5294[2]];
    signal tmp_5296[3] <== [tmp_5244[0] * 13046961313070095543, tmp_5244[1] * 13046961313070095543, tmp_5244[2] * 13046961313070095543];
    signal tmp_5297[3] <== [tmp_5296[0] + tmp_5248[0], tmp_5296[1] + tmp_5248[1], tmp_5296[2] + tmp_5248[2]];
    signal tmp_5298[3] <== [tmp_5295[0] + tmp_5297[0], tmp_5295[1] + tmp_5297[1], tmp_5295[2] + tmp_5297[2]];
    signal tmp_5299[3] <== [tmp_5247[0] * 8594143216561850811, tmp_5247[1] * 8594143216561850811, tmp_5247[2] * 8594143216561850811];
    signal tmp_5300[3] <== [tmp_5299[0] + tmp_5248[0], tmp_5299[1] + tmp_5248[1], tmp_5299[2] + tmp_5248[2]];
    signal tmp_5301[3] <== [tmp_5298[0] + tmp_5300[0], tmp_5298[1] + tmp_5300[1], tmp_5298[2] + tmp_5300[2]];
    signal tmp_5302[3] <== [tmp_5256[0] + tmp_5301[0], tmp_5256[1] + tmp_5301[1], tmp_5256[2] + tmp_5301[2]];
    signal tmp_5303[3] <== [evals[126][0] - tmp_5302[0], evals[126][1] - tmp_5302[1], evals[126][2] - tmp_5302[2]];
    signal tmp_5304[3] <== CMul()(evals[76], tmp_5303);
    signal tmp_5305[3] <== [tmp_5253[0] + tmp_5304[0], tmp_5253[1] + tmp_5304[1], tmp_5253[2] + tmp_5304[2]];
    signal tmp_5306[3] <== CMul()(challengeQ, tmp_5305);
    signal tmp_5307[3] <== [evals[126][0] + tmp_4577[0], evals[126][1] + tmp_4577[1], evals[126][2] + tmp_4577[2]];
    signal tmp_5308[3] <== CMul()(tmp_4581, tmp_5307);
    signal tmp_5309[3] <== [tmp_5308[0] * 16040574633112940480, tmp_5308[1] * 16040574633112940480, tmp_5308[2] * 16040574633112940480];
    signal tmp_5310[3] <== [tmp_5258[0] * 14263299814608977431, tmp_5258[1] * 14263299814608977431, tmp_5258[2] * 14263299814608977431];
    signal tmp_5311[3] <== [tmp_5310[0] + tmp_5301[0], tmp_5310[1] + tmp_5301[1], tmp_5310[2] + tmp_5301[2]];
    signal tmp_5312[3] <== [tmp_5308[0] + tmp_5311[0], tmp_5308[1] + tmp_5311[1], tmp_5308[2] + tmp_5311[2]];
    signal tmp_5313[3] <== [tmp_5261[0] * 770395855193680981, tmp_5261[1] * 770395855193680981, tmp_5261[2] * 770395855193680981];
    signal tmp_5314[3] <== [tmp_5313[0] + tmp_5301[0], tmp_5313[1] + tmp_5301[1], tmp_5313[2] + tmp_5301[2]];
    signal tmp_5315[3] <== [tmp_5312[0] + tmp_5314[0], tmp_5312[1] + tmp_5314[1], tmp_5312[2] + tmp_5314[2]];
    signal tmp_5316[3] <== [tmp_5264[0] * 3459277367440070515, tmp_5264[1] * 3459277367440070515, tmp_5264[2] * 3459277367440070515];
    signal tmp_5317[3] <== [tmp_5316[0] + tmp_5301[0], tmp_5316[1] + tmp_5301[1], tmp_5316[2] + tmp_5301[2]];
    signal tmp_5318[3] <== [tmp_5315[0] + tmp_5317[0], tmp_5315[1] + tmp_5317[1], tmp_5315[2] + tmp_5317[2]];
    signal tmp_5319[3] <== [tmp_5267[0] * 17087697094293314027, tmp_5267[1] * 17087697094293314027, tmp_5267[2] * 17087697094293314027];
    signal tmp_5320[3] <== [tmp_5319[0] + tmp_5301[0], tmp_5319[1] + tmp_5301[1], tmp_5319[2] + tmp_5301[2]];
    signal tmp_5321[3] <== [tmp_5318[0] + tmp_5320[0], tmp_5318[1] + tmp_5320[1], tmp_5318[2] + tmp_5320[2]];
    signal tmp_5322[3] <== [tmp_5270[0] * 6694380135428747348, tmp_5270[1] * 6694380135428747348, tmp_5270[2] * 6694380135428747348];
    signal tmp_5323[3] <== [tmp_5322[0] + tmp_5301[0], tmp_5322[1] + tmp_5301[1], tmp_5322[2] + tmp_5301[2]];
    signal tmp_5324[3] <== [tmp_5321[0] + tmp_5323[0], tmp_5321[1] + tmp_5323[1], tmp_5321[2] + tmp_5323[2]];
    signal tmp_5325[3] <== [tmp_5273[0] * 2034408310088972836, tmp_5273[1] * 2034408310088972836, tmp_5273[2] * 2034408310088972836];
    signal tmp_5326[3] <== [tmp_5325[0] + tmp_5301[0], tmp_5325[1] + tmp_5301[1], tmp_5325[2] + tmp_5301[2]];
    signal tmp_5327[3] <== [tmp_5324[0] + tmp_5326[0], tmp_5324[1] + tmp_5326[1], tmp_5324[2] + tmp_5326[2]];
    signal tmp_5328[3] <== [tmp_5276[0] * 3434575637390274478, tmp_5276[1] * 3434575637390274478, tmp_5276[2] * 3434575637390274478];
    signal tmp_5329[3] <== [tmp_5328[0] + tmp_5301[0], tmp_5328[1] + tmp_5301[1], tmp_5328[2] + tmp_5301[2]];
    signal tmp_5330[3] <== [tmp_5327[0] + tmp_5329[0], tmp_5327[1] + tmp_5329[1], tmp_5327[2] + tmp_5329[2]];
    signal tmp_5331[3] <== [tmp_5279[0] * 6052753985947965968, tmp_5279[1] * 6052753985947965968, tmp_5279[2] * 6052753985947965968];
    signal tmp_5332[3] <== [tmp_5331[0] + tmp_5301[0], tmp_5331[1] + tmp_5301[1], tmp_5331[2] + tmp_5301[2]];
    signal tmp_5333[3] <== [tmp_5330[0] + tmp_5332[0], tmp_5330[1] + tmp_5332[1], tmp_5330[2] + tmp_5332[2]];
    signal tmp_5334[3] <== [tmp_5282[0] * 13608362914817483670, tmp_5282[1] * 13608362914817483670, tmp_5282[2] * 13608362914817483670];
    signal tmp_5335[3] <== [tmp_5334[0] + tmp_5301[0], tmp_5334[1] + tmp_5301[1], tmp_5334[2] + tmp_5301[2]];
    signal tmp_5336[3] <== [tmp_5333[0] + tmp_5335[0], tmp_5333[1] + tmp_5335[1], tmp_5333[2] + tmp_5335[2]];
    signal tmp_5337[3] <== [tmp_5285[0] * 18163707672964630459, tmp_5285[1] * 18163707672964630459, tmp_5285[2] * 18163707672964630459];
    signal tmp_5338[3] <== [tmp_5337[0] + tmp_5301[0], tmp_5337[1] + tmp_5301[1], tmp_5337[2] + tmp_5301[2]];
    signal tmp_5339[3] <== [tmp_5336[0] + tmp_5338[0], tmp_5336[1] + tmp_5338[1], tmp_5336[2] + tmp_5338[2]];
    signal tmp_5340[3] <== [tmp_5288[0] * 14373610220374016704, tmp_5288[1] * 14373610220374016704, tmp_5288[2] * 14373610220374016704];
    signal tmp_5341[3] <== [tmp_5340[0] + tmp_5301[0], tmp_5340[1] + tmp_5301[1], tmp_5340[2] + tmp_5301[2]];
    signal tmp_5342[3] <== [tmp_5339[0] + tmp_5341[0], tmp_5339[1] + tmp_5341[1], tmp_5339[2] + tmp_5341[2]];
    signal tmp_5343[3] <== [tmp_5291[0] * 6226282807566121054, tmp_5291[1] * 6226282807566121054, tmp_5291[2] * 6226282807566121054];
    signal tmp_5344[3] <== [tmp_5343[0] + tmp_5301[0], tmp_5343[1] + tmp_5301[1], tmp_5343[2] + tmp_5301[2]];
    signal tmp_5345[3] <== [tmp_5342[0] + tmp_5344[0], tmp_5342[1] + tmp_5344[1], tmp_5342[2] + tmp_5344[2]];
    signal tmp_5346[3] <== [tmp_5294[0] * 3643354756180461803, tmp_5294[1] * 3643354756180461803, tmp_5294[2] * 3643354756180461803];
    signal tmp_5347[3] <== [tmp_5346[0] + tmp_5301[0], tmp_5346[1] + tmp_5301[1], tmp_5346[2] + tmp_5301[2]];
    signal tmp_5348[3] <== [tmp_5345[0] + tmp_5347[0], tmp_5345[1] + tmp_5347[1], tmp_5345[2] + tmp_5347[2]];
    signal tmp_5349[3] <== [tmp_5297[0] * 13046961313070095543, tmp_5297[1] * 13046961313070095543, tmp_5297[2] * 13046961313070095543];
    signal tmp_5350[3] <== [tmp_5349[0] + tmp_5301[0], tmp_5349[1] + tmp_5301[1], tmp_5349[2] + tmp_5301[2]];
    signal tmp_5351[3] <== [tmp_5348[0] + tmp_5350[0], tmp_5348[1] + tmp_5350[1], tmp_5348[2] + tmp_5350[2]];
    signal tmp_5352[3] <== [tmp_5300[0] * 8594143216561850811, tmp_5300[1] * 8594143216561850811, tmp_5300[2] * 8594143216561850811];
    signal tmp_5353[3] <== [tmp_5352[0] + tmp_5301[0], tmp_5352[1] + tmp_5301[1], tmp_5352[2] + tmp_5301[2]];
    signal tmp_5354[3] <== [tmp_5351[0] + tmp_5353[0], tmp_5351[1] + tmp_5353[1], tmp_5351[2] + tmp_5353[2]];
    signal tmp_5355[3] <== [tmp_5309[0] + tmp_5354[0], tmp_5309[1] + tmp_5354[1], tmp_5309[2] + tmp_5354[2]];
    signal tmp_5356[3] <== [evals[127][0] - tmp_5355[0], evals[127][1] - tmp_5355[1], evals[127][2] - tmp_5355[2]];
    signal tmp_5357[3] <== CMul()(evals[76], tmp_5356);
    signal tmp_5358[3] <== [tmp_5306[0] + tmp_5357[0], tmp_5306[1] + tmp_5357[1], tmp_5306[2] + tmp_5357[2]];
    signal tmp_5359[3] <== CMul()(challengeQ, tmp_5358);
    signal tmp_5360[3] <== [evals[127][0] + tmp_4600[0], evals[127][1] + tmp_4600[1], evals[127][2] + tmp_4600[2]];
    signal tmp_5361[3] <== CMul()(tmp_4604, tmp_5360);
    signal tmp_5362[3] <== [tmp_5361[0] * 16040574633112940480, tmp_5361[1] * 16040574633112940480, tmp_5361[2] * 16040574633112940480];
    signal tmp_5363[3] <== [tmp_5311[0] * 14263299814608977431, tmp_5311[1] * 14263299814608977431, tmp_5311[2] * 14263299814608977431];
    signal tmp_5364[3] <== [tmp_5363[0] + tmp_5354[0], tmp_5363[1] + tmp_5354[1], tmp_5363[2] + tmp_5354[2]];
    signal tmp_5365[3] <== [tmp_5361[0] + tmp_5364[0], tmp_5361[1] + tmp_5364[1], tmp_5361[2] + tmp_5364[2]];
    signal tmp_5366[3] <== [tmp_5314[0] * 770395855193680981, tmp_5314[1] * 770395855193680981, tmp_5314[2] * 770395855193680981];
    signal tmp_5367[3] <== [tmp_5366[0] + tmp_5354[0], tmp_5366[1] + tmp_5354[1], tmp_5366[2] + tmp_5354[2]];
    signal tmp_5368[3] <== [tmp_5365[0] + tmp_5367[0], tmp_5365[1] + tmp_5367[1], tmp_5365[2] + tmp_5367[2]];
    signal tmp_5369[3] <== [tmp_5317[0] * 3459277367440070515, tmp_5317[1] * 3459277367440070515, tmp_5317[2] * 3459277367440070515];
    signal tmp_5370[3] <== [tmp_5369[0] + tmp_5354[0], tmp_5369[1] + tmp_5354[1], tmp_5369[2] + tmp_5354[2]];
    signal tmp_5371[3] <== [tmp_5368[0] + tmp_5370[0], tmp_5368[1] + tmp_5370[1], tmp_5368[2] + tmp_5370[2]];
    signal tmp_5372[3] <== [tmp_5320[0] * 17087697094293314027, tmp_5320[1] * 17087697094293314027, tmp_5320[2] * 17087697094293314027];
    signal tmp_5373[3] <== [tmp_5372[0] + tmp_5354[0], tmp_5372[1] + tmp_5354[1], tmp_5372[2] + tmp_5354[2]];
    signal tmp_5374[3] <== [tmp_5371[0] + tmp_5373[0], tmp_5371[1] + tmp_5373[1], tmp_5371[2] + tmp_5373[2]];
    signal tmp_5375[3] <== [tmp_5323[0] * 6694380135428747348, tmp_5323[1] * 6694380135428747348, tmp_5323[2] * 6694380135428747348];
    signal tmp_5376[3] <== [tmp_5375[0] + tmp_5354[0], tmp_5375[1] + tmp_5354[1], tmp_5375[2] + tmp_5354[2]];
    signal tmp_5377[3] <== [tmp_5374[0] + tmp_5376[0], tmp_5374[1] + tmp_5376[1], tmp_5374[2] + tmp_5376[2]];
    signal tmp_5378[3] <== [tmp_5326[0] * 2034408310088972836, tmp_5326[1] * 2034408310088972836, tmp_5326[2] * 2034408310088972836];
    signal tmp_5379[3] <== [tmp_5378[0] + tmp_5354[0], tmp_5378[1] + tmp_5354[1], tmp_5378[2] + tmp_5354[2]];
    signal tmp_5380[3] <== [tmp_5377[0] + tmp_5379[0], tmp_5377[1] + tmp_5379[1], tmp_5377[2] + tmp_5379[2]];
    signal tmp_5381[3] <== [tmp_5329[0] * 3434575637390274478, tmp_5329[1] * 3434575637390274478, tmp_5329[2] * 3434575637390274478];
    signal tmp_5382[3] <== [tmp_5381[0] + tmp_5354[0], tmp_5381[1] + tmp_5354[1], tmp_5381[2] + tmp_5354[2]];
    signal tmp_5383[3] <== [tmp_5380[0] + tmp_5382[0], tmp_5380[1] + tmp_5382[1], tmp_5380[2] + tmp_5382[2]];
    signal tmp_5384[3] <== [tmp_5332[0] * 6052753985947965968, tmp_5332[1] * 6052753985947965968, tmp_5332[2] * 6052753985947965968];
    signal tmp_5385[3] <== [tmp_5384[0] + tmp_5354[0], tmp_5384[1] + tmp_5354[1], tmp_5384[2] + tmp_5354[2]];
    signal tmp_5386[3] <== [tmp_5383[0] + tmp_5385[0], tmp_5383[1] + tmp_5385[1], tmp_5383[2] + tmp_5385[2]];
    signal tmp_5387[3] <== [tmp_5335[0] * 13608362914817483670, tmp_5335[1] * 13608362914817483670, tmp_5335[2] * 13608362914817483670];
    signal tmp_5388[3] <== [tmp_5387[0] + tmp_5354[0], tmp_5387[1] + tmp_5354[1], tmp_5387[2] + tmp_5354[2]];
    signal tmp_5389[3] <== [tmp_5386[0] + tmp_5388[0], tmp_5386[1] + tmp_5388[1], tmp_5386[2] + tmp_5388[2]];
    signal tmp_5390[3] <== [tmp_5338[0] * 18163707672964630459, tmp_5338[1] * 18163707672964630459, tmp_5338[2] * 18163707672964630459];
    signal tmp_5391[3] <== [tmp_5390[0] + tmp_5354[0], tmp_5390[1] + tmp_5354[1], tmp_5390[2] + tmp_5354[2]];
    signal tmp_5392[3] <== [tmp_5389[0] + tmp_5391[0], tmp_5389[1] + tmp_5391[1], tmp_5389[2] + tmp_5391[2]];
    signal tmp_5393[3] <== [tmp_5341[0] * 14373610220374016704, tmp_5341[1] * 14373610220374016704, tmp_5341[2] * 14373610220374016704];
    signal tmp_5394[3] <== [tmp_5393[0] + tmp_5354[0], tmp_5393[1] + tmp_5354[1], tmp_5393[2] + tmp_5354[2]];
    signal tmp_5395[3] <== [tmp_5392[0] + tmp_5394[0], tmp_5392[1] + tmp_5394[1], tmp_5392[2] + tmp_5394[2]];
    signal tmp_5396[3] <== [tmp_5344[0] * 6226282807566121054, tmp_5344[1] * 6226282807566121054, tmp_5344[2] * 6226282807566121054];
    signal tmp_5397[3] <== [tmp_5396[0] + tmp_5354[0], tmp_5396[1] + tmp_5354[1], tmp_5396[2] + tmp_5354[2]];
    signal tmp_5398[3] <== [tmp_5395[0] + tmp_5397[0], tmp_5395[1] + tmp_5397[1], tmp_5395[2] + tmp_5397[2]];
    signal tmp_5399[3] <== [tmp_5347[0] * 3643354756180461803, tmp_5347[1] * 3643354756180461803, tmp_5347[2] * 3643354756180461803];
    signal tmp_5400[3] <== [tmp_5399[0] + tmp_5354[0], tmp_5399[1] + tmp_5354[1], tmp_5399[2] + tmp_5354[2]];
    signal tmp_5401[3] <== [tmp_5398[0] + tmp_5400[0], tmp_5398[1] + tmp_5400[1], tmp_5398[2] + tmp_5400[2]];
    signal tmp_5402[3] <== [tmp_5350[0] * 13046961313070095543, tmp_5350[1] * 13046961313070095543, tmp_5350[2] * 13046961313070095543];
    signal tmp_5403[3] <== [tmp_5402[0] + tmp_5354[0], tmp_5402[1] + tmp_5354[1], tmp_5402[2] + tmp_5354[2]];
    signal tmp_5404[3] <== [tmp_5401[0] + tmp_5403[0], tmp_5401[1] + tmp_5403[1], tmp_5401[2] + tmp_5403[2]];
    signal tmp_5405[3] <== [tmp_5353[0] * 8594143216561850811, tmp_5353[1] * 8594143216561850811, tmp_5353[2] * 8594143216561850811];
    signal tmp_5406[3] <== [tmp_5405[0] + tmp_5354[0], tmp_5405[1] + tmp_5354[1], tmp_5405[2] + tmp_5354[2]];
    signal tmp_5407[3] <== [tmp_5404[0] + tmp_5406[0], tmp_5404[1] + tmp_5406[1], tmp_5404[2] + tmp_5406[2]];
    signal tmp_5408[3] <== [tmp_5362[0] + tmp_5407[0], tmp_5362[1] + tmp_5407[1], tmp_5362[2] + tmp_5407[2]];
    signal tmp_5409[3] <== [evals[128][0] - tmp_5408[0], evals[128][1] - tmp_5408[1], evals[128][2] - tmp_5408[2]];
    signal tmp_5410[3] <== CMul()(evals[76], tmp_5409);
    signal tmp_5411[3] <== [tmp_5359[0] + tmp_5410[0], tmp_5359[1] + tmp_5410[1], tmp_5359[2] + tmp_5410[2]];
    signal tmp_5412[3] <== CMul()(challengeQ, tmp_5411);
    signal tmp_5413[3] <== [evals[128][0] + tmp_4629[0], evals[128][1] + tmp_4629[1], evals[128][2] + tmp_4629[2]];
    signal tmp_5414[3] <== CMul()(tmp_4633, tmp_5413);
    signal tmp_5415[3] <== [tmp_5414[0] * 16040574633112940480, tmp_5414[1] * 16040574633112940480, tmp_5414[2] * 16040574633112940480];
    signal tmp_5416[3] <== [tmp_5364[0] * 14263299814608977431, tmp_5364[1] * 14263299814608977431, tmp_5364[2] * 14263299814608977431];
    signal tmp_5417[3] <== [tmp_5416[0] + tmp_5407[0], tmp_5416[1] + tmp_5407[1], tmp_5416[2] + tmp_5407[2]];
    signal tmp_5418[3] <== [tmp_5414[0] + tmp_5417[0], tmp_5414[1] + tmp_5417[1], tmp_5414[2] + tmp_5417[2]];
    signal tmp_5419[3] <== [tmp_5367[0] * 770395855193680981, tmp_5367[1] * 770395855193680981, tmp_5367[2] * 770395855193680981];
    signal tmp_5420[3] <== [tmp_5419[0] + tmp_5407[0], tmp_5419[1] + tmp_5407[1], tmp_5419[2] + tmp_5407[2]];
    signal tmp_5421[3] <== [tmp_5418[0] + tmp_5420[0], tmp_5418[1] + tmp_5420[1], tmp_5418[2] + tmp_5420[2]];
    signal tmp_5422[3] <== [tmp_5370[0] * 3459277367440070515, tmp_5370[1] * 3459277367440070515, tmp_5370[2] * 3459277367440070515];
    signal tmp_5423[3] <== [tmp_5422[0] + tmp_5407[0], tmp_5422[1] + tmp_5407[1], tmp_5422[2] + tmp_5407[2]];
    signal tmp_5424[3] <== [tmp_5421[0] + tmp_5423[0], tmp_5421[1] + tmp_5423[1], tmp_5421[2] + tmp_5423[2]];
    signal tmp_5425[3] <== [tmp_5373[0] * 17087697094293314027, tmp_5373[1] * 17087697094293314027, tmp_5373[2] * 17087697094293314027];
    signal tmp_5426[3] <== [tmp_5425[0] + tmp_5407[0], tmp_5425[1] + tmp_5407[1], tmp_5425[2] + tmp_5407[2]];
    signal tmp_5427[3] <== [tmp_5424[0] + tmp_5426[0], tmp_5424[1] + tmp_5426[1], tmp_5424[2] + tmp_5426[2]];
    signal tmp_5428[3] <== [tmp_5376[0] * 6694380135428747348, tmp_5376[1] * 6694380135428747348, tmp_5376[2] * 6694380135428747348];
    signal tmp_5429[3] <== [tmp_5428[0] + tmp_5407[0], tmp_5428[1] + tmp_5407[1], tmp_5428[2] + tmp_5407[2]];
    signal tmp_5430[3] <== [tmp_5427[0] + tmp_5429[0], tmp_5427[1] + tmp_5429[1], tmp_5427[2] + tmp_5429[2]];
    signal tmp_5431[3] <== [tmp_5379[0] * 2034408310088972836, tmp_5379[1] * 2034408310088972836, tmp_5379[2] * 2034408310088972836];
    signal tmp_5432[3] <== [tmp_5431[0] + tmp_5407[0], tmp_5431[1] + tmp_5407[1], tmp_5431[2] + tmp_5407[2]];
    signal tmp_5433[3] <== [tmp_5430[0] + tmp_5432[0], tmp_5430[1] + tmp_5432[1], tmp_5430[2] + tmp_5432[2]];
    signal tmp_5434[3] <== [tmp_5382[0] * 3434575637390274478, tmp_5382[1] * 3434575637390274478, tmp_5382[2] * 3434575637390274478];
    signal tmp_5435[3] <== [tmp_5434[0] + tmp_5407[0], tmp_5434[1] + tmp_5407[1], tmp_5434[2] + tmp_5407[2]];
    signal tmp_5436[3] <== [tmp_5433[0] + tmp_5435[0], tmp_5433[1] + tmp_5435[1], tmp_5433[2] + tmp_5435[2]];
    signal tmp_5437[3] <== [tmp_5385[0] * 6052753985947965968, tmp_5385[1] * 6052753985947965968, tmp_5385[2] * 6052753985947965968];
    signal tmp_5438[3] <== [tmp_5437[0] + tmp_5407[0], tmp_5437[1] + tmp_5407[1], tmp_5437[2] + tmp_5407[2]];
    signal tmp_5439[3] <== [tmp_5436[0] + tmp_5438[0], tmp_5436[1] + tmp_5438[1], tmp_5436[2] + tmp_5438[2]];
    signal tmp_5440[3] <== [tmp_5388[0] * 13608362914817483670, tmp_5388[1] * 13608362914817483670, tmp_5388[2] * 13608362914817483670];
    signal tmp_5441[3] <== [tmp_5440[0] + tmp_5407[0], tmp_5440[1] + tmp_5407[1], tmp_5440[2] + tmp_5407[2]];
    signal tmp_5442[3] <== [tmp_5439[0] + tmp_5441[0], tmp_5439[1] + tmp_5441[1], tmp_5439[2] + tmp_5441[2]];
    signal tmp_5443[3] <== [tmp_5391[0] * 18163707672964630459, tmp_5391[1] * 18163707672964630459, tmp_5391[2] * 18163707672964630459];
    signal tmp_5444[3] <== [tmp_5443[0] + tmp_5407[0], tmp_5443[1] + tmp_5407[1], tmp_5443[2] + tmp_5407[2]];
    signal tmp_5445[3] <== [tmp_5442[0] + tmp_5444[0], tmp_5442[1] + tmp_5444[1], tmp_5442[2] + tmp_5444[2]];
    signal tmp_5446[3] <== [tmp_5394[0] * 14373610220374016704, tmp_5394[1] * 14373610220374016704, tmp_5394[2] * 14373610220374016704];
    signal tmp_5447[3] <== [tmp_5446[0] + tmp_5407[0], tmp_5446[1] + tmp_5407[1], tmp_5446[2] + tmp_5407[2]];
    signal tmp_5448[3] <== [tmp_5445[0] + tmp_5447[0], tmp_5445[1] + tmp_5447[1], tmp_5445[2] + tmp_5447[2]];
    signal tmp_5449[3] <== [tmp_5397[0] * 6226282807566121054, tmp_5397[1] * 6226282807566121054, tmp_5397[2] * 6226282807566121054];
    signal tmp_5450[3] <== [tmp_5449[0] + tmp_5407[0], tmp_5449[1] + tmp_5407[1], tmp_5449[2] + tmp_5407[2]];
    signal tmp_5451[3] <== [tmp_5448[0] + tmp_5450[0], tmp_5448[1] + tmp_5450[1], tmp_5448[2] + tmp_5450[2]];
    signal tmp_5452[3] <== [tmp_5400[0] * 3643354756180461803, tmp_5400[1] * 3643354756180461803, tmp_5400[2] * 3643354756180461803];
    signal tmp_5453[3] <== [tmp_5452[0] + tmp_5407[0], tmp_5452[1] + tmp_5407[1], tmp_5452[2] + tmp_5407[2]];
    signal tmp_5454[3] <== [tmp_5451[0] + tmp_5453[0], tmp_5451[1] + tmp_5453[1], tmp_5451[2] + tmp_5453[2]];
    signal tmp_5455[3] <== [tmp_5403[0] * 13046961313070095543, tmp_5403[1] * 13046961313070095543, tmp_5403[2] * 13046961313070095543];
    signal tmp_5456[3] <== [tmp_5455[0] + tmp_5407[0], tmp_5455[1] + tmp_5407[1], tmp_5455[2] + tmp_5407[2]];
    signal tmp_5457[3] <== [tmp_5454[0] + tmp_5456[0], tmp_5454[1] + tmp_5456[1], tmp_5454[2] + tmp_5456[2]];
    signal tmp_5458[3] <== [tmp_5406[0] * 8594143216561850811, tmp_5406[1] * 8594143216561850811, tmp_5406[2] * 8594143216561850811];
    signal tmp_5459[3] <== [tmp_5458[0] + tmp_5407[0], tmp_5458[1] + tmp_5407[1], tmp_5458[2] + tmp_5407[2]];
    signal tmp_5460[3] <== [tmp_5457[0] + tmp_5459[0], tmp_5457[1] + tmp_5459[1], tmp_5457[2] + tmp_5459[2]];
    signal tmp_5461[3] <== [tmp_5415[0] + tmp_5460[0], tmp_5415[1] + tmp_5460[1], tmp_5415[2] + tmp_5460[2]];
    signal tmp_5462[3] <== [evals[129][0] - tmp_5461[0], evals[129][1] - tmp_5461[1], evals[129][2] - tmp_5461[2]];
    signal tmp_5463[3] <== CMul()(evals[76], tmp_5462);
    signal tmp_5464[3] <== [tmp_5412[0] + tmp_5463[0], tmp_5412[1] + tmp_5463[1], tmp_5412[2] + tmp_5463[2]];
    signal tmp_5465[3] <== CMul()(challengeQ, tmp_5464);
    signal tmp_5466[3] <== [evals[129][0] + tmp_4553[0], evals[129][1] + tmp_4553[1], evals[129][2] + tmp_4553[2]];
    signal tmp_5467[3] <== CMul()(tmp_4557, tmp_5466);
    signal tmp_5468[3] <== [tmp_5467[0] * 16040574633112940480, tmp_5467[1] * 16040574633112940480, tmp_5467[2] * 16040574633112940480];
    signal tmp_5469[3] <== [tmp_5417[0] * 14263299814608977431, tmp_5417[1] * 14263299814608977431, tmp_5417[2] * 14263299814608977431];
    signal tmp_5470[3] <== [tmp_5469[0] + tmp_5460[0], tmp_5469[1] + tmp_5460[1], tmp_5469[2] + tmp_5460[2]];
    signal tmp_5471[3] <== [tmp_5467[0] + tmp_5470[0], tmp_5467[1] + tmp_5470[1], tmp_5467[2] + tmp_5470[2]];
    signal tmp_5472[3] <== [tmp_5420[0] * 770395855193680981, tmp_5420[1] * 770395855193680981, tmp_5420[2] * 770395855193680981];
    signal tmp_5473[3] <== [tmp_5472[0] + tmp_5460[0], tmp_5472[1] + tmp_5460[1], tmp_5472[2] + tmp_5460[2]];
    signal tmp_5474[3] <== [tmp_5471[0] + tmp_5473[0], tmp_5471[1] + tmp_5473[1], tmp_5471[2] + tmp_5473[2]];
    signal tmp_5475[3] <== [tmp_5423[0] * 3459277367440070515, tmp_5423[1] * 3459277367440070515, tmp_5423[2] * 3459277367440070515];
    signal tmp_5476[3] <== [tmp_5475[0] + tmp_5460[0], tmp_5475[1] + tmp_5460[1], tmp_5475[2] + tmp_5460[2]];
    signal tmp_5477[3] <== [tmp_5474[0] + tmp_5476[0], tmp_5474[1] + tmp_5476[1], tmp_5474[2] + tmp_5476[2]];
    signal tmp_5478[3] <== [tmp_5426[0] * 17087697094293314027, tmp_5426[1] * 17087697094293314027, tmp_5426[2] * 17087697094293314027];
    signal tmp_5479[3] <== [tmp_5478[0] + tmp_5460[0], tmp_5478[1] + tmp_5460[1], tmp_5478[2] + tmp_5460[2]];
    signal tmp_5480[3] <== [tmp_5477[0] + tmp_5479[0], tmp_5477[1] + tmp_5479[1], tmp_5477[2] + tmp_5479[2]];
    signal tmp_5481[3] <== [tmp_5429[0] * 6694380135428747348, tmp_5429[1] * 6694380135428747348, tmp_5429[2] * 6694380135428747348];
    signal tmp_5482[3] <== [tmp_5481[0] + tmp_5460[0], tmp_5481[1] + tmp_5460[1], tmp_5481[2] + tmp_5460[2]];
    signal tmp_5483[3] <== [tmp_5480[0] + tmp_5482[0], tmp_5480[1] + tmp_5482[1], tmp_5480[2] + tmp_5482[2]];
    signal tmp_5484[3] <== [tmp_5432[0] * 2034408310088972836, tmp_5432[1] * 2034408310088972836, tmp_5432[2] * 2034408310088972836];
    signal tmp_5485[3] <== [tmp_5484[0] + tmp_5460[0], tmp_5484[1] + tmp_5460[1], tmp_5484[2] + tmp_5460[2]];
    signal tmp_5486[3] <== [tmp_5483[0] + tmp_5485[0], tmp_5483[1] + tmp_5485[1], tmp_5483[2] + tmp_5485[2]];
    signal tmp_5487[3] <== [tmp_5435[0] * 3434575637390274478, tmp_5435[1] * 3434575637390274478, tmp_5435[2] * 3434575637390274478];
    signal tmp_5488[3] <== [tmp_5487[0] + tmp_5460[0], tmp_5487[1] + tmp_5460[1], tmp_5487[2] + tmp_5460[2]];
    signal tmp_5489[3] <== [tmp_5486[0] + tmp_5488[0], tmp_5486[1] + tmp_5488[1], tmp_5486[2] + tmp_5488[2]];
    signal tmp_5490[3] <== [tmp_5438[0] * 6052753985947965968, tmp_5438[1] * 6052753985947965968, tmp_5438[2] * 6052753985947965968];
    signal tmp_5491[3] <== [tmp_5490[0] + tmp_5460[0], tmp_5490[1] + tmp_5460[1], tmp_5490[2] + tmp_5460[2]];
    signal tmp_5492[3] <== [tmp_5489[0] + tmp_5491[0], tmp_5489[1] + tmp_5491[1], tmp_5489[2] + tmp_5491[2]];
    signal tmp_5493[3] <== [tmp_5441[0] * 13608362914817483670, tmp_5441[1] * 13608362914817483670, tmp_5441[2] * 13608362914817483670];
    signal tmp_5494[3] <== [tmp_5493[0] + tmp_5460[0], tmp_5493[1] + tmp_5460[1], tmp_5493[2] + tmp_5460[2]];
    signal tmp_5495[3] <== [tmp_5492[0] + tmp_5494[0], tmp_5492[1] + tmp_5494[1], tmp_5492[2] + tmp_5494[2]];
    signal tmp_5496[3] <== [tmp_5444[0] * 18163707672964630459, tmp_5444[1] * 18163707672964630459, tmp_5444[2] * 18163707672964630459];
    signal tmp_5497[3] <== [tmp_5496[0] + tmp_5460[0], tmp_5496[1] + tmp_5460[1], tmp_5496[2] + tmp_5460[2]];
    signal tmp_5498[3] <== [tmp_5495[0] + tmp_5497[0], tmp_5495[1] + tmp_5497[1], tmp_5495[2] + tmp_5497[2]];
    signal tmp_5499[3] <== [tmp_5447[0] * 14373610220374016704, tmp_5447[1] * 14373610220374016704, tmp_5447[2] * 14373610220374016704];
    signal tmp_5500[3] <== [tmp_5499[0] + tmp_5460[0], tmp_5499[1] + tmp_5460[1], tmp_5499[2] + tmp_5460[2]];
    signal tmp_5501[3] <== [tmp_5498[0] + tmp_5500[0], tmp_5498[1] + tmp_5500[1], tmp_5498[2] + tmp_5500[2]];
    signal tmp_5502[3] <== [tmp_5450[0] * 6226282807566121054, tmp_5450[1] * 6226282807566121054, tmp_5450[2] * 6226282807566121054];
    signal tmp_5503[3] <== [tmp_5502[0] + tmp_5460[0], tmp_5502[1] + tmp_5460[1], tmp_5502[2] + tmp_5460[2]];
    signal tmp_5504[3] <== [tmp_5501[0] + tmp_5503[0], tmp_5501[1] + tmp_5503[1], tmp_5501[2] + tmp_5503[2]];
    signal tmp_5505[3] <== [tmp_5453[0] * 3643354756180461803, tmp_5453[1] * 3643354756180461803, tmp_5453[2] * 3643354756180461803];
    signal tmp_5506[3] <== [tmp_5505[0] + tmp_5460[0], tmp_5505[1] + tmp_5460[1], tmp_5505[2] + tmp_5460[2]];
    signal tmp_5507[3] <== [tmp_5504[0] + tmp_5506[0], tmp_5504[1] + tmp_5506[1], tmp_5504[2] + tmp_5506[2]];
    signal tmp_5508[3] <== [tmp_5456[0] * 13046961313070095543, tmp_5456[1] * 13046961313070095543, tmp_5456[2] * 13046961313070095543];
    signal tmp_5509[3] <== [tmp_5508[0] + tmp_5460[0], tmp_5508[1] + tmp_5460[1], tmp_5508[2] + tmp_5460[2]];
    signal tmp_5510[3] <== [tmp_5507[0] + tmp_5509[0], tmp_5507[1] + tmp_5509[1], tmp_5507[2] + tmp_5509[2]];
    signal tmp_5511[3] <== [tmp_5459[0] * 8594143216561850811, tmp_5459[1] * 8594143216561850811, tmp_5459[2] * 8594143216561850811];
    signal tmp_5512[3] <== [tmp_5511[0] + tmp_5460[0], tmp_5511[1] + tmp_5460[1], tmp_5511[2] + tmp_5460[2]];
    signal tmp_5513[3] <== [tmp_5510[0] + tmp_5512[0], tmp_5510[1] + tmp_5512[1], tmp_5510[2] + tmp_5512[2]];
    signal tmp_5514[3] <== [tmp_5468[0] + tmp_5513[0], tmp_5468[1] + tmp_5513[1], tmp_5468[2] + tmp_5513[2]];
    signal tmp_5515[3] <== [evals[130][0] - tmp_5514[0], evals[130][1] - tmp_5514[1], evals[130][2] - tmp_5514[2]];
    signal tmp_5516[3] <== CMul()(evals[76], tmp_5515);
    signal tmp_5517[3] <== [tmp_5465[0] + tmp_5516[0], tmp_5465[1] + tmp_5516[1], tmp_5465[2] + tmp_5516[2]];
    signal tmp_5518[3] <== CMul()(challengeQ, tmp_5517);
    signal tmp_5519[3] <== [evals[130][0] + tmp_4683[0], evals[130][1] + tmp_4683[1], evals[130][2] + tmp_4683[2]];
    signal tmp_5520[3] <== CMul()(tmp_4687, tmp_5519);
    signal tmp_5521[3] <== [tmp_5520[0] * 16040574633112940480, tmp_5520[1] * 16040574633112940480, tmp_5520[2] * 16040574633112940480];
    signal tmp_5522[3] <== [tmp_5470[0] * 14263299814608977431, tmp_5470[1] * 14263299814608977431, tmp_5470[2] * 14263299814608977431];
    signal tmp_5523[3] <== [tmp_5522[0] + tmp_5513[0], tmp_5522[1] + tmp_5513[1], tmp_5522[2] + tmp_5513[2]];
    signal tmp_5524[3] <== [tmp_5520[0] + tmp_5523[0], tmp_5520[1] + tmp_5523[1], tmp_5520[2] + tmp_5523[2]];
    signal tmp_5525[3] <== [tmp_5473[0] * 770395855193680981, tmp_5473[1] * 770395855193680981, tmp_5473[2] * 770395855193680981];
    signal tmp_5526[3] <== [tmp_5525[0] + tmp_5513[0], tmp_5525[1] + tmp_5513[1], tmp_5525[2] + tmp_5513[2]];
    signal tmp_5527[3] <== [tmp_5524[0] + tmp_5526[0], tmp_5524[1] + tmp_5526[1], tmp_5524[2] + tmp_5526[2]];
    signal tmp_5528[3] <== [tmp_5476[0] * 3459277367440070515, tmp_5476[1] * 3459277367440070515, tmp_5476[2] * 3459277367440070515];
    signal tmp_5529[3] <== [tmp_5528[0] + tmp_5513[0], tmp_5528[1] + tmp_5513[1], tmp_5528[2] + tmp_5513[2]];
    signal tmp_5530[3] <== [tmp_5527[0] + tmp_5529[0], tmp_5527[1] + tmp_5529[1], tmp_5527[2] + tmp_5529[2]];
    signal tmp_5531[3] <== [tmp_5479[0] * 17087697094293314027, tmp_5479[1] * 17087697094293314027, tmp_5479[2] * 17087697094293314027];
    signal tmp_5532[3] <== [tmp_5531[0] + tmp_5513[0], tmp_5531[1] + tmp_5513[1], tmp_5531[2] + tmp_5513[2]];
    signal tmp_5533[3] <== [tmp_5530[0] + tmp_5532[0], tmp_5530[1] + tmp_5532[1], tmp_5530[2] + tmp_5532[2]];
    signal tmp_5534[3] <== [tmp_5482[0] * 6694380135428747348, tmp_5482[1] * 6694380135428747348, tmp_5482[2] * 6694380135428747348];
    signal tmp_5535[3] <== [tmp_5534[0] + tmp_5513[0], tmp_5534[1] + tmp_5513[1], tmp_5534[2] + tmp_5513[2]];
    signal tmp_5536[3] <== [tmp_5533[0] + tmp_5535[0], tmp_5533[1] + tmp_5535[1], tmp_5533[2] + tmp_5535[2]];
    signal tmp_5537[3] <== [tmp_5485[0] * 2034408310088972836, tmp_5485[1] * 2034408310088972836, tmp_5485[2] * 2034408310088972836];
    signal tmp_5538[3] <== [tmp_5537[0] + tmp_5513[0], tmp_5537[1] + tmp_5513[1], tmp_5537[2] + tmp_5513[2]];
    signal tmp_5539[3] <== [tmp_5536[0] + tmp_5538[0], tmp_5536[1] + tmp_5538[1], tmp_5536[2] + tmp_5538[2]];
    signal tmp_5540[3] <== [tmp_5488[0] * 3434575637390274478, tmp_5488[1] * 3434575637390274478, tmp_5488[2] * 3434575637390274478];
    signal tmp_5541[3] <== [tmp_5540[0] + tmp_5513[0], tmp_5540[1] + tmp_5513[1], tmp_5540[2] + tmp_5513[2]];
    signal tmp_5542[3] <== [tmp_5539[0] + tmp_5541[0], tmp_5539[1] + tmp_5541[1], tmp_5539[2] + tmp_5541[2]];
    signal tmp_5543[3] <== [tmp_5491[0] * 6052753985947965968, tmp_5491[1] * 6052753985947965968, tmp_5491[2] * 6052753985947965968];
    signal tmp_5544[3] <== [tmp_5543[0] + tmp_5513[0], tmp_5543[1] + tmp_5513[1], tmp_5543[2] + tmp_5513[2]];
    signal tmp_5545[3] <== [tmp_5542[0] + tmp_5544[0], tmp_5542[1] + tmp_5544[1], tmp_5542[2] + tmp_5544[2]];
    signal tmp_5546[3] <== [tmp_5494[0] * 13608362914817483670, tmp_5494[1] * 13608362914817483670, tmp_5494[2] * 13608362914817483670];
    signal tmp_5547[3] <== [tmp_5546[0] + tmp_5513[0], tmp_5546[1] + tmp_5513[1], tmp_5546[2] + tmp_5513[2]];
    signal tmp_5548[3] <== [tmp_5545[0] + tmp_5547[0], tmp_5545[1] + tmp_5547[1], tmp_5545[2] + tmp_5547[2]];
    signal tmp_5549[3] <== [tmp_5497[0] * 18163707672964630459, tmp_5497[1] * 18163707672964630459, tmp_5497[2] * 18163707672964630459];
    signal tmp_5550[3] <== [tmp_5549[0] + tmp_5513[0], tmp_5549[1] + tmp_5513[1], tmp_5549[2] + tmp_5513[2]];
    signal tmp_5551[3] <== [tmp_5548[0] + tmp_5550[0], tmp_5548[1] + tmp_5550[1], tmp_5548[2] + tmp_5550[2]];
    signal tmp_5552[3] <== [tmp_5500[0] * 14373610220374016704, tmp_5500[1] * 14373610220374016704, tmp_5500[2] * 14373610220374016704];
    signal tmp_5553[3] <== [tmp_5552[0] + tmp_5513[0], tmp_5552[1] + tmp_5513[1], tmp_5552[2] + tmp_5513[2]];
    signal tmp_5554[3] <== [tmp_5551[0] + tmp_5553[0], tmp_5551[1] + tmp_5553[1], tmp_5551[2] + tmp_5553[2]];
    signal tmp_5555[3] <== [tmp_5503[0] * 6226282807566121054, tmp_5503[1] * 6226282807566121054, tmp_5503[2] * 6226282807566121054];
    signal tmp_5556[3] <== [tmp_5555[0] + tmp_5513[0], tmp_5555[1] + tmp_5513[1], tmp_5555[2] + tmp_5513[2]];
    signal tmp_5557[3] <== [tmp_5554[0] + tmp_5556[0], tmp_5554[1] + tmp_5556[1], tmp_5554[2] + tmp_5556[2]];
    signal tmp_5558[3] <== [tmp_5506[0] * 3643354756180461803, tmp_5506[1] * 3643354756180461803, tmp_5506[2] * 3643354756180461803];
    signal tmp_5559[3] <== [tmp_5558[0] + tmp_5513[0], tmp_5558[1] + tmp_5513[1], tmp_5558[2] + tmp_5513[2]];
    signal tmp_5560[3] <== [tmp_5557[0] + tmp_5559[0], tmp_5557[1] + tmp_5559[1], tmp_5557[2] + tmp_5559[2]];
    signal tmp_5561[3] <== [tmp_5509[0] * 13046961313070095543, tmp_5509[1] * 13046961313070095543, tmp_5509[2] * 13046961313070095543];
    signal tmp_5562[3] <== [tmp_5561[0] + tmp_5513[0], tmp_5561[1] + tmp_5513[1], tmp_5561[2] + tmp_5513[2]];
    signal tmp_5563[3] <== [tmp_5560[0] + tmp_5562[0], tmp_5560[1] + tmp_5562[1], tmp_5560[2] + tmp_5562[2]];
    signal tmp_5564[3] <== [tmp_5512[0] * 8594143216561850811, tmp_5512[1] * 8594143216561850811, tmp_5512[2] * 8594143216561850811];
    signal tmp_5565[3] <== [tmp_5564[0] + tmp_5513[0], tmp_5564[1] + tmp_5513[1], tmp_5564[2] + tmp_5513[2]];
    signal tmp_5566[3] <== [tmp_5563[0] + tmp_5565[0], tmp_5563[1] + tmp_5565[1], tmp_5563[2] + tmp_5565[2]];
    signal tmp_5567[3] <== [tmp_5521[0] + tmp_5566[0], tmp_5521[1] + tmp_5566[1], tmp_5521[2] + tmp_5566[2]];
    signal tmp_5568[3] <== [evals[131][0] - tmp_5567[0], evals[131][1] - tmp_5567[1], evals[131][2] - tmp_5567[2]];
    signal tmp_5569[3] <== CMul()(evals[76], tmp_5568);
    signal tmp_5570[3] <== [tmp_5518[0] + tmp_5569[0], tmp_5518[1] + tmp_5569[1], tmp_5518[2] + tmp_5569[2]];
    signal tmp_5571[3] <== CMul()(challengeQ, tmp_5570);
    signal tmp_5572[3] <== [evals[131][0] + tmp_4706[0], evals[131][1] + tmp_4706[1], evals[131][2] + tmp_4706[2]];
    signal tmp_5573[3] <== CMul()(tmp_4710, tmp_5572);
    signal tmp_5574[3] <== [tmp_5573[0] * 16040574633112940480, tmp_5573[1] * 16040574633112940480, tmp_5573[2] * 16040574633112940480];
    signal tmp_5575[3] <== [tmp_5523[0] * 14263299814608977431, tmp_5523[1] * 14263299814608977431, tmp_5523[2] * 14263299814608977431];
    signal tmp_5576[3] <== [tmp_5575[0] + tmp_5566[0], tmp_5575[1] + tmp_5566[1], tmp_5575[2] + tmp_5566[2]];
    signal tmp_5577[3] <== [tmp_5573[0] + tmp_5576[0], tmp_5573[1] + tmp_5576[1], tmp_5573[2] + tmp_5576[2]];
    signal tmp_5578[3] <== [tmp_5526[0] * 770395855193680981, tmp_5526[1] * 770395855193680981, tmp_5526[2] * 770395855193680981];
    signal tmp_5579[3] <== [tmp_5578[0] + tmp_5566[0], tmp_5578[1] + tmp_5566[1], tmp_5578[2] + tmp_5566[2]];
    signal tmp_5580[3] <== [tmp_5577[0] + tmp_5579[0], tmp_5577[1] + tmp_5579[1], tmp_5577[2] + tmp_5579[2]];
    signal tmp_5581[3] <== [tmp_5529[0] * 3459277367440070515, tmp_5529[1] * 3459277367440070515, tmp_5529[2] * 3459277367440070515];
    signal tmp_5582[3] <== [tmp_5581[0] + tmp_5566[0], tmp_5581[1] + tmp_5566[1], tmp_5581[2] + tmp_5566[2]];
    signal tmp_5583[3] <== [tmp_5580[0] + tmp_5582[0], tmp_5580[1] + tmp_5582[1], tmp_5580[2] + tmp_5582[2]];
    signal tmp_5584[3] <== [tmp_5532[0] * 17087697094293314027, tmp_5532[1] * 17087697094293314027, tmp_5532[2] * 17087697094293314027];
    signal tmp_5585[3] <== [tmp_5584[0] + tmp_5566[0], tmp_5584[1] + tmp_5566[1], tmp_5584[2] + tmp_5566[2]];
    signal tmp_5586[3] <== [tmp_5583[0] + tmp_5585[0], tmp_5583[1] + tmp_5585[1], tmp_5583[2] + tmp_5585[2]];
    signal tmp_5587[3] <== [tmp_5535[0] * 6694380135428747348, tmp_5535[1] * 6694380135428747348, tmp_5535[2] * 6694380135428747348];
    signal tmp_5588[3] <== [tmp_5587[0] + tmp_5566[0], tmp_5587[1] + tmp_5566[1], tmp_5587[2] + tmp_5566[2]];
    signal tmp_5589[3] <== [tmp_5586[0] + tmp_5588[0], tmp_5586[1] + tmp_5588[1], tmp_5586[2] + tmp_5588[2]];
    signal tmp_5590[3] <== [tmp_5538[0] * 2034408310088972836, tmp_5538[1] * 2034408310088972836, tmp_5538[2] * 2034408310088972836];
    signal tmp_5591[3] <== [tmp_5590[0] + tmp_5566[0], tmp_5590[1] + tmp_5566[1], tmp_5590[2] + tmp_5566[2]];
    signal tmp_5592[3] <== [tmp_5589[0] + tmp_5591[0], tmp_5589[1] + tmp_5591[1], tmp_5589[2] + tmp_5591[2]];
    signal tmp_5593[3] <== [tmp_5541[0] * 3434575637390274478, tmp_5541[1] * 3434575637390274478, tmp_5541[2] * 3434575637390274478];
    signal tmp_5594[3] <== [tmp_5593[0] + tmp_5566[0], tmp_5593[1] + tmp_5566[1], tmp_5593[2] + tmp_5566[2]];
    signal tmp_5595[3] <== [tmp_5592[0] + tmp_5594[0], tmp_5592[1] + tmp_5594[1], tmp_5592[2] + tmp_5594[2]];
    signal tmp_5596[3] <== [tmp_5544[0] * 6052753985947965968, tmp_5544[1] * 6052753985947965968, tmp_5544[2] * 6052753985947965968];
    signal tmp_5597[3] <== [tmp_5596[0] + tmp_5566[0], tmp_5596[1] + tmp_5566[1], tmp_5596[2] + tmp_5566[2]];
    signal tmp_5598[3] <== [tmp_5595[0] + tmp_5597[0], tmp_5595[1] + tmp_5597[1], tmp_5595[2] + tmp_5597[2]];
    signal tmp_5599[3] <== [tmp_5547[0] * 13608362914817483670, tmp_5547[1] * 13608362914817483670, tmp_5547[2] * 13608362914817483670];
    signal tmp_5600[3] <== [tmp_5599[0] + tmp_5566[0], tmp_5599[1] + tmp_5566[1], tmp_5599[2] + tmp_5566[2]];
    signal tmp_5601[3] <== [tmp_5598[0] + tmp_5600[0], tmp_5598[1] + tmp_5600[1], tmp_5598[2] + tmp_5600[2]];
    signal tmp_5602[3] <== [tmp_5550[0] * 18163707672964630459, tmp_5550[1] * 18163707672964630459, tmp_5550[2] * 18163707672964630459];
    signal tmp_5603[3] <== [tmp_5602[0] + tmp_5566[0], tmp_5602[1] + tmp_5566[1], tmp_5602[2] + tmp_5566[2]];
    signal tmp_5604[3] <== [tmp_5601[0] + tmp_5603[0], tmp_5601[1] + tmp_5603[1], tmp_5601[2] + tmp_5603[2]];
    signal tmp_5605[3] <== [tmp_5553[0] * 14373610220374016704, tmp_5553[1] * 14373610220374016704, tmp_5553[2] * 14373610220374016704];
    signal tmp_5606[3] <== [tmp_5605[0] + tmp_5566[0], tmp_5605[1] + tmp_5566[1], tmp_5605[2] + tmp_5566[2]];
    signal tmp_5607[3] <== [tmp_5604[0] + tmp_5606[0], tmp_5604[1] + tmp_5606[1], tmp_5604[2] + tmp_5606[2]];
    signal tmp_5608[3] <== [tmp_5556[0] * 6226282807566121054, tmp_5556[1] * 6226282807566121054, tmp_5556[2] * 6226282807566121054];
    signal tmp_5609[3] <== [tmp_5608[0] + tmp_5566[0], tmp_5608[1] + tmp_5566[1], tmp_5608[2] + tmp_5566[2]];
    signal tmp_5610[3] <== [tmp_5607[0] + tmp_5609[0], tmp_5607[1] + tmp_5609[1], tmp_5607[2] + tmp_5609[2]];
    signal tmp_5611[3] <== [tmp_5559[0] * 3643354756180461803, tmp_5559[1] * 3643354756180461803, tmp_5559[2] * 3643354756180461803];
    signal tmp_5612[3] <== [tmp_5611[0] + tmp_5566[0], tmp_5611[1] + tmp_5566[1], tmp_5611[2] + tmp_5566[2]];
    signal tmp_5613[3] <== [tmp_5610[0] + tmp_5612[0], tmp_5610[1] + tmp_5612[1], tmp_5610[2] + tmp_5612[2]];
    signal tmp_5614[3] <== [tmp_5562[0] * 13046961313070095543, tmp_5562[1] * 13046961313070095543, tmp_5562[2] * 13046961313070095543];
    signal tmp_5615[3] <== [tmp_5614[0] + tmp_5566[0], tmp_5614[1] + tmp_5566[1], tmp_5614[2] + tmp_5566[2]];
    signal tmp_5616[3] <== [tmp_5613[0] + tmp_5615[0], tmp_5613[1] + tmp_5615[1], tmp_5613[2] + tmp_5615[2]];
    signal tmp_5617[3] <== [tmp_5565[0] * 8594143216561850811, tmp_5565[1] * 8594143216561850811, tmp_5565[2] * 8594143216561850811];
    signal tmp_5618[3] <== [tmp_5617[0] + tmp_5566[0], tmp_5617[1] + tmp_5566[1], tmp_5617[2] + tmp_5566[2]];
    signal tmp_5619[3] <== [tmp_5616[0] + tmp_5618[0], tmp_5616[1] + tmp_5618[1], tmp_5616[2] + tmp_5618[2]];
    signal tmp_5620[3] <== [tmp_5574[0] + tmp_5619[0], tmp_5574[1] + tmp_5619[1], tmp_5574[2] + tmp_5619[2]];
    signal tmp_5621[3] <== [evals[132][0] - tmp_5620[0], evals[132][1] - tmp_5620[1], evals[132][2] - tmp_5620[2]];
    signal tmp_5622[3] <== CMul()(evals[76], tmp_5621);
    signal tmp_5623[3] <== [tmp_5571[0] + tmp_5622[0], tmp_5571[1] + tmp_5622[1], tmp_5571[2] + tmp_5622[2]];
    signal tmp_5624[3] <== CMul()(challengeQ, tmp_5623);
    signal tmp_5625[3] <== [evals[132][0] + tmp_4735[0], evals[132][1] + tmp_4735[1], evals[132][2] + tmp_4735[2]];
    signal tmp_5626[3] <== CMul()(tmp_4739, tmp_5625);
    signal tmp_5627[3] <== [tmp_5626[0] * 16040574633112940480, tmp_5626[1] * 16040574633112940480, tmp_5626[2] * 16040574633112940480];
    signal tmp_5628[3] <== [tmp_5576[0] * 14263299814608977431, tmp_5576[1] * 14263299814608977431, tmp_5576[2] * 14263299814608977431];
    signal tmp_5629[3] <== [tmp_5628[0] + tmp_5619[0], tmp_5628[1] + tmp_5619[1], tmp_5628[2] + tmp_5619[2]];
    signal tmp_5630[3] <== [tmp_5626[0] + tmp_5629[0], tmp_5626[1] + tmp_5629[1], tmp_5626[2] + tmp_5629[2]];
    signal tmp_5631[3] <== [tmp_5579[0] * 770395855193680981, tmp_5579[1] * 770395855193680981, tmp_5579[2] * 770395855193680981];
    signal tmp_5632[3] <== [tmp_5631[0] + tmp_5619[0], tmp_5631[1] + tmp_5619[1], tmp_5631[2] + tmp_5619[2]];
    signal tmp_5633[3] <== [tmp_5630[0] + tmp_5632[0], tmp_5630[1] + tmp_5632[1], tmp_5630[2] + tmp_5632[2]];
    signal tmp_5634[3] <== [tmp_5582[0] * 3459277367440070515, tmp_5582[1] * 3459277367440070515, tmp_5582[2] * 3459277367440070515];
    signal tmp_5635[3] <== [tmp_5634[0] + tmp_5619[0], tmp_5634[1] + tmp_5619[1], tmp_5634[2] + tmp_5619[2]];
    signal tmp_5636[3] <== [tmp_5633[0] + tmp_5635[0], tmp_5633[1] + tmp_5635[1], tmp_5633[2] + tmp_5635[2]];
    signal tmp_5637[3] <== [tmp_5585[0] * 17087697094293314027, tmp_5585[1] * 17087697094293314027, tmp_5585[2] * 17087697094293314027];
    signal tmp_5638[3] <== [tmp_5637[0] + tmp_5619[0], tmp_5637[1] + tmp_5619[1], tmp_5637[2] + tmp_5619[2]];
    signal tmp_5639[3] <== [tmp_5636[0] + tmp_5638[0], tmp_5636[1] + tmp_5638[1], tmp_5636[2] + tmp_5638[2]];
    signal tmp_5640[3] <== [tmp_5588[0] * 6694380135428747348, tmp_5588[1] * 6694380135428747348, tmp_5588[2] * 6694380135428747348];
    signal tmp_5641[3] <== [tmp_5640[0] + tmp_5619[0], tmp_5640[1] + tmp_5619[1], tmp_5640[2] + tmp_5619[2]];
    signal tmp_5642[3] <== [tmp_5639[0] + tmp_5641[0], tmp_5639[1] + tmp_5641[1], tmp_5639[2] + tmp_5641[2]];
    signal tmp_5643[3] <== [tmp_5591[0] * 2034408310088972836, tmp_5591[1] * 2034408310088972836, tmp_5591[2] * 2034408310088972836];
    signal tmp_5644[3] <== [tmp_5643[0] + tmp_5619[0], tmp_5643[1] + tmp_5619[1], tmp_5643[2] + tmp_5619[2]];
    signal tmp_5645[3] <== [tmp_5642[0] + tmp_5644[0], tmp_5642[1] + tmp_5644[1], tmp_5642[2] + tmp_5644[2]];
    signal tmp_5646[3] <== [tmp_5594[0] * 3434575637390274478, tmp_5594[1] * 3434575637390274478, tmp_5594[2] * 3434575637390274478];
    signal tmp_5647[3] <== [tmp_5646[0] + tmp_5619[0], tmp_5646[1] + tmp_5619[1], tmp_5646[2] + tmp_5619[2]];
    signal tmp_5648[3] <== [tmp_5645[0] + tmp_5647[0], tmp_5645[1] + tmp_5647[1], tmp_5645[2] + tmp_5647[2]];
    signal tmp_5649[3] <== [tmp_5597[0] * 6052753985947965968, tmp_5597[1] * 6052753985947965968, tmp_5597[2] * 6052753985947965968];
    signal tmp_5650[3] <== [tmp_5649[0] + tmp_5619[0], tmp_5649[1] + tmp_5619[1], tmp_5649[2] + tmp_5619[2]];
    signal tmp_5651[3] <== [tmp_5648[0] + tmp_5650[0], tmp_5648[1] + tmp_5650[1], tmp_5648[2] + tmp_5650[2]];
    signal tmp_5652[3] <== [tmp_5600[0] * 13608362914817483670, tmp_5600[1] * 13608362914817483670, tmp_5600[2] * 13608362914817483670];
    signal tmp_5653[3] <== [tmp_5652[0] + tmp_5619[0], tmp_5652[1] + tmp_5619[1], tmp_5652[2] + tmp_5619[2]];
    signal tmp_5654[3] <== [tmp_5651[0] + tmp_5653[0], tmp_5651[1] + tmp_5653[1], tmp_5651[2] + tmp_5653[2]];
    signal tmp_5655[3] <== [tmp_5603[0] * 18163707672964630459, tmp_5603[1] * 18163707672964630459, tmp_5603[2] * 18163707672964630459];
    signal tmp_5656[3] <== [tmp_5655[0] + tmp_5619[0], tmp_5655[1] + tmp_5619[1], tmp_5655[2] + tmp_5619[2]];
    signal tmp_5657[3] <== [tmp_5654[0] + tmp_5656[0], tmp_5654[1] + tmp_5656[1], tmp_5654[2] + tmp_5656[2]];
    signal tmp_5658[3] <== [tmp_5606[0] * 14373610220374016704, tmp_5606[1] * 14373610220374016704, tmp_5606[2] * 14373610220374016704];
    signal tmp_5659[3] <== [tmp_5658[0] + tmp_5619[0], tmp_5658[1] + tmp_5619[1], tmp_5658[2] + tmp_5619[2]];
    signal tmp_5660[3] <== [tmp_5657[0] + tmp_5659[0], tmp_5657[1] + tmp_5659[1], tmp_5657[2] + tmp_5659[2]];
    signal tmp_5661[3] <== [tmp_5609[0] * 6226282807566121054, tmp_5609[1] * 6226282807566121054, tmp_5609[2] * 6226282807566121054];
    signal tmp_5662[3] <== [tmp_5661[0] + tmp_5619[0], tmp_5661[1] + tmp_5619[1], tmp_5661[2] + tmp_5619[2]];
    signal tmp_5663[3] <== [tmp_5660[0] + tmp_5662[0], tmp_5660[1] + tmp_5662[1], tmp_5660[2] + tmp_5662[2]];
    signal tmp_5664[3] <== [tmp_5612[0] * 3643354756180461803, tmp_5612[1] * 3643354756180461803, tmp_5612[2] * 3643354756180461803];
    signal tmp_5665[3] <== [tmp_5664[0] + tmp_5619[0], tmp_5664[1] + tmp_5619[1], tmp_5664[2] + tmp_5619[2]];
    signal tmp_5666[3] <== [tmp_5663[0] + tmp_5665[0], tmp_5663[1] + tmp_5665[1], tmp_5663[2] + tmp_5665[2]];
    signal tmp_5667[3] <== [tmp_5615[0] * 13046961313070095543, tmp_5615[1] * 13046961313070095543, tmp_5615[2] * 13046961313070095543];
    signal tmp_5668[3] <== [tmp_5667[0] + tmp_5619[0], tmp_5667[1] + tmp_5619[1], tmp_5667[2] + tmp_5619[2]];
    signal tmp_5669[3] <== [tmp_5666[0] + tmp_5668[0], tmp_5666[1] + tmp_5668[1], tmp_5666[2] + tmp_5668[2]];
    signal tmp_5670[3] <== [tmp_5618[0] * 8594143216561850811, tmp_5618[1] * 8594143216561850811, tmp_5618[2] * 8594143216561850811];
    signal tmp_5671[3] <== [tmp_5670[0] + tmp_5619[0], tmp_5670[1] + tmp_5619[1], tmp_5670[2] + tmp_5619[2]];
    signal tmp_5672[3] <== [tmp_5669[0] + tmp_5671[0], tmp_5669[1] + tmp_5671[1], tmp_5669[2] + tmp_5671[2]];
    signal tmp_5673[3] <== [tmp_5627[0] + tmp_5672[0], tmp_5627[1] + tmp_5672[1], tmp_5627[2] + tmp_5672[2]];
    signal tmp_5674[3] <== [evals[133][0] - tmp_5673[0], evals[133][1] - tmp_5673[1], evals[133][2] - tmp_5673[2]];
    signal tmp_5675[3] <== CMul()(evals[76], tmp_5674);
    signal tmp_5676[3] <== [tmp_5624[0] + tmp_5675[0], tmp_5624[1] + tmp_5675[1], tmp_5624[2] + tmp_5675[2]];
    signal tmp_5677[3] <== CMul()(challengeQ, tmp_5676);
    signal tmp_5678[3] <== [evals[133][0] + tmp_4659[0], evals[133][1] + tmp_4659[1], evals[133][2] + tmp_4659[2]];
    signal tmp_5679[3] <== CMul()(tmp_4663, tmp_5678);
    signal tmp_5680[3] <== [tmp_5679[0] * 16040574633112940480, tmp_5679[1] * 16040574633112940480, tmp_5679[2] * 16040574633112940480];
    signal tmp_5681[3] <== [tmp_5629[0] * 14263299814608977431, tmp_5629[1] * 14263299814608977431, tmp_5629[2] * 14263299814608977431];
    signal tmp_5682[3] <== [tmp_5681[0] + tmp_5672[0], tmp_5681[1] + tmp_5672[1], tmp_5681[2] + tmp_5672[2]];
    signal tmp_5683[3] <== [tmp_5679[0] + tmp_5682[0], tmp_5679[1] + tmp_5682[1], tmp_5679[2] + tmp_5682[2]];
    signal tmp_5684[3] <== [tmp_5632[0] * 770395855193680981, tmp_5632[1] * 770395855193680981, tmp_5632[2] * 770395855193680981];
    signal tmp_5685[3] <== [tmp_5684[0] + tmp_5672[0], tmp_5684[1] + tmp_5672[1], tmp_5684[2] + tmp_5672[2]];
    signal tmp_5686[3] <== [tmp_5683[0] + tmp_5685[0], tmp_5683[1] + tmp_5685[1], tmp_5683[2] + tmp_5685[2]];
    signal tmp_5687[3] <== [tmp_5635[0] * 3459277367440070515, tmp_5635[1] * 3459277367440070515, tmp_5635[2] * 3459277367440070515];
    signal tmp_5688[3] <== [tmp_5687[0] + tmp_5672[0], tmp_5687[1] + tmp_5672[1], tmp_5687[2] + tmp_5672[2]];
    signal tmp_5689[3] <== [tmp_5686[0] + tmp_5688[0], tmp_5686[1] + tmp_5688[1], tmp_5686[2] + tmp_5688[2]];
    signal tmp_5690[3] <== [tmp_5638[0] * 17087697094293314027, tmp_5638[1] * 17087697094293314027, tmp_5638[2] * 17087697094293314027];
    signal tmp_5691[3] <== [tmp_5690[0] + tmp_5672[0], tmp_5690[1] + tmp_5672[1], tmp_5690[2] + tmp_5672[2]];
    signal tmp_5692[3] <== [tmp_5689[0] + tmp_5691[0], tmp_5689[1] + tmp_5691[1], tmp_5689[2] + tmp_5691[2]];
    signal tmp_5693[3] <== [tmp_5641[0] * 6694380135428747348, tmp_5641[1] * 6694380135428747348, tmp_5641[2] * 6694380135428747348];
    signal tmp_5694[3] <== [tmp_5693[0] + tmp_5672[0], tmp_5693[1] + tmp_5672[1], tmp_5693[2] + tmp_5672[2]];
    signal tmp_5695[3] <== [tmp_5692[0] + tmp_5694[0], tmp_5692[1] + tmp_5694[1], tmp_5692[2] + tmp_5694[2]];
    signal tmp_5696[3] <== [tmp_5644[0] * 2034408310088972836, tmp_5644[1] * 2034408310088972836, tmp_5644[2] * 2034408310088972836];
    signal tmp_5697[3] <== [tmp_5696[0] + tmp_5672[0], tmp_5696[1] + tmp_5672[1], tmp_5696[2] + tmp_5672[2]];
    signal tmp_5698[3] <== [tmp_5695[0] + tmp_5697[0], tmp_5695[1] + tmp_5697[1], tmp_5695[2] + tmp_5697[2]];
    signal tmp_5699[3] <== [tmp_5647[0] * 3434575637390274478, tmp_5647[1] * 3434575637390274478, tmp_5647[2] * 3434575637390274478];
    signal tmp_5700[3] <== [tmp_5699[0] + tmp_5672[0], tmp_5699[1] + tmp_5672[1], tmp_5699[2] + tmp_5672[2]];
    signal tmp_5701[3] <== [tmp_5698[0] + tmp_5700[0], tmp_5698[1] + tmp_5700[1], tmp_5698[2] + tmp_5700[2]];
    signal tmp_5702[3] <== [tmp_5650[0] * 6052753985947965968, tmp_5650[1] * 6052753985947965968, tmp_5650[2] * 6052753985947965968];
    signal tmp_5703[3] <== [tmp_5702[0] + tmp_5672[0], tmp_5702[1] + tmp_5672[1], tmp_5702[2] + tmp_5672[2]];
    signal tmp_5704[3] <== [tmp_5701[0] + tmp_5703[0], tmp_5701[1] + tmp_5703[1], tmp_5701[2] + tmp_5703[2]];
    signal tmp_5705[3] <== [tmp_5653[0] * 13608362914817483670, tmp_5653[1] * 13608362914817483670, tmp_5653[2] * 13608362914817483670];
    signal tmp_5706[3] <== [tmp_5705[0] + tmp_5672[0], tmp_5705[1] + tmp_5672[1], tmp_5705[2] + tmp_5672[2]];
    signal tmp_5707[3] <== [tmp_5704[0] + tmp_5706[0], tmp_5704[1] + tmp_5706[1], tmp_5704[2] + tmp_5706[2]];
    signal tmp_5708[3] <== [tmp_5656[0] * 18163707672964630459, tmp_5656[1] * 18163707672964630459, tmp_5656[2] * 18163707672964630459];
    signal tmp_5709[3] <== [tmp_5708[0] + tmp_5672[0], tmp_5708[1] + tmp_5672[1], tmp_5708[2] + tmp_5672[2]];
    signal tmp_5710[3] <== [tmp_5707[0] + tmp_5709[0], tmp_5707[1] + tmp_5709[1], tmp_5707[2] + tmp_5709[2]];
    signal tmp_5711[3] <== [tmp_5659[0] * 14373610220374016704, tmp_5659[1] * 14373610220374016704, tmp_5659[2] * 14373610220374016704];
    signal tmp_5712[3] <== [tmp_5711[0] + tmp_5672[0], tmp_5711[1] + tmp_5672[1], tmp_5711[2] + tmp_5672[2]];
    signal tmp_5713[3] <== [tmp_5710[0] + tmp_5712[0], tmp_5710[1] + tmp_5712[1], tmp_5710[2] + tmp_5712[2]];
    signal tmp_5714[3] <== [tmp_5662[0] * 6226282807566121054, tmp_5662[1] * 6226282807566121054, tmp_5662[2] * 6226282807566121054];
    signal tmp_5715[3] <== [tmp_5714[0] + tmp_5672[0], tmp_5714[1] + tmp_5672[1], tmp_5714[2] + tmp_5672[2]];
    signal tmp_5716[3] <== [tmp_5713[0] + tmp_5715[0], tmp_5713[1] + tmp_5715[1], tmp_5713[2] + tmp_5715[2]];
    signal tmp_5717[3] <== [tmp_5665[0] * 3643354756180461803, tmp_5665[1] * 3643354756180461803, tmp_5665[2] * 3643354756180461803];
    signal tmp_5718[3] <== [tmp_5717[0] + tmp_5672[0], tmp_5717[1] + tmp_5672[1], tmp_5717[2] + tmp_5672[2]];
    signal tmp_5719[3] <== [tmp_5716[0] + tmp_5718[0], tmp_5716[1] + tmp_5718[1], tmp_5716[2] + tmp_5718[2]];
    signal tmp_5720[3] <== [tmp_5668[0] * 13046961313070095543, tmp_5668[1] * 13046961313070095543, tmp_5668[2] * 13046961313070095543];
    signal tmp_5721[3] <== [tmp_5720[0] + tmp_5672[0], tmp_5720[1] + tmp_5672[1], tmp_5720[2] + tmp_5672[2]];
    signal tmp_5722[3] <== [tmp_5719[0] + tmp_5721[0], tmp_5719[1] + tmp_5721[1], tmp_5719[2] + tmp_5721[2]];
    signal tmp_5723[3] <== [tmp_5671[0] * 8594143216561850811, tmp_5671[1] * 8594143216561850811, tmp_5671[2] * 8594143216561850811];
    signal tmp_5724[3] <== [tmp_5723[0] + tmp_5672[0], tmp_5723[1] + tmp_5672[1], tmp_5723[2] + tmp_5672[2]];
    signal tmp_5725[3] <== [tmp_5722[0] + tmp_5724[0], tmp_5722[1] + tmp_5724[1], tmp_5722[2] + tmp_5724[2]];
    signal tmp_5726[3] <== [tmp_5680[0] + tmp_5725[0], tmp_5680[1] + tmp_5725[1], tmp_5680[2] + tmp_5725[2]];
    signal tmp_5727[3] <== [evals[134][0] - tmp_5726[0], evals[134][1] - tmp_5726[1], evals[134][2] - tmp_5726[2]];
    signal tmp_5728[3] <== CMul()(evals[76], tmp_5727);
    signal tmp_5729[3] <== [tmp_5677[0] + tmp_5728[0], tmp_5677[1] + tmp_5728[1], tmp_5677[2] + tmp_5728[2]];
    signal tmp_5730[3] <== CMul()(challengeQ, tmp_5729);
    signal tmp_5731[3] <== [evals[134][0] + tmp_4789[0], evals[134][1] + tmp_4789[1], evals[134][2] + tmp_4789[2]];
    signal tmp_5732[3] <== CMul()(tmp_4793, tmp_5731);
    signal tmp_5733[3] <== [tmp_5732[0] * 16040574633112940480, tmp_5732[1] * 16040574633112940480, tmp_5732[2] * 16040574633112940480];
    signal tmp_5734[3] <== [tmp_5682[0] * 14263299814608977431, tmp_5682[1] * 14263299814608977431, tmp_5682[2] * 14263299814608977431];
    signal tmp_5735[3] <== [tmp_5734[0] + tmp_5725[0], tmp_5734[1] + tmp_5725[1], tmp_5734[2] + tmp_5725[2]];
    signal tmp_5736[3] <== [tmp_5732[0] + tmp_5735[0], tmp_5732[1] + tmp_5735[1], tmp_5732[2] + tmp_5735[2]];
    signal tmp_5737[3] <== [tmp_5685[0] * 770395855193680981, tmp_5685[1] * 770395855193680981, tmp_5685[2] * 770395855193680981];
    signal tmp_5738[3] <== [tmp_5737[0] + tmp_5725[0], tmp_5737[1] + tmp_5725[1], tmp_5737[2] + tmp_5725[2]];
    signal tmp_5739[3] <== [tmp_5736[0] + tmp_5738[0], tmp_5736[1] + tmp_5738[1], tmp_5736[2] + tmp_5738[2]];
    signal tmp_5740[3] <== [tmp_5688[0] * 3459277367440070515, tmp_5688[1] * 3459277367440070515, tmp_5688[2] * 3459277367440070515];
    signal tmp_5741[3] <== [tmp_5740[0] + tmp_5725[0], tmp_5740[1] + tmp_5725[1], tmp_5740[2] + tmp_5725[2]];
    signal tmp_5742[3] <== [tmp_5739[0] + tmp_5741[0], tmp_5739[1] + tmp_5741[1], tmp_5739[2] + tmp_5741[2]];
    signal tmp_5743[3] <== [tmp_5691[0] * 17087697094293314027, tmp_5691[1] * 17087697094293314027, tmp_5691[2] * 17087697094293314027];
    signal tmp_5744[3] <== [tmp_5743[0] + tmp_5725[0], tmp_5743[1] + tmp_5725[1], tmp_5743[2] + tmp_5725[2]];
    signal tmp_5745[3] <== [tmp_5742[0] + tmp_5744[0], tmp_5742[1] + tmp_5744[1], tmp_5742[2] + tmp_5744[2]];
    signal tmp_5746[3] <== [tmp_5694[0] * 6694380135428747348, tmp_5694[1] * 6694380135428747348, tmp_5694[2] * 6694380135428747348];
    signal tmp_5747[3] <== [tmp_5746[0] + tmp_5725[0], tmp_5746[1] + tmp_5725[1], tmp_5746[2] + tmp_5725[2]];
    signal tmp_5748[3] <== [tmp_5745[0] + tmp_5747[0], tmp_5745[1] + tmp_5747[1], tmp_5745[2] + tmp_5747[2]];
    signal tmp_5749[3] <== [tmp_5697[0] * 2034408310088972836, tmp_5697[1] * 2034408310088972836, tmp_5697[2] * 2034408310088972836];
    signal tmp_5750[3] <== [tmp_5749[0] + tmp_5725[0], tmp_5749[1] + tmp_5725[1], tmp_5749[2] + tmp_5725[2]];
    signal tmp_5751[3] <== [tmp_5748[0] + tmp_5750[0], tmp_5748[1] + tmp_5750[1], tmp_5748[2] + tmp_5750[2]];
    signal tmp_5752[3] <== [tmp_5700[0] * 3434575637390274478, tmp_5700[1] * 3434575637390274478, tmp_5700[2] * 3434575637390274478];
    signal tmp_5753[3] <== [tmp_5752[0] + tmp_5725[0], tmp_5752[1] + tmp_5725[1], tmp_5752[2] + tmp_5725[2]];
    signal tmp_5754[3] <== [tmp_5751[0] + tmp_5753[0], tmp_5751[1] + tmp_5753[1], tmp_5751[2] + tmp_5753[2]];
    signal tmp_5755[3] <== [tmp_5703[0] * 6052753985947965968, tmp_5703[1] * 6052753985947965968, tmp_5703[2] * 6052753985947965968];
    signal tmp_5756[3] <== [tmp_5755[0] + tmp_5725[0], tmp_5755[1] + tmp_5725[1], tmp_5755[2] + tmp_5725[2]];
    signal tmp_5757[3] <== [tmp_5754[0] + tmp_5756[0], tmp_5754[1] + tmp_5756[1], tmp_5754[2] + tmp_5756[2]];
    signal tmp_5758[3] <== [tmp_5706[0] * 13608362914817483670, tmp_5706[1] * 13608362914817483670, tmp_5706[2] * 13608362914817483670];
    signal tmp_5759[3] <== [tmp_5758[0] + tmp_5725[0], tmp_5758[1] + tmp_5725[1], tmp_5758[2] + tmp_5725[2]];
    signal tmp_5760[3] <== [tmp_5757[0] + tmp_5759[0], tmp_5757[1] + tmp_5759[1], tmp_5757[2] + tmp_5759[2]];
    signal tmp_5761[3] <== [tmp_5709[0] * 18163707672964630459, tmp_5709[1] * 18163707672964630459, tmp_5709[2] * 18163707672964630459];
    signal tmp_5762[3] <== [tmp_5761[0] + tmp_5725[0], tmp_5761[1] + tmp_5725[1], tmp_5761[2] + tmp_5725[2]];
    signal tmp_5763[3] <== [tmp_5760[0] + tmp_5762[0], tmp_5760[1] + tmp_5762[1], tmp_5760[2] + tmp_5762[2]];
    signal tmp_5764[3] <== [tmp_5712[0] * 14373610220374016704, tmp_5712[1] * 14373610220374016704, tmp_5712[2] * 14373610220374016704];
    signal tmp_5765[3] <== [tmp_5764[0] + tmp_5725[0], tmp_5764[1] + tmp_5725[1], tmp_5764[2] + tmp_5725[2]];
    signal tmp_5766[3] <== [tmp_5763[0] + tmp_5765[0], tmp_5763[1] + tmp_5765[1], tmp_5763[2] + tmp_5765[2]];
    signal tmp_5767[3] <== [tmp_5715[0] * 6226282807566121054, tmp_5715[1] * 6226282807566121054, tmp_5715[2] * 6226282807566121054];
    signal tmp_5768[3] <== [tmp_5767[0] + tmp_5725[0], tmp_5767[1] + tmp_5725[1], tmp_5767[2] + tmp_5725[2]];
    signal tmp_5769[3] <== [tmp_5766[0] + tmp_5768[0], tmp_5766[1] + tmp_5768[1], tmp_5766[2] + tmp_5768[2]];
    signal tmp_5770[3] <== [tmp_5718[0] * 3643354756180461803, tmp_5718[1] * 3643354756180461803, tmp_5718[2] * 3643354756180461803];
    signal tmp_5771[3] <== [tmp_5770[0] + tmp_5725[0], tmp_5770[1] + tmp_5725[1], tmp_5770[2] + tmp_5725[2]];
    signal tmp_5772[3] <== [tmp_5769[0] + tmp_5771[0], tmp_5769[1] + tmp_5771[1], tmp_5769[2] + tmp_5771[2]];
    signal tmp_5773[3] <== [tmp_5721[0] * 13046961313070095543, tmp_5721[1] * 13046961313070095543, tmp_5721[2] * 13046961313070095543];
    signal tmp_5774[3] <== [tmp_5773[0] + tmp_5725[0], tmp_5773[1] + tmp_5725[1], tmp_5773[2] + tmp_5725[2]];
    signal tmp_5775[3] <== [tmp_5772[0] + tmp_5774[0], tmp_5772[1] + tmp_5774[1], tmp_5772[2] + tmp_5774[2]];
    signal tmp_5776[3] <== [tmp_5724[0] * 8594143216561850811, tmp_5724[1] * 8594143216561850811, tmp_5724[2] * 8594143216561850811];
    signal tmp_5777[3] <== [tmp_5776[0] + tmp_5725[0], tmp_5776[1] + tmp_5725[1], tmp_5776[2] + tmp_5725[2]];
    signal tmp_5778[3] <== [tmp_5775[0] + tmp_5777[0], tmp_5775[1] + tmp_5777[1], tmp_5775[2] + tmp_5777[2]];
    signal tmp_5779[3] <== [tmp_5733[0] + tmp_5778[0], tmp_5733[1] + tmp_5778[1], tmp_5733[2] + tmp_5778[2]];
    signal tmp_5780[3] <== [evals[135][0] - tmp_5779[0], evals[135][1] - tmp_5779[1], evals[135][2] - tmp_5779[2]];
    signal tmp_5781[3] <== CMul()(evals[76], tmp_5780);
    signal tmp_5782[3] <== [tmp_5730[0] + tmp_5781[0], tmp_5730[1] + tmp_5781[1], tmp_5730[2] + tmp_5781[2]];
    signal tmp_5783[3] <== CMul()(challengeQ, tmp_5782);
    signal tmp_5784[3] <== [evals[135][0] + tmp_4812[0], evals[135][1] + tmp_4812[1], evals[135][2] + tmp_4812[2]];
    signal tmp_5785[3] <== CMul()(tmp_4816, tmp_5784);
    signal tmp_5786[3] <== [tmp_5785[0] * 16040574633112940480, tmp_5785[1] * 16040574633112940480, tmp_5785[2] * 16040574633112940480];
    signal tmp_5787[3] <== [tmp_5735[0] * 14263299814608977431, tmp_5735[1] * 14263299814608977431, tmp_5735[2] * 14263299814608977431];
    signal tmp_5788[3] <== [tmp_5787[0] + tmp_5778[0], tmp_5787[1] + tmp_5778[1], tmp_5787[2] + tmp_5778[2]];
    signal tmp_5789[3] <== [tmp_5785[0] + tmp_5788[0], tmp_5785[1] + tmp_5788[1], tmp_5785[2] + tmp_5788[2]];
    signal tmp_5790[3] <== [tmp_5738[0] * 770395855193680981, tmp_5738[1] * 770395855193680981, tmp_5738[2] * 770395855193680981];
    signal tmp_5791[3] <== [tmp_5790[0] + tmp_5778[0], tmp_5790[1] + tmp_5778[1], tmp_5790[2] + tmp_5778[2]];
    signal tmp_5792[3] <== [tmp_5789[0] + tmp_5791[0], tmp_5789[1] + tmp_5791[1], tmp_5789[2] + tmp_5791[2]];
    signal tmp_5793[3] <== [tmp_5741[0] * 3459277367440070515, tmp_5741[1] * 3459277367440070515, tmp_5741[2] * 3459277367440070515];
    signal tmp_5794[3] <== [tmp_5793[0] + tmp_5778[0], tmp_5793[1] + tmp_5778[1], tmp_5793[2] + tmp_5778[2]];
    signal tmp_5795[3] <== [tmp_5792[0] + tmp_5794[0], tmp_5792[1] + tmp_5794[1], tmp_5792[2] + tmp_5794[2]];
    signal tmp_5796[3] <== [tmp_5744[0] * 17087697094293314027, tmp_5744[1] * 17087697094293314027, tmp_5744[2] * 17087697094293314027];
    signal tmp_5797[3] <== [tmp_5796[0] + tmp_5778[0], tmp_5796[1] + tmp_5778[1], tmp_5796[2] + tmp_5778[2]];
    signal tmp_5798[3] <== [tmp_5795[0] + tmp_5797[0], tmp_5795[1] + tmp_5797[1], tmp_5795[2] + tmp_5797[2]];
    signal tmp_5799[3] <== [tmp_5747[0] * 6694380135428747348, tmp_5747[1] * 6694380135428747348, tmp_5747[2] * 6694380135428747348];
    signal tmp_5800[3] <== [tmp_5799[0] + tmp_5778[0], tmp_5799[1] + tmp_5778[1], tmp_5799[2] + tmp_5778[2]];
    signal tmp_5801[3] <== [tmp_5798[0] + tmp_5800[0], tmp_5798[1] + tmp_5800[1], tmp_5798[2] + tmp_5800[2]];
    signal tmp_5802[3] <== [tmp_5750[0] * 2034408310088972836, tmp_5750[1] * 2034408310088972836, tmp_5750[2] * 2034408310088972836];
    signal tmp_5803[3] <== [tmp_5802[0] + tmp_5778[0], tmp_5802[1] + tmp_5778[1], tmp_5802[2] + tmp_5778[2]];
    signal tmp_5804[3] <== [tmp_5801[0] + tmp_5803[0], tmp_5801[1] + tmp_5803[1], tmp_5801[2] + tmp_5803[2]];
    signal tmp_5805[3] <== [tmp_5753[0] * 3434575637390274478, tmp_5753[1] * 3434575637390274478, tmp_5753[2] * 3434575637390274478];
    signal tmp_5806[3] <== [tmp_5805[0] + tmp_5778[0], tmp_5805[1] + tmp_5778[1], tmp_5805[2] + tmp_5778[2]];
    signal tmp_5807[3] <== [tmp_5804[0] + tmp_5806[0], tmp_5804[1] + tmp_5806[1], tmp_5804[2] + tmp_5806[2]];
    signal tmp_5808[3] <== [tmp_5756[0] * 6052753985947965968, tmp_5756[1] * 6052753985947965968, tmp_5756[2] * 6052753985947965968];
    signal tmp_5809[3] <== [tmp_5808[0] + tmp_5778[0], tmp_5808[1] + tmp_5778[1], tmp_5808[2] + tmp_5778[2]];
    signal tmp_5810[3] <== [tmp_5807[0] + tmp_5809[0], tmp_5807[1] + tmp_5809[1], tmp_5807[2] + tmp_5809[2]];
    signal tmp_5811[3] <== [tmp_5759[0] * 13608362914817483670, tmp_5759[1] * 13608362914817483670, tmp_5759[2] * 13608362914817483670];
    tmp_5812 <== [tmp_5811[0] + tmp_5778[0], tmp_5811[1] + tmp_5778[1], tmp_5811[2] + tmp_5778[2]];
    signal tmp_5813[3] <== [tmp_5810[0] + tmp_5812[0], tmp_5810[1] + tmp_5812[1], tmp_5810[2] + tmp_5812[2]];
    signal tmp_5814[3] <== [tmp_5762[0] * 18163707672964630459, tmp_5762[1] * 18163707672964630459, tmp_5762[2] * 18163707672964630459];
    tmp_5815 <== [tmp_5814[0] + tmp_5778[0], tmp_5814[1] + tmp_5778[1], tmp_5814[2] + tmp_5778[2]];
    signal tmp_5816[3] <== [tmp_5813[0] + tmp_5815[0], tmp_5813[1] + tmp_5815[1], tmp_5813[2] + tmp_5815[2]];
    signal tmp_5817[3] <== [tmp_5765[0] * 14373610220374016704, tmp_5765[1] * 14373610220374016704, tmp_5765[2] * 14373610220374016704];
    tmp_5818 <== [tmp_5817[0] + tmp_5778[0], tmp_5817[1] + tmp_5778[1], tmp_5817[2] + tmp_5778[2]];
    signal tmp_5819[3] <== [tmp_5816[0] + tmp_5818[0], tmp_5816[1] + tmp_5818[1], tmp_5816[2] + tmp_5818[2]];
    signal tmp_5820[3] <== [tmp_5768[0] * 6226282807566121054, tmp_5768[1] * 6226282807566121054, tmp_5768[2] * 6226282807566121054];
    tmp_5821 <== [tmp_5820[0] + tmp_5778[0], tmp_5820[1] + tmp_5778[1], tmp_5820[2] + tmp_5778[2]];
    signal tmp_5822[3] <== [tmp_5819[0] + tmp_5821[0], tmp_5819[1] + tmp_5821[1], tmp_5819[2] + tmp_5821[2]];
    signal tmp_5823[3] <== [tmp_5771[0] * 3643354756180461803, tmp_5771[1] * 3643354756180461803, tmp_5771[2] * 3643354756180461803];
    tmp_5824 <== [tmp_5823[0] + tmp_5778[0], tmp_5823[1] + tmp_5778[1], tmp_5823[2] + tmp_5778[2]];
    signal tmp_5825[3] <== [tmp_5822[0] + tmp_5824[0], tmp_5822[1] + tmp_5824[1], tmp_5822[2] + tmp_5824[2]];
    signal tmp_5826[3] <== [tmp_5774[0] * 13046961313070095543, tmp_5774[1] * 13046961313070095543, tmp_5774[2] * 13046961313070095543];
    tmp_5827 <== [tmp_5826[0] + tmp_5778[0], tmp_5826[1] + tmp_5778[1], tmp_5826[2] + tmp_5778[2]];
    signal tmp_5828[3] <== [tmp_5825[0] + tmp_5827[0], tmp_5825[1] + tmp_5827[1], tmp_5825[2] + tmp_5827[2]];
    signal tmp_5829[3] <== [tmp_5777[0] * 8594143216561850811, tmp_5777[1] * 8594143216561850811, tmp_5777[2] * 8594143216561850811];
    tmp_5830 <== [tmp_5829[0] + tmp_5778[0], tmp_5829[1] + tmp_5778[1], tmp_5829[2] + tmp_5778[2]];
    tmp_5831 <== [tmp_5828[0] + tmp_5830[0], tmp_5828[1] + tmp_5830[1], tmp_5828[2] + tmp_5830[2]];
    signal tmp_5832[3] <== [tmp_5786[0] + tmp_5831[0], tmp_5786[1] + tmp_5831[1], tmp_5786[2] + tmp_5831[2]];
    signal tmp_5833[3] <== [evals[136][0] - tmp_5832[0], evals[136][1] - tmp_5832[1], evals[136][2] - tmp_5832[2]];
    signal tmp_5834[3] <== CMul()(evals[76], tmp_5833);
    signal tmp_5835[3] <== [tmp_5783[0] + tmp_5834[0], tmp_5783[1] + tmp_5834[1], tmp_5783[2] + tmp_5834[2]];
    tmp_5836 <== CMul()(challengeQ, tmp_5835);
    signal tmp_5837[3] <== [evals[136][0] + tmp_4841[0], evals[136][1] + tmp_4841[1], evals[136][2] + tmp_4841[2]];
    signal tmp_5838[3] <== CMul()(tmp_4845, tmp_5837);
    tmp_5839 <== [tmp_5838[0] * 16040574633112940480, tmp_5838[1] * 16040574633112940480, tmp_5838[2] * 16040574633112940480];
    signal tmp_5840[3] <== [tmp_5788[0] * 14263299814608977431, tmp_5788[1] * 14263299814608977431, tmp_5788[2] * 14263299814608977431];
    tmp_5841 <== [tmp_5840[0] + tmp_5831[0], tmp_5840[1] + tmp_5831[1], tmp_5840[2] + tmp_5831[2]];
    signal tmp_5842[3] <== [tmp_5838[0] + tmp_5841[0], tmp_5838[1] + tmp_5841[1], tmp_5838[2] + tmp_5841[2]];
    signal tmp_5843[3] <== [tmp_5791[0] * 770395855193680981, tmp_5791[1] * 770395855193680981, tmp_5791[2] * 770395855193680981];
    tmp_5844 <== [tmp_5843[0] + tmp_5831[0], tmp_5843[1] + tmp_5831[1], tmp_5843[2] + tmp_5831[2]];
    signal tmp_5845[3] <== [tmp_5842[0] + tmp_5844[0], tmp_5842[1] + tmp_5844[1], tmp_5842[2] + tmp_5844[2]];
    signal tmp_5846[3] <== [tmp_5794[0] * 3459277367440070515, tmp_5794[1] * 3459277367440070515, tmp_5794[2] * 3459277367440070515];
    tmp_5847 <== [tmp_5846[0] + tmp_5831[0], tmp_5846[1] + tmp_5831[1], tmp_5846[2] + tmp_5831[2]];
    signal tmp_5848[3] <== [tmp_5845[0] + tmp_5847[0], tmp_5845[1] + tmp_5847[1], tmp_5845[2] + tmp_5847[2]];
    signal tmp_5849[3] <== [tmp_5797[0] * 17087697094293314027, tmp_5797[1] * 17087697094293314027, tmp_5797[2] * 17087697094293314027];
    tmp_5850 <== [tmp_5849[0] + tmp_5831[0], tmp_5849[1] + tmp_5831[1], tmp_5849[2] + tmp_5831[2]];
    signal tmp_5851[3] <== [tmp_5848[0] + tmp_5850[0], tmp_5848[1] + tmp_5850[1], tmp_5848[2] + tmp_5850[2]];
    signal tmp_5852[3] <== [tmp_5800[0] * 6694380135428747348, tmp_5800[1] * 6694380135428747348, tmp_5800[2] * 6694380135428747348];
    tmp_5853 <== [tmp_5852[0] + tmp_5831[0], tmp_5852[1] + tmp_5831[1], tmp_5852[2] + tmp_5831[2]];
    signal tmp_5854[3] <== [tmp_5851[0] + tmp_5853[0], tmp_5851[1] + tmp_5853[1], tmp_5851[2] + tmp_5853[2]];
    signal tmp_5855[3] <== [tmp_5803[0] * 2034408310088972836, tmp_5803[1] * 2034408310088972836, tmp_5803[2] * 2034408310088972836];
    tmp_5856 <== [tmp_5855[0] + tmp_5831[0], tmp_5855[1] + tmp_5831[1], tmp_5855[2] + tmp_5831[2]];
    signal tmp_5857[3] <== [tmp_5854[0] + tmp_5856[0], tmp_5854[1] + tmp_5856[1], tmp_5854[2] + tmp_5856[2]];
    signal tmp_5858[3] <== [tmp_5806[0] * 3434575637390274478, tmp_5806[1] * 3434575637390274478, tmp_5806[2] * 3434575637390274478];
    tmp_5859 <== [tmp_5858[0] + tmp_5831[0], tmp_5858[1] + tmp_5831[1], tmp_5858[2] + tmp_5831[2]];
    tmp_5860 <== [tmp_5857[0] + tmp_5859[0], tmp_5857[1] + tmp_5859[1], tmp_5857[2] + tmp_5859[2]];
    tmp_5861 <== [tmp_5809[0] * 6052753985947965968, tmp_5809[1] * 6052753985947965968, tmp_5809[2] * 6052753985947965968];
}

template VerifyEvaluationsChunks2() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[198][3];
    signal input publics[391];

    signal input Zh[3];

    signal input tmp_4765[3];
 
    signal input tmp_4769[3];
 
    signal input tmp_5812[3];
 
    signal input tmp_5815[3];
 
    signal input tmp_5818[3];
 
    signal input tmp_5821[3];
 
    signal input tmp_5824[3];
 
    signal input tmp_5827[3];
 
    signal input tmp_5830[3];
 
    signal input tmp_5831[3];
 
    signal input tmp_5836[3];
 
    signal input tmp_5839[3];
 
    signal input tmp_5841[3];
 
    signal input tmp_5844[3];
 
    signal input tmp_5847[3];
 
    signal input tmp_5850[3];
 
    signal input tmp_5853[3];
 
    signal input tmp_5856[3];
 
    signal input tmp_5859[3];
 
    signal input tmp_5860[3];
 
    signal input tmp_5861[3];
 

    signal output tmp_5945[3];
 
    signal output tmp_6004[3];
 
    signal output tmp_6063[3];
 
    signal output tmp_6122[3];
 
    signal output tmp_6181[3];
 
    signal output tmp_6240[3];
 
    signal output tmp_6853[3];
 
    signal output tmp_6858[3];
 
    signal output tmp_6859[3];
 
    signal output tmp_6860[3];
 
    signal tmp_5862[3] <== [tmp_5861[0] + tmp_5831[0], tmp_5861[1] + tmp_5831[1], tmp_5861[2] + tmp_5831[2]];
    signal tmp_5863[3] <== [tmp_5860[0] + tmp_5862[0], tmp_5860[1] + tmp_5862[1], tmp_5860[2] + tmp_5862[2]];
    signal tmp_5864[3] <== [tmp_5812[0] * 13608362914817483670, tmp_5812[1] * 13608362914817483670, tmp_5812[2] * 13608362914817483670];
    signal tmp_5865[3] <== [tmp_5864[0] + tmp_5831[0], tmp_5864[1] + tmp_5831[1], tmp_5864[2] + tmp_5831[2]];
    signal tmp_5866[3] <== [tmp_5863[0] + tmp_5865[0], tmp_5863[1] + tmp_5865[1], tmp_5863[2] + tmp_5865[2]];
    signal tmp_5867[3] <== [tmp_5815[0] * 18163707672964630459, tmp_5815[1] * 18163707672964630459, tmp_5815[2] * 18163707672964630459];
    signal tmp_5868[3] <== [tmp_5867[0] + tmp_5831[0], tmp_5867[1] + tmp_5831[1], tmp_5867[2] + tmp_5831[2]];
    signal tmp_5869[3] <== [tmp_5866[0] + tmp_5868[0], tmp_5866[1] + tmp_5868[1], tmp_5866[2] + tmp_5868[2]];
    signal tmp_5870[3] <== [tmp_5818[0] * 14373610220374016704, tmp_5818[1] * 14373610220374016704, tmp_5818[2] * 14373610220374016704];
    signal tmp_5871[3] <== [tmp_5870[0] + tmp_5831[0], tmp_5870[1] + tmp_5831[1], tmp_5870[2] + tmp_5831[2]];
    signal tmp_5872[3] <== [tmp_5869[0] + tmp_5871[0], tmp_5869[1] + tmp_5871[1], tmp_5869[2] + tmp_5871[2]];
    signal tmp_5873[3] <== [tmp_5821[0] * 6226282807566121054, tmp_5821[1] * 6226282807566121054, tmp_5821[2] * 6226282807566121054];
    signal tmp_5874[3] <== [tmp_5873[0] + tmp_5831[0], tmp_5873[1] + tmp_5831[1], tmp_5873[2] + tmp_5831[2]];
    signal tmp_5875[3] <== [tmp_5872[0] + tmp_5874[0], tmp_5872[1] + tmp_5874[1], tmp_5872[2] + tmp_5874[2]];
    signal tmp_5876[3] <== [tmp_5824[0] * 3643354756180461803, tmp_5824[1] * 3643354756180461803, tmp_5824[2] * 3643354756180461803];
    signal tmp_5877[3] <== [tmp_5876[0] + tmp_5831[0], tmp_5876[1] + tmp_5831[1], tmp_5876[2] + tmp_5831[2]];
    signal tmp_5878[3] <== [tmp_5875[0] + tmp_5877[0], tmp_5875[1] + tmp_5877[1], tmp_5875[2] + tmp_5877[2]];
    signal tmp_5879[3] <== [tmp_5827[0] * 13046961313070095543, tmp_5827[1] * 13046961313070095543, tmp_5827[2] * 13046961313070095543];
    signal tmp_5880[3] <== [tmp_5879[0] + tmp_5831[0], tmp_5879[1] + tmp_5831[1], tmp_5879[2] + tmp_5831[2]];
    signal tmp_5881[3] <== [tmp_5878[0] + tmp_5880[0], tmp_5878[1] + tmp_5880[1], tmp_5878[2] + tmp_5880[2]];
    signal tmp_5882[3] <== [tmp_5830[0] * 8594143216561850811, tmp_5830[1] * 8594143216561850811, tmp_5830[2] * 8594143216561850811];
    signal tmp_5883[3] <== [tmp_5882[0] + tmp_5831[0], tmp_5882[1] + tmp_5831[1], tmp_5882[2] + tmp_5831[2]];
    signal tmp_5884[3] <== [tmp_5881[0] + tmp_5883[0], tmp_5881[1] + tmp_5883[1], tmp_5881[2] + tmp_5883[2]];
    signal tmp_5885[3] <== [tmp_5839[0] + tmp_5884[0], tmp_5839[1] + tmp_5884[1], tmp_5839[2] + tmp_5884[2]];
    signal tmp_5886[3] <== [evals[137][0] - tmp_5885[0], evals[137][1] - tmp_5885[1], evals[137][2] - tmp_5885[2]];
    signal tmp_5887[3] <== CMul()(evals[76], tmp_5886);
    signal tmp_5888[3] <== [tmp_5836[0] + tmp_5887[0], tmp_5836[1] + tmp_5887[1], tmp_5836[2] + tmp_5887[2]];
    signal tmp_5889[3] <== CMul()(challengeQ, tmp_5888);
    signal tmp_5890[3] <== [evals[137][0] + tmp_4765[0], evals[137][1] + tmp_4765[1], evals[137][2] + tmp_4765[2]];
    signal tmp_5891[3] <== CMul()(tmp_4769, tmp_5890);
    signal tmp_5892[3] <== [tmp_5891[0] * 16040574633112940480, tmp_5891[1] * 16040574633112940480, tmp_5891[2] * 16040574633112940480];
    signal tmp_5893[3] <== [tmp_5841[0] * 14263299814608977431, tmp_5841[1] * 14263299814608977431, tmp_5841[2] * 14263299814608977431];
    signal tmp_5894[3] <== [tmp_5893[0] + tmp_5884[0], tmp_5893[1] + tmp_5884[1], tmp_5893[2] + tmp_5884[2]];
    signal tmp_5895[3] <== [tmp_5891[0] + tmp_5894[0], tmp_5891[1] + tmp_5894[1], tmp_5891[2] + tmp_5894[2]];
    signal tmp_5896[3] <== [tmp_5844[0] * 770395855193680981, tmp_5844[1] * 770395855193680981, tmp_5844[2] * 770395855193680981];
    signal tmp_5897[3] <== [tmp_5896[0] + tmp_5884[0], tmp_5896[1] + tmp_5884[1], tmp_5896[2] + tmp_5884[2]];
    signal tmp_5898[3] <== [tmp_5895[0] + tmp_5897[0], tmp_5895[1] + tmp_5897[1], tmp_5895[2] + tmp_5897[2]];
    signal tmp_5899[3] <== [tmp_5847[0] * 3459277367440070515, tmp_5847[1] * 3459277367440070515, tmp_5847[2] * 3459277367440070515];
    signal tmp_5900[3] <== [tmp_5899[0] + tmp_5884[0], tmp_5899[1] + tmp_5884[1], tmp_5899[2] + tmp_5884[2]];
    signal tmp_5901[3] <== [tmp_5898[0] + tmp_5900[0], tmp_5898[1] + tmp_5900[1], tmp_5898[2] + tmp_5900[2]];
    signal tmp_5902[3] <== [tmp_5850[0] * 17087697094293314027, tmp_5850[1] * 17087697094293314027, tmp_5850[2] * 17087697094293314027];
    signal tmp_5903[3] <== [tmp_5902[0] + tmp_5884[0], tmp_5902[1] + tmp_5884[1], tmp_5902[2] + tmp_5884[2]];
    signal tmp_5904[3] <== [tmp_5901[0] + tmp_5903[0], tmp_5901[1] + tmp_5903[1], tmp_5901[2] + tmp_5903[2]];
    signal tmp_5905[3] <== [tmp_5853[0] * 6694380135428747348, tmp_5853[1] * 6694380135428747348, tmp_5853[2] * 6694380135428747348];
    signal tmp_5906[3] <== [tmp_5905[0] + tmp_5884[0], tmp_5905[1] + tmp_5884[1], tmp_5905[2] + tmp_5884[2]];
    signal tmp_5907[3] <== [tmp_5904[0] + tmp_5906[0], tmp_5904[1] + tmp_5906[1], tmp_5904[2] + tmp_5906[2]];
    signal tmp_5908[3] <== [tmp_5856[0] * 2034408310088972836, tmp_5856[1] * 2034408310088972836, tmp_5856[2] * 2034408310088972836];
    signal tmp_5909[3] <== [tmp_5908[0] + tmp_5884[0], tmp_5908[1] + tmp_5884[1], tmp_5908[2] + tmp_5884[2]];
    signal tmp_5910[3] <== [tmp_5907[0] + tmp_5909[0], tmp_5907[1] + tmp_5909[1], tmp_5907[2] + tmp_5909[2]];
    signal tmp_5911[3] <== [tmp_5859[0] * 3434575637390274478, tmp_5859[1] * 3434575637390274478, tmp_5859[2] * 3434575637390274478];
    signal tmp_5912[3] <== [tmp_5911[0] + tmp_5884[0], tmp_5911[1] + tmp_5884[1], tmp_5911[2] + tmp_5884[2]];
    signal tmp_5913[3] <== [tmp_5910[0] + tmp_5912[0], tmp_5910[1] + tmp_5912[1], tmp_5910[2] + tmp_5912[2]];
    signal tmp_5914[3] <== [tmp_5862[0] * 6052753985947965968, tmp_5862[1] * 6052753985947965968, tmp_5862[2] * 6052753985947965968];
    signal tmp_5915[3] <== [tmp_5914[0] + tmp_5884[0], tmp_5914[1] + tmp_5884[1], tmp_5914[2] + tmp_5884[2]];
    signal tmp_5916[3] <== [tmp_5913[0] + tmp_5915[0], tmp_5913[1] + tmp_5915[1], tmp_5913[2] + tmp_5915[2]];
    signal tmp_5917[3] <== [tmp_5865[0] * 13608362914817483670, tmp_5865[1] * 13608362914817483670, tmp_5865[2] * 13608362914817483670];
    signal tmp_5918[3] <== [tmp_5917[0] + tmp_5884[0], tmp_5917[1] + tmp_5884[1], tmp_5917[2] + tmp_5884[2]];
    signal tmp_5919[3] <== [tmp_5916[0] + tmp_5918[0], tmp_5916[1] + tmp_5918[1], tmp_5916[2] + tmp_5918[2]];
    signal tmp_5920[3] <== [tmp_5868[0] * 18163707672964630459, tmp_5868[1] * 18163707672964630459, tmp_5868[2] * 18163707672964630459];
    signal tmp_5921[3] <== [tmp_5920[0] + tmp_5884[0], tmp_5920[1] + tmp_5884[1], tmp_5920[2] + tmp_5884[2]];
    signal tmp_5922[3] <== [tmp_5919[0] + tmp_5921[0], tmp_5919[1] + tmp_5921[1], tmp_5919[2] + tmp_5921[2]];
    signal tmp_5923[3] <== [tmp_5871[0] * 14373610220374016704, tmp_5871[1] * 14373610220374016704, tmp_5871[2] * 14373610220374016704];
    signal tmp_5924[3] <== [tmp_5923[0] + tmp_5884[0], tmp_5923[1] + tmp_5884[1], tmp_5923[2] + tmp_5884[2]];
    signal tmp_5925[3] <== [tmp_5922[0] + tmp_5924[0], tmp_5922[1] + tmp_5924[1], tmp_5922[2] + tmp_5924[2]];
    signal tmp_5926[3] <== [tmp_5874[0] * 6226282807566121054, tmp_5874[1] * 6226282807566121054, tmp_5874[2] * 6226282807566121054];
    signal tmp_5927[3] <== [tmp_5926[0] + tmp_5884[0], tmp_5926[1] + tmp_5884[1], tmp_5926[2] + tmp_5884[2]];
    signal tmp_5928[3] <== [tmp_5925[0] + tmp_5927[0], tmp_5925[1] + tmp_5927[1], tmp_5925[2] + tmp_5927[2]];
    signal tmp_5929[3] <== [tmp_5877[0] * 3643354756180461803, tmp_5877[1] * 3643354756180461803, tmp_5877[2] * 3643354756180461803];
    signal tmp_5930[3] <== [tmp_5929[0] + tmp_5884[0], tmp_5929[1] + tmp_5884[1], tmp_5929[2] + tmp_5884[2]];
    signal tmp_5931[3] <== [tmp_5928[0] + tmp_5930[0], tmp_5928[1] + tmp_5930[1], tmp_5928[2] + tmp_5930[2]];
    signal tmp_5932[3] <== [tmp_5880[0] * 13046961313070095543, tmp_5880[1] * 13046961313070095543, tmp_5880[2] * 13046961313070095543];
    signal tmp_5933[3] <== [tmp_5932[0] + tmp_5884[0], tmp_5932[1] + tmp_5884[1], tmp_5932[2] + tmp_5884[2]];
    signal tmp_5934[3] <== [tmp_5931[0] + tmp_5933[0], tmp_5931[1] + tmp_5933[1], tmp_5931[2] + tmp_5933[2]];
    signal tmp_5935[3] <== [tmp_5883[0] * 8594143216561850811, tmp_5883[1] * 8594143216561850811, tmp_5883[2] * 8594143216561850811];
    signal tmp_5936[3] <== [tmp_5935[0] + tmp_5884[0], tmp_5935[1] + tmp_5884[1], tmp_5935[2] + tmp_5884[2]];
    signal tmp_5937[3] <== [tmp_5934[0] + tmp_5936[0], tmp_5934[1] + tmp_5936[1], tmp_5934[2] + tmp_5936[2]];
    signal tmp_5938[3] <== [tmp_5892[0] + tmp_5937[0], tmp_5892[1] + tmp_5937[1], tmp_5892[2] + tmp_5937[2]];
    signal tmp_5939[3] <== [evals[22][0] - tmp_5938[0], evals[22][1] - tmp_5938[1], evals[22][2] - tmp_5938[2]];
    signal tmp_5940[3] <== CMul()(evals[76], tmp_5939);
    signal tmp_5941[3] <== [tmp_5889[0] + tmp_5940[0], tmp_5889[1] + tmp_5940[1], tmp_5889[2] + tmp_5940[2]];
    signal tmp_5942[3] <== CMul()(challengeQ, tmp_5941);
    signal tmp_5943[3] <== [evals[22][0] + 17869527288639567155, evals[22][1], evals[22][2]];
    signal tmp_5944[3] <== [evals[22][0] + 17869527288639567155, evals[22][1], evals[22][2]];
    tmp_5945 <== CMul()(tmp_5943, tmp_5944);
    signal tmp_5946[3] <== CMul()(evals[163], tmp_5945);
    signal tmp_5947[3] <== [evals[22][0] + 17869527288639567155, evals[22][1], evals[22][2]];
    signal tmp_5948[3] <== CMul()(tmp_5946, tmp_5947);
    signal tmp_5949[3] <== [tmp_5948[0] * 16040574633112940480, tmp_5948[1] * 16040574633112940480, tmp_5948[2] * 16040574633112940480];
    signal tmp_5950[3] <== [evals[22][0] + 17869527288639567155, evals[22][1], evals[22][2]];
    signal tmp_5951[3] <== CMul()(tmp_5946, tmp_5950);
    signal tmp_5952[3] <== [tmp_5894[0] * 14263299814608977431, tmp_5894[1] * 14263299814608977431, tmp_5894[2] * 14263299814608977431];
    signal tmp_5953[3] <== [tmp_5952[0] + tmp_5937[0], tmp_5952[1] + tmp_5937[1], tmp_5952[2] + tmp_5937[2]];
    signal tmp_5954[3] <== [tmp_5951[0] + tmp_5953[0], tmp_5951[1] + tmp_5953[1], tmp_5951[2] + tmp_5953[2]];
    signal tmp_5955[3] <== [tmp_5897[0] * 770395855193680981, tmp_5897[1] * 770395855193680981, tmp_5897[2] * 770395855193680981];
    signal tmp_5956[3] <== [tmp_5955[0] + tmp_5937[0], tmp_5955[1] + tmp_5937[1], tmp_5955[2] + tmp_5937[2]];
    signal tmp_5957[3] <== [tmp_5954[0] + tmp_5956[0], tmp_5954[1] + tmp_5956[1], tmp_5954[2] + tmp_5956[2]];
    signal tmp_5958[3] <== [tmp_5900[0] * 3459277367440070515, tmp_5900[1] * 3459277367440070515, tmp_5900[2] * 3459277367440070515];
    signal tmp_5959[3] <== [tmp_5958[0] + tmp_5937[0], tmp_5958[1] + tmp_5937[1], tmp_5958[2] + tmp_5937[2]];
    signal tmp_5960[3] <== [tmp_5957[0] + tmp_5959[0], tmp_5957[1] + tmp_5959[1], tmp_5957[2] + tmp_5959[2]];
    signal tmp_5961[3] <== [tmp_5903[0] * 17087697094293314027, tmp_5903[1] * 17087697094293314027, tmp_5903[2] * 17087697094293314027];
    signal tmp_5962[3] <== [tmp_5961[0] + tmp_5937[0], tmp_5961[1] + tmp_5937[1], tmp_5961[2] + tmp_5937[2]];
    signal tmp_5963[3] <== [tmp_5960[0] + tmp_5962[0], tmp_5960[1] + tmp_5962[1], tmp_5960[2] + tmp_5962[2]];
    signal tmp_5964[3] <== [tmp_5906[0] * 6694380135428747348, tmp_5906[1] * 6694380135428747348, tmp_5906[2] * 6694380135428747348];
    signal tmp_5965[3] <== [tmp_5964[0] + tmp_5937[0], tmp_5964[1] + tmp_5937[1], tmp_5964[2] + tmp_5937[2]];
    signal tmp_5966[3] <== [tmp_5963[0] + tmp_5965[0], tmp_5963[1] + tmp_5965[1], tmp_5963[2] + tmp_5965[2]];
    signal tmp_5967[3] <== [tmp_5909[0] * 2034408310088972836, tmp_5909[1] * 2034408310088972836, tmp_5909[2] * 2034408310088972836];
    signal tmp_5968[3] <== [tmp_5967[0] + tmp_5937[0], tmp_5967[1] + tmp_5937[1], tmp_5967[2] + tmp_5937[2]];
    signal tmp_5969[3] <== [tmp_5966[0] + tmp_5968[0], tmp_5966[1] + tmp_5968[1], tmp_5966[2] + tmp_5968[2]];
    signal tmp_5970[3] <== [tmp_5912[0] * 3434575637390274478, tmp_5912[1] * 3434575637390274478, tmp_5912[2] * 3434575637390274478];
    signal tmp_5971[3] <== [tmp_5970[0] + tmp_5937[0], tmp_5970[1] + tmp_5937[1], tmp_5970[2] + tmp_5937[2]];
    signal tmp_5972[3] <== [tmp_5969[0] + tmp_5971[0], tmp_5969[1] + tmp_5971[1], tmp_5969[2] + tmp_5971[2]];
    signal tmp_5973[3] <== [tmp_5915[0] * 6052753985947965968, tmp_5915[1] * 6052753985947965968, tmp_5915[2] * 6052753985947965968];
    signal tmp_5974[3] <== [tmp_5973[0] + tmp_5937[0], tmp_5973[1] + tmp_5937[1], tmp_5973[2] + tmp_5937[2]];
    signal tmp_5975[3] <== [tmp_5972[0] + tmp_5974[0], tmp_5972[1] + tmp_5974[1], tmp_5972[2] + tmp_5974[2]];
    signal tmp_5976[3] <== [tmp_5918[0] * 13608362914817483670, tmp_5918[1] * 13608362914817483670, tmp_5918[2] * 13608362914817483670];
    signal tmp_5977[3] <== [tmp_5976[0] + tmp_5937[0], tmp_5976[1] + tmp_5937[1], tmp_5976[2] + tmp_5937[2]];
    signal tmp_5978[3] <== [tmp_5975[0] + tmp_5977[0], tmp_5975[1] + tmp_5977[1], tmp_5975[2] + tmp_5977[2]];
    signal tmp_5979[3] <== [tmp_5921[0] * 18163707672964630459, tmp_5921[1] * 18163707672964630459, tmp_5921[2] * 18163707672964630459];
    signal tmp_5980[3] <== [tmp_5979[0] + tmp_5937[0], tmp_5979[1] + tmp_5937[1], tmp_5979[2] + tmp_5937[2]];
    signal tmp_5981[3] <== [tmp_5978[0] + tmp_5980[0], tmp_5978[1] + tmp_5980[1], tmp_5978[2] + tmp_5980[2]];
    signal tmp_5982[3] <== [tmp_5924[0] * 14373610220374016704, tmp_5924[1] * 14373610220374016704, tmp_5924[2] * 14373610220374016704];
    signal tmp_5983[3] <== [tmp_5982[0] + tmp_5937[0], tmp_5982[1] + tmp_5937[1], tmp_5982[2] + tmp_5937[2]];
    signal tmp_5984[3] <== [tmp_5981[0] + tmp_5983[0], tmp_5981[1] + tmp_5983[1], tmp_5981[2] + tmp_5983[2]];
    signal tmp_5985[3] <== [tmp_5927[0] * 6226282807566121054, tmp_5927[1] * 6226282807566121054, tmp_5927[2] * 6226282807566121054];
    signal tmp_5986[3] <== [tmp_5985[0] + tmp_5937[0], tmp_5985[1] + tmp_5937[1], tmp_5985[2] + tmp_5937[2]];
    signal tmp_5987[3] <== [tmp_5984[0] + tmp_5986[0], tmp_5984[1] + tmp_5986[1], tmp_5984[2] + tmp_5986[2]];
    signal tmp_5988[3] <== [tmp_5930[0] * 3643354756180461803, tmp_5930[1] * 3643354756180461803, tmp_5930[2] * 3643354756180461803];
    signal tmp_5989[3] <== [tmp_5988[0] + tmp_5937[0], tmp_5988[1] + tmp_5937[1], tmp_5988[2] + tmp_5937[2]];
    signal tmp_5990[3] <== [tmp_5987[0] + tmp_5989[0], tmp_5987[1] + tmp_5989[1], tmp_5987[2] + tmp_5989[2]];
    signal tmp_5991[3] <== [tmp_5933[0] * 13046961313070095543, tmp_5933[1] * 13046961313070095543, tmp_5933[2] * 13046961313070095543];
    signal tmp_5992[3] <== [tmp_5991[0] + tmp_5937[0], tmp_5991[1] + tmp_5937[1], tmp_5991[2] + tmp_5937[2]];
    signal tmp_5993[3] <== [tmp_5990[0] + tmp_5992[0], tmp_5990[1] + tmp_5992[1], tmp_5990[2] + tmp_5992[2]];
    signal tmp_5994[3] <== [tmp_5936[0] * 8594143216561850811, tmp_5936[1] * 8594143216561850811, tmp_5936[2] * 8594143216561850811];
    signal tmp_5995[3] <== [tmp_5994[0] + tmp_5937[0], tmp_5994[1] + tmp_5937[1], tmp_5994[2] + tmp_5937[2]];
    signal tmp_5996[3] <== [tmp_5993[0] + tmp_5995[0], tmp_5993[1] + tmp_5995[1], tmp_5993[2] + tmp_5995[2]];
    signal tmp_5997[3] <== [tmp_5949[0] + tmp_5996[0], tmp_5949[1] + tmp_5996[1], tmp_5949[2] + tmp_5996[2]];
    signal tmp_5998[3] <== [evals[23][0] - tmp_5997[0], evals[23][1] - tmp_5997[1], evals[23][2] - tmp_5997[2]];
    signal tmp_5999[3] <== CMul()(evals[76], tmp_5998);
    signal tmp_6000[3] <== [tmp_5942[0] + tmp_5999[0], tmp_5942[1] + tmp_5999[1], tmp_5942[2] + tmp_5999[2]];
    signal tmp_6001[3] <== CMul()(challengeQ, tmp_6000);
    signal tmp_6002[3] <== [evals[23][0] + 7829055113315023688, evals[23][1], evals[23][2]];
    signal tmp_6003[3] <== [evals[23][0] + 7829055113315023688, evals[23][1], evals[23][2]];
    tmp_6004 <== CMul()(tmp_6002, tmp_6003);
    signal tmp_6005[3] <== CMul()(evals[164], tmp_6004);
    signal tmp_6006[3] <== [evals[23][0] + 7829055113315023688, evals[23][1], evals[23][2]];
    signal tmp_6007[3] <== CMul()(tmp_6005, tmp_6006);
    signal tmp_6008[3] <== [tmp_6007[0] * 16040574633112940480, tmp_6007[1] * 16040574633112940480, tmp_6007[2] * 16040574633112940480];
    signal tmp_6009[3] <== [evals[23][0] + 7829055113315023688, evals[23][1], evals[23][2]];
    signal tmp_6010[3] <== CMul()(tmp_6005, tmp_6009);
    signal tmp_6011[3] <== [tmp_5953[0] * 14263299814608977431, tmp_5953[1] * 14263299814608977431, tmp_5953[2] * 14263299814608977431];
    signal tmp_6012[3] <== [tmp_6011[0] + tmp_5996[0], tmp_6011[1] + tmp_5996[1], tmp_6011[2] + tmp_5996[2]];
    signal tmp_6013[3] <== [tmp_6010[0] + tmp_6012[0], tmp_6010[1] + tmp_6012[1], tmp_6010[2] + tmp_6012[2]];
    signal tmp_6014[3] <== [tmp_5956[0] * 770395855193680981, tmp_5956[1] * 770395855193680981, tmp_5956[2] * 770395855193680981];
    signal tmp_6015[3] <== [tmp_6014[0] + tmp_5996[0], tmp_6014[1] + tmp_5996[1], tmp_6014[2] + tmp_5996[2]];
    signal tmp_6016[3] <== [tmp_6013[0] + tmp_6015[0], tmp_6013[1] + tmp_6015[1], tmp_6013[2] + tmp_6015[2]];
    signal tmp_6017[3] <== [tmp_5959[0] * 3459277367440070515, tmp_5959[1] * 3459277367440070515, tmp_5959[2] * 3459277367440070515];
    signal tmp_6018[3] <== [tmp_6017[0] + tmp_5996[0], tmp_6017[1] + tmp_5996[1], tmp_6017[2] + tmp_5996[2]];
    signal tmp_6019[3] <== [tmp_6016[0] + tmp_6018[0], tmp_6016[1] + tmp_6018[1], tmp_6016[2] + tmp_6018[2]];
    signal tmp_6020[3] <== [tmp_5962[0] * 17087697094293314027, tmp_5962[1] * 17087697094293314027, tmp_5962[2] * 17087697094293314027];
    signal tmp_6021[3] <== [tmp_6020[0] + tmp_5996[0], tmp_6020[1] + tmp_5996[1], tmp_6020[2] + tmp_5996[2]];
    signal tmp_6022[3] <== [tmp_6019[0] + tmp_6021[0], tmp_6019[1] + tmp_6021[1], tmp_6019[2] + tmp_6021[2]];
    signal tmp_6023[3] <== [tmp_5965[0] * 6694380135428747348, tmp_5965[1] * 6694380135428747348, tmp_5965[2] * 6694380135428747348];
    signal tmp_6024[3] <== [tmp_6023[0] + tmp_5996[0], tmp_6023[1] + tmp_5996[1], tmp_6023[2] + tmp_5996[2]];
    signal tmp_6025[3] <== [tmp_6022[0] + tmp_6024[0], tmp_6022[1] + tmp_6024[1], tmp_6022[2] + tmp_6024[2]];
    signal tmp_6026[3] <== [tmp_5968[0] * 2034408310088972836, tmp_5968[1] * 2034408310088972836, tmp_5968[2] * 2034408310088972836];
    signal tmp_6027[3] <== [tmp_6026[0] + tmp_5996[0], tmp_6026[1] + tmp_5996[1], tmp_6026[2] + tmp_5996[2]];
    signal tmp_6028[3] <== [tmp_6025[0] + tmp_6027[0], tmp_6025[1] + tmp_6027[1], tmp_6025[2] + tmp_6027[2]];
    signal tmp_6029[3] <== [tmp_5971[0] * 3434575637390274478, tmp_5971[1] * 3434575637390274478, tmp_5971[2] * 3434575637390274478];
    signal tmp_6030[3] <== [tmp_6029[0] + tmp_5996[0], tmp_6029[1] + tmp_5996[1], tmp_6029[2] + tmp_5996[2]];
    signal tmp_6031[3] <== [tmp_6028[0] + tmp_6030[0], tmp_6028[1] + tmp_6030[1], tmp_6028[2] + tmp_6030[2]];
    signal tmp_6032[3] <== [tmp_5974[0] * 6052753985947965968, tmp_5974[1] * 6052753985947965968, tmp_5974[2] * 6052753985947965968];
    signal tmp_6033[3] <== [tmp_6032[0] + tmp_5996[0], tmp_6032[1] + tmp_5996[1], tmp_6032[2] + tmp_5996[2]];
    signal tmp_6034[3] <== [tmp_6031[0] + tmp_6033[0], tmp_6031[1] + tmp_6033[1], tmp_6031[2] + tmp_6033[2]];
    signal tmp_6035[3] <== [tmp_5977[0] * 13608362914817483670, tmp_5977[1] * 13608362914817483670, tmp_5977[2] * 13608362914817483670];
    signal tmp_6036[3] <== [tmp_6035[0] + tmp_5996[0], tmp_6035[1] + tmp_5996[1], tmp_6035[2] + tmp_5996[2]];
    signal tmp_6037[3] <== [tmp_6034[0] + tmp_6036[0], tmp_6034[1] + tmp_6036[1], tmp_6034[2] + tmp_6036[2]];
    signal tmp_6038[3] <== [tmp_5980[0] * 18163707672964630459, tmp_5980[1] * 18163707672964630459, tmp_5980[2] * 18163707672964630459];
    signal tmp_6039[3] <== [tmp_6038[0] + tmp_5996[0], tmp_6038[1] + tmp_5996[1], tmp_6038[2] + tmp_5996[2]];
    signal tmp_6040[3] <== [tmp_6037[0] + tmp_6039[0], tmp_6037[1] + tmp_6039[1], tmp_6037[2] + tmp_6039[2]];
    signal tmp_6041[3] <== [tmp_5983[0] * 14373610220374016704, tmp_5983[1] * 14373610220374016704, tmp_5983[2] * 14373610220374016704];
    signal tmp_6042[3] <== [tmp_6041[0] + tmp_5996[0], tmp_6041[1] + tmp_5996[1], tmp_6041[2] + tmp_5996[2]];
    signal tmp_6043[3] <== [tmp_6040[0] + tmp_6042[0], tmp_6040[1] + tmp_6042[1], tmp_6040[2] + tmp_6042[2]];
    signal tmp_6044[3] <== [tmp_5986[0] * 6226282807566121054, tmp_5986[1] * 6226282807566121054, tmp_5986[2] * 6226282807566121054];
    signal tmp_6045[3] <== [tmp_6044[0] + tmp_5996[0], tmp_6044[1] + tmp_5996[1], tmp_6044[2] + tmp_5996[2]];
    signal tmp_6046[3] <== [tmp_6043[0] + tmp_6045[0], tmp_6043[1] + tmp_6045[1], tmp_6043[2] + tmp_6045[2]];
    signal tmp_6047[3] <== [tmp_5989[0] * 3643354756180461803, tmp_5989[1] * 3643354756180461803, tmp_5989[2] * 3643354756180461803];
    signal tmp_6048[3] <== [tmp_6047[0] + tmp_5996[0], tmp_6047[1] + tmp_5996[1], tmp_6047[2] + tmp_5996[2]];
    signal tmp_6049[3] <== [tmp_6046[0] + tmp_6048[0], tmp_6046[1] + tmp_6048[1], tmp_6046[2] + tmp_6048[2]];
    signal tmp_6050[3] <== [tmp_5992[0] * 13046961313070095543, tmp_5992[1] * 13046961313070095543, tmp_5992[2] * 13046961313070095543];
    signal tmp_6051[3] <== [tmp_6050[0] + tmp_5996[0], tmp_6050[1] + tmp_5996[1], tmp_6050[2] + tmp_5996[2]];
    signal tmp_6052[3] <== [tmp_6049[0] + tmp_6051[0], tmp_6049[1] + tmp_6051[1], tmp_6049[2] + tmp_6051[2]];
    signal tmp_6053[3] <== [tmp_5995[0] * 8594143216561850811, tmp_5995[1] * 8594143216561850811, tmp_5995[2] * 8594143216561850811];
    signal tmp_6054[3] <== [tmp_6053[0] + tmp_5996[0], tmp_6053[1] + tmp_5996[1], tmp_6053[2] + tmp_5996[2]];
    signal tmp_6055[3] <== [tmp_6052[0] + tmp_6054[0], tmp_6052[1] + tmp_6054[1], tmp_6052[2] + tmp_6054[2]];
    signal tmp_6056[3] <== [tmp_6008[0] + tmp_6055[0], tmp_6008[1] + tmp_6055[1], tmp_6008[2] + tmp_6055[2]];
    signal tmp_6057[3] <== [evals[24][0] - tmp_6056[0], evals[24][1] - tmp_6056[1], evals[24][2] - tmp_6056[2]];
    signal tmp_6058[3] <== CMul()(evals[76], tmp_6057);
    signal tmp_6059[3] <== [tmp_6001[0] + tmp_6058[0], tmp_6001[1] + tmp_6058[1], tmp_6001[2] + tmp_6058[2]];
    signal tmp_6060[3] <== CMul()(challengeQ, tmp_6059);
    signal tmp_6061[3] <== [evals[24][0] + 3256047469251174543, evals[24][1], evals[24][2]];
    signal tmp_6062[3] <== [evals[24][0] + 3256047469251174543, evals[24][1], evals[24][2]];
    tmp_6063 <== CMul()(tmp_6061, tmp_6062);
    signal tmp_6064[3] <== CMul()(evals[165], tmp_6063);
    signal tmp_6065[3] <== [evals[24][0] + 3256047469251174543, evals[24][1], evals[24][2]];
    signal tmp_6066[3] <== CMul()(tmp_6064, tmp_6065);
    signal tmp_6067[3] <== [tmp_6066[0] * 16040574633112940480, tmp_6066[1] * 16040574633112940480, tmp_6066[2] * 16040574633112940480];
    signal tmp_6068[3] <== [evals[24][0] + 3256047469251174543, evals[24][1], evals[24][2]];
    signal tmp_6069[3] <== CMul()(tmp_6064, tmp_6068);
    signal tmp_6070[3] <== [tmp_6012[0] * 14263299814608977431, tmp_6012[1] * 14263299814608977431, tmp_6012[2] * 14263299814608977431];
    signal tmp_6071[3] <== [tmp_6070[0] + tmp_6055[0], tmp_6070[1] + tmp_6055[1], tmp_6070[2] + tmp_6055[2]];
    signal tmp_6072[3] <== [tmp_6069[0] + tmp_6071[0], tmp_6069[1] + tmp_6071[1], tmp_6069[2] + tmp_6071[2]];
    signal tmp_6073[3] <== [tmp_6015[0] * 770395855193680981, tmp_6015[1] * 770395855193680981, tmp_6015[2] * 770395855193680981];
    signal tmp_6074[3] <== [tmp_6073[0] + tmp_6055[0], tmp_6073[1] + tmp_6055[1], tmp_6073[2] + tmp_6055[2]];
    signal tmp_6075[3] <== [tmp_6072[0] + tmp_6074[0], tmp_6072[1] + tmp_6074[1], tmp_6072[2] + tmp_6074[2]];
    signal tmp_6076[3] <== [tmp_6018[0] * 3459277367440070515, tmp_6018[1] * 3459277367440070515, tmp_6018[2] * 3459277367440070515];
    signal tmp_6077[3] <== [tmp_6076[0] + tmp_6055[0], tmp_6076[1] + tmp_6055[1], tmp_6076[2] + tmp_6055[2]];
    signal tmp_6078[3] <== [tmp_6075[0] + tmp_6077[0], tmp_6075[1] + tmp_6077[1], tmp_6075[2] + tmp_6077[2]];
    signal tmp_6079[3] <== [tmp_6021[0] * 17087697094293314027, tmp_6021[1] * 17087697094293314027, tmp_6021[2] * 17087697094293314027];
    signal tmp_6080[3] <== [tmp_6079[0] + tmp_6055[0], tmp_6079[1] + tmp_6055[1], tmp_6079[2] + tmp_6055[2]];
    signal tmp_6081[3] <== [tmp_6078[0] + tmp_6080[0], tmp_6078[1] + tmp_6080[1], tmp_6078[2] + tmp_6080[2]];
    signal tmp_6082[3] <== [tmp_6024[0] * 6694380135428747348, tmp_6024[1] * 6694380135428747348, tmp_6024[2] * 6694380135428747348];
    signal tmp_6083[3] <== [tmp_6082[0] + tmp_6055[0], tmp_6082[1] + tmp_6055[1], tmp_6082[2] + tmp_6055[2]];
    signal tmp_6084[3] <== [tmp_6081[0] + tmp_6083[0], tmp_6081[1] + tmp_6083[1], tmp_6081[2] + tmp_6083[2]];
    signal tmp_6085[3] <== [tmp_6027[0] * 2034408310088972836, tmp_6027[1] * 2034408310088972836, tmp_6027[2] * 2034408310088972836];
    signal tmp_6086[3] <== [tmp_6085[0] + tmp_6055[0], tmp_6085[1] + tmp_6055[1], tmp_6085[2] + tmp_6055[2]];
    signal tmp_6087[3] <== [tmp_6084[0] + tmp_6086[0], tmp_6084[1] + tmp_6086[1], tmp_6084[2] + tmp_6086[2]];
    signal tmp_6088[3] <== [tmp_6030[0] * 3434575637390274478, tmp_6030[1] * 3434575637390274478, tmp_6030[2] * 3434575637390274478];
    signal tmp_6089[3] <== [tmp_6088[0] + tmp_6055[0], tmp_6088[1] + tmp_6055[1], tmp_6088[2] + tmp_6055[2]];
    signal tmp_6090[3] <== [tmp_6087[0] + tmp_6089[0], tmp_6087[1] + tmp_6089[1], tmp_6087[2] + tmp_6089[2]];
    signal tmp_6091[3] <== [tmp_6033[0] * 6052753985947965968, tmp_6033[1] * 6052753985947965968, tmp_6033[2] * 6052753985947965968];
    signal tmp_6092[3] <== [tmp_6091[0] + tmp_6055[0], tmp_6091[1] + tmp_6055[1], tmp_6091[2] + tmp_6055[2]];
    signal tmp_6093[3] <== [tmp_6090[0] + tmp_6092[0], tmp_6090[1] + tmp_6092[1], tmp_6090[2] + tmp_6092[2]];
    signal tmp_6094[3] <== [tmp_6036[0] * 13608362914817483670, tmp_6036[1] * 13608362914817483670, tmp_6036[2] * 13608362914817483670];
    signal tmp_6095[3] <== [tmp_6094[0] + tmp_6055[0], tmp_6094[1] + tmp_6055[1], tmp_6094[2] + tmp_6055[2]];
    signal tmp_6096[3] <== [tmp_6093[0] + tmp_6095[0], tmp_6093[1] + tmp_6095[1], tmp_6093[2] + tmp_6095[2]];
    signal tmp_6097[3] <== [tmp_6039[0] * 18163707672964630459, tmp_6039[1] * 18163707672964630459, tmp_6039[2] * 18163707672964630459];
    signal tmp_6098[3] <== [tmp_6097[0] + tmp_6055[0], tmp_6097[1] + tmp_6055[1], tmp_6097[2] + tmp_6055[2]];
    signal tmp_6099[3] <== [tmp_6096[0] + tmp_6098[0], tmp_6096[1] + tmp_6098[1], tmp_6096[2] + tmp_6098[2]];
    signal tmp_6100[3] <== [tmp_6042[0] * 14373610220374016704, tmp_6042[1] * 14373610220374016704, tmp_6042[2] * 14373610220374016704];
    signal tmp_6101[3] <== [tmp_6100[0] + tmp_6055[0], tmp_6100[1] + tmp_6055[1], tmp_6100[2] + tmp_6055[2]];
    signal tmp_6102[3] <== [tmp_6099[0] + tmp_6101[0], tmp_6099[1] + tmp_6101[1], tmp_6099[2] + tmp_6101[2]];
    signal tmp_6103[3] <== [tmp_6045[0] * 6226282807566121054, tmp_6045[1] * 6226282807566121054, tmp_6045[2] * 6226282807566121054];
    signal tmp_6104[3] <== [tmp_6103[0] + tmp_6055[0], tmp_6103[1] + tmp_6055[1], tmp_6103[2] + tmp_6055[2]];
    signal tmp_6105[3] <== [tmp_6102[0] + tmp_6104[0], tmp_6102[1] + tmp_6104[1], tmp_6102[2] + tmp_6104[2]];
    signal tmp_6106[3] <== [tmp_6048[0] * 3643354756180461803, tmp_6048[1] * 3643354756180461803, tmp_6048[2] * 3643354756180461803];
    signal tmp_6107[3] <== [tmp_6106[0] + tmp_6055[0], tmp_6106[1] + tmp_6055[1], tmp_6106[2] + tmp_6055[2]];
    signal tmp_6108[3] <== [tmp_6105[0] + tmp_6107[0], tmp_6105[1] + tmp_6107[1], tmp_6105[2] + tmp_6107[2]];
    signal tmp_6109[3] <== [tmp_6051[0] * 13046961313070095543, tmp_6051[1] * 13046961313070095543, tmp_6051[2] * 13046961313070095543];
    signal tmp_6110[3] <== [tmp_6109[0] + tmp_6055[0], tmp_6109[1] + tmp_6055[1], tmp_6109[2] + tmp_6055[2]];
    signal tmp_6111[3] <== [tmp_6108[0] + tmp_6110[0], tmp_6108[1] + tmp_6110[1], tmp_6108[2] + tmp_6110[2]];
    signal tmp_6112[3] <== [tmp_6054[0] * 8594143216561850811, tmp_6054[1] * 8594143216561850811, tmp_6054[2] * 8594143216561850811];
    signal tmp_6113[3] <== [tmp_6112[0] + tmp_6055[0], tmp_6112[1] + tmp_6055[1], tmp_6112[2] + tmp_6055[2]];
    signal tmp_6114[3] <== [tmp_6111[0] + tmp_6113[0], tmp_6111[1] + tmp_6113[1], tmp_6111[2] + tmp_6113[2]];
    signal tmp_6115[3] <== [tmp_6067[0] + tmp_6114[0], tmp_6067[1] + tmp_6114[1], tmp_6067[2] + tmp_6114[2]];
    signal tmp_6116[3] <== [evals[25][0] - tmp_6115[0], evals[25][1] - tmp_6115[1], evals[25][2] - tmp_6115[2]];
    signal tmp_6117[3] <== CMul()(evals[76], tmp_6116);
    signal tmp_6118[3] <== [tmp_6060[0] + tmp_6117[0], tmp_6060[1] + tmp_6117[1], tmp_6060[2] + tmp_6117[2]];
    signal tmp_6119[3] <== CMul()(challengeQ, tmp_6118);
    signal tmp_6120[3] <== [evals[25][0] + 3015723851705964382, evals[25][1], evals[25][2]];
    signal tmp_6121[3] <== [evals[25][0] + 3015723851705964382, evals[25][1], evals[25][2]];
    tmp_6122 <== CMul()(tmp_6120, tmp_6121);
    signal tmp_6123[3] <== CMul()(evals[166], tmp_6122);
    signal tmp_6124[3] <== [evals[25][0] + 3015723851705964382, evals[25][1], evals[25][2]];
    signal tmp_6125[3] <== CMul()(tmp_6123, tmp_6124);
    signal tmp_6126[3] <== [tmp_6125[0] * 16040574633112940480, tmp_6125[1] * 16040574633112940480, tmp_6125[2] * 16040574633112940480];
    signal tmp_6127[3] <== [evals[25][0] + 3015723851705964382, evals[25][1], evals[25][2]];
    signal tmp_6128[3] <== CMul()(tmp_6123, tmp_6127);
    signal tmp_6129[3] <== [tmp_6071[0] * 14263299814608977431, tmp_6071[1] * 14263299814608977431, tmp_6071[2] * 14263299814608977431];
    signal tmp_6130[3] <== [tmp_6129[0] + tmp_6114[0], tmp_6129[1] + tmp_6114[1], tmp_6129[2] + tmp_6114[2]];
    signal tmp_6131[3] <== [tmp_6128[0] + tmp_6130[0], tmp_6128[1] + tmp_6130[1], tmp_6128[2] + tmp_6130[2]];
    signal tmp_6132[3] <== [tmp_6074[0] * 770395855193680981, tmp_6074[1] * 770395855193680981, tmp_6074[2] * 770395855193680981];
    signal tmp_6133[3] <== [tmp_6132[0] + tmp_6114[0], tmp_6132[1] + tmp_6114[1], tmp_6132[2] + tmp_6114[2]];
    signal tmp_6134[3] <== [tmp_6131[0] + tmp_6133[0], tmp_6131[1] + tmp_6133[1], tmp_6131[2] + tmp_6133[2]];
    signal tmp_6135[3] <== [tmp_6077[0] * 3459277367440070515, tmp_6077[1] * 3459277367440070515, tmp_6077[2] * 3459277367440070515];
    signal tmp_6136[3] <== [tmp_6135[0] + tmp_6114[0], tmp_6135[1] + tmp_6114[1], tmp_6135[2] + tmp_6114[2]];
    signal tmp_6137[3] <== [tmp_6134[0] + tmp_6136[0], tmp_6134[1] + tmp_6136[1], tmp_6134[2] + tmp_6136[2]];
    signal tmp_6138[3] <== [tmp_6080[0] * 17087697094293314027, tmp_6080[1] * 17087697094293314027, tmp_6080[2] * 17087697094293314027];
    signal tmp_6139[3] <== [tmp_6138[0] + tmp_6114[0], tmp_6138[1] + tmp_6114[1], tmp_6138[2] + tmp_6114[2]];
    signal tmp_6140[3] <== [tmp_6137[0] + tmp_6139[0], tmp_6137[1] + tmp_6139[1], tmp_6137[2] + tmp_6139[2]];
    signal tmp_6141[3] <== [tmp_6083[0] * 6694380135428747348, tmp_6083[1] * 6694380135428747348, tmp_6083[2] * 6694380135428747348];
    signal tmp_6142[3] <== [tmp_6141[0] + tmp_6114[0], tmp_6141[1] + tmp_6114[1], tmp_6141[2] + tmp_6114[2]];
    signal tmp_6143[3] <== [tmp_6140[0] + tmp_6142[0], tmp_6140[1] + tmp_6142[1], tmp_6140[2] + tmp_6142[2]];
    signal tmp_6144[3] <== [tmp_6086[0] * 2034408310088972836, tmp_6086[1] * 2034408310088972836, tmp_6086[2] * 2034408310088972836];
    signal tmp_6145[3] <== [tmp_6144[0] + tmp_6114[0], tmp_6144[1] + tmp_6114[1], tmp_6144[2] + tmp_6114[2]];
    signal tmp_6146[3] <== [tmp_6143[0] + tmp_6145[0], tmp_6143[1] + tmp_6145[1], tmp_6143[2] + tmp_6145[2]];
    signal tmp_6147[3] <== [tmp_6089[0] * 3434575637390274478, tmp_6089[1] * 3434575637390274478, tmp_6089[2] * 3434575637390274478];
    signal tmp_6148[3] <== [tmp_6147[0] + tmp_6114[0], tmp_6147[1] + tmp_6114[1], tmp_6147[2] + tmp_6114[2]];
    signal tmp_6149[3] <== [tmp_6146[0] + tmp_6148[0], tmp_6146[1] + tmp_6148[1], tmp_6146[2] + tmp_6148[2]];
    signal tmp_6150[3] <== [tmp_6092[0] * 6052753985947965968, tmp_6092[1] * 6052753985947965968, tmp_6092[2] * 6052753985947965968];
    signal tmp_6151[3] <== [tmp_6150[0] + tmp_6114[0], tmp_6150[1] + tmp_6114[1], tmp_6150[2] + tmp_6114[2]];
    signal tmp_6152[3] <== [tmp_6149[0] + tmp_6151[0], tmp_6149[1] + tmp_6151[1], tmp_6149[2] + tmp_6151[2]];
    signal tmp_6153[3] <== [tmp_6095[0] * 13608362914817483670, tmp_6095[1] * 13608362914817483670, tmp_6095[2] * 13608362914817483670];
    signal tmp_6154[3] <== [tmp_6153[0] + tmp_6114[0], tmp_6153[1] + tmp_6114[1], tmp_6153[2] + tmp_6114[2]];
    signal tmp_6155[3] <== [tmp_6152[0] + tmp_6154[0], tmp_6152[1] + tmp_6154[1], tmp_6152[2] + tmp_6154[2]];
    signal tmp_6156[3] <== [tmp_6098[0] * 18163707672964630459, tmp_6098[1] * 18163707672964630459, tmp_6098[2] * 18163707672964630459];
    signal tmp_6157[3] <== [tmp_6156[0] + tmp_6114[0], tmp_6156[1] + tmp_6114[1], tmp_6156[2] + tmp_6114[2]];
    signal tmp_6158[3] <== [tmp_6155[0] + tmp_6157[0], tmp_6155[1] + tmp_6157[1], tmp_6155[2] + tmp_6157[2]];
    signal tmp_6159[3] <== [tmp_6101[0] * 14373610220374016704, tmp_6101[1] * 14373610220374016704, tmp_6101[2] * 14373610220374016704];
    signal tmp_6160[3] <== [tmp_6159[0] + tmp_6114[0], tmp_6159[1] + tmp_6114[1], tmp_6159[2] + tmp_6114[2]];
    signal tmp_6161[3] <== [tmp_6158[0] + tmp_6160[0], tmp_6158[1] + tmp_6160[1], tmp_6158[2] + tmp_6160[2]];
    signal tmp_6162[3] <== [tmp_6104[0] * 6226282807566121054, tmp_6104[1] * 6226282807566121054, tmp_6104[2] * 6226282807566121054];
    signal tmp_6163[3] <== [tmp_6162[0] + tmp_6114[0], tmp_6162[1] + tmp_6114[1], tmp_6162[2] + tmp_6114[2]];
    signal tmp_6164[3] <== [tmp_6161[0] + tmp_6163[0], tmp_6161[1] + tmp_6163[1], tmp_6161[2] + tmp_6163[2]];
    signal tmp_6165[3] <== [tmp_6107[0] * 3643354756180461803, tmp_6107[1] * 3643354756180461803, tmp_6107[2] * 3643354756180461803];
    signal tmp_6166[3] <== [tmp_6165[0] + tmp_6114[0], tmp_6165[1] + tmp_6114[1], tmp_6165[2] + tmp_6114[2]];
    signal tmp_6167[3] <== [tmp_6164[0] + tmp_6166[0], tmp_6164[1] + tmp_6166[1], tmp_6164[2] + tmp_6166[2]];
    signal tmp_6168[3] <== [tmp_6110[0] * 13046961313070095543, tmp_6110[1] * 13046961313070095543, tmp_6110[2] * 13046961313070095543];
    signal tmp_6169[3] <== [tmp_6168[0] + tmp_6114[0], tmp_6168[1] + tmp_6114[1], tmp_6168[2] + tmp_6114[2]];
    signal tmp_6170[3] <== [tmp_6167[0] + tmp_6169[0], tmp_6167[1] + tmp_6169[1], tmp_6167[2] + tmp_6169[2]];
    signal tmp_6171[3] <== [tmp_6113[0] * 8594143216561850811, tmp_6113[1] * 8594143216561850811, tmp_6113[2] * 8594143216561850811];
    signal tmp_6172[3] <== [tmp_6171[0] + tmp_6114[0], tmp_6171[1] + tmp_6114[1], tmp_6171[2] + tmp_6114[2]];
    signal tmp_6173[3] <== [tmp_6170[0] + tmp_6172[0], tmp_6170[1] + tmp_6172[1], tmp_6170[2] + tmp_6172[2]];
    signal tmp_6174[3] <== [tmp_6126[0] + tmp_6173[0], tmp_6126[1] + tmp_6173[1], tmp_6126[2] + tmp_6173[2]];
    signal tmp_6175[3] <== [evals[26][0] - tmp_6174[0], evals[26][1] - tmp_6174[1], evals[26][2] - tmp_6174[2]];
    signal tmp_6176[3] <== CMul()(evals[76], tmp_6175);
    signal tmp_6177[3] <== [tmp_6119[0] + tmp_6176[0], tmp_6119[1] + tmp_6176[1], tmp_6119[2] + tmp_6176[2]];
    signal tmp_6178[3] <== CMul()(challengeQ, tmp_6177);
    signal tmp_6179[3] <== [evals[26][0] + 5177282273995529875, evals[26][1], evals[26][2]];
    signal tmp_6180[3] <== [evals[26][0] + 5177282273995529875, evals[26][1], evals[26][2]];
    tmp_6181 <== CMul()(tmp_6179, tmp_6180);
    signal tmp_6182[3] <== CMul()(evals[167], tmp_6181);
    signal tmp_6183[3] <== [evals[26][0] + 5177282273995529875, evals[26][1], evals[26][2]];
    signal tmp_6184[3] <== CMul()(tmp_6182, tmp_6183);
    signal tmp_6185[3] <== [tmp_6184[0] * 16040574633112940480, tmp_6184[1] * 16040574633112940480, tmp_6184[2] * 16040574633112940480];
    signal tmp_6186[3] <== [evals[26][0] + 5177282273995529875, evals[26][1], evals[26][2]];
    signal tmp_6187[3] <== CMul()(tmp_6182, tmp_6186);
    signal tmp_6188[3] <== [tmp_6130[0] * 14263299814608977431, tmp_6130[1] * 14263299814608977431, tmp_6130[2] * 14263299814608977431];
    signal tmp_6189[3] <== [tmp_6188[0] + tmp_6173[0], tmp_6188[1] + tmp_6173[1], tmp_6188[2] + tmp_6173[2]];
    signal tmp_6190[3] <== [tmp_6187[0] + tmp_6189[0], tmp_6187[1] + tmp_6189[1], tmp_6187[2] + tmp_6189[2]];
    signal tmp_6191[3] <== [tmp_6133[0] * 770395855193680981, tmp_6133[1] * 770395855193680981, tmp_6133[2] * 770395855193680981];
    signal tmp_6192[3] <== [tmp_6191[0] + tmp_6173[0], tmp_6191[1] + tmp_6173[1], tmp_6191[2] + tmp_6173[2]];
    signal tmp_6193[3] <== [tmp_6190[0] + tmp_6192[0], tmp_6190[1] + tmp_6192[1], tmp_6190[2] + tmp_6192[2]];
    signal tmp_6194[3] <== [tmp_6136[0] * 3459277367440070515, tmp_6136[1] * 3459277367440070515, tmp_6136[2] * 3459277367440070515];
    signal tmp_6195[3] <== [tmp_6194[0] + tmp_6173[0], tmp_6194[1] + tmp_6173[1], tmp_6194[2] + tmp_6173[2]];
    signal tmp_6196[3] <== [tmp_6193[0] + tmp_6195[0], tmp_6193[1] + tmp_6195[1], tmp_6193[2] + tmp_6195[2]];
    signal tmp_6197[3] <== [tmp_6139[0] * 17087697094293314027, tmp_6139[1] * 17087697094293314027, tmp_6139[2] * 17087697094293314027];
    signal tmp_6198[3] <== [tmp_6197[0] + tmp_6173[0], tmp_6197[1] + tmp_6173[1], tmp_6197[2] + tmp_6173[2]];
    signal tmp_6199[3] <== [tmp_6196[0] + tmp_6198[0], tmp_6196[1] + tmp_6198[1], tmp_6196[2] + tmp_6198[2]];
    signal tmp_6200[3] <== [tmp_6142[0] * 6694380135428747348, tmp_6142[1] * 6694380135428747348, tmp_6142[2] * 6694380135428747348];
    signal tmp_6201[3] <== [tmp_6200[0] + tmp_6173[0], tmp_6200[1] + tmp_6173[1], tmp_6200[2] + tmp_6173[2]];
    signal tmp_6202[3] <== [tmp_6199[0] + tmp_6201[0], tmp_6199[1] + tmp_6201[1], tmp_6199[2] + tmp_6201[2]];
    signal tmp_6203[3] <== [tmp_6145[0] * 2034408310088972836, tmp_6145[1] * 2034408310088972836, tmp_6145[2] * 2034408310088972836];
    signal tmp_6204[3] <== [tmp_6203[0] + tmp_6173[0], tmp_6203[1] + tmp_6173[1], tmp_6203[2] + tmp_6173[2]];
    signal tmp_6205[3] <== [tmp_6202[0] + tmp_6204[0], tmp_6202[1] + tmp_6204[1], tmp_6202[2] + tmp_6204[2]];
    signal tmp_6206[3] <== [tmp_6148[0] * 3434575637390274478, tmp_6148[1] * 3434575637390274478, tmp_6148[2] * 3434575637390274478];
    signal tmp_6207[3] <== [tmp_6206[0] + tmp_6173[0], tmp_6206[1] + tmp_6173[1], tmp_6206[2] + tmp_6173[2]];
    signal tmp_6208[3] <== [tmp_6205[0] + tmp_6207[0], tmp_6205[1] + tmp_6207[1], tmp_6205[2] + tmp_6207[2]];
    signal tmp_6209[3] <== [tmp_6151[0] * 6052753985947965968, tmp_6151[1] * 6052753985947965968, tmp_6151[2] * 6052753985947965968];
    signal tmp_6210[3] <== [tmp_6209[0] + tmp_6173[0], tmp_6209[1] + tmp_6173[1], tmp_6209[2] + tmp_6173[2]];
    signal tmp_6211[3] <== [tmp_6208[0] + tmp_6210[0], tmp_6208[1] + tmp_6210[1], tmp_6208[2] + tmp_6210[2]];
    signal tmp_6212[3] <== [tmp_6154[0] * 13608362914817483670, tmp_6154[1] * 13608362914817483670, tmp_6154[2] * 13608362914817483670];
    signal tmp_6213[3] <== [tmp_6212[0] + tmp_6173[0], tmp_6212[1] + tmp_6173[1], tmp_6212[2] + tmp_6173[2]];
    signal tmp_6214[3] <== [tmp_6211[0] + tmp_6213[0], tmp_6211[1] + tmp_6213[1], tmp_6211[2] + tmp_6213[2]];
    signal tmp_6215[3] <== [tmp_6157[0] * 18163707672964630459, tmp_6157[1] * 18163707672964630459, tmp_6157[2] * 18163707672964630459];
    signal tmp_6216[3] <== [tmp_6215[0] + tmp_6173[0], tmp_6215[1] + tmp_6173[1], tmp_6215[2] + tmp_6173[2]];
    signal tmp_6217[3] <== [tmp_6214[0] + tmp_6216[0], tmp_6214[1] + tmp_6216[1], tmp_6214[2] + tmp_6216[2]];
    signal tmp_6218[3] <== [tmp_6160[0] * 14373610220374016704, tmp_6160[1] * 14373610220374016704, tmp_6160[2] * 14373610220374016704];
    signal tmp_6219[3] <== [tmp_6218[0] + tmp_6173[0], tmp_6218[1] + tmp_6173[1], tmp_6218[2] + tmp_6173[2]];
    signal tmp_6220[3] <== [tmp_6217[0] + tmp_6219[0], tmp_6217[1] + tmp_6219[1], tmp_6217[2] + tmp_6219[2]];
    signal tmp_6221[3] <== [tmp_6163[0] * 6226282807566121054, tmp_6163[1] * 6226282807566121054, tmp_6163[2] * 6226282807566121054];
    signal tmp_6222[3] <== [tmp_6221[0] + tmp_6173[0], tmp_6221[1] + tmp_6173[1], tmp_6221[2] + tmp_6173[2]];
    signal tmp_6223[3] <== [tmp_6220[0] + tmp_6222[0], tmp_6220[1] + tmp_6222[1], tmp_6220[2] + tmp_6222[2]];
    signal tmp_6224[3] <== [tmp_6166[0] * 3643354756180461803, tmp_6166[1] * 3643354756180461803, tmp_6166[2] * 3643354756180461803];
    signal tmp_6225[3] <== [tmp_6224[0] + tmp_6173[0], tmp_6224[1] + tmp_6173[1], tmp_6224[2] + tmp_6173[2]];
    signal tmp_6226[3] <== [tmp_6223[0] + tmp_6225[0], tmp_6223[1] + tmp_6225[1], tmp_6223[2] + tmp_6225[2]];
    signal tmp_6227[3] <== [tmp_6169[0] * 13046961313070095543, tmp_6169[1] * 13046961313070095543, tmp_6169[2] * 13046961313070095543];
    signal tmp_6228[3] <== [tmp_6227[0] + tmp_6173[0], tmp_6227[1] + tmp_6173[1], tmp_6227[2] + tmp_6173[2]];
    signal tmp_6229[3] <== [tmp_6226[0] + tmp_6228[0], tmp_6226[1] + tmp_6228[1], tmp_6226[2] + tmp_6228[2]];
    signal tmp_6230[3] <== [tmp_6172[0] * 8594143216561850811, tmp_6172[1] * 8594143216561850811, tmp_6172[2] * 8594143216561850811];
    signal tmp_6231[3] <== [tmp_6230[0] + tmp_6173[0], tmp_6230[1] + tmp_6173[1], tmp_6230[2] + tmp_6173[2]];
    signal tmp_6232[3] <== [tmp_6229[0] + tmp_6231[0], tmp_6229[1] + tmp_6231[1], tmp_6229[2] + tmp_6231[2]];
    signal tmp_6233[3] <== [tmp_6185[0] + tmp_6232[0], tmp_6185[1] + tmp_6232[1], tmp_6185[2] + tmp_6232[2]];
    signal tmp_6234[3] <== [evals[27][0] - tmp_6233[0], evals[27][1] - tmp_6233[1], evals[27][2] - tmp_6233[2]];
    signal tmp_6235[3] <== CMul()(evals[76], tmp_6234);
    signal tmp_6236[3] <== [tmp_6178[0] + tmp_6235[0], tmp_6178[1] + tmp_6235[1], tmp_6178[2] + tmp_6235[2]];
    signal tmp_6237[3] <== CMul()(challengeQ, tmp_6236);
    signal tmp_6238[3] <== [evals[27][0] + 16035152896984012190, evals[27][1], evals[27][2]];
    signal tmp_6239[3] <== [evals[27][0] + 16035152896984012190, evals[27][1], evals[27][2]];
    tmp_6240 <== CMul()(tmp_6238, tmp_6239);
    signal tmp_6241[3] <== CMul()(evals[168], tmp_6240);
    signal tmp_6242[3] <== [evals[27][0] + 16035152896984012190, evals[27][1], evals[27][2]];
    signal tmp_6243[3] <== CMul()(tmp_6241, tmp_6242);
    signal tmp_6244[3] <== [tmp_6243[0] * 16040574633112940480, tmp_6243[1] * 16040574633112940480, tmp_6243[2] * 16040574633112940480];
    signal tmp_6245[3] <== [evals[27][0] + 16035152896984012190, evals[27][1], evals[27][2]];
    signal tmp_6246[3] <== CMul()(tmp_6241, tmp_6245);
    signal tmp_6247[3] <== [tmp_6189[0] * 14263299814608977431, tmp_6189[1] * 14263299814608977431, tmp_6189[2] * 14263299814608977431];
    signal tmp_6248[3] <== [tmp_6247[0] + tmp_6232[0], tmp_6247[1] + tmp_6232[1], tmp_6247[2] + tmp_6232[2]];
    signal tmp_6249[3] <== [tmp_6246[0] + tmp_6248[0], tmp_6246[1] + tmp_6248[1], tmp_6246[2] + tmp_6248[2]];
    signal tmp_6250[3] <== [tmp_6192[0] * 770395855193680981, tmp_6192[1] * 770395855193680981, tmp_6192[2] * 770395855193680981];
    signal tmp_6251[3] <== [tmp_6250[0] + tmp_6232[0], tmp_6250[1] + tmp_6232[1], tmp_6250[2] + tmp_6232[2]];
    signal tmp_6252[3] <== [tmp_6249[0] + tmp_6251[0], tmp_6249[1] + tmp_6251[1], tmp_6249[2] + tmp_6251[2]];
    signal tmp_6253[3] <== [tmp_6195[0] * 3459277367440070515, tmp_6195[1] * 3459277367440070515, tmp_6195[2] * 3459277367440070515];
    signal tmp_6254[3] <== [tmp_6253[0] + tmp_6232[0], tmp_6253[1] + tmp_6232[1], tmp_6253[2] + tmp_6232[2]];
    signal tmp_6255[3] <== [tmp_6252[0] + tmp_6254[0], tmp_6252[1] + tmp_6254[1], tmp_6252[2] + tmp_6254[2]];
    signal tmp_6256[3] <== [tmp_6198[0] * 17087697094293314027, tmp_6198[1] * 17087697094293314027, tmp_6198[2] * 17087697094293314027];
    signal tmp_6257[3] <== [tmp_6256[0] + tmp_6232[0], tmp_6256[1] + tmp_6232[1], tmp_6256[2] + tmp_6232[2]];
    signal tmp_6258[3] <== [tmp_6255[0] + tmp_6257[0], tmp_6255[1] + tmp_6257[1], tmp_6255[2] + tmp_6257[2]];
    signal tmp_6259[3] <== [tmp_6201[0] * 6694380135428747348, tmp_6201[1] * 6694380135428747348, tmp_6201[2] * 6694380135428747348];
    signal tmp_6260[3] <== [tmp_6259[0] + tmp_6232[0], tmp_6259[1] + tmp_6232[1], tmp_6259[2] + tmp_6232[2]];
    signal tmp_6261[3] <== [tmp_6258[0] + tmp_6260[0], tmp_6258[1] + tmp_6260[1], tmp_6258[2] + tmp_6260[2]];
    signal tmp_6262[3] <== [tmp_6204[0] * 2034408310088972836, tmp_6204[1] * 2034408310088972836, tmp_6204[2] * 2034408310088972836];
    signal tmp_6263[3] <== [tmp_6262[0] + tmp_6232[0], tmp_6262[1] + tmp_6232[1], tmp_6262[2] + tmp_6232[2]];
    signal tmp_6264[3] <== [tmp_6261[0] + tmp_6263[0], tmp_6261[1] + tmp_6263[1], tmp_6261[2] + tmp_6263[2]];
    signal tmp_6265[3] <== [tmp_6207[0] * 3434575637390274478, tmp_6207[1] * 3434575637390274478, tmp_6207[2] * 3434575637390274478];
    signal tmp_6266[3] <== [tmp_6265[0] + tmp_6232[0], tmp_6265[1] + tmp_6232[1], tmp_6265[2] + tmp_6232[2]];
    signal tmp_6267[3] <== [tmp_6264[0] + tmp_6266[0], tmp_6264[1] + tmp_6266[1], tmp_6264[2] + tmp_6266[2]];
    signal tmp_6268[3] <== [tmp_6210[0] * 6052753985947965968, tmp_6210[1] * 6052753985947965968, tmp_6210[2] * 6052753985947965968];
    signal tmp_6269[3] <== [tmp_6268[0] + tmp_6232[0], tmp_6268[1] + tmp_6232[1], tmp_6268[2] + tmp_6232[2]];
    signal tmp_6270[3] <== [tmp_6267[0] + tmp_6269[0], tmp_6267[1] + tmp_6269[1], tmp_6267[2] + tmp_6269[2]];
    signal tmp_6271[3] <== [tmp_6213[0] * 13608362914817483670, tmp_6213[1] * 13608362914817483670, tmp_6213[2] * 13608362914817483670];
    signal tmp_6272[3] <== [tmp_6271[0] + tmp_6232[0], tmp_6271[1] + tmp_6232[1], tmp_6271[2] + tmp_6232[2]];
    signal tmp_6273[3] <== [tmp_6270[0] + tmp_6272[0], tmp_6270[1] + tmp_6272[1], tmp_6270[2] + tmp_6272[2]];
    signal tmp_6274[3] <== [tmp_6216[0] * 18163707672964630459, tmp_6216[1] * 18163707672964630459, tmp_6216[2] * 18163707672964630459];
    signal tmp_6275[3] <== [tmp_6274[0] + tmp_6232[0], tmp_6274[1] + tmp_6232[1], tmp_6274[2] + tmp_6232[2]];
    signal tmp_6276[3] <== [tmp_6273[0] + tmp_6275[0], tmp_6273[1] + tmp_6275[1], tmp_6273[2] + tmp_6275[2]];
    signal tmp_6277[3] <== [tmp_6219[0] * 14373610220374016704, tmp_6219[1] * 14373610220374016704, tmp_6219[2] * 14373610220374016704];
    signal tmp_6278[3] <== [tmp_6277[0] + tmp_6232[0], tmp_6277[1] + tmp_6232[1], tmp_6277[2] + tmp_6232[2]];
    signal tmp_6279[3] <== [tmp_6276[0] + tmp_6278[0], tmp_6276[1] + tmp_6278[1], tmp_6276[2] + tmp_6278[2]];
    signal tmp_6280[3] <== [tmp_6222[0] * 6226282807566121054, tmp_6222[1] * 6226282807566121054, tmp_6222[2] * 6226282807566121054];
    signal tmp_6281[3] <== [tmp_6280[0] + tmp_6232[0], tmp_6280[1] + tmp_6232[1], tmp_6280[2] + tmp_6232[2]];
    signal tmp_6282[3] <== [tmp_6279[0] + tmp_6281[0], tmp_6279[1] + tmp_6281[1], tmp_6279[2] + tmp_6281[2]];
    signal tmp_6283[3] <== [tmp_6225[0] * 3643354756180461803, tmp_6225[1] * 3643354756180461803, tmp_6225[2] * 3643354756180461803];
    signal tmp_6284[3] <== [tmp_6283[0] + tmp_6232[0], tmp_6283[1] + tmp_6232[1], tmp_6283[2] + tmp_6232[2]];
    signal tmp_6285[3] <== [tmp_6282[0] + tmp_6284[0], tmp_6282[1] + tmp_6284[1], tmp_6282[2] + tmp_6284[2]];
    signal tmp_6286[3] <== [tmp_6228[0] * 13046961313070095543, tmp_6228[1] * 13046961313070095543, tmp_6228[2] * 13046961313070095543];
    signal tmp_6287[3] <== [tmp_6286[0] + tmp_6232[0], tmp_6286[1] + tmp_6232[1], tmp_6286[2] + tmp_6232[2]];
    signal tmp_6288[3] <== [tmp_6285[0] + tmp_6287[0], tmp_6285[1] + tmp_6287[1], tmp_6285[2] + tmp_6287[2]];
    signal tmp_6289[3] <== [tmp_6231[0] * 8594143216561850811, tmp_6231[1] * 8594143216561850811, tmp_6231[2] * 8594143216561850811];
    signal tmp_6290[3] <== [tmp_6289[0] + tmp_6232[0], tmp_6289[1] + tmp_6232[1], tmp_6289[2] + tmp_6232[2]];
    signal tmp_6291[3] <== [tmp_6288[0] + tmp_6290[0], tmp_6288[1] + tmp_6290[1], tmp_6288[2] + tmp_6290[2]];
    signal tmp_6292[3] <== [tmp_6244[0] + tmp_6291[0], tmp_6244[1] + tmp_6291[1], tmp_6244[2] + tmp_6291[2]];
    signal tmp_6293[3] <== [evals[180][0] - tmp_6292[0], evals[180][1] - tmp_6292[1], evals[180][2] - tmp_6292[2]];
    signal tmp_6294[3] <== CMul()(evals[76], tmp_6293);
    signal tmp_6295[3] <== [tmp_6237[0] + tmp_6294[0], tmp_6237[1] + tmp_6294[1], tmp_6237[2] + tmp_6294[2]];
    signal tmp_6296[3] <== CMul()(challengeQ, tmp_6295);
    signal tmp_6297[3] <== [tmp_6248[0] * 14263299814608977431, tmp_6248[1] * 14263299814608977431, tmp_6248[2] * 14263299814608977431];
    signal tmp_6298[3] <== [tmp_6297[0] + tmp_6291[0], tmp_6297[1] + tmp_6291[1], tmp_6297[2] + tmp_6291[2]];
    signal tmp_6299[3] <== [evals[181][0] - tmp_6298[0], evals[181][1] - tmp_6298[1], evals[181][2] - tmp_6298[2]];
    signal tmp_6300[3] <== CMul()(evals[76], tmp_6299);
    signal tmp_6301[3] <== [tmp_6296[0] + tmp_6300[0], tmp_6296[1] + tmp_6300[1], tmp_6296[2] + tmp_6300[2]];
    signal tmp_6302[3] <== CMul()(challengeQ, tmp_6301);
    signal tmp_6303[3] <== [tmp_6251[0] * 770395855193680981, tmp_6251[1] * 770395855193680981, tmp_6251[2] * 770395855193680981];
    signal tmp_6304[3] <== [tmp_6303[0] + tmp_6291[0], tmp_6303[1] + tmp_6291[1], tmp_6303[2] + tmp_6291[2]];
    signal tmp_6305[3] <== [evals[182][0] - tmp_6304[0], evals[182][1] - tmp_6304[1], evals[182][2] - tmp_6304[2]];
    signal tmp_6306[3] <== CMul()(evals[76], tmp_6305);
    signal tmp_6307[3] <== [tmp_6302[0] + tmp_6306[0], tmp_6302[1] + tmp_6306[1], tmp_6302[2] + tmp_6306[2]];
    signal tmp_6308[3] <== CMul()(challengeQ, tmp_6307);
    signal tmp_6309[3] <== [tmp_6254[0] * 3459277367440070515, tmp_6254[1] * 3459277367440070515, tmp_6254[2] * 3459277367440070515];
    signal tmp_6310[3] <== [tmp_6309[0] + tmp_6291[0], tmp_6309[1] + tmp_6291[1], tmp_6309[2] + tmp_6291[2]];
    signal tmp_6311[3] <== [evals[183][0] - tmp_6310[0], evals[183][1] - tmp_6310[1], evals[183][2] - tmp_6310[2]];
    signal tmp_6312[3] <== CMul()(evals[76], tmp_6311);
    signal tmp_6313[3] <== [tmp_6308[0] + tmp_6312[0], tmp_6308[1] + tmp_6312[1], tmp_6308[2] + tmp_6312[2]];
    signal tmp_6314[3] <== CMul()(challengeQ, tmp_6313);
    signal tmp_6315[3] <== [tmp_6257[0] * 17087697094293314027, tmp_6257[1] * 17087697094293314027, tmp_6257[2] * 17087697094293314027];
    signal tmp_6316[3] <== [tmp_6315[0] + tmp_6291[0], tmp_6315[1] + tmp_6291[1], tmp_6315[2] + tmp_6291[2]];
    signal tmp_6317[3] <== [evals[184][0] - tmp_6316[0], evals[184][1] - tmp_6316[1], evals[184][2] - tmp_6316[2]];
    signal tmp_6318[3] <== CMul()(evals[76], tmp_6317);
    signal tmp_6319[3] <== [tmp_6314[0] + tmp_6318[0], tmp_6314[1] + tmp_6318[1], tmp_6314[2] + tmp_6318[2]];
    signal tmp_6320[3] <== CMul()(challengeQ, tmp_6319);
    signal tmp_6321[3] <== [tmp_6260[0] * 6694380135428747348, tmp_6260[1] * 6694380135428747348, tmp_6260[2] * 6694380135428747348];
    signal tmp_6322[3] <== [tmp_6321[0] + tmp_6291[0], tmp_6321[1] + tmp_6291[1], tmp_6321[2] + tmp_6291[2]];
    signal tmp_6323[3] <== [evals[185][0] - tmp_6322[0], evals[185][1] - tmp_6322[1], evals[185][2] - tmp_6322[2]];
    signal tmp_6324[3] <== CMul()(evals[76], tmp_6323);
    signal tmp_6325[3] <== [tmp_6320[0] + tmp_6324[0], tmp_6320[1] + tmp_6324[1], tmp_6320[2] + tmp_6324[2]];
    signal tmp_6326[3] <== CMul()(challengeQ, tmp_6325);
    signal tmp_6327[3] <== [tmp_6263[0] * 2034408310088972836, tmp_6263[1] * 2034408310088972836, tmp_6263[2] * 2034408310088972836];
    signal tmp_6328[3] <== [tmp_6327[0] + tmp_6291[0], tmp_6327[1] + tmp_6291[1], tmp_6327[2] + tmp_6291[2]];
    signal tmp_6329[3] <== [evals[186][0] - tmp_6328[0], evals[186][1] - tmp_6328[1], evals[186][2] - tmp_6328[2]];
    signal tmp_6330[3] <== CMul()(evals[76], tmp_6329);
    signal tmp_6331[3] <== [tmp_6326[0] + tmp_6330[0], tmp_6326[1] + tmp_6330[1], tmp_6326[2] + tmp_6330[2]];
    signal tmp_6332[3] <== CMul()(challengeQ, tmp_6331);
    signal tmp_6333[3] <== [tmp_6266[0] * 3434575637390274478, tmp_6266[1] * 3434575637390274478, tmp_6266[2] * 3434575637390274478];
    signal tmp_6334[3] <== [tmp_6333[0] + tmp_6291[0], tmp_6333[1] + tmp_6291[1], tmp_6333[2] + tmp_6291[2]];
    signal tmp_6335[3] <== [evals[187][0] - tmp_6334[0], evals[187][1] - tmp_6334[1], evals[187][2] - tmp_6334[2]];
    signal tmp_6336[3] <== CMul()(evals[76], tmp_6335);
    signal tmp_6337[3] <== [tmp_6332[0] + tmp_6336[0], tmp_6332[1] + tmp_6336[1], tmp_6332[2] + tmp_6336[2]];
    signal tmp_6338[3] <== CMul()(challengeQ, tmp_6337);
    signal tmp_6339[3] <== [tmp_6269[0] * 6052753985947965968, tmp_6269[1] * 6052753985947965968, tmp_6269[2] * 6052753985947965968];
    signal tmp_6340[3] <== [tmp_6339[0] + tmp_6291[0], tmp_6339[1] + tmp_6291[1], tmp_6339[2] + tmp_6291[2]];
    signal tmp_6341[3] <== [evals[188][0] - tmp_6340[0], evals[188][1] - tmp_6340[1], evals[188][2] - tmp_6340[2]];
    signal tmp_6342[3] <== CMul()(evals[76], tmp_6341);
    signal tmp_6343[3] <== [tmp_6338[0] + tmp_6342[0], tmp_6338[1] + tmp_6342[1], tmp_6338[2] + tmp_6342[2]];
    signal tmp_6344[3] <== CMul()(challengeQ, tmp_6343);
    signal tmp_6345[3] <== [tmp_6272[0] * 13608362914817483670, tmp_6272[1] * 13608362914817483670, tmp_6272[2] * 13608362914817483670];
    signal tmp_6346[3] <== [tmp_6345[0] + tmp_6291[0], tmp_6345[1] + tmp_6291[1], tmp_6345[2] + tmp_6291[2]];
    signal tmp_6347[3] <== [evals[189][0] - tmp_6346[0], evals[189][1] - tmp_6346[1], evals[189][2] - tmp_6346[2]];
    signal tmp_6348[3] <== CMul()(evals[76], tmp_6347);
    signal tmp_6349[3] <== [tmp_6344[0] + tmp_6348[0], tmp_6344[1] + tmp_6348[1], tmp_6344[2] + tmp_6348[2]];
    signal tmp_6350[3] <== CMul()(challengeQ, tmp_6349);
    signal tmp_6351[3] <== [tmp_6275[0] * 18163707672964630459, tmp_6275[1] * 18163707672964630459, tmp_6275[2] * 18163707672964630459];
    signal tmp_6352[3] <== [tmp_6351[0] + tmp_6291[0], tmp_6351[1] + tmp_6291[1], tmp_6351[2] + tmp_6291[2]];
    signal tmp_6353[3] <== [evals[190][0] - tmp_6352[0], evals[190][1] - tmp_6352[1], evals[190][2] - tmp_6352[2]];
    signal tmp_6354[3] <== CMul()(evals[76], tmp_6353);
    signal tmp_6355[3] <== [tmp_6350[0] + tmp_6354[0], tmp_6350[1] + tmp_6354[1], tmp_6350[2] + tmp_6354[2]];
    signal tmp_6356[3] <== CMul()(challengeQ, tmp_6355);
    signal tmp_6357[3] <== [tmp_6278[0] * 14373610220374016704, tmp_6278[1] * 14373610220374016704, tmp_6278[2] * 14373610220374016704];
    signal tmp_6358[3] <== [tmp_6357[0] + tmp_6291[0], tmp_6357[1] + tmp_6291[1], tmp_6357[2] + tmp_6291[2]];
    signal tmp_6359[3] <== [evals[191][0] - tmp_6358[0], evals[191][1] - tmp_6358[1], evals[191][2] - tmp_6358[2]];
    signal tmp_6360[3] <== CMul()(evals[76], tmp_6359);
    signal tmp_6361[3] <== [tmp_6356[0] + tmp_6360[0], tmp_6356[1] + tmp_6360[1], tmp_6356[2] + tmp_6360[2]];
    signal tmp_6362[3] <== CMul()(challengeQ, tmp_6361);
    signal tmp_6363[3] <== [tmp_6281[0] * 6226282807566121054, tmp_6281[1] * 6226282807566121054, tmp_6281[2] * 6226282807566121054];
    signal tmp_6364[3] <== [tmp_6363[0] + tmp_6291[0], tmp_6363[1] + tmp_6291[1], tmp_6363[2] + tmp_6291[2]];
    signal tmp_6365[3] <== [evals[192][0] - tmp_6364[0], evals[192][1] - tmp_6364[1], evals[192][2] - tmp_6364[2]];
    signal tmp_6366[3] <== CMul()(evals[76], tmp_6365);
    signal tmp_6367[3] <== [tmp_6362[0] + tmp_6366[0], tmp_6362[1] + tmp_6366[1], tmp_6362[2] + tmp_6366[2]];
    signal tmp_6368[3] <== CMul()(challengeQ, tmp_6367);
    signal tmp_6369[3] <== [tmp_6284[0] * 3643354756180461803, tmp_6284[1] * 3643354756180461803, tmp_6284[2] * 3643354756180461803];
    signal tmp_6370[3] <== [tmp_6369[0] + tmp_6291[0], tmp_6369[1] + tmp_6291[1], tmp_6369[2] + tmp_6291[2]];
    signal tmp_6371[3] <== [evals[193][0] - tmp_6370[0], evals[193][1] - tmp_6370[1], evals[193][2] - tmp_6370[2]];
    signal tmp_6372[3] <== CMul()(evals[76], tmp_6371);
    signal tmp_6373[3] <== [tmp_6368[0] + tmp_6372[0], tmp_6368[1] + tmp_6372[1], tmp_6368[2] + tmp_6372[2]];
    signal tmp_6374[3] <== CMul()(challengeQ, tmp_6373);
    signal tmp_6375[3] <== [tmp_6287[0] * 13046961313070095543, tmp_6287[1] * 13046961313070095543, tmp_6287[2] * 13046961313070095543];
    signal tmp_6376[3] <== [tmp_6375[0] + tmp_6291[0], tmp_6375[1] + tmp_6291[1], tmp_6375[2] + tmp_6291[2]];
    signal tmp_6377[3] <== [evals[194][0] - tmp_6376[0], evals[194][1] - tmp_6376[1], evals[194][2] - tmp_6376[2]];
    signal tmp_6378[3] <== CMul()(evals[76], tmp_6377);
    signal tmp_6379[3] <== [tmp_6374[0] + tmp_6378[0], tmp_6374[1] + tmp_6378[1], tmp_6374[2] + tmp_6378[2]];
    signal tmp_6380[3] <== CMul()(challengeQ, tmp_6379);
    signal tmp_6381[3] <== [tmp_6290[0] * 8594143216561850811, tmp_6290[1] * 8594143216561850811, tmp_6290[2] * 8594143216561850811];
    signal tmp_6382[3] <== [tmp_6381[0] + tmp_6291[0], tmp_6381[1] + tmp_6291[1], tmp_6381[2] + tmp_6291[2]];
    signal tmp_6383[3] <== [evals[195][0] - tmp_6382[0], evals[195][1] - tmp_6382[1], evals[195][2] - tmp_6382[2]];
    signal tmp_6384[3] <== CMul()(evals[76], tmp_6383);
    signal tmp_6385[3] <== [tmp_6380[0] + tmp_6384[0], tmp_6380[1] + tmp_6384[1], tmp_6380[2] + tmp_6384[2]];
    signal tmp_6386[3] <== CMul()(challengeQ, tmp_6385);
    signal tmp_6387[3] <== CMul()(evals[86], evals[89]);
    signal tmp_6388[3] <== CMul()(evals[87], evals[91]);
    signal tmp_6389[3] <== [tmp_6387[0] + tmp_6388[0], tmp_6387[1] + tmp_6388[1], tmp_6387[2] + tmp_6388[2]];
    signal tmp_6390[3] <== CMul()(evals[88], evals[90]);
    signal tmp_6391[3] <== [tmp_6389[0] + tmp_6390[0], tmp_6389[1] + tmp_6390[1], tmp_6389[2] + tmp_6390[2]];
    signal tmp_6392[3] <== [evals[92][0] - tmp_6391[0], evals[92][1] - tmp_6391[1], evals[92][2] - tmp_6391[2]];
    signal tmp_6393[3] <== CMul()(evals[78], tmp_6392);
    signal tmp_6394[3] <== [tmp_6386[0] + tmp_6393[0], tmp_6386[1] + tmp_6393[1], tmp_6386[2] + tmp_6393[2]];
    signal tmp_6395[3] <== CMul()(challengeQ, tmp_6394);
    signal tmp_6396[3] <== CMul()(evals[86], evals[90]);
    signal tmp_6397[3] <== CMul()(evals[87], evals[89]);
    signal tmp_6398[3] <== [tmp_6396[0] + tmp_6397[0], tmp_6396[1] + tmp_6397[1], tmp_6396[2] + tmp_6397[2]];
    signal tmp_6399[3] <== CMul()(evals[87], evals[91]);
    signal tmp_6400[3] <== [tmp_6398[0] + tmp_6399[0], tmp_6398[1] + tmp_6399[1], tmp_6398[2] + tmp_6399[2]];
    signal tmp_6401[3] <== CMul()(evals[88], evals[90]);
    signal tmp_6402[3] <== [tmp_6400[0] + tmp_6401[0], tmp_6400[1] + tmp_6401[1], tmp_6400[2] + tmp_6401[2]];
    signal tmp_6403[3] <== CMul()(evals[88], evals[91]);
    signal tmp_6404[3] <== [tmp_6402[0] + tmp_6403[0], tmp_6402[1] + tmp_6403[1], tmp_6402[2] + tmp_6403[2]];
    signal tmp_6405[3] <== [evals[93][0] - tmp_6404[0], evals[93][1] - tmp_6404[1], evals[93][2] - tmp_6404[2]];
    signal tmp_6406[3] <== CMul()(evals[78], tmp_6405);
    signal tmp_6407[3] <== [tmp_6395[0] + tmp_6406[0], tmp_6395[1] + tmp_6406[1], tmp_6395[2] + tmp_6406[2]];
    signal tmp_6408[3] <== CMul()(challengeQ, tmp_6407);
    signal tmp_6409[3] <== CMul()(evals[86], evals[91]);
    signal tmp_6410[3] <== CMul()(evals[88], evals[91]);
    signal tmp_6411[3] <== [tmp_6409[0] + tmp_6410[0], tmp_6409[1] + tmp_6410[1], tmp_6409[2] + tmp_6410[2]];
    signal tmp_6412[3] <== CMul()(evals[88], evals[89]);
    signal tmp_6413[3] <== [tmp_6411[0] + tmp_6412[0], tmp_6411[1] + tmp_6412[1], tmp_6411[2] + tmp_6412[2]];
    signal tmp_6414[3] <== CMul()(evals[87], evals[90]);
    signal tmp_6415[3] <== [tmp_6413[0] + tmp_6414[0], tmp_6413[1] + tmp_6414[1], tmp_6413[2] + tmp_6414[2]];
    signal tmp_6416[3] <== [evals[94][0] - tmp_6415[0], evals[94][1] - tmp_6415[1], evals[94][2] - tmp_6415[2]];
    signal tmp_6417[3] <== CMul()(evals[78], tmp_6416);
    signal tmp_6418[3] <== [tmp_6408[0] + tmp_6417[0], tmp_6408[1] + tmp_6417[1], tmp_6408[2] + tmp_6417[2]];
    signal tmp_6419[3] <== CMul()(challengeQ, tmp_6418);
    signal tmp_6420[3] <== CMul()(evals[95], evals[98]);
    signal tmp_6421[3] <== CMul()(evals[96], evals[100]);
    signal tmp_6422[3] <== [tmp_6420[0] + tmp_6421[0], tmp_6420[1] + tmp_6421[1], tmp_6420[2] + tmp_6421[2]];
    signal tmp_6423[3] <== CMul()(evals[97], evals[99]);
    signal tmp_6424[3] <== [tmp_6422[0] + tmp_6423[0], tmp_6422[1] + tmp_6423[1], tmp_6422[2] + tmp_6423[2]];
    signal tmp_6425[3] <== [evals[101][0] - tmp_6424[0], evals[101][1] - tmp_6424[1], evals[101][2] - tmp_6424[2]];
    signal tmp_6426[3] <== CMul()(evals[78], tmp_6425);
    signal tmp_6427[3] <== [tmp_6419[0] + tmp_6426[0], tmp_6419[1] + tmp_6426[1], tmp_6419[2] + tmp_6426[2]];
    signal tmp_6428[3] <== CMul()(challengeQ, tmp_6427);
    signal tmp_6429[3] <== CMul()(evals[95], evals[99]);
    signal tmp_6430[3] <== CMul()(evals[96], evals[98]);
    signal tmp_6431[3] <== [tmp_6429[0] + tmp_6430[0], tmp_6429[1] + tmp_6430[1], tmp_6429[2] + tmp_6430[2]];
    signal tmp_6432[3] <== CMul()(evals[96], evals[100]);
    signal tmp_6433[3] <== [tmp_6431[0] + tmp_6432[0], tmp_6431[1] + tmp_6432[1], tmp_6431[2] + tmp_6432[2]];
    signal tmp_6434[3] <== CMul()(evals[97], evals[99]);
    signal tmp_6435[3] <== [tmp_6433[0] + tmp_6434[0], tmp_6433[1] + tmp_6434[1], tmp_6433[2] + tmp_6434[2]];
    signal tmp_6436[3] <== CMul()(evals[97], evals[100]);
    signal tmp_6437[3] <== [tmp_6435[0] + tmp_6436[0], tmp_6435[1] + tmp_6436[1], tmp_6435[2] + tmp_6436[2]];
    signal tmp_6438[3] <== [evals[102][0] - tmp_6437[0], evals[102][1] - tmp_6437[1], evals[102][2] - tmp_6437[2]];
    signal tmp_6439[3] <== CMul()(evals[78], tmp_6438);
    signal tmp_6440[3] <== [tmp_6428[0] + tmp_6439[0], tmp_6428[1] + tmp_6439[1], tmp_6428[2] + tmp_6439[2]];
    signal tmp_6441[3] <== CMul()(challengeQ, tmp_6440);
    signal tmp_6442[3] <== CMul()(evals[95], evals[100]);
    signal tmp_6443[3] <== CMul()(evals[97], evals[100]);
    signal tmp_6444[3] <== [tmp_6442[0] + tmp_6443[0], tmp_6442[1] + tmp_6443[1], tmp_6442[2] + tmp_6443[2]];
    signal tmp_6445[3] <== CMul()(evals[97], evals[98]);
    signal tmp_6446[3] <== [tmp_6444[0] + tmp_6445[0], tmp_6444[1] + tmp_6445[1], tmp_6444[2] + tmp_6445[2]];
    signal tmp_6447[3] <== CMul()(evals[96], evals[99]);
    signal tmp_6448[3] <== [tmp_6446[0] + tmp_6447[0], tmp_6446[1] + tmp_6447[1], tmp_6446[2] + tmp_6447[2]];
    signal tmp_6449[3] <== [evals[103][0] - tmp_6448[0], evals[103][1] - tmp_6448[1], evals[103][2] - tmp_6448[2]];
    signal tmp_6450[3] <== CMul()(evals[78], tmp_6449);
    signal tmp_6451[3] <== [tmp_6441[0] + tmp_6450[0], tmp_6441[1] + tmp_6450[1], tmp_6441[2] + tmp_6450[2]];
    signal tmp_6452[3] <== CMul()(challengeQ, tmp_6451);
    signal tmp_6453[3] <== CMul()(evals[104], evals[107]);
    signal tmp_6454[3] <== CMul()(evals[105], evals[109]);
    signal tmp_6455[3] <== [tmp_6453[0] + tmp_6454[0], tmp_6453[1] + tmp_6454[1], tmp_6453[2] + tmp_6454[2]];
    signal tmp_6456[3] <== CMul()(evals[106], evals[108]);
    signal tmp_6457[3] <== [tmp_6455[0] + tmp_6456[0], tmp_6455[1] + tmp_6456[1], tmp_6455[2] + tmp_6456[2]];
    signal tmp_6458[3] <== [evals[110][0] - tmp_6457[0], evals[110][1] - tmp_6457[1], evals[110][2] - tmp_6457[2]];
    signal tmp_6459[3] <== CMul()(evals[78], tmp_6458);
    signal tmp_6460[3] <== [tmp_6452[0] + tmp_6459[0], tmp_6452[1] + tmp_6459[1], tmp_6452[2] + tmp_6459[2]];
    signal tmp_6461[3] <== CMul()(challengeQ, tmp_6460);
    signal tmp_6462[3] <== CMul()(evals[104], evals[108]);
    signal tmp_6463[3] <== CMul()(evals[105], evals[107]);
    signal tmp_6464[3] <== [tmp_6462[0] + tmp_6463[0], tmp_6462[1] + tmp_6463[1], tmp_6462[2] + tmp_6463[2]];
    signal tmp_6465[3] <== CMul()(evals[105], evals[109]);
    signal tmp_6466[3] <== [tmp_6464[0] + tmp_6465[0], tmp_6464[1] + tmp_6465[1], tmp_6464[2] + tmp_6465[2]];
    signal tmp_6467[3] <== CMul()(evals[106], evals[108]);
    signal tmp_6468[3] <== [tmp_6466[0] + tmp_6467[0], tmp_6466[1] + tmp_6467[1], tmp_6466[2] + tmp_6467[2]];
    signal tmp_6469[3] <== CMul()(evals[106], evals[109]);
    signal tmp_6470[3] <== [tmp_6468[0] + tmp_6469[0], tmp_6468[1] + tmp_6469[1], tmp_6468[2] + tmp_6469[2]];
    signal tmp_6471[3] <== [evals[111][0] - tmp_6470[0], evals[111][1] - tmp_6470[1], evals[111][2] - tmp_6470[2]];
    signal tmp_6472[3] <== CMul()(evals[78], tmp_6471);
    signal tmp_6473[3] <== [tmp_6461[0] + tmp_6472[0], tmp_6461[1] + tmp_6472[1], tmp_6461[2] + tmp_6472[2]];
    signal tmp_6474[3] <== CMul()(challengeQ, tmp_6473);
    signal tmp_6475[3] <== CMul()(evals[104], evals[109]);
    signal tmp_6476[3] <== CMul()(evals[106], evals[109]);
    signal tmp_6477[3] <== [tmp_6475[0] + tmp_6476[0], tmp_6475[1] + tmp_6476[1], tmp_6475[2] + tmp_6476[2]];
    signal tmp_6478[3] <== CMul()(evals[106], evals[107]);
    signal tmp_6479[3] <== [tmp_6477[0] + tmp_6478[0], tmp_6477[1] + tmp_6478[1], tmp_6477[2] + tmp_6478[2]];
    signal tmp_6480[3] <== CMul()(evals[105], evals[108]);
    signal tmp_6481[3] <== [tmp_6479[0] + tmp_6480[0], tmp_6479[1] + tmp_6480[1], tmp_6479[2] + tmp_6480[2]];
    signal tmp_6482[3] <== [evals[112][0] - tmp_6481[0], evals[112][1] - tmp_6481[1], evals[112][2] - tmp_6481[2]];
    signal tmp_6483[3] <== CMul()(evals[78], tmp_6482);
    signal tmp_6484[3] <== [tmp_6474[0] + tmp_6483[0], tmp_6474[1] + tmp_6483[1], tmp_6474[2] + tmp_6483[2]];
    signal tmp_6485[3] <== CMul()(challengeQ, tmp_6484);
    signal tmp_6486[3] <== CMul()(evals[113], evals[116]);
    signal tmp_6487[3] <== CMul()(evals[114], evals[118]);
    signal tmp_6488[3] <== [tmp_6486[0] + tmp_6487[0], tmp_6486[1] + tmp_6487[1], tmp_6486[2] + tmp_6487[2]];
    signal tmp_6489[3] <== CMul()(evals[115], evals[117]);
    signal tmp_6490[3] <== [tmp_6488[0] + tmp_6489[0], tmp_6488[1] + tmp_6489[1], tmp_6488[2] + tmp_6489[2]];
    signal tmp_6491[3] <== [evals[119][0] - tmp_6490[0], evals[119][1] - tmp_6490[1], evals[119][2] - tmp_6490[2]];
    signal tmp_6492[3] <== CMul()(evals[78], tmp_6491);
    signal tmp_6493[3] <== [tmp_6485[0] + tmp_6492[0], tmp_6485[1] + tmp_6492[1], tmp_6485[2] + tmp_6492[2]];
    signal tmp_6494[3] <== CMul()(challengeQ, tmp_6493);
    signal tmp_6495[3] <== CMul()(evals[113], evals[117]);
    signal tmp_6496[3] <== CMul()(evals[114], evals[116]);
    signal tmp_6497[3] <== [tmp_6495[0] + tmp_6496[0], tmp_6495[1] + tmp_6496[1], tmp_6495[2] + tmp_6496[2]];
    signal tmp_6498[3] <== CMul()(evals[114], evals[118]);
    signal tmp_6499[3] <== [tmp_6497[0] + tmp_6498[0], tmp_6497[1] + tmp_6498[1], tmp_6497[2] + tmp_6498[2]];
    signal tmp_6500[3] <== CMul()(evals[115], evals[117]);
    signal tmp_6501[3] <== [tmp_6499[0] + tmp_6500[0], tmp_6499[1] + tmp_6500[1], tmp_6499[2] + tmp_6500[2]];
    signal tmp_6502[3] <== CMul()(evals[115], evals[118]);
    signal tmp_6503[3] <== [tmp_6501[0] + tmp_6502[0], tmp_6501[1] + tmp_6502[1], tmp_6501[2] + tmp_6502[2]];
    signal tmp_6504[3] <== [evals[120][0] - tmp_6503[0], evals[120][1] - tmp_6503[1], evals[120][2] - tmp_6503[2]];
    signal tmp_6505[3] <== CMul()(evals[78], tmp_6504);
    signal tmp_6506[3] <== [tmp_6494[0] + tmp_6505[0], tmp_6494[1] + tmp_6505[1], tmp_6494[2] + tmp_6505[2]];
    signal tmp_6507[3] <== CMul()(challengeQ, tmp_6506);
    signal tmp_6508[3] <== CMul()(evals[113], evals[118]);
    signal tmp_6509[3] <== CMul()(evals[115], evals[118]);
    signal tmp_6510[3] <== [tmp_6508[0] + tmp_6509[0], tmp_6508[1] + tmp_6509[1], tmp_6508[2] + tmp_6509[2]];
    signal tmp_6511[3] <== CMul()(evals[115], evals[116]);
    signal tmp_6512[3] <== [tmp_6510[0] + tmp_6511[0], tmp_6510[1] + tmp_6511[1], tmp_6510[2] + tmp_6511[2]];
    signal tmp_6513[3] <== CMul()(evals[114], evals[117]);
    signal tmp_6514[3] <== [tmp_6512[0] + tmp_6513[0], tmp_6512[1] + tmp_6513[1], tmp_6512[2] + tmp_6513[2]];
    signal tmp_6515[3] <== [evals[121][0] - tmp_6514[0], evals[121][1] - tmp_6514[1], evals[121][2] - tmp_6514[2]];
    signal tmp_6516[3] <== CMul()(evals[78], tmp_6515);
    signal tmp_6517[3] <== [tmp_6507[0] + tmp_6516[0], tmp_6507[1] + tmp_6516[1], tmp_6507[2] + tmp_6516[2]];
    signal tmp_6518[3] <== CMul()(challengeQ, tmp_6517);
    signal tmp_6519[3] <== CMul()(evals[64], evals[86]);
    signal tmp_6520[3] <== CMul()(evals[65], evals[89]);
    signal tmp_6521[3] <== [tmp_6519[0] + tmp_6520[0], tmp_6519[1] + tmp_6520[1], tmp_6519[2] + tmp_6520[2]];
    signal tmp_6522[3] <== CMul()(evals[66], evals[92]);
    signal tmp_6523[3] <== [tmp_6521[0] + tmp_6522[0], tmp_6521[1] + tmp_6522[1], tmp_6521[2] + tmp_6522[2]];
    signal tmp_6524[3] <== CMul()(evals[67], evals[95]);
    signal tmp_6525[3] <== [tmp_6523[0] + tmp_6524[0], tmp_6523[1] + tmp_6524[1], tmp_6523[2] + tmp_6524[2]];
    signal tmp_6526[3] <== CMul()(evals[70], evals[86]);
    signal tmp_6527[3] <== [tmp_6525[0] + tmp_6526[0], tmp_6525[1] + tmp_6526[1], tmp_6525[2] + tmp_6526[2]];
    signal tmp_6528[3] <== CMul()(evals[71], evals[89]);
    signal tmp_6529[3] <== [tmp_6527[0] + tmp_6528[0], tmp_6527[1] + tmp_6528[1], tmp_6527[2] + tmp_6528[2]];
    signal tmp_6530[3] <== [evals[98][0] - tmp_6529[0], evals[98][1] - tmp_6529[1], evals[98][2] - tmp_6529[2]];
    signal tmp_6531[3] <== CMul()(evals[80], tmp_6530);
    signal tmp_6532[3] <== [tmp_6518[0] + tmp_6531[0], tmp_6518[1] + tmp_6531[1], tmp_6518[2] + tmp_6531[2]];
    signal tmp_6533[3] <== CMul()(challengeQ, tmp_6532);
    signal tmp_6534[3] <== CMul()(evals[64], evals[87]);
    signal tmp_6535[3] <== CMul()(evals[65], evals[90]);
    signal tmp_6536[3] <== [tmp_6534[0] + tmp_6535[0], tmp_6534[1] + tmp_6535[1], tmp_6534[2] + tmp_6535[2]];
    signal tmp_6537[3] <== CMul()(evals[66], evals[93]);
    signal tmp_6538[3] <== [tmp_6536[0] + tmp_6537[0], tmp_6536[1] + tmp_6537[1], tmp_6536[2] + tmp_6537[2]];
    signal tmp_6539[3] <== CMul()(evals[67], evals[96]);
    signal tmp_6540[3] <== [tmp_6538[0] + tmp_6539[0], tmp_6538[1] + tmp_6539[1], tmp_6538[2] + tmp_6539[2]];
    signal tmp_6541[3] <== CMul()(evals[70], evals[87]);
    signal tmp_6542[3] <== [tmp_6540[0] + tmp_6541[0], tmp_6540[1] + tmp_6541[1], tmp_6540[2] + tmp_6541[2]];
    signal tmp_6543[3] <== CMul()(evals[71], evals[90]);
    signal tmp_6544[3] <== [tmp_6542[0] + tmp_6543[0], tmp_6542[1] + tmp_6543[1], tmp_6542[2] + tmp_6543[2]];
    signal tmp_6545[3] <== [evals[99][0] - tmp_6544[0], evals[99][1] - tmp_6544[1], evals[99][2] - tmp_6544[2]];
    signal tmp_6546[3] <== CMul()(evals[80], tmp_6545);
    signal tmp_6547[3] <== [tmp_6533[0] + tmp_6546[0], tmp_6533[1] + tmp_6546[1], tmp_6533[2] + tmp_6546[2]];
    signal tmp_6548[3] <== CMul()(challengeQ, tmp_6547);
    signal tmp_6549[3] <== CMul()(evals[64], evals[88]);
    signal tmp_6550[3] <== CMul()(evals[65], evals[91]);
    signal tmp_6551[3] <== [tmp_6549[0] + tmp_6550[0], tmp_6549[1] + tmp_6550[1], tmp_6549[2] + tmp_6550[2]];
    signal tmp_6552[3] <== CMul()(evals[66], evals[94]);
    signal tmp_6553[3] <== [tmp_6551[0] + tmp_6552[0], tmp_6551[1] + tmp_6552[1], tmp_6551[2] + tmp_6552[2]];
    signal tmp_6554[3] <== CMul()(evals[67], evals[97]);
    signal tmp_6555[3] <== [tmp_6553[0] + tmp_6554[0], tmp_6553[1] + tmp_6554[1], tmp_6553[2] + tmp_6554[2]];
    signal tmp_6556[3] <== CMul()(evals[70], evals[88]);
    signal tmp_6557[3] <== [tmp_6555[0] + tmp_6556[0], tmp_6555[1] + tmp_6556[1], tmp_6555[2] + tmp_6556[2]];
    signal tmp_6558[3] <== CMul()(evals[71], evals[91]);
    signal tmp_6559[3] <== [tmp_6557[0] + tmp_6558[0], tmp_6557[1] + tmp_6558[1], tmp_6557[2] + tmp_6558[2]];
    signal tmp_6560[3] <== [evals[100][0] - tmp_6559[0], evals[100][1] - tmp_6559[1], evals[100][2] - tmp_6559[2]];
    signal tmp_6561[3] <== CMul()(evals[80], tmp_6560);
    signal tmp_6562[3] <== [tmp_6548[0] + tmp_6561[0], tmp_6548[1] + tmp_6561[1], tmp_6548[2] + tmp_6561[2]];
    signal tmp_6563[3] <== CMul()(challengeQ, tmp_6562);
    signal tmp_6564[3] <== CMul()(evals[64], evals[86]);
    signal tmp_6565[3] <== CMul()(evals[65], evals[89]);
    signal tmp_6566[3] <== [tmp_6564[0] - tmp_6565[0], tmp_6564[1] - tmp_6565[1], tmp_6564[2] - tmp_6565[2]];
    signal tmp_6567[3] <== CMul()(evals[68], evals[92]);
    signal tmp_6568[3] <== [tmp_6566[0] + tmp_6567[0], tmp_6566[1] + tmp_6567[1], tmp_6566[2] + tmp_6567[2]];
    signal tmp_6569[3] <== CMul()(evals[69], evals[95]);
    signal tmp_6570[3] <== [tmp_6568[0] - tmp_6569[0], tmp_6568[1] - tmp_6569[1], tmp_6568[2] - tmp_6569[2]];
    signal tmp_6571[3] <== CMul()(evals[70], evals[86]);
    signal tmp_6572[3] <== [tmp_6570[0] + tmp_6571[0], tmp_6570[1] + tmp_6571[1], tmp_6570[2] + tmp_6571[2]];
    signal tmp_6573[3] <== CMul()(evals[71], evals[89]);
    signal tmp_6574[3] <== [tmp_6572[0] - tmp_6573[0], tmp_6572[1] - tmp_6573[1], tmp_6572[2] - tmp_6573[2]];
    signal tmp_6575[3] <== [evals[101][0] - tmp_6574[0], evals[101][1] - tmp_6574[1], evals[101][2] - tmp_6574[2]];
    signal tmp_6576[3] <== CMul()(evals[80], tmp_6575);
    signal tmp_6577[3] <== [tmp_6563[0] + tmp_6576[0], tmp_6563[1] + tmp_6576[1], tmp_6563[2] + tmp_6576[2]];
    signal tmp_6578[3] <== CMul()(challengeQ, tmp_6577);
    signal tmp_6579[3] <== CMul()(evals[64], evals[87]);
    signal tmp_6580[3] <== CMul()(evals[65], evals[90]);
    signal tmp_6581[3] <== [tmp_6579[0] - tmp_6580[0], tmp_6579[1] - tmp_6580[1], tmp_6579[2] - tmp_6580[2]];
    signal tmp_6582[3] <== CMul()(evals[68], evals[93]);
    signal tmp_6583[3] <== [tmp_6581[0] + tmp_6582[0], tmp_6581[1] + tmp_6582[1], tmp_6581[2] + tmp_6582[2]];
    signal tmp_6584[3] <== CMul()(evals[69], evals[96]);
    signal tmp_6585[3] <== [tmp_6583[0] - tmp_6584[0], tmp_6583[1] - tmp_6584[1], tmp_6583[2] - tmp_6584[2]];
    signal tmp_6586[3] <== CMul()(evals[70], evals[87]);
    signal tmp_6587[3] <== [tmp_6585[0] + tmp_6586[0], tmp_6585[1] + tmp_6586[1], tmp_6585[2] + tmp_6586[2]];
    signal tmp_6588[3] <== CMul()(evals[71], evals[90]);
    signal tmp_6589[3] <== [tmp_6587[0] - tmp_6588[0], tmp_6587[1] - tmp_6588[1], tmp_6587[2] - tmp_6588[2]];
    signal tmp_6590[3] <== [evals[102][0] - tmp_6589[0], evals[102][1] - tmp_6589[1], evals[102][2] - tmp_6589[2]];
    signal tmp_6591[3] <== CMul()(evals[80], tmp_6590);
    signal tmp_6592[3] <== [tmp_6578[0] + tmp_6591[0], tmp_6578[1] + tmp_6591[1], tmp_6578[2] + tmp_6591[2]];
    signal tmp_6593[3] <== CMul()(challengeQ, tmp_6592);
    signal tmp_6594[3] <== CMul()(evals[64], evals[88]);
    signal tmp_6595[3] <== CMul()(evals[65], evals[91]);
    signal tmp_6596[3] <== [tmp_6594[0] - tmp_6595[0], tmp_6594[1] - tmp_6595[1], tmp_6594[2] - tmp_6595[2]];
    signal tmp_6597[3] <== CMul()(evals[68], evals[94]);
    signal tmp_6598[3] <== [tmp_6596[0] + tmp_6597[0], tmp_6596[1] + tmp_6597[1], tmp_6596[2] + tmp_6597[2]];
    signal tmp_6599[3] <== CMul()(evals[69], evals[97]);
    signal tmp_6600[3] <== [tmp_6598[0] - tmp_6599[0], tmp_6598[1] - tmp_6599[1], tmp_6598[2] - tmp_6599[2]];
    signal tmp_6601[3] <== CMul()(evals[70], evals[88]);
    signal tmp_6602[3] <== [tmp_6600[0] + tmp_6601[0], tmp_6600[1] + tmp_6601[1], tmp_6600[2] + tmp_6601[2]];
    signal tmp_6603[3] <== CMul()(evals[71], evals[91]);
    signal tmp_6604[3] <== [tmp_6602[0] - tmp_6603[0], tmp_6602[1] - tmp_6603[1], tmp_6602[2] - tmp_6603[2]];
    signal tmp_6605[3] <== [evals[103][0] - tmp_6604[0], evals[103][1] - tmp_6604[1], evals[103][2] - tmp_6604[2]];
    signal tmp_6606[3] <== CMul()(evals[80], tmp_6605);
    signal tmp_6607[3] <== [tmp_6593[0] + tmp_6606[0], tmp_6593[1] + tmp_6606[1], tmp_6593[2] + tmp_6606[2]];
    signal tmp_6608[3] <== CMul()(challengeQ, tmp_6607);
    signal tmp_6609[3] <== CMul()(evals[64], evals[86]);
    signal tmp_6610[3] <== CMul()(evals[65], evals[89]);
    signal tmp_6611[3] <== [tmp_6609[0] + tmp_6610[0], tmp_6609[1] + tmp_6610[1], tmp_6609[2] + tmp_6610[2]];
    signal tmp_6612[3] <== CMul()(evals[66], evals[92]);
    signal tmp_6613[3] <== [tmp_6611[0] - tmp_6612[0], tmp_6611[1] - tmp_6612[1], tmp_6611[2] - tmp_6612[2]];
    signal tmp_6614[3] <== CMul()(evals[67], evals[95]);
    signal tmp_6615[3] <== [tmp_6613[0] - tmp_6614[0], tmp_6613[1] - tmp_6614[1], tmp_6613[2] - tmp_6614[2]];
    signal tmp_6616[3] <== CMul()(evals[70], evals[92]);
    signal tmp_6617[3] <== [tmp_6615[0] + tmp_6616[0], tmp_6615[1] + tmp_6616[1], tmp_6615[2] + tmp_6616[2]];
    signal tmp_6618[3] <== CMul()(evals[72], evals[95]);
    signal tmp_6619[3] <== [tmp_6617[0] + tmp_6618[0], tmp_6617[1] + tmp_6618[1], tmp_6617[2] + tmp_6618[2]];
    signal tmp_6620[3] <== [evals[104][0] - tmp_6619[0], evals[104][1] - tmp_6619[1], evals[104][2] - tmp_6619[2]];
    signal tmp_6621[3] <== CMul()(evals[80], tmp_6620);
    signal tmp_6622[3] <== [tmp_6608[0] + tmp_6621[0], tmp_6608[1] + tmp_6621[1], tmp_6608[2] + tmp_6621[2]];
    signal tmp_6623[3] <== CMul()(challengeQ, tmp_6622);
    signal tmp_6624[3] <== CMul()(evals[64], evals[87]);
    signal tmp_6625[3] <== CMul()(evals[65], evals[90]);
    signal tmp_6626[3] <== [tmp_6624[0] + tmp_6625[0], tmp_6624[1] + tmp_6625[1], tmp_6624[2] + tmp_6625[2]];
    signal tmp_6627[3] <== CMul()(evals[66], evals[93]);
    signal tmp_6628[3] <== [tmp_6626[0] - tmp_6627[0], tmp_6626[1] - tmp_6627[1], tmp_6626[2] - tmp_6627[2]];
    signal tmp_6629[3] <== CMul()(evals[67], evals[96]);
    signal tmp_6630[3] <== [tmp_6628[0] - tmp_6629[0], tmp_6628[1] - tmp_6629[1], tmp_6628[2] - tmp_6629[2]];
    signal tmp_6631[3] <== CMul()(evals[70], evals[93]);
    signal tmp_6632[3] <== [tmp_6630[0] + tmp_6631[0], tmp_6630[1] + tmp_6631[1], tmp_6630[2] + tmp_6631[2]];
    signal tmp_6633[3] <== CMul()(evals[72], evals[96]);
    signal tmp_6634[3] <== [tmp_6632[0] + tmp_6633[0], tmp_6632[1] + tmp_6633[1], tmp_6632[2] + tmp_6633[2]];
    signal tmp_6635[3] <== [evals[105][0] - tmp_6634[0], evals[105][1] - tmp_6634[1], evals[105][2] - tmp_6634[2]];
    signal tmp_6636[3] <== CMul()(evals[80], tmp_6635);
    signal tmp_6637[3] <== [tmp_6623[0] + tmp_6636[0], tmp_6623[1] + tmp_6636[1], tmp_6623[2] + tmp_6636[2]];
    signal tmp_6638[3] <== CMul()(challengeQ, tmp_6637);
    signal tmp_6639[3] <== CMul()(evals[64], evals[88]);
    signal tmp_6640[3] <== CMul()(evals[65], evals[91]);
    signal tmp_6641[3] <== [tmp_6639[0] + tmp_6640[0], tmp_6639[1] + tmp_6640[1], tmp_6639[2] + tmp_6640[2]];
    signal tmp_6642[3] <== CMul()(evals[66], evals[94]);
    signal tmp_6643[3] <== [tmp_6641[0] - tmp_6642[0], tmp_6641[1] - tmp_6642[1], tmp_6641[2] - tmp_6642[2]];
    signal tmp_6644[3] <== CMul()(evals[67], evals[97]);
    signal tmp_6645[3] <== [tmp_6643[0] - tmp_6644[0], tmp_6643[1] - tmp_6644[1], tmp_6643[2] - tmp_6644[2]];
    signal tmp_6646[3] <== CMul()(evals[70], evals[94]);
    signal tmp_6647[3] <== [tmp_6645[0] + tmp_6646[0], tmp_6645[1] + tmp_6646[1], tmp_6645[2] + tmp_6646[2]];
    signal tmp_6648[3] <== CMul()(evals[72], evals[97]);
    signal tmp_6649[3] <== [tmp_6647[0] + tmp_6648[0], tmp_6647[1] + tmp_6648[1], tmp_6647[2] + tmp_6648[2]];
    signal tmp_6650[3] <== [evals[106][0] - tmp_6649[0], evals[106][1] - tmp_6649[1], evals[106][2] - tmp_6649[2]];
    signal tmp_6651[3] <== CMul()(evals[80], tmp_6650);
    signal tmp_6652[3] <== [tmp_6638[0] + tmp_6651[0], tmp_6638[1] + tmp_6651[1], tmp_6638[2] + tmp_6651[2]];
    signal tmp_6653[3] <== CMul()(challengeQ, tmp_6652);
    signal tmp_6654[3] <== CMul()(evals[64], evals[86]);
    signal tmp_6655[3] <== CMul()(evals[65], evals[89]);
    signal tmp_6656[3] <== [tmp_6654[0] - tmp_6655[0], tmp_6654[1] - tmp_6655[1], tmp_6654[2] - tmp_6655[2]];
    signal tmp_6657[3] <== CMul()(evals[68], evals[92]);
    signal tmp_6658[3] <== [tmp_6656[0] - tmp_6657[0], tmp_6656[1] - tmp_6657[1], tmp_6656[2] - tmp_6657[2]];
    signal tmp_6659[3] <== CMul()(evals[69], evals[95]);
    signal tmp_6660[3] <== [tmp_6658[0] + tmp_6659[0], tmp_6658[1] + tmp_6659[1], tmp_6658[2] + tmp_6659[2]];
    signal tmp_6661[3] <== CMul()(evals[70], evals[92]);
    signal tmp_6662[3] <== [tmp_6660[0] + tmp_6661[0], tmp_6660[1] + tmp_6661[1], tmp_6660[2] + tmp_6661[2]];
    signal tmp_6663[3] <== CMul()(evals[72], evals[95]);
    signal tmp_6664[3] <== [tmp_6662[0] - tmp_6663[0], tmp_6662[1] - tmp_6663[1], tmp_6662[2] - tmp_6663[2]];
    signal tmp_6665[3] <== [evals[107][0] - tmp_6664[0], evals[107][1] - tmp_6664[1], evals[107][2] - tmp_6664[2]];
    signal tmp_6666[3] <== CMul()(evals[80], tmp_6665);
    signal tmp_6667[3] <== [tmp_6653[0] + tmp_6666[0], tmp_6653[1] + tmp_6666[1], tmp_6653[2] + tmp_6666[2]];
    signal tmp_6668[3] <== CMul()(challengeQ, tmp_6667);
    signal tmp_6669[3] <== CMul()(evals[64], evals[87]);
    signal tmp_6670[3] <== CMul()(evals[65], evals[90]);
    signal tmp_6671[3] <== [tmp_6669[0] - tmp_6670[0], tmp_6669[1] - tmp_6670[1], tmp_6669[2] - tmp_6670[2]];
    signal tmp_6672[3] <== CMul()(evals[68], evals[93]);
    signal tmp_6673[3] <== [tmp_6671[0] - tmp_6672[0], tmp_6671[1] - tmp_6672[1], tmp_6671[2] - tmp_6672[2]];
    signal tmp_6674[3] <== CMul()(evals[69], evals[96]);
    signal tmp_6675[3] <== [tmp_6673[0] + tmp_6674[0], tmp_6673[1] + tmp_6674[1], tmp_6673[2] + tmp_6674[2]];
    signal tmp_6676[3] <== CMul()(evals[70], evals[93]);
    signal tmp_6677[3] <== [tmp_6675[0] + tmp_6676[0], tmp_6675[1] + tmp_6676[1], tmp_6675[2] + tmp_6676[2]];
    signal tmp_6678[3] <== CMul()(evals[72], evals[96]);
    signal tmp_6679[3] <== [tmp_6677[0] - tmp_6678[0], tmp_6677[1] - tmp_6678[1], tmp_6677[2] - tmp_6678[2]];
    signal tmp_6680[3] <== [evals[108][0] - tmp_6679[0], evals[108][1] - tmp_6679[1], evals[108][2] - tmp_6679[2]];
    signal tmp_6681[3] <== CMul()(evals[80], tmp_6680);
    signal tmp_6682[3] <== [tmp_6668[0] + tmp_6681[0], tmp_6668[1] + tmp_6681[1], tmp_6668[2] + tmp_6681[2]];
    signal tmp_6683[3] <== CMul()(challengeQ, tmp_6682);
    signal tmp_6684[3] <== CMul()(evals[64], evals[88]);
    signal tmp_6685[3] <== CMul()(evals[65], evals[91]);
    signal tmp_6686[3] <== [tmp_6684[0] - tmp_6685[0], tmp_6684[1] - tmp_6685[1], tmp_6684[2] - tmp_6685[2]];
    signal tmp_6687[3] <== CMul()(evals[68], evals[94]);
    signal tmp_6688[3] <== [tmp_6686[0] - tmp_6687[0], tmp_6686[1] - tmp_6687[1], tmp_6686[2] - tmp_6687[2]];
    signal tmp_6689[3] <== CMul()(evals[69], evals[97]);
    signal tmp_6690[3] <== [tmp_6688[0] + tmp_6689[0], tmp_6688[1] + tmp_6689[1], tmp_6688[2] + tmp_6689[2]];
    signal tmp_6691[3] <== CMul()(evals[70], evals[94]);
    signal tmp_6692[3] <== [tmp_6690[0] + tmp_6691[0], tmp_6690[1] + tmp_6691[1], tmp_6690[2] + tmp_6691[2]];
    signal tmp_6693[3] <== CMul()(evals[72], evals[97]);
    signal tmp_6694[3] <== [tmp_6692[0] - tmp_6693[0], tmp_6692[1] - tmp_6693[1], tmp_6692[2] - tmp_6693[2]];
    signal tmp_6695[3] <== [evals[109][0] - tmp_6694[0], evals[109][1] - tmp_6694[1], evals[109][2] - tmp_6694[2]];
    signal tmp_6696[3] <== CMul()(evals[80], tmp_6695);
    signal tmp_6697[3] <== [tmp_6683[0] + tmp_6696[0], tmp_6683[1] + tmp_6696[1], tmp_6683[2] + tmp_6696[2]];
    signal tmp_6698[3] <== CMul()(challengeQ, tmp_6697);
    signal tmp_6699[3] <== CMul()(evals[79], evals[169]);
    signal tmp_6700[3] <== [tmp_6698[0] + tmp_6699[0], tmp_6698[1] + tmp_6699[1], tmp_6698[2] + tmp_6699[2]];
    signal tmp_6701[3] <== CMul()(challengeQ, tmp_6700);
    signal tmp_6702[3] <== CMul()(evals[79], evals[170]);
    signal tmp_6703[3] <== [tmp_6701[0] + tmp_6702[0], tmp_6701[1] + tmp_6702[1], tmp_6701[2] + tmp_6702[2]];
    signal tmp_6704[3] <== CMul()(challengeQ, tmp_6703);
    signal tmp_6705[3] <== CMul()(evals[79], evals[171]);
    signal tmp_6706[3] <== [tmp_6704[0] + tmp_6705[0], tmp_6704[1] + tmp_6705[1], tmp_6704[2] + tmp_6705[2]];
    signal tmp_6707[3] <== CMul()(challengeQ, tmp_6706);
    signal tmp_6708[3] <== [1 - evals[98][0], -evals[98][1], -evals[98][2]];
    signal tmp_6709[3] <== [1 - evals[99][0], -evals[99][1], -evals[99][2]];
    signal tmp_6710[3] <== CMul()(tmp_6708, tmp_6709);
    signal tmp_6711[3] <== CMul()(evals[81], tmp_6710);
    signal tmp_6712[3] <== [evals[86][0] - evals[100][0], evals[86][1] - evals[100][1], evals[86][2] - evals[100][2]];
    signal tmp_6713[3] <== CMul()(tmp_6711, tmp_6712);
    signal tmp_6714[3] <== [tmp_6707[0] + tmp_6713[0], tmp_6707[1] + tmp_6713[1], tmp_6707[2] + tmp_6713[2]];
    signal tmp_6715[3] <== CMul()(challengeQ, tmp_6714);
    signal tmp_6716[3] <== CMul()(evals[81], tmp_6710);
    signal tmp_6717[3] <== [evals[87][0] - evals[101][0], evals[87][1] - evals[101][1], evals[87][2] - evals[101][2]];
    signal tmp_6718[3] <== CMul()(tmp_6716, tmp_6717);
    signal tmp_6719[3] <== [tmp_6715[0] + tmp_6718[0], tmp_6715[1] + tmp_6718[1], tmp_6715[2] + tmp_6718[2]];
    signal tmp_6720[3] <== CMul()(challengeQ, tmp_6719);
    signal tmp_6721[3] <== CMul()(evals[81], tmp_6710);
    signal tmp_6722[3] <== [evals[88][0] - evals[102][0], evals[88][1] - evals[102][1], evals[88][2] - evals[102][2]];
    signal tmp_6723[3] <== CMul()(tmp_6721, tmp_6722);
    signal tmp_6724[3] <== [tmp_6720[0] + tmp_6723[0], tmp_6720[1] + tmp_6723[1], tmp_6720[2] + tmp_6723[2]];
    signal tmp_6725[3] <== CMul()(challengeQ, tmp_6724);
    signal tmp_6726[3] <== [1 - evals[99][0], -evals[99][1], -evals[99][2]];
    signal tmp_6727[3] <== CMul()(evals[98], tmp_6726);
    signal tmp_6728[3] <== CMul()(evals[81], tmp_6727);
    signal tmp_6729[3] <== [evals[89][0] - evals[100][0], evals[89][1] - evals[100][1], evals[89][2] - evals[100][2]];
    signal tmp_6730[3] <== CMul()(tmp_6728, tmp_6729);
    signal tmp_6731[3] <== [tmp_6725[0] + tmp_6730[0], tmp_6725[1] + tmp_6730[1], tmp_6725[2] + tmp_6730[2]];
    signal tmp_6732[3] <== CMul()(challengeQ, tmp_6731);
    signal tmp_6733[3] <== CMul()(evals[81], tmp_6727);
    signal tmp_6734[3] <== [evals[90][0] - evals[101][0], evals[90][1] - evals[101][1], evals[90][2] - evals[101][2]];
    signal tmp_6735[3] <== CMul()(tmp_6733, tmp_6734);
    signal tmp_6736[3] <== [tmp_6732[0] + tmp_6735[0], tmp_6732[1] + tmp_6735[1], tmp_6732[2] + tmp_6735[2]];
    signal tmp_6737[3] <== CMul()(challengeQ, tmp_6736);
    signal tmp_6738[3] <== CMul()(evals[81], tmp_6727);
    signal tmp_6739[3] <== [evals[91][0] - evals[102][0], evals[91][1] - evals[102][1], evals[91][2] - evals[102][2]];
    signal tmp_6740[3] <== CMul()(tmp_6738, tmp_6739);
    signal tmp_6741[3] <== [tmp_6737[0] + tmp_6740[0], tmp_6737[1] + tmp_6740[1], tmp_6737[2] + tmp_6740[2]];
    signal tmp_6742[3] <== CMul()(challengeQ, tmp_6741);
    signal tmp_6743[3] <== [1 - evals[98][0], -evals[98][1], -evals[98][2]];
    signal tmp_6744[3] <== CMul()(tmp_6743, evals[99]);
    signal tmp_6745[3] <== CMul()(evals[81], tmp_6744);
    signal tmp_6746[3] <== [evals[92][0] - evals[100][0], evals[92][1] - evals[100][1], evals[92][2] - evals[100][2]];
    signal tmp_6747[3] <== CMul()(tmp_6745, tmp_6746);
    signal tmp_6748[3] <== [tmp_6742[0] + tmp_6747[0], tmp_6742[1] + tmp_6747[1], tmp_6742[2] + tmp_6747[2]];
    signal tmp_6749[3] <== CMul()(challengeQ, tmp_6748);
    signal tmp_6750[3] <== CMul()(evals[81], tmp_6744);
    signal tmp_6751[3] <== [evals[93][0] - evals[101][0], evals[93][1] - evals[101][1], evals[93][2] - evals[101][2]];
    signal tmp_6752[3] <== CMul()(tmp_6750, tmp_6751);
    signal tmp_6753[3] <== [tmp_6749[0] + tmp_6752[0], tmp_6749[1] + tmp_6752[1], tmp_6749[2] + tmp_6752[2]];
    signal tmp_6754[3] <== CMul()(challengeQ, tmp_6753);
    signal tmp_6755[3] <== CMul()(evals[81], tmp_6744);
    signal tmp_6756[3] <== [evals[94][0] - evals[102][0], evals[94][1] - evals[102][1], evals[94][2] - evals[102][2]];
    signal tmp_6757[3] <== CMul()(tmp_6755, tmp_6756);
    signal tmp_6758[3] <== [tmp_6754[0] + tmp_6757[0], tmp_6754[1] + tmp_6757[1], tmp_6754[2] + tmp_6757[2]];
    signal tmp_6759[3] <== CMul()(challengeQ, tmp_6758);
    signal tmp_6760[3] <== CMul()(evals[98], evals[99]);
    signal tmp_6761[3] <== CMul()(evals[81], tmp_6760);
    signal tmp_6762[3] <== [evals[95][0] - evals[100][0], evals[95][1] - evals[100][1], evals[95][2] - evals[100][2]];
    signal tmp_6763[3] <== CMul()(tmp_6761, tmp_6762);
    signal tmp_6764[3] <== [tmp_6759[0] + tmp_6763[0], tmp_6759[1] + tmp_6763[1], tmp_6759[2] + tmp_6763[2]];
    signal tmp_6765[3] <== CMul()(challengeQ, tmp_6764);
    signal tmp_6766[3] <== CMul()(evals[81], tmp_6760);
    signal tmp_6767[3] <== [evals[96][0] - evals[101][0], evals[96][1] - evals[101][1], evals[96][2] - evals[101][2]];
    signal tmp_6768[3] <== CMul()(tmp_6766, tmp_6767);
    signal tmp_6769[3] <== [tmp_6765[0] + tmp_6768[0], tmp_6765[1] + tmp_6768[1], tmp_6765[2] + tmp_6768[2]];
    signal tmp_6770[3] <== CMul()(challengeQ, tmp_6769);
    signal tmp_6771[3] <== CMul()(evals[81], tmp_6760);
    signal tmp_6772[3] <== [evals[97][0] - evals[102][0], evals[97][1] - evals[102][1], evals[97][2] - evals[102][2]];
    signal tmp_6773[3] <== CMul()(tmp_6771, tmp_6772);
    signal tmp_6774[3] <== [tmp_6770[0] + tmp_6773[0], tmp_6770[1] + tmp_6773[1], tmp_6770[2] + tmp_6773[2]];
    signal tmp_6775[3] <== CMul()(challengeQ, tmp_6774);
    signal tmp_6776[3] <== [1 - evals[98][0], -evals[98][1], -evals[98][2]];
    signal tmp_6777[3] <== CMul()(evals[98], tmp_6776);
    signal tmp_6778[3] <== CMul()(evals[81], tmp_6777);
    signal tmp_6779[3] <== [tmp_6775[0] + tmp_6778[0], tmp_6775[1] + tmp_6778[1], tmp_6775[2] + tmp_6778[2]];
    signal tmp_6780[3] <== CMul()(challengeQ, tmp_6779);
    signal tmp_6781[3] <== [1 - evals[99][0], -evals[99][1], -evals[99][2]];
    signal tmp_6782[3] <== CMul()(evals[99], tmp_6781);
    signal tmp_6783[3] <== CMul()(evals[81], tmp_6782);
    signal tmp_6784[3] <== [tmp_6780[0] + tmp_6783[0], tmp_6780[1] + tmp_6783[1], tmp_6780[2] + tmp_6783[2]];
    signal tmp_6785[3] <== CMul()(challengeQ, tmp_6784);
    signal tmp_6786[3] <== [1 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_6787[3] <== [1 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_6788[3] <== CMul()(tmp_6786, tmp_6787);
    signal tmp_6789[3] <== CMul()(evals[82], tmp_6788);
    signal tmp_6790[3] <== [evals[86][0] - evals[104][0], evals[86][1] - evals[104][1], evals[86][2] - evals[104][2]];
    signal tmp_6791[3] <== CMul()(tmp_6789, tmp_6790);
    signal tmp_6792[3] <== [tmp_6785[0] + tmp_6791[0], tmp_6785[1] + tmp_6791[1], tmp_6785[2] + tmp_6791[2]];
    signal tmp_6793[3] <== CMul()(challengeQ, tmp_6792);
    signal tmp_6794[3] <== CMul()(evals[82], tmp_6788);
    signal tmp_6795[3] <== [evals[87][0] - evals[105][0], evals[87][1] - evals[105][1], evals[87][2] - evals[105][2]];
    signal tmp_6796[3] <== CMul()(tmp_6794, tmp_6795);
    signal tmp_6797[3] <== [tmp_6793[0] + tmp_6796[0], tmp_6793[1] + tmp_6796[1], tmp_6793[2] + tmp_6796[2]];
    signal tmp_6798[3] <== CMul()(challengeQ, tmp_6797);
    signal tmp_6799[3] <== CMul()(evals[82], tmp_6788);
    signal tmp_6800[3] <== [evals[88][0] - evals[106][0], evals[88][1] - evals[106][1], evals[88][2] - evals[106][2]];
    signal tmp_6801[3] <== CMul()(tmp_6799, tmp_6800);
    signal tmp_6802[3] <== [tmp_6798[0] + tmp_6801[0], tmp_6798[1] + tmp_6801[1], tmp_6798[2] + tmp_6801[2]];
    signal tmp_6803[3] <== CMul()(challengeQ, tmp_6802);
    signal tmp_6804[3] <== CMul()(evals[82], tmp_6788);
    signal tmp_6805[3] <== [evals[89][0] - evals[107][0], evals[89][1] - evals[107][1], evals[89][2] - evals[107][2]];
    signal tmp_6806[3] <== CMul()(tmp_6804, tmp_6805);
    signal tmp_6807[3] <== [tmp_6803[0] + tmp_6806[0], tmp_6803[1] + tmp_6806[1], tmp_6803[2] + tmp_6806[2]];
    signal tmp_6808[3] <== CMul()(challengeQ, tmp_6807);
    signal tmp_6809[3] <== [1 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_6810[3] <== CMul()(evals[102], tmp_6809);
    signal tmp_6811[3] <== CMul()(evals[82], tmp_6810);
    signal tmp_6812[3] <== [evals[90][0] - evals[104][0], evals[90][1] - evals[104][1], evals[90][2] - evals[104][2]];
    signal tmp_6813[3] <== CMul()(tmp_6811, tmp_6812);
    signal tmp_6814[3] <== [tmp_6808[0] + tmp_6813[0], tmp_6808[1] + tmp_6813[1], tmp_6808[2] + tmp_6813[2]];
    signal tmp_6815[3] <== CMul()(challengeQ, tmp_6814);
    signal tmp_6816[3] <== CMul()(evals[82], tmp_6810);
    signal tmp_6817[3] <== [evals[91][0] - evals[105][0], evals[91][1] - evals[105][1], evals[91][2] - evals[105][2]];
    signal tmp_6818[3] <== CMul()(tmp_6816, tmp_6817);
    signal tmp_6819[3] <== [tmp_6815[0] + tmp_6818[0], tmp_6815[1] + tmp_6818[1], tmp_6815[2] + tmp_6818[2]];
    signal tmp_6820[3] <== CMul()(challengeQ, tmp_6819);
    signal tmp_6821[3] <== CMul()(evals[82], tmp_6810);
    signal tmp_6822[3] <== [evals[92][0] - evals[106][0], evals[92][1] - evals[106][1], evals[92][2] - evals[106][2]];
    signal tmp_6823[3] <== CMul()(tmp_6821, tmp_6822);
    signal tmp_6824[3] <== [tmp_6820[0] + tmp_6823[0], tmp_6820[1] + tmp_6823[1], tmp_6820[2] + tmp_6823[2]];
    signal tmp_6825[3] <== CMul()(challengeQ, tmp_6824);
    signal tmp_6826[3] <== CMul()(evals[82], tmp_6810);
    signal tmp_6827[3] <== [evals[93][0] - evals[107][0], evals[93][1] - evals[107][1], evals[93][2] - evals[107][2]];
    signal tmp_6828[3] <== CMul()(tmp_6826, tmp_6827);
    signal tmp_6829[3] <== [tmp_6825[0] + tmp_6828[0], tmp_6825[1] + tmp_6828[1], tmp_6825[2] + tmp_6828[2]];
    signal tmp_6830[3] <== CMul()(challengeQ, tmp_6829);
    signal tmp_6831[3] <== [1 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_6832[3] <== CMul()(tmp_6831, evals[103]);
    signal tmp_6833[3] <== CMul()(evals[82], tmp_6832);
    signal tmp_6834[3] <== [evals[94][0] - evals[104][0], evals[94][1] - evals[104][1], evals[94][2] - evals[104][2]];
    signal tmp_6835[3] <== CMul()(tmp_6833, tmp_6834);
    signal tmp_6836[3] <== [tmp_6830[0] + tmp_6835[0], tmp_6830[1] + tmp_6835[1], tmp_6830[2] + tmp_6835[2]];
    signal tmp_6837[3] <== CMul()(challengeQ, tmp_6836);
    signal tmp_6838[3] <== CMul()(evals[82], tmp_6832);
    signal tmp_6839[3] <== [evals[95][0] - evals[105][0], evals[95][1] - evals[105][1], evals[95][2] - evals[105][2]];
    signal tmp_6840[3] <== CMul()(tmp_6838, tmp_6839);
    signal tmp_6841[3] <== [tmp_6837[0] + tmp_6840[0], tmp_6837[1] + tmp_6840[1], tmp_6837[2] + tmp_6840[2]];
    signal tmp_6842[3] <== CMul()(challengeQ, tmp_6841);
    signal tmp_6843[3] <== CMul()(evals[82], tmp_6832);
    signal tmp_6844[3] <== [evals[96][0] - evals[106][0], evals[96][1] - evals[106][1], evals[96][2] - evals[106][2]];
    signal tmp_6845[3] <== CMul()(tmp_6843, tmp_6844);
    signal tmp_6846[3] <== [tmp_6842[0] + tmp_6845[0], tmp_6842[1] + tmp_6845[1], tmp_6842[2] + tmp_6845[2]];
    signal tmp_6847[3] <== CMul()(challengeQ, tmp_6846);
    signal tmp_6848[3] <== CMul()(evals[82], tmp_6832);
    signal tmp_6849[3] <== [evals[97][0] - evals[107][0], evals[97][1] - evals[107][1], evals[97][2] - evals[107][2]];
    signal tmp_6850[3] <== CMul()(tmp_6848, tmp_6849);
    signal tmp_6851[3] <== [tmp_6847[0] + tmp_6850[0], tmp_6847[1] + tmp_6850[1], tmp_6847[2] + tmp_6850[2]];
    signal tmp_6852[3] <== CMul()(challengeQ, tmp_6851);
    tmp_6853 <== CMul()(evals[102], evals[103]);
    signal tmp_6854[3] <== CMul()(evals[82], tmp_6853);
    signal tmp_6855[3] <== [evals[98][0] - evals[104][0], evals[98][1] - evals[104][1], evals[98][2] - evals[104][2]];
    signal tmp_6856[3] <== CMul()(tmp_6854, tmp_6855);
    signal tmp_6857[3] <== [tmp_6852[0] + tmp_6856[0], tmp_6852[1] + tmp_6856[1], tmp_6852[2] + tmp_6856[2]];
    tmp_6858 <== CMul()(challengeQ, tmp_6857);
    tmp_6859 <== CMul()(evals[82], tmp_6853);
    tmp_6860 <== [evals[99][0] - evals[105][0], evals[99][1] - evals[105][1], evals[99][2] - evals[105][2]];
}

template VerifyEvaluationsChunks3() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[198][3];
    signal input publics[391];

    signal input Zh[3];

    signal input tmp_4451[3];
 
    signal input tmp_4475[3];
 
    signal input tmp_4498[3];
 
    signal input tmp_4527[3];
 
    signal input tmp_4556[3];
 
    signal input tmp_4580[3];
 
    signal input tmp_4603[3];
 
    signal input tmp_4632[3];
 
    signal input tmp_4662[3];
 
    signal input tmp_4686[3];
 
    signal input tmp_4709[3];
 
    signal input tmp_4738[3];
 
    signal input tmp_4768[3];
 
    signal input tmp_4792[3];
 
    signal input tmp_4815[3];
 
    signal input tmp_4844[3];
 
    signal input tmp_5945[3];
 
    signal input tmp_6004[3];
 
    signal input tmp_6063[3];
 
    signal input tmp_6122[3];
 
    signal input tmp_6181[3];
 
    signal input tmp_6240[3];
 
    signal input tmp_6853[3];
 
    signal input tmp_6858[3];
 
    signal input tmp_6859[3];
 
    signal input tmp_6860[3];
 

    signal output tmp_7700[3];
 
    signal tmp_6861[3] <== CMul()(tmp_6859, tmp_6860);
    signal tmp_6862[3] <== [tmp_6858[0] + tmp_6861[0], tmp_6858[1] + tmp_6861[1], tmp_6858[2] + tmp_6861[2]];
    signal tmp_6863[3] <== CMul()(challengeQ, tmp_6862);
    signal tmp_6864[3] <== CMul()(evals[82], tmp_6853);
    signal tmp_6865[3] <== [evals[100][0] - evals[106][0], evals[100][1] - evals[106][1], evals[100][2] - evals[106][2]];
    signal tmp_6866[3] <== CMul()(tmp_6864, tmp_6865);
    signal tmp_6867[3] <== [tmp_6863[0] + tmp_6866[0], tmp_6863[1] + tmp_6866[1], tmp_6863[2] + tmp_6866[2]];
    signal tmp_6868[3] <== CMul()(challengeQ, tmp_6867);
    signal tmp_6869[3] <== CMul()(evals[82], tmp_6853);
    signal tmp_6870[3] <== [evals[101][0] - evals[107][0], evals[101][1] - evals[107][1], evals[101][2] - evals[107][2]];
    signal tmp_6871[3] <== CMul()(tmp_6869, tmp_6870);
    signal tmp_6872[3] <== [tmp_6868[0] + tmp_6871[0], tmp_6868[1] + tmp_6871[1], tmp_6868[2] + tmp_6871[2]];
    signal tmp_6873[3] <== CMul()(challengeQ, tmp_6872);
    signal tmp_6874[3] <== [1 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_6875[3] <== CMul()(evals[102], tmp_6874);
    signal tmp_6876[3] <== CMul()(evals[82], tmp_6875);
    signal tmp_6877[3] <== [tmp_6873[0] + tmp_6876[0], tmp_6873[1] + tmp_6876[1], tmp_6873[2] + tmp_6876[2]];
    signal tmp_6878[3] <== CMul()(challengeQ, tmp_6877);
    signal tmp_6879[3] <== [1 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_6880[3] <== CMul()(evals[103], tmp_6879);
    signal tmp_6881[3] <== CMul()(evals[82], tmp_6880);
    signal tmp_6882[3] <== [tmp_6878[0] + tmp_6881[0], tmp_6878[1] + tmp_6881[1], tmp_6878[2] + tmp_6881[2]];
    signal tmp_6883[3] <== CMul()(challengeQ, tmp_6882);
    signal tmp_6884[3] <== CMul()(evals[84], challengesStage2[0]);
    signal tmp_6885[3] <== [tmp_6884[0] + evals[86][0], tmp_6884[1] + evals[86][1], tmp_6884[2] + evals[86][2]];
    signal tmp_6886[3] <== CMul()(tmp_6885, challengesStage2[0]);
    signal tmp_6887[3] <== [tmp_6886[0] + 1, tmp_6886[1], tmp_6886[2]];
    signal tmp_6888[3] <== [tmp_6887[0] + challengesStage2[1][0], tmp_6887[1] + challengesStage2[1][1], tmp_6887[2] + challengesStage2[1][2]];
    signal tmp_6889[3] <== [tmp_6888[0] - 1, tmp_6888[1], tmp_6888[2]];
    signal tmp_6890[3] <== [tmp_6889[0] + 1, tmp_6889[1], tmp_6889[2]];
    signal tmp_6891[3] <== [12275445934081160404 * evals[84][0], 12275445934081160404 * evals[84][1], 12275445934081160404 * evals[84][2]];
    signal tmp_6892[3] <== CMul()(tmp_6891, challengesStage2[0]);
    signal tmp_6893[3] <== [tmp_6892[0] + evals[87][0], tmp_6892[1] + evals[87][1], tmp_6892[2] + evals[87][2]];
    signal tmp_6894[3] <== CMul()(tmp_6893, challengesStage2[0]);
    signal tmp_6895[3] <== [tmp_6894[0] + 1, tmp_6894[1], tmp_6894[2]];
    signal tmp_6896[3] <== [tmp_6895[0] + challengesStage2[1][0], tmp_6895[1] + challengesStage2[1][1], tmp_6895[2] + challengesStage2[1][2]];
    signal tmp_6897[3] <== [tmp_6896[0] - 1, tmp_6896[1], tmp_6896[2]];
    signal tmp_6898[3] <== [tmp_6897[0] + 1, tmp_6897[1], tmp_6897[2]];
    signal tmp_6899[3] <== CMul()(tmp_6890, tmp_6898);
    signal tmp_6900[3] <== [4756475762779100925 * evals[84][0], 4756475762779100925 * evals[84][1], 4756475762779100925 * evals[84][2]];
    signal tmp_6901[3] <== CMul()(tmp_6900, challengesStage2[0]);
    signal tmp_6902[3] <== [tmp_6901[0] + evals[88][0], tmp_6901[1] + evals[88][1], tmp_6901[2] + evals[88][2]];
    signal tmp_6903[3] <== CMul()(tmp_6902, challengesStage2[0]);
    signal tmp_6904[3] <== [tmp_6903[0] + 1, tmp_6903[1], tmp_6903[2]];
    signal tmp_6905[3] <== [tmp_6904[0] + challengesStage2[1][0], tmp_6904[1] + challengesStage2[1][1], tmp_6904[2] + challengesStage2[1][2]];
    signal tmp_6906[3] <== [tmp_6905[0] - 1, tmp_6905[1], tmp_6905[2]];
    signal tmp_6907[3] <== [tmp_6906[0] + 1, tmp_6906[1], tmp_6906[2]];
    signal tmp_6908[3] <== CMul()(tmp_6899, tmp_6907);
    signal tmp_6909[3] <== [1279992132519201448 * evals[84][0], 1279992132519201448 * evals[84][1], 1279992132519201448 * evals[84][2]];
    signal tmp_6910[3] <== CMul()(tmp_6909, challengesStage2[0]);
    signal tmp_6911[3] <== [tmp_6910[0] + evals[89][0], tmp_6910[1] + evals[89][1], tmp_6910[2] + evals[89][2]];
    signal tmp_6912[3] <== CMul()(tmp_6911, challengesStage2[0]);
    signal tmp_6913[3] <== [tmp_6912[0] + 1, tmp_6912[1], tmp_6912[2]];
    signal tmp_6914[3] <== [tmp_6913[0] + challengesStage2[1][0], tmp_6913[1] + challengesStage2[1][1], tmp_6913[2] + challengesStage2[1][2]];
    signal tmp_6915[3] <== [tmp_6914[0] - 1, tmp_6914[1], tmp_6914[2]];
    signal tmp_6916[3] <== [tmp_6915[0] + 1, tmp_6915[1], tmp_6915[2]];
    signal tmp_6917[3] <== CMul()(tmp_6908, tmp_6916);
    signal tmp_6918[3] <== CMul()(evals[139], tmp_6917);
    signal tmp_6919[3] <== CMul()(evals[28], challengesStage2[0]);
    signal tmp_6920[3] <== [tmp_6919[0] + evals[86][0], tmp_6919[1] + evals[86][1], tmp_6919[2] + evals[86][2]];
    signal tmp_6921[3] <== CMul()(tmp_6920, challengesStage2[0]);
    signal tmp_6922[3] <== [tmp_6921[0] + 1, tmp_6921[1], tmp_6921[2]];
    signal tmp_6923[3] <== [tmp_6922[0] + challengesStage2[1][0], tmp_6922[1] + challengesStage2[1][1], tmp_6922[2] + challengesStage2[1][2]];
    signal tmp_6924[3] <== [tmp_6923[0] - 1, tmp_6923[1], tmp_6923[2]];
    signal tmp_6925[3] <== [tmp_6924[0] + 1, tmp_6924[1], tmp_6924[2]];
    signal tmp_6926[3] <== CMul()(evals[29], challengesStage2[0]);
    signal tmp_6927[3] <== [tmp_6926[0] + evals[87][0], tmp_6926[1] + evals[87][1], tmp_6926[2] + evals[87][2]];
    signal tmp_6928[3] <== CMul()(tmp_6927, challengesStage2[0]);
    signal tmp_6929[3] <== [tmp_6928[0] + 1, tmp_6928[1], tmp_6928[2]];
    signal tmp_6930[3] <== [tmp_6929[0] + challengesStage2[1][0], tmp_6929[1] + challengesStage2[1][1], tmp_6929[2] + challengesStage2[1][2]];
    signal tmp_6931[3] <== [tmp_6930[0] - 1, tmp_6930[1], tmp_6930[2]];
    signal tmp_6932[3] <== [tmp_6931[0] + 1, tmp_6931[1], tmp_6931[2]];
    signal tmp_6933[3] <== CMul()(tmp_6925, tmp_6932);
    signal tmp_6934[3] <== CMul()(evals[30], challengesStage2[0]);
    signal tmp_6935[3] <== [tmp_6934[0] + evals[88][0], tmp_6934[1] + evals[88][1], tmp_6934[2] + evals[88][2]];
    signal tmp_6936[3] <== CMul()(tmp_6935, challengesStage2[0]);
    signal tmp_6937[3] <== [tmp_6936[0] + 1, tmp_6936[1], tmp_6936[2]];
    signal tmp_6938[3] <== [tmp_6937[0] + challengesStage2[1][0], tmp_6937[1] + challengesStage2[1][1], tmp_6937[2] + challengesStage2[1][2]];
    signal tmp_6939[3] <== [tmp_6938[0] - 1, tmp_6938[1], tmp_6938[2]];
    signal tmp_6940[3] <== [tmp_6939[0] + 1, tmp_6939[1], tmp_6939[2]];
    signal tmp_6941[3] <== CMul()(tmp_6933, tmp_6940);
    signal tmp_6942[3] <== CMul()(evals[31], challengesStage2[0]);
    signal tmp_6943[3] <== [tmp_6942[0] + evals[89][0], tmp_6942[1] + evals[89][1], tmp_6942[2] + evals[89][2]];
    signal tmp_6944[3] <== CMul()(tmp_6943, challengesStage2[0]);
    signal tmp_6945[3] <== [tmp_6944[0] + 1, tmp_6944[1], tmp_6944[2]];
    signal tmp_6946[3] <== [tmp_6945[0] + challengesStage2[1][0], tmp_6945[1] + challengesStage2[1][1], tmp_6945[2] + challengesStage2[1][2]];
    signal tmp_6947[3] <== [tmp_6946[0] - 1, tmp_6946[1], tmp_6946[2]];
    signal tmp_6948[3] <== [tmp_6947[0] + 1, tmp_6947[1], tmp_6947[2]];
    signal tmp_6949[3] <== CMul()(tmp_6941, tmp_6948);
    signal tmp_6950[3] <== CMul()(evals[32], challengesStage2[0]);
    signal tmp_6951[3] <== [tmp_6950[0] + evals[90][0], tmp_6950[1] + evals[90][1], tmp_6950[2] + evals[90][2]];
    signal tmp_6952[3] <== CMul()(tmp_6951, challengesStage2[0]);
    signal tmp_6953[3] <== [tmp_6952[0] + 1, tmp_6952[1], tmp_6952[2]];
    signal tmp_6954[3] <== [tmp_6953[0] + challengesStage2[1][0], tmp_6953[1] + challengesStage2[1][1], tmp_6953[2] + challengesStage2[1][2]];
    signal tmp_6955[3] <== [tmp_6954[0] - 1, tmp_6954[1], tmp_6954[2]];
    signal tmp_6956[3] <== [tmp_6955[0] + 1, tmp_6955[1], tmp_6955[2]];
    signal tmp_6957[3] <== CMul()(tmp_6949, tmp_6956);
    signal tmp_6958[3] <== [tmp_6918[0] - tmp_6957[0], tmp_6918[1] - tmp_6957[1], tmp_6918[2] - tmp_6957[2]];
    signal tmp_6959[3] <== [tmp_6883[0] + tmp_6958[0], tmp_6883[1] + tmp_6958[1], tmp_6883[2] + tmp_6958[2]];
    signal tmp_6960[3] <== CMul()(challengeQ, tmp_6959);
    signal tmp_6961[3] <== [8312008622371998338 * evals[84][0], 8312008622371998338 * evals[84][1], 8312008622371998338 * evals[84][2]];
    signal tmp_6962[3] <== CMul()(tmp_6961, challengesStage2[0]);
    signal tmp_6963[3] <== [tmp_6962[0] + evals[90][0], tmp_6962[1] + evals[90][1], tmp_6962[2] + evals[90][2]];
    signal tmp_6964[3] <== CMul()(tmp_6963, challengesStage2[0]);
    signal tmp_6965[3] <== [tmp_6964[0] + 1, tmp_6964[1], tmp_6964[2]];
    signal tmp_6966[3] <== [tmp_6965[0] + challengesStage2[1][0], tmp_6965[1] + challengesStage2[1][1], tmp_6965[2] + challengesStage2[1][2]];
    signal tmp_6967[3] <== [tmp_6966[0] - 1, tmp_6966[1], tmp_6966[2]];
    signal tmp_6968[3] <== [tmp_6967[0] + 1, tmp_6967[1], tmp_6967[2]];
    signal tmp_6969[3] <== [7781028390488215464 * evals[84][0], 7781028390488215464 * evals[84][1], 7781028390488215464 * evals[84][2]];
    signal tmp_6970[3] <== CMul()(tmp_6969, challengesStage2[0]);
    signal tmp_6971[3] <== [tmp_6970[0] + evals[91][0], tmp_6970[1] + evals[91][1], tmp_6970[2] + evals[91][2]];
    signal tmp_6972[3] <== CMul()(tmp_6971, challengesStage2[0]);
    signal tmp_6973[3] <== [tmp_6972[0] + 1, tmp_6972[1], tmp_6972[2]];
    signal tmp_6974[3] <== [tmp_6973[0] + challengesStage2[1][0], tmp_6973[1] + challengesStage2[1][1], tmp_6973[2] + challengesStage2[1][2]];
    signal tmp_6975[3] <== [tmp_6974[0] - 1, tmp_6974[1], tmp_6974[2]];
    signal tmp_6976[3] <== [tmp_6975[0] + 1, tmp_6975[1], tmp_6975[2]];
    signal tmp_6977[3] <== CMul()(tmp_6968, tmp_6976);
    signal tmp_6978[3] <== [11302600489504509467 * evals[84][0], 11302600489504509467 * evals[84][1], 11302600489504509467 * evals[84][2]];
    signal tmp_6979[3] <== CMul()(tmp_6978, challengesStage2[0]);
    signal tmp_6980[3] <== [tmp_6979[0] + evals[92][0], tmp_6979[1] + evals[92][1], tmp_6979[2] + evals[92][2]];
    signal tmp_6981[3] <== CMul()(tmp_6980, challengesStage2[0]);
    signal tmp_6982[3] <== [tmp_6981[0] + 1, tmp_6981[1], tmp_6981[2]];
    signal tmp_6983[3] <== [tmp_6982[0] + challengesStage2[1][0], tmp_6982[1] + challengesStage2[1][1], tmp_6982[2] + challengesStage2[1][2]];
    signal tmp_6984[3] <== [tmp_6983[0] - 1, tmp_6983[1], tmp_6983[2]];
    signal tmp_6985[3] <== [tmp_6984[0] + 1, tmp_6984[1], tmp_6984[2]];
    signal tmp_6986[3] <== CMul()(tmp_6977, tmp_6985);
    signal tmp_6987[3] <== [4549350404001778198 * evals[84][0], 4549350404001778198 * evals[84][1], 4549350404001778198 * evals[84][2]];
    signal tmp_6988[3] <== CMul()(tmp_6987, challengesStage2[0]);
    signal tmp_6989[3] <== [tmp_6988[0] + evals[93][0], tmp_6988[1] + evals[93][1], tmp_6988[2] + evals[93][2]];
    signal tmp_6990[3] <== CMul()(tmp_6989, challengesStage2[0]);
    signal tmp_6991[3] <== [tmp_6990[0] + 1, tmp_6990[1], tmp_6990[2]];
    signal tmp_6992[3] <== [tmp_6991[0] + challengesStage2[1][0], tmp_6991[1] + challengesStage2[1][1], tmp_6991[2] + challengesStage2[1][2]];
    signal tmp_6993[3] <== [tmp_6992[0] - 1, tmp_6992[1], tmp_6992[2]];
    signal tmp_6994[3] <== [tmp_6993[0] + 1, tmp_6993[1], tmp_6993[2]];
    signal tmp_6995[3] <== CMul()(tmp_6986, tmp_6994);
    signal tmp_6996[3] <== CMul()(evals[140], tmp_6995);
    signal tmp_6997[3] <== CMul()(evals[33], challengesStage2[0]);
    signal tmp_6998[3] <== [tmp_6997[0] + evals[91][0], tmp_6997[1] + evals[91][1], tmp_6997[2] + evals[91][2]];
    signal tmp_6999[3] <== CMul()(tmp_6998, challengesStage2[0]);
    signal tmp_7000[3] <== [tmp_6999[0] + 1, tmp_6999[1], tmp_6999[2]];
    signal tmp_7001[3] <== [tmp_7000[0] + challengesStage2[1][0], tmp_7000[1] + challengesStage2[1][1], tmp_7000[2] + challengesStage2[1][2]];
    signal tmp_7002[3] <== [tmp_7001[0] - 1, tmp_7001[1], tmp_7001[2]];
    signal tmp_7003[3] <== [tmp_7002[0] + 1, tmp_7002[1], tmp_7002[2]];
    signal tmp_7004[3] <== CMul()(evals[139], tmp_7003);
    signal tmp_7005[3] <== CMul()(evals[34], challengesStage2[0]);
    signal tmp_7006[3] <== [tmp_7005[0] + evals[92][0], tmp_7005[1] + evals[92][1], tmp_7005[2] + evals[92][2]];
    signal tmp_7007[3] <== CMul()(tmp_7006, challengesStage2[0]);
    signal tmp_7008[3] <== [tmp_7007[0] + 1, tmp_7007[1], tmp_7007[2]];
    signal tmp_7009[3] <== [tmp_7008[0] + challengesStage2[1][0], tmp_7008[1] + challengesStage2[1][1], tmp_7008[2] + challengesStage2[1][2]];
    signal tmp_7010[3] <== [tmp_7009[0] - 1, tmp_7009[1], tmp_7009[2]];
    signal tmp_7011[3] <== [tmp_7010[0] + 1, tmp_7010[1], tmp_7010[2]];
    signal tmp_7012[3] <== CMul()(tmp_7004, tmp_7011);
    signal tmp_7013[3] <== CMul()(evals[35], challengesStage2[0]);
    signal tmp_7014[3] <== [tmp_7013[0] + evals[93][0], tmp_7013[1] + evals[93][1], tmp_7013[2] + evals[93][2]];
    signal tmp_7015[3] <== CMul()(tmp_7014, challengesStage2[0]);
    signal tmp_7016[3] <== [tmp_7015[0] + 1, tmp_7015[1], tmp_7015[2]];
    signal tmp_7017[3] <== [tmp_7016[0] + challengesStage2[1][0], tmp_7016[1] + challengesStage2[1][1], tmp_7016[2] + challengesStage2[1][2]];
    signal tmp_7018[3] <== [tmp_7017[0] - 1, tmp_7017[1], tmp_7017[2]];
    signal tmp_7019[3] <== [tmp_7018[0] + 1, tmp_7018[1], tmp_7018[2]];
    signal tmp_7020[3] <== CMul()(tmp_7012, tmp_7019);
    signal tmp_7021[3] <== CMul()(evals[36], challengesStage2[0]);
    signal tmp_7022[3] <== [tmp_7021[0] + evals[94][0], tmp_7021[1] + evals[94][1], tmp_7021[2] + evals[94][2]];
    signal tmp_7023[3] <== CMul()(tmp_7022, challengesStage2[0]);
    signal tmp_7024[3] <== [tmp_7023[0] + 1, tmp_7023[1], tmp_7023[2]];
    signal tmp_7025[3] <== [tmp_7024[0] + challengesStage2[1][0], tmp_7024[1] + challengesStage2[1][1], tmp_7024[2] + challengesStage2[1][2]];
    signal tmp_7026[3] <== [tmp_7025[0] - 1, tmp_7025[1], tmp_7025[2]];
    signal tmp_7027[3] <== [tmp_7026[0] + 1, tmp_7026[1], tmp_7026[2]];
    signal tmp_7028[3] <== CMul()(tmp_7020, tmp_7027);
    signal tmp_7029[3] <== [tmp_6996[0] - tmp_7028[0], tmp_6996[1] - tmp_7028[1], tmp_6996[2] - tmp_7028[2]];
    signal tmp_7030[3] <== [tmp_6960[0] + tmp_7029[0], tmp_6960[1] + tmp_7029[1], tmp_6960[2] + tmp_7029[2]];
    signal tmp_7031[3] <== CMul()(challengeQ, tmp_7030);
    signal tmp_7032[3] <== [3688660304411827445 * evals[84][0], 3688660304411827445 * evals[84][1], 3688660304411827445 * evals[84][2]];
    signal tmp_7033[3] <== CMul()(tmp_7032, challengesStage2[0]);
    signal tmp_7034[3] <== [tmp_7033[0] + evals[94][0], tmp_7033[1] + evals[94][1], tmp_7033[2] + evals[94][2]];
    signal tmp_7035[3] <== CMul()(tmp_7034, challengesStage2[0]);
    signal tmp_7036[3] <== [tmp_7035[0] + 1, tmp_7035[1], tmp_7035[2]];
    signal tmp_7037[3] <== [tmp_7036[0] + challengesStage2[1][0], tmp_7036[1] + challengesStage2[1][1], tmp_7036[2] + challengesStage2[1][2]];
    signal tmp_7038[3] <== [tmp_7037[0] - 1, tmp_7037[1], tmp_7037[2]];
    signal tmp_7039[3] <== [tmp_7038[0] + 1, tmp_7038[1], tmp_7038[2]];
    signal tmp_7040[3] <== [16725109960945739746 * evals[84][0], 16725109960945739746 * evals[84][1], 16725109960945739746 * evals[84][2]];
    signal tmp_7041[3] <== CMul()(tmp_7040, challengesStage2[0]);
    signal tmp_7042[3] <== [tmp_7041[0] + evals[95][0], tmp_7041[1] + evals[95][1], tmp_7041[2] + evals[95][2]];
    signal tmp_7043[3] <== CMul()(tmp_7042, challengesStage2[0]);
    signal tmp_7044[3] <== [tmp_7043[0] + 1, tmp_7043[1], tmp_7043[2]];
    signal tmp_7045[3] <== [tmp_7044[0] + challengesStage2[1][0], tmp_7044[1] + challengesStage2[1][1], tmp_7044[2] + challengesStage2[1][2]];
    signal tmp_7046[3] <== [tmp_7045[0] - 1, tmp_7045[1], tmp_7045[2]];
    signal tmp_7047[3] <== [tmp_7046[0] + 1, tmp_7046[1], tmp_7046[2]];
    signal tmp_7048[3] <== CMul()(tmp_7039, tmp_7047);
    signal tmp_7049[3] <== [16538725463549498621 * evals[84][0], 16538725463549498621 * evals[84][1], 16538725463549498621 * evals[84][2]];
    signal tmp_7050[3] <== CMul()(tmp_7049, challengesStage2[0]);
    signal tmp_7051[3] <== [tmp_7050[0] + evals[96][0], tmp_7050[1] + evals[96][1], tmp_7050[2] + evals[96][2]];
    signal tmp_7052[3] <== CMul()(tmp_7051, challengesStage2[0]);
    signal tmp_7053[3] <== [tmp_7052[0] + 1, tmp_7052[1], tmp_7052[2]];
    signal tmp_7054[3] <== [tmp_7053[0] + challengesStage2[1][0], tmp_7053[1] + challengesStage2[1][1], tmp_7053[2] + challengesStage2[1][2]];
    signal tmp_7055[3] <== [tmp_7054[0] - 1, tmp_7054[1], tmp_7054[2]];
    signal tmp_7056[3] <== [tmp_7055[0] + 1, tmp_7055[1], tmp_7055[2]];
    signal tmp_7057[3] <== CMul()(tmp_7048, tmp_7056);
    signal tmp_7058[3] <== [12756200801261202346 * evals[84][0], 12756200801261202346 * evals[84][1], 12756200801261202346 * evals[84][2]];
    signal tmp_7059[3] <== CMul()(tmp_7058, challengesStage2[0]);
    signal tmp_7060[3] <== [tmp_7059[0] + evals[97][0], tmp_7059[1] + evals[97][1], tmp_7059[2] + evals[97][2]];
    signal tmp_7061[3] <== CMul()(tmp_7060, challengesStage2[0]);
    signal tmp_7062[3] <== [tmp_7061[0] + 1, tmp_7061[1], tmp_7061[2]];
    signal tmp_7063[3] <== [tmp_7062[0] + challengesStage2[1][0], tmp_7062[1] + challengesStage2[1][1], tmp_7062[2] + challengesStage2[1][2]];
    signal tmp_7064[3] <== [tmp_7063[0] - 1, tmp_7063[1], tmp_7063[2]];
    signal tmp_7065[3] <== [tmp_7064[0] + 1, tmp_7064[1], tmp_7064[2]];
    signal tmp_7066[3] <== CMul()(tmp_7057, tmp_7065);
    signal tmp_7067[3] <== CMul()(evals[141], tmp_7066);
    signal tmp_7068[3] <== CMul()(evals[37], challengesStage2[0]);
    signal tmp_7069[3] <== [tmp_7068[0] + evals[95][0], tmp_7068[1] + evals[95][1], tmp_7068[2] + evals[95][2]];
    signal tmp_7070[3] <== CMul()(tmp_7069, challengesStage2[0]);
    signal tmp_7071[3] <== [tmp_7070[0] + 1, tmp_7070[1], tmp_7070[2]];
    signal tmp_7072[3] <== [tmp_7071[0] + challengesStage2[1][0], tmp_7071[1] + challengesStage2[1][1], tmp_7071[2] + challengesStage2[1][2]];
    signal tmp_7073[3] <== [tmp_7072[0] - 1, tmp_7072[1], tmp_7072[2]];
    signal tmp_7074[3] <== [tmp_7073[0] + 1, tmp_7073[1], tmp_7073[2]];
    signal tmp_7075[3] <== CMul()(evals[140], tmp_7074);
    signal tmp_7076[3] <== CMul()(evals[38], challengesStage2[0]);
    signal tmp_7077[3] <== [tmp_7076[0] + evals[96][0], tmp_7076[1] + evals[96][1], tmp_7076[2] + evals[96][2]];
    signal tmp_7078[3] <== CMul()(tmp_7077, challengesStage2[0]);
    signal tmp_7079[3] <== [tmp_7078[0] + 1, tmp_7078[1], tmp_7078[2]];
    signal tmp_7080[3] <== [tmp_7079[0] + challengesStage2[1][0], tmp_7079[1] + challengesStage2[1][1], tmp_7079[2] + challengesStage2[1][2]];
    signal tmp_7081[3] <== [tmp_7080[0] - 1, tmp_7080[1], tmp_7080[2]];
    signal tmp_7082[3] <== [tmp_7081[0] + 1, tmp_7081[1], tmp_7081[2]];
    signal tmp_7083[3] <== CMul()(tmp_7075, tmp_7082);
    signal tmp_7084[3] <== CMul()(evals[39], challengesStage2[0]);
    signal tmp_7085[3] <== [tmp_7084[0] + evals[97][0], tmp_7084[1] + evals[97][1], tmp_7084[2] + evals[97][2]];
    signal tmp_7086[3] <== CMul()(tmp_7085, challengesStage2[0]);
    signal tmp_7087[3] <== [tmp_7086[0] + 1, tmp_7086[1], tmp_7086[2]];
    signal tmp_7088[3] <== [tmp_7087[0] + challengesStage2[1][0], tmp_7087[1] + challengesStage2[1][1], tmp_7087[2] + challengesStage2[1][2]];
    signal tmp_7089[3] <== [tmp_7088[0] - 1, tmp_7088[1], tmp_7088[2]];
    signal tmp_7090[3] <== [tmp_7089[0] + 1, tmp_7089[1], tmp_7089[2]];
    signal tmp_7091[3] <== CMul()(tmp_7083, tmp_7090);
    signal tmp_7092[3] <== CMul()(evals[40], challengesStage2[0]);
    signal tmp_7093[3] <== [tmp_7092[0] + evals[98][0], tmp_7092[1] + evals[98][1], tmp_7092[2] + evals[98][2]];
    signal tmp_7094[3] <== CMul()(tmp_7093, challengesStage2[0]);
    signal tmp_7095[3] <== [tmp_7094[0] + 1, tmp_7094[1], tmp_7094[2]];
    signal tmp_7096[3] <== [tmp_7095[0] + challengesStage2[1][0], tmp_7095[1] + challengesStage2[1][1], tmp_7095[2] + challengesStage2[1][2]];
    signal tmp_7097[3] <== [tmp_7096[0] - 1, tmp_7096[1], tmp_7096[2]];
    signal tmp_7098[3] <== [tmp_7097[0] + 1, tmp_7097[1], tmp_7097[2]];
    signal tmp_7099[3] <== CMul()(tmp_7091, tmp_7098);
    signal tmp_7100[3] <== [tmp_7067[0] - tmp_7099[0], tmp_7067[1] - tmp_7099[1], tmp_7067[2] - tmp_7099[2]];
    signal tmp_7101[3] <== [tmp_7031[0] + tmp_7100[0], tmp_7031[1] + tmp_7100[1], tmp_7031[2] + tmp_7100[2]];
    signal tmp_7102[3] <== CMul()(challengeQ, tmp_7101);
    signal tmp_7103[3] <== [15099809066790865939 * evals[84][0], 15099809066790865939 * evals[84][1], 15099809066790865939 * evals[84][2]];
    signal tmp_7104[3] <== CMul()(tmp_7103, challengesStage2[0]);
    signal tmp_7105[3] <== [tmp_7104[0] + evals[98][0], tmp_7104[1] + evals[98][1], tmp_7104[2] + evals[98][2]];
    signal tmp_7106[3] <== CMul()(tmp_7105, challengesStage2[0]);
    signal tmp_7107[3] <== [tmp_7106[0] + 1, tmp_7106[1], tmp_7106[2]];
    signal tmp_7108[3] <== [tmp_7107[0] + challengesStage2[1][0], tmp_7107[1] + challengesStage2[1][1], tmp_7107[2] + challengesStage2[1][2]];
    signal tmp_7109[3] <== [tmp_7108[0] - 1, tmp_7108[1], tmp_7108[2]];
    signal tmp_7110[3] <== [tmp_7109[0] + 1, tmp_7109[1], tmp_7109[2]];
    signal tmp_7111[3] <== [17214954929431464349 * evals[84][0], 17214954929431464349 * evals[84][1], 17214954929431464349 * evals[84][2]];
    signal tmp_7112[3] <== CMul()(tmp_7111, challengesStage2[0]);
    signal tmp_7113[3] <== [tmp_7112[0] + evals[99][0], tmp_7112[1] + evals[99][1], tmp_7112[2] + evals[99][2]];
    signal tmp_7114[3] <== CMul()(tmp_7113, challengesStage2[0]);
    signal tmp_7115[3] <== [tmp_7114[0] + 1, tmp_7114[1], tmp_7114[2]];
    signal tmp_7116[3] <== [tmp_7115[0] + challengesStage2[1][0], tmp_7115[1] + challengesStage2[1][1], tmp_7115[2] + challengesStage2[1][2]];
    signal tmp_7117[3] <== [tmp_7116[0] - 1, tmp_7116[1], tmp_7116[2]];
    signal tmp_7118[3] <== [tmp_7117[0] + 1, tmp_7117[1], tmp_7117[2]];
    signal tmp_7119[3] <== CMul()(tmp_7110, tmp_7118);
    signal tmp_7120[3] <== [11016800570561344835 * evals[84][0], 11016800570561344835 * evals[84][1], 11016800570561344835 * evals[84][2]];
    signal tmp_7121[3] <== CMul()(tmp_7120, challengesStage2[0]);
    signal tmp_7122[3] <== [tmp_7121[0] + evals[100][0], tmp_7121[1] + evals[100][1], tmp_7121[2] + evals[100][2]];
    signal tmp_7123[3] <== CMul()(tmp_7122, challengesStage2[0]);
    signal tmp_7124[3] <== [tmp_7123[0] + 1, tmp_7123[1], tmp_7123[2]];
    signal tmp_7125[3] <== [tmp_7124[0] + challengesStage2[1][0], tmp_7124[1] + challengesStage2[1][1], tmp_7124[2] + challengesStage2[1][2]];
    signal tmp_7126[3] <== [tmp_7125[0] - 1, tmp_7125[1], tmp_7125[2]];
    signal tmp_7127[3] <== [tmp_7126[0] + 1, tmp_7126[1], tmp_7126[2]];
    signal tmp_7128[3] <== CMul()(tmp_7119, tmp_7127);
    signal tmp_7129[3] <== [11274872323250451096 * evals[84][0], 11274872323250451096 * evals[84][1], 11274872323250451096 * evals[84][2]];
    signal tmp_7130[3] <== CMul()(tmp_7129, challengesStage2[0]);
    signal tmp_7131[3] <== [tmp_7130[0] + evals[101][0], tmp_7130[1] + evals[101][1], tmp_7130[2] + evals[101][2]];
    signal tmp_7132[3] <== CMul()(tmp_7131, challengesStage2[0]);
    signal tmp_7133[3] <== [tmp_7132[0] + 1, tmp_7132[1], tmp_7132[2]];
    signal tmp_7134[3] <== [tmp_7133[0] + challengesStage2[1][0], tmp_7133[1] + challengesStage2[1][1], tmp_7133[2] + challengesStage2[1][2]];
    signal tmp_7135[3] <== [tmp_7134[0] - 1, tmp_7134[1], tmp_7134[2]];
    signal tmp_7136[3] <== [tmp_7135[0] + 1, tmp_7135[1], tmp_7135[2]];
    signal tmp_7137[3] <== CMul()(tmp_7128, tmp_7136);
    signal tmp_7138[3] <== CMul()(evals[142], tmp_7137);
    signal tmp_7139[3] <== CMul()(evals[41], challengesStage2[0]);
    signal tmp_7140[3] <== [tmp_7139[0] + evals[99][0], tmp_7139[1] + evals[99][1], tmp_7139[2] + evals[99][2]];
    signal tmp_7141[3] <== CMul()(tmp_7140, challengesStage2[0]);
    signal tmp_7142[3] <== [tmp_7141[0] + 1, tmp_7141[1], tmp_7141[2]];
    signal tmp_7143[3] <== [tmp_7142[0] + challengesStage2[1][0], tmp_7142[1] + challengesStage2[1][1], tmp_7142[2] + challengesStage2[1][2]];
    signal tmp_7144[3] <== [tmp_7143[0] - 1, tmp_7143[1], tmp_7143[2]];
    signal tmp_7145[3] <== [tmp_7144[0] + 1, tmp_7144[1], tmp_7144[2]];
    signal tmp_7146[3] <== CMul()(evals[141], tmp_7145);
    signal tmp_7147[3] <== CMul()(evals[42], challengesStage2[0]);
    signal tmp_7148[3] <== [tmp_7147[0] + evals[100][0], tmp_7147[1] + evals[100][1], tmp_7147[2] + evals[100][2]];
    signal tmp_7149[3] <== CMul()(tmp_7148, challengesStage2[0]);
    signal tmp_7150[3] <== [tmp_7149[0] + 1, tmp_7149[1], tmp_7149[2]];
    signal tmp_7151[3] <== [tmp_7150[0] + challengesStage2[1][0], tmp_7150[1] + challengesStage2[1][1], tmp_7150[2] + challengesStage2[1][2]];
    signal tmp_7152[3] <== [tmp_7151[0] - 1, tmp_7151[1], tmp_7151[2]];
    signal tmp_7153[3] <== [tmp_7152[0] + 1, tmp_7152[1], tmp_7152[2]];
    signal tmp_7154[3] <== CMul()(tmp_7146, tmp_7153);
    signal tmp_7155[3] <== CMul()(evals[43], challengesStage2[0]);
    signal tmp_7156[3] <== [tmp_7155[0] + evals[101][0], tmp_7155[1] + evals[101][1], tmp_7155[2] + evals[101][2]];
    signal tmp_7157[3] <== CMul()(tmp_7156, challengesStage2[0]);
    signal tmp_7158[3] <== [tmp_7157[0] + 1, tmp_7157[1], tmp_7157[2]];
    signal tmp_7159[3] <== [tmp_7158[0] + challengesStage2[1][0], tmp_7158[1] + challengesStage2[1][1], tmp_7158[2] + challengesStage2[1][2]];
    signal tmp_7160[3] <== [tmp_7159[0] - 1, tmp_7159[1], tmp_7159[2]];
    signal tmp_7161[3] <== [tmp_7160[0] + 1, tmp_7160[1], tmp_7160[2]];
    signal tmp_7162[3] <== CMul()(tmp_7154, tmp_7161);
    signal tmp_7163[3] <== CMul()(evals[44], challengesStage2[0]);
    signal tmp_7164[3] <== [tmp_7163[0] + evals[102][0], tmp_7163[1] + evals[102][1], tmp_7163[2] + evals[102][2]];
    signal tmp_7165[3] <== CMul()(tmp_7164, challengesStage2[0]);
    signal tmp_7166[3] <== [tmp_7165[0] + 1, tmp_7165[1], tmp_7165[2]];
    signal tmp_7167[3] <== [tmp_7166[0] + challengesStage2[1][0], tmp_7166[1] + challengesStage2[1][1], tmp_7166[2] + challengesStage2[1][2]];
    signal tmp_7168[3] <== [tmp_7167[0] - 1, tmp_7167[1], tmp_7167[2]];
    signal tmp_7169[3] <== [tmp_7168[0] + 1, tmp_7168[1], tmp_7168[2]];
    signal tmp_7170[3] <== CMul()(tmp_7162, tmp_7169);
    signal tmp_7171[3] <== [tmp_7138[0] - tmp_7170[0], tmp_7138[1] - tmp_7170[1], tmp_7138[2] - tmp_7170[2]];
    signal tmp_7172[3] <== [tmp_7102[0] + tmp_7171[0], tmp_7102[1] + tmp_7171[1], tmp_7102[2] + tmp_7171[2]];
    signal tmp_7173[3] <== CMul()(challengeQ, tmp_7172);
    signal tmp_7174[3] <== [6534114114080170934 * evals[84][0], 6534114114080170934 * evals[84][1], 6534114114080170934 * evals[84][2]];
    signal tmp_7175[3] <== CMul()(tmp_7174, challengesStage2[0]);
    signal tmp_7176[3] <== [tmp_7175[0] + evals[102][0], tmp_7175[1] + evals[102][1], tmp_7175[2] + evals[102][2]];
    signal tmp_7177[3] <== CMul()(tmp_7176, challengesStage2[0]);
    signal tmp_7178[3] <== [tmp_7177[0] + 1, tmp_7177[1], tmp_7177[2]];
    signal tmp_7179[3] <== [tmp_7178[0] + challengesStage2[1][0], tmp_7178[1] + challengesStage2[1][1], tmp_7178[2] + challengesStage2[1][2]];
    signal tmp_7180[3] <== [tmp_7179[0] - 1, tmp_7179[1], tmp_7179[2]];
    signal tmp_7181[3] <== [tmp_7180[0] + 1, tmp_7180[1], tmp_7180[2]];
    signal tmp_7182[3] <== [13047390008333835222 * evals[84][0], 13047390008333835222 * evals[84][1], 13047390008333835222 * evals[84][2]];
    signal tmp_7183[3] <== CMul()(tmp_7182, challengesStage2[0]);
    signal tmp_7184[3] <== [tmp_7183[0] + evals[103][0], tmp_7183[1] + evals[103][1], tmp_7183[2] + evals[103][2]];
    signal tmp_7185[3] <== CMul()(tmp_7184, challengesStage2[0]);
    signal tmp_7186[3] <== [tmp_7185[0] + 1, tmp_7185[1], tmp_7185[2]];
    signal tmp_7187[3] <== [tmp_7186[0] + challengesStage2[1][0], tmp_7186[1] + challengesStage2[1][1], tmp_7186[2] + challengesStage2[1][2]];
    signal tmp_7188[3] <== [tmp_7187[0] - 1, tmp_7187[1], tmp_7187[2]];
    signal tmp_7189[3] <== [tmp_7188[0] + 1, tmp_7188[1], tmp_7188[2]];
    signal tmp_7190[3] <== CMul()(tmp_7181, tmp_7189);
    signal tmp_7191[3] <== [11189528522318044176 * evals[84][0], 11189528522318044176 * evals[84][1], 11189528522318044176 * evals[84][2]];
    signal tmp_7192[3] <== CMul()(tmp_7191, challengesStage2[0]);
    signal tmp_7193[3] <== [tmp_7192[0] + evals[104][0], tmp_7192[1] + evals[104][1], tmp_7192[2] + evals[104][2]];
    signal tmp_7194[3] <== CMul()(tmp_7193, challengesStage2[0]);
    signal tmp_7195[3] <== [tmp_7194[0] + 1, tmp_7194[1], tmp_7194[2]];
    signal tmp_7196[3] <== [tmp_7195[0] + challengesStage2[1][0], tmp_7195[1] + challengesStage2[1][1], tmp_7195[2] + challengesStage2[1][2]];
    signal tmp_7197[3] <== [tmp_7196[0] - 1, tmp_7196[1], tmp_7196[2]];
    signal tmp_7198[3] <== [tmp_7197[0] + 1, tmp_7197[1], tmp_7197[2]];
    signal tmp_7199[3] <== CMul()(tmp_7190, tmp_7198);
    signal tmp_7200[3] <== [3320735505586735876 * evals[84][0], 3320735505586735876 * evals[84][1], 3320735505586735876 * evals[84][2]];
    signal tmp_7201[3] <== CMul()(tmp_7200, challengesStage2[0]);
    signal tmp_7202[3] <== [tmp_7201[0] + evals[105][0], tmp_7201[1] + evals[105][1], tmp_7201[2] + evals[105][2]];
    signal tmp_7203[3] <== CMul()(tmp_7202, challengesStage2[0]);
    signal tmp_7204[3] <== [tmp_7203[0] + 1, tmp_7203[1], tmp_7203[2]];
    signal tmp_7205[3] <== [tmp_7204[0] + challengesStage2[1][0], tmp_7204[1] + challengesStage2[1][1], tmp_7204[2] + challengesStage2[1][2]];
    signal tmp_7206[3] <== [tmp_7205[0] - 1, tmp_7205[1], tmp_7205[2]];
    signal tmp_7207[3] <== [tmp_7206[0] + 1, tmp_7206[1], tmp_7206[2]];
    signal tmp_7208[3] <== CMul()(tmp_7199, tmp_7207);
    signal tmp_7209[3] <== CMul()(evals[143], tmp_7208);
    signal tmp_7210[3] <== CMul()(evals[45], challengesStage2[0]);
    signal tmp_7211[3] <== [tmp_7210[0] + evals[103][0], tmp_7210[1] + evals[103][1], tmp_7210[2] + evals[103][2]];
    signal tmp_7212[3] <== CMul()(tmp_7211, challengesStage2[0]);
    signal tmp_7213[3] <== [tmp_7212[0] + 1, tmp_7212[1], tmp_7212[2]];
    signal tmp_7214[3] <== [tmp_7213[0] + challengesStage2[1][0], tmp_7213[1] + challengesStage2[1][1], tmp_7213[2] + challengesStage2[1][2]];
    signal tmp_7215[3] <== [tmp_7214[0] - 1, tmp_7214[1], tmp_7214[2]];
    signal tmp_7216[3] <== [tmp_7215[0] + 1, tmp_7215[1], tmp_7215[2]];
    signal tmp_7217[3] <== CMul()(evals[142], tmp_7216);
    signal tmp_7218[3] <== CMul()(evals[46], challengesStage2[0]);
    signal tmp_7219[3] <== [tmp_7218[0] + evals[104][0], tmp_7218[1] + evals[104][1], tmp_7218[2] + evals[104][2]];
    signal tmp_7220[3] <== CMul()(tmp_7219, challengesStage2[0]);
    signal tmp_7221[3] <== [tmp_7220[0] + 1, tmp_7220[1], tmp_7220[2]];
    signal tmp_7222[3] <== [tmp_7221[0] + challengesStage2[1][0], tmp_7221[1] + challengesStage2[1][1], tmp_7221[2] + challengesStage2[1][2]];
    signal tmp_7223[3] <== [tmp_7222[0] - 1, tmp_7222[1], tmp_7222[2]];
    signal tmp_7224[3] <== [tmp_7223[0] + 1, tmp_7223[1], tmp_7223[2]];
    signal tmp_7225[3] <== CMul()(tmp_7217, tmp_7224);
    signal tmp_7226[3] <== CMul()(evals[47], challengesStage2[0]);
    signal tmp_7227[3] <== [tmp_7226[0] + evals[105][0], tmp_7226[1] + evals[105][1], tmp_7226[2] + evals[105][2]];
    signal tmp_7228[3] <== CMul()(tmp_7227, challengesStage2[0]);
    signal tmp_7229[3] <== [tmp_7228[0] + 1, tmp_7228[1], tmp_7228[2]];
    signal tmp_7230[3] <== [tmp_7229[0] + challengesStage2[1][0], tmp_7229[1] + challengesStage2[1][1], tmp_7229[2] + challengesStage2[1][2]];
    signal tmp_7231[3] <== [tmp_7230[0] - 1, tmp_7230[1], tmp_7230[2]];
    signal tmp_7232[3] <== [tmp_7231[0] + 1, tmp_7231[1], tmp_7231[2]];
    signal tmp_7233[3] <== CMul()(tmp_7225, tmp_7232);
    signal tmp_7234[3] <== CMul()(evals[48], challengesStage2[0]);
    signal tmp_7235[3] <== [tmp_7234[0] + evals[106][0], tmp_7234[1] + evals[106][1], tmp_7234[2] + evals[106][2]];
    signal tmp_7236[3] <== CMul()(tmp_7235, challengesStage2[0]);
    signal tmp_7237[3] <== [tmp_7236[0] + 1, tmp_7236[1], tmp_7236[2]];
    signal tmp_7238[3] <== [tmp_7237[0] + challengesStage2[1][0], tmp_7237[1] + challengesStage2[1][1], tmp_7237[2] + challengesStage2[1][2]];
    signal tmp_7239[3] <== [tmp_7238[0] - 1, tmp_7238[1], tmp_7238[2]];
    signal tmp_7240[3] <== [tmp_7239[0] + 1, tmp_7239[1], tmp_7239[2]];
    signal tmp_7241[3] <== CMul()(tmp_7233, tmp_7240);
    signal tmp_7242[3] <== [tmp_7209[0] - tmp_7241[0], tmp_7209[1] - tmp_7241[1], tmp_7209[2] - tmp_7241[2]];
    signal tmp_7243[3] <== [tmp_7173[0] + tmp_7242[0], tmp_7173[1] + tmp_7242[1], tmp_7173[2] + tmp_7242[2]];
    signal tmp_7244[3] <== CMul()(challengeQ, tmp_7243);
    signal tmp_7245[3] <== [7240278926970958133 * evals[84][0], 7240278926970958133 * evals[84][1], 7240278926970958133 * evals[84][2]];
    signal tmp_7246[3] <== CMul()(tmp_7245, challengesStage2[0]);
    signal tmp_7247[3] <== [tmp_7246[0] + evals[106][0], tmp_7246[1] + evals[106][1], tmp_7246[2] + evals[106][2]];
    signal tmp_7248[3] <== CMul()(tmp_7247, challengesStage2[0]);
    signal tmp_7249[3] <== [tmp_7248[0] + 1, tmp_7248[1], tmp_7248[2]];
    signal tmp_7250[3] <== [tmp_7249[0] + challengesStage2[1][0], tmp_7249[1] + challengesStage2[1][1], tmp_7249[2] + challengesStage2[1][2]];
    signal tmp_7251[3] <== [tmp_7250[0] - 1, tmp_7250[1], tmp_7250[2]];
    signal tmp_7252[3] <== [tmp_7251[0] + 1, tmp_7251[1], tmp_7251[2]];
    signal tmp_7253[3] <== [8246665031048405574 * evals[84][0], 8246665031048405574 * evals[84][1], 8246665031048405574 * evals[84][2]];
    signal tmp_7254[3] <== CMul()(tmp_7253, challengesStage2[0]);
    signal tmp_7255[3] <== [tmp_7254[0] + evals[107][0], tmp_7254[1] + evals[107][1], tmp_7254[2] + evals[107][2]];
    signal tmp_7256[3] <== CMul()(tmp_7255, challengesStage2[0]);
    signal tmp_7257[3] <== [tmp_7256[0] + 1, tmp_7256[1], tmp_7256[2]];
    signal tmp_7258[3] <== [tmp_7257[0] + challengesStage2[1][0], tmp_7257[1] + challengesStage2[1][1], tmp_7257[2] + challengesStage2[1][2]];
    signal tmp_7259[3] <== [tmp_7258[0] - 1, tmp_7258[1], tmp_7258[2]];
    signal tmp_7260[3] <== [tmp_7259[0] + 1, tmp_7259[1], tmp_7259[2]];
    signal tmp_7261[3] <== CMul()(tmp_7252, tmp_7260);
    signal tmp_7262[3] <== [12693612801792047873 * evals[84][0], 12693612801792047873 * evals[84][1], 12693612801792047873 * evals[84][2]];
    signal tmp_7263[3] <== CMul()(tmp_7262, challengesStage2[0]);
    signal tmp_7264[3] <== [tmp_7263[0] + evals[108][0], tmp_7263[1] + evals[108][1], tmp_7263[2] + evals[108][2]];
    signal tmp_7265[3] <== CMul()(tmp_7264, challengesStage2[0]);
    signal tmp_7266[3] <== [tmp_7265[0] + 1, tmp_7265[1], tmp_7265[2]];
    signal tmp_7267[3] <== [tmp_7266[0] + challengesStage2[1][0], tmp_7266[1] + challengesStage2[1][1], tmp_7266[2] + challengesStage2[1][2]];
    signal tmp_7268[3] <== [tmp_7267[0] - 1, tmp_7267[1], tmp_7267[2]];
    signal tmp_7269[3] <== [tmp_7268[0] + 1, tmp_7268[1], tmp_7268[2]];
    signal tmp_7270[3] <== CMul()(tmp_7261, tmp_7269);
    signal tmp_7271[3] <== [9404062091095256088 * evals[84][0], 9404062091095256088 * evals[84][1], 9404062091095256088 * evals[84][2]];
    signal tmp_7272[3] <== CMul()(tmp_7271, challengesStage2[0]);
    signal tmp_7273[3] <== [tmp_7272[0] + evals[109][0], tmp_7272[1] + evals[109][1], tmp_7272[2] + evals[109][2]];
    signal tmp_7274[3] <== CMul()(tmp_7273, challengesStage2[0]);
    signal tmp_7275[3] <== [tmp_7274[0] + 1, tmp_7274[1], tmp_7274[2]];
    signal tmp_7276[3] <== [tmp_7275[0] + challengesStage2[1][0], tmp_7275[1] + challengesStage2[1][1], tmp_7275[2] + challengesStage2[1][2]];
    signal tmp_7277[3] <== [tmp_7276[0] - 1, tmp_7276[1], tmp_7276[2]];
    signal tmp_7278[3] <== [tmp_7277[0] + 1, tmp_7277[1], tmp_7277[2]];
    signal tmp_7279[3] <== CMul()(tmp_7270, tmp_7278);
    signal tmp_7280[3] <== CMul()(evals[144], tmp_7279);
    signal tmp_7281[3] <== CMul()(evals[49], challengesStage2[0]);
    signal tmp_7282[3] <== [tmp_7281[0] + evals[107][0], tmp_7281[1] + evals[107][1], tmp_7281[2] + evals[107][2]];
    signal tmp_7283[3] <== CMul()(tmp_7282, challengesStage2[0]);
    signal tmp_7284[3] <== [tmp_7283[0] + 1, tmp_7283[1], tmp_7283[2]];
    signal tmp_7285[3] <== [tmp_7284[0] + challengesStage2[1][0], tmp_7284[1] + challengesStage2[1][1], tmp_7284[2] + challengesStage2[1][2]];
    signal tmp_7286[3] <== [tmp_7285[0] - 1, tmp_7285[1], tmp_7285[2]];
    signal tmp_7287[3] <== [tmp_7286[0] + 1, tmp_7286[1], tmp_7286[2]];
    signal tmp_7288[3] <== CMul()(evals[143], tmp_7287);
    signal tmp_7289[3] <== CMul()(evals[50], challengesStage2[0]);
    signal tmp_7290[3] <== [tmp_7289[0] + evals[108][0], tmp_7289[1] + evals[108][1], tmp_7289[2] + evals[108][2]];
    signal tmp_7291[3] <== CMul()(tmp_7290, challengesStage2[0]);
    signal tmp_7292[3] <== [tmp_7291[0] + 1, tmp_7291[1], tmp_7291[2]];
    signal tmp_7293[3] <== [tmp_7292[0] + challengesStage2[1][0], tmp_7292[1] + challengesStage2[1][1], tmp_7292[2] + challengesStage2[1][2]];
    signal tmp_7294[3] <== [tmp_7293[0] - 1, tmp_7293[1], tmp_7293[2]];
    signal tmp_7295[3] <== [tmp_7294[0] + 1, tmp_7294[1], tmp_7294[2]];
    signal tmp_7296[3] <== CMul()(tmp_7288, tmp_7295);
    signal tmp_7297[3] <== CMul()(evals[51], challengesStage2[0]);
    signal tmp_7298[3] <== [tmp_7297[0] + evals[109][0], tmp_7297[1] + evals[109][1], tmp_7297[2] + evals[109][2]];
    signal tmp_7299[3] <== CMul()(tmp_7298, challengesStage2[0]);
    signal tmp_7300[3] <== [tmp_7299[0] + 1, tmp_7299[1], tmp_7299[2]];
    signal tmp_7301[3] <== [tmp_7300[0] + challengesStage2[1][0], tmp_7300[1] + challengesStage2[1][1], tmp_7300[2] + challengesStage2[1][2]];
    signal tmp_7302[3] <== [tmp_7301[0] - 1, tmp_7301[1], tmp_7301[2]];
    signal tmp_7303[3] <== [tmp_7302[0] + 1, tmp_7302[1], tmp_7302[2]];
    signal tmp_7304[3] <== CMul()(tmp_7296, tmp_7303);
    signal tmp_7305[3] <== CMul()(evals[52], challengesStage2[0]);
    signal tmp_7306[3] <== [tmp_7305[0] + evals[110][0], tmp_7305[1] + evals[110][1], tmp_7305[2] + evals[110][2]];
    signal tmp_7307[3] <== CMul()(tmp_7306, challengesStage2[0]);
    signal tmp_7308[3] <== [tmp_7307[0] + 1, tmp_7307[1], tmp_7307[2]];
    signal tmp_7309[3] <== [tmp_7308[0] + challengesStage2[1][0], tmp_7308[1] + challengesStage2[1][1], tmp_7308[2] + challengesStage2[1][2]];
    signal tmp_7310[3] <== [tmp_7309[0] - 1, tmp_7309[1], tmp_7309[2]];
    signal tmp_7311[3] <== [tmp_7310[0] + 1, tmp_7310[1], tmp_7310[2]];
    signal tmp_7312[3] <== CMul()(tmp_7304, tmp_7311);
    signal tmp_7313[3] <== [tmp_7280[0] - tmp_7312[0], tmp_7280[1] - tmp_7312[1], tmp_7280[2] - tmp_7312[2]];
    signal tmp_7314[3] <== [tmp_7244[0] + tmp_7313[0], tmp_7244[1] + tmp_7313[1], tmp_7244[2] + tmp_7313[2]];
    signal tmp_7315[3] <== CMul()(challengeQ, tmp_7314);
    signal tmp_7316[3] <== [14878451024585790464 * evals[84][0], 14878451024585790464 * evals[84][1], 14878451024585790464 * evals[84][2]];
    signal tmp_7317[3] <== CMul()(tmp_7316, challengesStage2[0]);
    signal tmp_7318[3] <== [tmp_7317[0] + evals[110][0], tmp_7317[1] + evals[110][1], tmp_7317[2] + evals[110][2]];
    signal tmp_7319[3] <== CMul()(tmp_7318, challengesStage2[0]);
    signal tmp_7320[3] <== [tmp_7319[0] + 1, tmp_7319[1], tmp_7319[2]];
    signal tmp_7321[3] <== [tmp_7320[0] + challengesStage2[1][0], tmp_7320[1] + challengesStage2[1][1], tmp_7320[2] + challengesStage2[1][2]];
    signal tmp_7322[3] <== [tmp_7321[0] - 1, tmp_7321[1], tmp_7321[2]];
    signal tmp_7323[3] <== [tmp_7322[0] + 1, tmp_7322[1], tmp_7322[2]];
    signal tmp_7324[3] <== [11395689535312925715 * evals[84][0], 11395689535312925715 * evals[84][1], 11395689535312925715 * evals[84][2]];
    signal tmp_7325[3] <== CMul()(tmp_7324, challengesStage2[0]);
    signal tmp_7326[3] <== [tmp_7325[0] + evals[111][0], tmp_7325[1] + evals[111][1], tmp_7325[2] + evals[111][2]];
    signal tmp_7327[3] <== CMul()(tmp_7326, challengesStage2[0]);
    signal tmp_7328[3] <== [tmp_7327[0] + 1, tmp_7327[1], tmp_7327[2]];
    signal tmp_7329[3] <== [tmp_7328[0] + challengesStage2[1][0], tmp_7328[1] + challengesStage2[1][1], tmp_7328[2] + challengesStage2[1][2]];
    signal tmp_7330[3] <== [tmp_7329[0] - 1, tmp_7329[1], tmp_7329[2]];
    signal tmp_7331[3] <== [tmp_7330[0] + 1, tmp_7330[1], tmp_7330[2]];
    signal tmp_7332[3] <== CMul()(tmp_7323, tmp_7331);
    signal tmp_7333[3] <== [5975151027177897326 * evals[84][0], 5975151027177897326 * evals[84][1], 5975151027177897326 * evals[84][2]];
    signal tmp_7334[3] <== CMul()(tmp_7333, challengesStage2[0]);
    signal tmp_7335[3] <== [tmp_7334[0] + evals[112][0], tmp_7334[1] + evals[112][1], tmp_7334[2] + evals[112][2]];
    signal tmp_7336[3] <== CMul()(tmp_7335, challengesStage2[0]);
    signal tmp_7337[3] <== [tmp_7336[0] + 1, tmp_7336[1], tmp_7336[2]];
    signal tmp_7338[3] <== [tmp_7337[0] + challengesStage2[1][0], tmp_7337[1] + challengesStage2[1][1], tmp_7337[2] + challengesStage2[1][2]];
    signal tmp_7339[3] <== [tmp_7338[0] - 1, tmp_7338[1], tmp_7338[2]];
    signal tmp_7340[3] <== [tmp_7339[0] + 1, tmp_7339[1], tmp_7339[2]];
    signal tmp_7341[3] <== CMul()(tmp_7332, tmp_7340);
    signal tmp_7342[3] <== [3914263340919964335 * evals[84][0], 3914263340919964335 * evals[84][1], 3914263340919964335 * evals[84][2]];
    signal tmp_7343[3] <== CMul()(tmp_7342, challengesStage2[0]);
    signal tmp_7344[3] <== [tmp_7343[0] + evals[113][0], tmp_7343[1] + evals[113][1], tmp_7343[2] + evals[113][2]];
    signal tmp_7345[3] <== CMul()(tmp_7344, challengesStage2[0]);
    signal tmp_7346[3] <== [tmp_7345[0] + 1, tmp_7345[1], tmp_7345[2]];
    signal tmp_7347[3] <== [tmp_7346[0] + challengesStage2[1][0], tmp_7346[1] + challengesStage2[1][1], tmp_7346[2] + challengesStage2[1][2]];
    signal tmp_7348[3] <== [tmp_7347[0] - 1, tmp_7347[1], tmp_7347[2]];
    signal tmp_7349[3] <== [tmp_7348[0] + 1, tmp_7348[1], tmp_7348[2]];
    signal tmp_7350[3] <== CMul()(tmp_7341, tmp_7349);
    signal tmp_7351[3] <== CMul()(evals[145], tmp_7350);
    signal tmp_7352[3] <== CMul()(evals[53], challengesStage2[0]);
    signal tmp_7353[3] <== [tmp_7352[0] + evals[111][0], tmp_7352[1] + evals[111][1], tmp_7352[2] + evals[111][2]];
    signal tmp_7354[3] <== CMul()(tmp_7353, challengesStage2[0]);
    signal tmp_7355[3] <== [tmp_7354[0] + 1, tmp_7354[1], tmp_7354[2]];
    signal tmp_7356[3] <== [tmp_7355[0] + challengesStage2[1][0], tmp_7355[1] + challengesStage2[1][1], tmp_7355[2] + challengesStage2[1][2]];
    signal tmp_7357[3] <== [tmp_7356[0] - 1, tmp_7356[1], tmp_7356[2]];
    signal tmp_7358[3] <== [tmp_7357[0] + 1, tmp_7357[1], tmp_7357[2]];
    signal tmp_7359[3] <== CMul()(evals[144], tmp_7358);
    signal tmp_7360[3] <== CMul()(evals[54], challengesStage2[0]);
    signal tmp_7361[3] <== [tmp_7360[0] + evals[112][0], tmp_7360[1] + evals[112][1], tmp_7360[2] + evals[112][2]];
    signal tmp_7362[3] <== CMul()(tmp_7361, challengesStage2[0]);
    signal tmp_7363[3] <== [tmp_7362[0] + 1, tmp_7362[1], tmp_7362[2]];
    signal tmp_7364[3] <== [tmp_7363[0] + challengesStage2[1][0], tmp_7363[1] + challengesStage2[1][1], tmp_7363[2] + challengesStage2[1][2]];
    signal tmp_7365[3] <== [tmp_7364[0] - 1, tmp_7364[1], tmp_7364[2]];
    signal tmp_7366[3] <== [tmp_7365[0] + 1, tmp_7365[1], tmp_7365[2]];
    signal tmp_7367[3] <== CMul()(tmp_7359, tmp_7366);
    signal tmp_7368[3] <== CMul()(evals[55], challengesStage2[0]);
    signal tmp_7369[3] <== [tmp_7368[0] + evals[113][0], tmp_7368[1] + evals[113][1], tmp_7368[2] + evals[113][2]];
    signal tmp_7370[3] <== CMul()(tmp_7369, challengesStage2[0]);
    signal tmp_7371[3] <== [tmp_7370[0] + 1, tmp_7370[1], tmp_7370[2]];
    signal tmp_7372[3] <== [tmp_7371[0] + challengesStage2[1][0], tmp_7371[1] + challengesStage2[1][1], tmp_7371[2] + challengesStage2[1][2]];
    signal tmp_7373[3] <== [tmp_7372[0] - 1, tmp_7372[1], tmp_7372[2]];
    signal tmp_7374[3] <== [tmp_7373[0] + 1, tmp_7373[1], tmp_7373[2]];
    signal tmp_7375[3] <== CMul()(tmp_7367, tmp_7374);
    signal tmp_7376[3] <== CMul()(evals[56], challengesStage2[0]);
    signal tmp_7377[3] <== [tmp_7376[0] + evals[114][0], tmp_7376[1] + evals[114][1], tmp_7376[2] + evals[114][2]];
    signal tmp_7378[3] <== CMul()(tmp_7377, challengesStage2[0]);
    signal tmp_7379[3] <== [tmp_7378[0] + 1, tmp_7378[1], tmp_7378[2]];
    signal tmp_7380[3] <== [tmp_7379[0] + challengesStage2[1][0], tmp_7379[1] + challengesStage2[1][1], tmp_7379[2] + challengesStage2[1][2]];
    signal tmp_7381[3] <== [tmp_7380[0] - 1, tmp_7380[1], tmp_7380[2]];
    signal tmp_7382[3] <== [tmp_7381[0] + 1, tmp_7381[1], tmp_7381[2]];
    signal tmp_7383[3] <== CMul()(tmp_7375, tmp_7382);
    signal tmp_7384[3] <== [tmp_7351[0] - tmp_7383[0], tmp_7351[1] - tmp_7383[1], tmp_7351[2] - tmp_7383[2]];
    signal tmp_7385[3] <== [tmp_7315[0] + tmp_7384[0], tmp_7315[1] + tmp_7384[1], tmp_7315[2] + tmp_7384[2]];
    signal tmp_7386[3] <== CMul()(challengeQ, tmp_7385);
    signal tmp_7387[3] <== [7939331167302551519 * evals[84][0], 7939331167302551519 * evals[84][1], 7939331167302551519 * evals[84][2]];
    signal tmp_7388[3] <== CMul()(tmp_7387, challengesStage2[0]);
    signal tmp_7389[3] <== [tmp_7388[0] + evals[114][0], tmp_7388[1] + evals[114][1], tmp_7388[2] + evals[114][2]];
    signal tmp_7390[3] <== CMul()(tmp_7389, challengesStage2[0]);
    signal tmp_7391[3] <== [tmp_7390[0] + 1, tmp_7390[1], tmp_7390[2]];
    signal tmp_7392[3] <== [tmp_7391[0] + challengesStage2[1][0], tmp_7391[1] + challengesStage2[1][1], tmp_7391[2] + challengesStage2[1][2]];
    signal tmp_7393[3] <== [tmp_7392[0] - 1, tmp_7392[1], tmp_7392[2]];
    signal tmp_7394[3] <== [tmp_7393[0] + 1, tmp_7393[1], tmp_7393[2]];
    signal tmp_7395[3] <== [6372337341823991747 * evals[84][0], 6372337341823991747 * evals[84][1], 6372337341823991747 * evals[84][2]];
    signal tmp_7396[3] <== CMul()(tmp_7395, challengesStage2[0]);
    signal tmp_7397[3] <== [tmp_7396[0] + evals[115][0], tmp_7396[1] + evals[115][1], tmp_7396[2] + evals[115][2]];
    signal tmp_7398[3] <== CMul()(tmp_7397, challengesStage2[0]);
    signal tmp_7399[3] <== [tmp_7398[0] + 1, tmp_7398[1], tmp_7398[2]];
    signal tmp_7400[3] <== [tmp_7399[0] + challengesStage2[1][0], tmp_7399[1] + challengesStage2[1][1], tmp_7399[2] + challengesStage2[1][2]];
    signal tmp_7401[3] <== [tmp_7400[0] - 1, tmp_7400[1], tmp_7400[2]];
    signal tmp_7402[3] <== [tmp_7401[0] + 1, tmp_7401[1], tmp_7401[2]];
    signal tmp_7403[3] <== CMul()(tmp_7394, tmp_7402);
    signal tmp_7404[3] <== [12069298394184580155 * evals[84][0], 12069298394184580155 * evals[84][1], 12069298394184580155 * evals[84][2]];
    signal tmp_7405[3] <== CMul()(tmp_7404, challengesStage2[0]);
    signal tmp_7406[3] <== [tmp_7405[0] + evals[116][0], tmp_7405[1] + evals[116][1], tmp_7405[2] + evals[116][2]];
    signal tmp_7407[3] <== CMul()(tmp_7406, challengesStage2[0]);
    signal tmp_7408[3] <== [tmp_7407[0] + 1, tmp_7407[1], tmp_7407[2]];
    signal tmp_7409[3] <== [tmp_7408[0] + challengesStage2[1][0], tmp_7408[1] + challengesStage2[1][1], tmp_7408[2] + challengesStage2[1][2]];
    signal tmp_7410[3] <== [tmp_7409[0] - 1, tmp_7409[1], tmp_7409[2]];
    signal tmp_7411[3] <== [tmp_7410[0] + 1, tmp_7410[1], tmp_7410[2]];
    signal tmp_7412[3] <== CMul()(tmp_7403, tmp_7411);
    signal tmp_7413[3] <== [4304519570518954644 * evals[84][0], 4304519570518954644 * evals[84][1], 4304519570518954644 * evals[84][2]];
    signal tmp_7414[3] <== CMul()(tmp_7413, challengesStage2[0]);
    signal tmp_7415[3] <== [tmp_7414[0] + evals[117][0], tmp_7414[1] + evals[117][1], tmp_7414[2] + evals[117][2]];
    signal tmp_7416[3] <== CMul()(tmp_7415, challengesStage2[0]);
    signal tmp_7417[3] <== [tmp_7416[0] + 1, tmp_7416[1], tmp_7416[2]];
    signal tmp_7418[3] <== [tmp_7417[0] + challengesStage2[1][0], tmp_7417[1] + challengesStage2[1][1], tmp_7417[2] + challengesStage2[1][2]];
    signal tmp_7419[3] <== [tmp_7418[0] - 1, tmp_7418[1], tmp_7418[2]];
    signal tmp_7420[3] <== [tmp_7419[0] + 1, tmp_7419[1], tmp_7419[2]];
    signal tmp_7421[3] <== CMul()(tmp_7412, tmp_7420);
    signal tmp_7422[3] <== CMul()(evals[146], tmp_7421);
    signal tmp_7423[3] <== CMul()(evals[57], challengesStage2[0]);
    signal tmp_7424[3] <== [tmp_7423[0] + evals[115][0], tmp_7423[1] + evals[115][1], tmp_7423[2] + evals[115][2]];
    signal tmp_7425[3] <== CMul()(tmp_7424, challengesStage2[0]);
    signal tmp_7426[3] <== [tmp_7425[0] + 1, tmp_7425[1], tmp_7425[2]];
    signal tmp_7427[3] <== [tmp_7426[0] + challengesStage2[1][0], tmp_7426[1] + challengesStage2[1][1], tmp_7426[2] + challengesStage2[1][2]];
    signal tmp_7428[3] <== [tmp_7427[0] - 1, tmp_7427[1], tmp_7427[2]];
    signal tmp_7429[3] <== [tmp_7428[0] + 1, tmp_7428[1], tmp_7428[2]];
    signal tmp_7430[3] <== CMul()(evals[145], tmp_7429);
    signal tmp_7431[3] <== CMul()(evals[58], challengesStage2[0]);
    signal tmp_7432[3] <== [tmp_7431[0] + evals[116][0], tmp_7431[1] + evals[116][1], tmp_7431[2] + evals[116][2]];
    signal tmp_7433[3] <== CMul()(tmp_7432, challengesStage2[0]);
    signal tmp_7434[3] <== [tmp_7433[0] + 1, tmp_7433[1], tmp_7433[2]];
    signal tmp_7435[3] <== [tmp_7434[0] + challengesStage2[1][0], tmp_7434[1] + challengesStage2[1][1], tmp_7434[2] + challengesStage2[1][2]];
    signal tmp_7436[3] <== [tmp_7435[0] - 1, tmp_7435[1], tmp_7435[2]];
    signal tmp_7437[3] <== [tmp_7436[0] + 1, tmp_7436[1], tmp_7436[2]];
    signal tmp_7438[3] <== CMul()(tmp_7430, tmp_7437);
    signal tmp_7439[3] <== CMul()(evals[59], challengesStage2[0]);
    signal tmp_7440[3] <== [tmp_7439[0] + evals[117][0], tmp_7439[1] + evals[117][1], tmp_7439[2] + evals[117][2]];
    signal tmp_7441[3] <== CMul()(tmp_7440, challengesStage2[0]);
    signal tmp_7442[3] <== [tmp_7441[0] + 1, tmp_7441[1], tmp_7441[2]];
    signal tmp_7443[3] <== [tmp_7442[0] + challengesStage2[1][0], tmp_7442[1] + challengesStage2[1][1], tmp_7442[2] + challengesStage2[1][2]];
    signal tmp_7444[3] <== [tmp_7443[0] - 1, tmp_7443[1], tmp_7443[2]];
    signal tmp_7445[3] <== [tmp_7444[0] + 1, tmp_7444[1], tmp_7444[2]];
    signal tmp_7446[3] <== CMul()(tmp_7438, tmp_7445);
    signal tmp_7447[3] <== CMul()(evals[60], challengesStage2[0]);
    signal tmp_7448[3] <== [tmp_7447[0] + evals[118][0], tmp_7447[1] + evals[118][1], tmp_7447[2] + evals[118][2]];
    signal tmp_7449[3] <== CMul()(tmp_7448, challengesStage2[0]);
    signal tmp_7450[3] <== [tmp_7449[0] + 1, tmp_7449[1], tmp_7449[2]];
    signal tmp_7451[3] <== [tmp_7450[0] + challengesStage2[1][0], tmp_7450[1] + challengesStage2[1][1], tmp_7450[2] + challengesStage2[1][2]];
    signal tmp_7452[3] <== [tmp_7451[0] - 1, tmp_7451[1], tmp_7451[2]];
    signal tmp_7453[3] <== [tmp_7452[0] + 1, tmp_7452[1], tmp_7452[2]];
    signal tmp_7454[3] <== CMul()(tmp_7446, tmp_7453);
    signal tmp_7455[3] <== [tmp_7422[0] - tmp_7454[0], tmp_7422[1] - tmp_7454[1], tmp_7422[2] - tmp_7454[2]];
    signal tmp_7456[3] <== [tmp_7386[0] + tmp_7455[0], tmp_7386[1] + tmp_7455[1], tmp_7386[2] + tmp_7455[2]];
    signal tmp_7457[3] <== CMul()(challengeQ, tmp_7456);
    signal tmp_7458[3] <== [12241777603010464820 * evals[84][0], 12241777603010464820 * evals[84][1], 12241777603010464820 * evals[84][2]];
    signal tmp_7459[3] <== CMul()(tmp_7458, challengesStage2[0]);
    signal tmp_7460[3] <== [tmp_7459[0] + evals[118][0], tmp_7459[1] + evals[118][1], tmp_7459[2] + evals[118][2]];
    signal tmp_7461[3] <== CMul()(tmp_7460, challengesStage2[0]);
    signal tmp_7462[3] <== [tmp_7461[0] + 1, tmp_7461[1], tmp_7461[2]];
    signal tmp_7463[3] <== [tmp_7462[0] + challengesStage2[1][0], tmp_7462[1] + challengesStage2[1][1], tmp_7462[2] + challengesStage2[1][2]];
    signal tmp_7464[3] <== [tmp_7463[0] - 1, tmp_7463[1], tmp_7463[2]];
    signal tmp_7465[3] <== [tmp_7464[0] + 1, tmp_7464[1], tmp_7464[2]];
    signal tmp_7466[3] <== [17006090354634284996 * evals[84][0], 17006090354634284996 * evals[84][1], 17006090354634284996 * evals[84][2]];
    signal tmp_7467[3] <== CMul()(tmp_7466, challengesStage2[0]);
    signal tmp_7468[3] <== [tmp_7467[0] + evals[119][0], tmp_7467[1] + evals[119][1], tmp_7467[2] + evals[119][2]];
    signal tmp_7469[3] <== CMul()(tmp_7468, challengesStage2[0]);
    signal tmp_7470[3] <== [tmp_7469[0] + 1, tmp_7469[1], tmp_7469[2]];
    signal tmp_7471[3] <== [tmp_7470[0] + challengesStage2[1][0], tmp_7470[1] + challengesStage2[1][1], tmp_7470[2] + challengesStage2[1][2]];
    signal tmp_7472[3] <== [tmp_7471[0] - 1, tmp_7471[1], tmp_7471[2]];
    signal tmp_7473[3] <== [tmp_7472[0] + 1, tmp_7472[1], tmp_7472[2]];
    signal tmp_7474[3] <== CMul()(tmp_7465, tmp_7473);
    signal tmp_7475[3] <== [8160273888729427702 * evals[84][0], 8160273888729427702 * evals[84][1], 8160273888729427702 * evals[84][2]];
    signal tmp_7476[3] <== CMul()(tmp_7475, challengesStage2[0]);
    signal tmp_7477[3] <== [tmp_7476[0] + evals[120][0], tmp_7476[1] + evals[120][1], tmp_7476[2] + evals[120][2]];
    signal tmp_7478[3] <== CMul()(tmp_7477, challengesStage2[0]);
    signal tmp_7479[3] <== [tmp_7478[0] + 1, tmp_7478[1], tmp_7478[2]];
    signal tmp_7480[3] <== [tmp_7479[0] + challengesStage2[1][0], tmp_7479[1] + challengesStage2[1][1], tmp_7479[2] + challengesStage2[1][2]];
    signal tmp_7481[3] <== [tmp_7480[0] - 1, tmp_7480[1], tmp_7480[2]];
    signal tmp_7482[3] <== [tmp_7481[0] + 1, tmp_7481[1], tmp_7481[2]];
    signal tmp_7483[3] <== CMul()(tmp_7474, tmp_7482);
    signal tmp_7484[3] <== [4612251612089625347 * evals[84][0], 4612251612089625347 * evals[84][1], 4612251612089625347 * evals[84][2]];
    signal tmp_7485[3] <== CMul()(tmp_7484, challengesStage2[0]);
    signal tmp_7486[3] <== [tmp_7485[0] + evals[121][0], tmp_7485[1] + evals[121][1], tmp_7485[2] + evals[121][2]];
    signal tmp_7487[3] <== CMul()(tmp_7486, challengesStage2[0]);
    signal tmp_7488[3] <== [tmp_7487[0] + 1, tmp_7487[1], tmp_7487[2]];
    signal tmp_7489[3] <== [tmp_7488[0] + challengesStage2[1][0], tmp_7488[1] + challengesStage2[1][1], tmp_7488[2] + challengesStage2[1][2]];
    signal tmp_7490[3] <== [tmp_7489[0] - 1, tmp_7489[1], tmp_7489[2]];
    signal tmp_7491[3] <== [tmp_7490[0] + 1, tmp_7490[1], tmp_7490[2]];
    signal tmp_7492[3] <== CMul()(tmp_7483, tmp_7491);
    signal tmp_7493[3] <== CMul()(evals[138], tmp_7492);
    signal tmp_7494[3] <== CMul()(evals[61], challengesStage2[0]);
    signal tmp_7495[3] <== [tmp_7494[0] + evals[119][0], tmp_7494[1] + evals[119][1], tmp_7494[2] + evals[119][2]];
    signal tmp_7496[3] <== CMul()(tmp_7495, challengesStage2[0]);
    signal tmp_7497[3] <== [tmp_7496[0] + 1, tmp_7496[1], tmp_7496[2]];
    signal tmp_7498[3] <== [tmp_7497[0] + challengesStage2[1][0], tmp_7497[1] + challengesStage2[1][1], tmp_7497[2] + challengesStage2[1][2]];
    signal tmp_7499[3] <== [tmp_7498[0] - 1, tmp_7498[1], tmp_7498[2]];
    signal tmp_7500[3] <== [tmp_7499[0] + 1, tmp_7499[1], tmp_7499[2]];
    signal tmp_7501[3] <== CMul()(evals[146], tmp_7500);
    signal tmp_7502[3] <== CMul()(evals[62], challengesStage2[0]);
    signal tmp_7503[3] <== [tmp_7502[0] + evals[120][0], tmp_7502[1] + evals[120][1], tmp_7502[2] + evals[120][2]];
    signal tmp_7504[3] <== CMul()(tmp_7503, challengesStage2[0]);
    signal tmp_7505[3] <== [tmp_7504[0] + 1, tmp_7504[1], tmp_7504[2]];
    signal tmp_7506[3] <== [tmp_7505[0] + challengesStage2[1][0], tmp_7505[1] + challengesStage2[1][1], tmp_7505[2] + challengesStage2[1][2]];
    signal tmp_7507[3] <== [tmp_7506[0] - 1, tmp_7506[1], tmp_7506[2]];
    signal tmp_7508[3] <== [tmp_7507[0] + 1, tmp_7507[1], tmp_7507[2]];
    signal tmp_7509[3] <== CMul()(tmp_7501, tmp_7508);
    signal tmp_7510[3] <== CMul()(evals[63], challengesStage2[0]);
    signal tmp_7511[3] <== [tmp_7510[0] + evals[121][0], tmp_7510[1] + evals[121][1], tmp_7510[2] + evals[121][2]];
    signal tmp_7512[3] <== CMul()(tmp_7511, challengesStage2[0]);
    signal tmp_7513[3] <== [tmp_7512[0] + 1, tmp_7512[1], tmp_7512[2]];
    signal tmp_7514[3] <== [tmp_7513[0] + challengesStage2[1][0], tmp_7513[1] + challengesStage2[1][1], tmp_7513[2] + challengesStage2[1][2]];
    signal tmp_7515[3] <== [tmp_7514[0] - 1, tmp_7514[1], tmp_7514[2]];
    signal tmp_7516[3] <== [tmp_7515[0] + 1, tmp_7515[1], tmp_7515[2]];
    signal tmp_7517[3] <== CMul()(tmp_7509, tmp_7516);
    signal tmp_7518[3] <== CMul()(evals[172], tmp_7517);
    signal tmp_7519[3] <== [tmp_7493[0] - tmp_7518[0], tmp_7493[1] - tmp_7518[1], tmp_7493[2] - tmp_7518[2]];
    signal tmp_7520[3] <== [tmp_7457[0] + tmp_7519[0], tmp_7457[1] + tmp_7519[1], tmp_7457[2] + tmp_7519[2]];
    signal tmp_7521[3] <== CMul()(challengeQ, tmp_7520);
    signal tmp_7522[3] <== [1 - evals[138][0], -evals[138][1], -evals[138][2]];
    signal tmp_7523[3] <== CMul()(evals[179], tmp_7522);
    signal tmp_7524[3] <== [tmp_7521[0] + tmp_7523[0], tmp_7521[1] + tmp_7523[1], tmp_7521[2] + tmp_7523[2]];
    signal tmp_7525[3] <== CMul()(challengeQ, tmp_7524);
    signal tmp_3662[3] <== CMul()(tmp_4451, tmp_4451);
    signal tmp_7526[3] <== [evals[147][0] - tmp_3662[0], evals[147][1] - tmp_3662[1], evals[147][2] - tmp_3662[2]];
    signal tmp_7527[3] <== [tmp_7525[0] + tmp_7526[0], tmp_7525[1] + tmp_7526[1], tmp_7525[2] + tmp_7526[2]];
    signal tmp_7528[3] <== CMul()(challengeQ, tmp_7527);
    signal tmp_3666[3] <== CMul()(tmp_4475, tmp_4475);
    signal tmp_7529[3] <== [evals[148][0] - tmp_3666[0], evals[148][1] - tmp_3666[1], evals[148][2] - tmp_3666[2]];
    signal tmp_7530[3] <== [tmp_7528[0] + tmp_7529[0], tmp_7528[1] + tmp_7529[1], tmp_7528[2] + tmp_7529[2]];
    signal tmp_7531[3] <== CMul()(challengeQ, tmp_7530);
    signal tmp_3670[3] <== CMul()(tmp_4498, tmp_4498);
    signal tmp_7532[3] <== [evals[149][0] - tmp_3670[0], evals[149][1] - tmp_3670[1], evals[149][2] - tmp_3670[2]];
    signal tmp_7533[3] <== [tmp_7531[0] + tmp_7532[0], tmp_7531[1] + tmp_7532[1], tmp_7531[2] + tmp_7532[2]];
    signal tmp_7534[3] <== CMul()(challengeQ, tmp_7533);
    signal tmp_3674[3] <== CMul()(tmp_4527, tmp_4527);
    signal tmp_7535[3] <== [evals[150][0] - tmp_3674[0], evals[150][1] - tmp_3674[1], evals[150][2] - tmp_3674[2]];
    signal tmp_7536[3] <== [tmp_7534[0] + tmp_7535[0], tmp_7534[1] + tmp_7535[1], tmp_7534[2] + tmp_7535[2]];
    signal tmp_7537[3] <== CMul()(challengeQ, tmp_7536);
    signal tmp_3678[3] <== CMul()(tmp_4556, tmp_4556);
    signal tmp_7538[3] <== [evals[151][0] - tmp_3678[0], evals[151][1] - tmp_3678[1], evals[151][2] - tmp_3678[2]];
    signal tmp_7539[3] <== [tmp_7537[0] + tmp_7538[0], tmp_7537[1] + tmp_7538[1], tmp_7537[2] + tmp_7538[2]];
    signal tmp_7540[3] <== CMul()(challengeQ, tmp_7539);
    signal tmp_3682[3] <== CMul()(tmp_4580, tmp_4580);
    signal tmp_7541[3] <== [evals[152][0] - tmp_3682[0], evals[152][1] - tmp_3682[1], evals[152][2] - tmp_3682[2]];
    signal tmp_7542[3] <== [tmp_7540[0] + tmp_7541[0], tmp_7540[1] + tmp_7541[1], tmp_7540[2] + tmp_7541[2]];
    signal tmp_7543[3] <== CMul()(challengeQ, tmp_7542);
    signal tmp_3686[3] <== CMul()(tmp_4603, tmp_4603);
    signal tmp_7544[3] <== [evals[153][0] - tmp_3686[0], evals[153][1] - tmp_3686[1], evals[153][2] - tmp_3686[2]];
    signal tmp_7545[3] <== [tmp_7543[0] + tmp_7544[0], tmp_7543[1] + tmp_7544[1], tmp_7543[2] + tmp_7544[2]];
    signal tmp_7546[3] <== CMul()(challengeQ, tmp_7545);
    signal tmp_3690[3] <== CMul()(tmp_4632, tmp_4632);
    signal tmp_7547[3] <== [evals[154][0] - tmp_3690[0], evals[154][1] - tmp_3690[1], evals[154][2] - tmp_3690[2]];
    signal tmp_7548[3] <== [tmp_7546[0] + tmp_7547[0], tmp_7546[1] + tmp_7547[1], tmp_7546[2] + tmp_7547[2]];
    signal tmp_7549[3] <== CMul()(challengeQ, tmp_7548);
    signal tmp_3694[3] <== CMul()(tmp_4662, tmp_4662);
    signal tmp_7550[3] <== [evals[155][0] - tmp_3694[0], evals[155][1] - tmp_3694[1], evals[155][2] - tmp_3694[2]];
    signal tmp_7551[3] <== [tmp_7549[0] + tmp_7550[0], tmp_7549[1] + tmp_7550[1], tmp_7549[2] + tmp_7550[2]];
    signal tmp_7552[3] <== CMul()(challengeQ, tmp_7551);
    signal tmp_3698[3] <== CMul()(tmp_4686, tmp_4686);
    signal tmp_7553[3] <== [evals[156][0] - tmp_3698[0], evals[156][1] - tmp_3698[1], evals[156][2] - tmp_3698[2]];
    signal tmp_7554[3] <== [tmp_7552[0] + tmp_7553[0], tmp_7552[1] + tmp_7553[1], tmp_7552[2] + tmp_7553[2]];
    signal tmp_7555[3] <== CMul()(challengeQ, tmp_7554);
    signal tmp_3702[3] <== CMul()(tmp_4709, tmp_4709);
    signal tmp_7556[3] <== [evals[157][0] - tmp_3702[0], evals[157][1] - tmp_3702[1], evals[157][2] - tmp_3702[2]];
    signal tmp_7557[3] <== [tmp_7555[0] + tmp_7556[0], tmp_7555[1] + tmp_7556[1], tmp_7555[2] + tmp_7556[2]];
    signal tmp_7558[3] <== CMul()(challengeQ, tmp_7557);
    signal tmp_3706[3] <== CMul()(tmp_4738, tmp_4738);
    signal tmp_7559[3] <== [evals[158][0] - tmp_3706[0], evals[158][1] - tmp_3706[1], evals[158][2] - tmp_3706[2]];
    signal tmp_7560[3] <== [tmp_7558[0] + tmp_7559[0], tmp_7558[1] + tmp_7559[1], tmp_7558[2] + tmp_7559[2]];
    signal tmp_7561[3] <== CMul()(challengeQ, tmp_7560);
    signal tmp_3710[3] <== CMul()(tmp_4768, tmp_4768);
    signal tmp_7562[3] <== [evals[159][0] - tmp_3710[0], evals[159][1] - tmp_3710[1], evals[159][2] - tmp_3710[2]];
    signal tmp_7563[3] <== [tmp_7561[0] + tmp_7562[0], tmp_7561[1] + tmp_7562[1], tmp_7561[2] + tmp_7562[2]];
    signal tmp_7564[3] <== CMul()(challengeQ, tmp_7563);
    signal tmp_3714[3] <== CMul()(tmp_4792, tmp_4792);
    signal tmp_7565[3] <== [evals[160][0] - tmp_3714[0], evals[160][1] - tmp_3714[1], evals[160][2] - tmp_3714[2]];
    signal tmp_7566[3] <== [tmp_7564[0] + tmp_7565[0], tmp_7564[1] + tmp_7565[1], tmp_7564[2] + tmp_7565[2]];
    signal tmp_7567[3] <== CMul()(challengeQ, tmp_7566);
    signal tmp_3718[3] <== CMul()(tmp_4815, tmp_4815);
    signal tmp_7568[3] <== [evals[161][0] - tmp_3718[0], evals[161][1] - tmp_3718[1], evals[161][2] - tmp_3718[2]];
    signal tmp_7569[3] <== [tmp_7567[0] + tmp_7568[0], tmp_7567[1] + tmp_7568[1], tmp_7567[2] + tmp_7568[2]];
    signal tmp_7570[3] <== CMul()(challengeQ, tmp_7569);
    signal tmp_3722[3] <== CMul()(tmp_4844, tmp_4844);
    signal tmp_7571[3] <== [evals[162][0] - tmp_3722[0], evals[162][1] - tmp_3722[1], evals[162][2] - tmp_3722[2]];
    signal tmp_7572[3] <== [tmp_7570[0] + tmp_7571[0], tmp_7570[1] + tmp_7571[1], tmp_7570[2] + tmp_7571[2]];
    signal tmp_7573[3] <== CMul()(challengeQ, tmp_7572);
    signal tmp_3726[3] <== CMul()(tmp_5945, tmp_5945);
    signal tmp_7574[3] <== [evals[163][0] - tmp_3726[0], evals[163][1] - tmp_3726[1], evals[163][2] - tmp_3726[2]];
    signal tmp_7575[3] <== [tmp_7573[0] + tmp_7574[0], tmp_7573[1] + tmp_7574[1], tmp_7573[2] + tmp_7574[2]];
    signal tmp_7576[3] <== CMul()(challengeQ, tmp_7575);
    signal tmp_3730[3] <== CMul()(tmp_6004, tmp_6004);
    signal tmp_7577[3] <== [evals[164][0] - tmp_3730[0], evals[164][1] - tmp_3730[1], evals[164][2] - tmp_3730[2]];
    signal tmp_7578[3] <== [tmp_7576[0] + tmp_7577[0], tmp_7576[1] + tmp_7577[1], tmp_7576[2] + tmp_7577[2]];
    signal tmp_7579[3] <== CMul()(challengeQ, tmp_7578);
    signal tmp_3734[3] <== CMul()(tmp_6063, tmp_6063);
    signal tmp_7580[3] <== [evals[165][0] - tmp_3734[0], evals[165][1] - tmp_3734[1], evals[165][2] - tmp_3734[2]];
    signal tmp_7581[3] <== [tmp_7579[0] + tmp_7580[0], tmp_7579[1] + tmp_7580[1], tmp_7579[2] + tmp_7580[2]];
    signal tmp_7582[3] <== CMul()(challengeQ, tmp_7581);
    signal tmp_3738[3] <== CMul()(tmp_6122, tmp_6122);
    signal tmp_7583[3] <== [evals[166][0] - tmp_3738[0], evals[166][1] - tmp_3738[1], evals[166][2] - tmp_3738[2]];
    signal tmp_7584[3] <== [tmp_7582[0] + tmp_7583[0], tmp_7582[1] + tmp_7583[1], tmp_7582[2] + tmp_7583[2]];
    signal tmp_7585[3] <== CMul()(challengeQ, tmp_7584);
    signal tmp_3742[3] <== CMul()(tmp_6181, tmp_6181);
    signal tmp_7586[3] <== [evals[167][0] - tmp_3742[0], evals[167][1] - tmp_3742[1], evals[167][2] - tmp_3742[2]];
    signal tmp_7587[3] <== [tmp_7585[0] + tmp_7586[0], tmp_7585[1] + tmp_7586[1], tmp_7585[2] + tmp_7586[2]];
    signal tmp_7588[3] <== CMul()(challengeQ, tmp_7587);
    signal tmp_3746[3] <== CMul()(tmp_6240, tmp_6240);
    signal tmp_7589[3] <== [evals[168][0] - tmp_3746[0], evals[168][1] - tmp_3746[1], evals[168][2] - tmp_3746[2]];
    signal tmp_7590[3] <== [tmp_7588[0] + tmp_7589[0], tmp_7588[1] + tmp_7589[1], tmp_7588[2] + tmp_7589[2]];
    signal tmp_7591[3] <== CMul()(challengeQ, tmp_7590);
    signal tmp_7592[3] <== CMul()(evals[98], evals[101]);
    signal tmp_7593[3] <== CMul()(evals[99], evals[103]);
    signal tmp_7594[3] <== [tmp_7592[0] + tmp_7593[0], tmp_7592[1] + tmp_7593[1], tmp_7592[2] + tmp_7593[2]];
    signal tmp_7595[3] <== CMul()(evals[100], evals[102]);
    signal tmp_7596[3] <== [tmp_7594[0] + tmp_7595[0], tmp_7594[1] + tmp_7595[1], tmp_7594[2] + tmp_7595[2]];
    signal tmp_7597[3] <== [tmp_7596[0] + evals[95][0], tmp_7596[1] + evals[95][1], tmp_7596[2] + evals[95][2]];
    signal tmp_7598[3] <== CMul()(tmp_7597, evals[101]);
    signal tmp_7599[3] <== CMul()(evals[98], evals[102]);
    signal tmp_7600[3] <== CMul()(evals[99], evals[101]);
    signal tmp_7601[3] <== [tmp_7599[0] + tmp_7600[0], tmp_7599[1] + tmp_7600[1], tmp_7599[2] + tmp_7600[2]];
    signal tmp_7602[3] <== CMul()(evals[99], evals[103]);
    signal tmp_7603[3] <== [tmp_7601[0] + tmp_7602[0], tmp_7601[1] + tmp_7602[1], tmp_7601[2] + tmp_7602[2]];
    signal tmp_7604[3] <== CMul()(evals[100], evals[102]);
    signal tmp_7605[3] <== [tmp_7603[0] + tmp_7604[0], tmp_7603[1] + tmp_7604[1], tmp_7603[2] + tmp_7604[2]];
    signal tmp_7606[3] <== CMul()(evals[100], evals[103]);
    signal tmp_7607[3] <== [tmp_7605[0] + tmp_7606[0], tmp_7605[1] + tmp_7606[1], tmp_7605[2] + tmp_7606[2]];
    signal tmp_7608[3] <== [tmp_7607[0] + evals[96][0], tmp_7607[1] + evals[96][1], tmp_7607[2] + evals[96][2]];
    signal tmp_7609[3] <== CMul()(tmp_7608, evals[103]);
    signal tmp_7610[3] <== [tmp_7598[0] + tmp_7609[0], tmp_7598[1] + tmp_7609[1], tmp_7598[2] + tmp_7609[2]];
    signal tmp_7611[3] <== CMul()(evals[98], evals[103]);
    signal tmp_7612[3] <== CMul()(evals[100], evals[103]);
    signal tmp_7613[3] <== [tmp_7611[0] + tmp_7612[0], tmp_7611[1] + tmp_7612[1], tmp_7611[2] + tmp_7612[2]];
    signal tmp_7614[3] <== CMul()(evals[100], evals[101]);
    signal tmp_7615[3] <== [tmp_7613[0] + tmp_7614[0], tmp_7613[1] + tmp_7614[1], tmp_7613[2] + tmp_7614[2]];
    signal tmp_7616[3] <== CMul()(evals[99], evals[102]);
    signal tmp_7617[3] <== [tmp_7615[0] + tmp_7616[0], tmp_7615[1] + tmp_7616[1], tmp_7615[2] + tmp_7616[2]];
    signal tmp_7618[3] <== [tmp_7617[0] + evals[97][0], tmp_7617[1] + evals[97][1], tmp_7617[2] + evals[97][2]];
    signal tmp_7619[3] <== CMul()(tmp_7618, evals[102]);
    signal tmp_7620[3] <== [tmp_7610[0] + tmp_7619[0], tmp_7610[1] + tmp_7619[1], tmp_7610[2] + tmp_7619[2]];
    signal tmp_7621[3] <== [tmp_7620[0] + evals[92][0], tmp_7620[1] + evals[92][1], tmp_7620[2] + evals[92][2]];
    signal tmp_7622[3] <== CMul()(tmp_7621, evals[101]);
    signal tmp_7623[3] <== CMul()(tmp_7597, evals[102]);
    signal tmp_7624[3] <== CMul()(tmp_7608, evals[101]);
    signal tmp_7625[3] <== [tmp_7623[0] + tmp_7624[0], tmp_7623[1] + tmp_7624[1], tmp_7623[2] + tmp_7624[2]];
    signal tmp_7626[3] <== CMul()(tmp_7608, evals[103]);
    signal tmp_7627[3] <== [tmp_7625[0] + tmp_7626[0], tmp_7625[1] + tmp_7626[1], tmp_7625[2] + tmp_7626[2]];
    signal tmp_7628[3] <== CMul()(tmp_7618, evals[102]);
    signal tmp_7629[3] <== [tmp_7627[0] + tmp_7628[0], tmp_7627[1] + tmp_7628[1], tmp_7627[2] + tmp_7628[2]];
    signal tmp_7630[3] <== CMul()(tmp_7618, evals[103]);
    signal tmp_7631[3] <== [tmp_7629[0] + tmp_7630[0], tmp_7629[1] + tmp_7630[1], tmp_7629[2] + tmp_7630[2]];
    signal tmp_7632[3] <== [tmp_7631[0] + evals[93][0], tmp_7631[1] + evals[93][1], tmp_7631[2] + evals[93][2]];
    signal tmp_7633[3] <== CMul()(tmp_7632, evals[103]);
    signal tmp_7634[3] <== [tmp_7622[0] + tmp_7633[0], tmp_7622[1] + tmp_7633[1], tmp_7622[2] + tmp_7633[2]];
    signal tmp_7635[3] <== CMul()(tmp_7597, evals[103]);
    signal tmp_7636[3] <== CMul()(tmp_7618, evals[103]);
    signal tmp_7637[3] <== [tmp_7635[0] + tmp_7636[0], tmp_7635[1] + tmp_7636[1], tmp_7635[2] + tmp_7636[2]];
    signal tmp_7638[3] <== CMul()(tmp_7618, evals[101]);
    signal tmp_7639[3] <== [tmp_7637[0] + tmp_7638[0], tmp_7637[1] + tmp_7638[1], tmp_7637[2] + tmp_7638[2]];
    signal tmp_7640[3] <== CMul()(tmp_7608, evals[102]);
    signal tmp_7641[3] <== [tmp_7639[0] + tmp_7640[0], tmp_7639[1] + tmp_7640[1], tmp_7639[2] + tmp_7640[2]];
    signal tmp_7642[3] <== [tmp_7641[0] + evals[94][0], tmp_7641[1] + evals[94][1], tmp_7641[2] + evals[94][2]];
    signal tmp_7643[3] <== CMul()(tmp_7642, evals[102]);
    signal tmp_7644[3] <== [tmp_7634[0] + tmp_7643[0], tmp_7634[1] + tmp_7643[1], tmp_7634[2] + tmp_7643[2]];
    signal tmp_7645[3] <== [tmp_7644[0] + evals[89][0], tmp_7644[1] + evals[89][1], tmp_7644[2] + evals[89][2]];
    signal tmp_7646[3] <== CMul()(tmp_7645, evals[101]);
    signal tmp_7647[3] <== CMul()(tmp_7621, evals[102]);
    signal tmp_7648[3] <== CMul()(tmp_7632, evals[101]);
    signal tmp_7649[3] <== [tmp_7647[0] + tmp_7648[0], tmp_7647[1] + tmp_7648[1], tmp_7647[2] + tmp_7648[2]];
    signal tmp_7650[3] <== CMul()(tmp_7632, evals[103]);
    signal tmp_7651[3] <== [tmp_7649[0] + tmp_7650[0], tmp_7649[1] + tmp_7650[1], tmp_7649[2] + tmp_7650[2]];
    signal tmp_7652[3] <== CMul()(tmp_7642, evals[102]);
    signal tmp_7653[3] <== [tmp_7651[0] + tmp_7652[0], tmp_7651[1] + tmp_7652[1], tmp_7651[2] + tmp_7652[2]];
    signal tmp_7654[3] <== CMul()(tmp_7642, evals[103]);
    signal tmp_7655[3] <== [tmp_7653[0] + tmp_7654[0], tmp_7653[1] + tmp_7654[1], tmp_7653[2] + tmp_7654[2]];
    signal tmp_7656[3] <== [tmp_7655[0] + evals[90][0], tmp_7655[1] + evals[90][1], tmp_7655[2] + evals[90][2]];
    signal tmp_7657[3] <== CMul()(tmp_7656, evals[103]);
    signal tmp_7658[3] <== [tmp_7646[0] + tmp_7657[0], tmp_7646[1] + tmp_7657[1], tmp_7646[2] + tmp_7657[2]];
    signal tmp_7659[3] <== CMul()(tmp_7621, evals[103]);
    signal tmp_7660[3] <== CMul()(tmp_7642, evals[103]);
    signal tmp_7661[3] <== [tmp_7659[0] + tmp_7660[0], tmp_7659[1] + tmp_7660[1], tmp_7659[2] + tmp_7660[2]];
    signal tmp_7662[3] <== CMul()(tmp_7642, evals[101]);
    signal tmp_7663[3] <== [tmp_7661[0] + tmp_7662[0], tmp_7661[1] + tmp_7662[1], tmp_7661[2] + tmp_7662[2]];
    signal tmp_7664[3] <== CMul()(tmp_7632, evals[102]);
    signal tmp_7665[3] <== [tmp_7663[0] + tmp_7664[0], tmp_7663[1] + tmp_7664[1], tmp_7663[2] + tmp_7664[2]];
    signal tmp_7666[3] <== [tmp_7665[0] + evals[91][0], tmp_7665[1] + evals[91][1], tmp_7665[2] + evals[91][2]];
    signal tmp_7667[3] <== CMul()(tmp_7666, evals[102]);
    signal tmp_7668[3] <== [tmp_7658[0] + tmp_7667[0], tmp_7658[1] + tmp_7667[1], tmp_7658[2] + tmp_7667[2]];
    signal tmp_7669[3] <== [tmp_7668[0] + evals[86][0], tmp_7668[1] + evals[86][1], tmp_7668[2] + evals[86][2]];
    signal tmp_3828[3] <== [tmp_7669[0] - evals[104][0], tmp_7669[1] - evals[104][1], tmp_7669[2] - evals[104][2]];
    signal tmp_7670[3] <== [evals[169][0] - tmp_3828[0], evals[169][1] - tmp_3828[1], evals[169][2] - tmp_3828[2]];
    signal tmp_7671[3] <== [tmp_7591[0] + tmp_7670[0], tmp_7591[1] + tmp_7670[1], tmp_7591[2] + tmp_7670[2]];
    signal tmp_7672[3] <== CMul()(challengeQ, tmp_7671);
    signal tmp_7673[3] <== CMul()(tmp_7645, evals[102]);
    signal tmp_7674[3] <== CMul()(tmp_7656, evals[101]);
    signal tmp_7675[3] <== [tmp_7673[0] + tmp_7674[0], tmp_7673[1] + tmp_7674[1], tmp_7673[2] + tmp_7674[2]];
    signal tmp_7676[3] <== CMul()(tmp_7656, evals[103]);
    signal tmp_7677[3] <== [tmp_7675[0] + tmp_7676[0], tmp_7675[1] + tmp_7676[1], tmp_7675[2] + tmp_7676[2]];
    signal tmp_7678[3] <== CMul()(tmp_7666, evals[102]);
    signal tmp_7679[3] <== [tmp_7677[0] + tmp_7678[0], tmp_7677[1] + tmp_7678[1], tmp_7677[2] + tmp_7678[2]];
    signal tmp_7680[3] <== CMul()(tmp_7666, evals[103]);
    signal tmp_7681[3] <== [tmp_7679[0] + tmp_7680[0], tmp_7679[1] + tmp_7680[1], tmp_7679[2] + tmp_7680[2]];
    signal tmp_7682[3] <== [tmp_7681[0] + evals[87][0], tmp_7681[1] + evals[87][1], tmp_7681[2] + evals[87][2]];
    signal tmp_3842[3] <== [tmp_7682[0] - evals[105][0], tmp_7682[1] - evals[105][1], tmp_7682[2] - evals[105][2]];
    signal tmp_7683[3] <== [evals[170][0] - tmp_3842[0], evals[170][1] - tmp_3842[1], evals[170][2] - tmp_3842[2]];
    signal tmp_7684[3] <== [tmp_7672[0] + tmp_7683[0], tmp_7672[1] + tmp_7683[1], tmp_7672[2] + tmp_7683[2]];
    signal tmp_7685[3] <== CMul()(challengeQ, tmp_7684);
    signal tmp_7686[3] <== CMul()(tmp_7645, evals[103]);
    signal tmp_7687[3] <== CMul()(tmp_7666, evals[103]);
    signal tmp_7688[3] <== [tmp_7686[0] + tmp_7687[0], tmp_7686[1] + tmp_7687[1], tmp_7686[2] + tmp_7687[2]];
    signal tmp_7689[3] <== CMul()(tmp_7666, evals[101]);
    signal tmp_7690[3] <== [tmp_7688[0] + tmp_7689[0], tmp_7688[1] + tmp_7689[1], tmp_7688[2] + tmp_7689[2]];
    signal tmp_7691[3] <== CMul()(tmp_7656, evals[102]);
    signal tmp_7692[3] <== [tmp_7690[0] + tmp_7691[0], tmp_7690[1] + tmp_7691[1], tmp_7690[2] + tmp_7691[2]];
    signal tmp_7693[3] <== [tmp_7692[0] + evals[88][0], tmp_7692[1] + evals[88][1], tmp_7692[2] + evals[88][2]];
    signal tmp_3854[3] <== [tmp_7693[0] - evals[106][0], tmp_7693[1] - evals[106][1], tmp_7693[2] - evals[106][2]];
    signal tmp_7694[3] <== [evals[171][0] - tmp_3854[0], evals[171][1] - tmp_3854[1], evals[171][2] - tmp_3854[2]];
    signal tmp_7695[3] <== [tmp_7685[0] + tmp_7694[0], tmp_7685[1] + tmp_7694[1], tmp_7685[2] + tmp_7694[2]];
    signal tmp_7696[3] <== CMul()(challengeQ, tmp_7695);
    signal tmp_7697[3] <== [1 - evals[85][0], -evals[85][1], -evals[85][2]];
    signal tmp_7698[3] <== CMul()(evals[19], tmp_7697);
    signal tmp_3860[3] <== [tmp_7698[0] + evals[85][0], tmp_7698[1] + evals[85][1], tmp_7698[2] + evals[85][2]];
    signal tmp_7699[3] <== [evals[172][0] - tmp_3860[0], evals[172][1] - tmp_3860[1], evals[172][2] - tmp_3860[2]];
    signal tmp_3862[3] <== [tmp_7696[0] + tmp_7699[0], tmp_7696[1] + tmp_7699[1], tmp_7696[2] + tmp_7699[2]];
    tmp_7700 <== CMul()(tmp_3862, Zh);
}


template VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[198][3];
    signal input publics[391];
    signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeXi) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[18][3];
    for (var i=0; i< 18 ; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeXi, challengeXi);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[17][0] - 1, zMul[17][1], zMul[17][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_3864[3];
 
    signal tmp_3866[3];
 
    signal tmp_3978[3];
 
    signal tmp_4448[3];
 
    signal tmp_4451[3];
 
    signal tmp_4452[3];
 
    signal tmp_4472[3];
 
    signal tmp_4475[3];
 
    signal tmp_4476[3];
 
    signal tmp_4495[3];
 
    signal tmp_4498[3];
 
    signal tmp_4499[3];
 
    signal tmp_4503[3];
 
    signal tmp_4524[3];
 
    signal tmp_4527[3];
 
    signal tmp_4528[3];
 
    signal tmp_4533[3];
 
    signal tmp_4534[3];
 
    signal tmp_4535[3];
 
    signal tmp_4553[3];
 
    signal tmp_4556[3];
 
    signal tmp_4557[3];
 
    signal tmp_4577[3];
 
    signal tmp_4580[3];
 
    signal tmp_4581[3];
 
    signal tmp_4600[3];
 
    signal tmp_4603[3];
 
    signal tmp_4604[3];
 
    signal tmp_4608[3];
 
    signal tmp_4629[3];
 
    signal tmp_4632[3];
 
    signal tmp_4633[3];
 
    signal tmp_4638[3];
 
    signal tmp_4639[3];
 
    signal tmp_4640[3];
 
    signal tmp_4641[3];
 
    signal tmp_4659[3];
 
    signal tmp_4662[3];
 
    signal tmp_4663[3];
 
    signal tmp_4683[3];
 
    signal tmp_4686[3];
 
    signal tmp_4687[3];
 
    signal tmp_4706[3];
 
    signal tmp_4709[3];
 
    signal tmp_4710[3];
 
    signal tmp_4714[3];
 
    signal tmp_4735[3];
 
    signal tmp_4738[3];
 
    signal tmp_4739[3];
 
    signal tmp_4744[3];
 
    signal tmp_4745[3];
 
    signal tmp_4746[3];
 
    signal tmp_4747[3];
 
    signal tmp_4765[3];
 
    signal tmp_4768[3];
 
    signal tmp_4769[3];
 
    signal tmp_4789[3];
 
    signal tmp_4792[3];
 
    signal tmp_4793[3];
 
    signal tmp_4812[3];
 
    signal tmp_4815[3];
 
    signal tmp_4816[3];
 
    signal tmp_4820[3];
 
    signal tmp_4841[3];
 
    signal tmp_4844[3];
 
    signal tmp_4845[3];
 
    signal tmp_4850[3];
 
    signal tmp_4851[3];
 
    signal tmp_4852[3];
 
    signal tmp_4853[3];
 
    signal tmp_4854[3];
 
    signal tmp_4859[3];
 
    signal tmp_4862[3];
 
    (tmp_3864,tmp_3866,tmp_3978,tmp_4448,tmp_4451,tmp_4452,tmp_4472,tmp_4475,tmp_4476,tmp_4495,tmp_4498,tmp_4499,tmp_4503,tmp_4524,tmp_4527,tmp_4528,tmp_4533,tmp_4534,tmp_4535,tmp_4553,tmp_4556,tmp_4557,tmp_4577,tmp_4580,tmp_4581,tmp_4600,tmp_4603,tmp_4604,tmp_4608,tmp_4629,tmp_4632,tmp_4633,tmp_4638,tmp_4639,tmp_4640,tmp_4641,tmp_4659,tmp_4662,tmp_4663,tmp_4683,tmp_4686,tmp_4687,tmp_4706,tmp_4709,tmp_4710,tmp_4714,tmp_4735,tmp_4738,tmp_4739,tmp_4744,tmp_4745,tmp_4746,tmp_4747,tmp_4765,tmp_4768,tmp_4769,tmp_4789,tmp_4792,tmp_4793,tmp_4812,tmp_4815,tmp_4816,tmp_4820,tmp_4841,tmp_4844,tmp_4845,tmp_4850,tmp_4851,tmp_4852,tmp_4853,tmp_4854,tmp_4859,tmp_4862) <== VerifyEvaluationsChunks0()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh);
    signal tmp_5812[3];
 
    signal tmp_5815[3];
 
    signal tmp_5818[3];
 
    signal tmp_5821[3];
 
    signal tmp_5824[3];
 
    signal tmp_5827[3];
 
    signal tmp_5830[3];
 
    signal tmp_5831[3];
 
    signal tmp_5836[3];
 
    signal tmp_5839[3];
 
    signal tmp_5841[3];
 
    signal tmp_5844[3];
 
    signal tmp_5847[3];
 
    signal tmp_5850[3];
 
    signal tmp_5853[3];
 
    signal tmp_5856[3];
 
    signal tmp_5859[3];
 
    signal tmp_5860[3];
 
    signal tmp_5861[3];
 
    (tmp_5812,tmp_5815,tmp_5818,tmp_5821,tmp_5824,tmp_5827,tmp_5830,tmp_5831,tmp_5836,tmp_5839,tmp_5841,tmp_5844,tmp_5847,tmp_5850,tmp_5853,tmp_5856,tmp_5859,tmp_5860,tmp_5861) <== VerifyEvaluationsChunks1()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_3864,tmp_3866,tmp_3978,tmp_4448,tmp_4452,tmp_4472,tmp_4476,tmp_4495,tmp_4499,tmp_4503,tmp_4524,tmp_4528,tmp_4533,tmp_4534,tmp_4535,tmp_4553,tmp_4557,tmp_4577,tmp_4581,tmp_4600,tmp_4604,tmp_4608,tmp_4629,tmp_4633,tmp_4638,tmp_4639,tmp_4640,tmp_4641,tmp_4659,tmp_4663,tmp_4683,tmp_4687,tmp_4706,tmp_4710,tmp_4714,tmp_4735,tmp_4739,tmp_4744,tmp_4745,tmp_4746,tmp_4747,tmp_4789,tmp_4793,tmp_4812,tmp_4816,tmp_4820,tmp_4841,tmp_4845,tmp_4850,tmp_4851,tmp_4852,tmp_4853,tmp_4854,tmp_4859,tmp_4862);
    signal tmp_5945[3];
 
    signal tmp_6004[3];
 
    signal tmp_6063[3];
 
    signal tmp_6122[3];
 
    signal tmp_6181[3];
 
    signal tmp_6240[3];
 
    signal tmp_6853[3];
 
    signal tmp_6858[3];
 
    signal tmp_6859[3];
 
    signal tmp_6860[3];
 
    (tmp_5945,tmp_6004,tmp_6063,tmp_6122,tmp_6181,tmp_6240,tmp_6853,tmp_6858,tmp_6859,tmp_6860) <== VerifyEvaluationsChunks2()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_4765,tmp_4769,tmp_5812,tmp_5815,tmp_5818,tmp_5821,tmp_5824,tmp_5827,tmp_5830,tmp_5831,tmp_5836,tmp_5839,tmp_5841,tmp_5844,tmp_5847,tmp_5850,tmp_5853,tmp_5856,tmp_5859,tmp_5860,tmp_5861);
    signal tmp_7700[3];
 
    (tmp_7700) <== VerifyEvaluationsChunks3()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_4451,tmp_4475,tmp_4498,tmp_4527,tmp_4556,tmp_4580,tmp_4603,tmp_4632,tmp_4662,tmp_4686,tmp_4709,tmp_4738,tmp_4768,tmp_4792,tmp_4815,tmp_4844,tmp_5945,tmp_6004,tmp_6063,tmp_6122,tmp_6181,tmp_6240,tmp_6853,tmp_6858,tmp_6859,tmp_6860);

    signal xAcc[4][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[3][3]; // Stores the evaluations of Q_i
    signal qAcc[4][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 4; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[173+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[17]);
            qStep[i-1] <== CMul()(xAcc[i], evals[173+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i
    enable * (tmp_7700[0] - qAcc[3][0]) === 0;
    enable * (tmp_7700[1] - qAcc[3][1]) === 0;
    enable * (tmp_7700[2] - qAcc[3][2]) === 0;
}


template CalculateFRIPolChunks0() {
    signal input challengesFRI[2][3];
    signal input evals[198][3];

 
    signal input cm1[52];
 
    signal input cm2[55];
    signal input cm3[12];
    signal input consts[58];

    signal input xDivXSubXi[6][3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
 
    mapValues.vals1 <== cm1;
 
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;


    signal output tmp_598[3];
 
    signal tmp_0[3] <== [consts[46] - evals[0][0], -evals[0][1], -evals[0][2]];
    signal tmp_1[3] <== CMul()(tmp_0, challengesFRI[1]);
    signal tmp_2[3] <== [consts[47] - evals[1][0], -evals[1][1], -evals[1][2]];
    signal tmp_3[3] <== [tmp_1[0] + tmp_2[0], tmp_1[1] + tmp_2[1], tmp_1[2] + tmp_2[2]];
    signal tmp_4[3] <== CMul()(tmp_3, challengesFRI[1]);
    signal tmp_5[3] <== [consts[48] - evals[2][0], -evals[2][1], -evals[2][2]];
    signal tmp_6[3] <== [tmp_4[0] + tmp_5[0], tmp_4[1] + tmp_5[1], tmp_4[2] + tmp_5[2]];
    signal tmp_7[3] <== CMul()(tmp_6, challengesFRI[1]);
    signal tmp_8[3] <== [mapValues.cm1_36 - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_9[3] <== [tmp_7[0] + tmp_8[0], tmp_7[1] + tmp_8[1], tmp_7[2] + tmp_8[2]];
    signal tmp_10[3] <== CMul()(tmp_9, challengesFRI[1]);
    signal tmp_11[3] <== [mapValues.cm1_37 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_12[3] <== [tmp_10[0] + tmp_11[0], tmp_10[1] + tmp_11[1], tmp_10[2] + tmp_11[2]];
    signal tmp_13[3] <== CMul()(tmp_12, challengesFRI[1]);
    signal tmp_14[3] <== [mapValues.cm1_38 - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_15[3] <== [tmp_13[0] + tmp_14[0], tmp_13[1] + tmp_14[1], tmp_13[2] + tmp_14[2]];
    signal tmp_16[3] <== CMul()(tmp_15, challengesFRI[1]);
    signal tmp_17[3] <== [mapValues.cm1_39 - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_18[3] <== [tmp_16[0] + tmp_17[0], tmp_16[1] + tmp_17[1], tmp_16[2] + tmp_17[2]];
    signal tmp_19[3] <== CMul()(tmp_18, challengesFRI[1]);
    signal tmp_20[3] <== [mapValues.cm1_40 - evals[7][0], -evals[7][1], -evals[7][2]];
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_21, challengesFRI[1]);
    signal tmp_23[3] <== [mapValues.cm1_41 - evals[8][0], -evals[8][1], -evals[8][2]];
    signal tmp_24[3] <== [tmp_22[0] + tmp_23[0], tmp_22[1] + tmp_23[1], tmp_22[2] + tmp_23[2]];
    signal tmp_25[3] <== CMul()(tmp_24, challengesFRI[1]);
    signal tmp_26[3] <== [mapValues.cm1_42 - evals[9][0], -evals[9][1], -evals[9][2]];
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_27, challengesFRI[1]);
    signal tmp_29[3] <== [mapValues.cm1_43 - evals[10][0], -evals[10][1], -evals[10][2]];
    signal tmp_30[3] <== [tmp_28[0] + tmp_29[0], tmp_28[1] + tmp_29[1], tmp_28[2] + tmp_29[2]];
    signal tmp_31[3] <== CMul()(tmp_30, challengesFRI[1]);
    signal tmp_32[3] <== [mapValues.cm1_44 - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_33[3] <== [tmp_31[0] + tmp_32[0], tmp_31[1] + tmp_32[1], tmp_31[2] + tmp_32[2]];
    signal tmp_34[3] <== CMul()(tmp_33, challengesFRI[1]);
    signal tmp_35[3] <== [mapValues.cm1_45 - evals[12][0], -evals[12][1], -evals[12][2]];
    signal tmp_36[3] <== [tmp_34[0] + tmp_35[0], tmp_34[1] + tmp_35[1], tmp_34[2] + tmp_35[2]];
    signal tmp_37[3] <== CMul()(tmp_36, challengesFRI[1]);
    signal tmp_38[3] <== [mapValues.cm1_46 - evals[13][0], -evals[13][1], -evals[13][2]];
    signal tmp_39[3] <== [tmp_37[0] + tmp_38[0], tmp_37[1] + tmp_38[1], tmp_37[2] + tmp_38[2]];
    signal tmp_40[3] <== CMul()(tmp_39, challengesFRI[1]);
    signal tmp_41[3] <== [mapValues.cm1_47 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_42[3] <== [tmp_40[0] + tmp_41[0], tmp_40[1] + tmp_41[1], tmp_40[2] + tmp_41[2]];
    signal tmp_43[3] <== CMul()(tmp_42, challengesFRI[1]);
    signal tmp_44[3] <== [mapValues.cm1_48 - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_45[3] <== [tmp_43[0] + tmp_44[0], tmp_43[1] + tmp_44[1], tmp_43[2] + tmp_44[2]];
    signal tmp_46[3] <== CMul()(tmp_45, challengesFRI[1]);
    signal tmp_47[3] <== [mapValues.cm1_49 - evals[16][0], -evals[16][1], -evals[16][2]];
    signal tmp_48[3] <== [tmp_46[0] + tmp_47[0], tmp_46[1] + tmp_47[1], tmp_46[2] + tmp_47[2]];
    signal tmp_49[3] <== CMul()(tmp_48, challengesFRI[1]);
    signal tmp_50[3] <== [mapValues.cm1_50 - evals[17][0], -evals[17][1], -evals[17][2]];
    signal tmp_51[3] <== [tmp_49[0] + tmp_50[0], tmp_49[1] + tmp_50[1], tmp_49[2] + tmp_50[2]];
    signal tmp_52[3] <== CMul()(tmp_51, challengesFRI[1]);
    signal tmp_53[3] <== [mapValues.cm1_51 - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_54[3] <== [tmp_52[0] + tmp_53[0], tmp_52[1] + tmp_53[1], tmp_52[2] + tmp_53[2]];
    signal tmp_55[3] <== CMul()(tmp_54, challengesFRI[1]);
    signal tmp_56[3] <== [mapValues.cm2_0[0] - evals[19][0], mapValues.cm2_0[1] - evals[19][1], mapValues.cm2_0[2] - evals[19][2]];
    signal tmp_57[3] <== [tmp_55[0] + tmp_56[0], tmp_55[1] + tmp_56[1], tmp_55[2] + tmp_56[2]];
    signal tmp_58[3] <== CMul()(tmp_57, xDivXSubXi[0]);
    signal tmp_59[3] <== CMul()(challengesFRI[0], tmp_58);
    signal tmp_60[3] <== [consts[46] - evals[20][0], -evals[20][1], -evals[20][2]];
    signal tmp_61[3] <== CMul()(tmp_60, challengesFRI[1]);
    signal tmp_62[3] <== [consts[47] - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_63[3] <== [tmp_61[0] + tmp_62[0], tmp_61[1] + tmp_62[1], tmp_61[2] + tmp_62[2]];
    signal tmp_64[3] <== CMul()(tmp_63, xDivXSubXi[1]);
    signal tmp_65[3] <== [tmp_59[0] + tmp_64[0], tmp_59[1] + tmp_64[1], tmp_59[2] + tmp_64[2]];
    signal tmp_66[3] <== CMul()(challengesFRI[0], tmp_65);
    signal tmp_67[3] <== [mapValues.cm1_18 - evals[22][0], -evals[22][1], -evals[22][2]];
    signal tmp_68[3] <== CMul()(tmp_67, challengesFRI[1]);
    signal tmp_69[3] <== [mapValues.cm1_19 - evals[23][0], -evals[23][1], -evals[23][2]];
    signal tmp_70[3] <== [tmp_68[0] + tmp_69[0], tmp_68[1] + tmp_69[1], tmp_68[2] + tmp_69[2]];
    signal tmp_71[3] <== CMul()(tmp_70, challengesFRI[1]);
    signal tmp_72[3] <== [mapValues.cm1_20 - evals[24][0], -evals[24][1], -evals[24][2]];
    signal tmp_73[3] <== [tmp_71[0] + tmp_72[0], tmp_71[1] + tmp_72[1], tmp_71[2] + tmp_72[2]];
    signal tmp_74[3] <== CMul()(tmp_73, challengesFRI[1]);
    signal tmp_75[3] <== [mapValues.cm1_21 - evals[25][0], -evals[25][1], -evals[25][2]];
    signal tmp_76[3] <== [tmp_74[0] + tmp_75[0], tmp_74[1] + tmp_75[1], tmp_74[2] + tmp_75[2]];
    signal tmp_77[3] <== CMul()(tmp_76, challengesFRI[1]);
    signal tmp_78[3] <== [mapValues.cm1_22 - evals[26][0], -evals[26][1], -evals[26][2]];
    signal tmp_79[3] <== [tmp_77[0] + tmp_78[0], tmp_77[1] + tmp_78[1], tmp_77[2] + tmp_78[2]];
    signal tmp_80[3] <== CMul()(tmp_79, challengesFRI[1]);
    signal tmp_81[3] <== [mapValues.cm1_23 - evals[27][0], -evals[27][1], -evals[27][2]];
    signal tmp_82[3] <== [tmp_80[0] + tmp_81[0], tmp_80[1] + tmp_81[1], tmp_80[2] + tmp_81[2]];
    signal tmp_83[3] <== CMul()(tmp_82, xDivXSubXi[2]);
    signal tmp_84[3] <== [tmp_66[0] + tmp_83[0], tmp_66[1] + tmp_83[1], tmp_66[2] + tmp_83[2]];
    signal tmp_85[3] <== CMul()(challengesFRI[0], tmp_84);
    signal tmp_86[3] <== [consts[0] - evals[28][0], -evals[28][1], -evals[28][2]];
    signal tmp_87[3] <== CMul()(tmp_86, challengesFRI[1]);
    signal tmp_88[3] <== [consts[1] - evals[29][0], -evals[29][1], -evals[29][2]];
    signal tmp_89[3] <== [tmp_87[0] + tmp_88[0], tmp_87[1] + tmp_88[1], tmp_87[2] + tmp_88[2]];
    signal tmp_90[3] <== CMul()(tmp_89, challengesFRI[1]);
    signal tmp_91[3] <== [consts[2] - evals[30][0], -evals[30][1], -evals[30][2]];
    signal tmp_92[3] <== [tmp_90[0] + tmp_91[0], tmp_90[1] + tmp_91[1], tmp_90[2] + tmp_91[2]];
    signal tmp_93[3] <== CMul()(tmp_92, challengesFRI[1]);
    signal tmp_94[3] <== [consts[3] - evals[31][0], -evals[31][1], -evals[31][2]];
    signal tmp_95[3] <== [tmp_93[0] + tmp_94[0], tmp_93[1] + tmp_94[1], tmp_93[2] + tmp_94[2]];
    signal tmp_96[3] <== CMul()(tmp_95, challengesFRI[1]);
    signal tmp_97[3] <== [consts[4] - evals[32][0], -evals[32][1], -evals[32][2]];
    signal tmp_98[3] <== [tmp_96[0] + tmp_97[0], tmp_96[1] + tmp_97[1], tmp_96[2] + tmp_97[2]];
    signal tmp_99[3] <== CMul()(tmp_98, challengesFRI[1]);
    signal tmp_100[3] <== [consts[5] - evals[33][0], -evals[33][1], -evals[33][2]];
    signal tmp_101[3] <== [tmp_99[0] + tmp_100[0], tmp_99[1] + tmp_100[1], tmp_99[2] + tmp_100[2]];
    signal tmp_102[3] <== CMul()(tmp_101, challengesFRI[1]);
    signal tmp_103[3] <== [consts[6] - evals[34][0], -evals[34][1], -evals[34][2]];
    signal tmp_104[3] <== [tmp_102[0] + tmp_103[0], tmp_102[1] + tmp_103[1], tmp_102[2] + tmp_103[2]];
    signal tmp_105[3] <== CMul()(tmp_104, challengesFRI[1]);
    signal tmp_106[3] <== [consts[7] - evals[35][0], -evals[35][1], -evals[35][2]];
    signal tmp_107[3] <== [tmp_105[0] + tmp_106[0], tmp_105[1] + tmp_106[1], tmp_105[2] + tmp_106[2]];
    signal tmp_108[3] <== CMul()(tmp_107, challengesFRI[1]);
    signal tmp_109[3] <== [consts[8] - evals[36][0], -evals[36][1], -evals[36][2]];
    signal tmp_110[3] <== [tmp_108[0] + tmp_109[0], tmp_108[1] + tmp_109[1], tmp_108[2] + tmp_109[2]];
    signal tmp_111[3] <== CMul()(tmp_110, challengesFRI[1]);
    signal tmp_112[3] <== [consts[9] - evals[37][0], -evals[37][1], -evals[37][2]];
    signal tmp_113[3] <== [tmp_111[0] + tmp_112[0], tmp_111[1] + tmp_112[1], tmp_111[2] + tmp_112[2]];
    signal tmp_114[3] <== CMul()(tmp_113, challengesFRI[1]);
    signal tmp_115[3] <== [consts[10] - evals[38][0], -evals[38][1], -evals[38][2]];
    signal tmp_116[3] <== [tmp_114[0] + tmp_115[0], tmp_114[1] + tmp_115[1], tmp_114[2] + tmp_115[2]];
    signal tmp_117[3] <== CMul()(tmp_116, challengesFRI[1]);
    signal tmp_118[3] <== [consts[11] - evals[39][0], -evals[39][1], -evals[39][2]];
    signal tmp_119[3] <== [tmp_117[0] + tmp_118[0], tmp_117[1] + tmp_118[1], tmp_117[2] + tmp_118[2]];
    signal tmp_120[3] <== CMul()(tmp_119, challengesFRI[1]);
    signal tmp_121[3] <== [consts[12] - evals[40][0], -evals[40][1], -evals[40][2]];
    signal tmp_122[3] <== [tmp_120[0] + tmp_121[0], tmp_120[1] + tmp_121[1], tmp_120[2] + tmp_121[2]];
    signal tmp_123[3] <== CMul()(tmp_122, challengesFRI[1]);
    signal tmp_124[3] <== [consts[13] - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_125[3] <== [tmp_123[0] + tmp_124[0], tmp_123[1] + tmp_124[1], tmp_123[2] + tmp_124[2]];
    signal tmp_126[3] <== CMul()(tmp_125, challengesFRI[1]);
    signal tmp_127[3] <== [consts[14] - evals[42][0], -evals[42][1], -evals[42][2]];
    signal tmp_128[3] <== [tmp_126[0] + tmp_127[0], tmp_126[1] + tmp_127[1], tmp_126[2] + tmp_127[2]];
    signal tmp_129[3] <== CMul()(tmp_128, challengesFRI[1]);
    signal tmp_130[3] <== [consts[15] - evals[43][0], -evals[43][1], -evals[43][2]];
    signal tmp_131[3] <== [tmp_129[0] + tmp_130[0], tmp_129[1] + tmp_130[1], tmp_129[2] + tmp_130[2]];
    signal tmp_132[3] <== CMul()(tmp_131, challengesFRI[1]);
    signal tmp_133[3] <== [consts[16] - evals[44][0], -evals[44][1], -evals[44][2]];
    signal tmp_134[3] <== [tmp_132[0] + tmp_133[0], tmp_132[1] + tmp_133[1], tmp_132[2] + tmp_133[2]];
    signal tmp_135[3] <== CMul()(tmp_134, challengesFRI[1]);
    signal tmp_136[3] <== [consts[17] - evals[45][0], -evals[45][1], -evals[45][2]];
    signal tmp_137[3] <== [tmp_135[0] + tmp_136[0], tmp_135[1] + tmp_136[1], tmp_135[2] + tmp_136[2]];
    signal tmp_138[3] <== CMul()(tmp_137, challengesFRI[1]);
    signal tmp_139[3] <== [consts[18] - evals[46][0], -evals[46][1], -evals[46][2]];
    signal tmp_140[3] <== [tmp_138[0] + tmp_139[0], tmp_138[1] + tmp_139[1], tmp_138[2] + tmp_139[2]];
    signal tmp_141[3] <== CMul()(tmp_140, challengesFRI[1]);
    signal tmp_142[3] <== [consts[19] - evals[47][0], -evals[47][1], -evals[47][2]];
    signal tmp_143[3] <== [tmp_141[0] + tmp_142[0], tmp_141[1] + tmp_142[1], tmp_141[2] + tmp_142[2]];
    signal tmp_144[3] <== CMul()(tmp_143, challengesFRI[1]);
    signal tmp_145[3] <== [consts[20] - evals[48][0], -evals[48][1], -evals[48][2]];
    signal tmp_146[3] <== [tmp_144[0] + tmp_145[0], tmp_144[1] + tmp_145[1], tmp_144[2] + tmp_145[2]];
    signal tmp_147[3] <== CMul()(tmp_146, challengesFRI[1]);
    signal tmp_148[3] <== [consts[21] - evals[49][0], -evals[49][1], -evals[49][2]];
    signal tmp_149[3] <== [tmp_147[0] + tmp_148[0], tmp_147[1] + tmp_148[1], tmp_147[2] + tmp_148[2]];
    signal tmp_150[3] <== CMul()(tmp_149, challengesFRI[1]);
    signal tmp_151[3] <== [consts[22] - evals[50][0], -evals[50][1], -evals[50][2]];
    signal tmp_152[3] <== [tmp_150[0] + tmp_151[0], tmp_150[1] + tmp_151[1], tmp_150[2] + tmp_151[2]];
    signal tmp_153[3] <== CMul()(tmp_152, challengesFRI[1]);
    signal tmp_154[3] <== [consts[23] - evals[51][0], -evals[51][1], -evals[51][2]];
    signal tmp_155[3] <== [tmp_153[0] + tmp_154[0], tmp_153[1] + tmp_154[1], tmp_153[2] + tmp_154[2]];
    signal tmp_156[3] <== CMul()(tmp_155, challengesFRI[1]);
    signal tmp_157[3] <== [consts[24] - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_158[3] <== [tmp_156[0] + tmp_157[0], tmp_156[1] + tmp_157[1], tmp_156[2] + tmp_157[2]];
    signal tmp_159[3] <== CMul()(tmp_158, challengesFRI[1]);
    signal tmp_160[3] <== [consts[25] - evals[53][0], -evals[53][1], -evals[53][2]];
    signal tmp_161[3] <== [tmp_159[0] + tmp_160[0], tmp_159[1] + tmp_160[1], tmp_159[2] + tmp_160[2]];
    signal tmp_162[3] <== CMul()(tmp_161, challengesFRI[1]);
    signal tmp_163[3] <== [consts[26] - evals[54][0], -evals[54][1], -evals[54][2]];
    signal tmp_164[3] <== [tmp_162[0] + tmp_163[0], tmp_162[1] + tmp_163[1], tmp_162[2] + tmp_163[2]];
    signal tmp_165[3] <== CMul()(tmp_164, challengesFRI[1]);
    signal tmp_166[3] <== [consts[27] - evals[55][0], -evals[55][1], -evals[55][2]];
    signal tmp_167[3] <== [tmp_165[0] + tmp_166[0], tmp_165[1] + tmp_166[1], tmp_165[2] + tmp_166[2]];
    signal tmp_168[3] <== CMul()(tmp_167, challengesFRI[1]);
    signal tmp_169[3] <== [consts[28] - evals[56][0], -evals[56][1], -evals[56][2]];
    signal tmp_170[3] <== [tmp_168[0] + tmp_169[0], tmp_168[1] + tmp_169[1], tmp_168[2] + tmp_169[2]];
    signal tmp_171[3] <== CMul()(tmp_170, challengesFRI[1]);
    signal tmp_172[3] <== [consts[29] - evals[57][0], -evals[57][1], -evals[57][2]];
    signal tmp_173[3] <== [tmp_171[0] + tmp_172[0], tmp_171[1] + tmp_172[1], tmp_171[2] + tmp_172[2]];
    signal tmp_174[3] <== CMul()(tmp_173, challengesFRI[1]);
    signal tmp_175[3] <== [consts[30] - evals[58][0], -evals[58][1], -evals[58][2]];
    signal tmp_176[3] <== [tmp_174[0] + tmp_175[0], tmp_174[1] + tmp_175[1], tmp_174[2] + tmp_175[2]];
    signal tmp_177[3] <== CMul()(tmp_176, challengesFRI[1]);
    signal tmp_178[3] <== [consts[31] - evals[59][0], -evals[59][1], -evals[59][2]];
    signal tmp_179[3] <== [tmp_177[0] + tmp_178[0], tmp_177[1] + tmp_178[1], tmp_177[2] + tmp_178[2]];
    signal tmp_180[3] <== CMul()(tmp_179, challengesFRI[1]);
    signal tmp_181[3] <== [consts[32] - evals[60][0], -evals[60][1], -evals[60][2]];
    signal tmp_182[3] <== [tmp_180[0] + tmp_181[0], tmp_180[1] + tmp_181[1], tmp_180[2] + tmp_181[2]];
    signal tmp_183[3] <== CMul()(tmp_182, challengesFRI[1]);
    signal tmp_184[3] <== [consts[33] - evals[61][0], -evals[61][1], -evals[61][2]];
    signal tmp_185[3] <== [tmp_183[0] + tmp_184[0], tmp_183[1] + tmp_184[1], tmp_183[2] + tmp_184[2]];
    signal tmp_186[3] <== CMul()(tmp_185, challengesFRI[1]);
    signal tmp_187[3] <== [consts[34] - evals[62][0], -evals[62][1], -evals[62][2]];
    signal tmp_188[3] <== [tmp_186[0] + tmp_187[0], tmp_186[1] + tmp_187[1], tmp_186[2] + tmp_187[2]];
    signal tmp_189[3] <== CMul()(tmp_188, challengesFRI[1]);
    signal tmp_190[3] <== [consts[35] - evals[63][0], -evals[63][1], -evals[63][2]];
    signal tmp_191[3] <== [tmp_189[0] + tmp_190[0], tmp_189[1] + tmp_190[1], tmp_189[2] + tmp_190[2]];
    signal tmp_192[3] <== CMul()(tmp_191, challengesFRI[1]);
    signal tmp_193[3] <== [consts[36] - evals[64][0], -evals[64][1], -evals[64][2]];
    signal tmp_194[3] <== [tmp_192[0] + tmp_193[0], tmp_192[1] + tmp_193[1], tmp_192[2] + tmp_193[2]];
    signal tmp_195[3] <== CMul()(tmp_194, challengesFRI[1]);
    signal tmp_196[3] <== [consts[37] - evals[65][0], -evals[65][1], -evals[65][2]];
    signal tmp_197[3] <== [tmp_195[0] + tmp_196[0], tmp_195[1] + tmp_196[1], tmp_195[2] + tmp_196[2]];
    signal tmp_198[3] <== CMul()(tmp_197, challengesFRI[1]);
    signal tmp_199[3] <== [consts[38] - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_200[3] <== [tmp_198[0] + tmp_199[0], tmp_198[1] + tmp_199[1], tmp_198[2] + tmp_199[2]];
    signal tmp_201[3] <== CMul()(tmp_200, challengesFRI[1]);
    signal tmp_202[3] <== [consts[39] - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_203[3] <== [tmp_201[0] + tmp_202[0], tmp_201[1] + tmp_202[1], tmp_201[2] + tmp_202[2]];
    signal tmp_204[3] <== CMul()(tmp_203, challengesFRI[1]);
    signal tmp_205[3] <== [consts[40] - evals[68][0], -evals[68][1], -evals[68][2]];
    signal tmp_206[3] <== [tmp_204[0] + tmp_205[0], tmp_204[1] + tmp_205[1], tmp_204[2] + tmp_205[2]];
    signal tmp_207[3] <== CMul()(tmp_206, challengesFRI[1]);
    signal tmp_208[3] <== [consts[41] - evals[69][0], -evals[69][1], -evals[69][2]];
    signal tmp_209[3] <== [tmp_207[0] + tmp_208[0], tmp_207[1] + tmp_208[1], tmp_207[2] + tmp_208[2]];
    signal tmp_210[3] <== CMul()(tmp_209, challengesFRI[1]);
    signal tmp_211[3] <== [consts[42] - evals[70][0], -evals[70][1], -evals[70][2]];
    signal tmp_212[3] <== [tmp_210[0] + tmp_211[0], tmp_210[1] + tmp_211[1], tmp_210[2] + tmp_211[2]];
    signal tmp_213[3] <== CMul()(tmp_212, challengesFRI[1]);
    signal tmp_214[3] <== [consts[43] - evals[71][0], -evals[71][1], -evals[71][2]];
    signal tmp_215[3] <== [tmp_213[0] + tmp_214[0], tmp_213[1] + tmp_214[1], tmp_213[2] + tmp_214[2]];
    signal tmp_216[3] <== CMul()(tmp_215, challengesFRI[1]);
    signal tmp_217[3] <== [consts[44] - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_218[3] <== [tmp_216[0] + tmp_217[0], tmp_216[1] + tmp_217[1], tmp_216[2] + tmp_217[2]];
    signal tmp_219[3] <== CMul()(tmp_218, challengesFRI[1]);
    signal tmp_220[3] <== [consts[45] - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_221[3] <== [tmp_219[0] + tmp_220[0], tmp_219[1] + tmp_220[1], tmp_219[2] + tmp_220[2]];
    signal tmp_222[3] <== CMul()(tmp_221, challengesFRI[1]);
    signal tmp_223[3] <== [consts[46] - evals[74][0], -evals[74][1], -evals[74][2]];
    signal tmp_224[3] <== [tmp_222[0] + tmp_223[0], tmp_222[1] + tmp_223[1], tmp_222[2] + tmp_223[2]];
    signal tmp_225[3] <== CMul()(tmp_224, challengesFRI[1]);
    signal tmp_226[3] <== [consts[47] - evals[75][0], -evals[75][1], -evals[75][2]];
    signal tmp_227[3] <== [tmp_225[0] + tmp_226[0], tmp_225[1] + tmp_226[1], tmp_225[2] + tmp_226[2]];
    signal tmp_228[3] <== CMul()(tmp_227, challengesFRI[1]);
    signal tmp_229[3] <== [consts[48] - evals[76][0], -evals[76][1], -evals[76][2]];
    signal tmp_230[3] <== [tmp_228[0] + tmp_229[0], tmp_228[1] + tmp_229[1], tmp_228[2] + tmp_229[2]];
    signal tmp_231[3] <== CMul()(tmp_230, challengesFRI[1]);
    signal tmp_232[3] <== [consts[49] - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_233[3] <== [tmp_231[0] + tmp_232[0], tmp_231[1] + tmp_232[1], tmp_231[2] + tmp_232[2]];
    signal tmp_234[3] <== CMul()(tmp_233, challengesFRI[1]);
    signal tmp_235[3] <== [consts[50] - evals[78][0], -evals[78][1], -evals[78][2]];
    signal tmp_236[3] <== [tmp_234[0] + tmp_235[0], tmp_234[1] + tmp_235[1], tmp_234[2] + tmp_235[2]];
    signal tmp_237[3] <== CMul()(tmp_236, challengesFRI[1]);
    signal tmp_238[3] <== [consts[51] - evals[79][0], -evals[79][1], -evals[79][2]];
    signal tmp_239[3] <== [tmp_237[0] + tmp_238[0], tmp_237[1] + tmp_238[1], tmp_237[2] + tmp_238[2]];
    signal tmp_240[3] <== CMul()(tmp_239, challengesFRI[1]);
    signal tmp_241[3] <== [consts[52] - evals[80][0], -evals[80][1], -evals[80][2]];
    signal tmp_242[3] <== [tmp_240[0] + tmp_241[0], tmp_240[1] + tmp_241[1], tmp_240[2] + tmp_241[2]];
    signal tmp_243[3] <== CMul()(tmp_242, challengesFRI[1]);
    signal tmp_244[3] <== [consts[53] - evals[81][0], -evals[81][1], -evals[81][2]];
    signal tmp_245[3] <== [tmp_243[0] + tmp_244[0], tmp_243[1] + tmp_244[1], tmp_243[2] + tmp_244[2]];
    signal tmp_246[3] <== CMul()(tmp_245, challengesFRI[1]);
    signal tmp_247[3] <== [consts[54] - evals[82][0], -evals[82][1], -evals[82][2]];
    signal tmp_248[3] <== [tmp_246[0] + tmp_247[0], tmp_246[1] + tmp_247[1], tmp_246[2] + tmp_247[2]];
    signal tmp_249[3] <== CMul()(tmp_248, challengesFRI[1]);
    signal tmp_250[3] <== [consts[55] - evals[83][0], -evals[83][1], -evals[83][2]];
    signal tmp_251[3] <== [tmp_249[0] + tmp_250[0], tmp_249[1] + tmp_250[1], tmp_249[2] + tmp_250[2]];
    signal tmp_252[3] <== CMul()(tmp_251, challengesFRI[1]);
    signal tmp_253[3] <== [consts[56] - evals[84][0], -evals[84][1], -evals[84][2]];
    signal tmp_254[3] <== [tmp_252[0] + tmp_253[0], tmp_252[1] + tmp_253[1], tmp_252[2] + tmp_253[2]];
    signal tmp_255[3] <== CMul()(tmp_254, challengesFRI[1]);
    signal tmp_256[3] <== [consts[57] - evals[85][0], -evals[85][1], -evals[85][2]];
    signal tmp_257[3] <== [tmp_255[0] + tmp_256[0], tmp_255[1] + tmp_256[1], tmp_255[2] + tmp_256[2]];
    signal tmp_258[3] <== CMul()(tmp_257, challengesFRI[1]);
    signal tmp_259[3] <== [mapValues.cm1_0 - evals[86][0], -evals[86][1], -evals[86][2]];
    signal tmp_260[3] <== [tmp_258[0] + tmp_259[0], tmp_258[1] + tmp_259[1], tmp_258[2] + tmp_259[2]];
    signal tmp_261[3] <== CMul()(tmp_260, challengesFRI[1]);
    signal tmp_262[3] <== [mapValues.cm1_1 - evals[87][0], -evals[87][1], -evals[87][2]];
    signal tmp_263[3] <== [tmp_261[0] + tmp_262[0], tmp_261[1] + tmp_262[1], tmp_261[2] + tmp_262[2]];
    signal tmp_264[3] <== CMul()(tmp_263, challengesFRI[1]);
    signal tmp_265[3] <== [mapValues.cm1_2 - evals[88][0], -evals[88][1], -evals[88][2]];
    signal tmp_266[3] <== [tmp_264[0] + tmp_265[0], tmp_264[1] + tmp_265[1], tmp_264[2] + tmp_265[2]];
    signal tmp_267[3] <== CMul()(tmp_266, challengesFRI[1]);
    signal tmp_268[3] <== [mapValues.cm1_3 - evals[89][0], -evals[89][1], -evals[89][2]];
    signal tmp_269[3] <== [tmp_267[0] + tmp_268[0], tmp_267[1] + tmp_268[1], tmp_267[2] + tmp_268[2]];
    signal tmp_270[3] <== CMul()(tmp_269, challengesFRI[1]);
    signal tmp_271[3] <== [mapValues.cm1_4 - evals[90][0], -evals[90][1], -evals[90][2]];
    signal tmp_272[3] <== [tmp_270[0] + tmp_271[0], tmp_270[1] + tmp_271[1], tmp_270[2] + tmp_271[2]];
    signal tmp_273[3] <== CMul()(tmp_272, challengesFRI[1]);
    signal tmp_274[3] <== [mapValues.cm1_5 - evals[91][0], -evals[91][1], -evals[91][2]];
    signal tmp_275[3] <== [tmp_273[0] + tmp_274[0], tmp_273[1] + tmp_274[1], tmp_273[2] + tmp_274[2]];
    signal tmp_276[3] <== CMul()(tmp_275, challengesFRI[1]);
    signal tmp_277[3] <== [mapValues.cm1_6 - evals[92][0], -evals[92][1], -evals[92][2]];
    signal tmp_278[3] <== [tmp_276[0] + tmp_277[0], tmp_276[1] + tmp_277[1], tmp_276[2] + tmp_277[2]];
    signal tmp_279[3] <== CMul()(tmp_278, challengesFRI[1]);
    signal tmp_280[3] <== [mapValues.cm1_7 - evals[93][0], -evals[93][1], -evals[93][2]];
    signal tmp_281[3] <== [tmp_279[0] + tmp_280[0], tmp_279[1] + tmp_280[1], tmp_279[2] + tmp_280[2]];
    signal tmp_282[3] <== CMul()(tmp_281, challengesFRI[1]);
    signal tmp_283[3] <== [mapValues.cm1_8 - evals[94][0], -evals[94][1], -evals[94][2]];
    signal tmp_284[3] <== [tmp_282[0] + tmp_283[0], tmp_282[1] + tmp_283[1], tmp_282[2] + tmp_283[2]];
    signal tmp_285[3] <== CMul()(tmp_284, challengesFRI[1]);
    signal tmp_286[3] <== [mapValues.cm1_9 - evals[95][0], -evals[95][1], -evals[95][2]];
    signal tmp_287[3] <== [tmp_285[0] + tmp_286[0], tmp_285[1] + tmp_286[1], tmp_285[2] + tmp_286[2]];
    signal tmp_288[3] <== CMul()(tmp_287, challengesFRI[1]);
    signal tmp_289[3] <== [mapValues.cm1_10 - evals[96][0], -evals[96][1], -evals[96][2]];
    signal tmp_290[3] <== [tmp_288[0] + tmp_289[0], tmp_288[1] + tmp_289[1], tmp_288[2] + tmp_289[2]];
    signal tmp_291[3] <== CMul()(tmp_290, challengesFRI[1]);
    signal tmp_292[3] <== [mapValues.cm1_11 - evals[97][0], -evals[97][1], -evals[97][2]];
    signal tmp_293[3] <== [tmp_291[0] + tmp_292[0], tmp_291[1] + tmp_292[1], tmp_291[2] + tmp_292[2]];
    signal tmp_294[3] <== CMul()(tmp_293, challengesFRI[1]);
    signal tmp_295[3] <== [mapValues.cm1_12 - evals[98][0], -evals[98][1], -evals[98][2]];
    signal tmp_296[3] <== [tmp_294[0] + tmp_295[0], tmp_294[1] + tmp_295[1], tmp_294[2] + tmp_295[2]];
    signal tmp_297[3] <== CMul()(tmp_296, challengesFRI[1]);
    signal tmp_298[3] <== [mapValues.cm1_13 - evals[99][0], -evals[99][1], -evals[99][2]];
    signal tmp_299[3] <== [tmp_297[0] + tmp_298[0], tmp_297[1] + tmp_298[1], tmp_297[2] + tmp_298[2]];
    signal tmp_300[3] <== CMul()(tmp_299, challengesFRI[1]);
    signal tmp_301[3] <== [mapValues.cm1_14 - evals[100][0], -evals[100][1], -evals[100][2]];
    signal tmp_302[3] <== [tmp_300[0] + tmp_301[0], tmp_300[1] + tmp_301[1], tmp_300[2] + tmp_301[2]];
    signal tmp_303[3] <== CMul()(tmp_302, challengesFRI[1]);
    signal tmp_304[3] <== [mapValues.cm1_15 - evals[101][0], -evals[101][1], -evals[101][2]];
    signal tmp_305[3] <== [tmp_303[0] + tmp_304[0], tmp_303[1] + tmp_304[1], tmp_303[2] + tmp_304[2]];
    signal tmp_306[3] <== CMul()(tmp_305, challengesFRI[1]);
    signal tmp_307[3] <== [mapValues.cm1_16 - evals[102][0], -evals[102][1], -evals[102][2]];
    signal tmp_308[3] <== [tmp_306[0] + tmp_307[0], tmp_306[1] + tmp_307[1], tmp_306[2] + tmp_307[2]];
    signal tmp_309[3] <== CMul()(tmp_308, challengesFRI[1]);
    signal tmp_310[3] <== [mapValues.cm1_17 - evals[103][0], -evals[103][1], -evals[103][2]];
    signal tmp_311[3] <== [tmp_309[0] + tmp_310[0], tmp_309[1] + tmp_310[1], tmp_309[2] + tmp_310[2]];
    signal tmp_312[3] <== CMul()(tmp_311, challengesFRI[1]);
    signal tmp_313[3] <== [mapValues.cm1_18 - evals[104][0], -evals[104][1], -evals[104][2]];
    signal tmp_314[3] <== [tmp_312[0] + tmp_313[0], tmp_312[1] + tmp_313[1], tmp_312[2] + tmp_313[2]];
    signal tmp_315[3] <== CMul()(tmp_314, challengesFRI[1]);
    signal tmp_316[3] <== [mapValues.cm1_19 - evals[105][0], -evals[105][1], -evals[105][2]];
    signal tmp_317[3] <== [tmp_315[0] + tmp_316[0], tmp_315[1] + tmp_316[1], tmp_315[2] + tmp_316[2]];
    signal tmp_318[3] <== CMul()(tmp_317, challengesFRI[1]);
    signal tmp_319[3] <== [mapValues.cm1_20 - evals[106][0], -evals[106][1], -evals[106][2]];
    signal tmp_320[3] <== [tmp_318[0] + tmp_319[0], tmp_318[1] + tmp_319[1], tmp_318[2] + tmp_319[2]];
    signal tmp_321[3] <== CMul()(tmp_320, challengesFRI[1]);
    signal tmp_322[3] <== [mapValues.cm1_21 - evals[107][0], -evals[107][1], -evals[107][2]];
    signal tmp_323[3] <== [tmp_321[0] + tmp_322[0], tmp_321[1] + tmp_322[1], tmp_321[2] + tmp_322[2]];
    signal tmp_324[3] <== CMul()(tmp_323, challengesFRI[1]);
    signal tmp_325[3] <== [mapValues.cm1_22 - evals[108][0], -evals[108][1], -evals[108][2]];
    signal tmp_326[3] <== [tmp_324[0] + tmp_325[0], tmp_324[1] + tmp_325[1], tmp_324[2] + tmp_325[2]];
    signal tmp_327[3] <== CMul()(tmp_326, challengesFRI[1]);
    signal tmp_328[3] <== [mapValues.cm1_23 - evals[109][0], -evals[109][1], -evals[109][2]];
    signal tmp_329[3] <== [tmp_327[0] + tmp_328[0], tmp_327[1] + tmp_328[1], tmp_327[2] + tmp_328[2]];
    signal tmp_330[3] <== CMul()(tmp_329, challengesFRI[1]);
    signal tmp_331[3] <== [mapValues.cm1_24 - evals[110][0], -evals[110][1], -evals[110][2]];
    signal tmp_332[3] <== [tmp_330[0] + tmp_331[0], tmp_330[1] + tmp_331[1], tmp_330[2] + tmp_331[2]];
    signal tmp_333[3] <== CMul()(tmp_332, challengesFRI[1]);
    signal tmp_334[3] <== [mapValues.cm1_25 - evals[111][0], -evals[111][1], -evals[111][2]];
    signal tmp_335[3] <== [tmp_333[0] + tmp_334[0], tmp_333[1] + tmp_334[1], tmp_333[2] + tmp_334[2]];
    signal tmp_336[3] <== CMul()(tmp_335, challengesFRI[1]);
    signal tmp_337[3] <== [mapValues.cm1_26 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_338[3] <== [tmp_336[0] + tmp_337[0], tmp_336[1] + tmp_337[1], tmp_336[2] + tmp_337[2]];
    signal tmp_339[3] <== CMul()(tmp_338, challengesFRI[1]);
    signal tmp_340[3] <== [mapValues.cm1_27 - evals[113][0], -evals[113][1], -evals[113][2]];
    signal tmp_341[3] <== [tmp_339[0] + tmp_340[0], tmp_339[1] + tmp_340[1], tmp_339[2] + tmp_340[2]];
    signal tmp_342[3] <== CMul()(tmp_341, challengesFRI[1]);
    signal tmp_343[3] <== [mapValues.cm1_28 - evals[114][0], -evals[114][1], -evals[114][2]];
    signal tmp_344[3] <== [tmp_342[0] + tmp_343[0], tmp_342[1] + tmp_343[1], tmp_342[2] + tmp_343[2]];
    signal tmp_345[3] <== CMul()(tmp_344, challengesFRI[1]);
    signal tmp_346[3] <== [mapValues.cm1_29 - evals[115][0], -evals[115][1], -evals[115][2]];
    signal tmp_347[3] <== [tmp_345[0] + tmp_346[0], tmp_345[1] + tmp_346[1], tmp_345[2] + tmp_346[2]];
    signal tmp_348[3] <== CMul()(tmp_347, challengesFRI[1]);
    signal tmp_349[3] <== [mapValues.cm1_30 - evals[116][0], -evals[116][1], -evals[116][2]];
    signal tmp_350[3] <== [tmp_348[0] + tmp_349[0], tmp_348[1] + tmp_349[1], tmp_348[2] + tmp_349[2]];
    signal tmp_351[3] <== CMul()(tmp_350, challengesFRI[1]);
    signal tmp_352[3] <== [mapValues.cm1_31 - evals[117][0], -evals[117][1], -evals[117][2]];
    signal tmp_353[3] <== [tmp_351[0] + tmp_352[0], tmp_351[1] + tmp_352[1], tmp_351[2] + tmp_352[2]];
    signal tmp_354[3] <== CMul()(tmp_353, challengesFRI[1]);
    signal tmp_355[3] <== [mapValues.cm1_32 - evals[118][0], -evals[118][1], -evals[118][2]];
    signal tmp_356[3] <== [tmp_354[0] + tmp_355[0], tmp_354[1] + tmp_355[1], tmp_354[2] + tmp_355[2]];
    signal tmp_357[3] <== CMul()(tmp_356, challengesFRI[1]);
    signal tmp_358[3] <== [mapValues.cm1_33 - evals[119][0], -evals[119][1], -evals[119][2]];
    signal tmp_359[3] <== [tmp_357[0] + tmp_358[0], tmp_357[1] + tmp_358[1], tmp_357[2] + tmp_358[2]];
    signal tmp_360[3] <== CMul()(tmp_359, challengesFRI[1]);
    signal tmp_361[3] <== [mapValues.cm1_34 - evals[120][0], -evals[120][1], -evals[120][2]];
    signal tmp_362[3] <== [tmp_360[0] + tmp_361[0], tmp_360[1] + tmp_361[1], tmp_360[2] + tmp_361[2]];
    signal tmp_363[3] <== CMul()(tmp_362, challengesFRI[1]);
    signal tmp_364[3] <== [mapValues.cm1_35 - evals[121][0], -evals[121][1], -evals[121][2]];
    signal tmp_365[3] <== [tmp_363[0] + tmp_364[0], tmp_363[1] + tmp_364[1], tmp_363[2] + tmp_364[2]];
    signal tmp_366[3] <== CMul()(tmp_365, challengesFRI[1]);
    signal tmp_367[3] <== [mapValues.cm1_36 - evals[122][0], -evals[122][1], -evals[122][2]];
    signal tmp_368[3] <== [tmp_366[0] + tmp_367[0], tmp_366[1] + tmp_367[1], tmp_366[2] + tmp_367[2]];
    signal tmp_369[3] <== CMul()(tmp_368, challengesFRI[1]);
    signal tmp_370[3] <== [mapValues.cm1_37 - evals[123][0], -evals[123][1], -evals[123][2]];
    signal tmp_371[3] <== [tmp_369[0] + tmp_370[0], tmp_369[1] + tmp_370[1], tmp_369[2] + tmp_370[2]];
    signal tmp_372[3] <== CMul()(tmp_371, challengesFRI[1]);
    signal tmp_373[3] <== [mapValues.cm1_38 - evals[124][0], -evals[124][1], -evals[124][2]];
    signal tmp_374[3] <== [tmp_372[0] + tmp_373[0], tmp_372[1] + tmp_373[1], tmp_372[2] + tmp_373[2]];
    signal tmp_375[3] <== CMul()(tmp_374, challengesFRI[1]);
    signal tmp_376[3] <== [mapValues.cm1_39 - evals[125][0], -evals[125][1], -evals[125][2]];
    signal tmp_377[3] <== [tmp_375[0] + tmp_376[0], tmp_375[1] + tmp_376[1], tmp_375[2] + tmp_376[2]];
    signal tmp_378[3] <== CMul()(tmp_377, challengesFRI[1]);
    signal tmp_379[3] <== [mapValues.cm1_40 - evals[126][0], -evals[126][1], -evals[126][2]];
    signal tmp_380[3] <== [tmp_378[0] + tmp_379[0], tmp_378[1] + tmp_379[1], tmp_378[2] + tmp_379[2]];
    signal tmp_381[3] <== CMul()(tmp_380, challengesFRI[1]);
    signal tmp_382[3] <== [mapValues.cm1_41 - evals[127][0], -evals[127][1], -evals[127][2]];
    signal tmp_383[3] <== [tmp_381[0] + tmp_382[0], tmp_381[1] + tmp_382[1], tmp_381[2] + tmp_382[2]];
    signal tmp_384[3] <== CMul()(tmp_383, challengesFRI[1]);
    signal tmp_385[3] <== [mapValues.cm1_42 - evals[128][0], -evals[128][1], -evals[128][2]];
    signal tmp_386[3] <== [tmp_384[0] + tmp_385[0], tmp_384[1] + tmp_385[1], tmp_384[2] + tmp_385[2]];
    signal tmp_387[3] <== CMul()(tmp_386, challengesFRI[1]);
    signal tmp_388[3] <== [mapValues.cm1_43 - evals[129][0], -evals[129][1], -evals[129][2]];
    signal tmp_389[3] <== [tmp_387[0] + tmp_388[0], tmp_387[1] + tmp_388[1], tmp_387[2] + tmp_388[2]];
    signal tmp_390[3] <== CMul()(tmp_389, challengesFRI[1]);
    signal tmp_391[3] <== [mapValues.cm1_44 - evals[130][0], -evals[130][1], -evals[130][2]];
    signal tmp_392[3] <== [tmp_390[0] + tmp_391[0], tmp_390[1] + tmp_391[1], tmp_390[2] + tmp_391[2]];
    signal tmp_393[3] <== CMul()(tmp_392, challengesFRI[1]);
    signal tmp_394[3] <== [mapValues.cm1_45 - evals[131][0], -evals[131][1], -evals[131][2]];
    signal tmp_395[3] <== [tmp_393[0] + tmp_394[0], tmp_393[1] + tmp_394[1], tmp_393[2] + tmp_394[2]];
    signal tmp_396[3] <== CMul()(tmp_395, challengesFRI[1]);
    signal tmp_397[3] <== [mapValues.cm1_46 - evals[132][0], -evals[132][1], -evals[132][2]];
    signal tmp_398[3] <== [tmp_396[0] + tmp_397[0], tmp_396[1] + tmp_397[1], tmp_396[2] + tmp_397[2]];
    signal tmp_399[3] <== CMul()(tmp_398, challengesFRI[1]);
    signal tmp_400[3] <== [mapValues.cm1_47 - evals[133][0], -evals[133][1], -evals[133][2]];
    signal tmp_401[3] <== [tmp_399[0] + tmp_400[0], tmp_399[1] + tmp_400[1], tmp_399[2] + tmp_400[2]];
    signal tmp_402[3] <== CMul()(tmp_401, challengesFRI[1]);
    signal tmp_403[3] <== [mapValues.cm1_48 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_404[3] <== [tmp_402[0] + tmp_403[0], tmp_402[1] + tmp_403[1], tmp_402[2] + tmp_403[2]];
    signal tmp_405[3] <== CMul()(tmp_404, challengesFRI[1]);
    signal tmp_406[3] <== [mapValues.cm1_49 - evals[135][0], -evals[135][1], -evals[135][2]];
    signal tmp_407[3] <== [tmp_405[0] + tmp_406[0], tmp_405[1] + tmp_406[1], tmp_405[2] + tmp_406[2]];
    signal tmp_408[3] <== CMul()(tmp_407, challengesFRI[1]);
    signal tmp_409[3] <== [mapValues.cm1_50 - evals[136][0], -evals[136][1], -evals[136][2]];
    signal tmp_410[3] <== [tmp_408[0] + tmp_409[0], tmp_408[1] + tmp_409[1], tmp_408[2] + tmp_409[2]];
    signal tmp_411[3] <== CMul()(tmp_410, challengesFRI[1]);
    signal tmp_412[3] <== [mapValues.cm1_51 - evals[137][0], -evals[137][1], -evals[137][2]];
    signal tmp_413[3] <== [tmp_411[0] + tmp_412[0], tmp_411[1] + tmp_412[1], tmp_411[2] + tmp_412[2]];
    signal tmp_414[3] <== CMul()(tmp_413, challengesFRI[1]);
    signal tmp_415[3] <== [mapValues.cm2_0[0] - evals[138][0], mapValues.cm2_0[1] - evals[138][1], mapValues.cm2_0[2] - evals[138][2]];
    signal tmp_416[3] <== [tmp_414[0] + tmp_415[0], tmp_414[1] + tmp_415[1], tmp_414[2] + tmp_415[2]];
    signal tmp_417[3] <== CMul()(tmp_416, challengesFRI[1]);
    signal tmp_418[3] <== [mapValues.cm2_1[0] - evals[139][0], mapValues.cm2_1[1] - evals[139][1], mapValues.cm2_1[2] - evals[139][2]];
    signal tmp_419[3] <== [tmp_417[0] + tmp_418[0], tmp_417[1] + tmp_418[1], tmp_417[2] + tmp_418[2]];
    signal tmp_420[3] <== CMul()(tmp_419, challengesFRI[1]);
    signal tmp_421[3] <== [mapValues.cm2_2[0] - evals[140][0], mapValues.cm2_2[1] - evals[140][1], mapValues.cm2_2[2] - evals[140][2]];
    signal tmp_422[3] <== [tmp_420[0] + tmp_421[0], tmp_420[1] + tmp_421[1], tmp_420[2] + tmp_421[2]];
    signal tmp_423[3] <== CMul()(tmp_422, challengesFRI[1]);
    signal tmp_424[3] <== [mapValues.cm2_3[0] - evals[141][0], mapValues.cm2_3[1] - evals[141][1], mapValues.cm2_3[2] - evals[141][2]];
    signal tmp_425[3] <== [tmp_423[0] + tmp_424[0], tmp_423[1] + tmp_424[1], tmp_423[2] + tmp_424[2]];
    signal tmp_426[3] <== CMul()(tmp_425, challengesFRI[1]);
    signal tmp_427[3] <== [mapValues.cm2_4[0] - evals[142][0], mapValues.cm2_4[1] - evals[142][1], mapValues.cm2_4[2] - evals[142][2]];
    signal tmp_428[3] <== [tmp_426[0] + tmp_427[0], tmp_426[1] + tmp_427[1], tmp_426[2] + tmp_427[2]];
    signal tmp_429[3] <== CMul()(tmp_428, challengesFRI[1]);
    signal tmp_430[3] <== [mapValues.cm2_5[0] - evals[143][0], mapValues.cm2_5[1] - evals[143][1], mapValues.cm2_5[2] - evals[143][2]];
    signal tmp_431[3] <== [tmp_429[0] + tmp_430[0], tmp_429[1] + tmp_430[1], tmp_429[2] + tmp_430[2]];
    signal tmp_432[3] <== CMul()(tmp_431, challengesFRI[1]);
    signal tmp_433[3] <== [mapValues.cm2_6[0] - evals[144][0], mapValues.cm2_6[1] - evals[144][1], mapValues.cm2_6[2] - evals[144][2]];
    signal tmp_434[3] <== [tmp_432[0] + tmp_433[0], tmp_432[1] + tmp_433[1], tmp_432[2] + tmp_433[2]];
    signal tmp_435[3] <== CMul()(tmp_434, challengesFRI[1]);
    signal tmp_436[3] <== [mapValues.cm2_7[0] - evals[145][0], mapValues.cm2_7[1] - evals[145][1], mapValues.cm2_7[2] - evals[145][2]];
    signal tmp_437[3] <== [tmp_435[0] + tmp_436[0], tmp_435[1] + tmp_436[1], tmp_435[2] + tmp_436[2]];
    signal tmp_438[3] <== CMul()(tmp_437, challengesFRI[1]);
    signal tmp_439[3] <== [mapValues.cm2_8[0] - evals[146][0], mapValues.cm2_8[1] - evals[146][1], mapValues.cm2_8[2] - evals[146][2]];
    signal tmp_440[3] <== [tmp_438[0] + tmp_439[0], tmp_438[1] + tmp_439[1], tmp_438[2] + tmp_439[2]];
    signal tmp_441[3] <== CMul()(tmp_440, challengesFRI[1]);
    signal tmp_442[3] <== [mapValues.cm2_9 - evals[147][0], -evals[147][1], -evals[147][2]];
    signal tmp_443[3] <== [tmp_441[0] + tmp_442[0], tmp_441[1] + tmp_442[1], tmp_441[2] + tmp_442[2]];
    signal tmp_444[3] <== CMul()(tmp_443, challengesFRI[1]);
    signal tmp_445[3] <== [mapValues.cm2_10 - evals[148][0], -evals[148][1], -evals[148][2]];
    signal tmp_446[3] <== [tmp_444[0] + tmp_445[0], tmp_444[1] + tmp_445[1], tmp_444[2] + tmp_445[2]];
    signal tmp_447[3] <== CMul()(tmp_446, challengesFRI[1]);
    signal tmp_448[3] <== [mapValues.cm2_11 - evals[149][0], -evals[149][1], -evals[149][2]];
    signal tmp_449[3] <== [tmp_447[0] + tmp_448[0], tmp_447[1] + tmp_448[1], tmp_447[2] + tmp_448[2]];
    signal tmp_450[3] <== CMul()(tmp_449, challengesFRI[1]);
    signal tmp_451[3] <== [mapValues.cm2_12 - evals[150][0], -evals[150][1], -evals[150][2]];
    signal tmp_452[3] <== [tmp_450[0] + tmp_451[0], tmp_450[1] + tmp_451[1], tmp_450[2] + tmp_451[2]];
    signal tmp_453[3] <== CMul()(tmp_452, challengesFRI[1]);
    signal tmp_454[3] <== [mapValues.cm2_13 - evals[151][0], -evals[151][1], -evals[151][2]];
    signal tmp_455[3] <== [tmp_453[0] + tmp_454[0], tmp_453[1] + tmp_454[1], tmp_453[2] + tmp_454[2]];
    signal tmp_456[3] <== CMul()(tmp_455, challengesFRI[1]);
    signal tmp_457[3] <== [mapValues.cm2_14 - evals[152][0], -evals[152][1], -evals[152][2]];
    signal tmp_458[3] <== [tmp_456[0] + tmp_457[0], tmp_456[1] + tmp_457[1], tmp_456[2] + tmp_457[2]];
    signal tmp_459[3] <== CMul()(tmp_458, challengesFRI[1]);
    signal tmp_460[3] <== [mapValues.cm2_15 - evals[153][0], -evals[153][1], -evals[153][2]];
    signal tmp_461[3] <== [tmp_459[0] + tmp_460[0], tmp_459[1] + tmp_460[1], tmp_459[2] + tmp_460[2]];
    signal tmp_462[3] <== CMul()(tmp_461, challengesFRI[1]);
    signal tmp_463[3] <== [mapValues.cm2_16 - evals[154][0], -evals[154][1], -evals[154][2]];
    signal tmp_464[3] <== [tmp_462[0] + tmp_463[0], tmp_462[1] + tmp_463[1], tmp_462[2] + tmp_463[2]];
    signal tmp_465[3] <== CMul()(tmp_464, challengesFRI[1]);
    signal tmp_466[3] <== [mapValues.cm2_17 - evals[155][0], -evals[155][1], -evals[155][2]];
    signal tmp_467[3] <== [tmp_465[0] + tmp_466[0], tmp_465[1] + tmp_466[1], tmp_465[2] + tmp_466[2]];
    signal tmp_468[3] <== CMul()(tmp_467, challengesFRI[1]);
    signal tmp_469[3] <== [mapValues.cm2_18 - evals[156][0], -evals[156][1], -evals[156][2]];
    signal tmp_470[3] <== [tmp_468[0] + tmp_469[0], tmp_468[1] + tmp_469[1], tmp_468[2] + tmp_469[2]];
    signal tmp_471[3] <== CMul()(tmp_470, challengesFRI[1]);
    signal tmp_472[3] <== [mapValues.cm2_19 - evals[157][0], -evals[157][1], -evals[157][2]];
    signal tmp_473[3] <== [tmp_471[0] + tmp_472[0], tmp_471[1] + tmp_472[1], tmp_471[2] + tmp_472[2]];
    signal tmp_474[3] <== CMul()(tmp_473, challengesFRI[1]);
    signal tmp_475[3] <== [mapValues.cm2_20 - evals[158][0], -evals[158][1], -evals[158][2]];
    signal tmp_476[3] <== [tmp_474[0] + tmp_475[0], tmp_474[1] + tmp_475[1], tmp_474[2] + tmp_475[2]];
    signal tmp_477[3] <== CMul()(tmp_476, challengesFRI[1]);
    signal tmp_478[3] <== [mapValues.cm2_21 - evals[159][0], -evals[159][1], -evals[159][2]];
    signal tmp_479[3] <== [tmp_477[0] + tmp_478[0], tmp_477[1] + tmp_478[1], tmp_477[2] + tmp_478[2]];
    signal tmp_480[3] <== CMul()(tmp_479, challengesFRI[1]);
    signal tmp_481[3] <== [mapValues.cm2_22 - evals[160][0], -evals[160][1], -evals[160][2]];
    signal tmp_482[3] <== [tmp_480[0] + tmp_481[0], tmp_480[1] + tmp_481[1], tmp_480[2] + tmp_481[2]];
    signal tmp_483[3] <== CMul()(tmp_482, challengesFRI[1]);
    signal tmp_484[3] <== [mapValues.cm2_23 - evals[161][0], -evals[161][1], -evals[161][2]];
    signal tmp_485[3] <== [tmp_483[0] + tmp_484[0], tmp_483[1] + tmp_484[1], tmp_483[2] + tmp_484[2]];
    signal tmp_486[3] <== CMul()(tmp_485, challengesFRI[1]);
    signal tmp_487[3] <== [mapValues.cm2_24 - evals[162][0], -evals[162][1], -evals[162][2]];
    signal tmp_488[3] <== [tmp_486[0] + tmp_487[0], tmp_486[1] + tmp_487[1], tmp_486[2] + tmp_487[2]];
    signal tmp_489[3] <== CMul()(tmp_488, challengesFRI[1]);
    signal tmp_490[3] <== [mapValues.cm2_25 - evals[163][0], -evals[163][1], -evals[163][2]];
    signal tmp_491[3] <== [tmp_489[0] + tmp_490[0], tmp_489[1] + tmp_490[1], tmp_489[2] + tmp_490[2]];
    signal tmp_492[3] <== CMul()(tmp_491, challengesFRI[1]);
    signal tmp_493[3] <== [mapValues.cm2_26 - evals[164][0], -evals[164][1], -evals[164][2]];
    signal tmp_494[3] <== [tmp_492[0] + tmp_493[0], tmp_492[1] + tmp_493[1], tmp_492[2] + tmp_493[2]];
    signal tmp_495[3] <== CMul()(tmp_494, challengesFRI[1]);
    signal tmp_496[3] <== [mapValues.cm2_27 - evals[165][0], -evals[165][1], -evals[165][2]];
    signal tmp_497[3] <== [tmp_495[0] + tmp_496[0], tmp_495[1] + tmp_496[1], tmp_495[2] + tmp_496[2]];
    signal tmp_498[3] <== CMul()(tmp_497, challengesFRI[1]);
    signal tmp_499[3] <== [mapValues.cm2_28 - evals[166][0], -evals[166][1], -evals[166][2]];
    signal tmp_500[3] <== [tmp_498[0] + tmp_499[0], tmp_498[1] + tmp_499[1], tmp_498[2] + tmp_499[2]];
    signal tmp_501[3] <== CMul()(tmp_500, challengesFRI[1]);
    signal tmp_502[3] <== [mapValues.cm2_29 - evals[167][0], -evals[167][1], -evals[167][2]];
    signal tmp_503[3] <== [tmp_501[0] + tmp_502[0], tmp_501[1] + tmp_502[1], tmp_501[2] + tmp_502[2]];
    signal tmp_504[3] <== CMul()(tmp_503, challengesFRI[1]);
    signal tmp_505[3] <== [mapValues.cm2_30 - evals[168][0], -evals[168][1], -evals[168][2]];
    signal tmp_506[3] <== [tmp_504[0] + tmp_505[0], tmp_504[1] + tmp_505[1], tmp_504[2] + tmp_505[2]];
    signal tmp_507[3] <== CMul()(tmp_506, challengesFRI[1]);
    signal tmp_508[3] <== [mapValues.cm2_31 - evals[169][0], -evals[169][1], -evals[169][2]];
    signal tmp_509[3] <== [tmp_507[0] + tmp_508[0], tmp_507[1] + tmp_508[1], tmp_507[2] + tmp_508[2]];
    signal tmp_510[3] <== CMul()(tmp_509, challengesFRI[1]);
    signal tmp_511[3] <== [mapValues.cm2_32 - evals[170][0], -evals[170][1], -evals[170][2]];
    signal tmp_512[3] <== [tmp_510[0] + tmp_511[0], tmp_510[1] + tmp_511[1], tmp_510[2] + tmp_511[2]];
    signal tmp_513[3] <== CMul()(tmp_512, challengesFRI[1]);
    signal tmp_514[3] <== [mapValues.cm2_33 - evals[171][0], -evals[171][1], -evals[171][2]];
    signal tmp_515[3] <== [tmp_513[0] + tmp_514[0], tmp_513[1] + tmp_514[1], tmp_513[2] + tmp_514[2]];
    signal tmp_516[3] <== CMul()(tmp_515, challengesFRI[1]);
    signal tmp_517[3] <== [mapValues.cm2_34[0] - evals[172][0], mapValues.cm2_34[1] - evals[172][1], mapValues.cm2_34[2] - evals[172][2]];
    signal tmp_518[3] <== [tmp_516[0] + tmp_517[0], tmp_516[1] + tmp_517[1], tmp_516[2] + tmp_517[2]];
    signal tmp_519[3] <== CMul()(tmp_518, challengesFRI[1]);
    signal tmp_520[3] <== [mapValues.cm3_0[0] - evals[173][0], mapValues.cm3_0[1] - evals[173][1], mapValues.cm3_0[2] - evals[173][2]];
    signal tmp_521[3] <== [tmp_519[0] + tmp_520[0], tmp_519[1] + tmp_520[1], tmp_519[2] + tmp_520[2]];
    signal tmp_522[3] <== CMul()(tmp_521, challengesFRI[1]);
    signal tmp_523[3] <== [mapValues.cm3_1[0] - evals[174][0], mapValues.cm3_1[1] - evals[174][1], mapValues.cm3_1[2] - evals[174][2]];
    signal tmp_524[3] <== [tmp_522[0] + tmp_523[0], tmp_522[1] + tmp_523[1], tmp_522[2] + tmp_523[2]];
    signal tmp_525[3] <== CMul()(tmp_524, challengesFRI[1]);
    signal tmp_526[3] <== [mapValues.cm3_2[0] - evals[175][0], mapValues.cm3_2[1] - evals[175][1], mapValues.cm3_2[2] - evals[175][2]];
    signal tmp_527[3] <== [tmp_525[0] + tmp_526[0], tmp_525[1] + tmp_526[1], tmp_525[2] + tmp_526[2]];
    signal tmp_528[3] <== CMul()(tmp_527, challengesFRI[1]);
    signal tmp_529[3] <== [mapValues.cm3_3[0] - evals[176][0], mapValues.cm3_3[1] - evals[176][1], mapValues.cm3_3[2] - evals[176][2]];
    signal tmp_530[3] <== [tmp_528[0] + tmp_529[0], tmp_528[1] + tmp_529[1], tmp_528[2] + tmp_529[2]];
    signal tmp_531[3] <== CMul()(tmp_530, xDivXSubXi[3]);
    signal tmp_532[3] <== [tmp_85[0] + tmp_531[0], tmp_85[1] + tmp_531[1], tmp_85[2] + tmp_531[2]];
    signal tmp_533[3] <== CMul()(challengesFRI[0], tmp_532);
    signal tmp_534[3] <== [consts[48] - evals[177][0], -evals[177][1], -evals[177][2]];
    signal tmp_535[3] <== CMul()(tmp_534, challengesFRI[1]);
    signal tmp_536[3] <== [consts[49] - evals[178][0], -evals[178][1], -evals[178][2]];
    signal tmp_537[3] <== [tmp_535[0] + tmp_536[0], tmp_535[1] + tmp_536[1], tmp_535[2] + tmp_536[2]];
    signal tmp_538[3] <== CMul()(tmp_537, challengesFRI[1]);
    signal tmp_539[3] <== [consts[57] - evals[179][0], -evals[179][1], -evals[179][2]];
    signal tmp_540[3] <== [tmp_538[0] + tmp_539[0], tmp_538[1] + tmp_539[1], tmp_538[2] + tmp_539[2]];
    signal tmp_541[3] <== CMul()(tmp_540, challengesFRI[1]);
    signal tmp_542[3] <== [mapValues.cm1_36 - evals[180][0], -evals[180][1], -evals[180][2]];
    signal tmp_543[3] <== [tmp_541[0] + tmp_542[0], tmp_541[1] + tmp_542[1], tmp_541[2] + tmp_542[2]];
    signal tmp_544[3] <== CMul()(tmp_543, challengesFRI[1]);
    signal tmp_545[3] <== [mapValues.cm1_37 - evals[181][0], -evals[181][1], -evals[181][2]];
    signal tmp_546[3] <== [tmp_544[0] + tmp_545[0], tmp_544[1] + tmp_545[1], tmp_544[2] + tmp_545[2]];
    signal tmp_547[3] <== CMul()(tmp_546, challengesFRI[1]);
    signal tmp_548[3] <== [mapValues.cm1_38 - evals[182][0], -evals[182][1], -evals[182][2]];
    signal tmp_549[3] <== [tmp_547[0] + tmp_548[0], tmp_547[1] + tmp_548[1], tmp_547[2] + tmp_548[2]];
    signal tmp_550[3] <== CMul()(tmp_549, challengesFRI[1]);
    signal tmp_551[3] <== [mapValues.cm1_39 - evals[183][0], -evals[183][1], -evals[183][2]];
    signal tmp_552[3] <== [tmp_550[0] + tmp_551[0], tmp_550[1] + tmp_551[1], tmp_550[2] + tmp_551[2]];
    signal tmp_553[3] <== CMul()(tmp_552, challengesFRI[1]);
    signal tmp_554[3] <== [mapValues.cm1_40 - evals[184][0], -evals[184][1], -evals[184][2]];
    signal tmp_555[3] <== [tmp_553[0] + tmp_554[0], tmp_553[1] + tmp_554[1], tmp_553[2] + tmp_554[2]];
    signal tmp_556[3] <== CMul()(tmp_555, challengesFRI[1]);
    signal tmp_557[3] <== [mapValues.cm1_41 - evals[185][0], -evals[185][1], -evals[185][2]];
    signal tmp_558[3] <== [tmp_556[0] + tmp_557[0], tmp_556[1] + tmp_557[1], tmp_556[2] + tmp_557[2]];
    signal tmp_559[3] <== CMul()(tmp_558, challengesFRI[1]);
    signal tmp_560[3] <== [mapValues.cm1_42 - evals[186][0], -evals[186][1], -evals[186][2]];
    signal tmp_561[3] <== [tmp_559[0] + tmp_560[0], tmp_559[1] + tmp_560[1], tmp_559[2] + tmp_560[2]];
    signal tmp_562[3] <== CMul()(tmp_561, challengesFRI[1]);
    signal tmp_563[3] <== [mapValues.cm1_43 - evals[187][0], -evals[187][1], -evals[187][2]];
    signal tmp_564[3] <== [tmp_562[0] + tmp_563[0], tmp_562[1] + tmp_563[1], tmp_562[2] + tmp_563[2]];
    signal tmp_565[3] <== CMul()(tmp_564, challengesFRI[1]);
    signal tmp_566[3] <== [mapValues.cm1_44 - evals[188][0], -evals[188][1], -evals[188][2]];
    signal tmp_567[3] <== [tmp_565[0] + tmp_566[0], tmp_565[1] + tmp_566[1], tmp_565[2] + tmp_566[2]];
    signal tmp_568[3] <== CMul()(tmp_567, challengesFRI[1]);
    signal tmp_569[3] <== [mapValues.cm1_45 - evals[189][0], -evals[189][1], -evals[189][2]];
    signal tmp_570[3] <== [tmp_568[0] + tmp_569[0], tmp_568[1] + tmp_569[1], tmp_568[2] + tmp_569[2]];
    signal tmp_571[3] <== CMul()(tmp_570, challengesFRI[1]);
    signal tmp_572[3] <== [mapValues.cm1_46 - evals[190][0], -evals[190][1], -evals[190][2]];
    signal tmp_573[3] <== [tmp_571[0] + tmp_572[0], tmp_571[1] + tmp_572[1], tmp_571[2] + tmp_572[2]];
    signal tmp_574[3] <== CMul()(tmp_573, challengesFRI[1]);
    signal tmp_575[3] <== [mapValues.cm1_47 - evals[191][0], -evals[191][1], -evals[191][2]];
    signal tmp_576[3] <== [tmp_574[0] + tmp_575[0], tmp_574[1] + tmp_575[1], tmp_574[2] + tmp_575[2]];
    signal tmp_577[3] <== CMul()(tmp_576, challengesFRI[1]);
    signal tmp_578[3] <== [mapValues.cm1_48 - evals[192][0], -evals[192][1], -evals[192][2]];
    signal tmp_579[3] <== [tmp_577[0] + tmp_578[0], tmp_577[1] + tmp_578[1], tmp_577[2] + tmp_578[2]];
    signal tmp_580[3] <== CMul()(tmp_579, challengesFRI[1]);
    signal tmp_581[3] <== [mapValues.cm1_49 - evals[193][0], -evals[193][1], -evals[193][2]];
    signal tmp_582[3] <== [tmp_580[0] + tmp_581[0], tmp_580[1] + tmp_581[1], tmp_580[2] + tmp_581[2]];
    signal tmp_583[3] <== CMul()(tmp_582, challengesFRI[1]);
    signal tmp_584[3] <== [mapValues.cm1_50 - evals[194][0], -evals[194][1], -evals[194][2]];
    signal tmp_585[3] <== [tmp_583[0] + tmp_584[0], tmp_583[1] + tmp_584[1], tmp_583[2] + tmp_584[2]];
    signal tmp_586[3] <== CMul()(tmp_585, challengesFRI[1]);
    signal tmp_587[3] <== [mapValues.cm1_51 - evals[195][0], -evals[195][1], -evals[195][2]];
    signal tmp_588[3] <== [tmp_586[0] + tmp_587[0], tmp_586[1] + tmp_587[1], tmp_586[2] + tmp_587[2]];
    signal tmp_589[3] <== CMul()(tmp_588, xDivXSubXi[4]);
    signal tmp_590[3] <== [tmp_533[0] + tmp_589[0], tmp_533[1] + tmp_589[1], tmp_533[2] + tmp_589[2]];
    signal tmp_591[3] <== CMul()(challengesFRI[0], tmp_590);
    signal tmp_592[3] <== [consts[48] - evals[196][0], -evals[196][1], -evals[196][2]];
    signal tmp_593[3] <== CMul()(tmp_592, challengesFRI[1]);
    signal tmp_594[3] <== [consts[49] - evals[197][0], -evals[197][1], -evals[197][2]];
    signal tmp_595[3] <== [tmp_593[0] + tmp_594[0], tmp_593[1] + tmp_594[1], tmp_593[2] + tmp_594[2]];
    signal tmp_596[3] <== CMul()(tmp_595, xDivXSubXi[5]);
    tmp_598 <== [tmp_591[0] + tmp_596[0], tmp_591[1] + tmp_596[1], tmp_591[2] + tmp_596[2]];

}

/*  Calculate FRI polinomial */
template CalculateFRIPolValue0() {
    signal input {binary} queriesFRI[20];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[198][3];
 
    signal input cm1[52];
 
    signal input cm2[55];
    signal input cm3[12];
    signal input consts[58];
    
    signal output queryVals[3];

    signal xacc[20];
    xacc[0] <== queriesFRI[0]*(7 * roots(20)-7) + 7;
    for (var i=1; i<20; i++) {
        xacc[i] <== xacc[i-1] * ( queriesFRI[i]*(roots(20 - i) - 1) +1);
    }

    signal xDivXSubXi[6][3];

    xDivXSubXi[0] <== CInv()([xacc[19] - 14996013474702747840 * challengeXi[0], - 14996013474702747840 * challengeXi[1], - 14996013474702747840 * challengeXi[2]]);
    xDivXSubXi[1] <== CInv()([xacc[19] - 15139302138664925958 * challengeXi[0], - 15139302138664925958 * challengeXi[1], - 15139302138664925958 * challengeXi[2]]);
    xDivXSubXi[2] <== CInv()([xacc[19] - 11804306091609355736 * challengeXi[0], - 11804306091609355736 * challengeXi[1], - 11804306091609355736 * challengeXi[2]]);
    xDivXSubXi[3] <== CInv()([xacc[19] - 1 * challengeXi[0], - 1 * challengeXi[1], - 1 * challengeXi[2]]);
    xDivXSubXi[4] <== CInv()([xacc[19] - 3411401055030829696 * challengeXi[0], - 3411401055030829696 * challengeXi[1], - 3411401055030829696 * challengeXi[2]]);
    xDivXSubXi[5] <== CInv()([xacc[19] - 5718075921287398682 * challengeXi[0], - 5718075921287398682 * challengeXi[1], - 5718075921287398682 * challengeXi[2]]);

    signal tmp_598[3];
 
    (tmp_598) <== CalculateFRIPolChunks0()(challengesFRI,evals,cm1,cm2,cm3,consts,xDivXSubXi);

    queryVals[0] <== tmp_598[0];
    queryVals[1] <== tmp_598[1];
    queryVals[2] <== tmp_598[2];
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
 
    signal input vals1[52];
 
    signal input vals2[55];
    signal input vals3[12];
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
    signal output cm1_36;
    signal output cm1_37;
    signal output cm1_38;
    signal output cm1_39;
    signal output cm1_40;
    signal output cm1_41;
    signal output cm1_42;
    signal output cm1_43;
    signal output cm1_44;
    signal output cm1_45;
    signal output cm1_46;
    signal output cm1_47;
    signal output cm1_48;
    signal output cm1_49;
    signal output cm1_50;
    signal output cm1_51;
    signal output cm2_0[3];
    signal output cm2_1[3];
    signal output cm2_2[3];
    signal output cm2_3[3];
    signal output cm2_4[3];
    signal output cm2_5[3];
    signal output cm2_6[3];
    signal output cm2_7[3];
    signal output cm2_8[3];
    signal output cm2_9;
    signal output cm2_10;
    signal output cm2_11;
    signal output cm2_12;
    signal output cm2_13;
    signal output cm2_14;
    signal output cm2_15;
    signal output cm2_16;
    signal output cm2_17;
    signal output cm2_18;
    signal output cm2_19;
    signal output cm2_20;
    signal output cm2_21;
    signal output cm2_22;
    signal output cm2_23;
    signal output cm2_24;
    signal output cm2_25;
    signal output cm2_26;
    signal output cm2_27;
    signal output cm2_28;
    signal output cm2_29;
    signal output cm2_30;
    signal output cm2_31;
    signal output cm2_32;
    signal output cm2_33;
    signal output cm2_34[3];
    signal output cm3_0[3];
    signal output cm3_1[3];
    signal output cm3_2[3];
    signal output cm3_3[3];


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
    cm1_36 <== vals1[36];
    cm1_37 <== vals1[37];
    cm1_38 <== vals1[38];
    cm1_39 <== vals1[39];
    cm1_40 <== vals1[40];
    cm1_41 <== vals1[41];
    cm1_42 <== vals1[42];
    cm1_43 <== vals1[43];
    cm1_44 <== vals1[44];
    cm1_45 <== vals1[45];
    cm1_46 <== vals1[46];
    cm1_47 <== vals1[47];
    cm1_48 <== vals1[48];
    cm1_49 <== vals1[49];
    cm1_50 <== vals1[50];
    cm1_51 <== vals1[51];
    cm2_0 <== [vals2[0],vals2[1] , vals2[2]];
    cm2_1 <== [vals2[3],vals2[4] , vals2[5]];
    cm2_2 <== [vals2[6],vals2[7] , vals2[8]];
    cm2_3 <== [vals2[9],vals2[10] , vals2[11]];
    cm2_4 <== [vals2[12],vals2[13] , vals2[14]];
    cm2_5 <== [vals2[15],vals2[16] , vals2[17]];
    cm2_6 <== [vals2[18],vals2[19] , vals2[20]];
    cm2_7 <== [vals2[21],vals2[22] , vals2[23]];
    cm2_8 <== [vals2[24],vals2[25] , vals2[26]];
    cm2_9 <== vals2[27];
    cm2_10 <== vals2[28];
    cm2_11 <== vals2[29];
    cm2_12 <== vals2[30];
    cm2_13 <== vals2[31];
    cm2_14 <== vals2[32];
    cm2_15 <== vals2[33];
    cm2_16 <== vals2[34];
    cm2_17 <== vals2[35];
    cm2_18 <== vals2[36];
    cm2_19 <== vals2[37];
    cm2_20 <== vals2[38];
    cm2_21 <== vals2[39];
    cm2_22 <== vals2[40];
    cm2_23 <== vals2[41];
    cm2_24 <== vals2[42];
    cm2_25 <== vals2[43];
    cm2_26 <== vals2[44];
    cm2_27 <== vals2[45];
    cm2_28 <== vals2[46];
    cm2_29 <== vals2[47];
    cm2_30 <== vals2[48];
    cm2_31 <== vals2[49];
    cm2_32 <== vals2[50];
    cm2_33 <== vals2[51];
    cm2_34 <== [vals2[52],vals2[53] , vals2[54]];
    cm3_0 <== [vals3[0],vals3[1] , vals3[2]];
    cm3_1 <== [vals3[3],vals3[4] , vals3[5]];
    cm3_2 <== [vals3[6],vals3[7] , vals3[8]];
    cm3_3 <== [vals3[9],vals3[10] , vals3[11]];
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
    for (var k= 8; k< 32; k++) {
        for (var e=0; e<3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k= 0; k < 8; k++) {
        _ <== lastIFFT[k];
    }
}

template StarkVerifier0() {
    signal input publics[391]; // publics polynomials
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials

    signal rootC[4] <== [1666073231162020912,13476540770991823196,8478030186702083301,6587207277830154145 ]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[198][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
 
    signal input s0_vals1[110][52];
 
    signal input s0_vals2[110][55];
                                       
    signal input s0_vals3[110][12];
    signal input s0_valsC[110][58];


    // Merkle proofs for each of the evaluations
    signal input s0_siblings1[110][8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_siblings2[110][8][12];
    signal input s0_last_mt_levels2[16][4];
 
    signal input s0_siblings3[110][8][12];
    signal input s0_last_mt_levels3[16][4];
    signal input s0_siblingsC[110][8][12];
    
    signal input s0_last_mt_levelsC[16][4];
    // Contains the root of the original polynomial and all the intermediate FRI polynomials except for the last step
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];

    // For each intermediate FRI polynomial and the last one, we store at vals the values needed to check the queries.
    // Given a query r,  the verifier needs b points to check it out, being b = 2^u, where u is the difference between two consecutive step
    // and the sibling paths for each query.
    signal input s1_vals[110][24];
    signal input s1_siblings[110][7][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals[110][24];
    signal input s2_siblings[110][5][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals[110][24];
    signal input s3_siblings[110][4][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals[110][24];
    signal input s4_siblings[110][2][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals[110][24];
    signal input s5_siblings[110][1][12];
    signal input s5_last_mt_levels[16][4];

    // Evaluations of the final FRI polynomial over a set of points of size bounded its degree
    signal input finalPol[32][3];

    signal input nonce;

    signal {binary} enabled;
    enabled <== 1;


    signal queryVals[110][3];

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
    signal {binary} queriesFRI[110][20];


    ///////////
    // Calculate challenges, challengesFRISteps and queriesFRI
    ///////////

 

    (challengesStage2,challengeQ,challengeXi,challengesFRI,challengesFRISteps,queriesFRI) <== Transcript0()(publics,rootC,root1,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,s5_root,finalPol, nonce, enabled);

    ///////////
    // Check constraints polynomial in the evaluation point
    ///////////

 

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputsC so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
 
    var s0_vals1_p[110][52][1];
 
    var s0_vals2_p[110][55][1];
 
    var s0_vals3_p[110][12][1];
    var s0_valsC_p[110][58][1];
    var s0_vals_p[110][1][3]; 
    var s1_vals_p[110][8][3]; 
    var s2_vals_p[110][8][3]; 
    var s3_vals_p[110][8][3]; 
    var s4_vals_p[110][8][3]; 
    var s5_vals_p[110][8][3]; 

    for (var q=0; q<110; q++) {
        // Preprocess vals for the initial FRI polynomial
 
        for (var i = 0; i < 52; i++) {
            s0_vals1_p[q][i][0] = s0_vals1[q][i];
        }
 
        for (var i = 0; i < 55; i++) {
            s0_vals2_p[q][i][0] = s0_vals2[q][i];
        }
 
        for (var i = 0; i < 12; i++) {
            s0_vals3_p[q][i][0] = s0_vals3[q][i];
        }
        for (var i = 0; i < 58; i++) {
            s0_valsC_p[q][i][0] = s0_valsC[q][i];
        }

        // Preprocess vals for each folded polynomial
        for(var e=0; e < 3; e++) {
            for(var c=0; c < 8; c++) {
                s1_vals_p[q][c][e] = s1_vals[q][c*3+e];
            }
            for(var c=0; c < 8; c++) {
                s2_vals_p[q][c][e] = s2_vals[q][c*3+e];
            }
            for(var c=0; c < 8; c++) {
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

    signal {binary} queriesFRIBits[110][10][2];
    for(var i = 0; i < 110; i++) {
        for(var j = 0; j < 10; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 20) {
                    queriesFRIBits[i][j][k] <== 0;
                } else {
                    queriesFRIBits[i][j][k] <== queriesFRI[i][j*2 + k];
                }
            }
        }
    }

    //Calculate merkle root for s0 vals
 
    for (var q=0; q<110; q++) {
        VerifyMerkleHashUntilLevel(1, 52, 4, 8, 2, 1048576)(s0_vals1_p[q], s0_siblings1[q], queriesFRIBits[q], s0_last_mt_levels1, enabled);
    }
 
    for (var q=0; q<110; q++) {
        VerifyMerkleHashUntilLevel(1, 55, 4, 8, 2, 1048576)(s0_vals2_p[q], s0_siblings2[q], queriesFRIBits[q], s0_last_mt_levels2, enabled);
    }

    for (var q=0; q<110; q++) {
        VerifyMerkleHashUntilLevel(1, 12, 4, 8, 2, 1048576)(s0_vals3_p[q], s0_siblings3[q], queriesFRIBits[q], s0_last_mt_levels3, enabled);
    }

    for (var q=0; q<110; q++) {
        VerifyMerkleHashUntilLevel(1, 58, 4, 8, 2, 1048576)(s0_valsC_p[q], s0_siblingsC[q], queriesFRIBits[q], s0_last_mt_levelsC, enabled);
                                    
    }


    signal {binary} s1_keys_merkle_bits[110][9][2];
    for (var q=0; q<110; q++) {
        // Calculate merkle root for s1 vals

        for(var j = 0; j < 9; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 17) {
                    s1_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s1_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 7, 2, 131072)(s1_vals_p[q], s1_siblings[q], s1_keys_merkle_bits[q], s1_last_mt_levels, enabled);
    }
    signal {binary} s2_keys_merkle_bits[110][7][2];
    for (var q=0; q<110; q++) {
        // Calculate merkle root for s2 vals

        for(var j = 0; j < 7; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 14) {
                    s2_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s2_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 5, 2, 16384)(s2_vals_p[q], s2_siblings[q], s2_keys_merkle_bits[q], s2_last_mt_levels, enabled);
    }
    signal {binary} s3_keys_merkle_bits[110][6][2];
    for (var q=0; q<110; q++) {
        // Calculate merkle root for s3 vals

        for(var j = 0; j < 6; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 11) {
                    s3_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s3_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 4, 2, 2048)(s3_vals_p[q], s3_siblings[q], s3_keys_merkle_bits[q], s3_last_mt_levels, enabled);
    }
    signal {binary} s4_keys_merkle_bits[110][4][2];
    for (var q=0; q<110; q++) {
        // Calculate merkle root for s4 vals

        for(var j = 0; j < 4; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 8) {
                    s4_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s4_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 2, 2, 256)(s4_vals_p[q], s4_siblings[q], s4_keys_merkle_bits[q], s4_last_mt_levels, enabled);
    }
    signal {binary} s5_keys_merkle_bits[110][3][2];
    for (var q=0; q<110; q++) {
        // Calculate merkle root for s5 vals

        for(var j = 0; j < 3; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 5) {
                    s5_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s5_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 1, 2, 32)(s5_vals_p[q], s5_siblings[q], s5_keys_merkle_bits[q], s5_last_mt_levels, enabled);
    }

    VerifyMerkleRoot(2, 4, 1048576)(s0_last_mt_levels1, root1, enabled);
    VerifyMerkleRoot(2, 4, 1048576)(s0_last_mt_levels2, root2, enabled);

    VerifyMerkleRoot(2, 4, 1048576)(s0_last_mt_levels3, root3, enabled);

    VerifyMerkleRoot(2, 4, 1048576)(s0_last_mt_levelsC, rootC, enabled);


    VerifyMerkleRoot(2, 4, 131072)(s1_last_mt_levels, s1_root, enabled);
    VerifyMerkleRoot(2, 4, 16384)(s2_last_mt_levels, s2_root, enabled);
    VerifyMerkleRoot(2, 4, 2048)(s3_last_mt_levels, s3_root, enabled);
    VerifyMerkleRoot(2, 4, 256)(s4_last_mt_levels, s4_root, enabled);
    VerifyMerkleRoot(2, 4, 32)(s5_last_mt_levels, s5_root, enabled);
        

    ///////////
    // Calculate FRI Polinomial
    ///////////
    
    for (var q=0; q<110; q++) {
        // Reconstruct FRI polinomial from evaluations
        queryVals[q] <== CalculateFRIPolValue0()(queriesFRI[q], challengeXi, challengesFRI, evals, s0_vals1[q], s0_vals2[q], s0_vals3[q], s0_valsC[q]);
    }

    ///////////
    // Verify FRI Polinomial
    ///////////
    signal {binary} s1_queriesFRI[110][17];
    signal {binary} s2_queriesFRI[110][14];
    signal {binary} s3_queriesFRI[110][11];
    signal {binary} s4_queriesFRI[110][8];
    signal {binary} s5_queriesFRI[110][5];

    for (var q=0; q<110; q++) {
      
        // Verify that the query is properly constructed. This is done by checking that the linear combination of the set of 
        // polynomials committed during the different rounds evaluated at z matches with the commitment of the FRI polynomial
        VerifyQuery0(20, 17)(queriesFRI[q], queryVals[q], s1_vals_p[q], enabled);

        ///////////
        // Verify FRI construction
        ///////////

        // For each folding level we need to check that the polynomial is properly constructed
        // Remember that if the step between polynomials is b = 2^l, the next polynomial p_(i+1) will have degree deg(p_i) / b

        // Check S1
        for(var i = 0; i < 17; i++) { s1_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 20, 17, 14, 2635249152773512046)(s1_queriesFRI[q], challengesFRISteps[1], s1_vals_p[q], s2_vals_p[q], enabled);

        // Check S2
        for(var i = 0; i < 14; i++) { s2_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 17, 14, 11, 12421013511830570338)(s2_queriesFRI[q], challengesFRISteps[2], s2_vals_p[q], s3_vals_p[q], enabled);

        // Check S3
        for(var i = 0; i < 11; i++) { s3_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 14, 11, 8, 11143297345130450484)(s3_queriesFRI[q], challengesFRISteps[3], s3_vals_p[q], s4_vals_p[q], enabled);

        // Check S4
        for(var i = 0; i < 8; i++) { s4_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 11, 8, 5, 1138102428757299658)(s4_queriesFRI[q], challengesFRISteps[4], s4_vals_p[q], s5_vals_p[q], enabled);

        // Check S5
        for(var i = 0; i < 5; i++) { s5_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(20, 8, 5, 0, 140704680260498080)(s5_queriesFRI[q], challengesFRISteps[5], s5_vals_p[q], finalPol, enabled);
    }

    VerifyFinalPol0()(finalPol, enabled);
}
    
