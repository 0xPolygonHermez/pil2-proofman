pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "hash/poseidon2/pow.circom";
include "tree/treeselector4.circom";
include "hash/poseidon2/merklehash.circom";

/*
    FRI query derivation: 72 indices into L_0 (20 bits each) from a
    fresh transcript over the challenge drawn after the final polynomial and the
    grinding nonce, with the proof of work checked first.
*/
template calculateFRIQueries0() {
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queriesL0[72][20];

    VerifyPoW(21)(challengeFRIQueries, nonce, enable);


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
    for (var i = 7; i < 16; i++) {
        _ <== transcriptHash_friQueries_1[i]; // Unused transcript values        
    }

    // From each transcript hash converted to bits, we assign those bits to queriesL0[q] to define the query positions
    var q = 0; // Query number 
    var b = 0; // Bit number 
    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_0[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_0[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_1[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_1[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_2[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_2[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_3[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_3[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_4[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_4[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_5[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_5[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_6[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_6[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_7[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_7[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_8[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_8[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_9[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_9[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_10[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_10[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_11[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_11[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_12[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_12[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_13[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_14[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_14[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_15[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_15[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_16[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_16[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_17[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_17[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_18[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_18[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_19[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_19[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_20[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_20[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_21[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_21[63]; // Unused last bit

    for (var j = 0; j < 54; j++) {
        queriesL0[q][b] <== transcriptN2b_22[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    for (var j = 54; j < 64; j++) {
        _ <== transcriptN2b_22[j]; // Unused bits        
    }
}

/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input publics[399];
    signal input rootC[4];
    signal input root1[4];



    signal input root2[4];
    signal input root3[4];
    signal input evals[144][3]; 
    signal input s1_root[4]; // Root of a polynomial of degree lower than 2^17, evaluated over 2^20 points
    signal input s2_root[4]; // Root of a polynomial of degree lower than 2^14, evaluated over 2^17 points
    signal input s3_root[4]; // Root of a polynomial of degree lower than 2^11, evaluated over 2^14 points
    signal input s4_root[4]; // Root of a polynomial of degree lower than 2^8, evaluated over 2^11 points
    signal input s5_root[4]; // Root of a polynomial of degree lower than 2^5, evaluated over 2^8 points
    signal input finalPol[32][3]; // Evaluations over 2^5 points of a polynomial of degree lower than 2^2
    signal input nonce;
    signal input {binary} enable;

    signal output challengesStage2[2][3];
    signal output challengeQ[3];
    signal output challengeOut[3];
    signal output challengesDeep[2][3];

    // r^fold_s, s = 0..M−1: drawn after the root of f_s, folds it into f_{s+1}.
    signal output rFold[5][3];
    signal challengeQueries[3];
    signal output {binary} queriesL0[72][20];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];



    signal transcriptHash_publics_0[16] <== Poseidon2(4, 16)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7],publics[8],publics[9],publics[10],publics[11]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_0[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_1[16] <== Poseidon2(4, 16)([publics[12],publics[13],publics[14],publics[15],publics[16],publics[17],publics[18],publics[19],publics[20],publics[21],publics[22],publics[23]], [transcriptHash_publics_0[0],transcriptHash_publics_0[1],transcriptHash_publics_0[2],transcriptHash_publics_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_1[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_2[16] <== Poseidon2(4, 16)([publics[24],publics[25],publics[26],publics[27],publics[28],publics[29],publics[30],publics[31],publics[32],publics[33],publics[34],publics[35]], [transcriptHash_publics_1[0],transcriptHash_publics_1[1],transcriptHash_publics_1[2],transcriptHash_publics_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_2[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_3[16] <== Poseidon2(4, 16)([publics[36],publics[37],publics[38],publics[39],publics[40],publics[41],publics[42],publics[43],publics[44],publics[45],publics[46],publics[47]], [transcriptHash_publics_2[0],transcriptHash_publics_2[1],transcriptHash_publics_2[2],transcriptHash_publics_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_3[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_4[16] <== Poseidon2(4, 16)([publics[48],publics[49],publics[50],publics[51],publics[52],publics[53],publics[54],publics[55],publics[56],publics[57],publics[58],publics[59]], [transcriptHash_publics_3[0],transcriptHash_publics_3[1],transcriptHash_publics_3[2],transcriptHash_publics_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_4[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_5[16] <== Poseidon2(4, 16)([publics[60],publics[61],publics[62],publics[63],publics[64],publics[65],publics[66],publics[67],publics[68],publics[69],publics[70],publics[71]], [transcriptHash_publics_4[0],transcriptHash_publics_4[1],transcriptHash_publics_4[2],transcriptHash_publics_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_5[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_6[16] <== Poseidon2(4, 16)([publics[72],publics[73],publics[74],publics[75],publics[76],publics[77],publics[78],publics[79],publics[80],publics[81],publics[82],publics[83]], [transcriptHash_publics_5[0],transcriptHash_publics_5[1],transcriptHash_publics_5[2],transcriptHash_publics_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_6[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_7[16] <== Poseidon2(4, 16)([publics[84],publics[85],publics[86],publics[87],publics[88],publics[89],publics[90],publics[91],publics[92],publics[93],publics[94],publics[95]], [transcriptHash_publics_6[0],transcriptHash_publics_6[1],transcriptHash_publics_6[2],transcriptHash_publics_6[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_7[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_8[16] <== Poseidon2(4, 16)([publics[96],publics[97],publics[98],publics[99],publics[100],publics[101],publics[102],publics[103],publics[104],publics[105],publics[106],publics[107]], [transcriptHash_publics_7[0],transcriptHash_publics_7[1],transcriptHash_publics_7[2],transcriptHash_publics_7[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_8[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_9[16] <== Poseidon2(4, 16)([publics[108],publics[109],publics[110],publics[111],publics[112],publics[113],publics[114],publics[115],publics[116],publics[117],publics[118],publics[119]], [transcriptHash_publics_8[0],transcriptHash_publics_8[1],transcriptHash_publics_8[2],transcriptHash_publics_8[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_9[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_10[16] <== Poseidon2(4, 16)([publics[120],publics[121],publics[122],publics[123],publics[124],publics[125],publics[126],publics[127],publics[128],publics[129],publics[130],publics[131]], [transcriptHash_publics_9[0],transcriptHash_publics_9[1],transcriptHash_publics_9[2],transcriptHash_publics_9[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_10[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_11[16] <== Poseidon2(4, 16)([publics[132],publics[133],publics[134],publics[135],publics[136],publics[137],publics[138],publics[139],publics[140],publics[141],publics[142],publics[143]], [transcriptHash_publics_10[0],transcriptHash_publics_10[1],transcriptHash_publics_10[2],transcriptHash_publics_10[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_11[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_12[16] <== Poseidon2(4, 16)([publics[144],publics[145],publics[146],publics[147],publics[148],publics[149],publics[150],publics[151],publics[152],publics[153],publics[154],publics[155]], [transcriptHash_publics_11[0],transcriptHash_publics_11[1],transcriptHash_publics_11[2],transcriptHash_publics_11[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_12[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_13[16] <== Poseidon2(4, 16)([publics[156],publics[157],publics[158],publics[159],publics[160],publics[161],publics[162],publics[163],publics[164],publics[165],publics[166],publics[167]], [transcriptHash_publics_12[0],transcriptHash_publics_12[1],transcriptHash_publics_12[2],transcriptHash_publics_12[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_13[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_14[16] <== Poseidon2(4, 16)([publics[168],publics[169],publics[170],publics[171],publics[172],publics[173],publics[174],publics[175],publics[176],publics[177],publics[178],publics[179]], [transcriptHash_publics_13[0],transcriptHash_publics_13[1],transcriptHash_publics_13[2],transcriptHash_publics_13[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_14[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_15[16] <== Poseidon2(4, 16)([publics[180],publics[181],publics[182],publics[183],publics[184],publics[185],publics[186],publics[187],publics[188],publics[189],publics[190],publics[191]], [transcriptHash_publics_14[0],transcriptHash_publics_14[1],transcriptHash_publics_14[2],transcriptHash_publics_14[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_15[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_16[16] <== Poseidon2(4, 16)([publics[192],publics[193],publics[194],publics[195],publics[196],publics[197],publics[198],publics[199],publics[200],publics[201],publics[202],publics[203]], [transcriptHash_publics_15[0],transcriptHash_publics_15[1],transcriptHash_publics_15[2],transcriptHash_publics_15[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_16[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_17[16] <== Poseidon2(4, 16)([publics[204],publics[205],publics[206],publics[207],publics[208],publics[209],publics[210],publics[211],publics[212],publics[213],publics[214],publics[215]], [transcriptHash_publics_16[0],transcriptHash_publics_16[1],transcriptHash_publics_16[2],transcriptHash_publics_16[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_17[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_18[16] <== Poseidon2(4, 16)([publics[216],publics[217],publics[218],publics[219],publics[220],publics[221],publics[222],publics[223],publics[224],publics[225],publics[226],publics[227]], [transcriptHash_publics_17[0],transcriptHash_publics_17[1],transcriptHash_publics_17[2],transcriptHash_publics_17[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_18[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_19[16] <== Poseidon2(4, 16)([publics[228],publics[229],publics[230],publics[231],publics[232],publics[233],publics[234],publics[235],publics[236],publics[237],publics[238],publics[239]], [transcriptHash_publics_18[0],transcriptHash_publics_18[1],transcriptHash_publics_18[2],transcriptHash_publics_18[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_19[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_20[16] <== Poseidon2(4, 16)([publics[240],publics[241],publics[242],publics[243],publics[244],publics[245],publics[246],publics[247],publics[248],publics[249],publics[250],publics[251]], [transcriptHash_publics_19[0],transcriptHash_publics_19[1],transcriptHash_publics_19[2],transcriptHash_publics_19[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_20[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_21[16] <== Poseidon2(4, 16)([publics[252],publics[253],publics[254],publics[255],publics[256],publics[257],publics[258],publics[259],publics[260],publics[261],publics[262],publics[263]], [transcriptHash_publics_20[0],transcriptHash_publics_20[1],transcriptHash_publics_20[2],transcriptHash_publics_20[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_21[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_22[16] <== Poseidon2(4, 16)([publics[264],publics[265],publics[266],publics[267],publics[268],publics[269],publics[270],publics[271],publics[272],publics[273],publics[274],publics[275]], [transcriptHash_publics_21[0],transcriptHash_publics_21[1],transcriptHash_publics_21[2],transcriptHash_publics_21[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_22[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_23[16] <== Poseidon2(4, 16)([publics[276],publics[277],publics[278],publics[279],publics[280],publics[281],publics[282],publics[283],publics[284],publics[285],publics[286],publics[287]], [transcriptHash_publics_22[0],transcriptHash_publics_22[1],transcriptHash_publics_22[2],transcriptHash_publics_22[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_23[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_24[16] <== Poseidon2(4, 16)([publics[288],publics[289],publics[290],publics[291],publics[292],publics[293],publics[294],publics[295],publics[296],publics[297],publics[298],publics[299]], [transcriptHash_publics_23[0],transcriptHash_publics_23[1],transcriptHash_publics_23[2],transcriptHash_publics_23[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_24[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_25[16] <== Poseidon2(4, 16)([publics[300],publics[301],publics[302],publics[303],publics[304],publics[305],publics[306],publics[307],publics[308],publics[309],publics[310],publics[311]], [transcriptHash_publics_24[0],transcriptHash_publics_24[1],transcriptHash_publics_24[2],transcriptHash_publics_24[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_25[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_26[16] <== Poseidon2(4, 16)([publics[312],publics[313],publics[314],publics[315],publics[316],publics[317],publics[318],publics[319],publics[320],publics[321],publics[322],publics[323]], [transcriptHash_publics_25[0],transcriptHash_publics_25[1],transcriptHash_publics_25[2],transcriptHash_publics_25[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_26[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_27[16] <== Poseidon2(4, 16)([publics[324],publics[325],publics[326],publics[327],publics[328],publics[329],publics[330],publics[331],publics[332],publics[333],publics[334],publics[335]], [transcriptHash_publics_26[0],transcriptHash_publics_26[1],transcriptHash_publics_26[2],transcriptHash_publics_26[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_27[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_28[16] <== Poseidon2(4, 16)([publics[336],publics[337],publics[338],publics[339],publics[340],publics[341],publics[342],publics[343],publics[344],publics[345],publics[346],publics[347]], [transcriptHash_publics_27[0],transcriptHash_publics_27[1],transcriptHash_publics_27[2],transcriptHash_publics_27[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_28[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_29[16] <== Poseidon2(4, 16)([publics[348],publics[349],publics[350],publics[351],publics[352],publics[353],publics[354],publics[355],publics[356],publics[357],publics[358],publics[359]], [transcriptHash_publics_28[0],transcriptHash_publics_28[1],transcriptHash_publics_28[2],transcriptHash_publics_28[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_29[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_30[16] <== Poseidon2(4, 16)([publics[360],publics[361],publics[362],publics[363],publics[364],publics[365],publics[366],publics[367],publics[368],publics[369],publics[370],publics[371]], [transcriptHash_publics_29[0],transcriptHash_publics_29[1],transcriptHash_publics_29[2],transcriptHash_publics_29[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_30[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_31[16] <== Poseidon2(4, 16)([publics[372],publics[373],publics[374],publics[375],publics[376],publics[377],publics[378],publics[379],publics[380],publics[381],publics[382],publics[383]], [transcriptHash_publics_30[0],transcriptHash_publics_30[1],transcriptHash_publics_30[2],transcriptHash_publics_30[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_31[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_32[16] <== Poseidon2(4, 16)([publics[384],publics[385],publics[386],publics[387],publics[388],publics[389],publics[390],publics[391],publics[392],publics[393],publics[394],publics[395]], [transcriptHash_publics_31[0],transcriptHash_publics_31[1],transcriptHash_publics_31[2],transcriptHash_publics_31[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_32[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_33[16] <== Poseidon2(4, 16)([publics[396],publics[397],publics[398],0,0,0,0,0,0,0,0,0], [transcriptHash_publics_32[0],transcriptHash_publics_32[1],transcriptHash_publics_32[2],transcriptHash_publics_32[3]]);
    publicsHash <== [transcriptHash_publics_33[0], transcriptHash_publics_33[1], transcriptHash_publics_33[2], transcriptHash_publics_33[3]];

    signal transcriptHash_0[16] <== Poseidon2(4, 16)([rootC[0],rootC[1],rootC[2],rootC[3],publicsHash[0],publicsHash[1],publicsHash[2],publicsHash[3],root1[0],root1[1],root1[2],root1[3]], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for (var i = 6; i < 16; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }

    signal transcriptHash_1[16] <== Poseidon2(4, 16)([root2[0],root2[1],root2[2],root2[3],0,0,0,0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }

    signal transcriptHash_2[16] <== Poseidon2(4, 16)([root3[0],root3[1],root3[2],root3[3],0,0,0,0,0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeOut <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];

    signal transcriptHash_evals_0[16] <== Poseidon2(4, 16)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1],evals[2][2],evals[3][0],evals[3][1],evals[3][2]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_1[16] <== Poseidon2(4, 16)([evals[4][0],evals[4][1],evals[4][2],evals[5][0],evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_2[16] <== Poseidon2(4, 16)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1],evals[10][2],evals[11][0],evals[11][1],evals[11][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_3[16] <== Poseidon2(4, 16)([evals[12][0],evals[12][1],evals[12][2],evals[13][0],evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_4[16] <== Poseidon2(4, 16)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1],evals[18][2],evals[19][0],evals[19][1],evals[19][2]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_4[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_5[16] <== Poseidon2(4, 16)([evals[20][0],evals[20][1],evals[20][2],evals[21][0],evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],evals[23][0],evals[23][1],evals[23][2]], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_5[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_6[16] <== Poseidon2(4, 16)([evals[24][0],evals[24][1],evals[24][2],evals[25][0],evals[25][1],evals[25][2],evals[26][0],evals[26][1],evals[26][2],evals[27][0],evals[27][1],evals[27][2]], [transcriptHash_evals_5[0],transcriptHash_evals_5[1],transcriptHash_evals_5[2],transcriptHash_evals_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_6[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_7[16] <== Poseidon2(4, 16)([evals[28][0],evals[28][1],evals[28][2],evals[29][0],evals[29][1],evals[29][2],evals[30][0],evals[30][1],evals[30][2],evals[31][0],evals[31][1],evals[31][2]], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_7[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_8[16] <== Poseidon2(4, 16)([evals[32][0],evals[32][1],evals[32][2],evals[33][0],evals[33][1],evals[33][2],evals[34][0],evals[34][1],evals[34][2],evals[35][0],evals[35][1],evals[35][2]], [transcriptHash_evals_7[0],transcriptHash_evals_7[1],transcriptHash_evals_7[2],transcriptHash_evals_7[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_8[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_9[16] <== Poseidon2(4, 16)([evals[36][0],evals[36][1],evals[36][2],evals[37][0],evals[37][1],evals[37][2],evals[38][0],evals[38][1],evals[38][2],evals[39][0],evals[39][1],evals[39][2]], [transcriptHash_evals_8[0],transcriptHash_evals_8[1],transcriptHash_evals_8[2],transcriptHash_evals_8[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_9[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_10[16] <== Poseidon2(4, 16)([evals[40][0],evals[40][1],evals[40][2],evals[41][0],evals[41][1],evals[41][2],evals[42][0],evals[42][1],evals[42][2],evals[43][0],evals[43][1],evals[43][2]], [transcriptHash_evals_9[0],transcriptHash_evals_9[1],transcriptHash_evals_9[2],transcriptHash_evals_9[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_10[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_11[16] <== Poseidon2(4, 16)([evals[44][0],evals[44][1],evals[44][2],evals[45][0],evals[45][1],evals[45][2],evals[46][0],evals[46][1],evals[46][2],evals[47][0],evals[47][1],evals[47][2]], [transcriptHash_evals_10[0],transcriptHash_evals_10[1],transcriptHash_evals_10[2],transcriptHash_evals_10[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_11[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_12[16] <== Poseidon2(4, 16)([evals[48][0],evals[48][1],evals[48][2],evals[49][0],evals[49][1],evals[49][2],evals[50][0],evals[50][1],evals[50][2],evals[51][0],evals[51][1],evals[51][2]], [transcriptHash_evals_11[0],transcriptHash_evals_11[1],transcriptHash_evals_11[2],transcriptHash_evals_11[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_12[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_13[16] <== Poseidon2(4, 16)([evals[52][0],evals[52][1],evals[52][2],evals[53][0],evals[53][1],evals[53][2],evals[54][0],evals[54][1],evals[54][2],evals[55][0],evals[55][1],evals[55][2]], [transcriptHash_evals_12[0],transcriptHash_evals_12[1],transcriptHash_evals_12[2],transcriptHash_evals_12[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_13[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_14[16] <== Poseidon2(4, 16)([evals[56][0],evals[56][1],evals[56][2],evals[57][0],evals[57][1],evals[57][2],evals[58][0],evals[58][1],evals[58][2],evals[59][0],evals[59][1],evals[59][2]], [transcriptHash_evals_13[0],transcriptHash_evals_13[1],transcriptHash_evals_13[2],transcriptHash_evals_13[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_14[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_15[16] <== Poseidon2(4, 16)([evals[60][0],evals[60][1],evals[60][2],evals[61][0],evals[61][1],evals[61][2],evals[62][0],evals[62][1],evals[62][2],evals[63][0],evals[63][1],evals[63][2]], [transcriptHash_evals_14[0],transcriptHash_evals_14[1],transcriptHash_evals_14[2],transcriptHash_evals_14[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_15[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_16[16] <== Poseidon2(4, 16)([evals[64][0],evals[64][1],evals[64][2],evals[65][0],evals[65][1],evals[65][2],evals[66][0],evals[66][1],evals[66][2],evals[67][0],evals[67][1],evals[67][2]], [transcriptHash_evals_15[0],transcriptHash_evals_15[1],transcriptHash_evals_15[2],transcriptHash_evals_15[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_16[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_17[16] <== Poseidon2(4, 16)([evals[68][0],evals[68][1],evals[68][2],evals[69][0],evals[69][1],evals[69][2],evals[70][0],evals[70][1],evals[70][2],evals[71][0],evals[71][1],evals[71][2]], [transcriptHash_evals_16[0],transcriptHash_evals_16[1],transcriptHash_evals_16[2],transcriptHash_evals_16[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_17[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_18[16] <== Poseidon2(4, 16)([evals[72][0],evals[72][1],evals[72][2],evals[73][0],evals[73][1],evals[73][2],evals[74][0],evals[74][1],evals[74][2],evals[75][0],evals[75][1],evals[75][2]], [transcriptHash_evals_17[0],transcriptHash_evals_17[1],transcriptHash_evals_17[2],transcriptHash_evals_17[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_18[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_19[16] <== Poseidon2(4, 16)([evals[76][0],evals[76][1],evals[76][2],evals[77][0],evals[77][1],evals[77][2],evals[78][0],evals[78][1],evals[78][2],evals[79][0],evals[79][1],evals[79][2]], [transcriptHash_evals_18[0],transcriptHash_evals_18[1],transcriptHash_evals_18[2],transcriptHash_evals_18[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_19[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_20[16] <== Poseidon2(4, 16)([evals[80][0],evals[80][1],evals[80][2],evals[81][0],evals[81][1],evals[81][2],evals[82][0],evals[82][1],evals[82][2],evals[83][0],evals[83][1],evals[83][2]], [transcriptHash_evals_19[0],transcriptHash_evals_19[1],transcriptHash_evals_19[2],transcriptHash_evals_19[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_20[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_21[16] <== Poseidon2(4, 16)([evals[84][0],evals[84][1],evals[84][2],evals[85][0],evals[85][1],evals[85][2],evals[86][0],evals[86][1],evals[86][2],evals[87][0],evals[87][1],evals[87][2]], [transcriptHash_evals_20[0],transcriptHash_evals_20[1],transcriptHash_evals_20[2],transcriptHash_evals_20[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_21[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_22[16] <== Poseidon2(4, 16)([evals[88][0],evals[88][1],evals[88][2],evals[89][0],evals[89][1],evals[89][2],evals[90][0],evals[90][1],evals[90][2],evals[91][0],evals[91][1],evals[91][2]], [transcriptHash_evals_21[0],transcriptHash_evals_21[1],transcriptHash_evals_21[2],transcriptHash_evals_21[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_22[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_23[16] <== Poseidon2(4, 16)([evals[92][0],evals[92][1],evals[92][2],evals[93][0],evals[93][1],evals[93][2],evals[94][0],evals[94][1],evals[94][2],evals[95][0],evals[95][1],evals[95][2]], [transcriptHash_evals_22[0],transcriptHash_evals_22[1],transcriptHash_evals_22[2],transcriptHash_evals_22[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_23[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_24[16] <== Poseidon2(4, 16)([evals[96][0],evals[96][1],evals[96][2],evals[97][0],evals[97][1],evals[97][2],evals[98][0],evals[98][1],evals[98][2],evals[99][0],evals[99][1],evals[99][2]], [transcriptHash_evals_23[0],transcriptHash_evals_23[1],transcriptHash_evals_23[2],transcriptHash_evals_23[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_24[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_25[16] <== Poseidon2(4, 16)([evals[100][0],evals[100][1],evals[100][2],evals[101][0],evals[101][1],evals[101][2],evals[102][0],evals[102][1],evals[102][2],evals[103][0],evals[103][1],evals[103][2]], [transcriptHash_evals_24[0],transcriptHash_evals_24[1],transcriptHash_evals_24[2],transcriptHash_evals_24[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_25[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_26[16] <== Poseidon2(4, 16)([evals[104][0],evals[104][1],evals[104][2],evals[105][0],evals[105][1],evals[105][2],evals[106][0],evals[106][1],evals[106][2],evals[107][0],evals[107][1],evals[107][2]], [transcriptHash_evals_25[0],transcriptHash_evals_25[1],transcriptHash_evals_25[2],transcriptHash_evals_25[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_26[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_27[16] <== Poseidon2(4, 16)([evals[108][0],evals[108][1],evals[108][2],evals[109][0],evals[109][1],evals[109][2],evals[110][0],evals[110][1],evals[110][2],evals[111][0],evals[111][1],evals[111][2]], [transcriptHash_evals_26[0],transcriptHash_evals_26[1],transcriptHash_evals_26[2],transcriptHash_evals_26[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_27[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_28[16] <== Poseidon2(4, 16)([evals[112][0],evals[112][1],evals[112][2],evals[113][0],evals[113][1],evals[113][2],evals[114][0],evals[114][1],evals[114][2],evals[115][0],evals[115][1],evals[115][2]], [transcriptHash_evals_27[0],transcriptHash_evals_27[1],transcriptHash_evals_27[2],transcriptHash_evals_27[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_28[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_29[16] <== Poseidon2(4, 16)([evals[116][0],evals[116][1],evals[116][2],evals[117][0],evals[117][1],evals[117][2],evals[118][0],evals[118][1],evals[118][2],evals[119][0],evals[119][1],evals[119][2]], [transcriptHash_evals_28[0],transcriptHash_evals_28[1],transcriptHash_evals_28[2],transcriptHash_evals_28[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_29[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_30[16] <== Poseidon2(4, 16)([evals[120][0],evals[120][1],evals[120][2],evals[121][0],evals[121][1],evals[121][2],evals[122][0],evals[122][1],evals[122][2],evals[123][0],evals[123][1],evals[123][2]], [transcriptHash_evals_29[0],transcriptHash_evals_29[1],transcriptHash_evals_29[2],transcriptHash_evals_29[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_30[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_31[16] <== Poseidon2(4, 16)([evals[124][0],evals[124][1],evals[124][2],evals[125][0],evals[125][1],evals[125][2],evals[126][0],evals[126][1],evals[126][2],evals[127][0],evals[127][1],evals[127][2]], [transcriptHash_evals_30[0],transcriptHash_evals_30[1],transcriptHash_evals_30[2],transcriptHash_evals_30[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_31[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_32[16] <== Poseidon2(4, 16)([evals[128][0],evals[128][1],evals[128][2],evals[129][0],evals[129][1],evals[129][2],evals[130][0],evals[130][1],evals[130][2],evals[131][0],evals[131][1],evals[131][2]], [transcriptHash_evals_31[0],transcriptHash_evals_31[1],transcriptHash_evals_31[2],transcriptHash_evals_31[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_32[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_33[16] <== Poseidon2(4, 16)([evals[132][0],evals[132][1],evals[132][2],evals[133][0],evals[133][1],evals[133][2],evals[134][0],evals[134][1],evals[134][2],evals[135][0],evals[135][1],evals[135][2]], [transcriptHash_evals_32[0],transcriptHash_evals_32[1],transcriptHash_evals_32[2],transcriptHash_evals_32[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_33[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_34[16] <== Poseidon2(4, 16)([evals[136][0],evals[136][1],evals[136][2],evals[137][0],evals[137][1],evals[137][2],evals[138][0],evals[138][1],evals[138][2],evals[139][0],evals[139][1],evals[139][2]], [transcriptHash_evals_33[0],transcriptHash_evals_33[1],transcriptHash_evals_33[2],transcriptHash_evals_33[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_34[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_35[16] <== Poseidon2(4, 16)([evals[140][0],evals[140][1],evals[140][2],evals[141][0],evals[141][1],evals[141][2],evals[142][0],evals[142][1],evals[142][2],evals[143][0],evals[143][1],evals[143][2]], [transcriptHash_evals_34[0],transcriptHash_evals_34[1],transcriptHash_evals_34[2],transcriptHash_evals_34[3]]);
    evalsHash <== [transcriptHash_evals_35[0], transcriptHash_evals_35[1], transcriptHash_evals_35[2], transcriptHash_evals_35[3]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }

    signal transcriptHash_3[16] <== Poseidon2(4, 16)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0,0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesDeep[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesDeep[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    for (var i = 6; i < 16; i++) {
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }

    signal transcriptHash_4[16] <== Poseidon2(4, 16)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0,0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    rFold[0] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }

    signal transcriptHash_5[16] <== Poseidon2(4, 16)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0,0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    rFold[1] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }

    signal transcriptHash_6[16] <== Poseidon2(4, 16)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0,0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    rFold[2] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }

    signal transcriptHash_7[16] <== Poseidon2(4, 16)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0,0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    rFold[3] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }

    signal transcriptHash_8[16] <== Poseidon2(4, 16)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0,0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    rFold[4] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];

    signal transcriptHash_lastPolFRI_0[16] <== Poseidon2(4, 16)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1],finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_1[16] <== Poseidon2(4, 16)([finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0],finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_2[16] <== Poseidon2(4, 16)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1],finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_3[16] <== Poseidon2(4, 16)([finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0],finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_4[16] <== Poseidon2(4, 16)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1],finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_5[16] <== Poseidon2(4, 16)([finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0],finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_6[16] <== Poseidon2(4, 16)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1],finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_7[16] <== Poseidon2(4, 16)([finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0],finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_7[0], transcriptHash_lastPolFRI_7[1], transcriptHash_lastPolFRI_7[2], transcriptHash_lastPolFRI_7[3]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }

    signal transcriptHash_9[16] <== Poseidon2(4, 16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    challengeQueries <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];
    queriesL0 <== calculateFRIQueries0()(challengeQueries, nonce, enable);
}

/*
    Verify that FRI polynomials are built properly
*/
template VerifyFRI0(nBitsExt, prevStepBits, currStepBits, nextStepBits, e0) {
    var nextStep = currStepBits - nextStepBits; 
    var step = prevStepBits - currStepBits;

    signal input {binary} queriesL0[currStepBits];
    signal input friChallenge[3];
    signal input s_vals_curr[1<< step][3];
    signal input s_vals_next[1<< nextStep][3];
    signal input {binary} enable;

    signal sx[currStepBits];
    
    sx[0] <==  e0 *( queriesL0[0] * (invroots(prevStepBits) -1) + 1);
    for (var i = 1; i < currStepBits; i++) {
        sx[i] <== sx[i-1] *  ( queriesL0[i] * (invroots(prevStepBits -i) -1) +1);
    }
        
    // Perform an IFFT to obtain the coefficients of the polynomial given s_vals and evaluate it 
    signal coefs[1 << step][3] <== FFT(step, 3, 1)(s_vals_curr);
    signal evalXprime[3] <== [friChallenge[0] *  sx[currStepBits - 1], friChallenge[1] * sx[currStepBits - 1], friChallenge[2] *  sx[currStepBits - 1]];
    signal evalPol[3] <== EvalPol(1 << step)(coefs, evalXprime);

    signal {binary} keys_lowValues[nextStep];
    for (var i = 0; i < nextStep; i++) { keys_lowValues[i] <== queriesL0[i + nextStepBits]; } 
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s_vals_next, keys_lowValues);

    enable * (lowValues[0] - evalPol[0]) === 0;
    enable * (lowValues[1] - evalPol[1]) === 0;
    enable * (lowValues[2] - evalPol[2]) === 0;
}


/*
    The air's constraint algebra at the out-of-domain point z = challengeOut, as straight-line
    code generated from the air's expression DAG: it reads the claimed openings `evals` (plus
    publics and the airgroup/air/proof values), folds every constraint with powers of
    challengeQ and divides by the vanishing polynomials.
*/
template VerifyEvaluationsChunks0_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];

    signal input Zh[3];


    signal output tmp_6223[3];
    signal tmp_5225[3] <== [evals[64][0] - publics[0], evals[64][1], evals[64][2]];
    signal tmp_5226[3] <== CMul()(evals[61], tmp_5225);
    signal tmp_5227[3] <== CMul()(challengeQ, tmp_5226);
    signal tmp_5228[3] <== [evals[65][0] - publics[1], evals[65][1], evals[65][2]];
    signal tmp_5229[3] <== CMul()(evals[61], tmp_5228);
    signal tmp_5230[3] <== [tmp_5227[0] + tmp_5229[0], tmp_5227[1] + tmp_5229[1], tmp_5227[2] + tmp_5229[2]];
    signal tmp_5231[3] <== CMul()(challengeQ, tmp_5230);
    signal tmp_5232[3] <== [evals[66][0] - publics[2], evals[66][1], evals[66][2]];
    signal tmp_5233[3] <== CMul()(evals[61], tmp_5232);
    signal tmp_5234[3] <== [tmp_5231[0] + tmp_5233[0], tmp_5231[1] + tmp_5233[1], tmp_5231[2] + tmp_5233[2]];
    signal tmp_5235[3] <== CMul()(challengeQ, tmp_5234);
    signal tmp_5236[3] <== [evals[67][0] - publics[3], evals[67][1], evals[67][2]];
    signal tmp_5237[3] <== CMul()(evals[61], tmp_5236);
    signal tmp_5238[3] <== [tmp_5235[0] + tmp_5237[0], tmp_5235[1] + tmp_5237[1], tmp_5235[2] + tmp_5237[2]];
    signal tmp_5239[3] <== CMul()(challengeQ, tmp_5238);
    signal tmp_5240[3] <== [evals[68][0] - publics[4], evals[68][1], evals[68][2]];
    signal tmp_5241[3] <== CMul()(evals[61], tmp_5240);
    signal tmp_5242[3] <== [tmp_5239[0] + tmp_5241[0], tmp_5239[1] + tmp_5241[1], tmp_5239[2] + tmp_5241[2]];
    signal tmp_5243[3] <== CMul()(challengeQ, tmp_5242);
    signal tmp_5244[3] <== [evals[69][0] - publics[5], evals[69][1], evals[69][2]];
    signal tmp_5245[3] <== CMul()(evals[61], tmp_5244);
    signal tmp_5246[3] <== [tmp_5243[0] + tmp_5245[0], tmp_5243[1] + tmp_5245[1], tmp_5243[2] + tmp_5245[2]];
    signal tmp_5247[3] <== CMul()(challengeQ, tmp_5246);
    signal tmp_5248[3] <== [evals[70][0] - publics[6], evals[70][1], evals[70][2]];
    signal tmp_5249[3] <== CMul()(evals[61], tmp_5248);
    signal tmp_5250[3] <== [tmp_5247[0] + tmp_5249[0], tmp_5247[1] + tmp_5249[1], tmp_5247[2] + tmp_5249[2]];
    signal tmp_5251[3] <== CMul()(challengeQ, tmp_5250);
    signal tmp_5252[3] <== [evals[71][0] - publics[7], evals[71][1], evals[71][2]];
    signal tmp_5253[3] <== CMul()(evals[61], tmp_5252);
    signal tmp_5254[3] <== [tmp_5251[0] + tmp_5253[0], tmp_5251[1] + tmp_5253[1], tmp_5251[2] + tmp_5253[2]];
    signal tmp_5255[3] <== CMul()(challengeQ, tmp_5254);
    signal tmp_5256[3] <== [evals[72][0] - publics[8], evals[72][1], evals[72][2]];
    signal tmp_5257[3] <== CMul()(evals[61], tmp_5256);
    signal tmp_5258[3] <== [tmp_5255[0] + tmp_5257[0], tmp_5255[1] + tmp_5257[1], tmp_5255[2] + tmp_5257[2]];
    signal tmp_5259[3] <== CMul()(challengeQ, tmp_5258);
    signal tmp_5260[3] <== [evals[73][0] - publics[9], evals[73][1], evals[73][2]];
    signal tmp_5261[3] <== CMul()(evals[61], tmp_5260);
    signal tmp_5262[3] <== [tmp_5259[0] + tmp_5261[0], tmp_5259[1] + tmp_5261[1], tmp_5259[2] + tmp_5261[2]];
    signal tmp_5263[3] <== CMul()(challengeQ, tmp_5262);
    signal tmp_5264[3] <== [evals[74][0] - publics[10], evals[74][1], evals[74][2]];
    signal tmp_5265[3] <== CMul()(evals[61], tmp_5264);
    signal tmp_5266[3] <== [tmp_5263[0] + tmp_5265[0], tmp_5263[1] + tmp_5265[1], tmp_5263[2] + tmp_5265[2]];
    signal tmp_5267[3] <== CMul()(challengeQ, tmp_5266);
    signal tmp_5268[3] <== [evals[75][0] - publics[11], evals[75][1], evals[75][2]];
    signal tmp_5269[3] <== CMul()(evals[61], tmp_5268);
    signal tmp_5270[3] <== [tmp_5267[0] + tmp_5269[0], tmp_5267[1] + tmp_5269[1], tmp_5267[2] + tmp_5269[2]];
    signal tmp_5271[3] <== CMul()(challengeQ, tmp_5270);
    signal tmp_5272[3] <== [evals[76][0] - publics[12], evals[76][1], evals[76][2]];
    signal tmp_5273[3] <== CMul()(evals[61], tmp_5272);
    signal tmp_5274[3] <== [tmp_5271[0] + tmp_5273[0], tmp_5271[1] + tmp_5273[1], tmp_5271[2] + tmp_5273[2]];
    signal tmp_5275[3] <== CMul()(challengeQ, tmp_5274);
    signal tmp_5276[3] <== [evals[77][0] - publics[13], evals[77][1], evals[77][2]];
    signal tmp_5277[3] <== CMul()(evals[61], tmp_5276);
    signal tmp_5278[3] <== [tmp_5275[0] + tmp_5277[0], tmp_5275[1] + tmp_5277[1], tmp_5275[2] + tmp_5277[2]];
    signal tmp_5279[3] <== CMul()(challengeQ, tmp_5278);
    signal tmp_5280[3] <== [evals[78][0] - publics[14], evals[78][1], evals[78][2]];
    signal tmp_5281[3] <== CMul()(evals[61], tmp_5280);
    signal tmp_5282[3] <== [tmp_5279[0] + tmp_5281[0], tmp_5279[1] + tmp_5281[1], tmp_5279[2] + tmp_5281[2]];
    signal tmp_5283[3] <== CMul()(challengeQ, tmp_5282);
    signal tmp_5284[3] <== [evals[79][0] - publics[15], evals[79][1], evals[79][2]];
    signal tmp_5285[3] <== CMul()(evals[61], tmp_5284);
    signal tmp_5286[3] <== [tmp_5283[0] + tmp_5285[0], tmp_5283[1] + tmp_5285[1], tmp_5283[2] + tmp_5285[2]];
    signal tmp_5287[3] <== CMul()(challengeQ, tmp_5286);
    signal tmp_5288[3] <== [evals[80][0] - publics[16], evals[80][1], evals[80][2]];
    signal tmp_5289[3] <== CMul()(evals[61], tmp_5288);
    signal tmp_5290[3] <== [tmp_5287[0] + tmp_5289[0], tmp_5287[1] + tmp_5289[1], tmp_5287[2] + tmp_5289[2]];
    signal tmp_5291[3] <== CMul()(challengeQ, tmp_5290);
    signal tmp_5292[3] <== [evals[81][0] - publics[17], evals[81][1], evals[81][2]];
    signal tmp_5293[3] <== CMul()(evals[61], tmp_5292);
    signal tmp_5294[3] <== [tmp_5291[0] + tmp_5293[0], tmp_5291[1] + tmp_5293[1], tmp_5291[2] + tmp_5293[2]];
    signal tmp_5295[3] <== CMul()(challengeQ, tmp_5294);
    signal tmp_5296[3] <== [evals[82][0] - publics[18], evals[82][1], evals[82][2]];
    signal tmp_5297[3] <== CMul()(evals[61], tmp_5296);
    signal tmp_5298[3] <== [tmp_5295[0] + tmp_5297[0], tmp_5295[1] + tmp_5297[1], tmp_5295[2] + tmp_5297[2]];
    signal tmp_5299[3] <== CMul()(challengeQ, tmp_5298);
    signal tmp_5300[3] <== [evals[83][0] - publics[19], evals[83][1], evals[83][2]];
    signal tmp_5301[3] <== CMul()(evals[61], tmp_5300);
    signal tmp_5302[3] <== [tmp_5299[0] + tmp_5301[0], tmp_5299[1] + tmp_5301[1], tmp_5299[2] + tmp_5301[2]];
    signal tmp_5303[3] <== CMul()(challengeQ, tmp_5302);
    signal tmp_5304[3] <== [evals[84][0] - publics[20], evals[84][1], evals[84][2]];
    signal tmp_5305[3] <== CMul()(evals[61], tmp_5304);
    signal tmp_5306[3] <== [tmp_5303[0] + tmp_5305[0], tmp_5303[1] + tmp_5305[1], tmp_5303[2] + tmp_5305[2]];
    signal tmp_5307[3] <== CMul()(challengeQ, tmp_5306);
    signal tmp_5308[3] <== [evals[85][0] - publics[21], evals[85][1], evals[85][2]];
    signal tmp_5309[3] <== CMul()(evals[61], tmp_5308);
    signal tmp_5310[3] <== [tmp_5307[0] + tmp_5309[0], tmp_5307[1] + tmp_5309[1], tmp_5307[2] + tmp_5309[2]];
    signal tmp_5311[3] <== CMul()(challengeQ, tmp_5310);
    signal tmp_5312[3] <== [evals[86][0] - publics[22], evals[86][1], evals[86][2]];
    signal tmp_5313[3] <== CMul()(evals[61], tmp_5312);
    signal tmp_5314[3] <== [tmp_5311[0] + tmp_5313[0], tmp_5311[1] + tmp_5313[1], tmp_5311[2] + tmp_5313[2]];
    signal tmp_5315[3] <== CMul()(challengeQ, tmp_5314);
    signal tmp_5316[3] <== [evals[87][0] - publics[23], evals[87][1], evals[87][2]];
    signal tmp_5317[3] <== CMul()(evals[61], tmp_5316);
    signal tmp_5318[3] <== [tmp_5315[0] + tmp_5317[0], tmp_5315[1] + tmp_5317[1], tmp_5315[2] + tmp_5317[2]];
    signal tmp_5319[3] <== CMul()(challengeQ, tmp_5318);
    signal tmp_5320[3] <== [evals[64][0] - publics[24], evals[64][1], evals[64][2]];
    signal tmp_5321[3] <== CMul()(evals[15], tmp_5320);
    signal tmp_5322[3] <== [tmp_5319[0] + tmp_5321[0], tmp_5319[1] + tmp_5321[1], tmp_5319[2] + tmp_5321[2]];
    signal tmp_5323[3] <== CMul()(challengeQ, tmp_5322);
    signal tmp_5324[3] <== [evals[65][0] - publics[25], evals[65][1], evals[65][2]];
    signal tmp_5325[3] <== CMul()(evals[15], tmp_5324);
    signal tmp_5326[3] <== [tmp_5323[0] + tmp_5325[0], tmp_5323[1] + tmp_5325[1], tmp_5323[2] + tmp_5325[2]];
    signal tmp_5327[3] <== CMul()(challengeQ, tmp_5326);
    signal tmp_5328[3] <== [evals[66][0] - publics[26], evals[66][1], evals[66][2]];
    signal tmp_5329[3] <== CMul()(evals[15], tmp_5328);
    signal tmp_5330[3] <== [tmp_5327[0] + tmp_5329[0], tmp_5327[1] + tmp_5329[1], tmp_5327[2] + tmp_5329[2]];
    signal tmp_5331[3] <== CMul()(challengeQ, tmp_5330);
    signal tmp_5332[3] <== [evals[67][0] - publics[27], evals[67][1], evals[67][2]];
    signal tmp_5333[3] <== CMul()(evals[15], tmp_5332);
    signal tmp_5334[3] <== [tmp_5331[0] + tmp_5333[0], tmp_5331[1] + tmp_5333[1], tmp_5331[2] + tmp_5333[2]];
    signal tmp_5335[3] <== CMul()(challengeQ, tmp_5334);
    signal tmp_5336[3] <== [evals[68][0] - publics[28], evals[68][1], evals[68][2]];
    signal tmp_5337[3] <== CMul()(evals[15], tmp_5336);
    signal tmp_5338[3] <== [tmp_5335[0] + tmp_5337[0], tmp_5335[1] + tmp_5337[1], tmp_5335[2] + tmp_5337[2]];
    signal tmp_5339[3] <== CMul()(challengeQ, tmp_5338);
    signal tmp_5340[3] <== [evals[69][0] - publics[29], evals[69][1], evals[69][2]];
    signal tmp_5341[3] <== CMul()(evals[15], tmp_5340);
    signal tmp_5342[3] <== [tmp_5339[0] + tmp_5341[0], tmp_5339[1] + tmp_5341[1], tmp_5339[2] + tmp_5341[2]];
    signal tmp_5343[3] <== CMul()(challengeQ, tmp_5342);
    signal tmp_5344[3] <== [evals[70][0] - publics[30], evals[70][1], evals[70][2]];
    signal tmp_5345[3] <== CMul()(evals[15], tmp_5344);
    signal tmp_5346[3] <== [tmp_5343[0] + tmp_5345[0], tmp_5343[1] + tmp_5345[1], tmp_5343[2] + tmp_5345[2]];
    signal tmp_5347[3] <== CMul()(challengeQ, tmp_5346);
    signal tmp_5348[3] <== [evals[71][0] - publics[31], evals[71][1], evals[71][2]];
    signal tmp_5349[3] <== CMul()(evals[15], tmp_5348);
    signal tmp_5350[3] <== [tmp_5347[0] + tmp_5349[0], tmp_5347[1] + tmp_5349[1], tmp_5347[2] + tmp_5349[2]];
    signal tmp_5351[3] <== CMul()(challengeQ, tmp_5350);
    signal tmp_5352[3] <== [evals[72][0] - publics[32], evals[72][1], evals[72][2]];
    signal tmp_5353[3] <== CMul()(evals[15], tmp_5352);
    signal tmp_5354[3] <== [tmp_5351[0] + tmp_5353[0], tmp_5351[1] + tmp_5353[1], tmp_5351[2] + tmp_5353[2]];
    signal tmp_5355[3] <== CMul()(challengeQ, tmp_5354);
    signal tmp_5356[3] <== [evals[73][0] - publics[33], evals[73][1], evals[73][2]];
    signal tmp_5357[3] <== CMul()(evals[15], tmp_5356);
    signal tmp_5358[3] <== [tmp_5355[0] + tmp_5357[0], tmp_5355[1] + tmp_5357[1], tmp_5355[2] + tmp_5357[2]];
    signal tmp_5359[3] <== CMul()(challengeQ, tmp_5358);
    signal tmp_5360[3] <== [evals[74][0] - publics[34], evals[74][1], evals[74][2]];
    signal tmp_5361[3] <== CMul()(evals[15], tmp_5360);
    signal tmp_5362[3] <== [tmp_5359[0] + tmp_5361[0], tmp_5359[1] + tmp_5361[1], tmp_5359[2] + tmp_5361[2]];
    signal tmp_5363[3] <== CMul()(challengeQ, tmp_5362);
    signal tmp_5364[3] <== [evals[75][0] - publics[35], evals[75][1], evals[75][2]];
    signal tmp_5365[3] <== CMul()(evals[15], tmp_5364);
    signal tmp_5366[3] <== [tmp_5363[0] + tmp_5365[0], tmp_5363[1] + tmp_5365[1], tmp_5363[2] + tmp_5365[2]];
    signal tmp_5367[3] <== CMul()(challengeQ, tmp_5366);
    signal tmp_5368[3] <== [evals[76][0] - publics[36], evals[76][1], evals[76][2]];
    signal tmp_5369[3] <== CMul()(evals[15], tmp_5368);
    signal tmp_5370[3] <== [tmp_5367[0] + tmp_5369[0], tmp_5367[1] + tmp_5369[1], tmp_5367[2] + tmp_5369[2]];
    signal tmp_5371[3] <== CMul()(challengeQ, tmp_5370);
    signal tmp_5372[3] <== [evals[77][0] - publics[37], evals[77][1], evals[77][2]];
    signal tmp_5373[3] <== CMul()(evals[15], tmp_5372);
    signal tmp_5374[3] <== [tmp_5371[0] + tmp_5373[0], tmp_5371[1] + tmp_5373[1], tmp_5371[2] + tmp_5373[2]];
    signal tmp_5375[3] <== CMul()(challengeQ, tmp_5374);
    signal tmp_5376[3] <== [evals[78][0] - publics[38], evals[78][1], evals[78][2]];
    signal tmp_5377[3] <== CMul()(evals[15], tmp_5376);
    signal tmp_5378[3] <== [tmp_5375[0] + tmp_5377[0], tmp_5375[1] + tmp_5377[1], tmp_5375[2] + tmp_5377[2]];
    signal tmp_5379[3] <== CMul()(challengeQ, tmp_5378);
    signal tmp_5380[3] <== [evals[79][0] - publics[39], evals[79][1], evals[79][2]];
    signal tmp_5381[3] <== CMul()(evals[15], tmp_5380);
    signal tmp_5382[3] <== [tmp_5379[0] + tmp_5381[0], tmp_5379[1] + tmp_5381[1], tmp_5379[2] + tmp_5381[2]];
    signal tmp_5383[3] <== CMul()(challengeQ, tmp_5382);
    signal tmp_5384[3] <== [evals[80][0] - publics[40], evals[80][1], evals[80][2]];
    signal tmp_5385[3] <== CMul()(evals[15], tmp_5384);
    signal tmp_5386[3] <== [tmp_5383[0] + tmp_5385[0], tmp_5383[1] + tmp_5385[1], tmp_5383[2] + tmp_5385[2]];
    signal tmp_5387[3] <== CMul()(challengeQ, tmp_5386);
    signal tmp_5388[3] <== [evals[81][0] - publics[41], evals[81][1], evals[81][2]];
    signal tmp_5389[3] <== CMul()(evals[15], tmp_5388);
    signal tmp_5390[3] <== [tmp_5387[0] + tmp_5389[0], tmp_5387[1] + tmp_5389[1], tmp_5387[2] + tmp_5389[2]];
    signal tmp_5391[3] <== CMul()(challengeQ, tmp_5390);
    signal tmp_5392[3] <== [evals[82][0] - publics[42], evals[82][1], evals[82][2]];
    signal tmp_5393[3] <== CMul()(evals[15], tmp_5392);
    signal tmp_5394[3] <== [tmp_5391[0] + tmp_5393[0], tmp_5391[1] + tmp_5393[1], tmp_5391[2] + tmp_5393[2]];
    signal tmp_5395[3] <== CMul()(challengeQ, tmp_5394);
    signal tmp_5396[3] <== [evals[83][0] - publics[43], evals[83][1], evals[83][2]];
    signal tmp_5397[3] <== CMul()(evals[15], tmp_5396);
    signal tmp_5398[3] <== [tmp_5395[0] + tmp_5397[0], tmp_5395[1] + tmp_5397[1], tmp_5395[2] + tmp_5397[2]];
    signal tmp_5399[3] <== CMul()(challengeQ, tmp_5398);
    signal tmp_5400[3] <== [evals[84][0] - publics[44], evals[84][1], evals[84][2]];
    signal tmp_5401[3] <== CMul()(evals[15], tmp_5400);
    signal tmp_5402[3] <== [tmp_5399[0] + tmp_5401[0], tmp_5399[1] + tmp_5401[1], tmp_5399[2] + tmp_5401[2]];
    signal tmp_5403[3] <== CMul()(challengeQ, tmp_5402);
    signal tmp_5404[3] <== [evals[85][0] - publics[45], evals[85][1], evals[85][2]];
    signal tmp_5405[3] <== CMul()(evals[15], tmp_5404);
    signal tmp_5406[3] <== [tmp_5403[0] + tmp_5405[0], tmp_5403[1] + tmp_5405[1], tmp_5403[2] + tmp_5405[2]];
    signal tmp_5407[3] <== CMul()(challengeQ, tmp_5406);
    signal tmp_5408[3] <== [evals[86][0] - publics[46], evals[86][1], evals[86][2]];
    signal tmp_5409[3] <== CMul()(evals[15], tmp_5408);
    signal tmp_5410[3] <== [tmp_5407[0] + tmp_5409[0], tmp_5407[1] + tmp_5409[1], tmp_5407[2] + tmp_5409[2]];
    signal tmp_5411[3] <== CMul()(challengeQ, tmp_5410);
    signal tmp_5412[3] <== [evals[87][0] - publics[47], evals[87][1], evals[87][2]];
    signal tmp_5413[3] <== CMul()(evals[15], tmp_5412);
    signal tmp_5414[3] <== [tmp_5411[0] + tmp_5413[0], tmp_5411[1] + tmp_5413[1], tmp_5411[2] + tmp_5413[2]];
    signal tmp_5415[3] <== CMul()(challengeQ, tmp_5414);
    signal tmp_5416[3] <== [evals[64][0] - publics[48], evals[64][1], evals[64][2]];
    signal tmp_5417[3] <== CMul()(evals[14], tmp_5416);
    signal tmp_5418[3] <== [tmp_5415[0] + tmp_5417[0], tmp_5415[1] + tmp_5417[1], tmp_5415[2] + tmp_5417[2]];
    signal tmp_5419[3] <== CMul()(challengeQ, tmp_5418);
    signal tmp_5420[3] <== [evals[65][0] - publics[49], evals[65][1], evals[65][2]];
    signal tmp_5421[3] <== CMul()(evals[14], tmp_5420);
    signal tmp_5422[3] <== [tmp_5419[0] + tmp_5421[0], tmp_5419[1] + tmp_5421[1], tmp_5419[2] + tmp_5421[2]];
    signal tmp_5423[3] <== CMul()(challengeQ, tmp_5422);
    signal tmp_5424[3] <== [evals[66][0] - publics[50], evals[66][1], evals[66][2]];
    signal tmp_5425[3] <== CMul()(evals[14], tmp_5424);
    signal tmp_5426[3] <== [tmp_5423[0] + tmp_5425[0], tmp_5423[1] + tmp_5425[1], tmp_5423[2] + tmp_5425[2]];
    signal tmp_5427[3] <== CMul()(challengeQ, tmp_5426);
    signal tmp_5428[3] <== [evals[67][0] - publics[51], evals[67][1], evals[67][2]];
    signal tmp_5429[3] <== CMul()(evals[14], tmp_5428);
    signal tmp_5430[3] <== [tmp_5427[0] + tmp_5429[0], tmp_5427[1] + tmp_5429[1], tmp_5427[2] + tmp_5429[2]];
    signal tmp_5431[3] <== CMul()(challengeQ, tmp_5430);
    signal tmp_5432[3] <== [evals[68][0] - publics[52], evals[68][1], evals[68][2]];
    signal tmp_5433[3] <== CMul()(evals[14], tmp_5432);
    signal tmp_5434[3] <== [tmp_5431[0] + tmp_5433[0], tmp_5431[1] + tmp_5433[1], tmp_5431[2] + tmp_5433[2]];
    signal tmp_5435[3] <== CMul()(challengeQ, tmp_5434);
    signal tmp_5436[3] <== [evals[69][0] - publics[53], evals[69][1], evals[69][2]];
    signal tmp_5437[3] <== CMul()(evals[14], tmp_5436);
    signal tmp_5438[3] <== [tmp_5435[0] + tmp_5437[0], tmp_5435[1] + tmp_5437[1], tmp_5435[2] + tmp_5437[2]];
    signal tmp_5439[3] <== CMul()(challengeQ, tmp_5438);
    signal tmp_5440[3] <== [evals[70][0] - publics[54], evals[70][1], evals[70][2]];
    signal tmp_5441[3] <== CMul()(evals[14], tmp_5440);
    signal tmp_5442[3] <== [tmp_5439[0] + tmp_5441[0], tmp_5439[1] + tmp_5441[1], tmp_5439[2] + tmp_5441[2]];
    signal tmp_5443[3] <== CMul()(challengeQ, tmp_5442);
    signal tmp_5444[3] <== [evals[71][0] - publics[55], evals[71][1], evals[71][2]];
    signal tmp_5445[3] <== CMul()(evals[14], tmp_5444);
    signal tmp_5446[3] <== [tmp_5443[0] + tmp_5445[0], tmp_5443[1] + tmp_5445[1], tmp_5443[2] + tmp_5445[2]];
    signal tmp_5447[3] <== CMul()(challengeQ, tmp_5446);
    signal tmp_5448[3] <== [evals[72][0] - publics[56], evals[72][1], evals[72][2]];
    signal tmp_5449[3] <== CMul()(evals[14], tmp_5448);
    signal tmp_5450[3] <== [tmp_5447[0] + tmp_5449[0], tmp_5447[1] + tmp_5449[1], tmp_5447[2] + tmp_5449[2]];
    signal tmp_5451[3] <== CMul()(challengeQ, tmp_5450);
    signal tmp_5452[3] <== [evals[73][0] - publics[57], evals[73][1], evals[73][2]];
    signal tmp_5453[3] <== CMul()(evals[14], tmp_5452);
    signal tmp_5454[3] <== [tmp_5451[0] + tmp_5453[0], tmp_5451[1] + tmp_5453[1], tmp_5451[2] + tmp_5453[2]];
    signal tmp_5455[3] <== CMul()(challengeQ, tmp_5454);
    signal tmp_5456[3] <== [evals[74][0] - publics[58], evals[74][1], evals[74][2]];
    signal tmp_5457[3] <== CMul()(evals[14], tmp_5456);
    signal tmp_5458[3] <== [tmp_5455[0] + tmp_5457[0], tmp_5455[1] + tmp_5457[1], tmp_5455[2] + tmp_5457[2]];
    signal tmp_5459[3] <== CMul()(challengeQ, tmp_5458);
    signal tmp_5460[3] <== [evals[75][0] - publics[59], evals[75][1], evals[75][2]];
    signal tmp_5461[3] <== CMul()(evals[14], tmp_5460);
    signal tmp_5462[3] <== [tmp_5459[0] + tmp_5461[0], tmp_5459[1] + tmp_5461[1], tmp_5459[2] + tmp_5461[2]];
    signal tmp_5463[3] <== CMul()(challengeQ, tmp_5462);
    signal tmp_5464[3] <== [evals[76][0] - publics[60], evals[76][1], evals[76][2]];
    signal tmp_5465[3] <== CMul()(evals[14], tmp_5464);
    signal tmp_5466[3] <== [tmp_5463[0] + tmp_5465[0], tmp_5463[1] + tmp_5465[1], tmp_5463[2] + tmp_5465[2]];
    signal tmp_5467[3] <== CMul()(challengeQ, tmp_5466);
    signal tmp_5468[3] <== [evals[77][0] - publics[61], evals[77][1], evals[77][2]];
    signal tmp_5469[3] <== CMul()(evals[14], tmp_5468);
    signal tmp_5470[3] <== [tmp_5467[0] + tmp_5469[0], tmp_5467[1] + tmp_5469[1], tmp_5467[2] + tmp_5469[2]];
    signal tmp_5471[3] <== CMul()(challengeQ, tmp_5470);
    signal tmp_5472[3] <== [evals[78][0] - publics[62], evals[78][1], evals[78][2]];
    signal tmp_5473[3] <== CMul()(evals[14], tmp_5472);
    signal tmp_5474[3] <== [tmp_5471[0] + tmp_5473[0], tmp_5471[1] + tmp_5473[1], tmp_5471[2] + tmp_5473[2]];
    signal tmp_5475[3] <== CMul()(challengeQ, tmp_5474);
    signal tmp_5476[3] <== [evals[79][0] - publics[63], evals[79][1], evals[79][2]];
    signal tmp_5477[3] <== CMul()(evals[14], tmp_5476);
    signal tmp_5478[3] <== [tmp_5475[0] + tmp_5477[0], tmp_5475[1] + tmp_5477[1], tmp_5475[2] + tmp_5477[2]];
    signal tmp_5479[3] <== CMul()(challengeQ, tmp_5478);
    signal tmp_5480[3] <== [evals[80][0] - publics[64], evals[80][1], evals[80][2]];
    signal tmp_5481[3] <== CMul()(evals[14], tmp_5480);
    signal tmp_5482[3] <== [tmp_5479[0] + tmp_5481[0], tmp_5479[1] + tmp_5481[1], tmp_5479[2] + tmp_5481[2]];
    signal tmp_5483[3] <== CMul()(challengeQ, tmp_5482);
    signal tmp_5484[3] <== [evals[81][0] - publics[65], evals[81][1], evals[81][2]];
    signal tmp_5485[3] <== CMul()(evals[14], tmp_5484);
    signal tmp_5486[3] <== [tmp_5483[0] + tmp_5485[0], tmp_5483[1] + tmp_5485[1], tmp_5483[2] + tmp_5485[2]];
    signal tmp_5487[3] <== CMul()(challengeQ, tmp_5486);
    signal tmp_5488[3] <== [evals[82][0] - publics[66], evals[82][1], evals[82][2]];
    signal tmp_5489[3] <== CMul()(evals[14], tmp_5488);
    signal tmp_5490[3] <== [tmp_5487[0] + tmp_5489[0], tmp_5487[1] + tmp_5489[1], tmp_5487[2] + tmp_5489[2]];
    signal tmp_5491[3] <== CMul()(challengeQ, tmp_5490);
    signal tmp_5492[3] <== [evals[83][0] - publics[67], evals[83][1], evals[83][2]];
    signal tmp_5493[3] <== CMul()(evals[14], tmp_5492);
    signal tmp_5494[3] <== [tmp_5491[0] + tmp_5493[0], tmp_5491[1] + tmp_5493[1], tmp_5491[2] + tmp_5493[2]];
    signal tmp_5495[3] <== CMul()(challengeQ, tmp_5494);
    signal tmp_5496[3] <== [evals[84][0] - publics[68], evals[84][1], evals[84][2]];
    signal tmp_5497[3] <== CMul()(evals[14], tmp_5496);
    signal tmp_5498[3] <== [tmp_5495[0] + tmp_5497[0], tmp_5495[1] + tmp_5497[1], tmp_5495[2] + tmp_5497[2]];
    signal tmp_5499[3] <== CMul()(challengeQ, tmp_5498);
    signal tmp_5500[3] <== [evals[85][0] - publics[69], evals[85][1], evals[85][2]];
    signal tmp_5501[3] <== CMul()(evals[14], tmp_5500);
    signal tmp_5502[3] <== [tmp_5499[0] + tmp_5501[0], tmp_5499[1] + tmp_5501[1], tmp_5499[2] + tmp_5501[2]];
    signal tmp_5503[3] <== CMul()(challengeQ, tmp_5502);
    signal tmp_5504[3] <== [evals[86][0] - publics[70], evals[86][1], evals[86][2]];
    signal tmp_5505[3] <== CMul()(evals[14], tmp_5504);
    signal tmp_5506[3] <== [tmp_5503[0] + tmp_5505[0], tmp_5503[1] + tmp_5505[1], tmp_5503[2] + tmp_5505[2]];
    signal tmp_5507[3] <== CMul()(challengeQ, tmp_5506);
    signal tmp_5508[3] <== [evals[87][0] - publics[71], evals[87][1], evals[87][2]];
    signal tmp_5509[3] <== CMul()(evals[14], tmp_5508);
    signal tmp_5510[3] <== [tmp_5507[0] + tmp_5509[0], tmp_5507[1] + tmp_5509[1], tmp_5507[2] + tmp_5509[2]];
    signal tmp_5511[3] <== CMul()(challengeQ, tmp_5510);
    signal tmp_5512[3] <== [evals[64][0] - publics[72], evals[64][1], evals[64][2]];
    signal tmp_5513[3] <== CMul()(evals[13], tmp_5512);
    signal tmp_5514[3] <== [tmp_5511[0] + tmp_5513[0], tmp_5511[1] + tmp_5513[1], tmp_5511[2] + tmp_5513[2]];
    signal tmp_5515[3] <== CMul()(challengeQ, tmp_5514);
    signal tmp_5516[3] <== [evals[65][0] - publics[73], evals[65][1], evals[65][2]];
    signal tmp_5517[3] <== CMul()(evals[13], tmp_5516);
    signal tmp_5518[3] <== [tmp_5515[0] + tmp_5517[0], tmp_5515[1] + tmp_5517[1], tmp_5515[2] + tmp_5517[2]];
    signal tmp_5519[3] <== CMul()(challengeQ, tmp_5518);
    signal tmp_5520[3] <== [evals[66][0] - publics[74], evals[66][1], evals[66][2]];
    signal tmp_5521[3] <== CMul()(evals[13], tmp_5520);
    signal tmp_5522[3] <== [tmp_5519[0] + tmp_5521[0], tmp_5519[1] + tmp_5521[1], tmp_5519[2] + tmp_5521[2]];
    signal tmp_5523[3] <== CMul()(challengeQ, tmp_5522);
    signal tmp_5524[3] <== [evals[67][0] - publics[75], evals[67][1], evals[67][2]];
    signal tmp_5525[3] <== CMul()(evals[13], tmp_5524);
    signal tmp_5526[3] <== [tmp_5523[0] + tmp_5525[0], tmp_5523[1] + tmp_5525[1], tmp_5523[2] + tmp_5525[2]];
    signal tmp_5527[3] <== CMul()(challengeQ, tmp_5526);
    signal tmp_5528[3] <== [evals[68][0] - publics[76], evals[68][1], evals[68][2]];
    signal tmp_5529[3] <== CMul()(evals[13], tmp_5528);
    signal tmp_5530[3] <== [tmp_5527[0] + tmp_5529[0], tmp_5527[1] + tmp_5529[1], tmp_5527[2] + tmp_5529[2]];
    signal tmp_5531[3] <== CMul()(challengeQ, tmp_5530);
    signal tmp_5532[3] <== [evals[69][0] - publics[77], evals[69][1], evals[69][2]];
    signal tmp_5533[3] <== CMul()(evals[13], tmp_5532);
    signal tmp_5534[3] <== [tmp_5531[0] + tmp_5533[0], tmp_5531[1] + tmp_5533[1], tmp_5531[2] + tmp_5533[2]];
    signal tmp_5535[3] <== CMul()(challengeQ, tmp_5534);
    signal tmp_5536[3] <== [evals[70][0] - publics[78], evals[70][1], evals[70][2]];
    signal tmp_5537[3] <== CMul()(evals[13], tmp_5536);
    signal tmp_5538[3] <== [tmp_5535[0] + tmp_5537[0], tmp_5535[1] + tmp_5537[1], tmp_5535[2] + tmp_5537[2]];
    signal tmp_5539[3] <== CMul()(challengeQ, tmp_5538);
    signal tmp_5540[3] <== [evals[71][0] - publics[79], evals[71][1], evals[71][2]];
    signal tmp_5541[3] <== CMul()(evals[13], tmp_5540);
    signal tmp_5542[3] <== [tmp_5539[0] + tmp_5541[0], tmp_5539[1] + tmp_5541[1], tmp_5539[2] + tmp_5541[2]];
    signal tmp_5543[3] <== CMul()(challengeQ, tmp_5542);
    signal tmp_5544[3] <== [evals[72][0] - publics[80], evals[72][1], evals[72][2]];
    signal tmp_5545[3] <== CMul()(evals[13], tmp_5544);
    signal tmp_5546[3] <== [tmp_5543[0] + tmp_5545[0], tmp_5543[1] + tmp_5545[1], tmp_5543[2] + tmp_5545[2]];
    signal tmp_5547[3] <== CMul()(challengeQ, tmp_5546);
    signal tmp_5548[3] <== [evals[73][0] - publics[81], evals[73][1], evals[73][2]];
    signal tmp_5549[3] <== CMul()(evals[13], tmp_5548);
    signal tmp_5550[3] <== [tmp_5547[0] + tmp_5549[0], tmp_5547[1] + tmp_5549[1], tmp_5547[2] + tmp_5549[2]];
    signal tmp_5551[3] <== CMul()(challengeQ, tmp_5550);
    signal tmp_5552[3] <== [evals[74][0] - publics[82], evals[74][1], evals[74][2]];
    signal tmp_5553[3] <== CMul()(evals[13], tmp_5552);
    signal tmp_5554[3] <== [tmp_5551[0] + tmp_5553[0], tmp_5551[1] + tmp_5553[1], tmp_5551[2] + tmp_5553[2]];
    signal tmp_5555[3] <== CMul()(challengeQ, tmp_5554);
    signal tmp_5556[3] <== [evals[75][0] - publics[83], evals[75][1], evals[75][2]];
    signal tmp_5557[3] <== CMul()(evals[13], tmp_5556);
    signal tmp_5558[3] <== [tmp_5555[0] + tmp_5557[0], tmp_5555[1] + tmp_5557[1], tmp_5555[2] + tmp_5557[2]];
    signal tmp_5559[3] <== CMul()(challengeQ, tmp_5558);
    signal tmp_5560[3] <== [evals[76][0] - publics[84], evals[76][1], evals[76][2]];
    signal tmp_5561[3] <== CMul()(evals[13], tmp_5560);
    signal tmp_5562[3] <== [tmp_5559[0] + tmp_5561[0], tmp_5559[1] + tmp_5561[1], tmp_5559[2] + tmp_5561[2]];
    signal tmp_5563[3] <== CMul()(challengeQ, tmp_5562);
    signal tmp_5564[3] <== [evals[77][0] - publics[85], evals[77][1], evals[77][2]];
    signal tmp_5565[3] <== CMul()(evals[13], tmp_5564);
    signal tmp_5566[3] <== [tmp_5563[0] + tmp_5565[0], tmp_5563[1] + tmp_5565[1], tmp_5563[2] + tmp_5565[2]];
    signal tmp_5567[3] <== CMul()(challengeQ, tmp_5566);
    signal tmp_5568[3] <== [evals[78][0] - publics[86], evals[78][1], evals[78][2]];
    signal tmp_5569[3] <== CMul()(evals[13], tmp_5568);
    signal tmp_5570[3] <== [tmp_5567[0] + tmp_5569[0], tmp_5567[1] + tmp_5569[1], tmp_5567[2] + tmp_5569[2]];
    signal tmp_5571[3] <== CMul()(challengeQ, tmp_5570);
    signal tmp_5572[3] <== [evals[79][0] - publics[87], evals[79][1], evals[79][2]];
    signal tmp_5573[3] <== CMul()(evals[13], tmp_5572);
    signal tmp_5574[3] <== [tmp_5571[0] + tmp_5573[0], tmp_5571[1] + tmp_5573[1], tmp_5571[2] + tmp_5573[2]];
    signal tmp_5575[3] <== CMul()(challengeQ, tmp_5574);
    signal tmp_5576[3] <== [evals[80][0] - publics[88], evals[80][1], evals[80][2]];
    signal tmp_5577[3] <== CMul()(evals[13], tmp_5576);
    signal tmp_5578[3] <== [tmp_5575[0] + tmp_5577[0], tmp_5575[1] + tmp_5577[1], tmp_5575[2] + tmp_5577[2]];
    signal tmp_5579[3] <== CMul()(challengeQ, tmp_5578);
    signal tmp_5580[3] <== [evals[81][0] - publics[89], evals[81][1], evals[81][2]];
    signal tmp_5581[3] <== CMul()(evals[13], tmp_5580);
    signal tmp_5582[3] <== [tmp_5579[0] + tmp_5581[0], tmp_5579[1] + tmp_5581[1], tmp_5579[2] + tmp_5581[2]];
    signal tmp_5583[3] <== CMul()(challengeQ, tmp_5582);
    signal tmp_5584[3] <== [evals[82][0] - publics[90], evals[82][1], evals[82][2]];
    signal tmp_5585[3] <== CMul()(evals[13], tmp_5584);
    signal tmp_5586[3] <== [tmp_5583[0] + tmp_5585[0], tmp_5583[1] + tmp_5585[1], tmp_5583[2] + tmp_5585[2]];
    signal tmp_5587[3] <== CMul()(challengeQ, tmp_5586);
    signal tmp_5588[3] <== [evals[83][0] - publics[91], evals[83][1], evals[83][2]];
    signal tmp_5589[3] <== CMul()(evals[13], tmp_5588);
    signal tmp_5590[3] <== [tmp_5587[0] + tmp_5589[0], tmp_5587[1] + tmp_5589[1], tmp_5587[2] + tmp_5589[2]];
    signal tmp_5591[3] <== CMul()(challengeQ, tmp_5590);
    signal tmp_5592[3] <== [evals[84][0] - publics[92], evals[84][1], evals[84][2]];
    signal tmp_5593[3] <== CMul()(evals[13], tmp_5592);
    signal tmp_5594[3] <== [tmp_5591[0] + tmp_5593[0], tmp_5591[1] + tmp_5593[1], tmp_5591[2] + tmp_5593[2]];
    signal tmp_5595[3] <== CMul()(challengeQ, tmp_5594);
    signal tmp_5596[3] <== [evals[85][0] - publics[93], evals[85][1], evals[85][2]];
    signal tmp_5597[3] <== CMul()(evals[13], tmp_5596);
    signal tmp_5598[3] <== [tmp_5595[0] + tmp_5597[0], tmp_5595[1] + tmp_5597[1], tmp_5595[2] + tmp_5597[2]];
    signal tmp_5599[3] <== CMul()(challengeQ, tmp_5598);
    signal tmp_5600[3] <== [evals[86][0] - publics[94], evals[86][1], evals[86][2]];
    signal tmp_5601[3] <== CMul()(evals[13], tmp_5600);
    signal tmp_5602[3] <== [tmp_5599[0] + tmp_5601[0], tmp_5599[1] + tmp_5601[1], tmp_5599[2] + tmp_5601[2]];
    signal tmp_5603[3] <== CMul()(challengeQ, tmp_5602);
    signal tmp_5604[3] <== [evals[87][0] - publics[95], evals[87][1], evals[87][2]];
    signal tmp_5605[3] <== CMul()(evals[13], tmp_5604);
    signal tmp_5606[3] <== [tmp_5603[0] + tmp_5605[0], tmp_5603[1] + tmp_5605[1], tmp_5603[2] + tmp_5605[2]];
    signal tmp_5607[3] <== CMul()(challengeQ, tmp_5606);
    signal tmp_5608[3] <== [evals[64][0] - publics[96], evals[64][1], evals[64][2]];
    signal tmp_5609[3] <== CMul()(evals[12], tmp_5608);
    signal tmp_5610[3] <== [tmp_5607[0] + tmp_5609[0], tmp_5607[1] + tmp_5609[1], tmp_5607[2] + tmp_5609[2]];
    signal tmp_5611[3] <== CMul()(challengeQ, tmp_5610);
    signal tmp_5612[3] <== [evals[65][0] - publics[97], evals[65][1], evals[65][2]];
    signal tmp_5613[3] <== CMul()(evals[12], tmp_5612);
    signal tmp_5614[3] <== [tmp_5611[0] + tmp_5613[0], tmp_5611[1] + tmp_5613[1], tmp_5611[2] + tmp_5613[2]];
    signal tmp_5615[3] <== CMul()(challengeQ, tmp_5614);
    signal tmp_5616[3] <== [evals[66][0] - publics[98], evals[66][1], evals[66][2]];
    signal tmp_5617[3] <== CMul()(evals[12], tmp_5616);
    signal tmp_5618[3] <== [tmp_5615[0] + tmp_5617[0], tmp_5615[1] + tmp_5617[1], tmp_5615[2] + tmp_5617[2]];
    signal tmp_5619[3] <== CMul()(challengeQ, tmp_5618);
    signal tmp_5620[3] <== [evals[67][0] - publics[99], evals[67][1], evals[67][2]];
    signal tmp_5621[3] <== CMul()(evals[12], tmp_5620);
    signal tmp_5622[3] <== [tmp_5619[0] + tmp_5621[0], tmp_5619[1] + tmp_5621[1], tmp_5619[2] + tmp_5621[2]];
    signal tmp_5623[3] <== CMul()(challengeQ, tmp_5622);
    signal tmp_5624[3] <== [evals[68][0] - publics[100], evals[68][1], evals[68][2]];
    signal tmp_5625[3] <== CMul()(evals[12], tmp_5624);
    signal tmp_5626[3] <== [tmp_5623[0] + tmp_5625[0], tmp_5623[1] + tmp_5625[1], tmp_5623[2] + tmp_5625[2]];
    signal tmp_5627[3] <== CMul()(challengeQ, tmp_5626);
    signal tmp_5628[3] <== [evals[69][0] - publics[101], evals[69][1], evals[69][2]];
    signal tmp_5629[3] <== CMul()(evals[12], tmp_5628);
    signal tmp_5630[3] <== [tmp_5627[0] + tmp_5629[0], tmp_5627[1] + tmp_5629[1], tmp_5627[2] + tmp_5629[2]];
    signal tmp_5631[3] <== CMul()(challengeQ, tmp_5630);
    signal tmp_5632[3] <== [evals[70][0] - publics[102], evals[70][1], evals[70][2]];
    signal tmp_5633[3] <== CMul()(evals[12], tmp_5632);
    signal tmp_5634[3] <== [tmp_5631[0] + tmp_5633[0], tmp_5631[1] + tmp_5633[1], tmp_5631[2] + tmp_5633[2]];
    signal tmp_5635[3] <== CMul()(challengeQ, tmp_5634);
    signal tmp_5636[3] <== [evals[71][0] - publics[103], evals[71][1], evals[71][2]];
    signal tmp_5637[3] <== CMul()(evals[12], tmp_5636);
    signal tmp_5638[3] <== [tmp_5635[0] + tmp_5637[0], tmp_5635[1] + tmp_5637[1], tmp_5635[2] + tmp_5637[2]];
    signal tmp_5639[3] <== CMul()(challengeQ, tmp_5638);
    signal tmp_5640[3] <== [evals[72][0] - publics[104], evals[72][1], evals[72][2]];
    signal tmp_5641[3] <== CMul()(evals[12], tmp_5640);
    signal tmp_5642[3] <== [tmp_5639[0] + tmp_5641[0], tmp_5639[1] + tmp_5641[1], tmp_5639[2] + tmp_5641[2]];
    signal tmp_5643[3] <== CMul()(challengeQ, tmp_5642);
    signal tmp_5644[3] <== [evals[73][0] - publics[105], evals[73][1], evals[73][2]];
    signal tmp_5645[3] <== CMul()(evals[12], tmp_5644);
    signal tmp_5646[3] <== [tmp_5643[0] + tmp_5645[0], tmp_5643[1] + tmp_5645[1], tmp_5643[2] + tmp_5645[2]];
    signal tmp_5647[3] <== CMul()(challengeQ, tmp_5646);
    signal tmp_5648[3] <== [evals[74][0] - publics[106], evals[74][1], evals[74][2]];
    signal tmp_5649[3] <== CMul()(evals[12], tmp_5648);
    signal tmp_5650[3] <== [tmp_5647[0] + tmp_5649[0], tmp_5647[1] + tmp_5649[1], tmp_5647[2] + tmp_5649[2]];
    signal tmp_5651[3] <== CMul()(challengeQ, tmp_5650);
    signal tmp_5652[3] <== [evals[75][0] - publics[107], evals[75][1], evals[75][2]];
    signal tmp_5653[3] <== CMul()(evals[12], tmp_5652);
    signal tmp_5654[3] <== [tmp_5651[0] + tmp_5653[0], tmp_5651[1] + tmp_5653[1], tmp_5651[2] + tmp_5653[2]];
    signal tmp_5655[3] <== CMul()(challengeQ, tmp_5654);
    signal tmp_5656[3] <== [evals[76][0] - publics[108], evals[76][1], evals[76][2]];
    signal tmp_5657[3] <== CMul()(evals[12], tmp_5656);
    signal tmp_5658[3] <== [tmp_5655[0] + tmp_5657[0], tmp_5655[1] + tmp_5657[1], tmp_5655[2] + tmp_5657[2]];
    signal tmp_5659[3] <== CMul()(challengeQ, tmp_5658);
    signal tmp_5660[3] <== [evals[77][0] - publics[109], evals[77][1], evals[77][2]];
    signal tmp_5661[3] <== CMul()(evals[12], tmp_5660);
    signal tmp_5662[3] <== [tmp_5659[0] + tmp_5661[0], tmp_5659[1] + tmp_5661[1], tmp_5659[2] + tmp_5661[2]];
    signal tmp_5663[3] <== CMul()(challengeQ, tmp_5662);
    signal tmp_5664[3] <== [evals[78][0] - publics[110], evals[78][1], evals[78][2]];
    signal tmp_5665[3] <== CMul()(evals[12], tmp_5664);
    signal tmp_5666[3] <== [tmp_5663[0] + tmp_5665[0], tmp_5663[1] + tmp_5665[1], tmp_5663[2] + tmp_5665[2]];
    signal tmp_5667[3] <== CMul()(challengeQ, tmp_5666);
    signal tmp_5668[3] <== [evals[79][0] - publics[111], evals[79][1], evals[79][2]];
    signal tmp_5669[3] <== CMul()(evals[12], tmp_5668);
    signal tmp_5670[3] <== [tmp_5667[0] + tmp_5669[0], tmp_5667[1] + tmp_5669[1], tmp_5667[2] + tmp_5669[2]];
    signal tmp_5671[3] <== CMul()(challengeQ, tmp_5670);
    signal tmp_5672[3] <== [evals[80][0] - publics[112], evals[80][1], evals[80][2]];
    signal tmp_5673[3] <== CMul()(evals[12], tmp_5672);
    signal tmp_5674[3] <== [tmp_5671[0] + tmp_5673[0], tmp_5671[1] + tmp_5673[1], tmp_5671[2] + tmp_5673[2]];
    signal tmp_5675[3] <== CMul()(challengeQ, tmp_5674);
    signal tmp_5676[3] <== [evals[81][0] - publics[113], evals[81][1], evals[81][2]];
    signal tmp_5677[3] <== CMul()(evals[12], tmp_5676);
    signal tmp_5678[3] <== [tmp_5675[0] + tmp_5677[0], tmp_5675[1] + tmp_5677[1], tmp_5675[2] + tmp_5677[2]];
    signal tmp_5679[3] <== CMul()(challengeQ, tmp_5678);
    signal tmp_5680[3] <== [evals[82][0] - publics[114], evals[82][1], evals[82][2]];
    signal tmp_5681[3] <== CMul()(evals[12], tmp_5680);
    signal tmp_5682[3] <== [tmp_5679[0] + tmp_5681[0], tmp_5679[1] + tmp_5681[1], tmp_5679[2] + tmp_5681[2]];
    signal tmp_5683[3] <== CMul()(challengeQ, tmp_5682);
    signal tmp_5684[3] <== [evals[83][0] - publics[115], evals[83][1], evals[83][2]];
    signal tmp_5685[3] <== CMul()(evals[12], tmp_5684);
    signal tmp_5686[3] <== [tmp_5683[0] + tmp_5685[0], tmp_5683[1] + tmp_5685[1], tmp_5683[2] + tmp_5685[2]];
    signal tmp_5687[3] <== CMul()(challengeQ, tmp_5686);
    signal tmp_5688[3] <== [evals[84][0] - publics[116], evals[84][1], evals[84][2]];
    signal tmp_5689[3] <== CMul()(evals[12], tmp_5688);
    signal tmp_5690[3] <== [tmp_5687[0] + tmp_5689[0], tmp_5687[1] + tmp_5689[1], tmp_5687[2] + tmp_5689[2]];
    signal tmp_5691[3] <== CMul()(challengeQ, tmp_5690);
    signal tmp_5692[3] <== [evals[85][0] - publics[117], evals[85][1], evals[85][2]];
    signal tmp_5693[3] <== CMul()(evals[12], tmp_5692);
    signal tmp_5694[3] <== [tmp_5691[0] + tmp_5693[0], tmp_5691[1] + tmp_5693[1], tmp_5691[2] + tmp_5693[2]];
    signal tmp_5695[3] <== CMul()(challengeQ, tmp_5694);
    signal tmp_5696[3] <== [evals[86][0] - publics[118], evals[86][1], evals[86][2]];
    signal tmp_5697[3] <== CMul()(evals[12], tmp_5696);
    signal tmp_5698[3] <== [tmp_5695[0] + tmp_5697[0], tmp_5695[1] + tmp_5697[1], tmp_5695[2] + tmp_5697[2]];
    signal tmp_5699[3] <== CMul()(challengeQ, tmp_5698);
    signal tmp_5700[3] <== [evals[87][0] - publics[119], evals[87][1], evals[87][2]];
    signal tmp_5701[3] <== CMul()(evals[12], tmp_5700);
    signal tmp_5702[3] <== [tmp_5699[0] + tmp_5701[0], tmp_5699[1] + tmp_5701[1], tmp_5699[2] + tmp_5701[2]];
    signal tmp_5703[3] <== CMul()(challengeQ, tmp_5702);
    signal tmp_5704[3] <== [evals[64][0] - publics[120], evals[64][1], evals[64][2]];
    signal tmp_5705[3] <== CMul()(evals[11], tmp_5704);
    signal tmp_5706[3] <== [tmp_5703[0] + tmp_5705[0], tmp_5703[1] + tmp_5705[1], tmp_5703[2] + tmp_5705[2]];
    signal tmp_5707[3] <== CMul()(challengeQ, tmp_5706);
    signal tmp_5708[3] <== [evals[65][0] - publics[121], evals[65][1], evals[65][2]];
    signal tmp_5709[3] <== CMul()(evals[11], tmp_5708);
    signal tmp_5710[3] <== [tmp_5707[0] + tmp_5709[0], tmp_5707[1] + tmp_5709[1], tmp_5707[2] + tmp_5709[2]];
    signal tmp_5711[3] <== CMul()(challengeQ, tmp_5710);
    signal tmp_5712[3] <== [evals[66][0] - publics[122], evals[66][1], evals[66][2]];
    signal tmp_5713[3] <== CMul()(evals[11], tmp_5712);
    signal tmp_5714[3] <== [tmp_5711[0] + tmp_5713[0], tmp_5711[1] + tmp_5713[1], tmp_5711[2] + tmp_5713[2]];
    signal tmp_5715[3] <== CMul()(challengeQ, tmp_5714);
    signal tmp_5716[3] <== [evals[67][0] - publics[123], evals[67][1], evals[67][2]];
    signal tmp_5717[3] <== CMul()(evals[11], tmp_5716);
    signal tmp_5718[3] <== [tmp_5715[0] + tmp_5717[0], tmp_5715[1] + tmp_5717[1], tmp_5715[2] + tmp_5717[2]];
    signal tmp_5719[3] <== CMul()(challengeQ, tmp_5718);
    signal tmp_5720[3] <== [evals[68][0] - publics[124], evals[68][1], evals[68][2]];
    signal tmp_5721[3] <== CMul()(evals[11], tmp_5720);
    signal tmp_5722[3] <== [tmp_5719[0] + tmp_5721[0], tmp_5719[1] + tmp_5721[1], tmp_5719[2] + tmp_5721[2]];
    signal tmp_5723[3] <== CMul()(challengeQ, tmp_5722);
    signal tmp_5724[3] <== [evals[69][0] - publics[125], evals[69][1], evals[69][2]];
    signal tmp_5725[3] <== CMul()(evals[11], tmp_5724);
    signal tmp_5726[3] <== [tmp_5723[0] + tmp_5725[0], tmp_5723[1] + tmp_5725[1], tmp_5723[2] + tmp_5725[2]];
    signal tmp_5727[3] <== CMul()(challengeQ, tmp_5726);
    signal tmp_5728[3] <== [evals[70][0] - publics[126], evals[70][1], evals[70][2]];
    signal tmp_5729[3] <== CMul()(evals[11], tmp_5728);
    signal tmp_5730[3] <== [tmp_5727[0] + tmp_5729[0], tmp_5727[1] + tmp_5729[1], tmp_5727[2] + tmp_5729[2]];
    signal tmp_5731[3] <== CMul()(challengeQ, tmp_5730);
    signal tmp_5732[3] <== [evals[71][0] - publics[127], evals[71][1], evals[71][2]];
    signal tmp_5733[3] <== CMul()(evals[11], tmp_5732);
    signal tmp_5734[3] <== [tmp_5731[0] + tmp_5733[0], tmp_5731[1] + tmp_5733[1], tmp_5731[2] + tmp_5733[2]];
    signal tmp_5735[3] <== CMul()(challengeQ, tmp_5734);
    signal tmp_5736[3] <== [evals[72][0] - publics[128], evals[72][1], evals[72][2]];
    signal tmp_5737[3] <== CMul()(evals[11], tmp_5736);
    signal tmp_5738[3] <== [tmp_5735[0] + tmp_5737[0], tmp_5735[1] + tmp_5737[1], tmp_5735[2] + tmp_5737[2]];
    signal tmp_5739[3] <== CMul()(challengeQ, tmp_5738);
    signal tmp_5740[3] <== [evals[73][0] - publics[129], evals[73][1], evals[73][2]];
    signal tmp_5741[3] <== CMul()(evals[11], tmp_5740);
    signal tmp_5742[3] <== [tmp_5739[0] + tmp_5741[0], tmp_5739[1] + tmp_5741[1], tmp_5739[2] + tmp_5741[2]];
    signal tmp_5743[3] <== CMul()(challengeQ, tmp_5742);
    signal tmp_5744[3] <== [evals[74][0] - publics[130], evals[74][1], evals[74][2]];
    signal tmp_5745[3] <== CMul()(evals[11], tmp_5744);
    signal tmp_5746[3] <== [tmp_5743[0] + tmp_5745[0], tmp_5743[1] + tmp_5745[1], tmp_5743[2] + tmp_5745[2]];
    signal tmp_5747[3] <== CMul()(challengeQ, tmp_5746);
    signal tmp_5748[3] <== [evals[75][0] - publics[131], evals[75][1], evals[75][2]];
    signal tmp_5749[3] <== CMul()(evals[11], tmp_5748);
    signal tmp_5750[3] <== [tmp_5747[0] + tmp_5749[0], tmp_5747[1] + tmp_5749[1], tmp_5747[2] + tmp_5749[2]];
    signal tmp_5751[3] <== CMul()(challengeQ, tmp_5750);
    signal tmp_5752[3] <== [evals[76][0] - publics[132], evals[76][1], evals[76][2]];
    signal tmp_5753[3] <== CMul()(evals[11], tmp_5752);
    signal tmp_5754[3] <== [tmp_5751[0] + tmp_5753[0], tmp_5751[1] + tmp_5753[1], tmp_5751[2] + tmp_5753[2]];
    signal tmp_5755[3] <== CMul()(challengeQ, tmp_5754);
    signal tmp_5756[3] <== [evals[77][0] - publics[133], evals[77][1], evals[77][2]];
    signal tmp_5757[3] <== CMul()(evals[11], tmp_5756);
    signal tmp_5758[3] <== [tmp_5755[0] + tmp_5757[0], tmp_5755[1] + tmp_5757[1], tmp_5755[2] + tmp_5757[2]];
    signal tmp_5759[3] <== CMul()(challengeQ, tmp_5758);
    signal tmp_5760[3] <== [evals[78][0] - publics[134], evals[78][1], evals[78][2]];
    signal tmp_5761[3] <== CMul()(evals[11], tmp_5760);
    signal tmp_5762[3] <== [tmp_5759[0] + tmp_5761[0], tmp_5759[1] + tmp_5761[1], tmp_5759[2] + tmp_5761[2]];
    signal tmp_5763[3] <== CMul()(challengeQ, tmp_5762);
    signal tmp_5764[3] <== [evals[79][0] - publics[135], evals[79][1], evals[79][2]];
    signal tmp_5765[3] <== CMul()(evals[11], tmp_5764);
    signal tmp_5766[3] <== [tmp_5763[0] + tmp_5765[0], tmp_5763[1] + tmp_5765[1], tmp_5763[2] + tmp_5765[2]];
    signal tmp_5767[3] <== CMul()(challengeQ, tmp_5766);
    signal tmp_5768[3] <== [evals[80][0] - publics[136], evals[80][1], evals[80][2]];
    signal tmp_5769[3] <== CMul()(evals[11], tmp_5768);
    signal tmp_5770[3] <== [tmp_5767[0] + tmp_5769[0], tmp_5767[1] + tmp_5769[1], tmp_5767[2] + tmp_5769[2]];
    signal tmp_5771[3] <== CMul()(challengeQ, tmp_5770);
    signal tmp_5772[3] <== [evals[81][0] - publics[137], evals[81][1], evals[81][2]];
    signal tmp_5773[3] <== CMul()(evals[11], tmp_5772);
    signal tmp_5774[3] <== [tmp_5771[0] + tmp_5773[0], tmp_5771[1] + tmp_5773[1], tmp_5771[2] + tmp_5773[2]];
    signal tmp_5775[3] <== CMul()(challengeQ, tmp_5774);
    signal tmp_5776[3] <== [evals[82][0] - publics[138], evals[82][1], evals[82][2]];
    signal tmp_5777[3] <== CMul()(evals[11], tmp_5776);
    signal tmp_5778[3] <== [tmp_5775[0] + tmp_5777[0], tmp_5775[1] + tmp_5777[1], tmp_5775[2] + tmp_5777[2]];
    signal tmp_5779[3] <== CMul()(challengeQ, tmp_5778);
    signal tmp_5780[3] <== [evals[83][0] - publics[139], evals[83][1], evals[83][2]];
    signal tmp_5781[3] <== CMul()(evals[11], tmp_5780);
    signal tmp_5782[3] <== [tmp_5779[0] + tmp_5781[0], tmp_5779[1] + tmp_5781[1], tmp_5779[2] + tmp_5781[2]];
    signal tmp_5783[3] <== CMul()(challengeQ, tmp_5782);
    signal tmp_5784[3] <== [evals[84][0] - publics[140], evals[84][1], evals[84][2]];
    signal tmp_5785[3] <== CMul()(evals[11], tmp_5784);
    signal tmp_5786[3] <== [tmp_5783[0] + tmp_5785[0], tmp_5783[1] + tmp_5785[1], tmp_5783[2] + tmp_5785[2]];
    signal tmp_5787[3] <== CMul()(challengeQ, tmp_5786);
    signal tmp_5788[3] <== [evals[85][0] - publics[141], evals[85][1], evals[85][2]];
    signal tmp_5789[3] <== CMul()(evals[11], tmp_5788);
    signal tmp_5790[3] <== [tmp_5787[0] + tmp_5789[0], tmp_5787[1] + tmp_5789[1], tmp_5787[2] + tmp_5789[2]];
    signal tmp_5791[3] <== CMul()(challengeQ, tmp_5790);
    signal tmp_5792[3] <== [evals[86][0] - publics[142], evals[86][1], evals[86][2]];
    signal tmp_5793[3] <== CMul()(evals[11], tmp_5792);
    signal tmp_5794[3] <== [tmp_5791[0] + tmp_5793[0], tmp_5791[1] + tmp_5793[1], tmp_5791[2] + tmp_5793[2]];
    signal tmp_5795[3] <== CMul()(challengeQ, tmp_5794);
    signal tmp_5796[3] <== [evals[87][0] - publics[143], evals[87][1], evals[87][2]];
    signal tmp_5797[3] <== CMul()(evals[11], tmp_5796);
    signal tmp_5798[3] <== [tmp_5795[0] + tmp_5797[0], tmp_5795[1] + tmp_5797[1], tmp_5795[2] + tmp_5797[2]];
    signal tmp_5799[3] <== CMul()(challengeQ, tmp_5798);
    signal tmp_5800[3] <== [evals[64][0] - publics[144], evals[64][1], evals[64][2]];
    signal tmp_5801[3] <== CMul()(evals[10], tmp_5800);
    signal tmp_5802[3] <== [tmp_5799[0] + tmp_5801[0], tmp_5799[1] + tmp_5801[1], tmp_5799[2] + tmp_5801[2]];
    signal tmp_5803[3] <== CMul()(challengeQ, tmp_5802);
    signal tmp_5804[3] <== [evals[65][0] - publics[145], evals[65][1], evals[65][2]];
    signal tmp_5805[3] <== CMul()(evals[10], tmp_5804);
    signal tmp_5806[3] <== [tmp_5803[0] + tmp_5805[0], tmp_5803[1] + tmp_5805[1], tmp_5803[2] + tmp_5805[2]];
    signal tmp_5807[3] <== CMul()(challengeQ, tmp_5806);
    signal tmp_5808[3] <== [evals[66][0] - publics[146], evals[66][1], evals[66][2]];
    signal tmp_5809[3] <== CMul()(evals[10], tmp_5808);
    signal tmp_5810[3] <== [tmp_5807[0] + tmp_5809[0], tmp_5807[1] + tmp_5809[1], tmp_5807[2] + tmp_5809[2]];
    signal tmp_5811[3] <== CMul()(challengeQ, tmp_5810);
    signal tmp_5812[3] <== [evals[67][0] - publics[147], evals[67][1], evals[67][2]];
    signal tmp_5813[3] <== CMul()(evals[10], tmp_5812);
    signal tmp_5814[3] <== [tmp_5811[0] + tmp_5813[0], tmp_5811[1] + tmp_5813[1], tmp_5811[2] + tmp_5813[2]];
    signal tmp_5815[3] <== CMul()(challengeQ, tmp_5814);
    signal tmp_5816[3] <== [evals[68][0] - publics[148], evals[68][1], evals[68][2]];
    signal tmp_5817[3] <== CMul()(evals[10], tmp_5816);
    signal tmp_5818[3] <== [tmp_5815[0] + tmp_5817[0], tmp_5815[1] + tmp_5817[1], tmp_5815[2] + tmp_5817[2]];
    signal tmp_5819[3] <== CMul()(challengeQ, tmp_5818);
    signal tmp_5820[3] <== [evals[69][0] - publics[149], evals[69][1], evals[69][2]];
    signal tmp_5821[3] <== CMul()(evals[10], tmp_5820);
    signal tmp_5822[3] <== [tmp_5819[0] + tmp_5821[0], tmp_5819[1] + tmp_5821[1], tmp_5819[2] + tmp_5821[2]];
    signal tmp_5823[3] <== CMul()(challengeQ, tmp_5822);
    signal tmp_5824[3] <== [evals[70][0] - publics[150], evals[70][1], evals[70][2]];
    signal tmp_5825[3] <== CMul()(evals[10], tmp_5824);
    signal tmp_5826[3] <== [tmp_5823[0] + tmp_5825[0], tmp_5823[1] + tmp_5825[1], tmp_5823[2] + tmp_5825[2]];
    signal tmp_5827[3] <== CMul()(challengeQ, tmp_5826);
    signal tmp_5828[3] <== [evals[71][0] - publics[151], evals[71][1], evals[71][2]];
    signal tmp_5829[3] <== CMul()(evals[10], tmp_5828);
    signal tmp_5830[3] <== [tmp_5827[0] + tmp_5829[0], tmp_5827[1] + tmp_5829[1], tmp_5827[2] + tmp_5829[2]];
    signal tmp_5831[3] <== CMul()(challengeQ, tmp_5830);
    signal tmp_5832[3] <== [evals[72][0] - publics[152], evals[72][1], evals[72][2]];
    signal tmp_5833[3] <== CMul()(evals[10], tmp_5832);
    signal tmp_5834[3] <== [tmp_5831[0] + tmp_5833[0], tmp_5831[1] + tmp_5833[1], tmp_5831[2] + tmp_5833[2]];
    signal tmp_5835[3] <== CMul()(challengeQ, tmp_5834);
    signal tmp_5836[3] <== [evals[73][0] - publics[153], evals[73][1], evals[73][2]];
    signal tmp_5837[3] <== CMul()(evals[10], tmp_5836);
    signal tmp_5838[3] <== [tmp_5835[0] + tmp_5837[0], tmp_5835[1] + tmp_5837[1], tmp_5835[2] + tmp_5837[2]];
    signal tmp_5839[3] <== CMul()(challengeQ, tmp_5838);
    signal tmp_5840[3] <== [evals[74][0] - publics[154], evals[74][1], evals[74][2]];
    signal tmp_5841[3] <== CMul()(evals[10], tmp_5840);
    signal tmp_5842[3] <== [tmp_5839[0] + tmp_5841[0], tmp_5839[1] + tmp_5841[1], tmp_5839[2] + tmp_5841[2]];
    signal tmp_5843[3] <== CMul()(challengeQ, tmp_5842);
    signal tmp_5844[3] <== [evals[75][0] - publics[155], evals[75][1], evals[75][2]];
    signal tmp_5845[3] <== CMul()(evals[10], tmp_5844);
    signal tmp_5846[3] <== [tmp_5843[0] + tmp_5845[0], tmp_5843[1] + tmp_5845[1], tmp_5843[2] + tmp_5845[2]];
    signal tmp_5847[3] <== CMul()(challengeQ, tmp_5846);
    signal tmp_5848[3] <== [evals[76][0] - publics[156], evals[76][1], evals[76][2]];
    signal tmp_5849[3] <== CMul()(evals[10], tmp_5848);
    signal tmp_5850[3] <== [tmp_5847[0] + tmp_5849[0], tmp_5847[1] + tmp_5849[1], tmp_5847[2] + tmp_5849[2]];
    signal tmp_5851[3] <== CMul()(challengeQ, tmp_5850);
    signal tmp_5852[3] <== [evals[77][0] - publics[157], evals[77][1], evals[77][2]];
    signal tmp_5853[3] <== CMul()(evals[10], tmp_5852);
    signal tmp_5854[3] <== [tmp_5851[0] + tmp_5853[0], tmp_5851[1] + tmp_5853[1], tmp_5851[2] + tmp_5853[2]];
    signal tmp_5855[3] <== CMul()(challengeQ, tmp_5854);
    signal tmp_5856[3] <== [evals[78][0] - publics[158], evals[78][1], evals[78][2]];
    signal tmp_5857[3] <== CMul()(evals[10], tmp_5856);
    signal tmp_5858[3] <== [tmp_5855[0] + tmp_5857[0], tmp_5855[1] + tmp_5857[1], tmp_5855[2] + tmp_5857[2]];
    signal tmp_5859[3] <== CMul()(challengeQ, tmp_5858);
    signal tmp_5860[3] <== [evals[79][0] - publics[159], evals[79][1], evals[79][2]];
    signal tmp_5861[3] <== CMul()(evals[10], tmp_5860);
    signal tmp_5862[3] <== [tmp_5859[0] + tmp_5861[0], tmp_5859[1] + tmp_5861[1], tmp_5859[2] + tmp_5861[2]];
    signal tmp_5863[3] <== CMul()(challengeQ, tmp_5862);
    signal tmp_5864[3] <== [evals[80][0] - publics[160], evals[80][1], evals[80][2]];
    signal tmp_5865[3] <== CMul()(evals[10], tmp_5864);
    signal tmp_5866[3] <== [tmp_5863[0] + tmp_5865[0], tmp_5863[1] + tmp_5865[1], tmp_5863[2] + tmp_5865[2]];
    signal tmp_5867[3] <== CMul()(challengeQ, tmp_5866);
    signal tmp_5868[3] <== [evals[81][0] - publics[161], evals[81][1], evals[81][2]];
    signal tmp_5869[3] <== CMul()(evals[10], tmp_5868);
    signal tmp_5870[3] <== [tmp_5867[0] + tmp_5869[0], tmp_5867[1] + tmp_5869[1], tmp_5867[2] + tmp_5869[2]];
    signal tmp_5871[3] <== CMul()(challengeQ, tmp_5870);
    signal tmp_5872[3] <== [evals[82][0] - publics[162], evals[82][1], evals[82][2]];
    signal tmp_5873[3] <== CMul()(evals[10], tmp_5872);
    signal tmp_5874[3] <== [tmp_5871[0] + tmp_5873[0], tmp_5871[1] + tmp_5873[1], tmp_5871[2] + tmp_5873[2]];
    signal tmp_5875[3] <== CMul()(challengeQ, tmp_5874);
    signal tmp_5876[3] <== [evals[83][0] - publics[163], evals[83][1], evals[83][2]];
    signal tmp_5877[3] <== CMul()(evals[10], tmp_5876);
    signal tmp_5878[3] <== [tmp_5875[0] + tmp_5877[0], tmp_5875[1] + tmp_5877[1], tmp_5875[2] + tmp_5877[2]];
    signal tmp_5879[3] <== CMul()(challengeQ, tmp_5878);
    signal tmp_5880[3] <== [evals[84][0] - publics[164], evals[84][1], evals[84][2]];
    signal tmp_5881[3] <== CMul()(evals[10], tmp_5880);
    signal tmp_5882[3] <== [tmp_5879[0] + tmp_5881[0], tmp_5879[1] + tmp_5881[1], tmp_5879[2] + tmp_5881[2]];
    signal tmp_5883[3] <== CMul()(challengeQ, tmp_5882);
    signal tmp_5884[3] <== [evals[85][0] - publics[165], evals[85][1], evals[85][2]];
    signal tmp_5885[3] <== CMul()(evals[10], tmp_5884);
    signal tmp_5886[3] <== [tmp_5883[0] + tmp_5885[0], tmp_5883[1] + tmp_5885[1], tmp_5883[2] + tmp_5885[2]];
    signal tmp_5887[3] <== CMul()(challengeQ, tmp_5886);
    signal tmp_5888[3] <== [evals[86][0] - publics[166], evals[86][1], evals[86][2]];
    signal tmp_5889[3] <== CMul()(evals[10], tmp_5888);
    signal tmp_5890[3] <== [tmp_5887[0] + tmp_5889[0], tmp_5887[1] + tmp_5889[1], tmp_5887[2] + tmp_5889[2]];
    signal tmp_5891[3] <== CMul()(challengeQ, tmp_5890);
    signal tmp_5892[3] <== [evals[87][0] - publics[167], evals[87][1], evals[87][2]];
    signal tmp_5893[3] <== CMul()(evals[10], tmp_5892);
    signal tmp_5894[3] <== [tmp_5891[0] + tmp_5893[0], tmp_5891[1] + tmp_5893[1], tmp_5891[2] + tmp_5893[2]];
    signal tmp_5895[3] <== CMul()(challengeQ, tmp_5894);
    signal tmp_5896[3] <== [evals[64][0] - publics[168], evals[64][1], evals[64][2]];
    signal tmp_5897[3] <== CMul()(evals[9], tmp_5896);
    signal tmp_5898[3] <== [tmp_5895[0] + tmp_5897[0], tmp_5895[1] + tmp_5897[1], tmp_5895[2] + tmp_5897[2]];
    signal tmp_5899[3] <== CMul()(challengeQ, tmp_5898);
    signal tmp_5900[3] <== [evals[65][0] - publics[169], evals[65][1], evals[65][2]];
    signal tmp_5901[3] <== CMul()(evals[9], tmp_5900);
    signal tmp_5902[3] <== [tmp_5899[0] + tmp_5901[0], tmp_5899[1] + tmp_5901[1], tmp_5899[2] + tmp_5901[2]];
    signal tmp_5903[3] <== CMul()(challengeQ, tmp_5902);
    signal tmp_5904[3] <== [evals[66][0] - publics[170], evals[66][1], evals[66][2]];
    signal tmp_5905[3] <== CMul()(evals[9], tmp_5904);
    signal tmp_5906[3] <== [tmp_5903[0] + tmp_5905[0], tmp_5903[1] + tmp_5905[1], tmp_5903[2] + tmp_5905[2]];
    signal tmp_5907[3] <== CMul()(challengeQ, tmp_5906);
    signal tmp_5908[3] <== [evals[67][0] - publics[171], evals[67][1], evals[67][2]];
    signal tmp_5909[3] <== CMul()(evals[9], tmp_5908);
    signal tmp_5910[3] <== [tmp_5907[0] + tmp_5909[0], tmp_5907[1] + tmp_5909[1], tmp_5907[2] + tmp_5909[2]];
    signal tmp_5911[3] <== CMul()(challengeQ, tmp_5910);
    signal tmp_5912[3] <== [evals[68][0] - publics[172], evals[68][1], evals[68][2]];
    signal tmp_5913[3] <== CMul()(evals[9], tmp_5912);
    signal tmp_5914[3] <== [tmp_5911[0] + tmp_5913[0], tmp_5911[1] + tmp_5913[1], tmp_5911[2] + tmp_5913[2]];
    signal tmp_5915[3] <== CMul()(challengeQ, tmp_5914);
    signal tmp_5916[3] <== [evals[69][0] - publics[173], evals[69][1], evals[69][2]];
    signal tmp_5917[3] <== CMul()(evals[9], tmp_5916);
    signal tmp_5918[3] <== [tmp_5915[0] + tmp_5917[0], tmp_5915[1] + tmp_5917[1], tmp_5915[2] + tmp_5917[2]];
    signal tmp_5919[3] <== CMul()(challengeQ, tmp_5918);
    signal tmp_5920[3] <== [evals[70][0] - publics[174], evals[70][1], evals[70][2]];
    signal tmp_5921[3] <== CMul()(evals[9], tmp_5920);
    signal tmp_5922[3] <== [tmp_5919[0] + tmp_5921[0], tmp_5919[1] + tmp_5921[1], tmp_5919[2] + tmp_5921[2]];
    signal tmp_5923[3] <== CMul()(challengeQ, tmp_5922);
    signal tmp_5924[3] <== [evals[71][0] - publics[175], evals[71][1], evals[71][2]];
    signal tmp_5925[3] <== CMul()(evals[9], tmp_5924);
    signal tmp_5926[3] <== [tmp_5923[0] + tmp_5925[0], tmp_5923[1] + tmp_5925[1], tmp_5923[2] + tmp_5925[2]];
    signal tmp_5927[3] <== CMul()(challengeQ, tmp_5926);
    signal tmp_5928[3] <== [evals[72][0] - publics[176], evals[72][1], evals[72][2]];
    signal tmp_5929[3] <== CMul()(evals[9], tmp_5928);
    signal tmp_5930[3] <== [tmp_5927[0] + tmp_5929[0], tmp_5927[1] + tmp_5929[1], tmp_5927[2] + tmp_5929[2]];
    signal tmp_5931[3] <== CMul()(challengeQ, tmp_5930);
    signal tmp_5932[3] <== [evals[73][0] - publics[177], evals[73][1], evals[73][2]];
    signal tmp_5933[3] <== CMul()(evals[9], tmp_5932);
    signal tmp_5934[3] <== [tmp_5931[0] + tmp_5933[0], tmp_5931[1] + tmp_5933[1], tmp_5931[2] + tmp_5933[2]];
    signal tmp_5935[3] <== CMul()(challengeQ, tmp_5934);
    signal tmp_5936[3] <== [evals[74][0] - publics[178], evals[74][1], evals[74][2]];
    signal tmp_5937[3] <== CMul()(evals[9], tmp_5936);
    signal tmp_5938[3] <== [tmp_5935[0] + tmp_5937[0], tmp_5935[1] + tmp_5937[1], tmp_5935[2] + tmp_5937[2]];
    signal tmp_5939[3] <== CMul()(challengeQ, tmp_5938);
    signal tmp_5940[3] <== [evals[75][0] - publics[179], evals[75][1], evals[75][2]];
    signal tmp_5941[3] <== CMul()(evals[9], tmp_5940);
    signal tmp_5942[3] <== [tmp_5939[0] + tmp_5941[0], tmp_5939[1] + tmp_5941[1], tmp_5939[2] + tmp_5941[2]];
    signal tmp_5943[3] <== CMul()(challengeQ, tmp_5942);
    signal tmp_5944[3] <== [evals[76][0] - publics[180], evals[76][1], evals[76][2]];
    signal tmp_5945[3] <== CMul()(evals[9], tmp_5944);
    signal tmp_5946[3] <== [tmp_5943[0] + tmp_5945[0], tmp_5943[1] + tmp_5945[1], tmp_5943[2] + tmp_5945[2]];
    signal tmp_5947[3] <== CMul()(challengeQ, tmp_5946);
    signal tmp_5948[3] <== [evals[77][0] - publics[181], evals[77][1], evals[77][2]];
    signal tmp_5949[3] <== CMul()(evals[9], tmp_5948);
    signal tmp_5950[3] <== [tmp_5947[0] + tmp_5949[0], tmp_5947[1] + tmp_5949[1], tmp_5947[2] + tmp_5949[2]];
    signal tmp_5951[3] <== CMul()(challengeQ, tmp_5950);
    signal tmp_5952[3] <== [evals[78][0] - publics[182], evals[78][1], evals[78][2]];
    signal tmp_5953[3] <== CMul()(evals[9], tmp_5952);
    signal tmp_5954[3] <== [tmp_5951[0] + tmp_5953[0], tmp_5951[1] + tmp_5953[1], tmp_5951[2] + tmp_5953[2]];
    signal tmp_5955[3] <== CMul()(challengeQ, tmp_5954);
    signal tmp_5956[3] <== [evals[79][0] - publics[183], evals[79][1], evals[79][2]];
    signal tmp_5957[3] <== CMul()(evals[9], tmp_5956);
    signal tmp_5958[3] <== [tmp_5955[0] + tmp_5957[0], tmp_5955[1] + tmp_5957[1], tmp_5955[2] + tmp_5957[2]];
    signal tmp_5959[3] <== CMul()(challengeQ, tmp_5958);
    signal tmp_5960[3] <== [evals[80][0] - publics[184], evals[80][1], evals[80][2]];
    signal tmp_5961[3] <== CMul()(evals[9], tmp_5960);
    signal tmp_5962[3] <== [tmp_5959[0] + tmp_5961[0], tmp_5959[1] + tmp_5961[1], tmp_5959[2] + tmp_5961[2]];
    signal tmp_5963[3] <== CMul()(challengeQ, tmp_5962);
    signal tmp_5964[3] <== [evals[81][0] - publics[185], evals[81][1], evals[81][2]];
    signal tmp_5965[3] <== CMul()(evals[9], tmp_5964);
    signal tmp_5966[3] <== [tmp_5963[0] + tmp_5965[0], tmp_5963[1] + tmp_5965[1], tmp_5963[2] + tmp_5965[2]];
    signal tmp_5967[3] <== CMul()(challengeQ, tmp_5966);
    signal tmp_5968[3] <== [evals[82][0] - publics[186], evals[82][1], evals[82][2]];
    signal tmp_5969[3] <== CMul()(evals[9], tmp_5968);
    signal tmp_5970[3] <== [tmp_5967[0] + tmp_5969[0], tmp_5967[1] + tmp_5969[1], tmp_5967[2] + tmp_5969[2]];
    signal tmp_5971[3] <== CMul()(challengeQ, tmp_5970);
    signal tmp_5972[3] <== [evals[83][0] - publics[187], evals[83][1], evals[83][2]];
    signal tmp_5973[3] <== CMul()(evals[9], tmp_5972);
    signal tmp_5974[3] <== [tmp_5971[0] + tmp_5973[0], tmp_5971[1] + tmp_5973[1], tmp_5971[2] + tmp_5973[2]];
    signal tmp_5975[3] <== CMul()(challengeQ, tmp_5974);
    signal tmp_5976[3] <== [evals[84][0] - publics[188], evals[84][1], evals[84][2]];
    signal tmp_5977[3] <== CMul()(evals[9], tmp_5976);
    signal tmp_5978[3] <== [tmp_5975[0] + tmp_5977[0], tmp_5975[1] + tmp_5977[1], tmp_5975[2] + tmp_5977[2]];
    signal tmp_5979[3] <== CMul()(challengeQ, tmp_5978);
    signal tmp_5980[3] <== [evals[85][0] - publics[189], evals[85][1], evals[85][2]];
    signal tmp_5981[3] <== CMul()(evals[9], tmp_5980);
    signal tmp_5982[3] <== [tmp_5979[0] + tmp_5981[0], tmp_5979[1] + tmp_5981[1], tmp_5979[2] + tmp_5981[2]];
    signal tmp_5983[3] <== CMul()(challengeQ, tmp_5982);
    signal tmp_5984[3] <== [evals[86][0] - publics[190], evals[86][1], evals[86][2]];
    signal tmp_5985[3] <== CMul()(evals[9], tmp_5984);
    signal tmp_5986[3] <== [tmp_5983[0] + tmp_5985[0], tmp_5983[1] + tmp_5985[1], tmp_5983[2] + tmp_5985[2]];
    signal tmp_5987[3] <== CMul()(challengeQ, tmp_5986);
    signal tmp_5988[3] <== [evals[87][0] - publics[191], evals[87][1], evals[87][2]];
    signal tmp_5989[3] <== CMul()(evals[9], tmp_5988);
    signal tmp_5990[3] <== [tmp_5987[0] + tmp_5989[0], tmp_5987[1] + tmp_5989[1], tmp_5987[2] + tmp_5989[2]];
    signal tmp_5991[3] <== CMul()(challengeQ, tmp_5990);
    signal tmp_5992[3] <== [evals[64][0] - publics[192], evals[64][1], evals[64][2]];
    signal tmp_5993[3] <== CMul()(evals[8], tmp_5992);
    signal tmp_5994[3] <== [tmp_5991[0] + tmp_5993[0], tmp_5991[1] + tmp_5993[1], tmp_5991[2] + tmp_5993[2]];
    signal tmp_5995[3] <== CMul()(challengeQ, tmp_5994);
    signal tmp_5996[3] <== [evals[65][0] - publics[193], evals[65][1], evals[65][2]];
    signal tmp_5997[3] <== CMul()(evals[8], tmp_5996);
    signal tmp_5998[3] <== [tmp_5995[0] + tmp_5997[0], tmp_5995[1] + tmp_5997[1], tmp_5995[2] + tmp_5997[2]];
    signal tmp_5999[3] <== CMul()(challengeQ, tmp_5998);
    signal tmp_6000[3] <== [evals[66][0] - publics[194], evals[66][1], evals[66][2]];
    signal tmp_6001[3] <== CMul()(evals[8], tmp_6000);
    signal tmp_6002[3] <== [tmp_5999[0] + tmp_6001[0], tmp_5999[1] + tmp_6001[1], tmp_5999[2] + tmp_6001[2]];
    signal tmp_6003[3] <== CMul()(challengeQ, tmp_6002);
    signal tmp_6004[3] <== [evals[67][0] - publics[195], evals[67][1], evals[67][2]];
    signal tmp_6005[3] <== CMul()(evals[8], tmp_6004);
    signal tmp_6006[3] <== [tmp_6003[0] + tmp_6005[0], tmp_6003[1] + tmp_6005[1], tmp_6003[2] + tmp_6005[2]];
    signal tmp_6007[3] <== CMul()(challengeQ, tmp_6006);
    signal tmp_6008[3] <== [evals[68][0] - publics[196], evals[68][1], evals[68][2]];
    signal tmp_6009[3] <== CMul()(evals[8], tmp_6008);
    signal tmp_6010[3] <== [tmp_6007[0] + tmp_6009[0], tmp_6007[1] + tmp_6009[1], tmp_6007[2] + tmp_6009[2]];
    signal tmp_6011[3] <== CMul()(challengeQ, tmp_6010);
    signal tmp_6012[3] <== [evals[69][0] - publics[197], evals[69][1], evals[69][2]];
    signal tmp_6013[3] <== CMul()(evals[8], tmp_6012);
    signal tmp_6014[3] <== [tmp_6011[0] + tmp_6013[0], tmp_6011[1] + tmp_6013[1], tmp_6011[2] + tmp_6013[2]];
    signal tmp_6015[3] <== CMul()(challengeQ, tmp_6014);
    signal tmp_6016[3] <== [evals[70][0] - publics[198], evals[70][1], evals[70][2]];
    signal tmp_6017[3] <== CMul()(evals[8], tmp_6016);
    signal tmp_6018[3] <== [tmp_6015[0] + tmp_6017[0], tmp_6015[1] + tmp_6017[1], tmp_6015[2] + tmp_6017[2]];
    signal tmp_6019[3] <== CMul()(challengeQ, tmp_6018);
    signal tmp_6020[3] <== [evals[71][0] - publics[199], evals[71][1], evals[71][2]];
    signal tmp_6021[3] <== CMul()(evals[8], tmp_6020);
    signal tmp_6022[3] <== [tmp_6019[0] + tmp_6021[0], tmp_6019[1] + tmp_6021[1], tmp_6019[2] + tmp_6021[2]];
    signal tmp_6023[3] <== CMul()(challengeQ, tmp_6022);
    signal tmp_6024[3] <== [evals[72][0] - publics[200], evals[72][1], evals[72][2]];
    signal tmp_6025[3] <== CMul()(evals[8], tmp_6024);
    signal tmp_6026[3] <== [tmp_6023[0] + tmp_6025[0], tmp_6023[1] + tmp_6025[1], tmp_6023[2] + tmp_6025[2]];
    signal tmp_6027[3] <== CMul()(challengeQ, tmp_6026);
    signal tmp_6028[3] <== [evals[73][0] - publics[201], evals[73][1], evals[73][2]];
    signal tmp_6029[3] <== CMul()(evals[8], tmp_6028);
    signal tmp_6030[3] <== [tmp_6027[0] + tmp_6029[0], tmp_6027[1] + tmp_6029[1], tmp_6027[2] + tmp_6029[2]];
    signal tmp_6031[3] <== CMul()(challengeQ, tmp_6030);
    signal tmp_6032[3] <== [evals[74][0] - publics[202], evals[74][1], evals[74][2]];
    signal tmp_6033[3] <== CMul()(evals[8], tmp_6032);
    signal tmp_6034[3] <== [tmp_6031[0] + tmp_6033[0], tmp_6031[1] + tmp_6033[1], tmp_6031[2] + tmp_6033[2]];
    signal tmp_6035[3] <== CMul()(challengeQ, tmp_6034);
    signal tmp_6036[3] <== [evals[75][0] - publics[203], evals[75][1], evals[75][2]];
    signal tmp_6037[3] <== CMul()(evals[8], tmp_6036);
    signal tmp_6038[3] <== [tmp_6035[0] + tmp_6037[0], tmp_6035[1] + tmp_6037[1], tmp_6035[2] + tmp_6037[2]];
    signal tmp_6039[3] <== CMul()(challengeQ, tmp_6038);
    signal tmp_6040[3] <== [evals[76][0] - publics[204], evals[76][1], evals[76][2]];
    signal tmp_6041[3] <== CMul()(evals[8], tmp_6040);
    signal tmp_6042[3] <== [tmp_6039[0] + tmp_6041[0], tmp_6039[1] + tmp_6041[1], tmp_6039[2] + tmp_6041[2]];
    signal tmp_6043[3] <== CMul()(challengeQ, tmp_6042);
    signal tmp_6044[3] <== [evals[77][0] - publics[205], evals[77][1], evals[77][2]];
    signal tmp_6045[3] <== CMul()(evals[8], tmp_6044);
    signal tmp_6046[3] <== [tmp_6043[0] + tmp_6045[0], tmp_6043[1] + tmp_6045[1], tmp_6043[2] + tmp_6045[2]];
    signal tmp_6047[3] <== CMul()(challengeQ, tmp_6046);
    signal tmp_6048[3] <== [evals[78][0] - publics[206], evals[78][1], evals[78][2]];
    signal tmp_6049[3] <== CMul()(evals[8], tmp_6048);
    signal tmp_6050[3] <== [tmp_6047[0] + tmp_6049[0], tmp_6047[1] + tmp_6049[1], tmp_6047[2] + tmp_6049[2]];
    signal tmp_6051[3] <== CMul()(challengeQ, tmp_6050);
    signal tmp_6052[3] <== [evals[79][0] - publics[207], evals[79][1], evals[79][2]];
    signal tmp_6053[3] <== CMul()(evals[8], tmp_6052);
    signal tmp_6054[3] <== [tmp_6051[0] + tmp_6053[0], tmp_6051[1] + tmp_6053[1], tmp_6051[2] + tmp_6053[2]];
    signal tmp_6055[3] <== CMul()(challengeQ, tmp_6054);
    signal tmp_6056[3] <== [evals[80][0] - publics[208], evals[80][1], evals[80][2]];
    signal tmp_6057[3] <== CMul()(evals[8], tmp_6056);
    signal tmp_6058[3] <== [tmp_6055[0] + tmp_6057[0], tmp_6055[1] + tmp_6057[1], tmp_6055[2] + tmp_6057[2]];
    signal tmp_6059[3] <== CMul()(challengeQ, tmp_6058);
    signal tmp_6060[3] <== [evals[81][0] - publics[209], evals[81][1], evals[81][2]];
    signal tmp_6061[3] <== CMul()(evals[8], tmp_6060);
    signal tmp_6062[3] <== [tmp_6059[0] + tmp_6061[0], tmp_6059[1] + tmp_6061[1], tmp_6059[2] + tmp_6061[2]];
    signal tmp_6063[3] <== CMul()(challengeQ, tmp_6062);
    signal tmp_6064[3] <== [evals[82][0] - publics[210], evals[82][1], evals[82][2]];
    signal tmp_6065[3] <== CMul()(evals[8], tmp_6064);
    signal tmp_6066[3] <== [tmp_6063[0] + tmp_6065[0], tmp_6063[1] + tmp_6065[1], tmp_6063[2] + tmp_6065[2]];
    signal tmp_6067[3] <== CMul()(challengeQ, tmp_6066);
    signal tmp_6068[3] <== [evals[83][0] - publics[211], evals[83][1], evals[83][2]];
    signal tmp_6069[3] <== CMul()(evals[8], tmp_6068);
    signal tmp_6070[3] <== [tmp_6067[0] + tmp_6069[0], tmp_6067[1] + tmp_6069[1], tmp_6067[2] + tmp_6069[2]];
    signal tmp_6071[3] <== CMul()(challengeQ, tmp_6070);
    signal tmp_6072[3] <== [evals[84][0] - publics[212], evals[84][1], evals[84][2]];
    signal tmp_6073[3] <== CMul()(evals[8], tmp_6072);
    signal tmp_6074[3] <== [tmp_6071[0] + tmp_6073[0], tmp_6071[1] + tmp_6073[1], tmp_6071[2] + tmp_6073[2]];
    signal tmp_6075[3] <== CMul()(challengeQ, tmp_6074);
    signal tmp_6076[3] <== [evals[85][0] - publics[213], evals[85][1], evals[85][2]];
    signal tmp_6077[3] <== CMul()(evals[8], tmp_6076);
    signal tmp_6078[3] <== [tmp_6075[0] + tmp_6077[0], tmp_6075[1] + tmp_6077[1], tmp_6075[2] + tmp_6077[2]];
    signal tmp_6079[3] <== CMul()(challengeQ, tmp_6078);
    signal tmp_6080[3] <== [evals[86][0] - publics[214], evals[86][1], evals[86][2]];
    signal tmp_6081[3] <== CMul()(evals[8], tmp_6080);
    signal tmp_6082[3] <== [tmp_6079[0] + tmp_6081[0], tmp_6079[1] + tmp_6081[1], tmp_6079[2] + tmp_6081[2]];
    signal tmp_6083[3] <== CMul()(challengeQ, tmp_6082);
    signal tmp_6084[3] <== [evals[87][0] - publics[215], evals[87][1], evals[87][2]];
    signal tmp_6085[3] <== CMul()(evals[8], tmp_6084);
    signal tmp_6086[3] <== [tmp_6083[0] + tmp_6085[0], tmp_6083[1] + tmp_6085[1], tmp_6083[2] + tmp_6085[2]];
    signal tmp_6087[3] <== CMul()(challengeQ, tmp_6086);
    signal tmp_6088[3] <== [evals[64][0] - publics[216], evals[64][1], evals[64][2]];
    signal tmp_6089[3] <== CMul()(evals[7], tmp_6088);
    signal tmp_6090[3] <== [tmp_6087[0] + tmp_6089[0], tmp_6087[1] + tmp_6089[1], tmp_6087[2] + tmp_6089[2]];
    signal tmp_6091[3] <== CMul()(challengeQ, tmp_6090);
    signal tmp_6092[3] <== [evals[65][0] - publics[217], evals[65][1], evals[65][2]];
    signal tmp_6093[3] <== CMul()(evals[7], tmp_6092);
    signal tmp_6094[3] <== [tmp_6091[0] + tmp_6093[0], tmp_6091[1] + tmp_6093[1], tmp_6091[2] + tmp_6093[2]];
    signal tmp_6095[3] <== CMul()(challengeQ, tmp_6094);
    signal tmp_6096[3] <== [evals[66][0] - publics[218], evals[66][1], evals[66][2]];
    signal tmp_6097[3] <== CMul()(evals[7], tmp_6096);
    signal tmp_6098[3] <== [tmp_6095[0] + tmp_6097[0], tmp_6095[1] + tmp_6097[1], tmp_6095[2] + tmp_6097[2]];
    signal tmp_6099[3] <== CMul()(challengeQ, tmp_6098);
    signal tmp_6100[3] <== [evals[67][0] - publics[219], evals[67][1], evals[67][2]];
    signal tmp_6101[3] <== CMul()(evals[7], tmp_6100);
    signal tmp_6102[3] <== [tmp_6099[0] + tmp_6101[0], tmp_6099[1] + tmp_6101[1], tmp_6099[2] + tmp_6101[2]];
    signal tmp_6103[3] <== CMul()(challengeQ, tmp_6102);
    signal tmp_6104[3] <== [evals[68][0] - publics[220], evals[68][1], evals[68][2]];
    signal tmp_6105[3] <== CMul()(evals[7], tmp_6104);
    signal tmp_6106[3] <== [tmp_6103[0] + tmp_6105[0], tmp_6103[1] + tmp_6105[1], tmp_6103[2] + tmp_6105[2]];
    signal tmp_6107[3] <== CMul()(challengeQ, tmp_6106);
    signal tmp_6108[3] <== [evals[69][0] - publics[221], evals[69][1], evals[69][2]];
    signal tmp_6109[3] <== CMul()(evals[7], tmp_6108);
    signal tmp_6110[3] <== [tmp_6107[0] + tmp_6109[0], tmp_6107[1] + tmp_6109[1], tmp_6107[2] + tmp_6109[2]];
    signal tmp_6111[3] <== CMul()(challengeQ, tmp_6110);
    signal tmp_6112[3] <== [evals[70][0] - publics[222], evals[70][1], evals[70][2]];
    signal tmp_6113[3] <== CMul()(evals[7], tmp_6112);
    signal tmp_6114[3] <== [tmp_6111[0] + tmp_6113[0], tmp_6111[1] + tmp_6113[1], tmp_6111[2] + tmp_6113[2]];
    signal tmp_6115[3] <== CMul()(challengeQ, tmp_6114);
    signal tmp_6116[3] <== [evals[71][0] - publics[223], evals[71][1], evals[71][2]];
    signal tmp_6117[3] <== CMul()(evals[7], tmp_6116);
    signal tmp_6118[3] <== [tmp_6115[0] + tmp_6117[0], tmp_6115[1] + tmp_6117[1], tmp_6115[2] + tmp_6117[2]];
    signal tmp_6119[3] <== CMul()(challengeQ, tmp_6118);
    signal tmp_6120[3] <== [evals[72][0] - publics[224], evals[72][1], evals[72][2]];
    signal tmp_6121[3] <== CMul()(evals[7], tmp_6120);
    signal tmp_6122[3] <== [tmp_6119[0] + tmp_6121[0], tmp_6119[1] + tmp_6121[1], tmp_6119[2] + tmp_6121[2]];
    signal tmp_6123[3] <== CMul()(challengeQ, tmp_6122);
    signal tmp_6124[3] <== [evals[73][0] - publics[225], evals[73][1], evals[73][2]];
    signal tmp_6125[3] <== CMul()(evals[7], tmp_6124);
    signal tmp_6126[3] <== [tmp_6123[0] + tmp_6125[0], tmp_6123[1] + tmp_6125[1], tmp_6123[2] + tmp_6125[2]];
    signal tmp_6127[3] <== CMul()(challengeQ, tmp_6126);
    signal tmp_6128[3] <== [evals[74][0] - publics[226], evals[74][1], evals[74][2]];
    signal tmp_6129[3] <== CMul()(evals[7], tmp_6128);
    signal tmp_6130[3] <== [tmp_6127[0] + tmp_6129[0], tmp_6127[1] + tmp_6129[1], tmp_6127[2] + tmp_6129[2]];
    signal tmp_6131[3] <== CMul()(challengeQ, tmp_6130);
    signal tmp_6132[3] <== [evals[75][0] - publics[227], evals[75][1], evals[75][2]];
    signal tmp_6133[3] <== CMul()(evals[7], tmp_6132);
    signal tmp_6134[3] <== [tmp_6131[0] + tmp_6133[0], tmp_6131[1] + tmp_6133[1], tmp_6131[2] + tmp_6133[2]];
    signal tmp_6135[3] <== CMul()(challengeQ, tmp_6134);
    signal tmp_6136[3] <== [evals[76][0] - publics[228], evals[76][1], evals[76][2]];
    signal tmp_6137[3] <== CMul()(evals[7], tmp_6136);
    signal tmp_6138[3] <== [tmp_6135[0] + tmp_6137[0], tmp_6135[1] + tmp_6137[1], tmp_6135[2] + tmp_6137[2]];
    signal tmp_6139[3] <== CMul()(challengeQ, tmp_6138);
    signal tmp_6140[3] <== [evals[77][0] - publics[229], evals[77][1], evals[77][2]];
    signal tmp_6141[3] <== CMul()(evals[7], tmp_6140);
    signal tmp_6142[3] <== [tmp_6139[0] + tmp_6141[0], tmp_6139[1] + tmp_6141[1], tmp_6139[2] + tmp_6141[2]];
    signal tmp_6143[3] <== CMul()(challengeQ, tmp_6142);
    signal tmp_6144[3] <== [evals[78][0] - publics[230], evals[78][1], evals[78][2]];
    signal tmp_6145[3] <== CMul()(evals[7], tmp_6144);
    signal tmp_6146[3] <== [tmp_6143[0] + tmp_6145[0], tmp_6143[1] + tmp_6145[1], tmp_6143[2] + tmp_6145[2]];
    signal tmp_6147[3] <== CMul()(challengeQ, tmp_6146);
    signal tmp_6148[3] <== [evals[79][0] - publics[231], evals[79][1], evals[79][2]];
    signal tmp_6149[3] <== CMul()(evals[7], tmp_6148);
    signal tmp_6150[3] <== [tmp_6147[0] + tmp_6149[0], tmp_6147[1] + tmp_6149[1], tmp_6147[2] + tmp_6149[2]];
    signal tmp_6151[3] <== CMul()(challengeQ, tmp_6150);
    signal tmp_6152[3] <== [evals[80][0] - publics[232], evals[80][1], evals[80][2]];
    signal tmp_6153[3] <== CMul()(evals[7], tmp_6152);
    signal tmp_6154[3] <== [tmp_6151[0] + tmp_6153[0], tmp_6151[1] + tmp_6153[1], tmp_6151[2] + tmp_6153[2]];
    signal tmp_6155[3] <== CMul()(challengeQ, tmp_6154);
    signal tmp_6156[3] <== [evals[81][0] - publics[233], evals[81][1], evals[81][2]];
    signal tmp_6157[3] <== CMul()(evals[7], tmp_6156);
    signal tmp_6158[3] <== [tmp_6155[0] + tmp_6157[0], tmp_6155[1] + tmp_6157[1], tmp_6155[2] + tmp_6157[2]];
    signal tmp_6159[3] <== CMul()(challengeQ, tmp_6158);
    signal tmp_6160[3] <== [evals[82][0] - publics[234], evals[82][1], evals[82][2]];
    signal tmp_6161[3] <== CMul()(evals[7], tmp_6160);
    signal tmp_6162[3] <== [tmp_6159[0] + tmp_6161[0], tmp_6159[1] + tmp_6161[1], tmp_6159[2] + tmp_6161[2]];
    signal tmp_6163[3] <== CMul()(challengeQ, tmp_6162);
    signal tmp_6164[3] <== [evals[83][0] - publics[235], evals[83][1], evals[83][2]];
    signal tmp_6165[3] <== CMul()(evals[7], tmp_6164);
    signal tmp_6166[3] <== [tmp_6163[0] + tmp_6165[0], tmp_6163[1] + tmp_6165[1], tmp_6163[2] + tmp_6165[2]];
    signal tmp_6167[3] <== CMul()(challengeQ, tmp_6166);
    signal tmp_6168[3] <== [evals[84][0] - publics[236], evals[84][1], evals[84][2]];
    signal tmp_6169[3] <== CMul()(evals[7], tmp_6168);
    signal tmp_6170[3] <== [tmp_6167[0] + tmp_6169[0], tmp_6167[1] + tmp_6169[1], tmp_6167[2] + tmp_6169[2]];
    signal tmp_6171[3] <== CMul()(challengeQ, tmp_6170);
    signal tmp_6172[3] <== [evals[85][0] - publics[237], evals[85][1], evals[85][2]];
    signal tmp_6173[3] <== CMul()(evals[7], tmp_6172);
    signal tmp_6174[3] <== [tmp_6171[0] + tmp_6173[0], tmp_6171[1] + tmp_6173[1], tmp_6171[2] + tmp_6173[2]];
    signal tmp_6175[3] <== CMul()(challengeQ, tmp_6174);
    signal tmp_6176[3] <== [evals[86][0] - publics[238], evals[86][1], evals[86][2]];
    signal tmp_6177[3] <== CMul()(evals[7], tmp_6176);
    signal tmp_6178[3] <== [tmp_6175[0] + tmp_6177[0], tmp_6175[1] + tmp_6177[1], tmp_6175[2] + tmp_6177[2]];
    signal tmp_6179[3] <== CMul()(challengeQ, tmp_6178);
    signal tmp_6180[3] <== [evals[87][0] - publics[239], evals[87][1], evals[87][2]];
    signal tmp_6181[3] <== CMul()(evals[7], tmp_6180);
    signal tmp_6182[3] <== [tmp_6179[0] + tmp_6181[0], tmp_6179[1] + tmp_6181[1], tmp_6179[2] + tmp_6181[2]];
    signal tmp_6183[3] <== CMul()(challengeQ, tmp_6182);
    signal tmp_6184[3] <== [evals[64][0] - publics[240], evals[64][1], evals[64][2]];
    signal tmp_6185[3] <== CMul()(evals[6], tmp_6184);
    signal tmp_6186[3] <== [tmp_6183[0] + tmp_6185[0], tmp_6183[1] + tmp_6185[1], tmp_6183[2] + tmp_6185[2]];
    signal tmp_6187[3] <== CMul()(challengeQ, tmp_6186);
    signal tmp_6188[3] <== [evals[65][0] - publics[241], evals[65][1], evals[65][2]];
    signal tmp_6189[3] <== CMul()(evals[6], tmp_6188);
    signal tmp_6190[3] <== [tmp_6187[0] + tmp_6189[0], tmp_6187[1] + tmp_6189[1], tmp_6187[2] + tmp_6189[2]];
    signal tmp_6191[3] <== CMul()(challengeQ, tmp_6190);
    signal tmp_6192[3] <== [evals[66][0] - publics[242], evals[66][1], evals[66][2]];
    signal tmp_6193[3] <== CMul()(evals[6], tmp_6192);
    signal tmp_6194[3] <== [tmp_6191[0] + tmp_6193[0], tmp_6191[1] + tmp_6193[1], tmp_6191[2] + tmp_6193[2]];
    signal tmp_6195[3] <== CMul()(challengeQ, tmp_6194);
    signal tmp_6196[3] <== [evals[67][0] - publics[243], evals[67][1], evals[67][2]];
    signal tmp_6197[3] <== CMul()(evals[6], tmp_6196);
    signal tmp_6198[3] <== [tmp_6195[0] + tmp_6197[0], tmp_6195[1] + tmp_6197[1], tmp_6195[2] + tmp_6197[2]];
    signal tmp_6199[3] <== CMul()(challengeQ, tmp_6198);
    signal tmp_6200[3] <== [evals[68][0] - publics[244], evals[68][1], evals[68][2]];
    signal tmp_6201[3] <== CMul()(evals[6], tmp_6200);
    signal tmp_6202[3] <== [tmp_6199[0] + tmp_6201[0], tmp_6199[1] + tmp_6201[1], tmp_6199[2] + tmp_6201[2]];
    signal tmp_6203[3] <== CMul()(challengeQ, tmp_6202);
    signal tmp_6204[3] <== [evals[69][0] - publics[245], evals[69][1], evals[69][2]];
    signal tmp_6205[3] <== CMul()(evals[6], tmp_6204);
    signal tmp_6206[3] <== [tmp_6203[0] + tmp_6205[0], tmp_6203[1] + tmp_6205[1], tmp_6203[2] + tmp_6205[2]];
    signal tmp_6207[3] <== CMul()(challengeQ, tmp_6206);
    signal tmp_6208[3] <== [evals[70][0] - publics[246], evals[70][1], evals[70][2]];
    signal tmp_6209[3] <== CMul()(evals[6], tmp_6208);
    signal tmp_6210[3] <== [tmp_6207[0] + tmp_6209[0], tmp_6207[1] + tmp_6209[1], tmp_6207[2] + tmp_6209[2]];
    signal tmp_6211[3] <== CMul()(challengeQ, tmp_6210);
    signal tmp_6212[3] <== [evals[71][0] - publics[247], evals[71][1], evals[71][2]];
    signal tmp_6213[3] <== CMul()(evals[6], tmp_6212);
    signal tmp_6214[3] <== [tmp_6211[0] + tmp_6213[0], tmp_6211[1] + tmp_6213[1], tmp_6211[2] + tmp_6213[2]];
    signal tmp_6215[3] <== CMul()(challengeQ, tmp_6214);
    signal tmp_6216[3] <== [evals[72][0] - publics[248], evals[72][1], evals[72][2]];
    signal tmp_6217[3] <== CMul()(evals[6], tmp_6216);
    signal tmp_6218[3] <== [tmp_6215[0] + tmp_6217[0], tmp_6215[1] + tmp_6217[1], tmp_6215[2] + tmp_6217[2]];
    signal tmp_6219[3] <== CMul()(challengeQ, tmp_6218);
    signal tmp_6220[3] <== [evals[73][0] - publics[249], evals[73][1], evals[73][2]];
    signal tmp_6221[3] <== CMul()(evals[6], tmp_6220);
    signal tmp_6222[3] <== [tmp_6219[0] + tmp_6221[0], tmp_6219[1] + tmp_6221[1], tmp_6219[2] + tmp_6221[2]];
    tmp_6223 <== CMul()(challengeQ, tmp_6222);
}

template VerifyEvaluationsChunks1_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_6223[3];

    signal output tmp_6837[3];
    signal output tmp_6849[3];
    signal output tmp_6867[3];
    signal output tmp_6873[3];
    signal output tmp_6936[3];
    signal output tmp_7303[3];
    signal output tmp_7307[3];
    signal output tmp_7322[3];
    signal output tmp_7323[3];
    signal output tmp_7332[3];
    signal output tmp_7334[3];
    signal output tmp_7335[3];
    signal tmp_6224[3] <== [evals[74][0] - publics[250], evals[74][1], evals[74][2]];
    signal tmp_6225[3] <== CMul()(evals[6], tmp_6224);
    signal tmp_6226[3] <== [tmp_6223[0] + tmp_6225[0], tmp_6223[1] + tmp_6225[1], tmp_6223[2] + tmp_6225[2]];
    signal tmp_6227[3] <== CMul()(challengeQ, tmp_6226);
    signal tmp_6228[3] <== [evals[75][0] - publics[251], evals[75][1], evals[75][2]];
    signal tmp_6229[3] <== CMul()(evals[6], tmp_6228);
    signal tmp_6230[3] <== [tmp_6227[0] + tmp_6229[0], tmp_6227[1] + tmp_6229[1], tmp_6227[2] + tmp_6229[2]];
    signal tmp_6231[3] <== CMul()(challengeQ, tmp_6230);
    signal tmp_6232[3] <== [evals[76][0] - publics[252], evals[76][1], evals[76][2]];
    signal tmp_6233[3] <== CMul()(evals[6], tmp_6232);
    signal tmp_6234[3] <== [tmp_6231[0] + tmp_6233[0], tmp_6231[1] + tmp_6233[1], tmp_6231[2] + tmp_6233[2]];
    signal tmp_6235[3] <== CMul()(challengeQ, tmp_6234);
    signal tmp_6236[3] <== [evals[77][0] - publics[253], evals[77][1], evals[77][2]];
    signal tmp_6237[3] <== CMul()(evals[6], tmp_6236);
    signal tmp_6238[3] <== [tmp_6235[0] + tmp_6237[0], tmp_6235[1] + tmp_6237[1], tmp_6235[2] + tmp_6237[2]];
    signal tmp_6239[3] <== CMul()(challengeQ, tmp_6238);
    signal tmp_6240[3] <== [evals[78][0] - publics[254], evals[78][1], evals[78][2]];
    signal tmp_6241[3] <== CMul()(evals[6], tmp_6240);
    signal tmp_6242[3] <== [tmp_6239[0] + tmp_6241[0], tmp_6239[1] + tmp_6241[1], tmp_6239[2] + tmp_6241[2]];
    signal tmp_6243[3] <== CMul()(challengeQ, tmp_6242);
    signal tmp_6244[3] <== [evals[79][0] - publics[255], evals[79][1], evals[79][2]];
    signal tmp_6245[3] <== CMul()(evals[6], tmp_6244);
    signal tmp_6246[3] <== [tmp_6243[0] + tmp_6245[0], tmp_6243[1] + tmp_6245[1], tmp_6243[2] + tmp_6245[2]];
    signal tmp_6247[3] <== CMul()(challengeQ, tmp_6246);
    signal tmp_6248[3] <== [evals[80][0] - publics[256], evals[80][1], evals[80][2]];
    signal tmp_6249[3] <== CMul()(evals[6], tmp_6248);
    signal tmp_6250[3] <== [tmp_6247[0] + tmp_6249[0], tmp_6247[1] + tmp_6249[1], tmp_6247[2] + tmp_6249[2]];
    signal tmp_6251[3] <== CMul()(challengeQ, tmp_6250);
    signal tmp_6252[3] <== [evals[81][0] - publics[257], evals[81][1], evals[81][2]];
    signal tmp_6253[3] <== CMul()(evals[6], tmp_6252);
    signal tmp_6254[3] <== [tmp_6251[0] + tmp_6253[0], tmp_6251[1] + tmp_6253[1], tmp_6251[2] + tmp_6253[2]];
    signal tmp_6255[3] <== CMul()(challengeQ, tmp_6254);
    signal tmp_6256[3] <== [evals[82][0] - publics[258], evals[82][1], evals[82][2]];
    signal tmp_6257[3] <== CMul()(evals[6], tmp_6256);
    signal tmp_6258[3] <== [tmp_6255[0] + tmp_6257[0], tmp_6255[1] + tmp_6257[1], tmp_6255[2] + tmp_6257[2]];
    signal tmp_6259[3] <== CMul()(challengeQ, tmp_6258);
    signal tmp_6260[3] <== [evals[83][0] - publics[259], evals[83][1], evals[83][2]];
    signal tmp_6261[3] <== CMul()(evals[6], tmp_6260);
    signal tmp_6262[3] <== [tmp_6259[0] + tmp_6261[0], tmp_6259[1] + tmp_6261[1], tmp_6259[2] + tmp_6261[2]];
    signal tmp_6263[3] <== CMul()(challengeQ, tmp_6262);
    signal tmp_6264[3] <== [evals[84][0] - publics[260], evals[84][1], evals[84][2]];
    signal tmp_6265[3] <== CMul()(evals[6], tmp_6264);
    signal tmp_6266[3] <== [tmp_6263[0] + tmp_6265[0], tmp_6263[1] + tmp_6265[1], tmp_6263[2] + tmp_6265[2]];
    signal tmp_6267[3] <== CMul()(challengeQ, tmp_6266);
    signal tmp_6268[3] <== [evals[85][0] - publics[261], evals[85][1], evals[85][2]];
    signal tmp_6269[3] <== CMul()(evals[6], tmp_6268);
    signal tmp_6270[3] <== [tmp_6267[0] + tmp_6269[0], tmp_6267[1] + tmp_6269[1], tmp_6267[2] + tmp_6269[2]];
    signal tmp_6271[3] <== CMul()(challengeQ, tmp_6270);
    signal tmp_6272[3] <== [evals[86][0] - publics[262], evals[86][1], evals[86][2]];
    signal tmp_6273[3] <== CMul()(evals[6], tmp_6272);
    signal tmp_6274[3] <== [tmp_6271[0] + tmp_6273[0], tmp_6271[1] + tmp_6273[1], tmp_6271[2] + tmp_6273[2]];
    signal tmp_6275[3] <== CMul()(challengeQ, tmp_6274);
    signal tmp_6276[3] <== [evals[87][0] - publics[263], evals[87][1], evals[87][2]];
    signal tmp_6277[3] <== CMul()(evals[6], tmp_6276);
    signal tmp_6278[3] <== [tmp_6275[0] + tmp_6277[0], tmp_6275[1] + tmp_6277[1], tmp_6275[2] + tmp_6277[2]];
    signal tmp_6279[3] <== CMul()(challengeQ, tmp_6278);
    signal tmp_6280[3] <== [evals[64][0] - publics[264], evals[64][1], evals[64][2]];
    signal tmp_6281[3] <== CMul()(evals[5], tmp_6280);
    signal tmp_6282[3] <== [tmp_6279[0] + tmp_6281[0], tmp_6279[1] + tmp_6281[1], tmp_6279[2] + tmp_6281[2]];
    signal tmp_6283[3] <== CMul()(challengeQ, tmp_6282);
    signal tmp_6284[3] <== [evals[65][0] - publics[265], evals[65][1], evals[65][2]];
    signal tmp_6285[3] <== CMul()(evals[5], tmp_6284);
    signal tmp_6286[3] <== [tmp_6283[0] + tmp_6285[0], tmp_6283[1] + tmp_6285[1], tmp_6283[2] + tmp_6285[2]];
    signal tmp_6287[3] <== CMul()(challengeQ, tmp_6286);
    signal tmp_6288[3] <== [evals[66][0] - publics[266], evals[66][1], evals[66][2]];
    signal tmp_6289[3] <== CMul()(evals[5], tmp_6288);
    signal tmp_6290[3] <== [tmp_6287[0] + tmp_6289[0], tmp_6287[1] + tmp_6289[1], tmp_6287[2] + tmp_6289[2]];
    signal tmp_6291[3] <== CMul()(challengeQ, tmp_6290);
    signal tmp_6292[3] <== [evals[67][0] - publics[267], evals[67][1], evals[67][2]];
    signal tmp_6293[3] <== CMul()(evals[5], tmp_6292);
    signal tmp_6294[3] <== [tmp_6291[0] + tmp_6293[0], tmp_6291[1] + tmp_6293[1], tmp_6291[2] + tmp_6293[2]];
    signal tmp_6295[3] <== CMul()(challengeQ, tmp_6294);
    signal tmp_6296[3] <== [evals[68][0] - publics[268], evals[68][1], evals[68][2]];
    signal tmp_6297[3] <== CMul()(evals[5], tmp_6296);
    signal tmp_6298[3] <== [tmp_6295[0] + tmp_6297[0], tmp_6295[1] + tmp_6297[1], tmp_6295[2] + tmp_6297[2]];
    signal tmp_6299[3] <== CMul()(challengeQ, tmp_6298);
    signal tmp_6300[3] <== [evals[69][0] - publics[269], evals[69][1], evals[69][2]];
    signal tmp_6301[3] <== CMul()(evals[5], tmp_6300);
    signal tmp_6302[3] <== [tmp_6299[0] + tmp_6301[0], tmp_6299[1] + tmp_6301[1], tmp_6299[2] + tmp_6301[2]];
    signal tmp_6303[3] <== CMul()(challengeQ, tmp_6302);
    signal tmp_6304[3] <== [evals[70][0] - publics[270], evals[70][1], evals[70][2]];
    signal tmp_6305[3] <== CMul()(evals[5], tmp_6304);
    signal tmp_6306[3] <== [tmp_6303[0] + tmp_6305[0], tmp_6303[1] + tmp_6305[1], tmp_6303[2] + tmp_6305[2]];
    signal tmp_6307[3] <== CMul()(challengeQ, tmp_6306);
    signal tmp_6308[3] <== [evals[71][0] - publics[271], evals[71][1], evals[71][2]];
    signal tmp_6309[3] <== CMul()(evals[5], tmp_6308);
    signal tmp_6310[3] <== [tmp_6307[0] + tmp_6309[0], tmp_6307[1] + tmp_6309[1], tmp_6307[2] + tmp_6309[2]];
    signal tmp_6311[3] <== CMul()(challengeQ, tmp_6310);
    signal tmp_6312[3] <== [evals[72][0] - publics[272], evals[72][1], evals[72][2]];
    signal tmp_6313[3] <== CMul()(evals[5], tmp_6312);
    signal tmp_6314[3] <== [tmp_6311[0] + tmp_6313[0], tmp_6311[1] + tmp_6313[1], tmp_6311[2] + tmp_6313[2]];
    signal tmp_6315[3] <== CMul()(challengeQ, tmp_6314);
    signal tmp_6316[3] <== [evals[73][0] - publics[273], evals[73][1], evals[73][2]];
    signal tmp_6317[3] <== CMul()(evals[5], tmp_6316);
    signal tmp_6318[3] <== [tmp_6315[0] + tmp_6317[0], tmp_6315[1] + tmp_6317[1], tmp_6315[2] + tmp_6317[2]];
    signal tmp_6319[3] <== CMul()(challengeQ, tmp_6318);
    signal tmp_6320[3] <== [evals[74][0] - publics[274], evals[74][1], evals[74][2]];
    signal tmp_6321[3] <== CMul()(evals[5], tmp_6320);
    signal tmp_6322[3] <== [tmp_6319[0] + tmp_6321[0], tmp_6319[1] + tmp_6321[1], tmp_6319[2] + tmp_6321[2]];
    signal tmp_6323[3] <== CMul()(challengeQ, tmp_6322);
    signal tmp_6324[3] <== [evals[75][0] - publics[275], evals[75][1], evals[75][2]];
    signal tmp_6325[3] <== CMul()(evals[5], tmp_6324);
    signal tmp_6326[3] <== [tmp_6323[0] + tmp_6325[0], tmp_6323[1] + tmp_6325[1], tmp_6323[2] + tmp_6325[2]];
    signal tmp_6327[3] <== CMul()(challengeQ, tmp_6326);
    signal tmp_6328[3] <== [evals[76][0] - publics[276], evals[76][1], evals[76][2]];
    signal tmp_6329[3] <== CMul()(evals[5], tmp_6328);
    signal tmp_6330[3] <== [tmp_6327[0] + tmp_6329[0], tmp_6327[1] + tmp_6329[1], tmp_6327[2] + tmp_6329[2]];
    signal tmp_6331[3] <== CMul()(challengeQ, tmp_6330);
    signal tmp_6332[3] <== [evals[77][0] - publics[277], evals[77][1], evals[77][2]];
    signal tmp_6333[3] <== CMul()(evals[5], tmp_6332);
    signal tmp_6334[3] <== [tmp_6331[0] + tmp_6333[0], tmp_6331[1] + tmp_6333[1], tmp_6331[2] + tmp_6333[2]];
    signal tmp_6335[3] <== CMul()(challengeQ, tmp_6334);
    signal tmp_6336[3] <== [evals[78][0] - publics[278], evals[78][1], evals[78][2]];
    signal tmp_6337[3] <== CMul()(evals[5], tmp_6336);
    signal tmp_6338[3] <== [tmp_6335[0] + tmp_6337[0], tmp_6335[1] + tmp_6337[1], tmp_6335[2] + tmp_6337[2]];
    signal tmp_6339[3] <== CMul()(challengeQ, tmp_6338);
    signal tmp_6340[3] <== [evals[79][0] - publics[279], evals[79][1], evals[79][2]];
    signal tmp_6341[3] <== CMul()(evals[5], tmp_6340);
    signal tmp_6342[3] <== [tmp_6339[0] + tmp_6341[0], tmp_6339[1] + tmp_6341[1], tmp_6339[2] + tmp_6341[2]];
    signal tmp_6343[3] <== CMul()(challengeQ, tmp_6342);
    signal tmp_6344[3] <== [evals[80][0] - publics[280], evals[80][1], evals[80][2]];
    signal tmp_6345[3] <== CMul()(evals[5], tmp_6344);
    signal tmp_6346[3] <== [tmp_6343[0] + tmp_6345[0], tmp_6343[1] + tmp_6345[1], tmp_6343[2] + tmp_6345[2]];
    signal tmp_6347[3] <== CMul()(challengeQ, tmp_6346);
    signal tmp_6348[3] <== [evals[81][0] - publics[281], evals[81][1], evals[81][2]];
    signal tmp_6349[3] <== CMul()(evals[5], tmp_6348);
    signal tmp_6350[3] <== [tmp_6347[0] + tmp_6349[0], tmp_6347[1] + tmp_6349[1], tmp_6347[2] + tmp_6349[2]];
    signal tmp_6351[3] <== CMul()(challengeQ, tmp_6350);
    signal tmp_6352[3] <== [evals[82][0] - publics[282], evals[82][1], evals[82][2]];
    signal tmp_6353[3] <== CMul()(evals[5], tmp_6352);
    signal tmp_6354[3] <== [tmp_6351[0] + tmp_6353[0], tmp_6351[1] + tmp_6353[1], tmp_6351[2] + tmp_6353[2]];
    signal tmp_6355[3] <== CMul()(challengeQ, tmp_6354);
    signal tmp_6356[3] <== [evals[83][0] - publics[283], evals[83][1], evals[83][2]];
    signal tmp_6357[3] <== CMul()(evals[5], tmp_6356);
    signal tmp_6358[3] <== [tmp_6355[0] + tmp_6357[0], tmp_6355[1] + tmp_6357[1], tmp_6355[2] + tmp_6357[2]];
    signal tmp_6359[3] <== CMul()(challengeQ, tmp_6358);
    signal tmp_6360[3] <== [evals[84][0] - publics[284], evals[84][1], evals[84][2]];
    signal tmp_6361[3] <== CMul()(evals[5], tmp_6360);
    signal tmp_6362[3] <== [tmp_6359[0] + tmp_6361[0], tmp_6359[1] + tmp_6361[1], tmp_6359[2] + tmp_6361[2]];
    signal tmp_6363[3] <== CMul()(challengeQ, tmp_6362);
    signal tmp_6364[3] <== [evals[85][0] - publics[285], evals[85][1], evals[85][2]];
    signal tmp_6365[3] <== CMul()(evals[5], tmp_6364);
    signal tmp_6366[3] <== [tmp_6363[0] + tmp_6365[0], tmp_6363[1] + tmp_6365[1], tmp_6363[2] + tmp_6365[2]];
    signal tmp_6367[3] <== CMul()(challengeQ, tmp_6366);
    signal tmp_6368[3] <== [evals[86][0] - publics[286], evals[86][1], evals[86][2]];
    signal tmp_6369[3] <== CMul()(evals[5], tmp_6368);
    signal tmp_6370[3] <== [tmp_6367[0] + tmp_6369[0], tmp_6367[1] + tmp_6369[1], tmp_6367[2] + tmp_6369[2]];
    signal tmp_6371[3] <== CMul()(challengeQ, tmp_6370);
    signal tmp_6372[3] <== [evals[87][0] - publics[287], evals[87][1], evals[87][2]];
    signal tmp_6373[3] <== CMul()(evals[5], tmp_6372);
    signal tmp_6374[3] <== [tmp_6371[0] + tmp_6373[0], tmp_6371[1] + tmp_6373[1], tmp_6371[2] + tmp_6373[2]];
    signal tmp_6375[3] <== CMul()(challengeQ, tmp_6374);
    signal tmp_6376[3] <== [evals[64][0] - publics[288], evals[64][1], evals[64][2]];
    signal tmp_6377[3] <== CMul()(evals[4], tmp_6376);
    signal tmp_6378[3] <== [tmp_6375[0] + tmp_6377[0], tmp_6375[1] + tmp_6377[1], tmp_6375[2] + tmp_6377[2]];
    signal tmp_6379[3] <== CMul()(challengeQ, tmp_6378);
    signal tmp_6380[3] <== [evals[65][0] - publics[289], evals[65][1], evals[65][2]];
    signal tmp_6381[3] <== CMul()(evals[4], tmp_6380);
    signal tmp_6382[3] <== [tmp_6379[0] + tmp_6381[0], tmp_6379[1] + tmp_6381[1], tmp_6379[2] + tmp_6381[2]];
    signal tmp_6383[3] <== CMul()(challengeQ, tmp_6382);
    signal tmp_6384[3] <== [evals[66][0] - publics[290], evals[66][1], evals[66][2]];
    signal tmp_6385[3] <== CMul()(evals[4], tmp_6384);
    signal tmp_6386[3] <== [tmp_6383[0] + tmp_6385[0], tmp_6383[1] + tmp_6385[1], tmp_6383[2] + tmp_6385[2]];
    signal tmp_6387[3] <== CMul()(challengeQ, tmp_6386);
    signal tmp_6388[3] <== [evals[67][0] - publics[291], evals[67][1], evals[67][2]];
    signal tmp_6389[3] <== CMul()(evals[4], tmp_6388);
    signal tmp_6390[3] <== [tmp_6387[0] + tmp_6389[0], tmp_6387[1] + tmp_6389[1], tmp_6387[2] + tmp_6389[2]];
    signal tmp_6391[3] <== CMul()(challengeQ, tmp_6390);
    signal tmp_6392[3] <== [evals[68][0] - publics[292], evals[68][1], evals[68][2]];
    signal tmp_6393[3] <== CMul()(evals[4], tmp_6392);
    signal tmp_6394[3] <== [tmp_6391[0] + tmp_6393[0], tmp_6391[1] + tmp_6393[1], tmp_6391[2] + tmp_6393[2]];
    signal tmp_6395[3] <== CMul()(challengeQ, tmp_6394);
    signal tmp_6396[3] <== [evals[69][0] - publics[293], evals[69][1], evals[69][2]];
    signal tmp_6397[3] <== CMul()(evals[4], tmp_6396);
    signal tmp_6398[3] <== [tmp_6395[0] + tmp_6397[0], tmp_6395[1] + tmp_6397[1], tmp_6395[2] + tmp_6397[2]];
    signal tmp_6399[3] <== CMul()(challengeQ, tmp_6398);
    signal tmp_6400[3] <== [evals[70][0] - publics[294], evals[70][1], evals[70][2]];
    signal tmp_6401[3] <== CMul()(evals[4], tmp_6400);
    signal tmp_6402[3] <== [tmp_6399[0] + tmp_6401[0], tmp_6399[1] + tmp_6401[1], tmp_6399[2] + tmp_6401[2]];
    signal tmp_6403[3] <== CMul()(challengeQ, tmp_6402);
    signal tmp_6404[3] <== [evals[71][0] - publics[295], evals[71][1], evals[71][2]];
    signal tmp_6405[3] <== CMul()(evals[4], tmp_6404);
    signal tmp_6406[3] <== [tmp_6403[0] + tmp_6405[0], tmp_6403[1] + tmp_6405[1], tmp_6403[2] + tmp_6405[2]];
    signal tmp_6407[3] <== CMul()(challengeQ, tmp_6406);
    signal tmp_6408[3] <== [evals[72][0] - publics[296], evals[72][1], evals[72][2]];
    signal tmp_6409[3] <== CMul()(evals[4], tmp_6408);
    signal tmp_6410[3] <== [tmp_6407[0] + tmp_6409[0], tmp_6407[1] + tmp_6409[1], tmp_6407[2] + tmp_6409[2]];
    signal tmp_6411[3] <== CMul()(challengeQ, tmp_6410);
    signal tmp_6412[3] <== [evals[73][0] - publics[297], evals[73][1], evals[73][2]];
    signal tmp_6413[3] <== CMul()(evals[4], tmp_6412);
    signal tmp_6414[3] <== [tmp_6411[0] + tmp_6413[0], tmp_6411[1] + tmp_6413[1], tmp_6411[2] + tmp_6413[2]];
    signal tmp_6415[3] <== CMul()(challengeQ, tmp_6414);
    signal tmp_6416[3] <== [evals[74][0] - publics[298], evals[74][1], evals[74][2]];
    signal tmp_6417[3] <== CMul()(evals[4], tmp_6416);
    signal tmp_6418[3] <== [tmp_6415[0] + tmp_6417[0], tmp_6415[1] + tmp_6417[1], tmp_6415[2] + tmp_6417[2]];
    signal tmp_6419[3] <== CMul()(challengeQ, tmp_6418);
    signal tmp_6420[3] <== [evals[75][0] - publics[299], evals[75][1], evals[75][2]];
    signal tmp_6421[3] <== CMul()(evals[4], tmp_6420);
    signal tmp_6422[3] <== [tmp_6419[0] + tmp_6421[0], tmp_6419[1] + tmp_6421[1], tmp_6419[2] + tmp_6421[2]];
    signal tmp_6423[3] <== CMul()(challengeQ, tmp_6422);
    signal tmp_6424[3] <== [evals[76][0] - publics[300], evals[76][1], evals[76][2]];
    signal tmp_6425[3] <== CMul()(evals[4], tmp_6424);
    signal tmp_6426[3] <== [tmp_6423[0] + tmp_6425[0], tmp_6423[1] + tmp_6425[1], tmp_6423[2] + tmp_6425[2]];
    signal tmp_6427[3] <== CMul()(challengeQ, tmp_6426);
    signal tmp_6428[3] <== [evals[77][0] - publics[301], evals[77][1], evals[77][2]];
    signal tmp_6429[3] <== CMul()(evals[4], tmp_6428);
    signal tmp_6430[3] <== [tmp_6427[0] + tmp_6429[0], tmp_6427[1] + tmp_6429[1], tmp_6427[2] + tmp_6429[2]];
    signal tmp_6431[3] <== CMul()(challengeQ, tmp_6430);
    signal tmp_6432[3] <== [evals[78][0] - publics[302], evals[78][1], evals[78][2]];
    signal tmp_6433[3] <== CMul()(evals[4], tmp_6432);
    signal tmp_6434[3] <== [tmp_6431[0] + tmp_6433[0], tmp_6431[1] + tmp_6433[1], tmp_6431[2] + tmp_6433[2]];
    signal tmp_6435[3] <== CMul()(challengeQ, tmp_6434);
    signal tmp_6436[3] <== [evals[79][0] - publics[303], evals[79][1], evals[79][2]];
    signal tmp_6437[3] <== CMul()(evals[4], tmp_6436);
    signal tmp_6438[3] <== [tmp_6435[0] + tmp_6437[0], tmp_6435[1] + tmp_6437[1], tmp_6435[2] + tmp_6437[2]];
    signal tmp_6439[3] <== CMul()(challengeQ, tmp_6438);
    signal tmp_6440[3] <== [evals[80][0] - publics[304], evals[80][1], evals[80][2]];
    signal tmp_6441[3] <== CMul()(evals[4], tmp_6440);
    signal tmp_6442[3] <== [tmp_6439[0] + tmp_6441[0], tmp_6439[1] + tmp_6441[1], tmp_6439[2] + tmp_6441[2]];
    signal tmp_6443[3] <== CMul()(challengeQ, tmp_6442);
    signal tmp_6444[3] <== [evals[81][0] - publics[305], evals[81][1], evals[81][2]];
    signal tmp_6445[3] <== CMul()(evals[4], tmp_6444);
    signal tmp_6446[3] <== [tmp_6443[0] + tmp_6445[0], tmp_6443[1] + tmp_6445[1], tmp_6443[2] + tmp_6445[2]];
    signal tmp_6447[3] <== CMul()(challengeQ, tmp_6446);
    signal tmp_6448[3] <== [evals[82][0] - publics[306], evals[82][1], evals[82][2]];
    signal tmp_6449[3] <== CMul()(evals[4], tmp_6448);
    signal tmp_6450[3] <== [tmp_6447[0] + tmp_6449[0], tmp_6447[1] + tmp_6449[1], tmp_6447[2] + tmp_6449[2]];
    signal tmp_6451[3] <== CMul()(challengeQ, tmp_6450);
    signal tmp_6452[3] <== [evals[83][0] - publics[307], evals[83][1], evals[83][2]];
    signal tmp_6453[3] <== CMul()(evals[4], tmp_6452);
    signal tmp_6454[3] <== [tmp_6451[0] + tmp_6453[0], tmp_6451[1] + tmp_6453[1], tmp_6451[2] + tmp_6453[2]];
    signal tmp_6455[3] <== CMul()(challengeQ, tmp_6454);
    signal tmp_6456[3] <== [evals[84][0] - publics[308], evals[84][1], evals[84][2]];
    signal tmp_6457[3] <== CMul()(evals[4], tmp_6456);
    signal tmp_6458[3] <== [tmp_6455[0] + tmp_6457[0], tmp_6455[1] + tmp_6457[1], tmp_6455[2] + tmp_6457[2]];
    signal tmp_6459[3] <== CMul()(challengeQ, tmp_6458);
    signal tmp_6460[3] <== [evals[85][0] - publics[309], evals[85][1], evals[85][2]];
    signal tmp_6461[3] <== CMul()(evals[4], tmp_6460);
    signal tmp_6462[3] <== [tmp_6459[0] + tmp_6461[0], tmp_6459[1] + tmp_6461[1], tmp_6459[2] + tmp_6461[2]];
    signal tmp_6463[3] <== CMul()(challengeQ, tmp_6462);
    signal tmp_6464[3] <== [evals[86][0] - publics[310], evals[86][1], evals[86][2]];
    signal tmp_6465[3] <== CMul()(evals[4], tmp_6464);
    signal tmp_6466[3] <== [tmp_6463[0] + tmp_6465[0], tmp_6463[1] + tmp_6465[1], tmp_6463[2] + tmp_6465[2]];
    signal tmp_6467[3] <== CMul()(challengeQ, tmp_6466);
    signal tmp_6468[3] <== [evals[87][0] - publics[311], evals[87][1], evals[87][2]];
    signal tmp_6469[3] <== CMul()(evals[4], tmp_6468);
    signal tmp_6470[3] <== [tmp_6467[0] + tmp_6469[0], tmp_6467[1] + tmp_6469[1], tmp_6467[2] + tmp_6469[2]];
    signal tmp_6471[3] <== CMul()(challengeQ, tmp_6470);
    signal tmp_6472[3] <== [evals[64][0] - publics[312], evals[64][1], evals[64][2]];
    signal tmp_6473[3] <== CMul()(evals[3], tmp_6472);
    signal tmp_6474[3] <== [tmp_6471[0] + tmp_6473[0], tmp_6471[1] + tmp_6473[1], tmp_6471[2] + tmp_6473[2]];
    signal tmp_6475[3] <== CMul()(challengeQ, tmp_6474);
    signal tmp_6476[3] <== [evals[65][0] - publics[313], evals[65][1], evals[65][2]];
    signal tmp_6477[3] <== CMul()(evals[3], tmp_6476);
    signal tmp_6478[3] <== [tmp_6475[0] + tmp_6477[0], tmp_6475[1] + tmp_6477[1], tmp_6475[2] + tmp_6477[2]];
    signal tmp_6479[3] <== CMul()(challengeQ, tmp_6478);
    signal tmp_6480[3] <== [evals[66][0] - publics[314], evals[66][1], evals[66][2]];
    signal tmp_6481[3] <== CMul()(evals[3], tmp_6480);
    signal tmp_6482[3] <== [tmp_6479[0] + tmp_6481[0], tmp_6479[1] + tmp_6481[1], tmp_6479[2] + tmp_6481[2]];
    signal tmp_6483[3] <== CMul()(challengeQ, tmp_6482);
    signal tmp_6484[3] <== [evals[67][0] - publics[315], evals[67][1], evals[67][2]];
    signal tmp_6485[3] <== CMul()(evals[3], tmp_6484);
    signal tmp_6486[3] <== [tmp_6483[0] + tmp_6485[0], tmp_6483[1] + tmp_6485[1], tmp_6483[2] + tmp_6485[2]];
    signal tmp_6487[3] <== CMul()(challengeQ, tmp_6486);
    signal tmp_6488[3] <== [evals[68][0] - publics[316], evals[68][1], evals[68][2]];
    signal tmp_6489[3] <== CMul()(evals[3], tmp_6488);
    signal tmp_6490[3] <== [tmp_6487[0] + tmp_6489[0], tmp_6487[1] + tmp_6489[1], tmp_6487[2] + tmp_6489[2]];
    signal tmp_6491[3] <== CMul()(challengeQ, tmp_6490);
    signal tmp_6492[3] <== [evals[69][0] - publics[317], evals[69][1], evals[69][2]];
    signal tmp_6493[3] <== CMul()(evals[3], tmp_6492);
    signal tmp_6494[3] <== [tmp_6491[0] + tmp_6493[0], tmp_6491[1] + tmp_6493[1], tmp_6491[2] + tmp_6493[2]];
    signal tmp_6495[3] <== CMul()(challengeQ, tmp_6494);
    signal tmp_6496[3] <== [evals[70][0] - publics[318], evals[70][1], evals[70][2]];
    signal tmp_6497[3] <== CMul()(evals[3], tmp_6496);
    signal tmp_6498[3] <== [tmp_6495[0] + tmp_6497[0], tmp_6495[1] + tmp_6497[1], tmp_6495[2] + tmp_6497[2]];
    signal tmp_6499[3] <== CMul()(challengeQ, tmp_6498);
    signal tmp_6500[3] <== [evals[71][0] - publics[319], evals[71][1], evals[71][2]];
    signal tmp_6501[3] <== CMul()(evals[3], tmp_6500);
    signal tmp_6502[3] <== [tmp_6499[0] + tmp_6501[0], tmp_6499[1] + tmp_6501[1], tmp_6499[2] + tmp_6501[2]];
    signal tmp_6503[3] <== CMul()(challengeQ, tmp_6502);
    signal tmp_6504[3] <== [evals[72][0] - publics[320], evals[72][1], evals[72][2]];
    signal tmp_6505[3] <== CMul()(evals[3], tmp_6504);
    signal tmp_6506[3] <== [tmp_6503[0] + tmp_6505[0], tmp_6503[1] + tmp_6505[1], tmp_6503[2] + tmp_6505[2]];
    signal tmp_6507[3] <== CMul()(challengeQ, tmp_6506);
    signal tmp_6508[3] <== [evals[73][0] - publics[321], evals[73][1], evals[73][2]];
    signal tmp_6509[3] <== CMul()(evals[3], tmp_6508);
    signal tmp_6510[3] <== [tmp_6507[0] + tmp_6509[0], tmp_6507[1] + tmp_6509[1], tmp_6507[2] + tmp_6509[2]];
    signal tmp_6511[3] <== CMul()(challengeQ, tmp_6510);
    signal tmp_6512[3] <== [evals[74][0] - publics[322], evals[74][1], evals[74][2]];
    signal tmp_6513[3] <== CMul()(evals[3], tmp_6512);
    signal tmp_6514[3] <== [tmp_6511[0] + tmp_6513[0], tmp_6511[1] + tmp_6513[1], tmp_6511[2] + tmp_6513[2]];
    signal tmp_6515[3] <== CMul()(challengeQ, tmp_6514);
    signal tmp_6516[3] <== [evals[75][0] - publics[323], evals[75][1], evals[75][2]];
    signal tmp_6517[3] <== CMul()(evals[3], tmp_6516);
    signal tmp_6518[3] <== [tmp_6515[0] + tmp_6517[0], tmp_6515[1] + tmp_6517[1], tmp_6515[2] + tmp_6517[2]];
    signal tmp_6519[3] <== CMul()(challengeQ, tmp_6518);
    signal tmp_6520[3] <== [evals[76][0] - publics[324], evals[76][1], evals[76][2]];
    signal tmp_6521[3] <== CMul()(evals[3], tmp_6520);
    signal tmp_6522[3] <== [tmp_6519[0] + tmp_6521[0], tmp_6519[1] + tmp_6521[1], tmp_6519[2] + tmp_6521[2]];
    signal tmp_6523[3] <== CMul()(challengeQ, tmp_6522);
    signal tmp_6524[3] <== [evals[77][0] - publics[325], evals[77][1], evals[77][2]];
    signal tmp_6525[3] <== CMul()(evals[3], tmp_6524);
    signal tmp_6526[3] <== [tmp_6523[0] + tmp_6525[0], tmp_6523[1] + tmp_6525[1], tmp_6523[2] + tmp_6525[2]];
    signal tmp_6527[3] <== CMul()(challengeQ, tmp_6526);
    signal tmp_6528[3] <== [evals[78][0] - publics[326], evals[78][1], evals[78][2]];
    signal tmp_6529[3] <== CMul()(evals[3], tmp_6528);
    signal tmp_6530[3] <== [tmp_6527[0] + tmp_6529[0], tmp_6527[1] + tmp_6529[1], tmp_6527[2] + tmp_6529[2]];
    signal tmp_6531[3] <== CMul()(challengeQ, tmp_6530);
    signal tmp_6532[3] <== [evals[79][0] - publics[327], evals[79][1], evals[79][2]];
    signal tmp_6533[3] <== CMul()(evals[3], tmp_6532);
    signal tmp_6534[3] <== [tmp_6531[0] + tmp_6533[0], tmp_6531[1] + tmp_6533[1], tmp_6531[2] + tmp_6533[2]];
    signal tmp_6535[3] <== CMul()(challengeQ, tmp_6534);
    signal tmp_6536[3] <== [evals[80][0] - publics[328], evals[80][1], evals[80][2]];
    signal tmp_6537[3] <== CMul()(evals[3], tmp_6536);
    signal tmp_6538[3] <== [tmp_6535[0] + tmp_6537[0], tmp_6535[1] + tmp_6537[1], tmp_6535[2] + tmp_6537[2]];
    signal tmp_6539[3] <== CMul()(challengeQ, tmp_6538);
    signal tmp_6540[3] <== [evals[81][0] - publics[329], evals[81][1], evals[81][2]];
    signal tmp_6541[3] <== CMul()(evals[3], tmp_6540);
    signal tmp_6542[3] <== [tmp_6539[0] + tmp_6541[0], tmp_6539[1] + tmp_6541[1], tmp_6539[2] + tmp_6541[2]];
    signal tmp_6543[3] <== CMul()(challengeQ, tmp_6542);
    signal tmp_6544[3] <== [evals[82][0] - publics[330], evals[82][1], evals[82][2]];
    signal tmp_6545[3] <== CMul()(evals[3], tmp_6544);
    signal tmp_6546[3] <== [tmp_6543[0] + tmp_6545[0], tmp_6543[1] + tmp_6545[1], tmp_6543[2] + tmp_6545[2]];
    signal tmp_6547[3] <== CMul()(challengeQ, tmp_6546);
    signal tmp_6548[3] <== [evals[83][0] - publics[331], evals[83][1], evals[83][2]];
    signal tmp_6549[3] <== CMul()(evals[3], tmp_6548);
    signal tmp_6550[3] <== [tmp_6547[0] + tmp_6549[0], tmp_6547[1] + tmp_6549[1], tmp_6547[2] + tmp_6549[2]];
    signal tmp_6551[3] <== CMul()(challengeQ, tmp_6550);
    signal tmp_6552[3] <== [evals[84][0] - publics[332], evals[84][1], evals[84][2]];
    signal tmp_6553[3] <== CMul()(evals[3], tmp_6552);
    signal tmp_6554[3] <== [tmp_6551[0] + tmp_6553[0], tmp_6551[1] + tmp_6553[1], tmp_6551[2] + tmp_6553[2]];
    signal tmp_6555[3] <== CMul()(challengeQ, tmp_6554);
    signal tmp_6556[3] <== [evals[85][0] - publics[333], evals[85][1], evals[85][2]];
    signal tmp_6557[3] <== CMul()(evals[3], tmp_6556);
    signal tmp_6558[3] <== [tmp_6555[0] + tmp_6557[0], tmp_6555[1] + tmp_6557[1], tmp_6555[2] + tmp_6557[2]];
    signal tmp_6559[3] <== CMul()(challengeQ, tmp_6558);
    signal tmp_6560[3] <== [evals[86][0] - publics[334], evals[86][1], evals[86][2]];
    signal tmp_6561[3] <== CMul()(evals[3], tmp_6560);
    signal tmp_6562[3] <== [tmp_6559[0] + tmp_6561[0], tmp_6559[1] + tmp_6561[1], tmp_6559[2] + tmp_6561[2]];
    signal tmp_6563[3] <== CMul()(challengeQ, tmp_6562);
    signal tmp_6564[3] <== [evals[87][0] - publics[335], evals[87][1], evals[87][2]];
    signal tmp_6565[3] <== CMul()(evals[3], tmp_6564);
    signal tmp_6566[3] <== [tmp_6563[0] + tmp_6565[0], tmp_6563[1] + tmp_6565[1], tmp_6563[2] + tmp_6565[2]];
    signal tmp_6567[3] <== CMul()(challengeQ, tmp_6566);
    signal tmp_6568[3] <== [evals[64][0] - publics[336], evals[64][1], evals[64][2]];
    signal tmp_6569[3] <== CMul()(evals[2], tmp_6568);
    signal tmp_6570[3] <== [tmp_6567[0] + tmp_6569[0], tmp_6567[1] + tmp_6569[1], tmp_6567[2] + tmp_6569[2]];
    signal tmp_6571[3] <== CMul()(challengeQ, tmp_6570);
    signal tmp_6572[3] <== [evals[65][0] - publics[337], evals[65][1], evals[65][2]];
    signal tmp_6573[3] <== CMul()(evals[2], tmp_6572);
    signal tmp_6574[3] <== [tmp_6571[0] + tmp_6573[0], tmp_6571[1] + tmp_6573[1], tmp_6571[2] + tmp_6573[2]];
    signal tmp_6575[3] <== CMul()(challengeQ, tmp_6574);
    signal tmp_6576[3] <== [evals[66][0] - publics[338], evals[66][1], evals[66][2]];
    signal tmp_6577[3] <== CMul()(evals[2], tmp_6576);
    signal tmp_6578[3] <== [tmp_6575[0] + tmp_6577[0], tmp_6575[1] + tmp_6577[1], tmp_6575[2] + tmp_6577[2]];
    signal tmp_6579[3] <== CMul()(challengeQ, tmp_6578);
    signal tmp_6580[3] <== [evals[67][0] - publics[339], evals[67][1], evals[67][2]];
    signal tmp_6581[3] <== CMul()(evals[2], tmp_6580);
    signal tmp_6582[3] <== [tmp_6579[0] + tmp_6581[0], tmp_6579[1] + tmp_6581[1], tmp_6579[2] + tmp_6581[2]];
    signal tmp_6583[3] <== CMul()(challengeQ, tmp_6582);
    signal tmp_6584[3] <== [evals[68][0] - publics[340], evals[68][1], evals[68][2]];
    signal tmp_6585[3] <== CMul()(evals[2], tmp_6584);
    signal tmp_6586[3] <== [tmp_6583[0] + tmp_6585[0], tmp_6583[1] + tmp_6585[1], tmp_6583[2] + tmp_6585[2]];
    signal tmp_6587[3] <== CMul()(challengeQ, tmp_6586);
    signal tmp_6588[3] <== [evals[69][0] - publics[341], evals[69][1], evals[69][2]];
    signal tmp_6589[3] <== CMul()(evals[2], tmp_6588);
    signal tmp_6590[3] <== [tmp_6587[0] + tmp_6589[0], tmp_6587[1] + tmp_6589[1], tmp_6587[2] + tmp_6589[2]];
    signal tmp_6591[3] <== CMul()(challengeQ, tmp_6590);
    signal tmp_6592[3] <== [evals[70][0] - publics[342], evals[70][1], evals[70][2]];
    signal tmp_6593[3] <== CMul()(evals[2], tmp_6592);
    signal tmp_6594[3] <== [tmp_6591[0] + tmp_6593[0], tmp_6591[1] + tmp_6593[1], tmp_6591[2] + tmp_6593[2]];
    signal tmp_6595[3] <== CMul()(challengeQ, tmp_6594);
    signal tmp_6596[3] <== [evals[71][0] - publics[343], evals[71][1], evals[71][2]];
    signal tmp_6597[3] <== CMul()(evals[2], tmp_6596);
    signal tmp_6598[3] <== [tmp_6595[0] + tmp_6597[0], tmp_6595[1] + tmp_6597[1], tmp_6595[2] + tmp_6597[2]];
    signal tmp_6599[3] <== CMul()(challengeQ, tmp_6598);
    signal tmp_6600[3] <== [evals[72][0] - publics[344], evals[72][1], evals[72][2]];
    signal tmp_6601[3] <== CMul()(evals[2], tmp_6600);
    signal tmp_6602[3] <== [tmp_6599[0] + tmp_6601[0], tmp_6599[1] + tmp_6601[1], tmp_6599[2] + tmp_6601[2]];
    signal tmp_6603[3] <== CMul()(challengeQ, tmp_6602);
    signal tmp_6604[3] <== [evals[73][0] - publics[345], evals[73][1], evals[73][2]];
    signal tmp_6605[3] <== CMul()(evals[2], tmp_6604);
    signal tmp_6606[3] <== [tmp_6603[0] + tmp_6605[0], tmp_6603[1] + tmp_6605[1], tmp_6603[2] + tmp_6605[2]];
    signal tmp_6607[3] <== CMul()(challengeQ, tmp_6606);
    signal tmp_6608[3] <== [evals[74][0] - publics[346], evals[74][1], evals[74][2]];
    signal tmp_6609[3] <== CMul()(evals[2], tmp_6608);
    signal tmp_6610[3] <== [tmp_6607[0] + tmp_6609[0], tmp_6607[1] + tmp_6609[1], tmp_6607[2] + tmp_6609[2]];
    signal tmp_6611[3] <== CMul()(challengeQ, tmp_6610);
    signal tmp_6612[3] <== [evals[75][0] - publics[347], evals[75][1], evals[75][2]];
    signal tmp_6613[3] <== CMul()(evals[2], tmp_6612);
    signal tmp_6614[3] <== [tmp_6611[0] + tmp_6613[0], tmp_6611[1] + tmp_6613[1], tmp_6611[2] + tmp_6613[2]];
    signal tmp_6615[3] <== CMul()(challengeQ, tmp_6614);
    signal tmp_6616[3] <== [evals[76][0] - publics[348], evals[76][1], evals[76][2]];
    signal tmp_6617[3] <== CMul()(evals[2], tmp_6616);
    signal tmp_6618[3] <== [tmp_6615[0] + tmp_6617[0], tmp_6615[1] + tmp_6617[1], tmp_6615[2] + tmp_6617[2]];
    signal tmp_6619[3] <== CMul()(challengeQ, tmp_6618);
    signal tmp_6620[3] <== [evals[77][0] - publics[349], evals[77][1], evals[77][2]];
    signal tmp_6621[3] <== CMul()(evals[2], tmp_6620);
    signal tmp_6622[3] <== [tmp_6619[0] + tmp_6621[0], tmp_6619[1] + tmp_6621[1], tmp_6619[2] + tmp_6621[2]];
    signal tmp_6623[3] <== CMul()(challengeQ, tmp_6622);
    signal tmp_6624[3] <== [evals[78][0] - publics[350], evals[78][1], evals[78][2]];
    signal tmp_6625[3] <== CMul()(evals[2], tmp_6624);
    signal tmp_6626[3] <== [tmp_6623[0] + tmp_6625[0], tmp_6623[1] + tmp_6625[1], tmp_6623[2] + tmp_6625[2]];
    signal tmp_6627[3] <== CMul()(challengeQ, tmp_6626);
    signal tmp_6628[3] <== [evals[79][0] - publics[351], evals[79][1], evals[79][2]];
    signal tmp_6629[3] <== CMul()(evals[2], tmp_6628);
    signal tmp_6630[3] <== [tmp_6627[0] + tmp_6629[0], tmp_6627[1] + tmp_6629[1], tmp_6627[2] + tmp_6629[2]];
    signal tmp_6631[3] <== CMul()(challengeQ, tmp_6630);
    signal tmp_6632[3] <== [evals[80][0] - publics[352], evals[80][1], evals[80][2]];
    signal tmp_6633[3] <== CMul()(evals[2], tmp_6632);
    signal tmp_6634[3] <== [tmp_6631[0] + tmp_6633[0], tmp_6631[1] + tmp_6633[1], tmp_6631[2] + tmp_6633[2]];
    signal tmp_6635[3] <== CMul()(challengeQ, tmp_6634);
    signal tmp_6636[3] <== [evals[81][0] - publics[353], evals[81][1], evals[81][2]];
    signal tmp_6637[3] <== CMul()(evals[2], tmp_6636);
    signal tmp_6638[3] <== [tmp_6635[0] + tmp_6637[0], tmp_6635[1] + tmp_6637[1], tmp_6635[2] + tmp_6637[2]];
    signal tmp_6639[3] <== CMul()(challengeQ, tmp_6638);
    signal tmp_6640[3] <== [evals[82][0] - publics[354], evals[82][1], evals[82][2]];
    signal tmp_6641[3] <== CMul()(evals[2], tmp_6640);
    signal tmp_6642[3] <== [tmp_6639[0] + tmp_6641[0], tmp_6639[1] + tmp_6641[1], tmp_6639[2] + tmp_6641[2]];
    signal tmp_6643[3] <== CMul()(challengeQ, tmp_6642);
    signal tmp_6644[3] <== [evals[83][0] - publics[355], evals[83][1], evals[83][2]];
    signal tmp_6645[3] <== CMul()(evals[2], tmp_6644);
    signal tmp_6646[3] <== [tmp_6643[0] + tmp_6645[0], tmp_6643[1] + tmp_6645[1], tmp_6643[2] + tmp_6645[2]];
    signal tmp_6647[3] <== CMul()(challengeQ, tmp_6646);
    signal tmp_6648[3] <== [evals[84][0] - publics[356], evals[84][1], evals[84][2]];
    signal tmp_6649[3] <== CMul()(evals[2], tmp_6648);
    signal tmp_6650[3] <== [tmp_6647[0] + tmp_6649[0], tmp_6647[1] + tmp_6649[1], tmp_6647[2] + tmp_6649[2]];
    signal tmp_6651[3] <== CMul()(challengeQ, tmp_6650);
    signal tmp_6652[3] <== [evals[85][0] - publics[357], evals[85][1], evals[85][2]];
    signal tmp_6653[3] <== CMul()(evals[2], tmp_6652);
    signal tmp_6654[3] <== [tmp_6651[0] + tmp_6653[0], tmp_6651[1] + tmp_6653[1], tmp_6651[2] + tmp_6653[2]];
    signal tmp_6655[3] <== CMul()(challengeQ, tmp_6654);
    signal tmp_6656[3] <== [evals[86][0] - publics[358], evals[86][1], evals[86][2]];
    signal tmp_6657[3] <== CMul()(evals[2], tmp_6656);
    signal tmp_6658[3] <== [tmp_6655[0] + tmp_6657[0], tmp_6655[1] + tmp_6657[1], tmp_6655[2] + tmp_6657[2]];
    signal tmp_6659[3] <== CMul()(challengeQ, tmp_6658);
    signal tmp_6660[3] <== [evals[87][0] - publics[359], evals[87][1], evals[87][2]];
    signal tmp_6661[3] <== CMul()(evals[2], tmp_6660);
    signal tmp_6662[3] <== [tmp_6659[0] + tmp_6661[0], tmp_6659[1] + tmp_6661[1], tmp_6659[2] + tmp_6661[2]];
    signal tmp_6663[3] <== CMul()(challengeQ, tmp_6662);
    signal tmp_6664[3] <== [evals[64][0] - publics[360], evals[64][1], evals[64][2]];
    signal tmp_6665[3] <== CMul()(evals[1], tmp_6664);
    signal tmp_6666[3] <== [tmp_6663[0] + tmp_6665[0], tmp_6663[1] + tmp_6665[1], tmp_6663[2] + tmp_6665[2]];
    signal tmp_6667[3] <== CMul()(challengeQ, tmp_6666);
    signal tmp_6668[3] <== [evals[65][0] - publics[361], evals[65][1], evals[65][2]];
    signal tmp_6669[3] <== CMul()(evals[1], tmp_6668);
    signal tmp_6670[3] <== [tmp_6667[0] + tmp_6669[0], tmp_6667[1] + tmp_6669[1], tmp_6667[2] + tmp_6669[2]];
    signal tmp_6671[3] <== CMul()(challengeQ, tmp_6670);
    signal tmp_6672[3] <== [evals[66][0] - publics[362], evals[66][1], evals[66][2]];
    signal tmp_6673[3] <== CMul()(evals[1], tmp_6672);
    signal tmp_6674[3] <== [tmp_6671[0] + tmp_6673[0], tmp_6671[1] + tmp_6673[1], tmp_6671[2] + tmp_6673[2]];
    signal tmp_6675[3] <== CMul()(challengeQ, tmp_6674);
    signal tmp_6676[3] <== [evals[67][0] - publics[363], evals[67][1], evals[67][2]];
    signal tmp_6677[3] <== CMul()(evals[1], tmp_6676);
    signal tmp_6678[3] <== [tmp_6675[0] + tmp_6677[0], tmp_6675[1] + tmp_6677[1], tmp_6675[2] + tmp_6677[2]];
    signal tmp_6679[3] <== CMul()(challengeQ, tmp_6678);
    signal tmp_6680[3] <== [evals[68][0] - publics[364], evals[68][1], evals[68][2]];
    signal tmp_6681[3] <== CMul()(evals[1], tmp_6680);
    signal tmp_6682[3] <== [tmp_6679[0] + tmp_6681[0], tmp_6679[1] + tmp_6681[1], tmp_6679[2] + tmp_6681[2]];
    signal tmp_6683[3] <== CMul()(challengeQ, tmp_6682);
    signal tmp_6684[3] <== [evals[69][0] - publics[365], evals[69][1], evals[69][2]];
    signal tmp_6685[3] <== CMul()(evals[1], tmp_6684);
    signal tmp_6686[3] <== [tmp_6683[0] + tmp_6685[0], tmp_6683[1] + tmp_6685[1], tmp_6683[2] + tmp_6685[2]];
    signal tmp_6687[3] <== CMul()(challengeQ, tmp_6686);
    signal tmp_6688[3] <== [evals[70][0] - publics[366], evals[70][1], evals[70][2]];
    signal tmp_6689[3] <== CMul()(evals[1], tmp_6688);
    signal tmp_6690[3] <== [tmp_6687[0] + tmp_6689[0], tmp_6687[1] + tmp_6689[1], tmp_6687[2] + tmp_6689[2]];
    signal tmp_6691[3] <== CMul()(challengeQ, tmp_6690);
    signal tmp_6692[3] <== [evals[71][0] - publics[367], evals[71][1], evals[71][2]];
    signal tmp_6693[3] <== CMul()(evals[1], tmp_6692);
    signal tmp_6694[3] <== [tmp_6691[0] + tmp_6693[0], tmp_6691[1] + tmp_6693[1], tmp_6691[2] + tmp_6693[2]];
    signal tmp_6695[3] <== CMul()(challengeQ, tmp_6694);
    signal tmp_6696[3] <== [evals[72][0] - publics[368], evals[72][1], evals[72][2]];
    signal tmp_6697[3] <== CMul()(evals[1], tmp_6696);
    signal tmp_6698[3] <== [tmp_6695[0] + tmp_6697[0], tmp_6695[1] + tmp_6697[1], tmp_6695[2] + tmp_6697[2]];
    signal tmp_6699[3] <== CMul()(challengeQ, tmp_6698);
    signal tmp_6700[3] <== [evals[73][0] - publics[369], evals[73][1], evals[73][2]];
    signal tmp_6701[3] <== CMul()(evals[1], tmp_6700);
    signal tmp_6702[3] <== [tmp_6699[0] + tmp_6701[0], tmp_6699[1] + tmp_6701[1], tmp_6699[2] + tmp_6701[2]];
    signal tmp_6703[3] <== CMul()(challengeQ, tmp_6702);
    signal tmp_6704[3] <== [evals[74][0] - publics[370], evals[74][1], evals[74][2]];
    signal tmp_6705[3] <== CMul()(evals[1], tmp_6704);
    signal tmp_6706[3] <== [tmp_6703[0] + tmp_6705[0], tmp_6703[1] + tmp_6705[1], tmp_6703[2] + tmp_6705[2]];
    signal tmp_6707[3] <== CMul()(challengeQ, tmp_6706);
    signal tmp_6708[3] <== [evals[75][0] - publics[371], evals[75][1], evals[75][2]];
    signal tmp_6709[3] <== CMul()(evals[1], tmp_6708);
    signal tmp_6710[3] <== [tmp_6707[0] + tmp_6709[0], tmp_6707[1] + tmp_6709[1], tmp_6707[2] + tmp_6709[2]];
    signal tmp_6711[3] <== CMul()(challengeQ, tmp_6710);
    signal tmp_6712[3] <== [evals[76][0] - publics[372], evals[76][1], evals[76][2]];
    signal tmp_6713[3] <== CMul()(evals[1], tmp_6712);
    signal tmp_6714[3] <== [tmp_6711[0] + tmp_6713[0], tmp_6711[1] + tmp_6713[1], tmp_6711[2] + tmp_6713[2]];
    signal tmp_6715[3] <== CMul()(challengeQ, tmp_6714);
    signal tmp_6716[3] <== [evals[77][0] - publics[373], evals[77][1], evals[77][2]];
    signal tmp_6717[3] <== CMul()(evals[1], tmp_6716);
    signal tmp_6718[3] <== [tmp_6715[0] + tmp_6717[0], tmp_6715[1] + tmp_6717[1], tmp_6715[2] + tmp_6717[2]];
    signal tmp_6719[3] <== CMul()(challengeQ, tmp_6718);
    signal tmp_6720[3] <== [evals[78][0] - publics[374], evals[78][1], evals[78][2]];
    signal tmp_6721[3] <== CMul()(evals[1], tmp_6720);
    signal tmp_6722[3] <== [tmp_6719[0] + tmp_6721[0], tmp_6719[1] + tmp_6721[1], tmp_6719[2] + tmp_6721[2]];
    signal tmp_6723[3] <== CMul()(challengeQ, tmp_6722);
    signal tmp_6724[3] <== [evals[79][0] - publics[375], evals[79][1], evals[79][2]];
    signal tmp_6725[3] <== CMul()(evals[1], tmp_6724);
    signal tmp_6726[3] <== [tmp_6723[0] + tmp_6725[0], tmp_6723[1] + tmp_6725[1], tmp_6723[2] + tmp_6725[2]];
    signal tmp_6727[3] <== CMul()(challengeQ, tmp_6726);
    signal tmp_6728[3] <== [evals[80][0] - publics[376], evals[80][1], evals[80][2]];
    signal tmp_6729[3] <== CMul()(evals[1], tmp_6728);
    signal tmp_6730[3] <== [tmp_6727[0] + tmp_6729[0], tmp_6727[1] + tmp_6729[1], tmp_6727[2] + tmp_6729[2]];
    signal tmp_6731[3] <== CMul()(challengeQ, tmp_6730);
    signal tmp_6732[3] <== [evals[81][0] - publics[377], evals[81][1], evals[81][2]];
    signal tmp_6733[3] <== CMul()(evals[1], tmp_6732);
    signal tmp_6734[3] <== [tmp_6731[0] + tmp_6733[0], tmp_6731[1] + tmp_6733[1], tmp_6731[2] + tmp_6733[2]];
    signal tmp_6735[3] <== CMul()(challengeQ, tmp_6734);
    signal tmp_6736[3] <== [evals[82][0] - publics[378], evals[82][1], evals[82][2]];
    signal tmp_6737[3] <== CMul()(evals[1], tmp_6736);
    signal tmp_6738[3] <== [tmp_6735[0] + tmp_6737[0], tmp_6735[1] + tmp_6737[1], tmp_6735[2] + tmp_6737[2]];
    signal tmp_6739[3] <== CMul()(challengeQ, tmp_6738);
    signal tmp_6740[3] <== [evals[83][0] - publics[379], evals[83][1], evals[83][2]];
    signal tmp_6741[3] <== CMul()(evals[1], tmp_6740);
    signal tmp_6742[3] <== [tmp_6739[0] + tmp_6741[0], tmp_6739[1] + tmp_6741[1], tmp_6739[2] + tmp_6741[2]];
    signal tmp_6743[3] <== CMul()(challengeQ, tmp_6742);
    signal tmp_6744[3] <== [evals[84][0] - publics[380], evals[84][1], evals[84][2]];
    signal tmp_6745[3] <== CMul()(evals[1], tmp_6744);
    signal tmp_6746[3] <== [tmp_6743[0] + tmp_6745[0], tmp_6743[1] + tmp_6745[1], tmp_6743[2] + tmp_6745[2]];
    signal tmp_6747[3] <== CMul()(challengeQ, tmp_6746);
    signal tmp_6748[3] <== [evals[85][0] - publics[381], evals[85][1], evals[85][2]];
    signal tmp_6749[3] <== CMul()(evals[1], tmp_6748);
    signal tmp_6750[3] <== [tmp_6747[0] + tmp_6749[0], tmp_6747[1] + tmp_6749[1], tmp_6747[2] + tmp_6749[2]];
    signal tmp_6751[3] <== CMul()(challengeQ, tmp_6750);
    signal tmp_6752[3] <== [evals[86][0] - publics[382], evals[86][1], evals[86][2]];
    signal tmp_6753[3] <== CMul()(evals[1], tmp_6752);
    signal tmp_6754[3] <== [tmp_6751[0] + tmp_6753[0], tmp_6751[1] + tmp_6753[1], tmp_6751[2] + tmp_6753[2]];
    signal tmp_6755[3] <== CMul()(challengeQ, tmp_6754);
    signal tmp_6756[3] <== [evals[87][0] - publics[383], evals[87][1], evals[87][2]];
    signal tmp_6757[3] <== CMul()(evals[1], tmp_6756);
    signal tmp_6758[3] <== [tmp_6755[0] + tmp_6757[0], tmp_6755[1] + tmp_6757[1], tmp_6755[2] + tmp_6757[2]];
    signal tmp_6759[3] <== CMul()(challengeQ, tmp_6758);
    signal tmp_6760[3] <== [evals[64][0] - publics[384], evals[64][1], evals[64][2]];
    signal tmp_6761[3] <== CMul()(evals[0], tmp_6760);
    signal tmp_6762[3] <== [tmp_6759[0] + tmp_6761[0], tmp_6759[1] + tmp_6761[1], tmp_6759[2] + tmp_6761[2]];
    signal tmp_6763[3] <== CMul()(challengeQ, tmp_6762);
    signal tmp_6764[3] <== [evals[65][0] - publics[385], evals[65][1], evals[65][2]];
    signal tmp_6765[3] <== CMul()(evals[0], tmp_6764);
    signal tmp_6766[3] <== [tmp_6763[0] + tmp_6765[0], tmp_6763[1] + tmp_6765[1], tmp_6763[2] + tmp_6765[2]];
    signal tmp_6767[3] <== CMul()(challengeQ, tmp_6766);
    signal tmp_6768[3] <== [evals[66][0] - publics[386], evals[66][1], evals[66][2]];
    signal tmp_6769[3] <== CMul()(evals[0], tmp_6768);
    signal tmp_6770[3] <== [tmp_6767[0] + tmp_6769[0], tmp_6767[1] + tmp_6769[1], tmp_6767[2] + tmp_6769[2]];
    signal tmp_6771[3] <== CMul()(challengeQ, tmp_6770);
    signal tmp_6772[3] <== [evals[67][0] - publics[387], evals[67][1], evals[67][2]];
    signal tmp_6773[3] <== CMul()(evals[0], tmp_6772);
    signal tmp_6774[3] <== [tmp_6771[0] + tmp_6773[0], tmp_6771[1] + tmp_6773[1], tmp_6771[2] + tmp_6773[2]];
    signal tmp_6775[3] <== CMul()(challengeQ, tmp_6774);
    signal tmp_6776[3] <== [evals[68][0] - publics[388], evals[68][1], evals[68][2]];
    signal tmp_6777[3] <== CMul()(evals[0], tmp_6776);
    signal tmp_6778[3] <== [tmp_6775[0] + tmp_6777[0], tmp_6775[1] + tmp_6777[1], tmp_6775[2] + tmp_6777[2]];
    signal tmp_6779[3] <== CMul()(challengeQ, tmp_6778);
    signal tmp_6780[3] <== [evals[69][0] - publics[389], evals[69][1], evals[69][2]];
    signal tmp_6781[3] <== CMul()(evals[0], tmp_6780);
    signal tmp_6782[3] <== [tmp_6779[0] + tmp_6781[0], tmp_6779[1] + tmp_6781[1], tmp_6779[2] + tmp_6781[2]];
    signal tmp_6783[3] <== CMul()(challengeQ, tmp_6782);
    signal tmp_6784[3] <== [evals[70][0] - publics[390], evals[70][1], evals[70][2]];
    signal tmp_6785[3] <== CMul()(evals[0], tmp_6784);
    signal tmp_6786[3] <== [tmp_6783[0] + tmp_6785[0], tmp_6783[1] + tmp_6785[1], tmp_6783[2] + tmp_6785[2]];
    signal tmp_6787[3] <== CMul()(challengeQ, tmp_6786);
    signal tmp_6788[3] <== [evals[71][0] - publics[391], evals[71][1], evals[71][2]];
    signal tmp_6789[3] <== CMul()(evals[0], tmp_6788);
    signal tmp_6790[3] <== [tmp_6787[0] + tmp_6789[0], tmp_6787[1] + tmp_6789[1], tmp_6787[2] + tmp_6789[2]];
    signal tmp_6791[3] <== CMul()(challengeQ, tmp_6790);
    signal tmp_6792[3] <== [evals[72][0] - publics[392], evals[72][1], evals[72][2]];
    signal tmp_6793[3] <== CMul()(evals[0], tmp_6792);
    signal tmp_6794[3] <== [tmp_6791[0] + tmp_6793[0], tmp_6791[1] + tmp_6793[1], tmp_6791[2] + tmp_6793[2]];
    signal tmp_6795[3] <== CMul()(challengeQ, tmp_6794);
    signal tmp_6796[3] <== [evals[73][0] - publics[393], evals[73][1], evals[73][2]];
    signal tmp_6797[3] <== CMul()(evals[0], tmp_6796);
    signal tmp_6798[3] <== [tmp_6795[0] + tmp_6797[0], tmp_6795[1] + tmp_6797[1], tmp_6795[2] + tmp_6797[2]];
    signal tmp_6799[3] <== CMul()(challengeQ, tmp_6798);
    signal tmp_6800[3] <== [evals[74][0] - publics[394], evals[74][1], evals[74][2]];
    signal tmp_6801[3] <== CMul()(evals[0], tmp_6800);
    signal tmp_6802[3] <== [tmp_6799[0] + tmp_6801[0], tmp_6799[1] + tmp_6801[1], tmp_6799[2] + tmp_6801[2]];
    signal tmp_6803[3] <== CMul()(challengeQ, tmp_6802);
    signal tmp_6804[3] <== [evals[75][0] - publics[395], evals[75][1], evals[75][2]];
    signal tmp_6805[3] <== CMul()(evals[0], tmp_6804);
    signal tmp_6806[3] <== [tmp_6803[0] + tmp_6805[0], tmp_6803[1] + tmp_6805[1], tmp_6803[2] + tmp_6805[2]];
    signal tmp_6807[3] <== CMul()(challengeQ, tmp_6806);
    signal tmp_6808[3] <== [evals[76][0] - publics[396], evals[76][1], evals[76][2]];
    signal tmp_6809[3] <== CMul()(evals[0], tmp_6808);
    signal tmp_6810[3] <== [tmp_6807[0] + tmp_6809[0], tmp_6807[1] + tmp_6809[1], tmp_6807[2] + tmp_6809[2]];
    signal tmp_6811[3] <== CMul()(challengeQ, tmp_6810);
    signal tmp_6812[3] <== [evals[77][0] - publics[397], evals[77][1], evals[77][2]];
    signal tmp_6813[3] <== CMul()(evals[0], tmp_6812);
    signal tmp_6814[3] <== [tmp_6811[0] + tmp_6813[0], tmp_6811[1] + tmp_6813[1], tmp_6811[2] + tmp_6813[2]];
    signal tmp_6815[3] <== CMul()(challengeQ, tmp_6814);
    signal tmp_6816[3] <== [evals[78][0] - publics[398], evals[78][1], evals[78][2]];
    signal tmp_6817[3] <== CMul()(evals[0], tmp_6816);
    signal tmp_6818[3] <== [tmp_6815[0] + tmp_6817[0], tmp_6815[1] + tmp_6817[1], tmp_6815[2] + tmp_6817[2]];
    signal tmp_6819[3] <== CMul()(challengeQ, tmp_6818);
    signal tmp_6820[3] <== [evals[60][0] + evals[123][0], evals[60][1] + evals[123][1], evals[60][2] + evals[123][2]];
    signal tmp_6821[3] <== [tmp_6820[0] + evals[53][0], tmp_6820[1] + evals[53][1], tmp_6820[2] + evals[53][2]];
    signal tmp_6822[3] <== [tmp_6821[0] + evals[124][0], tmp_6821[1] + evals[124][1], tmp_6821[2] + evals[124][2]];
    signal tmp_6823[3] <== CMul()(evals[64], evals[65]);
    signal tmp_6824[3] <== CMul()(evals[41], tmp_6823);
    signal tmp_6825[3] <== CMul()(evals[42], evals[64]);
    signal tmp_6826[3] <== [tmp_6824[0] + tmp_6825[0], tmp_6824[1] + tmp_6825[1], tmp_6824[2] + tmp_6825[2]];
    signal tmp_6827[3] <== CMul()(evals[43], evals[65]);
    signal tmp_6828[3] <== [tmp_6826[0] + tmp_6827[0], tmp_6826[1] + tmp_6827[1], tmp_6826[2] + tmp_6827[2]];
    signal tmp_6829[3] <== CMul()(evals[44], evals[66]);
    signal tmp_6830[3] <== [tmp_6828[0] + tmp_6829[0], tmp_6828[1] + tmp_6829[1], tmp_6828[2] + tmp_6829[2]];
    signal tmp_6831[3] <== [tmp_6830[0] + evals[45][0], tmp_6830[1] + evals[45][1], tmp_6830[2] + evals[45][2]];
    signal tmp_6832[3] <== CMul()(tmp_6822, tmp_6831);
    signal tmp_6833[3] <== [tmp_6819[0] + tmp_6832[0], tmp_6819[1] + tmp_6832[1], tmp_6819[2] + tmp_6832[2]];
    signal tmp_6834[3] <== CMul()(challengeQ, tmp_6833);
    signal tmp_6835[3] <== CMul()(evals[67], evals[68]);
    signal tmp_6836[3] <== CMul()(evals[41], tmp_6835);
    tmp_6837 <== CMul()(evals[42], evals[67]);
    signal tmp_6838[3] <== [tmp_6836[0] + tmp_6837[0], tmp_6836[1] + tmp_6837[1], tmp_6836[2] + tmp_6837[2]];
    signal tmp_6839[3] <== CMul()(evals[43], evals[68]);
    signal tmp_6840[3] <== [tmp_6838[0] + tmp_6839[0], tmp_6838[1] + tmp_6839[1], tmp_6838[2] + tmp_6839[2]];
    signal tmp_6841[3] <== CMul()(evals[44], evals[69]);
    signal tmp_6842[3] <== [tmp_6840[0] + tmp_6841[0], tmp_6840[1] + tmp_6841[1], tmp_6840[2] + tmp_6841[2]];
    signal tmp_6843[3] <== [tmp_6842[0] + evals[45][0], tmp_6842[1] + evals[45][1], tmp_6842[2] + evals[45][2]];
    signal tmp_6844[3] <== CMul()(tmp_6822, tmp_6843);
    signal tmp_6845[3] <== [tmp_6834[0] + tmp_6844[0], tmp_6834[1] + tmp_6844[1], tmp_6834[2] + tmp_6844[2]];
    signal tmp_6846[3] <== CMul()(challengeQ, tmp_6845);
    signal tmp_6847[3] <== CMul()(evals[70], evals[71]);
    signal tmp_6848[3] <== CMul()(evals[46], tmp_6847);
    tmp_6849 <== CMul()(evals[47], evals[70]);
    signal tmp_6850[3] <== [tmp_6848[0] + tmp_6849[0], tmp_6848[1] + tmp_6849[1], tmp_6848[2] + tmp_6849[2]];
    signal tmp_6851[3] <== CMul()(evals[48], evals[71]);
    signal tmp_6852[3] <== [tmp_6850[0] + tmp_6851[0], tmp_6850[1] + tmp_6851[1], tmp_6850[2] + tmp_6851[2]];
    signal tmp_6853[3] <== CMul()(evals[49], evals[72]);
    signal tmp_6854[3] <== [tmp_6852[0] + tmp_6853[0], tmp_6852[1] + tmp_6853[1], tmp_6852[2] + tmp_6853[2]];
    signal tmp_6855[3] <== [tmp_6854[0] + evals[50][0], tmp_6854[1] + evals[50][1], tmp_6854[2] + evals[50][2]];
    signal tmp_6856[3] <== CMul()(tmp_6822, tmp_6855);
    signal tmp_6857[3] <== [tmp_6846[0] + tmp_6856[0], tmp_6846[1] + tmp_6856[1], tmp_6846[2] + tmp_6856[2]];
    signal tmp_6858[3] <== CMul()(challengeQ, tmp_6857);
    signal tmp_6860[3] <== [tmp_6820[0] + evals[124][0], tmp_6820[1] + evals[124][1], tmp_6820[2] + evals[124][2]];
    signal tmp_6861[3] <== CMul()(evals[73], evals[74]);
    signal tmp_6862[3] <== CMul()(evals[46], tmp_6861);
    signal tmp_6863[3] <== CMul()(evals[47], evals[73]);
    signal tmp_6864[3] <== [tmp_6862[0] + tmp_6863[0], tmp_6862[1] + tmp_6863[1], tmp_6862[2] + tmp_6863[2]];
    signal tmp_6865[3] <== CMul()(evals[48], evals[74]);
    signal tmp_6866[3] <== [tmp_6864[0] + tmp_6865[0], tmp_6864[1] + tmp_6865[1], tmp_6864[2] + tmp_6865[2]];
    tmp_6867 <== CMul()(evals[49], evals[75]);
    signal tmp_6868[3] <== [tmp_6866[0] + tmp_6867[0], tmp_6866[1] + tmp_6867[1], tmp_6866[2] + tmp_6867[2]];
    signal tmp_6869[3] <== [tmp_6868[0] + evals[50][0], tmp_6868[1] + evals[50][1], tmp_6868[2] + evals[50][2]];
    signal tmp_6870[3] <== CMul()(tmp_6860, tmp_6869);
    signal tmp_6871[3] <== [tmp_6858[0] + tmp_6870[0], tmp_6858[1] + tmp_6870[1], tmp_6858[2] + tmp_6870[2]];
    signal tmp_6872[3] <== CMul()(challengeQ, tmp_6871);
    tmp_6873 <== CMul()(evals[76], evals[77]);
    signal tmp_6874[3] <== CMul()(evals[46], tmp_6873);
    signal tmp_6875[3] <== CMul()(evals[47], evals[76]);
    signal tmp_6876[3] <== [tmp_6874[0] + tmp_6875[0], tmp_6874[1] + tmp_6875[1], tmp_6874[2] + tmp_6875[2]];
    signal tmp_6877[3] <== CMul()(evals[48], evals[77]);
    signal tmp_6878[3] <== [tmp_6876[0] + tmp_6877[0], tmp_6876[1] + tmp_6877[1], tmp_6876[2] + tmp_6877[2]];
    signal tmp_6879[3] <== CMul()(evals[49], evals[78]);
    signal tmp_6880[3] <== [tmp_6878[0] + tmp_6879[0], tmp_6878[1] + tmp_6879[1], tmp_6878[2] + tmp_6879[2]];
    signal tmp_6881[3] <== [tmp_6880[0] + evals[50][0], tmp_6880[1] + evals[50][1], tmp_6880[2] + evals[50][2]];
    signal tmp_6882[3] <== CMul()(tmp_6860, tmp_6881);
    signal tmp_6883[3] <== [tmp_6872[0] + tmp_6882[0], tmp_6872[1] + tmp_6882[1], tmp_6872[2] + tmp_6882[2]];
    signal tmp_6884[3] <== CMul()(challengeQ, tmp_6883);
    signal tmp_6885[3] <== CMul()(evals[79], evals[80]);
    signal tmp_6886[3] <== CMul()(evals[46], tmp_6885);
    signal tmp_6887[3] <== CMul()(evals[47], evals[79]);
    signal tmp_6888[3] <== [tmp_6886[0] + tmp_6887[0], tmp_6886[1] + tmp_6887[1], tmp_6886[2] + tmp_6887[2]];
    signal tmp_6889[3] <== CMul()(evals[48], evals[80]);
    signal tmp_6890[3] <== [tmp_6888[0] + tmp_6889[0], tmp_6888[1] + tmp_6889[1], tmp_6888[2] + tmp_6889[2]];
    signal tmp_6891[3] <== CMul()(evals[49], evals[81]);
    signal tmp_6892[3] <== [tmp_6890[0] + tmp_6891[0], tmp_6890[1] + tmp_6891[1], tmp_6890[2] + tmp_6891[2]];
    signal tmp_6893[3] <== [tmp_6892[0] + evals[50][0], tmp_6892[1] + evals[50][1], tmp_6892[2] + evals[50][2]];
    signal tmp_6894[3] <== CMul()(evals[60], tmp_6893);
    signal tmp_6895[3] <== [tmp_6884[0] + tmp_6894[0], tmp_6884[1] + tmp_6894[1], tmp_6884[2] + tmp_6894[2]];
    signal tmp_6896[3] <== CMul()(challengeQ, tmp_6895);
    signal tmp_6897[3] <== [evals[60][0] + evals[55][0], evals[60][1] + evals[55][1], evals[60][2] + evals[55][2]];
    signal tmp_6898[3] <== [tmp_6897[0] + evals[58][0], tmp_6897[1] + evals[58][1], tmp_6897[2] + evals[58][2]];
    signal tmp_6899[3] <== CMul()(evals[82], evals[83]);
    signal tmp_6900[3] <== CMul()(evals[46], tmp_6899);
    signal tmp_6901[3] <== CMul()(evals[47], evals[82]);
    signal tmp_6902[3] <== [tmp_6900[0] + tmp_6901[0], tmp_6900[1] + tmp_6901[1], tmp_6900[2] + tmp_6901[2]];
    signal tmp_6903[3] <== CMul()(evals[48], evals[83]);
    signal tmp_6904[3] <== [tmp_6902[0] + tmp_6903[0], tmp_6902[1] + tmp_6903[1], tmp_6902[2] + tmp_6903[2]];
    signal tmp_6905[3] <== CMul()(evals[49], evals[84]);
    signal tmp_6906[3] <== [tmp_6904[0] + tmp_6905[0], tmp_6904[1] + tmp_6905[1], tmp_6904[2] + tmp_6905[2]];
    signal tmp_6907[3] <== [tmp_6906[0] + evals[50][0], tmp_6906[1] + evals[50][1], tmp_6906[2] + evals[50][2]];
    signal tmp_6908[3] <== CMul()(tmp_6898, tmp_6907);
    signal tmp_6909[3] <== [tmp_6896[0] + tmp_6908[0], tmp_6896[1] + tmp_6908[1], tmp_6896[2] + tmp_6908[2]];
    signal tmp_6910[3] <== CMul()(challengeQ, tmp_6909);
    signal tmp_6913[3] <== [tmp_6898[0] + evals[56][0], tmp_6898[1] + evals[56][1], tmp_6898[2] + evals[56][2]];
    signal tmp_6914[3] <== CMul()(evals[85], evals[86]);
    signal tmp_6915[3] <== CMul()(evals[46], tmp_6914);
    signal tmp_6916[3] <== CMul()(evals[47], evals[85]);
    signal tmp_6917[3] <== [tmp_6915[0] + tmp_6916[0], tmp_6915[1] + tmp_6916[1], tmp_6915[2] + tmp_6916[2]];
    signal tmp_6918[3] <== CMul()(evals[48], evals[86]);
    signal tmp_6919[3] <== [tmp_6917[0] + tmp_6918[0], tmp_6917[1] + tmp_6918[1], tmp_6917[2] + tmp_6918[2]];
    signal tmp_6920[3] <== CMul()(evals[49], evals[87]);
    signal tmp_6921[3] <== [tmp_6919[0] + tmp_6920[0], tmp_6919[1] + tmp_6920[1], tmp_6919[2] + tmp_6920[2]];
    signal tmp_6922[3] <== [tmp_6921[0] + evals[50][0], tmp_6921[1] + evals[50][1], tmp_6921[2] + evals[50][2]];
    signal tmp_6923[3] <== CMul()(tmp_6913, tmp_6922);
    signal tmp_6924[3] <== [tmp_6910[0] + tmp_6923[0], tmp_6910[1] + tmp_6923[1], tmp_6910[2] + tmp_6923[2]];
    signal tmp_6925[3] <== CMul()(challengeQ, tmp_6924);
    signal tmp_6926[3] <== CMul()(evals[52], evals[126]);
    signal tmp_6927[3] <== [evals[126][0] - 1, evals[126][1], evals[126][2]];
    signal tmp_6928[3] <== CMul()(tmp_6926, tmp_6927);
    signal tmp_6929[3] <== [tmp_6925[0] + tmp_6928[0], tmp_6925[1] + tmp_6928[1], tmp_6925[2] + tmp_6928[2]];
    signal tmp_6930[3] <== CMul()(challengeQ, tmp_6929);
    signal tmp_6931[3] <== CMul()(evals[52], evals[143]);
    signal tmp_6932[3] <== [evals[143][0] - 1, evals[143][1], evals[143][2]];
    signal tmp_6933[3] <== CMul()(tmp_6931, tmp_6932);
    signal tmp_6934[3] <== [tmp_6930[0] + tmp_6933[0], tmp_6930[1] + tmp_6933[1], tmp_6930[2] + tmp_6933[2]];
    signal tmp_6935[3] <== CMul()(challengeQ, tmp_6934);
    tmp_6936 <== [evals[51][0] + evals[52][0], evals[51][1] + evals[52][1], evals[51][2] + evals[52][2]];
    signal tmp_6937[3] <== [1 - evals[126][0], -evals[126][1], -evals[126][2]];
    signal tmp_6938[3] <== [1 - evals[143][0], -evals[143][1], -evals[143][2]];
    signal tmp_6939[3] <== CMul()(tmp_6937, tmp_6938);
    signal tmp_6940[3] <== CMul()(tmp_6939, evals[67]);
    signal tmp_6942[3] <== CMul()(evals[126], tmp_6938);
    signal tmp_6944[3] <== CMul()(tmp_6937, evals[143]);
    signal tmp_6945[3] <== [tmp_6942[0] + tmp_6944[0], tmp_6942[1] + tmp_6944[1], tmp_6942[2] + tmp_6944[2]];
    signal tmp_6946[3] <== CMul()(evals[126], evals[143]);
    signal tmp_6947[3] <== [tmp_6945[0] + tmp_6946[0], tmp_6945[1] + tmp_6946[1], tmp_6945[2] + tmp_6946[2]];
    signal tmp_6948[3] <== CMul()(tmp_6947, evals[71]);
    signal tmp_6949[3] <== [tmp_6940[0] + tmp_6948[0], tmp_6940[1] + tmp_6948[1], tmp_6940[2] + tmp_6948[2]];
    signal tmp_6950[3] <== CMul()(evals[52], tmp_6949);
    signal tmp_6951[3] <== CMul()(evals[51], evals[67]);
    signal tmp_6952[3] <== [tmp_6950[0] + tmp_6951[0], tmp_6950[1] + tmp_6951[1], tmp_6950[2] + tmp_6951[2]];
    signal tmp_6953[3] <== [2 * tmp_6952[0], 2 * tmp_6952[1], 2 * tmp_6952[2]];
    signal tmp_6954[3] <== CMul()(tmp_6939, evals[64]);
    signal tmp_6957[3] <== CMul()(tmp_6947, evals[68]);
    signal tmp_6958[3] <== [tmp_6954[0] + tmp_6957[0], tmp_6954[1] + tmp_6957[1], tmp_6954[2] + tmp_6957[2]];
    signal tmp_6959[3] <== CMul()(evals[52], tmp_6958);
    signal tmp_6960[3] <== CMul()(evals[51], evals[64]);
    signal tmp_6961[3] <== [tmp_6959[0] + tmp_6960[0], tmp_6959[1] + tmp_6960[1], tmp_6959[2] + tmp_6960[2]];
    signal tmp_6962[3] <== CMul()(tmp_6939, evals[65]);
    signal tmp_6965[3] <== CMul()(tmp_6947, evals[69]);
    signal tmp_6966[3] <== [tmp_6962[0] + tmp_6965[0], tmp_6962[1] + tmp_6965[1], tmp_6962[2] + tmp_6965[2]];
    signal tmp_6967[3] <== CMul()(evals[52], tmp_6966);
    signal tmp_6968[3] <== CMul()(evals[51], evals[65]);
    signal tmp_6969[3] <== [tmp_6967[0] + tmp_6968[0], tmp_6967[1] + tmp_6968[1], tmp_6967[2] + tmp_6968[2]];
    signal tmp_6970[3] <== [tmp_6961[0] + tmp_6969[0], tmp_6961[1] + tmp_6969[1], tmp_6961[2] + tmp_6969[2]];
    signal tmp_6971[3] <== [tmp_6953[0] + tmp_6970[0], tmp_6953[1] + tmp_6970[1], tmp_6953[2] + tmp_6970[2]];
    signal tmp_6972[3] <== [4 * tmp_6970[0], 4 * tmp_6970[1], 4 * tmp_6970[2]];
    signal tmp_6981[3] <== [2 * tmp_6969[0], 2 * tmp_6969[1], 2 * tmp_6969[2]];
    signal tmp_6982[3] <== CMul()(tmp_6939, evals[66]);
    signal tmp_6985[3] <== CMul()(tmp_6947, evals[70]);
    signal tmp_6986[3] <== [tmp_6982[0] + tmp_6985[0], tmp_6982[1] + tmp_6985[1], tmp_6982[2] + tmp_6985[2]];
    signal tmp_6987[3] <== CMul()(evals[52], tmp_6986);
    signal tmp_6988[3] <== CMul()(evals[51], evals[66]);
    signal tmp_6989[3] <== [tmp_6987[0] + tmp_6988[0], tmp_6987[1] + tmp_6988[1], tmp_6987[2] + tmp_6988[2]];
    signal tmp_6998[3] <== [tmp_6989[0] + tmp_6952[0], tmp_6989[1] + tmp_6952[1], tmp_6989[2] + tmp_6952[2]];
    signal tmp_6999[3] <== [tmp_6981[0] + tmp_6998[0], tmp_6981[1] + tmp_6998[1], tmp_6981[2] + tmp_6998[2]];
    signal tmp_7000[3] <== [tmp_6972[0] + tmp_6999[0], tmp_6972[1] + tmp_6999[1], tmp_6972[2] + tmp_6999[2]];
    signal tmp_7001[3] <== [tmp_6971[0] + tmp_7000[0], tmp_6971[1] + tmp_7000[1], tmp_6971[2] + tmp_7000[2]];
    signal tmp_7002[3] <== CMul()(tmp_6939, evals[71]);
    signal tmp_7003[3] <== CMul()(tmp_6942, evals[67]);
    signal tmp_7004[3] <== [tmp_7002[0] + tmp_7003[0], tmp_7002[1] + tmp_7003[1], tmp_7002[2] + tmp_7003[2]];
    signal tmp_7005[3] <== [tmp_6944[0] + tmp_6946[0], tmp_6944[1] + tmp_6946[1], tmp_6944[2] + tmp_6946[2]];
    signal tmp_7006[3] <== CMul()(tmp_7005, evals[75]);
    signal tmp_7007[3] <== [tmp_7004[0] + tmp_7006[0], tmp_7004[1] + tmp_7006[1], tmp_7004[2] + tmp_7006[2]];
    signal tmp_7008[3] <== CMul()(evals[52], tmp_7007);
    signal tmp_7009[3] <== CMul()(evals[51], evals[71]);
    signal tmp_7010[3] <== [tmp_7008[0] + tmp_7009[0], tmp_7008[1] + tmp_7009[1], tmp_7008[2] + tmp_7009[2]];
    signal tmp_7011[3] <== [2 * tmp_7010[0], 2 * tmp_7010[1], 2 * tmp_7010[2]];
    signal tmp_7012[3] <== CMul()(tmp_6939, evals[68]);
    signal tmp_7013[3] <== CMul()(tmp_6942, evals[64]);
    signal tmp_7014[3] <== [tmp_7012[0] + tmp_7013[0], tmp_7012[1] + tmp_7013[1], tmp_7012[2] + tmp_7013[2]];
    signal tmp_7016[3] <== CMul()(tmp_7005, evals[72]);
    signal tmp_7017[3] <== [tmp_7014[0] + tmp_7016[0], tmp_7014[1] + tmp_7016[1], tmp_7014[2] + tmp_7016[2]];
    signal tmp_7018[3] <== CMul()(evals[52], tmp_7017);
    signal tmp_7019[3] <== CMul()(evals[51], evals[68]);
    signal tmp_7020[3] <== [tmp_7018[0] + tmp_7019[0], tmp_7018[1] + tmp_7019[1], tmp_7018[2] + tmp_7019[2]];
    signal tmp_7021[3] <== CMul()(tmp_6939, evals[69]);
    signal tmp_7022[3] <== CMul()(tmp_6942, evals[65]);
    signal tmp_7023[3] <== [tmp_7021[0] + tmp_7022[0], tmp_7021[1] + tmp_7022[1], tmp_7021[2] + tmp_7022[2]];
    signal tmp_7025[3] <== CMul()(tmp_7005, evals[73]);
    signal tmp_7026[3] <== [tmp_7023[0] + tmp_7025[0], tmp_7023[1] + tmp_7025[1], tmp_7023[2] + tmp_7025[2]];
    signal tmp_7027[3] <== CMul()(evals[52], tmp_7026);
    signal tmp_7028[3] <== CMul()(evals[51], evals[69]);
    signal tmp_7029[3] <== [tmp_7027[0] + tmp_7028[0], tmp_7027[1] + tmp_7028[1], tmp_7027[2] + tmp_7028[2]];
    signal tmp_7030[3] <== [tmp_7020[0] + tmp_7029[0], tmp_7020[1] + tmp_7029[1], tmp_7020[2] + tmp_7029[2]];
    signal tmp_7031[3] <== [tmp_7011[0] + tmp_7030[0], tmp_7011[1] + tmp_7030[1], tmp_7011[2] + tmp_7030[2]];
    signal tmp_7032[3] <== [4 * tmp_7030[0], 4 * tmp_7030[1], 4 * tmp_7030[2]];
    signal tmp_7042[3] <== [2 * tmp_7029[0], 2 * tmp_7029[1], 2 * tmp_7029[2]];
    signal tmp_7043[3] <== CMul()(tmp_6939, evals[70]);
    signal tmp_7044[3] <== CMul()(tmp_6942, evals[66]);
    signal tmp_7045[3] <== [tmp_7043[0] + tmp_7044[0], tmp_7043[1] + tmp_7044[1], tmp_7043[2] + tmp_7044[2]];
    signal tmp_7047[3] <== CMul()(tmp_7005, evals[74]);
    signal tmp_7048[3] <== [tmp_7045[0] + tmp_7047[0], tmp_7045[1] + tmp_7047[1], tmp_7045[2] + tmp_7047[2]];
    signal tmp_7049[3] <== CMul()(evals[52], tmp_7048);
    signal tmp_7050[3] <== CMul()(evals[51], evals[70]);
    signal tmp_7051[3] <== [tmp_7049[0] + tmp_7050[0], tmp_7049[1] + tmp_7050[1], tmp_7049[2] + tmp_7050[2]];
    signal tmp_7061[3] <== [tmp_7051[0] + tmp_7010[0], tmp_7051[1] + tmp_7010[1], tmp_7051[2] + tmp_7010[2]];
    signal tmp_7062[3] <== [tmp_7042[0] + tmp_7061[0], tmp_7042[1] + tmp_7061[1], tmp_7042[2] + tmp_7061[2]];
    signal tmp_7063[3] <== [tmp_7032[0] + tmp_7062[0], tmp_7032[1] + tmp_7062[1], tmp_7032[2] + tmp_7062[2]];
    signal tmp_7064[3] <== [tmp_7031[0] + tmp_7063[0], tmp_7031[1] + tmp_7063[1], tmp_7031[2] + tmp_7063[2]];
    signal tmp_7065[3] <== [tmp_7001[0] + tmp_7064[0], tmp_7001[1] + tmp_7064[1], tmp_7001[2] + tmp_7064[2]];
    signal tmp_7066[3] <== [tmp_6939[0] + tmp_6942[0], tmp_6939[1] + tmp_6942[1], tmp_6939[2] + tmp_6942[2]];
    signal tmp_7067[3] <== CMul()(tmp_7066, evals[75]);
    signal tmp_7068[3] <== CMul()(tmp_6944, evals[67]);
    signal tmp_7069[3] <== [tmp_7067[0] + tmp_7068[0], tmp_7067[1] + tmp_7068[1], tmp_7067[2] + tmp_7068[2]];
    signal tmp_7070[3] <== CMul()(tmp_6946, evals[79]);
    signal tmp_7071[3] <== [tmp_7069[0] + tmp_7070[0], tmp_7069[1] + tmp_7070[1], tmp_7069[2] + tmp_7070[2]];
    signal tmp_7072[3] <== CMul()(evals[52], tmp_7071);
    signal tmp_7073[3] <== CMul()(evals[51], evals[75]);
    signal tmp_7074[3] <== [tmp_7072[0] + tmp_7073[0], tmp_7072[1] + tmp_7073[1], tmp_7072[2] + tmp_7073[2]];
    signal tmp_7075[3] <== [2 * tmp_7074[0], 2 * tmp_7074[1], 2 * tmp_7074[2]];
    signal tmp_7077[3] <== CMul()(tmp_7066, evals[72]);
    signal tmp_7078[3] <== CMul()(tmp_6944, evals[64]);
    signal tmp_7079[3] <== [tmp_7077[0] + tmp_7078[0], tmp_7077[1] + tmp_7078[1], tmp_7077[2] + tmp_7078[2]];
    signal tmp_7080[3] <== CMul()(tmp_6946, evals[76]);
    signal tmp_7081[3] <== [tmp_7079[0] + tmp_7080[0], tmp_7079[1] + tmp_7080[1], tmp_7079[2] + tmp_7080[2]];
    signal tmp_7082[3] <== CMul()(evals[52], tmp_7081);
    signal tmp_7083[3] <== CMul()(evals[51], evals[72]);
    signal tmp_7084[3] <== [tmp_7082[0] + tmp_7083[0], tmp_7082[1] + tmp_7083[1], tmp_7082[2] + tmp_7083[2]];
    signal tmp_7086[3] <== CMul()(tmp_7066, evals[73]);
    signal tmp_7087[3] <== CMul()(tmp_6944, evals[65]);
    signal tmp_7088[3] <== [tmp_7086[0] + tmp_7087[0], tmp_7086[1] + tmp_7087[1], tmp_7086[2] + tmp_7087[2]];
    signal tmp_7089[3] <== CMul()(tmp_6946, evals[77]);
    signal tmp_7090[3] <== [tmp_7088[0] + tmp_7089[0], tmp_7088[1] + tmp_7089[1], tmp_7088[2] + tmp_7089[2]];
    signal tmp_7091[3] <== CMul()(evals[52], tmp_7090);
    signal tmp_7092[3] <== CMul()(evals[51], evals[73]);
    signal tmp_7093[3] <== [tmp_7091[0] + tmp_7092[0], tmp_7091[1] + tmp_7092[1], tmp_7091[2] + tmp_7092[2]];
    signal tmp_7094[3] <== [tmp_7084[0] + tmp_7093[0], tmp_7084[1] + tmp_7093[1], tmp_7084[2] + tmp_7093[2]];
    signal tmp_7095[3] <== [tmp_7075[0] + tmp_7094[0], tmp_7075[1] + tmp_7094[1], tmp_7075[2] + tmp_7094[2]];
    signal tmp_7096[3] <== [4 * tmp_7094[0], 4 * tmp_7094[1], 4 * tmp_7094[2]];
    signal tmp_7106[3] <== [2 * tmp_7093[0], 2 * tmp_7093[1], 2 * tmp_7093[2]];
    signal tmp_7108[3] <== CMul()(tmp_7066, evals[74]);
    signal tmp_7109[3] <== CMul()(tmp_6944, evals[66]);
    signal tmp_7110[3] <== [tmp_7108[0] + tmp_7109[0], tmp_7108[1] + tmp_7109[1], tmp_7108[2] + tmp_7109[2]];
    signal tmp_7111[3] <== CMul()(tmp_6946, evals[78]);
    signal tmp_7112[3] <== [tmp_7110[0] + tmp_7111[0], tmp_7110[1] + tmp_7111[1], tmp_7110[2] + tmp_7111[2]];
    signal tmp_7113[3] <== CMul()(evals[52], tmp_7112);
    signal tmp_7114[3] <== CMul()(evals[51], evals[74]);
    signal tmp_7115[3] <== [tmp_7113[0] + tmp_7114[0], tmp_7113[1] + tmp_7114[1], tmp_7113[2] + tmp_7114[2]];
    signal tmp_7125[3] <== [tmp_7115[0] + tmp_7074[0], tmp_7115[1] + tmp_7074[1], tmp_7115[2] + tmp_7074[2]];
    signal tmp_7126[3] <== [tmp_7106[0] + tmp_7125[0], tmp_7106[1] + tmp_7125[1], tmp_7106[2] + tmp_7125[2]];
    signal tmp_7127[3] <== [tmp_7096[0] + tmp_7126[0], tmp_7096[1] + tmp_7126[1], tmp_7096[2] + tmp_7126[2]];
    signal tmp_7128[3] <== [tmp_7095[0] + tmp_7127[0], tmp_7095[1] + tmp_7127[1], tmp_7095[2] + tmp_7127[2]];
    signal tmp_7129[3] <== [tmp_7065[0] + tmp_7128[0], tmp_7065[1] + tmp_7128[1], tmp_7065[2] + tmp_7128[2]];
    signal tmp_7131[3] <== [tmp_7066[0] + tmp_6944[0], tmp_7066[1] + tmp_6944[1], tmp_7066[2] + tmp_6944[2]];
    signal tmp_7132[3] <== CMul()(tmp_7131, evals[79]);
    signal tmp_7133[3] <== CMul()(tmp_6946, evals[67]);
    signal tmp_7134[3] <== [tmp_7132[0] + tmp_7133[0], tmp_7132[1] + tmp_7133[1], tmp_7132[2] + tmp_7133[2]];
    signal tmp_7135[3] <== CMul()(evals[52], tmp_7134);
    signal tmp_7136[3] <== CMul()(evals[51], evals[79]);
    signal tmp_7137[3] <== [tmp_7135[0] + tmp_7136[0], tmp_7135[1] + tmp_7136[1], tmp_7135[2] + tmp_7136[2]];
    signal tmp_7138[3] <== [2 * tmp_7137[0], 2 * tmp_7137[1], 2 * tmp_7137[2]];
    signal tmp_7141[3] <== CMul()(tmp_7131, evals[76]);
    signal tmp_7142[3] <== CMul()(tmp_6946, evals[64]);
    signal tmp_7143[3] <== [tmp_7141[0] + tmp_7142[0], tmp_7141[1] + tmp_7142[1], tmp_7141[2] + tmp_7142[2]];
    signal tmp_7144[3] <== CMul()(evals[52], tmp_7143);
    signal tmp_7145[3] <== CMul()(evals[51], evals[76]);
    signal tmp_7146[3] <== [tmp_7144[0] + tmp_7145[0], tmp_7144[1] + tmp_7145[1], tmp_7144[2] + tmp_7145[2]];
    signal tmp_7149[3] <== CMul()(tmp_7131, evals[77]);
    signal tmp_7150[3] <== CMul()(tmp_6946, evals[65]);
    signal tmp_7151[3] <== [tmp_7149[0] + tmp_7150[0], tmp_7149[1] + tmp_7150[1], tmp_7149[2] + tmp_7150[2]];
    signal tmp_7152[3] <== CMul()(evals[52], tmp_7151);
    signal tmp_7153[3] <== CMul()(evals[51], evals[77]);
    signal tmp_7154[3] <== [tmp_7152[0] + tmp_7153[0], tmp_7152[1] + tmp_7153[1], tmp_7152[2] + tmp_7153[2]];
    signal tmp_7155[3] <== [tmp_7146[0] + tmp_7154[0], tmp_7146[1] + tmp_7154[1], tmp_7146[2] + tmp_7154[2]];
    signal tmp_7156[3] <== [tmp_7138[0] + tmp_7155[0], tmp_7138[1] + tmp_7155[1], tmp_7138[2] + tmp_7155[2]];
    signal tmp_7157[3] <== [4 * tmp_7155[0], 4 * tmp_7155[1], 4 * tmp_7155[2]];
    signal tmp_7166[3] <== [2 * tmp_7154[0], 2 * tmp_7154[1], 2 * tmp_7154[2]];
    signal tmp_7169[3] <== CMul()(tmp_7131, evals[78]);
    signal tmp_7170[3] <== CMul()(tmp_6946, evals[66]);
    signal tmp_7171[3] <== [tmp_7169[0] + tmp_7170[0], tmp_7169[1] + tmp_7170[1], tmp_7169[2] + tmp_7170[2]];
    signal tmp_7172[3] <== CMul()(evals[52], tmp_7171);
    signal tmp_7173[3] <== CMul()(evals[51], evals[78]);
    signal tmp_7174[3] <== [tmp_7172[0] + tmp_7173[0], tmp_7172[1] + tmp_7173[1], tmp_7172[2] + tmp_7173[2]];
    signal tmp_7183[3] <== [tmp_7174[0] + tmp_7137[0], tmp_7174[1] + tmp_7137[1], tmp_7174[2] + tmp_7137[2]];
    signal tmp_7184[3] <== [tmp_7166[0] + tmp_7183[0], tmp_7166[1] + tmp_7183[1], tmp_7166[2] + tmp_7183[2]];
    signal tmp_7185[3] <== [tmp_7157[0] + tmp_7184[0], tmp_7157[1] + tmp_7184[1], tmp_7157[2] + tmp_7184[2]];
    signal tmp_7186[3] <== [tmp_7156[0] + tmp_7185[0], tmp_7156[1] + tmp_7185[1], tmp_7156[2] + tmp_7185[2]];
    signal tmp_7187[3] <== [tmp_7129[0] + tmp_7186[0], tmp_7129[1] + tmp_7186[1], tmp_7129[2] + tmp_7186[2]];
    signal tmp_7188[3] <== [tmp_7001[0] + tmp_7187[0], tmp_7001[1] + tmp_7187[1], tmp_7001[2] + tmp_7187[2]];
    signal tmp_7189[3] <== [evals[80][0] - tmp_7188[0], evals[80][1] - tmp_7188[1], evals[80][2] - tmp_7188[2]];
    signal tmp_7190[3] <== CMul()(tmp_6936, tmp_7189);
    signal tmp_7191[3] <== [tmp_6935[0] + tmp_7190[0], tmp_6935[1] + tmp_7190[1], tmp_6935[2] + tmp_7190[2]];
    signal tmp_7192[3] <== CMul()(challengeQ, tmp_7191);
    signal tmp_7194[3] <== [tmp_7000[0] + tmp_7063[0], tmp_7000[1] + tmp_7063[1], tmp_7000[2] + tmp_7063[2]];
    signal tmp_7195[3] <== [tmp_7194[0] + tmp_7127[0], tmp_7194[1] + tmp_7127[1], tmp_7194[2] + tmp_7127[2]];
    signal tmp_7196[3] <== [tmp_7195[0] + tmp_7185[0], tmp_7195[1] + tmp_7185[1], tmp_7195[2] + tmp_7185[2]];
    signal tmp_7197[3] <== [tmp_7000[0] + tmp_7196[0], tmp_7000[1] + tmp_7196[1], tmp_7000[2] + tmp_7196[2]];
    signal tmp_7198[3] <== [evals[81][0] - tmp_7197[0], evals[81][1] - tmp_7197[1], evals[81][2] - tmp_7197[2]];
    signal tmp_7199[3] <== CMul()(tmp_6936, tmp_7198);
    signal tmp_7200[3] <== [tmp_7192[0] + tmp_7199[0], tmp_7192[1] + tmp_7199[1], tmp_7192[2] + tmp_7199[2]];
    signal tmp_7201[3] <== CMul()(challengeQ, tmp_7200);
    signal tmp_7203[3] <== [4 * tmp_6998[0], 4 * tmp_6998[1], 4 * tmp_6998[2]];
    signal tmp_7204[3] <== [tmp_7203[0] + tmp_6971[0], tmp_7203[1] + tmp_6971[1], tmp_7203[2] + tmp_6971[2]];
    signal tmp_7205[3] <== [tmp_6999[0] + tmp_7204[0], tmp_6999[1] + tmp_7204[1], tmp_6999[2] + tmp_7204[2]];
    signal tmp_7206[3] <== [4 * tmp_7061[0], 4 * tmp_7061[1], 4 * tmp_7061[2]];
    signal tmp_7207[3] <== [tmp_7206[0] + tmp_7031[0], tmp_7206[1] + tmp_7031[1], tmp_7206[2] + tmp_7031[2]];
    signal tmp_7208[3] <== [tmp_7062[0] + tmp_7207[0], tmp_7062[1] + tmp_7207[1], tmp_7062[2] + tmp_7207[2]];
    signal tmp_7209[3] <== [tmp_7205[0] + tmp_7208[0], tmp_7205[1] + tmp_7208[1], tmp_7205[2] + tmp_7208[2]];
    signal tmp_7210[3] <== [4 * tmp_7125[0], 4 * tmp_7125[1], 4 * tmp_7125[2]];
    signal tmp_7211[3] <== [tmp_7210[0] + tmp_7095[0], tmp_7210[1] + tmp_7095[1], tmp_7210[2] + tmp_7095[2]];
    signal tmp_7212[3] <== [tmp_7126[0] + tmp_7211[0], tmp_7126[1] + tmp_7211[1], tmp_7126[2] + tmp_7211[2]];
    signal tmp_7213[3] <== [tmp_7209[0] + tmp_7212[0], tmp_7209[1] + tmp_7212[1], tmp_7209[2] + tmp_7212[2]];
    signal tmp_7214[3] <== [4 * tmp_7183[0], 4 * tmp_7183[1], 4 * tmp_7183[2]];
    signal tmp_7215[3] <== [tmp_7214[0] + tmp_7156[0], tmp_7214[1] + tmp_7156[1], tmp_7214[2] + tmp_7156[2]];
    signal tmp_7216[3] <== [tmp_7184[0] + tmp_7215[0], tmp_7184[1] + tmp_7215[1], tmp_7184[2] + tmp_7215[2]];
    signal tmp_7217[3] <== [tmp_7213[0] + tmp_7216[0], tmp_7213[1] + tmp_7216[1], tmp_7213[2] + tmp_7216[2]];
    signal tmp_7218[3] <== [tmp_7205[0] + tmp_7217[0], tmp_7205[1] + tmp_7217[1], tmp_7205[2] + tmp_7217[2]];
    signal tmp_7219[3] <== [evals[82][0] - tmp_7218[0], evals[82][1] - tmp_7218[1], evals[82][2] - tmp_7218[2]];
    signal tmp_7220[3] <== CMul()(tmp_6936, tmp_7219);
    signal tmp_7221[3] <== [tmp_7201[0] + tmp_7220[0], tmp_7201[1] + tmp_7220[1], tmp_7201[2] + tmp_7220[2]];
    signal tmp_7222[3] <== CMul()(challengeQ, tmp_7221);
    signal tmp_7224[3] <== [tmp_7204[0] + tmp_7207[0], tmp_7204[1] + tmp_7207[1], tmp_7204[2] + tmp_7207[2]];
    signal tmp_7225[3] <== [tmp_7224[0] + tmp_7211[0], tmp_7224[1] + tmp_7211[1], tmp_7224[2] + tmp_7211[2]];
    signal tmp_7226[3] <== [tmp_7225[0] + tmp_7215[0], tmp_7225[1] + tmp_7215[1], tmp_7225[2] + tmp_7215[2]];
    signal tmp_7227[3] <== [tmp_7204[0] + tmp_7226[0], tmp_7204[1] + tmp_7226[1], tmp_7204[2] + tmp_7226[2]];
    signal tmp_7228[3] <== [evals[83][0] - tmp_7227[0], evals[83][1] - tmp_7227[1], evals[83][2] - tmp_7227[2]];
    signal tmp_7229[3] <== CMul()(tmp_6936, tmp_7228);
    signal tmp_7230[3] <== [tmp_7222[0] + tmp_7229[0], tmp_7222[1] + tmp_7229[1], tmp_7222[2] + tmp_7229[2]];
    signal tmp_7231[3] <== CMul()(challengeQ, tmp_7230);
    signal tmp_7233[3] <== [tmp_7064[0] + tmp_7187[0], tmp_7064[1] + tmp_7187[1], tmp_7064[2] + tmp_7187[2]];
    signal tmp_7234[3] <== [evals[84][0] - tmp_7233[0], evals[84][1] - tmp_7233[1], evals[84][2] - tmp_7233[2]];
    signal tmp_7235[3] <== CMul()(tmp_6936, tmp_7234);
    signal tmp_7236[3] <== [tmp_7231[0] + tmp_7235[0], tmp_7231[1] + tmp_7235[1], tmp_7231[2] + tmp_7235[2]];
    signal tmp_7237[3] <== CMul()(challengeQ, tmp_7236);
    signal tmp_7239[3] <== [tmp_7063[0] + tmp_7196[0], tmp_7063[1] + tmp_7196[1], tmp_7063[2] + tmp_7196[2]];
    signal tmp_7240[3] <== [evals[85][0] - tmp_7239[0], evals[85][1] - tmp_7239[1], evals[85][2] - tmp_7239[2]];
    signal tmp_7241[3] <== CMul()(tmp_6936, tmp_7240);
    signal tmp_7242[3] <== [tmp_7237[0] + tmp_7241[0], tmp_7237[1] + tmp_7241[1], tmp_7237[2] + tmp_7241[2]];
    signal tmp_7243[3] <== CMul()(challengeQ, tmp_7242);
    signal tmp_7245[3] <== [tmp_7208[0] + tmp_7217[0], tmp_7208[1] + tmp_7217[1], tmp_7208[2] + tmp_7217[2]];
    signal tmp_7246[3] <== [evals[86][0] - tmp_7245[0], evals[86][1] - tmp_7245[1], evals[86][2] - tmp_7245[2]];
    signal tmp_7247[3] <== CMul()(tmp_6936, tmp_7246);
    signal tmp_7248[3] <== [tmp_7243[0] + tmp_7247[0], tmp_7243[1] + tmp_7247[1], tmp_7243[2] + tmp_7247[2]];
    signal tmp_7249[3] <== CMul()(challengeQ, tmp_7248);
    signal tmp_7251[3] <== [tmp_7207[0] + tmp_7226[0], tmp_7207[1] + tmp_7226[1], tmp_7207[2] + tmp_7226[2]];
    signal tmp_7252[3] <== [evals[87][0] - tmp_7251[0], evals[87][1] - tmp_7251[1], evals[87][2] - tmp_7251[2]];
    signal tmp_7253[3] <== CMul()(tmp_6936, tmp_7252);
    signal tmp_7254[3] <== [tmp_7249[0] + tmp_7253[0], tmp_7249[1] + tmp_7253[1], tmp_7249[2] + tmp_7253[2]];
    signal tmp_7255[3] <== CMul()(challengeQ, tmp_7254);
    signal tmp_7257[3] <== [tmp_7128[0] + tmp_7187[0], tmp_7128[1] + tmp_7187[1], tmp_7128[2] + tmp_7187[2]];
    signal tmp_7258[3] <== [evals[88][0] - tmp_7257[0], evals[88][1] - tmp_7257[1], evals[88][2] - tmp_7257[2]];
    signal tmp_7259[3] <== CMul()(tmp_6936, tmp_7258);
    signal tmp_7260[3] <== [tmp_7255[0] + tmp_7259[0], tmp_7255[1] + tmp_7259[1], tmp_7255[2] + tmp_7259[2]];
    signal tmp_7261[3] <== CMul()(challengeQ, tmp_7260);
    signal tmp_7263[3] <== [tmp_7127[0] + tmp_7196[0], tmp_7127[1] + tmp_7196[1], tmp_7127[2] + tmp_7196[2]];
    signal tmp_7264[3] <== [evals[89][0] - tmp_7263[0], evals[89][1] - tmp_7263[1], evals[89][2] - tmp_7263[2]];
    signal tmp_7265[3] <== CMul()(tmp_6936, tmp_7264);
    signal tmp_7266[3] <== [tmp_7261[0] + tmp_7265[0], tmp_7261[1] + tmp_7265[1], tmp_7261[2] + tmp_7265[2]];
    signal tmp_7267[3] <== CMul()(challengeQ, tmp_7266);
    signal tmp_7269[3] <== [tmp_7212[0] + tmp_7217[0], tmp_7212[1] + tmp_7217[1], tmp_7212[2] + tmp_7217[2]];
    signal tmp_7270[3] <== [evals[90][0] - tmp_7269[0], evals[90][1] - tmp_7269[1], evals[90][2] - tmp_7269[2]];
    signal tmp_7271[3] <== CMul()(tmp_6936, tmp_7270);
    signal tmp_7272[3] <== [tmp_7267[0] + tmp_7271[0], tmp_7267[1] + tmp_7271[1], tmp_7267[2] + tmp_7271[2]];
    signal tmp_7273[3] <== CMul()(challengeQ, tmp_7272);
    signal tmp_7275[3] <== [tmp_7211[0] + tmp_7226[0], tmp_7211[1] + tmp_7226[1], tmp_7211[2] + tmp_7226[2]];
    signal tmp_7276[3] <== [evals[91][0] - tmp_7275[0], evals[91][1] - tmp_7275[1], evals[91][2] - tmp_7275[2]];
    signal tmp_7277[3] <== CMul()(tmp_6936, tmp_7276);
    signal tmp_7278[3] <== [tmp_7273[0] + tmp_7277[0], tmp_7273[1] + tmp_7277[1], tmp_7273[2] + tmp_7277[2]];
    signal tmp_7279[3] <== CMul()(challengeQ, tmp_7278);
    signal tmp_7281[3] <== [tmp_7186[0] + tmp_7187[0], tmp_7186[1] + tmp_7187[1], tmp_7186[2] + tmp_7187[2]];
    signal tmp_7282[3] <== [evals[92][0] - tmp_7281[0], evals[92][1] - tmp_7281[1], evals[92][2] - tmp_7281[2]];
    signal tmp_7283[3] <== CMul()(tmp_6936, tmp_7282);
    signal tmp_7284[3] <== [tmp_7279[0] + tmp_7283[0], tmp_7279[1] + tmp_7283[1], tmp_7279[2] + tmp_7283[2]];
    signal tmp_7285[3] <== CMul()(challengeQ, tmp_7284);
    signal tmp_7287[3] <== [tmp_7185[0] + tmp_7196[0], tmp_7185[1] + tmp_7196[1], tmp_7185[2] + tmp_7196[2]];
    signal tmp_7288[3] <== [evals[93][0] - tmp_7287[0], evals[93][1] - tmp_7287[1], evals[93][2] - tmp_7287[2]];
    signal tmp_7289[3] <== CMul()(tmp_6936, tmp_7288);
    signal tmp_7290[3] <== [tmp_7285[0] + tmp_7289[0], tmp_7285[1] + tmp_7289[1], tmp_7285[2] + tmp_7289[2]];
    signal tmp_7291[3] <== CMul()(challengeQ, tmp_7290);
    signal tmp_7293[3] <== [tmp_7216[0] + tmp_7217[0], tmp_7216[1] + tmp_7217[1], tmp_7216[2] + tmp_7217[2]];
    signal tmp_7294[3] <== [evals[94][0] - tmp_7293[0], evals[94][1] - tmp_7293[1], evals[94][2] - tmp_7293[2]];
    signal tmp_7295[3] <== CMul()(tmp_6936, tmp_7294);
    signal tmp_7296[3] <== [tmp_7291[0] + tmp_7295[0], tmp_7291[1] + tmp_7295[1], tmp_7291[2] + tmp_7295[2]];
    signal tmp_7297[3] <== CMul()(challengeQ, tmp_7296);
    signal tmp_7299[3] <== [tmp_7215[0] + tmp_7226[0], tmp_7215[1] + tmp_7226[1], tmp_7215[2] + tmp_7226[2]];
    signal tmp_7300[3] <== [evals[95][0] - tmp_7299[0], evals[95][1] - tmp_7299[1], evals[95][2] - tmp_7299[2]];
    signal tmp_7301[3] <== CMul()(tmp_6936, tmp_7300);
    signal tmp_7302[3] <== [tmp_7297[0] + tmp_7301[0], tmp_7297[1] + tmp_7301[1], tmp_7297[2] + tmp_7301[2]];
    tmp_7303 <== CMul()(challengeQ, tmp_7302);
    signal tmp_7305[3] <== [tmp_6936[0] + evals[123][0], tmp_6936[1] + evals[123][1], tmp_6936[2] + evals[123][2]];
    signal tmp_7306[3] <== [tmp_7305[0] + evals[124][0], tmp_7305[1] + evals[124][1], tmp_7305[2] + evals[124][2]];
    tmp_7307 <== [tmp_7306[0] + evals[54][0], tmp_7306[1] + evals[54][1], tmp_7306[2] + evals[54][2]];
    signal tmp_7309[3] <== [tmp_6936[0] * 5625716564419252202, tmp_6936[1] * 5625716564419252202, tmp_6936[2] * 5625716564419252202];
    signal tmp_7310[3] <== [evals[123][0] * 11810209965918630788, evals[123][1] * 11810209965918630788, evals[123][2] * 11810209965918630788];
    signal tmp_7311[3] <== [tmp_7309[0] + tmp_7310[0], tmp_7309[1] + tmp_7310[1], tmp_7309[2] + tmp_7310[2]];
    signal tmp_7312[3] <== [evals[124][0] * 2197120128337974909, evals[124][1] * 2197120128337974909, evals[124][2] * 2197120128337974909];
    signal tmp_7313[3] <== [tmp_7311[0] + tmp_7312[0], tmp_7311[1] + tmp_7312[1], tmp_7311[2] + tmp_7312[2]];
    signal tmp_7314[3] <== [evals[54][0] * 16343835012250148340, evals[54][1] * 16343835012250148340, evals[54][2] * 16343835012250148340];
    signal tmp_7315[3] <== [tmp_7313[0] + tmp_7314[0], tmp_7313[1] + tmp_7314[1], tmp_7313[2] + tmp_7314[2]];
    signal tmp_7316[3] <== [evals[83][0] + tmp_7315[0], evals[83][1] + tmp_7315[1], evals[83][2] + tmp_7315[2]];
    signal tmp_7318[3] <== CMul()(tmp_7316, tmp_7316);
    signal tmp_7319[3] <== CMul()(tmp_7318, tmp_7318);
    signal tmp_7320[3] <== CMul()(tmp_7319, tmp_7318);
    tmp_7322 <== CMul()(tmp_7320, tmp_7316);
    tmp_7323 <== [2 * tmp_7322[0], 2 * tmp_7322[1], 2 * tmp_7322[2]];
    signal tmp_7325[3] <== [tmp_6936[0] * 1579613653969377219, tmp_6936[1] * 1579613653969377219, tmp_6936[2] * 1579613653969377219];
    signal tmp_7326[3] <== [evals[123][0] * 12437259172286625042, evals[123][1] * 12437259172286625042, evals[123][2] * 12437259172286625042];
    signal tmp_7327[3] <== [tmp_7325[0] + tmp_7326[0], tmp_7325[1] + tmp_7326[1], tmp_7325[2] + tmp_7326[2]];
    signal tmp_7328[3] <== [evals[124][0] * 15766898731627159621, evals[124][1] * 15766898731627159621, evals[124][2] * 15766898731627159621];
    signal tmp_7329[3] <== [tmp_7327[0] + tmp_7328[0], tmp_7327[1] + tmp_7328[1], tmp_7327[2] + tmp_7328[2]];
    signal tmp_7330[3] <== [evals[54][0] * 9738621014989498753, evals[54][1] * 9738621014989498753, evals[54][2] * 9738621014989498753];
    signal tmp_7331[3] <== [tmp_7329[0] + tmp_7330[0], tmp_7329[1] + tmp_7330[1], tmp_7329[2] + tmp_7330[2]];
    tmp_7332 <== [evals[80][0] + tmp_7331[0], evals[80][1] + tmp_7331[1], evals[80][2] + tmp_7331[2]];
    tmp_7334 <== CMul()(tmp_7332, tmp_7332);
    tmp_7335 <== CMul()(tmp_7334, tmp_7334);
}

template VerifyEvaluationsChunks2_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_6936[3];
    signal input tmp_7303[3];
    signal input tmp_7307[3];
    signal input tmp_7322[3];
    signal input tmp_7323[3];
    signal input tmp_7332[3];
    signal input tmp_7334[3];
    signal input tmp_7335[3];

    signal output tmp_7869[3];
    signal output tmp_7927[3];
    signal output tmp_7951[3];
    signal output tmp_7969[3];
    signal output tmp_7986[3];
    signal output tmp_8009[3];
    signal output tmp_8033[3];
    signal output tmp_8051[3];
    signal output tmp_8068[3];
    signal output tmp_8091[3];
    signal output tmp_8558[3];
    signal output tmp_8563[3];
    signal output tmp_8566[3];
    signal output tmp_8569[3];
    signal output tmp_8572[3];
    signal output tmp_8575[3];
    signal output tmp_8578[3];
    signal output tmp_8581[3];
    signal output tmp_8584[3];
    signal output tmp_8587[3];
    signal output tmp_8590[3];
    signal output tmp_8593[3];
    signal output tmp_8596[3];
    signal output tmp_8599[3];
    signal output tmp_8602[3];
    signal output tmp_8605[3];
    signal output tmp_8606[3];
    signal output tmp_8608[3];
    signal tmp_7336[3] <== CMul()(tmp_7335, tmp_7334);
    signal tmp_7338[3] <== CMul()(tmp_7336, tmp_7332);
    signal tmp_7340[3] <== [tmp_6936[0] * 15509510893087893340, tmp_6936[1] * 15509510893087893340, tmp_6936[2] * 15509510893087893340];
    signal tmp_7341[3] <== [evals[123][0] * 3143835545170851304, evals[123][1] * 3143835545170851304, evals[123][2] * 3143835545170851304];
    signal tmp_7342[3] <== [tmp_7340[0] + tmp_7341[0], tmp_7340[1] + tmp_7341[1], tmp_7340[2] + tmp_7341[2]];
    signal tmp_7343[3] <== [evals[124][0] * 6714370142978148338, evals[124][1] * 6714370142978148338, evals[124][2] * 6714370142978148338];
    signal tmp_7344[3] <== [tmp_7342[0] + tmp_7343[0], tmp_7342[1] + tmp_7343[1], tmp_7342[2] + tmp_7343[2]];
    signal tmp_7345[3] <== [evals[54][0] * 237412858668409460, evals[54][1] * 237412858668409460, evals[54][2] * 237412858668409460];
    signal tmp_7346[3] <== [tmp_7344[0] + tmp_7345[0], tmp_7344[1] + tmp_7345[1], tmp_7344[2] + tmp_7345[2]];
    signal tmp_7347[3] <== [evals[81][0] + tmp_7346[0], evals[81][1] + tmp_7346[1], evals[81][2] + tmp_7346[2]];
    signal tmp_7349[3] <== CMul()(tmp_7347, tmp_7347);
    signal tmp_7350[3] <== CMul()(tmp_7349, tmp_7349);
    signal tmp_7351[3] <== CMul()(tmp_7350, tmp_7349);
    signal tmp_7353[3] <== CMul()(tmp_7351, tmp_7347);
    signal tmp_7354[3] <== [tmp_7338[0] + tmp_7353[0], tmp_7338[1] + tmp_7353[1], tmp_7338[2] + tmp_7353[2]];
    signal tmp_7355[3] <== [tmp_7323[0] + tmp_7354[0], tmp_7323[1] + tmp_7354[1], tmp_7323[2] + tmp_7354[2]];
    signal tmp_7356[3] <== [4 * tmp_7354[0], 4 * tmp_7354[1], 4 * tmp_7354[2]];
    signal tmp_7359[3] <== [2 * tmp_7353[0], 2 * tmp_7353[1], 2 * tmp_7353[2]];
    signal tmp_7361[3] <== [tmp_6936[0] * 10090715174060125222, tmp_6936[1] * 10090715174060125222, tmp_6936[2] * 10090715174060125222];
    signal tmp_7362[3] <== [evals[123][0] * 14361575850007152549, evals[123][1] * 14361575850007152549, evals[123][2] * 14361575850007152549];
    signal tmp_7363[3] <== [tmp_7361[0] + tmp_7362[0], tmp_7361[1] + tmp_7362[1], tmp_7361[2] + tmp_7362[2]];
    signal tmp_7364[3] <== [evals[124][0] * 5906664861467821209, evals[124][1] * 5906664861467821209, evals[124][2] * 5906664861467821209];
    signal tmp_7365[3] <== [tmp_7363[0] + tmp_7364[0], tmp_7363[1] + tmp_7364[1], tmp_7363[2] + tmp_7364[2]];
    signal tmp_7366[3] <== [evals[54][0] * 6405904454922718574, evals[54][1] * 6405904454922718574, evals[54][2] * 6405904454922718574];
    signal tmp_7367[3] <== [tmp_7365[0] + tmp_7366[0], tmp_7365[1] + tmp_7366[1], tmp_7365[2] + tmp_7366[2]];
    signal tmp_7368[3] <== [evals[82][0] + tmp_7367[0], evals[82][1] + tmp_7367[1], evals[82][2] + tmp_7367[2]];
    signal tmp_7370[3] <== CMul()(tmp_7368, tmp_7368);
    signal tmp_7371[3] <== CMul()(tmp_7370, tmp_7370);
    signal tmp_7372[3] <== CMul()(tmp_7371, tmp_7370);
    signal tmp_7374[3] <== CMul()(tmp_7372, tmp_7368);
    signal tmp_7377[3] <== [tmp_7374[0] + tmp_7322[0], tmp_7374[1] + tmp_7322[1], tmp_7374[2] + tmp_7322[2]];
    signal tmp_7378[3] <== [tmp_7359[0] + tmp_7377[0], tmp_7359[1] + tmp_7377[1], tmp_7359[2] + tmp_7377[2]];
    signal tmp_7379[3] <== [tmp_7356[0] + tmp_7378[0], tmp_7356[1] + tmp_7378[1], tmp_7356[2] + tmp_7378[2]];
    signal tmp_7380[3] <== [tmp_7355[0] + tmp_7379[0], tmp_7355[1] + tmp_7379[1], tmp_7355[2] + tmp_7379[2]];
    signal tmp_7382[3] <== [tmp_6936[0] * 2027625550790675754, tmp_6936[1] * 2027625550790675754, tmp_6936[2] * 2027625550790675754];
    signal tmp_7383[3] <== [evals[123][0] * 1778669202480014256, evals[123][1] * 1778669202480014256, evals[123][2] * 1778669202480014256];
    signal tmp_7384[3] <== [tmp_7382[0] + tmp_7383[0], tmp_7382[1] + tmp_7383[1], tmp_7382[2] + tmp_7383[2]];
    signal tmp_7385[3] <== [evals[124][0] * 12399217337488717232, evals[124][1] * 12399217337488717232, evals[124][2] * 12399217337488717232];
    signal tmp_7386[3] <== [tmp_7384[0] + tmp_7385[0], tmp_7384[1] + tmp_7385[1], tmp_7384[2] + tmp_7385[2]];
    signal tmp_7387[3] <== [evals[54][0] * 16030622638137836774, evals[54][1] * 16030622638137836774, evals[54][2] * 16030622638137836774];
    signal tmp_7388[3] <== [tmp_7386[0] + tmp_7387[0], tmp_7386[1] + tmp_7387[1], tmp_7386[2] + tmp_7387[2]];
    signal tmp_7389[3] <== [evals[87][0] + tmp_7388[0], evals[87][1] + tmp_7388[1], evals[87][2] + tmp_7388[2]];
    signal tmp_7391[3] <== CMul()(tmp_7389, tmp_7389);
    signal tmp_7392[3] <== CMul()(tmp_7391, tmp_7391);
    signal tmp_7393[3] <== CMul()(tmp_7392, tmp_7391);
    signal tmp_7395[3] <== CMul()(tmp_7393, tmp_7389);
    signal tmp_7396[3] <== [2 * tmp_7395[0], 2 * tmp_7395[1], 2 * tmp_7395[2]];
    signal tmp_7398[3] <== [tmp_6936[0] * 3006005019077469174, tmp_6936[1] * 3006005019077469174, tmp_6936[2] * 3006005019077469174];
    signal tmp_7399[3] <== [evals[123][0] * 17667700303515849362, evals[123][1] * 17667700303515849362, evals[123][2] * 17667700303515849362];
    signal tmp_7400[3] <== [tmp_7398[0] + tmp_7399[0], tmp_7398[1] + tmp_7399[1], tmp_7398[2] + tmp_7399[2]];
    signal tmp_7401[3] <== [evals[124][0] * 9400421283898153864, evals[124][1] * 9400421283898153864, evals[124][2] * 9400421283898153864];
    signal tmp_7402[3] <== [tmp_7400[0] + tmp_7401[0], tmp_7400[1] + tmp_7401[1], tmp_7400[2] + tmp_7401[2]];
    signal tmp_7403[3] <== [evals[54][0] * 9342245617122491936, evals[54][1] * 9342245617122491936, evals[54][2] * 9342245617122491936];
    signal tmp_7404[3] <== [tmp_7402[0] + tmp_7403[0], tmp_7402[1] + tmp_7403[1], tmp_7402[2] + tmp_7403[2]];
    signal tmp_7405[3] <== [evals[84][0] + tmp_7404[0], evals[84][1] + tmp_7404[1], evals[84][2] + tmp_7404[2]];
    signal tmp_7407[3] <== CMul()(tmp_7405, tmp_7405);
    signal tmp_7408[3] <== CMul()(tmp_7407, tmp_7407);
    signal tmp_7409[3] <== CMul()(tmp_7408, tmp_7407);
    signal tmp_7411[3] <== CMul()(tmp_7409, tmp_7405);
    signal tmp_7413[3] <== [tmp_6936[0] * 18314693207014427912, tmp_6936[1] * 18314693207014427912, tmp_6936[2] * 18314693207014427912];
    signal tmp_7414[3] <== [evals[123][0] * 13451932870729425780, evals[123][1] * 13451932870729425780, evals[123][2] * 13451932870729425780];
    signal tmp_7415[3] <== [tmp_7413[0] + tmp_7414[0], tmp_7413[1] + tmp_7414[1], tmp_7413[2] + tmp_7414[2]];
    signal tmp_7416[3] <== [evals[124][0] * 16740473708727660723, evals[124][1] * 16740473708727660723, evals[124][2] * 16740473708727660723];
    signal tmp_7417[3] <== [tmp_7415[0] + tmp_7416[0], tmp_7415[1] + tmp_7416[1], tmp_7415[2] + tmp_7416[2]];
    signal tmp_7418[3] <== [evals[54][0] * 824922755340087111, evals[54][1] * 824922755340087111, evals[54][2] * 824922755340087111];
    signal tmp_7419[3] <== [tmp_7417[0] + tmp_7418[0], tmp_7417[1] + tmp_7418[1], tmp_7417[2] + tmp_7418[2]];
    signal tmp_7420[3] <== [evals[85][0] + tmp_7419[0], evals[85][1] + tmp_7419[1], evals[85][2] + tmp_7419[2]];
    signal tmp_7422[3] <== CMul()(tmp_7420, tmp_7420);
    signal tmp_7423[3] <== CMul()(tmp_7422, tmp_7422);
    signal tmp_7424[3] <== CMul()(tmp_7423, tmp_7422);
    signal tmp_7426[3] <== CMul()(tmp_7424, tmp_7420);
    signal tmp_7427[3] <== [tmp_7411[0] + tmp_7426[0], tmp_7411[1] + tmp_7426[1], tmp_7411[2] + tmp_7426[2]];
    signal tmp_7428[3] <== [tmp_7396[0] + tmp_7427[0], tmp_7396[1] + tmp_7427[1], tmp_7396[2] + tmp_7427[2]];
    signal tmp_7429[3] <== [4 * tmp_7427[0], 4 * tmp_7427[1], 4 * tmp_7427[2]];
    signal tmp_7432[3] <== [2 * tmp_7426[0], 2 * tmp_7426[1], 2 * tmp_7426[2]];
    signal tmp_7434[3] <== [tmp_6936[0] * 10170571510627764565, tmp_6936[1] * 10170571510627764565, tmp_6936[2] * 10170571510627764565];
    signal tmp_7435[3] <== [evals[123][0] * 17496644301504593329, evals[123][1] * 17496644301504593329, evals[123][2] * 17496644301504593329];
    signal tmp_7436[3] <== [tmp_7434[0] + tmp_7435[0], tmp_7434[1] + tmp_7435[1], tmp_7434[2] + tmp_7435[2]];
    signal tmp_7437[3] <== [evals[124][0] * 7287335615015370599, evals[124][1] * 7287335615015370599, evals[124][2] * 7287335615015370599];
    signal tmp_7438[3] <== [tmp_7436[0] + tmp_7437[0], tmp_7436[1] + tmp_7437[1], tmp_7436[2] + tmp_7437[2]];
    signal tmp_7439[3] <== [evals[54][0] * 5398764608665671277, evals[54][1] * 5398764608665671277, evals[54][2] * 5398764608665671277];
    signal tmp_7440[3] <== [tmp_7438[0] + tmp_7439[0], tmp_7438[1] + tmp_7439[1], tmp_7438[2] + tmp_7439[2]];
    signal tmp_7441[3] <== [evals[86][0] + tmp_7440[0], evals[86][1] + tmp_7440[1], evals[86][2] + tmp_7440[2]];
    signal tmp_7443[3] <== CMul()(tmp_7441, tmp_7441);
    signal tmp_7444[3] <== CMul()(tmp_7443, tmp_7443);
    signal tmp_7445[3] <== CMul()(tmp_7444, tmp_7443);
    signal tmp_7447[3] <== CMul()(tmp_7445, tmp_7441);
    signal tmp_7450[3] <== [tmp_7447[0] + tmp_7395[0], tmp_7447[1] + tmp_7395[1], tmp_7447[2] + tmp_7395[2]];
    signal tmp_7451[3] <== [tmp_7432[0] + tmp_7450[0], tmp_7432[1] + tmp_7450[1], tmp_7432[2] + tmp_7450[2]];
    signal tmp_7452[3] <== [tmp_7429[0] + tmp_7451[0], tmp_7429[1] + tmp_7451[1], tmp_7429[2] + tmp_7451[2]];
    signal tmp_7453[3] <== [tmp_7428[0] + tmp_7452[0], tmp_7428[1] + tmp_7452[1], tmp_7428[2] + tmp_7452[2]];
    signal tmp_7454[3] <== [tmp_7380[0] + tmp_7453[0], tmp_7380[1] + tmp_7453[1], tmp_7380[2] + tmp_7453[2]];
    signal tmp_7456[3] <== [tmp_6936[0] * 3151070045406026687, tmp_6936[1] * 3151070045406026687, tmp_6936[2] * 3151070045406026687];
    signal tmp_7457[3] <== [evals[123][0] * 15182093233826554989, evals[123][1] * 15182093233826554989, evals[123][2] * 15182093233826554989];
    signal tmp_7458[3] <== [tmp_7456[0] + tmp_7457[0], tmp_7456[1] + tmp_7457[1], tmp_7456[2] + tmp_7457[2]];
    signal tmp_7459[3] <== [evals[124][0] * 3483661079725019706, evals[124][1] * 3483661079725019706, evals[124][2] * 3483661079725019706];
    signal tmp_7460[3] <== [tmp_7458[0] + tmp_7459[0], tmp_7458[1] + tmp_7459[1], tmp_7458[2] + tmp_7459[2]];
    signal tmp_7461[3] <== [evals[54][0] * 587812931475326787, evals[54][1] * 587812931475326787, evals[54][2] * 587812931475326787];
    signal tmp_7462[3] <== [tmp_7460[0] + tmp_7461[0], tmp_7460[1] + tmp_7461[1], tmp_7460[2] + tmp_7461[2]];
    signal tmp_7463[3] <== [evals[91][0] + tmp_7462[0], evals[91][1] + tmp_7462[1], evals[91][2] + tmp_7462[2]];
    signal tmp_7465[3] <== CMul()(tmp_7463, tmp_7463);
    signal tmp_7466[3] <== CMul()(tmp_7465, tmp_7465);
    signal tmp_7467[3] <== CMul()(tmp_7466, tmp_7465);
    signal tmp_7469[3] <== CMul()(tmp_7467, tmp_7463);
    signal tmp_7470[3] <== [2 * tmp_7469[0], 2 * tmp_7469[1], 2 * tmp_7469[2]];
    signal tmp_7472[3] <== [tmp_6936[0] * 3983470257916202094, tmp_6936[1] * 3983470257916202094, tmp_6936[2] * 3983470257916202094];
    signal tmp_7473[3] <== [evals[123][0] * 11124488067381312952, evals[123][1] * 11124488067381312952, evals[123][2] * 11124488067381312952];
    signal tmp_7474[3] <== [tmp_7472[0] + tmp_7473[0], tmp_7472[1] + tmp_7473[1], tmp_7472[2] + tmp_7473[2]];
    signal tmp_7475[3] <== [evals[124][0] * 12655834778591227242, evals[124][1] * 12655834778591227242, evals[124][2] * 12655834778591227242];
    signal tmp_7476[3] <== [tmp_7474[0] + tmp_7475[0], tmp_7474[1] + tmp_7475[1], tmp_7474[2] + tmp_7475[2]];
    signal tmp_7477[3] <== [evals[54][0] * 4921899302518525668, evals[54][1] * 4921899302518525668, evals[54][2] * 4921899302518525668];
    signal tmp_7478[3] <== [tmp_7476[0] + tmp_7477[0], tmp_7476[1] + tmp_7477[1], tmp_7476[2] + tmp_7477[2]];
    signal tmp_7479[3] <== [evals[88][0] + tmp_7478[0], evals[88][1] + tmp_7478[1], evals[88][2] + tmp_7478[2]];
    signal tmp_7481[3] <== CMul()(tmp_7479, tmp_7479);
    signal tmp_7482[3] <== CMul()(tmp_7481, tmp_7481);
    signal tmp_7483[3] <== CMul()(tmp_7482, tmp_7481);
    signal tmp_7485[3] <== CMul()(tmp_7483, tmp_7479);
    signal tmp_7487[3] <== [tmp_6936[0] * 3423470109396435354, tmp_6936[1] * 3423470109396435354, tmp_6936[2] * 3423470109396435354];
    signal tmp_7488[3] <== [evals[123][0] * 14273583775934080755, evals[123][1] * 14273583775934080755, evals[123][2] * 14273583775934080755];
    signal tmp_7489[3] <== [tmp_7487[0] + tmp_7488[0], tmp_7487[1] + tmp_7488[1], tmp_7487[2] + tmp_7488[2]];
    signal tmp_7490[3] <== [evals[124][0] * 7516917516498792203, evals[124][1] * 7516917516498792203, evals[124][2] * 7516917516498792203];
    signal tmp_7491[3] <== [tmp_7489[0] + tmp_7490[0], tmp_7489[1] + tmp_7490[1], tmp_7489[2] + tmp_7490[2]];
    signal tmp_7492[3] <== [evals[54][0] * 10676667276121395674, evals[54][1] * 10676667276121395674, evals[54][2] * 10676667276121395674];
    signal tmp_7493[3] <== [tmp_7491[0] + tmp_7492[0], tmp_7491[1] + tmp_7492[1], tmp_7491[2] + tmp_7492[2]];
    signal tmp_7494[3] <== [evals[89][0] + tmp_7493[0], evals[89][1] + tmp_7493[1], evals[89][2] + tmp_7493[2]];
    signal tmp_7496[3] <== CMul()(tmp_7494, tmp_7494);
    signal tmp_7497[3] <== CMul()(tmp_7496, tmp_7496);
    signal tmp_7498[3] <== CMul()(tmp_7497, tmp_7496);
    signal tmp_7500[3] <== CMul()(tmp_7498, tmp_7494);
    signal tmp_7501[3] <== [tmp_7485[0] + tmp_7500[0], tmp_7485[1] + tmp_7500[1], tmp_7485[2] + tmp_7500[2]];
    signal tmp_7502[3] <== [tmp_7470[0] + tmp_7501[0], tmp_7470[1] + tmp_7501[1], tmp_7470[2] + tmp_7501[2]];
    signal tmp_7503[3] <== [4 * tmp_7501[0], 4 * tmp_7501[1], 4 * tmp_7501[2]];
    signal tmp_7506[3] <== [2 * tmp_7500[0], 2 * tmp_7500[1], 2 * tmp_7500[2]];
    signal tmp_7508[3] <== [tmp_6936[0] * 3450488264035752368, tmp_6936[1] * 3450488264035752368, tmp_6936[2] * 3450488264035752368];
    signal tmp_7509[3] <== [evals[123][0] * 13257379154210125715, evals[123][1] * 13257379154210125715, evals[123][2] * 13257379154210125715];
    signal tmp_7510[3] <== [tmp_7508[0] + tmp_7509[0], tmp_7508[1] + tmp_7509[1], tmp_7508[2] + tmp_7509[2]];
    signal tmp_7511[3] <== [evals[124][0] * 17291898461659961460, evals[124][1] * 17291898461659961460, evals[124][2] * 17291898461659961460];
    signal tmp_7512[3] <== [tmp_7510[0] + tmp_7511[0], tmp_7510[1] + tmp_7511[1], tmp_7510[2] + tmp_7511[2]];
    signal tmp_7513[3] <== [evals[54][0] * 16304445044992359788, evals[54][1] * 16304445044992359788, evals[54][2] * 16304445044992359788];
    signal tmp_7514[3] <== [tmp_7512[0] + tmp_7513[0], tmp_7512[1] + tmp_7513[1], tmp_7512[2] + tmp_7513[2]];
    signal tmp_7515[3] <== [evals[90][0] + tmp_7514[0], evals[90][1] + tmp_7514[1], evals[90][2] + tmp_7514[2]];
    signal tmp_7517[3] <== CMul()(tmp_7515, tmp_7515);
    signal tmp_7518[3] <== CMul()(tmp_7517, tmp_7517);
    signal tmp_7519[3] <== CMul()(tmp_7518, tmp_7517);
    signal tmp_7521[3] <== CMul()(tmp_7519, tmp_7515);
    signal tmp_7524[3] <== [tmp_7521[0] + tmp_7469[0], tmp_7521[1] + tmp_7469[1], tmp_7521[2] + tmp_7469[2]];
    signal tmp_7525[3] <== [tmp_7506[0] + tmp_7524[0], tmp_7506[1] + tmp_7524[1], tmp_7506[2] + tmp_7524[2]];
    signal tmp_7526[3] <== [tmp_7503[0] + tmp_7525[0], tmp_7503[1] + tmp_7525[1], tmp_7503[2] + tmp_7525[2]];
    signal tmp_7527[3] <== [tmp_7502[0] + tmp_7526[0], tmp_7502[1] + tmp_7526[1], tmp_7502[2] + tmp_7526[2]];
    signal tmp_7528[3] <== [tmp_7454[0] + tmp_7527[0], tmp_7454[1] + tmp_7527[1], tmp_7454[2] + tmp_7527[2]];
    signal tmp_7530[3] <== [tmp_6936[0] * 7780139165529418388, tmp_6936[1] * 7780139165529418388, tmp_6936[2] * 7780139165529418388];
    signal tmp_7531[3] <== [evals[123][0] * 3717595218375990594, evals[123][1] * 3717595218375990594, evals[123][2] * 3717595218375990594];
    signal tmp_7532[3] <== [tmp_7530[0] + tmp_7531[0], tmp_7530[1] + tmp_7531[1], tmp_7530[2] + tmp_7531[2]];
    signal tmp_7533[3] <== [evals[124][0] * 17138720803294251067, evals[124][1] * 17138720803294251067, evals[124][2] * 17138720803294251067];
    signal tmp_7534[3] <== [tmp_7532[0] + tmp_7533[0], tmp_7532[1] + tmp_7533[1], tmp_7532[2] + tmp_7533[2]];
    signal tmp_7535[3] <== [evals[54][0] * 15665352327749112015, evals[54][1] * 15665352327749112015, evals[54][2] * 15665352327749112015];
    signal tmp_7536[3] <== [tmp_7534[0] + tmp_7535[0], tmp_7534[1] + tmp_7535[1], tmp_7534[2] + tmp_7535[2]];
    signal tmp_7537[3] <== [evals[95][0] + tmp_7536[0], evals[95][1] + tmp_7536[1], evals[95][2] + tmp_7536[2]];
    signal tmp_7539[3] <== CMul()(tmp_7537, tmp_7537);
    signal tmp_7540[3] <== CMul()(tmp_7539, tmp_7539);
    signal tmp_7541[3] <== CMul()(tmp_7540, tmp_7539);
    signal tmp_7543[3] <== CMul()(tmp_7541, tmp_7537);
    signal tmp_7544[3] <== [2 * tmp_7543[0], 2 * tmp_7543[1], 2 * tmp_7543[2]];
    signal tmp_7546[3] <== [tmp_6936[0] * 13462781804006123550, tmp_6936[1] * 13462781804006123550, tmp_6936[2] * 13462781804006123550];
    signal tmp_7547[3] <== [evals[123][0] * 5343815959848316783, evals[123][1] * 5343815959848316783, evals[123][2] * 5343815959848316783];
    signal tmp_7548[3] <== [tmp_7546[0] + tmp_7547[0], tmp_7546[1] + tmp_7547[1], tmp_7546[2] + tmp_7547[2]];
    signal tmp_7549[3] <== [evals[124][0] * 18038566855851505564, evals[124][1] * 18038566855851505564, evals[124][2] * 18038566855851505564];
    signal tmp_7550[3] <== [tmp_7548[0] + tmp_7549[0], tmp_7548[1] + tmp_7549[1], tmp_7548[2] + tmp_7549[2]];
    signal tmp_7551[3] <== [evals[54][0] * 7627897473303434798, evals[54][1] * 7627897473303434798, evals[54][2] * 7627897473303434798];
    signal tmp_7552[3] <== [tmp_7550[0] + tmp_7551[0], tmp_7550[1] + tmp_7551[1], tmp_7550[2] + tmp_7551[2]];
    signal tmp_7553[3] <== [evals[92][0] + tmp_7552[0], evals[92][1] + tmp_7552[1], evals[92][2] + tmp_7552[2]];
    signal tmp_7555[3] <== CMul()(tmp_7553, tmp_7553);
    signal tmp_7556[3] <== CMul()(tmp_7555, tmp_7555);
    signal tmp_7557[3] <== CMul()(tmp_7556, tmp_7555);
    signal tmp_7559[3] <== CMul()(tmp_7557, tmp_7553);
    signal tmp_7561[3] <== [tmp_6936[0] * 13288575772684627216, tmp_6936[1] * 13288575772684627216, tmp_6936[2] * 13288575772684627216];
    signal tmp_7562[3] <== [evals[123][0] * 5696193414571279556, evals[123][1] * 5696193414571279556, evals[123][2] * 5696193414571279556];
    signal tmp_7563[3] <== [tmp_7561[0] + tmp_7562[0], tmp_7561[1] + tmp_7562[1], tmp_7561[2] + tmp_7562[2]];
    signal tmp_7564[3] <== [evals[124][0] * 6166154478921479976, evals[124][1] * 6166154478921479976, evals[124][2] * 6166154478921479976];
    signal tmp_7565[3] <== [tmp_7563[0] + tmp_7564[0], tmp_7563[1] + tmp_7564[1], tmp_7563[2] + tmp_7564[2]];
    signal tmp_7566[3] <== [evals[54][0] * 2633542348821978290, evals[54][1] * 2633542348821978290, evals[54][2] * 2633542348821978290];
    signal tmp_7567[3] <== [tmp_7565[0] + tmp_7566[0], tmp_7565[1] + tmp_7566[1], tmp_7565[2] + tmp_7566[2]];
    signal tmp_7568[3] <== [evals[93][0] + tmp_7567[0], evals[93][1] + tmp_7567[1], evals[93][2] + tmp_7567[2]];
    signal tmp_7570[3] <== CMul()(tmp_7568, tmp_7568);
    signal tmp_7571[3] <== CMul()(tmp_7570, tmp_7570);
    signal tmp_7572[3] <== CMul()(tmp_7571, tmp_7570);
    signal tmp_7574[3] <== CMul()(tmp_7572, tmp_7568);
    signal tmp_7575[3] <== [tmp_7559[0] + tmp_7574[0], tmp_7559[1] + tmp_7574[1], tmp_7559[2] + tmp_7574[2]];
    signal tmp_7576[3] <== [tmp_7544[0] + tmp_7575[0], tmp_7544[1] + tmp_7575[1], tmp_7544[2] + tmp_7575[2]];
    signal tmp_7577[3] <== [4 * tmp_7575[0], 4 * tmp_7575[1], 4 * tmp_7575[2]];
    signal tmp_7580[3] <== [2 * tmp_7574[0], 2 * tmp_7574[1], 2 * tmp_7574[2]];
    signal tmp_7582[3] <== [tmp_6936[0] * 13745549378090937523, tmp_6936[1] * 13745549378090937523, tmp_6936[2] * 13745549378090937523];
    signal tmp_7583[3] <== [evals[123][0] * 6089787229781325813, evals[123][1] * 6089787229781325813, evals[123][2] * 6089787229781325813];
    signal tmp_7584[3] <== [tmp_7582[0] + tmp_7583[0], tmp_7582[1] + tmp_7583[1], tmp_7582[2] + tmp_7583[2]];
    signal tmp_7585[3] <== [evals[124][0] * 8824538844032254328, evals[124][1] * 8824538844032254328, evals[124][2] * 8824538844032254328];
    signal tmp_7586[3] <== [tmp_7584[0] + tmp_7585[0], tmp_7584[1] + tmp_7585[1], tmp_7584[2] + tmp_7585[2]];
    signal tmp_7587[3] <== [evals[54][0] * 5714662808802647771, evals[54][1] * 5714662808802647771, evals[54][2] * 5714662808802647771];
    signal tmp_7588[3] <== [tmp_7586[0] + tmp_7587[0], tmp_7586[1] + tmp_7587[1], tmp_7586[2] + tmp_7587[2]];
    signal tmp_7589[3] <== [evals[94][0] + tmp_7588[0], evals[94][1] + tmp_7588[1], evals[94][2] + tmp_7588[2]];
    signal tmp_7591[3] <== CMul()(tmp_7589, tmp_7589);
    signal tmp_7592[3] <== CMul()(tmp_7591, tmp_7591);
    signal tmp_7593[3] <== CMul()(tmp_7592, tmp_7591);
    signal tmp_7595[3] <== CMul()(tmp_7593, tmp_7589);
    signal tmp_7598[3] <== [tmp_7595[0] + tmp_7543[0], tmp_7595[1] + tmp_7543[1], tmp_7595[2] + tmp_7543[2]];
    signal tmp_7599[3] <== [tmp_7580[0] + tmp_7598[0], tmp_7580[1] + tmp_7598[1], tmp_7580[2] + tmp_7598[2]];
    signal tmp_7600[3] <== [tmp_7577[0] + tmp_7599[0], tmp_7577[1] + tmp_7599[1], tmp_7577[2] + tmp_7599[2]];
    signal tmp_7601[3] <== [tmp_7576[0] + tmp_7600[0], tmp_7576[1] + tmp_7600[1], tmp_7576[2] + tmp_7600[2]];
    signal tmp_7602[3] <== [tmp_7528[0] + tmp_7601[0], tmp_7528[1] + tmp_7601[1], tmp_7528[2] + tmp_7601[2]];
    signal tmp_7603[3] <== [tmp_7380[0] + tmp_7602[0], tmp_7380[1] + tmp_7602[1], tmp_7380[2] + tmp_7602[2]];
    signal tmp_7604[3] <== [evals[96][0] - tmp_7603[0], evals[96][1] - tmp_7603[1], evals[96][2] - tmp_7603[2]];
    signal tmp_7605[3] <== CMul()(tmp_7307, tmp_7604);
    signal tmp_7606[3] <== [tmp_7303[0] + tmp_7605[0], tmp_7303[1] + tmp_7605[1], tmp_7303[2] + tmp_7605[2]];
    signal tmp_7607[3] <== CMul()(challengeQ, tmp_7606);
    signal tmp_7612[3] <== [tmp_7379[0] + tmp_7452[0], tmp_7379[1] + tmp_7452[1], tmp_7379[2] + tmp_7452[2]];
    signal tmp_7613[3] <== [tmp_7612[0] + tmp_7526[0], tmp_7612[1] + tmp_7526[1], tmp_7612[2] + tmp_7526[2]];
    signal tmp_7614[3] <== [tmp_7613[0] + tmp_7600[0], tmp_7613[1] + tmp_7600[1], tmp_7613[2] + tmp_7600[2]];
    signal tmp_7615[3] <== [tmp_7379[0] + tmp_7614[0], tmp_7379[1] + tmp_7614[1], tmp_7379[2] + tmp_7614[2]];
    signal tmp_7616[3] <== [evals[97][0] - tmp_7615[0], evals[97][1] - tmp_7615[1], evals[97][2] - tmp_7615[2]];
    signal tmp_7617[3] <== CMul()(tmp_7307, tmp_7616);
    signal tmp_7618[3] <== [tmp_7607[0] + tmp_7617[0], tmp_7607[1] + tmp_7617[1], tmp_7607[2] + tmp_7617[2]];
    signal tmp_7619[3] <== CMul()(challengeQ, tmp_7618);
    signal tmp_7624[3] <== [4 * tmp_7377[0], 4 * tmp_7377[1], 4 * tmp_7377[2]];
    signal tmp_7625[3] <== [tmp_7624[0] + tmp_7355[0], tmp_7624[1] + tmp_7355[1], tmp_7624[2] + tmp_7355[2]];
    signal tmp_7626[3] <== [tmp_7378[0] + tmp_7625[0], tmp_7378[1] + tmp_7625[1], tmp_7378[2] + tmp_7625[2]];
    signal tmp_7627[3] <== [4 * tmp_7450[0], 4 * tmp_7450[1], 4 * tmp_7450[2]];
    signal tmp_7628[3] <== [tmp_7627[0] + tmp_7428[0], tmp_7627[1] + tmp_7428[1], tmp_7627[2] + tmp_7428[2]];
    signal tmp_7629[3] <== [tmp_7451[0] + tmp_7628[0], tmp_7451[1] + tmp_7628[1], tmp_7451[2] + tmp_7628[2]];
    signal tmp_7630[3] <== [tmp_7626[0] + tmp_7629[0], tmp_7626[1] + tmp_7629[1], tmp_7626[2] + tmp_7629[2]];
    signal tmp_7631[3] <== [4 * tmp_7524[0], 4 * tmp_7524[1], 4 * tmp_7524[2]];
    signal tmp_7632[3] <== [tmp_7631[0] + tmp_7502[0], tmp_7631[1] + tmp_7502[1], tmp_7631[2] + tmp_7502[2]];
    signal tmp_7633[3] <== [tmp_7525[0] + tmp_7632[0], tmp_7525[1] + tmp_7632[1], tmp_7525[2] + tmp_7632[2]];
    signal tmp_7634[3] <== [tmp_7630[0] + tmp_7633[0], tmp_7630[1] + tmp_7633[1], tmp_7630[2] + tmp_7633[2]];
    signal tmp_7635[3] <== [4 * tmp_7598[0], 4 * tmp_7598[1], 4 * tmp_7598[2]];
    signal tmp_7636[3] <== [tmp_7635[0] + tmp_7576[0], tmp_7635[1] + tmp_7576[1], tmp_7635[2] + tmp_7576[2]];
    signal tmp_7637[3] <== [tmp_7599[0] + tmp_7636[0], tmp_7599[1] + tmp_7636[1], tmp_7599[2] + tmp_7636[2]];
    signal tmp_7638[3] <== [tmp_7634[0] + tmp_7637[0], tmp_7634[1] + tmp_7637[1], tmp_7634[2] + tmp_7637[2]];
    signal tmp_7639[3] <== [tmp_7626[0] + tmp_7638[0], tmp_7626[1] + tmp_7638[1], tmp_7626[2] + tmp_7638[2]];
    signal tmp_7640[3] <== [evals[98][0] - tmp_7639[0], evals[98][1] - tmp_7639[1], evals[98][2] - tmp_7639[2]];
    signal tmp_7641[3] <== CMul()(tmp_7307, tmp_7640);
    signal tmp_7642[3] <== [tmp_7619[0] + tmp_7641[0], tmp_7619[1] + tmp_7641[1], tmp_7619[2] + tmp_7641[2]];
    signal tmp_7643[3] <== CMul()(challengeQ, tmp_7642);
    signal tmp_7648[3] <== [tmp_7625[0] + tmp_7628[0], tmp_7625[1] + tmp_7628[1], tmp_7625[2] + tmp_7628[2]];
    signal tmp_7649[3] <== [tmp_7648[0] + tmp_7632[0], tmp_7648[1] + tmp_7632[1], tmp_7648[2] + tmp_7632[2]];
    signal tmp_7650[3] <== [tmp_7649[0] + tmp_7636[0], tmp_7649[1] + tmp_7636[1], tmp_7649[2] + tmp_7636[2]];
    signal tmp_7651[3] <== [tmp_7625[0] + tmp_7650[0], tmp_7625[1] + tmp_7650[1], tmp_7625[2] + tmp_7650[2]];
    signal tmp_7652[3] <== [evals[99][0] - tmp_7651[0], evals[99][1] - tmp_7651[1], evals[99][2] - tmp_7651[2]];
    signal tmp_7653[3] <== CMul()(tmp_7307, tmp_7652);
    signal tmp_7654[3] <== [tmp_7643[0] + tmp_7653[0], tmp_7643[1] + tmp_7653[1], tmp_7643[2] + tmp_7653[2]];
    signal tmp_7655[3] <== CMul()(challengeQ, tmp_7654);
    signal tmp_7660[3] <== [tmp_7453[0] + tmp_7602[0], tmp_7453[1] + tmp_7602[1], tmp_7453[2] + tmp_7602[2]];
    signal tmp_7661[3] <== [evals[100][0] - tmp_7660[0], evals[100][1] - tmp_7660[1], evals[100][2] - tmp_7660[2]];
    signal tmp_7662[3] <== CMul()(tmp_7307, tmp_7661);
    signal tmp_7663[3] <== [tmp_7655[0] + tmp_7662[0], tmp_7655[1] + tmp_7662[1], tmp_7655[2] + tmp_7662[2]];
    signal tmp_7664[3] <== CMul()(challengeQ, tmp_7663);
    signal tmp_7669[3] <== [tmp_7452[0] + tmp_7614[0], tmp_7452[1] + tmp_7614[1], tmp_7452[2] + tmp_7614[2]];
    signal tmp_7670[3] <== [evals[101][0] - tmp_7669[0], evals[101][1] - tmp_7669[1], evals[101][2] - tmp_7669[2]];
    signal tmp_7671[3] <== CMul()(tmp_7307, tmp_7670);
    signal tmp_7672[3] <== [tmp_7664[0] + tmp_7671[0], tmp_7664[1] + tmp_7671[1], tmp_7664[2] + tmp_7671[2]];
    signal tmp_7673[3] <== CMul()(challengeQ, tmp_7672);
    signal tmp_7678[3] <== [tmp_7629[0] + tmp_7638[0], tmp_7629[1] + tmp_7638[1], tmp_7629[2] + tmp_7638[2]];
    signal tmp_7679[3] <== [evals[102][0] - tmp_7678[0], evals[102][1] - tmp_7678[1], evals[102][2] - tmp_7678[2]];
    signal tmp_7680[3] <== CMul()(tmp_7307, tmp_7679);
    signal tmp_7681[3] <== [tmp_7673[0] + tmp_7680[0], tmp_7673[1] + tmp_7680[1], tmp_7673[2] + tmp_7680[2]];
    signal tmp_7682[3] <== CMul()(challengeQ, tmp_7681);
    signal tmp_7687[3] <== [tmp_7628[0] + tmp_7650[0], tmp_7628[1] + tmp_7650[1], tmp_7628[2] + tmp_7650[2]];
    signal tmp_7688[3] <== [evals[103][0] - tmp_7687[0], evals[103][1] - tmp_7687[1], evals[103][2] - tmp_7687[2]];
    signal tmp_7689[3] <== CMul()(tmp_7307, tmp_7688);
    signal tmp_7690[3] <== [tmp_7682[0] + tmp_7689[0], tmp_7682[1] + tmp_7689[1], tmp_7682[2] + tmp_7689[2]];
    signal tmp_7691[3] <== CMul()(challengeQ, tmp_7690);
    signal tmp_7696[3] <== [tmp_7527[0] + tmp_7602[0], tmp_7527[1] + tmp_7602[1], tmp_7527[2] + tmp_7602[2]];
    signal tmp_7697[3] <== [evals[104][0] - tmp_7696[0], evals[104][1] - tmp_7696[1], evals[104][2] - tmp_7696[2]];
    signal tmp_7698[3] <== CMul()(tmp_7307, tmp_7697);
    signal tmp_7699[3] <== [tmp_7691[0] + tmp_7698[0], tmp_7691[1] + tmp_7698[1], tmp_7691[2] + tmp_7698[2]];
    signal tmp_7700[3] <== CMul()(challengeQ, tmp_7699);
    signal tmp_7705[3] <== [tmp_7526[0] + tmp_7614[0], tmp_7526[1] + tmp_7614[1], tmp_7526[2] + tmp_7614[2]];
    signal tmp_7706[3] <== [evals[105][0] - tmp_7705[0], evals[105][1] - tmp_7705[1], evals[105][2] - tmp_7705[2]];
    signal tmp_7707[3] <== CMul()(tmp_7307, tmp_7706);
    signal tmp_7708[3] <== [tmp_7700[0] + tmp_7707[0], tmp_7700[1] + tmp_7707[1], tmp_7700[2] + tmp_7707[2]];
    signal tmp_7709[3] <== CMul()(challengeQ, tmp_7708);
    signal tmp_7714[3] <== [tmp_7633[0] + tmp_7638[0], tmp_7633[1] + tmp_7638[1], tmp_7633[2] + tmp_7638[2]];
    signal tmp_7715[3] <== [evals[106][0] - tmp_7714[0], evals[106][1] - tmp_7714[1], evals[106][2] - tmp_7714[2]];
    signal tmp_7716[3] <== CMul()(tmp_7307, tmp_7715);
    signal tmp_7717[3] <== [tmp_7709[0] + tmp_7716[0], tmp_7709[1] + tmp_7716[1], tmp_7709[2] + tmp_7716[2]];
    signal tmp_7718[3] <== CMul()(challengeQ, tmp_7717);
    signal tmp_7723[3] <== [tmp_7632[0] + tmp_7650[0], tmp_7632[1] + tmp_7650[1], tmp_7632[2] + tmp_7650[2]];
    signal tmp_7724[3] <== [evals[107][0] - tmp_7723[0], evals[107][1] - tmp_7723[1], evals[107][2] - tmp_7723[2]];
    signal tmp_7725[3] <== CMul()(tmp_7307, tmp_7724);
    signal tmp_7726[3] <== [tmp_7718[0] + tmp_7725[0], tmp_7718[1] + tmp_7725[1], tmp_7718[2] + tmp_7725[2]];
    signal tmp_7727[3] <== CMul()(challengeQ, tmp_7726);
    signal tmp_7732[3] <== [tmp_7601[0] + tmp_7602[0], tmp_7601[1] + tmp_7602[1], tmp_7601[2] + tmp_7602[2]];
    signal tmp_7733[3] <== [evals[108][0] - tmp_7732[0], evals[108][1] - tmp_7732[1], evals[108][2] - tmp_7732[2]];
    signal tmp_7734[3] <== CMul()(tmp_7307, tmp_7733);
    signal tmp_7735[3] <== [tmp_7727[0] + tmp_7734[0], tmp_7727[1] + tmp_7734[1], tmp_7727[2] + tmp_7734[2]];
    signal tmp_7736[3] <== CMul()(challengeQ, tmp_7735);
    signal tmp_7741[3] <== [tmp_7600[0] + tmp_7614[0], tmp_7600[1] + tmp_7614[1], tmp_7600[2] + tmp_7614[2]];
    signal tmp_7742[3] <== [evals[109][0] - tmp_7741[0], evals[109][1] - tmp_7741[1], evals[109][2] - tmp_7741[2]];
    signal tmp_7743[3] <== CMul()(tmp_7307, tmp_7742);
    signal tmp_7744[3] <== [tmp_7736[0] + tmp_7743[0], tmp_7736[1] + tmp_7743[1], tmp_7736[2] + tmp_7743[2]];
    signal tmp_7745[3] <== CMul()(challengeQ, tmp_7744);
    signal tmp_7750[3] <== [tmp_7637[0] + tmp_7638[0], tmp_7637[1] + tmp_7638[1], tmp_7637[2] + tmp_7638[2]];
    signal tmp_7751[3] <== [evals[110][0] - tmp_7750[0], evals[110][1] - tmp_7750[1], evals[110][2] - tmp_7750[2]];
    signal tmp_7752[3] <== CMul()(tmp_7307, tmp_7751);
    signal tmp_7753[3] <== [tmp_7745[0] + tmp_7752[0], tmp_7745[1] + tmp_7752[1], tmp_7745[2] + tmp_7752[2]];
    signal tmp_7754[3] <== CMul()(challengeQ, tmp_7753);
    signal tmp_7759[3] <== [tmp_7636[0] + tmp_7650[0], tmp_7636[1] + tmp_7650[1], tmp_7636[2] + tmp_7650[2]];
    signal tmp_7760[3] <== [evals[111][0] - tmp_7759[0], evals[111][1] - tmp_7759[1], evals[111][2] - tmp_7759[2]];
    signal tmp_7761[3] <== CMul()(tmp_7307, tmp_7760);
    signal tmp_7762[3] <== [tmp_7754[0] + tmp_7761[0], tmp_7754[1] + tmp_7761[1], tmp_7754[2] + tmp_7761[2]];
    signal tmp_7763[3] <== CMul()(challengeQ, tmp_7762);
    signal tmp_7768[3] <== CMul()(evals[54], evals[64]);
    signal tmp_7769[3] <== [1 - evals[54][0], -evals[54][1], -evals[54][2]];
    signal tmp_7770[3] <== CMul()(tmp_7769, evals[127]);
    signal tmp_7771[3] <== [tmp_7768[0] + tmp_7770[0], tmp_7768[1] + tmp_7770[1], tmp_7768[2] + tmp_7770[2]];
    signal tmp_7773[3] <== [tmp_6936[0] * 12037145371429260559, tmp_6936[1] * 12037145371429260559, tmp_6936[2] * 12037145371429260559];
    signal tmp_7774[3] <== [evals[123][0] * 12343504462597575239, evals[123][1] * 12343504462597575239, evals[123][2] * 12343504462597575239];
    signal tmp_7775[3] <== [tmp_7773[0] + tmp_7774[0], tmp_7773[1] + tmp_7774[1], tmp_7773[2] + tmp_7774[2]];
    signal tmp_7776[3] <== [evals[124][0] * 6108317238076822824, evals[124][1] * 6108317238076822824, evals[124][2] * 6108317238076822824];
    signal tmp_7777[3] <== [tmp_7775[0] + tmp_7776[0], tmp_7775[1] + tmp_7776[1], tmp_7775[2] + tmp_7776[2]];
    signal tmp_7778[3] <== [evals[54][0] * 9076340683484238287, evals[54][1] * 9076340683484238287, evals[54][2] * 9076340683484238287];
    signal tmp_7779[3] <== [tmp_7777[0] + tmp_7778[0], tmp_7777[1] + tmp_7778[1], tmp_7777[2] + tmp_7778[2]];
    signal tmp_7780[3] <== [evals[53][0] * 12399309956582731760, evals[53][1] * 12399309956582731760, evals[53][2] * 12399309956582731760];
    signal tmp_7781[3] <== [tmp_7779[0] + tmp_7780[0], tmp_7779[1] + tmp_7780[1], tmp_7779[2] + tmp_7780[2]];
    signal tmp_7782[3] <== [evals[99][0] + tmp_7781[0], evals[99][1] + tmp_7781[1], evals[99][2] + tmp_7781[2]];
    signal tmp_7784[3] <== CMul()(tmp_7782, tmp_7782);
    signal tmp_7785[3] <== CMul()(tmp_7784, tmp_7784);
    signal tmp_7786[3] <== CMul()(tmp_7785, tmp_7784);
    signal tmp_7788[3] <== CMul()(tmp_7786, tmp_7782);
    signal tmp_7789[3] <== [2 * tmp_7788[0], 2 * tmp_7788[1], 2 * tmp_7788[2]];
    signal tmp_7791[3] <== [tmp_6936[0] * 11699769881923825909, tmp_6936[1] * 11699769881923825909, tmp_6936[2] * 11699769881923825909];
    signal tmp_7792[3] <== [evals[123][0] * 12355842859741995537, evals[123][1] * 12355842859741995537, evals[123][2] * 12355842859741995537];
    signal tmp_7793[3] <== [tmp_7791[0] + tmp_7792[0], tmp_7791[1] + tmp_7792[1], tmp_7791[2] + tmp_7792[2]];
    signal tmp_7794[3] <== [evals[124][0] * 4158286797471309733, evals[124][1] * 4158286797471309733, evals[124][2] * 4158286797471309733];
    signal tmp_7795[3] <== [tmp_7793[0] + tmp_7794[0], tmp_7793[1] + tmp_7794[1], tmp_7793[2] + tmp_7794[2]];
    signal tmp_7796[3] <== [evals[54][0] * 8041781683128153420, evals[54][1] * 8041781683128153420, evals[54][2] * 8041781683128153420];
    signal tmp_7797[3] <== [tmp_7795[0] + tmp_7796[0], tmp_7795[1] + tmp_7796[1], tmp_7795[2] + tmp_7796[2]];
    signal tmp_7798[3] <== [evals[53][0] * 2949845317987848448, evals[53][1] * 2949845317987848448, evals[53][2] * 2949845317987848448];
    signal tmp_7799[3] <== [tmp_7797[0] + tmp_7798[0], tmp_7797[1] + tmp_7798[1], tmp_7797[2] + tmp_7798[2]];
    signal tmp_7800[3] <== [evals[96][0] + tmp_7799[0], evals[96][1] + tmp_7799[1], evals[96][2] + tmp_7799[2]];
    signal tmp_7802[3] <== CMul()(tmp_7800, tmp_7800);
    signal tmp_7803[3] <== CMul()(tmp_7802, tmp_7802);
    signal tmp_7804[3] <== CMul()(tmp_7803, tmp_7802);
    signal tmp_7806[3] <== CMul()(tmp_7804, tmp_7800);
    signal tmp_7808[3] <== [tmp_6936[0] * 4349260652212695338, tmp_6936[1] * 4349260652212695338, tmp_6936[2] * 4349260652212695338];
    signal tmp_7809[3] <== [evals[123][0] * 14160921043125059199, evals[123][1] * 14160921043125059199, evals[123][2] * 14160921043125059199];
    signal tmp_7810[3] <== [tmp_7808[0] + tmp_7809[0], tmp_7808[1] + tmp_7809[1], tmp_7808[2] + tmp_7809[2]];
    signal tmp_7811[3] <== [evals[124][0] * 1589482450994236426, evals[124][1] * 1589482450994236426, evals[124][2] * 1589482450994236426];
    signal tmp_7812[3] <== [tmp_7810[0] + tmp_7811[0], tmp_7810[1] + tmp_7811[1], tmp_7810[2] + tmp_7811[2]];
    signal tmp_7813[3] <== [evals[54][0] * 13172746155287932121, evals[54][1] * 13172746155287932121, evals[54][2] * 13172746155287932121];
    signal tmp_7814[3] <== [tmp_7812[0] + tmp_7813[0], tmp_7812[1] + tmp_7813[1], tmp_7812[2] + tmp_7813[2]];
    signal tmp_7815[3] <== [evals[53][0] * 6935557324168256069, evals[53][1] * 6935557324168256069, evals[53][2] * 6935557324168256069];
    signal tmp_7816[3] <== [tmp_7814[0] + tmp_7815[0], tmp_7814[1] + tmp_7815[1], tmp_7814[2] + tmp_7815[2]];
    signal tmp_7817[3] <== [evals[97][0] + tmp_7816[0], evals[97][1] + tmp_7816[1], evals[97][2] + tmp_7816[2]];
    signal tmp_7819[3] <== CMul()(tmp_7817, tmp_7817);
    signal tmp_7820[3] <== CMul()(tmp_7819, tmp_7819);
    signal tmp_7821[3] <== CMul()(tmp_7820, tmp_7819);
    signal tmp_7823[3] <== CMul()(tmp_7821, tmp_7817);
    signal tmp_7824[3] <== [tmp_7806[0] + tmp_7823[0], tmp_7806[1] + tmp_7823[1], tmp_7806[2] + tmp_7823[2]];
    signal tmp_7825[3] <== [tmp_7789[0] + tmp_7824[0], tmp_7789[1] + tmp_7824[1], tmp_7789[2] + tmp_7824[2]];
    signal tmp_7826[3] <== [4 * tmp_7824[0], 4 * tmp_7824[1], 4 * tmp_7824[2]];
    signal tmp_7829[3] <== [2 * tmp_7823[0], 2 * tmp_7823[1], 2 * tmp_7823[2]];
    signal tmp_7831[3] <== [tmp_6936[0] * 8228348357294662898, tmp_6936[1] * 8228348357294662898, tmp_6936[2] * 8228348357294662898];
    signal tmp_7832[3] <== [evals[123][0] * 13682913322779819091, evals[123][1] * 13682913322779819091, evals[123][2] * 13682913322779819091];
    signal tmp_7833[3] <== [tmp_7831[0] + tmp_7832[0], tmp_7831[1] + tmp_7832[1], tmp_7831[2] + tmp_7832[2]];
    signal tmp_7834[3] <== [evals[124][0] * 4836112672767737407, evals[124][1] * 4836112672767737407, evals[124][2] * 4836112672767737407];
    signal tmp_7835[3] <== [tmp_7833[0] + tmp_7834[0], tmp_7833[1] + tmp_7834[1], tmp_7833[2] + tmp_7834[2]];
    signal tmp_7836[3] <== [evals[54][0] * 6611413049825339003, evals[54][1] * 6611413049825339003, evals[54][2] * 6611413049825339003];
    signal tmp_7837[3] <== [tmp_7835[0] + tmp_7836[0], tmp_7835[1] + tmp_7836[1], tmp_7835[2] + tmp_7836[2]];
    signal tmp_7838[3] <== [evals[53][0] * 2055390472619727748, evals[53][1] * 2055390472619727748, evals[53][2] * 2055390472619727748];
    signal tmp_7839[3] <== [tmp_7837[0] + tmp_7838[0], tmp_7837[1] + tmp_7838[1], tmp_7837[2] + tmp_7838[2]];
    signal tmp_7840[3] <== [evals[98][0] + tmp_7839[0], evals[98][1] + tmp_7839[1], evals[98][2] + tmp_7839[2]];
    signal tmp_7842[3] <== CMul()(tmp_7840, tmp_7840);
    signal tmp_7843[3] <== CMul()(tmp_7842, tmp_7842);
    signal tmp_7844[3] <== CMul()(tmp_7843, tmp_7842);
    signal tmp_7846[3] <== CMul()(tmp_7844, tmp_7840);
    signal tmp_7849[3] <== [tmp_7846[0] + tmp_7788[0], tmp_7846[1] + tmp_7788[1], tmp_7846[2] + tmp_7788[2]];
    signal tmp_7850[3] <== [tmp_7829[0] + tmp_7849[0], tmp_7829[1] + tmp_7849[1], tmp_7829[2] + tmp_7849[2]];
    signal tmp_7851[3] <== [tmp_7826[0] + tmp_7850[0], tmp_7826[1] + tmp_7850[1], tmp_7826[2] + tmp_7850[2]];
    signal tmp_7852[3] <== [tmp_7825[0] + tmp_7851[0], tmp_7825[1] + tmp_7851[1], tmp_7825[2] + tmp_7851[2]];
    signal tmp_7854[3] <== [tmp_6936[0] * 10157815501749214163, tmp_6936[1] * 10157815501749214163, tmp_6936[2] * 10157815501749214163];
    signal tmp_7855[3] <== [evals[123][0] * 690520822232718983, evals[123][1] * 690520822232718983, evals[123][2] * 690520822232718983];
    signal tmp_7856[3] <== [tmp_7854[0] + tmp_7855[0], tmp_7854[1] + tmp_7855[1], tmp_7854[2] + tmp_7855[2]];
    signal tmp_7857[3] <== [evals[124][0] * 6643835433125335951, evals[124][1] * 6643835433125335951, evals[124][2] * 6643835433125335951];
    signal tmp_7858[3] <== [tmp_7856[0] + tmp_7857[0], tmp_7856[1] + tmp_7857[1], tmp_7856[2] + tmp_7857[2]];
    signal tmp_7859[3] <== [evals[54][0] * 16216654076620539925, evals[54][1] * 16216654076620539925, evals[54][2] * 16216654076620539925];
    signal tmp_7860[3] <== [tmp_7858[0] + tmp_7859[0], tmp_7858[1] + tmp_7859[1], tmp_7858[2] + tmp_7859[2]];
    signal tmp_7861[3] <== [evals[53][0] * 14821962908959377092, evals[53][1] * 14821962908959377092, evals[53][2] * 14821962908959377092];
    signal tmp_7862[3] <== [tmp_7860[0] + tmp_7861[0], tmp_7860[1] + tmp_7861[1], tmp_7860[2] + tmp_7861[2]];
    signal tmp_7863[3] <== [evals[103][0] + tmp_7862[0], evals[103][1] + tmp_7862[1], evals[103][2] + tmp_7862[2]];
    signal tmp_7865[3] <== CMul()(tmp_7863, tmp_7863);
    signal tmp_7866[3] <== CMul()(tmp_7865, tmp_7865);
    signal tmp_7867[3] <== CMul()(tmp_7866, tmp_7865);
    tmp_7869 <== CMul()(tmp_7867, tmp_7863);
    signal tmp_7870[3] <== [2 * tmp_7869[0], 2 * tmp_7869[1], 2 * tmp_7869[2]];
    signal tmp_7872[3] <== [tmp_6936[0] * 7743383997471469184, tmp_6936[1] * 7743383997471469184, tmp_6936[2] * 7743383997471469184];
    signal tmp_7873[3] <== [evals[123][0] * 14249643564311509087, evals[123][1] * 14249643564311509087, evals[123][2] * 14249643564311509087];
    signal tmp_7874[3] <== [tmp_7872[0] + tmp_7873[0], tmp_7872[1] + tmp_7873[1], tmp_7872[2] + tmp_7873[2]];
    signal tmp_7875[3] <== [evals[124][0] * 11174425919404088598, evals[124][1] * 11174425919404088598, evals[124][2] * 11174425919404088598];
    signal tmp_7876[3] <== [tmp_7874[0] + tmp_7875[0], tmp_7874[1] + tmp_7875[1], tmp_7874[2] + tmp_7875[2]];
    signal tmp_7877[3] <== [evals[54][0] * 12295614633782496176, evals[54][1] * 12295614633782496176, evals[54][2] * 12295614633782496176];
    signal tmp_7878[3] <== [tmp_7876[0] + tmp_7877[0], tmp_7876[1] + tmp_7877[1], tmp_7876[2] + tmp_7877[2]];
    signal tmp_7879[3] <== [evals[53][0] * 3088058410657255303, evals[53][1] * 3088058410657255303, evals[53][2] * 3088058410657255303];
    signal tmp_7880[3] <== [tmp_7878[0] + tmp_7879[0], tmp_7878[1] + tmp_7879[1], tmp_7878[2] + tmp_7879[2]];
    signal tmp_7881[3] <== [evals[100][0] + tmp_7880[0], evals[100][1] + tmp_7880[1], evals[100][2] + tmp_7880[2]];
    signal tmp_7883[3] <== CMul()(tmp_7881, tmp_7881);
    signal tmp_7884[3] <== CMul()(tmp_7883, tmp_7883);
    signal tmp_7885[3] <== CMul()(tmp_7884, tmp_7883);
    signal tmp_7887[3] <== CMul()(tmp_7885, tmp_7881);
    signal tmp_7889[3] <== [tmp_6936[0] * 7842006993218978420, tmp_6936[1] * 7842006993218978420, tmp_6936[2] * 7842006993218978420];
    signal tmp_7890[3] <== [evals[123][0] * 15849758561085927402, evals[123][1] * 15849758561085927402, evals[123][2] * 15849758561085927402];
    signal tmp_7891[3] <== [tmp_7889[0] + tmp_7890[0], tmp_7889[1] + tmp_7890[1], tmp_7889[2] + tmp_7890[2]];
    signal tmp_7892[3] <== [evals[124][0] * 2755155124184744585, evals[124][1] * 2755155124184744585, evals[124][2] * 2755155124184744585];
    signal tmp_7893[3] <== [tmp_7891[0] + tmp_7892[0], tmp_7891[1] + tmp_7892[1], tmp_7891[2] + tmp_7892[2]];
    signal tmp_7894[3] <== [evals[54][0] * 11414162089917699064, evals[54][1] * 11414162089917699064, evals[54][2] * 11414162089917699064];
    signal tmp_7895[3] <== [tmp_7893[0] + tmp_7894[0], tmp_7893[1] + tmp_7894[1], tmp_7893[2] + tmp_7894[2]];
    signal tmp_7896[3] <== [evals[53][0] * 8790959064194804235, evals[53][1] * 8790959064194804235, evals[53][2] * 8790959064194804235];
    signal tmp_7897[3] <== [tmp_7895[0] + tmp_7896[0], tmp_7895[1] + tmp_7896[1], tmp_7895[2] + tmp_7896[2]];
    signal tmp_7898[3] <== [evals[101][0] + tmp_7897[0], evals[101][1] + tmp_7897[1], evals[101][2] + tmp_7897[2]];
    signal tmp_7900[3] <== CMul()(tmp_7898, tmp_7898);
    signal tmp_7901[3] <== CMul()(tmp_7900, tmp_7900);
    signal tmp_7902[3] <== CMul()(tmp_7901, tmp_7900);
    signal tmp_7904[3] <== CMul()(tmp_7902, tmp_7898);
    signal tmp_7905[3] <== [tmp_7887[0] + tmp_7904[0], tmp_7887[1] + tmp_7904[1], tmp_7887[2] + tmp_7904[2]];
    signal tmp_7906[3] <== [tmp_7870[0] + tmp_7905[0], tmp_7870[1] + tmp_7905[1], tmp_7870[2] + tmp_7905[2]];
    signal tmp_7907[3] <== [4 * tmp_7905[0], 4 * tmp_7905[1], 4 * tmp_7905[2]];
    signal tmp_7910[3] <== [2 * tmp_7904[0], 2 * tmp_7904[1], 2 * tmp_7904[2]];
    signal tmp_7912[3] <== [tmp_6936[0] * 18369471830829996810, tmp_6936[1] * 18369471830829996810, tmp_6936[2] * 18369471830829996810];
    signal tmp_7913[3] <== [evals[123][0] * 12186564686511703262, evals[123][1] * 12186564686511703262, evals[123][2] * 12186564686511703262];
    signal tmp_7914[3] <== [tmp_7912[0] + tmp_7913[0], tmp_7912[1] + tmp_7913[1], tmp_7912[2] + tmp_7913[2]];
    signal tmp_7915[3] <== [evals[124][0] * 13202523778257173956, evals[124][1] * 13202523778257173956, evals[124][2] * 13202523778257173956];
    signal tmp_7916[3] <== [tmp_7914[0] + tmp_7915[0], tmp_7914[1] + tmp_7915[1], tmp_7914[2] + tmp_7915[2]];
    signal tmp_7917[3] <== [evals[54][0] * 13658053783093222498, evals[54][1] * 13658053783093222498, evals[54][2] * 13658053783093222498];
    signal tmp_7918[3] <== [tmp_7916[0] + tmp_7917[0], tmp_7916[1] + tmp_7917[1], tmp_7916[2] + tmp_7917[2]];
    signal tmp_7919[3] <== [evals[53][0] * 4582664136559626757, evals[53][1] * 4582664136559626757, evals[53][2] * 4582664136559626757];
    signal tmp_7920[3] <== [tmp_7918[0] + tmp_7919[0], tmp_7918[1] + tmp_7919[1], tmp_7918[2] + tmp_7919[2]];
    signal tmp_7921[3] <== [evals[102][0] + tmp_7920[0], evals[102][1] + tmp_7920[1], evals[102][2] + tmp_7920[2]];
    signal tmp_7923[3] <== CMul()(tmp_7921, tmp_7921);
    signal tmp_7924[3] <== CMul()(tmp_7923, tmp_7923);
    signal tmp_7925[3] <== CMul()(tmp_7924, tmp_7923);
    tmp_7927 <== CMul()(tmp_7925, tmp_7921);
    signal tmp_7930[3] <== [tmp_7927[0] + tmp_7869[0], tmp_7927[1] + tmp_7869[1], tmp_7927[2] + tmp_7869[2]];
    signal tmp_7931[3] <== [tmp_7910[0] + tmp_7930[0], tmp_7910[1] + tmp_7930[1], tmp_7910[2] + tmp_7930[2]];
    signal tmp_7932[3] <== [tmp_7907[0] + tmp_7931[0], tmp_7907[1] + tmp_7931[1], tmp_7907[2] + tmp_7931[2]];
    signal tmp_7933[3] <== [tmp_7906[0] + tmp_7932[0], tmp_7906[1] + tmp_7932[1], tmp_7906[2] + tmp_7932[2]];
    signal tmp_7934[3] <== [tmp_7852[0] + tmp_7933[0], tmp_7852[1] + tmp_7933[1], tmp_7852[2] + tmp_7933[2]];
    signal tmp_7936[3] <== [tmp_6936[0] * 6178857985076893256, tmp_6936[1] * 6178857985076893256, tmp_6936[2] * 6178857985076893256];
    signal tmp_7937[3] <== [evals[123][0] * 12083506578988585239, evals[123][1] * 12083506578988585239, evals[123][2] * 12083506578988585239];
    signal tmp_7938[3] <== [tmp_7936[0] + tmp_7937[0], tmp_7936[1] + tmp_7937[1], tmp_7936[2] + tmp_7937[2]];
    signal tmp_7939[3] <== [evals[124][0] * 257624298448087449, evals[124][1] * 257624298448087449, evals[124][2] * 257624298448087449];
    signal tmp_7940[3] <== [tmp_7938[0] + tmp_7939[0], tmp_7938[1] + tmp_7939[1], tmp_7938[2] + tmp_7939[2]];
    signal tmp_7941[3] <== [evals[54][0] * 13499421621575350806, evals[54][1] * 13499421621575350806, evals[54][2] * 13499421621575350806];
    signal tmp_7942[3] <== [tmp_7940[0] + tmp_7941[0], tmp_7940[1] + tmp_7941[1], tmp_7940[2] + tmp_7941[2]];
    signal tmp_7943[3] <== [evals[53][0] * 11973738501993105496, evals[53][1] * 11973738501993105496, evals[53][2] * 11973738501993105496];
    signal tmp_7944[3] <== [tmp_7942[0] + tmp_7943[0], tmp_7942[1] + tmp_7943[1], tmp_7942[2] + tmp_7943[2]];
    signal tmp_7945[3] <== [evals[107][0] + tmp_7944[0], evals[107][1] + tmp_7944[1], evals[107][2] + tmp_7944[2]];
    signal tmp_7947[3] <== CMul()(tmp_7945, tmp_7945);
    signal tmp_7948[3] <== CMul()(tmp_7947, tmp_7947);
    signal tmp_7949[3] <== CMul()(tmp_7948, tmp_7947);
    tmp_7951 <== CMul()(tmp_7949, tmp_7945);
    signal tmp_7952[3] <== [2 * tmp_7951[0], 2 * tmp_7951[1], 2 * tmp_7951[2]];
    signal tmp_7954[3] <== [tmp_6936[0] * 11128005290408113568, tmp_6936[1] * 11128005290408113568, tmp_6936[2] * 11128005290408113568];
    signal tmp_7955[3] <== [evals[123][0] * 684200944081655198, evals[123][1] * 684200944081655198, evals[123][2] * 684200944081655198];
    signal tmp_7956[3] <== [tmp_7954[0] + tmp_7955[0], tmp_7954[1] + tmp_7955[1], tmp_7954[2] + tmp_7955[2]];
    signal tmp_7957[3] <== [evals[124][0] * 2748965361441888477, evals[124][1] * 2748965361441888477, evals[124][2] * 2748965361441888477];
    signal tmp_7958[3] <== [tmp_7956[0] + tmp_7957[0], tmp_7956[1] + tmp_7957[1], tmp_7956[2] + tmp_7957[2]];
    signal tmp_7959[3] <== [evals[54][0] * 17782598995085730495, evals[54][1] * 17782598995085730495, evals[54][2] * 17782598995085730495];
    signal tmp_7960[3] <== [tmp_7958[0] + tmp_7959[0], tmp_7958[1] + tmp_7959[1], tmp_7958[2] + tmp_7959[2]];
    signal tmp_7961[3] <== [evals[53][0] * 2002685046256118388, evals[53][1] * 2002685046256118388, evals[53][2] * 2002685046256118388];
    signal tmp_7962[3] <== [tmp_7960[0] + tmp_7961[0], tmp_7960[1] + tmp_7961[1], tmp_7960[2] + tmp_7961[2]];
    signal tmp_7963[3] <== [evals[104][0] + tmp_7962[0], evals[104][1] + tmp_7962[1], evals[104][2] + tmp_7962[2]];
    signal tmp_7965[3] <== CMul()(tmp_7963, tmp_7963);
    signal tmp_7966[3] <== CMul()(tmp_7965, tmp_7965);
    signal tmp_7967[3] <== CMul()(tmp_7966, tmp_7965);
    tmp_7969 <== CMul()(tmp_7967, tmp_7963);
    signal tmp_7971[3] <== [tmp_6936[0] * 12272917402811871099, tmp_6936[1] * 12272917402811871099, tmp_6936[2] * 12272917402811871099];
    signal tmp_7972[3] <== [evals[123][0] * 6008958449718781068, evals[123][1] * 6008958449718781068, evals[123][2] * 6008958449718781068];
    signal tmp_7973[3] <== [tmp_7971[0] + tmp_7972[0], tmp_7971[1] + tmp_7972[1], tmp_7971[2] + tmp_7972[2]];
    signal tmp_7974[3] <== [evals[124][0] * 14297205492001640506, evals[124][1] * 14297205492001640506, evals[124][2] * 14297205492001640506];
    signal tmp_7975[3] <== [tmp_7973[0] + tmp_7974[0], tmp_7973[1] + tmp_7974[1], tmp_7973[2] + tmp_7974[2]];
    signal tmp_7976[3] <== [evals[54][0] * 18362903988365127450, evals[54][1] * 18362903988365127450, evals[54][2] * 18362903988365127450];
    signal tmp_7977[3] <== [tmp_7975[0] + tmp_7976[0], tmp_7975[1] + tmp_7976[1], tmp_7975[2] + tmp_7976[2]];
    signal tmp_7978[3] <== [evals[53][0] * 15689593639089632368, evals[53][1] * 15689593639089632368, evals[53][2] * 15689593639089632368];
    signal tmp_7979[3] <== [tmp_7977[0] + tmp_7978[0], tmp_7977[1] + tmp_7978[1], tmp_7977[2] + tmp_7978[2]];
    signal tmp_7980[3] <== [evals[105][0] + tmp_7979[0], evals[105][1] + tmp_7979[1], evals[105][2] + tmp_7979[2]];
    signal tmp_7982[3] <== CMul()(tmp_7980, tmp_7980);
    signal tmp_7983[3] <== CMul()(tmp_7982, tmp_7982);
    signal tmp_7984[3] <== CMul()(tmp_7983, tmp_7982);
    tmp_7986 <== CMul()(tmp_7984, tmp_7980);
    signal tmp_7987[3] <== [tmp_7969[0] + tmp_7986[0], tmp_7969[1] + tmp_7986[1], tmp_7969[2] + tmp_7986[2]];
    signal tmp_7988[3] <== [tmp_7952[0] + tmp_7987[0], tmp_7952[1] + tmp_7987[1], tmp_7952[2] + tmp_7987[2]];
    signal tmp_7989[3] <== [4 * tmp_7987[0], 4 * tmp_7987[1], 4 * tmp_7987[2]];
    signal tmp_7992[3] <== [2 * tmp_7986[0], 2 * tmp_7986[1], 2 * tmp_7986[2]];
    signal tmp_7994[3] <== [tmp_6936[0] * 106820976709770637, tmp_6936[1] * 106820976709770637, tmp_6936[2] * 106820976709770637];
    signal tmp_7995[3] <== [evals[123][0] * 3098079696536837500, evals[123][1] * 3098079696536837500, evals[123][2] * 3098079696536837500];
    signal tmp_7996[3] <== [tmp_7994[0] + tmp_7995[0], tmp_7994[1] + tmp_7995[1], tmp_7994[2] + tmp_7995[2]];
    signal tmp_7997[3] <== [evals[124][0] * 12595977886586773319, evals[124][1] * 12595977886586773319, evals[124][2] * 12595977886586773319];
    signal tmp_7998[3] <== [tmp_7996[0] + tmp_7997[0], tmp_7996[1] + tmp_7997[1], tmp_7996[2] + tmp_7997[2]];
    signal tmp_7999[3] <== [evals[54][0] * 11596821426909532371, evals[54][1] * 11596821426909532371, evals[54][2] * 11596821426909532371];
    signal tmp_8000[3] <== [tmp_7998[0] + tmp_7999[0], tmp_7998[1] + tmp_7999[1], tmp_7998[2] + tmp_7999[2]];
    signal tmp_8001[3] <== [evals[53][0] * 10512046881800463697, evals[53][1] * 10512046881800463697, evals[53][2] * 10512046881800463697];
    signal tmp_8002[3] <== [tmp_8000[0] + tmp_8001[0], tmp_8000[1] + tmp_8001[1], tmp_8000[2] + tmp_8001[2]];
    signal tmp_8003[3] <== [evals[106][0] + tmp_8002[0], evals[106][1] + tmp_8002[1], evals[106][2] + tmp_8002[2]];
    signal tmp_8005[3] <== CMul()(tmp_8003, tmp_8003);
    signal tmp_8006[3] <== CMul()(tmp_8005, tmp_8005);
    signal tmp_8007[3] <== CMul()(tmp_8006, tmp_8005);
    tmp_8009 <== CMul()(tmp_8007, tmp_8003);
    signal tmp_8012[3] <== [tmp_8009[0] + tmp_7951[0], tmp_8009[1] + tmp_7951[1], tmp_8009[2] + tmp_7951[2]];
    signal tmp_8013[3] <== [tmp_7992[0] + tmp_8012[0], tmp_7992[1] + tmp_8012[1], tmp_7992[2] + tmp_8012[2]];
    signal tmp_8014[3] <== [tmp_7989[0] + tmp_8013[0], tmp_7989[1] + tmp_8013[1], tmp_7989[2] + tmp_8013[2]];
    signal tmp_8015[3] <== [tmp_7988[0] + tmp_8014[0], tmp_7988[1] + tmp_8014[1], tmp_7988[2] + tmp_8014[2]];
    signal tmp_8016[3] <== [tmp_7934[0] + tmp_8015[0], tmp_7934[1] + tmp_8015[1], tmp_7934[2] + tmp_8015[2]];
    signal tmp_8018[3] <== [tmp_6936[0] * 10060334973073197601, tmp_6936[1] * 10060334973073197601, tmp_6936[2] * 10060334973073197601];
    signal tmp_8019[3] <== [evals[123][0] * 17095617549195818058, evals[123][1] * 17095617549195818058, evals[123][2] * 17095617549195818058];
    signal tmp_8020[3] <== [tmp_8018[0] + tmp_8019[0], tmp_8018[1] + tmp_8019[1], tmp_8018[2] + tmp_8019[2]];
    signal tmp_8021[3] <== [evals[124][0] * 7256201493271212720, evals[124][1] * 7256201493271212720, evals[124][2] * 7256201493271212720];
    signal tmp_8022[3] <== [tmp_8020[0] + tmp_8021[0], tmp_8020[1] + tmp_8021[1], tmp_8020[2] + tmp_8021[2]];
    signal tmp_8023[3] <== [evals[54][0] * 4615393526180652407, evals[54][1] * 4615393526180652407, evals[54][2] * 4615393526180652407];
    signal tmp_8024[3] <== [tmp_8022[0] + tmp_8023[0], tmp_8022[1] + tmp_8023[1], tmp_8022[2] + tmp_8023[2]];
    signal tmp_8025[3] <== [evals[53][0] * 17528314872631229800, evals[53][1] * 17528314872631229800, evals[53][2] * 17528314872631229800];
    signal tmp_8026[3] <== [tmp_8024[0] + tmp_8025[0], tmp_8024[1] + tmp_8025[1], tmp_8024[2] + tmp_8025[2]];
    signal tmp_8027[3] <== [evals[111][0] + tmp_8026[0], evals[111][1] + tmp_8026[1], evals[111][2] + tmp_8026[2]];
    signal tmp_8029[3] <== CMul()(tmp_8027, tmp_8027);
    signal tmp_8030[3] <== CMul()(tmp_8029, tmp_8029);
    signal tmp_8031[3] <== CMul()(tmp_8030, tmp_8029);
    tmp_8033 <== CMul()(tmp_8031, tmp_8027);
    signal tmp_8034[3] <== [2 * tmp_8033[0], 2 * tmp_8033[1], 2 * tmp_8033[2]];
    signal tmp_8036[3] <== [tmp_6936[0] * 14036787039817180193, tmp_6936[1] * 14036787039817180193, tmp_6936[2] * 14036787039817180193];
    signal tmp_8037[3] <== [evals[123][0] * 6489163969976034646, evals[123][1] * 6489163969976034646, evals[123][2] * 6489163969976034646];
    signal tmp_8038[3] <== [tmp_8036[0] + tmp_8037[0], tmp_8036[1] + tmp_8037[1], tmp_8036[2] + tmp_8037[2]];
    signal tmp_8039[3] <== [evals[124][0] * 2744326799038764281, evals[124][1] * 2744326799038764281, evals[124][2] * 2744326799038764281];
    signal tmp_8040[3] <== [tmp_8038[0] + tmp_8039[0], tmp_8038[1] + tmp_8039[1], tmp_8038[2] + tmp_8039[2]];
    signal tmp_8041[3] <== [evals[54][0] * 15256312875610823807, evals[54][1] * 15256312875610823807, evals[54][2] * 15256312875610823807];
    signal tmp_8042[3] <== [tmp_8040[0] + tmp_8041[0], tmp_8040[1] + tmp_8041[1], tmp_8040[2] + tmp_8041[2]];
    signal tmp_8043[3] <== [evals[53][0] * 13009070935614847359, evals[53][1] * 13009070935614847359, evals[53][2] * 13009070935614847359];
    signal tmp_8044[3] <== [tmp_8042[0] + tmp_8043[0], tmp_8042[1] + tmp_8043[1], tmp_8042[2] + tmp_8043[2]];
    signal tmp_8045[3] <== [evals[108][0] + tmp_8044[0], evals[108][1] + tmp_8044[1], evals[108][2] + tmp_8044[2]];
    signal tmp_8047[3] <== CMul()(tmp_8045, tmp_8045);
    signal tmp_8048[3] <== CMul()(tmp_8047, tmp_8047);
    signal tmp_8049[3] <== CMul()(tmp_8048, tmp_8047);
    tmp_8051 <== CMul()(tmp_8049, tmp_8045);
    signal tmp_8053[3] <== [tmp_6936[0] * 1659458828486927362, tmp_6936[1] * 1659458828486927362, tmp_6936[2] * 1659458828486927362];
    signal tmp_8054[3] <== [evals[123][0] * 1186411428667285248, evals[123][1] * 1186411428667285248, evals[123][2] * 1186411428667285248];
    signal tmp_8055[3] <== [tmp_8053[0] + tmp_8054[0], tmp_8053[1] + tmp_8054[1], tmp_8053[2] + tmp_8054[2]];
    signal tmp_8056[3] <== [evals[124][0] * 9097808034015176417, evals[124][1] * 9097808034015176417, evals[124][2] * 9097808034015176417];
    signal tmp_8057[3] <== [tmp_8055[0] + tmp_8056[0], tmp_8055[1] + tmp_8056[1], tmp_8055[2] + tmp_8056[2]];
    signal tmp_8058[3] <== [evals[54][0] * 1558929959921731273, evals[54][1] * 1558929959921731273, evals[54][2] * 1558929959921731273];
    signal tmp_8059[3] <== [tmp_8057[0] + tmp_8058[0], tmp_8057[1] + tmp_8058[1], tmp_8057[2] + tmp_8058[2]];
    signal tmp_8060[3] <== [evals[53][0] * 12908033183314664096, evals[53][1] * 12908033183314664096, evals[53][2] * 12908033183314664096];
    signal tmp_8061[3] <== [tmp_8059[0] + tmp_8060[0], tmp_8059[1] + tmp_8060[1], tmp_8059[2] + tmp_8060[2]];
    signal tmp_8062[3] <== [evals[109][0] + tmp_8061[0], evals[109][1] + tmp_8061[1], evals[109][2] + tmp_8061[2]];
    signal tmp_8064[3] <== CMul()(tmp_8062, tmp_8062);
    signal tmp_8065[3] <== CMul()(tmp_8064, tmp_8064);
    signal tmp_8066[3] <== CMul()(tmp_8065, tmp_8064);
    tmp_8068 <== CMul()(tmp_8066, tmp_8062);
    signal tmp_8069[3] <== [tmp_8051[0] + tmp_8068[0], tmp_8051[1] + tmp_8068[1], tmp_8051[2] + tmp_8068[2]];
    signal tmp_8070[3] <== [tmp_8034[0] + tmp_8069[0], tmp_8034[1] + tmp_8069[1], tmp_8034[2] + tmp_8069[2]];
    signal tmp_8071[3] <== [4 * tmp_8069[0], 4 * tmp_8069[1], 4 * tmp_8069[2]];
    signal tmp_8074[3] <== [2 * tmp_8068[0], 2 * tmp_8068[1], 2 * tmp_8068[2]];
    signal tmp_8076[3] <== [tmp_6936[0] * 15938183887757779615, tmp_6936[1] * 15938183887757779615, tmp_6936[2] * 15938183887757779615];
    signal tmp_8077[3] <== [evals[123][0] * 11192474480021157091, evals[123][1] * 11192474480021157091, evals[123][2] * 11192474480021157091];
    signal tmp_8078[3] <== [tmp_8076[0] + tmp_8077[0], tmp_8076[1] + tmp_8077[1], tmp_8076[2] + tmp_8077[2]];
    signal tmp_8079[3] <== [evals[124][0] * 5523099006222423939, evals[124][1] * 5523099006222423939, evals[124][2] * 5523099006222423939];
    signal tmp_8080[3] <== [tmp_8078[0] + tmp_8079[0], tmp_8078[1] + tmp_8079[1], tmp_8078[2] + tmp_8079[2]];
    signal tmp_8081[3] <== [evals[54][0] * 17567284148478398281, evals[54][1] * 17567284148478398281, evals[54][2] * 17567284148478398281];
    signal tmp_8082[3] <== [tmp_8080[0] + tmp_8081[0], tmp_8080[1] + tmp_8081[1], tmp_8080[2] + tmp_8081[2]];
    signal tmp_8083[3] <== [evals[53][0] * 9530849385991780809, evals[53][1] * 9530849385991780809, evals[53][2] * 9530849385991780809];
    signal tmp_8084[3] <== [tmp_8082[0] + tmp_8083[0], tmp_8082[1] + tmp_8083[1], tmp_8082[2] + tmp_8083[2]];
    signal tmp_8085[3] <== [evals[110][0] + tmp_8084[0], evals[110][1] + tmp_8084[1], evals[110][2] + tmp_8084[2]];
    signal tmp_8087[3] <== CMul()(tmp_8085, tmp_8085);
    signal tmp_8088[3] <== CMul()(tmp_8087, tmp_8087);
    signal tmp_8089[3] <== CMul()(tmp_8088, tmp_8087);
    tmp_8091 <== CMul()(tmp_8089, tmp_8085);
    signal tmp_8094[3] <== [tmp_8091[0] + tmp_8033[0], tmp_8091[1] + tmp_8033[1], tmp_8091[2] + tmp_8033[2]];
    signal tmp_8095[3] <== [tmp_8074[0] + tmp_8094[0], tmp_8074[1] + tmp_8094[1], tmp_8074[2] + tmp_8094[2]];
    signal tmp_8096[3] <== [tmp_8071[0] + tmp_8095[0], tmp_8071[1] + tmp_8095[1], tmp_8071[2] + tmp_8095[2]];
    signal tmp_8097[3] <== [tmp_8070[0] + tmp_8096[0], tmp_8070[1] + tmp_8096[1], tmp_8070[2] + tmp_8096[2]];
    signal tmp_8098[3] <== [tmp_8016[0] + tmp_8097[0], tmp_8016[1] + tmp_8097[1], tmp_8016[2] + tmp_8097[2]];
    signal tmp_8099[3] <== [tmp_7852[0] + tmp_8098[0], tmp_7852[1] + tmp_8098[1], tmp_7852[2] + tmp_8098[2]];
    signal tmp_8100[3] <== [tmp_7771[0] - tmp_8099[0], tmp_7771[1] - tmp_8099[1], tmp_7771[2] - tmp_8099[2]];
    signal tmp_8101[3] <== CMul()(tmp_7307, tmp_8100);
    signal tmp_8102[3] <== [tmp_7763[0] + tmp_8101[0], tmp_7763[1] + tmp_8101[1], tmp_7763[2] + tmp_8101[2]];
    signal tmp_8103[3] <== CMul()(challengeQ, tmp_8102);
    signal tmp_8108[3] <== CMul()(evals[54], evals[65]);
    signal tmp_8110[3] <== CMul()(tmp_7769, evals[128]);
    signal tmp_8111[3] <== [tmp_8108[0] + tmp_8110[0], tmp_8108[1] + tmp_8110[1], tmp_8108[2] + tmp_8110[2]];
    signal tmp_8112[3] <== [tmp_7851[0] + tmp_7932[0], tmp_7851[1] + tmp_7932[1], tmp_7851[2] + tmp_7932[2]];
    signal tmp_8113[3] <== [tmp_8112[0] + tmp_8014[0], tmp_8112[1] + tmp_8014[1], tmp_8112[2] + tmp_8014[2]];
    signal tmp_8114[3] <== [tmp_8113[0] + tmp_8096[0], tmp_8113[1] + tmp_8096[1], tmp_8113[2] + tmp_8096[2]];
    signal tmp_8115[3] <== [tmp_7851[0] + tmp_8114[0], tmp_7851[1] + tmp_8114[1], tmp_7851[2] + tmp_8114[2]];
    signal tmp_8116[3] <== [tmp_8111[0] - tmp_8115[0], tmp_8111[1] - tmp_8115[1], tmp_8111[2] - tmp_8115[2]];
    signal tmp_8117[3] <== CMul()(tmp_7307, tmp_8116);
    signal tmp_8118[3] <== [tmp_8103[0] + tmp_8117[0], tmp_8103[1] + tmp_8117[1], tmp_8103[2] + tmp_8117[2]];
    signal tmp_8119[3] <== CMul()(challengeQ, tmp_8118);
    signal tmp_8124[3] <== CMul()(evals[54], evals[66]);
    signal tmp_8126[3] <== CMul()(tmp_7769, evals[129]);
    signal tmp_8127[3] <== [tmp_8124[0] + tmp_8126[0], tmp_8124[1] + tmp_8126[1], tmp_8124[2] + tmp_8126[2]];
    signal tmp_8128[3] <== [4 * tmp_7849[0], 4 * tmp_7849[1], 4 * tmp_7849[2]];
    signal tmp_8129[3] <== [tmp_8128[0] + tmp_7825[0], tmp_8128[1] + tmp_7825[1], tmp_8128[2] + tmp_7825[2]];
    signal tmp_8130[3] <== [tmp_7850[0] + tmp_8129[0], tmp_7850[1] + tmp_8129[1], tmp_7850[2] + tmp_8129[2]];
    signal tmp_8131[3] <== [4 * tmp_7930[0], 4 * tmp_7930[1], 4 * tmp_7930[2]];
    signal tmp_8132[3] <== [tmp_8131[0] + tmp_7906[0], tmp_8131[1] + tmp_7906[1], tmp_8131[2] + tmp_7906[2]];
    signal tmp_8133[3] <== [tmp_7931[0] + tmp_8132[0], tmp_7931[1] + tmp_8132[1], tmp_7931[2] + tmp_8132[2]];
    signal tmp_8134[3] <== [tmp_8130[0] + tmp_8133[0], tmp_8130[1] + tmp_8133[1], tmp_8130[2] + tmp_8133[2]];
    signal tmp_8135[3] <== [4 * tmp_8012[0], 4 * tmp_8012[1], 4 * tmp_8012[2]];
    signal tmp_8136[3] <== [tmp_8135[0] + tmp_7988[0], tmp_8135[1] + tmp_7988[1], tmp_8135[2] + tmp_7988[2]];
    signal tmp_8137[3] <== [tmp_8013[0] + tmp_8136[0], tmp_8013[1] + tmp_8136[1], tmp_8013[2] + tmp_8136[2]];
    signal tmp_8138[3] <== [tmp_8134[0] + tmp_8137[0], tmp_8134[1] + tmp_8137[1], tmp_8134[2] + tmp_8137[2]];
    signal tmp_8139[3] <== [4 * tmp_8094[0], 4 * tmp_8094[1], 4 * tmp_8094[2]];
    signal tmp_8140[3] <== [tmp_8139[0] + tmp_8070[0], tmp_8139[1] + tmp_8070[1], tmp_8139[2] + tmp_8070[2]];
    signal tmp_8141[3] <== [tmp_8095[0] + tmp_8140[0], tmp_8095[1] + tmp_8140[1], tmp_8095[2] + tmp_8140[2]];
    signal tmp_8142[3] <== [tmp_8138[0] + tmp_8141[0], tmp_8138[1] + tmp_8141[1], tmp_8138[2] + tmp_8141[2]];
    signal tmp_8143[3] <== [tmp_8130[0] + tmp_8142[0], tmp_8130[1] + tmp_8142[1], tmp_8130[2] + tmp_8142[2]];
    signal tmp_8144[3] <== [tmp_8127[0] - tmp_8143[0], tmp_8127[1] - tmp_8143[1], tmp_8127[2] - tmp_8143[2]];
    signal tmp_8145[3] <== CMul()(tmp_7307, tmp_8144);
    signal tmp_8146[3] <== [tmp_8119[0] + tmp_8145[0], tmp_8119[1] + tmp_8145[1], tmp_8119[2] + tmp_8145[2]];
    signal tmp_8147[3] <== CMul()(challengeQ, tmp_8146);
    signal tmp_8152[3] <== CMul()(evals[54], evals[67]);
    signal tmp_8154[3] <== CMul()(tmp_7769, evals[130]);
    signal tmp_8155[3] <== [tmp_8152[0] + tmp_8154[0], tmp_8152[1] + tmp_8154[1], tmp_8152[2] + tmp_8154[2]];
    signal tmp_8156[3] <== [tmp_8129[0] + tmp_8132[0], tmp_8129[1] + tmp_8132[1], tmp_8129[2] + tmp_8132[2]];
    signal tmp_8157[3] <== [tmp_8156[0] + tmp_8136[0], tmp_8156[1] + tmp_8136[1], tmp_8156[2] + tmp_8136[2]];
    signal tmp_8158[3] <== [tmp_8157[0] + tmp_8140[0], tmp_8157[1] + tmp_8140[1], tmp_8157[2] + tmp_8140[2]];
    signal tmp_8159[3] <== [tmp_8129[0] + tmp_8158[0], tmp_8129[1] + tmp_8158[1], tmp_8129[2] + tmp_8158[2]];
    signal tmp_8160[3] <== [tmp_8155[0] - tmp_8159[0], tmp_8155[1] - tmp_8159[1], tmp_8155[2] - tmp_8159[2]];
    signal tmp_8161[3] <== CMul()(tmp_7307, tmp_8160);
    signal tmp_8162[3] <== [tmp_8147[0] + tmp_8161[0], tmp_8147[1] + tmp_8161[1], tmp_8147[2] + tmp_8161[2]];
    signal tmp_8163[3] <== CMul()(challengeQ, tmp_8162);
    signal tmp_8168[3] <== CMul()(evals[54], evals[68]);
    signal tmp_8170[3] <== CMul()(tmp_7769, evals[131]);
    signal tmp_8171[3] <== [tmp_8168[0] + tmp_8170[0], tmp_8168[1] + tmp_8170[1], tmp_8168[2] + tmp_8170[2]];
    signal tmp_8172[3] <== [tmp_7933[0] + tmp_8098[0], tmp_7933[1] + tmp_8098[1], tmp_7933[2] + tmp_8098[2]];
    signal tmp_8173[3] <== [tmp_8171[0] - tmp_8172[0], tmp_8171[1] - tmp_8172[1], tmp_8171[2] - tmp_8172[2]];
    signal tmp_8174[3] <== CMul()(tmp_7307, tmp_8173);
    signal tmp_8175[3] <== [tmp_8163[0] + tmp_8174[0], tmp_8163[1] + tmp_8174[1], tmp_8163[2] + tmp_8174[2]];
    signal tmp_8176[3] <== CMul()(challengeQ, tmp_8175);
    signal tmp_8181[3] <== CMul()(evals[54], evals[69]);
    signal tmp_8183[3] <== CMul()(tmp_7769, evals[132]);
    signal tmp_8184[3] <== [tmp_8181[0] + tmp_8183[0], tmp_8181[1] + tmp_8183[1], tmp_8181[2] + tmp_8183[2]];
    signal tmp_8185[3] <== [tmp_7932[0] + tmp_8114[0], tmp_7932[1] + tmp_8114[1], tmp_7932[2] + tmp_8114[2]];
    signal tmp_8186[3] <== [tmp_8184[0] - tmp_8185[0], tmp_8184[1] - tmp_8185[1], tmp_8184[2] - tmp_8185[2]];
    signal tmp_8187[3] <== CMul()(tmp_7307, tmp_8186);
    signal tmp_8188[3] <== [tmp_8176[0] + tmp_8187[0], tmp_8176[1] + tmp_8187[1], tmp_8176[2] + tmp_8187[2]];
    signal tmp_8189[3] <== CMul()(challengeQ, tmp_8188);
    signal tmp_8194[3] <== CMul()(evals[54], evals[70]);
    signal tmp_8196[3] <== CMul()(tmp_7769, evals[133]);
    signal tmp_8197[3] <== [tmp_8194[0] + tmp_8196[0], tmp_8194[1] + tmp_8196[1], tmp_8194[2] + tmp_8196[2]];
    signal tmp_8198[3] <== [tmp_8133[0] + tmp_8142[0], tmp_8133[1] + tmp_8142[1], tmp_8133[2] + tmp_8142[2]];
    signal tmp_8199[3] <== [tmp_8197[0] - tmp_8198[0], tmp_8197[1] - tmp_8198[1], tmp_8197[2] - tmp_8198[2]];
    signal tmp_8200[3] <== CMul()(tmp_7307, tmp_8199);
    signal tmp_8201[3] <== [tmp_8189[0] + tmp_8200[0], tmp_8189[1] + tmp_8200[1], tmp_8189[2] + tmp_8200[2]];
    signal tmp_8202[3] <== CMul()(challengeQ, tmp_8201);
    signal tmp_8207[3] <== CMul()(evals[54], evals[71]);
    signal tmp_8209[3] <== CMul()(tmp_7769, evals[134]);
    signal tmp_8210[3] <== [tmp_8207[0] + tmp_8209[0], tmp_8207[1] + tmp_8209[1], tmp_8207[2] + tmp_8209[2]];
    signal tmp_8211[3] <== [tmp_8132[0] + tmp_8158[0], tmp_8132[1] + tmp_8158[1], tmp_8132[2] + tmp_8158[2]];
    signal tmp_8212[3] <== [tmp_8210[0] - tmp_8211[0], tmp_8210[1] - tmp_8211[1], tmp_8210[2] - tmp_8211[2]];
    signal tmp_8213[3] <== CMul()(tmp_7307, tmp_8212);
    signal tmp_8214[3] <== [tmp_8202[0] + tmp_8213[0], tmp_8202[1] + tmp_8213[1], tmp_8202[2] + tmp_8213[2]];
    signal tmp_8215[3] <== CMul()(challengeQ, tmp_8214);
    signal tmp_8220[3] <== CMul()(evals[54], evals[72]);
    signal tmp_8222[3] <== CMul()(tmp_7769, evals[135]);
    signal tmp_8223[3] <== [tmp_8220[0] + tmp_8222[0], tmp_8220[1] + tmp_8222[1], tmp_8220[2] + tmp_8222[2]];
    signal tmp_8224[3] <== [tmp_8015[0] + tmp_8098[0], tmp_8015[1] + tmp_8098[1], tmp_8015[2] + tmp_8098[2]];
    signal tmp_8225[3] <== [tmp_8223[0] - tmp_8224[0], tmp_8223[1] - tmp_8224[1], tmp_8223[2] - tmp_8224[2]];
    signal tmp_8226[3] <== CMul()(tmp_7307, tmp_8225);
    signal tmp_8227[3] <== [tmp_8215[0] + tmp_8226[0], tmp_8215[1] + tmp_8226[1], tmp_8215[2] + tmp_8226[2]];
    signal tmp_8228[3] <== CMul()(challengeQ, tmp_8227);
    signal tmp_8233[3] <== CMul()(evals[54], evals[73]);
    signal tmp_8235[3] <== CMul()(tmp_7769, evals[136]);
    signal tmp_8236[3] <== [tmp_8233[0] + tmp_8235[0], tmp_8233[1] + tmp_8235[1], tmp_8233[2] + tmp_8235[2]];
    signal tmp_8237[3] <== [tmp_8014[0] + tmp_8114[0], tmp_8014[1] + tmp_8114[1], tmp_8014[2] + tmp_8114[2]];
    signal tmp_8238[3] <== [tmp_8236[0] - tmp_8237[0], tmp_8236[1] - tmp_8237[1], tmp_8236[2] - tmp_8237[2]];
    signal tmp_8239[3] <== CMul()(tmp_7307, tmp_8238);
    signal tmp_8240[3] <== [tmp_8228[0] + tmp_8239[0], tmp_8228[1] + tmp_8239[1], tmp_8228[2] + tmp_8239[2]];
    signal tmp_8241[3] <== CMul()(challengeQ, tmp_8240);
    signal tmp_8246[3] <== CMul()(evals[54], evals[74]);
    signal tmp_8248[3] <== CMul()(tmp_7769, evals[137]);
    signal tmp_8249[3] <== [tmp_8246[0] + tmp_8248[0], tmp_8246[1] + tmp_8248[1], tmp_8246[2] + tmp_8248[2]];
    signal tmp_8250[3] <== [tmp_8137[0] + tmp_8142[0], tmp_8137[1] + tmp_8142[1], tmp_8137[2] + tmp_8142[2]];
    signal tmp_8251[3] <== [tmp_8249[0] - tmp_8250[0], tmp_8249[1] - tmp_8250[1], tmp_8249[2] - tmp_8250[2]];
    signal tmp_8252[3] <== CMul()(tmp_7307, tmp_8251);
    signal tmp_8253[3] <== [tmp_8241[0] + tmp_8252[0], tmp_8241[1] + tmp_8252[1], tmp_8241[2] + tmp_8252[2]];
    signal tmp_8254[3] <== CMul()(challengeQ, tmp_8253);
    signal tmp_8259[3] <== CMul()(evals[54], evals[75]);
    signal tmp_8261[3] <== CMul()(tmp_7769, evals[138]);
    signal tmp_8262[3] <== [tmp_8259[0] + tmp_8261[0], tmp_8259[1] + tmp_8261[1], tmp_8259[2] + tmp_8261[2]];
    signal tmp_8263[3] <== [tmp_8136[0] + tmp_8158[0], tmp_8136[1] + tmp_8158[1], tmp_8136[2] + tmp_8158[2]];
    signal tmp_8264[3] <== [tmp_8262[0] - tmp_8263[0], tmp_8262[1] - tmp_8263[1], tmp_8262[2] - tmp_8263[2]];
    signal tmp_8265[3] <== CMul()(tmp_7307, tmp_8264);
    signal tmp_8266[3] <== [tmp_8254[0] + tmp_8265[0], tmp_8254[1] + tmp_8265[1], tmp_8254[2] + tmp_8265[2]];
    signal tmp_8267[3] <== CMul()(challengeQ, tmp_8266);
    signal tmp_8272[3] <== CMul()(evals[54], evals[76]);
    signal tmp_8274[3] <== CMul()(tmp_7769, evals[139]);
    signal tmp_8275[3] <== [tmp_8272[0] + tmp_8274[0], tmp_8272[1] + tmp_8274[1], tmp_8272[2] + tmp_8274[2]];
    signal tmp_8276[3] <== [tmp_8097[0] + tmp_8098[0], tmp_8097[1] + tmp_8098[1], tmp_8097[2] + tmp_8098[2]];
    signal tmp_8277[3] <== [tmp_8275[0] - tmp_8276[0], tmp_8275[1] - tmp_8276[1], tmp_8275[2] - tmp_8276[2]];
    signal tmp_8278[3] <== CMul()(tmp_7307, tmp_8277);
    signal tmp_8279[3] <== [tmp_8267[0] + tmp_8278[0], tmp_8267[1] + tmp_8278[1], tmp_8267[2] + tmp_8278[2]];
    signal tmp_8280[3] <== CMul()(challengeQ, tmp_8279);
    signal tmp_8285[3] <== CMul()(evals[54], evals[77]);
    signal tmp_8287[3] <== CMul()(tmp_7769, evals[140]);
    signal tmp_8288[3] <== [tmp_8285[0] + tmp_8287[0], tmp_8285[1] + tmp_8287[1], tmp_8285[2] + tmp_8287[2]];
    signal tmp_8289[3] <== [tmp_8096[0] + tmp_8114[0], tmp_8096[1] + tmp_8114[1], tmp_8096[2] + tmp_8114[2]];
    signal tmp_8290[3] <== [tmp_8288[0] - tmp_8289[0], tmp_8288[1] - tmp_8289[1], tmp_8288[2] - tmp_8289[2]];
    signal tmp_8291[3] <== CMul()(tmp_7307, tmp_8290);
    signal tmp_8292[3] <== [tmp_8280[0] + tmp_8291[0], tmp_8280[1] + tmp_8291[1], tmp_8280[2] + tmp_8291[2]];
    signal tmp_8293[3] <== CMul()(challengeQ, tmp_8292);
    signal tmp_8298[3] <== CMul()(evals[54], evals[78]);
    signal tmp_8300[3] <== CMul()(tmp_7769, evals[141]);
    signal tmp_8301[3] <== [tmp_8298[0] + tmp_8300[0], tmp_8298[1] + tmp_8300[1], tmp_8298[2] + tmp_8300[2]];
    signal tmp_8302[3] <== [tmp_8141[0] + tmp_8142[0], tmp_8141[1] + tmp_8142[1], tmp_8141[2] + tmp_8142[2]];
    signal tmp_8303[3] <== [tmp_8301[0] - tmp_8302[0], tmp_8301[1] - tmp_8302[1], tmp_8301[2] - tmp_8302[2]];
    signal tmp_8304[3] <== CMul()(tmp_7307, tmp_8303);
    signal tmp_8305[3] <== [tmp_8293[0] + tmp_8304[0], tmp_8293[1] + tmp_8304[1], tmp_8293[2] + tmp_8304[2]];
    signal tmp_8306[3] <== CMul()(challengeQ, tmp_8305);
    signal tmp_8311[3] <== CMul()(evals[54], evals[79]);
    signal tmp_8313[3] <== CMul()(tmp_7769, evals[142]);
    signal tmp_8314[3] <== [tmp_8311[0] + tmp_8313[0], tmp_8311[1] + tmp_8313[1], tmp_8311[2] + tmp_8313[2]];
    signal tmp_8315[3] <== [tmp_8140[0] + tmp_8158[0], tmp_8140[1] + tmp_8158[1], tmp_8140[2] + tmp_8158[2]];
    signal tmp_8316[3] <== [tmp_8314[0] - tmp_8315[0], tmp_8314[1] - tmp_8315[1], tmp_8314[2] - tmp_8315[2]];
    signal tmp_8317[3] <== CMul()(tmp_7307, tmp_8316);
    signal tmp_8318[3] <== [tmp_8306[0] + tmp_8317[0], tmp_8306[1] + tmp_8317[1], tmp_8306[2] + tmp_8317[2]];
    signal tmp_8319[3] <== CMul()(challengeQ, tmp_8318);
    signal tmp_8320[3] <== [evals[96][0] - evals[80][0], evals[96][1] - evals[80][1], evals[96][2] - evals[80][2]];
    signal tmp_8321[3] <== CMul()(evals[53], tmp_8320);
    signal tmp_8322[3] <== [tmp_8319[0] + tmp_8321[0], tmp_8319[1] + tmp_8321[1], tmp_8319[2] + tmp_8321[2]];
    signal tmp_8323[3] <== CMul()(challengeQ, tmp_8322);
    signal tmp_8326[3] <== [tmp_7806[0] * 16040574633112940480, tmp_7806[1] * 16040574633112940480, tmp_7806[2] * 16040574633112940480];
    signal tmp_8327[3] <== [tmp_7806[0] + evals[81][0], tmp_7806[1] + evals[81][1], tmp_7806[2] + evals[81][2]];
    signal tmp_8328[3] <== [tmp_8327[0] + evals[82][0], tmp_8327[1] + evals[82][1], tmp_8327[2] + evals[82][2]];
    signal tmp_8329[3] <== [tmp_8328[0] + evals[83][0], tmp_8328[1] + evals[83][1], tmp_8328[2] + evals[83][2]];
    signal tmp_8330[3] <== [tmp_8329[0] + evals[84][0], tmp_8329[1] + evals[84][1], tmp_8329[2] + evals[84][2]];
    signal tmp_8331[3] <== [tmp_8330[0] + evals[85][0], tmp_8330[1] + evals[85][1], tmp_8330[2] + evals[85][2]];
    signal tmp_8332[3] <== [tmp_8331[0] + evals[86][0], tmp_8331[1] + evals[86][1], tmp_8331[2] + evals[86][2]];
    signal tmp_8333[3] <== [tmp_8332[0] + evals[87][0], tmp_8332[1] + evals[87][1], tmp_8332[2] + evals[87][2]];
    signal tmp_8334[3] <== [tmp_8333[0] + evals[88][0], tmp_8333[1] + evals[88][1], tmp_8333[2] + evals[88][2]];
    signal tmp_8335[3] <== [tmp_8334[0] + evals[89][0], tmp_8334[1] + evals[89][1], tmp_8334[2] + evals[89][2]];
    signal tmp_8336[3] <== [tmp_8335[0] + evals[90][0], tmp_8335[1] + evals[90][1], tmp_8335[2] + evals[90][2]];
    signal tmp_8337[3] <== [tmp_8336[0] + evals[91][0], tmp_8336[1] + evals[91][1], tmp_8336[2] + evals[91][2]];
    signal tmp_8338[3] <== [tmp_8337[0] + evals[92][0], tmp_8337[1] + evals[92][1], tmp_8337[2] + evals[92][2]];
    signal tmp_8339[3] <== [tmp_8338[0] + evals[93][0], tmp_8338[1] + evals[93][1], tmp_8338[2] + evals[93][2]];
    signal tmp_8340[3] <== [tmp_8339[0] + evals[94][0], tmp_8339[1] + evals[94][1], tmp_8339[2] + evals[94][2]];
    signal tmp_8341[3] <== [tmp_8340[0] + evals[95][0], tmp_8340[1] + evals[95][1], tmp_8340[2] + evals[95][2]];
    signal tmp_8342[3] <== [tmp_8326[0] + tmp_8341[0], tmp_8326[1] + tmp_8341[1], tmp_8326[2] + tmp_8341[2]];
    signal tmp_8343[3] <== [evals[97][0] - tmp_8342[0], evals[97][1] - tmp_8342[1], evals[97][2] - tmp_8342[2]];
    signal tmp_8344[3] <== CMul()(evals[53], tmp_8343);
    signal tmp_8345[3] <== [tmp_8323[0] + tmp_8344[0], tmp_8323[1] + tmp_8344[1], tmp_8323[2] + tmp_8344[2]];
    signal tmp_8346[3] <== CMul()(challengeQ, tmp_8345);
    signal tmp_8349[3] <== [tmp_7823[0] * 16040574633112940480, tmp_7823[1] * 16040574633112940480, tmp_7823[2] * 16040574633112940480];
    signal tmp_8350[3] <== [evals[81][0] * 14263299814608977431, evals[81][1] * 14263299814608977431, evals[81][2] * 14263299814608977431];
    signal tmp_8351[3] <== [tmp_8350[0] + tmp_8341[0], tmp_8350[1] + tmp_8341[1], tmp_8350[2] + tmp_8341[2]];
    signal tmp_8352[3] <== [tmp_7823[0] + tmp_8351[0], tmp_7823[1] + tmp_8351[1], tmp_7823[2] + tmp_8351[2]];
    signal tmp_8353[3] <== [evals[82][0] * 770395855193680981, evals[82][1] * 770395855193680981, evals[82][2] * 770395855193680981];
    signal tmp_8354[3] <== [tmp_8353[0] + tmp_8341[0], tmp_8353[1] + tmp_8341[1], tmp_8353[2] + tmp_8341[2]];
    signal tmp_8355[3] <== [tmp_8352[0] + tmp_8354[0], tmp_8352[1] + tmp_8354[1], tmp_8352[2] + tmp_8354[2]];
    signal tmp_8356[3] <== [evals[83][0] * 3459277367440070515, evals[83][1] * 3459277367440070515, evals[83][2] * 3459277367440070515];
    signal tmp_8357[3] <== [tmp_8356[0] + tmp_8341[0], tmp_8356[1] + tmp_8341[1], tmp_8356[2] + tmp_8341[2]];
    signal tmp_8358[3] <== [tmp_8355[0] + tmp_8357[0], tmp_8355[1] + tmp_8357[1], tmp_8355[2] + tmp_8357[2]];
    signal tmp_8359[3] <== [evals[84][0] * 17087697094293314027, evals[84][1] * 17087697094293314027, evals[84][2] * 17087697094293314027];
    signal tmp_8360[3] <== [tmp_8359[0] + tmp_8341[0], tmp_8359[1] + tmp_8341[1], tmp_8359[2] + tmp_8341[2]];
    signal tmp_8361[3] <== [tmp_8358[0] + tmp_8360[0], tmp_8358[1] + tmp_8360[1], tmp_8358[2] + tmp_8360[2]];
    signal tmp_8362[3] <== [evals[85][0] * 6694380135428747348, evals[85][1] * 6694380135428747348, evals[85][2] * 6694380135428747348];
    signal tmp_8363[3] <== [tmp_8362[0] + tmp_8341[0], tmp_8362[1] + tmp_8341[1], tmp_8362[2] + tmp_8341[2]];
    signal tmp_8364[3] <== [tmp_8361[0] + tmp_8363[0], tmp_8361[1] + tmp_8363[1], tmp_8361[2] + tmp_8363[2]];
    signal tmp_8365[3] <== [evals[86][0] * 2034408310088972836, evals[86][1] * 2034408310088972836, evals[86][2] * 2034408310088972836];
    signal tmp_8366[3] <== [tmp_8365[0] + tmp_8341[0], tmp_8365[1] + tmp_8341[1], tmp_8365[2] + tmp_8341[2]];
    signal tmp_8367[3] <== [tmp_8364[0] + tmp_8366[0], tmp_8364[1] + tmp_8366[1], tmp_8364[2] + tmp_8366[2]];
    signal tmp_8368[3] <== [evals[87][0] * 3434575637390274478, evals[87][1] * 3434575637390274478, evals[87][2] * 3434575637390274478];
    signal tmp_8369[3] <== [tmp_8368[0] + tmp_8341[0], tmp_8368[1] + tmp_8341[1], tmp_8368[2] + tmp_8341[2]];
    signal tmp_8370[3] <== [tmp_8367[0] + tmp_8369[0], tmp_8367[1] + tmp_8369[1], tmp_8367[2] + tmp_8369[2]];
    signal tmp_8371[3] <== [evals[88][0] * 6052753985947965968, evals[88][1] * 6052753985947965968, evals[88][2] * 6052753985947965968];
    signal tmp_8372[3] <== [tmp_8371[0] + tmp_8341[0], tmp_8371[1] + tmp_8341[1], tmp_8371[2] + tmp_8341[2]];
    signal tmp_8373[3] <== [tmp_8370[0] + tmp_8372[0], tmp_8370[1] + tmp_8372[1], tmp_8370[2] + tmp_8372[2]];
    signal tmp_8374[3] <== [evals[89][0] * 13608362914817483670, evals[89][1] * 13608362914817483670, evals[89][2] * 13608362914817483670];
    signal tmp_8375[3] <== [tmp_8374[0] + tmp_8341[0], tmp_8374[1] + tmp_8341[1], tmp_8374[2] + tmp_8341[2]];
    signal tmp_8376[3] <== [tmp_8373[0] + tmp_8375[0], tmp_8373[1] + tmp_8375[1], tmp_8373[2] + tmp_8375[2]];
    signal tmp_8377[3] <== [evals[90][0] * 18163707672964630459, evals[90][1] * 18163707672964630459, evals[90][2] * 18163707672964630459];
    signal tmp_8378[3] <== [tmp_8377[0] + tmp_8341[0], tmp_8377[1] + tmp_8341[1], tmp_8377[2] + tmp_8341[2]];
    signal tmp_8379[3] <== [tmp_8376[0] + tmp_8378[0], tmp_8376[1] + tmp_8378[1], tmp_8376[2] + tmp_8378[2]];
    signal tmp_8380[3] <== [evals[91][0] * 14373610220374016704, evals[91][1] * 14373610220374016704, evals[91][2] * 14373610220374016704];
    signal tmp_8381[3] <== [tmp_8380[0] + tmp_8341[0], tmp_8380[1] + tmp_8341[1], tmp_8380[2] + tmp_8341[2]];
    signal tmp_8382[3] <== [tmp_8379[0] + tmp_8381[0], tmp_8379[1] + tmp_8381[1], tmp_8379[2] + tmp_8381[2]];
    signal tmp_8383[3] <== [evals[92][0] * 6226282807566121054, evals[92][1] * 6226282807566121054, evals[92][2] * 6226282807566121054];
    signal tmp_8384[3] <== [tmp_8383[0] + tmp_8341[0], tmp_8383[1] + tmp_8341[1], tmp_8383[2] + tmp_8341[2]];
    signal tmp_8385[3] <== [tmp_8382[0] + tmp_8384[0], tmp_8382[1] + tmp_8384[1], tmp_8382[2] + tmp_8384[2]];
    signal tmp_8386[3] <== [evals[93][0] * 3643354756180461803, evals[93][1] * 3643354756180461803, evals[93][2] * 3643354756180461803];
    signal tmp_8387[3] <== [tmp_8386[0] + tmp_8341[0], tmp_8386[1] + tmp_8341[1], tmp_8386[2] + tmp_8341[2]];
    signal tmp_8388[3] <== [tmp_8385[0] + tmp_8387[0], tmp_8385[1] + tmp_8387[1], tmp_8385[2] + tmp_8387[2]];
    signal tmp_8389[3] <== [evals[94][0] * 13046961313070095543, evals[94][1] * 13046961313070095543, evals[94][2] * 13046961313070095543];
    signal tmp_8390[3] <== [tmp_8389[0] + tmp_8341[0], tmp_8389[1] + tmp_8341[1], tmp_8389[2] + tmp_8341[2]];
    signal tmp_8391[3] <== [tmp_8388[0] + tmp_8390[0], tmp_8388[1] + tmp_8390[1], tmp_8388[2] + tmp_8390[2]];
    signal tmp_8392[3] <== [evals[95][0] * 8594143216561850811, evals[95][1] * 8594143216561850811, evals[95][2] * 8594143216561850811];
    signal tmp_8393[3] <== [tmp_8392[0] + tmp_8341[0], tmp_8392[1] + tmp_8341[1], tmp_8392[2] + tmp_8341[2]];
    signal tmp_8394[3] <== [tmp_8391[0] + tmp_8393[0], tmp_8391[1] + tmp_8393[1], tmp_8391[2] + tmp_8393[2]];
    signal tmp_8395[3] <== [tmp_8349[0] + tmp_8394[0], tmp_8349[1] + tmp_8394[1], tmp_8349[2] + tmp_8394[2]];
    signal tmp_8396[3] <== [evals[98][0] - tmp_8395[0], evals[98][1] - tmp_8395[1], evals[98][2] - tmp_8395[2]];
    signal tmp_8397[3] <== CMul()(evals[53], tmp_8396);
    signal tmp_8398[3] <== [tmp_8346[0] + tmp_8397[0], tmp_8346[1] + tmp_8397[1], tmp_8346[2] + tmp_8397[2]];
    signal tmp_8399[3] <== CMul()(challengeQ, tmp_8398);
    signal tmp_8402[3] <== [tmp_7846[0] * 16040574633112940480, tmp_7846[1] * 16040574633112940480, tmp_7846[2] * 16040574633112940480];
    signal tmp_8403[3] <== [tmp_8351[0] * 14263299814608977431, tmp_8351[1] * 14263299814608977431, tmp_8351[2] * 14263299814608977431];
    signal tmp_8404[3] <== [tmp_8403[0] + tmp_8394[0], tmp_8403[1] + tmp_8394[1], tmp_8403[2] + tmp_8394[2]];
    signal tmp_8405[3] <== [tmp_7846[0] + tmp_8404[0], tmp_7846[1] + tmp_8404[1], tmp_7846[2] + tmp_8404[2]];
    signal tmp_8406[3] <== [tmp_8354[0] * 770395855193680981, tmp_8354[1] * 770395855193680981, tmp_8354[2] * 770395855193680981];
    signal tmp_8407[3] <== [tmp_8406[0] + tmp_8394[0], tmp_8406[1] + tmp_8394[1], tmp_8406[2] + tmp_8394[2]];
    signal tmp_8408[3] <== [tmp_8405[0] + tmp_8407[0], tmp_8405[1] + tmp_8407[1], tmp_8405[2] + tmp_8407[2]];
    signal tmp_8409[3] <== [tmp_8357[0] * 3459277367440070515, tmp_8357[1] * 3459277367440070515, tmp_8357[2] * 3459277367440070515];
    signal tmp_8410[3] <== [tmp_8409[0] + tmp_8394[0], tmp_8409[1] + tmp_8394[1], tmp_8409[2] + tmp_8394[2]];
    signal tmp_8411[3] <== [tmp_8408[0] + tmp_8410[0], tmp_8408[1] + tmp_8410[1], tmp_8408[2] + tmp_8410[2]];
    signal tmp_8412[3] <== [tmp_8360[0] * 17087697094293314027, tmp_8360[1] * 17087697094293314027, tmp_8360[2] * 17087697094293314027];
    signal tmp_8413[3] <== [tmp_8412[0] + tmp_8394[0], tmp_8412[1] + tmp_8394[1], tmp_8412[2] + tmp_8394[2]];
    signal tmp_8414[3] <== [tmp_8411[0] + tmp_8413[0], tmp_8411[1] + tmp_8413[1], tmp_8411[2] + tmp_8413[2]];
    signal tmp_8415[3] <== [tmp_8363[0] * 6694380135428747348, tmp_8363[1] * 6694380135428747348, tmp_8363[2] * 6694380135428747348];
    signal tmp_8416[3] <== [tmp_8415[0] + tmp_8394[0], tmp_8415[1] + tmp_8394[1], tmp_8415[2] + tmp_8394[2]];
    signal tmp_8417[3] <== [tmp_8414[0] + tmp_8416[0], tmp_8414[1] + tmp_8416[1], tmp_8414[2] + tmp_8416[2]];
    signal tmp_8418[3] <== [tmp_8366[0] * 2034408310088972836, tmp_8366[1] * 2034408310088972836, tmp_8366[2] * 2034408310088972836];
    signal tmp_8419[3] <== [tmp_8418[0] + tmp_8394[0], tmp_8418[1] + tmp_8394[1], tmp_8418[2] + tmp_8394[2]];
    signal tmp_8420[3] <== [tmp_8417[0] + tmp_8419[0], tmp_8417[1] + tmp_8419[1], tmp_8417[2] + tmp_8419[2]];
    signal tmp_8421[3] <== [tmp_8369[0] * 3434575637390274478, tmp_8369[1] * 3434575637390274478, tmp_8369[2] * 3434575637390274478];
    signal tmp_8422[3] <== [tmp_8421[0] + tmp_8394[0], tmp_8421[1] + tmp_8394[1], tmp_8421[2] + tmp_8394[2]];
    signal tmp_8423[3] <== [tmp_8420[0] + tmp_8422[0], tmp_8420[1] + tmp_8422[1], tmp_8420[2] + tmp_8422[2]];
    signal tmp_8424[3] <== [tmp_8372[0] * 6052753985947965968, tmp_8372[1] * 6052753985947965968, tmp_8372[2] * 6052753985947965968];
    signal tmp_8425[3] <== [tmp_8424[0] + tmp_8394[0], tmp_8424[1] + tmp_8394[1], tmp_8424[2] + tmp_8394[2]];
    signal tmp_8426[3] <== [tmp_8423[0] + tmp_8425[0], tmp_8423[1] + tmp_8425[1], tmp_8423[2] + tmp_8425[2]];
    signal tmp_8427[3] <== [tmp_8375[0] * 13608362914817483670, tmp_8375[1] * 13608362914817483670, tmp_8375[2] * 13608362914817483670];
    signal tmp_8428[3] <== [tmp_8427[0] + tmp_8394[0], tmp_8427[1] + tmp_8394[1], tmp_8427[2] + tmp_8394[2]];
    signal tmp_8429[3] <== [tmp_8426[0] + tmp_8428[0], tmp_8426[1] + tmp_8428[1], tmp_8426[2] + tmp_8428[2]];
    signal tmp_8430[3] <== [tmp_8378[0] * 18163707672964630459, tmp_8378[1] * 18163707672964630459, tmp_8378[2] * 18163707672964630459];
    signal tmp_8431[3] <== [tmp_8430[0] + tmp_8394[0], tmp_8430[1] + tmp_8394[1], tmp_8430[2] + tmp_8394[2]];
    signal tmp_8432[3] <== [tmp_8429[0] + tmp_8431[0], tmp_8429[1] + tmp_8431[1], tmp_8429[2] + tmp_8431[2]];
    signal tmp_8433[3] <== [tmp_8381[0] * 14373610220374016704, tmp_8381[1] * 14373610220374016704, tmp_8381[2] * 14373610220374016704];
    signal tmp_8434[3] <== [tmp_8433[0] + tmp_8394[0], tmp_8433[1] + tmp_8394[1], tmp_8433[2] + tmp_8394[2]];
    signal tmp_8435[3] <== [tmp_8432[0] + tmp_8434[0], tmp_8432[1] + tmp_8434[1], tmp_8432[2] + tmp_8434[2]];
    signal tmp_8436[3] <== [tmp_8384[0] * 6226282807566121054, tmp_8384[1] * 6226282807566121054, tmp_8384[2] * 6226282807566121054];
    signal tmp_8437[3] <== [tmp_8436[0] + tmp_8394[0], tmp_8436[1] + tmp_8394[1], tmp_8436[2] + tmp_8394[2]];
    signal tmp_8438[3] <== [tmp_8435[0] + tmp_8437[0], tmp_8435[1] + tmp_8437[1], tmp_8435[2] + tmp_8437[2]];
    signal tmp_8439[3] <== [tmp_8387[0] * 3643354756180461803, tmp_8387[1] * 3643354756180461803, tmp_8387[2] * 3643354756180461803];
    signal tmp_8440[3] <== [tmp_8439[0] + tmp_8394[0], tmp_8439[1] + tmp_8394[1], tmp_8439[2] + tmp_8394[2]];
    signal tmp_8441[3] <== [tmp_8438[0] + tmp_8440[0], tmp_8438[1] + tmp_8440[1], tmp_8438[2] + tmp_8440[2]];
    signal tmp_8442[3] <== [tmp_8390[0] * 13046961313070095543, tmp_8390[1] * 13046961313070095543, tmp_8390[2] * 13046961313070095543];
    signal tmp_8443[3] <== [tmp_8442[0] + tmp_8394[0], tmp_8442[1] + tmp_8394[1], tmp_8442[2] + tmp_8394[2]];
    signal tmp_8444[3] <== [tmp_8441[0] + tmp_8443[0], tmp_8441[1] + tmp_8443[1], tmp_8441[2] + tmp_8443[2]];
    signal tmp_8445[3] <== [tmp_8393[0] * 8594143216561850811, tmp_8393[1] * 8594143216561850811, tmp_8393[2] * 8594143216561850811];
    signal tmp_8446[3] <== [tmp_8445[0] + tmp_8394[0], tmp_8445[1] + tmp_8394[1], tmp_8445[2] + tmp_8394[2]];
    signal tmp_8447[3] <== [tmp_8444[0] + tmp_8446[0], tmp_8444[1] + tmp_8446[1], tmp_8444[2] + tmp_8446[2]];
    signal tmp_8448[3] <== [tmp_8402[0] + tmp_8447[0], tmp_8402[1] + tmp_8447[1], tmp_8402[2] + tmp_8447[2]];
    signal tmp_8449[3] <== [evals[99][0] - tmp_8448[0], evals[99][1] - tmp_8448[1], evals[99][2] - tmp_8448[2]];
    signal tmp_8450[3] <== CMul()(evals[53], tmp_8449);
    signal tmp_8451[3] <== [tmp_8399[0] + tmp_8450[0], tmp_8399[1] + tmp_8450[1], tmp_8399[2] + tmp_8450[2]];
    signal tmp_8452[3] <== CMul()(challengeQ, tmp_8451);
    signal tmp_8455[3] <== [tmp_7788[0] * 16040574633112940480, tmp_7788[1] * 16040574633112940480, tmp_7788[2] * 16040574633112940480];
    signal tmp_8456[3] <== [tmp_8404[0] * 14263299814608977431, tmp_8404[1] * 14263299814608977431, tmp_8404[2] * 14263299814608977431];
    signal tmp_8457[3] <== [tmp_8456[0] + tmp_8447[0], tmp_8456[1] + tmp_8447[1], tmp_8456[2] + tmp_8447[2]];
    signal tmp_8458[3] <== [tmp_7788[0] + tmp_8457[0], tmp_7788[1] + tmp_8457[1], tmp_7788[2] + tmp_8457[2]];
    signal tmp_8459[3] <== [tmp_8407[0] * 770395855193680981, tmp_8407[1] * 770395855193680981, tmp_8407[2] * 770395855193680981];
    signal tmp_8460[3] <== [tmp_8459[0] + tmp_8447[0], tmp_8459[1] + tmp_8447[1], tmp_8459[2] + tmp_8447[2]];
    signal tmp_8461[3] <== [tmp_8458[0] + tmp_8460[0], tmp_8458[1] + tmp_8460[1], tmp_8458[2] + tmp_8460[2]];
    signal tmp_8462[3] <== [tmp_8410[0] * 3459277367440070515, tmp_8410[1] * 3459277367440070515, tmp_8410[2] * 3459277367440070515];
    signal tmp_8463[3] <== [tmp_8462[0] + tmp_8447[0], tmp_8462[1] + tmp_8447[1], tmp_8462[2] + tmp_8447[2]];
    signal tmp_8464[3] <== [tmp_8461[0] + tmp_8463[0], tmp_8461[1] + tmp_8463[1], tmp_8461[2] + tmp_8463[2]];
    signal tmp_8465[3] <== [tmp_8413[0] * 17087697094293314027, tmp_8413[1] * 17087697094293314027, tmp_8413[2] * 17087697094293314027];
    signal tmp_8466[3] <== [tmp_8465[0] + tmp_8447[0], tmp_8465[1] + tmp_8447[1], tmp_8465[2] + tmp_8447[2]];
    signal tmp_8467[3] <== [tmp_8464[0] + tmp_8466[0], tmp_8464[1] + tmp_8466[1], tmp_8464[2] + tmp_8466[2]];
    signal tmp_8468[3] <== [tmp_8416[0] * 6694380135428747348, tmp_8416[1] * 6694380135428747348, tmp_8416[2] * 6694380135428747348];
    signal tmp_8469[3] <== [tmp_8468[0] + tmp_8447[0], tmp_8468[1] + tmp_8447[1], tmp_8468[2] + tmp_8447[2]];
    signal tmp_8470[3] <== [tmp_8467[0] + tmp_8469[0], tmp_8467[1] + tmp_8469[1], tmp_8467[2] + tmp_8469[2]];
    signal tmp_8471[3] <== [tmp_8419[0] * 2034408310088972836, tmp_8419[1] * 2034408310088972836, tmp_8419[2] * 2034408310088972836];
    signal tmp_8472[3] <== [tmp_8471[0] + tmp_8447[0], tmp_8471[1] + tmp_8447[1], tmp_8471[2] + tmp_8447[2]];
    signal tmp_8473[3] <== [tmp_8470[0] + tmp_8472[0], tmp_8470[1] + tmp_8472[1], tmp_8470[2] + tmp_8472[2]];
    signal tmp_8474[3] <== [tmp_8422[0] * 3434575637390274478, tmp_8422[1] * 3434575637390274478, tmp_8422[2] * 3434575637390274478];
    signal tmp_8475[3] <== [tmp_8474[0] + tmp_8447[0], tmp_8474[1] + tmp_8447[1], tmp_8474[2] + tmp_8447[2]];
    signal tmp_8476[3] <== [tmp_8473[0] + tmp_8475[0], tmp_8473[1] + tmp_8475[1], tmp_8473[2] + tmp_8475[2]];
    signal tmp_8477[3] <== [tmp_8425[0] * 6052753985947965968, tmp_8425[1] * 6052753985947965968, tmp_8425[2] * 6052753985947965968];
    signal tmp_8478[3] <== [tmp_8477[0] + tmp_8447[0], tmp_8477[1] + tmp_8447[1], tmp_8477[2] + tmp_8447[2]];
    signal tmp_8479[3] <== [tmp_8476[0] + tmp_8478[0], tmp_8476[1] + tmp_8478[1], tmp_8476[2] + tmp_8478[2]];
    signal tmp_8480[3] <== [tmp_8428[0] * 13608362914817483670, tmp_8428[1] * 13608362914817483670, tmp_8428[2] * 13608362914817483670];
    signal tmp_8481[3] <== [tmp_8480[0] + tmp_8447[0], tmp_8480[1] + tmp_8447[1], tmp_8480[2] + tmp_8447[2]];
    signal tmp_8482[3] <== [tmp_8479[0] + tmp_8481[0], tmp_8479[1] + tmp_8481[1], tmp_8479[2] + tmp_8481[2]];
    signal tmp_8483[3] <== [tmp_8431[0] * 18163707672964630459, tmp_8431[1] * 18163707672964630459, tmp_8431[2] * 18163707672964630459];
    signal tmp_8484[3] <== [tmp_8483[0] + tmp_8447[0], tmp_8483[1] + tmp_8447[1], tmp_8483[2] + tmp_8447[2]];
    signal tmp_8485[3] <== [tmp_8482[0] + tmp_8484[0], tmp_8482[1] + tmp_8484[1], tmp_8482[2] + tmp_8484[2]];
    signal tmp_8486[3] <== [tmp_8434[0] * 14373610220374016704, tmp_8434[1] * 14373610220374016704, tmp_8434[2] * 14373610220374016704];
    signal tmp_8487[3] <== [tmp_8486[0] + tmp_8447[0], tmp_8486[1] + tmp_8447[1], tmp_8486[2] + tmp_8447[2]];
    signal tmp_8488[3] <== [tmp_8485[0] + tmp_8487[0], tmp_8485[1] + tmp_8487[1], tmp_8485[2] + tmp_8487[2]];
    signal tmp_8489[3] <== [tmp_8437[0] * 6226282807566121054, tmp_8437[1] * 6226282807566121054, tmp_8437[2] * 6226282807566121054];
    signal tmp_8490[3] <== [tmp_8489[0] + tmp_8447[0], tmp_8489[1] + tmp_8447[1], tmp_8489[2] + tmp_8447[2]];
    signal tmp_8491[3] <== [tmp_8488[0] + tmp_8490[0], tmp_8488[1] + tmp_8490[1], tmp_8488[2] + tmp_8490[2]];
    signal tmp_8492[3] <== [tmp_8440[0] * 3643354756180461803, tmp_8440[1] * 3643354756180461803, tmp_8440[2] * 3643354756180461803];
    signal tmp_8493[3] <== [tmp_8492[0] + tmp_8447[0], tmp_8492[1] + tmp_8447[1], tmp_8492[2] + tmp_8447[2]];
    signal tmp_8494[3] <== [tmp_8491[0] + tmp_8493[0], tmp_8491[1] + tmp_8493[1], tmp_8491[2] + tmp_8493[2]];
    signal tmp_8495[3] <== [tmp_8443[0] * 13046961313070095543, tmp_8443[1] * 13046961313070095543, tmp_8443[2] * 13046961313070095543];
    signal tmp_8496[3] <== [tmp_8495[0] + tmp_8447[0], tmp_8495[1] + tmp_8447[1], tmp_8495[2] + tmp_8447[2]];
    signal tmp_8497[3] <== [tmp_8494[0] + tmp_8496[0], tmp_8494[1] + tmp_8496[1], tmp_8494[2] + tmp_8496[2]];
    signal tmp_8498[3] <== [tmp_8446[0] * 8594143216561850811, tmp_8446[1] * 8594143216561850811, tmp_8446[2] * 8594143216561850811];
    signal tmp_8499[3] <== [tmp_8498[0] + tmp_8447[0], tmp_8498[1] + tmp_8447[1], tmp_8498[2] + tmp_8447[2]];
    signal tmp_8500[3] <== [tmp_8497[0] + tmp_8499[0], tmp_8497[1] + tmp_8499[1], tmp_8497[2] + tmp_8499[2]];
    signal tmp_8501[3] <== [tmp_8455[0] + tmp_8500[0], tmp_8455[1] + tmp_8500[1], tmp_8455[2] + tmp_8500[2]];
    signal tmp_8502[3] <== [evals[100][0] - tmp_8501[0], evals[100][1] - tmp_8501[1], evals[100][2] - tmp_8501[2]];
    signal tmp_8503[3] <== CMul()(evals[53], tmp_8502);
    signal tmp_8504[3] <== [tmp_8452[0] + tmp_8503[0], tmp_8452[1] + tmp_8503[1], tmp_8452[2] + tmp_8503[2]];
    signal tmp_8505[3] <== CMul()(challengeQ, tmp_8504);
    signal tmp_8508[3] <== [tmp_7887[0] * 16040574633112940480, tmp_7887[1] * 16040574633112940480, tmp_7887[2] * 16040574633112940480];
    signal tmp_8509[3] <== [tmp_8457[0] * 14263299814608977431, tmp_8457[1] * 14263299814608977431, tmp_8457[2] * 14263299814608977431];
    signal tmp_8510[3] <== [tmp_8509[0] + tmp_8500[0], tmp_8509[1] + tmp_8500[1], tmp_8509[2] + tmp_8500[2]];
    signal tmp_8511[3] <== [tmp_7887[0] + tmp_8510[0], tmp_7887[1] + tmp_8510[1], tmp_7887[2] + tmp_8510[2]];
    signal tmp_8512[3] <== [tmp_8460[0] * 770395855193680981, tmp_8460[1] * 770395855193680981, tmp_8460[2] * 770395855193680981];
    signal tmp_8513[3] <== [tmp_8512[0] + tmp_8500[0], tmp_8512[1] + tmp_8500[1], tmp_8512[2] + tmp_8500[2]];
    signal tmp_8514[3] <== [tmp_8511[0] + tmp_8513[0], tmp_8511[1] + tmp_8513[1], tmp_8511[2] + tmp_8513[2]];
    signal tmp_8515[3] <== [tmp_8463[0] * 3459277367440070515, tmp_8463[1] * 3459277367440070515, tmp_8463[2] * 3459277367440070515];
    signal tmp_8516[3] <== [tmp_8515[0] + tmp_8500[0], tmp_8515[1] + tmp_8500[1], tmp_8515[2] + tmp_8500[2]];
    signal tmp_8517[3] <== [tmp_8514[0] + tmp_8516[0], tmp_8514[1] + tmp_8516[1], tmp_8514[2] + tmp_8516[2]];
    signal tmp_8518[3] <== [tmp_8466[0] * 17087697094293314027, tmp_8466[1] * 17087697094293314027, tmp_8466[2] * 17087697094293314027];
    signal tmp_8519[3] <== [tmp_8518[0] + tmp_8500[0], tmp_8518[1] + tmp_8500[1], tmp_8518[2] + tmp_8500[2]];
    signal tmp_8520[3] <== [tmp_8517[0] + tmp_8519[0], tmp_8517[1] + tmp_8519[1], tmp_8517[2] + tmp_8519[2]];
    signal tmp_8521[3] <== [tmp_8469[0] * 6694380135428747348, tmp_8469[1] * 6694380135428747348, tmp_8469[2] * 6694380135428747348];
    signal tmp_8522[3] <== [tmp_8521[0] + tmp_8500[0], tmp_8521[1] + tmp_8500[1], tmp_8521[2] + tmp_8500[2]];
    signal tmp_8523[3] <== [tmp_8520[0] + tmp_8522[0], tmp_8520[1] + tmp_8522[1], tmp_8520[2] + tmp_8522[2]];
    signal tmp_8524[3] <== [tmp_8472[0] * 2034408310088972836, tmp_8472[1] * 2034408310088972836, tmp_8472[2] * 2034408310088972836];
    signal tmp_8525[3] <== [tmp_8524[0] + tmp_8500[0], tmp_8524[1] + tmp_8500[1], tmp_8524[2] + tmp_8500[2]];
    signal tmp_8526[3] <== [tmp_8523[0] + tmp_8525[0], tmp_8523[1] + tmp_8525[1], tmp_8523[2] + tmp_8525[2]];
    signal tmp_8527[3] <== [tmp_8475[0] * 3434575637390274478, tmp_8475[1] * 3434575637390274478, tmp_8475[2] * 3434575637390274478];
    signal tmp_8528[3] <== [tmp_8527[0] + tmp_8500[0], tmp_8527[1] + tmp_8500[1], tmp_8527[2] + tmp_8500[2]];
    signal tmp_8529[3] <== [tmp_8526[0] + tmp_8528[0], tmp_8526[1] + tmp_8528[1], tmp_8526[2] + tmp_8528[2]];
    signal tmp_8530[3] <== [tmp_8478[0] * 6052753985947965968, tmp_8478[1] * 6052753985947965968, tmp_8478[2] * 6052753985947965968];
    signal tmp_8531[3] <== [tmp_8530[0] + tmp_8500[0], tmp_8530[1] + tmp_8500[1], tmp_8530[2] + tmp_8500[2]];
    signal tmp_8532[3] <== [tmp_8529[0] + tmp_8531[0], tmp_8529[1] + tmp_8531[1], tmp_8529[2] + tmp_8531[2]];
    signal tmp_8533[3] <== [tmp_8481[0] * 13608362914817483670, tmp_8481[1] * 13608362914817483670, tmp_8481[2] * 13608362914817483670];
    signal tmp_8534[3] <== [tmp_8533[0] + tmp_8500[0], tmp_8533[1] + tmp_8500[1], tmp_8533[2] + tmp_8500[2]];
    signal tmp_8535[3] <== [tmp_8532[0] + tmp_8534[0], tmp_8532[1] + tmp_8534[1], tmp_8532[2] + tmp_8534[2]];
    signal tmp_8536[3] <== [tmp_8484[0] * 18163707672964630459, tmp_8484[1] * 18163707672964630459, tmp_8484[2] * 18163707672964630459];
    signal tmp_8537[3] <== [tmp_8536[0] + tmp_8500[0], tmp_8536[1] + tmp_8500[1], tmp_8536[2] + tmp_8500[2]];
    signal tmp_8538[3] <== [tmp_8535[0] + tmp_8537[0], tmp_8535[1] + tmp_8537[1], tmp_8535[2] + tmp_8537[2]];
    signal tmp_8539[3] <== [tmp_8487[0] * 14373610220374016704, tmp_8487[1] * 14373610220374016704, tmp_8487[2] * 14373610220374016704];
    signal tmp_8540[3] <== [tmp_8539[0] + tmp_8500[0], tmp_8539[1] + tmp_8500[1], tmp_8539[2] + tmp_8500[2]];
    signal tmp_8541[3] <== [tmp_8538[0] + tmp_8540[0], tmp_8538[1] + tmp_8540[1], tmp_8538[2] + tmp_8540[2]];
    signal tmp_8542[3] <== [tmp_8490[0] * 6226282807566121054, tmp_8490[1] * 6226282807566121054, tmp_8490[2] * 6226282807566121054];
    signal tmp_8543[3] <== [tmp_8542[0] + tmp_8500[0], tmp_8542[1] + tmp_8500[1], tmp_8542[2] + tmp_8500[2]];
    signal tmp_8544[3] <== [tmp_8541[0] + tmp_8543[0], tmp_8541[1] + tmp_8543[1], tmp_8541[2] + tmp_8543[2]];
    signal tmp_8545[3] <== [tmp_8493[0] * 3643354756180461803, tmp_8493[1] * 3643354756180461803, tmp_8493[2] * 3643354756180461803];
    signal tmp_8546[3] <== [tmp_8545[0] + tmp_8500[0], tmp_8545[1] + tmp_8500[1], tmp_8545[2] + tmp_8500[2]];
    signal tmp_8547[3] <== [tmp_8544[0] + tmp_8546[0], tmp_8544[1] + tmp_8546[1], tmp_8544[2] + tmp_8546[2]];
    signal tmp_8548[3] <== [tmp_8496[0] * 13046961313070095543, tmp_8496[1] * 13046961313070095543, tmp_8496[2] * 13046961313070095543];
    signal tmp_8549[3] <== [tmp_8548[0] + tmp_8500[0], tmp_8548[1] + tmp_8500[1], tmp_8548[2] + tmp_8500[2]];
    signal tmp_8550[3] <== [tmp_8547[0] + tmp_8549[0], tmp_8547[1] + tmp_8549[1], tmp_8547[2] + tmp_8549[2]];
    signal tmp_8551[3] <== [tmp_8499[0] * 8594143216561850811, tmp_8499[1] * 8594143216561850811, tmp_8499[2] * 8594143216561850811];
    signal tmp_8552[3] <== [tmp_8551[0] + tmp_8500[0], tmp_8551[1] + tmp_8500[1], tmp_8551[2] + tmp_8500[2]];
    signal tmp_8553[3] <== [tmp_8550[0] + tmp_8552[0], tmp_8550[1] + tmp_8552[1], tmp_8550[2] + tmp_8552[2]];
    signal tmp_8554[3] <== [tmp_8508[0] + tmp_8553[0], tmp_8508[1] + tmp_8553[1], tmp_8508[2] + tmp_8553[2]];
    signal tmp_8555[3] <== [evals[101][0] - tmp_8554[0], evals[101][1] - tmp_8554[1], evals[101][2] - tmp_8554[2]];
    signal tmp_8556[3] <== CMul()(evals[53], tmp_8555);
    signal tmp_8557[3] <== [tmp_8505[0] + tmp_8556[0], tmp_8505[1] + tmp_8556[1], tmp_8505[2] + tmp_8556[2]];
    tmp_8558 <== CMul()(challengeQ, tmp_8557);
    signal tmp_8561[3] <== [tmp_7904[0] * 16040574633112940480, tmp_7904[1] * 16040574633112940480, tmp_7904[2] * 16040574633112940480];
    signal tmp_8562[3] <== [tmp_8510[0] * 14263299814608977431, tmp_8510[1] * 14263299814608977431, tmp_8510[2] * 14263299814608977431];
    tmp_8563 <== [tmp_8562[0] + tmp_8553[0], tmp_8562[1] + tmp_8553[1], tmp_8562[2] + tmp_8553[2]];
    signal tmp_8564[3] <== [tmp_7904[0] + tmp_8563[0], tmp_7904[1] + tmp_8563[1], tmp_7904[2] + tmp_8563[2]];
    signal tmp_8565[3] <== [tmp_8513[0] * 770395855193680981, tmp_8513[1] * 770395855193680981, tmp_8513[2] * 770395855193680981];
    tmp_8566 <== [tmp_8565[0] + tmp_8553[0], tmp_8565[1] + tmp_8553[1], tmp_8565[2] + tmp_8553[2]];
    signal tmp_8567[3] <== [tmp_8564[0] + tmp_8566[0], tmp_8564[1] + tmp_8566[1], tmp_8564[2] + tmp_8566[2]];
    signal tmp_8568[3] <== [tmp_8516[0] * 3459277367440070515, tmp_8516[1] * 3459277367440070515, tmp_8516[2] * 3459277367440070515];
    tmp_8569 <== [tmp_8568[0] + tmp_8553[0], tmp_8568[1] + tmp_8553[1], tmp_8568[2] + tmp_8553[2]];
    signal tmp_8570[3] <== [tmp_8567[0] + tmp_8569[0], tmp_8567[1] + tmp_8569[1], tmp_8567[2] + tmp_8569[2]];
    signal tmp_8571[3] <== [tmp_8519[0] * 17087697094293314027, tmp_8519[1] * 17087697094293314027, tmp_8519[2] * 17087697094293314027];
    tmp_8572 <== [tmp_8571[0] + tmp_8553[0], tmp_8571[1] + tmp_8553[1], tmp_8571[2] + tmp_8553[2]];
    signal tmp_8573[3] <== [tmp_8570[0] + tmp_8572[0], tmp_8570[1] + tmp_8572[1], tmp_8570[2] + tmp_8572[2]];
    signal tmp_8574[3] <== [tmp_8522[0] * 6694380135428747348, tmp_8522[1] * 6694380135428747348, tmp_8522[2] * 6694380135428747348];
    tmp_8575 <== [tmp_8574[0] + tmp_8553[0], tmp_8574[1] + tmp_8553[1], tmp_8574[2] + tmp_8553[2]];
    signal tmp_8576[3] <== [tmp_8573[0] + tmp_8575[0], tmp_8573[1] + tmp_8575[1], tmp_8573[2] + tmp_8575[2]];
    signal tmp_8577[3] <== [tmp_8525[0] * 2034408310088972836, tmp_8525[1] * 2034408310088972836, tmp_8525[2] * 2034408310088972836];
    tmp_8578 <== [tmp_8577[0] + tmp_8553[0], tmp_8577[1] + tmp_8553[1], tmp_8577[2] + tmp_8553[2]];
    signal tmp_8579[3] <== [tmp_8576[0] + tmp_8578[0], tmp_8576[1] + tmp_8578[1], tmp_8576[2] + tmp_8578[2]];
    signal tmp_8580[3] <== [tmp_8528[0] * 3434575637390274478, tmp_8528[1] * 3434575637390274478, tmp_8528[2] * 3434575637390274478];
    tmp_8581 <== [tmp_8580[0] + tmp_8553[0], tmp_8580[1] + tmp_8553[1], tmp_8580[2] + tmp_8553[2]];
    signal tmp_8582[3] <== [tmp_8579[0] + tmp_8581[0], tmp_8579[1] + tmp_8581[1], tmp_8579[2] + tmp_8581[2]];
    signal tmp_8583[3] <== [tmp_8531[0] * 6052753985947965968, tmp_8531[1] * 6052753985947965968, tmp_8531[2] * 6052753985947965968];
    tmp_8584 <== [tmp_8583[0] + tmp_8553[0], tmp_8583[1] + tmp_8553[1], tmp_8583[2] + tmp_8553[2]];
    signal tmp_8585[3] <== [tmp_8582[0] + tmp_8584[0], tmp_8582[1] + tmp_8584[1], tmp_8582[2] + tmp_8584[2]];
    signal tmp_8586[3] <== [tmp_8534[0] * 13608362914817483670, tmp_8534[1] * 13608362914817483670, tmp_8534[2] * 13608362914817483670];
    tmp_8587 <== [tmp_8586[0] + tmp_8553[0], tmp_8586[1] + tmp_8553[1], tmp_8586[2] + tmp_8553[2]];
    signal tmp_8588[3] <== [tmp_8585[0] + tmp_8587[0], tmp_8585[1] + tmp_8587[1], tmp_8585[2] + tmp_8587[2]];
    signal tmp_8589[3] <== [tmp_8537[0] * 18163707672964630459, tmp_8537[1] * 18163707672964630459, tmp_8537[2] * 18163707672964630459];
    tmp_8590 <== [tmp_8589[0] + tmp_8553[0], tmp_8589[1] + tmp_8553[1], tmp_8589[2] + tmp_8553[2]];
    signal tmp_8591[3] <== [tmp_8588[0] + tmp_8590[0], tmp_8588[1] + tmp_8590[1], tmp_8588[2] + tmp_8590[2]];
    signal tmp_8592[3] <== [tmp_8540[0] * 14373610220374016704, tmp_8540[1] * 14373610220374016704, tmp_8540[2] * 14373610220374016704];
    tmp_8593 <== [tmp_8592[0] + tmp_8553[0], tmp_8592[1] + tmp_8553[1], tmp_8592[2] + tmp_8553[2]];
    signal tmp_8594[3] <== [tmp_8591[0] + tmp_8593[0], tmp_8591[1] + tmp_8593[1], tmp_8591[2] + tmp_8593[2]];
    signal tmp_8595[3] <== [tmp_8543[0] * 6226282807566121054, tmp_8543[1] * 6226282807566121054, tmp_8543[2] * 6226282807566121054];
    tmp_8596 <== [tmp_8595[0] + tmp_8553[0], tmp_8595[1] + tmp_8553[1], tmp_8595[2] + tmp_8553[2]];
    signal tmp_8597[3] <== [tmp_8594[0] + tmp_8596[0], tmp_8594[1] + tmp_8596[1], tmp_8594[2] + tmp_8596[2]];
    signal tmp_8598[3] <== [tmp_8546[0] * 3643354756180461803, tmp_8546[1] * 3643354756180461803, tmp_8546[2] * 3643354756180461803];
    tmp_8599 <== [tmp_8598[0] + tmp_8553[0], tmp_8598[1] + tmp_8553[1], tmp_8598[2] + tmp_8553[2]];
    signal tmp_8600[3] <== [tmp_8597[0] + tmp_8599[0], tmp_8597[1] + tmp_8599[1], tmp_8597[2] + tmp_8599[2]];
    signal tmp_8601[3] <== [tmp_8549[0] * 13046961313070095543, tmp_8549[1] * 13046961313070095543, tmp_8549[2] * 13046961313070095543];
    tmp_8602 <== [tmp_8601[0] + tmp_8553[0], tmp_8601[1] + tmp_8553[1], tmp_8601[2] + tmp_8553[2]];
    signal tmp_8603[3] <== [tmp_8600[0] + tmp_8602[0], tmp_8600[1] + tmp_8602[1], tmp_8600[2] + tmp_8602[2]];
    signal tmp_8604[3] <== [tmp_8552[0] * 8594143216561850811, tmp_8552[1] * 8594143216561850811, tmp_8552[2] * 8594143216561850811];
    tmp_8605 <== [tmp_8604[0] + tmp_8553[0], tmp_8604[1] + tmp_8553[1], tmp_8604[2] + tmp_8553[2]];
    tmp_8606 <== [tmp_8603[0] + tmp_8605[0], tmp_8603[1] + tmp_8605[1], tmp_8603[2] + tmp_8605[2]];
    signal tmp_8607[3] <== [tmp_8561[0] + tmp_8606[0], tmp_8561[1] + tmp_8606[1], tmp_8561[2] + tmp_8606[2]];
    tmp_8608 <== [evals[102][0] - tmp_8607[0], evals[102][1] - tmp_8607[1], evals[102][2] - tmp_8607[2]];
}

template VerifyEvaluationsChunks3_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_7869[3];
    signal input tmp_7927[3];
    signal input tmp_7951[3];
    signal input tmp_7969[3];
    signal input tmp_7986[3];
    signal input tmp_8009[3];
    signal input tmp_8033[3];
    signal input tmp_8051[3];
    signal input tmp_8068[3];
    signal input tmp_8091[3];
    signal input tmp_8558[3];
    signal input tmp_8563[3];
    signal input tmp_8566[3];
    signal input tmp_8569[3];
    signal input tmp_8572[3];
    signal input tmp_8575[3];
    signal input tmp_8578[3];
    signal input tmp_8581[3];
    signal input tmp_8584[3];
    signal input tmp_8587[3];
    signal input tmp_8590[3];
    signal input tmp_8593[3];
    signal input tmp_8596[3];
    signal input tmp_8599[3];
    signal input tmp_8602[3];
    signal input tmp_8605[3];
    signal input tmp_8606[3];
    signal input tmp_8608[3];

    signal output tmp_9657[3];
    signal tmp_8609[3] <== CMul()(evals[53], tmp_8608);
    signal tmp_8610[3] <== [tmp_8558[0] + tmp_8609[0], tmp_8558[1] + tmp_8609[1], tmp_8558[2] + tmp_8609[2]];
    signal tmp_8611[3] <== CMul()(challengeQ, tmp_8610);
    signal tmp_8614[3] <== [tmp_7927[0] * 16040574633112940480, tmp_7927[1] * 16040574633112940480, tmp_7927[2] * 16040574633112940480];
    signal tmp_8615[3] <== [tmp_8563[0] * 14263299814608977431, tmp_8563[1] * 14263299814608977431, tmp_8563[2] * 14263299814608977431];
    signal tmp_8616[3] <== [tmp_8615[0] + tmp_8606[0], tmp_8615[1] + tmp_8606[1], tmp_8615[2] + tmp_8606[2]];
    signal tmp_8617[3] <== [tmp_7927[0] + tmp_8616[0], tmp_7927[1] + tmp_8616[1], tmp_7927[2] + tmp_8616[2]];
    signal tmp_8618[3] <== [tmp_8566[0] * 770395855193680981, tmp_8566[1] * 770395855193680981, tmp_8566[2] * 770395855193680981];
    signal tmp_8619[3] <== [tmp_8618[0] + tmp_8606[0], tmp_8618[1] + tmp_8606[1], tmp_8618[2] + tmp_8606[2]];
    signal tmp_8620[3] <== [tmp_8617[0] + tmp_8619[0], tmp_8617[1] + tmp_8619[1], tmp_8617[2] + tmp_8619[2]];
    signal tmp_8621[3] <== [tmp_8569[0] * 3459277367440070515, tmp_8569[1] * 3459277367440070515, tmp_8569[2] * 3459277367440070515];
    signal tmp_8622[3] <== [tmp_8621[0] + tmp_8606[0], tmp_8621[1] + tmp_8606[1], tmp_8621[2] + tmp_8606[2]];
    signal tmp_8623[3] <== [tmp_8620[0] + tmp_8622[0], tmp_8620[1] + tmp_8622[1], tmp_8620[2] + tmp_8622[2]];
    signal tmp_8624[3] <== [tmp_8572[0] * 17087697094293314027, tmp_8572[1] * 17087697094293314027, tmp_8572[2] * 17087697094293314027];
    signal tmp_8625[3] <== [tmp_8624[0] + tmp_8606[0], tmp_8624[1] + tmp_8606[1], tmp_8624[2] + tmp_8606[2]];
    signal tmp_8626[3] <== [tmp_8623[0] + tmp_8625[0], tmp_8623[1] + tmp_8625[1], tmp_8623[2] + tmp_8625[2]];
    signal tmp_8627[3] <== [tmp_8575[0] * 6694380135428747348, tmp_8575[1] * 6694380135428747348, tmp_8575[2] * 6694380135428747348];
    signal tmp_8628[3] <== [tmp_8627[0] + tmp_8606[0], tmp_8627[1] + tmp_8606[1], tmp_8627[2] + tmp_8606[2]];
    signal tmp_8629[3] <== [tmp_8626[0] + tmp_8628[0], tmp_8626[1] + tmp_8628[1], tmp_8626[2] + tmp_8628[2]];
    signal tmp_8630[3] <== [tmp_8578[0] * 2034408310088972836, tmp_8578[1] * 2034408310088972836, tmp_8578[2] * 2034408310088972836];
    signal tmp_8631[3] <== [tmp_8630[0] + tmp_8606[0], tmp_8630[1] + tmp_8606[1], tmp_8630[2] + tmp_8606[2]];
    signal tmp_8632[3] <== [tmp_8629[0] + tmp_8631[0], tmp_8629[1] + tmp_8631[1], tmp_8629[2] + tmp_8631[2]];
    signal tmp_8633[3] <== [tmp_8581[0] * 3434575637390274478, tmp_8581[1] * 3434575637390274478, tmp_8581[2] * 3434575637390274478];
    signal tmp_8634[3] <== [tmp_8633[0] + tmp_8606[0], tmp_8633[1] + tmp_8606[1], tmp_8633[2] + tmp_8606[2]];
    signal tmp_8635[3] <== [tmp_8632[0] + tmp_8634[0], tmp_8632[1] + tmp_8634[1], tmp_8632[2] + tmp_8634[2]];
    signal tmp_8636[3] <== [tmp_8584[0] * 6052753985947965968, tmp_8584[1] * 6052753985947965968, tmp_8584[2] * 6052753985947965968];
    signal tmp_8637[3] <== [tmp_8636[0] + tmp_8606[0], tmp_8636[1] + tmp_8606[1], tmp_8636[2] + tmp_8606[2]];
    signal tmp_8638[3] <== [tmp_8635[0] + tmp_8637[0], tmp_8635[1] + tmp_8637[1], tmp_8635[2] + tmp_8637[2]];
    signal tmp_8639[3] <== [tmp_8587[0] * 13608362914817483670, tmp_8587[1] * 13608362914817483670, tmp_8587[2] * 13608362914817483670];
    signal tmp_8640[3] <== [tmp_8639[0] + tmp_8606[0], tmp_8639[1] + tmp_8606[1], tmp_8639[2] + tmp_8606[2]];
    signal tmp_8641[3] <== [tmp_8638[0] + tmp_8640[0], tmp_8638[1] + tmp_8640[1], tmp_8638[2] + tmp_8640[2]];
    signal tmp_8642[3] <== [tmp_8590[0] * 18163707672964630459, tmp_8590[1] * 18163707672964630459, tmp_8590[2] * 18163707672964630459];
    signal tmp_8643[3] <== [tmp_8642[0] + tmp_8606[0], tmp_8642[1] + tmp_8606[1], tmp_8642[2] + tmp_8606[2]];
    signal tmp_8644[3] <== [tmp_8641[0] + tmp_8643[0], tmp_8641[1] + tmp_8643[1], tmp_8641[2] + tmp_8643[2]];
    signal tmp_8645[3] <== [tmp_8593[0] * 14373610220374016704, tmp_8593[1] * 14373610220374016704, tmp_8593[2] * 14373610220374016704];
    signal tmp_8646[3] <== [tmp_8645[0] + tmp_8606[0], tmp_8645[1] + tmp_8606[1], tmp_8645[2] + tmp_8606[2]];
    signal tmp_8647[3] <== [tmp_8644[0] + tmp_8646[0], tmp_8644[1] + tmp_8646[1], tmp_8644[2] + tmp_8646[2]];
    signal tmp_8648[3] <== [tmp_8596[0] * 6226282807566121054, tmp_8596[1] * 6226282807566121054, tmp_8596[2] * 6226282807566121054];
    signal tmp_8649[3] <== [tmp_8648[0] + tmp_8606[0], tmp_8648[1] + tmp_8606[1], tmp_8648[2] + tmp_8606[2]];
    signal tmp_8650[3] <== [tmp_8647[0] + tmp_8649[0], tmp_8647[1] + tmp_8649[1], tmp_8647[2] + tmp_8649[2]];
    signal tmp_8651[3] <== [tmp_8599[0] * 3643354756180461803, tmp_8599[1] * 3643354756180461803, tmp_8599[2] * 3643354756180461803];
    signal tmp_8652[3] <== [tmp_8651[0] + tmp_8606[0], tmp_8651[1] + tmp_8606[1], tmp_8651[2] + tmp_8606[2]];
    signal tmp_8653[3] <== [tmp_8650[0] + tmp_8652[0], tmp_8650[1] + tmp_8652[1], tmp_8650[2] + tmp_8652[2]];
    signal tmp_8654[3] <== [tmp_8602[0] * 13046961313070095543, tmp_8602[1] * 13046961313070095543, tmp_8602[2] * 13046961313070095543];
    signal tmp_8655[3] <== [tmp_8654[0] + tmp_8606[0], tmp_8654[1] + tmp_8606[1], tmp_8654[2] + tmp_8606[2]];
    signal tmp_8656[3] <== [tmp_8653[0] + tmp_8655[0], tmp_8653[1] + tmp_8655[1], tmp_8653[2] + tmp_8655[2]];
    signal tmp_8657[3] <== [tmp_8605[0] * 8594143216561850811, tmp_8605[1] * 8594143216561850811, tmp_8605[2] * 8594143216561850811];
    signal tmp_8658[3] <== [tmp_8657[0] + tmp_8606[0], tmp_8657[1] + tmp_8606[1], tmp_8657[2] + tmp_8606[2]];
    signal tmp_8659[3] <== [tmp_8656[0] + tmp_8658[0], tmp_8656[1] + tmp_8658[1], tmp_8656[2] + tmp_8658[2]];
    signal tmp_8660[3] <== [tmp_8614[0] + tmp_8659[0], tmp_8614[1] + tmp_8659[1], tmp_8614[2] + tmp_8659[2]];
    signal tmp_8661[3] <== [evals[103][0] - tmp_8660[0], evals[103][1] - tmp_8660[1], evals[103][2] - tmp_8660[2]];
    signal tmp_8662[3] <== CMul()(evals[53], tmp_8661);
    signal tmp_8663[3] <== [tmp_8611[0] + tmp_8662[0], tmp_8611[1] + tmp_8662[1], tmp_8611[2] + tmp_8662[2]];
    signal tmp_8664[3] <== CMul()(challengeQ, tmp_8663);
    signal tmp_8667[3] <== [tmp_7869[0] * 16040574633112940480, tmp_7869[1] * 16040574633112940480, tmp_7869[2] * 16040574633112940480];
    signal tmp_8668[3] <== [tmp_8616[0] * 14263299814608977431, tmp_8616[1] * 14263299814608977431, tmp_8616[2] * 14263299814608977431];
    signal tmp_8669[3] <== [tmp_8668[0] + tmp_8659[0], tmp_8668[1] + tmp_8659[1], tmp_8668[2] + tmp_8659[2]];
    signal tmp_8670[3] <== [tmp_7869[0] + tmp_8669[0], tmp_7869[1] + tmp_8669[1], tmp_7869[2] + tmp_8669[2]];
    signal tmp_8671[3] <== [tmp_8619[0] * 770395855193680981, tmp_8619[1] * 770395855193680981, tmp_8619[2] * 770395855193680981];
    signal tmp_8672[3] <== [tmp_8671[0] + tmp_8659[0], tmp_8671[1] + tmp_8659[1], tmp_8671[2] + tmp_8659[2]];
    signal tmp_8673[3] <== [tmp_8670[0] + tmp_8672[0], tmp_8670[1] + tmp_8672[1], tmp_8670[2] + tmp_8672[2]];
    signal tmp_8674[3] <== [tmp_8622[0] * 3459277367440070515, tmp_8622[1] * 3459277367440070515, tmp_8622[2] * 3459277367440070515];
    signal tmp_8675[3] <== [tmp_8674[0] + tmp_8659[0], tmp_8674[1] + tmp_8659[1], tmp_8674[2] + tmp_8659[2]];
    signal tmp_8676[3] <== [tmp_8673[0] + tmp_8675[0], tmp_8673[1] + tmp_8675[1], tmp_8673[2] + tmp_8675[2]];
    signal tmp_8677[3] <== [tmp_8625[0] * 17087697094293314027, tmp_8625[1] * 17087697094293314027, tmp_8625[2] * 17087697094293314027];
    signal tmp_8678[3] <== [tmp_8677[0] + tmp_8659[0], tmp_8677[1] + tmp_8659[1], tmp_8677[2] + tmp_8659[2]];
    signal tmp_8679[3] <== [tmp_8676[0] + tmp_8678[0], tmp_8676[1] + tmp_8678[1], tmp_8676[2] + tmp_8678[2]];
    signal tmp_8680[3] <== [tmp_8628[0] * 6694380135428747348, tmp_8628[1] * 6694380135428747348, tmp_8628[2] * 6694380135428747348];
    signal tmp_8681[3] <== [tmp_8680[0] + tmp_8659[0], tmp_8680[1] + tmp_8659[1], tmp_8680[2] + tmp_8659[2]];
    signal tmp_8682[3] <== [tmp_8679[0] + tmp_8681[0], tmp_8679[1] + tmp_8681[1], tmp_8679[2] + tmp_8681[2]];
    signal tmp_8683[3] <== [tmp_8631[0] * 2034408310088972836, tmp_8631[1] * 2034408310088972836, tmp_8631[2] * 2034408310088972836];
    signal tmp_8684[3] <== [tmp_8683[0] + tmp_8659[0], tmp_8683[1] + tmp_8659[1], tmp_8683[2] + tmp_8659[2]];
    signal tmp_8685[3] <== [tmp_8682[0] + tmp_8684[0], tmp_8682[1] + tmp_8684[1], tmp_8682[2] + tmp_8684[2]];
    signal tmp_8686[3] <== [tmp_8634[0] * 3434575637390274478, tmp_8634[1] * 3434575637390274478, tmp_8634[2] * 3434575637390274478];
    signal tmp_8687[3] <== [tmp_8686[0] + tmp_8659[0], tmp_8686[1] + tmp_8659[1], tmp_8686[2] + tmp_8659[2]];
    signal tmp_8688[3] <== [tmp_8685[0] + tmp_8687[0], tmp_8685[1] + tmp_8687[1], tmp_8685[2] + tmp_8687[2]];
    signal tmp_8689[3] <== [tmp_8637[0] * 6052753985947965968, tmp_8637[1] * 6052753985947965968, tmp_8637[2] * 6052753985947965968];
    signal tmp_8690[3] <== [tmp_8689[0] + tmp_8659[0], tmp_8689[1] + tmp_8659[1], tmp_8689[2] + tmp_8659[2]];
    signal tmp_8691[3] <== [tmp_8688[0] + tmp_8690[0], tmp_8688[1] + tmp_8690[1], tmp_8688[2] + tmp_8690[2]];
    signal tmp_8692[3] <== [tmp_8640[0] * 13608362914817483670, tmp_8640[1] * 13608362914817483670, tmp_8640[2] * 13608362914817483670];
    signal tmp_8693[3] <== [tmp_8692[0] + tmp_8659[0], tmp_8692[1] + tmp_8659[1], tmp_8692[2] + tmp_8659[2]];
    signal tmp_8694[3] <== [tmp_8691[0] + tmp_8693[0], tmp_8691[1] + tmp_8693[1], tmp_8691[2] + tmp_8693[2]];
    signal tmp_8695[3] <== [tmp_8643[0] * 18163707672964630459, tmp_8643[1] * 18163707672964630459, tmp_8643[2] * 18163707672964630459];
    signal tmp_8696[3] <== [tmp_8695[0] + tmp_8659[0], tmp_8695[1] + tmp_8659[1], tmp_8695[2] + tmp_8659[2]];
    signal tmp_8697[3] <== [tmp_8694[0] + tmp_8696[0], tmp_8694[1] + tmp_8696[1], tmp_8694[2] + tmp_8696[2]];
    signal tmp_8698[3] <== [tmp_8646[0] * 14373610220374016704, tmp_8646[1] * 14373610220374016704, tmp_8646[2] * 14373610220374016704];
    signal tmp_8699[3] <== [tmp_8698[0] + tmp_8659[0], tmp_8698[1] + tmp_8659[1], tmp_8698[2] + tmp_8659[2]];
    signal tmp_8700[3] <== [tmp_8697[0] + tmp_8699[0], tmp_8697[1] + tmp_8699[1], tmp_8697[2] + tmp_8699[2]];
    signal tmp_8701[3] <== [tmp_8649[0] * 6226282807566121054, tmp_8649[1] * 6226282807566121054, tmp_8649[2] * 6226282807566121054];
    signal tmp_8702[3] <== [tmp_8701[0] + tmp_8659[0], tmp_8701[1] + tmp_8659[1], tmp_8701[2] + tmp_8659[2]];
    signal tmp_8703[3] <== [tmp_8700[0] + tmp_8702[0], tmp_8700[1] + tmp_8702[1], tmp_8700[2] + tmp_8702[2]];
    signal tmp_8704[3] <== [tmp_8652[0] * 3643354756180461803, tmp_8652[1] * 3643354756180461803, tmp_8652[2] * 3643354756180461803];
    signal tmp_8705[3] <== [tmp_8704[0] + tmp_8659[0], tmp_8704[1] + tmp_8659[1], tmp_8704[2] + tmp_8659[2]];
    signal tmp_8706[3] <== [tmp_8703[0] + tmp_8705[0], tmp_8703[1] + tmp_8705[1], tmp_8703[2] + tmp_8705[2]];
    signal tmp_8707[3] <== [tmp_8655[0] * 13046961313070095543, tmp_8655[1] * 13046961313070095543, tmp_8655[2] * 13046961313070095543];
    signal tmp_8708[3] <== [tmp_8707[0] + tmp_8659[0], tmp_8707[1] + tmp_8659[1], tmp_8707[2] + tmp_8659[2]];
    signal tmp_8709[3] <== [tmp_8706[0] + tmp_8708[0], tmp_8706[1] + tmp_8708[1], tmp_8706[2] + tmp_8708[2]];
    signal tmp_8710[3] <== [tmp_8658[0] * 8594143216561850811, tmp_8658[1] * 8594143216561850811, tmp_8658[2] * 8594143216561850811];
    signal tmp_8711[3] <== [tmp_8710[0] + tmp_8659[0], tmp_8710[1] + tmp_8659[1], tmp_8710[2] + tmp_8659[2]];
    signal tmp_8712[3] <== [tmp_8709[0] + tmp_8711[0], tmp_8709[1] + tmp_8711[1], tmp_8709[2] + tmp_8711[2]];
    signal tmp_8713[3] <== [tmp_8667[0] + tmp_8712[0], tmp_8667[1] + tmp_8712[1], tmp_8667[2] + tmp_8712[2]];
    signal tmp_8714[3] <== [evals[104][0] - tmp_8713[0], evals[104][1] - tmp_8713[1], evals[104][2] - tmp_8713[2]];
    signal tmp_8715[3] <== CMul()(evals[53], tmp_8714);
    signal tmp_8716[3] <== [tmp_8664[0] + tmp_8715[0], tmp_8664[1] + tmp_8715[1], tmp_8664[2] + tmp_8715[2]];
    signal tmp_8717[3] <== CMul()(challengeQ, tmp_8716);
    signal tmp_8720[3] <== [tmp_7969[0] * 16040574633112940480, tmp_7969[1] * 16040574633112940480, tmp_7969[2] * 16040574633112940480];
    signal tmp_8721[3] <== [tmp_8669[0] * 14263299814608977431, tmp_8669[1] * 14263299814608977431, tmp_8669[2] * 14263299814608977431];
    signal tmp_8722[3] <== [tmp_8721[0] + tmp_8712[0], tmp_8721[1] + tmp_8712[1], tmp_8721[2] + tmp_8712[2]];
    signal tmp_8723[3] <== [tmp_7969[0] + tmp_8722[0], tmp_7969[1] + tmp_8722[1], tmp_7969[2] + tmp_8722[2]];
    signal tmp_8724[3] <== [tmp_8672[0] * 770395855193680981, tmp_8672[1] * 770395855193680981, tmp_8672[2] * 770395855193680981];
    signal tmp_8725[3] <== [tmp_8724[0] + tmp_8712[0], tmp_8724[1] + tmp_8712[1], tmp_8724[2] + tmp_8712[2]];
    signal tmp_8726[3] <== [tmp_8723[0] + tmp_8725[0], tmp_8723[1] + tmp_8725[1], tmp_8723[2] + tmp_8725[2]];
    signal tmp_8727[3] <== [tmp_8675[0] * 3459277367440070515, tmp_8675[1] * 3459277367440070515, tmp_8675[2] * 3459277367440070515];
    signal tmp_8728[3] <== [tmp_8727[0] + tmp_8712[0], tmp_8727[1] + tmp_8712[1], tmp_8727[2] + tmp_8712[2]];
    signal tmp_8729[3] <== [tmp_8726[0] + tmp_8728[0], tmp_8726[1] + tmp_8728[1], tmp_8726[2] + tmp_8728[2]];
    signal tmp_8730[3] <== [tmp_8678[0] * 17087697094293314027, tmp_8678[1] * 17087697094293314027, tmp_8678[2] * 17087697094293314027];
    signal tmp_8731[3] <== [tmp_8730[0] + tmp_8712[0], tmp_8730[1] + tmp_8712[1], tmp_8730[2] + tmp_8712[2]];
    signal tmp_8732[3] <== [tmp_8729[0] + tmp_8731[0], tmp_8729[1] + tmp_8731[1], tmp_8729[2] + tmp_8731[2]];
    signal tmp_8733[3] <== [tmp_8681[0] * 6694380135428747348, tmp_8681[1] * 6694380135428747348, tmp_8681[2] * 6694380135428747348];
    signal tmp_8734[3] <== [tmp_8733[0] + tmp_8712[0], tmp_8733[1] + tmp_8712[1], tmp_8733[2] + tmp_8712[2]];
    signal tmp_8735[3] <== [tmp_8732[0] + tmp_8734[0], tmp_8732[1] + tmp_8734[1], tmp_8732[2] + tmp_8734[2]];
    signal tmp_8736[3] <== [tmp_8684[0] * 2034408310088972836, tmp_8684[1] * 2034408310088972836, tmp_8684[2] * 2034408310088972836];
    signal tmp_8737[3] <== [tmp_8736[0] + tmp_8712[0], tmp_8736[1] + tmp_8712[1], tmp_8736[2] + tmp_8712[2]];
    signal tmp_8738[3] <== [tmp_8735[0] + tmp_8737[0], tmp_8735[1] + tmp_8737[1], tmp_8735[2] + tmp_8737[2]];
    signal tmp_8739[3] <== [tmp_8687[0] * 3434575637390274478, tmp_8687[1] * 3434575637390274478, tmp_8687[2] * 3434575637390274478];
    signal tmp_8740[3] <== [tmp_8739[0] + tmp_8712[0], tmp_8739[1] + tmp_8712[1], tmp_8739[2] + tmp_8712[2]];
    signal tmp_8741[3] <== [tmp_8738[0] + tmp_8740[0], tmp_8738[1] + tmp_8740[1], tmp_8738[2] + tmp_8740[2]];
    signal tmp_8742[3] <== [tmp_8690[0] * 6052753985947965968, tmp_8690[1] * 6052753985947965968, tmp_8690[2] * 6052753985947965968];
    signal tmp_8743[3] <== [tmp_8742[0] + tmp_8712[0], tmp_8742[1] + tmp_8712[1], tmp_8742[2] + tmp_8712[2]];
    signal tmp_8744[3] <== [tmp_8741[0] + tmp_8743[0], tmp_8741[1] + tmp_8743[1], tmp_8741[2] + tmp_8743[2]];
    signal tmp_8745[3] <== [tmp_8693[0] * 13608362914817483670, tmp_8693[1] * 13608362914817483670, tmp_8693[2] * 13608362914817483670];
    signal tmp_8746[3] <== [tmp_8745[0] + tmp_8712[0], tmp_8745[1] + tmp_8712[1], tmp_8745[2] + tmp_8712[2]];
    signal tmp_8747[3] <== [tmp_8744[0] + tmp_8746[0], tmp_8744[1] + tmp_8746[1], tmp_8744[2] + tmp_8746[2]];
    signal tmp_8748[3] <== [tmp_8696[0] * 18163707672964630459, tmp_8696[1] * 18163707672964630459, tmp_8696[2] * 18163707672964630459];
    signal tmp_8749[3] <== [tmp_8748[0] + tmp_8712[0], tmp_8748[1] + tmp_8712[1], tmp_8748[2] + tmp_8712[2]];
    signal tmp_8750[3] <== [tmp_8747[0] + tmp_8749[0], tmp_8747[1] + tmp_8749[1], tmp_8747[2] + tmp_8749[2]];
    signal tmp_8751[3] <== [tmp_8699[0] * 14373610220374016704, tmp_8699[1] * 14373610220374016704, tmp_8699[2] * 14373610220374016704];
    signal tmp_8752[3] <== [tmp_8751[0] + tmp_8712[0], tmp_8751[1] + tmp_8712[1], tmp_8751[2] + tmp_8712[2]];
    signal tmp_8753[3] <== [tmp_8750[0] + tmp_8752[0], tmp_8750[1] + tmp_8752[1], tmp_8750[2] + tmp_8752[2]];
    signal tmp_8754[3] <== [tmp_8702[0] * 6226282807566121054, tmp_8702[1] * 6226282807566121054, tmp_8702[2] * 6226282807566121054];
    signal tmp_8755[3] <== [tmp_8754[0] + tmp_8712[0], tmp_8754[1] + tmp_8712[1], tmp_8754[2] + tmp_8712[2]];
    signal tmp_8756[3] <== [tmp_8753[0] + tmp_8755[0], tmp_8753[1] + tmp_8755[1], tmp_8753[2] + tmp_8755[2]];
    signal tmp_8757[3] <== [tmp_8705[0] * 3643354756180461803, tmp_8705[1] * 3643354756180461803, tmp_8705[2] * 3643354756180461803];
    signal tmp_8758[3] <== [tmp_8757[0] + tmp_8712[0], tmp_8757[1] + tmp_8712[1], tmp_8757[2] + tmp_8712[2]];
    signal tmp_8759[3] <== [tmp_8756[0] + tmp_8758[0], tmp_8756[1] + tmp_8758[1], tmp_8756[2] + tmp_8758[2]];
    signal tmp_8760[3] <== [tmp_8708[0] * 13046961313070095543, tmp_8708[1] * 13046961313070095543, tmp_8708[2] * 13046961313070095543];
    signal tmp_8761[3] <== [tmp_8760[0] + tmp_8712[0], tmp_8760[1] + tmp_8712[1], tmp_8760[2] + tmp_8712[2]];
    signal tmp_8762[3] <== [tmp_8759[0] + tmp_8761[0], tmp_8759[1] + tmp_8761[1], tmp_8759[2] + tmp_8761[2]];
    signal tmp_8763[3] <== [tmp_8711[0] * 8594143216561850811, tmp_8711[1] * 8594143216561850811, tmp_8711[2] * 8594143216561850811];
    signal tmp_8764[3] <== [tmp_8763[0] + tmp_8712[0], tmp_8763[1] + tmp_8712[1], tmp_8763[2] + tmp_8712[2]];
    signal tmp_8765[3] <== [tmp_8762[0] + tmp_8764[0], tmp_8762[1] + tmp_8764[1], tmp_8762[2] + tmp_8764[2]];
    signal tmp_8766[3] <== [tmp_8720[0] + tmp_8765[0], tmp_8720[1] + tmp_8765[1], tmp_8720[2] + tmp_8765[2]];
    signal tmp_8767[3] <== [evals[105][0] - tmp_8766[0], evals[105][1] - tmp_8766[1], evals[105][2] - tmp_8766[2]];
    signal tmp_8768[3] <== CMul()(evals[53], tmp_8767);
    signal tmp_8769[3] <== [tmp_8717[0] + tmp_8768[0], tmp_8717[1] + tmp_8768[1], tmp_8717[2] + tmp_8768[2]];
    signal tmp_8770[3] <== CMul()(challengeQ, tmp_8769);
    signal tmp_8773[3] <== [tmp_7986[0] * 16040574633112940480, tmp_7986[1] * 16040574633112940480, tmp_7986[2] * 16040574633112940480];
    signal tmp_8774[3] <== [tmp_8722[0] * 14263299814608977431, tmp_8722[1] * 14263299814608977431, tmp_8722[2] * 14263299814608977431];
    signal tmp_8775[3] <== [tmp_8774[0] + tmp_8765[0], tmp_8774[1] + tmp_8765[1], tmp_8774[2] + tmp_8765[2]];
    signal tmp_8776[3] <== [tmp_7986[0] + tmp_8775[0], tmp_7986[1] + tmp_8775[1], tmp_7986[2] + tmp_8775[2]];
    signal tmp_8777[3] <== [tmp_8725[0] * 770395855193680981, tmp_8725[1] * 770395855193680981, tmp_8725[2] * 770395855193680981];
    signal tmp_8778[3] <== [tmp_8777[0] + tmp_8765[0], tmp_8777[1] + tmp_8765[1], tmp_8777[2] + tmp_8765[2]];
    signal tmp_8779[3] <== [tmp_8776[0] + tmp_8778[0], tmp_8776[1] + tmp_8778[1], tmp_8776[2] + tmp_8778[2]];
    signal tmp_8780[3] <== [tmp_8728[0] * 3459277367440070515, tmp_8728[1] * 3459277367440070515, tmp_8728[2] * 3459277367440070515];
    signal tmp_8781[3] <== [tmp_8780[0] + tmp_8765[0], tmp_8780[1] + tmp_8765[1], tmp_8780[2] + tmp_8765[2]];
    signal tmp_8782[3] <== [tmp_8779[0] + tmp_8781[0], tmp_8779[1] + tmp_8781[1], tmp_8779[2] + tmp_8781[2]];
    signal tmp_8783[3] <== [tmp_8731[0] * 17087697094293314027, tmp_8731[1] * 17087697094293314027, tmp_8731[2] * 17087697094293314027];
    signal tmp_8784[3] <== [tmp_8783[0] + tmp_8765[0], tmp_8783[1] + tmp_8765[1], tmp_8783[2] + tmp_8765[2]];
    signal tmp_8785[3] <== [tmp_8782[0] + tmp_8784[0], tmp_8782[1] + tmp_8784[1], tmp_8782[2] + tmp_8784[2]];
    signal tmp_8786[3] <== [tmp_8734[0] * 6694380135428747348, tmp_8734[1] * 6694380135428747348, tmp_8734[2] * 6694380135428747348];
    signal tmp_8787[3] <== [tmp_8786[0] + tmp_8765[0], tmp_8786[1] + tmp_8765[1], tmp_8786[2] + tmp_8765[2]];
    signal tmp_8788[3] <== [tmp_8785[0] + tmp_8787[0], tmp_8785[1] + tmp_8787[1], tmp_8785[2] + tmp_8787[2]];
    signal tmp_8789[3] <== [tmp_8737[0] * 2034408310088972836, tmp_8737[1] * 2034408310088972836, tmp_8737[2] * 2034408310088972836];
    signal tmp_8790[3] <== [tmp_8789[0] + tmp_8765[0], tmp_8789[1] + tmp_8765[1], tmp_8789[2] + tmp_8765[2]];
    signal tmp_8791[3] <== [tmp_8788[0] + tmp_8790[0], tmp_8788[1] + tmp_8790[1], tmp_8788[2] + tmp_8790[2]];
    signal tmp_8792[3] <== [tmp_8740[0] * 3434575637390274478, tmp_8740[1] * 3434575637390274478, tmp_8740[2] * 3434575637390274478];
    signal tmp_8793[3] <== [tmp_8792[0] + tmp_8765[0], tmp_8792[1] + tmp_8765[1], tmp_8792[2] + tmp_8765[2]];
    signal tmp_8794[3] <== [tmp_8791[0] + tmp_8793[0], tmp_8791[1] + tmp_8793[1], tmp_8791[2] + tmp_8793[2]];
    signal tmp_8795[3] <== [tmp_8743[0] * 6052753985947965968, tmp_8743[1] * 6052753985947965968, tmp_8743[2] * 6052753985947965968];
    signal tmp_8796[3] <== [tmp_8795[0] + tmp_8765[0], tmp_8795[1] + tmp_8765[1], tmp_8795[2] + tmp_8765[2]];
    signal tmp_8797[3] <== [tmp_8794[0] + tmp_8796[0], tmp_8794[1] + tmp_8796[1], tmp_8794[2] + tmp_8796[2]];
    signal tmp_8798[3] <== [tmp_8746[0] * 13608362914817483670, tmp_8746[1] * 13608362914817483670, tmp_8746[2] * 13608362914817483670];
    signal tmp_8799[3] <== [tmp_8798[0] + tmp_8765[0], tmp_8798[1] + tmp_8765[1], tmp_8798[2] + tmp_8765[2]];
    signal tmp_8800[3] <== [tmp_8797[0] + tmp_8799[0], tmp_8797[1] + tmp_8799[1], tmp_8797[2] + tmp_8799[2]];
    signal tmp_8801[3] <== [tmp_8749[0] * 18163707672964630459, tmp_8749[1] * 18163707672964630459, tmp_8749[2] * 18163707672964630459];
    signal tmp_8802[3] <== [tmp_8801[0] + tmp_8765[0], tmp_8801[1] + tmp_8765[1], tmp_8801[2] + tmp_8765[2]];
    signal tmp_8803[3] <== [tmp_8800[0] + tmp_8802[0], tmp_8800[1] + tmp_8802[1], tmp_8800[2] + tmp_8802[2]];
    signal tmp_8804[3] <== [tmp_8752[0] * 14373610220374016704, tmp_8752[1] * 14373610220374016704, tmp_8752[2] * 14373610220374016704];
    signal tmp_8805[3] <== [tmp_8804[0] + tmp_8765[0], tmp_8804[1] + tmp_8765[1], tmp_8804[2] + tmp_8765[2]];
    signal tmp_8806[3] <== [tmp_8803[0] + tmp_8805[0], tmp_8803[1] + tmp_8805[1], tmp_8803[2] + tmp_8805[2]];
    signal tmp_8807[3] <== [tmp_8755[0] * 6226282807566121054, tmp_8755[1] * 6226282807566121054, tmp_8755[2] * 6226282807566121054];
    signal tmp_8808[3] <== [tmp_8807[0] + tmp_8765[0], tmp_8807[1] + tmp_8765[1], tmp_8807[2] + tmp_8765[2]];
    signal tmp_8809[3] <== [tmp_8806[0] + tmp_8808[0], tmp_8806[1] + tmp_8808[1], tmp_8806[2] + tmp_8808[2]];
    signal tmp_8810[3] <== [tmp_8758[0] * 3643354756180461803, tmp_8758[1] * 3643354756180461803, tmp_8758[2] * 3643354756180461803];
    signal tmp_8811[3] <== [tmp_8810[0] + tmp_8765[0], tmp_8810[1] + tmp_8765[1], tmp_8810[2] + tmp_8765[2]];
    signal tmp_8812[3] <== [tmp_8809[0] + tmp_8811[0], tmp_8809[1] + tmp_8811[1], tmp_8809[2] + tmp_8811[2]];
    signal tmp_8813[3] <== [tmp_8761[0] * 13046961313070095543, tmp_8761[1] * 13046961313070095543, tmp_8761[2] * 13046961313070095543];
    signal tmp_8814[3] <== [tmp_8813[0] + tmp_8765[0], tmp_8813[1] + tmp_8765[1], tmp_8813[2] + tmp_8765[2]];
    signal tmp_8815[3] <== [tmp_8812[0] + tmp_8814[0], tmp_8812[1] + tmp_8814[1], tmp_8812[2] + tmp_8814[2]];
    signal tmp_8816[3] <== [tmp_8764[0] * 8594143216561850811, tmp_8764[1] * 8594143216561850811, tmp_8764[2] * 8594143216561850811];
    signal tmp_8817[3] <== [tmp_8816[0] + tmp_8765[0], tmp_8816[1] + tmp_8765[1], tmp_8816[2] + tmp_8765[2]];
    signal tmp_8818[3] <== [tmp_8815[0] + tmp_8817[0], tmp_8815[1] + tmp_8817[1], tmp_8815[2] + tmp_8817[2]];
    signal tmp_8819[3] <== [tmp_8773[0] + tmp_8818[0], tmp_8773[1] + tmp_8818[1], tmp_8773[2] + tmp_8818[2]];
    signal tmp_8820[3] <== [evals[106][0] - tmp_8819[0], evals[106][1] - tmp_8819[1], evals[106][2] - tmp_8819[2]];
    signal tmp_8821[3] <== CMul()(evals[53], tmp_8820);
    signal tmp_8822[3] <== [tmp_8770[0] + tmp_8821[0], tmp_8770[1] + tmp_8821[1], tmp_8770[2] + tmp_8821[2]];
    signal tmp_8823[3] <== CMul()(challengeQ, tmp_8822);
    signal tmp_8826[3] <== [tmp_8009[0] * 16040574633112940480, tmp_8009[1] * 16040574633112940480, tmp_8009[2] * 16040574633112940480];
    signal tmp_8827[3] <== [tmp_8775[0] * 14263299814608977431, tmp_8775[1] * 14263299814608977431, tmp_8775[2] * 14263299814608977431];
    signal tmp_8828[3] <== [tmp_8827[0] + tmp_8818[0], tmp_8827[1] + tmp_8818[1], tmp_8827[2] + tmp_8818[2]];
    signal tmp_8829[3] <== [tmp_8009[0] + tmp_8828[0], tmp_8009[1] + tmp_8828[1], tmp_8009[2] + tmp_8828[2]];
    signal tmp_8830[3] <== [tmp_8778[0] * 770395855193680981, tmp_8778[1] * 770395855193680981, tmp_8778[2] * 770395855193680981];
    signal tmp_8831[3] <== [tmp_8830[0] + tmp_8818[0], tmp_8830[1] + tmp_8818[1], tmp_8830[2] + tmp_8818[2]];
    signal tmp_8832[3] <== [tmp_8829[0] + tmp_8831[0], tmp_8829[1] + tmp_8831[1], tmp_8829[2] + tmp_8831[2]];
    signal tmp_8833[3] <== [tmp_8781[0] * 3459277367440070515, tmp_8781[1] * 3459277367440070515, tmp_8781[2] * 3459277367440070515];
    signal tmp_8834[3] <== [tmp_8833[0] + tmp_8818[0], tmp_8833[1] + tmp_8818[1], tmp_8833[2] + tmp_8818[2]];
    signal tmp_8835[3] <== [tmp_8832[0] + tmp_8834[0], tmp_8832[1] + tmp_8834[1], tmp_8832[2] + tmp_8834[2]];
    signal tmp_8836[3] <== [tmp_8784[0] * 17087697094293314027, tmp_8784[1] * 17087697094293314027, tmp_8784[2] * 17087697094293314027];
    signal tmp_8837[3] <== [tmp_8836[0] + tmp_8818[0], tmp_8836[1] + tmp_8818[1], tmp_8836[2] + tmp_8818[2]];
    signal tmp_8838[3] <== [tmp_8835[0] + tmp_8837[0], tmp_8835[1] + tmp_8837[1], tmp_8835[2] + tmp_8837[2]];
    signal tmp_8839[3] <== [tmp_8787[0] * 6694380135428747348, tmp_8787[1] * 6694380135428747348, tmp_8787[2] * 6694380135428747348];
    signal tmp_8840[3] <== [tmp_8839[0] + tmp_8818[0], tmp_8839[1] + tmp_8818[1], tmp_8839[2] + tmp_8818[2]];
    signal tmp_8841[3] <== [tmp_8838[0] + tmp_8840[0], tmp_8838[1] + tmp_8840[1], tmp_8838[2] + tmp_8840[2]];
    signal tmp_8842[3] <== [tmp_8790[0] * 2034408310088972836, tmp_8790[1] * 2034408310088972836, tmp_8790[2] * 2034408310088972836];
    signal tmp_8843[3] <== [tmp_8842[0] + tmp_8818[0], tmp_8842[1] + tmp_8818[1], tmp_8842[2] + tmp_8818[2]];
    signal tmp_8844[3] <== [tmp_8841[0] + tmp_8843[0], tmp_8841[1] + tmp_8843[1], tmp_8841[2] + tmp_8843[2]];
    signal tmp_8845[3] <== [tmp_8793[0] * 3434575637390274478, tmp_8793[1] * 3434575637390274478, tmp_8793[2] * 3434575637390274478];
    signal tmp_8846[3] <== [tmp_8845[0] + tmp_8818[0], tmp_8845[1] + tmp_8818[1], tmp_8845[2] + tmp_8818[2]];
    signal tmp_8847[3] <== [tmp_8844[0] + tmp_8846[0], tmp_8844[1] + tmp_8846[1], tmp_8844[2] + tmp_8846[2]];
    signal tmp_8848[3] <== [tmp_8796[0] * 6052753985947965968, tmp_8796[1] * 6052753985947965968, tmp_8796[2] * 6052753985947965968];
    signal tmp_8849[3] <== [tmp_8848[0] + tmp_8818[0], tmp_8848[1] + tmp_8818[1], tmp_8848[2] + tmp_8818[2]];
    signal tmp_8850[3] <== [tmp_8847[0] + tmp_8849[0], tmp_8847[1] + tmp_8849[1], tmp_8847[2] + tmp_8849[2]];
    signal tmp_8851[3] <== [tmp_8799[0] * 13608362914817483670, tmp_8799[1] * 13608362914817483670, tmp_8799[2] * 13608362914817483670];
    signal tmp_8852[3] <== [tmp_8851[0] + tmp_8818[0], tmp_8851[1] + tmp_8818[1], tmp_8851[2] + tmp_8818[2]];
    signal tmp_8853[3] <== [tmp_8850[0] + tmp_8852[0], tmp_8850[1] + tmp_8852[1], tmp_8850[2] + tmp_8852[2]];
    signal tmp_8854[3] <== [tmp_8802[0] * 18163707672964630459, tmp_8802[1] * 18163707672964630459, tmp_8802[2] * 18163707672964630459];
    signal tmp_8855[3] <== [tmp_8854[0] + tmp_8818[0], tmp_8854[1] + tmp_8818[1], tmp_8854[2] + tmp_8818[2]];
    signal tmp_8856[3] <== [tmp_8853[0] + tmp_8855[0], tmp_8853[1] + tmp_8855[1], tmp_8853[2] + tmp_8855[2]];
    signal tmp_8857[3] <== [tmp_8805[0] * 14373610220374016704, tmp_8805[1] * 14373610220374016704, tmp_8805[2] * 14373610220374016704];
    signal tmp_8858[3] <== [tmp_8857[0] + tmp_8818[0], tmp_8857[1] + tmp_8818[1], tmp_8857[2] + tmp_8818[2]];
    signal tmp_8859[3] <== [tmp_8856[0] + tmp_8858[0], tmp_8856[1] + tmp_8858[1], tmp_8856[2] + tmp_8858[2]];
    signal tmp_8860[3] <== [tmp_8808[0] * 6226282807566121054, tmp_8808[1] * 6226282807566121054, tmp_8808[2] * 6226282807566121054];
    signal tmp_8861[3] <== [tmp_8860[0] + tmp_8818[0], tmp_8860[1] + tmp_8818[1], tmp_8860[2] + tmp_8818[2]];
    signal tmp_8862[3] <== [tmp_8859[0] + tmp_8861[0], tmp_8859[1] + tmp_8861[1], tmp_8859[2] + tmp_8861[2]];
    signal tmp_8863[3] <== [tmp_8811[0] * 3643354756180461803, tmp_8811[1] * 3643354756180461803, tmp_8811[2] * 3643354756180461803];
    signal tmp_8864[3] <== [tmp_8863[0] + tmp_8818[0], tmp_8863[1] + tmp_8818[1], tmp_8863[2] + tmp_8818[2]];
    signal tmp_8865[3] <== [tmp_8862[0] + tmp_8864[0], tmp_8862[1] + tmp_8864[1], tmp_8862[2] + tmp_8864[2]];
    signal tmp_8866[3] <== [tmp_8814[0] * 13046961313070095543, tmp_8814[1] * 13046961313070095543, tmp_8814[2] * 13046961313070095543];
    signal tmp_8867[3] <== [tmp_8866[0] + tmp_8818[0], tmp_8866[1] + tmp_8818[1], tmp_8866[2] + tmp_8818[2]];
    signal tmp_8868[3] <== [tmp_8865[0] + tmp_8867[0], tmp_8865[1] + tmp_8867[1], tmp_8865[2] + tmp_8867[2]];
    signal tmp_8869[3] <== [tmp_8817[0] * 8594143216561850811, tmp_8817[1] * 8594143216561850811, tmp_8817[2] * 8594143216561850811];
    signal tmp_8870[3] <== [tmp_8869[0] + tmp_8818[0], tmp_8869[1] + tmp_8818[1], tmp_8869[2] + tmp_8818[2]];
    signal tmp_8871[3] <== [tmp_8868[0] + tmp_8870[0], tmp_8868[1] + tmp_8870[1], tmp_8868[2] + tmp_8870[2]];
    signal tmp_8872[3] <== [tmp_8826[0] + tmp_8871[0], tmp_8826[1] + tmp_8871[1], tmp_8826[2] + tmp_8871[2]];
    signal tmp_8873[3] <== [evals[107][0] - tmp_8872[0], evals[107][1] - tmp_8872[1], evals[107][2] - tmp_8872[2]];
    signal tmp_8874[3] <== CMul()(evals[53], tmp_8873);
    signal tmp_8875[3] <== [tmp_8823[0] + tmp_8874[0], tmp_8823[1] + tmp_8874[1], tmp_8823[2] + tmp_8874[2]];
    signal tmp_8876[3] <== CMul()(challengeQ, tmp_8875);
    signal tmp_8879[3] <== [tmp_7951[0] * 16040574633112940480, tmp_7951[1] * 16040574633112940480, tmp_7951[2] * 16040574633112940480];
    signal tmp_8880[3] <== [tmp_8828[0] * 14263299814608977431, tmp_8828[1] * 14263299814608977431, tmp_8828[2] * 14263299814608977431];
    signal tmp_8881[3] <== [tmp_8880[0] + tmp_8871[0], tmp_8880[1] + tmp_8871[1], tmp_8880[2] + tmp_8871[2]];
    signal tmp_8882[3] <== [tmp_7951[0] + tmp_8881[0], tmp_7951[1] + tmp_8881[1], tmp_7951[2] + tmp_8881[2]];
    signal tmp_8883[3] <== [tmp_8831[0] * 770395855193680981, tmp_8831[1] * 770395855193680981, tmp_8831[2] * 770395855193680981];
    signal tmp_8884[3] <== [tmp_8883[0] + tmp_8871[0], tmp_8883[1] + tmp_8871[1], tmp_8883[2] + tmp_8871[2]];
    signal tmp_8885[3] <== [tmp_8882[0] + tmp_8884[0], tmp_8882[1] + tmp_8884[1], tmp_8882[2] + tmp_8884[2]];
    signal tmp_8886[3] <== [tmp_8834[0] * 3459277367440070515, tmp_8834[1] * 3459277367440070515, tmp_8834[2] * 3459277367440070515];
    signal tmp_8887[3] <== [tmp_8886[0] + tmp_8871[0], tmp_8886[1] + tmp_8871[1], tmp_8886[2] + tmp_8871[2]];
    signal tmp_8888[3] <== [tmp_8885[0] + tmp_8887[0], tmp_8885[1] + tmp_8887[1], tmp_8885[2] + tmp_8887[2]];
    signal tmp_8889[3] <== [tmp_8837[0] * 17087697094293314027, tmp_8837[1] * 17087697094293314027, tmp_8837[2] * 17087697094293314027];
    signal tmp_8890[3] <== [tmp_8889[0] + tmp_8871[0], tmp_8889[1] + tmp_8871[1], tmp_8889[2] + tmp_8871[2]];
    signal tmp_8891[3] <== [tmp_8888[0] + tmp_8890[0], tmp_8888[1] + tmp_8890[1], tmp_8888[2] + tmp_8890[2]];
    signal tmp_8892[3] <== [tmp_8840[0] * 6694380135428747348, tmp_8840[1] * 6694380135428747348, tmp_8840[2] * 6694380135428747348];
    signal tmp_8893[3] <== [tmp_8892[0] + tmp_8871[0], tmp_8892[1] + tmp_8871[1], tmp_8892[2] + tmp_8871[2]];
    signal tmp_8894[3] <== [tmp_8891[0] + tmp_8893[0], tmp_8891[1] + tmp_8893[1], tmp_8891[2] + tmp_8893[2]];
    signal tmp_8895[3] <== [tmp_8843[0] * 2034408310088972836, tmp_8843[1] * 2034408310088972836, tmp_8843[2] * 2034408310088972836];
    signal tmp_8896[3] <== [tmp_8895[0] + tmp_8871[0], tmp_8895[1] + tmp_8871[1], tmp_8895[2] + tmp_8871[2]];
    signal tmp_8897[3] <== [tmp_8894[0] + tmp_8896[0], tmp_8894[1] + tmp_8896[1], tmp_8894[2] + tmp_8896[2]];
    signal tmp_8898[3] <== [tmp_8846[0] * 3434575637390274478, tmp_8846[1] * 3434575637390274478, tmp_8846[2] * 3434575637390274478];
    signal tmp_8899[3] <== [tmp_8898[0] + tmp_8871[0], tmp_8898[1] + tmp_8871[1], tmp_8898[2] + tmp_8871[2]];
    signal tmp_8900[3] <== [tmp_8897[0] + tmp_8899[0], tmp_8897[1] + tmp_8899[1], tmp_8897[2] + tmp_8899[2]];
    signal tmp_8901[3] <== [tmp_8849[0] * 6052753985947965968, tmp_8849[1] * 6052753985947965968, tmp_8849[2] * 6052753985947965968];
    signal tmp_8902[3] <== [tmp_8901[0] + tmp_8871[0], tmp_8901[1] + tmp_8871[1], tmp_8901[2] + tmp_8871[2]];
    signal tmp_8903[3] <== [tmp_8900[0] + tmp_8902[0], tmp_8900[1] + tmp_8902[1], tmp_8900[2] + tmp_8902[2]];
    signal tmp_8904[3] <== [tmp_8852[0] * 13608362914817483670, tmp_8852[1] * 13608362914817483670, tmp_8852[2] * 13608362914817483670];
    signal tmp_8905[3] <== [tmp_8904[0] + tmp_8871[0], tmp_8904[1] + tmp_8871[1], tmp_8904[2] + tmp_8871[2]];
    signal tmp_8906[3] <== [tmp_8903[0] + tmp_8905[0], tmp_8903[1] + tmp_8905[1], tmp_8903[2] + tmp_8905[2]];
    signal tmp_8907[3] <== [tmp_8855[0] * 18163707672964630459, tmp_8855[1] * 18163707672964630459, tmp_8855[2] * 18163707672964630459];
    signal tmp_8908[3] <== [tmp_8907[0] + tmp_8871[0], tmp_8907[1] + tmp_8871[1], tmp_8907[2] + tmp_8871[2]];
    signal tmp_8909[3] <== [tmp_8906[0] + tmp_8908[0], tmp_8906[1] + tmp_8908[1], tmp_8906[2] + tmp_8908[2]];
    signal tmp_8910[3] <== [tmp_8858[0] * 14373610220374016704, tmp_8858[1] * 14373610220374016704, tmp_8858[2] * 14373610220374016704];
    signal tmp_8911[3] <== [tmp_8910[0] + tmp_8871[0], tmp_8910[1] + tmp_8871[1], tmp_8910[2] + tmp_8871[2]];
    signal tmp_8912[3] <== [tmp_8909[0] + tmp_8911[0], tmp_8909[1] + tmp_8911[1], tmp_8909[2] + tmp_8911[2]];
    signal tmp_8913[3] <== [tmp_8861[0] * 6226282807566121054, tmp_8861[1] * 6226282807566121054, tmp_8861[2] * 6226282807566121054];
    signal tmp_8914[3] <== [tmp_8913[0] + tmp_8871[0], tmp_8913[1] + tmp_8871[1], tmp_8913[2] + tmp_8871[2]];
    signal tmp_8915[3] <== [tmp_8912[0] + tmp_8914[0], tmp_8912[1] + tmp_8914[1], tmp_8912[2] + tmp_8914[2]];
    signal tmp_8916[3] <== [tmp_8864[0] * 3643354756180461803, tmp_8864[1] * 3643354756180461803, tmp_8864[2] * 3643354756180461803];
    signal tmp_8917[3] <== [tmp_8916[0] + tmp_8871[0], tmp_8916[1] + tmp_8871[1], tmp_8916[2] + tmp_8871[2]];
    signal tmp_8918[3] <== [tmp_8915[0] + tmp_8917[0], tmp_8915[1] + tmp_8917[1], tmp_8915[2] + tmp_8917[2]];
    signal tmp_8919[3] <== [tmp_8867[0] * 13046961313070095543, tmp_8867[1] * 13046961313070095543, tmp_8867[2] * 13046961313070095543];
    signal tmp_8920[3] <== [tmp_8919[0] + tmp_8871[0], tmp_8919[1] + tmp_8871[1], tmp_8919[2] + tmp_8871[2]];
    signal tmp_8921[3] <== [tmp_8918[0] + tmp_8920[0], tmp_8918[1] + tmp_8920[1], tmp_8918[2] + tmp_8920[2]];
    signal tmp_8922[3] <== [tmp_8870[0] * 8594143216561850811, tmp_8870[1] * 8594143216561850811, tmp_8870[2] * 8594143216561850811];
    signal tmp_8923[3] <== [tmp_8922[0] + tmp_8871[0], tmp_8922[1] + tmp_8871[1], tmp_8922[2] + tmp_8871[2]];
    signal tmp_8924[3] <== [tmp_8921[0] + tmp_8923[0], tmp_8921[1] + tmp_8923[1], tmp_8921[2] + tmp_8923[2]];
    signal tmp_8925[3] <== [tmp_8879[0] + tmp_8924[0], tmp_8879[1] + tmp_8924[1], tmp_8879[2] + tmp_8924[2]];
    signal tmp_8926[3] <== [evals[108][0] - tmp_8925[0], evals[108][1] - tmp_8925[1], evals[108][2] - tmp_8925[2]];
    signal tmp_8927[3] <== CMul()(evals[53], tmp_8926);
    signal tmp_8928[3] <== [tmp_8876[0] + tmp_8927[0], tmp_8876[1] + tmp_8927[1], tmp_8876[2] + tmp_8927[2]];
    signal tmp_8929[3] <== CMul()(challengeQ, tmp_8928);
    signal tmp_8932[3] <== [tmp_8051[0] * 16040574633112940480, tmp_8051[1] * 16040574633112940480, tmp_8051[2] * 16040574633112940480];
    signal tmp_8933[3] <== [tmp_8881[0] * 14263299814608977431, tmp_8881[1] * 14263299814608977431, tmp_8881[2] * 14263299814608977431];
    signal tmp_8934[3] <== [tmp_8933[0] + tmp_8924[0], tmp_8933[1] + tmp_8924[1], tmp_8933[2] + tmp_8924[2]];
    signal tmp_8935[3] <== [tmp_8051[0] + tmp_8934[0], tmp_8051[1] + tmp_8934[1], tmp_8051[2] + tmp_8934[2]];
    signal tmp_8936[3] <== [tmp_8884[0] * 770395855193680981, tmp_8884[1] * 770395855193680981, tmp_8884[2] * 770395855193680981];
    signal tmp_8937[3] <== [tmp_8936[0] + tmp_8924[0], tmp_8936[1] + tmp_8924[1], tmp_8936[2] + tmp_8924[2]];
    signal tmp_8938[3] <== [tmp_8935[0] + tmp_8937[0], tmp_8935[1] + tmp_8937[1], tmp_8935[2] + tmp_8937[2]];
    signal tmp_8939[3] <== [tmp_8887[0] * 3459277367440070515, tmp_8887[1] * 3459277367440070515, tmp_8887[2] * 3459277367440070515];
    signal tmp_8940[3] <== [tmp_8939[0] + tmp_8924[0], tmp_8939[1] + tmp_8924[1], tmp_8939[2] + tmp_8924[2]];
    signal tmp_8941[3] <== [tmp_8938[0] + tmp_8940[0], tmp_8938[1] + tmp_8940[1], tmp_8938[2] + tmp_8940[2]];
    signal tmp_8942[3] <== [tmp_8890[0] * 17087697094293314027, tmp_8890[1] * 17087697094293314027, tmp_8890[2] * 17087697094293314027];
    signal tmp_8943[3] <== [tmp_8942[0] + tmp_8924[0], tmp_8942[1] + tmp_8924[1], tmp_8942[2] + tmp_8924[2]];
    signal tmp_8944[3] <== [tmp_8941[0] + tmp_8943[0], tmp_8941[1] + tmp_8943[1], tmp_8941[2] + tmp_8943[2]];
    signal tmp_8945[3] <== [tmp_8893[0] * 6694380135428747348, tmp_8893[1] * 6694380135428747348, tmp_8893[2] * 6694380135428747348];
    signal tmp_8946[3] <== [tmp_8945[0] + tmp_8924[0], tmp_8945[1] + tmp_8924[1], tmp_8945[2] + tmp_8924[2]];
    signal tmp_8947[3] <== [tmp_8944[0] + tmp_8946[0], tmp_8944[1] + tmp_8946[1], tmp_8944[2] + tmp_8946[2]];
    signal tmp_8948[3] <== [tmp_8896[0] * 2034408310088972836, tmp_8896[1] * 2034408310088972836, tmp_8896[2] * 2034408310088972836];
    signal tmp_8949[3] <== [tmp_8948[0] + tmp_8924[0], tmp_8948[1] + tmp_8924[1], tmp_8948[2] + tmp_8924[2]];
    signal tmp_8950[3] <== [tmp_8947[0] + tmp_8949[0], tmp_8947[1] + tmp_8949[1], tmp_8947[2] + tmp_8949[2]];
    signal tmp_8951[3] <== [tmp_8899[0] * 3434575637390274478, tmp_8899[1] * 3434575637390274478, tmp_8899[2] * 3434575637390274478];
    signal tmp_8952[3] <== [tmp_8951[0] + tmp_8924[0], tmp_8951[1] + tmp_8924[1], tmp_8951[2] + tmp_8924[2]];
    signal tmp_8953[3] <== [tmp_8950[0] + tmp_8952[0], tmp_8950[1] + tmp_8952[1], tmp_8950[2] + tmp_8952[2]];
    signal tmp_8954[3] <== [tmp_8902[0] * 6052753985947965968, tmp_8902[1] * 6052753985947965968, tmp_8902[2] * 6052753985947965968];
    signal tmp_8955[3] <== [tmp_8954[0] + tmp_8924[0], tmp_8954[1] + tmp_8924[1], tmp_8954[2] + tmp_8924[2]];
    signal tmp_8956[3] <== [tmp_8953[0] + tmp_8955[0], tmp_8953[1] + tmp_8955[1], tmp_8953[2] + tmp_8955[2]];
    signal tmp_8957[3] <== [tmp_8905[0] * 13608362914817483670, tmp_8905[1] * 13608362914817483670, tmp_8905[2] * 13608362914817483670];
    signal tmp_8958[3] <== [tmp_8957[0] + tmp_8924[0], tmp_8957[1] + tmp_8924[1], tmp_8957[2] + tmp_8924[2]];
    signal tmp_8959[3] <== [tmp_8956[0] + tmp_8958[0], tmp_8956[1] + tmp_8958[1], tmp_8956[2] + tmp_8958[2]];
    signal tmp_8960[3] <== [tmp_8908[0] * 18163707672964630459, tmp_8908[1] * 18163707672964630459, tmp_8908[2] * 18163707672964630459];
    signal tmp_8961[3] <== [tmp_8960[0] + tmp_8924[0], tmp_8960[1] + tmp_8924[1], tmp_8960[2] + tmp_8924[2]];
    signal tmp_8962[3] <== [tmp_8959[0] + tmp_8961[0], tmp_8959[1] + tmp_8961[1], tmp_8959[2] + tmp_8961[2]];
    signal tmp_8963[3] <== [tmp_8911[0] * 14373610220374016704, tmp_8911[1] * 14373610220374016704, tmp_8911[2] * 14373610220374016704];
    signal tmp_8964[3] <== [tmp_8963[0] + tmp_8924[0], tmp_8963[1] + tmp_8924[1], tmp_8963[2] + tmp_8924[2]];
    signal tmp_8965[3] <== [tmp_8962[0] + tmp_8964[0], tmp_8962[1] + tmp_8964[1], tmp_8962[2] + tmp_8964[2]];
    signal tmp_8966[3] <== [tmp_8914[0] * 6226282807566121054, tmp_8914[1] * 6226282807566121054, tmp_8914[2] * 6226282807566121054];
    signal tmp_8967[3] <== [tmp_8966[0] + tmp_8924[0], tmp_8966[1] + tmp_8924[1], tmp_8966[2] + tmp_8924[2]];
    signal tmp_8968[3] <== [tmp_8965[0] + tmp_8967[0], tmp_8965[1] + tmp_8967[1], tmp_8965[2] + tmp_8967[2]];
    signal tmp_8969[3] <== [tmp_8917[0] * 3643354756180461803, tmp_8917[1] * 3643354756180461803, tmp_8917[2] * 3643354756180461803];
    signal tmp_8970[3] <== [tmp_8969[0] + tmp_8924[0], tmp_8969[1] + tmp_8924[1], tmp_8969[2] + tmp_8924[2]];
    signal tmp_8971[3] <== [tmp_8968[0] + tmp_8970[0], tmp_8968[1] + tmp_8970[1], tmp_8968[2] + tmp_8970[2]];
    signal tmp_8972[3] <== [tmp_8920[0] * 13046961313070095543, tmp_8920[1] * 13046961313070095543, tmp_8920[2] * 13046961313070095543];
    signal tmp_8973[3] <== [tmp_8972[0] + tmp_8924[0], tmp_8972[1] + tmp_8924[1], tmp_8972[2] + tmp_8924[2]];
    signal tmp_8974[3] <== [tmp_8971[0] + tmp_8973[0], tmp_8971[1] + tmp_8973[1], tmp_8971[2] + tmp_8973[2]];
    signal tmp_8975[3] <== [tmp_8923[0] * 8594143216561850811, tmp_8923[1] * 8594143216561850811, tmp_8923[2] * 8594143216561850811];
    signal tmp_8976[3] <== [tmp_8975[0] + tmp_8924[0], tmp_8975[1] + tmp_8924[1], tmp_8975[2] + tmp_8924[2]];
    signal tmp_8977[3] <== [tmp_8974[0] + tmp_8976[0], tmp_8974[1] + tmp_8976[1], tmp_8974[2] + tmp_8976[2]];
    signal tmp_8978[3] <== [tmp_8932[0] + tmp_8977[0], tmp_8932[1] + tmp_8977[1], tmp_8932[2] + tmp_8977[2]];
    signal tmp_8979[3] <== [evals[109][0] - tmp_8978[0], evals[109][1] - tmp_8978[1], evals[109][2] - tmp_8978[2]];
    signal tmp_8980[3] <== CMul()(evals[53], tmp_8979);
    signal tmp_8981[3] <== [tmp_8929[0] + tmp_8980[0], tmp_8929[1] + tmp_8980[1], tmp_8929[2] + tmp_8980[2]];
    signal tmp_8982[3] <== CMul()(challengeQ, tmp_8981);
    signal tmp_8985[3] <== [tmp_8068[0] * 16040574633112940480, tmp_8068[1] * 16040574633112940480, tmp_8068[2] * 16040574633112940480];
    signal tmp_8986[3] <== [tmp_8934[0] * 14263299814608977431, tmp_8934[1] * 14263299814608977431, tmp_8934[2] * 14263299814608977431];
    signal tmp_8987[3] <== [tmp_8986[0] + tmp_8977[0], tmp_8986[1] + tmp_8977[1], tmp_8986[2] + tmp_8977[2]];
    signal tmp_8988[3] <== [tmp_8068[0] + tmp_8987[0], tmp_8068[1] + tmp_8987[1], tmp_8068[2] + tmp_8987[2]];
    signal tmp_8989[3] <== [tmp_8937[0] * 770395855193680981, tmp_8937[1] * 770395855193680981, tmp_8937[2] * 770395855193680981];
    signal tmp_8990[3] <== [tmp_8989[0] + tmp_8977[0], tmp_8989[1] + tmp_8977[1], tmp_8989[2] + tmp_8977[2]];
    signal tmp_8991[3] <== [tmp_8988[0] + tmp_8990[0], tmp_8988[1] + tmp_8990[1], tmp_8988[2] + tmp_8990[2]];
    signal tmp_8992[3] <== [tmp_8940[0] * 3459277367440070515, tmp_8940[1] * 3459277367440070515, tmp_8940[2] * 3459277367440070515];
    signal tmp_8993[3] <== [tmp_8992[0] + tmp_8977[0], tmp_8992[1] + tmp_8977[1], tmp_8992[2] + tmp_8977[2]];
    signal tmp_8994[3] <== [tmp_8991[0] + tmp_8993[0], tmp_8991[1] + tmp_8993[1], tmp_8991[2] + tmp_8993[2]];
    signal tmp_8995[3] <== [tmp_8943[0] * 17087697094293314027, tmp_8943[1] * 17087697094293314027, tmp_8943[2] * 17087697094293314027];
    signal tmp_8996[3] <== [tmp_8995[0] + tmp_8977[0], tmp_8995[1] + tmp_8977[1], tmp_8995[2] + tmp_8977[2]];
    signal tmp_8997[3] <== [tmp_8994[0] + tmp_8996[0], tmp_8994[1] + tmp_8996[1], tmp_8994[2] + tmp_8996[2]];
    signal tmp_8998[3] <== [tmp_8946[0] * 6694380135428747348, tmp_8946[1] * 6694380135428747348, tmp_8946[2] * 6694380135428747348];
    signal tmp_8999[3] <== [tmp_8998[0] + tmp_8977[0], tmp_8998[1] + tmp_8977[1], tmp_8998[2] + tmp_8977[2]];
    signal tmp_9000[3] <== [tmp_8997[0] + tmp_8999[0], tmp_8997[1] + tmp_8999[1], tmp_8997[2] + tmp_8999[2]];
    signal tmp_9001[3] <== [tmp_8949[0] * 2034408310088972836, tmp_8949[1] * 2034408310088972836, tmp_8949[2] * 2034408310088972836];
    signal tmp_9002[3] <== [tmp_9001[0] + tmp_8977[0], tmp_9001[1] + tmp_8977[1], tmp_9001[2] + tmp_8977[2]];
    signal tmp_9003[3] <== [tmp_9000[0] + tmp_9002[0], tmp_9000[1] + tmp_9002[1], tmp_9000[2] + tmp_9002[2]];
    signal tmp_9004[3] <== [tmp_8952[0] * 3434575637390274478, tmp_8952[1] * 3434575637390274478, tmp_8952[2] * 3434575637390274478];
    signal tmp_9005[3] <== [tmp_9004[0] + tmp_8977[0], tmp_9004[1] + tmp_8977[1], tmp_9004[2] + tmp_8977[2]];
    signal tmp_9006[3] <== [tmp_9003[0] + tmp_9005[0], tmp_9003[1] + tmp_9005[1], tmp_9003[2] + tmp_9005[2]];
    signal tmp_9007[3] <== [tmp_8955[0] * 6052753985947965968, tmp_8955[1] * 6052753985947965968, tmp_8955[2] * 6052753985947965968];
    signal tmp_9008[3] <== [tmp_9007[0] + tmp_8977[0], tmp_9007[1] + tmp_8977[1], tmp_9007[2] + tmp_8977[2]];
    signal tmp_9009[3] <== [tmp_9006[0] + tmp_9008[0], tmp_9006[1] + tmp_9008[1], tmp_9006[2] + tmp_9008[2]];
    signal tmp_9010[3] <== [tmp_8958[0] * 13608362914817483670, tmp_8958[1] * 13608362914817483670, tmp_8958[2] * 13608362914817483670];
    signal tmp_9011[3] <== [tmp_9010[0] + tmp_8977[0], tmp_9010[1] + tmp_8977[1], tmp_9010[2] + tmp_8977[2]];
    signal tmp_9012[3] <== [tmp_9009[0] + tmp_9011[0], tmp_9009[1] + tmp_9011[1], tmp_9009[2] + tmp_9011[2]];
    signal tmp_9013[3] <== [tmp_8961[0] * 18163707672964630459, tmp_8961[1] * 18163707672964630459, tmp_8961[2] * 18163707672964630459];
    signal tmp_9014[3] <== [tmp_9013[0] + tmp_8977[0], tmp_9013[1] + tmp_8977[1], tmp_9013[2] + tmp_8977[2]];
    signal tmp_9015[3] <== [tmp_9012[0] + tmp_9014[0], tmp_9012[1] + tmp_9014[1], tmp_9012[2] + tmp_9014[2]];
    signal tmp_9016[3] <== [tmp_8964[0] * 14373610220374016704, tmp_8964[1] * 14373610220374016704, tmp_8964[2] * 14373610220374016704];
    signal tmp_9017[3] <== [tmp_9016[0] + tmp_8977[0], tmp_9016[1] + tmp_8977[1], tmp_9016[2] + tmp_8977[2]];
    signal tmp_9018[3] <== [tmp_9015[0] + tmp_9017[0], tmp_9015[1] + tmp_9017[1], tmp_9015[2] + tmp_9017[2]];
    signal tmp_9019[3] <== [tmp_8967[0] * 6226282807566121054, tmp_8967[1] * 6226282807566121054, tmp_8967[2] * 6226282807566121054];
    signal tmp_9020[3] <== [tmp_9019[0] + tmp_8977[0], tmp_9019[1] + tmp_8977[1], tmp_9019[2] + tmp_8977[2]];
    signal tmp_9021[3] <== [tmp_9018[0] + tmp_9020[0], tmp_9018[1] + tmp_9020[1], tmp_9018[2] + tmp_9020[2]];
    signal tmp_9022[3] <== [tmp_8970[0] * 3643354756180461803, tmp_8970[1] * 3643354756180461803, tmp_8970[2] * 3643354756180461803];
    signal tmp_9023[3] <== [tmp_9022[0] + tmp_8977[0], tmp_9022[1] + tmp_8977[1], tmp_9022[2] + tmp_8977[2]];
    signal tmp_9024[3] <== [tmp_9021[0] + tmp_9023[0], tmp_9021[1] + tmp_9023[1], tmp_9021[2] + tmp_9023[2]];
    signal tmp_9025[3] <== [tmp_8973[0] * 13046961313070095543, tmp_8973[1] * 13046961313070095543, tmp_8973[2] * 13046961313070095543];
    signal tmp_9026[3] <== [tmp_9025[0] + tmp_8977[0], tmp_9025[1] + tmp_8977[1], tmp_9025[2] + tmp_8977[2]];
    signal tmp_9027[3] <== [tmp_9024[0] + tmp_9026[0], tmp_9024[1] + tmp_9026[1], tmp_9024[2] + tmp_9026[2]];
    signal tmp_9028[3] <== [tmp_8976[0] * 8594143216561850811, tmp_8976[1] * 8594143216561850811, tmp_8976[2] * 8594143216561850811];
    signal tmp_9029[3] <== [tmp_9028[0] + tmp_8977[0], tmp_9028[1] + tmp_8977[1], tmp_9028[2] + tmp_8977[2]];
    signal tmp_9030[3] <== [tmp_9027[0] + tmp_9029[0], tmp_9027[1] + tmp_9029[1], tmp_9027[2] + tmp_9029[2]];
    signal tmp_9031[3] <== [tmp_8985[0] + tmp_9030[0], tmp_8985[1] + tmp_9030[1], tmp_8985[2] + tmp_9030[2]];
    signal tmp_9032[3] <== [evals[110][0] - tmp_9031[0], evals[110][1] - tmp_9031[1], evals[110][2] - tmp_9031[2]];
    signal tmp_9033[3] <== CMul()(evals[53], tmp_9032);
    signal tmp_9034[3] <== [tmp_8982[0] + tmp_9033[0], tmp_8982[1] + tmp_9033[1], tmp_8982[2] + tmp_9033[2]];
    signal tmp_9035[3] <== CMul()(challengeQ, tmp_9034);
    signal tmp_9038[3] <== [tmp_8091[0] * 16040574633112940480, tmp_8091[1] * 16040574633112940480, tmp_8091[2] * 16040574633112940480];
    signal tmp_9039[3] <== [tmp_8987[0] * 14263299814608977431, tmp_8987[1] * 14263299814608977431, tmp_8987[2] * 14263299814608977431];
    signal tmp_9040[3] <== [tmp_9039[0] + tmp_9030[0], tmp_9039[1] + tmp_9030[1], tmp_9039[2] + tmp_9030[2]];
    signal tmp_9041[3] <== [tmp_8091[0] + tmp_9040[0], tmp_8091[1] + tmp_9040[1], tmp_8091[2] + tmp_9040[2]];
    signal tmp_9042[3] <== [tmp_8990[0] * 770395855193680981, tmp_8990[1] * 770395855193680981, tmp_8990[2] * 770395855193680981];
    signal tmp_9043[3] <== [tmp_9042[0] + tmp_9030[0], tmp_9042[1] + tmp_9030[1], tmp_9042[2] + tmp_9030[2]];
    signal tmp_9044[3] <== [tmp_9041[0] + tmp_9043[0], tmp_9041[1] + tmp_9043[1], tmp_9041[2] + tmp_9043[2]];
    signal tmp_9045[3] <== [tmp_8993[0] * 3459277367440070515, tmp_8993[1] * 3459277367440070515, tmp_8993[2] * 3459277367440070515];
    signal tmp_9046[3] <== [tmp_9045[0] + tmp_9030[0], tmp_9045[1] + tmp_9030[1], tmp_9045[2] + tmp_9030[2]];
    signal tmp_9047[3] <== [tmp_9044[0] + tmp_9046[0], tmp_9044[1] + tmp_9046[1], tmp_9044[2] + tmp_9046[2]];
    signal tmp_9048[3] <== [tmp_8996[0] * 17087697094293314027, tmp_8996[1] * 17087697094293314027, tmp_8996[2] * 17087697094293314027];
    signal tmp_9049[3] <== [tmp_9048[0] + tmp_9030[0], tmp_9048[1] + tmp_9030[1], tmp_9048[2] + tmp_9030[2]];
    signal tmp_9050[3] <== [tmp_9047[0] + tmp_9049[0], tmp_9047[1] + tmp_9049[1], tmp_9047[2] + tmp_9049[2]];
    signal tmp_9051[3] <== [tmp_8999[0] * 6694380135428747348, tmp_8999[1] * 6694380135428747348, tmp_8999[2] * 6694380135428747348];
    signal tmp_9052[3] <== [tmp_9051[0] + tmp_9030[0], tmp_9051[1] + tmp_9030[1], tmp_9051[2] + tmp_9030[2]];
    signal tmp_9053[3] <== [tmp_9050[0] + tmp_9052[0], tmp_9050[1] + tmp_9052[1], tmp_9050[2] + tmp_9052[2]];
    signal tmp_9054[3] <== [tmp_9002[0] * 2034408310088972836, tmp_9002[1] * 2034408310088972836, tmp_9002[2] * 2034408310088972836];
    signal tmp_9055[3] <== [tmp_9054[0] + tmp_9030[0], tmp_9054[1] + tmp_9030[1], tmp_9054[2] + tmp_9030[2]];
    signal tmp_9056[3] <== [tmp_9053[0] + tmp_9055[0], tmp_9053[1] + tmp_9055[1], tmp_9053[2] + tmp_9055[2]];
    signal tmp_9057[3] <== [tmp_9005[0] * 3434575637390274478, tmp_9005[1] * 3434575637390274478, tmp_9005[2] * 3434575637390274478];
    signal tmp_9058[3] <== [tmp_9057[0] + tmp_9030[0], tmp_9057[1] + tmp_9030[1], tmp_9057[2] + tmp_9030[2]];
    signal tmp_9059[3] <== [tmp_9056[0] + tmp_9058[0], tmp_9056[1] + tmp_9058[1], tmp_9056[2] + tmp_9058[2]];
    signal tmp_9060[3] <== [tmp_9008[0] * 6052753985947965968, tmp_9008[1] * 6052753985947965968, tmp_9008[2] * 6052753985947965968];
    signal tmp_9061[3] <== [tmp_9060[0] + tmp_9030[0], tmp_9060[1] + tmp_9030[1], tmp_9060[2] + tmp_9030[2]];
    signal tmp_9062[3] <== [tmp_9059[0] + tmp_9061[0], tmp_9059[1] + tmp_9061[1], tmp_9059[2] + tmp_9061[2]];
    signal tmp_9063[3] <== [tmp_9011[0] * 13608362914817483670, tmp_9011[1] * 13608362914817483670, tmp_9011[2] * 13608362914817483670];
    signal tmp_9064[3] <== [tmp_9063[0] + tmp_9030[0], tmp_9063[1] + tmp_9030[1], tmp_9063[2] + tmp_9030[2]];
    signal tmp_9065[3] <== [tmp_9062[0] + tmp_9064[0], tmp_9062[1] + tmp_9064[1], tmp_9062[2] + tmp_9064[2]];
    signal tmp_9066[3] <== [tmp_9014[0] * 18163707672964630459, tmp_9014[1] * 18163707672964630459, tmp_9014[2] * 18163707672964630459];
    signal tmp_9067[3] <== [tmp_9066[0] + tmp_9030[0], tmp_9066[1] + tmp_9030[1], tmp_9066[2] + tmp_9030[2]];
    signal tmp_9068[3] <== [tmp_9065[0] + tmp_9067[0], tmp_9065[1] + tmp_9067[1], tmp_9065[2] + tmp_9067[2]];
    signal tmp_9069[3] <== [tmp_9017[0] * 14373610220374016704, tmp_9017[1] * 14373610220374016704, tmp_9017[2] * 14373610220374016704];
    signal tmp_9070[3] <== [tmp_9069[0] + tmp_9030[0], tmp_9069[1] + tmp_9030[1], tmp_9069[2] + tmp_9030[2]];
    signal tmp_9071[3] <== [tmp_9068[0] + tmp_9070[0], tmp_9068[1] + tmp_9070[1], tmp_9068[2] + tmp_9070[2]];
    signal tmp_9072[3] <== [tmp_9020[0] * 6226282807566121054, tmp_9020[1] * 6226282807566121054, tmp_9020[2] * 6226282807566121054];
    signal tmp_9073[3] <== [tmp_9072[0] + tmp_9030[0], tmp_9072[1] + tmp_9030[1], tmp_9072[2] + tmp_9030[2]];
    signal tmp_9074[3] <== [tmp_9071[0] + tmp_9073[0], tmp_9071[1] + tmp_9073[1], tmp_9071[2] + tmp_9073[2]];
    signal tmp_9075[3] <== [tmp_9023[0] * 3643354756180461803, tmp_9023[1] * 3643354756180461803, tmp_9023[2] * 3643354756180461803];
    signal tmp_9076[3] <== [tmp_9075[0] + tmp_9030[0], tmp_9075[1] + tmp_9030[1], tmp_9075[2] + tmp_9030[2]];
    signal tmp_9077[3] <== [tmp_9074[0] + tmp_9076[0], tmp_9074[1] + tmp_9076[1], tmp_9074[2] + tmp_9076[2]];
    signal tmp_9078[3] <== [tmp_9026[0] * 13046961313070095543, tmp_9026[1] * 13046961313070095543, tmp_9026[2] * 13046961313070095543];
    signal tmp_9079[3] <== [tmp_9078[0] + tmp_9030[0], tmp_9078[1] + tmp_9030[1], tmp_9078[2] + tmp_9030[2]];
    signal tmp_9080[3] <== [tmp_9077[0] + tmp_9079[0], tmp_9077[1] + tmp_9079[1], tmp_9077[2] + tmp_9079[2]];
    signal tmp_9081[3] <== [tmp_9029[0] * 8594143216561850811, tmp_9029[1] * 8594143216561850811, tmp_9029[2] * 8594143216561850811];
    signal tmp_9082[3] <== [tmp_9081[0] + tmp_9030[0], tmp_9081[1] + tmp_9030[1], tmp_9081[2] + tmp_9030[2]];
    signal tmp_9083[3] <== [tmp_9080[0] + tmp_9082[0], tmp_9080[1] + tmp_9082[1], tmp_9080[2] + tmp_9082[2]];
    signal tmp_9084[3] <== [tmp_9038[0] + tmp_9083[0], tmp_9038[1] + tmp_9083[1], tmp_9038[2] + tmp_9083[2]];
    signal tmp_9085[3] <== [evals[111][0] - tmp_9084[0], evals[111][1] - tmp_9084[1], evals[111][2] - tmp_9084[2]];
    signal tmp_9086[3] <== CMul()(evals[53], tmp_9085);
    signal tmp_9087[3] <== [tmp_9035[0] + tmp_9086[0], tmp_9035[1] + tmp_9086[1], tmp_9035[2] + tmp_9086[2]];
    signal tmp_9088[3] <== CMul()(challengeQ, tmp_9087);
    signal tmp_9091[3] <== [tmp_8033[0] * 16040574633112940480, tmp_8033[1] * 16040574633112940480, tmp_8033[2] * 16040574633112940480];
    signal tmp_9092[3] <== [tmp_9040[0] * 14263299814608977431, tmp_9040[1] * 14263299814608977431, tmp_9040[2] * 14263299814608977431];
    signal tmp_9093[3] <== [tmp_9092[0] + tmp_9083[0], tmp_9092[1] + tmp_9083[1], tmp_9092[2] + tmp_9083[2]];
    signal tmp_9094[3] <== [tmp_8033[0] + tmp_9093[0], tmp_8033[1] + tmp_9093[1], tmp_8033[2] + tmp_9093[2]];
    signal tmp_9095[3] <== [tmp_9043[0] * 770395855193680981, tmp_9043[1] * 770395855193680981, tmp_9043[2] * 770395855193680981];
    signal tmp_9096[3] <== [tmp_9095[0] + tmp_9083[0], tmp_9095[1] + tmp_9083[1], tmp_9095[2] + tmp_9083[2]];
    signal tmp_9097[3] <== [tmp_9094[0] + tmp_9096[0], tmp_9094[1] + tmp_9096[1], tmp_9094[2] + tmp_9096[2]];
    signal tmp_9098[3] <== [tmp_9046[0] * 3459277367440070515, tmp_9046[1] * 3459277367440070515, tmp_9046[2] * 3459277367440070515];
    signal tmp_9099[3] <== [tmp_9098[0] + tmp_9083[0], tmp_9098[1] + tmp_9083[1], tmp_9098[2] + tmp_9083[2]];
    signal tmp_9100[3] <== [tmp_9097[0] + tmp_9099[0], tmp_9097[1] + tmp_9099[1], tmp_9097[2] + tmp_9099[2]];
    signal tmp_9101[3] <== [tmp_9049[0] * 17087697094293314027, tmp_9049[1] * 17087697094293314027, tmp_9049[2] * 17087697094293314027];
    signal tmp_9102[3] <== [tmp_9101[0] + tmp_9083[0], tmp_9101[1] + tmp_9083[1], tmp_9101[2] + tmp_9083[2]];
    signal tmp_9103[3] <== [tmp_9100[0] + tmp_9102[0], tmp_9100[1] + tmp_9102[1], tmp_9100[2] + tmp_9102[2]];
    signal tmp_9104[3] <== [tmp_9052[0] * 6694380135428747348, tmp_9052[1] * 6694380135428747348, tmp_9052[2] * 6694380135428747348];
    signal tmp_9105[3] <== [tmp_9104[0] + tmp_9083[0], tmp_9104[1] + tmp_9083[1], tmp_9104[2] + tmp_9083[2]];
    signal tmp_9106[3] <== [tmp_9103[0] + tmp_9105[0], tmp_9103[1] + tmp_9105[1], tmp_9103[2] + tmp_9105[2]];
    signal tmp_9107[3] <== [tmp_9055[0] * 2034408310088972836, tmp_9055[1] * 2034408310088972836, tmp_9055[2] * 2034408310088972836];
    signal tmp_9108[3] <== [tmp_9107[0] + tmp_9083[0], tmp_9107[1] + tmp_9083[1], tmp_9107[2] + tmp_9083[2]];
    signal tmp_9109[3] <== [tmp_9106[0] + tmp_9108[0], tmp_9106[1] + tmp_9108[1], tmp_9106[2] + tmp_9108[2]];
    signal tmp_9110[3] <== [tmp_9058[0] * 3434575637390274478, tmp_9058[1] * 3434575637390274478, tmp_9058[2] * 3434575637390274478];
    signal tmp_9111[3] <== [tmp_9110[0] + tmp_9083[0], tmp_9110[1] + tmp_9083[1], tmp_9110[2] + tmp_9083[2]];
    signal tmp_9112[3] <== [tmp_9109[0] + tmp_9111[0], tmp_9109[1] + tmp_9111[1], tmp_9109[2] + tmp_9111[2]];
    signal tmp_9113[3] <== [tmp_9061[0] * 6052753985947965968, tmp_9061[1] * 6052753985947965968, tmp_9061[2] * 6052753985947965968];
    signal tmp_9114[3] <== [tmp_9113[0] + tmp_9083[0], tmp_9113[1] + tmp_9083[1], tmp_9113[2] + tmp_9083[2]];
    signal tmp_9115[3] <== [tmp_9112[0] + tmp_9114[0], tmp_9112[1] + tmp_9114[1], tmp_9112[2] + tmp_9114[2]];
    signal tmp_9116[3] <== [tmp_9064[0] * 13608362914817483670, tmp_9064[1] * 13608362914817483670, tmp_9064[2] * 13608362914817483670];
    signal tmp_9117[3] <== [tmp_9116[0] + tmp_9083[0], tmp_9116[1] + tmp_9083[1], tmp_9116[2] + tmp_9083[2]];
    signal tmp_9118[3] <== [tmp_9115[0] + tmp_9117[0], tmp_9115[1] + tmp_9117[1], tmp_9115[2] + tmp_9117[2]];
    signal tmp_9119[3] <== [tmp_9067[0] * 18163707672964630459, tmp_9067[1] * 18163707672964630459, tmp_9067[2] * 18163707672964630459];
    signal tmp_9120[3] <== [tmp_9119[0] + tmp_9083[0], tmp_9119[1] + tmp_9083[1], tmp_9119[2] + tmp_9083[2]];
    signal tmp_9121[3] <== [tmp_9118[0] + tmp_9120[0], tmp_9118[1] + tmp_9120[1], tmp_9118[2] + tmp_9120[2]];
    signal tmp_9122[3] <== [tmp_9070[0] * 14373610220374016704, tmp_9070[1] * 14373610220374016704, tmp_9070[2] * 14373610220374016704];
    signal tmp_9123[3] <== [tmp_9122[0] + tmp_9083[0], tmp_9122[1] + tmp_9083[1], tmp_9122[2] + tmp_9083[2]];
    signal tmp_9124[3] <== [tmp_9121[0] + tmp_9123[0], tmp_9121[1] + tmp_9123[1], tmp_9121[2] + tmp_9123[2]];
    signal tmp_9125[3] <== [tmp_9073[0] * 6226282807566121054, tmp_9073[1] * 6226282807566121054, tmp_9073[2] * 6226282807566121054];
    signal tmp_9126[3] <== [tmp_9125[0] + tmp_9083[0], tmp_9125[1] + tmp_9083[1], tmp_9125[2] + tmp_9083[2]];
    signal tmp_9127[3] <== [tmp_9124[0] + tmp_9126[0], tmp_9124[1] + tmp_9126[1], tmp_9124[2] + tmp_9126[2]];
    signal tmp_9128[3] <== [tmp_9076[0] * 3643354756180461803, tmp_9076[1] * 3643354756180461803, tmp_9076[2] * 3643354756180461803];
    signal tmp_9129[3] <== [tmp_9128[0] + tmp_9083[0], tmp_9128[1] + tmp_9083[1], tmp_9128[2] + tmp_9083[2]];
    signal tmp_9130[3] <== [tmp_9127[0] + tmp_9129[0], tmp_9127[1] + tmp_9129[1], tmp_9127[2] + tmp_9129[2]];
    signal tmp_9131[3] <== [tmp_9079[0] * 13046961313070095543, tmp_9079[1] * 13046961313070095543, tmp_9079[2] * 13046961313070095543];
    signal tmp_9132[3] <== [tmp_9131[0] + tmp_9083[0], tmp_9131[1] + tmp_9083[1], tmp_9131[2] + tmp_9083[2]];
    signal tmp_9133[3] <== [tmp_9130[0] + tmp_9132[0], tmp_9130[1] + tmp_9132[1], tmp_9130[2] + tmp_9132[2]];
    signal tmp_9134[3] <== [tmp_9082[0] * 8594143216561850811, tmp_9082[1] * 8594143216561850811, tmp_9082[2] * 8594143216561850811];
    signal tmp_9135[3] <== [tmp_9134[0] + tmp_9083[0], tmp_9134[1] + tmp_9083[1], tmp_9134[2] + tmp_9083[2]];
    signal tmp_9136[3] <== [tmp_9133[0] + tmp_9135[0], tmp_9133[1] + tmp_9135[1], tmp_9133[2] + tmp_9135[2]];
    signal tmp_9137[3] <== [tmp_9091[0] + tmp_9136[0], tmp_9091[1] + tmp_9136[1], tmp_9091[2] + tmp_9136[2]];
    signal tmp_9138[3] <== [evals[73][0] - tmp_9137[0], evals[73][1] - tmp_9137[1], evals[73][2] - tmp_9137[2]];
    signal tmp_9139[3] <== CMul()(evals[53], tmp_9138);
    signal tmp_9140[3] <== [tmp_9088[0] + tmp_9139[0], tmp_9088[1] + tmp_9139[1], tmp_9088[2] + tmp_9139[2]];
    signal tmp_9141[3] <== CMul()(challengeQ, tmp_9140);
    signal tmp_9142[3] <== [evals[73][0] + 17869527288639567155, evals[73][1], evals[73][2]];
    signal tmp_9144[3] <== CMul()(tmp_9142, tmp_9142);
    signal tmp_9145[3] <== CMul()(tmp_9144, tmp_9144);
    signal tmp_9146[3] <== CMul()(tmp_9145, tmp_9144);
    signal tmp_9148[3] <== CMul()(tmp_9146, tmp_9142);
    signal tmp_9149[3] <== [tmp_9148[0] * 16040574633112940480, tmp_9148[1] * 16040574633112940480, tmp_9148[2] * 16040574633112940480];
    signal tmp_9152[3] <== [tmp_9093[0] * 14263299814608977431, tmp_9093[1] * 14263299814608977431, tmp_9093[2] * 14263299814608977431];
    signal tmp_9153[3] <== [tmp_9152[0] + tmp_9136[0], tmp_9152[1] + tmp_9136[1], tmp_9152[2] + tmp_9136[2]];
    signal tmp_9154[3] <== [tmp_9148[0] + tmp_9153[0], tmp_9148[1] + tmp_9153[1], tmp_9148[2] + tmp_9153[2]];
    signal tmp_9155[3] <== [tmp_9096[0] * 770395855193680981, tmp_9096[1] * 770395855193680981, tmp_9096[2] * 770395855193680981];
    signal tmp_9156[3] <== [tmp_9155[0] + tmp_9136[0], tmp_9155[1] + tmp_9136[1], tmp_9155[2] + tmp_9136[2]];
    signal tmp_9157[3] <== [tmp_9154[0] + tmp_9156[0], tmp_9154[1] + tmp_9156[1], tmp_9154[2] + tmp_9156[2]];
    signal tmp_9158[3] <== [tmp_9099[0] * 3459277367440070515, tmp_9099[1] * 3459277367440070515, tmp_9099[2] * 3459277367440070515];
    signal tmp_9159[3] <== [tmp_9158[0] + tmp_9136[0], tmp_9158[1] + tmp_9136[1], tmp_9158[2] + tmp_9136[2]];
    signal tmp_9160[3] <== [tmp_9157[0] + tmp_9159[0], tmp_9157[1] + tmp_9159[1], tmp_9157[2] + tmp_9159[2]];
    signal tmp_9161[3] <== [tmp_9102[0] * 17087697094293314027, tmp_9102[1] * 17087697094293314027, tmp_9102[2] * 17087697094293314027];
    signal tmp_9162[3] <== [tmp_9161[0] + tmp_9136[0], tmp_9161[1] + tmp_9136[1], tmp_9161[2] + tmp_9136[2]];
    signal tmp_9163[3] <== [tmp_9160[0] + tmp_9162[0], tmp_9160[1] + tmp_9162[1], tmp_9160[2] + tmp_9162[2]];
    signal tmp_9164[3] <== [tmp_9105[0] * 6694380135428747348, tmp_9105[1] * 6694380135428747348, tmp_9105[2] * 6694380135428747348];
    signal tmp_9165[3] <== [tmp_9164[0] + tmp_9136[0], tmp_9164[1] + tmp_9136[1], tmp_9164[2] + tmp_9136[2]];
    signal tmp_9166[3] <== [tmp_9163[0] + tmp_9165[0], tmp_9163[1] + tmp_9165[1], tmp_9163[2] + tmp_9165[2]];
    signal tmp_9167[3] <== [tmp_9108[0] * 2034408310088972836, tmp_9108[1] * 2034408310088972836, tmp_9108[2] * 2034408310088972836];
    signal tmp_9168[3] <== [tmp_9167[0] + tmp_9136[0], tmp_9167[1] + tmp_9136[1], tmp_9167[2] + tmp_9136[2]];
    signal tmp_9169[3] <== [tmp_9166[0] + tmp_9168[0], tmp_9166[1] + tmp_9168[1], tmp_9166[2] + tmp_9168[2]];
    signal tmp_9170[3] <== [tmp_9111[0] * 3434575637390274478, tmp_9111[1] * 3434575637390274478, tmp_9111[2] * 3434575637390274478];
    signal tmp_9171[3] <== [tmp_9170[0] + tmp_9136[0], tmp_9170[1] + tmp_9136[1], tmp_9170[2] + tmp_9136[2]];
    signal tmp_9172[3] <== [tmp_9169[0] + tmp_9171[0], tmp_9169[1] + tmp_9171[1], tmp_9169[2] + tmp_9171[2]];
    signal tmp_9173[3] <== [tmp_9114[0] * 6052753985947965968, tmp_9114[1] * 6052753985947965968, tmp_9114[2] * 6052753985947965968];
    signal tmp_9174[3] <== [tmp_9173[0] + tmp_9136[0], tmp_9173[1] + tmp_9136[1], tmp_9173[2] + tmp_9136[2]];
    signal tmp_9175[3] <== [tmp_9172[0] + tmp_9174[0], tmp_9172[1] + tmp_9174[1], tmp_9172[2] + tmp_9174[2]];
    signal tmp_9176[3] <== [tmp_9117[0] * 13608362914817483670, tmp_9117[1] * 13608362914817483670, tmp_9117[2] * 13608362914817483670];
    signal tmp_9177[3] <== [tmp_9176[0] + tmp_9136[0], tmp_9176[1] + tmp_9136[1], tmp_9176[2] + tmp_9136[2]];
    signal tmp_9178[3] <== [tmp_9175[0] + tmp_9177[0], tmp_9175[1] + tmp_9177[1], tmp_9175[2] + tmp_9177[2]];
    signal tmp_9179[3] <== [tmp_9120[0] * 18163707672964630459, tmp_9120[1] * 18163707672964630459, tmp_9120[2] * 18163707672964630459];
    signal tmp_9180[3] <== [tmp_9179[0] + tmp_9136[0], tmp_9179[1] + tmp_9136[1], tmp_9179[2] + tmp_9136[2]];
    signal tmp_9181[3] <== [tmp_9178[0] + tmp_9180[0], tmp_9178[1] + tmp_9180[1], tmp_9178[2] + tmp_9180[2]];
    signal tmp_9182[3] <== [tmp_9123[0] * 14373610220374016704, tmp_9123[1] * 14373610220374016704, tmp_9123[2] * 14373610220374016704];
    signal tmp_9183[3] <== [tmp_9182[0] + tmp_9136[0], tmp_9182[1] + tmp_9136[1], tmp_9182[2] + tmp_9136[2]];
    signal tmp_9184[3] <== [tmp_9181[0] + tmp_9183[0], tmp_9181[1] + tmp_9183[1], tmp_9181[2] + tmp_9183[2]];
    signal tmp_9185[3] <== [tmp_9126[0] * 6226282807566121054, tmp_9126[1] * 6226282807566121054, tmp_9126[2] * 6226282807566121054];
    signal tmp_9186[3] <== [tmp_9185[0] + tmp_9136[0], tmp_9185[1] + tmp_9136[1], tmp_9185[2] + tmp_9136[2]];
    signal tmp_9187[3] <== [tmp_9184[0] + tmp_9186[0], tmp_9184[1] + tmp_9186[1], tmp_9184[2] + tmp_9186[2]];
    signal tmp_9188[3] <== [tmp_9129[0] * 3643354756180461803, tmp_9129[1] * 3643354756180461803, tmp_9129[2] * 3643354756180461803];
    signal tmp_9189[3] <== [tmp_9188[0] + tmp_9136[0], tmp_9188[1] + tmp_9136[1], tmp_9188[2] + tmp_9136[2]];
    signal tmp_9190[3] <== [tmp_9187[0] + tmp_9189[0], tmp_9187[1] + tmp_9189[1], tmp_9187[2] + tmp_9189[2]];
    signal tmp_9191[3] <== [tmp_9132[0] * 13046961313070095543, tmp_9132[1] * 13046961313070095543, tmp_9132[2] * 13046961313070095543];
    signal tmp_9192[3] <== [tmp_9191[0] + tmp_9136[0], tmp_9191[1] + tmp_9136[1], tmp_9191[2] + tmp_9136[2]];
    signal tmp_9193[3] <== [tmp_9190[0] + tmp_9192[0], tmp_9190[1] + tmp_9192[1], tmp_9190[2] + tmp_9192[2]];
    signal tmp_9194[3] <== [tmp_9135[0] * 8594143216561850811, tmp_9135[1] * 8594143216561850811, tmp_9135[2] * 8594143216561850811];
    signal tmp_9195[3] <== [tmp_9194[0] + tmp_9136[0], tmp_9194[1] + tmp_9136[1], tmp_9194[2] + tmp_9136[2]];
    signal tmp_9196[3] <== [tmp_9193[0] + tmp_9195[0], tmp_9193[1] + tmp_9195[1], tmp_9193[2] + tmp_9195[2]];
    signal tmp_9197[3] <== [tmp_9149[0] + tmp_9196[0], tmp_9149[1] + tmp_9196[1], tmp_9149[2] + tmp_9196[2]];
    signal tmp_9198[3] <== [evals[74][0] - tmp_9197[0], evals[74][1] - tmp_9197[1], evals[74][2] - tmp_9197[2]];
    signal tmp_9199[3] <== CMul()(evals[53], tmp_9198);
    signal tmp_9200[3] <== [tmp_9141[0] + tmp_9199[0], tmp_9141[1] + tmp_9199[1], tmp_9141[2] + tmp_9199[2]];
    signal tmp_9201[3] <== CMul()(challengeQ, tmp_9200);
    signal tmp_9202[3] <== [evals[74][0] + 7829055113315023688, evals[74][1], evals[74][2]];
    signal tmp_9204[3] <== CMul()(tmp_9202, tmp_9202);
    signal tmp_9205[3] <== CMul()(tmp_9204, tmp_9204);
    signal tmp_9206[3] <== CMul()(tmp_9205, tmp_9204);
    signal tmp_9208[3] <== CMul()(tmp_9206, tmp_9202);
    signal tmp_9209[3] <== [tmp_9208[0] * 16040574633112940480, tmp_9208[1] * 16040574633112940480, tmp_9208[2] * 16040574633112940480];
    signal tmp_9212[3] <== [tmp_9153[0] * 14263299814608977431, tmp_9153[1] * 14263299814608977431, tmp_9153[2] * 14263299814608977431];
    signal tmp_9213[3] <== [tmp_9212[0] + tmp_9196[0], tmp_9212[1] + tmp_9196[1], tmp_9212[2] + tmp_9196[2]];
    signal tmp_9214[3] <== [tmp_9208[0] + tmp_9213[0], tmp_9208[1] + tmp_9213[1], tmp_9208[2] + tmp_9213[2]];
    signal tmp_9215[3] <== [tmp_9156[0] * 770395855193680981, tmp_9156[1] * 770395855193680981, tmp_9156[2] * 770395855193680981];
    signal tmp_9216[3] <== [tmp_9215[0] + tmp_9196[0], tmp_9215[1] + tmp_9196[1], tmp_9215[2] + tmp_9196[2]];
    signal tmp_9217[3] <== [tmp_9214[0] + tmp_9216[0], tmp_9214[1] + tmp_9216[1], tmp_9214[2] + tmp_9216[2]];
    signal tmp_9218[3] <== [tmp_9159[0] * 3459277367440070515, tmp_9159[1] * 3459277367440070515, tmp_9159[2] * 3459277367440070515];
    signal tmp_9219[3] <== [tmp_9218[0] + tmp_9196[0], tmp_9218[1] + tmp_9196[1], tmp_9218[2] + tmp_9196[2]];
    signal tmp_9220[3] <== [tmp_9217[0] + tmp_9219[0], tmp_9217[1] + tmp_9219[1], tmp_9217[2] + tmp_9219[2]];
    signal tmp_9221[3] <== [tmp_9162[0] * 17087697094293314027, tmp_9162[1] * 17087697094293314027, tmp_9162[2] * 17087697094293314027];
    signal tmp_9222[3] <== [tmp_9221[0] + tmp_9196[0], tmp_9221[1] + tmp_9196[1], tmp_9221[2] + tmp_9196[2]];
    signal tmp_9223[3] <== [tmp_9220[0] + tmp_9222[0], tmp_9220[1] + tmp_9222[1], tmp_9220[2] + tmp_9222[2]];
    signal tmp_9224[3] <== [tmp_9165[0] * 6694380135428747348, tmp_9165[1] * 6694380135428747348, tmp_9165[2] * 6694380135428747348];
    signal tmp_9225[3] <== [tmp_9224[0] + tmp_9196[0], tmp_9224[1] + tmp_9196[1], tmp_9224[2] + tmp_9196[2]];
    signal tmp_9226[3] <== [tmp_9223[0] + tmp_9225[0], tmp_9223[1] + tmp_9225[1], tmp_9223[2] + tmp_9225[2]];
    signal tmp_9227[3] <== [tmp_9168[0] * 2034408310088972836, tmp_9168[1] * 2034408310088972836, tmp_9168[2] * 2034408310088972836];
    signal tmp_9228[3] <== [tmp_9227[0] + tmp_9196[0], tmp_9227[1] + tmp_9196[1], tmp_9227[2] + tmp_9196[2]];
    signal tmp_9229[3] <== [tmp_9226[0] + tmp_9228[0], tmp_9226[1] + tmp_9228[1], tmp_9226[2] + tmp_9228[2]];
    signal tmp_9230[3] <== [tmp_9171[0] * 3434575637390274478, tmp_9171[1] * 3434575637390274478, tmp_9171[2] * 3434575637390274478];
    signal tmp_9231[3] <== [tmp_9230[0] + tmp_9196[0], tmp_9230[1] + tmp_9196[1], tmp_9230[2] + tmp_9196[2]];
    signal tmp_9232[3] <== [tmp_9229[0] + tmp_9231[0], tmp_9229[1] + tmp_9231[1], tmp_9229[2] + tmp_9231[2]];
    signal tmp_9233[3] <== [tmp_9174[0] * 6052753985947965968, tmp_9174[1] * 6052753985947965968, tmp_9174[2] * 6052753985947965968];
    signal tmp_9234[3] <== [tmp_9233[0] + tmp_9196[0], tmp_9233[1] + tmp_9196[1], tmp_9233[2] + tmp_9196[2]];
    signal tmp_9235[3] <== [tmp_9232[0] + tmp_9234[0], tmp_9232[1] + tmp_9234[1], tmp_9232[2] + tmp_9234[2]];
    signal tmp_9236[3] <== [tmp_9177[0] * 13608362914817483670, tmp_9177[1] * 13608362914817483670, tmp_9177[2] * 13608362914817483670];
    signal tmp_9237[3] <== [tmp_9236[0] + tmp_9196[0], tmp_9236[1] + tmp_9196[1], tmp_9236[2] + tmp_9196[2]];
    signal tmp_9238[3] <== [tmp_9235[0] + tmp_9237[0], tmp_9235[1] + tmp_9237[1], tmp_9235[2] + tmp_9237[2]];
    signal tmp_9239[3] <== [tmp_9180[0] * 18163707672964630459, tmp_9180[1] * 18163707672964630459, tmp_9180[2] * 18163707672964630459];
    signal tmp_9240[3] <== [tmp_9239[0] + tmp_9196[0], tmp_9239[1] + tmp_9196[1], tmp_9239[2] + tmp_9196[2]];
    signal tmp_9241[3] <== [tmp_9238[0] + tmp_9240[0], tmp_9238[1] + tmp_9240[1], tmp_9238[2] + tmp_9240[2]];
    signal tmp_9242[3] <== [tmp_9183[0] * 14373610220374016704, tmp_9183[1] * 14373610220374016704, tmp_9183[2] * 14373610220374016704];
    signal tmp_9243[3] <== [tmp_9242[0] + tmp_9196[0], tmp_9242[1] + tmp_9196[1], tmp_9242[2] + tmp_9196[2]];
    signal tmp_9244[3] <== [tmp_9241[0] + tmp_9243[0], tmp_9241[1] + tmp_9243[1], tmp_9241[2] + tmp_9243[2]];
    signal tmp_9245[3] <== [tmp_9186[0] * 6226282807566121054, tmp_9186[1] * 6226282807566121054, tmp_9186[2] * 6226282807566121054];
    signal tmp_9246[3] <== [tmp_9245[0] + tmp_9196[0], tmp_9245[1] + tmp_9196[1], tmp_9245[2] + tmp_9196[2]];
    signal tmp_9247[3] <== [tmp_9244[0] + tmp_9246[0], tmp_9244[1] + tmp_9246[1], tmp_9244[2] + tmp_9246[2]];
    signal tmp_9248[3] <== [tmp_9189[0] * 3643354756180461803, tmp_9189[1] * 3643354756180461803, tmp_9189[2] * 3643354756180461803];
    signal tmp_9249[3] <== [tmp_9248[0] + tmp_9196[0], tmp_9248[1] + tmp_9196[1], tmp_9248[2] + tmp_9196[2]];
    signal tmp_9250[3] <== [tmp_9247[0] + tmp_9249[0], tmp_9247[1] + tmp_9249[1], tmp_9247[2] + tmp_9249[2]];
    signal tmp_9251[3] <== [tmp_9192[0] * 13046961313070095543, tmp_9192[1] * 13046961313070095543, tmp_9192[2] * 13046961313070095543];
    signal tmp_9252[3] <== [tmp_9251[0] + tmp_9196[0], tmp_9251[1] + tmp_9196[1], tmp_9251[2] + tmp_9196[2]];
    signal tmp_9253[3] <== [tmp_9250[0] + tmp_9252[0], tmp_9250[1] + tmp_9252[1], tmp_9250[2] + tmp_9252[2]];
    signal tmp_9254[3] <== [tmp_9195[0] * 8594143216561850811, tmp_9195[1] * 8594143216561850811, tmp_9195[2] * 8594143216561850811];
    signal tmp_9255[3] <== [tmp_9254[0] + tmp_9196[0], tmp_9254[1] + tmp_9196[1], tmp_9254[2] + tmp_9196[2]];
    signal tmp_9256[3] <== [tmp_9253[0] + tmp_9255[0], tmp_9253[1] + tmp_9255[1], tmp_9253[2] + tmp_9255[2]];
    signal tmp_9257[3] <== [tmp_9209[0] + tmp_9256[0], tmp_9209[1] + tmp_9256[1], tmp_9209[2] + tmp_9256[2]];
    signal tmp_9258[3] <== [evals[75][0] - tmp_9257[0], evals[75][1] - tmp_9257[1], evals[75][2] - tmp_9257[2]];
    signal tmp_9259[3] <== CMul()(evals[53], tmp_9258);
    signal tmp_9260[3] <== [tmp_9201[0] + tmp_9259[0], tmp_9201[1] + tmp_9259[1], tmp_9201[2] + tmp_9259[2]];
    signal tmp_9261[3] <== CMul()(challengeQ, tmp_9260);
    signal tmp_9262[3] <== [evals[75][0] + 3256047469251174543, evals[75][1], evals[75][2]];
    signal tmp_9264[3] <== CMul()(tmp_9262, tmp_9262);
    signal tmp_9265[3] <== CMul()(tmp_9264, tmp_9264);
    signal tmp_9266[3] <== CMul()(tmp_9265, tmp_9264);
    signal tmp_9268[3] <== CMul()(tmp_9266, tmp_9262);
    signal tmp_9269[3] <== [tmp_9268[0] * 16040574633112940480, tmp_9268[1] * 16040574633112940480, tmp_9268[2] * 16040574633112940480];
    signal tmp_9272[3] <== [tmp_9213[0] * 14263299814608977431, tmp_9213[1] * 14263299814608977431, tmp_9213[2] * 14263299814608977431];
    signal tmp_9273[3] <== [tmp_9272[0] + tmp_9256[0], tmp_9272[1] + tmp_9256[1], tmp_9272[2] + tmp_9256[2]];
    signal tmp_9274[3] <== [tmp_9268[0] + tmp_9273[0], tmp_9268[1] + tmp_9273[1], tmp_9268[2] + tmp_9273[2]];
    signal tmp_9275[3] <== [tmp_9216[0] * 770395855193680981, tmp_9216[1] * 770395855193680981, tmp_9216[2] * 770395855193680981];
    signal tmp_9276[3] <== [tmp_9275[0] + tmp_9256[0], tmp_9275[1] + tmp_9256[1], tmp_9275[2] + tmp_9256[2]];
    signal tmp_9277[3] <== [tmp_9274[0] + tmp_9276[0], tmp_9274[1] + tmp_9276[1], tmp_9274[2] + tmp_9276[2]];
    signal tmp_9278[3] <== [tmp_9219[0] * 3459277367440070515, tmp_9219[1] * 3459277367440070515, tmp_9219[2] * 3459277367440070515];
    signal tmp_9279[3] <== [tmp_9278[0] + tmp_9256[0], tmp_9278[1] + tmp_9256[1], tmp_9278[2] + tmp_9256[2]];
    signal tmp_9280[3] <== [tmp_9277[0] + tmp_9279[0], tmp_9277[1] + tmp_9279[1], tmp_9277[2] + tmp_9279[2]];
    signal tmp_9281[3] <== [tmp_9222[0] * 17087697094293314027, tmp_9222[1] * 17087697094293314027, tmp_9222[2] * 17087697094293314027];
    signal tmp_9282[3] <== [tmp_9281[0] + tmp_9256[0], tmp_9281[1] + tmp_9256[1], tmp_9281[2] + tmp_9256[2]];
    signal tmp_9283[3] <== [tmp_9280[0] + tmp_9282[0], tmp_9280[1] + tmp_9282[1], tmp_9280[2] + tmp_9282[2]];
    signal tmp_9284[3] <== [tmp_9225[0] * 6694380135428747348, tmp_9225[1] * 6694380135428747348, tmp_9225[2] * 6694380135428747348];
    signal tmp_9285[3] <== [tmp_9284[0] + tmp_9256[0], tmp_9284[1] + tmp_9256[1], tmp_9284[2] + tmp_9256[2]];
    signal tmp_9286[3] <== [tmp_9283[0] + tmp_9285[0], tmp_9283[1] + tmp_9285[1], tmp_9283[2] + tmp_9285[2]];
    signal tmp_9287[3] <== [tmp_9228[0] * 2034408310088972836, tmp_9228[1] * 2034408310088972836, tmp_9228[2] * 2034408310088972836];
    signal tmp_9288[3] <== [tmp_9287[0] + tmp_9256[0], tmp_9287[1] + tmp_9256[1], tmp_9287[2] + tmp_9256[2]];
    signal tmp_9289[3] <== [tmp_9286[0] + tmp_9288[0], tmp_9286[1] + tmp_9288[1], tmp_9286[2] + tmp_9288[2]];
    signal tmp_9290[3] <== [tmp_9231[0] * 3434575637390274478, tmp_9231[1] * 3434575637390274478, tmp_9231[2] * 3434575637390274478];
    signal tmp_9291[3] <== [tmp_9290[0] + tmp_9256[0], tmp_9290[1] + tmp_9256[1], tmp_9290[2] + tmp_9256[2]];
    signal tmp_9292[3] <== [tmp_9289[0] + tmp_9291[0], tmp_9289[1] + tmp_9291[1], tmp_9289[2] + tmp_9291[2]];
    signal tmp_9293[3] <== [tmp_9234[0] * 6052753985947965968, tmp_9234[1] * 6052753985947965968, tmp_9234[2] * 6052753985947965968];
    signal tmp_9294[3] <== [tmp_9293[0] + tmp_9256[0], tmp_9293[1] + tmp_9256[1], tmp_9293[2] + tmp_9256[2]];
    signal tmp_9295[3] <== [tmp_9292[0] + tmp_9294[0], tmp_9292[1] + tmp_9294[1], tmp_9292[2] + tmp_9294[2]];
    signal tmp_9296[3] <== [tmp_9237[0] * 13608362914817483670, tmp_9237[1] * 13608362914817483670, tmp_9237[2] * 13608362914817483670];
    signal tmp_9297[3] <== [tmp_9296[0] + tmp_9256[0], tmp_9296[1] + tmp_9256[1], tmp_9296[2] + tmp_9256[2]];
    signal tmp_9298[3] <== [tmp_9295[0] + tmp_9297[0], tmp_9295[1] + tmp_9297[1], tmp_9295[2] + tmp_9297[2]];
    signal tmp_9299[3] <== [tmp_9240[0] * 18163707672964630459, tmp_9240[1] * 18163707672964630459, tmp_9240[2] * 18163707672964630459];
    signal tmp_9300[3] <== [tmp_9299[0] + tmp_9256[0], tmp_9299[1] + tmp_9256[1], tmp_9299[2] + tmp_9256[2]];
    signal tmp_9301[3] <== [tmp_9298[0] + tmp_9300[0], tmp_9298[1] + tmp_9300[1], tmp_9298[2] + tmp_9300[2]];
    signal tmp_9302[3] <== [tmp_9243[0] * 14373610220374016704, tmp_9243[1] * 14373610220374016704, tmp_9243[2] * 14373610220374016704];
    signal tmp_9303[3] <== [tmp_9302[0] + tmp_9256[0], tmp_9302[1] + tmp_9256[1], tmp_9302[2] + tmp_9256[2]];
    signal tmp_9304[3] <== [tmp_9301[0] + tmp_9303[0], tmp_9301[1] + tmp_9303[1], tmp_9301[2] + tmp_9303[2]];
    signal tmp_9305[3] <== [tmp_9246[0] * 6226282807566121054, tmp_9246[1] * 6226282807566121054, tmp_9246[2] * 6226282807566121054];
    signal tmp_9306[3] <== [tmp_9305[0] + tmp_9256[0], tmp_9305[1] + tmp_9256[1], tmp_9305[2] + tmp_9256[2]];
    signal tmp_9307[3] <== [tmp_9304[0] + tmp_9306[0], tmp_9304[1] + tmp_9306[1], tmp_9304[2] + tmp_9306[2]];
    signal tmp_9308[3] <== [tmp_9249[0] * 3643354756180461803, tmp_9249[1] * 3643354756180461803, tmp_9249[2] * 3643354756180461803];
    signal tmp_9309[3] <== [tmp_9308[0] + tmp_9256[0], tmp_9308[1] + tmp_9256[1], tmp_9308[2] + tmp_9256[2]];
    signal tmp_9310[3] <== [tmp_9307[0] + tmp_9309[0], tmp_9307[1] + tmp_9309[1], tmp_9307[2] + tmp_9309[2]];
    signal tmp_9311[3] <== [tmp_9252[0] * 13046961313070095543, tmp_9252[1] * 13046961313070095543, tmp_9252[2] * 13046961313070095543];
    signal tmp_9312[3] <== [tmp_9311[0] + tmp_9256[0], tmp_9311[1] + tmp_9256[1], tmp_9311[2] + tmp_9256[2]];
    signal tmp_9313[3] <== [tmp_9310[0] + tmp_9312[0], tmp_9310[1] + tmp_9312[1], tmp_9310[2] + tmp_9312[2]];
    signal tmp_9314[3] <== [tmp_9255[0] * 8594143216561850811, tmp_9255[1] * 8594143216561850811, tmp_9255[2] * 8594143216561850811];
    signal tmp_9315[3] <== [tmp_9314[0] + tmp_9256[0], tmp_9314[1] + tmp_9256[1], tmp_9314[2] + tmp_9256[2]];
    signal tmp_9316[3] <== [tmp_9313[0] + tmp_9315[0], tmp_9313[1] + tmp_9315[1], tmp_9313[2] + tmp_9315[2]];
    signal tmp_9317[3] <== [tmp_9269[0] + tmp_9316[0], tmp_9269[1] + tmp_9316[1], tmp_9269[2] + tmp_9316[2]];
    signal tmp_9318[3] <== [evals[76][0] - tmp_9317[0], evals[76][1] - tmp_9317[1], evals[76][2] - tmp_9317[2]];
    signal tmp_9319[3] <== CMul()(evals[53], tmp_9318);
    signal tmp_9320[3] <== [tmp_9261[0] + tmp_9319[0], tmp_9261[1] + tmp_9319[1], tmp_9261[2] + tmp_9319[2]];
    signal tmp_9321[3] <== CMul()(challengeQ, tmp_9320);
    signal tmp_9322[3] <== [evals[76][0] + 3015723851705964382, evals[76][1], evals[76][2]];
    signal tmp_9324[3] <== CMul()(tmp_9322, tmp_9322);
    signal tmp_9325[3] <== CMul()(tmp_9324, tmp_9324);
    signal tmp_9326[3] <== CMul()(tmp_9325, tmp_9324);
    signal tmp_9328[3] <== CMul()(tmp_9326, tmp_9322);
    signal tmp_9329[3] <== [tmp_9328[0] * 16040574633112940480, tmp_9328[1] * 16040574633112940480, tmp_9328[2] * 16040574633112940480];
    signal tmp_9332[3] <== [tmp_9273[0] * 14263299814608977431, tmp_9273[1] * 14263299814608977431, tmp_9273[2] * 14263299814608977431];
    signal tmp_9333[3] <== [tmp_9332[0] + tmp_9316[0], tmp_9332[1] + tmp_9316[1], tmp_9332[2] + tmp_9316[2]];
    signal tmp_9334[3] <== [tmp_9328[0] + tmp_9333[0], tmp_9328[1] + tmp_9333[1], tmp_9328[2] + tmp_9333[2]];
    signal tmp_9335[3] <== [tmp_9276[0] * 770395855193680981, tmp_9276[1] * 770395855193680981, tmp_9276[2] * 770395855193680981];
    signal tmp_9336[3] <== [tmp_9335[0] + tmp_9316[0], tmp_9335[1] + tmp_9316[1], tmp_9335[2] + tmp_9316[2]];
    signal tmp_9337[3] <== [tmp_9334[0] + tmp_9336[0], tmp_9334[1] + tmp_9336[1], tmp_9334[2] + tmp_9336[2]];
    signal tmp_9338[3] <== [tmp_9279[0] * 3459277367440070515, tmp_9279[1] * 3459277367440070515, tmp_9279[2] * 3459277367440070515];
    signal tmp_9339[3] <== [tmp_9338[0] + tmp_9316[0], tmp_9338[1] + tmp_9316[1], tmp_9338[2] + tmp_9316[2]];
    signal tmp_9340[3] <== [tmp_9337[0] + tmp_9339[0], tmp_9337[1] + tmp_9339[1], tmp_9337[2] + tmp_9339[2]];
    signal tmp_9341[3] <== [tmp_9282[0] * 17087697094293314027, tmp_9282[1] * 17087697094293314027, tmp_9282[2] * 17087697094293314027];
    signal tmp_9342[3] <== [tmp_9341[0] + tmp_9316[0], tmp_9341[1] + tmp_9316[1], tmp_9341[2] + tmp_9316[2]];
    signal tmp_9343[3] <== [tmp_9340[0] + tmp_9342[0], tmp_9340[1] + tmp_9342[1], tmp_9340[2] + tmp_9342[2]];
    signal tmp_9344[3] <== [tmp_9285[0] * 6694380135428747348, tmp_9285[1] * 6694380135428747348, tmp_9285[2] * 6694380135428747348];
    signal tmp_9345[3] <== [tmp_9344[0] + tmp_9316[0], tmp_9344[1] + tmp_9316[1], tmp_9344[2] + tmp_9316[2]];
    signal tmp_9346[3] <== [tmp_9343[0] + tmp_9345[0], tmp_9343[1] + tmp_9345[1], tmp_9343[2] + tmp_9345[2]];
    signal tmp_9347[3] <== [tmp_9288[0] * 2034408310088972836, tmp_9288[1] * 2034408310088972836, tmp_9288[2] * 2034408310088972836];
    signal tmp_9348[3] <== [tmp_9347[0] + tmp_9316[0], tmp_9347[1] + tmp_9316[1], tmp_9347[2] + tmp_9316[2]];
    signal tmp_9349[3] <== [tmp_9346[0] + tmp_9348[0], tmp_9346[1] + tmp_9348[1], tmp_9346[2] + tmp_9348[2]];
    signal tmp_9350[3] <== [tmp_9291[0] * 3434575637390274478, tmp_9291[1] * 3434575637390274478, tmp_9291[2] * 3434575637390274478];
    signal tmp_9351[3] <== [tmp_9350[0] + tmp_9316[0], tmp_9350[1] + tmp_9316[1], tmp_9350[2] + tmp_9316[2]];
    signal tmp_9352[3] <== [tmp_9349[0] + tmp_9351[0], tmp_9349[1] + tmp_9351[1], tmp_9349[2] + tmp_9351[2]];
    signal tmp_9353[3] <== [tmp_9294[0] * 6052753985947965968, tmp_9294[1] * 6052753985947965968, tmp_9294[2] * 6052753985947965968];
    signal tmp_9354[3] <== [tmp_9353[0] + tmp_9316[0], tmp_9353[1] + tmp_9316[1], tmp_9353[2] + tmp_9316[2]];
    signal tmp_9355[3] <== [tmp_9352[0] + tmp_9354[0], tmp_9352[1] + tmp_9354[1], tmp_9352[2] + tmp_9354[2]];
    signal tmp_9356[3] <== [tmp_9297[0] * 13608362914817483670, tmp_9297[1] * 13608362914817483670, tmp_9297[2] * 13608362914817483670];
    signal tmp_9357[3] <== [tmp_9356[0] + tmp_9316[0], tmp_9356[1] + tmp_9316[1], tmp_9356[2] + tmp_9316[2]];
    signal tmp_9358[3] <== [tmp_9355[0] + tmp_9357[0], tmp_9355[1] + tmp_9357[1], tmp_9355[2] + tmp_9357[2]];
    signal tmp_9359[3] <== [tmp_9300[0] * 18163707672964630459, tmp_9300[1] * 18163707672964630459, tmp_9300[2] * 18163707672964630459];
    signal tmp_9360[3] <== [tmp_9359[0] + tmp_9316[0], tmp_9359[1] + tmp_9316[1], tmp_9359[2] + tmp_9316[2]];
    signal tmp_9361[3] <== [tmp_9358[0] + tmp_9360[0], tmp_9358[1] + tmp_9360[1], tmp_9358[2] + tmp_9360[2]];
    signal tmp_9362[3] <== [tmp_9303[0] * 14373610220374016704, tmp_9303[1] * 14373610220374016704, tmp_9303[2] * 14373610220374016704];
    signal tmp_9363[3] <== [tmp_9362[0] + tmp_9316[0], tmp_9362[1] + tmp_9316[1], tmp_9362[2] + tmp_9316[2]];
    signal tmp_9364[3] <== [tmp_9361[0] + tmp_9363[0], tmp_9361[1] + tmp_9363[1], tmp_9361[2] + tmp_9363[2]];
    signal tmp_9365[3] <== [tmp_9306[0] * 6226282807566121054, tmp_9306[1] * 6226282807566121054, tmp_9306[2] * 6226282807566121054];
    signal tmp_9366[3] <== [tmp_9365[0] + tmp_9316[0], tmp_9365[1] + tmp_9316[1], tmp_9365[2] + tmp_9316[2]];
    signal tmp_9367[3] <== [tmp_9364[0] + tmp_9366[0], tmp_9364[1] + tmp_9366[1], tmp_9364[2] + tmp_9366[2]];
    signal tmp_9368[3] <== [tmp_9309[0] * 3643354756180461803, tmp_9309[1] * 3643354756180461803, tmp_9309[2] * 3643354756180461803];
    signal tmp_9369[3] <== [tmp_9368[0] + tmp_9316[0], tmp_9368[1] + tmp_9316[1], tmp_9368[2] + tmp_9316[2]];
    signal tmp_9370[3] <== [tmp_9367[0] + tmp_9369[0], tmp_9367[1] + tmp_9369[1], tmp_9367[2] + tmp_9369[2]];
    signal tmp_9371[3] <== [tmp_9312[0] * 13046961313070095543, tmp_9312[1] * 13046961313070095543, tmp_9312[2] * 13046961313070095543];
    signal tmp_9372[3] <== [tmp_9371[0] + tmp_9316[0], tmp_9371[1] + tmp_9316[1], tmp_9371[2] + tmp_9316[2]];
    signal tmp_9373[3] <== [tmp_9370[0] + tmp_9372[0], tmp_9370[1] + tmp_9372[1], tmp_9370[2] + tmp_9372[2]];
    signal tmp_9374[3] <== [tmp_9315[0] * 8594143216561850811, tmp_9315[1] * 8594143216561850811, tmp_9315[2] * 8594143216561850811];
    signal tmp_9375[3] <== [tmp_9374[0] + tmp_9316[0], tmp_9374[1] + tmp_9316[1], tmp_9374[2] + tmp_9316[2]];
    signal tmp_9376[3] <== [tmp_9373[0] + tmp_9375[0], tmp_9373[1] + tmp_9375[1], tmp_9373[2] + tmp_9375[2]];
    signal tmp_9377[3] <== [tmp_9329[0] + tmp_9376[0], tmp_9329[1] + tmp_9376[1], tmp_9329[2] + tmp_9376[2]];
    signal tmp_9378[3] <== [evals[77][0] - tmp_9377[0], evals[77][1] - tmp_9377[1], evals[77][2] - tmp_9377[2]];
    signal tmp_9379[3] <== CMul()(evals[53], tmp_9378);
    signal tmp_9380[3] <== [tmp_9321[0] + tmp_9379[0], tmp_9321[1] + tmp_9379[1], tmp_9321[2] + tmp_9379[2]];
    signal tmp_9381[3] <== CMul()(challengeQ, tmp_9380);
    signal tmp_9382[3] <== [evals[77][0] + 5177282273995529875, evals[77][1], evals[77][2]];
    signal tmp_9384[3] <== CMul()(tmp_9382, tmp_9382);
    signal tmp_9385[3] <== CMul()(tmp_9384, tmp_9384);
    signal tmp_9386[3] <== CMul()(tmp_9385, tmp_9384);
    signal tmp_9388[3] <== CMul()(tmp_9386, tmp_9382);
    signal tmp_9389[3] <== [tmp_9388[0] * 16040574633112940480, tmp_9388[1] * 16040574633112940480, tmp_9388[2] * 16040574633112940480];
    signal tmp_9392[3] <== [tmp_9333[0] * 14263299814608977431, tmp_9333[1] * 14263299814608977431, tmp_9333[2] * 14263299814608977431];
    signal tmp_9393[3] <== [tmp_9392[0] + tmp_9376[0], tmp_9392[1] + tmp_9376[1], tmp_9392[2] + tmp_9376[2]];
    signal tmp_9394[3] <== [tmp_9388[0] + tmp_9393[0], tmp_9388[1] + tmp_9393[1], tmp_9388[2] + tmp_9393[2]];
    signal tmp_9395[3] <== [tmp_9336[0] * 770395855193680981, tmp_9336[1] * 770395855193680981, tmp_9336[2] * 770395855193680981];
    signal tmp_9396[3] <== [tmp_9395[0] + tmp_9376[0], tmp_9395[1] + tmp_9376[1], tmp_9395[2] + tmp_9376[2]];
    signal tmp_9397[3] <== [tmp_9394[0] + tmp_9396[0], tmp_9394[1] + tmp_9396[1], tmp_9394[2] + tmp_9396[2]];
    signal tmp_9398[3] <== [tmp_9339[0] * 3459277367440070515, tmp_9339[1] * 3459277367440070515, tmp_9339[2] * 3459277367440070515];
    signal tmp_9399[3] <== [tmp_9398[0] + tmp_9376[0], tmp_9398[1] + tmp_9376[1], tmp_9398[2] + tmp_9376[2]];
    signal tmp_9400[3] <== [tmp_9397[0] + tmp_9399[0], tmp_9397[1] + tmp_9399[1], tmp_9397[2] + tmp_9399[2]];
    signal tmp_9401[3] <== [tmp_9342[0] * 17087697094293314027, tmp_9342[1] * 17087697094293314027, tmp_9342[2] * 17087697094293314027];
    signal tmp_9402[3] <== [tmp_9401[0] + tmp_9376[0], tmp_9401[1] + tmp_9376[1], tmp_9401[2] + tmp_9376[2]];
    signal tmp_9403[3] <== [tmp_9400[0] + tmp_9402[0], tmp_9400[1] + tmp_9402[1], tmp_9400[2] + tmp_9402[2]];
    signal tmp_9404[3] <== [tmp_9345[0] * 6694380135428747348, tmp_9345[1] * 6694380135428747348, tmp_9345[2] * 6694380135428747348];
    signal tmp_9405[3] <== [tmp_9404[0] + tmp_9376[0], tmp_9404[1] + tmp_9376[1], tmp_9404[2] + tmp_9376[2]];
    signal tmp_9406[3] <== [tmp_9403[0] + tmp_9405[0], tmp_9403[1] + tmp_9405[1], tmp_9403[2] + tmp_9405[2]];
    signal tmp_9407[3] <== [tmp_9348[0] * 2034408310088972836, tmp_9348[1] * 2034408310088972836, tmp_9348[2] * 2034408310088972836];
    signal tmp_9408[3] <== [tmp_9407[0] + tmp_9376[0], tmp_9407[1] + tmp_9376[1], tmp_9407[2] + tmp_9376[2]];
    signal tmp_9409[3] <== [tmp_9406[0] + tmp_9408[0], tmp_9406[1] + tmp_9408[1], tmp_9406[2] + tmp_9408[2]];
    signal tmp_9410[3] <== [tmp_9351[0] * 3434575637390274478, tmp_9351[1] * 3434575637390274478, tmp_9351[2] * 3434575637390274478];
    signal tmp_9411[3] <== [tmp_9410[0] + tmp_9376[0], tmp_9410[1] + tmp_9376[1], tmp_9410[2] + tmp_9376[2]];
    signal tmp_9412[3] <== [tmp_9409[0] + tmp_9411[0], tmp_9409[1] + tmp_9411[1], tmp_9409[2] + tmp_9411[2]];
    signal tmp_9413[3] <== [tmp_9354[0] * 6052753985947965968, tmp_9354[1] * 6052753985947965968, tmp_9354[2] * 6052753985947965968];
    signal tmp_9414[3] <== [tmp_9413[0] + tmp_9376[0], tmp_9413[1] + tmp_9376[1], tmp_9413[2] + tmp_9376[2]];
    signal tmp_9415[3] <== [tmp_9412[0] + tmp_9414[0], tmp_9412[1] + tmp_9414[1], tmp_9412[2] + tmp_9414[2]];
    signal tmp_9416[3] <== [tmp_9357[0] * 13608362914817483670, tmp_9357[1] * 13608362914817483670, tmp_9357[2] * 13608362914817483670];
    signal tmp_9417[3] <== [tmp_9416[0] + tmp_9376[0], tmp_9416[1] + tmp_9376[1], tmp_9416[2] + tmp_9376[2]];
    signal tmp_9418[3] <== [tmp_9415[0] + tmp_9417[0], tmp_9415[1] + tmp_9417[1], tmp_9415[2] + tmp_9417[2]];
    signal tmp_9419[3] <== [tmp_9360[0] * 18163707672964630459, tmp_9360[1] * 18163707672964630459, tmp_9360[2] * 18163707672964630459];
    signal tmp_9420[3] <== [tmp_9419[0] + tmp_9376[0], tmp_9419[1] + tmp_9376[1], tmp_9419[2] + tmp_9376[2]];
    signal tmp_9421[3] <== [tmp_9418[0] + tmp_9420[0], tmp_9418[1] + tmp_9420[1], tmp_9418[2] + tmp_9420[2]];
    signal tmp_9422[3] <== [tmp_9363[0] * 14373610220374016704, tmp_9363[1] * 14373610220374016704, tmp_9363[2] * 14373610220374016704];
    signal tmp_9423[3] <== [tmp_9422[0] + tmp_9376[0], tmp_9422[1] + tmp_9376[1], tmp_9422[2] + tmp_9376[2]];
    signal tmp_9424[3] <== [tmp_9421[0] + tmp_9423[0], tmp_9421[1] + tmp_9423[1], tmp_9421[2] + tmp_9423[2]];
    signal tmp_9425[3] <== [tmp_9366[0] * 6226282807566121054, tmp_9366[1] * 6226282807566121054, tmp_9366[2] * 6226282807566121054];
    signal tmp_9426[3] <== [tmp_9425[0] + tmp_9376[0], tmp_9425[1] + tmp_9376[1], tmp_9425[2] + tmp_9376[2]];
    signal tmp_9427[3] <== [tmp_9424[0] + tmp_9426[0], tmp_9424[1] + tmp_9426[1], tmp_9424[2] + tmp_9426[2]];
    signal tmp_9428[3] <== [tmp_9369[0] * 3643354756180461803, tmp_9369[1] * 3643354756180461803, tmp_9369[2] * 3643354756180461803];
    signal tmp_9429[3] <== [tmp_9428[0] + tmp_9376[0], tmp_9428[1] + tmp_9376[1], tmp_9428[2] + tmp_9376[2]];
    signal tmp_9430[3] <== [tmp_9427[0] + tmp_9429[0], tmp_9427[1] + tmp_9429[1], tmp_9427[2] + tmp_9429[2]];
    signal tmp_9431[3] <== [tmp_9372[0] * 13046961313070095543, tmp_9372[1] * 13046961313070095543, tmp_9372[2] * 13046961313070095543];
    signal tmp_9432[3] <== [tmp_9431[0] + tmp_9376[0], tmp_9431[1] + tmp_9376[1], tmp_9431[2] + tmp_9376[2]];
    signal tmp_9433[3] <== [tmp_9430[0] + tmp_9432[0], tmp_9430[1] + tmp_9432[1], tmp_9430[2] + tmp_9432[2]];
    signal tmp_9434[3] <== [tmp_9375[0] * 8594143216561850811, tmp_9375[1] * 8594143216561850811, tmp_9375[2] * 8594143216561850811];
    signal tmp_9435[3] <== [tmp_9434[0] + tmp_9376[0], tmp_9434[1] + tmp_9376[1], tmp_9434[2] + tmp_9376[2]];
    signal tmp_9436[3] <== [tmp_9433[0] + tmp_9435[0], tmp_9433[1] + tmp_9435[1], tmp_9433[2] + tmp_9435[2]];
    signal tmp_9437[3] <== [tmp_9389[0] + tmp_9436[0], tmp_9389[1] + tmp_9436[1], tmp_9389[2] + tmp_9436[2]];
    signal tmp_9438[3] <== [evals[78][0] - tmp_9437[0], evals[78][1] - tmp_9437[1], evals[78][2] - tmp_9437[2]];
    signal tmp_9439[3] <== CMul()(evals[53], tmp_9438);
    signal tmp_9440[3] <== [tmp_9381[0] + tmp_9439[0], tmp_9381[1] + tmp_9439[1], tmp_9381[2] + tmp_9439[2]];
    signal tmp_9441[3] <== CMul()(challengeQ, tmp_9440);
    signal tmp_9442[3] <== [evals[78][0] + 16035152896984012190, evals[78][1], evals[78][2]];
    signal tmp_9444[3] <== CMul()(tmp_9442, tmp_9442);
    signal tmp_9445[3] <== CMul()(tmp_9444, tmp_9444);
    signal tmp_9446[3] <== CMul()(tmp_9445, tmp_9444);
    signal tmp_9448[3] <== CMul()(tmp_9446, tmp_9442);
    signal tmp_9449[3] <== [tmp_9448[0] * 16040574633112940480, tmp_9448[1] * 16040574633112940480, tmp_9448[2] * 16040574633112940480];
    signal tmp_9452[3] <== [tmp_9393[0] * 14263299814608977431, tmp_9393[1] * 14263299814608977431, tmp_9393[2] * 14263299814608977431];
    signal tmp_9453[3] <== [tmp_9452[0] + tmp_9436[0], tmp_9452[1] + tmp_9436[1], tmp_9452[2] + tmp_9436[2]];
    signal tmp_9454[3] <== [tmp_9448[0] + tmp_9453[0], tmp_9448[1] + tmp_9453[1], tmp_9448[2] + tmp_9453[2]];
    signal tmp_9455[3] <== [tmp_9396[0] * 770395855193680981, tmp_9396[1] * 770395855193680981, tmp_9396[2] * 770395855193680981];
    signal tmp_9456[3] <== [tmp_9455[0] + tmp_9436[0], tmp_9455[1] + tmp_9436[1], tmp_9455[2] + tmp_9436[2]];
    signal tmp_9457[3] <== [tmp_9454[0] + tmp_9456[0], tmp_9454[1] + tmp_9456[1], tmp_9454[2] + tmp_9456[2]];
    signal tmp_9458[3] <== [tmp_9399[0] * 3459277367440070515, tmp_9399[1] * 3459277367440070515, tmp_9399[2] * 3459277367440070515];
    signal tmp_9459[3] <== [tmp_9458[0] + tmp_9436[0], tmp_9458[1] + tmp_9436[1], tmp_9458[2] + tmp_9436[2]];
    signal tmp_9460[3] <== [tmp_9457[0] + tmp_9459[0], tmp_9457[1] + tmp_9459[1], tmp_9457[2] + tmp_9459[2]];
    signal tmp_9461[3] <== [tmp_9402[0] * 17087697094293314027, tmp_9402[1] * 17087697094293314027, tmp_9402[2] * 17087697094293314027];
    signal tmp_9462[3] <== [tmp_9461[0] + tmp_9436[0], tmp_9461[1] + tmp_9436[1], tmp_9461[2] + tmp_9436[2]];
    signal tmp_9463[3] <== [tmp_9460[0] + tmp_9462[0], tmp_9460[1] + tmp_9462[1], tmp_9460[2] + tmp_9462[2]];
    signal tmp_9464[3] <== [tmp_9405[0] * 6694380135428747348, tmp_9405[1] * 6694380135428747348, tmp_9405[2] * 6694380135428747348];
    signal tmp_9465[3] <== [tmp_9464[0] + tmp_9436[0], tmp_9464[1] + tmp_9436[1], tmp_9464[2] + tmp_9436[2]];
    signal tmp_9466[3] <== [tmp_9463[0] + tmp_9465[0], tmp_9463[1] + tmp_9465[1], tmp_9463[2] + tmp_9465[2]];
    signal tmp_9467[3] <== [tmp_9408[0] * 2034408310088972836, tmp_9408[1] * 2034408310088972836, tmp_9408[2] * 2034408310088972836];
    signal tmp_9468[3] <== [tmp_9467[0] + tmp_9436[0], tmp_9467[1] + tmp_9436[1], tmp_9467[2] + tmp_9436[2]];
    signal tmp_9469[3] <== [tmp_9466[0] + tmp_9468[0], tmp_9466[1] + tmp_9468[1], tmp_9466[2] + tmp_9468[2]];
    signal tmp_9470[3] <== [tmp_9411[0] * 3434575637390274478, tmp_9411[1] * 3434575637390274478, tmp_9411[2] * 3434575637390274478];
    signal tmp_9471[3] <== [tmp_9470[0] + tmp_9436[0], tmp_9470[1] + tmp_9436[1], tmp_9470[2] + tmp_9436[2]];
    signal tmp_9472[3] <== [tmp_9469[0] + tmp_9471[0], tmp_9469[1] + tmp_9471[1], tmp_9469[2] + tmp_9471[2]];
    signal tmp_9473[3] <== [tmp_9414[0] * 6052753985947965968, tmp_9414[1] * 6052753985947965968, tmp_9414[2] * 6052753985947965968];
    signal tmp_9474[3] <== [tmp_9473[0] + tmp_9436[0], tmp_9473[1] + tmp_9436[1], tmp_9473[2] + tmp_9436[2]];
    signal tmp_9475[3] <== [tmp_9472[0] + tmp_9474[0], tmp_9472[1] + tmp_9474[1], tmp_9472[2] + tmp_9474[2]];
    signal tmp_9476[3] <== [tmp_9417[0] * 13608362914817483670, tmp_9417[1] * 13608362914817483670, tmp_9417[2] * 13608362914817483670];
    signal tmp_9477[3] <== [tmp_9476[0] + tmp_9436[0], tmp_9476[1] + tmp_9436[1], tmp_9476[2] + tmp_9436[2]];
    signal tmp_9478[3] <== [tmp_9475[0] + tmp_9477[0], tmp_9475[1] + tmp_9477[1], tmp_9475[2] + tmp_9477[2]];
    signal tmp_9479[3] <== [tmp_9420[0] * 18163707672964630459, tmp_9420[1] * 18163707672964630459, tmp_9420[2] * 18163707672964630459];
    signal tmp_9480[3] <== [tmp_9479[0] + tmp_9436[0], tmp_9479[1] + tmp_9436[1], tmp_9479[2] + tmp_9436[2]];
    signal tmp_9481[3] <== [tmp_9478[0] + tmp_9480[0], tmp_9478[1] + tmp_9480[1], tmp_9478[2] + tmp_9480[2]];
    signal tmp_9482[3] <== [tmp_9423[0] * 14373610220374016704, tmp_9423[1] * 14373610220374016704, tmp_9423[2] * 14373610220374016704];
    signal tmp_9483[3] <== [tmp_9482[0] + tmp_9436[0], tmp_9482[1] + tmp_9436[1], tmp_9482[2] + tmp_9436[2]];
    signal tmp_9484[3] <== [tmp_9481[0] + tmp_9483[0], tmp_9481[1] + tmp_9483[1], tmp_9481[2] + tmp_9483[2]];
    signal tmp_9485[3] <== [tmp_9426[0] * 6226282807566121054, tmp_9426[1] * 6226282807566121054, tmp_9426[2] * 6226282807566121054];
    signal tmp_9486[3] <== [tmp_9485[0] + tmp_9436[0], tmp_9485[1] + tmp_9436[1], tmp_9485[2] + tmp_9436[2]];
    signal tmp_9487[3] <== [tmp_9484[0] + tmp_9486[0], tmp_9484[1] + tmp_9486[1], tmp_9484[2] + tmp_9486[2]];
    signal tmp_9488[3] <== [tmp_9429[0] * 3643354756180461803, tmp_9429[1] * 3643354756180461803, tmp_9429[2] * 3643354756180461803];
    signal tmp_9489[3] <== [tmp_9488[0] + tmp_9436[0], tmp_9488[1] + tmp_9436[1], tmp_9488[2] + tmp_9436[2]];
    signal tmp_9490[3] <== [tmp_9487[0] + tmp_9489[0], tmp_9487[1] + tmp_9489[1], tmp_9487[2] + tmp_9489[2]];
    signal tmp_9491[3] <== [tmp_9432[0] * 13046961313070095543, tmp_9432[1] * 13046961313070095543, tmp_9432[2] * 13046961313070095543];
    signal tmp_9492[3] <== [tmp_9491[0] + tmp_9436[0], tmp_9491[1] + tmp_9436[1], tmp_9491[2] + tmp_9436[2]];
    signal tmp_9493[3] <== [tmp_9490[0] + tmp_9492[0], tmp_9490[1] + tmp_9492[1], tmp_9490[2] + tmp_9492[2]];
    signal tmp_9494[3] <== [tmp_9435[0] * 8594143216561850811, tmp_9435[1] * 8594143216561850811, tmp_9435[2] * 8594143216561850811];
    signal tmp_9495[3] <== [tmp_9494[0] + tmp_9436[0], tmp_9494[1] + tmp_9436[1], tmp_9494[2] + tmp_9436[2]];
    signal tmp_9496[3] <== [tmp_9493[0] + tmp_9495[0], tmp_9493[1] + tmp_9495[1], tmp_9493[2] + tmp_9495[2]];
    signal tmp_9497[3] <== [tmp_9449[0] + tmp_9496[0], tmp_9449[1] + tmp_9496[1], tmp_9449[2] + tmp_9496[2]];
    signal tmp_9498[3] <== [evals[127][0] - tmp_9497[0], evals[127][1] - tmp_9497[1], evals[127][2] - tmp_9497[2]];
    signal tmp_9499[3] <== CMul()(evals[53], tmp_9498);
    signal tmp_9500[3] <== [tmp_9441[0] + tmp_9499[0], tmp_9441[1] + tmp_9499[1], tmp_9441[2] + tmp_9499[2]];
    signal tmp_9501[3] <== CMul()(challengeQ, tmp_9500);
    signal tmp_9502[3] <== [tmp_9453[0] * 14263299814608977431, tmp_9453[1] * 14263299814608977431, tmp_9453[2] * 14263299814608977431];
    signal tmp_9503[3] <== [tmp_9502[0] + tmp_9496[0], tmp_9502[1] + tmp_9496[1], tmp_9502[2] + tmp_9496[2]];
    signal tmp_9504[3] <== [evals[128][0] - tmp_9503[0], evals[128][1] - tmp_9503[1], evals[128][2] - tmp_9503[2]];
    signal tmp_9505[3] <== CMul()(evals[53], tmp_9504);
    signal tmp_9506[3] <== [tmp_9501[0] + tmp_9505[0], tmp_9501[1] + tmp_9505[1], tmp_9501[2] + tmp_9505[2]];
    signal tmp_9507[3] <== CMul()(challengeQ, tmp_9506);
    signal tmp_9508[3] <== [tmp_9456[0] * 770395855193680981, tmp_9456[1] * 770395855193680981, tmp_9456[2] * 770395855193680981];
    signal tmp_9509[3] <== [tmp_9508[0] + tmp_9496[0], tmp_9508[1] + tmp_9496[1], tmp_9508[2] + tmp_9496[2]];
    signal tmp_9510[3] <== [evals[129][0] - tmp_9509[0], evals[129][1] - tmp_9509[1], evals[129][2] - tmp_9509[2]];
    signal tmp_9511[3] <== CMul()(evals[53], tmp_9510);
    signal tmp_9512[3] <== [tmp_9507[0] + tmp_9511[0], tmp_9507[1] + tmp_9511[1], tmp_9507[2] + tmp_9511[2]];
    signal tmp_9513[3] <== CMul()(challengeQ, tmp_9512);
    signal tmp_9514[3] <== [tmp_9459[0] * 3459277367440070515, tmp_9459[1] * 3459277367440070515, tmp_9459[2] * 3459277367440070515];
    signal tmp_9515[3] <== [tmp_9514[0] + tmp_9496[0], tmp_9514[1] + tmp_9496[1], tmp_9514[2] + tmp_9496[2]];
    signal tmp_9516[3] <== [evals[130][0] - tmp_9515[0], evals[130][1] - tmp_9515[1], evals[130][2] - tmp_9515[2]];
    signal tmp_9517[3] <== CMul()(evals[53], tmp_9516);
    signal tmp_9518[3] <== [tmp_9513[0] + tmp_9517[0], tmp_9513[1] + tmp_9517[1], tmp_9513[2] + tmp_9517[2]];
    signal tmp_9519[3] <== CMul()(challengeQ, tmp_9518);
    signal tmp_9520[3] <== [tmp_9462[0] * 17087697094293314027, tmp_9462[1] * 17087697094293314027, tmp_9462[2] * 17087697094293314027];
    signal tmp_9521[3] <== [tmp_9520[0] + tmp_9496[0], tmp_9520[1] + tmp_9496[1], tmp_9520[2] + tmp_9496[2]];
    signal tmp_9522[3] <== [evals[131][0] - tmp_9521[0], evals[131][1] - tmp_9521[1], evals[131][2] - tmp_9521[2]];
    signal tmp_9523[3] <== CMul()(evals[53], tmp_9522);
    signal tmp_9524[3] <== [tmp_9519[0] + tmp_9523[0], tmp_9519[1] + tmp_9523[1], tmp_9519[2] + tmp_9523[2]];
    signal tmp_9525[3] <== CMul()(challengeQ, tmp_9524);
    signal tmp_9526[3] <== [tmp_9465[0] * 6694380135428747348, tmp_9465[1] * 6694380135428747348, tmp_9465[2] * 6694380135428747348];
    signal tmp_9527[3] <== [tmp_9526[0] + tmp_9496[0], tmp_9526[1] + tmp_9496[1], tmp_9526[2] + tmp_9496[2]];
    signal tmp_9528[3] <== [evals[132][0] - tmp_9527[0], evals[132][1] - tmp_9527[1], evals[132][2] - tmp_9527[2]];
    signal tmp_9529[3] <== CMul()(evals[53], tmp_9528);
    signal tmp_9530[3] <== [tmp_9525[0] + tmp_9529[0], tmp_9525[1] + tmp_9529[1], tmp_9525[2] + tmp_9529[2]];
    signal tmp_9531[3] <== CMul()(challengeQ, tmp_9530);
    signal tmp_9532[3] <== [tmp_9468[0] * 2034408310088972836, tmp_9468[1] * 2034408310088972836, tmp_9468[2] * 2034408310088972836];
    signal tmp_9533[3] <== [tmp_9532[0] + tmp_9496[0], tmp_9532[1] + tmp_9496[1], tmp_9532[2] + tmp_9496[2]];
    signal tmp_9534[3] <== [evals[133][0] - tmp_9533[0], evals[133][1] - tmp_9533[1], evals[133][2] - tmp_9533[2]];
    signal tmp_9535[3] <== CMul()(evals[53], tmp_9534);
    signal tmp_9536[3] <== [tmp_9531[0] + tmp_9535[0], tmp_9531[1] + tmp_9535[1], tmp_9531[2] + tmp_9535[2]];
    signal tmp_9537[3] <== CMul()(challengeQ, tmp_9536);
    signal tmp_9538[3] <== [tmp_9471[0] * 3434575637390274478, tmp_9471[1] * 3434575637390274478, tmp_9471[2] * 3434575637390274478];
    signal tmp_9539[3] <== [tmp_9538[0] + tmp_9496[0], tmp_9538[1] + tmp_9496[1], tmp_9538[2] + tmp_9496[2]];
    signal tmp_9540[3] <== [evals[134][0] - tmp_9539[0], evals[134][1] - tmp_9539[1], evals[134][2] - tmp_9539[2]];
    signal tmp_9541[3] <== CMul()(evals[53], tmp_9540);
    signal tmp_9542[3] <== [tmp_9537[0] + tmp_9541[0], tmp_9537[1] + tmp_9541[1], tmp_9537[2] + tmp_9541[2]];
    signal tmp_9543[3] <== CMul()(challengeQ, tmp_9542);
    signal tmp_9544[3] <== [tmp_9474[0] * 6052753985947965968, tmp_9474[1] * 6052753985947965968, tmp_9474[2] * 6052753985947965968];
    signal tmp_9545[3] <== [tmp_9544[0] + tmp_9496[0], tmp_9544[1] + tmp_9496[1], tmp_9544[2] + tmp_9496[2]];
    signal tmp_9546[3] <== [evals[135][0] - tmp_9545[0], evals[135][1] - tmp_9545[1], evals[135][2] - tmp_9545[2]];
    signal tmp_9547[3] <== CMul()(evals[53], tmp_9546);
    signal tmp_9548[3] <== [tmp_9543[0] + tmp_9547[0], tmp_9543[1] + tmp_9547[1], tmp_9543[2] + tmp_9547[2]];
    signal tmp_9549[3] <== CMul()(challengeQ, tmp_9548);
    signal tmp_9550[3] <== [tmp_9477[0] * 13608362914817483670, tmp_9477[1] * 13608362914817483670, tmp_9477[2] * 13608362914817483670];
    signal tmp_9551[3] <== [tmp_9550[0] + tmp_9496[0], tmp_9550[1] + tmp_9496[1], tmp_9550[2] + tmp_9496[2]];
    signal tmp_9552[3] <== [evals[136][0] - tmp_9551[0], evals[136][1] - tmp_9551[1], evals[136][2] - tmp_9551[2]];
    signal tmp_9553[3] <== CMul()(evals[53], tmp_9552);
    signal tmp_9554[3] <== [tmp_9549[0] + tmp_9553[0], tmp_9549[1] + tmp_9553[1], tmp_9549[2] + tmp_9553[2]];
    signal tmp_9555[3] <== CMul()(challengeQ, tmp_9554);
    signal tmp_9556[3] <== [tmp_9480[0] * 18163707672964630459, tmp_9480[1] * 18163707672964630459, tmp_9480[2] * 18163707672964630459];
    signal tmp_9557[3] <== [tmp_9556[0] + tmp_9496[0], tmp_9556[1] + tmp_9496[1], tmp_9556[2] + tmp_9496[2]];
    signal tmp_9558[3] <== [evals[137][0] - tmp_9557[0], evals[137][1] - tmp_9557[1], evals[137][2] - tmp_9557[2]];
    signal tmp_9559[3] <== CMul()(evals[53], tmp_9558);
    signal tmp_9560[3] <== [tmp_9555[0] + tmp_9559[0], tmp_9555[1] + tmp_9559[1], tmp_9555[2] + tmp_9559[2]];
    signal tmp_9561[3] <== CMul()(challengeQ, tmp_9560);
    signal tmp_9562[3] <== [tmp_9483[0] * 14373610220374016704, tmp_9483[1] * 14373610220374016704, tmp_9483[2] * 14373610220374016704];
    signal tmp_9563[3] <== [tmp_9562[0] + tmp_9496[0], tmp_9562[1] + tmp_9496[1], tmp_9562[2] + tmp_9496[2]];
    signal tmp_9564[3] <== [evals[138][0] - tmp_9563[0], evals[138][1] - tmp_9563[1], evals[138][2] - tmp_9563[2]];
    signal tmp_9565[3] <== CMul()(evals[53], tmp_9564);
    signal tmp_9566[3] <== [tmp_9561[0] + tmp_9565[0], tmp_9561[1] + tmp_9565[1], tmp_9561[2] + tmp_9565[2]];
    signal tmp_9567[3] <== CMul()(challengeQ, tmp_9566);
    signal tmp_9568[3] <== [tmp_9486[0] * 6226282807566121054, tmp_9486[1] * 6226282807566121054, tmp_9486[2] * 6226282807566121054];
    signal tmp_9569[3] <== [tmp_9568[0] + tmp_9496[0], tmp_9568[1] + tmp_9496[1], tmp_9568[2] + tmp_9496[2]];
    signal tmp_9570[3] <== [evals[139][0] - tmp_9569[0], evals[139][1] - tmp_9569[1], evals[139][2] - tmp_9569[2]];
    signal tmp_9571[3] <== CMul()(evals[53], tmp_9570);
    signal tmp_9572[3] <== [tmp_9567[0] + tmp_9571[0], tmp_9567[1] + tmp_9571[1], tmp_9567[2] + tmp_9571[2]];
    signal tmp_9573[3] <== CMul()(challengeQ, tmp_9572);
    signal tmp_9574[3] <== [tmp_9489[0] * 3643354756180461803, tmp_9489[1] * 3643354756180461803, tmp_9489[2] * 3643354756180461803];
    signal tmp_9575[3] <== [tmp_9574[0] + tmp_9496[0], tmp_9574[1] + tmp_9496[1], tmp_9574[2] + tmp_9496[2]];
    signal tmp_9576[3] <== [evals[140][0] - tmp_9575[0], evals[140][1] - tmp_9575[1], evals[140][2] - tmp_9575[2]];
    signal tmp_9577[3] <== CMul()(evals[53], tmp_9576);
    signal tmp_9578[3] <== [tmp_9573[0] + tmp_9577[0], tmp_9573[1] + tmp_9577[1], tmp_9573[2] + tmp_9577[2]];
    signal tmp_9579[3] <== CMul()(challengeQ, tmp_9578);
    signal tmp_9580[3] <== [tmp_9492[0] * 13046961313070095543, tmp_9492[1] * 13046961313070095543, tmp_9492[2] * 13046961313070095543];
    signal tmp_9581[3] <== [tmp_9580[0] + tmp_9496[0], tmp_9580[1] + tmp_9496[1], tmp_9580[2] + tmp_9496[2]];
    signal tmp_9582[3] <== [evals[141][0] - tmp_9581[0], evals[141][1] - tmp_9581[1], evals[141][2] - tmp_9581[2]];
    signal tmp_9583[3] <== CMul()(evals[53], tmp_9582);
    signal tmp_9584[3] <== [tmp_9579[0] + tmp_9583[0], tmp_9579[1] + tmp_9583[1], tmp_9579[2] + tmp_9583[2]];
    signal tmp_9585[3] <== CMul()(challengeQ, tmp_9584);
    signal tmp_9586[3] <== [tmp_9495[0] * 8594143216561850811, tmp_9495[1] * 8594143216561850811, tmp_9495[2] * 8594143216561850811];
    signal tmp_9587[3] <== [tmp_9586[0] + tmp_9496[0], tmp_9586[1] + tmp_9496[1], tmp_9586[2] + tmp_9496[2]];
    signal tmp_9588[3] <== [evals[142][0] - tmp_9587[0], evals[142][1] - tmp_9587[1], evals[142][2] - tmp_9587[2]];
    signal tmp_9589[3] <== CMul()(evals[53], tmp_9588);
    signal tmp_9590[3] <== [tmp_9585[0] + tmp_9589[0], tmp_9585[1] + tmp_9589[1], tmp_9585[2] + tmp_9589[2]];
    signal tmp_9591[3] <== CMul()(challengeQ, tmp_9590);
    signal tmp_9592[3] <== CMul()(evals[64], evals[67]);
    signal tmp_9593[3] <== CMul()(evals[65], evals[69]);
    signal tmp_9594[3] <== [tmp_9592[0] + tmp_9593[0], tmp_9592[1] + tmp_9593[1], tmp_9592[2] + tmp_9593[2]];
    signal tmp_9595[3] <== CMul()(evals[66], evals[68]);
    signal tmp_9596[3] <== [tmp_9594[0] + tmp_9595[0], tmp_9594[1] + tmp_9595[1], tmp_9594[2] + tmp_9595[2]];
    signal tmp_9597[3] <== [evals[70][0] - tmp_9596[0], evals[70][1] - tmp_9596[1], evals[70][2] - tmp_9596[2]];
    signal tmp_9598[3] <== CMul()(evals[55], tmp_9597);
    signal tmp_9599[3] <== [tmp_9591[0] + tmp_9598[0], tmp_9591[1] + tmp_9598[1], tmp_9591[2] + tmp_9598[2]];
    signal tmp_9600[3] <== CMul()(challengeQ, tmp_9599);
    signal tmp_9601[3] <== CMul()(evals[64], evals[68]);
    signal tmp_9602[3] <== CMul()(evals[65], evals[67]);
    signal tmp_9603[3] <== [tmp_9601[0] + tmp_9602[0], tmp_9601[1] + tmp_9602[1], tmp_9601[2] + tmp_9602[2]];
    signal tmp_9605[3] <== [tmp_9603[0] + tmp_9593[0], tmp_9603[1] + tmp_9593[1], tmp_9603[2] + tmp_9593[2]];
    signal tmp_9607[3] <== [tmp_9605[0] + tmp_9595[0], tmp_9605[1] + tmp_9595[1], tmp_9605[2] + tmp_9595[2]];
    signal tmp_9608[3] <== CMul()(evals[66], evals[69]);
    signal tmp_9609[3] <== [tmp_9607[0] + tmp_9608[0], tmp_9607[1] + tmp_9608[1], tmp_9607[2] + tmp_9608[2]];
    signal tmp_9610[3] <== [evals[71][0] - tmp_9609[0], evals[71][1] - tmp_9609[1], evals[71][2] - tmp_9609[2]];
    signal tmp_9611[3] <== CMul()(evals[55], tmp_9610);
    signal tmp_9612[3] <== [tmp_9600[0] + tmp_9611[0], tmp_9600[1] + tmp_9611[1], tmp_9600[2] + tmp_9611[2]];
    signal tmp_9613[3] <== CMul()(challengeQ, tmp_9612);
    signal tmp_9614[3] <== CMul()(evals[64], evals[69]);
    signal tmp_9616[3] <== [tmp_9614[0] + tmp_9608[0], tmp_9614[1] + tmp_9608[1], tmp_9614[2] + tmp_9608[2]];
    signal tmp_9617[3] <== CMul()(evals[66], evals[67]);
    signal tmp_9618[3] <== [tmp_9616[0] + tmp_9617[0], tmp_9616[1] + tmp_9617[1], tmp_9616[2] + tmp_9617[2]];
    signal tmp_9619[3] <== CMul()(evals[65], evals[68]);
    signal tmp_9620[3] <== [tmp_9618[0] + tmp_9619[0], tmp_9618[1] + tmp_9619[1], tmp_9618[2] + tmp_9619[2]];
    signal tmp_9621[3] <== [evals[72][0] - tmp_9620[0], evals[72][1] - tmp_9620[1], evals[72][2] - tmp_9620[2]];
    signal tmp_9622[3] <== CMul()(evals[55], tmp_9621);
    signal tmp_9623[3] <== [tmp_9613[0] + tmp_9622[0], tmp_9613[1] + tmp_9622[1], tmp_9613[2] + tmp_9622[2]];
    signal tmp_9624[3] <== CMul()(challengeQ, tmp_9623);
    signal tmp_9625[3] <== CMul()(evals[73], evals[76]);
    signal tmp_9626[3] <== CMul()(evals[74], evals[78]);
    signal tmp_9627[3] <== [tmp_9625[0] + tmp_9626[0], tmp_9625[1] + tmp_9626[1], tmp_9625[2] + tmp_9626[2]];
    signal tmp_9628[3] <== CMul()(evals[75], evals[77]);
    signal tmp_9629[3] <== [tmp_9627[0] + tmp_9628[0], tmp_9627[1] + tmp_9628[1], tmp_9627[2] + tmp_9628[2]];
    signal tmp_9630[3] <== [evals[79][0] - tmp_9629[0], evals[79][1] - tmp_9629[1], evals[79][2] - tmp_9629[2]];
    signal tmp_9631[3] <== CMul()(evals[55], tmp_9630);
    signal tmp_9632[3] <== [tmp_9624[0] + tmp_9631[0], tmp_9624[1] + tmp_9631[1], tmp_9624[2] + tmp_9631[2]];
    signal tmp_9633[3] <== CMul()(challengeQ, tmp_9632);
    signal tmp_9634[3] <== CMul()(evals[73], evals[77]);
    signal tmp_9635[3] <== CMul()(evals[74], evals[76]);
    signal tmp_9636[3] <== [tmp_9634[0] + tmp_9635[0], tmp_9634[1] + tmp_9635[1], tmp_9634[2] + tmp_9635[2]];
    signal tmp_9638[3] <== [tmp_9636[0] + tmp_9626[0], tmp_9636[1] + tmp_9626[1], tmp_9636[2] + tmp_9626[2]];
    signal tmp_9640[3] <== [tmp_9638[0] + tmp_9628[0], tmp_9638[1] + tmp_9628[1], tmp_9638[2] + tmp_9628[2]];
    signal tmp_9641[3] <== CMul()(evals[75], evals[78]);
    signal tmp_9642[3] <== [tmp_9640[0] + tmp_9641[0], tmp_9640[1] + tmp_9641[1], tmp_9640[2] + tmp_9641[2]];
    signal tmp_9643[3] <== [evals[80][0] - tmp_9642[0], evals[80][1] - tmp_9642[1], evals[80][2] - tmp_9642[2]];
    signal tmp_9644[3] <== CMul()(evals[55], tmp_9643);
    signal tmp_9645[3] <== [tmp_9633[0] + tmp_9644[0], tmp_9633[1] + tmp_9644[1], tmp_9633[2] + tmp_9644[2]];
    signal tmp_9646[3] <== CMul()(challengeQ, tmp_9645);
    signal tmp_9647[3] <== CMul()(evals[73], evals[78]);
    signal tmp_9649[3] <== [tmp_9647[0] + tmp_9641[0], tmp_9647[1] + tmp_9641[1], tmp_9647[2] + tmp_9641[2]];
    signal tmp_9650[3] <== CMul()(evals[75], evals[76]);
    signal tmp_9651[3] <== [tmp_9649[0] + tmp_9650[0], tmp_9649[1] + tmp_9650[1], tmp_9649[2] + tmp_9650[2]];
    signal tmp_9652[3] <== CMul()(evals[74], evals[77]);
    signal tmp_9653[3] <== [tmp_9651[0] + tmp_9652[0], tmp_9651[1] + tmp_9652[1], tmp_9651[2] + tmp_9652[2]];
    signal tmp_9654[3] <== [evals[81][0] - tmp_9653[0], evals[81][1] - tmp_9653[1], evals[81][2] - tmp_9653[2]];
    signal tmp_9655[3] <== CMul()(evals[55], tmp_9654);
    signal tmp_9656[3] <== [tmp_9646[0] + tmp_9655[0], tmp_9646[1] + tmp_9655[1], tmp_9646[2] + tmp_9655[2]];
    tmp_9657 <== CMul()(challengeQ, tmp_9656);
}

template VerifyEvaluationsChunks4_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_6837[3];
    signal input tmp_6849[3];
    signal input tmp_6867[3];
    signal input tmp_6873[3];
    signal input tmp_9657[3];

    signal output tmp_10448[3];
    signal tmp_9658[3] <== CMul()(evals[41], evals[64]);
    signal tmp_9660[3] <== [tmp_9658[0] + tmp_6837[0], tmp_9658[1] + tmp_6837[1], tmp_9658[2] + tmp_6837[2]];
    signal tmp_9661[3] <== CMul()(evals[43], evals[70]);
    signal tmp_9662[3] <== [tmp_9660[0] + tmp_9661[0], tmp_9660[1] + tmp_9661[1], tmp_9660[2] + tmp_9661[2]];
    signal tmp_9663[3] <== CMul()(evals[44], evals[73]);
    signal tmp_9664[3] <== [tmp_9662[0] + tmp_9663[0], tmp_9662[1] + tmp_9663[1], tmp_9662[2] + tmp_9663[2]];
    signal tmp_9665[3] <== CMul()(evals[47], evals[64]);
    signal tmp_9666[3] <== [tmp_9664[0] + tmp_9665[0], tmp_9664[1] + tmp_9665[1], tmp_9664[2] + tmp_9665[2]];
    signal tmp_9667[3] <== CMul()(evals[48], evals[67]);
    signal tmp_9668[3] <== [tmp_9666[0] + tmp_9667[0], tmp_9666[1] + tmp_9667[1], tmp_9666[2] + tmp_9667[2]];
    signal tmp_9669[3] <== [evals[76][0] - tmp_9668[0], evals[76][1] - tmp_9668[1], evals[76][2] - tmp_9668[2]];
    signal tmp_9670[3] <== CMul()(evals[57], tmp_9669);
    signal tmp_9671[3] <== [tmp_9657[0] + tmp_9670[0], tmp_9657[1] + tmp_9670[1], tmp_9657[2] + tmp_9670[2]];
    signal tmp_9672[3] <== CMul()(challengeQ, tmp_9671);
    signal tmp_9673[3] <== CMul()(evals[41], evals[65]);
    signal tmp_9674[3] <== CMul()(evals[42], evals[68]);
    signal tmp_9675[3] <== [tmp_9673[0] + tmp_9674[0], tmp_9673[1] + tmp_9674[1], tmp_9673[2] + tmp_9674[2]];
    signal tmp_9676[3] <== CMul()(evals[43], evals[71]);
    signal tmp_9677[3] <== [tmp_9675[0] + tmp_9676[0], tmp_9675[1] + tmp_9676[1], tmp_9675[2] + tmp_9676[2]];
    signal tmp_9678[3] <== CMul()(evals[44], evals[74]);
    signal tmp_9679[3] <== [tmp_9677[0] + tmp_9678[0], tmp_9677[1] + tmp_9678[1], tmp_9677[2] + tmp_9678[2]];
    signal tmp_9680[3] <== CMul()(evals[47], evals[65]);
    signal tmp_9681[3] <== [tmp_9679[0] + tmp_9680[0], tmp_9679[1] + tmp_9680[1], tmp_9679[2] + tmp_9680[2]];
    signal tmp_9682[3] <== CMul()(evals[48], evals[68]);
    signal tmp_9683[3] <== [tmp_9681[0] + tmp_9682[0], tmp_9681[1] + tmp_9682[1], tmp_9681[2] + tmp_9682[2]];
    signal tmp_9684[3] <== [evals[77][0] - tmp_9683[0], evals[77][1] - tmp_9683[1], evals[77][2] - tmp_9683[2]];
    signal tmp_9685[3] <== CMul()(evals[57], tmp_9684);
    signal tmp_9686[3] <== [tmp_9672[0] + tmp_9685[0], tmp_9672[1] + tmp_9685[1], tmp_9672[2] + tmp_9685[2]];
    signal tmp_9687[3] <== CMul()(challengeQ, tmp_9686);
    signal tmp_9688[3] <== CMul()(evals[41], evals[66]);
    signal tmp_9689[3] <== CMul()(evals[42], evals[69]);
    signal tmp_9690[3] <== [tmp_9688[0] + tmp_9689[0], tmp_9688[1] + tmp_9689[1], tmp_9688[2] + tmp_9689[2]];
    signal tmp_9691[3] <== CMul()(evals[43], evals[72]);
    signal tmp_9692[3] <== [tmp_9690[0] + tmp_9691[0], tmp_9690[1] + tmp_9691[1], tmp_9690[2] + tmp_9691[2]];
    signal tmp_9693[3] <== CMul()(evals[44], evals[75]);
    signal tmp_9694[3] <== [tmp_9692[0] + tmp_9693[0], tmp_9692[1] + tmp_9693[1], tmp_9692[2] + tmp_9693[2]];
    signal tmp_9695[3] <== CMul()(evals[47], evals[66]);
    signal tmp_9696[3] <== [tmp_9694[0] + tmp_9695[0], tmp_9694[1] + tmp_9695[1], tmp_9694[2] + tmp_9695[2]];
    signal tmp_9697[3] <== CMul()(evals[48], evals[69]);
    signal tmp_9698[3] <== [tmp_9696[0] + tmp_9697[0], tmp_9696[1] + tmp_9697[1], tmp_9696[2] + tmp_9697[2]];
    signal tmp_9699[3] <== [evals[78][0] - tmp_9698[0], evals[78][1] - tmp_9698[1], evals[78][2] - tmp_9698[2]];
    signal tmp_9700[3] <== CMul()(evals[57], tmp_9699);
    signal tmp_9701[3] <== [tmp_9687[0] + tmp_9700[0], tmp_9687[1] + tmp_9700[1], tmp_9687[2] + tmp_9700[2]];
    signal tmp_9702[3] <== CMul()(challengeQ, tmp_9701);
    signal tmp_9705[3] <== [tmp_9658[0] - tmp_6837[0], tmp_9658[1] - tmp_6837[1], tmp_9658[2] - tmp_6837[2]];
    signal tmp_9706[3] <== CMul()(evals[45], evals[70]);
    signal tmp_9707[3] <== [tmp_9705[0] + tmp_9706[0], tmp_9705[1] + tmp_9706[1], tmp_9705[2] + tmp_9706[2]];
    signal tmp_9708[3] <== CMul()(evals[46], evals[73]);
    signal tmp_9709[3] <== [tmp_9707[0] - tmp_9708[0], tmp_9707[1] - tmp_9708[1], tmp_9707[2] - tmp_9708[2]];
    signal tmp_9711[3] <== [tmp_9709[0] + tmp_9665[0], tmp_9709[1] + tmp_9665[1], tmp_9709[2] + tmp_9665[2]];
    signal tmp_9713[3] <== [tmp_9711[0] - tmp_9667[0], tmp_9711[1] - tmp_9667[1], tmp_9711[2] - tmp_9667[2]];
    signal tmp_9714[3] <== [evals[79][0] - tmp_9713[0], evals[79][1] - tmp_9713[1], evals[79][2] - tmp_9713[2]];
    signal tmp_9715[3] <== CMul()(evals[57], tmp_9714);
    signal tmp_9716[3] <== [tmp_9702[0] + tmp_9715[0], tmp_9702[1] + tmp_9715[1], tmp_9702[2] + tmp_9715[2]];
    signal tmp_9717[3] <== CMul()(challengeQ, tmp_9716);
    signal tmp_9720[3] <== [tmp_9673[0] - tmp_9674[0], tmp_9673[1] - tmp_9674[1], tmp_9673[2] - tmp_9674[2]];
    signal tmp_9721[3] <== CMul()(evals[45], evals[71]);
    signal tmp_9722[3] <== [tmp_9720[0] + tmp_9721[0], tmp_9720[1] + tmp_9721[1], tmp_9720[2] + tmp_9721[2]];
    signal tmp_9723[3] <== CMul()(evals[46], evals[74]);
    signal tmp_9724[3] <== [tmp_9722[0] - tmp_9723[0], tmp_9722[1] - tmp_9723[1], tmp_9722[2] - tmp_9723[2]];
    signal tmp_9726[3] <== [tmp_9724[0] + tmp_9680[0], tmp_9724[1] + tmp_9680[1], tmp_9724[2] + tmp_9680[2]];
    signal tmp_9728[3] <== [tmp_9726[0] - tmp_9682[0], tmp_9726[1] - tmp_9682[1], tmp_9726[2] - tmp_9682[2]];
    signal tmp_9729[3] <== [evals[80][0] - tmp_9728[0], evals[80][1] - tmp_9728[1], evals[80][2] - tmp_9728[2]];
    signal tmp_9730[3] <== CMul()(evals[57], tmp_9729);
    signal tmp_9731[3] <== [tmp_9717[0] + tmp_9730[0], tmp_9717[1] + tmp_9730[1], tmp_9717[2] + tmp_9730[2]];
    signal tmp_9732[3] <== CMul()(challengeQ, tmp_9731);
    signal tmp_9735[3] <== [tmp_9688[0] - tmp_9689[0], tmp_9688[1] - tmp_9689[1], tmp_9688[2] - tmp_9689[2]];
    signal tmp_9736[3] <== CMul()(evals[45], evals[72]);
    signal tmp_9737[3] <== [tmp_9735[0] + tmp_9736[0], tmp_9735[1] + tmp_9736[1], tmp_9735[2] + tmp_9736[2]];
    signal tmp_9738[3] <== CMul()(evals[46], evals[75]);
    signal tmp_9739[3] <== [tmp_9737[0] - tmp_9738[0], tmp_9737[1] - tmp_9738[1], tmp_9737[2] - tmp_9738[2]];
    signal tmp_9741[3] <== [tmp_9739[0] + tmp_9695[0], tmp_9739[1] + tmp_9695[1], tmp_9739[2] + tmp_9695[2]];
    signal tmp_9743[3] <== [tmp_9741[0] - tmp_9697[0], tmp_9741[1] - tmp_9697[1], tmp_9741[2] - tmp_9697[2]];
    signal tmp_9744[3] <== [evals[81][0] - tmp_9743[0], evals[81][1] - tmp_9743[1], evals[81][2] - tmp_9743[2]];
    signal tmp_9745[3] <== CMul()(evals[57], tmp_9744);
    signal tmp_9746[3] <== [tmp_9732[0] + tmp_9745[0], tmp_9732[1] + tmp_9745[1], tmp_9732[2] + tmp_9745[2]];
    signal tmp_9747[3] <== CMul()(challengeQ, tmp_9746);
    signal tmp_9752[3] <== [tmp_9660[0] - tmp_9661[0], tmp_9660[1] - tmp_9661[1], tmp_9660[2] - tmp_9661[2]];
    signal tmp_9754[3] <== [tmp_9752[0] - tmp_9663[0], tmp_9752[1] - tmp_9663[1], tmp_9752[2] - tmp_9663[2]];
    signal tmp_9756[3] <== [tmp_9754[0] + tmp_6849[0], tmp_9754[1] + tmp_6849[1], tmp_9754[2] + tmp_6849[2]];
    signal tmp_9757[3] <== CMul()(evals[49], evals[73]);
    signal tmp_9758[3] <== [tmp_9756[0] + tmp_9757[0], tmp_9756[1] + tmp_9757[1], tmp_9756[2] + tmp_9757[2]];
    signal tmp_9759[3] <== [evals[82][0] - tmp_9758[0], evals[82][1] - tmp_9758[1], evals[82][2] - tmp_9758[2]];
    signal tmp_9760[3] <== CMul()(evals[57], tmp_9759);
    signal tmp_9761[3] <== [tmp_9747[0] + tmp_9760[0], tmp_9747[1] + tmp_9760[1], tmp_9747[2] + tmp_9760[2]];
    signal tmp_9762[3] <== CMul()(challengeQ, tmp_9761);
    signal tmp_9767[3] <== [tmp_9675[0] - tmp_9676[0], tmp_9675[1] - tmp_9676[1], tmp_9675[2] - tmp_9676[2]];
    signal tmp_9769[3] <== [tmp_9767[0] - tmp_9678[0], tmp_9767[1] - tmp_9678[1], tmp_9767[2] - tmp_9678[2]];
    signal tmp_9770[3] <== CMul()(evals[47], evals[71]);
    signal tmp_9771[3] <== [tmp_9769[0] + tmp_9770[0], tmp_9769[1] + tmp_9770[1], tmp_9769[2] + tmp_9770[2]];
    signal tmp_9772[3] <== CMul()(evals[49], evals[74]);
    signal tmp_9773[3] <== [tmp_9771[0] + tmp_9772[0], tmp_9771[1] + tmp_9772[1], tmp_9771[2] + tmp_9772[2]];
    signal tmp_9774[3] <== [evals[83][0] - tmp_9773[0], evals[83][1] - tmp_9773[1], evals[83][2] - tmp_9773[2]];
    signal tmp_9775[3] <== CMul()(evals[57], tmp_9774);
    signal tmp_9776[3] <== [tmp_9762[0] + tmp_9775[0], tmp_9762[1] + tmp_9775[1], tmp_9762[2] + tmp_9775[2]];
    signal tmp_9777[3] <== CMul()(challengeQ, tmp_9776);
    signal tmp_9782[3] <== [tmp_9690[0] - tmp_9691[0], tmp_9690[1] - tmp_9691[1], tmp_9690[2] - tmp_9691[2]];
    signal tmp_9784[3] <== [tmp_9782[0] - tmp_9693[0], tmp_9782[1] - tmp_9693[1], tmp_9782[2] - tmp_9693[2]];
    signal tmp_9785[3] <== CMul()(evals[47], evals[72]);
    signal tmp_9786[3] <== [tmp_9784[0] + tmp_9785[0], tmp_9784[1] + tmp_9785[1], tmp_9784[2] + tmp_9785[2]];
    signal tmp_9788[3] <== [tmp_9786[0] + tmp_6867[0], tmp_9786[1] + tmp_6867[1], tmp_9786[2] + tmp_6867[2]];
    signal tmp_9789[3] <== [evals[84][0] - tmp_9788[0], evals[84][1] - tmp_9788[1], evals[84][2] - tmp_9788[2]];
    signal tmp_9790[3] <== CMul()(evals[57], tmp_9789);
    signal tmp_9791[3] <== [tmp_9777[0] + tmp_9790[0], tmp_9777[1] + tmp_9790[1], tmp_9777[2] + tmp_9790[2]];
    signal tmp_9792[3] <== CMul()(challengeQ, tmp_9791);
    signal tmp_9797[3] <== [tmp_9705[0] - tmp_9706[0], tmp_9705[1] - tmp_9706[1], tmp_9705[2] - tmp_9706[2]];
    signal tmp_9799[3] <== [tmp_9797[0] + tmp_9708[0], tmp_9797[1] + tmp_9708[1], tmp_9797[2] + tmp_9708[2]];
    signal tmp_9801[3] <== [tmp_9799[0] + tmp_6849[0], tmp_9799[1] + tmp_6849[1], tmp_9799[2] + tmp_6849[2]];
    signal tmp_9803[3] <== [tmp_9801[0] - tmp_9757[0], tmp_9801[1] - tmp_9757[1], tmp_9801[2] - tmp_9757[2]];
    signal tmp_9804[3] <== [evals[85][0] - tmp_9803[0], evals[85][1] - tmp_9803[1], evals[85][2] - tmp_9803[2]];
    signal tmp_9805[3] <== CMul()(evals[57], tmp_9804);
    signal tmp_9806[3] <== [tmp_9792[0] + tmp_9805[0], tmp_9792[1] + tmp_9805[1], tmp_9792[2] + tmp_9805[2]];
    signal tmp_9807[3] <== CMul()(challengeQ, tmp_9806);
    signal tmp_9812[3] <== [tmp_9720[0] - tmp_9721[0], tmp_9720[1] - tmp_9721[1], tmp_9720[2] - tmp_9721[2]];
    signal tmp_9814[3] <== [tmp_9812[0] + tmp_9723[0], tmp_9812[1] + tmp_9723[1], tmp_9812[2] + tmp_9723[2]];
    signal tmp_9816[3] <== [tmp_9814[0] + tmp_9770[0], tmp_9814[1] + tmp_9770[1], tmp_9814[2] + tmp_9770[2]];
    signal tmp_9818[3] <== [tmp_9816[0] - tmp_9772[0], tmp_9816[1] - tmp_9772[1], tmp_9816[2] - tmp_9772[2]];
    signal tmp_9819[3] <== [evals[86][0] - tmp_9818[0], evals[86][1] - tmp_9818[1], evals[86][2] - tmp_9818[2]];
    signal tmp_9820[3] <== CMul()(evals[57], tmp_9819);
    signal tmp_9821[3] <== [tmp_9807[0] + tmp_9820[0], tmp_9807[1] + tmp_9820[1], tmp_9807[2] + tmp_9820[2]];
    signal tmp_9822[3] <== CMul()(challengeQ, tmp_9821);
    signal tmp_9827[3] <== [tmp_9735[0] - tmp_9736[0], tmp_9735[1] - tmp_9736[1], tmp_9735[2] - tmp_9736[2]];
    signal tmp_9829[3] <== [tmp_9827[0] + tmp_9738[0], tmp_9827[1] + tmp_9738[1], tmp_9827[2] + tmp_9738[2]];
    signal tmp_9831[3] <== [tmp_9829[0] + tmp_9785[0], tmp_9829[1] + tmp_9785[1], tmp_9829[2] + tmp_9785[2]];
    signal tmp_9833[3] <== [tmp_9831[0] - tmp_6867[0], tmp_9831[1] - tmp_6867[1], tmp_9831[2] - tmp_6867[2]];
    signal tmp_9834[3] <== [evals[87][0] - tmp_9833[0], evals[87][1] - tmp_9833[1], evals[87][2] - tmp_9833[2]];
    signal tmp_9835[3] <== CMul()(evals[57], tmp_9834);
    signal tmp_9836[3] <== [tmp_9822[0] + tmp_9835[0], tmp_9822[1] + tmp_9835[1], tmp_9822[2] + tmp_9835[2]];
    signal tmp_9837[3] <== CMul()(challengeQ, tmp_9836);
    signal tmp_9838[3] <== CMul()(evals[76], evals[79]);
    signal tmp_9839[3] <== CMul()(evals[77], evals[81]);
    signal tmp_9840[3] <== [tmp_9838[0] + tmp_9839[0], tmp_9838[1] + tmp_9839[1], tmp_9838[2] + tmp_9839[2]];
    signal tmp_9841[3] <== CMul()(evals[78], evals[80]);
    signal tmp_9842[3] <== [tmp_9840[0] + tmp_9841[0], tmp_9840[1] + tmp_9841[1], tmp_9840[2] + tmp_9841[2]];
    signal tmp_9843[3] <== [tmp_9842[0] + evals[73][0], tmp_9842[1] + evals[73][1], tmp_9842[2] + evals[73][2]];
    signal tmp_9844[3] <== CMul()(tmp_9843, evals[79]);
    signal tmp_9845[3] <== CMul()(evals[76], evals[80]);
    signal tmp_9846[3] <== CMul()(evals[77], evals[79]);
    signal tmp_9847[3] <== [tmp_9845[0] + tmp_9846[0], tmp_9845[1] + tmp_9846[1], tmp_9845[2] + tmp_9846[2]];
    signal tmp_9849[3] <== [tmp_9847[0] + tmp_9839[0], tmp_9847[1] + tmp_9839[1], tmp_9847[2] + tmp_9839[2]];
    signal tmp_9851[3] <== [tmp_9849[0] + tmp_9841[0], tmp_9849[1] + tmp_9841[1], tmp_9849[2] + tmp_9841[2]];
    signal tmp_9852[3] <== CMul()(evals[78], evals[81]);
    signal tmp_9853[3] <== [tmp_9851[0] + tmp_9852[0], tmp_9851[1] + tmp_9852[1], tmp_9851[2] + tmp_9852[2]];
    signal tmp_9854[3] <== [tmp_9853[0] + evals[74][0], tmp_9853[1] + evals[74][1], tmp_9853[2] + evals[74][2]];
    signal tmp_9855[3] <== CMul()(tmp_9854, evals[81]);
    signal tmp_9856[3] <== [tmp_9844[0] + tmp_9855[0], tmp_9844[1] + tmp_9855[1], tmp_9844[2] + tmp_9855[2]];
    signal tmp_9857[3] <== CMul()(evals[76], evals[81]);
    signal tmp_9859[3] <== [tmp_9857[0] + tmp_9852[0], tmp_9857[1] + tmp_9852[1], tmp_9857[2] + tmp_9852[2]];
    signal tmp_9860[3] <== CMul()(evals[78], evals[79]);
    signal tmp_9861[3] <== [tmp_9859[0] + tmp_9860[0], tmp_9859[1] + tmp_9860[1], tmp_9859[2] + tmp_9860[2]];
    signal tmp_9862[3] <== CMul()(evals[77], evals[80]);
    signal tmp_9863[3] <== [tmp_9861[0] + tmp_9862[0], tmp_9861[1] + tmp_9862[1], tmp_9861[2] + tmp_9862[2]];
    signal tmp_9864[3] <== [tmp_9863[0] + evals[75][0], tmp_9863[1] + evals[75][1], tmp_9863[2] + evals[75][2]];
    signal tmp_9865[3] <== CMul()(tmp_9864, evals[80]);
    signal tmp_9866[3] <== [tmp_9856[0] + tmp_9865[0], tmp_9856[1] + tmp_9865[1], tmp_9856[2] + tmp_9865[2]];
    signal tmp_9867[3] <== [tmp_9866[0] + evals[70][0], tmp_9866[1] + evals[70][1], tmp_9866[2] + evals[70][2]];
    signal tmp_9868[3] <== CMul()(tmp_9867, evals[79]);
    signal tmp_9869[3] <== CMul()(tmp_9843, evals[80]);
    signal tmp_9870[3] <== CMul()(tmp_9854, evals[79]);
    signal tmp_9871[3] <== [tmp_9869[0] + tmp_9870[0], tmp_9869[1] + tmp_9870[1], tmp_9869[2] + tmp_9870[2]];
    signal tmp_9873[3] <== [tmp_9871[0] + tmp_9855[0], tmp_9871[1] + tmp_9855[1], tmp_9871[2] + tmp_9855[2]];
    signal tmp_9875[3] <== [tmp_9873[0] + tmp_9865[0], tmp_9873[1] + tmp_9865[1], tmp_9873[2] + tmp_9865[2]];
    signal tmp_9876[3] <== CMul()(tmp_9864, evals[81]);
    signal tmp_9877[3] <== [tmp_9875[0] + tmp_9876[0], tmp_9875[1] + tmp_9876[1], tmp_9875[2] + tmp_9876[2]];
    signal tmp_9878[3] <== [tmp_9877[0] + evals[71][0], tmp_9877[1] + evals[71][1], tmp_9877[2] + evals[71][2]];
    signal tmp_9879[3] <== CMul()(tmp_9878, evals[81]);
    signal tmp_9880[3] <== [tmp_9868[0] + tmp_9879[0], tmp_9868[1] + tmp_9879[1], tmp_9868[2] + tmp_9879[2]];
    signal tmp_9881[3] <== CMul()(tmp_9843, evals[81]);
    signal tmp_9883[3] <== [tmp_9881[0] + tmp_9876[0], tmp_9881[1] + tmp_9876[1], tmp_9881[2] + tmp_9876[2]];
    signal tmp_9884[3] <== CMul()(tmp_9864, evals[79]);
    signal tmp_9885[3] <== [tmp_9883[0] + tmp_9884[0], tmp_9883[1] + tmp_9884[1], tmp_9883[2] + tmp_9884[2]];
    signal tmp_9886[3] <== CMul()(tmp_9854, evals[80]);
    signal tmp_9887[3] <== [tmp_9885[0] + tmp_9886[0], tmp_9885[1] + tmp_9886[1], tmp_9885[2] + tmp_9886[2]];
    signal tmp_9888[3] <== [tmp_9887[0] + evals[72][0], tmp_9887[1] + evals[72][1], tmp_9887[2] + evals[72][2]];
    signal tmp_9889[3] <== CMul()(tmp_9888, evals[80]);
    signal tmp_9890[3] <== [tmp_9880[0] + tmp_9889[0], tmp_9880[1] + tmp_9889[1], tmp_9880[2] + tmp_9889[2]];
    signal tmp_9891[3] <== [tmp_9890[0] + evals[67][0], tmp_9890[1] + evals[67][1], tmp_9890[2] + evals[67][2]];
    signal tmp_9892[3] <== CMul()(tmp_9891, evals[79]);
    signal tmp_9893[3] <== CMul()(tmp_9867, evals[80]);
    signal tmp_9894[3] <== CMul()(tmp_9878, evals[79]);
    signal tmp_9895[3] <== [tmp_9893[0] + tmp_9894[0], tmp_9893[1] + tmp_9894[1], tmp_9893[2] + tmp_9894[2]];
    signal tmp_9897[3] <== [tmp_9895[0] + tmp_9879[0], tmp_9895[1] + tmp_9879[1], tmp_9895[2] + tmp_9879[2]];
    signal tmp_9899[3] <== [tmp_9897[0] + tmp_9889[0], tmp_9897[1] + tmp_9889[1], tmp_9897[2] + tmp_9889[2]];
    signal tmp_9900[3] <== CMul()(tmp_9888, evals[81]);
    signal tmp_9901[3] <== [tmp_9899[0] + tmp_9900[0], tmp_9899[1] + tmp_9900[1], tmp_9899[2] + tmp_9900[2]];
    signal tmp_9902[3] <== [tmp_9901[0] + evals[68][0], tmp_9901[1] + evals[68][1], tmp_9901[2] + evals[68][2]];
    signal tmp_9903[3] <== CMul()(tmp_9902, evals[81]);
    signal tmp_9904[3] <== [tmp_9892[0] + tmp_9903[0], tmp_9892[1] + tmp_9903[1], tmp_9892[2] + tmp_9903[2]];
    signal tmp_9905[3] <== CMul()(tmp_9867, evals[81]);
    signal tmp_9907[3] <== [tmp_9905[0] + tmp_9900[0], tmp_9905[1] + tmp_9900[1], tmp_9905[2] + tmp_9900[2]];
    signal tmp_9908[3] <== CMul()(tmp_9888, evals[79]);
    signal tmp_9909[3] <== [tmp_9907[0] + tmp_9908[0], tmp_9907[1] + tmp_9908[1], tmp_9907[2] + tmp_9908[2]];
    signal tmp_9910[3] <== CMul()(tmp_9878, evals[80]);
    signal tmp_9911[3] <== [tmp_9909[0] + tmp_9910[0], tmp_9909[1] + tmp_9910[1], tmp_9909[2] + tmp_9910[2]];
    signal tmp_9912[3] <== [tmp_9911[0] + evals[69][0], tmp_9911[1] + evals[69][1], tmp_9911[2] + evals[69][2]];
    signal tmp_9913[3] <== CMul()(tmp_9912, evals[80]);
    signal tmp_9914[3] <== [tmp_9904[0] + tmp_9913[0], tmp_9904[1] + tmp_9913[1], tmp_9904[2] + tmp_9913[2]];
    signal tmp_9915[3] <== [tmp_9914[0] + evals[64][0], tmp_9914[1] + evals[64][1], tmp_9914[2] + evals[64][2]];
    signal tmp_9916[3] <== [tmp_9915[0] - evals[82][0], tmp_9915[1] - evals[82][1], tmp_9915[2] - evals[82][2]];
    signal tmp_9917[3] <== CMul()(evals[56], tmp_9916);
    signal tmp_9918[3] <== [tmp_9837[0] + tmp_9917[0], tmp_9837[1] + tmp_9917[1], tmp_9837[2] + tmp_9917[2]];
    signal tmp_9919[3] <== CMul()(challengeQ, tmp_9918);
    signal tmp_9920[3] <== CMul()(tmp_9891, evals[80]);
    signal tmp_9921[3] <== CMul()(tmp_9902, evals[79]);
    signal tmp_9922[3] <== [tmp_9920[0] + tmp_9921[0], tmp_9920[1] + tmp_9921[1], tmp_9920[2] + tmp_9921[2]];
    signal tmp_9924[3] <== [tmp_9922[0] + tmp_9903[0], tmp_9922[1] + tmp_9903[1], tmp_9922[2] + tmp_9903[2]];
    signal tmp_9926[3] <== [tmp_9924[0] + tmp_9913[0], tmp_9924[1] + tmp_9913[1], tmp_9924[2] + tmp_9913[2]];
    signal tmp_9927[3] <== CMul()(tmp_9912, evals[81]);
    signal tmp_9928[3] <== [tmp_9926[0] + tmp_9927[0], tmp_9926[1] + tmp_9927[1], tmp_9926[2] + tmp_9927[2]];
    signal tmp_9929[3] <== [tmp_9928[0] + evals[65][0], tmp_9928[1] + evals[65][1], tmp_9928[2] + evals[65][2]];
    signal tmp_9930[3] <== [tmp_9929[0] - evals[83][0], tmp_9929[1] - evals[83][1], tmp_9929[2] - evals[83][2]];
    signal tmp_9931[3] <== CMul()(evals[56], tmp_9930);
    signal tmp_9932[3] <== [tmp_9919[0] + tmp_9931[0], tmp_9919[1] + tmp_9931[1], tmp_9919[2] + tmp_9931[2]];
    signal tmp_9933[3] <== CMul()(challengeQ, tmp_9932);
    signal tmp_9934[3] <== CMul()(tmp_9891, evals[81]);
    signal tmp_9936[3] <== [tmp_9934[0] + tmp_9927[0], tmp_9934[1] + tmp_9927[1], tmp_9934[2] + tmp_9927[2]];
    signal tmp_9937[3] <== CMul()(tmp_9912, evals[79]);
    signal tmp_9938[3] <== [tmp_9936[0] + tmp_9937[0], tmp_9936[1] + tmp_9937[1], tmp_9936[2] + tmp_9937[2]];
    signal tmp_9939[3] <== CMul()(tmp_9902, evals[80]);
    signal tmp_9940[3] <== [tmp_9938[0] + tmp_9939[0], tmp_9938[1] + tmp_9939[1], tmp_9938[2] + tmp_9939[2]];
    signal tmp_9941[3] <== [tmp_9940[0] + evals[66][0], tmp_9940[1] + evals[66][1], tmp_9940[2] + evals[66][2]];
    signal tmp_9942[3] <== [tmp_9941[0] - evals[84][0], tmp_9941[1] - evals[84][1], tmp_9941[2] - evals[84][2]];
    signal tmp_9943[3] <== CMul()(evals[56], tmp_9942);
    signal tmp_9944[3] <== [tmp_9933[0] + tmp_9943[0], tmp_9933[1] + tmp_9943[1], tmp_9933[2] + tmp_9943[2]];
    signal tmp_9945[3] <== CMul()(challengeQ, tmp_9944);
    signal tmp_9946[3] <== [1 - evals[76][0], -evals[76][1], -evals[76][2]];
    signal tmp_9947[3] <== [1 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_9948[3] <== CMul()(tmp_9946, tmp_9947);
    signal tmp_9949[3] <== CMul()(evals[58], tmp_9948);
    signal tmp_9950[3] <== [evals[64][0] - evals[78][0], evals[64][1] - evals[78][1], evals[64][2] - evals[78][2]];
    signal tmp_9951[3] <== CMul()(tmp_9949, tmp_9950);
    signal tmp_9952[3] <== [tmp_9945[0] + tmp_9951[0], tmp_9945[1] + tmp_9951[1], tmp_9945[2] + tmp_9951[2]];
    signal tmp_9953[3] <== CMul()(challengeQ, tmp_9952);
    signal tmp_9955[3] <== [evals[65][0] - evals[79][0], evals[65][1] - evals[79][1], evals[65][2] - evals[79][2]];
    signal tmp_9956[3] <== CMul()(tmp_9949, tmp_9955);
    signal tmp_9957[3] <== [tmp_9953[0] + tmp_9956[0], tmp_9953[1] + tmp_9956[1], tmp_9953[2] + tmp_9956[2]];
    signal tmp_9958[3] <== CMul()(challengeQ, tmp_9957);
    signal tmp_9960[3] <== [evals[66][0] - evals[80][0], evals[66][1] - evals[80][1], evals[66][2] - evals[80][2]];
    signal tmp_9961[3] <== CMul()(tmp_9949, tmp_9960);
    signal tmp_9962[3] <== [tmp_9958[0] + tmp_9961[0], tmp_9958[1] + tmp_9961[1], tmp_9958[2] + tmp_9961[2]];
    signal tmp_9963[3] <== CMul()(challengeQ, tmp_9962);
    signal tmp_9964[3] <== CMul()(evals[76], tmp_9947);
    signal tmp_9965[3] <== CMul()(evals[58], tmp_9964);
    signal tmp_9966[3] <== [evals[67][0] - evals[78][0], evals[67][1] - evals[78][1], evals[67][2] - evals[78][2]];
    signal tmp_9967[3] <== CMul()(tmp_9965, tmp_9966);
    signal tmp_9968[3] <== [tmp_9963[0] + tmp_9967[0], tmp_9963[1] + tmp_9967[1], tmp_9963[2] + tmp_9967[2]];
    signal tmp_9969[3] <== CMul()(challengeQ, tmp_9968);
    signal tmp_9971[3] <== [evals[68][0] - evals[79][0], evals[68][1] - evals[79][1], evals[68][2] - evals[79][2]];
    signal tmp_9972[3] <== CMul()(tmp_9965, tmp_9971);
    signal tmp_9973[3] <== [tmp_9969[0] + tmp_9972[0], tmp_9969[1] + tmp_9972[1], tmp_9969[2] + tmp_9972[2]];
    signal tmp_9974[3] <== CMul()(challengeQ, tmp_9973);
    signal tmp_9976[3] <== [evals[69][0] - evals[80][0], evals[69][1] - evals[80][1], evals[69][2] - evals[80][2]];
    signal tmp_9977[3] <== CMul()(tmp_9965, tmp_9976);
    signal tmp_9978[3] <== [tmp_9974[0] + tmp_9977[0], tmp_9974[1] + tmp_9977[1], tmp_9974[2] + tmp_9977[2]];
    signal tmp_9979[3] <== CMul()(challengeQ, tmp_9978);
    signal tmp_9980[3] <== CMul()(tmp_9946, evals[77]);
    signal tmp_9981[3] <== CMul()(evals[58], tmp_9980);
    signal tmp_9982[3] <== [evals[70][0] - evals[78][0], evals[70][1] - evals[78][1], evals[70][2] - evals[78][2]];
    signal tmp_9983[3] <== CMul()(tmp_9981, tmp_9982);
    signal tmp_9984[3] <== [tmp_9979[0] + tmp_9983[0], tmp_9979[1] + tmp_9983[1], tmp_9979[2] + tmp_9983[2]];
    signal tmp_9985[3] <== CMul()(challengeQ, tmp_9984);
    signal tmp_9987[3] <== [evals[71][0] - evals[79][0], evals[71][1] - evals[79][1], evals[71][2] - evals[79][2]];
    signal tmp_9988[3] <== CMul()(tmp_9981, tmp_9987);
    signal tmp_9989[3] <== [tmp_9985[0] + tmp_9988[0], tmp_9985[1] + tmp_9988[1], tmp_9985[2] + tmp_9988[2]];
    signal tmp_9990[3] <== CMul()(challengeQ, tmp_9989);
    signal tmp_9992[3] <== [evals[72][0] - evals[80][0], evals[72][1] - evals[80][1], evals[72][2] - evals[80][2]];
    signal tmp_9993[3] <== CMul()(tmp_9981, tmp_9992);
    signal tmp_9994[3] <== [tmp_9990[0] + tmp_9993[0], tmp_9990[1] + tmp_9993[1], tmp_9990[2] + tmp_9993[2]];
    signal tmp_9995[3] <== CMul()(challengeQ, tmp_9994);
    signal tmp_9997[3] <== CMul()(evals[58], tmp_6873);
    signal tmp_9998[3] <== [evals[73][0] - evals[78][0], evals[73][1] - evals[78][1], evals[73][2] - evals[78][2]];
    signal tmp_9999[3] <== CMul()(tmp_9997, tmp_9998);
    signal tmp_10000[3] <== [tmp_9995[0] + tmp_9999[0], tmp_9995[1] + tmp_9999[1], tmp_9995[2] + tmp_9999[2]];
    signal tmp_10001[3] <== CMul()(challengeQ, tmp_10000);
    signal tmp_10003[3] <== [evals[74][0] - evals[79][0], evals[74][1] - evals[79][1], evals[74][2] - evals[79][2]];
    signal tmp_10004[3] <== CMul()(tmp_9997, tmp_10003);
    signal tmp_10005[3] <== [tmp_10001[0] + tmp_10004[0], tmp_10001[1] + tmp_10004[1], tmp_10001[2] + tmp_10004[2]];
    signal tmp_10006[3] <== CMul()(challengeQ, tmp_10005);
    signal tmp_10008[3] <== [evals[75][0] - evals[80][0], evals[75][1] - evals[80][1], evals[75][2] - evals[80][2]];
    signal tmp_10009[3] <== CMul()(tmp_9997, tmp_10008);
    signal tmp_10010[3] <== [tmp_10006[0] + tmp_10009[0], tmp_10006[1] + tmp_10009[1], tmp_10006[2] + tmp_10009[2]];
    signal tmp_10011[3] <== CMul()(challengeQ, tmp_10010);
    signal tmp_10013[3] <== CMul()(evals[76], tmp_9946);
    signal tmp_10014[3] <== CMul()(evals[58], tmp_10013);
    signal tmp_10015[3] <== [tmp_10011[0] + tmp_10014[0], tmp_10011[1] + tmp_10014[1], tmp_10011[2] + tmp_10014[2]];
    signal tmp_10016[3] <== CMul()(challengeQ, tmp_10015);
    signal tmp_10018[3] <== CMul()(evals[77], tmp_9947);
    signal tmp_10019[3] <== CMul()(evals[58], tmp_10018);
    signal tmp_10020[3] <== [tmp_10016[0] + tmp_10019[0], tmp_10016[1] + tmp_10019[1], tmp_10016[2] + tmp_10019[2]];
    signal tmp_10021[3] <== CMul()(challengeQ, tmp_10020);
    signal tmp_10022[3] <== [1 - evals[80][0], -evals[80][1], -evals[80][2]];
    signal tmp_10023[3] <== [1 - evals[81][0], -evals[81][1], -evals[81][2]];
    signal tmp_10024[3] <== CMul()(tmp_10022, tmp_10023);
    signal tmp_10025[3] <== CMul()(evals[59], tmp_10024);
    signal tmp_10026[3] <== [evals[64][0] - evals[82][0], evals[64][1] - evals[82][1], evals[64][2] - evals[82][2]];
    signal tmp_10027[3] <== CMul()(tmp_10025, tmp_10026);
    signal tmp_10028[3] <== [tmp_10021[0] + tmp_10027[0], tmp_10021[1] + tmp_10027[1], tmp_10021[2] + tmp_10027[2]];
    signal tmp_10029[3] <== CMul()(challengeQ, tmp_10028);
    signal tmp_10031[3] <== [evals[65][0] - evals[83][0], evals[65][1] - evals[83][1], evals[65][2] - evals[83][2]];
    signal tmp_10032[3] <== CMul()(tmp_10025, tmp_10031);
    signal tmp_10033[3] <== [tmp_10029[0] + tmp_10032[0], tmp_10029[1] + tmp_10032[1], tmp_10029[2] + tmp_10032[2]];
    signal tmp_10034[3] <== CMul()(challengeQ, tmp_10033);
    signal tmp_10036[3] <== [evals[66][0] - evals[84][0], evals[66][1] - evals[84][1], evals[66][2] - evals[84][2]];
    signal tmp_10037[3] <== CMul()(tmp_10025, tmp_10036);
    signal tmp_10038[3] <== [tmp_10034[0] + tmp_10037[0], tmp_10034[1] + tmp_10037[1], tmp_10034[2] + tmp_10037[2]];
    signal tmp_10039[3] <== CMul()(challengeQ, tmp_10038);
    signal tmp_10041[3] <== [evals[67][0] - evals[85][0], evals[67][1] - evals[85][1], evals[67][2] - evals[85][2]];
    signal tmp_10042[3] <== CMul()(tmp_10025, tmp_10041);
    signal tmp_10043[3] <== [tmp_10039[0] + tmp_10042[0], tmp_10039[1] + tmp_10042[1], tmp_10039[2] + tmp_10042[2]];
    signal tmp_10044[3] <== CMul()(challengeQ, tmp_10043);
    signal tmp_10046[3] <== CMul()(evals[80], tmp_10023);
    signal tmp_10047[3] <== CMul()(evals[59], tmp_10046);
    signal tmp_10048[3] <== [evals[68][0] - evals[82][0], evals[68][1] - evals[82][1], evals[68][2] - evals[82][2]];
    signal tmp_10049[3] <== CMul()(tmp_10047, tmp_10048);
    signal tmp_10050[3] <== [tmp_10044[0] + tmp_10049[0], tmp_10044[1] + tmp_10049[1], tmp_10044[2] + tmp_10049[2]];
    signal tmp_10051[3] <== CMul()(challengeQ, tmp_10050);
    signal tmp_10053[3] <== [evals[69][0] - evals[83][0], evals[69][1] - evals[83][1], evals[69][2] - evals[83][2]];
    signal tmp_10054[3] <== CMul()(tmp_10047, tmp_10053);
    signal tmp_10055[3] <== [tmp_10051[0] + tmp_10054[0], tmp_10051[1] + tmp_10054[1], tmp_10051[2] + tmp_10054[2]];
    signal tmp_10056[3] <== CMul()(challengeQ, tmp_10055);
    signal tmp_10058[3] <== [evals[70][0] - evals[84][0], evals[70][1] - evals[84][1], evals[70][2] - evals[84][2]];
    signal tmp_10059[3] <== CMul()(tmp_10047, tmp_10058);
    signal tmp_10060[3] <== [tmp_10056[0] + tmp_10059[0], tmp_10056[1] + tmp_10059[1], tmp_10056[2] + tmp_10059[2]];
    signal tmp_10061[3] <== CMul()(challengeQ, tmp_10060);
    signal tmp_10063[3] <== [evals[71][0] - evals[85][0], evals[71][1] - evals[85][1], evals[71][2] - evals[85][2]];
    signal tmp_10064[3] <== CMul()(tmp_10047, tmp_10063);
    signal tmp_10065[3] <== [tmp_10061[0] + tmp_10064[0], tmp_10061[1] + tmp_10064[1], tmp_10061[2] + tmp_10064[2]];
    signal tmp_10066[3] <== CMul()(challengeQ, tmp_10065);
    signal tmp_10068[3] <== CMul()(tmp_10022, evals[81]);
    signal tmp_10069[3] <== CMul()(evals[59], tmp_10068);
    signal tmp_10070[3] <== [evals[72][0] - evals[82][0], evals[72][1] - evals[82][1], evals[72][2] - evals[82][2]];
    signal tmp_10071[3] <== CMul()(tmp_10069, tmp_10070);
    signal tmp_10072[3] <== [tmp_10066[0] + tmp_10071[0], tmp_10066[1] + tmp_10071[1], tmp_10066[2] + tmp_10071[2]];
    signal tmp_10073[3] <== CMul()(challengeQ, tmp_10072);
    signal tmp_10075[3] <== [evals[73][0] - evals[83][0], evals[73][1] - evals[83][1], evals[73][2] - evals[83][2]];
    signal tmp_10076[3] <== CMul()(tmp_10069, tmp_10075);
    signal tmp_10077[3] <== [tmp_10073[0] + tmp_10076[0], tmp_10073[1] + tmp_10076[1], tmp_10073[2] + tmp_10076[2]];
    signal tmp_10078[3] <== CMul()(challengeQ, tmp_10077);
    signal tmp_10080[3] <== [evals[74][0] - evals[84][0], evals[74][1] - evals[84][1], evals[74][2] - evals[84][2]];
    signal tmp_10081[3] <== CMul()(tmp_10069, tmp_10080);
    signal tmp_10082[3] <== [tmp_10078[0] + tmp_10081[0], tmp_10078[1] + tmp_10081[1], tmp_10078[2] + tmp_10081[2]];
    signal tmp_10083[3] <== CMul()(challengeQ, tmp_10082);
    signal tmp_10085[3] <== [evals[75][0] - evals[85][0], evals[75][1] - evals[85][1], evals[75][2] - evals[85][2]];
    signal tmp_10086[3] <== CMul()(tmp_10069, tmp_10085);
    signal tmp_10087[3] <== [tmp_10083[0] + tmp_10086[0], tmp_10083[1] + tmp_10086[1], tmp_10083[2] + tmp_10086[2]];
    signal tmp_10088[3] <== CMul()(challengeQ, tmp_10087);
    signal tmp_10089[3] <== CMul()(evals[80], evals[81]);
    signal tmp_10090[3] <== CMul()(evals[59], tmp_10089);
    signal tmp_10091[3] <== [evals[76][0] - evals[82][0], evals[76][1] - evals[82][1], evals[76][2] - evals[82][2]];
    signal tmp_10092[3] <== CMul()(tmp_10090, tmp_10091);
    signal tmp_10093[3] <== [tmp_10088[0] + tmp_10092[0], tmp_10088[1] + tmp_10092[1], tmp_10088[2] + tmp_10092[2]];
    signal tmp_10094[3] <== CMul()(challengeQ, tmp_10093);
    signal tmp_10096[3] <== [evals[77][0] - evals[83][0], evals[77][1] - evals[83][1], evals[77][2] - evals[83][2]];
    signal tmp_10097[3] <== CMul()(tmp_10090, tmp_10096);
    signal tmp_10098[3] <== [tmp_10094[0] + tmp_10097[0], tmp_10094[1] + tmp_10097[1], tmp_10094[2] + tmp_10097[2]];
    signal tmp_10099[3] <== CMul()(challengeQ, tmp_10098);
    signal tmp_10101[3] <== [evals[78][0] - evals[84][0], evals[78][1] - evals[84][1], evals[78][2] - evals[84][2]];
    signal tmp_10102[3] <== CMul()(tmp_10090, tmp_10101);
    signal tmp_10103[3] <== [tmp_10099[0] + tmp_10102[0], tmp_10099[1] + tmp_10102[1], tmp_10099[2] + tmp_10102[2]];
    signal tmp_10104[3] <== CMul()(challengeQ, tmp_10103);
    signal tmp_10106[3] <== [evals[79][0] - evals[85][0], evals[79][1] - evals[85][1], evals[79][2] - evals[85][2]];
    signal tmp_10107[3] <== CMul()(tmp_10090, tmp_10106);
    signal tmp_10108[3] <== [tmp_10104[0] + tmp_10107[0], tmp_10104[1] + tmp_10107[1], tmp_10104[2] + tmp_10107[2]];
    signal tmp_10109[3] <== CMul()(challengeQ, tmp_10108);
    signal tmp_10111[3] <== CMul()(evals[80], tmp_10022);
    signal tmp_10112[3] <== CMul()(evals[59], tmp_10111);
    signal tmp_10113[3] <== [tmp_10109[0] + tmp_10112[0], tmp_10109[1] + tmp_10112[1], tmp_10109[2] + tmp_10112[2]];
    signal tmp_10114[3] <== CMul()(challengeQ, tmp_10113);
    signal tmp_10116[3] <== CMul()(evals[81], tmp_10023);
    signal tmp_10117[3] <== CMul()(evals[59], tmp_10116);
    signal tmp_10118[3] <== [tmp_10114[0] + tmp_10117[0], tmp_10114[1] + tmp_10117[1], tmp_10114[2] + tmp_10117[2]];
    signal tmp_10119[3] <== CMul()(challengeQ, tmp_10118);
    signal tmp_10120[3] <== CMul()(evals[62], challengesStage2[0]);
    signal tmp_10121[3] <== [tmp_10120[0] + evals[64][0], tmp_10120[1] + evals[64][1], tmp_10120[2] + evals[64][2]];
    signal tmp_10122[3] <== CMul()(tmp_10121, challengesStage2[0]);
    signal tmp_10123[3] <== [tmp_10122[0] + 1, tmp_10122[1], tmp_10122[2]];
    signal tmp_10124[3] <== [tmp_10123[0] + challengesStage2[1][0], tmp_10123[1] + challengesStage2[1][1], tmp_10123[2] + challengesStage2[1][2]];
    signal tmp_10125[3] <== [12275445934081160404 * evals[62][0], 12275445934081160404 * evals[62][1], 12275445934081160404 * evals[62][2]];
    signal tmp_10126[3] <== CMul()(tmp_10125, challengesStage2[0]);
    signal tmp_10127[3] <== [tmp_10126[0] + evals[65][0], tmp_10126[1] + evals[65][1], tmp_10126[2] + evals[65][2]];
    signal tmp_10128[3] <== CMul()(tmp_10127, challengesStage2[0]);
    signal tmp_10129[3] <== [tmp_10128[0] + 1, tmp_10128[1], tmp_10128[2]];
    signal tmp_10130[3] <== [tmp_10129[0] + challengesStage2[1][0], tmp_10129[1] + challengesStage2[1][1], tmp_10129[2] + challengesStage2[1][2]];
    signal tmp_10131[3] <== CMul()(tmp_10124, tmp_10130);
    signal tmp_10132[3] <== [4756475762779100925 * evals[62][0], 4756475762779100925 * evals[62][1], 4756475762779100925 * evals[62][2]];
    signal tmp_10133[3] <== CMul()(tmp_10132, challengesStage2[0]);
    signal tmp_10134[3] <== [tmp_10133[0] + evals[66][0], tmp_10133[1] + evals[66][1], tmp_10133[2] + evals[66][2]];
    signal tmp_10135[3] <== CMul()(tmp_10134, challengesStage2[0]);
    signal tmp_10136[3] <== [tmp_10135[0] + 1, tmp_10135[1], tmp_10135[2]];
    signal tmp_10137[3] <== [tmp_10136[0] + challengesStage2[1][0], tmp_10136[1] + challengesStage2[1][1], tmp_10136[2] + challengesStage2[1][2]];
    signal tmp_10138[3] <== CMul()(tmp_10131, tmp_10137);
    signal tmp_10139[3] <== [1279992132519201448 * evals[62][0], 1279992132519201448 * evals[62][1], 1279992132519201448 * evals[62][2]];
    signal tmp_10140[3] <== CMul()(tmp_10139, challengesStage2[0]);
    signal tmp_10141[3] <== [tmp_10140[0] + evals[67][0], tmp_10140[1] + evals[67][1], tmp_10140[2] + evals[67][2]];
    signal tmp_10142[3] <== CMul()(tmp_10141, challengesStage2[0]);
    signal tmp_10143[3] <== [tmp_10142[0] + 1, tmp_10142[1], tmp_10142[2]];
    signal tmp_10144[3] <== [tmp_10143[0] + challengesStage2[1][0], tmp_10143[1] + challengesStage2[1][1], tmp_10143[2] + challengesStage2[1][2]];
    signal tmp_10145[3] <== CMul()(tmp_10138, tmp_10144);
    signal tmp_10146[3] <== [8312008622371998338 * evals[62][0], 8312008622371998338 * evals[62][1], 8312008622371998338 * evals[62][2]];
    signal tmp_10147[3] <== CMul()(tmp_10146, challengesStage2[0]);
    signal tmp_10148[3] <== [tmp_10147[0] + evals[68][0], tmp_10147[1] + evals[68][1], tmp_10147[2] + evals[68][2]];
    signal tmp_10149[3] <== CMul()(tmp_10148, challengesStage2[0]);
    signal tmp_10150[3] <== [tmp_10149[0] + 1, tmp_10149[1], tmp_10149[2]];
    signal tmp_10151[3] <== [tmp_10150[0] + challengesStage2[1][0], tmp_10150[1] + challengesStage2[1][1], tmp_10150[2] + challengesStage2[1][2]];
    signal tmp_10152[3] <== CMul()(tmp_10145, tmp_10151);
    signal tmp_10153[3] <== [7781028390488215464 * evals[62][0], 7781028390488215464 * evals[62][1], 7781028390488215464 * evals[62][2]];
    signal tmp_10154[3] <== CMul()(tmp_10153, challengesStage2[0]);
    signal tmp_10155[3] <== [tmp_10154[0] + evals[69][0], tmp_10154[1] + evals[69][1], tmp_10154[2] + evals[69][2]];
    signal tmp_10156[3] <== CMul()(tmp_10155, challengesStage2[0]);
    signal tmp_10157[3] <== [tmp_10156[0] + 1, tmp_10156[1], tmp_10156[2]];
    signal tmp_10158[3] <== [tmp_10157[0] + challengesStage2[1][0], tmp_10157[1] + challengesStage2[1][1], tmp_10157[2] + challengesStage2[1][2]];
    signal tmp_10159[3] <== CMul()(tmp_10152, tmp_10158);
    signal tmp_10160[3] <== [11302600489504509467 * evals[62][0], 11302600489504509467 * evals[62][1], 11302600489504509467 * evals[62][2]];
    signal tmp_10161[3] <== CMul()(tmp_10160, challengesStage2[0]);
    signal tmp_10162[3] <== [tmp_10161[0] + evals[70][0], tmp_10161[1] + evals[70][1], tmp_10161[2] + evals[70][2]];
    signal tmp_10163[3] <== CMul()(tmp_10162, challengesStage2[0]);
    signal tmp_10164[3] <== [tmp_10163[0] + 1, tmp_10163[1], tmp_10163[2]];
    signal tmp_10165[3] <== [tmp_10164[0] + challengesStage2[1][0], tmp_10164[1] + challengesStage2[1][1], tmp_10164[2] + challengesStage2[1][2]];
    signal tmp_10166[3] <== CMul()(tmp_10159, tmp_10165);
    signal tmp_10167[3] <== CMul()(evals[113], tmp_10166);
    signal tmp_10168[3] <== CMul()(evals[17], challengesStage2[0]);
    signal tmp_10169[3] <== [tmp_10168[0] + evals[64][0], tmp_10168[1] + evals[64][1], tmp_10168[2] + evals[64][2]];
    signal tmp_10170[3] <== CMul()(tmp_10169, challengesStage2[0]);
    signal tmp_10171[3] <== [tmp_10170[0] + 1, tmp_10170[1], tmp_10170[2]];
    signal tmp_10172[3] <== [tmp_10171[0] + challengesStage2[1][0], tmp_10171[1] + challengesStage2[1][1], tmp_10171[2] + challengesStage2[1][2]];
    signal tmp_10173[3] <== CMul()(evals[18], challengesStage2[0]);
    signal tmp_10174[3] <== [tmp_10173[0] + evals[65][0], tmp_10173[1] + evals[65][1], tmp_10173[2] + evals[65][2]];
    signal tmp_10175[3] <== CMul()(tmp_10174, challengesStage2[0]);
    signal tmp_10176[3] <== [tmp_10175[0] + 1, tmp_10175[1], tmp_10175[2]];
    signal tmp_10177[3] <== [tmp_10176[0] + challengesStage2[1][0], tmp_10176[1] + challengesStage2[1][1], tmp_10176[2] + challengesStage2[1][2]];
    signal tmp_10178[3] <== CMul()(tmp_10172, tmp_10177);
    signal tmp_10179[3] <== CMul()(evals[19], challengesStage2[0]);
    signal tmp_10180[3] <== [tmp_10179[0] + evals[66][0], tmp_10179[1] + evals[66][1], tmp_10179[2] + evals[66][2]];
    signal tmp_10181[3] <== CMul()(tmp_10180, challengesStage2[0]);
    signal tmp_10182[3] <== [tmp_10181[0] + 1, tmp_10181[1], tmp_10181[2]];
    signal tmp_10183[3] <== [tmp_10182[0] + challengesStage2[1][0], tmp_10182[1] + challengesStage2[1][1], tmp_10182[2] + challengesStage2[1][2]];
    signal tmp_10184[3] <== CMul()(tmp_10178, tmp_10183);
    signal tmp_10185[3] <== CMul()(evals[20], challengesStage2[0]);
    signal tmp_10186[3] <== [tmp_10185[0] + evals[67][0], tmp_10185[1] + evals[67][1], tmp_10185[2] + evals[67][2]];
    signal tmp_10187[3] <== CMul()(tmp_10186, challengesStage2[0]);
    signal tmp_10188[3] <== [tmp_10187[0] + 1, tmp_10187[1], tmp_10187[2]];
    signal tmp_10189[3] <== [tmp_10188[0] + challengesStage2[1][0], tmp_10188[1] + challengesStage2[1][1], tmp_10188[2] + challengesStage2[1][2]];
    signal tmp_10190[3] <== CMul()(tmp_10184, tmp_10189);
    signal tmp_10191[3] <== CMul()(evals[21], challengesStage2[0]);
    signal tmp_10192[3] <== [tmp_10191[0] + evals[68][0], tmp_10191[1] + evals[68][1], tmp_10191[2] + evals[68][2]];
    signal tmp_10193[3] <== CMul()(tmp_10192, challengesStage2[0]);
    signal tmp_10194[3] <== [tmp_10193[0] + 1, tmp_10193[1], tmp_10193[2]];
    signal tmp_10195[3] <== [tmp_10194[0] + challengesStage2[1][0], tmp_10194[1] + challengesStage2[1][1], tmp_10194[2] + challengesStage2[1][2]];
    signal tmp_10196[3] <== CMul()(tmp_10190, tmp_10195);
    signal tmp_10197[3] <== CMul()(evals[22], challengesStage2[0]);
    signal tmp_10198[3] <== [tmp_10197[0] + evals[69][0], tmp_10197[1] + evals[69][1], tmp_10197[2] + evals[69][2]];
    signal tmp_10199[3] <== CMul()(tmp_10198, challengesStage2[0]);
    signal tmp_10200[3] <== [tmp_10199[0] + 1, tmp_10199[1], tmp_10199[2]];
    signal tmp_10201[3] <== [tmp_10200[0] + challengesStage2[1][0], tmp_10200[1] + challengesStage2[1][1], tmp_10200[2] + challengesStage2[1][2]];
    signal tmp_10202[3] <== CMul()(tmp_10196, tmp_10201);
    signal tmp_10203[3] <== CMul()(evals[23], challengesStage2[0]);
    signal tmp_10204[3] <== [tmp_10203[0] + evals[70][0], tmp_10203[1] + evals[70][1], tmp_10203[2] + evals[70][2]];
    signal tmp_10205[3] <== CMul()(tmp_10204, challengesStage2[0]);
    signal tmp_10206[3] <== [tmp_10205[0] + 1, tmp_10205[1], tmp_10205[2]];
    signal tmp_10207[3] <== [tmp_10206[0] + challengesStage2[1][0], tmp_10206[1] + challengesStage2[1][1], tmp_10206[2] + challengesStage2[1][2]];
    signal tmp_10208[3] <== CMul()(tmp_10202, tmp_10207);
    signal tmp_10209[3] <== CMul()(evals[24], challengesStage2[0]);
    signal tmp_10210[3] <== [tmp_10209[0] + evals[71][0], tmp_10209[1] + evals[71][1], tmp_10209[2] + evals[71][2]];
    signal tmp_10211[3] <== CMul()(tmp_10210, challengesStage2[0]);
    signal tmp_10212[3] <== [tmp_10211[0] + 1, tmp_10211[1], tmp_10211[2]];
    signal tmp_10213[3] <== [tmp_10212[0] + challengesStage2[1][0], tmp_10212[1] + challengesStage2[1][1], tmp_10212[2] + challengesStage2[1][2]];
    signal tmp_10214[3] <== CMul()(tmp_10208, tmp_10213);
    signal tmp_10215[3] <== [tmp_10167[0] - tmp_10214[0], tmp_10167[1] - tmp_10214[1], tmp_10167[2] - tmp_10214[2]];
    signal tmp_10216[3] <== [tmp_10119[0] + tmp_10215[0], tmp_10119[1] + tmp_10215[1], tmp_10119[2] + tmp_10215[2]];
    signal tmp_10217[3] <== CMul()(challengeQ, tmp_10216);
    signal tmp_10218[3] <== [4549350404001778198 * evals[62][0], 4549350404001778198 * evals[62][1], 4549350404001778198 * evals[62][2]];
    signal tmp_10219[3] <== CMul()(tmp_10218, challengesStage2[0]);
    signal tmp_10220[3] <== [tmp_10219[0] + evals[71][0], tmp_10219[1] + evals[71][1], tmp_10219[2] + evals[71][2]];
    signal tmp_10221[3] <== CMul()(tmp_10220, challengesStage2[0]);
    signal tmp_10222[3] <== [tmp_10221[0] + 1, tmp_10221[1], tmp_10221[2]];
    signal tmp_10223[3] <== [tmp_10222[0] + challengesStage2[1][0], tmp_10222[1] + challengesStage2[1][1], tmp_10222[2] + challengesStage2[1][2]];
    signal tmp_10224[3] <== [3688660304411827445 * evals[62][0], 3688660304411827445 * evals[62][1], 3688660304411827445 * evals[62][2]];
    signal tmp_10225[3] <== CMul()(tmp_10224, challengesStage2[0]);
    signal tmp_10226[3] <== [tmp_10225[0] + evals[72][0], tmp_10225[1] + evals[72][1], tmp_10225[2] + evals[72][2]];
    signal tmp_10227[3] <== CMul()(tmp_10226, challengesStage2[0]);
    signal tmp_10228[3] <== [tmp_10227[0] + 1, tmp_10227[1], tmp_10227[2]];
    signal tmp_10229[3] <== [tmp_10228[0] + challengesStage2[1][0], tmp_10228[1] + challengesStage2[1][1], tmp_10228[2] + challengesStage2[1][2]];
    signal tmp_10230[3] <== CMul()(tmp_10223, tmp_10229);
    signal tmp_10231[3] <== [16725109960945739746 * evals[62][0], 16725109960945739746 * evals[62][1], 16725109960945739746 * evals[62][2]];
    signal tmp_10232[3] <== CMul()(tmp_10231, challengesStage2[0]);
    signal tmp_10233[3] <== [tmp_10232[0] + evals[73][0], tmp_10232[1] + evals[73][1], tmp_10232[2] + evals[73][2]];
    signal tmp_10234[3] <== CMul()(tmp_10233, challengesStage2[0]);
    signal tmp_10235[3] <== [tmp_10234[0] + 1, tmp_10234[1], tmp_10234[2]];
    signal tmp_10236[3] <== [tmp_10235[0] + challengesStage2[1][0], tmp_10235[1] + challengesStage2[1][1], tmp_10235[2] + challengesStage2[1][2]];
    signal tmp_10237[3] <== CMul()(tmp_10230, tmp_10236);
    signal tmp_10238[3] <== [16538725463549498621 * evals[62][0], 16538725463549498621 * evals[62][1], 16538725463549498621 * evals[62][2]];
    signal tmp_10239[3] <== CMul()(tmp_10238, challengesStage2[0]);
    signal tmp_10240[3] <== [tmp_10239[0] + evals[74][0], tmp_10239[1] + evals[74][1], tmp_10239[2] + evals[74][2]];
    signal tmp_10241[3] <== CMul()(tmp_10240, challengesStage2[0]);
    signal tmp_10242[3] <== [tmp_10241[0] + 1, tmp_10241[1], tmp_10241[2]];
    signal tmp_10243[3] <== [tmp_10242[0] + challengesStage2[1][0], tmp_10242[1] + challengesStage2[1][1], tmp_10242[2] + challengesStage2[1][2]];
    signal tmp_10244[3] <== CMul()(tmp_10237, tmp_10243);
    signal tmp_10245[3] <== [12756200801261202346 * evals[62][0], 12756200801261202346 * evals[62][1], 12756200801261202346 * evals[62][2]];
    signal tmp_10246[3] <== CMul()(tmp_10245, challengesStage2[0]);
    signal tmp_10247[3] <== [tmp_10246[0] + evals[75][0], tmp_10246[1] + evals[75][1], tmp_10246[2] + evals[75][2]];
    signal tmp_10248[3] <== CMul()(tmp_10247, challengesStage2[0]);
    signal tmp_10249[3] <== [tmp_10248[0] + 1, tmp_10248[1], tmp_10248[2]];
    signal tmp_10250[3] <== [tmp_10249[0] + challengesStage2[1][0], tmp_10249[1] + challengesStage2[1][1], tmp_10249[2] + challengesStage2[1][2]];
    signal tmp_10251[3] <== CMul()(tmp_10244, tmp_10250);
    signal tmp_10252[3] <== [15099809066790865939 * evals[62][0], 15099809066790865939 * evals[62][1], 15099809066790865939 * evals[62][2]];
    signal tmp_10253[3] <== CMul()(tmp_10252, challengesStage2[0]);
    signal tmp_10254[3] <== [tmp_10253[0] + evals[76][0], tmp_10253[1] + evals[76][1], tmp_10253[2] + evals[76][2]];
    signal tmp_10255[3] <== CMul()(tmp_10254, challengesStage2[0]);
    signal tmp_10256[3] <== [tmp_10255[0] + 1, tmp_10255[1], tmp_10255[2]];
    signal tmp_10257[3] <== [tmp_10256[0] + challengesStage2[1][0], tmp_10256[1] + challengesStage2[1][1], tmp_10256[2] + challengesStage2[1][2]];
    signal tmp_10258[3] <== CMul()(tmp_10251, tmp_10257);
    signal tmp_10259[3] <== [17214954929431464349 * evals[62][0], 17214954929431464349 * evals[62][1], 17214954929431464349 * evals[62][2]];
    signal tmp_10260[3] <== CMul()(tmp_10259, challengesStage2[0]);
    signal tmp_10261[3] <== [tmp_10260[0] + evals[77][0], tmp_10260[1] + evals[77][1], tmp_10260[2] + evals[77][2]];
    signal tmp_10262[3] <== CMul()(tmp_10261, challengesStage2[0]);
    signal tmp_10263[3] <== [tmp_10262[0] + 1, tmp_10262[1], tmp_10262[2]];
    signal tmp_10264[3] <== [tmp_10263[0] + challengesStage2[1][0], tmp_10263[1] + challengesStage2[1][1], tmp_10263[2] + challengesStage2[1][2]];
    signal tmp_10265[3] <== CMul()(tmp_10258, tmp_10264);
    signal tmp_10266[3] <== CMul()(evals[114], tmp_10265);
    signal tmp_10267[3] <== CMul()(evals[25], challengesStage2[0]);
    signal tmp_10268[3] <== [tmp_10267[0] + evals[72][0], tmp_10267[1] + evals[72][1], tmp_10267[2] + evals[72][2]];
    signal tmp_10269[3] <== CMul()(tmp_10268, challengesStage2[0]);
    signal tmp_10270[3] <== [tmp_10269[0] + 1, tmp_10269[1], tmp_10269[2]];
    signal tmp_10271[3] <== [tmp_10270[0] + challengesStage2[1][0], tmp_10270[1] + challengesStage2[1][1], tmp_10270[2] + challengesStage2[1][2]];
    signal tmp_10272[3] <== CMul()(evals[113], tmp_10271);
    signal tmp_10273[3] <== CMul()(evals[26], challengesStage2[0]);
    signal tmp_10274[3] <== [tmp_10273[0] + evals[73][0], tmp_10273[1] + evals[73][1], tmp_10273[2] + evals[73][2]];
    signal tmp_10275[3] <== CMul()(tmp_10274, challengesStage2[0]);
    signal tmp_10276[3] <== [tmp_10275[0] + 1, tmp_10275[1], tmp_10275[2]];
    signal tmp_10277[3] <== [tmp_10276[0] + challengesStage2[1][0], tmp_10276[1] + challengesStage2[1][1], tmp_10276[2] + challengesStage2[1][2]];
    signal tmp_10278[3] <== CMul()(tmp_10272, tmp_10277);
    signal tmp_10279[3] <== CMul()(evals[27], challengesStage2[0]);
    signal tmp_10280[3] <== [tmp_10279[0] + evals[74][0], tmp_10279[1] + evals[74][1], tmp_10279[2] + evals[74][2]];
    signal tmp_10281[3] <== CMul()(tmp_10280, challengesStage2[0]);
    signal tmp_10282[3] <== [tmp_10281[0] + 1, tmp_10281[1], tmp_10281[2]];
    signal tmp_10283[3] <== [tmp_10282[0] + challengesStage2[1][0], tmp_10282[1] + challengesStage2[1][1], tmp_10282[2] + challengesStage2[1][2]];
    signal tmp_10284[3] <== CMul()(tmp_10278, tmp_10283);
    signal tmp_10285[3] <== CMul()(evals[28], challengesStage2[0]);
    signal tmp_10286[3] <== [tmp_10285[0] + evals[75][0], tmp_10285[1] + evals[75][1], tmp_10285[2] + evals[75][2]];
    signal tmp_10287[3] <== CMul()(tmp_10286, challengesStage2[0]);
    signal tmp_10288[3] <== [tmp_10287[0] + 1, tmp_10287[1], tmp_10287[2]];
    signal tmp_10289[3] <== [tmp_10288[0] + challengesStage2[1][0], tmp_10288[1] + challengesStage2[1][1], tmp_10288[2] + challengesStage2[1][2]];
    signal tmp_10290[3] <== CMul()(tmp_10284, tmp_10289);
    signal tmp_10291[3] <== CMul()(evals[29], challengesStage2[0]);
    signal tmp_10292[3] <== [tmp_10291[0] + evals[76][0], tmp_10291[1] + evals[76][1], tmp_10291[2] + evals[76][2]];
    signal tmp_10293[3] <== CMul()(tmp_10292, challengesStage2[0]);
    signal tmp_10294[3] <== [tmp_10293[0] + 1, tmp_10293[1], tmp_10293[2]];
    signal tmp_10295[3] <== [tmp_10294[0] + challengesStage2[1][0], tmp_10294[1] + challengesStage2[1][1], tmp_10294[2] + challengesStage2[1][2]];
    signal tmp_10296[3] <== CMul()(tmp_10290, tmp_10295);
    signal tmp_10297[3] <== CMul()(evals[30], challengesStage2[0]);
    signal tmp_10298[3] <== [tmp_10297[0] + evals[77][0], tmp_10297[1] + evals[77][1], tmp_10297[2] + evals[77][2]];
    signal tmp_10299[3] <== CMul()(tmp_10298, challengesStage2[0]);
    signal tmp_10300[3] <== [tmp_10299[0] + 1, tmp_10299[1], tmp_10299[2]];
    signal tmp_10301[3] <== [tmp_10300[0] + challengesStage2[1][0], tmp_10300[1] + challengesStage2[1][1], tmp_10300[2] + challengesStage2[1][2]];
    signal tmp_10302[3] <== CMul()(tmp_10296, tmp_10301);
    signal tmp_10303[3] <== CMul()(evals[31], challengesStage2[0]);
    signal tmp_10304[3] <== [tmp_10303[0] + evals[78][0], tmp_10303[1] + evals[78][1], tmp_10303[2] + evals[78][2]];
    signal tmp_10305[3] <== CMul()(tmp_10304, challengesStage2[0]);
    signal tmp_10306[3] <== [tmp_10305[0] + 1, tmp_10305[1], tmp_10305[2]];
    signal tmp_10307[3] <== [tmp_10306[0] + challengesStage2[1][0], tmp_10306[1] + challengesStage2[1][1], tmp_10306[2] + challengesStage2[1][2]];
    signal tmp_10308[3] <== CMul()(tmp_10302, tmp_10307);
    signal tmp_10309[3] <== [tmp_10266[0] - tmp_10308[0], tmp_10266[1] - tmp_10308[1], tmp_10266[2] - tmp_10308[2]];
    signal tmp_10310[3] <== [tmp_10217[0] + tmp_10309[0], tmp_10217[1] + tmp_10309[1], tmp_10217[2] + tmp_10309[2]];
    signal tmp_10311[3] <== CMul()(challengeQ, tmp_10310);
    signal tmp_10312[3] <== [11016800570561344835 * evals[62][0], 11016800570561344835 * evals[62][1], 11016800570561344835 * evals[62][2]];
    signal tmp_10313[3] <== CMul()(tmp_10312, challengesStage2[0]);
    signal tmp_10314[3] <== [tmp_10313[0] + evals[78][0], tmp_10313[1] + evals[78][1], tmp_10313[2] + evals[78][2]];
    signal tmp_10315[3] <== CMul()(tmp_10314, challengesStage2[0]);
    signal tmp_10316[3] <== [tmp_10315[0] + 1, tmp_10315[1], tmp_10315[2]];
    signal tmp_10317[3] <== [tmp_10316[0] + challengesStage2[1][0], tmp_10316[1] + challengesStage2[1][1], tmp_10316[2] + challengesStage2[1][2]];
    signal tmp_10318[3] <== [11274872323250451096 * evals[62][0], 11274872323250451096 * evals[62][1], 11274872323250451096 * evals[62][2]];
    signal tmp_10319[3] <== CMul()(tmp_10318, challengesStage2[0]);
    signal tmp_10320[3] <== [tmp_10319[0] + evals[79][0], tmp_10319[1] + evals[79][1], tmp_10319[2] + evals[79][2]];
    signal tmp_10321[3] <== CMul()(tmp_10320, challengesStage2[0]);
    signal tmp_10322[3] <== [tmp_10321[0] + 1, tmp_10321[1], tmp_10321[2]];
    signal tmp_10323[3] <== [tmp_10322[0] + challengesStage2[1][0], tmp_10322[1] + challengesStage2[1][1], tmp_10322[2] + challengesStage2[1][2]];
    signal tmp_10324[3] <== CMul()(tmp_10317, tmp_10323);
    signal tmp_10325[3] <== [6534114114080170934 * evals[62][0], 6534114114080170934 * evals[62][1], 6534114114080170934 * evals[62][2]];
    signal tmp_10326[3] <== CMul()(tmp_10325, challengesStage2[0]);
    signal tmp_10327[3] <== [tmp_10326[0] + evals[80][0], tmp_10326[1] + evals[80][1], tmp_10326[2] + evals[80][2]];
    signal tmp_10328[3] <== CMul()(tmp_10327, challengesStage2[0]);
    signal tmp_10329[3] <== [tmp_10328[0] + 1, tmp_10328[1], tmp_10328[2]];
    signal tmp_10330[3] <== [tmp_10329[0] + challengesStage2[1][0], tmp_10329[1] + challengesStage2[1][1], tmp_10329[2] + challengesStage2[1][2]];
    signal tmp_10331[3] <== CMul()(tmp_10324, tmp_10330);
    signal tmp_10332[3] <== [13047390008333835222 * evals[62][0], 13047390008333835222 * evals[62][1], 13047390008333835222 * evals[62][2]];
    signal tmp_10333[3] <== CMul()(tmp_10332, challengesStage2[0]);
    signal tmp_10334[3] <== [tmp_10333[0] + evals[81][0], tmp_10333[1] + evals[81][1], tmp_10333[2] + evals[81][2]];
    signal tmp_10335[3] <== CMul()(tmp_10334, challengesStage2[0]);
    signal tmp_10336[3] <== [tmp_10335[0] + 1, tmp_10335[1], tmp_10335[2]];
    signal tmp_10337[3] <== [tmp_10336[0] + challengesStage2[1][0], tmp_10336[1] + challengesStage2[1][1], tmp_10336[2] + challengesStage2[1][2]];
    signal tmp_10338[3] <== CMul()(tmp_10331, tmp_10337);
    signal tmp_10339[3] <== [11189528522318044176 * evals[62][0], 11189528522318044176 * evals[62][1], 11189528522318044176 * evals[62][2]];
    signal tmp_10340[3] <== CMul()(tmp_10339, challengesStage2[0]);
    signal tmp_10341[3] <== [tmp_10340[0] + evals[82][0], tmp_10340[1] + evals[82][1], tmp_10340[2] + evals[82][2]];
    signal tmp_10342[3] <== CMul()(tmp_10341, challengesStage2[0]);
    signal tmp_10343[3] <== [tmp_10342[0] + 1, tmp_10342[1], tmp_10342[2]];
    signal tmp_10344[3] <== [tmp_10343[0] + challengesStage2[1][0], tmp_10343[1] + challengesStage2[1][1], tmp_10343[2] + challengesStage2[1][2]];
    signal tmp_10345[3] <== CMul()(tmp_10338, tmp_10344);
    signal tmp_10346[3] <== [3320735505586735876 * evals[62][0], 3320735505586735876 * evals[62][1], 3320735505586735876 * evals[62][2]];
    signal tmp_10347[3] <== CMul()(tmp_10346, challengesStage2[0]);
    signal tmp_10348[3] <== [tmp_10347[0] + evals[83][0], tmp_10347[1] + evals[83][1], tmp_10347[2] + evals[83][2]];
    signal tmp_10349[3] <== CMul()(tmp_10348, challengesStage2[0]);
    signal tmp_10350[3] <== [tmp_10349[0] + 1, tmp_10349[1], tmp_10349[2]];
    signal tmp_10351[3] <== [tmp_10350[0] + challengesStage2[1][0], tmp_10350[1] + challengesStage2[1][1], tmp_10350[2] + challengesStage2[1][2]];
    signal tmp_10352[3] <== CMul()(tmp_10345, tmp_10351);
    signal tmp_10353[3] <== [7240278926970958133 * evals[62][0], 7240278926970958133 * evals[62][1], 7240278926970958133 * evals[62][2]];
    signal tmp_10354[3] <== CMul()(tmp_10353, challengesStage2[0]);
    signal tmp_10355[3] <== [tmp_10354[0] + evals[84][0], tmp_10354[1] + evals[84][1], tmp_10354[2] + evals[84][2]];
    signal tmp_10356[3] <== CMul()(tmp_10355, challengesStage2[0]);
    signal tmp_10357[3] <== [tmp_10356[0] + 1, tmp_10356[1], tmp_10356[2]];
    signal tmp_10358[3] <== [tmp_10357[0] + challengesStage2[1][0], tmp_10357[1] + challengesStage2[1][1], tmp_10357[2] + challengesStage2[1][2]];
    signal tmp_10359[3] <== CMul()(tmp_10352, tmp_10358);
    signal tmp_10360[3] <== CMul()(evals[115], tmp_10359);
    signal tmp_10361[3] <== CMul()(evals[32], challengesStage2[0]);
    signal tmp_10362[3] <== [tmp_10361[0] + evals[79][0], tmp_10361[1] + evals[79][1], tmp_10361[2] + evals[79][2]];
    signal tmp_10363[3] <== CMul()(tmp_10362, challengesStage2[0]);
    signal tmp_10364[3] <== [tmp_10363[0] + 1, tmp_10363[1], tmp_10363[2]];
    signal tmp_10365[3] <== [tmp_10364[0] + challengesStage2[1][0], tmp_10364[1] + challengesStage2[1][1], tmp_10364[2] + challengesStage2[1][2]];
    signal tmp_10366[3] <== CMul()(evals[114], tmp_10365);
    signal tmp_10367[3] <== CMul()(evals[33], challengesStage2[0]);
    signal tmp_10368[3] <== [tmp_10367[0] + evals[80][0], tmp_10367[1] + evals[80][1], tmp_10367[2] + evals[80][2]];
    signal tmp_10369[3] <== CMul()(tmp_10368, challengesStage2[0]);
    signal tmp_10370[3] <== [tmp_10369[0] + 1, tmp_10369[1], tmp_10369[2]];
    signal tmp_10371[3] <== [tmp_10370[0] + challengesStage2[1][0], tmp_10370[1] + challengesStage2[1][1], tmp_10370[2] + challengesStage2[1][2]];
    signal tmp_10372[3] <== CMul()(tmp_10366, tmp_10371);
    signal tmp_10373[3] <== CMul()(evals[34], challengesStage2[0]);
    signal tmp_10374[3] <== [tmp_10373[0] + evals[81][0], tmp_10373[1] + evals[81][1], tmp_10373[2] + evals[81][2]];
    signal tmp_10375[3] <== CMul()(tmp_10374, challengesStage2[0]);
    signal tmp_10376[3] <== [tmp_10375[0] + 1, tmp_10375[1], tmp_10375[2]];
    signal tmp_10377[3] <== [tmp_10376[0] + challengesStage2[1][0], tmp_10376[1] + challengesStage2[1][1], tmp_10376[2] + challengesStage2[1][2]];
    signal tmp_10378[3] <== CMul()(tmp_10372, tmp_10377);
    signal tmp_10379[3] <== CMul()(evals[35], challengesStage2[0]);
    signal tmp_10380[3] <== [tmp_10379[0] + evals[82][0], tmp_10379[1] + evals[82][1], tmp_10379[2] + evals[82][2]];
    signal tmp_10381[3] <== CMul()(tmp_10380, challengesStage2[0]);
    signal tmp_10382[3] <== [tmp_10381[0] + 1, tmp_10381[1], tmp_10381[2]];
    signal tmp_10383[3] <== [tmp_10382[0] + challengesStage2[1][0], tmp_10382[1] + challengesStage2[1][1], tmp_10382[2] + challengesStage2[1][2]];
    signal tmp_10384[3] <== CMul()(tmp_10378, tmp_10383);
    signal tmp_10385[3] <== CMul()(evals[36], challengesStage2[0]);
    signal tmp_10386[3] <== [tmp_10385[0] + evals[83][0], tmp_10385[1] + evals[83][1], tmp_10385[2] + evals[83][2]];
    signal tmp_10387[3] <== CMul()(tmp_10386, challengesStage2[0]);
    signal tmp_10388[3] <== [tmp_10387[0] + 1, tmp_10387[1], tmp_10387[2]];
    signal tmp_10389[3] <== [tmp_10388[0] + challengesStage2[1][0], tmp_10388[1] + challengesStage2[1][1], tmp_10388[2] + challengesStage2[1][2]];
    signal tmp_10390[3] <== CMul()(tmp_10384, tmp_10389);
    signal tmp_10391[3] <== CMul()(evals[37], challengesStage2[0]);
    signal tmp_10392[3] <== [tmp_10391[0] + evals[84][0], tmp_10391[1] + evals[84][1], tmp_10391[2] + evals[84][2]];
    signal tmp_10393[3] <== CMul()(tmp_10392, challengesStage2[0]);
    signal tmp_10394[3] <== [tmp_10393[0] + 1, tmp_10393[1], tmp_10393[2]];
    signal tmp_10395[3] <== [tmp_10394[0] + challengesStage2[1][0], tmp_10394[1] + challengesStage2[1][1], tmp_10394[2] + challengesStage2[1][2]];
    signal tmp_10396[3] <== CMul()(tmp_10390, tmp_10395);
    signal tmp_10397[3] <== CMul()(evals[38], challengesStage2[0]);
    signal tmp_10398[3] <== [tmp_10397[0] + evals[85][0], tmp_10397[1] + evals[85][1], tmp_10397[2] + evals[85][2]];
    signal tmp_10399[3] <== CMul()(tmp_10398, challengesStage2[0]);
    signal tmp_10400[3] <== [tmp_10399[0] + 1, tmp_10399[1], tmp_10399[2]];
    signal tmp_10401[3] <== [tmp_10400[0] + challengesStage2[1][0], tmp_10400[1] + challengesStage2[1][1], tmp_10400[2] + challengesStage2[1][2]];
    signal tmp_10402[3] <== CMul()(tmp_10396, tmp_10401);
    signal tmp_10403[3] <== [tmp_10360[0] - tmp_10402[0], tmp_10360[1] - tmp_10402[1], tmp_10360[2] - tmp_10402[2]];
    signal tmp_10404[3] <== [tmp_10311[0] + tmp_10403[0], tmp_10311[1] + tmp_10403[1], tmp_10311[2] + tmp_10403[2]];
    signal tmp_10405[3] <== CMul()(challengeQ, tmp_10404);
    signal tmp_10406[3] <== [8246665031048405574 * evals[62][0], 8246665031048405574 * evals[62][1], 8246665031048405574 * evals[62][2]];
    signal tmp_10407[3] <== CMul()(tmp_10406, challengesStage2[0]);
    signal tmp_10408[3] <== [tmp_10407[0] + evals[85][0], tmp_10407[1] + evals[85][1], tmp_10407[2] + evals[85][2]];
    signal tmp_10409[3] <== CMul()(tmp_10408, challengesStage2[0]);
    signal tmp_10410[3] <== [tmp_10409[0] + 1, tmp_10409[1], tmp_10409[2]];
    signal tmp_10411[3] <== [tmp_10410[0] + challengesStage2[1][0], tmp_10410[1] + challengesStage2[1][1], tmp_10410[2] + challengesStage2[1][2]];
    signal tmp_10412[3] <== [12693612801792047873 * evals[62][0], 12693612801792047873 * evals[62][1], 12693612801792047873 * evals[62][2]];
    signal tmp_10413[3] <== CMul()(tmp_10412, challengesStage2[0]);
    signal tmp_10414[3] <== [tmp_10413[0] + evals[86][0], tmp_10413[1] + evals[86][1], tmp_10413[2] + evals[86][2]];
    signal tmp_10415[3] <== CMul()(tmp_10414, challengesStage2[0]);
    signal tmp_10416[3] <== [tmp_10415[0] + 1, tmp_10415[1], tmp_10415[2]];
    signal tmp_10417[3] <== [tmp_10416[0] + challengesStage2[1][0], tmp_10416[1] + challengesStage2[1][1], tmp_10416[2] + challengesStage2[1][2]];
    signal tmp_10418[3] <== CMul()(tmp_10411, tmp_10417);
    signal tmp_10419[3] <== [9404062091095256088 * evals[62][0], 9404062091095256088 * evals[62][1], 9404062091095256088 * evals[62][2]];
    signal tmp_10420[3] <== CMul()(tmp_10419, challengesStage2[0]);
    signal tmp_10421[3] <== [tmp_10420[0] + evals[87][0], tmp_10420[1] + evals[87][1], tmp_10420[2] + evals[87][2]];
    signal tmp_10422[3] <== CMul()(tmp_10421, challengesStage2[0]);
    signal tmp_10423[3] <== [tmp_10422[0] + 1, tmp_10422[1], tmp_10422[2]];
    signal tmp_10424[3] <== [tmp_10423[0] + challengesStage2[1][0], tmp_10423[1] + challengesStage2[1][1], tmp_10423[2] + challengesStage2[1][2]];
    signal tmp_10425[3] <== CMul()(tmp_10418, tmp_10424);
    signal tmp_10426[3] <== CMul()(evals[112], tmp_10425);
    signal tmp_10427[3] <== [1 - evals[63][0], -evals[63][1], -evals[63][2]];
    signal tmp_10428[3] <== CMul()(evals[16], tmp_10427);
    signal tmp_10429[3] <== [tmp_10428[0] + evals[63][0], tmp_10428[1] + evals[63][1], tmp_10428[2] + evals[63][2]];
    signal tmp_10430[3] <== CMul()(evals[39], challengesStage2[0]);
    signal tmp_10431[3] <== [tmp_10430[0] + evals[86][0], tmp_10430[1] + evals[86][1], tmp_10430[2] + evals[86][2]];
    signal tmp_10432[3] <== CMul()(tmp_10431, challengesStage2[0]);
    signal tmp_10433[3] <== [tmp_10432[0] + 1, tmp_10432[1], tmp_10432[2]];
    signal tmp_10434[3] <== [tmp_10433[0] + challengesStage2[1][0], tmp_10433[1] + challengesStage2[1][1], tmp_10433[2] + challengesStage2[1][2]];
    signal tmp_10435[3] <== CMul()(evals[115], tmp_10434);
    signal tmp_10436[3] <== CMul()(evals[40], challengesStage2[0]);
    signal tmp_10437[3] <== [tmp_10436[0] + evals[87][0], tmp_10436[1] + evals[87][1], tmp_10436[2] + evals[87][2]];
    signal tmp_10438[3] <== CMul()(tmp_10437, challengesStage2[0]);
    signal tmp_10439[3] <== [tmp_10438[0] + 1, tmp_10438[1], tmp_10438[2]];
    signal tmp_10440[3] <== [tmp_10439[0] + challengesStage2[1][0], tmp_10439[1] + challengesStage2[1][1], tmp_10439[2] + challengesStage2[1][2]];
    signal tmp_10441[3] <== CMul()(tmp_10435, tmp_10440);
    signal tmp_10442[3] <== CMul()(tmp_10429, tmp_10441);
    signal tmp_10443[3] <== [tmp_10426[0] - tmp_10442[0], tmp_10426[1] - tmp_10442[1], tmp_10426[2] - tmp_10442[2]];
    signal tmp_10444[3] <== [tmp_10405[0] + tmp_10443[0], tmp_10405[1] + tmp_10443[1], tmp_10405[2] + tmp_10443[2]];
    signal tmp_10445[3] <== CMul()(challengeQ, tmp_10444);
    signal tmp_10446[3] <== [1 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_10447[3] <== CMul()(evals[125], tmp_10446);
    signal tmp_5223[3] <== [tmp_10445[0] + tmp_10447[0], tmp_10445[1] + tmp_10447[1], tmp_10445[2] + tmp_10447[2]];
    tmp_10448 <== CMul()(tmp_5223, Zh);
}


/*
    The claimed evaluations satisfy the air: the constraint combination at z, computed by
    VerifyEvaluationsChunks*, must equal Q(z).

    Q, of degree < 7·2^17, is committed as 7 polynomials of degree < 2^17, so
    Q(z) = Σ_i z^(i·2^17)·Q_i(z) is reassembled here.
*/
template parallel VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[144][3];
    signal input publics[399];
    signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeOut) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[17][3];
    for (var i = 0; i < 17; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeOut, challengeOut);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[16][0] - 1, zMul[16][1], zMul[16][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_6223[3];
    (tmp_6223) <== VerifyEvaluationsChunks0_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh);
    signal tmp_6837[3];
    signal tmp_6849[3];
    signal tmp_6867[3];
    signal tmp_6873[3];
    signal tmp_6936[3];
    signal tmp_7303[3];
    signal tmp_7307[3];
    signal tmp_7322[3];
    signal tmp_7323[3];
    signal tmp_7332[3];
    signal tmp_7334[3];
    signal tmp_7335[3];
    (tmp_6837,tmp_6849,tmp_6867,tmp_6873,tmp_6936,tmp_7303,tmp_7307,tmp_7322,tmp_7323,tmp_7332,tmp_7334,tmp_7335) <== VerifyEvaluationsChunks1_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_6223);
    signal tmp_7869[3];
    signal tmp_7927[3];
    signal tmp_7951[3];
    signal tmp_7969[3];
    signal tmp_7986[3];
    signal tmp_8009[3];
    signal tmp_8033[3];
    signal tmp_8051[3];
    signal tmp_8068[3];
    signal tmp_8091[3];
    signal tmp_8558[3];
    signal tmp_8563[3];
    signal tmp_8566[3];
    signal tmp_8569[3];
    signal tmp_8572[3];
    signal tmp_8575[3];
    signal tmp_8578[3];
    signal tmp_8581[3];
    signal tmp_8584[3];
    signal tmp_8587[3];
    signal tmp_8590[3];
    signal tmp_8593[3];
    signal tmp_8596[3];
    signal tmp_8599[3];
    signal tmp_8602[3];
    signal tmp_8605[3];
    signal tmp_8606[3];
    signal tmp_8608[3];
    (tmp_7869,tmp_7927,tmp_7951,tmp_7969,tmp_7986,tmp_8009,tmp_8033,tmp_8051,tmp_8068,tmp_8091,tmp_8558,tmp_8563,tmp_8566,tmp_8569,tmp_8572,tmp_8575,tmp_8578,tmp_8581,tmp_8584,tmp_8587,tmp_8590,tmp_8593,tmp_8596,tmp_8599,tmp_8602,tmp_8605,tmp_8606,tmp_8608) <== VerifyEvaluationsChunks2_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_6936,tmp_7303,tmp_7307,tmp_7322,tmp_7323,tmp_7332,tmp_7334,tmp_7335);
    signal tmp_9657[3];
    (tmp_9657) <== VerifyEvaluationsChunks3_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_7869,tmp_7927,tmp_7951,tmp_7969,tmp_7986,tmp_8009,tmp_8033,tmp_8051,tmp_8068,tmp_8091,tmp_8558,tmp_8563,tmp_8566,tmp_8569,tmp_8572,tmp_8575,tmp_8578,tmp_8581,tmp_8584,tmp_8587,tmp_8590,tmp_8593,tmp_8596,tmp_8599,tmp_8602,tmp_8605,tmp_8606,tmp_8608);
    signal tmp_10448[3];
    (tmp_10448) <== VerifyEvaluationsChunks4_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_6837,tmp_6849,tmp_6867,tmp_6873,tmp_9657);

    signal xAcc[7][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[6][3]; // Stores the evaluations of Q_i
    signal qAcc[7][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i = 0; i < 7; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[116+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[16]);
            qStep[i-1] <== CMul()(xAcc[i], evals[116+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i
    enable * (tmp_10448[0] - qAcc[6][0]) === 0;
    enable * (tmp_10448[1] - qAcc[6][1]) === 0;
    enable * (tmp_10448[2] - qAcc[6][2]) === 0;
}

/*
    f_0(x) at one query point, out of that query's opened leaves: the arithmetic of the DEEP
    combination whose formula and cross-check live in VerifyQuery0 below.

    The leaves arrive as the flat sections the Merkle trees store (cm1..cm3, consts), and `mapValues` splits them into one signal per committed polynomial: dim 1 for a
    base-field polynomial, dim 3 for an extension one — which is why a dim-1 value is
    subtracted from limb 0 of its `evals` entry alone. Fixed polynomials are always dim 1 and
    are read straight out of `consts`.

    The emitted code is a nested Horner. Within one opening point g^s·z the differences
    (pol(x) - pol(g^s·z)) are folded with challengesDeep[1] = vf2, the result is multiplied by
    xDivXSubXi[s] = 1/(x - g^s·z), and the opening points are folded together with
    challengesDeep[0] = vf1; tmp_807 is f_0(x).

    The DAG is split into 1 chunk(s) by the same compile-cost device as
    VerifyEvaluationsChunks*, with tmp_ signals crossing the boundaries; each chunk builds its
    own `mapValues`, since the leaf values feed the whole DAG.
*/
template CalculateDeepPolChunks0_0() {
    signal input challengesDeep[2][3];
    signal input evals[144][3];

    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[47];

    signal input xDivXSubXi[19][3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
    mapValues.vals1 <== cm1;
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;

    signal input tmp_664[3];
    signal input tmp_665[3];
    signal input tmp_666[3];
    signal input tmp_667[3];
    signal input tmp_668[3];
    signal input tmp_669[3];
    signal input tmp_670[3];
    signal input tmp_671[3];
    signal input tmp_672[3];
    signal input tmp_673[3];
    signal input tmp_674[3];
    signal input tmp_675[3];
    signal input tmp_676[3];
    signal input tmp_677[3];
    signal input tmp_678[3];
    signal input tmp_679[3];
    signal input tmp_680[3];
    signal input tmp_698[3];
    signal input tmp_699[3];
    signal input tmp_700[3];
    signal input tmp_701[3];
    signal input tmp_702[3];
    signal input tmp_703[3];
    signal input tmp_704[3];
    signal input tmp_705[3];
    signal input tmp_706[3];
    signal input tmp_707[3];
    signal input tmp_708[3];
    signal input tmp_709[3];
    signal input tmp_710[3];
    signal input tmp_711[3];
    signal input tmp_712[3];
    signal input tmp_713[3];
    signal input tmp_714[3];
    signal input tmp_715[3];
    signal input tmp_716[3];
    signal input tmp_717[3];
    signal input tmp_718[3];
    signal input tmp_719[3];
    signal input tmp_720[3];
    signal input tmp_721[3];
    signal input tmp_722[3];
    signal input tmp_723[3];
    signal input tmp_724[3];
    signal input tmp_725[3];
    signal input tmp_726[3];
    signal input tmp_727[3];
    signal input tmp_728[3];
    signal input tmp_729[3];
    signal input tmp_730[3];
    signal input tmp_731[3];
    signal input tmp_732[3];
    signal input tmp_733[3];
    signal input tmp_734[3];
    signal input tmp_735[3];
    signal input tmp_736[3];
    signal input tmp_737[3];
    signal input tmp_738[3];
    signal input tmp_739[3];
    signal input tmp_740[3];
    signal input tmp_741[3];
    signal input tmp_742[3];
    signal input tmp_743[3];
    signal input tmp_744[3];
    signal input tmp_745[3];
    signal input tmp_746[3];
    signal input tmp_747[3];
    signal input tmp_748[3];
    signal input tmp_749[3];
    signal input tmp_750[3];
    signal input tmp_751[3];
    signal input tmp_752[3];
    signal input tmp_753[3];
    signal input tmp_754[3];
    signal input tmp_755[3];
    signal input tmp_756[3];
    signal input tmp_757[3];
    signal input tmp_758[3];
    signal input tmp_759[3];
    signal input tmp_760[3];
    signal input tmp_761[3];
    signal input tmp_762[3];
    signal input tmp_763[3];
    signal input tmp_764[3];
    signal input tmp_765[3];
    signal input tmp_766[3];
    signal input tmp_767[3];
    signal input tmp_768[3];
    signal input tmp_769[3];
    signal input tmp_770[3];
    signal input tmp_771[3];
    signal input tmp_772[3];
    signal input tmp_773[3];
    signal input tmp_774[3];
    signal input tmp_775[3];
    signal input tmp_776[3];
    signal input tmp_777[3];
    signal input tmp_778[3];
    signal input tmp_779[3];
    signal input tmp_780[3];
    signal input tmp_781[3];
    signal input tmp_782[3];
    signal input tmp_783[3];
    signal input tmp_784[3];
    signal input tmp_785[3];
    signal input tmp_786[3];
    signal input tmp_787[3];
    signal input tmp_788[3];
    signal input tmp_789[3];
    signal input tmp_790[3];
    signal input tmp_791[3];
    signal input tmp_792[3];
    signal input tmp_793[3];
    signal input tmp_794[3];
    signal input tmp_795[3];
    signal input tmp_796[3];
    signal input tmp_797[3];
    signal input tmp_798[3];
    signal input tmp_799[3];
    signal input tmp_800[3];
    signal input tmp_801[3];
    signal input tmp_803[3];
    signal input tmp_804[3];
    signal input tmp_805[3];
    signal input tmp_806[3];

    signal output tmp_807[3];
    signal tmp_681[3] <== CMul()(tmp_680, xDivXSubXi[0]);
    signal tmp_682[3] <== CMul()(tmp_679, xDivXSubXi[1]);
    signal tmp_683[3] <== CMul()(tmp_678, xDivXSubXi[2]);
    signal tmp_684[3] <== CMul()(tmp_677, xDivXSubXi[3]);
    signal tmp_685[3] <== CMul()(tmp_676, xDivXSubXi[4]);
    signal tmp_686[3] <== CMul()(tmp_675, xDivXSubXi[5]);
    signal tmp_687[3] <== CMul()(tmp_674, xDivXSubXi[6]);
    signal tmp_688[3] <== CMul()(tmp_673, xDivXSubXi[7]);
    signal tmp_689[3] <== CMul()(tmp_672, xDivXSubXi[8]);
    signal tmp_690[3] <== CMul()(tmp_671, xDivXSubXi[9]);
    signal tmp_691[3] <== CMul()(tmp_670, xDivXSubXi[10]);
    signal tmp_692[3] <== CMul()(tmp_669, xDivXSubXi[11]);
    signal tmp_693[3] <== CMul()(tmp_668, xDivXSubXi[12]);
    signal tmp_694[3] <== CMul()(tmp_667, xDivXSubXi[13]);
    signal tmp_695[3] <== CMul()(tmp_666, xDivXSubXi[14]);
    signal tmp_696[3] <== CMul()(tmp_665, xDivXSubXi[15]);
    signal tmp_697[3] <== CMul()(tmp_664, xDivXSubXi[16]);
    signal tmp_802[3] <== CMul()(challengesDeep[0], xDivXSubXi[17]);
    signal tmp_391[3] <== [tmp_681[0] + tmp_682[0], tmp_681[1] + tmp_682[1], tmp_681[2] + tmp_682[2]];
    signal tmp_392[3] <== [tmp_391[0] + tmp_683[0], tmp_391[1] + tmp_683[1], tmp_391[2] + tmp_683[2]];
    signal tmp_393[3] <== [tmp_392[0] + tmp_684[0], tmp_392[1] + tmp_684[1], tmp_392[2] + tmp_684[2]];
    signal tmp_394[3] <== [tmp_393[0] + tmp_685[0], tmp_393[1] + tmp_685[1], tmp_393[2] + tmp_685[2]];
    signal tmp_395[3] <== [tmp_394[0] + tmp_686[0], tmp_394[1] + tmp_686[1], tmp_394[2] + tmp_686[2]];
    signal tmp_396[3] <== [tmp_395[0] + tmp_687[0], tmp_395[1] + tmp_687[1], tmp_395[2] + tmp_687[2]];
    signal tmp_397[3] <== [tmp_396[0] + tmp_688[0], tmp_396[1] + tmp_688[1], tmp_396[2] + tmp_688[2]];
    signal tmp_398[3] <== [tmp_397[0] + tmp_689[0], tmp_397[1] + tmp_689[1], tmp_397[2] + tmp_689[2]];
    signal tmp_399[3] <== [tmp_398[0] + tmp_690[0], tmp_398[1] + tmp_690[1], tmp_398[2] + tmp_690[2]];
    signal tmp_400[3] <== [tmp_399[0] + tmp_691[0], tmp_399[1] + tmp_691[1], tmp_399[2] + tmp_691[2]];
    signal tmp_401[3] <== [tmp_400[0] + tmp_692[0], tmp_400[1] + tmp_692[1], tmp_400[2] + tmp_692[2]];
    signal tmp_402[3] <== [tmp_401[0] + tmp_693[0], tmp_401[1] + tmp_693[1], tmp_401[2] + tmp_693[2]];
    signal tmp_403[3] <== [tmp_402[0] + tmp_694[0], tmp_402[1] + tmp_694[1], tmp_402[2] + tmp_694[2]];
    signal tmp_404[3] <== [tmp_403[0] + tmp_695[0], tmp_403[1] + tmp_695[1], tmp_403[2] + tmp_695[2]];
    signal tmp_405[3] <== [tmp_404[0] + tmp_696[0], tmp_404[1] + tmp_696[1], tmp_404[2] + tmp_696[2]];
    signal tmp_406[3] <== [tmp_405[0] + tmp_697[0], tmp_405[1] + tmp_697[1], tmp_405[2] + tmp_697[2]];
    signal tmp_407[3] <== [consts[44] * tmp_406[0], consts[44] * tmp_406[1], consts[44] * tmp_406[2]];
    signal tmp_408[3] <== CMul()(challengesDeep[1], mapValues.cm2_0);
    signal tmp_409[3] <== [tmp_696[0] + tmp_697[0], tmp_696[1] + tmp_697[1], tmp_696[2] + tmp_697[2]];
    signal tmp_410[3] <== CMul()(tmp_408, tmp_409);
    signal tmp_411[3] <== [tmp_407[0] + tmp_410[0], tmp_407[1] + tmp_410[1], tmp_407[2] + tmp_410[2]];
    signal tmp_412[3] <== [tmp_698[0] * consts[0], tmp_698[1] * consts[0], tmp_698[2] * consts[0]];
    signal tmp_413[3] <== [tmp_699[0] * consts[1], tmp_699[1] * consts[1], tmp_699[2] * consts[1]];
    signal tmp_414[3] <== [tmp_412[0] + tmp_413[0], tmp_412[1] + tmp_413[1], tmp_412[2] + tmp_413[2]];
    signal tmp_415[3] <== [tmp_700[0] * consts[2], tmp_700[1] * consts[2], tmp_700[2] * consts[2]];
    signal tmp_416[3] <== [tmp_414[0] + tmp_415[0], tmp_414[1] + tmp_415[1], tmp_414[2] + tmp_415[2]];
    signal tmp_417[3] <== [tmp_701[0] * consts[3], tmp_701[1] * consts[3], tmp_701[2] * consts[3]];
    signal tmp_418[3] <== [tmp_416[0] + tmp_417[0], tmp_416[1] + tmp_417[1], tmp_416[2] + tmp_417[2]];
    signal tmp_419[3] <== [tmp_702[0] * consts[4], tmp_702[1] * consts[4], tmp_702[2] * consts[4]];
    signal tmp_420[3] <== [tmp_418[0] + tmp_419[0], tmp_418[1] + tmp_419[1], tmp_418[2] + tmp_419[2]];
    signal tmp_421[3] <== [tmp_703[0] * consts[5], tmp_703[1] * consts[5], tmp_703[2] * consts[5]];
    signal tmp_422[3] <== [tmp_420[0] + tmp_421[0], tmp_420[1] + tmp_421[1], tmp_420[2] + tmp_421[2]];
    signal tmp_423[3] <== [tmp_704[0] * consts[6], tmp_704[1] * consts[6], tmp_704[2] * consts[6]];
    signal tmp_424[3] <== [tmp_422[0] + tmp_423[0], tmp_422[1] + tmp_423[1], tmp_422[2] + tmp_423[2]];
    signal tmp_425[3] <== [tmp_705[0] * consts[7], tmp_705[1] * consts[7], tmp_705[2] * consts[7]];
    signal tmp_426[3] <== [tmp_424[0] + tmp_425[0], tmp_424[1] + tmp_425[1], tmp_424[2] + tmp_425[2]];
    signal tmp_427[3] <== [tmp_706[0] * consts[8], tmp_706[1] * consts[8], tmp_706[2] * consts[8]];
    signal tmp_428[3] <== [tmp_426[0] + tmp_427[0], tmp_426[1] + tmp_427[1], tmp_426[2] + tmp_427[2]];
    signal tmp_429[3] <== [tmp_707[0] * consts[9], tmp_707[1] * consts[9], tmp_707[2] * consts[9]];
    signal tmp_430[3] <== [tmp_428[0] + tmp_429[0], tmp_428[1] + tmp_429[1], tmp_428[2] + tmp_429[2]];
    signal tmp_431[3] <== [tmp_708[0] * consts[10], tmp_708[1] * consts[10], tmp_708[2] * consts[10]];
    signal tmp_432[3] <== [tmp_430[0] + tmp_431[0], tmp_430[1] + tmp_431[1], tmp_430[2] + tmp_431[2]];
    signal tmp_433[3] <== [tmp_709[0] * consts[11], tmp_709[1] * consts[11], tmp_709[2] * consts[11]];
    signal tmp_434[3] <== [tmp_432[0] + tmp_433[0], tmp_432[1] + tmp_433[1], tmp_432[2] + tmp_433[2]];
    signal tmp_435[3] <== [tmp_710[0] * consts[12], tmp_710[1] * consts[12], tmp_710[2] * consts[12]];
    signal tmp_436[3] <== [tmp_434[0] + tmp_435[0], tmp_434[1] + tmp_435[1], tmp_434[2] + tmp_435[2]];
    signal tmp_437[3] <== [tmp_711[0] * consts[13], tmp_711[1] * consts[13], tmp_711[2] * consts[13]];
    signal tmp_438[3] <== [tmp_436[0] + tmp_437[0], tmp_436[1] + tmp_437[1], tmp_436[2] + tmp_437[2]];
    signal tmp_439[3] <== [tmp_712[0] * consts[14], tmp_712[1] * consts[14], tmp_712[2] * consts[14]];
    signal tmp_440[3] <== [tmp_438[0] + tmp_439[0], tmp_438[1] + tmp_439[1], tmp_438[2] + tmp_439[2]];
    signal tmp_441[3] <== [tmp_713[0] * consts[15], tmp_713[1] * consts[15], tmp_713[2] * consts[15]];
    signal tmp_442[3] <== [tmp_440[0] + tmp_441[0], tmp_440[1] + tmp_441[1], tmp_440[2] + tmp_441[2]];
    signal tmp_443[3] <== [tmp_714[0] * consts[16], tmp_714[1] * consts[16], tmp_714[2] * consts[16]];
    signal tmp_444[3] <== [tmp_442[0] + tmp_443[0], tmp_442[1] + tmp_443[1], tmp_442[2] + tmp_443[2]];
    signal tmp_445[3] <== [tmp_715[0] * consts[17], tmp_715[1] * consts[17], tmp_715[2] * consts[17]];
    signal tmp_446[3] <== [tmp_444[0] + tmp_445[0], tmp_444[1] + tmp_445[1], tmp_444[2] + tmp_445[2]];
    signal tmp_447[3] <== [tmp_716[0] * consts[18], tmp_716[1] * consts[18], tmp_716[2] * consts[18]];
    signal tmp_448[3] <== [tmp_446[0] + tmp_447[0], tmp_446[1] + tmp_447[1], tmp_446[2] + tmp_447[2]];
    signal tmp_449[3] <== [tmp_717[0] * consts[19], tmp_717[1] * consts[19], tmp_717[2] * consts[19]];
    signal tmp_450[3] <== [tmp_448[0] + tmp_449[0], tmp_448[1] + tmp_449[1], tmp_448[2] + tmp_449[2]];
    signal tmp_451[3] <== [tmp_718[0] * consts[20], tmp_718[1] * consts[20], tmp_718[2] * consts[20]];
    signal tmp_452[3] <== [tmp_450[0] + tmp_451[0], tmp_450[1] + tmp_451[1], tmp_450[2] + tmp_451[2]];
    signal tmp_453[3] <== [tmp_719[0] * consts[21], tmp_719[1] * consts[21], tmp_719[2] * consts[21]];
    signal tmp_454[3] <== [tmp_452[0] + tmp_453[0], tmp_452[1] + tmp_453[1], tmp_452[2] + tmp_453[2]];
    signal tmp_455[3] <== [tmp_720[0] * consts[22], tmp_720[1] * consts[22], tmp_720[2] * consts[22]];
    signal tmp_456[3] <== [tmp_454[0] + tmp_455[0], tmp_454[1] + tmp_455[1], tmp_454[2] + tmp_455[2]];
    signal tmp_457[3] <== [tmp_721[0] * consts[23], tmp_721[1] * consts[23], tmp_721[2] * consts[23]];
    signal tmp_458[3] <== [tmp_456[0] + tmp_457[0], tmp_456[1] + tmp_457[1], tmp_456[2] + tmp_457[2]];
    signal tmp_459[3] <== [tmp_722[0] * consts[24], tmp_722[1] * consts[24], tmp_722[2] * consts[24]];
    signal tmp_460[3] <== [tmp_458[0] + tmp_459[0], tmp_458[1] + tmp_459[1], tmp_458[2] + tmp_459[2]];
    signal tmp_461[3] <== [tmp_723[0] * consts[25], tmp_723[1] * consts[25], tmp_723[2] * consts[25]];
    signal tmp_462[3] <== [tmp_460[0] + tmp_461[0], tmp_460[1] + tmp_461[1], tmp_460[2] + tmp_461[2]];
    signal tmp_463[3] <== [tmp_724[0] * consts[26], tmp_724[1] * consts[26], tmp_724[2] * consts[26]];
    signal tmp_464[3] <== [tmp_462[0] + tmp_463[0], tmp_462[1] + tmp_463[1], tmp_462[2] + tmp_463[2]];
    signal tmp_465[3] <== [tmp_725[0] * consts[27], tmp_725[1] * consts[27], tmp_725[2] * consts[27]];
    signal tmp_466[3] <== [tmp_464[0] + tmp_465[0], tmp_464[1] + tmp_465[1], tmp_464[2] + tmp_465[2]];
    signal tmp_467[3] <== [tmp_726[0] * consts[28], tmp_726[1] * consts[28], tmp_726[2] * consts[28]];
    signal tmp_468[3] <== [tmp_466[0] + tmp_467[0], tmp_466[1] + tmp_467[1], tmp_466[2] + tmp_467[2]];
    signal tmp_469[3] <== [tmp_727[0] * consts[29], tmp_727[1] * consts[29], tmp_727[2] * consts[29]];
    signal tmp_470[3] <== [tmp_468[0] + tmp_469[0], tmp_468[1] + tmp_469[1], tmp_468[2] + tmp_469[2]];
    signal tmp_471[3] <== [tmp_728[0] * consts[30], tmp_728[1] * consts[30], tmp_728[2] * consts[30]];
    signal tmp_472[3] <== [tmp_470[0] + tmp_471[0], tmp_470[1] + tmp_471[1], tmp_470[2] + tmp_471[2]];
    signal tmp_473[3] <== [tmp_729[0] * consts[31], tmp_729[1] * consts[31], tmp_729[2] * consts[31]];
    signal tmp_474[3] <== [tmp_472[0] + tmp_473[0], tmp_472[1] + tmp_473[1], tmp_472[2] + tmp_473[2]];
    signal tmp_475[3] <== [tmp_730[0] * consts[32], tmp_730[1] * consts[32], tmp_730[2] * consts[32]];
    signal tmp_476[3] <== [tmp_474[0] + tmp_475[0], tmp_474[1] + tmp_475[1], tmp_474[2] + tmp_475[2]];
    signal tmp_477[3] <== [tmp_731[0] * consts[33], tmp_731[1] * consts[33], tmp_731[2] * consts[33]];
    signal tmp_478[3] <== [tmp_476[0] + tmp_477[0], tmp_476[1] + tmp_477[1], tmp_476[2] + tmp_477[2]];
    signal tmp_479[3] <== [tmp_732[0] * consts[34], tmp_732[1] * consts[34], tmp_732[2] * consts[34]];
    signal tmp_480[3] <== [tmp_478[0] + tmp_479[0], tmp_478[1] + tmp_479[1], tmp_478[2] + tmp_479[2]];
    signal tmp_481[3] <== [tmp_733[0] * consts[35], tmp_733[1] * consts[35], tmp_733[2] * consts[35]];
    signal tmp_482[3] <== [tmp_480[0] + tmp_481[0], tmp_480[1] + tmp_481[1], tmp_480[2] + tmp_481[2]];
    signal tmp_483[3] <== [tmp_736[0] * consts[38], tmp_736[1] * consts[38], tmp_736[2] * consts[38]];
    signal tmp_484[3] <== [tmp_482[0] + tmp_483[0], tmp_482[1] + tmp_483[1], tmp_482[2] + tmp_483[2]];
    signal tmp_485[3] <== [tmp_737[0] * consts[39], tmp_737[1] * consts[39], tmp_737[2] * consts[39]];
    signal tmp_486[3] <== [tmp_484[0] + tmp_485[0], tmp_484[1] + tmp_485[1], tmp_484[2] + tmp_485[2]];
    signal tmp_487[3] <== [tmp_738[0] * consts[40], tmp_738[1] * consts[40], tmp_738[2] * consts[40]];
    signal tmp_488[3] <== [tmp_486[0] + tmp_487[0], tmp_486[1] + tmp_487[1], tmp_486[2] + tmp_487[2]];
    signal tmp_489[3] <== [tmp_739[0] * consts[41], tmp_739[1] * consts[41], tmp_739[2] * consts[41]];
    signal tmp_490[3] <== [tmp_488[0] + tmp_489[0], tmp_488[1] + tmp_489[1], tmp_488[2] + tmp_489[2]];
    signal tmp_491[3] <== [tmp_740[0] * consts[42], tmp_740[1] * consts[42], tmp_740[2] * consts[42]];
    signal tmp_492[3] <== [tmp_490[0] + tmp_491[0], tmp_490[1] + tmp_491[1], tmp_490[2] + tmp_491[2]];
    signal tmp_493[3] <== [tmp_741[0] * consts[43], tmp_741[1] * consts[43], tmp_741[2] * consts[43]];
    signal tmp_494[3] <== [tmp_492[0] + tmp_493[0], tmp_492[1] + tmp_493[1], tmp_492[2] + tmp_493[2]];
    signal tmp_495[3] <== [tmp_742[0] * consts[45], tmp_742[1] * consts[45], tmp_742[2] * consts[45]];
    signal tmp_496[3] <== [tmp_494[0] + tmp_495[0], tmp_494[1] + tmp_495[1], tmp_494[2] + tmp_495[2]];
    signal tmp_497[3] <== [tmp_744[0] * mapValues.cm1_0, tmp_744[1] * mapValues.cm1_0, tmp_744[2] * mapValues.cm1_0];
    signal tmp_498[3] <== [tmp_496[0] + tmp_497[0], tmp_496[1] + tmp_497[1], tmp_496[2] + tmp_497[2]];
    signal tmp_499[3] <== [tmp_745[0] * mapValues.cm1_1, tmp_745[1] * mapValues.cm1_1, tmp_745[2] * mapValues.cm1_1];
    signal tmp_500[3] <== [tmp_498[0] + tmp_499[0], tmp_498[1] + tmp_499[1], tmp_498[2] + tmp_499[2]];
    signal tmp_501[3] <== [tmp_746[0] * mapValues.cm1_2, tmp_746[1] * mapValues.cm1_2, tmp_746[2] * mapValues.cm1_2];
    signal tmp_502[3] <== [tmp_500[0] + tmp_501[0], tmp_500[1] + tmp_501[1], tmp_500[2] + tmp_501[2]];
    signal tmp_503[3] <== [tmp_747[0] * mapValues.cm1_3, tmp_747[1] * mapValues.cm1_3, tmp_747[2] * mapValues.cm1_3];
    signal tmp_504[3] <== [tmp_502[0] + tmp_503[0], tmp_502[1] + tmp_503[1], tmp_502[2] + tmp_503[2]];
    signal tmp_505[3] <== [tmp_748[0] * mapValues.cm1_4, tmp_748[1] * mapValues.cm1_4, tmp_748[2] * mapValues.cm1_4];
    signal tmp_506[3] <== [tmp_504[0] + tmp_505[0], tmp_504[1] + tmp_505[1], tmp_504[2] + tmp_505[2]];
    signal tmp_507[3] <== [tmp_749[0] * mapValues.cm1_5, tmp_749[1] * mapValues.cm1_5, tmp_749[2] * mapValues.cm1_5];
    signal tmp_508[3] <== [tmp_506[0] + tmp_507[0], tmp_506[1] + tmp_507[1], tmp_506[2] + tmp_507[2]];
    signal tmp_509[3] <== [tmp_750[0] * mapValues.cm1_6, tmp_750[1] * mapValues.cm1_6, tmp_750[2] * mapValues.cm1_6];
    signal tmp_510[3] <== [tmp_508[0] + tmp_509[0], tmp_508[1] + tmp_509[1], tmp_508[2] + tmp_509[2]];
    signal tmp_511[3] <== [tmp_751[0] * mapValues.cm1_7, tmp_751[1] * mapValues.cm1_7, tmp_751[2] * mapValues.cm1_7];
    signal tmp_512[3] <== [tmp_510[0] + tmp_511[0], tmp_510[1] + tmp_511[1], tmp_510[2] + tmp_511[2]];
    signal tmp_513[3] <== [tmp_752[0] * mapValues.cm1_8, tmp_752[1] * mapValues.cm1_8, tmp_752[2] * mapValues.cm1_8];
    signal tmp_514[3] <== [tmp_512[0] + tmp_513[0], tmp_512[1] + tmp_513[1], tmp_512[2] + tmp_513[2]];
    signal tmp_515[3] <== [tmp_753[0] * mapValues.cm1_9, tmp_753[1] * mapValues.cm1_9, tmp_753[2] * mapValues.cm1_9];
    signal tmp_516[3] <== [tmp_514[0] + tmp_515[0], tmp_514[1] + tmp_515[1], tmp_514[2] + tmp_515[2]];
    signal tmp_517[3] <== [tmp_754[0] * mapValues.cm1_10, tmp_754[1] * mapValues.cm1_10, tmp_754[2] * mapValues.cm1_10];
    signal tmp_518[3] <== [tmp_516[0] + tmp_517[0], tmp_516[1] + tmp_517[1], tmp_516[2] + tmp_517[2]];
    signal tmp_519[3] <== [tmp_755[0] * mapValues.cm1_11, tmp_755[1] * mapValues.cm1_11, tmp_755[2] * mapValues.cm1_11];
    signal tmp_520[3] <== [tmp_518[0] + tmp_519[0], tmp_518[1] + tmp_519[1], tmp_518[2] + tmp_519[2]];
    signal tmp_521[3] <== [tmp_756[0] * mapValues.cm1_12, tmp_756[1] * mapValues.cm1_12, tmp_756[2] * mapValues.cm1_12];
    signal tmp_522[3] <== [tmp_520[0] + tmp_521[0], tmp_520[1] + tmp_521[1], tmp_520[2] + tmp_521[2]];
    signal tmp_523[3] <== [tmp_757[0] * mapValues.cm1_13, tmp_757[1] * mapValues.cm1_13, tmp_757[2] * mapValues.cm1_13];
    signal tmp_524[3] <== [tmp_522[0] + tmp_523[0], tmp_522[1] + tmp_523[1], tmp_522[2] + tmp_523[2]];
    signal tmp_525[3] <== [tmp_758[0] * mapValues.cm1_14, tmp_758[1] * mapValues.cm1_14, tmp_758[2] * mapValues.cm1_14];
    signal tmp_526[3] <== [tmp_524[0] + tmp_525[0], tmp_524[1] + tmp_525[1], tmp_524[2] + tmp_525[2]];
    signal tmp_527[3] <== [tmp_776[0] * mapValues.cm1_32, tmp_776[1] * mapValues.cm1_32, tmp_776[2] * mapValues.cm1_32];
    signal tmp_528[3] <== [tmp_526[0] + tmp_527[0], tmp_526[1] + tmp_527[1], tmp_526[2] + tmp_527[2]];
    signal tmp_529[3] <== [tmp_777[0] * mapValues.cm1_33, tmp_777[1] * mapValues.cm1_33, tmp_777[2] * mapValues.cm1_33];
    signal tmp_530[3] <== [tmp_528[0] + tmp_529[0], tmp_528[1] + tmp_529[1], tmp_528[2] + tmp_529[2]];
    signal tmp_531[3] <== [tmp_778[0] * mapValues.cm1_34, tmp_778[1] * mapValues.cm1_34, tmp_778[2] * mapValues.cm1_34];
    signal tmp_532[3] <== [tmp_530[0] + tmp_531[0], tmp_530[1] + tmp_531[1], tmp_530[2] + tmp_531[2]];
    signal tmp_533[3] <== [tmp_779[0] * mapValues.cm1_35, tmp_779[1] * mapValues.cm1_35, tmp_779[2] * mapValues.cm1_35];
    signal tmp_534[3] <== [tmp_532[0] + tmp_533[0], tmp_532[1] + tmp_533[1], tmp_532[2] + tmp_533[2]];
    signal tmp_535[3] <== [tmp_780[0] * mapValues.cm1_36, tmp_780[1] * mapValues.cm1_36, tmp_780[2] * mapValues.cm1_36];
    signal tmp_536[3] <== [tmp_534[0] + tmp_535[0], tmp_534[1] + tmp_535[1], tmp_534[2] + tmp_535[2]];
    signal tmp_537[3] <== [tmp_781[0] * mapValues.cm1_37, tmp_781[1] * mapValues.cm1_37, tmp_781[2] * mapValues.cm1_37];
    signal tmp_538[3] <== [tmp_536[0] + tmp_537[0], tmp_536[1] + tmp_537[1], tmp_536[2] + tmp_537[2]];
    signal tmp_539[3] <== [tmp_782[0] * mapValues.cm1_38, tmp_782[1] * mapValues.cm1_38, tmp_782[2] * mapValues.cm1_38];
    signal tmp_540[3] <== [tmp_538[0] + tmp_539[0], tmp_538[1] + tmp_539[1], tmp_538[2] + tmp_539[2]];
    signal tmp_541[3] <== [tmp_783[0] * mapValues.cm1_39, tmp_783[1] * mapValues.cm1_39, tmp_783[2] * mapValues.cm1_39];
    signal tmp_542[3] <== [tmp_540[0] + tmp_541[0], tmp_540[1] + tmp_541[1], tmp_540[2] + tmp_541[2]];
    signal tmp_543[3] <== [tmp_784[0] * mapValues.cm1_40, tmp_784[1] * mapValues.cm1_40, tmp_784[2] * mapValues.cm1_40];
    signal tmp_544[3] <== [tmp_542[0] + tmp_543[0], tmp_542[1] + tmp_543[1], tmp_542[2] + tmp_543[2]];
    signal tmp_545[3] <== [tmp_785[0] * mapValues.cm1_41, tmp_785[1] * mapValues.cm1_41, tmp_785[2] * mapValues.cm1_41];
    signal tmp_546[3] <== [tmp_544[0] + tmp_545[0], tmp_544[1] + tmp_545[1], tmp_544[2] + tmp_545[2]];
    signal tmp_547[3] <== [tmp_786[0] * mapValues.cm1_42, tmp_786[1] * mapValues.cm1_42, tmp_786[2] * mapValues.cm1_42];
    signal tmp_548[3] <== [tmp_546[0] + tmp_547[0], tmp_546[1] + tmp_547[1], tmp_546[2] + tmp_547[2]];
    signal tmp_549[3] <== [tmp_787[0] * mapValues.cm1_43, tmp_787[1] * mapValues.cm1_43, tmp_787[2] * mapValues.cm1_43];
    signal tmp_550[3] <== [tmp_548[0] + tmp_549[0], tmp_548[1] + tmp_549[1], tmp_548[2] + tmp_549[2]];
    signal tmp_551[3] <== [tmp_788[0] * mapValues.cm1_44, tmp_788[1] * mapValues.cm1_44, tmp_788[2] * mapValues.cm1_44];
    signal tmp_552[3] <== [tmp_550[0] + tmp_551[0], tmp_550[1] + tmp_551[1], tmp_550[2] + tmp_551[2]];
    signal tmp_553[3] <== [tmp_789[0] * mapValues.cm1_45, tmp_789[1] * mapValues.cm1_45, tmp_789[2] * mapValues.cm1_45];
    signal tmp_554[3] <== [tmp_552[0] + tmp_553[0], tmp_552[1] + tmp_553[1], tmp_552[2] + tmp_553[2]];
    signal tmp_555[3] <== [tmp_790[0] * mapValues.cm1_46, tmp_790[1] * mapValues.cm1_46, tmp_790[2] * mapValues.cm1_46];
    signal tmp_556[3] <== [tmp_554[0] + tmp_555[0], tmp_554[1] + tmp_555[1], tmp_554[2] + tmp_555[2]];
    signal tmp_557[3] <== [tmp_791[0] * mapValues.cm1_47, tmp_791[1] * mapValues.cm1_47, tmp_791[2] * mapValues.cm1_47];
    signal tmp_558[3] <== [tmp_556[0] + tmp_557[0], tmp_556[1] + tmp_557[1], tmp_556[2] + tmp_557[2]];
    signal tmp_559[3] <== CMul()(tmp_792, mapValues.cm2_1);
    signal tmp_560[3] <== [tmp_558[0] + tmp_559[0], tmp_558[1] + tmp_559[1], tmp_558[2] + tmp_559[2]];
    signal tmp_561[3] <== CMul()(tmp_793, mapValues.cm2_2);
    signal tmp_562[3] <== [tmp_560[0] + tmp_561[0], tmp_560[1] + tmp_561[1], tmp_560[2] + tmp_561[2]];
    signal tmp_563[3] <== CMul()(tmp_794, mapValues.cm2_3);
    signal tmp_564[3] <== [tmp_562[0] + tmp_563[0], tmp_562[1] + tmp_563[1], tmp_562[2] + tmp_563[2]];
    signal tmp_565[3] <== CMul()(tmp_795, mapValues.cm3_0);
    signal tmp_566[3] <== [tmp_564[0] + tmp_565[0], tmp_564[1] + tmp_565[1], tmp_564[2] + tmp_565[2]];
    signal tmp_567[3] <== CMul()(tmp_796, mapValues.cm3_1);
    signal tmp_568[3] <== [tmp_566[0] + tmp_567[0], tmp_566[1] + tmp_567[1], tmp_566[2] + tmp_567[2]];
    signal tmp_569[3] <== CMul()(tmp_797, mapValues.cm3_2);
    signal tmp_570[3] <== [tmp_568[0] + tmp_569[0], tmp_568[1] + tmp_569[1], tmp_568[2] + tmp_569[2]];
    signal tmp_571[3] <== CMul()(tmp_798, mapValues.cm3_3);
    signal tmp_572[3] <== [tmp_570[0] + tmp_571[0], tmp_570[1] + tmp_571[1], tmp_570[2] + tmp_571[2]];
    signal tmp_573[3] <== CMul()(tmp_799, mapValues.cm3_4);
    signal tmp_574[3] <== [tmp_572[0] + tmp_573[0], tmp_572[1] + tmp_573[1], tmp_572[2] + tmp_573[2]];
    signal tmp_575[3] <== CMul()(tmp_800, mapValues.cm3_5);
    signal tmp_576[3] <== [tmp_574[0] + tmp_575[0], tmp_574[1] + tmp_575[1], tmp_574[2] + tmp_575[2]];
    signal tmp_577[3] <== CMul()(tmp_801, mapValues.cm3_6);
    signal tmp_578[3] <== [tmp_576[0] + tmp_577[0], tmp_576[1] + tmp_577[1], tmp_576[2] + tmp_577[2]];
    signal tmp_579[3] <== CMul()(tmp_578, tmp_697);
    signal tmp_580[3] <== [tmp_411[0] + tmp_579[0], tmp_411[1] + tmp_579[1], tmp_411[2] + tmp_579[2]];
    signal tmp_581[3] <== [tmp_734[0] * consts[36], tmp_734[1] * consts[36], tmp_734[2] * consts[36]];
    signal tmp_582[3] <== [tmp_735[0] * consts[37], tmp_735[1] * consts[37], tmp_735[2] * consts[37]];
    signal tmp_583[3] <== [tmp_581[0] + tmp_582[0], tmp_581[1] + tmp_582[1], tmp_581[2] + tmp_582[2]];
    signal tmp_584[3] <== [tmp_743[0] * consts[46], tmp_743[1] * consts[46], tmp_743[2] * consts[46]];
    signal tmp_585[3] <== [tmp_583[0] + tmp_584[0], tmp_583[1] + tmp_584[1], tmp_583[2] + tmp_584[2]];
    signal tmp_586[3] <== [tmp_760[0] * mapValues.cm1_16, tmp_760[1] * mapValues.cm1_16, tmp_760[2] * mapValues.cm1_16];
    signal tmp_587[3] <== [tmp_585[0] + tmp_586[0], tmp_585[1] + tmp_586[1], tmp_585[2] + tmp_586[2]];
    signal tmp_588[3] <== [tmp_761[0] * mapValues.cm1_17, tmp_761[1] * mapValues.cm1_17, tmp_761[2] * mapValues.cm1_17];
    signal tmp_589[3] <== [tmp_587[0] + tmp_588[0], tmp_587[1] + tmp_588[1], tmp_587[2] + tmp_588[2]];
    signal tmp_590[3] <== [tmp_762[0] * mapValues.cm1_18, tmp_762[1] * mapValues.cm1_18, tmp_762[2] * mapValues.cm1_18];
    signal tmp_591[3] <== [tmp_589[0] + tmp_590[0], tmp_589[1] + tmp_590[1], tmp_589[2] + tmp_590[2]];
    signal tmp_592[3] <== [tmp_763[0] * mapValues.cm1_19, tmp_763[1] * mapValues.cm1_19, tmp_763[2] * mapValues.cm1_19];
    signal tmp_593[3] <== [tmp_591[0] + tmp_592[0], tmp_591[1] + tmp_592[1], tmp_591[2] + tmp_592[2]];
    signal tmp_594[3] <== [tmp_764[0] * mapValues.cm1_20, tmp_764[1] * mapValues.cm1_20, tmp_764[2] * mapValues.cm1_20];
    signal tmp_595[3] <== [tmp_593[0] + tmp_594[0], tmp_593[1] + tmp_594[1], tmp_593[2] + tmp_594[2]];
    signal tmp_596[3] <== [tmp_765[0] * mapValues.cm1_21, tmp_765[1] * mapValues.cm1_21, tmp_765[2] * mapValues.cm1_21];
    signal tmp_597[3] <== [tmp_595[0] + tmp_596[0], tmp_595[1] + tmp_596[1], tmp_595[2] + tmp_596[2]];
    signal tmp_598[3] <== [tmp_766[0] * mapValues.cm1_22, tmp_766[1] * mapValues.cm1_22, tmp_766[2] * mapValues.cm1_22];
    signal tmp_599[3] <== [tmp_597[0] + tmp_598[0], tmp_597[1] + tmp_598[1], tmp_597[2] + tmp_598[2]];
    signal tmp_600[3] <== [tmp_767[0] * mapValues.cm1_23, tmp_767[1] * mapValues.cm1_23, tmp_767[2] * mapValues.cm1_23];
    signal tmp_601[3] <== [tmp_599[0] + tmp_600[0], tmp_599[1] + tmp_600[1], tmp_599[2] + tmp_600[2]];
    signal tmp_602[3] <== [tmp_768[0] * mapValues.cm1_24, tmp_768[1] * mapValues.cm1_24, tmp_768[2] * mapValues.cm1_24];
    signal tmp_603[3] <== [tmp_601[0] + tmp_602[0], tmp_601[1] + tmp_602[1], tmp_601[2] + tmp_602[2]];
    signal tmp_604[3] <== [tmp_769[0] * mapValues.cm1_25, tmp_769[1] * mapValues.cm1_25, tmp_769[2] * mapValues.cm1_25];
    signal tmp_605[3] <== [tmp_603[0] + tmp_604[0], tmp_603[1] + tmp_604[1], tmp_603[2] + tmp_604[2]];
    signal tmp_606[3] <== [tmp_770[0] * mapValues.cm1_26, tmp_770[1] * mapValues.cm1_26, tmp_770[2] * mapValues.cm1_26];
    signal tmp_607[3] <== [tmp_605[0] + tmp_606[0], tmp_605[1] + tmp_606[1], tmp_605[2] + tmp_606[2]];
    signal tmp_608[3] <== [tmp_771[0] * mapValues.cm1_27, tmp_771[1] * mapValues.cm1_27, tmp_771[2] * mapValues.cm1_27];
    signal tmp_609[3] <== [tmp_607[0] + tmp_608[0], tmp_607[1] + tmp_608[1], tmp_607[2] + tmp_608[2]];
    signal tmp_610[3] <== [tmp_772[0] * mapValues.cm1_28, tmp_772[1] * mapValues.cm1_28, tmp_772[2] * mapValues.cm1_28];
    signal tmp_611[3] <== [tmp_609[0] + tmp_610[0], tmp_609[1] + tmp_610[1], tmp_609[2] + tmp_610[2]];
    signal tmp_612[3] <== [tmp_773[0] * mapValues.cm1_29, tmp_773[1] * mapValues.cm1_29, tmp_773[2] * mapValues.cm1_29];
    signal tmp_613[3] <== [tmp_611[0] + tmp_612[0], tmp_611[1] + tmp_612[1], tmp_611[2] + tmp_612[2]];
    signal tmp_614[3] <== [tmp_774[0] * mapValues.cm1_30, tmp_774[1] * mapValues.cm1_30, tmp_774[2] * mapValues.cm1_30];
    signal tmp_615[3] <== [tmp_613[0] + tmp_614[0], tmp_613[1] + tmp_614[1], tmp_613[2] + tmp_614[2]];
    signal tmp_616[3] <== [tmp_775[0] * mapValues.cm1_31, tmp_775[1] * mapValues.cm1_31, tmp_775[2] * mapValues.cm1_31];
    signal tmp_617[3] <== [tmp_615[0] + tmp_616[0], tmp_615[1] + tmp_616[1], tmp_615[2] + tmp_616[2]];
    signal tmp_618[3] <== [tmp_697[0] + tmp_802[0], tmp_697[1] + tmp_802[1], tmp_697[2] + tmp_802[2]];
    signal tmp_619[3] <== CMul()(tmp_617, tmp_618);
    signal tmp_620[3] <== [tmp_580[0] + tmp_619[0], tmp_580[1] + tmp_619[1], tmp_580[2] + tmp_619[2]];
    signal tmp_621[3] <== [tmp_759[0] * mapValues.cm1_15, tmp_759[1] * mapValues.cm1_15, tmp_759[2] * mapValues.cm1_15];
    signal tmp_623[3] <== [tmp_618[0] + xDivXSubXi[18][0], tmp_618[1] + xDivXSubXi[18][1], tmp_618[2] + xDivXSubXi[18][2]];
    signal tmp_624[3] <== CMul()(tmp_621, tmp_623);
    signal tmp_625[3] <== [tmp_620[0] + tmp_624[0], tmp_620[1] + tmp_624[1], tmp_620[2] + tmp_624[2]];
    signal tmp_626[3] <== CMul()(tmp_681, evals[0]);
    signal tmp_627[3] <== [tmp_625[0] - tmp_626[0], tmp_625[1] - tmp_626[1], tmp_625[2] - tmp_626[2]];
    signal tmp_628[3] <== CMul()(tmp_682, evals[1]);
    signal tmp_629[3] <== [tmp_627[0] - tmp_628[0], tmp_627[1] - tmp_628[1], tmp_627[2] - tmp_628[2]];
    signal tmp_630[3] <== CMul()(tmp_683, evals[2]);
    signal tmp_631[3] <== [tmp_629[0] - tmp_630[0], tmp_629[1] - tmp_630[1], tmp_629[2] - tmp_630[2]];
    signal tmp_632[3] <== CMul()(tmp_684, evals[3]);
    signal tmp_633[3] <== [tmp_631[0] - tmp_632[0], tmp_631[1] - tmp_632[1], tmp_631[2] - tmp_632[2]];
    signal tmp_634[3] <== CMul()(tmp_685, evals[4]);
    signal tmp_635[3] <== [tmp_633[0] - tmp_634[0], tmp_633[1] - tmp_634[1], tmp_633[2] - tmp_634[2]];
    signal tmp_636[3] <== CMul()(tmp_686, evals[5]);
    signal tmp_637[3] <== [tmp_635[0] - tmp_636[0], tmp_635[1] - tmp_636[1], tmp_635[2] - tmp_636[2]];
    signal tmp_638[3] <== CMul()(tmp_687, evals[6]);
    signal tmp_639[3] <== [tmp_637[0] - tmp_638[0], tmp_637[1] - tmp_638[1], tmp_637[2] - tmp_638[2]];
    signal tmp_640[3] <== CMul()(tmp_688, evals[7]);
    signal tmp_641[3] <== [tmp_639[0] - tmp_640[0], tmp_639[1] - tmp_640[1], tmp_639[2] - tmp_640[2]];
    signal tmp_642[3] <== CMul()(tmp_689, evals[8]);
    signal tmp_643[3] <== [tmp_641[0] - tmp_642[0], tmp_641[1] - tmp_642[1], tmp_641[2] - tmp_642[2]];
    signal tmp_644[3] <== CMul()(tmp_690, evals[9]);
    signal tmp_645[3] <== [tmp_643[0] - tmp_644[0], tmp_643[1] - tmp_644[1], tmp_643[2] - tmp_644[2]];
    signal tmp_646[3] <== CMul()(tmp_691, evals[10]);
    signal tmp_647[3] <== [tmp_645[0] - tmp_646[0], tmp_645[1] - tmp_646[1], tmp_645[2] - tmp_646[2]];
    signal tmp_648[3] <== CMul()(tmp_692, evals[11]);
    signal tmp_649[3] <== [tmp_647[0] - tmp_648[0], tmp_647[1] - tmp_648[1], tmp_647[2] - tmp_648[2]];
    signal tmp_650[3] <== CMul()(tmp_693, evals[12]);
    signal tmp_651[3] <== [tmp_649[0] - tmp_650[0], tmp_649[1] - tmp_650[1], tmp_649[2] - tmp_650[2]];
    signal tmp_652[3] <== CMul()(tmp_694, evals[13]);
    signal tmp_653[3] <== [tmp_651[0] - tmp_652[0], tmp_651[1] - tmp_652[1], tmp_651[2] - tmp_652[2]];
    signal tmp_654[3] <== CMul()(tmp_695, evals[14]);
    signal tmp_655[3] <== [tmp_653[0] - tmp_654[0], tmp_653[1] - tmp_654[1], tmp_653[2] - tmp_654[2]];
    signal tmp_656[3] <== CMul()(tmp_696, tmp_803);
    signal tmp_657[3] <== [tmp_655[0] - tmp_656[0], tmp_655[1] - tmp_656[1], tmp_655[2] - tmp_656[2]];
    signal tmp_658[3] <== CMul()(tmp_697, tmp_804);
    signal tmp_659[3] <== [tmp_657[0] - tmp_658[0], tmp_657[1] - tmp_658[1], tmp_657[2] - tmp_658[2]];
    signal tmp_660[3] <== CMul()(tmp_802, tmp_805);
    signal tmp_661[3] <== [tmp_659[0] - tmp_660[0], tmp_659[1] - tmp_660[1], tmp_659[2] - tmp_660[2]];
    signal tmp_662[3] <== CMul()(xDivXSubXi[18], tmp_806);
    tmp_807 <== [tmp_661[0] - tmp_662[0], tmp_661[1] - tmp_662[1], tmp_661[2] - tmp_662[2]];

}


template CalculateFRIConstantsChunks0() {
    signal input challengesDeep[2][3];
    signal input evals[144][3];
    signal output tmp_664[3];
    signal output tmp_665[3];
    signal output tmp_666[3];
    signal output tmp_667[3];
    signal output tmp_668[3];
    signal output tmp_669[3];
    signal output tmp_670[3];
    signal output tmp_671[3];
    signal output tmp_672[3];
    signal output tmp_673[3];
    signal output tmp_674[3];
    signal output tmp_675[3];
    signal output tmp_676[3];
    signal output tmp_677[3];
    signal output tmp_678[3];
    signal output tmp_679[3];
    signal output tmp_680[3];
    signal output tmp_698[3];
    signal output tmp_699[3];
    signal output tmp_700[3];
    signal output tmp_701[3];
    signal output tmp_702[3];
    signal output tmp_703[3];
    signal output tmp_704[3];
    signal output tmp_705[3];
    signal output tmp_706[3];
    signal output tmp_707[3];
    signal output tmp_708[3];
    signal output tmp_709[3];
    signal output tmp_710[3];
    signal output tmp_711[3];
    signal output tmp_712[3];
    signal output tmp_713[3];
    signal output tmp_714[3];
    signal output tmp_715[3];
    signal output tmp_716[3];
    signal output tmp_717[3];
    signal output tmp_718[3];
    signal output tmp_719[3];
    signal output tmp_720[3];
    signal output tmp_721[3];
    signal output tmp_722[3];
    signal output tmp_723[3];
    signal output tmp_724[3];
    signal output tmp_725[3];
    signal output tmp_726[3];
    signal output tmp_727[3];
    signal output tmp_728[3];
    signal output tmp_729[3];
    signal output tmp_730[3];
    signal output tmp_731[3];
    signal output tmp_732[3];
    signal output tmp_733[3];
    signal output tmp_734[3];
    signal output tmp_735[3];
    signal output tmp_736[3];
    signal output tmp_737[3];
    signal output tmp_738[3];
    signal output tmp_739[3];
    signal output tmp_740[3];
    signal output tmp_741[3];
    signal output tmp_742[3];
    signal output tmp_743[3];
    signal output tmp_744[3];
    signal output tmp_745[3];
    signal output tmp_746[3];
    signal output tmp_747[3];
    signal output tmp_748[3];
    signal output tmp_749[3];
    signal output tmp_750[3];
    signal output tmp_751[3];
    signal output tmp_752[3];
    signal output tmp_753[3];
    signal output tmp_754[3];
    signal output tmp_755[3];
    signal output tmp_756[3];
    signal output tmp_757[3];
    signal output tmp_758[3];
    signal output tmp_759[3];
    signal output tmp_760[3];
    signal output tmp_761[3];
    signal output tmp_762[3];
    signal output tmp_763[3];
    signal output tmp_764[3];
    signal output tmp_765[3];
    signal output tmp_766[3];
    signal output tmp_767[3];
    signal output tmp_768[3];
    signal output tmp_769[3];
    signal output tmp_770[3];
    signal output tmp_771[3];
    signal output tmp_772[3];
    signal output tmp_773[3];
    signal output tmp_774[3];
    signal output tmp_775[3];
    signal output tmp_776[3];
    signal output tmp_777[3];
    signal output tmp_778[3];
    signal output tmp_779[3];
    signal output tmp_780[3];
    signal output tmp_781[3];
    signal output tmp_782[3];
    signal output tmp_783[3];
    signal output tmp_784[3];
    signal output tmp_785[3];
    signal output tmp_786[3];
    signal output tmp_787[3];
    signal output tmp_788[3];
    signal output tmp_789[3];
    signal output tmp_790[3];
    signal output tmp_791[3];
    signal output tmp_792[3];
    signal output tmp_793[3];
    signal output tmp_794[3];
    signal output tmp_795[3];
    signal output tmp_796[3];
    signal output tmp_797[3];
    signal output tmp_798[3];
    signal output tmp_799[3];
    signal output tmp_800[3];
    signal output tmp_801[3];
    signal output tmp_803[3];
    signal output tmp_804[3];
    signal output tmp_805[3];
    signal output tmp_806[3];
    tmp_664 <== CMul()(challengesDeep[0], challengesDeep[0]);
    tmp_665 <== CMul()(tmp_664, challengesDeep[0]);
    tmp_666 <== CMul()(tmp_665, challengesDeep[0]);
    tmp_667 <== CMul()(tmp_666, challengesDeep[0]);
    tmp_668 <== CMul()(tmp_667, challengesDeep[0]);
    tmp_669 <== CMul()(tmp_668, challengesDeep[0]);
    tmp_670 <== CMul()(tmp_669, challengesDeep[0]);
    tmp_671 <== CMul()(tmp_670, challengesDeep[0]);
    tmp_672 <== CMul()(tmp_671, challengesDeep[0]);
    tmp_673 <== CMul()(tmp_672, challengesDeep[0]);
    tmp_674 <== CMul()(tmp_673, challengesDeep[0]);
    tmp_675 <== CMul()(tmp_674, challengesDeep[0]);
    tmp_676 <== CMul()(tmp_675, challengesDeep[0]);
    tmp_677 <== CMul()(tmp_676, challengesDeep[0]);
    tmp_678 <== CMul()(tmp_677, challengesDeep[0]);
    tmp_679 <== CMul()(tmp_678, challengesDeep[0]);
    tmp_680 <== CMul()(tmp_679, challengesDeep[0]);
    tmp_698 <== CMul()(challengesDeep[1], challengesDeep[1]);
    tmp_699 <== CMul()(tmp_698, challengesDeep[1]);
    tmp_700 <== CMul()(tmp_699, challengesDeep[1]);
    tmp_701 <== CMul()(tmp_700, challengesDeep[1]);
    tmp_702 <== CMul()(tmp_701, challengesDeep[1]);
    tmp_703 <== CMul()(tmp_702, challengesDeep[1]);
    tmp_704 <== CMul()(tmp_703, challengesDeep[1]);
    tmp_705 <== CMul()(tmp_704, challengesDeep[1]);
    tmp_706 <== CMul()(tmp_705, challengesDeep[1]);
    tmp_707 <== CMul()(tmp_706, challengesDeep[1]);
    tmp_708 <== CMul()(tmp_707, challengesDeep[1]);
    tmp_709 <== CMul()(tmp_708, challengesDeep[1]);
    tmp_710 <== CMul()(tmp_709, challengesDeep[1]);
    tmp_711 <== CMul()(tmp_710, challengesDeep[1]);
    tmp_712 <== CMul()(tmp_711, challengesDeep[1]);
    tmp_713 <== CMul()(tmp_712, challengesDeep[1]);
    tmp_714 <== CMul()(tmp_713, challengesDeep[1]);
    tmp_715 <== CMul()(tmp_714, challengesDeep[1]);
    tmp_716 <== CMul()(tmp_715, challengesDeep[1]);
    tmp_717 <== CMul()(tmp_716, challengesDeep[1]);
    tmp_718 <== CMul()(tmp_717, challengesDeep[1]);
    tmp_719 <== CMul()(tmp_718, challengesDeep[1]);
    tmp_720 <== CMul()(tmp_719, challengesDeep[1]);
    tmp_721 <== CMul()(tmp_720, challengesDeep[1]);
    tmp_722 <== CMul()(tmp_721, challengesDeep[1]);
    tmp_723 <== CMul()(tmp_722, challengesDeep[1]);
    tmp_724 <== CMul()(tmp_723, challengesDeep[1]);
    tmp_725 <== CMul()(tmp_724, challengesDeep[1]);
    tmp_726 <== CMul()(tmp_725, challengesDeep[1]);
    tmp_727 <== CMul()(tmp_726, challengesDeep[1]);
    tmp_728 <== CMul()(tmp_727, challengesDeep[1]);
    tmp_729 <== CMul()(tmp_728, challengesDeep[1]);
    tmp_730 <== CMul()(tmp_729, challengesDeep[1]);
    tmp_731 <== CMul()(tmp_730, challengesDeep[1]);
    tmp_732 <== CMul()(tmp_731, challengesDeep[1]);
    tmp_733 <== CMul()(tmp_732, challengesDeep[1]);
    tmp_734 <== CMul()(tmp_733, challengesDeep[1]);
    tmp_735 <== CMul()(tmp_734, challengesDeep[1]);
    tmp_736 <== CMul()(tmp_735, challengesDeep[1]);
    tmp_737 <== CMul()(tmp_736, challengesDeep[1]);
    tmp_738 <== CMul()(tmp_737, challengesDeep[1]);
    tmp_739 <== CMul()(tmp_738, challengesDeep[1]);
    tmp_740 <== CMul()(tmp_739, challengesDeep[1]);
    tmp_741 <== CMul()(tmp_740, challengesDeep[1]);
    tmp_742 <== CMul()(tmp_741, challengesDeep[1]);
    tmp_743 <== CMul()(tmp_742, challengesDeep[1]);
    tmp_744 <== CMul()(tmp_743, challengesDeep[1]);
    tmp_745 <== CMul()(tmp_744, challengesDeep[1]);
    tmp_746 <== CMul()(tmp_745, challengesDeep[1]);
    tmp_747 <== CMul()(tmp_746, challengesDeep[1]);
    tmp_748 <== CMul()(tmp_747, challengesDeep[1]);
    tmp_749 <== CMul()(tmp_748, challengesDeep[1]);
    tmp_750 <== CMul()(tmp_749, challengesDeep[1]);
    tmp_751 <== CMul()(tmp_750, challengesDeep[1]);
    tmp_752 <== CMul()(tmp_751, challengesDeep[1]);
    tmp_753 <== CMul()(tmp_752, challengesDeep[1]);
    tmp_754 <== CMul()(tmp_753, challengesDeep[1]);
    tmp_755 <== CMul()(tmp_754, challengesDeep[1]);
    tmp_756 <== CMul()(tmp_755, challengesDeep[1]);
    tmp_757 <== CMul()(tmp_756, challengesDeep[1]);
    tmp_758 <== CMul()(tmp_757, challengesDeep[1]);
    tmp_759 <== CMul()(tmp_758, challengesDeep[1]);
    tmp_760 <== CMul()(tmp_759, challengesDeep[1]);
    tmp_761 <== CMul()(tmp_760, challengesDeep[1]);
    tmp_762 <== CMul()(tmp_761, challengesDeep[1]);
    tmp_763 <== CMul()(tmp_762, challengesDeep[1]);
    tmp_764 <== CMul()(tmp_763, challengesDeep[1]);
    tmp_765 <== CMul()(tmp_764, challengesDeep[1]);
    tmp_766 <== CMul()(tmp_765, challengesDeep[1]);
    tmp_767 <== CMul()(tmp_766, challengesDeep[1]);
    tmp_768 <== CMul()(tmp_767, challengesDeep[1]);
    tmp_769 <== CMul()(tmp_768, challengesDeep[1]);
    tmp_770 <== CMul()(tmp_769, challengesDeep[1]);
    tmp_771 <== CMul()(tmp_770, challengesDeep[1]);
    tmp_772 <== CMul()(tmp_771, challengesDeep[1]);
    tmp_773 <== CMul()(tmp_772, challengesDeep[1]);
    tmp_774 <== CMul()(tmp_773, challengesDeep[1]);
    tmp_775 <== CMul()(tmp_774, challengesDeep[1]);
    tmp_776 <== CMul()(tmp_775, challengesDeep[1]);
    tmp_777 <== CMul()(tmp_776, challengesDeep[1]);
    tmp_778 <== CMul()(tmp_777, challengesDeep[1]);
    tmp_779 <== CMul()(tmp_778, challengesDeep[1]);
    tmp_780 <== CMul()(tmp_779, challengesDeep[1]);
    tmp_781 <== CMul()(tmp_780, challengesDeep[1]);
    tmp_782 <== CMul()(tmp_781, challengesDeep[1]);
    tmp_783 <== CMul()(tmp_782, challengesDeep[1]);
    tmp_784 <== CMul()(tmp_783, challengesDeep[1]);
    tmp_785 <== CMul()(tmp_784, challengesDeep[1]);
    tmp_786 <== CMul()(tmp_785, challengesDeep[1]);
    tmp_787 <== CMul()(tmp_786, challengesDeep[1]);
    tmp_788 <== CMul()(tmp_787, challengesDeep[1]);
    tmp_789 <== CMul()(tmp_788, challengesDeep[1]);
    tmp_790 <== CMul()(tmp_789, challengesDeep[1]);
    tmp_791 <== CMul()(tmp_790, challengesDeep[1]);
    tmp_792 <== CMul()(tmp_791, challengesDeep[1]);
    tmp_793 <== CMul()(tmp_792, challengesDeep[1]);
    tmp_794 <== CMul()(tmp_793, challengesDeep[1]);
    tmp_795 <== CMul()(tmp_794, challengesDeep[1]);
    tmp_796 <== CMul()(tmp_795, challengesDeep[1]);
    tmp_797 <== CMul()(tmp_796, challengesDeep[1]);
    tmp_798 <== CMul()(tmp_797, challengesDeep[1]);
    tmp_799 <== CMul()(tmp_798, challengesDeep[1]);
    tmp_800 <== CMul()(tmp_799, challengesDeep[1]);
    tmp_801 <== CMul()(tmp_800, challengesDeep[1]);
    signal tmp_139[3] <== CMul()(challengesDeep[1], evals[16]);
    tmp_803 <== [evals[15][0] + tmp_139[0], evals[15][1] + tmp_139[1], evals[15][2] + tmp_139[2]];
    signal tmp_141[3] <== CMul()(challengesDeep[1], evals[112]);
    signal tmp_142[3] <== [evals[61][0] + tmp_141[0], evals[61][1] + tmp_141[1], evals[61][2] + tmp_141[2]];
    signal tmp_143[3] <== CMul()(tmp_698, evals[17]);
    signal tmp_144[3] <== [tmp_142[0] + tmp_143[0], tmp_142[1] + tmp_143[1], tmp_142[2] + tmp_143[2]];
    signal tmp_145[3] <== CMul()(tmp_699, evals[18]);
    signal tmp_146[3] <== [tmp_144[0] + tmp_145[0], tmp_144[1] + tmp_145[1], tmp_144[2] + tmp_145[2]];
    signal tmp_147[3] <== CMul()(tmp_700, evals[19]);
    signal tmp_148[3] <== [tmp_146[0] + tmp_147[0], tmp_146[1] + tmp_147[1], tmp_146[2] + tmp_147[2]];
    signal tmp_149[3] <== CMul()(tmp_701, evals[20]);
    signal tmp_150[3] <== [tmp_148[0] + tmp_149[0], tmp_148[1] + tmp_149[1], tmp_148[2] + tmp_149[2]];
    signal tmp_151[3] <== CMul()(tmp_702, evals[21]);
    signal tmp_152[3] <== [tmp_150[0] + tmp_151[0], tmp_150[1] + tmp_151[1], tmp_150[2] + tmp_151[2]];
    signal tmp_153[3] <== CMul()(tmp_703, evals[22]);
    signal tmp_154[3] <== [tmp_152[0] + tmp_153[0], tmp_152[1] + tmp_153[1], tmp_152[2] + tmp_153[2]];
    signal tmp_155[3] <== CMul()(tmp_704, evals[23]);
    signal tmp_156[3] <== [tmp_154[0] + tmp_155[0], tmp_154[1] + tmp_155[1], tmp_154[2] + tmp_155[2]];
    signal tmp_157[3] <== CMul()(tmp_705, evals[24]);
    signal tmp_158[3] <== [tmp_156[0] + tmp_157[0], tmp_156[1] + tmp_157[1], tmp_156[2] + tmp_157[2]];
    signal tmp_159[3] <== CMul()(tmp_706, evals[25]);
    signal tmp_160[3] <== [tmp_158[0] + tmp_159[0], tmp_158[1] + tmp_159[1], tmp_158[2] + tmp_159[2]];
    signal tmp_161[3] <== CMul()(tmp_707, evals[26]);
    signal tmp_162[3] <== [tmp_160[0] + tmp_161[0], tmp_160[1] + tmp_161[1], tmp_160[2] + tmp_161[2]];
    signal tmp_163[3] <== CMul()(tmp_708, evals[27]);
    signal tmp_164[3] <== [tmp_162[0] + tmp_163[0], tmp_162[1] + tmp_163[1], tmp_162[2] + tmp_163[2]];
    signal tmp_165[3] <== CMul()(tmp_709, evals[28]);
    signal tmp_166[3] <== [tmp_164[0] + tmp_165[0], tmp_164[1] + tmp_165[1], tmp_164[2] + tmp_165[2]];
    signal tmp_167[3] <== CMul()(tmp_710, evals[29]);
    signal tmp_168[3] <== [tmp_166[0] + tmp_167[0], tmp_166[1] + tmp_167[1], tmp_166[2] + tmp_167[2]];
    signal tmp_169[3] <== CMul()(tmp_711, evals[30]);
    signal tmp_170[3] <== [tmp_168[0] + tmp_169[0], tmp_168[1] + tmp_169[1], tmp_168[2] + tmp_169[2]];
    signal tmp_171[3] <== CMul()(tmp_712, evals[31]);
    signal tmp_172[3] <== [tmp_170[0] + tmp_171[0], tmp_170[1] + tmp_171[1], tmp_170[2] + tmp_171[2]];
    signal tmp_173[3] <== CMul()(tmp_713, evals[32]);
    signal tmp_174[3] <== [tmp_172[0] + tmp_173[0], tmp_172[1] + tmp_173[1], tmp_172[2] + tmp_173[2]];
    signal tmp_175[3] <== CMul()(tmp_714, evals[33]);
    signal tmp_176[3] <== [tmp_174[0] + tmp_175[0], tmp_174[1] + tmp_175[1], tmp_174[2] + tmp_175[2]];
    signal tmp_177[3] <== CMul()(tmp_715, evals[34]);
    signal tmp_178[3] <== [tmp_176[0] + tmp_177[0], tmp_176[1] + tmp_177[1], tmp_176[2] + tmp_177[2]];
    signal tmp_179[3] <== CMul()(tmp_716, evals[35]);
    signal tmp_180[3] <== [tmp_178[0] + tmp_179[0], tmp_178[1] + tmp_179[1], tmp_178[2] + tmp_179[2]];
    signal tmp_181[3] <== CMul()(tmp_717, evals[36]);
    signal tmp_182[3] <== [tmp_180[0] + tmp_181[0], tmp_180[1] + tmp_181[1], tmp_180[2] + tmp_181[2]];
    signal tmp_183[3] <== CMul()(tmp_718, evals[37]);
    signal tmp_184[3] <== [tmp_182[0] + tmp_183[0], tmp_182[1] + tmp_183[1], tmp_182[2] + tmp_183[2]];
    signal tmp_185[3] <== CMul()(tmp_719, evals[38]);
    signal tmp_186[3] <== [tmp_184[0] + tmp_185[0], tmp_184[1] + tmp_185[1], tmp_184[2] + tmp_185[2]];
    signal tmp_187[3] <== CMul()(tmp_720, evals[39]);
    signal tmp_188[3] <== [tmp_186[0] + tmp_187[0], tmp_186[1] + tmp_187[1], tmp_186[2] + tmp_187[2]];
    signal tmp_189[3] <== CMul()(tmp_721, evals[40]);
    signal tmp_190[3] <== [tmp_188[0] + tmp_189[0], tmp_188[1] + tmp_189[1], tmp_188[2] + tmp_189[2]];
    signal tmp_191[3] <== CMul()(tmp_722, evals[41]);
    signal tmp_192[3] <== [tmp_190[0] + tmp_191[0], tmp_190[1] + tmp_191[1], tmp_190[2] + tmp_191[2]];
    signal tmp_193[3] <== CMul()(tmp_723, evals[42]);
    signal tmp_194[3] <== [tmp_192[0] + tmp_193[0], tmp_192[1] + tmp_193[1], tmp_192[2] + tmp_193[2]];
    signal tmp_195[3] <== CMul()(tmp_724, evals[43]);
    signal tmp_196[3] <== [tmp_194[0] + tmp_195[0], tmp_194[1] + tmp_195[1], tmp_194[2] + tmp_195[2]];
    signal tmp_197[3] <== CMul()(tmp_725, evals[44]);
    signal tmp_198[3] <== [tmp_196[0] + tmp_197[0], tmp_196[1] + tmp_197[1], tmp_196[2] + tmp_197[2]];
    signal tmp_199[3] <== CMul()(tmp_726, evals[45]);
    signal tmp_200[3] <== [tmp_198[0] + tmp_199[0], tmp_198[1] + tmp_199[1], tmp_198[2] + tmp_199[2]];
    signal tmp_201[3] <== CMul()(tmp_727, evals[46]);
    signal tmp_202[3] <== [tmp_200[0] + tmp_201[0], tmp_200[1] + tmp_201[1], tmp_200[2] + tmp_201[2]];
    signal tmp_203[3] <== CMul()(tmp_728, evals[47]);
    signal tmp_204[3] <== [tmp_202[0] + tmp_203[0], tmp_202[1] + tmp_203[1], tmp_202[2] + tmp_203[2]];
    signal tmp_205[3] <== CMul()(tmp_729, evals[48]);
    signal tmp_206[3] <== [tmp_204[0] + tmp_205[0], tmp_204[1] + tmp_205[1], tmp_204[2] + tmp_205[2]];
    signal tmp_207[3] <== CMul()(tmp_730, evals[49]);
    signal tmp_208[3] <== [tmp_206[0] + tmp_207[0], tmp_206[1] + tmp_207[1], tmp_206[2] + tmp_207[2]];
    signal tmp_209[3] <== CMul()(tmp_731, evals[50]);
    signal tmp_210[3] <== [tmp_208[0] + tmp_209[0], tmp_208[1] + tmp_209[1], tmp_208[2] + tmp_209[2]];
    signal tmp_211[3] <== CMul()(tmp_732, evals[51]);
    signal tmp_212[3] <== [tmp_210[0] + tmp_211[0], tmp_210[1] + tmp_211[1], tmp_210[2] + tmp_211[2]];
    signal tmp_213[3] <== CMul()(tmp_733, evals[52]);
    signal tmp_214[3] <== [tmp_212[0] + tmp_213[0], tmp_212[1] + tmp_213[1], tmp_212[2] + tmp_213[2]];
    signal tmp_215[3] <== CMul()(tmp_734, evals[53]);
    signal tmp_216[3] <== [tmp_214[0] + tmp_215[0], tmp_214[1] + tmp_215[1], tmp_214[2] + tmp_215[2]];
    signal tmp_217[3] <== CMul()(tmp_735, evals[54]);
    signal tmp_218[3] <== [tmp_216[0] + tmp_217[0], tmp_216[1] + tmp_217[1], tmp_216[2] + tmp_217[2]];
    signal tmp_219[3] <== CMul()(tmp_736, evals[55]);
    signal tmp_220[3] <== [tmp_218[0] + tmp_219[0], tmp_218[1] + tmp_219[1], tmp_218[2] + tmp_219[2]];
    signal tmp_221[3] <== CMul()(tmp_737, evals[56]);
    signal tmp_222[3] <== [tmp_220[0] + tmp_221[0], tmp_220[1] + tmp_221[1], tmp_220[2] + tmp_221[2]];
    signal tmp_223[3] <== CMul()(tmp_738, evals[57]);
    signal tmp_224[3] <== [tmp_222[0] + tmp_223[0], tmp_222[1] + tmp_223[1], tmp_222[2] + tmp_223[2]];
    signal tmp_225[3] <== CMul()(tmp_739, evals[58]);
    signal tmp_226[3] <== [tmp_224[0] + tmp_225[0], tmp_224[1] + tmp_225[1], tmp_224[2] + tmp_225[2]];
    signal tmp_227[3] <== CMul()(tmp_740, evals[59]);
    signal tmp_228[3] <== [tmp_226[0] + tmp_227[0], tmp_226[1] + tmp_227[1], tmp_226[2] + tmp_227[2]];
    signal tmp_229[3] <== CMul()(tmp_741, evals[60]);
    signal tmp_230[3] <== [tmp_228[0] + tmp_229[0], tmp_228[1] + tmp_229[1], tmp_228[2] + tmp_229[2]];
    signal tmp_231[3] <== CMul()(tmp_742, evals[62]);
    signal tmp_232[3] <== [tmp_230[0] + tmp_231[0], tmp_230[1] + tmp_231[1], tmp_230[2] + tmp_231[2]];
    signal tmp_233[3] <== CMul()(tmp_743, evals[63]);
    signal tmp_234[3] <== [tmp_232[0] + tmp_233[0], tmp_232[1] + tmp_233[1], tmp_232[2] + tmp_233[2]];
    signal tmp_235[3] <== CMul()(tmp_744, evals[64]);
    signal tmp_236[3] <== [tmp_234[0] + tmp_235[0], tmp_234[1] + tmp_235[1], tmp_234[2] + tmp_235[2]];
    signal tmp_237[3] <== CMul()(tmp_745, evals[65]);
    signal tmp_238[3] <== [tmp_236[0] + tmp_237[0], tmp_236[1] + tmp_237[1], tmp_236[2] + tmp_237[2]];
    signal tmp_239[3] <== CMul()(tmp_746, evals[66]);
    signal tmp_240[3] <== [tmp_238[0] + tmp_239[0], tmp_238[1] + tmp_239[1], tmp_238[2] + tmp_239[2]];
    signal tmp_241[3] <== CMul()(tmp_747, evals[67]);
    signal tmp_242[3] <== [tmp_240[0] + tmp_241[0], tmp_240[1] + tmp_241[1], tmp_240[2] + tmp_241[2]];
    signal tmp_243[3] <== CMul()(tmp_748, evals[68]);
    signal tmp_244[3] <== [tmp_242[0] + tmp_243[0], tmp_242[1] + tmp_243[1], tmp_242[2] + tmp_243[2]];
    signal tmp_245[3] <== CMul()(tmp_749, evals[69]);
    signal tmp_246[3] <== [tmp_244[0] + tmp_245[0], tmp_244[1] + tmp_245[1], tmp_244[2] + tmp_245[2]];
    signal tmp_247[3] <== CMul()(tmp_750, evals[70]);
    signal tmp_248[3] <== [tmp_246[0] + tmp_247[0], tmp_246[1] + tmp_247[1], tmp_246[2] + tmp_247[2]];
    signal tmp_249[3] <== CMul()(tmp_751, evals[71]);
    signal tmp_250[3] <== [tmp_248[0] + tmp_249[0], tmp_248[1] + tmp_249[1], tmp_248[2] + tmp_249[2]];
    signal tmp_251[3] <== CMul()(tmp_752, evals[72]);
    signal tmp_252[3] <== [tmp_250[0] + tmp_251[0], tmp_250[1] + tmp_251[1], tmp_250[2] + tmp_251[2]];
    signal tmp_253[3] <== CMul()(tmp_753, evals[73]);
    signal tmp_254[3] <== [tmp_252[0] + tmp_253[0], tmp_252[1] + tmp_253[1], tmp_252[2] + tmp_253[2]];
    signal tmp_255[3] <== CMul()(tmp_754, evals[74]);
    signal tmp_256[3] <== [tmp_254[0] + tmp_255[0], tmp_254[1] + tmp_255[1], tmp_254[2] + tmp_255[2]];
    signal tmp_257[3] <== CMul()(tmp_755, evals[75]);
    signal tmp_258[3] <== [tmp_256[0] + tmp_257[0], tmp_256[1] + tmp_257[1], tmp_256[2] + tmp_257[2]];
    signal tmp_259[3] <== CMul()(tmp_756, evals[76]);
    signal tmp_260[3] <== [tmp_258[0] + tmp_259[0], tmp_258[1] + tmp_259[1], tmp_258[2] + tmp_259[2]];
    signal tmp_261[3] <== CMul()(tmp_757, evals[77]);
    signal tmp_262[3] <== [tmp_260[0] + tmp_261[0], tmp_260[1] + tmp_261[1], tmp_260[2] + tmp_261[2]];
    signal tmp_263[3] <== CMul()(tmp_758, evals[78]);
    signal tmp_264[3] <== [tmp_262[0] + tmp_263[0], tmp_262[1] + tmp_263[1], tmp_262[2] + tmp_263[2]];
    signal tmp_265[3] <== CMul()(tmp_759, evals[79]);
    signal tmp_266[3] <== [tmp_264[0] + tmp_265[0], tmp_264[1] + tmp_265[1], tmp_264[2] + tmp_265[2]];
    signal tmp_267[3] <== CMul()(tmp_760, evals[80]);
    signal tmp_268[3] <== [tmp_266[0] + tmp_267[0], tmp_266[1] + tmp_267[1], tmp_266[2] + tmp_267[2]];
    signal tmp_269[3] <== CMul()(tmp_761, evals[81]);
    signal tmp_270[3] <== [tmp_268[0] + tmp_269[0], tmp_268[1] + tmp_269[1], tmp_268[2] + tmp_269[2]];
    signal tmp_271[3] <== CMul()(tmp_762, evals[82]);
    signal tmp_272[3] <== [tmp_270[0] + tmp_271[0], tmp_270[1] + tmp_271[1], tmp_270[2] + tmp_271[2]];
    signal tmp_273[3] <== CMul()(tmp_763, evals[83]);
    signal tmp_274[3] <== [tmp_272[0] + tmp_273[0], tmp_272[1] + tmp_273[1], tmp_272[2] + tmp_273[2]];
    signal tmp_275[3] <== CMul()(tmp_764, evals[84]);
    signal tmp_276[3] <== [tmp_274[0] + tmp_275[0], tmp_274[1] + tmp_275[1], tmp_274[2] + tmp_275[2]];
    signal tmp_277[3] <== CMul()(tmp_765, evals[85]);
    signal tmp_278[3] <== [tmp_276[0] + tmp_277[0], tmp_276[1] + tmp_277[1], tmp_276[2] + tmp_277[2]];
    signal tmp_279[3] <== CMul()(tmp_766, evals[86]);
    signal tmp_280[3] <== [tmp_278[0] + tmp_279[0], tmp_278[1] + tmp_279[1], tmp_278[2] + tmp_279[2]];
    signal tmp_281[3] <== CMul()(tmp_767, evals[87]);
    signal tmp_282[3] <== [tmp_280[0] + tmp_281[0], tmp_280[1] + tmp_281[1], tmp_280[2] + tmp_281[2]];
    signal tmp_283[3] <== CMul()(tmp_768, evals[88]);
    signal tmp_284[3] <== [tmp_282[0] + tmp_283[0], tmp_282[1] + tmp_283[1], tmp_282[2] + tmp_283[2]];
    signal tmp_285[3] <== CMul()(tmp_769, evals[89]);
    signal tmp_286[3] <== [tmp_284[0] + tmp_285[0], tmp_284[1] + tmp_285[1], tmp_284[2] + tmp_285[2]];
    signal tmp_287[3] <== CMul()(tmp_770, evals[90]);
    signal tmp_288[3] <== [tmp_286[0] + tmp_287[0], tmp_286[1] + tmp_287[1], tmp_286[2] + tmp_287[2]];
    signal tmp_289[3] <== CMul()(tmp_771, evals[91]);
    signal tmp_290[3] <== [tmp_288[0] + tmp_289[0], tmp_288[1] + tmp_289[1], tmp_288[2] + tmp_289[2]];
    signal tmp_291[3] <== CMul()(tmp_772, evals[92]);
    signal tmp_292[3] <== [tmp_290[0] + tmp_291[0], tmp_290[1] + tmp_291[1], tmp_290[2] + tmp_291[2]];
    signal tmp_293[3] <== CMul()(tmp_773, evals[93]);
    signal tmp_294[3] <== [tmp_292[0] + tmp_293[0], tmp_292[1] + tmp_293[1], tmp_292[2] + tmp_293[2]];
    signal tmp_295[3] <== CMul()(tmp_774, evals[94]);
    signal tmp_296[3] <== [tmp_294[0] + tmp_295[0], tmp_294[1] + tmp_295[1], tmp_294[2] + tmp_295[2]];
    signal tmp_297[3] <== CMul()(tmp_775, evals[95]);
    signal tmp_298[3] <== [tmp_296[0] + tmp_297[0], tmp_296[1] + tmp_297[1], tmp_296[2] + tmp_297[2]];
    signal tmp_299[3] <== CMul()(tmp_776, evals[96]);
    signal tmp_300[3] <== [tmp_298[0] + tmp_299[0], tmp_298[1] + tmp_299[1], tmp_298[2] + tmp_299[2]];
    signal tmp_301[3] <== CMul()(tmp_777, evals[97]);
    signal tmp_302[3] <== [tmp_300[0] + tmp_301[0], tmp_300[1] + tmp_301[1], tmp_300[2] + tmp_301[2]];
    signal tmp_303[3] <== CMul()(tmp_778, evals[98]);
    signal tmp_304[3] <== [tmp_302[0] + tmp_303[0], tmp_302[1] + tmp_303[1], tmp_302[2] + tmp_303[2]];
    signal tmp_305[3] <== CMul()(tmp_779, evals[99]);
    signal tmp_306[3] <== [tmp_304[0] + tmp_305[0], tmp_304[1] + tmp_305[1], tmp_304[2] + tmp_305[2]];
    signal tmp_307[3] <== CMul()(tmp_780, evals[100]);
    signal tmp_308[3] <== [tmp_306[0] + tmp_307[0], tmp_306[1] + tmp_307[1], tmp_306[2] + tmp_307[2]];
    signal tmp_309[3] <== CMul()(tmp_781, evals[101]);
    signal tmp_310[3] <== [tmp_308[0] + tmp_309[0], tmp_308[1] + tmp_309[1], tmp_308[2] + tmp_309[2]];
    signal tmp_311[3] <== CMul()(tmp_782, evals[102]);
    signal tmp_312[3] <== [tmp_310[0] + tmp_311[0], tmp_310[1] + tmp_311[1], tmp_310[2] + tmp_311[2]];
    signal tmp_313[3] <== CMul()(tmp_783, evals[103]);
    signal tmp_314[3] <== [tmp_312[0] + tmp_313[0], tmp_312[1] + tmp_313[1], tmp_312[2] + tmp_313[2]];
    signal tmp_315[3] <== CMul()(tmp_784, evals[104]);
    signal tmp_316[3] <== [tmp_314[0] + tmp_315[0], tmp_314[1] + tmp_315[1], tmp_314[2] + tmp_315[2]];
    signal tmp_317[3] <== CMul()(tmp_785, evals[105]);
    signal tmp_318[3] <== [tmp_316[0] + tmp_317[0], tmp_316[1] + tmp_317[1], tmp_316[2] + tmp_317[2]];
    signal tmp_319[3] <== CMul()(tmp_786, evals[106]);
    signal tmp_320[3] <== [tmp_318[0] + tmp_319[0], tmp_318[1] + tmp_319[1], tmp_318[2] + tmp_319[2]];
    signal tmp_321[3] <== CMul()(tmp_787, evals[107]);
    signal tmp_322[3] <== [tmp_320[0] + tmp_321[0], tmp_320[1] + tmp_321[1], tmp_320[2] + tmp_321[2]];
    signal tmp_323[3] <== CMul()(tmp_788, evals[108]);
    signal tmp_324[3] <== [tmp_322[0] + tmp_323[0], tmp_322[1] + tmp_323[1], tmp_322[2] + tmp_323[2]];
    signal tmp_325[3] <== CMul()(tmp_789, evals[109]);
    signal tmp_326[3] <== [tmp_324[0] + tmp_325[0], tmp_324[1] + tmp_325[1], tmp_324[2] + tmp_325[2]];
    signal tmp_327[3] <== CMul()(tmp_790, evals[110]);
    signal tmp_328[3] <== [tmp_326[0] + tmp_327[0], tmp_326[1] + tmp_327[1], tmp_326[2] + tmp_327[2]];
    signal tmp_329[3] <== CMul()(tmp_791, evals[111]);
    signal tmp_330[3] <== [tmp_328[0] + tmp_329[0], tmp_328[1] + tmp_329[1], tmp_328[2] + tmp_329[2]];
    signal tmp_331[3] <== CMul()(tmp_792, evals[113]);
    signal tmp_332[3] <== [tmp_330[0] + tmp_331[0], tmp_330[1] + tmp_331[1], tmp_330[2] + tmp_331[2]];
    signal tmp_333[3] <== CMul()(tmp_793, evals[114]);
    signal tmp_334[3] <== [tmp_332[0] + tmp_333[0], tmp_332[1] + tmp_333[1], tmp_332[2] + tmp_333[2]];
    signal tmp_335[3] <== CMul()(tmp_794, evals[115]);
    signal tmp_336[3] <== [tmp_334[0] + tmp_335[0], tmp_334[1] + tmp_335[1], tmp_334[2] + tmp_335[2]];
    signal tmp_337[3] <== CMul()(tmp_795, evals[116]);
    signal tmp_338[3] <== [tmp_336[0] + tmp_337[0], tmp_336[1] + tmp_337[1], tmp_336[2] + tmp_337[2]];
    signal tmp_339[3] <== CMul()(tmp_796, evals[117]);
    signal tmp_340[3] <== [tmp_338[0] + tmp_339[0], tmp_338[1] + tmp_339[1], tmp_338[2] + tmp_339[2]];
    signal tmp_341[3] <== CMul()(tmp_797, evals[118]);
    signal tmp_342[3] <== [tmp_340[0] + tmp_341[0], tmp_340[1] + tmp_341[1], tmp_340[2] + tmp_341[2]];
    signal tmp_343[3] <== CMul()(tmp_798, evals[119]);
    signal tmp_344[3] <== [tmp_342[0] + tmp_343[0], tmp_342[1] + tmp_343[1], tmp_342[2] + tmp_343[2]];
    signal tmp_345[3] <== CMul()(tmp_799, evals[120]);
    signal tmp_346[3] <== [tmp_344[0] + tmp_345[0], tmp_344[1] + tmp_345[1], tmp_344[2] + tmp_345[2]];
    signal tmp_347[3] <== CMul()(tmp_800, evals[121]);
    signal tmp_348[3] <== [tmp_346[0] + tmp_347[0], tmp_346[1] + tmp_347[1], tmp_346[2] + tmp_347[2]];
    signal tmp_349[3] <== CMul()(tmp_801, evals[122]);
    tmp_804 <== [tmp_348[0] + tmp_349[0], tmp_348[1] + tmp_349[1], tmp_348[2] + tmp_349[2]];
    signal tmp_351[3] <== CMul()(tmp_734, evals[123]);
    signal tmp_352[3] <== CMul()(tmp_735, evals[124]);
    signal tmp_353[3] <== [tmp_351[0] + tmp_352[0], tmp_351[1] + tmp_352[1], tmp_351[2] + tmp_352[2]];
    signal tmp_354[3] <== CMul()(tmp_743, evals[125]);
    signal tmp_355[3] <== [tmp_353[0] + tmp_354[0], tmp_353[1] + tmp_354[1], tmp_353[2] + tmp_354[2]];
    signal tmp_356[3] <== CMul()(tmp_759, evals[126]);
    signal tmp_357[3] <== [tmp_355[0] + tmp_356[0], tmp_355[1] + tmp_356[1], tmp_355[2] + tmp_356[2]];
    signal tmp_358[3] <== CMul()(tmp_760, evals[127]);
    signal tmp_359[3] <== [tmp_357[0] + tmp_358[0], tmp_357[1] + tmp_358[1], tmp_357[2] + tmp_358[2]];
    signal tmp_360[3] <== CMul()(tmp_761, evals[128]);
    signal tmp_361[3] <== [tmp_359[0] + tmp_360[0], tmp_359[1] + tmp_360[1], tmp_359[2] + tmp_360[2]];
    signal tmp_362[3] <== CMul()(tmp_762, evals[129]);
    signal tmp_363[3] <== [tmp_361[0] + tmp_362[0], tmp_361[1] + tmp_362[1], tmp_361[2] + tmp_362[2]];
    signal tmp_364[3] <== CMul()(tmp_763, evals[130]);
    signal tmp_365[3] <== [tmp_363[0] + tmp_364[0], tmp_363[1] + tmp_364[1], tmp_363[2] + tmp_364[2]];
    signal tmp_366[3] <== CMul()(tmp_764, evals[131]);
    signal tmp_367[3] <== [tmp_365[0] + tmp_366[0], tmp_365[1] + tmp_366[1], tmp_365[2] + tmp_366[2]];
    signal tmp_368[3] <== CMul()(tmp_765, evals[132]);
    signal tmp_369[3] <== [tmp_367[0] + tmp_368[0], tmp_367[1] + tmp_368[1], tmp_367[2] + tmp_368[2]];
    signal tmp_370[3] <== CMul()(tmp_766, evals[133]);
    signal tmp_371[3] <== [tmp_369[0] + tmp_370[0], tmp_369[1] + tmp_370[1], tmp_369[2] + tmp_370[2]];
    signal tmp_372[3] <== CMul()(tmp_767, evals[134]);
    signal tmp_373[3] <== [tmp_371[0] + tmp_372[0], tmp_371[1] + tmp_372[1], tmp_371[2] + tmp_372[2]];
    signal tmp_374[3] <== CMul()(tmp_768, evals[135]);
    signal tmp_375[3] <== [tmp_373[0] + tmp_374[0], tmp_373[1] + tmp_374[1], tmp_373[2] + tmp_374[2]];
    signal tmp_376[3] <== CMul()(tmp_769, evals[136]);
    signal tmp_377[3] <== [tmp_375[0] + tmp_376[0], tmp_375[1] + tmp_376[1], tmp_375[2] + tmp_376[2]];
    signal tmp_378[3] <== CMul()(tmp_770, evals[137]);
    signal tmp_379[3] <== [tmp_377[0] + tmp_378[0], tmp_377[1] + tmp_378[1], tmp_377[2] + tmp_378[2]];
    signal tmp_380[3] <== CMul()(tmp_771, evals[138]);
    signal tmp_381[3] <== [tmp_379[0] + tmp_380[0], tmp_379[1] + tmp_380[1], tmp_379[2] + tmp_380[2]];
    signal tmp_382[3] <== CMul()(tmp_772, evals[139]);
    signal tmp_383[3] <== [tmp_381[0] + tmp_382[0], tmp_381[1] + tmp_382[1], tmp_381[2] + tmp_382[2]];
    signal tmp_384[3] <== CMul()(tmp_773, evals[140]);
    signal tmp_385[3] <== [tmp_383[0] + tmp_384[0], tmp_383[1] + tmp_384[1], tmp_383[2] + tmp_384[2]];
    signal tmp_386[3] <== CMul()(tmp_774, evals[141]);
    signal tmp_387[3] <== [tmp_385[0] + tmp_386[0], tmp_385[1] + tmp_386[1], tmp_385[2] + tmp_386[2]];
    signal tmp_388[3] <== CMul()(tmp_775, evals[142]);
    tmp_805 <== [tmp_387[0] + tmp_388[0], tmp_387[1] + tmp_388[1], tmp_387[2] + tmp_388[2]];
    tmp_806 <== CMul()(tmp_759, evals[143]);
}

/*
    The committed openings really are the openings, at one query point.

    Given query index i (LSB-first bits), x = shift * w^i in L_0 is accumulated bit by bit in
    `xacc`, and one CInv per opening point gives the denominators 1/(x - g^s*z) that the DEEP
    combination needs (the name xDivXSubXi is inherited from the prover's row helper).
    `CalculateDeepPolChunks*` then rebuilds
        f_0(x) = sum over opening points, vf1-folded, of
                 [ sum over polynomials, vf2-folded, of (pol(x) - pol(g^s*z)) ] / (x - g^s*z)
    from this query's opened leaves - the trace/quotient/fixed/custom values of row i, which
    the caller has already checked against their Merkle roots. The query-independent terms
    arrive precomputed in `friConsts`, whose first two entries are vf1 and vf2.

    The value must match the prover's committed f_0 oracle: the caller passes the whole opened
    coset of f_0 above x, and TreeSelector picks the member x corresponds to using the query
    bits above nextStepBits. The three equalities at the end are the join between the STARK
    part of the proof and the low-degree test.
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesL0[20];
    signal input challengeOut[3];
    signal input friConsts[127][3];
    signal input evals[144][3];
    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[47];

    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    signal xacc[20];
    xacc[0] <== queriesL0[0]*(7 * roots(20)-7) + 7;
    for (var i = 1; i < 20; i++) {
        xacc[i] <== xacc[i-1] * ( queriesL0[i]*(roots(20 - i) - 1) +1);
    }

    signal xDivXSubXi[19][3];

    xDivXSubXi[0] <== CInv()([xacc[19] - 11575992183625933494 * challengeOut[0], - 11575992183625933494 * challengeOut[1], - 11575992183625933494 * challengeOut[2]]);
    xDivXSubXi[1] <== CInv()([xacc[19] - 17891969694835432085 * challengeOut[0], - 17891969694835432085 * challengeOut[1], - 17891969694835432085 * challengeOut[2]]);
    xDivXSubXi[2] <== CInv()([xacc[19] - 16287335884293908063 * challengeOut[0], - 16287335884293908063 * challengeOut[1], - 16287335884293908063 * challengeOut[2]]);
    xDivXSubXi[3] <== CInv()([xacc[19] - 11970842550450932764 * challengeOut[0], - 11970842550450932764 * challengeOut[1], - 11970842550450932764 * challengeOut[2]]);
    xDivXSubXi[4] <== CInv()([xacc[19] - 12477578612481739161 * challengeOut[0], - 12477578612481739161 * challengeOut[1], - 12477578612481739161 * challengeOut[2]]);
    xDivXSubXi[5] <== CInv()([xacc[19] - 8201046199761910586 * challengeOut[0], - 8201046199761910586 * challengeOut[1], - 8201046199761910586 * challengeOut[2]]);
    xDivXSubXi[6] <== CInv()([xacc[19] - 17209560082449230913 * challengeOut[0], - 17209560082449230913 * challengeOut[1], - 17209560082449230913 * challengeOut[2]]);
    xDivXSubXi[7] <== CInv()([xacc[19] - 8992281176983196029 * challengeOut[0], - 8992281176983196029 * challengeOut[1], - 8992281176983196029 * challengeOut[2]]);
    xDivXSubXi[8] <== CInv()([xacc[19] - 2841727033376697931 * challengeOut[0], - 2841727033376697931 * challengeOut[1], - 2841727033376697931 * challengeOut[2]]);
    xDivXSubXi[9] <== CInv()([xacc[19] - 16403767884237967044 * challengeOut[0], - 16403767884237967044 * challengeOut[1], - 16403767884237967044 * challengeOut[2]]);
    xDivXSubXi[10] <== CInv()([xacc[19] - 25110052138364961 * challengeOut[0], - 25110052138364961 * challengeOut[1], - 25110052138364961 * challengeOut[2]]);
    xDivXSubXi[11] <== CInv()([xacc[19] - 14361513308298917157 * challengeOut[0], - 14361513308298917157 * challengeOut[1], - 14361513308298917157 * challengeOut[2]]);
    xDivXSubXi[12] <== CInv()([xacc[19] - 8892493137794983311 * challengeOut[0], - 8892493137794983311 * challengeOut[1], - 8892493137794983311 * challengeOut[2]]);
    xDivXSubXi[13] <== CInv()([xacc[19] - 13583859940902678234 * challengeOut[0], - 13583859940902678234 * challengeOut[1], - 13583859940902678234 * challengeOut[2]]);
    xDivXSubXi[14] <== CInv()([xacc[19] - 9071788333329385449 * challengeOut[0], - 9071788333329385449 * challengeOut[1], - 9071788333329385449 * challengeOut[2]]);
    xDivXSubXi[15] <== CInv()([xacc[19] - 15139302138664925958 * challengeOut[0], - 15139302138664925958 * challengeOut[1], - 15139302138664925958 * challengeOut[2]]);
    xDivXSubXi[16] <== CInv()([xacc[19] - 1 * challengeOut[0], - 1 * challengeOut[1], - 1 * challengeOut[2]]);
    xDivXSubXi[17] <== CInv()([xacc[19] - 5718075921287398682 * challengeOut[0], - 5718075921287398682 * challengeOut[1], - 5718075921287398682 * challengeOut[2]]);
    xDivXSubXi[18] <== CInv()([xacc[19] - 8167150655112846419 * challengeOut[0], - 8167150655112846419 * challengeOut[1], - 8167150655112846419 * challengeOut[2]]);

    // The query-independent terms, computed once in the StarkVerifier.
    signal challengesDeep[2][3] <== [friConsts[0], friConsts[1]];
    signal tmp_664[3] <== friConsts[2];
    signal tmp_665[3] <== friConsts[3];
    signal tmp_666[3] <== friConsts[4];
    signal tmp_667[3] <== friConsts[5];
    signal tmp_668[3] <== friConsts[6];
    signal tmp_669[3] <== friConsts[7];
    signal tmp_670[3] <== friConsts[8];
    signal tmp_671[3] <== friConsts[9];
    signal tmp_672[3] <== friConsts[10];
    signal tmp_673[3] <== friConsts[11];
    signal tmp_674[3] <== friConsts[12];
    signal tmp_675[3] <== friConsts[13];
    signal tmp_676[3] <== friConsts[14];
    signal tmp_677[3] <== friConsts[15];
    signal tmp_678[3] <== friConsts[16];
    signal tmp_679[3] <== friConsts[17];
    signal tmp_680[3] <== friConsts[18];
    signal tmp_698[3] <== friConsts[19];
    signal tmp_699[3] <== friConsts[20];
    signal tmp_700[3] <== friConsts[21];
    signal tmp_701[3] <== friConsts[22];
    signal tmp_702[3] <== friConsts[23];
    signal tmp_703[3] <== friConsts[24];
    signal tmp_704[3] <== friConsts[25];
    signal tmp_705[3] <== friConsts[26];
    signal tmp_706[3] <== friConsts[27];
    signal tmp_707[3] <== friConsts[28];
    signal tmp_708[3] <== friConsts[29];
    signal tmp_709[3] <== friConsts[30];
    signal tmp_710[3] <== friConsts[31];
    signal tmp_711[3] <== friConsts[32];
    signal tmp_712[3] <== friConsts[33];
    signal tmp_713[3] <== friConsts[34];
    signal tmp_714[3] <== friConsts[35];
    signal tmp_715[3] <== friConsts[36];
    signal tmp_716[3] <== friConsts[37];
    signal tmp_717[3] <== friConsts[38];
    signal tmp_718[3] <== friConsts[39];
    signal tmp_719[3] <== friConsts[40];
    signal tmp_720[3] <== friConsts[41];
    signal tmp_721[3] <== friConsts[42];
    signal tmp_722[3] <== friConsts[43];
    signal tmp_723[3] <== friConsts[44];
    signal tmp_724[3] <== friConsts[45];
    signal tmp_725[3] <== friConsts[46];
    signal tmp_726[3] <== friConsts[47];
    signal tmp_727[3] <== friConsts[48];
    signal tmp_728[3] <== friConsts[49];
    signal tmp_729[3] <== friConsts[50];
    signal tmp_730[3] <== friConsts[51];
    signal tmp_731[3] <== friConsts[52];
    signal tmp_732[3] <== friConsts[53];
    signal tmp_733[3] <== friConsts[54];
    signal tmp_734[3] <== friConsts[55];
    signal tmp_735[3] <== friConsts[56];
    signal tmp_736[3] <== friConsts[57];
    signal tmp_737[3] <== friConsts[58];
    signal tmp_738[3] <== friConsts[59];
    signal tmp_739[3] <== friConsts[60];
    signal tmp_740[3] <== friConsts[61];
    signal tmp_741[3] <== friConsts[62];
    signal tmp_742[3] <== friConsts[63];
    signal tmp_743[3] <== friConsts[64];
    signal tmp_744[3] <== friConsts[65];
    signal tmp_745[3] <== friConsts[66];
    signal tmp_746[3] <== friConsts[67];
    signal tmp_747[3] <== friConsts[68];
    signal tmp_748[3] <== friConsts[69];
    signal tmp_749[3] <== friConsts[70];
    signal tmp_750[3] <== friConsts[71];
    signal tmp_751[3] <== friConsts[72];
    signal tmp_752[3] <== friConsts[73];
    signal tmp_753[3] <== friConsts[74];
    signal tmp_754[3] <== friConsts[75];
    signal tmp_755[3] <== friConsts[76];
    signal tmp_756[3] <== friConsts[77];
    signal tmp_757[3] <== friConsts[78];
    signal tmp_758[3] <== friConsts[79];
    signal tmp_759[3] <== friConsts[80];
    signal tmp_760[3] <== friConsts[81];
    signal tmp_761[3] <== friConsts[82];
    signal tmp_762[3] <== friConsts[83];
    signal tmp_763[3] <== friConsts[84];
    signal tmp_764[3] <== friConsts[85];
    signal tmp_765[3] <== friConsts[86];
    signal tmp_766[3] <== friConsts[87];
    signal tmp_767[3] <== friConsts[88];
    signal tmp_768[3] <== friConsts[89];
    signal tmp_769[3] <== friConsts[90];
    signal tmp_770[3] <== friConsts[91];
    signal tmp_771[3] <== friConsts[92];
    signal tmp_772[3] <== friConsts[93];
    signal tmp_773[3] <== friConsts[94];
    signal tmp_774[3] <== friConsts[95];
    signal tmp_775[3] <== friConsts[96];
    signal tmp_776[3] <== friConsts[97];
    signal tmp_777[3] <== friConsts[98];
    signal tmp_778[3] <== friConsts[99];
    signal tmp_779[3] <== friConsts[100];
    signal tmp_780[3] <== friConsts[101];
    signal tmp_781[3] <== friConsts[102];
    signal tmp_782[3] <== friConsts[103];
    signal tmp_783[3] <== friConsts[104];
    signal tmp_784[3] <== friConsts[105];
    signal tmp_785[3] <== friConsts[106];
    signal tmp_786[3] <== friConsts[107];
    signal tmp_787[3] <== friConsts[108];
    signal tmp_788[3] <== friConsts[109];
    signal tmp_789[3] <== friConsts[110];
    signal tmp_790[3] <== friConsts[111];
    signal tmp_791[3] <== friConsts[112];
    signal tmp_792[3] <== friConsts[113];
    signal tmp_793[3] <== friConsts[114];
    signal tmp_794[3] <== friConsts[115];
    signal tmp_795[3] <== friConsts[116];
    signal tmp_796[3] <== friConsts[117];
    signal tmp_797[3] <== friConsts[118];
    signal tmp_798[3] <== friConsts[119];
    signal tmp_799[3] <== friConsts[120];
    signal tmp_800[3] <== friConsts[121];
    signal tmp_801[3] <== friConsts[122];
    signal tmp_803[3] <== friConsts[123];
    signal tmp_804[3] <== friConsts[124];
    signal tmp_805[3] <== friConsts[125];
    signal tmp_806[3] <== friConsts[126];

    signal tmp_807[3];
    (tmp_807) <== CalculateDeepPolChunks0_0()(challengesDeep,evals,cm1,cm2,cm3,consts,xDivXSubXi,tmp_664,tmp_665,tmp_666,tmp_667,tmp_668,tmp_669,tmp_670,tmp_671,tmp_672,tmp_673,tmp_674,tmp_675,tmp_676,tmp_677,tmp_678,tmp_679,tmp_680,tmp_698,tmp_699,tmp_700,tmp_701,tmp_702,tmp_703,tmp_704,tmp_705,tmp_706,tmp_707,tmp_708,tmp_709,tmp_710,tmp_711,tmp_712,tmp_713,tmp_714,tmp_715,tmp_716,tmp_717,tmp_718,tmp_719,tmp_720,tmp_721,tmp_722,tmp_723,tmp_724,tmp_725,tmp_726,tmp_727,tmp_728,tmp_729,tmp_730,tmp_731,tmp_732,tmp_733,tmp_734,tmp_735,tmp_736,tmp_737,tmp_738,tmp_739,tmp_740,tmp_741,tmp_742,tmp_743,tmp_744,tmp_745,tmp_746,tmp_747,tmp_748,tmp_749,tmp_750,tmp_751,tmp_752,tmp_753,tmp_754,tmp_755,tmp_756,tmp_757,tmp_758,tmp_759,tmp_760,tmp_761,tmp_762,tmp_763,tmp_764,tmp_765,tmp_766,tmp_767,tmp_768,tmp_769,tmp_770,tmp_771,tmp_772,tmp_773,tmp_774,tmp_775,tmp_776,tmp_777,tmp_778,tmp_779,tmp_780,tmp_781,tmp_782,tmp_783,tmp_784,tmp_785,tmp_786,tmp_787,tmp_788,tmp_789,tmp_790,tmp_791,tmp_792,tmp_793,tmp_794,tmp_795,tmp_796,tmp_797,tmp_798,tmp_799,tmp_800,tmp_801,tmp_803,tmp_804,tmp_805,tmp_806);

    signal queryVals[3];
    queryVals[0] <== tmp_807[0];
    queryVals[1] <== tmp_807[1];
    queryVals[2] <== tmp_807[2];

    signal {binary} s0_keys_lowValues[nextStep];
    for (var i = 0; i < nextStep; i++) {
        s0_keys_lowValues[i] <== queriesL0[i + nextStepBits];
    }

    for (var i = 0; i < nextStepBits; i++) {
        _ <== queriesL0[i];
    }
   
    signal lowValues[3] <== TreeSelector(nextStep, 3)(s1_vals, s0_keys_lowValues);

    enable * (lowValues[0] - queryVals[0]) === 0;
    enable * (lowValues[1] - queryVals[1]) === 0;
    enable * (lowValues[2] - queryVals[2]) === 0;
}

/*
    Polynomials can either have dimension 1 (if they are defined in the base field) or dimension 3 (if they are defined in the 
    extended field). In general, all initial polynomials (constants and tr) will have dim 1 and the other ones such as Z (grand product),
    Q (quotient) or h_i (plookup) will have dim 3.
    This function processes the values, which are stored in an array vals[n] and splits them in multiple signals of size 1 (vals_i) 
    or 3 (vals_i[3]) depending on its dimension.
*/
template MapValues0() {
    signal input vals1[48];
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
    cm2_0 <== [vals2[0], vals2[1], vals2[2]];
    cm2_1 <== [vals2[3], vals2[4], vals2[5]];
    cm2_2 <== [vals2[6], vals2[7], vals2[8]];
    cm2_3 <== [vals2[9], vals2[10], vals2[11]];
    cm3_0 <== [vals3[0], vals3[1], vals3[2]];
    cm3_1 <== [vals3[3], vals3[4], vals3[5]];
    cm3_2 <== [vals3[6], vals3[7], vals3[8]];
    cm3_3 <== [vals3[9], vals3[10], vals3[11]];
    cm3_4 <== [vals3[12], vals3[13], vals3[14]];
    cm3_5 <== [vals3[15], vals3[16], vals3[17]];
    cm3_6 <== [vals3[18], vals3[19], vals3[20]];
}

/*
    The degree of FRI's last oracle. `finalPol` is sent as evaluations over the last folded
    domain, so its degree bound has to be checked explicitly: one IFFT, shared by all queries,
    and every coefficient above the bound is pinned to zero.
*/
template VerifyFinalPol0() {
    ///////
    // Check Degree last pol
    ///////
    signal input finalPol[32][3];
    signal input {binary} enable;
    
    // Calculate the IFFT to get the coefficients of finalPol 
    signal lastIFFT[32][3] <== FFT(5, 3, 1)(finalPol);

    // Check that the degree of the final polynomial is bounded by the degree defined in the last step of the folding
    for (var k = 4; k < 32; k++) {
        for (var e = 0; e < 3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k = 0; k < 4; k++) {
        _ <== lastIFFT[k];
    }
}


/*
    Everything one query x in L_0 needs.

    Merkle-opens every L_0 tree at the query index (the stage trees, Q, the fixed polynomials
    and any custom commit, plus the f_0 oracle's coset), calls VerifyQuery0 to
    recompute f_0(x) from those leaves and cross-check it against the opened oracle, and then
    runs the FRI consistency check of every folding step, ending against `finalPol`.
*/
template VerifySingleQuery0() {
    signal input {binary} queriesL0[20];

    signal input challengeOut[3];
    signal input friConsts[127][3];
    signal input rFold[5][3];
    signal input evals[144][3];
    signal input {binary} enable;

    signal input s0_vals1[48];
    signal input s0_vals1_p[48][1];
    signal input s0_siblings1[8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[12];
    signal input s0_vals2_p[12][1];
    signal input s0_siblings2[8][12];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[21];
    signal input s0_vals3_p[21][1];
    signal input s0_siblings3[8][12];

    signal input s0_valsC[47];
    signal input s0_valsC_p[47][1];
    signal input s0_siblingsC[8][12];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];


    signal input s1_vals_p[8][3];
    signal input s1_siblings[7][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[8][3];
    signal input s2_siblings[5][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[8][3];
    signal input s3_siblings[4][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[8][3];
    signal input s4_siblings[2][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals_p[8][3];
    signal input s5_siblings[1][12];
    signal input s5_last_mt_levels[16][4];

    signal input finalPol[32][3];

    signal {binary} queriesL0Bits[10][2];
    for (var j = 0; j < 10; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 20) {
                queriesL0Bits[j][k] <== 0;
            } else {
                queriesL0Bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(1, 48, 4, 8, 2, 1048576)(s0_vals1_p, s0_siblings1, queriesL0Bits, s0_last_mt_levels1, enable);
    VerifyMerkleHashUntilLevel(1, 12, 4, 8, 2, 1048576)(s0_vals2_p, s0_siblings2, queriesL0Bits, s0_last_mt_levels2, enable);

    VerifyMerkleHashUntilLevel(1, 21, 4, 8, 2, 1048576)(s0_vals3_p, s0_siblings3, queriesL0Bits, s0_last_mt_levels3, enable);
    VerifyMerkleHashUntilLevel(1, 47, 4, 8, 2, 1048576)(s0_valsC_p, s0_siblingsC, queriesL0Bits, s0_last_mt_levelsC, enable);


    signal {binary} s1_keys_merkle_bits[9][2];
    for (var j = 0; j < 9; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 17) {
                s1_keys_merkle_bits[j][k] <== 0;
            } else {
                s1_keys_merkle_bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 7, 2, 131072)(s1_vals_p, s1_siblings, s1_keys_merkle_bits, s1_last_mt_levels, enable);
    signal {binary} s2_keys_merkle_bits[7][2];
    for (var j = 0; j < 7; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 14) {
                s2_keys_merkle_bits[j][k] <== 0;
            } else {
                s2_keys_merkle_bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 5, 2, 16384)(s2_vals_p, s2_siblings, s2_keys_merkle_bits, s2_last_mt_levels, enable);
    signal {binary} s3_keys_merkle_bits[6][2];
    for (var j = 0; j < 6; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 11) {
                s3_keys_merkle_bits[j][k] <== 0;
            } else {
                s3_keys_merkle_bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 4, 2, 2048)(s3_vals_p, s3_siblings, s3_keys_merkle_bits, s3_last_mt_levels, enable);
    signal {binary} s4_keys_merkle_bits[4][2];
    for (var j = 0; j < 4; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 8) {
                s4_keys_merkle_bits[j][k] <== 0;
            } else {
                s4_keys_merkle_bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 2, 2, 256)(s4_vals_p, s4_siblings, s4_keys_merkle_bits, s4_last_mt_levels, enable);
    signal {binary} s5_keys_merkle_bits[3][2];
    for (var j = 0; j < 3; j++) {
        for (var k = 0; k < 2; k++) {
            if (k + j * 2 >= 5) {
                s5_keys_merkle_bits[j][k] <== 0;
            } else {
                s5_keys_merkle_bits[j][k] <== queriesL0[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 1, 2, 32)(s5_vals_p, s5_siblings, s5_keys_merkle_bits, s5_last_mt_levels, enable);

    VerifyQuery0(20, 17)(queriesL0, challengeOut, friConsts, evals, s0_vals1, s0_vals2, s0_vals3, s0_valsC, s1_vals_p, enable);

    signal {binary} s1_queriesL0[17];
    for (var i = 0; i < 17; i++) { s1_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(20, 20, 17, 14, 2635249152773512046)(s1_queriesL0, rFold[0], s1_vals_p, s2_vals_p, enable);
    signal {binary} s2_queriesL0[14];
    for (var i = 0; i < 14; i++) { s2_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(20, 17, 14, 11, 12421013511830570338)(s2_queriesL0, rFold[1], s2_vals_p, s3_vals_p, enable);
    signal {binary} s3_queriesL0[11];
    for (var i = 0; i < 11; i++) { s3_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(20, 14, 11, 8, 11143297345130450484)(s3_queriesL0, rFold[2], s3_vals_p, s4_vals_p, enable);
    signal {binary} s4_queriesL0[8];
    for (var i = 0; i < 8; i++) { s4_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(20, 11, 8, 5, 1138102428757299658)(s4_queriesL0, rFold[3], s4_vals_p, s5_vals_p, enable);
    signal {binary} s5_queriesL0[5];
    for (var i = 0; i < 5; i++) { s5_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(20, 8, 5, 0, 140704680260498080)(s5_queriesL0, rFold[4], s5_vals_p, finalPol, enable);
}

/*
    A batch of nQueries independent VerifySingleQuery0 instances. Batching exists
    only to keep the compiler happy: one parallel component per query, or one over all
    72 of them, is far more expensive to compile than a handful of medium batches.
    Signals shared by all queries (roots, challenges, the tree's last levels sent in clear)
    are passed through unsliced.
*/
template parallel VerifyQueriesBatch0(nQueries) {
    signal input {binary} queriesL0[nQueries][20];

    signal input challengeOut[3];
    signal input friConsts[127][3];
    signal input rFold[5][3];
    signal input evals[144][3];
    signal input {binary} enable;

    signal input s0_vals1[nQueries][48];
    signal input s0_vals1_p[nQueries][48][1];
    signal input s0_siblings1[nQueries][8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[nQueries][12];
    signal input s0_vals2_p[nQueries][12][1];
    signal input s0_siblings2[nQueries][8][12];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[nQueries][21];
    signal input s0_vals3_p[nQueries][21][1];
    signal input s0_siblings3[nQueries][8][12];

    signal input s0_valsC[nQueries][47];
    signal input s0_valsC_p[nQueries][47][1];
    signal input s0_siblingsC[nQueries][8][12];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];


    signal input s1_vals_p[nQueries][8][3];
    signal input s1_siblings[nQueries][7][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[nQueries][8][3];
    signal input s2_siblings[nQueries][5][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[nQueries][8][3];
    signal input s3_siblings[nQueries][4][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[nQueries][8][3];
    signal input s4_siblings[nQueries][2][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals_p[nQueries][8][3];
    signal input s5_siblings[nQueries][1][12];
    signal input s5_last_mt_levels[16][4];

    signal input finalPol[32][3];

    for (var q = 0; q < nQueries; q++) {
        VerifySingleQuery0()(
            queriesL0[q],
            challengeOut,
            friConsts,
            rFold,
            evals,
            enable,
            s0_vals1[q],
            s0_vals1_p[q],
            s0_siblings1[q],
            s0_last_mt_levels1,
            s0_vals2[q],
            s0_vals2_p[q],
            s0_siblings2[q],
            s0_last_mt_levels2,
            s0_vals3[q],
            s0_vals3_p[q],
            s0_siblings3[q],
            s0_valsC[q],
            s0_valsC_p[q],
            s0_siblingsC[q],
            s0_last_mt_levels3,
            s0_last_mt_levelsC,
            s1_vals_p[q],
            s1_siblings[q],
            s1_last_mt_levels,
            s2_vals_p[q],
            s2_siblings[q],
            s2_last_mt_levels,
            s3_vals_p[q],
            s3_siblings[q],
            s3_last_mt_levels,
            s4_vals_p[q],
            s4_siblings[q],
            s4_last_mt_levels,
            s5_vals_p[q],
            s5_siblings[q],
            s5_last_mt_levels,
            finalPol
        );
    }
}

/*
    Verifies one STARK proof of the air 'FiboCPU_FibonacciSquare_recursive1' (Poseidon2 transcript and Merkle trees):
    replays the Fiat-Shamir transcript to derive every challenge, checks that the evaluations
    at the out-of-domain point satisfy the constraint polynomial, opens the committed trees at
    the query positions, and runs the low-degree test on the DEEP polynomial f_0 down to the
    final polynomial.

    Trace:        2^17 rows, 2 committed stage(s) plus the quotient, split in 7 polynomial(s), in stage 3
    Extension:    2^20 points (blowup 2^3)
    Commitments:  Merkle trees of arity 4, last 2 level(s) sent in clear, hashed commits
    Low-degree test: FRI
      folding factors (log2): [3, 3, 3, 3, 3]
      log2 domain sizes:      [20, 17, 14, 11, 8, 5]
      log2 degree bounds:     [17, 14, 11, 8, 5, 2] (final polynomial of degree lower than 2^2)
      queries:                72
      grinding bits:          21
*/
template StarkVerifier0() {

    signal input publics[399];

    signal input root1[4]; // Commitment to witness polynomials of stage 1
    signal input root2[4]; // Commitment to witness polynomials of stage 2
    signal input root3[4]; // Commitment to the quotient polynomial Q

    signal input rootC[4]; // Commitment to fixed polynomials

    
    signal input evals[144][3]; // Evaluations of commited polynomials at the out-of-domain sample point z and shifts gˢ·z

    // Leaf values of the merkle tree used to check all the queries
    signal input s0_vals1[72][48];
    signal input s0_vals2[72][12];
    signal input s0_vals3[72][21];
    signal input s0_valsC[72][47];


    // Merkle proofs for each of the evaluations
    signal input s0_siblings1[72][8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_siblings2[72][8][12];
    signal input s0_last_mt_levels2[16][4];
    signal input s0_siblings3[72][8][12];
    signal input s0_last_mt_levels3[16][4];
    signal input s0_siblingsC[72][8][12];
    signal input s0_last_mt_levelsC[16][4];

    // Contains the root of the DEEP polynomial and of every folded oracle.
    signal input s1_root[4]; // Root of a polynomial of degree lower than 2^17, evaluated over 2^20 points
    signal input s2_root[4]; // Root of a polynomial of degree lower than 2^14, evaluated over 2^17 points
    signal input s3_root[4]; // Root of a polynomial of degree lower than 2^11, evaluated over 2^14 points
    signal input s4_root[4]; // Root of a polynomial of degree lower than 2^8, evaluated over 2^11 points
    signal input s5_root[4]; // Root of a polynomial of degree lower than 2^5, evaluated over 2^8 points

    // For each committed oracle, the values needed to check its round's queries: a query x
    // into Fold(f_{i-1}, k_{i-1}, r^fold_{i-1}) : L_i --> 𝔽 involves querying f_{i-1} at all
    // 2^{k_{i-1}}-roots of x.
    signal input s1_vals[72][24];
    signal input s1_siblings[72][7][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals[72][24];
    signal input s2_siblings[72][5][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals[72][24];
    signal input s3_siblings[72][4][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals[72][24];
    signal input s4_siblings[72][2][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals[72][24];
    signal input s5_siblings[72][1][12];
    signal input s5_last_mt_levels[16][4];

    // The final polynomial is sent in clear as evaluations over the last folded domain.
    signal input finalPol[32][3]; // Evaluations over 2^5 points of a polynomial of degree lower than 2^2

    // The grinding nonce.
    signal input nonce;

    signal {binary} enabled;
    signal input enable;
    enable * (enable -1) === 0;
    enabled <== enable;


    signal queryVals[72][3];

    signal challengesStage2[2][3];

    signal challengeQ[3];
    signal challengeOut[3];
    signal challengesDeep[2][3];

    // r^fold_s: the random value the verifier provides once f_s is committed, under which the
    // prover folds f_s into f_{s+1} and commits that.
    signal rFold[5][3];

    // Challenges from which we derive all the queries
    signal {binary} queriesL0[72][20];

    ///////////
    // Check that the verifier's challenges are well derived from the prover messages via the Fiat-Shamir heuristic
    ///////////

    (challengesStage2,challengeQ,challengeOut,challengesDeep,rFold,queriesL0) <== Transcript0()(publics,rootC,root1,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,s5_root,finalPol, nonce, enabled);

    ///////////
    // DEEP polynomial terms that are the same for every query
    ///////////
    signal tmp_664[3];
    signal tmp_665[3];
    signal tmp_666[3];
    signal tmp_667[3];
    signal tmp_668[3];
    signal tmp_669[3];
    signal tmp_670[3];
    signal tmp_671[3];
    signal tmp_672[3];
    signal tmp_673[3];
    signal tmp_674[3];
    signal tmp_675[3];
    signal tmp_676[3];
    signal tmp_677[3];
    signal tmp_678[3];
    signal tmp_679[3];
    signal tmp_680[3];
    signal tmp_698[3];
    signal tmp_699[3];
    signal tmp_700[3];
    signal tmp_701[3];
    signal tmp_702[3];
    signal tmp_703[3];
    signal tmp_704[3];
    signal tmp_705[3];
    signal tmp_706[3];
    signal tmp_707[3];
    signal tmp_708[3];
    signal tmp_709[3];
    signal tmp_710[3];
    signal tmp_711[3];
    signal tmp_712[3];
    signal tmp_713[3];
    signal tmp_714[3];
    signal tmp_715[3];
    signal tmp_716[3];
    signal tmp_717[3];
    signal tmp_718[3];
    signal tmp_719[3];
    signal tmp_720[3];
    signal tmp_721[3];
    signal tmp_722[3];
    signal tmp_723[3];
    signal tmp_724[3];
    signal tmp_725[3];
    signal tmp_726[3];
    signal tmp_727[3];
    signal tmp_728[3];
    signal tmp_729[3];
    signal tmp_730[3];
    signal tmp_731[3];
    signal tmp_732[3];
    signal tmp_733[3];
    signal tmp_734[3];
    signal tmp_735[3];
    signal tmp_736[3];
    signal tmp_737[3];
    signal tmp_738[3];
    signal tmp_739[3];
    signal tmp_740[3];
    signal tmp_741[3];
    signal tmp_742[3];
    signal tmp_743[3];
    signal tmp_744[3];
    signal tmp_745[3];
    signal tmp_746[3];
    signal tmp_747[3];
    signal tmp_748[3];
    signal tmp_749[3];
    signal tmp_750[3];
    signal tmp_751[3];
    signal tmp_752[3];
    signal tmp_753[3];
    signal tmp_754[3];
    signal tmp_755[3];
    signal tmp_756[3];
    signal tmp_757[3];
    signal tmp_758[3];
    signal tmp_759[3];
    signal tmp_760[3];
    signal tmp_761[3];
    signal tmp_762[3];
    signal tmp_763[3];
    signal tmp_764[3];
    signal tmp_765[3];
    signal tmp_766[3];
    signal tmp_767[3];
    signal tmp_768[3];
    signal tmp_769[3];
    signal tmp_770[3];
    signal tmp_771[3];
    signal tmp_772[3];
    signal tmp_773[3];
    signal tmp_774[3];
    signal tmp_775[3];
    signal tmp_776[3];
    signal tmp_777[3];
    signal tmp_778[3];
    signal tmp_779[3];
    signal tmp_780[3];
    signal tmp_781[3];
    signal tmp_782[3];
    signal tmp_783[3];
    signal tmp_784[3];
    signal tmp_785[3];
    signal tmp_786[3];
    signal tmp_787[3];
    signal tmp_788[3];
    signal tmp_789[3];
    signal tmp_790[3];
    signal tmp_791[3];
    signal tmp_792[3];
    signal tmp_793[3];
    signal tmp_794[3];
    signal tmp_795[3];
    signal tmp_796[3];
    signal tmp_797[3];
    signal tmp_798[3];
    signal tmp_799[3];
    signal tmp_800[3];
    signal tmp_801[3];
    signal tmp_803[3];
    signal tmp_804[3];
    signal tmp_805[3];
    signal tmp_806[3];
    (tmp_664,tmp_665,tmp_666,tmp_667,tmp_668,tmp_669,tmp_670,tmp_671,tmp_672,tmp_673,tmp_674,tmp_675,tmp_676,tmp_677,tmp_678,tmp_679,tmp_680,tmp_698,tmp_699,tmp_700,tmp_701,tmp_702,tmp_703,tmp_704,tmp_705,tmp_706,tmp_707,tmp_708,tmp_709,tmp_710,tmp_711,tmp_712,tmp_713,tmp_714,tmp_715,tmp_716,tmp_717,tmp_718,tmp_719,tmp_720,tmp_721,tmp_722,tmp_723,tmp_724,tmp_725,tmp_726,tmp_727,tmp_728,tmp_729,tmp_730,tmp_731,tmp_732,tmp_733,tmp_734,tmp_735,tmp_736,tmp_737,tmp_738,tmp_739,tmp_740,tmp_741,tmp_742,tmp_743,tmp_744,tmp_745,tmp_746,tmp_747,tmp_748,tmp_749,tmp_750,tmp_751,tmp_752,tmp_753,tmp_754,tmp_755,tmp_756,tmp_757,tmp_758,tmp_759,tmp_760,tmp_761,tmp_762,tmp_763,tmp_764,tmp_765,tmp_766,tmp_767,tmp_768,tmp_769,tmp_770,tmp_771,tmp_772,tmp_773,tmp_774,tmp_775,tmp_776,tmp_777,tmp_778,tmp_779,tmp_780,tmp_781,tmp_782,tmp_783,tmp_784,tmp_785,tmp_786,tmp_787,tmp_788,tmp_789,tmp_790,tmp_791,tmp_792,tmp_793,tmp_794,tmp_795,tmp_796,tmp_797,tmp_798,tmp_799,tmp_800,tmp_801,tmp_803,tmp_804,tmp_805,tmp_806) <== CalculateFRIConstantsChunks0()(challengesDeep,evals);

    signal friConsts[127][3];
    friConsts[0] <== challengesDeep[0];
    friConsts[1] <== challengesDeep[1];
    friConsts[2] <== tmp_664;
    friConsts[3] <== tmp_665;
    friConsts[4] <== tmp_666;
    friConsts[5] <== tmp_667;
    friConsts[6] <== tmp_668;
    friConsts[7] <== tmp_669;
    friConsts[8] <== tmp_670;
    friConsts[9] <== tmp_671;
    friConsts[10] <== tmp_672;
    friConsts[11] <== tmp_673;
    friConsts[12] <== tmp_674;
    friConsts[13] <== tmp_675;
    friConsts[14] <== tmp_676;
    friConsts[15] <== tmp_677;
    friConsts[16] <== tmp_678;
    friConsts[17] <== tmp_679;
    friConsts[18] <== tmp_680;
    friConsts[19] <== tmp_698;
    friConsts[20] <== tmp_699;
    friConsts[21] <== tmp_700;
    friConsts[22] <== tmp_701;
    friConsts[23] <== tmp_702;
    friConsts[24] <== tmp_703;
    friConsts[25] <== tmp_704;
    friConsts[26] <== tmp_705;
    friConsts[27] <== tmp_706;
    friConsts[28] <== tmp_707;
    friConsts[29] <== tmp_708;
    friConsts[30] <== tmp_709;
    friConsts[31] <== tmp_710;
    friConsts[32] <== tmp_711;
    friConsts[33] <== tmp_712;
    friConsts[34] <== tmp_713;
    friConsts[35] <== tmp_714;
    friConsts[36] <== tmp_715;
    friConsts[37] <== tmp_716;
    friConsts[38] <== tmp_717;
    friConsts[39] <== tmp_718;
    friConsts[40] <== tmp_719;
    friConsts[41] <== tmp_720;
    friConsts[42] <== tmp_721;
    friConsts[43] <== tmp_722;
    friConsts[44] <== tmp_723;
    friConsts[45] <== tmp_724;
    friConsts[46] <== tmp_725;
    friConsts[47] <== tmp_726;
    friConsts[48] <== tmp_727;
    friConsts[49] <== tmp_728;
    friConsts[50] <== tmp_729;
    friConsts[51] <== tmp_730;
    friConsts[52] <== tmp_731;
    friConsts[53] <== tmp_732;
    friConsts[54] <== tmp_733;
    friConsts[55] <== tmp_734;
    friConsts[56] <== tmp_735;
    friConsts[57] <== tmp_736;
    friConsts[58] <== tmp_737;
    friConsts[59] <== tmp_738;
    friConsts[60] <== tmp_739;
    friConsts[61] <== tmp_740;
    friConsts[62] <== tmp_741;
    friConsts[63] <== tmp_742;
    friConsts[64] <== tmp_743;
    friConsts[65] <== tmp_744;
    friConsts[66] <== tmp_745;
    friConsts[67] <== tmp_746;
    friConsts[68] <== tmp_747;
    friConsts[69] <== tmp_748;
    friConsts[70] <== tmp_749;
    friConsts[71] <== tmp_750;
    friConsts[72] <== tmp_751;
    friConsts[73] <== tmp_752;
    friConsts[74] <== tmp_753;
    friConsts[75] <== tmp_754;
    friConsts[76] <== tmp_755;
    friConsts[77] <== tmp_756;
    friConsts[78] <== tmp_757;
    friConsts[79] <== tmp_758;
    friConsts[80] <== tmp_759;
    friConsts[81] <== tmp_760;
    friConsts[82] <== tmp_761;
    friConsts[83] <== tmp_762;
    friConsts[84] <== tmp_763;
    friConsts[85] <== tmp_764;
    friConsts[86] <== tmp_765;
    friConsts[87] <== tmp_766;
    friConsts[88] <== tmp_767;
    friConsts[89] <== tmp_768;
    friConsts[90] <== tmp_769;
    friConsts[91] <== tmp_770;
    friConsts[92] <== tmp_771;
    friConsts[93] <== tmp_772;
    friConsts[94] <== tmp_773;
    friConsts[95] <== tmp_774;
    friConsts[96] <== tmp_775;
    friConsts[97] <== tmp_776;
    friConsts[98] <== tmp_777;
    friConsts[99] <== tmp_778;
    friConsts[100] <== tmp_779;
    friConsts[101] <== tmp_780;
    friConsts[102] <== tmp_781;
    friConsts[103] <== tmp_782;
    friConsts[104] <== tmp_783;
    friConsts[105] <== tmp_784;
    friConsts[106] <== tmp_785;
    friConsts[107] <== tmp_786;
    friConsts[108] <== tmp_787;
    friConsts[109] <== tmp_788;
    friConsts[110] <== tmp_789;
    friConsts[111] <== tmp_790;
    friConsts[112] <== tmp_791;
    friConsts[113] <== tmp_792;
    friConsts[114] <== tmp_793;
    friConsts[115] <== tmp_794;
    friConsts[116] <== tmp_795;
    friConsts[117] <== tmp_796;
    friConsts[118] <== tmp_797;
    friConsts[119] <== tmp_798;
    friConsts[120] <== tmp_799;
    friConsts[121] <== tmp_800;
    friConsts[122] <== tmp_801;
    friConsts[123] <== tmp_803;
    friConsts[124] <== tmp_804;
    friConsts[125] <== tmp_805;
    friConsts[126] <== tmp_806;

    ///////////
    // Check constraints are consistent with Q polynomial at the out-of-domain points
    ///////////

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeOut, evals, publics, enabled);

    ///////////
    // Reshape the opened leaves for the Merkle templates
    ///////////

    // VerifyMerkleHash* takes a leaf as values[elementsInLinear][eSize]: eSize = 1 for the L_0
    // trees, whose leaves are base-field rows, and 3 for the low-degree-test oracles, whose
    // leaves are extension-field cosets. The proof wire carries neither shape: an L_0 leaf is
    // the flat row s0_vals*[q][i], and an opened coset is s{i}_vals[q][c*3+e], its 2^k members
    // with their limbs interleaved. The `_p` arrays are the same signals re-indexed — a
    // trailing length-1 axis for the rows, [2^k][3] for the cosets. They exist only because an
    // anonymous component call needs each argument to already have the parameter's shape.
    var s0_vals1_p[72][48][1];
    var s0_vals2_p[72][12][1];
    var s0_vals3_p[72][21][1];
    var s0_valsC_p[72][47][1];
    var s0_vals_p[72][1][3]; 
    var s1_vals_p[72][8][3]; 
    var s2_vals_p[72][8][3]; 
    var s3_vals_p[72][8][3]; 
    var s4_vals_p[72][8][3]; 
    var s5_vals_p[72][8][3]; 

    for (var q = 0; q < 72; q++) {
        for (var i = 0; i < 48; i++) {
            s0_vals1_p[q][i][0] = s0_vals1[q][i];
        }
        for (var i = 0; i < 12; i++) {
            s0_vals2_p[q][i][0] = s0_vals2[q][i];
        }
        for (var i = 0; i < 21; i++) {
            s0_vals3_p[q][i][0] = s0_vals3[q][i];
        }
        for (var i = 0; i < 47; i++) {
            s0_valsC_p[q][i][0] = s0_valsC[q][i];
        }

        // Preprocess vals for each folded polynomial
        for (var e = 0; e < 3; e++) {
            for (var c = 0; c < 8; c++) {
                s1_vals_p[q][c][e] = s1_vals[q][c*3+e];
            }
            for (var c = 0; c < 8; c++) {
                s2_vals_p[q][c][e] = s2_vals[q][c*3+e];
            }
            for (var c = 0; c < 8; c++) {
                s3_vals_p[q][c][e] = s3_vals[q][c*3+e];
            }
            for (var c = 0; c < 8; c++) {
                s4_vals_p[q][c][e] = s4_vals[q][c*3+e];
            }
            for (var c = 0; c < 8; c++) {
                s5_vals_p[q][c][e] = s5_vals[q][c*3+e];
            }
        }
    }

    ///////////
    // Per-query checks, in batches of QUERIES_BATCH_SIZE: each VerifySingleQuery0 opens
    // every L_0 tree at its query, cross-checks the DEEP value against f_0 and
    // runs the FRI folding steps
    ///////////

    // Batch-size parameters — change QUERIES_BATCH_SIZE and recompile to test different batch sizes.
    var N_QUERIES = 72;
    var QUERIES_BATCH_SIZE = 9;
    var N_FULL_BATCHES = N_QUERIES \ QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE = N_QUERIES - N_FULL_BATCHES * QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE_SAFE = LAST_BATCH_SIZE > 0 ? LAST_BATCH_SIZE : 1;

    // Work buffers for full batches
    signal {binary} batch_work_queriesL0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][20];

    signal batch_work_s0_vals1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][48];
    signal batch_work_s0_vals1_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][48][1];
    signal batch_work_s0_siblings1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];
    signal batch_work_s0_vals2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][12];
    signal batch_work_s0_vals2_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][12][1];
    signal batch_work_s0_siblings2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];

    signal batch_work_s0_vals3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][21];
    signal batch_work_s0_vals3_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][21][1];
    signal batch_work_s0_siblings3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];
    signal batch_work_s0_valsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][47];
    signal batch_work_s0_valsC_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][47][1];
    signal batch_work_s0_siblingsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];


    signal batch_work_s1_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s1_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][7][12];
    signal batch_work_s2_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s2_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5][12];
    signal batch_work_s3_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s3_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][4][12];
    signal batch_work_s4_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s4_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][12];
    signal batch_work_s5_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s5_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][1][12];

    // Process full batches
    for (var b = 0; b < N_FULL_BATCHES; b++) {
        var batchStart = b * QUERIES_BATCH_SIZE;

        // Fill work buffers for batch b
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 20; i++) {
                batch_work_queriesL0[b][q][i] <== queriesL0[batchStart + q][i];
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 48; i++) {
                batch_work_s0_vals1[b][q][i] <== s0_vals1[batchStart + q][i];
                batch_work_s0_vals1_p[b][q][i][0] <== s0_vals1_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 8; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s0_siblings1[b][q][j][k] <== s0_siblings1[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 12; i++) {
                batch_work_s0_vals2[b][q][i] <== s0_vals2[batchStart + q][i];
                batch_work_s0_vals2_p[b][q][i][0] <== s0_vals2_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 8; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s0_siblings2[b][q][j][k] <== s0_siblings2[batchStart + q][j][k];
                }
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 21; i++) {
                batch_work_s0_vals3[b][q][i] <== s0_vals3[batchStart + q][i];
                batch_work_s0_vals3_p[b][q][i][0] <== s0_vals3_p[batchStart + q][i][0];
            }
            for (var i = 0; i < 47; i++) {
                batch_work_s0_valsC[b][q][i] <== s0_valsC[batchStart + q][i];
                batch_work_s0_valsC_p[b][q][i][0] <== s0_valsC_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 8; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s0_siblings3[b][q][j][k] <== s0_siblings3[batchStart + q][j][k];
                    batch_work_s0_siblingsC[b][q][j][k] <== s0_siblingsC[batchStart + q][j][k];
                }
            }
        }


        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s1_vals_p[b][q][c][e] <== s1_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 7; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s1_siblings[b][q][j][k] <== s1_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s2_vals_p[b][q][c][e] <== s2_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 5; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s2_siblings[b][q][j][k] <== s2_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s3_vals_p[b][q][c][e] <== s3_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s3_siblings[b][q][j][k] <== s3_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s4_vals_p[b][q][c][e] <== s4_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 2; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s4_siblings[b][q][j][k] <== s4_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s5_vals_p[b][q][c][e] <== s5_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 1; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s5_siblings[b][q][j][k] <== s5_siblings[batchStart + q][j][k];
                }
            }
        }

        // Call batch verifier
        VerifyQueriesBatch0(QUERIES_BATCH_SIZE)(
            batch_work_queriesL0[b],
            challengeOut,
            friConsts,
            rFold,
            evals,
            enabled,
            batch_work_s0_vals1[b],
            batch_work_s0_vals1_p[b],
            batch_work_s0_siblings1[b],
            s0_last_mt_levels1,
            batch_work_s0_vals2[b],
            batch_work_s0_vals2_p[b],
            batch_work_s0_siblings2[b],
            s0_last_mt_levels2,
            batch_work_s0_vals3[b],
            batch_work_s0_vals3_p[b],
            batch_work_s0_siblings3[b],
            batch_work_s0_valsC[b],
            batch_work_s0_valsC_p[b],
            batch_work_s0_siblingsC[b],
            s0_last_mt_levels3,
            s0_last_mt_levelsC,
            batch_work_s1_vals_p[b],
            batch_work_s1_siblings[b],
            s1_last_mt_levels,
            batch_work_s2_vals_p[b],
            batch_work_s2_siblings[b],
            s2_last_mt_levels,
            batch_work_s3_vals_p[b],
            batch_work_s3_siblings[b],
            s3_last_mt_levels,
            batch_work_s4_vals_p[b],
            batch_work_s4_siblings[b],
            s4_last_mt_levels,
            batch_work_s5_vals_p[b],
            batch_work_s5_siblings[b],
            s5_last_mt_levels,
            finalPol
        );
    }

    // Remainder batch
    signal {binary} remainder_queriesL0[LAST_BATCH_SIZE_SAFE][20];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 20; i++) {
            remainder_queriesL0[q][i] <== queriesL0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
        }
    }

    signal remainder_s0_vals1[LAST_BATCH_SIZE_SAFE][48];
    signal remainder_s0_vals1_p[LAST_BATCH_SIZE_SAFE][48][1];
    signal remainder_s0_siblings1[LAST_BATCH_SIZE_SAFE][8][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 48; i++) {
            remainder_s0_vals1[q][i] <== s0_vals1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals1_p[q][i][0] <== s0_vals1_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 8; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings1[q][j][k] <== s0_siblings1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s0_vals2[LAST_BATCH_SIZE_SAFE][12];
    signal remainder_s0_vals2_p[LAST_BATCH_SIZE_SAFE][12][1];
    signal remainder_s0_siblings2[LAST_BATCH_SIZE_SAFE][8][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 12; i++) {
            remainder_s0_vals2[q][i] <== s0_vals2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals2_p[q][i][0] <== s0_vals2_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 8; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings2[q][j][k] <== s0_siblings2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s0_vals3[LAST_BATCH_SIZE_SAFE][21];
    signal remainder_s0_vals3_p[LAST_BATCH_SIZE_SAFE][21][1];
    signal remainder_s0_siblings3[LAST_BATCH_SIZE_SAFE][8][12];
    signal remainder_s0_valsC[LAST_BATCH_SIZE_SAFE][47];
    signal remainder_s0_valsC_p[LAST_BATCH_SIZE_SAFE][47][1];
    signal remainder_s0_siblingsC[LAST_BATCH_SIZE_SAFE][8][12];

    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 21; i++) {
            remainder_s0_vals3[q][i] <== s0_vals3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals3_p[q][i][0] <== s0_vals3_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var i = 0; i < 47; i++) {
            remainder_s0_valsC[q][i] <== s0_valsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_valsC_p[q][i][0] <== s0_valsC_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 8; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings3[q][j][k] <== s0_siblings3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
                remainder_s0_siblingsC[q][j][k] <== s0_siblingsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }


    signal remainder_s1_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s1_siblings[LAST_BATCH_SIZE_SAFE][7][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s1_vals_p[q][c][e] <== s1_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 7; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s1_siblings[q][j][k] <== s1_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s2_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s2_siblings[LAST_BATCH_SIZE_SAFE][5][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s2_vals_p[q][c][e] <== s2_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 5; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s2_siblings[q][j][k] <== s2_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s3_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s3_siblings[LAST_BATCH_SIZE_SAFE][4][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s3_vals_p[q][c][e] <== s3_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 4; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s3_siblings[q][j][k] <== s3_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s4_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s4_siblings[LAST_BATCH_SIZE_SAFE][2][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s4_vals_p[q][c][e] <== s4_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s4_siblings[q][j][k] <== s4_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s5_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s5_siblings[LAST_BATCH_SIZE_SAFE][1][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s5_vals_p[q][c][e] <== s5_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 1; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s5_siblings[q][j][k] <== s5_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    if (LAST_BATCH_SIZE > 0) {
        VerifyQueriesBatch0(LAST_BATCH_SIZE)(
            remainder_queriesL0,
            challengeOut,
            friConsts,
            rFold,
            evals,
            enabled,
            remainder_s0_vals1,
            remainder_s0_vals1_p,
            remainder_s0_siblings1,
            s0_last_mt_levels1,
            remainder_s0_vals2,
            remainder_s0_vals2_p,
            remainder_s0_siblings2,
            s0_last_mt_levels2,
            remainder_s0_vals3,
            remainder_s0_vals3_p,
            remainder_s0_siblings3,
            remainder_s0_valsC,
            remainder_s0_valsC_p,
            remainder_s0_siblingsC,
            s0_last_mt_levels3,
            s0_last_mt_levelsC,
            remainder_s1_vals_p,
            remainder_s1_siblings,
            s1_last_mt_levels,
            remainder_s2_vals_p,
            remainder_s2_siblings,
            s2_last_mt_levels,
            remainder_s3_vals_p,
            remainder_s3_siblings,
            s3_last_mt_levels,
            remainder_s4_vals_p,
            remainder_s4_siblings,
            s4_last_mt_levels,
            remainder_s5_vals_p,
            remainder_s5_siblings,
            s5_last_mt_levels,
            finalPol
        );
    }


    ///////////
    // Verify Merkle roots for optimized last levels
    ///////////

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
    // Check the degree of the final polynomial
    ///////////

    VerifyFinalPol0()(finalPol, enabled);
}

