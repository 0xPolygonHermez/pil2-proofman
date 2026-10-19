pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "hash/poseidon1/pow.circom";
include "tree/treeselector8.circom";
include "hash/poseidon1/merklehash.circom";

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


    signal transcriptHash_friQueries_0[16] <== Poseidon(16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],nonce,0,0,0,0,0,0,0,0], [0,0,0,0]);
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

    signal transcriptHash_friQueries_1[16] <== Poseidon(16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_0[0],transcriptHash_friQueries_0[1],transcriptHash_friQueries_0[2],transcriptHash_friQueries_0[3]]);
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
    signal input evals[152][3]; 
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



    signal transcriptHash_publics_0[16] <== Poseidon(16)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7],publics[8],publics[9],publics[10],publics[11]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_0[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_1[16] <== Poseidon(16)([publics[12],publics[13],publics[14],publics[15],publics[16],publics[17],publics[18],publics[19],publics[20],publics[21],publics[22],publics[23]], [transcriptHash_publics_0[0],transcriptHash_publics_0[1],transcriptHash_publics_0[2],transcriptHash_publics_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_1[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_2[16] <== Poseidon(16)([publics[24],publics[25],publics[26],publics[27],publics[28],publics[29],publics[30],publics[31],publics[32],publics[33],publics[34],publics[35]], [transcriptHash_publics_1[0],transcriptHash_publics_1[1],transcriptHash_publics_1[2],transcriptHash_publics_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_2[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_3[16] <== Poseidon(16)([publics[36],publics[37],publics[38],publics[39],publics[40],publics[41],publics[42],publics[43],publics[44],publics[45],publics[46],publics[47]], [transcriptHash_publics_2[0],transcriptHash_publics_2[1],transcriptHash_publics_2[2],transcriptHash_publics_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_3[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_4[16] <== Poseidon(16)([publics[48],publics[49],publics[50],publics[51],publics[52],publics[53],publics[54],publics[55],publics[56],publics[57],publics[58],publics[59]], [transcriptHash_publics_3[0],transcriptHash_publics_3[1],transcriptHash_publics_3[2],transcriptHash_publics_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_4[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_5[16] <== Poseidon(16)([publics[60],publics[61],publics[62],publics[63],publics[64],publics[65],publics[66],publics[67],publics[68],publics[69],publics[70],publics[71]], [transcriptHash_publics_4[0],transcriptHash_publics_4[1],transcriptHash_publics_4[2],transcriptHash_publics_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_5[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_6[16] <== Poseidon(16)([publics[72],publics[73],publics[74],publics[75],publics[76],publics[77],publics[78],publics[79],publics[80],publics[81],publics[82],publics[83]], [transcriptHash_publics_5[0],transcriptHash_publics_5[1],transcriptHash_publics_5[2],transcriptHash_publics_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_6[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_7[16] <== Poseidon(16)([publics[84],publics[85],publics[86],publics[87],publics[88],publics[89],publics[90],publics[91],publics[92],publics[93],publics[94],publics[95]], [transcriptHash_publics_6[0],transcriptHash_publics_6[1],transcriptHash_publics_6[2],transcriptHash_publics_6[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_7[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_8[16] <== Poseidon(16)([publics[96],publics[97],publics[98],publics[99],publics[100],publics[101],publics[102],publics[103],publics[104],publics[105],publics[106],publics[107]], [transcriptHash_publics_7[0],transcriptHash_publics_7[1],transcriptHash_publics_7[2],transcriptHash_publics_7[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_8[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_9[16] <== Poseidon(16)([publics[108],publics[109],publics[110],publics[111],publics[112],publics[113],publics[114],publics[115],publics[116],publics[117],publics[118],publics[119]], [transcriptHash_publics_8[0],transcriptHash_publics_8[1],transcriptHash_publics_8[2],transcriptHash_publics_8[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_9[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_10[16] <== Poseidon(16)([publics[120],publics[121],publics[122],publics[123],publics[124],publics[125],publics[126],publics[127],publics[128],publics[129],publics[130],publics[131]], [transcriptHash_publics_9[0],transcriptHash_publics_9[1],transcriptHash_publics_9[2],transcriptHash_publics_9[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_10[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_11[16] <== Poseidon(16)([publics[132],publics[133],publics[134],publics[135],publics[136],publics[137],publics[138],publics[139],publics[140],publics[141],publics[142],publics[143]], [transcriptHash_publics_10[0],transcriptHash_publics_10[1],transcriptHash_publics_10[2],transcriptHash_publics_10[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_11[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_12[16] <== Poseidon(16)([publics[144],publics[145],publics[146],publics[147],publics[148],publics[149],publics[150],publics[151],publics[152],publics[153],publics[154],publics[155]], [transcriptHash_publics_11[0],transcriptHash_publics_11[1],transcriptHash_publics_11[2],transcriptHash_publics_11[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_12[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_13[16] <== Poseidon(16)([publics[156],publics[157],publics[158],publics[159],publics[160],publics[161],publics[162],publics[163],publics[164],publics[165],publics[166],publics[167]], [transcriptHash_publics_12[0],transcriptHash_publics_12[1],transcriptHash_publics_12[2],transcriptHash_publics_12[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_13[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_14[16] <== Poseidon(16)([publics[168],publics[169],publics[170],publics[171],publics[172],publics[173],publics[174],publics[175],publics[176],publics[177],publics[178],publics[179]], [transcriptHash_publics_13[0],transcriptHash_publics_13[1],transcriptHash_publics_13[2],transcriptHash_publics_13[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_14[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_15[16] <== Poseidon(16)([publics[180],publics[181],publics[182],publics[183],publics[184],publics[185],publics[186],publics[187],publics[188],publics[189],publics[190],publics[191]], [transcriptHash_publics_14[0],transcriptHash_publics_14[1],transcriptHash_publics_14[2],transcriptHash_publics_14[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_15[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_16[16] <== Poseidon(16)([publics[192],publics[193],publics[194],publics[195],publics[196],publics[197],publics[198],publics[199],publics[200],publics[201],publics[202],publics[203]], [transcriptHash_publics_15[0],transcriptHash_publics_15[1],transcriptHash_publics_15[2],transcriptHash_publics_15[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_16[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_17[16] <== Poseidon(16)([publics[204],publics[205],publics[206],publics[207],publics[208],publics[209],publics[210],publics[211],publics[212],publics[213],publics[214],publics[215]], [transcriptHash_publics_16[0],transcriptHash_publics_16[1],transcriptHash_publics_16[2],transcriptHash_publics_16[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_17[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_18[16] <== Poseidon(16)([publics[216],publics[217],publics[218],publics[219],publics[220],publics[221],publics[222],publics[223],publics[224],publics[225],publics[226],publics[227]], [transcriptHash_publics_17[0],transcriptHash_publics_17[1],transcriptHash_publics_17[2],transcriptHash_publics_17[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_18[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_19[16] <== Poseidon(16)([publics[228],publics[229],publics[230],publics[231],publics[232],publics[233],publics[234],publics[235],publics[236],publics[237],publics[238],publics[239]], [transcriptHash_publics_18[0],transcriptHash_publics_18[1],transcriptHash_publics_18[2],transcriptHash_publics_18[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_19[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_20[16] <== Poseidon(16)([publics[240],publics[241],publics[242],publics[243],publics[244],publics[245],publics[246],publics[247],publics[248],publics[249],publics[250],publics[251]], [transcriptHash_publics_19[0],transcriptHash_publics_19[1],transcriptHash_publics_19[2],transcriptHash_publics_19[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_20[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_21[16] <== Poseidon(16)([publics[252],publics[253],publics[254],publics[255],publics[256],publics[257],publics[258],publics[259],publics[260],publics[261],publics[262],publics[263]], [transcriptHash_publics_20[0],transcriptHash_publics_20[1],transcriptHash_publics_20[2],transcriptHash_publics_20[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_21[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_22[16] <== Poseidon(16)([publics[264],publics[265],publics[266],publics[267],publics[268],publics[269],publics[270],publics[271],publics[272],publics[273],publics[274],publics[275]], [transcriptHash_publics_21[0],transcriptHash_publics_21[1],transcriptHash_publics_21[2],transcriptHash_publics_21[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_22[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_23[16] <== Poseidon(16)([publics[276],publics[277],publics[278],publics[279],publics[280],publics[281],publics[282],publics[283],publics[284],publics[285],publics[286],publics[287]], [transcriptHash_publics_22[0],transcriptHash_publics_22[1],transcriptHash_publics_22[2],transcriptHash_publics_22[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_23[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_24[16] <== Poseidon(16)([publics[288],publics[289],publics[290],publics[291],publics[292],publics[293],publics[294],publics[295],publics[296],publics[297],publics[298],publics[299]], [transcriptHash_publics_23[0],transcriptHash_publics_23[1],transcriptHash_publics_23[2],transcriptHash_publics_23[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_24[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_25[16] <== Poseidon(16)([publics[300],publics[301],publics[302],publics[303],publics[304],publics[305],publics[306],publics[307],publics[308],publics[309],publics[310],publics[311]], [transcriptHash_publics_24[0],transcriptHash_publics_24[1],transcriptHash_publics_24[2],transcriptHash_publics_24[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_25[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_26[16] <== Poseidon(16)([publics[312],publics[313],publics[314],publics[315],publics[316],publics[317],publics[318],publics[319],publics[320],publics[321],publics[322],publics[323]], [transcriptHash_publics_25[0],transcriptHash_publics_25[1],transcriptHash_publics_25[2],transcriptHash_publics_25[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_26[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_27[16] <== Poseidon(16)([publics[324],publics[325],publics[326],publics[327],publics[328],publics[329],publics[330],publics[331],publics[332],publics[333],publics[334],publics[335]], [transcriptHash_publics_26[0],transcriptHash_publics_26[1],transcriptHash_publics_26[2],transcriptHash_publics_26[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_27[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_28[16] <== Poseidon(16)([publics[336],publics[337],publics[338],publics[339],publics[340],publics[341],publics[342],publics[343],publics[344],publics[345],publics[346],publics[347]], [transcriptHash_publics_27[0],transcriptHash_publics_27[1],transcriptHash_publics_27[2],transcriptHash_publics_27[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_28[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_29[16] <== Poseidon(16)([publics[348],publics[349],publics[350],publics[351],publics[352],publics[353],publics[354],publics[355],publics[356],publics[357],publics[358],publics[359]], [transcriptHash_publics_28[0],transcriptHash_publics_28[1],transcriptHash_publics_28[2],transcriptHash_publics_28[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_29[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_30[16] <== Poseidon(16)([publics[360],publics[361],publics[362],publics[363],publics[364],publics[365],publics[366],publics[367],publics[368],publics[369],publics[370],publics[371]], [transcriptHash_publics_29[0],transcriptHash_publics_29[1],transcriptHash_publics_29[2],transcriptHash_publics_29[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_30[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_31[16] <== Poseidon(16)([publics[372],publics[373],publics[374],publics[375],publics[376],publics[377],publics[378],publics[379],publics[380],publics[381],publics[382],publics[383]], [transcriptHash_publics_30[0],transcriptHash_publics_30[1],transcriptHash_publics_30[2],transcriptHash_publics_30[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_31[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_32[16] <== Poseidon(16)([publics[384],publics[385],publics[386],publics[387],publics[388],publics[389],publics[390],publics[391],publics[392],publics[393],publics[394],publics[395]], [transcriptHash_publics_31[0],transcriptHash_publics_31[1],transcriptHash_publics_31[2],transcriptHash_publics_31[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_publics_32[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_33[16] <== Poseidon(16)([publics[396],publics[397],publics[398],0,0,0,0,0,0,0,0,0], [transcriptHash_publics_32[0],transcriptHash_publics_32[1],transcriptHash_publics_32[2],transcriptHash_publics_32[3]]);
    publicsHash <== [transcriptHash_publics_33[0], transcriptHash_publics_33[1], transcriptHash_publics_33[2], transcriptHash_publics_33[3]];

    signal transcriptHash_0[16] <== Poseidon(16)([rootC[0],rootC[1],rootC[2],rootC[3],publicsHash[0],publicsHash[1],publicsHash[2],publicsHash[3],root1[0],root1[1],root1[2],root1[3]], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for (var i = 6; i < 16; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }

    signal transcriptHash_1[16] <== Poseidon(16)([root2[0],root2[1],root2[2],root2[3],0,0,0,0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }

    signal transcriptHash_2[16] <== Poseidon(16)([root3[0],root3[1],root3[2],root3[3],0,0,0,0,0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeOut <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];

    signal transcriptHash_evals_0[16] <== Poseidon(16)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1],evals[2][2],evals[3][0],evals[3][1],evals[3][2]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_1[16] <== Poseidon(16)([evals[4][0],evals[4][1],evals[4][2],evals[5][0],evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_2[16] <== Poseidon(16)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1],evals[10][2],evals[11][0],evals[11][1],evals[11][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_3[16] <== Poseidon(16)([evals[12][0],evals[12][1],evals[12][2],evals[13][0],evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_4[16] <== Poseidon(16)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1],evals[18][2],evals[19][0],evals[19][1],evals[19][2]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_4[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_5[16] <== Poseidon(16)([evals[20][0],evals[20][1],evals[20][2],evals[21][0],evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],evals[23][0],evals[23][1],evals[23][2]], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_5[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_6[16] <== Poseidon(16)([evals[24][0],evals[24][1],evals[24][2],evals[25][0],evals[25][1],evals[25][2],evals[26][0],evals[26][1],evals[26][2],evals[27][0],evals[27][1],evals[27][2]], [transcriptHash_evals_5[0],transcriptHash_evals_5[1],transcriptHash_evals_5[2],transcriptHash_evals_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_6[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_7[16] <== Poseidon(16)([evals[28][0],evals[28][1],evals[28][2],evals[29][0],evals[29][1],evals[29][2],evals[30][0],evals[30][1],evals[30][2],evals[31][0],evals[31][1],evals[31][2]], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_7[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_8[16] <== Poseidon(16)([evals[32][0],evals[32][1],evals[32][2],evals[33][0],evals[33][1],evals[33][2],evals[34][0],evals[34][1],evals[34][2],evals[35][0],evals[35][1],evals[35][2]], [transcriptHash_evals_7[0],transcriptHash_evals_7[1],transcriptHash_evals_7[2],transcriptHash_evals_7[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_8[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_9[16] <== Poseidon(16)([evals[36][0],evals[36][1],evals[36][2],evals[37][0],evals[37][1],evals[37][2],evals[38][0],evals[38][1],evals[38][2],evals[39][0],evals[39][1],evals[39][2]], [transcriptHash_evals_8[0],transcriptHash_evals_8[1],transcriptHash_evals_8[2],transcriptHash_evals_8[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_9[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_10[16] <== Poseidon(16)([evals[40][0],evals[40][1],evals[40][2],evals[41][0],evals[41][1],evals[41][2],evals[42][0],evals[42][1],evals[42][2],evals[43][0],evals[43][1],evals[43][2]], [transcriptHash_evals_9[0],transcriptHash_evals_9[1],transcriptHash_evals_9[2],transcriptHash_evals_9[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_10[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_11[16] <== Poseidon(16)([evals[44][0],evals[44][1],evals[44][2],evals[45][0],evals[45][1],evals[45][2],evals[46][0],evals[46][1],evals[46][2],evals[47][0],evals[47][1],evals[47][2]], [transcriptHash_evals_10[0],transcriptHash_evals_10[1],transcriptHash_evals_10[2],transcriptHash_evals_10[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_11[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_12[16] <== Poseidon(16)([evals[48][0],evals[48][1],evals[48][2],evals[49][0],evals[49][1],evals[49][2],evals[50][0],evals[50][1],evals[50][2],evals[51][0],evals[51][1],evals[51][2]], [transcriptHash_evals_11[0],transcriptHash_evals_11[1],transcriptHash_evals_11[2],transcriptHash_evals_11[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_12[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_13[16] <== Poseidon(16)([evals[52][0],evals[52][1],evals[52][2],evals[53][0],evals[53][1],evals[53][2],evals[54][0],evals[54][1],evals[54][2],evals[55][0],evals[55][1],evals[55][2]], [transcriptHash_evals_12[0],transcriptHash_evals_12[1],transcriptHash_evals_12[2],transcriptHash_evals_12[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_13[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_14[16] <== Poseidon(16)([evals[56][0],evals[56][1],evals[56][2],evals[57][0],evals[57][1],evals[57][2],evals[58][0],evals[58][1],evals[58][2],evals[59][0],evals[59][1],evals[59][2]], [transcriptHash_evals_13[0],transcriptHash_evals_13[1],transcriptHash_evals_13[2],transcriptHash_evals_13[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_14[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_15[16] <== Poseidon(16)([evals[60][0],evals[60][1],evals[60][2],evals[61][0],evals[61][1],evals[61][2],evals[62][0],evals[62][1],evals[62][2],evals[63][0],evals[63][1],evals[63][2]], [transcriptHash_evals_14[0],transcriptHash_evals_14[1],transcriptHash_evals_14[2],transcriptHash_evals_14[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_15[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_16[16] <== Poseidon(16)([evals[64][0],evals[64][1],evals[64][2],evals[65][0],evals[65][1],evals[65][2],evals[66][0],evals[66][1],evals[66][2],evals[67][0],evals[67][1],evals[67][2]], [transcriptHash_evals_15[0],transcriptHash_evals_15[1],transcriptHash_evals_15[2],transcriptHash_evals_15[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_16[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_17[16] <== Poseidon(16)([evals[68][0],evals[68][1],evals[68][2],evals[69][0],evals[69][1],evals[69][2],evals[70][0],evals[70][1],evals[70][2],evals[71][0],evals[71][1],evals[71][2]], [transcriptHash_evals_16[0],transcriptHash_evals_16[1],transcriptHash_evals_16[2],transcriptHash_evals_16[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_17[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_18[16] <== Poseidon(16)([evals[72][0],evals[72][1],evals[72][2],evals[73][0],evals[73][1],evals[73][2],evals[74][0],evals[74][1],evals[74][2],evals[75][0],evals[75][1],evals[75][2]], [transcriptHash_evals_17[0],transcriptHash_evals_17[1],transcriptHash_evals_17[2],transcriptHash_evals_17[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_18[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_19[16] <== Poseidon(16)([evals[76][0],evals[76][1],evals[76][2],evals[77][0],evals[77][1],evals[77][2],evals[78][0],evals[78][1],evals[78][2],evals[79][0],evals[79][1],evals[79][2]], [transcriptHash_evals_18[0],transcriptHash_evals_18[1],transcriptHash_evals_18[2],transcriptHash_evals_18[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_19[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_20[16] <== Poseidon(16)([evals[80][0],evals[80][1],evals[80][2],evals[81][0],evals[81][1],evals[81][2],evals[82][0],evals[82][1],evals[82][2],evals[83][0],evals[83][1],evals[83][2]], [transcriptHash_evals_19[0],transcriptHash_evals_19[1],transcriptHash_evals_19[2],transcriptHash_evals_19[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_20[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_21[16] <== Poseidon(16)([evals[84][0],evals[84][1],evals[84][2],evals[85][0],evals[85][1],evals[85][2],evals[86][0],evals[86][1],evals[86][2],evals[87][0],evals[87][1],evals[87][2]], [transcriptHash_evals_20[0],transcriptHash_evals_20[1],transcriptHash_evals_20[2],transcriptHash_evals_20[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_21[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_22[16] <== Poseidon(16)([evals[88][0],evals[88][1],evals[88][2],evals[89][0],evals[89][1],evals[89][2],evals[90][0],evals[90][1],evals[90][2],evals[91][0],evals[91][1],evals[91][2]], [transcriptHash_evals_21[0],transcriptHash_evals_21[1],transcriptHash_evals_21[2],transcriptHash_evals_21[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_22[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_23[16] <== Poseidon(16)([evals[92][0],evals[92][1],evals[92][2],evals[93][0],evals[93][1],evals[93][2],evals[94][0],evals[94][1],evals[94][2],evals[95][0],evals[95][1],evals[95][2]], [transcriptHash_evals_22[0],transcriptHash_evals_22[1],transcriptHash_evals_22[2],transcriptHash_evals_22[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_23[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_24[16] <== Poseidon(16)([evals[96][0],evals[96][1],evals[96][2],evals[97][0],evals[97][1],evals[97][2],evals[98][0],evals[98][1],evals[98][2],evals[99][0],evals[99][1],evals[99][2]], [transcriptHash_evals_23[0],transcriptHash_evals_23[1],transcriptHash_evals_23[2],transcriptHash_evals_23[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_24[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_25[16] <== Poseidon(16)([evals[100][0],evals[100][1],evals[100][2],evals[101][0],evals[101][1],evals[101][2],evals[102][0],evals[102][1],evals[102][2],evals[103][0],evals[103][1],evals[103][2]], [transcriptHash_evals_24[0],transcriptHash_evals_24[1],transcriptHash_evals_24[2],transcriptHash_evals_24[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_25[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_26[16] <== Poseidon(16)([evals[104][0],evals[104][1],evals[104][2],evals[105][0],evals[105][1],evals[105][2],evals[106][0],evals[106][1],evals[106][2],evals[107][0],evals[107][1],evals[107][2]], [transcriptHash_evals_25[0],transcriptHash_evals_25[1],transcriptHash_evals_25[2],transcriptHash_evals_25[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_26[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_27[16] <== Poseidon(16)([evals[108][0],evals[108][1],evals[108][2],evals[109][0],evals[109][1],evals[109][2],evals[110][0],evals[110][1],evals[110][2],evals[111][0],evals[111][1],evals[111][2]], [transcriptHash_evals_26[0],transcriptHash_evals_26[1],transcriptHash_evals_26[2],transcriptHash_evals_26[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_27[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_28[16] <== Poseidon(16)([evals[112][0],evals[112][1],evals[112][2],evals[113][0],evals[113][1],evals[113][2],evals[114][0],evals[114][1],evals[114][2],evals[115][0],evals[115][1],evals[115][2]], [transcriptHash_evals_27[0],transcriptHash_evals_27[1],transcriptHash_evals_27[2],transcriptHash_evals_27[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_28[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_29[16] <== Poseidon(16)([evals[116][0],evals[116][1],evals[116][2],evals[117][0],evals[117][1],evals[117][2],evals[118][0],evals[118][1],evals[118][2],evals[119][0],evals[119][1],evals[119][2]], [transcriptHash_evals_28[0],transcriptHash_evals_28[1],transcriptHash_evals_28[2],transcriptHash_evals_28[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_29[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_30[16] <== Poseidon(16)([evals[120][0],evals[120][1],evals[120][2],evals[121][0],evals[121][1],evals[121][2],evals[122][0],evals[122][1],evals[122][2],evals[123][0],evals[123][1],evals[123][2]], [transcriptHash_evals_29[0],transcriptHash_evals_29[1],transcriptHash_evals_29[2],transcriptHash_evals_29[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_30[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_31[16] <== Poseidon(16)([evals[124][0],evals[124][1],evals[124][2],evals[125][0],evals[125][1],evals[125][2],evals[126][0],evals[126][1],evals[126][2],evals[127][0],evals[127][1],evals[127][2]], [transcriptHash_evals_30[0],transcriptHash_evals_30[1],transcriptHash_evals_30[2],transcriptHash_evals_30[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_31[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_32[16] <== Poseidon(16)([evals[128][0],evals[128][1],evals[128][2],evals[129][0],evals[129][1],evals[129][2],evals[130][0],evals[130][1],evals[130][2],evals[131][0],evals[131][1],evals[131][2]], [transcriptHash_evals_31[0],transcriptHash_evals_31[1],transcriptHash_evals_31[2],transcriptHash_evals_31[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_32[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_33[16] <== Poseidon(16)([evals[132][0],evals[132][1],evals[132][2],evals[133][0],evals[133][1],evals[133][2],evals[134][0],evals[134][1],evals[134][2],evals[135][0],evals[135][1],evals[135][2]], [transcriptHash_evals_32[0],transcriptHash_evals_32[1],transcriptHash_evals_32[2],transcriptHash_evals_32[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_33[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_34[16] <== Poseidon(16)([evals[136][0],evals[136][1],evals[136][2],evals[137][0],evals[137][1],evals[137][2],evals[138][0],evals[138][1],evals[138][2],evals[139][0],evals[139][1],evals[139][2]], [transcriptHash_evals_33[0],transcriptHash_evals_33[1],transcriptHash_evals_33[2],transcriptHash_evals_33[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_34[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_35[16] <== Poseidon(16)([evals[140][0],evals[140][1],evals[140][2],evals[141][0],evals[141][1],evals[141][2],evals[142][0],evals[142][1],evals[142][2],evals[143][0],evals[143][1],evals[143][2]], [transcriptHash_evals_34[0],transcriptHash_evals_34[1],transcriptHash_evals_34[2],transcriptHash_evals_34[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_35[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_36[16] <== Poseidon(16)([evals[144][0],evals[144][1],evals[144][2],evals[145][0],evals[145][1],evals[145][2],evals[146][0],evals[146][1],evals[146][2],evals[147][0],evals[147][1],evals[147][2]], [transcriptHash_evals_35[0],transcriptHash_evals_35[1],transcriptHash_evals_35[2],transcriptHash_evals_35[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_evals_36[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_37[16] <== Poseidon(16)([evals[148][0],evals[148][1],evals[148][2],evals[149][0],evals[149][1],evals[149][2],evals[150][0],evals[150][1],evals[150][2],evals[151][0],evals[151][1],evals[151][2]], [transcriptHash_evals_36[0],transcriptHash_evals_36[1],transcriptHash_evals_36[2],transcriptHash_evals_36[3]]);
    evalsHash <== [transcriptHash_evals_37[0], transcriptHash_evals_37[1], transcriptHash_evals_37[2], transcriptHash_evals_37[3]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }

    signal transcriptHash_3[16] <== Poseidon(16)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0,0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesDeep[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesDeep[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    for (var i = 6; i < 16; i++) {
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }

    signal transcriptHash_4[16] <== Poseidon(16)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0,0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    rFold[0] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }

    signal transcriptHash_5[16] <== Poseidon(16)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0,0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    rFold[1] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }

    signal transcriptHash_6[16] <== Poseidon(16)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0,0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    rFold[2] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }

    signal transcriptHash_7[16] <== Poseidon(16)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0,0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    rFold[3] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }

    signal transcriptHash_8[16] <== Poseidon(16)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0,0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    rFold[4] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];

    signal transcriptHash_lastPolFRI_0[16] <== Poseidon(16)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1],finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_1[16] <== Poseidon(16)([finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0],finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_2[16] <== Poseidon(16)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1],finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_3[16] <== Poseidon(16)([finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0],finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_4[16] <== Poseidon(16)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1],finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_5[16] <== Poseidon(16)([finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0],finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_6[16] <== Poseidon(16)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1],finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_7[16] <== Poseidon(16)([finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0],finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_7[0], transcriptHash_lastPolFRI_7[1], transcriptHash_lastPolFRI_7[2], transcriptHash_lastPolFRI_7[3]];
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }

    signal transcriptHash_9[16] <== Poseidon(16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
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
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];


    signal output tmp_8564[3];
    signal tmp_7566[3] <== [evals[67][0] - publics[0], evals[67][1], evals[67][2]];
    signal tmp_7567[3] <== CMul()(evals[64], tmp_7566);
    signal tmp_7568[3] <== CMul()(challengeQ, tmp_7567);
    signal tmp_7569[3] <== [evals[68][0] - publics[1], evals[68][1], evals[68][2]];
    signal tmp_7570[3] <== CMul()(evals[64], tmp_7569);
    signal tmp_7571[3] <== [tmp_7568[0] + tmp_7570[0], tmp_7568[1] + tmp_7570[1], tmp_7568[2] + tmp_7570[2]];
    signal tmp_7572[3] <== CMul()(challengeQ, tmp_7571);
    signal tmp_7573[3] <== [evals[69][0] - publics[2], evals[69][1], evals[69][2]];
    signal tmp_7574[3] <== CMul()(evals[64], tmp_7573);
    signal tmp_7575[3] <== [tmp_7572[0] + tmp_7574[0], tmp_7572[1] + tmp_7574[1], tmp_7572[2] + tmp_7574[2]];
    signal tmp_7576[3] <== CMul()(challengeQ, tmp_7575);
    signal tmp_7577[3] <== [evals[70][0] - publics[3], evals[70][1], evals[70][2]];
    signal tmp_7578[3] <== CMul()(evals[64], tmp_7577);
    signal tmp_7579[3] <== [tmp_7576[0] + tmp_7578[0], tmp_7576[1] + tmp_7578[1], tmp_7576[2] + tmp_7578[2]];
    signal tmp_7580[3] <== CMul()(challengeQ, tmp_7579);
    signal tmp_7581[3] <== [evals[71][0] - publics[4], evals[71][1], evals[71][2]];
    signal tmp_7582[3] <== CMul()(evals[64], tmp_7581);
    signal tmp_7583[3] <== [tmp_7580[0] + tmp_7582[0], tmp_7580[1] + tmp_7582[1], tmp_7580[2] + tmp_7582[2]];
    signal tmp_7584[3] <== CMul()(challengeQ, tmp_7583);
    signal tmp_7585[3] <== [evals[72][0] - publics[5], evals[72][1], evals[72][2]];
    signal tmp_7586[3] <== CMul()(evals[64], tmp_7585);
    signal tmp_7587[3] <== [tmp_7584[0] + tmp_7586[0], tmp_7584[1] + tmp_7586[1], tmp_7584[2] + tmp_7586[2]];
    signal tmp_7588[3] <== CMul()(challengeQ, tmp_7587);
    signal tmp_7589[3] <== [evals[73][0] - publics[6], evals[73][1], evals[73][2]];
    signal tmp_7590[3] <== CMul()(evals[64], tmp_7589);
    signal tmp_7591[3] <== [tmp_7588[0] + tmp_7590[0], tmp_7588[1] + tmp_7590[1], tmp_7588[2] + tmp_7590[2]];
    signal tmp_7592[3] <== CMul()(challengeQ, tmp_7591);
    signal tmp_7593[3] <== [evals[74][0] - publics[7], evals[74][1], evals[74][2]];
    signal tmp_7594[3] <== CMul()(evals[64], tmp_7593);
    signal tmp_7595[3] <== [tmp_7592[0] + tmp_7594[0], tmp_7592[1] + tmp_7594[1], tmp_7592[2] + tmp_7594[2]];
    signal tmp_7596[3] <== CMul()(challengeQ, tmp_7595);
    signal tmp_7597[3] <== [evals[75][0] - publics[8], evals[75][1], evals[75][2]];
    signal tmp_7598[3] <== CMul()(evals[64], tmp_7597);
    signal tmp_7599[3] <== [tmp_7596[0] + tmp_7598[0], tmp_7596[1] + tmp_7598[1], tmp_7596[2] + tmp_7598[2]];
    signal tmp_7600[3] <== CMul()(challengeQ, tmp_7599);
    signal tmp_7601[3] <== [evals[76][0] - publics[9], evals[76][1], evals[76][2]];
    signal tmp_7602[3] <== CMul()(evals[64], tmp_7601);
    signal tmp_7603[3] <== [tmp_7600[0] + tmp_7602[0], tmp_7600[1] + tmp_7602[1], tmp_7600[2] + tmp_7602[2]];
    signal tmp_7604[3] <== CMul()(challengeQ, tmp_7603);
    signal tmp_7605[3] <== [evals[77][0] - publics[10], evals[77][1], evals[77][2]];
    signal tmp_7606[3] <== CMul()(evals[64], tmp_7605);
    signal tmp_7607[3] <== [tmp_7604[0] + tmp_7606[0], tmp_7604[1] + tmp_7606[1], tmp_7604[2] + tmp_7606[2]];
    signal tmp_7608[3] <== CMul()(challengeQ, tmp_7607);
    signal tmp_7609[3] <== [evals[78][0] - publics[11], evals[78][1], evals[78][2]];
    signal tmp_7610[3] <== CMul()(evals[64], tmp_7609);
    signal tmp_7611[3] <== [tmp_7608[0] + tmp_7610[0], tmp_7608[1] + tmp_7610[1], tmp_7608[2] + tmp_7610[2]];
    signal tmp_7612[3] <== CMul()(challengeQ, tmp_7611);
    signal tmp_7613[3] <== [evals[79][0] - publics[12], evals[79][1], evals[79][2]];
    signal tmp_7614[3] <== CMul()(evals[64], tmp_7613);
    signal tmp_7615[3] <== [tmp_7612[0] + tmp_7614[0], tmp_7612[1] + tmp_7614[1], tmp_7612[2] + tmp_7614[2]];
    signal tmp_7616[3] <== CMul()(challengeQ, tmp_7615);
    signal tmp_7617[3] <== [evals[80][0] - publics[13], evals[80][1], evals[80][2]];
    signal tmp_7618[3] <== CMul()(evals[64], tmp_7617);
    signal tmp_7619[3] <== [tmp_7616[0] + tmp_7618[0], tmp_7616[1] + tmp_7618[1], tmp_7616[2] + tmp_7618[2]];
    signal tmp_7620[3] <== CMul()(challengeQ, tmp_7619);
    signal tmp_7621[3] <== [evals[81][0] - publics[14], evals[81][1], evals[81][2]];
    signal tmp_7622[3] <== CMul()(evals[64], tmp_7621);
    signal tmp_7623[3] <== [tmp_7620[0] + tmp_7622[0], tmp_7620[1] + tmp_7622[1], tmp_7620[2] + tmp_7622[2]];
    signal tmp_7624[3] <== CMul()(challengeQ, tmp_7623);
    signal tmp_7625[3] <== [evals[82][0] - publics[15], evals[82][1], evals[82][2]];
    signal tmp_7626[3] <== CMul()(evals[64], tmp_7625);
    signal tmp_7627[3] <== [tmp_7624[0] + tmp_7626[0], tmp_7624[1] + tmp_7626[1], tmp_7624[2] + tmp_7626[2]];
    signal tmp_7628[3] <== CMul()(challengeQ, tmp_7627);
    signal tmp_7629[3] <== [evals[83][0] - publics[16], evals[83][1], evals[83][2]];
    signal tmp_7630[3] <== CMul()(evals[64], tmp_7629);
    signal tmp_7631[3] <== [tmp_7628[0] + tmp_7630[0], tmp_7628[1] + tmp_7630[1], tmp_7628[2] + tmp_7630[2]];
    signal tmp_7632[3] <== CMul()(challengeQ, tmp_7631);
    signal tmp_7633[3] <== [evals[84][0] - publics[17], evals[84][1], evals[84][2]];
    signal tmp_7634[3] <== CMul()(evals[64], tmp_7633);
    signal tmp_7635[3] <== [tmp_7632[0] + tmp_7634[0], tmp_7632[1] + tmp_7634[1], tmp_7632[2] + tmp_7634[2]];
    signal tmp_7636[3] <== CMul()(challengeQ, tmp_7635);
    signal tmp_7637[3] <== [evals[85][0] - publics[18], evals[85][1], evals[85][2]];
    signal tmp_7638[3] <== CMul()(evals[64], tmp_7637);
    signal tmp_7639[3] <== [tmp_7636[0] + tmp_7638[0], tmp_7636[1] + tmp_7638[1], tmp_7636[2] + tmp_7638[2]];
    signal tmp_7640[3] <== CMul()(challengeQ, tmp_7639);
    signal tmp_7641[3] <== [evals[86][0] - publics[19], evals[86][1], evals[86][2]];
    signal tmp_7642[3] <== CMul()(evals[64], tmp_7641);
    signal tmp_7643[3] <== [tmp_7640[0] + tmp_7642[0], tmp_7640[1] + tmp_7642[1], tmp_7640[2] + tmp_7642[2]];
    signal tmp_7644[3] <== CMul()(challengeQ, tmp_7643);
    signal tmp_7645[3] <== [evals[87][0] - publics[20], evals[87][1], evals[87][2]];
    signal tmp_7646[3] <== CMul()(evals[64], tmp_7645);
    signal tmp_7647[3] <== [tmp_7644[0] + tmp_7646[0], tmp_7644[1] + tmp_7646[1], tmp_7644[2] + tmp_7646[2]];
    signal tmp_7648[3] <== CMul()(challengeQ, tmp_7647);
    signal tmp_7649[3] <== [evals[88][0] - publics[21], evals[88][1], evals[88][2]];
    signal tmp_7650[3] <== CMul()(evals[64], tmp_7649);
    signal tmp_7651[3] <== [tmp_7648[0] + tmp_7650[0], tmp_7648[1] + tmp_7650[1], tmp_7648[2] + tmp_7650[2]];
    signal tmp_7652[3] <== CMul()(challengeQ, tmp_7651);
    signal tmp_7653[3] <== [evals[89][0] - publics[22], evals[89][1], evals[89][2]];
    signal tmp_7654[3] <== CMul()(evals[64], tmp_7653);
    signal tmp_7655[3] <== [tmp_7652[0] + tmp_7654[0], tmp_7652[1] + tmp_7654[1], tmp_7652[2] + tmp_7654[2]];
    signal tmp_7656[3] <== CMul()(challengeQ, tmp_7655);
    signal tmp_7657[3] <== [evals[90][0] - publics[23], evals[90][1], evals[90][2]];
    signal tmp_7658[3] <== CMul()(evals[64], tmp_7657);
    signal tmp_7659[3] <== [tmp_7656[0] + tmp_7658[0], tmp_7656[1] + tmp_7658[1], tmp_7656[2] + tmp_7658[2]];
    signal tmp_7660[3] <== CMul()(challengeQ, tmp_7659);
    signal tmp_7661[3] <== [evals[67][0] - publics[24], evals[67][1], evals[67][2]];
    signal tmp_7662[3] <== CMul()(evals[19], tmp_7661);
    signal tmp_7663[3] <== [tmp_7660[0] + tmp_7662[0], tmp_7660[1] + tmp_7662[1], tmp_7660[2] + tmp_7662[2]];
    signal tmp_7664[3] <== CMul()(challengeQ, tmp_7663);
    signal tmp_7665[3] <== [evals[68][0] - publics[25], evals[68][1], evals[68][2]];
    signal tmp_7666[3] <== CMul()(evals[19], tmp_7665);
    signal tmp_7667[3] <== [tmp_7664[0] + tmp_7666[0], tmp_7664[1] + tmp_7666[1], tmp_7664[2] + tmp_7666[2]];
    signal tmp_7668[3] <== CMul()(challengeQ, tmp_7667);
    signal tmp_7669[3] <== [evals[69][0] - publics[26], evals[69][1], evals[69][2]];
    signal tmp_7670[3] <== CMul()(evals[19], tmp_7669);
    signal tmp_7671[3] <== [tmp_7668[0] + tmp_7670[0], tmp_7668[1] + tmp_7670[1], tmp_7668[2] + tmp_7670[2]];
    signal tmp_7672[3] <== CMul()(challengeQ, tmp_7671);
    signal tmp_7673[3] <== [evals[70][0] - publics[27], evals[70][1], evals[70][2]];
    signal tmp_7674[3] <== CMul()(evals[19], tmp_7673);
    signal tmp_7675[3] <== [tmp_7672[0] + tmp_7674[0], tmp_7672[1] + tmp_7674[1], tmp_7672[2] + tmp_7674[2]];
    signal tmp_7676[3] <== CMul()(challengeQ, tmp_7675);
    signal tmp_7677[3] <== [evals[71][0] - publics[28], evals[71][1], evals[71][2]];
    signal tmp_7678[3] <== CMul()(evals[19], tmp_7677);
    signal tmp_7679[3] <== [tmp_7676[0] + tmp_7678[0], tmp_7676[1] + tmp_7678[1], tmp_7676[2] + tmp_7678[2]];
    signal tmp_7680[3] <== CMul()(challengeQ, tmp_7679);
    signal tmp_7681[3] <== [evals[72][0] - publics[29], evals[72][1], evals[72][2]];
    signal tmp_7682[3] <== CMul()(evals[19], tmp_7681);
    signal tmp_7683[3] <== [tmp_7680[0] + tmp_7682[0], tmp_7680[1] + tmp_7682[1], tmp_7680[2] + tmp_7682[2]];
    signal tmp_7684[3] <== CMul()(challengeQ, tmp_7683);
    signal tmp_7685[3] <== [evals[73][0] - publics[30], evals[73][1], evals[73][2]];
    signal tmp_7686[3] <== CMul()(evals[19], tmp_7685);
    signal tmp_7687[3] <== [tmp_7684[0] + tmp_7686[0], tmp_7684[1] + tmp_7686[1], tmp_7684[2] + tmp_7686[2]];
    signal tmp_7688[3] <== CMul()(challengeQ, tmp_7687);
    signal tmp_7689[3] <== [evals[74][0] - publics[31], evals[74][1], evals[74][2]];
    signal tmp_7690[3] <== CMul()(evals[19], tmp_7689);
    signal tmp_7691[3] <== [tmp_7688[0] + tmp_7690[0], tmp_7688[1] + tmp_7690[1], tmp_7688[2] + tmp_7690[2]];
    signal tmp_7692[3] <== CMul()(challengeQ, tmp_7691);
    signal tmp_7693[3] <== [evals[75][0] - publics[32], evals[75][1], evals[75][2]];
    signal tmp_7694[3] <== CMul()(evals[19], tmp_7693);
    signal tmp_7695[3] <== [tmp_7692[0] + tmp_7694[0], tmp_7692[1] + tmp_7694[1], tmp_7692[2] + tmp_7694[2]];
    signal tmp_7696[3] <== CMul()(challengeQ, tmp_7695);
    signal tmp_7697[3] <== [evals[76][0] - publics[33], evals[76][1], evals[76][2]];
    signal tmp_7698[3] <== CMul()(evals[19], tmp_7697);
    signal tmp_7699[3] <== [tmp_7696[0] + tmp_7698[0], tmp_7696[1] + tmp_7698[1], tmp_7696[2] + tmp_7698[2]];
    signal tmp_7700[3] <== CMul()(challengeQ, tmp_7699);
    signal tmp_7701[3] <== [evals[77][0] - publics[34], evals[77][1], evals[77][2]];
    signal tmp_7702[3] <== CMul()(evals[19], tmp_7701);
    signal tmp_7703[3] <== [tmp_7700[0] + tmp_7702[0], tmp_7700[1] + tmp_7702[1], tmp_7700[2] + tmp_7702[2]];
    signal tmp_7704[3] <== CMul()(challengeQ, tmp_7703);
    signal tmp_7705[3] <== [evals[78][0] - publics[35], evals[78][1], evals[78][2]];
    signal tmp_7706[3] <== CMul()(evals[19], tmp_7705);
    signal tmp_7707[3] <== [tmp_7704[0] + tmp_7706[0], tmp_7704[1] + tmp_7706[1], tmp_7704[2] + tmp_7706[2]];
    signal tmp_7708[3] <== CMul()(challengeQ, tmp_7707);
    signal tmp_7709[3] <== [evals[79][0] - publics[36], evals[79][1], evals[79][2]];
    signal tmp_7710[3] <== CMul()(evals[19], tmp_7709);
    signal tmp_7711[3] <== [tmp_7708[0] + tmp_7710[0], tmp_7708[1] + tmp_7710[1], tmp_7708[2] + tmp_7710[2]];
    signal tmp_7712[3] <== CMul()(challengeQ, tmp_7711);
    signal tmp_7713[3] <== [evals[80][0] - publics[37], evals[80][1], evals[80][2]];
    signal tmp_7714[3] <== CMul()(evals[19], tmp_7713);
    signal tmp_7715[3] <== [tmp_7712[0] + tmp_7714[0], tmp_7712[1] + tmp_7714[1], tmp_7712[2] + tmp_7714[2]];
    signal tmp_7716[3] <== CMul()(challengeQ, tmp_7715);
    signal tmp_7717[3] <== [evals[81][0] - publics[38], evals[81][1], evals[81][2]];
    signal tmp_7718[3] <== CMul()(evals[19], tmp_7717);
    signal tmp_7719[3] <== [tmp_7716[0] + tmp_7718[0], tmp_7716[1] + tmp_7718[1], tmp_7716[2] + tmp_7718[2]];
    signal tmp_7720[3] <== CMul()(challengeQ, tmp_7719);
    signal tmp_7721[3] <== [evals[82][0] - publics[39], evals[82][1], evals[82][2]];
    signal tmp_7722[3] <== CMul()(evals[19], tmp_7721);
    signal tmp_7723[3] <== [tmp_7720[0] + tmp_7722[0], tmp_7720[1] + tmp_7722[1], tmp_7720[2] + tmp_7722[2]];
    signal tmp_7724[3] <== CMul()(challengeQ, tmp_7723);
    signal tmp_7725[3] <== [evals[83][0] - publics[40], evals[83][1], evals[83][2]];
    signal tmp_7726[3] <== CMul()(evals[19], tmp_7725);
    signal tmp_7727[3] <== [tmp_7724[0] + tmp_7726[0], tmp_7724[1] + tmp_7726[1], tmp_7724[2] + tmp_7726[2]];
    signal tmp_7728[3] <== CMul()(challengeQ, tmp_7727);
    signal tmp_7729[3] <== [evals[84][0] - publics[41], evals[84][1], evals[84][2]];
    signal tmp_7730[3] <== CMul()(evals[19], tmp_7729);
    signal tmp_7731[3] <== [tmp_7728[0] + tmp_7730[0], tmp_7728[1] + tmp_7730[1], tmp_7728[2] + tmp_7730[2]];
    signal tmp_7732[3] <== CMul()(challengeQ, tmp_7731);
    signal tmp_7733[3] <== [evals[85][0] - publics[42], evals[85][1], evals[85][2]];
    signal tmp_7734[3] <== CMul()(evals[19], tmp_7733);
    signal tmp_7735[3] <== [tmp_7732[0] + tmp_7734[0], tmp_7732[1] + tmp_7734[1], tmp_7732[2] + tmp_7734[2]];
    signal tmp_7736[3] <== CMul()(challengeQ, tmp_7735);
    signal tmp_7737[3] <== [evals[86][0] - publics[43], evals[86][1], evals[86][2]];
    signal tmp_7738[3] <== CMul()(evals[19], tmp_7737);
    signal tmp_7739[3] <== [tmp_7736[0] + tmp_7738[0], tmp_7736[1] + tmp_7738[1], tmp_7736[2] + tmp_7738[2]];
    signal tmp_7740[3] <== CMul()(challengeQ, tmp_7739);
    signal tmp_7741[3] <== [evals[87][0] - publics[44], evals[87][1], evals[87][2]];
    signal tmp_7742[3] <== CMul()(evals[19], tmp_7741);
    signal tmp_7743[3] <== [tmp_7740[0] + tmp_7742[0], tmp_7740[1] + tmp_7742[1], tmp_7740[2] + tmp_7742[2]];
    signal tmp_7744[3] <== CMul()(challengeQ, tmp_7743);
    signal tmp_7745[3] <== [evals[88][0] - publics[45], evals[88][1], evals[88][2]];
    signal tmp_7746[3] <== CMul()(evals[19], tmp_7745);
    signal tmp_7747[3] <== [tmp_7744[0] + tmp_7746[0], tmp_7744[1] + tmp_7746[1], tmp_7744[2] + tmp_7746[2]];
    signal tmp_7748[3] <== CMul()(challengeQ, tmp_7747);
    signal tmp_7749[3] <== [evals[89][0] - publics[46], evals[89][1], evals[89][2]];
    signal tmp_7750[3] <== CMul()(evals[19], tmp_7749);
    signal tmp_7751[3] <== [tmp_7748[0] + tmp_7750[0], tmp_7748[1] + tmp_7750[1], tmp_7748[2] + tmp_7750[2]];
    signal tmp_7752[3] <== CMul()(challengeQ, tmp_7751);
    signal tmp_7753[3] <== [evals[90][0] - publics[47], evals[90][1], evals[90][2]];
    signal tmp_7754[3] <== CMul()(evals[19], tmp_7753);
    signal tmp_7755[3] <== [tmp_7752[0] + tmp_7754[0], tmp_7752[1] + tmp_7754[1], tmp_7752[2] + tmp_7754[2]];
    signal tmp_7756[3] <== CMul()(challengeQ, tmp_7755);
    signal tmp_7757[3] <== [evals[67][0] - publics[48], evals[67][1], evals[67][2]];
    signal tmp_7758[3] <== CMul()(evals[16], tmp_7757);
    signal tmp_7759[3] <== [tmp_7756[0] + tmp_7758[0], tmp_7756[1] + tmp_7758[1], tmp_7756[2] + tmp_7758[2]];
    signal tmp_7760[3] <== CMul()(challengeQ, tmp_7759);
    signal tmp_7761[3] <== [evals[68][0] - publics[49], evals[68][1], evals[68][2]];
    signal tmp_7762[3] <== CMul()(evals[16], tmp_7761);
    signal tmp_7763[3] <== [tmp_7760[0] + tmp_7762[0], tmp_7760[1] + tmp_7762[1], tmp_7760[2] + tmp_7762[2]];
    signal tmp_7764[3] <== CMul()(challengeQ, tmp_7763);
    signal tmp_7765[3] <== [evals[69][0] - publics[50], evals[69][1], evals[69][2]];
    signal tmp_7766[3] <== CMul()(evals[16], tmp_7765);
    signal tmp_7767[3] <== [tmp_7764[0] + tmp_7766[0], tmp_7764[1] + tmp_7766[1], tmp_7764[2] + tmp_7766[2]];
    signal tmp_7768[3] <== CMul()(challengeQ, tmp_7767);
    signal tmp_7769[3] <== [evals[70][0] - publics[51], evals[70][1], evals[70][2]];
    signal tmp_7770[3] <== CMul()(evals[16], tmp_7769);
    signal tmp_7771[3] <== [tmp_7768[0] + tmp_7770[0], tmp_7768[1] + tmp_7770[1], tmp_7768[2] + tmp_7770[2]];
    signal tmp_7772[3] <== CMul()(challengeQ, tmp_7771);
    signal tmp_7773[3] <== [evals[71][0] - publics[52], evals[71][1], evals[71][2]];
    signal tmp_7774[3] <== CMul()(evals[16], tmp_7773);
    signal tmp_7775[3] <== [tmp_7772[0] + tmp_7774[0], tmp_7772[1] + tmp_7774[1], tmp_7772[2] + tmp_7774[2]];
    signal tmp_7776[3] <== CMul()(challengeQ, tmp_7775);
    signal tmp_7777[3] <== [evals[72][0] - publics[53], evals[72][1], evals[72][2]];
    signal tmp_7778[3] <== CMul()(evals[16], tmp_7777);
    signal tmp_7779[3] <== [tmp_7776[0] + tmp_7778[0], tmp_7776[1] + tmp_7778[1], tmp_7776[2] + tmp_7778[2]];
    signal tmp_7780[3] <== CMul()(challengeQ, tmp_7779);
    signal tmp_7781[3] <== [evals[73][0] - publics[54], evals[73][1], evals[73][2]];
    signal tmp_7782[3] <== CMul()(evals[16], tmp_7781);
    signal tmp_7783[3] <== [tmp_7780[0] + tmp_7782[0], tmp_7780[1] + tmp_7782[1], tmp_7780[2] + tmp_7782[2]];
    signal tmp_7784[3] <== CMul()(challengeQ, tmp_7783);
    signal tmp_7785[3] <== [evals[74][0] - publics[55], evals[74][1], evals[74][2]];
    signal tmp_7786[3] <== CMul()(evals[16], tmp_7785);
    signal tmp_7787[3] <== [tmp_7784[0] + tmp_7786[0], tmp_7784[1] + tmp_7786[1], tmp_7784[2] + tmp_7786[2]];
    signal tmp_7788[3] <== CMul()(challengeQ, tmp_7787);
    signal tmp_7789[3] <== [evals[75][0] - publics[56], evals[75][1], evals[75][2]];
    signal tmp_7790[3] <== CMul()(evals[16], tmp_7789);
    signal tmp_7791[3] <== [tmp_7788[0] + tmp_7790[0], tmp_7788[1] + tmp_7790[1], tmp_7788[2] + tmp_7790[2]];
    signal tmp_7792[3] <== CMul()(challengeQ, tmp_7791);
    signal tmp_7793[3] <== [evals[76][0] - publics[57], evals[76][1], evals[76][2]];
    signal tmp_7794[3] <== CMul()(evals[16], tmp_7793);
    signal tmp_7795[3] <== [tmp_7792[0] + tmp_7794[0], tmp_7792[1] + tmp_7794[1], tmp_7792[2] + tmp_7794[2]];
    signal tmp_7796[3] <== CMul()(challengeQ, tmp_7795);
    signal tmp_7797[3] <== [evals[77][0] - publics[58], evals[77][1], evals[77][2]];
    signal tmp_7798[3] <== CMul()(evals[16], tmp_7797);
    signal tmp_7799[3] <== [tmp_7796[0] + tmp_7798[0], tmp_7796[1] + tmp_7798[1], tmp_7796[2] + tmp_7798[2]];
    signal tmp_7800[3] <== CMul()(challengeQ, tmp_7799);
    signal tmp_7801[3] <== [evals[78][0] - publics[59], evals[78][1], evals[78][2]];
    signal tmp_7802[3] <== CMul()(evals[16], tmp_7801);
    signal tmp_7803[3] <== [tmp_7800[0] + tmp_7802[0], tmp_7800[1] + tmp_7802[1], tmp_7800[2] + tmp_7802[2]];
    signal tmp_7804[3] <== CMul()(challengeQ, tmp_7803);
    signal tmp_7805[3] <== [evals[79][0] - publics[60], evals[79][1], evals[79][2]];
    signal tmp_7806[3] <== CMul()(evals[16], tmp_7805);
    signal tmp_7807[3] <== [tmp_7804[0] + tmp_7806[0], tmp_7804[1] + tmp_7806[1], tmp_7804[2] + tmp_7806[2]];
    signal tmp_7808[3] <== CMul()(challengeQ, tmp_7807);
    signal tmp_7809[3] <== [evals[80][0] - publics[61], evals[80][1], evals[80][2]];
    signal tmp_7810[3] <== CMul()(evals[16], tmp_7809);
    signal tmp_7811[3] <== [tmp_7808[0] + tmp_7810[0], tmp_7808[1] + tmp_7810[1], tmp_7808[2] + tmp_7810[2]];
    signal tmp_7812[3] <== CMul()(challengeQ, tmp_7811);
    signal tmp_7813[3] <== [evals[81][0] - publics[62], evals[81][1], evals[81][2]];
    signal tmp_7814[3] <== CMul()(evals[16], tmp_7813);
    signal tmp_7815[3] <== [tmp_7812[0] + tmp_7814[0], tmp_7812[1] + tmp_7814[1], tmp_7812[2] + tmp_7814[2]];
    signal tmp_7816[3] <== CMul()(challengeQ, tmp_7815);
    signal tmp_7817[3] <== [evals[82][0] - publics[63], evals[82][1], evals[82][2]];
    signal tmp_7818[3] <== CMul()(evals[16], tmp_7817);
    signal tmp_7819[3] <== [tmp_7816[0] + tmp_7818[0], tmp_7816[1] + tmp_7818[1], tmp_7816[2] + tmp_7818[2]];
    signal tmp_7820[3] <== CMul()(challengeQ, tmp_7819);
    signal tmp_7821[3] <== [evals[83][0] - publics[64], evals[83][1], evals[83][2]];
    signal tmp_7822[3] <== CMul()(evals[16], tmp_7821);
    signal tmp_7823[3] <== [tmp_7820[0] + tmp_7822[0], tmp_7820[1] + tmp_7822[1], tmp_7820[2] + tmp_7822[2]];
    signal tmp_7824[3] <== CMul()(challengeQ, tmp_7823);
    signal tmp_7825[3] <== [evals[84][0] - publics[65], evals[84][1], evals[84][2]];
    signal tmp_7826[3] <== CMul()(evals[16], tmp_7825);
    signal tmp_7827[3] <== [tmp_7824[0] + tmp_7826[0], tmp_7824[1] + tmp_7826[1], tmp_7824[2] + tmp_7826[2]];
    signal tmp_7828[3] <== CMul()(challengeQ, tmp_7827);
    signal tmp_7829[3] <== [evals[85][0] - publics[66], evals[85][1], evals[85][2]];
    signal tmp_7830[3] <== CMul()(evals[16], tmp_7829);
    signal tmp_7831[3] <== [tmp_7828[0] + tmp_7830[0], tmp_7828[1] + tmp_7830[1], tmp_7828[2] + tmp_7830[2]];
    signal tmp_7832[3] <== CMul()(challengeQ, tmp_7831);
    signal tmp_7833[3] <== [evals[86][0] - publics[67], evals[86][1], evals[86][2]];
    signal tmp_7834[3] <== CMul()(evals[16], tmp_7833);
    signal tmp_7835[3] <== [tmp_7832[0] + tmp_7834[0], tmp_7832[1] + tmp_7834[1], tmp_7832[2] + tmp_7834[2]];
    signal tmp_7836[3] <== CMul()(challengeQ, tmp_7835);
    signal tmp_7837[3] <== [evals[87][0] - publics[68], evals[87][1], evals[87][2]];
    signal tmp_7838[3] <== CMul()(evals[16], tmp_7837);
    signal tmp_7839[3] <== [tmp_7836[0] + tmp_7838[0], tmp_7836[1] + tmp_7838[1], tmp_7836[2] + tmp_7838[2]];
    signal tmp_7840[3] <== CMul()(challengeQ, tmp_7839);
    signal tmp_7841[3] <== [evals[88][0] - publics[69], evals[88][1], evals[88][2]];
    signal tmp_7842[3] <== CMul()(evals[16], tmp_7841);
    signal tmp_7843[3] <== [tmp_7840[0] + tmp_7842[0], tmp_7840[1] + tmp_7842[1], tmp_7840[2] + tmp_7842[2]];
    signal tmp_7844[3] <== CMul()(challengeQ, tmp_7843);
    signal tmp_7845[3] <== [evals[89][0] - publics[70], evals[89][1], evals[89][2]];
    signal tmp_7846[3] <== CMul()(evals[16], tmp_7845);
    signal tmp_7847[3] <== [tmp_7844[0] + tmp_7846[0], tmp_7844[1] + tmp_7846[1], tmp_7844[2] + tmp_7846[2]];
    signal tmp_7848[3] <== CMul()(challengeQ, tmp_7847);
    signal tmp_7849[3] <== [evals[90][0] - publics[71], evals[90][1], evals[90][2]];
    signal tmp_7850[3] <== CMul()(evals[16], tmp_7849);
    signal tmp_7851[3] <== [tmp_7848[0] + tmp_7850[0], tmp_7848[1] + tmp_7850[1], tmp_7848[2] + tmp_7850[2]];
    signal tmp_7852[3] <== CMul()(challengeQ, tmp_7851);
    signal tmp_7853[3] <== [evals[67][0] - publics[72], evals[67][1], evals[67][2]];
    signal tmp_7854[3] <== CMul()(evals[13], tmp_7853);
    signal tmp_7855[3] <== [tmp_7852[0] + tmp_7854[0], tmp_7852[1] + tmp_7854[1], tmp_7852[2] + tmp_7854[2]];
    signal tmp_7856[3] <== CMul()(challengeQ, tmp_7855);
    signal tmp_7857[3] <== [evals[68][0] - publics[73], evals[68][1], evals[68][2]];
    signal tmp_7858[3] <== CMul()(evals[13], tmp_7857);
    signal tmp_7859[3] <== [tmp_7856[0] + tmp_7858[0], tmp_7856[1] + tmp_7858[1], tmp_7856[2] + tmp_7858[2]];
    signal tmp_7860[3] <== CMul()(challengeQ, tmp_7859);
    signal tmp_7861[3] <== [evals[69][0] - publics[74], evals[69][1], evals[69][2]];
    signal tmp_7862[3] <== CMul()(evals[13], tmp_7861);
    signal tmp_7863[3] <== [tmp_7860[0] + tmp_7862[0], tmp_7860[1] + tmp_7862[1], tmp_7860[2] + tmp_7862[2]];
    signal tmp_7864[3] <== CMul()(challengeQ, tmp_7863);
    signal tmp_7865[3] <== [evals[70][0] - publics[75], evals[70][1], evals[70][2]];
    signal tmp_7866[3] <== CMul()(evals[13], tmp_7865);
    signal tmp_7867[3] <== [tmp_7864[0] + tmp_7866[0], tmp_7864[1] + tmp_7866[1], tmp_7864[2] + tmp_7866[2]];
    signal tmp_7868[3] <== CMul()(challengeQ, tmp_7867);
    signal tmp_7869[3] <== [evals[71][0] - publics[76], evals[71][1], evals[71][2]];
    signal tmp_7870[3] <== CMul()(evals[13], tmp_7869);
    signal tmp_7871[3] <== [tmp_7868[0] + tmp_7870[0], tmp_7868[1] + tmp_7870[1], tmp_7868[2] + tmp_7870[2]];
    signal tmp_7872[3] <== CMul()(challengeQ, tmp_7871);
    signal tmp_7873[3] <== [evals[72][0] - publics[77], evals[72][1], evals[72][2]];
    signal tmp_7874[3] <== CMul()(evals[13], tmp_7873);
    signal tmp_7875[3] <== [tmp_7872[0] + tmp_7874[0], tmp_7872[1] + tmp_7874[1], tmp_7872[2] + tmp_7874[2]];
    signal tmp_7876[3] <== CMul()(challengeQ, tmp_7875);
    signal tmp_7877[3] <== [evals[73][0] - publics[78], evals[73][1], evals[73][2]];
    signal tmp_7878[3] <== CMul()(evals[13], tmp_7877);
    signal tmp_7879[3] <== [tmp_7876[0] + tmp_7878[0], tmp_7876[1] + tmp_7878[1], tmp_7876[2] + tmp_7878[2]];
    signal tmp_7880[3] <== CMul()(challengeQ, tmp_7879);
    signal tmp_7881[3] <== [evals[74][0] - publics[79], evals[74][1], evals[74][2]];
    signal tmp_7882[3] <== CMul()(evals[13], tmp_7881);
    signal tmp_7883[3] <== [tmp_7880[0] + tmp_7882[0], tmp_7880[1] + tmp_7882[1], tmp_7880[2] + tmp_7882[2]];
    signal tmp_7884[3] <== CMul()(challengeQ, tmp_7883);
    signal tmp_7885[3] <== [evals[75][0] - publics[80], evals[75][1], evals[75][2]];
    signal tmp_7886[3] <== CMul()(evals[13], tmp_7885);
    signal tmp_7887[3] <== [tmp_7884[0] + tmp_7886[0], tmp_7884[1] + tmp_7886[1], tmp_7884[2] + tmp_7886[2]];
    signal tmp_7888[3] <== CMul()(challengeQ, tmp_7887);
    signal tmp_7889[3] <== [evals[76][0] - publics[81], evals[76][1], evals[76][2]];
    signal tmp_7890[3] <== CMul()(evals[13], tmp_7889);
    signal tmp_7891[3] <== [tmp_7888[0] + tmp_7890[0], tmp_7888[1] + tmp_7890[1], tmp_7888[2] + tmp_7890[2]];
    signal tmp_7892[3] <== CMul()(challengeQ, tmp_7891);
    signal tmp_7893[3] <== [evals[77][0] - publics[82], evals[77][1], evals[77][2]];
    signal tmp_7894[3] <== CMul()(evals[13], tmp_7893);
    signal tmp_7895[3] <== [tmp_7892[0] + tmp_7894[0], tmp_7892[1] + tmp_7894[1], tmp_7892[2] + tmp_7894[2]];
    signal tmp_7896[3] <== CMul()(challengeQ, tmp_7895);
    signal tmp_7897[3] <== [evals[78][0] - publics[83], evals[78][1], evals[78][2]];
    signal tmp_7898[3] <== CMul()(evals[13], tmp_7897);
    signal tmp_7899[3] <== [tmp_7896[0] + tmp_7898[0], tmp_7896[1] + tmp_7898[1], tmp_7896[2] + tmp_7898[2]];
    signal tmp_7900[3] <== CMul()(challengeQ, tmp_7899);
    signal tmp_7901[3] <== [evals[79][0] - publics[84], evals[79][1], evals[79][2]];
    signal tmp_7902[3] <== CMul()(evals[13], tmp_7901);
    signal tmp_7903[3] <== [tmp_7900[0] + tmp_7902[0], tmp_7900[1] + tmp_7902[1], tmp_7900[2] + tmp_7902[2]];
    signal tmp_7904[3] <== CMul()(challengeQ, tmp_7903);
    signal tmp_7905[3] <== [evals[80][0] - publics[85], evals[80][1], evals[80][2]];
    signal tmp_7906[3] <== CMul()(evals[13], tmp_7905);
    signal tmp_7907[3] <== [tmp_7904[0] + tmp_7906[0], tmp_7904[1] + tmp_7906[1], tmp_7904[2] + tmp_7906[2]];
    signal tmp_7908[3] <== CMul()(challengeQ, tmp_7907);
    signal tmp_7909[3] <== [evals[81][0] - publics[86], evals[81][1], evals[81][2]];
    signal tmp_7910[3] <== CMul()(evals[13], tmp_7909);
    signal tmp_7911[3] <== [tmp_7908[0] + tmp_7910[0], tmp_7908[1] + tmp_7910[1], tmp_7908[2] + tmp_7910[2]];
    signal tmp_7912[3] <== CMul()(challengeQ, tmp_7911);
    signal tmp_7913[3] <== [evals[82][0] - publics[87], evals[82][1], evals[82][2]];
    signal tmp_7914[3] <== CMul()(evals[13], tmp_7913);
    signal tmp_7915[3] <== [tmp_7912[0] + tmp_7914[0], tmp_7912[1] + tmp_7914[1], tmp_7912[2] + tmp_7914[2]];
    signal tmp_7916[3] <== CMul()(challengeQ, tmp_7915);
    signal tmp_7917[3] <== [evals[83][0] - publics[88], evals[83][1], evals[83][2]];
    signal tmp_7918[3] <== CMul()(evals[13], tmp_7917);
    signal tmp_7919[3] <== [tmp_7916[0] + tmp_7918[0], tmp_7916[1] + tmp_7918[1], tmp_7916[2] + tmp_7918[2]];
    signal tmp_7920[3] <== CMul()(challengeQ, tmp_7919);
    signal tmp_7921[3] <== [evals[84][0] - publics[89], evals[84][1], evals[84][2]];
    signal tmp_7922[3] <== CMul()(evals[13], tmp_7921);
    signal tmp_7923[3] <== [tmp_7920[0] + tmp_7922[0], tmp_7920[1] + tmp_7922[1], tmp_7920[2] + tmp_7922[2]];
    signal tmp_7924[3] <== CMul()(challengeQ, tmp_7923);
    signal tmp_7925[3] <== [evals[85][0] - publics[90], evals[85][1], evals[85][2]];
    signal tmp_7926[3] <== CMul()(evals[13], tmp_7925);
    signal tmp_7927[3] <== [tmp_7924[0] + tmp_7926[0], tmp_7924[1] + tmp_7926[1], tmp_7924[2] + tmp_7926[2]];
    signal tmp_7928[3] <== CMul()(challengeQ, tmp_7927);
    signal tmp_7929[3] <== [evals[86][0] - publics[91], evals[86][1], evals[86][2]];
    signal tmp_7930[3] <== CMul()(evals[13], tmp_7929);
    signal tmp_7931[3] <== [tmp_7928[0] + tmp_7930[0], tmp_7928[1] + tmp_7930[1], tmp_7928[2] + tmp_7930[2]];
    signal tmp_7932[3] <== CMul()(challengeQ, tmp_7931);
    signal tmp_7933[3] <== [evals[87][0] - publics[92], evals[87][1], evals[87][2]];
    signal tmp_7934[3] <== CMul()(evals[13], tmp_7933);
    signal tmp_7935[3] <== [tmp_7932[0] + tmp_7934[0], tmp_7932[1] + tmp_7934[1], tmp_7932[2] + tmp_7934[2]];
    signal tmp_7936[3] <== CMul()(challengeQ, tmp_7935);
    signal tmp_7937[3] <== [evals[88][0] - publics[93], evals[88][1], evals[88][2]];
    signal tmp_7938[3] <== CMul()(evals[13], tmp_7937);
    signal tmp_7939[3] <== [tmp_7936[0] + tmp_7938[0], tmp_7936[1] + tmp_7938[1], tmp_7936[2] + tmp_7938[2]];
    signal tmp_7940[3] <== CMul()(challengeQ, tmp_7939);
    signal tmp_7941[3] <== [evals[89][0] - publics[94], evals[89][1], evals[89][2]];
    signal tmp_7942[3] <== CMul()(evals[13], tmp_7941);
    signal tmp_7943[3] <== [tmp_7940[0] + tmp_7942[0], tmp_7940[1] + tmp_7942[1], tmp_7940[2] + tmp_7942[2]];
    signal tmp_7944[3] <== CMul()(challengeQ, tmp_7943);
    signal tmp_7945[3] <== [evals[90][0] - publics[95], evals[90][1], evals[90][2]];
    signal tmp_7946[3] <== CMul()(evals[13], tmp_7945);
    signal tmp_7947[3] <== [tmp_7944[0] + tmp_7946[0], tmp_7944[1] + tmp_7946[1], tmp_7944[2] + tmp_7946[2]];
    signal tmp_7948[3] <== CMul()(challengeQ, tmp_7947);
    signal tmp_7949[3] <== [evals[67][0] - publics[96], evals[67][1], evals[67][2]];
    signal tmp_7950[3] <== CMul()(evals[12], tmp_7949);
    signal tmp_7951[3] <== [tmp_7948[0] + tmp_7950[0], tmp_7948[1] + tmp_7950[1], tmp_7948[2] + tmp_7950[2]];
    signal tmp_7952[3] <== CMul()(challengeQ, tmp_7951);
    signal tmp_7953[3] <== [evals[68][0] - publics[97], evals[68][1], evals[68][2]];
    signal tmp_7954[3] <== CMul()(evals[12], tmp_7953);
    signal tmp_7955[3] <== [tmp_7952[0] + tmp_7954[0], tmp_7952[1] + tmp_7954[1], tmp_7952[2] + tmp_7954[2]];
    signal tmp_7956[3] <== CMul()(challengeQ, tmp_7955);
    signal tmp_7957[3] <== [evals[69][0] - publics[98], evals[69][1], evals[69][2]];
    signal tmp_7958[3] <== CMul()(evals[12], tmp_7957);
    signal tmp_7959[3] <== [tmp_7956[0] + tmp_7958[0], tmp_7956[1] + tmp_7958[1], tmp_7956[2] + tmp_7958[2]];
    signal tmp_7960[3] <== CMul()(challengeQ, tmp_7959);
    signal tmp_7961[3] <== [evals[70][0] - publics[99], evals[70][1], evals[70][2]];
    signal tmp_7962[3] <== CMul()(evals[12], tmp_7961);
    signal tmp_7963[3] <== [tmp_7960[0] + tmp_7962[0], tmp_7960[1] + tmp_7962[1], tmp_7960[2] + tmp_7962[2]];
    signal tmp_7964[3] <== CMul()(challengeQ, tmp_7963);
    signal tmp_7965[3] <== [evals[71][0] - publics[100], evals[71][1], evals[71][2]];
    signal tmp_7966[3] <== CMul()(evals[12], tmp_7965);
    signal tmp_7967[3] <== [tmp_7964[0] + tmp_7966[0], tmp_7964[1] + tmp_7966[1], tmp_7964[2] + tmp_7966[2]];
    signal tmp_7968[3] <== CMul()(challengeQ, tmp_7967);
    signal tmp_7969[3] <== [evals[72][0] - publics[101], evals[72][1], evals[72][2]];
    signal tmp_7970[3] <== CMul()(evals[12], tmp_7969);
    signal tmp_7971[3] <== [tmp_7968[0] + tmp_7970[0], tmp_7968[1] + tmp_7970[1], tmp_7968[2] + tmp_7970[2]];
    signal tmp_7972[3] <== CMul()(challengeQ, tmp_7971);
    signal tmp_7973[3] <== [evals[73][0] - publics[102], evals[73][1], evals[73][2]];
    signal tmp_7974[3] <== CMul()(evals[12], tmp_7973);
    signal tmp_7975[3] <== [tmp_7972[0] + tmp_7974[0], tmp_7972[1] + tmp_7974[1], tmp_7972[2] + tmp_7974[2]];
    signal tmp_7976[3] <== CMul()(challengeQ, tmp_7975);
    signal tmp_7977[3] <== [evals[74][0] - publics[103], evals[74][1], evals[74][2]];
    signal tmp_7978[3] <== CMul()(evals[12], tmp_7977);
    signal tmp_7979[3] <== [tmp_7976[0] + tmp_7978[0], tmp_7976[1] + tmp_7978[1], tmp_7976[2] + tmp_7978[2]];
    signal tmp_7980[3] <== CMul()(challengeQ, tmp_7979);
    signal tmp_7981[3] <== [evals[75][0] - publics[104], evals[75][1], evals[75][2]];
    signal tmp_7982[3] <== CMul()(evals[12], tmp_7981);
    signal tmp_7983[3] <== [tmp_7980[0] + tmp_7982[0], tmp_7980[1] + tmp_7982[1], tmp_7980[2] + tmp_7982[2]];
    signal tmp_7984[3] <== CMul()(challengeQ, tmp_7983);
    signal tmp_7985[3] <== [evals[76][0] - publics[105], evals[76][1], evals[76][2]];
    signal tmp_7986[3] <== CMul()(evals[12], tmp_7985);
    signal tmp_7987[3] <== [tmp_7984[0] + tmp_7986[0], tmp_7984[1] + tmp_7986[1], tmp_7984[2] + tmp_7986[2]];
    signal tmp_7988[3] <== CMul()(challengeQ, tmp_7987);
    signal tmp_7989[3] <== [evals[77][0] - publics[106], evals[77][1], evals[77][2]];
    signal tmp_7990[3] <== CMul()(evals[12], tmp_7989);
    signal tmp_7991[3] <== [tmp_7988[0] + tmp_7990[0], tmp_7988[1] + tmp_7990[1], tmp_7988[2] + tmp_7990[2]];
    signal tmp_7992[3] <== CMul()(challengeQ, tmp_7991);
    signal tmp_7993[3] <== [evals[78][0] - publics[107], evals[78][1], evals[78][2]];
    signal tmp_7994[3] <== CMul()(evals[12], tmp_7993);
    signal tmp_7995[3] <== [tmp_7992[0] + tmp_7994[0], tmp_7992[1] + tmp_7994[1], tmp_7992[2] + tmp_7994[2]];
    signal tmp_7996[3] <== CMul()(challengeQ, tmp_7995);
    signal tmp_7997[3] <== [evals[79][0] - publics[108], evals[79][1], evals[79][2]];
    signal tmp_7998[3] <== CMul()(evals[12], tmp_7997);
    signal tmp_7999[3] <== [tmp_7996[0] + tmp_7998[0], tmp_7996[1] + tmp_7998[1], tmp_7996[2] + tmp_7998[2]];
    signal tmp_8000[3] <== CMul()(challengeQ, tmp_7999);
    signal tmp_8001[3] <== [evals[80][0] - publics[109], evals[80][1], evals[80][2]];
    signal tmp_8002[3] <== CMul()(evals[12], tmp_8001);
    signal tmp_8003[3] <== [tmp_8000[0] + tmp_8002[0], tmp_8000[1] + tmp_8002[1], tmp_8000[2] + tmp_8002[2]];
    signal tmp_8004[3] <== CMul()(challengeQ, tmp_8003);
    signal tmp_8005[3] <== [evals[81][0] - publics[110], evals[81][1], evals[81][2]];
    signal tmp_8006[3] <== CMul()(evals[12], tmp_8005);
    signal tmp_8007[3] <== [tmp_8004[0] + tmp_8006[0], tmp_8004[1] + tmp_8006[1], tmp_8004[2] + tmp_8006[2]];
    signal tmp_8008[3] <== CMul()(challengeQ, tmp_8007);
    signal tmp_8009[3] <== [evals[82][0] - publics[111], evals[82][1], evals[82][2]];
    signal tmp_8010[3] <== CMul()(evals[12], tmp_8009);
    signal tmp_8011[3] <== [tmp_8008[0] + tmp_8010[0], tmp_8008[1] + tmp_8010[1], tmp_8008[2] + tmp_8010[2]];
    signal tmp_8012[3] <== CMul()(challengeQ, tmp_8011);
    signal tmp_8013[3] <== [evals[83][0] - publics[112], evals[83][1], evals[83][2]];
    signal tmp_8014[3] <== CMul()(evals[12], tmp_8013);
    signal tmp_8015[3] <== [tmp_8012[0] + tmp_8014[0], tmp_8012[1] + tmp_8014[1], tmp_8012[2] + tmp_8014[2]];
    signal tmp_8016[3] <== CMul()(challengeQ, tmp_8015);
    signal tmp_8017[3] <== [evals[84][0] - publics[113], evals[84][1], evals[84][2]];
    signal tmp_8018[3] <== CMul()(evals[12], tmp_8017);
    signal tmp_8019[3] <== [tmp_8016[0] + tmp_8018[0], tmp_8016[1] + tmp_8018[1], tmp_8016[2] + tmp_8018[2]];
    signal tmp_8020[3] <== CMul()(challengeQ, tmp_8019);
    signal tmp_8021[3] <== [evals[85][0] - publics[114], evals[85][1], evals[85][2]];
    signal tmp_8022[3] <== CMul()(evals[12], tmp_8021);
    signal tmp_8023[3] <== [tmp_8020[0] + tmp_8022[0], tmp_8020[1] + tmp_8022[1], tmp_8020[2] + tmp_8022[2]];
    signal tmp_8024[3] <== CMul()(challengeQ, tmp_8023);
    signal tmp_8025[3] <== [evals[86][0] - publics[115], evals[86][1], evals[86][2]];
    signal tmp_8026[3] <== CMul()(evals[12], tmp_8025);
    signal tmp_8027[3] <== [tmp_8024[0] + tmp_8026[0], tmp_8024[1] + tmp_8026[1], tmp_8024[2] + tmp_8026[2]];
    signal tmp_8028[3] <== CMul()(challengeQ, tmp_8027);
    signal tmp_8029[3] <== [evals[87][0] - publics[116], evals[87][1], evals[87][2]];
    signal tmp_8030[3] <== CMul()(evals[12], tmp_8029);
    signal tmp_8031[3] <== [tmp_8028[0] + tmp_8030[0], tmp_8028[1] + tmp_8030[1], tmp_8028[2] + tmp_8030[2]];
    signal tmp_8032[3] <== CMul()(challengeQ, tmp_8031);
    signal tmp_8033[3] <== [evals[88][0] - publics[117], evals[88][1], evals[88][2]];
    signal tmp_8034[3] <== CMul()(evals[12], tmp_8033);
    signal tmp_8035[3] <== [tmp_8032[0] + tmp_8034[0], tmp_8032[1] + tmp_8034[1], tmp_8032[2] + tmp_8034[2]];
    signal tmp_8036[3] <== CMul()(challengeQ, tmp_8035);
    signal tmp_8037[3] <== [evals[89][0] - publics[118], evals[89][1], evals[89][2]];
    signal tmp_8038[3] <== CMul()(evals[12], tmp_8037);
    signal tmp_8039[3] <== [tmp_8036[0] + tmp_8038[0], tmp_8036[1] + tmp_8038[1], tmp_8036[2] + tmp_8038[2]];
    signal tmp_8040[3] <== CMul()(challengeQ, tmp_8039);
    signal tmp_8041[3] <== [evals[90][0] - publics[119], evals[90][1], evals[90][2]];
    signal tmp_8042[3] <== CMul()(evals[12], tmp_8041);
    signal tmp_8043[3] <== [tmp_8040[0] + tmp_8042[0], tmp_8040[1] + tmp_8042[1], tmp_8040[2] + tmp_8042[2]];
    signal tmp_8044[3] <== CMul()(challengeQ, tmp_8043);
    signal tmp_8045[3] <== [evals[67][0] - publics[120], evals[67][1], evals[67][2]];
    signal tmp_8046[3] <== CMul()(evals[11], tmp_8045);
    signal tmp_8047[3] <== [tmp_8044[0] + tmp_8046[0], tmp_8044[1] + tmp_8046[1], tmp_8044[2] + tmp_8046[2]];
    signal tmp_8048[3] <== CMul()(challengeQ, tmp_8047);
    signal tmp_8049[3] <== [evals[68][0] - publics[121], evals[68][1], evals[68][2]];
    signal tmp_8050[3] <== CMul()(evals[11], tmp_8049);
    signal tmp_8051[3] <== [tmp_8048[0] + tmp_8050[0], tmp_8048[1] + tmp_8050[1], tmp_8048[2] + tmp_8050[2]];
    signal tmp_8052[3] <== CMul()(challengeQ, tmp_8051);
    signal tmp_8053[3] <== [evals[69][0] - publics[122], evals[69][1], evals[69][2]];
    signal tmp_8054[3] <== CMul()(evals[11], tmp_8053);
    signal tmp_8055[3] <== [tmp_8052[0] + tmp_8054[0], tmp_8052[1] + tmp_8054[1], tmp_8052[2] + tmp_8054[2]];
    signal tmp_8056[3] <== CMul()(challengeQ, tmp_8055);
    signal tmp_8057[3] <== [evals[70][0] - publics[123], evals[70][1], evals[70][2]];
    signal tmp_8058[3] <== CMul()(evals[11], tmp_8057);
    signal tmp_8059[3] <== [tmp_8056[0] + tmp_8058[0], tmp_8056[1] + tmp_8058[1], tmp_8056[2] + tmp_8058[2]];
    signal tmp_8060[3] <== CMul()(challengeQ, tmp_8059);
    signal tmp_8061[3] <== [evals[71][0] - publics[124], evals[71][1], evals[71][2]];
    signal tmp_8062[3] <== CMul()(evals[11], tmp_8061);
    signal tmp_8063[3] <== [tmp_8060[0] + tmp_8062[0], tmp_8060[1] + tmp_8062[1], tmp_8060[2] + tmp_8062[2]];
    signal tmp_8064[3] <== CMul()(challengeQ, tmp_8063);
    signal tmp_8065[3] <== [evals[72][0] - publics[125], evals[72][1], evals[72][2]];
    signal tmp_8066[3] <== CMul()(evals[11], tmp_8065);
    signal tmp_8067[3] <== [tmp_8064[0] + tmp_8066[0], tmp_8064[1] + tmp_8066[1], tmp_8064[2] + tmp_8066[2]];
    signal tmp_8068[3] <== CMul()(challengeQ, tmp_8067);
    signal tmp_8069[3] <== [evals[73][0] - publics[126], evals[73][1], evals[73][2]];
    signal tmp_8070[3] <== CMul()(evals[11], tmp_8069);
    signal tmp_8071[3] <== [tmp_8068[0] + tmp_8070[0], tmp_8068[1] + tmp_8070[1], tmp_8068[2] + tmp_8070[2]];
    signal tmp_8072[3] <== CMul()(challengeQ, tmp_8071);
    signal tmp_8073[3] <== [evals[74][0] - publics[127], evals[74][1], evals[74][2]];
    signal tmp_8074[3] <== CMul()(evals[11], tmp_8073);
    signal tmp_8075[3] <== [tmp_8072[0] + tmp_8074[0], tmp_8072[1] + tmp_8074[1], tmp_8072[2] + tmp_8074[2]];
    signal tmp_8076[3] <== CMul()(challengeQ, tmp_8075);
    signal tmp_8077[3] <== [evals[75][0] - publics[128], evals[75][1], evals[75][2]];
    signal tmp_8078[3] <== CMul()(evals[11], tmp_8077);
    signal tmp_8079[3] <== [tmp_8076[0] + tmp_8078[0], tmp_8076[1] + tmp_8078[1], tmp_8076[2] + tmp_8078[2]];
    signal tmp_8080[3] <== CMul()(challengeQ, tmp_8079);
    signal tmp_8081[3] <== [evals[76][0] - publics[129], evals[76][1], evals[76][2]];
    signal tmp_8082[3] <== CMul()(evals[11], tmp_8081);
    signal tmp_8083[3] <== [tmp_8080[0] + tmp_8082[0], tmp_8080[1] + tmp_8082[1], tmp_8080[2] + tmp_8082[2]];
    signal tmp_8084[3] <== CMul()(challengeQ, tmp_8083);
    signal tmp_8085[3] <== [evals[77][0] - publics[130], evals[77][1], evals[77][2]];
    signal tmp_8086[3] <== CMul()(evals[11], tmp_8085);
    signal tmp_8087[3] <== [tmp_8084[0] + tmp_8086[0], tmp_8084[1] + tmp_8086[1], tmp_8084[2] + tmp_8086[2]];
    signal tmp_8088[3] <== CMul()(challengeQ, tmp_8087);
    signal tmp_8089[3] <== [evals[78][0] - publics[131], evals[78][1], evals[78][2]];
    signal tmp_8090[3] <== CMul()(evals[11], tmp_8089);
    signal tmp_8091[3] <== [tmp_8088[0] + tmp_8090[0], tmp_8088[1] + tmp_8090[1], tmp_8088[2] + tmp_8090[2]];
    signal tmp_8092[3] <== CMul()(challengeQ, tmp_8091);
    signal tmp_8093[3] <== [evals[79][0] - publics[132], evals[79][1], evals[79][2]];
    signal tmp_8094[3] <== CMul()(evals[11], tmp_8093);
    signal tmp_8095[3] <== [tmp_8092[0] + tmp_8094[0], tmp_8092[1] + tmp_8094[1], tmp_8092[2] + tmp_8094[2]];
    signal tmp_8096[3] <== CMul()(challengeQ, tmp_8095);
    signal tmp_8097[3] <== [evals[80][0] - publics[133], evals[80][1], evals[80][2]];
    signal tmp_8098[3] <== CMul()(evals[11], tmp_8097);
    signal tmp_8099[3] <== [tmp_8096[0] + tmp_8098[0], tmp_8096[1] + tmp_8098[1], tmp_8096[2] + tmp_8098[2]];
    signal tmp_8100[3] <== CMul()(challengeQ, tmp_8099);
    signal tmp_8101[3] <== [evals[81][0] - publics[134], evals[81][1], evals[81][2]];
    signal tmp_8102[3] <== CMul()(evals[11], tmp_8101);
    signal tmp_8103[3] <== [tmp_8100[0] + tmp_8102[0], tmp_8100[1] + tmp_8102[1], tmp_8100[2] + tmp_8102[2]];
    signal tmp_8104[3] <== CMul()(challengeQ, tmp_8103);
    signal tmp_8105[3] <== [evals[82][0] - publics[135], evals[82][1], evals[82][2]];
    signal tmp_8106[3] <== CMul()(evals[11], tmp_8105);
    signal tmp_8107[3] <== [tmp_8104[0] + tmp_8106[0], tmp_8104[1] + tmp_8106[1], tmp_8104[2] + tmp_8106[2]];
    signal tmp_8108[3] <== CMul()(challengeQ, tmp_8107);
    signal tmp_8109[3] <== [evals[83][0] - publics[136], evals[83][1], evals[83][2]];
    signal tmp_8110[3] <== CMul()(evals[11], tmp_8109);
    signal tmp_8111[3] <== [tmp_8108[0] + tmp_8110[0], tmp_8108[1] + tmp_8110[1], tmp_8108[2] + tmp_8110[2]];
    signal tmp_8112[3] <== CMul()(challengeQ, tmp_8111);
    signal tmp_8113[3] <== [evals[84][0] - publics[137], evals[84][1], evals[84][2]];
    signal tmp_8114[3] <== CMul()(evals[11], tmp_8113);
    signal tmp_8115[3] <== [tmp_8112[0] + tmp_8114[0], tmp_8112[1] + tmp_8114[1], tmp_8112[2] + tmp_8114[2]];
    signal tmp_8116[3] <== CMul()(challengeQ, tmp_8115);
    signal tmp_8117[3] <== [evals[85][0] - publics[138], evals[85][1], evals[85][2]];
    signal tmp_8118[3] <== CMul()(evals[11], tmp_8117);
    signal tmp_8119[3] <== [tmp_8116[0] + tmp_8118[0], tmp_8116[1] + tmp_8118[1], tmp_8116[2] + tmp_8118[2]];
    signal tmp_8120[3] <== CMul()(challengeQ, tmp_8119);
    signal tmp_8121[3] <== [evals[86][0] - publics[139], evals[86][1], evals[86][2]];
    signal tmp_8122[3] <== CMul()(evals[11], tmp_8121);
    signal tmp_8123[3] <== [tmp_8120[0] + tmp_8122[0], tmp_8120[1] + tmp_8122[1], tmp_8120[2] + tmp_8122[2]];
    signal tmp_8124[3] <== CMul()(challengeQ, tmp_8123);
    signal tmp_8125[3] <== [evals[87][0] - publics[140], evals[87][1], evals[87][2]];
    signal tmp_8126[3] <== CMul()(evals[11], tmp_8125);
    signal tmp_8127[3] <== [tmp_8124[0] + tmp_8126[0], tmp_8124[1] + tmp_8126[1], tmp_8124[2] + tmp_8126[2]];
    signal tmp_8128[3] <== CMul()(challengeQ, tmp_8127);
    signal tmp_8129[3] <== [evals[88][0] - publics[141], evals[88][1], evals[88][2]];
    signal tmp_8130[3] <== CMul()(evals[11], tmp_8129);
    signal tmp_8131[3] <== [tmp_8128[0] + tmp_8130[0], tmp_8128[1] + tmp_8130[1], tmp_8128[2] + tmp_8130[2]];
    signal tmp_8132[3] <== CMul()(challengeQ, tmp_8131);
    signal tmp_8133[3] <== [evals[89][0] - publics[142], evals[89][1], evals[89][2]];
    signal tmp_8134[3] <== CMul()(evals[11], tmp_8133);
    signal tmp_8135[3] <== [tmp_8132[0] + tmp_8134[0], tmp_8132[1] + tmp_8134[1], tmp_8132[2] + tmp_8134[2]];
    signal tmp_8136[3] <== CMul()(challengeQ, tmp_8135);
    signal tmp_8137[3] <== [evals[90][0] - publics[143], evals[90][1], evals[90][2]];
    signal tmp_8138[3] <== CMul()(evals[11], tmp_8137);
    signal tmp_8139[3] <== [tmp_8136[0] + tmp_8138[0], tmp_8136[1] + tmp_8138[1], tmp_8136[2] + tmp_8138[2]];
    signal tmp_8140[3] <== CMul()(challengeQ, tmp_8139);
    signal tmp_8141[3] <== [evals[67][0] - publics[144], evals[67][1], evals[67][2]];
    signal tmp_8142[3] <== CMul()(evals[10], tmp_8141);
    signal tmp_8143[3] <== [tmp_8140[0] + tmp_8142[0], tmp_8140[1] + tmp_8142[1], tmp_8140[2] + tmp_8142[2]];
    signal tmp_8144[3] <== CMul()(challengeQ, tmp_8143);
    signal tmp_8145[3] <== [evals[68][0] - publics[145], evals[68][1], evals[68][2]];
    signal tmp_8146[3] <== CMul()(evals[10], tmp_8145);
    signal tmp_8147[3] <== [tmp_8144[0] + tmp_8146[0], tmp_8144[1] + tmp_8146[1], tmp_8144[2] + tmp_8146[2]];
    signal tmp_8148[3] <== CMul()(challengeQ, tmp_8147);
    signal tmp_8149[3] <== [evals[69][0] - publics[146], evals[69][1], evals[69][2]];
    signal tmp_8150[3] <== CMul()(evals[10], tmp_8149);
    signal tmp_8151[3] <== [tmp_8148[0] + tmp_8150[0], tmp_8148[1] + tmp_8150[1], tmp_8148[2] + tmp_8150[2]];
    signal tmp_8152[3] <== CMul()(challengeQ, tmp_8151);
    signal tmp_8153[3] <== [evals[70][0] - publics[147], evals[70][1], evals[70][2]];
    signal tmp_8154[3] <== CMul()(evals[10], tmp_8153);
    signal tmp_8155[3] <== [tmp_8152[0] + tmp_8154[0], tmp_8152[1] + tmp_8154[1], tmp_8152[2] + tmp_8154[2]];
    signal tmp_8156[3] <== CMul()(challengeQ, tmp_8155);
    signal tmp_8157[3] <== [evals[71][0] - publics[148], evals[71][1], evals[71][2]];
    signal tmp_8158[3] <== CMul()(evals[10], tmp_8157);
    signal tmp_8159[3] <== [tmp_8156[0] + tmp_8158[0], tmp_8156[1] + tmp_8158[1], tmp_8156[2] + tmp_8158[2]];
    signal tmp_8160[3] <== CMul()(challengeQ, tmp_8159);
    signal tmp_8161[3] <== [evals[72][0] - publics[149], evals[72][1], evals[72][2]];
    signal tmp_8162[3] <== CMul()(evals[10], tmp_8161);
    signal tmp_8163[3] <== [tmp_8160[0] + tmp_8162[0], tmp_8160[1] + tmp_8162[1], tmp_8160[2] + tmp_8162[2]];
    signal tmp_8164[3] <== CMul()(challengeQ, tmp_8163);
    signal tmp_8165[3] <== [evals[73][0] - publics[150], evals[73][1], evals[73][2]];
    signal tmp_8166[3] <== CMul()(evals[10], tmp_8165);
    signal tmp_8167[3] <== [tmp_8164[0] + tmp_8166[0], tmp_8164[1] + tmp_8166[1], tmp_8164[2] + tmp_8166[2]];
    signal tmp_8168[3] <== CMul()(challengeQ, tmp_8167);
    signal tmp_8169[3] <== [evals[74][0] - publics[151], evals[74][1], evals[74][2]];
    signal tmp_8170[3] <== CMul()(evals[10], tmp_8169);
    signal tmp_8171[3] <== [tmp_8168[0] + tmp_8170[0], tmp_8168[1] + tmp_8170[1], tmp_8168[2] + tmp_8170[2]];
    signal tmp_8172[3] <== CMul()(challengeQ, tmp_8171);
    signal tmp_8173[3] <== [evals[75][0] - publics[152], evals[75][1], evals[75][2]];
    signal tmp_8174[3] <== CMul()(evals[10], tmp_8173);
    signal tmp_8175[3] <== [tmp_8172[0] + tmp_8174[0], tmp_8172[1] + tmp_8174[1], tmp_8172[2] + tmp_8174[2]];
    signal tmp_8176[3] <== CMul()(challengeQ, tmp_8175);
    signal tmp_8177[3] <== [evals[76][0] - publics[153], evals[76][1], evals[76][2]];
    signal tmp_8178[3] <== CMul()(evals[10], tmp_8177);
    signal tmp_8179[3] <== [tmp_8176[0] + tmp_8178[0], tmp_8176[1] + tmp_8178[1], tmp_8176[2] + tmp_8178[2]];
    signal tmp_8180[3] <== CMul()(challengeQ, tmp_8179);
    signal tmp_8181[3] <== [evals[77][0] - publics[154], evals[77][1], evals[77][2]];
    signal tmp_8182[3] <== CMul()(evals[10], tmp_8181);
    signal tmp_8183[3] <== [tmp_8180[0] + tmp_8182[0], tmp_8180[1] + tmp_8182[1], tmp_8180[2] + tmp_8182[2]];
    signal tmp_8184[3] <== CMul()(challengeQ, tmp_8183);
    signal tmp_8185[3] <== [evals[78][0] - publics[155], evals[78][1], evals[78][2]];
    signal tmp_8186[3] <== CMul()(evals[10], tmp_8185);
    signal tmp_8187[3] <== [tmp_8184[0] + tmp_8186[0], tmp_8184[1] + tmp_8186[1], tmp_8184[2] + tmp_8186[2]];
    signal tmp_8188[3] <== CMul()(challengeQ, tmp_8187);
    signal tmp_8189[3] <== [evals[79][0] - publics[156], evals[79][1], evals[79][2]];
    signal tmp_8190[3] <== CMul()(evals[10], tmp_8189);
    signal tmp_8191[3] <== [tmp_8188[0] + tmp_8190[0], tmp_8188[1] + tmp_8190[1], tmp_8188[2] + tmp_8190[2]];
    signal tmp_8192[3] <== CMul()(challengeQ, tmp_8191);
    signal tmp_8193[3] <== [evals[80][0] - publics[157], evals[80][1], evals[80][2]];
    signal tmp_8194[3] <== CMul()(evals[10], tmp_8193);
    signal tmp_8195[3] <== [tmp_8192[0] + tmp_8194[0], tmp_8192[1] + tmp_8194[1], tmp_8192[2] + tmp_8194[2]];
    signal tmp_8196[3] <== CMul()(challengeQ, tmp_8195);
    signal tmp_8197[3] <== [evals[81][0] - publics[158], evals[81][1], evals[81][2]];
    signal tmp_8198[3] <== CMul()(evals[10], tmp_8197);
    signal tmp_8199[3] <== [tmp_8196[0] + tmp_8198[0], tmp_8196[1] + tmp_8198[1], tmp_8196[2] + tmp_8198[2]];
    signal tmp_8200[3] <== CMul()(challengeQ, tmp_8199);
    signal tmp_8201[3] <== [evals[82][0] - publics[159], evals[82][1], evals[82][2]];
    signal tmp_8202[3] <== CMul()(evals[10], tmp_8201);
    signal tmp_8203[3] <== [tmp_8200[0] + tmp_8202[0], tmp_8200[1] + tmp_8202[1], tmp_8200[2] + tmp_8202[2]];
    signal tmp_8204[3] <== CMul()(challengeQ, tmp_8203);
    signal tmp_8205[3] <== [evals[83][0] - publics[160], evals[83][1], evals[83][2]];
    signal tmp_8206[3] <== CMul()(evals[10], tmp_8205);
    signal tmp_8207[3] <== [tmp_8204[0] + tmp_8206[0], tmp_8204[1] + tmp_8206[1], tmp_8204[2] + tmp_8206[2]];
    signal tmp_8208[3] <== CMul()(challengeQ, tmp_8207);
    signal tmp_8209[3] <== [evals[84][0] - publics[161], evals[84][1], evals[84][2]];
    signal tmp_8210[3] <== CMul()(evals[10], tmp_8209);
    signal tmp_8211[3] <== [tmp_8208[0] + tmp_8210[0], tmp_8208[1] + tmp_8210[1], tmp_8208[2] + tmp_8210[2]];
    signal tmp_8212[3] <== CMul()(challengeQ, tmp_8211);
    signal tmp_8213[3] <== [evals[85][0] - publics[162], evals[85][1], evals[85][2]];
    signal tmp_8214[3] <== CMul()(evals[10], tmp_8213);
    signal tmp_8215[3] <== [tmp_8212[0] + tmp_8214[0], tmp_8212[1] + tmp_8214[1], tmp_8212[2] + tmp_8214[2]];
    signal tmp_8216[3] <== CMul()(challengeQ, tmp_8215);
    signal tmp_8217[3] <== [evals[86][0] - publics[163], evals[86][1], evals[86][2]];
    signal tmp_8218[3] <== CMul()(evals[10], tmp_8217);
    signal tmp_8219[3] <== [tmp_8216[0] + tmp_8218[0], tmp_8216[1] + tmp_8218[1], tmp_8216[2] + tmp_8218[2]];
    signal tmp_8220[3] <== CMul()(challengeQ, tmp_8219);
    signal tmp_8221[3] <== [evals[87][0] - publics[164], evals[87][1], evals[87][2]];
    signal tmp_8222[3] <== CMul()(evals[10], tmp_8221);
    signal tmp_8223[3] <== [tmp_8220[0] + tmp_8222[0], tmp_8220[1] + tmp_8222[1], tmp_8220[2] + tmp_8222[2]];
    signal tmp_8224[3] <== CMul()(challengeQ, tmp_8223);
    signal tmp_8225[3] <== [evals[88][0] - publics[165], evals[88][1], evals[88][2]];
    signal tmp_8226[3] <== CMul()(evals[10], tmp_8225);
    signal tmp_8227[3] <== [tmp_8224[0] + tmp_8226[0], tmp_8224[1] + tmp_8226[1], tmp_8224[2] + tmp_8226[2]];
    signal tmp_8228[3] <== CMul()(challengeQ, tmp_8227);
    signal tmp_8229[3] <== [evals[89][0] - publics[166], evals[89][1], evals[89][2]];
    signal tmp_8230[3] <== CMul()(evals[10], tmp_8229);
    signal tmp_8231[3] <== [tmp_8228[0] + tmp_8230[0], tmp_8228[1] + tmp_8230[1], tmp_8228[2] + tmp_8230[2]];
    signal tmp_8232[3] <== CMul()(challengeQ, tmp_8231);
    signal tmp_8233[3] <== [evals[90][0] - publics[167], evals[90][1], evals[90][2]];
    signal tmp_8234[3] <== CMul()(evals[10], tmp_8233);
    signal tmp_8235[3] <== [tmp_8232[0] + tmp_8234[0], tmp_8232[1] + tmp_8234[1], tmp_8232[2] + tmp_8234[2]];
    signal tmp_8236[3] <== CMul()(challengeQ, tmp_8235);
    signal tmp_8237[3] <== [evals[67][0] - publics[168], evals[67][1], evals[67][2]];
    signal tmp_8238[3] <== CMul()(evals[9], tmp_8237);
    signal tmp_8239[3] <== [tmp_8236[0] + tmp_8238[0], tmp_8236[1] + tmp_8238[1], tmp_8236[2] + tmp_8238[2]];
    signal tmp_8240[3] <== CMul()(challengeQ, tmp_8239);
    signal tmp_8241[3] <== [evals[68][0] - publics[169], evals[68][1], evals[68][2]];
    signal tmp_8242[3] <== CMul()(evals[9], tmp_8241);
    signal tmp_8243[3] <== [tmp_8240[0] + tmp_8242[0], tmp_8240[1] + tmp_8242[1], tmp_8240[2] + tmp_8242[2]];
    signal tmp_8244[3] <== CMul()(challengeQ, tmp_8243);
    signal tmp_8245[3] <== [evals[69][0] - publics[170], evals[69][1], evals[69][2]];
    signal tmp_8246[3] <== CMul()(evals[9], tmp_8245);
    signal tmp_8247[3] <== [tmp_8244[0] + tmp_8246[0], tmp_8244[1] + tmp_8246[1], tmp_8244[2] + tmp_8246[2]];
    signal tmp_8248[3] <== CMul()(challengeQ, tmp_8247);
    signal tmp_8249[3] <== [evals[70][0] - publics[171], evals[70][1], evals[70][2]];
    signal tmp_8250[3] <== CMul()(evals[9], tmp_8249);
    signal tmp_8251[3] <== [tmp_8248[0] + tmp_8250[0], tmp_8248[1] + tmp_8250[1], tmp_8248[2] + tmp_8250[2]];
    signal tmp_8252[3] <== CMul()(challengeQ, tmp_8251);
    signal tmp_8253[3] <== [evals[71][0] - publics[172], evals[71][1], evals[71][2]];
    signal tmp_8254[3] <== CMul()(evals[9], tmp_8253);
    signal tmp_8255[3] <== [tmp_8252[0] + tmp_8254[0], tmp_8252[1] + tmp_8254[1], tmp_8252[2] + tmp_8254[2]];
    signal tmp_8256[3] <== CMul()(challengeQ, tmp_8255);
    signal tmp_8257[3] <== [evals[72][0] - publics[173], evals[72][1], evals[72][2]];
    signal tmp_8258[3] <== CMul()(evals[9], tmp_8257);
    signal tmp_8259[3] <== [tmp_8256[0] + tmp_8258[0], tmp_8256[1] + tmp_8258[1], tmp_8256[2] + tmp_8258[2]];
    signal tmp_8260[3] <== CMul()(challengeQ, tmp_8259);
    signal tmp_8261[3] <== [evals[73][0] - publics[174], evals[73][1], evals[73][2]];
    signal tmp_8262[3] <== CMul()(evals[9], tmp_8261);
    signal tmp_8263[3] <== [tmp_8260[0] + tmp_8262[0], tmp_8260[1] + tmp_8262[1], tmp_8260[2] + tmp_8262[2]];
    signal tmp_8264[3] <== CMul()(challengeQ, tmp_8263);
    signal tmp_8265[3] <== [evals[74][0] - publics[175], evals[74][1], evals[74][2]];
    signal tmp_8266[3] <== CMul()(evals[9], tmp_8265);
    signal tmp_8267[3] <== [tmp_8264[0] + tmp_8266[0], tmp_8264[1] + tmp_8266[1], tmp_8264[2] + tmp_8266[2]];
    signal tmp_8268[3] <== CMul()(challengeQ, tmp_8267);
    signal tmp_8269[3] <== [evals[75][0] - publics[176], evals[75][1], evals[75][2]];
    signal tmp_8270[3] <== CMul()(evals[9], tmp_8269);
    signal tmp_8271[3] <== [tmp_8268[0] + tmp_8270[0], tmp_8268[1] + tmp_8270[1], tmp_8268[2] + tmp_8270[2]];
    signal tmp_8272[3] <== CMul()(challengeQ, tmp_8271);
    signal tmp_8273[3] <== [evals[76][0] - publics[177], evals[76][1], evals[76][2]];
    signal tmp_8274[3] <== CMul()(evals[9], tmp_8273);
    signal tmp_8275[3] <== [tmp_8272[0] + tmp_8274[0], tmp_8272[1] + tmp_8274[1], tmp_8272[2] + tmp_8274[2]];
    signal tmp_8276[3] <== CMul()(challengeQ, tmp_8275);
    signal tmp_8277[3] <== [evals[77][0] - publics[178], evals[77][1], evals[77][2]];
    signal tmp_8278[3] <== CMul()(evals[9], tmp_8277);
    signal tmp_8279[3] <== [tmp_8276[0] + tmp_8278[0], tmp_8276[1] + tmp_8278[1], tmp_8276[2] + tmp_8278[2]];
    signal tmp_8280[3] <== CMul()(challengeQ, tmp_8279);
    signal tmp_8281[3] <== [evals[78][0] - publics[179], evals[78][1], evals[78][2]];
    signal tmp_8282[3] <== CMul()(evals[9], tmp_8281);
    signal tmp_8283[3] <== [tmp_8280[0] + tmp_8282[0], tmp_8280[1] + tmp_8282[1], tmp_8280[2] + tmp_8282[2]];
    signal tmp_8284[3] <== CMul()(challengeQ, tmp_8283);
    signal tmp_8285[3] <== [evals[79][0] - publics[180], evals[79][1], evals[79][2]];
    signal tmp_8286[3] <== CMul()(evals[9], tmp_8285);
    signal tmp_8287[3] <== [tmp_8284[0] + tmp_8286[0], tmp_8284[1] + tmp_8286[1], tmp_8284[2] + tmp_8286[2]];
    signal tmp_8288[3] <== CMul()(challengeQ, tmp_8287);
    signal tmp_8289[3] <== [evals[80][0] - publics[181], evals[80][1], evals[80][2]];
    signal tmp_8290[3] <== CMul()(evals[9], tmp_8289);
    signal tmp_8291[3] <== [tmp_8288[0] + tmp_8290[0], tmp_8288[1] + tmp_8290[1], tmp_8288[2] + tmp_8290[2]];
    signal tmp_8292[3] <== CMul()(challengeQ, tmp_8291);
    signal tmp_8293[3] <== [evals[81][0] - publics[182], evals[81][1], evals[81][2]];
    signal tmp_8294[3] <== CMul()(evals[9], tmp_8293);
    signal tmp_8295[3] <== [tmp_8292[0] + tmp_8294[0], tmp_8292[1] + tmp_8294[1], tmp_8292[2] + tmp_8294[2]];
    signal tmp_8296[3] <== CMul()(challengeQ, tmp_8295);
    signal tmp_8297[3] <== [evals[82][0] - publics[183], evals[82][1], evals[82][2]];
    signal tmp_8298[3] <== CMul()(evals[9], tmp_8297);
    signal tmp_8299[3] <== [tmp_8296[0] + tmp_8298[0], tmp_8296[1] + tmp_8298[1], tmp_8296[2] + tmp_8298[2]];
    signal tmp_8300[3] <== CMul()(challengeQ, tmp_8299);
    signal tmp_8301[3] <== [evals[83][0] - publics[184], evals[83][1], evals[83][2]];
    signal tmp_8302[3] <== CMul()(evals[9], tmp_8301);
    signal tmp_8303[3] <== [tmp_8300[0] + tmp_8302[0], tmp_8300[1] + tmp_8302[1], tmp_8300[2] + tmp_8302[2]];
    signal tmp_8304[3] <== CMul()(challengeQ, tmp_8303);
    signal tmp_8305[3] <== [evals[84][0] - publics[185], evals[84][1], evals[84][2]];
    signal tmp_8306[3] <== CMul()(evals[9], tmp_8305);
    signal tmp_8307[3] <== [tmp_8304[0] + tmp_8306[0], tmp_8304[1] + tmp_8306[1], tmp_8304[2] + tmp_8306[2]];
    signal tmp_8308[3] <== CMul()(challengeQ, tmp_8307);
    signal tmp_8309[3] <== [evals[85][0] - publics[186], evals[85][1], evals[85][2]];
    signal tmp_8310[3] <== CMul()(evals[9], tmp_8309);
    signal tmp_8311[3] <== [tmp_8308[0] + tmp_8310[0], tmp_8308[1] + tmp_8310[1], tmp_8308[2] + tmp_8310[2]];
    signal tmp_8312[3] <== CMul()(challengeQ, tmp_8311);
    signal tmp_8313[3] <== [evals[86][0] - publics[187], evals[86][1], evals[86][2]];
    signal tmp_8314[3] <== CMul()(evals[9], tmp_8313);
    signal tmp_8315[3] <== [tmp_8312[0] + tmp_8314[0], tmp_8312[1] + tmp_8314[1], tmp_8312[2] + tmp_8314[2]];
    signal tmp_8316[3] <== CMul()(challengeQ, tmp_8315);
    signal tmp_8317[3] <== [evals[87][0] - publics[188], evals[87][1], evals[87][2]];
    signal tmp_8318[3] <== CMul()(evals[9], tmp_8317);
    signal tmp_8319[3] <== [tmp_8316[0] + tmp_8318[0], tmp_8316[1] + tmp_8318[1], tmp_8316[2] + tmp_8318[2]];
    signal tmp_8320[3] <== CMul()(challengeQ, tmp_8319);
    signal tmp_8321[3] <== [evals[88][0] - publics[189], evals[88][1], evals[88][2]];
    signal tmp_8322[3] <== CMul()(evals[9], tmp_8321);
    signal tmp_8323[3] <== [tmp_8320[0] + tmp_8322[0], tmp_8320[1] + tmp_8322[1], tmp_8320[2] + tmp_8322[2]];
    signal tmp_8324[3] <== CMul()(challengeQ, tmp_8323);
    signal tmp_8325[3] <== [evals[89][0] - publics[190], evals[89][1], evals[89][2]];
    signal tmp_8326[3] <== CMul()(evals[9], tmp_8325);
    signal tmp_8327[3] <== [tmp_8324[0] + tmp_8326[0], tmp_8324[1] + tmp_8326[1], tmp_8324[2] + tmp_8326[2]];
    signal tmp_8328[3] <== CMul()(challengeQ, tmp_8327);
    signal tmp_8329[3] <== [evals[90][0] - publics[191], evals[90][1], evals[90][2]];
    signal tmp_8330[3] <== CMul()(evals[9], tmp_8329);
    signal tmp_8331[3] <== [tmp_8328[0] + tmp_8330[0], tmp_8328[1] + tmp_8330[1], tmp_8328[2] + tmp_8330[2]];
    signal tmp_8332[3] <== CMul()(challengeQ, tmp_8331);
    signal tmp_8333[3] <== [evals[67][0] - publics[192], evals[67][1], evals[67][2]];
    signal tmp_8334[3] <== CMul()(evals[8], tmp_8333);
    signal tmp_8335[3] <== [tmp_8332[0] + tmp_8334[0], tmp_8332[1] + tmp_8334[1], tmp_8332[2] + tmp_8334[2]];
    signal tmp_8336[3] <== CMul()(challengeQ, tmp_8335);
    signal tmp_8337[3] <== [evals[68][0] - publics[193], evals[68][1], evals[68][2]];
    signal tmp_8338[3] <== CMul()(evals[8], tmp_8337);
    signal tmp_8339[3] <== [tmp_8336[0] + tmp_8338[0], tmp_8336[1] + tmp_8338[1], tmp_8336[2] + tmp_8338[2]];
    signal tmp_8340[3] <== CMul()(challengeQ, tmp_8339);
    signal tmp_8341[3] <== [evals[69][0] - publics[194], evals[69][1], evals[69][2]];
    signal tmp_8342[3] <== CMul()(evals[8], tmp_8341);
    signal tmp_8343[3] <== [tmp_8340[0] + tmp_8342[0], tmp_8340[1] + tmp_8342[1], tmp_8340[2] + tmp_8342[2]];
    signal tmp_8344[3] <== CMul()(challengeQ, tmp_8343);
    signal tmp_8345[3] <== [evals[70][0] - publics[195], evals[70][1], evals[70][2]];
    signal tmp_8346[3] <== CMul()(evals[8], tmp_8345);
    signal tmp_8347[3] <== [tmp_8344[0] + tmp_8346[0], tmp_8344[1] + tmp_8346[1], tmp_8344[2] + tmp_8346[2]];
    signal tmp_8348[3] <== CMul()(challengeQ, tmp_8347);
    signal tmp_8349[3] <== [evals[71][0] - publics[196], evals[71][1], evals[71][2]];
    signal tmp_8350[3] <== CMul()(evals[8], tmp_8349);
    signal tmp_8351[3] <== [tmp_8348[0] + tmp_8350[0], tmp_8348[1] + tmp_8350[1], tmp_8348[2] + tmp_8350[2]];
    signal tmp_8352[3] <== CMul()(challengeQ, tmp_8351);
    signal tmp_8353[3] <== [evals[72][0] - publics[197], evals[72][1], evals[72][2]];
    signal tmp_8354[3] <== CMul()(evals[8], tmp_8353);
    signal tmp_8355[3] <== [tmp_8352[0] + tmp_8354[0], tmp_8352[1] + tmp_8354[1], tmp_8352[2] + tmp_8354[2]];
    signal tmp_8356[3] <== CMul()(challengeQ, tmp_8355);
    signal tmp_8357[3] <== [evals[73][0] - publics[198], evals[73][1], evals[73][2]];
    signal tmp_8358[3] <== CMul()(evals[8], tmp_8357);
    signal tmp_8359[3] <== [tmp_8356[0] + tmp_8358[0], tmp_8356[1] + tmp_8358[1], tmp_8356[2] + tmp_8358[2]];
    signal tmp_8360[3] <== CMul()(challengeQ, tmp_8359);
    signal tmp_8361[3] <== [evals[74][0] - publics[199], evals[74][1], evals[74][2]];
    signal tmp_8362[3] <== CMul()(evals[8], tmp_8361);
    signal tmp_8363[3] <== [tmp_8360[0] + tmp_8362[0], tmp_8360[1] + tmp_8362[1], tmp_8360[2] + tmp_8362[2]];
    signal tmp_8364[3] <== CMul()(challengeQ, tmp_8363);
    signal tmp_8365[3] <== [evals[75][0] - publics[200], evals[75][1], evals[75][2]];
    signal tmp_8366[3] <== CMul()(evals[8], tmp_8365);
    signal tmp_8367[3] <== [tmp_8364[0] + tmp_8366[0], tmp_8364[1] + tmp_8366[1], tmp_8364[2] + tmp_8366[2]];
    signal tmp_8368[3] <== CMul()(challengeQ, tmp_8367);
    signal tmp_8369[3] <== [evals[76][0] - publics[201], evals[76][1], evals[76][2]];
    signal tmp_8370[3] <== CMul()(evals[8], tmp_8369);
    signal tmp_8371[3] <== [tmp_8368[0] + tmp_8370[0], tmp_8368[1] + tmp_8370[1], tmp_8368[2] + tmp_8370[2]];
    signal tmp_8372[3] <== CMul()(challengeQ, tmp_8371);
    signal tmp_8373[3] <== [evals[77][0] - publics[202], evals[77][1], evals[77][2]];
    signal tmp_8374[3] <== CMul()(evals[8], tmp_8373);
    signal tmp_8375[3] <== [tmp_8372[0] + tmp_8374[0], tmp_8372[1] + tmp_8374[1], tmp_8372[2] + tmp_8374[2]];
    signal tmp_8376[3] <== CMul()(challengeQ, tmp_8375);
    signal tmp_8377[3] <== [evals[78][0] - publics[203], evals[78][1], evals[78][2]];
    signal tmp_8378[3] <== CMul()(evals[8], tmp_8377);
    signal tmp_8379[3] <== [tmp_8376[0] + tmp_8378[0], tmp_8376[1] + tmp_8378[1], tmp_8376[2] + tmp_8378[2]];
    signal tmp_8380[3] <== CMul()(challengeQ, tmp_8379);
    signal tmp_8381[3] <== [evals[79][0] - publics[204], evals[79][1], evals[79][2]];
    signal tmp_8382[3] <== CMul()(evals[8], tmp_8381);
    signal tmp_8383[3] <== [tmp_8380[0] + tmp_8382[0], tmp_8380[1] + tmp_8382[1], tmp_8380[2] + tmp_8382[2]];
    signal tmp_8384[3] <== CMul()(challengeQ, tmp_8383);
    signal tmp_8385[3] <== [evals[80][0] - publics[205], evals[80][1], evals[80][2]];
    signal tmp_8386[3] <== CMul()(evals[8], tmp_8385);
    signal tmp_8387[3] <== [tmp_8384[0] + tmp_8386[0], tmp_8384[1] + tmp_8386[1], tmp_8384[2] + tmp_8386[2]];
    signal tmp_8388[3] <== CMul()(challengeQ, tmp_8387);
    signal tmp_8389[3] <== [evals[81][0] - publics[206], evals[81][1], evals[81][2]];
    signal tmp_8390[3] <== CMul()(evals[8], tmp_8389);
    signal tmp_8391[3] <== [tmp_8388[0] + tmp_8390[0], tmp_8388[1] + tmp_8390[1], tmp_8388[2] + tmp_8390[2]];
    signal tmp_8392[3] <== CMul()(challengeQ, tmp_8391);
    signal tmp_8393[3] <== [evals[82][0] - publics[207], evals[82][1], evals[82][2]];
    signal tmp_8394[3] <== CMul()(evals[8], tmp_8393);
    signal tmp_8395[3] <== [tmp_8392[0] + tmp_8394[0], tmp_8392[1] + tmp_8394[1], tmp_8392[2] + tmp_8394[2]];
    signal tmp_8396[3] <== CMul()(challengeQ, tmp_8395);
    signal tmp_8397[3] <== [evals[83][0] - publics[208], evals[83][1], evals[83][2]];
    signal tmp_8398[3] <== CMul()(evals[8], tmp_8397);
    signal tmp_8399[3] <== [tmp_8396[0] + tmp_8398[0], tmp_8396[1] + tmp_8398[1], tmp_8396[2] + tmp_8398[2]];
    signal tmp_8400[3] <== CMul()(challengeQ, tmp_8399);
    signal tmp_8401[3] <== [evals[84][0] - publics[209], evals[84][1], evals[84][2]];
    signal tmp_8402[3] <== CMul()(evals[8], tmp_8401);
    signal tmp_8403[3] <== [tmp_8400[0] + tmp_8402[0], tmp_8400[1] + tmp_8402[1], tmp_8400[2] + tmp_8402[2]];
    signal tmp_8404[3] <== CMul()(challengeQ, tmp_8403);
    signal tmp_8405[3] <== [evals[85][0] - publics[210], evals[85][1], evals[85][2]];
    signal tmp_8406[3] <== CMul()(evals[8], tmp_8405);
    signal tmp_8407[3] <== [tmp_8404[0] + tmp_8406[0], tmp_8404[1] + tmp_8406[1], tmp_8404[2] + tmp_8406[2]];
    signal tmp_8408[3] <== CMul()(challengeQ, tmp_8407);
    signal tmp_8409[3] <== [evals[86][0] - publics[211], evals[86][1], evals[86][2]];
    signal tmp_8410[3] <== CMul()(evals[8], tmp_8409);
    signal tmp_8411[3] <== [tmp_8408[0] + tmp_8410[0], tmp_8408[1] + tmp_8410[1], tmp_8408[2] + tmp_8410[2]];
    signal tmp_8412[3] <== CMul()(challengeQ, tmp_8411);
    signal tmp_8413[3] <== [evals[87][0] - publics[212], evals[87][1], evals[87][2]];
    signal tmp_8414[3] <== CMul()(evals[8], tmp_8413);
    signal tmp_8415[3] <== [tmp_8412[0] + tmp_8414[0], tmp_8412[1] + tmp_8414[1], tmp_8412[2] + tmp_8414[2]];
    signal tmp_8416[3] <== CMul()(challengeQ, tmp_8415);
    signal tmp_8417[3] <== [evals[88][0] - publics[213], evals[88][1], evals[88][2]];
    signal tmp_8418[3] <== CMul()(evals[8], tmp_8417);
    signal tmp_8419[3] <== [tmp_8416[0] + tmp_8418[0], tmp_8416[1] + tmp_8418[1], tmp_8416[2] + tmp_8418[2]];
    signal tmp_8420[3] <== CMul()(challengeQ, tmp_8419);
    signal tmp_8421[3] <== [evals[89][0] - publics[214], evals[89][1], evals[89][2]];
    signal tmp_8422[3] <== CMul()(evals[8], tmp_8421);
    signal tmp_8423[3] <== [tmp_8420[0] + tmp_8422[0], tmp_8420[1] + tmp_8422[1], tmp_8420[2] + tmp_8422[2]];
    signal tmp_8424[3] <== CMul()(challengeQ, tmp_8423);
    signal tmp_8425[3] <== [evals[90][0] - publics[215], evals[90][1], evals[90][2]];
    signal tmp_8426[3] <== CMul()(evals[8], tmp_8425);
    signal tmp_8427[3] <== [tmp_8424[0] + tmp_8426[0], tmp_8424[1] + tmp_8426[1], tmp_8424[2] + tmp_8426[2]];
    signal tmp_8428[3] <== CMul()(challengeQ, tmp_8427);
    signal tmp_8429[3] <== [evals[67][0] - publics[216], evals[67][1], evals[67][2]];
    signal tmp_8430[3] <== CMul()(evals[7], tmp_8429);
    signal tmp_8431[3] <== [tmp_8428[0] + tmp_8430[0], tmp_8428[1] + tmp_8430[1], tmp_8428[2] + tmp_8430[2]];
    signal tmp_8432[3] <== CMul()(challengeQ, tmp_8431);
    signal tmp_8433[3] <== [evals[68][0] - publics[217], evals[68][1], evals[68][2]];
    signal tmp_8434[3] <== CMul()(evals[7], tmp_8433);
    signal tmp_8435[3] <== [tmp_8432[0] + tmp_8434[0], tmp_8432[1] + tmp_8434[1], tmp_8432[2] + tmp_8434[2]];
    signal tmp_8436[3] <== CMul()(challengeQ, tmp_8435);
    signal tmp_8437[3] <== [evals[69][0] - publics[218], evals[69][1], evals[69][2]];
    signal tmp_8438[3] <== CMul()(evals[7], tmp_8437);
    signal tmp_8439[3] <== [tmp_8436[0] + tmp_8438[0], tmp_8436[1] + tmp_8438[1], tmp_8436[2] + tmp_8438[2]];
    signal tmp_8440[3] <== CMul()(challengeQ, tmp_8439);
    signal tmp_8441[3] <== [evals[70][0] - publics[219], evals[70][1], evals[70][2]];
    signal tmp_8442[3] <== CMul()(evals[7], tmp_8441);
    signal tmp_8443[3] <== [tmp_8440[0] + tmp_8442[0], tmp_8440[1] + tmp_8442[1], tmp_8440[2] + tmp_8442[2]];
    signal tmp_8444[3] <== CMul()(challengeQ, tmp_8443);
    signal tmp_8445[3] <== [evals[71][0] - publics[220], evals[71][1], evals[71][2]];
    signal tmp_8446[3] <== CMul()(evals[7], tmp_8445);
    signal tmp_8447[3] <== [tmp_8444[0] + tmp_8446[0], tmp_8444[1] + tmp_8446[1], tmp_8444[2] + tmp_8446[2]];
    signal tmp_8448[3] <== CMul()(challengeQ, tmp_8447);
    signal tmp_8449[3] <== [evals[72][0] - publics[221], evals[72][1], evals[72][2]];
    signal tmp_8450[3] <== CMul()(evals[7], tmp_8449);
    signal tmp_8451[3] <== [tmp_8448[0] + tmp_8450[0], tmp_8448[1] + tmp_8450[1], tmp_8448[2] + tmp_8450[2]];
    signal tmp_8452[3] <== CMul()(challengeQ, tmp_8451);
    signal tmp_8453[3] <== [evals[73][0] - publics[222], evals[73][1], evals[73][2]];
    signal tmp_8454[3] <== CMul()(evals[7], tmp_8453);
    signal tmp_8455[3] <== [tmp_8452[0] + tmp_8454[0], tmp_8452[1] + tmp_8454[1], tmp_8452[2] + tmp_8454[2]];
    signal tmp_8456[3] <== CMul()(challengeQ, tmp_8455);
    signal tmp_8457[3] <== [evals[74][0] - publics[223], evals[74][1], evals[74][2]];
    signal tmp_8458[3] <== CMul()(evals[7], tmp_8457);
    signal tmp_8459[3] <== [tmp_8456[0] + tmp_8458[0], tmp_8456[1] + tmp_8458[1], tmp_8456[2] + tmp_8458[2]];
    signal tmp_8460[3] <== CMul()(challengeQ, tmp_8459);
    signal tmp_8461[3] <== [evals[75][0] - publics[224], evals[75][1], evals[75][2]];
    signal tmp_8462[3] <== CMul()(evals[7], tmp_8461);
    signal tmp_8463[3] <== [tmp_8460[0] + tmp_8462[0], tmp_8460[1] + tmp_8462[1], tmp_8460[2] + tmp_8462[2]];
    signal tmp_8464[3] <== CMul()(challengeQ, tmp_8463);
    signal tmp_8465[3] <== [evals[76][0] - publics[225], evals[76][1], evals[76][2]];
    signal tmp_8466[3] <== CMul()(evals[7], tmp_8465);
    signal tmp_8467[3] <== [tmp_8464[0] + tmp_8466[0], tmp_8464[1] + tmp_8466[1], tmp_8464[2] + tmp_8466[2]];
    signal tmp_8468[3] <== CMul()(challengeQ, tmp_8467);
    signal tmp_8469[3] <== [evals[77][0] - publics[226], evals[77][1], evals[77][2]];
    signal tmp_8470[3] <== CMul()(evals[7], tmp_8469);
    signal tmp_8471[3] <== [tmp_8468[0] + tmp_8470[0], tmp_8468[1] + tmp_8470[1], tmp_8468[2] + tmp_8470[2]];
    signal tmp_8472[3] <== CMul()(challengeQ, tmp_8471);
    signal tmp_8473[3] <== [evals[78][0] - publics[227], evals[78][1], evals[78][2]];
    signal tmp_8474[3] <== CMul()(evals[7], tmp_8473);
    signal tmp_8475[3] <== [tmp_8472[0] + tmp_8474[0], tmp_8472[1] + tmp_8474[1], tmp_8472[2] + tmp_8474[2]];
    signal tmp_8476[3] <== CMul()(challengeQ, tmp_8475);
    signal tmp_8477[3] <== [evals[79][0] - publics[228], evals[79][1], evals[79][2]];
    signal tmp_8478[3] <== CMul()(evals[7], tmp_8477);
    signal tmp_8479[3] <== [tmp_8476[0] + tmp_8478[0], tmp_8476[1] + tmp_8478[1], tmp_8476[2] + tmp_8478[2]];
    signal tmp_8480[3] <== CMul()(challengeQ, tmp_8479);
    signal tmp_8481[3] <== [evals[80][0] - publics[229], evals[80][1], evals[80][2]];
    signal tmp_8482[3] <== CMul()(evals[7], tmp_8481);
    signal tmp_8483[3] <== [tmp_8480[0] + tmp_8482[0], tmp_8480[1] + tmp_8482[1], tmp_8480[2] + tmp_8482[2]];
    signal tmp_8484[3] <== CMul()(challengeQ, tmp_8483);
    signal tmp_8485[3] <== [evals[81][0] - publics[230], evals[81][1], evals[81][2]];
    signal tmp_8486[3] <== CMul()(evals[7], tmp_8485);
    signal tmp_8487[3] <== [tmp_8484[0] + tmp_8486[0], tmp_8484[1] + tmp_8486[1], tmp_8484[2] + tmp_8486[2]];
    signal tmp_8488[3] <== CMul()(challengeQ, tmp_8487);
    signal tmp_8489[3] <== [evals[82][0] - publics[231], evals[82][1], evals[82][2]];
    signal tmp_8490[3] <== CMul()(evals[7], tmp_8489);
    signal tmp_8491[3] <== [tmp_8488[0] + tmp_8490[0], tmp_8488[1] + tmp_8490[1], tmp_8488[2] + tmp_8490[2]];
    signal tmp_8492[3] <== CMul()(challengeQ, tmp_8491);
    signal tmp_8493[3] <== [evals[83][0] - publics[232], evals[83][1], evals[83][2]];
    signal tmp_8494[3] <== CMul()(evals[7], tmp_8493);
    signal tmp_8495[3] <== [tmp_8492[0] + tmp_8494[0], tmp_8492[1] + tmp_8494[1], tmp_8492[2] + tmp_8494[2]];
    signal tmp_8496[3] <== CMul()(challengeQ, tmp_8495);
    signal tmp_8497[3] <== [evals[84][0] - publics[233], evals[84][1], evals[84][2]];
    signal tmp_8498[3] <== CMul()(evals[7], tmp_8497);
    signal tmp_8499[3] <== [tmp_8496[0] + tmp_8498[0], tmp_8496[1] + tmp_8498[1], tmp_8496[2] + tmp_8498[2]];
    signal tmp_8500[3] <== CMul()(challengeQ, tmp_8499);
    signal tmp_8501[3] <== [evals[85][0] - publics[234], evals[85][1], evals[85][2]];
    signal tmp_8502[3] <== CMul()(evals[7], tmp_8501);
    signal tmp_8503[3] <== [tmp_8500[0] + tmp_8502[0], tmp_8500[1] + tmp_8502[1], tmp_8500[2] + tmp_8502[2]];
    signal tmp_8504[3] <== CMul()(challengeQ, tmp_8503);
    signal tmp_8505[3] <== [evals[86][0] - publics[235], evals[86][1], evals[86][2]];
    signal tmp_8506[3] <== CMul()(evals[7], tmp_8505);
    signal tmp_8507[3] <== [tmp_8504[0] + tmp_8506[0], tmp_8504[1] + tmp_8506[1], tmp_8504[2] + tmp_8506[2]];
    signal tmp_8508[3] <== CMul()(challengeQ, tmp_8507);
    signal tmp_8509[3] <== [evals[87][0] - publics[236], evals[87][1], evals[87][2]];
    signal tmp_8510[3] <== CMul()(evals[7], tmp_8509);
    signal tmp_8511[3] <== [tmp_8508[0] + tmp_8510[0], tmp_8508[1] + tmp_8510[1], tmp_8508[2] + tmp_8510[2]];
    signal tmp_8512[3] <== CMul()(challengeQ, tmp_8511);
    signal tmp_8513[3] <== [evals[88][0] - publics[237], evals[88][1], evals[88][2]];
    signal tmp_8514[3] <== CMul()(evals[7], tmp_8513);
    signal tmp_8515[3] <== [tmp_8512[0] + tmp_8514[0], tmp_8512[1] + tmp_8514[1], tmp_8512[2] + tmp_8514[2]];
    signal tmp_8516[3] <== CMul()(challengeQ, tmp_8515);
    signal tmp_8517[3] <== [evals[89][0] - publics[238], evals[89][1], evals[89][2]];
    signal tmp_8518[3] <== CMul()(evals[7], tmp_8517);
    signal tmp_8519[3] <== [tmp_8516[0] + tmp_8518[0], tmp_8516[1] + tmp_8518[1], tmp_8516[2] + tmp_8518[2]];
    signal tmp_8520[3] <== CMul()(challengeQ, tmp_8519);
    signal tmp_8521[3] <== [evals[90][0] - publics[239], evals[90][1], evals[90][2]];
    signal tmp_8522[3] <== CMul()(evals[7], tmp_8521);
    signal tmp_8523[3] <== [tmp_8520[0] + tmp_8522[0], tmp_8520[1] + tmp_8522[1], tmp_8520[2] + tmp_8522[2]];
    signal tmp_8524[3] <== CMul()(challengeQ, tmp_8523);
    signal tmp_8525[3] <== [evals[67][0] - publics[240], evals[67][1], evals[67][2]];
    signal tmp_8526[3] <== CMul()(evals[6], tmp_8525);
    signal tmp_8527[3] <== [tmp_8524[0] + tmp_8526[0], tmp_8524[1] + tmp_8526[1], tmp_8524[2] + tmp_8526[2]];
    signal tmp_8528[3] <== CMul()(challengeQ, tmp_8527);
    signal tmp_8529[3] <== [evals[68][0] - publics[241], evals[68][1], evals[68][2]];
    signal tmp_8530[3] <== CMul()(evals[6], tmp_8529);
    signal tmp_8531[3] <== [tmp_8528[0] + tmp_8530[0], tmp_8528[1] + tmp_8530[1], tmp_8528[2] + tmp_8530[2]];
    signal tmp_8532[3] <== CMul()(challengeQ, tmp_8531);
    signal tmp_8533[3] <== [evals[69][0] - publics[242], evals[69][1], evals[69][2]];
    signal tmp_8534[3] <== CMul()(evals[6], tmp_8533);
    signal tmp_8535[3] <== [tmp_8532[0] + tmp_8534[0], tmp_8532[1] + tmp_8534[1], tmp_8532[2] + tmp_8534[2]];
    signal tmp_8536[3] <== CMul()(challengeQ, tmp_8535);
    signal tmp_8537[3] <== [evals[70][0] - publics[243], evals[70][1], evals[70][2]];
    signal tmp_8538[3] <== CMul()(evals[6], tmp_8537);
    signal tmp_8539[3] <== [tmp_8536[0] + tmp_8538[0], tmp_8536[1] + tmp_8538[1], tmp_8536[2] + tmp_8538[2]];
    signal tmp_8540[3] <== CMul()(challengeQ, tmp_8539);
    signal tmp_8541[3] <== [evals[71][0] - publics[244], evals[71][1], evals[71][2]];
    signal tmp_8542[3] <== CMul()(evals[6], tmp_8541);
    signal tmp_8543[3] <== [tmp_8540[0] + tmp_8542[0], tmp_8540[1] + tmp_8542[1], tmp_8540[2] + tmp_8542[2]];
    signal tmp_8544[3] <== CMul()(challengeQ, tmp_8543);
    signal tmp_8545[3] <== [evals[72][0] - publics[245], evals[72][1], evals[72][2]];
    signal tmp_8546[3] <== CMul()(evals[6], tmp_8545);
    signal tmp_8547[3] <== [tmp_8544[0] + tmp_8546[0], tmp_8544[1] + tmp_8546[1], tmp_8544[2] + tmp_8546[2]];
    signal tmp_8548[3] <== CMul()(challengeQ, tmp_8547);
    signal tmp_8549[3] <== [evals[73][0] - publics[246], evals[73][1], evals[73][2]];
    signal tmp_8550[3] <== CMul()(evals[6], tmp_8549);
    signal tmp_8551[3] <== [tmp_8548[0] + tmp_8550[0], tmp_8548[1] + tmp_8550[1], tmp_8548[2] + tmp_8550[2]];
    signal tmp_8552[3] <== CMul()(challengeQ, tmp_8551);
    signal tmp_8553[3] <== [evals[74][0] - publics[247], evals[74][1], evals[74][2]];
    signal tmp_8554[3] <== CMul()(evals[6], tmp_8553);
    signal tmp_8555[3] <== [tmp_8552[0] + tmp_8554[0], tmp_8552[1] + tmp_8554[1], tmp_8552[2] + tmp_8554[2]];
    signal tmp_8556[3] <== CMul()(challengeQ, tmp_8555);
    signal tmp_8557[3] <== [evals[75][0] - publics[248], evals[75][1], evals[75][2]];
    signal tmp_8558[3] <== CMul()(evals[6], tmp_8557);
    signal tmp_8559[3] <== [tmp_8556[0] + tmp_8558[0], tmp_8556[1] + tmp_8558[1], tmp_8556[2] + tmp_8558[2]];
    signal tmp_8560[3] <== CMul()(challengeQ, tmp_8559);
    signal tmp_8561[3] <== [evals[76][0] - publics[249], evals[76][1], evals[76][2]];
    signal tmp_8562[3] <== CMul()(evals[6], tmp_8561);
    signal tmp_8563[3] <== [tmp_8560[0] + tmp_8562[0], tmp_8560[1] + tmp_8562[1], tmp_8560[2] + tmp_8562[2]];
    tmp_8564 <== CMul()(challengeQ, tmp_8563);
}

template VerifyEvaluationsChunks1_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_8564[3];

    signal output tmp_9161[3];
    signal output tmp_9163[3];
    signal output tmp_9180[3];
    signal output tmp_9192[3];
    signal output tmp_9210[3];
    signal output tmp_9500[3];
    signal output tmp_9501[3];
    signal output tmp_9504[3];
    signal output tmp_9518[3];
    signal output tmp_9532[3];
    signal output tmp_9547[3];
    signal output tmp_9563[3];
    signal output tmp_9578[3];
    signal output tmp_9580[3];
    signal output tmp_9586[3];
    signal tmp_8565[3] <== [evals[77][0] - publics[250], evals[77][1], evals[77][2]];
    signal tmp_8566[3] <== CMul()(evals[6], tmp_8565);
    signal tmp_8567[3] <== [tmp_8564[0] + tmp_8566[0], tmp_8564[1] + tmp_8566[1], tmp_8564[2] + tmp_8566[2]];
    signal tmp_8568[3] <== CMul()(challengeQ, tmp_8567);
    signal tmp_8569[3] <== [evals[78][0] - publics[251], evals[78][1], evals[78][2]];
    signal tmp_8570[3] <== CMul()(evals[6], tmp_8569);
    signal tmp_8571[3] <== [tmp_8568[0] + tmp_8570[0], tmp_8568[1] + tmp_8570[1], tmp_8568[2] + tmp_8570[2]];
    signal tmp_8572[3] <== CMul()(challengeQ, tmp_8571);
    signal tmp_8573[3] <== [evals[79][0] - publics[252], evals[79][1], evals[79][2]];
    signal tmp_8574[3] <== CMul()(evals[6], tmp_8573);
    signal tmp_8575[3] <== [tmp_8572[0] + tmp_8574[0], tmp_8572[1] + tmp_8574[1], tmp_8572[2] + tmp_8574[2]];
    signal tmp_8576[3] <== CMul()(challengeQ, tmp_8575);
    signal tmp_8577[3] <== [evals[80][0] - publics[253], evals[80][1], evals[80][2]];
    signal tmp_8578[3] <== CMul()(evals[6], tmp_8577);
    signal tmp_8579[3] <== [tmp_8576[0] + tmp_8578[0], tmp_8576[1] + tmp_8578[1], tmp_8576[2] + tmp_8578[2]];
    signal tmp_8580[3] <== CMul()(challengeQ, tmp_8579);
    signal tmp_8581[3] <== [evals[81][0] - publics[254], evals[81][1], evals[81][2]];
    signal tmp_8582[3] <== CMul()(evals[6], tmp_8581);
    signal tmp_8583[3] <== [tmp_8580[0] + tmp_8582[0], tmp_8580[1] + tmp_8582[1], tmp_8580[2] + tmp_8582[2]];
    signal tmp_8584[3] <== CMul()(challengeQ, tmp_8583);
    signal tmp_8585[3] <== [evals[82][0] - publics[255], evals[82][1], evals[82][2]];
    signal tmp_8586[3] <== CMul()(evals[6], tmp_8585);
    signal tmp_8587[3] <== [tmp_8584[0] + tmp_8586[0], tmp_8584[1] + tmp_8586[1], tmp_8584[2] + tmp_8586[2]];
    signal tmp_8588[3] <== CMul()(challengeQ, tmp_8587);
    signal tmp_8589[3] <== [evals[83][0] - publics[256], evals[83][1], evals[83][2]];
    signal tmp_8590[3] <== CMul()(evals[6], tmp_8589);
    signal tmp_8591[3] <== [tmp_8588[0] + tmp_8590[0], tmp_8588[1] + tmp_8590[1], tmp_8588[2] + tmp_8590[2]];
    signal tmp_8592[3] <== CMul()(challengeQ, tmp_8591);
    signal tmp_8593[3] <== [evals[84][0] - publics[257], evals[84][1], evals[84][2]];
    signal tmp_8594[3] <== CMul()(evals[6], tmp_8593);
    signal tmp_8595[3] <== [tmp_8592[0] + tmp_8594[0], tmp_8592[1] + tmp_8594[1], tmp_8592[2] + tmp_8594[2]];
    signal tmp_8596[3] <== CMul()(challengeQ, tmp_8595);
    signal tmp_8597[3] <== [evals[85][0] - publics[258], evals[85][1], evals[85][2]];
    signal tmp_8598[3] <== CMul()(evals[6], tmp_8597);
    signal tmp_8599[3] <== [tmp_8596[0] + tmp_8598[0], tmp_8596[1] + tmp_8598[1], tmp_8596[2] + tmp_8598[2]];
    signal tmp_8600[3] <== CMul()(challengeQ, tmp_8599);
    signal tmp_8601[3] <== [evals[86][0] - publics[259], evals[86][1], evals[86][2]];
    signal tmp_8602[3] <== CMul()(evals[6], tmp_8601);
    signal tmp_8603[3] <== [tmp_8600[0] + tmp_8602[0], tmp_8600[1] + tmp_8602[1], tmp_8600[2] + tmp_8602[2]];
    signal tmp_8604[3] <== CMul()(challengeQ, tmp_8603);
    signal tmp_8605[3] <== [evals[87][0] - publics[260], evals[87][1], evals[87][2]];
    signal tmp_8606[3] <== CMul()(evals[6], tmp_8605);
    signal tmp_8607[3] <== [tmp_8604[0] + tmp_8606[0], tmp_8604[1] + tmp_8606[1], tmp_8604[2] + tmp_8606[2]];
    signal tmp_8608[3] <== CMul()(challengeQ, tmp_8607);
    signal tmp_8609[3] <== [evals[88][0] - publics[261], evals[88][1], evals[88][2]];
    signal tmp_8610[3] <== CMul()(evals[6], tmp_8609);
    signal tmp_8611[3] <== [tmp_8608[0] + tmp_8610[0], tmp_8608[1] + tmp_8610[1], tmp_8608[2] + tmp_8610[2]];
    signal tmp_8612[3] <== CMul()(challengeQ, tmp_8611);
    signal tmp_8613[3] <== [evals[89][0] - publics[262], evals[89][1], evals[89][2]];
    signal tmp_8614[3] <== CMul()(evals[6], tmp_8613);
    signal tmp_8615[3] <== [tmp_8612[0] + tmp_8614[0], tmp_8612[1] + tmp_8614[1], tmp_8612[2] + tmp_8614[2]];
    signal tmp_8616[3] <== CMul()(challengeQ, tmp_8615);
    signal tmp_8617[3] <== [evals[90][0] - publics[263], evals[90][1], evals[90][2]];
    signal tmp_8618[3] <== CMul()(evals[6], tmp_8617);
    signal tmp_8619[3] <== [tmp_8616[0] + tmp_8618[0], tmp_8616[1] + tmp_8618[1], tmp_8616[2] + tmp_8618[2]];
    signal tmp_8620[3] <== CMul()(challengeQ, tmp_8619);
    signal tmp_8621[3] <== [evals[67][0] - publics[264], evals[67][1], evals[67][2]];
    signal tmp_8622[3] <== CMul()(evals[5], tmp_8621);
    signal tmp_8623[3] <== [tmp_8620[0] + tmp_8622[0], tmp_8620[1] + tmp_8622[1], tmp_8620[2] + tmp_8622[2]];
    signal tmp_8624[3] <== CMul()(challengeQ, tmp_8623);
    signal tmp_8625[3] <== [evals[68][0] - publics[265], evals[68][1], evals[68][2]];
    signal tmp_8626[3] <== CMul()(evals[5], tmp_8625);
    signal tmp_8627[3] <== [tmp_8624[0] + tmp_8626[0], tmp_8624[1] + tmp_8626[1], tmp_8624[2] + tmp_8626[2]];
    signal tmp_8628[3] <== CMul()(challengeQ, tmp_8627);
    signal tmp_8629[3] <== [evals[69][0] - publics[266], evals[69][1], evals[69][2]];
    signal tmp_8630[3] <== CMul()(evals[5], tmp_8629);
    signal tmp_8631[3] <== [tmp_8628[0] + tmp_8630[0], tmp_8628[1] + tmp_8630[1], tmp_8628[2] + tmp_8630[2]];
    signal tmp_8632[3] <== CMul()(challengeQ, tmp_8631);
    signal tmp_8633[3] <== [evals[70][0] - publics[267], evals[70][1], evals[70][2]];
    signal tmp_8634[3] <== CMul()(evals[5], tmp_8633);
    signal tmp_8635[3] <== [tmp_8632[0] + tmp_8634[0], tmp_8632[1] + tmp_8634[1], tmp_8632[2] + tmp_8634[2]];
    signal tmp_8636[3] <== CMul()(challengeQ, tmp_8635);
    signal tmp_8637[3] <== [evals[71][0] - publics[268], evals[71][1], evals[71][2]];
    signal tmp_8638[3] <== CMul()(evals[5], tmp_8637);
    signal tmp_8639[3] <== [tmp_8636[0] + tmp_8638[0], tmp_8636[1] + tmp_8638[1], tmp_8636[2] + tmp_8638[2]];
    signal tmp_8640[3] <== CMul()(challengeQ, tmp_8639);
    signal tmp_8641[3] <== [evals[72][0] - publics[269], evals[72][1], evals[72][2]];
    signal tmp_8642[3] <== CMul()(evals[5], tmp_8641);
    signal tmp_8643[3] <== [tmp_8640[0] + tmp_8642[0], tmp_8640[1] + tmp_8642[1], tmp_8640[2] + tmp_8642[2]];
    signal tmp_8644[3] <== CMul()(challengeQ, tmp_8643);
    signal tmp_8645[3] <== [evals[73][0] - publics[270], evals[73][1], evals[73][2]];
    signal tmp_8646[3] <== CMul()(evals[5], tmp_8645);
    signal tmp_8647[3] <== [tmp_8644[0] + tmp_8646[0], tmp_8644[1] + tmp_8646[1], tmp_8644[2] + tmp_8646[2]];
    signal tmp_8648[3] <== CMul()(challengeQ, tmp_8647);
    signal tmp_8649[3] <== [evals[74][0] - publics[271], evals[74][1], evals[74][2]];
    signal tmp_8650[3] <== CMul()(evals[5], tmp_8649);
    signal tmp_8651[3] <== [tmp_8648[0] + tmp_8650[0], tmp_8648[1] + tmp_8650[1], tmp_8648[2] + tmp_8650[2]];
    signal tmp_8652[3] <== CMul()(challengeQ, tmp_8651);
    signal tmp_8653[3] <== [evals[75][0] - publics[272], evals[75][1], evals[75][2]];
    signal tmp_8654[3] <== CMul()(evals[5], tmp_8653);
    signal tmp_8655[3] <== [tmp_8652[0] + tmp_8654[0], tmp_8652[1] + tmp_8654[1], tmp_8652[2] + tmp_8654[2]];
    signal tmp_8656[3] <== CMul()(challengeQ, tmp_8655);
    signal tmp_8657[3] <== [evals[76][0] - publics[273], evals[76][1], evals[76][2]];
    signal tmp_8658[3] <== CMul()(evals[5], tmp_8657);
    signal tmp_8659[3] <== [tmp_8656[0] + tmp_8658[0], tmp_8656[1] + tmp_8658[1], tmp_8656[2] + tmp_8658[2]];
    signal tmp_8660[3] <== CMul()(challengeQ, tmp_8659);
    signal tmp_8661[3] <== [evals[77][0] - publics[274], evals[77][1], evals[77][2]];
    signal tmp_8662[3] <== CMul()(evals[5], tmp_8661);
    signal tmp_8663[3] <== [tmp_8660[0] + tmp_8662[0], tmp_8660[1] + tmp_8662[1], tmp_8660[2] + tmp_8662[2]];
    signal tmp_8664[3] <== CMul()(challengeQ, tmp_8663);
    signal tmp_8665[3] <== [evals[78][0] - publics[275], evals[78][1], evals[78][2]];
    signal tmp_8666[3] <== CMul()(evals[5], tmp_8665);
    signal tmp_8667[3] <== [tmp_8664[0] + tmp_8666[0], tmp_8664[1] + tmp_8666[1], tmp_8664[2] + tmp_8666[2]];
    signal tmp_8668[3] <== CMul()(challengeQ, tmp_8667);
    signal tmp_8669[3] <== [evals[79][0] - publics[276], evals[79][1], evals[79][2]];
    signal tmp_8670[3] <== CMul()(evals[5], tmp_8669);
    signal tmp_8671[3] <== [tmp_8668[0] + tmp_8670[0], tmp_8668[1] + tmp_8670[1], tmp_8668[2] + tmp_8670[2]];
    signal tmp_8672[3] <== CMul()(challengeQ, tmp_8671);
    signal tmp_8673[3] <== [evals[80][0] - publics[277], evals[80][1], evals[80][2]];
    signal tmp_8674[3] <== CMul()(evals[5], tmp_8673);
    signal tmp_8675[3] <== [tmp_8672[0] + tmp_8674[0], tmp_8672[1] + tmp_8674[1], tmp_8672[2] + tmp_8674[2]];
    signal tmp_8676[3] <== CMul()(challengeQ, tmp_8675);
    signal tmp_8677[3] <== [evals[81][0] - publics[278], evals[81][1], evals[81][2]];
    signal tmp_8678[3] <== CMul()(evals[5], tmp_8677);
    signal tmp_8679[3] <== [tmp_8676[0] + tmp_8678[0], tmp_8676[1] + tmp_8678[1], tmp_8676[2] + tmp_8678[2]];
    signal tmp_8680[3] <== CMul()(challengeQ, tmp_8679);
    signal tmp_8681[3] <== [evals[82][0] - publics[279], evals[82][1], evals[82][2]];
    signal tmp_8682[3] <== CMul()(evals[5], tmp_8681);
    signal tmp_8683[3] <== [tmp_8680[0] + tmp_8682[0], tmp_8680[1] + tmp_8682[1], tmp_8680[2] + tmp_8682[2]];
    signal tmp_8684[3] <== CMul()(challengeQ, tmp_8683);
    signal tmp_8685[3] <== [evals[83][0] - publics[280], evals[83][1], evals[83][2]];
    signal tmp_8686[3] <== CMul()(evals[5], tmp_8685);
    signal tmp_8687[3] <== [tmp_8684[0] + tmp_8686[0], tmp_8684[1] + tmp_8686[1], tmp_8684[2] + tmp_8686[2]];
    signal tmp_8688[3] <== CMul()(challengeQ, tmp_8687);
    signal tmp_8689[3] <== [evals[84][0] - publics[281], evals[84][1], evals[84][2]];
    signal tmp_8690[3] <== CMul()(evals[5], tmp_8689);
    signal tmp_8691[3] <== [tmp_8688[0] + tmp_8690[0], tmp_8688[1] + tmp_8690[1], tmp_8688[2] + tmp_8690[2]];
    signal tmp_8692[3] <== CMul()(challengeQ, tmp_8691);
    signal tmp_8693[3] <== [evals[85][0] - publics[282], evals[85][1], evals[85][2]];
    signal tmp_8694[3] <== CMul()(evals[5], tmp_8693);
    signal tmp_8695[3] <== [tmp_8692[0] + tmp_8694[0], tmp_8692[1] + tmp_8694[1], tmp_8692[2] + tmp_8694[2]];
    signal tmp_8696[3] <== CMul()(challengeQ, tmp_8695);
    signal tmp_8697[3] <== [evals[86][0] - publics[283], evals[86][1], evals[86][2]];
    signal tmp_8698[3] <== CMul()(evals[5], tmp_8697);
    signal tmp_8699[3] <== [tmp_8696[0] + tmp_8698[0], tmp_8696[1] + tmp_8698[1], tmp_8696[2] + tmp_8698[2]];
    signal tmp_8700[3] <== CMul()(challengeQ, tmp_8699);
    signal tmp_8701[3] <== [evals[87][0] - publics[284], evals[87][1], evals[87][2]];
    signal tmp_8702[3] <== CMul()(evals[5], tmp_8701);
    signal tmp_8703[3] <== [tmp_8700[0] + tmp_8702[0], tmp_8700[1] + tmp_8702[1], tmp_8700[2] + tmp_8702[2]];
    signal tmp_8704[3] <== CMul()(challengeQ, tmp_8703);
    signal tmp_8705[3] <== [evals[88][0] - publics[285], evals[88][1], evals[88][2]];
    signal tmp_8706[3] <== CMul()(evals[5], tmp_8705);
    signal tmp_8707[3] <== [tmp_8704[0] + tmp_8706[0], tmp_8704[1] + tmp_8706[1], tmp_8704[2] + tmp_8706[2]];
    signal tmp_8708[3] <== CMul()(challengeQ, tmp_8707);
    signal tmp_8709[3] <== [evals[89][0] - publics[286], evals[89][1], evals[89][2]];
    signal tmp_8710[3] <== CMul()(evals[5], tmp_8709);
    signal tmp_8711[3] <== [tmp_8708[0] + tmp_8710[0], tmp_8708[1] + tmp_8710[1], tmp_8708[2] + tmp_8710[2]];
    signal tmp_8712[3] <== CMul()(challengeQ, tmp_8711);
    signal tmp_8713[3] <== [evals[90][0] - publics[287], evals[90][1], evals[90][2]];
    signal tmp_8714[3] <== CMul()(evals[5], tmp_8713);
    signal tmp_8715[3] <== [tmp_8712[0] + tmp_8714[0], tmp_8712[1] + tmp_8714[1], tmp_8712[2] + tmp_8714[2]];
    signal tmp_8716[3] <== CMul()(challengeQ, tmp_8715);
    signal tmp_8717[3] <== [evals[67][0] - publics[288], evals[67][1], evals[67][2]];
    signal tmp_8718[3] <== CMul()(evals[4], tmp_8717);
    signal tmp_8719[3] <== [tmp_8716[0] + tmp_8718[0], tmp_8716[1] + tmp_8718[1], tmp_8716[2] + tmp_8718[2]];
    signal tmp_8720[3] <== CMul()(challengeQ, tmp_8719);
    signal tmp_8721[3] <== [evals[68][0] - publics[289], evals[68][1], evals[68][2]];
    signal tmp_8722[3] <== CMul()(evals[4], tmp_8721);
    signal tmp_8723[3] <== [tmp_8720[0] + tmp_8722[0], tmp_8720[1] + tmp_8722[1], tmp_8720[2] + tmp_8722[2]];
    signal tmp_8724[3] <== CMul()(challengeQ, tmp_8723);
    signal tmp_8725[3] <== [evals[69][0] - publics[290], evals[69][1], evals[69][2]];
    signal tmp_8726[3] <== CMul()(evals[4], tmp_8725);
    signal tmp_8727[3] <== [tmp_8724[0] + tmp_8726[0], tmp_8724[1] + tmp_8726[1], tmp_8724[2] + tmp_8726[2]];
    signal tmp_8728[3] <== CMul()(challengeQ, tmp_8727);
    signal tmp_8729[3] <== [evals[70][0] - publics[291], evals[70][1], evals[70][2]];
    signal tmp_8730[3] <== CMul()(evals[4], tmp_8729);
    signal tmp_8731[3] <== [tmp_8728[0] + tmp_8730[0], tmp_8728[1] + tmp_8730[1], tmp_8728[2] + tmp_8730[2]];
    signal tmp_8732[3] <== CMul()(challengeQ, tmp_8731);
    signal tmp_8733[3] <== [evals[71][0] - publics[292], evals[71][1], evals[71][2]];
    signal tmp_8734[3] <== CMul()(evals[4], tmp_8733);
    signal tmp_8735[3] <== [tmp_8732[0] + tmp_8734[0], tmp_8732[1] + tmp_8734[1], tmp_8732[2] + tmp_8734[2]];
    signal tmp_8736[3] <== CMul()(challengeQ, tmp_8735);
    signal tmp_8737[3] <== [evals[72][0] - publics[293], evals[72][1], evals[72][2]];
    signal tmp_8738[3] <== CMul()(evals[4], tmp_8737);
    signal tmp_8739[3] <== [tmp_8736[0] + tmp_8738[0], tmp_8736[1] + tmp_8738[1], tmp_8736[2] + tmp_8738[2]];
    signal tmp_8740[3] <== CMul()(challengeQ, tmp_8739);
    signal tmp_8741[3] <== [evals[73][0] - publics[294], evals[73][1], evals[73][2]];
    signal tmp_8742[3] <== CMul()(evals[4], tmp_8741);
    signal tmp_8743[3] <== [tmp_8740[0] + tmp_8742[0], tmp_8740[1] + tmp_8742[1], tmp_8740[2] + tmp_8742[2]];
    signal tmp_8744[3] <== CMul()(challengeQ, tmp_8743);
    signal tmp_8745[3] <== [evals[74][0] - publics[295], evals[74][1], evals[74][2]];
    signal tmp_8746[3] <== CMul()(evals[4], tmp_8745);
    signal tmp_8747[3] <== [tmp_8744[0] + tmp_8746[0], tmp_8744[1] + tmp_8746[1], tmp_8744[2] + tmp_8746[2]];
    signal tmp_8748[3] <== CMul()(challengeQ, tmp_8747);
    signal tmp_8749[3] <== [evals[75][0] - publics[296], evals[75][1], evals[75][2]];
    signal tmp_8750[3] <== CMul()(evals[4], tmp_8749);
    signal tmp_8751[3] <== [tmp_8748[0] + tmp_8750[0], tmp_8748[1] + tmp_8750[1], tmp_8748[2] + tmp_8750[2]];
    signal tmp_8752[3] <== CMul()(challengeQ, tmp_8751);
    signal tmp_8753[3] <== [evals[76][0] - publics[297], evals[76][1], evals[76][2]];
    signal tmp_8754[3] <== CMul()(evals[4], tmp_8753);
    signal tmp_8755[3] <== [tmp_8752[0] + tmp_8754[0], tmp_8752[1] + tmp_8754[1], tmp_8752[2] + tmp_8754[2]];
    signal tmp_8756[3] <== CMul()(challengeQ, tmp_8755);
    signal tmp_8757[3] <== [evals[77][0] - publics[298], evals[77][1], evals[77][2]];
    signal tmp_8758[3] <== CMul()(evals[4], tmp_8757);
    signal tmp_8759[3] <== [tmp_8756[0] + tmp_8758[0], tmp_8756[1] + tmp_8758[1], tmp_8756[2] + tmp_8758[2]];
    signal tmp_8760[3] <== CMul()(challengeQ, tmp_8759);
    signal tmp_8761[3] <== [evals[78][0] - publics[299], evals[78][1], evals[78][2]];
    signal tmp_8762[3] <== CMul()(evals[4], tmp_8761);
    signal tmp_8763[3] <== [tmp_8760[0] + tmp_8762[0], tmp_8760[1] + tmp_8762[1], tmp_8760[2] + tmp_8762[2]];
    signal tmp_8764[3] <== CMul()(challengeQ, tmp_8763);
    signal tmp_8765[3] <== [evals[79][0] - publics[300], evals[79][1], evals[79][2]];
    signal tmp_8766[3] <== CMul()(evals[4], tmp_8765);
    signal tmp_8767[3] <== [tmp_8764[0] + tmp_8766[0], tmp_8764[1] + tmp_8766[1], tmp_8764[2] + tmp_8766[2]];
    signal tmp_8768[3] <== CMul()(challengeQ, tmp_8767);
    signal tmp_8769[3] <== [evals[80][0] - publics[301], evals[80][1], evals[80][2]];
    signal tmp_8770[3] <== CMul()(evals[4], tmp_8769);
    signal tmp_8771[3] <== [tmp_8768[0] + tmp_8770[0], tmp_8768[1] + tmp_8770[1], tmp_8768[2] + tmp_8770[2]];
    signal tmp_8772[3] <== CMul()(challengeQ, tmp_8771);
    signal tmp_8773[3] <== [evals[81][0] - publics[302], evals[81][1], evals[81][2]];
    signal tmp_8774[3] <== CMul()(evals[4], tmp_8773);
    signal tmp_8775[3] <== [tmp_8772[0] + tmp_8774[0], tmp_8772[1] + tmp_8774[1], tmp_8772[2] + tmp_8774[2]];
    signal tmp_8776[3] <== CMul()(challengeQ, tmp_8775);
    signal tmp_8777[3] <== [evals[82][0] - publics[303], evals[82][1], evals[82][2]];
    signal tmp_8778[3] <== CMul()(evals[4], tmp_8777);
    signal tmp_8779[3] <== [tmp_8776[0] + tmp_8778[0], tmp_8776[1] + tmp_8778[1], tmp_8776[2] + tmp_8778[2]];
    signal tmp_8780[3] <== CMul()(challengeQ, tmp_8779);
    signal tmp_8781[3] <== [evals[83][0] - publics[304], evals[83][1], evals[83][2]];
    signal tmp_8782[3] <== CMul()(evals[4], tmp_8781);
    signal tmp_8783[3] <== [tmp_8780[0] + tmp_8782[0], tmp_8780[1] + tmp_8782[1], tmp_8780[2] + tmp_8782[2]];
    signal tmp_8784[3] <== CMul()(challengeQ, tmp_8783);
    signal tmp_8785[3] <== [evals[84][0] - publics[305], evals[84][1], evals[84][2]];
    signal tmp_8786[3] <== CMul()(evals[4], tmp_8785);
    signal tmp_8787[3] <== [tmp_8784[0] + tmp_8786[0], tmp_8784[1] + tmp_8786[1], tmp_8784[2] + tmp_8786[2]];
    signal tmp_8788[3] <== CMul()(challengeQ, tmp_8787);
    signal tmp_8789[3] <== [evals[85][0] - publics[306], evals[85][1], evals[85][2]];
    signal tmp_8790[3] <== CMul()(evals[4], tmp_8789);
    signal tmp_8791[3] <== [tmp_8788[0] + tmp_8790[0], tmp_8788[1] + tmp_8790[1], tmp_8788[2] + tmp_8790[2]];
    signal tmp_8792[3] <== CMul()(challengeQ, tmp_8791);
    signal tmp_8793[3] <== [evals[86][0] - publics[307], evals[86][1], evals[86][2]];
    signal tmp_8794[3] <== CMul()(evals[4], tmp_8793);
    signal tmp_8795[3] <== [tmp_8792[0] + tmp_8794[0], tmp_8792[1] + tmp_8794[1], tmp_8792[2] + tmp_8794[2]];
    signal tmp_8796[3] <== CMul()(challengeQ, tmp_8795);
    signal tmp_8797[3] <== [evals[87][0] - publics[308], evals[87][1], evals[87][2]];
    signal tmp_8798[3] <== CMul()(evals[4], tmp_8797);
    signal tmp_8799[3] <== [tmp_8796[0] + tmp_8798[0], tmp_8796[1] + tmp_8798[1], tmp_8796[2] + tmp_8798[2]];
    signal tmp_8800[3] <== CMul()(challengeQ, tmp_8799);
    signal tmp_8801[3] <== [evals[88][0] - publics[309], evals[88][1], evals[88][2]];
    signal tmp_8802[3] <== CMul()(evals[4], tmp_8801);
    signal tmp_8803[3] <== [tmp_8800[0] + tmp_8802[0], tmp_8800[1] + tmp_8802[1], tmp_8800[2] + tmp_8802[2]];
    signal tmp_8804[3] <== CMul()(challengeQ, tmp_8803);
    signal tmp_8805[3] <== [evals[89][0] - publics[310], evals[89][1], evals[89][2]];
    signal tmp_8806[3] <== CMul()(evals[4], tmp_8805);
    signal tmp_8807[3] <== [tmp_8804[0] + tmp_8806[0], tmp_8804[1] + tmp_8806[1], tmp_8804[2] + tmp_8806[2]];
    signal tmp_8808[3] <== CMul()(challengeQ, tmp_8807);
    signal tmp_8809[3] <== [evals[90][0] - publics[311], evals[90][1], evals[90][2]];
    signal tmp_8810[3] <== CMul()(evals[4], tmp_8809);
    signal tmp_8811[3] <== [tmp_8808[0] + tmp_8810[0], tmp_8808[1] + tmp_8810[1], tmp_8808[2] + tmp_8810[2]];
    signal tmp_8812[3] <== CMul()(challengeQ, tmp_8811);
    signal tmp_8813[3] <== [evals[67][0] - publics[312], evals[67][1], evals[67][2]];
    signal tmp_8814[3] <== CMul()(evals[3], tmp_8813);
    signal tmp_8815[3] <== [tmp_8812[0] + tmp_8814[0], tmp_8812[1] + tmp_8814[1], tmp_8812[2] + tmp_8814[2]];
    signal tmp_8816[3] <== CMul()(challengeQ, tmp_8815);
    signal tmp_8817[3] <== [evals[68][0] - publics[313], evals[68][1], evals[68][2]];
    signal tmp_8818[3] <== CMul()(evals[3], tmp_8817);
    signal tmp_8819[3] <== [tmp_8816[0] + tmp_8818[0], tmp_8816[1] + tmp_8818[1], tmp_8816[2] + tmp_8818[2]];
    signal tmp_8820[3] <== CMul()(challengeQ, tmp_8819);
    signal tmp_8821[3] <== [evals[69][0] - publics[314], evals[69][1], evals[69][2]];
    signal tmp_8822[3] <== CMul()(evals[3], tmp_8821);
    signal tmp_8823[3] <== [tmp_8820[0] + tmp_8822[0], tmp_8820[1] + tmp_8822[1], tmp_8820[2] + tmp_8822[2]];
    signal tmp_8824[3] <== CMul()(challengeQ, tmp_8823);
    signal tmp_8825[3] <== [evals[70][0] - publics[315], evals[70][1], evals[70][2]];
    signal tmp_8826[3] <== CMul()(evals[3], tmp_8825);
    signal tmp_8827[3] <== [tmp_8824[0] + tmp_8826[0], tmp_8824[1] + tmp_8826[1], tmp_8824[2] + tmp_8826[2]];
    signal tmp_8828[3] <== CMul()(challengeQ, tmp_8827);
    signal tmp_8829[3] <== [evals[71][0] - publics[316], evals[71][1], evals[71][2]];
    signal tmp_8830[3] <== CMul()(evals[3], tmp_8829);
    signal tmp_8831[3] <== [tmp_8828[0] + tmp_8830[0], tmp_8828[1] + tmp_8830[1], tmp_8828[2] + tmp_8830[2]];
    signal tmp_8832[3] <== CMul()(challengeQ, tmp_8831);
    signal tmp_8833[3] <== [evals[72][0] - publics[317], evals[72][1], evals[72][2]];
    signal tmp_8834[3] <== CMul()(evals[3], tmp_8833);
    signal tmp_8835[3] <== [tmp_8832[0] + tmp_8834[0], tmp_8832[1] + tmp_8834[1], tmp_8832[2] + tmp_8834[2]];
    signal tmp_8836[3] <== CMul()(challengeQ, tmp_8835);
    signal tmp_8837[3] <== [evals[73][0] - publics[318], evals[73][1], evals[73][2]];
    signal tmp_8838[3] <== CMul()(evals[3], tmp_8837);
    signal tmp_8839[3] <== [tmp_8836[0] + tmp_8838[0], tmp_8836[1] + tmp_8838[1], tmp_8836[2] + tmp_8838[2]];
    signal tmp_8840[3] <== CMul()(challengeQ, tmp_8839);
    signal tmp_8841[3] <== [evals[74][0] - publics[319], evals[74][1], evals[74][2]];
    signal tmp_8842[3] <== CMul()(evals[3], tmp_8841);
    signal tmp_8843[3] <== [tmp_8840[0] + tmp_8842[0], tmp_8840[1] + tmp_8842[1], tmp_8840[2] + tmp_8842[2]];
    signal tmp_8844[3] <== CMul()(challengeQ, tmp_8843);
    signal tmp_8845[3] <== [evals[75][0] - publics[320], evals[75][1], evals[75][2]];
    signal tmp_8846[3] <== CMul()(evals[3], tmp_8845);
    signal tmp_8847[3] <== [tmp_8844[0] + tmp_8846[0], tmp_8844[1] + tmp_8846[1], tmp_8844[2] + tmp_8846[2]];
    signal tmp_8848[3] <== CMul()(challengeQ, tmp_8847);
    signal tmp_8849[3] <== [evals[76][0] - publics[321], evals[76][1], evals[76][2]];
    signal tmp_8850[3] <== CMul()(evals[3], tmp_8849);
    signal tmp_8851[3] <== [tmp_8848[0] + tmp_8850[0], tmp_8848[1] + tmp_8850[1], tmp_8848[2] + tmp_8850[2]];
    signal tmp_8852[3] <== CMul()(challengeQ, tmp_8851);
    signal tmp_8853[3] <== [evals[77][0] - publics[322], evals[77][1], evals[77][2]];
    signal tmp_8854[3] <== CMul()(evals[3], tmp_8853);
    signal tmp_8855[3] <== [tmp_8852[0] + tmp_8854[0], tmp_8852[1] + tmp_8854[1], tmp_8852[2] + tmp_8854[2]];
    signal tmp_8856[3] <== CMul()(challengeQ, tmp_8855);
    signal tmp_8857[3] <== [evals[78][0] - publics[323], evals[78][1], evals[78][2]];
    signal tmp_8858[3] <== CMul()(evals[3], tmp_8857);
    signal tmp_8859[3] <== [tmp_8856[0] + tmp_8858[0], tmp_8856[1] + tmp_8858[1], tmp_8856[2] + tmp_8858[2]];
    signal tmp_8860[3] <== CMul()(challengeQ, tmp_8859);
    signal tmp_8861[3] <== [evals[79][0] - publics[324], evals[79][1], evals[79][2]];
    signal tmp_8862[3] <== CMul()(evals[3], tmp_8861);
    signal tmp_8863[3] <== [tmp_8860[0] + tmp_8862[0], tmp_8860[1] + tmp_8862[1], tmp_8860[2] + tmp_8862[2]];
    signal tmp_8864[3] <== CMul()(challengeQ, tmp_8863);
    signal tmp_8865[3] <== [evals[80][0] - publics[325], evals[80][1], evals[80][2]];
    signal tmp_8866[3] <== CMul()(evals[3], tmp_8865);
    signal tmp_8867[3] <== [tmp_8864[0] + tmp_8866[0], tmp_8864[1] + tmp_8866[1], tmp_8864[2] + tmp_8866[2]];
    signal tmp_8868[3] <== CMul()(challengeQ, tmp_8867);
    signal tmp_8869[3] <== [evals[81][0] - publics[326], evals[81][1], evals[81][2]];
    signal tmp_8870[3] <== CMul()(evals[3], tmp_8869);
    signal tmp_8871[3] <== [tmp_8868[0] + tmp_8870[0], tmp_8868[1] + tmp_8870[1], tmp_8868[2] + tmp_8870[2]];
    signal tmp_8872[3] <== CMul()(challengeQ, tmp_8871);
    signal tmp_8873[3] <== [evals[82][0] - publics[327], evals[82][1], evals[82][2]];
    signal tmp_8874[3] <== CMul()(evals[3], tmp_8873);
    signal tmp_8875[3] <== [tmp_8872[0] + tmp_8874[0], tmp_8872[1] + tmp_8874[1], tmp_8872[2] + tmp_8874[2]];
    signal tmp_8876[3] <== CMul()(challengeQ, tmp_8875);
    signal tmp_8877[3] <== [evals[83][0] - publics[328], evals[83][1], evals[83][2]];
    signal tmp_8878[3] <== CMul()(evals[3], tmp_8877);
    signal tmp_8879[3] <== [tmp_8876[0] + tmp_8878[0], tmp_8876[1] + tmp_8878[1], tmp_8876[2] + tmp_8878[2]];
    signal tmp_8880[3] <== CMul()(challengeQ, tmp_8879);
    signal tmp_8881[3] <== [evals[84][0] - publics[329], evals[84][1], evals[84][2]];
    signal tmp_8882[3] <== CMul()(evals[3], tmp_8881);
    signal tmp_8883[3] <== [tmp_8880[0] + tmp_8882[0], tmp_8880[1] + tmp_8882[1], tmp_8880[2] + tmp_8882[2]];
    signal tmp_8884[3] <== CMul()(challengeQ, tmp_8883);
    signal tmp_8885[3] <== [evals[85][0] - publics[330], evals[85][1], evals[85][2]];
    signal tmp_8886[3] <== CMul()(evals[3], tmp_8885);
    signal tmp_8887[3] <== [tmp_8884[0] + tmp_8886[0], tmp_8884[1] + tmp_8886[1], tmp_8884[2] + tmp_8886[2]];
    signal tmp_8888[3] <== CMul()(challengeQ, tmp_8887);
    signal tmp_8889[3] <== [evals[86][0] - publics[331], evals[86][1], evals[86][2]];
    signal tmp_8890[3] <== CMul()(evals[3], tmp_8889);
    signal tmp_8891[3] <== [tmp_8888[0] + tmp_8890[0], tmp_8888[1] + tmp_8890[1], tmp_8888[2] + tmp_8890[2]];
    signal tmp_8892[3] <== CMul()(challengeQ, tmp_8891);
    signal tmp_8893[3] <== [evals[87][0] - publics[332], evals[87][1], evals[87][2]];
    signal tmp_8894[3] <== CMul()(evals[3], tmp_8893);
    signal tmp_8895[3] <== [tmp_8892[0] + tmp_8894[0], tmp_8892[1] + tmp_8894[1], tmp_8892[2] + tmp_8894[2]];
    signal tmp_8896[3] <== CMul()(challengeQ, tmp_8895);
    signal tmp_8897[3] <== [evals[88][0] - publics[333], evals[88][1], evals[88][2]];
    signal tmp_8898[3] <== CMul()(evals[3], tmp_8897);
    signal tmp_8899[3] <== [tmp_8896[0] + tmp_8898[0], tmp_8896[1] + tmp_8898[1], tmp_8896[2] + tmp_8898[2]];
    signal tmp_8900[3] <== CMul()(challengeQ, tmp_8899);
    signal tmp_8901[3] <== [evals[89][0] - publics[334], evals[89][1], evals[89][2]];
    signal tmp_8902[3] <== CMul()(evals[3], tmp_8901);
    signal tmp_8903[3] <== [tmp_8900[0] + tmp_8902[0], tmp_8900[1] + tmp_8902[1], tmp_8900[2] + tmp_8902[2]];
    signal tmp_8904[3] <== CMul()(challengeQ, tmp_8903);
    signal tmp_8905[3] <== [evals[90][0] - publics[335], evals[90][1], evals[90][2]];
    signal tmp_8906[3] <== CMul()(evals[3], tmp_8905);
    signal tmp_8907[3] <== [tmp_8904[0] + tmp_8906[0], tmp_8904[1] + tmp_8906[1], tmp_8904[2] + tmp_8906[2]];
    signal tmp_8908[3] <== CMul()(challengeQ, tmp_8907);
    signal tmp_8909[3] <== [evals[67][0] - publics[336], evals[67][1], evals[67][2]];
    signal tmp_8910[3] <== CMul()(evals[2], tmp_8909);
    signal tmp_8911[3] <== [tmp_8908[0] + tmp_8910[0], tmp_8908[1] + tmp_8910[1], tmp_8908[2] + tmp_8910[2]];
    signal tmp_8912[3] <== CMul()(challengeQ, tmp_8911);
    signal tmp_8913[3] <== [evals[68][0] - publics[337], evals[68][1], evals[68][2]];
    signal tmp_8914[3] <== CMul()(evals[2], tmp_8913);
    signal tmp_8915[3] <== [tmp_8912[0] + tmp_8914[0], tmp_8912[1] + tmp_8914[1], tmp_8912[2] + tmp_8914[2]];
    signal tmp_8916[3] <== CMul()(challengeQ, tmp_8915);
    signal tmp_8917[3] <== [evals[69][0] - publics[338], evals[69][1], evals[69][2]];
    signal tmp_8918[3] <== CMul()(evals[2], tmp_8917);
    signal tmp_8919[3] <== [tmp_8916[0] + tmp_8918[0], tmp_8916[1] + tmp_8918[1], tmp_8916[2] + tmp_8918[2]];
    signal tmp_8920[3] <== CMul()(challengeQ, tmp_8919);
    signal tmp_8921[3] <== [evals[70][0] - publics[339], evals[70][1], evals[70][2]];
    signal tmp_8922[3] <== CMul()(evals[2], tmp_8921);
    signal tmp_8923[3] <== [tmp_8920[0] + tmp_8922[0], tmp_8920[1] + tmp_8922[1], tmp_8920[2] + tmp_8922[2]];
    signal tmp_8924[3] <== CMul()(challengeQ, tmp_8923);
    signal tmp_8925[3] <== [evals[71][0] - publics[340], evals[71][1], evals[71][2]];
    signal tmp_8926[3] <== CMul()(evals[2], tmp_8925);
    signal tmp_8927[3] <== [tmp_8924[0] + tmp_8926[0], tmp_8924[1] + tmp_8926[1], tmp_8924[2] + tmp_8926[2]];
    signal tmp_8928[3] <== CMul()(challengeQ, tmp_8927);
    signal tmp_8929[3] <== [evals[72][0] - publics[341], evals[72][1], evals[72][2]];
    signal tmp_8930[3] <== CMul()(evals[2], tmp_8929);
    signal tmp_8931[3] <== [tmp_8928[0] + tmp_8930[0], tmp_8928[1] + tmp_8930[1], tmp_8928[2] + tmp_8930[2]];
    signal tmp_8932[3] <== CMul()(challengeQ, tmp_8931);
    signal tmp_8933[3] <== [evals[73][0] - publics[342], evals[73][1], evals[73][2]];
    signal tmp_8934[3] <== CMul()(evals[2], tmp_8933);
    signal tmp_8935[3] <== [tmp_8932[0] + tmp_8934[0], tmp_8932[1] + tmp_8934[1], tmp_8932[2] + tmp_8934[2]];
    signal tmp_8936[3] <== CMul()(challengeQ, tmp_8935);
    signal tmp_8937[3] <== [evals[74][0] - publics[343], evals[74][1], evals[74][2]];
    signal tmp_8938[3] <== CMul()(evals[2], tmp_8937);
    signal tmp_8939[3] <== [tmp_8936[0] + tmp_8938[0], tmp_8936[1] + tmp_8938[1], tmp_8936[2] + tmp_8938[2]];
    signal tmp_8940[3] <== CMul()(challengeQ, tmp_8939);
    signal tmp_8941[3] <== [evals[75][0] - publics[344], evals[75][1], evals[75][2]];
    signal tmp_8942[3] <== CMul()(evals[2], tmp_8941);
    signal tmp_8943[3] <== [tmp_8940[0] + tmp_8942[0], tmp_8940[1] + tmp_8942[1], tmp_8940[2] + tmp_8942[2]];
    signal tmp_8944[3] <== CMul()(challengeQ, tmp_8943);
    signal tmp_8945[3] <== [evals[76][0] - publics[345], evals[76][1], evals[76][2]];
    signal tmp_8946[3] <== CMul()(evals[2], tmp_8945);
    signal tmp_8947[3] <== [tmp_8944[0] + tmp_8946[0], tmp_8944[1] + tmp_8946[1], tmp_8944[2] + tmp_8946[2]];
    signal tmp_8948[3] <== CMul()(challengeQ, tmp_8947);
    signal tmp_8949[3] <== [evals[77][0] - publics[346], evals[77][1], evals[77][2]];
    signal tmp_8950[3] <== CMul()(evals[2], tmp_8949);
    signal tmp_8951[3] <== [tmp_8948[0] + tmp_8950[0], tmp_8948[1] + tmp_8950[1], tmp_8948[2] + tmp_8950[2]];
    signal tmp_8952[3] <== CMul()(challengeQ, tmp_8951);
    signal tmp_8953[3] <== [evals[78][0] - publics[347], evals[78][1], evals[78][2]];
    signal tmp_8954[3] <== CMul()(evals[2], tmp_8953);
    signal tmp_8955[3] <== [tmp_8952[0] + tmp_8954[0], tmp_8952[1] + tmp_8954[1], tmp_8952[2] + tmp_8954[2]];
    signal tmp_8956[3] <== CMul()(challengeQ, tmp_8955);
    signal tmp_8957[3] <== [evals[79][0] - publics[348], evals[79][1], evals[79][2]];
    signal tmp_8958[3] <== CMul()(evals[2], tmp_8957);
    signal tmp_8959[3] <== [tmp_8956[0] + tmp_8958[0], tmp_8956[1] + tmp_8958[1], tmp_8956[2] + tmp_8958[2]];
    signal tmp_8960[3] <== CMul()(challengeQ, tmp_8959);
    signal tmp_8961[3] <== [evals[80][0] - publics[349], evals[80][1], evals[80][2]];
    signal tmp_8962[3] <== CMul()(evals[2], tmp_8961);
    signal tmp_8963[3] <== [tmp_8960[0] + tmp_8962[0], tmp_8960[1] + tmp_8962[1], tmp_8960[2] + tmp_8962[2]];
    signal tmp_8964[3] <== CMul()(challengeQ, tmp_8963);
    signal tmp_8965[3] <== [evals[81][0] - publics[350], evals[81][1], evals[81][2]];
    signal tmp_8966[3] <== CMul()(evals[2], tmp_8965);
    signal tmp_8967[3] <== [tmp_8964[0] + tmp_8966[0], tmp_8964[1] + tmp_8966[1], tmp_8964[2] + tmp_8966[2]];
    signal tmp_8968[3] <== CMul()(challengeQ, tmp_8967);
    signal tmp_8969[3] <== [evals[82][0] - publics[351], evals[82][1], evals[82][2]];
    signal tmp_8970[3] <== CMul()(evals[2], tmp_8969);
    signal tmp_8971[3] <== [tmp_8968[0] + tmp_8970[0], tmp_8968[1] + tmp_8970[1], tmp_8968[2] + tmp_8970[2]];
    signal tmp_8972[3] <== CMul()(challengeQ, tmp_8971);
    signal tmp_8973[3] <== [evals[83][0] - publics[352], evals[83][1], evals[83][2]];
    signal tmp_8974[3] <== CMul()(evals[2], tmp_8973);
    signal tmp_8975[3] <== [tmp_8972[0] + tmp_8974[0], tmp_8972[1] + tmp_8974[1], tmp_8972[2] + tmp_8974[2]];
    signal tmp_8976[3] <== CMul()(challengeQ, tmp_8975);
    signal tmp_8977[3] <== [evals[84][0] - publics[353], evals[84][1], evals[84][2]];
    signal tmp_8978[3] <== CMul()(evals[2], tmp_8977);
    signal tmp_8979[3] <== [tmp_8976[0] + tmp_8978[0], tmp_8976[1] + tmp_8978[1], tmp_8976[2] + tmp_8978[2]];
    signal tmp_8980[3] <== CMul()(challengeQ, tmp_8979);
    signal tmp_8981[3] <== [evals[85][0] - publics[354], evals[85][1], evals[85][2]];
    signal tmp_8982[3] <== CMul()(evals[2], tmp_8981);
    signal tmp_8983[3] <== [tmp_8980[0] + tmp_8982[0], tmp_8980[1] + tmp_8982[1], tmp_8980[2] + tmp_8982[2]];
    signal tmp_8984[3] <== CMul()(challengeQ, tmp_8983);
    signal tmp_8985[3] <== [evals[86][0] - publics[355], evals[86][1], evals[86][2]];
    signal tmp_8986[3] <== CMul()(evals[2], tmp_8985);
    signal tmp_8987[3] <== [tmp_8984[0] + tmp_8986[0], tmp_8984[1] + tmp_8986[1], tmp_8984[2] + tmp_8986[2]];
    signal tmp_8988[3] <== CMul()(challengeQ, tmp_8987);
    signal tmp_8989[3] <== [evals[87][0] - publics[356], evals[87][1], evals[87][2]];
    signal tmp_8990[3] <== CMul()(evals[2], tmp_8989);
    signal tmp_8991[3] <== [tmp_8988[0] + tmp_8990[0], tmp_8988[1] + tmp_8990[1], tmp_8988[2] + tmp_8990[2]];
    signal tmp_8992[3] <== CMul()(challengeQ, tmp_8991);
    signal tmp_8993[3] <== [evals[88][0] - publics[357], evals[88][1], evals[88][2]];
    signal tmp_8994[3] <== CMul()(evals[2], tmp_8993);
    signal tmp_8995[3] <== [tmp_8992[0] + tmp_8994[0], tmp_8992[1] + tmp_8994[1], tmp_8992[2] + tmp_8994[2]];
    signal tmp_8996[3] <== CMul()(challengeQ, tmp_8995);
    signal tmp_8997[3] <== [evals[89][0] - publics[358], evals[89][1], evals[89][2]];
    signal tmp_8998[3] <== CMul()(evals[2], tmp_8997);
    signal tmp_8999[3] <== [tmp_8996[0] + tmp_8998[0], tmp_8996[1] + tmp_8998[1], tmp_8996[2] + tmp_8998[2]];
    signal tmp_9000[3] <== CMul()(challengeQ, tmp_8999);
    signal tmp_9001[3] <== [evals[90][0] - publics[359], evals[90][1], evals[90][2]];
    signal tmp_9002[3] <== CMul()(evals[2], tmp_9001);
    signal tmp_9003[3] <== [tmp_9000[0] + tmp_9002[0], tmp_9000[1] + tmp_9002[1], tmp_9000[2] + tmp_9002[2]];
    signal tmp_9004[3] <== CMul()(challengeQ, tmp_9003);
    signal tmp_9005[3] <== [evals[67][0] - publics[360], evals[67][1], evals[67][2]];
    signal tmp_9006[3] <== CMul()(evals[1], tmp_9005);
    signal tmp_9007[3] <== [tmp_9004[0] + tmp_9006[0], tmp_9004[1] + tmp_9006[1], tmp_9004[2] + tmp_9006[2]];
    signal tmp_9008[3] <== CMul()(challengeQ, tmp_9007);
    signal tmp_9009[3] <== [evals[68][0] - publics[361], evals[68][1], evals[68][2]];
    signal tmp_9010[3] <== CMul()(evals[1], tmp_9009);
    signal tmp_9011[3] <== [tmp_9008[0] + tmp_9010[0], tmp_9008[1] + tmp_9010[1], tmp_9008[2] + tmp_9010[2]];
    signal tmp_9012[3] <== CMul()(challengeQ, tmp_9011);
    signal tmp_9013[3] <== [evals[69][0] - publics[362], evals[69][1], evals[69][2]];
    signal tmp_9014[3] <== CMul()(evals[1], tmp_9013);
    signal tmp_9015[3] <== [tmp_9012[0] + tmp_9014[0], tmp_9012[1] + tmp_9014[1], tmp_9012[2] + tmp_9014[2]];
    signal tmp_9016[3] <== CMul()(challengeQ, tmp_9015);
    signal tmp_9017[3] <== [evals[70][0] - publics[363], evals[70][1], evals[70][2]];
    signal tmp_9018[3] <== CMul()(evals[1], tmp_9017);
    signal tmp_9019[3] <== [tmp_9016[0] + tmp_9018[0], tmp_9016[1] + tmp_9018[1], tmp_9016[2] + tmp_9018[2]];
    signal tmp_9020[3] <== CMul()(challengeQ, tmp_9019);
    signal tmp_9021[3] <== [evals[71][0] - publics[364], evals[71][1], evals[71][2]];
    signal tmp_9022[3] <== CMul()(evals[1], tmp_9021);
    signal tmp_9023[3] <== [tmp_9020[0] + tmp_9022[0], tmp_9020[1] + tmp_9022[1], tmp_9020[2] + tmp_9022[2]];
    signal tmp_9024[3] <== CMul()(challengeQ, tmp_9023);
    signal tmp_9025[3] <== [evals[72][0] - publics[365], evals[72][1], evals[72][2]];
    signal tmp_9026[3] <== CMul()(evals[1], tmp_9025);
    signal tmp_9027[3] <== [tmp_9024[0] + tmp_9026[0], tmp_9024[1] + tmp_9026[1], tmp_9024[2] + tmp_9026[2]];
    signal tmp_9028[3] <== CMul()(challengeQ, tmp_9027);
    signal tmp_9029[3] <== [evals[73][0] - publics[366], evals[73][1], evals[73][2]];
    signal tmp_9030[3] <== CMul()(evals[1], tmp_9029);
    signal tmp_9031[3] <== [tmp_9028[0] + tmp_9030[0], tmp_9028[1] + tmp_9030[1], tmp_9028[2] + tmp_9030[2]];
    signal tmp_9032[3] <== CMul()(challengeQ, tmp_9031);
    signal tmp_9033[3] <== [evals[74][0] - publics[367], evals[74][1], evals[74][2]];
    signal tmp_9034[3] <== CMul()(evals[1], tmp_9033);
    signal tmp_9035[3] <== [tmp_9032[0] + tmp_9034[0], tmp_9032[1] + tmp_9034[1], tmp_9032[2] + tmp_9034[2]];
    signal tmp_9036[3] <== CMul()(challengeQ, tmp_9035);
    signal tmp_9037[3] <== [evals[75][0] - publics[368], evals[75][1], evals[75][2]];
    signal tmp_9038[3] <== CMul()(evals[1], tmp_9037);
    signal tmp_9039[3] <== [tmp_9036[0] + tmp_9038[0], tmp_9036[1] + tmp_9038[1], tmp_9036[2] + tmp_9038[2]];
    signal tmp_9040[3] <== CMul()(challengeQ, tmp_9039);
    signal tmp_9041[3] <== [evals[76][0] - publics[369], evals[76][1], evals[76][2]];
    signal tmp_9042[3] <== CMul()(evals[1], tmp_9041);
    signal tmp_9043[3] <== [tmp_9040[0] + tmp_9042[0], tmp_9040[1] + tmp_9042[1], tmp_9040[2] + tmp_9042[2]];
    signal tmp_9044[3] <== CMul()(challengeQ, tmp_9043);
    signal tmp_9045[3] <== [evals[77][0] - publics[370], evals[77][1], evals[77][2]];
    signal tmp_9046[3] <== CMul()(evals[1], tmp_9045);
    signal tmp_9047[3] <== [tmp_9044[0] + tmp_9046[0], tmp_9044[1] + tmp_9046[1], tmp_9044[2] + tmp_9046[2]];
    signal tmp_9048[3] <== CMul()(challengeQ, tmp_9047);
    signal tmp_9049[3] <== [evals[78][0] - publics[371], evals[78][1], evals[78][2]];
    signal tmp_9050[3] <== CMul()(evals[1], tmp_9049);
    signal tmp_9051[3] <== [tmp_9048[0] + tmp_9050[0], tmp_9048[1] + tmp_9050[1], tmp_9048[2] + tmp_9050[2]];
    signal tmp_9052[3] <== CMul()(challengeQ, tmp_9051);
    signal tmp_9053[3] <== [evals[79][0] - publics[372], evals[79][1], evals[79][2]];
    signal tmp_9054[3] <== CMul()(evals[1], tmp_9053);
    signal tmp_9055[3] <== [tmp_9052[0] + tmp_9054[0], tmp_9052[1] + tmp_9054[1], tmp_9052[2] + tmp_9054[2]];
    signal tmp_9056[3] <== CMul()(challengeQ, tmp_9055);
    signal tmp_9057[3] <== [evals[80][0] - publics[373], evals[80][1], evals[80][2]];
    signal tmp_9058[3] <== CMul()(evals[1], tmp_9057);
    signal tmp_9059[3] <== [tmp_9056[0] + tmp_9058[0], tmp_9056[1] + tmp_9058[1], tmp_9056[2] + tmp_9058[2]];
    signal tmp_9060[3] <== CMul()(challengeQ, tmp_9059);
    signal tmp_9061[3] <== [evals[81][0] - publics[374], evals[81][1], evals[81][2]];
    signal tmp_9062[3] <== CMul()(evals[1], tmp_9061);
    signal tmp_9063[3] <== [tmp_9060[0] + tmp_9062[0], tmp_9060[1] + tmp_9062[1], tmp_9060[2] + tmp_9062[2]];
    signal tmp_9064[3] <== CMul()(challengeQ, tmp_9063);
    signal tmp_9065[3] <== [evals[82][0] - publics[375], evals[82][1], evals[82][2]];
    signal tmp_9066[3] <== CMul()(evals[1], tmp_9065);
    signal tmp_9067[3] <== [tmp_9064[0] + tmp_9066[0], tmp_9064[1] + tmp_9066[1], tmp_9064[2] + tmp_9066[2]];
    signal tmp_9068[3] <== CMul()(challengeQ, tmp_9067);
    signal tmp_9069[3] <== [evals[83][0] - publics[376], evals[83][1], evals[83][2]];
    signal tmp_9070[3] <== CMul()(evals[1], tmp_9069);
    signal tmp_9071[3] <== [tmp_9068[0] + tmp_9070[0], tmp_9068[1] + tmp_9070[1], tmp_9068[2] + tmp_9070[2]];
    signal tmp_9072[3] <== CMul()(challengeQ, tmp_9071);
    signal tmp_9073[3] <== [evals[84][0] - publics[377], evals[84][1], evals[84][2]];
    signal tmp_9074[3] <== CMul()(evals[1], tmp_9073);
    signal tmp_9075[3] <== [tmp_9072[0] + tmp_9074[0], tmp_9072[1] + tmp_9074[1], tmp_9072[2] + tmp_9074[2]];
    signal tmp_9076[3] <== CMul()(challengeQ, tmp_9075);
    signal tmp_9077[3] <== [evals[85][0] - publics[378], evals[85][1], evals[85][2]];
    signal tmp_9078[3] <== CMul()(evals[1], tmp_9077);
    signal tmp_9079[3] <== [tmp_9076[0] + tmp_9078[0], tmp_9076[1] + tmp_9078[1], tmp_9076[2] + tmp_9078[2]];
    signal tmp_9080[3] <== CMul()(challengeQ, tmp_9079);
    signal tmp_9081[3] <== [evals[86][0] - publics[379], evals[86][1], evals[86][2]];
    signal tmp_9082[3] <== CMul()(evals[1], tmp_9081);
    signal tmp_9083[3] <== [tmp_9080[0] + tmp_9082[0], tmp_9080[1] + tmp_9082[1], tmp_9080[2] + tmp_9082[2]];
    signal tmp_9084[3] <== CMul()(challengeQ, tmp_9083);
    signal tmp_9085[3] <== [evals[87][0] - publics[380], evals[87][1], evals[87][2]];
    signal tmp_9086[3] <== CMul()(evals[1], tmp_9085);
    signal tmp_9087[3] <== [tmp_9084[0] + tmp_9086[0], tmp_9084[1] + tmp_9086[1], tmp_9084[2] + tmp_9086[2]];
    signal tmp_9088[3] <== CMul()(challengeQ, tmp_9087);
    signal tmp_9089[3] <== [evals[88][0] - publics[381], evals[88][1], evals[88][2]];
    signal tmp_9090[3] <== CMul()(evals[1], tmp_9089);
    signal tmp_9091[3] <== [tmp_9088[0] + tmp_9090[0], tmp_9088[1] + tmp_9090[1], tmp_9088[2] + tmp_9090[2]];
    signal tmp_9092[3] <== CMul()(challengeQ, tmp_9091);
    signal tmp_9093[3] <== [evals[89][0] - publics[382], evals[89][1], evals[89][2]];
    signal tmp_9094[3] <== CMul()(evals[1], tmp_9093);
    signal tmp_9095[3] <== [tmp_9092[0] + tmp_9094[0], tmp_9092[1] + tmp_9094[1], tmp_9092[2] + tmp_9094[2]];
    signal tmp_9096[3] <== CMul()(challengeQ, tmp_9095);
    signal tmp_9097[3] <== [evals[90][0] - publics[383], evals[90][1], evals[90][2]];
    signal tmp_9098[3] <== CMul()(evals[1], tmp_9097);
    signal tmp_9099[3] <== [tmp_9096[0] + tmp_9098[0], tmp_9096[1] + tmp_9098[1], tmp_9096[2] + tmp_9098[2]];
    signal tmp_9100[3] <== CMul()(challengeQ, tmp_9099);
    signal tmp_9101[3] <== [evals[67][0] - publics[384], evals[67][1], evals[67][2]];
    signal tmp_9102[3] <== CMul()(evals[0], tmp_9101);
    signal tmp_9103[3] <== [tmp_9100[0] + tmp_9102[0], tmp_9100[1] + tmp_9102[1], tmp_9100[2] + tmp_9102[2]];
    signal tmp_9104[3] <== CMul()(challengeQ, tmp_9103);
    signal tmp_9105[3] <== [evals[68][0] - publics[385], evals[68][1], evals[68][2]];
    signal tmp_9106[3] <== CMul()(evals[0], tmp_9105);
    signal tmp_9107[3] <== [tmp_9104[0] + tmp_9106[0], tmp_9104[1] + tmp_9106[1], tmp_9104[2] + tmp_9106[2]];
    signal tmp_9108[3] <== CMul()(challengeQ, tmp_9107);
    signal tmp_9109[3] <== [evals[69][0] - publics[386], evals[69][1], evals[69][2]];
    signal tmp_9110[3] <== CMul()(evals[0], tmp_9109);
    signal tmp_9111[3] <== [tmp_9108[0] + tmp_9110[0], tmp_9108[1] + tmp_9110[1], tmp_9108[2] + tmp_9110[2]];
    signal tmp_9112[3] <== CMul()(challengeQ, tmp_9111);
    signal tmp_9113[3] <== [evals[70][0] - publics[387], evals[70][1], evals[70][2]];
    signal tmp_9114[3] <== CMul()(evals[0], tmp_9113);
    signal tmp_9115[3] <== [tmp_9112[0] + tmp_9114[0], tmp_9112[1] + tmp_9114[1], tmp_9112[2] + tmp_9114[2]];
    signal tmp_9116[3] <== CMul()(challengeQ, tmp_9115);
    signal tmp_9117[3] <== [evals[71][0] - publics[388], evals[71][1], evals[71][2]];
    signal tmp_9118[3] <== CMul()(evals[0], tmp_9117);
    signal tmp_9119[3] <== [tmp_9116[0] + tmp_9118[0], tmp_9116[1] + tmp_9118[1], tmp_9116[2] + tmp_9118[2]];
    signal tmp_9120[3] <== CMul()(challengeQ, tmp_9119);
    signal tmp_9121[3] <== [evals[72][0] - publics[389], evals[72][1], evals[72][2]];
    signal tmp_9122[3] <== CMul()(evals[0], tmp_9121);
    signal tmp_9123[3] <== [tmp_9120[0] + tmp_9122[0], tmp_9120[1] + tmp_9122[1], tmp_9120[2] + tmp_9122[2]];
    signal tmp_9124[3] <== CMul()(challengeQ, tmp_9123);
    signal tmp_9125[3] <== [evals[73][0] - publics[390], evals[73][1], evals[73][2]];
    signal tmp_9126[3] <== CMul()(evals[0], tmp_9125);
    signal tmp_9127[3] <== [tmp_9124[0] + tmp_9126[0], tmp_9124[1] + tmp_9126[1], tmp_9124[2] + tmp_9126[2]];
    signal tmp_9128[3] <== CMul()(challengeQ, tmp_9127);
    signal tmp_9129[3] <== [evals[74][0] - publics[391], evals[74][1], evals[74][2]];
    signal tmp_9130[3] <== CMul()(evals[0], tmp_9129);
    signal tmp_9131[3] <== [tmp_9128[0] + tmp_9130[0], tmp_9128[1] + tmp_9130[1], tmp_9128[2] + tmp_9130[2]];
    signal tmp_9132[3] <== CMul()(challengeQ, tmp_9131);
    signal tmp_9133[3] <== [evals[75][0] - publics[392], evals[75][1], evals[75][2]];
    signal tmp_9134[3] <== CMul()(evals[0], tmp_9133);
    signal tmp_9135[3] <== [tmp_9132[0] + tmp_9134[0], tmp_9132[1] + tmp_9134[1], tmp_9132[2] + tmp_9134[2]];
    signal tmp_9136[3] <== CMul()(challengeQ, tmp_9135);
    signal tmp_9137[3] <== [evals[76][0] - publics[393], evals[76][1], evals[76][2]];
    signal tmp_9138[3] <== CMul()(evals[0], tmp_9137);
    signal tmp_9139[3] <== [tmp_9136[0] + tmp_9138[0], tmp_9136[1] + tmp_9138[1], tmp_9136[2] + tmp_9138[2]];
    signal tmp_9140[3] <== CMul()(challengeQ, tmp_9139);
    signal tmp_9141[3] <== [evals[77][0] - publics[394], evals[77][1], evals[77][2]];
    signal tmp_9142[3] <== CMul()(evals[0], tmp_9141);
    signal tmp_9143[3] <== [tmp_9140[0] + tmp_9142[0], tmp_9140[1] + tmp_9142[1], tmp_9140[2] + tmp_9142[2]];
    signal tmp_9144[3] <== CMul()(challengeQ, tmp_9143);
    signal tmp_9145[3] <== [evals[78][0] - publics[395], evals[78][1], evals[78][2]];
    signal tmp_9146[3] <== CMul()(evals[0], tmp_9145);
    signal tmp_9147[3] <== [tmp_9144[0] + tmp_9146[0], tmp_9144[1] + tmp_9146[1], tmp_9144[2] + tmp_9146[2]];
    signal tmp_9148[3] <== CMul()(challengeQ, tmp_9147);
    signal tmp_9149[3] <== [evals[79][0] - publics[396], evals[79][1], evals[79][2]];
    signal tmp_9150[3] <== CMul()(evals[0], tmp_9149);
    signal tmp_9151[3] <== [tmp_9148[0] + tmp_9150[0], tmp_9148[1] + tmp_9150[1], tmp_9148[2] + tmp_9150[2]];
    signal tmp_9152[3] <== CMul()(challengeQ, tmp_9151);
    signal tmp_9153[3] <== [evals[80][0] - publics[397], evals[80][1], evals[80][2]];
    signal tmp_9154[3] <== CMul()(evals[0], tmp_9153);
    signal tmp_9155[3] <== [tmp_9152[0] + tmp_9154[0], tmp_9152[1] + tmp_9154[1], tmp_9152[2] + tmp_9154[2]];
    signal tmp_9156[3] <== CMul()(challengeQ, tmp_9155);
    signal tmp_9157[3] <== [evals[81][0] - publics[398], evals[81][1], evals[81][2]];
    signal tmp_9158[3] <== CMul()(evals[0], tmp_9157);
    signal tmp_9159[3] <== [tmp_9156[0] + tmp_9158[0], tmp_9156[1] + tmp_9158[1], tmp_9156[2] + tmp_9158[2]];
    signal tmp_9160[3] <== CMul()(challengeQ, tmp_9159);
    tmp_9161 <== [evals[17][0] + evals[18][0], evals[17][1] + evals[18][1], evals[17][2] + evals[18][2]];
    signal tmp_9162[3] <== [evals[63][0] + tmp_9161[0], evals[63][1] + tmp_9161[1], evals[63][2] + tmp_9161[2]];
    tmp_9163 <== [evals[14][0] + evals[15][0], evals[14][1] + evals[15][1], evals[14][2] + evals[15][2]];
    signal tmp_9164[3] <== [tmp_9162[0] + tmp_9163[0], tmp_9162[1] + tmp_9163[1], tmp_9162[2] + tmp_9163[2]];
    signal tmp_9165[3] <== [tmp_9164[0] + evals[126][0], tmp_9164[1] + evals[126][1], tmp_9164[2] + evals[126][2]];
    signal tmp_9166[3] <== CMul()(evals[67], evals[68]);
    signal tmp_9167[3] <== CMul()(evals[45], tmp_9166);
    signal tmp_9168[3] <== CMul()(evals[46], evals[67]);
    signal tmp_9169[3] <== [tmp_9167[0] + tmp_9168[0], tmp_9167[1] + tmp_9168[1], tmp_9167[2] + tmp_9168[2]];
    signal tmp_9170[3] <== CMul()(evals[47], evals[68]);
    signal tmp_9171[3] <== [tmp_9169[0] + tmp_9170[0], tmp_9169[1] + tmp_9170[1], tmp_9169[2] + tmp_9170[2]];
    signal tmp_9172[3] <== CMul()(evals[48], evals[69]);
    signal tmp_9173[3] <== [tmp_9171[0] + tmp_9172[0], tmp_9171[1] + tmp_9172[1], tmp_9171[2] + tmp_9172[2]];
    signal tmp_9174[3] <== [tmp_9173[0] + evals[49][0], tmp_9173[1] + evals[49][1], tmp_9173[2] + evals[49][2]];
    signal tmp_9175[3] <== CMul()(tmp_9165, tmp_9174);
    signal tmp_9176[3] <== [tmp_9160[0] + tmp_9175[0], tmp_9160[1] + tmp_9175[1], tmp_9160[2] + tmp_9175[2]];
    signal tmp_9177[3] <== CMul()(challengeQ, tmp_9176);
    signal tmp_9178[3] <== CMul()(evals[70], evals[71]);
    signal tmp_9179[3] <== CMul()(evals[45], tmp_9178);
    tmp_9180 <== CMul()(evals[46], evals[70]);
    signal tmp_9181[3] <== [tmp_9179[0] + tmp_9180[0], tmp_9179[1] + tmp_9180[1], tmp_9179[2] + tmp_9180[2]];
    signal tmp_9182[3] <== CMul()(evals[47], evals[71]);
    signal tmp_9183[3] <== [tmp_9181[0] + tmp_9182[0], tmp_9181[1] + tmp_9182[1], tmp_9181[2] + tmp_9182[2]];
    signal tmp_9184[3] <== CMul()(evals[48], evals[72]);
    signal tmp_9185[3] <== [tmp_9183[0] + tmp_9184[0], tmp_9183[1] + tmp_9184[1], tmp_9183[2] + tmp_9184[2]];
    signal tmp_9186[3] <== [tmp_9185[0] + evals[49][0], tmp_9185[1] + evals[49][1], tmp_9185[2] + evals[49][2]];
    signal tmp_9187[3] <== CMul()(tmp_9165, tmp_9186);
    signal tmp_9188[3] <== [tmp_9177[0] + tmp_9187[0], tmp_9177[1] + tmp_9187[1], tmp_9177[2] + tmp_9187[2]];
    signal tmp_9189[3] <== CMul()(challengeQ, tmp_9188);
    signal tmp_9190[3] <== CMul()(evals[73], evals[74]);
    signal tmp_9191[3] <== CMul()(evals[50], tmp_9190);
    tmp_9192 <== CMul()(evals[51], evals[73]);
    signal tmp_9193[3] <== [tmp_9191[0] + tmp_9192[0], tmp_9191[1] + tmp_9192[1], tmp_9191[2] + tmp_9192[2]];
    signal tmp_9194[3] <== CMul()(evals[52], evals[74]);
    signal tmp_9195[3] <== [tmp_9193[0] + tmp_9194[0], tmp_9193[1] + tmp_9194[1], tmp_9193[2] + tmp_9194[2]];
    signal tmp_9196[3] <== CMul()(evals[53], evals[75]);
    signal tmp_9197[3] <== [tmp_9195[0] + tmp_9196[0], tmp_9195[1] + tmp_9196[1], tmp_9195[2] + tmp_9196[2]];
    signal tmp_9198[3] <== [tmp_9197[0] + evals[54][0], tmp_9197[1] + evals[54][1], tmp_9197[2] + evals[54][2]];
    signal tmp_9199[3] <== CMul()(tmp_9165, tmp_9198);
    signal tmp_9200[3] <== [tmp_9189[0] + tmp_9199[0], tmp_9189[1] + tmp_9199[1], tmp_9189[2] + tmp_9199[2]];
    signal tmp_9201[3] <== CMul()(challengeQ, tmp_9200);
    signal tmp_9203[3] <== [tmp_9162[0] + evals[126][0], tmp_9162[1] + evals[126][1], tmp_9162[2] + evals[126][2]];
    signal tmp_9204[3] <== CMul()(evals[76], evals[77]);
    signal tmp_9205[3] <== CMul()(evals[50], tmp_9204);
    signal tmp_9206[3] <== CMul()(evals[51], evals[76]);
    signal tmp_9207[3] <== [tmp_9205[0] + tmp_9206[0], tmp_9205[1] + tmp_9206[1], tmp_9205[2] + tmp_9206[2]];
    signal tmp_9208[3] <== CMul()(evals[52], evals[77]);
    signal tmp_9209[3] <== [tmp_9207[0] + tmp_9208[0], tmp_9207[1] + tmp_9208[1], tmp_9207[2] + tmp_9208[2]];
    tmp_9210 <== CMul()(evals[53], evals[78]);
    signal tmp_9211[3] <== [tmp_9209[0] + tmp_9210[0], tmp_9209[1] + tmp_9210[1], tmp_9209[2] + tmp_9210[2]];
    signal tmp_9212[3] <== [tmp_9211[0] + evals[54][0], tmp_9211[1] + evals[54][1], tmp_9211[2] + evals[54][2]];
    signal tmp_9213[3] <== CMul()(tmp_9203, tmp_9212);
    signal tmp_9214[3] <== [tmp_9201[0] + tmp_9213[0], tmp_9201[1] + tmp_9213[1], tmp_9201[2] + tmp_9213[2]];
    signal tmp_9215[3] <== CMul()(challengeQ, tmp_9214);
    signal tmp_9216[3] <== CMul()(evals[79], evals[80]);
    signal tmp_9217[3] <== CMul()(evals[50], tmp_9216);
    signal tmp_9218[3] <== CMul()(evals[51], evals[79]);
    signal tmp_9219[3] <== [tmp_9217[0] + tmp_9218[0], tmp_9217[1] + tmp_9218[1], tmp_9217[2] + tmp_9218[2]];
    signal tmp_9220[3] <== CMul()(evals[52], evals[80]);
    signal tmp_9221[3] <== [tmp_9219[0] + tmp_9220[0], tmp_9219[1] + tmp_9220[1], tmp_9219[2] + tmp_9220[2]];
    signal tmp_9222[3] <== CMul()(evals[53], evals[81]);
    signal tmp_9223[3] <== [tmp_9221[0] + tmp_9222[0], tmp_9221[1] + tmp_9222[1], tmp_9221[2] + tmp_9222[2]];
    signal tmp_9224[3] <== [tmp_9223[0] + evals[54][0], tmp_9223[1] + evals[54][1], tmp_9223[2] + evals[54][2]];
    signal tmp_9225[3] <== CMul()(tmp_9203, tmp_9224);
    signal tmp_9226[3] <== [tmp_9215[0] + tmp_9225[0], tmp_9215[1] + tmp_9225[1], tmp_9215[2] + tmp_9225[2]];
    signal tmp_9227[3] <== CMul()(challengeQ, tmp_9226);
    signal tmp_9228[3] <== CMul()(evals[82], evals[83]);
    signal tmp_9229[3] <== CMul()(evals[50], tmp_9228);
    signal tmp_9230[3] <== CMul()(evals[51], evals[82]);
    signal tmp_9231[3] <== [tmp_9229[0] + tmp_9230[0], tmp_9229[1] + tmp_9230[1], tmp_9229[2] + tmp_9230[2]];
    signal tmp_9232[3] <== CMul()(evals[52], evals[83]);
    signal tmp_9233[3] <== [tmp_9231[0] + tmp_9232[0], tmp_9231[1] + tmp_9232[1], tmp_9231[2] + tmp_9232[2]];
    signal tmp_9234[3] <== CMul()(evals[53], evals[84]);
    signal tmp_9235[3] <== [tmp_9233[0] + tmp_9234[0], tmp_9233[1] + tmp_9234[1], tmp_9233[2] + tmp_9234[2]];
    signal tmp_9236[3] <== [tmp_9235[0] + evals[54][0], tmp_9235[1] + evals[54][1], tmp_9235[2] + evals[54][2]];
    signal tmp_9237[3] <== CMul()(evals[63], tmp_9236);
    signal tmp_9238[3] <== [tmp_9227[0] + tmp_9237[0], tmp_9227[1] + tmp_9237[1], tmp_9227[2] + tmp_9237[2]];
    signal tmp_9239[3] <== CMul()(challengeQ, tmp_9238);
    signal tmp_9240[3] <== [evals[63][0] + evals[58][0], evals[63][1] + evals[58][1], evals[63][2] + evals[58][2]];
    signal tmp_9241[3] <== CMul()(evals[85], evals[86]);
    signal tmp_9242[3] <== CMul()(evals[50], tmp_9241);
    signal tmp_9243[3] <== CMul()(evals[51], evals[85]);
    signal tmp_9244[3] <== [tmp_9242[0] + tmp_9243[0], tmp_9242[1] + tmp_9243[1], tmp_9242[2] + tmp_9243[2]];
    signal tmp_9245[3] <== CMul()(evals[52], evals[86]);
    signal tmp_9246[3] <== [tmp_9244[0] + tmp_9245[0], tmp_9244[1] + tmp_9245[1], tmp_9244[2] + tmp_9245[2]];
    signal tmp_9247[3] <== CMul()(evals[53], evals[87]);
    signal tmp_9248[3] <== [tmp_9246[0] + tmp_9247[0], tmp_9246[1] + tmp_9247[1], tmp_9246[2] + tmp_9247[2]];
    signal tmp_9249[3] <== [tmp_9248[0] + evals[54][0], tmp_9248[1] + evals[54][1], tmp_9248[2] + evals[54][2]];
    signal tmp_9250[3] <== CMul()(tmp_9240, tmp_9249);
    signal tmp_9251[3] <== [tmp_9239[0] + tmp_9250[0], tmp_9239[1] + tmp_9250[1], tmp_9239[2] + tmp_9250[2]];
    signal tmp_9252[3] <== CMul()(challengeQ, tmp_9251);
    signal tmp_9254[3] <== [tmp_9240[0] + evals[59][0], tmp_9240[1] + evals[59][1], tmp_9240[2] + evals[59][2]];
    signal tmp_9255[3] <== CMul()(evals[88], evals[89]);
    signal tmp_9256[3] <== CMul()(evals[50], tmp_9255);
    signal tmp_9257[3] <== CMul()(evals[51], evals[88]);
    signal tmp_9258[3] <== [tmp_9256[0] + tmp_9257[0], tmp_9256[1] + tmp_9257[1], tmp_9256[2] + tmp_9257[2]];
    signal tmp_9259[3] <== CMul()(evals[52], evals[89]);
    signal tmp_9260[3] <== [tmp_9258[0] + tmp_9259[0], tmp_9258[1] + tmp_9259[1], tmp_9258[2] + tmp_9259[2]];
    signal tmp_9261[3] <== CMul()(evals[53], evals[90]);
    signal tmp_9262[3] <== [tmp_9260[0] + tmp_9261[0], tmp_9260[1] + tmp_9261[1], tmp_9260[2] + tmp_9261[2]];
    signal tmp_9263[3] <== [tmp_9262[0] + evals[54][0], tmp_9262[1] + evals[54][1], tmp_9262[2] + evals[54][2]];
    signal tmp_9264[3] <== CMul()(tmp_9254, tmp_9263);
    signal tmp_9265[3] <== [tmp_9252[0] + tmp_9264[0], tmp_9252[1] + tmp_9264[1], tmp_9252[2] + tmp_9264[2]];
    signal tmp_9266[3] <== CMul()(challengeQ, tmp_9265);
    signal tmp_9267[3] <== CMul()(evals[55], evals[134]);
    signal tmp_9268[3] <== [evals[134][0] - 1, evals[134][1], evals[134][2]];
    signal tmp_9269[3] <== CMul()(tmp_9267, tmp_9268);
    signal tmp_9270[3] <== [tmp_9266[0] + tmp_9269[0], tmp_9266[1] + tmp_9269[1], tmp_9266[2] + tmp_9269[2]];
    signal tmp_9271[3] <== CMul()(challengeQ, tmp_9270);
    signal tmp_9272[3] <== CMul()(evals[55], evals[151]);
    signal tmp_9273[3] <== [evals[151][0] - 1, evals[151][1], evals[151][2]];
    signal tmp_9274[3] <== CMul()(tmp_9272, tmp_9273);
    signal tmp_9275[3] <== [tmp_9271[0] + tmp_9274[0], tmp_9271[1] + tmp_9274[1], tmp_9271[2] + tmp_9274[2]];
    signal tmp_9276[3] <== CMul()(challengeQ, tmp_9275);
    signal tmp_9277[3] <== [1 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_9278[3] <== [1 - evals[151][0], -evals[151][1], -evals[151][2]];
    signal tmp_9279[3] <== CMul()(tmp_9277, tmp_9278);
    signal tmp_9280[3] <== CMul()(tmp_9279, evals[67]);
    signal tmp_9282[3] <== CMul()(evals[134], tmp_9278);
    signal tmp_9284[3] <== CMul()(tmp_9277, evals[151]);
    signal tmp_9285[3] <== [tmp_9282[0] + tmp_9284[0], tmp_9282[1] + tmp_9284[1], tmp_9282[2] + tmp_9284[2]];
    signal tmp_9286[3] <== CMul()(evals[134], evals[151]);
    signal tmp_9287[3] <== [tmp_9285[0] + tmp_9286[0], tmp_9285[1] + tmp_9286[1], tmp_9285[2] + tmp_9286[2]];
    signal tmp_9288[3] <== CMul()(tmp_9287, evals[71]);
    signal tmp_9289[3] <== [tmp_9280[0] + tmp_9288[0], tmp_9280[1] + tmp_9288[1], tmp_9280[2] + tmp_9288[2]];
    signal tmp_9290[3] <== [evals[83][0] - tmp_9289[0], evals[83][1] - tmp_9289[1], evals[83][2] - tmp_9289[2]];
    signal tmp_9291[3] <== CMul()(evals[55], tmp_9290);
    signal tmp_9292[3] <== [tmp_9276[0] + tmp_9291[0], tmp_9276[1] + tmp_9291[1], tmp_9276[2] + tmp_9291[2]];
    signal tmp_9293[3] <== CMul()(challengeQ, tmp_9292);
    signal tmp_9294[3] <== [evals[83][0] - evals[67][0], evals[83][1] - evals[67][1], evals[83][2] - evals[67][2]];
    signal tmp_9295[3] <== CMul()(evals[56], tmp_9294);
    signal tmp_9296[3] <== [tmp_9293[0] + tmp_9295[0], tmp_9293[1] + tmp_9295[1], tmp_9293[2] + tmp_9295[2]];
    signal tmp_9297[3] <== CMul()(challengeQ, tmp_9296);
    signal tmp_9298[3] <== CMul()(tmp_9279, evals[68]);
    signal tmp_9301[3] <== CMul()(tmp_9287, evals[72]);
    signal tmp_9302[3] <== [tmp_9298[0] + tmp_9301[0], tmp_9298[1] + tmp_9301[1], tmp_9298[2] + tmp_9301[2]];
    signal tmp_9303[3] <== [evals[84][0] - tmp_9302[0], evals[84][1] - tmp_9302[1], evals[84][2] - tmp_9302[2]];
    signal tmp_9304[3] <== CMul()(evals[55], tmp_9303);
    signal tmp_9305[3] <== [tmp_9297[0] + tmp_9304[0], tmp_9297[1] + tmp_9304[1], tmp_9297[2] + tmp_9304[2]];
    signal tmp_9306[3] <== CMul()(challengeQ, tmp_9305);
    signal tmp_9307[3] <== [evals[84][0] - evals[68][0], evals[84][1] - evals[68][1], evals[84][2] - evals[68][2]];
    signal tmp_9308[3] <== CMul()(evals[56], tmp_9307);
    signal tmp_9309[3] <== [tmp_9306[0] + tmp_9308[0], tmp_9306[1] + tmp_9308[1], tmp_9306[2] + tmp_9308[2]];
    signal tmp_9310[3] <== CMul()(challengeQ, tmp_9309);
    signal tmp_9311[3] <== CMul()(tmp_9279, evals[69]);
    signal tmp_9314[3] <== CMul()(tmp_9287, evals[73]);
    signal tmp_9315[3] <== [tmp_9311[0] + tmp_9314[0], tmp_9311[1] + tmp_9314[1], tmp_9311[2] + tmp_9314[2]];
    signal tmp_9316[3] <== [evals[85][0] - tmp_9315[0], evals[85][1] - tmp_9315[1], evals[85][2] - tmp_9315[2]];
    signal tmp_9317[3] <== CMul()(evals[55], tmp_9316);
    signal tmp_9318[3] <== [tmp_9310[0] + tmp_9317[0], tmp_9310[1] + tmp_9317[1], tmp_9310[2] + tmp_9317[2]];
    signal tmp_9319[3] <== CMul()(challengeQ, tmp_9318);
    signal tmp_9320[3] <== [evals[85][0] - evals[69][0], evals[85][1] - evals[69][1], evals[85][2] - evals[69][2]];
    signal tmp_9321[3] <== CMul()(evals[56], tmp_9320);
    signal tmp_9322[3] <== [tmp_9319[0] + tmp_9321[0], tmp_9319[1] + tmp_9321[1], tmp_9319[2] + tmp_9321[2]];
    signal tmp_9323[3] <== CMul()(challengeQ, tmp_9322);
    signal tmp_9324[3] <== CMul()(tmp_9279, evals[70]);
    signal tmp_9327[3] <== CMul()(tmp_9287, evals[74]);
    signal tmp_9328[3] <== [tmp_9324[0] + tmp_9327[0], tmp_9324[1] + tmp_9327[1], tmp_9324[2] + tmp_9327[2]];
    signal tmp_9329[3] <== [evals[86][0] - tmp_9328[0], evals[86][1] - tmp_9328[1], evals[86][2] - tmp_9328[2]];
    signal tmp_9330[3] <== CMul()(evals[55], tmp_9329);
    signal tmp_9331[3] <== [tmp_9323[0] + tmp_9330[0], tmp_9323[1] + tmp_9330[1], tmp_9323[2] + tmp_9330[2]];
    signal tmp_9332[3] <== CMul()(challengeQ, tmp_9331);
    signal tmp_9333[3] <== [evals[86][0] - evals[70][0], evals[86][1] - evals[70][1], evals[86][2] - evals[70][2]];
    signal tmp_9334[3] <== CMul()(evals[56], tmp_9333);
    signal tmp_9335[3] <== [tmp_9332[0] + tmp_9334[0], tmp_9332[1] + tmp_9334[1], tmp_9332[2] + tmp_9334[2]];
    signal tmp_9336[3] <== CMul()(challengeQ, tmp_9335);
    signal tmp_9337[3] <== CMul()(tmp_9279, evals[71]);
    signal tmp_9338[3] <== CMul()(tmp_9282, evals[67]);
    signal tmp_9339[3] <== [tmp_9337[0] + tmp_9338[0], tmp_9337[1] + tmp_9338[1], tmp_9337[2] + tmp_9338[2]];
    signal tmp_9340[3] <== [tmp_9284[0] + tmp_9286[0], tmp_9284[1] + tmp_9286[1], tmp_9284[2] + tmp_9286[2]];
    signal tmp_9341[3] <== CMul()(tmp_9340, evals[75]);
    signal tmp_9342[3] <== [tmp_9339[0] + tmp_9341[0], tmp_9339[1] + tmp_9341[1], tmp_9339[2] + tmp_9341[2]];
    signal tmp_9343[3] <== [evals[87][0] - tmp_9342[0], evals[87][1] - tmp_9342[1], evals[87][2] - tmp_9342[2]];
    signal tmp_9344[3] <== CMul()(evals[55], tmp_9343);
    signal tmp_9345[3] <== [tmp_9336[0] + tmp_9344[0], tmp_9336[1] + tmp_9344[1], tmp_9336[2] + tmp_9344[2]];
    signal tmp_9346[3] <== CMul()(challengeQ, tmp_9345);
    signal tmp_9347[3] <== [evals[87][0] - evals[71][0], evals[87][1] - evals[71][1], evals[87][2] - evals[71][2]];
    signal tmp_9348[3] <== CMul()(evals[56], tmp_9347);
    signal tmp_9349[3] <== [tmp_9346[0] + tmp_9348[0], tmp_9346[1] + tmp_9348[1], tmp_9346[2] + tmp_9348[2]];
    signal tmp_9350[3] <== CMul()(challengeQ, tmp_9349);
    signal tmp_9351[3] <== CMul()(tmp_9279, evals[72]);
    signal tmp_9352[3] <== CMul()(tmp_9282, evals[68]);
    signal tmp_9353[3] <== [tmp_9351[0] + tmp_9352[0], tmp_9351[1] + tmp_9352[1], tmp_9351[2] + tmp_9352[2]];
    signal tmp_9355[3] <== CMul()(tmp_9340, evals[76]);
    signal tmp_9356[3] <== [tmp_9353[0] + tmp_9355[0], tmp_9353[1] + tmp_9355[1], tmp_9353[2] + tmp_9355[2]];
    signal tmp_9357[3] <== [evals[88][0] - tmp_9356[0], evals[88][1] - tmp_9356[1], evals[88][2] - tmp_9356[2]];
    signal tmp_9358[3] <== CMul()(evals[55], tmp_9357);
    signal tmp_9359[3] <== [tmp_9350[0] + tmp_9358[0], tmp_9350[1] + tmp_9358[1], tmp_9350[2] + tmp_9358[2]];
    signal tmp_9360[3] <== CMul()(challengeQ, tmp_9359);
    signal tmp_9361[3] <== [evals[88][0] - evals[72][0], evals[88][1] - evals[72][1], evals[88][2] - evals[72][2]];
    signal tmp_9362[3] <== CMul()(evals[56], tmp_9361);
    signal tmp_9363[3] <== [tmp_9360[0] + tmp_9362[0], tmp_9360[1] + tmp_9362[1], tmp_9360[2] + tmp_9362[2]];
    signal tmp_9364[3] <== CMul()(challengeQ, tmp_9363);
    signal tmp_9365[3] <== CMul()(tmp_9279, evals[73]);
    signal tmp_9366[3] <== CMul()(tmp_9282, evals[69]);
    signal tmp_9367[3] <== [tmp_9365[0] + tmp_9366[0], tmp_9365[1] + tmp_9366[1], tmp_9365[2] + tmp_9366[2]];
    signal tmp_9369[3] <== CMul()(tmp_9340, evals[77]);
    signal tmp_9370[3] <== [tmp_9367[0] + tmp_9369[0], tmp_9367[1] + tmp_9369[1], tmp_9367[2] + tmp_9369[2]];
    signal tmp_9371[3] <== [evals[89][0] - tmp_9370[0], evals[89][1] - tmp_9370[1], evals[89][2] - tmp_9370[2]];
    signal tmp_9372[3] <== CMul()(evals[55], tmp_9371);
    signal tmp_9373[3] <== [tmp_9364[0] + tmp_9372[0], tmp_9364[1] + tmp_9372[1], tmp_9364[2] + tmp_9372[2]];
    signal tmp_9374[3] <== CMul()(challengeQ, tmp_9373);
    signal tmp_9375[3] <== [evals[89][0] - evals[73][0], evals[89][1] - evals[73][1], evals[89][2] - evals[73][2]];
    signal tmp_9376[3] <== CMul()(evals[56], tmp_9375);
    signal tmp_9377[3] <== [tmp_9374[0] + tmp_9376[0], tmp_9374[1] + tmp_9376[1], tmp_9374[2] + tmp_9376[2]];
    signal tmp_9378[3] <== CMul()(challengeQ, tmp_9377);
    signal tmp_9379[3] <== CMul()(tmp_9279, evals[74]);
    signal tmp_9380[3] <== CMul()(tmp_9282, evals[70]);
    signal tmp_9381[3] <== [tmp_9379[0] + tmp_9380[0], tmp_9379[1] + tmp_9380[1], tmp_9379[2] + tmp_9380[2]];
    signal tmp_9383[3] <== CMul()(tmp_9340, evals[78]);
    signal tmp_9384[3] <== [tmp_9381[0] + tmp_9383[0], tmp_9381[1] + tmp_9383[1], tmp_9381[2] + tmp_9383[2]];
    signal tmp_9385[3] <== [evals[90][0] - tmp_9384[0], evals[90][1] - tmp_9384[1], evals[90][2] - tmp_9384[2]];
    signal tmp_9386[3] <== CMul()(evals[55], tmp_9385);
    signal tmp_9387[3] <== [tmp_9378[0] + tmp_9386[0], tmp_9378[1] + tmp_9386[1], tmp_9378[2] + tmp_9386[2]];
    signal tmp_9388[3] <== CMul()(challengeQ, tmp_9387);
    signal tmp_9389[3] <== [evals[90][0] - evals[74][0], evals[90][1] - evals[74][1], evals[90][2] - evals[74][2]];
    signal tmp_9390[3] <== CMul()(evals[56], tmp_9389);
    signal tmp_9391[3] <== [tmp_9388[0] + tmp_9390[0], tmp_9388[1] + tmp_9390[1], tmp_9388[2] + tmp_9390[2]];
    signal tmp_9392[3] <== CMul()(challengeQ, tmp_9391);
    signal tmp_9393[3] <== [tmp_9279[0] + tmp_9282[0], tmp_9279[1] + tmp_9282[1], tmp_9279[2] + tmp_9282[2]];
    signal tmp_9394[3] <== CMul()(tmp_9393, evals[75]);
    signal tmp_9395[3] <== CMul()(tmp_9284, evals[67]);
    signal tmp_9396[3] <== [tmp_9394[0] + tmp_9395[0], tmp_9394[1] + tmp_9395[1], tmp_9394[2] + tmp_9395[2]];
    signal tmp_9397[3] <== CMul()(tmp_9286, evals[79]);
    signal tmp_9398[3] <== [tmp_9396[0] + tmp_9397[0], tmp_9396[1] + tmp_9397[1], tmp_9396[2] + tmp_9397[2]];
    signal tmp_9399[3] <== [evals[91][0] - tmp_9398[0], evals[91][1] - tmp_9398[1], evals[91][2] - tmp_9398[2]];
    signal tmp_9400[3] <== CMul()(evals[55], tmp_9399);
    signal tmp_9401[3] <== [tmp_9392[0] + tmp_9400[0], tmp_9392[1] + tmp_9400[1], tmp_9392[2] + tmp_9400[2]];
    signal tmp_9402[3] <== CMul()(challengeQ, tmp_9401);
    signal tmp_9403[3] <== [evals[91][0] - evals[75][0], evals[91][1] - evals[75][1], evals[91][2] - evals[75][2]];
    signal tmp_9404[3] <== CMul()(evals[56], tmp_9403);
    signal tmp_9405[3] <== [tmp_9402[0] + tmp_9404[0], tmp_9402[1] + tmp_9404[1], tmp_9402[2] + tmp_9404[2]];
    signal tmp_9406[3] <== CMul()(challengeQ, tmp_9405);
    signal tmp_9408[3] <== CMul()(tmp_9393, evals[76]);
    signal tmp_9409[3] <== CMul()(tmp_9284, evals[68]);
    signal tmp_9410[3] <== [tmp_9408[0] + tmp_9409[0], tmp_9408[1] + tmp_9409[1], tmp_9408[2] + tmp_9409[2]];
    signal tmp_9411[3] <== CMul()(tmp_9286, evals[80]);
    signal tmp_9412[3] <== [tmp_9410[0] + tmp_9411[0], tmp_9410[1] + tmp_9411[1], tmp_9410[2] + tmp_9411[2]];
    signal tmp_9413[3] <== [evals[92][0] - tmp_9412[0], evals[92][1] - tmp_9412[1], evals[92][2] - tmp_9412[2]];
    signal tmp_9414[3] <== CMul()(evals[55], tmp_9413);
    signal tmp_9415[3] <== [tmp_9406[0] + tmp_9414[0], tmp_9406[1] + tmp_9414[1], tmp_9406[2] + tmp_9414[2]];
    signal tmp_9416[3] <== CMul()(challengeQ, tmp_9415);
    signal tmp_9417[3] <== [evals[92][0] - evals[76][0], evals[92][1] - evals[76][1], evals[92][2] - evals[76][2]];
    signal tmp_9418[3] <== CMul()(evals[56], tmp_9417);
    signal tmp_9419[3] <== [tmp_9416[0] + tmp_9418[0], tmp_9416[1] + tmp_9418[1], tmp_9416[2] + tmp_9418[2]];
    signal tmp_9420[3] <== CMul()(challengeQ, tmp_9419);
    signal tmp_9422[3] <== CMul()(tmp_9393, evals[77]);
    signal tmp_9423[3] <== CMul()(tmp_9284, evals[69]);
    signal tmp_9424[3] <== [tmp_9422[0] + tmp_9423[0], tmp_9422[1] + tmp_9423[1], tmp_9422[2] + tmp_9423[2]];
    signal tmp_9425[3] <== CMul()(tmp_9286, evals[81]);
    signal tmp_9426[3] <== [tmp_9424[0] + tmp_9425[0], tmp_9424[1] + tmp_9425[1], tmp_9424[2] + tmp_9425[2]];
    signal tmp_9427[3] <== [evals[93][0] - tmp_9426[0], evals[93][1] - tmp_9426[1], evals[93][2] - tmp_9426[2]];
    signal tmp_9428[3] <== CMul()(evals[55], tmp_9427);
    signal tmp_9429[3] <== [tmp_9420[0] + tmp_9428[0], tmp_9420[1] + tmp_9428[1], tmp_9420[2] + tmp_9428[2]];
    signal tmp_9430[3] <== CMul()(challengeQ, tmp_9429);
    signal tmp_9431[3] <== [evals[93][0] - evals[77][0], evals[93][1] - evals[77][1], evals[93][2] - evals[77][2]];
    signal tmp_9432[3] <== CMul()(evals[56], tmp_9431);
    signal tmp_9433[3] <== [tmp_9430[0] + tmp_9432[0], tmp_9430[1] + tmp_9432[1], tmp_9430[2] + tmp_9432[2]];
    signal tmp_9434[3] <== CMul()(challengeQ, tmp_9433);
    signal tmp_9436[3] <== CMul()(tmp_9393, evals[78]);
    signal tmp_9437[3] <== CMul()(tmp_9284, evals[70]);
    signal tmp_9438[3] <== [tmp_9436[0] + tmp_9437[0], tmp_9436[1] + tmp_9437[1], tmp_9436[2] + tmp_9437[2]];
    signal tmp_9439[3] <== CMul()(tmp_9286, evals[82]);
    signal tmp_9440[3] <== [tmp_9438[0] + tmp_9439[0], tmp_9438[1] + tmp_9439[1], tmp_9438[2] + tmp_9439[2]];
    signal tmp_9441[3] <== [evals[94][0] - tmp_9440[0], evals[94][1] - tmp_9440[1], evals[94][2] - tmp_9440[2]];
    signal tmp_9442[3] <== CMul()(evals[55], tmp_9441);
    signal tmp_9443[3] <== [tmp_9434[0] + tmp_9442[0], tmp_9434[1] + tmp_9442[1], tmp_9434[2] + tmp_9442[2]];
    signal tmp_9444[3] <== CMul()(challengeQ, tmp_9443);
    signal tmp_9445[3] <== [evals[94][0] - evals[78][0], evals[94][1] - evals[78][1], evals[94][2] - evals[78][2]];
    signal tmp_9446[3] <== CMul()(evals[56], tmp_9445);
    signal tmp_9447[3] <== [tmp_9444[0] + tmp_9446[0], tmp_9444[1] + tmp_9446[1], tmp_9444[2] + tmp_9446[2]];
    signal tmp_9448[3] <== CMul()(challengeQ, tmp_9447);
    signal tmp_9450[3] <== [tmp_9393[0] + tmp_9284[0], tmp_9393[1] + tmp_9284[1], tmp_9393[2] + tmp_9284[2]];
    signal tmp_9451[3] <== CMul()(tmp_9450, evals[79]);
    signal tmp_9452[3] <== CMul()(tmp_9286, evals[67]);
    signal tmp_9453[3] <== [tmp_9451[0] + tmp_9452[0], tmp_9451[1] + tmp_9452[1], tmp_9451[2] + tmp_9452[2]];
    signal tmp_9454[3] <== [evals[95][0] - tmp_9453[0], evals[95][1] - tmp_9453[1], evals[95][2] - tmp_9453[2]];
    signal tmp_9455[3] <== CMul()(evals[55], tmp_9454);
    signal tmp_9456[3] <== [tmp_9448[0] + tmp_9455[0], tmp_9448[1] + tmp_9455[1], tmp_9448[2] + tmp_9455[2]];
    signal tmp_9457[3] <== CMul()(challengeQ, tmp_9456);
    signal tmp_9458[3] <== [evals[95][0] - evals[79][0], evals[95][1] - evals[79][1], evals[95][2] - evals[79][2]];
    signal tmp_9459[3] <== CMul()(evals[56], tmp_9458);
    signal tmp_9460[3] <== [tmp_9457[0] + tmp_9459[0], tmp_9457[1] + tmp_9459[1], tmp_9457[2] + tmp_9459[2]];
    signal tmp_9461[3] <== CMul()(challengeQ, tmp_9460);
    signal tmp_9464[3] <== CMul()(tmp_9450, evals[80]);
    signal tmp_9465[3] <== CMul()(tmp_9286, evals[68]);
    signal tmp_9466[3] <== [tmp_9464[0] + tmp_9465[0], tmp_9464[1] + tmp_9465[1], tmp_9464[2] + tmp_9465[2]];
    signal tmp_9467[3] <== [evals[96][0] - tmp_9466[0], evals[96][1] - tmp_9466[1], evals[96][2] - tmp_9466[2]];
    signal tmp_9468[3] <== CMul()(evals[55], tmp_9467);
    signal tmp_9469[3] <== [tmp_9461[0] + tmp_9468[0], tmp_9461[1] + tmp_9468[1], tmp_9461[2] + tmp_9468[2]];
    signal tmp_9470[3] <== CMul()(challengeQ, tmp_9469);
    signal tmp_9471[3] <== [evals[96][0] - evals[80][0], evals[96][1] - evals[80][1], evals[96][2] - evals[80][2]];
    signal tmp_9472[3] <== CMul()(evals[56], tmp_9471);
    signal tmp_9473[3] <== [tmp_9470[0] + tmp_9472[0], tmp_9470[1] + tmp_9472[1], tmp_9470[2] + tmp_9472[2]];
    signal tmp_9474[3] <== CMul()(challengeQ, tmp_9473);
    signal tmp_9477[3] <== CMul()(tmp_9450, evals[81]);
    signal tmp_9478[3] <== CMul()(tmp_9286, evals[69]);
    signal tmp_9479[3] <== [tmp_9477[0] + tmp_9478[0], tmp_9477[1] + tmp_9478[1], tmp_9477[2] + tmp_9478[2]];
    signal tmp_9480[3] <== [evals[97][0] - tmp_9479[0], evals[97][1] - tmp_9479[1], evals[97][2] - tmp_9479[2]];
    signal tmp_9481[3] <== CMul()(evals[55], tmp_9480);
    signal tmp_9482[3] <== [tmp_9474[0] + tmp_9481[0], tmp_9474[1] + tmp_9481[1], tmp_9474[2] + tmp_9481[2]];
    signal tmp_9483[3] <== CMul()(challengeQ, tmp_9482);
    signal tmp_9484[3] <== [evals[97][0] - evals[81][0], evals[97][1] - evals[81][1], evals[97][2] - evals[81][2]];
    signal tmp_9485[3] <== CMul()(evals[56], tmp_9484);
    signal tmp_9486[3] <== [tmp_9483[0] + tmp_9485[0], tmp_9483[1] + tmp_9485[1], tmp_9483[2] + tmp_9485[2]];
    signal tmp_9487[3] <== CMul()(challengeQ, tmp_9486);
    signal tmp_9490[3] <== CMul()(tmp_9450, evals[82]);
    signal tmp_9491[3] <== CMul()(tmp_9286, evals[70]);
    signal tmp_9492[3] <== [tmp_9490[0] + tmp_9491[0], tmp_9490[1] + tmp_9491[1], tmp_9490[2] + tmp_9491[2]];
    signal tmp_9493[3] <== [evals[98][0] - tmp_9492[0], evals[98][1] - tmp_9492[1], evals[98][2] - tmp_9492[2]];
    signal tmp_9494[3] <== CMul()(evals[55], tmp_9493);
    signal tmp_9495[3] <== [tmp_9487[0] + tmp_9494[0], tmp_9487[1] + tmp_9494[1], tmp_9487[2] + tmp_9494[2]];
    signal tmp_9496[3] <== CMul()(challengeQ, tmp_9495);
    signal tmp_9497[3] <== [evals[98][0] - evals[82][0], evals[98][1] - evals[82][1], evals[98][2] - evals[82][2]];
    signal tmp_9498[3] <== CMul()(evals[56], tmp_9497);
    signal tmp_9499[3] <== [tmp_9496[0] + tmp_9498[0], tmp_9496[1] + tmp_9498[1], tmp_9496[2] + tmp_9498[2]];
    tmp_9500 <== CMul()(challengeQ, tmp_9499);
    tmp_9501 <== [evals[55][0] + evals[56][0], evals[55][1] + evals[56][1], evals[55][2] + evals[56][2]];
    signal tmp_9502[3] <== [tmp_9501[0] + tmp_9161[0], tmp_9501[1] + tmp_9161[1], tmp_9501[2] + tmp_9161[2]];
    signal tmp_9503[3] <== [tmp_9502[0] + evals[126][0], tmp_9502[1] + evals[126][1], tmp_9502[2] + evals[126][2]];
    tmp_9504 <== [tmp_9503[0] + evals[57][0], tmp_9503[1] + evals[57][1], tmp_9503[2] + evals[57][2]];
    signal tmp_9505[3] <== [tmp_9501[0] * 1579613653969377219, tmp_9501[1] * 1579613653969377219, tmp_9501[2] * 1579613653969377219];
    signal tmp_9506[3] <== [evals[83][0] + tmp_9505[0], evals[83][1] + tmp_9505[1], evals[83][2] + tmp_9505[2]];
    signal tmp_9507[3] <== CMul()(tmp_9506, tmp_9506);
    signal tmp_9508[3] <== CMul()(tmp_9507, tmp_9507);
    signal tmp_9509[3] <== CMul()(tmp_9508, tmp_9507);
    signal tmp_9510[3] <== CMul()(tmp_9509, tmp_9506);
    signal tmp_9511[3] <== [tmp_9501[0] * 7845454723070343021, tmp_9501[1] * 7845454723070343021, tmp_9501[2] * 7845454723070343021];
    signal tmp_9512[3] <== [tmp_9161[0] * 14871271697671130741, tmp_9161[1] * 14871271697671130741, tmp_9161[2] * 14871271697671130741];
    signal tmp_9513[3] <== [tmp_9511[0] + tmp_9512[0], tmp_9511[1] + tmp_9512[1], tmp_9511[2] + tmp_9512[2]];
    signal tmp_9514[3] <== [evals[126][0] * 967731447871038855, evals[126][1] * 967731447871038855, evals[126][2] * 967731447871038855];
    signal tmp_9515[3] <== [tmp_9513[0] + tmp_9514[0], tmp_9513[1] + tmp_9514[1], tmp_9513[2] + tmp_9514[2]];
    signal tmp_9516[3] <== [evals[57][0] * 10880189488914803976, evals[57][1] * 10880189488914803976, evals[57][2] * 10880189488914803976];
    signal tmp_9517[3] <== [tmp_9515[0] + tmp_9516[0], tmp_9515[1] + tmp_9516[1], tmp_9515[2] + tmp_9516[2]];
    tmp_9518 <== [tmp_9510[0] + tmp_9517[0], tmp_9510[1] + tmp_9517[1], tmp_9510[2] + tmp_9517[2]];
    signal tmp_9519[3] <== [tmp_9501[0] * 15509510893087893340, tmp_9501[1] * 15509510893087893340, tmp_9501[2] * 15509510893087893340];
    signal tmp_9520[3] <== [evals[84][0] + tmp_9519[0], evals[84][1] + tmp_9519[1], evals[84][2] + tmp_9519[2]];
    signal tmp_9521[3] <== CMul()(tmp_9520, tmp_9520);
    signal tmp_9522[3] <== CMul()(tmp_9521, tmp_9521);
    signal tmp_9523[3] <== CMul()(tmp_9522, tmp_9521);
    signal tmp_9524[3] <== CMul()(tmp_9523, tmp_9520);
    signal tmp_9525[3] <== [tmp_9501[0] * 4429017498140175847, tmp_9501[1] * 4429017498140175847, tmp_9501[2] * 4429017498140175847];
    signal tmp_9526[3] <== [tmp_9161[0] * 7818640162547527607, tmp_9161[1] * 7818640162547527607, tmp_9161[2] * 7818640162547527607];
    signal tmp_9527[3] <== [tmp_9525[0] + tmp_9526[0], tmp_9525[1] + tmp_9526[1], tmp_9525[2] + tmp_9526[2]];
    signal tmp_9528[3] <== [evals[126][0] * 13497390221064521284, evals[126][1] * 13497390221064521284, evals[126][2] * 13497390221064521284];
    signal tmp_9529[3] <== [tmp_9527[0] + tmp_9528[0], tmp_9527[1] + tmp_9528[1], tmp_9527[2] + tmp_9528[2]];
    signal tmp_9530[3] <== [evals[57][0] * 10817563486901345506, evals[57][1] * 10817563486901345506, evals[57][2] * 10817563486901345506];
    signal tmp_9531[3] <== [tmp_9529[0] + tmp_9530[0], tmp_9529[1] + tmp_9530[1], tmp_9529[2] + tmp_9530[2]];
    tmp_9532 <== [tmp_9524[0] + tmp_9531[0], tmp_9524[1] + tmp_9531[1], tmp_9524[2] + tmp_9531[2]];
    signal tmp_9533[3] <== [tmp_9518[0] + tmp_9532[0], tmp_9518[1] + tmp_9532[1], tmp_9518[2] + tmp_9532[2]];
    signal tmp_9534[3] <== [tmp_9501[0] * 10090715174060125222, tmp_9501[1] * 10090715174060125222, tmp_9501[2] * 10090715174060125222];
    signal tmp_9535[3] <== [evals[85][0] + tmp_9534[0], evals[85][1] + tmp_9534[1], evals[85][2] + tmp_9534[2]];
    signal tmp_9536[3] <== CMul()(tmp_9535, tmp_9535);
    signal tmp_9537[3] <== CMul()(tmp_9536, tmp_9536);
    signal tmp_9538[3] <== CMul()(tmp_9537, tmp_9536);
    signal tmp_9539[3] <== CMul()(tmp_9538, tmp_9535);
    signal tmp_9540[3] <== [tmp_9501[0] * 2092211485884675791, tmp_9501[1] * 2092211485884675791, tmp_9501[2] * 2092211485884675791];
    signal tmp_9541[3] <== [tmp_9161[0] * 10207849655710394742, tmp_9161[1] * 10207849655710394742, tmp_9161[2] * 10207849655710394742];
    signal tmp_9542[3] <== [tmp_9540[0] + tmp_9541[0], tmp_9540[1] + tmp_9541[1], tmp_9540[2] + tmp_9541[2]];
    signal tmp_9543[3] <== [evals[126][0] * 7079992912276546290, evals[126][1] * 7079992912276546290, evals[126][2] * 7079992912276546290];
    signal tmp_9544[3] <== [tmp_9542[0] + tmp_9543[0], tmp_9542[1] + tmp_9543[1], tmp_9542[2] + tmp_9543[2]];
    signal tmp_9545[3] <== [evals[57][0] * 5595838894652252793, evals[57][1] * 5595838894652252793, evals[57][2] * 5595838894652252793];
    signal tmp_9546[3] <== [tmp_9544[0] + tmp_9545[0], tmp_9544[1] + tmp_9545[1], tmp_9544[2] + tmp_9545[2]];
    tmp_9547 <== [tmp_9539[0] + tmp_9546[0], tmp_9539[1] + tmp_9546[1], tmp_9539[2] + tmp_9546[2]];
    signal tmp_9548[3] <== [51 * tmp_9547[0], 51 * tmp_9547[1], 51 * tmp_9547[2]];
    signal tmp_9549[3] <== [tmp_9533[0] + tmp_9548[0], tmp_9533[1] + tmp_9548[1], tmp_9533[2] + tmp_9548[2]];
    signal tmp_9550[3] <== [tmp_9501[0] * 5625716564419252202, tmp_9501[1] * 5625716564419252202, tmp_9501[2] * 5625716564419252202];
    signal tmp_9551[3] <== [evals[86][0] + tmp_9550[0], evals[86][1] + tmp_9550[1], evals[86][2] + tmp_9550[2]];
    signal tmp_9552[3] <== CMul()(tmp_9551, tmp_9551);
    signal tmp_9553[3] <== CMul()(tmp_9552, tmp_9552);
    signal tmp_9554[3] <== CMul()(tmp_9553, tmp_9552);
    signal tmp_9555[3] <== CMul()(tmp_9554, tmp_9551);
    signal tmp_9556[3] <== [tmp_9501[0] * 3672938463800022916, tmp_9501[1] * 3672938463800022916, tmp_9501[2] * 3672938463800022916];
    signal tmp_9557[3] <== [tmp_9161[0] * 17997741013544777071, tmp_9161[1] * 17997741013544777071, tmp_9161[2] * 17997741013544777071];
    signal tmp_9558[3] <== [tmp_9556[0] + tmp_9557[0], tmp_9556[1] + tmp_9557[1], tmp_9556[2] + tmp_9557[2]];
    signal tmp_9559[3] <== [evals[126][0] * 4056766622830105470, evals[126][1] * 4056766622830105470, evals[126][2] * 4056766622830105470];
    signal tmp_9560[3] <== [tmp_9558[0] + tmp_9559[0], tmp_9558[1] + tmp_9559[1], tmp_9558[2] + tmp_9559[2]];
    signal tmp_9561[3] <== [evals[57][0] * 7339830491681595439, evals[57][1] * 7339830491681595439, evals[57][2] * 7339830491681595439];
    signal tmp_9562[3] <== [tmp_9560[0] + tmp_9561[0], tmp_9560[1] + tmp_9561[1], tmp_9560[2] + tmp_9561[2]];
    tmp_9563 <== [tmp_9555[0] + tmp_9562[0], tmp_9555[1] + tmp_9562[1], tmp_9555[2] + tmp_9562[2]];
    signal tmp_9564[3] <== [tmp_9549[0] + tmp_9563[0], tmp_9549[1] + tmp_9563[1], tmp_9549[2] + tmp_9563[2]];
    signal tmp_9565[3] <== [tmp_9501[0] * 3006005019077469174, tmp_9501[1] * 3006005019077469174, tmp_9501[2] * 3006005019077469174];
    signal tmp_9566[3] <== [evals[87][0] + tmp_9565[0], evals[87][1] + tmp_9565[1], evals[87][2] + tmp_9565[2]];
    signal tmp_9567[3] <== CMul()(tmp_9566, tmp_9566);
    signal tmp_9568[3] <== CMul()(tmp_9567, tmp_9567);
    signal tmp_9569[3] <== CMul()(tmp_9568, tmp_9567);
    signal tmp_9570[3] <== CMul()(tmp_9569, tmp_9566);
    signal tmp_9571[3] <== [tmp_9501[0] * 14908811154314031538, tmp_9501[1] * 14908811154314031538, tmp_9501[2] * 14908811154314031538];
    signal tmp_9572[3] <== [tmp_9161[0] * 16991056560447799587, tmp_9161[1] * 16991056560447799587, tmp_9161[2] * 16991056560447799587];
    signal tmp_9573[3] <== [tmp_9571[0] + tmp_9572[0], tmp_9571[1] + tmp_9572[1], tmp_9571[2] + tmp_9572[2]];
    signal tmp_9574[3] <== [evals[126][0] * 6323166457980539990, evals[126][1] * 6323166457980539990, evals[126][2] * 6323166457980539990];
    signal tmp_9575[3] <== [tmp_9573[0] + tmp_9574[0], tmp_9573[1] + tmp_9574[1], tmp_9573[2] + tmp_9574[2]];
    signal tmp_9576[3] <== [evals[57][0] * 13024184231170405476, evals[57][1] * 13024184231170405476, evals[57][2] * 13024184231170405476];
    signal tmp_9577[3] <== [tmp_9575[0] + tmp_9576[0], tmp_9575[1] + tmp_9576[1], tmp_9575[2] + tmp_9576[2]];
    tmp_9578 <== [tmp_9570[0] + tmp_9577[0], tmp_9570[1] + tmp_9577[1], tmp_9570[2] + tmp_9577[2]];
    signal tmp_9579[3] <== [11 * tmp_9578[0], 11 * tmp_9578[1], 11 * tmp_9578[2]];
    tmp_9580 <== [tmp_9564[0] + tmp_9579[0], tmp_9564[1] + tmp_9579[1], tmp_9564[2] + tmp_9579[2]];
    signal tmp_9581[3] <== [tmp_9501[0] * 18314693207014427912, tmp_9501[1] * 18314693207014427912, tmp_9501[2] * 18314693207014427912];
    signal tmp_9582[3] <== [evals[88][0] + tmp_9581[0], evals[88][1] + tmp_9581[1], evals[88][2] + tmp_9581[2]];
    signal tmp_9583[3] <== CMul()(tmp_9582, tmp_9582);
    signal tmp_9584[3] <== CMul()(tmp_9583, tmp_9583);
    signal tmp_9585[3] <== CMul()(tmp_9584, tmp_9583);
    tmp_9586 <== CMul()(tmp_9585, tmp_9582);
}

template VerifyEvaluationsChunks2_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_9161[3];
    signal input tmp_9500[3];
    signal input tmp_9501[3];
    signal input tmp_9504[3];
    signal input tmp_9518[3];
    signal input tmp_9532[3];
    signal input tmp_9547[3];
    signal input tmp_9563[3];
    signal input tmp_9578[3];
    signal input tmp_9580[3];
    signal input tmp_9586[3];

    signal output tmp_10511[3];
    signal output tmp_10513[3];
    signal output tmp_10519[3];
    signal output tmp_10525[3];
    signal output tmp_10529[3];
    signal output tmp_10535[3];
    signal output tmp_10540[3];
    signal output tmp_10546[3];
    signal output tmp_10552[3];
    signal output tmp_10558[3];
    signal output tmp_10563[3];
    signal output tmp_10569[3];
    signal output tmp_10575[3];
    signal output tmp_10581[3];
    signal output tmp_10587[3];
    signal output tmp_10593[3];
    signal output tmp_10594[3];
    signal output tmp_10599[3];
    signal output tmp_10605[3];
    signal output tmp_10610[3];
    signal output tmp_10616[3];
    signal output tmp_10622[3];
    signal output tmp_10628[3];
    signal output tmp_10634[3];
    signal output tmp_10640[3];
    signal output tmp_10646[3];
    signal output tmp_10652[3];
    signal output tmp_10658[3];
    signal output tmp_10664[3];
    signal output tmp_10670[3];
    signal output tmp_10676[3];
    signal output tmp_10682[3];
    signal output tmp_10688[3];
    signal output tmp_10694[3];
    signal output tmp_10700[3];
    signal output tmp_10702[3];
    signal output tmp_10731[3];
    signal output tmp_10745[3];
    signal output tmp_10787[3];
    signal output tmp_10801[3];
    signal output tmp_10843[3];
    signal output tmp_10857[3];
    signal output tmp_10899[3];
    signal output tmp_10913[3];
    signal output tmp_10926[3];
    signal output tmp_10971[3];
    signal output tmp_10977[3];
    signal output tmp_10981[3];
    signal output tmp_10988[3];
    signal output tmp_10990[3];
    signal tmp_9587[3] <== [tmp_9501[0] * 6573822840603111307, tmp_9501[1] * 6573822840603111307, tmp_9501[2] * 6573822840603111307];
    signal tmp_9588[3] <== [tmp_9161[0] * 348197253806644292, tmp_9161[1] * 348197253806644292, tmp_9161[2] * 348197253806644292];
    signal tmp_9589[3] <== [tmp_9587[0] + tmp_9588[0], tmp_9587[1] + tmp_9588[1], tmp_9587[2] + tmp_9588[2]];
    signal tmp_9590[3] <== [evals[126][0] * 13419047640011651400, evals[126][1] * 13419047640011651400, evals[126][2] * 13419047640011651400];
    signal tmp_9591[3] <== [tmp_9589[0] + tmp_9590[0], tmp_9589[1] + tmp_9590[1], tmp_9589[2] + tmp_9590[2]];
    signal tmp_9592[3] <== [evals[57][0] * 7496303001397288101, evals[57][1] * 7496303001397288101, evals[57][2] * 7496303001397288101];
    signal tmp_9593[3] <== [tmp_9591[0] + tmp_9592[0], tmp_9591[1] + tmp_9592[1], tmp_9591[2] + tmp_9592[2]];
    signal tmp_9594[3] <== [tmp_9586[0] + tmp_9593[0], tmp_9586[1] + tmp_9593[1], tmp_9586[2] + tmp_9593[2]];
    signal tmp_9595[3] <== [17 * tmp_9594[0], 17 * tmp_9594[1], 17 * tmp_9594[2]];
    signal tmp_9596[3] <== [tmp_9580[0] + tmp_9595[0], tmp_9580[1] + tmp_9595[1], tmp_9580[2] + tmp_9595[2]];
    signal tmp_9597[3] <== [tmp_9501[0] * 10170571510627764565, tmp_9501[1] * 10170571510627764565, tmp_9501[2] * 10170571510627764565];
    signal tmp_9598[3] <== [evals[89][0] + tmp_9597[0], evals[89][1] + tmp_9597[1], evals[89][2] + tmp_9597[2]];
    signal tmp_9599[3] <== CMul()(tmp_9598, tmp_9598);
    signal tmp_9600[3] <== CMul()(tmp_9599, tmp_9599);
    signal tmp_9601[3] <== CMul()(tmp_9600, tmp_9599);
    signal tmp_9602[3] <== CMul()(tmp_9601, tmp_9598);
    signal tmp_9603[3] <== [tmp_9501[0] * 10909691831624859236, tmp_9501[1] * 10909691831624859236, tmp_9501[2] * 10909691831624859236];
    signal tmp_9604[3] <== [tmp_9161[0] * 3804216850635206858, tmp_9161[1] * 3804216850635206858, tmp_9161[2] * 3804216850635206858];
    signal tmp_9605[3] <== [tmp_9603[0] + tmp_9604[0], tmp_9603[1] + tmp_9604[1], tmp_9603[2] + tmp_9604[2]];
    signal tmp_9606[3] <== [evals[126][0] * 12588714469852125705, evals[126][1] * 12588714469852125705, evals[126][2] * 12588714469852125705];
    signal tmp_9607[3] <== [tmp_9605[0] + tmp_9606[0], tmp_9605[1] + tmp_9606[1], tmp_9605[2] + tmp_9606[2]];
    signal tmp_9608[3] <== [evals[57][0] * 1481011955045677708, evals[57][1] * 1481011955045677708, evals[57][2] * 1481011955045677708];
    signal tmp_9609[3] <== [tmp_9607[0] + tmp_9608[0], tmp_9607[1] + tmp_9608[1], tmp_9607[2] + tmp_9608[2]];
    signal tmp_9610[3] <== [tmp_9602[0] + tmp_9609[0], tmp_9602[1] + tmp_9609[1], tmp_9602[2] + tmp_9609[2]];
    signal tmp_9611[3] <== [2 * tmp_9610[0], 2 * tmp_9610[1], 2 * tmp_9610[2]];
    signal tmp_9612[3] <== [tmp_9596[0] + tmp_9611[0], tmp_9596[1] + tmp_9611[1], tmp_9596[2] + tmp_9611[2]];
    signal tmp_9613[3] <== [tmp_9501[0] * 2027625550790675754, tmp_9501[1] * 2027625550790675754, tmp_9501[2] * 2027625550790675754];
    signal tmp_9614[3] <== [evals[90][0] + tmp_9613[0], evals[90][1] + tmp_9613[1], evals[90][2] + tmp_9613[2]];
    signal tmp_9615[3] <== CMul()(tmp_9614, tmp_9614);
    signal tmp_9616[3] <== CMul()(tmp_9615, tmp_9615);
    signal tmp_9617[3] <== CMul()(tmp_9616, tmp_9615);
    signal tmp_9618[3] <== CMul()(tmp_9617, tmp_9614);
    signal tmp_9619[3] <== [tmp_9501[0] * 2180104264643618635, tmp_9501[1] * 2180104264643618635, tmp_9501[2] * 2180104264643618635];
    signal tmp_9620[3] <== [tmp_9161[0] * 15878441634750804699, tmp_9161[1] * 15878441634750804699, tmp_9161[2] * 15878441634750804699];
    signal tmp_9621[3] <== [tmp_9619[0] + tmp_9620[0], tmp_9619[1] + tmp_9620[1], tmp_9619[2] + tmp_9620[2]];
    signal tmp_9622[3] <== [evals[126][0] * 9297270292248179409, evals[126][1] * 9297270292248179409, evals[126][2] * 9297270292248179409];
    signal tmp_9623[3] <== [tmp_9621[0] + tmp_9622[0], tmp_9621[1] + tmp_9622[1], tmp_9621[2] + tmp_9622[2]];
    signal tmp_9624[3] <== [evals[57][0] * 14060739868329031365, evals[57][1] * 14060739868329031365, evals[57][2] * 14060739868329031365];
    signal tmp_9625[3] <== [tmp_9623[0] + tmp_9624[0], tmp_9623[1] + tmp_9624[1], tmp_9623[2] + tmp_9624[2]];
    signal tmp_9626[3] <== [tmp_9618[0] + tmp_9625[0], tmp_9618[1] + tmp_9625[1], tmp_9618[2] + tmp_9625[2]];
    signal tmp_9627[3] <== [tmp_9612[0] + tmp_9626[0], tmp_9612[1] + tmp_9626[1], tmp_9612[2] + tmp_9626[2]];
    signal tmp_9628[3] <== [tmp_9501[0] * 3983470257916202094, tmp_9501[1] * 3983470257916202094, tmp_9501[2] * 3983470257916202094];
    signal tmp_9629[3] <== [evals[91][0] + tmp_9628[0], evals[91][1] + tmp_9628[1], evals[91][2] + tmp_9628[2]];
    signal tmp_9630[3] <== CMul()(tmp_9629, tmp_9629);
    signal tmp_9631[3] <== CMul()(tmp_9630, tmp_9630);
    signal tmp_9632[3] <== CMul()(tmp_9631, tmp_9630);
    signal tmp_9633[3] <== CMul()(tmp_9632, tmp_9629);
    signal tmp_9634[3] <== [tmp_9501[0] * 17674822478918492543, tmp_9501[1] * 17674822478918492543, tmp_9501[2] * 17674822478918492543];
    signal tmp_9635[3] <== [tmp_9161[0] * 10783876336999051732, tmp_9161[1] * 10783876336999051732, tmp_9161[2] * 10783876336999051732];
    signal tmp_9636[3] <== [tmp_9634[0] + tmp_9635[0], tmp_9634[1] + tmp_9635[1], tmp_9634[2] + tmp_9635[2]];
    signal tmp_9637[3] <== [evals[126][0] * 17023507959532517322, evals[126][1] * 17023507959532517322, evals[126][2] * 17023507959532517322];
    signal tmp_9638[3] <== [tmp_9636[0] + tmp_9637[0], tmp_9636[1] + tmp_9637[1], tmp_9636[2] + tmp_9637[2]];
    signal tmp_9639[3] <== [evals[57][0] * 13581824287786003086, evals[57][1] * 13581824287786003086, evals[57][2] * 13581824287786003086];
    signal tmp_9640[3] <== [tmp_9638[0] + tmp_9639[0], tmp_9638[1] + tmp_9639[1], tmp_9638[2] + tmp_9639[2]];
    signal tmp_9641[3] <== [tmp_9633[0] + tmp_9640[0], tmp_9633[1] + tmp_9640[1], tmp_9633[2] + tmp_9640[2]];
    signal tmp_9642[3] <== [101 * tmp_9641[0], 101 * tmp_9641[1], 101 * tmp_9641[2]];
    signal tmp_9643[3] <== [tmp_9627[0] + tmp_9642[0], tmp_9627[1] + tmp_9642[1], tmp_9627[2] + tmp_9642[2]];
    signal tmp_9644[3] <== [tmp_9501[0] * 3423470109396435354, tmp_9501[1] * 3423470109396435354, tmp_9501[2] * 3423470109396435354];
    signal tmp_9645[3] <== [evals[92][0] + tmp_9644[0], evals[92][1] + tmp_9644[1], evals[92][2] + tmp_9644[2]];
    signal tmp_9646[3] <== CMul()(tmp_9645, tmp_9645);
    signal tmp_9647[3] <== CMul()(tmp_9646, tmp_9646);
    signal tmp_9648[3] <== CMul()(tmp_9647, tmp_9646);
    signal tmp_9649[3] <== CMul()(tmp_9648, tmp_9645);
    signal tmp_9650[3] <== [tmp_9501[0] * 18417266132335741394, tmp_9501[1] * 18417266132335741394, tmp_9501[2] * 18417266132335741394];
    signal tmp_9651[3] <== [tmp_9161[0] * 6249355405258274459, tmp_9161[1] * 6249355405258274459, tmp_9161[2] * 6249355405258274459];
    signal tmp_9652[3] <== [tmp_9650[0] + tmp_9651[0], tmp_9650[1] + tmp_9651[1], tmp_9650[2] + tmp_9651[2]];
    signal tmp_9653[3] <== [evals[126][0] * 9382136940971401256, evals[126][1] * 9382136940971401256, evals[126][2] * 9382136940971401256];
    signal tmp_9654[3] <== [tmp_9652[0] + tmp_9653[0], tmp_9652[1] + tmp_9653[1], tmp_9652[2] + tmp_9653[2]];
    signal tmp_9655[3] <== [evals[57][0] * 101626575556905271, evals[57][1] * 101626575556905271, evals[57][2] * 101626575556905271];
    signal tmp_9656[3] <== [tmp_9654[0] + tmp_9655[0], tmp_9654[1] + tmp_9655[1], tmp_9654[2] + tmp_9655[2]];
    signal tmp_9657[3] <== [tmp_9649[0] + tmp_9656[0], tmp_9649[1] + tmp_9656[1], tmp_9649[2] + tmp_9656[2]];
    signal tmp_9658[3] <== [63 * tmp_9657[0], 63 * tmp_9657[1], 63 * tmp_9657[2]];
    signal tmp_9659[3] <== [tmp_9643[0] + tmp_9658[0], tmp_9643[1] + tmp_9658[1], tmp_9643[2] + tmp_9658[2]];
    signal tmp_9660[3] <== [tmp_9501[0] * 3450488264035752368, tmp_9501[1] * 3450488264035752368, tmp_9501[2] * 3450488264035752368];
    signal tmp_9661[3] <== [evals[93][0] + tmp_9660[0], evals[93][1] + tmp_9660[1], evals[93][2] + tmp_9660[2]];
    signal tmp_9662[3] <== CMul()(tmp_9661, tmp_9661);
    signal tmp_9663[3] <== CMul()(tmp_9662, tmp_9662);
    signal tmp_9664[3] <== CMul()(tmp_9663, tmp_9662);
    signal tmp_9665[3] <== CMul()(tmp_9664, tmp_9661);
    signal tmp_9666[3] <== [tmp_9501[0] * 11589569604681103119, tmp_9501[1] * 11589569604681103119, tmp_9501[2] * 11589569604681103119];
    signal tmp_9667[3] <== [tmp_9161[0] * 18338300173971397155, tmp_9161[1] * 18338300173971397155, tmp_9161[2] * 18338300173971397155];
    signal tmp_9668[3] <== [tmp_9666[0] + tmp_9667[0], tmp_9666[1] + tmp_9667[1], tmp_9666[2] + tmp_9667[2]];
    signal tmp_9669[3] <== [evals[126][0] * 14883732890265648356, evals[126][1] * 14883732890265648356, evals[126][2] * 14883732890265648356];
    signal tmp_9670[3] <== [tmp_9668[0] + tmp_9669[0], tmp_9668[1] + tmp_9669[1], tmp_9668[2] + tmp_9669[2]];
    signal tmp_9671[3] <== [evals[57][0] * 1195190930009648444, evals[57][1] * 1195190930009648444, evals[57][2] * 1195190930009648444];
    signal tmp_9672[3] <== [tmp_9670[0] + tmp_9671[0], tmp_9670[1] + tmp_9671[1], tmp_9670[2] + tmp_9671[2]];
    signal tmp_9673[3] <== [tmp_9665[0] + tmp_9672[0], tmp_9665[1] + tmp_9672[1], tmp_9665[2] + tmp_9672[2]];
    signal tmp_9674[3] <== [15 * tmp_9673[0], 15 * tmp_9673[1], 15 * tmp_9673[2]];
    signal tmp_9675[3] <== [tmp_9659[0] + tmp_9674[0], tmp_9659[1] + tmp_9674[1], tmp_9659[2] + tmp_9674[2]];
    signal tmp_9676[3] <== [tmp_9501[0] * 3151070045406026687, tmp_9501[1] * 3151070045406026687, tmp_9501[2] * 3151070045406026687];
    signal tmp_9677[3] <== [evals[94][0] + tmp_9676[0], evals[94][1] + tmp_9676[1], evals[94][2] + tmp_9676[2]];
    signal tmp_9678[3] <== CMul()(tmp_9677, tmp_9677);
    signal tmp_9679[3] <== CMul()(tmp_9678, tmp_9678);
    signal tmp_9680[3] <== CMul()(tmp_9679, tmp_9678);
    signal tmp_9681[3] <== CMul()(tmp_9680, tmp_9677);
    signal tmp_9682[3] <== [tmp_9501[0] * 18392483772870992139, tmp_9501[1] * 18392483772870992139, tmp_9501[2] * 18392483772870992139];
    signal tmp_9683[3] <== [tmp_9161[0] * 18008467388369492623, tmp_9161[1] * 18008467388369492623, tmp_9161[2] * 18008467388369492623];
    signal tmp_9684[3] <== [tmp_9682[0] + tmp_9683[0], tmp_9682[1] + tmp_9683[1], tmp_9682[2] + tmp_9683[2]];
    signal tmp_9685[3] <== [evals[126][0] * 17444658652079921032, evals[126][1] * 17444658652079921032, evals[126][2] * 17444658652079921032];
    signal tmp_9686[3] <== [tmp_9684[0] + tmp_9685[0], tmp_9684[1] + tmp_9685[1], tmp_9684[2] + tmp_9685[2]];
    signal tmp_9687[3] <== [evals[57][0] * 1911629430057473607, evals[57][1] * 1911629430057473607, evals[57][2] * 1911629430057473607];
    signal tmp_9688[3] <== [tmp_9686[0] + tmp_9687[0], tmp_9686[1] + tmp_9687[1], tmp_9686[2] + tmp_9687[2]];
    signal tmp_9689[3] <== [tmp_9681[0] + tmp_9688[0], tmp_9681[1] + tmp_9688[1], tmp_9681[2] + tmp_9688[2]];
    signal tmp_9690[3] <== [2 * tmp_9689[0], 2 * tmp_9689[1], 2 * tmp_9689[2]];
    signal tmp_9691[3] <== [tmp_9675[0] + tmp_9690[0], tmp_9675[1] + tmp_9690[1], tmp_9675[2] + tmp_9690[2]];
    signal tmp_9692[3] <== [tmp_9501[0] * 13462781804006123550, tmp_9501[1] * 13462781804006123550, tmp_9501[2] * 13462781804006123550];
    signal tmp_9693[3] <== [evals[95][0] + tmp_9692[0], evals[95][1] + tmp_9692[1], evals[95][2] + tmp_9692[2]];
    signal tmp_9694[3] <== CMul()(tmp_9693, tmp_9693);
    signal tmp_9695[3] <== CMul()(tmp_9694, tmp_9694);
    signal tmp_9696[3] <== CMul()(tmp_9695, tmp_9694);
    signal tmp_9697[3] <== CMul()(tmp_9696, tmp_9693);
    signal tmp_9698[3] <== [tmp_9501[0] * 9901024618406354614, tmp_9501[1] * 9901024618406354614, tmp_9501[2] * 9901024618406354614];
    signal tmp_9699[3] <== [tmp_9161[0] * 5454689878460956614, tmp_9161[1] * 5454689878460956614, tmp_9161[2] * 5454689878460956614];
    signal tmp_9700[3] <== [tmp_9698[0] + tmp_9699[0], tmp_9698[1] + tmp_9699[1], tmp_9698[2] + tmp_9699[2]];
    signal tmp_9701[3] <== [evals[126][0] * 8737622480559860280, evals[126][1] * 8737622480559860280, evals[126][2] * 8737622480559860280];
    signal tmp_9702[3] <== [tmp_9700[0] + tmp_9701[0], tmp_9700[1] + tmp_9701[1], tmp_9700[2] + tmp_9701[2]];
    signal tmp_9703[3] <== [evals[57][0] * 4398090650023947233, evals[57][1] * 4398090650023947233, evals[57][2] * 4398090650023947233];
    signal tmp_9704[3] <== [tmp_9702[0] + tmp_9703[0], tmp_9702[1] + tmp_9703[1], tmp_9702[2] + tmp_9703[2]];
    signal tmp_9705[3] <== [tmp_9697[0] + tmp_9704[0], tmp_9697[1] + tmp_9704[1], tmp_9697[2] + tmp_9704[2]];
    signal tmp_9706[3] <== [67 * tmp_9705[0], 67 * tmp_9705[1], 67 * tmp_9705[2]];
    signal tmp_9707[3] <== [tmp_9691[0] + tmp_9706[0], tmp_9691[1] + tmp_9706[1], tmp_9691[2] + tmp_9706[2]];
    signal tmp_9708[3] <== [tmp_9501[0] * 13288575772684627216, tmp_9501[1] * 13288575772684627216, tmp_9501[2] * 13288575772684627216];
    signal tmp_9709[3] <== [evals[96][0] + tmp_9708[0], evals[96][1] + tmp_9708[1], evals[96][2] + tmp_9708[2]];
    signal tmp_9710[3] <== CMul()(tmp_9709, tmp_9709);
    signal tmp_9711[3] <== CMul()(tmp_9710, tmp_9710);
    signal tmp_9712[3] <== CMul()(tmp_9711, tmp_9710);
    signal tmp_9713[3] <== CMul()(tmp_9712, tmp_9709);
    signal tmp_9714[3] <== [tmp_9501[0] * 841962207097860172, tmp_9501[1] * 841962207097860172, tmp_9501[2] * 841962207097860172];
    signal tmp_9715[3] <== [tmp_9161[0] * 15539651071574326605, tmp_9161[1] * 15539651071574326605, tmp_9161[2] * 15539651071574326605];
    signal tmp_9716[3] <== [tmp_9714[0] + tmp_9715[0], tmp_9714[1] + tmp_9715[1], tmp_9714[2] + tmp_9715[2]];
    signal tmp_9717[3] <== [evals[126][0] * 9445526981937155820, evals[126][1] * 9445526981937155820, evals[126][2] * 9445526981937155820];
    signal tmp_9718[3] <== [tmp_9716[0] + tmp_9717[0], tmp_9716[1] + tmp_9717[1], tmp_9716[2] + tmp_9717[2]];
    signal tmp_9719[3] <== [evals[57][0] * 5194367552957880421, evals[57][1] * 5194367552957880421, evals[57][2] * 5194367552957880421];
    signal tmp_9720[3] <== [tmp_9718[0] + tmp_9719[0], tmp_9718[1] + tmp_9719[1], tmp_9718[2] + tmp_9719[2]];
    signal tmp_9721[3] <== [tmp_9713[0] + tmp_9720[0], tmp_9713[1] + tmp_9720[1], tmp_9713[2] + tmp_9720[2]];
    signal tmp_9722[3] <== [22 * tmp_9721[0], 22 * tmp_9721[1], 22 * tmp_9721[2]];
    signal tmp_9723[3] <== [tmp_9707[0] + tmp_9722[0], tmp_9707[1] + tmp_9722[1], tmp_9707[2] + tmp_9722[2]];
    signal tmp_9724[3] <== [tmp_9501[0] * 13745549378090937523, tmp_9501[1] * 13745549378090937523, tmp_9501[2] * 13745549378090937523];
    signal tmp_9725[3] <== [evals[97][0] + tmp_9724[0], evals[97][1] + tmp_9724[1], evals[97][2] + tmp_9724[2]];
    signal tmp_9726[3] <== CMul()(tmp_9725, tmp_9725);
    signal tmp_9727[3] <== CMul()(tmp_9726, tmp_9726);
    signal tmp_9728[3] <== CMul()(tmp_9727, tmp_9726);
    signal tmp_9729[3] <== CMul()(tmp_9728, tmp_9725);
    signal tmp_9730[3] <== [tmp_9501[0] * 1419811072696625524, tmp_9501[1] * 1419811072696625524, tmp_9501[2] * 1419811072696625524];
    signal tmp_9731[3] <== [tmp_9161[0] * 9583365915190359557, tmp_9161[1] * 9583365915190359557, tmp_9161[2] * 9583365915190359557];
    signal tmp_9732[3] <== [tmp_9730[0] + tmp_9731[0], tmp_9730[1] + tmp_9731[1], tmp_9730[2] + tmp_9731[2]];
    signal tmp_9733[3] <== [evals[126][0] * 14154961912193710379, evals[126][1] * 14154961912193710379, evals[126][2] * 14154961912193710379];
    signal tmp_9734[3] <== [tmp_9732[0] + tmp_9733[0], tmp_9732[1] + tmp_9733[1], tmp_9732[2] + tmp_9733[2]];
    signal tmp_9735[3] <== [evals[57][0] * 7348962900982106793, evals[57][1] * 7348962900982106793, evals[57][2] * 7348962900982106793];
    signal tmp_9736[3] <== [tmp_9734[0] + tmp_9735[0], tmp_9734[1] + tmp_9735[1], tmp_9734[2] + tmp_9735[2]];
    signal tmp_9737[3] <== [tmp_9729[0] + tmp_9736[0], tmp_9729[1] + tmp_9736[1], tmp_9729[2] + tmp_9736[2]];
    signal tmp_9738[3] <== [13 * tmp_9737[0], 13 * tmp_9737[1], 13 * tmp_9737[2]];
    signal tmp_9739[3] <== [tmp_9723[0] + tmp_9738[0], tmp_9723[1] + tmp_9738[1], tmp_9723[2] + tmp_9738[2]];
    signal tmp_9740[3] <== [tmp_9501[0] * 7780139165529418388, tmp_9501[1] * 7780139165529418388, tmp_9501[2] * 7780139165529418388];
    signal tmp_9741[3] <== [evals[98][0] + tmp_9740[0], evals[98][1] + tmp_9740[1], evals[98][2] + tmp_9740[2]];
    signal tmp_9742[3] <== CMul()(tmp_9741, tmp_9741);
    signal tmp_9743[3] <== CMul()(tmp_9742, tmp_9742);
    signal tmp_9744[3] <== CMul()(tmp_9743, tmp_9742);
    signal tmp_9745[3] <== CMul()(tmp_9744, tmp_9741);
    signal tmp_9746[3] <== [tmp_9501[0] * 10322277472159862957, tmp_9501[1] * 10322277472159862957, tmp_9501[2] * 10322277472159862957];
    signal tmp_9747[3] <== [tmp_9161[0] * 1022758405964188465, tmp_9161[1] * 1022758405964188465, tmp_9161[2] * 1022758405964188465];
    signal tmp_9748[3] <== [tmp_9746[0] + tmp_9747[0], tmp_9746[1] + tmp_9747[1], tmp_9746[2] + tmp_9747[2]];
    signal tmp_9749[3] <== [evals[126][0] * 5087036264788036553, evals[126][1] * 5087036264788036553, evals[126][2] * 5087036264788036553];
    signal tmp_9750[3] <== [tmp_9748[0] + tmp_9749[0], tmp_9748[1] + tmp_9749[1], tmp_9748[2] + tmp_9749[2]];
    signal tmp_9751[3] <== [evals[57][0] * 9087509974770738024, evals[57][1] * 9087509974770738024, evals[57][2] * 9087509974770738024];
    signal tmp_9752[3] <== [tmp_9750[0] + tmp_9751[0], tmp_9750[1] + tmp_9751[1], tmp_9750[2] + tmp_9751[2]];
    signal tmp_9753[3] <== [tmp_9745[0] + tmp_9752[0], tmp_9745[1] + tmp_9752[1], tmp_9745[2] + tmp_9752[2]];
    signal tmp_9754[3] <== [3 * tmp_9753[0], 3 * tmp_9753[1], 3 * tmp_9753[2]];
    signal tmp_9755[3] <== [tmp_9739[0] + tmp_9754[0], tmp_9739[1] + tmp_9754[1], tmp_9739[2] + tmp_9754[2]];
    signal tmp_9756[3] <== [evals[99][0] - tmp_9755[0], evals[99][1] - tmp_9755[1], evals[99][2] - tmp_9755[2]];
    signal tmp_9757[3] <== CMul()(tmp_9504, tmp_9756);
    signal tmp_9758[3] <== [tmp_9500[0] + tmp_9757[0], tmp_9500[1] + tmp_9757[1], tmp_9500[2] + tmp_9757[2]];
    signal tmp_9759[3] <== CMul()(challengeQ, tmp_9758);
    signal tmp_9764[3] <== [3 * tmp_9518[0], 3 * tmp_9518[1], 3 * tmp_9518[2]];
    signal tmp_9766[3] <== [tmp_9764[0] + tmp_9532[0], tmp_9764[1] + tmp_9532[1], tmp_9764[2] + tmp_9532[2]];
    signal tmp_9768[3] <== [tmp_9766[0] + tmp_9547[0], tmp_9766[1] + tmp_9547[1], tmp_9766[2] + tmp_9547[2]];
    signal tmp_9770[3] <== [51 * tmp_9563[0], 51 * tmp_9563[1], 51 * tmp_9563[2]];
    signal tmp_9771[3] <== [tmp_9768[0] + tmp_9770[0], tmp_9768[1] + tmp_9770[1], tmp_9768[2] + tmp_9770[2]];
    signal tmp_9773[3] <== [tmp_9771[0] + tmp_9578[0], tmp_9771[1] + tmp_9578[1], tmp_9771[2] + tmp_9578[2]];
    signal tmp_9775[3] <== [11 * tmp_9594[0], 11 * tmp_9594[1], 11 * tmp_9594[2]];
    signal tmp_9776[3] <== [tmp_9773[0] + tmp_9775[0], tmp_9773[1] + tmp_9775[1], tmp_9773[2] + tmp_9775[2]];
    signal tmp_9778[3] <== [17 * tmp_9610[0], 17 * tmp_9610[1], 17 * tmp_9610[2]];
    signal tmp_9779[3] <== [tmp_9776[0] + tmp_9778[0], tmp_9776[1] + tmp_9778[1], tmp_9776[2] + tmp_9778[2]];
    signal tmp_9781[3] <== [2 * tmp_9626[0], 2 * tmp_9626[1], 2 * tmp_9626[2]];
    signal tmp_9782[3] <== [tmp_9779[0] + tmp_9781[0], tmp_9779[1] + tmp_9781[1], tmp_9779[2] + tmp_9781[2]];
    signal tmp_9784[3] <== [tmp_9782[0] + tmp_9641[0], tmp_9782[1] + tmp_9641[1], tmp_9782[2] + tmp_9641[2]];
    signal tmp_9786[3] <== [101 * tmp_9657[0], 101 * tmp_9657[1], 101 * tmp_9657[2]];
    signal tmp_9787[3] <== [tmp_9784[0] + tmp_9786[0], tmp_9784[1] + tmp_9786[1], tmp_9784[2] + tmp_9786[2]];
    signal tmp_9789[3] <== [63 * tmp_9673[0], 63 * tmp_9673[1], 63 * tmp_9673[2]];
    signal tmp_9790[3] <== [tmp_9787[0] + tmp_9789[0], tmp_9787[1] + tmp_9789[1], tmp_9787[2] + tmp_9789[2]];
    signal tmp_9792[3] <== [15 * tmp_9689[0], 15 * tmp_9689[1], 15 * tmp_9689[2]];
    signal tmp_9793[3] <== [tmp_9790[0] + tmp_9792[0], tmp_9790[1] + tmp_9792[1], tmp_9790[2] + tmp_9792[2]];
    signal tmp_9795[3] <== [2 * tmp_9705[0], 2 * tmp_9705[1], 2 * tmp_9705[2]];
    signal tmp_9796[3] <== [tmp_9793[0] + tmp_9795[0], tmp_9793[1] + tmp_9795[1], tmp_9793[2] + tmp_9795[2]];
    signal tmp_9798[3] <== [67 * tmp_9721[0], 67 * tmp_9721[1], 67 * tmp_9721[2]];
    signal tmp_9799[3] <== [tmp_9796[0] + tmp_9798[0], tmp_9796[1] + tmp_9798[1], tmp_9796[2] + tmp_9798[2]];
    signal tmp_9801[3] <== [22 * tmp_9737[0], 22 * tmp_9737[1], 22 * tmp_9737[2]];
    signal tmp_9802[3] <== [tmp_9799[0] + tmp_9801[0], tmp_9799[1] + tmp_9801[1], tmp_9799[2] + tmp_9801[2]];
    signal tmp_9804[3] <== [13 * tmp_9753[0], 13 * tmp_9753[1], 13 * tmp_9753[2]];
    signal tmp_9805[3] <== [tmp_9802[0] + tmp_9804[0], tmp_9802[1] + tmp_9804[1], tmp_9802[2] + tmp_9804[2]];
    signal tmp_9806[3] <== [evals[100][0] - tmp_9805[0], evals[100][1] - tmp_9805[1], evals[100][2] - tmp_9805[2]];
    signal tmp_9807[3] <== CMul()(tmp_9504, tmp_9806);
    signal tmp_9808[3] <== [tmp_9759[0] + tmp_9807[0], tmp_9759[1] + tmp_9807[1], tmp_9759[2] + tmp_9807[2]];
    signal tmp_9809[3] <== CMul()(challengeQ, tmp_9808);
    signal tmp_9814[3] <== [13 * tmp_9518[0], 13 * tmp_9518[1], 13 * tmp_9518[2]];
    signal tmp_9816[3] <== [3 * tmp_9532[0], 3 * tmp_9532[1], 3 * tmp_9532[2]];
    signal tmp_9817[3] <== [tmp_9814[0] + tmp_9816[0], tmp_9814[1] + tmp_9816[1], tmp_9814[2] + tmp_9816[2]];
    signal tmp_9819[3] <== [tmp_9817[0] + tmp_9547[0], tmp_9817[1] + tmp_9547[1], tmp_9817[2] + tmp_9547[2]];
    signal tmp_9821[3] <== [tmp_9819[0] + tmp_9563[0], tmp_9819[1] + tmp_9563[1], tmp_9819[2] + tmp_9563[2]];
    signal tmp_9823[3] <== [51 * tmp_9578[0], 51 * tmp_9578[1], 51 * tmp_9578[2]];
    signal tmp_9824[3] <== [tmp_9821[0] + tmp_9823[0], tmp_9821[1] + tmp_9823[1], tmp_9821[2] + tmp_9823[2]];
    signal tmp_9826[3] <== [tmp_9824[0] + tmp_9594[0], tmp_9824[1] + tmp_9594[1], tmp_9824[2] + tmp_9594[2]];
    signal tmp_9828[3] <== [11 * tmp_9610[0], 11 * tmp_9610[1], 11 * tmp_9610[2]];
    signal tmp_9829[3] <== [tmp_9826[0] + tmp_9828[0], tmp_9826[1] + tmp_9828[1], tmp_9826[2] + tmp_9828[2]];
    signal tmp_9831[3] <== [17 * tmp_9626[0], 17 * tmp_9626[1], 17 * tmp_9626[2]];
    signal tmp_9832[3] <== [tmp_9829[0] + tmp_9831[0], tmp_9829[1] + tmp_9831[1], tmp_9829[2] + tmp_9831[2]];
    signal tmp_9834[3] <== [2 * tmp_9641[0], 2 * tmp_9641[1], 2 * tmp_9641[2]];
    signal tmp_9835[3] <== [tmp_9832[0] + tmp_9834[0], tmp_9832[1] + tmp_9834[1], tmp_9832[2] + tmp_9834[2]];
    signal tmp_9837[3] <== [tmp_9835[0] + tmp_9657[0], tmp_9835[1] + tmp_9657[1], tmp_9835[2] + tmp_9657[2]];
    signal tmp_9839[3] <== [101 * tmp_9673[0], 101 * tmp_9673[1], 101 * tmp_9673[2]];
    signal tmp_9840[3] <== [tmp_9837[0] + tmp_9839[0], tmp_9837[1] + tmp_9839[1], tmp_9837[2] + tmp_9839[2]];
    signal tmp_9842[3] <== [63 * tmp_9689[0], 63 * tmp_9689[1], 63 * tmp_9689[2]];
    signal tmp_9843[3] <== [tmp_9840[0] + tmp_9842[0], tmp_9840[1] + tmp_9842[1], tmp_9840[2] + tmp_9842[2]];
    signal tmp_9845[3] <== [15 * tmp_9705[0], 15 * tmp_9705[1], 15 * tmp_9705[2]];
    signal tmp_9846[3] <== [tmp_9843[0] + tmp_9845[0], tmp_9843[1] + tmp_9845[1], tmp_9843[2] + tmp_9845[2]];
    signal tmp_9848[3] <== [2 * tmp_9721[0], 2 * tmp_9721[1], 2 * tmp_9721[2]];
    signal tmp_9849[3] <== [tmp_9846[0] + tmp_9848[0], tmp_9846[1] + tmp_9848[1], tmp_9846[2] + tmp_9848[2]];
    signal tmp_9851[3] <== [67 * tmp_9737[0], 67 * tmp_9737[1], 67 * tmp_9737[2]];
    signal tmp_9852[3] <== [tmp_9849[0] + tmp_9851[0], tmp_9849[1] + tmp_9851[1], tmp_9849[2] + tmp_9851[2]];
    signal tmp_9854[3] <== [22 * tmp_9753[0], 22 * tmp_9753[1], 22 * tmp_9753[2]];
    signal tmp_9855[3] <== [tmp_9852[0] + tmp_9854[0], tmp_9852[1] + tmp_9854[1], tmp_9852[2] + tmp_9854[2]];
    signal tmp_9856[3] <== [evals[101][0] - tmp_9855[0], evals[101][1] - tmp_9855[1], evals[101][2] - tmp_9855[2]];
    signal tmp_9857[3] <== CMul()(tmp_9504, tmp_9856);
    signal tmp_9858[3] <== [tmp_9809[0] + tmp_9857[0], tmp_9809[1] + tmp_9857[1], tmp_9809[2] + tmp_9857[2]];
    signal tmp_9859[3] <== CMul()(challengeQ, tmp_9858);
    signal tmp_9864[3] <== [22 * tmp_9518[0], 22 * tmp_9518[1], 22 * tmp_9518[2]];
    signal tmp_9866[3] <== [13 * tmp_9532[0], 13 * tmp_9532[1], 13 * tmp_9532[2]];
    signal tmp_9867[3] <== [tmp_9864[0] + tmp_9866[0], tmp_9864[1] + tmp_9866[1], tmp_9864[2] + tmp_9866[2]];
    signal tmp_9869[3] <== [3 * tmp_9547[0], 3 * tmp_9547[1], 3 * tmp_9547[2]];
    signal tmp_9870[3] <== [tmp_9867[0] + tmp_9869[0], tmp_9867[1] + tmp_9869[1], tmp_9867[2] + tmp_9869[2]];
    signal tmp_9872[3] <== [tmp_9870[0] + tmp_9563[0], tmp_9870[1] + tmp_9563[1], tmp_9870[2] + tmp_9563[2]];
    signal tmp_9874[3] <== [tmp_9872[0] + tmp_9578[0], tmp_9872[1] + tmp_9578[1], tmp_9872[2] + tmp_9578[2]];
    signal tmp_9876[3] <== [51 * tmp_9594[0], 51 * tmp_9594[1], 51 * tmp_9594[2]];
    signal tmp_9877[3] <== [tmp_9874[0] + tmp_9876[0], tmp_9874[1] + tmp_9876[1], tmp_9874[2] + tmp_9876[2]];
    signal tmp_9879[3] <== [tmp_9877[0] + tmp_9610[0], tmp_9877[1] + tmp_9610[1], tmp_9877[2] + tmp_9610[2]];
    signal tmp_9881[3] <== [11 * tmp_9626[0], 11 * tmp_9626[1], 11 * tmp_9626[2]];
    signal tmp_9882[3] <== [tmp_9879[0] + tmp_9881[0], tmp_9879[1] + tmp_9881[1], tmp_9879[2] + tmp_9881[2]];
    signal tmp_9884[3] <== [17 * tmp_9641[0], 17 * tmp_9641[1], 17 * tmp_9641[2]];
    signal tmp_9885[3] <== [tmp_9882[0] + tmp_9884[0], tmp_9882[1] + tmp_9884[1], tmp_9882[2] + tmp_9884[2]];
    signal tmp_9887[3] <== [2 * tmp_9657[0], 2 * tmp_9657[1], 2 * tmp_9657[2]];
    signal tmp_9888[3] <== [tmp_9885[0] + tmp_9887[0], tmp_9885[1] + tmp_9887[1], tmp_9885[2] + tmp_9887[2]];
    signal tmp_9890[3] <== [tmp_9888[0] + tmp_9673[0], tmp_9888[1] + tmp_9673[1], tmp_9888[2] + tmp_9673[2]];
    signal tmp_9892[3] <== [101 * tmp_9689[0], 101 * tmp_9689[1], 101 * tmp_9689[2]];
    signal tmp_9893[3] <== [tmp_9890[0] + tmp_9892[0], tmp_9890[1] + tmp_9892[1], tmp_9890[2] + tmp_9892[2]];
    signal tmp_9895[3] <== [63 * tmp_9705[0], 63 * tmp_9705[1], 63 * tmp_9705[2]];
    signal tmp_9896[3] <== [tmp_9893[0] + tmp_9895[0], tmp_9893[1] + tmp_9895[1], tmp_9893[2] + tmp_9895[2]];
    signal tmp_9898[3] <== [15 * tmp_9721[0], 15 * tmp_9721[1], 15 * tmp_9721[2]];
    signal tmp_9899[3] <== [tmp_9896[0] + tmp_9898[0], tmp_9896[1] + tmp_9898[1], tmp_9896[2] + tmp_9898[2]];
    signal tmp_9901[3] <== [2 * tmp_9737[0], 2 * tmp_9737[1], 2 * tmp_9737[2]];
    signal tmp_9902[3] <== [tmp_9899[0] + tmp_9901[0], tmp_9899[1] + tmp_9901[1], tmp_9899[2] + tmp_9901[2]];
    signal tmp_9904[3] <== [67 * tmp_9753[0], 67 * tmp_9753[1], 67 * tmp_9753[2]];
    signal tmp_9905[3] <== [tmp_9902[0] + tmp_9904[0], tmp_9902[1] + tmp_9904[1], tmp_9902[2] + tmp_9904[2]];
    signal tmp_9906[3] <== [evals[102][0] - tmp_9905[0], evals[102][1] - tmp_9905[1], evals[102][2] - tmp_9905[2]];
    signal tmp_9907[3] <== CMul()(tmp_9504, tmp_9906);
    signal tmp_9908[3] <== [tmp_9859[0] + tmp_9907[0], tmp_9859[1] + tmp_9907[1], tmp_9859[2] + tmp_9907[2]];
    signal tmp_9909[3] <== CMul()(challengeQ, tmp_9908);
    signal tmp_9914[3] <== [67 * tmp_9518[0], 67 * tmp_9518[1], 67 * tmp_9518[2]];
    signal tmp_9916[3] <== [22 * tmp_9532[0], 22 * tmp_9532[1], 22 * tmp_9532[2]];
    signal tmp_9917[3] <== [tmp_9914[0] + tmp_9916[0], tmp_9914[1] + tmp_9916[1], tmp_9914[2] + tmp_9916[2]];
    signal tmp_9919[3] <== [13 * tmp_9547[0], 13 * tmp_9547[1], 13 * tmp_9547[2]];
    signal tmp_9920[3] <== [tmp_9917[0] + tmp_9919[0], tmp_9917[1] + tmp_9919[1], tmp_9917[2] + tmp_9919[2]];
    signal tmp_9922[3] <== [3 * tmp_9563[0], 3 * tmp_9563[1], 3 * tmp_9563[2]];
    signal tmp_9923[3] <== [tmp_9920[0] + tmp_9922[0], tmp_9920[1] + tmp_9922[1], tmp_9920[2] + tmp_9922[2]];
    signal tmp_9925[3] <== [tmp_9923[0] + tmp_9578[0], tmp_9923[1] + tmp_9578[1], tmp_9923[2] + tmp_9578[2]];
    signal tmp_9927[3] <== [tmp_9925[0] + tmp_9594[0], tmp_9925[1] + tmp_9594[1], tmp_9925[2] + tmp_9594[2]];
    signal tmp_9929[3] <== [51 * tmp_9610[0], 51 * tmp_9610[1], 51 * tmp_9610[2]];
    signal tmp_9930[3] <== [tmp_9927[0] + tmp_9929[0], tmp_9927[1] + tmp_9929[1], tmp_9927[2] + tmp_9929[2]];
    signal tmp_9932[3] <== [tmp_9930[0] + tmp_9626[0], tmp_9930[1] + tmp_9626[1], tmp_9930[2] + tmp_9626[2]];
    signal tmp_9934[3] <== [11 * tmp_9641[0], 11 * tmp_9641[1], 11 * tmp_9641[2]];
    signal tmp_9935[3] <== [tmp_9932[0] + tmp_9934[0], tmp_9932[1] + tmp_9934[1], tmp_9932[2] + tmp_9934[2]];
    signal tmp_9937[3] <== [17 * tmp_9657[0], 17 * tmp_9657[1], 17 * tmp_9657[2]];
    signal tmp_9938[3] <== [tmp_9935[0] + tmp_9937[0], tmp_9935[1] + tmp_9937[1], tmp_9935[2] + tmp_9937[2]];
    signal tmp_9940[3] <== [2 * tmp_9673[0], 2 * tmp_9673[1], 2 * tmp_9673[2]];
    signal tmp_9941[3] <== [tmp_9938[0] + tmp_9940[0], tmp_9938[1] + tmp_9940[1], tmp_9938[2] + tmp_9940[2]];
    signal tmp_9943[3] <== [tmp_9941[0] + tmp_9689[0], tmp_9941[1] + tmp_9689[1], tmp_9941[2] + tmp_9689[2]];
    signal tmp_9945[3] <== [101 * tmp_9705[0], 101 * tmp_9705[1], 101 * tmp_9705[2]];
    signal tmp_9946[3] <== [tmp_9943[0] + tmp_9945[0], tmp_9943[1] + tmp_9945[1], tmp_9943[2] + tmp_9945[2]];
    signal tmp_9948[3] <== [63 * tmp_9721[0], 63 * tmp_9721[1], 63 * tmp_9721[2]];
    signal tmp_9949[3] <== [tmp_9946[0] + tmp_9948[0], tmp_9946[1] + tmp_9948[1], tmp_9946[2] + tmp_9948[2]];
    signal tmp_9951[3] <== [15 * tmp_9737[0], 15 * tmp_9737[1], 15 * tmp_9737[2]];
    signal tmp_9952[3] <== [tmp_9949[0] + tmp_9951[0], tmp_9949[1] + tmp_9951[1], tmp_9949[2] + tmp_9951[2]];
    signal tmp_9954[3] <== [2 * tmp_9753[0], 2 * tmp_9753[1], 2 * tmp_9753[2]];
    signal tmp_9955[3] <== [tmp_9952[0] + tmp_9954[0], tmp_9952[1] + tmp_9954[1], tmp_9952[2] + tmp_9954[2]];
    signal tmp_9956[3] <== [evals[103][0] - tmp_9955[0], evals[103][1] - tmp_9955[1], evals[103][2] - tmp_9955[2]];
    signal tmp_9957[3] <== CMul()(tmp_9504, tmp_9956);
    signal tmp_9958[3] <== [tmp_9909[0] + tmp_9957[0], tmp_9909[1] + tmp_9957[1], tmp_9909[2] + tmp_9957[2]];
    signal tmp_9959[3] <== CMul()(challengeQ, tmp_9958);
    signal tmp_9964[3] <== [2 * tmp_9518[0], 2 * tmp_9518[1], 2 * tmp_9518[2]];
    signal tmp_9966[3] <== [67 * tmp_9532[0], 67 * tmp_9532[1], 67 * tmp_9532[2]];
    signal tmp_9967[3] <== [tmp_9964[0] + tmp_9966[0], tmp_9964[1] + tmp_9966[1], tmp_9964[2] + tmp_9966[2]];
    signal tmp_9969[3] <== [22 * tmp_9547[0], 22 * tmp_9547[1], 22 * tmp_9547[2]];
    signal tmp_9970[3] <== [tmp_9967[0] + tmp_9969[0], tmp_9967[1] + tmp_9969[1], tmp_9967[2] + tmp_9969[2]];
    signal tmp_9972[3] <== [13 * tmp_9563[0], 13 * tmp_9563[1], 13 * tmp_9563[2]];
    signal tmp_9973[3] <== [tmp_9970[0] + tmp_9972[0], tmp_9970[1] + tmp_9972[1], tmp_9970[2] + tmp_9972[2]];
    signal tmp_9975[3] <== [3 * tmp_9578[0], 3 * tmp_9578[1], 3 * tmp_9578[2]];
    signal tmp_9976[3] <== [tmp_9973[0] + tmp_9975[0], tmp_9973[1] + tmp_9975[1], tmp_9973[2] + tmp_9975[2]];
    signal tmp_9978[3] <== [tmp_9976[0] + tmp_9594[0], tmp_9976[1] + tmp_9594[1], tmp_9976[2] + tmp_9594[2]];
    signal tmp_9980[3] <== [tmp_9978[0] + tmp_9610[0], tmp_9978[1] + tmp_9610[1], tmp_9978[2] + tmp_9610[2]];
    signal tmp_9982[3] <== [51 * tmp_9626[0], 51 * tmp_9626[1], 51 * tmp_9626[2]];
    signal tmp_9983[3] <== [tmp_9980[0] + tmp_9982[0], tmp_9980[1] + tmp_9982[1], tmp_9980[2] + tmp_9982[2]];
    signal tmp_9985[3] <== [tmp_9983[0] + tmp_9641[0], tmp_9983[1] + tmp_9641[1], tmp_9983[2] + tmp_9641[2]];
    signal tmp_9987[3] <== [11 * tmp_9657[0], 11 * tmp_9657[1], 11 * tmp_9657[2]];
    signal tmp_9988[3] <== [tmp_9985[0] + tmp_9987[0], tmp_9985[1] + tmp_9987[1], tmp_9985[2] + tmp_9987[2]];
    signal tmp_9990[3] <== [17 * tmp_9673[0], 17 * tmp_9673[1], 17 * tmp_9673[2]];
    signal tmp_9991[3] <== [tmp_9988[0] + tmp_9990[0], tmp_9988[1] + tmp_9990[1], tmp_9988[2] + tmp_9990[2]];
    signal tmp_9994[3] <== [tmp_9991[0] + tmp_9690[0], tmp_9991[1] + tmp_9690[1], tmp_9991[2] + tmp_9690[2]];
    signal tmp_9996[3] <== [tmp_9994[0] + tmp_9705[0], tmp_9994[1] + tmp_9705[1], tmp_9994[2] + tmp_9705[2]];
    signal tmp_9998[3] <== [101 * tmp_9721[0], 101 * tmp_9721[1], 101 * tmp_9721[2]];
    signal tmp_9999[3] <== [tmp_9996[0] + tmp_9998[0], tmp_9996[1] + tmp_9998[1], tmp_9996[2] + tmp_9998[2]];
    signal tmp_10001[3] <== [63 * tmp_9737[0], 63 * tmp_9737[1], 63 * tmp_9737[2]];
    signal tmp_10002[3] <== [tmp_9999[0] + tmp_10001[0], tmp_9999[1] + tmp_10001[1], tmp_9999[2] + tmp_10001[2]];
    signal tmp_10004[3] <== [15 * tmp_9753[0], 15 * tmp_9753[1], 15 * tmp_9753[2]];
    signal tmp_10005[3] <== [tmp_10002[0] + tmp_10004[0], tmp_10002[1] + tmp_10004[1], tmp_10002[2] + tmp_10004[2]];
    signal tmp_10006[3] <== [evals[104][0] - tmp_10005[0], evals[104][1] - tmp_10005[1], evals[104][2] - tmp_10005[2]];
    signal tmp_10007[3] <== CMul()(tmp_9504, tmp_10006);
    signal tmp_10008[3] <== [tmp_9959[0] + tmp_10007[0], tmp_9959[1] + tmp_10007[1], tmp_9959[2] + tmp_10007[2]];
    signal tmp_10009[3] <== CMul()(challengeQ, tmp_10008);
    signal tmp_10014[3] <== [15 * tmp_9518[0], 15 * tmp_9518[1], 15 * tmp_9518[2]];
    signal tmp_10016[3] <== [2 * tmp_9532[0], 2 * tmp_9532[1], 2 * tmp_9532[2]];
    signal tmp_10017[3] <== [tmp_10014[0] + tmp_10016[0], tmp_10014[1] + tmp_10016[1], tmp_10014[2] + tmp_10016[2]];
    signal tmp_10019[3] <== [67 * tmp_9547[0], 67 * tmp_9547[1], 67 * tmp_9547[2]];
    signal tmp_10020[3] <== [tmp_10017[0] + tmp_10019[0], tmp_10017[1] + tmp_10019[1], tmp_10017[2] + tmp_10019[2]];
    signal tmp_10022[3] <== [22 * tmp_9563[0], 22 * tmp_9563[1], 22 * tmp_9563[2]];
    signal tmp_10023[3] <== [tmp_10020[0] + tmp_10022[0], tmp_10020[1] + tmp_10022[1], tmp_10020[2] + tmp_10022[2]];
    signal tmp_10025[3] <== [13 * tmp_9578[0], 13 * tmp_9578[1], 13 * tmp_9578[2]];
    signal tmp_10026[3] <== [tmp_10023[0] + tmp_10025[0], tmp_10023[1] + tmp_10025[1], tmp_10023[2] + tmp_10025[2]];
    signal tmp_10028[3] <== [3 * tmp_9594[0], 3 * tmp_9594[1], 3 * tmp_9594[2]];
    signal tmp_10029[3] <== [tmp_10026[0] + tmp_10028[0], tmp_10026[1] + tmp_10028[1], tmp_10026[2] + tmp_10028[2]];
    signal tmp_10031[3] <== [tmp_10029[0] + tmp_9610[0], tmp_10029[1] + tmp_9610[1], tmp_10029[2] + tmp_9610[2]];
    signal tmp_10033[3] <== [tmp_10031[0] + tmp_9626[0], tmp_10031[1] + tmp_9626[1], tmp_10031[2] + tmp_9626[2]];
    signal tmp_10035[3] <== [51 * tmp_9641[0], 51 * tmp_9641[1], 51 * tmp_9641[2]];
    signal tmp_10036[3] <== [tmp_10033[0] + tmp_10035[0], tmp_10033[1] + tmp_10035[1], tmp_10033[2] + tmp_10035[2]];
    signal tmp_10038[3] <== [tmp_10036[0] + tmp_9657[0], tmp_10036[1] + tmp_9657[1], tmp_10036[2] + tmp_9657[2]];
    signal tmp_10040[3] <== [11 * tmp_9673[0], 11 * tmp_9673[1], 11 * tmp_9673[2]];
    signal tmp_10041[3] <== [tmp_10038[0] + tmp_10040[0], tmp_10038[1] + tmp_10040[1], tmp_10038[2] + tmp_10040[2]];
    signal tmp_10043[3] <== [17 * tmp_9689[0], 17 * tmp_9689[1], 17 * tmp_9689[2]];
    signal tmp_10044[3] <== [tmp_10041[0] + tmp_10043[0], tmp_10041[1] + tmp_10043[1], tmp_10041[2] + tmp_10043[2]];
    signal tmp_10047[3] <== [tmp_10044[0] + tmp_9795[0], tmp_10044[1] + tmp_9795[1], tmp_10044[2] + tmp_9795[2]];
    signal tmp_10049[3] <== [tmp_10047[0] + tmp_9721[0], tmp_10047[1] + tmp_9721[1], tmp_10047[2] + tmp_9721[2]];
    signal tmp_10051[3] <== [101 * tmp_9737[0], 101 * tmp_9737[1], 101 * tmp_9737[2]];
    signal tmp_10052[3] <== [tmp_10049[0] + tmp_10051[0], tmp_10049[1] + tmp_10051[1], tmp_10049[2] + tmp_10051[2]];
    signal tmp_10054[3] <== [63 * tmp_9753[0], 63 * tmp_9753[1], 63 * tmp_9753[2]];
    signal tmp_10055[3] <== [tmp_10052[0] + tmp_10054[0], tmp_10052[1] + tmp_10054[1], tmp_10052[2] + tmp_10054[2]];
    signal tmp_10056[3] <== [evals[105][0] - tmp_10055[0], evals[105][1] - tmp_10055[1], evals[105][2] - tmp_10055[2]];
    signal tmp_10057[3] <== CMul()(tmp_9504, tmp_10056);
    signal tmp_10058[3] <== [tmp_10009[0] + tmp_10057[0], tmp_10009[1] + tmp_10057[1], tmp_10009[2] + tmp_10057[2]];
    signal tmp_10059[3] <== CMul()(challengeQ, tmp_10058);
    signal tmp_10064[3] <== [63 * tmp_9518[0], 63 * tmp_9518[1], 63 * tmp_9518[2]];
    signal tmp_10066[3] <== [15 * tmp_9532[0], 15 * tmp_9532[1], 15 * tmp_9532[2]];
    signal tmp_10067[3] <== [tmp_10064[0] + tmp_10066[0], tmp_10064[1] + tmp_10066[1], tmp_10064[2] + tmp_10066[2]];
    signal tmp_10069[3] <== [2 * tmp_9547[0], 2 * tmp_9547[1], 2 * tmp_9547[2]];
    signal tmp_10070[3] <== [tmp_10067[0] + tmp_10069[0], tmp_10067[1] + tmp_10069[1], tmp_10067[2] + tmp_10069[2]];
    signal tmp_10072[3] <== [67 * tmp_9563[0], 67 * tmp_9563[1], 67 * tmp_9563[2]];
    signal tmp_10073[3] <== [tmp_10070[0] + tmp_10072[0], tmp_10070[1] + tmp_10072[1], tmp_10070[2] + tmp_10072[2]];
    signal tmp_10075[3] <== [22 * tmp_9578[0], 22 * tmp_9578[1], 22 * tmp_9578[2]];
    signal tmp_10076[3] <== [tmp_10073[0] + tmp_10075[0], tmp_10073[1] + tmp_10075[1], tmp_10073[2] + tmp_10075[2]];
    signal tmp_10078[3] <== [13 * tmp_9594[0], 13 * tmp_9594[1], 13 * tmp_9594[2]];
    signal tmp_10079[3] <== [tmp_10076[0] + tmp_10078[0], tmp_10076[1] + tmp_10078[1], tmp_10076[2] + tmp_10078[2]];
    signal tmp_10081[3] <== [3 * tmp_9610[0], 3 * tmp_9610[1], 3 * tmp_9610[2]];
    signal tmp_10082[3] <== [tmp_10079[0] + tmp_10081[0], tmp_10079[1] + tmp_10081[1], tmp_10079[2] + tmp_10081[2]];
    signal tmp_10084[3] <== [tmp_10082[0] + tmp_9626[0], tmp_10082[1] + tmp_9626[1], tmp_10082[2] + tmp_9626[2]];
    signal tmp_10086[3] <== [tmp_10084[0] + tmp_9641[0], tmp_10084[1] + tmp_9641[1], tmp_10084[2] + tmp_9641[2]];
    signal tmp_10088[3] <== [51 * tmp_9657[0], 51 * tmp_9657[1], 51 * tmp_9657[2]];
    signal tmp_10089[3] <== [tmp_10086[0] + tmp_10088[0], tmp_10086[1] + tmp_10088[1], tmp_10086[2] + tmp_10088[2]];
    signal tmp_10091[3] <== [tmp_10089[0] + tmp_9673[0], tmp_10089[1] + tmp_9673[1], tmp_10089[2] + tmp_9673[2]];
    signal tmp_10093[3] <== [11 * tmp_9689[0], 11 * tmp_9689[1], 11 * tmp_9689[2]];
    signal tmp_10094[3] <== [tmp_10091[0] + tmp_10093[0], tmp_10091[1] + tmp_10093[1], tmp_10091[2] + tmp_10093[2]];
    signal tmp_10096[3] <== [17 * tmp_9705[0], 17 * tmp_9705[1], 17 * tmp_9705[2]];
    signal tmp_10097[3] <== [tmp_10094[0] + tmp_10096[0], tmp_10094[1] + tmp_10096[1], tmp_10094[2] + tmp_10096[2]];
    signal tmp_10100[3] <== [tmp_10097[0] + tmp_9848[0], tmp_10097[1] + tmp_9848[1], tmp_10097[2] + tmp_9848[2]];
    signal tmp_10102[3] <== [tmp_10100[0] + tmp_9737[0], tmp_10100[1] + tmp_9737[1], tmp_10100[2] + tmp_9737[2]];
    signal tmp_10104[3] <== [101 * tmp_9753[0], 101 * tmp_9753[1], 101 * tmp_9753[2]];
    signal tmp_10105[3] <== [tmp_10102[0] + tmp_10104[0], tmp_10102[1] + tmp_10104[1], tmp_10102[2] + tmp_10104[2]];
    signal tmp_10106[3] <== [evals[106][0] - tmp_10105[0], evals[106][1] - tmp_10105[1], evals[106][2] - tmp_10105[2]];
    signal tmp_10107[3] <== CMul()(tmp_9504, tmp_10106);
    signal tmp_10108[3] <== [tmp_10059[0] + tmp_10107[0], tmp_10059[1] + tmp_10107[1], tmp_10059[2] + tmp_10107[2]];
    signal tmp_10109[3] <== CMul()(challengeQ, tmp_10108);
    signal tmp_10114[3] <== [101 * tmp_9518[0], 101 * tmp_9518[1], 101 * tmp_9518[2]];
    signal tmp_10116[3] <== [63 * tmp_9532[0], 63 * tmp_9532[1], 63 * tmp_9532[2]];
    signal tmp_10117[3] <== [tmp_10114[0] + tmp_10116[0], tmp_10114[1] + tmp_10116[1], tmp_10114[2] + tmp_10116[2]];
    signal tmp_10119[3] <== [15 * tmp_9547[0], 15 * tmp_9547[1], 15 * tmp_9547[2]];
    signal tmp_10120[3] <== [tmp_10117[0] + tmp_10119[0], tmp_10117[1] + tmp_10119[1], tmp_10117[2] + tmp_10119[2]];
    signal tmp_10122[3] <== [2 * tmp_9563[0], 2 * tmp_9563[1], 2 * tmp_9563[2]];
    signal tmp_10123[3] <== [tmp_10120[0] + tmp_10122[0], tmp_10120[1] + tmp_10122[1], tmp_10120[2] + tmp_10122[2]];
    signal tmp_10125[3] <== [67 * tmp_9578[0], 67 * tmp_9578[1], 67 * tmp_9578[2]];
    signal tmp_10126[3] <== [tmp_10123[0] + tmp_10125[0], tmp_10123[1] + tmp_10125[1], tmp_10123[2] + tmp_10125[2]];
    signal tmp_10128[3] <== [22 * tmp_9594[0], 22 * tmp_9594[1], 22 * tmp_9594[2]];
    signal tmp_10129[3] <== [tmp_10126[0] + tmp_10128[0], tmp_10126[1] + tmp_10128[1], tmp_10126[2] + tmp_10128[2]];
    signal tmp_10131[3] <== [13 * tmp_9610[0], 13 * tmp_9610[1], 13 * tmp_9610[2]];
    signal tmp_10132[3] <== [tmp_10129[0] + tmp_10131[0], tmp_10129[1] + tmp_10131[1], tmp_10129[2] + tmp_10131[2]];
    signal tmp_10134[3] <== [3 * tmp_9626[0], 3 * tmp_9626[1], 3 * tmp_9626[2]];
    signal tmp_10135[3] <== [tmp_10132[0] + tmp_10134[0], tmp_10132[1] + tmp_10134[1], tmp_10132[2] + tmp_10134[2]];
    signal tmp_10137[3] <== [tmp_10135[0] + tmp_9641[0], tmp_10135[1] + tmp_9641[1], tmp_10135[2] + tmp_9641[2]];
    signal tmp_10139[3] <== [tmp_10137[0] + tmp_9657[0], tmp_10137[1] + tmp_9657[1], tmp_10137[2] + tmp_9657[2]];
    signal tmp_10141[3] <== [51 * tmp_9673[0], 51 * tmp_9673[1], 51 * tmp_9673[2]];
    signal tmp_10142[3] <== [tmp_10139[0] + tmp_10141[0], tmp_10139[1] + tmp_10141[1], tmp_10139[2] + tmp_10141[2]];
    signal tmp_10144[3] <== [tmp_10142[0] + tmp_9689[0], tmp_10142[1] + tmp_9689[1], tmp_10142[2] + tmp_9689[2]];
    signal tmp_10146[3] <== [11 * tmp_9705[0], 11 * tmp_9705[1], 11 * tmp_9705[2]];
    signal tmp_10147[3] <== [tmp_10144[0] + tmp_10146[0], tmp_10144[1] + tmp_10146[1], tmp_10144[2] + tmp_10146[2]];
    signal tmp_10149[3] <== [17 * tmp_9721[0], 17 * tmp_9721[1], 17 * tmp_9721[2]];
    signal tmp_10150[3] <== [tmp_10147[0] + tmp_10149[0], tmp_10147[1] + tmp_10149[1], tmp_10147[2] + tmp_10149[2]];
    signal tmp_10153[3] <== [tmp_10150[0] + tmp_9901[0], tmp_10150[1] + tmp_9901[1], tmp_10150[2] + tmp_9901[2]];
    signal tmp_10155[3] <== [tmp_10153[0] + tmp_9753[0], tmp_10153[1] + tmp_9753[1], tmp_10153[2] + tmp_9753[2]];
    signal tmp_10156[3] <== [evals[107][0] - tmp_10155[0], evals[107][1] - tmp_10155[1], evals[107][2] - tmp_10155[2]];
    signal tmp_10157[3] <== CMul()(tmp_9504, tmp_10156);
    signal tmp_10158[3] <== [tmp_10109[0] + tmp_10157[0], tmp_10109[1] + tmp_10157[1], tmp_10109[2] + tmp_10157[2]];
    signal tmp_10159[3] <== CMul()(challengeQ, tmp_10158);
    signal tmp_10165[3] <== [101 * tmp_9532[0], 101 * tmp_9532[1], 101 * tmp_9532[2]];
    signal tmp_10166[3] <== [tmp_9518[0] + tmp_10165[0], tmp_9518[1] + tmp_10165[1], tmp_9518[2] + tmp_10165[2]];
    signal tmp_10168[3] <== [63 * tmp_9547[0], 63 * tmp_9547[1], 63 * tmp_9547[2]];
    signal tmp_10169[3] <== [tmp_10166[0] + tmp_10168[0], tmp_10166[1] + tmp_10168[1], tmp_10166[2] + tmp_10168[2]];
    signal tmp_10171[3] <== [15 * tmp_9563[0], 15 * tmp_9563[1], 15 * tmp_9563[2]];
    signal tmp_10172[3] <== [tmp_10169[0] + tmp_10171[0], tmp_10169[1] + tmp_10171[1], tmp_10169[2] + tmp_10171[2]];
    signal tmp_10174[3] <== [2 * tmp_9578[0], 2 * tmp_9578[1], 2 * tmp_9578[2]];
    signal tmp_10175[3] <== [tmp_10172[0] + tmp_10174[0], tmp_10172[1] + tmp_10174[1], tmp_10172[2] + tmp_10174[2]];
    signal tmp_10177[3] <== [67 * tmp_9594[0], 67 * tmp_9594[1], 67 * tmp_9594[2]];
    signal tmp_10178[3] <== [tmp_10175[0] + tmp_10177[0], tmp_10175[1] + tmp_10177[1], tmp_10175[2] + tmp_10177[2]];
    signal tmp_10180[3] <== [22 * tmp_9610[0], 22 * tmp_9610[1], 22 * tmp_9610[2]];
    signal tmp_10181[3] <== [tmp_10178[0] + tmp_10180[0], tmp_10178[1] + tmp_10180[1], tmp_10178[2] + tmp_10180[2]];
    signal tmp_10183[3] <== [13 * tmp_9626[0], 13 * tmp_9626[1], 13 * tmp_9626[2]];
    signal tmp_10184[3] <== [tmp_10181[0] + tmp_10183[0], tmp_10181[1] + tmp_10183[1], tmp_10181[2] + tmp_10183[2]];
    signal tmp_10186[3] <== [3 * tmp_9641[0], 3 * tmp_9641[1], 3 * tmp_9641[2]];
    signal tmp_10187[3] <== [tmp_10184[0] + tmp_10186[0], tmp_10184[1] + tmp_10186[1], tmp_10184[2] + tmp_10186[2]];
    signal tmp_10189[3] <== [tmp_10187[0] + tmp_9657[0], tmp_10187[1] + tmp_9657[1], tmp_10187[2] + tmp_9657[2]];
    signal tmp_10191[3] <== [tmp_10189[0] + tmp_9673[0], tmp_10189[1] + tmp_9673[1], tmp_10189[2] + tmp_9673[2]];
    signal tmp_10193[3] <== [51 * tmp_9689[0], 51 * tmp_9689[1], 51 * tmp_9689[2]];
    signal tmp_10194[3] <== [tmp_10191[0] + tmp_10193[0], tmp_10191[1] + tmp_10193[1], tmp_10191[2] + tmp_10193[2]];
    signal tmp_10196[3] <== [tmp_10194[0] + tmp_9705[0], tmp_10194[1] + tmp_9705[1], tmp_10194[2] + tmp_9705[2]];
    signal tmp_10198[3] <== [11 * tmp_9721[0], 11 * tmp_9721[1], 11 * tmp_9721[2]];
    signal tmp_10199[3] <== [tmp_10196[0] + tmp_10198[0], tmp_10196[1] + tmp_10198[1], tmp_10196[2] + tmp_10198[2]];
    signal tmp_10201[3] <== [17 * tmp_9737[0], 17 * tmp_9737[1], 17 * tmp_9737[2]];
    signal tmp_10202[3] <== [tmp_10199[0] + tmp_10201[0], tmp_10199[1] + tmp_10201[1], tmp_10199[2] + tmp_10201[2]];
    signal tmp_10205[3] <== [tmp_10202[0] + tmp_9954[0], tmp_10202[1] + tmp_9954[1], tmp_10202[2] + tmp_9954[2]];
    signal tmp_10206[3] <== [evals[108][0] - tmp_10205[0], evals[108][1] - tmp_10205[1], evals[108][2] - tmp_10205[2]];
    signal tmp_10207[3] <== CMul()(tmp_9504, tmp_10206);
    signal tmp_10208[3] <== [tmp_10159[0] + tmp_10207[0], tmp_10159[1] + tmp_10207[1], tmp_10159[2] + tmp_10207[2]];
    signal tmp_10209[3] <== CMul()(challengeQ, tmp_10208);
    signal tmp_10216[3] <== [tmp_9964[0] + tmp_9532[0], tmp_9964[1] + tmp_9532[1], tmp_9964[2] + tmp_9532[2]];
    signal tmp_10218[3] <== [101 * tmp_9547[0], 101 * tmp_9547[1], 101 * tmp_9547[2]];
    signal tmp_10219[3] <== [tmp_10216[0] + tmp_10218[0], tmp_10216[1] + tmp_10218[1], tmp_10216[2] + tmp_10218[2]];
    signal tmp_10221[3] <== [63 * tmp_9563[0], 63 * tmp_9563[1], 63 * tmp_9563[2]];
    signal tmp_10222[3] <== [tmp_10219[0] + tmp_10221[0], tmp_10219[1] + tmp_10221[1], tmp_10219[2] + tmp_10221[2]];
    signal tmp_10224[3] <== [15 * tmp_9578[0], 15 * tmp_9578[1], 15 * tmp_9578[2]];
    signal tmp_10225[3] <== [tmp_10222[0] + tmp_10224[0], tmp_10222[1] + tmp_10224[1], tmp_10222[2] + tmp_10224[2]];
    signal tmp_10227[3] <== [2 * tmp_9594[0], 2 * tmp_9594[1], 2 * tmp_9594[2]];
    signal tmp_10228[3] <== [tmp_10225[0] + tmp_10227[0], tmp_10225[1] + tmp_10227[1], tmp_10225[2] + tmp_10227[2]];
    signal tmp_10230[3] <== [67 * tmp_9610[0], 67 * tmp_9610[1], 67 * tmp_9610[2]];
    signal tmp_10231[3] <== [tmp_10228[0] + tmp_10230[0], tmp_10228[1] + tmp_10230[1], tmp_10228[2] + tmp_10230[2]];
    signal tmp_10233[3] <== [22 * tmp_9626[0], 22 * tmp_9626[1], 22 * tmp_9626[2]];
    signal tmp_10234[3] <== [tmp_10231[0] + tmp_10233[0], tmp_10231[1] + tmp_10233[1], tmp_10231[2] + tmp_10233[2]];
    signal tmp_10236[3] <== [13 * tmp_9641[0], 13 * tmp_9641[1], 13 * tmp_9641[2]];
    signal tmp_10237[3] <== [tmp_10234[0] + tmp_10236[0], tmp_10234[1] + tmp_10236[1], tmp_10234[2] + tmp_10236[2]];
    signal tmp_10239[3] <== [3 * tmp_9657[0], 3 * tmp_9657[1], 3 * tmp_9657[2]];
    signal tmp_10240[3] <== [tmp_10237[0] + tmp_10239[0], tmp_10237[1] + tmp_10239[1], tmp_10237[2] + tmp_10239[2]];
    signal tmp_10242[3] <== [tmp_10240[0] + tmp_9673[0], tmp_10240[1] + tmp_9673[1], tmp_10240[2] + tmp_9673[2]];
    signal tmp_10244[3] <== [tmp_10242[0] + tmp_9689[0], tmp_10242[1] + tmp_9689[1], tmp_10242[2] + tmp_9689[2]];
    signal tmp_10246[3] <== [51 * tmp_9705[0], 51 * tmp_9705[1], 51 * tmp_9705[2]];
    signal tmp_10247[3] <== [tmp_10244[0] + tmp_10246[0], tmp_10244[1] + tmp_10246[1], tmp_10244[2] + tmp_10246[2]];
    signal tmp_10249[3] <== [tmp_10247[0] + tmp_9721[0], tmp_10247[1] + tmp_9721[1], tmp_10247[2] + tmp_9721[2]];
    signal tmp_10251[3] <== [11 * tmp_9737[0], 11 * tmp_9737[1], 11 * tmp_9737[2]];
    signal tmp_10252[3] <== [tmp_10249[0] + tmp_10251[0], tmp_10249[1] + tmp_10251[1], tmp_10249[2] + tmp_10251[2]];
    signal tmp_10254[3] <== [17 * tmp_9753[0], 17 * tmp_9753[1], 17 * tmp_9753[2]];
    signal tmp_10255[3] <== [tmp_10252[0] + tmp_10254[0], tmp_10252[1] + tmp_10254[1], tmp_10252[2] + tmp_10254[2]];
    signal tmp_10256[3] <== [evals[109][0] - tmp_10255[0], evals[109][1] - tmp_10255[1], evals[109][2] - tmp_10255[2]];
    signal tmp_10257[3] <== CMul()(tmp_9504, tmp_10256);
    signal tmp_10258[3] <== [tmp_10209[0] + tmp_10257[0], tmp_10209[1] + tmp_10257[1], tmp_10209[2] + tmp_10257[2]];
    signal tmp_10259[3] <== CMul()(challengeQ, tmp_10258);
    signal tmp_10264[3] <== [17 * tmp_9518[0], 17 * tmp_9518[1], 17 * tmp_9518[2]];
    signal tmp_10267[3] <== [tmp_10264[0] + tmp_10016[0], tmp_10264[1] + tmp_10016[1], tmp_10264[2] + tmp_10016[2]];
    signal tmp_10269[3] <== [tmp_10267[0] + tmp_9547[0], tmp_10267[1] + tmp_9547[1], tmp_10267[2] + tmp_9547[2]];
    signal tmp_10271[3] <== [101 * tmp_9563[0], 101 * tmp_9563[1], 101 * tmp_9563[2]];
    signal tmp_10272[3] <== [tmp_10269[0] + tmp_10271[0], tmp_10269[1] + tmp_10271[1], tmp_10269[2] + tmp_10271[2]];
    signal tmp_10274[3] <== [63 * tmp_9578[0], 63 * tmp_9578[1], 63 * tmp_9578[2]];
    signal tmp_10275[3] <== [tmp_10272[0] + tmp_10274[0], tmp_10272[1] + tmp_10274[1], tmp_10272[2] + tmp_10274[2]];
    signal tmp_10277[3] <== [15 * tmp_9594[0], 15 * tmp_9594[1], 15 * tmp_9594[2]];
    signal tmp_10278[3] <== [tmp_10275[0] + tmp_10277[0], tmp_10275[1] + tmp_10277[1], tmp_10275[2] + tmp_10277[2]];
    signal tmp_10281[3] <== [tmp_10278[0] + tmp_9611[0], tmp_10278[1] + tmp_9611[1], tmp_10278[2] + tmp_9611[2]];
    signal tmp_10283[3] <== [67 * tmp_9626[0], 67 * tmp_9626[1], 67 * tmp_9626[2]];
    signal tmp_10284[3] <== [tmp_10281[0] + tmp_10283[0], tmp_10281[1] + tmp_10283[1], tmp_10281[2] + tmp_10283[2]];
    signal tmp_10286[3] <== [22 * tmp_9641[0], 22 * tmp_9641[1], 22 * tmp_9641[2]];
    signal tmp_10287[3] <== [tmp_10284[0] + tmp_10286[0], tmp_10284[1] + tmp_10286[1], tmp_10284[2] + tmp_10286[2]];
    signal tmp_10289[3] <== [13 * tmp_9657[0], 13 * tmp_9657[1], 13 * tmp_9657[2]];
    signal tmp_10290[3] <== [tmp_10287[0] + tmp_10289[0], tmp_10287[1] + tmp_10289[1], tmp_10287[2] + tmp_10289[2]];
    signal tmp_10292[3] <== [3 * tmp_9673[0], 3 * tmp_9673[1], 3 * tmp_9673[2]];
    signal tmp_10293[3] <== [tmp_10290[0] + tmp_10292[0], tmp_10290[1] + tmp_10292[1], tmp_10290[2] + tmp_10292[2]];
    signal tmp_10295[3] <== [tmp_10293[0] + tmp_9689[0], tmp_10293[1] + tmp_9689[1], tmp_10293[2] + tmp_9689[2]];
    signal tmp_10297[3] <== [tmp_10295[0] + tmp_9705[0], tmp_10295[1] + tmp_9705[1], tmp_10295[2] + tmp_9705[2]];
    signal tmp_10299[3] <== [51 * tmp_9721[0], 51 * tmp_9721[1], 51 * tmp_9721[2]];
    signal tmp_10300[3] <== [tmp_10297[0] + tmp_10299[0], tmp_10297[1] + tmp_10299[1], tmp_10297[2] + tmp_10299[2]];
    signal tmp_10302[3] <== [tmp_10300[0] + tmp_9737[0], tmp_10300[1] + tmp_9737[1], tmp_10300[2] + tmp_9737[2]];
    signal tmp_10304[3] <== [11 * tmp_9753[0], 11 * tmp_9753[1], 11 * tmp_9753[2]];
    signal tmp_10305[3] <== [tmp_10302[0] + tmp_10304[0], tmp_10302[1] + tmp_10304[1], tmp_10302[2] + tmp_10304[2]];
    signal tmp_10306[3] <== [evals[110][0] - tmp_10305[0], evals[110][1] - tmp_10305[1], evals[110][2] - tmp_10305[2]];
    signal tmp_10307[3] <== CMul()(tmp_9504, tmp_10306);
    signal tmp_10308[3] <== [tmp_10259[0] + tmp_10307[0], tmp_10259[1] + tmp_10307[1], tmp_10259[2] + tmp_10307[2]];
    signal tmp_10309[3] <== CMul()(challengeQ, tmp_10308);
    signal tmp_10314[3] <== [11 * tmp_9518[0], 11 * tmp_9518[1], 11 * tmp_9518[2]];
    signal tmp_10316[3] <== [17 * tmp_9532[0], 17 * tmp_9532[1], 17 * tmp_9532[2]];
    signal tmp_10317[3] <== [tmp_10314[0] + tmp_10316[0], tmp_10314[1] + tmp_10316[1], tmp_10314[2] + tmp_10316[2]];
    signal tmp_10320[3] <== [tmp_10317[0] + tmp_10069[0], tmp_10317[1] + tmp_10069[1], tmp_10317[2] + tmp_10069[2]];
    signal tmp_10322[3] <== [tmp_10320[0] + tmp_9563[0], tmp_10320[1] + tmp_9563[1], tmp_10320[2] + tmp_9563[2]];
    signal tmp_10324[3] <== [101 * tmp_9578[0], 101 * tmp_9578[1], 101 * tmp_9578[2]];
    signal tmp_10325[3] <== [tmp_10322[0] + tmp_10324[0], tmp_10322[1] + tmp_10324[1], tmp_10322[2] + tmp_10324[2]];
    signal tmp_10327[3] <== [63 * tmp_9594[0], 63 * tmp_9594[1], 63 * tmp_9594[2]];
    signal tmp_10328[3] <== [tmp_10325[0] + tmp_10327[0], tmp_10325[1] + tmp_10327[1], tmp_10325[2] + tmp_10327[2]];
    signal tmp_10330[3] <== [15 * tmp_9610[0], 15 * tmp_9610[1], 15 * tmp_9610[2]];
    signal tmp_10331[3] <== [tmp_10328[0] + tmp_10330[0], tmp_10328[1] + tmp_10330[1], tmp_10328[2] + tmp_10330[2]];
    signal tmp_10334[3] <== [tmp_10331[0] + tmp_9781[0], tmp_10331[1] + tmp_9781[1], tmp_10331[2] + tmp_9781[2]];
    signal tmp_10336[3] <== [67 * tmp_9641[0], 67 * tmp_9641[1], 67 * tmp_9641[2]];
    signal tmp_10337[3] <== [tmp_10334[0] + tmp_10336[0], tmp_10334[1] + tmp_10336[1], tmp_10334[2] + tmp_10336[2]];
    signal tmp_10339[3] <== [22 * tmp_9657[0], 22 * tmp_9657[1], 22 * tmp_9657[2]];
    signal tmp_10340[3] <== [tmp_10337[0] + tmp_10339[0], tmp_10337[1] + tmp_10339[1], tmp_10337[2] + tmp_10339[2]];
    signal tmp_10342[3] <== [13 * tmp_9673[0], 13 * tmp_9673[1], 13 * tmp_9673[2]];
    signal tmp_10343[3] <== [tmp_10340[0] + tmp_10342[0], tmp_10340[1] + tmp_10342[1], tmp_10340[2] + tmp_10342[2]];
    signal tmp_10345[3] <== [3 * tmp_9689[0], 3 * tmp_9689[1], 3 * tmp_9689[2]];
    signal tmp_10346[3] <== [tmp_10343[0] + tmp_10345[0], tmp_10343[1] + tmp_10345[1], tmp_10343[2] + tmp_10345[2]];
    signal tmp_10348[3] <== [tmp_10346[0] + tmp_9705[0], tmp_10346[1] + tmp_9705[1], tmp_10346[2] + tmp_9705[2]];
    signal tmp_10350[3] <== [tmp_10348[0] + tmp_9721[0], tmp_10348[1] + tmp_9721[1], tmp_10348[2] + tmp_9721[2]];
    signal tmp_10352[3] <== [51 * tmp_9737[0], 51 * tmp_9737[1], 51 * tmp_9737[2]];
    signal tmp_10353[3] <== [tmp_10350[0] + tmp_10352[0], tmp_10350[1] + tmp_10352[1], tmp_10350[2] + tmp_10352[2]];
    signal tmp_10355[3] <== [tmp_10353[0] + tmp_9753[0], tmp_10353[1] + tmp_9753[1], tmp_10353[2] + tmp_9753[2]];
    signal tmp_10356[3] <== [evals[111][0] - tmp_10355[0], evals[111][1] - tmp_10355[1], evals[111][2] - tmp_10355[2]];
    signal tmp_10357[3] <== CMul()(tmp_9504, tmp_10356);
    signal tmp_10358[3] <== [tmp_10309[0] + tmp_10357[0], tmp_10309[1] + tmp_10357[1], tmp_10309[2] + tmp_10357[2]];
    signal tmp_10359[3] <== CMul()(challengeQ, tmp_10358);
    signal tmp_10365[3] <== [11 * tmp_9532[0], 11 * tmp_9532[1], 11 * tmp_9532[2]];
    signal tmp_10366[3] <== [tmp_9518[0] + tmp_10365[0], tmp_9518[1] + tmp_10365[1], tmp_9518[2] + tmp_10365[2]];
    signal tmp_10368[3] <== [17 * tmp_9547[0], 17 * tmp_9547[1], 17 * tmp_9547[2]];
    signal tmp_10369[3] <== [tmp_10366[0] + tmp_10368[0], tmp_10366[1] + tmp_10368[1], tmp_10366[2] + tmp_10368[2]];
    signal tmp_10372[3] <== [tmp_10369[0] + tmp_10122[0], tmp_10369[1] + tmp_10122[1], tmp_10369[2] + tmp_10122[2]];
    signal tmp_10374[3] <== [tmp_10372[0] + tmp_9578[0], tmp_10372[1] + tmp_9578[1], tmp_10372[2] + tmp_9578[2]];
    signal tmp_10376[3] <== [101 * tmp_9594[0], 101 * tmp_9594[1], 101 * tmp_9594[2]];
    signal tmp_10377[3] <== [tmp_10374[0] + tmp_10376[0], tmp_10374[1] + tmp_10376[1], tmp_10374[2] + tmp_10376[2]];
    signal tmp_10379[3] <== [63 * tmp_9610[0], 63 * tmp_9610[1], 63 * tmp_9610[2]];
    signal tmp_10380[3] <== [tmp_10377[0] + tmp_10379[0], tmp_10377[1] + tmp_10379[1], tmp_10377[2] + tmp_10379[2]];
    signal tmp_10382[3] <== [15 * tmp_9626[0], 15 * tmp_9626[1], 15 * tmp_9626[2]];
    signal tmp_10383[3] <== [tmp_10380[0] + tmp_10382[0], tmp_10380[1] + tmp_10382[1], tmp_10380[2] + tmp_10382[2]];
    signal tmp_10386[3] <== [tmp_10383[0] + tmp_9834[0], tmp_10383[1] + tmp_9834[1], tmp_10383[2] + tmp_9834[2]];
    signal tmp_10388[3] <== [67 * tmp_9657[0], 67 * tmp_9657[1], 67 * tmp_9657[2]];
    signal tmp_10389[3] <== [tmp_10386[0] + tmp_10388[0], tmp_10386[1] + tmp_10388[1], tmp_10386[2] + tmp_10388[2]];
    signal tmp_10391[3] <== [22 * tmp_9673[0], 22 * tmp_9673[1], 22 * tmp_9673[2]];
    signal tmp_10392[3] <== [tmp_10389[0] + tmp_10391[0], tmp_10389[1] + tmp_10391[1], tmp_10389[2] + tmp_10391[2]];
    signal tmp_10394[3] <== [13 * tmp_9689[0], 13 * tmp_9689[1], 13 * tmp_9689[2]];
    signal tmp_10395[3] <== [tmp_10392[0] + tmp_10394[0], tmp_10392[1] + tmp_10394[1], tmp_10392[2] + tmp_10394[2]];
    signal tmp_10397[3] <== [3 * tmp_9705[0], 3 * tmp_9705[1], 3 * tmp_9705[2]];
    signal tmp_10398[3] <== [tmp_10395[0] + tmp_10397[0], tmp_10395[1] + tmp_10397[1], tmp_10395[2] + tmp_10397[2]];
    signal tmp_10400[3] <== [tmp_10398[0] + tmp_9721[0], tmp_10398[1] + tmp_9721[1], tmp_10398[2] + tmp_9721[2]];
    signal tmp_10402[3] <== [tmp_10400[0] + tmp_9737[0], tmp_10400[1] + tmp_9737[1], tmp_10400[2] + tmp_9737[2]];
    signal tmp_10404[3] <== [51 * tmp_9753[0], 51 * tmp_9753[1], 51 * tmp_9753[2]];
    signal tmp_10405[3] <== [tmp_10402[0] + tmp_10404[0], tmp_10402[1] + tmp_10404[1], tmp_10402[2] + tmp_10404[2]];
    signal tmp_10406[3] <== [evals[112][0] - tmp_10405[0], evals[112][1] - tmp_10405[1], evals[112][2] - tmp_10405[2]];
    signal tmp_10407[3] <== CMul()(tmp_9504, tmp_10406);
    signal tmp_10408[3] <== [tmp_10359[0] + tmp_10407[0], tmp_10359[1] + tmp_10407[1], tmp_10359[2] + tmp_10407[2]];
    signal tmp_10409[3] <== CMul()(challengeQ, tmp_10408);
    signal tmp_10414[3] <== [51 * tmp_9518[0], 51 * tmp_9518[1], 51 * tmp_9518[2]];
    signal tmp_10416[3] <== [tmp_10414[0] + tmp_9532[0], tmp_10414[1] + tmp_9532[1], tmp_10414[2] + tmp_9532[2]];
    signal tmp_10418[3] <== [11 * tmp_9547[0], 11 * tmp_9547[1], 11 * tmp_9547[2]];
    signal tmp_10419[3] <== [tmp_10416[0] + tmp_10418[0], tmp_10416[1] + tmp_10418[1], tmp_10416[2] + tmp_10418[2]];
    signal tmp_10421[3] <== [17 * tmp_9563[0], 17 * tmp_9563[1], 17 * tmp_9563[2]];
    signal tmp_10422[3] <== [tmp_10419[0] + tmp_10421[0], tmp_10419[1] + tmp_10421[1], tmp_10419[2] + tmp_10421[2]];
    signal tmp_10425[3] <== [tmp_10422[0] + tmp_10174[0], tmp_10422[1] + tmp_10174[1], tmp_10422[2] + tmp_10174[2]];
    signal tmp_10427[3] <== [tmp_10425[0] + tmp_9594[0], tmp_10425[1] + tmp_9594[1], tmp_10425[2] + tmp_9594[2]];
    signal tmp_10429[3] <== [101 * tmp_9610[0], 101 * tmp_9610[1], 101 * tmp_9610[2]];
    signal tmp_10430[3] <== [tmp_10427[0] + tmp_10429[0], tmp_10427[1] + tmp_10429[1], tmp_10427[2] + tmp_10429[2]];
    signal tmp_10432[3] <== [63 * tmp_9626[0], 63 * tmp_9626[1], 63 * tmp_9626[2]];
    signal tmp_10433[3] <== [tmp_10430[0] + tmp_10432[0], tmp_10430[1] + tmp_10432[1], tmp_10430[2] + tmp_10432[2]];
    signal tmp_10435[3] <== [15 * tmp_9641[0], 15 * tmp_9641[1], 15 * tmp_9641[2]];
    signal tmp_10436[3] <== [tmp_10433[0] + tmp_10435[0], tmp_10433[1] + tmp_10435[1], tmp_10433[2] + tmp_10435[2]];
    signal tmp_10439[3] <== [tmp_10436[0] + tmp_9887[0], tmp_10436[1] + tmp_9887[1], tmp_10436[2] + tmp_9887[2]];
    signal tmp_10441[3] <== [67 * tmp_9673[0], 67 * tmp_9673[1], 67 * tmp_9673[2]];
    signal tmp_10442[3] <== [tmp_10439[0] + tmp_10441[0], tmp_10439[1] + tmp_10441[1], tmp_10439[2] + tmp_10441[2]];
    signal tmp_10444[3] <== [22 * tmp_9689[0], 22 * tmp_9689[1], 22 * tmp_9689[2]];
    signal tmp_10445[3] <== [tmp_10442[0] + tmp_10444[0], tmp_10442[1] + tmp_10444[1], tmp_10442[2] + tmp_10444[2]];
    signal tmp_10447[3] <== [13 * tmp_9705[0], 13 * tmp_9705[1], 13 * tmp_9705[2]];
    signal tmp_10448[3] <== [tmp_10445[0] + tmp_10447[0], tmp_10445[1] + tmp_10447[1], tmp_10445[2] + tmp_10447[2]];
    signal tmp_10450[3] <== [3 * tmp_9721[0], 3 * tmp_9721[1], 3 * tmp_9721[2]];
    signal tmp_10451[3] <== [tmp_10448[0] + tmp_10450[0], tmp_10448[1] + tmp_10450[1], tmp_10448[2] + tmp_10450[2]];
    signal tmp_10453[3] <== [tmp_10451[0] + tmp_9737[0], tmp_10451[1] + tmp_9737[1], tmp_10451[2] + tmp_9737[2]];
    signal tmp_10455[3] <== [tmp_10453[0] + tmp_9753[0], tmp_10453[1] + tmp_9753[1], tmp_10453[2] + tmp_9753[2]];
    signal tmp_10456[3] <== [evals[113][0] - tmp_10455[0], evals[113][1] - tmp_10455[1], evals[113][2] - tmp_10455[2]];
    signal tmp_10457[3] <== CMul()(tmp_9504, tmp_10456);
    signal tmp_10458[3] <== [tmp_10409[0] + tmp_10457[0], tmp_10409[1] + tmp_10457[1], tmp_10409[2] + tmp_10457[2]];
    signal tmp_10459[3] <== CMul()(challengeQ, tmp_10458);
    signal tmp_10465[3] <== [51 * tmp_9532[0], 51 * tmp_9532[1], 51 * tmp_9532[2]];
    signal tmp_10466[3] <== [tmp_9518[0] + tmp_10465[0], tmp_9518[1] + tmp_10465[1], tmp_9518[2] + tmp_10465[2]];
    signal tmp_10468[3] <== [tmp_10466[0] + tmp_9547[0], tmp_10466[1] + tmp_9547[1], tmp_10466[2] + tmp_9547[2]];
    signal tmp_10470[3] <== [11 * tmp_9563[0], 11 * tmp_9563[1], 11 * tmp_9563[2]];
    signal tmp_10471[3] <== [tmp_10468[0] + tmp_10470[0], tmp_10468[1] + tmp_10470[1], tmp_10468[2] + tmp_10470[2]];
    signal tmp_10473[3] <== [17 * tmp_9578[0], 17 * tmp_9578[1], 17 * tmp_9578[2]];
    signal tmp_10474[3] <== [tmp_10471[0] + tmp_10473[0], tmp_10471[1] + tmp_10473[1], tmp_10471[2] + tmp_10473[2]];
    signal tmp_10477[3] <== [tmp_10474[0] + tmp_10227[0], tmp_10474[1] + tmp_10227[1], tmp_10474[2] + tmp_10227[2]];
    signal tmp_10479[3] <== [tmp_10477[0] + tmp_9610[0], tmp_10477[1] + tmp_9610[1], tmp_10477[2] + tmp_9610[2]];
    signal tmp_10481[3] <== [101 * tmp_9626[0], 101 * tmp_9626[1], 101 * tmp_9626[2]];
    signal tmp_10482[3] <== [tmp_10479[0] + tmp_10481[0], tmp_10479[1] + tmp_10481[1], tmp_10479[2] + tmp_10481[2]];
    signal tmp_10484[3] <== [63 * tmp_9641[0], 63 * tmp_9641[1], 63 * tmp_9641[2]];
    signal tmp_10485[3] <== [tmp_10482[0] + tmp_10484[0], tmp_10482[1] + tmp_10484[1], tmp_10482[2] + tmp_10484[2]];
    signal tmp_10487[3] <== [15 * tmp_9657[0], 15 * tmp_9657[1], 15 * tmp_9657[2]];
    signal tmp_10488[3] <== [tmp_10485[0] + tmp_10487[0], tmp_10485[1] + tmp_10487[1], tmp_10485[2] + tmp_10487[2]];
    signal tmp_10491[3] <== [tmp_10488[0] + tmp_9940[0], tmp_10488[1] + tmp_9940[1], tmp_10488[2] + tmp_9940[2]];
    signal tmp_10493[3] <== [67 * tmp_9689[0], 67 * tmp_9689[1], 67 * tmp_9689[2]];
    signal tmp_10494[3] <== [tmp_10491[0] + tmp_10493[0], tmp_10491[1] + tmp_10493[1], tmp_10491[2] + tmp_10493[2]];
    signal tmp_10496[3] <== [22 * tmp_9705[0], 22 * tmp_9705[1], 22 * tmp_9705[2]];
    signal tmp_10497[3] <== [tmp_10494[0] + tmp_10496[0], tmp_10494[1] + tmp_10496[1], tmp_10494[2] + tmp_10496[2]];
    signal tmp_10499[3] <== [13 * tmp_9721[0], 13 * tmp_9721[1], 13 * tmp_9721[2]];
    signal tmp_10500[3] <== [tmp_10497[0] + tmp_10499[0], tmp_10497[1] + tmp_10499[1], tmp_10497[2] + tmp_10499[2]];
    signal tmp_10502[3] <== [3 * tmp_9737[0], 3 * tmp_9737[1], 3 * tmp_9737[2]];
    signal tmp_10503[3] <== [tmp_10500[0] + tmp_10502[0], tmp_10500[1] + tmp_10502[1], tmp_10500[2] + tmp_10502[2]];
    signal tmp_10505[3] <== [tmp_10503[0] + tmp_9753[0], tmp_10503[1] + tmp_9753[1], tmp_10503[2] + tmp_9753[2]];
    signal tmp_10506[3] <== [evals[114][0] - tmp_10505[0], evals[114][1] - tmp_10505[1], evals[114][2] - tmp_10505[2]];
    signal tmp_10507[3] <== CMul()(tmp_9504, tmp_10506);
    signal tmp_10508[3] <== [tmp_10459[0] + tmp_10507[0], tmp_10459[1] + tmp_10507[1], tmp_10459[2] + tmp_10507[2]];
    signal tmp_10509[3] <== CMul()(challengeQ, tmp_10508);
    signal tmp_10510[3] <== [tmp_9501[0] + evals[126][0], tmp_9501[1] + evals[126][1], tmp_9501[2] + evals[126][2]];
    tmp_10511 <== [tmp_10510[0] + evals[57][0], tmp_10510[1] + evals[57][1], tmp_10510[2] + evals[57][2]];
    signal tmp_10512[3] <== CMul()(evals[57], evals[67]);
    tmp_10513 <== [1 - evals[57][0], -evals[57][1], -evals[57][2]];
    signal tmp_10514[3] <== CMul()(tmp_10513, evals[135]);
    signal tmp_10515[3] <== [tmp_10512[0] + tmp_10514[0], tmp_10512[1] + tmp_10514[1], tmp_10512[2] + tmp_10514[2]];
    signal tmp_10516[3] <== CMul()(evals[99], evals[99]);
    signal tmp_10517[3] <== CMul()(tmp_10516, tmp_10516);
    signal tmp_10518[3] <== CMul()(tmp_10517, tmp_10516);
    tmp_10519 <== CMul()(tmp_10518, evals[99]);
    signal tmp_10520[3] <== [tmp_9501[0] * 9372203521948909212, tmp_9501[1] * 9372203521948909212, tmp_9501[2] * 9372203521948909212];
    signal tmp_10521[3] <== [tmp_9161[0] * 11584860397836334902, tmp_9161[1] * 11584860397836334902, tmp_9161[2] * 11584860397836334902];
    signal tmp_10522[3] <== [tmp_10520[0] + tmp_10521[0], tmp_10520[1] + tmp_10521[1], tmp_10520[2] + tmp_10521[2]];
    signal tmp_10523[3] <== [evals[126][0] * 8298154099590067129, evals[126][1] * 8298154099590067129, evals[126][2] * 8298154099590067129];
    signal tmp_10524[3] <== [tmp_10522[0] + tmp_10523[0], tmp_10522[1] + tmp_10523[1], tmp_10522[2] + tmp_10523[2]];
    tmp_10525 <== [tmp_10519[0] + tmp_10524[0], tmp_10519[1] + tmp_10524[1], tmp_10519[2] + tmp_10524[2]];
    signal tmp_10526[3] <== CMul()(evals[100], evals[100]);
    signal tmp_10527[3] <== CMul()(tmp_10526, tmp_10526);
    signal tmp_10528[3] <== CMul()(tmp_10527, tmp_10526);
    tmp_10529 <== CMul()(tmp_10528, evals[100]);
    signal tmp_10530[3] <== [tmp_9501[0] * 13735366479310483840, tmp_9501[1] * 13735366479310483840, tmp_9501[2] * 13735366479310483840];
    signal tmp_10531[3] <== [tmp_9161[0] * 3020861528779719921, tmp_9161[1] * 3020861528779719921, tmp_9161[2] * 3020861528779719921];
    signal tmp_10532[3] <== [tmp_10530[0] + tmp_10531[0], tmp_10530[1] + tmp_10531[1], tmp_10530[2] + tmp_10531[2]];
    signal tmp_10533[3] <== [evals[126][0] * 10708178624497052143, evals[126][1] * 10708178624497052143, evals[126][2] * 10708178624497052143];
    signal tmp_10534[3] <== [tmp_10532[0] + tmp_10533[0], tmp_10532[1] + tmp_10533[1], tmp_10532[2] + tmp_10533[2]];
    tmp_10535 <== [tmp_10529[0] + tmp_10534[0], tmp_10529[1] + tmp_10534[1], tmp_10529[2] + tmp_10534[2]];
    signal tmp_10536[3] <== [tmp_10525[0] + tmp_10535[0], tmp_10525[1] + tmp_10535[1], tmp_10525[2] + tmp_10535[2]];
    signal tmp_10537[3] <== CMul()(evals[101], evals[101]);
    signal tmp_10538[3] <== CMul()(tmp_10537, tmp_10537);
    signal tmp_10539[3] <== CMul()(tmp_10538, tmp_10537);
    tmp_10540 <== CMul()(tmp_10539, evals[101]);
    signal tmp_10541[3] <== [tmp_9501[0] * 8803886850367906518, tmp_9501[1] * 8803886850367906518, tmp_9501[2] * 8803886850367906518];
    signal tmp_10542[3] <== [tmp_9161[0] * 8049325709262789663, tmp_9161[1] * 8049325709262789663, tmp_9161[2] * 8049325709262789663];
    signal tmp_10543[3] <== [tmp_10541[0] + tmp_10542[0], tmp_10541[1] + tmp_10542[1], tmp_10541[2] + tmp_10542[2]];
    signal tmp_10544[3] <== [evals[126][0] * 7005743732139600032, evals[126][1] * 7005743732139600032, evals[126][2] * 7005743732139600032];
    signal tmp_10545[3] <== [tmp_10543[0] + tmp_10544[0], tmp_10543[1] + tmp_10544[1], tmp_10543[2] + tmp_10544[2]];
    tmp_10546 <== [tmp_10540[0] + tmp_10545[0], tmp_10540[1] + tmp_10545[1], tmp_10540[2] + tmp_10545[2]];
    signal tmp_10547[3] <== [51 * tmp_10546[0], 51 * tmp_10546[1], 51 * tmp_10546[2]];
    signal tmp_10548[3] <== [tmp_10536[0] + tmp_10547[0], tmp_10536[1] + tmp_10547[1], tmp_10536[2] + tmp_10547[2]];
    signal tmp_10549[3] <== CMul()(evals[102], evals[102]);
    signal tmp_10550[3] <== CMul()(tmp_10549, tmp_10549);
    signal tmp_10551[3] <== CMul()(tmp_10550, tmp_10549);
    tmp_10552 <== CMul()(tmp_10551, evals[102]);
    signal tmp_10553[3] <== [tmp_9501[0] * 15518531785207090320, tmp_9501[1] * 15518531785207090320, tmp_9501[2] * 15518531785207090320];
    signal tmp_10554[3] <== [tmp_9161[0] * 9276251546345170559, tmp_9161[1] * 9276251546345170559, tmp_9161[2] * 9276251546345170559];
    signal tmp_10555[3] <== [tmp_10553[0] + tmp_10554[0], tmp_10553[1] + tmp_10554[1], tmp_10553[2] + tmp_10554[2]];
    signal tmp_10556[3] <== [evals[126][0] * 13988198031851058492, evals[126][1] * 13988198031851058492, evals[126][2] * 13988198031851058492];
    signal tmp_10557[3] <== [tmp_10555[0] + tmp_10556[0], tmp_10555[1] + tmp_10556[1], tmp_10555[2] + tmp_10556[2]];
    tmp_10558 <== [tmp_10552[0] + tmp_10557[0], tmp_10552[1] + tmp_10557[1], tmp_10552[2] + tmp_10557[2]];
    signal tmp_10559[3] <== [tmp_10548[0] + tmp_10558[0], tmp_10548[1] + tmp_10558[1], tmp_10548[2] + tmp_10558[2]];
    signal tmp_10560[3] <== CMul()(evals[103], evals[103]);
    signal tmp_10561[3] <== CMul()(tmp_10560, tmp_10560);
    signal tmp_10562[3] <== CMul()(tmp_10561, tmp_10560);
    tmp_10563 <== CMul()(tmp_10562, evals[103]);
    signal tmp_10564[3] <== [tmp_9501[0] * 11602150072024304307, tmp_9501[1] * 11602150072024304307, tmp_9501[2] * 11602150072024304307];
    signal tmp_10565[3] <== [tmp_9161[0] * 14380888688809393725, tmp_9161[1] * 14380888688809393725, tmp_9161[2] * 14380888688809393725];
    signal tmp_10566[3] <== [tmp_10564[0] + tmp_10565[0], tmp_10564[1] + tmp_10565[1], tmp_10564[2] + tmp_10565[2]];
    signal tmp_10567[3] <== [evals[126][0] * 10268836078569396437, evals[126][1] * 10268836078569396437, evals[126][2] * 10268836078569396437];
    signal tmp_10568[3] <== [tmp_10566[0] + tmp_10567[0], tmp_10566[1] + tmp_10567[1], tmp_10566[2] + tmp_10567[2]];
    tmp_10569 <== [tmp_10563[0] + tmp_10568[0], tmp_10563[1] + tmp_10568[1], tmp_10563[2] + tmp_10568[2]];
    signal tmp_10570[3] <== [11 * tmp_10569[0], 11 * tmp_10569[1], 11 * tmp_10569[2]];
    signal tmp_10571[3] <== [tmp_10559[0] + tmp_10570[0], tmp_10559[1] + tmp_10570[1], tmp_10559[2] + tmp_10570[2]];
    signal tmp_10572[3] <== CMul()(evals[104], evals[104]);
    signal tmp_10573[3] <== CMul()(tmp_10572, tmp_10572);
    signal tmp_10574[3] <== CMul()(tmp_10573, tmp_10572);
    tmp_10575 <== CMul()(tmp_10574, evals[104]);
    signal tmp_10576[3] <== [tmp_9501[0] * 2912873838477326782, tmp_9501[1] * 2912873838477326782, tmp_9501[2] * 2912873838477326782];
    signal tmp_10577[3] <== [tmp_9161[0] * 3586048355069849144, tmp_9161[1] * 3586048355069849144, tmp_9161[2] * 3586048355069849144];
    signal tmp_10578[3] <== [tmp_10576[0] + tmp_10577[0], tmp_10576[1] + tmp_10577[1], tmp_10576[2] + tmp_10577[2]];
    signal tmp_10579[3] <== [evals[126][0] * 9672572720882889100, evals[126][1] * 9672572720882889100, evals[126][2] * 9672572720882889100];
    signal tmp_10580[3] <== [tmp_10578[0] + tmp_10579[0], tmp_10578[1] + tmp_10579[1], tmp_10578[2] + tmp_10579[2]];
    tmp_10581 <== [tmp_10575[0] + tmp_10580[0], tmp_10575[1] + tmp_10580[1], tmp_10575[2] + tmp_10580[2]];
    signal tmp_10582[3] <== [17 * tmp_10581[0], 17 * tmp_10581[1], 17 * tmp_10581[2]];
    signal tmp_10583[3] <== [tmp_10571[0] + tmp_10582[0], tmp_10571[1] + tmp_10582[1], tmp_10571[2] + tmp_10582[2]];
    signal tmp_10584[3] <== CMul()(evals[105], evals[105]);
    signal tmp_10585[3] <== CMul()(tmp_10584, tmp_10584);
    signal tmp_10586[3] <== CMul()(tmp_10585, tmp_10584);
    tmp_10587 <== CMul()(tmp_10586, evals[105]);
    signal tmp_10588[3] <== [tmp_9501[0] * 14450902135526609586, tmp_9501[1] * 14450902135526609586, tmp_9501[2] * 14450902135526609586];
    signal tmp_10589[3] <== [tmp_9161[0] * 2685528509139446373, tmp_9161[1] * 2685528509139446373, tmp_9161[2] * 2685528509139446373];
    signal tmp_10590[3] <== [tmp_10588[0] + tmp_10589[0], tmp_10588[1] + tmp_10589[1], tmp_10588[2] + tmp_10589[2]];
    signal tmp_10591[3] <== [evals[126][0] * 7773309602241386138, evals[126][1] * 7773309602241386138, evals[126][2] * 7773309602241386138];
    signal tmp_10592[3] <== [tmp_10590[0] + tmp_10591[0], tmp_10590[1] + tmp_10591[1], tmp_10590[2] + tmp_10591[2]];
    tmp_10593 <== [tmp_10587[0] + tmp_10592[0], tmp_10587[1] + tmp_10592[1], tmp_10587[2] + tmp_10592[2]];
    tmp_10594 <== [2 * tmp_10593[0], 2 * tmp_10593[1], 2 * tmp_10593[2]];
    signal tmp_10595[3] <== [tmp_10583[0] + tmp_10594[0], tmp_10583[1] + tmp_10594[1], tmp_10583[2] + tmp_10594[2]];
    signal tmp_10596[3] <== CMul()(evals[106], evals[106]);
    signal tmp_10597[3] <== CMul()(tmp_10596, tmp_10596);
    signal tmp_10598[3] <== CMul()(tmp_10597, tmp_10596);
    tmp_10599 <== CMul()(tmp_10598, evals[106]);
    signal tmp_10600[3] <== [tmp_9501[0] * 9299676334145756732, tmp_9501[1] * 9299676334145756732, tmp_9501[2] * 9299676334145756732];
    signal tmp_10601[3] <== [tmp_9161[0] * 7618863604684140457, tmp_9161[1] * 7618863604684140457, tmp_9161[2] * 7618863604684140457];
    signal tmp_10602[3] <== [tmp_10600[0] + tmp_10601[0], tmp_10600[1] + tmp_10601[1], tmp_10600[2] + tmp_10601[2]];
    signal tmp_10603[3] <== [evals[126][0] * 13455068909528507193, evals[126][1] * 13455068909528507193, evals[126][2] * 13455068909528507193];
    signal tmp_10604[3] <== [tmp_10602[0] + tmp_10603[0], tmp_10602[1] + tmp_10603[1], tmp_10602[2] + tmp_10603[2]];
    tmp_10605 <== [tmp_10599[0] + tmp_10604[0], tmp_10599[1] + tmp_10604[1], tmp_10599[2] + tmp_10604[2]];
    signal tmp_10606[3] <== [tmp_10595[0] + tmp_10605[0], tmp_10595[1] + tmp_10605[1], tmp_10595[2] + tmp_10605[2]];
    signal tmp_10607[3] <== CMul()(evals[107], evals[107]);
    signal tmp_10608[3] <== CMul()(tmp_10607, tmp_10607);
    signal tmp_10609[3] <== CMul()(tmp_10608, tmp_10607);
    tmp_10610 <== CMul()(tmp_10609, evals[107]);
    signal tmp_10611[3] <== [tmp_9501[0] * 7761860768169821826, tmp_9501[1] * 7761860768169821826, tmp_9501[2] * 7761860768169821826];
    signal tmp_10612[3] <== [tmp_9161[0] * 15645272489037253163, tmp_9161[1] * 15645272489037253163, tmp_9161[2] * 15645272489037253163];
    signal tmp_10613[3] <== [tmp_10611[0] + tmp_10612[0], tmp_10611[1] + tmp_10612[1], tmp_10611[2] + tmp_10612[2]];
    signal tmp_10614[3] <== [evals[126][0] * 17592051316228424355, evals[126][1] * 17592051316228424355, evals[126][2] * 17592051316228424355];
    signal tmp_10615[3] <== [tmp_10613[0] + tmp_10614[0], tmp_10613[1] + tmp_10614[1], tmp_10613[2] + tmp_10614[2]];
    tmp_10616 <== [tmp_10610[0] + tmp_10615[0], tmp_10610[1] + tmp_10615[1], tmp_10610[2] + tmp_10615[2]];
    signal tmp_10617[3] <== [101 * tmp_10616[0], 101 * tmp_10616[1], 101 * tmp_10616[2]];
    signal tmp_10618[3] <== [tmp_10606[0] + tmp_10617[0], tmp_10606[1] + tmp_10617[1], tmp_10606[2] + tmp_10617[2]];
    signal tmp_10619[3] <== CMul()(evals[108], evals[108]);
    signal tmp_10620[3] <== CMul()(tmp_10619, tmp_10619);
    signal tmp_10621[3] <== CMul()(tmp_10620, tmp_10619);
    tmp_10622 <== CMul()(tmp_10621, evals[108]);
    signal tmp_10623[3] <== [tmp_9501[0] * 8088122716594392876, tmp_9501[1] * 8088122716594392876, tmp_9501[2] * 8088122716594392876];
    signal tmp_10624[3] <== [tmp_9161[0] * 14391536196044986046, tmp_9161[1] * 14391536196044986046, tmp_9161[2] * 14391536196044986046];
    signal tmp_10625[3] <== [tmp_10623[0] + tmp_10624[0], tmp_10623[1] + tmp_10624[1], tmp_10623[2] + tmp_10624[2]];
    signal tmp_10626[3] <== [evals[126][0] * 7038125295028811227, evals[126][1] * 7038125295028811227, evals[126][2] * 7038125295028811227];
    signal tmp_10627[3] <== [tmp_10625[0] + tmp_10626[0], tmp_10625[1] + tmp_10626[1], tmp_10625[2] + tmp_10626[2]];
    tmp_10628 <== [tmp_10622[0] + tmp_10627[0], tmp_10622[1] + tmp_10627[1], tmp_10622[2] + tmp_10627[2]];
    signal tmp_10629[3] <== [63 * tmp_10628[0], 63 * tmp_10628[1], 63 * tmp_10628[2]];
    signal tmp_10630[3] <== [tmp_10618[0] + tmp_10629[0], tmp_10618[1] + tmp_10629[1], tmp_10618[2] + tmp_10629[2]];
    signal tmp_10631[3] <== CMul()(evals[109], evals[109]);
    signal tmp_10632[3] <== CMul()(tmp_10631, tmp_10631);
    signal tmp_10633[3] <== CMul()(tmp_10632, tmp_10631);
    tmp_10634 <== CMul()(tmp_10633, evals[109]);
    signal tmp_10635[3] <== [tmp_9501[0] * 7439479476952963757, tmp_9501[1] * 7439479476952963757, tmp_9501[2] * 7439479476952963757];
    signal tmp_10636[3] <== [tmp_9161[0] * 13677749754501759967, tmp_9161[1] * 13677749754501759967, tmp_9161[2] * 13677749754501759967];
    signal tmp_10637[3] <== [tmp_10635[0] + tmp_10636[0], tmp_10635[1] + tmp_10636[1], tmp_10635[2] + tmp_10636[2]];
    signal tmp_10638[3] <== [evals[126][0] * 9310165980888335421, evals[126][1] * 9310165980888335421, evals[126][2] * 9310165980888335421];
    signal tmp_10639[3] <== [tmp_10637[0] + tmp_10638[0], tmp_10637[1] + tmp_10638[1], tmp_10637[2] + tmp_10638[2]];
    tmp_10640 <== [tmp_10634[0] + tmp_10639[0], tmp_10634[1] + tmp_10639[1], tmp_10634[2] + tmp_10639[2]];
    signal tmp_10641[3] <== [15 * tmp_10640[0], 15 * tmp_10640[1], 15 * tmp_10640[2]];
    signal tmp_10642[3] <== [tmp_10630[0] + tmp_10641[0], tmp_10630[1] + tmp_10641[1], tmp_10630[2] + tmp_10641[2]];
    signal tmp_10643[3] <== CMul()(evals[110], evals[110]);
    signal tmp_10644[3] <== CMul()(tmp_10643, tmp_10643);
    signal tmp_10645[3] <== CMul()(tmp_10644, tmp_10643);
    tmp_10646 <== CMul()(tmp_10645, evals[110]);
    signal tmp_10647[3] <== [tmp_9501[0] * 13142435206192622686, tmp_9501[1] * 13142435206192622686, tmp_9501[2] * 13142435206192622686];
    signal tmp_10648[3] <== [tmp_9161[0] * 4455490635243459169, tmp_9161[1] * 4455490635243459169, tmp_9161[2] * 4455490635243459169];
    signal tmp_10649[3] <== [tmp_10647[0] + tmp_10648[0], tmp_10647[1] + tmp_10648[1], tmp_10647[2] + tmp_10648[2]];
    signal tmp_10650[3] <== [evals[126][0] * 7124955715283261894, evals[126][1] * 7124955715283261894, evals[126][2] * 7124955715283261894];
    signal tmp_10651[3] <== [tmp_10649[0] + tmp_10650[0], tmp_10649[1] + tmp_10650[1], tmp_10649[2] + tmp_10650[2]];
    tmp_10652 <== [tmp_10646[0] + tmp_10651[0], tmp_10646[1] + tmp_10651[1], tmp_10646[2] + tmp_10651[2]];
    signal tmp_10653[3] <== [2 * tmp_10652[0], 2 * tmp_10652[1], 2 * tmp_10652[2]];
    signal tmp_10654[3] <== [tmp_10642[0] + tmp_10653[0], tmp_10642[1] + tmp_10653[1], tmp_10642[2] + tmp_10653[2]];
    signal tmp_10655[3] <== CMul()(evals[111], evals[111]);
    signal tmp_10656[3] <== CMul()(tmp_10655, tmp_10655);
    signal tmp_10657[3] <== CMul()(tmp_10656, tmp_10655);
    tmp_10658 <== CMul()(tmp_10657, evals[111]);
    signal tmp_10659[3] <== [tmp_9501[0] * 15091535460073700241, tmp_9501[1] * 15091535460073700241, tmp_9501[2] * 15091535460073700241];
    signal tmp_10660[3] <== [tmp_9161[0] * 4104818627819342438, tmp_9161[1] * 4104818627819342438, tmp_9161[2] * 4104818627819342438];
    signal tmp_10661[3] <== [tmp_10659[0] + tmp_10660[0], tmp_10659[1] + tmp_10660[1], tmp_10659[2] + tmp_10660[2]];
    signal tmp_10662[3] <== [evals[126][0] * 6894779481115729216, evals[126][1] * 6894779481115729216, evals[126][2] * 6894779481115729216];
    signal tmp_10663[3] <== [tmp_10661[0] + tmp_10662[0], tmp_10661[1] + tmp_10662[1], tmp_10661[2] + tmp_10662[2]];
    tmp_10664 <== [tmp_10658[0] + tmp_10663[0], tmp_10658[1] + tmp_10663[1], tmp_10658[2] + tmp_10663[2]];
    signal tmp_10665[3] <== [67 * tmp_10664[0], 67 * tmp_10664[1], 67 * tmp_10664[2]];
    signal tmp_10666[3] <== [tmp_10654[0] + tmp_10665[0], tmp_10654[1] + tmp_10665[1], tmp_10654[2] + tmp_10665[2]];
    signal tmp_10667[3] <== CMul()(evals[112], evals[112]);
    signal tmp_10668[3] <== CMul()(tmp_10667, tmp_10667);
    signal tmp_10669[3] <== CMul()(tmp_10668, tmp_10667);
    tmp_10670 <== CMul()(tmp_10669, evals[112]);
    signal tmp_10671[3] <== [tmp_9501[0] * 1246503809481338515, tmp_9501[1] * 1246503809481338515, tmp_9501[2] * 1246503809481338515];
    signal tmp_10672[3] <== [tmp_9161[0] * 11257046194955612097, tmp_9161[1] * 11257046194955612097, tmp_9161[2] * 11257046194955612097];
    signal tmp_10673[3] <== [tmp_10671[0] + tmp_10672[0], tmp_10671[1] + tmp_10672[1], tmp_10671[2] + tmp_10672[2]];
    signal tmp_10674[3] <== [evals[126][0] * 1173635621545091517, evals[126][1] * 1173635621545091517, evals[126][2] * 1173635621545091517];
    signal tmp_10675[3] <== [tmp_10673[0] + tmp_10674[0], tmp_10673[1] + tmp_10674[1], tmp_10673[2] + tmp_10674[2]];
    tmp_10676 <== [tmp_10670[0] + tmp_10675[0], tmp_10670[1] + tmp_10675[1], tmp_10670[2] + tmp_10675[2]];
    signal tmp_10677[3] <== [22 * tmp_10676[0], 22 * tmp_10676[1], 22 * tmp_10676[2]];
    signal tmp_10678[3] <== [tmp_10666[0] + tmp_10677[0], tmp_10666[1] + tmp_10677[1], tmp_10666[2] + tmp_10677[2]];
    signal tmp_10679[3] <== CMul()(evals[113], evals[113]);
    signal tmp_10680[3] <== CMul()(tmp_10679, tmp_10679);
    signal tmp_10681[3] <== CMul()(tmp_10680, tmp_10679);
    tmp_10682 <== CMul()(tmp_10681, evals[113]);
    signal tmp_10683[3] <== [tmp_9501[0] * 15424727577574863466, tmp_9501[1] * 15424727577574863466, tmp_9501[2] * 15424727577574863466];
    signal tmp_10684[3] <== [tmp_9161[0] * 17740901046700926130, tmp_9161[1] * 17740901046700926130, tmp_9161[2] * 17740901046700926130];
    signal tmp_10685[3] <== [tmp_10683[0] + tmp_10684[0], tmp_10683[1] + tmp_10684[1], tmp_10683[2] + tmp_10684[2]];
    signal tmp_10686[3] <== [evals[126][0] * 2648585141728028210, evals[126][1] * 2648585141728028210, evals[126][2] * 2648585141728028210];
    signal tmp_10687[3] <== [tmp_10685[0] + tmp_10686[0], tmp_10685[1] + tmp_10686[1], tmp_10685[2] + tmp_10686[2]];
    tmp_10688 <== [tmp_10682[0] + tmp_10687[0], tmp_10682[1] + tmp_10687[1], tmp_10682[2] + tmp_10687[2]];
    signal tmp_10689[3] <== [13 * tmp_10688[0], 13 * tmp_10688[1], 13 * tmp_10688[2]];
    signal tmp_10690[3] <== [tmp_10678[0] + tmp_10689[0], tmp_10678[1] + tmp_10689[1], tmp_10678[2] + tmp_10689[2]];
    signal tmp_10691[3] <== CMul()(evals[114], evals[114]);
    signal tmp_10692[3] <== CMul()(tmp_10691, tmp_10691);
    signal tmp_10693[3] <== CMul()(tmp_10692, tmp_10691);
    tmp_10694 <== CMul()(tmp_10693, evals[114]);
    signal tmp_10695[3] <== [tmp_9501[0] * 4922985543021971848, tmp_9501[1] * 4922985543021971848, tmp_9501[2] * 4922985543021971848];
    signal tmp_10696[3] <== [tmp_9161[0] * 10900350254403521826, tmp_9161[1] * 10900350254403521826, tmp_9161[2] * 10900350254403521826];
    signal tmp_10697[3] <== [tmp_10695[0] + tmp_10696[0], tmp_10695[1] + tmp_10696[1], tmp_10695[2] + tmp_10696[2]];
    signal tmp_10698[3] <== [evals[126][0] * 10841588854694617600, evals[126][1] * 10841588854694617600, evals[126][2] * 10841588854694617600];
    signal tmp_10699[3] <== [tmp_10697[0] + tmp_10698[0], tmp_10697[1] + tmp_10698[1], tmp_10697[2] + tmp_10698[2]];
    tmp_10700 <== [tmp_10694[0] + tmp_10699[0], tmp_10694[1] + tmp_10699[1], tmp_10694[2] + tmp_10699[2]];
    signal tmp_10701[3] <== [3 * tmp_10700[0], 3 * tmp_10700[1], 3 * tmp_10700[2]];
    tmp_10702 <== [tmp_10690[0] + tmp_10701[0], tmp_10690[1] + tmp_10701[1], tmp_10690[2] + tmp_10701[2]];
    signal tmp_10703[3] <== [tmp_10515[0] - tmp_10702[0], tmp_10515[1] - tmp_10702[1], tmp_10515[2] - tmp_10702[2]];
    signal tmp_10704[3] <== CMul()(tmp_10511, tmp_10703);
    signal tmp_10705[3] <== [tmp_10509[0] + tmp_10704[0], tmp_10509[1] + tmp_10704[1], tmp_10509[2] + tmp_10704[2]];
    signal tmp_10706[3] <== CMul()(challengeQ, tmp_10705);
    signal tmp_10709[3] <== CMul()(evals[57], evals[68]);
    signal tmp_10711[3] <== CMul()(tmp_10513, evals[136]);
    signal tmp_10712[3] <== [tmp_10709[0] + tmp_10711[0], tmp_10709[1] + tmp_10711[1], tmp_10709[2] + tmp_10711[2]];
    signal tmp_10714[3] <== [3 * tmp_10525[0], 3 * tmp_10525[1], 3 * tmp_10525[2]];
    signal tmp_10716[3] <== [tmp_10714[0] + tmp_10535[0], tmp_10714[1] + tmp_10535[1], tmp_10714[2] + tmp_10535[2]];
    signal tmp_10718[3] <== [tmp_10716[0] + tmp_10546[0], tmp_10716[1] + tmp_10546[1], tmp_10716[2] + tmp_10546[2]];
    signal tmp_10720[3] <== [51 * tmp_10558[0], 51 * tmp_10558[1], 51 * tmp_10558[2]];
    signal tmp_10721[3] <== [tmp_10718[0] + tmp_10720[0], tmp_10718[1] + tmp_10720[1], tmp_10718[2] + tmp_10720[2]];
    signal tmp_10723[3] <== [tmp_10721[0] + tmp_10569[0], tmp_10721[1] + tmp_10569[1], tmp_10721[2] + tmp_10569[2]];
    signal tmp_10725[3] <== [11 * tmp_10581[0], 11 * tmp_10581[1], 11 * tmp_10581[2]];
    signal tmp_10726[3] <== [tmp_10723[0] + tmp_10725[0], tmp_10723[1] + tmp_10725[1], tmp_10723[2] + tmp_10725[2]];
    signal tmp_10728[3] <== [17 * tmp_10593[0], 17 * tmp_10593[1], 17 * tmp_10593[2]];
    signal tmp_10729[3] <== [tmp_10726[0] + tmp_10728[0], tmp_10726[1] + tmp_10728[1], tmp_10726[2] + tmp_10728[2]];
    tmp_10731 <== [2 * tmp_10605[0], 2 * tmp_10605[1], 2 * tmp_10605[2]];
    signal tmp_10732[3] <== [tmp_10729[0] + tmp_10731[0], tmp_10729[1] + tmp_10731[1], tmp_10729[2] + tmp_10731[2]];
    signal tmp_10734[3] <== [tmp_10732[0] + tmp_10616[0], tmp_10732[1] + tmp_10616[1], tmp_10732[2] + tmp_10616[2]];
    signal tmp_10736[3] <== [101 * tmp_10628[0], 101 * tmp_10628[1], 101 * tmp_10628[2]];
    signal tmp_10737[3] <== [tmp_10734[0] + tmp_10736[0], tmp_10734[1] + tmp_10736[1], tmp_10734[2] + tmp_10736[2]];
    signal tmp_10739[3] <== [63 * tmp_10640[0], 63 * tmp_10640[1], 63 * tmp_10640[2]];
    signal tmp_10740[3] <== [tmp_10737[0] + tmp_10739[0], tmp_10737[1] + tmp_10739[1], tmp_10737[2] + tmp_10739[2]];
    signal tmp_10742[3] <== [15 * tmp_10652[0], 15 * tmp_10652[1], 15 * tmp_10652[2]];
    signal tmp_10743[3] <== [tmp_10740[0] + tmp_10742[0], tmp_10740[1] + tmp_10742[1], tmp_10740[2] + tmp_10742[2]];
    tmp_10745 <== [2 * tmp_10664[0], 2 * tmp_10664[1], 2 * tmp_10664[2]];
    signal tmp_10746[3] <== [tmp_10743[0] + tmp_10745[0], tmp_10743[1] + tmp_10745[1], tmp_10743[2] + tmp_10745[2]];
    signal tmp_10748[3] <== [67 * tmp_10676[0], 67 * tmp_10676[1], 67 * tmp_10676[2]];
    signal tmp_10749[3] <== [tmp_10746[0] + tmp_10748[0], tmp_10746[1] + tmp_10748[1], tmp_10746[2] + tmp_10748[2]];
    signal tmp_10751[3] <== [22 * tmp_10688[0], 22 * tmp_10688[1], 22 * tmp_10688[2]];
    signal tmp_10752[3] <== [tmp_10749[0] + tmp_10751[0], tmp_10749[1] + tmp_10751[1], tmp_10749[2] + tmp_10751[2]];
    signal tmp_10754[3] <== [13 * tmp_10700[0], 13 * tmp_10700[1], 13 * tmp_10700[2]];
    signal tmp_10755[3] <== [tmp_10752[0] + tmp_10754[0], tmp_10752[1] + tmp_10754[1], tmp_10752[2] + tmp_10754[2]];
    signal tmp_10756[3] <== [tmp_10712[0] - tmp_10755[0], tmp_10712[1] - tmp_10755[1], tmp_10712[2] - tmp_10755[2]];
    signal tmp_10757[3] <== CMul()(tmp_10511, tmp_10756);
    signal tmp_10758[3] <== [tmp_10706[0] + tmp_10757[0], tmp_10706[1] + tmp_10757[1], tmp_10706[2] + tmp_10757[2]];
    signal tmp_10759[3] <== CMul()(challengeQ, tmp_10758);
    signal tmp_10762[3] <== CMul()(evals[57], evals[69]);
    signal tmp_10764[3] <== CMul()(tmp_10513, evals[137]);
    signal tmp_10765[3] <== [tmp_10762[0] + tmp_10764[0], tmp_10762[1] + tmp_10764[1], tmp_10762[2] + tmp_10764[2]];
    signal tmp_10767[3] <== [13 * tmp_10525[0], 13 * tmp_10525[1], 13 * tmp_10525[2]];
    signal tmp_10769[3] <== [3 * tmp_10535[0], 3 * tmp_10535[1], 3 * tmp_10535[2]];
    signal tmp_10770[3] <== [tmp_10767[0] + tmp_10769[0], tmp_10767[1] + tmp_10769[1], tmp_10767[2] + tmp_10769[2]];
    signal tmp_10772[3] <== [tmp_10770[0] + tmp_10546[0], tmp_10770[1] + tmp_10546[1], tmp_10770[2] + tmp_10546[2]];
    signal tmp_10774[3] <== [tmp_10772[0] + tmp_10558[0], tmp_10772[1] + tmp_10558[1], tmp_10772[2] + tmp_10558[2]];
    signal tmp_10776[3] <== [51 * tmp_10569[0], 51 * tmp_10569[1], 51 * tmp_10569[2]];
    signal tmp_10777[3] <== [tmp_10774[0] + tmp_10776[0], tmp_10774[1] + tmp_10776[1], tmp_10774[2] + tmp_10776[2]];
    signal tmp_10779[3] <== [tmp_10777[0] + tmp_10581[0], tmp_10777[1] + tmp_10581[1], tmp_10777[2] + tmp_10581[2]];
    signal tmp_10781[3] <== [11 * tmp_10593[0], 11 * tmp_10593[1], 11 * tmp_10593[2]];
    signal tmp_10782[3] <== [tmp_10779[0] + tmp_10781[0], tmp_10779[1] + tmp_10781[1], tmp_10779[2] + tmp_10781[2]];
    signal tmp_10784[3] <== [17 * tmp_10605[0], 17 * tmp_10605[1], 17 * tmp_10605[2]];
    signal tmp_10785[3] <== [tmp_10782[0] + tmp_10784[0], tmp_10782[1] + tmp_10784[1], tmp_10782[2] + tmp_10784[2]];
    tmp_10787 <== [2 * tmp_10616[0], 2 * tmp_10616[1], 2 * tmp_10616[2]];
    signal tmp_10788[3] <== [tmp_10785[0] + tmp_10787[0], tmp_10785[1] + tmp_10787[1], tmp_10785[2] + tmp_10787[2]];
    signal tmp_10790[3] <== [tmp_10788[0] + tmp_10628[0], tmp_10788[1] + tmp_10628[1], tmp_10788[2] + tmp_10628[2]];
    signal tmp_10792[3] <== [101 * tmp_10640[0], 101 * tmp_10640[1], 101 * tmp_10640[2]];
    signal tmp_10793[3] <== [tmp_10790[0] + tmp_10792[0], tmp_10790[1] + tmp_10792[1], tmp_10790[2] + tmp_10792[2]];
    signal tmp_10795[3] <== [63 * tmp_10652[0], 63 * tmp_10652[1], 63 * tmp_10652[2]];
    signal tmp_10796[3] <== [tmp_10793[0] + tmp_10795[0], tmp_10793[1] + tmp_10795[1], tmp_10793[2] + tmp_10795[2]];
    signal tmp_10798[3] <== [15 * tmp_10664[0], 15 * tmp_10664[1], 15 * tmp_10664[2]];
    signal tmp_10799[3] <== [tmp_10796[0] + tmp_10798[0], tmp_10796[1] + tmp_10798[1], tmp_10796[2] + tmp_10798[2]];
    tmp_10801 <== [2 * tmp_10676[0], 2 * tmp_10676[1], 2 * tmp_10676[2]];
    signal tmp_10802[3] <== [tmp_10799[0] + tmp_10801[0], tmp_10799[1] + tmp_10801[1], tmp_10799[2] + tmp_10801[2]];
    signal tmp_10804[3] <== [67 * tmp_10688[0], 67 * tmp_10688[1], 67 * tmp_10688[2]];
    signal tmp_10805[3] <== [tmp_10802[0] + tmp_10804[0], tmp_10802[1] + tmp_10804[1], tmp_10802[2] + tmp_10804[2]];
    signal tmp_10807[3] <== [22 * tmp_10700[0], 22 * tmp_10700[1], 22 * tmp_10700[2]];
    signal tmp_10808[3] <== [tmp_10805[0] + tmp_10807[0], tmp_10805[1] + tmp_10807[1], tmp_10805[2] + tmp_10807[2]];
    signal tmp_10809[3] <== [tmp_10765[0] - tmp_10808[0], tmp_10765[1] - tmp_10808[1], tmp_10765[2] - tmp_10808[2]];
    signal tmp_10810[3] <== CMul()(tmp_10511, tmp_10809);
    signal tmp_10811[3] <== [tmp_10759[0] + tmp_10810[0], tmp_10759[1] + tmp_10810[1], tmp_10759[2] + tmp_10810[2]];
    signal tmp_10812[3] <== CMul()(challengeQ, tmp_10811);
    signal tmp_10815[3] <== CMul()(evals[57], evals[70]);
    signal tmp_10817[3] <== CMul()(tmp_10513, evals[138]);
    signal tmp_10818[3] <== [tmp_10815[0] + tmp_10817[0], tmp_10815[1] + tmp_10817[1], tmp_10815[2] + tmp_10817[2]];
    signal tmp_10820[3] <== [22 * tmp_10525[0], 22 * tmp_10525[1], 22 * tmp_10525[2]];
    signal tmp_10822[3] <== [13 * tmp_10535[0], 13 * tmp_10535[1], 13 * tmp_10535[2]];
    signal tmp_10823[3] <== [tmp_10820[0] + tmp_10822[0], tmp_10820[1] + tmp_10822[1], tmp_10820[2] + tmp_10822[2]];
    signal tmp_10825[3] <== [3 * tmp_10546[0], 3 * tmp_10546[1], 3 * tmp_10546[2]];
    signal tmp_10826[3] <== [tmp_10823[0] + tmp_10825[0], tmp_10823[1] + tmp_10825[1], tmp_10823[2] + tmp_10825[2]];
    signal tmp_10828[3] <== [tmp_10826[0] + tmp_10558[0], tmp_10826[1] + tmp_10558[1], tmp_10826[2] + tmp_10558[2]];
    signal tmp_10830[3] <== [tmp_10828[0] + tmp_10569[0], tmp_10828[1] + tmp_10569[1], tmp_10828[2] + tmp_10569[2]];
    signal tmp_10832[3] <== [51 * tmp_10581[0], 51 * tmp_10581[1], 51 * tmp_10581[2]];
    signal tmp_10833[3] <== [tmp_10830[0] + tmp_10832[0], tmp_10830[1] + tmp_10832[1], tmp_10830[2] + tmp_10832[2]];
    signal tmp_10835[3] <== [tmp_10833[0] + tmp_10593[0], tmp_10833[1] + tmp_10593[1], tmp_10833[2] + tmp_10593[2]];
    signal tmp_10837[3] <== [11 * tmp_10605[0], 11 * tmp_10605[1], 11 * tmp_10605[2]];
    signal tmp_10838[3] <== [tmp_10835[0] + tmp_10837[0], tmp_10835[1] + tmp_10837[1], tmp_10835[2] + tmp_10837[2]];
    signal tmp_10840[3] <== [17 * tmp_10616[0], 17 * tmp_10616[1], 17 * tmp_10616[2]];
    signal tmp_10841[3] <== [tmp_10838[0] + tmp_10840[0], tmp_10838[1] + tmp_10840[1], tmp_10838[2] + tmp_10840[2]];
    tmp_10843 <== [2 * tmp_10628[0], 2 * tmp_10628[1], 2 * tmp_10628[2]];
    signal tmp_10844[3] <== [tmp_10841[0] + tmp_10843[0], tmp_10841[1] + tmp_10843[1], tmp_10841[2] + tmp_10843[2]];
    signal tmp_10846[3] <== [tmp_10844[0] + tmp_10640[0], tmp_10844[1] + tmp_10640[1], tmp_10844[2] + tmp_10640[2]];
    signal tmp_10848[3] <== [101 * tmp_10652[0], 101 * tmp_10652[1], 101 * tmp_10652[2]];
    signal tmp_10849[3] <== [tmp_10846[0] + tmp_10848[0], tmp_10846[1] + tmp_10848[1], tmp_10846[2] + tmp_10848[2]];
    signal tmp_10851[3] <== [63 * tmp_10664[0], 63 * tmp_10664[1], 63 * tmp_10664[2]];
    signal tmp_10852[3] <== [tmp_10849[0] + tmp_10851[0], tmp_10849[1] + tmp_10851[1], tmp_10849[2] + tmp_10851[2]];
    signal tmp_10854[3] <== [15 * tmp_10676[0], 15 * tmp_10676[1], 15 * tmp_10676[2]];
    signal tmp_10855[3] <== [tmp_10852[0] + tmp_10854[0], tmp_10852[1] + tmp_10854[1], tmp_10852[2] + tmp_10854[2]];
    tmp_10857 <== [2 * tmp_10688[0], 2 * tmp_10688[1], 2 * tmp_10688[2]];
    signal tmp_10858[3] <== [tmp_10855[0] + tmp_10857[0], tmp_10855[1] + tmp_10857[1], tmp_10855[2] + tmp_10857[2]];
    signal tmp_10860[3] <== [67 * tmp_10700[0], 67 * tmp_10700[1], 67 * tmp_10700[2]];
    signal tmp_10861[3] <== [tmp_10858[0] + tmp_10860[0], tmp_10858[1] + tmp_10860[1], tmp_10858[2] + tmp_10860[2]];
    signal tmp_10862[3] <== [tmp_10818[0] - tmp_10861[0], tmp_10818[1] - tmp_10861[1], tmp_10818[2] - tmp_10861[2]];
    signal tmp_10863[3] <== CMul()(tmp_10511, tmp_10862);
    signal tmp_10864[3] <== [tmp_10812[0] + tmp_10863[0], tmp_10812[1] + tmp_10863[1], tmp_10812[2] + tmp_10863[2]];
    signal tmp_10865[3] <== CMul()(challengeQ, tmp_10864);
    signal tmp_10868[3] <== CMul()(evals[57], evals[71]);
    signal tmp_10870[3] <== CMul()(tmp_10513, evals[139]);
    signal tmp_10871[3] <== [tmp_10868[0] + tmp_10870[0], tmp_10868[1] + tmp_10870[1], tmp_10868[2] + tmp_10870[2]];
    signal tmp_10873[3] <== [67 * tmp_10525[0], 67 * tmp_10525[1], 67 * tmp_10525[2]];
    signal tmp_10875[3] <== [22 * tmp_10535[0], 22 * tmp_10535[1], 22 * tmp_10535[2]];
    signal tmp_10876[3] <== [tmp_10873[0] + tmp_10875[0], tmp_10873[1] + tmp_10875[1], tmp_10873[2] + tmp_10875[2]];
    signal tmp_10878[3] <== [13 * tmp_10546[0], 13 * tmp_10546[1], 13 * tmp_10546[2]];
    signal tmp_10879[3] <== [tmp_10876[0] + tmp_10878[0], tmp_10876[1] + tmp_10878[1], tmp_10876[2] + tmp_10878[2]];
    signal tmp_10881[3] <== [3 * tmp_10558[0], 3 * tmp_10558[1], 3 * tmp_10558[2]];
    signal tmp_10882[3] <== [tmp_10879[0] + tmp_10881[0], tmp_10879[1] + tmp_10881[1], tmp_10879[2] + tmp_10881[2]];
    signal tmp_10884[3] <== [tmp_10882[0] + tmp_10569[0], tmp_10882[1] + tmp_10569[1], tmp_10882[2] + tmp_10569[2]];
    signal tmp_10886[3] <== [tmp_10884[0] + tmp_10581[0], tmp_10884[1] + tmp_10581[1], tmp_10884[2] + tmp_10581[2]];
    signal tmp_10888[3] <== [51 * tmp_10593[0], 51 * tmp_10593[1], 51 * tmp_10593[2]];
    signal tmp_10889[3] <== [tmp_10886[0] + tmp_10888[0], tmp_10886[1] + tmp_10888[1], tmp_10886[2] + tmp_10888[2]];
    signal tmp_10891[3] <== [tmp_10889[0] + tmp_10605[0], tmp_10889[1] + tmp_10605[1], tmp_10889[2] + tmp_10605[2]];
    signal tmp_10893[3] <== [11 * tmp_10616[0], 11 * tmp_10616[1], 11 * tmp_10616[2]];
    signal tmp_10894[3] <== [tmp_10891[0] + tmp_10893[0], tmp_10891[1] + tmp_10893[1], tmp_10891[2] + tmp_10893[2]];
    signal tmp_10896[3] <== [17 * tmp_10628[0], 17 * tmp_10628[1], 17 * tmp_10628[2]];
    signal tmp_10897[3] <== [tmp_10894[0] + tmp_10896[0], tmp_10894[1] + tmp_10896[1], tmp_10894[2] + tmp_10896[2]];
    tmp_10899 <== [2 * tmp_10640[0], 2 * tmp_10640[1], 2 * tmp_10640[2]];
    signal tmp_10900[3] <== [tmp_10897[0] + tmp_10899[0], tmp_10897[1] + tmp_10899[1], tmp_10897[2] + tmp_10899[2]];
    signal tmp_10902[3] <== [tmp_10900[0] + tmp_10652[0], tmp_10900[1] + tmp_10652[1], tmp_10900[2] + tmp_10652[2]];
    signal tmp_10904[3] <== [101 * tmp_10664[0], 101 * tmp_10664[1], 101 * tmp_10664[2]];
    signal tmp_10905[3] <== [tmp_10902[0] + tmp_10904[0], tmp_10902[1] + tmp_10904[1], tmp_10902[2] + tmp_10904[2]];
    signal tmp_10907[3] <== [63 * tmp_10676[0], 63 * tmp_10676[1], 63 * tmp_10676[2]];
    signal tmp_10908[3] <== [tmp_10905[0] + tmp_10907[0], tmp_10905[1] + tmp_10907[1], tmp_10905[2] + tmp_10907[2]];
    signal tmp_10910[3] <== [15 * tmp_10688[0], 15 * tmp_10688[1], 15 * tmp_10688[2]];
    signal tmp_10911[3] <== [tmp_10908[0] + tmp_10910[0], tmp_10908[1] + tmp_10910[1], tmp_10908[2] + tmp_10910[2]];
    tmp_10913 <== [2 * tmp_10700[0], 2 * tmp_10700[1], 2 * tmp_10700[2]];
    signal tmp_10914[3] <== [tmp_10911[0] + tmp_10913[0], tmp_10911[1] + tmp_10913[1], tmp_10911[2] + tmp_10913[2]];
    signal tmp_10915[3] <== [tmp_10871[0] - tmp_10914[0], tmp_10871[1] - tmp_10914[1], tmp_10871[2] - tmp_10914[2]];
    signal tmp_10916[3] <== CMul()(tmp_10511, tmp_10915);
    signal tmp_10917[3] <== [tmp_10865[0] + tmp_10916[0], tmp_10865[1] + tmp_10916[1], tmp_10865[2] + tmp_10916[2]];
    signal tmp_10918[3] <== CMul()(challengeQ, tmp_10917);
    signal tmp_10921[3] <== CMul()(evals[57], evals[72]);
    signal tmp_10923[3] <== CMul()(tmp_10513, evals[140]);
    signal tmp_10924[3] <== [tmp_10921[0] + tmp_10923[0], tmp_10921[1] + tmp_10923[1], tmp_10921[2] + tmp_10923[2]];
    tmp_10926 <== [2 * tmp_10525[0], 2 * tmp_10525[1], 2 * tmp_10525[2]];
    signal tmp_10928[3] <== [67 * tmp_10535[0], 67 * tmp_10535[1], 67 * tmp_10535[2]];
    signal tmp_10929[3] <== [tmp_10926[0] + tmp_10928[0], tmp_10926[1] + tmp_10928[1], tmp_10926[2] + tmp_10928[2]];
    signal tmp_10931[3] <== [22 * tmp_10546[0], 22 * tmp_10546[1], 22 * tmp_10546[2]];
    signal tmp_10932[3] <== [tmp_10929[0] + tmp_10931[0], tmp_10929[1] + tmp_10931[1], tmp_10929[2] + tmp_10931[2]];
    signal tmp_10934[3] <== [13 * tmp_10558[0], 13 * tmp_10558[1], 13 * tmp_10558[2]];
    signal tmp_10935[3] <== [tmp_10932[0] + tmp_10934[0], tmp_10932[1] + tmp_10934[1], tmp_10932[2] + tmp_10934[2]];
    signal tmp_10937[3] <== [3 * tmp_10569[0], 3 * tmp_10569[1], 3 * tmp_10569[2]];
    signal tmp_10938[3] <== [tmp_10935[0] + tmp_10937[0], tmp_10935[1] + tmp_10937[1], tmp_10935[2] + tmp_10937[2]];
    signal tmp_10940[3] <== [tmp_10938[0] + tmp_10581[0], tmp_10938[1] + tmp_10581[1], tmp_10938[2] + tmp_10581[2]];
    signal tmp_10942[3] <== [tmp_10940[0] + tmp_10593[0], tmp_10940[1] + tmp_10593[1], tmp_10940[2] + tmp_10593[2]];
    signal tmp_10944[3] <== [51 * tmp_10605[0], 51 * tmp_10605[1], 51 * tmp_10605[2]];
    signal tmp_10945[3] <== [tmp_10942[0] + tmp_10944[0], tmp_10942[1] + tmp_10944[1], tmp_10942[2] + tmp_10944[2]];
    signal tmp_10947[3] <== [tmp_10945[0] + tmp_10616[0], tmp_10945[1] + tmp_10616[1], tmp_10945[2] + tmp_10616[2]];
    signal tmp_10949[3] <== [11 * tmp_10628[0], 11 * tmp_10628[1], 11 * tmp_10628[2]];
    signal tmp_10950[3] <== [tmp_10947[0] + tmp_10949[0], tmp_10947[1] + tmp_10949[1], tmp_10947[2] + tmp_10949[2]];
    signal tmp_10952[3] <== [17 * tmp_10640[0], 17 * tmp_10640[1], 17 * tmp_10640[2]];
    signal tmp_10953[3] <== [tmp_10950[0] + tmp_10952[0], tmp_10950[1] + tmp_10952[1], tmp_10950[2] + tmp_10952[2]];
    signal tmp_10956[3] <== [tmp_10953[0] + tmp_10653[0], tmp_10953[1] + tmp_10653[1], tmp_10953[2] + tmp_10653[2]];
    signal tmp_10958[3] <== [tmp_10956[0] + tmp_10664[0], tmp_10956[1] + tmp_10664[1], tmp_10956[2] + tmp_10664[2]];
    signal tmp_10960[3] <== [101 * tmp_10676[0], 101 * tmp_10676[1], 101 * tmp_10676[2]];
    signal tmp_10961[3] <== [tmp_10958[0] + tmp_10960[0], tmp_10958[1] + tmp_10960[1], tmp_10958[2] + tmp_10960[2]];
    signal tmp_10963[3] <== [63 * tmp_10688[0], 63 * tmp_10688[1], 63 * tmp_10688[2]];
    signal tmp_10964[3] <== [tmp_10961[0] + tmp_10963[0], tmp_10961[1] + tmp_10963[1], tmp_10961[2] + tmp_10963[2]];
    signal tmp_10966[3] <== [15 * tmp_10700[0], 15 * tmp_10700[1], 15 * tmp_10700[2]];
    signal tmp_10967[3] <== [tmp_10964[0] + tmp_10966[0], tmp_10964[1] + tmp_10966[1], tmp_10964[2] + tmp_10966[2]];
    signal tmp_10968[3] <== [tmp_10924[0] - tmp_10967[0], tmp_10924[1] - tmp_10967[1], tmp_10924[2] - tmp_10967[2]];
    signal tmp_10969[3] <== CMul()(tmp_10511, tmp_10968);
    signal tmp_10970[3] <== [tmp_10918[0] + tmp_10969[0], tmp_10918[1] + tmp_10969[1], tmp_10918[2] + tmp_10969[2]];
    tmp_10971 <== CMul()(challengeQ, tmp_10970);
    signal tmp_10974[3] <== CMul()(evals[57], evals[73]);
    signal tmp_10976[3] <== CMul()(tmp_10513, evals[141]);
    tmp_10977 <== [tmp_10974[0] + tmp_10976[0], tmp_10974[1] + tmp_10976[1], tmp_10974[2] + tmp_10976[2]];
    signal tmp_10979[3] <== [15 * tmp_10525[0], 15 * tmp_10525[1], 15 * tmp_10525[2]];
    tmp_10981 <== [2 * tmp_10535[0], 2 * tmp_10535[1], 2 * tmp_10535[2]];
    signal tmp_10982[3] <== [tmp_10979[0] + tmp_10981[0], tmp_10979[1] + tmp_10981[1], tmp_10979[2] + tmp_10981[2]];
    signal tmp_10984[3] <== [67 * tmp_10546[0], 67 * tmp_10546[1], 67 * tmp_10546[2]];
    signal tmp_10985[3] <== [tmp_10982[0] + tmp_10984[0], tmp_10982[1] + tmp_10984[1], tmp_10982[2] + tmp_10984[2]];
    signal tmp_10987[3] <== [22 * tmp_10558[0], 22 * tmp_10558[1], 22 * tmp_10558[2]];
    tmp_10988 <== [tmp_10985[0] + tmp_10987[0], tmp_10985[1] + tmp_10987[1], tmp_10985[2] + tmp_10987[2]];
    tmp_10990 <== [13 * tmp_10569[0], 13 * tmp_10569[1], 13 * tmp_10569[2]];
}

template VerifyEvaluationsChunks3_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_9161[3];
    signal input tmp_9163[3];
    signal input tmp_10511[3];
    signal input tmp_10513[3];
    signal input tmp_10519[3];
    signal input tmp_10525[3];
    signal input tmp_10529[3];
    signal input tmp_10535[3];
    signal input tmp_10540[3];
    signal input tmp_10546[3];
    signal input tmp_10558[3];
    signal input tmp_10569[3];
    signal input tmp_10581[3];
    signal input tmp_10593[3];
    signal input tmp_10594[3];
    signal input tmp_10605[3];
    signal input tmp_10616[3];
    signal input tmp_10628[3];
    signal input tmp_10640[3];
    signal input tmp_10652[3];
    signal input tmp_10664[3];
    signal input tmp_10676[3];
    signal input tmp_10688[3];
    signal input tmp_10700[3];
    signal input tmp_10702[3];
    signal input tmp_10731[3];
    signal input tmp_10745[3];
    signal input tmp_10787[3];
    signal input tmp_10801[3];
    signal input tmp_10843[3];
    signal input tmp_10857[3];
    signal input tmp_10899[3];
    signal input tmp_10913[3];
    signal input tmp_10926[3];
    signal input tmp_10971[3];
    signal input tmp_10977[3];
    signal input tmp_10981[3];
    signal input tmp_10988[3];
    signal input tmp_10990[3];

    signal output tmp_12354[3];
    signal output tmp_12417[3];
    signal output tmp_12422[3];
    signal output tmp_12427[3];
    signal output tmp_12434[3];
    signal output tmp_12435[3];
    signal output tmp_12438[3];
    signal output tmp_12443[3];
    signal output tmp_12448[3];
    signal output tmp_12453[3];
    signal output tmp_12458[3];
    signal output tmp_12463[3];
    signal output tmp_12468[3];
    signal output tmp_12473[3];
    signal output tmp_12478[3];
    signal output tmp_12483[3];
    signal output tmp_12488[3];
    signal output tmp_12493[3];
    signal output tmp_12495[3];
    signal output tmp_12497[3];
    signal tmp_10991[3] <== [tmp_10988[0] + tmp_10990[0], tmp_10988[1] + tmp_10990[1], tmp_10988[2] + tmp_10990[2]];
    signal tmp_10993[3] <== [3 * tmp_10581[0], 3 * tmp_10581[1], 3 * tmp_10581[2]];
    signal tmp_10994[3] <== [tmp_10991[0] + tmp_10993[0], tmp_10991[1] + tmp_10993[1], tmp_10991[2] + tmp_10993[2]];
    signal tmp_10996[3] <== [tmp_10994[0] + tmp_10593[0], tmp_10994[1] + tmp_10593[1], tmp_10994[2] + tmp_10593[2]];
    signal tmp_10998[3] <== [tmp_10996[0] + tmp_10605[0], tmp_10996[1] + tmp_10605[1], tmp_10996[2] + tmp_10605[2]];
    signal tmp_11000[3] <== [51 * tmp_10616[0], 51 * tmp_10616[1], 51 * tmp_10616[2]];
    signal tmp_11001[3] <== [tmp_10998[0] + tmp_11000[0], tmp_10998[1] + tmp_11000[1], tmp_10998[2] + tmp_11000[2]];
    signal tmp_11003[3] <== [tmp_11001[0] + tmp_10628[0], tmp_11001[1] + tmp_10628[1], tmp_11001[2] + tmp_10628[2]];
    signal tmp_11005[3] <== [11 * tmp_10640[0], 11 * tmp_10640[1], 11 * tmp_10640[2]];
    signal tmp_11006[3] <== [tmp_11003[0] + tmp_11005[0], tmp_11003[1] + tmp_11005[1], tmp_11003[2] + tmp_11005[2]];
    signal tmp_11008[3] <== [17 * tmp_10652[0], 17 * tmp_10652[1], 17 * tmp_10652[2]];
    signal tmp_11009[3] <== [tmp_11006[0] + tmp_11008[0], tmp_11006[1] + tmp_11008[1], tmp_11006[2] + tmp_11008[2]];
    signal tmp_11012[3] <== [tmp_11009[0] + tmp_10745[0], tmp_11009[1] + tmp_10745[1], tmp_11009[2] + tmp_10745[2]];
    signal tmp_11014[3] <== [tmp_11012[0] + tmp_10676[0], tmp_11012[1] + tmp_10676[1], tmp_11012[2] + tmp_10676[2]];
    signal tmp_11016[3] <== [101 * tmp_10688[0], 101 * tmp_10688[1], 101 * tmp_10688[2]];
    signal tmp_11017[3] <== [tmp_11014[0] + tmp_11016[0], tmp_11014[1] + tmp_11016[1], tmp_11014[2] + tmp_11016[2]];
    signal tmp_11019[3] <== [63 * tmp_10700[0], 63 * tmp_10700[1], 63 * tmp_10700[2]];
    signal tmp_11020[3] <== [tmp_11017[0] + tmp_11019[0], tmp_11017[1] + tmp_11019[1], tmp_11017[2] + tmp_11019[2]];
    signal tmp_11021[3] <== [tmp_10977[0] - tmp_11020[0], tmp_10977[1] - tmp_11020[1], tmp_10977[2] - tmp_11020[2]];
    signal tmp_11022[3] <== CMul()(tmp_10511, tmp_11021);
    signal tmp_11023[3] <== [tmp_10971[0] + tmp_11022[0], tmp_10971[1] + tmp_11022[1], tmp_10971[2] + tmp_11022[2]];
    signal tmp_11024[3] <== CMul()(challengeQ, tmp_11023);
    signal tmp_11027[3] <== CMul()(evals[57], evals[74]);
    signal tmp_11029[3] <== CMul()(tmp_10513, evals[142]);
    signal tmp_11030[3] <== [tmp_11027[0] + tmp_11029[0], tmp_11027[1] + tmp_11029[1], tmp_11027[2] + tmp_11029[2]];
    signal tmp_11032[3] <== [63 * tmp_10525[0], 63 * tmp_10525[1], 63 * tmp_10525[2]];
    signal tmp_11034[3] <== [15 * tmp_10535[0], 15 * tmp_10535[1], 15 * tmp_10535[2]];
    signal tmp_11035[3] <== [tmp_11032[0] + tmp_11034[0], tmp_11032[1] + tmp_11034[1], tmp_11032[2] + tmp_11034[2]];
    signal tmp_11037[3] <== [2 * tmp_10546[0], 2 * tmp_10546[1], 2 * tmp_10546[2]];
    signal tmp_11038[3] <== [tmp_11035[0] + tmp_11037[0], tmp_11035[1] + tmp_11037[1], tmp_11035[2] + tmp_11037[2]];
    signal tmp_11040[3] <== [67 * tmp_10558[0], 67 * tmp_10558[1], 67 * tmp_10558[2]];
    signal tmp_11041[3] <== [tmp_11038[0] + tmp_11040[0], tmp_11038[1] + tmp_11040[1], tmp_11038[2] + tmp_11040[2]];
    signal tmp_11043[3] <== [22 * tmp_10569[0], 22 * tmp_10569[1], 22 * tmp_10569[2]];
    signal tmp_11044[3] <== [tmp_11041[0] + tmp_11043[0], tmp_11041[1] + tmp_11043[1], tmp_11041[2] + tmp_11043[2]];
    signal tmp_11046[3] <== [13 * tmp_10581[0], 13 * tmp_10581[1], 13 * tmp_10581[2]];
    signal tmp_11047[3] <== [tmp_11044[0] + tmp_11046[0], tmp_11044[1] + tmp_11046[1], tmp_11044[2] + tmp_11046[2]];
    signal tmp_11049[3] <== [3 * tmp_10593[0], 3 * tmp_10593[1], 3 * tmp_10593[2]];
    signal tmp_11050[3] <== [tmp_11047[0] + tmp_11049[0], tmp_11047[1] + tmp_11049[1], tmp_11047[2] + tmp_11049[2]];
    signal tmp_11052[3] <== [tmp_11050[0] + tmp_10605[0], tmp_11050[1] + tmp_10605[1], tmp_11050[2] + tmp_10605[2]];
    signal tmp_11054[3] <== [tmp_11052[0] + tmp_10616[0], tmp_11052[1] + tmp_10616[1], tmp_11052[2] + tmp_10616[2]];
    signal tmp_11056[3] <== [51 * tmp_10628[0], 51 * tmp_10628[1], 51 * tmp_10628[2]];
    signal tmp_11057[3] <== [tmp_11054[0] + tmp_11056[0], tmp_11054[1] + tmp_11056[1], tmp_11054[2] + tmp_11056[2]];
    signal tmp_11059[3] <== [tmp_11057[0] + tmp_10640[0], tmp_11057[1] + tmp_10640[1], tmp_11057[2] + tmp_10640[2]];
    signal tmp_11061[3] <== [11 * tmp_10652[0], 11 * tmp_10652[1], 11 * tmp_10652[2]];
    signal tmp_11062[3] <== [tmp_11059[0] + tmp_11061[0], tmp_11059[1] + tmp_11061[1], tmp_11059[2] + tmp_11061[2]];
    signal tmp_11064[3] <== [17 * tmp_10664[0], 17 * tmp_10664[1], 17 * tmp_10664[2]];
    signal tmp_11065[3] <== [tmp_11062[0] + tmp_11064[0], tmp_11062[1] + tmp_11064[1], tmp_11062[2] + tmp_11064[2]];
    signal tmp_11068[3] <== [tmp_11065[0] + tmp_10801[0], tmp_11065[1] + tmp_10801[1], tmp_11065[2] + tmp_10801[2]];
    signal tmp_11070[3] <== [tmp_11068[0] + tmp_10688[0], tmp_11068[1] + tmp_10688[1], tmp_11068[2] + tmp_10688[2]];
    signal tmp_11072[3] <== [101 * tmp_10700[0], 101 * tmp_10700[1], 101 * tmp_10700[2]];
    signal tmp_11073[3] <== [tmp_11070[0] + tmp_11072[0], tmp_11070[1] + tmp_11072[1], tmp_11070[2] + tmp_11072[2]];
    signal tmp_11074[3] <== [tmp_11030[0] - tmp_11073[0], tmp_11030[1] - tmp_11073[1], tmp_11030[2] - tmp_11073[2]];
    signal tmp_11075[3] <== CMul()(tmp_10511, tmp_11074);
    signal tmp_11076[3] <== [tmp_11024[0] + tmp_11075[0], tmp_11024[1] + tmp_11075[1], tmp_11024[2] + tmp_11075[2]];
    signal tmp_11077[3] <== CMul()(challengeQ, tmp_11076);
    signal tmp_11080[3] <== CMul()(evals[57], evals[75]);
    signal tmp_11082[3] <== CMul()(tmp_10513, evals[143]);
    signal tmp_11083[3] <== [tmp_11080[0] + tmp_11082[0], tmp_11080[1] + tmp_11082[1], tmp_11080[2] + tmp_11082[2]];
    signal tmp_11085[3] <== [101 * tmp_10525[0], 101 * tmp_10525[1], 101 * tmp_10525[2]];
    signal tmp_11087[3] <== [63 * tmp_10535[0], 63 * tmp_10535[1], 63 * tmp_10535[2]];
    signal tmp_11088[3] <== [tmp_11085[0] + tmp_11087[0], tmp_11085[1] + tmp_11087[1], tmp_11085[2] + tmp_11087[2]];
    signal tmp_11090[3] <== [15 * tmp_10546[0], 15 * tmp_10546[1], 15 * tmp_10546[2]];
    signal tmp_11091[3] <== [tmp_11088[0] + tmp_11090[0], tmp_11088[1] + tmp_11090[1], tmp_11088[2] + tmp_11090[2]];
    signal tmp_11093[3] <== [2 * tmp_10558[0], 2 * tmp_10558[1], 2 * tmp_10558[2]];
    signal tmp_11094[3] <== [tmp_11091[0] + tmp_11093[0], tmp_11091[1] + tmp_11093[1], tmp_11091[2] + tmp_11093[2]];
    signal tmp_11096[3] <== [67 * tmp_10569[0], 67 * tmp_10569[1], 67 * tmp_10569[2]];
    signal tmp_11097[3] <== [tmp_11094[0] + tmp_11096[0], tmp_11094[1] + tmp_11096[1], tmp_11094[2] + tmp_11096[2]];
    signal tmp_11099[3] <== [22 * tmp_10581[0], 22 * tmp_10581[1], 22 * tmp_10581[2]];
    signal tmp_11100[3] <== [tmp_11097[0] + tmp_11099[0], tmp_11097[1] + tmp_11099[1], tmp_11097[2] + tmp_11099[2]];
    signal tmp_11102[3] <== [13 * tmp_10593[0], 13 * tmp_10593[1], 13 * tmp_10593[2]];
    signal tmp_11103[3] <== [tmp_11100[0] + tmp_11102[0], tmp_11100[1] + tmp_11102[1], tmp_11100[2] + tmp_11102[2]];
    signal tmp_11105[3] <== [3 * tmp_10605[0], 3 * tmp_10605[1], 3 * tmp_10605[2]];
    signal tmp_11106[3] <== [tmp_11103[0] + tmp_11105[0], tmp_11103[1] + tmp_11105[1], tmp_11103[2] + tmp_11105[2]];
    signal tmp_11108[3] <== [tmp_11106[0] + tmp_10616[0], tmp_11106[1] + tmp_10616[1], tmp_11106[2] + tmp_10616[2]];
    signal tmp_11110[3] <== [tmp_11108[0] + tmp_10628[0], tmp_11108[1] + tmp_10628[1], tmp_11108[2] + tmp_10628[2]];
    signal tmp_11112[3] <== [51 * tmp_10640[0], 51 * tmp_10640[1], 51 * tmp_10640[2]];
    signal tmp_11113[3] <== [tmp_11110[0] + tmp_11112[0], tmp_11110[1] + tmp_11112[1], tmp_11110[2] + tmp_11112[2]];
    signal tmp_11115[3] <== [tmp_11113[0] + tmp_10652[0], tmp_11113[1] + tmp_10652[1], tmp_11113[2] + tmp_10652[2]];
    signal tmp_11117[3] <== [11 * tmp_10664[0], 11 * tmp_10664[1], 11 * tmp_10664[2]];
    signal tmp_11118[3] <== [tmp_11115[0] + tmp_11117[0], tmp_11115[1] + tmp_11117[1], tmp_11115[2] + tmp_11117[2]];
    signal tmp_11120[3] <== [17 * tmp_10676[0], 17 * tmp_10676[1], 17 * tmp_10676[2]];
    signal tmp_11121[3] <== [tmp_11118[0] + tmp_11120[0], tmp_11118[1] + tmp_11120[1], tmp_11118[2] + tmp_11120[2]];
    signal tmp_11124[3] <== [tmp_11121[0] + tmp_10857[0], tmp_11121[1] + tmp_10857[1], tmp_11121[2] + tmp_10857[2]];
    signal tmp_11126[3] <== [tmp_11124[0] + tmp_10700[0], tmp_11124[1] + tmp_10700[1], tmp_11124[2] + tmp_10700[2]];
    signal tmp_11127[3] <== [tmp_11083[0] - tmp_11126[0], tmp_11083[1] - tmp_11126[1], tmp_11083[2] - tmp_11126[2]];
    signal tmp_11128[3] <== CMul()(tmp_10511, tmp_11127);
    signal tmp_11129[3] <== [tmp_11077[0] + tmp_11128[0], tmp_11077[1] + tmp_11128[1], tmp_11077[2] + tmp_11128[2]];
    signal tmp_11130[3] <== CMul()(challengeQ, tmp_11129);
    signal tmp_11133[3] <== CMul()(evals[57], evals[76]);
    signal tmp_11135[3] <== CMul()(tmp_10513, evals[144]);
    signal tmp_11136[3] <== [tmp_11133[0] + tmp_11135[0], tmp_11133[1] + tmp_11135[1], tmp_11133[2] + tmp_11135[2]];
    signal tmp_11139[3] <== [101 * tmp_10535[0], 101 * tmp_10535[1], 101 * tmp_10535[2]];
    signal tmp_11140[3] <== [tmp_10525[0] + tmp_11139[0], tmp_10525[1] + tmp_11139[1], tmp_10525[2] + tmp_11139[2]];
    signal tmp_11142[3] <== [63 * tmp_10546[0], 63 * tmp_10546[1], 63 * tmp_10546[2]];
    signal tmp_11143[3] <== [tmp_11140[0] + tmp_11142[0], tmp_11140[1] + tmp_11142[1], tmp_11140[2] + tmp_11142[2]];
    signal tmp_11145[3] <== [15 * tmp_10558[0], 15 * tmp_10558[1], 15 * tmp_10558[2]];
    signal tmp_11146[3] <== [tmp_11143[0] + tmp_11145[0], tmp_11143[1] + tmp_11145[1], tmp_11143[2] + tmp_11145[2]];
    signal tmp_11148[3] <== [2 * tmp_10569[0], 2 * tmp_10569[1], 2 * tmp_10569[2]];
    signal tmp_11149[3] <== [tmp_11146[0] + tmp_11148[0], tmp_11146[1] + tmp_11148[1], tmp_11146[2] + tmp_11148[2]];
    signal tmp_11151[3] <== [67 * tmp_10581[0], 67 * tmp_10581[1], 67 * tmp_10581[2]];
    signal tmp_11152[3] <== [tmp_11149[0] + tmp_11151[0], tmp_11149[1] + tmp_11151[1], tmp_11149[2] + tmp_11151[2]];
    signal tmp_11154[3] <== [22 * tmp_10593[0], 22 * tmp_10593[1], 22 * tmp_10593[2]];
    signal tmp_11155[3] <== [tmp_11152[0] + tmp_11154[0], tmp_11152[1] + tmp_11154[1], tmp_11152[2] + tmp_11154[2]];
    signal tmp_11157[3] <== [13 * tmp_10605[0], 13 * tmp_10605[1], 13 * tmp_10605[2]];
    signal tmp_11158[3] <== [tmp_11155[0] + tmp_11157[0], tmp_11155[1] + tmp_11157[1], tmp_11155[2] + tmp_11157[2]];
    signal tmp_11160[3] <== [3 * tmp_10616[0], 3 * tmp_10616[1], 3 * tmp_10616[2]];
    signal tmp_11161[3] <== [tmp_11158[0] + tmp_11160[0], tmp_11158[1] + tmp_11160[1], tmp_11158[2] + tmp_11160[2]];
    signal tmp_11163[3] <== [tmp_11161[0] + tmp_10628[0], tmp_11161[1] + tmp_10628[1], tmp_11161[2] + tmp_10628[2]];
    signal tmp_11165[3] <== [tmp_11163[0] + tmp_10640[0], tmp_11163[1] + tmp_10640[1], tmp_11163[2] + tmp_10640[2]];
    signal tmp_11167[3] <== [51 * tmp_10652[0], 51 * tmp_10652[1], 51 * tmp_10652[2]];
    signal tmp_11168[3] <== [tmp_11165[0] + tmp_11167[0], tmp_11165[1] + tmp_11167[1], tmp_11165[2] + tmp_11167[2]];
    signal tmp_11170[3] <== [tmp_11168[0] + tmp_10664[0], tmp_11168[1] + tmp_10664[1], tmp_11168[2] + tmp_10664[2]];
    signal tmp_11172[3] <== [11 * tmp_10676[0], 11 * tmp_10676[1], 11 * tmp_10676[2]];
    signal tmp_11173[3] <== [tmp_11170[0] + tmp_11172[0], tmp_11170[1] + tmp_11172[1], tmp_11170[2] + tmp_11172[2]];
    signal tmp_11175[3] <== [17 * tmp_10688[0], 17 * tmp_10688[1], 17 * tmp_10688[2]];
    signal tmp_11176[3] <== [tmp_11173[0] + tmp_11175[0], tmp_11173[1] + tmp_11175[1], tmp_11173[2] + tmp_11175[2]];
    signal tmp_11179[3] <== [tmp_11176[0] + tmp_10913[0], tmp_11176[1] + tmp_10913[1], tmp_11176[2] + tmp_10913[2]];
    signal tmp_11180[3] <== [tmp_11136[0] - tmp_11179[0], tmp_11136[1] - tmp_11179[1], tmp_11136[2] - tmp_11179[2]];
    signal tmp_11181[3] <== CMul()(tmp_10511, tmp_11180);
    signal tmp_11182[3] <== [tmp_11130[0] + tmp_11181[0], tmp_11130[1] + tmp_11181[1], tmp_11130[2] + tmp_11181[2]];
    signal tmp_11183[3] <== CMul()(challengeQ, tmp_11182);
    signal tmp_11186[3] <== CMul()(evals[57], evals[77]);
    signal tmp_11188[3] <== CMul()(tmp_10513, evals[145]);
    signal tmp_11189[3] <== [tmp_11186[0] + tmp_11188[0], tmp_11186[1] + tmp_11188[1], tmp_11186[2] + tmp_11188[2]];
    signal tmp_11193[3] <== [tmp_10926[0] + tmp_10535[0], tmp_10926[1] + tmp_10535[1], tmp_10926[2] + tmp_10535[2]];
    signal tmp_11195[3] <== [101 * tmp_10546[0], 101 * tmp_10546[1], 101 * tmp_10546[2]];
    signal tmp_11196[3] <== [tmp_11193[0] + tmp_11195[0], tmp_11193[1] + tmp_11195[1], tmp_11193[2] + tmp_11195[2]];
    signal tmp_11198[3] <== [63 * tmp_10558[0], 63 * tmp_10558[1], 63 * tmp_10558[2]];
    signal tmp_11199[3] <== [tmp_11196[0] + tmp_11198[0], tmp_11196[1] + tmp_11198[1], tmp_11196[2] + tmp_11198[2]];
    signal tmp_11201[3] <== [15 * tmp_10569[0], 15 * tmp_10569[1], 15 * tmp_10569[2]];
    signal tmp_11202[3] <== [tmp_11199[0] + tmp_11201[0], tmp_11199[1] + tmp_11201[1], tmp_11199[2] + tmp_11201[2]];
    signal tmp_11204[3] <== [2 * tmp_10581[0], 2 * tmp_10581[1], 2 * tmp_10581[2]];
    signal tmp_11205[3] <== [tmp_11202[0] + tmp_11204[0], tmp_11202[1] + tmp_11204[1], tmp_11202[2] + tmp_11204[2]];
    signal tmp_11207[3] <== [67 * tmp_10593[0], 67 * tmp_10593[1], 67 * tmp_10593[2]];
    signal tmp_11208[3] <== [tmp_11205[0] + tmp_11207[0], tmp_11205[1] + tmp_11207[1], tmp_11205[2] + tmp_11207[2]];
    signal tmp_11210[3] <== [22 * tmp_10605[0], 22 * tmp_10605[1], 22 * tmp_10605[2]];
    signal tmp_11211[3] <== [tmp_11208[0] + tmp_11210[0], tmp_11208[1] + tmp_11210[1], tmp_11208[2] + tmp_11210[2]];
    signal tmp_11213[3] <== [13 * tmp_10616[0], 13 * tmp_10616[1], 13 * tmp_10616[2]];
    signal tmp_11214[3] <== [tmp_11211[0] + tmp_11213[0], tmp_11211[1] + tmp_11213[1], tmp_11211[2] + tmp_11213[2]];
    signal tmp_11216[3] <== [3 * tmp_10628[0], 3 * tmp_10628[1], 3 * tmp_10628[2]];
    signal tmp_11217[3] <== [tmp_11214[0] + tmp_11216[0], tmp_11214[1] + tmp_11216[1], tmp_11214[2] + tmp_11216[2]];
    signal tmp_11219[3] <== [tmp_11217[0] + tmp_10640[0], tmp_11217[1] + tmp_10640[1], tmp_11217[2] + tmp_10640[2]];
    signal tmp_11221[3] <== [tmp_11219[0] + tmp_10652[0], tmp_11219[1] + tmp_10652[1], tmp_11219[2] + tmp_10652[2]];
    signal tmp_11223[3] <== [51 * tmp_10664[0], 51 * tmp_10664[1], 51 * tmp_10664[2]];
    signal tmp_11224[3] <== [tmp_11221[0] + tmp_11223[0], tmp_11221[1] + tmp_11223[1], tmp_11221[2] + tmp_11223[2]];
    signal tmp_11226[3] <== [tmp_11224[0] + tmp_10676[0], tmp_11224[1] + tmp_10676[1], tmp_11224[2] + tmp_10676[2]];
    signal tmp_11228[3] <== [11 * tmp_10688[0], 11 * tmp_10688[1], 11 * tmp_10688[2]];
    signal tmp_11229[3] <== [tmp_11226[0] + tmp_11228[0], tmp_11226[1] + tmp_11228[1], tmp_11226[2] + tmp_11228[2]];
    signal tmp_11231[3] <== [17 * tmp_10700[0], 17 * tmp_10700[1], 17 * tmp_10700[2]];
    signal tmp_11232[3] <== [tmp_11229[0] + tmp_11231[0], tmp_11229[1] + tmp_11231[1], tmp_11229[2] + tmp_11231[2]];
    signal tmp_11233[3] <== [tmp_11189[0] - tmp_11232[0], tmp_11189[1] - tmp_11232[1], tmp_11189[2] - tmp_11232[2]];
    signal tmp_11234[3] <== CMul()(tmp_10511, tmp_11233);
    signal tmp_11235[3] <== [tmp_11183[0] + tmp_11234[0], tmp_11183[1] + tmp_11234[1], tmp_11183[2] + tmp_11234[2]];
    signal tmp_11236[3] <== CMul()(challengeQ, tmp_11235);
    signal tmp_11239[3] <== CMul()(evals[57], evals[78]);
    signal tmp_11241[3] <== CMul()(tmp_10513, evals[146]);
    signal tmp_11242[3] <== [tmp_11239[0] + tmp_11241[0], tmp_11239[1] + tmp_11241[1], tmp_11239[2] + tmp_11241[2]];
    signal tmp_11244[3] <== [17 * tmp_10525[0], 17 * tmp_10525[1], 17 * tmp_10525[2]];
    signal tmp_11247[3] <== [tmp_11244[0] + tmp_10981[0], tmp_11244[1] + tmp_10981[1], tmp_11244[2] + tmp_10981[2]];
    signal tmp_11249[3] <== [tmp_11247[0] + tmp_10546[0], tmp_11247[1] + tmp_10546[1], tmp_11247[2] + tmp_10546[2]];
    signal tmp_11251[3] <== [101 * tmp_10558[0], 101 * tmp_10558[1], 101 * tmp_10558[2]];
    signal tmp_11252[3] <== [tmp_11249[0] + tmp_11251[0], tmp_11249[1] + tmp_11251[1], tmp_11249[2] + tmp_11251[2]];
    signal tmp_11254[3] <== [63 * tmp_10569[0], 63 * tmp_10569[1], 63 * tmp_10569[2]];
    signal tmp_11255[3] <== [tmp_11252[0] + tmp_11254[0], tmp_11252[1] + tmp_11254[1], tmp_11252[2] + tmp_11254[2]];
    signal tmp_11257[3] <== [15 * tmp_10581[0], 15 * tmp_10581[1], 15 * tmp_10581[2]];
    signal tmp_11258[3] <== [tmp_11255[0] + tmp_11257[0], tmp_11255[1] + tmp_11257[1], tmp_11255[2] + tmp_11257[2]];
    signal tmp_11261[3] <== [tmp_11258[0] + tmp_10594[0], tmp_11258[1] + tmp_10594[1], tmp_11258[2] + tmp_10594[2]];
    signal tmp_11263[3] <== [67 * tmp_10605[0], 67 * tmp_10605[1], 67 * tmp_10605[2]];
    signal tmp_11264[3] <== [tmp_11261[0] + tmp_11263[0], tmp_11261[1] + tmp_11263[1], tmp_11261[2] + tmp_11263[2]];
    signal tmp_11266[3] <== [22 * tmp_10616[0], 22 * tmp_10616[1], 22 * tmp_10616[2]];
    signal tmp_11267[3] <== [tmp_11264[0] + tmp_11266[0], tmp_11264[1] + tmp_11266[1], tmp_11264[2] + tmp_11266[2]];
    signal tmp_11269[3] <== [13 * tmp_10628[0], 13 * tmp_10628[1], 13 * tmp_10628[2]];
    signal tmp_11270[3] <== [tmp_11267[0] + tmp_11269[0], tmp_11267[1] + tmp_11269[1], tmp_11267[2] + tmp_11269[2]];
    signal tmp_11272[3] <== [3 * tmp_10640[0], 3 * tmp_10640[1], 3 * tmp_10640[2]];
    signal tmp_11273[3] <== [tmp_11270[0] + tmp_11272[0], tmp_11270[1] + tmp_11272[1], tmp_11270[2] + tmp_11272[2]];
    signal tmp_11275[3] <== [tmp_11273[0] + tmp_10652[0], tmp_11273[1] + tmp_10652[1], tmp_11273[2] + tmp_10652[2]];
    signal tmp_11277[3] <== [tmp_11275[0] + tmp_10664[0], tmp_11275[1] + tmp_10664[1], tmp_11275[2] + tmp_10664[2]];
    signal tmp_11279[3] <== [51 * tmp_10676[0], 51 * tmp_10676[1], 51 * tmp_10676[2]];
    signal tmp_11280[3] <== [tmp_11277[0] + tmp_11279[0], tmp_11277[1] + tmp_11279[1], tmp_11277[2] + tmp_11279[2]];
    signal tmp_11282[3] <== [tmp_11280[0] + tmp_10688[0], tmp_11280[1] + tmp_10688[1], tmp_11280[2] + tmp_10688[2]];
    signal tmp_11284[3] <== [11 * tmp_10700[0], 11 * tmp_10700[1], 11 * tmp_10700[2]];
    signal tmp_11285[3] <== [tmp_11282[0] + tmp_11284[0], tmp_11282[1] + tmp_11284[1], tmp_11282[2] + tmp_11284[2]];
    signal tmp_11286[3] <== [tmp_11242[0] - tmp_11285[0], tmp_11242[1] - tmp_11285[1], tmp_11242[2] - tmp_11285[2]];
    signal tmp_11287[3] <== CMul()(tmp_10511, tmp_11286);
    signal tmp_11288[3] <== [tmp_11236[0] + tmp_11287[0], tmp_11236[1] + tmp_11287[1], tmp_11236[2] + tmp_11287[2]];
    signal tmp_11289[3] <== CMul()(challengeQ, tmp_11288);
    signal tmp_11292[3] <== CMul()(evals[57], evals[79]);
    signal tmp_11294[3] <== CMul()(tmp_10513, evals[147]);
    signal tmp_11295[3] <== [tmp_11292[0] + tmp_11294[0], tmp_11292[1] + tmp_11294[1], tmp_11292[2] + tmp_11294[2]];
    signal tmp_11297[3] <== [11 * tmp_10525[0], 11 * tmp_10525[1], 11 * tmp_10525[2]];
    signal tmp_11299[3] <== [17 * tmp_10535[0], 17 * tmp_10535[1], 17 * tmp_10535[2]];
    signal tmp_11300[3] <== [tmp_11297[0] + tmp_11299[0], tmp_11297[1] + tmp_11299[1], tmp_11297[2] + tmp_11299[2]];
    signal tmp_11303[3] <== [tmp_11300[0] + tmp_11037[0], tmp_11300[1] + tmp_11037[1], tmp_11300[2] + tmp_11037[2]];
    signal tmp_11305[3] <== [tmp_11303[0] + tmp_10558[0], tmp_11303[1] + tmp_10558[1], tmp_11303[2] + tmp_10558[2]];
    signal tmp_11307[3] <== [101 * tmp_10569[0], 101 * tmp_10569[1], 101 * tmp_10569[2]];
    signal tmp_11308[3] <== [tmp_11305[0] + tmp_11307[0], tmp_11305[1] + tmp_11307[1], tmp_11305[2] + tmp_11307[2]];
    signal tmp_11310[3] <== [63 * tmp_10581[0], 63 * tmp_10581[1], 63 * tmp_10581[2]];
    signal tmp_11311[3] <== [tmp_11308[0] + tmp_11310[0], tmp_11308[1] + tmp_11310[1], tmp_11308[2] + tmp_11310[2]];
    signal tmp_11313[3] <== [15 * tmp_10593[0], 15 * tmp_10593[1], 15 * tmp_10593[2]];
    signal tmp_11314[3] <== [tmp_11311[0] + tmp_11313[0], tmp_11311[1] + tmp_11313[1], tmp_11311[2] + tmp_11313[2]];
    signal tmp_11317[3] <== [tmp_11314[0] + tmp_10731[0], tmp_11314[1] + tmp_10731[1], tmp_11314[2] + tmp_10731[2]];
    signal tmp_11319[3] <== [67 * tmp_10616[0], 67 * tmp_10616[1], 67 * tmp_10616[2]];
    signal tmp_11320[3] <== [tmp_11317[0] + tmp_11319[0], tmp_11317[1] + tmp_11319[1], tmp_11317[2] + tmp_11319[2]];
    signal tmp_11322[3] <== [22 * tmp_10628[0], 22 * tmp_10628[1], 22 * tmp_10628[2]];
    signal tmp_11323[3] <== [tmp_11320[0] + tmp_11322[0], tmp_11320[1] + tmp_11322[1], tmp_11320[2] + tmp_11322[2]];
    signal tmp_11325[3] <== [13 * tmp_10640[0], 13 * tmp_10640[1], 13 * tmp_10640[2]];
    signal tmp_11326[3] <== [tmp_11323[0] + tmp_11325[0], tmp_11323[1] + tmp_11325[1], tmp_11323[2] + tmp_11325[2]];
    signal tmp_11328[3] <== [3 * tmp_10652[0], 3 * tmp_10652[1], 3 * tmp_10652[2]];
    signal tmp_11329[3] <== [tmp_11326[0] + tmp_11328[0], tmp_11326[1] + tmp_11328[1], tmp_11326[2] + tmp_11328[2]];
    signal tmp_11331[3] <== [tmp_11329[0] + tmp_10664[0], tmp_11329[1] + tmp_10664[1], tmp_11329[2] + tmp_10664[2]];
    signal tmp_11333[3] <== [tmp_11331[0] + tmp_10676[0], tmp_11331[1] + tmp_10676[1], tmp_11331[2] + tmp_10676[2]];
    signal tmp_11335[3] <== [51 * tmp_10688[0], 51 * tmp_10688[1], 51 * tmp_10688[2]];
    signal tmp_11336[3] <== [tmp_11333[0] + tmp_11335[0], tmp_11333[1] + tmp_11335[1], tmp_11333[2] + tmp_11335[2]];
    signal tmp_11338[3] <== [tmp_11336[0] + tmp_10700[0], tmp_11336[1] + tmp_10700[1], tmp_11336[2] + tmp_10700[2]];
    signal tmp_11339[3] <== [tmp_11295[0] - tmp_11338[0], tmp_11295[1] - tmp_11338[1], tmp_11295[2] - tmp_11338[2]];
    signal tmp_11340[3] <== CMul()(tmp_10511, tmp_11339);
    signal tmp_11341[3] <== [tmp_11289[0] + tmp_11340[0], tmp_11289[1] + tmp_11340[1], tmp_11289[2] + tmp_11340[2]];
    signal tmp_11342[3] <== CMul()(challengeQ, tmp_11341);
    signal tmp_11345[3] <== CMul()(evals[57], evals[80]);
    signal tmp_11347[3] <== CMul()(tmp_10513, evals[148]);
    signal tmp_11348[3] <== [tmp_11345[0] + tmp_11347[0], tmp_11345[1] + tmp_11347[1], tmp_11345[2] + tmp_11347[2]];
    signal tmp_11351[3] <== [11 * tmp_10535[0], 11 * tmp_10535[1], 11 * tmp_10535[2]];
    signal tmp_11352[3] <== [tmp_10525[0] + tmp_11351[0], tmp_10525[1] + tmp_11351[1], tmp_10525[2] + tmp_11351[2]];
    signal tmp_11354[3] <== [17 * tmp_10546[0], 17 * tmp_10546[1], 17 * tmp_10546[2]];
    signal tmp_11355[3] <== [tmp_11352[0] + tmp_11354[0], tmp_11352[1] + tmp_11354[1], tmp_11352[2] + tmp_11354[2]];
    signal tmp_11358[3] <== [tmp_11355[0] + tmp_11093[0], tmp_11355[1] + tmp_11093[1], tmp_11355[2] + tmp_11093[2]];
    signal tmp_11360[3] <== [tmp_11358[0] + tmp_10569[0], tmp_11358[1] + tmp_10569[1], tmp_11358[2] + tmp_10569[2]];
    signal tmp_11362[3] <== [101 * tmp_10581[0], 101 * tmp_10581[1], 101 * tmp_10581[2]];
    signal tmp_11363[3] <== [tmp_11360[0] + tmp_11362[0], tmp_11360[1] + tmp_11362[1], tmp_11360[2] + tmp_11362[2]];
    signal tmp_11365[3] <== [63 * tmp_10593[0], 63 * tmp_10593[1], 63 * tmp_10593[2]];
    signal tmp_11366[3] <== [tmp_11363[0] + tmp_11365[0], tmp_11363[1] + tmp_11365[1], tmp_11363[2] + tmp_11365[2]];
    signal tmp_11368[3] <== [15 * tmp_10605[0], 15 * tmp_10605[1], 15 * tmp_10605[2]];
    signal tmp_11369[3] <== [tmp_11366[0] + tmp_11368[0], tmp_11366[1] + tmp_11368[1], tmp_11366[2] + tmp_11368[2]];
    signal tmp_11372[3] <== [tmp_11369[0] + tmp_10787[0], tmp_11369[1] + tmp_10787[1], tmp_11369[2] + tmp_10787[2]];
    signal tmp_11374[3] <== [67 * tmp_10628[0], 67 * tmp_10628[1], 67 * tmp_10628[2]];
    signal tmp_11375[3] <== [tmp_11372[0] + tmp_11374[0], tmp_11372[1] + tmp_11374[1], tmp_11372[2] + tmp_11374[2]];
    signal tmp_11377[3] <== [22 * tmp_10640[0], 22 * tmp_10640[1], 22 * tmp_10640[2]];
    signal tmp_11378[3] <== [tmp_11375[0] + tmp_11377[0], tmp_11375[1] + tmp_11377[1], tmp_11375[2] + tmp_11377[2]];
    signal tmp_11380[3] <== [13 * tmp_10652[0], 13 * tmp_10652[1], 13 * tmp_10652[2]];
    signal tmp_11381[3] <== [tmp_11378[0] + tmp_11380[0], tmp_11378[1] + tmp_11380[1], tmp_11378[2] + tmp_11380[2]];
    signal tmp_11383[3] <== [3 * tmp_10664[0], 3 * tmp_10664[1], 3 * tmp_10664[2]];
    signal tmp_11384[3] <== [tmp_11381[0] + tmp_11383[0], tmp_11381[1] + tmp_11383[1], tmp_11381[2] + tmp_11383[2]];
    signal tmp_11386[3] <== [tmp_11384[0] + tmp_10676[0], tmp_11384[1] + tmp_10676[1], tmp_11384[2] + tmp_10676[2]];
    signal tmp_11388[3] <== [tmp_11386[0] + tmp_10688[0], tmp_11386[1] + tmp_10688[1], tmp_11386[2] + tmp_10688[2]];
    signal tmp_11390[3] <== [51 * tmp_10700[0], 51 * tmp_10700[1], 51 * tmp_10700[2]];
    signal tmp_11391[3] <== [tmp_11388[0] + tmp_11390[0], tmp_11388[1] + tmp_11390[1], tmp_11388[2] + tmp_11390[2]];
    signal tmp_11392[3] <== [tmp_11348[0] - tmp_11391[0], tmp_11348[1] - tmp_11391[1], tmp_11348[2] - tmp_11391[2]];
    signal tmp_11393[3] <== CMul()(tmp_10511, tmp_11392);
    signal tmp_11394[3] <== [tmp_11342[0] + tmp_11393[0], tmp_11342[1] + tmp_11393[1], tmp_11342[2] + tmp_11393[2]];
    signal tmp_11395[3] <== CMul()(challengeQ, tmp_11394);
    signal tmp_11398[3] <== CMul()(evals[57], evals[81]);
    signal tmp_11400[3] <== CMul()(tmp_10513, evals[149]);
    signal tmp_11401[3] <== [tmp_11398[0] + tmp_11400[0], tmp_11398[1] + tmp_11400[1], tmp_11398[2] + tmp_11400[2]];
    signal tmp_11403[3] <== [51 * tmp_10525[0], 51 * tmp_10525[1], 51 * tmp_10525[2]];
    signal tmp_11405[3] <== [tmp_11403[0] + tmp_10535[0], tmp_11403[1] + tmp_10535[1], tmp_11403[2] + tmp_10535[2]];
    signal tmp_11407[3] <== [11 * tmp_10546[0], 11 * tmp_10546[1], 11 * tmp_10546[2]];
    signal tmp_11408[3] <== [tmp_11405[0] + tmp_11407[0], tmp_11405[1] + tmp_11407[1], tmp_11405[2] + tmp_11407[2]];
    signal tmp_11410[3] <== [17 * tmp_10558[0], 17 * tmp_10558[1], 17 * tmp_10558[2]];
    signal tmp_11411[3] <== [tmp_11408[0] + tmp_11410[0], tmp_11408[1] + tmp_11410[1], tmp_11408[2] + tmp_11410[2]];
    signal tmp_11414[3] <== [tmp_11411[0] + tmp_11148[0], tmp_11411[1] + tmp_11148[1], tmp_11411[2] + tmp_11148[2]];
    signal tmp_11416[3] <== [tmp_11414[0] + tmp_10581[0], tmp_11414[1] + tmp_10581[1], tmp_11414[2] + tmp_10581[2]];
    signal tmp_11418[3] <== [101 * tmp_10593[0], 101 * tmp_10593[1], 101 * tmp_10593[2]];
    signal tmp_11419[3] <== [tmp_11416[0] + tmp_11418[0], tmp_11416[1] + tmp_11418[1], tmp_11416[2] + tmp_11418[2]];
    signal tmp_11421[3] <== [63 * tmp_10605[0], 63 * tmp_10605[1], 63 * tmp_10605[2]];
    signal tmp_11422[3] <== [tmp_11419[0] + tmp_11421[0], tmp_11419[1] + tmp_11421[1], tmp_11419[2] + tmp_11421[2]];
    signal tmp_11424[3] <== [15 * tmp_10616[0], 15 * tmp_10616[1], 15 * tmp_10616[2]];
    signal tmp_11425[3] <== [tmp_11422[0] + tmp_11424[0], tmp_11422[1] + tmp_11424[1], tmp_11422[2] + tmp_11424[2]];
    signal tmp_11428[3] <== [tmp_11425[0] + tmp_10843[0], tmp_11425[1] + tmp_10843[1], tmp_11425[2] + tmp_10843[2]];
    signal tmp_11430[3] <== [67 * tmp_10640[0], 67 * tmp_10640[1], 67 * tmp_10640[2]];
    signal tmp_11431[3] <== [tmp_11428[0] + tmp_11430[0], tmp_11428[1] + tmp_11430[1], tmp_11428[2] + tmp_11430[2]];
    signal tmp_11433[3] <== [22 * tmp_10652[0], 22 * tmp_10652[1], 22 * tmp_10652[2]];
    signal tmp_11434[3] <== [tmp_11431[0] + tmp_11433[0], tmp_11431[1] + tmp_11433[1], tmp_11431[2] + tmp_11433[2]];
    signal tmp_11436[3] <== [13 * tmp_10664[0], 13 * tmp_10664[1], 13 * tmp_10664[2]];
    signal tmp_11437[3] <== [tmp_11434[0] + tmp_11436[0], tmp_11434[1] + tmp_11436[1], tmp_11434[2] + tmp_11436[2]];
    signal tmp_11439[3] <== [3 * tmp_10676[0], 3 * tmp_10676[1], 3 * tmp_10676[2]];
    signal tmp_11440[3] <== [tmp_11437[0] + tmp_11439[0], tmp_11437[1] + tmp_11439[1], tmp_11437[2] + tmp_11439[2]];
    signal tmp_11442[3] <== [tmp_11440[0] + tmp_10688[0], tmp_11440[1] + tmp_10688[1], tmp_11440[2] + tmp_10688[2]];
    signal tmp_11444[3] <== [tmp_11442[0] + tmp_10700[0], tmp_11442[1] + tmp_10700[1], tmp_11442[2] + tmp_10700[2]];
    signal tmp_11445[3] <== [tmp_11401[0] - tmp_11444[0], tmp_11401[1] - tmp_11444[1], tmp_11401[2] - tmp_11444[2]];
    signal tmp_11446[3] <== CMul()(tmp_10511, tmp_11445);
    signal tmp_11447[3] <== [tmp_11395[0] + tmp_11446[0], tmp_11395[1] + tmp_11446[1], tmp_11395[2] + tmp_11446[2]];
    signal tmp_11448[3] <== CMul()(challengeQ, tmp_11447);
    signal tmp_11451[3] <== CMul()(evals[57], evals[82]);
    signal tmp_11453[3] <== CMul()(tmp_10513, evals[150]);
    signal tmp_11454[3] <== [tmp_11451[0] + tmp_11453[0], tmp_11451[1] + tmp_11453[1], tmp_11451[2] + tmp_11453[2]];
    signal tmp_11457[3] <== [51 * tmp_10535[0], 51 * tmp_10535[1], 51 * tmp_10535[2]];
    signal tmp_11458[3] <== [tmp_10525[0] + tmp_11457[0], tmp_10525[1] + tmp_11457[1], tmp_10525[2] + tmp_11457[2]];
    signal tmp_11460[3] <== [tmp_11458[0] + tmp_10546[0], tmp_11458[1] + tmp_10546[1], tmp_11458[2] + tmp_10546[2]];
    signal tmp_11462[3] <== [11 * tmp_10558[0], 11 * tmp_10558[1], 11 * tmp_10558[2]];
    signal tmp_11463[3] <== [tmp_11460[0] + tmp_11462[0], tmp_11460[1] + tmp_11462[1], tmp_11460[2] + tmp_11462[2]];
    signal tmp_11465[3] <== [17 * tmp_10569[0], 17 * tmp_10569[1], 17 * tmp_10569[2]];
    signal tmp_11466[3] <== [tmp_11463[0] + tmp_11465[0], tmp_11463[1] + tmp_11465[1], tmp_11463[2] + tmp_11465[2]];
    signal tmp_11469[3] <== [tmp_11466[0] + tmp_11204[0], tmp_11466[1] + tmp_11204[1], tmp_11466[2] + tmp_11204[2]];
    signal tmp_11471[3] <== [tmp_11469[0] + tmp_10593[0], tmp_11469[1] + tmp_10593[1], tmp_11469[2] + tmp_10593[2]];
    signal tmp_11473[3] <== [101 * tmp_10605[0], 101 * tmp_10605[1], 101 * tmp_10605[2]];
    signal tmp_11474[3] <== [tmp_11471[0] + tmp_11473[0], tmp_11471[1] + tmp_11473[1], tmp_11471[2] + tmp_11473[2]];
    signal tmp_11476[3] <== [63 * tmp_10616[0], 63 * tmp_10616[1], 63 * tmp_10616[2]];
    signal tmp_11477[3] <== [tmp_11474[0] + tmp_11476[0], tmp_11474[1] + tmp_11476[1], tmp_11474[2] + tmp_11476[2]];
    signal tmp_11479[3] <== [15 * tmp_10628[0], 15 * tmp_10628[1], 15 * tmp_10628[2]];
    signal tmp_11480[3] <== [tmp_11477[0] + tmp_11479[0], tmp_11477[1] + tmp_11479[1], tmp_11477[2] + tmp_11479[2]];
    signal tmp_11483[3] <== [tmp_11480[0] + tmp_10899[0], tmp_11480[1] + tmp_10899[1], tmp_11480[2] + tmp_10899[2]];
    signal tmp_11485[3] <== [67 * tmp_10652[0], 67 * tmp_10652[1], 67 * tmp_10652[2]];
    signal tmp_11486[3] <== [tmp_11483[0] + tmp_11485[0], tmp_11483[1] + tmp_11485[1], tmp_11483[2] + tmp_11485[2]];
    signal tmp_11488[3] <== [22 * tmp_10664[0], 22 * tmp_10664[1], 22 * tmp_10664[2]];
    signal tmp_11489[3] <== [tmp_11486[0] + tmp_11488[0], tmp_11486[1] + tmp_11488[1], tmp_11486[2] + tmp_11488[2]];
    signal tmp_11491[3] <== [13 * tmp_10676[0], 13 * tmp_10676[1], 13 * tmp_10676[2]];
    signal tmp_11492[3] <== [tmp_11489[0] + tmp_11491[0], tmp_11489[1] + tmp_11491[1], tmp_11489[2] + tmp_11491[2]];
    signal tmp_11494[3] <== [3 * tmp_10688[0], 3 * tmp_10688[1], 3 * tmp_10688[2]];
    signal tmp_11495[3] <== [tmp_11492[0] + tmp_11494[0], tmp_11492[1] + tmp_11494[1], tmp_11492[2] + tmp_11494[2]];
    signal tmp_11497[3] <== [tmp_11495[0] + tmp_10700[0], tmp_11495[1] + tmp_10700[1], tmp_11495[2] + tmp_10700[2]];
    signal tmp_11498[3] <== [tmp_11454[0] - tmp_11497[0], tmp_11454[1] - tmp_11497[1], tmp_11454[2] - tmp_11497[2]];
    signal tmp_11499[3] <== CMul()(tmp_10511, tmp_11498);
    signal tmp_11500[3] <== [tmp_11448[0] + tmp_11499[0], tmp_11448[1] + tmp_11499[1], tmp_11448[2] + tmp_11499[2]];
    signal tmp_11501[3] <== CMul()(challengeQ, tmp_11500);
    signal tmp_11545[3] <== [evals[135][0] - tmp_10702[0], evals[135][1] - tmp_10702[1], evals[135][2] - tmp_10702[2]];
    signal tmp_11546[3] <== CMul()(tmp_9161, tmp_11545);
    signal tmp_11547[3] <== [tmp_11501[0] + tmp_11546[0], tmp_11501[1] + tmp_11546[1], tmp_11501[2] + tmp_11546[2]];
    signal tmp_11548[3] <== CMul()(challengeQ, tmp_11547);
    signal tmp_11550[3] <== [2256277865552202420 * tmp_10525[0], 2256277865552202420 * tmp_10525[1], 2256277865552202420 * tmp_10525[2]];
    signal tmp_11552[3] <== [4068529056800825848 * tmp_10535[0], 4068529056800825848 * tmp_10535[1], 4068529056800825848 * tmp_10535[2]];
    signal tmp_11553[3] <== [tmp_11550[0] + tmp_11552[0], tmp_11550[1] + tmp_11552[1], tmp_11550[2] + tmp_11552[2]];
    signal tmp_11555[3] <== [8593802027805519400 * tmp_10546[0], 8593802027805519400 * tmp_10546[1], 8593802027805519400 * tmp_10546[2]];
    signal tmp_11556[3] <== [tmp_11553[0] + tmp_11555[0], tmp_11553[1] + tmp_11555[1], tmp_11553[2] + tmp_11555[2]];
    signal tmp_11558[3] <== [10290793996632644659 * tmp_10558[0], 10290793996632644659 * tmp_10558[1], 10290793996632644659 * tmp_10558[2]];
    signal tmp_11559[3] <== [tmp_11556[0] + tmp_11558[0], tmp_11556[1] + tmp_11558[1], tmp_11556[2] + tmp_11558[2]];
    signal tmp_11561[3] <== [18408474278105675815 * tmp_10569[0], 18408474278105675815 * tmp_10569[1], 18408474278105675815 * tmp_10569[2]];
    signal tmp_11562[3] <== [tmp_11559[0] + tmp_11561[0], tmp_11559[1] + tmp_11561[1], tmp_11559[2] + tmp_11561[2]];
    signal tmp_11564[3] <== [17884228943875298379 * tmp_10581[0], 17884228943875298379 * tmp_10581[1], 17884228943875298379 * tmp_10581[2]];
    signal tmp_11565[3] <== [tmp_11562[0] + tmp_11564[0], tmp_11562[1] + tmp_11564[1], tmp_11562[2] + tmp_11564[2]];
    signal tmp_11567[3] <== [6702954500202032351 * tmp_10593[0], 6702954500202032351 * tmp_10593[1], 6702954500202032351 * tmp_10593[2]];
    signal tmp_11568[3] <== [tmp_11565[0] + tmp_11567[0], tmp_11565[1] + tmp_11567[1], tmp_11565[2] + tmp_11567[2]];
    signal tmp_11570[3] <== [11180511074533951034 * tmp_10605[0], 11180511074533951034 * tmp_10605[1], 11180511074533951034 * tmp_10605[2]];
    signal tmp_11571[3] <== [tmp_11568[0] + tmp_11570[0], tmp_11568[1] + tmp_11570[1], tmp_11568[2] + tmp_11570[2]];
    signal tmp_11573[3] <== [6468404447315445157 * tmp_10616[0], 6468404447315445157 * tmp_10616[1], 6468404447315445157 * tmp_10616[2]];
    signal tmp_11574[3] <== [tmp_11571[0] + tmp_11573[0], tmp_11571[1] + tmp_11573[1], tmp_11571[2] + tmp_11573[2]];
    signal tmp_11576[3] <== [12266684712140129864 * tmp_10628[0], 12266684712140129864 * tmp_10628[1], 12266684712140129864 * tmp_10628[2]];
    signal tmp_11577[3] <== [tmp_11574[0] + tmp_11576[0], tmp_11574[1] + tmp_11576[1], tmp_11574[2] + tmp_11576[2]];
    signal tmp_11579[3] <== [6782251596454435859 * tmp_10640[0], 6782251596454435859 * tmp_10640[1], 6782251596454435859 * tmp_10640[2]];
    signal tmp_11580[3] <== [tmp_11577[0] + tmp_11579[0], tmp_11577[1] + tmp_11579[1], tmp_11577[2] + tmp_11579[2]];
    signal tmp_11582[3] <== [2765050638571938600 * tmp_10652[0], 2765050638571938600 * tmp_10652[1], 2765050638571938600 * tmp_10652[2]];
    signal tmp_11583[3] <== [tmp_11580[0] + tmp_11582[0], tmp_11580[1] + tmp_11582[1], tmp_11580[2] + tmp_11582[2]];
    signal tmp_11585[3] <== [17337375643812333090 * tmp_10664[0], 17337375643812333090 * tmp_10664[1], 17337375643812333090 * tmp_10664[2]];
    signal tmp_11586[3] <== [tmp_11583[0] + tmp_11585[0], tmp_11583[1] + tmp_11585[1], tmp_11583[2] + tmp_11585[2]];
    signal tmp_11588[3] <== [13578549160148989543 * tmp_10676[0], 13578549160148989543 * tmp_10676[1], 13578549160148989543 * tmp_10676[2]];
    signal tmp_11589[3] <== [tmp_11586[0] + tmp_11588[0], tmp_11586[1] + tmp_11588[1], tmp_11586[2] + tmp_11588[2]];
    signal tmp_11591[3] <== [4068336395193158675 * tmp_10688[0], 4068336395193158675 * tmp_10688[1], 4068336395193158675 * tmp_10688[2]];
    signal tmp_11592[3] <== [tmp_11589[0] + tmp_11591[0], tmp_11589[1] + tmp_11591[1], tmp_11589[2] + tmp_11591[2]];
    signal tmp_11594[3] <== [1571908583333476476 * tmp_10700[0], 1571908583333476476 * tmp_10700[1], 1571908583333476476 * tmp_10700[2]];
    signal tmp_11595[3] <== [tmp_11592[0] + tmp_11594[0], tmp_11592[1] + tmp_11594[1], tmp_11592[2] + tmp_11594[2]];
    signal tmp_11596[3] <== [evals[136][0] - tmp_11595[0], evals[136][1] - tmp_11595[1], evals[136][2] - tmp_11595[2]];
    signal tmp_11597[3] <== CMul()(tmp_9161, tmp_11596);
    signal tmp_11598[3] <== [tmp_11548[0] + tmp_11597[0], tmp_11548[1] + tmp_11597[1], tmp_11548[2] + tmp_11597[2]];
    signal tmp_11599[3] <== CMul()(challengeQ, tmp_11598);
    signal tmp_11601[3] <== [17717269319234059614 * tmp_10525[0], 17717269319234059614 * tmp_10525[1], 17717269319234059614 * tmp_10525[2]];
    signal tmp_11603[3] <== [17521923232000336525 * tmp_10535[0], 17521923232000336525 * tmp_10535[1], 17521923232000336525 * tmp_10535[2]];
    signal tmp_11604[3] <== [tmp_11601[0] + tmp_11603[0], tmp_11601[1] + tmp_11603[1], tmp_11601[2] + tmp_11603[2]];
    signal tmp_11606[3] <== [7891351762273827007 * tmp_10546[0], 7891351762273827007 * tmp_10546[1], 7891351762273827007 * tmp_10546[2]];
    signal tmp_11607[3] <== [tmp_11604[0] + tmp_11606[0], tmp_11604[1] + tmp_11606[1], tmp_11604[2] + tmp_11606[2]];
    signal tmp_11609[3] <== [14226032251042260572 * tmp_10558[0], 14226032251042260572 * tmp_10558[1], 14226032251042260572 * tmp_10558[2]];
    signal tmp_11610[3] <== [tmp_11607[0] + tmp_11609[0], tmp_11607[1] + tmp_11609[1], tmp_11607[2] + tmp_11609[2]];
    signal tmp_11612[3] <== [17611042948119270535 * tmp_10569[0], 17611042948119270535 * tmp_10569[1], 17611042948119270535 * tmp_10569[2]];
    signal tmp_11613[3] <== [tmp_11610[0] + tmp_11612[0], tmp_11610[1] + tmp_11612[1], tmp_11610[2] + tmp_11612[2]];
    signal tmp_11615[3] <== [1810923409258252859 * tmp_10581[0], 1810923409258252859 * tmp_10581[1], 1810923409258252859 * tmp_10581[2]];
    signal tmp_11616[3] <== [tmp_11613[0] + tmp_11615[0], tmp_11613[1] + tmp_11615[1], tmp_11613[2] + tmp_11615[2]];
    signal tmp_11618[3] <== [6551497314953504346 * tmp_10593[0], 6551497314953504346 * tmp_10593[1], 6551497314953504346 * tmp_10593[2]];
    signal tmp_11619[3] <== [tmp_11616[0] + tmp_11618[0], tmp_11616[1] + tmp_11618[1], tmp_11616[2] + tmp_11618[2]];
    signal tmp_11621[3] <== [14645553078015703863 * tmp_10605[0], 14645553078015703863 * tmp_10605[1], 14645553078015703863 * tmp_10605[2]];
    signal tmp_11622[3] <== [tmp_11619[0] + tmp_11621[0], tmp_11619[1] + tmp_11621[1], tmp_11619[2] + tmp_11621[2]];
    signal tmp_11624[3] <== [17162061841860881922 * tmp_10616[0], 17162061841860881922 * tmp_10616[1], 17162061841860881922 * tmp_10616[2]];
    signal tmp_11625[3] <== [tmp_11622[0] + tmp_11624[0], tmp_11622[1] + tmp_11624[1], tmp_11622[2] + tmp_11624[2]];
    signal tmp_11627[3] <== [16803151265961688719 * tmp_10628[0], 16803151265961688719 * tmp_10628[1], 16803151265961688719 * tmp_10628[2]];
    signal tmp_11628[3] <== [tmp_11625[0] + tmp_11627[0], tmp_11625[1] + tmp_11627[1], tmp_11625[2] + tmp_11627[2]];
    signal tmp_11630[3] <== [1725490462187460494 * tmp_10640[0], 1725490462187460494 * tmp_10640[1], 1725490462187460494 * tmp_10640[2]];
    signal tmp_11631[3] <== [tmp_11628[0] + tmp_11630[0], tmp_11628[1] + tmp_11630[1], tmp_11628[2] + tmp_11630[2]];
    signal tmp_11633[3] <== [9867063047527519896 * tmp_10652[0], 9867063047527519896 * tmp_10652[1], 9867063047527519896 * tmp_10652[2]];
    signal tmp_11634[3] <== [tmp_11631[0] + tmp_11633[0], tmp_11631[1] + tmp_11633[1], tmp_11631[2] + tmp_11633[2]];
    signal tmp_11636[3] <== [14201875104048534679 * tmp_10664[0], 14201875104048534679 * tmp_10664[1], 14201875104048534679 * tmp_10664[2]];
    signal tmp_11637[3] <== [tmp_11634[0] + tmp_11636[0], tmp_11634[1] + tmp_11636[1], tmp_11634[2] + tmp_11636[2]];
    signal tmp_11639[3] <== [218778007896712464 * tmp_10676[0], 218778007896712464 * tmp_10676[1], 218778007896712464 * tmp_10676[2]];
    signal tmp_11640[3] <== [tmp_11637[0] + tmp_11639[0], tmp_11637[1] + tmp_11639[1], tmp_11637[2] + tmp_11639[2]];
    signal tmp_11642[3] <== [12636278992549723093 * tmp_10688[0], 12636278992549723093 * tmp_10688[1], 12636278992549723093 * tmp_10688[2]];
    signal tmp_11643[3] <== [tmp_11640[0] + tmp_11642[0], tmp_11640[1] + tmp_11642[1], tmp_11640[2] + tmp_11642[2]];
    signal tmp_11645[3] <== [4068336395193158675 * tmp_10700[0], 4068336395193158675 * tmp_10700[1], 4068336395193158675 * tmp_10700[2]];
    signal tmp_11646[3] <== [tmp_11643[0] + tmp_11645[0], tmp_11643[1] + tmp_11645[1], tmp_11643[2] + tmp_11645[2]];
    signal tmp_11647[3] <== [evals[137][0] - tmp_11646[0], evals[137][1] - tmp_11646[1], evals[137][2] - tmp_11646[2]];
    signal tmp_11648[3] <== CMul()(tmp_9161, tmp_11647);
    signal tmp_11649[3] <== [tmp_11599[0] + tmp_11648[0], tmp_11599[1] + tmp_11648[1], tmp_11599[2] + tmp_11648[2]];
    signal tmp_11650[3] <== CMul()(challengeQ, tmp_11649);
    signal tmp_11652[3] <== [12035225151448622303 * tmp_10525[0], 12035225151448622303 * tmp_10525[1], 12035225151448622303 * tmp_10525[2]];
    signal tmp_11654[3] <== [8855639374359979570 * tmp_10535[0], 8855639374359979570 * tmp_10535[1], 8855639374359979570 * tmp_10535[2]];
    signal tmp_11655[3] <== [tmp_11652[0] + tmp_11654[0], tmp_11652[1] + tmp_11654[1], tmp_11652[2] + tmp_11654[2]];
    signal tmp_11657[3] <== [10872182090012722023 * tmp_10546[0], 10872182090012722023 * tmp_10546[1], 10872182090012722023 * tmp_10546[2]];
    signal tmp_11658[3] <== [tmp_11655[0] + tmp_11657[0], tmp_11655[1] + tmp_11657[1], tmp_11655[2] + tmp_11657[2]];
    signal tmp_11660[3] <== [17245004066890521540 * tmp_10558[0], 17245004066890521540 * tmp_10558[1], 17245004066890521540 * tmp_10558[2]];
    signal tmp_11661[3] <== [tmp_11658[0] + tmp_11660[0], tmp_11658[1] + tmp_11660[1], tmp_11658[2] + tmp_11660[2]];
    signal tmp_11663[3] <== [1582298761054006274 * tmp_10569[0], 1582298761054006274 * tmp_10569[1], 1582298761054006274 * tmp_10569[2]];
    signal tmp_11664[3] <== [tmp_11661[0] + tmp_11663[0], tmp_11661[1] + tmp_11663[1], tmp_11661[2] + tmp_11663[2]];
    signal tmp_11666[3] <== [16023586798997043070 * tmp_10581[0], 16023586798997043070 * tmp_10581[1], 16023586798997043070 * tmp_10581[2]];
    signal tmp_11667[3] <== [tmp_11664[0] + tmp_11666[0], tmp_11664[1] + tmp_11666[1], tmp_11664[2] + tmp_11666[2]];
    signal tmp_11669[3] <== [9167682708495677368 * tmp_10593[0], 9167682708495677368 * tmp_10593[1], 9167682708495677368 * tmp_10593[2]];
    signal tmp_11670[3] <== [tmp_11667[0] + tmp_11669[0], tmp_11667[1] + tmp_11669[1], tmp_11667[2] + tmp_11669[2]];
    signal tmp_11672[3] <== [1309677305456162249 * tmp_10605[0], 1309677305456162249 * tmp_10605[1], 1309677305456162249 * tmp_10605[2]];
    signal tmp_11673[3] <== [tmp_11670[0] + tmp_11672[0], tmp_11670[1] + tmp_11672[1], tmp_11670[2] + tmp_11672[2]];
    signal tmp_11675[3] <== [1052212424660468163 * tmp_10616[0], 1052212424660468163 * tmp_10616[1], 1052212424660468163 * tmp_10616[2]];
    signal tmp_11676[3] <== [tmp_11673[0] + tmp_11675[0], tmp_11673[1] + tmp_11675[1], tmp_11673[2] + tmp_11675[2]];
    signal tmp_11678[3] <== [10584314677809541083 * tmp_10628[0], 10584314677809541083 * tmp_10628[1], 10584314677809541083 * tmp_10628[2]];
    signal tmp_11679[3] <== [tmp_11676[0] + tmp_11678[0], tmp_11676[1] + tmp_11678[1], tmp_11676[2] + tmp_11678[2]];
    signal tmp_11681[3] <== [14090216998035345570 * tmp_10640[0], 14090216998035345570 * tmp_10640[1], 14090216998035345570 * tmp_10640[2]];
    signal tmp_11682[3] <== [tmp_11679[0] + tmp_11681[0], tmp_11679[1] + tmp_11681[1], tmp_11679[2] + tmp_11681[2]];
    signal tmp_11684[3] <== [18211453817785255223 * tmp_10652[0], 18211453817785255223 * tmp_10652[1], 18211453817785255223 * tmp_10652[2]];
    signal tmp_11685[3] <== [tmp_11682[0] + tmp_11684[0], tmp_11682[1] + tmp_11684[1], tmp_11682[2] + tmp_11684[2]];
    signal tmp_11687[3] <== [7540469726131798188 * tmp_10664[0], 7540469726131798188 * tmp_10664[1], 7540469726131798188 * tmp_10664[2]];
    signal tmp_11688[3] <== [tmp_11685[0] + tmp_11687[0], tmp_11685[1] + tmp_11687[1], tmp_11685[2] + tmp_11687[2]];
    signal tmp_11690[3] <== [8480085599690188340 * tmp_10676[0], 8480085599690188340 * tmp_10676[1], 8480085599690188340 * tmp_10676[2]];
    signal tmp_11691[3] <== [tmp_11688[0] + tmp_11690[0], tmp_11688[1] + tmp_11690[1], tmp_11688[2] + tmp_11690[2]];
    signal tmp_11693[3] <== [218778007896712464 * tmp_10688[0], 218778007896712464 * tmp_10688[1], 218778007896712464 * tmp_10688[2]];
    signal tmp_11694[3] <== [tmp_11691[0] + tmp_11693[0], tmp_11691[1] + tmp_11693[1], tmp_11691[2] + tmp_11693[2]];
    signal tmp_11696[3] <== [13578549160148989543 * tmp_10700[0], 13578549160148989543 * tmp_10700[1], 13578549160148989543 * tmp_10700[2]];
    signal tmp_11697[3] <== [tmp_11694[0] + tmp_11696[0], tmp_11694[1] + tmp_11696[1], tmp_11694[2] + tmp_11696[2]];
    signal tmp_11698[3] <== [evals[138][0] - tmp_11697[0], evals[138][1] - tmp_11697[1], evals[138][2] - tmp_11697[2]];
    signal tmp_11699[3] <== CMul()(tmp_9161, tmp_11698);
    signal tmp_11700[3] <== [tmp_11650[0] + tmp_11699[0], tmp_11650[1] + tmp_11699[1], tmp_11650[2] + tmp_11699[2]];
    signal tmp_11701[3] <== CMul()(challengeQ, tmp_11700);
    signal tmp_11703[3] <== [10833525987555107485 * tmp_10525[0], 10833525987555107485 * tmp_10525[1], 10833525987555107485 * tmp_10525[2]];
    signal tmp_11705[3] <== [3224171887150947390 * tmp_10535[0], 3224171887150947390 * tmp_10535[1], 3224171887150947390 * tmp_10535[2]];
    signal tmp_11706[3] <== [tmp_11703[0] + tmp_11705[0], tmp_11703[1] + tmp_11705[1], tmp_11703[2] + tmp_11705[2]];
    signal tmp_11708[3] <== [15832068917340433117 * tmp_10546[0], 15832068917340433117 * tmp_10546[1], 15832068917340433117 * tmp_10546[2]];
    signal tmp_11709[3] <== [tmp_11706[0] + tmp_11708[0], tmp_11706[1] + tmp_11708[1], tmp_11706[2] + tmp_11708[2]];
    signal tmp_11711[3] <== [7595102312128106020 * tmp_10558[0], 7595102312128106020 * tmp_10558[1], 7595102312128106020 * tmp_10558[2]];
    signal tmp_11712[3] <== [tmp_11709[0] + tmp_11711[0], tmp_11709[1] + tmp_11711[1], tmp_11709[2] + tmp_11711[2]];
    signal tmp_11714[3] <== [7700354089512536567 * tmp_10569[0], 7700354089512536567 * tmp_10569[1], 7700354089512536567 * tmp_10569[2]];
    signal tmp_11715[3] <== [tmp_11712[0] + tmp_11714[0], tmp_11712[1] + tmp_11714[1], tmp_11712[2] + tmp_11714[2]];
    signal tmp_11717[3] <== [12431191248332837556 * tmp_10581[0], 12431191248332837556 * tmp_10581[1], 12431191248332837556 * tmp_10581[2]];
    signal tmp_11718[3] <== [tmp_11715[0] + tmp_11717[0], tmp_11715[1] + tmp_11717[1], tmp_11715[2] + tmp_11717[2]];
    signal tmp_11720[3] <== [8615081588084801800 * tmp_10593[0], 8615081588084801800 * tmp_10593[1], 8615081588084801800 * tmp_10593[2]];
    signal tmp_11721[3] <== [tmp_11718[0] + tmp_11720[0], tmp_11718[1] + tmp_11720[1], tmp_11718[2] + tmp_11720[2]];
    signal tmp_11723[3] <== [17713155928766469063 * tmp_10605[0], 17713155928766469063 * tmp_10605[1], 17713155928766469063 * tmp_10605[2]];
    signal tmp_11724[3] <== [tmp_11721[0] + tmp_11723[0], tmp_11721[1] + tmp_11723[1], tmp_11721[2] + tmp_11723[2]];
    signal tmp_11726[3] <== [13520163937605726164 * tmp_10616[0], 13520163937605726164 * tmp_10616[1], 13520163937605726164 * tmp_10616[2]];
    signal tmp_11727[3] <== [tmp_11724[0] + tmp_11726[0], tmp_11724[1] + tmp_11726[1], tmp_11724[2] + tmp_11726[2]];
    signal tmp_11729[3] <== [13134043455414658692 * tmp_10628[0], 13134043455414658692 * tmp_10628[1], 13134043455414658692 * tmp_10628[2]];
    signal tmp_11730[3] <== [tmp_11727[0] + tmp_11729[0], tmp_11727[1] + tmp_11729[1], tmp_11727[2] + tmp_11729[2]];
    signal tmp_11732[3] <== [5807446223844687465 * tmp_10640[0], 5807446223844687465 * tmp_10640[1], 5807446223844687465 * tmp_10640[2]];
    signal tmp_11733[3] <== [tmp_11730[0] + tmp_11732[0], tmp_11730[1] + tmp_11732[1], tmp_11730[2] + tmp_11732[2]];
    signal tmp_11735[3] <== [6129425131043046986 * tmp_10652[0], 6129425131043046986 * tmp_10652[1], 6129425131043046986 * tmp_10652[2]];
    signal tmp_11736[3] <== [tmp_11733[0] + tmp_11735[0], tmp_11733[1] + tmp_11735[1], tmp_11733[2] + tmp_11735[2]];
    signal tmp_11738[3] <== [16009746502472456190 * tmp_10664[0], 16009746502472456190 * tmp_10664[1], 16009746502472456190 * tmp_10664[2]];
    signal tmp_11739[3] <== [tmp_11736[0] + tmp_11738[0], tmp_11736[1] + tmp_11738[1], tmp_11736[2] + tmp_11738[2]];
    signal tmp_11741[3] <== [7540469726131798188 * tmp_10676[0], 7540469726131798188 * tmp_10676[1], 7540469726131798188 * tmp_10676[2]];
    signal tmp_11742[3] <== [tmp_11739[0] + tmp_11741[0], tmp_11739[1] + tmp_11741[1], tmp_11739[2] + tmp_11741[2]];
    signal tmp_11744[3] <== [14201875104048534679 * tmp_10688[0], 14201875104048534679 * tmp_10688[1], 14201875104048534679 * tmp_10688[2]];
    signal tmp_11745[3] <== [tmp_11742[0] + tmp_11744[0], tmp_11742[1] + tmp_11744[1], tmp_11742[2] + tmp_11744[2]];
    signal tmp_11747[3] <== [17337375643812333090 * tmp_10700[0], 17337375643812333090 * tmp_10700[1], 17337375643812333090 * tmp_10700[2]];
    signal tmp_11748[3] <== [tmp_11745[0] + tmp_11747[0], tmp_11745[1] + tmp_11747[1], tmp_11745[2] + tmp_11747[2]];
    signal tmp_11749[3] <== [evals[139][0] - tmp_11748[0], evals[139][1] - tmp_11748[1], evals[139][2] - tmp_11748[2]];
    signal tmp_11750[3] <== CMul()(tmp_9161, tmp_11749);
    signal tmp_11751[3] <== [tmp_11701[0] + tmp_11750[0], tmp_11701[1] + tmp_11750[1], tmp_11701[2] + tmp_11750[2]];
    signal tmp_11752[3] <== CMul()(challengeQ, tmp_11751);
    signal tmp_11754[3] <== [8429731608347594559 * tmp_10525[0], 8429731608347594559 * tmp_10525[1], 8429731608347594559 * tmp_10525[2]];
    signal tmp_11756[3] <== [1843095456038194143 * tmp_10535[0], 1843095456038194143 * tmp_10535[1], 1843095456038194143 * tmp_10535[2]];
    signal tmp_11757[3] <== [tmp_11754[0] + tmp_11756[0], tmp_11754[1] + tmp_11756[1], tmp_11754[2] + tmp_11756[2]];
    signal tmp_11759[3] <== [4800348490672825001 * tmp_10546[0], 4800348490672825001 * tmp_10546[1], 4800348490672825001 * tmp_10546[2]];
    signal tmp_11760[3] <== [tmp_11757[0] + tmp_11759[0], tmp_11757[1] + tmp_11759[1], tmp_11757[2] + tmp_11759[2]];
    signal tmp_11762[3] <== [15175373777650174966 * tmp_10558[0], 15175373777650174966 * tmp_10558[1], 15175373777650174966 * tmp_10558[2]];
    signal tmp_11763[3] <== [tmp_11760[0] + tmp_11762[0], tmp_11760[1] + tmp_11762[1], tmp_11760[2] + tmp_11762[2]];
    signal tmp_11765[3] <== [15743243826515774520 * tmp_10569[0], 15743243826515774520 * tmp_10569[1], 15743243826515774520 * tmp_10569[2]];
    signal tmp_11766[3] <== [tmp_11763[0] + tmp_11765[0], tmp_11763[1] + tmp_11765[1], tmp_11763[2] + tmp_11765[2]];
    signal tmp_11768[3] <== [2498880319480651868 * tmp_10581[0], 2498880319480651868 * tmp_10581[1], 2498880319480651868 * tmp_10581[2]];
    signal tmp_11769[3] <== [tmp_11766[0] + tmp_11768[0], tmp_11766[1] + tmp_11768[1], tmp_11766[2] + tmp_11768[2]];
    signal tmp_11771[3] <== [7546636406606303166 * tmp_10593[0], 7546636406606303166 * tmp_10593[1], 7546636406606303166 * tmp_10593[2]];
    signal tmp_11772[3] <== [tmp_11769[0] + tmp_11771[0], tmp_11769[1] + tmp_11771[1], tmp_11769[2] + tmp_11771[2]];
    signal tmp_11774[3] <== [15350435609153813292 * tmp_10605[0], 15350435609153813292 * tmp_10605[1], 15350435609153813292 * tmp_10605[2]];
    signal tmp_11775[3] <== [tmp_11772[0] + tmp_11774[0], tmp_11772[1] + tmp_11774[1], tmp_11772[2] + tmp_11774[2]];
    signal tmp_11777[3] <== [16548416728592494912 * tmp_10616[0], 16548416728592494912 * tmp_10616[1], 16548416728592494912 * tmp_10616[2]];
    signal tmp_11778[3] <== [tmp_11775[0] + tmp_11777[0], tmp_11775[1] + tmp_11777[1], tmp_11775[2] + tmp_11777[2]];
    signal tmp_11780[3] <== [17993594930758673496 * tmp_10628[0], 17993594930758673496 * tmp_10628[1], 17993594930758673496 * tmp_10628[2]];
    signal tmp_11781[3] <== [tmp_11778[0] + tmp_11780[0], tmp_11778[1] + tmp_11780[1], tmp_11778[2] + tmp_11780[2]];
    signal tmp_11783[3] <== [13912658301238544598 * tmp_10640[0], 13912658301238544598 * tmp_10640[1], 13912658301238544598 * tmp_10640[2]];
    signal tmp_11784[3] <== [tmp_11781[0] + tmp_11783[0], tmp_11781[1] + tmp_11783[1], tmp_11781[2] + tmp_11783[2]];
    signal tmp_11786[3] <== [18075372928619437347 * tmp_10652[0], 18075372928619437347 * tmp_10652[1], 18075372928619437347 * tmp_10652[2]];
    signal tmp_11787[3] <== [tmp_11784[0] + tmp_11786[0], tmp_11784[1] + tmp_11786[1], tmp_11784[2] + tmp_11786[2]];
    signal tmp_11789[3] <== [6129425131043046986 * tmp_10664[0], 6129425131043046986 * tmp_10664[1], 6129425131043046986 * tmp_10664[2]];
    signal tmp_11790[3] <== [tmp_11787[0] + tmp_11789[0], tmp_11787[1] + tmp_11789[1], tmp_11787[2] + tmp_11789[2]];
    signal tmp_11792[3] <== [18211453817785255223 * tmp_10676[0], 18211453817785255223 * tmp_10676[1], 18211453817785255223 * tmp_10676[2]];
    signal tmp_11793[3] <== [tmp_11790[0] + tmp_11792[0], tmp_11790[1] + tmp_11792[1], tmp_11790[2] + tmp_11792[2]];
    signal tmp_11795[3] <== [9867063047527519896 * tmp_10688[0], 9867063047527519896 * tmp_10688[1], 9867063047527519896 * tmp_10688[2]];
    signal tmp_11796[3] <== [tmp_11793[0] + tmp_11795[0], tmp_11793[1] + tmp_11795[1], tmp_11793[2] + tmp_11795[2]];
    signal tmp_11798[3] <== [2765050638571938600 * tmp_10700[0], 2765050638571938600 * tmp_10700[1], 2765050638571938600 * tmp_10700[2]];
    signal tmp_11799[3] <== [tmp_11796[0] + tmp_11798[0], tmp_11796[1] + tmp_11798[1], tmp_11796[2] + tmp_11798[2]];
    signal tmp_11800[3] <== [evals[140][0] - tmp_11799[0], evals[140][1] - tmp_11799[1], evals[140][2] - tmp_11799[2]];
    signal tmp_11801[3] <== CMul()(tmp_9161, tmp_11800);
    signal tmp_11802[3] <== [tmp_11752[0] + tmp_11801[0], tmp_11752[1] + tmp_11801[1], tmp_11752[2] + tmp_11801[2]];
    signal tmp_11803[3] <== CMul()(challengeQ, tmp_11802);
    signal tmp_11805[3] <== [15303535044422768803 * tmp_10525[0], 15303535044422768803 * tmp_10525[1], 15303535044422768803 * tmp_10525[2]];
    signal tmp_11807[3] <== [6296951922972871295 * tmp_10535[0], 6296951922972871295 * tmp_10535[1], 6296951922972871295 * tmp_10535[2]];
    signal tmp_11808[3] <== [tmp_11805[0] + tmp_11807[0], tmp_11805[1] + tmp_11807[1], tmp_11805[2] + tmp_11807[2]];
    signal tmp_11810[3] <== [5823573138716677219 * tmp_10546[0], 5823573138716677219 * tmp_10546[1], 5823573138716677219 * tmp_10546[2]];
    signal tmp_11811[3] <== [tmp_11808[0] + tmp_11810[0], tmp_11808[1] + tmp_11810[1], tmp_11808[2] + tmp_11810[2]];
    signal tmp_11813[3] <== [2680431002152629139 * tmp_10558[0], 2680431002152629139 * tmp_10558[1], 2680431002152629139 * tmp_10558[2]];
    signal tmp_11814[3] <== [tmp_11811[0] + tmp_11813[0], tmp_11811[1] + tmp_11813[1], tmp_11811[2] + tmp_11813[2]];
    signal tmp_11816[3] <== [7366465902859006190 * tmp_10569[0], 7366465902859006190 * tmp_10569[1], 7366465902859006190 * tmp_10569[2]];
    signal tmp_11817[3] <== [tmp_11814[0] + tmp_11816[0], tmp_11814[1] + tmp_11816[1], tmp_11814[2] + tmp_11816[2]];
    signal tmp_11819[3] <== [4072077849300314190 * tmp_10581[0], 4072077849300314190 * tmp_10581[1], 4072077849300314190 * tmp_10581[2]];
    signal tmp_11820[3] <== [tmp_11817[0] + tmp_11819[0], tmp_11817[1] + tmp_11819[1], tmp_11817[2] + tmp_11819[2]];
    signal tmp_11822[3] <== [16140126883652446716 * tmp_10593[0], 16140126883652446716 * tmp_10593[1], 16140126883652446716 * tmp_10593[2]];
    signal tmp_11823[3] <== [tmp_11820[0] + tmp_11822[0], tmp_11820[1] + tmp_11822[1], tmp_11820[2] + tmp_11822[2]];
    signal tmp_11825[3] <== [2601451337846871084 * tmp_10605[0], 2601451337846871084 * tmp_10605[1], 2601451337846871084 * tmp_10605[2]];
    signal tmp_11826[3] <== [tmp_11823[0] + tmp_11825[0], tmp_11823[1] + tmp_11825[1], tmp_11823[2] + tmp_11825[2]];
    signal tmp_11828[3] <== [15725013897929440397 * tmp_10616[0], 15725013897929440397 * tmp_10616[1], 15725013897929440397 * tmp_10616[2]];
    signal tmp_11829[3] <== [tmp_11826[0] + tmp_11828[0], tmp_11826[1] + tmp_11828[1], tmp_11826[2] + tmp_11828[2]];
    signal tmp_11831[3] <== [4389532408180633377 * tmp_10628[0], 4389532408180633377 * tmp_10628[1], 4389532408180633377 * tmp_10628[2]];
    signal tmp_11832[3] <== [tmp_11829[0] + tmp_11831[0], tmp_11829[1] + tmp_11831[1], tmp_11829[2] + tmp_11831[2]];
    signal tmp_11834[3] <== [1476671367976490131 * tmp_10640[0], 1476671367976490131 * tmp_10640[1], 1476671367976490131 * tmp_10640[2]];
    signal tmp_11835[3] <== [tmp_11832[0] + tmp_11834[0], tmp_11832[1] + tmp_11834[1], tmp_11832[2] + tmp_11834[2]];
    signal tmp_11837[3] <== [13912658301238544598 * tmp_10652[0], 13912658301238544598 * tmp_10652[1], 13912658301238544598 * tmp_10652[2]];
    signal tmp_11838[3] <== [tmp_11835[0] + tmp_11837[0], tmp_11835[1] + tmp_11837[1], tmp_11835[2] + tmp_11837[2]];
    signal tmp_11840[3] <== [5807446223844687465 * tmp_10664[0], 5807446223844687465 * tmp_10664[1], 5807446223844687465 * tmp_10664[2]];
    signal tmp_11841[3] <== [tmp_11838[0] + tmp_11840[0], tmp_11838[1] + tmp_11840[1], tmp_11838[2] + tmp_11840[2]];
    signal tmp_11843[3] <== [14090216998035345570 * tmp_10676[0], 14090216998035345570 * tmp_10676[1], 14090216998035345570 * tmp_10676[2]];
    signal tmp_11844[3] <== [tmp_11841[0] + tmp_11843[0], tmp_11841[1] + tmp_11843[1], tmp_11841[2] + tmp_11843[2]];
    signal tmp_11846[3] <== [1725490462187460494 * tmp_10688[0], 1725490462187460494 * tmp_10688[1], 1725490462187460494 * tmp_10688[2]];
    signal tmp_11847[3] <== [tmp_11844[0] + tmp_11846[0], tmp_11844[1] + tmp_11846[1], tmp_11844[2] + tmp_11846[2]];
    signal tmp_11849[3] <== [6782251596454435859 * tmp_10700[0], 6782251596454435859 * tmp_10700[1], 6782251596454435859 * tmp_10700[2]];
    signal tmp_11850[3] <== [tmp_11847[0] + tmp_11849[0], tmp_11847[1] + tmp_11849[1], tmp_11847[2] + tmp_11849[2]];
    signal tmp_11851[3] <== [evals[141][0] - tmp_11850[0], evals[141][1] - tmp_11850[1], evals[141][2] - tmp_11850[2]];
    signal tmp_11852[3] <== CMul()(tmp_9161, tmp_11851);
    signal tmp_11853[3] <== [tmp_11803[0] + tmp_11852[0], tmp_11803[1] + tmp_11852[1], tmp_11803[2] + tmp_11852[2]];
    signal tmp_11854[3] <== CMul()(challengeQ, tmp_11853);
    signal tmp_11856[3] <== [10595610446894069209 * tmp_10525[0], 10595610446894069209 * tmp_10525[1], 10595610446894069209 * tmp_10525[2]];
    signal tmp_11858[3] <== [3473665614188209191 * tmp_10535[0], 3473665614188209191 * tmp_10535[1], 3473665614188209191 * tmp_10535[2]];
    signal tmp_11859[3] <== [tmp_11856[0] + tmp_11858[0], tmp_11856[1] + tmp_11858[1], tmp_11856[2] + tmp_11858[2]];
    signal tmp_11861[3] <== [13074116889890804777 * tmp_10546[0], 13074116889890804777 * tmp_10546[1], 13074116889890804777 * tmp_10546[2]];
    signal tmp_11862[3] <== [tmp_11859[0] + tmp_11861[0], tmp_11859[1] + tmp_11861[1], tmp_11859[2] + tmp_11861[2]];
    signal tmp_11864[3] <== [16359376528724855414 * tmp_10558[0], 16359376528724855414 * tmp_10558[1], 16359376528724855414 * tmp_10558[2]];
    signal tmp_11865[3] <== [tmp_11862[0] + tmp_11864[0], tmp_11862[1] + tmp_11864[1], tmp_11862[2] + tmp_11864[2]];
    signal tmp_11867[3] <== [16204753907425352648 * tmp_10569[0], 16204753907425352648 * tmp_10569[1], 16204753907425352648 * tmp_10569[2]];
    signal tmp_11868[3] <== [tmp_11865[0] + tmp_11867[0], tmp_11865[1] + tmp_11867[1], tmp_11865[2] + tmp_11867[2]];
    signal tmp_11870[3] <== [1099103502109499253 * tmp_10581[0], 1099103502109499253 * tmp_10581[1], 1099103502109499253 * tmp_10581[2]];
    signal tmp_11871[3] <== [tmp_11868[0] + tmp_11870[0], tmp_11868[1] + tmp_11870[1], tmp_11868[2] + tmp_11870[2]];
    signal tmp_11873[3] <== [4616609025864245522 * tmp_10593[0], 4616609025864245522 * tmp_10593[1], 4616609025864245522 * tmp_10593[2]];
    signal tmp_11874[3] <== [tmp_11871[0] + tmp_11873[0], tmp_11871[1] + tmp_11873[1], tmp_11871[2] + tmp_11873[2]];
    signal tmp_11876[3] <== [7362869829681961507 * tmp_10605[0], 7362869829681961507 * tmp_10605[1], 7362869829681961507 * tmp_10605[2]];
    signal tmp_11877[3] <== [tmp_11874[0] + tmp_11876[0], tmp_11874[1] + tmp_11876[1], tmp_11874[2] + tmp_11876[2]];
    signal tmp_11879[3] <== [231791526046105962 * tmp_10616[0], 231791526046105962 * tmp_10616[1], 231791526046105962 * tmp_10616[2]];
    signal tmp_11880[3] <== [tmp_11877[0] + tmp_11879[0], tmp_11877[1] + tmp_11879[1], tmp_11877[2] + tmp_11879[2]];
    signal tmp_11882[3] <== [140048027282429797 * tmp_10628[0], 140048027282429797 * tmp_10628[1], 140048027282429797 * tmp_10628[2]];
    signal tmp_11883[3] <== [tmp_11880[0] + tmp_11882[0], tmp_11880[1] + tmp_11882[1], tmp_11880[2] + tmp_11882[2]];
    signal tmp_11885[3] <== [4389532408180633377 * tmp_10640[0], 4389532408180633377 * tmp_10640[1], 4389532408180633377 * tmp_10640[2]];
    signal tmp_11886[3] <== [tmp_11883[0] + tmp_11885[0], tmp_11883[1] + tmp_11885[1], tmp_11883[2] + tmp_11885[2]];
    signal tmp_11888[3] <== [17993594930758673496 * tmp_10652[0], 17993594930758673496 * tmp_10652[1], 17993594930758673496 * tmp_10652[2]];
    signal tmp_11889[3] <== [tmp_11886[0] + tmp_11888[0], tmp_11886[1] + tmp_11888[1], tmp_11886[2] + tmp_11888[2]];
    signal tmp_11891[3] <== [13134043455414658692 * tmp_10664[0], 13134043455414658692 * tmp_10664[1], 13134043455414658692 * tmp_10664[2]];
    signal tmp_11892[3] <== [tmp_11889[0] + tmp_11891[0], tmp_11889[1] + tmp_11891[1], tmp_11889[2] + tmp_11891[2]];
    signal tmp_11894[3] <== [10584314677809541083 * tmp_10676[0], 10584314677809541083 * tmp_10676[1], 10584314677809541083 * tmp_10676[2]];
    signal tmp_11895[3] <== [tmp_11892[0] + tmp_11894[0], tmp_11892[1] + tmp_11894[1], tmp_11892[2] + tmp_11894[2]];
    signal tmp_11897[3] <== [16803151265961688719 * tmp_10688[0], 16803151265961688719 * tmp_10688[1], 16803151265961688719 * tmp_10688[2]];
    signal tmp_11898[3] <== [tmp_11895[0] + tmp_11897[0], tmp_11895[1] + tmp_11897[1], tmp_11895[2] + tmp_11897[2]];
    signal tmp_11900[3] <== [12266684712140129864 * tmp_10700[0], 12266684712140129864 * tmp_10700[1], 12266684712140129864 * tmp_10700[2]];
    signal tmp_11901[3] <== [tmp_11898[0] + tmp_11900[0], tmp_11898[1] + tmp_11900[1], tmp_11898[2] + tmp_11900[2]];
    signal tmp_11902[3] <== [evals[142][0] - tmp_11901[0], evals[142][1] - tmp_11901[1], evals[142][2] - tmp_11901[2]];
    signal tmp_11903[3] <== CMul()(tmp_9161, tmp_11902);
    signal tmp_11904[3] <== [tmp_11854[0] + tmp_11903[0], tmp_11854[1] + tmp_11903[1], tmp_11854[2] + tmp_11903[2]];
    signal tmp_11905[3] <== CMul()(challengeQ, tmp_11904);
    signal tmp_11907[3] <== [13368554547082880001 * tmp_10525[0], 13368554547082880001 * tmp_10525[1], 13368554547082880001 * tmp_10525[2]];
    signal tmp_11909[3] <== [16995162224826066164 * tmp_10535[0], 16995162224826066164 * tmp_10535[1], 16995162224826066164 * tmp_10535[2]];
    signal tmp_11910[3] <== [tmp_11907[0] + tmp_11909[0], tmp_11907[1] + tmp_11909[1], tmp_11907[2] + tmp_11909[2]];
    signal tmp_11912[3] <== [10542619009725154399 * tmp_10546[0], 10542619009725154399 * tmp_10546[1], 10542619009725154399 * tmp_10546[2]];
    signal tmp_11913[3] <== [tmp_11910[0] + tmp_11912[0], tmp_11910[1] + tmp_11912[1], tmp_11910[2] + tmp_11912[2]];
    signal tmp_11915[3] <== [16073073456050006575 * tmp_10558[0], 16073073456050006575 * tmp_10558[1], 16073073456050006575 * tmp_10558[2]];
    signal tmp_11916[3] <== [tmp_11913[0] + tmp_11915[0], tmp_11913[1] + tmp_11915[1], tmp_11913[2] + tmp_11915[2]];
    signal tmp_11918[3] <== [9008401972144753939 * tmp_10569[0], 9008401972144753939 * tmp_10569[1], 9008401972144753939 * tmp_10569[2]];
    signal tmp_11919[3] <== [tmp_11916[0] + tmp_11918[0], tmp_11916[1] + tmp_11918[1], tmp_11916[2] + tmp_11918[2]];
    signal tmp_11921[3] <== [1749680724535600815 * tmp_10581[0], 1749680724535600815 * tmp_10581[1], 1749680724535600815 * tmp_10581[2]];
    signal tmp_11922[3] <== [tmp_11919[0] + tmp_11921[0], tmp_11919[1] + tmp_11921[1], tmp_11919[2] + tmp_11921[2]];
    signal tmp_11924[3] <== [5348209698235102331 * tmp_10593[0], 5348209698235102331 * tmp_10593[1], 5348209698235102331 * tmp_10593[2]];
    signal tmp_11925[3] <== [tmp_11922[0] + tmp_11924[0], tmp_11922[1] + tmp_11924[1], tmp_11922[2] + tmp_11924[2]];
    signal tmp_11927[3] <== [5200922087361472057 * tmp_10605[0], 5200922087361472057 * tmp_10605[1], 5200922087361472057 * tmp_10605[2]];
    signal tmp_11928[3] <== [tmp_11925[0] + tmp_11927[0], tmp_11925[1] + tmp_11927[1], tmp_11925[2] + tmp_11927[2]];
    signal tmp_11930[3] <== [16325322198676206943 * tmp_10616[0], 16325322198676206943 * tmp_10616[1], 16325322198676206943 * tmp_10616[2]];
    signal tmp_11931[3] <== [tmp_11928[0] + tmp_11930[0], tmp_11928[1] + tmp_11930[1], tmp_11928[2] + tmp_11930[2]];
    signal tmp_11933[3] <== [231791526046105962 * tmp_10628[0], 231791526046105962 * tmp_10628[1], 231791526046105962 * tmp_10628[2]];
    signal tmp_11934[3] <== [tmp_11931[0] + tmp_11933[0], tmp_11931[1] + tmp_11933[1], tmp_11931[2] + tmp_11933[2]];
    signal tmp_11936[3] <== [15725013897929440397 * tmp_10640[0], 15725013897929440397 * tmp_10640[1], 15725013897929440397 * tmp_10640[2]];
    signal tmp_11937[3] <== [tmp_11934[0] + tmp_11936[0], tmp_11934[1] + tmp_11936[1], tmp_11934[2] + tmp_11936[2]];
    signal tmp_11939[3] <== [16548416728592494912 * tmp_10652[0], 16548416728592494912 * tmp_10652[1], 16548416728592494912 * tmp_10652[2]];
    signal tmp_11940[3] <== [tmp_11937[0] + tmp_11939[0], tmp_11937[1] + tmp_11939[1], tmp_11937[2] + tmp_11939[2]];
    signal tmp_11942[3] <== [13520163937605726164 * tmp_10664[0], 13520163937605726164 * tmp_10664[1], 13520163937605726164 * tmp_10664[2]];
    signal tmp_11943[3] <== [tmp_11940[0] + tmp_11942[0], tmp_11940[1] + tmp_11942[1], tmp_11940[2] + tmp_11942[2]];
    signal tmp_11945[3] <== [1052212424660468163 * tmp_10676[0], 1052212424660468163 * tmp_10676[1], 1052212424660468163 * tmp_10676[2]];
    signal tmp_11946[3] <== [tmp_11943[0] + tmp_11945[0], tmp_11943[1] + tmp_11945[1], tmp_11943[2] + tmp_11945[2]];
    signal tmp_11948[3] <== [17162061841860881922 * tmp_10688[0], 17162061841860881922 * tmp_10688[1], 17162061841860881922 * tmp_10688[2]];
    signal tmp_11949[3] <== [tmp_11946[0] + tmp_11948[0], tmp_11946[1] + tmp_11948[1], tmp_11946[2] + tmp_11948[2]];
    signal tmp_11951[3] <== [6468404447315445157 * tmp_10700[0], 6468404447315445157 * tmp_10700[1], 6468404447315445157 * tmp_10700[2]];
    signal tmp_11952[3] <== [tmp_11949[0] + tmp_11951[0], tmp_11949[1] + tmp_11951[1], tmp_11949[2] + tmp_11951[2]];
    signal tmp_11953[3] <== [evals[143][0] - tmp_11952[0], evals[143][1] - tmp_11952[1], evals[143][2] - tmp_11952[2]];
    signal tmp_11954[3] <== CMul()(tmp_9161, tmp_11953);
    signal tmp_11955[3] <== [tmp_11905[0] + tmp_11954[0], tmp_11905[1] + tmp_11954[1], tmp_11905[2] + tmp_11954[2]];
    signal tmp_11956[3] <== CMul()(challengeQ, tmp_11955);
    signal tmp_11958[3] <== [16236357391915579752 * tmp_10525[0], 16236357391915579752 * tmp_10525[1], 16236357391915579752 * tmp_10525[2]];
    signal tmp_11960[3] <== [2360357678915503787 * tmp_10535[0], 2360357678915503787 * tmp_10535[1], 2360357678915503787 * tmp_10535[2]];
    signal tmp_11961[3] <== [tmp_11958[0] + tmp_11960[0], tmp_11958[1] + tmp_11960[1], tmp_11958[2] + tmp_11960[2]];
    signal tmp_11963[3] <== [5269705790554862459 * tmp_10546[0], 5269705790554862459 * tmp_10546[1], 5269705790554862459 * tmp_10546[2]];
    signal tmp_11964[3] <== [tmp_11961[0] + tmp_11963[0], tmp_11961[1] + tmp_11963[1], tmp_11961[2] + tmp_11963[2]];
    signal tmp_11966[3] <== [9707166089812971789 * tmp_10558[0], 9707166089812971789 * tmp_10558[1], 9707166089812971789 * tmp_10558[2]];
    signal tmp_11967[3] <== [tmp_11964[0] + tmp_11966[0], tmp_11964[1] + tmp_11966[1], tmp_11964[2] + tmp_11966[2]];
    signal tmp_11969[3] <== [13551115817015094532 * tmp_10569[0], 13551115817015094532 * tmp_10569[1], 13551115817015094532 * tmp_10569[2]];
    signal tmp_11970[3] <== [tmp_11967[0] + tmp_11969[0], tmp_11967[1] + tmp_11969[1], tmp_11967[2] + tmp_11969[2]];
    signal tmp_11972[3] <== [11909911263973988019 * tmp_10581[0], 11909911263973988019 * tmp_10581[1], 11909911263973988019 * tmp_10581[2]];
    signal tmp_11973[3] <== [tmp_11970[0] + tmp_11972[0], tmp_11970[1] + tmp_11972[1], tmp_11970[2] + tmp_11972[2]];
    signal tmp_11975[3] <== [5477818864382526223 * tmp_10593[0], 5477818864382526223 * tmp_10593[1], 5477818864382526223 * tmp_10593[2]];
    signal tmp_11976[3] <== [tmp_11973[0] + tmp_11975[0], tmp_11973[1] + tmp_11975[1], tmp_11973[2] + tmp_11975[2]];
    signal tmp_11978[3] <== [6662788181627778720 * tmp_10605[0], 6662788181627778720 * tmp_10605[1], 6662788181627778720 * tmp_10605[2]];
    signal tmp_11979[3] <== [tmp_11976[0] + tmp_11978[0], tmp_11976[1] + tmp_11978[1], tmp_11976[2] + tmp_11978[2]];
    signal tmp_11981[3] <== [5200922087361472057 * tmp_10616[0], 5200922087361472057 * tmp_10616[1], 5200922087361472057 * tmp_10616[2]];
    signal tmp_11982[3] <== [tmp_11979[0] + tmp_11981[0], tmp_11979[1] + tmp_11981[1], tmp_11979[2] + tmp_11981[2]];
    signal tmp_11984[3] <== [7362869829681961507 * tmp_10628[0], 7362869829681961507 * tmp_10628[1], 7362869829681961507 * tmp_10628[2]];
    signal tmp_11985[3] <== [tmp_11982[0] + tmp_11984[0], tmp_11982[1] + tmp_11984[1], tmp_11982[2] + tmp_11984[2]];
    signal tmp_11987[3] <== [2601451337846871084 * tmp_10640[0], 2601451337846871084 * tmp_10640[1], 2601451337846871084 * tmp_10640[2]];
    signal tmp_11988[3] <== [tmp_11985[0] + tmp_11987[0], tmp_11985[1] + tmp_11987[1], tmp_11985[2] + tmp_11987[2]];
    signal tmp_11990[3] <== [15350435609153813292 * tmp_10652[0], 15350435609153813292 * tmp_10652[1], 15350435609153813292 * tmp_10652[2]];
    signal tmp_11991[3] <== [tmp_11988[0] + tmp_11990[0], tmp_11988[1] + tmp_11990[1], tmp_11988[2] + tmp_11990[2]];
    signal tmp_11993[3] <== [17713155928766469063 * tmp_10664[0], 17713155928766469063 * tmp_10664[1], 17713155928766469063 * tmp_10664[2]];
    signal tmp_11994[3] <== [tmp_11991[0] + tmp_11993[0], tmp_11991[1] + tmp_11993[1], tmp_11991[2] + tmp_11993[2]];
    signal tmp_11996[3] <== [1309677305456162249 * tmp_10676[0], 1309677305456162249 * tmp_10676[1], 1309677305456162249 * tmp_10676[2]];
    signal tmp_11997[3] <== [tmp_11994[0] + tmp_11996[0], tmp_11994[1] + tmp_11996[1], tmp_11994[2] + tmp_11996[2]];
    signal tmp_11999[3] <== [14645553078015703863 * tmp_10688[0], 14645553078015703863 * tmp_10688[1], 14645553078015703863 * tmp_10688[2]];
    signal tmp_12000[3] <== [tmp_11997[0] + tmp_11999[0], tmp_11997[1] + tmp_11999[1], tmp_11997[2] + tmp_11999[2]];
    signal tmp_12002[3] <== [11180511074533951034 * tmp_10700[0], 11180511074533951034 * tmp_10700[1], 11180511074533951034 * tmp_10700[2]];
    signal tmp_12003[3] <== [tmp_12000[0] + tmp_12002[0], tmp_12000[1] + tmp_12002[1], tmp_12000[2] + tmp_12002[2]];
    signal tmp_12004[3] <== [evals[144][0] - tmp_12003[0], evals[144][1] - tmp_12003[1], evals[144][2] - tmp_12003[2]];
    signal tmp_12005[3] <== CMul()(tmp_9161, tmp_12004);
    signal tmp_12006[3] <== [tmp_11956[0] + tmp_12005[0], tmp_11956[1] + tmp_12005[1], tmp_11956[2] + tmp_12005[2]];
    signal tmp_12007[3] <== CMul()(challengeQ, tmp_12006);
    signal tmp_12009[3] <== [8046651557188262448 * tmp_10525[0], 8046651557188262448 * tmp_10525[1], 8046651557188262448 * tmp_10525[2]];
    signal tmp_12011[3] <== [14505687488637668062 * tmp_10535[0], 14505687488637668062 * tmp_10535[1], 14505687488637668062 * tmp_10535[2]];
    signal tmp_12012[3] <== [tmp_12009[0] + tmp_12011[0], tmp_12009[1] + tmp_12011[1], tmp_12009[2] + tmp_12011[2]];
    signal tmp_12014[3] <== [10449762191515258658 * tmp_10546[0], 10449762191515258658 * tmp_10546[1], 10449762191515258658 * tmp_10546[2]];
    signal tmp_12015[3] <== [tmp_12012[0] + tmp_12014[0], tmp_12012[1] + tmp_12014[1], tmp_12012[2] + tmp_12014[2]];
    signal tmp_12017[3] <== [12361291211423778677 * tmp_10558[0], 12361291211423778677 * tmp_10558[1], 12361291211423778677 * tmp_10558[2]];
    signal tmp_12018[3] <== [tmp_12015[0] + tmp_12017[0], tmp_12015[1] + tmp_12017[1], tmp_12015[2] + tmp_12017[2]];
    signal tmp_12020[3] <== [14725357302690445139 * tmp_10569[0], 14725357302690445139 * tmp_10569[1], 14725357302690445139 * tmp_10569[2]];
    signal tmp_12021[3] <== [tmp_12018[0] + tmp_12020[0], tmp_12018[1] + tmp_12020[1], tmp_12018[2] + tmp_12020[2]];
    signal tmp_12023[3] <== [13834032822097992727 * tmp_10581[0], 13834032822097992727 * tmp_10581[1], 13834032822097992727 * tmp_10581[2]];
    signal tmp_12024[3] <== [tmp_12021[0] + tmp_12023[0], tmp_12021[1] + tmp_12023[1], tmp_12021[2] + tmp_12023[2]];
    signal tmp_12026[3] <== [7412077663147604037 * tmp_10593[0], 7412077663147604037 * tmp_10593[1], 7412077663147604037 * tmp_10593[2]];
    signal tmp_12027[3] <== [tmp_12024[0] + tmp_12026[0], tmp_12024[1] + tmp_12026[1], tmp_12024[2] + tmp_12026[2]];
    signal tmp_12029[3] <== [5477818864382526223 * tmp_10605[0], 5477818864382526223 * tmp_10605[1], 5477818864382526223 * tmp_10605[2]];
    signal tmp_12030[3] <== [tmp_12027[0] + tmp_12029[0], tmp_12027[1] + tmp_12029[1], tmp_12027[2] + tmp_12029[2]];
    signal tmp_12032[3] <== [5348209698235102331 * tmp_10616[0], 5348209698235102331 * tmp_10616[1], 5348209698235102331 * tmp_10616[2]];
    signal tmp_12033[3] <== [tmp_12030[0] + tmp_12032[0], tmp_12030[1] + tmp_12032[1], tmp_12030[2] + tmp_12032[2]];
    signal tmp_12035[3] <== [4616609025864245522 * tmp_10628[0], 4616609025864245522 * tmp_10628[1], 4616609025864245522 * tmp_10628[2]];
    signal tmp_12036[3] <== [tmp_12033[0] + tmp_12035[0], tmp_12033[1] + tmp_12035[1], tmp_12033[2] + tmp_12035[2]];
    signal tmp_12038[3] <== [16140126883652446716 * tmp_10640[0], 16140126883652446716 * tmp_10640[1], 16140126883652446716 * tmp_10640[2]];
    signal tmp_12039[3] <== [tmp_12036[0] + tmp_12038[0], tmp_12036[1] + tmp_12038[1], tmp_12036[2] + tmp_12038[2]];
    signal tmp_12041[3] <== [7546636406606303166 * tmp_10652[0], 7546636406606303166 * tmp_10652[1], 7546636406606303166 * tmp_10652[2]];
    signal tmp_12042[3] <== [tmp_12039[0] + tmp_12041[0], tmp_12039[1] + tmp_12041[1], tmp_12039[2] + tmp_12041[2]];
    signal tmp_12044[3] <== [8615081588084801800 * tmp_10664[0], 8615081588084801800 * tmp_10664[1], 8615081588084801800 * tmp_10664[2]];
    signal tmp_12045[3] <== [tmp_12042[0] + tmp_12044[0], tmp_12042[1] + tmp_12044[1], tmp_12042[2] + tmp_12044[2]];
    signal tmp_12047[3] <== [9167682708495677368 * tmp_10676[0], 9167682708495677368 * tmp_10676[1], 9167682708495677368 * tmp_10676[2]];
    signal tmp_12048[3] <== [tmp_12045[0] + tmp_12047[0], tmp_12045[1] + tmp_12047[1], tmp_12045[2] + tmp_12047[2]];
    signal tmp_12050[3] <== [6551497314953504346 * tmp_10688[0], 6551497314953504346 * tmp_10688[1], 6551497314953504346 * tmp_10688[2]];
    signal tmp_12051[3] <== [tmp_12048[0] + tmp_12050[0], tmp_12048[1] + tmp_12050[1], tmp_12048[2] + tmp_12050[2]];
    signal tmp_12053[3] <== [6702954500202032351 * tmp_10700[0], 6702954500202032351 * tmp_10700[1], 6702954500202032351 * tmp_10700[2]];
    signal tmp_12054[3] <== [tmp_12051[0] + tmp_12053[0], tmp_12051[1] + tmp_12053[1], tmp_12051[2] + tmp_12053[2]];
    signal tmp_12055[3] <== [evals[145][0] - tmp_12054[0], evals[145][1] - tmp_12054[1], evals[145][2] - tmp_12054[2]];
    signal tmp_12056[3] <== CMul()(tmp_9161, tmp_12055);
    signal tmp_12057[3] <== [tmp_12007[0] + tmp_12056[0], tmp_12007[1] + tmp_12056[1], tmp_12007[2] + tmp_12056[2]];
    signal tmp_12058[3] <== CMul()(challengeQ, tmp_12057);
    signal tmp_12060[3] <== [10319242957211935185 * tmp_10525[0], 10319242957211935185 * tmp_10525[1], 10319242957211935185 * tmp_10525[2]];
    signal tmp_12062[3] <== [2525078022317723924 * tmp_10535[0], 2525078022317723924 * tmp_10535[1], 2525078022317723924 * tmp_10535[2]];
    signal tmp_12063[3] <== [tmp_12060[0] + tmp_12062[0], tmp_12060[1] + tmp_12062[1], tmp_12060[2] + tmp_12062[2]];
    signal tmp_12065[3] <== [7063454713635591383 * tmp_10546[0], 7063454713635591383 * tmp_10546[1], 7063454713635591383 * tmp_10546[2]];
    signal tmp_12066[3] <== [tmp_12063[0] + tmp_12065[0], tmp_12063[1] + tmp_12065[1], tmp_12063[2] + tmp_12065[2]];
    signal tmp_12068[3] <== [17138871444129789158 * tmp_10558[0], 17138871444129789158 * tmp_10558[1], 17138871444129789158 * tmp_10558[2]];
    signal tmp_12069[3] <== [tmp_12066[0] + tmp_12068[0], tmp_12066[1] + tmp_12068[1], tmp_12066[2] + tmp_12068[2]];
    signal tmp_12071[3] <== [16653803537962124952 * tmp_10569[0], 16653803537962124952 * tmp_10569[1], 16653803537962124952 * tmp_10569[2]];
    signal tmp_12072[3] <== [tmp_12069[0] + tmp_12071[0], tmp_12069[1] + tmp_12071[1], tmp_12069[2] + tmp_12071[2]];
    signal tmp_12074[3] <== [4015853678780844755 * tmp_10581[0], 4015853678780844755 * tmp_10581[1], 4015853678780844755 * tmp_10581[2]];
    signal tmp_12075[3] <== [tmp_12072[0] + tmp_12074[0], tmp_12072[1] + tmp_12074[1], tmp_12072[2] + tmp_12074[2]];
    signal tmp_12077[3] <== [13834032822097992727 * tmp_10593[0], 13834032822097992727 * tmp_10593[1], 13834032822097992727 * tmp_10593[2]];
    signal tmp_12078[3] <== [tmp_12075[0] + tmp_12077[0], tmp_12075[1] + tmp_12077[1], tmp_12075[2] + tmp_12077[2]];
    signal tmp_12080[3] <== [11909911263973988019 * tmp_10605[0], 11909911263973988019 * tmp_10605[1], 11909911263973988019 * tmp_10605[2]];
    signal tmp_12081[3] <== [tmp_12078[0] + tmp_12080[0], tmp_12078[1] + tmp_12080[1], tmp_12078[2] + tmp_12080[2]];
    signal tmp_12083[3] <== [1749680724535600815 * tmp_10616[0], 1749680724535600815 * tmp_10616[1], 1749680724535600815 * tmp_10616[2]];
    signal tmp_12084[3] <== [tmp_12081[0] + tmp_12083[0], tmp_12081[1] + tmp_12083[1], tmp_12081[2] + tmp_12083[2]];
    signal tmp_12086[3] <== [1099103502109499253 * tmp_10628[0], 1099103502109499253 * tmp_10628[1], 1099103502109499253 * tmp_10628[2]];
    signal tmp_12087[3] <== [tmp_12084[0] + tmp_12086[0], tmp_12084[1] + tmp_12086[1], tmp_12084[2] + tmp_12086[2]];
    signal tmp_12089[3] <== [4072077849300314190 * tmp_10640[0], 4072077849300314190 * tmp_10640[1], 4072077849300314190 * tmp_10640[2]];
    signal tmp_12090[3] <== [tmp_12087[0] + tmp_12089[0], tmp_12087[1] + tmp_12089[1], tmp_12087[2] + tmp_12089[2]];
    signal tmp_12092[3] <== [2498880319480651868 * tmp_10652[0], 2498880319480651868 * tmp_10652[1], 2498880319480651868 * tmp_10652[2]];
    signal tmp_12093[3] <== [tmp_12090[0] + tmp_12092[0], tmp_12090[1] + tmp_12092[1], tmp_12090[2] + tmp_12092[2]];
    signal tmp_12095[3] <== [12431191248332837556 * tmp_10664[0], 12431191248332837556 * tmp_10664[1], 12431191248332837556 * tmp_10664[2]];
    signal tmp_12096[3] <== [tmp_12093[0] + tmp_12095[0], tmp_12093[1] + tmp_12095[1], tmp_12093[2] + tmp_12095[2]];
    signal tmp_12098[3] <== [16023586798997043070 * tmp_10676[0], 16023586798997043070 * tmp_10676[1], 16023586798997043070 * tmp_10676[2]];
    signal tmp_12099[3] <== [tmp_12096[0] + tmp_12098[0], tmp_12096[1] + tmp_12098[1], tmp_12096[2] + tmp_12098[2]];
    signal tmp_12101[3] <== [1810923409258252859 * tmp_10688[0], 1810923409258252859 * tmp_10688[1], 1810923409258252859 * tmp_10688[2]];
    signal tmp_12102[3] <== [tmp_12099[0] + tmp_12101[0], tmp_12099[1] + tmp_12101[1], tmp_12099[2] + tmp_12101[2]];
    signal tmp_12104[3] <== [17884228943875298379 * tmp_10700[0], 17884228943875298379 * tmp_10700[1], 17884228943875298379 * tmp_10700[2]];
    signal tmp_12105[3] <== [tmp_12102[0] + tmp_12104[0], tmp_12102[1] + tmp_12104[1], tmp_12102[2] + tmp_12104[2]];
    signal tmp_12106[3] <== [evals[146][0] - tmp_12105[0], evals[146][1] - tmp_12105[1], evals[146][2] - tmp_12105[2]];
    signal tmp_12107[3] <== CMul()(tmp_9161, tmp_12106);
    signal tmp_12108[3] <== [tmp_12058[0] + tmp_12107[0], tmp_12058[1] + tmp_12107[1], tmp_12058[2] + tmp_12107[2]];
    signal tmp_12109[3] <== CMul()(challengeQ, tmp_12108);
    signal tmp_12111[3] <== [16137054013242276452 * tmp_10525[0], 16137054013242276452 * tmp_10525[1], 16137054013242276452 * tmp_10525[2]];
    signal tmp_12113[3] <== [3252585408322091368 * tmp_10535[0], 3252585408322091368 * tmp_10535[1], 3252585408322091368 * tmp_10535[2]];
    signal tmp_12114[3] <== [tmp_12111[0] + tmp_12113[0], tmp_12111[1] + tmp_12113[1], tmp_12111[2] + tmp_12113[2]];
    signal tmp_12116[3] <== [14728006497068635074 * tmp_10546[0], 14728006497068635074 * tmp_10546[1], 14728006497068635074 * tmp_10546[2]];
    signal tmp_12117[3] <== [tmp_12114[0] + tmp_12116[0], tmp_12114[1] + tmp_12116[1], tmp_12114[2] + tmp_12116[2]];
    signal tmp_12119[3] <== [3628630143580991743 * tmp_10558[0], 3628630143580991743 * tmp_10558[1], 3628630143580991743 * tmp_10558[2]];
    signal tmp_12120[3] <== [tmp_12117[0] + tmp_12119[0], tmp_12117[1] + tmp_12119[1], tmp_12117[2] + tmp_12119[2]];
    signal tmp_12122[3] <== [5578449105840657656 * tmp_10569[0], 5578449105840657656 * tmp_10569[1], 5578449105840657656 * tmp_10569[2]];
    signal tmp_12123[3] <== [tmp_12120[0] + tmp_12122[0], tmp_12120[1] + tmp_12122[1], tmp_12120[2] + tmp_12122[2]];
    signal tmp_12125[3] <== [16653803537962124952 * tmp_10581[0], 16653803537962124952 * tmp_10581[1], 16653803537962124952 * tmp_10581[2]];
    signal tmp_12126[3] <== [tmp_12123[0] + tmp_12125[0], tmp_12123[1] + tmp_12125[1], tmp_12123[2] + tmp_12125[2]];
    signal tmp_12128[3] <== [14725357302690445139 * tmp_10593[0], 14725357302690445139 * tmp_10593[1], 14725357302690445139 * tmp_10593[2]];
    signal tmp_12129[3] <== [tmp_12126[0] + tmp_12128[0], tmp_12126[1] + tmp_12128[1], tmp_12126[2] + tmp_12128[2]];
    signal tmp_12131[3] <== [13551115817015094532 * tmp_10605[0], 13551115817015094532 * tmp_10605[1], 13551115817015094532 * tmp_10605[2]];
    signal tmp_12132[3] <== [tmp_12129[0] + tmp_12131[0], tmp_12129[1] + tmp_12131[1], tmp_12129[2] + tmp_12131[2]];
    signal tmp_12134[3] <== [9008401972144753939 * tmp_10616[0], 9008401972144753939 * tmp_10616[1], 9008401972144753939 * tmp_10616[2]];
    signal tmp_12135[3] <== [tmp_12132[0] + tmp_12134[0], tmp_12132[1] + tmp_12134[1], tmp_12132[2] + tmp_12134[2]];
    signal tmp_12137[3] <== [16204753907425352648 * tmp_10628[0], 16204753907425352648 * tmp_10628[1], 16204753907425352648 * tmp_10628[2]];
    signal tmp_12138[3] <== [tmp_12135[0] + tmp_12137[0], tmp_12135[1] + tmp_12137[1], tmp_12135[2] + tmp_12137[2]];
    signal tmp_12140[3] <== [7366465902859006190 * tmp_10640[0], 7366465902859006190 * tmp_10640[1], 7366465902859006190 * tmp_10640[2]];
    signal tmp_12141[3] <== [tmp_12138[0] + tmp_12140[0], tmp_12138[1] + tmp_12140[1], tmp_12138[2] + tmp_12140[2]];
    signal tmp_12143[3] <== [15743243826515774520 * tmp_10652[0], 15743243826515774520 * tmp_10652[1], 15743243826515774520 * tmp_10652[2]];
    signal tmp_12144[3] <== [tmp_12141[0] + tmp_12143[0], tmp_12141[1] + tmp_12143[1], tmp_12141[2] + tmp_12143[2]];
    signal tmp_12146[3] <== [7700354089512536567 * tmp_10664[0], 7700354089512536567 * tmp_10664[1], 7700354089512536567 * tmp_10664[2]];
    signal tmp_12147[3] <== [tmp_12144[0] + tmp_12146[0], tmp_12144[1] + tmp_12146[1], tmp_12144[2] + tmp_12146[2]];
    signal tmp_12149[3] <== [1582298761054006274 * tmp_10676[0], 1582298761054006274 * tmp_10676[1], 1582298761054006274 * tmp_10676[2]];
    signal tmp_12150[3] <== [tmp_12147[0] + tmp_12149[0], tmp_12147[1] + tmp_12149[1], tmp_12147[2] + tmp_12149[2]];
    signal tmp_12152[3] <== [17611042948119270535 * tmp_10688[0], 17611042948119270535 * tmp_10688[1], 17611042948119270535 * tmp_10688[2]];
    signal tmp_12153[3] <== [tmp_12150[0] + tmp_12152[0], tmp_12150[1] + tmp_12152[1], tmp_12150[2] + tmp_12152[2]];
    signal tmp_12155[3] <== [18408474278105675815 * tmp_10700[0], 18408474278105675815 * tmp_10700[1], 18408474278105675815 * tmp_10700[2]];
    signal tmp_12156[3] <== [tmp_12153[0] + tmp_12155[0], tmp_12153[1] + tmp_12155[1], tmp_12153[2] + tmp_12155[2]];
    signal tmp_12157[3] <== [evals[147][0] - tmp_12156[0], evals[147][1] - tmp_12156[1], evals[147][2] - tmp_12156[2]];
    signal tmp_12158[3] <== CMul()(tmp_9161, tmp_12157);
    signal tmp_12159[3] <== [tmp_12109[0] + tmp_12158[0], tmp_12109[1] + tmp_12158[1], tmp_12109[2] + tmp_12158[2]];
    signal tmp_12160[3] <== CMul()(challengeQ, tmp_12159);
    signal tmp_12162[3] <== [8024158912291823733 * tmp_10525[0], 8024158912291823733 * tmp_10525[1], 8024158912291823733 * tmp_10525[2]];
    signal tmp_12164[3] <== [7624167817472438013 * tmp_10535[0], 7624167817472438013 * tmp_10535[1], 7624167817472438013 * tmp_10535[2]];
    signal tmp_12165[3] <== [tmp_12162[0] + tmp_12164[0], tmp_12162[1] + tmp_12164[1], tmp_12162[2] + tmp_12164[2]];
    signal tmp_12167[3] <== [14208464745322834529 * tmp_10546[0], 14208464745322834529 * tmp_10546[1], 14208464745322834529 * tmp_10546[2]];
    signal tmp_12168[3] <== [tmp_12165[0] + tmp_12167[0], tmp_12165[1] + tmp_12167[1], tmp_12165[2] + tmp_12167[2]];
    signal tmp_12170[3] <== [812825342001813471 * tmp_10558[0], 812825342001813471 * tmp_10558[1], 812825342001813471 * tmp_10558[2]];
    signal tmp_12171[3] <== [tmp_12168[0] + tmp_12170[0], tmp_12168[1] + tmp_12170[1], tmp_12168[2] + tmp_12170[2]];
    signal tmp_12173[3] <== [3628630143580991743 * tmp_10569[0], 3628630143580991743 * tmp_10569[1], 3628630143580991743 * tmp_10569[2]];
    signal tmp_12174[3] <== [tmp_12171[0] + tmp_12173[0], tmp_12171[1] + tmp_12173[1], tmp_12171[2] + tmp_12173[2]];
    signal tmp_12176[3] <== [17138871444129789158 * tmp_10581[0], 17138871444129789158 * tmp_10581[1], 17138871444129789158 * tmp_10581[2]];
    signal tmp_12177[3] <== [tmp_12174[0] + tmp_12176[0], tmp_12174[1] + tmp_12176[1], tmp_12174[2] + tmp_12176[2]];
    signal tmp_12179[3] <== [12361291211423778677 * tmp_10593[0], 12361291211423778677 * tmp_10593[1], 12361291211423778677 * tmp_10593[2]];
    signal tmp_12180[3] <== [tmp_12177[0] + tmp_12179[0], tmp_12177[1] + tmp_12179[1], tmp_12177[2] + tmp_12179[2]];
    signal tmp_12182[3] <== [9707166089812971789 * tmp_10605[0], 9707166089812971789 * tmp_10605[1], 9707166089812971789 * tmp_10605[2]];
    signal tmp_12183[3] <== [tmp_12180[0] + tmp_12182[0], tmp_12180[1] + tmp_12182[1], tmp_12180[2] + tmp_12182[2]];
    signal tmp_12185[3] <== [16073073456050006575 * tmp_10616[0], 16073073456050006575 * tmp_10616[1], 16073073456050006575 * tmp_10616[2]];
    signal tmp_12186[3] <== [tmp_12183[0] + tmp_12185[0], tmp_12183[1] + tmp_12185[1], tmp_12183[2] + tmp_12185[2]];
    signal tmp_12188[3] <== [16359376528724855414 * tmp_10628[0], 16359376528724855414 * tmp_10628[1], 16359376528724855414 * tmp_10628[2]];
    signal tmp_12189[3] <== [tmp_12186[0] + tmp_12188[0], tmp_12186[1] + tmp_12188[1], tmp_12186[2] + tmp_12188[2]];
    signal tmp_12191[3] <== [2680431002152629139 * tmp_10640[0], 2680431002152629139 * tmp_10640[1], 2680431002152629139 * tmp_10640[2]];
    signal tmp_12192[3] <== [tmp_12189[0] + tmp_12191[0], tmp_12189[1] + tmp_12191[1], tmp_12189[2] + tmp_12191[2]];
    signal tmp_12194[3] <== [15175373777650174966 * tmp_10652[0], 15175373777650174966 * tmp_10652[1], 15175373777650174966 * tmp_10652[2]];
    signal tmp_12195[3] <== [tmp_12192[0] + tmp_12194[0], tmp_12192[1] + tmp_12194[1], tmp_12192[2] + tmp_12194[2]];
    signal tmp_12197[3] <== [7595102312128106020 * tmp_10664[0], 7595102312128106020 * tmp_10664[1], 7595102312128106020 * tmp_10664[2]];
    signal tmp_12198[3] <== [tmp_12195[0] + tmp_12197[0], tmp_12195[1] + tmp_12197[1], tmp_12195[2] + tmp_12197[2]];
    signal tmp_12200[3] <== [17245004066890521540 * tmp_10676[0], 17245004066890521540 * tmp_10676[1], 17245004066890521540 * tmp_10676[2]];
    signal tmp_12201[3] <== [tmp_12198[0] + tmp_12200[0], tmp_12198[1] + tmp_12200[1], tmp_12198[2] + tmp_12200[2]];
    signal tmp_12203[3] <== [14226032251042260572 * tmp_10688[0], 14226032251042260572 * tmp_10688[1], 14226032251042260572 * tmp_10688[2]];
    signal tmp_12204[3] <== [tmp_12201[0] + tmp_12203[0], tmp_12201[1] + tmp_12203[1], tmp_12201[2] + tmp_12203[2]];
    signal tmp_12206[3] <== [10290793996632644659 * tmp_10700[0], 10290793996632644659 * tmp_10700[1], 10290793996632644659 * tmp_10700[2]];
    signal tmp_12207[3] <== [tmp_12204[0] + tmp_12206[0], tmp_12204[1] + tmp_12206[1], tmp_12204[2] + tmp_12206[2]];
    signal tmp_12208[3] <== [evals[148][0] - tmp_12207[0], evals[148][1] - tmp_12207[1], evals[148][2] - tmp_12207[2]];
    signal tmp_12209[3] <== CMul()(tmp_9161, tmp_12208);
    signal tmp_12210[3] <== [tmp_12160[0] + tmp_12209[0], tmp_12160[1] + tmp_12209[1], tmp_12160[2] + tmp_12209[2]];
    signal tmp_12211[3] <== CMul()(challengeQ, tmp_12210);
    signal tmp_12213[3] <== [9638737643153160474 * tmp_10525[0], 9638737643153160474 * tmp_10525[1], 9638737643153160474 * tmp_10525[2]];
    signal tmp_12215[3] <== [16887945882953670782 * tmp_10535[0], 16887945882953670782 * tmp_10535[1], 16887945882953670782 * tmp_10535[2]];
    signal tmp_12216[3] <== [tmp_12213[0] + tmp_12215[0], tmp_12213[1] + tmp_12215[1], tmp_12213[2] + tmp_12215[2]];
    signal tmp_12218[3] <== [11859617026755975295 * tmp_10546[0], 11859617026755975295 * tmp_10546[1], 11859617026755975295 * tmp_10546[2]];
    signal tmp_12219[3] <== [tmp_12216[0] + tmp_12218[0], tmp_12216[1] + tmp_12218[1], tmp_12216[2] + tmp_12218[2]];
    signal tmp_12221[3] <== [14208464745322834529 * tmp_10558[0], 14208464745322834529 * tmp_10558[1], 14208464745322834529 * tmp_10558[2]];
    signal tmp_12222[3] <== [tmp_12219[0] + tmp_12221[0], tmp_12219[1] + tmp_12221[1], tmp_12219[2] + tmp_12221[2]];
    signal tmp_12224[3] <== [14728006497068635074 * tmp_10569[0], 14728006497068635074 * tmp_10569[1], 14728006497068635074 * tmp_10569[2]];
    signal tmp_12225[3] <== [tmp_12222[0] + tmp_12224[0], tmp_12222[1] + tmp_12224[1], tmp_12222[2] + tmp_12224[2]];
    signal tmp_12227[3] <== [7063454713635591383 * tmp_10581[0], 7063454713635591383 * tmp_10581[1], 7063454713635591383 * tmp_10581[2]];
    signal tmp_12228[3] <== [tmp_12225[0] + tmp_12227[0], tmp_12225[1] + tmp_12227[1], tmp_12225[2] + tmp_12227[2]];
    signal tmp_12230[3] <== [10449762191515258658 * tmp_10593[0], 10449762191515258658 * tmp_10593[1], 10449762191515258658 * tmp_10593[2]];
    signal tmp_12231[3] <== [tmp_12228[0] + tmp_12230[0], tmp_12228[1] + tmp_12230[1], tmp_12228[2] + tmp_12230[2]];
    signal tmp_12233[3] <== [5269705790554862459 * tmp_10605[0], 5269705790554862459 * tmp_10605[1], 5269705790554862459 * tmp_10605[2]];
    signal tmp_12234[3] <== [tmp_12231[0] + tmp_12233[0], tmp_12231[1] + tmp_12233[1], tmp_12231[2] + tmp_12233[2]];
    signal tmp_12236[3] <== [10542619009725154399 * tmp_10616[0], 10542619009725154399 * tmp_10616[1], 10542619009725154399 * tmp_10616[2]];
    signal tmp_12237[3] <== [tmp_12234[0] + tmp_12236[0], tmp_12234[1] + tmp_12236[1], tmp_12234[2] + tmp_12236[2]];
    signal tmp_12239[3] <== [13074116889890804777 * tmp_10628[0], 13074116889890804777 * tmp_10628[1], 13074116889890804777 * tmp_10628[2]];
    signal tmp_12240[3] <== [tmp_12237[0] + tmp_12239[0], tmp_12237[1] + tmp_12239[1], tmp_12237[2] + tmp_12239[2]];
    signal tmp_12242[3] <== [5823573138716677219 * tmp_10640[0], 5823573138716677219 * tmp_10640[1], 5823573138716677219 * tmp_10640[2]];
    signal tmp_12243[3] <== [tmp_12240[0] + tmp_12242[0], tmp_12240[1] + tmp_12242[1], tmp_12240[2] + tmp_12242[2]];
    signal tmp_12245[3] <== [4800348490672825001 * tmp_10652[0], 4800348490672825001 * tmp_10652[1], 4800348490672825001 * tmp_10652[2]];
    signal tmp_12246[3] <== [tmp_12243[0] + tmp_12245[0], tmp_12243[1] + tmp_12245[1], tmp_12243[2] + tmp_12245[2]];
    signal tmp_12248[3] <== [15832068917340433117 * tmp_10664[0], 15832068917340433117 * tmp_10664[1], 15832068917340433117 * tmp_10664[2]];
    signal tmp_12249[3] <== [tmp_12246[0] + tmp_12248[0], tmp_12246[1] + tmp_12248[1], tmp_12246[2] + tmp_12248[2]];
    signal tmp_12251[3] <== [10872182090012722023 * tmp_10676[0], 10872182090012722023 * tmp_10676[1], 10872182090012722023 * tmp_10676[2]];
    signal tmp_12252[3] <== [tmp_12249[0] + tmp_12251[0], tmp_12249[1] + tmp_12251[1], tmp_12249[2] + tmp_12251[2]];
    signal tmp_12254[3] <== [7891351762273827007 * tmp_10688[0], 7891351762273827007 * tmp_10688[1], 7891351762273827007 * tmp_10688[2]];
    signal tmp_12255[3] <== [tmp_12252[0] + tmp_12254[0], tmp_12252[1] + tmp_12254[1], tmp_12252[2] + tmp_12254[2]];
    signal tmp_12257[3] <== [8593802027805519400 * tmp_10700[0], 8593802027805519400 * tmp_10700[1], 8593802027805519400 * tmp_10700[2]];
    signal tmp_12258[3] <== [tmp_12255[0] + tmp_12257[0], tmp_12255[1] + tmp_12257[1], tmp_12255[2] + tmp_12257[2]];
    signal tmp_12259[3] <== [evals[149][0] - tmp_12258[0], evals[149][1] - tmp_12258[1], evals[149][2] - tmp_12258[2]];
    signal tmp_12260[3] <== CMul()(tmp_9161, tmp_12259);
    signal tmp_12261[3] <== [tmp_12211[0] + tmp_12260[0], tmp_12211[1] + tmp_12260[1], tmp_12211[2] + tmp_12260[2]];
    signal tmp_12262[3] <== CMul()(challengeQ, tmp_12261);
    signal tmp_12264[3] <== [15737625755013050897 * tmp_10525[0], 15737625755013050897 * tmp_10525[1], 15737625755013050897 * tmp_10525[2]];
    signal tmp_12266[3] <== [6288199092898320258 * tmp_10535[0], 6288199092898320258 * tmp_10535[1], 6288199092898320258 * tmp_10535[2]];
    signal tmp_12267[3] <== [tmp_12264[0] + tmp_12266[0], tmp_12264[1] + tmp_12266[1], tmp_12264[2] + tmp_12266[2]];
    signal tmp_12269[3] <== [16887945882953670782 * tmp_10546[0], 16887945882953670782 * tmp_10546[1], 16887945882953670782 * tmp_10546[2]];
    signal tmp_12270[3] <== [tmp_12267[0] + tmp_12269[0], tmp_12267[1] + tmp_12269[1], tmp_12267[2] + tmp_12269[2]];
    signal tmp_12272[3] <== [7624167817472438013 * tmp_10558[0], 7624167817472438013 * tmp_10558[1], 7624167817472438013 * tmp_10558[2]];
    signal tmp_12273[3] <== [tmp_12270[0] + tmp_12272[0], tmp_12270[1] + tmp_12272[1], tmp_12270[2] + tmp_12272[2]];
    signal tmp_12275[3] <== [3252585408322091368 * tmp_10569[0], 3252585408322091368 * tmp_10569[1], 3252585408322091368 * tmp_10569[2]];
    signal tmp_12276[3] <== [tmp_12273[0] + tmp_12275[0], tmp_12273[1] + tmp_12275[1], tmp_12273[2] + tmp_12275[2]];
    signal tmp_12278[3] <== [2525078022317723924 * tmp_10581[0], 2525078022317723924 * tmp_10581[1], 2525078022317723924 * tmp_10581[2]];
    signal tmp_12279[3] <== [tmp_12276[0] + tmp_12278[0], tmp_12276[1] + tmp_12278[1], tmp_12276[2] + tmp_12278[2]];
    signal tmp_12281[3] <== [14505687488637668062 * tmp_10593[0], 14505687488637668062 * tmp_10593[1], 14505687488637668062 * tmp_10593[2]];
    signal tmp_12282[3] <== [tmp_12279[0] + tmp_12281[0], tmp_12279[1] + tmp_12281[1], tmp_12279[2] + tmp_12281[2]];
    signal tmp_12284[3] <== [2360357678915503787 * tmp_10605[0], 2360357678915503787 * tmp_10605[1], 2360357678915503787 * tmp_10605[2]];
    signal tmp_12285[3] <== [tmp_12282[0] + tmp_12284[0], tmp_12282[1] + tmp_12284[1], tmp_12282[2] + tmp_12284[2]];
    signal tmp_12287[3] <== [16995162224826066164 * tmp_10616[0], 16995162224826066164 * tmp_10616[1], 16995162224826066164 * tmp_10616[2]];
    signal tmp_12288[3] <== [tmp_12285[0] + tmp_12287[0], tmp_12285[1] + tmp_12287[1], tmp_12285[2] + tmp_12287[2]];
    signal tmp_12290[3] <== [3473665614188209191 * tmp_10628[0], 3473665614188209191 * tmp_10628[1], 3473665614188209191 * tmp_10628[2]];
    signal tmp_12291[3] <== [tmp_12288[0] + tmp_12290[0], tmp_12288[1] + tmp_12290[1], tmp_12288[2] + tmp_12290[2]];
    signal tmp_12293[3] <== [6296951922972871295 * tmp_10640[0], 6296951922972871295 * tmp_10640[1], 6296951922972871295 * tmp_10640[2]];
    signal tmp_12294[3] <== [tmp_12291[0] + tmp_12293[0], tmp_12291[1] + tmp_12293[1], tmp_12291[2] + tmp_12293[2]];
    signal tmp_12296[3] <== [1843095456038194143 * tmp_10652[0], 1843095456038194143 * tmp_10652[1], 1843095456038194143 * tmp_10652[2]];
    signal tmp_12297[3] <== [tmp_12294[0] + tmp_12296[0], tmp_12294[1] + tmp_12296[1], tmp_12294[2] + tmp_12296[2]];
    signal tmp_12299[3] <== [3224171887150947390 * tmp_10664[0], 3224171887150947390 * tmp_10664[1], 3224171887150947390 * tmp_10664[2]];
    signal tmp_12300[3] <== [tmp_12297[0] + tmp_12299[0], tmp_12297[1] + tmp_12299[1], tmp_12297[2] + tmp_12299[2]];
    signal tmp_12302[3] <== [8855639374359979570 * tmp_10676[0], 8855639374359979570 * tmp_10676[1], 8855639374359979570 * tmp_10676[2]];
    signal tmp_12303[3] <== [tmp_12300[0] + tmp_12302[0], tmp_12300[1] + tmp_12302[1], tmp_12300[2] + tmp_12302[2]];
    signal tmp_12305[3] <== [17521923232000336525 * tmp_10688[0], 17521923232000336525 * tmp_10688[1], 17521923232000336525 * tmp_10688[2]];
    signal tmp_12306[3] <== [tmp_12303[0] + tmp_12305[0], tmp_12303[1] + tmp_12305[1], tmp_12303[2] + tmp_12305[2]];
    signal tmp_12308[3] <== [4068529056800825848 * tmp_10700[0], 4068529056800825848 * tmp_10700[1], 4068529056800825848 * tmp_10700[2]];
    signal tmp_12309[3] <== [tmp_12306[0] + tmp_12308[0], tmp_12306[1] + tmp_12308[1], tmp_12306[2] + tmp_12308[2]];
    signal tmp_12310[3] <== [evals[150][0] - tmp_12309[0], evals[150][1] - tmp_12309[1], evals[150][2] - tmp_12309[2]];
    signal tmp_12311[3] <== CMul()(tmp_9161, tmp_12310);
    signal tmp_12312[3] <== [tmp_12262[0] + tmp_12311[0], tmp_12262[1] + tmp_12311[1], tmp_12262[2] + tmp_12311[2]];
    signal tmp_12313[3] <== CMul()(challengeQ, tmp_12312);
    signal tmp_12314[3] <== [evals[99][0] - evals[83][0], evals[99][1] - evals[83][1], evals[99][2] - evals[83][2]];
    signal tmp_12315[3] <== CMul()(tmp_9163, tmp_12314);
    signal tmp_12316[3] <== [tmp_12313[0] + tmp_12315[0], tmp_12313[1] + tmp_12315[1], tmp_12313[2] + tmp_12315[2]];
    signal tmp_12317[3] <== CMul()(challengeQ, tmp_12316);
    signal tmp_12318[3] <== [tmp_10519[0] + 8111169986958259496, tmp_10519[1], tmp_10519[2]];
    signal tmp_12319[3] <== [8669423204701701828 * evals[84][0], 8669423204701701828 * evals[84][1], 8669423204701701828 * evals[84][2]];
    signal tmp_12320[3] <== [tmp_12318[0] + tmp_12319[0], tmp_12318[1] + tmp_12319[1], tmp_12318[2] + tmp_12319[2]];
    signal tmp_12321[3] <== [11218806973091596230 * evals[85][0], 11218806973091596230 * evals[85][1], 11218806973091596230 * evals[85][2]];
    signal tmp_12322[3] <== [tmp_12320[0] + tmp_12321[0], tmp_12320[1] + tmp_12321[1], tmp_12320[2] + tmp_12321[2]];
    signal tmp_12323[3] <== [15262373047951269198 * evals[86][0], 15262373047951269198 * evals[86][1], 15262373047951269198 * evals[86][2]];
    signal tmp_12324[3] <== [tmp_12322[0] + tmp_12323[0], tmp_12322[1] + tmp_12323[1], tmp_12322[2] + tmp_12323[2]];
    signal tmp_12325[3] <== [1029025466846055820 * evals[87][0], 1029025466846055820 * evals[87][1], 1029025466846055820 * evals[87][2]];
    signal tmp_12326[3] <== [tmp_12324[0] + tmp_12325[0], tmp_12324[1] + tmp_12325[1], tmp_12324[2] + tmp_12325[2]];
    signal tmp_12327[3] <== [4974512540632580574 * evals[88][0], 4974512540632580574 * evals[88][1], 4974512540632580574 * evals[88][2]];
    signal tmp_12328[3] <== [tmp_12326[0] + tmp_12327[0], tmp_12326[1] + tmp_12327[1], tmp_12326[2] + tmp_12327[2]];
    signal tmp_12329[3] <== [9182455040231019020 * evals[89][0], 9182455040231019020 * evals[89][1], 9182455040231019020 * evals[89][2]];
    signal tmp_12330[3] <== [tmp_12328[0] + tmp_12329[0], tmp_12328[1] + tmp_12329[1], tmp_12328[2] + tmp_12329[2]];
    signal tmp_12331[3] <== [6370260108846926617 * evals[90][0], 6370260108846926617 * evals[90][1], 6370260108846926617 * evals[90][2]];
    signal tmp_12332[3] <== [tmp_12330[0] + tmp_12331[0], tmp_12330[1] + tmp_12331[1], tmp_12330[2] + tmp_12331[2]];
    signal tmp_12333[3] <== [13260731818101756883 * evals[91][0], 13260731818101756883 * evals[91][1], 13260731818101756883 * evals[91][2]];
    signal tmp_12334[3] <== [tmp_12332[0] + tmp_12333[0], tmp_12332[1] + tmp_12333[1], tmp_12332[2] + tmp_12333[2]];
    signal tmp_12335[3] <== [7710124073693265452 * evals[92][0], 7710124073693265452 * evals[92][1], 7710124073693265452 * evals[92][2]];
    signal tmp_12336[3] <== [tmp_12334[0] + tmp_12335[0], tmp_12334[1] + tmp_12335[1], tmp_12334[2] + tmp_12335[2]];
    signal tmp_12337[3] <== [15423058398499793658 * evals[93][0], 15423058398499793658 * evals[93][1], 15423058398499793658 * evals[93][2]];
    signal tmp_12338[3] <== [tmp_12336[0] + tmp_12337[0], tmp_12336[1] + tmp_12337[1], tmp_12336[2] + tmp_12337[2]];
    signal tmp_12339[3] <== [5050813558536356212 * evals[94][0], 5050813558536356212 * evals[94][1], 5050813558536356212 * evals[94][2]];
    signal tmp_12340[3] <== [tmp_12338[0] + tmp_12339[0], tmp_12338[1] + tmp_12339[1], tmp_12338[2] + tmp_12339[2]];
    signal tmp_12341[3] <== [7839532349345071368 * evals[95][0], 7839532349345071368 * evals[95][1], 7839532349345071368 * evals[95][2]];
    signal tmp_12342[3] <== [tmp_12340[0] + tmp_12341[0], tmp_12340[1] + tmp_12341[1], tmp_12340[2] + tmp_12341[2]];
    signal tmp_12343[3] <== [13946888797241243093 * evals[96][0], 13946888797241243093 * evals[96][1], 13946888797241243093 * evals[96][2]];
    signal tmp_12344[3] <== [tmp_12342[0] + tmp_12343[0], tmp_12342[1] + tmp_12343[1], tmp_12342[2] + tmp_12343[2]];
    signal tmp_12345[3] <== [1910481884837990028 * evals[97][0], 1910481884837990028 * evals[97][1], 1910481884837990028 * evals[97][2]];
    signal tmp_12346[3] <== [tmp_12344[0] + tmp_12345[0], tmp_12344[1] + tmp_12345[1], tmp_12344[2] + tmp_12345[2]];
    signal tmp_12347[3] <== [2968031798288424027 * evals[98][0], 2968031798288424027 * evals[98][1], 2968031798288424027 * evals[98][2]];
    signal tmp_12348[3] <== [tmp_12346[0] + tmp_12347[0], tmp_12346[1] + tmp_12347[1], tmp_12346[2] + tmp_12347[2]];
    signal tmp_12349[3] <== [tmp_12348[0] * 1, tmp_12348[1] * 1, tmp_12348[2] * 1];
    signal tmp_12350[3] <== [evals[100][0] - tmp_12349[0], evals[100][1] - tmp_12349[1], evals[100][2] - tmp_12349[2]];
    signal tmp_12351[3] <== CMul()(tmp_9163, tmp_12350);
    signal tmp_12352[3] <== [tmp_12317[0] + tmp_12351[0], tmp_12317[1] + tmp_12351[1], tmp_12317[2] + tmp_12351[2]];
    signal tmp_12353[3] <== CMul()(challengeQ, tmp_12352);
    tmp_12354 <== [tmp_10529[0] + 9086063564225054097, tmp_10529[1], tmp_10529[2]];
    signal tmp_12356[3] <== [tmp_12318[0] * 9426926376014476876, tmp_12318[1] * 9426926376014476876, tmp_12318[2] * 9426926376014476876];
    signal tmp_12357[3] <== [evals[84][0] + tmp_12356[0], evals[84][1] + tmp_12356[1], evals[84][2] + tmp_12356[2]];
    signal tmp_12358[3] <== [4769303276973324461 * tmp_12357[0], 4769303276973324461 * tmp_12357[1], 4769303276973324461 * tmp_12357[2]];
    signal tmp_12359[3] <== [tmp_12354[0] + tmp_12358[0], tmp_12354[1] + tmp_12358[1], tmp_12354[2] + tmp_12358[2]];
    signal tmp_12361[3] <== [tmp_12318[0] * 6636074244061498014, tmp_12318[1] * 6636074244061498014, tmp_12318[2] * 6636074244061498014];
    signal tmp_12362[3] <== [evals[85][0] + tmp_12361[0], evals[85][1] + tmp_12361[1], evals[85][2] + tmp_12361[2]];
    signal tmp_12363[3] <== [1784300138210151456 * tmp_12362[0], 1784300138210151456 * tmp_12362[1], 1784300138210151456 * tmp_12362[2]];
    signal tmp_12364[3] <== [tmp_12359[0] + tmp_12363[0], tmp_12359[1] + tmp_12363[1], tmp_12359[2] + tmp_12363[2]];
    signal tmp_12366[3] <== [tmp_12318[0] * 10264791899337939894, tmp_12318[1] * 10264791899337939894, tmp_12318[2] * 10264791899337939894];
    signal tmp_12367[3] <== [evals[86][0] + tmp_12366[0], evals[86][1] + tmp_12366[1], evals[86][2] + tmp_12366[2]];
    signal tmp_12368[3] <== [12324891265250247020 * tmp_12367[0], 12324891265250247020 * tmp_12367[1], 12324891265250247020 * tmp_12367[2]];
    signal tmp_12369[3] <== [tmp_12364[0] + tmp_12368[0], tmp_12364[1] + tmp_12368[1], tmp_12364[2] + tmp_12368[2]];
    signal tmp_12371[3] <== [tmp_12318[0] * 7084964930551891640, tmp_12318[1] * 7084964930551891640, tmp_12318[2] * 7084964930551891640];
    signal tmp_12372[3] <== [evals[87][0] + tmp_12371[0], evals[87][1] + tmp_12371[1], evals[87][2] + tmp_12371[2]];
    signal tmp_12373[3] <== [12174743494559533987 * tmp_12372[0], 12174743494559533987 * tmp_12372[1], 12174743494559533987 * tmp_12372[2]];
    signal tmp_12374[3] <== [tmp_12369[0] + tmp_12373[0], tmp_12369[1] + tmp_12373[1], tmp_12369[2] + tmp_12373[2]];
    signal tmp_12376[3] <== [tmp_12318[0] * 80533688234975742, tmp_12318[1] * 80533688234975742, tmp_12318[2] * 80533688234975742];
    signal tmp_12377[3] <== [evals[88][0] + tmp_12376[0], evals[88][1] + tmp_12376[1], evals[88][2] + tmp_12376[2]];
    signal tmp_12378[3] <== [8147334966940084329 * tmp_12377[0], 8147334966940084329 * tmp_12377[1], 8147334966940084329 * tmp_12377[2]];
    signal tmp_12379[3] <== [tmp_12374[0] + tmp_12378[0], tmp_12374[1] + tmp_12378[1], tmp_12374[2] + tmp_12378[2]];
    signal tmp_12381[3] <== [tmp_12318[0] * 14562009953595175769, tmp_12318[1] * 14562009953595175769, tmp_12318[2] * 14562009953595175769];
    signal tmp_12382[3] <== [evals[89][0] + tmp_12381[0], evals[89][1] + tmp_12381[1], evals[89][2] + tmp_12381[2]];
    signal tmp_12383[3] <== [16397648617955447197 * tmp_12382[0], 16397648617955447197 * tmp_12382[1], 16397648617955447197 * tmp_12382[2]];
    signal tmp_12384[3] <== [tmp_12379[0] + tmp_12383[0], tmp_12379[1] + tmp_12383[1], tmp_12379[2] + tmp_12383[2]];
    signal tmp_12386[3] <== [tmp_12318[0] * 326023128915630064, tmp_12318[1] * 326023128915630064, tmp_12318[2] * 326023128915630064];
    signal tmp_12387[3] <== [evals[90][0] + tmp_12386[0], evals[90][1] + tmp_12386[1], evals[90][2] + tmp_12386[2]];
    signal tmp_12388[3] <== [6037293384829534708 * tmp_12387[0], 6037293384829534708 * tmp_12387[1], 6037293384829534708 * tmp_12387[2]];
    signal tmp_12389[3] <== [tmp_12384[0] + tmp_12388[0], tmp_12384[1] + tmp_12388[1], tmp_12384[2] + tmp_12388[2]];
    signal tmp_12391[3] <== [tmp_12318[0] * 11214699076825281994, tmp_12318[1] * 11214699076825281994, tmp_12318[2] * 11214699076825281994];
    signal tmp_12392[3] <== [evals[91][0] + tmp_12391[0], evals[91][1] + tmp_12391[1], evals[91][2] + tmp_12391[2]];
    signal tmp_12393[3] <== [16763181795158500563 * tmp_12392[0], 16763181795158500563 * tmp_12392[1], 16763181795158500563 * tmp_12392[2]];
    signal tmp_12394[3] <== [tmp_12389[0] + tmp_12393[0], tmp_12389[1] + tmp_12393[1], tmp_12389[2] + tmp_12393[2]];
    signal tmp_12396[3] <== [tmp_12318[0] * 12885915700130647489, tmp_12318[1] * 12885915700130647489, tmp_12318[2] * 12885915700130647489];
    signal tmp_12397[3] <== [evals[92][0] + tmp_12396[0], evals[92][1] + tmp_12396[1], evals[92][2] + tmp_12396[2]];
    signal tmp_12398[3] <== [10796483624199473587 * tmp_12397[0], 10796483624199473587 * tmp_12397[1], 10796483624199473587 * tmp_12397[2]];
    signal tmp_12399[3] <== [tmp_12394[0] + tmp_12398[0], tmp_12394[1] + tmp_12398[1], tmp_12394[2] + tmp_12398[2]];
    signal tmp_12401[3] <== [tmp_12318[0] * 2933396631256123616, tmp_12318[1] * 2933396631256123616, tmp_12318[2] * 2933396631256123616];
    signal tmp_12402[3] <== [evals[93][0] + tmp_12401[0], evals[93][1] + tmp_12401[1], evals[93][2] + tmp_12401[2]];
    signal tmp_12403[3] <== [9857846934271670796 * tmp_12402[0], 9857846934271670796 * tmp_12402[1], 9857846934271670796 * tmp_12402[2]];
    signal tmp_12404[3] <== [tmp_12399[0] + tmp_12403[0], tmp_12399[1] + tmp_12403[1], tmp_12399[2] + tmp_12403[2]];
    signal tmp_12406[3] <== [tmp_12318[0] * 254842111671127473, tmp_12318[1] * 254842111671127473, tmp_12318[2] * 254842111671127473];
    signal tmp_12407[3] <== [evals[94][0] + tmp_12406[0], evals[94][1] + tmp_12406[1], evals[94][2] + tmp_12406[2]];
    signal tmp_12408[3] <== [1711698425798584877 * tmp_12407[0], 1711698425798584877 * tmp_12407[1], 1711698425798584877 * tmp_12407[2]];
    signal tmp_12409[3] <== [tmp_12404[0] + tmp_12408[0], tmp_12404[1] + tmp_12408[1], tmp_12404[2] + tmp_12408[2]];
    signal tmp_12411[3] <== [tmp_12318[0] * 15960520344296920510, tmp_12318[1] * 15960520344296920510, tmp_12318[2] * 15960520344296920510];
    signal tmp_12412[3] <== [evals[95][0] + tmp_12411[0], evals[95][1] + tmp_12411[1], evals[95][2] + tmp_12411[2]];
    signal tmp_12413[3] <== [14317037178658024350 * tmp_12412[0], 14317037178658024350 * tmp_12412[1], 14317037178658024350 * tmp_12412[2]];
    signal tmp_12414[3] <== [tmp_12409[0] + tmp_12413[0], tmp_12409[1] + tmp_12413[1], tmp_12409[2] + tmp_12413[2]];
    signal tmp_12416[3] <== [tmp_12318[0] * 472404601837029917, tmp_12318[1] * 472404601837029917, tmp_12318[2] * 472404601837029917];
    tmp_12417 <== [evals[96][0] + tmp_12416[0], evals[96][1] + tmp_12416[1], evals[96][2] + tmp_12416[2]];
    signal tmp_12418[3] <== [4727210660209380191 * tmp_12417[0], 4727210660209380191 * tmp_12417[1], 4727210660209380191 * tmp_12417[2]];
    signal tmp_12419[3] <== [tmp_12414[0] + tmp_12418[0], tmp_12414[1] + tmp_12418[1], tmp_12414[2] + tmp_12418[2]];
    signal tmp_12421[3] <== [tmp_12318[0] * 5788109941016889361, tmp_12318[1] * 5788109941016889361, tmp_12318[2] * 5788109941016889361];
    tmp_12422 <== [evals[97][0] + tmp_12421[0], evals[97][1] + tmp_12421[1], evals[97][2] + tmp_12421[2]];
    signal tmp_12423[3] <== [4043120510603784989 * tmp_12422[0], 4043120510603784989 * tmp_12422[1], 4043120510603784989 * tmp_12422[2]];
    signal tmp_12424[3] <== [tmp_12419[0] + tmp_12423[0], tmp_12419[1] + tmp_12423[1], tmp_12419[2] + tmp_12423[2]];
    signal tmp_12426[3] <== [tmp_12318[0] * 15962084738108841624, tmp_12318[1] * 15962084738108841624, tmp_12318[2] * 15962084738108841624];
    tmp_12427 <== [evals[98][0] + tmp_12426[0], evals[98][1] + tmp_12426[1], evals[98][2] + tmp_12426[2]];
    signal tmp_12428[3] <== [6414177337645848225 * tmp_12427[0], 6414177337645848225 * tmp_12427[1], 6414177337645848225 * tmp_12427[2]];
    signal tmp_12429[3] <== [tmp_12424[0] + tmp_12428[0], tmp_12424[1] + tmp_12428[1], tmp_12424[2] + tmp_12428[2]];
    signal tmp_12430[3] <== [tmp_12429[0] * 1, tmp_12429[1] * 1, tmp_12429[2] * 1];
    signal tmp_12431[3] <== [evals[101][0] - tmp_12430[0], evals[101][1] - tmp_12430[1], evals[101][2] - tmp_12430[2]];
    signal tmp_12432[3] <== CMul()(tmp_9163, tmp_12431);
    signal tmp_12433[3] <== [tmp_12353[0] + tmp_12432[0], tmp_12353[1] + tmp_12432[1], tmp_12353[2] + tmp_12432[2]];
    tmp_12434 <== CMul()(challengeQ, tmp_12433);
    tmp_12435 <== [tmp_10540[0] + 11571460868925408324, tmp_10540[1], tmp_10540[2]];
    signal tmp_12437[3] <== [tmp_12354[0] * 8808574163396808168, tmp_12354[1] * 8808574163396808168, tmp_12354[2] * 8808574163396808168];
    tmp_12438 <== [tmp_12357[0] + tmp_12437[0], tmp_12357[1] + tmp_12437[1], tmp_12357[2] + tmp_12437[2]];
    signal tmp_12439[3] <== [1514946122061859950 * tmp_12438[0], 1514946122061859950 * tmp_12438[1], 1514946122061859950 * tmp_12438[2]];
    signal tmp_12440[3] <== [tmp_12435[0] + tmp_12439[0], tmp_12435[1] + tmp_12439[1], tmp_12435[2] + tmp_12439[2]];
    signal tmp_12442[3] <== [tmp_12354[0] * 2762212780120881912, tmp_12354[1] * 2762212780120881912, tmp_12354[2] * 2762212780120881912];
    tmp_12443 <== [tmp_12362[0] + tmp_12442[0], tmp_12362[1] + tmp_12442[1], tmp_12362[2] + tmp_12442[2]];
    signal tmp_12444[3] <== [1955599193479362137 * tmp_12443[0], 1955599193479362137 * tmp_12443[1], 1955599193479362137 * tmp_12443[2]];
    signal tmp_12445[3] <== [tmp_12440[0] + tmp_12444[0], tmp_12440[1] + tmp_12444[1], tmp_12440[2] + tmp_12444[2]];
    signal tmp_12447[3] <== [tmp_12354[0] * 17951693132780115109, tmp_12354[1] * 17951693132780115109, tmp_12354[2] * 17951693132780115109];
    tmp_12448 <== [tmp_12367[0] + tmp_12447[0], tmp_12367[1] + tmp_12447[1], tmp_12367[2] + tmp_12447[2]];
    signal tmp_12449[3] <== [11647134996302165546 * tmp_12448[0], 11647134996302165546 * tmp_12448[1], 11647134996302165546 * tmp_12448[2]];
    signal tmp_12450[3] <== [tmp_12445[0] + tmp_12449[0], tmp_12445[1] + tmp_12449[1], tmp_12445[2] + tmp_12449[2]];
    signal tmp_12452[3] <== [tmp_12354[0] * 9961063800679125687, tmp_12354[1] * 9961063800679125687, tmp_12354[2] * 9961063800679125687];
    tmp_12453 <== [tmp_12372[0] + tmp_12452[0], tmp_12372[1] + tmp_12452[1], tmp_12372[2] + tmp_12452[2]];
    signal tmp_12454[3] <== [9465069334725615251 * tmp_12453[0], 9465069334725615251 * tmp_12453[1], 9465069334725615251 * tmp_12453[2]];
    signal tmp_12455[3] <== [tmp_12450[0] + tmp_12454[0], tmp_12450[1] + tmp_12454[1], tmp_12450[2] + tmp_12454[2]];
    signal tmp_12457[3] <== [tmp_12354[0] * 6634243339771071547, tmp_12354[1] * 6634243339771071547, tmp_12354[2] * 6634243339771071547];
    tmp_12458 <== [tmp_12377[0] + tmp_12457[0], tmp_12377[1] + tmp_12457[1], tmp_12377[2] + tmp_12457[2]];
    signal tmp_12459[3] <== [7347175090247155478 * tmp_12458[0], 7347175090247155478 * tmp_12458[1], 7347175090247155478 * tmp_12458[2]];
    signal tmp_12460[3] <== [tmp_12455[0] + tmp_12459[0], tmp_12455[1] + tmp_12459[1], tmp_12455[2] + tmp_12459[2]];
    signal tmp_12462[3] <== [tmp_12354[0] * 3730820316519738416, tmp_12354[1] * 3730820316519738416, tmp_12354[2] * 3730820316519738416];
    tmp_12463 <== [tmp_12382[0] + tmp_12462[0], tmp_12382[1] + tmp_12462[1], tmp_12382[2] + tmp_12462[2]];
    signal tmp_12464[3] <== [929610534534885463 * tmp_12463[0], 929610534534885463 * tmp_12463[1], 929610534534885463 * tmp_12463[2]];
    signal tmp_12465[3] <== [tmp_12460[0] + tmp_12464[0], tmp_12460[1] + tmp_12464[1], tmp_12460[2] + tmp_12464[2]];
    signal tmp_12467[3] <== [tmp_12354[0] * 6178874009151806010, tmp_12354[1] * 6178874009151806010, tmp_12354[2] * 6178874009151806010];
    tmp_12468 <== [tmp_12387[0] + tmp_12467[0], tmp_12387[1] + tmp_12467[1], tmp_12387[2] + tmp_12467[2]];
    signal tmp_12469[3] <== [10037966693745455517 * tmp_12468[0], 10037966693745455517 * tmp_12468[1], 10037966693745455517 * tmp_12468[2]];
    signal tmp_12470[3] <== [tmp_12465[0] + tmp_12469[0], tmp_12465[1] + tmp_12469[1], tmp_12465[2] + tmp_12469[2]];
    signal tmp_12472[3] <== [tmp_12354[0] * 151426042370448923, tmp_12354[1] * 151426042370448923, tmp_12354[2] * 151426042370448923];
    tmp_12473 <== [tmp_12392[0] + tmp_12472[0], tmp_12392[1] + tmp_12472[1], tmp_12392[2] + tmp_12472[2]];
    signal tmp_12474[3] <== [9847255082563255028 * tmp_12473[0], 9847255082563255028 * tmp_12473[1], 9847255082563255028 * tmp_12473[2]];
    signal tmp_12475[3] <== [tmp_12470[0] + tmp_12474[0], tmp_12470[1] + tmp_12474[1], tmp_12470[2] + tmp_12474[2]];
    signal tmp_12477[3] <== [tmp_12354[0] * 17283463014546786540, tmp_12354[1] * 17283463014546786540, tmp_12354[2] * 17283463014546786540];
    tmp_12478 <== [tmp_12397[0] + tmp_12477[0], tmp_12397[1] + tmp_12477[1], tmp_12397[2] + tmp_12477[2]];
    signal tmp_12479[3] <== [7498640650554676844 * tmp_12478[0], 7498640650554676844 * tmp_12478[1], 7498640650554676844 * tmp_12478[2]];
    signal tmp_12480[3] <== [tmp_12475[0] + tmp_12479[0], tmp_12475[1] + tmp_12479[1], tmp_12475[2] + tmp_12479[2]];
    signal tmp_12482[3] <== [tmp_12354[0] * 2716553630958499013, tmp_12354[1] * 2716553630958499013, tmp_12354[2] * 2716553630958499013];
    tmp_12483 <== [tmp_12402[0] + tmp_12482[0], tmp_12402[1] + tmp_12482[1], tmp_12402[2] + tmp_12482[2]];
    signal tmp_12484[3] <== [9423900797682883949 * tmp_12483[0], 9423900797682883949 * tmp_12483[1], 9423900797682883949 * tmp_12483[2]];
    signal tmp_12485[3] <== [tmp_12480[0] + tmp_12484[0], tmp_12480[1] + tmp_12484[1], tmp_12480[2] + tmp_12484[2]];
    signal tmp_12487[3] <== [tmp_12354[0] * 17502570475072947141, tmp_12354[1] * 17502570475072947141, tmp_12354[2] * 17502570475072947141];
    tmp_12488 <== [tmp_12407[0] + tmp_12487[0], tmp_12407[1] + tmp_12487[1], tmp_12407[2] + tmp_12487[2]];
    signal tmp_12489[3] <== [16088378537411909982 * tmp_12488[0], 16088378537411909982 * tmp_12488[1], 16088378537411909982 * tmp_12488[2]];
    signal tmp_12490[3] <== [tmp_12485[0] + tmp_12489[0], tmp_12485[1] + tmp_12489[1], tmp_12485[2] + tmp_12489[2]];
    signal tmp_12492[3] <== [tmp_12354[0] * 7861949515711048574, tmp_12354[1] * 7861949515711048574, tmp_12354[2] * 7861949515711048574];
    tmp_12493 <== [tmp_12412[0] + tmp_12492[0], tmp_12412[1] + tmp_12492[1], tmp_12412[2] + tmp_12492[2]];
    signal tmp_12494[3] <== [8768480041621124523 * tmp_12493[0], 8768480041621124523 * tmp_12493[1], 8768480041621124523 * tmp_12493[2]];
    tmp_12495 <== [tmp_12490[0] + tmp_12494[0], tmp_12490[1] + tmp_12494[1], tmp_12490[2] + tmp_12494[2]];
    tmp_12497 <== [tmp_12354[0] * 16107154087629722508, tmp_12354[1] * 16107154087629722508, tmp_12354[2] * 16107154087629722508];
}

template VerifyEvaluationsChunks4_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_9163[3];
    signal input tmp_10552[3];
    signal input tmp_10563[3];
    signal input tmp_10575[3];
    signal input tmp_10587[3];
    signal input tmp_10599[3];
    signal input tmp_10610[3];
    signal input tmp_10622[3];
    signal input tmp_10634[3];
    signal input tmp_10646[3];
    signal input tmp_10658[3];
    signal input tmp_10670[3];
    signal input tmp_10682[3];
    signal input tmp_10694[3];
    signal input tmp_12354[3];
    signal input tmp_12417[3];
    signal input tmp_12422[3];
    signal input tmp_12427[3];
    signal input tmp_12434[3];
    signal input tmp_12435[3];
    signal input tmp_12438[3];
    signal input tmp_12443[3];
    signal input tmp_12448[3];
    signal input tmp_12453[3];
    signal input tmp_12458[3];
    signal input tmp_12463[3];
    signal input tmp_12468[3];
    signal input tmp_12473[3];
    signal input tmp_12478[3];
    signal input tmp_12483[3];
    signal input tmp_12488[3];
    signal input tmp_12493[3];
    signal input tmp_12495[3];
    signal input tmp_12497[3];

    signal output tmp_13573[3];
    signal output tmp_13641[3];
    signal output tmp_13646[3];
    signal output tmp_13653[3];
    signal output tmp_13658[3];
    signal output tmp_13661[3];
    signal output tmp_13666[3];
    signal output tmp_13671[3];
    signal output tmp_13676[3];
    signal output tmp_13681[3];
    signal output tmp_13686[3];
    signal output tmp_13691[3];
    signal output tmp_13696[3];
    signal output tmp_13701[3];
    signal output tmp_13706[3];
    signal output tmp_13711[3];
    signal output tmp_13716[3];
    signal output tmp_13718[3];
    signal output tmp_13721[3];
    signal tmp_12498[3] <== [tmp_12417[0] + tmp_12497[0], tmp_12417[1] + tmp_12497[1], tmp_12417[2] + tmp_12497[2]];
    signal tmp_12499[3] <== [11676865538804947598 * tmp_12498[0], 11676865538804947598 * tmp_12498[1], 11676865538804947598 * tmp_12498[2]];
    signal tmp_12500[3] <== [tmp_12495[0] + tmp_12499[0], tmp_12495[1] + tmp_12499[1], tmp_12495[2] + tmp_12499[2]];
    signal tmp_12502[3] <== [tmp_12354[0] * 16444612208872340826, tmp_12354[1] * 16444612208872340826, tmp_12354[2] * 16444612208872340826];
    signal tmp_12503[3] <== [tmp_12422[0] + tmp_12502[0], tmp_12422[1] + tmp_12502[1], tmp_12422[2] + tmp_12502[2]];
    signal tmp_12504[3] <== [3273168334829907114 * tmp_12503[0], 3273168334829907114 * tmp_12503[1], 3273168334829907114 * tmp_12503[2]];
    signal tmp_12505[3] <== [tmp_12500[0] + tmp_12504[0], tmp_12500[1] + tmp_12504[1], tmp_12500[2] + tmp_12504[2]];
    signal tmp_12507[3] <== [tmp_12354[0] * 16364131105478793415, tmp_12354[1] * 16364131105478793415, tmp_12354[2] * 16364131105478793415];
    signal tmp_12508[3] <== [tmp_12427[0] + tmp_12507[0], tmp_12427[1] + tmp_12507[1], tmp_12427[2] + tmp_12507[2]];
    signal tmp_12509[3] <== [15744851004062856881 * tmp_12508[0], 15744851004062856881 * tmp_12508[1], 15744851004062856881 * tmp_12508[2]];
    signal tmp_12510[3] <== [tmp_12505[0] + tmp_12509[0], tmp_12505[1] + tmp_12509[1], tmp_12505[2] + tmp_12509[2]];
    signal tmp_12511[3] <== [tmp_12510[0] * 1, tmp_12510[1] * 1, tmp_12510[2] * 1];
    signal tmp_12512[3] <== [evals[102][0] - tmp_12511[0], evals[102][1] - tmp_12511[1], evals[102][2] - tmp_12511[2]];
    signal tmp_12513[3] <== CMul()(tmp_9163, tmp_12512);
    signal tmp_12514[3] <== [tmp_12434[0] + tmp_12513[0], tmp_12434[1] + tmp_12513[1], tmp_12434[2] + tmp_12513[2]];
    signal tmp_12515[3] <== CMul()(challengeQ, tmp_12514);
    signal tmp_12516[3] <== [tmp_10552[0] + 2855538343746738937, tmp_10552[1], tmp_10552[2]];
    signal tmp_12518[3] <== [tmp_12435[0] * 5047845749023635250, tmp_12435[1] * 5047845749023635250, tmp_12435[2] * 5047845749023635250];
    signal tmp_12519[3] <== [tmp_12438[0] + tmp_12518[0], tmp_12438[1] + tmp_12518[1], tmp_12438[2] + tmp_12518[2]];
    signal tmp_12520[3] <== [9944788307894914133 * tmp_12519[0], 9944788307894914133 * tmp_12519[1], 9944788307894914133 * tmp_12519[2]];
    signal tmp_12521[3] <== [tmp_12516[0] + tmp_12520[0], tmp_12516[1] + tmp_12520[1], tmp_12516[2] + tmp_12520[2]];
    signal tmp_12523[3] <== [tmp_12435[0] * 8434851656909622170, tmp_12435[1] * 8434851656909622170, tmp_12435[2] * 8434851656909622170];
    signal tmp_12524[3] <== [tmp_12443[0] + tmp_12523[0], tmp_12443[1] + tmp_12523[1], tmp_12443[2] + tmp_12523[2]];
    signal tmp_12525[3] <== [10771689630961936167 * tmp_12524[0], 10771689630961936167 * tmp_12524[1], 10771689630961936167 * tmp_12524[2]];
    signal tmp_12526[3] <== [tmp_12521[0] + tmp_12525[0], tmp_12521[1] + tmp_12525[1], tmp_12521[2] + tmp_12525[2]];
    signal tmp_12528[3] <== [tmp_12435[0] * 12922301140896192660, tmp_12435[1] * 12922301140896192660, tmp_12435[2] * 12922301140896192660];
    signal tmp_12529[3] <== [tmp_12448[0] + tmp_12528[0], tmp_12448[1] + tmp_12528[1], tmp_12448[2] + tmp_12528[2]];
    signal tmp_12530[3] <== [14652833956370356968 * tmp_12529[0], 14652833956370356968 * tmp_12529[1], 14652833956370356968 * tmp_12529[2]];
    signal tmp_12531[3] <== [tmp_12526[0] + tmp_12530[0], tmp_12526[1] + tmp_12530[1], tmp_12526[2] + tmp_12530[2]];
    signal tmp_12533[3] <== [tmp_12435[0] * 14175851439986065151, tmp_12435[1] * 14175851439986065151, tmp_12435[2] * 14175851439986065151];
    signal tmp_12534[3] <== [tmp_12453[0] + tmp_12533[0], tmp_12453[1] + tmp_12533[1], tmp_12453[2] + tmp_12533[2]];
    signal tmp_12535[3] <== [2762159236071112109 * tmp_12534[0], 2762159236071112109 * tmp_12534[1], 2762159236071112109 * tmp_12534[2]];
    signal tmp_12536[3] <== [tmp_12531[0] + tmp_12535[0], tmp_12531[1] + tmp_12535[1], tmp_12531[2] + tmp_12535[2]];
    signal tmp_12538[3] <== [tmp_12435[0] * 7308936446108866250, tmp_12435[1] * 7308936446108866250, tmp_12435[2] * 7308936446108866250];
    signal tmp_12539[3] <== [tmp_12458[0] + tmp_12538[0], tmp_12458[1] + tmp_12538[1], tmp_12458[2] + tmp_12538[2]];
    signal tmp_12540[3] <== [3599708650553142297 * tmp_12539[0], 3599708650553142297 * tmp_12539[1], 3599708650553142297 * tmp_12539[2]];
    signal tmp_12541[3] <== [tmp_12536[0] + tmp_12540[0], tmp_12536[1] + tmp_12540[1], tmp_12536[2] + tmp_12540[2]];
    signal tmp_12543[3] <== [tmp_12435[0] * 1728188381301272932, tmp_12435[1] * 1728188381301272932, tmp_12435[2] * 1728188381301272932];
    signal tmp_12544[3] <== [tmp_12463[0] + tmp_12543[0], tmp_12463[1] + tmp_12543[1], tmp_12463[2] + tmp_12543[2]];
    signal tmp_12545[3] <== [1524268929892782060 * tmp_12544[0], 1524268929892782060 * tmp_12544[1], 1524268929892782060 * tmp_12544[2]];
    signal tmp_12546[3] <== [tmp_12541[0] + tmp_12545[0], tmp_12541[1] + tmp_12545[1], tmp_12541[2] + tmp_12545[2]];
    signal tmp_12548[3] <== [tmp_12435[0] * 14315181752530606135, tmp_12435[1] * 14315181752530606135, tmp_12435[2] * 14315181752530606135];
    signal tmp_12549[3] <== [tmp_12468[0] + tmp_12548[0], tmp_12468[1] + tmp_12548[1], tmp_12468[2] + tmp_12548[2]];
    signal tmp_12550[3] <== [13329681733957674057 * tmp_12549[0], 13329681733957674057 * tmp_12549[1], 13329681733957674057 * tmp_12549[2]];
    signal tmp_12551[3] <== [tmp_12546[0] + tmp_12550[0], tmp_12546[1] + tmp_12550[1], tmp_12546[2] + tmp_12550[2]];
    signal tmp_12553[3] <== [tmp_12435[0] * 6491065457156410005, tmp_12435[1] * 6491065457156410005, tmp_12435[2] * 6491065457156410005];
    signal tmp_12554[3] <== [tmp_12473[0] + tmp_12553[0], tmp_12473[1] + tmp_12553[1], tmp_12473[2] + tmp_12553[2]];
    signal tmp_12555[3] <== [5797791035413508849 * tmp_12554[0], 5797791035413508849 * tmp_12554[1], 5797791035413508849 * tmp_12554[2]];
    signal tmp_12556[3] <== [tmp_12551[0] + tmp_12555[0], tmp_12551[1] + tmp_12555[1], tmp_12551[2] + tmp_12555[2]];
    signal tmp_12558[3] <== [tmp_12435[0] * 4175603234939884368, tmp_12435[1] * 4175603234939884368, tmp_12435[2] * 4175603234939884368];
    signal tmp_12559[3] <== [tmp_12478[0] + tmp_12558[0], tmp_12478[1] + tmp_12558[1], tmp_12478[2] + tmp_12558[2]];
    signal tmp_12560[3] <== [7515887947763182304 * tmp_12559[0], 7515887947763182304 * tmp_12559[1], 7515887947763182304 * tmp_12559[2]];
    signal tmp_12561[3] <== [tmp_12556[0] + tmp_12560[0], tmp_12556[1] + tmp_12560[1], tmp_12556[2] + tmp_12560[2]];
    signal tmp_12563[3] <== [tmp_12435[0] * 1239436087648515890, tmp_12435[1] * 1239436087648515890, tmp_12435[2] * 1239436087648515890];
    signal tmp_12564[3] <== [tmp_12483[0] + tmp_12563[0], tmp_12483[1] + tmp_12563[1], tmp_12483[2] + tmp_12563[2]];
    signal tmp_12565[3] <== [16010783934711236424 * tmp_12564[0], 16010783934711236424 * tmp_12564[1], 16010783934711236424 * tmp_12564[2]];
    signal tmp_12566[3] <== [tmp_12561[0] + tmp_12565[0], tmp_12561[1] + tmp_12565[1], tmp_12561[2] + tmp_12565[2]];
    signal tmp_12568[3] <== [tmp_12435[0] * 7915700559009623006, tmp_12435[1] * 7915700559009623006, tmp_12435[2] * 7915700559009623006];
    signal tmp_12569[3] <== [tmp_12488[0] + tmp_12568[0], tmp_12488[1] + tmp_12568[1], tmp_12488[2] + tmp_12568[2]];
    signal tmp_12570[3] <== [4190904235179964140 * tmp_12569[0], 4190904235179964140 * tmp_12569[1], 4190904235179964140 * tmp_12569[2]];
    signal tmp_12571[3] <== [tmp_12566[0] + tmp_12570[0], tmp_12566[1] + tmp_12570[1], tmp_12566[2] + tmp_12570[2]];
    signal tmp_12573[3] <== [tmp_12435[0] * 14767908959930482966, tmp_12435[1] * 14767908959930482966, tmp_12435[2] * 14767908959930482966];
    signal tmp_12574[3] <== [tmp_12493[0] + tmp_12573[0], tmp_12493[1] + tmp_12573[1], tmp_12493[2] + tmp_12573[2]];
    signal tmp_12575[3] <== [14187663514905569572 * tmp_12574[0], 14187663514905569572 * tmp_12574[1], 14187663514905569572 * tmp_12574[2]];
    signal tmp_12576[3] <== [tmp_12571[0] + tmp_12575[0], tmp_12571[1] + tmp_12575[1], tmp_12571[2] + tmp_12575[2]];
    signal tmp_12578[3] <== [tmp_12435[0] * 13675360068971462516, tmp_12435[1] * 13675360068971462516, tmp_12435[2] * 13675360068971462516];
    signal tmp_12579[3] <== [tmp_12498[0] + tmp_12578[0], tmp_12498[1] + tmp_12578[1], tmp_12498[2] + tmp_12578[2]];
    signal tmp_12580[3] <== [2777661931104600049 * tmp_12579[0], 2777661931104600049 * tmp_12579[1], 2777661931104600049 * tmp_12579[2]];
    signal tmp_12581[3] <== [tmp_12576[0] + tmp_12580[0], tmp_12576[1] + tmp_12580[1], tmp_12576[2] + tmp_12580[2]];
    signal tmp_12583[3] <== [tmp_12435[0] * 15228155835675626627, tmp_12435[1] * 15228155835675626627, tmp_12435[2] * 15228155835675626627];
    signal tmp_12584[3] <== [tmp_12503[0] + tmp_12583[0], tmp_12503[1] + tmp_12583[1], tmp_12503[2] + tmp_12583[2]];
    signal tmp_12585[3] <== [12517723326974261293 * tmp_12584[0], 12517723326974261293 * tmp_12584[1], 12517723326974261293 * tmp_12584[2]];
    signal tmp_12586[3] <== [tmp_12581[0] + tmp_12585[0], tmp_12581[1] + tmp_12585[1], tmp_12581[2] + tmp_12585[2]];
    signal tmp_12588[3] <== [tmp_12435[0] * 4870139207699630301, tmp_12435[1] * 4870139207699630301, tmp_12435[2] * 4870139207699630301];
    signal tmp_12589[3] <== [tmp_12508[0] + tmp_12588[0], tmp_12508[1] + tmp_12588[1], tmp_12508[2] + tmp_12588[2]];
    signal tmp_12590[3] <== [4677019694129625027 * tmp_12589[0], 4677019694129625027 * tmp_12589[1], 4677019694129625027 * tmp_12589[2]];
    signal tmp_12591[3] <== [tmp_12586[0] + tmp_12590[0], tmp_12586[1] + tmp_12590[1], tmp_12586[2] + tmp_12590[2]];
    signal tmp_12592[3] <== [tmp_12591[0] * 1, tmp_12591[1] * 1, tmp_12591[2] * 1];
    signal tmp_12593[3] <== [evals[103][0] - tmp_12592[0], evals[103][1] - tmp_12592[1], evals[103][2] - tmp_12592[2]];
    signal tmp_12594[3] <== CMul()(tmp_9163, tmp_12593);
    signal tmp_12595[3] <== [tmp_12515[0] + tmp_12594[0], tmp_12515[1] + tmp_12594[1], tmp_12515[2] + tmp_12594[2]];
    signal tmp_12596[3] <== CMul()(challengeQ, tmp_12595);
    signal tmp_12597[3] <== [tmp_10563[0] + 14270783606042555221, tmp_10563[1], tmp_10563[2]];
    signal tmp_12599[3] <== [tmp_12516[0] * 1665021560400291078, tmp_12516[1] * 1665021560400291078, tmp_12516[2] * 1665021560400291078];
    signal tmp_12600[3] <== [tmp_12519[0] + tmp_12599[0], tmp_12519[1] + tmp_12599[1], tmp_12519[2] + tmp_12599[2]];
    signal tmp_12601[3] <== [11160115342809132588 * tmp_12600[0], 11160115342809132588 * tmp_12600[1], 11160115342809132588 * tmp_12600[2]];
    signal tmp_12602[3] <== [tmp_12597[0] + tmp_12601[0], tmp_12597[1] + tmp_12601[1], tmp_12597[2] + tmp_12601[2]];
    signal tmp_12604[3] <== [tmp_12516[0] * 6249272897953199859, tmp_12516[1] * 6249272897953199859, tmp_12516[2] * 6249272897953199859];
    signal tmp_12605[3] <== [tmp_12524[0] + tmp_12604[0], tmp_12524[1] + tmp_12604[1], tmp_12524[2] + tmp_12604[2]];
    signal tmp_12606[3] <== [15052831855997744501 * tmp_12605[0], 15052831855997744501 * tmp_12605[1], 15052831855997744501 * tmp_12605[2]];
    signal tmp_12607[3] <== [tmp_12602[0] + tmp_12606[0], tmp_12602[1] + tmp_12606[1], tmp_12602[2] + tmp_12606[2]];
    signal tmp_12609[3] <== [tmp_12516[0] * 7260683056416925359, tmp_12516[1] * 7260683056416925359, tmp_12516[2] * 7260683056416925359];
    signal tmp_12610[3] <== [tmp_12529[0] + tmp_12609[0], tmp_12529[1] + tmp_12609[1], tmp_12529[2] + tmp_12609[2]];
    signal tmp_12611[3] <== [15739451022504817202 * tmp_12610[0], 15739451022504817202 * tmp_12610[1], 15739451022504817202 * tmp_12610[2]];
    signal tmp_12612[3] <== [tmp_12607[0] + tmp_12611[0], tmp_12607[1] + tmp_12611[1], tmp_12607[2] + tmp_12611[2]];
    signal tmp_12614[3] <== [tmp_12516[0] * 11363141252265098044, tmp_12516[1] * 11363141252265098044, tmp_12516[2] * 11363141252265098044];
    signal tmp_12615[3] <== [tmp_12534[0] + tmp_12614[0], tmp_12534[1] + tmp_12614[1], tmp_12534[2] + tmp_12614[2]];
    signal tmp_12616[3] <== [9671855798646808088 * tmp_12615[0], 9671855798646808088 * tmp_12615[1], 9671855798646808088 * tmp_12615[2]];
    signal tmp_12617[3] <== [tmp_12612[0] + tmp_12616[0], tmp_12612[1] + tmp_12616[1], tmp_12612[2] + tmp_12616[2]];
    signal tmp_12619[3] <== [tmp_12516[0] * 8012231655085569456, tmp_12516[1] * 8012231655085569456, tmp_12516[2] * 8012231655085569456];
    signal tmp_12620[3] <== [tmp_12539[0] + tmp_12619[0], tmp_12539[1] + tmp_12619[1], tmp_12539[2] + tmp_12619[2]];
    signal tmp_12621[3] <== [3993080706899827970 * tmp_12620[0], 3993080706899827970 * tmp_12620[1], 3993080706899827970 * tmp_12620[2]];
    signal tmp_12622[3] <== [tmp_12617[0] + tmp_12621[0], tmp_12617[1] + tmp_12621[1], tmp_12617[2] + tmp_12621[2]];
    signal tmp_12624[3] <== [tmp_12516[0] * 13251561609088352545, tmp_12516[1] * 13251561609088352545, tmp_12516[2] * 13251561609088352545];
    signal tmp_12625[3] <== [tmp_12544[0] + tmp_12624[0], tmp_12544[1] + tmp_12624[1], tmp_12544[2] + tmp_12624[2]];
    signal tmp_12626[3] <== [17244704481465347826 * tmp_12625[0], 17244704481465347826 * tmp_12625[1], 17244704481465347826 * tmp_12625[2]];
    signal tmp_12627[3] <== [tmp_12622[0] + tmp_12626[0], tmp_12622[1] + tmp_12626[1], tmp_12622[2] + tmp_12626[2]];
    signal tmp_12629[3] <== [tmp_12516[0] * 15473520463141521750, tmp_12516[1] * 15473520463141521750, tmp_12516[2] * 15473520463141521750];
    signal tmp_12630[3] <== [tmp_12549[0] + tmp_12629[0], tmp_12549[1] + tmp_12629[1], tmp_12549[2] + tmp_12629[2]];
    signal tmp_12631[3] <== [6669884464780361495 * tmp_12630[0], 6669884464780361495 * tmp_12630[1], 6669884464780361495 * tmp_12630[2]];
    signal tmp_12632[3] <== [tmp_12627[0] + tmp_12631[0], tmp_12627[1] + tmp_12631[1], tmp_12627[2] + tmp_12631[2]];
    signal tmp_12634[3] <== [tmp_12516[0] * 11088129738238123803, tmp_12516[1] * 11088129738238123803, tmp_12516[2] * 11088129738238123803];
    signal tmp_12635[3] <== [tmp_12554[0] + tmp_12634[0], tmp_12554[1] + tmp_12634[1], tmp_12554[2] + tmp_12634[2]];
    signal tmp_12636[3] <== [14600555312004421850 * tmp_12635[0], 14600555312004421850 * tmp_12635[1], 14600555312004421850 * tmp_12635[2]];
    signal tmp_12637[3] <== [tmp_12632[0] + tmp_12636[0], tmp_12632[1] + tmp_12636[1], tmp_12632[2] + tmp_12636[2]];
    signal tmp_12639[3] <== [tmp_12516[0] * 1281025690561882185, tmp_12516[1] * 1281025690561882185, tmp_12516[2] * 1281025690561882185];
    signal tmp_12640[3] <== [tmp_12559[0] + tmp_12639[0], tmp_12559[1] + tmp_12639[1], tmp_12559[2] + tmp_12639[2]];
    signal tmp_12641[3] <== [12885797354206706070 * tmp_12640[0], 12885797354206706070 * tmp_12640[1], 12885797354206706070 * tmp_12640[2]];
    signal tmp_12642[3] <== [tmp_12637[0] + tmp_12641[0], tmp_12637[1] + tmp_12641[1], tmp_12637[2] + tmp_12641[2]];
    signal tmp_12644[3] <== [tmp_12516[0] * 8655250499426779365, tmp_12516[1] * 8655250499426779365, tmp_12516[2] * 8655250499426779365];
    signal tmp_12645[3] <== [tmp_12564[0] + tmp_12644[0], tmp_12564[1] + tmp_12644[1], tmp_12564[2] + tmp_12644[2]];
    signal tmp_12646[3] <== [4680617225974204348 * tmp_12645[0], 4680617225974204348 * tmp_12645[1], 4680617225974204348 * tmp_12645[2]];
    signal tmp_12647[3] <== [tmp_12642[0] + tmp_12646[0], tmp_12642[1] + tmp_12646[1], tmp_12642[2] + tmp_12646[2]];
    signal tmp_12649[3] <== [tmp_12516[0] * 1321126874272112769, tmp_12516[1] * 1321126874272112769, tmp_12516[2] * 1321126874272112769];
    signal tmp_12650[3] <== [tmp_12569[0] + tmp_12649[0], tmp_12569[1] + tmp_12649[1], tmp_12569[2] + tmp_12649[2]];
    signal tmp_12651[3] <== [6736135874728218206 * tmp_12650[0], 6736135874728218206 * tmp_12650[1], 6736135874728218206 * tmp_12650[2]];
    signal tmp_12652[3] <== [tmp_12647[0] + tmp_12651[0], tmp_12647[1] + tmp_12651[1], tmp_12647[2] + tmp_12651[2]];
    signal tmp_12654[3] <== [tmp_12516[0] * 11480921900711623217, tmp_12516[1] * 11480921900711623217, tmp_12516[2] * 11480921900711623217];
    signal tmp_12655[3] <== [tmp_12574[0] + tmp_12654[0], tmp_12574[1] + tmp_12654[1], tmp_12574[2] + tmp_12654[2]];
    signal tmp_12656[3] <== [11543984753251934208 * tmp_12655[0], 11543984753251934208 * tmp_12655[1], 11543984753251934208 * tmp_12655[2]];
    signal tmp_12657[3] <== [tmp_12652[0] + tmp_12656[0], tmp_12652[1] + tmp_12656[1], tmp_12652[2] + tmp_12656[2]];
    signal tmp_12659[3] <== [tmp_12516[0] * 8534648876771497041, tmp_12516[1] * 8534648876771497041, tmp_12516[2] * 8534648876771497041];
    signal tmp_12660[3] <== [tmp_12579[0] + tmp_12659[0], tmp_12579[1] + tmp_12659[1], tmp_12579[2] + tmp_12659[2]];
    signal tmp_12661[3] <== [7462450122751074327 * tmp_12660[0], 7462450122751074327 * tmp_12660[1], 7462450122751074327 * tmp_12660[2]];
    signal tmp_12662[3] <== [tmp_12657[0] + tmp_12661[0], tmp_12657[1] + tmp_12661[1], tmp_12657[2] + tmp_12661[2]];
    signal tmp_12664[3] <== [tmp_12516[0] * 2345799062308732216, tmp_12516[1] * 2345799062308732216, tmp_12516[2] * 2345799062308732216];
    signal tmp_12665[3] <== [tmp_12584[0] + tmp_12664[0], tmp_12584[1] + tmp_12664[1], tmp_12584[2] + tmp_12664[2]];
    signal tmp_12666[3] <== [5997876138824023765 * tmp_12665[0], 5997876138824023765 * tmp_12665[1], 5997876138824023765 * tmp_12665[2]];
    signal tmp_12667[3] <== [tmp_12662[0] + tmp_12666[0], tmp_12662[1] + tmp_12666[1], tmp_12662[2] + tmp_12666[2]];
    signal tmp_12669[3] <== [tmp_12516[0] * 8793821150452954879, tmp_12516[1] * 8793821150452954879, tmp_12516[2] * 8793821150452954879];
    signal tmp_12670[3] <== [tmp_12589[0] + tmp_12669[0], tmp_12589[1] + tmp_12669[1], tmp_12589[2] + tmp_12669[2]];
    signal tmp_12671[3] <== [4694625700920314643 * tmp_12670[0], 4694625700920314643 * tmp_12670[1], 4694625700920314643 * tmp_12670[2]];
    signal tmp_12672[3] <== [tmp_12667[0] + tmp_12671[0], tmp_12667[1] + tmp_12671[1], tmp_12667[2] + tmp_12671[2]];
    signal tmp_12673[3] <== [tmp_12672[0] * 1, tmp_12672[1] * 1, tmp_12672[2] * 1];
    signal tmp_12674[3] <== [evals[104][0] - tmp_12673[0], evals[104][1] - tmp_12673[1], evals[104][2] - tmp_12673[2]];
    signal tmp_12675[3] <== CMul()(tmp_9163, tmp_12674);
    signal tmp_12676[3] <== [tmp_12596[0] + tmp_12675[0], tmp_12596[1] + tmp_12675[1], tmp_12596[2] + tmp_12675[2]];
    signal tmp_12677[3] <== CMul()(challengeQ, tmp_12676);
    signal tmp_12678[3] <== [tmp_10575[0] + 18107705583323614825, tmp_10575[1], tmp_10575[2]];
    signal tmp_12680[3] <== [tmp_12597[0] * 13330801039170540745, tmp_12597[1] * 13330801039170540745, tmp_12597[2] * 13330801039170540745];
    signal tmp_12681[3] <== [tmp_12600[0] + tmp_12680[0], tmp_12600[1] + tmp_12680[1], tmp_12600[2] + tmp_12680[2]];
    signal tmp_12682[3] <== [4989688843737548554 * tmp_12681[0], 4989688843737548554 * tmp_12681[1], 4989688843737548554 * tmp_12681[2]];
    signal tmp_12683[3] <== [tmp_12678[0] + tmp_12682[0], tmp_12678[1] + tmp_12682[1], tmp_12678[2] + tmp_12682[2]];
    signal tmp_12685[3] <== [tmp_12597[0] * 9691413076657869892, tmp_12597[1] * 9691413076657869892, tmp_12597[2] * 9691413076657869892];
    signal tmp_12686[3] <== [tmp_12605[0] + tmp_12685[0], tmp_12605[1] + tmp_12685[1], tmp_12605[2] + tmp_12685[2]];
    signal tmp_12687[3] <== [13821113963424948846 * tmp_12686[0], 13821113963424948846 * tmp_12686[1], 13821113963424948846 * tmp_12686[2]];
    signal tmp_12688[3] <== [tmp_12683[0] + tmp_12687[0], tmp_12683[1] + tmp_12687[1], tmp_12683[2] + tmp_12687[2]];
    signal tmp_12690[3] <== [tmp_12597[0] * 18097630306776785732, tmp_12597[1] * 18097630306776785732, tmp_12597[2] * 18097630306776785732];
    signal tmp_12691[3] <== [tmp_12610[0] + tmp_12690[0], tmp_12610[1] + tmp_12690[1], tmp_12610[2] + tmp_12690[2]];
    signal tmp_12692[3] <== [7247806441173525246 * tmp_12691[0], 7247806441173525246 * tmp_12691[1], 7247806441173525246 * tmp_12691[2]];
    signal tmp_12693[3] <== [tmp_12688[0] + tmp_12692[0], tmp_12688[1] + tmp_12692[1], tmp_12688[2] + tmp_12692[2]];
    signal tmp_12695[3] <== [tmp_12597[0] * 5719637133317020548, tmp_12597[1] * 5719637133317020548, tmp_12597[2] * 5719637133317020548];
    signal tmp_12696[3] <== [tmp_12615[0] + tmp_12695[0], tmp_12615[1] + tmp_12695[1], tmp_12615[2] + tmp_12695[2]];
    signal tmp_12697[3] <== [3364393074731955357 * tmp_12696[0], 3364393074731955357 * tmp_12696[1], 3364393074731955357 * tmp_12696[2]];
    signal tmp_12698[3] <== [tmp_12693[0] + tmp_12697[0], tmp_12693[1] + tmp_12697[1], tmp_12693[2] + tmp_12697[2]];
    signal tmp_12700[3] <== [tmp_12597[0] * 4029888846798459355, tmp_12597[1] * 4029888846798459355, tmp_12597[2] * 4029888846798459355];
    signal tmp_12701[3] <== [tmp_12620[0] + tmp_12700[0], tmp_12620[1] + tmp_12700[1], tmp_12620[2] + tmp_12700[2]];
    signal tmp_12702[3] <== [10146937751881836190 * tmp_12701[0], 10146937751881836190 * tmp_12701[1], 10146937751881836190 * tmp_12701[2]];
    signal tmp_12703[3] <== [tmp_12698[0] + tmp_12702[0], tmp_12698[1] + tmp_12702[1], tmp_12698[2] + tmp_12702[2]];
    signal tmp_12705[3] <== [tmp_12597[0] * 18356621544169502341, tmp_12597[1] * 18356621544169502341, tmp_12597[2] * 18356621544169502341];
    signal tmp_12706[3] <== [tmp_12625[0] + tmp_12705[0], tmp_12625[1] + tmp_12705[1], tmp_12625[2] + tmp_12705[2]];
    signal tmp_12707[3] <== [8218226966171798065 * tmp_12706[0], 8218226966171798065 * tmp_12706[1], 8218226966171798065 * tmp_12706[2]];
    signal tmp_12708[3] <== [tmp_12703[0] + tmp_12707[0], tmp_12703[1] + tmp_12707[1], tmp_12703[2] + tmp_12707[2]];
    signal tmp_12710[3] <== [tmp_12597[0] * 3193305903015148388, tmp_12597[1] * 3193305903015148388, tmp_12597[2] * 3193305903015148388];
    signal tmp_12711[3] <== [tmp_12630[0] + tmp_12710[0], tmp_12630[1] + tmp_12710[1], tmp_12630[2] + tmp_12710[2]];
    signal tmp_12712[3] <== [6275505124750343799 * tmp_12711[0], 6275505124750343799 * tmp_12711[1], 6275505124750343799 * tmp_12711[2]];
    signal tmp_12713[3] <== [tmp_12708[0] + tmp_12712[0], tmp_12708[1] + tmp_12712[1], tmp_12708[2] + tmp_12712[2]];
    signal tmp_12715[3] <== [tmp_12597[0] * 13178443391748614197, tmp_12597[1] * 13178443391748614197, tmp_12597[2] * 13178443391748614197];
    signal tmp_12716[3] <== [tmp_12635[0] + tmp_12715[0], tmp_12635[1] + tmp_12715[1], tmp_12635[2] + tmp_12715[2]];
    signal tmp_12717[3] <== [11654032185587768765 * tmp_12716[0], 11654032185587768765 * tmp_12716[1], 11654032185587768765 * tmp_12716[2]];
    signal tmp_12718[3] <== [tmp_12713[0] + tmp_12717[0], tmp_12713[1] + tmp_12717[1], tmp_12713[2] + tmp_12717[2]];
    signal tmp_12720[3] <== [tmp_12597[0] * 14481071249376583383, tmp_12597[1] * 14481071249376583383, tmp_12597[2] * 14481071249376583383];
    signal tmp_12721[3] <== [tmp_12640[0] + tmp_12720[0], tmp_12640[1] + tmp_12720[1], tmp_12640[2] + tmp_12720[2]];
    signal tmp_12722[3] <== [11697929231275965966 * tmp_12721[0], 11697929231275965966 * tmp_12721[1], 11697929231275965966 * tmp_12721[2]];
    signal tmp_12723[3] <== [tmp_12718[0] + tmp_12722[0], tmp_12718[1] + tmp_12722[1], tmp_12718[2] + tmp_12722[2]];
    signal tmp_12725[3] <== [tmp_12597[0] * 13298473208691573067, tmp_12597[1] * 13298473208691573067, tmp_12597[2] * 13298473208691573067];
    signal tmp_12726[3] <== [tmp_12645[0] + tmp_12725[0], tmp_12645[1] + tmp_12725[1], tmp_12645[2] + tmp_12725[2]];
    signal tmp_12727[3] <== [9256045630802942941 * tmp_12726[0], 9256045630802942941 * tmp_12726[1], 9256045630802942941 * tmp_12726[2]];
    signal tmp_12728[3] <== [tmp_12723[0] + tmp_12727[0], tmp_12723[1] + tmp_12727[1], tmp_12723[2] + tmp_12727[2]];
    signal tmp_12730[3] <== [tmp_12597[0] * 5559358937569616668, tmp_12597[1] * 5559358937569616668, tmp_12597[2] * 5559358937569616668];
    signal tmp_12731[3] <== [tmp_12650[0] + tmp_12730[0], tmp_12650[1] + tmp_12730[1], tmp_12650[2] + tmp_12730[2]];
    signal tmp_12732[3] <== [15946532889053694625 * tmp_12731[0], 15946532889053694625 * tmp_12731[1], 15946532889053694625 * tmp_12731[2]];
    signal tmp_12733[3] <== [tmp_12728[0] + tmp_12732[0], tmp_12728[1] + tmp_12732[1], tmp_12728[2] + tmp_12732[2]];
    signal tmp_12735[3] <== [tmp_12597[0] * 5477091296498352424, tmp_12597[1] * 5477091296498352424, tmp_12597[2] * 5477091296498352424];
    signal tmp_12736[3] <== [tmp_12655[0] + tmp_12735[0], tmp_12655[1] + tmp_12735[1], tmp_12655[2] + tmp_12735[2]];
    signal tmp_12737[3] <== [14178555464781916994 * tmp_12736[0], 14178555464781916994 * tmp_12736[1], 14178555464781916994 * tmp_12736[2]];
    signal tmp_12738[3] <== [tmp_12733[0] + tmp_12737[0], tmp_12733[1] + tmp_12737[1], tmp_12733[2] + tmp_12737[2]];
    signal tmp_12740[3] <== [tmp_12597[0] * 4033222126236138968, tmp_12597[1] * 4033222126236138968, tmp_12597[2] * 4033222126236138968];
    signal tmp_12741[3] <== [tmp_12660[0] + tmp_12740[0], tmp_12660[1] + tmp_12740[1], tmp_12660[2] + tmp_12740[2]];
    signal tmp_12742[3] <== [15652577753663522945 * tmp_12741[0], 15652577753663522945 * tmp_12741[1], 15652577753663522945 * tmp_12741[2]];
    signal tmp_12743[3] <== [tmp_12738[0] + tmp_12742[0], tmp_12738[1] + tmp_12742[1], tmp_12738[2] + tmp_12742[2]];
    signal tmp_12745[3] <== [tmp_12597[0] * 12567047098625210052, tmp_12597[1] * 12567047098625210052, tmp_12597[2] * 12567047098625210052];
    signal tmp_12746[3] <== [tmp_12665[0] + tmp_12745[0], tmp_12665[1] + tmp_12745[1], tmp_12665[2] + tmp_12745[2]];
    signal tmp_12747[3] <== [18353293019555892376 * tmp_12746[0], 18353293019555892376 * tmp_12746[1], 18353293019555892376 * tmp_12746[2]];
    signal tmp_12748[3] <== [tmp_12743[0] + tmp_12747[0], tmp_12743[1] + tmp_12747[1], tmp_12743[2] + tmp_12747[2]];
    signal tmp_12750[3] <== [tmp_12597[0] * 5194999122761668273, tmp_12597[1] * 5194999122761668273, tmp_12597[2] * 5194999122761668273];
    signal tmp_12751[3] <== [tmp_12670[0] + tmp_12750[0], tmp_12670[1] + tmp_12750[1], tmp_12670[2] + tmp_12750[2]];
    signal tmp_12752[3] <== [913259685371457409 * tmp_12751[0], 913259685371457409 * tmp_12751[1], 913259685371457409 * tmp_12751[2]];
    signal tmp_12753[3] <== [tmp_12748[0] + tmp_12752[0], tmp_12748[1] + tmp_12752[1], tmp_12748[2] + tmp_12752[2]];
    signal tmp_12754[3] <== [tmp_12753[0] * 1, tmp_12753[1] * 1, tmp_12753[2] * 1];
    signal tmp_12755[3] <== [evals[105][0] - tmp_12754[0], evals[105][1] - tmp_12754[1], evals[105][2] - tmp_12754[2]];
    signal tmp_12756[3] <== CMul()(tmp_9163, tmp_12755);
    signal tmp_12757[3] <== [tmp_12677[0] + tmp_12756[0], tmp_12677[1] + tmp_12756[1], tmp_12677[2] + tmp_12756[2]];
    signal tmp_12758[3] <== CMul()(challengeQ, tmp_12757);
    signal tmp_12759[3] <== [tmp_10587[0] + 14981760075018578211, tmp_10587[1], tmp_10587[2]];
    signal tmp_12761[3] <== [tmp_12678[0] * 2839286413723076283, tmp_12678[1] * 2839286413723076283, tmp_12678[2] * 2839286413723076283];
    signal tmp_12762[3] <== [tmp_12681[0] + tmp_12761[0], tmp_12681[1] + tmp_12761[1], tmp_12681[2] + tmp_12761[2]];
    signal tmp_12763[3] <== [8997952006844223763 * tmp_12762[0], 8997952006844223763 * tmp_12762[1], 8997952006844223763 * tmp_12762[2]];
    signal tmp_12764[3] <== [tmp_12759[0] + tmp_12763[0], tmp_12759[1] + tmp_12763[1], tmp_12759[2] + tmp_12763[2]];
    signal tmp_12766[3] <== [tmp_12678[0] * 8227928784918533600, tmp_12678[1] * 8227928784918533600, tmp_12678[2] * 8227928784918533600];
    signal tmp_12767[3] <== [tmp_12686[0] + tmp_12766[0], tmp_12686[1] + tmp_12766[1], tmp_12686[2] + tmp_12766[2]];
    signal tmp_12768[3] <== [2353549075741107334 * tmp_12767[0], 2353549075741107334 * tmp_12767[1], 2353549075741107334 * tmp_12767[2]];
    signal tmp_12769[3] <== [tmp_12764[0] + tmp_12768[0], tmp_12764[1] + tmp_12768[1], tmp_12764[2] + tmp_12768[2]];
    signal tmp_12771[3] <== [tmp_12678[0] * 6659444090177547341, tmp_12678[1] * 6659444090177547341, tmp_12678[2] * 6659444090177547341];
    signal tmp_12772[3] <== [tmp_12691[0] + tmp_12771[0], tmp_12691[1] + tmp_12771[1], tmp_12691[2] + tmp_12771[2]];
    signal tmp_12773[3] <== [15713716106319331569 * tmp_12772[0], 15713716106319331569 * tmp_12772[1], 15713716106319331569 * tmp_12772[2]];
    signal tmp_12774[3] <== [tmp_12769[0] + tmp_12773[0], tmp_12769[1] + tmp_12773[1], tmp_12769[2] + tmp_12773[2]];
    signal tmp_12776[3] <== [tmp_12678[0] * 7600262705963786111, tmp_12678[1] * 7600262705963786111, tmp_12678[2] * 7600262705963786111];
    signal tmp_12777[3] <== [tmp_12696[0] + tmp_12776[0], tmp_12696[1] + tmp_12776[1], tmp_12696[2] + tmp_12776[2]];
    signal tmp_12778[3] <== [16205952151624674977 * tmp_12777[0], 16205952151624674977 * tmp_12777[1], 16205952151624674977 * tmp_12777[2]];
    signal tmp_12779[3] <== [tmp_12774[0] + tmp_12778[0], tmp_12774[1] + tmp_12778[1], tmp_12774[2] + tmp_12778[2]];
    signal tmp_12781[3] <== [tmp_12678[0] * 14410065364859218948, tmp_12678[1] * 14410065364859218948, tmp_12678[2] * 14410065364859218948];
    signal tmp_12782[3] <== [tmp_12701[0] + tmp_12781[0], tmp_12701[1] + tmp_12781[1], tmp_12701[2] + tmp_12781[2]];
    signal tmp_12783[3] <== [15307752746251443250 * tmp_12782[0], 15307752746251443250 * tmp_12782[1], 15307752746251443250 * tmp_12782[2]];
    signal tmp_12784[3] <== [tmp_12779[0] + tmp_12783[0], tmp_12779[1] + tmp_12783[1], tmp_12779[2] + tmp_12783[2]];
    signal tmp_12786[3] <== [tmp_12678[0] * 5216672470358585092, tmp_12678[1] * 5216672470358585092, tmp_12678[2] * 5216672470358585092];
    signal tmp_12787[3] <== [tmp_12706[0] + tmp_12786[0], tmp_12706[1] + tmp_12786[1], tmp_12706[2] + tmp_12786[2]];
    signal tmp_12788[3] <== [10826281796282448866 * tmp_12787[0], 10826281796282448866 * tmp_12787[1], 10826281796282448866 * tmp_12787[2]];
    signal tmp_12789[3] <== [tmp_12784[0] + tmp_12788[0], tmp_12784[1] + tmp_12788[1], tmp_12784[2] + tmp_12788[2]];
    signal tmp_12791[3] <== [tmp_12678[0] * 10300497666947344511, tmp_12678[1] * 10300497666947344511, tmp_12678[2] * 10300497666947344511];
    signal tmp_12792[3] <== [tmp_12711[0] + tmp_12791[0], tmp_12711[1] + tmp_12791[1], tmp_12711[2] + tmp_12791[2]];
    signal tmp_12793[3] <== [12268463758085038250 * tmp_12792[0], 12268463758085038250 * tmp_12792[1], 12268463758085038250 * tmp_12792[2]];
    signal tmp_12794[3] <== [tmp_12789[0] + tmp_12793[0], tmp_12789[1] + tmp_12793[1], tmp_12789[2] + tmp_12793[2]];
    signal tmp_12796[3] <== [tmp_12678[0] * 10505442326626626850, tmp_12678[1] * 10505442326626626850, tmp_12678[2] * 10505442326626626850];
    signal tmp_12797[3] <== [tmp_12716[0] + tmp_12796[0], tmp_12716[1] + tmp_12796[1], tmp_12716[2] + tmp_12796[2]];
    signal tmp_12798[3] <== [6991270770333944507 * tmp_12797[0], 6991270770333944507 * tmp_12797[1], 6991270770333944507 * tmp_12797[2]];
    signal tmp_12799[3] <== [tmp_12794[0] + tmp_12798[0], tmp_12794[1] + tmp_12798[1], tmp_12794[2] + tmp_12798[2]];
    signal tmp_12801[3] <== [tmp_12678[0] * 2741298027419160918, tmp_12678[1] * 2741298027419160918, tmp_12678[2] * 2741298027419160918];
    signal tmp_12802[3] <== [tmp_12721[0] + tmp_12801[0], tmp_12721[1] + tmp_12801[1], tmp_12721[2] + tmp_12801[2]];
    signal tmp_12803[3] <== [5612088705630314148 * tmp_12802[0], 5612088705630314148 * tmp_12802[1], 5612088705630314148 * tmp_12802[2]];
    signal tmp_12804[3] <== [tmp_12799[0] + tmp_12803[0], tmp_12799[1] + tmp_12803[1], tmp_12799[2] + tmp_12803[2]];
    signal tmp_12806[3] <== [tmp_12678[0] * 15044675986841164042, tmp_12678[1] * 15044675986841164042, tmp_12678[2] * 15044675986841164042];
    signal tmp_12807[3] <== [tmp_12726[0] + tmp_12806[0], tmp_12726[1] + tmp_12806[1], tmp_12726[2] + tmp_12806[2]];
    signal tmp_12808[3] <== [7985909294534842852 * tmp_12807[0], 7985909294534842852 * tmp_12807[1], 7985909294534842852 * tmp_12807[2]];
    signal tmp_12809[3] <== [tmp_12804[0] + tmp_12808[0], tmp_12804[1] + tmp_12808[1], tmp_12804[2] + tmp_12808[2]];
    signal tmp_12811[3] <== [tmp_12678[0] * 16581973004291491442, tmp_12678[1] * 16581973004291491442, tmp_12678[2] * 16581973004291491442];
    signal tmp_12812[3] <== [tmp_12731[0] + tmp_12811[0], tmp_12731[1] + tmp_12811[1], tmp_12731[2] + tmp_12811[2]];
    signal tmp_12813[3] <== [7806714111671690128 * tmp_12812[0], 7806714111671690128 * tmp_12812[1], 7806714111671690128 * tmp_12812[2]];
    signal tmp_12814[3] <== [tmp_12809[0] + tmp_12813[0], tmp_12809[1] + tmp_12813[1], tmp_12809[2] + tmp_12813[2]];
    signal tmp_12816[3] <== [tmp_12678[0] * 18135546514968986064, tmp_12678[1] * 18135546514968986064, tmp_12678[2] * 18135546514968986064];
    signal tmp_12817[3] <== [tmp_12736[0] + tmp_12816[0], tmp_12736[1] + tmp_12816[1], tmp_12736[2] + tmp_12816[2]];
    signal tmp_12818[3] <== [10214446832758516695 * tmp_12817[0], 10214446832758516695 * tmp_12817[1], 10214446832758516695 * tmp_12817[2]];
    signal tmp_12819[3] <== [tmp_12814[0] + tmp_12818[0], tmp_12814[1] + tmp_12818[1], tmp_12814[2] + tmp_12818[2]];
    signal tmp_12821[3] <== [tmp_12678[0] * 15991008099224797738, tmp_12678[1] * 15991008099224797738, tmp_12678[2] * 15991008099224797738];
    signal tmp_12822[3] <== [tmp_12741[0] + tmp_12821[0], tmp_12741[1] + tmp_12821[1], tmp_12741[2] + tmp_12821[2]];
    signal tmp_12823[3] <== [4397775621482708994 * tmp_12822[0], 4397775621482708994 * tmp_12822[1], 4397775621482708994 * tmp_12822[2]];
    signal tmp_12824[3] <== [tmp_12819[0] + tmp_12823[0], tmp_12819[1] + tmp_12823[1], tmp_12819[2] + tmp_12823[2]];
    signal tmp_12826[3] <== [tmp_12678[0] * 3751874127382110996, tmp_12678[1] * 3751874127382110996, tmp_12678[2] * 3751874127382110996];
    signal tmp_12827[3] <== [tmp_12746[0] + tmp_12826[0], tmp_12746[1] + tmp_12826[1], tmp_12746[2] + tmp_12826[2]];
    signal tmp_12828[3] <== [5208611042143950342 * tmp_12827[0], 5208611042143950342 * tmp_12827[1], 5208611042143950342 * tmp_12827[2]];
    signal tmp_12829[3] <== [tmp_12824[0] + tmp_12828[0], tmp_12824[1] + tmp_12828[1], tmp_12824[2] + tmp_12828[2]];
    signal tmp_12831[3] <== [tmp_12678[0] * 5416412077983775726, tmp_12678[1] * 5416412077983775726, tmp_12678[2] * 5416412077983775726];
    signal tmp_12832[3] <== [tmp_12751[0] + tmp_12831[0], tmp_12751[1] + tmp_12831[1], tmp_12751[2] + tmp_12831[2]];
    signal tmp_12833[3] <== [7274343743337614775 * tmp_12832[0], 7274343743337614775 * tmp_12832[1], 7274343743337614775 * tmp_12832[2]];
    signal tmp_12834[3] <== [tmp_12829[0] + tmp_12833[0], tmp_12829[1] + tmp_12833[1], tmp_12829[2] + tmp_12833[2]];
    signal tmp_12835[3] <== [tmp_12834[0] * 1, tmp_12834[1] * 1, tmp_12834[2] * 1];
    signal tmp_12836[3] <== [evals[106][0] - tmp_12835[0], evals[106][1] - tmp_12835[1], evals[106][2] - tmp_12835[2]];
    signal tmp_12837[3] <== CMul()(tmp_9163, tmp_12836);
    signal tmp_12838[3] <== [tmp_12758[0] + tmp_12837[0], tmp_12758[1] + tmp_12837[1], tmp_12758[2] + tmp_12837[2]];
    signal tmp_12839[3] <== CMul()(challengeQ, tmp_12838);
    signal tmp_12840[3] <== [tmp_10599[0] + 2982104952642540993, tmp_10599[1], tmp_10599[2]];
    signal tmp_12842[3] <== [tmp_12759[0] * 4076524286822267596, tmp_12759[1] * 4076524286822267596, tmp_12759[2] * 4076524286822267596];
    signal tmp_12843[3] <== [tmp_12762[0] + tmp_12842[0], tmp_12762[1] + tmp_12842[1], tmp_12762[2] + tmp_12842[2]];
    signal tmp_12844[3] <== [6164038266681533795 * tmp_12843[0], 6164038266681533795 * tmp_12843[1], 6164038266681533795 * tmp_12843[2]];
    signal tmp_12845[3] <== [tmp_12840[0] + tmp_12844[0], tmp_12840[1] + tmp_12844[1], tmp_12840[2] + tmp_12844[2]];
    signal tmp_12847[3] <== [tmp_12759[0] * 8800108983290110025, tmp_12759[1] * 8800108983290110025, tmp_12759[2] * 8800108983290110025];
    signal tmp_12848[3] <== [tmp_12767[0] + tmp_12847[0], tmp_12767[1] + tmp_12847[1], tmp_12767[2] + tmp_12847[2]];
    signal tmp_12849[3] <== [2954388482741701990 * tmp_12848[0], 2954388482741701990 * tmp_12848[1], 2954388482741701990 * tmp_12848[2]];
    signal tmp_12850[3] <== [tmp_12845[0] + tmp_12849[0], tmp_12845[1] + tmp_12849[1], tmp_12845[2] + tmp_12849[2]];
    signal tmp_12852[3] <== [tmp_12759[0] * 291536037642081575, tmp_12759[1] * 291536037642081575, tmp_12759[2] * 291536037642081575];
    signal tmp_12853[3] <== [tmp_12772[0] + tmp_12852[0], tmp_12772[1] + tmp_12852[1], tmp_12772[2] + tmp_12852[2]];
    signal tmp_12854[3] <== [4767760292258986237 * tmp_12853[0], 4767760292258986237 * tmp_12853[1], 4767760292258986237 * tmp_12853[2]];
    signal tmp_12855[3] <== [tmp_12850[0] + tmp_12854[0], tmp_12850[1] + tmp_12854[1], tmp_12850[2] + tmp_12854[2]];
    signal tmp_12857[3] <== [tmp_12759[0] * 5510510448504677345, tmp_12759[1] * 5510510448504677345, tmp_12759[2] * 5510510448504677345];
    signal tmp_12858[3] <== [tmp_12777[0] + tmp_12857[0], tmp_12777[1] + tmp_12857[1], tmp_12777[2] + tmp_12857[2]];
    signal tmp_12859[3] <== [1216591495300834932 * tmp_12858[0], 1216591495300834932 * tmp_12858[1], 1216591495300834932 * tmp_12858[2]];
    signal tmp_12860[3] <== [tmp_12855[0] + tmp_12859[0], tmp_12855[1] + tmp_12859[1], tmp_12855[2] + tmp_12859[2]];
    signal tmp_12862[3] <== [tmp_12759[0] * 16087089554932036377, tmp_12759[1] * 16087089554932036377, tmp_12759[2] * 16087089554932036377];
    signal tmp_12863[3] <== [tmp_12782[0] + tmp_12862[0], tmp_12782[1] + tmp_12862[1], tmp_12782[2] + tmp_12862[2]];
    signal tmp_12864[3] <== [407520068134907189 * tmp_12863[0], 407520068134907189 * tmp_12863[1], 407520068134907189 * tmp_12863[2]];
    signal tmp_12865[3] <== [tmp_12860[0] + tmp_12864[0], tmp_12860[1] + tmp_12864[1], tmp_12860[2] + tmp_12864[2]];
    signal tmp_12867[3] <== [tmp_12759[0] * 9490752218128621864, tmp_12759[1] * 9490752218128621864, tmp_12759[2] * 9490752218128621864];
    signal tmp_12868[3] <== [tmp_12787[0] + tmp_12867[0], tmp_12787[1] + tmp_12867[1], tmp_12787[2] + tmp_12867[2]];
    signal tmp_12869[3] <== [8748275194504987666 * tmp_12868[0], 8748275194504987666 * tmp_12868[1], 8748275194504987666 * tmp_12868[2]];
    signal tmp_12870[3] <== [tmp_12865[0] + tmp_12869[0], tmp_12865[1] + tmp_12869[1], tmp_12865[2] + tmp_12869[2]];
    signal tmp_12872[3] <== [tmp_12759[0] * 17141708827443995602, tmp_12759[1] * 17141708827443995602, tmp_12759[2] * 17141708827443995602];
    signal tmp_12873[3] <== [tmp_12792[0] + tmp_12872[0], tmp_12792[1] + tmp_12872[1], tmp_12792[2] + tmp_12872[2]];
    signal tmp_12874[3] <== [4398221156344358190 * tmp_12873[0], 4398221156344358190 * tmp_12873[1], 4398221156344358190 * tmp_12873[2]];
    signal tmp_12875[3] <== [tmp_12870[0] + tmp_12874[0], tmp_12870[1] + tmp_12874[1], tmp_12870[2] + tmp_12874[2]];
    signal tmp_12877[3] <== [tmp_12759[0] * 363174800131535307, tmp_12759[1] * 363174800131535307, tmp_12759[2] * 363174800131535307];
    signal tmp_12878[3] <== [tmp_12797[0] + tmp_12877[0], tmp_12797[1] + tmp_12877[1], tmp_12797[2] + tmp_12877[2]];
    signal tmp_12879[3] <== [3691238772910702249 * tmp_12878[0], 3691238772910702249 * tmp_12878[1], 3691238772910702249 * tmp_12878[2]];
    signal tmp_12880[3] <== [tmp_12875[0] + tmp_12879[0], tmp_12875[1] + tmp_12879[1], tmp_12875[2] + tmp_12879[2]];
    signal tmp_12882[3] <== [tmp_12759[0] * 9926144393408541385, tmp_12759[1] * 9926144393408541385, tmp_12759[2] * 9926144393408541385];
    signal tmp_12883[3] <== [tmp_12802[0] + tmp_12882[0], tmp_12802[1] + tmp_12882[1], tmp_12802[2] + tmp_12882[2]];
    signal tmp_12884[3] <== [14469725748730639111 * tmp_12883[0], 14469725748730639111 * tmp_12883[1], 14469725748730639111 * tmp_12883[2]];
    signal tmp_12885[3] <== [tmp_12880[0] + tmp_12884[0], tmp_12880[1] + tmp_12884[1], tmp_12880[2] + tmp_12884[2]];
    signal tmp_12887[3] <== [tmp_12759[0] * 6235228401204354743, tmp_12759[1] * 6235228401204354743, tmp_12759[2] * 6235228401204354743];
    signal tmp_12888[3] <== [tmp_12807[0] + tmp_12887[0], tmp_12807[1] + tmp_12887[1], tmp_12807[2] + tmp_12887[2]];
    signal tmp_12889[3] <== [17322574795334626440 * tmp_12888[0], 17322574795334626440 * tmp_12888[1], 17322574795334626440 * tmp_12888[2]];
    signal tmp_12890[3] <== [tmp_12885[0] + tmp_12889[0], tmp_12885[1] + tmp_12889[1], tmp_12885[2] + tmp_12889[2]];
    signal tmp_12892[3] <== [tmp_12759[0] * 8054369299943103922, tmp_12759[1] * 8054369299943103922, tmp_12759[2] * 8054369299943103922];
    signal tmp_12893[3] <== [tmp_12812[0] + tmp_12892[0], tmp_12812[1] + tmp_12892[1], tmp_12812[2] + tmp_12892[2]];
    signal tmp_12894[3] <== [6754651456942327221 * tmp_12893[0], 6754651456942327221 * tmp_12893[1], 6754651456942327221 * tmp_12893[2]];
    signal tmp_12895[3] <== [tmp_12890[0] + tmp_12894[0], tmp_12890[1] + tmp_12894[1], tmp_12890[2] + tmp_12894[2]];
    signal tmp_12897[3] <== [tmp_12759[0] * 1181948739023406256, tmp_12759[1] * 1181948739023406256, tmp_12759[2] * 1181948739023406256];
    signal tmp_12898[3] <== [tmp_12817[0] + tmp_12897[0], tmp_12817[1] + tmp_12897[1], tmp_12817[2] + tmp_12897[2]];
    signal tmp_12899[3] <== [3445987485532732891 * tmp_12898[0], 3445987485532732891 * tmp_12898[1], 3445987485532732891 * tmp_12898[2]];
    signal tmp_12900[3] <== [tmp_12895[0] + tmp_12899[0], tmp_12895[1] + tmp_12899[1], tmp_12895[2] + tmp_12899[2]];
    signal tmp_12902[3] <== [tmp_12759[0] * 5256255887623871511, tmp_12759[1] * 5256255887623871511, tmp_12759[2] * 5256255887623871511];
    signal tmp_12903[3] <== [tmp_12822[0] + tmp_12902[0], tmp_12822[1] + tmp_12902[1], tmp_12822[2] + tmp_12902[2]];
    signal tmp_12904[3] <== [6696862914220145683 * tmp_12903[0], 6696862914220145683 * tmp_12903[1], 6696862914220145683 * tmp_12903[2]];
    signal tmp_12905[3] <== [tmp_12900[0] + tmp_12904[0], tmp_12900[1] + tmp_12904[1], tmp_12900[2] + tmp_12904[2]];
    signal tmp_12907[3] <== [tmp_12759[0] * 15192748352493407589, tmp_12759[1] * 15192748352493407589, tmp_12759[2] * 15192748352493407589];
    signal tmp_12908[3] <== [tmp_12827[0] + tmp_12907[0], tmp_12827[1] + tmp_12907[1], tmp_12827[2] + tmp_12907[2]];
    signal tmp_12909[3] <== [8135626180423978532 * tmp_12908[0], 8135626180423978532 * tmp_12908[1], 8135626180423978532 * tmp_12908[2]];
    signal tmp_12910[3] <== [tmp_12905[0] + tmp_12909[0], tmp_12905[1] + tmp_12909[1], tmp_12905[2] + tmp_12909[2]];
    signal tmp_12912[3] <== [tmp_12759[0] * 6531696571495839637, tmp_12759[1] * 6531696571495839637, tmp_12759[2] * 6531696571495839637];
    signal tmp_12913[3] <== [tmp_12832[0] + tmp_12912[0], tmp_12832[1] + tmp_12912[1], tmp_12832[2] + tmp_12912[2]];
    signal tmp_12914[3] <== [17786688261949651693 * tmp_12913[0], 17786688261949651693 * tmp_12913[1], 17786688261949651693 * tmp_12913[2]];
    signal tmp_12915[3] <== [tmp_12910[0] + tmp_12914[0], tmp_12910[1] + tmp_12914[1], tmp_12910[2] + tmp_12914[2]];
    signal tmp_12916[3] <== [tmp_12915[0] * 1, tmp_12915[1] * 1, tmp_12915[2] * 1];
    signal tmp_12917[3] <== [evals[107][0] - tmp_12916[0], evals[107][1] - tmp_12916[1], evals[107][2] - tmp_12916[2]];
    signal tmp_12918[3] <== CMul()(tmp_9163, tmp_12917);
    signal tmp_12919[3] <== [tmp_12839[0] + tmp_12918[0], tmp_12839[1] + tmp_12918[1], tmp_12839[2] + tmp_12918[2]];
    signal tmp_12920[3] <== CMul()(challengeQ, tmp_12919);
    signal tmp_12921[3] <== [tmp_10610[0] + 3183805956781626142, tmp_10610[1], tmp_10610[2]];
    signal tmp_12923[3] <== [tmp_12840[0] * 15537013571472701229, tmp_12840[1] * 15537013571472701229, tmp_12840[2] * 15537013571472701229];
    signal tmp_12924[3] <== [tmp_12843[0] + tmp_12923[0], tmp_12843[1] + tmp_12923[1], tmp_12843[2] + tmp_12923[2]];
    signal tmp_12925[3] <== [16947245409080232829 * tmp_12924[0], 16947245409080232829 * tmp_12924[1], 16947245409080232829 * tmp_12924[2]];
    signal tmp_12926[3] <== [tmp_12921[0] + tmp_12925[0], tmp_12921[1] + tmp_12925[1], tmp_12921[2] + tmp_12925[2]];
    signal tmp_12928[3] <== [tmp_12840[0] * 16947559881337334737, tmp_12840[1] * 16947559881337334737, tmp_12840[2] * 16947559881337334737];
    signal tmp_12929[3] <== [tmp_12848[0] + tmp_12928[0], tmp_12848[1] + tmp_12928[1], tmp_12848[2] + tmp_12928[2]];
    signal tmp_12930[3] <== [15778599751024291825 * tmp_12929[0], 15778599751024291825 * tmp_12929[1], 15778599751024291825 * tmp_12929[2]];
    signal tmp_12931[3] <== [tmp_12926[0] + tmp_12930[0], tmp_12926[1] + tmp_12930[1], tmp_12926[2] + tmp_12930[2]];
    signal tmp_12933[3] <== [tmp_12840[0] * 7493831729793585316, tmp_12840[1] * 7493831729793585316, tmp_12840[2] * 7493831729793585316];
    signal tmp_12934[3] <== [tmp_12853[0] + tmp_12933[0], tmp_12853[1] + tmp_12933[1], tmp_12853[2] + tmp_12933[2]];
    signal tmp_12935[3] <== [12554573928367352644 * tmp_12934[0], 12554573928367352644 * tmp_12934[1], 12554573928367352644 * tmp_12934[2]];
    signal tmp_12936[3] <== [tmp_12931[0] + tmp_12935[0], tmp_12931[1] + tmp_12935[1], tmp_12931[2] + tmp_12935[2]];
    signal tmp_12938[3] <== [tmp_12840[0] * 1623918663904207169, tmp_12840[1] * 1623918663904207169, tmp_12840[2] * 1623918663904207169];
    signal tmp_12939[3] <== [tmp_12858[0] + tmp_12938[0], tmp_12858[1] + tmp_12938[1], tmp_12858[2] + tmp_12938[2]];
    signal tmp_12940[3] <== [17305511116721035138 * tmp_12939[0], 17305511116721035138 * tmp_12939[1], 17305511116721035138 * tmp_12939[2]];
    signal tmp_12941[3] <== [tmp_12936[0] + tmp_12940[0], tmp_12936[1] + tmp_12940[1], tmp_12936[2] + tmp_12940[2]];
    signal tmp_12943[3] <== [tmp_12840[0] * 4687160973583834774, tmp_12840[1] * 4687160973583834774, tmp_12840[2] * 4687160973583834774];
    signal tmp_12944[3] <== [tmp_12863[0] + tmp_12943[0], tmp_12863[1] + tmp_12943[1], tmp_12863[2] + tmp_12943[2]];
    signal tmp_12945[3] <== [14036371303403364899 * tmp_12944[0], 14036371303403364899 * tmp_12944[1], 14036371303403364899 * tmp_12944[2]];
    signal tmp_12946[3] <== [tmp_12941[0] + tmp_12945[0], tmp_12941[1] + tmp_12945[1], tmp_12941[2] + tmp_12945[2]];
    signal tmp_12948[3] <== [tmp_12840[0] * 7481496270331462357, tmp_12840[1] * 7481496270331462357, tmp_12840[2] * 7481496270331462357];
    signal tmp_12949[3] <== [tmp_12868[0] + tmp_12948[0], tmp_12868[1] + tmp_12948[1], tmp_12868[2] + tmp_12948[2]];
    signal tmp_12950[3] <== [17810786690047806733 * tmp_12949[0], 17810786690047806733 * tmp_12949[1], 17810786690047806733 * tmp_12949[2]];
    signal tmp_12951[3] <== [tmp_12946[0] + tmp_12950[0], tmp_12946[1] + tmp_12950[1], tmp_12946[2] + tmp_12950[2]];
    signal tmp_12953[3] <== [tmp_12840[0] * 15777438269644270173, tmp_12840[1] * 15777438269644270173, tmp_12840[2] * 15777438269644270173];
    signal tmp_12954[3] <== [tmp_12873[0] + tmp_12953[0], tmp_12873[1] + tmp_12953[1], tmp_12873[2] + tmp_12953[2]];
    signal tmp_12955[3] <== [2212888412728595072 * tmp_12954[0], 2212888412728595072 * tmp_12954[1], 2212888412728595072 * tmp_12954[2]];
    signal tmp_12956[3] <== [tmp_12951[0] + tmp_12955[0], tmp_12951[1] + tmp_12955[1], tmp_12951[2] + tmp_12955[2]];
    signal tmp_12958[3] <== [tmp_12840[0] * 12845383926406511601, tmp_12840[1] * 12845383926406511601, tmp_12840[2] * 12845383926406511601];
    signal tmp_12959[3] <== [tmp_12878[0] + tmp_12958[0], tmp_12878[1] + tmp_12958[1], tmp_12878[2] + tmp_12958[2]];
    signal tmp_12960[3] <== [11824922540475901574 * tmp_12959[0], 11824922540475901574 * tmp_12959[1], 11824922540475901574 * tmp_12959[2]];
    signal tmp_12961[3] <== [tmp_12956[0] + tmp_12960[0], tmp_12956[1] + tmp_12960[1], tmp_12956[2] + tmp_12960[2]];
    signal tmp_12963[3] <== [tmp_12840[0] * 11148510750866669007, tmp_12840[1] * 11148510750866669007, tmp_12840[2] * 11148510750866669007];
    signal tmp_12964[3] <== [tmp_12883[0] + tmp_12963[0], tmp_12883[1] + tmp_12963[1], tmp_12883[2] + tmp_12963[2]];
    signal tmp_12965[3] <== [14117971470759675313 * tmp_12964[0], 14117971470759675313 * tmp_12964[1], 14117971470759675313 * tmp_12964[2]];
    signal tmp_12966[3] <== [tmp_12961[0] + tmp_12965[0], tmp_12961[1] + tmp_12965[1], tmp_12961[2] + tmp_12965[2]];
    signal tmp_12968[3] <== [tmp_12840[0] * 15318591593009208330, tmp_12840[1] * 15318591593009208330, tmp_12840[2] * 15318591593009208330];
    signal tmp_12969[3] <== [tmp_12888[0] + tmp_12968[0], tmp_12888[1] + tmp_12968[1], tmp_12888[2] + tmp_12968[2]];
    signal tmp_12970[3] <== [6803345862685226999 * tmp_12969[0], 6803345862685226999 * tmp_12969[1], 6803345862685226999 * tmp_12969[2]];
    signal tmp_12971[3] <== [tmp_12966[0] + tmp_12970[0], tmp_12966[1] + tmp_12970[1], tmp_12966[2] + tmp_12970[2]];
    signal tmp_12973[3] <== [tmp_12840[0] * 12116635019874345744, tmp_12840[1] * 12116635019874345744, tmp_12840[2] * 12116635019874345744];
    signal tmp_12974[3] <== [tmp_12893[0] + tmp_12973[0], tmp_12893[1] + tmp_12973[1], tmp_12893[2] + tmp_12973[2]];
    signal tmp_12975[3] <== [1790404652192668679 * tmp_12974[0], 1790404652192668679 * tmp_12974[1], 1790404652192668679 * tmp_12974[2]];
    signal tmp_12976[3] <== [tmp_12971[0] + tmp_12975[0], tmp_12971[1] + tmp_12975[1], tmp_12971[2] + tmp_12975[2]];
    signal tmp_12978[3] <== [tmp_12840[0] * 10427984200609223037, tmp_12840[1] * 10427984200609223037, tmp_12840[2] * 10427984200609223037];
    signal tmp_12979[3] <== [tmp_12898[0] + tmp_12978[0], tmp_12898[1] + tmp_12978[1], tmp_12898[2] + tmp_12978[2]];
    signal tmp_12980[3] <== [9306702297937622557 * tmp_12979[0], 9306702297937622557 * tmp_12979[1], 9306702297937622557 * tmp_12979[2]];
    signal tmp_12981[3] <== [tmp_12976[0] + tmp_12980[0], tmp_12976[1] + tmp_12980[1], tmp_12976[2] + tmp_12980[2]];
    signal tmp_12983[3] <== [tmp_12840[0] * 1275063075010481368, tmp_12840[1] * 1275063075010481368, tmp_12840[2] * 1275063075010481368];
    signal tmp_12984[3] <== [tmp_12903[0] + tmp_12983[0], tmp_12903[1] + tmp_12983[1], tmp_12903[2] + tmp_12983[2]];
    signal tmp_12985[3] <== [17976990147333300636 * tmp_12984[0], 17976990147333300636 * tmp_12984[1], 17976990147333300636 * tmp_12984[2]];
    signal tmp_12986[3] <== [tmp_12981[0] + tmp_12985[0], tmp_12981[1] + tmp_12985[1], tmp_12981[2] + tmp_12985[2]];
    signal tmp_12988[3] <== [tmp_12840[0] * 17363335480109898214, tmp_12840[1] * 17363335480109898214, tmp_12840[2] * 17363335480109898214];
    signal tmp_12989[3] <== [tmp_12908[0] + tmp_12988[0], tmp_12908[1] + tmp_12988[1], tmp_12908[2] + tmp_12988[2]];
    signal tmp_12990[3] <== [14238877083147544792 * tmp_12989[0], 14238877083147544792 * tmp_12989[1], 14238877083147544792 * tmp_12989[2]];
    signal tmp_12991[3] <== [tmp_12986[0] + tmp_12990[0], tmp_12986[1] + tmp_12990[1], tmp_12986[2] + tmp_12990[2]];
    signal tmp_12993[3] <== [tmp_12840[0] * 8340043298775904670, tmp_12840[1] * 8340043298775904670, tmp_12840[2] * 8340043298775904670];
    signal tmp_12994[3] <== [tmp_12913[0] + tmp_12993[0], tmp_12913[1] + tmp_12993[1], tmp_12913[2] + tmp_12993[2]];
    signal tmp_12995[3] <== [8744547697144458331 * tmp_12994[0], 8744547697144458331 * tmp_12994[1], 8744547697144458331 * tmp_12994[2]];
    signal tmp_12996[3] <== [tmp_12991[0] + tmp_12995[0], tmp_12991[1] + tmp_12995[1], tmp_12991[2] + tmp_12995[2]];
    signal tmp_12997[3] <== [tmp_12996[0] * 1, tmp_12996[1] * 1, tmp_12996[2] * 1];
    signal tmp_12998[3] <== [evals[108][0] - tmp_12997[0], evals[108][1] - tmp_12997[1], evals[108][2] - tmp_12997[2]];
    signal tmp_12999[3] <== CMul()(tmp_9163, tmp_12998);
    signal tmp_13000[3] <== [tmp_12920[0] + tmp_12999[0], tmp_12920[1] + tmp_12999[1], tmp_12920[2] + tmp_12999[2]];
    signal tmp_13001[3] <== CMul()(challengeQ, tmp_13000);
    signal tmp_13002[3] <== [tmp_10622[0] + 9076472642853520518, tmp_10622[1], tmp_10622[2]];
    signal tmp_13004[3] <== [tmp_12921[0] * 18432038191863482059, tmp_12921[1] * 18432038191863482059, tmp_12921[2] * 18432038191863482059];
    signal tmp_13005[3] <== [tmp_12924[0] + tmp_13004[0], tmp_12924[1] + tmp_13004[1], tmp_12924[2] + tmp_13004[2]];
    signal tmp_13006[3] <== [3220543931095753511 * tmp_13005[0], 3220543931095753511 * tmp_13005[1], 3220543931095753511 * tmp_13005[2]];
    signal tmp_13007[3] <== [tmp_13002[0] + tmp_13006[0], tmp_13002[1] + tmp_13006[1], tmp_13002[2] + tmp_13006[2]];
    signal tmp_13009[3] <== [tmp_12921[0] * 5432413814338603108, tmp_12921[1] * 5432413814338603108, tmp_12921[2] * 5432413814338603108];
    signal tmp_13010[3] <== [tmp_12929[0] + tmp_13009[0], tmp_12929[1] + tmp_13009[1], tmp_12929[2] + tmp_13009[2]];
    signal tmp_13011[3] <== [7401724255027812990 * tmp_13010[0], 7401724255027812990 * tmp_13010[1], 7401724255027812990 * tmp_13010[2]];
    signal tmp_13012[3] <== [tmp_13007[0] + tmp_13011[0], tmp_13007[1] + tmp_13011[1], tmp_13007[2] + tmp_13011[2]];
    signal tmp_13014[3] <== [tmp_12921[0] * 11454376156859769836, tmp_12921[1] * 11454376156859769836, tmp_12921[2] * 11454376156859769836];
    signal tmp_13015[3] <== [tmp_12934[0] + tmp_13014[0], tmp_12934[1] + tmp_13014[1], tmp_12934[2] + tmp_13014[2]];
    signal tmp_13016[3] <== [17813499879760129016 * tmp_13015[0], 17813499879760129016 * tmp_13015[1], 17813499879760129016 * tmp_13015[2]];
    signal tmp_13017[3] <== [tmp_13012[0] + tmp_13016[0], tmp_13012[1] + tmp_13016[1], tmp_13012[2] + tmp_13016[2]];
    signal tmp_13019[3] <== [tmp_12921[0] * 3406012298082584781, tmp_12921[1] * 3406012298082584781, tmp_12921[2] * 3406012298082584781];
    signal tmp_13020[3] <== [tmp_12939[0] + tmp_13019[0], tmp_12939[1] + tmp_13019[1], tmp_12939[2] + tmp_13019[2]];
    signal tmp_13021[3] <== [5428548591833763345 * tmp_13020[0], 5428548591833763345 * tmp_13020[1], 5428548591833763345 * tmp_13020[2]];
    signal tmp_13022[3] <== [tmp_13017[0] + tmp_13021[0], tmp_13017[1] + tmp_13021[1], tmp_13017[2] + tmp_13021[2]];
    signal tmp_13024[3] <== [tmp_12921[0] * 14588634923383545112, tmp_12921[1] * 14588634923383545112, tmp_12921[2] * 14588634923383545112];
    signal tmp_13025[3] <== [tmp_12944[0] + tmp_13024[0], tmp_12944[1] + tmp_13024[1], tmp_12944[2] + tmp_13024[2]];
    signal tmp_13026[3] <== [6131968098119340952 * tmp_13025[0], 6131968098119340952 * tmp_13025[1], 6131968098119340952 * tmp_13025[2]];
    signal tmp_13027[3] <== [tmp_13022[0] + tmp_13026[0], tmp_13022[1] + tmp_13026[1], tmp_13022[2] + tmp_13026[2]];
    signal tmp_13029[3] <== [tmp_12921[0] * 7967410531777373627, tmp_12921[1] * 7967410531777373627, tmp_12921[2] * 7967410531777373627];
    signal tmp_13030[3] <== [tmp_12949[0] + tmp_13029[0], tmp_12949[1] + tmp_13029[1], tmp_12949[2] + tmp_13029[2]];
    signal tmp_13031[3] <== [2074496711530634450 * tmp_13030[0], 2074496711530634450 * tmp_13030[1], 2074496711530634450 * tmp_13030[2]];
    signal tmp_13032[3] <== [tmp_13027[0] + tmp_13031[0], tmp_13027[1] + tmp_13031[1], tmp_13027[2] + tmp_13031[2]];
    signal tmp_13034[3] <== [tmp_12921[0] * 5132439848537977263, tmp_12921[1] * 5132439848537977263, tmp_12921[2] * 5132439848537977263];
    signal tmp_13035[3] <== [tmp_12954[0] + tmp_13034[0], tmp_12954[1] + tmp_13034[1], tmp_12954[2] + tmp_13034[2]];
    signal tmp_13036[3] <== [5882024272570382639 * tmp_13035[0], 5882024272570382639 * tmp_13035[1], 5882024272570382639 * tmp_13035[2]];
    signal tmp_13037[3] <== [tmp_13032[0] + tmp_13036[0], tmp_13032[1] + tmp_13036[1], tmp_13032[2] + tmp_13036[2]];
    signal tmp_13039[3] <== [tmp_12921[0] * 9090563597932543104, tmp_12921[1] * 9090563597932543104, tmp_12921[2] * 9090563597932543104];
    signal tmp_13040[3] <== [tmp_12959[0] + tmp_13039[0], tmp_12959[1] + tmp_13039[1], tmp_12959[2] + tmp_13039[2]];
    signal tmp_13041[3] <== [12970340649830174380 * tmp_13040[0], 12970340649830174380 * tmp_13040[1], 12970340649830174380 * tmp_13040[2]];
    signal tmp_13042[3] <== [tmp_13037[0] + tmp_13041[0], tmp_13037[1] + tmp_13041[1], tmp_13037[2] + tmp_13041[2]];
    signal tmp_13044[3] <== [tmp_12921[0] * 16222775232712330818, tmp_12921[1] * 16222775232712330818, tmp_12921[2] * 16222775232712330818];
    signal tmp_13045[3] <== [tmp_12964[0] + tmp_13044[0], tmp_12964[1] + tmp_13044[1], tmp_12964[2] + tmp_13044[2]];
    signal tmp_13046[3] <== [4251380188844404593 * tmp_13045[0], 4251380188844404593 * tmp_13045[1], 4251380188844404593 * tmp_13045[2]];
    signal tmp_13047[3] <== [tmp_13042[0] + tmp_13046[0], tmp_13042[1] + tmp_13046[1], tmp_13042[2] + tmp_13046[2]];
    signal tmp_13049[3] <== [tmp_12921[0] * 14947628689583127907, tmp_12921[1] * 14947628689583127907, tmp_12921[2] * 14947628689583127907];
    signal tmp_13050[3] <== [tmp_12969[0] + tmp_13049[0], tmp_12969[1] + tmp_13049[1], tmp_12969[2] + tmp_13049[2]];
    signal tmp_13051[3] <== [5246780099262719987 * tmp_13050[0], 5246780099262719987 * tmp_13050[1], 5246780099262719987 * tmp_13050[2]];
    signal tmp_13052[3] <== [tmp_13047[0] + tmp_13051[0], tmp_13047[1] + tmp_13051[1], tmp_13047[2] + tmp_13051[2]];
    signal tmp_13054[3] <== [tmp_12921[0] * 800947794585510760, tmp_12921[1] * 800947794585510760, tmp_12921[2] * 800947794585510760];
    signal tmp_13055[3] <== [tmp_12974[0] + tmp_13054[0], tmp_12974[1] + tmp_13054[1], tmp_12974[2] + tmp_13054[2]];
    signal tmp_13056[3] <== [12265306642981888259 * tmp_13055[0], 12265306642981888259 * tmp_13055[1], 12265306642981888259 * tmp_13055[2]];
    signal tmp_13057[3] <== [tmp_13052[0] + tmp_13056[0], tmp_13052[1] + tmp_13056[1], tmp_13052[2] + tmp_13056[2]];
    signal tmp_13059[3] <== [tmp_12921[0] * 3863721566387520445, tmp_12921[1] * 3863721566387520445, tmp_12921[2] * 3863721566387520445];
    signal tmp_13060[3] <== [tmp_12979[0] + tmp_13059[0], tmp_12979[1] + tmp_13059[1], tmp_12979[2] + tmp_13059[2]];
    signal tmp_13061[3] <== [11845579913726276983 * tmp_13060[0], 11845579913726276983 * tmp_13060[1], 11845579913726276983 * tmp_13060[2]];
    signal tmp_13062[3] <== [tmp_13057[0] + tmp_13061[0], tmp_13057[1] + tmp_13061[1], tmp_13057[2] + tmp_13061[2]];
    signal tmp_13064[3] <== [tmp_12921[0] * 8395171295366482951, tmp_12921[1] * 8395171295366482951, tmp_12921[2] * 8395171295366482951];
    signal tmp_13065[3] <== [tmp_12984[0] + tmp_13064[0], tmp_12984[1] + tmp_13064[1], tmp_12984[2] + tmp_13064[2]];
    signal tmp_13066[3] <== [4161015590580980279 * tmp_13065[0], 4161015590580980279 * tmp_13065[1], 4161015590580980279 * tmp_13065[2]];
    signal tmp_13067[3] <== [tmp_13062[0] + tmp_13066[0], tmp_13062[1] + tmp_13066[1], tmp_13062[2] + tmp_13066[2]];
    signal tmp_13069[3] <== [tmp_12921[0] * 13583398811995500171, tmp_12921[1] * 13583398811995500171, tmp_12921[2] * 13583398811995500171];
    signal tmp_13070[3] <== [tmp_12989[0] + tmp_13069[0], tmp_12989[1] + tmp_13069[1], tmp_12989[2] + tmp_13069[2]];
    signal tmp_13071[3] <== [16142970107084859093 * tmp_13070[0], 16142970107084859093 * tmp_13070[1], 16142970107084859093 * tmp_13070[2]];
    signal tmp_13072[3] <== [tmp_13067[0] + tmp_13071[0], tmp_13067[1] + tmp_13071[1], tmp_13067[2] + tmp_13071[2]];
    signal tmp_13074[3] <== [tmp_12921[0] * 11308879932982657557, tmp_12921[1] * 11308879932982657557, tmp_12921[2] * 11308879932982657557];
    signal tmp_13075[3] <== [tmp_12994[0] + tmp_13074[0], tmp_12994[1] + tmp_13074[1], tmp_12994[2] + tmp_13074[2]];
    signal tmp_13076[3] <== [10242614418821665472 * tmp_13075[0], 10242614418821665472 * tmp_13075[1], 10242614418821665472 * tmp_13075[2]];
    signal tmp_13077[3] <== [tmp_13072[0] + tmp_13076[0], tmp_13072[1] + tmp_13076[1], tmp_13072[2] + tmp_13076[2]];
    signal tmp_13078[3] <== [tmp_13077[0] * 1, tmp_13077[1] * 1, tmp_13077[2] * 1];
    signal tmp_13079[3] <== [evals[109][0] - tmp_13078[0], evals[109][1] - tmp_13078[1], evals[109][2] - tmp_13078[2]];
    signal tmp_13080[3] <== CMul()(tmp_9163, tmp_13079);
    signal tmp_13081[3] <== [tmp_13001[0] + tmp_13080[0], tmp_13001[1] + tmp_13080[1], tmp_13001[2] + tmp_13080[2]];
    signal tmp_13082[3] <== CMul()(challengeQ, tmp_13081);
    signal tmp_13083[3] <== [tmp_10634[0] + 14285914829807850365, tmp_10634[1], tmp_10634[2]];
    signal tmp_13085[3] <== [tmp_13002[0] * 3130238529562802618, tmp_13002[1] * 3130238529562802618, tmp_13002[2] * 3130238529562802618];
    signal tmp_13086[3] <== [tmp_13005[0] + tmp_13085[0], tmp_13005[1] + tmp_13085[1], tmp_13005[2] + tmp_13085[2]];
    signal tmp_13087[3] <== [8578786096637244792 * tmp_13086[0], 8578786096637244792 * tmp_13086[1], 8578786096637244792 * tmp_13086[2]];
    signal tmp_13088[3] <== [tmp_13083[0] + tmp_13087[0], tmp_13083[1] + tmp_13087[1], tmp_13083[2] + tmp_13087[2]];
    signal tmp_13090[3] <== [tmp_13002[0] * 17571383761285423616, tmp_13002[1] * 17571383761285423616, tmp_13002[2] * 17571383761285423616];
    signal tmp_13091[3] <== [tmp_13010[0] + tmp_13090[0], tmp_13010[1] + tmp_13090[1], tmp_13010[2] + tmp_13090[2]];
    signal tmp_13092[3] <== [10017940286670940083 * tmp_13091[0], 10017940286670940083 * tmp_13091[1], 10017940286670940083 * tmp_13091[2]];
    signal tmp_13093[3] <== [tmp_13088[0] + tmp_13092[0], tmp_13088[1] + tmp_13092[1], tmp_13088[2] + tmp_13092[2]];
    signal tmp_13095[3] <== [tmp_13002[0] * 11604431936999842251, tmp_13002[1] * 11604431936999842251, tmp_13002[2] * 11604431936999842251];
    signal tmp_13096[3] <== [tmp_13015[0] + tmp_13095[0], tmp_13015[1] + tmp_13095[1], tmp_13015[2] + tmp_13095[2]];
    signal tmp_13097[3] <== [6061757955429109029 * tmp_13096[0], 6061757955429109029 * tmp_13096[1], 6061757955429109029 * tmp_13096[2]];
    signal tmp_13098[3] <== [tmp_13093[0] + tmp_13097[0], tmp_13093[1] + tmp_13097[1], tmp_13093[2] + tmp_13097[2]];
    signal tmp_13100[3] <== [tmp_13002[0] * 1643416478018396715, tmp_13002[1] * 1643416478018396715, tmp_13002[2] * 1643416478018396715];
    signal tmp_13101[3] <== [tmp_13020[0] + tmp_13100[0], tmp_13020[1] + tmp_13100[1], tmp_13020[2] + tmp_13100[2]];
    signal tmp_13102[3] <== [6830443121764588145 * tmp_13101[0], 6830443121764588145 * tmp_13101[1], 6830443121764588145 * tmp_13101[2]];
    signal tmp_13103[3] <== [tmp_13098[0] + tmp_13102[0], tmp_13098[1] + tmp_13102[1], tmp_13098[2] + tmp_13102[2]];
    signal tmp_13105[3] <== [tmp_13002[0] * 9529320288398332687, tmp_13002[1] * 9529320288398332687, tmp_13002[2] * 9529320288398332687];
    signal tmp_13106[3] <== [tmp_13025[0] + tmp_13105[0], tmp_13025[1] + tmp_13105[1], tmp_13025[2] + tmp_13105[2]];
    signal tmp_13107[3] <== [6572675368485775153 * tmp_13106[0], 6572675368485775153 * tmp_13106[1], 6572675368485775153 * tmp_13106[2]];
    signal tmp_13108[3] <== [tmp_13103[0] + tmp_13107[0], tmp_13103[1] + tmp_13107[1], tmp_13103[2] + tmp_13107[2]];
    signal tmp_13110[3] <== [tmp_13002[0] * 2943500820182364631, tmp_13002[1] * 2943500820182364631, tmp_13002[2] * 2943500820182364631];
    signal tmp_13111[3] <== [tmp_13030[0] + tmp_13110[0], tmp_13030[1] + tmp_13110[1], tmp_13030[2] + tmp_13110[2]];
    signal tmp_13112[3] <== [2570198071091962115 * tmp_13111[0], 2570198071091962115 * tmp_13111[1], 2570198071091962115 * tmp_13111[2]];
    signal tmp_13113[3] <== [tmp_13108[0] + tmp_13112[0], tmp_13108[1] + tmp_13112[1], tmp_13108[2] + tmp_13112[2]];
    signal tmp_13115[3] <== [tmp_13002[0] * 5642253925908086286, tmp_13002[1] * 5642253925908086286, tmp_13002[2] * 5642253925908086286];
    signal tmp_13116[3] <== [tmp_13035[0] + tmp_13115[0], tmp_13035[1] + tmp_13115[1], tmp_13035[2] + tmp_13115[2]];
    signal tmp_13117[3] <== [4859557487763691165 * tmp_13116[0], 4859557487763691165 * tmp_13116[1], 4859557487763691165 * tmp_13116[2]];
    signal tmp_13118[3] <== [tmp_13113[0] + tmp_13117[0], tmp_13113[1] + tmp_13117[1], tmp_13113[2] + tmp_13117[2]];
    signal tmp_13120[3] <== [tmp_13002[0] * 1584863777448828811, tmp_13002[1] * 1584863777448828811, tmp_13002[2] * 1584863777448828811];
    signal tmp_13121[3] <== [tmp_13040[0] + tmp_13120[0], tmp_13040[1] + tmp_13120[1], tmp_13040[2] + tmp_13120[2]];
    signal tmp_13122[3] <== [5904682032025033039 * tmp_13121[0], 5904682032025033039 * tmp_13121[1], 5904682032025033039 * tmp_13121[2]];
    signal tmp_13123[3] <== [tmp_13118[0] + tmp_13122[0], tmp_13118[1] + tmp_13122[1], tmp_13118[2] + tmp_13122[2]];
    signal tmp_13125[3] <== [tmp_13002[0] * 12918821123083034440, tmp_13002[1] * 12918821123083034440, tmp_13002[2] * 12918821123083034440];
    signal tmp_13126[3] <== [tmp_13045[0] + tmp_13125[0], tmp_13045[1] + tmp_13125[1], tmp_13045[2] + tmp_13125[2]];
    signal tmp_13127[3] <== [13831434945980884721 * tmp_13126[0], 13831434945980884721 * tmp_13126[1], 13831434945980884721 * tmp_13126[2]];
    signal tmp_13128[3] <== [tmp_13123[0] + tmp_13127[0], tmp_13123[1] + tmp_13127[1], tmp_13123[2] + tmp_13127[2]];
    signal tmp_13130[3] <== [tmp_13002[0] * 4794647653414383806, tmp_13002[1] * 4794647653414383806, tmp_13002[2] * 4794647653414383806];
    signal tmp_13131[3] <== [tmp_13050[0] + tmp_13130[0], tmp_13050[1] + tmp_13130[1], tmp_13050[2] + tmp_13130[2]];
    signal tmp_13132[3] <== [13071285619755217600 * tmp_13131[0], 13071285619755217600 * tmp_13131[1], 13071285619755217600 * tmp_13131[2]];
    signal tmp_13133[3] <== [tmp_13128[0] + tmp_13132[0], tmp_13128[1] + tmp_13132[1], tmp_13128[2] + tmp_13132[2]];
    signal tmp_13135[3] <== [tmp_13002[0] * 12339757025454833750, tmp_13002[1] * 12339757025454833750, tmp_13002[2] * 12339757025454833750];
    signal tmp_13136[3] <== [tmp_13055[0] + tmp_13135[0], tmp_13055[1] + tmp_13135[1], tmp_13055[2] + tmp_13135[2]];
    signal tmp_13137[3] <== [9745460330714099568 * tmp_13136[0], 9745460330714099568 * tmp_13136[1], 9745460330714099568 * tmp_13136[2]];
    signal tmp_13138[3] <== [tmp_13133[0] + tmp_13137[0], tmp_13133[1] + tmp_13137[1], tmp_13133[2] + tmp_13137[2]];
    signal tmp_13140[3] <== [tmp_13002[0] * 15902923065000667949, tmp_13002[1] * 15902923065000667949, tmp_13002[2] * 15902923065000667949];
    signal tmp_13141[3] <== [tmp_13060[0] + tmp_13140[0], tmp_13060[1] + tmp_13140[1], tmp_13060[2] + tmp_13140[2]];
    signal tmp_13142[3] <== [17565589964402351072 * tmp_13141[0], 17565589964402351072 * tmp_13141[1], 17565589964402351072 * tmp_13141[2]];
    signal tmp_13143[3] <== [tmp_13138[0] + tmp_13142[0], tmp_13138[1] + tmp_13142[1], tmp_13138[2] + tmp_13142[2]];
    signal tmp_13145[3] <== [tmp_13002[0] * 945086332862338525, tmp_13002[1] * 945086332862338525, tmp_13002[2] * 945086332862338525];
    signal tmp_13146[3] <== [tmp_13065[0] + tmp_13145[0], tmp_13065[1] + tmp_13145[1], tmp_13065[2] + tmp_13145[2]];
    signal tmp_13147[3] <== [4478734179481674309 * tmp_13146[0], 4478734179481674309 * tmp_13146[1], 4478734179481674309 * tmp_13146[2]];
    signal tmp_13148[3] <== [tmp_13143[0] + tmp_13147[0], tmp_13143[1] + tmp_13147[1], tmp_13143[2] + tmp_13147[2]];
    signal tmp_13150[3] <== [tmp_13002[0] * 13520575880496481741, tmp_13002[1] * 13520575880496481741, tmp_13002[2] * 13520575880496481741];
    signal tmp_13151[3] <== [tmp_13070[0] + tmp_13150[0], tmp_13070[1] + tmp_13150[1], tmp_13070[2] + tmp_13150[2]];
    signal tmp_13152[3] <== [5964515412475082128 * tmp_13151[0], 5964515412475082128 * tmp_13151[1], 5964515412475082128 * tmp_13151[2]];
    signal tmp_13153[3] <== [tmp_13148[0] + tmp_13152[0], tmp_13148[1] + tmp_13152[1], tmp_13148[2] + tmp_13152[2]];
    signal tmp_13155[3] <== [tmp_13002[0] * 7361764812588359828, tmp_13002[1] * 7361764812588359828, tmp_13002[2] * 7361764812588359828];
    signal tmp_13156[3] <== [tmp_13075[0] + tmp_13155[0], tmp_13075[1] + tmp_13155[1], tmp_13075[2] + tmp_13155[2]];
    signal tmp_13157[3] <== [10182176813386478138 * tmp_13156[0], 10182176813386478138 * tmp_13156[1], 10182176813386478138 * tmp_13156[2]];
    signal tmp_13158[3] <== [tmp_13153[0] + tmp_13157[0], tmp_13153[1] + tmp_13157[1], tmp_13153[2] + tmp_13157[2]];
    signal tmp_13159[3] <== [tmp_13158[0] * 1, tmp_13158[1] * 1, tmp_13158[2] * 1];
    signal tmp_13160[3] <== [evals[110][0] - tmp_13159[0], evals[110][1] - tmp_13159[1], evals[110][2] - tmp_13159[2]];
    signal tmp_13161[3] <== CMul()(tmp_9163, tmp_13160);
    signal tmp_13162[3] <== [tmp_13082[0] + tmp_13161[0], tmp_13082[1] + tmp_13161[1], tmp_13082[2] + tmp_13161[2]];
    signal tmp_13163[3] <== CMul()(challengeQ, tmp_13162);
    signal tmp_13164[3] <== [tmp_10646[0] + 15492711980379895475, tmp_10646[1], tmp_10646[2]];
    signal tmp_13166[3] <== [tmp_13083[0] * 6014095723071284775, tmp_13083[1] * 6014095723071284775, tmp_13083[2] * 6014095723071284775];
    signal tmp_13167[3] <== [tmp_13086[0] + tmp_13166[0], tmp_13086[1] + tmp_13166[1], tmp_13086[2] + tmp_13166[2]];
    signal tmp_13168[3] <== [11235818329862088340 * tmp_13167[0], 11235818329862088340 * tmp_13167[1], 11235818329862088340 * tmp_13167[2]];
    signal tmp_13169[3] <== [tmp_13164[0] + tmp_13168[0], tmp_13164[1] + tmp_13168[1], tmp_13164[2] + tmp_13168[2]];
    signal tmp_13171[3] <== [tmp_13083[0] * 9204778852131994952, tmp_13083[1] * 9204778852131994952, tmp_13083[2] * 9204778852131994952];
    signal tmp_13172[3] <== [tmp_13091[0] + tmp_13171[0], tmp_13091[1] + tmp_13171[1], tmp_13091[2] + tmp_13171[2]];
    signal tmp_13173[3] <== [14470889044174589649 * tmp_13172[0], 14470889044174589649 * tmp_13172[1], 14470889044174589649 * tmp_13172[2]];
    signal tmp_13174[3] <== [tmp_13169[0] + tmp_13173[0], tmp_13169[1] + tmp_13173[1], tmp_13169[2] + tmp_13173[2]];
    signal tmp_13176[3] <== [tmp_13083[0] * 4971091590805790158, tmp_13083[1] * 4971091590805790158, tmp_13083[2] * 4971091590805790158];
    signal tmp_13177[3] <== [tmp_13096[0] + tmp_13176[0], tmp_13096[1] + tmp_13176[1], tmp_13096[2] + tmp_13176[2]];
    signal tmp_13178[3] <== [17369024544835219425 * tmp_13177[0], 17369024544835219425 * tmp_13177[1], 17369024544835219425 * tmp_13177[2]];
    signal tmp_13179[3] <== [tmp_13174[0] + tmp_13178[0], tmp_13174[1] + tmp_13178[1], tmp_13174[2] + tmp_13178[2]];
    signal tmp_13181[3] <== [tmp_13083[0] * 13132173167903710685, tmp_13083[1] * 13132173167903710685, tmp_13083[2] * 13132173167903710685];
    signal tmp_13182[3] <== [tmp_13101[0] + tmp_13181[0], tmp_13101[1] + tmp_13181[1], tmp_13101[2] + tmp_13181[2]];
    signal tmp_13183[3] <== [15549761383035162395 * tmp_13182[0], 15549761383035162395 * tmp_13182[1], 15549761383035162395 * tmp_13182[2]];
    signal tmp_13184[3] <== [tmp_13179[0] + tmp_13183[0], tmp_13179[1] + tmp_13183[1], tmp_13179[2] + tmp_13183[2]];
    signal tmp_13186[3] <== [tmp_13083[0] * 7151135351816094214, tmp_13083[1] * 7151135351816094214, tmp_13083[2] * 7151135351816094214];
    signal tmp_13187[3] <== [tmp_13106[0] + tmp_13186[0], tmp_13106[1] + tmp_13186[1], tmp_13106[2] + tmp_13186[2]];
    signal tmp_13188[3] <== [10101825905546027637 * tmp_13187[0], 10101825905546027637 * tmp_13187[1], 10101825905546027637 * tmp_13187[2]];
    signal tmp_13189[3] <== [tmp_13184[0] + tmp_13188[0], tmp_13184[1] + tmp_13188[1], tmp_13184[2] + tmp_13188[2]];
    signal tmp_13191[3] <== [tmp_13083[0] * 17995317069560704617, tmp_13083[1] * 17995317069560704617, tmp_13083[2] * 17995317069560704617];
    signal tmp_13192[3] <== [tmp_13111[0] + tmp_13191[0], tmp_13111[1] + tmp_13191[1], tmp_13111[2] + tmp_13191[2]];
    signal tmp_13193[3] <== [6574923001367612596 * tmp_13192[0], 6574923001367612596 * tmp_13192[1], 6574923001367612596 * tmp_13192[2]];
    signal tmp_13194[3] <== [tmp_13189[0] + tmp_13193[0], tmp_13189[1] + tmp_13193[1], tmp_13189[2] + tmp_13193[2]];
    signal tmp_13196[3] <== [tmp_13083[0] * 3798127953781171227, tmp_13083[1] * 3798127953781171227, tmp_13083[2] * 3798127953781171227];
    signal tmp_13197[3] <== [tmp_13116[0] + tmp_13196[0], tmp_13116[1] + tmp_13196[1], tmp_13116[2] + tmp_13196[2]];
    signal tmp_13198[3] <== [1494928778108945573 * tmp_13197[0], 1494928778108945573 * tmp_13197[1], 1494928778108945573 * tmp_13197[2]];
    signal tmp_13199[3] <== [tmp_13194[0] + tmp_13198[0], tmp_13194[1] + tmp_13198[1], tmp_13194[2] + tmp_13198[2]];
    signal tmp_13201[3] <== [tmp_13083[0] * 4569124295887334937, tmp_13083[1] * 4569124295887334937, tmp_13083[2] * 4569124295887334937];
    signal tmp_13202[3] <== [tmp_13121[0] + tmp_13201[0], tmp_13121[1] + tmp_13201[1], tmp_13121[2] + tmp_13201[2]];
    signal tmp_13203[3] <== [4609820205985602917 * tmp_13202[0], 4609820205985602917 * tmp_13202[1], 4609820205985602917 * tmp_13202[2]];
    signal tmp_13204[3] <== [tmp_13199[0] + tmp_13203[0], tmp_13199[1] + tmp_13203[1], tmp_13199[2] + tmp_13203[2]];
    signal tmp_13206[3] <== [tmp_13083[0] * 8937451988206592672, tmp_13083[1] * 8937451988206592672, tmp_13083[2] * 8937451988206592672];
    signal tmp_13207[3] <== [tmp_13126[0] + tmp_13206[0], tmp_13126[1] + tmp_13206[1], tmp_13126[2] + tmp_13206[2]];
    signal tmp_13208[3] <== [3061013547631929280 * tmp_13207[0], 3061013547631929280 * tmp_13207[1], 3061013547631929280 * tmp_13207[2]];
    signal tmp_13209[3] <== [tmp_13204[0] + tmp_13208[0], tmp_13204[1] + tmp_13208[1], tmp_13204[2] + tmp_13208[2]];
    signal tmp_13211[3] <== [tmp_13083[0] * 3515999524512876842, tmp_13083[1] * 3515999524512876842, tmp_13083[2] * 3515999524512876842];
    signal tmp_13212[3] <== [tmp_13131[0] + tmp_13211[0], tmp_13131[1] + tmp_13211[1], tmp_13131[2] + tmp_13211[2]];
    signal tmp_13213[3] <== [17109849163472234437 * tmp_13212[0], 17109849163472234437 * tmp_13212[1], 17109849163472234437 * tmp_13212[2]];
    signal tmp_13214[3] <== [tmp_13209[0] + tmp_13213[0], tmp_13209[1] + tmp_13213[1], tmp_13209[2] + tmp_13213[2]];
    signal tmp_13216[3] <== [tmp_13083[0] * 16966389141222274672, tmp_13083[1] * 16966389141222274672, tmp_13083[2] * 16966389141222274672];
    signal tmp_13217[3] <== [tmp_13136[0] + tmp_13216[0], tmp_13136[1] + tmp_13216[1], tmp_13136[2] + tmp_13216[2]];
    signal tmp_13218[3] <== [11697494665722803962 * tmp_13217[0], 11697494665722803962 * tmp_13217[1], 11697494665722803962 * tmp_13217[2]];
    signal tmp_13219[3] <== [tmp_13214[0] + tmp_13218[0], tmp_13214[1] + tmp_13218[1], tmp_13214[2] + tmp_13218[2]];
    signal tmp_13221[3] <== [tmp_13083[0] * 8034835433986334001, tmp_13083[1] * 8034835433986334001, tmp_13083[2] * 8034835433986334001];
    signal tmp_13222[3] <== [tmp_13141[0] + tmp_13221[0], tmp_13141[1] + tmp_13221[1], tmp_13141[2] + tmp_13221[2]];
    signal tmp_13223[3] <== [4706605688950171455 * tmp_13222[0], 4706605688950171455 * tmp_13222[1], 4706605688950171455 * tmp_13222[2]];
    signal tmp_13224[3] <== [tmp_13219[0] + tmp_13223[0], tmp_13219[1] + tmp_13223[1], tmp_13219[2] + tmp_13223[2]];
    signal tmp_13226[3] <== [tmp_13083[0] * 7985528271787349844, tmp_13083[1] * 7985528271787349844, tmp_13083[2] * 7985528271787349844];
    signal tmp_13227[3] <== [tmp_13146[0] + tmp_13226[0], tmp_13146[1] + tmp_13226[1], tmp_13146[2] + tmp_13226[2]];
    signal tmp_13228[3] <== [15438292502104786232 * tmp_13227[0], 15438292502104786232 * tmp_13227[1], 15438292502104786232 * tmp_13227[2]];
    signal tmp_13229[3] <== [tmp_13224[0] + tmp_13228[0], tmp_13224[1] + tmp_13228[1], tmp_13224[2] + tmp_13228[2]];
    signal tmp_13231[3] <== [tmp_13083[0] * 1825496924806641273, tmp_13083[1] * 1825496924806641273, tmp_13083[2] * 1825496924806641273];
    signal tmp_13232[3] <== [tmp_13151[0] + tmp_13231[0], tmp_13151[1] + tmp_13231[1], tmp_13151[2] + tmp_13231[2]];
    signal tmp_13233[3] <== [1979256218175145661 * tmp_13232[0], 1979256218175145661 * tmp_13232[1], 1979256218175145661 * tmp_13232[2]];
    signal tmp_13234[3] <== [tmp_13229[0] + tmp_13233[0], tmp_13229[1] + tmp_13233[1], tmp_13229[2] + tmp_13233[2]];
    signal tmp_13236[3] <== [tmp_13083[0] * 5094552554489483119, tmp_13083[1] * 5094552554489483119, tmp_13083[2] * 5094552554489483119];
    signal tmp_13237[3] <== [tmp_13156[0] + tmp_13236[0], tmp_13156[1] + tmp_13236[1], tmp_13156[2] + tmp_13236[2]];
    signal tmp_13238[3] <== [6777118549370094671 * tmp_13237[0], 6777118549370094671 * tmp_13237[1], 6777118549370094671 * tmp_13237[2]];
    signal tmp_13239[3] <== [tmp_13234[0] + tmp_13238[0], tmp_13234[1] + tmp_13238[1], tmp_13234[2] + tmp_13238[2]];
    signal tmp_13240[3] <== [tmp_13239[0] * 1, tmp_13239[1] * 1, tmp_13239[2] * 1];
    signal tmp_13241[3] <== [evals[111][0] - tmp_13240[0], evals[111][1] - tmp_13240[1], evals[111][2] - tmp_13240[2]];
    signal tmp_13242[3] <== CMul()(tmp_9163, tmp_13241);
    signal tmp_13243[3] <== [tmp_13163[0] + tmp_13242[0], tmp_13163[1] + tmp_13242[1], tmp_13163[2] + tmp_13242[2]];
    signal tmp_13244[3] <== CMul()(challengeQ, tmp_13243);
    signal tmp_13245[3] <== [tmp_10658[0] + 1773355790966798977, tmp_10658[1], tmp_10658[2]];
    signal tmp_13247[3] <== [tmp_13164[0] * 13058054427636566156, tmp_13164[1] * 13058054427636566156, tmp_13164[2] * 13058054427636566156];
    signal tmp_13248[3] <== [tmp_13167[0] + tmp_13247[0], tmp_13167[1] + tmp_13247[1], tmp_13167[2] + tmp_13247[2]];
    signal tmp_13249[3] <== [8370938923534542930 * tmp_13248[0], 8370938923534542930 * tmp_13248[1], 8370938923534542930 * tmp_13248[2]];
    signal tmp_13250[3] <== [tmp_13245[0] + tmp_13249[0], tmp_13245[1] + tmp_13249[1], tmp_13245[2] + tmp_13249[2]];
    signal tmp_13252[3] <== [tmp_13164[0] * 17494898710713280458, tmp_13164[1] * 17494898710713280458, tmp_13164[2] * 17494898710713280458];
    signal tmp_13253[3] <== [tmp_13172[0] + tmp_13252[0], tmp_13172[1] + tmp_13252[1], tmp_13172[2] + tmp_13252[2]];
    signal tmp_13254[3] <== [1280128970885626285 * tmp_13253[0], 1280128970885626285 * tmp_13253[1], 1280128970885626285 * tmp_13253[2]];
    signal tmp_13255[3] <== [tmp_13250[0] + tmp_13254[0], tmp_13250[1] + tmp_13254[1], tmp_13250[2] + tmp_13254[2]];
    signal tmp_13257[3] <== [tmp_13164[0] * 10556138419339233346, tmp_13164[1] * 10556138419339233346, tmp_13164[2] * 10556138419339233346];
    signal tmp_13258[3] <== [tmp_13177[0] + tmp_13257[0], tmp_13177[1] + tmp_13257[1], tmp_13177[2] + tmp_13257[2]];
    signal tmp_13259[3] <== [17941471055037515299 * tmp_13258[0], 17941471055037515299 * tmp_13258[1], 17941471055037515299 * tmp_13258[2]];
    signal tmp_13260[3] <== [tmp_13255[0] + tmp_13259[0], tmp_13255[1] + tmp_13259[1], tmp_13255[2] + tmp_13259[2]];
    signal tmp_13262[3] <== [tmp_13164[0] * 9446845158877430994, tmp_13164[1] * 9446845158877430994, tmp_13164[2] * 9446845158877430994];
    signal tmp_13263[3] <== [tmp_13182[0] + tmp_13262[0], tmp_13182[1] + tmp_13262[1], tmp_13182[2] + tmp_13262[2]];
    signal tmp_13264[3] <== [8584145824237830166 * tmp_13263[0], 8584145824237830166 * tmp_13263[1], 8584145824237830166 * tmp_13263[2]];
    signal tmp_13265[3] <== [tmp_13260[0] + tmp_13264[0], tmp_13260[1] + tmp_13264[1], tmp_13260[2] + tmp_13264[2]];
    signal tmp_13267[3] <== [tmp_13164[0] * 8319398780236037084, tmp_13164[1] * 8319398780236037084, tmp_13164[2] * 8319398780236037084];
    signal tmp_13268[3] <== [tmp_13187[0] + tmp_13267[0], tmp_13187[1] + tmp_13267[1], tmp_13187[2] + tmp_13267[2]];
    signal tmp_13269[3] <== [8210952786580820323 * tmp_13268[0], 8210952786580820323 * tmp_13268[1], 8210952786580820323 * tmp_13268[2]];
    signal tmp_13270[3] <== [tmp_13265[0] + tmp_13269[0], tmp_13265[1] + tmp_13269[1], tmp_13265[2] + tmp_13269[2]];
    signal tmp_13272[3] <== [tmp_13164[0] * 13582814433783971289, tmp_13164[1] * 13582814433783971289, tmp_13164[2] * 13582814433783971289];
    signal tmp_13273[3] <== [tmp_13192[0] + tmp_13272[0], tmp_13192[1] + tmp_13272[1], tmp_13192[2] + tmp_13272[2]];
    signal tmp_13274[3] <== [2433132954413441756 * tmp_13273[0], 2433132954413441756 * tmp_13273[1], 2433132954413441756 * tmp_13273[2]];
    signal tmp_13275[3] <== [tmp_13270[0] + tmp_13274[0], tmp_13270[1] + tmp_13274[1], tmp_13270[2] + tmp_13274[2]];
    signal tmp_13277[3] <== [tmp_13164[0] * 15000404693370233461, tmp_13164[1] * 15000404693370233461, tmp_13164[2] * 15000404693370233461];
    signal tmp_13278[3] <== [tmp_13197[0] + tmp_13277[0], tmp_13197[1] + tmp_13277[1], tmp_13197[2] + tmp_13277[2]];
    signal tmp_13279[3] <== [15662973610146691206 * tmp_13278[0], 15662973610146691206 * tmp_13278[1], 15662973610146691206 * tmp_13278[2]];
    signal tmp_13280[3] <== [tmp_13275[0] + tmp_13279[0], tmp_13275[1] + tmp_13279[1], tmp_13275[2] + tmp_13279[2]];
    signal tmp_13282[3] <== [tmp_13164[0] * 10753866658284722577, tmp_13164[1] * 10753866658284722577, tmp_13164[2] * 10753866658284722577];
    signal tmp_13283[3] <== [tmp_13202[0] + tmp_13282[0], tmp_13202[1] + tmp_13282[1], tmp_13202[2] + tmp_13282[2]];
    signal tmp_13284[3] <== [10026165261844553317 * tmp_13283[0], 10026165261844553317 * tmp_13283[1], 10026165261844553317 * tmp_13283[2]];
    signal tmp_13285[3] <== [tmp_13280[0] + tmp_13284[0], tmp_13280[1] + tmp_13284[1], tmp_13280[2] + tmp_13284[2]];
    signal tmp_13287[3] <== [tmp_13164[0] * 7462165540945749814, tmp_13164[1] * 7462165540945749814, tmp_13164[2] * 7462165540945749814];
    signal tmp_13288[3] <== [tmp_13207[0] + tmp_13287[0], tmp_13207[1] + tmp_13287[1], tmp_13207[2] + tmp_13287[2]];
    signal tmp_13289[3] <== [16296674388465118728 * tmp_13288[0], 16296674388465118728 * tmp_13288[1], 16296674388465118728 * tmp_13288[2]];
    signal tmp_13290[3] <== [tmp_13285[0] + tmp_13289[0], tmp_13285[1] + tmp_13289[1], tmp_13285[2] + tmp_13289[2]];
    signal tmp_13292[3] <== [tmp_13164[0] * 3904712121826807540, tmp_13164[1] * 3904712121826807540, tmp_13164[2] * 3904712121826807540];
    signal tmp_13293[3] <== [tmp_13212[0] + tmp_13292[0], tmp_13212[1] + tmp_13292[1], tmp_13212[2] + tmp_13292[2]];
    signal tmp_13294[3] <== [17271463318160608022 * tmp_13293[0], 17271463318160608022 * tmp_13293[1], 17271463318160608022 * tmp_13293[2]];
    signal tmp_13295[3] <== [tmp_13290[0] + tmp_13294[0], tmp_13290[1] + tmp_13294[1], tmp_13290[2] + tmp_13294[2]];
    signal tmp_13297[3] <== [tmp_13164[0] * 15631290702336306626, tmp_13164[1] * 15631290702336306626, tmp_13164[2] * 15631290702336306626];
    signal tmp_13298[3] <== [tmp_13217[0] + tmp_13297[0], tmp_13217[1] + tmp_13297[1], tmp_13217[2] + tmp_13297[2]];
    signal tmp_13299[3] <== [8140140055656693761 * tmp_13298[0], 8140140055656693761 * tmp_13298[1], 8140140055656693761 * tmp_13298[2]];
    signal tmp_13300[3] <== [tmp_13295[0] + tmp_13299[0], tmp_13295[1] + tmp_13299[1], tmp_13295[2] + tmp_13299[2]];
    signal tmp_13302[3] <== [tmp_13164[0] * 8501326043416122835, tmp_13164[1] * 8501326043416122835, tmp_13164[2] * 8501326043416122835];
    signal tmp_13303[3] <== [tmp_13222[0] + tmp_13302[0], tmp_13222[1] + tmp_13302[1], tmp_13222[2] + tmp_13302[2]];
    signal tmp_13304[3] <== [3268211924158099194 * tmp_13303[0], 3268211924158099194 * tmp_13303[1], 3268211924158099194 * tmp_13303[2]];
    signal tmp_13305[3] <== [tmp_13300[0] + tmp_13304[0], tmp_13300[1] + tmp_13304[1], tmp_13300[2] + tmp_13304[2]];
    signal tmp_13307[3] <== [tmp_13164[0] * 15945260208470094587, tmp_13164[1] * 15945260208470094587, tmp_13164[2] * 15945260208470094587];
    signal tmp_13308[3] <== [tmp_13227[0] + tmp_13307[0], tmp_13227[1] + tmp_13307[1], tmp_13227[2] + tmp_13307[2]];
    signal tmp_13309[3] <== [17850261561957541568 * tmp_13308[0], 17850261561957541568 * tmp_13308[1], 17850261561957541568 * tmp_13308[2]];
    signal tmp_13310[3] <== [tmp_13305[0] + tmp_13309[0], tmp_13305[1] + tmp_13309[1], tmp_13305[2] + tmp_13309[2]];
    signal tmp_13312[3] <== [tmp_13164[0] * 8246979520807589148, tmp_13164[1] * 8246979520807589148, tmp_13164[2] * 8246979520807589148];
    signal tmp_13313[3] <== [tmp_13232[0] + tmp_13312[0], tmp_13232[1] + tmp_13312[1], tmp_13232[2] + tmp_13312[2]];
    signal tmp_13314[3] <== [5233791213354052332 * tmp_13313[0], 5233791213354052332 * tmp_13313[1], 5233791213354052332 * tmp_13313[2]];
    signal tmp_13315[3] <== [tmp_13310[0] + tmp_13314[0], tmp_13310[1] + tmp_13314[1], tmp_13310[2] + tmp_13314[2]];
    signal tmp_13317[3] <== [tmp_13164[0] * 13778737560392521094, tmp_13164[1] * 13778737560392521094, tmp_13164[2] * 13778737560392521094];
    signal tmp_13318[3] <== [tmp_13237[0] + tmp_13317[0], tmp_13237[1] + tmp_13317[1], tmp_13237[2] + tmp_13317[2]];
    signal tmp_13319[3] <== [9957237719655515451 * tmp_13318[0], 9957237719655515451 * tmp_13318[1], 9957237719655515451 * tmp_13318[2]];
    signal tmp_13320[3] <== [tmp_13315[0] + tmp_13319[0], tmp_13315[1] + tmp_13319[1], tmp_13315[2] + tmp_13319[2]];
    signal tmp_13321[3] <== [tmp_13320[0] * 1, tmp_13320[1] * 1, tmp_13320[2] * 1];
    signal tmp_13322[3] <== [evals[112][0] - tmp_13321[0], evals[112][1] - tmp_13321[1], evals[112][2] - tmp_13321[2]];
    signal tmp_13323[3] <== CMul()(tmp_9163, tmp_13322);
    signal tmp_13324[3] <== [tmp_13244[0] + tmp_13323[0], tmp_13244[1] + tmp_13323[1], tmp_13244[2] + tmp_13323[2]];
    signal tmp_13325[3] <== CMul()(challengeQ, tmp_13324);
    signal tmp_13326[3] <== [tmp_10670[0] + 14396910818521180817, tmp_10670[1], tmp_10670[2]];
    signal tmp_13328[3] <== [tmp_13245[0] * 14943873981466403895, tmp_13245[1] * 14943873981466403895, tmp_13245[2] * 14943873981466403895];
    signal tmp_13329[3] <== [tmp_13248[0] + tmp_13328[0], tmp_13248[1] + tmp_13328[1], tmp_13248[2] + tmp_13328[2]];
    signal tmp_13330[3] <== [2284659241919280696 * tmp_13329[0], 2284659241919280696 * tmp_13329[1], 2284659241919280696 * tmp_13329[2]];
    signal tmp_13331[3] <== [tmp_13326[0] + tmp_13330[0], tmp_13326[1] + tmp_13330[1], tmp_13326[2] + tmp_13330[2]];
    signal tmp_13333[3] <== [tmp_13245[0] * 7858901551218927609, tmp_13245[1] * 7858901551218927609, tmp_13245[2] * 7858901551218927609];
    signal tmp_13334[3] <== [tmp_13253[0] + tmp_13333[0], tmp_13253[1] + tmp_13333[1], tmp_13253[2] + tmp_13333[2]];
    signal tmp_13335[3] <== [1847394796741791100 * tmp_13334[0], 1847394796741791100 * tmp_13334[1], 1847394796741791100 * tmp_13334[2]];
    signal tmp_13336[3] <== [tmp_13331[0] + tmp_13335[0], tmp_13331[1] + tmp_13335[1], tmp_13331[2] + tmp_13335[2]];
    signal tmp_13338[3] <== [tmp_13245[0] * 12183924058797585540, tmp_13245[1] * 12183924058797585540, tmp_13245[2] * 12183924058797585540];
    signal tmp_13339[3] <== [tmp_13258[0] + tmp_13338[0], tmp_13258[1] + tmp_13338[1], tmp_13258[2] + tmp_13338[2]];
    signal tmp_13340[3] <== [17083607968162884940 * tmp_13339[0], 17083607968162884940 * tmp_13339[1], 17083607968162884940 * tmp_13339[2]];
    signal tmp_13341[3] <== [tmp_13336[0] + tmp_13340[0], tmp_13336[1] + tmp_13340[1], tmp_13336[2] + tmp_13340[2]];
    signal tmp_13343[3] <== [tmp_13245[0] * 11287081087228760936, tmp_13245[1] * 11287081087228760936, tmp_13245[2] * 11287081087228760936];
    signal tmp_13344[3] <== [tmp_13263[0] + tmp_13343[0], tmp_13263[1] + tmp_13343[1], tmp_13263[2] + tmp_13343[2]];
    signal tmp_13345[3] <== [14649174299425920385 * tmp_13344[0], 14649174299425920385 * tmp_13344[1], 14649174299425920385 * tmp_13344[2]];
    signal tmp_13346[3] <== [tmp_13341[0] + tmp_13345[0], tmp_13341[1] + tmp_13345[1], tmp_13341[2] + tmp_13345[2]];
    signal tmp_13348[3] <== [tmp_13245[0] * 1768132666423206446, tmp_13245[1] * 1768132666423206446, tmp_13245[2] * 1768132666423206446];
    signal tmp_13349[3] <== [tmp_13268[0] + tmp_13348[0], tmp_13268[1] + tmp_13348[1], tmp_13268[2] + tmp_13348[2]];
    signal tmp_13350[3] <== [16742036463763035056 * tmp_13349[0], 16742036463763035056 * tmp_13349[1], 16742036463763035056 * tmp_13349[2]];
    signal tmp_13351[3] <== [tmp_13346[0] + tmp_13350[0], tmp_13346[1] + tmp_13350[1], tmp_13346[2] + tmp_13350[2]];
    signal tmp_13353[3] <== [tmp_13245[0] * 13818240357693031756, tmp_13245[1] * 13818240357693031756, tmp_13245[2] * 13818240357693031756];
    signal tmp_13354[3] <== [tmp_13273[0] + tmp_13353[0], tmp_13273[1] + tmp_13353[1], tmp_13273[2] + tmp_13353[2]];
    signal tmp_13355[3] <== [4411458753033679918 * tmp_13354[0], 4411458753033679918 * tmp_13354[1], 4411458753033679918 * tmp_13354[2]];
    signal tmp_13356[3] <== [tmp_13351[0] + tmp_13355[0], tmp_13351[1] + tmp_13355[1], tmp_13351[2] + tmp_13355[2]];
    signal tmp_13358[3] <== [tmp_13245[0] * 13951006846344947715, tmp_13245[1] * 13951006846344947715, tmp_13245[2] * 13951006846344947715];
    signal tmp_13359[3] <== [tmp_13278[0] + tmp_13358[0], tmp_13278[1] + tmp_13358[1], tmp_13278[2] + tmp_13358[2]];
    signal tmp_13360[3] <== [8754998449733673829 * tmp_13359[0], 8754998449733673829 * tmp_13359[1], 8754998449733673829 * tmp_13359[2]];
    signal tmp_13361[3] <== [tmp_13356[0] + tmp_13360[0], tmp_13356[1] + tmp_13360[1], tmp_13356[2] + tmp_13360[2]];
    signal tmp_13363[3] <== [tmp_13245[0] * 7593135614815777901, tmp_13245[1] * 7593135614815777901, tmp_13245[2] * 7593135614815777901];
    signal tmp_13364[3] <== [tmp_13283[0] + tmp_13363[0], tmp_13283[1] + tmp_13363[1], tmp_13283[2] + tmp_13363[2]];
    signal tmp_13365[3] <== [7535759880903270203 * tmp_13364[0], 7535759880903270203 * tmp_13364[1], 7535759880903270203 * tmp_13364[2]];
    signal tmp_13366[3] <== [tmp_13361[0] + tmp_13365[0], tmp_13361[1] + tmp_13365[1], tmp_13361[2] + tmp_13365[2]];
    signal tmp_13368[3] <== [tmp_13245[0] * 13292507633934451261, tmp_13245[1] * 13292507633934451261, tmp_13245[2] * 13292507633934451261];
    signal tmp_13369[3] <== [tmp_13288[0] + tmp_13368[0], tmp_13288[1] + tmp_13368[1], tmp_13288[2] + tmp_13368[2]];
    signal tmp_13370[3] <== [8448917368593535896 * tmp_13369[0], 8448917368593535896 * tmp_13369[1], 8448917368593535896 * tmp_13369[2]];
    signal tmp_13371[3] <== [tmp_13366[0] + tmp_13370[0], tmp_13366[1] + tmp_13370[1], tmp_13366[2] + tmp_13370[2]];
    signal tmp_13373[3] <== [tmp_13245[0] * 605032684208182921, tmp_13245[1] * 605032684208182921, tmp_13245[2] * 605032684208182921];
    signal tmp_13374[3] <== [tmp_13293[0] + tmp_13373[0], tmp_13293[1] + tmp_13373[1], tmp_13293[2] + tmp_13373[2]];
    signal tmp_13375[3] <== [16823524365863196542 * tmp_13374[0], 16823524365863196542 * tmp_13374[1], 16823524365863196542 * tmp_13374[2]];
    signal tmp_13376[3] <== [tmp_13371[0] + tmp_13375[0], tmp_13371[1] + tmp_13375[1], tmp_13371[2] + tmp_13375[2]];
    signal tmp_13378[3] <== [tmp_13245[0] * 6876471548335849320, tmp_13245[1] * 6876471548335849320, tmp_13245[2] * 6876471548335849320];
    signal tmp_13379[3] <== [tmp_13298[0] + tmp_13378[0], tmp_13298[1] + tmp_13378[1], tmp_13298[2] + tmp_13378[2]];
    signal tmp_13380[3] <== [15596667798790275845 * tmp_13379[0], 15596667798790275845 * tmp_13379[1], 15596667798790275845 * tmp_13379[2]];
    signal tmp_13381[3] <== [tmp_13376[0] + tmp_13380[0], tmp_13376[1] + tmp_13380[1], tmp_13376[2] + tmp_13380[2]];
    signal tmp_13383[3] <== [tmp_13245[0] * 14846539071992201527, tmp_13245[1] * 14846539071992201527, tmp_13245[2] * 14846539071992201527];
    signal tmp_13384[3] <== [tmp_13303[0] + tmp_13383[0], tmp_13303[1] + tmp_13383[1], tmp_13303[2] + tmp_13383[2]];
    signal tmp_13385[3] <== [1187691146794617430 * tmp_13384[0], 1187691146794617430 * tmp_13384[1], 1187691146794617430 * tmp_13384[2]];
    signal tmp_13386[3] <== [tmp_13381[0] + tmp_13385[0], tmp_13381[1] + tmp_13385[1], tmp_13381[2] + tmp_13385[2]];
    signal tmp_13388[3] <== [tmp_13245[0] * 9468985681611683708, tmp_13245[1] * 9468985681611683708, tmp_13245[2] * 9468985681611683708];
    signal tmp_13389[3] <== [tmp_13308[0] + tmp_13388[0], tmp_13308[1] + tmp_13388[1], tmp_13308[2] + tmp_13388[2]];
    signal tmp_13390[3] <== [11952521614018033253 * tmp_13389[0], 11952521614018033253 * tmp_13389[1], 11952521614018033253 * tmp_13389[2]];
    signal tmp_13391[3] <== [tmp_13386[0] + tmp_13390[0], tmp_13386[1] + tmp_13390[1], tmp_13386[2] + tmp_13390[2]];
    signal tmp_13393[3] <== [tmp_13245[0] * 16311784362809925742, tmp_13245[1] * 16311784362809925742, tmp_13245[2] * 16311784362809925742];
    signal tmp_13394[3] <== [tmp_13313[0] + tmp_13393[0], tmp_13313[1] + tmp_13393[1], tmp_13313[2] + tmp_13393[2]];
    signal tmp_13395[3] <== [9669076931479314330 * tmp_13394[0], 9669076931479314330 * tmp_13394[1], 9669076931479314330 * tmp_13394[2]];
    signal tmp_13396[3] <== [tmp_13391[0] + tmp_13395[0], tmp_13391[1] + tmp_13395[1], tmp_13391[2] + tmp_13395[2]];
    signal tmp_13398[3] <== [tmp_13245[0] * 3645094389087780366, tmp_13245[1] * 3645094389087780366, tmp_13245[2] * 3645094389087780366];
    signal tmp_13399[3] <== [tmp_13318[0] + tmp_13398[0], tmp_13318[1] + tmp_13398[1], tmp_13318[2] + tmp_13398[2]];
    signal tmp_13400[3] <== [12896624684325687180 * tmp_13399[0], 12896624684325687180 * tmp_13399[1], 12896624684325687180 * tmp_13399[2]];
    signal tmp_13401[3] <== [tmp_13396[0] + tmp_13400[0], tmp_13396[1] + tmp_13400[1], tmp_13396[2] + tmp_13400[2]];
    signal tmp_13402[3] <== [tmp_13401[0] * 1, tmp_13401[1] * 1, tmp_13401[2] * 1];
    signal tmp_13403[3] <== [evals[113][0] - tmp_13402[0], evals[113][1] - tmp_13402[1], evals[113][2] - tmp_13402[2]];
    signal tmp_13404[3] <== CMul()(tmp_9163, tmp_13403);
    signal tmp_13405[3] <== [tmp_13325[0] + tmp_13404[0], tmp_13325[1] + tmp_13404[1], tmp_13325[2] + tmp_13404[2]];
    signal tmp_13406[3] <== CMul()(challengeQ, tmp_13405);
    signal tmp_13407[3] <== [tmp_10682[0] + 3603452348219013210, tmp_10682[1], tmp_10682[2]];
    signal tmp_13409[3] <== [tmp_13326[0] * 4167332706053537795, tmp_13326[1] * 4167332706053537795, tmp_13326[2] * 4167332706053537795];
    signal tmp_13410[3] <== [tmp_13329[0] + tmp_13409[0], tmp_13329[1] + tmp_13409[1], tmp_13329[2] + tmp_13409[2]];
    signal tmp_13411[3] <== [6292265836956544172 * tmp_13410[0], 6292265836956544172 * tmp_13410[1], 6292265836956544172 * tmp_13410[2]];
    signal tmp_13412[3] <== [tmp_13407[0] + tmp_13411[0], tmp_13407[1] + tmp_13411[1], tmp_13407[2] + tmp_13411[2]];
    signal tmp_13414[3] <== [tmp_13326[0] * 1835961824923514021, tmp_13326[1] * 1835961824923514021, tmp_13326[2] * 1835961824923514021];
    signal tmp_13415[3] <== [tmp_13334[0] + tmp_13414[0], tmp_13334[1] + tmp_13414[1], tmp_13334[2] + tmp_13414[2]];
    signal tmp_13416[3] <== [15025127703762308473 * tmp_13415[0], 15025127703762308473 * tmp_13415[1], 15025127703762308473 * tmp_13415[2]];
    signal tmp_13417[3] <== [tmp_13412[0] + tmp_13416[0], tmp_13412[1] + tmp_13416[1], tmp_13412[2] + tmp_13416[2]];
    signal tmp_13419[3] <== [tmp_13326[0] * 2966705105152395108, tmp_13326[1] * 2966705105152395108, tmp_13326[2] * 2966705105152395108];
    signal tmp_13420[3] <== [tmp_13339[0] + tmp_13419[0], tmp_13339[1] + tmp_13419[1], tmp_13339[2] + tmp_13419[2]];
    signal tmp_13421[3] <== [1549382727242080154 * tmp_13420[0], 1549382727242080154 * tmp_13420[1], 1549382727242080154 * tmp_13420[2]];
    signal tmp_13422[3] <== [tmp_13417[0] + tmp_13421[0], tmp_13417[1] + tmp_13421[1], tmp_13417[2] + tmp_13421[2]];
    signal tmp_13424[3] <== [tmp_13326[0] * 973577938650894194, tmp_13326[1] * 973577938650894194, tmp_13326[2] * 973577938650894194];
    signal tmp_13425[3] <== [tmp_13344[0] + tmp_13424[0], tmp_13344[1] + tmp_13424[1], tmp_13344[2] + tmp_13424[2]];
    signal tmp_13426[3] <== [9995718145205770324 * tmp_13425[0], 9995718145205770324 * tmp_13425[1], 9995718145205770324 * tmp_13425[2]];
    signal tmp_13427[3] <== [tmp_13422[0] + tmp_13426[0], tmp_13422[1] + tmp_13426[1], tmp_13422[2] + tmp_13426[2]];
    signal tmp_13429[3] <== [tmp_13326[0] * 14560278670602761416, tmp_13326[1] * 14560278670602761416, tmp_13326[2] * 14560278670602761416];
    signal tmp_13430[3] <== [tmp_13349[0] + tmp_13429[0], tmp_13349[1] + tmp_13429[1], tmp_13349[2] + tmp_13429[2]];
    signal tmp_13431[3] <== [16328597422766744836 * tmp_13430[0], 16328597422766744836 * tmp_13430[1], 16328597422766744836 * tmp_13430[2]];
    signal tmp_13432[3] <== [tmp_13427[0] + tmp_13431[0], tmp_13427[1] + tmp_13431[1], tmp_13427[2] + tmp_13431[2]];
    signal tmp_13434[3] <== [tmp_13326[0] * 397212481041799485, tmp_13326[1] * 397212481041799485, tmp_13326[2] * 397212481041799485];
    signal tmp_13435[3] <== [tmp_13354[0] + tmp_13434[0], tmp_13354[1] + tmp_13434[1], tmp_13354[2] + tmp_13434[2]];
    signal tmp_13436[3] <== [14994994730468297602 * tmp_13435[0], 14994994730468297602 * tmp_13435[1], 14994994730468297602 * tmp_13435[2]];
    signal tmp_13437[3] <== [tmp_13432[0] + tmp_13436[0], tmp_13432[1] + tmp_13436[1], tmp_13432[2] + tmp_13436[2]];
    signal tmp_13439[3] <== [tmp_13326[0] * 4220306037658238426, tmp_13326[1] * 4220306037658238426, tmp_13326[2] * 4220306037658238426];
    signal tmp_13440[3] <== [tmp_13359[0] + tmp_13439[0], tmp_13359[1] + tmp_13439[1], tmp_13359[2] + tmp_13439[2]];
    signal tmp_13441[3] <== [5652613299116160344 * tmp_13440[0], 5652613299116160344 * tmp_13440[1], 5652613299116160344 * tmp_13440[2]];
    signal tmp_13442[3] <== [tmp_13437[0] + tmp_13441[0], tmp_13437[1] + tmp_13441[1], tmp_13437[2] + tmp_13441[2]];
    signal tmp_13444[3] <== [tmp_13326[0] * 270403227330701921, tmp_13326[1] * 270403227330701921, tmp_13326[2] * 270403227330701921];
    signal tmp_13445[3] <== [tmp_13364[0] + tmp_13444[0], tmp_13364[1] + tmp_13444[1], tmp_13364[2] + tmp_13444[2]];
    signal tmp_13446[3] <== [2558771485279565859 * tmp_13445[0], 2558771485279565859 * tmp_13445[1], 2558771485279565859 * tmp_13445[2]];
    signal tmp_13447[3] <== [tmp_13442[0] + tmp_13446[0], tmp_13442[1] + tmp_13446[1], tmp_13442[2] + tmp_13446[2]];
    signal tmp_13449[3] <== [tmp_13326[0] * 1308298291121009953, tmp_13326[1] * 1308298291121009953, tmp_13326[2] * 1308298291121009953];
    signal tmp_13450[3] <== [tmp_13369[0] + tmp_13449[0], tmp_13369[1] + tmp_13449[1], tmp_13369[2] + tmp_13449[2]];
    signal tmp_13451[3] <== [6410586043754003422 * tmp_13450[0], 6410586043754003422 * tmp_13450[1], 6410586043754003422 * tmp_13450[2]];
    signal tmp_13452[3] <== [tmp_13447[0] + tmp_13451[0], tmp_13447[1] + tmp_13451[1], tmp_13447[2] + tmp_13451[2]];
    signal tmp_13454[3] <== [tmp_13326[0] * 11943868125602043586, tmp_13326[1] * 11943868125602043586, tmp_13326[2] * 11943868125602043586];
    signal tmp_13455[3] <== [tmp_13374[0] + tmp_13454[0], tmp_13374[1] + tmp_13454[1], tmp_13374[2] + tmp_13454[2]];
    signal tmp_13456[3] <== [5825523164635395574 * tmp_13455[0], 5825523164635395574 * tmp_13455[1], 5825523164635395574 * tmp_13455[2]];
    signal tmp_13457[3] <== [tmp_13452[0] + tmp_13456[0], tmp_13452[1] + tmp_13456[1], tmp_13452[2] + tmp_13456[2]];
    signal tmp_13459[3] <== [tmp_13326[0] * 202395478849929974, tmp_13326[1] * 202395478849929974, tmp_13326[2] * 202395478849929974];
    signal tmp_13460[3] <== [tmp_13379[0] + tmp_13459[0], tmp_13379[1] + tmp_13459[1], tmp_13379[2] + tmp_13459[2]];
    signal tmp_13461[3] <== [2094054892975333961 * tmp_13460[0], 2094054892975333961 * tmp_13460[1], 2094054892975333961 * tmp_13460[2]];
    signal tmp_13462[3] <== [tmp_13457[0] + tmp_13461[0], tmp_13457[1] + tmp_13461[1], tmp_13457[2] + tmp_13461[2]];
    signal tmp_13464[3] <== [tmp_13326[0] * 17179636232347603881, tmp_13326[1] * 17179636232347603881, tmp_13326[2] * 17179636232347603881];
    signal tmp_13465[3] <== [tmp_13384[0] + tmp_13464[0], tmp_13384[1] + tmp_13464[1], tmp_13384[2] + tmp_13464[2]];
    signal tmp_13466[3] <== [4957600268439660491 * tmp_13465[0], 4957600268439660491 * tmp_13465[1], 4957600268439660491 * tmp_13465[2]];
    signal tmp_13467[3] <== [tmp_13462[0] + tmp_13466[0], tmp_13462[1] + tmp_13466[1], tmp_13462[2] + tmp_13466[2]];
    signal tmp_13469[3] <== [tmp_13326[0] * 4425027275485648465, tmp_13326[1] * 4425027275485648465, tmp_13326[2] * 4425027275485648465];
    signal tmp_13470[3] <== [tmp_13389[0] + tmp_13469[0], tmp_13389[1] + tmp_13469[1], tmp_13389[2] + tmp_13469[2]];
    signal tmp_13471[3] <== [12244729162271932904 * tmp_13470[0], 12244729162271932904 * tmp_13470[1], 12244729162271932904 * tmp_13470[2]];
    signal tmp_13472[3] <== [tmp_13467[0] + tmp_13471[0], tmp_13467[1] + tmp_13471[1], tmp_13467[2] + tmp_13471[2]];
    signal tmp_13474[3] <== [tmp_13326[0] * 9617076952929794796, tmp_13326[1] * 9617076952929794796, tmp_13326[2] * 9617076952929794796];
    signal tmp_13475[3] <== [tmp_13394[0] + tmp_13474[0], tmp_13394[1] + tmp_13474[1], tmp_13394[2] + tmp_13474[2]];
    signal tmp_13476[3] <== [722213697716749189 * tmp_13475[0], 722213697716749189 * tmp_13475[1], 722213697716749189 * tmp_13475[2]];
    signal tmp_13477[3] <== [tmp_13472[0] + tmp_13476[0], tmp_13472[1] + tmp_13476[1], tmp_13472[2] + tmp_13476[2]];
    signal tmp_13479[3] <== [tmp_13326[0] * 263806961605756236, tmp_13326[1] * 263806961605756236, tmp_13326[2] * 263806961605756236];
    signal tmp_13480[3] <== [tmp_13399[0] + tmp_13479[0], tmp_13399[1] + tmp_13479[1], tmp_13399[2] + tmp_13479[2]];
    signal tmp_13481[3] <== [9743689501105433264 * tmp_13480[0], 9743689501105433264 * tmp_13480[1], 9743689501105433264 * tmp_13480[2]];
    signal tmp_13482[3] <== [tmp_13477[0] + tmp_13481[0], tmp_13477[1] + tmp_13481[1], tmp_13477[2] + tmp_13481[2]];
    signal tmp_13483[3] <== [tmp_13482[0] * 1, tmp_13482[1] * 1, tmp_13482[2] * 1];
    signal tmp_13484[3] <== [evals[114][0] - tmp_13483[0], evals[114][1] - tmp_13483[1], evals[114][2] - tmp_13483[2]];
    signal tmp_13485[3] <== CMul()(tmp_9163, tmp_13484);
    signal tmp_13486[3] <== [tmp_13406[0] + tmp_13485[0], tmp_13406[1] + tmp_13485[1], tmp_13406[2] + tmp_13485[2]];
    signal tmp_13487[3] <== CMul()(challengeQ, tmp_13486);
    signal tmp_13488[3] <== [tmp_10694[0] + 6021258740816487473, tmp_10694[1], tmp_10694[2]];
    signal tmp_13490[3] <== [tmp_13407[0] * 16189423883189207660, tmp_13407[1] * 16189423883189207660, tmp_13407[2] * 16189423883189207660];
    signal tmp_13491[3] <== [tmp_13410[0] + tmp_13490[0], tmp_13410[1] + tmp_13490[1], tmp_13410[2] + tmp_13490[2]];
    signal tmp_13492[3] <== [10251643030888928632 * tmp_13491[0], 10251643030888928632 * tmp_13491[1], 10251643030888928632 * tmp_13491[2]];
    signal tmp_13493[3] <== [tmp_13488[0] + tmp_13492[0], tmp_13488[1] + tmp_13492[1], tmp_13488[2] + tmp_13492[2]];
    signal tmp_13495[3] <== [tmp_13407[0] * 14579641431978311466, tmp_13407[1] * 14579641431978311466, tmp_13407[2] * 14579641431978311466];
    signal tmp_13496[3] <== [tmp_13415[0] + tmp_13495[0], tmp_13415[1] + tmp_13495[1], tmp_13415[2] + tmp_13495[2]];
    signal tmp_13497[3] <== [14134563997332366361 * tmp_13496[0], 14134563997332366361 * tmp_13496[1], 14134563997332366361 * tmp_13496[2]];
    signal tmp_13498[3] <== [tmp_13493[0] + tmp_13497[0], tmp_13493[1] + tmp_13497[1], tmp_13493[2] + tmp_13497[2]];
    signal tmp_13500[3] <== [tmp_13407[0] * 14758776138231707159, tmp_13407[1] * 14758776138231707159, tmp_13407[2] * 14758776138231707159];
    signal tmp_13501[3] <== [tmp_13420[0] + tmp_13500[0], tmp_13420[1] + tmp_13500[1], tmp_13420[2] + tmp_13500[2]];
    signal tmp_13502[3] <== [14114253400931675624 * tmp_13501[0], 14114253400931675624 * tmp_13501[1], 14114253400931675624 * tmp_13501[2]];
    signal tmp_13503[3] <== [tmp_13498[0] + tmp_13502[0], tmp_13498[1] + tmp_13502[1], tmp_13498[2] + tmp_13502[2]];
    signal tmp_13505[3] <== [tmp_13407[0] * 12860387175592582496, tmp_13407[1] * 12860387175592582496, tmp_13407[2] * 12860387175592582496];
    signal tmp_13506[3] <== [tmp_13425[0] + tmp_13505[0], tmp_13425[1] + tmp_13505[1], tmp_13425[2] + tmp_13505[2]];
    signal tmp_13507[3] <== [3957736064180453745 * tmp_13506[0], 3957736064180453745 * tmp_13506[1], 3957736064180453745 * tmp_13506[2]];
    signal tmp_13508[3] <== [tmp_13503[0] + tmp_13507[0], tmp_13503[1] + tmp_13507[1], tmp_13503[2] + tmp_13507[2]];
    signal tmp_13510[3] <== [tmp_13407[0] * 15858261595013926463, tmp_13407[1] * 15858261595013926463, tmp_13407[2] * 15858261595013926463];
    signal tmp_13511[3] <== [tmp_13430[0] + tmp_13510[0], tmp_13430[1] + tmp_13510[1], tmp_13430[2] + tmp_13510[2]];
    signal tmp_13512[3] <== [8095276956282841616 * tmp_13511[0], 8095276956282841616 * tmp_13511[1], 8095276956282841616 * tmp_13511[2]];
    signal tmp_13513[3] <== [tmp_13508[0] + tmp_13512[0], tmp_13508[1] + tmp_13512[1], tmp_13508[2] + tmp_13512[2]];
    signal tmp_13515[3] <== [tmp_13407[0] * 14181919498407542627, tmp_13407[1] * 14181919498407542627, tmp_13407[2] * 14181919498407542627];
    signal tmp_13516[3] <== [tmp_13435[0] + tmp_13515[0], tmp_13435[1] + tmp_13515[1], tmp_13435[2] + tmp_13515[2]];
    signal tmp_13517[3] <== [4831909714697294330 * tmp_13516[0], 4831909714697294330 * tmp_13516[1], 4831909714697294330 * tmp_13516[2]];
    signal tmp_13518[3] <== [tmp_13513[0] + tmp_13517[0], tmp_13513[1] + tmp_13517[1], tmp_13513[2] + tmp_13517[2]];
    signal tmp_13520[3] <== [tmp_13407[0] * 13242793143982098963, tmp_13407[1] * 13242793143982098963, tmp_13407[2] * 13242793143982098963];
    signal tmp_13521[3] <== [tmp_13440[0] + tmp_13520[0], tmp_13440[1] + tmp_13520[1], tmp_13440[2] + tmp_13520[2]];
    signal tmp_13522[3] <== [1013946114800042668 * tmp_13521[0], 1013946114800042668 * tmp_13521[1], 1013946114800042668 * tmp_13521[2]];
    signal tmp_13523[3] <== [tmp_13518[0] + tmp_13522[0], tmp_13518[1] + tmp_13522[1], tmp_13518[2] + tmp_13522[2]];
    signal tmp_13525[3] <== [tmp_13407[0] * 11497321389989836778, tmp_13407[1] * 11497321389989836778, tmp_13407[2] * 11497321389989836778];
    signal tmp_13526[3] <== [tmp_13445[0] + tmp_13525[0], tmp_13445[1] + tmp_13525[1], tmp_13445[2] + tmp_13525[2]];
    signal tmp_13527[3] <== [18251392153846354131 * tmp_13526[0], 18251392153846354131 * tmp_13526[1], 18251392153846354131 * tmp_13526[2]];
    signal tmp_13528[3] <== [tmp_13523[0] + tmp_13527[0], tmp_13523[1] + tmp_13527[1], tmp_13523[2] + tmp_13527[2]];
    signal tmp_13530[3] <== [tmp_13407[0] * 16137066962847815728, tmp_13407[1] * 16137066962847815728, tmp_13407[2] * 16137066962847815728];
    signal tmp_13531[3] <== [tmp_13450[0] + tmp_13530[0], tmp_13450[1] + tmp_13530[1], tmp_13450[2] + tmp_13530[2]];
    signal tmp_13532[3] <== [2225048953983193835 * tmp_13531[0], 2225048953983193835 * tmp_13531[1], 2225048953983193835 * tmp_13531[2]];
    signal tmp_13533[3] <== [tmp_13528[0] + tmp_13532[0], tmp_13528[1] + tmp_13532[1], tmp_13528[2] + tmp_13532[2]];
    signal tmp_13535[3] <== [tmp_13407[0] * 15056884893890740908, tmp_13407[1] * 15056884893890740908, tmp_13407[2] * 15056884893890740908];
    signal tmp_13536[3] <== [tmp_13455[0] + tmp_13535[0], tmp_13455[1] + tmp_13535[1], tmp_13455[2] + tmp_13535[2]];
    signal tmp_13537[3] <== [1367030755796680700 * tmp_13536[0], 1367030755796680700 * tmp_13536[1], 1367030755796680700 * tmp_13536[2]];
    signal tmp_13538[3] <== [tmp_13533[0] + tmp_13537[0], tmp_13533[1] + tmp_13537[1], tmp_13533[2] + tmp_13537[2]];
    signal tmp_13540[3] <== [tmp_13407[0] * 14449506859106810187, tmp_13407[1] * 14449506859106810187, tmp_13407[2] * 14449506859106810187];
    signal tmp_13541[3] <== [tmp_13460[0] + tmp_13540[0], tmp_13460[1] + tmp_13540[1], tmp_13460[2] + tmp_13540[2]];
    signal tmp_13542[3] <== [7271551381474385426 * tmp_13541[0], 7271551381474385426 * tmp_13541[1], 7271551381474385426 * tmp_13541[2]];
    signal tmp_13543[3] <== [tmp_13538[0] + tmp_13542[0], tmp_13538[1] + tmp_13542[1], tmp_13538[2] + tmp_13542[2]];
    signal tmp_13545[3] <== [tmp_13407[0] * 14294938476450530176, tmp_13407[1] * 14294938476450530176, tmp_13407[2] * 14294938476450530176];
    signal tmp_13546[3] <== [tmp_13465[0] + tmp_13545[0], tmp_13465[1] + tmp_13545[1], tmp_13465[2] + tmp_13545[2]];
    signal tmp_13547[3] <== [2265568358314072054 * tmp_13546[0], 2265568358314072054 * tmp_13546[1], 2265568358314072054 * tmp_13546[2]];
    signal tmp_13548[3] <== [tmp_13543[0] + tmp_13547[0], tmp_13543[1] + tmp_13547[1], tmp_13543[2] + tmp_13547[2]];
    signal tmp_13550[3] <== [tmp_13407[0] * 16127203105503375405, tmp_13407[1] * 16127203105503375405, tmp_13407[2] * 16127203105503375405];
    signal tmp_13551[3] <== [tmp_13470[0] + tmp_13550[0], tmp_13470[1] + tmp_13550[1], tmp_13470[2] + tmp_13550[2]];
    signal tmp_13552[3] <== [16819938243526879572 * tmp_13551[0], 16819938243526879572 * tmp_13551[1], 16819938243526879572 * tmp_13551[2]];
    signal tmp_13553[3] <== [tmp_13548[0] + tmp_13552[0], tmp_13548[1] + tmp_13552[1], tmp_13548[2] + tmp_13552[2]];
    signal tmp_13555[3] <== [tmp_13407[0] * 12902790681243023167, tmp_13407[1] * 12902790681243023167, tmp_13407[2] * 12902790681243023167];
    signal tmp_13556[3] <== [tmp_13475[0] + tmp_13555[0], tmp_13475[1] + tmp_13555[1], tmp_13475[2] + tmp_13555[2]];
    signal tmp_13557[3] <== [5766489137465308436 * tmp_13556[0], 5766489137465308436 * tmp_13556[1], 5766489137465308436 * tmp_13556[2]];
    signal tmp_13558[3] <== [tmp_13553[0] + tmp_13557[0], tmp_13553[1] + tmp_13557[1], tmp_13553[2] + tmp_13557[2]];
    signal tmp_13560[3] <== [tmp_13407[0] * 14802832860197476755, tmp_13407[1] * 14802832860197476755, tmp_13407[2] * 14802832860197476755];
    signal tmp_13561[3] <== [tmp_13480[0] + tmp_13560[0], tmp_13480[1] + tmp_13560[1], tmp_13480[2] + tmp_13560[2]];
    signal tmp_13562[3] <== [11873533365200282612 * tmp_13561[0], 11873533365200282612 * tmp_13561[1], 11873533365200282612 * tmp_13561[2]];
    signal tmp_13563[3] <== [tmp_13558[0] + tmp_13562[0], tmp_13558[1] + tmp_13562[1], tmp_13558[2] + tmp_13562[2]];
    signal tmp_13564[3] <== [tmp_13563[0] * 1, tmp_13563[1] * 1, tmp_13563[2] * 1];
    signal tmp_13565[3] <== [evals[76][0] - tmp_13564[0], evals[76][1] - tmp_13564[1], evals[76][2] - tmp_13564[2]];
    signal tmp_13566[3] <== CMul()(tmp_9163, tmp_13565);
    signal tmp_13567[3] <== [tmp_13487[0] + tmp_13566[0], tmp_13487[1] + tmp_13566[1], tmp_13487[2] + tmp_13566[2]];
    signal tmp_13568[3] <== CMul()(challengeQ, tmp_13567);
    signal tmp_13569[3] <== CMul()(evals[76], evals[76]);
    signal tmp_13570[3] <== CMul()(tmp_13569, tmp_13569);
    signal tmp_13571[3] <== CMul()(tmp_13570, tmp_13569);
    signal tmp_13572[3] <== CMul()(tmp_13571, evals[76]);
    tmp_13573 <== [tmp_13572[0] + 13644528514839338320, tmp_13572[1], tmp_13572[2]];
    signal tmp_13575[3] <== [tmp_13488[0] * 46774797835064597, tmp_13488[1] * 46774797835064597, tmp_13488[2] * 46774797835064597];
    signal tmp_13576[3] <== [tmp_13491[0] + tmp_13575[0], tmp_13491[1] + tmp_13575[1], tmp_13491[2] + tmp_13575[2]];
    signal tmp_13577[3] <== [16742843303846205239 * tmp_13576[0], 16742843303846205239 * tmp_13576[1], 16742843303846205239 * tmp_13576[2]];
    signal tmp_13578[3] <== [tmp_13573[0] + tmp_13577[0], tmp_13573[1] + tmp_13577[1], tmp_13573[2] + tmp_13577[2]];
    signal tmp_13580[3] <== [tmp_13488[0] * 42260813756980111, tmp_13488[1] * 42260813756980111, tmp_13488[2] * 42260813756980111];
    signal tmp_13581[3] <== [tmp_13496[0] + tmp_13580[0], tmp_13496[1] + tmp_13580[1], tmp_13496[2] + tmp_13580[2]];
    signal tmp_13582[3] <== [16027339315972474841 * tmp_13581[0], 16027339315972474841 * tmp_13581[1], 16027339315972474841 * tmp_13581[2]];
    signal tmp_13583[3] <== [tmp_13578[0] + tmp_13582[0], tmp_13578[1] + tmp_13582[1], tmp_13578[2] + tmp_13582[2]];
    signal tmp_13585[3] <== [tmp_13488[0] * 42061006075881976, tmp_13488[1] * 42061006075881976, tmp_13488[2] * 42061006075881976];
    signal tmp_13586[3] <== [tmp_13501[0] + tmp_13585[0], tmp_13501[1] + tmp_13585[1], tmp_13501[2] + tmp_13585[2]];
    signal tmp_13587[3] <== [5155237477011543063 * tmp_13586[0], 5155237477011543063 * tmp_13586[1], 5155237477011543063 * tmp_13586[2]];
    signal tmp_13588[3] <== [tmp_13583[0] + tmp_13587[0], tmp_13583[1] + tmp_13587[1], tmp_13583[2] + tmp_13587[2]];
    signal tmp_13590[3] <== [tmp_13488[0] * 37387884336802444, tmp_13488[1] * 37387884336802444, tmp_13488[2] * 37387884336802444];
    signal tmp_13591[3] <== [tmp_13506[0] + tmp_13590[0], tmp_13506[1] + tmp_13590[1], tmp_13506[2] + tmp_13590[2]];
    signal tmp_13592[3] <== [16706543232980973010 * tmp_13591[0], 16706543232980973010 * tmp_13591[1], 16706543232980973010 * tmp_13591[2]];
    signal tmp_13593[3] <== [tmp_13588[0] + tmp_13592[0], tmp_13588[1] + tmp_13592[1], tmp_13588[2] + tmp_13592[2]];
    signal tmp_13595[3] <== [tmp_13488[0] * 45874323791659250, tmp_13488[1] * 45874323791659250, tmp_13488[2] * 45874323791659250];
    signal tmp_13596[3] <== [tmp_13511[0] + tmp_13595[0], tmp_13511[1] + tmp_13595[1], tmp_13511[2] + tmp_13595[2]];
    signal tmp_13597[3] <== [7286314172953842238 * tmp_13596[0], 7286314172953842238 * tmp_13596[1], 7286314172953842238 * tmp_13596[2]];
    signal tmp_13598[3] <== [tmp_13593[0] + tmp_13597[0], tmp_13593[1] + tmp_13597[1], tmp_13593[2] + tmp_13597[2]];
    signal tmp_13600[3] <== [tmp_13488[0] * 39843739646961688, tmp_13488[1] * 39843739646961688, tmp_13488[2] * 39843739646961688];
    signal tmp_13601[3] <== [tmp_13516[0] + tmp_13600[0], tmp_13516[1] + tmp_13600[1], tmp_13516[2] + tmp_13600[2]];
    signal tmp_13602[3] <== [4083460614057513700 * tmp_13601[0], 4083460614057513700 * tmp_13601[1], 4083460614057513700 * tmp_13601[2]];
    signal tmp_13603[3] <== [tmp_13598[0] + tmp_13602[0], tmp_13598[1] + tmp_13602[1], tmp_13598[2] + tmp_13602[2]];
    signal tmp_13605[3] <== [tmp_13488[0] * 37505668333798549, tmp_13488[1] * 37505668333798549, tmp_13488[2] * 37505668333798549];
    signal tmp_13606[3] <== [tmp_13521[0] + tmp_13605[0], tmp_13521[1] + tmp_13605[1], tmp_13521[2] + tmp_13605[2]];
    signal tmp_13607[3] <== [12377749838518084086 * tmp_13606[0], 12377749838518084086 * tmp_13606[1], 12377749838518084086 * tmp_13606[2]];
    signal tmp_13608[3] <== [tmp_13603[0] + tmp_13607[0], tmp_13603[1] + tmp_13607[1], tmp_13603[2] + tmp_13607[2]];
    signal tmp_13610[3] <== [tmp_13488[0] * 33555193038270055, tmp_13488[1] * 33555193038270055, tmp_13488[2] * 33555193038270055];
    signal tmp_13611[3] <== [tmp_13526[0] + tmp_13610[0], tmp_13526[1] + tmp_13610[1], tmp_13526[2] + tmp_13610[2]];
    signal tmp_13612[3] <== [10835584132090636100 * tmp_13611[0], 10835584132090636100 * tmp_13611[1], 10835584132090636100 * tmp_13611[2]];
    signal tmp_13613[3] <== [tmp_13608[0] + tmp_13612[0], tmp_13608[1] + tmp_13612[1], tmp_13608[2] + tmp_13612[2]];
    signal tmp_13615[3] <== [tmp_13488[0] * 46401355029888991, tmp_13488[1] * 46401355029888991, tmp_13488[2] * 46401355029888991];
    signal tmp_13616[3] <== [tmp_13531[0] + tmp_13615[0], tmp_13531[1] + tmp_13615[1], tmp_13531[2] + tmp_13615[2]];
    signal tmp_13617[3] <== [9162173973000276298 * tmp_13616[0], 9162173973000276298 * tmp_13616[1], 9162173973000276298 * tmp_13616[2]];
    signal tmp_13618[3] <== [tmp_13613[0] + tmp_13617[0], tmp_13613[1] + tmp_13617[1], tmp_13613[2] + tmp_13617[2]];
    signal tmp_13620[3] <== [tmp_13488[0] * 43291015365600457, tmp_13488[1] * 43291015365600457, tmp_13488[2] * 43291015365600457];
    signal tmp_13621[3] <== [tmp_13536[0] + tmp_13620[0], tmp_13536[1] + tmp_13620[1], tmp_13536[2] + tmp_13620[2]];
    signal tmp_13622[3] <== [13242512893966320078 * tmp_13621[0], 13242512893966320078 * tmp_13621[1], 13242512893966320078 * tmp_13621[2]];
    signal tmp_13623[3] <== [tmp_13618[0] + tmp_13622[0], tmp_13618[1] + tmp_13622[1], tmp_13618[2] + tmp_13622[2]];
    signal tmp_13625[3] <== [tmp_13488[0] * 41779174606149599, tmp_13488[1] * 41779174606149599, tmp_13488[2] * 41779174606149599];
    signal tmp_13626[3] <== [tmp_13541[0] + tmp_13625[0], tmp_13541[1] + tmp_13625[1], tmp_13541[2] + tmp_13625[2]];
    signal tmp_13627[3] <== [16079292348832412129 * tmp_13626[0], 16079292348832412129 * tmp_13626[1], 16079292348832412129 * tmp_13626[2]];
    signal tmp_13628[3] <== [tmp_13623[0] + tmp_13627[0], tmp_13623[1] + tmp_13627[1], tmp_13623[2] + tmp_13627[2]];
    signal tmp_13630[3] <== [tmp_13488[0] * 40954303879913243, tmp_13488[1] * 40954303879913243, tmp_13488[2] * 40954303879913243];
    signal tmp_13631[3] <== [tmp_13546[0] + tmp_13630[0], tmp_13546[1] + tmp_13630[1], tmp_13546[2] + tmp_13630[2]];
    signal tmp_13632[3] <== [11725903253960910652 * tmp_13631[0], 11725903253960910652 * tmp_13631[1], 11725903253960910652 * tmp_13631[2]];
    signal tmp_13633[3] <== [tmp_13628[0] + tmp_13632[0], tmp_13628[1] + tmp_13632[1], tmp_13628[2] + tmp_13632[2]];
    signal tmp_13635[3] <== [tmp_13488[0] * 46379175729155491, tmp_13488[1] * 46379175729155491, tmp_13488[2] * 46379175729155491];
    signal tmp_13636[3] <== [tmp_13551[0] + tmp_13635[0], tmp_13551[1] + tmp_13635[1], tmp_13551[2] + tmp_13635[2]];
    signal tmp_13637[3] <== [2230603834375356574 * tmp_13636[0], 2230603834375356574 * tmp_13636[1], 2230603834375356574 * tmp_13636[2]];
    signal tmp_13638[3] <== [tmp_13633[0] + tmp_13637[0], tmp_13633[1] + tmp_13637[1], tmp_13633[2] + tmp_13637[2]];
    signal tmp_13640[3] <== [tmp_13488[0] * 38174104222724431, tmp_13488[1] * 38174104222724431, tmp_13488[2] * 38174104222724431];
    tmp_13641 <== [tmp_13556[0] + tmp_13640[0], tmp_13556[1] + tmp_13640[1], tmp_13556[2] + tmp_13640[2]];
    signal tmp_13642[3] <== [2370340271898460524 * tmp_13641[0], 2370340271898460524 * tmp_13641[1], 2370340271898460524 * tmp_13641[2]];
    signal tmp_13643[3] <== [tmp_13638[0] + tmp_13642[0], tmp_13638[1] + tmp_13642[1], tmp_13638[2] + tmp_13642[2]];
    signal tmp_13645[3] <== [tmp_13488[0] * 42896868640066877, tmp_13488[1] * 42896868640066877, tmp_13488[2] * 42896868640066877];
    tmp_13646 <== [tmp_13561[0] + tmp_13645[0], tmp_13561[1] + tmp_13645[1], tmp_13561[2] + tmp_13645[2]];
    signal tmp_13647[3] <== [15115197718548534897 * tmp_13646[0], 15115197718548534897 * tmp_13646[1], 15115197718548534897 * tmp_13646[2]];
    signal tmp_13648[3] <== [tmp_13643[0] + tmp_13647[0], tmp_13643[1] + tmp_13647[1], tmp_13643[2] + tmp_13647[2]];
    signal tmp_13649[3] <== [tmp_13648[0] * 1, tmp_13648[1] * 1, tmp_13648[2] * 1];
    signal tmp_13650[3] <== [evals[77][0] - tmp_13649[0], evals[77][1] - tmp_13649[1], evals[77][2] - tmp_13649[2]];
    signal tmp_13651[3] <== CMul()(tmp_9163, tmp_13650);
    signal tmp_13652[3] <== [tmp_13568[0] + tmp_13651[0], tmp_13568[1] + tmp_13651[1], tmp_13568[2] + tmp_13651[2]];
    tmp_13653 <== CMul()(challengeQ, tmp_13652);
    signal tmp_13654[3] <== CMul()(evals[77], evals[77]);
    signal tmp_13655[3] <== CMul()(tmp_13654, tmp_13654);
    signal tmp_13656[3] <== CMul()(tmp_13655, tmp_13654);
    signal tmp_13657[3] <== CMul()(tmp_13656, evals[77]);
    tmp_13658 <== [tmp_13657[0] + 13846825240309565987, tmp_13657[1], tmp_13657[2]];
    signal tmp_13660[3] <== [tmp_13573[0] * 136526419311991, tmp_13573[1] * 136526419311991, tmp_13573[2] * 136526419311991];
    tmp_13661 <== [tmp_13576[0] + tmp_13660[0], tmp_13576[1] + tmp_13660[1], tmp_13576[2] + tmp_13660[2]];
    signal tmp_13662[3] <== [13471986203177128393 * tmp_13661[0], 13471986203177128393 * tmp_13661[1], 13471986203177128393 * tmp_13661[2]];
    signal tmp_13663[3] <== [tmp_13658[0] + tmp_13662[0], tmp_13658[1] + tmp_13662[1], tmp_13658[2] + tmp_13662[2]];
    signal tmp_13665[3] <== [tmp_13573[0] * 118328038511571, tmp_13573[1] * 118328038511571, tmp_13573[2] * 118328038511571];
    tmp_13666 <== [tmp_13581[0] + tmp_13665[0], tmp_13581[1] + tmp_13665[1], tmp_13581[2] + tmp_13665[2]];
    signal tmp_13667[3] <== [5366496057517922993 * tmp_13666[0], 5366496057517922993 * tmp_13666[1], 5366496057517922993 * tmp_13666[2]];
    signal tmp_13668[3] <== [tmp_13663[0] + tmp_13667[0], tmp_13663[1] + tmp_13667[1], tmp_13663[2] + tmp_13667[2]];
    signal tmp_13670[3] <== [tmp_13573[0] * 119953340504678, tmp_13573[1] * 119953340504678, tmp_13573[2] * 119953340504678];
    tmp_13671 <== [tmp_13586[0] + tmp_13670[0], tmp_13586[1] + tmp_13670[1], tmp_13586[2] + tmp_13670[2]];
    signal tmp_13672[3] <== [1488645808239492428 * tmp_13671[0], 1488645808239492428 * tmp_13671[1], 1488645808239492428 * tmp_13671[2]];
    signal tmp_13673[3] <== [tmp_13668[0] + tmp_13672[0], tmp_13668[1] + tmp_13672[1], tmp_13668[2] + tmp_13672[2]];
    signal tmp_13675[3] <== [tmp_13573[0] * 108745180907795, tmp_13573[1] * 108745180907795, tmp_13573[2] * 108745180907795];
    tmp_13676 <== [tmp_13591[0] + tmp_13675[0], tmp_13591[1] + tmp_13675[1], tmp_13591[2] + tmp_13675[2]];
    signal tmp_13677[3] <== [14201553218769780296 * tmp_13676[0], 14201553218769780296 * tmp_13676[1], 14201553218769780296 * tmp_13676[2]];
    signal tmp_13678[3] <== [tmp_13673[0] + tmp_13677[0], tmp_13673[1] + tmp_13677[1], tmp_13673[2] + tmp_13677[2]];
    signal tmp_13680[3] <== [tmp_13573[0] * 130471040155374, tmp_13573[1] * 130471040155374, tmp_13573[2] * 130471040155374];
    tmp_13681 <== [tmp_13596[0] + tmp_13680[0], tmp_13596[1] + tmp_13680[1], tmp_13596[2] + tmp_13680[2]];
    signal tmp_13682[3] <== [9804649820399473624 * tmp_13681[0], 9804649820399473624 * tmp_13681[1], 9804649820399473624 * tmp_13681[2]];
    signal tmp_13683[3] <== [tmp_13678[0] + tmp_13682[0], tmp_13678[1] + tmp_13682[1], tmp_13678[2] + tmp_13682[2]];
    signal tmp_13685[3] <== [tmp_13573[0] * 120388179181929, tmp_13573[1] * 120388179181929, tmp_13573[2] * 120388179181929];
    tmp_13686 <== [tmp_13601[0] + tmp_13685[0], tmp_13601[1] + tmp_13685[1], tmp_13601[2] + tmp_13685[2]];
    signal tmp_13687[3] <== [8107836565913188653 * tmp_13686[0], 8107836565913188653 * tmp_13686[1], 8107836565913188653 * tmp_13686[2]];
    signal tmp_13688[3] <== [tmp_13683[0] + tmp_13687[0], tmp_13683[1] + tmp_13687[1], tmp_13683[2] + tmp_13687[2]];
    signal tmp_13690[3] <== [tmp_13573[0] * 111878367465032, tmp_13573[1] * 111878367465032, tmp_13573[2] * 111878367465032];
    tmp_13691 <== [tmp_13606[0] + tmp_13690[0], tmp_13606[1] + tmp_13690[1], tmp_13606[2] + tmp_13690[2]];
    signal tmp_13692[3] <== [17624536480452545801 * tmp_13691[0], 17624536480452545801 * tmp_13691[1], 17624536480452545801 * tmp_13691[2]];
    signal tmp_13693[3] <== [tmp_13688[0] + tmp_13692[0], tmp_13688[1] + tmp_13692[1], tmp_13688[2] + tmp_13692[2]];
    signal tmp_13695[3] <== [tmp_13573[0] * 93831285341340, tmp_13573[1] * 93831285341340, tmp_13573[2] * 93831285341340];
    tmp_13696 <== [tmp_13611[0] + tmp_13695[0], tmp_13611[1] + tmp_13695[1], tmp_13611[2] + tmp_13695[2]];
    signal tmp_13697[3] <== [12565506485641087099 * tmp_13696[0], 12565506485641087099 * tmp_13696[1], 12565506485641087099 * tmp_13696[2]];
    signal tmp_13698[3] <== [tmp_13693[0] + tmp_13697[0], tmp_13693[1] + tmp_13697[1], tmp_13693[2] + tmp_13697[2]];
    signal tmp_13700[3] <== [tmp_13573[0] * 132190456129009, tmp_13573[1] * 132190456129009, tmp_13573[2] * 132190456129009];
    tmp_13701 <== [tmp_13616[0] + tmp_13700[0], tmp_13616[1] + tmp_13700[1], tmp_13616[2] + tmp_13700[2]];
    signal tmp_13702[3] <== [8957615728977470678 * tmp_13701[0], 8957615728977470678 * tmp_13701[1], 8957615728977470678 * tmp_13701[2]];
    signal tmp_13703[3] <== [tmp_13698[0] + tmp_13702[0], tmp_13698[1] + tmp_13702[1], tmp_13698[2] + tmp_13702[2]];
    signal tmp_13705[3] <== [tmp_13573[0] * 129381316454358, tmp_13573[1] * 129381316454358, tmp_13573[2] * 129381316454358];
    tmp_13706 <== [tmp_13621[0] + tmp_13705[0], tmp_13621[1] + tmp_13705[1], tmp_13621[2] + tmp_13705[2]];
    signal tmp_13707[3] <== [14839456847523021051 * tmp_13706[0], 14839456847523021051 * tmp_13706[1], 14839456847523021051 * tmp_13706[2]];
    signal tmp_13708[3] <== [tmp_13703[0] + tmp_13707[0], tmp_13703[1] + tmp_13707[1], tmp_13703[2] + tmp_13707[2]];
    signal tmp_13710[3] <== [tmp_13573[0] * 119515447190824, tmp_13573[1] * 119515447190824, tmp_13573[2] * 119515447190824];
    tmp_13711 <== [tmp_13626[0] + tmp_13710[0], tmp_13626[1] + tmp_13710[1], tmp_13626[2] + tmp_13710[2]];
    signal tmp_13712[3] <== [2411490870786259255 * tmp_13711[0], 2411490870786259255 * tmp_13711[1], 2411490870786259255 * tmp_13711[2]];
    signal tmp_13713[3] <== [tmp_13708[0] + tmp_13712[0], tmp_13708[1] + tmp_13712[1], tmp_13708[2] + tmp_13712[2]];
    signal tmp_13715[3] <== [tmp_13573[0] * 118467363703132, tmp_13573[1] * 118467363703132, tmp_13573[2] * 118467363703132];
    tmp_13716 <== [tmp_13631[0] + tmp_13715[0], tmp_13631[1] + tmp_13715[1], tmp_13631[2] + tmp_13715[2]];
    signal tmp_13717[3] <== [5683566100576615076 * tmp_13716[0], 5683566100576615076 * tmp_13716[1], 5683566100576615076 * tmp_13716[2]];
    tmp_13718 <== [tmp_13713[0] + tmp_13717[0], tmp_13713[1] + tmp_13717[1], tmp_13713[2] + tmp_13717[2]];
    signal tmp_13720[3] <== [tmp_13573[0] * 135504422706996, tmp_13573[1] * 135504422706996, tmp_13573[2] * 135504422706996];
    tmp_13721 <== [tmp_13636[0] + tmp_13720[0], tmp_13636[1] + tmp_13720[1], tmp_13636[2] + tmp_13720[2]];
}

template VerifyEvaluationsChunks5_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_9163[3];
    signal input tmp_9180[3];
    signal input tmp_9192[3];
    signal input tmp_9210[3];
    signal input tmp_13573[3];
    signal input tmp_13641[3];
    signal input tmp_13646[3];
    signal input tmp_13653[3];
    signal input tmp_13658[3];
    signal input tmp_13661[3];
    signal input tmp_13666[3];
    signal input tmp_13671[3];
    signal input tmp_13676[3];
    signal input tmp_13681[3];
    signal input tmp_13686[3];
    signal input tmp_13691[3];
    signal input tmp_13696[3];
    signal input tmp_13701[3];
    signal input tmp_13706[3];
    signal input tmp_13711[3];
    signal input tmp_13716[3];
    signal input tmp_13718[3];
    signal input tmp_13721[3];

    signal output tmp_14899[3];
    signal output tmp_14905[3];
    signal output tmp_14906[3];
    signal tmp_13722[3] <== [12384678008916204197 * tmp_13721[0], 12384678008916204197 * tmp_13721[1], 12384678008916204197 * tmp_13721[2]];
    signal tmp_13723[3] <== [tmp_13718[0] + tmp_13722[0], tmp_13718[1] + tmp_13722[1], tmp_13718[2] + tmp_13722[2]];
    signal tmp_13725[3] <== [tmp_13573[0] * 106598700061332, tmp_13573[1] * 106598700061332, tmp_13573[2] * 106598700061332];
    signal tmp_13726[3] <== [tmp_13641[0] + tmp_13725[0], tmp_13641[1] + tmp_13725[1], tmp_13641[2] + tmp_13725[2]];
    signal tmp_13727[3] <== [13849032862488033002 * tmp_13726[0], 13849032862488033002 * tmp_13726[1], 13849032862488033002 * tmp_13726[2]];
    signal tmp_13728[3] <== [tmp_13723[0] + tmp_13727[0], tmp_13723[1] + tmp_13727[1], tmp_13723[2] + tmp_13727[2]];
    signal tmp_13730[3] <== [tmp_13573[0] * 118332012221556, tmp_13573[1] * 118332012221556, tmp_13573[2] * 118332012221556];
    signal tmp_13731[3] <== [tmp_13646[0] + tmp_13730[0], tmp_13646[1] + tmp_13730[1], tmp_13646[2] + tmp_13730[2]];
    signal tmp_13732[3] <== [3958502407134855424 * tmp_13731[0], 3958502407134855424 * tmp_13731[1], 3958502407134855424 * tmp_13731[2]];
    signal tmp_13733[3] <== [tmp_13728[0] + tmp_13732[0], tmp_13728[1] + tmp_13732[1], tmp_13728[2] + tmp_13732[2]];
    signal tmp_13734[3] <== [tmp_13733[0] * 1, tmp_13733[1] * 1, tmp_13733[2] * 1];
    signal tmp_13735[3] <== [evals[78][0] - tmp_13734[0], evals[78][1] - tmp_13734[1], evals[78][2] - tmp_13734[2]];
    signal tmp_13736[3] <== CMul()(tmp_9163, tmp_13735);
    signal tmp_13737[3] <== [tmp_13653[0] + tmp_13736[0], tmp_13653[1] + tmp_13736[1], tmp_13653[2] + tmp_13736[2]];
    signal tmp_13738[3] <== CMul()(challengeQ, tmp_13737);
    signal tmp_13739[3] <== CMul()(evals[78], evals[78]);
    signal tmp_13740[3] <== CMul()(tmp_13739, tmp_13739);
    signal tmp_13741[3] <== CMul()(tmp_13740, tmp_13739);
    signal tmp_13742[3] <== CMul()(tmp_13741, evals[78]);
    signal tmp_13743[3] <== [tmp_13742[0] + 4463709219992588910, tmp_13742[1], tmp_13742[2]];
    signal tmp_13745[3] <== [tmp_13658[0] * 369323044064, tmp_13658[1] * 369323044064, tmp_13658[2] * 369323044064];
    signal tmp_13746[3] <== [tmp_13661[0] + tmp_13745[0], tmp_13661[1] + tmp_13745[1], tmp_13661[2] + tmp_13745[2]];
    signal tmp_13747[3] <== [13905431266086421911 * tmp_13746[0], 13905431266086421911 * tmp_13746[1], 13905431266086421911 * tmp_13746[2]];
    signal tmp_13748[3] <== [tmp_13743[0] + tmp_13747[0], tmp_13743[1] + tmp_13747[1], tmp_13743[2] + tmp_13747[2]];
    signal tmp_13750[3] <== [tmp_13658[0] * 376274279152, tmp_13658[1] * 376274279152, tmp_13658[2] * 376274279152];
    signal tmp_13751[3] <== [tmp_13666[0] + tmp_13750[0], tmp_13666[1] + tmp_13750[1], tmp_13666[2] + tmp_13750[2]];
    signal tmp_13752[3] <== [305991708664976914 * tmp_13751[0], 305991708664976914 * tmp_13751[1], 305991708664976914 * tmp_13751[2]];
    signal tmp_13753[3] <== [tmp_13748[0] + tmp_13752[0], tmp_13748[1] + tmp_13752[1], tmp_13748[2] + tmp_13752[2]];
    signal tmp_13755[3] <== [tmp_13658[0] * 361996018532, tmp_13658[1] * 361996018532, tmp_13658[2] * 361996018532];
    signal tmp_13756[3] <== [tmp_13671[0] + tmp_13755[0], tmp_13671[1] + tmp_13755[1], tmp_13671[2] + tmp_13755[2]];
    signal tmp_13757[3] <== [15209999190551796912 * tmp_13756[0], 15209999190551796912 * tmp_13756[1], 15209999190551796912 * tmp_13756[2]];
    signal tmp_13758[3] <== [tmp_13753[0] + tmp_13757[0], tmp_13753[1] + tmp_13757[1], tmp_13753[2] + tmp_13757[2]];
    signal tmp_13760[3] <== [tmp_13658[0] * 301447583530, tmp_13658[1] * 301447583530, tmp_13658[2] * 301447583530];
    signal tmp_13761[3] <== [tmp_13676[0] + tmp_13760[0], tmp_13676[1] + tmp_13760[1], tmp_13676[2] + tmp_13760[2]];
    signal tmp_13762[3] <== [11028393659978236139 * tmp_13761[0], 11028393659978236139 * tmp_13761[1], 11028393659978236139 * tmp_13761[2]];
    signal tmp_13763[3] <== [tmp_13758[0] + tmp_13762[0], tmp_13758[1] + tmp_13762[1], tmp_13758[2] + tmp_13762[2]];
    signal tmp_13765[3] <== [tmp_13658[0] * 377601912356, tmp_13658[1] * 377601912356, tmp_13658[2] * 377601912356];
    signal tmp_13766[3] <== [tmp_13681[0] + tmp_13765[0], tmp_13681[1] + tmp_13765[1], tmp_13681[2] + tmp_13765[2]];
    signal tmp_13767[3] <== [17904306168706867216 * tmp_13766[0], 17904306168706867216 * tmp_13766[1], 17904306168706867216 * tmp_13766[2]];
    signal tmp_13768[3] <== [tmp_13763[0] + tmp_13767[0], tmp_13763[1] + tmp_13767[1], tmp_13763[2] + tmp_13767[2]];
    signal tmp_13770[3] <== [tmp_13658[0] * 345235981905, tmp_13658[1] * 345235981905, tmp_13658[2] * 345235981905];
    signal tmp_13771[3] <== [tmp_13686[0] + tmp_13770[0], tmp_13686[1] + tmp_13770[1], tmp_13686[2] + tmp_13770[2]];
    signal tmp_13772[3] <== [8374188644407616260 * tmp_13771[0], 8374188644407616260 * tmp_13771[1], 8374188644407616260 * tmp_13771[2]];
    signal tmp_13773[3] <== [tmp_13768[0] + tmp_13772[0], tmp_13768[1] + tmp_13772[1], tmp_13768[2] + tmp_13772[2]];
    signal tmp_13775[3] <== [tmp_13658[0] * 287662427631, tmp_13658[1] * 287662427631, tmp_13658[2] * 287662427631];
    signal tmp_13776[3] <== [tmp_13691[0] + tmp_13775[0], tmp_13691[1] + tmp_13775[1], tmp_13691[2] + tmp_13775[2]];
    signal tmp_13777[3] <== [680800612778784853 * tmp_13776[0], 680800612778784853 * tmp_13776[1], 680800612778784853 * tmp_13776[2]];
    signal tmp_13778[3] <== [tmp_13773[0] + tmp_13777[0], tmp_13773[1] + tmp_13777[1], tmp_13773[2] + tmp_13777[2]];
    signal tmp_13780[3] <== [tmp_13658[0] * 284662670472, tmp_13658[1] * 284662670472, tmp_13658[2] * 284662670472];
    signal tmp_13781[3] <== [tmp_13696[0] + tmp_13780[0], tmp_13696[1] + tmp_13780[1], tmp_13696[2] + tmp_13780[2]];
    signal tmp_13782[3] <== [5417240084914372136 * tmp_13781[0], 5417240084914372136 * tmp_13781[1], 5417240084914372136 * tmp_13781[2]];
    signal tmp_13783[3] <== [tmp_13778[0] + tmp_13782[0], tmp_13778[1] + tmp_13782[1], tmp_13778[2] + tmp_13782[2]];
    signal tmp_13785[3] <== [tmp_13658[0] * 404765160652, tmp_13658[1] * 404765160652, tmp_13658[2] * 404765160652];
    signal tmp_13786[3] <== [tmp_13701[0] + tmp_13785[0], tmp_13701[1] + tmp_13785[1], tmp_13701[2] + tmp_13785[2]];
    signal tmp_13787[3] <== [1393710566264274938 * tmp_13786[0], 1393710566264274938 * tmp_13786[1], 1393710566264274938 * tmp_13786[2]];
    signal tmp_13788[3] <== [tmp_13783[0] + tmp_13787[0], tmp_13783[1] + tmp_13787[1], tmp_13783[2] + tmp_13787[2]];
    signal tmp_13790[3] <== [tmp_13658[0] * 351806900899, tmp_13658[1] * 351806900899, tmp_13658[2] * 351806900899];
    signal tmp_13791[3] <== [tmp_13706[0] + tmp_13790[0], tmp_13706[1] + tmp_13790[1], tmp_13706[2] + tmp_13790[2]];
    signal tmp_13792[3] <== [14148046894039047274 * tmp_13791[0], 14148046894039047274 * tmp_13791[1], 14148046894039047274 * tmp_13791[2]];
    signal tmp_13793[3] <== [tmp_13788[0] + tmp_13792[0], tmp_13788[1] + tmp_13792[1], tmp_13788[2] + tmp_13792[2]];
    signal tmp_13795[3] <== [tmp_13658[0] * 315580522525, tmp_13658[1] * 315580522525, tmp_13658[2] * 315580522525];
    signal tmp_13796[3] <== [tmp_13711[0] + tmp_13795[0], tmp_13711[1] + tmp_13795[1], tmp_13711[2] + tmp_13795[2]];
    signal tmp_13797[3] <== [3244735513351203558 * tmp_13796[0], 3244735513351203558 * tmp_13796[1], 3244735513351203558 * tmp_13796[2]];
    signal tmp_13798[3] <== [tmp_13793[0] + tmp_13797[0], tmp_13793[1] + tmp_13797[1], tmp_13793[2] + tmp_13797[2]];
    signal tmp_13800[3] <== [tmp_13658[0] * 348317304582, tmp_13658[1] * 348317304582, tmp_13658[2] * 348317304582];
    signal tmp_13801[3] <== [tmp_13716[0] + tmp_13800[0], tmp_13716[1] + tmp_13800[1], tmp_13716[2] + tmp_13800[2]];
    signal tmp_13802[3] <== [4744494926695798520 * tmp_13801[0], 4744494926695798520 * tmp_13801[1], 4744494926695798520 * tmp_13801[2]];
    signal tmp_13803[3] <== [tmp_13798[0] + tmp_13802[0], tmp_13798[1] + tmp_13802[1], tmp_13798[2] + tmp_13802[2]];
    signal tmp_13805[3] <== [tmp_13658[0] * 385400627347, tmp_13658[1] * 385400627347, tmp_13658[2] * 385400627347];
    signal tmp_13806[3] <== [tmp_13721[0] + tmp_13805[0], tmp_13721[1] + tmp_13805[1], tmp_13721[2] + tmp_13805[2]];
    signal tmp_13807[3] <== [12411244701441950267 * tmp_13806[0], 12411244701441950267 * tmp_13806[1], 12411244701441950267 * tmp_13806[2]];
    signal tmp_13808[3] <== [tmp_13803[0] + tmp_13807[0], tmp_13803[1] + tmp_13807[1], tmp_13803[2] + tmp_13807[2]];
    signal tmp_13810[3] <== [tmp_13658[0] * 323348446961, tmp_13658[1] * 323348446961, tmp_13658[2] * 323348446961];
    signal tmp_13811[3] <== [tmp_13726[0] + tmp_13810[0], tmp_13726[1] + tmp_13810[1], tmp_13726[2] + tmp_13810[2]];
    signal tmp_13812[3] <== [8803037078966971424 * tmp_13811[0], 8803037078966971424 * tmp_13811[1], 8803037078966971424 * tmp_13811[2]];
    signal tmp_13813[3] <== [tmp_13808[0] + tmp_13812[0], tmp_13808[1] + tmp_13812[1], tmp_13808[2] + tmp_13812[2]];
    signal tmp_13815[3] <== [tmp_13658[0] * 363383916243, tmp_13658[1] * 363383916243, tmp_13658[2] * 363383916243];
    signal tmp_13816[3] <== [tmp_13731[0] + tmp_13815[0], tmp_13731[1] + tmp_13815[1], tmp_13731[2] + tmp_13815[2]];
    signal tmp_13817[3] <== [5194306366755844852 * tmp_13816[0], 5194306366755844852 * tmp_13816[1], 5194306366755844852 * tmp_13816[2]];
    signal tmp_13818[3] <== [tmp_13813[0] + tmp_13817[0], tmp_13813[1] + tmp_13817[1], tmp_13813[2] + tmp_13817[2]];
    signal tmp_13819[3] <== [tmp_13818[0] * 1, tmp_13818[1] * 1, tmp_13818[2] * 1];
    signal tmp_13820[3] <== [evals[79][0] - tmp_13819[0], evals[79][1] - tmp_13819[1], evals[79][2] - tmp_13819[2]];
    signal tmp_13821[3] <== CMul()(tmp_9163, tmp_13820);
    signal tmp_13822[3] <== [tmp_13738[0] + tmp_13821[0], tmp_13738[1] + tmp_13821[1], tmp_13738[2] + tmp_13821[2]];
    signal tmp_13823[3] <== CMul()(challengeQ, tmp_13822);
    signal tmp_13824[3] <== CMul()(evals[79], evals[79]);
    signal tmp_13825[3] <== CMul()(tmp_13824, tmp_13824);
    signal tmp_13826[3] <== CMul()(tmp_13825, tmp_13824);
    signal tmp_13827[3] <== CMul()(tmp_13826, evals[79]);
    signal tmp_13828[3] <== [tmp_13827[0] + 15514028834701829198, tmp_13827[1], tmp_13827[2]];
    signal tmp_13830[3] <== [tmp_13743[0] * 1188745580, tmp_13743[1] * 1188745580, tmp_13743[2] * 1188745580];
    signal tmp_13831[3] <== [tmp_13746[0] + tmp_13830[0], tmp_13746[1] + tmp_13830[1], tmp_13746[2] + tmp_13830[2]];
    signal tmp_13832[3] <== [530243823762044158 * tmp_13831[0], 530243823762044158 * tmp_13831[1], 530243823762044158 * tmp_13831[2]];
    signal tmp_13833[3] <== [tmp_13828[0] + tmp_13832[0], tmp_13828[1] + tmp_13832[1], tmp_13828[2] + tmp_13832[2]];
    signal tmp_13835[3] <== [tmp_13743[0] * 1055003602, tmp_13743[1] * 1055003602, tmp_13743[2] * 1055003602];
    signal tmp_13836[3] <== [tmp_13751[0] + tmp_13835[0], tmp_13751[1] + tmp_13835[1], tmp_13751[2] + tmp_13835[2]];
    signal tmp_13837[3] <== [2698957419264379100 * tmp_13836[0], 2698957419264379100 * tmp_13836[1], 2698957419264379100 * tmp_13836[2]];
    signal tmp_13838[3] <== [tmp_13833[0] + tmp_13837[0], tmp_13833[1] + tmp_13837[1], tmp_13833[2] + tmp_13837[2]];
    signal tmp_13840[3] <== [tmp_13743[0] * 785416201, tmp_13743[1] * 785416201, tmp_13743[2] * 785416201];
    signal tmp_13841[3] <== [tmp_13756[0] + tmp_13840[0], tmp_13756[1] + tmp_13840[1], tmp_13756[2] + tmp_13840[2]];
    signal tmp_13842[3] <== [14296668284422018113 * tmp_13841[0], 14296668284422018113 * tmp_13841[1], 14296668284422018113 * tmp_13841[2]];
    signal tmp_13843[3] <== [tmp_13838[0] + tmp_13842[0], tmp_13838[1] + tmp_13842[1], tmp_13838[2] + tmp_13842[2]];
    signal tmp_13845[3] <== [tmp_13743[0] * 868051025, tmp_13743[1] * 868051025, tmp_13743[2] * 868051025];
    signal tmp_13846[3] <== [tmp_13761[0] + tmp_13845[0], tmp_13761[1] + tmp_13845[1], tmp_13761[2] + tmp_13845[2]];
    signal tmp_13847[3] <== [8333713848018243814 * tmp_13846[0], 8333713848018243814 * tmp_13846[1], 8333713848018243814 * tmp_13846[2]];
    signal tmp_13848[3] <== [tmp_13843[0] + tmp_13847[0], tmp_13843[1] + tmp_13847[1], tmp_13843[2] + tmp_13847[2]];
    signal tmp_13850[3] <== [tmp_13743[0] * 1135832507, tmp_13743[1] * 1135832507, tmp_13743[2] * 1135832507];
    signal tmp_13851[3] <== [tmp_13766[0] + tmp_13850[0], tmp_13766[1] + tmp_13850[1], tmp_13766[2] + tmp_13850[2]];
    signal tmp_13852[3] <== [12238342728063130848 * tmp_13851[0], 12238342728063130848 * tmp_13851[1], 12238342728063130848 * tmp_13851[2]];
    signal tmp_13853[3] <== [tmp_13848[0] + tmp_13852[0], tmp_13848[1] + tmp_13852[1], tmp_13848[2] + tmp_13852[2]];
    signal tmp_13855[3] <== [tmp_13743[0] * 1004853599, tmp_13743[1] * 1004853599, tmp_13743[2] * 1004853599];
    signal tmp_13856[3] <== [tmp_13771[0] + tmp_13855[0], tmp_13771[1] + tmp_13855[1], tmp_13771[2] + tmp_13855[2]];
    signal tmp_13857[3] <== [14599889067722749515 * tmp_13856[0], 14599889067722749515 * tmp_13856[1], 14599889067722749515 * tmp_13856[2]];
    signal tmp_13858[3] <== [tmp_13853[0] + tmp_13857[0], tmp_13853[1] + tmp_13857[1], tmp_13853[2] + tmp_13857[2]];
    signal tmp_13860[3] <== [tmp_13743[0] * 904741729, tmp_13743[1] * 904741729, tmp_13743[2] * 904741729];
    signal tmp_13861[3] <== [tmp_13776[0] + tmp_13860[0], tmp_13776[1] + tmp_13860[1], tmp_13776[2] + tmp_13860[2]];
    signal tmp_13862[3] <== [1498123092862756804 * tmp_13861[0], 1498123092862756804 * tmp_13861[1], 1498123092862756804 * tmp_13861[2]];
    signal tmp_13863[3] <== [tmp_13858[0] + tmp_13862[0], tmp_13858[1] + tmp_13862[1], tmp_13858[2] + tmp_13862[2]];
    signal tmp_13865[3] <== [tmp_13743[0] * 809824679, tmp_13743[1] * 809824679, tmp_13743[2] * 809824679];
    signal tmp_13866[3] <== [tmp_13781[0] + tmp_13865[0], tmp_13781[1] + tmp_13865[1], tmp_13781[2] + tmp_13865[2]];
    signal tmp_13867[3] <== [6951489160695905236 * tmp_13866[0], 6951489160695905236 * tmp_13866[1], 6951489160695905236 * tmp_13866[2]];
    signal tmp_13868[3] <== [tmp_13863[0] + tmp_13867[0], tmp_13863[1] + tmp_13867[1], tmp_13863[2] + tmp_13867[2]];
    signal tmp_13870[3] <== [tmp_13743[0] * 980810992, tmp_13743[1] * 980810992, tmp_13743[2] * 980810992];
    signal tmp_13871[3] <== [tmp_13786[0] + tmp_13870[0], tmp_13786[1] + tmp_13870[1], tmp_13786[2] + tmp_13870[2]];
    signal tmp_13872[3] <== [15663438810331591677 * tmp_13871[0], 15663438810331591677 * tmp_13871[1], 15663438810331591677 * tmp_13871[2]];
    signal tmp_13873[3] <== [tmp_13868[0] + tmp_13872[0], tmp_13868[1] + tmp_13872[1], tmp_13868[2] + tmp_13872[2]];
    signal tmp_13875[3] <== [tmp_13743[0] * 1178194302, tmp_13743[1] * 1178194302, tmp_13743[2] * 1178194302];
    signal tmp_13876[3] <== [tmp_13791[0] + tmp_13875[0], tmp_13791[1] + tmp_13875[1], tmp_13791[2] + tmp_13875[2]];
    signal tmp_13877[3] <== [13759896603824577231 * tmp_13876[0], 13759896603824577231 * tmp_13876[1], 13759896603824577231 * tmp_13876[2]];
    signal tmp_13878[3] <== [tmp_13873[0] + tmp_13877[0], tmp_13873[1] + tmp_13877[1], tmp_13873[2] + tmp_13877[2]];
    signal tmp_13880[3] <== [tmp_13743[0] * 1159788697, tmp_13743[1] * 1159788697, tmp_13743[2] * 1159788697];
    signal tmp_13881[3] <== [tmp_13796[0] + tmp_13880[0], tmp_13796[1] + tmp_13880[1], tmp_13796[2] + tmp_13880[2]];
    signal tmp_13882[3] <== [4898543744152164832 * tmp_13881[0], 4898543744152164832 * tmp_13881[1], 4898543744152164832 * tmp_13881[2]];
    signal tmp_13883[3] <== [tmp_13878[0] + tmp_13882[0], tmp_13878[1] + tmp_13882[1], tmp_13878[2] + tmp_13882[2]];
    signal tmp_13885[3] <== [tmp_13743[0] * 949043013, tmp_13743[1] * 949043013, tmp_13743[2] * 949043013];
    signal tmp_13886[3] <== [tmp_13801[0] + tmp_13885[0], tmp_13801[1] + tmp_13885[1], tmp_13801[2] + tmp_13885[2]];
    signal tmp_13887[3] <== [15908946426636755274 * tmp_13886[0], 15908946426636755274 * tmp_13886[1], 15908946426636755274 * tmp_13886[2]];
    signal tmp_13888[3] <== [tmp_13883[0] + tmp_13887[0], tmp_13883[1] + tmp_13887[1], tmp_13883[2] + tmp_13887[2]];
    signal tmp_13890[3] <== [tmp_13743[0] * 1001466621, tmp_13743[1] * 1001466621, tmp_13743[2] * 1001466621];
    signal tmp_13891[3] <== [tmp_13806[0] + tmp_13890[0], tmp_13806[1] + tmp_13890[1], tmp_13806[2] + tmp_13890[2]];
    signal tmp_13892[3] <== [13145973700442701961 * tmp_13891[0], 13145973700442701961 * tmp_13891[1], 13145973700442701961 * tmp_13891[2]];
    signal tmp_13893[3] <== [tmp_13888[0] + tmp_13892[0], tmp_13888[1] + tmp_13892[1], tmp_13888[2] + tmp_13892[2]];
    signal tmp_13895[3] <== [tmp_13743[0] * 1011628637, tmp_13743[1] * 1011628637, tmp_13743[2] * 1011628637];
    signal tmp_13896[3] <== [tmp_13811[0] + tmp_13895[0], tmp_13811[1] + tmp_13895[1], tmp_13811[2] + tmp_13895[2]];
    signal tmp_13897[3] <== [17400180121643364107 * tmp_13896[0], 17400180121643364107 * tmp_13896[1], 17400180121643364107 * tmp_13896[2]];
    signal tmp_13898[3] <== [tmp_13893[0] + tmp_13897[0], tmp_13893[1] + tmp_13897[1], tmp_13893[2] + tmp_13897[2]];
    signal tmp_13900[3] <== [tmp_13743[0] * 924759953, tmp_13743[1] * 924759953, tmp_13743[2] * 924759953];
    signal tmp_13901[3] <== [tmp_13816[0] + tmp_13900[0], tmp_13816[1] + tmp_13900[1], tmp_13816[2] + tmp_13900[2]];
    signal tmp_13902[3] <== [4391335980556159683 * tmp_13901[0], 4391335980556159683 * tmp_13901[1], 4391335980556159683 * tmp_13901[2]];
    signal tmp_13903[3] <== [tmp_13898[0] + tmp_13902[0], tmp_13898[1] + tmp_13902[1], tmp_13898[2] + tmp_13902[2]];
    signal tmp_13904[3] <== [tmp_13903[0] * 1, tmp_13903[1] * 1, tmp_13903[2] * 1];
    signal tmp_13905[3] <== [evals[80][0] - tmp_13904[0], evals[80][1] - tmp_13904[1], evals[80][2] - tmp_13904[2]];
    signal tmp_13906[3] <== CMul()(tmp_9163, tmp_13905);
    signal tmp_13907[3] <== [tmp_13823[0] + tmp_13906[0], tmp_13823[1] + tmp_13906[1], tmp_13823[2] + tmp_13906[2]];
    signal tmp_13908[3] <== CMul()(challengeQ, tmp_13907);
    signal tmp_13909[3] <== CMul()(evals[80], evals[80]);
    signal tmp_13910[3] <== CMul()(tmp_13909, tmp_13909);
    signal tmp_13911[3] <== CMul()(tmp_13910, tmp_13909);
    signal tmp_13912[3] <== CMul()(tmp_13911, evals[80]);
    signal tmp_13913[3] <== [tmp_13912[0] + 1027211584317448035, tmp_13912[1], tmp_13912[2]];
    signal tmp_13915[3] <== [tmp_13828[0] * 2475856, tmp_13828[1] * 2475856, tmp_13828[2] * 2475856];
    signal tmp_13916[3] <== [tmp_13831[0] + tmp_13915[0], tmp_13831[1] + tmp_13915[1], tmp_13831[2] + tmp_13915[2]];
    signal tmp_13917[3] <== [1787101036502119356 * tmp_13916[0], 1787101036502119356 * tmp_13916[1], 1787101036502119356 * tmp_13916[2]];
    signal tmp_13918[3] <== [tmp_13913[0] + tmp_13917[0], tmp_13913[1] + tmp_13917[1], tmp_13913[2] + tmp_13917[2]];
    signal tmp_13920[3] <== [tmp_13828[0] * 3337618, tmp_13828[1] * 3337618, tmp_13828[2] * 3337618];
    signal tmp_13921[3] <== [tmp_13836[0] + tmp_13920[0], tmp_13836[1] + tmp_13920[1], tmp_13836[2] + tmp_13920[2]];
    signal tmp_13922[3] <== [6714829819836869612 * tmp_13921[0], 6714829819836869612 * tmp_13921[1], 6714829819836869612 * tmp_13921[2]];
    signal tmp_13923[3] <== [tmp_13918[0] + tmp_13922[0], tmp_13918[1] + tmp_13922[1], tmp_13918[2] + tmp_13922[2]];
    signal tmp_13925[3] <== [tmp_13828[0] * 4161263, tmp_13828[1] * 4161263, tmp_13828[2] * 4161263];
    signal tmp_13926[3] <== [tmp_13841[0] + tmp_13925[0], tmp_13841[1] + tmp_13925[1], tmp_13841[2] + tmp_13925[2]];
    signal tmp_13927[3] <== [8614678868105523772 * tmp_13926[0], 8614678868105523772 * tmp_13926[1], 8614678868105523772 * tmp_13926[2]];
    signal tmp_13928[3] <== [tmp_13923[0] + tmp_13927[0], tmp_13923[1] + tmp_13927[1], tmp_13923[2] + tmp_13927[2]];
    signal tmp_13930[3] <== [tmp_13828[0] * 3129126, tmp_13828[1] * 3129126, tmp_13828[2] * 3129126];
    signal tmp_13931[3] <== [tmp_13846[0] + tmp_13930[0], tmp_13846[1] + tmp_13930[1], tmp_13846[2] + tmp_13930[2]];
    signal tmp_13932[3] <== [7372503606097811728 * tmp_13931[0], 7372503606097811728 * tmp_13931[1], 7372503606097811728 * tmp_13931[2]];
    signal tmp_13933[3] <== [tmp_13928[0] + tmp_13932[0], tmp_13928[1] + tmp_13932[1], tmp_13928[2] + tmp_13932[2]];
    signal tmp_13935[3] <== [tmp_13828[0] * 2071505, tmp_13828[1] * 2071505, tmp_13828[2] * 2071505];
    signal tmp_13936[3] <== [tmp_13851[0] + tmp_13935[0], tmp_13851[1] + tmp_13935[1], tmp_13851[2] + tmp_13935[2]];
    signal tmp_13937[3] <== [16967913077348192867 * tmp_13936[0], 16967913077348192867 * tmp_13936[1], 16967913077348192867 * tmp_13936[2]];
    signal tmp_13938[3] <== [tmp_13933[0] + tmp_13937[0], tmp_13933[1] + tmp_13937[1], tmp_13933[2] + tmp_13937[2]];
    signal tmp_13940[3] <== [tmp_13828[0] * 3373463, tmp_13828[1] * 3373463, tmp_13828[2] * 3373463];
    signal tmp_13941[3] <== [tmp_13856[0] + tmp_13940[0], tmp_13856[1] + tmp_13940[1], tmp_13856[2] + tmp_13940[2]];
    signal tmp_13942[3] <== [746855177740798579 * tmp_13941[0], 746855177740798579 * tmp_13941[1], 746855177740798579 * tmp_13941[2]];
    signal tmp_13943[3] <== [tmp_13938[0] + tmp_13942[0], tmp_13938[1] + tmp_13942[1], tmp_13938[2] + tmp_13942[2]];
    signal tmp_13945[3] <== [tmp_13828[0] * 2975691, tmp_13828[1] * 2975691, tmp_13828[2] * 2975691];
    signal tmp_13946[3] <== [tmp_13861[0] + tmp_13945[0], tmp_13861[1] + tmp_13945[1], tmp_13861[2] + tmp_13945[2]];
    signal tmp_13947[3] <== [8948543211894560314 * tmp_13946[0], 8948543211894560314 * tmp_13946[1], 8948543211894560314 * tmp_13946[2]];
    signal tmp_13948[3] <== [tmp_13943[0] + tmp_13947[0], tmp_13943[1] + tmp_13947[1], tmp_13943[2] + tmp_13947[2]];
    signal tmp_13950[3] <== [tmp_13828[0] * 1742470, tmp_13828[1] * 1742470, tmp_13828[2] * 1742470];
    signal tmp_13951[3] <== [tmp_13866[0] + tmp_13950[0], tmp_13866[1] + tmp_13950[1], tmp_13866[2] + tmp_13950[2]];
    signal tmp_13952[3] <== [7250314975625247352 * tmp_13951[0], 7250314975625247352 * tmp_13951[1], 7250314975625247352 * tmp_13951[2]];
    signal tmp_13953[3] <== [tmp_13948[0] + tmp_13952[0], tmp_13948[1] + tmp_13952[1], tmp_13948[2] + tmp_13952[2]];
    signal tmp_13955[3] <== [tmp_13828[0] * 2828204, tmp_13828[1] * 2828204, tmp_13828[2] * 2828204];
    signal tmp_13956[3] <== [tmp_13871[0] + tmp_13955[0], tmp_13871[1] + tmp_13955[1], tmp_13871[2] + tmp_13955[2]];
    signal tmp_13957[3] <== [11154498990878763202 * tmp_13956[0], 11154498990878763202 * tmp_13956[1], 11154498990878763202 * tmp_13956[2]];
    signal tmp_13958[3] <== [tmp_13953[0] + tmp_13957[0], tmp_13953[1] + tmp_13957[1], tmp_13953[2] + tmp_13957[2]];
    signal tmp_13960[3] <== [tmp_13828[0] * 3695590, tmp_13828[1] * 3695590, tmp_13828[2] * 3695590];
    signal tmp_13961[3] <== [tmp_13876[0] + tmp_13960[0], tmp_13876[1] + tmp_13960[1], tmp_13876[2] + tmp_13960[2]];
    signal tmp_13962[3] <== [9368653935206961209 * tmp_13961[0], 9368653935206961209 * tmp_13961[1], 9368653935206961209 * tmp_13961[2]];
    signal tmp_13963[3] <== [tmp_13958[0] + tmp_13962[0], tmp_13958[1] + tmp_13962[1], tmp_13958[2] + tmp_13962[2]];
    signal tmp_13965[3] <== [tmp_13828[0] * 1809935, tmp_13828[1] * 1809935, tmp_13828[2] * 1809935];
    signal tmp_13966[3] <== [tmp_13881[0] + tmp_13965[0], tmp_13881[1] + tmp_13965[1], tmp_13881[2] + tmp_13965[2]];
    signal tmp_13967[3] <== [5524036893624759211 * tmp_13966[0], 5524036893624759211 * tmp_13966[1], 5524036893624759211 * tmp_13966[2]];
    signal tmp_13968[3] <== [tmp_13963[0] + tmp_13967[0], tmp_13963[1] + tmp_13967[1], tmp_13963[2] + tmp_13967[2]];
    signal tmp_13970[3] <== [tmp_13828[0] * 2316312, tmp_13828[1] * 2316312, tmp_13828[2] * 2316312];
    signal tmp_13971[3] <== [tmp_13886[0] + tmp_13970[0], tmp_13886[1] + tmp_13970[1], tmp_13886[2] + tmp_13970[2]];
    signal tmp_13972[3] <== [2521926361859468511 * tmp_13971[0], 2521926361859468511 * tmp_13971[1], 2521926361859468511 * tmp_13971[2]];
    signal tmp_13973[3] <== [tmp_13968[0] + tmp_13972[0], tmp_13968[1] + tmp_13972[1], tmp_13968[2] + tmp_13972[2]];
    signal tmp_13975[3] <== [tmp_13828[0] * 3448583, tmp_13828[1] * 3448583, tmp_13828[2] * 3448583];
    signal tmp_13976[3] <== [tmp_13891[0] + tmp_13975[0], tmp_13891[1] + tmp_13975[1], tmp_13891[2] + tmp_13975[2]];
    signal tmp_13977[3] <== [14797218746248438904 * tmp_13976[0], 14797218746248438904 * tmp_13976[1], 14797218746248438904 * tmp_13976[2]];
    signal tmp_13978[3] <== [tmp_13973[0] + tmp_13977[0], tmp_13973[1] + tmp_13977[1], tmp_13973[2] + tmp_13977[2]];
    signal tmp_13980[3] <== [tmp_13828[0] * 2986173, tmp_13828[1] * 2986173, tmp_13828[2] * 2986173];
    signal tmp_13981[3] <== [tmp_13896[0] + tmp_13980[0], tmp_13896[1] + tmp_13980[1], tmp_13896[2] + tmp_13980[2]];
    signal tmp_13982[3] <== [10896085731851505890 * tmp_13981[0], 10896085731851505890 * tmp_13981[1], 10896085731851505890 * tmp_13981[2]];
    signal tmp_13983[3] <== [tmp_13978[0] + tmp_13982[0], tmp_13978[1] + tmp_13982[1], tmp_13978[2] + tmp_13982[2]];
    signal tmp_13985[3] <== [tmp_13828[0] * 2518923, tmp_13828[1] * 2518923, tmp_13828[2] * 2518923];
    signal tmp_13986[3] <== [tmp_13901[0] + tmp_13985[0], tmp_13901[1] + tmp_13985[1], tmp_13901[2] + tmp_13985[2]];
    signal tmp_13987[3] <== [9304799240492850786 * tmp_13986[0], 9304799240492850786 * tmp_13986[1], 9304799240492850786 * tmp_13986[2]];
    signal tmp_13988[3] <== [tmp_13983[0] + tmp_13987[0], tmp_13983[1] + tmp_13987[1], tmp_13983[2] + tmp_13987[2]];
    signal tmp_13989[3] <== [tmp_13988[0] * 1, tmp_13988[1] * 1, tmp_13988[2] * 1];
    signal tmp_13990[3] <== [evals[81][0] - tmp_13989[0], evals[81][1] - tmp_13989[1], evals[81][2] - tmp_13989[2]];
    signal tmp_13991[3] <== CMul()(tmp_9163, tmp_13990);
    signal tmp_13992[3] <== [tmp_13908[0] + tmp_13991[0], tmp_13908[1] + tmp_13991[1], tmp_13908[2] + tmp_13991[2]];
    signal tmp_13993[3] <== CMul()(challengeQ, tmp_13992);
    signal tmp_13994[3] <== CMul()(evals[81], evals[81]);
    signal tmp_13995[3] <== CMul()(tmp_13994, tmp_13994);
    signal tmp_13996[3] <== CMul()(tmp_13995, tmp_13994);
    signal tmp_13997[3] <== CMul()(tmp_13996, evals[81]);
    signal tmp_13998[3] <== [tmp_13997[0] + 9358535959712383718, tmp_13997[1], tmp_13997[2]];
    signal tmp_14000[3] <== [tmp_13913[0] * 3415, tmp_13913[1] * 3415, tmp_13913[2] * 3415];
    signal tmp_14001[3] <== [tmp_13916[0] + tmp_14000[0], tmp_13916[1] + tmp_14000[1], tmp_13916[2] + tmp_14000[2]];
    signal tmp_14002[3] <== [8193989947050030928 * tmp_14001[0], 8193989947050030928 * tmp_14001[1], 8193989947050030928 * tmp_14001[2]];
    signal tmp_14003[3] <== [tmp_13998[0] + tmp_14002[0], tmp_13998[1] + tmp_14002[1], tmp_13998[2] + tmp_14002[2]];
    signal tmp_14005[3] <== [tmp_13913[0] * 9781, tmp_13913[1] * 9781, tmp_13913[2] * 9781];
    signal tmp_14006[3] <== [tmp_13921[0] + tmp_14005[0], tmp_13921[1] + tmp_14005[1], tmp_13921[2] + tmp_14005[2]];
    signal tmp_14007[3] <== [8752166280136533348 * tmp_14006[0], 8752166280136533348 * tmp_14006[1], 8752166280136533348 * tmp_14006[2]];
    signal tmp_14008[3] <== [tmp_14003[0] + tmp_14007[0], tmp_14003[1] + tmp_14007[1], tmp_14003[2] + tmp_14007[2]];
    signal tmp_14010[3] <== [tmp_13913[0] * 5292, tmp_13913[1] * 5292, tmp_13913[2] * 5292];
    signal tmp_14011[3] <== [tmp_13926[0] + tmp_14010[0], tmp_13926[1] + tmp_14010[1], tmp_13926[2] + tmp_14010[2]];
    signal tmp_14012[3] <== [13837989622971627817 * tmp_14011[0], 13837989622971627817 * tmp_14011[1], 13837989622971627817 * tmp_14011[2]];
    signal tmp_14013[3] <== [tmp_14008[0] + tmp_14012[0], tmp_14008[1] + tmp_14012[1], tmp_14008[2] + tmp_14012[2]];
    signal tmp_14015[3] <== [tmp_13913[0] * 4288, tmp_13913[1] * 4288, tmp_13913[2] * 4288];
    signal tmp_14016[3] <== [tmp_13931[0] + tmp_14015[0], tmp_13931[1] + tmp_14015[1], tmp_13931[2] + tmp_14015[2]];
    signal tmp_14017[3] <== [16232594342971813546 * tmp_14016[0], 16232594342971813546 * tmp_14016[1], 16232594342971813546 * tmp_14016[2]];
    signal tmp_14018[3] <== [tmp_14013[0] + tmp_14017[0], tmp_14013[1] + tmp_14017[1], tmp_14013[2] + tmp_14017[2]];
    signal tmp_14020[3] <== [tmp_13913[0] * 7724, tmp_13913[1] * 7724, tmp_13913[2] * 7724];
    signal tmp_14021[3] <== [tmp_13936[0] + tmp_14020[0], tmp_13936[1] + tmp_14020[1], tmp_13936[2] + tmp_14020[2]];
    signal tmp_14022[3] <== [16475524618737509996 * tmp_14021[0], 16475524618737509996 * tmp_14021[1], 16475524618737509996 * tmp_14021[2]];
    signal tmp_14023[3] <== [tmp_14018[0] + tmp_14022[0], tmp_14018[1] + tmp_14022[1], tmp_14018[2] + tmp_14022[2]];
    signal tmp_14025[3] <== [tmp_13913[0] * 13016, tmp_13913[1] * 13016, tmp_13913[2] * 13016];
    signal tmp_14026[3] <== [tmp_13941[0] + tmp_14025[0], tmp_13941[1] + tmp_14025[1], tmp_13941[2] + tmp_14025[2]];
    signal tmp_14027[3] <== [13012351369794380536 * tmp_14026[0], 13012351369794380536 * tmp_14026[1], 13012351369794380536 * tmp_14026[2]];
    signal tmp_14028[3] <== [tmp_14023[0] + tmp_14027[0], tmp_14023[1] + tmp_14027[1], tmp_14023[2] + tmp_14027[2]];
    signal tmp_14030[3] <== [tmp_13913[0] * 3835, tmp_13913[1] * 3835, tmp_13913[2] * 3835];
    signal tmp_14031[3] <== [tmp_13946[0] + tmp_14030[0], tmp_13946[1] + tmp_14030[1], tmp_13946[2] + tmp_14030[2]];
    signal tmp_14032[3] <== [1790980186545924701 * tmp_14031[0], 1790980186545924701 * tmp_14031[1], 1790980186545924701 * tmp_14031[2]];
    signal tmp_14033[3] <== [tmp_14028[0] + tmp_14032[0], tmp_14028[1] + tmp_14032[1], tmp_14028[2] + tmp_14032[2]];
    signal tmp_14035[3] <== [tmp_13913[0] * 5807, tmp_13913[1] * 5807, tmp_13913[2] * 5807];
    signal tmp_14036[3] <== [tmp_13951[0] + tmp_14035[0], tmp_13951[1] + tmp_14035[1], tmp_13951[2] + tmp_14035[2]];
    signal tmp_14037[3] <== [7932420325258865009 * tmp_14036[0], 7932420325258865009 * tmp_14036[1], 7932420325258865009 * tmp_14036[2]];
    signal tmp_14038[3] <== [tmp_14033[0] + tmp_14037[0], tmp_14033[1] + tmp_14037[1], tmp_14033[2] + tmp_14037[2]];
    signal tmp_14040[3] <== [tmp_13913[0] * 4933, tmp_13913[1] * 4933, tmp_13913[2] * 4933];
    signal tmp_14041[3] <== [tmp_13956[0] + tmp_14040[0], tmp_13956[1] + tmp_14040[1], tmp_13956[2] + tmp_14040[2]];
    signal tmp_14042[3] <== [12730743194106063118 * tmp_14041[0], 12730743194106063118 * tmp_14041[1], 12730743194106063118 * tmp_14041[2]];
    signal tmp_14043[3] <== [tmp_14038[0] + tmp_14042[0], tmp_14038[1] + tmp_14042[1], tmp_14038[2] + tmp_14042[2]];
    signal tmp_14045[3] <== [tmp_13913[0] * 8577, tmp_13913[1] * 8577, tmp_13913[2] * 8577];
    signal tmp_14046[3] <== [tmp_13961[0] + tmp_14045[0], tmp_13961[1] + tmp_14045[1], tmp_13961[2] + tmp_14045[2]];
    signal tmp_14047[3] <== [14158202809073107037 * tmp_14046[0], 14158202809073107037 * tmp_14046[1], 14158202809073107037 * tmp_14046[2]];
    signal tmp_14048[3] <== [tmp_14043[0] + tmp_14047[0], tmp_14043[1] + tmp_14047[1], tmp_14043[2] + tmp_14047[2]];
    signal tmp_14050[3] <== [tmp_13913[0] * 13125, tmp_13913[1] * 13125, tmp_13913[2] * 13125];
    signal tmp_14051[3] <== [tmp_13966[0] + tmp_14050[0], tmp_13966[1] + tmp_14050[1], tmp_13966[2] + tmp_14050[2]];
    signal tmp_14052[3] <== [8949967597911797979 * tmp_14051[0], 8949967597911797979 * tmp_14051[1], 8949967597911797979 * tmp_14051[2]];
    signal tmp_14053[3] <== [tmp_14048[0] + tmp_14052[0], tmp_14048[1] + tmp_14052[1], tmp_14048[2] + tmp_14052[2]];
    signal tmp_14055[3] <== [tmp_13913[0] * 16823, tmp_13913[1] * 16823, tmp_13913[2] * 16823];
    signal tmp_14056[3] <== [tmp_13971[0] + tmp_14055[0], tmp_13971[1] + tmp_14055[1], tmp_13971[2] + tmp_14055[2]];
    signal tmp_14057[3] <== [2528834519776872471 * tmp_14056[0], 2528834519776872471 * tmp_14056[1], 2528834519776872471 * tmp_14056[2]];
    signal tmp_14058[3] <== [tmp_14053[0] + tmp_14057[0], tmp_14053[1] + tmp_14057[1], tmp_14053[2] + tmp_14057[2]];
    signal tmp_14060[3] <== [tmp_13913[0] * 3127, tmp_13913[1] * 3127, tmp_13913[2] * 3127];
    signal tmp_14061[3] <== [tmp_13976[0] + tmp_14060[0], tmp_13976[1] + tmp_14060[1], tmp_13976[2] + tmp_14060[2]];
    signal tmp_14062[3] <== [12882437260298977882 * tmp_14061[0], 12882437260298977882 * tmp_14061[1], 12882437260298977882 * tmp_14061[2]];
    signal tmp_14063[3] <== [tmp_14058[0] + tmp_14062[0], tmp_14058[1] + tmp_14062[1], tmp_14058[2] + tmp_14062[2]];
    signal tmp_14065[3] <== [tmp_13913[0] * 8363, tmp_13913[1] * 8363, tmp_13913[2] * 8363];
    signal tmp_14066[3] <== [tmp_13981[0] + tmp_14065[0], tmp_13981[1] + tmp_14065[1], tmp_13981[2] + tmp_14065[2]];
    signal tmp_14067[3] <== [14815685759369386413 * tmp_14066[0], 14815685759369386413 * tmp_14066[1], 14815685759369386413 * tmp_14066[2]];
    signal tmp_14068[3] <== [tmp_14063[0] + tmp_14067[0], tmp_14063[1] + tmp_14067[1], tmp_14063[2] + tmp_14067[2]];
    signal tmp_14070[3] <== [tmp_13913[0] * 15859, tmp_13913[1] * 15859, tmp_13913[2] * 15859];
    signal tmp_14071[3] <== [tmp_13986[0] + tmp_14070[0], tmp_13986[1] + tmp_14070[1], tmp_13986[2] + tmp_14070[2]];
    signal tmp_14072[3] <== [6859761478933209387 * tmp_14071[0], 6859761478933209387 * tmp_14071[1], 6859761478933209387 * tmp_14071[2]];
    signal tmp_14073[3] <== [tmp_14068[0] + tmp_14072[0], tmp_14068[1] + tmp_14072[1], tmp_14068[2] + tmp_14072[2]];
    signal tmp_14074[3] <== [tmp_14073[0] * 1, tmp_14073[1] * 1, tmp_14073[2] * 1];
    signal tmp_14075[3] <== [evals[135][0] - tmp_14074[0], evals[135][1] - tmp_14074[1], evals[135][2] - tmp_14074[2]];
    signal tmp_14076[3] <== CMul()(tmp_9163, tmp_14075);
    signal tmp_14077[3] <== [tmp_13993[0] + tmp_14076[0], tmp_13993[1] + tmp_14076[1], tmp_13993[2] + tmp_14076[2]];
    signal tmp_14078[3] <== CMul()(challengeQ, tmp_14077);
    signal tmp_14080[3] <== [tmp_13998[0] * 3, tmp_13998[1] * 3, tmp_13998[2] * 3];
    signal tmp_14081[3] <== [tmp_14001[0] + tmp_14080[0], tmp_14001[1] + tmp_14080[1], tmp_14001[2] + tmp_14080[2]];
    signal tmp_14082[3] <== [tmp_14081[0] * 1, tmp_14081[1] * 1, tmp_14081[2] * 1];
    signal tmp_14083[3] <== [evals[136][0] - tmp_14082[0], evals[136][1] - tmp_14082[1], evals[136][2] - tmp_14082[2]];
    signal tmp_14084[3] <== CMul()(tmp_9163, tmp_14083);
    signal tmp_14085[3] <== [tmp_14078[0] + tmp_14084[0], tmp_14078[1] + tmp_14084[1], tmp_14078[2] + tmp_14084[2]];
    signal tmp_14086[3] <== CMul()(challengeQ, tmp_14085);
    signal tmp_14088[3] <== [tmp_13998[0] * 13, tmp_13998[1] * 13, tmp_13998[2] * 13];
    signal tmp_14089[3] <== [tmp_14006[0] + tmp_14088[0], tmp_14006[1] + tmp_14088[1], tmp_14006[2] + tmp_14088[2]];
    signal tmp_14090[3] <== [tmp_14089[0] * 1, tmp_14089[1] * 1, tmp_14089[2] * 1];
    signal tmp_14091[3] <== [evals[137][0] - tmp_14090[0], evals[137][1] - tmp_14090[1], evals[137][2] - tmp_14090[2]];
    signal tmp_14092[3] <== CMul()(tmp_9163, tmp_14091);
    signal tmp_14093[3] <== [tmp_14086[0] + tmp_14092[0], tmp_14086[1] + tmp_14092[1], tmp_14086[2] + tmp_14092[2]];
    signal tmp_14094[3] <== CMul()(challengeQ, tmp_14093);
    signal tmp_14096[3] <== [tmp_13998[0] * 22, tmp_13998[1] * 22, tmp_13998[2] * 22];
    signal tmp_14097[3] <== [tmp_14011[0] + tmp_14096[0], tmp_14011[1] + tmp_14096[1], tmp_14011[2] + tmp_14096[2]];
    signal tmp_14098[3] <== [tmp_14097[0] * 1, tmp_14097[1] * 1, tmp_14097[2] * 1];
    signal tmp_14099[3] <== [evals[138][0] - tmp_14098[0], evals[138][1] - tmp_14098[1], evals[138][2] - tmp_14098[2]];
    signal tmp_14100[3] <== CMul()(tmp_9163, tmp_14099);
    signal tmp_14101[3] <== [tmp_14094[0] + tmp_14100[0], tmp_14094[1] + tmp_14100[1], tmp_14094[2] + tmp_14100[2]];
    signal tmp_14102[3] <== CMul()(challengeQ, tmp_14101);
    signal tmp_14104[3] <== [tmp_13998[0] * 67, tmp_13998[1] * 67, tmp_13998[2] * 67];
    signal tmp_14105[3] <== [tmp_14016[0] + tmp_14104[0], tmp_14016[1] + tmp_14104[1], tmp_14016[2] + tmp_14104[2]];
    signal tmp_14106[3] <== [tmp_14105[0] * 1, tmp_14105[1] * 1, tmp_14105[2] * 1];
    signal tmp_14107[3] <== [evals[139][0] - tmp_14106[0], evals[139][1] - tmp_14106[1], evals[139][2] - tmp_14106[2]];
    signal tmp_14108[3] <== CMul()(tmp_9163, tmp_14107);
    signal tmp_14109[3] <== [tmp_14102[0] + tmp_14108[0], tmp_14102[1] + tmp_14108[1], tmp_14102[2] + tmp_14108[2]];
    signal tmp_14110[3] <== CMul()(challengeQ, tmp_14109);
    signal tmp_14112[3] <== [tmp_13998[0] * 2, tmp_13998[1] * 2, tmp_13998[2] * 2];
    signal tmp_14113[3] <== [tmp_14021[0] + tmp_14112[0], tmp_14021[1] + tmp_14112[1], tmp_14021[2] + tmp_14112[2]];
    signal tmp_14114[3] <== [tmp_14113[0] * 1, tmp_14113[1] * 1, tmp_14113[2] * 1];
    signal tmp_14115[3] <== [evals[140][0] - tmp_14114[0], evals[140][1] - tmp_14114[1], evals[140][2] - tmp_14114[2]];
    signal tmp_14116[3] <== CMul()(tmp_9163, tmp_14115);
    signal tmp_14117[3] <== [tmp_14110[0] + tmp_14116[0], tmp_14110[1] + tmp_14116[1], tmp_14110[2] + tmp_14116[2]];
    signal tmp_14118[3] <== CMul()(challengeQ, tmp_14117);
    signal tmp_14120[3] <== [tmp_13998[0] * 15, tmp_13998[1] * 15, tmp_13998[2] * 15];
    signal tmp_14121[3] <== [tmp_14026[0] + tmp_14120[0], tmp_14026[1] + tmp_14120[1], tmp_14026[2] + tmp_14120[2]];
    signal tmp_14122[3] <== [tmp_14121[0] * 1, tmp_14121[1] * 1, tmp_14121[2] * 1];
    signal tmp_14123[3] <== [evals[141][0] - tmp_14122[0], evals[141][1] - tmp_14122[1], evals[141][2] - tmp_14122[2]];
    signal tmp_14124[3] <== CMul()(tmp_9163, tmp_14123);
    signal tmp_14125[3] <== [tmp_14118[0] + tmp_14124[0], tmp_14118[1] + tmp_14124[1], tmp_14118[2] + tmp_14124[2]];
    signal tmp_14126[3] <== CMul()(challengeQ, tmp_14125);
    signal tmp_14128[3] <== [tmp_13998[0] * 63, tmp_13998[1] * 63, tmp_13998[2] * 63];
    signal tmp_14129[3] <== [tmp_14031[0] + tmp_14128[0], tmp_14031[1] + tmp_14128[1], tmp_14031[2] + tmp_14128[2]];
    signal tmp_14130[3] <== [tmp_14129[0] * 1, tmp_14129[1] * 1, tmp_14129[2] * 1];
    signal tmp_14131[3] <== [evals[142][0] - tmp_14130[0], evals[142][1] - tmp_14130[1], evals[142][2] - tmp_14130[2]];
    signal tmp_14132[3] <== CMul()(tmp_9163, tmp_14131);
    signal tmp_14133[3] <== [tmp_14126[0] + tmp_14132[0], tmp_14126[1] + tmp_14132[1], tmp_14126[2] + tmp_14132[2]];
    signal tmp_14134[3] <== CMul()(challengeQ, tmp_14133);
    signal tmp_14136[3] <== [tmp_13998[0] * 101, tmp_13998[1] * 101, tmp_13998[2] * 101];
    signal tmp_14137[3] <== [tmp_14036[0] + tmp_14136[0], tmp_14036[1] + tmp_14136[1], tmp_14036[2] + tmp_14136[2]];
    signal tmp_14138[3] <== [tmp_14137[0] * 1, tmp_14137[1] * 1, tmp_14137[2] * 1];
    signal tmp_14139[3] <== [evals[143][0] - tmp_14138[0], evals[143][1] - tmp_14138[1], evals[143][2] - tmp_14138[2]];
    signal tmp_14140[3] <== CMul()(tmp_9163, tmp_14139);
    signal tmp_14141[3] <== [tmp_14134[0] + tmp_14140[0], tmp_14134[1] + tmp_14140[1], tmp_14134[2] + tmp_14140[2]];
    signal tmp_14142[3] <== CMul()(challengeQ, tmp_14141);
    signal tmp_14144[3] <== [tmp_14041[0] + tmp_13998[0], tmp_14041[1] + tmp_13998[1], tmp_14041[2] + tmp_13998[2]];
    signal tmp_14145[3] <== [tmp_14144[0] * 1, tmp_14144[1] * 1, tmp_14144[2] * 1];
    signal tmp_14146[3] <== [evals[144][0] - tmp_14145[0], evals[144][1] - tmp_14145[1], evals[144][2] - tmp_14145[2]];
    signal tmp_14147[3] <== CMul()(tmp_9163, tmp_14146);
    signal tmp_14148[3] <== [tmp_14142[0] + tmp_14147[0], tmp_14142[1] + tmp_14147[1], tmp_14142[2] + tmp_14147[2]];
    signal tmp_14149[3] <== CMul()(challengeQ, tmp_14148);
    signal tmp_14152[3] <== [tmp_14046[0] + tmp_14112[0], tmp_14046[1] + tmp_14112[1], tmp_14046[2] + tmp_14112[2]];
    signal tmp_14153[3] <== [tmp_14152[0] * 1, tmp_14152[1] * 1, tmp_14152[2] * 1];
    signal tmp_14154[3] <== [evals[145][0] - tmp_14153[0], evals[145][1] - tmp_14153[1], evals[145][2] - tmp_14153[2]];
    signal tmp_14155[3] <== CMul()(tmp_9163, tmp_14154);
    signal tmp_14156[3] <== [tmp_14149[0] + tmp_14155[0], tmp_14149[1] + tmp_14155[1], tmp_14149[2] + tmp_14155[2]];
    signal tmp_14157[3] <== CMul()(challengeQ, tmp_14156);
    signal tmp_14159[3] <== [tmp_13998[0] * 17, tmp_13998[1] * 17, tmp_13998[2] * 17];
    signal tmp_14160[3] <== [tmp_14051[0] + tmp_14159[0], tmp_14051[1] + tmp_14159[1], tmp_14051[2] + tmp_14159[2]];
    signal tmp_14161[3] <== [tmp_14160[0] * 1, tmp_14160[1] * 1, tmp_14160[2] * 1];
    signal tmp_14162[3] <== [evals[146][0] - tmp_14161[0], evals[146][1] - tmp_14161[1], evals[146][2] - tmp_14161[2]];
    signal tmp_14163[3] <== CMul()(tmp_9163, tmp_14162);
    signal tmp_14164[3] <== [tmp_14157[0] + tmp_14163[0], tmp_14157[1] + tmp_14163[1], tmp_14157[2] + tmp_14163[2]];
    signal tmp_14165[3] <== CMul()(challengeQ, tmp_14164);
    signal tmp_14167[3] <== [tmp_13998[0] * 11, tmp_13998[1] * 11, tmp_13998[2] * 11];
    signal tmp_14168[3] <== [tmp_14056[0] + tmp_14167[0], tmp_14056[1] + tmp_14167[1], tmp_14056[2] + tmp_14167[2]];
    signal tmp_14169[3] <== [tmp_14168[0] * 1, tmp_14168[1] * 1, tmp_14168[2] * 1];
    signal tmp_14170[3] <== [evals[147][0] - tmp_14169[0], evals[147][1] - tmp_14169[1], evals[147][2] - tmp_14169[2]];
    signal tmp_14171[3] <== CMul()(tmp_9163, tmp_14170);
    signal tmp_14172[3] <== [tmp_14165[0] + tmp_14171[0], tmp_14165[1] + tmp_14171[1], tmp_14165[2] + tmp_14171[2]];
    signal tmp_14173[3] <== CMul()(challengeQ, tmp_14172);
    signal tmp_14175[3] <== [tmp_14061[0] + tmp_13998[0], tmp_14061[1] + tmp_13998[1], tmp_14061[2] + tmp_13998[2]];
    signal tmp_14176[3] <== [tmp_14175[0] * 1, tmp_14175[1] * 1, tmp_14175[2] * 1];
    signal tmp_14177[3] <== [evals[148][0] - tmp_14176[0], evals[148][1] - tmp_14176[1], evals[148][2] - tmp_14176[2]];
    signal tmp_14178[3] <== CMul()(tmp_9163, tmp_14177);
    signal tmp_14179[3] <== [tmp_14173[0] + tmp_14178[0], tmp_14173[1] + tmp_14178[1], tmp_14173[2] + tmp_14178[2]];
    signal tmp_14180[3] <== CMul()(challengeQ, tmp_14179);
    signal tmp_14182[3] <== [tmp_13998[0] * 51, tmp_13998[1] * 51, tmp_13998[2] * 51];
    signal tmp_14183[3] <== [tmp_14066[0] + tmp_14182[0], tmp_14066[1] + tmp_14182[1], tmp_14066[2] + tmp_14182[2]];
    signal tmp_14184[3] <== [tmp_14183[0] * 1, tmp_14183[1] * 1, tmp_14183[2] * 1];
    signal tmp_14185[3] <== [evals[149][0] - tmp_14184[0], evals[149][1] - tmp_14184[1], evals[149][2] - tmp_14184[2]];
    signal tmp_14186[3] <== CMul()(tmp_9163, tmp_14185);
    signal tmp_14187[3] <== [tmp_14180[0] + tmp_14186[0], tmp_14180[1] + tmp_14186[1], tmp_14180[2] + tmp_14186[2]];
    signal tmp_14188[3] <== CMul()(challengeQ, tmp_14187);
    signal tmp_14190[3] <== [tmp_14071[0] + tmp_13998[0], tmp_14071[1] + tmp_13998[1], tmp_14071[2] + tmp_13998[2]];
    signal tmp_14191[3] <== [tmp_14190[0] * 1, tmp_14190[1] * 1, tmp_14190[2] * 1];
    signal tmp_14192[3] <== [evals[150][0] - tmp_14191[0], evals[150][1] - tmp_14191[1], evals[150][2] - tmp_14191[2]];
    signal tmp_14193[3] <== CMul()(tmp_9163, tmp_14192);
    signal tmp_14194[3] <== [tmp_14188[0] + tmp_14193[0], tmp_14188[1] + tmp_14193[1], tmp_14188[2] + tmp_14193[2]];
    signal tmp_14195[3] <== CMul()(challengeQ, tmp_14194);
    signal tmp_14196[3] <== CMul()(evals[67], evals[70]);
    signal tmp_14197[3] <== CMul()(evals[68], evals[72]);
    signal tmp_14198[3] <== [tmp_14196[0] + tmp_14197[0], tmp_14196[1] + tmp_14197[1], tmp_14196[2] + tmp_14197[2]];
    signal tmp_14199[3] <== CMul()(evals[69], evals[71]);
    signal tmp_14200[3] <== [tmp_14198[0] + tmp_14199[0], tmp_14198[1] + tmp_14199[1], tmp_14198[2] + tmp_14199[2]];
    signal tmp_14201[3] <== [evals[73][0] - tmp_14200[0], evals[73][1] - tmp_14200[1], evals[73][2] - tmp_14200[2]];
    signal tmp_14202[3] <== CMul()(evals[58], tmp_14201);
    signal tmp_14203[3] <== [tmp_14195[0] + tmp_14202[0], tmp_14195[1] + tmp_14202[1], tmp_14195[2] + tmp_14202[2]];
    signal tmp_14204[3] <== CMul()(challengeQ, tmp_14203);
    signal tmp_14205[3] <== CMul()(evals[67], evals[71]);
    signal tmp_14206[3] <== CMul()(evals[68], evals[70]);
    signal tmp_14207[3] <== [tmp_14205[0] + tmp_14206[0], tmp_14205[1] + tmp_14206[1], tmp_14205[2] + tmp_14206[2]];
    signal tmp_14209[3] <== [tmp_14207[0] + tmp_14197[0], tmp_14207[1] + tmp_14197[1], tmp_14207[2] + tmp_14197[2]];
    signal tmp_14211[3] <== [tmp_14209[0] + tmp_14199[0], tmp_14209[1] + tmp_14199[1], tmp_14209[2] + tmp_14199[2]];
    signal tmp_14212[3] <== CMul()(evals[69], evals[72]);
    signal tmp_14213[3] <== [tmp_14211[0] + tmp_14212[0], tmp_14211[1] + tmp_14212[1], tmp_14211[2] + tmp_14212[2]];
    signal tmp_14214[3] <== [evals[74][0] - tmp_14213[0], evals[74][1] - tmp_14213[1], evals[74][2] - tmp_14213[2]];
    signal tmp_14215[3] <== CMul()(evals[58], tmp_14214);
    signal tmp_14216[3] <== [tmp_14204[0] + tmp_14215[0], tmp_14204[1] + tmp_14215[1], tmp_14204[2] + tmp_14215[2]];
    signal tmp_14217[3] <== CMul()(challengeQ, tmp_14216);
    signal tmp_14218[3] <== CMul()(evals[67], evals[72]);
    signal tmp_14220[3] <== [tmp_14218[0] + tmp_14212[0], tmp_14218[1] + tmp_14212[1], tmp_14218[2] + tmp_14212[2]];
    signal tmp_14221[3] <== CMul()(evals[69], evals[70]);
    signal tmp_14222[3] <== [tmp_14220[0] + tmp_14221[0], tmp_14220[1] + tmp_14221[1], tmp_14220[2] + tmp_14221[2]];
    signal tmp_14223[3] <== CMul()(evals[68], evals[71]);
    signal tmp_14224[3] <== [tmp_14222[0] + tmp_14223[0], tmp_14222[1] + tmp_14223[1], tmp_14222[2] + tmp_14223[2]];
    signal tmp_14225[3] <== [evals[75][0] - tmp_14224[0], evals[75][1] - tmp_14224[1], evals[75][2] - tmp_14224[2]];
    signal tmp_14226[3] <== CMul()(evals[58], tmp_14225);
    signal tmp_14227[3] <== [tmp_14217[0] + tmp_14226[0], tmp_14217[1] + tmp_14226[1], tmp_14217[2] + tmp_14226[2]];
    signal tmp_14228[3] <== CMul()(challengeQ, tmp_14227);
    signal tmp_14229[3] <== CMul()(evals[76], evals[79]);
    signal tmp_14230[3] <== CMul()(evals[77], evals[81]);
    signal tmp_14231[3] <== [tmp_14229[0] + tmp_14230[0], tmp_14229[1] + tmp_14230[1], tmp_14229[2] + tmp_14230[2]];
    signal tmp_14232[3] <== CMul()(evals[78], evals[80]);
    signal tmp_14233[3] <== [tmp_14231[0] + tmp_14232[0], tmp_14231[1] + tmp_14232[1], tmp_14231[2] + tmp_14232[2]];
    signal tmp_14234[3] <== [evals[82][0] - tmp_14233[0], evals[82][1] - tmp_14233[1], evals[82][2] - tmp_14233[2]];
    signal tmp_14235[3] <== CMul()(evals[58], tmp_14234);
    signal tmp_14236[3] <== [tmp_14228[0] + tmp_14235[0], tmp_14228[1] + tmp_14235[1], tmp_14228[2] + tmp_14235[2]];
    signal tmp_14237[3] <== CMul()(challengeQ, tmp_14236);
    signal tmp_14238[3] <== CMul()(evals[76], evals[80]);
    signal tmp_14239[3] <== CMul()(evals[77], evals[79]);
    signal tmp_14240[3] <== [tmp_14238[0] + tmp_14239[0], tmp_14238[1] + tmp_14239[1], tmp_14238[2] + tmp_14239[2]];
    signal tmp_14242[3] <== [tmp_14240[0] + tmp_14230[0], tmp_14240[1] + tmp_14230[1], tmp_14240[2] + tmp_14230[2]];
    signal tmp_14244[3] <== [tmp_14242[0] + tmp_14232[0], tmp_14242[1] + tmp_14232[1], tmp_14242[2] + tmp_14232[2]];
    signal tmp_14245[3] <== CMul()(evals[78], evals[81]);
    signal tmp_14246[3] <== [tmp_14244[0] + tmp_14245[0], tmp_14244[1] + tmp_14245[1], tmp_14244[2] + tmp_14245[2]];
    signal tmp_14247[3] <== [evals[83][0] - tmp_14246[0], evals[83][1] - tmp_14246[1], evals[83][2] - tmp_14246[2]];
    signal tmp_14248[3] <== CMul()(evals[58], tmp_14247);
    signal tmp_14249[3] <== [tmp_14237[0] + tmp_14248[0], tmp_14237[1] + tmp_14248[1], tmp_14237[2] + tmp_14248[2]];
    signal tmp_14250[3] <== CMul()(challengeQ, tmp_14249);
    signal tmp_14251[3] <== CMul()(evals[76], evals[81]);
    signal tmp_14253[3] <== [tmp_14251[0] + tmp_14245[0], tmp_14251[1] + tmp_14245[1], tmp_14251[2] + tmp_14245[2]];
    signal tmp_14254[3] <== CMul()(evals[78], evals[79]);
    signal tmp_14255[3] <== [tmp_14253[0] + tmp_14254[0], tmp_14253[1] + tmp_14254[1], tmp_14253[2] + tmp_14254[2]];
    signal tmp_14256[3] <== CMul()(evals[77], evals[80]);
    signal tmp_14257[3] <== [tmp_14255[0] + tmp_14256[0], tmp_14255[1] + tmp_14256[1], tmp_14255[2] + tmp_14256[2]];
    signal tmp_14258[3] <== [evals[84][0] - tmp_14257[0], evals[84][1] - tmp_14257[1], evals[84][2] - tmp_14257[2]];
    signal tmp_14259[3] <== CMul()(evals[58], tmp_14258);
    signal tmp_14260[3] <== [tmp_14250[0] + tmp_14259[0], tmp_14250[1] + tmp_14259[1], tmp_14250[2] + tmp_14259[2]];
    signal tmp_14261[3] <== CMul()(challengeQ, tmp_14260);
    signal tmp_14262[3] <== CMul()(evals[45], evals[67]);
    signal tmp_14264[3] <== [tmp_14262[0] + tmp_9180[0], tmp_14262[1] + tmp_9180[1], tmp_14262[2] + tmp_9180[2]];
    signal tmp_14265[3] <== CMul()(evals[47], evals[73]);
    signal tmp_14266[3] <== [tmp_14264[0] + tmp_14265[0], tmp_14264[1] + tmp_14265[1], tmp_14264[2] + tmp_14265[2]];
    signal tmp_14267[3] <== CMul()(evals[48], evals[76]);
    signal tmp_14268[3] <== [tmp_14266[0] + tmp_14267[0], tmp_14266[1] + tmp_14267[1], tmp_14266[2] + tmp_14267[2]];
    signal tmp_14269[3] <== CMul()(evals[51], evals[67]);
    signal tmp_14270[3] <== [tmp_14268[0] + tmp_14269[0], tmp_14268[1] + tmp_14269[1], tmp_14268[2] + tmp_14269[2]];
    signal tmp_14271[3] <== CMul()(evals[52], evals[70]);
    signal tmp_14272[3] <== [tmp_14270[0] + tmp_14271[0], tmp_14270[1] + tmp_14271[1], tmp_14270[2] + tmp_14271[2]];
    signal tmp_14273[3] <== [evals[79][0] - tmp_14272[0], evals[79][1] - tmp_14272[1], evals[79][2] - tmp_14272[2]];
    signal tmp_14274[3] <== CMul()(evals[60], tmp_14273);
    signal tmp_14275[3] <== [tmp_14261[0] + tmp_14274[0], tmp_14261[1] + tmp_14274[1], tmp_14261[2] + tmp_14274[2]];
    signal tmp_14276[3] <== CMul()(challengeQ, tmp_14275);
    signal tmp_14277[3] <== CMul()(evals[45], evals[68]);
    signal tmp_14278[3] <== CMul()(evals[46], evals[71]);
    signal tmp_14279[3] <== [tmp_14277[0] + tmp_14278[0], tmp_14277[1] + tmp_14278[1], tmp_14277[2] + tmp_14278[2]];
    signal tmp_14280[3] <== CMul()(evals[47], evals[74]);
    signal tmp_14281[3] <== [tmp_14279[0] + tmp_14280[0], tmp_14279[1] + tmp_14280[1], tmp_14279[2] + tmp_14280[2]];
    signal tmp_14282[3] <== CMul()(evals[48], evals[77]);
    signal tmp_14283[3] <== [tmp_14281[0] + tmp_14282[0], tmp_14281[1] + tmp_14282[1], tmp_14281[2] + tmp_14282[2]];
    signal tmp_14284[3] <== CMul()(evals[51], evals[68]);
    signal tmp_14285[3] <== [tmp_14283[0] + tmp_14284[0], tmp_14283[1] + tmp_14284[1], tmp_14283[2] + tmp_14284[2]];
    signal tmp_14286[3] <== CMul()(evals[52], evals[71]);
    signal tmp_14287[3] <== [tmp_14285[0] + tmp_14286[0], tmp_14285[1] + tmp_14286[1], tmp_14285[2] + tmp_14286[2]];
    signal tmp_14288[3] <== [evals[80][0] - tmp_14287[0], evals[80][1] - tmp_14287[1], evals[80][2] - tmp_14287[2]];
    signal tmp_14289[3] <== CMul()(evals[60], tmp_14288);
    signal tmp_14290[3] <== [tmp_14276[0] + tmp_14289[0], tmp_14276[1] + tmp_14289[1], tmp_14276[2] + tmp_14289[2]];
    signal tmp_14291[3] <== CMul()(challengeQ, tmp_14290);
    signal tmp_14292[3] <== CMul()(evals[45], evals[69]);
    signal tmp_14293[3] <== CMul()(evals[46], evals[72]);
    signal tmp_14294[3] <== [tmp_14292[0] + tmp_14293[0], tmp_14292[1] + tmp_14293[1], tmp_14292[2] + tmp_14293[2]];
    signal tmp_14295[3] <== CMul()(evals[47], evals[75]);
    signal tmp_14296[3] <== [tmp_14294[0] + tmp_14295[0], tmp_14294[1] + tmp_14295[1], tmp_14294[2] + tmp_14295[2]];
    signal tmp_14297[3] <== CMul()(evals[48], evals[78]);
    signal tmp_14298[3] <== [tmp_14296[0] + tmp_14297[0], tmp_14296[1] + tmp_14297[1], tmp_14296[2] + tmp_14297[2]];
    signal tmp_14299[3] <== CMul()(evals[51], evals[69]);
    signal tmp_14300[3] <== [tmp_14298[0] + tmp_14299[0], tmp_14298[1] + tmp_14299[1], tmp_14298[2] + tmp_14299[2]];
    signal tmp_14301[3] <== CMul()(evals[52], evals[72]);
    signal tmp_14302[3] <== [tmp_14300[0] + tmp_14301[0], tmp_14300[1] + tmp_14301[1], tmp_14300[2] + tmp_14301[2]];
    signal tmp_14303[3] <== [evals[81][0] - tmp_14302[0], evals[81][1] - tmp_14302[1], evals[81][2] - tmp_14302[2]];
    signal tmp_14304[3] <== CMul()(evals[60], tmp_14303);
    signal tmp_14305[3] <== [tmp_14291[0] + tmp_14304[0], tmp_14291[1] + tmp_14304[1], tmp_14291[2] + tmp_14304[2]];
    signal tmp_14306[3] <== CMul()(challengeQ, tmp_14305);
    signal tmp_14309[3] <== [tmp_14262[0] - tmp_9180[0], tmp_14262[1] - tmp_9180[1], tmp_14262[2] - tmp_9180[2]];
    signal tmp_14310[3] <== CMul()(evals[49], evals[73]);
    signal tmp_14311[3] <== [tmp_14309[0] + tmp_14310[0], tmp_14309[1] + tmp_14310[1], tmp_14309[2] + tmp_14310[2]];
    signal tmp_14312[3] <== CMul()(evals[50], evals[76]);
    signal tmp_14313[3] <== [tmp_14311[0] - tmp_14312[0], tmp_14311[1] - tmp_14312[1], tmp_14311[2] - tmp_14312[2]];
    signal tmp_14315[3] <== [tmp_14313[0] + tmp_14269[0], tmp_14313[1] + tmp_14269[1], tmp_14313[2] + tmp_14269[2]];
    signal tmp_14317[3] <== [tmp_14315[0] - tmp_14271[0], tmp_14315[1] - tmp_14271[1], tmp_14315[2] - tmp_14271[2]];
    signal tmp_14318[3] <== [evals[82][0] - tmp_14317[0], evals[82][1] - tmp_14317[1], evals[82][2] - tmp_14317[2]];
    signal tmp_14319[3] <== CMul()(evals[60], tmp_14318);
    signal tmp_14320[3] <== [tmp_14306[0] + tmp_14319[0], tmp_14306[1] + tmp_14319[1], tmp_14306[2] + tmp_14319[2]];
    signal tmp_14321[3] <== CMul()(challengeQ, tmp_14320);
    signal tmp_14324[3] <== [tmp_14277[0] - tmp_14278[0], tmp_14277[1] - tmp_14278[1], tmp_14277[2] - tmp_14278[2]];
    signal tmp_14325[3] <== CMul()(evals[49], evals[74]);
    signal tmp_14326[3] <== [tmp_14324[0] + tmp_14325[0], tmp_14324[1] + tmp_14325[1], tmp_14324[2] + tmp_14325[2]];
    signal tmp_14327[3] <== CMul()(evals[50], evals[77]);
    signal tmp_14328[3] <== [tmp_14326[0] - tmp_14327[0], tmp_14326[1] - tmp_14327[1], tmp_14326[2] - tmp_14327[2]];
    signal tmp_14330[3] <== [tmp_14328[0] + tmp_14284[0], tmp_14328[1] + tmp_14284[1], tmp_14328[2] + tmp_14284[2]];
    signal tmp_14332[3] <== [tmp_14330[0] - tmp_14286[0], tmp_14330[1] - tmp_14286[1], tmp_14330[2] - tmp_14286[2]];
    signal tmp_14333[3] <== [evals[83][0] - tmp_14332[0], evals[83][1] - tmp_14332[1], evals[83][2] - tmp_14332[2]];
    signal tmp_14334[3] <== CMul()(evals[60], tmp_14333);
    signal tmp_14335[3] <== [tmp_14321[0] + tmp_14334[0], tmp_14321[1] + tmp_14334[1], tmp_14321[2] + tmp_14334[2]];
    signal tmp_14336[3] <== CMul()(challengeQ, tmp_14335);
    signal tmp_14339[3] <== [tmp_14292[0] - tmp_14293[0], tmp_14292[1] - tmp_14293[1], tmp_14292[2] - tmp_14293[2]];
    signal tmp_14340[3] <== CMul()(evals[49], evals[75]);
    signal tmp_14341[3] <== [tmp_14339[0] + tmp_14340[0], tmp_14339[1] + tmp_14340[1], tmp_14339[2] + tmp_14340[2]];
    signal tmp_14342[3] <== CMul()(evals[50], evals[78]);
    signal tmp_14343[3] <== [tmp_14341[0] - tmp_14342[0], tmp_14341[1] - tmp_14342[1], tmp_14341[2] - tmp_14342[2]];
    signal tmp_14345[3] <== [tmp_14343[0] + tmp_14299[0], tmp_14343[1] + tmp_14299[1], tmp_14343[2] + tmp_14299[2]];
    signal tmp_14347[3] <== [tmp_14345[0] - tmp_14301[0], tmp_14345[1] - tmp_14301[1], tmp_14345[2] - tmp_14301[2]];
    signal tmp_14348[3] <== [evals[84][0] - tmp_14347[0], evals[84][1] - tmp_14347[1], evals[84][2] - tmp_14347[2]];
    signal tmp_14349[3] <== CMul()(evals[60], tmp_14348);
    signal tmp_14350[3] <== [tmp_14336[0] + tmp_14349[0], tmp_14336[1] + tmp_14349[1], tmp_14336[2] + tmp_14349[2]];
    signal tmp_14351[3] <== CMul()(challengeQ, tmp_14350);
    signal tmp_14356[3] <== [tmp_14264[0] - tmp_14265[0], tmp_14264[1] - tmp_14265[1], tmp_14264[2] - tmp_14265[2]];
    signal tmp_14358[3] <== [tmp_14356[0] - tmp_14267[0], tmp_14356[1] - tmp_14267[1], tmp_14356[2] - tmp_14267[2]];
    signal tmp_14360[3] <== [tmp_14358[0] + tmp_9192[0], tmp_14358[1] + tmp_9192[1], tmp_14358[2] + tmp_9192[2]];
    signal tmp_14361[3] <== CMul()(evals[53], evals[76]);
    signal tmp_14362[3] <== [tmp_14360[0] + tmp_14361[0], tmp_14360[1] + tmp_14361[1], tmp_14360[2] + tmp_14361[2]];
    signal tmp_14363[3] <== [evals[85][0] - tmp_14362[0], evals[85][1] - tmp_14362[1], evals[85][2] - tmp_14362[2]];
    signal tmp_14364[3] <== CMul()(evals[60], tmp_14363);
    signal tmp_14365[3] <== [tmp_14351[0] + tmp_14364[0], tmp_14351[1] + tmp_14364[1], tmp_14351[2] + tmp_14364[2]];
    signal tmp_14366[3] <== CMul()(challengeQ, tmp_14365);
    signal tmp_14371[3] <== [tmp_14279[0] - tmp_14280[0], tmp_14279[1] - tmp_14280[1], tmp_14279[2] - tmp_14280[2]];
    signal tmp_14373[3] <== [tmp_14371[0] - tmp_14282[0], tmp_14371[1] - tmp_14282[1], tmp_14371[2] - tmp_14282[2]];
    signal tmp_14374[3] <== CMul()(evals[51], evals[74]);
    signal tmp_14375[3] <== [tmp_14373[0] + tmp_14374[0], tmp_14373[1] + tmp_14374[1], tmp_14373[2] + tmp_14374[2]];
    signal tmp_14376[3] <== CMul()(evals[53], evals[77]);
    signal tmp_14377[3] <== [tmp_14375[0] + tmp_14376[0], tmp_14375[1] + tmp_14376[1], tmp_14375[2] + tmp_14376[2]];
    signal tmp_14378[3] <== [evals[86][0] - tmp_14377[0], evals[86][1] - tmp_14377[1], evals[86][2] - tmp_14377[2]];
    signal tmp_14379[3] <== CMul()(evals[60], tmp_14378);
    signal tmp_14380[3] <== [tmp_14366[0] + tmp_14379[0], tmp_14366[1] + tmp_14379[1], tmp_14366[2] + tmp_14379[2]];
    signal tmp_14381[3] <== CMul()(challengeQ, tmp_14380);
    signal tmp_14386[3] <== [tmp_14294[0] - tmp_14295[0], tmp_14294[1] - tmp_14295[1], tmp_14294[2] - tmp_14295[2]];
    signal tmp_14388[3] <== [tmp_14386[0] - tmp_14297[0], tmp_14386[1] - tmp_14297[1], tmp_14386[2] - tmp_14297[2]];
    signal tmp_14389[3] <== CMul()(evals[51], evals[75]);
    signal tmp_14390[3] <== [tmp_14388[0] + tmp_14389[0], tmp_14388[1] + tmp_14389[1], tmp_14388[2] + tmp_14389[2]];
    signal tmp_14392[3] <== [tmp_14390[0] + tmp_9210[0], tmp_14390[1] + tmp_9210[1], tmp_14390[2] + tmp_9210[2]];
    signal tmp_14393[3] <== [evals[87][0] - tmp_14392[0], evals[87][1] - tmp_14392[1], evals[87][2] - tmp_14392[2]];
    signal tmp_14394[3] <== CMul()(evals[60], tmp_14393);
    signal tmp_14395[3] <== [tmp_14381[0] + tmp_14394[0], tmp_14381[1] + tmp_14394[1], tmp_14381[2] + tmp_14394[2]];
    signal tmp_14396[3] <== CMul()(challengeQ, tmp_14395);
    signal tmp_14401[3] <== [tmp_14309[0] - tmp_14310[0], tmp_14309[1] - tmp_14310[1], tmp_14309[2] - tmp_14310[2]];
    signal tmp_14403[3] <== [tmp_14401[0] + tmp_14312[0], tmp_14401[1] + tmp_14312[1], tmp_14401[2] + tmp_14312[2]];
    signal tmp_14405[3] <== [tmp_14403[0] + tmp_9192[0], tmp_14403[1] + tmp_9192[1], tmp_14403[2] + tmp_9192[2]];
    signal tmp_14407[3] <== [tmp_14405[0] - tmp_14361[0], tmp_14405[1] - tmp_14361[1], tmp_14405[2] - tmp_14361[2]];
    signal tmp_14408[3] <== [evals[88][0] - tmp_14407[0], evals[88][1] - tmp_14407[1], evals[88][2] - tmp_14407[2]];
    signal tmp_14409[3] <== CMul()(evals[60], tmp_14408);
    signal tmp_14410[3] <== [tmp_14396[0] + tmp_14409[0], tmp_14396[1] + tmp_14409[1], tmp_14396[2] + tmp_14409[2]];
    signal tmp_14411[3] <== CMul()(challengeQ, tmp_14410);
    signal tmp_14416[3] <== [tmp_14324[0] - tmp_14325[0], tmp_14324[1] - tmp_14325[1], tmp_14324[2] - tmp_14325[2]];
    signal tmp_14418[3] <== [tmp_14416[0] + tmp_14327[0], tmp_14416[1] + tmp_14327[1], tmp_14416[2] + tmp_14327[2]];
    signal tmp_14420[3] <== [tmp_14418[0] + tmp_14374[0], tmp_14418[1] + tmp_14374[1], tmp_14418[2] + tmp_14374[2]];
    signal tmp_14422[3] <== [tmp_14420[0] - tmp_14376[0], tmp_14420[1] - tmp_14376[1], tmp_14420[2] - tmp_14376[2]];
    signal tmp_14423[3] <== [evals[89][0] - tmp_14422[0], evals[89][1] - tmp_14422[1], evals[89][2] - tmp_14422[2]];
    signal tmp_14424[3] <== CMul()(evals[60], tmp_14423);
    signal tmp_14425[3] <== [tmp_14411[0] + tmp_14424[0], tmp_14411[1] + tmp_14424[1], tmp_14411[2] + tmp_14424[2]];
    signal tmp_14426[3] <== CMul()(challengeQ, tmp_14425);
    signal tmp_14431[3] <== [tmp_14339[0] - tmp_14340[0], tmp_14339[1] - tmp_14340[1], tmp_14339[2] - tmp_14340[2]];
    signal tmp_14433[3] <== [tmp_14431[0] + tmp_14342[0], tmp_14431[1] + tmp_14342[1], tmp_14431[2] + tmp_14342[2]];
    signal tmp_14435[3] <== [tmp_14433[0] + tmp_14389[0], tmp_14433[1] + tmp_14389[1], tmp_14433[2] + tmp_14389[2]];
    signal tmp_14437[3] <== [tmp_14435[0] - tmp_9210[0], tmp_14435[1] - tmp_9210[1], tmp_14435[2] - tmp_9210[2]];
    signal tmp_14438[3] <== [evals[90][0] - tmp_14437[0], evals[90][1] - tmp_14437[1], evals[90][2] - tmp_14437[2]];
    signal tmp_14439[3] <== CMul()(evals[60], tmp_14438);
    signal tmp_14440[3] <== [tmp_14426[0] + tmp_14439[0], tmp_14426[1] + tmp_14439[1], tmp_14426[2] + tmp_14439[2]];
    signal tmp_14441[3] <== CMul()(challengeQ, tmp_14440);
    signal tmp_14442[3] <== CMul()(evals[79], evals[82]);
    signal tmp_14443[3] <== CMul()(evals[80], evals[84]);
    signal tmp_14444[3] <== [tmp_14442[0] + tmp_14443[0], tmp_14442[1] + tmp_14443[1], tmp_14442[2] + tmp_14443[2]];
    signal tmp_14445[3] <== CMul()(evals[81], evals[83]);
    signal tmp_14446[3] <== [tmp_14444[0] + tmp_14445[0], tmp_14444[1] + tmp_14445[1], tmp_14444[2] + tmp_14445[2]];
    signal tmp_14447[3] <== [tmp_14446[0] + evals[76][0], tmp_14446[1] + evals[76][1], tmp_14446[2] + evals[76][2]];
    signal tmp_14448[3] <== CMul()(tmp_14447, evals[82]);
    signal tmp_14449[3] <== CMul()(evals[79], evals[83]);
    signal tmp_14450[3] <== CMul()(evals[80], evals[82]);
    signal tmp_14451[3] <== [tmp_14449[0] + tmp_14450[0], tmp_14449[1] + tmp_14450[1], tmp_14449[2] + tmp_14450[2]];
    signal tmp_14453[3] <== [tmp_14451[0] + tmp_14443[0], tmp_14451[1] + tmp_14443[1], tmp_14451[2] + tmp_14443[2]];
    signal tmp_14455[3] <== [tmp_14453[0] + tmp_14445[0], tmp_14453[1] + tmp_14445[1], tmp_14453[2] + tmp_14445[2]];
    signal tmp_14456[3] <== CMul()(evals[81], evals[84]);
    signal tmp_14457[3] <== [tmp_14455[0] + tmp_14456[0], tmp_14455[1] + tmp_14456[1], tmp_14455[2] + tmp_14456[2]];
    signal tmp_14458[3] <== [tmp_14457[0] + evals[77][0], tmp_14457[1] + evals[77][1], tmp_14457[2] + evals[77][2]];
    signal tmp_14459[3] <== CMul()(tmp_14458, evals[84]);
    signal tmp_14460[3] <== [tmp_14448[0] + tmp_14459[0], tmp_14448[1] + tmp_14459[1], tmp_14448[2] + tmp_14459[2]];
    signal tmp_14461[3] <== CMul()(evals[79], evals[84]);
    signal tmp_14463[3] <== [tmp_14461[0] + tmp_14456[0], tmp_14461[1] + tmp_14456[1], tmp_14461[2] + tmp_14456[2]];
    signal tmp_14464[3] <== CMul()(evals[81], evals[82]);
    signal tmp_14465[3] <== [tmp_14463[0] + tmp_14464[0], tmp_14463[1] + tmp_14464[1], tmp_14463[2] + tmp_14464[2]];
    signal tmp_14466[3] <== CMul()(evals[80], evals[83]);
    signal tmp_14467[3] <== [tmp_14465[0] + tmp_14466[0], tmp_14465[1] + tmp_14466[1], tmp_14465[2] + tmp_14466[2]];
    signal tmp_14468[3] <== [tmp_14467[0] + evals[78][0], tmp_14467[1] + evals[78][1], tmp_14467[2] + evals[78][2]];
    signal tmp_14469[3] <== CMul()(tmp_14468, evals[83]);
    signal tmp_14470[3] <== [tmp_14460[0] + tmp_14469[0], tmp_14460[1] + tmp_14469[1], tmp_14460[2] + tmp_14469[2]];
    signal tmp_14471[3] <== [tmp_14470[0] + evals[73][0], tmp_14470[1] + evals[73][1], tmp_14470[2] + evals[73][2]];
    signal tmp_14472[3] <== CMul()(tmp_14471, evals[82]);
    signal tmp_14473[3] <== CMul()(tmp_14447, evals[83]);
    signal tmp_14474[3] <== CMul()(tmp_14458, evals[82]);
    signal tmp_14475[3] <== [tmp_14473[0] + tmp_14474[0], tmp_14473[1] + tmp_14474[1], tmp_14473[2] + tmp_14474[2]];
    signal tmp_14477[3] <== [tmp_14475[0] + tmp_14459[0], tmp_14475[1] + tmp_14459[1], tmp_14475[2] + tmp_14459[2]];
    signal tmp_14479[3] <== [tmp_14477[0] + tmp_14469[0], tmp_14477[1] + tmp_14469[1], tmp_14477[2] + tmp_14469[2]];
    signal tmp_14480[3] <== CMul()(tmp_14468, evals[84]);
    signal tmp_14481[3] <== [tmp_14479[0] + tmp_14480[0], tmp_14479[1] + tmp_14480[1], tmp_14479[2] + tmp_14480[2]];
    signal tmp_14482[3] <== [tmp_14481[0] + evals[74][0], tmp_14481[1] + evals[74][1], tmp_14481[2] + evals[74][2]];
    signal tmp_14483[3] <== CMul()(tmp_14482, evals[84]);
    signal tmp_14484[3] <== [tmp_14472[0] + tmp_14483[0], tmp_14472[1] + tmp_14483[1], tmp_14472[2] + tmp_14483[2]];
    signal tmp_14485[3] <== CMul()(tmp_14447, evals[84]);
    signal tmp_14487[3] <== [tmp_14485[0] + tmp_14480[0], tmp_14485[1] + tmp_14480[1], tmp_14485[2] + tmp_14480[2]];
    signal tmp_14488[3] <== CMul()(tmp_14468, evals[82]);
    signal tmp_14489[3] <== [tmp_14487[0] + tmp_14488[0], tmp_14487[1] + tmp_14488[1], tmp_14487[2] + tmp_14488[2]];
    signal tmp_14490[3] <== CMul()(tmp_14458, evals[83]);
    signal tmp_14491[3] <== [tmp_14489[0] + tmp_14490[0], tmp_14489[1] + tmp_14490[1], tmp_14489[2] + tmp_14490[2]];
    signal tmp_14492[3] <== [tmp_14491[0] + evals[75][0], tmp_14491[1] + evals[75][1], tmp_14491[2] + evals[75][2]];
    signal tmp_14493[3] <== CMul()(tmp_14492, evals[83]);
    signal tmp_14494[3] <== [tmp_14484[0] + tmp_14493[0], tmp_14484[1] + tmp_14493[1], tmp_14484[2] + tmp_14493[2]];
    signal tmp_14495[3] <== [tmp_14494[0] + evals[70][0], tmp_14494[1] + evals[70][1], tmp_14494[2] + evals[70][2]];
    signal tmp_14496[3] <== CMul()(tmp_14495, evals[82]);
    signal tmp_14497[3] <== CMul()(tmp_14471, evals[83]);
    signal tmp_14498[3] <== CMul()(tmp_14482, evals[82]);
    signal tmp_14499[3] <== [tmp_14497[0] + tmp_14498[0], tmp_14497[1] + tmp_14498[1], tmp_14497[2] + tmp_14498[2]];
    signal tmp_14501[3] <== [tmp_14499[0] + tmp_14483[0], tmp_14499[1] + tmp_14483[1], tmp_14499[2] + tmp_14483[2]];
    signal tmp_14503[3] <== [tmp_14501[0] + tmp_14493[0], tmp_14501[1] + tmp_14493[1], tmp_14501[2] + tmp_14493[2]];
    signal tmp_14504[3] <== CMul()(tmp_14492, evals[84]);
    signal tmp_14505[3] <== [tmp_14503[0] + tmp_14504[0], tmp_14503[1] + tmp_14504[1], tmp_14503[2] + tmp_14504[2]];
    signal tmp_14506[3] <== [tmp_14505[0] + evals[71][0], tmp_14505[1] + evals[71][1], tmp_14505[2] + evals[71][2]];
    signal tmp_14507[3] <== CMul()(tmp_14506, evals[84]);
    signal tmp_14508[3] <== [tmp_14496[0] + tmp_14507[0], tmp_14496[1] + tmp_14507[1], tmp_14496[2] + tmp_14507[2]];
    signal tmp_14509[3] <== CMul()(tmp_14471, evals[84]);
    signal tmp_14511[3] <== [tmp_14509[0] + tmp_14504[0], tmp_14509[1] + tmp_14504[1], tmp_14509[2] + tmp_14504[2]];
    signal tmp_14512[3] <== CMul()(tmp_14492, evals[82]);
    signal tmp_14513[3] <== [tmp_14511[0] + tmp_14512[0], tmp_14511[1] + tmp_14512[1], tmp_14511[2] + tmp_14512[2]];
    signal tmp_14514[3] <== CMul()(tmp_14482, evals[83]);
    signal tmp_14515[3] <== [tmp_14513[0] + tmp_14514[0], tmp_14513[1] + tmp_14514[1], tmp_14513[2] + tmp_14514[2]];
    signal tmp_14516[3] <== [tmp_14515[0] + evals[72][0], tmp_14515[1] + evals[72][1], tmp_14515[2] + evals[72][2]];
    signal tmp_14517[3] <== CMul()(tmp_14516, evals[83]);
    signal tmp_14518[3] <== [tmp_14508[0] + tmp_14517[0], tmp_14508[1] + tmp_14517[1], tmp_14508[2] + tmp_14517[2]];
    signal tmp_14519[3] <== [tmp_14518[0] + evals[67][0], tmp_14518[1] + evals[67][1], tmp_14518[2] + evals[67][2]];
    signal tmp_14520[3] <== [tmp_14519[0] - evals[85][0], tmp_14519[1] - evals[85][1], tmp_14519[2] - evals[85][2]];
    signal tmp_14521[3] <== CMul()(evals[59], tmp_14520);
    signal tmp_14522[3] <== [tmp_14441[0] + tmp_14521[0], tmp_14441[1] + tmp_14521[1], tmp_14441[2] + tmp_14521[2]];
    signal tmp_14523[3] <== CMul()(challengeQ, tmp_14522);
    signal tmp_14524[3] <== CMul()(tmp_14495, evals[83]);
    signal tmp_14525[3] <== CMul()(tmp_14506, evals[82]);
    signal tmp_14526[3] <== [tmp_14524[0] + tmp_14525[0], tmp_14524[1] + tmp_14525[1], tmp_14524[2] + tmp_14525[2]];
    signal tmp_14528[3] <== [tmp_14526[0] + tmp_14507[0], tmp_14526[1] + tmp_14507[1], tmp_14526[2] + tmp_14507[2]];
    signal tmp_14530[3] <== [tmp_14528[0] + tmp_14517[0], tmp_14528[1] + tmp_14517[1], tmp_14528[2] + tmp_14517[2]];
    signal tmp_14531[3] <== CMul()(tmp_14516, evals[84]);
    signal tmp_14532[3] <== [tmp_14530[0] + tmp_14531[0], tmp_14530[1] + tmp_14531[1], tmp_14530[2] + tmp_14531[2]];
    signal tmp_14533[3] <== [tmp_14532[0] + evals[68][0], tmp_14532[1] + evals[68][1], tmp_14532[2] + evals[68][2]];
    signal tmp_14534[3] <== [tmp_14533[0] - evals[86][0], tmp_14533[1] - evals[86][1], tmp_14533[2] - evals[86][2]];
    signal tmp_14535[3] <== CMul()(evals[59], tmp_14534);
    signal tmp_14536[3] <== [tmp_14523[0] + tmp_14535[0], tmp_14523[1] + tmp_14535[1], tmp_14523[2] + tmp_14535[2]];
    signal tmp_14537[3] <== CMul()(challengeQ, tmp_14536);
    signal tmp_14538[3] <== CMul()(tmp_14495, evals[84]);
    signal tmp_14540[3] <== [tmp_14538[0] + tmp_14531[0], tmp_14538[1] + tmp_14531[1], tmp_14538[2] + tmp_14531[2]];
    signal tmp_14541[3] <== CMul()(tmp_14516, evals[82]);
    signal tmp_14542[3] <== [tmp_14540[0] + tmp_14541[0], tmp_14540[1] + tmp_14541[1], tmp_14540[2] + tmp_14541[2]];
    signal tmp_14543[3] <== CMul()(tmp_14506, evals[83]);
    signal tmp_14544[3] <== [tmp_14542[0] + tmp_14543[0], tmp_14542[1] + tmp_14543[1], tmp_14542[2] + tmp_14543[2]];
    signal tmp_14545[3] <== [tmp_14544[0] + evals[69][0], tmp_14544[1] + evals[69][1], tmp_14544[2] + evals[69][2]];
    signal tmp_14546[3] <== [tmp_14545[0] - evals[87][0], tmp_14545[1] - evals[87][1], tmp_14545[2] - evals[87][2]];
    signal tmp_14547[3] <== CMul()(evals[59], tmp_14546);
    signal tmp_14548[3] <== [tmp_14537[0] + tmp_14547[0], tmp_14537[1] + tmp_14547[1], tmp_14537[2] + tmp_14547[2]];
    signal tmp_14549[3] <== CMul()(challengeQ, tmp_14548);
    signal tmp_14550[3] <== [1 - evals[128][0], -evals[128][1], -evals[128][2]];
    signal tmp_14551[3] <== [1 - evals[129][0], -evals[129][1], -evals[129][2]];
    signal tmp_14552[3] <== CMul()(tmp_14550, tmp_14551);
    signal tmp_14553[3] <== [1 - evals[130][0], -evals[130][1], -evals[130][2]];
    signal tmp_14554[3] <== CMul()(tmp_14552, tmp_14553);
    signal tmp_14555[3] <== CMul()(evals[61], tmp_14554);
    signal tmp_14556[3] <== [evals[67][0] - evals[131][0], evals[67][1] - evals[131][1], evals[67][2] - evals[131][2]];
    signal tmp_14557[3] <== CMul()(tmp_14555, tmp_14556);
    signal tmp_14558[3] <== [tmp_14549[0] + tmp_14557[0], tmp_14549[1] + tmp_14557[1], tmp_14549[2] + tmp_14557[2]];
    signal tmp_14559[3] <== CMul()(challengeQ, tmp_14558);
    signal tmp_14561[3] <== [evals[68][0] - evals[132][0], evals[68][1] - evals[132][1], evals[68][2] - evals[132][2]];
    signal tmp_14562[3] <== CMul()(tmp_14555, tmp_14561);
    signal tmp_14563[3] <== [tmp_14559[0] + tmp_14562[0], tmp_14559[1] + tmp_14562[1], tmp_14559[2] + tmp_14562[2]];
    signal tmp_14564[3] <== CMul()(challengeQ, tmp_14563);
    signal tmp_14566[3] <== [evals[69][0] - evals[133][0], evals[69][1] - evals[133][1], evals[69][2] - evals[133][2]];
    signal tmp_14567[3] <== CMul()(tmp_14555, tmp_14566);
    signal tmp_14568[3] <== [tmp_14564[0] + tmp_14567[0], tmp_14564[1] + tmp_14567[1], tmp_14564[2] + tmp_14567[2]];
    signal tmp_14569[3] <== CMul()(challengeQ, tmp_14568);
    signal tmp_14570[3] <== CMul()(evals[128], tmp_14551);
    signal tmp_14571[3] <== CMul()(tmp_14570, tmp_14553);
    signal tmp_14572[3] <== CMul()(evals[61], tmp_14571);
    signal tmp_14573[3] <== [evals[70][0] - evals[131][0], evals[70][1] - evals[131][1], evals[70][2] - evals[131][2]];
    signal tmp_14574[3] <== CMul()(tmp_14572, tmp_14573);
    signal tmp_14575[3] <== [tmp_14569[0] + tmp_14574[0], tmp_14569[1] + tmp_14574[1], tmp_14569[2] + tmp_14574[2]];
    signal tmp_14576[3] <== CMul()(challengeQ, tmp_14575);
    signal tmp_14578[3] <== [evals[71][0] - evals[132][0], evals[71][1] - evals[132][1], evals[71][2] - evals[132][2]];
    signal tmp_14579[3] <== CMul()(tmp_14572, tmp_14578);
    signal tmp_14580[3] <== [tmp_14576[0] + tmp_14579[0], tmp_14576[1] + tmp_14579[1], tmp_14576[2] + tmp_14579[2]];
    signal tmp_14581[3] <== CMul()(challengeQ, tmp_14580);
    signal tmp_14583[3] <== [evals[72][0] - evals[133][0], evals[72][1] - evals[133][1], evals[72][2] - evals[133][2]];
    signal tmp_14584[3] <== CMul()(tmp_14572, tmp_14583);
    signal tmp_14585[3] <== [tmp_14581[0] + tmp_14584[0], tmp_14581[1] + tmp_14584[1], tmp_14581[2] + tmp_14584[2]];
    signal tmp_14586[3] <== CMul()(challengeQ, tmp_14585);
    signal tmp_14587[3] <== CMul()(tmp_14550, evals[129]);
    signal tmp_14588[3] <== CMul()(tmp_14587, tmp_14553);
    signal tmp_14589[3] <== CMul()(evals[61], tmp_14588);
    signal tmp_14590[3] <== [evals[73][0] - evals[131][0], evals[73][1] - evals[131][1], evals[73][2] - evals[131][2]];
    signal tmp_14591[3] <== CMul()(tmp_14589, tmp_14590);
    signal tmp_14592[3] <== [tmp_14586[0] + tmp_14591[0], tmp_14586[1] + tmp_14591[1], tmp_14586[2] + tmp_14591[2]];
    signal tmp_14593[3] <== CMul()(challengeQ, tmp_14592);
    signal tmp_14595[3] <== [evals[74][0] - evals[132][0], evals[74][1] - evals[132][1], evals[74][2] - evals[132][2]];
    signal tmp_14596[3] <== CMul()(tmp_14589, tmp_14595);
    signal tmp_14597[3] <== [tmp_14593[0] + tmp_14596[0], tmp_14593[1] + tmp_14596[1], tmp_14593[2] + tmp_14596[2]];
    signal tmp_14598[3] <== CMul()(challengeQ, tmp_14597);
    signal tmp_14600[3] <== [evals[75][0] - evals[133][0], evals[75][1] - evals[133][1], evals[75][2] - evals[133][2]];
    signal tmp_14601[3] <== CMul()(tmp_14589, tmp_14600);
    signal tmp_14602[3] <== [tmp_14598[0] + tmp_14601[0], tmp_14598[1] + tmp_14601[1], tmp_14598[2] + tmp_14601[2]];
    signal tmp_14603[3] <== CMul()(challengeQ, tmp_14602);
    signal tmp_14604[3] <== CMul()(evals[128], evals[129]);
    signal tmp_14605[3] <== CMul()(tmp_14604, tmp_14553);
    signal tmp_14606[3] <== CMul()(evals[61], tmp_14605);
    signal tmp_14607[3] <== [evals[76][0] - evals[131][0], evals[76][1] - evals[131][1], evals[76][2] - evals[131][2]];
    signal tmp_14608[3] <== CMul()(tmp_14606, tmp_14607);
    signal tmp_14609[3] <== [tmp_14603[0] + tmp_14608[0], tmp_14603[1] + tmp_14608[1], tmp_14603[2] + tmp_14608[2]];
    signal tmp_14610[3] <== CMul()(challengeQ, tmp_14609);
    signal tmp_14612[3] <== [evals[77][0] - evals[132][0], evals[77][1] - evals[132][1], evals[77][2] - evals[132][2]];
    signal tmp_14613[3] <== CMul()(tmp_14606, tmp_14612);
    signal tmp_14614[3] <== [tmp_14610[0] + tmp_14613[0], tmp_14610[1] + tmp_14613[1], tmp_14610[2] + tmp_14613[2]];
    signal tmp_14615[3] <== CMul()(challengeQ, tmp_14614);
    signal tmp_14617[3] <== [evals[78][0] - evals[133][0], evals[78][1] - evals[133][1], evals[78][2] - evals[133][2]];
    signal tmp_14618[3] <== CMul()(tmp_14606, tmp_14617);
    signal tmp_14619[3] <== [tmp_14615[0] + tmp_14618[0], tmp_14615[1] + tmp_14618[1], tmp_14615[2] + tmp_14618[2]];
    signal tmp_14620[3] <== CMul()(challengeQ, tmp_14619);
    signal tmp_14622[3] <== CMul()(tmp_14552, evals[130]);
    signal tmp_14623[3] <== CMul()(evals[61], tmp_14622);
    signal tmp_14624[3] <== [evals[79][0] - evals[131][0], evals[79][1] - evals[131][1], evals[79][2] - evals[131][2]];
    signal tmp_14625[3] <== CMul()(tmp_14623, tmp_14624);
    signal tmp_14626[3] <== [tmp_14620[0] + tmp_14625[0], tmp_14620[1] + tmp_14625[1], tmp_14620[2] + tmp_14625[2]];
    signal tmp_14627[3] <== CMul()(challengeQ, tmp_14626);
    signal tmp_14629[3] <== [evals[80][0] - evals[132][0], evals[80][1] - evals[132][1], evals[80][2] - evals[132][2]];
    signal tmp_14630[3] <== CMul()(tmp_14623, tmp_14629);
    signal tmp_14631[3] <== [tmp_14627[0] + tmp_14630[0], tmp_14627[1] + tmp_14630[1], tmp_14627[2] + tmp_14630[2]];
    signal tmp_14632[3] <== CMul()(challengeQ, tmp_14631);
    signal tmp_14634[3] <== [evals[81][0] - evals[133][0], evals[81][1] - evals[133][1], evals[81][2] - evals[133][2]];
    signal tmp_14635[3] <== CMul()(tmp_14623, tmp_14634);
    signal tmp_14636[3] <== [tmp_14632[0] + tmp_14635[0], tmp_14632[1] + tmp_14635[1], tmp_14632[2] + tmp_14635[2]];
    signal tmp_14637[3] <== CMul()(challengeQ, tmp_14636);
    signal tmp_14639[3] <== CMul()(tmp_14570, evals[130]);
    signal tmp_14640[3] <== CMul()(evals[61], tmp_14639);
    signal tmp_14641[3] <== [evals[82][0] - evals[131][0], evals[82][1] - evals[131][1], evals[82][2] - evals[131][2]];
    signal tmp_14642[3] <== CMul()(tmp_14640, tmp_14641);
    signal tmp_14643[3] <== [tmp_14637[0] + tmp_14642[0], tmp_14637[1] + tmp_14642[1], tmp_14637[2] + tmp_14642[2]];
    signal tmp_14644[3] <== CMul()(challengeQ, tmp_14643);
    signal tmp_14646[3] <== [evals[83][0] - evals[132][0], evals[83][1] - evals[132][1], evals[83][2] - evals[132][2]];
    signal tmp_14647[3] <== CMul()(tmp_14640, tmp_14646);
    signal tmp_14648[3] <== [tmp_14644[0] + tmp_14647[0], tmp_14644[1] + tmp_14647[1], tmp_14644[2] + tmp_14647[2]];
    signal tmp_14649[3] <== CMul()(challengeQ, tmp_14648);
    signal tmp_14651[3] <== [evals[84][0] - evals[133][0], evals[84][1] - evals[133][1], evals[84][2] - evals[133][2]];
    signal tmp_14652[3] <== CMul()(tmp_14640, tmp_14651);
    signal tmp_14653[3] <== [tmp_14649[0] + tmp_14652[0], tmp_14649[1] + tmp_14652[1], tmp_14649[2] + tmp_14652[2]];
    signal tmp_14654[3] <== CMul()(challengeQ, tmp_14653);
    signal tmp_14656[3] <== CMul()(tmp_14587, evals[130]);
    signal tmp_14657[3] <== CMul()(evals[61], tmp_14656);
    signal tmp_14658[3] <== [evals[85][0] - evals[131][0], evals[85][1] - evals[131][1], evals[85][2] - evals[131][2]];
    signal tmp_14659[3] <== CMul()(tmp_14657, tmp_14658);
    signal tmp_14660[3] <== [tmp_14654[0] + tmp_14659[0], tmp_14654[1] + tmp_14659[1], tmp_14654[2] + tmp_14659[2]];
    signal tmp_14661[3] <== CMul()(challengeQ, tmp_14660);
    signal tmp_14663[3] <== [evals[86][0] - evals[132][0], evals[86][1] - evals[132][1], evals[86][2] - evals[132][2]];
    signal tmp_14664[3] <== CMul()(tmp_14657, tmp_14663);
    signal tmp_14665[3] <== [tmp_14661[0] + tmp_14664[0], tmp_14661[1] + tmp_14664[1], tmp_14661[2] + tmp_14664[2]];
    signal tmp_14666[3] <== CMul()(challengeQ, tmp_14665);
    signal tmp_14668[3] <== [evals[87][0] - evals[133][0], evals[87][1] - evals[133][1], evals[87][2] - evals[133][2]];
    signal tmp_14669[3] <== CMul()(tmp_14657, tmp_14668);
    signal tmp_14670[3] <== [tmp_14666[0] + tmp_14669[0], tmp_14666[1] + tmp_14669[1], tmp_14666[2] + tmp_14669[2]];
    signal tmp_14671[3] <== CMul()(challengeQ, tmp_14670);
    signal tmp_14673[3] <== CMul()(tmp_14604, evals[130]);
    signal tmp_14674[3] <== CMul()(evals[61], tmp_14673);
    signal tmp_14675[3] <== [evals[88][0] - evals[131][0], evals[88][1] - evals[131][1], evals[88][2] - evals[131][2]];
    signal tmp_14676[3] <== CMul()(tmp_14674, tmp_14675);
    signal tmp_14677[3] <== [tmp_14671[0] + tmp_14676[0], tmp_14671[1] + tmp_14676[1], tmp_14671[2] + tmp_14676[2]];
    signal tmp_14678[3] <== CMul()(challengeQ, tmp_14677);
    signal tmp_14680[3] <== [evals[89][0] - evals[132][0], evals[89][1] - evals[132][1], evals[89][2] - evals[132][2]];
    signal tmp_14681[3] <== CMul()(tmp_14674, tmp_14680);
    signal tmp_14682[3] <== [tmp_14678[0] + tmp_14681[0], tmp_14678[1] + tmp_14681[1], tmp_14678[2] + tmp_14681[2]];
    signal tmp_14683[3] <== CMul()(challengeQ, tmp_14682);
    signal tmp_14685[3] <== [evals[90][0] - evals[133][0], evals[90][1] - evals[133][1], evals[90][2] - evals[133][2]];
    signal tmp_14686[3] <== CMul()(tmp_14674, tmp_14685);
    signal tmp_14687[3] <== [tmp_14683[0] + tmp_14686[0], tmp_14683[1] + tmp_14686[1], tmp_14683[2] + tmp_14686[2]];
    signal tmp_14688[3] <== CMul()(challengeQ, tmp_14687);
    signal tmp_14690[3] <== CMul()(evals[128], tmp_14550);
    signal tmp_14691[3] <== CMul()(evals[61], tmp_14690);
    signal tmp_14692[3] <== [tmp_14688[0] + tmp_14691[0], tmp_14688[1] + tmp_14691[1], tmp_14688[2] + tmp_14691[2]];
    signal tmp_14693[3] <== CMul()(challengeQ, tmp_14692);
    signal tmp_14695[3] <== CMul()(evals[129], tmp_14551);
    signal tmp_14696[3] <== CMul()(evals[61], tmp_14695);
    signal tmp_14697[3] <== [tmp_14693[0] + tmp_14696[0], tmp_14693[1] + tmp_14696[1], tmp_14693[2] + tmp_14696[2]];
    signal tmp_14698[3] <== CMul()(challengeQ, tmp_14697);
    signal tmp_14700[3] <== CMul()(evals[130], tmp_14553);
    signal tmp_14701[3] <== CMul()(evals[61], tmp_14700);
    signal tmp_14702[3] <== [tmp_14698[0] + tmp_14701[0], tmp_14698[1] + tmp_14701[1], tmp_14698[2] + tmp_14701[2]];
    signal tmp_14703[3] <== CMul()(challengeQ, tmp_14702);
    signal tmp_14704[3] <== [1 - evals[83][0], -evals[83][1], -evals[83][2]];
    signal tmp_14705[3] <== [1 - evals[84][0], -evals[84][1], -evals[84][2]];
    signal tmp_14706[3] <== CMul()(tmp_14704, tmp_14705);
    signal tmp_14707[3] <== CMul()(evals[62], tmp_14706);
    signal tmp_14708[3] <== [evals[67][0] - evals[85][0], evals[67][1] - evals[85][1], evals[67][2] - evals[85][2]];
    signal tmp_14709[3] <== CMul()(tmp_14707, tmp_14708);
    signal tmp_14710[3] <== [tmp_14703[0] + tmp_14709[0], tmp_14703[1] + tmp_14709[1], tmp_14703[2] + tmp_14709[2]];
    signal tmp_14711[3] <== CMul()(challengeQ, tmp_14710);
    signal tmp_14713[3] <== [evals[68][0] - evals[86][0], evals[68][1] - evals[86][1], evals[68][2] - evals[86][2]];
    signal tmp_14714[3] <== CMul()(tmp_14707, tmp_14713);
    signal tmp_14715[3] <== [tmp_14711[0] + tmp_14714[0], tmp_14711[1] + tmp_14714[1], tmp_14711[2] + tmp_14714[2]];
    signal tmp_14716[3] <== CMul()(challengeQ, tmp_14715);
    signal tmp_14718[3] <== [evals[69][0] - evals[87][0], evals[69][1] - evals[87][1], evals[69][2] - evals[87][2]];
    signal tmp_14719[3] <== CMul()(tmp_14707, tmp_14718);
    signal tmp_14720[3] <== [tmp_14716[0] + tmp_14719[0], tmp_14716[1] + tmp_14719[1], tmp_14716[2] + tmp_14719[2]];
    signal tmp_14721[3] <== CMul()(challengeQ, tmp_14720);
    signal tmp_14723[3] <== [evals[70][0] - evals[88][0], evals[70][1] - evals[88][1], evals[70][2] - evals[88][2]];
    signal tmp_14724[3] <== CMul()(tmp_14707, tmp_14723);
    signal tmp_14725[3] <== [tmp_14721[0] + tmp_14724[0], tmp_14721[1] + tmp_14724[1], tmp_14721[2] + tmp_14724[2]];
    signal tmp_14726[3] <== CMul()(challengeQ, tmp_14725);
    signal tmp_14728[3] <== CMul()(evals[83], tmp_14705);
    signal tmp_14729[3] <== CMul()(evals[62], tmp_14728);
    signal tmp_14730[3] <== [evals[71][0] - evals[85][0], evals[71][1] - evals[85][1], evals[71][2] - evals[85][2]];
    signal tmp_14731[3] <== CMul()(tmp_14729, tmp_14730);
    signal tmp_14732[3] <== [tmp_14726[0] + tmp_14731[0], tmp_14726[1] + tmp_14731[1], tmp_14726[2] + tmp_14731[2]];
    signal tmp_14733[3] <== CMul()(challengeQ, tmp_14732);
    signal tmp_14735[3] <== [evals[72][0] - evals[86][0], evals[72][1] - evals[86][1], evals[72][2] - evals[86][2]];
    signal tmp_14736[3] <== CMul()(tmp_14729, tmp_14735);
    signal tmp_14737[3] <== [tmp_14733[0] + tmp_14736[0], tmp_14733[1] + tmp_14736[1], tmp_14733[2] + tmp_14736[2]];
    signal tmp_14738[3] <== CMul()(challengeQ, tmp_14737);
    signal tmp_14740[3] <== [evals[73][0] - evals[87][0], evals[73][1] - evals[87][1], evals[73][2] - evals[87][2]];
    signal tmp_14741[3] <== CMul()(tmp_14729, tmp_14740);
    signal tmp_14742[3] <== [tmp_14738[0] + tmp_14741[0], tmp_14738[1] + tmp_14741[1], tmp_14738[2] + tmp_14741[2]];
    signal tmp_14743[3] <== CMul()(challengeQ, tmp_14742);
    signal tmp_14745[3] <== [evals[74][0] - evals[88][0], evals[74][1] - evals[88][1], evals[74][2] - evals[88][2]];
    signal tmp_14746[3] <== CMul()(tmp_14729, tmp_14745);
    signal tmp_14747[3] <== [tmp_14743[0] + tmp_14746[0], tmp_14743[1] + tmp_14746[1], tmp_14743[2] + tmp_14746[2]];
    signal tmp_14748[3] <== CMul()(challengeQ, tmp_14747);
    signal tmp_14750[3] <== CMul()(tmp_14704, evals[84]);
    signal tmp_14751[3] <== CMul()(evals[62], tmp_14750);
    signal tmp_14752[3] <== [evals[75][0] - evals[85][0], evals[75][1] - evals[85][1], evals[75][2] - evals[85][2]];
    signal tmp_14753[3] <== CMul()(tmp_14751, tmp_14752);
    signal tmp_14754[3] <== [tmp_14748[0] + tmp_14753[0], tmp_14748[1] + tmp_14753[1], tmp_14748[2] + tmp_14753[2]];
    signal tmp_14755[3] <== CMul()(challengeQ, tmp_14754);
    signal tmp_14757[3] <== [evals[76][0] - evals[86][0], evals[76][1] - evals[86][1], evals[76][2] - evals[86][2]];
    signal tmp_14758[3] <== CMul()(tmp_14751, tmp_14757);
    signal tmp_14759[3] <== [tmp_14755[0] + tmp_14758[0], tmp_14755[1] + tmp_14758[1], tmp_14755[2] + tmp_14758[2]];
    signal tmp_14760[3] <== CMul()(challengeQ, tmp_14759);
    signal tmp_14762[3] <== [evals[77][0] - evals[87][0], evals[77][1] - evals[87][1], evals[77][2] - evals[87][2]];
    signal tmp_14763[3] <== CMul()(tmp_14751, tmp_14762);
    signal tmp_14764[3] <== [tmp_14760[0] + tmp_14763[0], tmp_14760[1] + tmp_14763[1], tmp_14760[2] + tmp_14763[2]];
    signal tmp_14765[3] <== CMul()(challengeQ, tmp_14764);
    signal tmp_14767[3] <== [evals[78][0] - evals[88][0], evals[78][1] - evals[88][1], evals[78][2] - evals[88][2]];
    signal tmp_14768[3] <== CMul()(tmp_14751, tmp_14767);
    signal tmp_14769[3] <== [tmp_14765[0] + tmp_14768[0], tmp_14765[1] + tmp_14768[1], tmp_14765[2] + tmp_14768[2]];
    signal tmp_14770[3] <== CMul()(challengeQ, tmp_14769);
    signal tmp_14771[3] <== CMul()(evals[83], evals[84]);
    signal tmp_14772[3] <== CMul()(evals[62], tmp_14771);
    signal tmp_14773[3] <== [evals[79][0] - evals[85][0], evals[79][1] - evals[85][1], evals[79][2] - evals[85][2]];
    signal tmp_14774[3] <== CMul()(tmp_14772, tmp_14773);
    signal tmp_14775[3] <== [tmp_14770[0] + tmp_14774[0], tmp_14770[1] + tmp_14774[1], tmp_14770[2] + tmp_14774[2]];
    signal tmp_14776[3] <== CMul()(challengeQ, tmp_14775);
    signal tmp_14778[3] <== [evals[80][0] - evals[86][0], evals[80][1] - evals[86][1], evals[80][2] - evals[86][2]];
    signal tmp_14779[3] <== CMul()(tmp_14772, tmp_14778);
    signal tmp_14780[3] <== [tmp_14776[0] + tmp_14779[0], tmp_14776[1] + tmp_14779[1], tmp_14776[2] + tmp_14779[2]];
    signal tmp_14781[3] <== CMul()(challengeQ, tmp_14780);
    signal tmp_14783[3] <== [evals[81][0] - evals[87][0], evals[81][1] - evals[87][1], evals[81][2] - evals[87][2]];
    signal tmp_14784[3] <== CMul()(tmp_14772, tmp_14783);
    signal tmp_14785[3] <== [tmp_14781[0] + tmp_14784[0], tmp_14781[1] + tmp_14784[1], tmp_14781[2] + tmp_14784[2]];
    signal tmp_14786[3] <== CMul()(challengeQ, tmp_14785);
    signal tmp_14788[3] <== [evals[82][0] - evals[88][0], evals[82][1] - evals[88][1], evals[82][2] - evals[88][2]];
    signal tmp_14789[3] <== CMul()(tmp_14772, tmp_14788);
    signal tmp_14790[3] <== [tmp_14786[0] + tmp_14789[0], tmp_14786[1] + tmp_14789[1], tmp_14786[2] + tmp_14789[2]];
    signal tmp_14791[3] <== CMul()(challengeQ, tmp_14790);
    signal tmp_14793[3] <== CMul()(evals[83], tmp_14704);
    signal tmp_14794[3] <== CMul()(evals[62], tmp_14793);
    signal tmp_14795[3] <== [tmp_14791[0] + tmp_14794[0], tmp_14791[1] + tmp_14794[1], tmp_14791[2] + tmp_14794[2]];
    signal tmp_14796[3] <== CMul()(challengeQ, tmp_14795);
    signal tmp_14798[3] <== CMul()(evals[84], tmp_14705);
    signal tmp_14799[3] <== CMul()(evals[62], tmp_14798);
    signal tmp_14800[3] <== [tmp_14796[0] + tmp_14799[0], tmp_14796[1] + tmp_14799[1], tmp_14796[2] + tmp_14799[2]];
    signal tmp_14801[3] <== CMul()(challengeQ, tmp_14800);
    signal tmp_14802[3] <== CMul()(evals[65], challengesStage2[0]);
    signal tmp_14803[3] <== [tmp_14802[0] + evals[67][0], tmp_14802[1] + evals[67][1], tmp_14802[2] + evals[67][2]];
    signal tmp_14804[3] <== CMul()(tmp_14803, challengesStage2[0]);
    signal tmp_14805[3] <== [tmp_14804[0] + 1, tmp_14804[1], tmp_14804[2]];
    signal tmp_14806[3] <== [tmp_14805[0] + challengesStage2[1][0], tmp_14805[1] + challengesStage2[1][1], tmp_14805[2] + challengesStage2[1][2]];
    signal tmp_14807[3] <== [12275445934081160404 * evals[65][0], 12275445934081160404 * evals[65][1], 12275445934081160404 * evals[65][2]];
    signal tmp_14808[3] <== CMul()(tmp_14807, challengesStage2[0]);
    signal tmp_14809[3] <== [tmp_14808[0] + evals[68][0], tmp_14808[1] + evals[68][1], tmp_14808[2] + evals[68][2]];
    signal tmp_14810[3] <== CMul()(tmp_14809, challengesStage2[0]);
    signal tmp_14811[3] <== [tmp_14810[0] + 1, tmp_14810[1], tmp_14810[2]];
    signal tmp_14812[3] <== [tmp_14811[0] + challengesStage2[1][0], tmp_14811[1] + challengesStage2[1][1], tmp_14811[2] + challengesStage2[1][2]];
    signal tmp_14813[3] <== CMul()(tmp_14806, tmp_14812);
    signal tmp_14814[3] <== [4756475762779100925 * evals[65][0], 4756475762779100925 * evals[65][1], 4756475762779100925 * evals[65][2]];
    signal tmp_14815[3] <== CMul()(tmp_14814, challengesStage2[0]);
    signal tmp_14816[3] <== [tmp_14815[0] + evals[69][0], tmp_14815[1] + evals[69][1], tmp_14815[2] + evals[69][2]];
    signal tmp_14817[3] <== CMul()(tmp_14816, challengesStage2[0]);
    signal tmp_14818[3] <== [tmp_14817[0] + 1, tmp_14817[1], tmp_14817[2]];
    signal tmp_14819[3] <== [tmp_14818[0] + challengesStage2[1][0], tmp_14818[1] + challengesStage2[1][1], tmp_14818[2] + challengesStage2[1][2]];
    signal tmp_14820[3] <== CMul()(tmp_14813, tmp_14819);
    signal tmp_14821[3] <== [1279992132519201448 * evals[65][0], 1279992132519201448 * evals[65][1], 1279992132519201448 * evals[65][2]];
    signal tmp_14822[3] <== CMul()(tmp_14821, challengesStage2[0]);
    signal tmp_14823[3] <== [tmp_14822[0] + evals[70][0], tmp_14822[1] + evals[70][1], tmp_14822[2] + evals[70][2]];
    signal tmp_14824[3] <== CMul()(tmp_14823, challengesStage2[0]);
    signal tmp_14825[3] <== [tmp_14824[0] + 1, tmp_14824[1], tmp_14824[2]];
    signal tmp_14826[3] <== [tmp_14825[0] + challengesStage2[1][0], tmp_14825[1] + challengesStage2[1][1], tmp_14825[2] + challengesStage2[1][2]];
    signal tmp_14827[3] <== CMul()(tmp_14820, tmp_14826);
    signal tmp_14828[3] <== [8312008622371998338 * evals[65][0], 8312008622371998338 * evals[65][1], 8312008622371998338 * evals[65][2]];
    signal tmp_14829[3] <== CMul()(tmp_14828, challengesStage2[0]);
    signal tmp_14830[3] <== [tmp_14829[0] + evals[71][0], tmp_14829[1] + evals[71][1], tmp_14829[2] + evals[71][2]];
    signal tmp_14831[3] <== CMul()(tmp_14830, challengesStage2[0]);
    signal tmp_14832[3] <== [tmp_14831[0] + 1, tmp_14831[1], tmp_14831[2]];
    signal tmp_14833[3] <== [tmp_14832[0] + challengesStage2[1][0], tmp_14832[1] + challengesStage2[1][1], tmp_14832[2] + challengesStage2[1][2]];
    signal tmp_14834[3] <== CMul()(tmp_14827, tmp_14833);
    signal tmp_14835[3] <== [7781028390488215464 * evals[65][0], 7781028390488215464 * evals[65][1], 7781028390488215464 * evals[65][2]];
    signal tmp_14836[3] <== CMul()(tmp_14835, challengesStage2[0]);
    signal tmp_14837[3] <== [tmp_14836[0] + evals[72][0], tmp_14836[1] + evals[72][1], tmp_14836[2] + evals[72][2]];
    signal tmp_14838[3] <== CMul()(tmp_14837, challengesStage2[0]);
    signal tmp_14839[3] <== [tmp_14838[0] + 1, tmp_14838[1], tmp_14838[2]];
    signal tmp_14840[3] <== [tmp_14839[0] + challengesStage2[1][0], tmp_14839[1] + challengesStage2[1][1], tmp_14839[2] + challengesStage2[1][2]];
    signal tmp_14841[3] <== CMul()(tmp_14834, tmp_14840);
    signal tmp_14842[3] <== [11302600489504509467 * evals[65][0], 11302600489504509467 * evals[65][1], 11302600489504509467 * evals[65][2]];
    signal tmp_14843[3] <== CMul()(tmp_14842, challengesStage2[0]);
    signal tmp_14844[3] <== [tmp_14843[0] + evals[73][0], tmp_14843[1] + evals[73][1], tmp_14843[2] + evals[73][2]];
    signal tmp_14845[3] <== CMul()(tmp_14844, challengesStage2[0]);
    signal tmp_14846[3] <== [tmp_14845[0] + 1, tmp_14845[1], tmp_14845[2]];
    signal tmp_14847[3] <== [tmp_14846[0] + challengesStage2[1][0], tmp_14846[1] + challengesStage2[1][1], tmp_14846[2] + challengesStage2[1][2]];
    signal tmp_14848[3] <== CMul()(tmp_14841, tmp_14847);
    signal tmp_14849[3] <== CMul()(evals[116], tmp_14848);
    signal tmp_14850[3] <== CMul()(evals[21], challengesStage2[0]);
    signal tmp_14851[3] <== [tmp_14850[0] + evals[67][0], tmp_14850[1] + evals[67][1], tmp_14850[2] + evals[67][2]];
    signal tmp_14852[3] <== CMul()(tmp_14851, challengesStage2[0]);
    signal tmp_14853[3] <== [tmp_14852[0] + 1, tmp_14852[1], tmp_14852[2]];
    signal tmp_14854[3] <== [tmp_14853[0] + challengesStage2[1][0], tmp_14853[1] + challengesStage2[1][1], tmp_14853[2] + challengesStage2[1][2]];
    signal tmp_14855[3] <== CMul()(evals[22], challengesStage2[0]);
    signal tmp_14856[3] <== [tmp_14855[0] + evals[68][0], tmp_14855[1] + evals[68][1], tmp_14855[2] + evals[68][2]];
    signal tmp_14857[3] <== CMul()(tmp_14856, challengesStage2[0]);
    signal tmp_14858[3] <== [tmp_14857[0] + 1, tmp_14857[1], tmp_14857[2]];
    signal tmp_14859[3] <== [tmp_14858[0] + challengesStage2[1][0], tmp_14858[1] + challengesStage2[1][1], tmp_14858[2] + challengesStage2[1][2]];
    signal tmp_14860[3] <== CMul()(tmp_14854, tmp_14859);
    signal tmp_14861[3] <== CMul()(evals[23], challengesStage2[0]);
    signal tmp_14862[3] <== [tmp_14861[0] + evals[69][0], tmp_14861[1] + evals[69][1], tmp_14861[2] + evals[69][2]];
    signal tmp_14863[3] <== CMul()(tmp_14862, challengesStage2[0]);
    signal tmp_14864[3] <== [tmp_14863[0] + 1, tmp_14863[1], tmp_14863[2]];
    signal tmp_14865[3] <== [tmp_14864[0] + challengesStage2[1][0], tmp_14864[1] + challengesStage2[1][1], tmp_14864[2] + challengesStage2[1][2]];
    signal tmp_14866[3] <== CMul()(tmp_14860, tmp_14865);
    signal tmp_14867[3] <== CMul()(evals[24], challengesStage2[0]);
    signal tmp_14868[3] <== [tmp_14867[0] + evals[70][0], tmp_14867[1] + evals[70][1], tmp_14867[2] + evals[70][2]];
    signal tmp_14869[3] <== CMul()(tmp_14868, challengesStage2[0]);
    signal tmp_14870[3] <== [tmp_14869[0] + 1, tmp_14869[1], tmp_14869[2]];
    signal tmp_14871[3] <== [tmp_14870[0] + challengesStage2[1][0], tmp_14870[1] + challengesStage2[1][1], tmp_14870[2] + challengesStage2[1][2]];
    signal tmp_14872[3] <== CMul()(tmp_14866, tmp_14871);
    signal tmp_14873[3] <== CMul()(evals[25], challengesStage2[0]);
    signal tmp_14874[3] <== [tmp_14873[0] + evals[71][0], tmp_14873[1] + evals[71][1], tmp_14873[2] + evals[71][2]];
    signal tmp_14875[3] <== CMul()(tmp_14874, challengesStage2[0]);
    signal tmp_14876[3] <== [tmp_14875[0] + 1, tmp_14875[1], tmp_14875[2]];
    signal tmp_14877[3] <== [tmp_14876[0] + challengesStage2[1][0], tmp_14876[1] + challengesStage2[1][1], tmp_14876[2] + challengesStage2[1][2]];
    signal tmp_14878[3] <== CMul()(tmp_14872, tmp_14877);
    signal tmp_14879[3] <== CMul()(evals[26], challengesStage2[0]);
    signal tmp_14880[3] <== [tmp_14879[0] + evals[72][0], tmp_14879[1] + evals[72][1], tmp_14879[2] + evals[72][2]];
    signal tmp_14881[3] <== CMul()(tmp_14880, challengesStage2[0]);
    signal tmp_14882[3] <== [tmp_14881[0] + 1, tmp_14881[1], tmp_14881[2]];
    signal tmp_14883[3] <== [tmp_14882[0] + challengesStage2[1][0], tmp_14882[1] + challengesStage2[1][1], tmp_14882[2] + challengesStage2[1][2]];
    signal tmp_14884[3] <== CMul()(tmp_14878, tmp_14883);
    signal tmp_14885[3] <== CMul()(evals[27], challengesStage2[0]);
    signal tmp_14886[3] <== [tmp_14885[0] + evals[73][0], tmp_14885[1] + evals[73][1], tmp_14885[2] + evals[73][2]];
    signal tmp_14887[3] <== CMul()(tmp_14886, challengesStage2[0]);
    signal tmp_14888[3] <== [tmp_14887[0] + 1, tmp_14887[1], tmp_14887[2]];
    signal tmp_14889[3] <== [tmp_14888[0] + challengesStage2[1][0], tmp_14888[1] + challengesStage2[1][1], tmp_14888[2] + challengesStage2[1][2]];
    signal tmp_14890[3] <== CMul()(tmp_14884, tmp_14889);
    signal tmp_14891[3] <== CMul()(evals[28], challengesStage2[0]);
    signal tmp_14892[3] <== [tmp_14891[0] + evals[74][0], tmp_14891[1] + evals[74][1], tmp_14891[2] + evals[74][2]];
    signal tmp_14893[3] <== CMul()(tmp_14892, challengesStage2[0]);
    signal tmp_14894[3] <== [tmp_14893[0] + 1, tmp_14893[1], tmp_14893[2]];
    signal tmp_14895[3] <== [tmp_14894[0] + challengesStage2[1][0], tmp_14894[1] + challengesStage2[1][1], tmp_14894[2] + challengesStage2[1][2]];
    signal tmp_14896[3] <== CMul()(tmp_14890, tmp_14895);
    signal tmp_14897[3] <== [tmp_14849[0] - tmp_14896[0], tmp_14849[1] - tmp_14896[1], tmp_14849[2] - tmp_14896[2]];
    signal tmp_14898[3] <== [tmp_14801[0] + tmp_14897[0], tmp_14801[1] + tmp_14897[1], tmp_14801[2] + tmp_14897[2]];
    tmp_14899 <== CMul()(challengeQ, tmp_14898);
    signal tmp_14900[3] <== [4549350404001778198 * evals[65][0], 4549350404001778198 * evals[65][1], 4549350404001778198 * evals[65][2]];
    signal tmp_14901[3] <== CMul()(tmp_14900, challengesStage2[0]);
    signal tmp_14902[3] <== [tmp_14901[0] + evals[74][0], tmp_14901[1] + evals[74][1], tmp_14901[2] + evals[74][2]];
    signal tmp_14903[3] <== CMul()(tmp_14902, challengesStage2[0]);
    signal tmp_14904[3] <== [tmp_14903[0] + 1, tmp_14903[1], tmp_14903[2]];
    tmp_14905 <== [tmp_14904[0] + challengesStage2[1][0], tmp_14904[1] + challengesStage2[1][1], tmp_14904[2] + challengesStage2[1][2]];
    tmp_14906 <== [3688660304411827445 * evals[65][0], 3688660304411827445 * evals[65][1], 3688660304411827445 * evals[65][2]];
}

template VerifyEvaluationsChunks6_0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[152][3];
    signal input publics[399];

    signal input Zh[3];

    signal input tmp_14899[3];
    signal input tmp_14905[3];
    signal input tmp_14906[3];

    signal output tmp_15130[3];
    signal tmp_14907[3] <== CMul()(tmp_14906, challengesStage2[0]);
    signal tmp_14908[3] <== [tmp_14907[0] + evals[75][0], tmp_14907[1] + evals[75][1], tmp_14907[2] + evals[75][2]];
    signal tmp_14909[3] <== CMul()(tmp_14908, challengesStage2[0]);
    signal tmp_14910[3] <== [tmp_14909[0] + 1, tmp_14909[1], tmp_14909[2]];
    signal tmp_14911[3] <== [tmp_14910[0] + challengesStage2[1][0], tmp_14910[1] + challengesStage2[1][1], tmp_14910[2] + challengesStage2[1][2]];
    signal tmp_14912[3] <== CMul()(tmp_14905, tmp_14911);
    signal tmp_14913[3] <== [16725109960945739746 * evals[65][0], 16725109960945739746 * evals[65][1], 16725109960945739746 * evals[65][2]];
    signal tmp_14914[3] <== CMul()(tmp_14913, challengesStage2[0]);
    signal tmp_14915[3] <== [tmp_14914[0] + evals[76][0], tmp_14914[1] + evals[76][1], tmp_14914[2] + evals[76][2]];
    signal tmp_14916[3] <== CMul()(tmp_14915, challengesStage2[0]);
    signal tmp_14917[3] <== [tmp_14916[0] + 1, tmp_14916[1], tmp_14916[2]];
    signal tmp_14918[3] <== [tmp_14917[0] + challengesStage2[1][0], tmp_14917[1] + challengesStage2[1][1], tmp_14917[2] + challengesStage2[1][2]];
    signal tmp_14919[3] <== CMul()(tmp_14912, tmp_14918);
    signal tmp_14920[3] <== [16538725463549498621 * evals[65][0], 16538725463549498621 * evals[65][1], 16538725463549498621 * evals[65][2]];
    signal tmp_14921[3] <== CMul()(tmp_14920, challengesStage2[0]);
    signal tmp_14922[3] <== [tmp_14921[0] + evals[77][0], tmp_14921[1] + evals[77][1], tmp_14921[2] + evals[77][2]];
    signal tmp_14923[3] <== CMul()(tmp_14922, challengesStage2[0]);
    signal tmp_14924[3] <== [tmp_14923[0] + 1, tmp_14923[1], tmp_14923[2]];
    signal tmp_14925[3] <== [tmp_14924[0] + challengesStage2[1][0], tmp_14924[1] + challengesStage2[1][1], tmp_14924[2] + challengesStage2[1][2]];
    signal tmp_14926[3] <== CMul()(tmp_14919, tmp_14925);
    signal tmp_14927[3] <== [12756200801261202346 * evals[65][0], 12756200801261202346 * evals[65][1], 12756200801261202346 * evals[65][2]];
    signal tmp_14928[3] <== CMul()(tmp_14927, challengesStage2[0]);
    signal tmp_14929[3] <== [tmp_14928[0] + evals[78][0], tmp_14928[1] + evals[78][1], tmp_14928[2] + evals[78][2]];
    signal tmp_14930[3] <== CMul()(tmp_14929, challengesStage2[0]);
    signal tmp_14931[3] <== [tmp_14930[0] + 1, tmp_14930[1], tmp_14930[2]];
    signal tmp_14932[3] <== [tmp_14931[0] + challengesStage2[1][0], tmp_14931[1] + challengesStage2[1][1], tmp_14931[2] + challengesStage2[1][2]];
    signal tmp_14933[3] <== CMul()(tmp_14926, tmp_14932);
    signal tmp_14934[3] <== [15099809066790865939 * evals[65][0], 15099809066790865939 * evals[65][1], 15099809066790865939 * evals[65][2]];
    signal tmp_14935[3] <== CMul()(tmp_14934, challengesStage2[0]);
    signal tmp_14936[3] <== [tmp_14935[0] + evals[79][0], tmp_14935[1] + evals[79][1], tmp_14935[2] + evals[79][2]];
    signal tmp_14937[3] <== CMul()(tmp_14936, challengesStage2[0]);
    signal tmp_14938[3] <== [tmp_14937[0] + 1, tmp_14937[1], tmp_14937[2]];
    signal tmp_14939[3] <== [tmp_14938[0] + challengesStage2[1][0], tmp_14938[1] + challengesStage2[1][1], tmp_14938[2] + challengesStage2[1][2]];
    signal tmp_14940[3] <== CMul()(tmp_14933, tmp_14939);
    signal tmp_14941[3] <== [17214954929431464349 * evals[65][0], 17214954929431464349 * evals[65][1], 17214954929431464349 * evals[65][2]];
    signal tmp_14942[3] <== CMul()(tmp_14941, challengesStage2[0]);
    signal tmp_14943[3] <== [tmp_14942[0] + evals[80][0], tmp_14942[1] + evals[80][1], tmp_14942[2] + evals[80][2]];
    signal tmp_14944[3] <== CMul()(tmp_14943, challengesStage2[0]);
    signal tmp_14945[3] <== [tmp_14944[0] + 1, tmp_14944[1], tmp_14944[2]];
    signal tmp_14946[3] <== [tmp_14945[0] + challengesStage2[1][0], tmp_14945[1] + challengesStage2[1][1], tmp_14945[2] + challengesStage2[1][2]];
    signal tmp_14947[3] <== CMul()(tmp_14940, tmp_14946);
    signal tmp_14948[3] <== CMul()(evals[117], tmp_14947);
    signal tmp_14949[3] <== CMul()(evals[29], challengesStage2[0]);
    signal tmp_14950[3] <== [tmp_14949[0] + evals[75][0], tmp_14949[1] + evals[75][1], tmp_14949[2] + evals[75][2]];
    signal tmp_14951[3] <== CMul()(tmp_14950, challengesStage2[0]);
    signal tmp_14952[3] <== [tmp_14951[0] + 1, tmp_14951[1], tmp_14951[2]];
    signal tmp_14953[3] <== [tmp_14952[0] + challengesStage2[1][0], tmp_14952[1] + challengesStage2[1][1], tmp_14952[2] + challengesStage2[1][2]];
    signal tmp_14954[3] <== CMul()(evals[116], tmp_14953);
    signal tmp_14955[3] <== CMul()(evals[30], challengesStage2[0]);
    signal tmp_14956[3] <== [tmp_14955[0] + evals[76][0], tmp_14955[1] + evals[76][1], tmp_14955[2] + evals[76][2]];
    signal tmp_14957[3] <== CMul()(tmp_14956, challengesStage2[0]);
    signal tmp_14958[3] <== [tmp_14957[0] + 1, tmp_14957[1], tmp_14957[2]];
    signal tmp_14959[3] <== [tmp_14958[0] + challengesStage2[1][0], tmp_14958[1] + challengesStage2[1][1], tmp_14958[2] + challengesStage2[1][2]];
    signal tmp_14960[3] <== CMul()(tmp_14954, tmp_14959);
    signal tmp_14961[3] <== CMul()(evals[31], challengesStage2[0]);
    signal tmp_14962[3] <== [tmp_14961[0] + evals[77][0], tmp_14961[1] + evals[77][1], tmp_14961[2] + evals[77][2]];
    signal tmp_14963[3] <== CMul()(tmp_14962, challengesStage2[0]);
    signal tmp_14964[3] <== [tmp_14963[0] + 1, tmp_14963[1], tmp_14963[2]];
    signal tmp_14965[3] <== [tmp_14964[0] + challengesStage2[1][0], tmp_14964[1] + challengesStage2[1][1], tmp_14964[2] + challengesStage2[1][2]];
    signal tmp_14966[3] <== CMul()(tmp_14960, tmp_14965);
    signal tmp_14967[3] <== CMul()(evals[32], challengesStage2[0]);
    signal tmp_14968[3] <== [tmp_14967[0] + evals[78][0], tmp_14967[1] + evals[78][1], tmp_14967[2] + evals[78][2]];
    signal tmp_14969[3] <== CMul()(tmp_14968, challengesStage2[0]);
    signal tmp_14970[3] <== [tmp_14969[0] + 1, tmp_14969[1], tmp_14969[2]];
    signal tmp_14971[3] <== [tmp_14970[0] + challengesStage2[1][0], tmp_14970[1] + challengesStage2[1][1], tmp_14970[2] + challengesStage2[1][2]];
    signal tmp_14972[3] <== CMul()(tmp_14966, tmp_14971);
    signal tmp_14973[3] <== CMul()(evals[33], challengesStage2[0]);
    signal tmp_14974[3] <== [tmp_14973[0] + evals[79][0], tmp_14973[1] + evals[79][1], tmp_14973[2] + evals[79][2]];
    signal tmp_14975[3] <== CMul()(tmp_14974, challengesStage2[0]);
    signal tmp_14976[3] <== [tmp_14975[0] + 1, tmp_14975[1], tmp_14975[2]];
    signal tmp_14977[3] <== [tmp_14976[0] + challengesStage2[1][0], tmp_14976[1] + challengesStage2[1][1], tmp_14976[2] + challengesStage2[1][2]];
    signal tmp_14978[3] <== CMul()(tmp_14972, tmp_14977);
    signal tmp_14979[3] <== CMul()(evals[34], challengesStage2[0]);
    signal tmp_14980[3] <== [tmp_14979[0] + evals[80][0], tmp_14979[1] + evals[80][1], tmp_14979[2] + evals[80][2]];
    signal tmp_14981[3] <== CMul()(tmp_14980, challengesStage2[0]);
    signal tmp_14982[3] <== [tmp_14981[0] + 1, tmp_14981[1], tmp_14981[2]];
    signal tmp_14983[3] <== [tmp_14982[0] + challengesStage2[1][0], tmp_14982[1] + challengesStage2[1][1], tmp_14982[2] + challengesStage2[1][2]];
    signal tmp_14984[3] <== CMul()(tmp_14978, tmp_14983);
    signal tmp_14985[3] <== CMul()(evals[35], challengesStage2[0]);
    signal tmp_14986[3] <== [tmp_14985[0] + evals[81][0], tmp_14985[1] + evals[81][1], tmp_14985[2] + evals[81][2]];
    signal tmp_14987[3] <== CMul()(tmp_14986, challengesStage2[0]);
    signal tmp_14988[3] <== [tmp_14987[0] + 1, tmp_14987[1], tmp_14987[2]];
    signal tmp_14989[3] <== [tmp_14988[0] + challengesStage2[1][0], tmp_14988[1] + challengesStage2[1][1], tmp_14988[2] + challengesStage2[1][2]];
    signal tmp_14990[3] <== CMul()(tmp_14984, tmp_14989);
    signal tmp_14991[3] <== [tmp_14948[0] - tmp_14990[0], tmp_14948[1] - tmp_14990[1], tmp_14948[2] - tmp_14990[2]];
    signal tmp_14992[3] <== [tmp_14899[0] + tmp_14991[0], tmp_14899[1] + tmp_14991[1], tmp_14899[2] + tmp_14991[2]];
    signal tmp_14993[3] <== CMul()(challengeQ, tmp_14992);
    signal tmp_14994[3] <== [11016800570561344835 * evals[65][0], 11016800570561344835 * evals[65][1], 11016800570561344835 * evals[65][2]];
    signal tmp_14995[3] <== CMul()(tmp_14994, challengesStage2[0]);
    signal tmp_14996[3] <== [tmp_14995[0] + evals[81][0], tmp_14995[1] + evals[81][1], tmp_14995[2] + evals[81][2]];
    signal tmp_14997[3] <== CMul()(tmp_14996, challengesStage2[0]);
    signal tmp_14998[3] <== [tmp_14997[0] + 1, tmp_14997[1], tmp_14997[2]];
    signal tmp_14999[3] <== [tmp_14998[0] + challengesStage2[1][0], tmp_14998[1] + challengesStage2[1][1], tmp_14998[2] + challengesStage2[1][2]];
    signal tmp_15000[3] <== [11274872323250451096 * evals[65][0], 11274872323250451096 * evals[65][1], 11274872323250451096 * evals[65][2]];
    signal tmp_15001[3] <== CMul()(tmp_15000, challengesStage2[0]);
    signal tmp_15002[3] <== [tmp_15001[0] + evals[82][0], tmp_15001[1] + evals[82][1], tmp_15001[2] + evals[82][2]];
    signal tmp_15003[3] <== CMul()(tmp_15002, challengesStage2[0]);
    signal tmp_15004[3] <== [tmp_15003[0] + 1, tmp_15003[1], tmp_15003[2]];
    signal tmp_15005[3] <== [tmp_15004[0] + challengesStage2[1][0], tmp_15004[1] + challengesStage2[1][1], tmp_15004[2] + challengesStage2[1][2]];
    signal tmp_15006[3] <== CMul()(tmp_14999, tmp_15005);
    signal tmp_15007[3] <== [6534114114080170934 * evals[65][0], 6534114114080170934 * evals[65][1], 6534114114080170934 * evals[65][2]];
    signal tmp_15008[3] <== CMul()(tmp_15007, challengesStage2[0]);
    signal tmp_15009[3] <== [tmp_15008[0] + evals[83][0], tmp_15008[1] + evals[83][1], tmp_15008[2] + evals[83][2]];
    signal tmp_15010[3] <== CMul()(tmp_15009, challengesStage2[0]);
    signal tmp_15011[3] <== [tmp_15010[0] + 1, tmp_15010[1], tmp_15010[2]];
    signal tmp_15012[3] <== [tmp_15011[0] + challengesStage2[1][0], tmp_15011[1] + challengesStage2[1][1], tmp_15011[2] + challengesStage2[1][2]];
    signal tmp_15013[3] <== CMul()(tmp_15006, tmp_15012);
    signal tmp_15014[3] <== [13047390008333835222 * evals[65][0], 13047390008333835222 * evals[65][1], 13047390008333835222 * evals[65][2]];
    signal tmp_15015[3] <== CMul()(tmp_15014, challengesStage2[0]);
    signal tmp_15016[3] <== [tmp_15015[0] + evals[84][0], tmp_15015[1] + evals[84][1], tmp_15015[2] + evals[84][2]];
    signal tmp_15017[3] <== CMul()(tmp_15016, challengesStage2[0]);
    signal tmp_15018[3] <== [tmp_15017[0] + 1, tmp_15017[1], tmp_15017[2]];
    signal tmp_15019[3] <== [tmp_15018[0] + challengesStage2[1][0], tmp_15018[1] + challengesStage2[1][1], tmp_15018[2] + challengesStage2[1][2]];
    signal tmp_15020[3] <== CMul()(tmp_15013, tmp_15019);
    signal tmp_15021[3] <== [11189528522318044176 * evals[65][0], 11189528522318044176 * evals[65][1], 11189528522318044176 * evals[65][2]];
    signal tmp_15022[3] <== CMul()(tmp_15021, challengesStage2[0]);
    signal tmp_15023[3] <== [tmp_15022[0] + evals[85][0], tmp_15022[1] + evals[85][1], tmp_15022[2] + evals[85][2]];
    signal tmp_15024[3] <== CMul()(tmp_15023, challengesStage2[0]);
    signal tmp_15025[3] <== [tmp_15024[0] + 1, tmp_15024[1], tmp_15024[2]];
    signal tmp_15026[3] <== [tmp_15025[0] + challengesStage2[1][0], tmp_15025[1] + challengesStage2[1][1], tmp_15025[2] + challengesStage2[1][2]];
    signal tmp_15027[3] <== CMul()(tmp_15020, tmp_15026);
    signal tmp_15028[3] <== [3320735505586735876 * evals[65][0], 3320735505586735876 * evals[65][1], 3320735505586735876 * evals[65][2]];
    signal tmp_15029[3] <== CMul()(tmp_15028, challengesStage2[0]);
    signal tmp_15030[3] <== [tmp_15029[0] + evals[86][0], tmp_15029[1] + evals[86][1], tmp_15029[2] + evals[86][2]];
    signal tmp_15031[3] <== CMul()(tmp_15030, challengesStage2[0]);
    signal tmp_15032[3] <== [tmp_15031[0] + 1, tmp_15031[1], tmp_15031[2]];
    signal tmp_15033[3] <== [tmp_15032[0] + challengesStage2[1][0], tmp_15032[1] + challengesStage2[1][1], tmp_15032[2] + challengesStage2[1][2]];
    signal tmp_15034[3] <== CMul()(tmp_15027, tmp_15033);
    signal tmp_15035[3] <== [7240278926970958133 * evals[65][0], 7240278926970958133 * evals[65][1], 7240278926970958133 * evals[65][2]];
    signal tmp_15036[3] <== CMul()(tmp_15035, challengesStage2[0]);
    signal tmp_15037[3] <== [tmp_15036[0] + evals[87][0], tmp_15036[1] + evals[87][1], tmp_15036[2] + evals[87][2]];
    signal tmp_15038[3] <== CMul()(tmp_15037, challengesStage2[0]);
    signal tmp_15039[3] <== [tmp_15038[0] + 1, tmp_15038[1], tmp_15038[2]];
    signal tmp_15040[3] <== [tmp_15039[0] + challengesStage2[1][0], tmp_15039[1] + challengesStage2[1][1], tmp_15039[2] + challengesStage2[1][2]];
    signal tmp_15041[3] <== CMul()(tmp_15034, tmp_15040);
    signal tmp_15042[3] <== CMul()(evals[118], tmp_15041);
    signal tmp_15043[3] <== CMul()(evals[36], challengesStage2[0]);
    signal tmp_15044[3] <== [tmp_15043[0] + evals[82][0], tmp_15043[1] + evals[82][1], tmp_15043[2] + evals[82][2]];
    signal tmp_15045[3] <== CMul()(tmp_15044, challengesStage2[0]);
    signal tmp_15046[3] <== [tmp_15045[0] + 1, tmp_15045[1], tmp_15045[2]];
    signal tmp_15047[3] <== [tmp_15046[0] + challengesStage2[1][0], tmp_15046[1] + challengesStage2[1][1], tmp_15046[2] + challengesStage2[1][2]];
    signal tmp_15048[3] <== CMul()(evals[117], tmp_15047);
    signal tmp_15049[3] <== CMul()(evals[37], challengesStage2[0]);
    signal tmp_15050[3] <== [tmp_15049[0] + evals[83][0], tmp_15049[1] + evals[83][1], tmp_15049[2] + evals[83][2]];
    signal tmp_15051[3] <== CMul()(tmp_15050, challengesStage2[0]);
    signal tmp_15052[3] <== [tmp_15051[0] + 1, tmp_15051[1], tmp_15051[2]];
    signal tmp_15053[3] <== [tmp_15052[0] + challengesStage2[1][0], tmp_15052[1] + challengesStage2[1][1], tmp_15052[2] + challengesStage2[1][2]];
    signal tmp_15054[3] <== CMul()(tmp_15048, tmp_15053);
    signal tmp_15055[3] <== CMul()(evals[38], challengesStage2[0]);
    signal tmp_15056[3] <== [tmp_15055[0] + evals[84][0], tmp_15055[1] + evals[84][1], tmp_15055[2] + evals[84][2]];
    signal tmp_15057[3] <== CMul()(tmp_15056, challengesStage2[0]);
    signal tmp_15058[3] <== [tmp_15057[0] + 1, tmp_15057[1], tmp_15057[2]];
    signal tmp_15059[3] <== [tmp_15058[0] + challengesStage2[1][0], tmp_15058[1] + challengesStage2[1][1], tmp_15058[2] + challengesStage2[1][2]];
    signal tmp_15060[3] <== CMul()(tmp_15054, tmp_15059);
    signal tmp_15061[3] <== CMul()(evals[39], challengesStage2[0]);
    signal tmp_15062[3] <== [tmp_15061[0] + evals[85][0], tmp_15061[1] + evals[85][1], tmp_15061[2] + evals[85][2]];
    signal tmp_15063[3] <== CMul()(tmp_15062, challengesStage2[0]);
    signal tmp_15064[3] <== [tmp_15063[0] + 1, tmp_15063[1], tmp_15063[2]];
    signal tmp_15065[3] <== [tmp_15064[0] + challengesStage2[1][0], tmp_15064[1] + challengesStage2[1][1], tmp_15064[2] + challengesStage2[1][2]];
    signal tmp_15066[3] <== CMul()(tmp_15060, tmp_15065);
    signal tmp_15067[3] <== CMul()(evals[40], challengesStage2[0]);
    signal tmp_15068[3] <== [tmp_15067[0] + evals[86][0], tmp_15067[1] + evals[86][1], tmp_15067[2] + evals[86][2]];
    signal tmp_15069[3] <== CMul()(tmp_15068, challengesStage2[0]);
    signal tmp_15070[3] <== [tmp_15069[0] + 1, tmp_15069[1], tmp_15069[2]];
    signal tmp_15071[3] <== [tmp_15070[0] + challengesStage2[1][0], tmp_15070[1] + challengesStage2[1][1], tmp_15070[2] + challengesStage2[1][2]];
    signal tmp_15072[3] <== CMul()(tmp_15066, tmp_15071);
    signal tmp_15073[3] <== CMul()(evals[41], challengesStage2[0]);
    signal tmp_15074[3] <== [tmp_15073[0] + evals[87][0], tmp_15073[1] + evals[87][1], tmp_15073[2] + evals[87][2]];
    signal tmp_15075[3] <== CMul()(tmp_15074, challengesStage2[0]);
    signal tmp_15076[3] <== [tmp_15075[0] + 1, tmp_15075[1], tmp_15075[2]];
    signal tmp_15077[3] <== [tmp_15076[0] + challengesStage2[1][0], tmp_15076[1] + challengesStage2[1][1], tmp_15076[2] + challengesStage2[1][2]];
    signal tmp_15078[3] <== CMul()(tmp_15072, tmp_15077);
    signal tmp_15079[3] <== CMul()(evals[42], challengesStage2[0]);
    signal tmp_15080[3] <== [tmp_15079[0] + evals[88][0], tmp_15079[1] + evals[88][1], tmp_15079[2] + evals[88][2]];
    signal tmp_15081[3] <== CMul()(tmp_15080, challengesStage2[0]);
    signal tmp_15082[3] <== [tmp_15081[0] + 1, tmp_15081[1], tmp_15081[2]];
    signal tmp_15083[3] <== [tmp_15082[0] + challengesStage2[1][0], tmp_15082[1] + challengesStage2[1][1], tmp_15082[2] + challengesStage2[1][2]];
    signal tmp_15084[3] <== CMul()(tmp_15078, tmp_15083);
    signal tmp_15085[3] <== [tmp_15042[0] - tmp_15084[0], tmp_15042[1] - tmp_15084[1], tmp_15042[2] - tmp_15084[2]];
    signal tmp_15086[3] <== [tmp_14993[0] + tmp_15085[0], tmp_14993[1] + tmp_15085[1], tmp_14993[2] + tmp_15085[2]];
    signal tmp_15087[3] <== CMul()(challengeQ, tmp_15086);
    signal tmp_15088[3] <== [8246665031048405574 * evals[65][0], 8246665031048405574 * evals[65][1], 8246665031048405574 * evals[65][2]];
    signal tmp_15089[3] <== CMul()(tmp_15088, challengesStage2[0]);
    signal tmp_15090[3] <== [tmp_15089[0] + evals[88][0], tmp_15089[1] + evals[88][1], tmp_15089[2] + evals[88][2]];
    signal tmp_15091[3] <== CMul()(tmp_15090, challengesStage2[0]);
    signal tmp_15092[3] <== [tmp_15091[0] + 1, tmp_15091[1], tmp_15091[2]];
    signal tmp_15093[3] <== [tmp_15092[0] + challengesStage2[1][0], tmp_15092[1] + challengesStage2[1][1], tmp_15092[2] + challengesStage2[1][2]];
    signal tmp_15094[3] <== [12693612801792047873 * evals[65][0], 12693612801792047873 * evals[65][1], 12693612801792047873 * evals[65][2]];
    signal tmp_15095[3] <== CMul()(tmp_15094, challengesStage2[0]);
    signal tmp_15096[3] <== [tmp_15095[0] + evals[89][0], tmp_15095[1] + evals[89][1], tmp_15095[2] + evals[89][2]];
    signal tmp_15097[3] <== CMul()(tmp_15096, challengesStage2[0]);
    signal tmp_15098[3] <== [tmp_15097[0] + 1, tmp_15097[1], tmp_15097[2]];
    signal tmp_15099[3] <== [tmp_15098[0] + challengesStage2[1][0], tmp_15098[1] + challengesStage2[1][1], tmp_15098[2] + challengesStage2[1][2]];
    signal tmp_15100[3] <== CMul()(tmp_15093, tmp_15099);
    signal tmp_15101[3] <== [9404062091095256088 * evals[65][0], 9404062091095256088 * evals[65][1], 9404062091095256088 * evals[65][2]];
    signal tmp_15102[3] <== CMul()(tmp_15101, challengesStage2[0]);
    signal tmp_15103[3] <== [tmp_15102[0] + evals[90][0], tmp_15102[1] + evals[90][1], tmp_15102[2] + evals[90][2]];
    signal tmp_15104[3] <== CMul()(tmp_15103, challengesStage2[0]);
    signal tmp_15105[3] <== [tmp_15104[0] + 1, tmp_15104[1], tmp_15104[2]];
    signal tmp_15106[3] <== [tmp_15105[0] + challengesStage2[1][0], tmp_15105[1] + challengesStage2[1][1], tmp_15105[2] + challengesStage2[1][2]];
    signal tmp_15107[3] <== CMul()(tmp_15100, tmp_15106);
    signal tmp_15108[3] <== CMul()(evals[115], tmp_15107);
    signal tmp_15109[3] <== [1 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_15110[3] <== CMul()(evals[20], tmp_15109);
    signal tmp_15111[3] <== [tmp_15110[0] + evals[66][0], tmp_15110[1] + evals[66][1], tmp_15110[2] + evals[66][2]];
    signal tmp_15112[3] <== CMul()(evals[43], challengesStage2[0]);
    signal tmp_15113[3] <== [tmp_15112[0] + evals[89][0], tmp_15112[1] + evals[89][1], tmp_15112[2] + evals[89][2]];
    signal tmp_15114[3] <== CMul()(tmp_15113, challengesStage2[0]);
    signal tmp_15115[3] <== [tmp_15114[0] + 1, tmp_15114[1], tmp_15114[2]];
    signal tmp_15116[3] <== [tmp_15115[0] + challengesStage2[1][0], tmp_15115[1] + challengesStage2[1][1], tmp_15115[2] + challengesStage2[1][2]];
    signal tmp_15117[3] <== CMul()(evals[118], tmp_15116);
    signal tmp_15118[3] <== CMul()(evals[44], challengesStage2[0]);
    signal tmp_15119[3] <== [tmp_15118[0] + evals[90][0], tmp_15118[1] + evals[90][1], tmp_15118[2] + evals[90][2]];
    signal tmp_15120[3] <== CMul()(tmp_15119, challengesStage2[0]);
    signal tmp_15121[3] <== [tmp_15120[0] + 1, tmp_15120[1], tmp_15120[2]];
    signal tmp_15122[3] <== [tmp_15121[0] + challengesStage2[1][0], tmp_15121[1] + challengesStage2[1][1], tmp_15121[2] + challengesStage2[1][2]];
    signal tmp_15123[3] <== CMul()(tmp_15117, tmp_15122);
    signal tmp_15124[3] <== CMul()(tmp_15111, tmp_15123);
    signal tmp_15125[3] <== [tmp_15108[0] - tmp_15124[0], tmp_15108[1] - tmp_15124[1], tmp_15108[2] - tmp_15124[2]];
    signal tmp_15126[3] <== [tmp_15087[0] + tmp_15125[0], tmp_15087[1] + tmp_15125[1], tmp_15087[2] + tmp_15125[2]];
    signal tmp_15127[3] <== CMul()(challengeQ, tmp_15126);
    signal tmp_15128[3] <== [1 - evals[115][0], -evals[115][1], -evals[115][2]];
    signal tmp_15129[3] <== CMul()(evals[127], tmp_15128);
    signal tmp_7564[3] <== [tmp_15127[0] + tmp_15129[0], tmp_15127[1] + tmp_15129[1], tmp_15127[2] + tmp_15129[2]];
    tmp_15130 <== CMul()(tmp_7564, Zh);
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
    signal input evals[152][3];
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
    signal tmp_8564[3];
    (tmp_8564) <== VerifyEvaluationsChunks0_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh);
    signal tmp_9161[3];
    signal tmp_9163[3];
    signal tmp_9180[3];
    signal tmp_9192[3];
    signal tmp_9210[3];
    signal tmp_9500[3];
    signal tmp_9501[3];
    signal tmp_9504[3];
    signal tmp_9518[3];
    signal tmp_9532[3];
    signal tmp_9547[3];
    signal tmp_9563[3];
    signal tmp_9578[3];
    signal tmp_9580[3];
    signal tmp_9586[3];
    (tmp_9161,tmp_9163,tmp_9180,tmp_9192,tmp_9210,tmp_9500,tmp_9501,tmp_9504,tmp_9518,tmp_9532,tmp_9547,tmp_9563,tmp_9578,tmp_9580,tmp_9586) <== VerifyEvaluationsChunks1_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_8564);
    signal tmp_10511[3];
    signal tmp_10513[3];
    signal tmp_10519[3];
    signal tmp_10525[3];
    signal tmp_10529[3];
    signal tmp_10535[3];
    signal tmp_10540[3];
    signal tmp_10546[3];
    signal tmp_10552[3];
    signal tmp_10558[3];
    signal tmp_10563[3];
    signal tmp_10569[3];
    signal tmp_10575[3];
    signal tmp_10581[3];
    signal tmp_10587[3];
    signal tmp_10593[3];
    signal tmp_10594[3];
    signal tmp_10599[3];
    signal tmp_10605[3];
    signal tmp_10610[3];
    signal tmp_10616[3];
    signal tmp_10622[3];
    signal tmp_10628[3];
    signal tmp_10634[3];
    signal tmp_10640[3];
    signal tmp_10646[3];
    signal tmp_10652[3];
    signal tmp_10658[3];
    signal tmp_10664[3];
    signal tmp_10670[3];
    signal tmp_10676[3];
    signal tmp_10682[3];
    signal tmp_10688[3];
    signal tmp_10694[3];
    signal tmp_10700[3];
    signal tmp_10702[3];
    signal tmp_10731[3];
    signal tmp_10745[3];
    signal tmp_10787[3];
    signal tmp_10801[3];
    signal tmp_10843[3];
    signal tmp_10857[3];
    signal tmp_10899[3];
    signal tmp_10913[3];
    signal tmp_10926[3];
    signal tmp_10971[3];
    signal tmp_10977[3];
    signal tmp_10981[3];
    signal tmp_10988[3];
    signal tmp_10990[3];
    (tmp_10511,tmp_10513,tmp_10519,tmp_10525,tmp_10529,tmp_10535,tmp_10540,tmp_10546,tmp_10552,tmp_10558,tmp_10563,tmp_10569,tmp_10575,tmp_10581,tmp_10587,tmp_10593,tmp_10594,tmp_10599,tmp_10605,tmp_10610,tmp_10616,tmp_10622,tmp_10628,tmp_10634,tmp_10640,tmp_10646,tmp_10652,tmp_10658,tmp_10664,tmp_10670,tmp_10676,tmp_10682,tmp_10688,tmp_10694,tmp_10700,tmp_10702,tmp_10731,tmp_10745,tmp_10787,tmp_10801,tmp_10843,tmp_10857,tmp_10899,tmp_10913,tmp_10926,tmp_10971,tmp_10977,tmp_10981,tmp_10988,tmp_10990) <== VerifyEvaluationsChunks2_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_9161,tmp_9500,tmp_9501,tmp_9504,tmp_9518,tmp_9532,tmp_9547,tmp_9563,tmp_9578,tmp_9580,tmp_9586);
    signal tmp_12354[3];
    signal tmp_12417[3];
    signal tmp_12422[3];
    signal tmp_12427[3];
    signal tmp_12434[3];
    signal tmp_12435[3];
    signal tmp_12438[3];
    signal tmp_12443[3];
    signal tmp_12448[3];
    signal tmp_12453[3];
    signal tmp_12458[3];
    signal tmp_12463[3];
    signal tmp_12468[3];
    signal tmp_12473[3];
    signal tmp_12478[3];
    signal tmp_12483[3];
    signal tmp_12488[3];
    signal tmp_12493[3];
    signal tmp_12495[3];
    signal tmp_12497[3];
    (tmp_12354,tmp_12417,tmp_12422,tmp_12427,tmp_12434,tmp_12435,tmp_12438,tmp_12443,tmp_12448,tmp_12453,tmp_12458,tmp_12463,tmp_12468,tmp_12473,tmp_12478,tmp_12483,tmp_12488,tmp_12493,tmp_12495,tmp_12497) <== VerifyEvaluationsChunks3_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_9161,tmp_9163,tmp_10511,tmp_10513,tmp_10519,tmp_10525,tmp_10529,tmp_10535,tmp_10540,tmp_10546,tmp_10558,tmp_10569,tmp_10581,tmp_10593,tmp_10594,tmp_10605,tmp_10616,tmp_10628,tmp_10640,tmp_10652,tmp_10664,tmp_10676,tmp_10688,tmp_10700,tmp_10702,tmp_10731,tmp_10745,tmp_10787,tmp_10801,tmp_10843,tmp_10857,tmp_10899,tmp_10913,tmp_10926,tmp_10971,tmp_10977,tmp_10981,tmp_10988,tmp_10990);
    signal tmp_13573[3];
    signal tmp_13641[3];
    signal tmp_13646[3];
    signal tmp_13653[3];
    signal tmp_13658[3];
    signal tmp_13661[3];
    signal tmp_13666[3];
    signal tmp_13671[3];
    signal tmp_13676[3];
    signal tmp_13681[3];
    signal tmp_13686[3];
    signal tmp_13691[3];
    signal tmp_13696[3];
    signal tmp_13701[3];
    signal tmp_13706[3];
    signal tmp_13711[3];
    signal tmp_13716[3];
    signal tmp_13718[3];
    signal tmp_13721[3];
    (tmp_13573,tmp_13641,tmp_13646,tmp_13653,tmp_13658,tmp_13661,tmp_13666,tmp_13671,tmp_13676,tmp_13681,tmp_13686,tmp_13691,tmp_13696,tmp_13701,tmp_13706,tmp_13711,tmp_13716,tmp_13718,tmp_13721) <== VerifyEvaluationsChunks4_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_9163,tmp_10552,tmp_10563,tmp_10575,tmp_10587,tmp_10599,tmp_10610,tmp_10622,tmp_10634,tmp_10646,tmp_10658,tmp_10670,tmp_10682,tmp_10694,tmp_12354,tmp_12417,tmp_12422,tmp_12427,tmp_12434,tmp_12435,tmp_12438,tmp_12443,tmp_12448,tmp_12453,tmp_12458,tmp_12463,tmp_12468,tmp_12473,tmp_12478,tmp_12483,tmp_12488,tmp_12493,tmp_12495,tmp_12497);
    signal tmp_14899[3];
    signal tmp_14905[3];
    signal tmp_14906[3];
    (tmp_14899,tmp_14905,tmp_14906) <== VerifyEvaluationsChunks5_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_9163,tmp_9180,tmp_9192,tmp_9210,tmp_13573,tmp_13641,tmp_13646,tmp_13653,tmp_13658,tmp_13661,tmp_13666,tmp_13671,tmp_13676,tmp_13681,tmp_13686,tmp_13691,tmp_13696,tmp_13701,tmp_13706,tmp_13711,tmp_13716,tmp_13718,tmp_13721);
    signal tmp_15130[3];
    (tmp_15130) <== VerifyEvaluationsChunks6_0()(challengesStage2,challengeQ,challengeOut,evals,publics,Zh,tmp_14899,tmp_14905,tmp_14906);

    signal xAcc[7][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[6][3]; // Stores the evaluations of Q_i
    signal qAcc[7][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i = 0; i < 7; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[119+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[16]);
            qStep[i-1] <== CMul()(xAcc[i], evals[119+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i
    enable * (tmp_15130[0] - qAcc[6][0]) === 0;
    enable * (tmp_15130[1] - qAcc[6][1]) === 0;
    enable * (tmp_15130[2] - qAcc[6][2]) === 0;
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
    challengesDeep[0] = vf1; tmp_823 is f_0(x).

    The DAG is split into 1 chunk(s) by the same compile-cost device as
    VerifyEvaluationsChunks*, with tmp_ signals crossing the boundaries; each chunk builds its
    own `mapValues`, since the leaf values feed the whole DAG.
*/
template CalculateDeepPolChunks0_0() {
    signal input challengesDeep[2][3];
    signal input evals[152][3];

    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[46];

    signal input xDivXSubXi[19][3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
    mapValues.vals1 <== cm1;
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;

    signal input tmp_680[3];
    signal input tmp_681[3];
    signal input tmp_682[3];
    signal input tmp_683[3];
    signal input tmp_684[3];
    signal input tmp_685[3];
    signal input tmp_686[3];
    signal input tmp_687[3];
    signal input tmp_688[3];
    signal input tmp_689[3];
    signal input tmp_690[3];
    signal input tmp_691[3];
    signal input tmp_692[3];
    signal input tmp_693[3];
    signal input tmp_694[3];
    signal input tmp_695[3];
    signal input tmp_696[3];
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
    signal input tmp_802[3];
    signal input tmp_803[3];
    signal input tmp_804[3];
    signal input tmp_805[3];
    signal input tmp_806[3];
    signal input tmp_807[3];
    signal input tmp_808[3];
    signal input tmp_809[3];
    signal input tmp_810[3];
    signal input tmp_811[3];
    signal input tmp_812[3];
    signal input tmp_813[3];
    signal input tmp_814[3];
    signal input tmp_815[3];
    signal input tmp_816[3];
    signal input tmp_818[3];
    signal input tmp_819[3];
    signal input tmp_820[3];
    signal input tmp_821[3];
    signal input tmp_822[3];

    signal output tmp_823[3];
    signal tmp_697[3] <== CMul()(tmp_696, xDivXSubXi[0]);
    signal tmp_698[3] <== CMul()(tmp_695, xDivXSubXi[1]);
    signal tmp_699[3] <== CMul()(tmp_694, xDivXSubXi[2]);
    signal tmp_700[3] <== CMul()(tmp_693, xDivXSubXi[3]);
    signal tmp_701[3] <== CMul()(tmp_692, xDivXSubXi[4]);
    signal tmp_702[3] <== CMul()(tmp_691, xDivXSubXi[5]);
    signal tmp_703[3] <== CMul()(tmp_690, xDivXSubXi[6]);
    signal tmp_704[3] <== CMul()(tmp_689, xDivXSubXi[7]);
    signal tmp_705[3] <== CMul()(tmp_688, xDivXSubXi[8]);
    signal tmp_706[3] <== CMul()(tmp_687, xDivXSubXi[9]);
    signal tmp_707[3] <== CMul()(tmp_686, xDivXSubXi[10]);
    signal tmp_708[3] <== CMul()(tmp_685, xDivXSubXi[11]);
    signal tmp_709[3] <== CMul()(tmp_684, xDivXSubXi[12]);
    signal tmp_710[3] <== CMul()(tmp_683, xDivXSubXi[13]);
    signal tmp_711[3] <== CMul()(tmp_682, xDivXSubXi[14]);
    signal tmp_712[3] <== CMul()(tmp_681, xDivXSubXi[15]);
    signal tmp_713[3] <== CMul()(tmp_680, xDivXSubXi[16]);
    signal tmp_817[3] <== CMul()(challengesDeep[0], xDivXSubXi[17]);
    signal tmp_406[3] <== [tmp_697[0] + tmp_698[0], tmp_697[1] + tmp_698[1], tmp_697[2] + tmp_698[2]];
    signal tmp_407[3] <== [tmp_406[0] + tmp_699[0], tmp_406[1] + tmp_699[1], tmp_406[2] + tmp_699[2]];
    signal tmp_408[3] <== [tmp_407[0] + tmp_700[0], tmp_407[1] + tmp_700[1], tmp_407[2] + tmp_700[2]];
    signal tmp_409[3] <== [tmp_408[0] + tmp_701[0], tmp_408[1] + tmp_701[1], tmp_408[2] + tmp_701[2]];
    signal tmp_410[3] <== [tmp_409[0] + tmp_702[0], tmp_409[1] + tmp_702[1], tmp_409[2] + tmp_702[2]];
    signal tmp_411[3] <== [tmp_410[0] + tmp_703[0], tmp_410[1] + tmp_703[1], tmp_410[2] + tmp_703[2]];
    signal tmp_412[3] <== [tmp_411[0] + tmp_704[0], tmp_411[1] + tmp_704[1], tmp_411[2] + tmp_704[2]];
    signal tmp_413[3] <== [tmp_412[0] + tmp_705[0], tmp_412[1] + tmp_705[1], tmp_412[2] + tmp_705[2]];
    signal tmp_414[3] <== [tmp_413[0] + tmp_706[0], tmp_413[1] + tmp_706[1], tmp_413[2] + tmp_706[2]];
    signal tmp_415[3] <== [tmp_414[0] + tmp_707[0], tmp_414[1] + tmp_707[1], tmp_414[2] + tmp_707[2]];
    signal tmp_416[3] <== [tmp_415[0] + tmp_708[0], tmp_415[1] + tmp_708[1], tmp_415[2] + tmp_708[2]];
    signal tmp_417[3] <== [tmp_416[0] + tmp_709[0], tmp_416[1] + tmp_709[1], tmp_416[2] + tmp_709[2]];
    signal tmp_418[3] <== [tmp_417[0] + tmp_710[0], tmp_417[1] + tmp_710[1], tmp_417[2] + tmp_710[2]];
    signal tmp_419[3] <== [tmp_418[0] + tmp_711[0], tmp_418[1] + tmp_711[1], tmp_418[2] + tmp_711[2]];
    signal tmp_420[3] <== [tmp_419[0] + tmp_712[0], tmp_419[1] + tmp_712[1], tmp_419[2] + tmp_712[2]];
    signal tmp_421[3] <== [tmp_420[0] + tmp_713[0], tmp_420[1] + tmp_713[1], tmp_420[2] + tmp_713[2]];
    signal tmp_422[3] <== [consts[43] * tmp_421[0], consts[43] * tmp_421[1], consts[43] * tmp_421[2]];
    signal tmp_423[3] <== [challengesDeep[1][0] * consts[34], challengesDeep[1][1] * consts[34], challengesDeep[1][2] * consts[34]];
    signal tmp_424[3] <== [tmp_714[0] * consts[35], tmp_714[1] * consts[35], tmp_714[2] * consts[35]];
    signal tmp_425[3] <== [tmp_423[0] + tmp_424[0], tmp_423[1] + tmp_424[1], tmp_423[2] + tmp_424[2]];
    signal tmp_426[3] <== [tmp_711[0] + tmp_712[0], tmp_711[1] + tmp_712[1], tmp_711[2] + tmp_712[2]];
    signal tmp_427[3] <== [tmp_426[0] + tmp_713[0], tmp_426[1] + tmp_713[1], tmp_426[2] + tmp_713[2]];
    signal tmp_428[3] <== CMul()(tmp_425, tmp_427);
    signal tmp_429[3] <== [tmp_422[0] + tmp_428[0], tmp_422[1] + tmp_428[1], tmp_422[2] + tmp_428[2]];
    signal tmp_430[3] <== CMul()(tmp_715, mapValues.cm2_0);
    signal tmp_431[3] <== [tmp_712[0] + tmp_713[0], tmp_712[1] + tmp_713[1], tmp_712[2] + tmp_713[2]];
    signal tmp_432[3] <== CMul()(tmp_430, tmp_431);
    signal tmp_433[3] <== [tmp_429[0] + tmp_432[0], tmp_429[1] + tmp_432[1], tmp_429[2] + tmp_432[2]];
    signal tmp_434[3] <== [tmp_716[0] * consts[0], tmp_716[1] * consts[0], tmp_716[2] * consts[0]];
    signal tmp_435[3] <== [tmp_717[0] * consts[1], tmp_717[1] * consts[1], tmp_717[2] * consts[1]];
    signal tmp_436[3] <== [tmp_434[0] + tmp_435[0], tmp_434[1] + tmp_435[1], tmp_434[2] + tmp_435[2]];
    signal tmp_437[3] <== [tmp_718[0] * consts[2], tmp_718[1] * consts[2], tmp_718[2] * consts[2]];
    signal tmp_438[3] <== [tmp_436[0] + tmp_437[0], tmp_436[1] + tmp_437[1], tmp_436[2] + tmp_437[2]];
    signal tmp_439[3] <== [tmp_719[0] * consts[3], tmp_719[1] * consts[3], tmp_719[2] * consts[3]];
    signal tmp_440[3] <== [tmp_438[0] + tmp_439[0], tmp_438[1] + tmp_439[1], tmp_438[2] + tmp_439[2]];
    signal tmp_441[3] <== [tmp_720[0] * consts[4], tmp_720[1] * consts[4], tmp_720[2] * consts[4]];
    signal tmp_442[3] <== [tmp_440[0] + tmp_441[0], tmp_440[1] + tmp_441[1], tmp_440[2] + tmp_441[2]];
    signal tmp_443[3] <== [tmp_721[0] * consts[5], tmp_721[1] * consts[5], tmp_721[2] * consts[5]];
    signal tmp_444[3] <== [tmp_442[0] + tmp_443[0], tmp_442[1] + tmp_443[1], tmp_442[2] + tmp_443[2]];
    signal tmp_445[3] <== [tmp_722[0] * consts[6], tmp_722[1] * consts[6], tmp_722[2] * consts[6]];
    signal tmp_446[3] <== [tmp_444[0] + tmp_445[0], tmp_444[1] + tmp_445[1], tmp_444[2] + tmp_445[2]];
    signal tmp_447[3] <== [tmp_723[0] * consts[7], tmp_723[1] * consts[7], tmp_723[2] * consts[7]];
    signal tmp_448[3] <== [tmp_446[0] + tmp_447[0], tmp_446[1] + tmp_447[1], tmp_446[2] + tmp_447[2]];
    signal tmp_449[3] <== [tmp_724[0] * consts[8], tmp_724[1] * consts[8], tmp_724[2] * consts[8]];
    signal tmp_450[3] <== [tmp_448[0] + tmp_449[0], tmp_448[1] + tmp_449[1], tmp_448[2] + tmp_449[2]];
    signal tmp_451[3] <== [tmp_725[0] * consts[9], tmp_725[1] * consts[9], tmp_725[2] * consts[9]];
    signal tmp_452[3] <== [tmp_450[0] + tmp_451[0], tmp_450[1] + tmp_451[1], tmp_450[2] + tmp_451[2]];
    signal tmp_453[3] <== [tmp_726[0] * consts[10], tmp_726[1] * consts[10], tmp_726[2] * consts[10]];
    signal tmp_454[3] <== [tmp_452[0] + tmp_453[0], tmp_452[1] + tmp_453[1], tmp_452[2] + tmp_453[2]];
    signal tmp_455[3] <== [tmp_727[0] * consts[11], tmp_727[1] * consts[11], tmp_727[2] * consts[11]];
    signal tmp_456[3] <== [tmp_454[0] + tmp_455[0], tmp_454[1] + tmp_455[1], tmp_454[2] + tmp_455[2]];
    signal tmp_457[3] <== [tmp_728[0] * consts[12], tmp_728[1] * consts[12], tmp_728[2] * consts[12]];
    signal tmp_458[3] <== [tmp_456[0] + tmp_457[0], tmp_456[1] + tmp_457[1], tmp_456[2] + tmp_457[2]];
    signal tmp_459[3] <== [tmp_729[0] * consts[13], tmp_729[1] * consts[13], tmp_729[2] * consts[13]];
    signal tmp_460[3] <== [tmp_458[0] + tmp_459[0], tmp_458[1] + tmp_459[1], tmp_458[2] + tmp_459[2]];
    signal tmp_461[3] <== [tmp_730[0] * consts[14], tmp_730[1] * consts[14], tmp_730[2] * consts[14]];
    signal tmp_462[3] <== [tmp_460[0] + tmp_461[0], tmp_460[1] + tmp_461[1], tmp_460[2] + tmp_461[2]];
    signal tmp_463[3] <== [tmp_731[0] * consts[15], tmp_731[1] * consts[15], tmp_731[2] * consts[15]];
    signal tmp_464[3] <== [tmp_462[0] + tmp_463[0], tmp_462[1] + tmp_463[1], tmp_462[2] + tmp_463[2]];
    signal tmp_465[3] <== [tmp_732[0] * consts[16], tmp_732[1] * consts[16], tmp_732[2] * consts[16]];
    signal tmp_466[3] <== [tmp_464[0] + tmp_465[0], tmp_464[1] + tmp_465[1], tmp_464[2] + tmp_465[2]];
    signal tmp_467[3] <== [tmp_733[0] * consts[17], tmp_733[1] * consts[17], tmp_733[2] * consts[17]];
    signal tmp_468[3] <== [tmp_466[0] + tmp_467[0], tmp_466[1] + tmp_467[1], tmp_466[2] + tmp_467[2]];
    signal tmp_469[3] <== [tmp_734[0] * consts[18], tmp_734[1] * consts[18], tmp_734[2] * consts[18]];
    signal tmp_470[3] <== [tmp_468[0] + tmp_469[0], tmp_468[1] + tmp_469[1], tmp_468[2] + tmp_469[2]];
    signal tmp_471[3] <== [tmp_735[0] * consts[19], tmp_735[1] * consts[19], tmp_735[2] * consts[19]];
    signal tmp_472[3] <== [tmp_470[0] + tmp_471[0], tmp_470[1] + tmp_471[1], tmp_470[2] + tmp_471[2]];
    signal tmp_473[3] <== [tmp_736[0] * consts[20], tmp_736[1] * consts[20], tmp_736[2] * consts[20]];
    signal tmp_474[3] <== [tmp_472[0] + tmp_473[0], tmp_472[1] + tmp_473[1], tmp_472[2] + tmp_473[2]];
    signal tmp_475[3] <== [tmp_737[0] * consts[21], tmp_737[1] * consts[21], tmp_737[2] * consts[21]];
    signal tmp_476[3] <== [tmp_474[0] + tmp_475[0], tmp_474[1] + tmp_475[1], tmp_474[2] + tmp_475[2]];
    signal tmp_477[3] <== [tmp_738[0] * consts[22], tmp_738[1] * consts[22], tmp_738[2] * consts[22]];
    signal tmp_478[3] <== [tmp_476[0] + tmp_477[0], tmp_476[1] + tmp_477[1], tmp_476[2] + tmp_477[2]];
    signal tmp_479[3] <== [tmp_739[0] * consts[23], tmp_739[1] * consts[23], tmp_739[2] * consts[23]];
    signal tmp_480[3] <== [tmp_478[0] + tmp_479[0], tmp_478[1] + tmp_479[1], tmp_478[2] + tmp_479[2]];
    signal tmp_481[3] <== [tmp_740[0] * consts[24], tmp_740[1] * consts[24], tmp_740[2] * consts[24]];
    signal tmp_482[3] <== [tmp_480[0] + tmp_481[0], tmp_480[1] + tmp_481[1], tmp_480[2] + tmp_481[2]];
    signal tmp_483[3] <== [tmp_741[0] * consts[25], tmp_741[1] * consts[25], tmp_741[2] * consts[25]];
    signal tmp_484[3] <== [tmp_482[0] + tmp_483[0], tmp_482[1] + tmp_483[1], tmp_482[2] + tmp_483[2]];
    signal tmp_485[3] <== [tmp_742[0] * consts[26], tmp_742[1] * consts[26], tmp_742[2] * consts[26]];
    signal tmp_486[3] <== [tmp_484[0] + tmp_485[0], tmp_484[1] + tmp_485[1], tmp_484[2] + tmp_485[2]];
    signal tmp_487[3] <== [tmp_743[0] * consts[27], tmp_743[1] * consts[27], tmp_743[2] * consts[27]];
    signal tmp_488[3] <== [tmp_486[0] + tmp_487[0], tmp_486[1] + tmp_487[1], tmp_486[2] + tmp_487[2]];
    signal tmp_489[3] <== [tmp_744[0] * consts[28], tmp_744[1] * consts[28], tmp_744[2] * consts[28]];
    signal tmp_490[3] <== [tmp_488[0] + tmp_489[0], tmp_488[1] + tmp_489[1], tmp_488[2] + tmp_489[2]];
    signal tmp_491[3] <== [tmp_745[0] * consts[29], tmp_745[1] * consts[29], tmp_745[2] * consts[29]];
    signal tmp_492[3] <== [tmp_490[0] + tmp_491[0], tmp_490[1] + tmp_491[1], tmp_490[2] + tmp_491[2]];
    signal tmp_493[3] <== [tmp_746[0] * consts[30], tmp_746[1] * consts[30], tmp_746[2] * consts[30]];
    signal tmp_494[3] <== [tmp_492[0] + tmp_493[0], tmp_492[1] + tmp_493[1], tmp_492[2] + tmp_493[2]];
    signal tmp_495[3] <== [tmp_747[0] * consts[31], tmp_747[1] * consts[31], tmp_747[2] * consts[31]];
    signal tmp_496[3] <== [tmp_494[0] + tmp_495[0], tmp_494[1] + tmp_495[1], tmp_494[2] + tmp_495[2]];
    signal tmp_497[3] <== [tmp_748[0] * consts[32], tmp_748[1] * consts[32], tmp_748[2] * consts[32]];
    signal tmp_498[3] <== [tmp_496[0] + tmp_497[0], tmp_496[1] + tmp_497[1], tmp_496[2] + tmp_497[2]];
    signal tmp_499[3] <== [tmp_749[0] * consts[33], tmp_749[1] * consts[33], tmp_749[2] * consts[33]];
    signal tmp_500[3] <== [tmp_498[0] + tmp_499[0], tmp_498[1] + tmp_499[1], tmp_498[2] + tmp_499[2]];
    signal tmp_501[3] <== [tmp_751[0] * consts[37], tmp_751[1] * consts[37], tmp_751[2] * consts[37]];
    signal tmp_502[3] <== [tmp_500[0] + tmp_501[0], tmp_500[1] + tmp_501[1], tmp_500[2] + tmp_501[2]];
    signal tmp_503[3] <== [tmp_752[0] * consts[38], tmp_752[1] * consts[38], tmp_752[2] * consts[38]];
    signal tmp_504[3] <== [tmp_502[0] + tmp_503[0], tmp_502[1] + tmp_503[1], tmp_502[2] + tmp_503[2]];
    signal tmp_505[3] <== [tmp_753[0] * consts[39], tmp_753[1] * consts[39], tmp_753[2] * consts[39]];
    signal tmp_506[3] <== [tmp_504[0] + tmp_505[0], tmp_504[1] + tmp_505[1], tmp_504[2] + tmp_505[2]];
    signal tmp_507[3] <== [tmp_754[0] * consts[40], tmp_754[1] * consts[40], tmp_754[2] * consts[40]];
    signal tmp_508[3] <== [tmp_506[0] + tmp_507[0], tmp_506[1] + tmp_507[1], tmp_506[2] + tmp_507[2]];
    signal tmp_509[3] <== [tmp_755[0] * consts[41], tmp_755[1] * consts[41], tmp_755[2] * consts[41]];
    signal tmp_510[3] <== [tmp_508[0] + tmp_509[0], tmp_508[1] + tmp_509[1], tmp_508[2] + tmp_509[2]];
    signal tmp_511[3] <== [tmp_756[0] * consts[42], tmp_756[1] * consts[42], tmp_756[2] * consts[42]];
    signal tmp_512[3] <== [tmp_510[0] + tmp_511[0], tmp_510[1] + tmp_511[1], tmp_510[2] + tmp_511[2]];
    signal tmp_513[3] <== [tmp_757[0] * consts[44], tmp_757[1] * consts[44], tmp_757[2] * consts[44]];
    signal tmp_514[3] <== [tmp_512[0] + tmp_513[0], tmp_512[1] + tmp_513[1], tmp_512[2] + tmp_513[2]];
    signal tmp_515[3] <== [tmp_765[0] * mapValues.cm1_6, tmp_765[1] * mapValues.cm1_6, tmp_765[2] * mapValues.cm1_6];
    signal tmp_516[3] <== [tmp_514[0] + tmp_515[0], tmp_514[1] + tmp_515[1], tmp_514[2] + tmp_515[2]];
    signal tmp_517[3] <== [tmp_766[0] * mapValues.cm1_7, tmp_766[1] * mapValues.cm1_7, tmp_766[2] * mapValues.cm1_7];
    signal tmp_518[3] <== [tmp_516[0] + tmp_517[0], tmp_516[1] + tmp_517[1], tmp_516[2] + tmp_517[2]];
    signal tmp_519[3] <== [tmp_767[0] * mapValues.cm1_8, tmp_767[1] * mapValues.cm1_8, tmp_767[2] * mapValues.cm1_8];
    signal tmp_520[3] <== [tmp_518[0] + tmp_519[0], tmp_518[1] + tmp_519[1], tmp_518[2] + tmp_519[2]];
    signal tmp_521[3] <== [tmp_768[0] * mapValues.cm1_9, tmp_768[1] * mapValues.cm1_9, tmp_768[2] * mapValues.cm1_9];
    signal tmp_522[3] <== [tmp_520[0] + tmp_521[0], tmp_520[1] + tmp_521[1], tmp_520[2] + tmp_521[2]];
    signal tmp_523[3] <== [tmp_769[0] * mapValues.cm1_10, tmp_769[1] * mapValues.cm1_10, tmp_769[2] * mapValues.cm1_10];
    signal tmp_524[3] <== [tmp_522[0] + tmp_523[0], tmp_522[1] + tmp_523[1], tmp_522[2] + tmp_523[2]];
    signal tmp_525[3] <== [tmp_770[0] * mapValues.cm1_11, tmp_770[1] * mapValues.cm1_11, tmp_770[2] * mapValues.cm1_11];
    signal tmp_526[3] <== [tmp_524[0] + tmp_525[0], tmp_524[1] + tmp_525[1], tmp_524[2] + tmp_525[2]];
    signal tmp_527[3] <== [tmp_771[0] * mapValues.cm1_12, tmp_771[1] * mapValues.cm1_12, tmp_771[2] * mapValues.cm1_12];
    signal tmp_528[3] <== [tmp_526[0] + tmp_527[0], tmp_526[1] + tmp_527[1], tmp_526[2] + tmp_527[2]];
    signal tmp_529[3] <== [tmp_772[0] * mapValues.cm1_13, tmp_772[1] * mapValues.cm1_13, tmp_772[2] * mapValues.cm1_13];
    signal tmp_530[3] <== [tmp_528[0] + tmp_529[0], tmp_528[1] + tmp_529[1], tmp_528[2] + tmp_529[2]];
    signal tmp_531[3] <== [tmp_773[0] * mapValues.cm1_14, tmp_773[1] * mapValues.cm1_14, tmp_773[2] * mapValues.cm1_14];
    signal tmp_532[3] <== [tmp_530[0] + tmp_531[0], tmp_530[1] + tmp_531[1], tmp_530[2] + tmp_531[2]];
    signal tmp_533[3] <== [tmp_791[0] * mapValues.cm1_32, tmp_791[1] * mapValues.cm1_32, tmp_791[2] * mapValues.cm1_32];
    signal tmp_534[3] <== [tmp_532[0] + tmp_533[0], tmp_532[1] + tmp_533[1], tmp_532[2] + tmp_533[2]];
    signal tmp_535[3] <== [tmp_792[0] * mapValues.cm1_33, tmp_792[1] * mapValues.cm1_33, tmp_792[2] * mapValues.cm1_33];
    signal tmp_536[3] <== [tmp_534[0] + tmp_535[0], tmp_534[1] + tmp_535[1], tmp_534[2] + tmp_535[2]];
    signal tmp_537[3] <== [tmp_793[0] * mapValues.cm1_34, tmp_793[1] * mapValues.cm1_34, tmp_793[2] * mapValues.cm1_34];
    signal tmp_538[3] <== [tmp_536[0] + tmp_537[0], tmp_536[1] + tmp_537[1], tmp_536[2] + tmp_537[2]];
    signal tmp_539[3] <== [tmp_794[0] * mapValues.cm1_35, tmp_794[1] * mapValues.cm1_35, tmp_794[2] * mapValues.cm1_35];
    signal tmp_540[3] <== [tmp_538[0] + tmp_539[0], tmp_538[1] + tmp_539[1], tmp_538[2] + tmp_539[2]];
    signal tmp_541[3] <== [tmp_795[0] * mapValues.cm1_36, tmp_795[1] * mapValues.cm1_36, tmp_795[2] * mapValues.cm1_36];
    signal tmp_542[3] <== [tmp_540[0] + tmp_541[0], tmp_540[1] + tmp_541[1], tmp_540[2] + tmp_541[2]];
    signal tmp_543[3] <== [tmp_796[0] * mapValues.cm1_37, tmp_796[1] * mapValues.cm1_37, tmp_796[2] * mapValues.cm1_37];
    signal tmp_544[3] <== [tmp_542[0] + tmp_543[0], tmp_542[1] + tmp_543[1], tmp_542[2] + tmp_543[2]];
    signal tmp_545[3] <== [tmp_797[0] * mapValues.cm1_38, tmp_797[1] * mapValues.cm1_38, tmp_797[2] * mapValues.cm1_38];
    signal tmp_546[3] <== [tmp_544[0] + tmp_545[0], tmp_544[1] + tmp_545[1], tmp_544[2] + tmp_545[2]];
    signal tmp_547[3] <== [tmp_798[0] * mapValues.cm1_39, tmp_798[1] * mapValues.cm1_39, tmp_798[2] * mapValues.cm1_39];
    signal tmp_548[3] <== [tmp_546[0] + tmp_547[0], tmp_546[1] + tmp_547[1], tmp_546[2] + tmp_547[2]];
    signal tmp_549[3] <== [tmp_799[0] * mapValues.cm1_40, tmp_799[1] * mapValues.cm1_40, tmp_799[2] * mapValues.cm1_40];
    signal tmp_550[3] <== [tmp_548[0] + tmp_549[0], tmp_548[1] + tmp_549[1], tmp_548[2] + tmp_549[2]];
    signal tmp_551[3] <== [tmp_800[0] * mapValues.cm1_41, tmp_800[1] * mapValues.cm1_41, tmp_800[2] * mapValues.cm1_41];
    signal tmp_552[3] <== [tmp_550[0] + tmp_551[0], tmp_550[1] + tmp_551[1], tmp_550[2] + tmp_551[2]];
    signal tmp_553[3] <== [tmp_801[0] * mapValues.cm1_42, tmp_801[1] * mapValues.cm1_42, tmp_801[2] * mapValues.cm1_42];
    signal tmp_554[3] <== [tmp_552[0] + tmp_553[0], tmp_552[1] + tmp_553[1], tmp_552[2] + tmp_553[2]];
    signal tmp_555[3] <== [tmp_802[0] * mapValues.cm1_43, tmp_802[1] * mapValues.cm1_43, tmp_802[2] * mapValues.cm1_43];
    signal tmp_556[3] <== [tmp_554[0] + tmp_555[0], tmp_554[1] + tmp_555[1], tmp_554[2] + tmp_555[2]];
    signal tmp_557[3] <== [tmp_803[0] * mapValues.cm1_44, tmp_803[1] * mapValues.cm1_44, tmp_803[2] * mapValues.cm1_44];
    signal tmp_558[3] <== [tmp_556[0] + tmp_557[0], tmp_556[1] + tmp_557[1], tmp_556[2] + tmp_557[2]];
    signal tmp_559[3] <== [tmp_804[0] * mapValues.cm1_45, tmp_804[1] * mapValues.cm1_45, tmp_804[2] * mapValues.cm1_45];
    signal tmp_560[3] <== [tmp_558[0] + tmp_559[0], tmp_558[1] + tmp_559[1], tmp_558[2] + tmp_559[2]];
    signal tmp_561[3] <== [tmp_805[0] * mapValues.cm1_46, tmp_805[1] * mapValues.cm1_46, tmp_805[2] * mapValues.cm1_46];
    signal tmp_562[3] <== [tmp_560[0] + tmp_561[0], tmp_560[1] + tmp_561[1], tmp_560[2] + tmp_561[2]];
    signal tmp_563[3] <== [tmp_806[0] * mapValues.cm1_47, tmp_806[1] * mapValues.cm1_47, tmp_806[2] * mapValues.cm1_47];
    signal tmp_564[3] <== [tmp_562[0] + tmp_563[0], tmp_562[1] + tmp_563[1], tmp_562[2] + tmp_563[2]];
    signal tmp_565[3] <== CMul()(tmp_807, mapValues.cm2_1);
    signal tmp_566[3] <== [tmp_564[0] + tmp_565[0], tmp_564[1] + tmp_565[1], tmp_564[2] + tmp_565[2]];
    signal tmp_567[3] <== CMul()(tmp_808, mapValues.cm2_2);
    signal tmp_568[3] <== [tmp_566[0] + tmp_567[0], tmp_566[1] + tmp_567[1], tmp_566[2] + tmp_567[2]];
    signal tmp_569[3] <== CMul()(tmp_809, mapValues.cm2_3);
    signal tmp_570[3] <== [tmp_568[0] + tmp_569[0], tmp_568[1] + tmp_569[1], tmp_568[2] + tmp_569[2]];
    signal tmp_571[3] <== CMul()(tmp_810, mapValues.cm3_0);
    signal tmp_572[3] <== [tmp_570[0] + tmp_571[0], tmp_570[1] + tmp_571[1], tmp_570[2] + tmp_571[2]];
    signal tmp_573[3] <== CMul()(tmp_811, mapValues.cm3_1);
    signal tmp_574[3] <== [tmp_572[0] + tmp_573[0], tmp_572[1] + tmp_573[1], tmp_572[2] + tmp_573[2]];
    signal tmp_575[3] <== CMul()(tmp_812, mapValues.cm3_2);
    signal tmp_576[3] <== [tmp_574[0] + tmp_575[0], tmp_574[1] + tmp_575[1], tmp_574[2] + tmp_575[2]];
    signal tmp_577[3] <== CMul()(tmp_813, mapValues.cm3_3);
    signal tmp_578[3] <== [tmp_576[0] + tmp_577[0], tmp_576[1] + tmp_577[1], tmp_576[2] + tmp_577[2]];
    signal tmp_579[3] <== CMul()(tmp_814, mapValues.cm3_4);
    signal tmp_580[3] <== [tmp_578[0] + tmp_579[0], tmp_578[1] + tmp_579[1], tmp_578[2] + tmp_579[2]];
    signal tmp_581[3] <== CMul()(tmp_815, mapValues.cm3_5);
    signal tmp_582[3] <== [tmp_580[0] + tmp_581[0], tmp_580[1] + tmp_581[1], tmp_580[2] + tmp_581[2]];
    signal tmp_583[3] <== CMul()(tmp_816, mapValues.cm3_6);
    signal tmp_584[3] <== [tmp_582[0] + tmp_583[0], tmp_582[1] + tmp_583[1], tmp_582[2] + tmp_583[2]];
    signal tmp_585[3] <== CMul()(tmp_584, tmp_713);
    signal tmp_586[3] <== [tmp_433[0] + tmp_585[0], tmp_433[1] + tmp_585[1], tmp_433[2] + tmp_585[2]];
    signal tmp_587[3] <== [tmp_750[0] * consts[36], tmp_750[1] * consts[36], tmp_750[2] * consts[36]];
    signal tmp_588[3] <== [tmp_758[0] * consts[45], tmp_758[1] * consts[45], tmp_758[2] * consts[45]];
    signal tmp_589[3] <== [tmp_587[0] + tmp_588[0], tmp_587[1] + tmp_588[1], tmp_587[2] + tmp_588[2]];
    signal tmp_590[3] <== [tmp_759[0] * mapValues.cm1_0, tmp_759[1] * mapValues.cm1_0, tmp_759[2] * mapValues.cm1_0];
    signal tmp_591[3] <== [tmp_589[0] + tmp_590[0], tmp_589[1] + tmp_590[1], tmp_589[2] + tmp_590[2]];
    signal tmp_592[3] <== [tmp_760[0] * mapValues.cm1_1, tmp_760[1] * mapValues.cm1_1, tmp_760[2] * mapValues.cm1_1];
    signal tmp_593[3] <== [tmp_591[0] + tmp_592[0], tmp_591[1] + tmp_592[1], tmp_591[2] + tmp_592[2]];
    signal tmp_594[3] <== [tmp_761[0] * mapValues.cm1_2, tmp_761[1] * mapValues.cm1_2, tmp_761[2] * mapValues.cm1_2];
    signal tmp_595[3] <== [tmp_593[0] + tmp_594[0], tmp_593[1] + tmp_594[1], tmp_593[2] + tmp_594[2]];
    signal tmp_596[3] <== [tmp_762[0] * mapValues.cm1_3, tmp_762[1] * mapValues.cm1_3, tmp_762[2] * mapValues.cm1_3];
    signal tmp_597[3] <== [tmp_595[0] + tmp_596[0], tmp_595[1] + tmp_596[1], tmp_595[2] + tmp_596[2]];
    signal tmp_598[3] <== [tmp_763[0] * mapValues.cm1_4, tmp_763[1] * mapValues.cm1_4, tmp_763[2] * mapValues.cm1_4];
    signal tmp_599[3] <== [tmp_597[0] + tmp_598[0], tmp_597[1] + tmp_598[1], tmp_597[2] + tmp_598[2]];
    signal tmp_600[3] <== [tmp_764[0] * mapValues.cm1_5, tmp_764[1] * mapValues.cm1_5, tmp_764[2] * mapValues.cm1_5];
    signal tmp_601[3] <== [tmp_599[0] + tmp_600[0], tmp_599[1] + tmp_600[1], tmp_599[2] + tmp_600[2]];
    signal tmp_602[3] <== [tmp_775[0] * mapValues.cm1_16, tmp_775[1] * mapValues.cm1_16, tmp_775[2] * mapValues.cm1_16];
    signal tmp_603[3] <== [tmp_601[0] + tmp_602[0], tmp_601[1] + tmp_602[1], tmp_601[2] + tmp_602[2]];
    signal tmp_604[3] <== [tmp_776[0] * mapValues.cm1_17, tmp_776[1] * mapValues.cm1_17, tmp_776[2] * mapValues.cm1_17];
    signal tmp_605[3] <== [tmp_603[0] + tmp_604[0], tmp_603[1] + tmp_604[1], tmp_603[2] + tmp_604[2]];
    signal tmp_606[3] <== [tmp_777[0] * mapValues.cm1_18, tmp_777[1] * mapValues.cm1_18, tmp_777[2] * mapValues.cm1_18];
    signal tmp_607[3] <== [tmp_605[0] + tmp_606[0], tmp_605[1] + tmp_606[1], tmp_605[2] + tmp_606[2]];
    signal tmp_608[3] <== [tmp_778[0] * mapValues.cm1_19, tmp_778[1] * mapValues.cm1_19, tmp_778[2] * mapValues.cm1_19];
    signal tmp_609[3] <== [tmp_607[0] + tmp_608[0], tmp_607[1] + tmp_608[1], tmp_607[2] + tmp_608[2]];
    signal tmp_610[3] <== [tmp_779[0] * mapValues.cm1_20, tmp_779[1] * mapValues.cm1_20, tmp_779[2] * mapValues.cm1_20];
    signal tmp_611[3] <== [tmp_609[0] + tmp_610[0], tmp_609[1] + tmp_610[1], tmp_609[2] + tmp_610[2]];
    signal tmp_612[3] <== [tmp_780[0] * mapValues.cm1_21, tmp_780[1] * mapValues.cm1_21, tmp_780[2] * mapValues.cm1_21];
    signal tmp_613[3] <== [tmp_611[0] + tmp_612[0], tmp_611[1] + tmp_612[1], tmp_611[2] + tmp_612[2]];
    signal tmp_614[3] <== [tmp_781[0] * mapValues.cm1_22, tmp_781[1] * mapValues.cm1_22, tmp_781[2] * mapValues.cm1_22];
    signal tmp_615[3] <== [tmp_613[0] + tmp_614[0], tmp_613[1] + tmp_614[1], tmp_613[2] + tmp_614[2]];
    signal tmp_616[3] <== [tmp_782[0] * mapValues.cm1_23, tmp_782[1] * mapValues.cm1_23, tmp_782[2] * mapValues.cm1_23];
    signal tmp_617[3] <== [tmp_615[0] + tmp_616[0], tmp_615[1] + tmp_616[1], tmp_615[2] + tmp_616[2]];
    signal tmp_618[3] <== [tmp_783[0] * mapValues.cm1_24, tmp_783[1] * mapValues.cm1_24, tmp_783[2] * mapValues.cm1_24];
    signal tmp_619[3] <== [tmp_617[0] + tmp_618[0], tmp_617[1] + tmp_618[1], tmp_617[2] + tmp_618[2]];
    signal tmp_620[3] <== [tmp_784[0] * mapValues.cm1_25, tmp_784[1] * mapValues.cm1_25, tmp_784[2] * mapValues.cm1_25];
    signal tmp_621[3] <== [tmp_619[0] + tmp_620[0], tmp_619[1] + tmp_620[1], tmp_619[2] + tmp_620[2]];
    signal tmp_622[3] <== [tmp_785[0] * mapValues.cm1_26, tmp_785[1] * mapValues.cm1_26, tmp_785[2] * mapValues.cm1_26];
    signal tmp_623[3] <== [tmp_621[0] + tmp_622[0], tmp_621[1] + tmp_622[1], tmp_621[2] + tmp_622[2]];
    signal tmp_624[3] <== [tmp_786[0] * mapValues.cm1_27, tmp_786[1] * mapValues.cm1_27, tmp_786[2] * mapValues.cm1_27];
    signal tmp_625[3] <== [tmp_623[0] + tmp_624[0], tmp_623[1] + tmp_624[1], tmp_623[2] + tmp_624[2]];
    signal tmp_626[3] <== [tmp_787[0] * mapValues.cm1_28, tmp_787[1] * mapValues.cm1_28, tmp_787[2] * mapValues.cm1_28];
    signal tmp_627[3] <== [tmp_625[0] + tmp_626[0], tmp_625[1] + tmp_626[1], tmp_625[2] + tmp_626[2]];
    signal tmp_628[3] <== [tmp_788[0] * mapValues.cm1_29, tmp_788[1] * mapValues.cm1_29, tmp_788[2] * mapValues.cm1_29];
    signal tmp_629[3] <== [tmp_627[0] + tmp_628[0], tmp_627[1] + tmp_628[1], tmp_627[2] + tmp_628[2]];
    signal tmp_630[3] <== [tmp_789[0] * mapValues.cm1_30, tmp_789[1] * mapValues.cm1_30, tmp_789[2] * mapValues.cm1_30];
    signal tmp_631[3] <== [tmp_629[0] + tmp_630[0], tmp_629[1] + tmp_630[1], tmp_629[2] + tmp_630[2]];
    signal tmp_632[3] <== [tmp_790[0] * mapValues.cm1_31, tmp_790[1] * mapValues.cm1_31, tmp_790[2] * mapValues.cm1_31];
    signal tmp_633[3] <== [tmp_631[0] + tmp_632[0], tmp_631[1] + tmp_632[1], tmp_631[2] + tmp_632[2]];
    signal tmp_634[3] <== [tmp_713[0] + tmp_817[0], tmp_713[1] + tmp_817[1], tmp_713[2] + tmp_817[2]];
    signal tmp_635[3] <== CMul()(tmp_633, tmp_634);
    signal tmp_636[3] <== [tmp_586[0] + tmp_635[0], tmp_586[1] + tmp_635[1], tmp_586[2] + tmp_635[2]];
    signal tmp_637[3] <== [tmp_774[0] * mapValues.cm1_15, tmp_774[1] * mapValues.cm1_15, tmp_774[2] * mapValues.cm1_15];
    signal tmp_639[3] <== [tmp_634[0] + xDivXSubXi[18][0], tmp_634[1] + xDivXSubXi[18][1], tmp_634[2] + xDivXSubXi[18][2]];
    signal tmp_640[3] <== CMul()(tmp_637, tmp_639);
    signal tmp_641[3] <== [tmp_636[0] + tmp_640[0], tmp_636[1] + tmp_640[1], tmp_636[2] + tmp_640[2]];
    signal tmp_642[3] <== CMul()(tmp_697, evals[0]);
    signal tmp_643[3] <== [tmp_641[0] - tmp_642[0], tmp_641[1] - tmp_642[1], tmp_641[2] - tmp_642[2]];
    signal tmp_644[3] <== CMul()(tmp_698, evals[1]);
    signal tmp_645[3] <== [tmp_643[0] - tmp_644[0], tmp_643[1] - tmp_644[1], tmp_643[2] - tmp_644[2]];
    signal tmp_646[3] <== CMul()(tmp_699, evals[2]);
    signal tmp_647[3] <== [tmp_645[0] - tmp_646[0], tmp_645[1] - tmp_646[1], tmp_645[2] - tmp_646[2]];
    signal tmp_648[3] <== CMul()(tmp_700, evals[3]);
    signal tmp_649[3] <== [tmp_647[0] - tmp_648[0], tmp_647[1] - tmp_648[1], tmp_647[2] - tmp_648[2]];
    signal tmp_650[3] <== CMul()(tmp_701, evals[4]);
    signal tmp_651[3] <== [tmp_649[0] - tmp_650[0], tmp_649[1] - tmp_650[1], tmp_649[2] - tmp_650[2]];
    signal tmp_652[3] <== CMul()(tmp_702, evals[5]);
    signal tmp_653[3] <== [tmp_651[0] - tmp_652[0], tmp_651[1] - tmp_652[1], tmp_651[2] - tmp_652[2]];
    signal tmp_654[3] <== CMul()(tmp_703, evals[6]);
    signal tmp_655[3] <== [tmp_653[0] - tmp_654[0], tmp_653[1] - tmp_654[1], tmp_653[2] - tmp_654[2]];
    signal tmp_656[3] <== CMul()(tmp_704, evals[7]);
    signal tmp_657[3] <== [tmp_655[0] - tmp_656[0], tmp_655[1] - tmp_656[1], tmp_655[2] - tmp_656[2]];
    signal tmp_658[3] <== CMul()(tmp_705, evals[8]);
    signal tmp_659[3] <== [tmp_657[0] - tmp_658[0], tmp_657[1] - tmp_658[1], tmp_657[2] - tmp_658[2]];
    signal tmp_660[3] <== CMul()(tmp_706, evals[9]);
    signal tmp_661[3] <== [tmp_659[0] - tmp_660[0], tmp_659[1] - tmp_660[1], tmp_659[2] - tmp_660[2]];
    signal tmp_662[3] <== CMul()(tmp_707, evals[10]);
    signal tmp_663[3] <== [tmp_661[0] - tmp_662[0], tmp_661[1] - tmp_662[1], tmp_661[2] - tmp_662[2]];
    signal tmp_664[3] <== CMul()(tmp_708, evals[11]);
    signal tmp_665[3] <== [tmp_663[0] - tmp_664[0], tmp_663[1] - tmp_664[1], tmp_663[2] - tmp_664[2]];
    signal tmp_666[3] <== CMul()(tmp_709, evals[12]);
    signal tmp_667[3] <== [tmp_665[0] - tmp_666[0], tmp_665[1] - tmp_666[1], tmp_665[2] - tmp_666[2]];
    signal tmp_668[3] <== CMul()(tmp_710, evals[13]);
    signal tmp_669[3] <== [tmp_667[0] - tmp_668[0], tmp_667[1] - tmp_668[1], tmp_667[2] - tmp_668[2]];
    signal tmp_670[3] <== CMul()(tmp_711, tmp_818);
    signal tmp_671[3] <== [tmp_669[0] - tmp_670[0], tmp_669[1] - tmp_670[1], tmp_669[2] - tmp_670[2]];
    signal tmp_672[3] <== CMul()(tmp_712, tmp_819);
    signal tmp_673[3] <== [tmp_671[0] - tmp_672[0], tmp_671[1] - tmp_672[1], tmp_671[2] - tmp_672[2]];
    signal tmp_674[3] <== CMul()(tmp_713, tmp_820);
    signal tmp_675[3] <== [tmp_673[0] - tmp_674[0], tmp_673[1] - tmp_674[1], tmp_673[2] - tmp_674[2]];
    signal tmp_676[3] <== CMul()(tmp_817, tmp_821);
    signal tmp_677[3] <== [tmp_675[0] - tmp_676[0], tmp_675[1] - tmp_676[1], tmp_675[2] - tmp_676[2]];
    signal tmp_678[3] <== CMul()(xDivXSubXi[18], tmp_822);
    tmp_823 <== [tmp_677[0] - tmp_678[0], tmp_677[1] - tmp_678[1], tmp_677[2] - tmp_678[2]];

}


template CalculateFRIConstantsChunks0() {
    signal input challengesDeep[2][3];
    signal input evals[152][3];
    signal output tmp_680[3];
    signal output tmp_681[3];
    signal output tmp_682[3];
    signal output tmp_683[3];
    signal output tmp_684[3];
    signal output tmp_685[3];
    signal output tmp_686[3];
    signal output tmp_687[3];
    signal output tmp_688[3];
    signal output tmp_689[3];
    signal output tmp_690[3];
    signal output tmp_691[3];
    signal output tmp_692[3];
    signal output tmp_693[3];
    signal output tmp_694[3];
    signal output tmp_695[3];
    signal output tmp_696[3];
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
    signal output tmp_802[3];
    signal output tmp_803[3];
    signal output tmp_804[3];
    signal output tmp_805[3];
    signal output tmp_806[3];
    signal output tmp_807[3];
    signal output tmp_808[3];
    signal output tmp_809[3];
    signal output tmp_810[3];
    signal output tmp_811[3];
    signal output tmp_812[3];
    signal output tmp_813[3];
    signal output tmp_814[3];
    signal output tmp_815[3];
    signal output tmp_816[3];
    signal output tmp_818[3];
    signal output tmp_819[3];
    signal output tmp_820[3];
    signal output tmp_821[3];
    signal output tmp_822[3];
    tmp_680 <== CMul()(challengesDeep[0], challengesDeep[0]);
    tmp_681 <== CMul()(tmp_680, challengesDeep[0]);
    tmp_682 <== CMul()(tmp_681, challengesDeep[0]);
    tmp_683 <== CMul()(tmp_682, challengesDeep[0]);
    tmp_684 <== CMul()(tmp_683, challengesDeep[0]);
    tmp_685 <== CMul()(tmp_684, challengesDeep[0]);
    tmp_686 <== CMul()(tmp_685, challengesDeep[0]);
    tmp_687 <== CMul()(tmp_686, challengesDeep[0]);
    tmp_688 <== CMul()(tmp_687, challengesDeep[0]);
    tmp_689 <== CMul()(tmp_688, challengesDeep[0]);
    tmp_690 <== CMul()(tmp_689, challengesDeep[0]);
    tmp_691 <== CMul()(tmp_690, challengesDeep[0]);
    tmp_692 <== CMul()(tmp_691, challengesDeep[0]);
    tmp_693 <== CMul()(tmp_692, challengesDeep[0]);
    tmp_694 <== CMul()(tmp_693, challengesDeep[0]);
    tmp_695 <== CMul()(tmp_694, challengesDeep[0]);
    tmp_696 <== CMul()(tmp_695, challengesDeep[0]);
    tmp_714 <== CMul()(challengesDeep[1], challengesDeep[1]);
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
    tmp_802 <== CMul()(tmp_801, challengesDeep[1]);
    tmp_803 <== CMul()(tmp_802, challengesDeep[1]);
    tmp_804 <== CMul()(tmp_803, challengesDeep[1]);
    tmp_805 <== CMul()(tmp_804, challengesDeep[1]);
    tmp_806 <== CMul()(tmp_805, challengesDeep[1]);
    tmp_807 <== CMul()(tmp_806, challengesDeep[1]);
    tmp_808 <== CMul()(tmp_807, challengesDeep[1]);
    tmp_809 <== CMul()(tmp_808, challengesDeep[1]);
    tmp_810 <== CMul()(tmp_809, challengesDeep[1]);
    tmp_811 <== CMul()(tmp_810, challengesDeep[1]);
    tmp_812 <== CMul()(tmp_811, challengesDeep[1]);
    tmp_813 <== CMul()(tmp_812, challengesDeep[1]);
    tmp_814 <== CMul()(tmp_813, challengesDeep[1]);
    tmp_815 <== CMul()(tmp_814, challengesDeep[1]);
    tmp_816 <== CMul()(tmp_815, challengesDeep[1]);
    signal tmp_138[3] <== CMul()(challengesDeep[1], evals[14]);
    signal tmp_139[3] <== [evals[16][0] + tmp_138[0], evals[16][1] + tmp_138[1], evals[16][2] + tmp_138[2]];
    signal tmp_140[3] <== CMul()(tmp_714, evals[15]);
    tmp_818 <== [tmp_139[0] + tmp_140[0], tmp_139[1] + tmp_140[1], tmp_139[2] + tmp_140[2]];
    signal tmp_142[3] <== CMul()(challengesDeep[1], evals[17]);
    signal tmp_143[3] <== [evals[19][0] + tmp_142[0], evals[19][1] + tmp_142[1], evals[19][2] + tmp_142[2]];
    signal tmp_144[3] <== CMul()(tmp_714, evals[18]);
    signal tmp_145[3] <== [tmp_143[0] + tmp_144[0], tmp_143[1] + tmp_144[1], tmp_143[2] + tmp_144[2]];
    signal tmp_146[3] <== CMul()(tmp_715, evals[20]);
    tmp_819 <== [tmp_145[0] + tmp_146[0], tmp_145[1] + tmp_146[1], tmp_145[2] + tmp_146[2]];
    signal tmp_148[3] <== CMul()(challengesDeep[1], evals[55]);
    signal tmp_149[3] <== [evals[64][0] + tmp_148[0], evals[64][1] + tmp_148[1], evals[64][2] + tmp_148[2]];
    signal tmp_150[3] <== CMul()(tmp_714, evals[56]);
    signal tmp_151[3] <== [tmp_149[0] + tmp_150[0], tmp_149[1] + tmp_150[1], tmp_149[2] + tmp_150[2]];
    signal tmp_152[3] <== CMul()(tmp_715, evals[115]);
    signal tmp_153[3] <== [tmp_151[0] + tmp_152[0], tmp_151[1] + tmp_152[1], tmp_151[2] + tmp_152[2]];
    signal tmp_154[3] <== CMul()(tmp_716, evals[21]);
    signal tmp_155[3] <== [tmp_153[0] + tmp_154[0], tmp_153[1] + tmp_154[1], tmp_153[2] + tmp_154[2]];
    signal tmp_156[3] <== CMul()(tmp_717, evals[22]);
    signal tmp_157[3] <== [tmp_155[0] + tmp_156[0], tmp_155[1] + tmp_156[1], tmp_155[2] + tmp_156[2]];
    signal tmp_158[3] <== CMul()(tmp_718, evals[23]);
    signal tmp_159[3] <== [tmp_157[0] + tmp_158[0], tmp_157[1] + tmp_158[1], tmp_157[2] + tmp_158[2]];
    signal tmp_160[3] <== CMul()(tmp_719, evals[24]);
    signal tmp_161[3] <== [tmp_159[0] + tmp_160[0], tmp_159[1] + tmp_160[1], tmp_159[2] + tmp_160[2]];
    signal tmp_162[3] <== CMul()(tmp_720, evals[25]);
    signal tmp_163[3] <== [tmp_161[0] + tmp_162[0], tmp_161[1] + tmp_162[1], tmp_161[2] + tmp_162[2]];
    signal tmp_164[3] <== CMul()(tmp_721, evals[26]);
    signal tmp_165[3] <== [tmp_163[0] + tmp_164[0], tmp_163[1] + tmp_164[1], tmp_163[2] + tmp_164[2]];
    signal tmp_166[3] <== CMul()(tmp_722, evals[27]);
    signal tmp_167[3] <== [tmp_165[0] + tmp_166[0], tmp_165[1] + tmp_166[1], tmp_165[2] + tmp_166[2]];
    signal tmp_168[3] <== CMul()(tmp_723, evals[28]);
    signal tmp_169[3] <== [tmp_167[0] + tmp_168[0], tmp_167[1] + tmp_168[1], tmp_167[2] + tmp_168[2]];
    signal tmp_170[3] <== CMul()(tmp_724, evals[29]);
    signal tmp_171[3] <== [tmp_169[0] + tmp_170[0], tmp_169[1] + tmp_170[1], tmp_169[2] + tmp_170[2]];
    signal tmp_172[3] <== CMul()(tmp_725, evals[30]);
    signal tmp_173[3] <== [tmp_171[0] + tmp_172[0], tmp_171[1] + tmp_172[1], tmp_171[2] + tmp_172[2]];
    signal tmp_174[3] <== CMul()(tmp_726, evals[31]);
    signal tmp_175[3] <== [tmp_173[0] + tmp_174[0], tmp_173[1] + tmp_174[1], tmp_173[2] + tmp_174[2]];
    signal tmp_176[3] <== CMul()(tmp_727, evals[32]);
    signal tmp_177[3] <== [tmp_175[0] + tmp_176[0], tmp_175[1] + tmp_176[1], tmp_175[2] + tmp_176[2]];
    signal tmp_178[3] <== CMul()(tmp_728, evals[33]);
    signal tmp_179[3] <== [tmp_177[0] + tmp_178[0], tmp_177[1] + tmp_178[1], tmp_177[2] + tmp_178[2]];
    signal tmp_180[3] <== CMul()(tmp_729, evals[34]);
    signal tmp_181[3] <== [tmp_179[0] + tmp_180[0], tmp_179[1] + tmp_180[1], tmp_179[2] + tmp_180[2]];
    signal tmp_182[3] <== CMul()(tmp_730, evals[35]);
    signal tmp_183[3] <== [tmp_181[0] + tmp_182[0], tmp_181[1] + tmp_182[1], tmp_181[2] + tmp_182[2]];
    signal tmp_184[3] <== CMul()(tmp_731, evals[36]);
    signal tmp_185[3] <== [tmp_183[0] + tmp_184[0], tmp_183[1] + tmp_184[1], tmp_183[2] + tmp_184[2]];
    signal tmp_186[3] <== CMul()(tmp_732, evals[37]);
    signal tmp_187[3] <== [tmp_185[0] + tmp_186[0], tmp_185[1] + tmp_186[1], tmp_185[2] + tmp_186[2]];
    signal tmp_188[3] <== CMul()(tmp_733, evals[38]);
    signal tmp_189[3] <== [tmp_187[0] + tmp_188[0], tmp_187[1] + tmp_188[1], tmp_187[2] + tmp_188[2]];
    signal tmp_190[3] <== CMul()(tmp_734, evals[39]);
    signal tmp_191[3] <== [tmp_189[0] + tmp_190[0], tmp_189[1] + tmp_190[1], tmp_189[2] + tmp_190[2]];
    signal tmp_192[3] <== CMul()(tmp_735, evals[40]);
    signal tmp_193[3] <== [tmp_191[0] + tmp_192[0], tmp_191[1] + tmp_192[1], tmp_191[2] + tmp_192[2]];
    signal tmp_194[3] <== CMul()(tmp_736, evals[41]);
    signal tmp_195[3] <== [tmp_193[0] + tmp_194[0], tmp_193[1] + tmp_194[1], tmp_193[2] + tmp_194[2]];
    signal tmp_196[3] <== CMul()(tmp_737, evals[42]);
    signal tmp_197[3] <== [tmp_195[0] + tmp_196[0], tmp_195[1] + tmp_196[1], tmp_195[2] + tmp_196[2]];
    signal tmp_198[3] <== CMul()(tmp_738, evals[43]);
    signal tmp_199[3] <== [tmp_197[0] + tmp_198[0], tmp_197[1] + tmp_198[1], tmp_197[2] + tmp_198[2]];
    signal tmp_200[3] <== CMul()(tmp_739, evals[44]);
    signal tmp_201[3] <== [tmp_199[0] + tmp_200[0], tmp_199[1] + tmp_200[1], tmp_199[2] + tmp_200[2]];
    signal tmp_202[3] <== CMul()(tmp_740, evals[45]);
    signal tmp_203[3] <== [tmp_201[0] + tmp_202[0], tmp_201[1] + tmp_202[1], tmp_201[2] + tmp_202[2]];
    signal tmp_204[3] <== CMul()(tmp_741, evals[46]);
    signal tmp_205[3] <== [tmp_203[0] + tmp_204[0], tmp_203[1] + tmp_204[1], tmp_203[2] + tmp_204[2]];
    signal tmp_206[3] <== CMul()(tmp_742, evals[47]);
    signal tmp_207[3] <== [tmp_205[0] + tmp_206[0], tmp_205[1] + tmp_206[1], tmp_205[2] + tmp_206[2]];
    signal tmp_208[3] <== CMul()(tmp_743, evals[48]);
    signal tmp_209[3] <== [tmp_207[0] + tmp_208[0], tmp_207[1] + tmp_208[1], tmp_207[2] + tmp_208[2]];
    signal tmp_210[3] <== CMul()(tmp_744, evals[49]);
    signal tmp_211[3] <== [tmp_209[0] + tmp_210[0], tmp_209[1] + tmp_210[1], tmp_209[2] + tmp_210[2]];
    signal tmp_212[3] <== CMul()(tmp_745, evals[50]);
    signal tmp_213[3] <== [tmp_211[0] + tmp_212[0], tmp_211[1] + tmp_212[1], tmp_211[2] + tmp_212[2]];
    signal tmp_214[3] <== CMul()(tmp_746, evals[51]);
    signal tmp_215[3] <== [tmp_213[0] + tmp_214[0], tmp_213[1] + tmp_214[1], tmp_213[2] + tmp_214[2]];
    signal tmp_216[3] <== CMul()(tmp_747, evals[52]);
    signal tmp_217[3] <== [tmp_215[0] + tmp_216[0], tmp_215[1] + tmp_216[1], tmp_215[2] + tmp_216[2]];
    signal tmp_218[3] <== CMul()(tmp_748, evals[53]);
    signal tmp_219[3] <== [tmp_217[0] + tmp_218[0], tmp_217[1] + tmp_218[1], tmp_217[2] + tmp_218[2]];
    signal tmp_220[3] <== CMul()(tmp_749, evals[54]);
    signal tmp_221[3] <== [tmp_219[0] + tmp_220[0], tmp_219[1] + tmp_220[1], tmp_219[2] + tmp_220[2]];
    signal tmp_222[3] <== CMul()(tmp_750, evals[57]);
    signal tmp_223[3] <== [tmp_221[0] + tmp_222[0], tmp_221[1] + tmp_222[1], tmp_221[2] + tmp_222[2]];
    signal tmp_224[3] <== CMul()(tmp_751, evals[58]);
    signal tmp_225[3] <== [tmp_223[0] + tmp_224[0], tmp_223[1] + tmp_224[1], tmp_223[2] + tmp_224[2]];
    signal tmp_226[3] <== CMul()(tmp_752, evals[59]);
    signal tmp_227[3] <== [tmp_225[0] + tmp_226[0], tmp_225[1] + tmp_226[1], tmp_225[2] + tmp_226[2]];
    signal tmp_228[3] <== CMul()(tmp_753, evals[60]);
    signal tmp_229[3] <== [tmp_227[0] + tmp_228[0], tmp_227[1] + tmp_228[1], tmp_227[2] + tmp_228[2]];
    signal tmp_230[3] <== CMul()(tmp_754, evals[61]);
    signal tmp_231[3] <== [tmp_229[0] + tmp_230[0], tmp_229[1] + tmp_230[1], tmp_229[2] + tmp_230[2]];
    signal tmp_232[3] <== CMul()(tmp_755, evals[62]);
    signal tmp_233[3] <== [tmp_231[0] + tmp_232[0], tmp_231[1] + tmp_232[1], tmp_231[2] + tmp_232[2]];
    signal tmp_234[3] <== CMul()(tmp_756, evals[63]);
    signal tmp_235[3] <== [tmp_233[0] + tmp_234[0], tmp_233[1] + tmp_234[1], tmp_233[2] + tmp_234[2]];
    signal tmp_236[3] <== CMul()(tmp_757, evals[65]);
    signal tmp_237[3] <== [tmp_235[0] + tmp_236[0], tmp_235[1] + tmp_236[1], tmp_235[2] + tmp_236[2]];
    signal tmp_238[3] <== CMul()(tmp_758, evals[66]);
    signal tmp_239[3] <== [tmp_237[0] + tmp_238[0], tmp_237[1] + tmp_238[1], tmp_237[2] + tmp_238[2]];
    signal tmp_240[3] <== CMul()(tmp_759, evals[67]);
    signal tmp_241[3] <== [tmp_239[0] + tmp_240[0], tmp_239[1] + tmp_240[1], tmp_239[2] + tmp_240[2]];
    signal tmp_242[3] <== CMul()(tmp_760, evals[68]);
    signal tmp_243[3] <== [tmp_241[0] + tmp_242[0], tmp_241[1] + tmp_242[1], tmp_241[2] + tmp_242[2]];
    signal tmp_244[3] <== CMul()(tmp_761, evals[69]);
    signal tmp_245[3] <== [tmp_243[0] + tmp_244[0], tmp_243[1] + tmp_244[1], tmp_243[2] + tmp_244[2]];
    signal tmp_246[3] <== CMul()(tmp_762, evals[70]);
    signal tmp_247[3] <== [tmp_245[0] + tmp_246[0], tmp_245[1] + tmp_246[1], tmp_245[2] + tmp_246[2]];
    signal tmp_248[3] <== CMul()(tmp_763, evals[71]);
    signal tmp_249[3] <== [tmp_247[0] + tmp_248[0], tmp_247[1] + tmp_248[1], tmp_247[2] + tmp_248[2]];
    signal tmp_250[3] <== CMul()(tmp_764, evals[72]);
    signal tmp_251[3] <== [tmp_249[0] + tmp_250[0], tmp_249[1] + tmp_250[1], tmp_249[2] + tmp_250[2]];
    signal tmp_252[3] <== CMul()(tmp_765, evals[73]);
    signal tmp_253[3] <== [tmp_251[0] + tmp_252[0], tmp_251[1] + tmp_252[1], tmp_251[2] + tmp_252[2]];
    signal tmp_254[3] <== CMul()(tmp_766, evals[74]);
    signal tmp_255[3] <== [tmp_253[0] + tmp_254[0], tmp_253[1] + tmp_254[1], tmp_253[2] + tmp_254[2]];
    signal tmp_256[3] <== CMul()(tmp_767, evals[75]);
    signal tmp_257[3] <== [tmp_255[0] + tmp_256[0], tmp_255[1] + tmp_256[1], tmp_255[2] + tmp_256[2]];
    signal tmp_258[3] <== CMul()(tmp_768, evals[76]);
    signal tmp_259[3] <== [tmp_257[0] + tmp_258[0], tmp_257[1] + tmp_258[1], tmp_257[2] + tmp_258[2]];
    signal tmp_260[3] <== CMul()(tmp_769, evals[77]);
    signal tmp_261[3] <== [tmp_259[0] + tmp_260[0], tmp_259[1] + tmp_260[1], tmp_259[2] + tmp_260[2]];
    signal tmp_262[3] <== CMul()(tmp_770, evals[78]);
    signal tmp_263[3] <== [tmp_261[0] + tmp_262[0], tmp_261[1] + tmp_262[1], tmp_261[2] + tmp_262[2]];
    signal tmp_264[3] <== CMul()(tmp_771, evals[79]);
    signal tmp_265[3] <== [tmp_263[0] + tmp_264[0], tmp_263[1] + tmp_264[1], tmp_263[2] + tmp_264[2]];
    signal tmp_266[3] <== CMul()(tmp_772, evals[80]);
    signal tmp_267[3] <== [tmp_265[0] + tmp_266[0], tmp_265[1] + tmp_266[1], tmp_265[2] + tmp_266[2]];
    signal tmp_268[3] <== CMul()(tmp_773, evals[81]);
    signal tmp_269[3] <== [tmp_267[0] + tmp_268[0], tmp_267[1] + tmp_268[1], tmp_267[2] + tmp_268[2]];
    signal tmp_270[3] <== CMul()(tmp_774, evals[82]);
    signal tmp_271[3] <== [tmp_269[0] + tmp_270[0], tmp_269[1] + tmp_270[1], tmp_269[2] + tmp_270[2]];
    signal tmp_272[3] <== CMul()(tmp_775, evals[83]);
    signal tmp_273[3] <== [tmp_271[0] + tmp_272[0], tmp_271[1] + tmp_272[1], tmp_271[2] + tmp_272[2]];
    signal tmp_274[3] <== CMul()(tmp_776, evals[84]);
    signal tmp_275[3] <== [tmp_273[0] + tmp_274[0], tmp_273[1] + tmp_274[1], tmp_273[2] + tmp_274[2]];
    signal tmp_276[3] <== CMul()(tmp_777, evals[85]);
    signal tmp_277[3] <== [tmp_275[0] + tmp_276[0], tmp_275[1] + tmp_276[1], tmp_275[2] + tmp_276[2]];
    signal tmp_278[3] <== CMul()(tmp_778, evals[86]);
    signal tmp_279[3] <== [tmp_277[0] + tmp_278[0], tmp_277[1] + tmp_278[1], tmp_277[2] + tmp_278[2]];
    signal tmp_280[3] <== CMul()(tmp_779, evals[87]);
    signal tmp_281[3] <== [tmp_279[0] + tmp_280[0], tmp_279[1] + tmp_280[1], tmp_279[2] + tmp_280[2]];
    signal tmp_282[3] <== CMul()(tmp_780, evals[88]);
    signal tmp_283[3] <== [tmp_281[0] + tmp_282[0], tmp_281[1] + tmp_282[1], tmp_281[2] + tmp_282[2]];
    signal tmp_284[3] <== CMul()(tmp_781, evals[89]);
    signal tmp_285[3] <== [tmp_283[0] + tmp_284[0], tmp_283[1] + tmp_284[1], tmp_283[2] + tmp_284[2]];
    signal tmp_286[3] <== CMul()(tmp_782, evals[90]);
    signal tmp_287[3] <== [tmp_285[0] + tmp_286[0], tmp_285[1] + tmp_286[1], tmp_285[2] + tmp_286[2]];
    signal tmp_288[3] <== CMul()(tmp_783, evals[91]);
    signal tmp_289[3] <== [tmp_287[0] + tmp_288[0], tmp_287[1] + tmp_288[1], tmp_287[2] + tmp_288[2]];
    signal tmp_290[3] <== CMul()(tmp_784, evals[92]);
    signal tmp_291[3] <== [tmp_289[0] + tmp_290[0], tmp_289[1] + tmp_290[1], tmp_289[2] + tmp_290[2]];
    signal tmp_292[3] <== CMul()(tmp_785, evals[93]);
    signal tmp_293[3] <== [tmp_291[0] + tmp_292[0], tmp_291[1] + tmp_292[1], tmp_291[2] + tmp_292[2]];
    signal tmp_294[3] <== CMul()(tmp_786, evals[94]);
    signal tmp_295[3] <== [tmp_293[0] + tmp_294[0], tmp_293[1] + tmp_294[1], tmp_293[2] + tmp_294[2]];
    signal tmp_296[3] <== CMul()(tmp_787, evals[95]);
    signal tmp_297[3] <== [tmp_295[0] + tmp_296[0], tmp_295[1] + tmp_296[1], tmp_295[2] + tmp_296[2]];
    signal tmp_298[3] <== CMul()(tmp_788, evals[96]);
    signal tmp_299[3] <== [tmp_297[0] + tmp_298[0], tmp_297[1] + tmp_298[1], tmp_297[2] + tmp_298[2]];
    signal tmp_300[3] <== CMul()(tmp_789, evals[97]);
    signal tmp_301[3] <== [tmp_299[0] + tmp_300[0], tmp_299[1] + tmp_300[1], tmp_299[2] + tmp_300[2]];
    signal tmp_302[3] <== CMul()(tmp_790, evals[98]);
    signal tmp_303[3] <== [tmp_301[0] + tmp_302[0], tmp_301[1] + tmp_302[1], tmp_301[2] + tmp_302[2]];
    signal tmp_304[3] <== CMul()(tmp_791, evals[99]);
    signal tmp_305[3] <== [tmp_303[0] + tmp_304[0], tmp_303[1] + tmp_304[1], tmp_303[2] + tmp_304[2]];
    signal tmp_306[3] <== CMul()(tmp_792, evals[100]);
    signal tmp_307[3] <== [tmp_305[0] + tmp_306[0], tmp_305[1] + tmp_306[1], tmp_305[2] + tmp_306[2]];
    signal tmp_308[3] <== CMul()(tmp_793, evals[101]);
    signal tmp_309[3] <== [tmp_307[0] + tmp_308[0], tmp_307[1] + tmp_308[1], tmp_307[2] + tmp_308[2]];
    signal tmp_310[3] <== CMul()(tmp_794, evals[102]);
    signal tmp_311[3] <== [tmp_309[0] + tmp_310[0], tmp_309[1] + tmp_310[1], tmp_309[2] + tmp_310[2]];
    signal tmp_312[3] <== CMul()(tmp_795, evals[103]);
    signal tmp_313[3] <== [tmp_311[0] + tmp_312[0], tmp_311[1] + tmp_312[1], tmp_311[2] + tmp_312[2]];
    signal tmp_314[3] <== CMul()(tmp_796, evals[104]);
    signal tmp_315[3] <== [tmp_313[0] + tmp_314[0], tmp_313[1] + tmp_314[1], tmp_313[2] + tmp_314[2]];
    signal tmp_316[3] <== CMul()(tmp_797, evals[105]);
    signal tmp_317[3] <== [tmp_315[0] + tmp_316[0], tmp_315[1] + tmp_316[1], tmp_315[2] + tmp_316[2]];
    signal tmp_318[3] <== CMul()(tmp_798, evals[106]);
    signal tmp_319[3] <== [tmp_317[0] + tmp_318[0], tmp_317[1] + tmp_318[1], tmp_317[2] + tmp_318[2]];
    signal tmp_320[3] <== CMul()(tmp_799, evals[107]);
    signal tmp_321[3] <== [tmp_319[0] + tmp_320[0], tmp_319[1] + tmp_320[1], tmp_319[2] + tmp_320[2]];
    signal tmp_322[3] <== CMul()(tmp_800, evals[108]);
    signal tmp_323[3] <== [tmp_321[0] + tmp_322[0], tmp_321[1] + tmp_322[1], tmp_321[2] + tmp_322[2]];
    signal tmp_324[3] <== CMul()(tmp_801, evals[109]);
    signal tmp_325[3] <== [tmp_323[0] + tmp_324[0], tmp_323[1] + tmp_324[1], tmp_323[2] + tmp_324[2]];
    signal tmp_326[3] <== CMul()(tmp_802, evals[110]);
    signal tmp_327[3] <== [tmp_325[0] + tmp_326[0], tmp_325[1] + tmp_326[1], tmp_325[2] + tmp_326[2]];
    signal tmp_328[3] <== CMul()(tmp_803, evals[111]);
    signal tmp_329[3] <== [tmp_327[0] + tmp_328[0], tmp_327[1] + tmp_328[1], tmp_327[2] + tmp_328[2]];
    signal tmp_330[3] <== CMul()(tmp_804, evals[112]);
    signal tmp_331[3] <== [tmp_329[0] + tmp_330[0], tmp_329[1] + tmp_330[1], tmp_329[2] + tmp_330[2]];
    signal tmp_332[3] <== CMul()(tmp_805, evals[113]);
    signal tmp_333[3] <== [tmp_331[0] + tmp_332[0], tmp_331[1] + tmp_332[1], tmp_331[2] + tmp_332[2]];
    signal tmp_334[3] <== CMul()(tmp_806, evals[114]);
    signal tmp_335[3] <== [tmp_333[0] + tmp_334[0], tmp_333[1] + tmp_334[1], tmp_333[2] + tmp_334[2]];
    signal tmp_336[3] <== CMul()(tmp_807, evals[116]);
    signal tmp_337[3] <== [tmp_335[0] + tmp_336[0], tmp_335[1] + tmp_336[1], tmp_335[2] + tmp_336[2]];
    signal tmp_338[3] <== CMul()(tmp_808, evals[117]);
    signal tmp_339[3] <== [tmp_337[0] + tmp_338[0], tmp_337[1] + tmp_338[1], tmp_337[2] + tmp_338[2]];
    signal tmp_340[3] <== CMul()(tmp_809, evals[118]);
    signal tmp_341[3] <== [tmp_339[0] + tmp_340[0], tmp_339[1] + tmp_340[1], tmp_339[2] + tmp_340[2]];
    signal tmp_342[3] <== CMul()(tmp_810, evals[119]);
    signal tmp_343[3] <== [tmp_341[0] + tmp_342[0], tmp_341[1] + tmp_342[1], tmp_341[2] + tmp_342[2]];
    signal tmp_344[3] <== CMul()(tmp_811, evals[120]);
    signal tmp_345[3] <== [tmp_343[0] + tmp_344[0], tmp_343[1] + tmp_344[1], tmp_343[2] + tmp_344[2]];
    signal tmp_346[3] <== CMul()(tmp_812, evals[121]);
    signal tmp_347[3] <== [tmp_345[0] + tmp_346[0], tmp_345[1] + tmp_346[1], tmp_345[2] + tmp_346[2]];
    signal tmp_348[3] <== CMul()(tmp_813, evals[122]);
    signal tmp_349[3] <== [tmp_347[0] + tmp_348[0], tmp_347[1] + tmp_348[1], tmp_347[2] + tmp_348[2]];
    signal tmp_350[3] <== CMul()(tmp_814, evals[123]);
    signal tmp_351[3] <== [tmp_349[0] + tmp_350[0], tmp_349[1] + tmp_350[1], tmp_349[2] + tmp_350[2]];
    signal tmp_352[3] <== CMul()(tmp_815, evals[124]);
    signal tmp_353[3] <== [tmp_351[0] + tmp_352[0], tmp_351[1] + tmp_352[1], tmp_351[2] + tmp_352[2]];
    signal tmp_354[3] <== CMul()(tmp_816, evals[125]);
    tmp_820 <== [tmp_353[0] + tmp_354[0], tmp_353[1] + tmp_354[1], tmp_353[2] + tmp_354[2]];
    signal tmp_356[3] <== CMul()(tmp_750, evals[126]);
    signal tmp_357[3] <== CMul()(tmp_758, evals[127]);
    signal tmp_358[3] <== [tmp_356[0] + tmp_357[0], tmp_356[1] + tmp_357[1], tmp_356[2] + tmp_357[2]];
    signal tmp_359[3] <== CMul()(tmp_759, evals[128]);
    signal tmp_360[3] <== [tmp_358[0] + tmp_359[0], tmp_358[1] + tmp_359[1], tmp_358[2] + tmp_359[2]];
    signal tmp_361[3] <== CMul()(tmp_760, evals[129]);
    signal tmp_362[3] <== [tmp_360[0] + tmp_361[0], tmp_360[1] + tmp_361[1], tmp_360[2] + tmp_361[2]];
    signal tmp_363[3] <== CMul()(tmp_761, evals[130]);
    signal tmp_364[3] <== [tmp_362[0] + tmp_363[0], tmp_362[1] + tmp_363[1], tmp_362[2] + tmp_363[2]];
    signal tmp_365[3] <== CMul()(tmp_762, evals[131]);
    signal tmp_366[3] <== [tmp_364[0] + tmp_365[0], tmp_364[1] + tmp_365[1], tmp_364[2] + tmp_365[2]];
    signal tmp_367[3] <== CMul()(tmp_763, evals[132]);
    signal tmp_368[3] <== [tmp_366[0] + tmp_367[0], tmp_366[1] + tmp_367[1], tmp_366[2] + tmp_367[2]];
    signal tmp_369[3] <== CMul()(tmp_764, evals[133]);
    signal tmp_370[3] <== [tmp_368[0] + tmp_369[0], tmp_368[1] + tmp_369[1], tmp_368[2] + tmp_369[2]];
    signal tmp_371[3] <== CMul()(tmp_774, evals[134]);
    signal tmp_372[3] <== [tmp_370[0] + tmp_371[0], tmp_370[1] + tmp_371[1], tmp_370[2] + tmp_371[2]];
    signal tmp_373[3] <== CMul()(tmp_775, evals[135]);
    signal tmp_374[3] <== [tmp_372[0] + tmp_373[0], tmp_372[1] + tmp_373[1], tmp_372[2] + tmp_373[2]];
    signal tmp_375[3] <== CMul()(tmp_776, evals[136]);
    signal tmp_376[3] <== [tmp_374[0] + tmp_375[0], tmp_374[1] + tmp_375[1], tmp_374[2] + tmp_375[2]];
    signal tmp_377[3] <== CMul()(tmp_777, evals[137]);
    signal tmp_378[3] <== [tmp_376[0] + tmp_377[0], tmp_376[1] + tmp_377[1], tmp_376[2] + tmp_377[2]];
    signal tmp_379[3] <== CMul()(tmp_778, evals[138]);
    signal tmp_380[3] <== [tmp_378[0] + tmp_379[0], tmp_378[1] + tmp_379[1], tmp_378[2] + tmp_379[2]];
    signal tmp_381[3] <== CMul()(tmp_779, evals[139]);
    signal tmp_382[3] <== [tmp_380[0] + tmp_381[0], tmp_380[1] + tmp_381[1], tmp_380[2] + tmp_381[2]];
    signal tmp_383[3] <== CMul()(tmp_780, evals[140]);
    signal tmp_384[3] <== [tmp_382[0] + tmp_383[0], tmp_382[1] + tmp_383[1], tmp_382[2] + tmp_383[2]];
    signal tmp_385[3] <== CMul()(tmp_781, evals[141]);
    signal tmp_386[3] <== [tmp_384[0] + tmp_385[0], tmp_384[1] + tmp_385[1], tmp_384[2] + tmp_385[2]];
    signal tmp_387[3] <== CMul()(tmp_782, evals[142]);
    signal tmp_388[3] <== [tmp_386[0] + tmp_387[0], tmp_386[1] + tmp_387[1], tmp_386[2] + tmp_387[2]];
    signal tmp_389[3] <== CMul()(tmp_783, evals[143]);
    signal tmp_390[3] <== [tmp_388[0] + tmp_389[0], tmp_388[1] + tmp_389[1], tmp_388[2] + tmp_389[2]];
    signal tmp_391[3] <== CMul()(tmp_784, evals[144]);
    signal tmp_392[3] <== [tmp_390[0] + tmp_391[0], tmp_390[1] + tmp_391[1], tmp_390[2] + tmp_391[2]];
    signal tmp_393[3] <== CMul()(tmp_785, evals[145]);
    signal tmp_394[3] <== [tmp_392[0] + tmp_393[0], tmp_392[1] + tmp_393[1], tmp_392[2] + tmp_393[2]];
    signal tmp_395[3] <== CMul()(tmp_786, evals[146]);
    signal tmp_396[3] <== [tmp_394[0] + tmp_395[0], tmp_394[1] + tmp_395[1], tmp_394[2] + tmp_395[2]];
    signal tmp_397[3] <== CMul()(tmp_787, evals[147]);
    signal tmp_398[3] <== [tmp_396[0] + tmp_397[0], tmp_396[1] + tmp_397[1], tmp_396[2] + tmp_397[2]];
    signal tmp_399[3] <== CMul()(tmp_788, evals[148]);
    signal tmp_400[3] <== [tmp_398[0] + tmp_399[0], tmp_398[1] + tmp_399[1], tmp_398[2] + tmp_399[2]];
    signal tmp_401[3] <== CMul()(tmp_789, evals[149]);
    signal tmp_402[3] <== [tmp_400[0] + tmp_401[0], tmp_400[1] + tmp_401[1], tmp_400[2] + tmp_401[2]];
    signal tmp_403[3] <== CMul()(tmp_790, evals[150]);
    tmp_821 <== [tmp_402[0] + tmp_403[0], tmp_402[1] + tmp_403[1], tmp_402[2] + tmp_403[2]];
    tmp_822 <== CMul()(tmp_774, evals[151]);
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
    signal input evals[152][3];
    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[46];

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
    signal tmp_680[3] <== friConsts[2];
    signal tmp_681[3] <== friConsts[3];
    signal tmp_682[3] <== friConsts[4];
    signal tmp_683[3] <== friConsts[5];
    signal tmp_684[3] <== friConsts[6];
    signal tmp_685[3] <== friConsts[7];
    signal tmp_686[3] <== friConsts[8];
    signal tmp_687[3] <== friConsts[9];
    signal tmp_688[3] <== friConsts[10];
    signal tmp_689[3] <== friConsts[11];
    signal tmp_690[3] <== friConsts[12];
    signal tmp_691[3] <== friConsts[13];
    signal tmp_692[3] <== friConsts[14];
    signal tmp_693[3] <== friConsts[15];
    signal tmp_694[3] <== friConsts[16];
    signal tmp_695[3] <== friConsts[17];
    signal tmp_696[3] <== friConsts[18];
    signal tmp_714[3] <== friConsts[19];
    signal tmp_715[3] <== friConsts[20];
    signal tmp_716[3] <== friConsts[21];
    signal tmp_717[3] <== friConsts[22];
    signal tmp_718[3] <== friConsts[23];
    signal tmp_719[3] <== friConsts[24];
    signal tmp_720[3] <== friConsts[25];
    signal tmp_721[3] <== friConsts[26];
    signal tmp_722[3] <== friConsts[27];
    signal tmp_723[3] <== friConsts[28];
    signal tmp_724[3] <== friConsts[29];
    signal tmp_725[3] <== friConsts[30];
    signal tmp_726[3] <== friConsts[31];
    signal tmp_727[3] <== friConsts[32];
    signal tmp_728[3] <== friConsts[33];
    signal tmp_729[3] <== friConsts[34];
    signal tmp_730[3] <== friConsts[35];
    signal tmp_731[3] <== friConsts[36];
    signal tmp_732[3] <== friConsts[37];
    signal tmp_733[3] <== friConsts[38];
    signal tmp_734[3] <== friConsts[39];
    signal tmp_735[3] <== friConsts[40];
    signal tmp_736[3] <== friConsts[41];
    signal tmp_737[3] <== friConsts[42];
    signal tmp_738[3] <== friConsts[43];
    signal tmp_739[3] <== friConsts[44];
    signal tmp_740[3] <== friConsts[45];
    signal tmp_741[3] <== friConsts[46];
    signal tmp_742[3] <== friConsts[47];
    signal tmp_743[3] <== friConsts[48];
    signal tmp_744[3] <== friConsts[49];
    signal tmp_745[3] <== friConsts[50];
    signal tmp_746[3] <== friConsts[51];
    signal tmp_747[3] <== friConsts[52];
    signal tmp_748[3] <== friConsts[53];
    signal tmp_749[3] <== friConsts[54];
    signal tmp_750[3] <== friConsts[55];
    signal tmp_751[3] <== friConsts[56];
    signal tmp_752[3] <== friConsts[57];
    signal tmp_753[3] <== friConsts[58];
    signal tmp_754[3] <== friConsts[59];
    signal tmp_755[3] <== friConsts[60];
    signal tmp_756[3] <== friConsts[61];
    signal tmp_757[3] <== friConsts[62];
    signal tmp_758[3] <== friConsts[63];
    signal tmp_759[3] <== friConsts[64];
    signal tmp_760[3] <== friConsts[65];
    signal tmp_761[3] <== friConsts[66];
    signal tmp_762[3] <== friConsts[67];
    signal tmp_763[3] <== friConsts[68];
    signal tmp_764[3] <== friConsts[69];
    signal tmp_765[3] <== friConsts[70];
    signal tmp_766[3] <== friConsts[71];
    signal tmp_767[3] <== friConsts[72];
    signal tmp_768[3] <== friConsts[73];
    signal tmp_769[3] <== friConsts[74];
    signal tmp_770[3] <== friConsts[75];
    signal tmp_771[3] <== friConsts[76];
    signal tmp_772[3] <== friConsts[77];
    signal tmp_773[3] <== friConsts[78];
    signal tmp_774[3] <== friConsts[79];
    signal tmp_775[3] <== friConsts[80];
    signal tmp_776[3] <== friConsts[81];
    signal tmp_777[3] <== friConsts[82];
    signal tmp_778[3] <== friConsts[83];
    signal tmp_779[3] <== friConsts[84];
    signal tmp_780[3] <== friConsts[85];
    signal tmp_781[3] <== friConsts[86];
    signal tmp_782[3] <== friConsts[87];
    signal tmp_783[3] <== friConsts[88];
    signal tmp_784[3] <== friConsts[89];
    signal tmp_785[3] <== friConsts[90];
    signal tmp_786[3] <== friConsts[91];
    signal tmp_787[3] <== friConsts[92];
    signal tmp_788[3] <== friConsts[93];
    signal tmp_789[3] <== friConsts[94];
    signal tmp_790[3] <== friConsts[95];
    signal tmp_791[3] <== friConsts[96];
    signal tmp_792[3] <== friConsts[97];
    signal tmp_793[3] <== friConsts[98];
    signal tmp_794[3] <== friConsts[99];
    signal tmp_795[3] <== friConsts[100];
    signal tmp_796[3] <== friConsts[101];
    signal tmp_797[3] <== friConsts[102];
    signal tmp_798[3] <== friConsts[103];
    signal tmp_799[3] <== friConsts[104];
    signal tmp_800[3] <== friConsts[105];
    signal tmp_801[3] <== friConsts[106];
    signal tmp_802[3] <== friConsts[107];
    signal tmp_803[3] <== friConsts[108];
    signal tmp_804[3] <== friConsts[109];
    signal tmp_805[3] <== friConsts[110];
    signal tmp_806[3] <== friConsts[111];
    signal tmp_807[3] <== friConsts[112];
    signal tmp_808[3] <== friConsts[113];
    signal tmp_809[3] <== friConsts[114];
    signal tmp_810[3] <== friConsts[115];
    signal tmp_811[3] <== friConsts[116];
    signal tmp_812[3] <== friConsts[117];
    signal tmp_813[3] <== friConsts[118];
    signal tmp_814[3] <== friConsts[119];
    signal tmp_815[3] <== friConsts[120];
    signal tmp_816[3] <== friConsts[121];
    signal tmp_818[3] <== friConsts[122];
    signal tmp_819[3] <== friConsts[123];
    signal tmp_820[3] <== friConsts[124];
    signal tmp_821[3] <== friConsts[125];
    signal tmp_822[3] <== friConsts[126];

    signal tmp_823[3];
    (tmp_823) <== CalculateDeepPolChunks0_0()(challengesDeep,evals,cm1,cm2,cm3,consts,xDivXSubXi,tmp_680,tmp_681,tmp_682,tmp_683,tmp_684,tmp_685,tmp_686,tmp_687,tmp_688,tmp_689,tmp_690,tmp_691,tmp_692,tmp_693,tmp_694,tmp_695,tmp_696,tmp_714,tmp_715,tmp_716,tmp_717,tmp_718,tmp_719,tmp_720,tmp_721,tmp_722,tmp_723,tmp_724,tmp_725,tmp_726,tmp_727,tmp_728,tmp_729,tmp_730,tmp_731,tmp_732,tmp_733,tmp_734,tmp_735,tmp_736,tmp_737,tmp_738,tmp_739,tmp_740,tmp_741,tmp_742,tmp_743,tmp_744,tmp_745,tmp_746,tmp_747,tmp_748,tmp_749,tmp_750,tmp_751,tmp_752,tmp_753,tmp_754,tmp_755,tmp_756,tmp_757,tmp_758,tmp_759,tmp_760,tmp_761,tmp_762,tmp_763,tmp_764,tmp_765,tmp_766,tmp_767,tmp_768,tmp_769,tmp_770,tmp_771,tmp_772,tmp_773,tmp_774,tmp_775,tmp_776,tmp_777,tmp_778,tmp_779,tmp_780,tmp_781,tmp_782,tmp_783,tmp_784,tmp_785,tmp_786,tmp_787,tmp_788,tmp_789,tmp_790,tmp_791,tmp_792,tmp_793,tmp_794,tmp_795,tmp_796,tmp_797,tmp_798,tmp_799,tmp_800,tmp_801,tmp_802,tmp_803,tmp_804,tmp_805,tmp_806,tmp_807,tmp_808,tmp_809,tmp_810,tmp_811,tmp_812,tmp_813,tmp_814,tmp_815,tmp_816,tmp_818,tmp_819,tmp_820,tmp_821,tmp_822);

    signal queryVals[3];
    queryVals[0] <== tmp_823[0];
    queryVals[1] <== tmp_823[1];
    queryVals[2] <== tmp_823[2];

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
    signal input evals[152][3];
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

    signal input s0_valsC[46];
    signal input s0_valsC_p[46][1];
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
    VerifyMerkleHashUntilLevel(1, 46, 4, 8, 2, 1048576)(s0_valsC_p, s0_siblingsC, queriesL0Bits, s0_last_mt_levelsC, enable);


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
    signal input evals[152][3];
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

    signal input s0_valsC[nQueries][46];
    signal input s0_valsC_p[nQueries][46][1];
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
    Verifies one STARK proof of the air 'FiboCPU_FibonacciSquare_recursive1' (Poseidon1 transcript and Merkle trees):
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

    
    signal input evals[152][3]; // Evaluations of commited polynomials at the out-of-domain sample point z and shifts gˢ·z

    // Leaf values of the merkle tree used to check all the queries
    signal input s0_vals1[72][48];
    signal input s0_vals2[72][12];
    signal input s0_vals3[72][21];
    signal input s0_valsC[72][46];


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
    signal tmp_680[3];
    signal tmp_681[3];
    signal tmp_682[3];
    signal tmp_683[3];
    signal tmp_684[3];
    signal tmp_685[3];
    signal tmp_686[3];
    signal tmp_687[3];
    signal tmp_688[3];
    signal tmp_689[3];
    signal tmp_690[3];
    signal tmp_691[3];
    signal tmp_692[3];
    signal tmp_693[3];
    signal tmp_694[3];
    signal tmp_695[3];
    signal tmp_696[3];
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
    signal tmp_802[3];
    signal tmp_803[3];
    signal tmp_804[3];
    signal tmp_805[3];
    signal tmp_806[3];
    signal tmp_807[3];
    signal tmp_808[3];
    signal tmp_809[3];
    signal tmp_810[3];
    signal tmp_811[3];
    signal tmp_812[3];
    signal tmp_813[3];
    signal tmp_814[3];
    signal tmp_815[3];
    signal tmp_816[3];
    signal tmp_818[3];
    signal tmp_819[3];
    signal tmp_820[3];
    signal tmp_821[3];
    signal tmp_822[3];
    (tmp_680,tmp_681,tmp_682,tmp_683,tmp_684,tmp_685,tmp_686,tmp_687,tmp_688,tmp_689,tmp_690,tmp_691,tmp_692,tmp_693,tmp_694,tmp_695,tmp_696,tmp_714,tmp_715,tmp_716,tmp_717,tmp_718,tmp_719,tmp_720,tmp_721,tmp_722,tmp_723,tmp_724,tmp_725,tmp_726,tmp_727,tmp_728,tmp_729,tmp_730,tmp_731,tmp_732,tmp_733,tmp_734,tmp_735,tmp_736,tmp_737,tmp_738,tmp_739,tmp_740,tmp_741,tmp_742,tmp_743,tmp_744,tmp_745,tmp_746,tmp_747,tmp_748,tmp_749,tmp_750,tmp_751,tmp_752,tmp_753,tmp_754,tmp_755,tmp_756,tmp_757,tmp_758,tmp_759,tmp_760,tmp_761,tmp_762,tmp_763,tmp_764,tmp_765,tmp_766,tmp_767,tmp_768,tmp_769,tmp_770,tmp_771,tmp_772,tmp_773,tmp_774,tmp_775,tmp_776,tmp_777,tmp_778,tmp_779,tmp_780,tmp_781,tmp_782,tmp_783,tmp_784,tmp_785,tmp_786,tmp_787,tmp_788,tmp_789,tmp_790,tmp_791,tmp_792,tmp_793,tmp_794,tmp_795,tmp_796,tmp_797,tmp_798,tmp_799,tmp_800,tmp_801,tmp_802,tmp_803,tmp_804,tmp_805,tmp_806,tmp_807,tmp_808,tmp_809,tmp_810,tmp_811,tmp_812,tmp_813,tmp_814,tmp_815,tmp_816,tmp_818,tmp_819,tmp_820,tmp_821,tmp_822) <== CalculateFRIConstantsChunks0()(challengesDeep,evals);

    signal friConsts[127][3];
    friConsts[0] <== challengesDeep[0];
    friConsts[1] <== challengesDeep[1];
    friConsts[2] <== tmp_680;
    friConsts[3] <== tmp_681;
    friConsts[4] <== tmp_682;
    friConsts[5] <== tmp_683;
    friConsts[6] <== tmp_684;
    friConsts[7] <== tmp_685;
    friConsts[8] <== tmp_686;
    friConsts[9] <== tmp_687;
    friConsts[10] <== tmp_688;
    friConsts[11] <== tmp_689;
    friConsts[12] <== tmp_690;
    friConsts[13] <== tmp_691;
    friConsts[14] <== tmp_692;
    friConsts[15] <== tmp_693;
    friConsts[16] <== tmp_694;
    friConsts[17] <== tmp_695;
    friConsts[18] <== tmp_696;
    friConsts[19] <== tmp_714;
    friConsts[20] <== tmp_715;
    friConsts[21] <== tmp_716;
    friConsts[22] <== tmp_717;
    friConsts[23] <== tmp_718;
    friConsts[24] <== tmp_719;
    friConsts[25] <== tmp_720;
    friConsts[26] <== tmp_721;
    friConsts[27] <== tmp_722;
    friConsts[28] <== tmp_723;
    friConsts[29] <== tmp_724;
    friConsts[30] <== tmp_725;
    friConsts[31] <== tmp_726;
    friConsts[32] <== tmp_727;
    friConsts[33] <== tmp_728;
    friConsts[34] <== tmp_729;
    friConsts[35] <== tmp_730;
    friConsts[36] <== tmp_731;
    friConsts[37] <== tmp_732;
    friConsts[38] <== tmp_733;
    friConsts[39] <== tmp_734;
    friConsts[40] <== tmp_735;
    friConsts[41] <== tmp_736;
    friConsts[42] <== tmp_737;
    friConsts[43] <== tmp_738;
    friConsts[44] <== tmp_739;
    friConsts[45] <== tmp_740;
    friConsts[46] <== tmp_741;
    friConsts[47] <== tmp_742;
    friConsts[48] <== tmp_743;
    friConsts[49] <== tmp_744;
    friConsts[50] <== tmp_745;
    friConsts[51] <== tmp_746;
    friConsts[52] <== tmp_747;
    friConsts[53] <== tmp_748;
    friConsts[54] <== tmp_749;
    friConsts[55] <== tmp_750;
    friConsts[56] <== tmp_751;
    friConsts[57] <== tmp_752;
    friConsts[58] <== tmp_753;
    friConsts[59] <== tmp_754;
    friConsts[60] <== tmp_755;
    friConsts[61] <== tmp_756;
    friConsts[62] <== tmp_757;
    friConsts[63] <== tmp_758;
    friConsts[64] <== tmp_759;
    friConsts[65] <== tmp_760;
    friConsts[66] <== tmp_761;
    friConsts[67] <== tmp_762;
    friConsts[68] <== tmp_763;
    friConsts[69] <== tmp_764;
    friConsts[70] <== tmp_765;
    friConsts[71] <== tmp_766;
    friConsts[72] <== tmp_767;
    friConsts[73] <== tmp_768;
    friConsts[74] <== tmp_769;
    friConsts[75] <== tmp_770;
    friConsts[76] <== tmp_771;
    friConsts[77] <== tmp_772;
    friConsts[78] <== tmp_773;
    friConsts[79] <== tmp_774;
    friConsts[80] <== tmp_775;
    friConsts[81] <== tmp_776;
    friConsts[82] <== tmp_777;
    friConsts[83] <== tmp_778;
    friConsts[84] <== tmp_779;
    friConsts[85] <== tmp_780;
    friConsts[86] <== tmp_781;
    friConsts[87] <== tmp_782;
    friConsts[88] <== tmp_783;
    friConsts[89] <== tmp_784;
    friConsts[90] <== tmp_785;
    friConsts[91] <== tmp_786;
    friConsts[92] <== tmp_787;
    friConsts[93] <== tmp_788;
    friConsts[94] <== tmp_789;
    friConsts[95] <== tmp_790;
    friConsts[96] <== tmp_791;
    friConsts[97] <== tmp_792;
    friConsts[98] <== tmp_793;
    friConsts[99] <== tmp_794;
    friConsts[100] <== tmp_795;
    friConsts[101] <== tmp_796;
    friConsts[102] <== tmp_797;
    friConsts[103] <== tmp_798;
    friConsts[104] <== tmp_799;
    friConsts[105] <== tmp_800;
    friConsts[106] <== tmp_801;
    friConsts[107] <== tmp_802;
    friConsts[108] <== tmp_803;
    friConsts[109] <== tmp_804;
    friConsts[110] <== tmp_805;
    friConsts[111] <== tmp_806;
    friConsts[112] <== tmp_807;
    friConsts[113] <== tmp_808;
    friConsts[114] <== tmp_809;
    friConsts[115] <== tmp_810;
    friConsts[116] <== tmp_811;
    friConsts[117] <== tmp_812;
    friConsts[118] <== tmp_813;
    friConsts[119] <== tmp_814;
    friConsts[120] <== tmp_815;
    friConsts[121] <== tmp_816;
    friConsts[122] <== tmp_818;
    friConsts[123] <== tmp_819;
    friConsts[124] <== tmp_820;
    friConsts[125] <== tmp_821;
    friConsts[126] <== tmp_822;

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
    var s0_valsC_p[72][46][1];
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
        for (var i = 0; i < 46; i++) {
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
    signal batch_work_s0_valsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][46];
    signal batch_work_s0_valsC_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][46][1];
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
            for (var i = 0; i < 46; i++) {
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
    signal remainder_s0_valsC[LAST_BATCH_SIZE_SAFE][46];
    signal remainder_s0_valsC_p[LAST_BATCH_SIZE_SAFE][46][1];
    signal remainder_s0_siblingsC[LAST_BATCH_SIZE_SAFE][8][12];

    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 21; i++) {
            remainder_s0_vals3[q][i] <== s0_vals3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals3_p[q][i][0] <== s0_vals3_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var i = 0; i < 46; i++) {
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

