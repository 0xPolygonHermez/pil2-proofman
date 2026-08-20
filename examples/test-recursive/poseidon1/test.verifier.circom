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
    Calculate FRI Queries
*/
template calculateFRIQueries0() {
    
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queriesFRI[73][20];

    VerifyPoW(20)(challengeFRIQueries, nonce, enable);


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
    signal {binary} transcriptN2b_23[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[7]);
    for(var i = 8; i < 16; i++){
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

    for(var j = 0; j < 11; j++) {
        queriesFRI[q][b] <== transcriptN2b_23[j];
        b++;
        if(b == 20) {
            b = 0; 
            q++;
        }
    }
    for(var j = 11; j < 64; j++) {
        _ <== transcriptN2b_23[j]; // Unused bits        
    }
}

/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input publics[395];
    signal input rootC[4];
    signal input root1[4];



    signal input root2[4];
    signal input root3[4];
    signal input evals[135][3]; 
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
    signal output {binary} queriesFRI[73][20];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];



    signal transcriptHash_publics_0[16] <== Poseidon(16)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7],publics[8],publics[9],publics[10],publics[11]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_0[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_1[16] <== Poseidon(16)([publics[12],publics[13],publics[14],publics[15],publics[16],publics[17],publics[18],publics[19],publics[20],publics[21],publics[22],publics[23]], [transcriptHash_publics_0[0],transcriptHash_publics_0[1],transcriptHash_publics_0[2],transcriptHash_publics_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_1[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_2[16] <== Poseidon(16)([publics[24],publics[25],publics[26],publics[27],publics[28],publics[29],publics[30],publics[31],publics[32],publics[33],publics[34],publics[35]], [transcriptHash_publics_1[0],transcriptHash_publics_1[1],transcriptHash_publics_1[2],transcriptHash_publics_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_2[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_3[16] <== Poseidon(16)([publics[36],publics[37],publics[38],publics[39],publics[40],publics[41],publics[42],publics[43],publics[44],publics[45],publics[46],publics[47]], [transcriptHash_publics_2[0],transcriptHash_publics_2[1],transcriptHash_publics_2[2],transcriptHash_publics_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_3[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_4[16] <== Poseidon(16)([publics[48],publics[49],publics[50],publics[51],publics[52],publics[53],publics[54],publics[55],publics[56],publics[57],publics[58],publics[59]], [transcriptHash_publics_3[0],transcriptHash_publics_3[1],transcriptHash_publics_3[2],transcriptHash_publics_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_4[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_5[16] <== Poseidon(16)([publics[60],publics[61],publics[62],publics[63],publics[64],publics[65],publics[66],publics[67],publics[68],publics[69],publics[70],publics[71]], [transcriptHash_publics_4[0],transcriptHash_publics_4[1],transcriptHash_publics_4[2],transcriptHash_publics_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_5[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_6[16] <== Poseidon(16)([publics[72],publics[73],publics[74],publics[75],publics[76],publics[77],publics[78],publics[79],publics[80],publics[81],publics[82],publics[83]], [transcriptHash_publics_5[0],transcriptHash_publics_5[1],transcriptHash_publics_5[2],transcriptHash_publics_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_6[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_7[16] <== Poseidon(16)([publics[84],publics[85],publics[86],publics[87],publics[88],publics[89],publics[90],publics[91],publics[92],publics[93],publics[94],publics[95]], [transcriptHash_publics_6[0],transcriptHash_publics_6[1],transcriptHash_publics_6[2],transcriptHash_publics_6[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_7[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_8[16] <== Poseidon(16)([publics[96],publics[97],publics[98],publics[99],publics[100],publics[101],publics[102],publics[103],publics[104],publics[105],publics[106],publics[107]], [transcriptHash_publics_7[0],transcriptHash_publics_7[1],transcriptHash_publics_7[2],transcriptHash_publics_7[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_8[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_9[16] <== Poseidon(16)([publics[108],publics[109],publics[110],publics[111],publics[112],publics[113],publics[114],publics[115],publics[116],publics[117],publics[118],publics[119]], [transcriptHash_publics_8[0],transcriptHash_publics_8[1],transcriptHash_publics_8[2],transcriptHash_publics_8[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_9[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_10[16] <== Poseidon(16)([publics[120],publics[121],publics[122],publics[123],publics[124],publics[125],publics[126],publics[127],publics[128],publics[129],publics[130],publics[131]], [transcriptHash_publics_9[0],transcriptHash_publics_9[1],transcriptHash_publics_9[2],transcriptHash_publics_9[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_10[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_11[16] <== Poseidon(16)([publics[132],publics[133],publics[134],publics[135],publics[136],publics[137],publics[138],publics[139],publics[140],publics[141],publics[142],publics[143]], [transcriptHash_publics_10[0],transcriptHash_publics_10[1],transcriptHash_publics_10[2],transcriptHash_publics_10[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_11[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_12[16] <== Poseidon(16)([publics[144],publics[145],publics[146],publics[147],publics[148],publics[149],publics[150],publics[151],publics[152],publics[153],publics[154],publics[155]], [transcriptHash_publics_11[0],transcriptHash_publics_11[1],transcriptHash_publics_11[2],transcriptHash_publics_11[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_12[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_13[16] <== Poseidon(16)([publics[156],publics[157],publics[158],publics[159],publics[160],publics[161],publics[162],publics[163],publics[164],publics[165],publics[166],publics[167]], [transcriptHash_publics_12[0],transcriptHash_publics_12[1],transcriptHash_publics_12[2],transcriptHash_publics_12[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_13[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_14[16] <== Poseidon(16)([publics[168],publics[169],publics[170],publics[171],publics[172],publics[173],publics[174],publics[175],publics[176],publics[177],publics[178],publics[179]], [transcriptHash_publics_13[0],transcriptHash_publics_13[1],transcriptHash_publics_13[2],transcriptHash_publics_13[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_14[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_15[16] <== Poseidon(16)([publics[180],publics[181],publics[182],publics[183],publics[184],publics[185],publics[186],publics[187],publics[188],publics[189],publics[190],publics[191]], [transcriptHash_publics_14[0],transcriptHash_publics_14[1],transcriptHash_publics_14[2],transcriptHash_publics_14[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_15[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_16[16] <== Poseidon(16)([publics[192],publics[193],publics[194],publics[195],publics[196],publics[197],publics[198],publics[199],publics[200],publics[201],publics[202],publics[203]], [transcriptHash_publics_15[0],transcriptHash_publics_15[1],transcriptHash_publics_15[2],transcriptHash_publics_15[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_16[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_17[16] <== Poseidon(16)([publics[204],publics[205],publics[206],publics[207],publics[208],publics[209],publics[210],publics[211],publics[212],publics[213],publics[214],publics[215]], [transcriptHash_publics_16[0],transcriptHash_publics_16[1],transcriptHash_publics_16[2],transcriptHash_publics_16[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_17[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_18[16] <== Poseidon(16)([publics[216],publics[217],publics[218],publics[219],publics[220],publics[221],publics[222],publics[223],publics[224],publics[225],publics[226],publics[227]], [transcriptHash_publics_17[0],transcriptHash_publics_17[1],transcriptHash_publics_17[2],transcriptHash_publics_17[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_18[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_19[16] <== Poseidon(16)([publics[228],publics[229],publics[230],publics[231],publics[232],publics[233],publics[234],publics[235],publics[236],publics[237],publics[238],publics[239]], [transcriptHash_publics_18[0],transcriptHash_publics_18[1],transcriptHash_publics_18[2],transcriptHash_publics_18[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_19[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_20[16] <== Poseidon(16)([publics[240],publics[241],publics[242],publics[243],publics[244],publics[245],publics[246],publics[247],publics[248],publics[249],publics[250],publics[251]], [transcriptHash_publics_19[0],transcriptHash_publics_19[1],transcriptHash_publics_19[2],transcriptHash_publics_19[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_20[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_21[16] <== Poseidon(16)([publics[252],publics[253],publics[254],publics[255],publics[256],publics[257],publics[258],publics[259],publics[260],publics[261],publics[262],publics[263]], [transcriptHash_publics_20[0],transcriptHash_publics_20[1],transcriptHash_publics_20[2],transcriptHash_publics_20[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_21[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_22[16] <== Poseidon(16)([publics[264],publics[265],publics[266],publics[267],publics[268],publics[269],publics[270],publics[271],publics[272],publics[273],publics[274],publics[275]], [transcriptHash_publics_21[0],transcriptHash_publics_21[1],transcriptHash_publics_21[2],transcriptHash_publics_21[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_22[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_23[16] <== Poseidon(16)([publics[276],publics[277],publics[278],publics[279],publics[280],publics[281],publics[282],publics[283],publics[284],publics[285],publics[286],publics[287]], [transcriptHash_publics_22[0],transcriptHash_publics_22[1],transcriptHash_publics_22[2],transcriptHash_publics_22[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_23[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_24[16] <== Poseidon(16)([publics[288],publics[289],publics[290],publics[291],publics[292],publics[293],publics[294],publics[295],publics[296],publics[297],publics[298],publics[299]], [transcriptHash_publics_23[0],transcriptHash_publics_23[1],transcriptHash_publics_23[2],transcriptHash_publics_23[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_24[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_25[16] <== Poseidon(16)([publics[300],publics[301],publics[302],publics[303],publics[304],publics[305],publics[306],publics[307],publics[308],publics[309],publics[310],publics[311]], [transcriptHash_publics_24[0],transcriptHash_publics_24[1],transcriptHash_publics_24[2],transcriptHash_publics_24[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_25[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_26[16] <== Poseidon(16)([publics[312],publics[313],publics[314],publics[315],publics[316],publics[317],publics[318],publics[319],publics[320],publics[321],publics[322],publics[323]], [transcriptHash_publics_25[0],transcriptHash_publics_25[1],transcriptHash_publics_25[2],transcriptHash_publics_25[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_26[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_27[16] <== Poseidon(16)([publics[324],publics[325],publics[326],publics[327],publics[328],publics[329],publics[330],publics[331],publics[332],publics[333],publics[334],publics[335]], [transcriptHash_publics_26[0],transcriptHash_publics_26[1],transcriptHash_publics_26[2],transcriptHash_publics_26[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_27[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_28[16] <== Poseidon(16)([publics[336],publics[337],publics[338],publics[339],publics[340],publics[341],publics[342],publics[343],publics[344],publics[345],publics[346],publics[347]], [transcriptHash_publics_27[0],transcriptHash_publics_27[1],transcriptHash_publics_27[2],transcriptHash_publics_27[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_28[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_29[16] <== Poseidon(16)([publics[348],publics[349],publics[350],publics[351],publics[352],publics[353],publics[354],publics[355],publics[356],publics[357],publics[358],publics[359]], [transcriptHash_publics_28[0],transcriptHash_publics_28[1],transcriptHash_publics_28[2],transcriptHash_publics_28[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_29[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_30[16] <== Poseidon(16)([publics[360],publics[361],publics[362],publics[363],publics[364],publics[365],publics[366],publics[367],publics[368],publics[369],publics[370],publics[371]], [transcriptHash_publics_29[0],transcriptHash_publics_29[1],transcriptHash_publics_29[2],transcriptHash_publics_29[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_30[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_31[16] <== Poseidon(16)([publics[372],publics[373],publics[374],publics[375],publics[376],publics[377],publics[378],publics[379],publics[380],publics[381],publics[382],publics[383]], [transcriptHash_publics_30[0],transcriptHash_publics_30[1],transcriptHash_publics_30[2],transcriptHash_publics_30[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_publics_31[i]; // Unused transcript values 
    }

    signal transcriptHash_publics_32[16] <== Poseidon(16)([publics[384],publics[385],publics[386],publics[387],publics[388],publics[389],publics[390],publics[391],publics[392],publics[393],publics[394],0], [transcriptHash_publics_31[0],transcriptHash_publics_31[1],transcriptHash_publics_31[2],transcriptHash_publics_31[3]]);
    publicsHash <== [transcriptHash_publics_32[0], transcriptHash_publics_32[1], transcriptHash_publics_32[2], transcriptHash_publics_32[3]];

    signal transcriptHash_0[16] <== Poseidon(16)([rootC[0],rootC[1],rootC[2],rootC[3],publicsHash[0],publicsHash[1],publicsHash[2],publicsHash[3],root1[0],root1[1],root1[2],root1[3]], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for(var i = 6; i < 16; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }

    signal transcriptHash_1[16] <== Poseidon(16)([root2[0],root2[1],root2[2],root2[3],0,0,0,0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }

    signal transcriptHash_2[16] <== Poseidon(16)([root3[0],root3[1],root3[2],root3[3],0,0,0,0,0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeXi <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];

    signal transcriptHash_evals_0[16] <== Poseidon(16)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1],evals[2][2],evals[3][0],evals[3][1],evals[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_1[16] <== Poseidon(16)([evals[4][0],evals[4][1],evals[4][2],evals[5][0],evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_2[16] <== Poseidon(16)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1],evals[10][2],evals[11][0],evals[11][1],evals[11][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_3[16] <== Poseidon(16)([evals[12][0],evals[12][1],evals[12][2],evals[13][0],evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_4[16] <== Poseidon(16)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1],evals[18][2],evals[19][0],evals[19][1],evals[19][2]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_4[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_5[16] <== Poseidon(16)([evals[20][0],evals[20][1],evals[20][2],evals[21][0],evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],evals[23][0],evals[23][1],evals[23][2]], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_5[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_6[16] <== Poseidon(16)([evals[24][0],evals[24][1],evals[24][2],evals[25][0],evals[25][1],evals[25][2],evals[26][0],evals[26][1],evals[26][2],evals[27][0],evals[27][1],evals[27][2]], [transcriptHash_evals_5[0],transcriptHash_evals_5[1],transcriptHash_evals_5[2],transcriptHash_evals_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_6[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_7[16] <== Poseidon(16)([evals[28][0],evals[28][1],evals[28][2],evals[29][0],evals[29][1],evals[29][2],evals[30][0],evals[30][1],evals[30][2],evals[31][0],evals[31][1],evals[31][2]], [transcriptHash_evals_6[0],transcriptHash_evals_6[1],transcriptHash_evals_6[2],transcriptHash_evals_6[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_7[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_8[16] <== Poseidon(16)([evals[32][0],evals[32][1],evals[32][2],evals[33][0],evals[33][1],evals[33][2],evals[34][0],evals[34][1],evals[34][2],evals[35][0],evals[35][1],evals[35][2]], [transcriptHash_evals_7[0],transcriptHash_evals_7[1],transcriptHash_evals_7[2],transcriptHash_evals_7[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_8[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_9[16] <== Poseidon(16)([evals[36][0],evals[36][1],evals[36][2],evals[37][0],evals[37][1],evals[37][2],evals[38][0],evals[38][1],evals[38][2],evals[39][0],evals[39][1],evals[39][2]], [transcriptHash_evals_8[0],transcriptHash_evals_8[1],transcriptHash_evals_8[2],transcriptHash_evals_8[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_9[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_10[16] <== Poseidon(16)([evals[40][0],evals[40][1],evals[40][2],evals[41][0],evals[41][1],evals[41][2],evals[42][0],evals[42][1],evals[42][2],evals[43][0],evals[43][1],evals[43][2]], [transcriptHash_evals_9[0],transcriptHash_evals_9[1],transcriptHash_evals_9[2],transcriptHash_evals_9[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_10[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_11[16] <== Poseidon(16)([evals[44][0],evals[44][1],evals[44][2],evals[45][0],evals[45][1],evals[45][2],evals[46][0],evals[46][1],evals[46][2],evals[47][0],evals[47][1],evals[47][2]], [transcriptHash_evals_10[0],transcriptHash_evals_10[1],transcriptHash_evals_10[2],transcriptHash_evals_10[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_11[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_12[16] <== Poseidon(16)([evals[48][0],evals[48][1],evals[48][2],evals[49][0],evals[49][1],evals[49][2],evals[50][0],evals[50][1],evals[50][2],evals[51][0],evals[51][1],evals[51][2]], [transcriptHash_evals_11[0],transcriptHash_evals_11[1],transcriptHash_evals_11[2],transcriptHash_evals_11[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_12[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_13[16] <== Poseidon(16)([evals[52][0],evals[52][1],evals[52][2],evals[53][0],evals[53][1],evals[53][2],evals[54][0],evals[54][1],evals[54][2],evals[55][0],evals[55][1],evals[55][2]], [transcriptHash_evals_12[0],transcriptHash_evals_12[1],transcriptHash_evals_12[2],transcriptHash_evals_12[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_13[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_14[16] <== Poseidon(16)([evals[56][0],evals[56][1],evals[56][2],evals[57][0],evals[57][1],evals[57][2],evals[58][0],evals[58][1],evals[58][2],evals[59][0],evals[59][1],evals[59][2]], [transcriptHash_evals_13[0],transcriptHash_evals_13[1],transcriptHash_evals_13[2],transcriptHash_evals_13[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_14[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_15[16] <== Poseidon(16)([evals[60][0],evals[60][1],evals[60][2],evals[61][0],evals[61][1],evals[61][2],evals[62][0],evals[62][1],evals[62][2],evals[63][0],evals[63][1],evals[63][2]], [transcriptHash_evals_14[0],transcriptHash_evals_14[1],transcriptHash_evals_14[2],transcriptHash_evals_14[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_15[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_16[16] <== Poseidon(16)([evals[64][0],evals[64][1],evals[64][2],evals[65][0],evals[65][1],evals[65][2],evals[66][0],evals[66][1],evals[66][2],evals[67][0],evals[67][1],evals[67][2]], [transcriptHash_evals_15[0],transcriptHash_evals_15[1],transcriptHash_evals_15[2],transcriptHash_evals_15[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_16[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_17[16] <== Poseidon(16)([evals[68][0],evals[68][1],evals[68][2],evals[69][0],evals[69][1],evals[69][2],evals[70][0],evals[70][1],evals[70][2],evals[71][0],evals[71][1],evals[71][2]], [transcriptHash_evals_16[0],transcriptHash_evals_16[1],transcriptHash_evals_16[2],transcriptHash_evals_16[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_17[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_18[16] <== Poseidon(16)([evals[72][0],evals[72][1],evals[72][2],evals[73][0],evals[73][1],evals[73][2],evals[74][0],evals[74][1],evals[74][2],evals[75][0],evals[75][1],evals[75][2]], [transcriptHash_evals_17[0],transcriptHash_evals_17[1],transcriptHash_evals_17[2],transcriptHash_evals_17[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_18[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_19[16] <== Poseidon(16)([evals[76][0],evals[76][1],evals[76][2],evals[77][0],evals[77][1],evals[77][2],evals[78][0],evals[78][1],evals[78][2],evals[79][0],evals[79][1],evals[79][2]], [transcriptHash_evals_18[0],transcriptHash_evals_18[1],transcriptHash_evals_18[2],transcriptHash_evals_18[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_19[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_20[16] <== Poseidon(16)([evals[80][0],evals[80][1],evals[80][2],evals[81][0],evals[81][1],evals[81][2],evals[82][0],evals[82][1],evals[82][2],evals[83][0],evals[83][1],evals[83][2]], [transcriptHash_evals_19[0],transcriptHash_evals_19[1],transcriptHash_evals_19[2],transcriptHash_evals_19[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_20[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_21[16] <== Poseidon(16)([evals[84][0],evals[84][1],evals[84][2],evals[85][0],evals[85][1],evals[85][2],evals[86][0],evals[86][1],evals[86][2],evals[87][0],evals[87][1],evals[87][2]], [transcriptHash_evals_20[0],transcriptHash_evals_20[1],transcriptHash_evals_20[2],transcriptHash_evals_20[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_21[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_22[16] <== Poseidon(16)([evals[88][0],evals[88][1],evals[88][2],evals[89][0],evals[89][1],evals[89][2],evals[90][0],evals[90][1],evals[90][2],evals[91][0],evals[91][1],evals[91][2]], [transcriptHash_evals_21[0],transcriptHash_evals_21[1],transcriptHash_evals_21[2],transcriptHash_evals_21[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_22[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_23[16] <== Poseidon(16)([evals[92][0],evals[92][1],evals[92][2],evals[93][0],evals[93][1],evals[93][2],evals[94][0],evals[94][1],evals[94][2],evals[95][0],evals[95][1],evals[95][2]], [transcriptHash_evals_22[0],transcriptHash_evals_22[1],transcriptHash_evals_22[2],transcriptHash_evals_22[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_23[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_24[16] <== Poseidon(16)([evals[96][0],evals[96][1],evals[96][2],evals[97][0],evals[97][1],evals[97][2],evals[98][0],evals[98][1],evals[98][2],evals[99][0],evals[99][1],evals[99][2]], [transcriptHash_evals_23[0],transcriptHash_evals_23[1],transcriptHash_evals_23[2],transcriptHash_evals_23[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_24[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_25[16] <== Poseidon(16)([evals[100][0],evals[100][1],evals[100][2],evals[101][0],evals[101][1],evals[101][2],evals[102][0],evals[102][1],evals[102][2],evals[103][0],evals[103][1],evals[103][2]], [transcriptHash_evals_24[0],transcriptHash_evals_24[1],transcriptHash_evals_24[2],transcriptHash_evals_24[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_25[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_26[16] <== Poseidon(16)([evals[104][0],evals[104][1],evals[104][2],evals[105][0],evals[105][1],evals[105][2],evals[106][0],evals[106][1],evals[106][2],evals[107][0],evals[107][1],evals[107][2]], [transcriptHash_evals_25[0],transcriptHash_evals_25[1],transcriptHash_evals_25[2],transcriptHash_evals_25[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_26[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_27[16] <== Poseidon(16)([evals[108][0],evals[108][1],evals[108][2],evals[109][0],evals[109][1],evals[109][2],evals[110][0],evals[110][1],evals[110][2],evals[111][0],evals[111][1],evals[111][2]], [transcriptHash_evals_26[0],transcriptHash_evals_26[1],transcriptHash_evals_26[2],transcriptHash_evals_26[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_27[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_28[16] <== Poseidon(16)([evals[112][0],evals[112][1],evals[112][2],evals[113][0],evals[113][1],evals[113][2],evals[114][0],evals[114][1],evals[114][2],evals[115][0],evals[115][1],evals[115][2]], [transcriptHash_evals_27[0],transcriptHash_evals_27[1],transcriptHash_evals_27[2],transcriptHash_evals_27[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_28[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_29[16] <== Poseidon(16)([evals[116][0],evals[116][1],evals[116][2],evals[117][0],evals[117][1],evals[117][2],evals[118][0],evals[118][1],evals[118][2],evals[119][0],evals[119][1],evals[119][2]], [transcriptHash_evals_28[0],transcriptHash_evals_28[1],transcriptHash_evals_28[2],transcriptHash_evals_28[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_29[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_30[16] <== Poseidon(16)([evals[120][0],evals[120][1],evals[120][2],evals[121][0],evals[121][1],evals[121][2],evals[122][0],evals[122][1],evals[122][2],evals[123][0],evals[123][1],evals[123][2]], [transcriptHash_evals_29[0],transcriptHash_evals_29[1],transcriptHash_evals_29[2],transcriptHash_evals_29[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_30[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_31[16] <== Poseidon(16)([evals[124][0],evals[124][1],evals[124][2],evals[125][0],evals[125][1],evals[125][2],evals[126][0],evals[126][1],evals[126][2],evals[127][0],evals[127][1],evals[127][2]], [transcriptHash_evals_30[0],transcriptHash_evals_30[1],transcriptHash_evals_30[2],transcriptHash_evals_30[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_31[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_32[16] <== Poseidon(16)([evals[128][0],evals[128][1],evals[128][2],evals[129][0],evals[129][1],evals[129][2],evals[130][0],evals[130][1],evals[130][2],evals[131][0],evals[131][1],evals[131][2]], [transcriptHash_evals_31[0],transcriptHash_evals_31[1],transcriptHash_evals_31[2],transcriptHash_evals_31[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_32[i]; // Unused transcript values 
    }

    signal transcriptHash_evals_33[16] <== Poseidon(16)([evals[132][0],evals[132][1],evals[132][2],evals[133][0],evals[133][1],evals[133][2],evals[134][0],evals[134][1],evals[134][2],0,0,0], [transcriptHash_evals_32[0],transcriptHash_evals_32[1],transcriptHash_evals_32[2],transcriptHash_evals_32[3]]);
    evalsHash <== [transcriptHash_evals_33[0], transcriptHash_evals_33[1], transcriptHash_evals_33[2], transcriptHash_evals_33[3]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }

    signal transcriptHash_3[16] <== Poseidon(16)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0,0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesFRI[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesFRI[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    challengesFRISteps[0] <== [transcriptHash_3[6], transcriptHash_3[7], transcriptHash_3[8]];
    for(var i = 9; i < 16; i++){
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }

    signal transcriptHash_4[16] <== Poseidon(16)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0,0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    challengesFRISteps[1] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }

    signal transcriptHash_5[16] <== Poseidon(16)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0,0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    challengesFRISteps[2] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }

    signal transcriptHash_6[16] <== Poseidon(16)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0,0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    challengesFRISteps[3] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }

    signal transcriptHash_7[16] <== Poseidon(16)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0,0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    challengesFRISteps[4] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }

    signal transcriptHash_8[16] <== Poseidon(16)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0,0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    challengesFRISteps[5] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];

    signal transcriptHash_lastPolFRI_0[16] <== Poseidon(16)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1],finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_1[16] <== Poseidon(16)([finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0],finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_2[16] <== Poseidon(16)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1],finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_3[16] <== Poseidon(16)([finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0],finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_4[16] <== Poseidon(16)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1],finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_5[16] <== Poseidon(16)([finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0],finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_6[16] <== Poseidon(16)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1],finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }

    signal transcriptHash_lastPolFRI_7[16] <== Poseidon(16)([finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0],finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_7[0], transcriptHash_lastPolFRI_7[1], transcriptHash_lastPolFRI_7[2], transcriptHash_lastPolFRI_7[3]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }

    signal transcriptHash_9[16] <== Poseidon(16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
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
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];


    signal output tmp_6066[3];
    signal output tmp_6068[3];
    signal output tmp_6405[3];
    signal output tmp_6414[3];
    signal output tmp_6421[3];
    signal output tmp_6428[3];
    signal output tmp_6435[3];
    signal output tmp_6443[3];
    signal output tmp_6450[3];
    signal output tmp_6459[3];
    signal output tmp_6466[3];
    signal output tmp_6474[3];
    signal output tmp_6481[3];
    signal output tmp_6490[3];
    signal output tmp_6497[3];
    signal output tmp_6506[3];
    signal output tmp_6513[3];
    signal output tmp_6522[3];
    signal output tmp_6529[3];
    signal output tmp_6537[3];
    signal output tmp_6544[3];
    signal output tmp_6553[3];
    signal output tmp_6560[3];
    signal output tmp_6569[3];
    signal output tmp_6576[3];
    signal output tmp_6585[3];
    signal output tmp_6592[3];
    signal output tmp_6601[3];
    signal output tmp_6608[3];
    signal output tmp_6617[3];
    signal output tmp_6624[3];
    signal output tmp_6633[3];
    signal output tmp_6640[3];
    signal output tmp_6649[3];
    signal output tmp_6656[3];
    signal output tmp_7063[3];
    signal output tmp_7064[3];
    tmp_6066 <== [evals[2][0] + evals[3][0], evals[2][1] + evals[3][1], evals[2][2] + evals[3][2]];
    signal tmp_6067[3] <== [evals[47][0] + tmp_6066[0], evals[47][1] + tmp_6066[1], evals[47][2] + tmp_6066[2]];
    tmp_6068 <== [evals[0][0] + evals[1][0], evals[0][1] + evals[1][1], evals[0][2] + evals[1][2]];
    signal tmp_6069[3] <== [tmp_6067[0] + tmp_6068[0], tmp_6067[1] + tmp_6068[1], tmp_6067[2] + tmp_6068[2]];
    signal tmp_6070[3] <== [tmp_6069[0] + evals[109][0], tmp_6069[1] + evals[109][1], tmp_6069[2] + evals[109][2]];
    signal tmp_6071[3] <== CMul()(evals[50], evals[51]);
    signal tmp_6072[3] <== CMul()(evals[29], tmp_6071);
    signal tmp_6073[3] <== CMul()(evals[30], evals[50]);
    signal tmp_6074[3] <== [tmp_6072[0] + tmp_6073[0], tmp_6072[1] + tmp_6073[1], tmp_6072[2] + tmp_6073[2]];
    signal tmp_6075[3] <== CMul()(evals[31], evals[51]);
    signal tmp_6076[3] <== [tmp_6074[0] + tmp_6075[0], tmp_6074[1] + tmp_6075[1], tmp_6074[2] + tmp_6075[2]];
    signal tmp_6077[3] <== CMul()(evals[32], evals[52]);
    signal tmp_6078[3] <== [tmp_6076[0] + tmp_6077[0], tmp_6076[1] + tmp_6077[1], tmp_6076[2] + tmp_6077[2]];
    signal tmp_6079[3] <== [tmp_6078[0] + evals[33][0], tmp_6078[1] + evals[33][1], tmp_6078[2] + evals[33][2]];
    signal tmp_6080[3] <== CMul()(tmp_6070, tmp_6079);
    signal tmp_6081[3] <== CMul()(challengeQ, tmp_6080);
    signal tmp_6082[3] <== CMul()(evals[53], evals[54]);
    signal tmp_6083[3] <== CMul()(evals[29], tmp_6082);
    signal tmp_6084[3] <== CMul()(evals[30], evals[53]);
    signal tmp_6085[3] <== [tmp_6083[0] + tmp_6084[0], tmp_6083[1] + tmp_6084[1], tmp_6083[2] + tmp_6084[2]];
    signal tmp_6086[3] <== CMul()(evals[31], evals[54]);
    signal tmp_6087[3] <== [tmp_6085[0] + tmp_6086[0], tmp_6085[1] + tmp_6086[1], tmp_6085[2] + tmp_6086[2]];
    signal tmp_6088[3] <== CMul()(evals[32], evals[55]);
    signal tmp_6089[3] <== [tmp_6087[0] + tmp_6088[0], tmp_6087[1] + tmp_6088[1], tmp_6087[2] + tmp_6088[2]];
    signal tmp_6090[3] <== [tmp_6089[0] + evals[33][0], tmp_6089[1] + evals[33][1], tmp_6089[2] + evals[33][2]];
    signal tmp_6091[3] <== CMul()(tmp_6070, tmp_6090);
    signal tmp_6092[3] <== [tmp_6081[0] + tmp_6091[0], tmp_6081[1] + tmp_6091[1], tmp_6081[2] + tmp_6091[2]];
    signal tmp_6093[3] <== CMul()(challengeQ, tmp_6092);
    signal tmp_6094[3] <== CMul()(evals[56], evals[57]);
    signal tmp_6095[3] <== CMul()(evals[34], tmp_6094);
    signal tmp_6096[3] <== CMul()(evals[35], evals[56]);
    signal tmp_6097[3] <== [tmp_6095[0] + tmp_6096[0], tmp_6095[1] + tmp_6096[1], tmp_6095[2] + tmp_6096[2]];
    signal tmp_6098[3] <== CMul()(evals[36], evals[57]);
    signal tmp_6099[3] <== [tmp_6097[0] + tmp_6098[0], tmp_6097[1] + tmp_6098[1], tmp_6097[2] + tmp_6098[2]];
    signal tmp_6100[3] <== CMul()(evals[37], evals[58]);
    signal tmp_6101[3] <== [tmp_6099[0] + tmp_6100[0], tmp_6099[1] + tmp_6100[1], tmp_6099[2] + tmp_6100[2]];
    signal tmp_6102[3] <== [tmp_6101[0] + evals[38][0], tmp_6101[1] + evals[38][1], tmp_6101[2] + evals[38][2]];
    signal tmp_6103[3] <== CMul()(tmp_6070, tmp_6102);
    signal tmp_6104[3] <== [tmp_6093[0] + tmp_6103[0], tmp_6093[1] + tmp_6103[1], tmp_6093[2] + tmp_6103[2]];
    signal tmp_6105[3] <== CMul()(challengeQ, tmp_6104);
    signal tmp_6106[3] <== [evals[47][0] + tmp_6066[0], evals[47][1] + tmp_6066[1], evals[47][2] + tmp_6066[2]];
    signal tmp_6107[3] <== [tmp_6106[0] + evals[109][0], tmp_6106[1] + evals[109][1], tmp_6106[2] + evals[109][2]];
    signal tmp_6108[3] <== CMul()(evals[59], evals[60]);
    signal tmp_6109[3] <== CMul()(evals[34], tmp_6108);
    signal tmp_6110[3] <== CMul()(evals[35], evals[59]);
    signal tmp_6111[3] <== [tmp_6109[0] + tmp_6110[0], tmp_6109[1] + tmp_6110[1], tmp_6109[2] + tmp_6110[2]];
    signal tmp_6112[3] <== CMul()(evals[36], evals[60]);
    signal tmp_6113[3] <== [tmp_6111[0] + tmp_6112[0], tmp_6111[1] + tmp_6112[1], tmp_6111[2] + tmp_6112[2]];
    signal tmp_6114[3] <== CMul()(evals[37], evals[61]);
    signal tmp_6115[3] <== [tmp_6113[0] + tmp_6114[0], tmp_6113[1] + tmp_6114[1], tmp_6113[2] + tmp_6114[2]];
    signal tmp_6116[3] <== [tmp_6115[0] + evals[38][0], tmp_6115[1] + evals[38][1], tmp_6115[2] + evals[38][2]];
    signal tmp_6117[3] <== CMul()(tmp_6107, tmp_6116);
    signal tmp_6118[3] <== [tmp_6105[0] + tmp_6117[0], tmp_6105[1] + tmp_6117[1], tmp_6105[2] + tmp_6117[2]];
    signal tmp_6119[3] <== CMul()(challengeQ, tmp_6118);
    signal tmp_6120[3] <== CMul()(evals[62], evals[63]);
    signal tmp_6121[3] <== CMul()(evals[34], tmp_6120);
    signal tmp_6122[3] <== CMul()(evals[35], evals[62]);
    signal tmp_6123[3] <== [tmp_6121[0] + tmp_6122[0], tmp_6121[1] + tmp_6122[1], tmp_6121[2] + tmp_6122[2]];
    signal tmp_6124[3] <== CMul()(evals[36], evals[63]);
    signal tmp_6125[3] <== [tmp_6123[0] + tmp_6124[0], tmp_6123[1] + tmp_6124[1], tmp_6123[2] + tmp_6124[2]];
    signal tmp_6126[3] <== CMul()(evals[37], evals[64]);
    signal tmp_6127[3] <== [tmp_6125[0] + tmp_6126[0], tmp_6125[1] + tmp_6126[1], tmp_6125[2] + tmp_6126[2]];
    signal tmp_6128[3] <== [tmp_6127[0] + evals[38][0], tmp_6127[1] + evals[38][1], tmp_6127[2] + evals[38][2]];
    signal tmp_6129[3] <== CMul()(tmp_6107, tmp_6128);
    signal tmp_6130[3] <== [tmp_6119[0] + tmp_6129[0], tmp_6119[1] + tmp_6129[1], tmp_6119[2] + tmp_6129[2]];
    signal tmp_6131[3] <== CMul()(challengeQ, tmp_6130);
    signal tmp_6132[3] <== CMul()(evals[65], evals[66]);
    signal tmp_6133[3] <== CMul()(evals[34], tmp_6132);
    signal tmp_6134[3] <== CMul()(evals[35], evals[65]);
    signal tmp_6135[3] <== [tmp_6133[0] + tmp_6134[0], tmp_6133[1] + tmp_6134[1], tmp_6133[2] + tmp_6134[2]];
    signal tmp_6136[3] <== CMul()(evals[36], evals[66]);
    signal tmp_6137[3] <== [tmp_6135[0] + tmp_6136[0], tmp_6135[1] + tmp_6136[1], tmp_6135[2] + tmp_6136[2]];
    signal tmp_6138[3] <== CMul()(evals[37], evals[67]);
    signal tmp_6139[3] <== [tmp_6137[0] + tmp_6138[0], tmp_6137[1] + tmp_6138[1], tmp_6137[2] + tmp_6138[2]];
    signal tmp_6140[3] <== [tmp_6139[0] + evals[38][0], tmp_6139[1] + evals[38][1], tmp_6139[2] + evals[38][2]];
    signal tmp_6141[3] <== CMul()(evals[47], tmp_6140);
    signal tmp_6142[3] <== [tmp_6131[0] + tmp_6141[0], tmp_6131[1] + tmp_6141[1], tmp_6131[2] + tmp_6141[2]];
    signal tmp_6143[3] <== CMul()(challengeQ, tmp_6142);
    signal tmp_6144[3] <== [evals[47][0] + evals[42][0], evals[47][1] + evals[42][1], evals[47][2] + evals[42][2]];
    signal tmp_6145[3] <== CMul()(evals[68], evals[69]);
    signal tmp_6146[3] <== CMul()(evals[34], tmp_6145);
    signal tmp_6147[3] <== CMul()(evals[35], evals[68]);
    signal tmp_6148[3] <== [tmp_6146[0] + tmp_6147[0], tmp_6146[1] + tmp_6147[1], tmp_6146[2] + tmp_6147[2]];
    signal tmp_6149[3] <== CMul()(evals[36], evals[69]);
    signal tmp_6150[3] <== [tmp_6148[0] + tmp_6149[0], tmp_6148[1] + tmp_6149[1], tmp_6148[2] + tmp_6149[2]];
    signal tmp_6151[3] <== CMul()(evals[37], evals[70]);
    signal tmp_6152[3] <== [tmp_6150[0] + tmp_6151[0], tmp_6150[1] + tmp_6151[1], tmp_6150[2] + tmp_6151[2]];
    signal tmp_6153[3] <== [tmp_6152[0] + evals[38][0], tmp_6152[1] + evals[38][1], tmp_6152[2] + evals[38][2]];
    signal tmp_6154[3] <== CMul()(tmp_6144, tmp_6153);
    signal tmp_6155[3] <== [tmp_6143[0] + tmp_6154[0], tmp_6143[1] + tmp_6154[1], tmp_6143[2] + tmp_6154[2]];
    signal tmp_6156[3] <== CMul()(challengeQ, tmp_6155);
    signal tmp_6157[3] <== [evals[47][0] + evals[42][0], evals[47][1] + evals[42][1], evals[47][2] + evals[42][2]];
    signal tmp_6158[3] <== [tmp_6157[0] + evals[43][0], tmp_6157[1] + evals[43][1], tmp_6157[2] + evals[43][2]];
    signal tmp_6159[3] <== CMul()(evals[71], evals[72]);
    signal tmp_6160[3] <== CMul()(evals[34], tmp_6159);
    signal tmp_6161[3] <== CMul()(evals[35], evals[71]);
    signal tmp_6162[3] <== [tmp_6160[0] + tmp_6161[0], tmp_6160[1] + tmp_6161[1], tmp_6160[2] + tmp_6161[2]];
    signal tmp_6163[3] <== CMul()(evals[36], evals[72]);
    signal tmp_6164[3] <== [tmp_6162[0] + tmp_6163[0], tmp_6162[1] + tmp_6163[1], tmp_6162[2] + tmp_6163[2]];
    signal tmp_6165[3] <== CMul()(evals[37], evals[73]);
    signal tmp_6166[3] <== [tmp_6164[0] + tmp_6165[0], tmp_6164[1] + tmp_6165[1], tmp_6164[2] + tmp_6165[2]];
    signal tmp_6167[3] <== [tmp_6166[0] + evals[38][0], tmp_6166[1] + evals[38][1], tmp_6166[2] + evals[38][2]];
    signal tmp_6168[3] <== CMul()(tmp_6158, tmp_6167);
    signal tmp_6169[3] <== [tmp_6156[0] + tmp_6168[0], tmp_6156[1] + tmp_6168[1], tmp_6156[2] + tmp_6168[2]];
    signal tmp_6170[3] <== CMul()(challengeQ, tmp_6169);
    signal tmp_6171[3] <== CMul()(evals[39], evals[117]);
    signal tmp_6172[3] <== [evals[117][0] - 1, evals[117][1], evals[117][2]];
    signal tmp_6173[3] <== CMul()(tmp_6171, tmp_6172);
    signal tmp_6174[3] <== [tmp_6170[0] + tmp_6173[0], tmp_6170[1] + tmp_6173[1], tmp_6170[2] + tmp_6173[2]];
    signal tmp_6175[3] <== CMul()(challengeQ, tmp_6174);
    signal tmp_6176[3] <== CMul()(evals[39], evals[134]);
    signal tmp_6177[3] <== [evals[134][0] - 1, evals[134][1], evals[134][2]];
    signal tmp_6178[3] <== CMul()(tmp_6176, tmp_6177);
    signal tmp_6179[3] <== [tmp_6175[0] + tmp_6178[0], tmp_6175[1] + tmp_6178[1], tmp_6175[2] + tmp_6178[2]];
    signal tmp_6180[3] <== CMul()(challengeQ, tmp_6179);
    signal tmp_6181[3] <== [1 - evals[117][0], -evals[117][1], -evals[117][2]];
    signal tmp_6182[3] <== [1 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_6183[3] <== CMul()(tmp_6181, tmp_6182);
    signal tmp_6184[3] <== CMul()(tmp_6183, evals[50]);
    signal tmp_6185[3] <== [1 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_6186[3] <== CMul()(evals[117], tmp_6185);
    signal tmp_6187[3] <== [1 - evals[117][0], -evals[117][1], -evals[117][2]];
    signal tmp_6188[3] <== CMul()(tmp_6187, evals[134]);
    signal tmp_6189[3] <== [tmp_6186[0] + tmp_6188[0], tmp_6186[1] + tmp_6188[1], tmp_6186[2] + tmp_6188[2]];
    signal tmp_6190[3] <== CMul()(evals[117], evals[134]);
    signal tmp_6191[3] <== [tmp_6189[0] + tmp_6190[0], tmp_6189[1] + tmp_6190[1], tmp_6189[2] + tmp_6190[2]];
    signal tmp_6192[3] <== CMul()(tmp_6191, evals[54]);
    signal tmp_6193[3] <== [tmp_6184[0] + tmp_6192[0], tmp_6184[1] + tmp_6192[1], tmp_6184[2] + tmp_6192[2]];
    signal tmp_6194[3] <== [evals[66][0] - tmp_6193[0], evals[66][1] - tmp_6193[1], evals[66][2] - tmp_6193[2]];
    signal tmp_6195[3] <== CMul()(evals[39], tmp_6194);
    signal tmp_6196[3] <== [tmp_6180[0] + tmp_6195[0], tmp_6180[1] + tmp_6195[1], tmp_6180[2] + tmp_6195[2]];
    signal tmp_6197[3] <== CMul()(challengeQ, tmp_6196);
    signal tmp_6198[3] <== [evals[66][0] - evals[50][0], evals[66][1] - evals[50][1], evals[66][2] - evals[50][2]];
    signal tmp_6199[3] <== CMul()(evals[40], tmp_6198);
    signal tmp_6200[3] <== [tmp_6197[0] + tmp_6199[0], tmp_6197[1] + tmp_6199[1], tmp_6197[2] + tmp_6199[2]];
    signal tmp_6201[3] <== CMul()(challengeQ, tmp_6200);
    signal tmp_6202[3] <== CMul()(tmp_6183, evals[51]);
    signal tmp_6203[3] <== [tmp_6186[0] + tmp_6188[0], tmp_6186[1] + tmp_6188[1], tmp_6186[2] + tmp_6188[2]];
    signal tmp_6204[3] <== [tmp_6203[0] + tmp_6190[0], tmp_6203[1] + tmp_6190[1], tmp_6203[2] + tmp_6190[2]];
    signal tmp_6205[3] <== CMul()(tmp_6204, evals[55]);
    signal tmp_6206[3] <== [tmp_6202[0] + tmp_6205[0], tmp_6202[1] + tmp_6205[1], tmp_6202[2] + tmp_6205[2]];
    signal tmp_6207[3] <== [evals[67][0] - tmp_6206[0], evals[67][1] - tmp_6206[1], evals[67][2] - tmp_6206[2]];
    signal tmp_6208[3] <== CMul()(evals[39], tmp_6207);
    signal tmp_6209[3] <== [tmp_6201[0] + tmp_6208[0], tmp_6201[1] + tmp_6208[1], tmp_6201[2] + tmp_6208[2]];
    signal tmp_6210[3] <== CMul()(challengeQ, tmp_6209);
    signal tmp_6211[3] <== [evals[67][0] - evals[51][0], evals[67][1] - evals[51][1], evals[67][2] - evals[51][2]];
    signal tmp_6212[3] <== CMul()(evals[40], tmp_6211);
    signal tmp_6213[3] <== [tmp_6210[0] + tmp_6212[0], tmp_6210[1] + tmp_6212[1], tmp_6210[2] + tmp_6212[2]];
    signal tmp_6214[3] <== CMul()(challengeQ, tmp_6213);
    signal tmp_6215[3] <== CMul()(tmp_6183, evals[52]);
    signal tmp_6216[3] <== [tmp_6186[0] + tmp_6188[0], tmp_6186[1] + tmp_6188[1], tmp_6186[2] + tmp_6188[2]];
    signal tmp_6217[3] <== [tmp_6216[0] + tmp_6190[0], tmp_6216[1] + tmp_6190[1], tmp_6216[2] + tmp_6190[2]];
    signal tmp_6218[3] <== CMul()(tmp_6217, evals[56]);
    signal tmp_6219[3] <== [tmp_6215[0] + tmp_6218[0], tmp_6215[1] + tmp_6218[1], tmp_6215[2] + tmp_6218[2]];
    signal tmp_6220[3] <== [evals[68][0] - tmp_6219[0], evals[68][1] - tmp_6219[1], evals[68][2] - tmp_6219[2]];
    signal tmp_6221[3] <== CMul()(evals[39], tmp_6220);
    signal tmp_6222[3] <== [tmp_6214[0] + tmp_6221[0], tmp_6214[1] + tmp_6221[1], tmp_6214[2] + tmp_6221[2]];
    signal tmp_6223[3] <== CMul()(challengeQ, tmp_6222);
    signal tmp_6224[3] <== [evals[68][0] - evals[52][0], evals[68][1] - evals[52][1], evals[68][2] - evals[52][2]];
    signal tmp_6225[3] <== CMul()(evals[40], tmp_6224);
    signal tmp_6226[3] <== [tmp_6223[0] + tmp_6225[0], tmp_6223[1] + tmp_6225[1], tmp_6223[2] + tmp_6225[2]];
    signal tmp_6227[3] <== CMul()(challengeQ, tmp_6226);
    signal tmp_6228[3] <== CMul()(tmp_6183, evals[53]);
    signal tmp_6229[3] <== [tmp_6186[0] + tmp_6188[0], tmp_6186[1] + tmp_6188[1], tmp_6186[2] + tmp_6188[2]];
    signal tmp_6230[3] <== [tmp_6229[0] + tmp_6190[0], tmp_6229[1] + tmp_6190[1], tmp_6229[2] + tmp_6190[2]];
    signal tmp_6231[3] <== CMul()(tmp_6230, evals[57]);
    signal tmp_6232[3] <== [tmp_6228[0] + tmp_6231[0], tmp_6228[1] + tmp_6231[1], tmp_6228[2] + tmp_6231[2]];
    signal tmp_6233[3] <== [evals[69][0] - tmp_6232[0], evals[69][1] - tmp_6232[1], evals[69][2] - tmp_6232[2]];
    signal tmp_6234[3] <== CMul()(evals[39], tmp_6233);
    signal tmp_6235[3] <== [tmp_6227[0] + tmp_6234[0], tmp_6227[1] + tmp_6234[1], tmp_6227[2] + tmp_6234[2]];
    signal tmp_6236[3] <== CMul()(challengeQ, tmp_6235);
    signal tmp_6237[3] <== [evals[69][0] - evals[53][0], evals[69][1] - evals[53][1], evals[69][2] - evals[53][2]];
    signal tmp_6238[3] <== CMul()(evals[40], tmp_6237);
    signal tmp_6239[3] <== [tmp_6236[0] + tmp_6238[0], tmp_6236[1] + tmp_6238[1], tmp_6236[2] + tmp_6238[2]];
    signal tmp_6240[3] <== CMul()(challengeQ, tmp_6239);
    signal tmp_6241[3] <== CMul()(tmp_6183, evals[54]);
    signal tmp_6242[3] <== CMul()(tmp_6186, evals[50]);
    signal tmp_6243[3] <== [tmp_6241[0] + tmp_6242[0], tmp_6241[1] + tmp_6242[1], tmp_6241[2] + tmp_6242[2]];
    signal tmp_6244[3] <== [tmp_6188[0] + tmp_6190[0], tmp_6188[1] + tmp_6190[1], tmp_6188[2] + tmp_6190[2]];
    signal tmp_6245[3] <== CMul()(tmp_6244, evals[58]);
    signal tmp_6246[3] <== [tmp_6243[0] + tmp_6245[0], tmp_6243[1] + tmp_6245[1], tmp_6243[2] + tmp_6245[2]];
    signal tmp_6247[3] <== [evals[70][0] - tmp_6246[0], evals[70][1] - tmp_6246[1], evals[70][2] - tmp_6246[2]];
    signal tmp_6248[3] <== CMul()(evals[39], tmp_6247);
    signal tmp_6249[3] <== [tmp_6240[0] + tmp_6248[0], tmp_6240[1] + tmp_6248[1], tmp_6240[2] + tmp_6248[2]];
    signal tmp_6250[3] <== CMul()(challengeQ, tmp_6249);
    signal tmp_6251[3] <== [evals[70][0] - evals[54][0], evals[70][1] - evals[54][1], evals[70][2] - evals[54][2]];
    signal tmp_6252[3] <== CMul()(evals[40], tmp_6251);
    signal tmp_6253[3] <== [tmp_6250[0] + tmp_6252[0], tmp_6250[1] + tmp_6252[1], tmp_6250[2] + tmp_6252[2]];
    signal tmp_6254[3] <== CMul()(challengeQ, tmp_6253);
    signal tmp_6255[3] <== CMul()(tmp_6183, evals[55]);
    signal tmp_6256[3] <== CMul()(tmp_6186, evals[51]);
    signal tmp_6257[3] <== [tmp_6255[0] + tmp_6256[0], tmp_6255[1] + tmp_6256[1], tmp_6255[2] + tmp_6256[2]];
    signal tmp_6258[3] <== [tmp_6188[0] + tmp_6190[0], tmp_6188[1] + tmp_6190[1], tmp_6188[2] + tmp_6190[2]];
    signal tmp_6259[3] <== CMul()(tmp_6258, evals[59]);
    signal tmp_6260[3] <== [tmp_6257[0] + tmp_6259[0], tmp_6257[1] + tmp_6259[1], tmp_6257[2] + tmp_6259[2]];
    signal tmp_6261[3] <== [evals[71][0] - tmp_6260[0], evals[71][1] - tmp_6260[1], evals[71][2] - tmp_6260[2]];
    signal tmp_6262[3] <== CMul()(evals[39], tmp_6261);
    signal tmp_6263[3] <== [tmp_6254[0] + tmp_6262[0], tmp_6254[1] + tmp_6262[1], tmp_6254[2] + tmp_6262[2]];
    signal tmp_6264[3] <== CMul()(challengeQ, tmp_6263);
    signal tmp_6265[3] <== [evals[71][0] - evals[55][0], evals[71][1] - evals[55][1], evals[71][2] - evals[55][2]];
    signal tmp_6266[3] <== CMul()(evals[40], tmp_6265);
    signal tmp_6267[3] <== [tmp_6264[0] + tmp_6266[0], tmp_6264[1] + tmp_6266[1], tmp_6264[2] + tmp_6266[2]];
    signal tmp_6268[3] <== CMul()(challengeQ, tmp_6267);
    signal tmp_6269[3] <== CMul()(tmp_6183, evals[56]);
    signal tmp_6270[3] <== CMul()(tmp_6186, evals[52]);
    signal tmp_6271[3] <== [tmp_6269[0] + tmp_6270[0], tmp_6269[1] + tmp_6270[1], tmp_6269[2] + tmp_6270[2]];
    signal tmp_6272[3] <== [tmp_6188[0] + tmp_6190[0], tmp_6188[1] + tmp_6190[1], tmp_6188[2] + tmp_6190[2]];
    signal tmp_6273[3] <== CMul()(tmp_6272, evals[60]);
    signal tmp_6274[3] <== [tmp_6271[0] + tmp_6273[0], tmp_6271[1] + tmp_6273[1], tmp_6271[2] + tmp_6273[2]];
    signal tmp_6275[3] <== [evals[72][0] - tmp_6274[0], evals[72][1] - tmp_6274[1], evals[72][2] - tmp_6274[2]];
    signal tmp_6276[3] <== CMul()(evals[39], tmp_6275);
    signal tmp_6277[3] <== [tmp_6268[0] + tmp_6276[0], tmp_6268[1] + tmp_6276[1], tmp_6268[2] + tmp_6276[2]];
    signal tmp_6278[3] <== CMul()(challengeQ, tmp_6277);
    signal tmp_6279[3] <== [evals[72][0] - evals[56][0], evals[72][1] - evals[56][1], evals[72][2] - evals[56][2]];
    signal tmp_6280[3] <== CMul()(evals[40], tmp_6279);
    signal tmp_6281[3] <== [tmp_6278[0] + tmp_6280[0], tmp_6278[1] + tmp_6280[1], tmp_6278[2] + tmp_6280[2]];
    signal tmp_6282[3] <== CMul()(challengeQ, tmp_6281);
    signal tmp_6283[3] <== CMul()(tmp_6183, evals[57]);
    signal tmp_6284[3] <== CMul()(tmp_6186, evals[53]);
    signal tmp_6285[3] <== [tmp_6283[0] + tmp_6284[0], tmp_6283[1] + tmp_6284[1], tmp_6283[2] + tmp_6284[2]];
    signal tmp_6286[3] <== [tmp_6188[0] + tmp_6190[0], tmp_6188[1] + tmp_6190[1], tmp_6188[2] + tmp_6190[2]];
    signal tmp_6287[3] <== CMul()(tmp_6286, evals[61]);
    signal tmp_6288[3] <== [tmp_6285[0] + tmp_6287[0], tmp_6285[1] + tmp_6287[1], tmp_6285[2] + tmp_6287[2]];
    signal tmp_6289[3] <== [evals[73][0] - tmp_6288[0], evals[73][1] - tmp_6288[1], evals[73][2] - tmp_6288[2]];
    signal tmp_6290[3] <== CMul()(evals[39], tmp_6289);
    signal tmp_6291[3] <== [tmp_6282[0] + tmp_6290[0], tmp_6282[1] + tmp_6290[1], tmp_6282[2] + tmp_6290[2]];
    signal tmp_6292[3] <== CMul()(challengeQ, tmp_6291);
    signal tmp_6293[3] <== [evals[73][0] - evals[57][0], evals[73][1] - evals[57][1], evals[73][2] - evals[57][2]];
    signal tmp_6294[3] <== CMul()(evals[40], tmp_6293);
    signal tmp_6295[3] <== [tmp_6292[0] + tmp_6294[0], tmp_6292[1] + tmp_6294[1], tmp_6292[2] + tmp_6294[2]];
    signal tmp_6296[3] <== CMul()(challengeQ, tmp_6295);
    signal tmp_6297[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6298[3] <== CMul()(tmp_6297, evals[58]);
    signal tmp_6299[3] <== CMul()(tmp_6188, evals[50]);
    signal tmp_6300[3] <== [tmp_6298[0] + tmp_6299[0], tmp_6298[1] + tmp_6299[1], tmp_6298[2] + tmp_6299[2]];
    signal tmp_6301[3] <== CMul()(tmp_6190, evals[62]);
    signal tmp_6302[3] <== [tmp_6300[0] + tmp_6301[0], tmp_6300[1] + tmp_6301[1], tmp_6300[2] + tmp_6301[2]];
    signal tmp_6303[3] <== [evals[74][0] - tmp_6302[0], evals[74][1] - tmp_6302[1], evals[74][2] - tmp_6302[2]];
    signal tmp_6304[3] <== CMul()(evals[39], tmp_6303);
    signal tmp_6305[3] <== [tmp_6296[0] + tmp_6304[0], tmp_6296[1] + tmp_6304[1], tmp_6296[2] + tmp_6304[2]];
    signal tmp_6306[3] <== CMul()(challengeQ, tmp_6305);
    signal tmp_6307[3] <== [evals[74][0] - evals[58][0], evals[74][1] - evals[58][1], evals[74][2] - evals[58][2]];
    signal tmp_6308[3] <== CMul()(evals[40], tmp_6307);
    signal tmp_6309[3] <== [tmp_6306[0] + tmp_6308[0], tmp_6306[1] + tmp_6308[1], tmp_6306[2] + tmp_6308[2]];
    signal tmp_6310[3] <== CMul()(challengeQ, tmp_6309);
    signal tmp_6311[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6312[3] <== CMul()(tmp_6311, evals[59]);
    signal tmp_6313[3] <== CMul()(tmp_6188, evals[51]);
    signal tmp_6314[3] <== [tmp_6312[0] + tmp_6313[0], tmp_6312[1] + tmp_6313[1], tmp_6312[2] + tmp_6313[2]];
    signal tmp_6315[3] <== CMul()(tmp_6190, evals[63]);
    signal tmp_6316[3] <== [tmp_6314[0] + tmp_6315[0], tmp_6314[1] + tmp_6315[1], tmp_6314[2] + tmp_6315[2]];
    signal tmp_6317[3] <== [evals[75][0] - tmp_6316[0], evals[75][1] - tmp_6316[1], evals[75][2] - tmp_6316[2]];
    signal tmp_6318[3] <== CMul()(evals[39], tmp_6317);
    signal tmp_6319[3] <== [tmp_6310[0] + tmp_6318[0], tmp_6310[1] + tmp_6318[1], tmp_6310[2] + tmp_6318[2]];
    signal tmp_6320[3] <== CMul()(challengeQ, tmp_6319);
    signal tmp_6321[3] <== [evals[75][0] - evals[59][0], evals[75][1] - evals[59][1], evals[75][2] - evals[59][2]];
    signal tmp_6322[3] <== CMul()(evals[40], tmp_6321);
    signal tmp_6323[3] <== [tmp_6320[0] + tmp_6322[0], tmp_6320[1] + tmp_6322[1], tmp_6320[2] + tmp_6322[2]];
    signal tmp_6324[3] <== CMul()(challengeQ, tmp_6323);
    signal tmp_6325[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6326[3] <== CMul()(tmp_6325, evals[60]);
    signal tmp_6327[3] <== CMul()(tmp_6188, evals[52]);
    signal tmp_6328[3] <== [tmp_6326[0] + tmp_6327[0], tmp_6326[1] + tmp_6327[1], tmp_6326[2] + tmp_6327[2]];
    signal tmp_6329[3] <== CMul()(tmp_6190, evals[64]);
    signal tmp_6330[3] <== [tmp_6328[0] + tmp_6329[0], tmp_6328[1] + tmp_6329[1], tmp_6328[2] + tmp_6329[2]];
    signal tmp_6331[3] <== [evals[76][0] - tmp_6330[0], evals[76][1] - tmp_6330[1], evals[76][2] - tmp_6330[2]];
    signal tmp_6332[3] <== CMul()(evals[39], tmp_6331);
    signal tmp_6333[3] <== [tmp_6324[0] + tmp_6332[0], tmp_6324[1] + tmp_6332[1], tmp_6324[2] + tmp_6332[2]];
    signal tmp_6334[3] <== CMul()(challengeQ, tmp_6333);
    signal tmp_6335[3] <== [evals[76][0] - evals[60][0], evals[76][1] - evals[60][1], evals[76][2] - evals[60][2]];
    signal tmp_6336[3] <== CMul()(evals[40], tmp_6335);
    signal tmp_6337[3] <== [tmp_6334[0] + tmp_6336[0], tmp_6334[1] + tmp_6336[1], tmp_6334[2] + tmp_6336[2]];
    signal tmp_6338[3] <== CMul()(challengeQ, tmp_6337);
    signal tmp_6339[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6340[3] <== CMul()(tmp_6339, evals[61]);
    signal tmp_6341[3] <== CMul()(tmp_6188, evals[53]);
    signal tmp_6342[3] <== [tmp_6340[0] + tmp_6341[0], tmp_6340[1] + tmp_6341[1], tmp_6340[2] + tmp_6341[2]];
    signal tmp_6343[3] <== CMul()(tmp_6190, evals[65]);
    signal tmp_6344[3] <== [tmp_6342[0] + tmp_6343[0], tmp_6342[1] + tmp_6343[1], tmp_6342[2] + tmp_6343[2]];
    signal tmp_6345[3] <== [evals[77][0] - tmp_6344[0], evals[77][1] - tmp_6344[1], evals[77][2] - tmp_6344[2]];
    signal tmp_6346[3] <== CMul()(evals[39], tmp_6345);
    signal tmp_6347[3] <== [tmp_6338[0] + tmp_6346[0], tmp_6338[1] + tmp_6346[1], tmp_6338[2] + tmp_6346[2]];
    signal tmp_6348[3] <== CMul()(challengeQ, tmp_6347);
    signal tmp_6349[3] <== [evals[77][0] - evals[61][0], evals[77][1] - evals[61][1], evals[77][2] - evals[61][2]];
    signal tmp_6350[3] <== CMul()(evals[40], tmp_6349);
    signal tmp_6351[3] <== [tmp_6348[0] + tmp_6350[0], tmp_6348[1] + tmp_6350[1], tmp_6348[2] + tmp_6350[2]];
    signal tmp_6352[3] <== CMul()(challengeQ, tmp_6351);
    signal tmp_6353[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6354[3] <== [tmp_6353[0] + tmp_6188[0], tmp_6353[1] + tmp_6188[1], tmp_6353[2] + tmp_6188[2]];
    signal tmp_6355[3] <== CMul()(tmp_6354, evals[62]);
    signal tmp_6356[3] <== CMul()(tmp_6190, evals[50]);
    signal tmp_6357[3] <== [tmp_6355[0] + tmp_6356[0], tmp_6355[1] + tmp_6356[1], tmp_6355[2] + tmp_6356[2]];
    signal tmp_6358[3] <== [evals[78][0] - tmp_6357[0], evals[78][1] - tmp_6357[1], evals[78][2] - tmp_6357[2]];
    signal tmp_6359[3] <== CMul()(evals[39], tmp_6358);
    signal tmp_6360[3] <== [tmp_6352[0] + tmp_6359[0], tmp_6352[1] + tmp_6359[1], tmp_6352[2] + tmp_6359[2]];
    signal tmp_6361[3] <== CMul()(challengeQ, tmp_6360);
    signal tmp_6362[3] <== [evals[78][0] - evals[62][0], evals[78][1] - evals[62][1], evals[78][2] - evals[62][2]];
    signal tmp_6363[3] <== CMul()(evals[40], tmp_6362);
    signal tmp_6364[3] <== [tmp_6361[0] + tmp_6363[0], tmp_6361[1] + tmp_6363[1], tmp_6361[2] + tmp_6363[2]];
    signal tmp_6365[3] <== CMul()(challengeQ, tmp_6364);
    signal tmp_6366[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6367[3] <== [tmp_6366[0] + tmp_6188[0], tmp_6366[1] + tmp_6188[1], tmp_6366[2] + tmp_6188[2]];
    signal tmp_6368[3] <== CMul()(tmp_6367, evals[63]);
    signal tmp_6369[3] <== CMul()(tmp_6190, evals[51]);
    signal tmp_6370[3] <== [tmp_6368[0] + tmp_6369[0], tmp_6368[1] + tmp_6369[1], tmp_6368[2] + tmp_6369[2]];
    signal tmp_6371[3] <== [evals[79][0] - tmp_6370[0], evals[79][1] - tmp_6370[1], evals[79][2] - tmp_6370[2]];
    signal tmp_6372[3] <== CMul()(evals[39], tmp_6371);
    signal tmp_6373[3] <== [tmp_6365[0] + tmp_6372[0], tmp_6365[1] + tmp_6372[1], tmp_6365[2] + tmp_6372[2]];
    signal tmp_6374[3] <== CMul()(challengeQ, tmp_6373);
    signal tmp_6375[3] <== [evals[79][0] - evals[63][0], evals[79][1] - evals[63][1], evals[79][2] - evals[63][2]];
    signal tmp_6376[3] <== CMul()(evals[40], tmp_6375);
    signal tmp_6377[3] <== [tmp_6374[0] + tmp_6376[0], tmp_6374[1] + tmp_6376[1], tmp_6374[2] + tmp_6376[2]];
    signal tmp_6378[3] <== CMul()(challengeQ, tmp_6377);
    signal tmp_6379[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6380[3] <== [tmp_6379[0] + tmp_6188[0], tmp_6379[1] + tmp_6188[1], tmp_6379[2] + tmp_6188[2]];
    signal tmp_6381[3] <== CMul()(tmp_6380, evals[64]);
    signal tmp_6382[3] <== CMul()(tmp_6190, evals[52]);
    signal tmp_6383[3] <== [tmp_6381[0] + tmp_6382[0], tmp_6381[1] + tmp_6382[1], tmp_6381[2] + tmp_6382[2]];
    signal tmp_6384[3] <== [evals[80][0] - tmp_6383[0], evals[80][1] - tmp_6383[1], evals[80][2] - tmp_6383[2]];
    signal tmp_6385[3] <== CMul()(evals[39], tmp_6384);
    signal tmp_6386[3] <== [tmp_6378[0] + tmp_6385[0], tmp_6378[1] + tmp_6385[1], tmp_6378[2] + tmp_6385[2]];
    signal tmp_6387[3] <== CMul()(challengeQ, tmp_6386);
    signal tmp_6388[3] <== [evals[80][0] - evals[64][0], evals[80][1] - evals[64][1], evals[80][2] - evals[64][2]];
    signal tmp_6389[3] <== CMul()(evals[40], tmp_6388);
    signal tmp_6390[3] <== [tmp_6387[0] + tmp_6389[0], tmp_6387[1] + tmp_6389[1], tmp_6387[2] + tmp_6389[2]];
    signal tmp_6391[3] <== CMul()(challengeQ, tmp_6390);
    signal tmp_6392[3] <== [tmp_6183[0] + tmp_6186[0], tmp_6183[1] + tmp_6186[1], tmp_6183[2] + tmp_6186[2]];
    signal tmp_6393[3] <== [tmp_6392[0] + tmp_6188[0], tmp_6392[1] + tmp_6188[1], tmp_6392[2] + tmp_6188[2]];
    signal tmp_6394[3] <== CMul()(tmp_6393, evals[65]);
    signal tmp_6395[3] <== CMul()(tmp_6190, evals[53]);
    signal tmp_6396[3] <== [tmp_6394[0] + tmp_6395[0], tmp_6394[1] + tmp_6395[1], tmp_6394[2] + tmp_6395[2]];
    signal tmp_6397[3] <== [evals[81][0] - tmp_6396[0], evals[81][1] - tmp_6396[1], evals[81][2] - tmp_6396[2]];
    signal tmp_6398[3] <== CMul()(evals[39], tmp_6397);
    signal tmp_6399[3] <== [tmp_6391[0] + tmp_6398[0], tmp_6391[1] + tmp_6398[1], tmp_6391[2] + tmp_6398[2]];
    signal tmp_6400[3] <== CMul()(challengeQ, tmp_6399);
    signal tmp_6401[3] <== [evals[81][0] - evals[65][0], evals[81][1] - evals[65][1], evals[81][2] - evals[65][2]];
    signal tmp_6402[3] <== CMul()(evals[40], tmp_6401);
    signal tmp_6403[3] <== [tmp_6400[0] + tmp_6402[0], tmp_6400[1] + tmp_6402[1], tmp_6400[2] + tmp_6402[2]];
    signal tmp_6404[3] <== CMul()(challengeQ, tmp_6403);
    tmp_6405 <== [evals[39][0] + evals[40][0], evals[39][1] + evals[40][1], evals[39][2] + evals[40][2]];
    signal tmp_6406[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6407[3] <== [tmp_6406[0] + evals[109][0], tmp_6406[1] + evals[109][1], tmp_6406[2] + evals[109][2]];
    signal tmp_6408[3] <== [tmp_6407[0] + evals[41][0], tmp_6407[1] + evals[41][1], tmp_6407[2] + evals[41][2]];
    signal tmp_6409[3] <== [tmp_6405[0] * 1579613653969377219, tmp_6405[1] * 1579613653969377219, tmp_6405[2] * 1579613653969377219];
    signal tmp_6410[3] <== [evals[66][0] + tmp_6409[0], evals[66][1] + tmp_6409[1], evals[66][2] + tmp_6409[2]];
    signal tmp_6411[3] <== CMul()(tmp_6410, tmp_6410);
    signal tmp_6412[3] <== CMul()(tmp_6411, tmp_6411);
    signal tmp_6413[3] <== CMul()(tmp_6412, tmp_6411);
    tmp_6414 <== CMul()(tmp_6413, tmp_6410);
    signal tmp_6415[3] <== [tmp_6405[0] * 7845454723070343021, tmp_6405[1] * 7845454723070343021, tmp_6405[2] * 7845454723070343021];
    signal tmp_6416[3] <== [tmp_6066[0] * 14871271697671130741, tmp_6066[1] * 14871271697671130741, tmp_6066[2] * 14871271697671130741];
    signal tmp_6417[3] <== [tmp_6415[0] + tmp_6416[0], tmp_6415[1] + tmp_6416[1], tmp_6415[2] + tmp_6416[2]];
    signal tmp_6418[3] <== [evals[109][0] * 967731447871038855, evals[109][1] * 967731447871038855, evals[109][2] * 967731447871038855];
    signal tmp_6419[3] <== [tmp_6417[0] + tmp_6418[0], tmp_6417[1] + tmp_6418[1], tmp_6417[2] + tmp_6418[2]];
    signal tmp_6420[3] <== [evals[41][0] * 10880189488914803976, evals[41][1] * 10880189488914803976, evals[41][2] * 10880189488914803976];
    tmp_6421 <== [tmp_6419[0] + tmp_6420[0], tmp_6419[1] + tmp_6420[1], tmp_6419[2] + tmp_6420[2]];
    signal tmp_6422[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6423[3] <== [tmp_6405[0] * 15509510893087893340, tmp_6405[1] * 15509510893087893340, tmp_6405[2] * 15509510893087893340];
    signal tmp_6424[3] <== [evals[67][0] + tmp_6423[0], evals[67][1] + tmp_6423[1], evals[67][2] + tmp_6423[2]];
    signal tmp_6425[3] <== CMul()(tmp_6424, tmp_6424);
    signal tmp_6426[3] <== CMul()(tmp_6425, tmp_6425);
    signal tmp_6427[3] <== CMul()(tmp_6426, tmp_6425);
    tmp_6428 <== CMul()(tmp_6427, tmp_6424);
    signal tmp_6429[3] <== [tmp_6405[0] * 4429017498140175847, tmp_6405[1] * 4429017498140175847, tmp_6405[2] * 4429017498140175847];
    signal tmp_6430[3] <== [tmp_6066[0] * 7818640162547527607, tmp_6066[1] * 7818640162547527607, tmp_6066[2] * 7818640162547527607];
    signal tmp_6431[3] <== [tmp_6429[0] + tmp_6430[0], tmp_6429[1] + tmp_6430[1], tmp_6429[2] + tmp_6430[2]];
    signal tmp_6432[3] <== [evals[109][0] * 13497390221064521284, evals[109][1] * 13497390221064521284, evals[109][2] * 13497390221064521284];
    signal tmp_6433[3] <== [tmp_6431[0] + tmp_6432[0], tmp_6431[1] + tmp_6432[1], tmp_6431[2] + tmp_6432[2]];
    signal tmp_6434[3] <== [evals[41][0] * 10817563486901345506, evals[41][1] * 10817563486901345506, evals[41][2] * 10817563486901345506];
    tmp_6435 <== [tmp_6433[0] + tmp_6434[0], tmp_6433[1] + tmp_6434[1], tmp_6433[2] + tmp_6434[2]];
    signal tmp_6436[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6437[3] <== [tmp_6422[0] + tmp_6436[0], tmp_6422[1] + tmp_6436[1], tmp_6422[2] + tmp_6436[2]];
    signal tmp_6438[3] <== [tmp_6405[0] * 10090715174060125222, tmp_6405[1] * 10090715174060125222, tmp_6405[2] * 10090715174060125222];
    signal tmp_6439[3] <== [evals[68][0] + tmp_6438[0], evals[68][1] + tmp_6438[1], evals[68][2] + tmp_6438[2]];
    signal tmp_6440[3] <== CMul()(tmp_6439, tmp_6439);
    signal tmp_6441[3] <== CMul()(tmp_6440, tmp_6440);
    signal tmp_6442[3] <== CMul()(tmp_6441, tmp_6440);
    tmp_6443 <== CMul()(tmp_6442, tmp_6439);
    signal tmp_6444[3] <== [tmp_6405[0] * 2092211485884675791, tmp_6405[1] * 2092211485884675791, tmp_6405[2] * 2092211485884675791];
    signal tmp_6445[3] <== [tmp_6066[0] * 10207849655710394742, tmp_6066[1] * 10207849655710394742, tmp_6066[2] * 10207849655710394742];
    signal tmp_6446[3] <== [tmp_6444[0] + tmp_6445[0], tmp_6444[1] + tmp_6445[1], tmp_6444[2] + tmp_6445[2]];
    signal tmp_6447[3] <== [evals[109][0] * 7079992912276546290, evals[109][1] * 7079992912276546290, evals[109][2] * 7079992912276546290];
    signal tmp_6448[3] <== [tmp_6446[0] + tmp_6447[0], tmp_6446[1] + tmp_6447[1], tmp_6446[2] + tmp_6447[2]];
    signal tmp_6449[3] <== [evals[41][0] * 5595838894652252793, evals[41][1] * 5595838894652252793, evals[41][2] * 5595838894652252793];
    tmp_6450 <== [tmp_6448[0] + tmp_6449[0], tmp_6448[1] + tmp_6449[1], tmp_6448[2] + tmp_6449[2]];
    signal tmp_6451[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6452[3] <== [51 * tmp_6451[0], 51 * tmp_6451[1], 51 * tmp_6451[2]];
    signal tmp_6453[3] <== [tmp_6437[0] + tmp_6452[0], tmp_6437[1] + tmp_6452[1], tmp_6437[2] + tmp_6452[2]];
    signal tmp_6454[3] <== [tmp_6405[0] * 5625716564419252202, tmp_6405[1] * 5625716564419252202, tmp_6405[2] * 5625716564419252202];
    signal tmp_6455[3] <== [evals[69][0] + tmp_6454[0], evals[69][1] + tmp_6454[1], evals[69][2] + tmp_6454[2]];
    signal tmp_6456[3] <== CMul()(tmp_6455, tmp_6455);
    signal tmp_6457[3] <== CMul()(tmp_6456, tmp_6456);
    signal tmp_6458[3] <== CMul()(tmp_6457, tmp_6456);
    tmp_6459 <== CMul()(tmp_6458, tmp_6455);
    signal tmp_6460[3] <== [tmp_6405[0] * 3672938463800022916, tmp_6405[1] * 3672938463800022916, tmp_6405[2] * 3672938463800022916];
    signal tmp_6461[3] <== [tmp_6066[0] * 17997741013544777071, tmp_6066[1] * 17997741013544777071, tmp_6066[2] * 17997741013544777071];
    signal tmp_6462[3] <== [tmp_6460[0] + tmp_6461[0], tmp_6460[1] + tmp_6461[1], tmp_6460[2] + tmp_6461[2]];
    signal tmp_6463[3] <== [evals[109][0] * 4056766622830105470, evals[109][1] * 4056766622830105470, evals[109][2] * 4056766622830105470];
    signal tmp_6464[3] <== [tmp_6462[0] + tmp_6463[0], tmp_6462[1] + tmp_6463[1], tmp_6462[2] + tmp_6463[2]];
    signal tmp_6465[3] <== [evals[41][0] * 7339830491681595439, evals[41][1] * 7339830491681595439, evals[41][2] * 7339830491681595439];
    tmp_6466 <== [tmp_6464[0] + tmp_6465[0], tmp_6464[1] + tmp_6465[1], tmp_6464[2] + tmp_6465[2]];
    signal tmp_6467[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6468[3] <== [tmp_6453[0] + tmp_6467[0], tmp_6453[1] + tmp_6467[1], tmp_6453[2] + tmp_6467[2]];
    signal tmp_6469[3] <== [tmp_6405[0] * 3006005019077469174, tmp_6405[1] * 3006005019077469174, tmp_6405[2] * 3006005019077469174];
    signal tmp_6470[3] <== [evals[70][0] + tmp_6469[0], evals[70][1] + tmp_6469[1], evals[70][2] + tmp_6469[2]];
    signal tmp_6471[3] <== CMul()(tmp_6470, tmp_6470);
    signal tmp_6472[3] <== CMul()(tmp_6471, tmp_6471);
    signal tmp_6473[3] <== CMul()(tmp_6472, tmp_6471);
    tmp_6474 <== CMul()(tmp_6473, tmp_6470);
    signal tmp_6475[3] <== [tmp_6405[0] * 14908811154314031538, tmp_6405[1] * 14908811154314031538, tmp_6405[2] * 14908811154314031538];
    signal tmp_6476[3] <== [tmp_6066[0] * 16991056560447799587, tmp_6066[1] * 16991056560447799587, tmp_6066[2] * 16991056560447799587];
    signal tmp_6477[3] <== [tmp_6475[0] + tmp_6476[0], tmp_6475[1] + tmp_6476[1], tmp_6475[2] + tmp_6476[2]];
    signal tmp_6478[3] <== [evals[109][0] * 6323166457980539990, evals[109][1] * 6323166457980539990, evals[109][2] * 6323166457980539990];
    signal tmp_6479[3] <== [tmp_6477[0] + tmp_6478[0], tmp_6477[1] + tmp_6478[1], tmp_6477[2] + tmp_6478[2]];
    signal tmp_6480[3] <== [evals[41][0] * 13024184231170405476, evals[41][1] * 13024184231170405476, evals[41][2] * 13024184231170405476];
    tmp_6481 <== [tmp_6479[0] + tmp_6480[0], tmp_6479[1] + tmp_6480[1], tmp_6479[2] + tmp_6480[2]];
    signal tmp_6482[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6483[3] <== [11 * tmp_6482[0], 11 * tmp_6482[1], 11 * tmp_6482[2]];
    signal tmp_6484[3] <== [tmp_6468[0] + tmp_6483[0], tmp_6468[1] + tmp_6483[1], tmp_6468[2] + tmp_6483[2]];
    signal tmp_6485[3] <== [tmp_6405[0] * 18314693207014427912, tmp_6405[1] * 18314693207014427912, tmp_6405[2] * 18314693207014427912];
    signal tmp_6486[3] <== [evals[71][0] + tmp_6485[0], evals[71][1] + tmp_6485[1], evals[71][2] + tmp_6485[2]];
    signal tmp_6487[3] <== CMul()(tmp_6486, tmp_6486);
    signal tmp_6488[3] <== CMul()(tmp_6487, tmp_6487);
    signal tmp_6489[3] <== CMul()(tmp_6488, tmp_6487);
    tmp_6490 <== CMul()(tmp_6489, tmp_6486);
    signal tmp_6491[3] <== [tmp_6405[0] * 6573822840603111307, tmp_6405[1] * 6573822840603111307, tmp_6405[2] * 6573822840603111307];
    signal tmp_6492[3] <== [tmp_6066[0] * 348197253806644292, tmp_6066[1] * 348197253806644292, tmp_6066[2] * 348197253806644292];
    signal tmp_6493[3] <== [tmp_6491[0] + tmp_6492[0], tmp_6491[1] + tmp_6492[1], tmp_6491[2] + tmp_6492[2]];
    signal tmp_6494[3] <== [evals[109][0] * 13419047640011651400, evals[109][1] * 13419047640011651400, evals[109][2] * 13419047640011651400];
    signal tmp_6495[3] <== [tmp_6493[0] + tmp_6494[0], tmp_6493[1] + tmp_6494[1], tmp_6493[2] + tmp_6494[2]];
    signal tmp_6496[3] <== [evals[41][0] * 7496303001397288101, evals[41][1] * 7496303001397288101, evals[41][2] * 7496303001397288101];
    tmp_6497 <== [tmp_6495[0] + tmp_6496[0], tmp_6495[1] + tmp_6496[1], tmp_6495[2] + tmp_6496[2]];
    signal tmp_6498[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6499[3] <== [17 * tmp_6498[0], 17 * tmp_6498[1], 17 * tmp_6498[2]];
    signal tmp_6500[3] <== [tmp_6484[0] + tmp_6499[0], tmp_6484[1] + tmp_6499[1], tmp_6484[2] + tmp_6499[2]];
    signal tmp_6501[3] <== [tmp_6405[0] * 10170571510627764565, tmp_6405[1] * 10170571510627764565, tmp_6405[2] * 10170571510627764565];
    signal tmp_6502[3] <== [evals[72][0] + tmp_6501[0], evals[72][1] + tmp_6501[1], evals[72][2] + tmp_6501[2]];
    signal tmp_6503[3] <== CMul()(tmp_6502, tmp_6502);
    signal tmp_6504[3] <== CMul()(tmp_6503, tmp_6503);
    signal tmp_6505[3] <== CMul()(tmp_6504, tmp_6503);
    tmp_6506 <== CMul()(tmp_6505, tmp_6502);
    signal tmp_6507[3] <== [tmp_6405[0] * 10909691831624859236, tmp_6405[1] * 10909691831624859236, tmp_6405[2] * 10909691831624859236];
    signal tmp_6508[3] <== [tmp_6066[0] * 3804216850635206858, tmp_6066[1] * 3804216850635206858, tmp_6066[2] * 3804216850635206858];
    signal tmp_6509[3] <== [tmp_6507[0] + tmp_6508[0], tmp_6507[1] + tmp_6508[1], tmp_6507[2] + tmp_6508[2]];
    signal tmp_6510[3] <== [evals[109][0] * 12588714469852125705, evals[109][1] * 12588714469852125705, evals[109][2] * 12588714469852125705];
    signal tmp_6511[3] <== [tmp_6509[0] + tmp_6510[0], tmp_6509[1] + tmp_6510[1], tmp_6509[2] + tmp_6510[2]];
    signal tmp_6512[3] <== [evals[41][0] * 1481011955045677708, evals[41][1] * 1481011955045677708, evals[41][2] * 1481011955045677708];
    tmp_6513 <== [tmp_6511[0] + tmp_6512[0], tmp_6511[1] + tmp_6512[1], tmp_6511[2] + tmp_6512[2]];
    signal tmp_6514[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6515[3] <== [2 * tmp_6514[0], 2 * tmp_6514[1], 2 * tmp_6514[2]];
    signal tmp_6516[3] <== [tmp_6500[0] + tmp_6515[0], tmp_6500[1] + tmp_6515[1], tmp_6500[2] + tmp_6515[2]];
    signal tmp_6517[3] <== [tmp_6405[0] * 2027625550790675754, tmp_6405[1] * 2027625550790675754, tmp_6405[2] * 2027625550790675754];
    signal tmp_6518[3] <== [evals[73][0] + tmp_6517[0], evals[73][1] + tmp_6517[1], evals[73][2] + tmp_6517[2]];
    signal tmp_6519[3] <== CMul()(tmp_6518, tmp_6518);
    signal tmp_6520[3] <== CMul()(tmp_6519, tmp_6519);
    signal tmp_6521[3] <== CMul()(tmp_6520, tmp_6519);
    tmp_6522 <== CMul()(tmp_6521, tmp_6518);
    signal tmp_6523[3] <== [tmp_6405[0] * 2180104264643618635, tmp_6405[1] * 2180104264643618635, tmp_6405[2] * 2180104264643618635];
    signal tmp_6524[3] <== [tmp_6066[0] * 15878441634750804699, tmp_6066[1] * 15878441634750804699, tmp_6066[2] * 15878441634750804699];
    signal tmp_6525[3] <== [tmp_6523[0] + tmp_6524[0], tmp_6523[1] + tmp_6524[1], tmp_6523[2] + tmp_6524[2]];
    signal tmp_6526[3] <== [evals[109][0] * 9297270292248179409, evals[109][1] * 9297270292248179409, evals[109][2] * 9297270292248179409];
    signal tmp_6527[3] <== [tmp_6525[0] + tmp_6526[0], tmp_6525[1] + tmp_6526[1], tmp_6525[2] + tmp_6526[2]];
    signal tmp_6528[3] <== [evals[41][0] * 14060739868329031365, evals[41][1] * 14060739868329031365, evals[41][2] * 14060739868329031365];
    tmp_6529 <== [tmp_6527[0] + tmp_6528[0], tmp_6527[1] + tmp_6528[1], tmp_6527[2] + tmp_6528[2]];
    signal tmp_6530[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6531[3] <== [tmp_6516[0] + tmp_6530[0], tmp_6516[1] + tmp_6530[1], tmp_6516[2] + tmp_6530[2]];
    signal tmp_6532[3] <== [tmp_6405[0] * 3983470257916202094, tmp_6405[1] * 3983470257916202094, tmp_6405[2] * 3983470257916202094];
    signal tmp_6533[3] <== [evals[74][0] + tmp_6532[0], evals[74][1] + tmp_6532[1], evals[74][2] + tmp_6532[2]];
    signal tmp_6534[3] <== CMul()(tmp_6533, tmp_6533);
    signal tmp_6535[3] <== CMul()(tmp_6534, tmp_6534);
    signal tmp_6536[3] <== CMul()(tmp_6535, tmp_6534);
    tmp_6537 <== CMul()(tmp_6536, tmp_6533);
    signal tmp_6538[3] <== [tmp_6405[0] * 17674822478918492543, tmp_6405[1] * 17674822478918492543, tmp_6405[2] * 17674822478918492543];
    signal tmp_6539[3] <== [tmp_6066[0] * 10783876336999051732, tmp_6066[1] * 10783876336999051732, tmp_6066[2] * 10783876336999051732];
    signal tmp_6540[3] <== [tmp_6538[0] + tmp_6539[0], tmp_6538[1] + tmp_6539[1], tmp_6538[2] + tmp_6539[2]];
    signal tmp_6541[3] <== [evals[109][0] * 17023507959532517322, evals[109][1] * 17023507959532517322, evals[109][2] * 17023507959532517322];
    signal tmp_6542[3] <== [tmp_6540[0] + tmp_6541[0], tmp_6540[1] + tmp_6541[1], tmp_6540[2] + tmp_6541[2]];
    signal tmp_6543[3] <== [evals[41][0] * 13581824287786003086, evals[41][1] * 13581824287786003086, evals[41][2] * 13581824287786003086];
    tmp_6544 <== [tmp_6542[0] + tmp_6543[0], tmp_6542[1] + tmp_6543[1], tmp_6542[2] + tmp_6543[2]];
    signal tmp_6545[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6546[3] <== [101 * tmp_6545[0], 101 * tmp_6545[1], 101 * tmp_6545[2]];
    signal tmp_6547[3] <== [tmp_6531[0] + tmp_6546[0], tmp_6531[1] + tmp_6546[1], tmp_6531[2] + tmp_6546[2]];
    signal tmp_6548[3] <== [tmp_6405[0] * 3423470109396435354, tmp_6405[1] * 3423470109396435354, tmp_6405[2] * 3423470109396435354];
    signal tmp_6549[3] <== [evals[75][0] + tmp_6548[0], evals[75][1] + tmp_6548[1], evals[75][2] + tmp_6548[2]];
    signal tmp_6550[3] <== CMul()(tmp_6549, tmp_6549);
    signal tmp_6551[3] <== CMul()(tmp_6550, tmp_6550);
    signal tmp_6552[3] <== CMul()(tmp_6551, tmp_6550);
    tmp_6553 <== CMul()(tmp_6552, tmp_6549);
    signal tmp_6554[3] <== [tmp_6405[0] * 18417266132335741394, tmp_6405[1] * 18417266132335741394, tmp_6405[2] * 18417266132335741394];
    signal tmp_6555[3] <== [tmp_6066[0] * 6249355405258274459, tmp_6066[1] * 6249355405258274459, tmp_6066[2] * 6249355405258274459];
    signal tmp_6556[3] <== [tmp_6554[0] + tmp_6555[0], tmp_6554[1] + tmp_6555[1], tmp_6554[2] + tmp_6555[2]];
    signal tmp_6557[3] <== [evals[109][0] * 9382136940971401256, evals[109][1] * 9382136940971401256, evals[109][2] * 9382136940971401256];
    signal tmp_6558[3] <== [tmp_6556[0] + tmp_6557[0], tmp_6556[1] + tmp_6557[1], tmp_6556[2] + tmp_6557[2]];
    signal tmp_6559[3] <== [evals[41][0] * 101626575556905271, evals[41][1] * 101626575556905271, evals[41][2] * 101626575556905271];
    tmp_6560 <== [tmp_6558[0] + tmp_6559[0], tmp_6558[1] + tmp_6559[1], tmp_6558[2] + tmp_6559[2]];
    signal tmp_6561[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6562[3] <== [63 * tmp_6561[0], 63 * tmp_6561[1], 63 * tmp_6561[2]];
    signal tmp_6563[3] <== [tmp_6547[0] + tmp_6562[0], tmp_6547[1] + tmp_6562[1], tmp_6547[2] + tmp_6562[2]];
    signal tmp_6564[3] <== [tmp_6405[0] * 3450488264035752368, tmp_6405[1] * 3450488264035752368, tmp_6405[2] * 3450488264035752368];
    signal tmp_6565[3] <== [evals[76][0] + tmp_6564[0], evals[76][1] + tmp_6564[1], evals[76][2] + tmp_6564[2]];
    signal tmp_6566[3] <== CMul()(tmp_6565, tmp_6565);
    signal tmp_6567[3] <== CMul()(tmp_6566, tmp_6566);
    signal tmp_6568[3] <== CMul()(tmp_6567, tmp_6566);
    tmp_6569 <== CMul()(tmp_6568, tmp_6565);
    signal tmp_6570[3] <== [tmp_6405[0] * 11589569604681103119, tmp_6405[1] * 11589569604681103119, tmp_6405[2] * 11589569604681103119];
    signal tmp_6571[3] <== [tmp_6066[0] * 18338300173971397155, tmp_6066[1] * 18338300173971397155, tmp_6066[2] * 18338300173971397155];
    signal tmp_6572[3] <== [tmp_6570[0] + tmp_6571[0], tmp_6570[1] + tmp_6571[1], tmp_6570[2] + tmp_6571[2]];
    signal tmp_6573[3] <== [evals[109][0] * 14883732890265648356, evals[109][1] * 14883732890265648356, evals[109][2] * 14883732890265648356];
    signal tmp_6574[3] <== [tmp_6572[0] + tmp_6573[0], tmp_6572[1] + tmp_6573[1], tmp_6572[2] + tmp_6573[2]];
    signal tmp_6575[3] <== [evals[41][0] * 1195190930009648444, evals[41][1] * 1195190930009648444, evals[41][2] * 1195190930009648444];
    tmp_6576 <== [tmp_6574[0] + tmp_6575[0], tmp_6574[1] + tmp_6575[1], tmp_6574[2] + tmp_6575[2]];
    signal tmp_6577[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6578[3] <== [15 * tmp_6577[0], 15 * tmp_6577[1], 15 * tmp_6577[2]];
    signal tmp_6579[3] <== [tmp_6563[0] + tmp_6578[0], tmp_6563[1] + tmp_6578[1], tmp_6563[2] + tmp_6578[2]];
    signal tmp_6580[3] <== [tmp_6405[0] * 3151070045406026687, tmp_6405[1] * 3151070045406026687, tmp_6405[2] * 3151070045406026687];
    signal tmp_6581[3] <== [evals[77][0] + tmp_6580[0], evals[77][1] + tmp_6580[1], evals[77][2] + tmp_6580[2]];
    signal tmp_6582[3] <== CMul()(tmp_6581, tmp_6581);
    signal tmp_6583[3] <== CMul()(tmp_6582, tmp_6582);
    signal tmp_6584[3] <== CMul()(tmp_6583, tmp_6582);
    tmp_6585 <== CMul()(tmp_6584, tmp_6581);
    signal tmp_6586[3] <== [tmp_6405[0] * 18392483772870992139, tmp_6405[1] * 18392483772870992139, tmp_6405[2] * 18392483772870992139];
    signal tmp_6587[3] <== [tmp_6066[0] * 18008467388369492623, tmp_6066[1] * 18008467388369492623, tmp_6066[2] * 18008467388369492623];
    signal tmp_6588[3] <== [tmp_6586[0] + tmp_6587[0], tmp_6586[1] + tmp_6587[1], tmp_6586[2] + tmp_6587[2]];
    signal tmp_6589[3] <== [evals[109][0] * 17444658652079921032, evals[109][1] * 17444658652079921032, evals[109][2] * 17444658652079921032];
    signal tmp_6590[3] <== [tmp_6588[0] + tmp_6589[0], tmp_6588[1] + tmp_6589[1], tmp_6588[2] + tmp_6589[2]];
    signal tmp_6591[3] <== [evals[41][0] * 1911629430057473607, evals[41][1] * 1911629430057473607, evals[41][2] * 1911629430057473607];
    tmp_6592 <== [tmp_6590[0] + tmp_6591[0], tmp_6590[1] + tmp_6591[1], tmp_6590[2] + tmp_6591[2]];
    signal tmp_6593[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6594[3] <== [2 * tmp_6593[0], 2 * tmp_6593[1], 2 * tmp_6593[2]];
    signal tmp_6595[3] <== [tmp_6579[0] + tmp_6594[0], tmp_6579[1] + tmp_6594[1], tmp_6579[2] + tmp_6594[2]];
    signal tmp_6596[3] <== [tmp_6405[0] * 13462781804006123550, tmp_6405[1] * 13462781804006123550, tmp_6405[2] * 13462781804006123550];
    signal tmp_6597[3] <== [evals[78][0] + tmp_6596[0], evals[78][1] + tmp_6596[1], evals[78][2] + tmp_6596[2]];
    signal tmp_6598[3] <== CMul()(tmp_6597, tmp_6597);
    signal tmp_6599[3] <== CMul()(tmp_6598, tmp_6598);
    signal tmp_6600[3] <== CMul()(tmp_6599, tmp_6598);
    tmp_6601 <== CMul()(tmp_6600, tmp_6597);
    signal tmp_6602[3] <== [tmp_6405[0] * 9901024618406354614, tmp_6405[1] * 9901024618406354614, tmp_6405[2] * 9901024618406354614];
    signal tmp_6603[3] <== [tmp_6066[0] * 5454689878460956614, tmp_6066[1] * 5454689878460956614, tmp_6066[2] * 5454689878460956614];
    signal tmp_6604[3] <== [tmp_6602[0] + tmp_6603[0], tmp_6602[1] + tmp_6603[1], tmp_6602[2] + tmp_6603[2]];
    signal tmp_6605[3] <== [evals[109][0] * 8737622480559860280, evals[109][1] * 8737622480559860280, evals[109][2] * 8737622480559860280];
    signal tmp_6606[3] <== [tmp_6604[0] + tmp_6605[0], tmp_6604[1] + tmp_6605[1], tmp_6604[2] + tmp_6605[2]];
    signal tmp_6607[3] <== [evals[41][0] * 4398090650023947233, evals[41][1] * 4398090650023947233, evals[41][2] * 4398090650023947233];
    tmp_6608 <== [tmp_6606[0] + tmp_6607[0], tmp_6606[1] + tmp_6607[1], tmp_6606[2] + tmp_6607[2]];
    signal tmp_6609[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6610[3] <== [67 * tmp_6609[0], 67 * tmp_6609[1], 67 * tmp_6609[2]];
    signal tmp_6611[3] <== [tmp_6595[0] + tmp_6610[0], tmp_6595[1] + tmp_6610[1], tmp_6595[2] + tmp_6610[2]];
    signal tmp_6612[3] <== [tmp_6405[0] * 13288575772684627216, tmp_6405[1] * 13288575772684627216, tmp_6405[2] * 13288575772684627216];
    signal tmp_6613[3] <== [evals[79][0] + tmp_6612[0], evals[79][1] + tmp_6612[1], evals[79][2] + tmp_6612[2]];
    signal tmp_6614[3] <== CMul()(tmp_6613, tmp_6613);
    signal tmp_6615[3] <== CMul()(tmp_6614, tmp_6614);
    signal tmp_6616[3] <== CMul()(tmp_6615, tmp_6614);
    tmp_6617 <== CMul()(tmp_6616, tmp_6613);
    signal tmp_6618[3] <== [tmp_6405[0] * 841962207097860172, tmp_6405[1] * 841962207097860172, tmp_6405[2] * 841962207097860172];
    signal tmp_6619[3] <== [tmp_6066[0] * 15539651071574326605, tmp_6066[1] * 15539651071574326605, tmp_6066[2] * 15539651071574326605];
    signal tmp_6620[3] <== [tmp_6618[0] + tmp_6619[0], tmp_6618[1] + tmp_6619[1], tmp_6618[2] + tmp_6619[2]];
    signal tmp_6621[3] <== [evals[109][0] * 9445526981937155820, evals[109][1] * 9445526981937155820, evals[109][2] * 9445526981937155820];
    signal tmp_6622[3] <== [tmp_6620[0] + tmp_6621[0], tmp_6620[1] + tmp_6621[1], tmp_6620[2] + tmp_6621[2]];
    signal tmp_6623[3] <== [evals[41][0] * 5194367552957880421, evals[41][1] * 5194367552957880421, evals[41][2] * 5194367552957880421];
    tmp_6624 <== [tmp_6622[0] + tmp_6623[0], tmp_6622[1] + tmp_6623[1], tmp_6622[2] + tmp_6623[2]];
    signal tmp_6625[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6626[3] <== [22 * tmp_6625[0], 22 * tmp_6625[1], 22 * tmp_6625[2]];
    signal tmp_6627[3] <== [tmp_6611[0] + tmp_6626[0], tmp_6611[1] + tmp_6626[1], tmp_6611[2] + tmp_6626[2]];
    signal tmp_6628[3] <== [tmp_6405[0] * 13745549378090937523, tmp_6405[1] * 13745549378090937523, tmp_6405[2] * 13745549378090937523];
    signal tmp_6629[3] <== [evals[80][0] + tmp_6628[0], evals[80][1] + tmp_6628[1], evals[80][2] + tmp_6628[2]];
    signal tmp_6630[3] <== CMul()(tmp_6629, tmp_6629);
    signal tmp_6631[3] <== CMul()(tmp_6630, tmp_6630);
    signal tmp_6632[3] <== CMul()(tmp_6631, tmp_6630);
    tmp_6633 <== CMul()(tmp_6632, tmp_6629);
    signal tmp_6634[3] <== [tmp_6405[0] * 1419811072696625524, tmp_6405[1] * 1419811072696625524, tmp_6405[2] * 1419811072696625524];
    signal tmp_6635[3] <== [tmp_6066[0] * 9583365915190359557, tmp_6066[1] * 9583365915190359557, tmp_6066[2] * 9583365915190359557];
    signal tmp_6636[3] <== [tmp_6634[0] + tmp_6635[0], tmp_6634[1] + tmp_6635[1], tmp_6634[2] + tmp_6635[2]];
    signal tmp_6637[3] <== [evals[109][0] * 14154961912193710379, evals[109][1] * 14154961912193710379, evals[109][2] * 14154961912193710379];
    signal tmp_6638[3] <== [tmp_6636[0] + tmp_6637[0], tmp_6636[1] + tmp_6637[1], tmp_6636[2] + tmp_6637[2]];
    signal tmp_6639[3] <== [evals[41][0] * 7348962900982106793, evals[41][1] * 7348962900982106793, evals[41][2] * 7348962900982106793];
    tmp_6640 <== [tmp_6638[0] + tmp_6639[0], tmp_6638[1] + tmp_6639[1], tmp_6638[2] + tmp_6639[2]];
    signal tmp_6641[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6642[3] <== [13 * tmp_6641[0], 13 * tmp_6641[1], 13 * tmp_6641[2]];
    signal tmp_6643[3] <== [tmp_6627[0] + tmp_6642[0], tmp_6627[1] + tmp_6642[1], tmp_6627[2] + tmp_6642[2]];
    signal tmp_6644[3] <== [tmp_6405[0] * 7780139165529418388, tmp_6405[1] * 7780139165529418388, tmp_6405[2] * 7780139165529418388];
    signal tmp_6645[3] <== [evals[81][0] + tmp_6644[0], evals[81][1] + tmp_6644[1], evals[81][2] + tmp_6644[2]];
    signal tmp_6646[3] <== CMul()(tmp_6645, tmp_6645);
    signal tmp_6647[3] <== CMul()(tmp_6646, tmp_6646);
    signal tmp_6648[3] <== CMul()(tmp_6647, tmp_6646);
    tmp_6649 <== CMul()(tmp_6648, tmp_6645);
    signal tmp_6650[3] <== [tmp_6405[0] * 10322277472159862957, tmp_6405[1] * 10322277472159862957, tmp_6405[2] * 10322277472159862957];
    signal tmp_6651[3] <== [tmp_6066[0] * 1022758405964188465, tmp_6066[1] * 1022758405964188465, tmp_6066[2] * 1022758405964188465];
    signal tmp_6652[3] <== [tmp_6650[0] + tmp_6651[0], tmp_6650[1] + tmp_6651[1], tmp_6650[2] + tmp_6651[2]];
    signal tmp_6653[3] <== [evals[109][0] * 5087036264788036553, evals[109][1] * 5087036264788036553, evals[109][2] * 5087036264788036553];
    signal tmp_6654[3] <== [tmp_6652[0] + tmp_6653[0], tmp_6652[1] + tmp_6653[1], tmp_6652[2] + tmp_6653[2]];
    signal tmp_6655[3] <== [evals[41][0] * 9087509974770738024, evals[41][1] * 9087509974770738024, evals[41][2] * 9087509974770738024];
    tmp_6656 <== [tmp_6654[0] + tmp_6655[0], tmp_6654[1] + tmp_6655[1], tmp_6654[2] + tmp_6655[2]];
    signal tmp_6657[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6658[3] <== [3 * tmp_6657[0], 3 * tmp_6657[1], 3 * tmp_6657[2]];
    signal tmp_6659[3] <== [tmp_6643[0] + tmp_6658[0], tmp_6643[1] + tmp_6658[1], tmp_6643[2] + tmp_6658[2]];
    signal tmp_6660[3] <== [evals[82][0] - tmp_6659[0], evals[82][1] - tmp_6659[1], evals[82][2] - tmp_6659[2]];
    signal tmp_6661[3] <== CMul()(tmp_6408, tmp_6660);
    signal tmp_6662[3] <== [tmp_6404[0] + tmp_6661[0], tmp_6404[1] + tmp_6661[1], tmp_6404[2] + tmp_6661[2]];
    signal tmp_6663[3] <== CMul()(challengeQ, tmp_6662);
    signal tmp_6664[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6665[3] <== [tmp_6664[0] + evals[109][0], tmp_6664[1] + evals[109][1], tmp_6664[2] + evals[109][2]];
    signal tmp_6666[3] <== [tmp_6665[0] + evals[41][0], tmp_6665[1] + evals[41][1], tmp_6665[2] + evals[41][2]];
    signal tmp_6667[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6668[3] <== [3 * tmp_6667[0], 3 * tmp_6667[1], 3 * tmp_6667[2]];
    signal tmp_6669[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6670[3] <== [tmp_6668[0] + tmp_6669[0], tmp_6668[1] + tmp_6669[1], tmp_6668[2] + tmp_6669[2]];
    signal tmp_6671[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6672[3] <== [tmp_6670[0] + tmp_6671[0], tmp_6670[1] + tmp_6671[1], tmp_6670[2] + tmp_6671[2]];
    signal tmp_6673[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6674[3] <== [51 * tmp_6673[0], 51 * tmp_6673[1], 51 * tmp_6673[2]];
    signal tmp_6675[3] <== [tmp_6672[0] + tmp_6674[0], tmp_6672[1] + tmp_6674[1], tmp_6672[2] + tmp_6674[2]];
    signal tmp_6676[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6677[3] <== [tmp_6675[0] + tmp_6676[0], tmp_6675[1] + tmp_6676[1], tmp_6675[2] + tmp_6676[2]];
    signal tmp_6678[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6679[3] <== [11 * tmp_6678[0], 11 * tmp_6678[1], 11 * tmp_6678[2]];
    signal tmp_6680[3] <== [tmp_6677[0] + tmp_6679[0], tmp_6677[1] + tmp_6679[1], tmp_6677[2] + tmp_6679[2]];
    signal tmp_6681[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6682[3] <== [17 * tmp_6681[0], 17 * tmp_6681[1], 17 * tmp_6681[2]];
    signal tmp_6683[3] <== [tmp_6680[0] + tmp_6682[0], tmp_6680[1] + tmp_6682[1], tmp_6680[2] + tmp_6682[2]];
    signal tmp_6684[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6685[3] <== [2 * tmp_6684[0], 2 * tmp_6684[1], 2 * tmp_6684[2]];
    signal tmp_6686[3] <== [tmp_6683[0] + tmp_6685[0], tmp_6683[1] + tmp_6685[1], tmp_6683[2] + tmp_6685[2]];
    signal tmp_6687[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6688[3] <== [tmp_6686[0] + tmp_6687[0], tmp_6686[1] + tmp_6687[1], tmp_6686[2] + tmp_6687[2]];
    signal tmp_6689[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6690[3] <== [101 * tmp_6689[0], 101 * tmp_6689[1], 101 * tmp_6689[2]];
    signal tmp_6691[3] <== [tmp_6688[0] + tmp_6690[0], tmp_6688[1] + tmp_6690[1], tmp_6688[2] + tmp_6690[2]];
    signal tmp_6692[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6693[3] <== [63 * tmp_6692[0], 63 * tmp_6692[1], 63 * tmp_6692[2]];
    signal tmp_6694[3] <== [tmp_6691[0] + tmp_6693[0], tmp_6691[1] + tmp_6693[1], tmp_6691[2] + tmp_6693[2]];
    signal tmp_6695[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6696[3] <== [15 * tmp_6695[0], 15 * tmp_6695[1], 15 * tmp_6695[2]];
    signal tmp_6697[3] <== [tmp_6694[0] + tmp_6696[0], tmp_6694[1] + tmp_6696[1], tmp_6694[2] + tmp_6696[2]];
    signal tmp_6698[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6699[3] <== [2 * tmp_6698[0], 2 * tmp_6698[1], 2 * tmp_6698[2]];
    signal tmp_6700[3] <== [tmp_6697[0] + tmp_6699[0], tmp_6697[1] + tmp_6699[1], tmp_6697[2] + tmp_6699[2]];
    signal tmp_6701[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6702[3] <== [67 * tmp_6701[0], 67 * tmp_6701[1], 67 * tmp_6701[2]];
    signal tmp_6703[3] <== [tmp_6700[0] + tmp_6702[0], tmp_6700[1] + tmp_6702[1], tmp_6700[2] + tmp_6702[2]];
    signal tmp_6704[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6705[3] <== [22 * tmp_6704[0], 22 * tmp_6704[1], 22 * tmp_6704[2]];
    signal tmp_6706[3] <== [tmp_6703[0] + tmp_6705[0], tmp_6703[1] + tmp_6705[1], tmp_6703[2] + tmp_6705[2]];
    signal tmp_6707[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6708[3] <== [13 * tmp_6707[0], 13 * tmp_6707[1], 13 * tmp_6707[2]];
    signal tmp_6709[3] <== [tmp_6706[0] + tmp_6708[0], tmp_6706[1] + tmp_6708[1], tmp_6706[2] + tmp_6708[2]];
    signal tmp_6710[3] <== [evals[83][0] - tmp_6709[0], evals[83][1] - tmp_6709[1], evals[83][2] - tmp_6709[2]];
    signal tmp_6711[3] <== CMul()(tmp_6666, tmp_6710);
    signal tmp_6712[3] <== [tmp_6663[0] + tmp_6711[0], tmp_6663[1] + tmp_6711[1], tmp_6663[2] + tmp_6711[2]];
    signal tmp_6713[3] <== CMul()(challengeQ, tmp_6712);
    signal tmp_6714[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6715[3] <== [tmp_6714[0] + evals[109][0], tmp_6714[1] + evals[109][1], tmp_6714[2] + evals[109][2]];
    signal tmp_6716[3] <== [tmp_6715[0] + evals[41][0], tmp_6715[1] + evals[41][1], tmp_6715[2] + evals[41][2]];
    signal tmp_6717[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6718[3] <== [13 * tmp_6717[0], 13 * tmp_6717[1], 13 * tmp_6717[2]];
    signal tmp_6719[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6720[3] <== [3 * tmp_6719[0], 3 * tmp_6719[1], 3 * tmp_6719[2]];
    signal tmp_6721[3] <== [tmp_6718[0] + tmp_6720[0], tmp_6718[1] + tmp_6720[1], tmp_6718[2] + tmp_6720[2]];
    signal tmp_6722[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6723[3] <== [tmp_6721[0] + tmp_6722[0], tmp_6721[1] + tmp_6722[1], tmp_6721[2] + tmp_6722[2]];
    signal tmp_6724[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6725[3] <== [tmp_6723[0] + tmp_6724[0], tmp_6723[1] + tmp_6724[1], tmp_6723[2] + tmp_6724[2]];
    signal tmp_6726[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6727[3] <== [51 * tmp_6726[0], 51 * tmp_6726[1], 51 * tmp_6726[2]];
    signal tmp_6728[3] <== [tmp_6725[0] + tmp_6727[0], tmp_6725[1] + tmp_6727[1], tmp_6725[2] + tmp_6727[2]];
    signal tmp_6729[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6730[3] <== [tmp_6728[0] + tmp_6729[0], tmp_6728[1] + tmp_6729[1], tmp_6728[2] + tmp_6729[2]];
    signal tmp_6731[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6732[3] <== [11 * tmp_6731[0], 11 * tmp_6731[1], 11 * tmp_6731[2]];
    signal tmp_6733[3] <== [tmp_6730[0] + tmp_6732[0], tmp_6730[1] + tmp_6732[1], tmp_6730[2] + tmp_6732[2]];
    signal tmp_6734[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6735[3] <== [17 * tmp_6734[0], 17 * tmp_6734[1], 17 * tmp_6734[2]];
    signal tmp_6736[3] <== [tmp_6733[0] + tmp_6735[0], tmp_6733[1] + tmp_6735[1], tmp_6733[2] + tmp_6735[2]];
    signal tmp_6737[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6738[3] <== [2 * tmp_6737[0], 2 * tmp_6737[1], 2 * tmp_6737[2]];
    signal tmp_6739[3] <== [tmp_6736[0] + tmp_6738[0], tmp_6736[1] + tmp_6738[1], tmp_6736[2] + tmp_6738[2]];
    signal tmp_6740[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6741[3] <== [tmp_6739[0] + tmp_6740[0], tmp_6739[1] + tmp_6740[1], tmp_6739[2] + tmp_6740[2]];
    signal tmp_6742[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6743[3] <== [101 * tmp_6742[0], 101 * tmp_6742[1], 101 * tmp_6742[2]];
    signal tmp_6744[3] <== [tmp_6741[0] + tmp_6743[0], tmp_6741[1] + tmp_6743[1], tmp_6741[2] + tmp_6743[2]];
    signal tmp_6745[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6746[3] <== [63 * tmp_6745[0], 63 * tmp_6745[1], 63 * tmp_6745[2]];
    signal tmp_6747[3] <== [tmp_6744[0] + tmp_6746[0], tmp_6744[1] + tmp_6746[1], tmp_6744[2] + tmp_6746[2]];
    signal tmp_6748[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6749[3] <== [15 * tmp_6748[0], 15 * tmp_6748[1], 15 * tmp_6748[2]];
    signal tmp_6750[3] <== [tmp_6747[0] + tmp_6749[0], tmp_6747[1] + tmp_6749[1], tmp_6747[2] + tmp_6749[2]];
    signal tmp_6751[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6752[3] <== [2 * tmp_6751[0], 2 * tmp_6751[1], 2 * tmp_6751[2]];
    signal tmp_6753[3] <== [tmp_6750[0] + tmp_6752[0], tmp_6750[1] + tmp_6752[1], tmp_6750[2] + tmp_6752[2]];
    signal tmp_6754[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6755[3] <== [67 * tmp_6754[0], 67 * tmp_6754[1], 67 * tmp_6754[2]];
    signal tmp_6756[3] <== [tmp_6753[0] + tmp_6755[0], tmp_6753[1] + tmp_6755[1], tmp_6753[2] + tmp_6755[2]];
    signal tmp_6757[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6758[3] <== [22 * tmp_6757[0], 22 * tmp_6757[1], 22 * tmp_6757[2]];
    signal tmp_6759[3] <== [tmp_6756[0] + tmp_6758[0], tmp_6756[1] + tmp_6758[1], tmp_6756[2] + tmp_6758[2]];
    signal tmp_6760[3] <== [evals[84][0] - tmp_6759[0], evals[84][1] - tmp_6759[1], evals[84][2] - tmp_6759[2]];
    signal tmp_6761[3] <== CMul()(tmp_6716, tmp_6760);
    signal tmp_6762[3] <== [tmp_6713[0] + tmp_6761[0], tmp_6713[1] + tmp_6761[1], tmp_6713[2] + tmp_6761[2]];
    signal tmp_6763[3] <== CMul()(challengeQ, tmp_6762);
    signal tmp_6764[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6765[3] <== [tmp_6764[0] + evals[109][0], tmp_6764[1] + evals[109][1], tmp_6764[2] + evals[109][2]];
    signal tmp_6766[3] <== [tmp_6765[0] + evals[41][0], tmp_6765[1] + evals[41][1], tmp_6765[2] + evals[41][2]];
    signal tmp_6767[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6768[3] <== [22 * tmp_6767[0], 22 * tmp_6767[1], 22 * tmp_6767[2]];
    signal tmp_6769[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6770[3] <== [13 * tmp_6769[0], 13 * tmp_6769[1], 13 * tmp_6769[2]];
    signal tmp_6771[3] <== [tmp_6768[0] + tmp_6770[0], tmp_6768[1] + tmp_6770[1], tmp_6768[2] + tmp_6770[2]];
    signal tmp_6772[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6773[3] <== [3 * tmp_6772[0], 3 * tmp_6772[1], 3 * tmp_6772[2]];
    signal tmp_6774[3] <== [tmp_6771[0] + tmp_6773[0], tmp_6771[1] + tmp_6773[1], tmp_6771[2] + tmp_6773[2]];
    signal tmp_6775[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6776[3] <== [tmp_6774[0] + tmp_6775[0], tmp_6774[1] + tmp_6775[1], tmp_6774[2] + tmp_6775[2]];
    signal tmp_6777[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6778[3] <== [tmp_6776[0] + tmp_6777[0], tmp_6776[1] + tmp_6777[1], tmp_6776[2] + tmp_6777[2]];
    signal tmp_6779[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6780[3] <== [51 * tmp_6779[0], 51 * tmp_6779[1], 51 * tmp_6779[2]];
    signal tmp_6781[3] <== [tmp_6778[0] + tmp_6780[0], tmp_6778[1] + tmp_6780[1], tmp_6778[2] + tmp_6780[2]];
    signal tmp_6782[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6783[3] <== [tmp_6781[0] + tmp_6782[0], tmp_6781[1] + tmp_6782[1], tmp_6781[2] + tmp_6782[2]];
    signal tmp_6784[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6785[3] <== [11 * tmp_6784[0], 11 * tmp_6784[1], 11 * tmp_6784[2]];
    signal tmp_6786[3] <== [tmp_6783[0] + tmp_6785[0], tmp_6783[1] + tmp_6785[1], tmp_6783[2] + tmp_6785[2]];
    signal tmp_6787[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6788[3] <== [17 * tmp_6787[0], 17 * tmp_6787[1], 17 * tmp_6787[2]];
    signal tmp_6789[3] <== [tmp_6786[0] + tmp_6788[0], tmp_6786[1] + tmp_6788[1], tmp_6786[2] + tmp_6788[2]];
    signal tmp_6790[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6791[3] <== [2 * tmp_6790[0], 2 * tmp_6790[1], 2 * tmp_6790[2]];
    signal tmp_6792[3] <== [tmp_6789[0] + tmp_6791[0], tmp_6789[1] + tmp_6791[1], tmp_6789[2] + tmp_6791[2]];
    signal tmp_6793[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6794[3] <== [tmp_6792[0] + tmp_6793[0], tmp_6792[1] + tmp_6793[1], tmp_6792[2] + tmp_6793[2]];
    signal tmp_6795[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6796[3] <== [101 * tmp_6795[0], 101 * tmp_6795[1], 101 * tmp_6795[2]];
    signal tmp_6797[3] <== [tmp_6794[0] + tmp_6796[0], tmp_6794[1] + tmp_6796[1], tmp_6794[2] + tmp_6796[2]];
    signal tmp_6798[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6799[3] <== [63 * tmp_6798[0], 63 * tmp_6798[1], 63 * tmp_6798[2]];
    signal tmp_6800[3] <== [tmp_6797[0] + tmp_6799[0], tmp_6797[1] + tmp_6799[1], tmp_6797[2] + tmp_6799[2]];
    signal tmp_6801[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6802[3] <== [15 * tmp_6801[0], 15 * tmp_6801[1], 15 * tmp_6801[2]];
    signal tmp_6803[3] <== [tmp_6800[0] + tmp_6802[0], tmp_6800[1] + tmp_6802[1], tmp_6800[2] + tmp_6802[2]];
    signal tmp_6804[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6805[3] <== [2 * tmp_6804[0], 2 * tmp_6804[1], 2 * tmp_6804[2]];
    signal tmp_6806[3] <== [tmp_6803[0] + tmp_6805[0], tmp_6803[1] + tmp_6805[1], tmp_6803[2] + tmp_6805[2]];
    signal tmp_6807[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6808[3] <== [67 * tmp_6807[0], 67 * tmp_6807[1], 67 * tmp_6807[2]];
    signal tmp_6809[3] <== [tmp_6806[0] + tmp_6808[0], tmp_6806[1] + tmp_6808[1], tmp_6806[2] + tmp_6808[2]];
    signal tmp_6810[3] <== [evals[85][0] - tmp_6809[0], evals[85][1] - tmp_6809[1], evals[85][2] - tmp_6809[2]];
    signal tmp_6811[3] <== CMul()(tmp_6766, tmp_6810);
    signal tmp_6812[3] <== [tmp_6763[0] + tmp_6811[0], tmp_6763[1] + tmp_6811[1], tmp_6763[2] + tmp_6811[2]];
    signal tmp_6813[3] <== CMul()(challengeQ, tmp_6812);
    signal tmp_6814[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6815[3] <== [tmp_6814[0] + evals[109][0], tmp_6814[1] + evals[109][1], tmp_6814[2] + evals[109][2]];
    signal tmp_6816[3] <== [tmp_6815[0] + evals[41][0], tmp_6815[1] + evals[41][1], tmp_6815[2] + evals[41][2]];
    signal tmp_6817[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6818[3] <== [67 * tmp_6817[0], 67 * tmp_6817[1], 67 * tmp_6817[2]];
    signal tmp_6819[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6820[3] <== [22 * tmp_6819[0], 22 * tmp_6819[1], 22 * tmp_6819[2]];
    signal tmp_6821[3] <== [tmp_6818[0] + tmp_6820[0], tmp_6818[1] + tmp_6820[1], tmp_6818[2] + tmp_6820[2]];
    signal tmp_6822[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6823[3] <== [13 * tmp_6822[0], 13 * tmp_6822[1], 13 * tmp_6822[2]];
    signal tmp_6824[3] <== [tmp_6821[0] + tmp_6823[0], tmp_6821[1] + tmp_6823[1], tmp_6821[2] + tmp_6823[2]];
    signal tmp_6825[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6826[3] <== [3 * tmp_6825[0], 3 * tmp_6825[1], 3 * tmp_6825[2]];
    signal tmp_6827[3] <== [tmp_6824[0] + tmp_6826[0], tmp_6824[1] + tmp_6826[1], tmp_6824[2] + tmp_6826[2]];
    signal tmp_6828[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6829[3] <== [tmp_6827[0] + tmp_6828[0], tmp_6827[1] + tmp_6828[1], tmp_6827[2] + tmp_6828[2]];
    signal tmp_6830[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6831[3] <== [tmp_6829[0] + tmp_6830[0], tmp_6829[1] + tmp_6830[1], tmp_6829[2] + tmp_6830[2]];
    signal tmp_6832[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6833[3] <== [51 * tmp_6832[0], 51 * tmp_6832[1], 51 * tmp_6832[2]];
    signal tmp_6834[3] <== [tmp_6831[0] + tmp_6833[0], tmp_6831[1] + tmp_6833[1], tmp_6831[2] + tmp_6833[2]];
    signal tmp_6835[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6836[3] <== [tmp_6834[0] + tmp_6835[0], tmp_6834[1] + tmp_6835[1], tmp_6834[2] + tmp_6835[2]];
    signal tmp_6837[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6838[3] <== [11 * tmp_6837[0], 11 * tmp_6837[1], 11 * tmp_6837[2]];
    signal tmp_6839[3] <== [tmp_6836[0] + tmp_6838[0], tmp_6836[1] + tmp_6838[1], tmp_6836[2] + tmp_6838[2]];
    signal tmp_6840[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6841[3] <== [17 * tmp_6840[0], 17 * tmp_6840[1], 17 * tmp_6840[2]];
    signal tmp_6842[3] <== [tmp_6839[0] + tmp_6841[0], tmp_6839[1] + tmp_6841[1], tmp_6839[2] + tmp_6841[2]];
    signal tmp_6843[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6844[3] <== [2 * tmp_6843[0], 2 * tmp_6843[1], 2 * tmp_6843[2]];
    signal tmp_6845[3] <== [tmp_6842[0] + tmp_6844[0], tmp_6842[1] + tmp_6844[1], tmp_6842[2] + tmp_6844[2]];
    signal tmp_6846[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6847[3] <== [tmp_6845[0] + tmp_6846[0], tmp_6845[1] + tmp_6846[1], tmp_6845[2] + tmp_6846[2]];
    signal tmp_6848[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6849[3] <== [101 * tmp_6848[0], 101 * tmp_6848[1], 101 * tmp_6848[2]];
    signal tmp_6850[3] <== [tmp_6847[0] + tmp_6849[0], tmp_6847[1] + tmp_6849[1], tmp_6847[2] + tmp_6849[2]];
    signal tmp_6851[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6852[3] <== [63 * tmp_6851[0], 63 * tmp_6851[1], 63 * tmp_6851[2]];
    signal tmp_6853[3] <== [tmp_6850[0] + tmp_6852[0], tmp_6850[1] + tmp_6852[1], tmp_6850[2] + tmp_6852[2]];
    signal tmp_6854[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6855[3] <== [15 * tmp_6854[0], 15 * tmp_6854[1], 15 * tmp_6854[2]];
    signal tmp_6856[3] <== [tmp_6853[0] + tmp_6855[0], tmp_6853[1] + tmp_6855[1], tmp_6853[2] + tmp_6855[2]];
    signal tmp_6857[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6858[3] <== [2 * tmp_6857[0], 2 * tmp_6857[1], 2 * tmp_6857[2]];
    signal tmp_6859[3] <== [tmp_6856[0] + tmp_6858[0], tmp_6856[1] + tmp_6858[1], tmp_6856[2] + tmp_6858[2]];
    signal tmp_6860[3] <== [evals[86][0] - tmp_6859[0], evals[86][1] - tmp_6859[1], evals[86][2] - tmp_6859[2]];
    signal tmp_6861[3] <== CMul()(tmp_6816, tmp_6860);
    signal tmp_6862[3] <== [tmp_6813[0] + tmp_6861[0], tmp_6813[1] + tmp_6861[1], tmp_6813[2] + tmp_6861[2]];
    signal tmp_6863[3] <== CMul()(challengeQ, tmp_6862);
    signal tmp_6864[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6865[3] <== [tmp_6864[0] + evals[109][0], tmp_6864[1] + evals[109][1], tmp_6864[2] + evals[109][2]];
    signal tmp_6866[3] <== [tmp_6865[0] + evals[41][0], tmp_6865[1] + evals[41][1], tmp_6865[2] + evals[41][2]];
    signal tmp_6867[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6868[3] <== [2 * tmp_6867[0], 2 * tmp_6867[1], 2 * tmp_6867[2]];
    signal tmp_6869[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6870[3] <== [67 * tmp_6869[0], 67 * tmp_6869[1], 67 * tmp_6869[2]];
    signal tmp_6871[3] <== [tmp_6868[0] + tmp_6870[0], tmp_6868[1] + tmp_6870[1], tmp_6868[2] + tmp_6870[2]];
    signal tmp_6872[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6873[3] <== [22 * tmp_6872[0], 22 * tmp_6872[1], 22 * tmp_6872[2]];
    signal tmp_6874[3] <== [tmp_6871[0] + tmp_6873[0], tmp_6871[1] + tmp_6873[1], tmp_6871[2] + tmp_6873[2]];
    signal tmp_6875[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6876[3] <== [13 * tmp_6875[0], 13 * tmp_6875[1], 13 * tmp_6875[2]];
    signal tmp_6877[3] <== [tmp_6874[0] + tmp_6876[0], tmp_6874[1] + tmp_6876[1], tmp_6874[2] + tmp_6876[2]];
    signal tmp_6878[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6879[3] <== [3 * tmp_6878[0], 3 * tmp_6878[1], 3 * tmp_6878[2]];
    signal tmp_6880[3] <== [tmp_6877[0] + tmp_6879[0], tmp_6877[1] + tmp_6879[1], tmp_6877[2] + tmp_6879[2]];
    signal tmp_6881[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6882[3] <== [tmp_6880[0] + tmp_6881[0], tmp_6880[1] + tmp_6881[1], tmp_6880[2] + tmp_6881[2]];
    signal tmp_6883[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6884[3] <== [tmp_6882[0] + tmp_6883[0], tmp_6882[1] + tmp_6883[1], tmp_6882[2] + tmp_6883[2]];
    signal tmp_6885[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6886[3] <== [51 * tmp_6885[0], 51 * tmp_6885[1], 51 * tmp_6885[2]];
    signal tmp_6887[3] <== [tmp_6884[0] + tmp_6886[0], tmp_6884[1] + tmp_6886[1], tmp_6884[2] + tmp_6886[2]];
    signal tmp_6888[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6889[3] <== [tmp_6887[0] + tmp_6888[0], tmp_6887[1] + tmp_6888[1], tmp_6887[2] + tmp_6888[2]];
    signal tmp_6890[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6891[3] <== [11 * tmp_6890[0], 11 * tmp_6890[1], 11 * tmp_6890[2]];
    signal tmp_6892[3] <== [tmp_6889[0] + tmp_6891[0], tmp_6889[1] + tmp_6891[1], tmp_6889[2] + tmp_6891[2]];
    signal tmp_6893[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6894[3] <== [17 * tmp_6893[0], 17 * tmp_6893[1], 17 * tmp_6893[2]];
    signal tmp_6895[3] <== [tmp_6892[0] + tmp_6894[0], tmp_6892[1] + tmp_6894[1], tmp_6892[2] + tmp_6894[2]];
    signal tmp_6896[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6897[3] <== [2 * tmp_6896[0], 2 * tmp_6896[1], 2 * tmp_6896[2]];
    signal tmp_6898[3] <== [tmp_6895[0] + tmp_6897[0], tmp_6895[1] + tmp_6897[1], tmp_6895[2] + tmp_6897[2]];
    signal tmp_6899[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6900[3] <== [tmp_6898[0] + tmp_6899[0], tmp_6898[1] + tmp_6899[1], tmp_6898[2] + tmp_6899[2]];
    signal tmp_6901[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6902[3] <== [101 * tmp_6901[0], 101 * tmp_6901[1], 101 * tmp_6901[2]];
    signal tmp_6903[3] <== [tmp_6900[0] + tmp_6902[0], tmp_6900[1] + tmp_6902[1], tmp_6900[2] + tmp_6902[2]];
    signal tmp_6904[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6905[3] <== [63 * tmp_6904[0], 63 * tmp_6904[1], 63 * tmp_6904[2]];
    signal tmp_6906[3] <== [tmp_6903[0] + tmp_6905[0], tmp_6903[1] + tmp_6905[1], tmp_6903[2] + tmp_6905[2]];
    signal tmp_6907[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6908[3] <== [15 * tmp_6907[0], 15 * tmp_6907[1], 15 * tmp_6907[2]];
    signal tmp_6909[3] <== [tmp_6906[0] + tmp_6908[0], tmp_6906[1] + tmp_6908[1], tmp_6906[2] + tmp_6908[2]];
    signal tmp_6910[3] <== [evals[87][0] - tmp_6909[0], evals[87][1] - tmp_6909[1], evals[87][2] - tmp_6909[2]];
    signal tmp_6911[3] <== CMul()(tmp_6866, tmp_6910);
    signal tmp_6912[3] <== [tmp_6863[0] + tmp_6911[0], tmp_6863[1] + tmp_6911[1], tmp_6863[2] + tmp_6911[2]];
    signal tmp_6913[3] <== CMul()(challengeQ, tmp_6912);
    signal tmp_6914[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6915[3] <== [tmp_6914[0] + evals[109][0], tmp_6914[1] + evals[109][1], tmp_6914[2] + evals[109][2]];
    signal tmp_6916[3] <== [tmp_6915[0] + evals[41][0], tmp_6915[1] + evals[41][1], tmp_6915[2] + evals[41][2]];
    signal tmp_6917[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6918[3] <== [15 * tmp_6917[0], 15 * tmp_6917[1], 15 * tmp_6917[2]];
    signal tmp_6919[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6920[3] <== [2 * tmp_6919[0], 2 * tmp_6919[1], 2 * tmp_6919[2]];
    signal tmp_6921[3] <== [tmp_6918[0] + tmp_6920[0], tmp_6918[1] + tmp_6920[1], tmp_6918[2] + tmp_6920[2]];
    signal tmp_6922[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6923[3] <== [67 * tmp_6922[0], 67 * tmp_6922[1], 67 * tmp_6922[2]];
    signal tmp_6924[3] <== [tmp_6921[0] + tmp_6923[0], tmp_6921[1] + tmp_6923[1], tmp_6921[2] + tmp_6923[2]];
    signal tmp_6925[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6926[3] <== [22 * tmp_6925[0], 22 * tmp_6925[1], 22 * tmp_6925[2]];
    signal tmp_6927[3] <== [tmp_6924[0] + tmp_6926[0], tmp_6924[1] + tmp_6926[1], tmp_6924[2] + tmp_6926[2]];
    signal tmp_6928[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6929[3] <== [13 * tmp_6928[0], 13 * tmp_6928[1], 13 * tmp_6928[2]];
    signal tmp_6930[3] <== [tmp_6927[0] + tmp_6929[0], tmp_6927[1] + tmp_6929[1], tmp_6927[2] + tmp_6929[2]];
    signal tmp_6931[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6932[3] <== [3 * tmp_6931[0], 3 * tmp_6931[1], 3 * tmp_6931[2]];
    signal tmp_6933[3] <== [tmp_6930[0] + tmp_6932[0], tmp_6930[1] + tmp_6932[1], tmp_6930[2] + tmp_6932[2]];
    signal tmp_6934[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6935[3] <== [tmp_6933[0] + tmp_6934[0], tmp_6933[1] + tmp_6934[1], tmp_6933[2] + tmp_6934[2]];
    signal tmp_6936[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6937[3] <== [tmp_6935[0] + tmp_6936[0], tmp_6935[1] + tmp_6936[1], tmp_6935[2] + tmp_6936[2]];
    signal tmp_6938[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6939[3] <== [51 * tmp_6938[0], 51 * tmp_6938[1], 51 * tmp_6938[2]];
    signal tmp_6940[3] <== [tmp_6937[0] + tmp_6939[0], tmp_6937[1] + tmp_6939[1], tmp_6937[2] + tmp_6939[2]];
    signal tmp_6941[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6942[3] <== [tmp_6940[0] + tmp_6941[0], tmp_6940[1] + tmp_6941[1], tmp_6940[2] + tmp_6941[2]];
    signal tmp_6943[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6944[3] <== [11 * tmp_6943[0], 11 * tmp_6943[1], 11 * tmp_6943[2]];
    signal tmp_6945[3] <== [tmp_6942[0] + tmp_6944[0], tmp_6942[1] + tmp_6944[1], tmp_6942[2] + tmp_6944[2]];
    signal tmp_6946[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6947[3] <== [17 * tmp_6946[0], 17 * tmp_6946[1], 17 * tmp_6946[2]];
    signal tmp_6948[3] <== [tmp_6945[0] + tmp_6947[0], tmp_6945[1] + tmp_6947[1], tmp_6945[2] + tmp_6947[2]];
    signal tmp_6949[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_6950[3] <== [2 * tmp_6949[0], 2 * tmp_6949[1], 2 * tmp_6949[2]];
    signal tmp_6951[3] <== [tmp_6948[0] + tmp_6950[0], tmp_6948[1] + tmp_6950[1], tmp_6948[2] + tmp_6950[2]];
    signal tmp_6952[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_6953[3] <== [tmp_6951[0] + tmp_6952[0], tmp_6951[1] + tmp_6952[1], tmp_6951[2] + tmp_6952[2]];
    signal tmp_6954[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_6955[3] <== [101 * tmp_6954[0], 101 * tmp_6954[1], 101 * tmp_6954[2]];
    signal tmp_6956[3] <== [tmp_6953[0] + tmp_6955[0], tmp_6953[1] + tmp_6955[1], tmp_6953[2] + tmp_6955[2]];
    signal tmp_6957[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_6958[3] <== [63 * tmp_6957[0], 63 * tmp_6957[1], 63 * tmp_6957[2]];
    signal tmp_6959[3] <== [tmp_6956[0] + tmp_6958[0], tmp_6956[1] + tmp_6958[1], tmp_6956[2] + tmp_6958[2]];
    signal tmp_6960[3] <== [evals[88][0] - tmp_6959[0], evals[88][1] - tmp_6959[1], evals[88][2] - tmp_6959[2]];
    signal tmp_6961[3] <== CMul()(tmp_6916, tmp_6960);
    signal tmp_6962[3] <== [tmp_6913[0] + tmp_6961[0], tmp_6913[1] + tmp_6961[1], tmp_6913[2] + tmp_6961[2]];
    signal tmp_6963[3] <== CMul()(challengeQ, tmp_6962);
    signal tmp_6964[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_6965[3] <== [tmp_6964[0] + evals[109][0], tmp_6964[1] + evals[109][1], tmp_6964[2] + evals[109][2]];
    signal tmp_6966[3] <== [tmp_6965[0] + evals[41][0], tmp_6965[1] + evals[41][1], tmp_6965[2] + evals[41][2]];
    signal tmp_6967[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_6968[3] <== [63 * tmp_6967[0], 63 * tmp_6967[1], 63 * tmp_6967[2]];
    signal tmp_6969[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_6970[3] <== [15 * tmp_6969[0], 15 * tmp_6969[1], 15 * tmp_6969[2]];
    signal tmp_6971[3] <== [tmp_6968[0] + tmp_6970[0], tmp_6968[1] + tmp_6970[1], tmp_6968[2] + tmp_6970[2]];
    signal tmp_6972[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_6973[3] <== [2 * tmp_6972[0], 2 * tmp_6972[1], 2 * tmp_6972[2]];
    signal tmp_6974[3] <== [tmp_6971[0] + tmp_6973[0], tmp_6971[1] + tmp_6973[1], tmp_6971[2] + tmp_6973[2]];
    signal tmp_6975[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_6976[3] <== [67 * tmp_6975[0], 67 * tmp_6975[1], 67 * tmp_6975[2]];
    signal tmp_6977[3] <== [tmp_6974[0] + tmp_6976[0], tmp_6974[1] + tmp_6976[1], tmp_6974[2] + tmp_6976[2]];
    signal tmp_6978[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_6979[3] <== [22 * tmp_6978[0], 22 * tmp_6978[1], 22 * tmp_6978[2]];
    signal tmp_6980[3] <== [tmp_6977[0] + tmp_6979[0], tmp_6977[1] + tmp_6979[1], tmp_6977[2] + tmp_6979[2]];
    signal tmp_6981[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_6982[3] <== [13 * tmp_6981[0], 13 * tmp_6981[1], 13 * tmp_6981[2]];
    signal tmp_6983[3] <== [tmp_6980[0] + tmp_6982[0], tmp_6980[1] + tmp_6982[1], tmp_6980[2] + tmp_6982[2]];
    signal tmp_6984[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_6985[3] <== [3 * tmp_6984[0], 3 * tmp_6984[1], 3 * tmp_6984[2]];
    signal tmp_6986[3] <== [tmp_6983[0] + tmp_6985[0], tmp_6983[1] + tmp_6985[1], tmp_6983[2] + tmp_6985[2]];
    signal tmp_6987[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_6988[3] <== [tmp_6986[0] + tmp_6987[0], tmp_6986[1] + tmp_6987[1], tmp_6986[2] + tmp_6987[2]];
    signal tmp_6989[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_6990[3] <== [tmp_6988[0] + tmp_6989[0], tmp_6988[1] + tmp_6989[1], tmp_6988[2] + tmp_6989[2]];
    signal tmp_6991[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_6992[3] <== [51 * tmp_6991[0], 51 * tmp_6991[1], 51 * tmp_6991[2]];
    signal tmp_6993[3] <== [tmp_6990[0] + tmp_6992[0], tmp_6990[1] + tmp_6992[1], tmp_6990[2] + tmp_6992[2]];
    signal tmp_6994[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_6995[3] <== [tmp_6993[0] + tmp_6994[0], tmp_6993[1] + tmp_6994[1], tmp_6993[2] + tmp_6994[2]];
    signal tmp_6996[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_6997[3] <== [11 * tmp_6996[0], 11 * tmp_6996[1], 11 * tmp_6996[2]];
    signal tmp_6998[3] <== [tmp_6995[0] + tmp_6997[0], tmp_6995[1] + tmp_6997[1], tmp_6995[2] + tmp_6997[2]];
    signal tmp_6999[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7000[3] <== [17 * tmp_6999[0], 17 * tmp_6999[1], 17 * tmp_6999[2]];
    signal tmp_7001[3] <== [tmp_6998[0] + tmp_7000[0], tmp_6998[1] + tmp_7000[1], tmp_6998[2] + tmp_7000[2]];
    signal tmp_7002[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7003[3] <== [2 * tmp_7002[0], 2 * tmp_7002[1], 2 * tmp_7002[2]];
    signal tmp_7004[3] <== [tmp_7001[0] + tmp_7003[0], tmp_7001[1] + tmp_7003[1], tmp_7001[2] + tmp_7003[2]];
    signal tmp_7005[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7006[3] <== [tmp_7004[0] + tmp_7005[0], tmp_7004[1] + tmp_7005[1], tmp_7004[2] + tmp_7005[2]];
    signal tmp_7007[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7008[3] <== [101 * tmp_7007[0], 101 * tmp_7007[1], 101 * tmp_7007[2]];
    signal tmp_7009[3] <== [tmp_7006[0] + tmp_7008[0], tmp_7006[1] + tmp_7008[1], tmp_7006[2] + tmp_7008[2]];
    signal tmp_7010[3] <== [evals[89][0] - tmp_7009[0], evals[89][1] - tmp_7009[1], evals[89][2] - tmp_7009[2]];
    signal tmp_7011[3] <== CMul()(tmp_6966, tmp_7010);
    signal tmp_7012[3] <== [tmp_6963[0] + tmp_7011[0], tmp_6963[1] + tmp_7011[1], tmp_6963[2] + tmp_7011[2]];
    signal tmp_7013[3] <== CMul()(challengeQ, tmp_7012);
    signal tmp_7014[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7015[3] <== [tmp_7014[0] + evals[109][0], tmp_7014[1] + evals[109][1], tmp_7014[2] + evals[109][2]];
    signal tmp_7016[3] <== [tmp_7015[0] + evals[41][0], tmp_7015[1] + evals[41][1], tmp_7015[2] + evals[41][2]];
    signal tmp_7017[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7018[3] <== [101 * tmp_7017[0], 101 * tmp_7017[1], 101 * tmp_7017[2]];
    signal tmp_7019[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7020[3] <== [63 * tmp_7019[0], 63 * tmp_7019[1], 63 * tmp_7019[2]];
    signal tmp_7021[3] <== [tmp_7018[0] + tmp_7020[0], tmp_7018[1] + tmp_7020[1], tmp_7018[2] + tmp_7020[2]];
    signal tmp_7022[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7023[3] <== [15 * tmp_7022[0], 15 * tmp_7022[1], 15 * tmp_7022[2]];
    signal tmp_7024[3] <== [tmp_7021[0] + tmp_7023[0], tmp_7021[1] + tmp_7023[1], tmp_7021[2] + tmp_7023[2]];
    signal tmp_7025[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7026[3] <== [2 * tmp_7025[0], 2 * tmp_7025[1], 2 * tmp_7025[2]];
    signal tmp_7027[3] <== [tmp_7024[0] + tmp_7026[0], tmp_7024[1] + tmp_7026[1], tmp_7024[2] + tmp_7026[2]];
    signal tmp_7028[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7029[3] <== [67 * tmp_7028[0], 67 * tmp_7028[1], 67 * tmp_7028[2]];
    signal tmp_7030[3] <== [tmp_7027[0] + tmp_7029[0], tmp_7027[1] + tmp_7029[1], tmp_7027[2] + tmp_7029[2]];
    signal tmp_7031[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7032[3] <== [22 * tmp_7031[0], 22 * tmp_7031[1], 22 * tmp_7031[2]];
    signal tmp_7033[3] <== [tmp_7030[0] + tmp_7032[0], tmp_7030[1] + tmp_7032[1], tmp_7030[2] + tmp_7032[2]];
    signal tmp_7034[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7035[3] <== [13 * tmp_7034[0], 13 * tmp_7034[1], 13 * tmp_7034[2]];
    signal tmp_7036[3] <== [tmp_7033[0] + tmp_7035[0], tmp_7033[1] + tmp_7035[1], tmp_7033[2] + tmp_7035[2]];
    signal tmp_7037[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7038[3] <== [3 * tmp_7037[0], 3 * tmp_7037[1], 3 * tmp_7037[2]];
    signal tmp_7039[3] <== [tmp_7036[0] + tmp_7038[0], tmp_7036[1] + tmp_7038[1], tmp_7036[2] + tmp_7038[2]];
    signal tmp_7040[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7041[3] <== [tmp_7039[0] + tmp_7040[0], tmp_7039[1] + tmp_7040[1], tmp_7039[2] + tmp_7040[2]];
    signal tmp_7042[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7043[3] <== [tmp_7041[0] + tmp_7042[0], tmp_7041[1] + tmp_7042[1], tmp_7041[2] + tmp_7042[2]];
    signal tmp_7044[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7045[3] <== [51 * tmp_7044[0], 51 * tmp_7044[1], 51 * tmp_7044[2]];
    signal tmp_7046[3] <== [tmp_7043[0] + tmp_7045[0], tmp_7043[1] + tmp_7045[1], tmp_7043[2] + tmp_7045[2]];
    signal tmp_7047[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7048[3] <== [tmp_7046[0] + tmp_7047[0], tmp_7046[1] + tmp_7047[1], tmp_7046[2] + tmp_7047[2]];
    signal tmp_7049[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7050[3] <== [11 * tmp_7049[0], 11 * tmp_7049[1], 11 * tmp_7049[2]];
    signal tmp_7051[3] <== [tmp_7048[0] + tmp_7050[0], tmp_7048[1] + tmp_7050[1], tmp_7048[2] + tmp_7050[2]];
    signal tmp_7052[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7053[3] <== [17 * tmp_7052[0], 17 * tmp_7052[1], 17 * tmp_7052[2]];
    signal tmp_7054[3] <== [tmp_7051[0] + tmp_7053[0], tmp_7051[1] + tmp_7053[1], tmp_7051[2] + tmp_7053[2]];
    signal tmp_7055[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7056[3] <== [2 * tmp_7055[0], 2 * tmp_7055[1], 2 * tmp_7055[2]];
    signal tmp_7057[3] <== [tmp_7054[0] + tmp_7056[0], tmp_7054[1] + tmp_7056[1], tmp_7054[2] + tmp_7056[2]];
    signal tmp_7058[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7059[3] <== [tmp_7057[0] + tmp_7058[0], tmp_7057[1] + tmp_7058[1], tmp_7057[2] + tmp_7058[2]];
    signal tmp_7060[3] <== [evals[90][0] - tmp_7059[0], evals[90][1] - tmp_7059[1], evals[90][2] - tmp_7059[2]];
    signal tmp_7061[3] <== CMul()(tmp_7016, tmp_7060);
    signal tmp_7062[3] <== [tmp_7013[0] + tmp_7061[0], tmp_7013[1] + tmp_7061[1], tmp_7013[2] + tmp_7061[2]];
    tmp_7063 <== CMul()(challengeQ, tmp_7062);
    tmp_7064 <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
}

template VerifyEvaluationsChunks1() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_6066[3];
    signal input tmp_6405[3];
    signal input tmp_6414[3];
    signal input tmp_6421[3];
    signal input tmp_6428[3];
    signal input tmp_6435[3];
    signal input tmp_6443[3];
    signal input tmp_6450[3];
    signal input tmp_6459[3];
    signal input tmp_6466[3];
    signal input tmp_6474[3];
    signal input tmp_6481[3];
    signal input tmp_6490[3];
    signal input tmp_6497[3];
    signal input tmp_6506[3];
    signal input tmp_6513[3];
    signal input tmp_6522[3];
    signal input tmp_6529[3];
    signal input tmp_6537[3];
    signal input tmp_6544[3];
    signal input tmp_6553[3];
    signal input tmp_6560[3];
    signal input tmp_6569[3];
    signal input tmp_6576[3];
    signal input tmp_6585[3];
    signal input tmp_6592[3];
    signal input tmp_6601[3];
    signal input tmp_6608[3];
    signal input tmp_6617[3];
    signal input tmp_6624[3];
    signal input tmp_6633[3];
    signal input tmp_6640[3];
    signal input tmp_6649[3];
    signal input tmp_6656[3];
    signal input tmp_7063[3];
    signal input tmp_7064[3];

    signal output tmp_7423[3];
    signal output tmp_7428[3];
    signal output tmp_7433[3];
    signal output tmp_7438[3];
    signal output tmp_7444[3];
    signal output tmp_7449[3];
    signal output tmp_7456[3];
    signal output tmp_7461[3];
    signal output tmp_7467[3];
    signal output tmp_7472[3];
    signal output tmp_7479[3];
    signal output tmp_7484[3];
    signal output tmp_7491[3];
    signal output tmp_7496[3];
    signal output tmp_7503[3];
    signal output tmp_7508[3];
    signal output tmp_7514[3];
    signal output tmp_7519[3];
    signal output tmp_7526[3];
    signal output tmp_7531[3];
    signal output tmp_7538[3];
    signal output tmp_7543[3];
    signal output tmp_7550[3];
    signal output tmp_7555[3];
    signal output tmp_7562[3];
    signal output tmp_7567[3];
    signal output tmp_7574[3];
    signal output tmp_7579[3];
    signal output tmp_7586[3];
    signal output tmp_7591[3];
    signal output tmp_7598[3];
    signal output tmp_7603[3];
    signal output tmp_8034[3];
    signal output tmp_8036[3];
    signal output tmp_8040[3];
    signal output tmp_8062[3];
    signal output tmp_8063[3];
    signal tmp_7065[3] <== [tmp_7064[0] + evals[109][0], tmp_7064[1] + evals[109][1], tmp_7064[2] + evals[109][2]];
    signal tmp_7066[3] <== [tmp_7065[0] + evals[41][0], tmp_7065[1] + evals[41][1], tmp_7065[2] + evals[41][2]];
    signal tmp_7067[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7068[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7069[3] <== [101 * tmp_7068[0], 101 * tmp_7068[1], 101 * tmp_7068[2]];
    signal tmp_7070[3] <== [tmp_7067[0] + tmp_7069[0], tmp_7067[1] + tmp_7069[1], tmp_7067[2] + tmp_7069[2]];
    signal tmp_7071[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7072[3] <== [63 * tmp_7071[0], 63 * tmp_7071[1], 63 * tmp_7071[2]];
    signal tmp_7073[3] <== [tmp_7070[0] + tmp_7072[0], tmp_7070[1] + tmp_7072[1], tmp_7070[2] + tmp_7072[2]];
    signal tmp_7074[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7075[3] <== [15 * tmp_7074[0], 15 * tmp_7074[1], 15 * tmp_7074[2]];
    signal tmp_7076[3] <== [tmp_7073[0] + tmp_7075[0], tmp_7073[1] + tmp_7075[1], tmp_7073[2] + tmp_7075[2]];
    signal tmp_7077[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7078[3] <== [2 * tmp_7077[0], 2 * tmp_7077[1], 2 * tmp_7077[2]];
    signal tmp_7079[3] <== [tmp_7076[0] + tmp_7078[0], tmp_7076[1] + tmp_7078[1], tmp_7076[2] + tmp_7078[2]];
    signal tmp_7080[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7081[3] <== [67 * tmp_7080[0], 67 * tmp_7080[1], 67 * tmp_7080[2]];
    signal tmp_7082[3] <== [tmp_7079[0] + tmp_7081[0], tmp_7079[1] + tmp_7081[1], tmp_7079[2] + tmp_7081[2]];
    signal tmp_7083[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7084[3] <== [22 * tmp_7083[0], 22 * tmp_7083[1], 22 * tmp_7083[2]];
    signal tmp_7085[3] <== [tmp_7082[0] + tmp_7084[0], tmp_7082[1] + tmp_7084[1], tmp_7082[2] + tmp_7084[2]];
    signal tmp_7086[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7087[3] <== [13 * tmp_7086[0], 13 * tmp_7086[1], 13 * tmp_7086[2]];
    signal tmp_7088[3] <== [tmp_7085[0] + tmp_7087[0], tmp_7085[1] + tmp_7087[1], tmp_7085[2] + tmp_7087[2]];
    signal tmp_7089[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7090[3] <== [3 * tmp_7089[0], 3 * tmp_7089[1], 3 * tmp_7089[2]];
    signal tmp_7091[3] <== [tmp_7088[0] + tmp_7090[0], tmp_7088[1] + tmp_7090[1], tmp_7088[2] + tmp_7090[2]];
    signal tmp_7092[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7093[3] <== [tmp_7091[0] + tmp_7092[0], tmp_7091[1] + tmp_7092[1], tmp_7091[2] + tmp_7092[2]];
    signal tmp_7094[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7095[3] <== [tmp_7093[0] + tmp_7094[0], tmp_7093[1] + tmp_7094[1], tmp_7093[2] + tmp_7094[2]];
    signal tmp_7096[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7097[3] <== [51 * tmp_7096[0], 51 * tmp_7096[1], 51 * tmp_7096[2]];
    signal tmp_7098[3] <== [tmp_7095[0] + tmp_7097[0], tmp_7095[1] + tmp_7097[1], tmp_7095[2] + tmp_7097[2]];
    signal tmp_7099[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7100[3] <== [tmp_7098[0] + tmp_7099[0], tmp_7098[1] + tmp_7099[1], tmp_7098[2] + tmp_7099[2]];
    signal tmp_7101[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7102[3] <== [11 * tmp_7101[0], 11 * tmp_7101[1], 11 * tmp_7101[2]];
    signal tmp_7103[3] <== [tmp_7100[0] + tmp_7102[0], tmp_7100[1] + tmp_7102[1], tmp_7100[2] + tmp_7102[2]];
    signal tmp_7104[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7105[3] <== [17 * tmp_7104[0], 17 * tmp_7104[1], 17 * tmp_7104[2]];
    signal tmp_7106[3] <== [tmp_7103[0] + tmp_7105[0], tmp_7103[1] + tmp_7105[1], tmp_7103[2] + tmp_7105[2]];
    signal tmp_7107[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7108[3] <== [2 * tmp_7107[0], 2 * tmp_7107[1], 2 * tmp_7107[2]];
    signal tmp_7109[3] <== [tmp_7106[0] + tmp_7108[0], tmp_7106[1] + tmp_7108[1], tmp_7106[2] + tmp_7108[2]];
    signal tmp_7110[3] <== [evals[91][0] - tmp_7109[0], evals[91][1] - tmp_7109[1], evals[91][2] - tmp_7109[2]];
    signal tmp_7111[3] <== CMul()(tmp_7066, tmp_7110);
    signal tmp_7112[3] <== [tmp_7063[0] + tmp_7111[0], tmp_7063[1] + tmp_7111[1], tmp_7063[2] + tmp_7111[2]];
    signal tmp_7113[3] <== CMul()(challengeQ, tmp_7112);
    signal tmp_7114[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7115[3] <== [tmp_7114[0] + evals[109][0], tmp_7114[1] + evals[109][1], tmp_7114[2] + evals[109][2]];
    signal tmp_7116[3] <== [tmp_7115[0] + evals[41][0], tmp_7115[1] + evals[41][1], tmp_7115[2] + evals[41][2]];
    signal tmp_7117[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7118[3] <== [2 * tmp_7117[0], 2 * tmp_7117[1], 2 * tmp_7117[2]];
    signal tmp_7119[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7120[3] <== [tmp_7118[0] + tmp_7119[0], tmp_7118[1] + tmp_7119[1], tmp_7118[2] + tmp_7119[2]];
    signal tmp_7121[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7122[3] <== [101 * tmp_7121[0], 101 * tmp_7121[1], 101 * tmp_7121[2]];
    signal tmp_7123[3] <== [tmp_7120[0] + tmp_7122[0], tmp_7120[1] + tmp_7122[1], tmp_7120[2] + tmp_7122[2]];
    signal tmp_7124[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7125[3] <== [63 * tmp_7124[0], 63 * tmp_7124[1], 63 * tmp_7124[2]];
    signal tmp_7126[3] <== [tmp_7123[0] + tmp_7125[0], tmp_7123[1] + tmp_7125[1], tmp_7123[2] + tmp_7125[2]];
    signal tmp_7127[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7128[3] <== [15 * tmp_7127[0], 15 * tmp_7127[1], 15 * tmp_7127[2]];
    signal tmp_7129[3] <== [tmp_7126[0] + tmp_7128[0], tmp_7126[1] + tmp_7128[1], tmp_7126[2] + tmp_7128[2]];
    signal tmp_7130[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7131[3] <== [2 * tmp_7130[0], 2 * tmp_7130[1], 2 * tmp_7130[2]];
    signal tmp_7132[3] <== [tmp_7129[0] + tmp_7131[0], tmp_7129[1] + tmp_7131[1], tmp_7129[2] + tmp_7131[2]];
    signal tmp_7133[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7134[3] <== [67 * tmp_7133[0], 67 * tmp_7133[1], 67 * tmp_7133[2]];
    signal tmp_7135[3] <== [tmp_7132[0] + tmp_7134[0], tmp_7132[1] + tmp_7134[1], tmp_7132[2] + tmp_7134[2]];
    signal tmp_7136[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7137[3] <== [22 * tmp_7136[0], 22 * tmp_7136[1], 22 * tmp_7136[2]];
    signal tmp_7138[3] <== [tmp_7135[0] + tmp_7137[0], tmp_7135[1] + tmp_7137[1], tmp_7135[2] + tmp_7137[2]];
    signal tmp_7139[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7140[3] <== [13 * tmp_7139[0], 13 * tmp_7139[1], 13 * tmp_7139[2]];
    signal tmp_7141[3] <== [tmp_7138[0] + tmp_7140[0], tmp_7138[1] + tmp_7140[1], tmp_7138[2] + tmp_7140[2]];
    signal tmp_7142[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7143[3] <== [3 * tmp_7142[0], 3 * tmp_7142[1], 3 * tmp_7142[2]];
    signal tmp_7144[3] <== [tmp_7141[0] + tmp_7143[0], tmp_7141[1] + tmp_7143[1], tmp_7141[2] + tmp_7143[2]];
    signal tmp_7145[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7146[3] <== [tmp_7144[0] + tmp_7145[0], tmp_7144[1] + tmp_7145[1], tmp_7144[2] + tmp_7145[2]];
    signal tmp_7147[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7148[3] <== [tmp_7146[0] + tmp_7147[0], tmp_7146[1] + tmp_7147[1], tmp_7146[2] + tmp_7147[2]];
    signal tmp_7149[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7150[3] <== [51 * tmp_7149[0], 51 * tmp_7149[1], 51 * tmp_7149[2]];
    signal tmp_7151[3] <== [tmp_7148[0] + tmp_7150[0], tmp_7148[1] + tmp_7150[1], tmp_7148[2] + tmp_7150[2]];
    signal tmp_7152[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7153[3] <== [tmp_7151[0] + tmp_7152[0], tmp_7151[1] + tmp_7152[1], tmp_7151[2] + tmp_7152[2]];
    signal tmp_7154[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7155[3] <== [11 * tmp_7154[0], 11 * tmp_7154[1], 11 * tmp_7154[2]];
    signal tmp_7156[3] <== [tmp_7153[0] + tmp_7155[0], tmp_7153[1] + tmp_7155[1], tmp_7153[2] + tmp_7155[2]];
    signal tmp_7157[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7158[3] <== [17 * tmp_7157[0], 17 * tmp_7157[1], 17 * tmp_7157[2]];
    signal tmp_7159[3] <== [tmp_7156[0] + tmp_7158[0], tmp_7156[1] + tmp_7158[1], tmp_7156[2] + tmp_7158[2]];
    signal tmp_7160[3] <== [evals[92][0] - tmp_7159[0], evals[92][1] - tmp_7159[1], evals[92][2] - tmp_7159[2]];
    signal tmp_7161[3] <== CMul()(tmp_7116, tmp_7160);
    signal tmp_7162[3] <== [tmp_7113[0] + tmp_7161[0], tmp_7113[1] + tmp_7161[1], tmp_7113[2] + tmp_7161[2]];
    signal tmp_7163[3] <== CMul()(challengeQ, tmp_7162);
    signal tmp_7164[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7165[3] <== [tmp_7164[0] + evals[109][0], tmp_7164[1] + evals[109][1], tmp_7164[2] + evals[109][2]];
    signal tmp_7166[3] <== [tmp_7165[0] + evals[41][0], tmp_7165[1] + evals[41][1], tmp_7165[2] + evals[41][2]];
    signal tmp_7167[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7168[3] <== [17 * tmp_7167[0], 17 * tmp_7167[1], 17 * tmp_7167[2]];
    signal tmp_7169[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7170[3] <== [2 * tmp_7169[0], 2 * tmp_7169[1], 2 * tmp_7169[2]];
    signal tmp_7171[3] <== [tmp_7168[0] + tmp_7170[0], tmp_7168[1] + tmp_7170[1], tmp_7168[2] + tmp_7170[2]];
    signal tmp_7172[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7173[3] <== [tmp_7171[0] + tmp_7172[0], tmp_7171[1] + tmp_7172[1], tmp_7171[2] + tmp_7172[2]];
    signal tmp_7174[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7175[3] <== [101 * tmp_7174[0], 101 * tmp_7174[1], 101 * tmp_7174[2]];
    signal tmp_7176[3] <== [tmp_7173[0] + tmp_7175[0], tmp_7173[1] + tmp_7175[1], tmp_7173[2] + tmp_7175[2]];
    signal tmp_7177[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7178[3] <== [63 * tmp_7177[0], 63 * tmp_7177[1], 63 * tmp_7177[2]];
    signal tmp_7179[3] <== [tmp_7176[0] + tmp_7178[0], tmp_7176[1] + tmp_7178[1], tmp_7176[2] + tmp_7178[2]];
    signal tmp_7180[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7181[3] <== [15 * tmp_7180[0], 15 * tmp_7180[1], 15 * tmp_7180[2]];
    signal tmp_7182[3] <== [tmp_7179[0] + tmp_7181[0], tmp_7179[1] + tmp_7181[1], tmp_7179[2] + tmp_7181[2]];
    signal tmp_7183[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7184[3] <== [2 * tmp_7183[0], 2 * tmp_7183[1], 2 * tmp_7183[2]];
    signal tmp_7185[3] <== [tmp_7182[0] + tmp_7184[0], tmp_7182[1] + tmp_7184[1], tmp_7182[2] + tmp_7184[2]];
    signal tmp_7186[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7187[3] <== [67 * tmp_7186[0], 67 * tmp_7186[1], 67 * tmp_7186[2]];
    signal tmp_7188[3] <== [tmp_7185[0] + tmp_7187[0], tmp_7185[1] + tmp_7187[1], tmp_7185[2] + tmp_7187[2]];
    signal tmp_7189[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7190[3] <== [22 * tmp_7189[0], 22 * tmp_7189[1], 22 * tmp_7189[2]];
    signal tmp_7191[3] <== [tmp_7188[0] + tmp_7190[0], tmp_7188[1] + tmp_7190[1], tmp_7188[2] + tmp_7190[2]];
    signal tmp_7192[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7193[3] <== [13 * tmp_7192[0], 13 * tmp_7192[1], 13 * tmp_7192[2]];
    signal tmp_7194[3] <== [tmp_7191[0] + tmp_7193[0], tmp_7191[1] + tmp_7193[1], tmp_7191[2] + tmp_7193[2]];
    signal tmp_7195[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7196[3] <== [3 * tmp_7195[0], 3 * tmp_7195[1], 3 * tmp_7195[2]];
    signal tmp_7197[3] <== [tmp_7194[0] + tmp_7196[0], tmp_7194[1] + tmp_7196[1], tmp_7194[2] + tmp_7196[2]];
    signal tmp_7198[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7199[3] <== [tmp_7197[0] + tmp_7198[0], tmp_7197[1] + tmp_7198[1], tmp_7197[2] + tmp_7198[2]];
    signal tmp_7200[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7201[3] <== [tmp_7199[0] + tmp_7200[0], tmp_7199[1] + tmp_7200[1], tmp_7199[2] + tmp_7200[2]];
    signal tmp_7202[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7203[3] <== [51 * tmp_7202[0], 51 * tmp_7202[1], 51 * tmp_7202[2]];
    signal tmp_7204[3] <== [tmp_7201[0] + tmp_7203[0], tmp_7201[1] + tmp_7203[1], tmp_7201[2] + tmp_7203[2]];
    signal tmp_7205[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7206[3] <== [tmp_7204[0] + tmp_7205[0], tmp_7204[1] + tmp_7205[1], tmp_7204[2] + tmp_7205[2]];
    signal tmp_7207[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7208[3] <== [11 * tmp_7207[0], 11 * tmp_7207[1], 11 * tmp_7207[2]];
    signal tmp_7209[3] <== [tmp_7206[0] + tmp_7208[0], tmp_7206[1] + tmp_7208[1], tmp_7206[2] + tmp_7208[2]];
    signal tmp_7210[3] <== [evals[93][0] - tmp_7209[0], evals[93][1] - tmp_7209[1], evals[93][2] - tmp_7209[2]];
    signal tmp_7211[3] <== CMul()(tmp_7166, tmp_7210);
    signal tmp_7212[3] <== [tmp_7163[0] + tmp_7211[0], tmp_7163[1] + tmp_7211[1], tmp_7163[2] + tmp_7211[2]];
    signal tmp_7213[3] <== CMul()(challengeQ, tmp_7212);
    signal tmp_7214[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7215[3] <== [tmp_7214[0] + evals[109][0], tmp_7214[1] + evals[109][1], tmp_7214[2] + evals[109][2]];
    signal tmp_7216[3] <== [tmp_7215[0] + evals[41][0], tmp_7215[1] + evals[41][1], tmp_7215[2] + evals[41][2]];
    signal tmp_7217[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7218[3] <== [11 * tmp_7217[0], 11 * tmp_7217[1], 11 * tmp_7217[2]];
    signal tmp_7219[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7220[3] <== [17 * tmp_7219[0], 17 * tmp_7219[1], 17 * tmp_7219[2]];
    signal tmp_7221[3] <== [tmp_7218[0] + tmp_7220[0], tmp_7218[1] + tmp_7220[1], tmp_7218[2] + tmp_7220[2]];
    signal tmp_7222[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7223[3] <== [2 * tmp_7222[0], 2 * tmp_7222[1], 2 * tmp_7222[2]];
    signal tmp_7224[3] <== [tmp_7221[0] + tmp_7223[0], tmp_7221[1] + tmp_7223[1], tmp_7221[2] + tmp_7223[2]];
    signal tmp_7225[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7226[3] <== [tmp_7224[0] + tmp_7225[0], tmp_7224[1] + tmp_7225[1], tmp_7224[2] + tmp_7225[2]];
    signal tmp_7227[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7228[3] <== [101 * tmp_7227[0], 101 * tmp_7227[1], 101 * tmp_7227[2]];
    signal tmp_7229[3] <== [tmp_7226[0] + tmp_7228[0], tmp_7226[1] + tmp_7228[1], tmp_7226[2] + tmp_7228[2]];
    signal tmp_7230[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7231[3] <== [63 * tmp_7230[0], 63 * tmp_7230[1], 63 * tmp_7230[2]];
    signal tmp_7232[3] <== [tmp_7229[0] + tmp_7231[0], tmp_7229[1] + tmp_7231[1], tmp_7229[2] + tmp_7231[2]];
    signal tmp_7233[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7234[3] <== [15 * tmp_7233[0], 15 * tmp_7233[1], 15 * tmp_7233[2]];
    signal tmp_7235[3] <== [tmp_7232[0] + tmp_7234[0], tmp_7232[1] + tmp_7234[1], tmp_7232[2] + tmp_7234[2]];
    signal tmp_7236[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7237[3] <== [2 * tmp_7236[0], 2 * tmp_7236[1], 2 * tmp_7236[2]];
    signal tmp_7238[3] <== [tmp_7235[0] + tmp_7237[0], tmp_7235[1] + tmp_7237[1], tmp_7235[2] + tmp_7237[2]];
    signal tmp_7239[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7240[3] <== [67 * tmp_7239[0], 67 * tmp_7239[1], 67 * tmp_7239[2]];
    signal tmp_7241[3] <== [tmp_7238[0] + tmp_7240[0], tmp_7238[1] + tmp_7240[1], tmp_7238[2] + tmp_7240[2]];
    signal tmp_7242[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7243[3] <== [22 * tmp_7242[0], 22 * tmp_7242[1], 22 * tmp_7242[2]];
    signal tmp_7244[3] <== [tmp_7241[0] + tmp_7243[0], tmp_7241[1] + tmp_7243[1], tmp_7241[2] + tmp_7243[2]];
    signal tmp_7245[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7246[3] <== [13 * tmp_7245[0], 13 * tmp_7245[1], 13 * tmp_7245[2]];
    signal tmp_7247[3] <== [tmp_7244[0] + tmp_7246[0], tmp_7244[1] + tmp_7246[1], tmp_7244[2] + tmp_7246[2]];
    signal tmp_7248[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7249[3] <== [3 * tmp_7248[0], 3 * tmp_7248[1], 3 * tmp_7248[2]];
    signal tmp_7250[3] <== [tmp_7247[0] + tmp_7249[0], tmp_7247[1] + tmp_7249[1], tmp_7247[2] + tmp_7249[2]];
    signal tmp_7251[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7252[3] <== [tmp_7250[0] + tmp_7251[0], tmp_7250[1] + tmp_7251[1], tmp_7250[2] + tmp_7251[2]];
    signal tmp_7253[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7254[3] <== [tmp_7252[0] + tmp_7253[0], tmp_7252[1] + tmp_7253[1], tmp_7252[2] + tmp_7253[2]];
    signal tmp_7255[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7256[3] <== [51 * tmp_7255[0], 51 * tmp_7255[1], 51 * tmp_7255[2]];
    signal tmp_7257[3] <== [tmp_7254[0] + tmp_7256[0], tmp_7254[1] + tmp_7256[1], tmp_7254[2] + tmp_7256[2]];
    signal tmp_7258[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7259[3] <== [tmp_7257[0] + tmp_7258[0], tmp_7257[1] + tmp_7258[1], tmp_7257[2] + tmp_7258[2]];
    signal tmp_7260[3] <== [evals[94][0] - tmp_7259[0], evals[94][1] - tmp_7259[1], evals[94][2] - tmp_7259[2]];
    signal tmp_7261[3] <== CMul()(tmp_7216, tmp_7260);
    signal tmp_7262[3] <== [tmp_7213[0] + tmp_7261[0], tmp_7213[1] + tmp_7261[1], tmp_7213[2] + tmp_7261[2]];
    signal tmp_7263[3] <== CMul()(challengeQ, tmp_7262);
    signal tmp_7264[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7265[3] <== [tmp_7264[0] + evals[109][0], tmp_7264[1] + evals[109][1], tmp_7264[2] + evals[109][2]];
    signal tmp_7266[3] <== [tmp_7265[0] + evals[41][0], tmp_7265[1] + evals[41][1], tmp_7265[2] + evals[41][2]];
    signal tmp_7267[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7268[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7269[3] <== [11 * tmp_7268[0], 11 * tmp_7268[1], 11 * tmp_7268[2]];
    signal tmp_7270[3] <== [tmp_7267[0] + tmp_7269[0], tmp_7267[1] + tmp_7269[1], tmp_7267[2] + tmp_7269[2]];
    signal tmp_7271[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7272[3] <== [17 * tmp_7271[0], 17 * tmp_7271[1], 17 * tmp_7271[2]];
    signal tmp_7273[3] <== [tmp_7270[0] + tmp_7272[0], tmp_7270[1] + tmp_7272[1], tmp_7270[2] + tmp_7272[2]];
    signal tmp_7274[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7275[3] <== [2 * tmp_7274[0], 2 * tmp_7274[1], 2 * tmp_7274[2]];
    signal tmp_7276[3] <== [tmp_7273[0] + tmp_7275[0], tmp_7273[1] + tmp_7275[1], tmp_7273[2] + tmp_7275[2]];
    signal tmp_7277[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7278[3] <== [tmp_7276[0] + tmp_7277[0], tmp_7276[1] + tmp_7277[1], tmp_7276[2] + tmp_7277[2]];
    signal tmp_7279[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7280[3] <== [101 * tmp_7279[0], 101 * tmp_7279[1], 101 * tmp_7279[2]];
    signal tmp_7281[3] <== [tmp_7278[0] + tmp_7280[0], tmp_7278[1] + tmp_7280[1], tmp_7278[2] + tmp_7280[2]];
    signal tmp_7282[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7283[3] <== [63 * tmp_7282[0], 63 * tmp_7282[1], 63 * tmp_7282[2]];
    signal tmp_7284[3] <== [tmp_7281[0] + tmp_7283[0], tmp_7281[1] + tmp_7283[1], tmp_7281[2] + tmp_7283[2]];
    signal tmp_7285[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7286[3] <== [15 * tmp_7285[0], 15 * tmp_7285[1], 15 * tmp_7285[2]];
    signal tmp_7287[3] <== [tmp_7284[0] + tmp_7286[0], tmp_7284[1] + tmp_7286[1], tmp_7284[2] + tmp_7286[2]];
    signal tmp_7288[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7289[3] <== [2 * tmp_7288[0], 2 * tmp_7288[1], 2 * tmp_7288[2]];
    signal tmp_7290[3] <== [tmp_7287[0] + tmp_7289[0], tmp_7287[1] + tmp_7289[1], tmp_7287[2] + tmp_7289[2]];
    signal tmp_7291[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7292[3] <== [67 * tmp_7291[0], 67 * tmp_7291[1], 67 * tmp_7291[2]];
    signal tmp_7293[3] <== [tmp_7290[0] + tmp_7292[0], tmp_7290[1] + tmp_7292[1], tmp_7290[2] + tmp_7292[2]];
    signal tmp_7294[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7295[3] <== [22 * tmp_7294[0], 22 * tmp_7294[1], 22 * tmp_7294[2]];
    signal tmp_7296[3] <== [tmp_7293[0] + tmp_7295[0], tmp_7293[1] + tmp_7295[1], tmp_7293[2] + tmp_7295[2]];
    signal tmp_7297[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7298[3] <== [13 * tmp_7297[0], 13 * tmp_7297[1], 13 * tmp_7297[2]];
    signal tmp_7299[3] <== [tmp_7296[0] + tmp_7298[0], tmp_7296[1] + tmp_7298[1], tmp_7296[2] + tmp_7298[2]];
    signal tmp_7300[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7301[3] <== [3 * tmp_7300[0], 3 * tmp_7300[1], 3 * tmp_7300[2]];
    signal tmp_7302[3] <== [tmp_7299[0] + tmp_7301[0], tmp_7299[1] + tmp_7301[1], tmp_7299[2] + tmp_7301[2]];
    signal tmp_7303[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7304[3] <== [tmp_7302[0] + tmp_7303[0], tmp_7302[1] + tmp_7303[1], tmp_7302[2] + tmp_7303[2]];
    signal tmp_7305[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7306[3] <== [tmp_7304[0] + tmp_7305[0], tmp_7304[1] + tmp_7305[1], tmp_7304[2] + tmp_7305[2]];
    signal tmp_7307[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7308[3] <== [51 * tmp_7307[0], 51 * tmp_7307[1], 51 * tmp_7307[2]];
    signal tmp_7309[3] <== [tmp_7306[0] + tmp_7308[0], tmp_7306[1] + tmp_7308[1], tmp_7306[2] + tmp_7308[2]];
    signal tmp_7310[3] <== [evals[95][0] - tmp_7309[0], evals[95][1] - tmp_7309[1], evals[95][2] - tmp_7309[2]];
    signal tmp_7311[3] <== CMul()(tmp_7266, tmp_7310);
    signal tmp_7312[3] <== [tmp_7263[0] + tmp_7311[0], tmp_7263[1] + tmp_7311[1], tmp_7263[2] + tmp_7311[2]];
    signal tmp_7313[3] <== CMul()(challengeQ, tmp_7312);
    signal tmp_7314[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7315[3] <== [tmp_7314[0] + evals[109][0], tmp_7314[1] + evals[109][1], tmp_7314[2] + evals[109][2]];
    signal tmp_7316[3] <== [tmp_7315[0] + evals[41][0], tmp_7315[1] + evals[41][1], tmp_7315[2] + evals[41][2]];
    signal tmp_7317[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7318[3] <== [51 * tmp_7317[0], 51 * tmp_7317[1], 51 * tmp_7317[2]];
    signal tmp_7319[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7320[3] <== [tmp_7318[0] + tmp_7319[0], tmp_7318[1] + tmp_7319[1], tmp_7318[2] + tmp_7319[2]];
    signal tmp_7321[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7322[3] <== [11 * tmp_7321[0], 11 * tmp_7321[1], 11 * tmp_7321[2]];
    signal tmp_7323[3] <== [tmp_7320[0] + tmp_7322[0], tmp_7320[1] + tmp_7322[1], tmp_7320[2] + tmp_7322[2]];
    signal tmp_7324[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7325[3] <== [17 * tmp_7324[0], 17 * tmp_7324[1], 17 * tmp_7324[2]];
    signal tmp_7326[3] <== [tmp_7323[0] + tmp_7325[0], tmp_7323[1] + tmp_7325[1], tmp_7323[2] + tmp_7325[2]];
    signal tmp_7327[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7328[3] <== [2 * tmp_7327[0], 2 * tmp_7327[1], 2 * tmp_7327[2]];
    signal tmp_7329[3] <== [tmp_7326[0] + tmp_7328[0], tmp_7326[1] + tmp_7328[1], tmp_7326[2] + tmp_7328[2]];
    signal tmp_7330[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7331[3] <== [tmp_7329[0] + tmp_7330[0], tmp_7329[1] + tmp_7330[1], tmp_7329[2] + tmp_7330[2]];
    signal tmp_7332[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7333[3] <== [101 * tmp_7332[0], 101 * tmp_7332[1], 101 * tmp_7332[2]];
    signal tmp_7334[3] <== [tmp_7331[0] + tmp_7333[0], tmp_7331[1] + tmp_7333[1], tmp_7331[2] + tmp_7333[2]];
    signal tmp_7335[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7336[3] <== [63 * tmp_7335[0], 63 * tmp_7335[1], 63 * tmp_7335[2]];
    signal tmp_7337[3] <== [tmp_7334[0] + tmp_7336[0], tmp_7334[1] + tmp_7336[1], tmp_7334[2] + tmp_7336[2]];
    signal tmp_7338[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7339[3] <== [15 * tmp_7338[0], 15 * tmp_7338[1], 15 * tmp_7338[2]];
    signal tmp_7340[3] <== [tmp_7337[0] + tmp_7339[0], tmp_7337[1] + tmp_7339[1], tmp_7337[2] + tmp_7339[2]];
    signal tmp_7341[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7342[3] <== [2 * tmp_7341[0], 2 * tmp_7341[1], 2 * tmp_7341[2]];
    signal tmp_7343[3] <== [tmp_7340[0] + tmp_7342[0], tmp_7340[1] + tmp_7342[1], tmp_7340[2] + tmp_7342[2]];
    signal tmp_7344[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7345[3] <== [67 * tmp_7344[0], 67 * tmp_7344[1], 67 * tmp_7344[2]];
    signal tmp_7346[3] <== [tmp_7343[0] + tmp_7345[0], tmp_7343[1] + tmp_7345[1], tmp_7343[2] + tmp_7345[2]];
    signal tmp_7347[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7348[3] <== [22 * tmp_7347[0], 22 * tmp_7347[1], 22 * tmp_7347[2]];
    signal tmp_7349[3] <== [tmp_7346[0] + tmp_7348[0], tmp_7346[1] + tmp_7348[1], tmp_7346[2] + tmp_7348[2]];
    signal tmp_7350[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7351[3] <== [13 * tmp_7350[0], 13 * tmp_7350[1], 13 * tmp_7350[2]];
    signal tmp_7352[3] <== [tmp_7349[0] + tmp_7351[0], tmp_7349[1] + tmp_7351[1], tmp_7349[2] + tmp_7351[2]];
    signal tmp_7353[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7354[3] <== [3 * tmp_7353[0], 3 * tmp_7353[1], 3 * tmp_7353[2]];
    signal tmp_7355[3] <== [tmp_7352[0] + tmp_7354[0], tmp_7352[1] + tmp_7354[1], tmp_7352[2] + tmp_7354[2]];
    signal tmp_7356[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7357[3] <== [tmp_7355[0] + tmp_7356[0], tmp_7355[1] + tmp_7356[1], tmp_7355[2] + tmp_7356[2]];
    signal tmp_7358[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7359[3] <== [tmp_7357[0] + tmp_7358[0], tmp_7357[1] + tmp_7358[1], tmp_7357[2] + tmp_7358[2]];
    signal tmp_7360[3] <== [evals[96][0] - tmp_7359[0], evals[96][1] - tmp_7359[1], evals[96][2] - tmp_7359[2]];
    signal tmp_7361[3] <== CMul()(tmp_7316, tmp_7360);
    signal tmp_7362[3] <== [tmp_7313[0] + tmp_7361[0], tmp_7313[1] + tmp_7361[1], tmp_7313[2] + tmp_7361[2]];
    signal tmp_7363[3] <== CMul()(challengeQ, tmp_7362);
    signal tmp_7364[3] <== [tmp_6405[0] + tmp_6066[0], tmp_6405[1] + tmp_6066[1], tmp_6405[2] + tmp_6066[2]];
    signal tmp_7365[3] <== [tmp_7364[0] + evals[109][0], tmp_7364[1] + evals[109][1], tmp_7364[2] + evals[109][2]];
    signal tmp_7366[3] <== [tmp_7365[0] + evals[41][0], tmp_7365[1] + evals[41][1], tmp_7365[2] + evals[41][2]];
    signal tmp_7367[3] <== [tmp_6414[0] + tmp_6421[0], tmp_6414[1] + tmp_6421[1], tmp_6414[2] + tmp_6421[2]];
    signal tmp_7368[3] <== [tmp_6428[0] + tmp_6435[0], tmp_6428[1] + tmp_6435[1], tmp_6428[2] + tmp_6435[2]];
    signal tmp_7369[3] <== [51 * tmp_7368[0], 51 * tmp_7368[1], 51 * tmp_7368[2]];
    signal tmp_7370[3] <== [tmp_7367[0] + tmp_7369[0], tmp_7367[1] + tmp_7369[1], tmp_7367[2] + tmp_7369[2]];
    signal tmp_7371[3] <== [tmp_6443[0] + tmp_6450[0], tmp_6443[1] + tmp_6450[1], tmp_6443[2] + tmp_6450[2]];
    signal tmp_7372[3] <== [tmp_7370[0] + tmp_7371[0], tmp_7370[1] + tmp_7371[1], tmp_7370[2] + tmp_7371[2]];
    signal tmp_7373[3] <== [tmp_6459[0] + tmp_6466[0], tmp_6459[1] + tmp_6466[1], tmp_6459[2] + tmp_6466[2]];
    signal tmp_7374[3] <== [11 * tmp_7373[0], 11 * tmp_7373[1], 11 * tmp_7373[2]];
    signal tmp_7375[3] <== [tmp_7372[0] + tmp_7374[0], tmp_7372[1] + tmp_7374[1], tmp_7372[2] + tmp_7374[2]];
    signal tmp_7376[3] <== [tmp_6474[0] + tmp_6481[0], tmp_6474[1] + tmp_6481[1], tmp_6474[2] + tmp_6481[2]];
    signal tmp_7377[3] <== [17 * tmp_7376[0], 17 * tmp_7376[1], 17 * tmp_7376[2]];
    signal tmp_7378[3] <== [tmp_7375[0] + tmp_7377[0], tmp_7375[1] + tmp_7377[1], tmp_7375[2] + tmp_7377[2]];
    signal tmp_7379[3] <== [tmp_6490[0] + tmp_6497[0], tmp_6490[1] + tmp_6497[1], tmp_6490[2] + tmp_6497[2]];
    signal tmp_7380[3] <== [2 * tmp_7379[0], 2 * tmp_7379[1], 2 * tmp_7379[2]];
    signal tmp_7381[3] <== [tmp_7378[0] + tmp_7380[0], tmp_7378[1] + tmp_7380[1], tmp_7378[2] + tmp_7380[2]];
    signal tmp_7382[3] <== [tmp_6506[0] + tmp_6513[0], tmp_6506[1] + tmp_6513[1], tmp_6506[2] + tmp_6513[2]];
    signal tmp_7383[3] <== [tmp_7381[0] + tmp_7382[0], tmp_7381[1] + tmp_7382[1], tmp_7381[2] + tmp_7382[2]];
    signal tmp_7384[3] <== [tmp_6522[0] + tmp_6529[0], tmp_6522[1] + tmp_6529[1], tmp_6522[2] + tmp_6529[2]];
    signal tmp_7385[3] <== [101 * tmp_7384[0], 101 * tmp_7384[1], 101 * tmp_7384[2]];
    signal tmp_7386[3] <== [tmp_7383[0] + tmp_7385[0], tmp_7383[1] + tmp_7385[1], tmp_7383[2] + tmp_7385[2]];
    signal tmp_7387[3] <== [tmp_6537[0] + tmp_6544[0], tmp_6537[1] + tmp_6544[1], tmp_6537[2] + tmp_6544[2]];
    signal tmp_7388[3] <== [63 * tmp_7387[0], 63 * tmp_7387[1], 63 * tmp_7387[2]];
    signal tmp_7389[3] <== [tmp_7386[0] + tmp_7388[0], tmp_7386[1] + tmp_7388[1], tmp_7386[2] + tmp_7388[2]];
    signal tmp_7390[3] <== [tmp_6553[0] + tmp_6560[0], tmp_6553[1] + tmp_6560[1], tmp_6553[2] + tmp_6560[2]];
    signal tmp_7391[3] <== [15 * tmp_7390[0], 15 * tmp_7390[1], 15 * tmp_7390[2]];
    signal tmp_7392[3] <== [tmp_7389[0] + tmp_7391[0], tmp_7389[1] + tmp_7391[1], tmp_7389[2] + tmp_7391[2]];
    signal tmp_7393[3] <== [tmp_6569[0] + tmp_6576[0], tmp_6569[1] + tmp_6576[1], tmp_6569[2] + tmp_6576[2]];
    signal tmp_7394[3] <== [2 * tmp_7393[0], 2 * tmp_7393[1], 2 * tmp_7393[2]];
    signal tmp_7395[3] <== [tmp_7392[0] + tmp_7394[0], tmp_7392[1] + tmp_7394[1], tmp_7392[2] + tmp_7394[2]];
    signal tmp_7396[3] <== [tmp_6585[0] + tmp_6592[0], tmp_6585[1] + tmp_6592[1], tmp_6585[2] + tmp_6592[2]];
    signal tmp_7397[3] <== [67 * tmp_7396[0], 67 * tmp_7396[1], 67 * tmp_7396[2]];
    signal tmp_7398[3] <== [tmp_7395[0] + tmp_7397[0], tmp_7395[1] + tmp_7397[1], tmp_7395[2] + tmp_7397[2]];
    signal tmp_7399[3] <== [tmp_6601[0] + tmp_6608[0], tmp_6601[1] + tmp_6608[1], tmp_6601[2] + tmp_6608[2]];
    signal tmp_7400[3] <== [22 * tmp_7399[0], 22 * tmp_7399[1], 22 * tmp_7399[2]];
    signal tmp_7401[3] <== [tmp_7398[0] + tmp_7400[0], tmp_7398[1] + tmp_7400[1], tmp_7398[2] + tmp_7400[2]];
    signal tmp_7402[3] <== [tmp_6617[0] + tmp_6624[0], tmp_6617[1] + tmp_6624[1], tmp_6617[2] + tmp_6624[2]];
    signal tmp_7403[3] <== [13 * tmp_7402[0], 13 * tmp_7402[1], 13 * tmp_7402[2]];
    signal tmp_7404[3] <== [tmp_7401[0] + tmp_7403[0], tmp_7401[1] + tmp_7403[1], tmp_7401[2] + tmp_7403[2]];
    signal tmp_7405[3] <== [tmp_6633[0] + tmp_6640[0], tmp_6633[1] + tmp_6640[1], tmp_6633[2] + tmp_6640[2]];
    signal tmp_7406[3] <== [3 * tmp_7405[0], 3 * tmp_7405[1], 3 * tmp_7405[2]];
    signal tmp_7407[3] <== [tmp_7404[0] + tmp_7406[0], tmp_7404[1] + tmp_7406[1], tmp_7404[2] + tmp_7406[2]];
    signal tmp_7408[3] <== [tmp_6649[0] + tmp_6656[0], tmp_6649[1] + tmp_6656[1], tmp_6649[2] + tmp_6656[2]];
    signal tmp_7409[3] <== [tmp_7407[0] + tmp_7408[0], tmp_7407[1] + tmp_7408[1], tmp_7407[2] + tmp_7408[2]];
    signal tmp_7410[3] <== [evals[97][0] - tmp_7409[0], evals[97][1] - tmp_7409[1], evals[97][2] - tmp_7409[2]];
    signal tmp_7411[3] <== CMul()(tmp_7366, tmp_7410);
    signal tmp_7412[3] <== [tmp_7363[0] + tmp_7411[0], tmp_7363[1] + tmp_7411[1], tmp_7363[2] + tmp_7411[2]];
    signal tmp_7413[3] <== CMul()(challengeQ, tmp_7412);
    signal tmp_7414[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7415[3] <== [tmp_7414[0] + evals[41][0], tmp_7414[1] + evals[41][1], tmp_7414[2] + evals[41][2]];
    signal tmp_7416[3] <== CMul()(evals[41], evals[50]);
    signal tmp_7417[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7418[3] <== CMul()(tmp_7417, evals[118]);
    signal tmp_7419[3] <== [tmp_7416[0] + tmp_7418[0], tmp_7416[1] + tmp_7418[1], tmp_7416[2] + tmp_7418[2]];
    signal tmp_7420[3] <== CMul()(evals[82], evals[82]);
    signal tmp_7421[3] <== CMul()(tmp_7420, tmp_7420);
    signal tmp_7422[3] <== CMul()(tmp_7421, tmp_7420);
    tmp_7423 <== CMul()(tmp_7422, evals[82]);
    signal tmp_7424[3] <== [tmp_6405[0] * 9372203521948909212, tmp_6405[1] * 9372203521948909212, tmp_6405[2] * 9372203521948909212];
    signal tmp_7425[3] <== [tmp_6066[0] * 11584860397836334902, tmp_6066[1] * 11584860397836334902, tmp_6066[2] * 11584860397836334902];
    signal tmp_7426[3] <== [tmp_7424[0] + tmp_7425[0], tmp_7424[1] + tmp_7425[1], tmp_7424[2] + tmp_7425[2]];
    signal tmp_7427[3] <== [evals[109][0] * 8298154099590067129, evals[109][1] * 8298154099590067129, evals[109][2] * 8298154099590067129];
    tmp_7428 <== [tmp_7426[0] + tmp_7427[0], tmp_7426[1] + tmp_7427[1], tmp_7426[2] + tmp_7427[2]];
    signal tmp_7429[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7430[3] <== CMul()(evals[83], evals[83]);
    signal tmp_7431[3] <== CMul()(tmp_7430, tmp_7430);
    signal tmp_7432[3] <== CMul()(tmp_7431, tmp_7430);
    tmp_7433 <== CMul()(tmp_7432, evals[83]);
    signal tmp_7434[3] <== [tmp_6405[0] * 13735366479310483840, tmp_6405[1] * 13735366479310483840, tmp_6405[2] * 13735366479310483840];
    signal tmp_7435[3] <== [tmp_6066[0] * 3020861528779719921, tmp_6066[1] * 3020861528779719921, tmp_6066[2] * 3020861528779719921];
    signal tmp_7436[3] <== [tmp_7434[0] + tmp_7435[0], tmp_7434[1] + tmp_7435[1], tmp_7434[2] + tmp_7435[2]];
    signal tmp_7437[3] <== [evals[109][0] * 10708178624497052143, evals[109][1] * 10708178624497052143, evals[109][2] * 10708178624497052143];
    tmp_7438 <== [tmp_7436[0] + tmp_7437[0], tmp_7436[1] + tmp_7437[1], tmp_7436[2] + tmp_7437[2]];
    signal tmp_7439[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7440[3] <== [tmp_7429[0] + tmp_7439[0], tmp_7429[1] + tmp_7439[1], tmp_7429[2] + tmp_7439[2]];
    signal tmp_7441[3] <== CMul()(evals[84], evals[84]);
    signal tmp_7442[3] <== CMul()(tmp_7441, tmp_7441);
    signal tmp_7443[3] <== CMul()(tmp_7442, tmp_7441);
    tmp_7444 <== CMul()(tmp_7443, evals[84]);
    signal tmp_7445[3] <== [tmp_6405[0] * 8803886850367906518, tmp_6405[1] * 8803886850367906518, tmp_6405[2] * 8803886850367906518];
    signal tmp_7446[3] <== [tmp_6066[0] * 8049325709262789663, tmp_6066[1] * 8049325709262789663, tmp_6066[2] * 8049325709262789663];
    signal tmp_7447[3] <== [tmp_7445[0] + tmp_7446[0], tmp_7445[1] + tmp_7446[1], tmp_7445[2] + tmp_7446[2]];
    signal tmp_7448[3] <== [evals[109][0] * 7005743732139600032, evals[109][1] * 7005743732139600032, evals[109][2] * 7005743732139600032];
    tmp_7449 <== [tmp_7447[0] + tmp_7448[0], tmp_7447[1] + tmp_7448[1], tmp_7447[2] + tmp_7448[2]];
    signal tmp_7450[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7451[3] <== [51 * tmp_7450[0], 51 * tmp_7450[1], 51 * tmp_7450[2]];
    signal tmp_7452[3] <== [tmp_7440[0] + tmp_7451[0], tmp_7440[1] + tmp_7451[1], tmp_7440[2] + tmp_7451[2]];
    signal tmp_7453[3] <== CMul()(evals[85], evals[85]);
    signal tmp_7454[3] <== CMul()(tmp_7453, tmp_7453);
    signal tmp_7455[3] <== CMul()(tmp_7454, tmp_7453);
    tmp_7456 <== CMul()(tmp_7455, evals[85]);
    signal tmp_7457[3] <== [tmp_6405[0] * 15518531785207090320, tmp_6405[1] * 15518531785207090320, tmp_6405[2] * 15518531785207090320];
    signal tmp_7458[3] <== [tmp_6066[0] * 9276251546345170559, tmp_6066[1] * 9276251546345170559, tmp_6066[2] * 9276251546345170559];
    signal tmp_7459[3] <== [tmp_7457[0] + tmp_7458[0], tmp_7457[1] + tmp_7458[1], tmp_7457[2] + tmp_7458[2]];
    signal tmp_7460[3] <== [evals[109][0] * 13988198031851058492, evals[109][1] * 13988198031851058492, evals[109][2] * 13988198031851058492];
    tmp_7461 <== [tmp_7459[0] + tmp_7460[0], tmp_7459[1] + tmp_7460[1], tmp_7459[2] + tmp_7460[2]];
    signal tmp_7462[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7463[3] <== [tmp_7452[0] + tmp_7462[0], tmp_7452[1] + tmp_7462[1], tmp_7452[2] + tmp_7462[2]];
    signal tmp_7464[3] <== CMul()(evals[86], evals[86]);
    signal tmp_7465[3] <== CMul()(tmp_7464, tmp_7464);
    signal tmp_7466[3] <== CMul()(tmp_7465, tmp_7464);
    tmp_7467 <== CMul()(tmp_7466, evals[86]);
    signal tmp_7468[3] <== [tmp_6405[0] * 11602150072024304307, tmp_6405[1] * 11602150072024304307, tmp_6405[2] * 11602150072024304307];
    signal tmp_7469[3] <== [tmp_6066[0] * 14380888688809393725, tmp_6066[1] * 14380888688809393725, tmp_6066[2] * 14380888688809393725];
    signal tmp_7470[3] <== [tmp_7468[0] + tmp_7469[0], tmp_7468[1] + tmp_7469[1], tmp_7468[2] + tmp_7469[2]];
    signal tmp_7471[3] <== [evals[109][0] * 10268836078569396437, evals[109][1] * 10268836078569396437, evals[109][2] * 10268836078569396437];
    tmp_7472 <== [tmp_7470[0] + tmp_7471[0], tmp_7470[1] + tmp_7471[1], tmp_7470[2] + tmp_7471[2]];
    signal tmp_7473[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7474[3] <== [11 * tmp_7473[0], 11 * tmp_7473[1], 11 * tmp_7473[2]];
    signal tmp_7475[3] <== [tmp_7463[0] + tmp_7474[0], tmp_7463[1] + tmp_7474[1], tmp_7463[2] + tmp_7474[2]];
    signal tmp_7476[3] <== CMul()(evals[87], evals[87]);
    signal tmp_7477[3] <== CMul()(tmp_7476, tmp_7476);
    signal tmp_7478[3] <== CMul()(tmp_7477, tmp_7476);
    tmp_7479 <== CMul()(tmp_7478, evals[87]);
    signal tmp_7480[3] <== [tmp_6405[0] * 2912873838477326782, tmp_6405[1] * 2912873838477326782, tmp_6405[2] * 2912873838477326782];
    signal tmp_7481[3] <== [tmp_6066[0] * 3586048355069849144, tmp_6066[1] * 3586048355069849144, tmp_6066[2] * 3586048355069849144];
    signal tmp_7482[3] <== [tmp_7480[0] + tmp_7481[0], tmp_7480[1] + tmp_7481[1], tmp_7480[2] + tmp_7481[2]];
    signal tmp_7483[3] <== [evals[109][0] * 9672572720882889100, evals[109][1] * 9672572720882889100, evals[109][2] * 9672572720882889100];
    tmp_7484 <== [tmp_7482[0] + tmp_7483[0], tmp_7482[1] + tmp_7483[1], tmp_7482[2] + tmp_7483[2]];
    signal tmp_7485[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7486[3] <== [17 * tmp_7485[0], 17 * tmp_7485[1], 17 * tmp_7485[2]];
    signal tmp_7487[3] <== [tmp_7475[0] + tmp_7486[0], tmp_7475[1] + tmp_7486[1], tmp_7475[2] + tmp_7486[2]];
    signal tmp_7488[3] <== CMul()(evals[88], evals[88]);
    signal tmp_7489[3] <== CMul()(tmp_7488, tmp_7488);
    signal tmp_7490[3] <== CMul()(tmp_7489, tmp_7488);
    tmp_7491 <== CMul()(tmp_7490, evals[88]);
    signal tmp_7492[3] <== [tmp_6405[0] * 14450902135526609586, tmp_6405[1] * 14450902135526609586, tmp_6405[2] * 14450902135526609586];
    signal tmp_7493[3] <== [tmp_6066[0] * 2685528509139446373, tmp_6066[1] * 2685528509139446373, tmp_6066[2] * 2685528509139446373];
    signal tmp_7494[3] <== [tmp_7492[0] + tmp_7493[0], tmp_7492[1] + tmp_7493[1], tmp_7492[2] + tmp_7493[2]];
    signal tmp_7495[3] <== [evals[109][0] * 7773309602241386138, evals[109][1] * 7773309602241386138, evals[109][2] * 7773309602241386138];
    tmp_7496 <== [tmp_7494[0] + tmp_7495[0], tmp_7494[1] + tmp_7495[1], tmp_7494[2] + tmp_7495[2]];
    signal tmp_7497[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7498[3] <== [2 * tmp_7497[0], 2 * tmp_7497[1], 2 * tmp_7497[2]];
    signal tmp_7499[3] <== [tmp_7487[0] + tmp_7498[0], tmp_7487[1] + tmp_7498[1], tmp_7487[2] + tmp_7498[2]];
    signal tmp_7500[3] <== CMul()(evals[89], evals[89]);
    signal tmp_7501[3] <== CMul()(tmp_7500, tmp_7500);
    signal tmp_7502[3] <== CMul()(tmp_7501, tmp_7500);
    tmp_7503 <== CMul()(tmp_7502, evals[89]);
    signal tmp_7504[3] <== [tmp_6405[0] * 9299676334145756732, tmp_6405[1] * 9299676334145756732, tmp_6405[2] * 9299676334145756732];
    signal tmp_7505[3] <== [tmp_6066[0] * 7618863604684140457, tmp_6066[1] * 7618863604684140457, tmp_6066[2] * 7618863604684140457];
    signal tmp_7506[3] <== [tmp_7504[0] + tmp_7505[0], tmp_7504[1] + tmp_7505[1], tmp_7504[2] + tmp_7505[2]];
    signal tmp_7507[3] <== [evals[109][0] * 13455068909528507193, evals[109][1] * 13455068909528507193, evals[109][2] * 13455068909528507193];
    tmp_7508 <== [tmp_7506[0] + tmp_7507[0], tmp_7506[1] + tmp_7507[1], tmp_7506[2] + tmp_7507[2]];
    signal tmp_7509[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7510[3] <== [tmp_7499[0] + tmp_7509[0], tmp_7499[1] + tmp_7509[1], tmp_7499[2] + tmp_7509[2]];
    signal tmp_7511[3] <== CMul()(evals[90], evals[90]);
    signal tmp_7512[3] <== CMul()(tmp_7511, tmp_7511);
    signal tmp_7513[3] <== CMul()(tmp_7512, tmp_7511);
    tmp_7514 <== CMul()(tmp_7513, evals[90]);
    signal tmp_7515[3] <== [tmp_6405[0] * 7761860768169821826, tmp_6405[1] * 7761860768169821826, tmp_6405[2] * 7761860768169821826];
    signal tmp_7516[3] <== [tmp_6066[0] * 15645272489037253163, tmp_6066[1] * 15645272489037253163, tmp_6066[2] * 15645272489037253163];
    signal tmp_7517[3] <== [tmp_7515[0] + tmp_7516[0], tmp_7515[1] + tmp_7516[1], tmp_7515[2] + tmp_7516[2]];
    signal tmp_7518[3] <== [evals[109][0] * 17592051316228424355, evals[109][1] * 17592051316228424355, evals[109][2] * 17592051316228424355];
    tmp_7519 <== [tmp_7517[0] + tmp_7518[0], tmp_7517[1] + tmp_7518[1], tmp_7517[2] + tmp_7518[2]];
    signal tmp_7520[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7521[3] <== [101 * tmp_7520[0], 101 * tmp_7520[1], 101 * tmp_7520[2]];
    signal tmp_7522[3] <== [tmp_7510[0] + tmp_7521[0], tmp_7510[1] + tmp_7521[1], tmp_7510[2] + tmp_7521[2]];
    signal tmp_7523[3] <== CMul()(evals[91], evals[91]);
    signal tmp_7524[3] <== CMul()(tmp_7523, tmp_7523);
    signal tmp_7525[3] <== CMul()(tmp_7524, tmp_7523);
    tmp_7526 <== CMul()(tmp_7525, evals[91]);
    signal tmp_7527[3] <== [tmp_6405[0] * 8088122716594392876, tmp_6405[1] * 8088122716594392876, tmp_6405[2] * 8088122716594392876];
    signal tmp_7528[3] <== [tmp_6066[0] * 14391536196044986046, tmp_6066[1] * 14391536196044986046, tmp_6066[2] * 14391536196044986046];
    signal tmp_7529[3] <== [tmp_7527[0] + tmp_7528[0], tmp_7527[1] + tmp_7528[1], tmp_7527[2] + tmp_7528[2]];
    signal tmp_7530[3] <== [evals[109][0] * 7038125295028811227, evals[109][1] * 7038125295028811227, evals[109][2] * 7038125295028811227];
    tmp_7531 <== [tmp_7529[0] + tmp_7530[0], tmp_7529[1] + tmp_7530[1], tmp_7529[2] + tmp_7530[2]];
    signal tmp_7532[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7533[3] <== [63 * tmp_7532[0], 63 * tmp_7532[1], 63 * tmp_7532[2]];
    signal tmp_7534[3] <== [tmp_7522[0] + tmp_7533[0], tmp_7522[1] + tmp_7533[1], tmp_7522[2] + tmp_7533[2]];
    signal tmp_7535[3] <== CMul()(evals[92], evals[92]);
    signal tmp_7536[3] <== CMul()(tmp_7535, tmp_7535);
    signal tmp_7537[3] <== CMul()(tmp_7536, tmp_7535);
    tmp_7538 <== CMul()(tmp_7537, evals[92]);
    signal tmp_7539[3] <== [tmp_6405[0] * 7439479476952963757, tmp_6405[1] * 7439479476952963757, tmp_6405[2] * 7439479476952963757];
    signal tmp_7540[3] <== [tmp_6066[0] * 13677749754501759967, tmp_6066[1] * 13677749754501759967, tmp_6066[2] * 13677749754501759967];
    signal tmp_7541[3] <== [tmp_7539[0] + tmp_7540[0], tmp_7539[1] + tmp_7540[1], tmp_7539[2] + tmp_7540[2]];
    signal tmp_7542[3] <== [evals[109][0] * 9310165980888335421, evals[109][1] * 9310165980888335421, evals[109][2] * 9310165980888335421];
    tmp_7543 <== [tmp_7541[0] + tmp_7542[0], tmp_7541[1] + tmp_7542[1], tmp_7541[2] + tmp_7542[2]];
    signal tmp_7544[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7545[3] <== [15 * tmp_7544[0], 15 * tmp_7544[1], 15 * tmp_7544[2]];
    signal tmp_7546[3] <== [tmp_7534[0] + tmp_7545[0], tmp_7534[1] + tmp_7545[1], tmp_7534[2] + tmp_7545[2]];
    signal tmp_7547[3] <== CMul()(evals[93], evals[93]);
    signal tmp_7548[3] <== CMul()(tmp_7547, tmp_7547);
    signal tmp_7549[3] <== CMul()(tmp_7548, tmp_7547);
    tmp_7550 <== CMul()(tmp_7549, evals[93]);
    signal tmp_7551[3] <== [tmp_6405[0] * 13142435206192622686, tmp_6405[1] * 13142435206192622686, tmp_6405[2] * 13142435206192622686];
    signal tmp_7552[3] <== [tmp_6066[0] * 4455490635243459169, tmp_6066[1] * 4455490635243459169, tmp_6066[2] * 4455490635243459169];
    signal tmp_7553[3] <== [tmp_7551[0] + tmp_7552[0], tmp_7551[1] + tmp_7552[1], tmp_7551[2] + tmp_7552[2]];
    signal tmp_7554[3] <== [evals[109][0] * 7124955715283261894, evals[109][1] * 7124955715283261894, evals[109][2] * 7124955715283261894];
    tmp_7555 <== [tmp_7553[0] + tmp_7554[0], tmp_7553[1] + tmp_7554[1], tmp_7553[2] + tmp_7554[2]];
    signal tmp_7556[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7557[3] <== [2 * tmp_7556[0], 2 * tmp_7556[1], 2 * tmp_7556[2]];
    signal tmp_7558[3] <== [tmp_7546[0] + tmp_7557[0], tmp_7546[1] + tmp_7557[1], tmp_7546[2] + tmp_7557[2]];
    signal tmp_7559[3] <== CMul()(evals[94], evals[94]);
    signal tmp_7560[3] <== CMul()(tmp_7559, tmp_7559);
    signal tmp_7561[3] <== CMul()(tmp_7560, tmp_7559);
    tmp_7562 <== CMul()(tmp_7561, evals[94]);
    signal tmp_7563[3] <== [tmp_6405[0] * 15091535460073700241, tmp_6405[1] * 15091535460073700241, tmp_6405[2] * 15091535460073700241];
    signal tmp_7564[3] <== [tmp_6066[0] * 4104818627819342438, tmp_6066[1] * 4104818627819342438, tmp_6066[2] * 4104818627819342438];
    signal tmp_7565[3] <== [tmp_7563[0] + tmp_7564[0], tmp_7563[1] + tmp_7564[1], tmp_7563[2] + tmp_7564[2]];
    signal tmp_7566[3] <== [evals[109][0] * 6894779481115729216, evals[109][1] * 6894779481115729216, evals[109][2] * 6894779481115729216];
    tmp_7567 <== [tmp_7565[0] + tmp_7566[0], tmp_7565[1] + tmp_7566[1], tmp_7565[2] + tmp_7566[2]];
    signal tmp_7568[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7569[3] <== [67 * tmp_7568[0], 67 * tmp_7568[1], 67 * tmp_7568[2]];
    signal tmp_7570[3] <== [tmp_7558[0] + tmp_7569[0], tmp_7558[1] + tmp_7569[1], tmp_7558[2] + tmp_7569[2]];
    signal tmp_7571[3] <== CMul()(evals[95], evals[95]);
    signal tmp_7572[3] <== CMul()(tmp_7571, tmp_7571);
    signal tmp_7573[3] <== CMul()(tmp_7572, tmp_7571);
    tmp_7574 <== CMul()(tmp_7573, evals[95]);
    signal tmp_7575[3] <== [tmp_6405[0] * 1246503809481338515, tmp_6405[1] * 1246503809481338515, tmp_6405[2] * 1246503809481338515];
    signal tmp_7576[3] <== [tmp_6066[0] * 11257046194955612097, tmp_6066[1] * 11257046194955612097, tmp_6066[2] * 11257046194955612097];
    signal tmp_7577[3] <== [tmp_7575[0] + tmp_7576[0], tmp_7575[1] + tmp_7576[1], tmp_7575[2] + tmp_7576[2]];
    signal tmp_7578[3] <== [evals[109][0] * 1173635621545091517, evals[109][1] * 1173635621545091517, evals[109][2] * 1173635621545091517];
    tmp_7579 <== [tmp_7577[0] + tmp_7578[0], tmp_7577[1] + tmp_7578[1], tmp_7577[2] + tmp_7578[2]];
    signal tmp_7580[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7581[3] <== [22 * tmp_7580[0], 22 * tmp_7580[1], 22 * tmp_7580[2]];
    signal tmp_7582[3] <== [tmp_7570[0] + tmp_7581[0], tmp_7570[1] + tmp_7581[1], tmp_7570[2] + tmp_7581[2]];
    signal tmp_7583[3] <== CMul()(evals[96], evals[96]);
    signal tmp_7584[3] <== CMul()(tmp_7583, tmp_7583);
    signal tmp_7585[3] <== CMul()(tmp_7584, tmp_7583);
    tmp_7586 <== CMul()(tmp_7585, evals[96]);
    signal tmp_7587[3] <== [tmp_6405[0] * 15424727577574863466, tmp_6405[1] * 15424727577574863466, tmp_6405[2] * 15424727577574863466];
    signal tmp_7588[3] <== [tmp_6066[0] * 17740901046700926130, tmp_6066[1] * 17740901046700926130, tmp_6066[2] * 17740901046700926130];
    signal tmp_7589[3] <== [tmp_7587[0] + tmp_7588[0], tmp_7587[1] + tmp_7588[1], tmp_7587[2] + tmp_7588[2]];
    signal tmp_7590[3] <== [evals[109][0] * 2648585141728028210, evals[109][1] * 2648585141728028210, evals[109][2] * 2648585141728028210];
    tmp_7591 <== [tmp_7589[0] + tmp_7590[0], tmp_7589[1] + tmp_7590[1], tmp_7589[2] + tmp_7590[2]];
    signal tmp_7592[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7593[3] <== [13 * tmp_7592[0], 13 * tmp_7592[1], 13 * tmp_7592[2]];
    signal tmp_7594[3] <== [tmp_7582[0] + tmp_7593[0], tmp_7582[1] + tmp_7593[1], tmp_7582[2] + tmp_7593[2]];
    signal tmp_7595[3] <== CMul()(evals[97], evals[97]);
    signal tmp_7596[3] <== CMul()(tmp_7595, tmp_7595);
    signal tmp_7597[3] <== CMul()(tmp_7596, tmp_7595);
    tmp_7598 <== CMul()(tmp_7597, evals[97]);
    signal tmp_7599[3] <== [tmp_6405[0] * 4922985543021971848, tmp_6405[1] * 4922985543021971848, tmp_6405[2] * 4922985543021971848];
    signal tmp_7600[3] <== [tmp_6066[0] * 10900350254403521826, tmp_6066[1] * 10900350254403521826, tmp_6066[2] * 10900350254403521826];
    signal tmp_7601[3] <== [tmp_7599[0] + tmp_7600[0], tmp_7599[1] + tmp_7600[1], tmp_7599[2] + tmp_7600[2]];
    signal tmp_7602[3] <== [evals[109][0] * 10841588854694617600, evals[109][1] * 10841588854694617600, evals[109][2] * 10841588854694617600];
    tmp_7603 <== [tmp_7601[0] + tmp_7602[0], tmp_7601[1] + tmp_7602[1], tmp_7601[2] + tmp_7602[2]];
    signal tmp_7604[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7605[3] <== [3 * tmp_7604[0], 3 * tmp_7604[1], 3 * tmp_7604[2]];
    signal tmp_7606[3] <== [tmp_7594[0] + tmp_7605[0], tmp_7594[1] + tmp_7605[1], tmp_7594[2] + tmp_7605[2]];
    signal tmp_7607[3] <== [tmp_7419[0] - tmp_7606[0], tmp_7419[1] - tmp_7606[1], tmp_7419[2] - tmp_7606[2]];
    signal tmp_7608[3] <== CMul()(tmp_7415, tmp_7607);
    signal tmp_7609[3] <== [tmp_7413[0] + tmp_7608[0], tmp_7413[1] + tmp_7608[1], tmp_7413[2] + tmp_7608[2]];
    signal tmp_7610[3] <== CMul()(challengeQ, tmp_7609);
    signal tmp_7611[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7612[3] <== [tmp_7611[0] + evals[41][0], tmp_7611[1] + evals[41][1], tmp_7611[2] + evals[41][2]];
    signal tmp_7613[3] <== CMul()(evals[41], evals[51]);
    signal tmp_7614[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7615[3] <== CMul()(tmp_7614, evals[119]);
    signal tmp_7616[3] <== [tmp_7613[0] + tmp_7615[0], tmp_7613[1] + tmp_7615[1], tmp_7613[2] + tmp_7615[2]];
    signal tmp_7617[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7618[3] <== [3 * tmp_7617[0], 3 * tmp_7617[1], 3 * tmp_7617[2]];
    signal tmp_7619[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7620[3] <== [tmp_7618[0] + tmp_7619[0], tmp_7618[1] + tmp_7619[1], tmp_7618[2] + tmp_7619[2]];
    signal tmp_7621[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7622[3] <== [tmp_7620[0] + tmp_7621[0], tmp_7620[1] + tmp_7621[1], tmp_7620[2] + tmp_7621[2]];
    signal tmp_7623[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7624[3] <== [51 * tmp_7623[0], 51 * tmp_7623[1], 51 * tmp_7623[2]];
    signal tmp_7625[3] <== [tmp_7622[0] + tmp_7624[0], tmp_7622[1] + tmp_7624[1], tmp_7622[2] + tmp_7624[2]];
    signal tmp_7626[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7627[3] <== [tmp_7625[0] + tmp_7626[0], tmp_7625[1] + tmp_7626[1], tmp_7625[2] + tmp_7626[2]];
    signal tmp_7628[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7629[3] <== [11 * tmp_7628[0], 11 * tmp_7628[1], 11 * tmp_7628[2]];
    signal tmp_7630[3] <== [tmp_7627[0] + tmp_7629[0], tmp_7627[1] + tmp_7629[1], tmp_7627[2] + tmp_7629[2]];
    signal tmp_7631[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7632[3] <== [17 * tmp_7631[0], 17 * tmp_7631[1], 17 * tmp_7631[2]];
    signal tmp_7633[3] <== [tmp_7630[0] + tmp_7632[0], tmp_7630[1] + tmp_7632[1], tmp_7630[2] + tmp_7632[2]];
    signal tmp_7634[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7635[3] <== [2 * tmp_7634[0], 2 * tmp_7634[1], 2 * tmp_7634[2]];
    signal tmp_7636[3] <== [tmp_7633[0] + tmp_7635[0], tmp_7633[1] + tmp_7635[1], tmp_7633[2] + tmp_7635[2]];
    signal tmp_7637[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7638[3] <== [tmp_7636[0] + tmp_7637[0], tmp_7636[1] + tmp_7637[1], tmp_7636[2] + tmp_7637[2]];
    signal tmp_7639[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7640[3] <== [101 * tmp_7639[0], 101 * tmp_7639[1], 101 * tmp_7639[2]];
    signal tmp_7641[3] <== [tmp_7638[0] + tmp_7640[0], tmp_7638[1] + tmp_7640[1], tmp_7638[2] + tmp_7640[2]];
    signal tmp_7642[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7643[3] <== [63 * tmp_7642[0], 63 * tmp_7642[1], 63 * tmp_7642[2]];
    signal tmp_7644[3] <== [tmp_7641[0] + tmp_7643[0], tmp_7641[1] + tmp_7643[1], tmp_7641[2] + tmp_7643[2]];
    signal tmp_7645[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7646[3] <== [15 * tmp_7645[0], 15 * tmp_7645[1], 15 * tmp_7645[2]];
    signal tmp_7647[3] <== [tmp_7644[0] + tmp_7646[0], tmp_7644[1] + tmp_7646[1], tmp_7644[2] + tmp_7646[2]];
    signal tmp_7648[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7649[3] <== [2 * tmp_7648[0], 2 * tmp_7648[1], 2 * tmp_7648[2]];
    signal tmp_7650[3] <== [tmp_7647[0] + tmp_7649[0], tmp_7647[1] + tmp_7649[1], tmp_7647[2] + tmp_7649[2]];
    signal tmp_7651[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7652[3] <== [67 * tmp_7651[0], 67 * tmp_7651[1], 67 * tmp_7651[2]];
    signal tmp_7653[3] <== [tmp_7650[0] + tmp_7652[0], tmp_7650[1] + tmp_7652[1], tmp_7650[2] + tmp_7652[2]];
    signal tmp_7654[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7655[3] <== [22 * tmp_7654[0], 22 * tmp_7654[1], 22 * tmp_7654[2]];
    signal tmp_7656[3] <== [tmp_7653[0] + tmp_7655[0], tmp_7653[1] + tmp_7655[1], tmp_7653[2] + tmp_7655[2]];
    signal tmp_7657[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7658[3] <== [13 * tmp_7657[0], 13 * tmp_7657[1], 13 * tmp_7657[2]];
    signal tmp_7659[3] <== [tmp_7656[0] + tmp_7658[0], tmp_7656[1] + tmp_7658[1], tmp_7656[2] + tmp_7658[2]];
    signal tmp_7660[3] <== [tmp_7616[0] - tmp_7659[0], tmp_7616[1] - tmp_7659[1], tmp_7616[2] - tmp_7659[2]];
    signal tmp_7661[3] <== CMul()(tmp_7612, tmp_7660);
    signal tmp_7662[3] <== [tmp_7610[0] + tmp_7661[0], tmp_7610[1] + tmp_7661[1], tmp_7610[2] + tmp_7661[2]];
    signal tmp_7663[3] <== CMul()(challengeQ, tmp_7662);
    signal tmp_7664[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7665[3] <== [tmp_7664[0] + evals[41][0], tmp_7664[1] + evals[41][1], tmp_7664[2] + evals[41][2]];
    signal tmp_7666[3] <== CMul()(evals[41], evals[52]);
    signal tmp_7667[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7668[3] <== CMul()(tmp_7667, evals[120]);
    signal tmp_7669[3] <== [tmp_7666[0] + tmp_7668[0], tmp_7666[1] + tmp_7668[1], tmp_7666[2] + tmp_7668[2]];
    signal tmp_7670[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7671[3] <== [13 * tmp_7670[0], 13 * tmp_7670[1], 13 * tmp_7670[2]];
    signal tmp_7672[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7673[3] <== [3 * tmp_7672[0], 3 * tmp_7672[1], 3 * tmp_7672[2]];
    signal tmp_7674[3] <== [tmp_7671[0] + tmp_7673[0], tmp_7671[1] + tmp_7673[1], tmp_7671[2] + tmp_7673[2]];
    signal tmp_7675[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7676[3] <== [tmp_7674[0] + tmp_7675[0], tmp_7674[1] + tmp_7675[1], tmp_7674[2] + tmp_7675[2]];
    signal tmp_7677[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7678[3] <== [tmp_7676[0] + tmp_7677[0], tmp_7676[1] + tmp_7677[1], tmp_7676[2] + tmp_7677[2]];
    signal tmp_7679[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7680[3] <== [51 * tmp_7679[0], 51 * tmp_7679[1], 51 * tmp_7679[2]];
    signal tmp_7681[3] <== [tmp_7678[0] + tmp_7680[0], tmp_7678[1] + tmp_7680[1], tmp_7678[2] + tmp_7680[2]];
    signal tmp_7682[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7683[3] <== [tmp_7681[0] + tmp_7682[0], tmp_7681[1] + tmp_7682[1], tmp_7681[2] + tmp_7682[2]];
    signal tmp_7684[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7685[3] <== [11 * tmp_7684[0], 11 * tmp_7684[1], 11 * tmp_7684[2]];
    signal tmp_7686[3] <== [tmp_7683[0] + tmp_7685[0], tmp_7683[1] + tmp_7685[1], tmp_7683[2] + tmp_7685[2]];
    signal tmp_7687[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7688[3] <== [17 * tmp_7687[0], 17 * tmp_7687[1], 17 * tmp_7687[2]];
    signal tmp_7689[3] <== [tmp_7686[0] + tmp_7688[0], tmp_7686[1] + tmp_7688[1], tmp_7686[2] + tmp_7688[2]];
    signal tmp_7690[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7691[3] <== [2 * tmp_7690[0], 2 * tmp_7690[1], 2 * tmp_7690[2]];
    signal tmp_7692[3] <== [tmp_7689[0] + tmp_7691[0], tmp_7689[1] + tmp_7691[1], tmp_7689[2] + tmp_7691[2]];
    signal tmp_7693[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7694[3] <== [tmp_7692[0] + tmp_7693[0], tmp_7692[1] + tmp_7693[1], tmp_7692[2] + tmp_7693[2]];
    signal tmp_7695[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7696[3] <== [101 * tmp_7695[0], 101 * tmp_7695[1], 101 * tmp_7695[2]];
    signal tmp_7697[3] <== [tmp_7694[0] + tmp_7696[0], tmp_7694[1] + tmp_7696[1], tmp_7694[2] + tmp_7696[2]];
    signal tmp_7698[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7699[3] <== [63 * tmp_7698[0], 63 * tmp_7698[1], 63 * tmp_7698[2]];
    signal tmp_7700[3] <== [tmp_7697[0] + tmp_7699[0], tmp_7697[1] + tmp_7699[1], tmp_7697[2] + tmp_7699[2]];
    signal tmp_7701[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7702[3] <== [15 * tmp_7701[0], 15 * tmp_7701[1], 15 * tmp_7701[2]];
    signal tmp_7703[3] <== [tmp_7700[0] + tmp_7702[0], tmp_7700[1] + tmp_7702[1], tmp_7700[2] + tmp_7702[2]];
    signal tmp_7704[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7705[3] <== [2 * tmp_7704[0], 2 * tmp_7704[1], 2 * tmp_7704[2]];
    signal tmp_7706[3] <== [tmp_7703[0] + tmp_7705[0], tmp_7703[1] + tmp_7705[1], tmp_7703[2] + tmp_7705[2]];
    signal tmp_7707[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7708[3] <== [67 * tmp_7707[0], 67 * tmp_7707[1], 67 * tmp_7707[2]];
    signal tmp_7709[3] <== [tmp_7706[0] + tmp_7708[0], tmp_7706[1] + tmp_7708[1], tmp_7706[2] + tmp_7708[2]];
    signal tmp_7710[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7711[3] <== [22 * tmp_7710[0], 22 * tmp_7710[1], 22 * tmp_7710[2]];
    signal tmp_7712[3] <== [tmp_7709[0] + tmp_7711[0], tmp_7709[1] + tmp_7711[1], tmp_7709[2] + tmp_7711[2]];
    signal tmp_7713[3] <== [tmp_7669[0] - tmp_7712[0], tmp_7669[1] - tmp_7712[1], tmp_7669[2] - tmp_7712[2]];
    signal tmp_7714[3] <== CMul()(tmp_7665, tmp_7713);
    signal tmp_7715[3] <== [tmp_7663[0] + tmp_7714[0], tmp_7663[1] + tmp_7714[1], tmp_7663[2] + tmp_7714[2]];
    signal tmp_7716[3] <== CMul()(challengeQ, tmp_7715);
    signal tmp_7717[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7718[3] <== [tmp_7717[0] + evals[41][0], tmp_7717[1] + evals[41][1], tmp_7717[2] + evals[41][2]];
    signal tmp_7719[3] <== CMul()(evals[41], evals[53]);
    signal tmp_7720[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7721[3] <== CMul()(tmp_7720, evals[121]);
    signal tmp_7722[3] <== [tmp_7719[0] + tmp_7721[0], tmp_7719[1] + tmp_7721[1], tmp_7719[2] + tmp_7721[2]];
    signal tmp_7723[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7724[3] <== [22 * tmp_7723[0], 22 * tmp_7723[1], 22 * tmp_7723[2]];
    signal tmp_7725[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7726[3] <== [13 * tmp_7725[0], 13 * tmp_7725[1], 13 * tmp_7725[2]];
    signal tmp_7727[3] <== [tmp_7724[0] + tmp_7726[0], tmp_7724[1] + tmp_7726[1], tmp_7724[2] + tmp_7726[2]];
    signal tmp_7728[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7729[3] <== [3 * tmp_7728[0], 3 * tmp_7728[1], 3 * tmp_7728[2]];
    signal tmp_7730[3] <== [tmp_7727[0] + tmp_7729[0], tmp_7727[1] + tmp_7729[1], tmp_7727[2] + tmp_7729[2]];
    signal tmp_7731[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7732[3] <== [tmp_7730[0] + tmp_7731[0], tmp_7730[1] + tmp_7731[1], tmp_7730[2] + tmp_7731[2]];
    signal tmp_7733[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7734[3] <== [tmp_7732[0] + tmp_7733[0], tmp_7732[1] + tmp_7733[1], tmp_7732[2] + tmp_7733[2]];
    signal tmp_7735[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7736[3] <== [51 * tmp_7735[0], 51 * tmp_7735[1], 51 * tmp_7735[2]];
    signal tmp_7737[3] <== [tmp_7734[0] + tmp_7736[0], tmp_7734[1] + tmp_7736[1], tmp_7734[2] + tmp_7736[2]];
    signal tmp_7738[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7739[3] <== [tmp_7737[0] + tmp_7738[0], tmp_7737[1] + tmp_7738[1], tmp_7737[2] + tmp_7738[2]];
    signal tmp_7740[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7741[3] <== [11 * tmp_7740[0], 11 * tmp_7740[1], 11 * tmp_7740[2]];
    signal tmp_7742[3] <== [tmp_7739[0] + tmp_7741[0], tmp_7739[1] + tmp_7741[1], tmp_7739[2] + tmp_7741[2]];
    signal tmp_7743[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7744[3] <== [17 * tmp_7743[0], 17 * tmp_7743[1], 17 * tmp_7743[2]];
    signal tmp_7745[3] <== [tmp_7742[0] + tmp_7744[0], tmp_7742[1] + tmp_7744[1], tmp_7742[2] + tmp_7744[2]];
    signal tmp_7746[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7747[3] <== [2 * tmp_7746[0], 2 * tmp_7746[1], 2 * tmp_7746[2]];
    signal tmp_7748[3] <== [tmp_7745[0] + tmp_7747[0], tmp_7745[1] + tmp_7747[1], tmp_7745[2] + tmp_7747[2]];
    signal tmp_7749[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7750[3] <== [tmp_7748[0] + tmp_7749[0], tmp_7748[1] + tmp_7749[1], tmp_7748[2] + tmp_7749[2]];
    signal tmp_7751[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7752[3] <== [101 * tmp_7751[0], 101 * tmp_7751[1], 101 * tmp_7751[2]];
    signal tmp_7753[3] <== [tmp_7750[0] + tmp_7752[0], tmp_7750[1] + tmp_7752[1], tmp_7750[2] + tmp_7752[2]];
    signal tmp_7754[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7755[3] <== [63 * tmp_7754[0], 63 * tmp_7754[1], 63 * tmp_7754[2]];
    signal tmp_7756[3] <== [tmp_7753[0] + tmp_7755[0], tmp_7753[1] + tmp_7755[1], tmp_7753[2] + tmp_7755[2]];
    signal tmp_7757[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7758[3] <== [15 * tmp_7757[0], 15 * tmp_7757[1], 15 * tmp_7757[2]];
    signal tmp_7759[3] <== [tmp_7756[0] + tmp_7758[0], tmp_7756[1] + tmp_7758[1], tmp_7756[2] + tmp_7758[2]];
    signal tmp_7760[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7761[3] <== [2 * tmp_7760[0], 2 * tmp_7760[1], 2 * tmp_7760[2]];
    signal tmp_7762[3] <== [tmp_7759[0] + tmp_7761[0], tmp_7759[1] + tmp_7761[1], tmp_7759[2] + tmp_7761[2]];
    signal tmp_7763[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7764[3] <== [67 * tmp_7763[0], 67 * tmp_7763[1], 67 * tmp_7763[2]];
    signal tmp_7765[3] <== [tmp_7762[0] + tmp_7764[0], tmp_7762[1] + tmp_7764[1], tmp_7762[2] + tmp_7764[2]];
    signal tmp_7766[3] <== [tmp_7722[0] - tmp_7765[0], tmp_7722[1] - tmp_7765[1], tmp_7722[2] - tmp_7765[2]];
    signal tmp_7767[3] <== CMul()(tmp_7718, tmp_7766);
    signal tmp_7768[3] <== [tmp_7716[0] + tmp_7767[0], tmp_7716[1] + tmp_7767[1], tmp_7716[2] + tmp_7767[2]];
    signal tmp_7769[3] <== CMul()(challengeQ, tmp_7768);
    signal tmp_7770[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7771[3] <== [tmp_7770[0] + evals[41][0], tmp_7770[1] + evals[41][1], tmp_7770[2] + evals[41][2]];
    signal tmp_7772[3] <== CMul()(evals[41], evals[54]);
    signal tmp_7773[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7774[3] <== CMul()(tmp_7773, evals[122]);
    signal tmp_7775[3] <== [tmp_7772[0] + tmp_7774[0], tmp_7772[1] + tmp_7774[1], tmp_7772[2] + tmp_7774[2]];
    signal tmp_7776[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7777[3] <== [67 * tmp_7776[0], 67 * tmp_7776[1], 67 * tmp_7776[2]];
    signal tmp_7778[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7779[3] <== [22 * tmp_7778[0], 22 * tmp_7778[1], 22 * tmp_7778[2]];
    signal tmp_7780[3] <== [tmp_7777[0] + tmp_7779[0], tmp_7777[1] + tmp_7779[1], tmp_7777[2] + tmp_7779[2]];
    signal tmp_7781[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7782[3] <== [13 * tmp_7781[0], 13 * tmp_7781[1], 13 * tmp_7781[2]];
    signal tmp_7783[3] <== [tmp_7780[0] + tmp_7782[0], tmp_7780[1] + tmp_7782[1], tmp_7780[2] + tmp_7782[2]];
    signal tmp_7784[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7785[3] <== [3 * tmp_7784[0], 3 * tmp_7784[1], 3 * tmp_7784[2]];
    signal tmp_7786[3] <== [tmp_7783[0] + tmp_7785[0], tmp_7783[1] + tmp_7785[1], tmp_7783[2] + tmp_7785[2]];
    signal tmp_7787[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7788[3] <== [tmp_7786[0] + tmp_7787[0], tmp_7786[1] + tmp_7787[1], tmp_7786[2] + tmp_7787[2]];
    signal tmp_7789[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7790[3] <== [tmp_7788[0] + tmp_7789[0], tmp_7788[1] + tmp_7789[1], tmp_7788[2] + tmp_7789[2]];
    signal tmp_7791[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7792[3] <== [51 * tmp_7791[0], 51 * tmp_7791[1], 51 * tmp_7791[2]];
    signal tmp_7793[3] <== [tmp_7790[0] + tmp_7792[0], tmp_7790[1] + tmp_7792[1], tmp_7790[2] + tmp_7792[2]];
    signal tmp_7794[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7795[3] <== [tmp_7793[0] + tmp_7794[0], tmp_7793[1] + tmp_7794[1], tmp_7793[2] + tmp_7794[2]];
    signal tmp_7796[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7797[3] <== [11 * tmp_7796[0], 11 * tmp_7796[1], 11 * tmp_7796[2]];
    signal tmp_7798[3] <== [tmp_7795[0] + tmp_7797[0], tmp_7795[1] + tmp_7797[1], tmp_7795[2] + tmp_7797[2]];
    signal tmp_7799[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7800[3] <== [17 * tmp_7799[0], 17 * tmp_7799[1], 17 * tmp_7799[2]];
    signal tmp_7801[3] <== [tmp_7798[0] + tmp_7800[0], tmp_7798[1] + tmp_7800[1], tmp_7798[2] + tmp_7800[2]];
    signal tmp_7802[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7803[3] <== [2 * tmp_7802[0], 2 * tmp_7802[1], 2 * tmp_7802[2]];
    signal tmp_7804[3] <== [tmp_7801[0] + tmp_7803[0], tmp_7801[1] + tmp_7803[1], tmp_7801[2] + tmp_7803[2]];
    signal tmp_7805[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7806[3] <== [tmp_7804[0] + tmp_7805[0], tmp_7804[1] + tmp_7805[1], tmp_7804[2] + tmp_7805[2]];
    signal tmp_7807[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7808[3] <== [101 * tmp_7807[0], 101 * tmp_7807[1], 101 * tmp_7807[2]];
    signal tmp_7809[3] <== [tmp_7806[0] + tmp_7808[0], tmp_7806[1] + tmp_7808[1], tmp_7806[2] + tmp_7808[2]];
    signal tmp_7810[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7811[3] <== [63 * tmp_7810[0], 63 * tmp_7810[1], 63 * tmp_7810[2]];
    signal tmp_7812[3] <== [tmp_7809[0] + tmp_7811[0], tmp_7809[1] + tmp_7811[1], tmp_7809[2] + tmp_7811[2]];
    signal tmp_7813[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7814[3] <== [15 * tmp_7813[0], 15 * tmp_7813[1], 15 * tmp_7813[2]];
    signal tmp_7815[3] <== [tmp_7812[0] + tmp_7814[0], tmp_7812[1] + tmp_7814[1], tmp_7812[2] + tmp_7814[2]];
    signal tmp_7816[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7817[3] <== [2 * tmp_7816[0], 2 * tmp_7816[1], 2 * tmp_7816[2]];
    signal tmp_7818[3] <== [tmp_7815[0] + tmp_7817[0], tmp_7815[1] + tmp_7817[1], tmp_7815[2] + tmp_7817[2]];
    signal tmp_7819[3] <== [tmp_7775[0] - tmp_7818[0], tmp_7775[1] - tmp_7818[1], tmp_7775[2] - tmp_7818[2]];
    signal tmp_7820[3] <== CMul()(tmp_7771, tmp_7819);
    signal tmp_7821[3] <== [tmp_7769[0] + tmp_7820[0], tmp_7769[1] + tmp_7820[1], tmp_7769[2] + tmp_7820[2]];
    signal tmp_7822[3] <== CMul()(challengeQ, tmp_7821);
    signal tmp_7823[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7824[3] <== [tmp_7823[0] + evals[41][0], tmp_7823[1] + evals[41][1], tmp_7823[2] + evals[41][2]];
    signal tmp_7825[3] <== CMul()(evals[41], evals[55]);
    signal tmp_7826[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7827[3] <== CMul()(tmp_7826, evals[123]);
    signal tmp_7828[3] <== [tmp_7825[0] + tmp_7827[0], tmp_7825[1] + tmp_7827[1], tmp_7825[2] + tmp_7827[2]];
    signal tmp_7829[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7830[3] <== [2 * tmp_7829[0], 2 * tmp_7829[1], 2 * tmp_7829[2]];
    signal tmp_7831[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7832[3] <== [67 * tmp_7831[0], 67 * tmp_7831[1], 67 * tmp_7831[2]];
    signal tmp_7833[3] <== [tmp_7830[0] + tmp_7832[0], tmp_7830[1] + tmp_7832[1], tmp_7830[2] + tmp_7832[2]];
    signal tmp_7834[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7835[3] <== [22 * tmp_7834[0], 22 * tmp_7834[1], 22 * tmp_7834[2]];
    signal tmp_7836[3] <== [tmp_7833[0] + tmp_7835[0], tmp_7833[1] + tmp_7835[1], tmp_7833[2] + tmp_7835[2]];
    signal tmp_7837[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7838[3] <== [13 * tmp_7837[0], 13 * tmp_7837[1], 13 * tmp_7837[2]];
    signal tmp_7839[3] <== [tmp_7836[0] + tmp_7838[0], tmp_7836[1] + tmp_7838[1], tmp_7836[2] + tmp_7838[2]];
    signal tmp_7840[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7841[3] <== [3 * tmp_7840[0], 3 * tmp_7840[1], 3 * tmp_7840[2]];
    signal tmp_7842[3] <== [tmp_7839[0] + tmp_7841[0], tmp_7839[1] + tmp_7841[1], tmp_7839[2] + tmp_7841[2]];
    signal tmp_7843[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7844[3] <== [tmp_7842[0] + tmp_7843[0], tmp_7842[1] + tmp_7843[1], tmp_7842[2] + tmp_7843[2]];
    signal tmp_7845[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7846[3] <== [tmp_7844[0] + tmp_7845[0], tmp_7844[1] + tmp_7845[1], tmp_7844[2] + tmp_7845[2]];
    signal tmp_7847[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7848[3] <== [51 * tmp_7847[0], 51 * tmp_7847[1], 51 * tmp_7847[2]];
    signal tmp_7849[3] <== [tmp_7846[0] + tmp_7848[0], tmp_7846[1] + tmp_7848[1], tmp_7846[2] + tmp_7848[2]];
    signal tmp_7850[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7851[3] <== [tmp_7849[0] + tmp_7850[0], tmp_7849[1] + tmp_7850[1], tmp_7849[2] + tmp_7850[2]];
    signal tmp_7852[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7853[3] <== [11 * tmp_7852[0], 11 * tmp_7852[1], 11 * tmp_7852[2]];
    signal tmp_7854[3] <== [tmp_7851[0] + tmp_7853[0], tmp_7851[1] + tmp_7853[1], tmp_7851[2] + tmp_7853[2]];
    signal tmp_7855[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7856[3] <== [17 * tmp_7855[0], 17 * tmp_7855[1], 17 * tmp_7855[2]];
    signal tmp_7857[3] <== [tmp_7854[0] + tmp_7856[0], tmp_7854[1] + tmp_7856[1], tmp_7854[2] + tmp_7856[2]];
    signal tmp_7858[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7859[3] <== [2 * tmp_7858[0], 2 * tmp_7858[1], 2 * tmp_7858[2]];
    signal tmp_7860[3] <== [tmp_7857[0] + tmp_7859[0], tmp_7857[1] + tmp_7859[1], tmp_7857[2] + tmp_7859[2]];
    signal tmp_7861[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7862[3] <== [tmp_7860[0] + tmp_7861[0], tmp_7860[1] + tmp_7861[1], tmp_7860[2] + tmp_7861[2]];
    signal tmp_7863[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7864[3] <== [101 * tmp_7863[0], 101 * tmp_7863[1], 101 * tmp_7863[2]];
    signal tmp_7865[3] <== [tmp_7862[0] + tmp_7864[0], tmp_7862[1] + tmp_7864[1], tmp_7862[2] + tmp_7864[2]];
    signal tmp_7866[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7867[3] <== [63 * tmp_7866[0], 63 * tmp_7866[1], 63 * tmp_7866[2]];
    signal tmp_7868[3] <== [tmp_7865[0] + tmp_7867[0], tmp_7865[1] + tmp_7867[1], tmp_7865[2] + tmp_7867[2]];
    signal tmp_7869[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7870[3] <== [15 * tmp_7869[0], 15 * tmp_7869[1], 15 * tmp_7869[2]];
    signal tmp_7871[3] <== [tmp_7868[0] + tmp_7870[0], tmp_7868[1] + tmp_7870[1], tmp_7868[2] + tmp_7870[2]];
    signal tmp_7872[3] <== [tmp_7828[0] - tmp_7871[0], tmp_7828[1] - tmp_7871[1], tmp_7828[2] - tmp_7871[2]];
    signal tmp_7873[3] <== CMul()(tmp_7824, tmp_7872);
    signal tmp_7874[3] <== [tmp_7822[0] + tmp_7873[0], tmp_7822[1] + tmp_7873[1], tmp_7822[2] + tmp_7873[2]];
    signal tmp_7875[3] <== CMul()(challengeQ, tmp_7874);
    signal tmp_7876[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7877[3] <== [tmp_7876[0] + evals[41][0], tmp_7876[1] + evals[41][1], tmp_7876[2] + evals[41][2]];
    signal tmp_7878[3] <== CMul()(evals[41], evals[56]);
    signal tmp_7879[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7880[3] <== CMul()(tmp_7879, evals[124]);
    signal tmp_7881[3] <== [tmp_7878[0] + tmp_7880[0], tmp_7878[1] + tmp_7880[1], tmp_7878[2] + tmp_7880[2]];
    signal tmp_7882[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7883[3] <== [15 * tmp_7882[0], 15 * tmp_7882[1], 15 * tmp_7882[2]];
    signal tmp_7884[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7885[3] <== [2 * tmp_7884[0], 2 * tmp_7884[1], 2 * tmp_7884[2]];
    signal tmp_7886[3] <== [tmp_7883[0] + tmp_7885[0], tmp_7883[1] + tmp_7885[1], tmp_7883[2] + tmp_7885[2]];
    signal tmp_7887[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7888[3] <== [67 * tmp_7887[0], 67 * tmp_7887[1], 67 * tmp_7887[2]];
    signal tmp_7889[3] <== [tmp_7886[0] + tmp_7888[0], tmp_7886[1] + tmp_7888[1], tmp_7886[2] + tmp_7888[2]];
    signal tmp_7890[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7891[3] <== [22 * tmp_7890[0], 22 * tmp_7890[1], 22 * tmp_7890[2]];
    signal tmp_7892[3] <== [tmp_7889[0] + tmp_7891[0], tmp_7889[1] + tmp_7891[1], tmp_7889[2] + tmp_7891[2]];
    signal tmp_7893[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7894[3] <== [13 * tmp_7893[0], 13 * tmp_7893[1], 13 * tmp_7893[2]];
    signal tmp_7895[3] <== [tmp_7892[0] + tmp_7894[0], tmp_7892[1] + tmp_7894[1], tmp_7892[2] + tmp_7894[2]];
    signal tmp_7896[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7897[3] <== [3 * tmp_7896[0], 3 * tmp_7896[1], 3 * tmp_7896[2]];
    signal tmp_7898[3] <== [tmp_7895[0] + tmp_7897[0], tmp_7895[1] + tmp_7897[1], tmp_7895[2] + tmp_7897[2]];
    signal tmp_7899[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7900[3] <== [tmp_7898[0] + tmp_7899[0], tmp_7898[1] + tmp_7899[1], tmp_7898[2] + tmp_7899[2]];
    signal tmp_7901[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7902[3] <== [tmp_7900[0] + tmp_7901[0], tmp_7900[1] + tmp_7901[1], tmp_7900[2] + tmp_7901[2]];
    signal tmp_7903[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7904[3] <== [51 * tmp_7903[0], 51 * tmp_7903[1], 51 * tmp_7903[2]];
    signal tmp_7905[3] <== [tmp_7902[0] + tmp_7904[0], tmp_7902[1] + tmp_7904[1], tmp_7902[2] + tmp_7904[2]];
    signal tmp_7906[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7907[3] <== [tmp_7905[0] + tmp_7906[0], tmp_7905[1] + tmp_7906[1], tmp_7905[2] + tmp_7906[2]];
    signal tmp_7908[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7909[3] <== [11 * tmp_7908[0], 11 * tmp_7908[1], 11 * tmp_7908[2]];
    signal tmp_7910[3] <== [tmp_7907[0] + tmp_7909[0], tmp_7907[1] + tmp_7909[1], tmp_7907[2] + tmp_7909[2]];
    signal tmp_7911[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7912[3] <== [17 * tmp_7911[0], 17 * tmp_7911[1], 17 * tmp_7911[2]];
    signal tmp_7913[3] <== [tmp_7910[0] + tmp_7912[0], tmp_7910[1] + tmp_7912[1], tmp_7910[2] + tmp_7912[2]];
    signal tmp_7914[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7915[3] <== [2 * tmp_7914[0], 2 * tmp_7914[1], 2 * tmp_7914[2]];
    signal tmp_7916[3] <== [tmp_7913[0] + tmp_7915[0], tmp_7913[1] + tmp_7915[1], tmp_7913[2] + tmp_7915[2]];
    signal tmp_7917[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7918[3] <== [tmp_7916[0] + tmp_7917[0], tmp_7916[1] + tmp_7917[1], tmp_7916[2] + tmp_7917[2]];
    signal tmp_7919[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7920[3] <== [101 * tmp_7919[0], 101 * tmp_7919[1], 101 * tmp_7919[2]];
    signal tmp_7921[3] <== [tmp_7918[0] + tmp_7920[0], tmp_7918[1] + tmp_7920[1], tmp_7918[2] + tmp_7920[2]];
    signal tmp_7922[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7923[3] <== [63 * tmp_7922[0], 63 * tmp_7922[1], 63 * tmp_7922[2]];
    signal tmp_7924[3] <== [tmp_7921[0] + tmp_7923[0], tmp_7921[1] + tmp_7923[1], tmp_7921[2] + tmp_7923[2]];
    signal tmp_7925[3] <== [tmp_7881[0] - tmp_7924[0], tmp_7881[1] - tmp_7924[1], tmp_7881[2] - tmp_7924[2]];
    signal tmp_7926[3] <== CMul()(tmp_7877, tmp_7925);
    signal tmp_7927[3] <== [tmp_7875[0] + tmp_7926[0], tmp_7875[1] + tmp_7926[1], tmp_7875[2] + tmp_7926[2]];
    signal tmp_7928[3] <== CMul()(challengeQ, tmp_7927);
    signal tmp_7929[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7930[3] <== [tmp_7929[0] + evals[41][0], tmp_7929[1] + evals[41][1], tmp_7929[2] + evals[41][2]];
    signal tmp_7931[3] <== CMul()(evals[41], evals[57]);
    signal tmp_7932[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7933[3] <== CMul()(tmp_7932, evals[125]);
    signal tmp_7934[3] <== [tmp_7931[0] + tmp_7933[0], tmp_7931[1] + tmp_7933[1], tmp_7931[2] + tmp_7933[2]];
    signal tmp_7935[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7936[3] <== [63 * tmp_7935[0], 63 * tmp_7935[1], 63 * tmp_7935[2]];
    signal tmp_7937[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7938[3] <== [15 * tmp_7937[0], 15 * tmp_7937[1], 15 * tmp_7937[2]];
    signal tmp_7939[3] <== [tmp_7936[0] + tmp_7938[0], tmp_7936[1] + tmp_7938[1], tmp_7936[2] + tmp_7938[2]];
    signal tmp_7940[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7941[3] <== [2 * tmp_7940[0], 2 * tmp_7940[1], 2 * tmp_7940[2]];
    signal tmp_7942[3] <== [tmp_7939[0] + tmp_7941[0], tmp_7939[1] + tmp_7941[1], tmp_7939[2] + tmp_7941[2]];
    signal tmp_7943[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7944[3] <== [67 * tmp_7943[0], 67 * tmp_7943[1], 67 * tmp_7943[2]];
    signal tmp_7945[3] <== [tmp_7942[0] + tmp_7944[0], tmp_7942[1] + tmp_7944[1], tmp_7942[2] + tmp_7944[2]];
    signal tmp_7946[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_7947[3] <== [22 * tmp_7946[0], 22 * tmp_7946[1], 22 * tmp_7946[2]];
    signal tmp_7948[3] <== [tmp_7945[0] + tmp_7947[0], tmp_7945[1] + tmp_7947[1], tmp_7945[2] + tmp_7947[2]];
    signal tmp_7949[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_7950[3] <== [13 * tmp_7949[0], 13 * tmp_7949[1], 13 * tmp_7949[2]];
    signal tmp_7951[3] <== [tmp_7948[0] + tmp_7950[0], tmp_7948[1] + tmp_7950[1], tmp_7948[2] + tmp_7950[2]];
    signal tmp_7952[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_7953[3] <== [3 * tmp_7952[0], 3 * tmp_7952[1], 3 * tmp_7952[2]];
    signal tmp_7954[3] <== [tmp_7951[0] + tmp_7953[0], tmp_7951[1] + tmp_7953[1], tmp_7951[2] + tmp_7953[2]];
    signal tmp_7955[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_7956[3] <== [tmp_7954[0] + tmp_7955[0], tmp_7954[1] + tmp_7955[1], tmp_7954[2] + tmp_7955[2]];
    signal tmp_7957[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_7958[3] <== [tmp_7956[0] + tmp_7957[0], tmp_7956[1] + tmp_7957[1], tmp_7956[2] + tmp_7957[2]];
    signal tmp_7959[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_7960[3] <== [51 * tmp_7959[0], 51 * tmp_7959[1], 51 * tmp_7959[2]];
    signal tmp_7961[3] <== [tmp_7958[0] + tmp_7960[0], tmp_7958[1] + tmp_7960[1], tmp_7958[2] + tmp_7960[2]];
    signal tmp_7962[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_7963[3] <== [tmp_7961[0] + tmp_7962[0], tmp_7961[1] + tmp_7962[1], tmp_7961[2] + tmp_7962[2]];
    signal tmp_7964[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_7965[3] <== [11 * tmp_7964[0], 11 * tmp_7964[1], 11 * tmp_7964[2]];
    signal tmp_7966[3] <== [tmp_7963[0] + tmp_7965[0], tmp_7963[1] + tmp_7965[1], tmp_7963[2] + tmp_7965[2]];
    signal tmp_7967[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_7968[3] <== [17 * tmp_7967[0], 17 * tmp_7967[1], 17 * tmp_7967[2]];
    signal tmp_7969[3] <== [tmp_7966[0] + tmp_7968[0], tmp_7966[1] + tmp_7968[1], tmp_7966[2] + tmp_7968[2]];
    signal tmp_7970[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_7971[3] <== [2 * tmp_7970[0], 2 * tmp_7970[1], 2 * tmp_7970[2]];
    signal tmp_7972[3] <== [tmp_7969[0] + tmp_7971[0], tmp_7969[1] + tmp_7971[1], tmp_7969[2] + tmp_7971[2]];
    signal tmp_7973[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_7974[3] <== [tmp_7972[0] + tmp_7973[0], tmp_7972[1] + tmp_7973[1], tmp_7972[2] + tmp_7973[2]];
    signal tmp_7975[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_7976[3] <== [101 * tmp_7975[0], 101 * tmp_7975[1], 101 * tmp_7975[2]];
    signal tmp_7977[3] <== [tmp_7974[0] + tmp_7976[0], tmp_7974[1] + tmp_7976[1], tmp_7974[2] + tmp_7976[2]];
    signal tmp_7978[3] <== [tmp_7934[0] - tmp_7977[0], tmp_7934[1] - tmp_7977[1], tmp_7934[2] - tmp_7977[2]];
    signal tmp_7979[3] <== CMul()(tmp_7930, tmp_7978);
    signal tmp_7980[3] <== [tmp_7928[0] + tmp_7979[0], tmp_7928[1] + tmp_7979[1], tmp_7928[2] + tmp_7979[2]];
    signal tmp_7981[3] <== CMul()(challengeQ, tmp_7980);
    signal tmp_7982[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_7983[3] <== [tmp_7982[0] + evals[41][0], tmp_7982[1] + evals[41][1], tmp_7982[2] + evals[41][2]];
    signal tmp_7984[3] <== CMul()(evals[41], evals[58]);
    signal tmp_7985[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_7986[3] <== CMul()(tmp_7985, evals[126]);
    signal tmp_7987[3] <== [tmp_7984[0] + tmp_7986[0], tmp_7984[1] + tmp_7986[1], tmp_7984[2] + tmp_7986[2]];
    signal tmp_7988[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_7989[3] <== [101 * tmp_7988[0], 101 * tmp_7988[1], 101 * tmp_7988[2]];
    signal tmp_7990[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_7991[3] <== [63 * tmp_7990[0], 63 * tmp_7990[1], 63 * tmp_7990[2]];
    signal tmp_7992[3] <== [tmp_7989[0] + tmp_7991[0], tmp_7989[1] + tmp_7991[1], tmp_7989[2] + tmp_7991[2]];
    signal tmp_7993[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_7994[3] <== [15 * tmp_7993[0], 15 * tmp_7993[1], 15 * tmp_7993[2]];
    signal tmp_7995[3] <== [tmp_7992[0] + tmp_7994[0], tmp_7992[1] + tmp_7994[1], tmp_7992[2] + tmp_7994[2]];
    signal tmp_7996[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_7997[3] <== [2 * tmp_7996[0], 2 * tmp_7996[1], 2 * tmp_7996[2]];
    signal tmp_7998[3] <== [tmp_7995[0] + tmp_7997[0], tmp_7995[1] + tmp_7997[1], tmp_7995[2] + tmp_7997[2]];
    signal tmp_7999[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8000[3] <== [67 * tmp_7999[0], 67 * tmp_7999[1], 67 * tmp_7999[2]];
    signal tmp_8001[3] <== [tmp_7998[0] + tmp_8000[0], tmp_7998[1] + tmp_8000[1], tmp_7998[2] + tmp_8000[2]];
    signal tmp_8002[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8003[3] <== [22 * tmp_8002[0], 22 * tmp_8002[1], 22 * tmp_8002[2]];
    signal tmp_8004[3] <== [tmp_8001[0] + tmp_8003[0], tmp_8001[1] + tmp_8003[1], tmp_8001[2] + tmp_8003[2]];
    signal tmp_8005[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8006[3] <== [13 * tmp_8005[0], 13 * tmp_8005[1], 13 * tmp_8005[2]];
    signal tmp_8007[3] <== [tmp_8004[0] + tmp_8006[0], tmp_8004[1] + tmp_8006[1], tmp_8004[2] + tmp_8006[2]];
    signal tmp_8008[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8009[3] <== [3 * tmp_8008[0], 3 * tmp_8008[1], 3 * tmp_8008[2]];
    signal tmp_8010[3] <== [tmp_8007[0] + tmp_8009[0], tmp_8007[1] + tmp_8009[1], tmp_8007[2] + tmp_8009[2]];
    signal tmp_8011[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8012[3] <== [tmp_8010[0] + tmp_8011[0], tmp_8010[1] + tmp_8011[1], tmp_8010[2] + tmp_8011[2]];
    signal tmp_8013[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8014[3] <== [tmp_8012[0] + tmp_8013[0], tmp_8012[1] + tmp_8013[1], tmp_8012[2] + tmp_8013[2]];
    signal tmp_8015[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8016[3] <== [51 * tmp_8015[0], 51 * tmp_8015[1], 51 * tmp_8015[2]];
    signal tmp_8017[3] <== [tmp_8014[0] + tmp_8016[0], tmp_8014[1] + tmp_8016[1], tmp_8014[2] + tmp_8016[2]];
    signal tmp_8018[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8019[3] <== [tmp_8017[0] + tmp_8018[0], tmp_8017[1] + tmp_8018[1], tmp_8017[2] + tmp_8018[2]];
    signal tmp_8020[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8021[3] <== [11 * tmp_8020[0], 11 * tmp_8020[1], 11 * tmp_8020[2]];
    signal tmp_8022[3] <== [tmp_8019[0] + tmp_8021[0], tmp_8019[1] + tmp_8021[1], tmp_8019[2] + tmp_8021[2]];
    signal tmp_8023[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8024[3] <== [17 * tmp_8023[0], 17 * tmp_8023[1], 17 * tmp_8023[2]];
    signal tmp_8025[3] <== [tmp_8022[0] + tmp_8024[0], tmp_8022[1] + tmp_8024[1], tmp_8022[2] + tmp_8024[2]];
    signal tmp_8026[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8027[3] <== [2 * tmp_8026[0], 2 * tmp_8026[1], 2 * tmp_8026[2]];
    signal tmp_8028[3] <== [tmp_8025[0] + tmp_8027[0], tmp_8025[1] + tmp_8027[1], tmp_8025[2] + tmp_8027[2]];
    signal tmp_8029[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8030[3] <== [tmp_8028[0] + tmp_8029[0], tmp_8028[1] + tmp_8029[1], tmp_8028[2] + tmp_8029[2]];
    signal tmp_8031[3] <== [tmp_7987[0] - tmp_8030[0], tmp_7987[1] - tmp_8030[1], tmp_7987[2] - tmp_8030[2]];
    signal tmp_8032[3] <== CMul()(tmp_7983, tmp_8031);
    signal tmp_8033[3] <== [tmp_7981[0] + tmp_8032[0], tmp_7981[1] + tmp_8032[1], tmp_7981[2] + tmp_8032[2]];
    tmp_8034 <== CMul()(challengeQ, tmp_8033);
    signal tmp_8035[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    tmp_8036 <== [tmp_8035[0] + evals[41][0], tmp_8035[1] + evals[41][1], tmp_8035[2] + evals[41][2]];
    signal tmp_8037[3] <== CMul()(evals[41], evals[59]);
    signal tmp_8038[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8039[3] <== CMul()(tmp_8038, evals[127]);
    tmp_8040 <== [tmp_8037[0] + tmp_8039[0], tmp_8037[1] + tmp_8039[1], tmp_8037[2] + tmp_8039[2]];
    signal tmp_8041[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8042[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8043[3] <== [101 * tmp_8042[0], 101 * tmp_8042[1], 101 * tmp_8042[2]];
    signal tmp_8044[3] <== [tmp_8041[0] + tmp_8043[0], tmp_8041[1] + tmp_8043[1], tmp_8041[2] + tmp_8043[2]];
    signal tmp_8045[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8046[3] <== [63 * tmp_8045[0], 63 * tmp_8045[1], 63 * tmp_8045[2]];
    signal tmp_8047[3] <== [tmp_8044[0] + tmp_8046[0], tmp_8044[1] + tmp_8046[1], tmp_8044[2] + tmp_8046[2]];
    signal tmp_8048[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8049[3] <== [15 * tmp_8048[0], 15 * tmp_8048[1], 15 * tmp_8048[2]];
    signal tmp_8050[3] <== [tmp_8047[0] + tmp_8049[0], tmp_8047[1] + tmp_8049[1], tmp_8047[2] + tmp_8049[2]];
    signal tmp_8051[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8052[3] <== [2 * tmp_8051[0], 2 * tmp_8051[1], 2 * tmp_8051[2]];
    signal tmp_8053[3] <== [tmp_8050[0] + tmp_8052[0], tmp_8050[1] + tmp_8052[1], tmp_8050[2] + tmp_8052[2]];
    signal tmp_8054[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8055[3] <== [67 * tmp_8054[0], 67 * tmp_8054[1], 67 * tmp_8054[2]];
    signal tmp_8056[3] <== [tmp_8053[0] + tmp_8055[0], tmp_8053[1] + tmp_8055[1], tmp_8053[2] + tmp_8055[2]];
    signal tmp_8057[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8058[3] <== [22 * tmp_8057[0], 22 * tmp_8057[1], 22 * tmp_8057[2]];
    signal tmp_8059[3] <== [tmp_8056[0] + tmp_8058[0], tmp_8056[1] + tmp_8058[1], tmp_8056[2] + tmp_8058[2]];
    signal tmp_8060[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8061[3] <== [13 * tmp_8060[0], 13 * tmp_8060[1], 13 * tmp_8060[2]];
    tmp_8062 <== [tmp_8059[0] + tmp_8061[0], tmp_8059[1] + tmp_8061[1], tmp_8059[2] + tmp_8061[2]];
    tmp_8063 <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
}

template VerifyEvaluationsChunks2() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_6066[3];
    signal input tmp_6405[3];
    signal input tmp_7423[3];
    signal input tmp_7428[3];
    signal input tmp_7433[3];
    signal input tmp_7438[3];
    signal input tmp_7444[3];
    signal input tmp_7449[3];
    signal input tmp_7456[3];
    signal input tmp_7461[3];
    signal input tmp_7467[3];
    signal input tmp_7472[3];
    signal input tmp_7479[3];
    signal input tmp_7484[3];
    signal input tmp_7491[3];
    signal input tmp_7496[3];
    signal input tmp_7503[3];
    signal input tmp_7508[3];
    signal input tmp_7514[3];
    signal input tmp_7519[3];
    signal input tmp_7526[3];
    signal input tmp_7531[3];
    signal input tmp_7538[3];
    signal input tmp_7543[3];
    signal input tmp_7550[3];
    signal input tmp_7555[3];
    signal input tmp_7562[3];
    signal input tmp_7567[3];
    signal input tmp_7574[3];
    signal input tmp_7579[3];
    signal input tmp_7586[3];
    signal input tmp_7591[3];
    signal input tmp_7598[3];
    signal input tmp_7603[3];
    signal input tmp_8034[3];
    signal input tmp_8036[3];
    signal input tmp_8040[3];
    signal input tmp_8062[3];
    signal input tmp_8063[3];

    signal output tmp_9013[3];
    signal output tmp_9062[3];
    signal tmp_8064[3] <== [3 * tmp_8063[0], 3 * tmp_8063[1], 3 * tmp_8063[2]];
    signal tmp_8065[3] <== [tmp_8062[0] + tmp_8064[0], tmp_8062[1] + tmp_8064[1], tmp_8062[2] + tmp_8064[2]];
    signal tmp_8066[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8067[3] <== [tmp_8065[0] + tmp_8066[0], tmp_8065[1] + tmp_8066[1], tmp_8065[2] + tmp_8066[2]];
    signal tmp_8068[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8069[3] <== [tmp_8067[0] + tmp_8068[0], tmp_8067[1] + tmp_8068[1], tmp_8067[2] + tmp_8068[2]];
    signal tmp_8070[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8071[3] <== [51 * tmp_8070[0], 51 * tmp_8070[1], 51 * tmp_8070[2]];
    signal tmp_8072[3] <== [tmp_8069[0] + tmp_8071[0], tmp_8069[1] + tmp_8071[1], tmp_8069[2] + tmp_8071[2]];
    signal tmp_8073[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8074[3] <== [tmp_8072[0] + tmp_8073[0], tmp_8072[1] + tmp_8073[1], tmp_8072[2] + tmp_8073[2]];
    signal tmp_8075[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8076[3] <== [11 * tmp_8075[0], 11 * tmp_8075[1], 11 * tmp_8075[2]];
    signal tmp_8077[3] <== [tmp_8074[0] + tmp_8076[0], tmp_8074[1] + tmp_8076[1], tmp_8074[2] + tmp_8076[2]];
    signal tmp_8078[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8079[3] <== [17 * tmp_8078[0], 17 * tmp_8078[1], 17 * tmp_8078[2]];
    signal tmp_8080[3] <== [tmp_8077[0] + tmp_8079[0], tmp_8077[1] + tmp_8079[1], tmp_8077[2] + tmp_8079[2]];
    signal tmp_8081[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8082[3] <== [2 * tmp_8081[0], 2 * tmp_8081[1], 2 * tmp_8081[2]];
    signal tmp_8083[3] <== [tmp_8080[0] + tmp_8082[0], tmp_8080[1] + tmp_8082[1], tmp_8080[2] + tmp_8082[2]];
    signal tmp_8084[3] <== [tmp_8040[0] - tmp_8083[0], tmp_8040[1] - tmp_8083[1], tmp_8040[2] - tmp_8083[2]];
    signal tmp_8085[3] <== CMul()(tmp_8036, tmp_8084);
    signal tmp_8086[3] <== [tmp_8034[0] + tmp_8085[0], tmp_8034[1] + tmp_8085[1], tmp_8034[2] + tmp_8085[2]];
    signal tmp_8087[3] <== CMul()(challengeQ, tmp_8086);
    signal tmp_8088[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8089[3] <== [tmp_8088[0] + evals[41][0], tmp_8088[1] + evals[41][1], tmp_8088[2] + evals[41][2]];
    signal tmp_8090[3] <== CMul()(evals[41], evals[60]);
    signal tmp_8091[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8092[3] <== CMul()(tmp_8091, evals[128]);
    signal tmp_8093[3] <== [tmp_8090[0] + tmp_8092[0], tmp_8090[1] + tmp_8092[1], tmp_8090[2] + tmp_8092[2]];
    signal tmp_8094[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8095[3] <== [2 * tmp_8094[0], 2 * tmp_8094[1], 2 * tmp_8094[2]];
    signal tmp_8096[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8097[3] <== [tmp_8095[0] + tmp_8096[0], tmp_8095[1] + tmp_8096[1], tmp_8095[2] + tmp_8096[2]];
    signal tmp_8098[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8099[3] <== [101 * tmp_8098[0], 101 * tmp_8098[1], 101 * tmp_8098[2]];
    signal tmp_8100[3] <== [tmp_8097[0] + tmp_8099[0], tmp_8097[1] + tmp_8099[1], tmp_8097[2] + tmp_8099[2]];
    signal tmp_8101[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8102[3] <== [63 * tmp_8101[0], 63 * tmp_8101[1], 63 * tmp_8101[2]];
    signal tmp_8103[3] <== [tmp_8100[0] + tmp_8102[0], tmp_8100[1] + tmp_8102[1], tmp_8100[2] + tmp_8102[2]];
    signal tmp_8104[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8105[3] <== [15 * tmp_8104[0], 15 * tmp_8104[1], 15 * tmp_8104[2]];
    signal tmp_8106[3] <== [tmp_8103[0] + tmp_8105[0], tmp_8103[1] + tmp_8105[1], tmp_8103[2] + tmp_8105[2]];
    signal tmp_8107[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8108[3] <== [2 * tmp_8107[0], 2 * tmp_8107[1], 2 * tmp_8107[2]];
    signal tmp_8109[3] <== [tmp_8106[0] + tmp_8108[0], tmp_8106[1] + tmp_8108[1], tmp_8106[2] + tmp_8108[2]];
    signal tmp_8110[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8111[3] <== [67 * tmp_8110[0], 67 * tmp_8110[1], 67 * tmp_8110[2]];
    signal tmp_8112[3] <== [tmp_8109[0] + tmp_8111[0], tmp_8109[1] + tmp_8111[1], tmp_8109[2] + tmp_8111[2]];
    signal tmp_8113[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8114[3] <== [22 * tmp_8113[0], 22 * tmp_8113[1], 22 * tmp_8113[2]];
    signal tmp_8115[3] <== [tmp_8112[0] + tmp_8114[0], tmp_8112[1] + tmp_8114[1], tmp_8112[2] + tmp_8114[2]];
    signal tmp_8116[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8117[3] <== [13 * tmp_8116[0], 13 * tmp_8116[1], 13 * tmp_8116[2]];
    signal tmp_8118[3] <== [tmp_8115[0] + tmp_8117[0], tmp_8115[1] + tmp_8117[1], tmp_8115[2] + tmp_8117[2]];
    signal tmp_8119[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8120[3] <== [3 * tmp_8119[0], 3 * tmp_8119[1], 3 * tmp_8119[2]];
    signal tmp_8121[3] <== [tmp_8118[0] + tmp_8120[0], tmp_8118[1] + tmp_8120[1], tmp_8118[2] + tmp_8120[2]];
    signal tmp_8122[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8123[3] <== [tmp_8121[0] + tmp_8122[0], tmp_8121[1] + tmp_8122[1], tmp_8121[2] + tmp_8122[2]];
    signal tmp_8124[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8125[3] <== [tmp_8123[0] + tmp_8124[0], tmp_8123[1] + tmp_8124[1], tmp_8123[2] + tmp_8124[2]];
    signal tmp_8126[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8127[3] <== [51 * tmp_8126[0], 51 * tmp_8126[1], 51 * tmp_8126[2]];
    signal tmp_8128[3] <== [tmp_8125[0] + tmp_8127[0], tmp_8125[1] + tmp_8127[1], tmp_8125[2] + tmp_8127[2]];
    signal tmp_8129[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8130[3] <== [tmp_8128[0] + tmp_8129[0], tmp_8128[1] + tmp_8129[1], tmp_8128[2] + tmp_8129[2]];
    signal tmp_8131[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8132[3] <== [11 * tmp_8131[0], 11 * tmp_8131[1], 11 * tmp_8131[2]];
    signal tmp_8133[3] <== [tmp_8130[0] + tmp_8132[0], tmp_8130[1] + tmp_8132[1], tmp_8130[2] + tmp_8132[2]];
    signal tmp_8134[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8135[3] <== [17 * tmp_8134[0], 17 * tmp_8134[1], 17 * tmp_8134[2]];
    signal tmp_8136[3] <== [tmp_8133[0] + tmp_8135[0], tmp_8133[1] + tmp_8135[1], tmp_8133[2] + tmp_8135[2]];
    signal tmp_8137[3] <== [tmp_8093[0] - tmp_8136[0], tmp_8093[1] - tmp_8136[1], tmp_8093[2] - tmp_8136[2]];
    signal tmp_8138[3] <== CMul()(tmp_8089, tmp_8137);
    signal tmp_8139[3] <== [tmp_8087[0] + tmp_8138[0], tmp_8087[1] + tmp_8138[1], tmp_8087[2] + tmp_8138[2]];
    signal tmp_8140[3] <== CMul()(challengeQ, tmp_8139);
    signal tmp_8141[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8142[3] <== [tmp_8141[0] + evals[41][0], tmp_8141[1] + evals[41][1], tmp_8141[2] + evals[41][2]];
    signal tmp_8143[3] <== CMul()(evals[41], evals[61]);
    signal tmp_8144[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8145[3] <== CMul()(tmp_8144, evals[129]);
    signal tmp_8146[3] <== [tmp_8143[0] + tmp_8145[0], tmp_8143[1] + tmp_8145[1], tmp_8143[2] + tmp_8145[2]];
    signal tmp_8147[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8148[3] <== [17 * tmp_8147[0], 17 * tmp_8147[1], 17 * tmp_8147[2]];
    signal tmp_8149[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8150[3] <== [2 * tmp_8149[0], 2 * tmp_8149[1], 2 * tmp_8149[2]];
    signal tmp_8151[3] <== [tmp_8148[0] + tmp_8150[0], tmp_8148[1] + tmp_8150[1], tmp_8148[2] + tmp_8150[2]];
    signal tmp_8152[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8153[3] <== [tmp_8151[0] + tmp_8152[0], tmp_8151[1] + tmp_8152[1], tmp_8151[2] + tmp_8152[2]];
    signal tmp_8154[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8155[3] <== [101 * tmp_8154[0], 101 * tmp_8154[1], 101 * tmp_8154[2]];
    signal tmp_8156[3] <== [tmp_8153[0] + tmp_8155[0], tmp_8153[1] + tmp_8155[1], tmp_8153[2] + tmp_8155[2]];
    signal tmp_8157[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8158[3] <== [63 * tmp_8157[0], 63 * tmp_8157[1], 63 * tmp_8157[2]];
    signal tmp_8159[3] <== [tmp_8156[0] + tmp_8158[0], tmp_8156[1] + tmp_8158[1], tmp_8156[2] + tmp_8158[2]];
    signal tmp_8160[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8161[3] <== [15 * tmp_8160[0], 15 * tmp_8160[1], 15 * tmp_8160[2]];
    signal tmp_8162[3] <== [tmp_8159[0] + tmp_8161[0], tmp_8159[1] + tmp_8161[1], tmp_8159[2] + tmp_8161[2]];
    signal tmp_8163[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8164[3] <== [2 * tmp_8163[0], 2 * tmp_8163[1], 2 * tmp_8163[2]];
    signal tmp_8165[3] <== [tmp_8162[0] + tmp_8164[0], tmp_8162[1] + tmp_8164[1], tmp_8162[2] + tmp_8164[2]];
    signal tmp_8166[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8167[3] <== [67 * tmp_8166[0], 67 * tmp_8166[1], 67 * tmp_8166[2]];
    signal tmp_8168[3] <== [tmp_8165[0] + tmp_8167[0], tmp_8165[1] + tmp_8167[1], tmp_8165[2] + tmp_8167[2]];
    signal tmp_8169[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8170[3] <== [22 * tmp_8169[0], 22 * tmp_8169[1], 22 * tmp_8169[2]];
    signal tmp_8171[3] <== [tmp_8168[0] + tmp_8170[0], tmp_8168[1] + tmp_8170[1], tmp_8168[2] + tmp_8170[2]];
    signal tmp_8172[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8173[3] <== [13 * tmp_8172[0], 13 * tmp_8172[1], 13 * tmp_8172[2]];
    signal tmp_8174[3] <== [tmp_8171[0] + tmp_8173[0], tmp_8171[1] + tmp_8173[1], tmp_8171[2] + tmp_8173[2]];
    signal tmp_8175[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8176[3] <== [3 * tmp_8175[0], 3 * tmp_8175[1], 3 * tmp_8175[2]];
    signal tmp_8177[3] <== [tmp_8174[0] + tmp_8176[0], tmp_8174[1] + tmp_8176[1], tmp_8174[2] + tmp_8176[2]];
    signal tmp_8178[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8179[3] <== [tmp_8177[0] + tmp_8178[0], tmp_8177[1] + tmp_8178[1], tmp_8177[2] + tmp_8178[2]];
    signal tmp_8180[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8181[3] <== [tmp_8179[0] + tmp_8180[0], tmp_8179[1] + tmp_8180[1], tmp_8179[2] + tmp_8180[2]];
    signal tmp_8182[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8183[3] <== [51 * tmp_8182[0], 51 * tmp_8182[1], 51 * tmp_8182[2]];
    signal tmp_8184[3] <== [tmp_8181[0] + tmp_8183[0], tmp_8181[1] + tmp_8183[1], tmp_8181[2] + tmp_8183[2]];
    signal tmp_8185[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8186[3] <== [tmp_8184[0] + tmp_8185[0], tmp_8184[1] + tmp_8185[1], tmp_8184[2] + tmp_8185[2]];
    signal tmp_8187[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8188[3] <== [11 * tmp_8187[0], 11 * tmp_8187[1], 11 * tmp_8187[2]];
    signal tmp_8189[3] <== [tmp_8186[0] + tmp_8188[0], tmp_8186[1] + tmp_8188[1], tmp_8186[2] + tmp_8188[2]];
    signal tmp_8190[3] <== [tmp_8146[0] - tmp_8189[0], tmp_8146[1] - tmp_8189[1], tmp_8146[2] - tmp_8189[2]];
    signal tmp_8191[3] <== CMul()(tmp_8142, tmp_8190);
    signal tmp_8192[3] <== [tmp_8140[0] + tmp_8191[0], tmp_8140[1] + tmp_8191[1], tmp_8140[2] + tmp_8191[2]];
    signal tmp_8193[3] <== CMul()(challengeQ, tmp_8192);
    signal tmp_8194[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8195[3] <== [tmp_8194[0] + evals[41][0], tmp_8194[1] + evals[41][1], tmp_8194[2] + evals[41][2]];
    signal tmp_8196[3] <== CMul()(evals[41], evals[62]);
    signal tmp_8197[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8198[3] <== CMul()(tmp_8197, evals[130]);
    signal tmp_8199[3] <== [tmp_8196[0] + tmp_8198[0], tmp_8196[1] + tmp_8198[1], tmp_8196[2] + tmp_8198[2]];
    signal tmp_8200[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8201[3] <== [11 * tmp_8200[0], 11 * tmp_8200[1], 11 * tmp_8200[2]];
    signal tmp_8202[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8203[3] <== [17 * tmp_8202[0], 17 * tmp_8202[1], 17 * tmp_8202[2]];
    signal tmp_8204[3] <== [tmp_8201[0] + tmp_8203[0], tmp_8201[1] + tmp_8203[1], tmp_8201[2] + tmp_8203[2]];
    signal tmp_8205[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8206[3] <== [2 * tmp_8205[0], 2 * tmp_8205[1], 2 * tmp_8205[2]];
    signal tmp_8207[3] <== [tmp_8204[0] + tmp_8206[0], tmp_8204[1] + tmp_8206[1], tmp_8204[2] + tmp_8206[2]];
    signal tmp_8208[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8209[3] <== [tmp_8207[0] + tmp_8208[0], tmp_8207[1] + tmp_8208[1], tmp_8207[2] + tmp_8208[2]];
    signal tmp_8210[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8211[3] <== [101 * tmp_8210[0], 101 * tmp_8210[1], 101 * tmp_8210[2]];
    signal tmp_8212[3] <== [tmp_8209[0] + tmp_8211[0], tmp_8209[1] + tmp_8211[1], tmp_8209[2] + tmp_8211[2]];
    signal tmp_8213[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8214[3] <== [63 * tmp_8213[0], 63 * tmp_8213[1], 63 * tmp_8213[2]];
    signal tmp_8215[3] <== [tmp_8212[0] + tmp_8214[0], tmp_8212[1] + tmp_8214[1], tmp_8212[2] + tmp_8214[2]];
    signal tmp_8216[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8217[3] <== [15 * tmp_8216[0], 15 * tmp_8216[1], 15 * tmp_8216[2]];
    signal tmp_8218[3] <== [tmp_8215[0] + tmp_8217[0], tmp_8215[1] + tmp_8217[1], tmp_8215[2] + tmp_8217[2]];
    signal tmp_8219[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8220[3] <== [2 * tmp_8219[0], 2 * tmp_8219[1], 2 * tmp_8219[2]];
    signal tmp_8221[3] <== [tmp_8218[0] + tmp_8220[0], tmp_8218[1] + tmp_8220[1], tmp_8218[2] + tmp_8220[2]];
    signal tmp_8222[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8223[3] <== [67 * tmp_8222[0], 67 * tmp_8222[1], 67 * tmp_8222[2]];
    signal tmp_8224[3] <== [tmp_8221[0] + tmp_8223[0], tmp_8221[1] + tmp_8223[1], tmp_8221[2] + tmp_8223[2]];
    signal tmp_8225[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8226[3] <== [22 * tmp_8225[0], 22 * tmp_8225[1], 22 * tmp_8225[2]];
    signal tmp_8227[3] <== [tmp_8224[0] + tmp_8226[0], tmp_8224[1] + tmp_8226[1], tmp_8224[2] + tmp_8226[2]];
    signal tmp_8228[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8229[3] <== [13 * tmp_8228[0], 13 * tmp_8228[1], 13 * tmp_8228[2]];
    signal tmp_8230[3] <== [tmp_8227[0] + tmp_8229[0], tmp_8227[1] + tmp_8229[1], tmp_8227[2] + tmp_8229[2]];
    signal tmp_8231[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8232[3] <== [3 * tmp_8231[0], 3 * tmp_8231[1], 3 * tmp_8231[2]];
    signal tmp_8233[3] <== [tmp_8230[0] + tmp_8232[0], tmp_8230[1] + tmp_8232[1], tmp_8230[2] + tmp_8232[2]];
    signal tmp_8234[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8235[3] <== [tmp_8233[0] + tmp_8234[0], tmp_8233[1] + tmp_8234[1], tmp_8233[2] + tmp_8234[2]];
    signal tmp_8236[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8237[3] <== [tmp_8235[0] + tmp_8236[0], tmp_8235[1] + tmp_8236[1], tmp_8235[2] + tmp_8236[2]];
    signal tmp_8238[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8239[3] <== [51 * tmp_8238[0], 51 * tmp_8238[1], 51 * tmp_8238[2]];
    signal tmp_8240[3] <== [tmp_8237[0] + tmp_8239[0], tmp_8237[1] + tmp_8239[1], tmp_8237[2] + tmp_8239[2]];
    signal tmp_8241[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8242[3] <== [tmp_8240[0] + tmp_8241[0], tmp_8240[1] + tmp_8241[1], tmp_8240[2] + tmp_8241[2]];
    signal tmp_8243[3] <== [tmp_8199[0] - tmp_8242[0], tmp_8199[1] - tmp_8242[1], tmp_8199[2] - tmp_8242[2]];
    signal tmp_8244[3] <== CMul()(tmp_8195, tmp_8243);
    signal tmp_8245[3] <== [tmp_8193[0] + tmp_8244[0], tmp_8193[1] + tmp_8244[1], tmp_8193[2] + tmp_8244[2]];
    signal tmp_8246[3] <== CMul()(challengeQ, tmp_8245);
    signal tmp_8247[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8248[3] <== [tmp_8247[0] + evals[41][0], tmp_8247[1] + evals[41][1], tmp_8247[2] + evals[41][2]];
    signal tmp_8249[3] <== CMul()(evals[41], evals[63]);
    signal tmp_8250[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8251[3] <== CMul()(tmp_8250, evals[131]);
    signal tmp_8252[3] <== [tmp_8249[0] + tmp_8251[0], tmp_8249[1] + tmp_8251[1], tmp_8249[2] + tmp_8251[2]];
    signal tmp_8253[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8254[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8255[3] <== [11 * tmp_8254[0], 11 * tmp_8254[1], 11 * tmp_8254[2]];
    signal tmp_8256[3] <== [tmp_8253[0] + tmp_8255[0], tmp_8253[1] + tmp_8255[1], tmp_8253[2] + tmp_8255[2]];
    signal tmp_8257[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8258[3] <== [17 * tmp_8257[0], 17 * tmp_8257[1], 17 * tmp_8257[2]];
    signal tmp_8259[3] <== [tmp_8256[0] + tmp_8258[0], tmp_8256[1] + tmp_8258[1], tmp_8256[2] + tmp_8258[2]];
    signal tmp_8260[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8261[3] <== [2 * tmp_8260[0], 2 * tmp_8260[1], 2 * tmp_8260[2]];
    signal tmp_8262[3] <== [tmp_8259[0] + tmp_8261[0], tmp_8259[1] + tmp_8261[1], tmp_8259[2] + tmp_8261[2]];
    signal tmp_8263[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8264[3] <== [tmp_8262[0] + tmp_8263[0], tmp_8262[1] + tmp_8263[1], tmp_8262[2] + tmp_8263[2]];
    signal tmp_8265[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8266[3] <== [101 * tmp_8265[0], 101 * tmp_8265[1], 101 * tmp_8265[2]];
    signal tmp_8267[3] <== [tmp_8264[0] + tmp_8266[0], tmp_8264[1] + tmp_8266[1], tmp_8264[2] + tmp_8266[2]];
    signal tmp_8268[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8269[3] <== [63 * tmp_8268[0], 63 * tmp_8268[1], 63 * tmp_8268[2]];
    signal tmp_8270[3] <== [tmp_8267[0] + tmp_8269[0], tmp_8267[1] + tmp_8269[1], tmp_8267[2] + tmp_8269[2]];
    signal tmp_8271[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8272[3] <== [15 * tmp_8271[0], 15 * tmp_8271[1], 15 * tmp_8271[2]];
    signal tmp_8273[3] <== [tmp_8270[0] + tmp_8272[0], tmp_8270[1] + tmp_8272[1], tmp_8270[2] + tmp_8272[2]];
    signal tmp_8274[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8275[3] <== [2 * tmp_8274[0], 2 * tmp_8274[1], 2 * tmp_8274[2]];
    signal tmp_8276[3] <== [tmp_8273[0] + tmp_8275[0], tmp_8273[1] + tmp_8275[1], tmp_8273[2] + tmp_8275[2]];
    signal tmp_8277[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8278[3] <== [67 * tmp_8277[0], 67 * tmp_8277[1], 67 * tmp_8277[2]];
    signal tmp_8279[3] <== [tmp_8276[0] + tmp_8278[0], tmp_8276[1] + tmp_8278[1], tmp_8276[2] + tmp_8278[2]];
    signal tmp_8280[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8281[3] <== [22 * tmp_8280[0], 22 * tmp_8280[1], 22 * tmp_8280[2]];
    signal tmp_8282[3] <== [tmp_8279[0] + tmp_8281[0], tmp_8279[1] + tmp_8281[1], tmp_8279[2] + tmp_8281[2]];
    signal tmp_8283[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8284[3] <== [13 * tmp_8283[0], 13 * tmp_8283[1], 13 * tmp_8283[2]];
    signal tmp_8285[3] <== [tmp_8282[0] + tmp_8284[0], tmp_8282[1] + tmp_8284[1], tmp_8282[2] + tmp_8284[2]];
    signal tmp_8286[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8287[3] <== [3 * tmp_8286[0], 3 * tmp_8286[1], 3 * tmp_8286[2]];
    signal tmp_8288[3] <== [tmp_8285[0] + tmp_8287[0], tmp_8285[1] + tmp_8287[1], tmp_8285[2] + tmp_8287[2]];
    signal tmp_8289[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8290[3] <== [tmp_8288[0] + tmp_8289[0], tmp_8288[1] + tmp_8289[1], tmp_8288[2] + tmp_8289[2]];
    signal tmp_8291[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8292[3] <== [tmp_8290[0] + tmp_8291[0], tmp_8290[1] + tmp_8291[1], tmp_8290[2] + tmp_8291[2]];
    signal tmp_8293[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8294[3] <== [51 * tmp_8293[0], 51 * tmp_8293[1], 51 * tmp_8293[2]];
    signal tmp_8295[3] <== [tmp_8292[0] + tmp_8294[0], tmp_8292[1] + tmp_8294[1], tmp_8292[2] + tmp_8294[2]];
    signal tmp_8296[3] <== [tmp_8252[0] - tmp_8295[0], tmp_8252[1] - tmp_8295[1], tmp_8252[2] - tmp_8295[2]];
    signal tmp_8297[3] <== CMul()(tmp_8248, tmp_8296);
    signal tmp_8298[3] <== [tmp_8246[0] + tmp_8297[0], tmp_8246[1] + tmp_8297[1], tmp_8246[2] + tmp_8297[2]];
    signal tmp_8299[3] <== CMul()(challengeQ, tmp_8298);
    signal tmp_8300[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8301[3] <== [tmp_8300[0] + evals[41][0], tmp_8300[1] + evals[41][1], tmp_8300[2] + evals[41][2]];
    signal tmp_8302[3] <== CMul()(evals[41], evals[64]);
    signal tmp_8303[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8304[3] <== CMul()(tmp_8303, evals[132]);
    signal tmp_8305[3] <== [tmp_8302[0] + tmp_8304[0], tmp_8302[1] + tmp_8304[1], tmp_8302[2] + tmp_8304[2]];
    signal tmp_8306[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8307[3] <== [51 * tmp_8306[0], 51 * tmp_8306[1], 51 * tmp_8306[2]];
    signal tmp_8308[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8309[3] <== [tmp_8307[0] + tmp_8308[0], tmp_8307[1] + tmp_8308[1], tmp_8307[2] + tmp_8308[2]];
    signal tmp_8310[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8311[3] <== [11 * tmp_8310[0], 11 * tmp_8310[1], 11 * tmp_8310[2]];
    signal tmp_8312[3] <== [tmp_8309[0] + tmp_8311[0], tmp_8309[1] + tmp_8311[1], tmp_8309[2] + tmp_8311[2]];
    signal tmp_8313[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8314[3] <== [17 * tmp_8313[0], 17 * tmp_8313[1], 17 * tmp_8313[2]];
    signal tmp_8315[3] <== [tmp_8312[0] + tmp_8314[0], tmp_8312[1] + tmp_8314[1], tmp_8312[2] + tmp_8314[2]];
    signal tmp_8316[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8317[3] <== [2 * tmp_8316[0], 2 * tmp_8316[1], 2 * tmp_8316[2]];
    signal tmp_8318[3] <== [tmp_8315[0] + tmp_8317[0], tmp_8315[1] + tmp_8317[1], tmp_8315[2] + tmp_8317[2]];
    signal tmp_8319[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8320[3] <== [tmp_8318[0] + tmp_8319[0], tmp_8318[1] + tmp_8319[1], tmp_8318[2] + tmp_8319[2]];
    signal tmp_8321[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8322[3] <== [101 * tmp_8321[0], 101 * tmp_8321[1], 101 * tmp_8321[2]];
    signal tmp_8323[3] <== [tmp_8320[0] + tmp_8322[0], tmp_8320[1] + tmp_8322[1], tmp_8320[2] + tmp_8322[2]];
    signal tmp_8324[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8325[3] <== [63 * tmp_8324[0], 63 * tmp_8324[1], 63 * tmp_8324[2]];
    signal tmp_8326[3] <== [tmp_8323[0] + tmp_8325[0], tmp_8323[1] + tmp_8325[1], tmp_8323[2] + tmp_8325[2]];
    signal tmp_8327[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8328[3] <== [15 * tmp_8327[0], 15 * tmp_8327[1], 15 * tmp_8327[2]];
    signal tmp_8329[3] <== [tmp_8326[0] + tmp_8328[0], tmp_8326[1] + tmp_8328[1], tmp_8326[2] + tmp_8328[2]];
    signal tmp_8330[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8331[3] <== [2 * tmp_8330[0], 2 * tmp_8330[1], 2 * tmp_8330[2]];
    signal tmp_8332[3] <== [tmp_8329[0] + tmp_8331[0], tmp_8329[1] + tmp_8331[1], tmp_8329[2] + tmp_8331[2]];
    signal tmp_8333[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8334[3] <== [67 * tmp_8333[0], 67 * tmp_8333[1], 67 * tmp_8333[2]];
    signal tmp_8335[3] <== [tmp_8332[0] + tmp_8334[0], tmp_8332[1] + tmp_8334[1], tmp_8332[2] + tmp_8334[2]];
    signal tmp_8336[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8337[3] <== [22 * tmp_8336[0], 22 * tmp_8336[1], 22 * tmp_8336[2]];
    signal tmp_8338[3] <== [tmp_8335[0] + tmp_8337[0], tmp_8335[1] + tmp_8337[1], tmp_8335[2] + tmp_8337[2]];
    signal tmp_8339[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8340[3] <== [13 * tmp_8339[0], 13 * tmp_8339[1], 13 * tmp_8339[2]];
    signal tmp_8341[3] <== [tmp_8338[0] + tmp_8340[0], tmp_8338[1] + tmp_8340[1], tmp_8338[2] + tmp_8340[2]];
    signal tmp_8342[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8343[3] <== [3 * tmp_8342[0], 3 * tmp_8342[1], 3 * tmp_8342[2]];
    signal tmp_8344[3] <== [tmp_8341[0] + tmp_8343[0], tmp_8341[1] + tmp_8343[1], tmp_8341[2] + tmp_8343[2]];
    signal tmp_8345[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8346[3] <== [tmp_8344[0] + tmp_8345[0], tmp_8344[1] + tmp_8345[1], tmp_8344[2] + tmp_8345[2]];
    signal tmp_8347[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8348[3] <== [tmp_8346[0] + tmp_8347[0], tmp_8346[1] + tmp_8347[1], tmp_8346[2] + tmp_8347[2]];
    signal tmp_8349[3] <== [tmp_8305[0] - tmp_8348[0], tmp_8305[1] - tmp_8348[1], tmp_8305[2] - tmp_8348[2]];
    signal tmp_8350[3] <== CMul()(tmp_8301, tmp_8349);
    signal tmp_8351[3] <== [tmp_8299[0] + tmp_8350[0], tmp_8299[1] + tmp_8350[1], tmp_8299[2] + tmp_8350[2]];
    signal tmp_8352[3] <== CMul()(challengeQ, tmp_8351);
    signal tmp_8353[3] <== [tmp_6405[0] + evals[109][0], tmp_6405[1] + evals[109][1], tmp_6405[2] + evals[109][2]];
    signal tmp_8354[3] <== [tmp_8353[0] + evals[41][0], tmp_8353[1] + evals[41][1], tmp_8353[2] + evals[41][2]];
    signal tmp_8355[3] <== CMul()(evals[41], evals[65]);
    signal tmp_8356[3] <== [1 - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_8357[3] <== CMul()(tmp_8356, evals[133]);
    signal tmp_8358[3] <== [tmp_8355[0] + tmp_8357[0], tmp_8355[1] + tmp_8357[1], tmp_8355[2] + tmp_8357[2]];
    signal tmp_8359[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8360[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8361[3] <== [51 * tmp_8360[0], 51 * tmp_8360[1], 51 * tmp_8360[2]];
    signal tmp_8362[3] <== [tmp_8359[0] + tmp_8361[0], tmp_8359[1] + tmp_8361[1], tmp_8359[2] + tmp_8361[2]];
    signal tmp_8363[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8364[3] <== [tmp_8362[0] + tmp_8363[0], tmp_8362[1] + tmp_8363[1], tmp_8362[2] + tmp_8363[2]];
    signal tmp_8365[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8366[3] <== [11 * tmp_8365[0], 11 * tmp_8365[1], 11 * tmp_8365[2]];
    signal tmp_8367[3] <== [tmp_8364[0] + tmp_8366[0], tmp_8364[1] + tmp_8366[1], tmp_8364[2] + tmp_8366[2]];
    signal tmp_8368[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8369[3] <== [17 * tmp_8368[0], 17 * tmp_8368[1], 17 * tmp_8368[2]];
    signal tmp_8370[3] <== [tmp_8367[0] + tmp_8369[0], tmp_8367[1] + tmp_8369[1], tmp_8367[2] + tmp_8369[2]];
    signal tmp_8371[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8372[3] <== [2 * tmp_8371[0], 2 * tmp_8371[1], 2 * tmp_8371[2]];
    signal tmp_8373[3] <== [tmp_8370[0] + tmp_8372[0], tmp_8370[1] + tmp_8372[1], tmp_8370[2] + tmp_8372[2]];
    signal tmp_8374[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8375[3] <== [tmp_8373[0] + tmp_8374[0], tmp_8373[1] + tmp_8374[1], tmp_8373[2] + tmp_8374[2]];
    signal tmp_8376[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8377[3] <== [101 * tmp_8376[0], 101 * tmp_8376[1], 101 * tmp_8376[2]];
    signal tmp_8378[3] <== [tmp_8375[0] + tmp_8377[0], tmp_8375[1] + tmp_8377[1], tmp_8375[2] + tmp_8377[2]];
    signal tmp_8379[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8380[3] <== [63 * tmp_8379[0], 63 * tmp_8379[1], 63 * tmp_8379[2]];
    signal tmp_8381[3] <== [tmp_8378[0] + tmp_8380[0], tmp_8378[1] + tmp_8380[1], tmp_8378[2] + tmp_8380[2]];
    signal tmp_8382[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8383[3] <== [15 * tmp_8382[0], 15 * tmp_8382[1], 15 * tmp_8382[2]];
    signal tmp_8384[3] <== [tmp_8381[0] + tmp_8383[0], tmp_8381[1] + tmp_8383[1], tmp_8381[2] + tmp_8383[2]];
    signal tmp_8385[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8386[3] <== [2 * tmp_8385[0], 2 * tmp_8385[1], 2 * tmp_8385[2]];
    signal tmp_8387[3] <== [tmp_8384[0] + tmp_8386[0], tmp_8384[1] + tmp_8386[1], tmp_8384[2] + tmp_8386[2]];
    signal tmp_8388[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8389[3] <== [67 * tmp_8388[0], 67 * tmp_8388[1], 67 * tmp_8388[2]];
    signal tmp_8390[3] <== [tmp_8387[0] + tmp_8389[0], tmp_8387[1] + tmp_8389[1], tmp_8387[2] + tmp_8389[2]];
    signal tmp_8391[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8392[3] <== [22 * tmp_8391[0], 22 * tmp_8391[1], 22 * tmp_8391[2]];
    signal tmp_8393[3] <== [tmp_8390[0] + tmp_8392[0], tmp_8390[1] + tmp_8392[1], tmp_8390[2] + tmp_8392[2]];
    signal tmp_8394[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8395[3] <== [13 * tmp_8394[0], 13 * tmp_8394[1], 13 * tmp_8394[2]];
    signal tmp_8396[3] <== [tmp_8393[0] + tmp_8395[0], tmp_8393[1] + tmp_8395[1], tmp_8393[2] + tmp_8395[2]];
    signal tmp_8397[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8398[3] <== [3 * tmp_8397[0], 3 * tmp_8397[1], 3 * tmp_8397[2]];
    signal tmp_8399[3] <== [tmp_8396[0] + tmp_8398[0], tmp_8396[1] + tmp_8398[1], tmp_8396[2] + tmp_8398[2]];
    signal tmp_8400[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8401[3] <== [tmp_8399[0] + tmp_8400[0], tmp_8399[1] + tmp_8400[1], tmp_8399[2] + tmp_8400[2]];
    signal tmp_8402[3] <== [tmp_8358[0] - tmp_8401[0], tmp_8358[1] - tmp_8401[1], tmp_8358[2] - tmp_8401[2]];
    signal tmp_8403[3] <== CMul()(tmp_8354, tmp_8402);
    signal tmp_8404[3] <== [tmp_8352[0] + tmp_8403[0], tmp_8352[1] + tmp_8403[1], tmp_8352[2] + tmp_8403[2]];
    signal tmp_8405[3] <== CMul()(challengeQ, tmp_8404);
    signal tmp_8406[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8407[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8408[3] <== [tmp_8406[0] + tmp_8407[0], tmp_8406[1] + tmp_8407[1], tmp_8406[2] + tmp_8407[2]];
    signal tmp_8409[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8410[3] <== [51 * tmp_8409[0], 51 * tmp_8409[1], 51 * tmp_8409[2]];
    signal tmp_8411[3] <== [tmp_8408[0] + tmp_8410[0], tmp_8408[1] + tmp_8410[1], tmp_8408[2] + tmp_8410[2]];
    signal tmp_8412[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8413[3] <== [tmp_8411[0] + tmp_8412[0], tmp_8411[1] + tmp_8412[1], tmp_8411[2] + tmp_8412[2]];
    signal tmp_8414[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8415[3] <== [11 * tmp_8414[0], 11 * tmp_8414[1], 11 * tmp_8414[2]];
    signal tmp_8416[3] <== [tmp_8413[0] + tmp_8415[0], tmp_8413[1] + tmp_8415[1], tmp_8413[2] + tmp_8415[2]];
    signal tmp_8417[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8418[3] <== [17 * tmp_8417[0], 17 * tmp_8417[1], 17 * tmp_8417[2]];
    signal tmp_8419[3] <== [tmp_8416[0] + tmp_8418[0], tmp_8416[1] + tmp_8418[1], tmp_8416[2] + tmp_8418[2]];
    signal tmp_8420[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8421[3] <== [2 * tmp_8420[0], 2 * tmp_8420[1], 2 * tmp_8420[2]];
    signal tmp_8422[3] <== [tmp_8419[0] + tmp_8421[0], tmp_8419[1] + tmp_8421[1], tmp_8419[2] + tmp_8421[2]];
    signal tmp_8423[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8424[3] <== [tmp_8422[0] + tmp_8423[0], tmp_8422[1] + tmp_8423[1], tmp_8422[2] + tmp_8423[2]];
    signal tmp_8425[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8426[3] <== [101 * tmp_8425[0], 101 * tmp_8425[1], 101 * tmp_8425[2]];
    signal tmp_8427[3] <== [tmp_8424[0] + tmp_8426[0], tmp_8424[1] + tmp_8426[1], tmp_8424[2] + tmp_8426[2]];
    signal tmp_8428[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8429[3] <== [63 * tmp_8428[0], 63 * tmp_8428[1], 63 * tmp_8428[2]];
    signal tmp_8430[3] <== [tmp_8427[0] + tmp_8429[0], tmp_8427[1] + tmp_8429[1], tmp_8427[2] + tmp_8429[2]];
    signal tmp_8431[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8432[3] <== [15 * tmp_8431[0], 15 * tmp_8431[1], 15 * tmp_8431[2]];
    signal tmp_8433[3] <== [tmp_8430[0] + tmp_8432[0], tmp_8430[1] + tmp_8432[1], tmp_8430[2] + tmp_8432[2]];
    signal tmp_8434[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8435[3] <== [2 * tmp_8434[0], 2 * tmp_8434[1], 2 * tmp_8434[2]];
    signal tmp_8436[3] <== [tmp_8433[0] + tmp_8435[0], tmp_8433[1] + tmp_8435[1], tmp_8433[2] + tmp_8435[2]];
    signal tmp_8437[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8438[3] <== [67 * tmp_8437[0], 67 * tmp_8437[1], 67 * tmp_8437[2]];
    signal tmp_8439[3] <== [tmp_8436[0] + tmp_8438[0], tmp_8436[1] + tmp_8438[1], tmp_8436[2] + tmp_8438[2]];
    signal tmp_8440[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8441[3] <== [22 * tmp_8440[0], 22 * tmp_8440[1], 22 * tmp_8440[2]];
    signal tmp_8442[3] <== [tmp_8439[0] + tmp_8441[0], tmp_8439[1] + tmp_8441[1], tmp_8439[2] + tmp_8441[2]];
    signal tmp_8443[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8444[3] <== [13 * tmp_8443[0], 13 * tmp_8443[1], 13 * tmp_8443[2]];
    signal tmp_8445[3] <== [tmp_8442[0] + tmp_8444[0], tmp_8442[1] + tmp_8444[1], tmp_8442[2] + tmp_8444[2]];
    signal tmp_8446[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8447[3] <== [3 * tmp_8446[0], 3 * tmp_8446[1], 3 * tmp_8446[2]];
    signal tmp_8448[3] <== [tmp_8445[0] + tmp_8447[0], tmp_8445[1] + tmp_8447[1], tmp_8445[2] + tmp_8447[2]];
    signal tmp_8449[3] <== [evals[118][0] - tmp_8448[0], evals[118][1] - tmp_8448[1], evals[118][2] - tmp_8448[2]];
    signal tmp_8450[3] <== CMul()(tmp_6066, tmp_8449);
    signal tmp_8451[3] <== [tmp_8405[0] + tmp_8450[0], tmp_8405[1] + tmp_8450[1], tmp_8405[2] + tmp_8450[2]];
    signal tmp_8452[3] <== CMul()(challengeQ, tmp_8451);
    signal tmp_8453[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8454[3] <== [2256277865552202420 * tmp_8453[0], 2256277865552202420 * tmp_8453[1], 2256277865552202420 * tmp_8453[2]];
    signal tmp_8455[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8456[3] <== [4068529056800825848 * tmp_8455[0], 4068529056800825848 * tmp_8455[1], 4068529056800825848 * tmp_8455[2]];
    signal tmp_8457[3] <== [tmp_8454[0] + tmp_8456[0], tmp_8454[1] + tmp_8456[1], tmp_8454[2] + tmp_8456[2]];
    signal tmp_8458[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8459[3] <== [8593802027805519400 * tmp_8458[0], 8593802027805519400 * tmp_8458[1], 8593802027805519400 * tmp_8458[2]];
    signal tmp_8460[3] <== [tmp_8457[0] + tmp_8459[0], tmp_8457[1] + tmp_8459[1], tmp_8457[2] + tmp_8459[2]];
    signal tmp_8461[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8462[3] <== [10290793996632644659 * tmp_8461[0], 10290793996632644659 * tmp_8461[1], 10290793996632644659 * tmp_8461[2]];
    signal tmp_8463[3] <== [tmp_8460[0] + tmp_8462[0], tmp_8460[1] + tmp_8462[1], tmp_8460[2] + tmp_8462[2]];
    signal tmp_8464[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8465[3] <== [18408474278105675815 * tmp_8464[0], 18408474278105675815 * tmp_8464[1], 18408474278105675815 * tmp_8464[2]];
    signal tmp_8466[3] <== [tmp_8463[0] + tmp_8465[0], tmp_8463[1] + tmp_8465[1], tmp_8463[2] + tmp_8465[2]];
    signal tmp_8467[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8468[3] <== [17884228943875298379 * tmp_8467[0], 17884228943875298379 * tmp_8467[1], 17884228943875298379 * tmp_8467[2]];
    signal tmp_8469[3] <== [tmp_8466[0] + tmp_8468[0], tmp_8466[1] + tmp_8468[1], tmp_8466[2] + tmp_8468[2]];
    signal tmp_8470[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8471[3] <== [6702954500202032351 * tmp_8470[0], 6702954500202032351 * tmp_8470[1], 6702954500202032351 * tmp_8470[2]];
    signal tmp_8472[3] <== [tmp_8469[0] + tmp_8471[0], tmp_8469[1] + tmp_8471[1], tmp_8469[2] + tmp_8471[2]];
    signal tmp_8473[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8474[3] <== [11180511074533951034 * tmp_8473[0], 11180511074533951034 * tmp_8473[1], 11180511074533951034 * tmp_8473[2]];
    signal tmp_8475[3] <== [tmp_8472[0] + tmp_8474[0], tmp_8472[1] + tmp_8474[1], tmp_8472[2] + tmp_8474[2]];
    signal tmp_8476[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8477[3] <== [6468404447315445157 * tmp_8476[0], 6468404447315445157 * tmp_8476[1], 6468404447315445157 * tmp_8476[2]];
    signal tmp_8478[3] <== [tmp_8475[0] + tmp_8477[0], tmp_8475[1] + tmp_8477[1], tmp_8475[2] + tmp_8477[2]];
    signal tmp_8479[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8480[3] <== [12266684712140129864 * tmp_8479[0], 12266684712140129864 * tmp_8479[1], 12266684712140129864 * tmp_8479[2]];
    signal tmp_8481[3] <== [tmp_8478[0] + tmp_8480[0], tmp_8478[1] + tmp_8480[1], tmp_8478[2] + tmp_8480[2]];
    signal tmp_8482[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8483[3] <== [6782251596454435859 * tmp_8482[0], 6782251596454435859 * tmp_8482[1], 6782251596454435859 * tmp_8482[2]];
    signal tmp_8484[3] <== [tmp_8481[0] + tmp_8483[0], tmp_8481[1] + tmp_8483[1], tmp_8481[2] + tmp_8483[2]];
    signal tmp_8485[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8486[3] <== [2765050638571938600 * tmp_8485[0], 2765050638571938600 * tmp_8485[1], 2765050638571938600 * tmp_8485[2]];
    signal tmp_8487[3] <== [tmp_8484[0] + tmp_8486[0], tmp_8484[1] + tmp_8486[1], tmp_8484[2] + tmp_8486[2]];
    signal tmp_8488[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8489[3] <== [17337375643812333090 * tmp_8488[0], 17337375643812333090 * tmp_8488[1], 17337375643812333090 * tmp_8488[2]];
    signal tmp_8490[3] <== [tmp_8487[0] + tmp_8489[0], tmp_8487[1] + tmp_8489[1], tmp_8487[2] + tmp_8489[2]];
    signal tmp_8491[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8492[3] <== [13578549160148989543 * tmp_8491[0], 13578549160148989543 * tmp_8491[1], 13578549160148989543 * tmp_8491[2]];
    signal tmp_8493[3] <== [tmp_8490[0] + tmp_8492[0], tmp_8490[1] + tmp_8492[1], tmp_8490[2] + tmp_8492[2]];
    signal tmp_8494[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8495[3] <== [4068336395193158675 * tmp_8494[0], 4068336395193158675 * tmp_8494[1], 4068336395193158675 * tmp_8494[2]];
    signal tmp_8496[3] <== [tmp_8493[0] + tmp_8495[0], tmp_8493[1] + tmp_8495[1], tmp_8493[2] + tmp_8495[2]];
    signal tmp_8497[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8498[3] <== [1571908583333476476 * tmp_8497[0], 1571908583333476476 * tmp_8497[1], 1571908583333476476 * tmp_8497[2]];
    signal tmp_8499[3] <== [tmp_8496[0] + tmp_8498[0], tmp_8496[1] + tmp_8498[1], tmp_8496[2] + tmp_8498[2]];
    signal tmp_8500[3] <== [evals[119][0] - tmp_8499[0], evals[119][1] - tmp_8499[1], evals[119][2] - tmp_8499[2]];
    signal tmp_8501[3] <== CMul()(tmp_6066, tmp_8500);
    signal tmp_8502[3] <== [tmp_8452[0] + tmp_8501[0], tmp_8452[1] + tmp_8501[1], tmp_8452[2] + tmp_8501[2]];
    signal tmp_8503[3] <== CMul()(challengeQ, tmp_8502);
    signal tmp_8504[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8505[3] <== [17717269319234059614 * tmp_8504[0], 17717269319234059614 * tmp_8504[1], 17717269319234059614 * tmp_8504[2]];
    signal tmp_8506[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8507[3] <== [17521923232000336525 * tmp_8506[0], 17521923232000336525 * tmp_8506[1], 17521923232000336525 * tmp_8506[2]];
    signal tmp_8508[3] <== [tmp_8505[0] + tmp_8507[0], tmp_8505[1] + tmp_8507[1], tmp_8505[2] + tmp_8507[2]];
    signal tmp_8509[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8510[3] <== [7891351762273827007 * tmp_8509[0], 7891351762273827007 * tmp_8509[1], 7891351762273827007 * tmp_8509[2]];
    signal tmp_8511[3] <== [tmp_8508[0] + tmp_8510[0], tmp_8508[1] + tmp_8510[1], tmp_8508[2] + tmp_8510[2]];
    signal tmp_8512[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8513[3] <== [14226032251042260572 * tmp_8512[0], 14226032251042260572 * tmp_8512[1], 14226032251042260572 * tmp_8512[2]];
    signal tmp_8514[3] <== [tmp_8511[0] + tmp_8513[0], tmp_8511[1] + tmp_8513[1], tmp_8511[2] + tmp_8513[2]];
    signal tmp_8515[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8516[3] <== [17611042948119270535 * tmp_8515[0], 17611042948119270535 * tmp_8515[1], 17611042948119270535 * tmp_8515[2]];
    signal tmp_8517[3] <== [tmp_8514[0] + tmp_8516[0], tmp_8514[1] + tmp_8516[1], tmp_8514[2] + tmp_8516[2]];
    signal tmp_8518[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8519[3] <== [1810923409258252859 * tmp_8518[0], 1810923409258252859 * tmp_8518[1], 1810923409258252859 * tmp_8518[2]];
    signal tmp_8520[3] <== [tmp_8517[0] + tmp_8519[0], tmp_8517[1] + tmp_8519[1], tmp_8517[2] + tmp_8519[2]];
    signal tmp_8521[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8522[3] <== [6551497314953504346 * tmp_8521[0], 6551497314953504346 * tmp_8521[1], 6551497314953504346 * tmp_8521[2]];
    signal tmp_8523[3] <== [tmp_8520[0] + tmp_8522[0], tmp_8520[1] + tmp_8522[1], tmp_8520[2] + tmp_8522[2]];
    signal tmp_8524[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8525[3] <== [14645553078015703863 * tmp_8524[0], 14645553078015703863 * tmp_8524[1], 14645553078015703863 * tmp_8524[2]];
    signal tmp_8526[3] <== [tmp_8523[0] + tmp_8525[0], tmp_8523[1] + tmp_8525[1], tmp_8523[2] + tmp_8525[2]];
    signal tmp_8527[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8528[3] <== [17162061841860881922 * tmp_8527[0], 17162061841860881922 * tmp_8527[1], 17162061841860881922 * tmp_8527[2]];
    signal tmp_8529[3] <== [tmp_8526[0] + tmp_8528[0], tmp_8526[1] + tmp_8528[1], tmp_8526[2] + tmp_8528[2]];
    signal tmp_8530[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8531[3] <== [16803151265961688719 * tmp_8530[0], 16803151265961688719 * tmp_8530[1], 16803151265961688719 * tmp_8530[2]];
    signal tmp_8532[3] <== [tmp_8529[0] + tmp_8531[0], tmp_8529[1] + tmp_8531[1], tmp_8529[2] + tmp_8531[2]];
    signal tmp_8533[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8534[3] <== [1725490462187460494 * tmp_8533[0], 1725490462187460494 * tmp_8533[1], 1725490462187460494 * tmp_8533[2]];
    signal tmp_8535[3] <== [tmp_8532[0] + tmp_8534[0], tmp_8532[1] + tmp_8534[1], tmp_8532[2] + tmp_8534[2]];
    signal tmp_8536[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8537[3] <== [9867063047527519896 * tmp_8536[0], 9867063047527519896 * tmp_8536[1], 9867063047527519896 * tmp_8536[2]];
    signal tmp_8538[3] <== [tmp_8535[0] + tmp_8537[0], tmp_8535[1] + tmp_8537[1], tmp_8535[2] + tmp_8537[2]];
    signal tmp_8539[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8540[3] <== [14201875104048534679 * tmp_8539[0], 14201875104048534679 * tmp_8539[1], 14201875104048534679 * tmp_8539[2]];
    signal tmp_8541[3] <== [tmp_8538[0] + tmp_8540[0], tmp_8538[1] + tmp_8540[1], tmp_8538[2] + tmp_8540[2]];
    signal tmp_8542[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8543[3] <== [218778007896712464 * tmp_8542[0], 218778007896712464 * tmp_8542[1], 218778007896712464 * tmp_8542[2]];
    signal tmp_8544[3] <== [tmp_8541[0] + tmp_8543[0], tmp_8541[1] + tmp_8543[1], tmp_8541[2] + tmp_8543[2]];
    signal tmp_8545[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8546[3] <== [12636278992549723093 * tmp_8545[0], 12636278992549723093 * tmp_8545[1], 12636278992549723093 * tmp_8545[2]];
    signal tmp_8547[3] <== [tmp_8544[0] + tmp_8546[0], tmp_8544[1] + tmp_8546[1], tmp_8544[2] + tmp_8546[2]];
    signal tmp_8548[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8549[3] <== [4068336395193158675 * tmp_8548[0], 4068336395193158675 * tmp_8548[1], 4068336395193158675 * tmp_8548[2]];
    signal tmp_8550[3] <== [tmp_8547[0] + tmp_8549[0], tmp_8547[1] + tmp_8549[1], tmp_8547[2] + tmp_8549[2]];
    signal tmp_8551[3] <== [evals[120][0] - tmp_8550[0], evals[120][1] - tmp_8550[1], evals[120][2] - tmp_8550[2]];
    signal tmp_8552[3] <== CMul()(tmp_6066, tmp_8551);
    signal tmp_8553[3] <== [tmp_8503[0] + tmp_8552[0], tmp_8503[1] + tmp_8552[1], tmp_8503[2] + tmp_8552[2]];
    signal tmp_8554[3] <== CMul()(challengeQ, tmp_8553);
    signal tmp_8555[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8556[3] <== [12035225151448622303 * tmp_8555[0], 12035225151448622303 * tmp_8555[1], 12035225151448622303 * tmp_8555[2]];
    signal tmp_8557[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8558[3] <== [8855639374359979570 * tmp_8557[0], 8855639374359979570 * tmp_8557[1], 8855639374359979570 * tmp_8557[2]];
    signal tmp_8559[3] <== [tmp_8556[0] + tmp_8558[0], tmp_8556[1] + tmp_8558[1], tmp_8556[2] + tmp_8558[2]];
    signal tmp_8560[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8561[3] <== [10872182090012722023 * tmp_8560[0], 10872182090012722023 * tmp_8560[1], 10872182090012722023 * tmp_8560[2]];
    signal tmp_8562[3] <== [tmp_8559[0] + tmp_8561[0], tmp_8559[1] + tmp_8561[1], tmp_8559[2] + tmp_8561[2]];
    signal tmp_8563[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8564[3] <== [17245004066890521540 * tmp_8563[0], 17245004066890521540 * tmp_8563[1], 17245004066890521540 * tmp_8563[2]];
    signal tmp_8565[3] <== [tmp_8562[0] + tmp_8564[0], tmp_8562[1] + tmp_8564[1], tmp_8562[2] + tmp_8564[2]];
    signal tmp_8566[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8567[3] <== [1582298761054006274 * tmp_8566[0], 1582298761054006274 * tmp_8566[1], 1582298761054006274 * tmp_8566[2]];
    signal tmp_8568[3] <== [tmp_8565[0] + tmp_8567[0], tmp_8565[1] + tmp_8567[1], tmp_8565[2] + tmp_8567[2]];
    signal tmp_8569[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8570[3] <== [16023586798997043070 * tmp_8569[0], 16023586798997043070 * tmp_8569[1], 16023586798997043070 * tmp_8569[2]];
    signal tmp_8571[3] <== [tmp_8568[0] + tmp_8570[0], tmp_8568[1] + tmp_8570[1], tmp_8568[2] + tmp_8570[2]];
    signal tmp_8572[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8573[3] <== [9167682708495677368 * tmp_8572[0], 9167682708495677368 * tmp_8572[1], 9167682708495677368 * tmp_8572[2]];
    signal tmp_8574[3] <== [tmp_8571[0] + tmp_8573[0], tmp_8571[1] + tmp_8573[1], tmp_8571[2] + tmp_8573[2]];
    signal tmp_8575[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8576[3] <== [1309677305456162249 * tmp_8575[0], 1309677305456162249 * tmp_8575[1], 1309677305456162249 * tmp_8575[2]];
    signal tmp_8577[3] <== [tmp_8574[0] + tmp_8576[0], tmp_8574[1] + tmp_8576[1], tmp_8574[2] + tmp_8576[2]];
    signal tmp_8578[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8579[3] <== [1052212424660468163 * tmp_8578[0], 1052212424660468163 * tmp_8578[1], 1052212424660468163 * tmp_8578[2]];
    signal tmp_8580[3] <== [tmp_8577[0] + tmp_8579[0], tmp_8577[1] + tmp_8579[1], tmp_8577[2] + tmp_8579[2]];
    signal tmp_8581[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8582[3] <== [10584314677809541083 * tmp_8581[0], 10584314677809541083 * tmp_8581[1], 10584314677809541083 * tmp_8581[2]];
    signal tmp_8583[3] <== [tmp_8580[0] + tmp_8582[0], tmp_8580[1] + tmp_8582[1], tmp_8580[2] + tmp_8582[2]];
    signal tmp_8584[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8585[3] <== [14090216998035345570 * tmp_8584[0], 14090216998035345570 * tmp_8584[1], 14090216998035345570 * tmp_8584[2]];
    signal tmp_8586[3] <== [tmp_8583[0] + tmp_8585[0], tmp_8583[1] + tmp_8585[1], tmp_8583[2] + tmp_8585[2]];
    signal tmp_8587[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8588[3] <== [18211453817785255223 * tmp_8587[0], 18211453817785255223 * tmp_8587[1], 18211453817785255223 * tmp_8587[2]];
    signal tmp_8589[3] <== [tmp_8586[0] + tmp_8588[0], tmp_8586[1] + tmp_8588[1], tmp_8586[2] + tmp_8588[2]];
    signal tmp_8590[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8591[3] <== [7540469726131798188 * tmp_8590[0], 7540469726131798188 * tmp_8590[1], 7540469726131798188 * tmp_8590[2]];
    signal tmp_8592[3] <== [tmp_8589[0] + tmp_8591[0], tmp_8589[1] + tmp_8591[1], tmp_8589[2] + tmp_8591[2]];
    signal tmp_8593[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8594[3] <== [8480085599690188340 * tmp_8593[0], 8480085599690188340 * tmp_8593[1], 8480085599690188340 * tmp_8593[2]];
    signal tmp_8595[3] <== [tmp_8592[0] + tmp_8594[0], tmp_8592[1] + tmp_8594[1], tmp_8592[2] + tmp_8594[2]];
    signal tmp_8596[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8597[3] <== [218778007896712464 * tmp_8596[0], 218778007896712464 * tmp_8596[1], 218778007896712464 * tmp_8596[2]];
    signal tmp_8598[3] <== [tmp_8595[0] + tmp_8597[0], tmp_8595[1] + tmp_8597[1], tmp_8595[2] + tmp_8597[2]];
    signal tmp_8599[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8600[3] <== [13578549160148989543 * tmp_8599[0], 13578549160148989543 * tmp_8599[1], 13578549160148989543 * tmp_8599[2]];
    signal tmp_8601[3] <== [tmp_8598[0] + tmp_8600[0], tmp_8598[1] + tmp_8600[1], tmp_8598[2] + tmp_8600[2]];
    signal tmp_8602[3] <== [evals[121][0] - tmp_8601[0], evals[121][1] - tmp_8601[1], evals[121][2] - tmp_8601[2]];
    signal tmp_8603[3] <== CMul()(tmp_6066, tmp_8602);
    signal tmp_8604[3] <== [tmp_8554[0] + tmp_8603[0], tmp_8554[1] + tmp_8603[1], tmp_8554[2] + tmp_8603[2]];
    signal tmp_8605[3] <== CMul()(challengeQ, tmp_8604);
    signal tmp_8606[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8607[3] <== [10833525987555107485 * tmp_8606[0], 10833525987555107485 * tmp_8606[1], 10833525987555107485 * tmp_8606[2]];
    signal tmp_8608[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8609[3] <== [3224171887150947390 * tmp_8608[0], 3224171887150947390 * tmp_8608[1], 3224171887150947390 * tmp_8608[2]];
    signal tmp_8610[3] <== [tmp_8607[0] + tmp_8609[0], tmp_8607[1] + tmp_8609[1], tmp_8607[2] + tmp_8609[2]];
    signal tmp_8611[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8612[3] <== [15832068917340433117 * tmp_8611[0], 15832068917340433117 * tmp_8611[1], 15832068917340433117 * tmp_8611[2]];
    signal tmp_8613[3] <== [tmp_8610[0] + tmp_8612[0], tmp_8610[1] + tmp_8612[1], tmp_8610[2] + tmp_8612[2]];
    signal tmp_8614[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8615[3] <== [7595102312128106020 * tmp_8614[0], 7595102312128106020 * tmp_8614[1], 7595102312128106020 * tmp_8614[2]];
    signal tmp_8616[3] <== [tmp_8613[0] + tmp_8615[0], tmp_8613[1] + tmp_8615[1], tmp_8613[2] + tmp_8615[2]];
    signal tmp_8617[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8618[3] <== [7700354089512536567 * tmp_8617[0], 7700354089512536567 * tmp_8617[1], 7700354089512536567 * tmp_8617[2]];
    signal tmp_8619[3] <== [tmp_8616[0] + tmp_8618[0], tmp_8616[1] + tmp_8618[1], tmp_8616[2] + tmp_8618[2]];
    signal tmp_8620[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8621[3] <== [12431191248332837556 * tmp_8620[0], 12431191248332837556 * tmp_8620[1], 12431191248332837556 * tmp_8620[2]];
    signal tmp_8622[3] <== [tmp_8619[0] + tmp_8621[0], tmp_8619[1] + tmp_8621[1], tmp_8619[2] + tmp_8621[2]];
    signal tmp_8623[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8624[3] <== [8615081588084801800 * tmp_8623[0], 8615081588084801800 * tmp_8623[1], 8615081588084801800 * tmp_8623[2]];
    signal tmp_8625[3] <== [tmp_8622[0] + tmp_8624[0], tmp_8622[1] + tmp_8624[1], tmp_8622[2] + tmp_8624[2]];
    signal tmp_8626[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8627[3] <== [17713155928766469063 * tmp_8626[0], 17713155928766469063 * tmp_8626[1], 17713155928766469063 * tmp_8626[2]];
    signal tmp_8628[3] <== [tmp_8625[0] + tmp_8627[0], tmp_8625[1] + tmp_8627[1], tmp_8625[2] + tmp_8627[2]];
    signal tmp_8629[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8630[3] <== [13520163937605726164 * tmp_8629[0], 13520163937605726164 * tmp_8629[1], 13520163937605726164 * tmp_8629[2]];
    signal tmp_8631[3] <== [tmp_8628[0] + tmp_8630[0], tmp_8628[1] + tmp_8630[1], tmp_8628[2] + tmp_8630[2]];
    signal tmp_8632[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8633[3] <== [13134043455414658692 * tmp_8632[0], 13134043455414658692 * tmp_8632[1], 13134043455414658692 * tmp_8632[2]];
    signal tmp_8634[3] <== [tmp_8631[0] + tmp_8633[0], tmp_8631[1] + tmp_8633[1], tmp_8631[2] + tmp_8633[2]];
    signal tmp_8635[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8636[3] <== [5807446223844687465 * tmp_8635[0], 5807446223844687465 * tmp_8635[1], 5807446223844687465 * tmp_8635[2]];
    signal tmp_8637[3] <== [tmp_8634[0] + tmp_8636[0], tmp_8634[1] + tmp_8636[1], tmp_8634[2] + tmp_8636[2]];
    signal tmp_8638[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8639[3] <== [6129425131043046986 * tmp_8638[0], 6129425131043046986 * tmp_8638[1], 6129425131043046986 * tmp_8638[2]];
    signal tmp_8640[3] <== [tmp_8637[0] + tmp_8639[0], tmp_8637[1] + tmp_8639[1], tmp_8637[2] + tmp_8639[2]];
    signal tmp_8641[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8642[3] <== [16009746502472456190 * tmp_8641[0], 16009746502472456190 * tmp_8641[1], 16009746502472456190 * tmp_8641[2]];
    signal tmp_8643[3] <== [tmp_8640[0] + tmp_8642[0], tmp_8640[1] + tmp_8642[1], tmp_8640[2] + tmp_8642[2]];
    signal tmp_8644[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8645[3] <== [7540469726131798188 * tmp_8644[0], 7540469726131798188 * tmp_8644[1], 7540469726131798188 * tmp_8644[2]];
    signal tmp_8646[3] <== [tmp_8643[0] + tmp_8645[0], tmp_8643[1] + tmp_8645[1], tmp_8643[2] + tmp_8645[2]];
    signal tmp_8647[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8648[3] <== [14201875104048534679 * tmp_8647[0], 14201875104048534679 * tmp_8647[1], 14201875104048534679 * tmp_8647[2]];
    signal tmp_8649[3] <== [tmp_8646[0] + tmp_8648[0], tmp_8646[1] + tmp_8648[1], tmp_8646[2] + tmp_8648[2]];
    signal tmp_8650[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8651[3] <== [17337375643812333090 * tmp_8650[0], 17337375643812333090 * tmp_8650[1], 17337375643812333090 * tmp_8650[2]];
    signal tmp_8652[3] <== [tmp_8649[0] + tmp_8651[0], tmp_8649[1] + tmp_8651[1], tmp_8649[2] + tmp_8651[2]];
    signal tmp_8653[3] <== [evals[122][0] - tmp_8652[0], evals[122][1] - tmp_8652[1], evals[122][2] - tmp_8652[2]];
    signal tmp_8654[3] <== CMul()(tmp_6066, tmp_8653);
    signal tmp_8655[3] <== [tmp_8605[0] + tmp_8654[0], tmp_8605[1] + tmp_8654[1], tmp_8605[2] + tmp_8654[2]];
    signal tmp_8656[3] <== CMul()(challengeQ, tmp_8655);
    signal tmp_8657[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8658[3] <== [8429731608347594559 * tmp_8657[0], 8429731608347594559 * tmp_8657[1], 8429731608347594559 * tmp_8657[2]];
    signal tmp_8659[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8660[3] <== [1843095456038194143 * tmp_8659[0], 1843095456038194143 * tmp_8659[1], 1843095456038194143 * tmp_8659[2]];
    signal tmp_8661[3] <== [tmp_8658[0] + tmp_8660[0], tmp_8658[1] + tmp_8660[1], tmp_8658[2] + tmp_8660[2]];
    signal tmp_8662[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8663[3] <== [4800348490672825001 * tmp_8662[0], 4800348490672825001 * tmp_8662[1], 4800348490672825001 * tmp_8662[2]];
    signal tmp_8664[3] <== [tmp_8661[0] + tmp_8663[0], tmp_8661[1] + tmp_8663[1], tmp_8661[2] + tmp_8663[2]];
    signal tmp_8665[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8666[3] <== [15175373777650174966 * tmp_8665[0], 15175373777650174966 * tmp_8665[1], 15175373777650174966 * tmp_8665[2]];
    signal tmp_8667[3] <== [tmp_8664[0] + tmp_8666[0], tmp_8664[1] + tmp_8666[1], tmp_8664[2] + tmp_8666[2]];
    signal tmp_8668[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8669[3] <== [15743243826515774520 * tmp_8668[0], 15743243826515774520 * tmp_8668[1], 15743243826515774520 * tmp_8668[2]];
    signal tmp_8670[3] <== [tmp_8667[0] + tmp_8669[0], tmp_8667[1] + tmp_8669[1], tmp_8667[2] + tmp_8669[2]];
    signal tmp_8671[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8672[3] <== [2498880319480651868 * tmp_8671[0], 2498880319480651868 * tmp_8671[1], 2498880319480651868 * tmp_8671[2]];
    signal tmp_8673[3] <== [tmp_8670[0] + tmp_8672[0], tmp_8670[1] + tmp_8672[1], tmp_8670[2] + tmp_8672[2]];
    signal tmp_8674[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8675[3] <== [7546636406606303166 * tmp_8674[0], 7546636406606303166 * tmp_8674[1], 7546636406606303166 * tmp_8674[2]];
    signal tmp_8676[3] <== [tmp_8673[0] + tmp_8675[0], tmp_8673[1] + tmp_8675[1], tmp_8673[2] + tmp_8675[2]];
    signal tmp_8677[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8678[3] <== [15350435609153813292 * tmp_8677[0], 15350435609153813292 * tmp_8677[1], 15350435609153813292 * tmp_8677[2]];
    signal tmp_8679[3] <== [tmp_8676[0] + tmp_8678[0], tmp_8676[1] + tmp_8678[1], tmp_8676[2] + tmp_8678[2]];
    signal tmp_8680[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8681[3] <== [16548416728592494912 * tmp_8680[0], 16548416728592494912 * tmp_8680[1], 16548416728592494912 * tmp_8680[2]];
    signal tmp_8682[3] <== [tmp_8679[0] + tmp_8681[0], tmp_8679[1] + tmp_8681[1], tmp_8679[2] + tmp_8681[2]];
    signal tmp_8683[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8684[3] <== [17993594930758673496 * tmp_8683[0], 17993594930758673496 * tmp_8683[1], 17993594930758673496 * tmp_8683[2]];
    signal tmp_8685[3] <== [tmp_8682[0] + tmp_8684[0], tmp_8682[1] + tmp_8684[1], tmp_8682[2] + tmp_8684[2]];
    signal tmp_8686[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8687[3] <== [13912658301238544598 * tmp_8686[0], 13912658301238544598 * tmp_8686[1], 13912658301238544598 * tmp_8686[2]];
    signal tmp_8688[3] <== [tmp_8685[0] + tmp_8687[0], tmp_8685[1] + tmp_8687[1], tmp_8685[2] + tmp_8687[2]];
    signal tmp_8689[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8690[3] <== [18075372928619437347 * tmp_8689[0], 18075372928619437347 * tmp_8689[1], 18075372928619437347 * tmp_8689[2]];
    signal tmp_8691[3] <== [tmp_8688[0] + tmp_8690[0], tmp_8688[1] + tmp_8690[1], tmp_8688[2] + tmp_8690[2]];
    signal tmp_8692[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8693[3] <== [6129425131043046986 * tmp_8692[0], 6129425131043046986 * tmp_8692[1], 6129425131043046986 * tmp_8692[2]];
    signal tmp_8694[3] <== [tmp_8691[0] + tmp_8693[0], tmp_8691[1] + tmp_8693[1], tmp_8691[2] + tmp_8693[2]];
    signal tmp_8695[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8696[3] <== [18211453817785255223 * tmp_8695[0], 18211453817785255223 * tmp_8695[1], 18211453817785255223 * tmp_8695[2]];
    signal tmp_8697[3] <== [tmp_8694[0] + tmp_8696[0], tmp_8694[1] + tmp_8696[1], tmp_8694[2] + tmp_8696[2]];
    signal tmp_8698[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8699[3] <== [9867063047527519896 * tmp_8698[0], 9867063047527519896 * tmp_8698[1], 9867063047527519896 * tmp_8698[2]];
    signal tmp_8700[3] <== [tmp_8697[0] + tmp_8699[0], tmp_8697[1] + tmp_8699[1], tmp_8697[2] + tmp_8699[2]];
    signal tmp_8701[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8702[3] <== [2765050638571938600 * tmp_8701[0], 2765050638571938600 * tmp_8701[1], 2765050638571938600 * tmp_8701[2]];
    signal tmp_8703[3] <== [tmp_8700[0] + tmp_8702[0], tmp_8700[1] + tmp_8702[1], tmp_8700[2] + tmp_8702[2]];
    signal tmp_8704[3] <== [evals[123][0] - tmp_8703[0], evals[123][1] - tmp_8703[1], evals[123][2] - tmp_8703[2]];
    signal tmp_8705[3] <== CMul()(tmp_6066, tmp_8704);
    signal tmp_8706[3] <== [tmp_8656[0] + tmp_8705[0], tmp_8656[1] + tmp_8705[1], tmp_8656[2] + tmp_8705[2]];
    signal tmp_8707[3] <== CMul()(challengeQ, tmp_8706);
    signal tmp_8708[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8709[3] <== [15303535044422768803 * tmp_8708[0], 15303535044422768803 * tmp_8708[1], 15303535044422768803 * tmp_8708[2]];
    signal tmp_8710[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8711[3] <== [6296951922972871295 * tmp_8710[0], 6296951922972871295 * tmp_8710[1], 6296951922972871295 * tmp_8710[2]];
    signal tmp_8712[3] <== [tmp_8709[0] + tmp_8711[0], tmp_8709[1] + tmp_8711[1], tmp_8709[2] + tmp_8711[2]];
    signal tmp_8713[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8714[3] <== [5823573138716677219 * tmp_8713[0], 5823573138716677219 * tmp_8713[1], 5823573138716677219 * tmp_8713[2]];
    signal tmp_8715[3] <== [tmp_8712[0] + tmp_8714[0], tmp_8712[1] + tmp_8714[1], tmp_8712[2] + tmp_8714[2]];
    signal tmp_8716[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8717[3] <== [2680431002152629139 * tmp_8716[0], 2680431002152629139 * tmp_8716[1], 2680431002152629139 * tmp_8716[2]];
    signal tmp_8718[3] <== [tmp_8715[0] + tmp_8717[0], tmp_8715[1] + tmp_8717[1], tmp_8715[2] + tmp_8717[2]];
    signal tmp_8719[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8720[3] <== [7366465902859006190 * tmp_8719[0], 7366465902859006190 * tmp_8719[1], 7366465902859006190 * tmp_8719[2]];
    signal tmp_8721[3] <== [tmp_8718[0] + tmp_8720[0], tmp_8718[1] + tmp_8720[1], tmp_8718[2] + tmp_8720[2]];
    signal tmp_8722[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8723[3] <== [4072077849300314190 * tmp_8722[0], 4072077849300314190 * tmp_8722[1], 4072077849300314190 * tmp_8722[2]];
    signal tmp_8724[3] <== [tmp_8721[0] + tmp_8723[0], tmp_8721[1] + tmp_8723[1], tmp_8721[2] + tmp_8723[2]];
    signal tmp_8725[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8726[3] <== [16140126883652446716 * tmp_8725[0], 16140126883652446716 * tmp_8725[1], 16140126883652446716 * tmp_8725[2]];
    signal tmp_8727[3] <== [tmp_8724[0] + tmp_8726[0], tmp_8724[1] + tmp_8726[1], tmp_8724[2] + tmp_8726[2]];
    signal tmp_8728[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8729[3] <== [2601451337846871084 * tmp_8728[0], 2601451337846871084 * tmp_8728[1], 2601451337846871084 * tmp_8728[2]];
    signal tmp_8730[3] <== [tmp_8727[0] + tmp_8729[0], tmp_8727[1] + tmp_8729[1], tmp_8727[2] + tmp_8729[2]];
    signal tmp_8731[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8732[3] <== [15725013897929440397 * tmp_8731[0], 15725013897929440397 * tmp_8731[1], 15725013897929440397 * tmp_8731[2]];
    signal tmp_8733[3] <== [tmp_8730[0] + tmp_8732[0], tmp_8730[1] + tmp_8732[1], tmp_8730[2] + tmp_8732[2]];
    signal tmp_8734[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8735[3] <== [4389532408180633377 * tmp_8734[0], 4389532408180633377 * tmp_8734[1], 4389532408180633377 * tmp_8734[2]];
    signal tmp_8736[3] <== [tmp_8733[0] + tmp_8735[0], tmp_8733[1] + tmp_8735[1], tmp_8733[2] + tmp_8735[2]];
    signal tmp_8737[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8738[3] <== [1476671367976490131 * tmp_8737[0], 1476671367976490131 * tmp_8737[1], 1476671367976490131 * tmp_8737[2]];
    signal tmp_8739[3] <== [tmp_8736[0] + tmp_8738[0], tmp_8736[1] + tmp_8738[1], tmp_8736[2] + tmp_8738[2]];
    signal tmp_8740[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8741[3] <== [13912658301238544598 * tmp_8740[0], 13912658301238544598 * tmp_8740[1], 13912658301238544598 * tmp_8740[2]];
    signal tmp_8742[3] <== [tmp_8739[0] + tmp_8741[0], tmp_8739[1] + tmp_8741[1], tmp_8739[2] + tmp_8741[2]];
    signal tmp_8743[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8744[3] <== [5807446223844687465 * tmp_8743[0], 5807446223844687465 * tmp_8743[1], 5807446223844687465 * tmp_8743[2]];
    signal tmp_8745[3] <== [tmp_8742[0] + tmp_8744[0], tmp_8742[1] + tmp_8744[1], tmp_8742[2] + tmp_8744[2]];
    signal tmp_8746[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8747[3] <== [14090216998035345570 * tmp_8746[0], 14090216998035345570 * tmp_8746[1], 14090216998035345570 * tmp_8746[2]];
    signal tmp_8748[3] <== [tmp_8745[0] + tmp_8747[0], tmp_8745[1] + tmp_8747[1], tmp_8745[2] + tmp_8747[2]];
    signal tmp_8749[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8750[3] <== [1725490462187460494 * tmp_8749[0], 1725490462187460494 * tmp_8749[1], 1725490462187460494 * tmp_8749[2]];
    signal tmp_8751[3] <== [tmp_8748[0] + tmp_8750[0], tmp_8748[1] + tmp_8750[1], tmp_8748[2] + tmp_8750[2]];
    signal tmp_8752[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8753[3] <== [6782251596454435859 * tmp_8752[0], 6782251596454435859 * tmp_8752[1], 6782251596454435859 * tmp_8752[2]];
    signal tmp_8754[3] <== [tmp_8751[0] + tmp_8753[0], tmp_8751[1] + tmp_8753[1], tmp_8751[2] + tmp_8753[2]];
    signal tmp_8755[3] <== [evals[124][0] - tmp_8754[0], evals[124][1] - tmp_8754[1], evals[124][2] - tmp_8754[2]];
    signal tmp_8756[3] <== CMul()(tmp_6066, tmp_8755);
    signal tmp_8757[3] <== [tmp_8707[0] + tmp_8756[0], tmp_8707[1] + tmp_8756[1], tmp_8707[2] + tmp_8756[2]];
    signal tmp_8758[3] <== CMul()(challengeQ, tmp_8757);
    signal tmp_8759[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8760[3] <== [10595610446894069209 * tmp_8759[0], 10595610446894069209 * tmp_8759[1], 10595610446894069209 * tmp_8759[2]];
    signal tmp_8761[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8762[3] <== [3473665614188209191 * tmp_8761[0], 3473665614188209191 * tmp_8761[1], 3473665614188209191 * tmp_8761[2]];
    signal tmp_8763[3] <== [tmp_8760[0] + tmp_8762[0], tmp_8760[1] + tmp_8762[1], tmp_8760[2] + tmp_8762[2]];
    signal tmp_8764[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8765[3] <== [13074116889890804777 * tmp_8764[0], 13074116889890804777 * tmp_8764[1], 13074116889890804777 * tmp_8764[2]];
    signal tmp_8766[3] <== [tmp_8763[0] + tmp_8765[0], tmp_8763[1] + tmp_8765[1], tmp_8763[2] + tmp_8765[2]];
    signal tmp_8767[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8768[3] <== [16359376528724855414 * tmp_8767[0], 16359376528724855414 * tmp_8767[1], 16359376528724855414 * tmp_8767[2]];
    signal tmp_8769[3] <== [tmp_8766[0] + tmp_8768[0], tmp_8766[1] + tmp_8768[1], tmp_8766[2] + tmp_8768[2]];
    signal tmp_8770[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8771[3] <== [16204753907425352648 * tmp_8770[0], 16204753907425352648 * tmp_8770[1], 16204753907425352648 * tmp_8770[2]];
    signal tmp_8772[3] <== [tmp_8769[0] + tmp_8771[0], tmp_8769[1] + tmp_8771[1], tmp_8769[2] + tmp_8771[2]];
    signal tmp_8773[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8774[3] <== [1099103502109499253 * tmp_8773[0], 1099103502109499253 * tmp_8773[1], 1099103502109499253 * tmp_8773[2]];
    signal tmp_8775[3] <== [tmp_8772[0] + tmp_8774[0], tmp_8772[1] + tmp_8774[1], tmp_8772[2] + tmp_8774[2]];
    signal tmp_8776[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8777[3] <== [4616609025864245522 * tmp_8776[0], 4616609025864245522 * tmp_8776[1], 4616609025864245522 * tmp_8776[2]];
    signal tmp_8778[3] <== [tmp_8775[0] + tmp_8777[0], tmp_8775[1] + tmp_8777[1], tmp_8775[2] + tmp_8777[2]];
    signal tmp_8779[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8780[3] <== [7362869829681961507 * tmp_8779[0], 7362869829681961507 * tmp_8779[1], 7362869829681961507 * tmp_8779[2]];
    signal tmp_8781[3] <== [tmp_8778[0] + tmp_8780[0], tmp_8778[1] + tmp_8780[1], tmp_8778[2] + tmp_8780[2]];
    signal tmp_8782[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8783[3] <== [231791526046105962 * tmp_8782[0], 231791526046105962 * tmp_8782[1], 231791526046105962 * tmp_8782[2]];
    signal tmp_8784[3] <== [tmp_8781[0] + tmp_8783[0], tmp_8781[1] + tmp_8783[1], tmp_8781[2] + tmp_8783[2]];
    signal tmp_8785[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8786[3] <== [140048027282429797 * tmp_8785[0], 140048027282429797 * tmp_8785[1], 140048027282429797 * tmp_8785[2]];
    signal tmp_8787[3] <== [tmp_8784[0] + tmp_8786[0], tmp_8784[1] + tmp_8786[1], tmp_8784[2] + tmp_8786[2]];
    signal tmp_8788[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8789[3] <== [4389532408180633377 * tmp_8788[0], 4389532408180633377 * tmp_8788[1], 4389532408180633377 * tmp_8788[2]];
    signal tmp_8790[3] <== [tmp_8787[0] + tmp_8789[0], tmp_8787[1] + tmp_8789[1], tmp_8787[2] + tmp_8789[2]];
    signal tmp_8791[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8792[3] <== [17993594930758673496 * tmp_8791[0], 17993594930758673496 * tmp_8791[1], 17993594930758673496 * tmp_8791[2]];
    signal tmp_8793[3] <== [tmp_8790[0] + tmp_8792[0], tmp_8790[1] + tmp_8792[1], tmp_8790[2] + tmp_8792[2]];
    signal tmp_8794[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8795[3] <== [13134043455414658692 * tmp_8794[0], 13134043455414658692 * tmp_8794[1], 13134043455414658692 * tmp_8794[2]];
    signal tmp_8796[3] <== [tmp_8793[0] + tmp_8795[0], tmp_8793[1] + tmp_8795[1], tmp_8793[2] + tmp_8795[2]];
    signal tmp_8797[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8798[3] <== [10584314677809541083 * tmp_8797[0], 10584314677809541083 * tmp_8797[1], 10584314677809541083 * tmp_8797[2]];
    signal tmp_8799[3] <== [tmp_8796[0] + tmp_8798[0], tmp_8796[1] + tmp_8798[1], tmp_8796[2] + tmp_8798[2]];
    signal tmp_8800[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8801[3] <== [16803151265961688719 * tmp_8800[0], 16803151265961688719 * tmp_8800[1], 16803151265961688719 * tmp_8800[2]];
    signal tmp_8802[3] <== [tmp_8799[0] + tmp_8801[0], tmp_8799[1] + tmp_8801[1], tmp_8799[2] + tmp_8801[2]];
    signal tmp_8803[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8804[3] <== [12266684712140129864 * tmp_8803[0], 12266684712140129864 * tmp_8803[1], 12266684712140129864 * tmp_8803[2]];
    signal tmp_8805[3] <== [tmp_8802[0] + tmp_8804[0], tmp_8802[1] + tmp_8804[1], tmp_8802[2] + tmp_8804[2]];
    signal tmp_8806[3] <== [evals[125][0] - tmp_8805[0], evals[125][1] - tmp_8805[1], evals[125][2] - tmp_8805[2]];
    signal tmp_8807[3] <== CMul()(tmp_6066, tmp_8806);
    signal tmp_8808[3] <== [tmp_8758[0] + tmp_8807[0], tmp_8758[1] + tmp_8807[1], tmp_8758[2] + tmp_8807[2]];
    signal tmp_8809[3] <== CMul()(challengeQ, tmp_8808);
    signal tmp_8810[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8811[3] <== [13368554547082880001 * tmp_8810[0], 13368554547082880001 * tmp_8810[1], 13368554547082880001 * tmp_8810[2]];
    signal tmp_8812[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8813[3] <== [16995162224826066164 * tmp_8812[0], 16995162224826066164 * tmp_8812[1], 16995162224826066164 * tmp_8812[2]];
    signal tmp_8814[3] <== [tmp_8811[0] + tmp_8813[0], tmp_8811[1] + tmp_8813[1], tmp_8811[2] + tmp_8813[2]];
    signal tmp_8815[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8816[3] <== [10542619009725154399 * tmp_8815[0], 10542619009725154399 * tmp_8815[1], 10542619009725154399 * tmp_8815[2]];
    signal tmp_8817[3] <== [tmp_8814[0] + tmp_8816[0], tmp_8814[1] + tmp_8816[1], tmp_8814[2] + tmp_8816[2]];
    signal tmp_8818[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8819[3] <== [16073073456050006575 * tmp_8818[0], 16073073456050006575 * tmp_8818[1], 16073073456050006575 * tmp_8818[2]];
    signal tmp_8820[3] <== [tmp_8817[0] + tmp_8819[0], tmp_8817[1] + tmp_8819[1], tmp_8817[2] + tmp_8819[2]];
    signal tmp_8821[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8822[3] <== [9008401972144753939 * tmp_8821[0], 9008401972144753939 * tmp_8821[1], 9008401972144753939 * tmp_8821[2]];
    signal tmp_8823[3] <== [tmp_8820[0] + tmp_8822[0], tmp_8820[1] + tmp_8822[1], tmp_8820[2] + tmp_8822[2]];
    signal tmp_8824[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8825[3] <== [1749680724535600815 * tmp_8824[0], 1749680724535600815 * tmp_8824[1], 1749680724535600815 * tmp_8824[2]];
    signal tmp_8826[3] <== [tmp_8823[0] + tmp_8825[0], tmp_8823[1] + tmp_8825[1], tmp_8823[2] + tmp_8825[2]];
    signal tmp_8827[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8828[3] <== [5348209698235102331 * tmp_8827[0], 5348209698235102331 * tmp_8827[1], 5348209698235102331 * tmp_8827[2]];
    signal tmp_8829[3] <== [tmp_8826[0] + tmp_8828[0], tmp_8826[1] + tmp_8828[1], tmp_8826[2] + tmp_8828[2]];
    signal tmp_8830[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8831[3] <== [5200922087361472057 * tmp_8830[0], 5200922087361472057 * tmp_8830[1], 5200922087361472057 * tmp_8830[2]];
    signal tmp_8832[3] <== [tmp_8829[0] + tmp_8831[0], tmp_8829[1] + tmp_8831[1], tmp_8829[2] + tmp_8831[2]];
    signal tmp_8833[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8834[3] <== [16325322198676206943 * tmp_8833[0], 16325322198676206943 * tmp_8833[1], 16325322198676206943 * tmp_8833[2]];
    signal tmp_8835[3] <== [tmp_8832[0] + tmp_8834[0], tmp_8832[1] + tmp_8834[1], tmp_8832[2] + tmp_8834[2]];
    signal tmp_8836[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8837[3] <== [231791526046105962 * tmp_8836[0], 231791526046105962 * tmp_8836[1], 231791526046105962 * tmp_8836[2]];
    signal tmp_8838[3] <== [tmp_8835[0] + tmp_8837[0], tmp_8835[1] + tmp_8837[1], tmp_8835[2] + tmp_8837[2]];
    signal tmp_8839[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8840[3] <== [15725013897929440397 * tmp_8839[0], 15725013897929440397 * tmp_8839[1], 15725013897929440397 * tmp_8839[2]];
    signal tmp_8841[3] <== [tmp_8838[0] + tmp_8840[0], tmp_8838[1] + tmp_8840[1], tmp_8838[2] + tmp_8840[2]];
    signal tmp_8842[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8843[3] <== [16548416728592494912 * tmp_8842[0], 16548416728592494912 * tmp_8842[1], 16548416728592494912 * tmp_8842[2]];
    signal tmp_8844[3] <== [tmp_8841[0] + tmp_8843[0], tmp_8841[1] + tmp_8843[1], tmp_8841[2] + tmp_8843[2]];
    signal tmp_8845[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8846[3] <== [13520163937605726164 * tmp_8845[0], 13520163937605726164 * tmp_8845[1], 13520163937605726164 * tmp_8845[2]];
    signal tmp_8847[3] <== [tmp_8844[0] + tmp_8846[0], tmp_8844[1] + tmp_8846[1], tmp_8844[2] + tmp_8846[2]];
    signal tmp_8848[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8849[3] <== [1052212424660468163 * tmp_8848[0], 1052212424660468163 * tmp_8848[1], 1052212424660468163 * tmp_8848[2]];
    signal tmp_8850[3] <== [tmp_8847[0] + tmp_8849[0], tmp_8847[1] + tmp_8849[1], tmp_8847[2] + tmp_8849[2]];
    signal tmp_8851[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8852[3] <== [17162061841860881922 * tmp_8851[0], 17162061841860881922 * tmp_8851[1], 17162061841860881922 * tmp_8851[2]];
    signal tmp_8853[3] <== [tmp_8850[0] + tmp_8852[0], tmp_8850[1] + tmp_8852[1], tmp_8850[2] + tmp_8852[2]];
    signal tmp_8854[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8855[3] <== [6468404447315445157 * tmp_8854[0], 6468404447315445157 * tmp_8854[1], 6468404447315445157 * tmp_8854[2]];
    signal tmp_8856[3] <== [tmp_8853[0] + tmp_8855[0], tmp_8853[1] + tmp_8855[1], tmp_8853[2] + tmp_8855[2]];
    signal tmp_8857[3] <== [evals[126][0] - tmp_8856[0], evals[126][1] - tmp_8856[1], evals[126][2] - tmp_8856[2]];
    signal tmp_8858[3] <== CMul()(tmp_6066, tmp_8857);
    signal tmp_8859[3] <== [tmp_8809[0] + tmp_8858[0], tmp_8809[1] + tmp_8858[1], tmp_8809[2] + tmp_8858[2]];
    signal tmp_8860[3] <== CMul()(challengeQ, tmp_8859);
    signal tmp_8861[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8862[3] <== [16236357391915579752 * tmp_8861[0], 16236357391915579752 * tmp_8861[1], 16236357391915579752 * tmp_8861[2]];
    signal tmp_8863[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8864[3] <== [2360357678915503787 * tmp_8863[0], 2360357678915503787 * tmp_8863[1], 2360357678915503787 * tmp_8863[2]];
    signal tmp_8865[3] <== [tmp_8862[0] + tmp_8864[0], tmp_8862[1] + tmp_8864[1], tmp_8862[2] + tmp_8864[2]];
    signal tmp_8866[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8867[3] <== [5269705790554862459 * tmp_8866[0], 5269705790554862459 * tmp_8866[1], 5269705790554862459 * tmp_8866[2]];
    signal tmp_8868[3] <== [tmp_8865[0] + tmp_8867[0], tmp_8865[1] + tmp_8867[1], tmp_8865[2] + tmp_8867[2]];
    signal tmp_8869[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8870[3] <== [9707166089812971789 * tmp_8869[0], 9707166089812971789 * tmp_8869[1], 9707166089812971789 * tmp_8869[2]];
    signal tmp_8871[3] <== [tmp_8868[0] + tmp_8870[0], tmp_8868[1] + tmp_8870[1], tmp_8868[2] + tmp_8870[2]];
    signal tmp_8872[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8873[3] <== [13551115817015094532 * tmp_8872[0], 13551115817015094532 * tmp_8872[1], 13551115817015094532 * tmp_8872[2]];
    signal tmp_8874[3] <== [tmp_8871[0] + tmp_8873[0], tmp_8871[1] + tmp_8873[1], tmp_8871[2] + tmp_8873[2]];
    signal tmp_8875[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8876[3] <== [11909911263973988019 * tmp_8875[0], 11909911263973988019 * tmp_8875[1], 11909911263973988019 * tmp_8875[2]];
    signal tmp_8877[3] <== [tmp_8874[0] + tmp_8876[0], tmp_8874[1] + tmp_8876[1], tmp_8874[2] + tmp_8876[2]];
    signal tmp_8878[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8879[3] <== [5477818864382526223 * tmp_8878[0], 5477818864382526223 * tmp_8878[1], 5477818864382526223 * tmp_8878[2]];
    signal tmp_8880[3] <== [tmp_8877[0] + tmp_8879[0], tmp_8877[1] + tmp_8879[1], tmp_8877[2] + tmp_8879[2]];
    signal tmp_8881[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8882[3] <== [6662788181627778720 * tmp_8881[0], 6662788181627778720 * tmp_8881[1], 6662788181627778720 * tmp_8881[2]];
    signal tmp_8883[3] <== [tmp_8880[0] + tmp_8882[0], tmp_8880[1] + tmp_8882[1], tmp_8880[2] + tmp_8882[2]];
    signal tmp_8884[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8885[3] <== [5200922087361472057 * tmp_8884[0], 5200922087361472057 * tmp_8884[1], 5200922087361472057 * tmp_8884[2]];
    signal tmp_8886[3] <== [tmp_8883[0] + tmp_8885[0], tmp_8883[1] + tmp_8885[1], tmp_8883[2] + tmp_8885[2]];
    signal tmp_8887[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8888[3] <== [7362869829681961507 * tmp_8887[0], 7362869829681961507 * tmp_8887[1], 7362869829681961507 * tmp_8887[2]];
    signal tmp_8889[3] <== [tmp_8886[0] + tmp_8888[0], tmp_8886[1] + tmp_8888[1], tmp_8886[2] + tmp_8888[2]];
    signal tmp_8890[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8891[3] <== [2601451337846871084 * tmp_8890[0], 2601451337846871084 * tmp_8890[1], 2601451337846871084 * tmp_8890[2]];
    signal tmp_8892[3] <== [tmp_8889[0] + tmp_8891[0], tmp_8889[1] + tmp_8891[1], tmp_8889[2] + tmp_8891[2]];
    signal tmp_8893[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8894[3] <== [15350435609153813292 * tmp_8893[0], 15350435609153813292 * tmp_8893[1], 15350435609153813292 * tmp_8893[2]];
    signal tmp_8895[3] <== [tmp_8892[0] + tmp_8894[0], tmp_8892[1] + tmp_8894[1], tmp_8892[2] + tmp_8894[2]];
    signal tmp_8896[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8897[3] <== [17713155928766469063 * tmp_8896[0], 17713155928766469063 * tmp_8896[1], 17713155928766469063 * tmp_8896[2]];
    signal tmp_8898[3] <== [tmp_8895[0] + tmp_8897[0], tmp_8895[1] + tmp_8897[1], tmp_8895[2] + tmp_8897[2]];
    signal tmp_8899[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8900[3] <== [1309677305456162249 * tmp_8899[0], 1309677305456162249 * tmp_8899[1], 1309677305456162249 * tmp_8899[2]];
    signal tmp_8901[3] <== [tmp_8898[0] + tmp_8900[0], tmp_8898[1] + tmp_8900[1], tmp_8898[2] + tmp_8900[2]];
    signal tmp_8902[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8903[3] <== [14645553078015703863 * tmp_8902[0], 14645553078015703863 * tmp_8902[1], 14645553078015703863 * tmp_8902[2]];
    signal tmp_8904[3] <== [tmp_8901[0] + tmp_8903[0], tmp_8901[1] + tmp_8903[1], tmp_8901[2] + tmp_8903[2]];
    signal tmp_8905[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8906[3] <== [11180511074533951034 * tmp_8905[0], 11180511074533951034 * tmp_8905[1], 11180511074533951034 * tmp_8905[2]];
    signal tmp_8907[3] <== [tmp_8904[0] + tmp_8906[0], tmp_8904[1] + tmp_8906[1], tmp_8904[2] + tmp_8906[2]];
    signal tmp_8908[3] <== [evals[127][0] - tmp_8907[0], evals[127][1] - tmp_8907[1], evals[127][2] - tmp_8907[2]];
    signal tmp_8909[3] <== CMul()(tmp_6066, tmp_8908);
    signal tmp_8910[3] <== [tmp_8860[0] + tmp_8909[0], tmp_8860[1] + tmp_8909[1], tmp_8860[2] + tmp_8909[2]];
    signal tmp_8911[3] <== CMul()(challengeQ, tmp_8910);
    signal tmp_8912[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8913[3] <== [8046651557188262448 * tmp_8912[0], 8046651557188262448 * tmp_8912[1], 8046651557188262448 * tmp_8912[2]];
    signal tmp_8914[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8915[3] <== [14505687488637668062 * tmp_8914[0], 14505687488637668062 * tmp_8914[1], 14505687488637668062 * tmp_8914[2]];
    signal tmp_8916[3] <== [tmp_8913[0] + tmp_8915[0], tmp_8913[1] + tmp_8915[1], tmp_8913[2] + tmp_8915[2]];
    signal tmp_8917[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8918[3] <== [10449762191515258658 * tmp_8917[0], 10449762191515258658 * tmp_8917[1], 10449762191515258658 * tmp_8917[2]];
    signal tmp_8919[3] <== [tmp_8916[0] + tmp_8918[0], tmp_8916[1] + tmp_8918[1], tmp_8916[2] + tmp_8918[2]];
    signal tmp_8920[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8921[3] <== [12361291211423778677 * tmp_8920[0], 12361291211423778677 * tmp_8920[1], 12361291211423778677 * tmp_8920[2]];
    signal tmp_8922[3] <== [tmp_8919[0] + tmp_8921[0], tmp_8919[1] + tmp_8921[1], tmp_8919[2] + tmp_8921[2]];
    signal tmp_8923[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8924[3] <== [14725357302690445139 * tmp_8923[0], 14725357302690445139 * tmp_8923[1], 14725357302690445139 * tmp_8923[2]];
    signal tmp_8925[3] <== [tmp_8922[0] + tmp_8924[0], tmp_8922[1] + tmp_8924[1], tmp_8922[2] + tmp_8924[2]];
    signal tmp_8926[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8927[3] <== [13834032822097992727 * tmp_8926[0], 13834032822097992727 * tmp_8926[1], 13834032822097992727 * tmp_8926[2]];
    signal tmp_8928[3] <== [tmp_8925[0] + tmp_8927[0], tmp_8925[1] + tmp_8927[1], tmp_8925[2] + tmp_8927[2]];
    signal tmp_8929[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8930[3] <== [7412077663147604037 * tmp_8929[0], 7412077663147604037 * tmp_8929[1], 7412077663147604037 * tmp_8929[2]];
    signal tmp_8931[3] <== [tmp_8928[0] + tmp_8930[0], tmp_8928[1] + tmp_8930[1], tmp_8928[2] + tmp_8930[2]];
    signal tmp_8932[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8933[3] <== [5477818864382526223 * tmp_8932[0], 5477818864382526223 * tmp_8932[1], 5477818864382526223 * tmp_8932[2]];
    signal tmp_8934[3] <== [tmp_8931[0] + tmp_8933[0], tmp_8931[1] + tmp_8933[1], tmp_8931[2] + tmp_8933[2]];
    signal tmp_8935[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8936[3] <== [5348209698235102331 * tmp_8935[0], 5348209698235102331 * tmp_8935[1], 5348209698235102331 * tmp_8935[2]];
    signal tmp_8937[3] <== [tmp_8934[0] + tmp_8936[0], tmp_8934[1] + tmp_8936[1], tmp_8934[2] + tmp_8936[2]];
    signal tmp_8938[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8939[3] <== [4616609025864245522 * tmp_8938[0], 4616609025864245522 * tmp_8938[1], 4616609025864245522 * tmp_8938[2]];
    signal tmp_8940[3] <== [tmp_8937[0] + tmp_8939[0], tmp_8937[1] + tmp_8939[1], tmp_8937[2] + tmp_8939[2]];
    signal tmp_8941[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8942[3] <== [16140126883652446716 * tmp_8941[0], 16140126883652446716 * tmp_8941[1], 16140126883652446716 * tmp_8941[2]];
    signal tmp_8943[3] <== [tmp_8940[0] + tmp_8942[0], tmp_8940[1] + tmp_8942[1], tmp_8940[2] + tmp_8942[2]];
    signal tmp_8944[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8945[3] <== [7546636406606303166 * tmp_8944[0], 7546636406606303166 * tmp_8944[1], 7546636406606303166 * tmp_8944[2]];
    signal tmp_8946[3] <== [tmp_8943[0] + tmp_8945[0], tmp_8943[1] + tmp_8945[1], tmp_8943[2] + tmp_8945[2]];
    signal tmp_8947[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8948[3] <== [8615081588084801800 * tmp_8947[0], 8615081588084801800 * tmp_8947[1], 8615081588084801800 * tmp_8947[2]];
    signal tmp_8949[3] <== [tmp_8946[0] + tmp_8948[0], tmp_8946[1] + tmp_8948[1], tmp_8946[2] + tmp_8948[2]];
    signal tmp_8950[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_8951[3] <== [9167682708495677368 * tmp_8950[0], 9167682708495677368 * tmp_8950[1], 9167682708495677368 * tmp_8950[2]];
    signal tmp_8952[3] <== [tmp_8949[0] + tmp_8951[0], tmp_8949[1] + tmp_8951[1], tmp_8949[2] + tmp_8951[2]];
    signal tmp_8953[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_8954[3] <== [6551497314953504346 * tmp_8953[0], 6551497314953504346 * tmp_8953[1], 6551497314953504346 * tmp_8953[2]];
    signal tmp_8955[3] <== [tmp_8952[0] + tmp_8954[0], tmp_8952[1] + tmp_8954[1], tmp_8952[2] + tmp_8954[2]];
    signal tmp_8956[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_8957[3] <== [6702954500202032351 * tmp_8956[0], 6702954500202032351 * tmp_8956[1], 6702954500202032351 * tmp_8956[2]];
    signal tmp_8958[3] <== [tmp_8955[0] + tmp_8957[0], tmp_8955[1] + tmp_8957[1], tmp_8955[2] + tmp_8957[2]];
    signal tmp_8959[3] <== [evals[128][0] - tmp_8958[0], evals[128][1] - tmp_8958[1], evals[128][2] - tmp_8958[2]];
    signal tmp_8960[3] <== CMul()(tmp_6066, tmp_8959);
    signal tmp_8961[3] <== [tmp_8911[0] + tmp_8960[0], tmp_8911[1] + tmp_8960[1], tmp_8911[2] + tmp_8960[2]];
    signal tmp_8962[3] <== CMul()(challengeQ, tmp_8961);
    signal tmp_8963[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_8964[3] <== [10319242957211935185 * tmp_8963[0], 10319242957211935185 * tmp_8963[1], 10319242957211935185 * tmp_8963[2]];
    signal tmp_8965[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_8966[3] <== [2525078022317723924 * tmp_8965[0], 2525078022317723924 * tmp_8965[1], 2525078022317723924 * tmp_8965[2]];
    signal tmp_8967[3] <== [tmp_8964[0] + tmp_8966[0], tmp_8964[1] + tmp_8966[1], tmp_8964[2] + tmp_8966[2]];
    signal tmp_8968[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_8969[3] <== [7063454713635591383 * tmp_8968[0], 7063454713635591383 * tmp_8968[1], 7063454713635591383 * tmp_8968[2]];
    signal tmp_8970[3] <== [tmp_8967[0] + tmp_8969[0], tmp_8967[1] + tmp_8969[1], tmp_8967[2] + tmp_8969[2]];
    signal tmp_8971[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_8972[3] <== [17138871444129789158 * tmp_8971[0], 17138871444129789158 * tmp_8971[1], 17138871444129789158 * tmp_8971[2]];
    signal tmp_8973[3] <== [tmp_8970[0] + tmp_8972[0], tmp_8970[1] + tmp_8972[1], tmp_8970[2] + tmp_8972[2]];
    signal tmp_8974[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_8975[3] <== [16653803537962124952 * tmp_8974[0], 16653803537962124952 * tmp_8974[1], 16653803537962124952 * tmp_8974[2]];
    signal tmp_8976[3] <== [tmp_8973[0] + tmp_8975[0], tmp_8973[1] + tmp_8975[1], tmp_8973[2] + tmp_8975[2]];
    signal tmp_8977[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_8978[3] <== [4015853678780844755 * tmp_8977[0], 4015853678780844755 * tmp_8977[1], 4015853678780844755 * tmp_8977[2]];
    signal tmp_8979[3] <== [tmp_8976[0] + tmp_8978[0], tmp_8976[1] + tmp_8978[1], tmp_8976[2] + tmp_8978[2]];
    signal tmp_8980[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_8981[3] <== [13834032822097992727 * tmp_8980[0], 13834032822097992727 * tmp_8980[1], 13834032822097992727 * tmp_8980[2]];
    signal tmp_8982[3] <== [tmp_8979[0] + tmp_8981[0], tmp_8979[1] + tmp_8981[1], tmp_8979[2] + tmp_8981[2]];
    signal tmp_8983[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_8984[3] <== [11909911263973988019 * tmp_8983[0], 11909911263973988019 * tmp_8983[1], 11909911263973988019 * tmp_8983[2]];
    signal tmp_8985[3] <== [tmp_8982[0] + tmp_8984[0], tmp_8982[1] + tmp_8984[1], tmp_8982[2] + tmp_8984[2]];
    signal tmp_8986[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_8987[3] <== [1749680724535600815 * tmp_8986[0], 1749680724535600815 * tmp_8986[1], 1749680724535600815 * tmp_8986[2]];
    signal tmp_8988[3] <== [tmp_8985[0] + tmp_8987[0], tmp_8985[1] + tmp_8987[1], tmp_8985[2] + tmp_8987[2]];
    signal tmp_8989[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_8990[3] <== [1099103502109499253 * tmp_8989[0], 1099103502109499253 * tmp_8989[1], 1099103502109499253 * tmp_8989[2]];
    signal tmp_8991[3] <== [tmp_8988[0] + tmp_8990[0], tmp_8988[1] + tmp_8990[1], tmp_8988[2] + tmp_8990[2]];
    signal tmp_8992[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_8993[3] <== [4072077849300314190 * tmp_8992[0], 4072077849300314190 * tmp_8992[1], 4072077849300314190 * tmp_8992[2]];
    signal tmp_8994[3] <== [tmp_8991[0] + tmp_8993[0], tmp_8991[1] + tmp_8993[1], tmp_8991[2] + tmp_8993[2]];
    signal tmp_8995[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_8996[3] <== [2498880319480651868 * tmp_8995[0], 2498880319480651868 * tmp_8995[1], 2498880319480651868 * tmp_8995[2]];
    signal tmp_8997[3] <== [tmp_8994[0] + tmp_8996[0], tmp_8994[1] + tmp_8996[1], tmp_8994[2] + tmp_8996[2]];
    signal tmp_8998[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_8999[3] <== [12431191248332837556 * tmp_8998[0], 12431191248332837556 * tmp_8998[1], 12431191248332837556 * tmp_8998[2]];
    signal tmp_9000[3] <== [tmp_8997[0] + tmp_8999[0], tmp_8997[1] + tmp_8999[1], tmp_8997[2] + tmp_8999[2]];
    signal tmp_9001[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_9002[3] <== [16023586798997043070 * tmp_9001[0], 16023586798997043070 * tmp_9001[1], 16023586798997043070 * tmp_9001[2]];
    signal tmp_9003[3] <== [tmp_9000[0] + tmp_9002[0], tmp_9000[1] + tmp_9002[1], tmp_9000[2] + tmp_9002[2]];
    signal tmp_9004[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_9005[3] <== [1810923409258252859 * tmp_9004[0], 1810923409258252859 * tmp_9004[1], 1810923409258252859 * tmp_9004[2]];
    signal tmp_9006[3] <== [tmp_9003[0] + tmp_9005[0], tmp_9003[1] + tmp_9005[1], tmp_9003[2] + tmp_9005[2]];
    signal tmp_9007[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_9008[3] <== [17884228943875298379 * tmp_9007[0], 17884228943875298379 * tmp_9007[1], 17884228943875298379 * tmp_9007[2]];
    signal tmp_9009[3] <== [tmp_9006[0] + tmp_9008[0], tmp_9006[1] + tmp_9008[1], tmp_9006[2] + tmp_9008[2]];
    signal tmp_9010[3] <== [evals[129][0] - tmp_9009[0], evals[129][1] - tmp_9009[1], evals[129][2] - tmp_9009[2]];
    signal tmp_9011[3] <== CMul()(tmp_6066, tmp_9010);
    signal tmp_9012[3] <== [tmp_8962[0] + tmp_9011[0], tmp_8962[1] + tmp_9011[1], tmp_8962[2] + tmp_9011[2]];
    tmp_9013 <== CMul()(challengeQ, tmp_9012);
    signal tmp_9014[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_9015[3] <== [16137054013242276452 * tmp_9014[0], 16137054013242276452 * tmp_9014[1], 16137054013242276452 * tmp_9014[2]];
    signal tmp_9016[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_9017[3] <== [3252585408322091368 * tmp_9016[0], 3252585408322091368 * tmp_9016[1], 3252585408322091368 * tmp_9016[2]];
    signal tmp_9018[3] <== [tmp_9015[0] + tmp_9017[0], tmp_9015[1] + tmp_9017[1], tmp_9015[2] + tmp_9017[2]];
    signal tmp_9019[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_9020[3] <== [14728006497068635074 * tmp_9019[0], 14728006497068635074 * tmp_9019[1], 14728006497068635074 * tmp_9019[2]];
    signal tmp_9021[3] <== [tmp_9018[0] + tmp_9020[0], tmp_9018[1] + tmp_9020[1], tmp_9018[2] + tmp_9020[2]];
    signal tmp_9022[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_9023[3] <== [3628630143580991743 * tmp_9022[0], 3628630143580991743 * tmp_9022[1], 3628630143580991743 * tmp_9022[2]];
    signal tmp_9024[3] <== [tmp_9021[0] + tmp_9023[0], tmp_9021[1] + tmp_9023[1], tmp_9021[2] + tmp_9023[2]];
    signal tmp_9025[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_9026[3] <== [5578449105840657656 * tmp_9025[0], 5578449105840657656 * tmp_9025[1], 5578449105840657656 * tmp_9025[2]];
    signal tmp_9027[3] <== [tmp_9024[0] + tmp_9026[0], tmp_9024[1] + tmp_9026[1], tmp_9024[2] + tmp_9026[2]];
    signal tmp_9028[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_9029[3] <== [16653803537962124952 * tmp_9028[0], 16653803537962124952 * tmp_9028[1], 16653803537962124952 * tmp_9028[2]];
    signal tmp_9030[3] <== [tmp_9027[0] + tmp_9029[0], tmp_9027[1] + tmp_9029[1], tmp_9027[2] + tmp_9029[2]];
    signal tmp_9031[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_9032[3] <== [14725357302690445139 * tmp_9031[0], 14725357302690445139 * tmp_9031[1], 14725357302690445139 * tmp_9031[2]];
    signal tmp_9033[3] <== [tmp_9030[0] + tmp_9032[0], tmp_9030[1] + tmp_9032[1], tmp_9030[2] + tmp_9032[2]];
    signal tmp_9034[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_9035[3] <== [13551115817015094532 * tmp_9034[0], 13551115817015094532 * tmp_9034[1], 13551115817015094532 * tmp_9034[2]];
    signal tmp_9036[3] <== [tmp_9033[0] + tmp_9035[0], tmp_9033[1] + tmp_9035[1], tmp_9033[2] + tmp_9035[2]];
    signal tmp_9037[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_9038[3] <== [9008401972144753939 * tmp_9037[0], 9008401972144753939 * tmp_9037[1], 9008401972144753939 * tmp_9037[2]];
    signal tmp_9039[3] <== [tmp_9036[0] + tmp_9038[0], tmp_9036[1] + tmp_9038[1], tmp_9036[2] + tmp_9038[2]];
    signal tmp_9040[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_9041[3] <== [16204753907425352648 * tmp_9040[0], 16204753907425352648 * tmp_9040[1], 16204753907425352648 * tmp_9040[2]];
    signal tmp_9042[3] <== [tmp_9039[0] + tmp_9041[0], tmp_9039[1] + tmp_9041[1], tmp_9039[2] + tmp_9041[2]];
    signal tmp_9043[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_9044[3] <== [7366465902859006190 * tmp_9043[0], 7366465902859006190 * tmp_9043[1], 7366465902859006190 * tmp_9043[2]];
    signal tmp_9045[3] <== [tmp_9042[0] + tmp_9044[0], tmp_9042[1] + tmp_9044[1], tmp_9042[2] + tmp_9044[2]];
    signal tmp_9046[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_9047[3] <== [15743243826515774520 * tmp_9046[0], 15743243826515774520 * tmp_9046[1], 15743243826515774520 * tmp_9046[2]];
    signal tmp_9048[3] <== [tmp_9045[0] + tmp_9047[0], tmp_9045[1] + tmp_9047[1], tmp_9045[2] + tmp_9047[2]];
    signal tmp_9049[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_9050[3] <== [7700354089512536567 * tmp_9049[0], 7700354089512536567 * tmp_9049[1], 7700354089512536567 * tmp_9049[2]];
    signal tmp_9051[3] <== [tmp_9048[0] + tmp_9050[0], tmp_9048[1] + tmp_9050[1], tmp_9048[2] + tmp_9050[2]];
    signal tmp_9052[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_9053[3] <== [1582298761054006274 * tmp_9052[0], 1582298761054006274 * tmp_9052[1], 1582298761054006274 * tmp_9052[2]];
    signal tmp_9054[3] <== [tmp_9051[0] + tmp_9053[0], tmp_9051[1] + tmp_9053[1], tmp_9051[2] + tmp_9053[2]];
    signal tmp_9055[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_9056[3] <== [17611042948119270535 * tmp_9055[0], 17611042948119270535 * tmp_9055[1], 17611042948119270535 * tmp_9055[2]];
    signal tmp_9057[3] <== [tmp_9054[0] + tmp_9056[0], tmp_9054[1] + tmp_9056[1], tmp_9054[2] + tmp_9056[2]];
    signal tmp_9058[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_9059[3] <== [18408474278105675815 * tmp_9058[0], 18408474278105675815 * tmp_9058[1], 18408474278105675815 * tmp_9058[2]];
    signal tmp_9060[3] <== [tmp_9057[0] + tmp_9059[0], tmp_9057[1] + tmp_9059[1], tmp_9057[2] + tmp_9059[2]];
    signal tmp_9061[3] <== [evals[130][0] - tmp_9060[0], evals[130][1] - tmp_9060[1], evals[130][2] - tmp_9060[2]];
    tmp_9062 <== CMul()(tmp_6066, tmp_9061);
}

template VerifyEvaluationsChunks3() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_6066[3];
    signal input tmp_6068[3];
    signal input tmp_7423[3];
    signal input tmp_7428[3];
    signal input tmp_7433[3];
    signal input tmp_7438[3];
    signal input tmp_7444[3];
    signal input tmp_7449[3];
    signal input tmp_7456[3];
    signal input tmp_7461[3];
    signal input tmp_7467[3];
    signal input tmp_7472[3];
    signal input tmp_7479[3];
    signal input tmp_7484[3];
    signal input tmp_7491[3];
    signal input tmp_7496[3];
    signal input tmp_7503[3];
    signal input tmp_7508[3];
    signal input tmp_7514[3];
    signal input tmp_7519[3];
    signal input tmp_7526[3];
    signal input tmp_7531[3];
    signal input tmp_7538[3];
    signal input tmp_7543[3];
    signal input tmp_7550[3];
    signal input tmp_7555[3];
    signal input tmp_7562[3];
    signal input tmp_7567[3];
    signal input tmp_7574[3];
    signal input tmp_7579[3];
    signal input tmp_7586[3];
    signal input tmp_7591[3];
    signal input tmp_7598[3];
    signal input tmp_7603[3];
    signal input tmp_9013[3];
    signal input tmp_9062[3];

    signal output tmp_9986[3];
    signal output tmp_9990[3];
    signal output tmp_9995[3];
    signal output tmp_10000[3];
    signal output tmp_10005[3];
    signal output tmp_10010[3];
    signal output tmp_10015[3];
    signal output tmp_10020[3];
    signal output tmp_10025[3];
    signal output tmp_10030[3];
    signal output tmp_10035[3];
    signal output tmp_10040[3];
    signal output tmp_10045[3];
    signal output tmp_10050[3];
    signal output tmp_10055[3];
    signal output tmp_10057[3];
    signal output tmp_10060[3];
    signal output tmp_10061[3];
    signal tmp_9063[3] <== [tmp_9013[0] + tmp_9062[0], tmp_9013[1] + tmp_9062[1], tmp_9013[2] + tmp_9062[2]];
    signal tmp_9064[3] <== CMul()(challengeQ, tmp_9063);
    signal tmp_9065[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_9066[3] <== [8024158912291823733 * tmp_9065[0], 8024158912291823733 * tmp_9065[1], 8024158912291823733 * tmp_9065[2]];
    signal tmp_9067[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_9068[3] <== [7624167817472438013 * tmp_9067[0], 7624167817472438013 * tmp_9067[1], 7624167817472438013 * tmp_9067[2]];
    signal tmp_9069[3] <== [tmp_9066[0] + tmp_9068[0], tmp_9066[1] + tmp_9068[1], tmp_9066[2] + tmp_9068[2]];
    signal tmp_9070[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_9071[3] <== [14208464745322834529 * tmp_9070[0], 14208464745322834529 * tmp_9070[1], 14208464745322834529 * tmp_9070[2]];
    signal tmp_9072[3] <== [tmp_9069[0] + tmp_9071[0], tmp_9069[1] + tmp_9071[1], tmp_9069[2] + tmp_9071[2]];
    signal tmp_9073[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_9074[3] <== [812825342001813471 * tmp_9073[0], 812825342001813471 * tmp_9073[1], 812825342001813471 * tmp_9073[2]];
    signal tmp_9075[3] <== [tmp_9072[0] + tmp_9074[0], tmp_9072[1] + tmp_9074[1], tmp_9072[2] + tmp_9074[2]];
    signal tmp_9076[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_9077[3] <== [3628630143580991743 * tmp_9076[0], 3628630143580991743 * tmp_9076[1], 3628630143580991743 * tmp_9076[2]];
    signal tmp_9078[3] <== [tmp_9075[0] + tmp_9077[0], tmp_9075[1] + tmp_9077[1], tmp_9075[2] + tmp_9077[2]];
    signal tmp_9079[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_9080[3] <== [17138871444129789158 * tmp_9079[0], 17138871444129789158 * tmp_9079[1], 17138871444129789158 * tmp_9079[2]];
    signal tmp_9081[3] <== [tmp_9078[0] + tmp_9080[0], tmp_9078[1] + tmp_9080[1], tmp_9078[2] + tmp_9080[2]];
    signal tmp_9082[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_9083[3] <== [12361291211423778677 * tmp_9082[0], 12361291211423778677 * tmp_9082[1], 12361291211423778677 * tmp_9082[2]];
    signal tmp_9084[3] <== [tmp_9081[0] + tmp_9083[0], tmp_9081[1] + tmp_9083[1], tmp_9081[2] + tmp_9083[2]];
    signal tmp_9085[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_9086[3] <== [9707166089812971789 * tmp_9085[0], 9707166089812971789 * tmp_9085[1], 9707166089812971789 * tmp_9085[2]];
    signal tmp_9087[3] <== [tmp_9084[0] + tmp_9086[0], tmp_9084[1] + tmp_9086[1], tmp_9084[2] + tmp_9086[2]];
    signal tmp_9088[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_9089[3] <== [16073073456050006575 * tmp_9088[0], 16073073456050006575 * tmp_9088[1], 16073073456050006575 * tmp_9088[2]];
    signal tmp_9090[3] <== [tmp_9087[0] + tmp_9089[0], tmp_9087[1] + tmp_9089[1], tmp_9087[2] + tmp_9089[2]];
    signal tmp_9091[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_9092[3] <== [16359376528724855414 * tmp_9091[0], 16359376528724855414 * tmp_9091[1], 16359376528724855414 * tmp_9091[2]];
    signal tmp_9093[3] <== [tmp_9090[0] + tmp_9092[0], tmp_9090[1] + tmp_9092[1], tmp_9090[2] + tmp_9092[2]];
    signal tmp_9094[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_9095[3] <== [2680431002152629139 * tmp_9094[0], 2680431002152629139 * tmp_9094[1], 2680431002152629139 * tmp_9094[2]];
    signal tmp_9096[3] <== [tmp_9093[0] + tmp_9095[0], tmp_9093[1] + tmp_9095[1], tmp_9093[2] + tmp_9095[2]];
    signal tmp_9097[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_9098[3] <== [15175373777650174966 * tmp_9097[0], 15175373777650174966 * tmp_9097[1], 15175373777650174966 * tmp_9097[2]];
    signal tmp_9099[3] <== [tmp_9096[0] + tmp_9098[0], tmp_9096[1] + tmp_9098[1], tmp_9096[2] + tmp_9098[2]];
    signal tmp_9100[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_9101[3] <== [7595102312128106020 * tmp_9100[0], 7595102312128106020 * tmp_9100[1], 7595102312128106020 * tmp_9100[2]];
    signal tmp_9102[3] <== [tmp_9099[0] + tmp_9101[0], tmp_9099[1] + tmp_9101[1], tmp_9099[2] + tmp_9101[2]];
    signal tmp_9103[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_9104[3] <== [17245004066890521540 * tmp_9103[0], 17245004066890521540 * tmp_9103[1], 17245004066890521540 * tmp_9103[2]];
    signal tmp_9105[3] <== [tmp_9102[0] + tmp_9104[0], tmp_9102[1] + tmp_9104[1], tmp_9102[2] + tmp_9104[2]];
    signal tmp_9106[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_9107[3] <== [14226032251042260572 * tmp_9106[0], 14226032251042260572 * tmp_9106[1], 14226032251042260572 * tmp_9106[2]];
    signal tmp_9108[3] <== [tmp_9105[0] + tmp_9107[0], tmp_9105[1] + tmp_9107[1], tmp_9105[2] + tmp_9107[2]];
    signal tmp_9109[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_9110[3] <== [10290793996632644659 * tmp_9109[0], 10290793996632644659 * tmp_9109[1], 10290793996632644659 * tmp_9109[2]];
    signal tmp_9111[3] <== [tmp_9108[0] + tmp_9110[0], tmp_9108[1] + tmp_9110[1], tmp_9108[2] + tmp_9110[2]];
    signal tmp_9112[3] <== [evals[131][0] - tmp_9111[0], evals[131][1] - tmp_9111[1], evals[131][2] - tmp_9111[2]];
    signal tmp_9113[3] <== CMul()(tmp_6066, tmp_9112);
    signal tmp_9114[3] <== [tmp_9064[0] + tmp_9113[0], tmp_9064[1] + tmp_9113[1], tmp_9064[2] + tmp_9113[2]];
    signal tmp_9115[3] <== CMul()(challengeQ, tmp_9114);
    signal tmp_9116[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_9117[3] <== [9638737643153160474 * tmp_9116[0], 9638737643153160474 * tmp_9116[1], 9638737643153160474 * tmp_9116[2]];
    signal tmp_9118[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_9119[3] <== [16887945882953670782 * tmp_9118[0], 16887945882953670782 * tmp_9118[1], 16887945882953670782 * tmp_9118[2]];
    signal tmp_9120[3] <== [tmp_9117[0] + tmp_9119[0], tmp_9117[1] + tmp_9119[1], tmp_9117[2] + tmp_9119[2]];
    signal tmp_9121[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_9122[3] <== [11859617026755975295 * tmp_9121[0], 11859617026755975295 * tmp_9121[1], 11859617026755975295 * tmp_9121[2]];
    signal tmp_9123[3] <== [tmp_9120[0] + tmp_9122[0], tmp_9120[1] + tmp_9122[1], tmp_9120[2] + tmp_9122[2]];
    signal tmp_9124[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_9125[3] <== [14208464745322834529 * tmp_9124[0], 14208464745322834529 * tmp_9124[1], 14208464745322834529 * tmp_9124[2]];
    signal tmp_9126[3] <== [tmp_9123[0] + tmp_9125[0], tmp_9123[1] + tmp_9125[1], tmp_9123[2] + tmp_9125[2]];
    signal tmp_9127[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_9128[3] <== [14728006497068635074 * tmp_9127[0], 14728006497068635074 * tmp_9127[1], 14728006497068635074 * tmp_9127[2]];
    signal tmp_9129[3] <== [tmp_9126[0] + tmp_9128[0], tmp_9126[1] + tmp_9128[1], tmp_9126[2] + tmp_9128[2]];
    signal tmp_9130[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_9131[3] <== [7063454713635591383 * tmp_9130[0], 7063454713635591383 * tmp_9130[1], 7063454713635591383 * tmp_9130[2]];
    signal tmp_9132[3] <== [tmp_9129[0] + tmp_9131[0], tmp_9129[1] + tmp_9131[1], tmp_9129[2] + tmp_9131[2]];
    signal tmp_9133[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_9134[3] <== [10449762191515258658 * tmp_9133[0], 10449762191515258658 * tmp_9133[1], 10449762191515258658 * tmp_9133[2]];
    signal tmp_9135[3] <== [tmp_9132[0] + tmp_9134[0], tmp_9132[1] + tmp_9134[1], tmp_9132[2] + tmp_9134[2]];
    signal tmp_9136[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_9137[3] <== [5269705790554862459 * tmp_9136[0], 5269705790554862459 * tmp_9136[1], 5269705790554862459 * tmp_9136[2]];
    signal tmp_9138[3] <== [tmp_9135[0] + tmp_9137[0], tmp_9135[1] + tmp_9137[1], tmp_9135[2] + tmp_9137[2]];
    signal tmp_9139[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_9140[3] <== [10542619009725154399 * tmp_9139[0], 10542619009725154399 * tmp_9139[1], 10542619009725154399 * tmp_9139[2]];
    signal tmp_9141[3] <== [tmp_9138[0] + tmp_9140[0], tmp_9138[1] + tmp_9140[1], tmp_9138[2] + tmp_9140[2]];
    signal tmp_9142[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_9143[3] <== [13074116889890804777 * tmp_9142[0], 13074116889890804777 * tmp_9142[1], 13074116889890804777 * tmp_9142[2]];
    signal tmp_9144[3] <== [tmp_9141[0] + tmp_9143[0], tmp_9141[1] + tmp_9143[1], tmp_9141[2] + tmp_9143[2]];
    signal tmp_9145[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_9146[3] <== [5823573138716677219 * tmp_9145[0], 5823573138716677219 * tmp_9145[1], 5823573138716677219 * tmp_9145[2]];
    signal tmp_9147[3] <== [tmp_9144[0] + tmp_9146[0], tmp_9144[1] + tmp_9146[1], tmp_9144[2] + tmp_9146[2]];
    signal tmp_9148[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_9149[3] <== [4800348490672825001 * tmp_9148[0], 4800348490672825001 * tmp_9148[1], 4800348490672825001 * tmp_9148[2]];
    signal tmp_9150[3] <== [tmp_9147[0] + tmp_9149[0], tmp_9147[1] + tmp_9149[1], tmp_9147[2] + tmp_9149[2]];
    signal tmp_9151[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_9152[3] <== [15832068917340433117 * tmp_9151[0], 15832068917340433117 * tmp_9151[1], 15832068917340433117 * tmp_9151[2]];
    signal tmp_9153[3] <== [tmp_9150[0] + tmp_9152[0], tmp_9150[1] + tmp_9152[1], tmp_9150[2] + tmp_9152[2]];
    signal tmp_9154[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_9155[3] <== [10872182090012722023 * tmp_9154[0], 10872182090012722023 * tmp_9154[1], 10872182090012722023 * tmp_9154[2]];
    signal tmp_9156[3] <== [tmp_9153[0] + tmp_9155[0], tmp_9153[1] + tmp_9155[1], tmp_9153[2] + tmp_9155[2]];
    signal tmp_9157[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_9158[3] <== [7891351762273827007 * tmp_9157[0], 7891351762273827007 * tmp_9157[1], 7891351762273827007 * tmp_9157[2]];
    signal tmp_9159[3] <== [tmp_9156[0] + tmp_9158[0], tmp_9156[1] + tmp_9158[1], tmp_9156[2] + tmp_9158[2]];
    signal tmp_9160[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_9161[3] <== [8593802027805519400 * tmp_9160[0], 8593802027805519400 * tmp_9160[1], 8593802027805519400 * tmp_9160[2]];
    signal tmp_9162[3] <== [tmp_9159[0] + tmp_9161[0], tmp_9159[1] + tmp_9161[1], tmp_9159[2] + tmp_9161[2]];
    signal tmp_9163[3] <== [evals[132][0] - tmp_9162[0], evals[132][1] - tmp_9162[1], evals[132][2] - tmp_9162[2]];
    signal tmp_9164[3] <== CMul()(tmp_6066, tmp_9163);
    signal tmp_9165[3] <== [tmp_9115[0] + tmp_9164[0], tmp_9115[1] + tmp_9164[1], tmp_9115[2] + tmp_9164[2]];
    signal tmp_9166[3] <== CMul()(challengeQ, tmp_9165);
    signal tmp_9167[3] <== [tmp_7423[0] + tmp_7428[0], tmp_7423[1] + tmp_7428[1], tmp_7423[2] + tmp_7428[2]];
    signal tmp_9168[3] <== [15737625755013050897 * tmp_9167[0], 15737625755013050897 * tmp_9167[1], 15737625755013050897 * tmp_9167[2]];
    signal tmp_9169[3] <== [tmp_7433[0] + tmp_7438[0], tmp_7433[1] + tmp_7438[1], tmp_7433[2] + tmp_7438[2]];
    signal tmp_9170[3] <== [6288199092898320258 * tmp_9169[0], 6288199092898320258 * tmp_9169[1], 6288199092898320258 * tmp_9169[2]];
    signal tmp_9171[3] <== [tmp_9168[0] + tmp_9170[0], tmp_9168[1] + tmp_9170[1], tmp_9168[2] + tmp_9170[2]];
    signal tmp_9172[3] <== [tmp_7444[0] + tmp_7449[0], tmp_7444[1] + tmp_7449[1], tmp_7444[2] + tmp_7449[2]];
    signal tmp_9173[3] <== [16887945882953670782 * tmp_9172[0], 16887945882953670782 * tmp_9172[1], 16887945882953670782 * tmp_9172[2]];
    signal tmp_9174[3] <== [tmp_9171[0] + tmp_9173[0], tmp_9171[1] + tmp_9173[1], tmp_9171[2] + tmp_9173[2]];
    signal tmp_9175[3] <== [tmp_7456[0] + tmp_7461[0], tmp_7456[1] + tmp_7461[1], tmp_7456[2] + tmp_7461[2]];
    signal tmp_9176[3] <== [7624167817472438013 * tmp_9175[0], 7624167817472438013 * tmp_9175[1], 7624167817472438013 * tmp_9175[2]];
    signal tmp_9177[3] <== [tmp_9174[0] + tmp_9176[0], tmp_9174[1] + tmp_9176[1], tmp_9174[2] + tmp_9176[2]];
    signal tmp_9178[3] <== [tmp_7467[0] + tmp_7472[0], tmp_7467[1] + tmp_7472[1], tmp_7467[2] + tmp_7472[2]];
    signal tmp_9179[3] <== [3252585408322091368 * tmp_9178[0], 3252585408322091368 * tmp_9178[1], 3252585408322091368 * tmp_9178[2]];
    signal tmp_9180[3] <== [tmp_9177[0] + tmp_9179[0], tmp_9177[1] + tmp_9179[1], tmp_9177[2] + tmp_9179[2]];
    signal tmp_9181[3] <== [tmp_7479[0] + tmp_7484[0], tmp_7479[1] + tmp_7484[1], tmp_7479[2] + tmp_7484[2]];
    signal tmp_9182[3] <== [2525078022317723924 * tmp_9181[0], 2525078022317723924 * tmp_9181[1], 2525078022317723924 * tmp_9181[2]];
    signal tmp_9183[3] <== [tmp_9180[0] + tmp_9182[0], tmp_9180[1] + tmp_9182[1], tmp_9180[2] + tmp_9182[2]];
    signal tmp_9184[3] <== [tmp_7491[0] + tmp_7496[0], tmp_7491[1] + tmp_7496[1], tmp_7491[2] + tmp_7496[2]];
    signal tmp_9185[3] <== [14505687488637668062 * tmp_9184[0], 14505687488637668062 * tmp_9184[1], 14505687488637668062 * tmp_9184[2]];
    signal tmp_9186[3] <== [tmp_9183[0] + tmp_9185[0], tmp_9183[1] + tmp_9185[1], tmp_9183[2] + tmp_9185[2]];
    signal tmp_9187[3] <== [tmp_7503[0] + tmp_7508[0], tmp_7503[1] + tmp_7508[1], tmp_7503[2] + tmp_7508[2]];
    signal tmp_9188[3] <== [2360357678915503787 * tmp_9187[0], 2360357678915503787 * tmp_9187[1], 2360357678915503787 * tmp_9187[2]];
    signal tmp_9189[3] <== [tmp_9186[0] + tmp_9188[0], tmp_9186[1] + tmp_9188[1], tmp_9186[2] + tmp_9188[2]];
    signal tmp_9190[3] <== [tmp_7514[0] + tmp_7519[0], tmp_7514[1] + tmp_7519[1], tmp_7514[2] + tmp_7519[2]];
    signal tmp_9191[3] <== [16995162224826066164 * tmp_9190[0], 16995162224826066164 * tmp_9190[1], 16995162224826066164 * tmp_9190[2]];
    signal tmp_9192[3] <== [tmp_9189[0] + tmp_9191[0], tmp_9189[1] + tmp_9191[1], tmp_9189[2] + tmp_9191[2]];
    signal tmp_9193[3] <== [tmp_7526[0] + tmp_7531[0], tmp_7526[1] + tmp_7531[1], tmp_7526[2] + tmp_7531[2]];
    signal tmp_9194[3] <== [3473665614188209191 * tmp_9193[0], 3473665614188209191 * tmp_9193[1], 3473665614188209191 * tmp_9193[2]];
    signal tmp_9195[3] <== [tmp_9192[0] + tmp_9194[0], tmp_9192[1] + tmp_9194[1], tmp_9192[2] + tmp_9194[2]];
    signal tmp_9196[3] <== [tmp_7538[0] + tmp_7543[0], tmp_7538[1] + tmp_7543[1], tmp_7538[2] + tmp_7543[2]];
    signal tmp_9197[3] <== [6296951922972871295 * tmp_9196[0], 6296951922972871295 * tmp_9196[1], 6296951922972871295 * tmp_9196[2]];
    signal tmp_9198[3] <== [tmp_9195[0] + tmp_9197[0], tmp_9195[1] + tmp_9197[1], tmp_9195[2] + tmp_9197[2]];
    signal tmp_9199[3] <== [tmp_7550[0] + tmp_7555[0], tmp_7550[1] + tmp_7555[1], tmp_7550[2] + tmp_7555[2]];
    signal tmp_9200[3] <== [1843095456038194143 * tmp_9199[0], 1843095456038194143 * tmp_9199[1], 1843095456038194143 * tmp_9199[2]];
    signal tmp_9201[3] <== [tmp_9198[0] + tmp_9200[0], tmp_9198[1] + tmp_9200[1], tmp_9198[2] + tmp_9200[2]];
    signal tmp_9202[3] <== [tmp_7562[0] + tmp_7567[0], tmp_7562[1] + tmp_7567[1], tmp_7562[2] + tmp_7567[2]];
    signal tmp_9203[3] <== [3224171887150947390 * tmp_9202[0], 3224171887150947390 * tmp_9202[1], 3224171887150947390 * tmp_9202[2]];
    signal tmp_9204[3] <== [tmp_9201[0] + tmp_9203[0], tmp_9201[1] + tmp_9203[1], tmp_9201[2] + tmp_9203[2]];
    signal tmp_9205[3] <== [tmp_7574[0] + tmp_7579[0], tmp_7574[1] + tmp_7579[1], tmp_7574[2] + tmp_7579[2]];
    signal tmp_9206[3] <== [8855639374359979570 * tmp_9205[0], 8855639374359979570 * tmp_9205[1], 8855639374359979570 * tmp_9205[2]];
    signal tmp_9207[3] <== [tmp_9204[0] + tmp_9206[0], tmp_9204[1] + tmp_9206[1], tmp_9204[2] + tmp_9206[2]];
    signal tmp_9208[3] <== [tmp_7586[0] + tmp_7591[0], tmp_7586[1] + tmp_7591[1], tmp_7586[2] + tmp_7591[2]];
    signal tmp_9209[3] <== [17521923232000336525 * tmp_9208[0], 17521923232000336525 * tmp_9208[1], 17521923232000336525 * tmp_9208[2]];
    signal tmp_9210[3] <== [tmp_9207[0] + tmp_9209[0], tmp_9207[1] + tmp_9209[1], tmp_9207[2] + tmp_9209[2]];
    signal tmp_9211[3] <== [tmp_7598[0] + tmp_7603[0], tmp_7598[1] + tmp_7603[1], tmp_7598[2] + tmp_7603[2]];
    signal tmp_9212[3] <== [4068529056800825848 * tmp_9211[0], 4068529056800825848 * tmp_9211[1], 4068529056800825848 * tmp_9211[2]];
    signal tmp_9213[3] <== [tmp_9210[0] + tmp_9212[0], tmp_9210[1] + tmp_9212[1], tmp_9210[2] + tmp_9212[2]];
    signal tmp_9214[3] <== [evals[133][0] - tmp_9213[0], evals[133][1] - tmp_9213[1], evals[133][2] - tmp_9213[2]];
    signal tmp_9215[3] <== CMul()(tmp_6066, tmp_9214);
    signal tmp_9216[3] <== [tmp_9166[0] + tmp_9215[0], tmp_9166[1] + tmp_9215[1], tmp_9166[2] + tmp_9215[2]];
    signal tmp_9217[3] <== CMul()(challengeQ, tmp_9216);
    signal tmp_9218[3] <== [evals[82][0] - evals[66][0], evals[82][1] - evals[66][1], evals[82][2] - evals[66][2]];
    signal tmp_9219[3] <== CMul()(tmp_6068, tmp_9218);
    signal tmp_9220[3] <== [tmp_9217[0] + tmp_9219[0], tmp_9217[1] + tmp_9219[1], tmp_9217[2] + tmp_9219[2]];
    signal tmp_9221[3] <== CMul()(challengeQ, tmp_9220);
    signal tmp_9222[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9223[3] <== [8669423204701701828 * evals[67][0], 8669423204701701828 * evals[67][1], 8669423204701701828 * evals[67][2]];
    signal tmp_9224[3] <== [tmp_9222[0] + tmp_9223[0], tmp_9222[1] + tmp_9223[1], tmp_9222[2] + tmp_9223[2]];
    signal tmp_9225[3] <== [11218806973091596230 * evals[68][0], 11218806973091596230 * evals[68][1], 11218806973091596230 * evals[68][2]];
    signal tmp_9226[3] <== [tmp_9224[0] + tmp_9225[0], tmp_9224[1] + tmp_9225[1], tmp_9224[2] + tmp_9225[2]];
    signal tmp_9227[3] <== [15262373047951269198 * evals[69][0], 15262373047951269198 * evals[69][1], 15262373047951269198 * evals[69][2]];
    signal tmp_9228[3] <== [tmp_9226[0] + tmp_9227[0], tmp_9226[1] + tmp_9227[1], tmp_9226[2] + tmp_9227[2]];
    signal tmp_9229[3] <== [1029025466846055820 * evals[70][0], 1029025466846055820 * evals[70][1], 1029025466846055820 * evals[70][2]];
    signal tmp_9230[3] <== [tmp_9228[0] + tmp_9229[0], tmp_9228[1] + tmp_9229[1], tmp_9228[2] + tmp_9229[2]];
    signal tmp_9231[3] <== [4974512540632580574 * evals[71][0], 4974512540632580574 * evals[71][1], 4974512540632580574 * evals[71][2]];
    signal tmp_9232[3] <== [tmp_9230[0] + tmp_9231[0], tmp_9230[1] + tmp_9231[1], tmp_9230[2] + tmp_9231[2]];
    signal tmp_9233[3] <== [9182455040231019020 * evals[72][0], 9182455040231019020 * evals[72][1], 9182455040231019020 * evals[72][2]];
    signal tmp_9234[3] <== [tmp_9232[0] + tmp_9233[0], tmp_9232[1] + tmp_9233[1], tmp_9232[2] + tmp_9233[2]];
    signal tmp_9235[3] <== [6370260108846926617 * evals[73][0], 6370260108846926617 * evals[73][1], 6370260108846926617 * evals[73][2]];
    signal tmp_9236[3] <== [tmp_9234[0] + tmp_9235[0], tmp_9234[1] + tmp_9235[1], tmp_9234[2] + tmp_9235[2]];
    signal tmp_9237[3] <== [13260731818101756883 * evals[74][0], 13260731818101756883 * evals[74][1], 13260731818101756883 * evals[74][2]];
    signal tmp_9238[3] <== [tmp_9236[0] + tmp_9237[0], tmp_9236[1] + tmp_9237[1], tmp_9236[2] + tmp_9237[2]];
    signal tmp_9239[3] <== [7710124073693265452 * evals[75][0], 7710124073693265452 * evals[75][1], 7710124073693265452 * evals[75][2]];
    signal tmp_9240[3] <== [tmp_9238[0] + tmp_9239[0], tmp_9238[1] + tmp_9239[1], tmp_9238[2] + tmp_9239[2]];
    signal tmp_9241[3] <== [15423058398499793658 * evals[76][0], 15423058398499793658 * evals[76][1], 15423058398499793658 * evals[76][2]];
    signal tmp_9242[3] <== [tmp_9240[0] + tmp_9241[0], tmp_9240[1] + tmp_9241[1], tmp_9240[2] + tmp_9241[2]];
    signal tmp_9243[3] <== [5050813558536356212 * evals[77][0], 5050813558536356212 * evals[77][1], 5050813558536356212 * evals[77][2]];
    signal tmp_9244[3] <== [tmp_9242[0] + tmp_9243[0], tmp_9242[1] + tmp_9243[1], tmp_9242[2] + tmp_9243[2]];
    signal tmp_9245[3] <== [7839532349345071368 * evals[78][0], 7839532349345071368 * evals[78][1], 7839532349345071368 * evals[78][2]];
    signal tmp_9246[3] <== [tmp_9244[0] + tmp_9245[0], tmp_9244[1] + tmp_9245[1], tmp_9244[2] + tmp_9245[2]];
    signal tmp_9247[3] <== [13946888797241243093 * evals[79][0], 13946888797241243093 * evals[79][1], 13946888797241243093 * evals[79][2]];
    signal tmp_9248[3] <== [tmp_9246[0] + tmp_9247[0], tmp_9246[1] + tmp_9247[1], tmp_9246[2] + tmp_9247[2]];
    signal tmp_9249[3] <== [1910481884837990028 * evals[80][0], 1910481884837990028 * evals[80][1], 1910481884837990028 * evals[80][2]];
    signal tmp_9250[3] <== [tmp_9248[0] + tmp_9249[0], tmp_9248[1] + tmp_9249[1], tmp_9248[2] + tmp_9249[2]];
    signal tmp_9251[3] <== [2968031798288424027 * evals[81][0], 2968031798288424027 * evals[81][1], 2968031798288424027 * evals[81][2]];
    signal tmp_9252[3] <== [tmp_9250[0] + tmp_9251[0], tmp_9250[1] + tmp_9251[1], tmp_9250[2] + tmp_9251[2]];
    signal tmp_9253[3] <== [tmp_9252[0] * 1, tmp_9252[1] * 1, tmp_9252[2] * 1];
    signal tmp_9254[3] <== [evals[83][0] - tmp_9253[0], evals[83][1] - tmp_9253[1], evals[83][2] - tmp_9253[2]];
    signal tmp_9255[3] <== CMul()(tmp_6068, tmp_9254);
    signal tmp_9256[3] <== [tmp_9221[0] + tmp_9255[0], tmp_9221[1] + tmp_9255[1], tmp_9221[2] + tmp_9255[2]];
    signal tmp_9257[3] <== CMul()(challengeQ, tmp_9256);
    signal tmp_9258[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9259[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9260[3] <== [tmp_9259[0] * 9426926376014476876, tmp_9259[1] * 9426926376014476876, tmp_9259[2] * 9426926376014476876];
    signal tmp_9261[3] <== [evals[67][0] + tmp_9260[0], evals[67][1] + tmp_9260[1], evals[67][2] + tmp_9260[2]];
    signal tmp_9262[3] <== [4769303276973324461 * tmp_9261[0], 4769303276973324461 * tmp_9261[1], 4769303276973324461 * tmp_9261[2]];
    signal tmp_9263[3] <== [tmp_9258[0] + tmp_9262[0], tmp_9258[1] + tmp_9262[1], tmp_9258[2] + tmp_9262[2]];
    signal tmp_9264[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9265[3] <== [tmp_9264[0] * 6636074244061498014, tmp_9264[1] * 6636074244061498014, tmp_9264[2] * 6636074244061498014];
    signal tmp_9266[3] <== [evals[68][0] + tmp_9265[0], evals[68][1] + tmp_9265[1], evals[68][2] + tmp_9265[2]];
    signal tmp_9267[3] <== [1784300138210151456 * tmp_9266[0], 1784300138210151456 * tmp_9266[1], 1784300138210151456 * tmp_9266[2]];
    signal tmp_9268[3] <== [tmp_9263[0] + tmp_9267[0], tmp_9263[1] + tmp_9267[1], tmp_9263[2] + tmp_9267[2]];
    signal tmp_9269[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9270[3] <== [tmp_9269[0] * 10264791899337939894, tmp_9269[1] * 10264791899337939894, tmp_9269[2] * 10264791899337939894];
    signal tmp_9271[3] <== [evals[69][0] + tmp_9270[0], evals[69][1] + tmp_9270[1], evals[69][2] + tmp_9270[2]];
    signal tmp_9272[3] <== [12324891265250247020 * tmp_9271[0], 12324891265250247020 * tmp_9271[1], 12324891265250247020 * tmp_9271[2]];
    signal tmp_9273[3] <== [tmp_9268[0] + tmp_9272[0], tmp_9268[1] + tmp_9272[1], tmp_9268[2] + tmp_9272[2]];
    signal tmp_9274[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9275[3] <== [tmp_9274[0] * 7084964930551891640, tmp_9274[1] * 7084964930551891640, tmp_9274[2] * 7084964930551891640];
    signal tmp_9276[3] <== [evals[70][0] + tmp_9275[0], evals[70][1] + tmp_9275[1], evals[70][2] + tmp_9275[2]];
    signal tmp_9277[3] <== [12174743494559533987 * tmp_9276[0], 12174743494559533987 * tmp_9276[1], 12174743494559533987 * tmp_9276[2]];
    signal tmp_9278[3] <== [tmp_9273[0] + tmp_9277[0], tmp_9273[1] + tmp_9277[1], tmp_9273[2] + tmp_9277[2]];
    signal tmp_9279[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9280[3] <== [tmp_9279[0] * 80533688234975742, tmp_9279[1] * 80533688234975742, tmp_9279[2] * 80533688234975742];
    signal tmp_9281[3] <== [evals[71][0] + tmp_9280[0], evals[71][1] + tmp_9280[1], evals[71][2] + tmp_9280[2]];
    signal tmp_9282[3] <== [8147334966940084329 * tmp_9281[0], 8147334966940084329 * tmp_9281[1], 8147334966940084329 * tmp_9281[2]];
    signal tmp_9283[3] <== [tmp_9278[0] + tmp_9282[0], tmp_9278[1] + tmp_9282[1], tmp_9278[2] + tmp_9282[2]];
    signal tmp_9284[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9285[3] <== [tmp_9284[0] * 14562009953595175769, tmp_9284[1] * 14562009953595175769, tmp_9284[2] * 14562009953595175769];
    signal tmp_9286[3] <== [evals[72][0] + tmp_9285[0], evals[72][1] + tmp_9285[1], evals[72][2] + tmp_9285[2]];
    signal tmp_9287[3] <== [16397648617955447197 * tmp_9286[0], 16397648617955447197 * tmp_9286[1], 16397648617955447197 * tmp_9286[2]];
    signal tmp_9288[3] <== [tmp_9283[0] + tmp_9287[0], tmp_9283[1] + tmp_9287[1], tmp_9283[2] + tmp_9287[2]];
    signal tmp_9289[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9290[3] <== [tmp_9289[0] * 326023128915630064, tmp_9289[1] * 326023128915630064, tmp_9289[2] * 326023128915630064];
    signal tmp_9291[3] <== [evals[73][0] + tmp_9290[0], evals[73][1] + tmp_9290[1], evals[73][2] + tmp_9290[2]];
    signal tmp_9292[3] <== [6037293384829534708 * tmp_9291[0], 6037293384829534708 * tmp_9291[1], 6037293384829534708 * tmp_9291[2]];
    signal tmp_9293[3] <== [tmp_9288[0] + tmp_9292[0], tmp_9288[1] + tmp_9292[1], tmp_9288[2] + tmp_9292[2]];
    signal tmp_9294[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9295[3] <== [tmp_9294[0] * 11214699076825281994, tmp_9294[1] * 11214699076825281994, tmp_9294[2] * 11214699076825281994];
    signal tmp_9296[3] <== [evals[74][0] + tmp_9295[0], evals[74][1] + tmp_9295[1], evals[74][2] + tmp_9295[2]];
    signal tmp_9297[3] <== [16763181795158500563 * tmp_9296[0], 16763181795158500563 * tmp_9296[1], 16763181795158500563 * tmp_9296[2]];
    signal tmp_9298[3] <== [tmp_9293[0] + tmp_9297[0], tmp_9293[1] + tmp_9297[1], tmp_9293[2] + tmp_9297[2]];
    signal tmp_9299[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9300[3] <== [tmp_9299[0] * 12885915700130647489, tmp_9299[1] * 12885915700130647489, tmp_9299[2] * 12885915700130647489];
    signal tmp_9301[3] <== [evals[75][0] + tmp_9300[0], evals[75][1] + tmp_9300[1], evals[75][2] + tmp_9300[2]];
    signal tmp_9302[3] <== [10796483624199473587 * tmp_9301[0], 10796483624199473587 * tmp_9301[1], 10796483624199473587 * tmp_9301[2]];
    signal tmp_9303[3] <== [tmp_9298[0] + tmp_9302[0], tmp_9298[1] + tmp_9302[1], tmp_9298[2] + tmp_9302[2]];
    signal tmp_9304[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9305[3] <== [tmp_9304[0] * 2933396631256123616, tmp_9304[1] * 2933396631256123616, tmp_9304[2] * 2933396631256123616];
    signal tmp_9306[3] <== [evals[76][0] + tmp_9305[0], evals[76][1] + tmp_9305[1], evals[76][2] + tmp_9305[2]];
    signal tmp_9307[3] <== [9857846934271670796 * tmp_9306[0], 9857846934271670796 * tmp_9306[1], 9857846934271670796 * tmp_9306[2]];
    signal tmp_9308[3] <== [tmp_9303[0] + tmp_9307[0], tmp_9303[1] + tmp_9307[1], tmp_9303[2] + tmp_9307[2]];
    signal tmp_9309[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9310[3] <== [tmp_9309[0] * 254842111671127473, tmp_9309[1] * 254842111671127473, tmp_9309[2] * 254842111671127473];
    signal tmp_9311[3] <== [evals[77][0] + tmp_9310[0], evals[77][1] + tmp_9310[1], evals[77][2] + tmp_9310[2]];
    signal tmp_9312[3] <== [1711698425798584877 * tmp_9311[0], 1711698425798584877 * tmp_9311[1], 1711698425798584877 * tmp_9311[2]];
    signal tmp_9313[3] <== [tmp_9308[0] + tmp_9312[0], tmp_9308[1] + tmp_9312[1], tmp_9308[2] + tmp_9312[2]];
    signal tmp_9314[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9315[3] <== [tmp_9314[0] * 15960520344296920510, tmp_9314[1] * 15960520344296920510, tmp_9314[2] * 15960520344296920510];
    signal tmp_9316[3] <== [evals[78][0] + tmp_9315[0], evals[78][1] + tmp_9315[1], evals[78][2] + tmp_9315[2]];
    signal tmp_9317[3] <== [14317037178658024350 * tmp_9316[0], 14317037178658024350 * tmp_9316[1], 14317037178658024350 * tmp_9316[2]];
    signal tmp_9318[3] <== [tmp_9313[0] + tmp_9317[0], tmp_9313[1] + tmp_9317[1], tmp_9313[2] + tmp_9317[2]];
    signal tmp_9319[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9320[3] <== [tmp_9319[0] * 472404601837029917, tmp_9319[1] * 472404601837029917, tmp_9319[2] * 472404601837029917];
    signal tmp_9321[3] <== [evals[79][0] + tmp_9320[0], evals[79][1] + tmp_9320[1], evals[79][2] + tmp_9320[2]];
    signal tmp_9322[3] <== [4727210660209380191 * tmp_9321[0], 4727210660209380191 * tmp_9321[1], 4727210660209380191 * tmp_9321[2]];
    signal tmp_9323[3] <== [tmp_9318[0] + tmp_9322[0], tmp_9318[1] + tmp_9322[1], tmp_9318[2] + tmp_9322[2]];
    signal tmp_9324[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9325[3] <== [tmp_9324[0] * 5788109941016889361, tmp_9324[1] * 5788109941016889361, tmp_9324[2] * 5788109941016889361];
    signal tmp_9326[3] <== [evals[80][0] + tmp_9325[0], evals[80][1] + tmp_9325[1], evals[80][2] + tmp_9325[2]];
    signal tmp_9327[3] <== [4043120510603784989 * tmp_9326[0], 4043120510603784989 * tmp_9326[1], 4043120510603784989 * tmp_9326[2]];
    signal tmp_9328[3] <== [tmp_9323[0] + tmp_9327[0], tmp_9323[1] + tmp_9327[1], tmp_9323[2] + tmp_9327[2]];
    signal tmp_9329[3] <== [tmp_7423[0] + 8111169986958259496, tmp_7423[1], tmp_7423[2]];
    signal tmp_9330[3] <== [tmp_9329[0] * 15962084738108841624, tmp_9329[1] * 15962084738108841624, tmp_9329[2] * 15962084738108841624];
    signal tmp_9331[3] <== [evals[81][0] + tmp_9330[0], evals[81][1] + tmp_9330[1], evals[81][2] + tmp_9330[2]];
    signal tmp_9332[3] <== [6414177337645848225 * tmp_9331[0], 6414177337645848225 * tmp_9331[1], 6414177337645848225 * tmp_9331[2]];
    signal tmp_9333[3] <== [tmp_9328[0] + tmp_9332[0], tmp_9328[1] + tmp_9332[1], tmp_9328[2] + tmp_9332[2]];
    signal tmp_9334[3] <== [tmp_9333[0] * 1, tmp_9333[1] * 1, tmp_9333[2] * 1];
    signal tmp_9335[3] <== [evals[84][0] - tmp_9334[0], evals[84][1] - tmp_9334[1], evals[84][2] - tmp_9334[2]];
    signal tmp_9336[3] <== CMul()(tmp_6068, tmp_9335);
    signal tmp_9337[3] <== [tmp_9257[0] + tmp_9336[0], tmp_9257[1] + tmp_9336[1], tmp_9257[2] + tmp_9336[2]];
    signal tmp_9338[3] <== CMul()(challengeQ, tmp_9337);
    signal tmp_9339[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9340[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9341[3] <== [tmp_9340[0] * 8808574163396808168, tmp_9340[1] * 8808574163396808168, tmp_9340[2] * 8808574163396808168];
    signal tmp_9342[3] <== [tmp_9261[0] + tmp_9341[0], tmp_9261[1] + tmp_9341[1], tmp_9261[2] + tmp_9341[2]];
    signal tmp_9343[3] <== [1514946122061859950 * tmp_9342[0], 1514946122061859950 * tmp_9342[1], 1514946122061859950 * tmp_9342[2]];
    signal tmp_9344[3] <== [tmp_9339[0] + tmp_9343[0], tmp_9339[1] + tmp_9343[1], tmp_9339[2] + tmp_9343[2]];
    signal tmp_9345[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9346[3] <== [tmp_9345[0] * 2762212780120881912, tmp_9345[1] * 2762212780120881912, tmp_9345[2] * 2762212780120881912];
    signal tmp_9347[3] <== [tmp_9266[0] + tmp_9346[0], tmp_9266[1] + tmp_9346[1], tmp_9266[2] + tmp_9346[2]];
    signal tmp_9348[3] <== [1955599193479362137 * tmp_9347[0], 1955599193479362137 * tmp_9347[1], 1955599193479362137 * tmp_9347[2]];
    signal tmp_9349[3] <== [tmp_9344[0] + tmp_9348[0], tmp_9344[1] + tmp_9348[1], tmp_9344[2] + tmp_9348[2]];
    signal tmp_9350[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9351[3] <== [tmp_9350[0] * 17951693132780115109, tmp_9350[1] * 17951693132780115109, tmp_9350[2] * 17951693132780115109];
    signal tmp_9352[3] <== [tmp_9271[0] + tmp_9351[0], tmp_9271[1] + tmp_9351[1], tmp_9271[2] + tmp_9351[2]];
    signal tmp_9353[3] <== [11647134996302165546 * tmp_9352[0], 11647134996302165546 * tmp_9352[1], 11647134996302165546 * tmp_9352[2]];
    signal tmp_9354[3] <== [tmp_9349[0] + tmp_9353[0], tmp_9349[1] + tmp_9353[1], tmp_9349[2] + tmp_9353[2]];
    signal tmp_9355[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9356[3] <== [tmp_9355[0] * 9961063800679125687, tmp_9355[1] * 9961063800679125687, tmp_9355[2] * 9961063800679125687];
    signal tmp_9357[3] <== [tmp_9276[0] + tmp_9356[0], tmp_9276[1] + tmp_9356[1], tmp_9276[2] + tmp_9356[2]];
    signal tmp_9358[3] <== [9465069334725615251 * tmp_9357[0], 9465069334725615251 * tmp_9357[1], 9465069334725615251 * tmp_9357[2]];
    signal tmp_9359[3] <== [tmp_9354[0] + tmp_9358[0], tmp_9354[1] + tmp_9358[1], tmp_9354[2] + tmp_9358[2]];
    signal tmp_9360[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9361[3] <== [tmp_9360[0] * 6634243339771071547, tmp_9360[1] * 6634243339771071547, tmp_9360[2] * 6634243339771071547];
    signal tmp_9362[3] <== [tmp_9281[0] + tmp_9361[0], tmp_9281[1] + tmp_9361[1], tmp_9281[2] + tmp_9361[2]];
    signal tmp_9363[3] <== [7347175090247155478 * tmp_9362[0], 7347175090247155478 * tmp_9362[1], 7347175090247155478 * tmp_9362[2]];
    signal tmp_9364[3] <== [tmp_9359[0] + tmp_9363[0], tmp_9359[1] + tmp_9363[1], tmp_9359[2] + tmp_9363[2]];
    signal tmp_9365[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9366[3] <== [tmp_9365[0] * 3730820316519738416, tmp_9365[1] * 3730820316519738416, tmp_9365[2] * 3730820316519738416];
    signal tmp_9367[3] <== [tmp_9286[0] + tmp_9366[0], tmp_9286[1] + tmp_9366[1], tmp_9286[2] + tmp_9366[2]];
    signal tmp_9368[3] <== [929610534534885463 * tmp_9367[0], 929610534534885463 * tmp_9367[1], 929610534534885463 * tmp_9367[2]];
    signal tmp_9369[3] <== [tmp_9364[0] + tmp_9368[0], tmp_9364[1] + tmp_9368[1], tmp_9364[2] + tmp_9368[2]];
    signal tmp_9370[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9371[3] <== [tmp_9370[0] * 6178874009151806010, tmp_9370[1] * 6178874009151806010, tmp_9370[2] * 6178874009151806010];
    signal tmp_9372[3] <== [tmp_9291[0] + tmp_9371[0], tmp_9291[1] + tmp_9371[1], tmp_9291[2] + tmp_9371[2]];
    signal tmp_9373[3] <== [10037966693745455517 * tmp_9372[0], 10037966693745455517 * tmp_9372[1], 10037966693745455517 * tmp_9372[2]];
    signal tmp_9374[3] <== [tmp_9369[0] + tmp_9373[0], tmp_9369[1] + tmp_9373[1], tmp_9369[2] + tmp_9373[2]];
    signal tmp_9375[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9376[3] <== [tmp_9375[0] * 151426042370448923, tmp_9375[1] * 151426042370448923, tmp_9375[2] * 151426042370448923];
    signal tmp_9377[3] <== [tmp_9296[0] + tmp_9376[0], tmp_9296[1] + tmp_9376[1], tmp_9296[2] + tmp_9376[2]];
    signal tmp_9378[3] <== [9847255082563255028 * tmp_9377[0], 9847255082563255028 * tmp_9377[1], 9847255082563255028 * tmp_9377[2]];
    signal tmp_9379[3] <== [tmp_9374[0] + tmp_9378[0], tmp_9374[1] + tmp_9378[1], tmp_9374[2] + tmp_9378[2]];
    signal tmp_9380[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9381[3] <== [tmp_9380[0] * 17283463014546786540, tmp_9380[1] * 17283463014546786540, tmp_9380[2] * 17283463014546786540];
    signal tmp_9382[3] <== [tmp_9301[0] + tmp_9381[0], tmp_9301[1] + tmp_9381[1], tmp_9301[2] + tmp_9381[2]];
    signal tmp_9383[3] <== [7498640650554676844 * tmp_9382[0], 7498640650554676844 * tmp_9382[1], 7498640650554676844 * tmp_9382[2]];
    signal tmp_9384[3] <== [tmp_9379[0] + tmp_9383[0], tmp_9379[1] + tmp_9383[1], tmp_9379[2] + tmp_9383[2]];
    signal tmp_9385[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9386[3] <== [tmp_9385[0] * 2716553630958499013, tmp_9385[1] * 2716553630958499013, tmp_9385[2] * 2716553630958499013];
    signal tmp_9387[3] <== [tmp_9306[0] + tmp_9386[0], tmp_9306[1] + tmp_9386[1], tmp_9306[2] + tmp_9386[2]];
    signal tmp_9388[3] <== [9423900797682883949 * tmp_9387[0], 9423900797682883949 * tmp_9387[1], 9423900797682883949 * tmp_9387[2]];
    signal tmp_9389[3] <== [tmp_9384[0] + tmp_9388[0], tmp_9384[1] + tmp_9388[1], tmp_9384[2] + tmp_9388[2]];
    signal tmp_9390[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9391[3] <== [tmp_9390[0] * 17502570475072947141, tmp_9390[1] * 17502570475072947141, tmp_9390[2] * 17502570475072947141];
    signal tmp_9392[3] <== [tmp_9311[0] + tmp_9391[0], tmp_9311[1] + tmp_9391[1], tmp_9311[2] + tmp_9391[2]];
    signal tmp_9393[3] <== [16088378537411909982 * tmp_9392[0], 16088378537411909982 * tmp_9392[1], 16088378537411909982 * tmp_9392[2]];
    signal tmp_9394[3] <== [tmp_9389[0] + tmp_9393[0], tmp_9389[1] + tmp_9393[1], tmp_9389[2] + tmp_9393[2]];
    signal tmp_9395[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9396[3] <== [tmp_9395[0] * 7861949515711048574, tmp_9395[1] * 7861949515711048574, tmp_9395[2] * 7861949515711048574];
    signal tmp_9397[3] <== [tmp_9316[0] + tmp_9396[0], tmp_9316[1] + tmp_9396[1], tmp_9316[2] + tmp_9396[2]];
    signal tmp_9398[3] <== [8768480041621124523 * tmp_9397[0], 8768480041621124523 * tmp_9397[1], 8768480041621124523 * tmp_9397[2]];
    signal tmp_9399[3] <== [tmp_9394[0] + tmp_9398[0], tmp_9394[1] + tmp_9398[1], tmp_9394[2] + tmp_9398[2]];
    signal tmp_9400[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9401[3] <== [tmp_9400[0] * 16107154087629722508, tmp_9400[1] * 16107154087629722508, tmp_9400[2] * 16107154087629722508];
    signal tmp_9402[3] <== [tmp_9321[0] + tmp_9401[0], tmp_9321[1] + tmp_9401[1], tmp_9321[2] + tmp_9401[2]];
    signal tmp_9403[3] <== [11676865538804947598 * tmp_9402[0], 11676865538804947598 * tmp_9402[1], 11676865538804947598 * tmp_9402[2]];
    signal tmp_9404[3] <== [tmp_9399[0] + tmp_9403[0], tmp_9399[1] + tmp_9403[1], tmp_9399[2] + tmp_9403[2]];
    signal tmp_9405[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9406[3] <== [tmp_9405[0] * 16444612208872340826, tmp_9405[1] * 16444612208872340826, tmp_9405[2] * 16444612208872340826];
    signal tmp_9407[3] <== [tmp_9326[0] + tmp_9406[0], tmp_9326[1] + tmp_9406[1], tmp_9326[2] + tmp_9406[2]];
    signal tmp_9408[3] <== [3273168334829907114 * tmp_9407[0], 3273168334829907114 * tmp_9407[1], 3273168334829907114 * tmp_9407[2]];
    signal tmp_9409[3] <== [tmp_9404[0] + tmp_9408[0], tmp_9404[1] + tmp_9408[1], tmp_9404[2] + tmp_9408[2]];
    signal tmp_9410[3] <== [tmp_7433[0] + 9086063564225054097, tmp_7433[1], tmp_7433[2]];
    signal tmp_9411[3] <== [tmp_9410[0] * 16364131105478793415, tmp_9410[1] * 16364131105478793415, tmp_9410[2] * 16364131105478793415];
    signal tmp_9412[3] <== [tmp_9331[0] + tmp_9411[0], tmp_9331[1] + tmp_9411[1], tmp_9331[2] + tmp_9411[2]];
    signal tmp_9413[3] <== [15744851004062856881 * tmp_9412[0], 15744851004062856881 * tmp_9412[1], 15744851004062856881 * tmp_9412[2]];
    signal tmp_9414[3] <== [tmp_9409[0] + tmp_9413[0], tmp_9409[1] + tmp_9413[1], tmp_9409[2] + tmp_9413[2]];
    signal tmp_9415[3] <== [tmp_9414[0] * 1, tmp_9414[1] * 1, tmp_9414[2] * 1];
    signal tmp_9416[3] <== [evals[85][0] - tmp_9415[0], evals[85][1] - tmp_9415[1], evals[85][2] - tmp_9415[2]];
    signal tmp_9417[3] <== CMul()(tmp_6068, tmp_9416);
    signal tmp_9418[3] <== [tmp_9338[0] + tmp_9417[0], tmp_9338[1] + tmp_9417[1], tmp_9338[2] + tmp_9417[2]];
    signal tmp_9419[3] <== CMul()(challengeQ, tmp_9418);
    signal tmp_9420[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9421[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9422[3] <== [tmp_9421[0] * 5047845749023635250, tmp_9421[1] * 5047845749023635250, tmp_9421[2] * 5047845749023635250];
    signal tmp_9423[3] <== [tmp_9342[0] + tmp_9422[0], tmp_9342[1] + tmp_9422[1], tmp_9342[2] + tmp_9422[2]];
    signal tmp_9424[3] <== [9944788307894914133 * tmp_9423[0], 9944788307894914133 * tmp_9423[1], 9944788307894914133 * tmp_9423[2]];
    signal tmp_9425[3] <== [tmp_9420[0] + tmp_9424[0], tmp_9420[1] + tmp_9424[1], tmp_9420[2] + tmp_9424[2]];
    signal tmp_9426[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9427[3] <== [tmp_9426[0] * 8434851656909622170, tmp_9426[1] * 8434851656909622170, tmp_9426[2] * 8434851656909622170];
    signal tmp_9428[3] <== [tmp_9347[0] + tmp_9427[0], tmp_9347[1] + tmp_9427[1], tmp_9347[2] + tmp_9427[2]];
    signal tmp_9429[3] <== [10771689630961936167 * tmp_9428[0], 10771689630961936167 * tmp_9428[1], 10771689630961936167 * tmp_9428[2]];
    signal tmp_9430[3] <== [tmp_9425[0] + tmp_9429[0], tmp_9425[1] + tmp_9429[1], tmp_9425[2] + tmp_9429[2]];
    signal tmp_9431[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9432[3] <== [tmp_9431[0] * 12922301140896192660, tmp_9431[1] * 12922301140896192660, tmp_9431[2] * 12922301140896192660];
    signal tmp_9433[3] <== [tmp_9352[0] + tmp_9432[0], tmp_9352[1] + tmp_9432[1], tmp_9352[2] + tmp_9432[2]];
    signal tmp_9434[3] <== [14652833956370356968 * tmp_9433[0], 14652833956370356968 * tmp_9433[1], 14652833956370356968 * tmp_9433[2]];
    signal tmp_9435[3] <== [tmp_9430[0] + tmp_9434[0], tmp_9430[1] + tmp_9434[1], tmp_9430[2] + tmp_9434[2]];
    signal tmp_9436[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9437[3] <== [tmp_9436[0] * 14175851439986065151, tmp_9436[1] * 14175851439986065151, tmp_9436[2] * 14175851439986065151];
    signal tmp_9438[3] <== [tmp_9357[0] + tmp_9437[0], tmp_9357[1] + tmp_9437[1], tmp_9357[2] + tmp_9437[2]];
    signal tmp_9439[3] <== [2762159236071112109 * tmp_9438[0], 2762159236071112109 * tmp_9438[1], 2762159236071112109 * tmp_9438[2]];
    signal tmp_9440[3] <== [tmp_9435[0] + tmp_9439[0], tmp_9435[1] + tmp_9439[1], tmp_9435[2] + tmp_9439[2]];
    signal tmp_9441[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9442[3] <== [tmp_9441[0] * 7308936446108866250, tmp_9441[1] * 7308936446108866250, tmp_9441[2] * 7308936446108866250];
    signal tmp_9443[3] <== [tmp_9362[0] + tmp_9442[0], tmp_9362[1] + tmp_9442[1], tmp_9362[2] + tmp_9442[2]];
    signal tmp_9444[3] <== [3599708650553142297 * tmp_9443[0], 3599708650553142297 * tmp_9443[1], 3599708650553142297 * tmp_9443[2]];
    signal tmp_9445[3] <== [tmp_9440[0] + tmp_9444[0], tmp_9440[1] + tmp_9444[1], tmp_9440[2] + tmp_9444[2]];
    signal tmp_9446[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9447[3] <== [tmp_9446[0] * 1728188381301272932, tmp_9446[1] * 1728188381301272932, tmp_9446[2] * 1728188381301272932];
    signal tmp_9448[3] <== [tmp_9367[0] + tmp_9447[0], tmp_9367[1] + tmp_9447[1], tmp_9367[2] + tmp_9447[2]];
    signal tmp_9449[3] <== [1524268929892782060 * tmp_9448[0], 1524268929892782060 * tmp_9448[1], 1524268929892782060 * tmp_9448[2]];
    signal tmp_9450[3] <== [tmp_9445[0] + tmp_9449[0], tmp_9445[1] + tmp_9449[1], tmp_9445[2] + tmp_9449[2]];
    signal tmp_9451[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9452[3] <== [tmp_9451[0] * 14315181752530606135, tmp_9451[1] * 14315181752530606135, tmp_9451[2] * 14315181752530606135];
    signal tmp_9453[3] <== [tmp_9372[0] + tmp_9452[0], tmp_9372[1] + tmp_9452[1], tmp_9372[2] + tmp_9452[2]];
    signal tmp_9454[3] <== [13329681733957674057 * tmp_9453[0], 13329681733957674057 * tmp_9453[1], 13329681733957674057 * tmp_9453[2]];
    signal tmp_9455[3] <== [tmp_9450[0] + tmp_9454[0], tmp_9450[1] + tmp_9454[1], tmp_9450[2] + tmp_9454[2]];
    signal tmp_9456[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9457[3] <== [tmp_9456[0] * 6491065457156410005, tmp_9456[1] * 6491065457156410005, tmp_9456[2] * 6491065457156410005];
    signal tmp_9458[3] <== [tmp_9377[0] + tmp_9457[0], tmp_9377[1] + tmp_9457[1], tmp_9377[2] + tmp_9457[2]];
    signal tmp_9459[3] <== [5797791035413508849 * tmp_9458[0], 5797791035413508849 * tmp_9458[1], 5797791035413508849 * tmp_9458[2]];
    signal tmp_9460[3] <== [tmp_9455[0] + tmp_9459[0], tmp_9455[1] + tmp_9459[1], tmp_9455[2] + tmp_9459[2]];
    signal tmp_9461[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9462[3] <== [tmp_9461[0] * 4175603234939884368, tmp_9461[1] * 4175603234939884368, tmp_9461[2] * 4175603234939884368];
    signal tmp_9463[3] <== [tmp_9382[0] + tmp_9462[0], tmp_9382[1] + tmp_9462[1], tmp_9382[2] + tmp_9462[2]];
    signal tmp_9464[3] <== [7515887947763182304 * tmp_9463[0], 7515887947763182304 * tmp_9463[1], 7515887947763182304 * tmp_9463[2]];
    signal tmp_9465[3] <== [tmp_9460[0] + tmp_9464[0], tmp_9460[1] + tmp_9464[1], tmp_9460[2] + tmp_9464[2]];
    signal tmp_9466[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9467[3] <== [tmp_9466[0] * 1239436087648515890, tmp_9466[1] * 1239436087648515890, tmp_9466[2] * 1239436087648515890];
    signal tmp_9468[3] <== [tmp_9387[0] + tmp_9467[0], tmp_9387[1] + tmp_9467[1], tmp_9387[2] + tmp_9467[2]];
    signal tmp_9469[3] <== [16010783934711236424 * tmp_9468[0], 16010783934711236424 * tmp_9468[1], 16010783934711236424 * tmp_9468[2]];
    signal tmp_9470[3] <== [tmp_9465[0] + tmp_9469[0], tmp_9465[1] + tmp_9469[1], tmp_9465[2] + tmp_9469[2]];
    signal tmp_9471[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9472[3] <== [tmp_9471[0] * 7915700559009623006, tmp_9471[1] * 7915700559009623006, tmp_9471[2] * 7915700559009623006];
    signal tmp_9473[3] <== [tmp_9392[0] + tmp_9472[0], tmp_9392[1] + tmp_9472[1], tmp_9392[2] + tmp_9472[2]];
    signal tmp_9474[3] <== [4190904235179964140 * tmp_9473[0], 4190904235179964140 * tmp_9473[1], 4190904235179964140 * tmp_9473[2]];
    signal tmp_9475[3] <== [tmp_9470[0] + tmp_9474[0], tmp_9470[1] + tmp_9474[1], tmp_9470[2] + tmp_9474[2]];
    signal tmp_9476[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9477[3] <== [tmp_9476[0] * 14767908959930482966, tmp_9476[1] * 14767908959930482966, tmp_9476[2] * 14767908959930482966];
    signal tmp_9478[3] <== [tmp_9397[0] + tmp_9477[0], tmp_9397[1] + tmp_9477[1], tmp_9397[2] + tmp_9477[2]];
    signal tmp_9479[3] <== [14187663514905569572 * tmp_9478[0], 14187663514905569572 * tmp_9478[1], 14187663514905569572 * tmp_9478[2]];
    signal tmp_9480[3] <== [tmp_9475[0] + tmp_9479[0], tmp_9475[1] + tmp_9479[1], tmp_9475[2] + tmp_9479[2]];
    signal tmp_9481[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9482[3] <== [tmp_9481[0] * 13675360068971462516, tmp_9481[1] * 13675360068971462516, tmp_9481[2] * 13675360068971462516];
    signal tmp_9483[3] <== [tmp_9402[0] + tmp_9482[0], tmp_9402[1] + tmp_9482[1], tmp_9402[2] + tmp_9482[2]];
    signal tmp_9484[3] <== [2777661931104600049 * tmp_9483[0], 2777661931104600049 * tmp_9483[1], 2777661931104600049 * tmp_9483[2]];
    signal tmp_9485[3] <== [tmp_9480[0] + tmp_9484[0], tmp_9480[1] + tmp_9484[1], tmp_9480[2] + tmp_9484[2]];
    signal tmp_9486[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9487[3] <== [tmp_9486[0] * 15228155835675626627, tmp_9486[1] * 15228155835675626627, tmp_9486[2] * 15228155835675626627];
    signal tmp_9488[3] <== [tmp_9407[0] + tmp_9487[0], tmp_9407[1] + tmp_9487[1], tmp_9407[2] + tmp_9487[2]];
    signal tmp_9489[3] <== [12517723326974261293 * tmp_9488[0], 12517723326974261293 * tmp_9488[1], 12517723326974261293 * tmp_9488[2]];
    signal tmp_9490[3] <== [tmp_9485[0] + tmp_9489[0], tmp_9485[1] + tmp_9489[1], tmp_9485[2] + tmp_9489[2]];
    signal tmp_9491[3] <== [tmp_7444[0] + 11571460868925408324, tmp_7444[1], tmp_7444[2]];
    signal tmp_9492[3] <== [tmp_9491[0] * 4870139207699630301, tmp_9491[1] * 4870139207699630301, tmp_9491[2] * 4870139207699630301];
    signal tmp_9493[3] <== [tmp_9412[0] + tmp_9492[0], tmp_9412[1] + tmp_9492[1], tmp_9412[2] + tmp_9492[2]];
    signal tmp_9494[3] <== [4677019694129625027 * tmp_9493[0], 4677019694129625027 * tmp_9493[1], 4677019694129625027 * tmp_9493[2]];
    signal tmp_9495[3] <== [tmp_9490[0] + tmp_9494[0], tmp_9490[1] + tmp_9494[1], tmp_9490[2] + tmp_9494[2]];
    signal tmp_9496[3] <== [tmp_9495[0] * 1, tmp_9495[1] * 1, tmp_9495[2] * 1];
    signal tmp_9497[3] <== [evals[86][0] - tmp_9496[0], evals[86][1] - tmp_9496[1], evals[86][2] - tmp_9496[2]];
    signal tmp_9498[3] <== CMul()(tmp_6068, tmp_9497);
    signal tmp_9499[3] <== [tmp_9419[0] + tmp_9498[0], tmp_9419[1] + tmp_9498[1], tmp_9419[2] + tmp_9498[2]];
    signal tmp_9500[3] <== CMul()(challengeQ, tmp_9499);
    signal tmp_9501[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9502[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9503[3] <== [tmp_9502[0] * 1665021560400291078, tmp_9502[1] * 1665021560400291078, tmp_9502[2] * 1665021560400291078];
    signal tmp_9504[3] <== [tmp_9423[0] + tmp_9503[0], tmp_9423[1] + tmp_9503[1], tmp_9423[2] + tmp_9503[2]];
    signal tmp_9505[3] <== [11160115342809132588 * tmp_9504[0], 11160115342809132588 * tmp_9504[1], 11160115342809132588 * tmp_9504[2]];
    signal tmp_9506[3] <== [tmp_9501[0] + tmp_9505[0], tmp_9501[1] + tmp_9505[1], tmp_9501[2] + tmp_9505[2]];
    signal tmp_9507[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9508[3] <== [tmp_9507[0] * 6249272897953199859, tmp_9507[1] * 6249272897953199859, tmp_9507[2] * 6249272897953199859];
    signal tmp_9509[3] <== [tmp_9428[0] + tmp_9508[0], tmp_9428[1] + tmp_9508[1], tmp_9428[2] + tmp_9508[2]];
    signal tmp_9510[3] <== [15052831855997744501 * tmp_9509[0], 15052831855997744501 * tmp_9509[1], 15052831855997744501 * tmp_9509[2]];
    signal tmp_9511[3] <== [tmp_9506[0] + tmp_9510[0], tmp_9506[1] + tmp_9510[1], tmp_9506[2] + tmp_9510[2]];
    signal tmp_9512[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9513[3] <== [tmp_9512[0] * 7260683056416925359, tmp_9512[1] * 7260683056416925359, tmp_9512[2] * 7260683056416925359];
    signal tmp_9514[3] <== [tmp_9433[0] + tmp_9513[0], tmp_9433[1] + tmp_9513[1], tmp_9433[2] + tmp_9513[2]];
    signal tmp_9515[3] <== [15739451022504817202 * tmp_9514[0], 15739451022504817202 * tmp_9514[1], 15739451022504817202 * tmp_9514[2]];
    signal tmp_9516[3] <== [tmp_9511[0] + tmp_9515[0], tmp_9511[1] + tmp_9515[1], tmp_9511[2] + tmp_9515[2]];
    signal tmp_9517[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9518[3] <== [tmp_9517[0] * 11363141252265098044, tmp_9517[1] * 11363141252265098044, tmp_9517[2] * 11363141252265098044];
    signal tmp_9519[3] <== [tmp_9438[0] + tmp_9518[0], tmp_9438[1] + tmp_9518[1], tmp_9438[2] + tmp_9518[2]];
    signal tmp_9520[3] <== [9671855798646808088 * tmp_9519[0], 9671855798646808088 * tmp_9519[1], 9671855798646808088 * tmp_9519[2]];
    signal tmp_9521[3] <== [tmp_9516[0] + tmp_9520[0], tmp_9516[1] + tmp_9520[1], tmp_9516[2] + tmp_9520[2]];
    signal tmp_9522[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9523[3] <== [tmp_9522[0] * 8012231655085569456, tmp_9522[1] * 8012231655085569456, tmp_9522[2] * 8012231655085569456];
    signal tmp_9524[3] <== [tmp_9443[0] + tmp_9523[0], tmp_9443[1] + tmp_9523[1], tmp_9443[2] + tmp_9523[2]];
    signal tmp_9525[3] <== [3993080706899827970 * tmp_9524[0], 3993080706899827970 * tmp_9524[1], 3993080706899827970 * tmp_9524[2]];
    signal tmp_9526[3] <== [tmp_9521[0] + tmp_9525[0], tmp_9521[1] + tmp_9525[1], tmp_9521[2] + tmp_9525[2]];
    signal tmp_9527[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9528[3] <== [tmp_9527[0] * 13251561609088352545, tmp_9527[1] * 13251561609088352545, tmp_9527[2] * 13251561609088352545];
    signal tmp_9529[3] <== [tmp_9448[0] + tmp_9528[0], tmp_9448[1] + tmp_9528[1], tmp_9448[2] + tmp_9528[2]];
    signal tmp_9530[3] <== [17244704481465347826 * tmp_9529[0], 17244704481465347826 * tmp_9529[1], 17244704481465347826 * tmp_9529[2]];
    signal tmp_9531[3] <== [tmp_9526[0] + tmp_9530[0], tmp_9526[1] + tmp_9530[1], tmp_9526[2] + tmp_9530[2]];
    signal tmp_9532[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9533[3] <== [tmp_9532[0] * 15473520463141521750, tmp_9532[1] * 15473520463141521750, tmp_9532[2] * 15473520463141521750];
    signal tmp_9534[3] <== [tmp_9453[0] + tmp_9533[0], tmp_9453[1] + tmp_9533[1], tmp_9453[2] + tmp_9533[2]];
    signal tmp_9535[3] <== [6669884464780361495 * tmp_9534[0], 6669884464780361495 * tmp_9534[1], 6669884464780361495 * tmp_9534[2]];
    signal tmp_9536[3] <== [tmp_9531[0] + tmp_9535[0], tmp_9531[1] + tmp_9535[1], tmp_9531[2] + tmp_9535[2]];
    signal tmp_9537[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9538[3] <== [tmp_9537[0] * 11088129738238123803, tmp_9537[1] * 11088129738238123803, tmp_9537[2] * 11088129738238123803];
    signal tmp_9539[3] <== [tmp_9458[0] + tmp_9538[0], tmp_9458[1] + tmp_9538[1], tmp_9458[2] + tmp_9538[2]];
    signal tmp_9540[3] <== [14600555312004421850 * tmp_9539[0], 14600555312004421850 * tmp_9539[1], 14600555312004421850 * tmp_9539[2]];
    signal tmp_9541[3] <== [tmp_9536[0] + tmp_9540[0], tmp_9536[1] + tmp_9540[1], tmp_9536[2] + tmp_9540[2]];
    signal tmp_9542[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9543[3] <== [tmp_9542[0] * 1281025690561882185, tmp_9542[1] * 1281025690561882185, tmp_9542[2] * 1281025690561882185];
    signal tmp_9544[3] <== [tmp_9463[0] + tmp_9543[0], tmp_9463[1] + tmp_9543[1], tmp_9463[2] + tmp_9543[2]];
    signal tmp_9545[3] <== [12885797354206706070 * tmp_9544[0], 12885797354206706070 * tmp_9544[1], 12885797354206706070 * tmp_9544[2]];
    signal tmp_9546[3] <== [tmp_9541[0] + tmp_9545[0], tmp_9541[1] + tmp_9545[1], tmp_9541[2] + tmp_9545[2]];
    signal tmp_9547[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9548[3] <== [tmp_9547[0] * 8655250499426779365, tmp_9547[1] * 8655250499426779365, tmp_9547[2] * 8655250499426779365];
    signal tmp_9549[3] <== [tmp_9468[0] + tmp_9548[0], tmp_9468[1] + tmp_9548[1], tmp_9468[2] + tmp_9548[2]];
    signal tmp_9550[3] <== [4680617225974204348 * tmp_9549[0], 4680617225974204348 * tmp_9549[1], 4680617225974204348 * tmp_9549[2]];
    signal tmp_9551[3] <== [tmp_9546[0] + tmp_9550[0], tmp_9546[1] + tmp_9550[1], tmp_9546[2] + tmp_9550[2]];
    signal tmp_9552[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9553[3] <== [tmp_9552[0] * 1321126874272112769, tmp_9552[1] * 1321126874272112769, tmp_9552[2] * 1321126874272112769];
    signal tmp_9554[3] <== [tmp_9473[0] + tmp_9553[0], tmp_9473[1] + tmp_9553[1], tmp_9473[2] + tmp_9553[2]];
    signal tmp_9555[3] <== [6736135874728218206 * tmp_9554[0], 6736135874728218206 * tmp_9554[1], 6736135874728218206 * tmp_9554[2]];
    signal tmp_9556[3] <== [tmp_9551[0] + tmp_9555[0], tmp_9551[1] + tmp_9555[1], tmp_9551[2] + tmp_9555[2]];
    signal tmp_9557[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9558[3] <== [tmp_9557[0] * 11480921900711623217, tmp_9557[1] * 11480921900711623217, tmp_9557[2] * 11480921900711623217];
    signal tmp_9559[3] <== [tmp_9478[0] + tmp_9558[0], tmp_9478[1] + tmp_9558[1], tmp_9478[2] + tmp_9558[2]];
    signal tmp_9560[3] <== [11543984753251934208 * tmp_9559[0], 11543984753251934208 * tmp_9559[1], 11543984753251934208 * tmp_9559[2]];
    signal tmp_9561[3] <== [tmp_9556[0] + tmp_9560[0], tmp_9556[1] + tmp_9560[1], tmp_9556[2] + tmp_9560[2]];
    signal tmp_9562[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9563[3] <== [tmp_9562[0] * 8534648876771497041, tmp_9562[1] * 8534648876771497041, tmp_9562[2] * 8534648876771497041];
    signal tmp_9564[3] <== [tmp_9483[0] + tmp_9563[0], tmp_9483[1] + tmp_9563[1], tmp_9483[2] + tmp_9563[2]];
    signal tmp_9565[3] <== [7462450122751074327 * tmp_9564[0], 7462450122751074327 * tmp_9564[1], 7462450122751074327 * tmp_9564[2]];
    signal tmp_9566[3] <== [tmp_9561[0] + tmp_9565[0], tmp_9561[1] + tmp_9565[1], tmp_9561[2] + tmp_9565[2]];
    signal tmp_9567[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9568[3] <== [tmp_9567[0] * 2345799062308732216, tmp_9567[1] * 2345799062308732216, tmp_9567[2] * 2345799062308732216];
    signal tmp_9569[3] <== [tmp_9488[0] + tmp_9568[0], tmp_9488[1] + tmp_9568[1], tmp_9488[2] + tmp_9568[2]];
    signal tmp_9570[3] <== [5997876138824023765 * tmp_9569[0], 5997876138824023765 * tmp_9569[1], 5997876138824023765 * tmp_9569[2]];
    signal tmp_9571[3] <== [tmp_9566[0] + tmp_9570[0], tmp_9566[1] + tmp_9570[1], tmp_9566[2] + tmp_9570[2]];
    signal tmp_9572[3] <== [tmp_7456[0] + 2855538343746738937, tmp_7456[1], tmp_7456[2]];
    signal tmp_9573[3] <== [tmp_9572[0] * 8793821150452954879, tmp_9572[1] * 8793821150452954879, tmp_9572[2] * 8793821150452954879];
    signal tmp_9574[3] <== [tmp_9493[0] + tmp_9573[0], tmp_9493[1] + tmp_9573[1], tmp_9493[2] + tmp_9573[2]];
    signal tmp_9575[3] <== [4694625700920314643 * tmp_9574[0], 4694625700920314643 * tmp_9574[1], 4694625700920314643 * tmp_9574[2]];
    signal tmp_9576[3] <== [tmp_9571[0] + tmp_9575[0], tmp_9571[1] + tmp_9575[1], tmp_9571[2] + tmp_9575[2]];
    signal tmp_9577[3] <== [tmp_9576[0] * 1, tmp_9576[1] * 1, tmp_9576[2] * 1];
    signal tmp_9578[3] <== [evals[87][0] - tmp_9577[0], evals[87][1] - tmp_9577[1], evals[87][2] - tmp_9577[2]];
    signal tmp_9579[3] <== CMul()(tmp_6068, tmp_9578);
    signal tmp_9580[3] <== [tmp_9500[0] + tmp_9579[0], tmp_9500[1] + tmp_9579[1], tmp_9500[2] + tmp_9579[2]];
    signal tmp_9581[3] <== CMul()(challengeQ, tmp_9580);
    signal tmp_9582[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9583[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9584[3] <== [tmp_9583[0] * 13330801039170540745, tmp_9583[1] * 13330801039170540745, tmp_9583[2] * 13330801039170540745];
    signal tmp_9585[3] <== [tmp_9504[0] + tmp_9584[0], tmp_9504[1] + tmp_9584[1], tmp_9504[2] + tmp_9584[2]];
    signal tmp_9586[3] <== [4989688843737548554 * tmp_9585[0], 4989688843737548554 * tmp_9585[1], 4989688843737548554 * tmp_9585[2]];
    signal tmp_9587[3] <== [tmp_9582[0] + tmp_9586[0], tmp_9582[1] + tmp_9586[1], tmp_9582[2] + tmp_9586[2]];
    signal tmp_9588[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9589[3] <== [tmp_9588[0] * 9691413076657869892, tmp_9588[1] * 9691413076657869892, tmp_9588[2] * 9691413076657869892];
    signal tmp_9590[3] <== [tmp_9509[0] + tmp_9589[0], tmp_9509[1] + tmp_9589[1], tmp_9509[2] + tmp_9589[2]];
    signal tmp_9591[3] <== [13821113963424948846 * tmp_9590[0], 13821113963424948846 * tmp_9590[1], 13821113963424948846 * tmp_9590[2]];
    signal tmp_9592[3] <== [tmp_9587[0] + tmp_9591[0], tmp_9587[1] + tmp_9591[1], tmp_9587[2] + tmp_9591[2]];
    signal tmp_9593[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9594[3] <== [tmp_9593[0] * 18097630306776785732, tmp_9593[1] * 18097630306776785732, tmp_9593[2] * 18097630306776785732];
    signal tmp_9595[3] <== [tmp_9514[0] + tmp_9594[0], tmp_9514[1] + tmp_9594[1], tmp_9514[2] + tmp_9594[2]];
    signal tmp_9596[3] <== [7247806441173525246 * tmp_9595[0], 7247806441173525246 * tmp_9595[1], 7247806441173525246 * tmp_9595[2]];
    signal tmp_9597[3] <== [tmp_9592[0] + tmp_9596[0], tmp_9592[1] + tmp_9596[1], tmp_9592[2] + tmp_9596[2]];
    signal tmp_9598[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9599[3] <== [tmp_9598[0] * 5719637133317020548, tmp_9598[1] * 5719637133317020548, tmp_9598[2] * 5719637133317020548];
    signal tmp_9600[3] <== [tmp_9519[0] + tmp_9599[0], tmp_9519[1] + tmp_9599[1], tmp_9519[2] + tmp_9599[2]];
    signal tmp_9601[3] <== [3364393074731955357 * tmp_9600[0], 3364393074731955357 * tmp_9600[1], 3364393074731955357 * tmp_9600[2]];
    signal tmp_9602[3] <== [tmp_9597[0] + tmp_9601[0], tmp_9597[1] + tmp_9601[1], tmp_9597[2] + tmp_9601[2]];
    signal tmp_9603[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9604[3] <== [tmp_9603[0] * 4029888846798459355, tmp_9603[1] * 4029888846798459355, tmp_9603[2] * 4029888846798459355];
    signal tmp_9605[3] <== [tmp_9524[0] + tmp_9604[0], tmp_9524[1] + tmp_9604[1], tmp_9524[2] + tmp_9604[2]];
    signal tmp_9606[3] <== [10146937751881836190 * tmp_9605[0], 10146937751881836190 * tmp_9605[1], 10146937751881836190 * tmp_9605[2]];
    signal tmp_9607[3] <== [tmp_9602[0] + tmp_9606[0], tmp_9602[1] + tmp_9606[1], tmp_9602[2] + tmp_9606[2]];
    signal tmp_9608[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9609[3] <== [tmp_9608[0] * 18356621544169502341, tmp_9608[1] * 18356621544169502341, tmp_9608[2] * 18356621544169502341];
    signal tmp_9610[3] <== [tmp_9529[0] + tmp_9609[0], tmp_9529[1] + tmp_9609[1], tmp_9529[2] + tmp_9609[2]];
    signal tmp_9611[3] <== [8218226966171798065 * tmp_9610[0], 8218226966171798065 * tmp_9610[1], 8218226966171798065 * tmp_9610[2]];
    signal tmp_9612[3] <== [tmp_9607[0] + tmp_9611[0], tmp_9607[1] + tmp_9611[1], tmp_9607[2] + tmp_9611[2]];
    signal tmp_9613[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9614[3] <== [tmp_9613[0] * 3193305903015148388, tmp_9613[1] * 3193305903015148388, tmp_9613[2] * 3193305903015148388];
    signal tmp_9615[3] <== [tmp_9534[0] + tmp_9614[0], tmp_9534[1] + tmp_9614[1], tmp_9534[2] + tmp_9614[2]];
    signal tmp_9616[3] <== [6275505124750343799 * tmp_9615[0], 6275505124750343799 * tmp_9615[1], 6275505124750343799 * tmp_9615[2]];
    signal tmp_9617[3] <== [tmp_9612[0] + tmp_9616[0], tmp_9612[1] + tmp_9616[1], tmp_9612[2] + tmp_9616[2]];
    signal tmp_9618[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9619[3] <== [tmp_9618[0] * 13178443391748614197, tmp_9618[1] * 13178443391748614197, tmp_9618[2] * 13178443391748614197];
    signal tmp_9620[3] <== [tmp_9539[0] + tmp_9619[0], tmp_9539[1] + tmp_9619[1], tmp_9539[2] + tmp_9619[2]];
    signal tmp_9621[3] <== [11654032185587768765 * tmp_9620[0], 11654032185587768765 * tmp_9620[1], 11654032185587768765 * tmp_9620[2]];
    signal tmp_9622[3] <== [tmp_9617[0] + tmp_9621[0], tmp_9617[1] + tmp_9621[1], tmp_9617[2] + tmp_9621[2]];
    signal tmp_9623[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9624[3] <== [tmp_9623[0] * 14481071249376583383, tmp_9623[1] * 14481071249376583383, tmp_9623[2] * 14481071249376583383];
    signal tmp_9625[3] <== [tmp_9544[0] + tmp_9624[0], tmp_9544[1] + tmp_9624[1], tmp_9544[2] + tmp_9624[2]];
    signal tmp_9626[3] <== [11697929231275965966 * tmp_9625[0], 11697929231275965966 * tmp_9625[1], 11697929231275965966 * tmp_9625[2]];
    signal tmp_9627[3] <== [tmp_9622[0] + tmp_9626[0], tmp_9622[1] + tmp_9626[1], tmp_9622[2] + tmp_9626[2]];
    signal tmp_9628[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9629[3] <== [tmp_9628[0] * 13298473208691573067, tmp_9628[1] * 13298473208691573067, tmp_9628[2] * 13298473208691573067];
    signal tmp_9630[3] <== [tmp_9549[0] + tmp_9629[0], tmp_9549[1] + tmp_9629[1], tmp_9549[2] + tmp_9629[2]];
    signal tmp_9631[3] <== [9256045630802942941 * tmp_9630[0], 9256045630802942941 * tmp_9630[1], 9256045630802942941 * tmp_9630[2]];
    signal tmp_9632[3] <== [tmp_9627[0] + tmp_9631[0], tmp_9627[1] + tmp_9631[1], tmp_9627[2] + tmp_9631[2]];
    signal tmp_9633[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9634[3] <== [tmp_9633[0] * 5559358937569616668, tmp_9633[1] * 5559358937569616668, tmp_9633[2] * 5559358937569616668];
    signal tmp_9635[3] <== [tmp_9554[0] + tmp_9634[0], tmp_9554[1] + tmp_9634[1], tmp_9554[2] + tmp_9634[2]];
    signal tmp_9636[3] <== [15946532889053694625 * tmp_9635[0], 15946532889053694625 * tmp_9635[1], 15946532889053694625 * tmp_9635[2]];
    signal tmp_9637[3] <== [tmp_9632[0] + tmp_9636[0], tmp_9632[1] + tmp_9636[1], tmp_9632[2] + tmp_9636[2]];
    signal tmp_9638[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9639[3] <== [tmp_9638[0] * 5477091296498352424, tmp_9638[1] * 5477091296498352424, tmp_9638[2] * 5477091296498352424];
    signal tmp_9640[3] <== [tmp_9559[0] + tmp_9639[0], tmp_9559[1] + tmp_9639[1], tmp_9559[2] + tmp_9639[2]];
    signal tmp_9641[3] <== [14178555464781916994 * tmp_9640[0], 14178555464781916994 * tmp_9640[1], 14178555464781916994 * tmp_9640[2]];
    signal tmp_9642[3] <== [tmp_9637[0] + tmp_9641[0], tmp_9637[1] + tmp_9641[1], tmp_9637[2] + tmp_9641[2]];
    signal tmp_9643[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9644[3] <== [tmp_9643[0] * 4033222126236138968, tmp_9643[1] * 4033222126236138968, tmp_9643[2] * 4033222126236138968];
    signal tmp_9645[3] <== [tmp_9564[0] + tmp_9644[0], tmp_9564[1] + tmp_9644[1], tmp_9564[2] + tmp_9644[2]];
    signal tmp_9646[3] <== [15652577753663522945 * tmp_9645[0], 15652577753663522945 * tmp_9645[1], 15652577753663522945 * tmp_9645[2]];
    signal tmp_9647[3] <== [tmp_9642[0] + tmp_9646[0], tmp_9642[1] + tmp_9646[1], tmp_9642[2] + tmp_9646[2]];
    signal tmp_9648[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9649[3] <== [tmp_9648[0] * 12567047098625210052, tmp_9648[1] * 12567047098625210052, tmp_9648[2] * 12567047098625210052];
    signal tmp_9650[3] <== [tmp_9569[0] + tmp_9649[0], tmp_9569[1] + tmp_9649[1], tmp_9569[2] + tmp_9649[2]];
    signal tmp_9651[3] <== [18353293019555892376 * tmp_9650[0], 18353293019555892376 * tmp_9650[1], 18353293019555892376 * tmp_9650[2]];
    signal tmp_9652[3] <== [tmp_9647[0] + tmp_9651[0], tmp_9647[1] + tmp_9651[1], tmp_9647[2] + tmp_9651[2]];
    signal tmp_9653[3] <== [tmp_7467[0] + 14270783606042555221, tmp_7467[1], tmp_7467[2]];
    signal tmp_9654[3] <== [tmp_9653[0] * 5194999122761668273, tmp_9653[1] * 5194999122761668273, tmp_9653[2] * 5194999122761668273];
    signal tmp_9655[3] <== [tmp_9574[0] + tmp_9654[0], tmp_9574[1] + tmp_9654[1], tmp_9574[2] + tmp_9654[2]];
    signal tmp_9656[3] <== [913259685371457409 * tmp_9655[0], 913259685371457409 * tmp_9655[1], 913259685371457409 * tmp_9655[2]];
    signal tmp_9657[3] <== [tmp_9652[0] + tmp_9656[0], tmp_9652[1] + tmp_9656[1], tmp_9652[2] + tmp_9656[2]];
    signal tmp_9658[3] <== [tmp_9657[0] * 1, tmp_9657[1] * 1, tmp_9657[2] * 1];
    signal tmp_9659[3] <== [evals[88][0] - tmp_9658[0], evals[88][1] - tmp_9658[1], evals[88][2] - tmp_9658[2]];
    signal tmp_9660[3] <== CMul()(tmp_6068, tmp_9659);
    signal tmp_9661[3] <== [tmp_9581[0] + tmp_9660[0], tmp_9581[1] + tmp_9660[1], tmp_9581[2] + tmp_9660[2]];
    signal tmp_9662[3] <== CMul()(challengeQ, tmp_9661);
    signal tmp_9663[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9664[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9665[3] <== [tmp_9664[0] * 2839286413723076283, tmp_9664[1] * 2839286413723076283, tmp_9664[2] * 2839286413723076283];
    signal tmp_9666[3] <== [tmp_9585[0] + tmp_9665[0], tmp_9585[1] + tmp_9665[1], tmp_9585[2] + tmp_9665[2]];
    signal tmp_9667[3] <== [8997952006844223763 * tmp_9666[0], 8997952006844223763 * tmp_9666[1], 8997952006844223763 * tmp_9666[2]];
    signal tmp_9668[3] <== [tmp_9663[0] + tmp_9667[0], tmp_9663[1] + tmp_9667[1], tmp_9663[2] + tmp_9667[2]];
    signal tmp_9669[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9670[3] <== [tmp_9669[0] * 8227928784918533600, tmp_9669[1] * 8227928784918533600, tmp_9669[2] * 8227928784918533600];
    signal tmp_9671[3] <== [tmp_9590[0] + tmp_9670[0], tmp_9590[1] + tmp_9670[1], tmp_9590[2] + tmp_9670[2]];
    signal tmp_9672[3] <== [2353549075741107334 * tmp_9671[0], 2353549075741107334 * tmp_9671[1], 2353549075741107334 * tmp_9671[2]];
    signal tmp_9673[3] <== [tmp_9668[0] + tmp_9672[0], tmp_9668[1] + tmp_9672[1], tmp_9668[2] + tmp_9672[2]];
    signal tmp_9674[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9675[3] <== [tmp_9674[0] * 6659444090177547341, tmp_9674[1] * 6659444090177547341, tmp_9674[2] * 6659444090177547341];
    signal tmp_9676[3] <== [tmp_9595[0] + tmp_9675[0], tmp_9595[1] + tmp_9675[1], tmp_9595[2] + tmp_9675[2]];
    signal tmp_9677[3] <== [15713716106319331569 * tmp_9676[0], 15713716106319331569 * tmp_9676[1], 15713716106319331569 * tmp_9676[2]];
    signal tmp_9678[3] <== [tmp_9673[0] + tmp_9677[0], tmp_9673[1] + tmp_9677[1], tmp_9673[2] + tmp_9677[2]];
    signal tmp_9679[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9680[3] <== [tmp_9679[0] * 7600262705963786111, tmp_9679[1] * 7600262705963786111, tmp_9679[2] * 7600262705963786111];
    signal tmp_9681[3] <== [tmp_9600[0] + tmp_9680[0], tmp_9600[1] + tmp_9680[1], tmp_9600[2] + tmp_9680[2]];
    signal tmp_9682[3] <== [16205952151624674977 * tmp_9681[0], 16205952151624674977 * tmp_9681[1], 16205952151624674977 * tmp_9681[2]];
    signal tmp_9683[3] <== [tmp_9678[0] + tmp_9682[0], tmp_9678[1] + tmp_9682[1], tmp_9678[2] + tmp_9682[2]];
    signal tmp_9684[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9685[3] <== [tmp_9684[0] * 14410065364859218948, tmp_9684[1] * 14410065364859218948, tmp_9684[2] * 14410065364859218948];
    signal tmp_9686[3] <== [tmp_9605[0] + tmp_9685[0], tmp_9605[1] + tmp_9685[1], tmp_9605[2] + tmp_9685[2]];
    signal tmp_9687[3] <== [15307752746251443250 * tmp_9686[0], 15307752746251443250 * tmp_9686[1], 15307752746251443250 * tmp_9686[2]];
    signal tmp_9688[3] <== [tmp_9683[0] + tmp_9687[0], tmp_9683[1] + tmp_9687[1], tmp_9683[2] + tmp_9687[2]];
    signal tmp_9689[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9690[3] <== [tmp_9689[0] * 5216672470358585092, tmp_9689[1] * 5216672470358585092, tmp_9689[2] * 5216672470358585092];
    signal tmp_9691[3] <== [tmp_9610[0] + tmp_9690[0], tmp_9610[1] + tmp_9690[1], tmp_9610[2] + tmp_9690[2]];
    signal tmp_9692[3] <== [10826281796282448866 * tmp_9691[0], 10826281796282448866 * tmp_9691[1], 10826281796282448866 * tmp_9691[2]];
    signal tmp_9693[3] <== [tmp_9688[0] + tmp_9692[0], tmp_9688[1] + tmp_9692[1], tmp_9688[2] + tmp_9692[2]];
    signal tmp_9694[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9695[3] <== [tmp_9694[0] * 10300497666947344511, tmp_9694[1] * 10300497666947344511, tmp_9694[2] * 10300497666947344511];
    signal tmp_9696[3] <== [tmp_9615[0] + tmp_9695[0], tmp_9615[1] + tmp_9695[1], tmp_9615[2] + tmp_9695[2]];
    signal tmp_9697[3] <== [12268463758085038250 * tmp_9696[0], 12268463758085038250 * tmp_9696[1], 12268463758085038250 * tmp_9696[2]];
    signal tmp_9698[3] <== [tmp_9693[0] + tmp_9697[0], tmp_9693[1] + tmp_9697[1], tmp_9693[2] + tmp_9697[2]];
    signal tmp_9699[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9700[3] <== [tmp_9699[0] * 10505442326626626850, tmp_9699[1] * 10505442326626626850, tmp_9699[2] * 10505442326626626850];
    signal tmp_9701[3] <== [tmp_9620[0] + tmp_9700[0], tmp_9620[1] + tmp_9700[1], tmp_9620[2] + tmp_9700[2]];
    signal tmp_9702[3] <== [6991270770333944507 * tmp_9701[0], 6991270770333944507 * tmp_9701[1], 6991270770333944507 * tmp_9701[2]];
    signal tmp_9703[3] <== [tmp_9698[0] + tmp_9702[0], tmp_9698[1] + tmp_9702[1], tmp_9698[2] + tmp_9702[2]];
    signal tmp_9704[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9705[3] <== [tmp_9704[0] * 2741298027419160918, tmp_9704[1] * 2741298027419160918, tmp_9704[2] * 2741298027419160918];
    signal tmp_9706[3] <== [tmp_9625[0] + tmp_9705[0], tmp_9625[1] + tmp_9705[1], tmp_9625[2] + tmp_9705[2]];
    signal tmp_9707[3] <== [5612088705630314148 * tmp_9706[0], 5612088705630314148 * tmp_9706[1], 5612088705630314148 * tmp_9706[2]];
    signal tmp_9708[3] <== [tmp_9703[0] + tmp_9707[0], tmp_9703[1] + tmp_9707[1], tmp_9703[2] + tmp_9707[2]];
    signal tmp_9709[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9710[3] <== [tmp_9709[0] * 15044675986841164042, tmp_9709[1] * 15044675986841164042, tmp_9709[2] * 15044675986841164042];
    signal tmp_9711[3] <== [tmp_9630[0] + tmp_9710[0], tmp_9630[1] + tmp_9710[1], tmp_9630[2] + tmp_9710[2]];
    signal tmp_9712[3] <== [7985909294534842852 * tmp_9711[0], 7985909294534842852 * tmp_9711[1], 7985909294534842852 * tmp_9711[2]];
    signal tmp_9713[3] <== [tmp_9708[0] + tmp_9712[0], tmp_9708[1] + tmp_9712[1], tmp_9708[2] + tmp_9712[2]];
    signal tmp_9714[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9715[3] <== [tmp_9714[0] * 16581973004291491442, tmp_9714[1] * 16581973004291491442, tmp_9714[2] * 16581973004291491442];
    signal tmp_9716[3] <== [tmp_9635[0] + tmp_9715[0], tmp_9635[1] + tmp_9715[1], tmp_9635[2] + tmp_9715[2]];
    signal tmp_9717[3] <== [7806714111671690128 * tmp_9716[0], 7806714111671690128 * tmp_9716[1], 7806714111671690128 * tmp_9716[2]];
    signal tmp_9718[3] <== [tmp_9713[0] + tmp_9717[0], tmp_9713[1] + tmp_9717[1], tmp_9713[2] + tmp_9717[2]];
    signal tmp_9719[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9720[3] <== [tmp_9719[0] * 18135546514968986064, tmp_9719[1] * 18135546514968986064, tmp_9719[2] * 18135546514968986064];
    signal tmp_9721[3] <== [tmp_9640[0] + tmp_9720[0], tmp_9640[1] + tmp_9720[1], tmp_9640[2] + tmp_9720[2]];
    signal tmp_9722[3] <== [10214446832758516695 * tmp_9721[0], 10214446832758516695 * tmp_9721[1], 10214446832758516695 * tmp_9721[2]];
    signal tmp_9723[3] <== [tmp_9718[0] + tmp_9722[0], tmp_9718[1] + tmp_9722[1], tmp_9718[2] + tmp_9722[2]];
    signal tmp_9724[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9725[3] <== [tmp_9724[0] * 15991008099224797738, tmp_9724[1] * 15991008099224797738, tmp_9724[2] * 15991008099224797738];
    signal tmp_9726[3] <== [tmp_9645[0] + tmp_9725[0], tmp_9645[1] + tmp_9725[1], tmp_9645[2] + tmp_9725[2]];
    signal tmp_9727[3] <== [4397775621482708994 * tmp_9726[0], 4397775621482708994 * tmp_9726[1], 4397775621482708994 * tmp_9726[2]];
    signal tmp_9728[3] <== [tmp_9723[0] + tmp_9727[0], tmp_9723[1] + tmp_9727[1], tmp_9723[2] + tmp_9727[2]];
    signal tmp_9729[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9730[3] <== [tmp_9729[0] * 3751874127382110996, tmp_9729[1] * 3751874127382110996, tmp_9729[2] * 3751874127382110996];
    signal tmp_9731[3] <== [tmp_9650[0] + tmp_9730[0], tmp_9650[1] + tmp_9730[1], tmp_9650[2] + tmp_9730[2]];
    signal tmp_9732[3] <== [5208611042143950342 * tmp_9731[0], 5208611042143950342 * tmp_9731[1], 5208611042143950342 * tmp_9731[2]];
    signal tmp_9733[3] <== [tmp_9728[0] + tmp_9732[0], tmp_9728[1] + tmp_9732[1], tmp_9728[2] + tmp_9732[2]];
    signal tmp_9734[3] <== [tmp_7479[0] + 18107705583323614825, tmp_7479[1], tmp_7479[2]];
    signal tmp_9735[3] <== [tmp_9734[0] * 5416412077983775726, tmp_9734[1] * 5416412077983775726, tmp_9734[2] * 5416412077983775726];
    signal tmp_9736[3] <== [tmp_9655[0] + tmp_9735[0], tmp_9655[1] + tmp_9735[1], tmp_9655[2] + tmp_9735[2]];
    signal tmp_9737[3] <== [7274343743337614775 * tmp_9736[0], 7274343743337614775 * tmp_9736[1], 7274343743337614775 * tmp_9736[2]];
    signal tmp_9738[3] <== [tmp_9733[0] + tmp_9737[0], tmp_9733[1] + tmp_9737[1], tmp_9733[2] + tmp_9737[2]];
    signal tmp_9739[3] <== [tmp_9738[0] * 1, tmp_9738[1] * 1, tmp_9738[2] * 1];
    signal tmp_9740[3] <== [evals[89][0] - tmp_9739[0], evals[89][1] - tmp_9739[1], evals[89][2] - tmp_9739[2]];
    signal tmp_9741[3] <== CMul()(tmp_6068, tmp_9740);
    signal tmp_9742[3] <== [tmp_9662[0] + tmp_9741[0], tmp_9662[1] + tmp_9741[1], tmp_9662[2] + tmp_9741[2]];
    signal tmp_9743[3] <== CMul()(challengeQ, tmp_9742);
    signal tmp_9744[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9745[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9746[3] <== [tmp_9745[0] * 4076524286822267596, tmp_9745[1] * 4076524286822267596, tmp_9745[2] * 4076524286822267596];
    signal tmp_9747[3] <== [tmp_9666[0] + tmp_9746[0], tmp_9666[1] + tmp_9746[1], tmp_9666[2] + tmp_9746[2]];
    signal tmp_9748[3] <== [6164038266681533795 * tmp_9747[0], 6164038266681533795 * tmp_9747[1], 6164038266681533795 * tmp_9747[2]];
    signal tmp_9749[3] <== [tmp_9744[0] + tmp_9748[0], tmp_9744[1] + tmp_9748[1], tmp_9744[2] + tmp_9748[2]];
    signal tmp_9750[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9751[3] <== [tmp_9750[0] * 8800108983290110025, tmp_9750[1] * 8800108983290110025, tmp_9750[2] * 8800108983290110025];
    signal tmp_9752[3] <== [tmp_9671[0] + tmp_9751[0], tmp_9671[1] + tmp_9751[1], tmp_9671[2] + tmp_9751[2]];
    signal tmp_9753[3] <== [2954388482741701990 * tmp_9752[0], 2954388482741701990 * tmp_9752[1], 2954388482741701990 * tmp_9752[2]];
    signal tmp_9754[3] <== [tmp_9749[0] + tmp_9753[0], tmp_9749[1] + tmp_9753[1], tmp_9749[2] + tmp_9753[2]];
    signal tmp_9755[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9756[3] <== [tmp_9755[0] * 291536037642081575, tmp_9755[1] * 291536037642081575, tmp_9755[2] * 291536037642081575];
    signal tmp_9757[3] <== [tmp_9676[0] + tmp_9756[0], tmp_9676[1] + tmp_9756[1], tmp_9676[2] + tmp_9756[2]];
    signal tmp_9758[3] <== [4767760292258986237 * tmp_9757[0], 4767760292258986237 * tmp_9757[1], 4767760292258986237 * tmp_9757[2]];
    signal tmp_9759[3] <== [tmp_9754[0] + tmp_9758[0], tmp_9754[1] + tmp_9758[1], tmp_9754[2] + tmp_9758[2]];
    signal tmp_9760[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9761[3] <== [tmp_9760[0] * 5510510448504677345, tmp_9760[1] * 5510510448504677345, tmp_9760[2] * 5510510448504677345];
    signal tmp_9762[3] <== [tmp_9681[0] + tmp_9761[0], tmp_9681[1] + tmp_9761[1], tmp_9681[2] + tmp_9761[2]];
    signal tmp_9763[3] <== [1216591495300834932 * tmp_9762[0], 1216591495300834932 * tmp_9762[1], 1216591495300834932 * tmp_9762[2]];
    signal tmp_9764[3] <== [tmp_9759[0] + tmp_9763[0], tmp_9759[1] + tmp_9763[1], tmp_9759[2] + tmp_9763[2]];
    signal tmp_9765[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9766[3] <== [tmp_9765[0] * 16087089554932036377, tmp_9765[1] * 16087089554932036377, tmp_9765[2] * 16087089554932036377];
    signal tmp_9767[3] <== [tmp_9686[0] + tmp_9766[0], tmp_9686[1] + tmp_9766[1], tmp_9686[2] + tmp_9766[2]];
    signal tmp_9768[3] <== [407520068134907189 * tmp_9767[0], 407520068134907189 * tmp_9767[1], 407520068134907189 * tmp_9767[2]];
    signal tmp_9769[3] <== [tmp_9764[0] + tmp_9768[0], tmp_9764[1] + tmp_9768[1], tmp_9764[2] + tmp_9768[2]];
    signal tmp_9770[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9771[3] <== [tmp_9770[0] * 9490752218128621864, tmp_9770[1] * 9490752218128621864, tmp_9770[2] * 9490752218128621864];
    signal tmp_9772[3] <== [tmp_9691[0] + tmp_9771[0], tmp_9691[1] + tmp_9771[1], tmp_9691[2] + tmp_9771[2]];
    signal tmp_9773[3] <== [8748275194504987666 * tmp_9772[0], 8748275194504987666 * tmp_9772[1], 8748275194504987666 * tmp_9772[2]];
    signal tmp_9774[3] <== [tmp_9769[0] + tmp_9773[0], tmp_9769[1] + tmp_9773[1], tmp_9769[2] + tmp_9773[2]];
    signal tmp_9775[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9776[3] <== [tmp_9775[0] * 17141708827443995602, tmp_9775[1] * 17141708827443995602, tmp_9775[2] * 17141708827443995602];
    signal tmp_9777[3] <== [tmp_9696[0] + tmp_9776[0], tmp_9696[1] + tmp_9776[1], tmp_9696[2] + tmp_9776[2]];
    signal tmp_9778[3] <== [4398221156344358190 * tmp_9777[0], 4398221156344358190 * tmp_9777[1], 4398221156344358190 * tmp_9777[2]];
    signal tmp_9779[3] <== [tmp_9774[0] + tmp_9778[0], tmp_9774[1] + tmp_9778[1], tmp_9774[2] + tmp_9778[2]];
    signal tmp_9780[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9781[3] <== [tmp_9780[0] * 363174800131535307, tmp_9780[1] * 363174800131535307, tmp_9780[2] * 363174800131535307];
    signal tmp_9782[3] <== [tmp_9701[0] + tmp_9781[0], tmp_9701[1] + tmp_9781[1], tmp_9701[2] + tmp_9781[2]];
    signal tmp_9783[3] <== [3691238772910702249 * tmp_9782[0], 3691238772910702249 * tmp_9782[1], 3691238772910702249 * tmp_9782[2]];
    signal tmp_9784[3] <== [tmp_9779[0] + tmp_9783[0], tmp_9779[1] + tmp_9783[1], tmp_9779[2] + tmp_9783[2]];
    signal tmp_9785[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9786[3] <== [tmp_9785[0] * 9926144393408541385, tmp_9785[1] * 9926144393408541385, tmp_9785[2] * 9926144393408541385];
    signal tmp_9787[3] <== [tmp_9706[0] + tmp_9786[0], tmp_9706[1] + tmp_9786[1], tmp_9706[2] + tmp_9786[2]];
    signal tmp_9788[3] <== [14469725748730639111 * tmp_9787[0], 14469725748730639111 * tmp_9787[1], 14469725748730639111 * tmp_9787[2]];
    signal tmp_9789[3] <== [tmp_9784[0] + tmp_9788[0], tmp_9784[1] + tmp_9788[1], tmp_9784[2] + tmp_9788[2]];
    signal tmp_9790[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9791[3] <== [tmp_9790[0] * 6235228401204354743, tmp_9790[1] * 6235228401204354743, tmp_9790[2] * 6235228401204354743];
    signal tmp_9792[3] <== [tmp_9711[0] + tmp_9791[0], tmp_9711[1] + tmp_9791[1], tmp_9711[2] + tmp_9791[2]];
    signal tmp_9793[3] <== [17322574795334626440 * tmp_9792[0], 17322574795334626440 * tmp_9792[1], 17322574795334626440 * tmp_9792[2]];
    signal tmp_9794[3] <== [tmp_9789[0] + tmp_9793[0], tmp_9789[1] + tmp_9793[1], tmp_9789[2] + tmp_9793[2]];
    signal tmp_9795[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9796[3] <== [tmp_9795[0] * 8054369299943103922, tmp_9795[1] * 8054369299943103922, tmp_9795[2] * 8054369299943103922];
    signal tmp_9797[3] <== [tmp_9716[0] + tmp_9796[0], tmp_9716[1] + tmp_9796[1], tmp_9716[2] + tmp_9796[2]];
    signal tmp_9798[3] <== [6754651456942327221 * tmp_9797[0], 6754651456942327221 * tmp_9797[1], 6754651456942327221 * tmp_9797[2]];
    signal tmp_9799[3] <== [tmp_9794[0] + tmp_9798[0], tmp_9794[1] + tmp_9798[1], tmp_9794[2] + tmp_9798[2]];
    signal tmp_9800[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9801[3] <== [tmp_9800[0] * 1181948739023406256, tmp_9800[1] * 1181948739023406256, tmp_9800[2] * 1181948739023406256];
    signal tmp_9802[3] <== [tmp_9721[0] + tmp_9801[0], tmp_9721[1] + tmp_9801[1], tmp_9721[2] + tmp_9801[2]];
    signal tmp_9803[3] <== [3445987485532732891 * tmp_9802[0], 3445987485532732891 * tmp_9802[1], 3445987485532732891 * tmp_9802[2]];
    signal tmp_9804[3] <== [tmp_9799[0] + tmp_9803[0], tmp_9799[1] + tmp_9803[1], tmp_9799[2] + tmp_9803[2]];
    signal tmp_9805[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9806[3] <== [tmp_9805[0] * 5256255887623871511, tmp_9805[1] * 5256255887623871511, tmp_9805[2] * 5256255887623871511];
    signal tmp_9807[3] <== [tmp_9726[0] + tmp_9806[0], tmp_9726[1] + tmp_9806[1], tmp_9726[2] + tmp_9806[2]];
    signal tmp_9808[3] <== [6696862914220145683 * tmp_9807[0], 6696862914220145683 * tmp_9807[1], 6696862914220145683 * tmp_9807[2]];
    signal tmp_9809[3] <== [tmp_9804[0] + tmp_9808[0], tmp_9804[1] + tmp_9808[1], tmp_9804[2] + tmp_9808[2]];
    signal tmp_9810[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9811[3] <== [tmp_9810[0] * 15192748352493407589, tmp_9810[1] * 15192748352493407589, tmp_9810[2] * 15192748352493407589];
    signal tmp_9812[3] <== [tmp_9731[0] + tmp_9811[0], tmp_9731[1] + tmp_9811[1], tmp_9731[2] + tmp_9811[2]];
    signal tmp_9813[3] <== [8135626180423978532 * tmp_9812[0], 8135626180423978532 * tmp_9812[1], 8135626180423978532 * tmp_9812[2]];
    signal tmp_9814[3] <== [tmp_9809[0] + tmp_9813[0], tmp_9809[1] + tmp_9813[1], tmp_9809[2] + tmp_9813[2]];
    signal tmp_9815[3] <== [tmp_7491[0] + 14981760075018578211, tmp_7491[1], tmp_7491[2]];
    signal tmp_9816[3] <== [tmp_9815[0] * 6531696571495839637, tmp_9815[1] * 6531696571495839637, tmp_9815[2] * 6531696571495839637];
    signal tmp_9817[3] <== [tmp_9736[0] + tmp_9816[0], tmp_9736[1] + tmp_9816[1], tmp_9736[2] + tmp_9816[2]];
    signal tmp_9818[3] <== [17786688261949651693 * tmp_9817[0], 17786688261949651693 * tmp_9817[1], 17786688261949651693 * tmp_9817[2]];
    signal tmp_9819[3] <== [tmp_9814[0] + tmp_9818[0], tmp_9814[1] + tmp_9818[1], tmp_9814[2] + tmp_9818[2]];
    signal tmp_9820[3] <== [tmp_9819[0] * 1, tmp_9819[1] * 1, tmp_9819[2] * 1];
    signal tmp_9821[3] <== [evals[90][0] - tmp_9820[0], evals[90][1] - tmp_9820[1], evals[90][2] - tmp_9820[2]];
    signal tmp_9822[3] <== CMul()(tmp_6068, tmp_9821);
    signal tmp_9823[3] <== [tmp_9743[0] + tmp_9822[0], tmp_9743[1] + tmp_9822[1], tmp_9743[2] + tmp_9822[2]];
    signal tmp_9824[3] <== CMul()(challengeQ, tmp_9823);
    signal tmp_9825[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9826[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9827[3] <== [tmp_9826[0] * 15537013571472701229, tmp_9826[1] * 15537013571472701229, tmp_9826[2] * 15537013571472701229];
    signal tmp_9828[3] <== [tmp_9747[0] + tmp_9827[0], tmp_9747[1] + tmp_9827[1], tmp_9747[2] + tmp_9827[2]];
    signal tmp_9829[3] <== [16947245409080232829 * tmp_9828[0], 16947245409080232829 * tmp_9828[1], 16947245409080232829 * tmp_9828[2]];
    signal tmp_9830[3] <== [tmp_9825[0] + tmp_9829[0], tmp_9825[1] + tmp_9829[1], tmp_9825[2] + tmp_9829[2]];
    signal tmp_9831[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9832[3] <== [tmp_9831[0] * 16947559881337334737, tmp_9831[1] * 16947559881337334737, tmp_9831[2] * 16947559881337334737];
    signal tmp_9833[3] <== [tmp_9752[0] + tmp_9832[0], tmp_9752[1] + tmp_9832[1], tmp_9752[2] + tmp_9832[2]];
    signal tmp_9834[3] <== [15778599751024291825 * tmp_9833[0], 15778599751024291825 * tmp_9833[1], 15778599751024291825 * tmp_9833[2]];
    signal tmp_9835[3] <== [tmp_9830[0] + tmp_9834[0], tmp_9830[1] + tmp_9834[1], tmp_9830[2] + tmp_9834[2]];
    signal tmp_9836[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9837[3] <== [tmp_9836[0] * 7493831729793585316, tmp_9836[1] * 7493831729793585316, tmp_9836[2] * 7493831729793585316];
    signal tmp_9838[3] <== [tmp_9757[0] + tmp_9837[0], tmp_9757[1] + tmp_9837[1], tmp_9757[2] + tmp_9837[2]];
    signal tmp_9839[3] <== [12554573928367352644 * tmp_9838[0], 12554573928367352644 * tmp_9838[1], 12554573928367352644 * tmp_9838[2]];
    signal tmp_9840[3] <== [tmp_9835[0] + tmp_9839[0], tmp_9835[1] + tmp_9839[1], tmp_9835[2] + tmp_9839[2]];
    signal tmp_9841[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9842[3] <== [tmp_9841[0] * 1623918663904207169, tmp_9841[1] * 1623918663904207169, tmp_9841[2] * 1623918663904207169];
    signal tmp_9843[3] <== [tmp_9762[0] + tmp_9842[0], tmp_9762[1] + tmp_9842[1], tmp_9762[2] + tmp_9842[2]];
    signal tmp_9844[3] <== [17305511116721035138 * tmp_9843[0], 17305511116721035138 * tmp_9843[1], 17305511116721035138 * tmp_9843[2]];
    signal tmp_9845[3] <== [tmp_9840[0] + tmp_9844[0], tmp_9840[1] + tmp_9844[1], tmp_9840[2] + tmp_9844[2]];
    signal tmp_9846[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9847[3] <== [tmp_9846[0] * 4687160973583834774, tmp_9846[1] * 4687160973583834774, tmp_9846[2] * 4687160973583834774];
    signal tmp_9848[3] <== [tmp_9767[0] + tmp_9847[0], tmp_9767[1] + tmp_9847[1], tmp_9767[2] + tmp_9847[2]];
    signal tmp_9849[3] <== [14036371303403364899 * tmp_9848[0], 14036371303403364899 * tmp_9848[1], 14036371303403364899 * tmp_9848[2]];
    signal tmp_9850[3] <== [tmp_9845[0] + tmp_9849[0], tmp_9845[1] + tmp_9849[1], tmp_9845[2] + tmp_9849[2]];
    signal tmp_9851[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9852[3] <== [tmp_9851[0] * 7481496270331462357, tmp_9851[1] * 7481496270331462357, tmp_9851[2] * 7481496270331462357];
    signal tmp_9853[3] <== [tmp_9772[0] + tmp_9852[0], tmp_9772[1] + tmp_9852[1], tmp_9772[2] + tmp_9852[2]];
    signal tmp_9854[3] <== [17810786690047806733 * tmp_9853[0], 17810786690047806733 * tmp_9853[1], 17810786690047806733 * tmp_9853[2]];
    signal tmp_9855[3] <== [tmp_9850[0] + tmp_9854[0], tmp_9850[1] + tmp_9854[1], tmp_9850[2] + tmp_9854[2]];
    signal tmp_9856[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9857[3] <== [tmp_9856[0] * 15777438269644270173, tmp_9856[1] * 15777438269644270173, tmp_9856[2] * 15777438269644270173];
    signal tmp_9858[3] <== [tmp_9777[0] + tmp_9857[0], tmp_9777[1] + tmp_9857[1], tmp_9777[2] + tmp_9857[2]];
    signal tmp_9859[3] <== [2212888412728595072 * tmp_9858[0], 2212888412728595072 * tmp_9858[1], 2212888412728595072 * tmp_9858[2]];
    signal tmp_9860[3] <== [tmp_9855[0] + tmp_9859[0], tmp_9855[1] + tmp_9859[1], tmp_9855[2] + tmp_9859[2]];
    signal tmp_9861[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9862[3] <== [tmp_9861[0] * 12845383926406511601, tmp_9861[1] * 12845383926406511601, tmp_9861[2] * 12845383926406511601];
    signal tmp_9863[3] <== [tmp_9782[0] + tmp_9862[0], tmp_9782[1] + tmp_9862[1], tmp_9782[2] + tmp_9862[2]];
    signal tmp_9864[3] <== [11824922540475901574 * tmp_9863[0], 11824922540475901574 * tmp_9863[1], 11824922540475901574 * tmp_9863[2]];
    signal tmp_9865[3] <== [tmp_9860[0] + tmp_9864[0], tmp_9860[1] + tmp_9864[1], tmp_9860[2] + tmp_9864[2]];
    signal tmp_9866[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9867[3] <== [tmp_9866[0] * 11148510750866669007, tmp_9866[1] * 11148510750866669007, tmp_9866[2] * 11148510750866669007];
    signal tmp_9868[3] <== [tmp_9787[0] + tmp_9867[0], tmp_9787[1] + tmp_9867[1], tmp_9787[2] + tmp_9867[2]];
    signal tmp_9869[3] <== [14117971470759675313 * tmp_9868[0], 14117971470759675313 * tmp_9868[1], 14117971470759675313 * tmp_9868[2]];
    signal tmp_9870[3] <== [tmp_9865[0] + tmp_9869[0], tmp_9865[1] + tmp_9869[1], tmp_9865[2] + tmp_9869[2]];
    signal tmp_9871[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9872[3] <== [tmp_9871[0] * 15318591593009208330, tmp_9871[1] * 15318591593009208330, tmp_9871[2] * 15318591593009208330];
    signal tmp_9873[3] <== [tmp_9792[0] + tmp_9872[0], tmp_9792[1] + tmp_9872[1], tmp_9792[2] + tmp_9872[2]];
    signal tmp_9874[3] <== [6803345862685226999 * tmp_9873[0], 6803345862685226999 * tmp_9873[1], 6803345862685226999 * tmp_9873[2]];
    signal tmp_9875[3] <== [tmp_9870[0] + tmp_9874[0], tmp_9870[1] + tmp_9874[1], tmp_9870[2] + tmp_9874[2]];
    signal tmp_9876[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9877[3] <== [tmp_9876[0] * 12116635019874345744, tmp_9876[1] * 12116635019874345744, tmp_9876[2] * 12116635019874345744];
    signal tmp_9878[3] <== [tmp_9797[0] + tmp_9877[0], tmp_9797[1] + tmp_9877[1], tmp_9797[2] + tmp_9877[2]];
    signal tmp_9879[3] <== [1790404652192668679 * tmp_9878[0], 1790404652192668679 * tmp_9878[1], 1790404652192668679 * tmp_9878[2]];
    signal tmp_9880[3] <== [tmp_9875[0] + tmp_9879[0], tmp_9875[1] + tmp_9879[1], tmp_9875[2] + tmp_9879[2]];
    signal tmp_9881[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9882[3] <== [tmp_9881[0] * 10427984200609223037, tmp_9881[1] * 10427984200609223037, tmp_9881[2] * 10427984200609223037];
    signal tmp_9883[3] <== [tmp_9802[0] + tmp_9882[0], tmp_9802[1] + tmp_9882[1], tmp_9802[2] + tmp_9882[2]];
    signal tmp_9884[3] <== [9306702297937622557 * tmp_9883[0], 9306702297937622557 * tmp_9883[1], 9306702297937622557 * tmp_9883[2]];
    signal tmp_9885[3] <== [tmp_9880[0] + tmp_9884[0], tmp_9880[1] + tmp_9884[1], tmp_9880[2] + tmp_9884[2]];
    signal tmp_9886[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9887[3] <== [tmp_9886[0] * 1275063075010481368, tmp_9886[1] * 1275063075010481368, tmp_9886[2] * 1275063075010481368];
    signal tmp_9888[3] <== [tmp_9807[0] + tmp_9887[0], tmp_9807[1] + tmp_9887[1], tmp_9807[2] + tmp_9887[2]];
    signal tmp_9889[3] <== [17976990147333300636 * tmp_9888[0], 17976990147333300636 * tmp_9888[1], 17976990147333300636 * tmp_9888[2]];
    signal tmp_9890[3] <== [tmp_9885[0] + tmp_9889[0], tmp_9885[1] + tmp_9889[1], tmp_9885[2] + tmp_9889[2]];
    signal tmp_9891[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9892[3] <== [tmp_9891[0] * 17363335480109898214, tmp_9891[1] * 17363335480109898214, tmp_9891[2] * 17363335480109898214];
    signal tmp_9893[3] <== [tmp_9812[0] + tmp_9892[0], tmp_9812[1] + tmp_9892[1], tmp_9812[2] + tmp_9892[2]];
    signal tmp_9894[3] <== [14238877083147544792 * tmp_9893[0], 14238877083147544792 * tmp_9893[1], 14238877083147544792 * tmp_9893[2]];
    signal tmp_9895[3] <== [tmp_9890[0] + tmp_9894[0], tmp_9890[1] + tmp_9894[1], tmp_9890[2] + tmp_9894[2]];
    signal tmp_9896[3] <== [tmp_7503[0] + 2982104952642540993, tmp_7503[1], tmp_7503[2]];
    signal tmp_9897[3] <== [tmp_9896[0] * 8340043298775904670, tmp_9896[1] * 8340043298775904670, tmp_9896[2] * 8340043298775904670];
    signal tmp_9898[3] <== [tmp_9817[0] + tmp_9897[0], tmp_9817[1] + tmp_9897[1], tmp_9817[2] + tmp_9897[2]];
    signal tmp_9899[3] <== [8744547697144458331 * tmp_9898[0], 8744547697144458331 * tmp_9898[1], 8744547697144458331 * tmp_9898[2]];
    signal tmp_9900[3] <== [tmp_9895[0] + tmp_9899[0], tmp_9895[1] + tmp_9899[1], tmp_9895[2] + tmp_9899[2]];
    signal tmp_9901[3] <== [tmp_9900[0] * 1, tmp_9900[1] * 1, tmp_9900[2] * 1];
    signal tmp_9902[3] <== [evals[91][0] - tmp_9901[0], evals[91][1] - tmp_9901[1], evals[91][2] - tmp_9901[2]];
    signal tmp_9903[3] <== CMul()(tmp_6068, tmp_9902);
    signal tmp_9904[3] <== [tmp_9824[0] + tmp_9903[0], tmp_9824[1] + tmp_9903[1], tmp_9824[2] + tmp_9903[2]];
    signal tmp_9905[3] <== CMul()(challengeQ, tmp_9904);
    signal tmp_9906[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_9907[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9908[3] <== [tmp_9907[0] * 18432038191863482059, tmp_9907[1] * 18432038191863482059, tmp_9907[2] * 18432038191863482059];
    signal tmp_9909[3] <== [tmp_9828[0] + tmp_9908[0], tmp_9828[1] + tmp_9908[1], tmp_9828[2] + tmp_9908[2]];
    signal tmp_9910[3] <== [3220543931095753511 * tmp_9909[0], 3220543931095753511 * tmp_9909[1], 3220543931095753511 * tmp_9909[2]];
    signal tmp_9911[3] <== [tmp_9906[0] + tmp_9910[0], tmp_9906[1] + tmp_9910[1], tmp_9906[2] + tmp_9910[2]];
    signal tmp_9912[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9913[3] <== [tmp_9912[0] * 5432413814338603108, tmp_9912[1] * 5432413814338603108, tmp_9912[2] * 5432413814338603108];
    signal tmp_9914[3] <== [tmp_9833[0] + tmp_9913[0], tmp_9833[1] + tmp_9913[1], tmp_9833[2] + tmp_9913[2]];
    signal tmp_9915[3] <== [7401724255027812990 * tmp_9914[0], 7401724255027812990 * tmp_9914[1], 7401724255027812990 * tmp_9914[2]];
    signal tmp_9916[3] <== [tmp_9911[0] + tmp_9915[0], tmp_9911[1] + tmp_9915[1], tmp_9911[2] + tmp_9915[2]];
    signal tmp_9917[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9918[3] <== [tmp_9917[0] * 11454376156859769836, tmp_9917[1] * 11454376156859769836, tmp_9917[2] * 11454376156859769836];
    signal tmp_9919[3] <== [tmp_9838[0] + tmp_9918[0], tmp_9838[1] + tmp_9918[1], tmp_9838[2] + tmp_9918[2]];
    signal tmp_9920[3] <== [17813499879760129016 * tmp_9919[0], 17813499879760129016 * tmp_9919[1], 17813499879760129016 * tmp_9919[2]];
    signal tmp_9921[3] <== [tmp_9916[0] + tmp_9920[0], tmp_9916[1] + tmp_9920[1], tmp_9916[2] + tmp_9920[2]];
    signal tmp_9922[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9923[3] <== [tmp_9922[0] * 3406012298082584781, tmp_9922[1] * 3406012298082584781, tmp_9922[2] * 3406012298082584781];
    signal tmp_9924[3] <== [tmp_9843[0] + tmp_9923[0], tmp_9843[1] + tmp_9923[1], tmp_9843[2] + tmp_9923[2]];
    signal tmp_9925[3] <== [5428548591833763345 * tmp_9924[0], 5428548591833763345 * tmp_9924[1], 5428548591833763345 * tmp_9924[2]];
    signal tmp_9926[3] <== [tmp_9921[0] + tmp_9925[0], tmp_9921[1] + tmp_9925[1], tmp_9921[2] + tmp_9925[2]];
    signal tmp_9927[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9928[3] <== [tmp_9927[0] * 14588634923383545112, tmp_9927[1] * 14588634923383545112, tmp_9927[2] * 14588634923383545112];
    signal tmp_9929[3] <== [tmp_9848[0] + tmp_9928[0], tmp_9848[1] + tmp_9928[1], tmp_9848[2] + tmp_9928[2]];
    signal tmp_9930[3] <== [6131968098119340952 * tmp_9929[0], 6131968098119340952 * tmp_9929[1], 6131968098119340952 * tmp_9929[2]];
    signal tmp_9931[3] <== [tmp_9926[0] + tmp_9930[0], tmp_9926[1] + tmp_9930[1], tmp_9926[2] + tmp_9930[2]];
    signal tmp_9932[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9933[3] <== [tmp_9932[0] * 7967410531777373627, tmp_9932[1] * 7967410531777373627, tmp_9932[2] * 7967410531777373627];
    signal tmp_9934[3] <== [tmp_9853[0] + tmp_9933[0], tmp_9853[1] + tmp_9933[1], tmp_9853[2] + tmp_9933[2]];
    signal tmp_9935[3] <== [2074496711530634450 * tmp_9934[0], 2074496711530634450 * tmp_9934[1], 2074496711530634450 * tmp_9934[2]];
    signal tmp_9936[3] <== [tmp_9931[0] + tmp_9935[0], tmp_9931[1] + tmp_9935[1], tmp_9931[2] + tmp_9935[2]];
    signal tmp_9937[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9938[3] <== [tmp_9937[0] * 5132439848537977263, tmp_9937[1] * 5132439848537977263, tmp_9937[2] * 5132439848537977263];
    signal tmp_9939[3] <== [tmp_9858[0] + tmp_9938[0], tmp_9858[1] + tmp_9938[1], tmp_9858[2] + tmp_9938[2]];
    signal tmp_9940[3] <== [5882024272570382639 * tmp_9939[0], 5882024272570382639 * tmp_9939[1], 5882024272570382639 * tmp_9939[2]];
    signal tmp_9941[3] <== [tmp_9936[0] + tmp_9940[0], tmp_9936[1] + tmp_9940[1], tmp_9936[2] + tmp_9940[2]];
    signal tmp_9942[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9943[3] <== [tmp_9942[0] * 9090563597932543104, tmp_9942[1] * 9090563597932543104, tmp_9942[2] * 9090563597932543104];
    signal tmp_9944[3] <== [tmp_9863[0] + tmp_9943[0], tmp_9863[1] + tmp_9943[1], tmp_9863[2] + tmp_9943[2]];
    signal tmp_9945[3] <== [12970340649830174380 * tmp_9944[0], 12970340649830174380 * tmp_9944[1], 12970340649830174380 * tmp_9944[2]];
    signal tmp_9946[3] <== [tmp_9941[0] + tmp_9945[0], tmp_9941[1] + tmp_9945[1], tmp_9941[2] + tmp_9945[2]];
    signal tmp_9947[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9948[3] <== [tmp_9947[0] * 16222775232712330818, tmp_9947[1] * 16222775232712330818, tmp_9947[2] * 16222775232712330818];
    signal tmp_9949[3] <== [tmp_9868[0] + tmp_9948[0], tmp_9868[1] + tmp_9948[1], tmp_9868[2] + tmp_9948[2]];
    signal tmp_9950[3] <== [4251380188844404593 * tmp_9949[0], 4251380188844404593 * tmp_9949[1], 4251380188844404593 * tmp_9949[2]];
    signal tmp_9951[3] <== [tmp_9946[0] + tmp_9950[0], tmp_9946[1] + tmp_9950[1], tmp_9946[2] + tmp_9950[2]];
    signal tmp_9952[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9953[3] <== [tmp_9952[0] * 14947628689583127907, tmp_9952[1] * 14947628689583127907, tmp_9952[2] * 14947628689583127907];
    signal tmp_9954[3] <== [tmp_9873[0] + tmp_9953[0], tmp_9873[1] + tmp_9953[1], tmp_9873[2] + tmp_9953[2]];
    signal tmp_9955[3] <== [5246780099262719987 * tmp_9954[0], 5246780099262719987 * tmp_9954[1], 5246780099262719987 * tmp_9954[2]];
    signal tmp_9956[3] <== [tmp_9951[0] + tmp_9955[0], tmp_9951[1] + tmp_9955[1], tmp_9951[2] + tmp_9955[2]];
    signal tmp_9957[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9958[3] <== [tmp_9957[0] * 800947794585510760, tmp_9957[1] * 800947794585510760, tmp_9957[2] * 800947794585510760];
    signal tmp_9959[3] <== [tmp_9878[0] + tmp_9958[0], tmp_9878[1] + tmp_9958[1], tmp_9878[2] + tmp_9958[2]];
    signal tmp_9960[3] <== [12265306642981888259 * tmp_9959[0], 12265306642981888259 * tmp_9959[1], 12265306642981888259 * tmp_9959[2]];
    signal tmp_9961[3] <== [tmp_9956[0] + tmp_9960[0], tmp_9956[1] + tmp_9960[1], tmp_9956[2] + tmp_9960[2]];
    signal tmp_9962[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9963[3] <== [tmp_9962[0] * 3863721566387520445, tmp_9962[1] * 3863721566387520445, tmp_9962[2] * 3863721566387520445];
    signal tmp_9964[3] <== [tmp_9883[0] + tmp_9963[0], tmp_9883[1] + tmp_9963[1], tmp_9883[2] + tmp_9963[2]];
    signal tmp_9965[3] <== [11845579913726276983 * tmp_9964[0], 11845579913726276983 * tmp_9964[1], 11845579913726276983 * tmp_9964[2]];
    signal tmp_9966[3] <== [tmp_9961[0] + tmp_9965[0], tmp_9961[1] + tmp_9965[1], tmp_9961[2] + tmp_9965[2]];
    signal tmp_9967[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9968[3] <== [tmp_9967[0] * 8395171295366482951, tmp_9967[1] * 8395171295366482951, tmp_9967[2] * 8395171295366482951];
    signal tmp_9969[3] <== [tmp_9888[0] + tmp_9968[0], tmp_9888[1] + tmp_9968[1], tmp_9888[2] + tmp_9968[2]];
    signal tmp_9970[3] <== [4161015590580980279 * tmp_9969[0], 4161015590580980279 * tmp_9969[1], 4161015590580980279 * tmp_9969[2]];
    signal tmp_9971[3] <== [tmp_9966[0] + tmp_9970[0], tmp_9966[1] + tmp_9970[1], tmp_9966[2] + tmp_9970[2]];
    signal tmp_9972[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9973[3] <== [tmp_9972[0] * 13583398811995500171, tmp_9972[1] * 13583398811995500171, tmp_9972[2] * 13583398811995500171];
    signal tmp_9974[3] <== [tmp_9893[0] + tmp_9973[0], tmp_9893[1] + tmp_9973[1], tmp_9893[2] + tmp_9973[2]];
    signal tmp_9975[3] <== [16142970107084859093 * tmp_9974[0], 16142970107084859093 * tmp_9974[1], 16142970107084859093 * tmp_9974[2]];
    signal tmp_9976[3] <== [tmp_9971[0] + tmp_9975[0], tmp_9971[1] + tmp_9975[1], tmp_9971[2] + tmp_9975[2]];
    signal tmp_9977[3] <== [tmp_7514[0] + 3183805956781626142, tmp_7514[1], tmp_7514[2]];
    signal tmp_9978[3] <== [tmp_9977[0] * 11308879932982657557, tmp_9977[1] * 11308879932982657557, tmp_9977[2] * 11308879932982657557];
    signal tmp_9979[3] <== [tmp_9898[0] + tmp_9978[0], tmp_9898[1] + tmp_9978[1], tmp_9898[2] + tmp_9978[2]];
    signal tmp_9980[3] <== [10242614418821665472 * tmp_9979[0], 10242614418821665472 * tmp_9979[1], 10242614418821665472 * tmp_9979[2]];
    signal tmp_9981[3] <== [tmp_9976[0] + tmp_9980[0], tmp_9976[1] + tmp_9980[1], tmp_9976[2] + tmp_9980[2]];
    signal tmp_9982[3] <== [tmp_9981[0] * 1, tmp_9981[1] * 1, tmp_9981[2] * 1];
    signal tmp_9983[3] <== [evals[92][0] - tmp_9982[0], evals[92][1] - tmp_9982[1], evals[92][2] - tmp_9982[2]];
    signal tmp_9984[3] <== CMul()(tmp_6068, tmp_9983);
    signal tmp_9985[3] <== [tmp_9905[0] + tmp_9984[0], tmp_9905[1] + tmp_9984[1], tmp_9905[2] + tmp_9984[2]];
    tmp_9986 <== CMul()(challengeQ, tmp_9985);
    signal tmp_9987[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_9988[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_9989[3] <== [tmp_9988[0] * 3130238529562802618, tmp_9988[1] * 3130238529562802618, tmp_9988[2] * 3130238529562802618];
    tmp_9990 <== [tmp_9909[0] + tmp_9989[0], tmp_9909[1] + tmp_9989[1], tmp_9909[2] + tmp_9989[2]];
    signal tmp_9991[3] <== [8578786096637244792 * tmp_9990[0], 8578786096637244792 * tmp_9990[1], 8578786096637244792 * tmp_9990[2]];
    signal tmp_9992[3] <== [tmp_9987[0] + tmp_9991[0], tmp_9987[1] + tmp_9991[1], tmp_9987[2] + tmp_9991[2]];
    signal tmp_9993[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_9994[3] <== [tmp_9993[0] * 17571383761285423616, tmp_9993[1] * 17571383761285423616, tmp_9993[2] * 17571383761285423616];
    tmp_9995 <== [tmp_9914[0] + tmp_9994[0], tmp_9914[1] + tmp_9994[1], tmp_9914[2] + tmp_9994[2]];
    signal tmp_9996[3] <== [10017940286670940083 * tmp_9995[0], 10017940286670940083 * tmp_9995[1], 10017940286670940083 * tmp_9995[2]];
    signal tmp_9997[3] <== [tmp_9992[0] + tmp_9996[0], tmp_9992[1] + tmp_9996[1], tmp_9992[2] + tmp_9996[2]];
    signal tmp_9998[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_9999[3] <== [tmp_9998[0] * 11604431936999842251, tmp_9998[1] * 11604431936999842251, tmp_9998[2] * 11604431936999842251];
    tmp_10000 <== [tmp_9919[0] + tmp_9999[0], tmp_9919[1] + tmp_9999[1], tmp_9919[2] + tmp_9999[2]];
    signal tmp_10001[3] <== [6061757955429109029 * tmp_10000[0], 6061757955429109029 * tmp_10000[1], 6061757955429109029 * tmp_10000[2]];
    signal tmp_10002[3] <== [tmp_9997[0] + tmp_10001[0], tmp_9997[1] + tmp_10001[1], tmp_9997[2] + tmp_10001[2]];
    signal tmp_10003[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10004[3] <== [tmp_10003[0] * 1643416478018396715, tmp_10003[1] * 1643416478018396715, tmp_10003[2] * 1643416478018396715];
    tmp_10005 <== [tmp_9924[0] + tmp_10004[0], tmp_9924[1] + tmp_10004[1], tmp_9924[2] + tmp_10004[2]];
    signal tmp_10006[3] <== [6830443121764588145 * tmp_10005[0], 6830443121764588145 * tmp_10005[1], 6830443121764588145 * tmp_10005[2]];
    signal tmp_10007[3] <== [tmp_10002[0] + tmp_10006[0], tmp_10002[1] + tmp_10006[1], tmp_10002[2] + tmp_10006[2]];
    signal tmp_10008[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10009[3] <== [tmp_10008[0] * 9529320288398332687, tmp_10008[1] * 9529320288398332687, tmp_10008[2] * 9529320288398332687];
    tmp_10010 <== [tmp_9929[0] + tmp_10009[0], tmp_9929[1] + tmp_10009[1], tmp_9929[2] + tmp_10009[2]];
    signal tmp_10011[3] <== [6572675368485775153 * tmp_10010[0], 6572675368485775153 * tmp_10010[1], 6572675368485775153 * tmp_10010[2]];
    signal tmp_10012[3] <== [tmp_10007[0] + tmp_10011[0], tmp_10007[1] + tmp_10011[1], tmp_10007[2] + tmp_10011[2]];
    signal tmp_10013[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10014[3] <== [tmp_10013[0] * 2943500820182364631, tmp_10013[1] * 2943500820182364631, tmp_10013[2] * 2943500820182364631];
    tmp_10015 <== [tmp_9934[0] + tmp_10014[0], tmp_9934[1] + tmp_10014[1], tmp_9934[2] + tmp_10014[2]];
    signal tmp_10016[3] <== [2570198071091962115 * tmp_10015[0], 2570198071091962115 * tmp_10015[1], 2570198071091962115 * tmp_10015[2]];
    signal tmp_10017[3] <== [tmp_10012[0] + tmp_10016[0], tmp_10012[1] + tmp_10016[1], tmp_10012[2] + tmp_10016[2]];
    signal tmp_10018[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10019[3] <== [tmp_10018[0] * 5642253925908086286, tmp_10018[1] * 5642253925908086286, tmp_10018[2] * 5642253925908086286];
    tmp_10020 <== [tmp_9939[0] + tmp_10019[0], tmp_9939[1] + tmp_10019[1], tmp_9939[2] + tmp_10019[2]];
    signal tmp_10021[3] <== [4859557487763691165 * tmp_10020[0], 4859557487763691165 * tmp_10020[1], 4859557487763691165 * tmp_10020[2]];
    signal tmp_10022[3] <== [tmp_10017[0] + tmp_10021[0], tmp_10017[1] + tmp_10021[1], tmp_10017[2] + tmp_10021[2]];
    signal tmp_10023[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10024[3] <== [tmp_10023[0] * 1584863777448828811, tmp_10023[1] * 1584863777448828811, tmp_10023[2] * 1584863777448828811];
    tmp_10025 <== [tmp_9944[0] + tmp_10024[0], tmp_9944[1] + tmp_10024[1], tmp_9944[2] + tmp_10024[2]];
    signal tmp_10026[3] <== [5904682032025033039 * tmp_10025[0], 5904682032025033039 * tmp_10025[1], 5904682032025033039 * tmp_10025[2]];
    signal tmp_10027[3] <== [tmp_10022[0] + tmp_10026[0], tmp_10022[1] + tmp_10026[1], tmp_10022[2] + tmp_10026[2]];
    signal tmp_10028[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10029[3] <== [tmp_10028[0] * 12918821123083034440, tmp_10028[1] * 12918821123083034440, tmp_10028[2] * 12918821123083034440];
    tmp_10030 <== [tmp_9949[0] + tmp_10029[0], tmp_9949[1] + tmp_10029[1], tmp_9949[2] + tmp_10029[2]];
    signal tmp_10031[3] <== [13831434945980884721 * tmp_10030[0], 13831434945980884721 * tmp_10030[1], 13831434945980884721 * tmp_10030[2]];
    signal tmp_10032[3] <== [tmp_10027[0] + tmp_10031[0], tmp_10027[1] + tmp_10031[1], tmp_10027[2] + tmp_10031[2]];
    signal tmp_10033[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10034[3] <== [tmp_10033[0] * 4794647653414383806, tmp_10033[1] * 4794647653414383806, tmp_10033[2] * 4794647653414383806];
    tmp_10035 <== [tmp_9954[0] + tmp_10034[0], tmp_9954[1] + tmp_10034[1], tmp_9954[2] + tmp_10034[2]];
    signal tmp_10036[3] <== [13071285619755217600 * tmp_10035[0], 13071285619755217600 * tmp_10035[1], 13071285619755217600 * tmp_10035[2]];
    signal tmp_10037[3] <== [tmp_10032[0] + tmp_10036[0], tmp_10032[1] + tmp_10036[1], tmp_10032[2] + tmp_10036[2]];
    signal tmp_10038[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10039[3] <== [tmp_10038[0] * 12339757025454833750, tmp_10038[1] * 12339757025454833750, tmp_10038[2] * 12339757025454833750];
    tmp_10040 <== [tmp_9959[0] + tmp_10039[0], tmp_9959[1] + tmp_10039[1], tmp_9959[2] + tmp_10039[2]];
    signal tmp_10041[3] <== [9745460330714099568 * tmp_10040[0], 9745460330714099568 * tmp_10040[1], 9745460330714099568 * tmp_10040[2]];
    signal tmp_10042[3] <== [tmp_10037[0] + tmp_10041[0], tmp_10037[1] + tmp_10041[1], tmp_10037[2] + tmp_10041[2]];
    signal tmp_10043[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10044[3] <== [tmp_10043[0] * 15902923065000667949, tmp_10043[1] * 15902923065000667949, tmp_10043[2] * 15902923065000667949];
    tmp_10045 <== [tmp_9964[0] + tmp_10044[0], tmp_9964[1] + tmp_10044[1], tmp_9964[2] + tmp_10044[2]];
    signal tmp_10046[3] <== [17565589964402351072 * tmp_10045[0], 17565589964402351072 * tmp_10045[1], 17565589964402351072 * tmp_10045[2]];
    signal tmp_10047[3] <== [tmp_10042[0] + tmp_10046[0], tmp_10042[1] + tmp_10046[1], tmp_10042[2] + tmp_10046[2]];
    signal tmp_10048[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10049[3] <== [tmp_10048[0] * 945086332862338525, tmp_10048[1] * 945086332862338525, tmp_10048[2] * 945086332862338525];
    tmp_10050 <== [tmp_9969[0] + tmp_10049[0], tmp_9969[1] + tmp_10049[1], tmp_9969[2] + tmp_10049[2]];
    signal tmp_10051[3] <== [4478734179481674309 * tmp_10050[0], 4478734179481674309 * tmp_10050[1], 4478734179481674309 * tmp_10050[2]];
    signal tmp_10052[3] <== [tmp_10047[0] + tmp_10051[0], tmp_10047[1] + tmp_10051[1], tmp_10047[2] + tmp_10051[2]];
    signal tmp_10053[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10054[3] <== [tmp_10053[0] * 13520575880496481741, tmp_10053[1] * 13520575880496481741, tmp_10053[2] * 13520575880496481741];
    tmp_10055 <== [tmp_9974[0] + tmp_10054[0], tmp_9974[1] + tmp_10054[1], tmp_9974[2] + tmp_10054[2]];
    signal tmp_10056[3] <== [5964515412475082128 * tmp_10055[0], 5964515412475082128 * tmp_10055[1], 5964515412475082128 * tmp_10055[2]];
    tmp_10057 <== [tmp_10052[0] + tmp_10056[0], tmp_10052[1] + tmp_10056[1], tmp_10052[2] + tmp_10056[2]];
    signal tmp_10058[3] <== [tmp_7526[0] + 9076472642853520518, tmp_7526[1], tmp_7526[2]];
    signal tmp_10059[3] <== [tmp_10058[0] * 7361764812588359828, tmp_10058[1] * 7361764812588359828, tmp_10058[2] * 7361764812588359828];
    tmp_10060 <== [tmp_9979[0] + tmp_10059[0], tmp_9979[1] + tmp_10059[1], tmp_9979[2] + tmp_10059[2]];
    tmp_10061 <== [10182176813386478138 * tmp_10060[0], 10182176813386478138 * tmp_10060[1], 10182176813386478138 * tmp_10060[2]];
}

template VerifyEvaluationsChunks4() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_6068[3];
    signal input tmp_7538[3];
    signal input tmp_7550[3];
    signal input tmp_7562[3];
    signal input tmp_7574[3];
    signal input tmp_7586[3];
    signal input tmp_7598[3];
    signal input tmp_9986[3];
    signal input tmp_9990[3];
    signal input tmp_9995[3];
    signal input tmp_10000[3];
    signal input tmp_10005[3];
    signal input tmp_10010[3];
    signal input tmp_10015[3];
    signal input tmp_10020[3];
    signal input tmp_10025[3];
    signal input tmp_10030[3];
    signal input tmp_10035[3];
    signal input tmp_10040[3];
    signal input tmp_10045[3];
    signal input tmp_10050[3];
    signal input tmp_10055[3];
    signal input tmp_10057[3];
    signal input tmp_10060[3];
    signal input tmp_10061[3];

    signal output tmp_10901[3];
    signal output tmp_10955[3];
    signal output tmp_10960[3];
    signal output tmp_10965[3];
    signal output tmp_10970[3];
    signal output tmp_10975[3];
    signal output tmp_11060[3];
    signal tmp_10062[3] <== [tmp_10057[0] + tmp_10061[0], tmp_10057[1] + tmp_10061[1], tmp_10057[2] + tmp_10061[2]];
    signal tmp_10063[3] <== [tmp_10062[0] * 1, tmp_10062[1] * 1, tmp_10062[2] * 1];
    signal tmp_10064[3] <== [evals[93][0] - tmp_10063[0], evals[93][1] - tmp_10063[1], evals[93][2] - tmp_10063[2]];
    signal tmp_10065[3] <== CMul()(tmp_6068, tmp_10064);
    signal tmp_10066[3] <== [tmp_9986[0] + tmp_10065[0], tmp_9986[1] + tmp_10065[1], tmp_9986[2] + tmp_10065[2]];
    signal tmp_10067[3] <== CMul()(challengeQ, tmp_10066);
    signal tmp_10068[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10069[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10070[3] <== [tmp_10069[0] * 6014095723071284775, tmp_10069[1] * 6014095723071284775, tmp_10069[2] * 6014095723071284775];
    signal tmp_10071[3] <== [tmp_9990[0] + tmp_10070[0], tmp_9990[1] + tmp_10070[1], tmp_9990[2] + tmp_10070[2]];
    signal tmp_10072[3] <== [11235818329862088340 * tmp_10071[0], 11235818329862088340 * tmp_10071[1], 11235818329862088340 * tmp_10071[2]];
    signal tmp_10073[3] <== [tmp_10068[0] + tmp_10072[0], tmp_10068[1] + tmp_10072[1], tmp_10068[2] + tmp_10072[2]];
    signal tmp_10074[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10075[3] <== [tmp_10074[0] * 9204778852131994952, tmp_10074[1] * 9204778852131994952, tmp_10074[2] * 9204778852131994952];
    signal tmp_10076[3] <== [tmp_9995[0] + tmp_10075[0], tmp_9995[1] + tmp_10075[1], tmp_9995[2] + tmp_10075[2]];
    signal tmp_10077[3] <== [14470889044174589649 * tmp_10076[0], 14470889044174589649 * tmp_10076[1], 14470889044174589649 * tmp_10076[2]];
    signal tmp_10078[3] <== [tmp_10073[0] + tmp_10077[0], tmp_10073[1] + tmp_10077[1], tmp_10073[2] + tmp_10077[2]];
    signal tmp_10079[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10080[3] <== [tmp_10079[0] * 4971091590805790158, tmp_10079[1] * 4971091590805790158, tmp_10079[2] * 4971091590805790158];
    signal tmp_10081[3] <== [tmp_10000[0] + tmp_10080[0], tmp_10000[1] + tmp_10080[1], tmp_10000[2] + tmp_10080[2]];
    signal tmp_10082[3] <== [17369024544835219425 * tmp_10081[0], 17369024544835219425 * tmp_10081[1], 17369024544835219425 * tmp_10081[2]];
    signal tmp_10083[3] <== [tmp_10078[0] + tmp_10082[0], tmp_10078[1] + tmp_10082[1], tmp_10078[2] + tmp_10082[2]];
    signal tmp_10084[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10085[3] <== [tmp_10084[0] * 13132173167903710685, tmp_10084[1] * 13132173167903710685, tmp_10084[2] * 13132173167903710685];
    signal tmp_10086[3] <== [tmp_10005[0] + tmp_10085[0], tmp_10005[1] + tmp_10085[1], tmp_10005[2] + tmp_10085[2]];
    signal tmp_10087[3] <== [15549761383035162395 * tmp_10086[0], 15549761383035162395 * tmp_10086[1], 15549761383035162395 * tmp_10086[2]];
    signal tmp_10088[3] <== [tmp_10083[0] + tmp_10087[0], tmp_10083[1] + tmp_10087[1], tmp_10083[2] + tmp_10087[2]];
    signal tmp_10089[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10090[3] <== [tmp_10089[0] * 7151135351816094214, tmp_10089[1] * 7151135351816094214, tmp_10089[2] * 7151135351816094214];
    signal tmp_10091[3] <== [tmp_10010[0] + tmp_10090[0], tmp_10010[1] + tmp_10090[1], tmp_10010[2] + tmp_10090[2]];
    signal tmp_10092[3] <== [10101825905546027637 * tmp_10091[0], 10101825905546027637 * tmp_10091[1], 10101825905546027637 * tmp_10091[2]];
    signal tmp_10093[3] <== [tmp_10088[0] + tmp_10092[0], tmp_10088[1] + tmp_10092[1], tmp_10088[2] + tmp_10092[2]];
    signal tmp_10094[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10095[3] <== [tmp_10094[0] * 17995317069560704617, tmp_10094[1] * 17995317069560704617, tmp_10094[2] * 17995317069560704617];
    signal tmp_10096[3] <== [tmp_10015[0] + tmp_10095[0], tmp_10015[1] + tmp_10095[1], tmp_10015[2] + tmp_10095[2]];
    signal tmp_10097[3] <== [6574923001367612596 * tmp_10096[0], 6574923001367612596 * tmp_10096[1], 6574923001367612596 * tmp_10096[2]];
    signal tmp_10098[3] <== [tmp_10093[0] + tmp_10097[0], tmp_10093[1] + tmp_10097[1], tmp_10093[2] + tmp_10097[2]];
    signal tmp_10099[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10100[3] <== [tmp_10099[0] * 3798127953781171227, tmp_10099[1] * 3798127953781171227, tmp_10099[2] * 3798127953781171227];
    signal tmp_10101[3] <== [tmp_10020[0] + tmp_10100[0], tmp_10020[1] + tmp_10100[1], tmp_10020[2] + tmp_10100[2]];
    signal tmp_10102[3] <== [1494928778108945573 * tmp_10101[0], 1494928778108945573 * tmp_10101[1], 1494928778108945573 * tmp_10101[2]];
    signal tmp_10103[3] <== [tmp_10098[0] + tmp_10102[0], tmp_10098[1] + tmp_10102[1], tmp_10098[2] + tmp_10102[2]];
    signal tmp_10104[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10105[3] <== [tmp_10104[0] * 4569124295887334937, tmp_10104[1] * 4569124295887334937, tmp_10104[2] * 4569124295887334937];
    signal tmp_10106[3] <== [tmp_10025[0] + tmp_10105[0], tmp_10025[1] + tmp_10105[1], tmp_10025[2] + tmp_10105[2]];
    signal tmp_10107[3] <== [4609820205985602917 * tmp_10106[0], 4609820205985602917 * tmp_10106[1], 4609820205985602917 * tmp_10106[2]];
    signal tmp_10108[3] <== [tmp_10103[0] + tmp_10107[0], tmp_10103[1] + tmp_10107[1], tmp_10103[2] + tmp_10107[2]];
    signal tmp_10109[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10110[3] <== [tmp_10109[0] * 8937451988206592672, tmp_10109[1] * 8937451988206592672, tmp_10109[2] * 8937451988206592672];
    signal tmp_10111[3] <== [tmp_10030[0] + tmp_10110[0], tmp_10030[1] + tmp_10110[1], tmp_10030[2] + tmp_10110[2]];
    signal tmp_10112[3] <== [3061013547631929280 * tmp_10111[0], 3061013547631929280 * tmp_10111[1], 3061013547631929280 * tmp_10111[2]];
    signal tmp_10113[3] <== [tmp_10108[0] + tmp_10112[0], tmp_10108[1] + tmp_10112[1], tmp_10108[2] + tmp_10112[2]];
    signal tmp_10114[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10115[3] <== [tmp_10114[0] * 3515999524512876842, tmp_10114[1] * 3515999524512876842, tmp_10114[2] * 3515999524512876842];
    signal tmp_10116[3] <== [tmp_10035[0] + tmp_10115[0], tmp_10035[1] + tmp_10115[1], tmp_10035[2] + tmp_10115[2]];
    signal tmp_10117[3] <== [17109849163472234437 * tmp_10116[0], 17109849163472234437 * tmp_10116[1], 17109849163472234437 * tmp_10116[2]];
    signal tmp_10118[3] <== [tmp_10113[0] + tmp_10117[0], tmp_10113[1] + tmp_10117[1], tmp_10113[2] + tmp_10117[2]];
    signal tmp_10119[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10120[3] <== [tmp_10119[0] * 16966389141222274672, tmp_10119[1] * 16966389141222274672, tmp_10119[2] * 16966389141222274672];
    signal tmp_10121[3] <== [tmp_10040[0] + tmp_10120[0], tmp_10040[1] + tmp_10120[1], tmp_10040[2] + tmp_10120[2]];
    signal tmp_10122[3] <== [11697494665722803962 * tmp_10121[0], 11697494665722803962 * tmp_10121[1], 11697494665722803962 * tmp_10121[2]];
    signal tmp_10123[3] <== [tmp_10118[0] + tmp_10122[0], tmp_10118[1] + tmp_10122[1], tmp_10118[2] + tmp_10122[2]];
    signal tmp_10124[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10125[3] <== [tmp_10124[0] * 8034835433986334001, tmp_10124[1] * 8034835433986334001, tmp_10124[2] * 8034835433986334001];
    signal tmp_10126[3] <== [tmp_10045[0] + tmp_10125[0], tmp_10045[1] + tmp_10125[1], tmp_10045[2] + tmp_10125[2]];
    signal tmp_10127[3] <== [4706605688950171455 * tmp_10126[0], 4706605688950171455 * tmp_10126[1], 4706605688950171455 * tmp_10126[2]];
    signal tmp_10128[3] <== [tmp_10123[0] + tmp_10127[0], tmp_10123[1] + tmp_10127[1], tmp_10123[2] + tmp_10127[2]];
    signal tmp_10129[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10130[3] <== [tmp_10129[0] * 7985528271787349844, tmp_10129[1] * 7985528271787349844, tmp_10129[2] * 7985528271787349844];
    signal tmp_10131[3] <== [tmp_10050[0] + tmp_10130[0], tmp_10050[1] + tmp_10130[1], tmp_10050[2] + tmp_10130[2]];
    signal tmp_10132[3] <== [15438292502104786232 * tmp_10131[0], 15438292502104786232 * tmp_10131[1], 15438292502104786232 * tmp_10131[2]];
    signal tmp_10133[3] <== [tmp_10128[0] + tmp_10132[0], tmp_10128[1] + tmp_10132[1], tmp_10128[2] + tmp_10132[2]];
    signal tmp_10134[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10135[3] <== [tmp_10134[0] * 1825496924806641273, tmp_10134[1] * 1825496924806641273, tmp_10134[2] * 1825496924806641273];
    signal tmp_10136[3] <== [tmp_10055[0] + tmp_10135[0], tmp_10055[1] + tmp_10135[1], tmp_10055[2] + tmp_10135[2]];
    signal tmp_10137[3] <== [1979256218175145661 * tmp_10136[0], 1979256218175145661 * tmp_10136[1], 1979256218175145661 * tmp_10136[2]];
    signal tmp_10138[3] <== [tmp_10133[0] + tmp_10137[0], tmp_10133[1] + tmp_10137[1], tmp_10133[2] + tmp_10137[2]];
    signal tmp_10139[3] <== [tmp_7538[0] + 14285914829807850365, tmp_7538[1], tmp_7538[2]];
    signal tmp_10140[3] <== [tmp_10139[0] * 5094552554489483119, tmp_10139[1] * 5094552554489483119, tmp_10139[2] * 5094552554489483119];
    signal tmp_10141[3] <== [tmp_10060[0] + tmp_10140[0], tmp_10060[1] + tmp_10140[1], tmp_10060[2] + tmp_10140[2]];
    signal tmp_10142[3] <== [6777118549370094671 * tmp_10141[0], 6777118549370094671 * tmp_10141[1], 6777118549370094671 * tmp_10141[2]];
    signal tmp_10143[3] <== [tmp_10138[0] + tmp_10142[0], tmp_10138[1] + tmp_10142[1], tmp_10138[2] + tmp_10142[2]];
    signal tmp_10144[3] <== [tmp_10143[0] * 1, tmp_10143[1] * 1, tmp_10143[2] * 1];
    signal tmp_10145[3] <== [evals[94][0] - tmp_10144[0], evals[94][1] - tmp_10144[1], evals[94][2] - tmp_10144[2]];
    signal tmp_10146[3] <== CMul()(tmp_6068, tmp_10145);
    signal tmp_10147[3] <== [tmp_10067[0] + tmp_10146[0], tmp_10067[1] + tmp_10146[1], tmp_10067[2] + tmp_10146[2]];
    signal tmp_10148[3] <== CMul()(challengeQ, tmp_10147);
    signal tmp_10149[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10150[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10151[3] <== [tmp_10150[0] * 13058054427636566156, tmp_10150[1] * 13058054427636566156, tmp_10150[2] * 13058054427636566156];
    signal tmp_10152[3] <== [tmp_10071[0] + tmp_10151[0], tmp_10071[1] + tmp_10151[1], tmp_10071[2] + tmp_10151[2]];
    signal tmp_10153[3] <== [8370938923534542930 * tmp_10152[0], 8370938923534542930 * tmp_10152[1], 8370938923534542930 * tmp_10152[2]];
    signal tmp_10154[3] <== [tmp_10149[0] + tmp_10153[0], tmp_10149[1] + tmp_10153[1], tmp_10149[2] + tmp_10153[2]];
    signal tmp_10155[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10156[3] <== [tmp_10155[0] * 17494898710713280458, tmp_10155[1] * 17494898710713280458, tmp_10155[2] * 17494898710713280458];
    signal tmp_10157[3] <== [tmp_10076[0] + tmp_10156[0], tmp_10076[1] + tmp_10156[1], tmp_10076[2] + tmp_10156[2]];
    signal tmp_10158[3] <== [1280128970885626285 * tmp_10157[0], 1280128970885626285 * tmp_10157[1], 1280128970885626285 * tmp_10157[2]];
    signal tmp_10159[3] <== [tmp_10154[0] + tmp_10158[0], tmp_10154[1] + tmp_10158[1], tmp_10154[2] + tmp_10158[2]];
    signal tmp_10160[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10161[3] <== [tmp_10160[0] * 10556138419339233346, tmp_10160[1] * 10556138419339233346, tmp_10160[2] * 10556138419339233346];
    signal tmp_10162[3] <== [tmp_10081[0] + tmp_10161[0], tmp_10081[1] + tmp_10161[1], tmp_10081[2] + tmp_10161[2]];
    signal tmp_10163[3] <== [17941471055037515299 * tmp_10162[0], 17941471055037515299 * tmp_10162[1], 17941471055037515299 * tmp_10162[2]];
    signal tmp_10164[3] <== [tmp_10159[0] + tmp_10163[0], tmp_10159[1] + tmp_10163[1], tmp_10159[2] + tmp_10163[2]];
    signal tmp_10165[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10166[3] <== [tmp_10165[0] * 9446845158877430994, tmp_10165[1] * 9446845158877430994, tmp_10165[2] * 9446845158877430994];
    signal tmp_10167[3] <== [tmp_10086[0] + tmp_10166[0], tmp_10086[1] + tmp_10166[1], tmp_10086[2] + tmp_10166[2]];
    signal tmp_10168[3] <== [8584145824237830166 * tmp_10167[0], 8584145824237830166 * tmp_10167[1], 8584145824237830166 * tmp_10167[2]];
    signal tmp_10169[3] <== [tmp_10164[0] + tmp_10168[0], tmp_10164[1] + tmp_10168[1], tmp_10164[2] + tmp_10168[2]];
    signal tmp_10170[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10171[3] <== [tmp_10170[0] * 8319398780236037084, tmp_10170[1] * 8319398780236037084, tmp_10170[2] * 8319398780236037084];
    signal tmp_10172[3] <== [tmp_10091[0] + tmp_10171[0], tmp_10091[1] + tmp_10171[1], tmp_10091[2] + tmp_10171[2]];
    signal tmp_10173[3] <== [8210952786580820323 * tmp_10172[0], 8210952786580820323 * tmp_10172[1], 8210952786580820323 * tmp_10172[2]];
    signal tmp_10174[3] <== [tmp_10169[0] + tmp_10173[0], tmp_10169[1] + tmp_10173[1], tmp_10169[2] + tmp_10173[2]];
    signal tmp_10175[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10176[3] <== [tmp_10175[0] * 13582814433783971289, tmp_10175[1] * 13582814433783971289, tmp_10175[2] * 13582814433783971289];
    signal tmp_10177[3] <== [tmp_10096[0] + tmp_10176[0], tmp_10096[1] + tmp_10176[1], tmp_10096[2] + tmp_10176[2]];
    signal tmp_10178[3] <== [2433132954413441756 * tmp_10177[0], 2433132954413441756 * tmp_10177[1], 2433132954413441756 * tmp_10177[2]];
    signal tmp_10179[3] <== [tmp_10174[0] + tmp_10178[0], tmp_10174[1] + tmp_10178[1], tmp_10174[2] + tmp_10178[2]];
    signal tmp_10180[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10181[3] <== [tmp_10180[0] * 15000404693370233461, tmp_10180[1] * 15000404693370233461, tmp_10180[2] * 15000404693370233461];
    signal tmp_10182[3] <== [tmp_10101[0] + tmp_10181[0], tmp_10101[1] + tmp_10181[1], tmp_10101[2] + tmp_10181[2]];
    signal tmp_10183[3] <== [15662973610146691206 * tmp_10182[0], 15662973610146691206 * tmp_10182[1], 15662973610146691206 * tmp_10182[2]];
    signal tmp_10184[3] <== [tmp_10179[0] + tmp_10183[0], tmp_10179[1] + tmp_10183[1], tmp_10179[2] + tmp_10183[2]];
    signal tmp_10185[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10186[3] <== [tmp_10185[0] * 10753866658284722577, tmp_10185[1] * 10753866658284722577, tmp_10185[2] * 10753866658284722577];
    signal tmp_10187[3] <== [tmp_10106[0] + tmp_10186[0], tmp_10106[1] + tmp_10186[1], tmp_10106[2] + tmp_10186[2]];
    signal tmp_10188[3] <== [10026165261844553317 * tmp_10187[0], 10026165261844553317 * tmp_10187[1], 10026165261844553317 * tmp_10187[2]];
    signal tmp_10189[3] <== [tmp_10184[0] + tmp_10188[0], tmp_10184[1] + tmp_10188[1], tmp_10184[2] + tmp_10188[2]];
    signal tmp_10190[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10191[3] <== [tmp_10190[0] * 7462165540945749814, tmp_10190[1] * 7462165540945749814, tmp_10190[2] * 7462165540945749814];
    signal tmp_10192[3] <== [tmp_10111[0] + tmp_10191[0], tmp_10111[1] + tmp_10191[1], tmp_10111[2] + tmp_10191[2]];
    signal tmp_10193[3] <== [16296674388465118728 * tmp_10192[0], 16296674388465118728 * tmp_10192[1], 16296674388465118728 * tmp_10192[2]];
    signal tmp_10194[3] <== [tmp_10189[0] + tmp_10193[0], tmp_10189[1] + tmp_10193[1], tmp_10189[2] + tmp_10193[2]];
    signal tmp_10195[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10196[3] <== [tmp_10195[0] * 3904712121826807540, tmp_10195[1] * 3904712121826807540, tmp_10195[2] * 3904712121826807540];
    signal tmp_10197[3] <== [tmp_10116[0] + tmp_10196[0], tmp_10116[1] + tmp_10196[1], tmp_10116[2] + tmp_10196[2]];
    signal tmp_10198[3] <== [17271463318160608022 * tmp_10197[0], 17271463318160608022 * tmp_10197[1], 17271463318160608022 * tmp_10197[2]];
    signal tmp_10199[3] <== [tmp_10194[0] + tmp_10198[0], tmp_10194[1] + tmp_10198[1], tmp_10194[2] + tmp_10198[2]];
    signal tmp_10200[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10201[3] <== [tmp_10200[0] * 15631290702336306626, tmp_10200[1] * 15631290702336306626, tmp_10200[2] * 15631290702336306626];
    signal tmp_10202[3] <== [tmp_10121[0] + tmp_10201[0], tmp_10121[1] + tmp_10201[1], tmp_10121[2] + tmp_10201[2]];
    signal tmp_10203[3] <== [8140140055656693761 * tmp_10202[0], 8140140055656693761 * tmp_10202[1], 8140140055656693761 * tmp_10202[2]];
    signal tmp_10204[3] <== [tmp_10199[0] + tmp_10203[0], tmp_10199[1] + tmp_10203[1], tmp_10199[2] + tmp_10203[2]];
    signal tmp_10205[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10206[3] <== [tmp_10205[0] * 8501326043416122835, tmp_10205[1] * 8501326043416122835, tmp_10205[2] * 8501326043416122835];
    signal tmp_10207[3] <== [tmp_10126[0] + tmp_10206[0], tmp_10126[1] + tmp_10206[1], tmp_10126[2] + tmp_10206[2]];
    signal tmp_10208[3] <== [3268211924158099194 * tmp_10207[0], 3268211924158099194 * tmp_10207[1], 3268211924158099194 * tmp_10207[2]];
    signal tmp_10209[3] <== [tmp_10204[0] + tmp_10208[0], tmp_10204[1] + tmp_10208[1], tmp_10204[2] + tmp_10208[2]];
    signal tmp_10210[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10211[3] <== [tmp_10210[0] * 15945260208470094587, tmp_10210[1] * 15945260208470094587, tmp_10210[2] * 15945260208470094587];
    signal tmp_10212[3] <== [tmp_10131[0] + tmp_10211[0], tmp_10131[1] + tmp_10211[1], tmp_10131[2] + tmp_10211[2]];
    signal tmp_10213[3] <== [17850261561957541568 * tmp_10212[0], 17850261561957541568 * tmp_10212[1], 17850261561957541568 * tmp_10212[2]];
    signal tmp_10214[3] <== [tmp_10209[0] + tmp_10213[0], tmp_10209[1] + tmp_10213[1], tmp_10209[2] + tmp_10213[2]];
    signal tmp_10215[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10216[3] <== [tmp_10215[0] * 8246979520807589148, tmp_10215[1] * 8246979520807589148, tmp_10215[2] * 8246979520807589148];
    signal tmp_10217[3] <== [tmp_10136[0] + tmp_10216[0], tmp_10136[1] + tmp_10216[1], tmp_10136[2] + tmp_10216[2]];
    signal tmp_10218[3] <== [5233791213354052332 * tmp_10217[0], 5233791213354052332 * tmp_10217[1], 5233791213354052332 * tmp_10217[2]];
    signal tmp_10219[3] <== [tmp_10214[0] + tmp_10218[0], tmp_10214[1] + tmp_10218[1], tmp_10214[2] + tmp_10218[2]];
    signal tmp_10220[3] <== [tmp_7550[0] + 15492711980379895475, tmp_7550[1], tmp_7550[2]];
    signal tmp_10221[3] <== [tmp_10220[0] * 13778737560392521094, tmp_10220[1] * 13778737560392521094, tmp_10220[2] * 13778737560392521094];
    signal tmp_10222[3] <== [tmp_10141[0] + tmp_10221[0], tmp_10141[1] + tmp_10221[1], tmp_10141[2] + tmp_10221[2]];
    signal tmp_10223[3] <== [9957237719655515451 * tmp_10222[0], 9957237719655515451 * tmp_10222[1], 9957237719655515451 * tmp_10222[2]];
    signal tmp_10224[3] <== [tmp_10219[0] + tmp_10223[0], tmp_10219[1] + tmp_10223[1], tmp_10219[2] + tmp_10223[2]];
    signal tmp_10225[3] <== [tmp_10224[0] * 1, tmp_10224[1] * 1, tmp_10224[2] * 1];
    signal tmp_10226[3] <== [evals[95][0] - tmp_10225[0], evals[95][1] - tmp_10225[1], evals[95][2] - tmp_10225[2]];
    signal tmp_10227[3] <== CMul()(tmp_6068, tmp_10226);
    signal tmp_10228[3] <== [tmp_10148[0] + tmp_10227[0], tmp_10148[1] + tmp_10227[1], tmp_10148[2] + tmp_10227[2]];
    signal tmp_10229[3] <== CMul()(challengeQ, tmp_10228);
    signal tmp_10230[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10231[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10232[3] <== [tmp_10231[0] * 14943873981466403895, tmp_10231[1] * 14943873981466403895, tmp_10231[2] * 14943873981466403895];
    signal tmp_10233[3] <== [tmp_10152[0] + tmp_10232[0], tmp_10152[1] + tmp_10232[1], tmp_10152[2] + tmp_10232[2]];
    signal tmp_10234[3] <== [2284659241919280696 * tmp_10233[0], 2284659241919280696 * tmp_10233[1], 2284659241919280696 * tmp_10233[2]];
    signal tmp_10235[3] <== [tmp_10230[0] + tmp_10234[0], tmp_10230[1] + tmp_10234[1], tmp_10230[2] + tmp_10234[2]];
    signal tmp_10236[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10237[3] <== [tmp_10236[0] * 7858901551218927609, tmp_10236[1] * 7858901551218927609, tmp_10236[2] * 7858901551218927609];
    signal tmp_10238[3] <== [tmp_10157[0] + tmp_10237[0], tmp_10157[1] + tmp_10237[1], tmp_10157[2] + tmp_10237[2]];
    signal tmp_10239[3] <== [1847394796741791100 * tmp_10238[0], 1847394796741791100 * tmp_10238[1], 1847394796741791100 * tmp_10238[2]];
    signal tmp_10240[3] <== [tmp_10235[0] + tmp_10239[0], tmp_10235[1] + tmp_10239[1], tmp_10235[2] + tmp_10239[2]];
    signal tmp_10241[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10242[3] <== [tmp_10241[0] * 12183924058797585540, tmp_10241[1] * 12183924058797585540, tmp_10241[2] * 12183924058797585540];
    signal tmp_10243[3] <== [tmp_10162[0] + tmp_10242[0], tmp_10162[1] + tmp_10242[1], tmp_10162[2] + tmp_10242[2]];
    signal tmp_10244[3] <== [17083607968162884940 * tmp_10243[0], 17083607968162884940 * tmp_10243[1], 17083607968162884940 * tmp_10243[2]];
    signal tmp_10245[3] <== [tmp_10240[0] + tmp_10244[0], tmp_10240[1] + tmp_10244[1], tmp_10240[2] + tmp_10244[2]];
    signal tmp_10246[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10247[3] <== [tmp_10246[0] * 11287081087228760936, tmp_10246[1] * 11287081087228760936, tmp_10246[2] * 11287081087228760936];
    signal tmp_10248[3] <== [tmp_10167[0] + tmp_10247[0], tmp_10167[1] + tmp_10247[1], tmp_10167[2] + tmp_10247[2]];
    signal tmp_10249[3] <== [14649174299425920385 * tmp_10248[0], 14649174299425920385 * tmp_10248[1], 14649174299425920385 * tmp_10248[2]];
    signal tmp_10250[3] <== [tmp_10245[0] + tmp_10249[0], tmp_10245[1] + tmp_10249[1], tmp_10245[2] + tmp_10249[2]];
    signal tmp_10251[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10252[3] <== [tmp_10251[0] * 1768132666423206446, tmp_10251[1] * 1768132666423206446, tmp_10251[2] * 1768132666423206446];
    signal tmp_10253[3] <== [tmp_10172[0] + tmp_10252[0], tmp_10172[1] + tmp_10252[1], tmp_10172[2] + tmp_10252[2]];
    signal tmp_10254[3] <== [16742036463763035056 * tmp_10253[0], 16742036463763035056 * tmp_10253[1], 16742036463763035056 * tmp_10253[2]];
    signal tmp_10255[3] <== [tmp_10250[0] + tmp_10254[0], tmp_10250[1] + tmp_10254[1], tmp_10250[2] + tmp_10254[2]];
    signal tmp_10256[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10257[3] <== [tmp_10256[0] * 13818240357693031756, tmp_10256[1] * 13818240357693031756, tmp_10256[2] * 13818240357693031756];
    signal tmp_10258[3] <== [tmp_10177[0] + tmp_10257[0], tmp_10177[1] + tmp_10257[1], tmp_10177[2] + tmp_10257[2]];
    signal tmp_10259[3] <== [4411458753033679918 * tmp_10258[0], 4411458753033679918 * tmp_10258[1], 4411458753033679918 * tmp_10258[2]];
    signal tmp_10260[3] <== [tmp_10255[0] + tmp_10259[0], tmp_10255[1] + tmp_10259[1], tmp_10255[2] + tmp_10259[2]];
    signal tmp_10261[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10262[3] <== [tmp_10261[0] * 13951006846344947715, tmp_10261[1] * 13951006846344947715, tmp_10261[2] * 13951006846344947715];
    signal tmp_10263[3] <== [tmp_10182[0] + tmp_10262[0], tmp_10182[1] + tmp_10262[1], tmp_10182[2] + tmp_10262[2]];
    signal tmp_10264[3] <== [8754998449733673829 * tmp_10263[0], 8754998449733673829 * tmp_10263[1], 8754998449733673829 * tmp_10263[2]];
    signal tmp_10265[3] <== [tmp_10260[0] + tmp_10264[0], tmp_10260[1] + tmp_10264[1], tmp_10260[2] + tmp_10264[2]];
    signal tmp_10266[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10267[3] <== [tmp_10266[0] * 7593135614815777901, tmp_10266[1] * 7593135614815777901, tmp_10266[2] * 7593135614815777901];
    signal tmp_10268[3] <== [tmp_10187[0] + tmp_10267[0], tmp_10187[1] + tmp_10267[1], tmp_10187[2] + tmp_10267[2]];
    signal tmp_10269[3] <== [7535759880903270203 * tmp_10268[0], 7535759880903270203 * tmp_10268[1], 7535759880903270203 * tmp_10268[2]];
    signal tmp_10270[3] <== [tmp_10265[0] + tmp_10269[0], tmp_10265[1] + tmp_10269[1], tmp_10265[2] + tmp_10269[2]];
    signal tmp_10271[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10272[3] <== [tmp_10271[0] * 13292507633934451261, tmp_10271[1] * 13292507633934451261, tmp_10271[2] * 13292507633934451261];
    signal tmp_10273[3] <== [tmp_10192[0] + tmp_10272[0], tmp_10192[1] + tmp_10272[1], tmp_10192[2] + tmp_10272[2]];
    signal tmp_10274[3] <== [8448917368593535896 * tmp_10273[0], 8448917368593535896 * tmp_10273[1], 8448917368593535896 * tmp_10273[2]];
    signal tmp_10275[3] <== [tmp_10270[0] + tmp_10274[0], tmp_10270[1] + tmp_10274[1], tmp_10270[2] + tmp_10274[2]];
    signal tmp_10276[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10277[3] <== [tmp_10276[0] * 605032684208182921, tmp_10276[1] * 605032684208182921, tmp_10276[2] * 605032684208182921];
    signal tmp_10278[3] <== [tmp_10197[0] + tmp_10277[0], tmp_10197[1] + tmp_10277[1], tmp_10197[2] + tmp_10277[2]];
    signal tmp_10279[3] <== [16823524365863196542 * tmp_10278[0], 16823524365863196542 * tmp_10278[1], 16823524365863196542 * tmp_10278[2]];
    signal tmp_10280[3] <== [tmp_10275[0] + tmp_10279[0], tmp_10275[1] + tmp_10279[1], tmp_10275[2] + tmp_10279[2]];
    signal tmp_10281[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10282[3] <== [tmp_10281[0] * 6876471548335849320, tmp_10281[1] * 6876471548335849320, tmp_10281[2] * 6876471548335849320];
    signal tmp_10283[3] <== [tmp_10202[0] + tmp_10282[0], tmp_10202[1] + tmp_10282[1], tmp_10202[2] + tmp_10282[2]];
    signal tmp_10284[3] <== [15596667798790275845 * tmp_10283[0], 15596667798790275845 * tmp_10283[1], 15596667798790275845 * tmp_10283[2]];
    signal tmp_10285[3] <== [tmp_10280[0] + tmp_10284[0], tmp_10280[1] + tmp_10284[1], tmp_10280[2] + tmp_10284[2]];
    signal tmp_10286[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10287[3] <== [tmp_10286[0] * 14846539071992201527, tmp_10286[1] * 14846539071992201527, tmp_10286[2] * 14846539071992201527];
    signal tmp_10288[3] <== [tmp_10207[0] + tmp_10287[0], tmp_10207[1] + tmp_10287[1], tmp_10207[2] + tmp_10287[2]];
    signal tmp_10289[3] <== [1187691146794617430 * tmp_10288[0], 1187691146794617430 * tmp_10288[1], 1187691146794617430 * tmp_10288[2]];
    signal tmp_10290[3] <== [tmp_10285[0] + tmp_10289[0], tmp_10285[1] + tmp_10289[1], tmp_10285[2] + tmp_10289[2]];
    signal tmp_10291[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10292[3] <== [tmp_10291[0] * 9468985681611683708, tmp_10291[1] * 9468985681611683708, tmp_10291[2] * 9468985681611683708];
    signal tmp_10293[3] <== [tmp_10212[0] + tmp_10292[0], tmp_10212[1] + tmp_10292[1], tmp_10212[2] + tmp_10292[2]];
    signal tmp_10294[3] <== [11952521614018033253 * tmp_10293[0], 11952521614018033253 * tmp_10293[1], 11952521614018033253 * tmp_10293[2]];
    signal tmp_10295[3] <== [tmp_10290[0] + tmp_10294[0], tmp_10290[1] + tmp_10294[1], tmp_10290[2] + tmp_10294[2]];
    signal tmp_10296[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10297[3] <== [tmp_10296[0] * 16311784362809925742, tmp_10296[1] * 16311784362809925742, tmp_10296[2] * 16311784362809925742];
    signal tmp_10298[3] <== [tmp_10217[0] + tmp_10297[0], tmp_10217[1] + tmp_10297[1], tmp_10217[2] + tmp_10297[2]];
    signal tmp_10299[3] <== [9669076931479314330 * tmp_10298[0], 9669076931479314330 * tmp_10298[1], 9669076931479314330 * tmp_10298[2]];
    signal tmp_10300[3] <== [tmp_10295[0] + tmp_10299[0], tmp_10295[1] + tmp_10299[1], tmp_10295[2] + tmp_10299[2]];
    signal tmp_10301[3] <== [tmp_7562[0] + 1773355790966798977, tmp_7562[1], tmp_7562[2]];
    signal tmp_10302[3] <== [tmp_10301[0] * 3645094389087780366, tmp_10301[1] * 3645094389087780366, tmp_10301[2] * 3645094389087780366];
    signal tmp_10303[3] <== [tmp_10222[0] + tmp_10302[0], tmp_10222[1] + tmp_10302[1], tmp_10222[2] + tmp_10302[2]];
    signal tmp_10304[3] <== [12896624684325687180 * tmp_10303[0], 12896624684325687180 * tmp_10303[1], 12896624684325687180 * tmp_10303[2]];
    signal tmp_10305[3] <== [tmp_10300[0] + tmp_10304[0], tmp_10300[1] + tmp_10304[1], tmp_10300[2] + tmp_10304[2]];
    signal tmp_10306[3] <== [tmp_10305[0] * 1, tmp_10305[1] * 1, tmp_10305[2] * 1];
    signal tmp_10307[3] <== [evals[96][0] - tmp_10306[0], evals[96][1] - tmp_10306[1], evals[96][2] - tmp_10306[2]];
    signal tmp_10308[3] <== CMul()(tmp_6068, tmp_10307);
    signal tmp_10309[3] <== [tmp_10229[0] + tmp_10308[0], tmp_10229[1] + tmp_10308[1], tmp_10229[2] + tmp_10308[2]];
    signal tmp_10310[3] <== CMul()(challengeQ, tmp_10309);
    signal tmp_10311[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10312[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10313[3] <== [tmp_10312[0] * 4167332706053537795, tmp_10312[1] * 4167332706053537795, tmp_10312[2] * 4167332706053537795];
    signal tmp_10314[3] <== [tmp_10233[0] + tmp_10313[0], tmp_10233[1] + tmp_10313[1], tmp_10233[2] + tmp_10313[2]];
    signal tmp_10315[3] <== [6292265836956544172 * tmp_10314[0], 6292265836956544172 * tmp_10314[1], 6292265836956544172 * tmp_10314[2]];
    signal tmp_10316[3] <== [tmp_10311[0] + tmp_10315[0], tmp_10311[1] + tmp_10315[1], tmp_10311[2] + tmp_10315[2]];
    signal tmp_10317[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10318[3] <== [tmp_10317[0] * 1835961824923514021, tmp_10317[1] * 1835961824923514021, tmp_10317[2] * 1835961824923514021];
    signal tmp_10319[3] <== [tmp_10238[0] + tmp_10318[0], tmp_10238[1] + tmp_10318[1], tmp_10238[2] + tmp_10318[2]];
    signal tmp_10320[3] <== [15025127703762308473 * tmp_10319[0], 15025127703762308473 * tmp_10319[1], 15025127703762308473 * tmp_10319[2]];
    signal tmp_10321[3] <== [tmp_10316[0] + tmp_10320[0], tmp_10316[1] + tmp_10320[1], tmp_10316[2] + tmp_10320[2]];
    signal tmp_10322[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10323[3] <== [tmp_10322[0] * 2966705105152395108, tmp_10322[1] * 2966705105152395108, tmp_10322[2] * 2966705105152395108];
    signal tmp_10324[3] <== [tmp_10243[0] + tmp_10323[0], tmp_10243[1] + tmp_10323[1], tmp_10243[2] + tmp_10323[2]];
    signal tmp_10325[3] <== [1549382727242080154 * tmp_10324[0], 1549382727242080154 * tmp_10324[1], 1549382727242080154 * tmp_10324[2]];
    signal tmp_10326[3] <== [tmp_10321[0] + tmp_10325[0], tmp_10321[1] + tmp_10325[1], tmp_10321[2] + tmp_10325[2]];
    signal tmp_10327[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10328[3] <== [tmp_10327[0] * 973577938650894194, tmp_10327[1] * 973577938650894194, tmp_10327[2] * 973577938650894194];
    signal tmp_10329[3] <== [tmp_10248[0] + tmp_10328[0], tmp_10248[1] + tmp_10328[1], tmp_10248[2] + tmp_10328[2]];
    signal tmp_10330[3] <== [9995718145205770324 * tmp_10329[0], 9995718145205770324 * tmp_10329[1], 9995718145205770324 * tmp_10329[2]];
    signal tmp_10331[3] <== [tmp_10326[0] + tmp_10330[0], tmp_10326[1] + tmp_10330[1], tmp_10326[2] + tmp_10330[2]];
    signal tmp_10332[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10333[3] <== [tmp_10332[0] * 14560278670602761416, tmp_10332[1] * 14560278670602761416, tmp_10332[2] * 14560278670602761416];
    signal tmp_10334[3] <== [tmp_10253[0] + tmp_10333[0], tmp_10253[1] + tmp_10333[1], tmp_10253[2] + tmp_10333[2]];
    signal tmp_10335[3] <== [16328597422766744836 * tmp_10334[0], 16328597422766744836 * tmp_10334[1], 16328597422766744836 * tmp_10334[2]];
    signal tmp_10336[3] <== [tmp_10331[0] + tmp_10335[0], tmp_10331[1] + tmp_10335[1], tmp_10331[2] + tmp_10335[2]];
    signal tmp_10337[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10338[3] <== [tmp_10337[0] * 397212481041799485, tmp_10337[1] * 397212481041799485, tmp_10337[2] * 397212481041799485];
    signal tmp_10339[3] <== [tmp_10258[0] + tmp_10338[0], tmp_10258[1] + tmp_10338[1], tmp_10258[2] + tmp_10338[2]];
    signal tmp_10340[3] <== [14994994730468297602 * tmp_10339[0], 14994994730468297602 * tmp_10339[1], 14994994730468297602 * tmp_10339[2]];
    signal tmp_10341[3] <== [tmp_10336[0] + tmp_10340[0], tmp_10336[1] + tmp_10340[1], tmp_10336[2] + tmp_10340[2]];
    signal tmp_10342[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10343[3] <== [tmp_10342[0] * 4220306037658238426, tmp_10342[1] * 4220306037658238426, tmp_10342[2] * 4220306037658238426];
    signal tmp_10344[3] <== [tmp_10263[0] + tmp_10343[0], tmp_10263[1] + tmp_10343[1], tmp_10263[2] + tmp_10343[2]];
    signal tmp_10345[3] <== [5652613299116160344 * tmp_10344[0], 5652613299116160344 * tmp_10344[1], 5652613299116160344 * tmp_10344[2]];
    signal tmp_10346[3] <== [tmp_10341[0] + tmp_10345[0], tmp_10341[1] + tmp_10345[1], tmp_10341[2] + tmp_10345[2]];
    signal tmp_10347[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10348[3] <== [tmp_10347[0] * 270403227330701921, tmp_10347[1] * 270403227330701921, tmp_10347[2] * 270403227330701921];
    signal tmp_10349[3] <== [tmp_10268[0] + tmp_10348[0], tmp_10268[1] + tmp_10348[1], tmp_10268[2] + tmp_10348[2]];
    signal tmp_10350[3] <== [2558771485279565859 * tmp_10349[0], 2558771485279565859 * tmp_10349[1], 2558771485279565859 * tmp_10349[2]];
    signal tmp_10351[3] <== [tmp_10346[0] + tmp_10350[0], tmp_10346[1] + tmp_10350[1], tmp_10346[2] + tmp_10350[2]];
    signal tmp_10352[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10353[3] <== [tmp_10352[0] * 1308298291121009953, tmp_10352[1] * 1308298291121009953, tmp_10352[2] * 1308298291121009953];
    signal tmp_10354[3] <== [tmp_10273[0] + tmp_10353[0], tmp_10273[1] + tmp_10353[1], tmp_10273[2] + tmp_10353[2]];
    signal tmp_10355[3] <== [6410586043754003422 * tmp_10354[0], 6410586043754003422 * tmp_10354[1], 6410586043754003422 * tmp_10354[2]];
    signal tmp_10356[3] <== [tmp_10351[0] + tmp_10355[0], tmp_10351[1] + tmp_10355[1], tmp_10351[2] + tmp_10355[2]];
    signal tmp_10357[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10358[3] <== [tmp_10357[0] * 11943868125602043586, tmp_10357[1] * 11943868125602043586, tmp_10357[2] * 11943868125602043586];
    signal tmp_10359[3] <== [tmp_10278[0] + tmp_10358[0], tmp_10278[1] + tmp_10358[1], tmp_10278[2] + tmp_10358[2]];
    signal tmp_10360[3] <== [5825523164635395574 * tmp_10359[0], 5825523164635395574 * tmp_10359[1], 5825523164635395574 * tmp_10359[2]];
    signal tmp_10361[3] <== [tmp_10356[0] + tmp_10360[0], tmp_10356[1] + tmp_10360[1], tmp_10356[2] + tmp_10360[2]];
    signal tmp_10362[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10363[3] <== [tmp_10362[0] * 202395478849929974, tmp_10362[1] * 202395478849929974, tmp_10362[2] * 202395478849929974];
    signal tmp_10364[3] <== [tmp_10283[0] + tmp_10363[0], tmp_10283[1] + tmp_10363[1], tmp_10283[2] + tmp_10363[2]];
    signal tmp_10365[3] <== [2094054892975333961 * tmp_10364[0], 2094054892975333961 * tmp_10364[1], 2094054892975333961 * tmp_10364[2]];
    signal tmp_10366[3] <== [tmp_10361[0] + tmp_10365[0], tmp_10361[1] + tmp_10365[1], tmp_10361[2] + tmp_10365[2]];
    signal tmp_10367[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10368[3] <== [tmp_10367[0] * 17179636232347603881, tmp_10367[1] * 17179636232347603881, tmp_10367[2] * 17179636232347603881];
    signal tmp_10369[3] <== [tmp_10288[0] + tmp_10368[0], tmp_10288[1] + tmp_10368[1], tmp_10288[2] + tmp_10368[2]];
    signal tmp_10370[3] <== [4957600268439660491 * tmp_10369[0], 4957600268439660491 * tmp_10369[1], 4957600268439660491 * tmp_10369[2]];
    signal tmp_10371[3] <== [tmp_10366[0] + tmp_10370[0], tmp_10366[1] + tmp_10370[1], tmp_10366[2] + tmp_10370[2]];
    signal tmp_10372[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10373[3] <== [tmp_10372[0] * 4425027275485648465, tmp_10372[1] * 4425027275485648465, tmp_10372[2] * 4425027275485648465];
    signal tmp_10374[3] <== [tmp_10293[0] + tmp_10373[0], tmp_10293[1] + tmp_10373[1], tmp_10293[2] + tmp_10373[2]];
    signal tmp_10375[3] <== [12244729162271932904 * tmp_10374[0], 12244729162271932904 * tmp_10374[1], 12244729162271932904 * tmp_10374[2]];
    signal tmp_10376[3] <== [tmp_10371[0] + tmp_10375[0], tmp_10371[1] + tmp_10375[1], tmp_10371[2] + tmp_10375[2]];
    signal tmp_10377[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10378[3] <== [tmp_10377[0] * 9617076952929794796, tmp_10377[1] * 9617076952929794796, tmp_10377[2] * 9617076952929794796];
    signal tmp_10379[3] <== [tmp_10298[0] + tmp_10378[0], tmp_10298[1] + tmp_10378[1], tmp_10298[2] + tmp_10378[2]];
    signal tmp_10380[3] <== [722213697716749189 * tmp_10379[0], 722213697716749189 * tmp_10379[1], 722213697716749189 * tmp_10379[2]];
    signal tmp_10381[3] <== [tmp_10376[0] + tmp_10380[0], tmp_10376[1] + tmp_10380[1], tmp_10376[2] + tmp_10380[2]];
    signal tmp_10382[3] <== [tmp_7574[0] + 14396910818521180817, tmp_7574[1], tmp_7574[2]];
    signal tmp_10383[3] <== [tmp_10382[0] * 263806961605756236, tmp_10382[1] * 263806961605756236, tmp_10382[2] * 263806961605756236];
    signal tmp_10384[3] <== [tmp_10303[0] + tmp_10383[0], tmp_10303[1] + tmp_10383[1], tmp_10303[2] + tmp_10383[2]];
    signal tmp_10385[3] <== [9743689501105433264 * tmp_10384[0], 9743689501105433264 * tmp_10384[1], 9743689501105433264 * tmp_10384[2]];
    signal tmp_10386[3] <== [tmp_10381[0] + tmp_10385[0], tmp_10381[1] + tmp_10385[1], tmp_10381[2] + tmp_10385[2]];
    signal tmp_10387[3] <== [tmp_10386[0] * 1, tmp_10386[1] * 1, tmp_10386[2] * 1];
    signal tmp_10388[3] <== [evals[97][0] - tmp_10387[0], evals[97][1] - tmp_10387[1], evals[97][2] - tmp_10387[2]];
    signal tmp_10389[3] <== CMul()(tmp_6068, tmp_10388);
    signal tmp_10390[3] <== [tmp_10310[0] + tmp_10389[0], tmp_10310[1] + tmp_10389[1], tmp_10310[2] + tmp_10389[2]];
    signal tmp_10391[3] <== CMul()(challengeQ, tmp_10390);
    signal tmp_10392[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10393[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10394[3] <== [tmp_10393[0] * 16189423883189207660, tmp_10393[1] * 16189423883189207660, tmp_10393[2] * 16189423883189207660];
    signal tmp_10395[3] <== [tmp_10314[0] + tmp_10394[0], tmp_10314[1] + tmp_10394[1], tmp_10314[2] + tmp_10394[2]];
    signal tmp_10396[3] <== [10251643030888928632 * tmp_10395[0], 10251643030888928632 * tmp_10395[1], 10251643030888928632 * tmp_10395[2]];
    signal tmp_10397[3] <== [tmp_10392[0] + tmp_10396[0], tmp_10392[1] + tmp_10396[1], tmp_10392[2] + tmp_10396[2]];
    signal tmp_10398[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10399[3] <== [tmp_10398[0] * 14579641431978311466, tmp_10398[1] * 14579641431978311466, tmp_10398[2] * 14579641431978311466];
    signal tmp_10400[3] <== [tmp_10319[0] + tmp_10399[0], tmp_10319[1] + tmp_10399[1], tmp_10319[2] + tmp_10399[2]];
    signal tmp_10401[3] <== [14134563997332366361 * tmp_10400[0], 14134563997332366361 * tmp_10400[1], 14134563997332366361 * tmp_10400[2]];
    signal tmp_10402[3] <== [tmp_10397[0] + tmp_10401[0], tmp_10397[1] + tmp_10401[1], tmp_10397[2] + tmp_10401[2]];
    signal tmp_10403[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10404[3] <== [tmp_10403[0] * 14758776138231707159, tmp_10403[1] * 14758776138231707159, tmp_10403[2] * 14758776138231707159];
    signal tmp_10405[3] <== [tmp_10324[0] + tmp_10404[0], tmp_10324[1] + tmp_10404[1], tmp_10324[2] + tmp_10404[2]];
    signal tmp_10406[3] <== [14114253400931675624 * tmp_10405[0], 14114253400931675624 * tmp_10405[1], 14114253400931675624 * tmp_10405[2]];
    signal tmp_10407[3] <== [tmp_10402[0] + tmp_10406[0], tmp_10402[1] + tmp_10406[1], tmp_10402[2] + tmp_10406[2]];
    signal tmp_10408[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10409[3] <== [tmp_10408[0] * 12860387175592582496, tmp_10408[1] * 12860387175592582496, tmp_10408[2] * 12860387175592582496];
    signal tmp_10410[3] <== [tmp_10329[0] + tmp_10409[0], tmp_10329[1] + tmp_10409[1], tmp_10329[2] + tmp_10409[2]];
    signal tmp_10411[3] <== [3957736064180453745 * tmp_10410[0], 3957736064180453745 * tmp_10410[1], 3957736064180453745 * tmp_10410[2]];
    signal tmp_10412[3] <== [tmp_10407[0] + tmp_10411[0], tmp_10407[1] + tmp_10411[1], tmp_10407[2] + tmp_10411[2]];
    signal tmp_10413[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10414[3] <== [tmp_10413[0] * 15858261595013926463, tmp_10413[1] * 15858261595013926463, tmp_10413[2] * 15858261595013926463];
    signal tmp_10415[3] <== [tmp_10334[0] + tmp_10414[0], tmp_10334[1] + tmp_10414[1], tmp_10334[2] + tmp_10414[2]];
    signal tmp_10416[3] <== [8095276956282841616 * tmp_10415[0], 8095276956282841616 * tmp_10415[1], 8095276956282841616 * tmp_10415[2]];
    signal tmp_10417[3] <== [tmp_10412[0] + tmp_10416[0], tmp_10412[1] + tmp_10416[1], tmp_10412[2] + tmp_10416[2]];
    signal tmp_10418[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10419[3] <== [tmp_10418[0] * 14181919498407542627, tmp_10418[1] * 14181919498407542627, tmp_10418[2] * 14181919498407542627];
    signal tmp_10420[3] <== [tmp_10339[0] + tmp_10419[0], tmp_10339[1] + tmp_10419[1], tmp_10339[2] + tmp_10419[2]];
    signal tmp_10421[3] <== [4831909714697294330 * tmp_10420[0], 4831909714697294330 * tmp_10420[1], 4831909714697294330 * tmp_10420[2]];
    signal tmp_10422[3] <== [tmp_10417[0] + tmp_10421[0], tmp_10417[1] + tmp_10421[1], tmp_10417[2] + tmp_10421[2]];
    signal tmp_10423[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10424[3] <== [tmp_10423[0] * 13242793143982098963, tmp_10423[1] * 13242793143982098963, tmp_10423[2] * 13242793143982098963];
    signal tmp_10425[3] <== [tmp_10344[0] + tmp_10424[0], tmp_10344[1] + tmp_10424[1], tmp_10344[2] + tmp_10424[2]];
    signal tmp_10426[3] <== [1013946114800042668 * tmp_10425[0], 1013946114800042668 * tmp_10425[1], 1013946114800042668 * tmp_10425[2]];
    signal tmp_10427[3] <== [tmp_10422[0] + tmp_10426[0], tmp_10422[1] + tmp_10426[1], tmp_10422[2] + tmp_10426[2]];
    signal tmp_10428[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10429[3] <== [tmp_10428[0] * 11497321389989836778, tmp_10428[1] * 11497321389989836778, tmp_10428[2] * 11497321389989836778];
    signal tmp_10430[3] <== [tmp_10349[0] + tmp_10429[0], tmp_10349[1] + tmp_10429[1], tmp_10349[2] + tmp_10429[2]];
    signal tmp_10431[3] <== [18251392153846354131 * tmp_10430[0], 18251392153846354131 * tmp_10430[1], 18251392153846354131 * tmp_10430[2]];
    signal tmp_10432[3] <== [tmp_10427[0] + tmp_10431[0], tmp_10427[1] + tmp_10431[1], tmp_10427[2] + tmp_10431[2]];
    signal tmp_10433[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10434[3] <== [tmp_10433[0] * 16137066962847815728, tmp_10433[1] * 16137066962847815728, tmp_10433[2] * 16137066962847815728];
    signal tmp_10435[3] <== [tmp_10354[0] + tmp_10434[0], tmp_10354[1] + tmp_10434[1], tmp_10354[2] + tmp_10434[2]];
    signal tmp_10436[3] <== [2225048953983193835 * tmp_10435[0], 2225048953983193835 * tmp_10435[1], 2225048953983193835 * tmp_10435[2]];
    signal tmp_10437[3] <== [tmp_10432[0] + tmp_10436[0], tmp_10432[1] + tmp_10436[1], tmp_10432[2] + tmp_10436[2]];
    signal tmp_10438[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10439[3] <== [tmp_10438[0] * 15056884893890740908, tmp_10438[1] * 15056884893890740908, tmp_10438[2] * 15056884893890740908];
    signal tmp_10440[3] <== [tmp_10359[0] + tmp_10439[0], tmp_10359[1] + tmp_10439[1], tmp_10359[2] + tmp_10439[2]];
    signal tmp_10441[3] <== [1367030755796680700 * tmp_10440[0], 1367030755796680700 * tmp_10440[1], 1367030755796680700 * tmp_10440[2]];
    signal tmp_10442[3] <== [tmp_10437[0] + tmp_10441[0], tmp_10437[1] + tmp_10441[1], tmp_10437[2] + tmp_10441[2]];
    signal tmp_10443[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10444[3] <== [tmp_10443[0] * 14449506859106810187, tmp_10443[1] * 14449506859106810187, tmp_10443[2] * 14449506859106810187];
    signal tmp_10445[3] <== [tmp_10364[0] + tmp_10444[0], tmp_10364[1] + tmp_10444[1], tmp_10364[2] + tmp_10444[2]];
    signal tmp_10446[3] <== [7271551381474385426 * tmp_10445[0], 7271551381474385426 * tmp_10445[1], 7271551381474385426 * tmp_10445[2]];
    signal tmp_10447[3] <== [tmp_10442[0] + tmp_10446[0], tmp_10442[1] + tmp_10446[1], tmp_10442[2] + tmp_10446[2]];
    signal tmp_10448[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10449[3] <== [tmp_10448[0] * 14294938476450530176, tmp_10448[1] * 14294938476450530176, tmp_10448[2] * 14294938476450530176];
    signal tmp_10450[3] <== [tmp_10369[0] + tmp_10449[0], tmp_10369[1] + tmp_10449[1], tmp_10369[2] + tmp_10449[2]];
    signal tmp_10451[3] <== [2265568358314072054 * tmp_10450[0], 2265568358314072054 * tmp_10450[1], 2265568358314072054 * tmp_10450[2]];
    signal tmp_10452[3] <== [tmp_10447[0] + tmp_10451[0], tmp_10447[1] + tmp_10451[1], tmp_10447[2] + tmp_10451[2]];
    signal tmp_10453[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10454[3] <== [tmp_10453[0] * 16127203105503375405, tmp_10453[1] * 16127203105503375405, tmp_10453[2] * 16127203105503375405];
    signal tmp_10455[3] <== [tmp_10374[0] + tmp_10454[0], tmp_10374[1] + tmp_10454[1], tmp_10374[2] + tmp_10454[2]];
    signal tmp_10456[3] <== [16819938243526879572 * tmp_10455[0], 16819938243526879572 * tmp_10455[1], 16819938243526879572 * tmp_10455[2]];
    signal tmp_10457[3] <== [tmp_10452[0] + tmp_10456[0], tmp_10452[1] + tmp_10456[1], tmp_10452[2] + tmp_10456[2]];
    signal tmp_10458[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10459[3] <== [tmp_10458[0] * 12902790681243023167, tmp_10458[1] * 12902790681243023167, tmp_10458[2] * 12902790681243023167];
    signal tmp_10460[3] <== [tmp_10379[0] + tmp_10459[0], tmp_10379[1] + tmp_10459[1], tmp_10379[2] + tmp_10459[2]];
    signal tmp_10461[3] <== [5766489137465308436 * tmp_10460[0], 5766489137465308436 * tmp_10460[1], 5766489137465308436 * tmp_10460[2]];
    signal tmp_10462[3] <== [tmp_10457[0] + tmp_10461[0], tmp_10457[1] + tmp_10461[1], tmp_10457[2] + tmp_10461[2]];
    signal tmp_10463[3] <== [tmp_7586[0] + 3603452348219013210, tmp_7586[1], tmp_7586[2]];
    signal tmp_10464[3] <== [tmp_10463[0] * 14802832860197476755, tmp_10463[1] * 14802832860197476755, tmp_10463[2] * 14802832860197476755];
    signal tmp_10465[3] <== [tmp_10384[0] + tmp_10464[0], tmp_10384[1] + tmp_10464[1], tmp_10384[2] + tmp_10464[2]];
    signal tmp_10466[3] <== [11873533365200282612 * tmp_10465[0], 11873533365200282612 * tmp_10465[1], 11873533365200282612 * tmp_10465[2]];
    signal tmp_10467[3] <== [tmp_10462[0] + tmp_10466[0], tmp_10462[1] + tmp_10466[1], tmp_10462[2] + tmp_10466[2]];
    signal tmp_10468[3] <== [tmp_10467[0] * 1, tmp_10467[1] * 1, tmp_10467[2] * 1];
    signal tmp_10469[3] <== [evals[59][0] - tmp_10468[0], evals[59][1] - tmp_10468[1], evals[59][2] - tmp_10468[2]];
    signal tmp_10470[3] <== CMul()(tmp_6068, tmp_10469);
    signal tmp_10471[3] <== [tmp_10391[0] + tmp_10470[0], tmp_10391[1] + tmp_10470[1], tmp_10391[2] + tmp_10470[2]];
    signal tmp_10472[3] <== CMul()(challengeQ, tmp_10471);
    signal tmp_10473[3] <== CMul()(evals[59], evals[59]);
    signal tmp_10474[3] <== CMul()(tmp_10473, tmp_10473);
    signal tmp_10475[3] <== CMul()(tmp_10474, tmp_10473);
    signal tmp_10476[3] <== CMul()(tmp_10475, evals[59]);
    signal tmp_10477[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10478[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10479[3] <== [tmp_10478[0] * 46774797835064597, tmp_10478[1] * 46774797835064597, tmp_10478[2] * 46774797835064597];
    signal tmp_10480[3] <== [tmp_10395[0] + tmp_10479[0], tmp_10395[1] + tmp_10479[1], tmp_10395[2] + tmp_10479[2]];
    signal tmp_10481[3] <== [16742843303846205239 * tmp_10480[0], 16742843303846205239 * tmp_10480[1], 16742843303846205239 * tmp_10480[2]];
    signal tmp_10482[3] <== [tmp_10477[0] + tmp_10481[0], tmp_10477[1] + tmp_10481[1], tmp_10477[2] + tmp_10481[2]];
    signal tmp_10483[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10484[3] <== [tmp_10483[0] * 42260813756980111, tmp_10483[1] * 42260813756980111, tmp_10483[2] * 42260813756980111];
    signal tmp_10485[3] <== [tmp_10400[0] + tmp_10484[0], tmp_10400[1] + tmp_10484[1], tmp_10400[2] + tmp_10484[2]];
    signal tmp_10486[3] <== [16027339315972474841 * tmp_10485[0], 16027339315972474841 * tmp_10485[1], 16027339315972474841 * tmp_10485[2]];
    signal tmp_10487[3] <== [tmp_10482[0] + tmp_10486[0], tmp_10482[1] + tmp_10486[1], tmp_10482[2] + tmp_10486[2]];
    signal tmp_10488[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10489[3] <== [tmp_10488[0] * 42061006075881976, tmp_10488[1] * 42061006075881976, tmp_10488[2] * 42061006075881976];
    signal tmp_10490[3] <== [tmp_10405[0] + tmp_10489[0], tmp_10405[1] + tmp_10489[1], tmp_10405[2] + tmp_10489[2]];
    signal tmp_10491[3] <== [5155237477011543063 * tmp_10490[0], 5155237477011543063 * tmp_10490[1], 5155237477011543063 * tmp_10490[2]];
    signal tmp_10492[3] <== [tmp_10487[0] + tmp_10491[0], tmp_10487[1] + tmp_10491[1], tmp_10487[2] + tmp_10491[2]];
    signal tmp_10493[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10494[3] <== [tmp_10493[0] * 37387884336802444, tmp_10493[1] * 37387884336802444, tmp_10493[2] * 37387884336802444];
    signal tmp_10495[3] <== [tmp_10410[0] + tmp_10494[0], tmp_10410[1] + tmp_10494[1], tmp_10410[2] + tmp_10494[2]];
    signal tmp_10496[3] <== [16706543232980973010 * tmp_10495[0], 16706543232980973010 * tmp_10495[1], 16706543232980973010 * tmp_10495[2]];
    signal tmp_10497[3] <== [tmp_10492[0] + tmp_10496[0], tmp_10492[1] + tmp_10496[1], tmp_10492[2] + tmp_10496[2]];
    signal tmp_10498[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10499[3] <== [tmp_10498[0] * 45874323791659250, tmp_10498[1] * 45874323791659250, tmp_10498[2] * 45874323791659250];
    signal tmp_10500[3] <== [tmp_10415[0] + tmp_10499[0], tmp_10415[1] + tmp_10499[1], tmp_10415[2] + tmp_10499[2]];
    signal tmp_10501[3] <== [7286314172953842238 * tmp_10500[0], 7286314172953842238 * tmp_10500[1], 7286314172953842238 * tmp_10500[2]];
    signal tmp_10502[3] <== [tmp_10497[0] + tmp_10501[0], tmp_10497[1] + tmp_10501[1], tmp_10497[2] + tmp_10501[2]];
    signal tmp_10503[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10504[3] <== [tmp_10503[0] * 39843739646961688, tmp_10503[1] * 39843739646961688, tmp_10503[2] * 39843739646961688];
    signal tmp_10505[3] <== [tmp_10420[0] + tmp_10504[0], tmp_10420[1] + tmp_10504[1], tmp_10420[2] + tmp_10504[2]];
    signal tmp_10506[3] <== [4083460614057513700 * tmp_10505[0], 4083460614057513700 * tmp_10505[1], 4083460614057513700 * tmp_10505[2]];
    signal tmp_10507[3] <== [tmp_10502[0] + tmp_10506[0], tmp_10502[1] + tmp_10506[1], tmp_10502[2] + tmp_10506[2]];
    signal tmp_10508[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10509[3] <== [tmp_10508[0] * 37505668333798549, tmp_10508[1] * 37505668333798549, tmp_10508[2] * 37505668333798549];
    signal tmp_10510[3] <== [tmp_10425[0] + tmp_10509[0], tmp_10425[1] + tmp_10509[1], tmp_10425[2] + tmp_10509[2]];
    signal tmp_10511[3] <== [12377749838518084086 * tmp_10510[0], 12377749838518084086 * tmp_10510[1], 12377749838518084086 * tmp_10510[2]];
    signal tmp_10512[3] <== [tmp_10507[0] + tmp_10511[0], tmp_10507[1] + tmp_10511[1], tmp_10507[2] + tmp_10511[2]];
    signal tmp_10513[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10514[3] <== [tmp_10513[0] * 33555193038270055, tmp_10513[1] * 33555193038270055, tmp_10513[2] * 33555193038270055];
    signal tmp_10515[3] <== [tmp_10430[0] + tmp_10514[0], tmp_10430[1] + tmp_10514[1], tmp_10430[2] + tmp_10514[2]];
    signal tmp_10516[3] <== [10835584132090636100 * tmp_10515[0], 10835584132090636100 * tmp_10515[1], 10835584132090636100 * tmp_10515[2]];
    signal tmp_10517[3] <== [tmp_10512[0] + tmp_10516[0], tmp_10512[1] + tmp_10516[1], tmp_10512[2] + tmp_10516[2]];
    signal tmp_10518[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10519[3] <== [tmp_10518[0] * 46401355029888991, tmp_10518[1] * 46401355029888991, tmp_10518[2] * 46401355029888991];
    signal tmp_10520[3] <== [tmp_10435[0] + tmp_10519[0], tmp_10435[1] + tmp_10519[1], tmp_10435[2] + tmp_10519[2]];
    signal tmp_10521[3] <== [9162173973000276298 * tmp_10520[0], 9162173973000276298 * tmp_10520[1], 9162173973000276298 * tmp_10520[2]];
    signal tmp_10522[3] <== [tmp_10517[0] + tmp_10521[0], tmp_10517[1] + tmp_10521[1], tmp_10517[2] + tmp_10521[2]];
    signal tmp_10523[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10524[3] <== [tmp_10523[0] * 43291015365600457, tmp_10523[1] * 43291015365600457, tmp_10523[2] * 43291015365600457];
    signal tmp_10525[3] <== [tmp_10440[0] + tmp_10524[0], tmp_10440[1] + tmp_10524[1], tmp_10440[2] + tmp_10524[2]];
    signal tmp_10526[3] <== [13242512893966320078 * tmp_10525[0], 13242512893966320078 * tmp_10525[1], 13242512893966320078 * tmp_10525[2]];
    signal tmp_10527[3] <== [tmp_10522[0] + tmp_10526[0], tmp_10522[1] + tmp_10526[1], tmp_10522[2] + tmp_10526[2]];
    signal tmp_10528[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10529[3] <== [tmp_10528[0] * 41779174606149599, tmp_10528[1] * 41779174606149599, tmp_10528[2] * 41779174606149599];
    signal tmp_10530[3] <== [tmp_10445[0] + tmp_10529[0], tmp_10445[1] + tmp_10529[1], tmp_10445[2] + tmp_10529[2]];
    signal tmp_10531[3] <== [16079292348832412129 * tmp_10530[0], 16079292348832412129 * tmp_10530[1], 16079292348832412129 * tmp_10530[2]];
    signal tmp_10532[3] <== [tmp_10527[0] + tmp_10531[0], tmp_10527[1] + tmp_10531[1], tmp_10527[2] + tmp_10531[2]];
    signal tmp_10533[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10534[3] <== [tmp_10533[0] * 40954303879913243, tmp_10533[1] * 40954303879913243, tmp_10533[2] * 40954303879913243];
    signal tmp_10535[3] <== [tmp_10450[0] + tmp_10534[0], tmp_10450[1] + tmp_10534[1], tmp_10450[2] + tmp_10534[2]];
    signal tmp_10536[3] <== [11725903253960910652 * tmp_10535[0], 11725903253960910652 * tmp_10535[1], 11725903253960910652 * tmp_10535[2]];
    signal tmp_10537[3] <== [tmp_10532[0] + tmp_10536[0], tmp_10532[1] + tmp_10536[1], tmp_10532[2] + tmp_10536[2]];
    signal tmp_10538[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10539[3] <== [tmp_10538[0] * 46379175729155491, tmp_10538[1] * 46379175729155491, tmp_10538[2] * 46379175729155491];
    signal tmp_10540[3] <== [tmp_10455[0] + tmp_10539[0], tmp_10455[1] + tmp_10539[1], tmp_10455[2] + tmp_10539[2]];
    signal tmp_10541[3] <== [2230603834375356574 * tmp_10540[0], 2230603834375356574 * tmp_10540[1], 2230603834375356574 * tmp_10540[2]];
    signal tmp_10542[3] <== [tmp_10537[0] + tmp_10541[0], tmp_10537[1] + tmp_10541[1], tmp_10537[2] + tmp_10541[2]];
    signal tmp_10543[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10544[3] <== [tmp_10543[0] * 38174104222724431, tmp_10543[1] * 38174104222724431, tmp_10543[2] * 38174104222724431];
    signal tmp_10545[3] <== [tmp_10460[0] + tmp_10544[0], tmp_10460[1] + tmp_10544[1], tmp_10460[2] + tmp_10544[2]];
    signal tmp_10546[3] <== [2370340271898460524 * tmp_10545[0], 2370340271898460524 * tmp_10545[1], 2370340271898460524 * tmp_10545[2]];
    signal tmp_10547[3] <== [tmp_10542[0] + tmp_10546[0], tmp_10542[1] + tmp_10546[1], tmp_10542[2] + tmp_10546[2]];
    signal tmp_10548[3] <== [tmp_7598[0] + 6021258740816487473, tmp_7598[1], tmp_7598[2]];
    signal tmp_10549[3] <== [tmp_10548[0] * 42896868640066877, tmp_10548[1] * 42896868640066877, tmp_10548[2] * 42896868640066877];
    signal tmp_10550[3] <== [tmp_10465[0] + tmp_10549[0], tmp_10465[1] + tmp_10549[1], tmp_10465[2] + tmp_10549[2]];
    signal tmp_10551[3] <== [15115197718548534897 * tmp_10550[0], 15115197718548534897 * tmp_10550[1], 15115197718548534897 * tmp_10550[2]];
    signal tmp_10552[3] <== [tmp_10547[0] + tmp_10551[0], tmp_10547[1] + tmp_10551[1], tmp_10547[2] + tmp_10551[2]];
    signal tmp_10553[3] <== [tmp_10552[0] * 1, tmp_10552[1] * 1, tmp_10552[2] * 1];
    signal tmp_10554[3] <== [evals[60][0] - tmp_10553[0], evals[60][1] - tmp_10553[1], evals[60][2] - tmp_10553[2]];
    signal tmp_10555[3] <== CMul()(tmp_6068, tmp_10554);
    signal tmp_10556[3] <== [tmp_10472[0] + tmp_10555[0], tmp_10472[1] + tmp_10555[1], tmp_10472[2] + tmp_10555[2]];
    signal tmp_10557[3] <== CMul()(challengeQ, tmp_10556);
    signal tmp_10558[3] <== CMul()(evals[60], evals[60]);
    signal tmp_10559[3] <== CMul()(tmp_10558, tmp_10558);
    signal tmp_10560[3] <== CMul()(tmp_10559, tmp_10558);
    signal tmp_10561[3] <== CMul()(tmp_10560, evals[60]);
    signal tmp_10562[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10563[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10564[3] <== [tmp_10563[0] * 136526419311991, tmp_10563[1] * 136526419311991, tmp_10563[2] * 136526419311991];
    signal tmp_10565[3] <== [tmp_10480[0] + tmp_10564[0], tmp_10480[1] + tmp_10564[1], tmp_10480[2] + tmp_10564[2]];
    signal tmp_10566[3] <== [13471986203177128393 * tmp_10565[0], 13471986203177128393 * tmp_10565[1], 13471986203177128393 * tmp_10565[2]];
    signal tmp_10567[3] <== [tmp_10562[0] + tmp_10566[0], tmp_10562[1] + tmp_10566[1], tmp_10562[2] + tmp_10566[2]];
    signal tmp_10568[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10569[3] <== [tmp_10568[0] * 118328038511571, tmp_10568[1] * 118328038511571, tmp_10568[2] * 118328038511571];
    signal tmp_10570[3] <== [tmp_10485[0] + tmp_10569[0], tmp_10485[1] + tmp_10569[1], tmp_10485[2] + tmp_10569[2]];
    signal tmp_10571[3] <== [5366496057517922993 * tmp_10570[0], 5366496057517922993 * tmp_10570[1], 5366496057517922993 * tmp_10570[2]];
    signal tmp_10572[3] <== [tmp_10567[0] + tmp_10571[0], tmp_10567[1] + tmp_10571[1], tmp_10567[2] + tmp_10571[2]];
    signal tmp_10573[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10574[3] <== [tmp_10573[0] * 119953340504678, tmp_10573[1] * 119953340504678, tmp_10573[2] * 119953340504678];
    signal tmp_10575[3] <== [tmp_10490[0] + tmp_10574[0], tmp_10490[1] + tmp_10574[1], tmp_10490[2] + tmp_10574[2]];
    signal tmp_10576[3] <== [1488645808239492428 * tmp_10575[0], 1488645808239492428 * tmp_10575[1], 1488645808239492428 * tmp_10575[2]];
    signal tmp_10577[3] <== [tmp_10572[0] + tmp_10576[0], tmp_10572[1] + tmp_10576[1], tmp_10572[2] + tmp_10576[2]];
    signal tmp_10578[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10579[3] <== [tmp_10578[0] * 108745180907795, tmp_10578[1] * 108745180907795, tmp_10578[2] * 108745180907795];
    signal tmp_10580[3] <== [tmp_10495[0] + tmp_10579[0], tmp_10495[1] + tmp_10579[1], tmp_10495[2] + tmp_10579[2]];
    signal tmp_10581[3] <== [14201553218769780296 * tmp_10580[0], 14201553218769780296 * tmp_10580[1], 14201553218769780296 * tmp_10580[2]];
    signal tmp_10582[3] <== [tmp_10577[0] + tmp_10581[0], tmp_10577[1] + tmp_10581[1], tmp_10577[2] + tmp_10581[2]];
    signal tmp_10583[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10584[3] <== [tmp_10583[0] * 130471040155374, tmp_10583[1] * 130471040155374, tmp_10583[2] * 130471040155374];
    signal tmp_10585[3] <== [tmp_10500[0] + tmp_10584[0], tmp_10500[1] + tmp_10584[1], tmp_10500[2] + tmp_10584[2]];
    signal tmp_10586[3] <== [9804649820399473624 * tmp_10585[0], 9804649820399473624 * tmp_10585[1], 9804649820399473624 * tmp_10585[2]];
    signal tmp_10587[3] <== [tmp_10582[0] + tmp_10586[0], tmp_10582[1] + tmp_10586[1], tmp_10582[2] + tmp_10586[2]];
    signal tmp_10588[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10589[3] <== [tmp_10588[0] * 120388179181929, tmp_10588[1] * 120388179181929, tmp_10588[2] * 120388179181929];
    signal tmp_10590[3] <== [tmp_10505[0] + tmp_10589[0], tmp_10505[1] + tmp_10589[1], tmp_10505[2] + tmp_10589[2]];
    signal tmp_10591[3] <== [8107836565913188653 * tmp_10590[0], 8107836565913188653 * tmp_10590[1], 8107836565913188653 * tmp_10590[2]];
    signal tmp_10592[3] <== [tmp_10587[0] + tmp_10591[0], tmp_10587[1] + tmp_10591[1], tmp_10587[2] + tmp_10591[2]];
    signal tmp_10593[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10594[3] <== [tmp_10593[0] * 111878367465032, tmp_10593[1] * 111878367465032, tmp_10593[2] * 111878367465032];
    signal tmp_10595[3] <== [tmp_10510[0] + tmp_10594[0], tmp_10510[1] + tmp_10594[1], tmp_10510[2] + tmp_10594[2]];
    signal tmp_10596[3] <== [17624536480452545801 * tmp_10595[0], 17624536480452545801 * tmp_10595[1], 17624536480452545801 * tmp_10595[2]];
    signal tmp_10597[3] <== [tmp_10592[0] + tmp_10596[0], tmp_10592[1] + tmp_10596[1], tmp_10592[2] + tmp_10596[2]];
    signal tmp_10598[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10599[3] <== [tmp_10598[0] * 93831285341340, tmp_10598[1] * 93831285341340, tmp_10598[2] * 93831285341340];
    signal tmp_10600[3] <== [tmp_10515[0] + tmp_10599[0], tmp_10515[1] + tmp_10599[1], tmp_10515[2] + tmp_10599[2]];
    signal tmp_10601[3] <== [12565506485641087099 * tmp_10600[0], 12565506485641087099 * tmp_10600[1], 12565506485641087099 * tmp_10600[2]];
    signal tmp_10602[3] <== [tmp_10597[0] + tmp_10601[0], tmp_10597[1] + tmp_10601[1], tmp_10597[2] + tmp_10601[2]];
    signal tmp_10603[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10604[3] <== [tmp_10603[0] * 132190456129009, tmp_10603[1] * 132190456129009, tmp_10603[2] * 132190456129009];
    signal tmp_10605[3] <== [tmp_10520[0] + tmp_10604[0], tmp_10520[1] + tmp_10604[1], tmp_10520[2] + tmp_10604[2]];
    signal tmp_10606[3] <== [8957615728977470678 * tmp_10605[0], 8957615728977470678 * tmp_10605[1], 8957615728977470678 * tmp_10605[2]];
    signal tmp_10607[3] <== [tmp_10602[0] + tmp_10606[0], tmp_10602[1] + tmp_10606[1], tmp_10602[2] + tmp_10606[2]];
    signal tmp_10608[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10609[3] <== [tmp_10608[0] * 129381316454358, tmp_10608[1] * 129381316454358, tmp_10608[2] * 129381316454358];
    signal tmp_10610[3] <== [tmp_10525[0] + tmp_10609[0], tmp_10525[1] + tmp_10609[1], tmp_10525[2] + tmp_10609[2]];
    signal tmp_10611[3] <== [14839456847523021051 * tmp_10610[0], 14839456847523021051 * tmp_10610[1], 14839456847523021051 * tmp_10610[2]];
    signal tmp_10612[3] <== [tmp_10607[0] + tmp_10611[0], tmp_10607[1] + tmp_10611[1], tmp_10607[2] + tmp_10611[2]];
    signal tmp_10613[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10614[3] <== [tmp_10613[0] * 119515447190824, tmp_10613[1] * 119515447190824, tmp_10613[2] * 119515447190824];
    signal tmp_10615[3] <== [tmp_10530[0] + tmp_10614[0], tmp_10530[1] + tmp_10614[1], tmp_10530[2] + tmp_10614[2]];
    signal tmp_10616[3] <== [2411490870786259255 * tmp_10615[0], 2411490870786259255 * tmp_10615[1], 2411490870786259255 * tmp_10615[2]];
    signal tmp_10617[3] <== [tmp_10612[0] + tmp_10616[0], tmp_10612[1] + tmp_10616[1], tmp_10612[2] + tmp_10616[2]];
    signal tmp_10618[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10619[3] <== [tmp_10618[0] * 118467363703132, tmp_10618[1] * 118467363703132, tmp_10618[2] * 118467363703132];
    signal tmp_10620[3] <== [tmp_10535[0] + tmp_10619[0], tmp_10535[1] + tmp_10619[1], tmp_10535[2] + tmp_10619[2]];
    signal tmp_10621[3] <== [5683566100576615076 * tmp_10620[0], 5683566100576615076 * tmp_10620[1], 5683566100576615076 * tmp_10620[2]];
    signal tmp_10622[3] <== [tmp_10617[0] + tmp_10621[0], tmp_10617[1] + tmp_10621[1], tmp_10617[2] + tmp_10621[2]];
    signal tmp_10623[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10624[3] <== [tmp_10623[0] * 135504422706996, tmp_10623[1] * 135504422706996, tmp_10623[2] * 135504422706996];
    signal tmp_10625[3] <== [tmp_10540[0] + tmp_10624[0], tmp_10540[1] + tmp_10624[1], tmp_10540[2] + tmp_10624[2]];
    signal tmp_10626[3] <== [12384678008916204197 * tmp_10625[0], 12384678008916204197 * tmp_10625[1], 12384678008916204197 * tmp_10625[2]];
    signal tmp_10627[3] <== [tmp_10622[0] + tmp_10626[0], tmp_10622[1] + tmp_10626[1], tmp_10622[2] + tmp_10626[2]];
    signal tmp_10628[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10629[3] <== [tmp_10628[0] * 106598700061332, tmp_10628[1] * 106598700061332, tmp_10628[2] * 106598700061332];
    signal tmp_10630[3] <== [tmp_10545[0] + tmp_10629[0], tmp_10545[1] + tmp_10629[1], tmp_10545[2] + tmp_10629[2]];
    signal tmp_10631[3] <== [13849032862488033002 * tmp_10630[0], 13849032862488033002 * tmp_10630[1], 13849032862488033002 * tmp_10630[2]];
    signal tmp_10632[3] <== [tmp_10627[0] + tmp_10631[0], tmp_10627[1] + tmp_10631[1], tmp_10627[2] + tmp_10631[2]];
    signal tmp_10633[3] <== [tmp_10476[0] + 13644528514839338320, tmp_10476[1], tmp_10476[2]];
    signal tmp_10634[3] <== [tmp_10633[0] * 118332012221556, tmp_10633[1] * 118332012221556, tmp_10633[2] * 118332012221556];
    signal tmp_10635[3] <== [tmp_10550[0] + tmp_10634[0], tmp_10550[1] + tmp_10634[1], tmp_10550[2] + tmp_10634[2]];
    signal tmp_10636[3] <== [3958502407134855424 * tmp_10635[0], 3958502407134855424 * tmp_10635[1], 3958502407134855424 * tmp_10635[2]];
    signal tmp_10637[3] <== [tmp_10632[0] + tmp_10636[0], tmp_10632[1] + tmp_10636[1], tmp_10632[2] + tmp_10636[2]];
    signal tmp_10638[3] <== [tmp_10637[0] * 1, tmp_10637[1] * 1, tmp_10637[2] * 1];
    signal tmp_10639[3] <== [evals[61][0] - tmp_10638[0], evals[61][1] - tmp_10638[1], evals[61][2] - tmp_10638[2]];
    signal tmp_10640[3] <== CMul()(tmp_6068, tmp_10639);
    signal tmp_10641[3] <== [tmp_10557[0] + tmp_10640[0], tmp_10557[1] + tmp_10640[1], tmp_10557[2] + tmp_10640[2]];
    signal tmp_10642[3] <== CMul()(challengeQ, tmp_10641);
    signal tmp_10643[3] <== CMul()(evals[61], evals[61]);
    signal tmp_10644[3] <== CMul()(tmp_10643, tmp_10643);
    signal tmp_10645[3] <== CMul()(tmp_10644, tmp_10643);
    signal tmp_10646[3] <== CMul()(tmp_10645, evals[61]);
    signal tmp_10647[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10648[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10649[3] <== [tmp_10648[0] * 369323044064, tmp_10648[1] * 369323044064, tmp_10648[2] * 369323044064];
    signal tmp_10650[3] <== [tmp_10565[0] + tmp_10649[0], tmp_10565[1] + tmp_10649[1], tmp_10565[2] + tmp_10649[2]];
    signal tmp_10651[3] <== [13905431266086421911 * tmp_10650[0], 13905431266086421911 * tmp_10650[1], 13905431266086421911 * tmp_10650[2]];
    signal tmp_10652[3] <== [tmp_10647[0] + tmp_10651[0], tmp_10647[1] + tmp_10651[1], tmp_10647[2] + tmp_10651[2]];
    signal tmp_10653[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10654[3] <== [tmp_10653[0] * 376274279152, tmp_10653[1] * 376274279152, tmp_10653[2] * 376274279152];
    signal tmp_10655[3] <== [tmp_10570[0] + tmp_10654[0], tmp_10570[1] + tmp_10654[1], tmp_10570[2] + tmp_10654[2]];
    signal tmp_10656[3] <== [305991708664976914 * tmp_10655[0], 305991708664976914 * tmp_10655[1], 305991708664976914 * tmp_10655[2]];
    signal tmp_10657[3] <== [tmp_10652[0] + tmp_10656[0], tmp_10652[1] + tmp_10656[1], tmp_10652[2] + tmp_10656[2]];
    signal tmp_10658[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10659[3] <== [tmp_10658[0] * 361996018532, tmp_10658[1] * 361996018532, tmp_10658[2] * 361996018532];
    signal tmp_10660[3] <== [tmp_10575[0] + tmp_10659[0], tmp_10575[1] + tmp_10659[1], tmp_10575[2] + tmp_10659[2]];
    signal tmp_10661[3] <== [15209999190551796912 * tmp_10660[0], 15209999190551796912 * tmp_10660[1], 15209999190551796912 * tmp_10660[2]];
    signal tmp_10662[3] <== [tmp_10657[0] + tmp_10661[0], tmp_10657[1] + tmp_10661[1], tmp_10657[2] + tmp_10661[2]];
    signal tmp_10663[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10664[3] <== [tmp_10663[0] * 301447583530, tmp_10663[1] * 301447583530, tmp_10663[2] * 301447583530];
    signal tmp_10665[3] <== [tmp_10580[0] + tmp_10664[0], tmp_10580[1] + tmp_10664[1], tmp_10580[2] + tmp_10664[2]];
    signal tmp_10666[3] <== [11028393659978236139 * tmp_10665[0], 11028393659978236139 * tmp_10665[1], 11028393659978236139 * tmp_10665[2]];
    signal tmp_10667[3] <== [tmp_10662[0] + tmp_10666[0], tmp_10662[1] + tmp_10666[1], tmp_10662[2] + tmp_10666[2]];
    signal tmp_10668[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10669[3] <== [tmp_10668[0] * 377601912356, tmp_10668[1] * 377601912356, tmp_10668[2] * 377601912356];
    signal tmp_10670[3] <== [tmp_10585[0] + tmp_10669[0], tmp_10585[1] + tmp_10669[1], tmp_10585[2] + tmp_10669[2]];
    signal tmp_10671[3] <== [17904306168706867216 * tmp_10670[0], 17904306168706867216 * tmp_10670[1], 17904306168706867216 * tmp_10670[2]];
    signal tmp_10672[3] <== [tmp_10667[0] + tmp_10671[0], tmp_10667[1] + tmp_10671[1], tmp_10667[2] + tmp_10671[2]];
    signal tmp_10673[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10674[3] <== [tmp_10673[0] * 345235981905, tmp_10673[1] * 345235981905, tmp_10673[2] * 345235981905];
    signal tmp_10675[3] <== [tmp_10590[0] + tmp_10674[0], tmp_10590[1] + tmp_10674[1], tmp_10590[2] + tmp_10674[2]];
    signal tmp_10676[3] <== [8374188644407616260 * tmp_10675[0], 8374188644407616260 * tmp_10675[1], 8374188644407616260 * tmp_10675[2]];
    signal tmp_10677[3] <== [tmp_10672[0] + tmp_10676[0], tmp_10672[1] + tmp_10676[1], tmp_10672[2] + tmp_10676[2]];
    signal tmp_10678[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10679[3] <== [tmp_10678[0] * 287662427631, tmp_10678[1] * 287662427631, tmp_10678[2] * 287662427631];
    signal tmp_10680[3] <== [tmp_10595[0] + tmp_10679[0], tmp_10595[1] + tmp_10679[1], tmp_10595[2] + tmp_10679[2]];
    signal tmp_10681[3] <== [680800612778784853 * tmp_10680[0], 680800612778784853 * tmp_10680[1], 680800612778784853 * tmp_10680[2]];
    signal tmp_10682[3] <== [tmp_10677[0] + tmp_10681[0], tmp_10677[1] + tmp_10681[1], tmp_10677[2] + tmp_10681[2]];
    signal tmp_10683[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10684[3] <== [tmp_10683[0] * 284662670472, tmp_10683[1] * 284662670472, tmp_10683[2] * 284662670472];
    signal tmp_10685[3] <== [tmp_10600[0] + tmp_10684[0], tmp_10600[1] + tmp_10684[1], tmp_10600[2] + tmp_10684[2]];
    signal tmp_10686[3] <== [5417240084914372136 * tmp_10685[0], 5417240084914372136 * tmp_10685[1], 5417240084914372136 * tmp_10685[2]];
    signal tmp_10687[3] <== [tmp_10682[0] + tmp_10686[0], tmp_10682[1] + tmp_10686[1], tmp_10682[2] + tmp_10686[2]];
    signal tmp_10688[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10689[3] <== [tmp_10688[0] * 404765160652, tmp_10688[1] * 404765160652, tmp_10688[2] * 404765160652];
    signal tmp_10690[3] <== [tmp_10605[0] + tmp_10689[0], tmp_10605[1] + tmp_10689[1], tmp_10605[2] + tmp_10689[2]];
    signal tmp_10691[3] <== [1393710566264274938 * tmp_10690[0], 1393710566264274938 * tmp_10690[1], 1393710566264274938 * tmp_10690[2]];
    signal tmp_10692[3] <== [tmp_10687[0] + tmp_10691[0], tmp_10687[1] + tmp_10691[1], tmp_10687[2] + tmp_10691[2]];
    signal tmp_10693[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10694[3] <== [tmp_10693[0] * 351806900899, tmp_10693[1] * 351806900899, tmp_10693[2] * 351806900899];
    signal tmp_10695[3] <== [tmp_10610[0] + tmp_10694[0], tmp_10610[1] + tmp_10694[1], tmp_10610[2] + tmp_10694[2]];
    signal tmp_10696[3] <== [14148046894039047274 * tmp_10695[0], 14148046894039047274 * tmp_10695[1], 14148046894039047274 * tmp_10695[2]];
    signal tmp_10697[3] <== [tmp_10692[0] + tmp_10696[0], tmp_10692[1] + tmp_10696[1], tmp_10692[2] + tmp_10696[2]];
    signal tmp_10698[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10699[3] <== [tmp_10698[0] * 315580522525, tmp_10698[1] * 315580522525, tmp_10698[2] * 315580522525];
    signal tmp_10700[3] <== [tmp_10615[0] + tmp_10699[0], tmp_10615[1] + tmp_10699[1], tmp_10615[2] + tmp_10699[2]];
    signal tmp_10701[3] <== [3244735513351203558 * tmp_10700[0], 3244735513351203558 * tmp_10700[1], 3244735513351203558 * tmp_10700[2]];
    signal tmp_10702[3] <== [tmp_10697[0] + tmp_10701[0], tmp_10697[1] + tmp_10701[1], tmp_10697[2] + tmp_10701[2]];
    signal tmp_10703[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10704[3] <== [tmp_10703[0] * 348317304582, tmp_10703[1] * 348317304582, tmp_10703[2] * 348317304582];
    signal tmp_10705[3] <== [tmp_10620[0] + tmp_10704[0], tmp_10620[1] + tmp_10704[1], tmp_10620[2] + tmp_10704[2]];
    signal tmp_10706[3] <== [4744494926695798520 * tmp_10705[0], 4744494926695798520 * tmp_10705[1], 4744494926695798520 * tmp_10705[2]];
    signal tmp_10707[3] <== [tmp_10702[0] + tmp_10706[0], tmp_10702[1] + tmp_10706[1], tmp_10702[2] + tmp_10706[2]];
    signal tmp_10708[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10709[3] <== [tmp_10708[0] * 385400627347, tmp_10708[1] * 385400627347, tmp_10708[2] * 385400627347];
    signal tmp_10710[3] <== [tmp_10625[0] + tmp_10709[0], tmp_10625[1] + tmp_10709[1], tmp_10625[2] + tmp_10709[2]];
    signal tmp_10711[3] <== [12411244701441950267 * tmp_10710[0], 12411244701441950267 * tmp_10710[1], 12411244701441950267 * tmp_10710[2]];
    signal tmp_10712[3] <== [tmp_10707[0] + tmp_10711[0], tmp_10707[1] + tmp_10711[1], tmp_10707[2] + tmp_10711[2]];
    signal tmp_10713[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10714[3] <== [tmp_10713[0] * 323348446961, tmp_10713[1] * 323348446961, tmp_10713[2] * 323348446961];
    signal tmp_10715[3] <== [tmp_10630[0] + tmp_10714[0], tmp_10630[1] + tmp_10714[1], tmp_10630[2] + tmp_10714[2]];
    signal tmp_10716[3] <== [8803037078966971424 * tmp_10715[0], 8803037078966971424 * tmp_10715[1], 8803037078966971424 * tmp_10715[2]];
    signal tmp_10717[3] <== [tmp_10712[0] + tmp_10716[0], tmp_10712[1] + tmp_10716[1], tmp_10712[2] + tmp_10716[2]];
    signal tmp_10718[3] <== [tmp_10561[0] + 13846825240309565987, tmp_10561[1], tmp_10561[2]];
    signal tmp_10719[3] <== [tmp_10718[0] * 363383916243, tmp_10718[1] * 363383916243, tmp_10718[2] * 363383916243];
    signal tmp_10720[3] <== [tmp_10635[0] + tmp_10719[0], tmp_10635[1] + tmp_10719[1], tmp_10635[2] + tmp_10719[2]];
    signal tmp_10721[3] <== [5194306366755844852 * tmp_10720[0], 5194306366755844852 * tmp_10720[1], 5194306366755844852 * tmp_10720[2]];
    signal tmp_10722[3] <== [tmp_10717[0] + tmp_10721[0], tmp_10717[1] + tmp_10721[1], tmp_10717[2] + tmp_10721[2]];
    signal tmp_10723[3] <== [tmp_10722[0] * 1, tmp_10722[1] * 1, tmp_10722[2] * 1];
    signal tmp_10724[3] <== [evals[62][0] - tmp_10723[0], evals[62][1] - tmp_10723[1], evals[62][2] - tmp_10723[2]];
    signal tmp_10725[3] <== CMul()(tmp_6068, tmp_10724);
    signal tmp_10726[3] <== [tmp_10642[0] + tmp_10725[0], tmp_10642[1] + tmp_10725[1], tmp_10642[2] + tmp_10725[2]];
    signal tmp_10727[3] <== CMul()(challengeQ, tmp_10726);
    signal tmp_10728[3] <== CMul()(evals[62], evals[62]);
    signal tmp_10729[3] <== CMul()(tmp_10728, tmp_10728);
    signal tmp_10730[3] <== CMul()(tmp_10729, tmp_10728);
    signal tmp_10731[3] <== CMul()(tmp_10730, evals[62]);
    signal tmp_10732[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10733[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10734[3] <== [tmp_10733[0] * 1188745580, tmp_10733[1] * 1188745580, tmp_10733[2] * 1188745580];
    signal tmp_10735[3] <== [tmp_10650[0] + tmp_10734[0], tmp_10650[1] + tmp_10734[1], tmp_10650[2] + tmp_10734[2]];
    signal tmp_10736[3] <== [530243823762044158 * tmp_10735[0], 530243823762044158 * tmp_10735[1], 530243823762044158 * tmp_10735[2]];
    signal tmp_10737[3] <== [tmp_10732[0] + tmp_10736[0], tmp_10732[1] + tmp_10736[1], tmp_10732[2] + tmp_10736[2]];
    signal tmp_10738[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10739[3] <== [tmp_10738[0] * 1055003602, tmp_10738[1] * 1055003602, tmp_10738[2] * 1055003602];
    signal tmp_10740[3] <== [tmp_10655[0] + tmp_10739[0], tmp_10655[1] + tmp_10739[1], tmp_10655[2] + tmp_10739[2]];
    signal tmp_10741[3] <== [2698957419264379100 * tmp_10740[0], 2698957419264379100 * tmp_10740[1], 2698957419264379100 * tmp_10740[2]];
    signal tmp_10742[3] <== [tmp_10737[0] + tmp_10741[0], tmp_10737[1] + tmp_10741[1], tmp_10737[2] + tmp_10741[2]];
    signal tmp_10743[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10744[3] <== [tmp_10743[0] * 785416201, tmp_10743[1] * 785416201, tmp_10743[2] * 785416201];
    signal tmp_10745[3] <== [tmp_10660[0] + tmp_10744[0], tmp_10660[1] + tmp_10744[1], tmp_10660[2] + tmp_10744[2]];
    signal tmp_10746[3] <== [14296668284422018113 * tmp_10745[0], 14296668284422018113 * tmp_10745[1], 14296668284422018113 * tmp_10745[2]];
    signal tmp_10747[3] <== [tmp_10742[0] + tmp_10746[0], tmp_10742[1] + tmp_10746[1], tmp_10742[2] + tmp_10746[2]];
    signal tmp_10748[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10749[3] <== [tmp_10748[0] * 868051025, tmp_10748[1] * 868051025, tmp_10748[2] * 868051025];
    signal tmp_10750[3] <== [tmp_10665[0] + tmp_10749[0], tmp_10665[1] + tmp_10749[1], tmp_10665[2] + tmp_10749[2]];
    signal tmp_10751[3] <== [8333713848018243814 * tmp_10750[0], 8333713848018243814 * tmp_10750[1], 8333713848018243814 * tmp_10750[2]];
    signal tmp_10752[3] <== [tmp_10747[0] + tmp_10751[0], tmp_10747[1] + tmp_10751[1], tmp_10747[2] + tmp_10751[2]];
    signal tmp_10753[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10754[3] <== [tmp_10753[0] * 1135832507, tmp_10753[1] * 1135832507, tmp_10753[2] * 1135832507];
    signal tmp_10755[3] <== [tmp_10670[0] + tmp_10754[0], tmp_10670[1] + tmp_10754[1], tmp_10670[2] + tmp_10754[2]];
    signal tmp_10756[3] <== [12238342728063130848 * tmp_10755[0], 12238342728063130848 * tmp_10755[1], 12238342728063130848 * tmp_10755[2]];
    signal tmp_10757[3] <== [tmp_10752[0] + tmp_10756[0], tmp_10752[1] + tmp_10756[1], tmp_10752[2] + tmp_10756[2]];
    signal tmp_10758[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10759[3] <== [tmp_10758[0] * 1004853599, tmp_10758[1] * 1004853599, tmp_10758[2] * 1004853599];
    signal tmp_10760[3] <== [tmp_10675[0] + tmp_10759[0], tmp_10675[1] + tmp_10759[1], tmp_10675[2] + tmp_10759[2]];
    signal tmp_10761[3] <== [14599889067722749515 * tmp_10760[0], 14599889067722749515 * tmp_10760[1], 14599889067722749515 * tmp_10760[2]];
    signal tmp_10762[3] <== [tmp_10757[0] + tmp_10761[0], tmp_10757[1] + tmp_10761[1], tmp_10757[2] + tmp_10761[2]];
    signal tmp_10763[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10764[3] <== [tmp_10763[0] * 904741729, tmp_10763[1] * 904741729, tmp_10763[2] * 904741729];
    signal tmp_10765[3] <== [tmp_10680[0] + tmp_10764[0], tmp_10680[1] + tmp_10764[1], tmp_10680[2] + tmp_10764[2]];
    signal tmp_10766[3] <== [1498123092862756804 * tmp_10765[0], 1498123092862756804 * tmp_10765[1], 1498123092862756804 * tmp_10765[2]];
    signal tmp_10767[3] <== [tmp_10762[0] + tmp_10766[0], tmp_10762[1] + tmp_10766[1], tmp_10762[2] + tmp_10766[2]];
    signal tmp_10768[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10769[3] <== [tmp_10768[0] * 809824679, tmp_10768[1] * 809824679, tmp_10768[2] * 809824679];
    signal tmp_10770[3] <== [tmp_10685[0] + tmp_10769[0], tmp_10685[1] + tmp_10769[1], tmp_10685[2] + tmp_10769[2]];
    signal tmp_10771[3] <== [6951489160695905236 * tmp_10770[0], 6951489160695905236 * tmp_10770[1], 6951489160695905236 * tmp_10770[2]];
    signal tmp_10772[3] <== [tmp_10767[0] + tmp_10771[0], tmp_10767[1] + tmp_10771[1], tmp_10767[2] + tmp_10771[2]];
    signal tmp_10773[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10774[3] <== [tmp_10773[0] * 980810992, tmp_10773[1] * 980810992, tmp_10773[2] * 980810992];
    signal tmp_10775[3] <== [tmp_10690[0] + tmp_10774[0], tmp_10690[1] + tmp_10774[1], tmp_10690[2] + tmp_10774[2]];
    signal tmp_10776[3] <== [15663438810331591677 * tmp_10775[0], 15663438810331591677 * tmp_10775[1], 15663438810331591677 * tmp_10775[2]];
    signal tmp_10777[3] <== [tmp_10772[0] + tmp_10776[0], tmp_10772[1] + tmp_10776[1], tmp_10772[2] + tmp_10776[2]];
    signal tmp_10778[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10779[3] <== [tmp_10778[0] * 1178194302, tmp_10778[1] * 1178194302, tmp_10778[2] * 1178194302];
    signal tmp_10780[3] <== [tmp_10695[0] + tmp_10779[0], tmp_10695[1] + tmp_10779[1], tmp_10695[2] + tmp_10779[2]];
    signal tmp_10781[3] <== [13759896603824577231 * tmp_10780[0], 13759896603824577231 * tmp_10780[1], 13759896603824577231 * tmp_10780[2]];
    signal tmp_10782[3] <== [tmp_10777[0] + tmp_10781[0], tmp_10777[1] + tmp_10781[1], tmp_10777[2] + tmp_10781[2]];
    signal tmp_10783[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10784[3] <== [tmp_10783[0] * 1159788697, tmp_10783[1] * 1159788697, tmp_10783[2] * 1159788697];
    signal tmp_10785[3] <== [tmp_10700[0] + tmp_10784[0], tmp_10700[1] + tmp_10784[1], tmp_10700[2] + tmp_10784[2]];
    signal tmp_10786[3] <== [4898543744152164832 * tmp_10785[0], 4898543744152164832 * tmp_10785[1], 4898543744152164832 * tmp_10785[2]];
    signal tmp_10787[3] <== [tmp_10782[0] + tmp_10786[0], tmp_10782[1] + tmp_10786[1], tmp_10782[2] + tmp_10786[2]];
    signal tmp_10788[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10789[3] <== [tmp_10788[0] * 949043013, tmp_10788[1] * 949043013, tmp_10788[2] * 949043013];
    signal tmp_10790[3] <== [tmp_10705[0] + tmp_10789[0], tmp_10705[1] + tmp_10789[1], tmp_10705[2] + tmp_10789[2]];
    signal tmp_10791[3] <== [15908946426636755274 * tmp_10790[0], 15908946426636755274 * tmp_10790[1], 15908946426636755274 * tmp_10790[2]];
    signal tmp_10792[3] <== [tmp_10787[0] + tmp_10791[0], tmp_10787[1] + tmp_10791[1], tmp_10787[2] + tmp_10791[2]];
    signal tmp_10793[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10794[3] <== [tmp_10793[0] * 1001466621, tmp_10793[1] * 1001466621, tmp_10793[2] * 1001466621];
    signal tmp_10795[3] <== [tmp_10710[0] + tmp_10794[0], tmp_10710[1] + tmp_10794[1], tmp_10710[2] + tmp_10794[2]];
    signal tmp_10796[3] <== [13145973700442701961 * tmp_10795[0], 13145973700442701961 * tmp_10795[1], 13145973700442701961 * tmp_10795[2]];
    signal tmp_10797[3] <== [tmp_10792[0] + tmp_10796[0], tmp_10792[1] + tmp_10796[1], tmp_10792[2] + tmp_10796[2]];
    signal tmp_10798[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10799[3] <== [tmp_10798[0] * 1011628637, tmp_10798[1] * 1011628637, tmp_10798[2] * 1011628637];
    signal tmp_10800[3] <== [tmp_10715[0] + tmp_10799[0], tmp_10715[1] + tmp_10799[1], tmp_10715[2] + tmp_10799[2]];
    signal tmp_10801[3] <== [17400180121643364107 * tmp_10800[0], 17400180121643364107 * tmp_10800[1], 17400180121643364107 * tmp_10800[2]];
    signal tmp_10802[3] <== [tmp_10797[0] + tmp_10801[0], tmp_10797[1] + tmp_10801[1], tmp_10797[2] + tmp_10801[2]];
    signal tmp_10803[3] <== [tmp_10646[0] + 4463709219992588910, tmp_10646[1], tmp_10646[2]];
    signal tmp_10804[3] <== [tmp_10803[0] * 924759953, tmp_10803[1] * 924759953, tmp_10803[2] * 924759953];
    signal tmp_10805[3] <== [tmp_10720[0] + tmp_10804[0], tmp_10720[1] + tmp_10804[1], tmp_10720[2] + tmp_10804[2]];
    signal tmp_10806[3] <== [4391335980556159683 * tmp_10805[0], 4391335980556159683 * tmp_10805[1], 4391335980556159683 * tmp_10805[2]];
    signal tmp_10807[3] <== [tmp_10802[0] + tmp_10806[0], tmp_10802[1] + tmp_10806[1], tmp_10802[2] + tmp_10806[2]];
    signal tmp_10808[3] <== [tmp_10807[0] * 1, tmp_10807[1] * 1, tmp_10807[2] * 1];
    signal tmp_10809[3] <== [evals[63][0] - tmp_10808[0], evals[63][1] - tmp_10808[1], evals[63][2] - tmp_10808[2]];
    signal tmp_10810[3] <== CMul()(tmp_6068, tmp_10809);
    signal tmp_10811[3] <== [tmp_10727[0] + tmp_10810[0], tmp_10727[1] + tmp_10810[1], tmp_10727[2] + tmp_10810[2]];
    signal tmp_10812[3] <== CMul()(challengeQ, tmp_10811);
    signal tmp_10813[3] <== CMul()(evals[63], evals[63]);
    signal tmp_10814[3] <== CMul()(tmp_10813, tmp_10813);
    signal tmp_10815[3] <== CMul()(tmp_10814, tmp_10813);
    signal tmp_10816[3] <== CMul()(tmp_10815, evals[63]);
    signal tmp_10817[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10818[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10819[3] <== [tmp_10818[0] * 2475856, tmp_10818[1] * 2475856, tmp_10818[2] * 2475856];
    signal tmp_10820[3] <== [tmp_10735[0] + tmp_10819[0], tmp_10735[1] + tmp_10819[1], tmp_10735[2] + tmp_10819[2]];
    signal tmp_10821[3] <== [1787101036502119356 * tmp_10820[0], 1787101036502119356 * tmp_10820[1], 1787101036502119356 * tmp_10820[2]];
    signal tmp_10822[3] <== [tmp_10817[0] + tmp_10821[0], tmp_10817[1] + tmp_10821[1], tmp_10817[2] + tmp_10821[2]];
    signal tmp_10823[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10824[3] <== [tmp_10823[0] * 3337618, tmp_10823[1] * 3337618, tmp_10823[2] * 3337618];
    signal tmp_10825[3] <== [tmp_10740[0] + tmp_10824[0], tmp_10740[1] + tmp_10824[1], tmp_10740[2] + tmp_10824[2]];
    signal tmp_10826[3] <== [6714829819836869612 * tmp_10825[0], 6714829819836869612 * tmp_10825[1], 6714829819836869612 * tmp_10825[2]];
    signal tmp_10827[3] <== [tmp_10822[0] + tmp_10826[0], tmp_10822[1] + tmp_10826[1], tmp_10822[2] + tmp_10826[2]];
    signal tmp_10828[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10829[3] <== [tmp_10828[0] * 4161263, tmp_10828[1] * 4161263, tmp_10828[2] * 4161263];
    signal tmp_10830[3] <== [tmp_10745[0] + tmp_10829[0], tmp_10745[1] + tmp_10829[1], tmp_10745[2] + tmp_10829[2]];
    signal tmp_10831[3] <== [8614678868105523772 * tmp_10830[0], 8614678868105523772 * tmp_10830[1], 8614678868105523772 * tmp_10830[2]];
    signal tmp_10832[3] <== [tmp_10827[0] + tmp_10831[0], tmp_10827[1] + tmp_10831[1], tmp_10827[2] + tmp_10831[2]];
    signal tmp_10833[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10834[3] <== [tmp_10833[0] * 3129126, tmp_10833[1] * 3129126, tmp_10833[2] * 3129126];
    signal tmp_10835[3] <== [tmp_10750[0] + tmp_10834[0], tmp_10750[1] + tmp_10834[1], tmp_10750[2] + tmp_10834[2]];
    signal tmp_10836[3] <== [7372503606097811728 * tmp_10835[0], 7372503606097811728 * tmp_10835[1], 7372503606097811728 * tmp_10835[2]];
    signal tmp_10837[3] <== [tmp_10832[0] + tmp_10836[0], tmp_10832[1] + tmp_10836[1], tmp_10832[2] + tmp_10836[2]];
    signal tmp_10838[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10839[3] <== [tmp_10838[0] * 2071505, tmp_10838[1] * 2071505, tmp_10838[2] * 2071505];
    signal tmp_10840[3] <== [tmp_10755[0] + tmp_10839[0], tmp_10755[1] + tmp_10839[1], tmp_10755[2] + tmp_10839[2]];
    signal tmp_10841[3] <== [16967913077348192867 * tmp_10840[0], 16967913077348192867 * tmp_10840[1], 16967913077348192867 * tmp_10840[2]];
    signal tmp_10842[3] <== [tmp_10837[0] + tmp_10841[0], tmp_10837[1] + tmp_10841[1], tmp_10837[2] + tmp_10841[2]];
    signal tmp_10843[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10844[3] <== [tmp_10843[0] * 3373463, tmp_10843[1] * 3373463, tmp_10843[2] * 3373463];
    signal tmp_10845[3] <== [tmp_10760[0] + tmp_10844[0], tmp_10760[1] + tmp_10844[1], tmp_10760[2] + tmp_10844[2]];
    signal tmp_10846[3] <== [746855177740798579 * tmp_10845[0], 746855177740798579 * tmp_10845[1], 746855177740798579 * tmp_10845[2]];
    signal tmp_10847[3] <== [tmp_10842[0] + tmp_10846[0], tmp_10842[1] + tmp_10846[1], tmp_10842[2] + tmp_10846[2]];
    signal tmp_10848[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10849[3] <== [tmp_10848[0] * 2975691, tmp_10848[1] * 2975691, tmp_10848[2] * 2975691];
    signal tmp_10850[3] <== [tmp_10765[0] + tmp_10849[0], tmp_10765[1] + tmp_10849[1], tmp_10765[2] + tmp_10849[2]];
    signal tmp_10851[3] <== [8948543211894560314 * tmp_10850[0], 8948543211894560314 * tmp_10850[1], 8948543211894560314 * tmp_10850[2]];
    signal tmp_10852[3] <== [tmp_10847[0] + tmp_10851[0], tmp_10847[1] + tmp_10851[1], tmp_10847[2] + tmp_10851[2]];
    signal tmp_10853[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10854[3] <== [tmp_10853[0] * 1742470, tmp_10853[1] * 1742470, tmp_10853[2] * 1742470];
    signal tmp_10855[3] <== [tmp_10770[0] + tmp_10854[0], tmp_10770[1] + tmp_10854[1], tmp_10770[2] + tmp_10854[2]];
    signal tmp_10856[3] <== [7250314975625247352 * tmp_10855[0], 7250314975625247352 * tmp_10855[1], 7250314975625247352 * tmp_10855[2]];
    signal tmp_10857[3] <== [tmp_10852[0] + tmp_10856[0], tmp_10852[1] + tmp_10856[1], tmp_10852[2] + tmp_10856[2]];
    signal tmp_10858[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10859[3] <== [tmp_10858[0] * 2828204, tmp_10858[1] * 2828204, tmp_10858[2] * 2828204];
    signal tmp_10860[3] <== [tmp_10775[0] + tmp_10859[0], tmp_10775[1] + tmp_10859[1], tmp_10775[2] + tmp_10859[2]];
    signal tmp_10861[3] <== [11154498990878763202 * tmp_10860[0], 11154498990878763202 * tmp_10860[1], 11154498990878763202 * tmp_10860[2]];
    signal tmp_10862[3] <== [tmp_10857[0] + tmp_10861[0], tmp_10857[1] + tmp_10861[1], tmp_10857[2] + tmp_10861[2]];
    signal tmp_10863[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10864[3] <== [tmp_10863[0] * 3695590, tmp_10863[1] * 3695590, tmp_10863[2] * 3695590];
    signal tmp_10865[3] <== [tmp_10780[0] + tmp_10864[0], tmp_10780[1] + tmp_10864[1], tmp_10780[2] + tmp_10864[2]];
    signal tmp_10866[3] <== [9368653935206961209 * tmp_10865[0], 9368653935206961209 * tmp_10865[1], 9368653935206961209 * tmp_10865[2]];
    signal tmp_10867[3] <== [tmp_10862[0] + tmp_10866[0], tmp_10862[1] + tmp_10866[1], tmp_10862[2] + tmp_10866[2]];
    signal tmp_10868[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10869[3] <== [tmp_10868[0] * 1809935, tmp_10868[1] * 1809935, tmp_10868[2] * 1809935];
    signal tmp_10870[3] <== [tmp_10785[0] + tmp_10869[0], tmp_10785[1] + tmp_10869[1], tmp_10785[2] + tmp_10869[2]];
    signal tmp_10871[3] <== [5524036893624759211 * tmp_10870[0], 5524036893624759211 * tmp_10870[1], 5524036893624759211 * tmp_10870[2]];
    signal tmp_10872[3] <== [tmp_10867[0] + tmp_10871[0], tmp_10867[1] + tmp_10871[1], tmp_10867[2] + tmp_10871[2]];
    signal tmp_10873[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10874[3] <== [tmp_10873[0] * 2316312, tmp_10873[1] * 2316312, tmp_10873[2] * 2316312];
    signal tmp_10875[3] <== [tmp_10790[0] + tmp_10874[0], tmp_10790[1] + tmp_10874[1], tmp_10790[2] + tmp_10874[2]];
    signal tmp_10876[3] <== [2521926361859468511 * tmp_10875[0], 2521926361859468511 * tmp_10875[1], 2521926361859468511 * tmp_10875[2]];
    signal tmp_10877[3] <== [tmp_10872[0] + tmp_10876[0], tmp_10872[1] + tmp_10876[1], tmp_10872[2] + tmp_10876[2]];
    signal tmp_10878[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10879[3] <== [tmp_10878[0] * 3448583, tmp_10878[1] * 3448583, tmp_10878[2] * 3448583];
    signal tmp_10880[3] <== [tmp_10795[0] + tmp_10879[0], tmp_10795[1] + tmp_10879[1], tmp_10795[2] + tmp_10879[2]];
    signal tmp_10881[3] <== [14797218746248438904 * tmp_10880[0], 14797218746248438904 * tmp_10880[1], 14797218746248438904 * tmp_10880[2]];
    signal tmp_10882[3] <== [tmp_10877[0] + tmp_10881[0], tmp_10877[1] + tmp_10881[1], tmp_10877[2] + tmp_10881[2]];
    signal tmp_10883[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10884[3] <== [tmp_10883[0] * 2986173, tmp_10883[1] * 2986173, tmp_10883[2] * 2986173];
    signal tmp_10885[3] <== [tmp_10800[0] + tmp_10884[0], tmp_10800[1] + tmp_10884[1], tmp_10800[2] + tmp_10884[2]];
    signal tmp_10886[3] <== [10896085731851505890 * tmp_10885[0], 10896085731851505890 * tmp_10885[1], 10896085731851505890 * tmp_10885[2]];
    signal tmp_10887[3] <== [tmp_10882[0] + tmp_10886[0], tmp_10882[1] + tmp_10886[1], tmp_10882[2] + tmp_10886[2]];
    signal tmp_10888[3] <== [tmp_10731[0] + 15514028834701829198, tmp_10731[1], tmp_10731[2]];
    signal tmp_10889[3] <== [tmp_10888[0] * 2518923, tmp_10888[1] * 2518923, tmp_10888[2] * 2518923];
    signal tmp_10890[3] <== [tmp_10805[0] + tmp_10889[0], tmp_10805[1] + tmp_10889[1], tmp_10805[2] + tmp_10889[2]];
    signal tmp_10891[3] <== [9304799240492850786 * tmp_10890[0], 9304799240492850786 * tmp_10890[1], 9304799240492850786 * tmp_10890[2]];
    signal tmp_10892[3] <== [tmp_10887[0] + tmp_10891[0], tmp_10887[1] + tmp_10891[1], tmp_10887[2] + tmp_10891[2]];
    signal tmp_10893[3] <== [tmp_10892[0] * 1, tmp_10892[1] * 1, tmp_10892[2] * 1];
    signal tmp_10894[3] <== [evals[64][0] - tmp_10893[0], evals[64][1] - tmp_10893[1], evals[64][2] - tmp_10893[2]];
    signal tmp_10895[3] <== CMul()(tmp_6068, tmp_10894);
    signal tmp_10896[3] <== [tmp_10812[0] + tmp_10895[0], tmp_10812[1] + tmp_10895[1], tmp_10812[2] + tmp_10895[2]];
    signal tmp_10897[3] <== CMul()(challengeQ, tmp_10896);
    signal tmp_10898[3] <== CMul()(evals[64], evals[64]);
    signal tmp_10899[3] <== CMul()(tmp_10898, tmp_10898);
    signal tmp_10900[3] <== CMul()(tmp_10899, tmp_10898);
    tmp_10901 <== CMul()(tmp_10900, evals[64]);
    signal tmp_10902[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_10903[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10904[3] <== [tmp_10903[0] * 3415, tmp_10903[1] * 3415, tmp_10903[2] * 3415];
    signal tmp_10905[3] <== [tmp_10820[0] + tmp_10904[0], tmp_10820[1] + tmp_10904[1], tmp_10820[2] + tmp_10904[2]];
    signal tmp_10906[3] <== [8193989947050030928 * tmp_10905[0], 8193989947050030928 * tmp_10905[1], 8193989947050030928 * tmp_10905[2]];
    signal tmp_10907[3] <== [tmp_10902[0] + tmp_10906[0], tmp_10902[1] + tmp_10906[1], tmp_10902[2] + tmp_10906[2]];
    signal tmp_10908[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10909[3] <== [tmp_10908[0] * 9781, tmp_10908[1] * 9781, tmp_10908[2] * 9781];
    signal tmp_10910[3] <== [tmp_10825[0] + tmp_10909[0], tmp_10825[1] + tmp_10909[1], tmp_10825[2] + tmp_10909[2]];
    signal tmp_10911[3] <== [8752166280136533348 * tmp_10910[0], 8752166280136533348 * tmp_10910[1], 8752166280136533348 * tmp_10910[2]];
    signal tmp_10912[3] <== [tmp_10907[0] + tmp_10911[0], tmp_10907[1] + tmp_10911[1], tmp_10907[2] + tmp_10911[2]];
    signal tmp_10913[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10914[3] <== [tmp_10913[0] * 5292, tmp_10913[1] * 5292, tmp_10913[2] * 5292];
    signal tmp_10915[3] <== [tmp_10830[0] + tmp_10914[0], tmp_10830[1] + tmp_10914[1], tmp_10830[2] + tmp_10914[2]];
    signal tmp_10916[3] <== [13837989622971627817 * tmp_10915[0], 13837989622971627817 * tmp_10915[1], 13837989622971627817 * tmp_10915[2]];
    signal tmp_10917[3] <== [tmp_10912[0] + tmp_10916[0], tmp_10912[1] + tmp_10916[1], tmp_10912[2] + tmp_10916[2]];
    signal tmp_10918[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10919[3] <== [tmp_10918[0] * 4288, tmp_10918[1] * 4288, tmp_10918[2] * 4288];
    signal tmp_10920[3] <== [tmp_10835[0] + tmp_10919[0], tmp_10835[1] + tmp_10919[1], tmp_10835[2] + tmp_10919[2]];
    signal tmp_10921[3] <== [16232594342971813546 * tmp_10920[0], 16232594342971813546 * tmp_10920[1], 16232594342971813546 * tmp_10920[2]];
    signal tmp_10922[3] <== [tmp_10917[0] + tmp_10921[0], tmp_10917[1] + tmp_10921[1], tmp_10917[2] + tmp_10921[2]];
    signal tmp_10923[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10924[3] <== [tmp_10923[0] * 7724, tmp_10923[1] * 7724, tmp_10923[2] * 7724];
    signal tmp_10925[3] <== [tmp_10840[0] + tmp_10924[0], tmp_10840[1] + tmp_10924[1], tmp_10840[2] + tmp_10924[2]];
    signal tmp_10926[3] <== [16475524618737509996 * tmp_10925[0], 16475524618737509996 * tmp_10925[1], 16475524618737509996 * tmp_10925[2]];
    signal tmp_10927[3] <== [tmp_10922[0] + tmp_10926[0], tmp_10922[1] + tmp_10926[1], tmp_10922[2] + tmp_10926[2]];
    signal tmp_10928[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10929[3] <== [tmp_10928[0] * 13016, tmp_10928[1] * 13016, tmp_10928[2] * 13016];
    signal tmp_10930[3] <== [tmp_10845[0] + tmp_10929[0], tmp_10845[1] + tmp_10929[1], tmp_10845[2] + tmp_10929[2]];
    signal tmp_10931[3] <== [13012351369794380536 * tmp_10930[0], 13012351369794380536 * tmp_10930[1], 13012351369794380536 * tmp_10930[2]];
    signal tmp_10932[3] <== [tmp_10927[0] + tmp_10931[0], tmp_10927[1] + tmp_10931[1], tmp_10927[2] + tmp_10931[2]];
    signal tmp_10933[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10934[3] <== [tmp_10933[0] * 3835, tmp_10933[1] * 3835, tmp_10933[2] * 3835];
    signal tmp_10935[3] <== [tmp_10850[0] + tmp_10934[0], tmp_10850[1] + tmp_10934[1], tmp_10850[2] + tmp_10934[2]];
    signal tmp_10936[3] <== [1790980186545924701 * tmp_10935[0], 1790980186545924701 * tmp_10935[1], 1790980186545924701 * tmp_10935[2]];
    signal tmp_10937[3] <== [tmp_10932[0] + tmp_10936[0], tmp_10932[1] + tmp_10936[1], tmp_10932[2] + tmp_10936[2]];
    signal tmp_10938[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10939[3] <== [tmp_10938[0] * 5807, tmp_10938[1] * 5807, tmp_10938[2] * 5807];
    signal tmp_10940[3] <== [tmp_10855[0] + tmp_10939[0], tmp_10855[1] + tmp_10939[1], tmp_10855[2] + tmp_10939[2]];
    signal tmp_10941[3] <== [7932420325258865009 * tmp_10940[0], 7932420325258865009 * tmp_10940[1], 7932420325258865009 * tmp_10940[2]];
    signal tmp_10942[3] <== [tmp_10937[0] + tmp_10941[0], tmp_10937[1] + tmp_10941[1], tmp_10937[2] + tmp_10941[2]];
    signal tmp_10943[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10944[3] <== [tmp_10943[0] * 4933, tmp_10943[1] * 4933, tmp_10943[2] * 4933];
    signal tmp_10945[3] <== [tmp_10860[0] + tmp_10944[0], tmp_10860[1] + tmp_10944[1], tmp_10860[2] + tmp_10944[2]];
    signal tmp_10946[3] <== [12730743194106063118 * tmp_10945[0], 12730743194106063118 * tmp_10945[1], 12730743194106063118 * tmp_10945[2]];
    signal tmp_10947[3] <== [tmp_10942[0] + tmp_10946[0], tmp_10942[1] + tmp_10946[1], tmp_10942[2] + tmp_10946[2]];
    signal tmp_10948[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10949[3] <== [tmp_10948[0] * 8577, tmp_10948[1] * 8577, tmp_10948[2] * 8577];
    signal tmp_10950[3] <== [tmp_10865[0] + tmp_10949[0], tmp_10865[1] + tmp_10949[1], tmp_10865[2] + tmp_10949[2]];
    signal tmp_10951[3] <== [14158202809073107037 * tmp_10950[0], 14158202809073107037 * tmp_10950[1], 14158202809073107037 * tmp_10950[2]];
    signal tmp_10952[3] <== [tmp_10947[0] + tmp_10951[0], tmp_10947[1] + tmp_10951[1], tmp_10947[2] + tmp_10951[2]];
    signal tmp_10953[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10954[3] <== [tmp_10953[0] * 13125, tmp_10953[1] * 13125, tmp_10953[2] * 13125];
    tmp_10955 <== [tmp_10870[0] + tmp_10954[0], tmp_10870[1] + tmp_10954[1], tmp_10870[2] + tmp_10954[2]];
    signal tmp_10956[3] <== [8949967597911797979 * tmp_10955[0], 8949967597911797979 * tmp_10955[1], 8949967597911797979 * tmp_10955[2]];
    signal tmp_10957[3] <== [tmp_10952[0] + tmp_10956[0], tmp_10952[1] + tmp_10956[1], tmp_10952[2] + tmp_10956[2]];
    signal tmp_10958[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10959[3] <== [tmp_10958[0] * 16823, tmp_10958[1] * 16823, tmp_10958[2] * 16823];
    tmp_10960 <== [tmp_10875[0] + tmp_10959[0], tmp_10875[1] + tmp_10959[1], tmp_10875[2] + tmp_10959[2]];
    signal tmp_10961[3] <== [2528834519776872471 * tmp_10960[0], 2528834519776872471 * tmp_10960[1], 2528834519776872471 * tmp_10960[2]];
    signal tmp_10962[3] <== [tmp_10957[0] + tmp_10961[0], tmp_10957[1] + tmp_10961[1], tmp_10957[2] + tmp_10961[2]];
    signal tmp_10963[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10964[3] <== [tmp_10963[0] * 3127, tmp_10963[1] * 3127, tmp_10963[2] * 3127];
    tmp_10965 <== [tmp_10880[0] + tmp_10964[0], tmp_10880[1] + tmp_10964[1], tmp_10880[2] + tmp_10964[2]];
    signal tmp_10966[3] <== [12882437260298977882 * tmp_10965[0], 12882437260298977882 * tmp_10965[1], 12882437260298977882 * tmp_10965[2]];
    signal tmp_10967[3] <== [tmp_10962[0] + tmp_10966[0], tmp_10962[1] + tmp_10966[1], tmp_10962[2] + tmp_10966[2]];
    signal tmp_10968[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10969[3] <== [tmp_10968[0] * 8363, tmp_10968[1] * 8363, tmp_10968[2] * 8363];
    tmp_10970 <== [tmp_10885[0] + tmp_10969[0], tmp_10885[1] + tmp_10969[1], tmp_10885[2] + tmp_10969[2]];
    signal tmp_10971[3] <== [14815685759369386413 * tmp_10970[0], 14815685759369386413 * tmp_10970[1], 14815685759369386413 * tmp_10970[2]];
    signal tmp_10972[3] <== [tmp_10967[0] + tmp_10971[0], tmp_10967[1] + tmp_10971[1], tmp_10967[2] + tmp_10971[2]];
    signal tmp_10973[3] <== [tmp_10816[0] + 1027211584317448035, tmp_10816[1], tmp_10816[2]];
    signal tmp_10974[3] <== [tmp_10973[0] * 15859, tmp_10973[1] * 15859, tmp_10973[2] * 15859];
    tmp_10975 <== [tmp_10890[0] + tmp_10974[0], tmp_10890[1] + tmp_10974[1], tmp_10890[2] + tmp_10974[2]];
    signal tmp_10976[3] <== [6859761478933209387 * tmp_10975[0], 6859761478933209387 * tmp_10975[1], 6859761478933209387 * tmp_10975[2]];
    signal tmp_10977[3] <== [tmp_10972[0] + tmp_10976[0], tmp_10972[1] + tmp_10976[1], tmp_10972[2] + tmp_10976[2]];
    signal tmp_10978[3] <== [tmp_10977[0] * 1, tmp_10977[1] * 1, tmp_10977[2] * 1];
    signal tmp_10979[3] <== [evals[118][0] - tmp_10978[0], evals[118][1] - tmp_10978[1], evals[118][2] - tmp_10978[2]];
    signal tmp_10980[3] <== CMul()(tmp_6068, tmp_10979);
    signal tmp_10981[3] <== [tmp_10897[0] + tmp_10980[0], tmp_10897[1] + tmp_10980[1], tmp_10897[2] + tmp_10980[2]];
    signal tmp_10982[3] <== CMul()(challengeQ, tmp_10981);
    signal tmp_10983[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_10984[3] <== [tmp_10983[0] * 3, tmp_10983[1] * 3, tmp_10983[2] * 3];
    signal tmp_10985[3] <== [tmp_10905[0] + tmp_10984[0], tmp_10905[1] + tmp_10984[1], tmp_10905[2] + tmp_10984[2]];
    signal tmp_10986[3] <== [tmp_10985[0] * 1, tmp_10985[1] * 1, tmp_10985[2] * 1];
    signal tmp_10987[3] <== [evals[119][0] - tmp_10986[0], evals[119][1] - tmp_10986[1], evals[119][2] - tmp_10986[2]];
    signal tmp_10988[3] <== CMul()(tmp_6068, tmp_10987);
    signal tmp_10989[3] <== [tmp_10982[0] + tmp_10988[0], tmp_10982[1] + tmp_10988[1], tmp_10982[2] + tmp_10988[2]];
    signal tmp_10990[3] <== CMul()(challengeQ, tmp_10989);
    signal tmp_10991[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_10992[3] <== [tmp_10991[0] * 13, tmp_10991[1] * 13, tmp_10991[2] * 13];
    signal tmp_10993[3] <== [tmp_10910[0] + tmp_10992[0], tmp_10910[1] + tmp_10992[1], tmp_10910[2] + tmp_10992[2]];
    signal tmp_10994[3] <== [tmp_10993[0] * 1, tmp_10993[1] * 1, tmp_10993[2] * 1];
    signal tmp_10995[3] <== [evals[120][0] - tmp_10994[0], evals[120][1] - tmp_10994[1], evals[120][2] - tmp_10994[2]];
    signal tmp_10996[3] <== CMul()(tmp_6068, tmp_10995);
    signal tmp_10997[3] <== [tmp_10990[0] + tmp_10996[0], tmp_10990[1] + tmp_10996[1], tmp_10990[2] + tmp_10996[2]];
    signal tmp_10998[3] <== CMul()(challengeQ, tmp_10997);
    signal tmp_10999[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11000[3] <== [tmp_10999[0] * 22, tmp_10999[1] * 22, tmp_10999[2] * 22];
    signal tmp_11001[3] <== [tmp_10915[0] + tmp_11000[0], tmp_10915[1] + tmp_11000[1], tmp_10915[2] + tmp_11000[2]];
    signal tmp_11002[3] <== [tmp_11001[0] * 1, tmp_11001[1] * 1, tmp_11001[2] * 1];
    signal tmp_11003[3] <== [evals[121][0] - tmp_11002[0], evals[121][1] - tmp_11002[1], evals[121][2] - tmp_11002[2]];
    signal tmp_11004[3] <== CMul()(tmp_6068, tmp_11003);
    signal tmp_11005[3] <== [tmp_10998[0] + tmp_11004[0], tmp_10998[1] + tmp_11004[1], tmp_10998[2] + tmp_11004[2]];
    signal tmp_11006[3] <== CMul()(challengeQ, tmp_11005);
    signal tmp_11007[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11008[3] <== [tmp_11007[0] * 67, tmp_11007[1] * 67, tmp_11007[2] * 67];
    signal tmp_11009[3] <== [tmp_10920[0] + tmp_11008[0], tmp_10920[1] + tmp_11008[1], tmp_10920[2] + tmp_11008[2]];
    signal tmp_11010[3] <== [tmp_11009[0] * 1, tmp_11009[1] * 1, tmp_11009[2] * 1];
    signal tmp_11011[3] <== [evals[122][0] - tmp_11010[0], evals[122][1] - tmp_11010[1], evals[122][2] - tmp_11010[2]];
    signal tmp_11012[3] <== CMul()(tmp_6068, tmp_11011);
    signal tmp_11013[3] <== [tmp_11006[0] + tmp_11012[0], tmp_11006[1] + tmp_11012[1], tmp_11006[2] + tmp_11012[2]];
    signal tmp_11014[3] <== CMul()(challengeQ, tmp_11013);
    signal tmp_11015[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11016[3] <== [tmp_11015[0] * 2, tmp_11015[1] * 2, tmp_11015[2] * 2];
    signal tmp_11017[3] <== [tmp_10925[0] + tmp_11016[0], tmp_10925[1] + tmp_11016[1], tmp_10925[2] + tmp_11016[2]];
    signal tmp_11018[3] <== [tmp_11017[0] * 1, tmp_11017[1] * 1, tmp_11017[2] * 1];
    signal tmp_11019[3] <== [evals[123][0] - tmp_11018[0], evals[123][1] - tmp_11018[1], evals[123][2] - tmp_11018[2]];
    signal tmp_11020[3] <== CMul()(tmp_6068, tmp_11019);
    signal tmp_11021[3] <== [tmp_11014[0] + tmp_11020[0], tmp_11014[1] + tmp_11020[1], tmp_11014[2] + tmp_11020[2]];
    signal tmp_11022[3] <== CMul()(challengeQ, tmp_11021);
    signal tmp_11023[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11024[3] <== [tmp_11023[0] * 15, tmp_11023[1] * 15, tmp_11023[2] * 15];
    signal tmp_11025[3] <== [tmp_10930[0] + tmp_11024[0], tmp_10930[1] + tmp_11024[1], tmp_10930[2] + tmp_11024[2]];
    signal tmp_11026[3] <== [tmp_11025[0] * 1, tmp_11025[1] * 1, tmp_11025[2] * 1];
    signal tmp_11027[3] <== [evals[124][0] - tmp_11026[0], evals[124][1] - tmp_11026[1], evals[124][2] - tmp_11026[2]];
    signal tmp_11028[3] <== CMul()(tmp_6068, tmp_11027);
    signal tmp_11029[3] <== [tmp_11022[0] + tmp_11028[0], tmp_11022[1] + tmp_11028[1], tmp_11022[2] + tmp_11028[2]];
    signal tmp_11030[3] <== CMul()(challengeQ, tmp_11029);
    signal tmp_11031[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11032[3] <== [tmp_11031[0] * 63, tmp_11031[1] * 63, tmp_11031[2] * 63];
    signal tmp_11033[3] <== [tmp_10935[0] + tmp_11032[0], tmp_10935[1] + tmp_11032[1], tmp_10935[2] + tmp_11032[2]];
    signal tmp_11034[3] <== [tmp_11033[0] * 1, tmp_11033[1] * 1, tmp_11033[2] * 1];
    signal tmp_11035[3] <== [evals[125][0] - tmp_11034[0], evals[125][1] - tmp_11034[1], evals[125][2] - tmp_11034[2]];
    signal tmp_11036[3] <== CMul()(tmp_6068, tmp_11035);
    signal tmp_11037[3] <== [tmp_11030[0] + tmp_11036[0], tmp_11030[1] + tmp_11036[1], tmp_11030[2] + tmp_11036[2]];
    signal tmp_11038[3] <== CMul()(challengeQ, tmp_11037);
    signal tmp_11039[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11040[3] <== [tmp_11039[0] * 101, tmp_11039[1] * 101, tmp_11039[2] * 101];
    signal tmp_11041[3] <== [tmp_10940[0] + tmp_11040[0], tmp_10940[1] + tmp_11040[1], tmp_10940[2] + tmp_11040[2]];
    signal tmp_11042[3] <== [tmp_11041[0] * 1, tmp_11041[1] * 1, tmp_11041[2] * 1];
    signal tmp_11043[3] <== [evals[126][0] - tmp_11042[0], evals[126][1] - tmp_11042[1], evals[126][2] - tmp_11042[2]];
    signal tmp_11044[3] <== CMul()(tmp_6068, tmp_11043);
    signal tmp_11045[3] <== [tmp_11038[0] + tmp_11044[0], tmp_11038[1] + tmp_11044[1], tmp_11038[2] + tmp_11044[2]];
    signal tmp_11046[3] <== CMul()(challengeQ, tmp_11045);
    signal tmp_11047[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11048[3] <== [tmp_10945[0] + tmp_11047[0], tmp_10945[1] + tmp_11047[1], tmp_10945[2] + tmp_11047[2]];
    signal tmp_11049[3] <== [tmp_11048[0] * 1, tmp_11048[1] * 1, tmp_11048[2] * 1];
    signal tmp_11050[3] <== [evals[127][0] - tmp_11049[0], evals[127][1] - tmp_11049[1], evals[127][2] - tmp_11049[2]];
    signal tmp_11051[3] <== CMul()(tmp_6068, tmp_11050);
    signal tmp_11052[3] <== [tmp_11046[0] + tmp_11051[0], tmp_11046[1] + tmp_11051[1], tmp_11046[2] + tmp_11051[2]];
    signal tmp_11053[3] <== CMul()(challengeQ, tmp_11052);
    signal tmp_11054[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11055[3] <== [tmp_11054[0] * 2, tmp_11054[1] * 2, tmp_11054[2] * 2];
    signal tmp_11056[3] <== [tmp_10950[0] + tmp_11055[0], tmp_10950[1] + tmp_11055[1], tmp_10950[2] + tmp_11055[2]];
    signal tmp_11057[3] <== [tmp_11056[0] * 1, tmp_11056[1] * 1, tmp_11056[2] * 1];
    signal tmp_11058[3] <== [evals[128][0] - tmp_11057[0], evals[128][1] - tmp_11057[1], evals[128][2] - tmp_11057[2]];
    signal tmp_11059[3] <== CMul()(tmp_6068, tmp_11058);
    tmp_11060 <== [tmp_11053[0] + tmp_11059[0], tmp_11053[1] + tmp_11059[1], tmp_11053[2] + tmp_11059[2]];
}

template VerifyEvaluationsChunks5() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_6068[3];
    signal input tmp_10901[3];
    signal input tmp_10955[3];
    signal input tmp_10960[3];
    signal input tmp_10965[3];
    signal input tmp_10970[3];
    signal input tmp_10975[3];
    signal input tmp_11060[3];

    signal output tmp_11955[3];
    signal output tmp_12018[3];
    signal output tmp_12058[3];
    signal output tmp_12059[3];
    signal tmp_11061[3] <== CMul()(challengeQ, tmp_11060);
    signal tmp_11062[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11063[3] <== [tmp_11062[0] * 17, tmp_11062[1] * 17, tmp_11062[2] * 17];
    signal tmp_11064[3] <== [tmp_10955[0] + tmp_11063[0], tmp_10955[1] + tmp_11063[1], tmp_10955[2] + tmp_11063[2]];
    signal tmp_11065[3] <== [tmp_11064[0] * 1, tmp_11064[1] * 1, tmp_11064[2] * 1];
    signal tmp_11066[3] <== [evals[129][0] - tmp_11065[0], evals[129][1] - tmp_11065[1], evals[129][2] - tmp_11065[2]];
    signal tmp_11067[3] <== CMul()(tmp_6068, tmp_11066);
    signal tmp_11068[3] <== [tmp_11061[0] + tmp_11067[0], tmp_11061[1] + tmp_11067[1], tmp_11061[2] + tmp_11067[2]];
    signal tmp_11069[3] <== CMul()(challengeQ, tmp_11068);
    signal tmp_11070[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11071[3] <== [tmp_11070[0] * 11, tmp_11070[1] * 11, tmp_11070[2] * 11];
    signal tmp_11072[3] <== [tmp_10960[0] + tmp_11071[0], tmp_10960[1] + tmp_11071[1], tmp_10960[2] + tmp_11071[2]];
    signal tmp_11073[3] <== [tmp_11072[0] * 1, tmp_11072[1] * 1, tmp_11072[2] * 1];
    signal tmp_11074[3] <== [evals[130][0] - tmp_11073[0], evals[130][1] - tmp_11073[1], evals[130][2] - tmp_11073[2]];
    signal tmp_11075[3] <== CMul()(tmp_6068, tmp_11074);
    signal tmp_11076[3] <== [tmp_11069[0] + tmp_11075[0], tmp_11069[1] + tmp_11075[1], tmp_11069[2] + tmp_11075[2]];
    signal tmp_11077[3] <== CMul()(challengeQ, tmp_11076);
    signal tmp_11078[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11079[3] <== [tmp_10965[0] + tmp_11078[0], tmp_10965[1] + tmp_11078[1], tmp_10965[2] + tmp_11078[2]];
    signal tmp_11080[3] <== [tmp_11079[0] * 1, tmp_11079[1] * 1, tmp_11079[2] * 1];
    signal tmp_11081[3] <== [evals[131][0] - tmp_11080[0], evals[131][1] - tmp_11080[1], evals[131][2] - tmp_11080[2]];
    signal tmp_11082[3] <== CMul()(tmp_6068, tmp_11081);
    signal tmp_11083[3] <== [tmp_11077[0] + tmp_11082[0], tmp_11077[1] + tmp_11082[1], tmp_11077[2] + tmp_11082[2]];
    signal tmp_11084[3] <== CMul()(challengeQ, tmp_11083);
    signal tmp_11085[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11086[3] <== [tmp_11085[0] * 51, tmp_11085[1] * 51, tmp_11085[2] * 51];
    signal tmp_11087[3] <== [tmp_10970[0] + tmp_11086[0], tmp_10970[1] + tmp_11086[1], tmp_10970[2] + tmp_11086[2]];
    signal tmp_11088[3] <== [tmp_11087[0] * 1, tmp_11087[1] * 1, tmp_11087[2] * 1];
    signal tmp_11089[3] <== [evals[132][0] - tmp_11088[0], evals[132][1] - tmp_11088[1], evals[132][2] - tmp_11088[2]];
    signal tmp_11090[3] <== CMul()(tmp_6068, tmp_11089);
    signal tmp_11091[3] <== [tmp_11084[0] + tmp_11090[0], tmp_11084[1] + tmp_11090[1], tmp_11084[2] + tmp_11090[2]];
    signal tmp_11092[3] <== CMul()(challengeQ, tmp_11091);
    signal tmp_11093[3] <== [tmp_10901[0] + 9358535959712383718, tmp_10901[1], tmp_10901[2]];
    signal tmp_11094[3] <== [tmp_10975[0] + tmp_11093[0], tmp_10975[1] + tmp_11093[1], tmp_10975[2] + tmp_11093[2]];
    signal tmp_11095[3] <== [tmp_11094[0] * 1, tmp_11094[1] * 1, tmp_11094[2] * 1];
    signal tmp_11096[3] <== [evals[133][0] - tmp_11095[0], evals[133][1] - tmp_11095[1], evals[133][2] - tmp_11095[2]];
    signal tmp_11097[3] <== CMul()(tmp_6068, tmp_11096);
    signal tmp_11098[3] <== [tmp_11092[0] + tmp_11097[0], tmp_11092[1] + tmp_11097[1], tmp_11092[2] + tmp_11097[2]];
    signal tmp_11099[3] <== CMul()(challengeQ, tmp_11098);
    signal tmp_11100[3] <== CMul()(evals[50], evals[53]);
    signal tmp_11101[3] <== CMul()(evals[51], evals[55]);
    signal tmp_11102[3] <== [tmp_11100[0] + tmp_11101[0], tmp_11100[1] + tmp_11101[1], tmp_11100[2] + tmp_11101[2]];
    signal tmp_11103[3] <== CMul()(evals[52], evals[54]);
    signal tmp_11104[3] <== [tmp_11102[0] + tmp_11103[0], tmp_11102[1] + tmp_11103[1], tmp_11102[2] + tmp_11103[2]];
    signal tmp_11105[3] <== [evals[56][0] - tmp_11104[0], evals[56][1] - tmp_11104[1], evals[56][2] - tmp_11104[2]];
    signal tmp_11106[3] <== CMul()(evals[42], tmp_11105);
    signal tmp_11107[3] <== [tmp_11099[0] + tmp_11106[0], tmp_11099[1] + tmp_11106[1], tmp_11099[2] + tmp_11106[2]];
    signal tmp_11108[3] <== CMul()(challengeQ, tmp_11107);
    signal tmp_11109[3] <== CMul()(evals[50], evals[54]);
    signal tmp_11110[3] <== CMul()(evals[51], evals[53]);
    signal tmp_11111[3] <== [tmp_11109[0] + tmp_11110[0], tmp_11109[1] + tmp_11110[1], tmp_11109[2] + tmp_11110[2]];
    signal tmp_11112[3] <== CMul()(evals[51], evals[55]);
    signal tmp_11113[3] <== [tmp_11111[0] + tmp_11112[0], tmp_11111[1] + tmp_11112[1], tmp_11111[2] + tmp_11112[2]];
    signal tmp_11114[3] <== CMul()(evals[52], evals[54]);
    signal tmp_11115[3] <== [tmp_11113[0] + tmp_11114[0], tmp_11113[1] + tmp_11114[1], tmp_11113[2] + tmp_11114[2]];
    signal tmp_11116[3] <== CMul()(evals[52], evals[55]);
    signal tmp_11117[3] <== [tmp_11115[0] + tmp_11116[0], tmp_11115[1] + tmp_11116[1], tmp_11115[2] + tmp_11116[2]];
    signal tmp_11118[3] <== [evals[57][0] - tmp_11117[0], evals[57][1] - tmp_11117[1], evals[57][2] - tmp_11117[2]];
    signal tmp_11119[3] <== CMul()(evals[42], tmp_11118);
    signal tmp_11120[3] <== [tmp_11108[0] + tmp_11119[0], tmp_11108[1] + tmp_11119[1], tmp_11108[2] + tmp_11119[2]];
    signal tmp_11121[3] <== CMul()(challengeQ, tmp_11120);
    signal tmp_11122[3] <== CMul()(evals[50], evals[55]);
    signal tmp_11123[3] <== CMul()(evals[52], evals[55]);
    signal tmp_11124[3] <== [tmp_11122[0] + tmp_11123[0], tmp_11122[1] + tmp_11123[1], tmp_11122[2] + tmp_11123[2]];
    signal tmp_11125[3] <== CMul()(evals[52], evals[53]);
    signal tmp_11126[3] <== [tmp_11124[0] + tmp_11125[0], tmp_11124[1] + tmp_11125[1], tmp_11124[2] + tmp_11125[2]];
    signal tmp_11127[3] <== CMul()(evals[51], evals[54]);
    signal tmp_11128[3] <== [tmp_11126[0] + tmp_11127[0], tmp_11126[1] + tmp_11127[1], tmp_11126[2] + tmp_11127[2]];
    signal tmp_11129[3] <== [evals[58][0] - tmp_11128[0], evals[58][1] - tmp_11128[1], evals[58][2] - tmp_11128[2]];
    signal tmp_11130[3] <== CMul()(evals[42], tmp_11129);
    signal tmp_11131[3] <== [tmp_11121[0] + tmp_11130[0], tmp_11121[1] + tmp_11130[1], tmp_11121[2] + tmp_11130[2]];
    signal tmp_11132[3] <== CMul()(challengeQ, tmp_11131);
    signal tmp_11133[3] <== CMul()(evals[59], evals[62]);
    signal tmp_11134[3] <== CMul()(evals[60], evals[64]);
    signal tmp_11135[3] <== [tmp_11133[0] + tmp_11134[0], tmp_11133[1] + tmp_11134[1], tmp_11133[2] + tmp_11134[2]];
    signal tmp_11136[3] <== CMul()(evals[61], evals[63]);
    signal tmp_11137[3] <== [tmp_11135[0] + tmp_11136[0], tmp_11135[1] + tmp_11136[1], tmp_11135[2] + tmp_11136[2]];
    signal tmp_11138[3] <== [evals[65][0] - tmp_11137[0], evals[65][1] - tmp_11137[1], evals[65][2] - tmp_11137[2]];
    signal tmp_11139[3] <== CMul()(evals[42], tmp_11138);
    signal tmp_11140[3] <== [tmp_11132[0] + tmp_11139[0], tmp_11132[1] + tmp_11139[1], tmp_11132[2] + tmp_11139[2]];
    signal tmp_11141[3] <== CMul()(challengeQ, tmp_11140);
    signal tmp_11142[3] <== CMul()(evals[59], evals[63]);
    signal tmp_11143[3] <== CMul()(evals[60], evals[62]);
    signal tmp_11144[3] <== [tmp_11142[0] + tmp_11143[0], tmp_11142[1] + tmp_11143[1], tmp_11142[2] + tmp_11143[2]];
    signal tmp_11145[3] <== CMul()(evals[60], evals[64]);
    signal tmp_11146[3] <== [tmp_11144[0] + tmp_11145[0], tmp_11144[1] + tmp_11145[1], tmp_11144[2] + tmp_11145[2]];
    signal tmp_11147[3] <== CMul()(evals[61], evals[63]);
    signal tmp_11148[3] <== [tmp_11146[0] + tmp_11147[0], tmp_11146[1] + tmp_11147[1], tmp_11146[2] + tmp_11147[2]];
    signal tmp_11149[3] <== CMul()(evals[61], evals[64]);
    signal tmp_11150[3] <== [tmp_11148[0] + tmp_11149[0], tmp_11148[1] + tmp_11149[1], tmp_11148[2] + tmp_11149[2]];
    signal tmp_11151[3] <== [evals[66][0] - tmp_11150[0], evals[66][1] - tmp_11150[1], evals[66][2] - tmp_11150[2]];
    signal tmp_11152[3] <== CMul()(evals[42], tmp_11151);
    signal tmp_11153[3] <== [tmp_11141[0] + tmp_11152[0], tmp_11141[1] + tmp_11152[1], tmp_11141[2] + tmp_11152[2]];
    signal tmp_11154[3] <== CMul()(challengeQ, tmp_11153);
    signal tmp_11155[3] <== CMul()(evals[59], evals[64]);
    signal tmp_11156[3] <== CMul()(evals[61], evals[64]);
    signal tmp_11157[3] <== [tmp_11155[0] + tmp_11156[0], tmp_11155[1] + tmp_11156[1], tmp_11155[2] + tmp_11156[2]];
    signal tmp_11158[3] <== CMul()(evals[61], evals[62]);
    signal tmp_11159[3] <== [tmp_11157[0] + tmp_11158[0], tmp_11157[1] + tmp_11158[1], tmp_11157[2] + tmp_11158[2]];
    signal tmp_11160[3] <== CMul()(evals[60], evals[63]);
    signal tmp_11161[3] <== [tmp_11159[0] + tmp_11160[0], tmp_11159[1] + tmp_11160[1], tmp_11159[2] + tmp_11160[2]];
    signal tmp_11162[3] <== [evals[67][0] - tmp_11161[0], evals[67][1] - tmp_11161[1], evals[67][2] - tmp_11161[2]];
    signal tmp_11163[3] <== CMul()(evals[42], tmp_11162);
    signal tmp_11164[3] <== [tmp_11154[0] + tmp_11163[0], tmp_11154[1] + tmp_11163[1], tmp_11154[2] + tmp_11163[2]];
    signal tmp_11165[3] <== CMul()(challengeQ, tmp_11164);
    signal tmp_11166[3] <== CMul()(evals[29], evals[50]);
    signal tmp_11167[3] <== CMul()(evals[30], evals[53]);
    signal tmp_11168[3] <== [tmp_11166[0] + tmp_11167[0], tmp_11166[1] + tmp_11167[1], tmp_11166[2] + tmp_11167[2]];
    signal tmp_11169[3] <== CMul()(evals[31], evals[56]);
    signal tmp_11170[3] <== [tmp_11168[0] + tmp_11169[0], tmp_11168[1] + tmp_11169[1], tmp_11168[2] + tmp_11169[2]];
    signal tmp_11171[3] <== CMul()(evals[32], evals[59]);
    signal tmp_11172[3] <== [tmp_11170[0] + tmp_11171[0], tmp_11170[1] + tmp_11171[1], tmp_11170[2] + tmp_11171[2]];
    signal tmp_11173[3] <== CMul()(evals[35], evals[50]);
    signal tmp_11174[3] <== [tmp_11172[0] + tmp_11173[0], tmp_11172[1] + tmp_11173[1], tmp_11172[2] + tmp_11173[2]];
    signal tmp_11175[3] <== CMul()(evals[36], evals[53]);
    signal tmp_11176[3] <== [tmp_11174[0] + tmp_11175[0], tmp_11174[1] + tmp_11175[1], tmp_11174[2] + tmp_11175[2]];
    signal tmp_11177[3] <== [evals[62][0] - tmp_11176[0], evals[62][1] - tmp_11176[1], evals[62][2] - tmp_11176[2]];
    signal tmp_11178[3] <== CMul()(evals[44], tmp_11177);
    signal tmp_11179[3] <== [tmp_11165[0] + tmp_11178[0], tmp_11165[1] + tmp_11178[1], tmp_11165[2] + tmp_11178[2]];
    signal tmp_11180[3] <== CMul()(challengeQ, tmp_11179);
    signal tmp_11181[3] <== CMul()(evals[29], evals[51]);
    signal tmp_11182[3] <== CMul()(evals[30], evals[54]);
    signal tmp_11183[3] <== [tmp_11181[0] + tmp_11182[0], tmp_11181[1] + tmp_11182[1], tmp_11181[2] + tmp_11182[2]];
    signal tmp_11184[3] <== CMul()(evals[31], evals[57]);
    signal tmp_11185[3] <== [tmp_11183[0] + tmp_11184[0], tmp_11183[1] + tmp_11184[1], tmp_11183[2] + tmp_11184[2]];
    signal tmp_11186[3] <== CMul()(evals[32], evals[60]);
    signal tmp_11187[3] <== [tmp_11185[0] + tmp_11186[0], tmp_11185[1] + tmp_11186[1], tmp_11185[2] + tmp_11186[2]];
    signal tmp_11188[3] <== CMul()(evals[35], evals[51]);
    signal tmp_11189[3] <== [tmp_11187[0] + tmp_11188[0], tmp_11187[1] + tmp_11188[1], tmp_11187[2] + tmp_11188[2]];
    signal tmp_11190[3] <== CMul()(evals[36], evals[54]);
    signal tmp_11191[3] <== [tmp_11189[0] + tmp_11190[0], tmp_11189[1] + tmp_11190[1], tmp_11189[2] + tmp_11190[2]];
    signal tmp_11192[3] <== [evals[63][0] - tmp_11191[0], evals[63][1] - tmp_11191[1], evals[63][2] - tmp_11191[2]];
    signal tmp_11193[3] <== CMul()(evals[44], tmp_11192);
    signal tmp_11194[3] <== [tmp_11180[0] + tmp_11193[0], tmp_11180[1] + tmp_11193[1], tmp_11180[2] + tmp_11193[2]];
    signal tmp_11195[3] <== CMul()(challengeQ, tmp_11194);
    signal tmp_11196[3] <== CMul()(evals[29], evals[52]);
    signal tmp_11197[3] <== CMul()(evals[30], evals[55]);
    signal tmp_11198[3] <== [tmp_11196[0] + tmp_11197[0], tmp_11196[1] + tmp_11197[1], tmp_11196[2] + tmp_11197[2]];
    signal tmp_11199[3] <== CMul()(evals[31], evals[58]);
    signal tmp_11200[3] <== [tmp_11198[0] + tmp_11199[0], tmp_11198[1] + tmp_11199[1], tmp_11198[2] + tmp_11199[2]];
    signal tmp_11201[3] <== CMul()(evals[32], evals[61]);
    signal tmp_11202[3] <== [tmp_11200[0] + tmp_11201[0], tmp_11200[1] + tmp_11201[1], tmp_11200[2] + tmp_11201[2]];
    signal tmp_11203[3] <== CMul()(evals[35], evals[52]);
    signal tmp_11204[3] <== [tmp_11202[0] + tmp_11203[0], tmp_11202[1] + tmp_11203[1], tmp_11202[2] + tmp_11203[2]];
    signal tmp_11205[3] <== CMul()(evals[36], evals[55]);
    signal tmp_11206[3] <== [tmp_11204[0] + tmp_11205[0], tmp_11204[1] + tmp_11205[1], tmp_11204[2] + tmp_11205[2]];
    signal tmp_11207[3] <== [evals[64][0] - tmp_11206[0], evals[64][1] - tmp_11206[1], evals[64][2] - tmp_11206[2]];
    signal tmp_11208[3] <== CMul()(evals[44], tmp_11207);
    signal tmp_11209[3] <== [tmp_11195[0] + tmp_11208[0], tmp_11195[1] + tmp_11208[1], tmp_11195[2] + tmp_11208[2]];
    signal tmp_11210[3] <== CMul()(challengeQ, tmp_11209);
    signal tmp_11211[3] <== CMul()(evals[29], evals[50]);
    signal tmp_11212[3] <== CMul()(evals[30], evals[53]);
    signal tmp_11213[3] <== [tmp_11211[0] - tmp_11212[0], tmp_11211[1] - tmp_11212[1], tmp_11211[2] - tmp_11212[2]];
    signal tmp_11214[3] <== CMul()(evals[33], evals[56]);
    signal tmp_11215[3] <== [tmp_11213[0] + tmp_11214[0], tmp_11213[1] + tmp_11214[1], tmp_11213[2] + tmp_11214[2]];
    signal tmp_11216[3] <== CMul()(evals[34], evals[59]);
    signal tmp_11217[3] <== [tmp_11215[0] - tmp_11216[0], tmp_11215[1] - tmp_11216[1], tmp_11215[2] - tmp_11216[2]];
    signal tmp_11218[3] <== CMul()(evals[35], evals[50]);
    signal tmp_11219[3] <== [tmp_11217[0] + tmp_11218[0], tmp_11217[1] + tmp_11218[1], tmp_11217[2] + tmp_11218[2]];
    signal tmp_11220[3] <== CMul()(evals[36], evals[53]);
    signal tmp_11221[3] <== [tmp_11219[0] - tmp_11220[0], tmp_11219[1] - tmp_11220[1], tmp_11219[2] - tmp_11220[2]];
    signal tmp_11222[3] <== [evals[65][0] - tmp_11221[0], evals[65][1] - tmp_11221[1], evals[65][2] - tmp_11221[2]];
    signal tmp_11223[3] <== CMul()(evals[44], tmp_11222);
    signal tmp_11224[3] <== [tmp_11210[0] + tmp_11223[0], tmp_11210[1] + tmp_11223[1], tmp_11210[2] + tmp_11223[2]];
    signal tmp_11225[3] <== CMul()(challengeQ, tmp_11224);
    signal tmp_11226[3] <== CMul()(evals[29], evals[51]);
    signal tmp_11227[3] <== CMul()(evals[30], evals[54]);
    signal tmp_11228[3] <== [tmp_11226[0] - tmp_11227[0], tmp_11226[1] - tmp_11227[1], tmp_11226[2] - tmp_11227[2]];
    signal tmp_11229[3] <== CMul()(evals[33], evals[57]);
    signal tmp_11230[3] <== [tmp_11228[0] + tmp_11229[0], tmp_11228[1] + tmp_11229[1], tmp_11228[2] + tmp_11229[2]];
    signal tmp_11231[3] <== CMul()(evals[34], evals[60]);
    signal tmp_11232[3] <== [tmp_11230[0] - tmp_11231[0], tmp_11230[1] - tmp_11231[1], tmp_11230[2] - tmp_11231[2]];
    signal tmp_11233[3] <== CMul()(evals[35], evals[51]);
    signal tmp_11234[3] <== [tmp_11232[0] + tmp_11233[0], tmp_11232[1] + tmp_11233[1], tmp_11232[2] + tmp_11233[2]];
    signal tmp_11235[3] <== CMul()(evals[36], evals[54]);
    signal tmp_11236[3] <== [tmp_11234[0] - tmp_11235[0], tmp_11234[1] - tmp_11235[1], tmp_11234[2] - tmp_11235[2]];
    signal tmp_11237[3] <== [evals[66][0] - tmp_11236[0], evals[66][1] - tmp_11236[1], evals[66][2] - tmp_11236[2]];
    signal tmp_11238[3] <== CMul()(evals[44], tmp_11237);
    signal tmp_11239[3] <== [tmp_11225[0] + tmp_11238[0], tmp_11225[1] + tmp_11238[1], tmp_11225[2] + tmp_11238[2]];
    signal tmp_11240[3] <== CMul()(challengeQ, tmp_11239);
    signal tmp_11241[3] <== CMul()(evals[29], evals[52]);
    signal tmp_11242[3] <== CMul()(evals[30], evals[55]);
    signal tmp_11243[3] <== [tmp_11241[0] - tmp_11242[0], tmp_11241[1] - tmp_11242[1], tmp_11241[2] - tmp_11242[2]];
    signal tmp_11244[3] <== CMul()(evals[33], evals[58]);
    signal tmp_11245[3] <== [tmp_11243[0] + tmp_11244[0], tmp_11243[1] + tmp_11244[1], tmp_11243[2] + tmp_11244[2]];
    signal tmp_11246[3] <== CMul()(evals[34], evals[61]);
    signal tmp_11247[3] <== [tmp_11245[0] - tmp_11246[0], tmp_11245[1] - tmp_11246[1], tmp_11245[2] - tmp_11246[2]];
    signal tmp_11248[3] <== CMul()(evals[35], evals[52]);
    signal tmp_11249[3] <== [tmp_11247[0] + tmp_11248[0], tmp_11247[1] + tmp_11248[1], tmp_11247[2] + tmp_11248[2]];
    signal tmp_11250[3] <== CMul()(evals[36], evals[55]);
    signal tmp_11251[3] <== [tmp_11249[0] - tmp_11250[0], tmp_11249[1] - tmp_11250[1], tmp_11249[2] - tmp_11250[2]];
    signal tmp_11252[3] <== [evals[67][0] - tmp_11251[0], evals[67][1] - tmp_11251[1], evals[67][2] - tmp_11251[2]];
    signal tmp_11253[3] <== CMul()(evals[44], tmp_11252);
    signal tmp_11254[3] <== [tmp_11240[0] + tmp_11253[0], tmp_11240[1] + tmp_11253[1], tmp_11240[2] + tmp_11253[2]];
    signal tmp_11255[3] <== CMul()(challengeQ, tmp_11254);
    signal tmp_11256[3] <== CMul()(evals[29], evals[50]);
    signal tmp_11257[3] <== CMul()(evals[30], evals[53]);
    signal tmp_11258[3] <== [tmp_11256[0] + tmp_11257[0], tmp_11256[1] + tmp_11257[1], tmp_11256[2] + tmp_11257[2]];
    signal tmp_11259[3] <== CMul()(evals[31], evals[56]);
    signal tmp_11260[3] <== [tmp_11258[0] - tmp_11259[0], tmp_11258[1] - tmp_11259[1], tmp_11258[2] - tmp_11259[2]];
    signal tmp_11261[3] <== CMul()(evals[32], evals[59]);
    signal tmp_11262[3] <== [tmp_11260[0] - tmp_11261[0], tmp_11260[1] - tmp_11261[1], tmp_11260[2] - tmp_11261[2]];
    signal tmp_11263[3] <== CMul()(evals[35], evals[56]);
    signal tmp_11264[3] <== [tmp_11262[0] + tmp_11263[0], tmp_11262[1] + tmp_11263[1], tmp_11262[2] + tmp_11263[2]];
    signal tmp_11265[3] <== CMul()(evals[37], evals[59]);
    signal tmp_11266[3] <== [tmp_11264[0] + tmp_11265[0], tmp_11264[1] + tmp_11265[1], tmp_11264[2] + tmp_11265[2]];
    signal tmp_11267[3] <== [evals[68][0] - tmp_11266[0], evals[68][1] - tmp_11266[1], evals[68][2] - tmp_11266[2]];
    signal tmp_11268[3] <== CMul()(evals[44], tmp_11267);
    signal tmp_11269[3] <== [tmp_11255[0] + tmp_11268[0], tmp_11255[1] + tmp_11268[1], tmp_11255[2] + tmp_11268[2]];
    signal tmp_11270[3] <== CMul()(challengeQ, tmp_11269);
    signal tmp_11271[3] <== CMul()(evals[29], evals[51]);
    signal tmp_11272[3] <== CMul()(evals[30], evals[54]);
    signal tmp_11273[3] <== [tmp_11271[0] + tmp_11272[0], tmp_11271[1] + tmp_11272[1], tmp_11271[2] + tmp_11272[2]];
    signal tmp_11274[3] <== CMul()(evals[31], evals[57]);
    signal tmp_11275[3] <== [tmp_11273[0] - tmp_11274[0], tmp_11273[1] - tmp_11274[1], tmp_11273[2] - tmp_11274[2]];
    signal tmp_11276[3] <== CMul()(evals[32], evals[60]);
    signal tmp_11277[3] <== [tmp_11275[0] - tmp_11276[0], tmp_11275[1] - tmp_11276[1], tmp_11275[2] - tmp_11276[2]];
    signal tmp_11278[3] <== CMul()(evals[35], evals[57]);
    signal tmp_11279[3] <== [tmp_11277[0] + tmp_11278[0], tmp_11277[1] + tmp_11278[1], tmp_11277[2] + tmp_11278[2]];
    signal tmp_11280[3] <== CMul()(evals[37], evals[60]);
    signal tmp_11281[3] <== [tmp_11279[0] + tmp_11280[0], tmp_11279[1] + tmp_11280[1], tmp_11279[2] + tmp_11280[2]];
    signal tmp_11282[3] <== [evals[69][0] - tmp_11281[0], evals[69][1] - tmp_11281[1], evals[69][2] - tmp_11281[2]];
    signal tmp_11283[3] <== CMul()(evals[44], tmp_11282);
    signal tmp_11284[3] <== [tmp_11270[0] + tmp_11283[0], tmp_11270[1] + tmp_11283[1], tmp_11270[2] + tmp_11283[2]];
    signal tmp_11285[3] <== CMul()(challengeQ, tmp_11284);
    signal tmp_11286[3] <== CMul()(evals[29], evals[52]);
    signal tmp_11287[3] <== CMul()(evals[30], evals[55]);
    signal tmp_11288[3] <== [tmp_11286[0] + tmp_11287[0], tmp_11286[1] + tmp_11287[1], tmp_11286[2] + tmp_11287[2]];
    signal tmp_11289[3] <== CMul()(evals[31], evals[58]);
    signal tmp_11290[3] <== [tmp_11288[0] - tmp_11289[0], tmp_11288[1] - tmp_11289[1], tmp_11288[2] - tmp_11289[2]];
    signal tmp_11291[3] <== CMul()(evals[32], evals[61]);
    signal tmp_11292[3] <== [tmp_11290[0] - tmp_11291[0], tmp_11290[1] - tmp_11291[1], tmp_11290[2] - tmp_11291[2]];
    signal tmp_11293[3] <== CMul()(evals[35], evals[58]);
    signal tmp_11294[3] <== [tmp_11292[0] + tmp_11293[0], tmp_11292[1] + tmp_11293[1], tmp_11292[2] + tmp_11293[2]];
    signal tmp_11295[3] <== CMul()(evals[37], evals[61]);
    signal tmp_11296[3] <== [tmp_11294[0] + tmp_11295[0], tmp_11294[1] + tmp_11295[1], tmp_11294[2] + tmp_11295[2]];
    signal tmp_11297[3] <== [evals[70][0] - tmp_11296[0], evals[70][1] - tmp_11296[1], evals[70][2] - tmp_11296[2]];
    signal tmp_11298[3] <== CMul()(evals[44], tmp_11297);
    signal tmp_11299[3] <== [tmp_11285[0] + tmp_11298[0], tmp_11285[1] + tmp_11298[1], tmp_11285[2] + tmp_11298[2]];
    signal tmp_11300[3] <== CMul()(challengeQ, tmp_11299);
    signal tmp_11301[3] <== CMul()(evals[29], evals[50]);
    signal tmp_11302[3] <== CMul()(evals[30], evals[53]);
    signal tmp_11303[3] <== [tmp_11301[0] - tmp_11302[0], tmp_11301[1] - tmp_11302[1], tmp_11301[2] - tmp_11302[2]];
    signal tmp_11304[3] <== CMul()(evals[33], evals[56]);
    signal tmp_11305[3] <== [tmp_11303[0] - tmp_11304[0], tmp_11303[1] - tmp_11304[1], tmp_11303[2] - tmp_11304[2]];
    signal tmp_11306[3] <== CMul()(evals[34], evals[59]);
    signal tmp_11307[3] <== [tmp_11305[0] + tmp_11306[0], tmp_11305[1] + tmp_11306[1], tmp_11305[2] + tmp_11306[2]];
    signal tmp_11308[3] <== CMul()(evals[35], evals[56]);
    signal tmp_11309[3] <== [tmp_11307[0] + tmp_11308[0], tmp_11307[1] + tmp_11308[1], tmp_11307[2] + tmp_11308[2]];
    signal tmp_11310[3] <== CMul()(evals[37], evals[59]);
    signal tmp_11311[3] <== [tmp_11309[0] - tmp_11310[0], tmp_11309[1] - tmp_11310[1], tmp_11309[2] - tmp_11310[2]];
    signal tmp_11312[3] <== [evals[71][0] - tmp_11311[0], evals[71][1] - tmp_11311[1], evals[71][2] - tmp_11311[2]];
    signal tmp_11313[3] <== CMul()(evals[44], tmp_11312);
    signal tmp_11314[3] <== [tmp_11300[0] + tmp_11313[0], tmp_11300[1] + tmp_11313[1], tmp_11300[2] + tmp_11313[2]];
    signal tmp_11315[3] <== CMul()(challengeQ, tmp_11314);
    signal tmp_11316[3] <== CMul()(evals[29], evals[51]);
    signal tmp_11317[3] <== CMul()(evals[30], evals[54]);
    signal tmp_11318[3] <== [tmp_11316[0] - tmp_11317[0], tmp_11316[1] - tmp_11317[1], tmp_11316[2] - tmp_11317[2]];
    signal tmp_11319[3] <== CMul()(evals[33], evals[57]);
    signal tmp_11320[3] <== [tmp_11318[0] - tmp_11319[0], tmp_11318[1] - tmp_11319[1], tmp_11318[2] - tmp_11319[2]];
    signal tmp_11321[3] <== CMul()(evals[34], evals[60]);
    signal tmp_11322[3] <== [tmp_11320[0] + tmp_11321[0], tmp_11320[1] + tmp_11321[1], tmp_11320[2] + tmp_11321[2]];
    signal tmp_11323[3] <== CMul()(evals[35], evals[57]);
    signal tmp_11324[3] <== [tmp_11322[0] + tmp_11323[0], tmp_11322[1] + tmp_11323[1], tmp_11322[2] + tmp_11323[2]];
    signal tmp_11325[3] <== CMul()(evals[37], evals[60]);
    signal tmp_11326[3] <== [tmp_11324[0] - tmp_11325[0], tmp_11324[1] - tmp_11325[1], tmp_11324[2] - tmp_11325[2]];
    signal tmp_11327[3] <== [evals[72][0] - tmp_11326[0], evals[72][1] - tmp_11326[1], evals[72][2] - tmp_11326[2]];
    signal tmp_11328[3] <== CMul()(evals[44], tmp_11327);
    signal tmp_11329[3] <== [tmp_11315[0] + tmp_11328[0], tmp_11315[1] + tmp_11328[1], tmp_11315[2] + tmp_11328[2]];
    signal tmp_11330[3] <== CMul()(challengeQ, tmp_11329);
    signal tmp_11331[3] <== CMul()(evals[29], evals[52]);
    signal tmp_11332[3] <== CMul()(evals[30], evals[55]);
    signal tmp_11333[3] <== [tmp_11331[0] - tmp_11332[0], tmp_11331[1] - tmp_11332[1], tmp_11331[2] - tmp_11332[2]];
    signal tmp_11334[3] <== CMul()(evals[33], evals[58]);
    signal tmp_11335[3] <== [tmp_11333[0] - tmp_11334[0], tmp_11333[1] - tmp_11334[1], tmp_11333[2] - tmp_11334[2]];
    signal tmp_11336[3] <== CMul()(evals[34], evals[61]);
    signal tmp_11337[3] <== [tmp_11335[0] + tmp_11336[0], tmp_11335[1] + tmp_11336[1], tmp_11335[2] + tmp_11336[2]];
    signal tmp_11338[3] <== CMul()(evals[35], evals[58]);
    signal tmp_11339[3] <== [tmp_11337[0] + tmp_11338[0], tmp_11337[1] + tmp_11338[1], tmp_11337[2] + tmp_11338[2]];
    signal tmp_11340[3] <== CMul()(evals[37], evals[61]);
    signal tmp_11341[3] <== [tmp_11339[0] - tmp_11340[0], tmp_11339[1] - tmp_11340[1], tmp_11339[2] - tmp_11340[2]];
    signal tmp_11342[3] <== [evals[73][0] - tmp_11341[0], evals[73][1] - tmp_11341[1], evals[73][2] - tmp_11341[2]];
    signal tmp_11343[3] <== CMul()(evals[44], tmp_11342);
    signal tmp_11344[3] <== [tmp_11330[0] + tmp_11343[0], tmp_11330[1] + tmp_11343[1], tmp_11330[2] + tmp_11343[2]];
    signal tmp_11345[3] <== CMul()(challengeQ, tmp_11344);
    signal tmp_11346[3] <== CMul()(evals[62], evals[65]);
    signal tmp_11347[3] <== CMul()(evals[63], evals[67]);
    signal tmp_11348[3] <== [tmp_11346[0] + tmp_11347[0], tmp_11346[1] + tmp_11347[1], tmp_11346[2] + tmp_11347[2]];
    signal tmp_11349[3] <== CMul()(evals[64], evals[66]);
    signal tmp_11350[3] <== [tmp_11348[0] + tmp_11349[0], tmp_11348[1] + tmp_11349[1], tmp_11348[2] + tmp_11349[2]];
    signal tmp_11351[3] <== [tmp_11350[0] + evals[59][0], tmp_11350[1] + evals[59][1], tmp_11350[2] + evals[59][2]];
    signal tmp_11352[3] <== CMul()(tmp_11351, evals[65]);
    signal tmp_11353[3] <== CMul()(evals[62], evals[66]);
    signal tmp_11354[3] <== CMul()(evals[63], evals[65]);
    signal tmp_11355[3] <== [tmp_11353[0] + tmp_11354[0], tmp_11353[1] + tmp_11354[1], tmp_11353[2] + tmp_11354[2]];
    signal tmp_11356[3] <== CMul()(evals[63], evals[67]);
    signal tmp_11357[3] <== [tmp_11355[0] + tmp_11356[0], tmp_11355[1] + tmp_11356[1], tmp_11355[2] + tmp_11356[2]];
    signal tmp_11358[3] <== CMul()(evals[64], evals[66]);
    signal tmp_11359[3] <== [tmp_11357[0] + tmp_11358[0], tmp_11357[1] + tmp_11358[1], tmp_11357[2] + tmp_11358[2]];
    signal tmp_11360[3] <== CMul()(evals[64], evals[67]);
    signal tmp_11361[3] <== [tmp_11359[0] + tmp_11360[0], tmp_11359[1] + tmp_11360[1], tmp_11359[2] + tmp_11360[2]];
    signal tmp_11362[3] <== [tmp_11361[0] + evals[60][0], tmp_11361[1] + evals[60][1], tmp_11361[2] + evals[60][2]];
    signal tmp_11363[3] <== CMul()(tmp_11362, evals[67]);
    signal tmp_11364[3] <== [tmp_11352[0] + tmp_11363[0], tmp_11352[1] + tmp_11363[1], tmp_11352[2] + tmp_11363[2]];
    signal tmp_11365[3] <== CMul()(evals[62], evals[67]);
    signal tmp_11366[3] <== CMul()(evals[64], evals[67]);
    signal tmp_11367[3] <== [tmp_11365[0] + tmp_11366[0], tmp_11365[1] + tmp_11366[1], tmp_11365[2] + tmp_11366[2]];
    signal tmp_11368[3] <== CMul()(evals[64], evals[65]);
    signal tmp_11369[3] <== [tmp_11367[0] + tmp_11368[0], tmp_11367[1] + tmp_11368[1], tmp_11367[2] + tmp_11368[2]];
    signal tmp_11370[3] <== CMul()(evals[63], evals[66]);
    signal tmp_11371[3] <== [tmp_11369[0] + tmp_11370[0], tmp_11369[1] + tmp_11370[1], tmp_11369[2] + tmp_11370[2]];
    signal tmp_11372[3] <== [tmp_11371[0] + evals[61][0], tmp_11371[1] + evals[61][1], tmp_11371[2] + evals[61][2]];
    signal tmp_11373[3] <== CMul()(tmp_11372, evals[66]);
    signal tmp_11374[3] <== [tmp_11364[0] + tmp_11373[0], tmp_11364[1] + tmp_11373[1], tmp_11364[2] + tmp_11373[2]];
    signal tmp_11375[3] <== [tmp_11374[0] + evals[56][0], tmp_11374[1] + evals[56][1], tmp_11374[2] + evals[56][2]];
    signal tmp_11376[3] <== CMul()(tmp_11375, evals[65]);
    signal tmp_11377[3] <== CMul()(tmp_11351, evals[66]);
    signal tmp_11378[3] <== CMul()(tmp_11362, evals[65]);
    signal tmp_11379[3] <== [tmp_11377[0] + tmp_11378[0], tmp_11377[1] + tmp_11378[1], tmp_11377[2] + tmp_11378[2]];
    signal tmp_11380[3] <== CMul()(tmp_11362, evals[67]);
    signal tmp_11381[3] <== [tmp_11379[0] + tmp_11380[0], tmp_11379[1] + tmp_11380[1], tmp_11379[2] + tmp_11380[2]];
    signal tmp_11382[3] <== CMul()(tmp_11372, evals[66]);
    signal tmp_11383[3] <== [tmp_11381[0] + tmp_11382[0], tmp_11381[1] + tmp_11382[1], tmp_11381[2] + tmp_11382[2]];
    signal tmp_11384[3] <== CMul()(tmp_11372, evals[67]);
    signal tmp_11385[3] <== [tmp_11383[0] + tmp_11384[0], tmp_11383[1] + tmp_11384[1], tmp_11383[2] + tmp_11384[2]];
    signal tmp_11386[3] <== [tmp_11385[0] + evals[57][0], tmp_11385[1] + evals[57][1], tmp_11385[2] + evals[57][2]];
    signal tmp_11387[3] <== CMul()(tmp_11386, evals[67]);
    signal tmp_11388[3] <== [tmp_11376[0] + tmp_11387[0], tmp_11376[1] + tmp_11387[1], tmp_11376[2] + tmp_11387[2]];
    signal tmp_11389[3] <== CMul()(tmp_11351, evals[67]);
    signal tmp_11390[3] <== CMul()(tmp_11372, evals[67]);
    signal tmp_11391[3] <== [tmp_11389[0] + tmp_11390[0], tmp_11389[1] + tmp_11390[1], tmp_11389[2] + tmp_11390[2]];
    signal tmp_11392[3] <== CMul()(tmp_11372, evals[65]);
    signal tmp_11393[3] <== [tmp_11391[0] + tmp_11392[0], tmp_11391[1] + tmp_11392[1], tmp_11391[2] + tmp_11392[2]];
    signal tmp_11394[3] <== CMul()(tmp_11362, evals[66]);
    signal tmp_11395[3] <== [tmp_11393[0] + tmp_11394[0], tmp_11393[1] + tmp_11394[1], tmp_11393[2] + tmp_11394[2]];
    signal tmp_11396[3] <== [tmp_11395[0] + evals[58][0], tmp_11395[1] + evals[58][1], tmp_11395[2] + evals[58][2]];
    signal tmp_11397[3] <== CMul()(tmp_11396, evals[66]);
    signal tmp_11398[3] <== [tmp_11388[0] + tmp_11397[0], tmp_11388[1] + tmp_11397[1], tmp_11388[2] + tmp_11397[2]];
    signal tmp_11399[3] <== [tmp_11398[0] + evals[53][0], tmp_11398[1] + evals[53][1], tmp_11398[2] + evals[53][2]];
    signal tmp_11400[3] <== CMul()(tmp_11399, evals[65]);
    signal tmp_11401[3] <== CMul()(tmp_11375, evals[66]);
    signal tmp_11402[3] <== CMul()(tmp_11386, evals[65]);
    signal tmp_11403[3] <== [tmp_11401[0] + tmp_11402[0], tmp_11401[1] + tmp_11402[1], tmp_11401[2] + tmp_11402[2]];
    signal tmp_11404[3] <== CMul()(tmp_11386, evals[67]);
    signal tmp_11405[3] <== [tmp_11403[0] + tmp_11404[0], tmp_11403[1] + tmp_11404[1], tmp_11403[2] + tmp_11404[2]];
    signal tmp_11406[3] <== CMul()(tmp_11396, evals[66]);
    signal tmp_11407[3] <== [tmp_11405[0] + tmp_11406[0], tmp_11405[1] + tmp_11406[1], tmp_11405[2] + tmp_11406[2]];
    signal tmp_11408[3] <== CMul()(tmp_11396, evals[67]);
    signal tmp_11409[3] <== [tmp_11407[0] + tmp_11408[0], tmp_11407[1] + tmp_11408[1], tmp_11407[2] + tmp_11408[2]];
    signal tmp_11410[3] <== [tmp_11409[0] + evals[54][0], tmp_11409[1] + evals[54][1], tmp_11409[2] + evals[54][2]];
    signal tmp_11411[3] <== CMul()(tmp_11410, evals[67]);
    signal tmp_11412[3] <== [tmp_11400[0] + tmp_11411[0], tmp_11400[1] + tmp_11411[1], tmp_11400[2] + tmp_11411[2]];
    signal tmp_11413[3] <== CMul()(tmp_11375, evals[67]);
    signal tmp_11414[3] <== CMul()(tmp_11396, evals[67]);
    signal tmp_11415[3] <== [tmp_11413[0] + tmp_11414[0], tmp_11413[1] + tmp_11414[1], tmp_11413[2] + tmp_11414[2]];
    signal tmp_11416[3] <== CMul()(tmp_11396, evals[65]);
    signal tmp_11417[3] <== [tmp_11415[0] + tmp_11416[0], tmp_11415[1] + tmp_11416[1], tmp_11415[2] + tmp_11416[2]];
    signal tmp_11418[3] <== CMul()(tmp_11386, evals[66]);
    signal tmp_11419[3] <== [tmp_11417[0] + tmp_11418[0], tmp_11417[1] + tmp_11418[1], tmp_11417[2] + tmp_11418[2]];
    signal tmp_11420[3] <== [tmp_11419[0] + evals[55][0], tmp_11419[1] + evals[55][1], tmp_11419[2] + evals[55][2]];
    signal tmp_11421[3] <== CMul()(tmp_11420, evals[66]);
    signal tmp_11422[3] <== [tmp_11412[0] + tmp_11421[0], tmp_11412[1] + tmp_11421[1], tmp_11412[2] + tmp_11421[2]];
    signal tmp_11423[3] <== [tmp_11422[0] + evals[50][0], tmp_11422[1] + evals[50][1], tmp_11422[2] + evals[50][2]];
    signal tmp_11424[3] <== [tmp_11423[0] - evals[68][0], tmp_11423[1] - evals[68][1], tmp_11423[2] - evals[68][2]];
    signal tmp_11425[3] <== CMul()(evals[43], tmp_11424);
    signal tmp_11426[3] <== [tmp_11345[0] + tmp_11425[0], tmp_11345[1] + tmp_11425[1], tmp_11345[2] + tmp_11425[2]];
    signal tmp_11427[3] <== CMul()(challengeQ, tmp_11426);
    signal tmp_11428[3] <== CMul()(tmp_11399, evals[66]);
    signal tmp_11429[3] <== CMul()(tmp_11410, evals[65]);
    signal tmp_11430[3] <== [tmp_11428[0] + tmp_11429[0], tmp_11428[1] + tmp_11429[1], tmp_11428[2] + tmp_11429[2]];
    signal tmp_11431[3] <== CMul()(tmp_11410, evals[67]);
    signal tmp_11432[3] <== [tmp_11430[0] + tmp_11431[0], tmp_11430[1] + tmp_11431[1], tmp_11430[2] + tmp_11431[2]];
    signal tmp_11433[3] <== CMul()(tmp_11420, evals[66]);
    signal tmp_11434[3] <== [tmp_11432[0] + tmp_11433[0], tmp_11432[1] + tmp_11433[1], tmp_11432[2] + tmp_11433[2]];
    signal tmp_11435[3] <== CMul()(tmp_11420, evals[67]);
    signal tmp_11436[3] <== [tmp_11434[0] + tmp_11435[0], tmp_11434[1] + tmp_11435[1], tmp_11434[2] + tmp_11435[2]];
    signal tmp_11437[3] <== [tmp_11436[0] + evals[51][0], tmp_11436[1] + evals[51][1], tmp_11436[2] + evals[51][2]];
    signal tmp_11438[3] <== [tmp_11437[0] - evals[69][0], tmp_11437[1] - evals[69][1], tmp_11437[2] - evals[69][2]];
    signal tmp_11439[3] <== CMul()(evals[43], tmp_11438);
    signal tmp_11440[3] <== [tmp_11427[0] + tmp_11439[0], tmp_11427[1] + tmp_11439[1], tmp_11427[2] + tmp_11439[2]];
    signal tmp_11441[3] <== CMul()(challengeQ, tmp_11440);
    signal tmp_11442[3] <== CMul()(tmp_11399, evals[67]);
    signal tmp_11443[3] <== CMul()(tmp_11420, evals[67]);
    signal tmp_11444[3] <== [tmp_11442[0] + tmp_11443[0], tmp_11442[1] + tmp_11443[1], tmp_11442[2] + tmp_11443[2]];
    signal tmp_11445[3] <== CMul()(tmp_11420, evals[65]);
    signal tmp_11446[3] <== [tmp_11444[0] + tmp_11445[0], tmp_11444[1] + tmp_11445[1], tmp_11444[2] + tmp_11445[2]];
    signal tmp_11447[3] <== CMul()(tmp_11410, evals[66]);
    signal tmp_11448[3] <== [tmp_11446[0] + tmp_11447[0], tmp_11446[1] + tmp_11447[1], tmp_11446[2] + tmp_11447[2]];
    signal tmp_11449[3] <== [tmp_11448[0] + evals[52][0], tmp_11448[1] + evals[52][1], tmp_11448[2] + evals[52][2]];
    signal tmp_11450[3] <== [tmp_11449[0] - evals[70][0], tmp_11449[1] - evals[70][1], tmp_11449[2] - evals[70][2]];
    signal tmp_11451[3] <== CMul()(evals[43], tmp_11450);
    signal tmp_11452[3] <== [tmp_11441[0] + tmp_11451[0], tmp_11441[1] + tmp_11451[1], tmp_11441[2] + tmp_11451[2]];
    signal tmp_11453[3] <== CMul()(challengeQ, tmp_11452);
    signal tmp_11454[3] <== [1 - evals[111][0], -evals[111][1], -evals[111][2]];
    signal tmp_11455[3] <== [1 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_11456[3] <== CMul()(tmp_11454, tmp_11455);
    signal tmp_11457[3] <== [1 - evals[113][0], -evals[113][1], -evals[113][2]];
    signal tmp_11458[3] <== CMul()(tmp_11456, tmp_11457);
    signal tmp_11459[3] <== CMul()(evals[45], tmp_11458);
    signal tmp_11460[3] <== [evals[50][0] - evals[114][0], evals[50][1] - evals[114][1], evals[50][2] - evals[114][2]];
    signal tmp_11461[3] <== CMul()(tmp_11459, tmp_11460);
    signal tmp_11462[3] <== [tmp_11453[0] + tmp_11461[0], tmp_11453[1] + tmp_11461[1], tmp_11453[2] + tmp_11461[2]];
    signal tmp_11463[3] <== CMul()(challengeQ, tmp_11462);
    signal tmp_11464[3] <== CMul()(evals[45], tmp_11458);
    signal tmp_11465[3] <== [evals[51][0] - evals[115][0], evals[51][1] - evals[115][1], evals[51][2] - evals[115][2]];
    signal tmp_11466[3] <== CMul()(tmp_11464, tmp_11465);
    signal tmp_11467[3] <== [tmp_11463[0] + tmp_11466[0], tmp_11463[1] + tmp_11466[1], tmp_11463[2] + tmp_11466[2]];
    signal tmp_11468[3] <== CMul()(challengeQ, tmp_11467);
    signal tmp_11469[3] <== CMul()(evals[45], tmp_11458);
    signal tmp_11470[3] <== [evals[52][0] - evals[116][0], evals[52][1] - evals[116][1], evals[52][2] - evals[116][2]];
    signal tmp_11471[3] <== CMul()(tmp_11469, tmp_11470);
    signal tmp_11472[3] <== [tmp_11468[0] + tmp_11471[0], tmp_11468[1] + tmp_11471[1], tmp_11468[2] + tmp_11471[2]];
    signal tmp_11473[3] <== CMul()(challengeQ, tmp_11472);
    signal tmp_11474[3] <== CMul()(evals[111], tmp_11455);
    signal tmp_11475[3] <== CMul()(tmp_11474, tmp_11457);
    signal tmp_11476[3] <== CMul()(evals[45], tmp_11475);
    signal tmp_11477[3] <== [evals[53][0] - evals[114][0], evals[53][1] - evals[114][1], evals[53][2] - evals[114][2]];
    signal tmp_11478[3] <== CMul()(tmp_11476, tmp_11477);
    signal tmp_11479[3] <== [tmp_11473[0] + tmp_11478[0], tmp_11473[1] + tmp_11478[1], tmp_11473[2] + tmp_11478[2]];
    signal tmp_11480[3] <== CMul()(challengeQ, tmp_11479);
    signal tmp_11481[3] <== CMul()(evals[45], tmp_11475);
    signal tmp_11482[3] <== [evals[54][0] - evals[115][0], evals[54][1] - evals[115][1], evals[54][2] - evals[115][2]];
    signal tmp_11483[3] <== CMul()(tmp_11481, tmp_11482);
    signal tmp_11484[3] <== [tmp_11480[0] + tmp_11483[0], tmp_11480[1] + tmp_11483[1], tmp_11480[2] + tmp_11483[2]];
    signal tmp_11485[3] <== CMul()(challengeQ, tmp_11484);
    signal tmp_11486[3] <== CMul()(evals[45], tmp_11475);
    signal tmp_11487[3] <== [evals[55][0] - evals[116][0], evals[55][1] - evals[116][1], evals[55][2] - evals[116][2]];
    signal tmp_11488[3] <== CMul()(tmp_11486, tmp_11487);
    signal tmp_11489[3] <== [tmp_11485[0] + tmp_11488[0], tmp_11485[1] + tmp_11488[1], tmp_11485[2] + tmp_11488[2]];
    signal tmp_11490[3] <== CMul()(challengeQ, tmp_11489);
    signal tmp_11491[3] <== CMul()(tmp_11454, evals[112]);
    signal tmp_11492[3] <== CMul()(tmp_11491, tmp_11457);
    signal tmp_11493[3] <== CMul()(evals[45], tmp_11492);
    signal tmp_11494[3] <== [evals[56][0] - evals[114][0], evals[56][1] - evals[114][1], evals[56][2] - evals[114][2]];
    signal tmp_11495[3] <== CMul()(tmp_11493, tmp_11494);
    signal tmp_11496[3] <== [tmp_11490[0] + tmp_11495[0], tmp_11490[1] + tmp_11495[1], tmp_11490[2] + tmp_11495[2]];
    signal tmp_11497[3] <== CMul()(challengeQ, tmp_11496);
    signal tmp_11498[3] <== CMul()(evals[45], tmp_11492);
    signal tmp_11499[3] <== [evals[57][0] - evals[115][0], evals[57][1] - evals[115][1], evals[57][2] - evals[115][2]];
    signal tmp_11500[3] <== CMul()(tmp_11498, tmp_11499);
    signal tmp_11501[3] <== [tmp_11497[0] + tmp_11500[0], tmp_11497[1] + tmp_11500[1], tmp_11497[2] + tmp_11500[2]];
    signal tmp_11502[3] <== CMul()(challengeQ, tmp_11501);
    signal tmp_11503[3] <== CMul()(evals[45], tmp_11492);
    signal tmp_11504[3] <== [evals[58][0] - evals[116][0], evals[58][1] - evals[116][1], evals[58][2] - evals[116][2]];
    signal tmp_11505[3] <== CMul()(tmp_11503, tmp_11504);
    signal tmp_11506[3] <== [tmp_11502[0] + tmp_11505[0], tmp_11502[1] + tmp_11505[1], tmp_11502[2] + tmp_11505[2]];
    signal tmp_11507[3] <== CMul()(challengeQ, tmp_11506);
    signal tmp_11508[3] <== CMul()(evals[111], evals[112]);
    signal tmp_11509[3] <== CMul()(tmp_11508, tmp_11457);
    signal tmp_11510[3] <== CMul()(evals[45], tmp_11509);
    signal tmp_11511[3] <== [evals[59][0] - evals[114][0], evals[59][1] - evals[114][1], evals[59][2] - evals[114][2]];
    signal tmp_11512[3] <== CMul()(tmp_11510, tmp_11511);
    signal tmp_11513[3] <== [tmp_11507[0] + tmp_11512[0], tmp_11507[1] + tmp_11512[1], tmp_11507[2] + tmp_11512[2]];
    signal tmp_11514[3] <== CMul()(challengeQ, tmp_11513);
    signal tmp_11515[3] <== CMul()(evals[45], tmp_11509);
    signal tmp_11516[3] <== [evals[60][0] - evals[115][0], evals[60][1] - evals[115][1], evals[60][2] - evals[115][2]];
    signal tmp_11517[3] <== CMul()(tmp_11515, tmp_11516);
    signal tmp_11518[3] <== [tmp_11514[0] + tmp_11517[0], tmp_11514[1] + tmp_11517[1], tmp_11514[2] + tmp_11517[2]];
    signal tmp_11519[3] <== CMul()(challengeQ, tmp_11518);
    signal tmp_11520[3] <== CMul()(evals[45], tmp_11509);
    signal tmp_11521[3] <== [evals[61][0] - evals[116][0], evals[61][1] - evals[116][1], evals[61][2] - evals[116][2]];
    signal tmp_11522[3] <== CMul()(tmp_11520, tmp_11521);
    signal tmp_11523[3] <== [tmp_11519[0] + tmp_11522[0], tmp_11519[1] + tmp_11522[1], tmp_11519[2] + tmp_11522[2]];
    signal tmp_11524[3] <== CMul()(challengeQ, tmp_11523);
    signal tmp_11525[3] <== CMul()(tmp_11454, tmp_11455);
    signal tmp_11526[3] <== CMul()(tmp_11525, evals[113]);
    signal tmp_11527[3] <== CMul()(evals[45], tmp_11526);
    signal tmp_11528[3] <== [evals[62][0] - evals[114][0], evals[62][1] - evals[114][1], evals[62][2] - evals[114][2]];
    signal tmp_11529[3] <== CMul()(tmp_11527, tmp_11528);
    signal tmp_11530[3] <== [tmp_11524[0] + tmp_11529[0], tmp_11524[1] + tmp_11529[1], tmp_11524[2] + tmp_11529[2]];
    signal tmp_11531[3] <== CMul()(challengeQ, tmp_11530);
    signal tmp_11532[3] <== CMul()(evals[45], tmp_11526);
    signal tmp_11533[3] <== [evals[63][0] - evals[115][0], evals[63][1] - evals[115][1], evals[63][2] - evals[115][2]];
    signal tmp_11534[3] <== CMul()(tmp_11532, tmp_11533);
    signal tmp_11535[3] <== [tmp_11531[0] + tmp_11534[0], tmp_11531[1] + tmp_11534[1], tmp_11531[2] + tmp_11534[2]];
    signal tmp_11536[3] <== CMul()(challengeQ, tmp_11535);
    signal tmp_11537[3] <== CMul()(evals[45], tmp_11526);
    signal tmp_11538[3] <== [evals[64][0] - evals[116][0], evals[64][1] - evals[116][1], evals[64][2] - evals[116][2]];
    signal tmp_11539[3] <== CMul()(tmp_11537, tmp_11538);
    signal tmp_11540[3] <== [tmp_11536[0] + tmp_11539[0], tmp_11536[1] + tmp_11539[1], tmp_11536[2] + tmp_11539[2]];
    signal tmp_11541[3] <== CMul()(challengeQ, tmp_11540);
    signal tmp_11542[3] <== CMul()(evals[111], tmp_11455);
    signal tmp_11543[3] <== CMul()(tmp_11542, evals[113]);
    signal tmp_11544[3] <== CMul()(evals[45], tmp_11543);
    signal tmp_11545[3] <== [evals[65][0] - evals[114][0], evals[65][1] - evals[114][1], evals[65][2] - evals[114][2]];
    signal tmp_11546[3] <== CMul()(tmp_11544, tmp_11545);
    signal tmp_11547[3] <== [tmp_11541[0] + tmp_11546[0], tmp_11541[1] + tmp_11546[1], tmp_11541[2] + tmp_11546[2]];
    signal tmp_11548[3] <== CMul()(challengeQ, tmp_11547);
    signal tmp_11549[3] <== CMul()(evals[45], tmp_11543);
    signal tmp_11550[3] <== [evals[66][0] - evals[115][0], evals[66][1] - evals[115][1], evals[66][2] - evals[115][2]];
    signal tmp_11551[3] <== CMul()(tmp_11549, tmp_11550);
    signal tmp_11552[3] <== [tmp_11548[0] + tmp_11551[0], tmp_11548[1] + tmp_11551[1], tmp_11548[2] + tmp_11551[2]];
    signal tmp_11553[3] <== CMul()(challengeQ, tmp_11552);
    signal tmp_11554[3] <== CMul()(evals[45], tmp_11543);
    signal tmp_11555[3] <== [evals[67][0] - evals[116][0], evals[67][1] - evals[116][1], evals[67][2] - evals[116][2]];
    signal tmp_11556[3] <== CMul()(tmp_11554, tmp_11555);
    signal tmp_11557[3] <== [tmp_11553[0] + tmp_11556[0], tmp_11553[1] + tmp_11556[1], tmp_11553[2] + tmp_11556[2]];
    signal tmp_11558[3] <== CMul()(challengeQ, tmp_11557);
    signal tmp_11559[3] <== CMul()(tmp_11454, evals[112]);
    signal tmp_11560[3] <== CMul()(tmp_11559, evals[113]);
    signal tmp_11561[3] <== CMul()(evals[45], tmp_11560);
    signal tmp_11562[3] <== [evals[68][0] - evals[114][0], evals[68][1] - evals[114][1], evals[68][2] - evals[114][2]];
    signal tmp_11563[3] <== CMul()(tmp_11561, tmp_11562);
    signal tmp_11564[3] <== [tmp_11558[0] + tmp_11563[0], tmp_11558[1] + tmp_11563[1], tmp_11558[2] + tmp_11563[2]];
    signal tmp_11565[3] <== CMul()(challengeQ, tmp_11564);
    signal tmp_11566[3] <== CMul()(evals[45], tmp_11560);
    signal tmp_11567[3] <== [evals[69][0] - evals[115][0], evals[69][1] - evals[115][1], evals[69][2] - evals[115][2]];
    signal tmp_11568[3] <== CMul()(tmp_11566, tmp_11567);
    signal tmp_11569[3] <== [tmp_11565[0] + tmp_11568[0], tmp_11565[1] + tmp_11568[1], tmp_11565[2] + tmp_11568[2]];
    signal tmp_11570[3] <== CMul()(challengeQ, tmp_11569);
    signal tmp_11571[3] <== CMul()(evals[45], tmp_11560);
    signal tmp_11572[3] <== [evals[70][0] - evals[116][0], evals[70][1] - evals[116][1], evals[70][2] - evals[116][2]];
    signal tmp_11573[3] <== CMul()(tmp_11571, tmp_11572);
    signal tmp_11574[3] <== [tmp_11570[0] + tmp_11573[0], tmp_11570[1] + tmp_11573[1], tmp_11570[2] + tmp_11573[2]];
    signal tmp_11575[3] <== CMul()(challengeQ, tmp_11574);
    signal tmp_11576[3] <== CMul()(evals[111], evals[112]);
    signal tmp_11577[3] <== CMul()(tmp_11576, evals[113]);
    signal tmp_11578[3] <== CMul()(evals[45], tmp_11577);
    signal tmp_11579[3] <== [evals[71][0] - evals[114][0], evals[71][1] - evals[114][1], evals[71][2] - evals[114][2]];
    signal tmp_11580[3] <== CMul()(tmp_11578, tmp_11579);
    signal tmp_11581[3] <== [tmp_11575[0] + tmp_11580[0], tmp_11575[1] + tmp_11580[1], tmp_11575[2] + tmp_11580[2]];
    signal tmp_11582[3] <== CMul()(challengeQ, tmp_11581);
    signal tmp_11583[3] <== CMul()(evals[45], tmp_11577);
    signal tmp_11584[3] <== [evals[72][0] - evals[115][0], evals[72][1] - evals[115][1], evals[72][2] - evals[115][2]];
    signal tmp_11585[3] <== CMul()(tmp_11583, tmp_11584);
    signal tmp_11586[3] <== [tmp_11582[0] + tmp_11585[0], tmp_11582[1] + tmp_11585[1], tmp_11582[2] + tmp_11585[2]];
    signal tmp_11587[3] <== CMul()(challengeQ, tmp_11586);
    signal tmp_11588[3] <== CMul()(evals[45], tmp_11577);
    signal tmp_11589[3] <== [evals[73][0] - evals[116][0], evals[73][1] - evals[116][1], evals[73][2] - evals[116][2]];
    signal tmp_11590[3] <== CMul()(tmp_11588, tmp_11589);
    signal tmp_11591[3] <== [tmp_11587[0] + tmp_11590[0], tmp_11587[1] + tmp_11590[1], tmp_11587[2] + tmp_11590[2]];
    signal tmp_11592[3] <== CMul()(challengeQ, tmp_11591);
    signal tmp_11593[3] <== [1 - evals[111][0], -evals[111][1], -evals[111][2]];
    signal tmp_11594[3] <== CMul()(evals[111], tmp_11593);
    signal tmp_11595[3] <== CMul()(evals[45], tmp_11594);
    signal tmp_11596[3] <== [tmp_11592[0] + tmp_11595[0], tmp_11592[1] + tmp_11595[1], tmp_11592[2] + tmp_11595[2]];
    signal tmp_11597[3] <== CMul()(challengeQ, tmp_11596);
    signal tmp_11598[3] <== [1 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_11599[3] <== CMul()(evals[112], tmp_11598);
    signal tmp_11600[3] <== CMul()(evals[45], tmp_11599);
    signal tmp_11601[3] <== [tmp_11597[0] + tmp_11600[0], tmp_11597[1] + tmp_11600[1], tmp_11597[2] + tmp_11600[2]];
    signal tmp_11602[3] <== CMul()(challengeQ, tmp_11601);
    signal tmp_11603[3] <== [1 - evals[113][0], -evals[113][1], -evals[113][2]];
    signal tmp_11604[3] <== CMul()(evals[113], tmp_11603);
    signal tmp_11605[3] <== CMul()(evals[45], tmp_11604);
    signal tmp_11606[3] <== [tmp_11602[0] + tmp_11605[0], tmp_11602[1] + tmp_11605[1], tmp_11602[2] + tmp_11605[2]];
    signal tmp_11607[3] <== CMul()(challengeQ, tmp_11606);
    signal tmp_11608[3] <== [1 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_11609[3] <== [1 - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_11610[3] <== CMul()(tmp_11608, tmp_11609);
    signal tmp_11611[3] <== CMul()(evals[46], tmp_11610);
    signal tmp_11612[3] <== [evals[50][0] - evals[68][0], evals[50][1] - evals[68][1], evals[50][2] - evals[68][2]];
    signal tmp_11613[3] <== CMul()(tmp_11611, tmp_11612);
    signal tmp_11614[3] <== [tmp_11607[0] + tmp_11613[0], tmp_11607[1] + tmp_11613[1], tmp_11607[2] + tmp_11613[2]];
    signal tmp_11615[3] <== CMul()(challengeQ, tmp_11614);
    signal tmp_11616[3] <== CMul()(evals[46], tmp_11610);
    signal tmp_11617[3] <== [evals[51][0] - evals[69][0], evals[51][1] - evals[69][1], evals[51][2] - evals[69][2]];
    signal tmp_11618[3] <== CMul()(tmp_11616, tmp_11617);
    signal tmp_11619[3] <== [tmp_11615[0] + tmp_11618[0], tmp_11615[1] + tmp_11618[1], tmp_11615[2] + tmp_11618[2]];
    signal tmp_11620[3] <== CMul()(challengeQ, tmp_11619);
    signal tmp_11621[3] <== CMul()(evals[46], tmp_11610);
    signal tmp_11622[3] <== [evals[52][0] - evals[70][0], evals[52][1] - evals[70][1], evals[52][2] - evals[70][2]];
    signal tmp_11623[3] <== CMul()(tmp_11621, tmp_11622);
    signal tmp_11624[3] <== [tmp_11620[0] + tmp_11623[0], tmp_11620[1] + tmp_11623[1], tmp_11620[2] + tmp_11623[2]];
    signal tmp_11625[3] <== CMul()(challengeQ, tmp_11624);
    signal tmp_11626[3] <== CMul()(evals[46], tmp_11610);
    signal tmp_11627[3] <== [evals[53][0] - evals[71][0], evals[53][1] - evals[71][1], evals[53][2] - evals[71][2]];
    signal tmp_11628[3] <== CMul()(tmp_11626, tmp_11627);
    signal tmp_11629[3] <== [tmp_11625[0] + tmp_11628[0], tmp_11625[1] + tmp_11628[1], tmp_11625[2] + tmp_11628[2]];
    signal tmp_11630[3] <== CMul()(challengeQ, tmp_11629);
    signal tmp_11631[3] <== [1 - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_11632[3] <== CMul()(evals[66], tmp_11631);
    signal tmp_11633[3] <== CMul()(evals[46], tmp_11632);
    signal tmp_11634[3] <== [evals[54][0] - evals[68][0], evals[54][1] - evals[68][1], evals[54][2] - evals[68][2]];
    signal tmp_11635[3] <== CMul()(tmp_11633, tmp_11634);
    signal tmp_11636[3] <== [tmp_11630[0] + tmp_11635[0], tmp_11630[1] + tmp_11635[1], tmp_11630[2] + tmp_11635[2]];
    signal tmp_11637[3] <== CMul()(challengeQ, tmp_11636);
    signal tmp_11638[3] <== CMul()(evals[46], tmp_11632);
    signal tmp_11639[3] <== [evals[55][0] - evals[69][0], evals[55][1] - evals[69][1], evals[55][2] - evals[69][2]];
    signal tmp_11640[3] <== CMul()(tmp_11638, tmp_11639);
    signal tmp_11641[3] <== [tmp_11637[0] + tmp_11640[0], tmp_11637[1] + tmp_11640[1], tmp_11637[2] + tmp_11640[2]];
    signal tmp_11642[3] <== CMul()(challengeQ, tmp_11641);
    signal tmp_11643[3] <== CMul()(evals[46], tmp_11632);
    signal tmp_11644[3] <== [evals[56][0] - evals[70][0], evals[56][1] - evals[70][1], evals[56][2] - evals[70][2]];
    signal tmp_11645[3] <== CMul()(tmp_11643, tmp_11644);
    signal tmp_11646[3] <== [tmp_11642[0] + tmp_11645[0], tmp_11642[1] + tmp_11645[1], tmp_11642[2] + tmp_11645[2]];
    signal tmp_11647[3] <== CMul()(challengeQ, tmp_11646);
    signal tmp_11648[3] <== CMul()(evals[46], tmp_11632);
    signal tmp_11649[3] <== [evals[57][0] - evals[71][0], evals[57][1] - evals[71][1], evals[57][2] - evals[71][2]];
    signal tmp_11650[3] <== CMul()(tmp_11648, tmp_11649);
    signal tmp_11651[3] <== [tmp_11647[0] + tmp_11650[0], tmp_11647[1] + tmp_11650[1], tmp_11647[2] + tmp_11650[2]];
    signal tmp_11652[3] <== CMul()(challengeQ, tmp_11651);
    signal tmp_11653[3] <== [1 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_11654[3] <== CMul()(tmp_11653, evals[67]);
    signal tmp_11655[3] <== CMul()(evals[46], tmp_11654);
    signal tmp_11656[3] <== [evals[58][0] - evals[68][0], evals[58][1] - evals[68][1], evals[58][2] - evals[68][2]];
    signal tmp_11657[3] <== CMul()(tmp_11655, tmp_11656);
    signal tmp_11658[3] <== [tmp_11652[0] + tmp_11657[0], tmp_11652[1] + tmp_11657[1], tmp_11652[2] + tmp_11657[2]];
    signal tmp_11659[3] <== CMul()(challengeQ, tmp_11658);
    signal tmp_11660[3] <== CMul()(evals[46], tmp_11654);
    signal tmp_11661[3] <== [evals[59][0] - evals[69][0], evals[59][1] - evals[69][1], evals[59][2] - evals[69][2]];
    signal tmp_11662[3] <== CMul()(tmp_11660, tmp_11661);
    signal tmp_11663[3] <== [tmp_11659[0] + tmp_11662[0], tmp_11659[1] + tmp_11662[1], tmp_11659[2] + tmp_11662[2]];
    signal tmp_11664[3] <== CMul()(challengeQ, tmp_11663);
    signal tmp_11665[3] <== CMul()(evals[46], tmp_11654);
    signal tmp_11666[3] <== [evals[60][0] - evals[70][0], evals[60][1] - evals[70][1], evals[60][2] - evals[70][2]];
    signal tmp_11667[3] <== CMul()(tmp_11665, tmp_11666);
    signal tmp_11668[3] <== [tmp_11664[0] + tmp_11667[0], tmp_11664[1] + tmp_11667[1], tmp_11664[2] + tmp_11667[2]];
    signal tmp_11669[3] <== CMul()(challengeQ, tmp_11668);
    signal tmp_11670[3] <== CMul()(evals[46], tmp_11654);
    signal tmp_11671[3] <== [evals[61][0] - evals[71][0], evals[61][1] - evals[71][1], evals[61][2] - evals[71][2]];
    signal tmp_11672[3] <== CMul()(tmp_11670, tmp_11671);
    signal tmp_11673[3] <== [tmp_11669[0] + tmp_11672[0], tmp_11669[1] + tmp_11672[1], tmp_11669[2] + tmp_11672[2]];
    signal tmp_11674[3] <== CMul()(challengeQ, tmp_11673);
    signal tmp_11675[3] <== CMul()(evals[66], evals[67]);
    signal tmp_11676[3] <== CMul()(evals[46], tmp_11675);
    signal tmp_11677[3] <== [evals[62][0] - evals[68][0], evals[62][1] - evals[68][1], evals[62][2] - evals[68][2]];
    signal tmp_11678[3] <== CMul()(tmp_11676, tmp_11677);
    signal tmp_11679[3] <== [tmp_11674[0] + tmp_11678[0], tmp_11674[1] + tmp_11678[1], tmp_11674[2] + tmp_11678[2]];
    signal tmp_11680[3] <== CMul()(challengeQ, tmp_11679);
    signal tmp_11681[3] <== CMul()(evals[46], tmp_11675);
    signal tmp_11682[3] <== [evals[63][0] - evals[69][0], evals[63][1] - evals[69][1], evals[63][2] - evals[69][2]];
    signal tmp_11683[3] <== CMul()(tmp_11681, tmp_11682);
    signal tmp_11684[3] <== [tmp_11680[0] + tmp_11683[0], tmp_11680[1] + tmp_11683[1], tmp_11680[2] + tmp_11683[2]];
    signal tmp_11685[3] <== CMul()(challengeQ, tmp_11684);
    signal tmp_11686[3] <== CMul()(evals[46], tmp_11675);
    signal tmp_11687[3] <== [evals[64][0] - evals[70][0], evals[64][1] - evals[70][1], evals[64][2] - evals[70][2]];
    signal tmp_11688[3] <== CMul()(tmp_11686, tmp_11687);
    signal tmp_11689[3] <== [tmp_11685[0] + tmp_11688[0], tmp_11685[1] + tmp_11688[1], tmp_11685[2] + tmp_11688[2]];
    signal tmp_11690[3] <== CMul()(challengeQ, tmp_11689);
    signal tmp_11691[3] <== CMul()(evals[46], tmp_11675);
    signal tmp_11692[3] <== [evals[65][0] - evals[71][0], evals[65][1] - evals[71][1], evals[65][2] - evals[71][2]];
    signal tmp_11693[3] <== CMul()(tmp_11691, tmp_11692);
    signal tmp_11694[3] <== [tmp_11690[0] + tmp_11693[0], tmp_11690[1] + tmp_11693[1], tmp_11690[2] + tmp_11693[2]];
    signal tmp_11695[3] <== CMul()(challengeQ, tmp_11694);
    signal tmp_11696[3] <== [1 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_11697[3] <== CMul()(evals[66], tmp_11696);
    signal tmp_11698[3] <== CMul()(evals[46], tmp_11697);
    signal tmp_11699[3] <== [tmp_11695[0] + tmp_11698[0], tmp_11695[1] + tmp_11698[1], tmp_11695[2] + tmp_11698[2]];
    signal tmp_11700[3] <== CMul()(challengeQ, tmp_11699);
    signal tmp_11701[3] <== [1 - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_11702[3] <== CMul()(evals[67], tmp_11701);
    signal tmp_11703[3] <== CMul()(evals[46], tmp_11702);
    signal tmp_11704[3] <== [tmp_11700[0] + tmp_11703[0], tmp_11700[1] + tmp_11703[1], tmp_11700[2] + tmp_11703[2]];
    signal tmp_11705[3] <== CMul()(challengeQ, tmp_11704);
    signal tmp_11706[3] <== CMul()(evals[48], challengesStage2[0]);
    signal tmp_11707[3] <== [tmp_11706[0] + evals[50][0], tmp_11706[1] + evals[50][1], tmp_11706[2] + evals[50][2]];
    signal tmp_11708[3] <== CMul()(tmp_11707, challengesStage2[0]);
    signal tmp_11709[3] <== [tmp_11708[0] + 1, tmp_11708[1], tmp_11708[2]];
    signal tmp_11710[3] <== [tmp_11709[0] + challengesStage2[1][0], tmp_11709[1] + challengesStage2[1][1], tmp_11709[2] + challengesStage2[1][2]];
    signal tmp_11711[3] <== [tmp_11710[0] - 1, tmp_11710[1], tmp_11710[2]];
    signal tmp_11712[3] <== [tmp_11711[0] + 1, tmp_11711[1], tmp_11711[2]];
    signal tmp_11713[3] <== [12275445934081160404 * evals[48][0], 12275445934081160404 * evals[48][1], 12275445934081160404 * evals[48][2]];
    signal tmp_11714[3] <== CMul()(tmp_11713, challengesStage2[0]);
    signal tmp_11715[3] <== [tmp_11714[0] + evals[51][0], tmp_11714[1] + evals[51][1], tmp_11714[2] + evals[51][2]];
    signal tmp_11716[3] <== CMul()(tmp_11715, challengesStage2[0]);
    signal tmp_11717[3] <== [tmp_11716[0] + 1, tmp_11716[1], tmp_11716[2]];
    signal tmp_11718[3] <== [tmp_11717[0] + challengesStage2[1][0], tmp_11717[1] + challengesStage2[1][1], tmp_11717[2] + challengesStage2[1][2]];
    signal tmp_11719[3] <== [tmp_11718[0] - 1, tmp_11718[1], tmp_11718[2]];
    signal tmp_11720[3] <== [tmp_11719[0] + 1, tmp_11719[1], tmp_11719[2]];
    signal tmp_11721[3] <== CMul()(tmp_11712, tmp_11720);
    signal tmp_11722[3] <== [4756475762779100925 * evals[48][0], 4756475762779100925 * evals[48][1], 4756475762779100925 * evals[48][2]];
    signal tmp_11723[3] <== CMul()(tmp_11722, challengesStage2[0]);
    signal tmp_11724[3] <== [tmp_11723[0] + evals[52][0], tmp_11723[1] + evals[52][1], tmp_11723[2] + evals[52][2]];
    signal tmp_11725[3] <== CMul()(tmp_11724, challengesStage2[0]);
    signal tmp_11726[3] <== [tmp_11725[0] + 1, tmp_11725[1], tmp_11725[2]];
    signal tmp_11727[3] <== [tmp_11726[0] + challengesStage2[1][0], tmp_11726[1] + challengesStage2[1][1], tmp_11726[2] + challengesStage2[1][2]];
    signal tmp_11728[3] <== [tmp_11727[0] - 1, tmp_11727[1], tmp_11727[2]];
    signal tmp_11729[3] <== [tmp_11728[0] + 1, tmp_11728[1], tmp_11728[2]];
    signal tmp_11730[3] <== CMul()(tmp_11721, tmp_11729);
    signal tmp_11731[3] <== [1279992132519201448 * evals[48][0], 1279992132519201448 * evals[48][1], 1279992132519201448 * evals[48][2]];
    signal tmp_11732[3] <== CMul()(tmp_11731, challengesStage2[0]);
    signal tmp_11733[3] <== [tmp_11732[0] + evals[53][0], tmp_11732[1] + evals[53][1], tmp_11732[2] + evals[53][2]];
    signal tmp_11734[3] <== CMul()(tmp_11733, challengesStage2[0]);
    signal tmp_11735[3] <== [tmp_11734[0] + 1, tmp_11734[1], tmp_11734[2]];
    signal tmp_11736[3] <== [tmp_11735[0] + challengesStage2[1][0], tmp_11735[1] + challengesStage2[1][1], tmp_11735[2] + challengesStage2[1][2]];
    signal tmp_11737[3] <== [tmp_11736[0] - 1, tmp_11736[1], tmp_11736[2]];
    signal tmp_11738[3] <== [tmp_11737[0] + 1, tmp_11737[1], tmp_11737[2]];
    signal tmp_11739[3] <== CMul()(tmp_11730, tmp_11738);
    signal tmp_11740[3] <== [8312008622371998338 * evals[48][0], 8312008622371998338 * evals[48][1], 8312008622371998338 * evals[48][2]];
    signal tmp_11741[3] <== CMul()(tmp_11740, challengesStage2[0]);
    signal tmp_11742[3] <== [tmp_11741[0] + evals[54][0], tmp_11741[1] + evals[54][1], tmp_11741[2] + evals[54][2]];
    signal tmp_11743[3] <== CMul()(tmp_11742, challengesStage2[0]);
    signal tmp_11744[3] <== [tmp_11743[0] + 1, tmp_11743[1], tmp_11743[2]];
    signal tmp_11745[3] <== [tmp_11744[0] + challengesStage2[1][0], tmp_11744[1] + challengesStage2[1][1], tmp_11744[2] + challengesStage2[1][2]];
    signal tmp_11746[3] <== [tmp_11745[0] - 1, tmp_11745[1], tmp_11745[2]];
    signal tmp_11747[3] <== [tmp_11746[0] + 1, tmp_11746[1], tmp_11746[2]];
    signal tmp_11748[3] <== CMul()(tmp_11739, tmp_11747);
    signal tmp_11749[3] <== [7781028390488215464 * evals[48][0], 7781028390488215464 * evals[48][1], 7781028390488215464 * evals[48][2]];
    signal tmp_11750[3] <== CMul()(tmp_11749, challengesStage2[0]);
    signal tmp_11751[3] <== [tmp_11750[0] + evals[55][0], tmp_11750[1] + evals[55][1], tmp_11750[2] + evals[55][2]];
    signal tmp_11752[3] <== CMul()(tmp_11751, challengesStage2[0]);
    signal tmp_11753[3] <== [tmp_11752[0] + 1, tmp_11752[1], tmp_11752[2]];
    signal tmp_11754[3] <== [tmp_11753[0] + challengesStage2[1][0], tmp_11753[1] + challengesStage2[1][1], tmp_11753[2] + challengesStage2[1][2]];
    signal tmp_11755[3] <== [tmp_11754[0] - 1, tmp_11754[1], tmp_11754[2]];
    signal tmp_11756[3] <== [tmp_11755[0] + 1, tmp_11755[1], tmp_11755[2]];
    signal tmp_11757[3] <== CMul()(tmp_11748, tmp_11756);
    signal tmp_11758[3] <== [11302600489504509467 * evals[48][0], 11302600489504509467 * evals[48][1], 11302600489504509467 * evals[48][2]];
    signal tmp_11759[3] <== CMul()(tmp_11758, challengesStage2[0]);
    signal tmp_11760[3] <== [tmp_11759[0] + evals[56][0], tmp_11759[1] + evals[56][1], tmp_11759[2] + evals[56][2]];
    signal tmp_11761[3] <== CMul()(tmp_11760, challengesStage2[0]);
    signal tmp_11762[3] <== [tmp_11761[0] + 1, tmp_11761[1], tmp_11761[2]];
    signal tmp_11763[3] <== [tmp_11762[0] + challengesStage2[1][0], tmp_11762[1] + challengesStage2[1][1], tmp_11762[2] + challengesStage2[1][2]];
    signal tmp_11764[3] <== [tmp_11763[0] - 1, tmp_11763[1], tmp_11763[2]];
    signal tmp_11765[3] <== [tmp_11764[0] + 1, tmp_11764[1], tmp_11764[2]];
    signal tmp_11766[3] <== CMul()(tmp_11757, tmp_11765);
    signal tmp_11767[3] <== CMul()(evals[99], tmp_11766);
    signal tmp_11768[3] <== CMul()(evals[5], challengesStage2[0]);
    signal tmp_11769[3] <== [tmp_11768[0] + evals[50][0], tmp_11768[1] + evals[50][1], tmp_11768[2] + evals[50][2]];
    signal tmp_11770[3] <== CMul()(tmp_11769, challengesStage2[0]);
    signal tmp_11771[3] <== [tmp_11770[0] + 1, tmp_11770[1], tmp_11770[2]];
    signal tmp_11772[3] <== [tmp_11771[0] + challengesStage2[1][0], tmp_11771[1] + challengesStage2[1][1], tmp_11771[2] + challengesStage2[1][2]];
    signal tmp_11773[3] <== [tmp_11772[0] - 1, tmp_11772[1], tmp_11772[2]];
    signal tmp_11774[3] <== [tmp_11773[0] + 1, tmp_11773[1], tmp_11773[2]];
    signal tmp_11775[3] <== CMul()(evals[6], challengesStage2[0]);
    signal tmp_11776[3] <== [tmp_11775[0] + evals[51][0], tmp_11775[1] + evals[51][1], tmp_11775[2] + evals[51][2]];
    signal tmp_11777[3] <== CMul()(tmp_11776, challengesStage2[0]);
    signal tmp_11778[3] <== [tmp_11777[0] + 1, tmp_11777[1], tmp_11777[2]];
    signal tmp_11779[3] <== [tmp_11778[0] + challengesStage2[1][0], tmp_11778[1] + challengesStage2[1][1], tmp_11778[2] + challengesStage2[1][2]];
    signal tmp_11780[3] <== [tmp_11779[0] - 1, tmp_11779[1], tmp_11779[2]];
    signal tmp_11781[3] <== [tmp_11780[0] + 1, tmp_11780[1], tmp_11780[2]];
    signal tmp_11782[3] <== CMul()(tmp_11774, tmp_11781);
    signal tmp_11783[3] <== CMul()(evals[7], challengesStage2[0]);
    signal tmp_11784[3] <== [tmp_11783[0] + evals[52][0], tmp_11783[1] + evals[52][1], tmp_11783[2] + evals[52][2]];
    signal tmp_11785[3] <== CMul()(tmp_11784, challengesStage2[0]);
    signal tmp_11786[3] <== [tmp_11785[0] + 1, tmp_11785[1], tmp_11785[2]];
    signal tmp_11787[3] <== [tmp_11786[0] + challengesStage2[1][0], tmp_11786[1] + challengesStage2[1][1], tmp_11786[2] + challengesStage2[1][2]];
    signal tmp_11788[3] <== [tmp_11787[0] - 1, tmp_11787[1], tmp_11787[2]];
    signal tmp_11789[3] <== [tmp_11788[0] + 1, tmp_11788[1], tmp_11788[2]];
    signal tmp_11790[3] <== CMul()(tmp_11782, tmp_11789);
    signal tmp_11791[3] <== CMul()(evals[8], challengesStage2[0]);
    signal tmp_11792[3] <== [tmp_11791[0] + evals[53][0], tmp_11791[1] + evals[53][1], tmp_11791[2] + evals[53][2]];
    signal tmp_11793[3] <== CMul()(tmp_11792, challengesStage2[0]);
    signal tmp_11794[3] <== [tmp_11793[0] + 1, tmp_11793[1], tmp_11793[2]];
    signal tmp_11795[3] <== [tmp_11794[0] + challengesStage2[1][0], tmp_11794[1] + challengesStage2[1][1], tmp_11794[2] + challengesStage2[1][2]];
    signal tmp_11796[3] <== [tmp_11795[0] - 1, tmp_11795[1], tmp_11795[2]];
    signal tmp_11797[3] <== [tmp_11796[0] + 1, tmp_11796[1], tmp_11796[2]];
    signal tmp_11798[3] <== CMul()(tmp_11790, tmp_11797);
    signal tmp_11799[3] <== CMul()(evals[9], challengesStage2[0]);
    signal tmp_11800[3] <== [tmp_11799[0] + evals[54][0], tmp_11799[1] + evals[54][1], tmp_11799[2] + evals[54][2]];
    signal tmp_11801[3] <== CMul()(tmp_11800, challengesStage2[0]);
    signal tmp_11802[3] <== [tmp_11801[0] + 1, tmp_11801[1], tmp_11801[2]];
    signal tmp_11803[3] <== [tmp_11802[0] + challengesStage2[1][0], tmp_11802[1] + challengesStage2[1][1], tmp_11802[2] + challengesStage2[1][2]];
    signal tmp_11804[3] <== [tmp_11803[0] - 1, tmp_11803[1], tmp_11803[2]];
    signal tmp_11805[3] <== [tmp_11804[0] + 1, tmp_11804[1], tmp_11804[2]];
    signal tmp_11806[3] <== CMul()(tmp_11798, tmp_11805);
    signal tmp_11807[3] <== CMul()(evals[10], challengesStage2[0]);
    signal tmp_11808[3] <== [tmp_11807[0] + evals[55][0], tmp_11807[1] + evals[55][1], tmp_11807[2] + evals[55][2]];
    signal tmp_11809[3] <== CMul()(tmp_11808, challengesStage2[0]);
    signal tmp_11810[3] <== [tmp_11809[0] + 1, tmp_11809[1], tmp_11809[2]];
    signal tmp_11811[3] <== [tmp_11810[0] + challengesStage2[1][0], tmp_11810[1] + challengesStage2[1][1], tmp_11810[2] + challengesStage2[1][2]];
    signal tmp_11812[3] <== [tmp_11811[0] - 1, tmp_11811[1], tmp_11811[2]];
    signal tmp_11813[3] <== [tmp_11812[0] + 1, tmp_11812[1], tmp_11812[2]];
    signal tmp_11814[3] <== CMul()(tmp_11806, tmp_11813);
    signal tmp_11815[3] <== CMul()(evals[11], challengesStage2[0]);
    signal tmp_11816[3] <== [tmp_11815[0] + evals[56][0], tmp_11815[1] + evals[56][1], tmp_11815[2] + evals[56][2]];
    signal tmp_11817[3] <== CMul()(tmp_11816, challengesStage2[0]);
    signal tmp_11818[3] <== [tmp_11817[0] + 1, tmp_11817[1], tmp_11817[2]];
    signal tmp_11819[3] <== [tmp_11818[0] + challengesStage2[1][0], tmp_11818[1] + challengesStage2[1][1], tmp_11818[2] + challengesStage2[1][2]];
    signal tmp_11820[3] <== [tmp_11819[0] - 1, tmp_11819[1], tmp_11819[2]];
    signal tmp_11821[3] <== [tmp_11820[0] + 1, tmp_11820[1], tmp_11820[2]];
    signal tmp_11822[3] <== CMul()(tmp_11814, tmp_11821);
    signal tmp_11823[3] <== CMul()(evals[12], challengesStage2[0]);
    signal tmp_11824[3] <== [tmp_11823[0] + evals[57][0], tmp_11823[1] + evals[57][1], tmp_11823[2] + evals[57][2]];
    signal tmp_11825[3] <== CMul()(tmp_11824, challengesStage2[0]);
    signal tmp_11826[3] <== [tmp_11825[0] + 1, tmp_11825[1], tmp_11825[2]];
    signal tmp_11827[3] <== [tmp_11826[0] + challengesStage2[1][0], tmp_11826[1] + challengesStage2[1][1], tmp_11826[2] + challengesStage2[1][2]];
    signal tmp_11828[3] <== [tmp_11827[0] - 1, tmp_11827[1], tmp_11827[2]];
    signal tmp_11829[3] <== [tmp_11828[0] + 1, tmp_11828[1], tmp_11828[2]];
    signal tmp_11830[3] <== CMul()(tmp_11822, tmp_11829);
    signal tmp_11831[3] <== [tmp_11767[0] - tmp_11830[0], tmp_11767[1] - tmp_11830[1], tmp_11767[2] - tmp_11830[2]];
    signal tmp_11832[3] <== [tmp_11705[0] + tmp_11831[0], tmp_11705[1] + tmp_11831[1], tmp_11705[2] + tmp_11831[2]];
    signal tmp_11833[3] <== CMul()(challengeQ, tmp_11832);
    signal tmp_11834[3] <== [4549350404001778198 * evals[48][0], 4549350404001778198 * evals[48][1], 4549350404001778198 * evals[48][2]];
    signal tmp_11835[3] <== CMul()(tmp_11834, challengesStage2[0]);
    signal tmp_11836[3] <== [tmp_11835[0] + evals[57][0], tmp_11835[1] + evals[57][1], tmp_11835[2] + evals[57][2]];
    signal tmp_11837[3] <== CMul()(tmp_11836, challengesStage2[0]);
    signal tmp_11838[3] <== [tmp_11837[0] + 1, tmp_11837[1], tmp_11837[2]];
    signal tmp_11839[3] <== [tmp_11838[0] + challengesStage2[1][0], tmp_11838[1] + challengesStage2[1][1], tmp_11838[2] + challengesStage2[1][2]];
    signal tmp_11840[3] <== [tmp_11839[0] - 1, tmp_11839[1], tmp_11839[2]];
    signal tmp_11841[3] <== [tmp_11840[0] + 1, tmp_11840[1], tmp_11840[2]];
    signal tmp_11842[3] <== [3688660304411827445 * evals[48][0], 3688660304411827445 * evals[48][1], 3688660304411827445 * evals[48][2]];
    signal tmp_11843[3] <== CMul()(tmp_11842, challengesStage2[0]);
    signal tmp_11844[3] <== [tmp_11843[0] + evals[58][0], tmp_11843[1] + evals[58][1], tmp_11843[2] + evals[58][2]];
    signal tmp_11845[3] <== CMul()(tmp_11844, challengesStage2[0]);
    signal tmp_11846[3] <== [tmp_11845[0] + 1, tmp_11845[1], tmp_11845[2]];
    signal tmp_11847[3] <== [tmp_11846[0] + challengesStage2[1][0], tmp_11846[1] + challengesStage2[1][1], tmp_11846[2] + challengesStage2[1][2]];
    signal tmp_11848[3] <== [tmp_11847[0] - 1, tmp_11847[1], tmp_11847[2]];
    signal tmp_11849[3] <== [tmp_11848[0] + 1, tmp_11848[1], tmp_11848[2]];
    signal tmp_11850[3] <== CMul()(tmp_11841, tmp_11849);
    signal tmp_11851[3] <== [16725109960945739746 * evals[48][0], 16725109960945739746 * evals[48][1], 16725109960945739746 * evals[48][2]];
    signal tmp_11852[3] <== CMul()(tmp_11851, challengesStage2[0]);
    signal tmp_11853[3] <== [tmp_11852[0] + evals[59][0], tmp_11852[1] + evals[59][1], tmp_11852[2] + evals[59][2]];
    signal tmp_11854[3] <== CMul()(tmp_11853, challengesStage2[0]);
    signal tmp_11855[3] <== [tmp_11854[0] + 1, tmp_11854[1], tmp_11854[2]];
    signal tmp_11856[3] <== [tmp_11855[0] + challengesStage2[1][0], tmp_11855[1] + challengesStage2[1][1], tmp_11855[2] + challengesStage2[1][2]];
    signal tmp_11857[3] <== [tmp_11856[0] - 1, tmp_11856[1], tmp_11856[2]];
    signal tmp_11858[3] <== [tmp_11857[0] + 1, tmp_11857[1], tmp_11857[2]];
    signal tmp_11859[3] <== CMul()(tmp_11850, tmp_11858);
    signal tmp_11860[3] <== [16538725463549498621 * evals[48][0], 16538725463549498621 * evals[48][1], 16538725463549498621 * evals[48][2]];
    signal tmp_11861[3] <== CMul()(tmp_11860, challengesStage2[0]);
    signal tmp_11862[3] <== [tmp_11861[0] + evals[60][0], tmp_11861[1] + evals[60][1], tmp_11861[2] + evals[60][2]];
    signal tmp_11863[3] <== CMul()(tmp_11862, challengesStage2[0]);
    signal tmp_11864[3] <== [tmp_11863[0] + 1, tmp_11863[1], tmp_11863[2]];
    signal tmp_11865[3] <== [tmp_11864[0] + challengesStage2[1][0], tmp_11864[1] + challengesStage2[1][1], tmp_11864[2] + challengesStage2[1][2]];
    signal tmp_11866[3] <== [tmp_11865[0] - 1, tmp_11865[1], tmp_11865[2]];
    signal tmp_11867[3] <== [tmp_11866[0] + 1, tmp_11866[1], tmp_11866[2]];
    signal tmp_11868[3] <== CMul()(tmp_11859, tmp_11867);
    signal tmp_11869[3] <== [12756200801261202346 * evals[48][0], 12756200801261202346 * evals[48][1], 12756200801261202346 * evals[48][2]];
    signal tmp_11870[3] <== CMul()(tmp_11869, challengesStage2[0]);
    signal tmp_11871[3] <== [tmp_11870[0] + evals[61][0], tmp_11870[1] + evals[61][1], tmp_11870[2] + evals[61][2]];
    signal tmp_11872[3] <== CMul()(tmp_11871, challengesStage2[0]);
    signal tmp_11873[3] <== [tmp_11872[0] + 1, tmp_11872[1], tmp_11872[2]];
    signal tmp_11874[3] <== [tmp_11873[0] + challengesStage2[1][0], tmp_11873[1] + challengesStage2[1][1], tmp_11873[2] + challengesStage2[1][2]];
    signal tmp_11875[3] <== [tmp_11874[0] - 1, tmp_11874[1], tmp_11874[2]];
    signal tmp_11876[3] <== [tmp_11875[0] + 1, tmp_11875[1], tmp_11875[2]];
    signal tmp_11877[3] <== CMul()(tmp_11868, tmp_11876);
    signal tmp_11878[3] <== [15099809066790865939 * evals[48][0], 15099809066790865939 * evals[48][1], 15099809066790865939 * evals[48][2]];
    signal tmp_11879[3] <== CMul()(tmp_11878, challengesStage2[0]);
    signal tmp_11880[3] <== [tmp_11879[0] + evals[62][0], tmp_11879[1] + evals[62][1], tmp_11879[2] + evals[62][2]];
    signal tmp_11881[3] <== CMul()(tmp_11880, challengesStage2[0]);
    signal tmp_11882[3] <== [tmp_11881[0] + 1, tmp_11881[1], tmp_11881[2]];
    signal tmp_11883[3] <== [tmp_11882[0] + challengesStage2[1][0], tmp_11882[1] + challengesStage2[1][1], tmp_11882[2] + challengesStage2[1][2]];
    signal tmp_11884[3] <== [tmp_11883[0] - 1, tmp_11883[1], tmp_11883[2]];
    signal tmp_11885[3] <== [tmp_11884[0] + 1, tmp_11884[1], tmp_11884[2]];
    signal tmp_11886[3] <== CMul()(tmp_11877, tmp_11885);
    signal tmp_11887[3] <== [17214954929431464349 * evals[48][0], 17214954929431464349 * evals[48][1], 17214954929431464349 * evals[48][2]];
    signal tmp_11888[3] <== CMul()(tmp_11887, challengesStage2[0]);
    signal tmp_11889[3] <== [tmp_11888[0] + evals[63][0], tmp_11888[1] + evals[63][1], tmp_11888[2] + evals[63][2]];
    signal tmp_11890[3] <== CMul()(tmp_11889, challengesStage2[0]);
    signal tmp_11891[3] <== [tmp_11890[0] + 1, tmp_11890[1], tmp_11890[2]];
    signal tmp_11892[3] <== [tmp_11891[0] + challengesStage2[1][0], tmp_11891[1] + challengesStage2[1][1], tmp_11891[2] + challengesStage2[1][2]];
    signal tmp_11893[3] <== [tmp_11892[0] - 1, tmp_11892[1], tmp_11892[2]];
    signal tmp_11894[3] <== [tmp_11893[0] + 1, tmp_11893[1], tmp_11893[2]];
    signal tmp_11895[3] <== CMul()(tmp_11886, tmp_11894);
    signal tmp_11896[3] <== CMul()(evals[100], tmp_11895);
    signal tmp_11897[3] <== CMul()(evals[13], challengesStage2[0]);
    signal tmp_11898[3] <== [tmp_11897[0] + evals[58][0], tmp_11897[1] + evals[58][1], tmp_11897[2] + evals[58][2]];
    signal tmp_11899[3] <== CMul()(tmp_11898, challengesStage2[0]);
    signal tmp_11900[3] <== [tmp_11899[0] + 1, tmp_11899[1], tmp_11899[2]];
    signal tmp_11901[3] <== [tmp_11900[0] + challengesStage2[1][0], tmp_11900[1] + challengesStage2[1][1], tmp_11900[2] + challengesStage2[1][2]];
    signal tmp_11902[3] <== [tmp_11901[0] - 1, tmp_11901[1], tmp_11901[2]];
    signal tmp_11903[3] <== [tmp_11902[0] + 1, tmp_11902[1], tmp_11902[2]];
    signal tmp_11904[3] <== CMul()(evals[99], tmp_11903);
    signal tmp_11905[3] <== CMul()(evals[14], challengesStage2[0]);
    signal tmp_11906[3] <== [tmp_11905[0] + evals[59][0], tmp_11905[1] + evals[59][1], tmp_11905[2] + evals[59][2]];
    signal tmp_11907[3] <== CMul()(tmp_11906, challengesStage2[0]);
    signal tmp_11908[3] <== [tmp_11907[0] + 1, tmp_11907[1], tmp_11907[2]];
    signal tmp_11909[3] <== [tmp_11908[0] + challengesStage2[1][0], tmp_11908[1] + challengesStage2[1][1], tmp_11908[2] + challengesStage2[1][2]];
    signal tmp_11910[3] <== [tmp_11909[0] - 1, tmp_11909[1], tmp_11909[2]];
    signal tmp_11911[3] <== [tmp_11910[0] + 1, tmp_11910[1], tmp_11910[2]];
    signal tmp_11912[3] <== CMul()(tmp_11904, tmp_11911);
    signal tmp_11913[3] <== CMul()(evals[15], challengesStage2[0]);
    signal tmp_11914[3] <== [tmp_11913[0] + evals[60][0], tmp_11913[1] + evals[60][1], tmp_11913[2] + evals[60][2]];
    signal tmp_11915[3] <== CMul()(tmp_11914, challengesStage2[0]);
    signal tmp_11916[3] <== [tmp_11915[0] + 1, tmp_11915[1], tmp_11915[2]];
    signal tmp_11917[3] <== [tmp_11916[0] + challengesStage2[1][0], tmp_11916[1] + challengesStage2[1][1], tmp_11916[2] + challengesStage2[1][2]];
    signal tmp_11918[3] <== [tmp_11917[0] - 1, tmp_11917[1], tmp_11917[2]];
    signal tmp_11919[3] <== [tmp_11918[0] + 1, tmp_11918[1], tmp_11918[2]];
    signal tmp_11920[3] <== CMul()(tmp_11912, tmp_11919);
    signal tmp_11921[3] <== CMul()(evals[16], challengesStage2[0]);
    signal tmp_11922[3] <== [tmp_11921[0] + evals[61][0], tmp_11921[1] + evals[61][1], tmp_11921[2] + evals[61][2]];
    signal tmp_11923[3] <== CMul()(tmp_11922, challengesStage2[0]);
    signal tmp_11924[3] <== [tmp_11923[0] + 1, tmp_11923[1], tmp_11923[2]];
    signal tmp_11925[3] <== [tmp_11924[0] + challengesStage2[1][0], tmp_11924[1] + challengesStage2[1][1], tmp_11924[2] + challengesStage2[1][2]];
    signal tmp_11926[3] <== [tmp_11925[0] - 1, tmp_11925[1], tmp_11925[2]];
    signal tmp_11927[3] <== [tmp_11926[0] + 1, tmp_11926[1], tmp_11926[2]];
    signal tmp_11928[3] <== CMul()(tmp_11920, tmp_11927);
    signal tmp_11929[3] <== CMul()(evals[17], challengesStage2[0]);
    signal tmp_11930[3] <== [tmp_11929[0] + evals[62][0], tmp_11929[1] + evals[62][1], tmp_11929[2] + evals[62][2]];
    signal tmp_11931[3] <== CMul()(tmp_11930, challengesStage2[0]);
    signal tmp_11932[3] <== [tmp_11931[0] + 1, tmp_11931[1], tmp_11931[2]];
    signal tmp_11933[3] <== [tmp_11932[0] + challengesStage2[1][0], tmp_11932[1] + challengesStage2[1][1], tmp_11932[2] + challengesStage2[1][2]];
    signal tmp_11934[3] <== [tmp_11933[0] - 1, tmp_11933[1], tmp_11933[2]];
    signal tmp_11935[3] <== [tmp_11934[0] + 1, tmp_11934[1], tmp_11934[2]];
    signal tmp_11936[3] <== CMul()(tmp_11928, tmp_11935);
    signal tmp_11937[3] <== CMul()(evals[18], challengesStage2[0]);
    signal tmp_11938[3] <== [tmp_11937[0] + evals[63][0], tmp_11937[1] + evals[63][1], tmp_11937[2] + evals[63][2]];
    signal tmp_11939[3] <== CMul()(tmp_11938, challengesStage2[0]);
    signal tmp_11940[3] <== [tmp_11939[0] + 1, tmp_11939[1], tmp_11939[2]];
    signal tmp_11941[3] <== [tmp_11940[0] + challengesStage2[1][0], tmp_11940[1] + challengesStage2[1][1], tmp_11940[2] + challengesStage2[1][2]];
    signal tmp_11942[3] <== [tmp_11941[0] - 1, tmp_11941[1], tmp_11941[2]];
    signal tmp_11943[3] <== [tmp_11942[0] + 1, tmp_11942[1], tmp_11942[2]];
    signal tmp_11944[3] <== CMul()(tmp_11936, tmp_11943);
    signal tmp_11945[3] <== CMul()(evals[19], challengesStage2[0]);
    signal tmp_11946[3] <== [tmp_11945[0] + evals[64][0], tmp_11945[1] + evals[64][1], tmp_11945[2] + evals[64][2]];
    signal tmp_11947[3] <== CMul()(tmp_11946, challengesStage2[0]);
    signal tmp_11948[3] <== [tmp_11947[0] + 1, tmp_11947[1], tmp_11947[2]];
    signal tmp_11949[3] <== [tmp_11948[0] + challengesStage2[1][0], tmp_11948[1] + challengesStage2[1][1], tmp_11948[2] + challengesStage2[1][2]];
    signal tmp_11950[3] <== [tmp_11949[0] - 1, tmp_11949[1], tmp_11949[2]];
    signal tmp_11951[3] <== [tmp_11950[0] + 1, tmp_11950[1], tmp_11950[2]];
    signal tmp_11952[3] <== CMul()(tmp_11944, tmp_11951);
    signal tmp_11953[3] <== [tmp_11896[0] - tmp_11952[0], tmp_11896[1] - tmp_11952[1], tmp_11896[2] - tmp_11952[2]];
    signal tmp_11954[3] <== [tmp_11833[0] + tmp_11953[0], tmp_11833[1] + tmp_11953[1], tmp_11833[2] + tmp_11953[2]];
    tmp_11955 <== CMul()(challengeQ, tmp_11954);
    signal tmp_11956[3] <== [11016800570561344835 * evals[48][0], 11016800570561344835 * evals[48][1], 11016800570561344835 * evals[48][2]];
    signal tmp_11957[3] <== CMul()(tmp_11956, challengesStage2[0]);
    signal tmp_11958[3] <== [tmp_11957[0] + evals[64][0], tmp_11957[1] + evals[64][1], tmp_11957[2] + evals[64][2]];
    signal tmp_11959[3] <== CMul()(tmp_11958, challengesStage2[0]);
    signal tmp_11960[3] <== [tmp_11959[0] + 1, tmp_11959[1], tmp_11959[2]];
    signal tmp_11961[3] <== [tmp_11960[0] + challengesStage2[1][0], tmp_11960[1] + challengesStage2[1][1], tmp_11960[2] + challengesStage2[1][2]];
    signal tmp_11962[3] <== [tmp_11961[0] - 1, tmp_11961[1], tmp_11961[2]];
    signal tmp_11963[3] <== [tmp_11962[0] + 1, tmp_11962[1], tmp_11962[2]];
    signal tmp_11964[3] <== [11274872323250451096 * evals[48][0], 11274872323250451096 * evals[48][1], 11274872323250451096 * evals[48][2]];
    signal tmp_11965[3] <== CMul()(tmp_11964, challengesStage2[0]);
    signal tmp_11966[3] <== [tmp_11965[0] + evals[65][0], tmp_11965[1] + evals[65][1], tmp_11965[2] + evals[65][2]];
    signal tmp_11967[3] <== CMul()(tmp_11966, challengesStage2[0]);
    signal tmp_11968[3] <== [tmp_11967[0] + 1, tmp_11967[1], tmp_11967[2]];
    signal tmp_11969[3] <== [tmp_11968[0] + challengesStage2[1][0], tmp_11968[1] + challengesStage2[1][1], tmp_11968[2] + challengesStage2[1][2]];
    signal tmp_11970[3] <== [tmp_11969[0] - 1, tmp_11969[1], tmp_11969[2]];
    signal tmp_11971[3] <== [tmp_11970[0] + 1, tmp_11970[1], tmp_11970[2]];
    signal tmp_11972[3] <== CMul()(tmp_11963, tmp_11971);
    signal tmp_11973[3] <== [6534114114080170934 * evals[48][0], 6534114114080170934 * evals[48][1], 6534114114080170934 * evals[48][2]];
    signal tmp_11974[3] <== CMul()(tmp_11973, challengesStage2[0]);
    signal tmp_11975[3] <== [tmp_11974[0] + evals[66][0], tmp_11974[1] + evals[66][1], tmp_11974[2] + evals[66][2]];
    signal tmp_11976[3] <== CMul()(tmp_11975, challengesStage2[0]);
    signal tmp_11977[3] <== [tmp_11976[0] + 1, tmp_11976[1], tmp_11976[2]];
    signal tmp_11978[3] <== [tmp_11977[0] + challengesStage2[1][0], tmp_11977[1] + challengesStage2[1][1], tmp_11977[2] + challengesStage2[1][2]];
    signal tmp_11979[3] <== [tmp_11978[0] - 1, tmp_11978[1], tmp_11978[2]];
    signal tmp_11980[3] <== [tmp_11979[0] + 1, tmp_11979[1], tmp_11979[2]];
    signal tmp_11981[3] <== CMul()(tmp_11972, tmp_11980);
    signal tmp_11982[3] <== [13047390008333835222 * evals[48][0], 13047390008333835222 * evals[48][1], 13047390008333835222 * evals[48][2]];
    signal tmp_11983[3] <== CMul()(tmp_11982, challengesStage2[0]);
    signal tmp_11984[3] <== [tmp_11983[0] + evals[67][0], tmp_11983[1] + evals[67][1], tmp_11983[2] + evals[67][2]];
    signal tmp_11985[3] <== CMul()(tmp_11984, challengesStage2[0]);
    signal tmp_11986[3] <== [tmp_11985[0] + 1, tmp_11985[1], tmp_11985[2]];
    signal tmp_11987[3] <== [tmp_11986[0] + challengesStage2[1][0], tmp_11986[1] + challengesStage2[1][1], tmp_11986[2] + challengesStage2[1][2]];
    signal tmp_11988[3] <== [tmp_11987[0] - 1, tmp_11987[1], tmp_11987[2]];
    signal tmp_11989[3] <== [tmp_11988[0] + 1, tmp_11988[1], tmp_11988[2]];
    signal tmp_11990[3] <== CMul()(tmp_11981, tmp_11989);
    signal tmp_11991[3] <== [11189528522318044176 * evals[48][0], 11189528522318044176 * evals[48][1], 11189528522318044176 * evals[48][2]];
    signal tmp_11992[3] <== CMul()(tmp_11991, challengesStage2[0]);
    signal tmp_11993[3] <== [tmp_11992[0] + evals[68][0], tmp_11992[1] + evals[68][1], tmp_11992[2] + evals[68][2]];
    signal tmp_11994[3] <== CMul()(tmp_11993, challengesStage2[0]);
    signal tmp_11995[3] <== [tmp_11994[0] + 1, tmp_11994[1], tmp_11994[2]];
    signal tmp_11996[3] <== [tmp_11995[0] + challengesStage2[1][0], tmp_11995[1] + challengesStage2[1][1], tmp_11995[2] + challengesStage2[1][2]];
    signal tmp_11997[3] <== [tmp_11996[0] - 1, tmp_11996[1], tmp_11996[2]];
    signal tmp_11998[3] <== [tmp_11997[0] + 1, tmp_11997[1], tmp_11997[2]];
    signal tmp_11999[3] <== CMul()(tmp_11990, tmp_11998);
    signal tmp_12000[3] <== [3320735505586735876 * evals[48][0], 3320735505586735876 * evals[48][1], 3320735505586735876 * evals[48][2]];
    signal tmp_12001[3] <== CMul()(tmp_12000, challengesStage2[0]);
    signal tmp_12002[3] <== [tmp_12001[0] + evals[69][0], tmp_12001[1] + evals[69][1], tmp_12001[2] + evals[69][2]];
    signal tmp_12003[3] <== CMul()(tmp_12002, challengesStage2[0]);
    signal tmp_12004[3] <== [tmp_12003[0] + 1, tmp_12003[1], tmp_12003[2]];
    signal tmp_12005[3] <== [tmp_12004[0] + challengesStage2[1][0], tmp_12004[1] + challengesStage2[1][1], tmp_12004[2] + challengesStage2[1][2]];
    signal tmp_12006[3] <== [tmp_12005[0] - 1, tmp_12005[1], tmp_12005[2]];
    signal tmp_12007[3] <== [tmp_12006[0] + 1, tmp_12006[1], tmp_12006[2]];
    signal tmp_12008[3] <== CMul()(tmp_11999, tmp_12007);
    signal tmp_12009[3] <== [7240278926970958133 * evals[48][0], 7240278926970958133 * evals[48][1], 7240278926970958133 * evals[48][2]];
    signal tmp_12010[3] <== CMul()(tmp_12009, challengesStage2[0]);
    signal tmp_12011[3] <== [tmp_12010[0] + evals[70][0], tmp_12010[1] + evals[70][1], tmp_12010[2] + evals[70][2]];
    signal tmp_12012[3] <== CMul()(tmp_12011, challengesStage2[0]);
    signal tmp_12013[3] <== [tmp_12012[0] + 1, tmp_12012[1], tmp_12012[2]];
    signal tmp_12014[3] <== [tmp_12013[0] + challengesStage2[1][0], tmp_12013[1] + challengesStage2[1][1], tmp_12013[2] + challengesStage2[1][2]];
    signal tmp_12015[3] <== [tmp_12014[0] - 1, tmp_12014[1], tmp_12014[2]];
    signal tmp_12016[3] <== [tmp_12015[0] + 1, tmp_12015[1], tmp_12015[2]];
    signal tmp_12017[3] <== CMul()(tmp_12008, tmp_12016);
    tmp_12018 <== CMul()(evals[101], tmp_12017);
    signal tmp_12019[3] <== CMul()(evals[20], challengesStage2[0]);
    signal tmp_12020[3] <== [tmp_12019[0] + evals[65][0], tmp_12019[1] + evals[65][1], tmp_12019[2] + evals[65][2]];
    signal tmp_12021[3] <== CMul()(tmp_12020, challengesStage2[0]);
    signal tmp_12022[3] <== [tmp_12021[0] + 1, tmp_12021[1], tmp_12021[2]];
    signal tmp_12023[3] <== [tmp_12022[0] + challengesStage2[1][0], tmp_12022[1] + challengesStage2[1][1], tmp_12022[2] + challengesStage2[1][2]];
    signal tmp_12024[3] <== [tmp_12023[0] - 1, tmp_12023[1], tmp_12023[2]];
    signal tmp_12025[3] <== [tmp_12024[0] + 1, tmp_12024[1], tmp_12024[2]];
    signal tmp_12026[3] <== CMul()(evals[100], tmp_12025);
    signal tmp_12027[3] <== CMul()(evals[21], challengesStage2[0]);
    signal tmp_12028[3] <== [tmp_12027[0] + evals[66][0], tmp_12027[1] + evals[66][1], tmp_12027[2] + evals[66][2]];
    signal tmp_12029[3] <== CMul()(tmp_12028, challengesStage2[0]);
    signal tmp_12030[3] <== [tmp_12029[0] + 1, tmp_12029[1], tmp_12029[2]];
    signal tmp_12031[3] <== [tmp_12030[0] + challengesStage2[1][0], tmp_12030[1] + challengesStage2[1][1], tmp_12030[2] + challengesStage2[1][2]];
    signal tmp_12032[3] <== [tmp_12031[0] - 1, tmp_12031[1], tmp_12031[2]];
    signal tmp_12033[3] <== [tmp_12032[0] + 1, tmp_12032[1], tmp_12032[2]];
    signal tmp_12034[3] <== CMul()(tmp_12026, tmp_12033);
    signal tmp_12035[3] <== CMul()(evals[22], challengesStage2[0]);
    signal tmp_12036[3] <== [tmp_12035[0] + evals[67][0], tmp_12035[1] + evals[67][1], tmp_12035[2] + evals[67][2]];
    signal tmp_12037[3] <== CMul()(tmp_12036, challengesStage2[0]);
    signal tmp_12038[3] <== [tmp_12037[0] + 1, tmp_12037[1], tmp_12037[2]];
    signal tmp_12039[3] <== [tmp_12038[0] + challengesStage2[1][0], tmp_12038[1] + challengesStage2[1][1], tmp_12038[2] + challengesStage2[1][2]];
    signal tmp_12040[3] <== [tmp_12039[0] - 1, tmp_12039[1], tmp_12039[2]];
    signal tmp_12041[3] <== [tmp_12040[0] + 1, tmp_12040[1], tmp_12040[2]];
    signal tmp_12042[3] <== CMul()(tmp_12034, tmp_12041);
    signal tmp_12043[3] <== CMul()(evals[23], challengesStage2[0]);
    signal tmp_12044[3] <== [tmp_12043[0] + evals[68][0], tmp_12043[1] + evals[68][1], tmp_12043[2] + evals[68][2]];
    signal tmp_12045[3] <== CMul()(tmp_12044, challengesStage2[0]);
    signal tmp_12046[3] <== [tmp_12045[0] + 1, tmp_12045[1], tmp_12045[2]];
    signal tmp_12047[3] <== [tmp_12046[0] + challengesStage2[1][0], tmp_12046[1] + challengesStage2[1][1], tmp_12046[2] + challengesStage2[1][2]];
    signal tmp_12048[3] <== [tmp_12047[0] - 1, tmp_12047[1], tmp_12047[2]];
    signal tmp_12049[3] <== [tmp_12048[0] + 1, tmp_12048[1], tmp_12048[2]];
    signal tmp_12050[3] <== CMul()(tmp_12042, tmp_12049);
    signal tmp_12051[3] <== CMul()(evals[24], challengesStage2[0]);
    signal tmp_12052[3] <== [tmp_12051[0] + evals[69][0], tmp_12051[1] + evals[69][1], tmp_12051[2] + evals[69][2]];
    signal tmp_12053[3] <== CMul()(tmp_12052, challengesStage2[0]);
    signal tmp_12054[3] <== [tmp_12053[0] + 1, tmp_12053[1], tmp_12053[2]];
    signal tmp_12055[3] <== [tmp_12054[0] + challengesStage2[1][0], tmp_12054[1] + challengesStage2[1][1], tmp_12054[2] + challengesStage2[1][2]];
    signal tmp_12056[3] <== [tmp_12055[0] - 1, tmp_12055[1], tmp_12055[2]];
    signal tmp_12057[3] <== [tmp_12056[0] + 1, tmp_12056[1], tmp_12056[2]];
    tmp_12058 <== CMul()(tmp_12050, tmp_12057);
    tmp_12059 <== CMul()(evals[25], challengesStage2[0]);
}

template VerifyEvaluationsChunks6() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];

    signal input Zh[3];

    signal input tmp_11955[3];
    signal input tmp_12018[3];
    signal input tmp_12058[3];
    signal input tmp_12059[3];

    signal output tmp_12130[3];
    signal tmp_12060[3] <== [tmp_12059[0] + evals[70][0], tmp_12059[1] + evals[70][1], tmp_12059[2] + evals[70][2]];
    signal tmp_12061[3] <== CMul()(tmp_12060, challengesStage2[0]);
    signal tmp_12062[3] <== [tmp_12061[0] + 1, tmp_12061[1], tmp_12061[2]];
    signal tmp_12063[3] <== [tmp_12062[0] + challengesStage2[1][0], tmp_12062[1] + challengesStage2[1][1], tmp_12062[2] + challengesStage2[1][2]];
    signal tmp_12064[3] <== [tmp_12063[0] - 1, tmp_12063[1], tmp_12063[2]];
    signal tmp_12065[3] <== [tmp_12064[0] + 1, tmp_12064[1], tmp_12064[2]];
    signal tmp_12066[3] <== CMul()(tmp_12058, tmp_12065);
    signal tmp_12067[3] <== CMul()(evals[26], challengesStage2[0]);
    signal tmp_12068[3] <== [tmp_12067[0] + evals[71][0], tmp_12067[1] + evals[71][1], tmp_12067[2] + evals[71][2]];
    signal tmp_12069[3] <== CMul()(tmp_12068, challengesStage2[0]);
    signal tmp_12070[3] <== [tmp_12069[0] + 1, tmp_12069[1], tmp_12069[2]];
    signal tmp_12071[3] <== [tmp_12070[0] + challengesStage2[1][0], tmp_12070[1] + challengesStage2[1][1], tmp_12070[2] + challengesStage2[1][2]];
    signal tmp_12072[3] <== [tmp_12071[0] - 1, tmp_12071[1], tmp_12071[2]];
    signal tmp_12073[3] <== [tmp_12072[0] + 1, tmp_12072[1], tmp_12072[2]];
    signal tmp_12074[3] <== CMul()(tmp_12066, tmp_12073);
    signal tmp_12075[3] <== [tmp_12018[0] - tmp_12074[0], tmp_12018[1] - tmp_12074[1], tmp_12018[2] - tmp_12074[2]];
    signal tmp_12076[3] <== [tmp_11955[0] + tmp_12075[0], tmp_11955[1] + tmp_12075[1], tmp_11955[2] + tmp_12075[2]];
    signal tmp_12077[3] <== CMul()(challengeQ, tmp_12076);
    signal tmp_12078[3] <== [8246665031048405574 * evals[48][0], 8246665031048405574 * evals[48][1], 8246665031048405574 * evals[48][2]];
    signal tmp_12079[3] <== CMul()(tmp_12078, challengesStage2[0]);
    signal tmp_12080[3] <== [tmp_12079[0] + evals[71][0], tmp_12079[1] + evals[71][1], tmp_12079[2] + evals[71][2]];
    signal tmp_12081[3] <== CMul()(tmp_12080, challengesStage2[0]);
    signal tmp_12082[3] <== [tmp_12081[0] + 1, tmp_12081[1], tmp_12081[2]];
    signal tmp_12083[3] <== [tmp_12082[0] + challengesStage2[1][0], tmp_12082[1] + challengesStage2[1][1], tmp_12082[2] + challengesStage2[1][2]];
    signal tmp_12084[3] <== [tmp_12083[0] - 1, tmp_12083[1], tmp_12083[2]];
    signal tmp_12085[3] <== [tmp_12084[0] + 1, tmp_12084[1], tmp_12084[2]];
    signal tmp_12086[3] <== [12693612801792047873 * evals[48][0], 12693612801792047873 * evals[48][1], 12693612801792047873 * evals[48][2]];
    signal tmp_12087[3] <== CMul()(tmp_12086, challengesStage2[0]);
    signal tmp_12088[3] <== [tmp_12087[0] + evals[72][0], tmp_12087[1] + evals[72][1], tmp_12087[2] + evals[72][2]];
    signal tmp_12089[3] <== CMul()(tmp_12088, challengesStage2[0]);
    signal tmp_12090[3] <== [tmp_12089[0] + 1, tmp_12089[1], tmp_12089[2]];
    signal tmp_12091[3] <== [tmp_12090[0] + challengesStage2[1][0], tmp_12090[1] + challengesStage2[1][1], tmp_12090[2] + challengesStage2[1][2]];
    signal tmp_12092[3] <== [tmp_12091[0] - 1, tmp_12091[1], tmp_12091[2]];
    signal tmp_12093[3] <== [tmp_12092[0] + 1, tmp_12092[1], tmp_12092[2]];
    signal tmp_12094[3] <== CMul()(tmp_12085, tmp_12093);
    signal tmp_12095[3] <== [9404062091095256088 * evals[48][0], 9404062091095256088 * evals[48][1], 9404062091095256088 * evals[48][2]];
    signal tmp_12096[3] <== CMul()(tmp_12095, challengesStage2[0]);
    signal tmp_12097[3] <== [tmp_12096[0] + evals[73][0], tmp_12096[1] + evals[73][1], tmp_12096[2] + evals[73][2]];
    signal tmp_12098[3] <== CMul()(tmp_12097, challengesStage2[0]);
    signal tmp_12099[3] <== [tmp_12098[0] + 1, tmp_12098[1], tmp_12098[2]];
    signal tmp_12100[3] <== [tmp_12099[0] + challengesStage2[1][0], tmp_12099[1] + challengesStage2[1][1], tmp_12099[2] + challengesStage2[1][2]];
    signal tmp_12101[3] <== [tmp_12100[0] - 1, tmp_12100[1], tmp_12100[2]];
    signal tmp_12102[3] <== [tmp_12101[0] + 1, tmp_12101[1], tmp_12101[2]];
    signal tmp_12103[3] <== CMul()(tmp_12094, tmp_12102);
    signal tmp_12104[3] <== CMul()(evals[98], tmp_12103);
    signal tmp_12105[3] <== [1 - evals[49][0], -evals[49][1], -evals[49][2]];
    signal tmp_12106[3] <== CMul()(evals[4], tmp_12105);
    signal tmp_12107[3] <== [tmp_12106[0] + evals[49][0], tmp_12106[1] + evals[49][1], tmp_12106[2] + evals[49][2]];
    signal tmp_12108[3] <== CMul()(evals[27], challengesStage2[0]);
    signal tmp_12109[3] <== [tmp_12108[0] + evals[72][0], tmp_12108[1] + evals[72][1], tmp_12108[2] + evals[72][2]];
    signal tmp_12110[3] <== CMul()(tmp_12109, challengesStage2[0]);
    signal tmp_12111[3] <== [tmp_12110[0] + 1, tmp_12110[1], tmp_12110[2]];
    signal tmp_12112[3] <== [tmp_12111[0] + challengesStage2[1][0], tmp_12111[1] + challengesStage2[1][1], tmp_12111[2] + challengesStage2[1][2]];
    signal tmp_12113[3] <== [tmp_12112[0] - 1, tmp_12112[1], tmp_12112[2]];
    signal tmp_12114[3] <== [tmp_12113[0] + 1, tmp_12113[1], tmp_12113[2]];
    signal tmp_12115[3] <== CMul()(evals[101], tmp_12114);
    signal tmp_12116[3] <== CMul()(evals[28], challengesStage2[0]);
    signal tmp_12117[3] <== [tmp_12116[0] + evals[73][0], tmp_12116[1] + evals[73][1], tmp_12116[2] + evals[73][2]];
    signal tmp_12118[3] <== CMul()(tmp_12117, challengesStage2[0]);
    signal tmp_12119[3] <== [tmp_12118[0] + 1, tmp_12118[1], tmp_12118[2]];
    signal tmp_12120[3] <== [tmp_12119[0] + challengesStage2[1][0], tmp_12119[1] + challengesStage2[1][1], tmp_12119[2] + challengesStage2[1][2]];
    signal tmp_12121[3] <== [tmp_12120[0] - 1, tmp_12120[1], tmp_12120[2]];
    signal tmp_12122[3] <== [tmp_12121[0] + 1, tmp_12121[1], tmp_12121[2]];
    signal tmp_12123[3] <== CMul()(tmp_12115, tmp_12122);
    signal tmp_12124[3] <== CMul()(tmp_12107, tmp_12123);
    signal tmp_12125[3] <== [tmp_12104[0] - tmp_12124[0], tmp_12104[1] - tmp_12124[1], tmp_12104[2] - tmp_12124[2]];
    signal tmp_12126[3] <== [tmp_12077[0] + tmp_12125[0], tmp_12077[1] + tmp_12125[1], tmp_12077[2] + tmp_12125[2]];
    signal tmp_12127[3] <== CMul()(challengeQ, tmp_12126);
    signal tmp_12128[3] <== [1 - evals[98][0], -evals[98][1], -evals[98][2]];
    signal tmp_12129[3] <== CMul()(evals[110], tmp_12128);
    signal tmp_6064[3] <== [tmp_12127[0] + tmp_12129[0], tmp_12127[1] + tmp_12129[1], tmp_12127[2] + tmp_12129[2]];
    tmp_12130 <== CMul()(tmp_6064, Zh);
}


template parallel VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[135][3];
    signal input publics[395];
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
    signal tmp_6066[3];
    signal tmp_6068[3];
    signal tmp_6405[3];
    signal tmp_6414[3];
    signal tmp_6421[3];
    signal tmp_6428[3];
    signal tmp_6435[3];
    signal tmp_6443[3];
    signal tmp_6450[3];
    signal tmp_6459[3];
    signal tmp_6466[3];
    signal tmp_6474[3];
    signal tmp_6481[3];
    signal tmp_6490[3];
    signal tmp_6497[3];
    signal tmp_6506[3];
    signal tmp_6513[3];
    signal tmp_6522[3];
    signal tmp_6529[3];
    signal tmp_6537[3];
    signal tmp_6544[3];
    signal tmp_6553[3];
    signal tmp_6560[3];
    signal tmp_6569[3];
    signal tmp_6576[3];
    signal tmp_6585[3];
    signal tmp_6592[3];
    signal tmp_6601[3];
    signal tmp_6608[3];
    signal tmp_6617[3];
    signal tmp_6624[3];
    signal tmp_6633[3];
    signal tmp_6640[3];
    signal tmp_6649[3];
    signal tmp_6656[3];
    signal tmp_7063[3];
    signal tmp_7064[3];
    (tmp_6066,tmp_6068,tmp_6405,tmp_6414,tmp_6421,tmp_6428,tmp_6435,tmp_6443,tmp_6450,tmp_6459,tmp_6466,tmp_6474,tmp_6481,tmp_6490,tmp_6497,tmp_6506,tmp_6513,tmp_6522,tmp_6529,tmp_6537,tmp_6544,tmp_6553,tmp_6560,tmp_6569,tmp_6576,tmp_6585,tmp_6592,tmp_6601,tmp_6608,tmp_6617,tmp_6624,tmp_6633,tmp_6640,tmp_6649,tmp_6656,tmp_7063,tmp_7064) <== VerifyEvaluationsChunks0()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh);
    signal tmp_7423[3];
    signal tmp_7428[3];
    signal tmp_7433[3];
    signal tmp_7438[3];
    signal tmp_7444[3];
    signal tmp_7449[3];
    signal tmp_7456[3];
    signal tmp_7461[3];
    signal tmp_7467[3];
    signal tmp_7472[3];
    signal tmp_7479[3];
    signal tmp_7484[3];
    signal tmp_7491[3];
    signal tmp_7496[3];
    signal tmp_7503[3];
    signal tmp_7508[3];
    signal tmp_7514[3];
    signal tmp_7519[3];
    signal tmp_7526[3];
    signal tmp_7531[3];
    signal tmp_7538[3];
    signal tmp_7543[3];
    signal tmp_7550[3];
    signal tmp_7555[3];
    signal tmp_7562[3];
    signal tmp_7567[3];
    signal tmp_7574[3];
    signal tmp_7579[3];
    signal tmp_7586[3];
    signal tmp_7591[3];
    signal tmp_7598[3];
    signal tmp_7603[3];
    signal tmp_8034[3];
    signal tmp_8036[3];
    signal tmp_8040[3];
    signal tmp_8062[3];
    signal tmp_8063[3];
    (tmp_7423,tmp_7428,tmp_7433,tmp_7438,tmp_7444,tmp_7449,tmp_7456,tmp_7461,tmp_7467,tmp_7472,tmp_7479,tmp_7484,tmp_7491,tmp_7496,tmp_7503,tmp_7508,tmp_7514,tmp_7519,tmp_7526,tmp_7531,tmp_7538,tmp_7543,tmp_7550,tmp_7555,tmp_7562,tmp_7567,tmp_7574,tmp_7579,tmp_7586,tmp_7591,tmp_7598,tmp_7603,tmp_8034,tmp_8036,tmp_8040,tmp_8062,tmp_8063) <== VerifyEvaluationsChunks1()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_6066,tmp_6405,tmp_6414,tmp_6421,tmp_6428,tmp_6435,tmp_6443,tmp_6450,tmp_6459,tmp_6466,tmp_6474,tmp_6481,tmp_6490,tmp_6497,tmp_6506,tmp_6513,tmp_6522,tmp_6529,tmp_6537,tmp_6544,tmp_6553,tmp_6560,tmp_6569,tmp_6576,tmp_6585,tmp_6592,tmp_6601,tmp_6608,tmp_6617,tmp_6624,tmp_6633,tmp_6640,tmp_6649,tmp_6656,tmp_7063,tmp_7064);
    signal tmp_9013[3];
    signal tmp_9062[3];
    (tmp_9013,tmp_9062) <== VerifyEvaluationsChunks2()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_6066,tmp_6405,tmp_7423,tmp_7428,tmp_7433,tmp_7438,tmp_7444,tmp_7449,tmp_7456,tmp_7461,tmp_7467,tmp_7472,tmp_7479,tmp_7484,tmp_7491,tmp_7496,tmp_7503,tmp_7508,tmp_7514,tmp_7519,tmp_7526,tmp_7531,tmp_7538,tmp_7543,tmp_7550,tmp_7555,tmp_7562,tmp_7567,tmp_7574,tmp_7579,tmp_7586,tmp_7591,tmp_7598,tmp_7603,tmp_8034,tmp_8036,tmp_8040,tmp_8062,tmp_8063);
    signal tmp_9986[3];
    signal tmp_9990[3];
    signal tmp_9995[3];
    signal tmp_10000[3];
    signal tmp_10005[3];
    signal tmp_10010[3];
    signal tmp_10015[3];
    signal tmp_10020[3];
    signal tmp_10025[3];
    signal tmp_10030[3];
    signal tmp_10035[3];
    signal tmp_10040[3];
    signal tmp_10045[3];
    signal tmp_10050[3];
    signal tmp_10055[3];
    signal tmp_10057[3];
    signal tmp_10060[3];
    signal tmp_10061[3];
    (tmp_9986,tmp_9990,tmp_9995,tmp_10000,tmp_10005,tmp_10010,tmp_10015,tmp_10020,tmp_10025,tmp_10030,tmp_10035,tmp_10040,tmp_10045,tmp_10050,tmp_10055,tmp_10057,tmp_10060,tmp_10061) <== VerifyEvaluationsChunks3()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_6066,tmp_6068,tmp_7423,tmp_7428,tmp_7433,tmp_7438,tmp_7444,tmp_7449,tmp_7456,tmp_7461,tmp_7467,tmp_7472,tmp_7479,tmp_7484,tmp_7491,tmp_7496,tmp_7503,tmp_7508,tmp_7514,tmp_7519,tmp_7526,tmp_7531,tmp_7538,tmp_7543,tmp_7550,tmp_7555,tmp_7562,tmp_7567,tmp_7574,tmp_7579,tmp_7586,tmp_7591,tmp_7598,tmp_7603,tmp_9013,tmp_9062);
    signal tmp_10901[3];
    signal tmp_10955[3];
    signal tmp_10960[3];
    signal tmp_10965[3];
    signal tmp_10970[3];
    signal tmp_10975[3];
    signal tmp_11060[3];
    (tmp_10901,tmp_10955,tmp_10960,tmp_10965,tmp_10970,tmp_10975,tmp_11060) <== VerifyEvaluationsChunks4()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_6068,tmp_7538,tmp_7550,tmp_7562,tmp_7574,tmp_7586,tmp_7598,tmp_9986,tmp_9990,tmp_9995,tmp_10000,tmp_10005,tmp_10010,tmp_10015,tmp_10020,tmp_10025,tmp_10030,tmp_10035,tmp_10040,tmp_10045,tmp_10050,tmp_10055,tmp_10057,tmp_10060,tmp_10061);
    signal tmp_11955[3];
    signal tmp_12018[3];
    signal tmp_12058[3];
    signal tmp_12059[3];
    (tmp_11955,tmp_12018,tmp_12058,tmp_12059) <== VerifyEvaluationsChunks5()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_6068,tmp_10901,tmp_10955,tmp_10960,tmp_10965,tmp_10970,tmp_10975,tmp_11060);
    signal tmp_12130[3];
    (tmp_12130) <== VerifyEvaluationsChunks6()(challengesStage2,challengeQ,challengeXi,evals,publics,Zh,tmp_11955,tmp_12018,tmp_12058,tmp_12059);

    signal xAcc[7][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[6][3]; // Stores the evaluations of Q_i
    signal qAcc[7][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 7; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[102+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[16]);
            qStep[i-1] <== CMul()(xAcc[i], evals[102+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i
    enable * (tmp_12130[0] - qAcc[6][0]) === 0;
    enable * (tmp_12130[1] - qAcc[6][1]) === 0;
    enable * (tmp_12130[2] - qAcc[6][2]) === 0;
}

template CalculateFRIPolChunks0() {
    signal input challengesFRI[2][3];
    signal input evals[135][3];

    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[45];

    signal input xDivXSubXi[5][3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
    mapValues.vals1 <== cm1;
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;


    signal output tmp_408[3];
    signal tmp_0[3] <== [consts[34] - evals[0][0], -evals[0][1], -evals[0][2]];
    signal tmp_1[3] <== CMul()(tmp_0, challengesFRI[1]);
    signal tmp_2[3] <== [consts[35] - evals[1][0], -evals[1][1], -evals[1][2]];
    signal tmp_3[3] <== [tmp_1[0] + tmp_2[0], tmp_1[1] + tmp_2[1], tmp_1[2] + tmp_2[2]];
    signal tmp_4[3] <== CMul()(tmp_3, xDivXSubXi[0]);
    signal tmp_5[3] <== CMul()(challengesFRI[0], tmp_4);
    signal tmp_6[3] <== [consts[34] - evals[2][0], -evals[2][1], -evals[2][2]];
    signal tmp_7[3] <== CMul()(tmp_6, challengesFRI[1]);
    signal tmp_8[3] <== [consts[35] - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_9[3] <== [tmp_7[0] + tmp_8[0], tmp_7[1] + tmp_8[1], tmp_7[2] + tmp_8[2]];
    signal tmp_10[3] <== CMul()(tmp_9, challengesFRI[1]);
    signal tmp_11[3] <== [mapValues.cm2_0[0] - evals[4][0], mapValues.cm2_0[1] - evals[4][1], mapValues.cm2_0[2] - evals[4][2]];
    signal tmp_12[3] <== [tmp_10[0] + tmp_11[0], tmp_10[1] + tmp_11[1], tmp_10[2] + tmp_11[2]];
    signal tmp_13[3] <== CMul()(tmp_12, xDivXSubXi[1]);
    signal tmp_14[3] <== [tmp_5[0] + tmp_13[0], tmp_5[1] + tmp_13[1], tmp_5[2] + tmp_13[2]];
    signal tmp_15[3] <== CMul()(challengesFRI[0], tmp_14);
    signal tmp_16[3] <== [consts[0] - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_17[3] <== CMul()(tmp_16, challengesFRI[1]);
    signal tmp_18[3] <== [consts[1] - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_19[3] <== [tmp_17[0] + tmp_18[0], tmp_17[1] + tmp_18[1], tmp_17[2] + tmp_18[2]];
    signal tmp_20[3] <== CMul()(tmp_19, challengesFRI[1]);
    signal tmp_21[3] <== [consts[2] - evals[7][0], -evals[7][1], -evals[7][2]];
    signal tmp_22[3] <== [tmp_20[0] + tmp_21[0], tmp_20[1] + tmp_21[1], tmp_20[2] + tmp_21[2]];
    signal tmp_23[3] <== CMul()(tmp_22, challengesFRI[1]);
    signal tmp_24[3] <== [consts[3] - evals[8][0], -evals[8][1], -evals[8][2]];
    signal tmp_25[3] <== [tmp_23[0] + tmp_24[0], tmp_23[1] + tmp_24[1], tmp_23[2] + tmp_24[2]];
    signal tmp_26[3] <== CMul()(tmp_25, challengesFRI[1]);
    signal tmp_27[3] <== [consts[4] - evals[9][0], -evals[9][1], -evals[9][2]];
    signal tmp_28[3] <== [tmp_26[0] + tmp_27[0], tmp_26[1] + tmp_27[1], tmp_26[2] + tmp_27[2]];
    signal tmp_29[3] <== CMul()(tmp_28, challengesFRI[1]);
    signal tmp_30[3] <== [consts[5] - evals[10][0], -evals[10][1], -evals[10][2]];
    signal tmp_31[3] <== [tmp_29[0] + tmp_30[0], tmp_29[1] + tmp_30[1], tmp_29[2] + tmp_30[2]];
    signal tmp_32[3] <== CMul()(tmp_31, challengesFRI[1]);
    signal tmp_33[3] <== [consts[6] - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_34[3] <== [tmp_32[0] + tmp_33[0], tmp_32[1] + tmp_33[1], tmp_32[2] + tmp_33[2]];
    signal tmp_35[3] <== CMul()(tmp_34, challengesFRI[1]);
    signal tmp_36[3] <== [consts[7] - evals[12][0], -evals[12][1], -evals[12][2]];
    signal tmp_37[3] <== [tmp_35[0] + tmp_36[0], tmp_35[1] + tmp_36[1], tmp_35[2] + tmp_36[2]];
    signal tmp_38[3] <== CMul()(tmp_37, challengesFRI[1]);
    signal tmp_39[3] <== [consts[8] - evals[13][0], -evals[13][1], -evals[13][2]];
    signal tmp_40[3] <== [tmp_38[0] + tmp_39[0], tmp_38[1] + tmp_39[1], tmp_38[2] + tmp_39[2]];
    signal tmp_41[3] <== CMul()(tmp_40, challengesFRI[1]);
    signal tmp_42[3] <== [consts[9] - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_43[3] <== [tmp_41[0] + tmp_42[0], tmp_41[1] + tmp_42[1], tmp_41[2] + tmp_42[2]];
    signal tmp_44[3] <== CMul()(tmp_43, challengesFRI[1]);
    signal tmp_45[3] <== [consts[10] - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_46[3] <== [tmp_44[0] + tmp_45[0], tmp_44[1] + tmp_45[1], tmp_44[2] + tmp_45[2]];
    signal tmp_47[3] <== CMul()(tmp_46, challengesFRI[1]);
    signal tmp_48[3] <== [consts[11] - evals[16][0], -evals[16][1], -evals[16][2]];
    signal tmp_49[3] <== [tmp_47[0] + tmp_48[0], tmp_47[1] + tmp_48[1], tmp_47[2] + tmp_48[2]];
    signal tmp_50[3] <== CMul()(tmp_49, challengesFRI[1]);
    signal tmp_51[3] <== [consts[12] - evals[17][0], -evals[17][1], -evals[17][2]];
    signal tmp_52[3] <== [tmp_50[0] + tmp_51[0], tmp_50[1] + tmp_51[1], tmp_50[2] + tmp_51[2]];
    signal tmp_53[3] <== CMul()(tmp_52, challengesFRI[1]);
    signal tmp_54[3] <== [consts[13] - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_55[3] <== [tmp_53[0] + tmp_54[0], tmp_53[1] + tmp_54[1], tmp_53[2] + tmp_54[2]];
    signal tmp_56[3] <== CMul()(tmp_55, challengesFRI[1]);
    signal tmp_57[3] <== [consts[14] - evals[19][0], -evals[19][1], -evals[19][2]];
    signal tmp_58[3] <== [tmp_56[0] + tmp_57[0], tmp_56[1] + tmp_57[1], tmp_56[2] + tmp_57[2]];
    signal tmp_59[3] <== CMul()(tmp_58, challengesFRI[1]);
    signal tmp_60[3] <== [consts[15] - evals[20][0], -evals[20][1], -evals[20][2]];
    signal tmp_61[3] <== [tmp_59[0] + tmp_60[0], tmp_59[1] + tmp_60[1], tmp_59[2] + tmp_60[2]];
    signal tmp_62[3] <== CMul()(tmp_61, challengesFRI[1]);
    signal tmp_63[3] <== [consts[16] - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_64[3] <== [tmp_62[0] + tmp_63[0], tmp_62[1] + tmp_63[1], tmp_62[2] + tmp_63[2]];
    signal tmp_65[3] <== CMul()(tmp_64, challengesFRI[1]);
    signal tmp_66[3] <== [consts[17] - evals[22][0], -evals[22][1], -evals[22][2]];
    signal tmp_67[3] <== [tmp_65[0] + tmp_66[0], tmp_65[1] + tmp_66[1], tmp_65[2] + tmp_66[2]];
    signal tmp_68[3] <== CMul()(tmp_67, challengesFRI[1]);
    signal tmp_69[3] <== [consts[18] - evals[23][0], -evals[23][1], -evals[23][2]];
    signal tmp_70[3] <== [tmp_68[0] + tmp_69[0], tmp_68[1] + tmp_69[1], tmp_68[2] + tmp_69[2]];
    signal tmp_71[3] <== CMul()(tmp_70, challengesFRI[1]);
    signal tmp_72[3] <== [consts[19] - evals[24][0], -evals[24][1], -evals[24][2]];
    signal tmp_73[3] <== [tmp_71[0] + tmp_72[0], tmp_71[1] + tmp_72[1], tmp_71[2] + tmp_72[2]];
    signal tmp_74[3] <== CMul()(tmp_73, challengesFRI[1]);
    signal tmp_75[3] <== [consts[20] - evals[25][0], -evals[25][1], -evals[25][2]];
    signal tmp_76[3] <== [tmp_74[0] + tmp_75[0], tmp_74[1] + tmp_75[1], tmp_74[2] + tmp_75[2]];
    signal tmp_77[3] <== CMul()(tmp_76, challengesFRI[1]);
    signal tmp_78[3] <== [consts[21] - evals[26][0], -evals[26][1], -evals[26][2]];
    signal tmp_79[3] <== [tmp_77[0] + tmp_78[0], tmp_77[1] + tmp_78[1], tmp_77[2] + tmp_78[2]];
    signal tmp_80[3] <== CMul()(tmp_79, challengesFRI[1]);
    signal tmp_81[3] <== [consts[22] - evals[27][0], -evals[27][1], -evals[27][2]];
    signal tmp_82[3] <== [tmp_80[0] + tmp_81[0], tmp_80[1] + tmp_81[1], tmp_80[2] + tmp_81[2]];
    signal tmp_83[3] <== CMul()(tmp_82, challengesFRI[1]);
    signal tmp_84[3] <== [consts[23] - evals[28][0], -evals[28][1], -evals[28][2]];
    signal tmp_85[3] <== [tmp_83[0] + tmp_84[0], tmp_83[1] + tmp_84[1], tmp_83[2] + tmp_84[2]];
    signal tmp_86[3] <== CMul()(tmp_85, challengesFRI[1]);
    signal tmp_87[3] <== [consts[24] - evals[29][0], -evals[29][1], -evals[29][2]];
    signal tmp_88[3] <== [tmp_86[0] + tmp_87[0], tmp_86[1] + tmp_87[1], tmp_86[2] + tmp_87[2]];
    signal tmp_89[3] <== CMul()(tmp_88, challengesFRI[1]);
    signal tmp_90[3] <== [consts[25] - evals[30][0], -evals[30][1], -evals[30][2]];
    signal tmp_91[3] <== [tmp_89[0] + tmp_90[0], tmp_89[1] + tmp_90[1], tmp_89[2] + tmp_90[2]];
    signal tmp_92[3] <== CMul()(tmp_91, challengesFRI[1]);
    signal tmp_93[3] <== [consts[26] - evals[31][0], -evals[31][1], -evals[31][2]];
    signal tmp_94[3] <== [tmp_92[0] + tmp_93[0], tmp_92[1] + tmp_93[1], tmp_92[2] + tmp_93[2]];
    signal tmp_95[3] <== CMul()(tmp_94, challengesFRI[1]);
    signal tmp_96[3] <== [consts[27] - evals[32][0], -evals[32][1], -evals[32][2]];
    signal tmp_97[3] <== [tmp_95[0] + tmp_96[0], tmp_95[1] + tmp_96[1], tmp_95[2] + tmp_96[2]];
    signal tmp_98[3] <== CMul()(tmp_97, challengesFRI[1]);
    signal tmp_99[3] <== [consts[28] - evals[33][0], -evals[33][1], -evals[33][2]];
    signal tmp_100[3] <== [tmp_98[0] + tmp_99[0], tmp_98[1] + tmp_99[1], tmp_98[2] + tmp_99[2]];
    signal tmp_101[3] <== CMul()(tmp_100, challengesFRI[1]);
    signal tmp_102[3] <== [consts[29] - evals[34][0], -evals[34][1], -evals[34][2]];
    signal tmp_103[3] <== [tmp_101[0] + tmp_102[0], tmp_101[1] + tmp_102[1], tmp_101[2] + tmp_102[2]];
    signal tmp_104[3] <== CMul()(tmp_103, challengesFRI[1]);
    signal tmp_105[3] <== [consts[30] - evals[35][0], -evals[35][1], -evals[35][2]];
    signal tmp_106[3] <== [tmp_104[0] + tmp_105[0], tmp_104[1] + tmp_105[1], tmp_104[2] + tmp_105[2]];
    signal tmp_107[3] <== CMul()(tmp_106, challengesFRI[1]);
    signal tmp_108[3] <== [consts[31] - evals[36][0], -evals[36][1], -evals[36][2]];
    signal tmp_109[3] <== [tmp_107[0] + tmp_108[0], tmp_107[1] + tmp_108[1], tmp_107[2] + tmp_108[2]];
    signal tmp_110[3] <== CMul()(tmp_109, challengesFRI[1]);
    signal tmp_111[3] <== [consts[32] - evals[37][0], -evals[37][1], -evals[37][2]];
    signal tmp_112[3] <== [tmp_110[0] + tmp_111[0], tmp_110[1] + tmp_111[1], tmp_110[2] + tmp_111[2]];
    signal tmp_113[3] <== CMul()(tmp_112, challengesFRI[1]);
    signal tmp_114[3] <== [consts[33] - evals[38][0], -evals[38][1], -evals[38][2]];
    signal tmp_115[3] <== [tmp_113[0] + tmp_114[0], tmp_113[1] + tmp_114[1], tmp_113[2] + tmp_114[2]];
    signal tmp_116[3] <== CMul()(tmp_115, challengesFRI[1]);
    signal tmp_117[3] <== [consts[34] - evals[39][0], -evals[39][1], -evals[39][2]];
    signal tmp_118[3] <== [tmp_116[0] + tmp_117[0], tmp_116[1] + tmp_117[1], tmp_116[2] + tmp_117[2]];
    signal tmp_119[3] <== CMul()(tmp_118, challengesFRI[1]);
    signal tmp_120[3] <== [consts[35] - evals[40][0], -evals[40][1], -evals[40][2]];
    signal tmp_121[3] <== [tmp_119[0] + tmp_120[0], tmp_119[1] + tmp_120[1], tmp_119[2] + tmp_120[2]];
    signal tmp_122[3] <== CMul()(tmp_121, challengesFRI[1]);
    signal tmp_123[3] <== [consts[36] - evals[41][0], -evals[41][1], -evals[41][2]];
    signal tmp_124[3] <== [tmp_122[0] + tmp_123[0], tmp_122[1] + tmp_123[1], tmp_122[2] + tmp_123[2]];
    signal tmp_125[3] <== CMul()(tmp_124, challengesFRI[1]);
    signal tmp_126[3] <== [consts[37] - evals[42][0], -evals[42][1], -evals[42][2]];
    signal tmp_127[3] <== [tmp_125[0] + tmp_126[0], tmp_125[1] + tmp_126[1], tmp_125[2] + tmp_126[2]];
    signal tmp_128[3] <== CMul()(tmp_127, challengesFRI[1]);
    signal tmp_129[3] <== [consts[38] - evals[43][0], -evals[43][1], -evals[43][2]];
    signal tmp_130[3] <== [tmp_128[0] + tmp_129[0], tmp_128[1] + tmp_129[1], tmp_128[2] + tmp_129[2]];
    signal tmp_131[3] <== CMul()(tmp_130, challengesFRI[1]);
    signal tmp_132[3] <== [consts[39] - evals[44][0], -evals[44][1], -evals[44][2]];
    signal tmp_133[3] <== [tmp_131[0] + tmp_132[0], tmp_131[1] + tmp_132[1], tmp_131[2] + tmp_132[2]];
    signal tmp_134[3] <== CMul()(tmp_133, challengesFRI[1]);
    signal tmp_135[3] <== [consts[40] - evals[45][0], -evals[45][1], -evals[45][2]];
    signal tmp_136[3] <== [tmp_134[0] + tmp_135[0], tmp_134[1] + tmp_135[1], tmp_134[2] + tmp_135[2]];
    signal tmp_137[3] <== CMul()(tmp_136, challengesFRI[1]);
    signal tmp_138[3] <== [consts[41] - evals[46][0], -evals[46][1], -evals[46][2]];
    signal tmp_139[3] <== [tmp_137[0] + tmp_138[0], tmp_137[1] + tmp_138[1], tmp_137[2] + tmp_138[2]];
    signal tmp_140[3] <== CMul()(tmp_139, challengesFRI[1]);
    signal tmp_141[3] <== [consts[42] - evals[47][0], -evals[47][1], -evals[47][2]];
    signal tmp_142[3] <== [tmp_140[0] + tmp_141[0], tmp_140[1] + tmp_141[1], tmp_140[2] + tmp_141[2]];
    signal tmp_143[3] <== CMul()(tmp_142, challengesFRI[1]);
    signal tmp_144[3] <== [consts[43] - evals[48][0], -evals[48][1], -evals[48][2]];
    signal tmp_145[3] <== [tmp_143[0] + tmp_144[0], tmp_143[1] + tmp_144[1], tmp_143[2] + tmp_144[2]];
    signal tmp_146[3] <== CMul()(tmp_145, challengesFRI[1]);
    signal tmp_147[3] <== [consts[44] - evals[49][0], -evals[49][1], -evals[49][2]];
    signal tmp_148[3] <== [tmp_146[0] + tmp_147[0], tmp_146[1] + tmp_147[1], tmp_146[2] + tmp_147[2]];
    signal tmp_149[3] <== CMul()(tmp_148, challengesFRI[1]);
    signal tmp_150[3] <== [mapValues.cm1_0 - evals[50][0], -evals[50][1], -evals[50][2]];
    signal tmp_151[3] <== [tmp_149[0] + tmp_150[0], tmp_149[1] + tmp_150[1], tmp_149[2] + tmp_150[2]];
    signal tmp_152[3] <== CMul()(tmp_151, challengesFRI[1]);
    signal tmp_153[3] <== [mapValues.cm1_1 - evals[51][0], -evals[51][1], -evals[51][2]];
    signal tmp_154[3] <== [tmp_152[0] + tmp_153[0], tmp_152[1] + tmp_153[1], tmp_152[2] + tmp_153[2]];
    signal tmp_155[3] <== CMul()(tmp_154, challengesFRI[1]);
    signal tmp_156[3] <== [mapValues.cm1_2 - evals[52][0], -evals[52][1], -evals[52][2]];
    signal tmp_157[3] <== [tmp_155[0] + tmp_156[0], tmp_155[1] + tmp_156[1], tmp_155[2] + tmp_156[2]];
    signal tmp_158[3] <== CMul()(tmp_157, challengesFRI[1]);
    signal tmp_159[3] <== [mapValues.cm1_3 - evals[53][0], -evals[53][1], -evals[53][2]];
    signal tmp_160[3] <== [tmp_158[0] + tmp_159[0], tmp_158[1] + tmp_159[1], tmp_158[2] + tmp_159[2]];
    signal tmp_161[3] <== CMul()(tmp_160, challengesFRI[1]);
    signal tmp_162[3] <== [mapValues.cm1_4 - evals[54][0], -evals[54][1], -evals[54][2]];
    signal tmp_163[3] <== [tmp_161[0] + tmp_162[0], tmp_161[1] + tmp_162[1], tmp_161[2] + tmp_162[2]];
    signal tmp_164[3] <== CMul()(tmp_163, challengesFRI[1]);
    signal tmp_165[3] <== [mapValues.cm1_5 - evals[55][0], -evals[55][1], -evals[55][2]];
    signal tmp_166[3] <== [tmp_164[0] + tmp_165[0], tmp_164[1] + tmp_165[1], tmp_164[2] + tmp_165[2]];
    signal tmp_167[3] <== CMul()(tmp_166, challengesFRI[1]);
    signal tmp_168[3] <== [mapValues.cm1_6 - evals[56][0], -evals[56][1], -evals[56][2]];
    signal tmp_169[3] <== [tmp_167[0] + tmp_168[0], tmp_167[1] + tmp_168[1], tmp_167[2] + tmp_168[2]];
    signal tmp_170[3] <== CMul()(tmp_169, challengesFRI[1]);
    signal tmp_171[3] <== [mapValues.cm1_7 - evals[57][0], -evals[57][1], -evals[57][2]];
    signal tmp_172[3] <== [tmp_170[0] + tmp_171[0], tmp_170[1] + tmp_171[1], tmp_170[2] + tmp_171[2]];
    signal tmp_173[3] <== CMul()(tmp_172, challengesFRI[1]);
    signal tmp_174[3] <== [mapValues.cm1_8 - evals[58][0], -evals[58][1], -evals[58][2]];
    signal tmp_175[3] <== [tmp_173[0] + tmp_174[0], tmp_173[1] + tmp_174[1], tmp_173[2] + tmp_174[2]];
    signal tmp_176[3] <== CMul()(tmp_175, challengesFRI[1]);
    signal tmp_177[3] <== [mapValues.cm1_9 - evals[59][0], -evals[59][1], -evals[59][2]];
    signal tmp_178[3] <== [tmp_176[0] + tmp_177[0], tmp_176[1] + tmp_177[1], tmp_176[2] + tmp_177[2]];
    signal tmp_179[3] <== CMul()(tmp_178, challengesFRI[1]);
    signal tmp_180[3] <== [mapValues.cm1_10 - evals[60][0], -evals[60][1], -evals[60][2]];
    signal tmp_181[3] <== [tmp_179[0] + tmp_180[0], tmp_179[1] + tmp_180[1], tmp_179[2] + tmp_180[2]];
    signal tmp_182[3] <== CMul()(tmp_181, challengesFRI[1]);
    signal tmp_183[3] <== [mapValues.cm1_11 - evals[61][0], -evals[61][1], -evals[61][2]];
    signal tmp_184[3] <== [tmp_182[0] + tmp_183[0], tmp_182[1] + tmp_183[1], tmp_182[2] + tmp_183[2]];
    signal tmp_185[3] <== CMul()(tmp_184, challengesFRI[1]);
    signal tmp_186[3] <== [mapValues.cm1_12 - evals[62][0], -evals[62][1], -evals[62][2]];
    signal tmp_187[3] <== [tmp_185[0] + tmp_186[0], tmp_185[1] + tmp_186[1], tmp_185[2] + tmp_186[2]];
    signal tmp_188[3] <== CMul()(tmp_187, challengesFRI[1]);
    signal tmp_189[3] <== [mapValues.cm1_13 - evals[63][0], -evals[63][1], -evals[63][2]];
    signal tmp_190[3] <== [tmp_188[0] + tmp_189[0], tmp_188[1] + tmp_189[1], tmp_188[2] + tmp_189[2]];
    signal tmp_191[3] <== CMul()(tmp_190, challengesFRI[1]);
    signal tmp_192[3] <== [mapValues.cm1_14 - evals[64][0], -evals[64][1], -evals[64][2]];
    signal tmp_193[3] <== [tmp_191[0] + tmp_192[0], tmp_191[1] + tmp_192[1], tmp_191[2] + tmp_192[2]];
    signal tmp_194[3] <== CMul()(tmp_193, challengesFRI[1]);
    signal tmp_195[3] <== [mapValues.cm1_15 - evals[65][0], -evals[65][1], -evals[65][2]];
    signal tmp_196[3] <== [tmp_194[0] + tmp_195[0], tmp_194[1] + tmp_195[1], tmp_194[2] + tmp_195[2]];
    signal tmp_197[3] <== CMul()(tmp_196, challengesFRI[1]);
    signal tmp_198[3] <== [mapValues.cm1_16 - evals[66][0], -evals[66][1], -evals[66][2]];
    signal tmp_199[3] <== [tmp_197[0] + tmp_198[0], tmp_197[1] + tmp_198[1], tmp_197[2] + tmp_198[2]];
    signal tmp_200[3] <== CMul()(tmp_199, challengesFRI[1]);
    signal tmp_201[3] <== [mapValues.cm1_17 - evals[67][0], -evals[67][1], -evals[67][2]];
    signal tmp_202[3] <== [tmp_200[0] + tmp_201[0], tmp_200[1] + tmp_201[1], tmp_200[2] + tmp_201[2]];
    signal tmp_203[3] <== CMul()(tmp_202, challengesFRI[1]);
    signal tmp_204[3] <== [mapValues.cm1_18 - evals[68][0], -evals[68][1], -evals[68][2]];
    signal tmp_205[3] <== [tmp_203[0] + tmp_204[0], tmp_203[1] + tmp_204[1], tmp_203[2] + tmp_204[2]];
    signal tmp_206[3] <== CMul()(tmp_205, challengesFRI[1]);
    signal tmp_207[3] <== [mapValues.cm1_19 - evals[69][0], -evals[69][1], -evals[69][2]];
    signal tmp_208[3] <== [tmp_206[0] + tmp_207[0], tmp_206[1] + tmp_207[1], tmp_206[2] + tmp_207[2]];
    signal tmp_209[3] <== CMul()(tmp_208, challengesFRI[1]);
    signal tmp_210[3] <== [mapValues.cm1_20 - evals[70][0], -evals[70][1], -evals[70][2]];
    signal tmp_211[3] <== [tmp_209[0] + tmp_210[0], tmp_209[1] + tmp_210[1], tmp_209[2] + tmp_210[2]];
    signal tmp_212[3] <== CMul()(tmp_211, challengesFRI[1]);
    signal tmp_213[3] <== [mapValues.cm1_21 - evals[71][0], -evals[71][1], -evals[71][2]];
    signal tmp_214[3] <== [tmp_212[0] + tmp_213[0], tmp_212[1] + tmp_213[1], tmp_212[2] + tmp_213[2]];
    signal tmp_215[3] <== CMul()(tmp_214, challengesFRI[1]);
    signal tmp_216[3] <== [mapValues.cm1_22 - evals[72][0], -evals[72][1], -evals[72][2]];
    signal tmp_217[3] <== [tmp_215[0] + tmp_216[0], tmp_215[1] + tmp_216[1], tmp_215[2] + tmp_216[2]];
    signal tmp_218[3] <== CMul()(tmp_217, challengesFRI[1]);
    signal tmp_219[3] <== [mapValues.cm1_23 - evals[73][0], -evals[73][1], -evals[73][2]];
    signal tmp_220[3] <== [tmp_218[0] + tmp_219[0], tmp_218[1] + tmp_219[1], tmp_218[2] + tmp_219[2]];
    signal tmp_221[3] <== CMul()(tmp_220, challengesFRI[1]);
    signal tmp_222[3] <== [mapValues.cm1_24 - evals[74][0], -evals[74][1], -evals[74][2]];
    signal tmp_223[3] <== [tmp_221[0] + tmp_222[0], tmp_221[1] + tmp_222[1], tmp_221[2] + tmp_222[2]];
    signal tmp_224[3] <== CMul()(tmp_223, challengesFRI[1]);
    signal tmp_225[3] <== [mapValues.cm1_25 - evals[75][0], -evals[75][1], -evals[75][2]];
    signal tmp_226[3] <== [tmp_224[0] + tmp_225[0], tmp_224[1] + tmp_225[1], tmp_224[2] + tmp_225[2]];
    signal tmp_227[3] <== CMul()(tmp_226, challengesFRI[1]);
    signal tmp_228[3] <== [mapValues.cm1_26 - evals[76][0], -evals[76][1], -evals[76][2]];
    signal tmp_229[3] <== [tmp_227[0] + tmp_228[0], tmp_227[1] + tmp_228[1], tmp_227[2] + tmp_228[2]];
    signal tmp_230[3] <== CMul()(tmp_229, challengesFRI[1]);
    signal tmp_231[3] <== [mapValues.cm1_27 - evals[77][0], -evals[77][1], -evals[77][2]];
    signal tmp_232[3] <== [tmp_230[0] + tmp_231[0], tmp_230[1] + tmp_231[1], tmp_230[2] + tmp_231[2]];
    signal tmp_233[3] <== CMul()(tmp_232, challengesFRI[1]);
    signal tmp_234[3] <== [mapValues.cm1_28 - evals[78][0], -evals[78][1], -evals[78][2]];
    signal tmp_235[3] <== [tmp_233[0] + tmp_234[0], tmp_233[1] + tmp_234[1], tmp_233[2] + tmp_234[2]];
    signal tmp_236[3] <== CMul()(tmp_235, challengesFRI[1]);
    signal tmp_237[3] <== [mapValues.cm1_29 - evals[79][0], -evals[79][1], -evals[79][2]];
    signal tmp_238[3] <== [tmp_236[0] + tmp_237[0], tmp_236[1] + tmp_237[1], tmp_236[2] + tmp_237[2]];
    signal tmp_239[3] <== CMul()(tmp_238, challengesFRI[1]);
    signal tmp_240[3] <== [mapValues.cm1_30 - evals[80][0], -evals[80][1], -evals[80][2]];
    signal tmp_241[3] <== [tmp_239[0] + tmp_240[0], tmp_239[1] + tmp_240[1], tmp_239[2] + tmp_240[2]];
    signal tmp_242[3] <== CMul()(tmp_241, challengesFRI[1]);
    signal tmp_243[3] <== [mapValues.cm1_31 - evals[81][0], -evals[81][1], -evals[81][2]];
    signal tmp_244[3] <== [tmp_242[0] + tmp_243[0], tmp_242[1] + tmp_243[1], tmp_242[2] + tmp_243[2]];
    signal tmp_245[3] <== CMul()(tmp_244, challengesFRI[1]);
    signal tmp_246[3] <== [mapValues.cm1_32 - evals[82][0], -evals[82][1], -evals[82][2]];
    signal tmp_247[3] <== [tmp_245[0] + tmp_246[0], tmp_245[1] + tmp_246[1], tmp_245[2] + tmp_246[2]];
    signal tmp_248[3] <== CMul()(tmp_247, challengesFRI[1]);
    signal tmp_249[3] <== [mapValues.cm1_33 - evals[83][0], -evals[83][1], -evals[83][2]];
    signal tmp_250[3] <== [tmp_248[0] + tmp_249[0], tmp_248[1] + tmp_249[1], tmp_248[2] + tmp_249[2]];
    signal tmp_251[3] <== CMul()(tmp_250, challengesFRI[1]);
    signal tmp_252[3] <== [mapValues.cm1_34 - evals[84][0], -evals[84][1], -evals[84][2]];
    signal tmp_253[3] <== [tmp_251[0] + tmp_252[0], tmp_251[1] + tmp_252[1], tmp_251[2] + tmp_252[2]];
    signal tmp_254[3] <== CMul()(tmp_253, challengesFRI[1]);
    signal tmp_255[3] <== [mapValues.cm1_35 - evals[85][0], -evals[85][1], -evals[85][2]];
    signal tmp_256[3] <== [tmp_254[0] + tmp_255[0], tmp_254[1] + tmp_255[1], tmp_254[2] + tmp_255[2]];
    signal tmp_257[3] <== CMul()(tmp_256, challengesFRI[1]);
    signal tmp_258[3] <== [mapValues.cm1_36 - evals[86][0], -evals[86][1], -evals[86][2]];
    signal tmp_259[3] <== [tmp_257[0] + tmp_258[0], tmp_257[1] + tmp_258[1], tmp_257[2] + tmp_258[2]];
    signal tmp_260[3] <== CMul()(tmp_259, challengesFRI[1]);
    signal tmp_261[3] <== [mapValues.cm1_37 - evals[87][0], -evals[87][1], -evals[87][2]];
    signal tmp_262[3] <== [tmp_260[0] + tmp_261[0], tmp_260[1] + tmp_261[1], tmp_260[2] + tmp_261[2]];
    signal tmp_263[3] <== CMul()(tmp_262, challengesFRI[1]);
    signal tmp_264[3] <== [mapValues.cm1_38 - evals[88][0], -evals[88][1], -evals[88][2]];
    signal tmp_265[3] <== [tmp_263[0] + tmp_264[0], tmp_263[1] + tmp_264[1], tmp_263[2] + tmp_264[2]];
    signal tmp_266[3] <== CMul()(tmp_265, challengesFRI[1]);
    signal tmp_267[3] <== [mapValues.cm1_39 - evals[89][0], -evals[89][1], -evals[89][2]];
    signal tmp_268[3] <== [tmp_266[0] + tmp_267[0], tmp_266[1] + tmp_267[1], tmp_266[2] + tmp_267[2]];
    signal tmp_269[3] <== CMul()(tmp_268, challengesFRI[1]);
    signal tmp_270[3] <== [mapValues.cm1_40 - evals[90][0], -evals[90][1], -evals[90][2]];
    signal tmp_271[3] <== [tmp_269[0] + tmp_270[0], tmp_269[1] + tmp_270[1], tmp_269[2] + tmp_270[2]];
    signal tmp_272[3] <== CMul()(tmp_271, challengesFRI[1]);
    signal tmp_273[3] <== [mapValues.cm1_41 - evals[91][0], -evals[91][1], -evals[91][2]];
    signal tmp_274[3] <== [tmp_272[0] + tmp_273[0], tmp_272[1] + tmp_273[1], tmp_272[2] + tmp_273[2]];
    signal tmp_275[3] <== CMul()(tmp_274, challengesFRI[1]);
    signal tmp_276[3] <== [mapValues.cm1_42 - evals[92][0], -evals[92][1], -evals[92][2]];
    signal tmp_277[3] <== [tmp_275[0] + tmp_276[0], tmp_275[1] + tmp_276[1], tmp_275[2] + tmp_276[2]];
    signal tmp_278[3] <== CMul()(tmp_277, challengesFRI[1]);
    signal tmp_279[3] <== [mapValues.cm1_43 - evals[93][0], -evals[93][1], -evals[93][2]];
    signal tmp_280[3] <== [tmp_278[0] + tmp_279[0], tmp_278[1] + tmp_279[1], tmp_278[2] + tmp_279[2]];
    signal tmp_281[3] <== CMul()(tmp_280, challengesFRI[1]);
    signal tmp_282[3] <== [mapValues.cm1_44 - evals[94][0], -evals[94][1], -evals[94][2]];
    signal tmp_283[3] <== [tmp_281[0] + tmp_282[0], tmp_281[1] + tmp_282[1], tmp_281[2] + tmp_282[2]];
    signal tmp_284[3] <== CMul()(tmp_283, challengesFRI[1]);
    signal tmp_285[3] <== [mapValues.cm1_45 - evals[95][0], -evals[95][1], -evals[95][2]];
    signal tmp_286[3] <== [tmp_284[0] + tmp_285[0], tmp_284[1] + tmp_285[1], tmp_284[2] + tmp_285[2]];
    signal tmp_287[3] <== CMul()(tmp_286, challengesFRI[1]);
    signal tmp_288[3] <== [mapValues.cm1_46 - evals[96][0], -evals[96][1], -evals[96][2]];
    signal tmp_289[3] <== [tmp_287[0] + tmp_288[0], tmp_287[1] + tmp_288[1], tmp_287[2] + tmp_288[2]];
    signal tmp_290[3] <== CMul()(tmp_289, challengesFRI[1]);
    signal tmp_291[3] <== [mapValues.cm1_47 - evals[97][0], -evals[97][1], -evals[97][2]];
    signal tmp_292[3] <== [tmp_290[0] + tmp_291[0], tmp_290[1] + tmp_291[1], tmp_290[2] + tmp_291[2]];
    signal tmp_293[3] <== CMul()(tmp_292, challengesFRI[1]);
    signal tmp_294[3] <== [mapValues.cm2_0[0] - evals[98][0], mapValues.cm2_0[1] - evals[98][1], mapValues.cm2_0[2] - evals[98][2]];
    signal tmp_295[3] <== [tmp_293[0] + tmp_294[0], tmp_293[1] + tmp_294[1], tmp_293[2] + tmp_294[2]];
    signal tmp_296[3] <== CMul()(tmp_295, challengesFRI[1]);
    signal tmp_297[3] <== [mapValues.cm2_1[0] - evals[99][0], mapValues.cm2_1[1] - evals[99][1], mapValues.cm2_1[2] - evals[99][2]];
    signal tmp_298[3] <== [tmp_296[0] + tmp_297[0], tmp_296[1] + tmp_297[1], tmp_296[2] + tmp_297[2]];
    signal tmp_299[3] <== CMul()(tmp_298, challengesFRI[1]);
    signal tmp_300[3] <== [mapValues.cm2_2[0] - evals[100][0], mapValues.cm2_2[1] - evals[100][1], mapValues.cm2_2[2] - evals[100][2]];
    signal tmp_301[3] <== [tmp_299[0] + tmp_300[0], tmp_299[1] + tmp_300[1], tmp_299[2] + tmp_300[2]];
    signal tmp_302[3] <== CMul()(tmp_301, challengesFRI[1]);
    signal tmp_303[3] <== [mapValues.cm2_3[0] - evals[101][0], mapValues.cm2_3[1] - evals[101][1], mapValues.cm2_3[2] - evals[101][2]];
    signal tmp_304[3] <== [tmp_302[0] + tmp_303[0], tmp_302[1] + tmp_303[1], tmp_302[2] + tmp_303[2]];
    signal tmp_305[3] <== CMul()(tmp_304, challengesFRI[1]);
    signal tmp_306[3] <== [mapValues.cm3_0[0] - evals[102][0], mapValues.cm3_0[1] - evals[102][1], mapValues.cm3_0[2] - evals[102][2]];
    signal tmp_307[3] <== [tmp_305[0] + tmp_306[0], tmp_305[1] + tmp_306[1], tmp_305[2] + tmp_306[2]];
    signal tmp_308[3] <== CMul()(tmp_307, challengesFRI[1]);
    signal tmp_309[3] <== [mapValues.cm3_1[0] - evals[103][0], mapValues.cm3_1[1] - evals[103][1], mapValues.cm3_1[2] - evals[103][2]];
    signal tmp_310[3] <== [tmp_308[0] + tmp_309[0], tmp_308[1] + tmp_309[1], tmp_308[2] + tmp_309[2]];
    signal tmp_311[3] <== CMul()(tmp_310, challengesFRI[1]);
    signal tmp_312[3] <== [mapValues.cm3_2[0] - evals[104][0], mapValues.cm3_2[1] - evals[104][1], mapValues.cm3_2[2] - evals[104][2]];
    signal tmp_313[3] <== [tmp_311[0] + tmp_312[0], tmp_311[1] + tmp_312[1], tmp_311[2] + tmp_312[2]];
    signal tmp_314[3] <== CMul()(tmp_313, challengesFRI[1]);
    signal tmp_315[3] <== [mapValues.cm3_3[0] - evals[105][0], mapValues.cm3_3[1] - evals[105][1], mapValues.cm3_3[2] - evals[105][2]];
    signal tmp_316[3] <== [tmp_314[0] + tmp_315[0], tmp_314[1] + tmp_315[1], tmp_314[2] + tmp_315[2]];
    signal tmp_317[3] <== CMul()(tmp_316, challengesFRI[1]);
    signal tmp_318[3] <== [mapValues.cm3_4[0] - evals[106][0], mapValues.cm3_4[1] - evals[106][1], mapValues.cm3_4[2] - evals[106][2]];
    signal tmp_319[3] <== [tmp_317[0] + tmp_318[0], tmp_317[1] + tmp_318[1], tmp_317[2] + tmp_318[2]];
    signal tmp_320[3] <== CMul()(tmp_319, challengesFRI[1]);
    signal tmp_321[3] <== [mapValues.cm3_5[0] - evals[107][0], mapValues.cm3_5[1] - evals[107][1], mapValues.cm3_5[2] - evals[107][2]];
    signal tmp_322[3] <== [tmp_320[0] + tmp_321[0], tmp_320[1] + tmp_321[1], tmp_320[2] + tmp_321[2]];
    signal tmp_323[3] <== CMul()(tmp_322, challengesFRI[1]);
    signal tmp_324[3] <== [mapValues.cm3_6[0] - evals[108][0], mapValues.cm3_6[1] - evals[108][1], mapValues.cm3_6[2] - evals[108][2]];
    signal tmp_325[3] <== [tmp_323[0] + tmp_324[0], tmp_323[1] + tmp_324[1], tmp_323[2] + tmp_324[2]];
    signal tmp_326[3] <== CMul()(tmp_325, xDivXSubXi[2]);
    signal tmp_327[3] <== [tmp_15[0] + tmp_326[0], tmp_15[1] + tmp_326[1], tmp_15[2] + tmp_326[2]];
    signal tmp_328[3] <== CMul()(challengesFRI[0], tmp_327);
    signal tmp_329[3] <== [consts[36] - evals[109][0], -evals[109][1], -evals[109][2]];
    signal tmp_330[3] <== CMul()(tmp_329, challengesFRI[1]);
    signal tmp_331[3] <== [consts[44] - evals[110][0], -evals[110][1], -evals[110][2]];
    signal tmp_332[3] <== [tmp_330[0] + tmp_331[0], tmp_330[1] + tmp_331[1], tmp_330[2] + tmp_331[2]];
    signal tmp_333[3] <== CMul()(tmp_332, challengesFRI[1]);
    signal tmp_334[3] <== [mapValues.cm1_0 - evals[111][0], -evals[111][1], -evals[111][2]];
    signal tmp_335[3] <== [tmp_333[0] + tmp_334[0], tmp_333[1] + tmp_334[1], tmp_333[2] + tmp_334[2]];
    signal tmp_336[3] <== CMul()(tmp_335, challengesFRI[1]);
    signal tmp_337[3] <== [mapValues.cm1_1 - evals[112][0], -evals[112][1], -evals[112][2]];
    signal tmp_338[3] <== [tmp_336[0] + tmp_337[0], tmp_336[1] + tmp_337[1], tmp_336[2] + tmp_337[2]];
    signal tmp_339[3] <== CMul()(tmp_338, challengesFRI[1]);
    signal tmp_340[3] <== [mapValues.cm1_2 - evals[113][0], -evals[113][1], -evals[113][2]];
    signal tmp_341[3] <== [tmp_339[0] + tmp_340[0], tmp_339[1] + tmp_340[1], tmp_339[2] + tmp_340[2]];
    signal tmp_342[3] <== CMul()(tmp_341, challengesFRI[1]);
    signal tmp_343[3] <== [mapValues.cm1_3 - evals[114][0], -evals[114][1], -evals[114][2]];
    signal tmp_344[3] <== [tmp_342[0] + tmp_343[0], tmp_342[1] + tmp_343[1], tmp_342[2] + tmp_343[2]];
    signal tmp_345[3] <== CMul()(tmp_344, challengesFRI[1]);
    signal tmp_346[3] <== [mapValues.cm1_4 - evals[115][0], -evals[115][1], -evals[115][2]];
    signal tmp_347[3] <== [tmp_345[0] + tmp_346[0], tmp_345[1] + tmp_346[1], tmp_345[2] + tmp_346[2]];
    signal tmp_348[3] <== CMul()(tmp_347, challengesFRI[1]);
    signal tmp_349[3] <== [mapValues.cm1_5 - evals[116][0], -evals[116][1], -evals[116][2]];
    signal tmp_350[3] <== [tmp_348[0] + tmp_349[0], tmp_348[1] + tmp_349[1], tmp_348[2] + tmp_349[2]];
    signal tmp_351[3] <== CMul()(tmp_350, challengesFRI[1]);
    signal tmp_352[3] <== [mapValues.cm1_15 - evals[117][0], -evals[117][1], -evals[117][2]];
    signal tmp_353[3] <== [tmp_351[0] + tmp_352[0], tmp_351[1] + tmp_352[1], tmp_351[2] + tmp_352[2]];
    signal tmp_354[3] <== CMul()(tmp_353, challengesFRI[1]);
    signal tmp_355[3] <== [mapValues.cm1_16 - evals[118][0], -evals[118][1], -evals[118][2]];
    signal tmp_356[3] <== [tmp_354[0] + tmp_355[0], tmp_354[1] + tmp_355[1], tmp_354[2] + tmp_355[2]];
    signal tmp_357[3] <== CMul()(tmp_356, challengesFRI[1]);
    signal tmp_358[3] <== [mapValues.cm1_17 - evals[119][0], -evals[119][1], -evals[119][2]];
    signal tmp_359[3] <== [tmp_357[0] + tmp_358[0], tmp_357[1] + tmp_358[1], tmp_357[2] + tmp_358[2]];
    signal tmp_360[3] <== CMul()(tmp_359, challengesFRI[1]);
    signal tmp_361[3] <== [mapValues.cm1_18 - evals[120][0], -evals[120][1], -evals[120][2]];
    signal tmp_362[3] <== [tmp_360[0] + tmp_361[0], tmp_360[1] + tmp_361[1], tmp_360[2] + tmp_361[2]];
    signal tmp_363[3] <== CMul()(tmp_362, challengesFRI[1]);
    signal tmp_364[3] <== [mapValues.cm1_19 - evals[121][0], -evals[121][1], -evals[121][2]];
    signal tmp_365[3] <== [tmp_363[0] + tmp_364[0], tmp_363[1] + tmp_364[1], tmp_363[2] + tmp_364[2]];
    signal tmp_366[3] <== CMul()(tmp_365, challengesFRI[1]);
    signal tmp_367[3] <== [mapValues.cm1_20 - evals[122][0], -evals[122][1], -evals[122][2]];
    signal tmp_368[3] <== [tmp_366[0] + tmp_367[0], tmp_366[1] + tmp_367[1], tmp_366[2] + tmp_367[2]];
    signal tmp_369[3] <== CMul()(tmp_368, challengesFRI[1]);
    signal tmp_370[3] <== [mapValues.cm1_21 - evals[123][0], -evals[123][1], -evals[123][2]];
    signal tmp_371[3] <== [tmp_369[0] + tmp_370[0], tmp_369[1] + tmp_370[1], tmp_369[2] + tmp_370[2]];
    signal tmp_372[3] <== CMul()(tmp_371, challengesFRI[1]);
    signal tmp_373[3] <== [mapValues.cm1_22 - evals[124][0], -evals[124][1], -evals[124][2]];
    signal tmp_374[3] <== [tmp_372[0] + tmp_373[0], tmp_372[1] + tmp_373[1], tmp_372[2] + tmp_373[2]];
    signal tmp_375[3] <== CMul()(tmp_374, challengesFRI[1]);
    signal tmp_376[3] <== [mapValues.cm1_23 - evals[125][0], -evals[125][1], -evals[125][2]];
    signal tmp_377[3] <== [tmp_375[0] + tmp_376[0], tmp_375[1] + tmp_376[1], tmp_375[2] + tmp_376[2]];
    signal tmp_378[3] <== CMul()(tmp_377, challengesFRI[1]);
    signal tmp_379[3] <== [mapValues.cm1_24 - evals[126][0], -evals[126][1], -evals[126][2]];
    signal tmp_380[3] <== [tmp_378[0] + tmp_379[0], tmp_378[1] + tmp_379[1], tmp_378[2] + tmp_379[2]];
    signal tmp_381[3] <== CMul()(tmp_380, challengesFRI[1]);
    signal tmp_382[3] <== [mapValues.cm1_25 - evals[127][0], -evals[127][1], -evals[127][2]];
    signal tmp_383[3] <== [tmp_381[0] + tmp_382[0], tmp_381[1] + tmp_382[1], tmp_381[2] + tmp_382[2]];
    signal tmp_384[3] <== CMul()(tmp_383, challengesFRI[1]);
    signal tmp_385[3] <== [mapValues.cm1_26 - evals[128][0], -evals[128][1], -evals[128][2]];
    signal tmp_386[3] <== [tmp_384[0] + tmp_385[0], tmp_384[1] + tmp_385[1], tmp_384[2] + tmp_385[2]];
    signal tmp_387[3] <== CMul()(tmp_386, challengesFRI[1]);
    signal tmp_388[3] <== [mapValues.cm1_27 - evals[129][0], -evals[129][1], -evals[129][2]];
    signal tmp_389[3] <== [tmp_387[0] + tmp_388[0], tmp_387[1] + tmp_388[1], tmp_387[2] + tmp_388[2]];
    signal tmp_390[3] <== CMul()(tmp_389, challengesFRI[1]);
    signal tmp_391[3] <== [mapValues.cm1_28 - evals[130][0], -evals[130][1], -evals[130][2]];
    signal tmp_392[3] <== [tmp_390[0] + tmp_391[0], tmp_390[1] + tmp_391[1], tmp_390[2] + tmp_391[2]];
    signal tmp_393[3] <== CMul()(tmp_392, challengesFRI[1]);
    signal tmp_394[3] <== [mapValues.cm1_29 - evals[131][0], -evals[131][1], -evals[131][2]];
    signal tmp_395[3] <== [tmp_393[0] + tmp_394[0], tmp_393[1] + tmp_394[1], tmp_393[2] + tmp_394[2]];
    signal tmp_396[3] <== CMul()(tmp_395, challengesFRI[1]);
    signal tmp_397[3] <== [mapValues.cm1_30 - evals[132][0], -evals[132][1], -evals[132][2]];
    signal tmp_398[3] <== [tmp_396[0] + tmp_397[0], tmp_396[1] + tmp_397[1], tmp_396[2] + tmp_397[2]];
    signal tmp_399[3] <== CMul()(tmp_398, challengesFRI[1]);
    signal tmp_400[3] <== [mapValues.cm1_31 - evals[133][0], -evals[133][1], -evals[133][2]];
    signal tmp_401[3] <== [tmp_399[0] + tmp_400[0], tmp_399[1] + tmp_400[1], tmp_399[2] + tmp_400[2]];
    signal tmp_402[3] <== CMul()(tmp_401, xDivXSubXi[3]);
    signal tmp_403[3] <== [tmp_328[0] + tmp_402[0], tmp_328[1] + tmp_402[1], tmp_328[2] + tmp_402[2]];
    signal tmp_404[3] <== CMul()(challengesFRI[0], tmp_403);
    signal tmp_405[3] <== [mapValues.cm1_15 - evals[134][0], -evals[134][1], -evals[134][2]];
    signal tmp_406[3] <== CMul()(tmp_405, xDivXSubXi[4]);
    tmp_408 <== [tmp_404[0] + tmp_406[0], tmp_404[1] + tmp_406[1], tmp_404[2] + tmp_406[2]];

}


/* 
    Verify that the initial FRI polynomial, which is the lineal combination of the committed polynomials
    during the STARK phases, is built properly
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesFRI[20];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[135][3];
    signal input cm1[48];
    signal input cm2[12];
    signal input cm3[21];
    signal input consts[45];

    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    signal xacc[20];
    xacc[0] <== queriesFRI[0]*(7 * roots(20)-7) + 7;
    for (var i=1; i<20; i++) {
        xacc[i] <== xacc[i-1] * ( queriesFRI[i]*(roots(20 - i) - 1) +1);
    }

    signal xDivXSubXi[5][3];

    xDivXSubXi[0] <== CInv()([xacc[19] - 9071788333329385449 * challengeXi[0], - 9071788333329385449 * challengeXi[1], - 9071788333329385449 * challengeXi[2]]);
    xDivXSubXi[1] <== CInv()([xacc[19] - 15139302138664925958 * challengeXi[0], - 15139302138664925958 * challengeXi[1], - 15139302138664925958 * challengeXi[2]]);
    xDivXSubXi[2] <== CInv()([xacc[19] - 1 * challengeXi[0], - 1 * challengeXi[1], - 1 * challengeXi[2]]);
    xDivXSubXi[3] <== CInv()([xacc[19] - 5718075921287398682 * challengeXi[0], - 5718075921287398682 * challengeXi[1], - 5718075921287398682 * challengeXi[2]]);
    xDivXSubXi[4] <== CInv()([xacc[19] - 8167150655112846419 * challengeXi[0], - 8167150655112846419 * challengeXi[1], - 8167150655112846419 * challengeXi[2]]);

    signal tmp_408[3];
    (tmp_408) <== CalculateFRIPolChunks0()(challengesFRI,evals,cm1,cm2,cm3,consts,xDivXSubXi);

    signal queryVals[3];
    queryVals[0] <== tmp_408[0];
    queryVals[1] <== tmp_408[1];
    queryVals[2] <== tmp_408[2];

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

template VerifySingleQuery0() {
    signal input {binary} queriesFRI[20];

    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input challengesFRISteps[7][3];
    signal input evals[135][3];
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

    signal input s0_valsC[45];
    signal input s0_valsC_p[45][1];
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

    signal {binary} queriesFRIBits[10][2];
    for(var j = 0; j < 10; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 20) {
                queriesFRIBits[j][k] <== 0;
            } else {
                queriesFRIBits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(1, 48, 4, 8, 2, 1048576)(s0_vals1_p, s0_siblings1, queriesFRIBits, s0_last_mt_levels1, enable);
    VerifyMerkleHashUntilLevel(1, 12, 4, 8, 2, 1048576)(s0_vals2_p, s0_siblings2, queriesFRIBits, s0_last_mt_levels2, enable);

    VerifyMerkleHashUntilLevel(1, 21, 4, 8, 2, 1048576)(s0_vals3_p, s0_siblings3, queriesFRIBits, s0_last_mt_levels3, enable);
    VerifyMerkleHashUntilLevel(1, 45, 4, 8, 2, 1048576)(s0_valsC_p, s0_siblingsC, queriesFRIBits, s0_last_mt_levelsC, enable);


    signal {binary} s1_keys_merkle_bits[9][2];
    for(var j = 0; j < 9; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 17) {
                s1_keys_merkle_bits[j][k] <== 0;
            } else {
                s1_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 7, 2, 131072)(s1_vals_p, s1_siblings, s1_keys_merkle_bits, s1_last_mt_levels, enable);
    signal {binary} s2_keys_merkle_bits[7][2];
    for(var j = 0; j < 7; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 14) {
                s2_keys_merkle_bits[j][k] <== 0;
            } else {
                s2_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 5, 2, 16384)(s2_vals_p, s2_siblings, s2_keys_merkle_bits, s2_last_mt_levels, enable);
    signal {binary} s3_keys_merkle_bits[6][2];
    for(var j = 0; j < 6; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 11) {
                s3_keys_merkle_bits[j][k] <== 0;
            } else {
                s3_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 4, 2, 2048)(s3_vals_p, s3_siblings, s3_keys_merkle_bits, s3_last_mt_levels, enable);
    signal {binary} s4_keys_merkle_bits[4][2];
    for(var j = 0; j < 4; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 8) {
                s4_keys_merkle_bits[j][k] <== 0;
            } else {
                s4_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 2, 2, 256)(s4_vals_p, s4_siblings, s4_keys_merkle_bits, s4_last_mt_levels, enable);
    signal {binary} s5_keys_merkle_bits[3][2];
    for(var j = 0; j < 3; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 5) {
                s5_keys_merkle_bits[j][k] <== 0;
            } else {
                s5_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 1, 2, 32)(s5_vals_p, s5_siblings, s5_keys_merkle_bits, s5_last_mt_levels, enable);

    VerifyQuery0(20, 17)(queriesFRI, challengeXi, challengesFRI, evals, s0_vals1, s0_vals2, s0_vals3, s0_valsC, s1_vals_p, enable);

    signal {binary} s1_queriesFRI[17];
    for(var i = 0; i < 17; i++) { s1_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(20, 20, 17, 14, 2635249152773512046)(s1_queriesFRI, challengesFRISteps[1], s1_vals_p, s2_vals_p, enable);
    signal {binary} s2_queriesFRI[14];
    for(var i = 0; i < 14; i++) { s2_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(20, 17, 14, 11, 12421013511830570338)(s2_queriesFRI, challengesFRISteps[2], s2_vals_p, s3_vals_p, enable);
    signal {binary} s3_queriesFRI[11];
    for(var i = 0; i < 11; i++) { s3_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(20, 14, 11, 8, 11143297345130450484)(s3_queriesFRI, challengesFRISteps[3], s3_vals_p, s4_vals_p, enable);
    signal {binary} s4_queriesFRI[8];
    for(var i = 0; i < 8; i++) { s4_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(20, 11, 8, 5, 1138102428757299658)(s4_queriesFRI, challengesFRISteps[4], s4_vals_p, s5_vals_p, enable);
    signal {binary} s5_queriesFRI[5];
    for(var i = 0; i < 5; i++) { s5_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(20, 8, 5, 0, 140704680260498080)(s5_queriesFRI, challengesFRISteps[5], s5_vals_p, finalPol, enable);
}

template parallel VerifyQueriesBatch0(nQueries) {
    signal input {binary} queriesFRI[nQueries][20];

    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input challengesFRISteps[7][3];
    signal input evals[135][3];
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

    signal input s0_valsC[nQueries][45];
    signal input s0_valsC_p[nQueries][45][1];
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
            queriesFRI[q],
            challengeXi,
            challengesFRI,
            challengesFRISteps,
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

template StarkVerifier0() {
    signal input publics[395]; // publics polynomials
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials

    signal input rootC[4]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[135][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
    signal input s0_vals1[73][48];
    signal input s0_vals2[73][12];
    signal input s0_vals3[73][21];
    signal input s0_valsC[73][45];


    // Merkle proofs for each of the evaluations
    signal input s0_siblings1[73][8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_siblings2[73][8][12];
    signal input s0_last_mt_levels2[16][4];
    signal input s0_siblings3[73][8][12];
    signal input s0_last_mt_levels3[16][4];
    signal input s0_siblingsC[73][8][12];
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
    signal input s1_vals[73][24];
    signal input s1_siblings[73][7][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals[73][24];
    signal input s2_siblings[73][5][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals[73][24];
    signal input s3_siblings[73][4][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals[73][24];
    signal input s4_siblings[73][2][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals[73][24];
    signal input s5_siblings[73][1][12];
    signal input s5_last_mt_levels[16][4];

    // Evaluations of the final FRI polynomial over a set of points of size bounded its degree
    signal input finalPol[32][3];

    signal input nonce;

    signal {binary} enabled;
    signal input enable;
    enable * (enable -1) === 0;
    enabled <== enable;




    signal queryVals[73][3];


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
    signal {binary} queriesFRI[73][20];


    ///////////
    // Calculate challenges, challengesFRISteps and queriesFRI
    ///////////

    (challengesStage2,challengeQ,challengeXi,challengesFRI,challengesFRISteps,queriesFRI) <== Transcript0()(publics,rootC,root1,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,s5_root,finalPol, nonce, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputsC so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
    var s0_vals1_p[73][48][1];
    var s0_vals2_p[73][12][1];
    var s0_vals3_p[73][21][1];
    var s0_valsC_p[73][45][1];
    var s0_vals_p[73][1][3]; 
    var s1_vals_p[73][8][3]; 
    var s2_vals_p[73][8][3]; 
    var s3_vals_p[73][8][3]; 
    var s4_vals_p[73][8][3]; 
    var s5_vals_p[73][8][3]; 

    for (var q=0; q<73; q++) {
        // Preprocess vals for the initial FRI polynomial
        for (var i = 0; i < 48; i++) {
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
    // Verify Merkle roots and FRI constraints per query
    ///////////

    // Batch-size parameters — change QUERIES_BATCH_SIZE and recompile to test different batch sizes.
    // N_FULL_BATCHES and LAST_BATCH_SIZE are derived automatically.
    var N_QUERIES = 73;
    var QUERIES_BATCH_SIZE = 10;
    var N_FULL_BATCHES = N_QUERIES \ QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE = N_QUERIES - N_FULL_BATCHES * QUERIES_BATCH_SIZE;
    // Used as signal dimension when LAST_BATCH_SIZE may be 0 (size-0 signals are invalid in Circom).
    var LAST_BATCH_SIZE_SAFE = LAST_BATCH_SIZE > 0 ? LAST_BATCH_SIZE : 1;

// Work buffers for full batches
    signal {binary} batch_work_queriesFRI[N_FULL_BATCHES][QUERIES_BATCH_SIZE][20];

    signal batch_work_s0_vals1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][48];
    signal batch_work_s0_vals1_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][48][1];
    signal batch_work_s0_siblings1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];
    signal batch_work_s0_vals2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][12];
    signal batch_work_s0_vals2_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][12][1];
    signal batch_work_s0_siblings2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];

    signal batch_work_s0_vals3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][21];
    signal batch_work_s0_vals3_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][21][1];
    signal batch_work_s0_siblings3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];
    signal batch_work_s0_valsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][45];
    signal batch_work_s0_valsC_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][45][1];
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

    // Process full batches with Circom loop
    for (var b = 0; b < N_FULL_BATCHES; b++) {
        var batchStart = b * QUERIES_BATCH_SIZE;

        // Fill work buffers for batch b
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 20; i++) {
                batch_work_queriesFRI[b][q][i] <== queriesFRI[batchStart + q][i];
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
            for (var i = 0; i < 45; i++) {
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

        // Call batch verifier with slice [b] of work buffers
        VerifyQueriesBatch0(QUERIES_BATCH_SIZE)(
            batch_work_queriesFRI[b],
            challengeXi,
            challengesFRI,
            challengesFRISteps,
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

// Remainder batch — signal declarations always emitted; call guarded by Circom if (LAST_BATCH_SIZE > 0)
    signal {binary} remainder_queriesFRI[LAST_BATCH_SIZE_SAFE][20];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 20; i++) {
            remainder_queriesFRI[q][i] <== queriesFRI[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
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
    signal remainder_s0_valsC[LAST_BATCH_SIZE_SAFE][45];
    signal remainder_s0_valsC_p[LAST_BATCH_SIZE_SAFE][45][1];
    signal remainder_s0_siblingsC[LAST_BATCH_SIZE_SAFE][8][12];

    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 21; i++) {
            remainder_s0_vals3[q][i] <== s0_vals3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals3_p[q][i][0] <== s0_vals3_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var i = 0; i < 45; i++) {
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
        remainder_queriesFRI,
        challengeXi,
        challengesFRI,
        challengesFRISteps,
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
    // Check constraints polynomial in the evaluation point
    ///////////

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, enabled);


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
    // Verify Merkle roots for optimized last levels (shared by all queries)
    ///////////

    VerifyFinalPol0()(finalPol, enabled);
}

