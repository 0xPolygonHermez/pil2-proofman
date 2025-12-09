pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "test.verifier.circom";


include "elliptic_curve.circom";

template CalculateStage1Hash() {
    signal input rootC[4];
    signal input root1[4];

    signal input airValues[3][3];

    signal output values[368];

    _ <== airValues[2]; // Unused air values at stage 1

    signal transcriptHash_0[16] <== Poseidon2(4,16)([rootC[0],rootC[1],rootC[2],rootC[3],root1[0],root1[1],root1[2],root1[3],airValues[0][0],airValues[1][0],0,0], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values
    }

    values[0] <== transcriptHash_0[0];
    values[1] <== transcriptHash_0[1];
    values[2] <== transcriptHash_0[2];
    values[3] <== transcriptHash_0[3];
    values[4] <== transcriptHash_0[4];
    values[5] <== transcriptHash_0[5];
    values[6] <== transcriptHash_0[6];
    values[7] <== transcriptHash_0[7];
    values[8] <== transcriptHash_0[8];
    values[9] <== transcriptHash_0[9];
    values[10] <== transcriptHash_0[10];
    values[11] <== transcriptHash_0[11];
    values[12] <== transcriptHash_0[12];
    values[13] <== transcriptHash_0[13];
    values[14] <== transcriptHash_0[14];
    values[15] <== transcriptHash_0[15];
    signal transcriptHash_1[16] <== Poseidon2(4, 16)([values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11]], [values[12], values[13], values[14], values[15]]);
    for (var j = 0; j < 16; j++) {
        values[16 + j] <== transcriptHash_1[j];
    }

    signal transcriptHash_2[16] <== Poseidon2(4, 16)([values[16], values[17], values[18], values[19], values[20], values[21], values[22], values[23], values[24], values[25], values[26], values[27]], [values[28], values[29], values[30], values[31]]);
    for (var j = 0; j < 16; j++) {
        values[32 + j] <== transcriptHash_2[j];
    }

    signal transcriptHash_3[16] <== Poseidon2(4, 16)([values[32], values[33], values[34], values[35], values[36], values[37], values[38], values[39], values[40], values[41], values[42], values[43]], [values[44], values[45], values[46], values[47]]);
    for (var j = 0; j < 16; j++) {
        values[48 + j] <== transcriptHash_3[j];
    }

    signal transcriptHash_4[16] <== Poseidon2(4, 16)([values[48], values[49], values[50], values[51], values[52], values[53], values[54], values[55], values[56], values[57], values[58], values[59]], [values[60], values[61], values[62], values[63]]);
    for (var j = 0; j < 16; j++) {
        values[64 + j] <== transcriptHash_4[j];
    }

    signal transcriptHash_5[16] <== Poseidon2(4, 16)([values[64], values[65], values[66], values[67], values[68], values[69], values[70], values[71], values[72], values[73], values[74], values[75]], [values[76], values[77], values[78], values[79]]);
    for (var j = 0; j < 16; j++) {
        values[80 + j] <== transcriptHash_5[j];
    }

    signal transcriptHash_6[16] <== Poseidon2(4, 16)([values[80], values[81], values[82], values[83], values[84], values[85], values[86], values[87], values[88], values[89], values[90], values[91]], [values[92], values[93], values[94], values[95]]);
    for (var j = 0; j < 16; j++) {
        values[96 + j] <== transcriptHash_6[j];
    }

    signal transcriptHash_7[16] <== Poseidon2(4, 16)([values[96], values[97], values[98], values[99], values[100], values[101], values[102], values[103], values[104], values[105], values[106], values[107]], [values[108], values[109], values[110], values[111]]);
    for (var j = 0; j < 16; j++) {
        values[112 + j] <== transcriptHash_7[j];
    }

    signal transcriptHash_8[16] <== Poseidon2(4, 16)([values[112], values[113], values[114], values[115], values[116], values[117], values[118], values[119], values[120], values[121], values[122], values[123]], [values[124], values[125], values[126], values[127]]);
    for (var j = 0; j < 16; j++) {
        values[128 + j] <== transcriptHash_8[j];
    }

    signal transcriptHash_9[16] <== Poseidon2(4, 16)([values[128], values[129], values[130], values[131], values[132], values[133], values[134], values[135], values[136], values[137], values[138], values[139]], [values[140], values[141], values[142], values[143]]);
    for (var j = 0; j < 16; j++) {
        values[144 + j] <== transcriptHash_9[j];
    }

    signal transcriptHash_10[16] <== Poseidon2(4, 16)([values[144], values[145], values[146], values[147], values[148], values[149], values[150], values[151], values[152], values[153], values[154], values[155]], [values[156], values[157], values[158], values[159]]);
    for (var j = 0; j < 16; j++) {
        values[160 + j] <== transcriptHash_10[j];
    }

    signal transcriptHash_11[16] <== Poseidon2(4, 16)([values[160], values[161], values[162], values[163], values[164], values[165], values[166], values[167], values[168], values[169], values[170], values[171]], [values[172], values[173], values[174], values[175]]);
    for (var j = 0; j < 16; j++) {
        values[176 + j] <== transcriptHash_11[j];
    }

    signal transcriptHash_12[16] <== Poseidon2(4, 16)([values[176], values[177], values[178], values[179], values[180], values[181], values[182], values[183], values[184], values[185], values[186], values[187]], [values[188], values[189], values[190], values[191]]);
    for (var j = 0; j < 16; j++) {
        values[192 + j] <== transcriptHash_12[j];
    }

    signal transcriptHash_13[16] <== Poseidon2(4, 16)([values[192], values[193], values[194], values[195], values[196], values[197], values[198], values[199], values[200], values[201], values[202], values[203]], [values[204], values[205], values[206], values[207]]);
    for (var j = 0; j < 16; j++) {
        values[208 + j] <== transcriptHash_13[j];
    }

    signal transcriptHash_14[16] <== Poseidon2(4, 16)([values[208], values[209], values[210], values[211], values[212], values[213], values[214], values[215], values[216], values[217], values[218], values[219]], [values[220], values[221], values[222], values[223]]);
    for (var j = 0; j < 16; j++) {
        values[224 + j] <== transcriptHash_14[j];
    }

    signal transcriptHash_15[16] <== Poseidon2(4, 16)([values[224], values[225], values[226], values[227], values[228], values[229], values[230], values[231], values[232], values[233], values[234], values[235]], [values[236], values[237], values[238], values[239]]);
    for (var j = 0; j < 16; j++) {
        values[240 + j] <== transcriptHash_15[j];
    }

    signal transcriptHash_16[16] <== Poseidon2(4, 16)([values[240], values[241], values[242], values[243], values[244], values[245], values[246], values[247], values[248], values[249], values[250], values[251]], [values[252], values[253], values[254], values[255]]);
    for (var j = 0; j < 16; j++) {
        values[256 + j] <== transcriptHash_16[j];
    }

    signal transcriptHash_17[16] <== Poseidon2(4, 16)([values[256], values[257], values[258], values[259], values[260], values[261], values[262], values[263], values[264], values[265], values[266], values[267]], [values[268], values[269], values[270], values[271]]);
    for (var j = 0; j < 16; j++) {
        values[272 + j] <== transcriptHash_17[j];
    }

    signal transcriptHash_18[16] <== Poseidon2(4, 16)([values[272], values[273], values[274], values[275], values[276], values[277], values[278], values[279], values[280], values[281], values[282], values[283]], [values[284], values[285], values[286], values[287]]);
    for (var j = 0; j < 16; j++) {
        values[288 + j] <== transcriptHash_18[j];
    }

    signal transcriptHash_19[16] <== Poseidon2(4, 16)([values[288], values[289], values[290], values[291], values[292], values[293], values[294], values[295], values[296], values[297], values[298], values[299]], [values[300], values[301], values[302], values[303]]);
    for (var j = 0; j < 16; j++) {
        values[304 + j] <== transcriptHash_19[j];
    }

    signal transcriptHash_20[16] <== Poseidon2(4, 16)([values[304], values[305], values[306], values[307], values[308], values[309], values[310], values[311], values[312], values[313], values[314], values[315]], [values[316], values[317], values[318], values[319]]);
    for (var j = 0; j < 16; j++) {
        values[320 + j] <== transcriptHash_20[j];
    }

    signal transcriptHash_21[16] <== Poseidon2(4, 16)([values[320], values[321], values[322], values[323], values[324], values[325], values[326], values[327], values[328], values[329], values[330], values[331]], [values[332], values[333], values[334], values[335]]);
    for (var j = 0; j < 16; j++) {
        values[336 + j] <== transcriptHash_21[j];
    }

    signal transcriptHash_22[16] <== Poseidon2(4, 16)([values[336], values[337], values[338], values[339], values[340], values[341], values[342], values[343], values[344], values[345], values[346], values[347]], [values[348], values[349], values[350], values[351]]);
    for (var j = 0; j < 16; j++) {
        values[352 + j] <== transcriptHash_22[j];
    }

}

template Recursive1() {


    signal output sv_circuitType;

    signal output sv_aggregatedProofs;

    signal output sv_aggregationTypes[1];
    signal output sv_airgroupvalues[1][3];

    signal output sv_stage1Hash[368];


    signal input airgroupvalues[1][3];

    signal input airvalues[3][3];

    signal input root1[4];
    signal input root2[4];
    signal input root3[4];

    signal input evals[20][3]; // Evaluations of the set polynomials at a challenge value z and gz

    signal input s0_valsC[128][2];
    signal input s0_siblingsC[128][12][12];

    signal input s0_vals_rom_0[128][2];
    signal input s0_siblings_rom_0[128][12][12];

    signal input s0_vals1[128][2];
    signal input s0_siblings1[128][12][12];
    signal input s0_vals2[128][9];
    signal input s0_siblings2[128][12][12];
    signal input s0_vals3[128][3];
    signal input s0_siblings3[128][12][12];

    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];
    signal input s6_root[4];

    signal input s1_vals[128][24];
    signal input s1_siblings[128][10][12];
    signal input s2_vals[128][24];
    signal input s2_siblings[128][9][12];
    signal input s3_vals[128][24];
    signal input s3_siblings[128][7][12];
    signal input s4_vals[128][24];
    signal input s4_siblings[128][6][12];
    signal input s5_vals[128][24];
    signal input s5_siblings[128][4][12];
    signal input s6_vals[128][24];
    signal input s6_siblings[128][3][12];

    signal input finalPol[32][3];

    signal input publics[8];
    
    signal input proofValues[2][3];
    
    signal input globalChallenge[3];

    signal input rootCAgg[4];



    component sV = StarkVerifier0();

    for (var i=0; i< 8; i++) {
        sV.publics[i] <== publics[i];
    }

    sV.airgroupvalues <== airgroupvalues;

    sV.airvalues <== airvalues;

    sV.proofvalues <== proofValues;

    sV.root1 <== root1;
    sV.root2 <== root2;
    sV.root3 <== root3;

    sV.evals <== evals;

    sV.s0_valsC <== s0_valsC;
    sV.s0_siblingsC <== s0_siblingsC;

    sV.s0_vals_rom_0 <== s0_vals_rom_0;
    sV.s0_siblings_rom_0 <== s0_siblings_rom_0;

    sV.s0_vals1 <== s0_vals1;
    sV.s0_siblings1 <== s0_siblings1;
    sV.s0_vals2 <== s0_vals2;
    sV.s0_siblings2 <== s0_siblings2;
    sV.s0_vals3 <== s0_vals3;
    sV.s0_siblings3 <== s0_siblings3;

    sV.s1_root <== s1_root;
    sV.s2_root <== s2_root;
    sV.s3_root <== s3_root;
    sV.s4_root <== s4_root;
    sV.s5_root <== s5_root;
    sV.s6_root <== s6_root;
    sV.s1_vals <== s1_vals;
    sV.s1_siblings <== s1_siblings;
    sV.s2_vals <== s2_vals;
    sV.s2_siblings <== s2_siblings;
    sV.s3_vals <== s3_vals;
    sV.s3_siblings <== s3_siblings;
    sV.s4_vals <== s4_vals;
    sV.s4_siblings <== s4_siblings;
    sV.s5_vals <== s5_vals;
    sV.s5_siblings <== s5_siblings;
    sV.s6_vals <== s6_vals;
    sV.s6_siblings <== s6_siblings;

    sV.finalPol <== finalPol;


    


    sV.globalChallenge <== globalChallenge;

    // --> Assign the VADCOP data
    sv_circuitType <== 2;
    
    sv_aggregatedProofs <== 1;
    sv_aggregationTypes <== [0];

    sv_airgroupvalues[0] <== airgroupvalues[0];

    sv_stage1Hash <== CalculateStage1Hash()(sV.rootC, root1, airvalues);

}

    
component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive1();

