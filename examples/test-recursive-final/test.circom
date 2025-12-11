pragma circom 2.1.0;
pragma custom_templates;

include "test.verifier.circom";
include "iszero.circom";
include "select_vk.circom";
include "agg_values.circom";


template VerifyGlobalChallenges() {

    signal input publics[8];
    signal input proofValues[2][3];
    signal input stage1Hash[1][368];
    
    signal input globalChallenge[3];
    signal calculatedGlobalChallenge[3];

    
    signal transcriptHash_0[16] <== Poseidon2(4, 16)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7],proofValues[0][0],proofValues[1][0],stage1Hash[0][0],stage1Hash[0][1]], [0,0,0,0]);
    for (var i = 4; i < 16; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values
    }

    
    signal transcriptHash_1[16] <== Poseidon2(4, 16)([stage1Hash[0][2],stage1Hash[0][3],stage1Hash[0][4],stage1Hash[0][5],stage1Hash[0][6],stage1Hash[0][7],stage1Hash[0][8],stage1Hash[0][9],stage1Hash[0][10],stage1Hash[0][11],stage1Hash[0][12],stage1Hash[0][13]], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_1[i]; // Unused transcript values
    }

    
    signal transcriptHash_2[16] <== Poseidon2(4, 16)([stage1Hash[0][14],stage1Hash[0][15],stage1Hash[0][16],stage1Hash[0][17],stage1Hash[0][18],stage1Hash[0][19],stage1Hash[0][20],stage1Hash[0][21],stage1Hash[0][22],stage1Hash[0][23],stage1Hash[0][24],stage1Hash[0][25]], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_2[i]; // Unused transcript values
    }

    
    signal transcriptHash_3[16] <== Poseidon2(4, 16)([stage1Hash[0][26],stage1Hash[0][27],stage1Hash[0][28],stage1Hash[0][29],stage1Hash[0][30],stage1Hash[0][31],stage1Hash[0][32],stage1Hash[0][33],stage1Hash[0][34],stage1Hash[0][35],stage1Hash[0][36],stage1Hash[0][37]], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_3[i]; // Unused transcript values
    }

    
    signal transcriptHash_4[16] <== Poseidon2(4, 16)([stage1Hash[0][38],stage1Hash[0][39],stage1Hash[0][40],stage1Hash[0][41],stage1Hash[0][42],stage1Hash[0][43],stage1Hash[0][44],stage1Hash[0][45],stage1Hash[0][46],stage1Hash[0][47],stage1Hash[0][48],stage1Hash[0][49]], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_4[i]; // Unused transcript values
    }

    
    signal transcriptHash_5[16] <== Poseidon2(4, 16)([stage1Hash[0][50],stage1Hash[0][51],stage1Hash[0][52],stage1Hash[0][53],stage1Hash[0][54],stage1Hash[0][55],stage1Hash[0][56],stage1Hash[0][57],stage1Hash[0][58],stage1Hash[0][59],stage1Hash[0][60],stage1Hash[0][61]], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_5[i]; // Unused transcript values
    }

    
    signal transcriptHash_6[16] <== Poseidon2(4, 16)([stage1Hash[0][62],stage1Hash[0][63],stage1Hash[0][64],stage1Hash[0][65],stage1Hash[0][66],stage1Hash[0][67],stage1Hash[0][68],stage1Hash[0][69],stage1Hash[0][70],stage1Hash[0][71],stage1Hash[0][72],stage1Hash[0][73]], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_6[i]; // Unused transcript values
    }

    
    signal transcriptHash_7[16] <== Poseidon2(4, 16)([stage1Hash[0][74],stage1Hash[0][75],stage1Hash[0][76],stage1Hash[0][77],stage1Hash[0][78],stage1Hash[0][79],stage1Hash[0][80],stage1Hash[0][81],stage1Hash[0][82],stage1Hash[0][83],stage1Hash[0][84],stage1Hash[0][85]], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_7[i]; // Unused transcript values
    }

    
    signal transcriptHash_8[16] <== Poseidon2(4, 16)([stage1Hash[0][86],stage1Hash[0][87],stage1Hash[0][88],stage1Hash[0][89],stage1Hash[0][90],stage1Hash[0][91],stage1Hash[0][92],stage1Hash[0][93],stage1Hash[0][94],stage1Hash[0][95],stage1Hash[0][96],stage1Hash[0][97]], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_8[i]; // Unused transcript values
    }

    
    signal transcriptHash_9[16] <== Poseidon2(4, 16)([stage1Hash[0][98],stage1Hash[0][99],stage1Hash[0][100],stage1Hash[0][101],stage1Hash[0][102],stage1Hash[0][103],stage1Hash[0][104],stage1Hash[0][105],stage1Hash[0][106],stage1Hash[0][107],stage1Hash[0][108],stage1Hash[0][109]], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_9[i]; // Unused transcript values
    }

    
    signal transcriptHash_10[16] <== Poseidon2(4, 16)([stage1Hash[0][110],stage1Hash[0][111],stage1Hash[0][112],stage1Hash[0][113],stage1Hash[0][114],stage1Hash[0][115],stage1Hash[0][116],stage1Hash[0][117],stage1Hash[0][118],stage1Hash[0][119],stage1Hash[0][120],stage1Hash[0][121]], [transcriptHash_9[0],transcriptHash_9[1],transcriptHash_9[2],transcriptHash_9[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_10[i]; // Unused transcript values
    }

    
    signal transcriptHash_11[16] <== Poseidon2(4, 16)([stage1Hash[0][122],stage1Hash[0][123],stage1Hash[0][124],stage1Hash[0][125],stage1Hash[0][126],stage1Hash[0][127],stage1Hash[0][128],stage1Hash[0][129],stage1Hash[0][130],stage1Hash[0][131],stage1Hash[0][132],stage1Hash[0][133]], [transcriptHash_10[0],transcriptHash_10[1],transcriptHash_10[2],transcriptHash_10[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_11[i]; // Unused transcript values
    }

    
    signal transcriptHash_12[16] <== Poseidon2(4, 16)([stage1Hash[0][134],stage1Hash[0][135],stage1Hash[0][136],stage1Hash[0][137],stage1Hash[0][138],stage1Hash[0][139],stage1Hash[0][140],stage1Hash[0][141],stage1Hash[0][142],stage1Hash[0][143],stage1Hash[0][144],stage1Hash[0][145]], [transcriptHash_11[0],transcriptHash_11[1],transcriptHash_11[2],transcriptHash_11[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_12[i]; // Unused transcript values
    }

    
    signal transcriptHash_13[16] <== Poseidon2(4, 16)([stage1Hash[0][146],stage1Hash[0][147],stage1Hash[0][148],stage1Hash[0][149],stage1Hash[0][150],stage1Hash[0][151],stage1Hash[0][152],stage1Hash[0][153],stage1Hash[0][154],stage1Hash[0][155],stage1Hash[0][156],stage1Hash[0][157]], [transcriptHash_12[0],transcriptHash_12[1],transcriptHash_12[2],transcriptHash_12[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_13[i]; // Unused transcript values
    }

    
    signal transcriptHash_14[16] <== Poseidon2(4, 16)([stage1Hash[0][158],stage1Hash[0][159],stage1Hash[0][160],stage1Hash[0][161],stage1Hash[0][162],stage1Hash[0][163],stage1Hash[0][164],stage1Hash[0][165],stage1Hash[0][166],stage1Hash[0][167],stage1Hash[0][168],stage1Hash[0][169]], [transcriptHash_13[0],transcriptHash_13[1],transcriptHash_13[2],transcriptHash_13[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_14[i]; // Unused transcript values
    }

    
    signal transcriptHash_15[16] <== Poseidon2(4, 16)([stage1Hash[0][170],stage1Hash[0][171],stage1Hash[0][172],stage1Hash[0][173],stage1Hash[0][174],stage1Hash[0][175],stage1Hash[0][176],stage1Hash[0][177],stage1Hash[0][178],stage1Hash[0][179],stage1Hash[0][180],stage1Hash[0][181]], [transcriptHash_14[0],transcriptHash_14[1],transcriptHash_14[2],transcriptHash_14[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_15[i]; // Unused transcript values
    }

    
    signal transcriptHash_16[16] <== Poseidon2(4, 16)([stage1Hash[0][182],stage1Hash[0][183],stage1Hash[0][184],stage1Hash[0][185],stage1Hash[0][186],stage1Hash[0][187],stage1Hash[0][188],stage1Hash[0][189],stage1Hash[0][190],stage1Hash[0][191],stage1Hash[0][192],stage1Hash[0][193]], [transcriptHash_15[0],transcriptHash_15[1],transcriptHash_15[2],transcriptHash_15[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_16[i]; // Unused transcript values
    }

    
    signal transcriptHash_17[16] <== Poseidon2(4, 16)([stage1Hash[0][194],stage1Hash[0][195],stage1Hash[0][196],stage1Hash[0][197],stage1Hash[0][198],stage1Hash[0][199],stage1Hash[0][200],stage1Hash[0][201],stage1Hash[0][202],stage1Hash[0][203],stage1Hash[0][204],stage1Hash[0][205]], [transcriptHash_16[0],transcriptHash_16[1],transcriptHash_16[2],transcriptHash_16[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_17[i]; // Unused transcript values
    }

    
    signal transcriptHash_18[16] <== Poseidon2(4, 16)([stage1Hash[0][206],stage1Hash[0][207],stage1Hash[0][208],stage1Hash[0][209],stage1Hash[0][210],stage1Hash[0][211],stage1Hash[0][212],stage1Hash[0][213],stage1Hash[0][214],stage1Hash[0][215],stage1Hash[0][216],stage1Hash[0][217]], [transcriptHash_17[0],transcriptHash_17[1],transcriptHash_17[2],transcriptHash_17[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_18[i]; // Unused transcript values
    }

    
    signal transcriptHash_19[16] <== Poseidon2(4, 16)([stage1Hash[0][218],stage1Hash[0][219],stage1Hash[0][220],stage1Hash[0][221],stage1Hash[0][222],stage1Hash[0][223],stage1Hash[0][224],stage1Hash[0][225],stage1Hash[0][226],stage1Hash[0][227],stage1Hash[0][228],stage1Hash[0][229]], [transcriptHash_18[0],transcriptHash_18[1],transcriptHash_18[2],transcriptHash_18[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_19[i]; // Unused transcript values
    }

    
    signal transcriptHash_20[16] <== Poseidon2(4, 16)([stage1Hash[0][230],stage1Hash[0][231],stage1Hash[0][232],stage1Hash[0][233],stage1Hash[0][234],stage1Hash[0][235],stage1Hash[0][236],stage1Hash[0][237],stage1Hash[0][238],stage1Hash[0][239],stage1Hash[0][240],stage1Hash[0][241]], [transcriptHash_19[0],transcriptHash_19[1],transcriptHash_19[2],transcriptHash_19[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_20[i]; // Unused transcript values
    }

    
    signal transcriptHash_21[16] <== Poseidon2(4, 16)([stage1Hash[0][242],stage1Hash[0][243],stage1Hash[0][244],stage1Hash[0][245],stage1Hash[0][246],stage1Hash[0][247],stage1Hash[0][248],stage1Hash[0][249],stage1Hash[0][250],stage1Hash[0][251],stage1Hash[0][252],stage1Hash[0][253]], [transcriptHash_20[0],transcriptHash_20[1],transcriptHash_20[2],transcriptHash_20[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_21[i]; // Unused transcript values
    }

    
    signal transcriptHash_22[16] <== Poseidon2(4, 16)([stage1Hash[0][254],stage1Hash[0][255],stage1Hash[0][256],stage1Hash[0][257],stage1Hash[0][258],stage1Hash[0][259],stage1Hash[0][260],stage1Hash[0][261],stage1Hash[0][262],stage1Hash[0][263],stage1Hash[0][264],stage1Hash[0][265]], [transcriptHash_21[0],transcriptHash_21[1],transcriptHash_21[2],transcriptHash_21[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_22[i]; // Unused transcript values
    }

    
    signal transcriptHash_23[16] <== Poseidon2(4, 16)([stage1Hash[0][266],stage1Hash[0][267],stage1Hash[0][268],stage1Hash[0][269],stage1Hash[0][270],stage1Hash[0][271],stage1Hash[0][272],stage1Hash[0][273],stage1Hash[0][274],stage1Hash[0][275],stage1Hash[0][276],stage1Hash[0][277]], [transcriptHash_22[0],transcriptHash_22[1],transcriptHash_22[2],transcriptHash_22[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_23[i]; // Unused transcript values
    }

    
    signal transcriptHash_24[16] <== Poseidon2(4, 16)([stage1Hash[0][278],stage1Hash[0][279],stage1Hash[0][280],stage1Hash[0][281],stage1Hash[0][282],stage1Hash[0][283],stage1Hash[0][284],stage1Hash[0][285],stage1Hash[0][286],stage1Hash[0][287],stage1Hash[0][288],stage1Hash[0][289]], [transcriptHash_23[0],transcriptHash_23[1],transcriptHash_23[2],transcriptHash_23[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_24[i]; // Unused transcript values
    }

    
    signal transcriptHash_25[16] <== Poseidon2(4, 16)([stage1Hash[0][290],stage1Hash[0][291],stage1Hash[0][292],stage1Hash[0][293],stage1Hash[0][294],stage1Hash[0][295],stage1Hash[0][296],stage1Hash[0][297],stage1Hash[0][298],stage1Hash[0][299],stage1Hash[0][300],stage1Hash[0][301]], [transcriptHash_24[0],transcriptHash_24[1],transcriptHash_24[2],transcriptHash_24[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_25[i]; // Unused transcript values
    }

    
    signal transcriptHash_26[16] <== Poseidon2(4, 16)([stage1Hash[0][302],stage1Hash[0][303],stage1Hash[0][304],stage1Hash[0][305],stage1Hash[0][306],stage1Hash[0][307],stage1Hash[0][308],stage1Hash[0][309],stage1Hash[0][310],stage1Hash[0][311],stage1Hash[0][312],stage1Hash[0][313]], [transcriptHash_25[0],transcriptHash_25[1],transcriptHash_25[2],transcriptHash_25[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_26[i]; // Unused transcript values
    }

    
    signal transcriptHash_27[16] <== Poseidon2(4, 16)([stage1Hash[0][314],stage1Hash[0][315],stage1Hash[0][316],stage1Hash[0][317],stage1Hash[0][318],stage1Hash[0][319],stage1Hash[0][320],stage1Hash[0][321],stage1Hash[0][322],stage1Hash[0][323],stage1Hash[0][324],stage1Hash[0][325]], [transcriptHash_26[0],transcriptHash_26[1],transcriptHash_26[2],transcriptHash_26[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_27[i]; // Unused transcript values
    }

    
    signal transcriptHash_28[16] <== Poseidon2(4, 16)([stage1Hash[0][326],stage1Hash[0][327],stage1Hash[0][328],stage1Hash[0][329],stage1Hash[0][330],stage1Hash[0][331],stage1Hash[0][332],stage1Hash[0][333],stage1Hash[0][334],stage1Hash[0][335],stage1Hash[0][336],stage1Hash[0][337]], [transcriptHash_27[0],transcriptHash_27[1],transcriptHash_27[2],transcriptHash_27[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_28[i]; // Unused transcript values
    }

    
    signal transcriptHash_29[16] <== Poseidon2(4, 16)([stage1Hash[0][338],stage1Hash[0][339],stage1Hash[0][340],stage1Hash[0][341],stage1Hash[0][342],stage1Hash[0][343],stage1Hash[0][344],stage1Hash[0][345],stage1Hash[0][346],stage1Hash[0][347],stage1Hash[0][348],stage1Hash[0][349]], [transcriptHash_28[0],transcriptHash_28[1],transcriptHash_28[2],transcriptHash_28[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_29[i]; // Unused transcript values
    }

    
    signal transcriptHash_30[16] <== Poseidon2(4, 16)([stage1Hash[0][350],stage1Hash[0][351],stage1Hash[0][352],stage1Hash[0][353],stage1Hash[0][354],stage1Hash[0][355],stage1Hash[0][356],stage1Hash[0][357],stage1Hash[0][358],stage1Hash[0][359],stage1Hash[0][360],stage1Hash[0][361]], [transcriptHash_29[0],transcriptHash_29[1],transcriptHash_29[2],transcriptHash_29[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_30[i]; // Unused transcript values
    }

    
    signal transcriptHash_31[16] <== Poseidon2(4, 16)([stage1Hash[0][362],stage1Hash[0][363],stage1Hash[0][364],stage1Hash[0][365],stage1Hash[0][366],stage1Hash[0][367],0,0,0,0,0,0], [transcriptHash_30[0],transcriptHash_30[1],transcriptHash_30[2],transcriptHash_30[3]]);
    for (var i = 3; i < 16; i++) {
        _ <== transcriptHash_31[i]; // Unused transcript values
    }

    calculatedGlobalChallenge <== [transcriptHash_31[0], transcriptHash_31[1], transcriptHash_31[2]];

    globalChallenge === calculatedGlobalChallenge;
}






template GlobalConstraint0_chunk0() {
    signal input s0_airgroupvalues[1][3];
    signal input publics[8];
    signal input proofValues[2][3];

    signal input challenges[2][3];


    signal output tmp_3;
 

    signal tmp_2 <== publics[0] * proofValues[0][0];
    tmp_3 <== tmp_2 - proofValues[1][0];
}
template GlobalConstraint1_chunk0() {
    signal input s0_airgroupvalues[1][3];
    signal input publics[8];
    signal input proofValues[2][3];

    signal input challenges[2][3];


    signal output tmp_5[3];
 

    tmp_5 <== [challenges[0][0] - challenges[0][0], challenges[0][1] - challenges[0][1], challenges[0][2] - challenges[0][2]];
}
template GlobalConstraint2_chunk0() {
    signal input s0_airgroupvalues[1][3];
    signal input publics[8];
    signal input proofValues[2][3];

    signal input challenges[2][3];


    signal output tmp_7[3];
 

    tmp_7 <== [s0_airgroupvalues[0][0] * 1, s0_airgroupvalues[0][1] * 1, s0_airgroupvalues[0][2] * 1];
}

template VerifyGlobalConstraints() {

    signal input s0_airgroupvalues[1][3];
    signal input publics[8];
    signal input proofValues[2][3];

    signal input globalChallenge[3];

    signal challenges[2][3];
    
    signal transcriptHash_0[16] <== Poseidon2(4, 16)([globalChallenge[0],globalChallenge[1],globalChallenge[2],0,0,0,0,0,0,0,0,0], [0,0,0,0]);
    challenges[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challenges[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];

    // Verify global constraints
    signal output tmp_3;
 
    (tmp_3) <== GlobalConstraint0_chunk0()(s0_airgroupvalues,publics,proofValues,challenges);
    tmp_3 === 0;
    signal output tmp_5[3];
 
    (tmp_5) <== GlobalConstraint1_chunk0()(s0_airgroupvalues,publics,proofValues,challenges);
    tmp_5[0] === 0;
    tmp_5[1] === 0;
    tmp_5[2] === 0;
    signal output tmp_7[3];
 
    (tmp_7) <== GlobalConstraint2_chunk0()(s0_airgroupvalues,publics,proofValues,challenges);
    tmp_7[0] === 0;
    tmp_7[1] === 0;
    tmp_7[2] === 0;
}

template Final() {

    signal input publics[8];

    signal input proofValues[2][3];

    signal input globalChallenge[3];


    signal input s0_sv_circuitType;

    signal input s0_sv_aggregatedProofs;

    signal input s0_sv_aggregationTypes[1];
    signal input s0_sv_airgroupvalues[1][3];

    signal input s0_sv_stage1Hash[368];



    signal input s0_root1[4];
    signal input s0_root2[4];
    signal input s0_root3[4];

    signal input s0_evals[145][3]; // Evaluations of the set polynomials at a challenge value z and gz

    signal input s0_s0_valsC[73][49];
    signal input s0_s0_siblingsC[73][9][12];
    signal input s0_s0_last_mt_levelsC[4][4];


    signal input s0_s0_vals1[73][59];
    signal input s0_s0_siblings1[73][9][12];
    signal input s0_s0_last_mt_levels1[4][4];
    signal input s0_s0_vals2[73][12];
    signal input s0_s0_siblings2[73][9][12];
    signal input s0_s0_last_mt_levels2[4][4];
    signal input s0_s0_vals3[73][21];
    signal input s0_s0_siblings3[73][9][12];
    signal input s0_s0_last_mt_levels3[4][4];

    signal input s0_s1_root[4];
    signal input s0_s2_root[4];
    signal input s0_s3_root[4];
    signal input s0_s4_root[4];
    signal input s0_s5_root[4];

    signal input s0_s1_vals[73][24];
    signal input s0_s1_siblings[73][8][12];
    signal input s0_s1_last_mt_levels[4][4];
    signal input s0_s2_vals[73][24];
    signal input s0_s2_siblings[73][6][12];
    signal input s0_s2_last_mt_levels[4][4];
    signal input s0_s3_vals[73][24];
    signal input s0_s3_siblings[73][5][12];
    signal input s0_s3_last_mt_levels[4][4];
    signal input s0_s4_vals[73][24];
    signal input s0_s4_siblings[73][3][12];
    signal input s0_s4_last_mt_levels[4][4];
    signal input s0_s5_vals[73][24];
    signal input s0_s5_siblings[73][2][12];
    signal input s0_s5_last_mt_levels[4][4];

    signal input s0_finalPol[32][3];

    signal input s0_nonce;



    component sV0 = StarkVerifier0();





    sV0.root1 <== s0_root1;
    sV0.root2 <== s0_root2;
    sV0.root3 <== s0_root3;

    sV0.evals <== s0_evals;

    sV0.s0_valsC <== s0_s0_valsC;
    sV0.s0_siblingsC <== s0_s0_siblingsC;
    sV0.s0_last_mt_levelsC <== s0_s0_last_mt_levelsC;


    sV0.s0_vals1 <== s0_s0_vals1;
    sV0.s0_siblings1 <== s0_s0_siblings1;
    sV0.s0_last_mt_levels1 <== s0_s0_last_mt_levels1;
    sV0.s0_vals2 <== s0_s0_vals2;
    sV0.s0_siblings2 <== s0_s0_siblings2;
    sV0.s0_last_mt_levels2 <== s0_s0_last_mt_levels2;
    sV0.s0_vals3 <== s0_s0_vals3;
    sV0.s0_siblings3 <== s0_s0_siblings3;
    sV0.s0_last_mt_levels3 <== s0_s0_last_mt_levels3;

    sV0.s1_root <== s0_s1_root;
    sV0.s2_root <== s0_s2_root;
    sV0.s3_root <== s0_s3_root;
    sV0.s4_root <== s0_s4_root;
    sV0.s5_root <== s0_s5_root;
    sV0.s1_vals <== s0_s1_vals;
    sV0.s1_siblings <== s0_s1_siblings;
    sV0.s1_last_mt_levels <== s0_s1_last_mt_levels;
    sV0.s2_vals <== s0_s2_vals;
    sV0.s2_siblings <== s0_s2_siblings;
    sV0.s2_last_mt_levels <== s0_s2_last_mt_levels;
    sV0.s3_vals <== s0_s3_vals;
    sV0.s3_siblings <== s0_s3_siblings;
    sV0.s3_last_mt_levels <== s0_s3_last_mt_levels;
    sV0.s4_vals <== s0_s4_vals;
    sV0.s4_siblings <== s0_s4_siblings;
    sV0.s4_last_mt_levels <== s0_s4_last_mt_levels;
    sV0.s5_vals <== s0_s5_vals;
    sV0.s5_siblings <== s0_s5_siblings;
    sV0.s5_last_mt_levels <== s0_s5_last_mt_levels;

    sV0.finalPol <== s0_finalPol;
    sV0.nonce <== s0_nonce;



    sV0.publics[0] <== s0_sv_circuitType;

    sV0.publics[1] <== s0_sv_aggregatedProofs;

    for(var i = 0; i < 1; i++) {
        sV0.publics[2 + i] <== s0_sv_aggregationTypes[i];
    }

    for(var i = 0; i < 1; i++) {
        sV0.publics[3 + 3*i] <== s0_sv_airgroupvalues[i][0];
        sV0.publics[3 + 3*i + 1] <== s0_sv_airgroupvalues[i][1];
        sV0.publics[3 + 3*i + 2] <== s0_sv_airgroupvalues[i][2];
    }

    for (var i = 0; i < 368; i++) {
        sV0.publics[6 + i] <== s0_sv_stage1Hash[i];
    }

    for(var i = 0; i < 8; i++) {
        sV0.publics[374 + i] <== publics[i];
    }

    for(var i = 0; i < 2; i++) {
        sV0.publics[382 + 3*i] <== proofValues[i][0];
        sV0.publics[382 + 3*i + 1] <== proofValues[i][1];
        sV0.publics[382 + 3*i + 2] <== proofValues[i][2];

    }

    sV0.publics[388] <== globalChallenge[0];
    sV0.publics[388 +1] <== globalChallenge[1];
    sV0.publics[388 +2] <== globalChallenge[2];

    signal {binary} s0_sv_isNull <== IsZero()(s0_sv_circuitType);

    sV0.enable <== 1 - s0_sv_isNull;


    var s0_sv_rootCAgg[4] = [3323133768409455890,17015215400481695111,4033101897939059200,9095535402787741050];
    var s0_sv_rootCBasics[3][4];

    s0_sv_rootCBasics[0] = [3248249964652270553,7451285207662492861,18311142931443783459,15212363843520699034];
    s0_sv_rootCBasics[1] = [13286600280173231541,9872213368401534385,13266461048166577248,13568061374073846954];
    s0_sv_rootCBasics[2] = [8927598614966140770,14022275257384979796,14848462804449467272,6573124761383370019];

 
    sV0.rootC <== SelectVerificationKeyNull(3)(s0_sv_circuitType, s0_sv_rootCBasics, s0_sv_rootCAgg);

    for (var i=0; i<4; i++) {
        sV0.publics[391 + i] <== s0_sv_rootCAgg[i];
    }

    // Calculate transcript and check that matches with the global challenges
    component verifyChallenges = VerifyGlobalChallenges();
    verifyChallenges.publics <== publics;
    verifyChallenges.proofValues <== proofValues;
    verifyChallenges.stage1Hash[0] <== s0_sv_stage1Hash;
    verifyChallenges.globalChallenge <== globalChallenge;

    // Verify global constraints
    component verifyGlobalConstraints = VerifyGlobalConstraints();
    verifyGlobalConstraints.publics <== publics;
    verifyGlobalConstraints.proofValues <== proofValues;
    verifyGlobalConstraints.globalChallenge <== globalChallenge;
    verifyGlobalConstraints.s0_airgroupvalues <== s0_sv_airgroupvalues;

    signal {binary} isNull[1];
    signal nAggregatedProofs[1];
    isNull[0] <== IsZero()(s0_sv_circuitType);
    nAggregatedProofs[0] <== s0_sv_aggregatedProofs;
    _ <== AggregateProofsNull(1)(nAggregatedProofs, isNull);

}

component main {public [publics]}= Final();
