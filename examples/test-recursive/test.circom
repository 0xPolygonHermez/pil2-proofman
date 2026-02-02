pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "test.verifier.circom";



template Recursive1() {


    signal input sv_circuitType;

    signal input sv_aggregatedProofs;

    signal input sv_aggregationTypes[1];
    signal input sv_airgroupvalues[1][3];

    signal input sv_stage1Hash[368];




    signal input root1[4];
    signal input root2[4];
    signal input root3[4];

    signal input evals[198][3]; // Evaluations of the set polynomials at a challenge value z and gz

    signal input s0_valsC[110][58];
    signal input s0_siblingsC[110][8][12];
    signal input s0_last_mt_levelsC[16][4];


    signal input s0_vals1[110][52];
    signal input s0_siblings1[110][8][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[110][55];
    signal input s0_siblings2[110][8][12];
    signal input s0_last_mt_levels2[16][4];
    signal input s0_vals3[110][12];
    signal input s0_siblings3[110][8][12];
    signal input s0_last_mt_levels3[16][4];

    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];

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

    signal input finalPol[32][3];

    signal input nonce;


    signal input publics[8];
    
    signal input proofValues[2][3];
    
    signal input globalChallenge[3];

    signal input rootCAgg[4];



    component sV = StarkVerifier0();





    sV.root1 <== root1;
    sV.root2 <== root2;
    sV.root3 <== root3;

    sV.evals <== evals;

    sV.s0_valsC <== s0_valsC;
    sV.s0_siblingsC <== s0_siblingsC;
    sV.s0_last_mt_levelsC <== s0_last_mt_levelsC;


    sV.s0_vals1 <== s0_vals1;
    sV.s0_siblings1 <== s0_siblings1;
    sV.s0_last_mt_levels1 <== s0_last_mt_levels1;
    sV.s0_vals2 <== s0_vals2;
    sV.s0_siblings2 <== s0_siblings2;
    sV.s0_last_mt_levels2 <== s0_last_mt_levels2;
    sV.s0_vals3 <== s0_vals3;
    sV.s0_siblings3 <== s0_siblings3;
    sV.s0_last_mt_levels3 <== s0_last_mt_levels3;

    sV.s1_root <== s1_root;
    sV.s2_root <== s2_root;
    sV.s3_root <== s3_root;
    sV.s4_root <== s4_root;
    sV.s5_root <== s5_root;
    sV.s1_vals <== s1_vals;
    sV.s1_siblings <== s1_siblings;
    sV.s1_last_mt_levels <== s1_last_mt_levels;
    sV.s2_vals <== s2_vals;
    sV.s2_siblings <== s2_siblings;
    sV.s2_last_mt_levels <== s2_last_mt_levels;
    sV.s3_vals <== s3_vals;
    sV.s3_siblings <== s3_siblings;
    sV.s3_last_mt_levels <== s3_last_mt_levels;
    sV.s4_vals <== s4_vals;
    sV.s4_siblings <== s4_siblings;
    sV.s4_last_mt_levels <== s4_last_mt_levels;
    sV.s5_vals <== s5_vals;
    sV.s5_siblings <== s5_siblings;
    sV.s5_last_mt_levels <== s5_last_mt_levels;

    sV.finalPol <== finalPol;
    sV.nonce <== nonce;


    

    sV.publics[0] <== sv_circuitType;

    sV.publics[1] <== sv_aggregatedProofs;

    for(var i = 0; i < 1; i++) {
        sV.publics[2 + i] <== sv_aggregationTypes[i];
    }

    for(var i = 0; i < 1; i++) {
        sV.publics[3 + 3*i] <== sv_airgroupvalues[i][0];
        sV.publics[3 + 3*i + 1] <== sv_airgroupvalues[i][1];
        sV.publics[3 + 3*i + 2] <== sv_airgroupvalues[i][2];
    }

    for (var i = 0; i < 368; i++) {
        sV.publics[6 + i] <== sv_stage1Hash[i];
    }

    for(var i = 0; i < 8; i++) {
        sV.publics[374 + i] <== publics[i];
    }

    for(var i = 0; i < 2; i++) {
        sV.publics[382 + 3*i] <== proofValues[i][0];
        sV.publics[382 + 3*i + 1] <== proofValues[i][1];
        sV.publics[382 + 3*i + 2] <== proofValues[i][2];

    }

    sV.publics[388] <== globalChallenge[0];
    sV.publics[388 +1] <== globalChallenge[1];
    sV.publics[388 +2] <== globalChallenge[2];


}

    
component main {public [sv_circuitType, sv_aggregatedProofs, sv_aggregationTypes, sv_airgroupvalues, sv_stage1Hash, publics, proofValues, globalChallenge, rootCAgg]} = Recursive1();

