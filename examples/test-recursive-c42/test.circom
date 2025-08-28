pragma circom 2.1.0;
pragma custom_templates;

include "iszero.circom";
include "test.verifier.circom";


include "elliptic_curve.circom";

template CalculateStage1HashToEC() {
    signal input rootC[4];
    signal input root1[4];

    signal input airValues[3][3];

    signal output P[2][5];

    _ <== airValues[2]; // Unused air values at stage 1

    signal transcriptHash_0[12] <== Poseidon2(12)([rootC[0],rootC[1],rootC[2],rootC[3],root1[0],root1[1],root1[2],root1[3]], [0,0,0,0]);
    for (var i = 4; i < 12; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values
    }

    signal transcriptHash_1[12] <== Poseidon2(12)([airValues[0][0],airValues[1][0],0,0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    for (var i = 10; i < 12; i++) {
        _ <== transcriptHash_1[i]; // Unused transcript values
    }

    signal x[5] <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2], transcriptHash_1[3], transcriptHash_1[4]];
    signal y[5] <== [transcriptHash_1[5], transcriptHash_1[6], transcriptHash_1[7], transcriptHash_1[8], transcriptHash_1[9]];

    // Constants for the EcGFp5 curve
    var A[5] = [6148914689804861439,263,0,0,0];
    var B[5] = [15713893096167979237,6148914689804861265,0,0,0];
    var Z[5] = [18446744069414584317,18446744069414584320,0,0,0];
    var C1[5] = [6585749426319121644,16990361517133133838,3264760655763595284,16784740989273302855,13434657726302040770];
    var C2[5] = [4795794222525505369,3412737461722269738,8370187669276724726,7130825117388110979,12052351772713910496];
    P <== HashToCurve(A, B, Z, C1, C2)(x,y);
}


template Recursive1() {


    signal output sv_circuitType;

    signal output sv_aggregationTypes[1];
    signal output sv_airgroupvalues[1][3];

    signal output sv_stage1HashToEC[2][5];


    signal input airgroupvalues[1][3];

    signal input airvalues[3][3];

    signal input root1[4];
    signal input root2[4];
    signal input root3[4];

    signal input evals[20][3]; // Evaluations of the set polynomials at a challenge value z and gz

    signal input s0_valsC[128][2];
    signal input s0_siblingsC[128][15][8];

    signal input s0_vals_rom_0[128][2];
    signal input s0_siblings_rom_0[128][15][8];

    signal input s0_vals1[128][2];
    signal input s0_siblings1[128][15][8];
    signal input s0_vals2[128][9];
    signal input s0_siblings2[128][15][8];
    signal input s0_vals3[128][3];
    signal input s0_siblings3[128][15][8];

    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];

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

    sV.finalPol <== finalPol;


    


    sV.globalChallenge <== globalChallenge;

    // --> Assign the VADCOP data
    sv_circuitType <== 2;
    
    sv_aggregationTypes <== [0];

    sv_airgroupvalues[0] <== airgroupvalues[0];

    sv_stage1HashToEC <== CalculateStage1HashToEC()(sV.rootC, root1, airvalues);

}

    
component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive1();

