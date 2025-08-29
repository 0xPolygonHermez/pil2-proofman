pragma circom 2.1.0;
pragma custom_templates;

include "test.verifier.circom";
include "iszero.circom";
include "select_vk.circom";
include "agg_values.circom";


template VerifyGlobalChallenges() {

    signal input publics[8];
    signal input proofValues[2][3];
    signal input stage1HashToEC[1][2][5];
    
    signal input globalChallenge[3];
    signal calculatedGlobalChallenge[3];

    
    signal transcriptHash_0[12] <== Poseidon2(12)([publics[0],publics[1],publics[2],publics[3],publics[4],publics[5],publics[6],publics[7]], [0,0,0,0]);
    for (var i = 4; i < 12; i++) {
        _ <== transcriptHash_0[i]; // Unused transcript values
    }

    
    signal transcriptHash_1[12] <== Poseidon2(12)([proofValues[0][0],proofValues[1][0],stage1HashToEC[0][0][0],stage1HashToEC[0][0][1],stage1HashToEC[0][0][2],stage1HashToEC[0][0][3],stage1HashToEC[0][0][4],stage1HashToEC[0][1][0]], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    for (var i = 4; i < 12; i++) {
        _ <== transcriptHash_1[i]; // Unused transcript values
    }

    
    signal transcriptHash_2[12] <== Poseidon2(12)([stage1HashToEC[0][1][1],stage1HashToEC[0][1][2],stage1HashToEC[0][1][3],stage1HashToEC[0][1][4],0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    for (var i = 3; i < 12; i++) {
        _ <== transcriptHash_2[i]; // Unused transcript values
    }

    calculatedGlobalChallenge <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];

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
    
    signal transcriptHash_0[12] <== Poseidon2(12)([globalChallenge[0],globalChallenge[1],globalChallenge[2],0,0,0,0,0], [0,0,0,0]);
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

    signal input s0_sv_stage1HashToEC[2][5];



    signal input s0_root1[4];
    signal input s0_root2[4];
    signal input s0_root3[4];

    signal input s0_evals[136][3]; // Evaluations of the set polynomials at a challenge value z and gz

    signal input s0_s0_valsC[43][45];
    signal input s0_s0_siblingsC[43][13][8];


    signal input s0_s0_vals1[43][36];
    signal input s0_s0_siblings1[43][13][8];
    signal input s0_s0_vals2[43][12];
    signal input s0_s0_siblings2[43][13][8];
    signal input s0_s0_vals3[43][21];
    signal input s0_s0_siblings3[43][13][8];

    signal input s0_s1_root[4];
    signal input s0_s2_root[4];
    signal input s0_s3_root[4];
    signal input s0_s4_root[4];

    signal input s0_s1_vals[43][48];
    signal input s0_s1_siblings[43][11][8];
    signal input s0_s2_vals[43][48];
    signal input s0_s2_siblings[43][8][8];
    signal input s0_s3_vals[43][48];
    signal input s0_s3_siblings[43][6][8];
    signal input s0_s4_vals[43][24];
    signal input s0_s4_siblings[43][4][8];

    signal input s0_finalPol[32][3];



    component sV0 = StarkVerifier0();





    sV0.root1 <== s0_root1;
    sV0.root2 <== s0_root2;
    sV0.root3 <== s0_root3;

    sV0.evals <== s0_evals;

    sV0.s0_valsC <== s0_s0_valsC;
    sV0.s0_siblingsC <== s0_s0_siblingsC;


    sV0.s0_vals1 <== s0_s0_vals1;
    sV0.s0_siblings1 <== s0_s0_siblings1;
    sV0.s0_vals2 <== s0_s0_vals2;
    sV0.s0_siblings2 <== s0_s0_siblings2;
    sV0.s0_vals3 <== s0_s0_vals3;
    sV0.s0_siblings3 <== s0_s0_siblings3;

    sV0.s1_root <== s0_s1_root;
    sV0.s2_root <== s0_s2_root;
    sV0.s3_root <== s0_s3_root;
    sV0.s4_root <== s0_s4_root;
    sV0.s1_vals <== s0_s1_vals;
    sV0.s1_siblings <== s0_s1_siblings;
    sV0.s2_vals <== s0_s2_vals;
    sV0.s2_siblings <== s0_s2_siblings;
    sV0.s3_vals <== s0_s3_vals;
    sV0.s3_siblings <== s0_s3_siblings;
    sV0.s4_vals <== s0_s4_vals;
    sV0.s4_siblings <== s0_s4_siblings;

    sV0.finalPol <== s0_finalPol;



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

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 5; j++) {
            sV0.publics[6 + 5*i + j] <== s0_sv_stage1HashToEC[i][j];
        }
    }

    for(var i = 0; i < 8; i++) {
        sV0.publics[16 + i] <== publics[i];
    }

    for(var i = 0; i < 2; i++) {
        sV0.publics[24 + 3*i] <== proofValues[i][0];
        sV0.publics[24 + 3*i + 1] <== proofValues[i][1];
        sV0.publics[24 + 3*i + 2] <== proofValues[i][2];

    }

    sV0.publics[30] <== globalChallenge[0];
    sV0.publics[30 +1] <== globalChallenge[1];
    sV0.publics[30 +2] <== globalChallenge[2];

    signal {binary} s0_sv_isNull <== IsZero()(s0_sv_circuitType);

    sV0.enable <== 1 - s0_sv_isNull;


    var s0_sv_rootCAgg[4] = [11626181605790575848,8853108949076634256,15998141566640291178,4573511429172678416];
    var s0_sv_rootCBasics[3][4];

    s0_sv_rootCBasics[0] = [12142227524572299132,7085954174139102859,2284939305328682278,788007566113243968];
    s0_sv_rootCBasics[1] = [531655343931675804,3718506447799678638,860429515158736850,16784892460450743376];
    s0_sv_rootCBasics[2] = [7429220990672227919,8561915747746241746,4909028731814883218,10821784384353250326];

 
    sV0.rootC <== SelectVerificationKeyNull(3)(s0_sv_circuitType, s0_sv_rootCBasics, s0_sv_rootCAgg);

    for (var i=0; i<4; i++) {
        sV0.publics[33 + i] <== s0_sv_rootCAgg[i];
    }

    // Calculate transcript and check that matches with the global challenges
    component verifyChallenges = VerifyGlobalChallenges();
    verifyChallenges.publics <== publics;
    verifyChallenges.proofValues <== proofValues;
    verifyChallenges.stage1HashToEC[0] <== s0_sv_stage1HashToEC;
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
