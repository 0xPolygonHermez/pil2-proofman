pragma circom 2.1.0;
pragma custom_templates;

include "select_vk.circom";
include "agg_values.circom";
include "acc_points.circom";

include "test.verifier.circom";

template Recursive2() {
    var rootCBasics[3][4];

    rootCBasics[0] = [401015088417336861,1839093663775868517,16290737316236619748,15328078766022373262];
    rootCBasics[1] = [14721882686236306867,11958401589948326049,4044159605597597871,3129498467973948371];
    rootCBasics[2] = [15787016206107144348,4082135310990252254,6020773157712459187,4490841628642139404];
    signal output sv_circuitType;
    signal output sv_aggregatedProofs;
    signal output sv_aggregationTypes[1];
    signal output sv_airgroupvalues[1][3];
    signal output sv_stage1Hash[368];

    signal input publics[8];

    signal input proofValues[2][3];

    signal input globalChallenge[3];
    signal input rootCAgg[4];

    signal input a_sv_circuitType;
    signal input a_sv_aggregatedProofs;
    signal input a_sv_aggregationTypes[1];
    signal input a_sv_airgroupvalues[1][3];
    signal input a_sv_stage1Hash[368];

    signal input a_root1[4];
    signal input a_root2[4];
    signal input a_root3[4];
    signal input a_evals[127][3];

    signal input a_s0_valsC[73][46];
    signal input a_s0_siblingsC[73][8][12];
    signal input a_s0_last_mt_levelsC[16][4];
    signal input a_s0_vals1[73][48];
    signal input a_s0_siblings1[73][8][12];
    signal input a_s0_last_mt_levels1[16][4];
    signal input a_s0_vals2[73][12];
    signal input a_s0_siblings2[73][8][12];
    signal input a_s0_last_mt_levels2[16][4];
    signal input a_s0_vals3[73][21];
    signal input a_s0_siblings3[73][8][12];
    signal input a_s0_last_mt_levels3[16][4];
    signal input a_s1_root[4];
    signal input a_s2_root[4];
    signal input a_s3_root[4];
    signal input a_s4_root[4];
    signal input a_s5_root[4];
    signal input a_s1_vals[73][24];
    signal input a_s1_siblings[73][7][12];
    signal input a_s1_last_mt_levels[16][4];
    signal input a_s2_vals[73][24];
    signal input a_s2_siblings[73][5][12];
    signal input a_s2_last_mt_levels[16][4];
    signal input a_s3_vals[73][24];
    signal input a_s3_siblings[73][4][12];
    signal input a_s3_last_mt_levels[16][4];
    signal input a_s4_vals[73][24];
    signal input a_s4_siblings[73][2][12];
    signal input a_s4_last_mt_levels[16][4];
    signal input a_s5_vals[73][24];
    signal input a_s5_siblings[73][1][12];
    signal input a_s5_last_mt_levels[16][4];
    signal input a_finalPol[32][3];
    signal input a_nonce;

    signal input b_sv_circuitType;
    signal input b_sv_aggregatedProofs;
    signal input b_sv_aggregationTypes[1];
    signal input b_sv_airgroupvalues[1][3];
    signal input b_sv_stage1Hash[368];

    signal input b_root1[4];
    signal input b_root2[4];
    signal input b_root3[4];
    signal input b_evals[127][3];

    signal input b_s0_valsC[73][46];
    signal input b_s0_siblingsC[73][8][12];
    signal input b_s0_last_mt_levelsC[16][4];
    signal input b_s0_vals1[73][48];
    signal input b_s0_siblings1[73][8][12];
    signal input b_s0_last_mt_levels1[16][4];
    signal input b_s0_vals2[73][12];
    signal input b_s0_siblings2[73][8][12];
    signal input b_s0_last_mt_levels2[16][4];
    signal input b_s0_vals3[73][21];
    signal input b_s0_siblings3[73][8][12];
    signal input b_s0_last_mt_levels3[16][4];
    signal input b_s1_root[4];
    signal input b_s2_root[4];
    signal input b_s3_root[4];
    signal input b_s4_root[4];
    signal input b_s5_root[4];
    signal input b_s1_vals[73][24];
    signal input b_s1_siblings[73][7][12];
    signal input b_s1_last_mt_levels[16][4];
    signal input b_s2_vals[73][24];
    signal input b_s2_siblings[73][5][12];
    signal input b_s2_last_mt_levels[16][4];
    signal input b_s3_vals[73][24];
    signal input b_s3_siblings[73][4][12];
    signal input b_s3_last_mt_levels[16][4];
    signal input b_s4_vals[73][24];
    signal input b_s4_siblings[73][2][12];
    signal input b_s4_last_mt_levels[16][4];
    signal input b_s5_vals[73][24];
    signal input b_s5_siblings[73][1][12];
    signal input b_s5_last_mt_levels[16][4];
    signal input b_finalPol[32][3];
    signal input b_nonce;

    signal input c_sv_circuitType;
    signal input c_sv_aggregatedProofs;
    signal input c_sv_aggregationTypes[1];
    signal input c_sv_airgroupvalues[1][3];
    signal input c_sv_stage1Hash[368];

    signal input c_root1[4];
    signal input c_root2[4];
    signal input c_root3[4];
    signal input c_evals[127][3];

    signal input c_s0_valsC[73][46];
    signal input c_s0_siblingsC[73][8][12];
    signal input c_s0_last_mt_levelsC[16][4];
    signal input c_s0_vals1[73][48];
    signal input c_s0_siblings1[73][8][12];
    signal input c_s0_last_mt_levels1[16][4];
    signal input c_s0_vals2[73][12];
    signal input c_s0_siblings2[73][8][12];
    signal input c_s0_last_mt_levels2[16][4];
    signal input c_s0_vals3[73][21];
    signal input c_s0_siblings3[73][8][12];
    signal input c_s0_last_mt_levels3[16][4];
    signal input c_s1_root[4];
    signal input c_s2_root[4];
    signal input c_s3_root[4];
    signal input c_s4_root[4];
    signal input c_s5_root[4];
    signal input c_s1_vals[73][24];
    signal input c_s1_siblings[73][7][12];
    signal input c_s1_last_mt_levels[16][4];
    signal input c_s2_vals[73][24];
    signal input c_s2_siblings[73][5][12];
    signal input c_s2_last_mt_levels[16][4];
    signal input c_s3_vals[73][24];
    signal input c_s3_siblings[73][4][12];
    signal input c_s3_last_mt_levels[16][4];
    signal input c_s4_vals[73][24];
    signal input c_s4_siblings[73][2][12];
    signal input c_s4_last_mt_levels[16][4];
    signal input c_s5_vals[73][24];
    signal input c_s5_siblings[73][1][12];
    signal input c_s5_last_mt_levels[16][4];
    signal input c_finalPol[32][3];
    signal input c_nonce;

    signal aggregationTypes[1];
    for(var i = 0; i < 1; i++) {
        aggregationTypes[i] <== a_sv_aggregationTypes[i];
        a_sv_aggregationTypes[i] === b_sv_aggregationTypes[i];
        a_sv_aggregationTypes[i] === c_sv_aggregationTypes[i];
    }
    component vA = parallel StarkVerifier0();
    vA.root1 <== a_root1;
    vA.root2 <== a_root2;
    vA.root3 <== a_root3;
    vA.evals <== a_evals;
    vA.s0_valsC <== a_s0_valsC;
    vA.s0_siblingsC <== a_s0_siblingsC;
    vA.s0_last_mt_levelsC <== a_s0_last_mt_levelsC;
    vA.s0_vals1 <== a_s0_vals1;
    vA.s0_siblings1 <== a_s0_siblings1;
    vA.s0_last_mt_levels1 <== a_s0_last_mt_levels1;
    vA.s0_vals2 <== a_s0_vals2;
    vA.s0_siblings2 <== a_s0_siblings2;
    vA.s0_last_mt_levels2 <== a_s0_last_mt_levels2;
    vA.s0_vals3 <== a_s0_vals3;
    vA.s0_siblings3 <== a_s0_siblings3;
    vA.s0_last_mt_levels3 <== a_s0_last_mt_levels3;
    vA.s1_root <== a_s1_root;
    vA.s2_root <== a_s2_root;
    vA.s3_root <== a_s3_root;
    vA.s4_root <== a_s4_root;
    vA.s5_root <== a_s5_root;
    vA.s1_vals <== a_s1_vals;
    vA.s1_siblings <== a_s1_siblings;
    vA.s1_last_mt_levels <== a_s1_last_mt_levels;
    vA.s2_vals <== a_s2_vals;
    vA.s2_siblings <== a_s2_siblings;
    vA.s2_last_mt_levels <== a_s2_last_mt_levels;
    vA.s3_vals <== a_s3_vals;
    vA.s3_siblings <== a_s3_siblings;
    vA.s3_last_mt_levels <== a_s3_last_mt_levels;
    vA.s4_vals <== a_s4_vals;
    vA.s4_siblings <== a_s4_siblings;
    vA.s4_last_mt_levels <== a_s4_last_mt_levels;
    vA.s5_vals <== a_s5_vals;
    vA.s5_siblings <== a_s5_siblings;
    vA.s5_last_mt_levels <== a_s5_last_mt_levels;
    vA.finalPol <== a_finalPol;
    vA.nonce <== a_nonce;

    component vB = parallel StarkVerifier0();
    vB.root1 <== b_root1;
    vB.root2 <== b_root2;
    vB.root3 <== b_root3;
    vB.evals <== b_evals;
    vB.s0_valsC <== b_s0_valsC;
    vB.s0_siblingsC <== b_s0_siblingsC;
    vB.s0_last_mt_levelsC <== b_s0_last_mt_levelsC;
    vB.s0_vals1 <== b_s0_vals1;
    vB.s0_siblings1 <== b_s0_siblings1;
    vB.s0_last_mt_levels1 <== b_s0_last_mt_levels1;
    vB.s0_vals2 <== b_s0_vals2;
    vB.s0_siblings2 <== b_s0_siblings2;
    vB.s0_last_mt_levels2 <== b_s0_last_mt_levels2;
    vB.s0_vals3 <== b_s0_vals3;
    vB.s0_siblings3 <== b_s0_siblings3;
    vB.s0_last_mt_levels3 <== b_s0_last_mt_levels3;
    vB.s1_root <== b_s1_root;
    vB.s2_root <== b_s2_root;
    vB.s3_root <== b_s3_root;
    vB.s4_root <== b_s4_root;
    vB.s5_root <== b_s5_root;
    vB.s1_vals <== b_s1_vals;
    vB.s1_siblings <== b_s1_siblings;
    vB.s1_last_mt_levels <== b_s1_last_mt_levels;
    vB.s2_vals <== b_s2_vals;
    vB.s2_siblings <== b_s2_siblings;
    vB.s2_last_mt_levels <== b_s2_last_mt_levels;
    vB.s3_vals <== b_s3_vals;
    vB.s3_siblings <== b_s3_siblings;
    vB.s3_last_mt_levels <== b_s3_last_mt_levels;
    vB.s4_vals <== b_s4_vals;
    vB.s4_siblings <== b_s4_siblings;
    vB.s4_last_mt_levels <== b_s4_last_mt_levels;
    vB.s5_vals <== b_s5_vals;
    vB.s5_siblings <== b_s5_siblings;
    vB.s5_last_mt_levels <== b_s5_last_mt_levels;
    vB.finalPol <== b_finalPol;
    vB.nonce <== b_nonce;

    component vC = parallel StarkVerifier0();
    vC.root1 <== c_root1;
    vC.root2 <== c_root2;
    vC.root3 <== c_root3;
    vC.evals <== c_evals;
    vC.s0_valsC <== c_s0_valsC;
    vC.s0_siblingsC <== c_s0_siblingsC;
    vC.s0_last_mt_levelsC <== c_s0_last_mt_levelsC;
    vC.s0_vals1 <== c_s0_vals1;
    vC.s0_siblings1 <== c_s0_siblings1;
    vC.s0_last_mt_levels1 <== c_s0_last_mt_levels1;
    vC.s0_vals2 <== c_s0_vals2;
    vC.s0_siblings2 <== c_s0_siblings2;
    vC.s0_last_mt_levels2 <== c_s0_last_mt_levels2;
    vC.s0_vals3 <== c_s0_vals3;
    vC.s0_siblings3 <== c_s0_siblings3;
    vC.s0_last_mt_levels3 <== c_s0_last_mt_levels3;
    vC.s1_root <== c_s1_root;
    vC.s2_root <== c_s2_root;
    vC.s3_root <== c_s3_root;
    vC.s4_root <== c_s4_root;
    vC.s5_root <== c_s5_root;
    vC.s1_vals <== c_s1_vals;
    vC.s1_siblings <== c_s1_siblings;
    vC.s1_last_mt_levels <== c_s1_last_mt_levels;
    vC.s2_vals <== c_s2_vals;
    vC.s2_siblings <== c_s2_siblings;
    vC.s2_last_mt_levels <== c_s2_last_mt_levels;
    vC.s3_vals <== c_s3_vals;
    vC.s3_siblings <== c_s3_siblings;
    vC.s3_last_mt_levels <== c_s3_last_mt_levels;
    vC.s4_vals <== c_s4_vals;
    vC.s4_siblings <== c_s4_siblings;
    vC.s4_last_mt_levels <== c_s4_last_mt_levels;
    vC.s5_vals <== c_s5_vals;
    vC.s5_siblings <== c_s5_siblings;
    vC.s5_last_mt_levels <== c_s5_last_mt_levels;
    vC.finalPol <== c_finalPol;
    vC.nonce <== c_nonce;

    vA.publics[0] <== a_sv_circuitType;
    vA.publics[1] <== a_sv_aggregatedProofs;
    for(var i = 0; i < 1; i++) {
        vA.publics[2 + i] <== a_sv_aggregationTypes[i];
    }
    for(var i = 0; i < 1; i++) {
        vA.publics[3 + 3*i] <== a_sv_airgroupvalues[i][0];
        vA.publics[3 + 3*i + 1] <== a_sv_airgroupvalues[i][1];
        vA.publics[3 + 3*i + 2] <== a_sv_airgroupvalues[i][2];
    }
    for (var i = 0; i < 368; i++) {
        vA.publics[6 + i] <== a_sv_stage1Hash[i];
    }
    for(var i = 0; i < 8; i++) {
        vA.publics[374 + i] <== publics[i];
    }
    for(var i = 0; i < 2; i++) {
        vA.publics[382 + 3*i] <== proofValues[i][0];
        vA.publics[382 + 3*i + 1] <== proofValues[i][1];
        vA.publics[382 + 3*i + 2] <== proofValues[i][2];
    }
    vA.publics[388] <== globalChallenge[0];
    vA.publics[388 +1] <== globalChallenge[1];
    vA.publics[388 +2] <== globalChallenge[2];
    signal {binary} a_sv_isNull <== IsZero()(a_sv_circuitType);
    vA.enable <== 1 - a_sv_isNull;

    vB.publics[0] <== b_sv_circuitType;
    vB.publics[1] <== b_sv_aggregatedProofs;
    for(var i = 0; i < 1; i++) {
        vB.publics[2 + i] <== b_sv_aggregationTypes[i];
    }
    for(var i = 0; i < 1; i++) {
        vB.publics[3 + 3*i] <== b_sv_airgroupvalues[i][0];
        vB.publics[3 + 3*i + 1] <== b_sv_airgroupvalues[i][1];
        vB.publics[3 + 3*i + 2] <== b_sv_airgroupvalues[i][2];
    }
    for (var i = 0; i < 368; i++) {
        vB.publics[6 + i] <== b_sv_stage1Hash[i];
    }
    for(var i = 0; i < 8; i++) {
        vB.publics[374 + i] <== publics[i];
    }
    for(var i = 0; i < 2; i++) {
        vB.publics[382 + 3*i] <== proofValues[i][0];
        vB.publics[382 + 3*i + 1] <== proofValues[i][1];
        vB.publics[382 + 3*i + 2] <== proofValues[i][2];
    }
    vB.publics[388] <== globalChallenge[0];
    vB.publics[388 +1] <== globalChallenge[1];
    vB.publics[388 +2] <== globalChallenge[2];
    signal {binary} b_sv_isNull <== IsZero()(b_sv_circuitType);
    vB.enable <== 1 - b_sv_isNull;

    vC.publics[0] <== c_sv_circuitType;
    vC.publics[1] <== c_sv_aggregatedProofs;
    for(var i = 0; i < 1; i++) {
        vC.publics[2 + i] <== c_sv_aggregationTypes[i];
    }
    for(var i = 0; i < 1; i++) {
        vC.publics[3 + 3*i] <== c_sv_airgroupvalues[i][0];
        vC.publics[3 + 3*i + 1] <== c_sv_airgroupvalues[i][1];
        vC.publics[3 + 3*i + 2] <== c_sv_airgroupvalues[i][2];
    }
    for (var i = 0; i < 368; i++) {
        vC.publics[6 + i] <== c_sv_stage1Hash[i];
    }
    for(var i = 0; i < 8; i++) {
        vC.publics[374 + i] <== publics[i];
    }
    for(var i = 0; i < 2; i++) {
        vC.publics[382 + 3*i] <== proofValues[i][0];
        vC.publics[382 + 3*i + 1] <== proofValues[i][1];
        vC.publics[382 + 3*i + 2] <== proofValues[i][2];
    }
    vC.publics[388] <== globalChallenge[0];
    vC.publics[388 +1] <== globalChallenge[1];
    vC.publics[388 +2] <== globalChallenge[2];
    signal {binary} c_sv_isNull <== IsZero()(c_sv_circuitType);
    vC.enable <== 1 - c_sv_isNull;

    vA.rootC <== SelectVerificationKeyNull(3)(a_sv_circuitType, rootCBasics, rootCAgg);
    vB.rootC <== SelectVerificationKeyNull(3)(b_sv_circuitType, rootCBasics, rootCAgg);
    vC.rootC <== SelectVerificationKeyNull(3)(c_sv_circuitType, rootCBasics, rootCAgg);

    sv_circuitType <== 1;

    sv_aggregationTypes <== aggregationTypes;
    signal {binary} aggTypes[1];
    for (var i = 0; i < 1; i++) {
        sv_aggregationTypes[i] * (sv_aggregationTypes[i] - 1) === 0;
        aggTypes[i] <== sv_aggregationTypes[i];
    }

    signal {binary} AB_isNull <== IsZero()(2 - a_sv_isNull - b_sv_isNull);
    signal airgroupValues_AB[1][3];
    for (var i = 0; i < 1; i++) {
        airgroupValues_AB[i] <== AggregateAirgroupValuesNull()(a_sv_airgroupvalues[i], b_sv_airgroupvalues[i], aggTypes[i], a_sv_isNull, b_sv_isNull);
        sv_airgroupvalues[i] <== AggregateAirgroupValuesNull()(airgroupValues_AB[i], c_sv_airgroupvalues[i], aggTypes[i], AB_isNull, c_sv_isNull);
    }

    signal {binary} isNull[3] <== [a_sv_isNull, b_sv_isNull, c_sv_isNull];
    sv_aggregatedProofs <== AggregateProofsNull(3)([a_sv_aggregatedProofs, b_sv_aggregatedProofs, c_sv_aggregatedProofs], isNull);

    signal AB_stage1Hash[368] <== AggregateValuesNull(368)(a_sv_stage1Hash, b_sv_stage1Hash, a_sv_isNull, b_sv_isNull);
    sv_stage1Hash <== AggregateValuesNull(368)(AB_stage1Hash, c_sv_stage1Hash, AB_isNull, c_sv_isNull);


    for (var i=0; i<4; i++) {
        vA.publics[391 + i] <== rootCAgg[i];
        vB.publics[391 + i] <== rootCAgg[i];
        vC.publics[391 + i] <== rootCAgg[i];
    }
}

component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive2();
