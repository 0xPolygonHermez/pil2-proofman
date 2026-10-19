pragma circom 2.1.0;
pragma custom_templates;

include "cmul.circom";
include "cinv.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "hash/blake3/pow.circom";
include "tree/treeselector4.circom";
include "hash/blake3/merklehash.circom";

/*
    FRI query derivation: 211 indices into L_0 (23 bits each) from a
    fresh transcript over the challenge drawn after the final polynomial and the
    grinding nonce, with the proof of work checked first.
*/
template calculateFRIQueries0() {
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queriesL0[211][23];

    VerifyPoW(24)(challengeFRIQueries, nonce, enable);


    signal b3fin_friQueries_0[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 0);
    signal {binary} transcriptN2b_0[64] <== Num2Bits_strict()(b3fin_friQueries_0[0]);
    signal {binary} transcriptN2b_1[64] <== Num2Bits_strict()(b3fin_friQueries_0[1]);
    signal {binary} transcriptN2b_2[64] <== Num2Bits_strict()(b3fin_friQueries_0[2]);
    signal {binary} transcriptN2b_3[64] <== Num2Bits_strict()(b3fin_friQueries_0[3]);
    signal {binary} transcriptN2b_4[64] <== Num2Bits_strict()(b3fin_friQueries_0[4]);
    signal {binary} transcriptN2b_5[64] <== Num2Bits_strict()(b3fin_friQueries_0[5]);
    signal {binary} transcriptN2b_6[64] <== Num2Bits_strict()(b3fin_friQueries_0[6]);
    signal {binary} transcriptN2b_7[64] <== Num2Bits_strict()(b3fin_friQueries_0[7]);

    signal b3fin_friQueries_1[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 1);
    signal {binary} transcriptN2b_8[64] <== Num2Bits_strict()(b3fin_friQueries_1[0]);
    signal {binary} transcriptN2b_9[64] <== Num2Bits_strict()(b3fin_friQueries_1[1]);
    signal {binary} transcriptN2b_10[64] <== Num2Bits_strict()(b3fin_friQueries_1[2]);
    signal {binary} transcriptN2b_11[64] <== Num2Bits_strict()(b3fin_friQueries_1[3]);
    signal {binary} transcriptN2b_12[64] <== Num2Bits_strict()(b3fin_friQueries_1[4]);
    signal {binary} transcriptN2b_13[64] <== Num2Bits_strict()(b3fin_friQueries_1[5]);
    signal {binary} transcriptN2b_14[64] <== Num2Bits_strict()(b3fin_friQueries_1[6]);
    signal {binary} transcriptN2b_15[64] <== Num2Bits_strict()(b3fin_friQueries_1[7]);

    signal b3fin_friQueries_2[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 2);
    signal {binary} transcriptN2b_16[64] <== Num2Bits_strict()(b3fin_friQueries_2[0]);
    signal {binary} transcriptN2b_17[64] <== Num2Bits_strict()(b3fin_friQueries_2[1]);
    signal {binary} transcriptN2b_18[64] <== Num2Bits_strict()(b3fin_friQueries_2[2]);
    signal {binary} transcriptN2b_19[64] <== Num2Bits_strict()(b3fin_friQueries_2[3]);
    signal {binary} transcriptN2b_20[64] <== Num2Bits_strict()(b3fin_friQueries_2[4]);
    signal {binary} transcriptN2b_21[64] <== Num2Bits_strict()(b3fin_friQueries_2[5]);
    signal {binary} transcriptN2b_22[64] <== Num2Bits_strict()(b3fin_friQueries_2[6]);
    signal {binary} transcriptN2b_23[64] <== Num2Bits_strict()(b3fin_friQueries_2[7]);

    signal b3fin_friQueries_3[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 3);
    signal {binary} transcriptN2b_24[64] <== Num2Bits_strict()(b3fin_friQueries_3[0]);
    signal {binary} transcriptN2b_25[64] <== Num2Bits_strict()(b3fin_friQueries_3[1]);
    signal {binary} transcriptN2b_26[64] <== Num2Bits_strict()(b3fin_friQueries_3[2]);
    signal {binary} transcriptN2b_27[64] <== Num2Bits_strict()(b3fin_friQueries_3[3]);
    signal {binary} transcriptN2b_28[64] <== Num2Bits_strict()(b3fin_friQueries_3[4]);
    signal {binary} transcriptN2b_29[64] <== Num2Bits_strict()(b3fin_friQueries_3[5]);
    signal {binary} transcriptN2b_30[64] <== Num2Bits_strict()(b3fin_friQueries_3[6]);
    signal {binary} transcriptN2b_31[64] <== Num2Bits_strict()(b3fin_friQueries_3[7]);

    signal b3fin_friQueries_4[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 4);
    signal {binary} transcriptN2b_32[64] <== Num2Bits_strict()(b3fin_friQueries_4[0]);
    signal {binary} transcriptN2b_33[64] <== Num2Bits_strict()(b3fin_friQueries_4[1]);
    signal {binary} transcriptN2b_34[64] <== Num2Bits_strict()(b3fin_friQueries_4[2]);
    signal {binary} transcriptN2b_35[64] <== Num2Bits_strict()(b3fin_friQueries_4[3]);
    signal {binary} transcriptN2b_36[64] <== Num2Bits_strict()(b3fin_friQueries_4[4]);
    signal {binary} transcriptN2b_37[64] <== Num2Bits_strict()(b3fin_friQueries_4[5]);
    signal {binary} transcriptN2b_38[64] <== Num2Bits_strict()(b3fin_friQueries_4[6]);
    signal {binary} transcriptN2b_39[64] <== Num2Bits_strict()(b3fin_friQueries_4[7]);

    signal b3fin_friQueries_5[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 5);
    signal {binary} transcriptN2b_40[64] <== Num2Bits_strict()(b3fin_friQueries_5[0]);
    signal {binary} transcriptN2b_41[64] <== Num2Bits_strict()(b3fin_friQueries_5[1]);
    signal {binary} transcriptN2b_42[64] <== Num2Bits_strict()(b3fin_friQueries_5[2]);
    signal {binary} transcriptN2b_43[64] <== Num2Bits_strict()(b3fin_friQueries_5[3]);
    signal {binary} transcriptN2b_44[64] <== Num2Bits_strict()(b3fin_friQueries_5[4]);
    signal {binary} transcriptN2b_45[64] <== Num2Bits_strict()(b3fin_friQueries_5[5]);
    signal {binary} transcriptN2b_46[64] <== Num2Bits_strict()(b3fin_friQueries_5[6]);
    signal {binary} transcriptN2b_47[64] <== Num2Bits_strict()(b3fin_friQueries_5[7]);

    signal b3fin_friQueries_6[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 6);
    signal {binary} transcriptN2b_48[64] <== Num2Bits_strict()(b3fin_friQueries_6[0]);
    signal {binary} transcriptN2b_49[64] <== Num2Bits_strict()(b3fin_friQueries_6[1]);
    signal {binary} transcriptN2b_50[64] <== Num2Bits_strict()(b3fin_friQueries_6[2]);
    signal {binary} transcriptN2b_51[64] <== Num2Bits_strict()(b3fin_friQueries_6[3]);
    signal {binary} transcriptN2b_52[64] <== Num2Bits_strict()(b3fin_friQueries_6[4]);
    signal {binary} transcriptN2b_53[64] <== Num2Bits_strict()(b3fin_friQueries_6[5]);
    signal {binary} transcriptN2b_54[64] <== Num2Bits_strict()(b3fin_friQueries_6[6]);
    signal {binary} transcriptN2b_55[64] <== Num2Bits_strict()(b3fin_friQueries_6[7]);

    signal b3fin_friQueries_7[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 7);
    signal {binary} transcriptN2b_56[64] <== Num2Bits_strict()(b3fin_friQueries_7[0]);
    signal {binary} transcriptN2b_57[64] <== Num2Bits_strict()(b3fin_friQueries_7[1]);
    signal {binary} transcriptN2b_58[64] <== Num2Bits_strict()(b3fin_friQueries_7[2]);
    signal {binary} transcriptN2b_59[64] <== Num2Bits_strict()(b3fin_friQueries_7[3]);
    signal {binary} transcriptN2b_60[64] <== Num2Bits_strict()(b3fin_friQueries_7[4]);
    signal {binary} transcriptN2b_61[64] <== Num2Bits_strict()(b3fin_friQueries_7[5]);
    signal {binary} transcriptN2b_62[64] <== Num2Bits_strict()(b3fin_friQueries_7[6]);
    signal {binary} transcriptN2b_63[64] <== Num2Bits_strict()(b3fin_friQueries_7[7]);

    signal b3fin_friQueries_8[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 8);
    signal {binary} transcriptN2b_64[64] <== Num2Bits_strict()(b3fin_friQueries_8[0]);
    signal {binary} transcriptN2b_65[64] <== Num2Bits_strict()(b3fin_friQueries_8[1]);
    signal {binary} transcriptN2b_66[64] <== Num2Bits_strict()(b3fin_friQueries_8[2]);
    signal {binary} transcriptN2b_67[64] <== Num2Bits_strict()(b3fin_friQueries_8[3]);
    signal {binary} transcriptN2b_68[64] <== Num2Bits_strict()(b3fin_friQueries_8[4]);
    signal {binary} transcriptN2b_69[64] <== Num2Bits_strict()(b3fin_friQueries_8[5]);
    signal {binary} transcriptN2b_70[64] <== Num2Bits_strict()(b3fin_friQueries_8[6]);
    signal {binary} transcriptN2b_71[64] <== Num2Bits_strict()(b3fin_friQueries_8[7]);

    signal b3fin_friQueries_9[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [challengeFRIQueries[0], challengeFRIQueries[1], challengeFRIQueries[2], nonce, 0, 0, 0, 0], 32, 9);
    signal {binary} transcriptN2b_72[64] <== Num2Bits_strict()(b3fin_friQueries_9[0]);
    signal {binary} transcriptN2b_73[64] <== Num2Bits_strict()(b3fin_friQueries_9[1]);
    signal {binary} transcriptN2b_74[64] <== Num2Bits_strict()(b3fin_friQueries_9[2]);
    signal {binary} transcriptN2b_75[64] <== Num2Bits_strict()(b3fin_friQueries_9[3]);
    signal {binary} transcriptN2b_76[64] <== Num2Bits_strict()(b3fin_friQueries_9[4]);
    signal {binary} transcriptN2b_77[64] <== Num2Bits_strict()(b3fin_friQueries_9[5]);
    // From each transcript hash converted to bits, we assign those bits to queriesL0[q] to define the query positions
    var q = 0; // Query number 
    var b = 0; // Bit number 
    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_0[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_0[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_1[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_1[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_2[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_2[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_3[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_3[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_4[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_4[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_5[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_5[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_6[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_6[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_7[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_7[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_8[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_8[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_9[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_9[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_10[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_10[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_11[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_11[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_12[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_12[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_13[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_14[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_14[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_15[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_15[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_16[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_16[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_17[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_17[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_18[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_18[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_19[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_19[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_20[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_20[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_21[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_21[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_22[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_22[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_23[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_23[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_24[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_24[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_25[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_25[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_26[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_26[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_27[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_27[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_28[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_28[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_29[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_29[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_30[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_30[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_31[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_31[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_32[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_32[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_33[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_33[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_34[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_34[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_35[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_35[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_36[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_36[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_37[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_37[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_38[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_38[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_39[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_39[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_40[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_40[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_41[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_41[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_42[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_42[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_43[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_43[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_44[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_44[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_45[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_45[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_46[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_46[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_47[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_47[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_48[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_48[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_49[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_49[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_50[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_50[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_51[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_51[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_52[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_52[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_53[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_53[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_54[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_54[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_55[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_55[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_56[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_56[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_57[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_57[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_58[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_58[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_59[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_59[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_60[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_60[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_61[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_61[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_62[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_62[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_63[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_63[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_64[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_64[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_65[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_65[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_66[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_66[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_67[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_67[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_68[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_68[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_69[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_69[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_70[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_70[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_71[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_71[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_72[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_72[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_73[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_73[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_74[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_74[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_75[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_75[63]; // Unused last bit

    for (var j = 0; j < 63; j++) {
        queriesL0[q][b] <== transcriptN2b_76[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_76[63]; // Unused last bit

    for (var j = 0; j < 2; j++) {
        queriesL0[q][b] <== transcriptN2b_77[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    for (var j = 2; j < 64; j++) {
        _ <== transcriptN2b_77[j]; // Unused bits        
    }
}

/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input globalChallenge[3]; 


    signal input airValues[3][3];
    signal input airgroupvalues[2][3];

    signal input root2[4];
    signal input root3[4];
    signal input evals[23][3]; 
    signal input s1_root[4]; // Root of a polynomial of degree lower than 2^22, evaluated over 2^23 points
    signal input s2_root[4]; // Root of a polynomial of degree lower than 2^17, evaluated over 2^18 points
    signal input s3_root[4]; // Root of a polynomial of degree lower than 2^13, evaluated over 2^14 points
    signal input s4_root[4]; // Root of a polynomial of degree lower than 2^9, evaluated over 2^10 points
    signal input finalPol[64][3]; // Evaluations over 2^6 points of a polynomial of degree lower than 2^5
    signal input nonce;
    signal input {binary} enable;

    signal output challengesStage2[2][3];
    signal output challengeQ[3];
    signal output challengeOut[3];
    signal output challengesDeep[2][3];

    // r^fold_s, s = 0..M−1: drawn after the root of f_s, folds it into f_{s+1}.
    signal output rFold[4][3];
    signal challengeQueries[3];
    signal output {binary} queriesL0[211][23];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];



    signal b3fin_0[8] <== Blake3FinalizeChunk(11)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [globalChallenge[0], globalChallenge[1], globalChallenge[2], 0, 0, 0, 0, 0], 24, 0);
    challengesStage2[0] <== [b3fin_0[0], b3fin_0[1], b3fin_0[2]];
    challengesStage2[1] <== [b3fin_0[3], b3fin_0[4], b3fin_0[5]];

    signal b3blk_1[8] <== Blake3AbsorbBlock(1)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [globalChallenge[0], globalChallenge[1], globalChallenge[2], 2, root2[0], root2[1], root2[2], root2[3]], 64, 0);

    signal b3fin_2[8] <== Blake3FinalizeChunk(10)(b3blk_1, [airValues[2][0], airValues[2][1], airValues[2][2], airgroupvalues[0][0], airgroupvalues[1][0], airgroupvalues[1][1], airgroupvalues[1][2], 0], 56, 0);
    challengeQ <== [b3fin_2[0], b3fin_2[1], b3fin_2[2]];

    signal b3blk_3[8] <== Blake3AbsorbBlock(0)(b3blk_1, [airValues[2][0], airValues[2][1], airValues[2][2], airgroupvalues[0][0], airgroupvalues[1][0], airgroupvalues[1][1], airgroupvalues[1][2], root3[0]], 64, 0);

    signal b3fin_4[8] <== Blake3FinalizeChunk(10)(b3blk_3, [root3[1], root3[2], root3[3], 0, 0, 0, 0, 0], 24, 0);
    challengeOut <== [b3fin_4[0], b3fin_4[1], b3fin_4[2]];

    signal b3blk_evals_0[8] <== Blake3AbsorbBlock(1)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [evals[0][0], evals[0][1], evals[0][2], evals[1][0], evals[1][1], evals[1][2], evals[2][0], evals[2][1]], 64, 0);

    signal b3blk_evals_1[8] <== Blake3AbsorbBlock(0)(b3blk_evals_0, [evals[2][2], evals[3][0], evals[3][1], evals[3][2], evals[4][0], evals[4][1], evals[4][2], evals[5][0]], 64, 0);

    signal b3blk_evals_2[8] <== Blake3AbsorbBlock(0)(b3blk_evals_1, [evals[5][1], evals[5][2], evals[6][0], evals[6][1], evals[6][2], evals[7][0], evals[7][1], evals[7][2]], 64, 0);

    signal b3blk_evals_3[8] <== Blake3AbsorbBlock(0)(b3blk_evals_2, [evals[8][0], evals[8][1], evals[8][2], evals[9][0], evals[9][1], evals[9][2], evals[10][0], evals[10][1]], 64, 0);

    signal b3blk_evals_4[8] <== Blake3AbsorbBlock(0)(b3blk_evals_3, [evals[10][2], evals[11][0], evals[11][1], evals[11][2], evals[12][0], evals[12][1], evals[12][2], evals[13][0]], 64, 0);

    signal b3blk_evals_5[8] <== Blake3AbsorbBlock(0)(b3blk_evals_4, [evals[13][1], evals[13][2], evals[14][0], evals[14][1], evals[14][2], evals[15][0], evals[15][1], evals[15][2]], 64, 0);

    signal b3blk_evals_6[8] <== Blake3AbsorbBlock(0)(b3blk_evals_5, [evals[16][0], evals[16][1], evals[16][2], evals[17][0], evals[17][1], evals[17][2], evals[18][0], evals[18][1]], 64, 0);

    signal b3blk_evals_7[8] <== Blake3AbsorbBlock(0)(b3blk_evals_6, [evals[18][2], evals[19][0], evals[19][1], evals[19][2], evals[20][0], evals[20][1], evals[20][2], evals[21][0]], 64, 0);

    signal b3fin_evals_8[8] <== Blake3FinalizeChunk(10)(b3blk_evals_7, [evals[21][1], evals[21][2], evals[22][0], evals[22][1], evals[22][2], 0, 0, 0], 40, 0);
    evalsHash <== [b3fin_evals_8[0], b3fin_evals_8[1], b3fin_evals_8[2], b3fin_evals_8[3]];

    signal b3fin_5[8] <== Blake3FinalizeChunk(10)(b3blk_3, [root3[1], root3[2], root3[3], evalsHash[0], evalsHash[1], evalsHash[2], evalsHash[3], 0], 56, 0);
    challengesDeep[0] <== [b3fin_5[0], b3fin_5[1], b3fin_5[2]];
    challengesDeep[1] <== [b3fin_5[3], b3fin_5[4], b3fin_5[5]];

    signal b3blk_6[8] <== Blake3AbsorbBlock(0)(b3blk_3, [root3[1], root3[2], root3[3], evalsHash[0], evalsHash[1], evalsHash[2], evalsHash[3], s1_root[0]], 64, 0);

    signal b3fin_7[8] <== Blake3FinalizeChunk(10)(b3blk_6, [s1_root[1], s1_root[2], s1_root[3], 0, 0, 0, 0, 0], 24, 0);
    rFold[0] <== [b3fin_7[0], b3fin_7[1], b3fin_7[2]];

    signal b3fin_8[8] <== Blake3FinalizeChunk(10)(b3blk_6, [s1_root[1], s1_root[2], s1_root[3], s2_root[0], s2_root[1], s2_root[2], s2_root[3], 0], 56, 0);
    rFold[1] <== [b3fin_8[0], b3fin_8[1], b3fin_8[2]];

    signal b3blk_9[8] <== Blake3AbsorbBlock(0)(b3blk_6, [s1_root[1], s1_root[2], s1_root[3], s2_root[0], s2_root[1], s2_root[2], s2_root[3], s3_root[0]], 64, 0);

    signal b3fin_10[8] <== Blake3FinalizeChunk(10)(b3blk_9, [s3_root[1], s3_root[2], s3_root[3], 0, 0, 0, 0, 0], 24, 0);
    rFold[2] <== [b3fin_10[0], b3fin_10[1], b3fin_10[2]];

    signal b3fin_11[8] <== Blake3FinalizeChunk(10)(b3blk_9, [s3_root[1], s3_root[2], s3_root[3], s4_root[0], s4_root[1], s4_root[2], s4_root[3], 0], 56, 0);
    rFold[3] <== [b3fin_11[0], b3fin_11[1], b3fin_11[2]];

    signal b3blk_lastPolFRI_0[8] <== Blake3AbsorbBlock(1)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [finalPol[0][0], finalPol[0][1], finalPol[0][2], finalPol[1][0], finalPol[1][1], finalPol[1][2], finalPol[2][0], finalPol[2][1]], 64, 0);

    signal b3blk_lastPolFRI_1[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_0, [finalPol[2][2], finalPol[3][0], finalPol[3][1], finalPol[3][2], finalPol[4][0], finalPol[4][1], finalPol[4][2], finalPol[5][0]], 64, 0);

    signal b3blk_lastPolFRI_2[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_1, [finalPol[5][1], finalPol[5][2], finalPol[6][0], finalPol[6][1], finalPol[6][2], finalPol[7][0], finalPol[7][1], finalPol[7][2]], 64, 0);

    signal b3blk_lastPolFRI_3[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_2, [finalPol[8][0], finalPol[8][1], finalPol[8][2], finalPol[9][0], finalPol[9][1], finalPol[9][2], finalPol[10][0], finalPol[10][1]], 64, 0);

    signal b3blk_lastPolFRI_4[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_3, [finalPol[10][2], finalPol[11][0], finalPol[11][1], finalPol[11][2], finalPol[12][0], finalPol[12][1], finalPol[12][2], finalPol[13][0]], 64, 0);

    signal b3blk_lastPolFRI_5[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_4, [finalPol[13][1], finalPol[13][2], finalPol[14][0], finalPol[14][1], finalPol[14][2], finalPol[15][0], finalPol[15][1], finalPol[15][2]], 64, 0);

    signal b3blk_lastPolFRI_6[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_5, [finalPol[16][0], finalPol[16][1], finalPol[16][2], finalPol[17][0], finalPol[17][1], finalPol[17][2], finalPol[18][0], finalPol[18][1]], 64, 0);

    signal b3blk_lastPolFRI_7[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_6, [finalPol[18][2], finalPol[19][0], finalPol[19][1], finalPol[19][2], finalPol[20][0], finalPol[20][1], finalPol[20][2], finalPol[21][0]], 64, 0);

    signal b3blk_lastPolFRI_8[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_7, [finalPol[21][1], finalPol[21][2], finalPol[22][0], finalPol[22][1], finalPol[22][2], finalPol[23][0], finalPol[23][1], finalPol[23][2]], 64, 0);

    signal b3blk_lastPolFRI_9[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_8, [finalPol[24][0], finalPol[24][1], finalPol[24][2], finalPol[25][0], finalPol[25][1], finalPol[25][2], finalPol[26][0], finalPol[26][1]], 64, 0);

    signal b3blk_lastPolFRI_10[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_9, [finalPol[26][2], finalPol[27][0], finalPol[27][1], finalPol[27][2], finalPol[28][0], finalPol[28][1], finalPol[28][2], finalPol[29][0]], 64, 0);

    signal b3blk_lastPolFRI_11[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_10, [finalPol[29][1], finalPol[29][2], finalPol[30][0], finalPol[30][1], finalPol[30][2], finalPol[31][0], finalPol[31][1], finalPol[31][2]], 64, 0);

    signal b3blk_lastPolFRI_12[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_11, [finalPol[32][0], finalPol[32][1], finalPol[32][2], finalPol[33][0], finalPol[33][1], finalPol[33][2], finalPol[34][0], finalPol[34][1]], 64, 0);

    signal b3blk_lastPolFRI_13[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_12, [finalPol[34][2], finalPol[35][0], finalPol[35][1], finalPol[35][2], finalPol[36][0], finalPol[36][1], finalPol[36][2], finalPol[37][0]], 64, 0);

    signal b3blk_lastPolFRI_14[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_13, [finalPol[37][1], finalPol[37][2], finalPol[38][0], finalPol[38][1], finalPol[38][2], finalPol[39][0], finalPol[39][1], finalPol[39][2]], 64, 0);

    signal b3blk_lastPolFRI_15[8] <== Blake3AbsorbBlock(2)(b3blk_lastPolFRI_14, [finalPol[40][0], finalPol[40][1], finalPol[40][2], finalPol[41][0], finalPol[41][1], finalPol[41][2], finalPol[42][0], finalPol[42][1]], 64, 0);

    signal b3blk_lastPolFRI_16[8] <== Blake3AbsorbBlock(1)([1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225], [finalPol[42][2], finalPol[43][0], finalPol[43][1], finalPol[43][2], finalPol[44][0], finalPol[44][1], finalPol[44][2], finalPol[45][0]], 64, 1);

    signal b3blk_lastPolFRI_17[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_16, [finalPol[45][1], finalPol[45][2], finalPol[46][0], finalPol[46][1], finalPol[46][2], finalPol[47][0], finalPol[47][1], finalPol[47][2]], 64, 1);

    signal b3blk_lastPolFRI_18[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_17, [finalPol[48][0], finalPol[48][1], finalPol[48][2], finalPol[49][0], finalPol[49][1], finalPol[49][2], finalPol[50][0], finalPol[50][1]], 64, 1);

    signal b3blk_lastPolFRI_19[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_18, [finalPol[50][2], finalPol[51][0], finalPol[51][1], finalPol[51][2], finalPol[52][0], finalPol[52][1], finalPol[52][2], finalPol[53][0]], 64, 1);

    signal b3blk_lastPolFRI_20[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_19, [finalPol[53][1], finalPol[53][2], finalPol[54][0], finalPol[54][1], finalPol[54][2], finalPol[55][0], finalPol[55][1], finalPol[55][2]], 64, 1);

    signal b3blk_lastPolFRI_21[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_20, [finalPol[56][0], finalPol[56][1], finalPol[56][2], finalPol[57][0], finalPol[57][1], finalPol[57][2], finalPol[58][0], finalPol[58][1]], 64, 1);

    signal b3blk_lastPolFRI_22[8] <== Blake3AbsorbBlock(0)(b3blk_lastPolFRI_21, [finalPol[58][2], finalPol[59][0], finalPol[59][1], finalPol[59][2], finalPol[60][0], finalPol[60][1], finalPol[60][2], finalPol[61][0]], 64, 1);

    signal b3cls_lastPolFRI_23[8] <== Blake3AbsorbBlock(2)(b3blk_lastPolFRI_22, [finalPol[61][1], finalPol[61][2], finalPol[62][0], finalPol[62][1], finalPol[62][2], finalPol[63][0], finalPol[63][1], finalPol[63][2]], 64, 1);

    signal b3fin_lastPolFRI_24[8] <== Blake3FinalizeParent()(b3blk_lastPolFRI_15, b3cls_lastPolFRI_23, 0);
    lastPolFRIHash <== [b3fin_lastPolFRI_24[0], b3fin_lastPolFRI_24[1], b3fin_lastPolFRI_24[2], b3fin_lastPolFRI_24[3]];

    signal b3blk_12[8] <== Blake3AbsorbBlock(0)(b3blk_9, [s3_root[1], s3_root[2], s3_root[3], s4_root[0], s4_root[1], s4_root[2], s4_root[3], lastPolFRIHash[0]], 64, 0);

    signal b3fin_13[8] <== Blake3FinalizeChunk(10)(b3blk_12, [lastPolFRIHash[1], lastPolFRIHash[2], lastPolFRIHash[3], 0, 0, 0, 0, 0], 24, 0);
    challengeQueries <== [b3fin_13[0], b3fin_13[1], b3fin_13[2]];
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
    signal input evals[23][3];
    signal input publics[8];
    signal input airgroupvalues[2][3];
    signal input airvalues[3][3];
    signal input proofvalues[2][3];

    signal input Zh[3];


    signal output tmp_199[3];
    signal tmp_101[3] <== [evals[5][0] - publics[1], evals[5][1], evals[5][2]];
    signal tmp_102[3] <== CMul()(evals[3], tmp_101);
    signal tmp_103[3] <== CMul()(challengeQ, tmp_102);
    signal tmp_104[3] <== [evals[6][0] - publics[2], evals[6][1], evals[6][2]];
    signal tmp_105[3] <== CMul()(evals[3], tmp_104);
    signal tmp_106[3] <== [tmp_103[0] + tmp_105[0], tmp_103[1] + tmp_105[1], tmp_103[2] + tmp_105[2]];
    signal tmp_107[3] <== CMul()(challengeQ, tmp_106);
    signal tmp_108[3] <== [evals[6][0] - publics[3], evals[6][1], evals[6][2]];
    signal tmp_109[3] <== CMul()(evals[14], tmp_108);
    signal tmp_110[3] <== [tmp_107[0] + tmp_109[0], tmp_107[1] + tmp_109[1], tmp_107[2] + tmp_109[2]];
    signal tmp_111[3] <== CMul()(challengeQ, tmp_110);
    signal tmp_112[3] <== [evals[16][0] - evals[6][0], evals[16][1] - evals[6][1], evals[16][2] - evals[6][2]];
    signal tmp_113[3] <== [1 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_114[3] <== CMul()(tmp_112, tmp_113);
    signal tmp_115[3] <== [tmp_111[0] + tmp_114[0], tmp_111[1] + tmp_114[1], tmp_111[2] + tmp_114[2]];
    signal tmp_116[3] <== CMul()(challengeQ, tmp_115);
    signal tmp_117[3] <== [evals[19][0] - evals[17][0], evals[19][1] - evals[17][1], evals[19][2] - evals[17][2]];
    signal tmp_118[3] <== [1 - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_119[3] <== CMul()(tmp_117, tmp_118);
    signal tmp_120[3] <== [tmp_116[0] + tmp_119[0], tmp_116[1] + tmp_119[1], tmp_116[2] + tmp_119[2]];
    signal tmp_121[3] <== CMul()(challengeQ, tmp_120);
    signal tmp_122[3] <== [evals[22][0] - evals[20][0], evals[22][1] - evals[20][1], evals[22][2] - evals[20][2]];
    signal tmp_123[3] <== [1 - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_124[3] <== CMul()(tmp_122, tmp_123);
    signal tmp_125[3] <== [tmp_121[0] + tmp_124[0], tmp_121[1] + tmp_124[1], tmp_121[2] + tmp_124[2]];
    signal tmp_126[3] <== CMul()(challengeQ, tmp_125);
    signal tmp_127 <== publics[0] * proofvalues[0][0];
    signal tmp_128 <== tmp_127 - proofvalues[1][0];
    signal tmp_129[3] <== [tmp_126[0] + tmp_128, tmp_126[1], tmp_126[2]];
    signal tmp_130[3] <== CMul()(challengeQ, tmp_129);
    signal tmp_131 <== 2 * airvalues[0][0];
    signal tmp_132 <== tmp_131 - airvalues[1][0];
    signal tmp_133[3] <== [tmp_130[0] + tmp_132, tmp_130[1], tmp_130[2]];
    signal tmp_134[3] <== CMul()(challengeQ, tmp_133);
    signal tmp_135 <== airgroupvalues[0][0] - airvalues[0][0];
    signal tmp_136[3] <== [evals[3][0] * tmp_135, evals[3][1] * tmp_135, evals[3][2] * tmp_135];
    signal tmp_137[3] <== [tmp_134[0] + tmp_136[0], tmp_134[1] + tmp_136[1], tmp_134[2] + tmp_136[2]];
    signal tmp_138[3] <== CMul()(challengeQ, tmp_137);
    signal tmp_139[3] <== [evals[2][0] + 1, evals[2][1], evals[2][2]];
    signal tmp_140[3] <== [evals[1][0] - tmp_139[0], evals[1][1] - tmp_139[1], evals[1][2] - tmp_139[2]];
    signal tmp_141[3] <== [tmp_138[0] + tmp_140[0], tmp_138[1] + tmp_140[1], tmp_138[2] + tmp_140[2]];
    signal tmp_142[3] <== CMul()(challengeQ, tmp_141);
    signal tmp_143[3] <== [2 * evals[5][0], 2 * evals[5][1], 2 * evals[5][2]];
    signal tmp_144[3] <== [tmp_143[0] + evals[6][0], tmp_143[1] + evals[6][1], tmp_143[2] + evals[6][2]];
    signal tmp_145[3] <== [tmp_144[0] + airvalues[0][0], tmp_144[1], tmp_144[2]];
    signal tmp_146[3] <== [54 * evals[1][0], 54 * evals[1][1], 54 * evals[1][2]];
    signal tmp_147[3] <== [tmp_145[0] + tmp_146[0], tmp_145[1] + tmp_146[1], tmp_145[2] + tmp_146[2]];
    signal tmp_148[3] <== [evals[3][0] * publics[1], evals[3][1] * publics[1], evals[3][2] * publics[1]];
    signal tmp_149[3] <== [tmp_147[0] + tmp_148[0], tmp_147[1] + tmp_148[1], tmp_147[2] + tmp_148[2]];
    signal tmp_150[3] <== [evals[7][0] - tmp_149[0], evals[7][1] - tmp_149[1], evals[7][2] - tmp_149[2]];
    signal tmp_151[3] <== [tmp_142[0] + tmp_150[0], tmp_142[1] + tmp_150[1], tmp_142[2] + tmp_150[2]];
    signal tmp_152[3] <== CMul()(challengeQ, tmp_151);
    signal tmp_155[3] <== [evals[8][0] - tmp_144[0], evals[8][1] - tmp_144[1], evals[8][2] - tmp_144[2]];
    signal tmp_156[3] <== [tmp_152[0] + tmp_155[0], tmp_152[1] + tmp_155[1], tmp_152[2] + tmp_155[2]];
    signal tmp_157[3] <== CMul()(challengeQ, tmp_156);
    signal tmp_158[3] <== [evals[8][0] + airvalues[0][0], evals[8][1], evals[8][2]];
    signal tmp_160[3] <== [tmp_158[0] + tmp_146[0], tmp_158[1] + tmp_146[1], tmp_158[2] + tmp_146[2]];
    signal tmp_162[3] <== [tmp_160[0] + tmp_148[0], tmp_160[1] + tmp_148[1], tmp_160[2] + tmp_148[2]];
    signal tmp_163[3] <== [evals[9][0] - tmp_162[0], evals[9][1] - tmp_162[1], evals[9][2] - tmp_162[2]];
    signal tmp_164[3] <== [tmp_157[0] + tmp_163[0], tmp_157[1] + tmp_163[1], tmp_157[2] + tmp_163[2]];
    signal tmp_165[3] <== CMul()(challengeQ, tmp_164);
    signal tmp_166[3] <== [evals[7][0] - evals[9][0], evals[7][1] - evals[9][1], evals[7][2] - evals[9][2]];
    signal tmp_167[3] <== [tmp_165[0] + tmp_166[0], tmp_165[1] + tmp_166[1], tmp_165[2] + tmp_166[2]];
    signal tmp_168[3] <== CMul()(challengeQ, tmp_167);
    signal tmp_171[3] <== [tmp_113[0] - 1, tmp_113[1], tmp_113[2]];
    signal tmp_172[3] <== CMul()(tmp_113, tmp_171);
    signal tmp_173[3] <== [tmp_168[0] + tmp_172[0], tmp_168[1] + tmp_172[1], tmp_168[2] + tmp_172[2]];
    signal tmp_174[3] <== CMul()(challengeQ, tmp_173);
    signal tmp_175[3] <== CMul()(evals[11], evals[12]);
    signal tmp_177[3] <== [0 - tmp_113[0], -tmp_113[1], -tmp_113[2]];
    signal tmp_178[3] <== [tmp_175[0] - tmp_177[0], tmp_175[1] - tmp_177[1], tmp_175[2] - tmp_177[2]];
    signal tmp_179[3] <== [tmp_174[0] + tmp_178[0], tmp_174[1] + tmp_178[1], tmp_174[2] + tmp_178[2]];
    signal tmp_180[3] <== CMul()(challengeQ, tmp_179);
    signal tmp_181[3] <== [1 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_182[3] <== CMul()(evals[0], tmp_181);
    signal tmp_183[3] <== [evals[10][0] - tmp_182[0], evals[10][1] - tmp_182[1], evals[10][2] - tmp_182[2]];
    signal tmp_184[3] <== [tmp_183[0] - evals[11][0], tmp_183[1] - evals[11][1], tmp_183[2] - evals[11][2]];
    signal tmp_185[3] <== [tmp_180[0] + tmp_184[0], tmp_180[1] + tmp_184[1], tmp_180[2] + tmp_184[2]];
    signal tmp_186[3] <== CMul()(challengeQ, tmp_185);
    signal tmp_187[3] <== [airgroupvalues[1][0] - evals[10][0], airgroupvalues[1][1] - evals[10][1], airgroupvalues[1][2] - evals[10][2]];
    signal tmp_188[3] <== CMul()(evals[15], tmp_187);
    signal tmp_189[3] <== [tmp_186[0] + tmp_188[0], tmp_186[1] + tmp_188[1], tmp_186[2] + tmp_188[2]];
    signal tmp_190[3] <== CMul()(challengeQ, tmp_189);
    signal tmp_191[3] <== CMul()(evals[17], challengesStage2[0]);
    signal tmp_192[3] <== CMul()(evals[5], evals[5]);
    signal tmp_193[3] <== CMul()(evals[6], evals[6]);
    signal tmp_194[3] <== [tmp_192[0] + tmp_193[0], tmp_192[1] + tmp_193[1], tmp_192[2] + tmp_193[2]];
    signal tmp_195[3] <== [tmp_191[0] + tmp_194[0], tmp_191[1] + tmp_194[1], tmp_191[2] + tmp_194[2]];
    signal tmp_196[3] <== CMul()(tmp_195, challengesStage2[0]);
    signal tmp_197[3] <== [tmp_196[0] + 1, tmp_196[1], tmp_196[2]];
    signal tmp_97[3] <== [tmp_197[0] + challengesStage2[1][0], tmp_197[1] + challengesStage2[1][1], tmp_197[2] + challengesStage2[1][2]];
    signal tmp_198[3] <== [evals[12][0] - tmp_97[0], evals[12][1] - tmp_97[1], evals[12][2] - tmp_97[2]];
    signal tmp_99[3] <== [tmp_190[0] + tmp_198[0], tmp_190[1] + tmp_198[1], tmp_190[2] + tmp_198[2]];
    tmp_199 <== CMul()(tmp_99, Zh);
}


/*
    The claimed evaluations satisfy the air: the constraint combination at z, computed by
    VerifyEvaluationsChunks*, must equal Q(z).

    Q has degree < 2^22 here, so it is committed unsplit and there is no need to reassemble
    it from multiple polynomials.
*/
template parallel VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeOut[3];
    signal input evals[23][3];
    signal input publics[8];
    signal input airgroupvalues[2][3];
    signal input airvalues[3][3];
    signal input proofvalues[2][3];
    signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeOut) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[22][3];
    for (var i = 0; i < 22; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeOut, challengeOut);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[21][0] - 1, zMul[21][1], zMul[21][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_199[3];
    (tmp_199) <== VerifyEvaluationsChunks0_0()(challengesStage2,challengeQ,challengeOut,evals,publics,airgroupvalues,airvalues,proofvalues,Zh);

    signal xAcc[1][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[0][3]; // Stores the evaluations of Q_i
    signal qAcc[1][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i = 0; i < 1; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[13+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[21]);
            qStep[i-1] <== CMul()(xAcc[i], evals[13+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i
    enable * (tmp_199[0] - qAcc[0][0]) === 0;
    enable * (tmp_199[1] - qAcc[0][1]) === 0;
    enable * (tmp_199[2] - qAcc[0][2]) === 0;
}

/*
    f_0(x) at one query point, out of that query's opened leaves: the arithmetic of the DEEP
    combination whose formula and cross-check live in VerifyQuery0 below.

    The leaves arrive as the flat sections the Merkle trees store (cm1..cm3, consts,
    custom_*), and `mapValues` splits them into one signal per committed polynomial: dim 1 for a
    base-field polynomial, dim 3 for an extension one — which is why a dim-1 value is
    subtracted from limb 0 of its `evals` entry alone. Fixed polynomials are always dim 1 and
    are read straight out of `consts`.

    The emitted code is a nested Horner. Within one opening point g^s·z the differences
    (pol(x) - pol(g^s·z)) are folded with challengesDeep[1] = vf2, the result is multiplied by
    xDivXSubXi[s] = 1/(x - g^s·z), and the opening points are folded together with
    challengesDeep[0] = vf1; tmp_125 is f_0(x).

    The DAG is split into 1 chunk(s) by the same compile-cost device as
    VerifyEvaluationsChunks*, with tmp_ signals crossing the boundaries; each chunk builds its
    own `mapValues`, since the leaf values feed the whole DAG.
*/
template CalculateDeepPolChunks0_0() {
    signal input challengesDeep[2][3];
    signal input evals[23][3];

    signal input cm1[5];
    signal input cm2[9];
    signal input cm3[3];
    signal input consts[2];
    signal input custom_rom_0[2];

    signal input xDivXSubXi[5][3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
    mapValues.vals1 <== cm1;
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;
    mapValues.vals_rom_0 <== custom_rom_0;

    signal input tmp_103[3];
    signal input tmp_104[3];
    signal input tmp_105[3];
    signal input tmp_108[3];
    signal input tmp_109[3];
    signal input tmp_110[3];
    signal input tmp_111[3];
    signal input tmp_112[3];
    signal input tmp_113[3];
    signal input tmp_114[3];
    signal input tmp_115[3];
    signal input tmp_116[3];
    signal input tmp_117[3];
    signal input tmp_118[3];
    signal input tmp_121[3];
    signal input tmp_122[3];
    signal input tmp_123[3];
    signal input tmp_124[3];

    signal output tmp_125[3];
    signal tmp_106[3] <== CMul()(tmp_105, xDivXSubXi[0]);
    signal tmp_107[3] <== CMul()(tmp_104, xDivXSubXi[1]);
    signal tmp_119[3] <== CMul()(tmp_103, xDivXSubXi[2]);
    signal tmp_120[3] <== CMul()(challengesDeep[0], xDivXSubXi[3]);
    signal tmp_57[3] <== [tmp_106[0] + tmp_107[0], tmp_106[1] + tmp_107[1], tmp_106[2] + tmp_107[2]];
    signal tmp_58[3] <== CMul()(mapValues.cm2_0, tmp_57);
    signal tmp_59[3] <== [challengesDeep[1][0] * mapValues.custom_rom_0_0, challengesDeep[1][1] * mapValues.custom_rom_0_0, challengesDeep[1][2] * mapValues.custom_rom_0_0];
    signal tmp_60[3] <== [tmp_108[0] * mapValues.custom_rom_0_1, tmp_108[1] * mapValues.custom_rom_0_1, tmp_108[2] * mapValues.custom_rom_0_1];
    signal tmp_61[3] <== [tmp_59[0] + tmp_60[0], tmp_59[1] + tmp_60[1], tmp_59[2] + tmp_60[2]];
    signal tmp_62[3] <== [tmp_113[0] * mapValues.cm1_2, tmp_113[1] * mapValues.cm1_2, tmp_113[2] * mapValues.cm1_2];
    signal tmp_63[3] <== [tmp_61[0] + tmp_62[0], tmp_61[1] + tmp_62[1], tmp_61[2] + tmp_62[2]];
    signal tmp_64[3] <== [tmp_114[0] * mapValues.cm1_3, tmp_114[1] * mapValues.cm1_3, tmp_114[2] * mapValues.cm1_3];
    signal tmp_65[3] <== [tmp_63[0] + tmp_64[0], tmp_63[1] + tmp_64[1], tmp_63[2] + tmp_64[2]];
    signal tmp_66[3] <== [tmp_115[0] * mapValues.cm1_4, tmp_115[1] * mapValues.cm1_4, tmp_115[2] * mapValues.cm1_4];
    signal tmp_67[3] <== [tmp_65[0] + tmp_66[0], tmp_65[1] + tmp_66[1], tmp_65[2] + tmp_66[2]];
    signal tmp_68[3] <== CMul()(tmp_116, mapValues.cm2_1);
    signal tmp_69[3] <== [tmp_67[0] + tmp_68[0], tmp_67[1] + tmp_68[1], tmp_67[2] + tmp_68[2]];
    signal tmp_70[3] <== CMul()(tmp_117, mapValues.cm2_2);
    signal tmp_71[3] <== [tmp_69[0] + tmp_70[0], tmp_69[1] + tmp_70[1], tmp_69[2] + tmp_70[2]];
    signal tmp_72[3] <== CMul()(tmp_118, mapValues.cm3_0);
    signal tmp_73[3] <== [tmp_71[0] + tmp_72[0], tmp_71[1] + tmp_72[1], tmp_71[2] + tmp_72[2]];
    signal tmp_74[3] <== CMul()(tmp_73, tmp_107);
    signal tmp_75[3] <== [tmp_58[0] + tmp_74[0], tmp_58[1] + tmp_74[1], tmp_58[2] + tmp_74[2]];
    signal tmp_76[3] <== [tmp_110[0] * consts[1], tmp_110[1] * consts[1], tmp_110[2] * consts[1]];
    signal tmp_77[3] <== [tmp_107[0] + tmp_119[0], tmp_107[1] + tmp_119[1], tmp_107[2] + tmp_119[2]];
    signal tmp_78[3] <== CMul()(tmp_76, tmp_77);
    signal tmp_79[3] <== [tmp_75[0] + tmp_78[0], tmp_75[1] + tmp_78[1], tmp_75[2] + tmp_78[2]];
    signal tmp_80[3] <== [tmp_112[0] * mapValues.cm1_1, tmp_112[1] * mapValues.cm1_1, tmp_112[2] * mapValues.cm1_1];
    signal tmp_82[3] <== [tmp_77[0] + tmp_120[0], tmp_77[1] + tmp_120[1], tmp_77[2] + tmp_120[2]];
    signal tmp_83[3] <== CMul()(tmp_80, tmp_82);
    signal tmp_84[3] <== [tmp_79[0] + tmp_83[0], tmp_79[1] + tmp_83[1], tmp_79[2] + tmp_83[2]];
    signal tmp_85[3] <== [tmp_109[0] * consts[0], tmp_109[1] * consts[0], tmp_109[2] * consts[0]];
    signal tmp_86[3] <== [tmp_111[0] * mapValues.cm1_0, tmp_111[1] * mapValues.cm1_0, tmp_111[2] * mapValues.cm1_0];
    signal tmp_87[3] <== [tmp_85[0] + tmp_86[0], tmp_85[1] + tmp_86[1], tmp_85[2] + tmp_86[2]];
    signal tmp_90[3] <== [tmp_82[0] + xDivXSubXi[4][0], tmp_82[1] + xDivXSubXi[4][1], tmp_82[2] + xDivXSubXi[4][2]];
    signal tmp_91[3] <== CMul()(tmp_87, tmp_90);
    signal tmp_92[3] <== [tmp_84[0] + tmp_91[0], tmp_84[1] + tmp_91[1], tmp_84[2] + tmp_91[2]];
    signal tmp_93[3] <== CMul()(tmp_106, evals[0]);
    signal tmp_94[3] <== [tmp_92[0] - tmp_93[0], tmp_92[1] - tmp_93[1], tmp_92[2] - tmp_93[2]];
    signal tmp_95[3] <== CMul()(tmp_107, tmp_121);
    signal tmp_96[3] <== [tmp_94[0] - tmp_95[0], tmp_94[1] - tmp_95[1], tmp_94[2] - tmp_95[2]];
    signal tmp_97[3] <== CMul()(tmp_119, tmp_122);
    signal tmp_98[3] <== [tmp_96[0] - tmp_97[0], tmp_96[1] - tmp_97[1], tmp_96[2] - tmp_97[2]];
    signal tmp_99[3] <== CMul()(tmp_120, tmp_123);
    signal tmp_100[3] <== [tmp_98[0] - tmp_99[0], tmp_98[1] - tmp_99[1], tmp_98[2] - tmp_99[2]];
    signal tmp_101[3] <== CMul()(xDivXSubXi[4], tmp_124);
    tmp_125 <== [tmp_100[0] - tmp_101[0], tmp_100[1] - tmp_101[1], tmp_100[2] - tmp_101[2]];

}


template CalculateFRIConstantsChunks0() {
    signal input challengesDeep[2][3];
    signal input evals[23][3];
    signal output tmp_103[3];
    signal output tmp_104[3];
    signal output tmp_105[3];
    signal output tmp_108[3];
    signal output tmp_109[3];
    signal output tmp_110[3];
    signal output tmp_111[3];
    signal output tmp_112[3];
    signal output tmp_113[3];
    signal output tmp_114[3];
    signal output tmp_115[3];
    signal output tmp_116[3];
    signal output tmp_117[3];
    signal output tmp_118[3];
    signal output tmp_121[3];
    signal output tmp_122[3];
    signal output tmp_123[3];
    signal output tmp_124[3];
    tmp_103 <== CMul()(challengesDeep[0], challengesDeep[0]);
    tmp_104 <== CMul()(tmp_103, challengesDeep[0]);
    tmp_105 <== CMul()(tmp_104, challengesDeep[0]);
    tmp_108 <== CMul()(challengesDeep[1], challengesDeep[1]);
    tmp_109 <== CMul()(tmp_108, challengesDeep[1]);
    tmp_110 <== CMul()(tmp_109, challengesDeep[1]);
    tmp_111 <== CMul()(tmp_110, challengesDeep[1]);
    tmp_112 <== CMul()(tmp_111, challengesDeep[1]);
    tmp_113 <== CMul()(tmp_112, challengesDeep[1]);
    tmp_114 <== CMul()(tmp_113, challengesDeep[1]);
    tmp_115 <== CMul()(tmp_114, challengesDeep[1]);
    tmp_116 <== CMul()(tmp_115, challengesDeep[1]);
    tmp_117 <== CMul()(tmp_116, challengesDeep[1]);
    tmp_118 <== CMul()(tmp_117, challengesDeep[1]);
    signal tmp_18[3] <== CMul()(challengesDeep[1], evals[1]);
    signal tmp_19[3] <== [evals[10][0] + tmp_18[0], evals[10][1] + tmp_18[1], evals[10][2] + tmp_18[2]];
    signal tmp_20[3] <== CMul()(tmp_108, evals[2]);
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_109, evals[3]);
    signal tmp_23[3] <== [tmp_21[0] + tmp_22[0], tmp_21[1] + tmp_22[1], tmp_21[2] + tmp_22[2]];
    signal tmp_24[3] <== CMul()(tmp_110, evals[4]);
    signal tmp_25[3] <== [tmp_23[0] + tmp_24[0], tmp_23[1] + tmp_24[1], tmp_23[2] + tmp_24[2]];
    signal tmp_26[3] <== CMul()(tmp_111, evals[5]);
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_112, evals[6]);
    signal tmp_29[3] <== [tmp_27[0] + tmp_28[0], tmp_27[1] + tmp_28[1], tmp_27[2] + tmp_28[2]];
    signal tmp_30[3] <== CMul()(tmp_113, evals[7]);
    signal tmp_31[3] <== [tmp_29[0] + tmp_30[0], tmp_29[1] + tmp_30[1], tmp_29[2] + tmp_30[2]];
    signal tmp_32[3] <== CMul()(tmp_114, evals[8]);
    signal tmp_33[3] <== [tmp_31[0] + tmp_32[0], tmp_31[1] + tmp_32[1], tmp_31[2] + tmp_32[2]];
    signal tmp_34[3] <== CMul()(tmp_115, evals[9]);
    signal tmp_35[3] <== [tmp_33[0] + tmp_34[0], tmp_33[1] + tmp_34[1], tmp_33[2] + tmp_34[2]];
    signal tmp_36[3] <== CMul()(tmp_116, evals[11]);
    signal tmp_37[3] <== [tmp_35[0] + tmp_36[0], tmp_35[1] + tmp_36[1], tmp_35[2] + tmp_36[2]];
    signal tmp_38[3] <== CMul()(tmp_117, evals[12]);
    signal tmp_39[3] <== [tmp_37[0] + tmp_38[0], tmp_37[1] + tmp_38[1], tmp_37[2] + tmp_38[2]];
    signal tmp_40[3] <== CMul()(tmp_118, evals[13]);
    tmp_121 <== [tmp_39[0] + tmp_40[0], tmp_39[1] + tmp_40[1], tmp_39[2] + tmp_40[2]];
    signal tmp_42[3] <== CMul()(tmp_109, evals[14]);
    signal tmp_43[3] <== CMul()(tmp_110, evals[15]);
    signal tmp_44[3] <== [tmp_42[0] + tmp_43[0], tmp_42[1] + tmp_43[1], tmp_42[2] + tmp_43[2]];
    signal tmp_45[3] <== CMul()(tmp_111, evals[16]);
    signal tmp_46[3] <== [tmp_44[0] + tmp_45[0], tmp_44[1] + tmp_45[1], tmp_44[2] + tmp_45[2]];
    signal tmp_47[3] <== CMul()(tmp_112, evals[17]);
    tmp_122 <== [tmp_46[0] + tmp_47[0], tmp_46[1] + tmp_47[1], tmp_46[2] + tmp_47[2]];
    signal tmp_49[3] <== CMul()(tmp_109, evals[18]);
    signal tmp_50[3] <== CMul()(tmp_111, evals[19]);
    signal tmp_51[3] <== [tmp_49[0] + tmp_50[0], tmp_49[1] + tmp_50[1], tmp_49[2] + tmp_50[2]];
    signal tmp_52[3] <== CMul()(tmp_112, evals[20]);
    tmp_123 <== [tmp_51[0] + tmp_52[0], tmp_51[1] + tmp_52[1], tmp_51[2] + tmp_52[2]];
    signal tmp_54[3] <== CMul()(tmp_109, evals[21]);
    signal tmp_55[3] <== CMul()(tmp_111, evals[22]);
    tmp_124 <== [tmp_54[0] + tmp_55[0], tmp_54[1] + tmp_55[1], tmp_54[2] + tmp_55[2]];
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
    signal input {binary} queriesL0[23];
    signal input challengeOut[3];
    signal input friConsts[20][3];
    signal input evals[23][3];
    signal input cm1[5];
    signal input cm2[9];
    signal input cm3[3];
    signal input consts[2];
    signal input custom_rom_0[2];

    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    signal xacc[23];
    xacc[0] <== queriesL0[0]*(7 * roots(23)-7) + 7;
    for (var i = 1; i < 23; i++) {
        xacc[i] <== xacc[i-1] * ( queriesL0[i]*(roots(23 - i) - 1) +1);
    }

    signal xDivXSubXi[5][3];

    xDivXSubXi[0] <== CInv()([xacc[22] - 10420286214021487819 * challengeOut[0], - 10420286214021487819 * challengeOut[1], - 10420286214021487819 * challengeOut[2]]);
    xDivXSubXi[1] <== CInv()([xacc[22] - 1 * challengeOut[0], - 1 * challengeOut[1], - 1 * challengeOut[2]]);
    xDivXSubXi[2] <== CInv()([xacc[22] - 8124823329697072476 * challengeOut[0], - 8124823329697072476 * challengeOut[1], - 8124823329697072476 * challengeOut[2]]);
    xDivXSubXi[3] <== CInv()([xacc[22] - 6553637399136210105 * challengeOut[0], - 6553637399136210105 * challengeOut[1], - 6553637399136210105 * challengeOut[2]]);
    xDivXSubXi[4] <== CInv()([xacc[22] - 331116024603048646 * challengeOut[0], - 331116024603048646 * challengeOut[1], - 331116024603048646 * challengeOut[2]]);

    // The query-independent terms, computed once in the StarkVerifier.
    signal challengesDeep[2][3] <== [friConsts[0], friConsts[1]];
    signal tmp_103[3] <== friConsts[2];
    signal tmp_104[3] <== friConsts[3];
    signal tmp_105[3] <== friConsts[4];
    signal tmp_108[3] <== friConsts[5];
    signal tmp_109[3] <== friConsts[6];
    signal tmp_110[3] <== friConsts[7];
    signal tmp_111[3] <== friConsts[8];
    signal tmp_112[3] <== friConsts[9];
    signal tmp_113[3] <== friConsts[10];
    signal tmp_114[3] <== friConsts[11];
    signal tmp_115[3] <== friConsts[12];
    signal tmp_116[3] <== friConsts[13];
    signal tmp_117[3] <== friConsts[14];
    signal tmp_118[3] <== friConsts[15];
    signal tmp_121[3] <== friConsts[16];
    signal tmp_122[3] <== friConsts[17];
    signal tmp_123[3] <== friConsts[18];
    signal tmp_124[3] <== friConsts[19];

    signal tmp_125[3];
    (tmp_125) <== CalculateDeepPolChunks0_0()(challengesDeep,evals,cm1,cm2,cm3,consts,custom_rom_0,xDivXSubXi,tmp_103,tmp_104,tmp_105,tmp_108,tmp_109,tmp_110,tmp_111,tmp_112,tmp_113,tmp_114,tmp_115,tmp_116,tmp_117,tmp_118,tmp_121,tmp_122,tmp_123,tmp_124);

    signal queryVals[3];
    queryVals[0] <== tmp_125[0];
    queryVals[1] <== tmp_125[1];
    queryVals[2] <== tmp_125[2];

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
    signal input vals1[5];
    signal input vals2[9];
    signal input vals3[3];
    signal input vals_rom_0[2];
    signal output cm1_0;
    signal output cm1_1;
    signal output cm1_2;
    signal output cm1_3;
    signal output cm1_4;
    signal output cm2_0[3];
    signal output cm2_1[3];
    signal output cm2_2[3];
    signal output cm3_0[3];

    signal output custom_rom_0_0;
    signal output custom_rom_0_1;


    custom_rom_0_0 <== vals_rom_0[0];
    custom_rom_0_1 <== vals_rom_0[1];

    cm1_0 <== vals1[0];
    cm1_1 <== vals1[1];
    cm1_2 <== vals1[2];
    cm1_3 <== vals1[3];
    cm1_4 <== vals1[4];
    cm2_0 <== [vals2[0], vals2[1], vals2[2]];
    cm2_1 <== [vals2[3], vals2[4], vals2[5]];
    cm2_2 <== [vals2[6], vals2[7], vals2[8]];
    cm3_0 <== [vals3[0], vals3[1], vals3[2]];
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
    signal input finalPol[64][3];
    signal input {binary} enable;
    
    // Calculate the IFFT to get the coefficients of finalPol 
    signal lastIFFT[64][3] <== FFT(6, 3, 1)(finalPol);

    // Check that the degree of the final polynomial is bounded by the degree defined in the last step of the folding
    for (var k = 32; k < 64; k++) {
        for (var e = 0; e < 3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k = 0; k < 32; k++) {
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
    signal input {binary} queriesL0[23];

    signal input challengeOut[3];
    signal input friConsts[20][3];
    signal input rFold[4][3];
    signal input evals[23][3];
    signal input {binary} enable;

    signal input s0_vals1[5];
    signal input s0_vals1_p[5][1];
    signal input s0_siblings1[19][4];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[9];
    signal input s0_vals2_p[9][1];
    signal input s0_siblings2[19][4];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[3];
    signal input s0_vals3_p[3][1];
    signal input s0_siblings3[19][4];

    signal input s0_valsC[2];
    signal input s0_valsC_p[2][1];
    signal input s0_siblingsC[19][4];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];

    signal input s0_vals_rom_0[2];
    signal input s0_vals_rom_0_p[2][1];
    signal input root_rom_0[4];
    signal input s0_siblings_rom_0[19][4];
    signal input s0_last_mt_levels_rom_0[16][4];

    signal input s1_vals_p[32][3];
    signal input s1_siblings[14][4];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[16][3];
    signal input s2_siblings[10][4];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[16][3];
    signal input s3_siblings[6][4];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[16][3];
    signal input s4_siblings[2][4];
    signal input s4_last_mt_levels[16][4];

    signal input finalPol[64][3];

    signal {binary} queriesL0Bits[23][1];
    for (var j = 0; j < 23; j++) {
        for (var k = 0; k < 1; k++) {
            if (k + j * 1 >= 23) {
                queriesL0Bits[j][k] <== 0;
            } else {
                queriesL0Bits[j][k] <== queriesL0[j*1 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(1, 5, 2, 19, 4, 8388608)(s0_vals1_p, s0_siblings1, queriesL0Bits, s0_last_mt_levels1, enable);
    VerifyMerkleHashUntilLevel(1, 9, 2, 19, 4, 8388608)(s0_vals2_p, s0_siblings2, queriesL0Bits, s0_last_mt_levels2, enable);

    VerifyMerkleHashUntilLevel(1, 3, 2, 19, 4, 8388608)(s0_vals3_p, s0_siblings3, queriesL0Bits, s0_last_mt_levels3, enable);
    VerifyMerkleHashUntilLevel(1, 2, 2, 19, 4, 8388608)(s0_valsC_p, s0_siblingsC, queriesL0Bits, s0_last_mt_levelsC, enable);

    VerifyMerkleHashUntilLevel(1, 2, 2, 19, 4, 8388608)(s0_vals_rom_0_p, s0_siblings_rom_0, queriesL0Bits, s0_last_mt_levels_rom_0, enable);

    signal {binary} s1_keys_merkle_bits[18][1];
    for (var j = 0; j < 18; j++) {
        for (var k = 0; k < 1; k++) {
            if (k + j * 1 >= 18) {
                s1_keys_merkle_bits[j][k] <== 0;
            } else {
                s1_keys_merkle_bits[j][k] <== queriesL0[j*1 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 32, 2, 14, 4, 262144)(s1_vals_p, s1_siblings, s1_keys_merkle_bits, s1_last_mt_levels, enable);
    signal {binary} s2_keys_merkle_bits[14][1];
    for (var j = 0; j < 14; j++) {
        for (var k = 0; k < 1; k++) {
            if (k + j * 1 >= 14) {
                s2_keys_merkle_bits[j][k] <== 0;
            } else {
                s2_keys_merkle_bits[j][k] <== queriesL0[j*1 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 16, 2, 10, 4, 16384)(s2_vals_p, s2_siblings, s2_keys_merkle_bits, s2_last_mt_levels, enable);
    signal {binary} s3_keys_merkle_bits[10][1];
    for (var j = 0; j < 10; j++) {
        for (var k = 0; k < 1; k++) {
            if (k + j * 1 >= 10) {
                s3_keys_merkle_bits[j][k] <== 0;
            } else {
                s3_keys_merkle_bits[j][k] <== queriesL0[j*1 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 16, 2, 6, 4, 1024)(s3_vals_p, s3_siblings, s3_keys_merkle_bits, s3_last_mt_levels, enable);
    signal {binary} s4_keys_merkle_bits[6][1];
    for (var j = 0; j < 6; j++) {
        for (var k = 0; k < 1; k++) {
            if (k + j * 1 >= 6) {
                s4_keys_merkle_bits[j][k] <== 0;
            } else {
                s4_keys_merkle_bits[j][k] <== queriesL0[j*1 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 16, 2, 2, 4, 64)(s4_vals_p, s4_siblings, s4_keys_merkle_bits, s4_last_mt_levels, enable);

    VerifyQuery0(23, 18)(queriesL0, challengeOut, friConsts, evals, s0_vals1, s0_vals2, s0_vals3, s0_valsC, s0_vals_rom_0, s1_vals_p, enable);

    signal {binary} s1_queriesL0[18];
    for (var i = 0; i < 18; i++) { s1_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(23, 23, 18, 14, 2635249152773512046)(s1_queriesL0, rFold[0], s1_vals_p, s2_vals_p, enable);
    signal {binary} s2_queriesL0[14];
    for (var i = 0; i < 14; i++) { s2_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(23, 18, 14, 10, 2058312433970512828)(s2_queriesL0, rFold[1], s2_vals_p, s3_vals_p, enable);
    signal {binary} s3_queriesL0[10];
    for (var i = 0; i < 10; i++) { s3_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(23, 14, 10, 6, 1138102428757299658)(s3_queriesL0, rFold[2], s3_vals_p, s4_vals_p, enable);
    signal {binary} s4_queriesL0[6];
    for (var i = 0; i < 6; i++) { s4_queriesL0[i] <== queriesL0[i]; }
    VerifyFRI0(23, 10, 6, 0, 13315202466730381196)(s4_queriesL0, rFold[3], s4_vals_p, finalPol, enable);
}

/*
    A batch of nQueries independent VerifySingleQuery0 instances. Batching exists
    only to keep the compiler happy: one parallel component per query, or one over all
    211 of them, is far more expensive to compile than a handful of medium batches.
    Signals shared by all queries (roots, challenges, the tree's last levels sent in clear)
    are passed through unsliced.
*/
template parallel VerifyQueriesBatch0(nQueries) {
    signal input {binary} queriesL0[nQueries][23];

    signal input challengeOut[3];
    signal input friConsts[20][3];
    signal input rFold[4][3];
    signal input evals[23][3];
    signal input {binary} enable;

    signal input s0_vals1[nQueries][5];
    signal input s0_vals1_p[nQueries][5][1];
    signal input s0_siblings1[nQueries][19][4];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[nQueries][9];
    signal input s0_vals2_p[nQueries][9][1];
    signal input s0_siblings2[nQueries][19][4];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[nQueries][3];
    signal input s0_vals3_p[nQueries][3][1];
    signal input s0_siblings3[nQueries][19][4];

    signal input s0_valsC[nQueries][2];
    signal input s0_valsC_p[nQueries][2][1];
    signal input s0_siblingsC[nQueries][19][4];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];

    signal input s0_vals_rom_0[nQueries][2];
    signal input s0_vals_rom_0_p[nQueries][2][1];
    signal input root_rom_0[4];
    signal input s0_siblings_rom_0[nQueries][19][4];
    signal input s0_last_mt_levels_rom_0[16][4];

    signal input s1_vals_p[nQueries][32][3];
    signal input s1_siblings[nQueries][14][4];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[nQueries][16][3];
    signal input s2_siblings[nQueries][10][4];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[nQueries][16][3];
    signal input s3_siblings[nQueries][6][4];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[nQueries][16][3];
    signal input s4_siblings[nQueries][2][4];
    signal input s4_last_mt_levels[16][4];

    signal input finalPol[64][3];

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
            s0_vals_rom_0[q],
            s0_vals_rom_0_p[q],
            root_rom_0,
            s0_siblings_rom_0[q],
            s0_last_mt_levels_rom_0,
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
            finalPol
        );
    }
}

/*
    Verifies one STARK proof of the air 'FibonacciSquare' (blake3 transcript and Merkle trees):
    replays the Fiat-Shamir transcript to derive every challenge, checks that the evaluations
    at the out-of-domain point satisfy the constraint polynomial, opens the committed trees at
    the query positions, and runs the low-degree test on the DEEP polynomial f_0 down to the
    final polynomial.

    Trace:        2^22 rows, 2 committed stage(s) plus the quotient, split in 1 polynomial(s), in stage 3
    Extension:    2^23 points (blowup 2^1)
    Commitments:  Merkle trees of arity 2, last 4 level(s) sent in clear, hashed commits
    Low-degree test: FRI
      folding factors (log2): [5, 4, 4, 4]
      log2 domain sizes:      [23, 18, 14, 10, 6]
      log2 degree bounds:     [22, 17, 13, 9, 5] (final polynomial of degree lower than 2^5)
      queries:                211
      grinding bits:          24
*/
template StarkVerifier0() {
    // Global challenge for the shared Fiat-Shamir transcript
    signal input globalChallenge[3];

    signal input publics[8];

    signal input airvalues[3][3];
    signal input airgroupvalues[2][3];
    signal input proofvalues[2][3];

    signal input root1[4]; // Commitment to witness polynomials of stage 1
    signal input root2[4]; // Commitment to witness polynomials of stage 2
    signal input root3[4]; // Commitment to the quotient polynomial Q

    signal output rootC[4] <== [17284145787926719296,716892807869615064,12286782145902210190,5219202236615833098 ]; // Commitment to fixed polynomials

    signal root_rom_0[4] <== [publics[4], publics[5], publics[6], publics[7]]; // Commitment to the custom commit 'rom', bound through the publics
    
    signal input evals[23][3]; // Evaluations of commited polynomials at the out-of-domain sample point z and shifts gˢ·z

    airgroupvalues[0][1] === 0;
    airgroupvalues[0][2] === 0;
    // Leaf values of the merkle tree used to check all the queries
    signal input s0_vals1[211][5];
    signal input s0_vals2[211][9];
    signal input s0_vals3[211][3];
    signal input s0_valsC[211][2];

    signal input s0_vals_rom_0[211][2];

    // Merkle proofs for each of the evaluations
    signal input s0_siblings1[211][19][4];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_siblings2[211][19][4];
    signal input s0_last_mt_levels2[16][4];
    signal input s0_siblings3[211][19][4];
    signal input s0_last_mt_levels3[16][4];
    signal input s0_siblingsC[211][19][4];
    signal input s0_last_mt_levelsC[16][4];
    signal input s0_siblings_rom_0[211][19][4];
    signal input s0_last_mt_levels_rom_0[16][4];

    // Contains the root of the DEEP polynomial and of every folded oracle.
    signal input s1_root[4]; // Root of a polynomial of degree lower than 2^22, evaluated over 2^23 points
    signal input s2_root[4]; // Root of a polynomial of degree lower than 2^17, evaluated over 2^18 points
    signal input s3_root[4]; // Root of a polynomial of degree lower than 2^13, evaluated over 2^14 points
    signal input s4_root[4]; // Root of a polynomial of degree lower than 2^9, evaluated over 2^10 points

    // For each committed oracle, the values needed to check its round's queries: a query x
    // into Fold(f_{i-1}, k_{i-1}, r^fold_{i-1}) : L_i --> 𝔽 involves querying f_{i-1} at all
    // 2^{k_{i-1}}-roots of x.
    signal input s1_vals[211][96];
    signal input s1_siblings[211][14][4];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals[211][48];
    signal input s2_siblings[211][10][4];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals[211][48];
    signal input s3_siblings[211][6][4];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals[211][48];
    signal input s4_siblings[211][2][4];
    signal input s4_last_mt_levels[16][4];

    // The final polynomial is sent in clear as evaluations over the last folded domain.
    signal input finalPol[64][3]; // Evaluations over 2^6 points of a polynomial of degree lower than 2^5

    // The grinding nonce.
    signal input nonce;

    signal {binary} enabled;
    enabled <== 1;


    signal queryVals[211][3];

    signal challengesStage2[2][3];

    signal challengeQ[3];
    signal challengeOut[3];
    signal challengesDeep[2][3];

    // r^fold_s: the random value the verifier provides once f_s is committed, under which the
    // prover folds f_s into f_{s+1} and commits that.
    signal rFold[4][3];

    // Challenges from which we derive all the queries
    signal {binary} queriesL0[211][23];

    ///////////
    // Check that the verifier's challenges are well derived from the prover messages via the Fiat-Shamir heuristic
    ///////////

    (challengesStage2,challengeQ,challengeOut,challengesDeep,rFold,queriesL0) <== Transcript0()(globalChallenge,airvalues,airgroupvalues,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,finalPol, nonce, enabled);

    ///////////
    // DEEP polynomial terms that are the same for every query
    ///////////
    signal tmp_103[3];
    signal tmp_104[3];
    signal tmp_105[3];
    signal tmp_108[3];
    signal tmp_109[3];
    signal tmp_110[3];
    signal tmp_111[3];
    signal tmp_112[3];
    signal tmp_113[3];
    signal tmp_114[3];
    signal tmp_115[3];
    signal tmp_116[3];
    signal tmp_117[3];
    signal tmp_118[3];
    signal tmp_121[3];
    signal tmp_122[3];
    signal tmp_123[3];
    signal tmp_124[3];
    (tmp_103,tmp_104,tmp_105,tmp_108,tmp_109,tmp_110,tmp_111,tmp_112,tmp_113,tmp_114,tmp_115,tmp_116,tmp_117,tmp_118,tmp_121,tmp_122,tmp_123,tmp_124) <== CalculateFRIConstantsChunks0()(challengesDeep,evals);

    signal friConsts[20][3];
    friConsts[0] <== challengesDeep[0];
    friConsts[1] <== challengesDeep[1];
    friConsts[2] <== tmp_103;
    friConsts[3] <== tmp_104;
    friConsts[4] <== tmp_105;
    friConsts[5] <== tmp_108;
    friConsts[6] <== tmp_109;
    friConsts[7] <== tmp_110;
    friConsts[8] <== tmp_111;
    friConsts[9] <== tmp_112;
    friConsts[10] <== tmp_113;
    friConsts[11] <== tmp_114;
    friConsts[12] <== tmp_115;
    friConsts[13] <== tmp_116;
    friConsts[14] <== tmp_117;
    friConsts[15] <== tmp_118;
    friConsts[16] <== tmp_121;
    friConsts[17] <== tmp_122;
    friConsts[18] <== tmp_123;
    friConsts[19] <== tmp_124;

    ///////////
    // Check constraints are consistent with Q polynomial at the out-of-domain points
    ///////////

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeOut, evals, publics, airgroupvalues, airvalues, proofvalues, enabled);

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
    var s0_vals1_p[211][5][1];
    var s0_vals2_p[211][9][1];
    var s0_vals3_p[211][3][1];
    var s0_valsC_p[211][2][1];
    var s0_vals_rom_0_p[211][2][1];
    var s0_vals_p[211][1][3]; 
    var s1_vals_p[211][32][3]; 
    var s2_vals_p[211][16][3]; 
    var s3_vals_p[211][16][3]; 
    var s4_vals_p[211][16][3]; 

    for (var q = 0; q < 211; q++) {
        for (var i = 0; i < 5; i++) {
            s0_vals1_p[q][i][0] = s0_vals1[q][i];
        }
        for (var i = 0; i < 9; i++) {
            s0_vals2_p[q][i][0] = s0_vals2[q][i];
        }
        for (var i = 0; i < 3; i++) {
            s0_vals3_p[q][i][0] = s0_vals3[q][i];
        }
        for (var i = 0; i < 2; i++) {
            s0_valsC_p[q][i][0] = s0_valsC[q][i];
        }
        for (var i = 0; i < 2; i++) {
            s0_vals_rom_0_p[q][i][0] = s0_vals_rom_0[q][i];
        }

        // Preprocess vals for each folded polynomial
        for (var e = 0; e < 3; e++) {
            for (var c = 0; c < 32; c++) {
                s1_vals_p[q][c][e] = s1_vals[q][c*3+e];
            }
            for (var c = 0; c < 16; c++) {
                s2_vals_p[q][c][e] = s2_vals[q][c*3+e];
            }
            for (var c = 0; c < 16; c++) {
                s3_vals_p[q][c][e] = s3_vals[q][c*3+e];
            }
            for (var c = 0; c < 16; c++) {
                s4_vals_p[q][c][e] = s4_vals[q][c*3+e];
            }
        }
    }

    ///////////
    // Per-query checks, in batches of QUERIES_BATCH_SIZE: each VerifySingleQuery0 opens
    // every L_0 tree at its query, cross-checks the DEEP value against f_0 and
    // runs the FRI folding steps
    ///////////

    // Batch-size parameters — change QUERIES_BATCH_SIZE and recompile to test different batch sizes.
    var N_QUERIES = 211;
    var QUERIES_BATCH_SIZE = 27;
    var N_FULL_BATCHES = N_QUERIES \ QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE = N_QUERIES - N_FULL_BATCHES * QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE_SAFE = LAST_BATCH_SIZE > 0 ? LAST_BATCH_SIZE : 1;

    // Work buffers for full batches
    signal {binary} batch_work_queriesL0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][23];

    signal batch_work_s0_vals1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5];
    signal batch_work_s0_vals1_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5][1];
    signal batch_work_s0_siblings1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][19][4];
    signal batch_work_s0_vals2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][9];
    signal batch_work_s0_vals2_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][9][1];
    signal batch_work_s0_siblings2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][19][4];

    signal batch_work_s0_vals3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][3];
    signal batch_work_s0_vals3_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][3][1];
    signal batch_work_s0_siblings3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][19][4];
    signal batch_work_s0_valsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2];
    signal batch_work_s0_valsC_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][1];
    signal batch_work_s0_siblingsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][19][4];

    signal batch_work_s0_vals_rom_0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2];
    signal batch_work_s0_vals_rom_0_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][1];
    signal batch_work_s0_siblings_rom_0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][19][4];

    signal batch_work_s1_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][32][3];
    signal batch_work_s1_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][14][4];
    signal batch_work_s2_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][16][3];
    signal batch_work_s2_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][4];
    signal batch_work_s3_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][16][3];
    signal batch_work_s3_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][6][4];
    signal batch_work_s4_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][16][3];
    signal batch_work_s4_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][4];

    // Process full batches
    for (var b = 0; b < N_FULL_BATCHES; b++) {
        var batchStart = b * QUERIES_BATCH_SIZE;

        // Fill work buffers for batch b
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 23; i++) {
                batch_work_queriesL0[b][q][i] <== queriesL0[batchStart + q][i];
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 5; i++) {
                batch_work_s0_vals1[b][q][i] <== s0_vals1[batchStart + q][i];
                batch_work_s0_vals1_p[b][q][i][0] <== s0_vals1_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 19; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s0_siblings1[b][q][j][k] <== s0_siblings1[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 9; i++) {
                batch_work_s0_vals2[b][q][i] <== s0_vals2[batchStart + q][i];
                batch_work_s0_vals2_p[b][q][i][0] <== s0_vals2_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 19; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s0_siblings2[b][q][j][k] <== s0_siblings2[batchStart + q][j][k];
                }
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 3; i++) {
                batch_work_s0_vals3[b][q][i] <== s0_vals3[batchStart + q][i];
                batch_work_s0_vals3_p[b][q][i][0] <== s0_vals3_p[batchStart + q][i][0];
            }
            for (var i = 0; i < 2; i++) {
                batch_work_s0_valsC[b][q][i] <== s0_valsC[batchStart + q][i];
                batch_work_s0_valsC_p[b][q][i][0] <== s0_valsC_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 19; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s0_siblings3[b][q][j][k] <== s0_siblings3[batchStart + q][j][k];
                    batch_work_s0_siblingsC[b][q][j][k] <== s0_siblingsC[batchStart + q][j][k];
                }
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 2; i++) {
                batch_work_s0_vals_rom_0[b][q][i] <== s0_vals_rom_0[batchStart + q][i];
                batch_work_s0_vals_rom_0_p[b][q][i][0] <== s0_vals_rom_0_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 19; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s0_siblings_rom_0[b][q][j][k] <== s0_siblings_rom_0[batchStart + q][j][k];
                }
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 32; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s1_vals_p[b][q][c][e] <== s1_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 14; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s1_siblings[b][q][j][k] <== s1_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 16; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s2_vals_p[b][q][c][e] <== s2_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 10; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s2_siblings[b][q][j][k] <== s2_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 16; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s3_vals_p[b][q][c][e] <== s3_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 6; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s3_siblings[b][q][j][k] <== s3_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 16; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s4_vals_p[b][q][c][e] <== s4_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 2; j++) {
                for (var k = 0; k < 4; k++) {
                    batch_work_s4_siblings[b][q][j][k] <== s4_siblings[batchStart + q][j][k];
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
            batch_work_s0_vals_rom_0[b],
            batch_work_s0_vals_rom_0_p[b],
            root_rom_0,
            batch_work_s0_siblings_rom_0[b],
            s0_last_mt_levels_rom_0,
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
            finalPol
        );
    }

    // Remainder batch
    signal {binary} remainder_queriesL0[LAST_BATCH_SIZE_SAFE][23];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 23; i++) {
            remainder_queriesL0[q][i] <== queriesL0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
        }
    }

    signal remainder_s0_vals1[LAST_BATCH_SIZE_SAFE][5];
    signal remainder_s0_vals1_p[LAST_BATCH_SIZE_SAFE][5][1];
    signal remainder_s0_siblings1[LAST_BATCH_SIZE_SAFE][19][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 5; i++) {
            remainder_s0_vals1[q][i] <== s0_vals1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals1_p[q][i][0] <== s0_vals1_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 19; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s0_siblings1[q][j][k] <== s0_siblings1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s0_vals2[LAST_BATCH_SIZE_SAFE][9];
    signal remainder_s0_vals2_p[LAST_BATCH_SIZE_SAFE][9][1];
    signal remainder_s0_siblings2[LAST_BATCH_SIZE_SAFE][19][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 9; i++) {
            remainder_s0_vals2[q][i] <== s0_vals2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals2_p[q][i][0] <== s0_vals2_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 19; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s0_siblings2[q][j][k] <== s0_siblings2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s0_vals3[LAST_BATCH_SIZE_SAFE][3];
    signal remainder_s0_vals3_p[LAST_BATCH_SIZE_SAFE][3][1];
    signal remainder_s0_siblings3[LAST_BATCH_SIZE_SAFE][19][4];
    signal remainder_s0_valsC[LAST_BATCH_SIZE_SAFE][2];
    signal remainder_s0_valsC_p[LAST_BATCH_SIZE_SAFE][2][1];
    signal remainder_s0_siblingsC[LAST_BATCH_SIZE_SAFE][19][4];

    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 3; i++) {
            remainder_s0_vals3[q][i] <== s0_vals3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals3_p[q][i][0] <== s0_vals3_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var i = 0; i < 2; i++) {
            remainder_s0_valsC[q][i] <== s0_valsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_valsC_p[q][i][0] <== s0_valsC_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 19; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s0_siblings3[q][j][k] <== s0_siblings3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
                remainder_s0_siblingsC[q][j][k] <== s0_siblingsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s0_vals_rom_0[LAST_BATCH_SIZE_SAFE][2];
    signal remainder_s0_vals_rom_0_p[LAST_BATCH_SIZE_SAFE][2][1];
    signal remainder_s0_siblings_rom_0[LAST_BATCH_SIZE_SAFE][19][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 2; i++) {
            remainder_s0_vals_rom_0[q][i] <== s0_vals_rom_0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals_rom_0_p[q][i][0] <== s0_vals_rom_0_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 19; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s0_siblings_rom_0[q][j][k] <== s0_siblings_rom_0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s1_vals_p[LAST_BATCH_SIZE_SAFE][32][3];
    signal remainder_s1_siblings[LAST_BATCH_SIZE_SAFE][14][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 32; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s1_vals_p[q][c][e] <== s1_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 14; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s1_siblings[q][j][k] <== s1_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s2_vals_p[LAST_BATCH_SIZE_SAFE][16][3];
    signal remainder_s2_siblings[LAST_BATCH_SIZE_SAFE][10][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 16; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s2_vals_p[q][c][e] <== s2_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 10; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s2_siblings[q][j][k] <== s2_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s3_vals_p[LAST_BATCH_SIZE_SAFE][16][3];
    signal remainder_s3_siblings[LAST_BATCH_SIZE_SAFE][6][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 16; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s3_vals_p[q][c][e] <== s3_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 6; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s3_siblings[q][j][k] <== s3_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s4_vals_p[LAST_BATCH_SIZE_SAFE][16][3];
    signal remainder_s4_siblings[LAST_BATCH_SIZE_SAFE][2][4];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 16; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s4_vals_p[q][c][e] <== s4_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 4; k++) {
                remainder_s4_siblings[q][j][k] <== s4_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
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
            remainder_s0_vals_rom_0,
            remainder_s0_vals_rom_0_p,
            root_rom_0,
            remainder_s0_siblings_rom_0,
            s0_last_mt_levels_rom_0,
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
            finalPol
        );
    }


    ///////////
    // Verify Merkle roots for optimized last levels
    ///////////

    VerifyMerkleRoot(4, 2, 8388608)(s0_last_mt_levels1, root1, enabled);
    VerifyMerkleRoot(4, 2, 8388608)(s0_last_mt_levels2, root2, enabled);

    VerifyMerkleRoot(4, 2, 8388608)(s0_last_mt_levels3, root3, enabled);

    VerifyMerkleRoot(4, 2, 8388608)(s0_last_mt_levelsC, rootC, enabled);

    VerifyMerkleRoot(4, 2, 8388608)(s0_last_mt_levels_rom_0, root_rom_0, enabled);

    VerifyMerkleRoot(4, 2, 262144)(s1_last_mt_levels, s1_root, enabled);
    VerifyMerkleRoot(4, 2, 16384)(s2_last_mt_levels, s2_root, enabled);
    VerifyMerkleRoot(4, 2, 1024)(s3_last_mt_levels, s3_root, enabled);
    VerifyMerkleRoot(4, 2, 64)(s4_last_mt_levels, s4_root, enabled);


    ///////////
    // Check the degree of the final polynomial
    ///////////

    VerifyFinalPol0()(finalPol, enabled);
}

