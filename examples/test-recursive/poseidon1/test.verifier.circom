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
    signal output {binary} queriesFRI[229][23];

    VerifyPoW(16)(challengeFRIQueries, nonce, enable);


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
    signal {binary} transcriptN2b_24[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[8]);
    signal {binary} transcriptN2b_25[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[9]);
    signal {binary} transcriptN2b_26[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[10]);
    signal {binary} transcriptN2b_27[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[11]);
    signal {binary} transcriptN2b_28[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[12]);
    signal {binary} transcriptN2b_29[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[13]);
    signal {binary} transcriptN2b_30[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[14]);
    signal {binary} transcriptN2b_31[64] <== Num2Bits_strict()(transcriptHash_friQueries_1[15]);

    signal transcriptHash_friQueries_2[16] <== Poseidon(16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_1[0],transcriptHash_friQueries_1[1],transcriptHash_friQueries_1[2],transcriptHash_friQueries_1[3]]);
    signal {binary} transcriptN2b_32[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[0]);
    signal {binary} transcriptN2b_33[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[1]);
    signal {binary} transcriptN2b_34[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[2]);
    signal {binary} transcriptN2b_35[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[3]);
    signal {binary} transcriptN2b_36[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[4]);
    signal {binary} transcriptN2b_37[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[5]);
    signal {binary} transcriptN2b_38[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[6]);
    signal {binary} transcriptN2b_39[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[7]);
    signal {binary} transcriptN2b_40[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[8]);
    signal {binary} transcriptN2b_41[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[9]);
    signal {binary} transcriptN2b_42[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[10]);
    signal {binary} transcriptN2b_43[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[11]);
    signal {binary} transcriptN2b_44[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[12]);
    signal {binary} transcriptN2b_45[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[13]);
    signal {binary} transcriptN2b_46[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[14]);
    signal {binary} transcriptN2b_47[64] <== Num2Bits_strict()(transcriptHash_friQueries_2[15]);

    signal transcriptHash_friQueries_3[16] <== Poseidon(16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_2[0],transcriptHash_friQueries_2[1],transcriptHash_friQueries_2[2],transcriptHash_friQueries_2[3]]);
    signal {binary} transcriptN2b_48[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[0]);
    signal {binary} transcriptN2b_49[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[1]);
    signal {binary} transcriptN2b_50[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[2]);
    signal {binary} transcriptN2b_51[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[3]);
    signal {binary} transcriptN2b_52[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[4]);
    signal {binary} transcriptN2b_53[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[5]);
    signal {binary} transcriptN2b_54[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[6]);
    signal {binary} transcriptN2b_55[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[7]);
    signal {binary} transcriptN2b_56[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[8]);
    signal {binary} transcriptN2b_57[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[9]);
    signal {binary} transcriptN2b_58[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[10]);
    signal {binary} transcriptN2b_59[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[11]);
    signal {binary} transcriptN2b_60[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[12]);
    signal {binary} transcriptN2b_61[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[13]);
    signal {binary} transcriptN2b_62[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[14]);
    signal {binary} transcriptN2b_63[64] <== Num2Bits_strict()(transcriptHash_friQueries_3[15]);

    signal transcriptHash_friQueries_4[16] <== Poseidon(16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_3[0],transcriptHash_friQueries_3[1],transcriptHash_friQueries_3[2],transcriptHash_friQueries_3[3]]);
    signal {binary} transcriptN2b_64[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[0]);
    signal {binary} transcriptN2b_65[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[1]);
    signal {binary} transcriptN2b_66[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[2]);
    signal {binary} transcriptN2b_67[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[3]);
    signal {binary} transcriptN2b_68[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[4]);
    signal {binary} transcriptN2b_69[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[5]);
    signal {binary} transcriptN2b_70[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[6]);
    signal {binary} transcriptN2b_71[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[7]);
    signal {binary} transcriptN2b_72[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[8]);
    signal {binary} transcriptN2b_73[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[9]);
    signal {binary} transcriptN2b_74[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[10]);
    signal {binary} transcriptN2b_75[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[11]);
    signal {binary} transcriptN2b_76[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[12]);
    signal {binary} transcriptN2b_77[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[13]);
    signal {binary} transcriptN2b_78[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[14]);
    signal {binary} transcriptN2b_79[64] <== Num2Bits_strict()(transcriptHash_friQueries_4[15]);

    signal transcriptHash_friQueries_5[16] <== Poseidon(16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_4[0],transcriptHash_friQueries_4[1],transcriptHash_friQueries_4[2],transcriptHash_friQueries_4[3]]);
    signal {binary} transcriptN2b_80[64] <== Num2Bits_strict()(transcriptHash_friQueries_5[0]);
    signal {binary} transcriptN2b_81[64] <== Num2Bits_strict()(transcriptHash_friQueries_5[1]);
    signal {binary} transcriptN2b_82[64] <== Num2Bits_strict()(transcriptHash_friQueries_5[2]);
    signal {binary} transcriptN2b_83[64] <== Num2Bits_strict()(transcriptHash_friQueries_5[3]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_friQueries_5[i]; // Unused transcript values        
    }

    // From each transcript hash converted to bits, we assign those bits to queriesFRI[q] to define the query positions
    var q = 0; // Query number 
    var b = 0; // Bit number 
    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_0[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_0[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_1[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_1[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_2[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_2[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_3[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_3[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_4[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_4[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_5[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_5[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_6[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_6[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_7[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_7[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_8[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_8[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_9[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_9[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_10[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_10[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_11[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_11[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_12[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_12[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_13[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_13[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_14[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_14[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_15[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_15[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_16[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_16[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_17[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_17[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_18[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_18[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_19[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_19[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_20[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_20[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_21[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_21[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_22[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_22[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_23[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_23[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_24[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_24[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_25[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_25[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_26[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_26[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_27[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_27[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_28[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_28[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_29[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_29[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_30[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_30[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_31[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_31[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_32[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_32[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_33[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_33[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_34[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_34[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_35[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_35[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_36[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_36[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_37[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_37[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_38[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_38[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_39[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_39[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_40[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_40[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_41[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_41[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_42[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_42[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_43[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_43[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_44[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_44[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_45[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_45[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_46[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_46[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_47[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_47[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_48[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_48[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_49[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_49[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_50[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_50[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_51[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_51[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_52[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_52[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_53[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_53[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_54[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_54[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_55[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_55[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_56[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_56[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_57[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_57[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_58[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_58[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_59[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_59[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_60[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_60[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_61[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_61[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_62[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_62[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_63[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_63[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_64[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_64[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_65[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_65[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_66[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_66[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_67[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_67[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_68[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_68[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_69[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_69[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_70[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_70[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_71[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_71[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_72[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_72[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_73[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_73[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_74[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_74[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_75[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_75[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_76[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_76[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_77[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_77[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_78[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_78[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_79[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_79[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_80[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_80[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_81[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_81[63]; // Unused last bit

    for(var j = 0; j < 63; j++) {
        queriesFRI[q][b] <== transcriptN2b_82[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    _ <== transcriptN2b_82[63]; // Unused last bit

    for(var j = 0; j < 38; j++) {
        queriesFRI[q][b] <== transcriptN2b_83[j];
        b++;
        if(b == 23) {
            b = 0; 
            q++;
        }
    }
    for(var j = 38; j < 64; j++) {
        _ <== transcriptN2b_83[j]; // Unused bits        
    }
}

/* 
    Calculate the transcript
*/ 
template Transcript0() {
    signal input globalChallenge[3]; 


    signal input airValues[3][3];

    signal input root2[4];
    signal input root3[4];
    signal input evals[23][3]; 
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];
    signal input s6_root[4];
    signal input finalPol[32][3];
    signal input nonce;
    signal input {binary} enable;

    signal output challengesStage2[2][3];
    signal output challengeQ[3];
    signal output challengeXi[3];
    signal output challengesFRI[2][3];
    signal output challengesFRISteps[8][3];
    signal output {binary} queriesFRI[229][23];

    signal publicsHash[4];
    signal evalsHash[4];
    signal lastPolFRIHash[4];



    signal transcriptHash_0[16] <== Poseidon(16)([globalChallenge[0],globalChallenge[1],globalChallenge[2],0,0,0,0,0,0,0,0,0], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for(var i = 6; i < 16; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }

    signal transcriptHash_1[16] <== Poseidon(16)([root2[0],root2[1],root2[2],root2[3],airValues[2][0],airValues[2][1],airValues[2][2],0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
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

    signal transcriptHash_evals_5[16] <== Poseidon(16)([evals[20][0],evals[20][1],evals[20][2],evals[21][0],evals[21][1],evals[21][2],evals[22][0],evals[22][1],evals[22][2],0,0,0], [transcriptHash_evals_4[0],transcriptHash_evals_4[1],transcriptHash_evals_4[2],transcriptHash_evals_4[3]]);
    evalsHash <== [transcriptHash_evals_5[0], transcriptHash_evals_5[1], transcriptHash_evals_5[2], transcriptHash_evals_5[3]];
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
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }

    signal transcriptHash_9[16] <== Poseidon(16)([s6_root[0],s6_root[1],s6_root[2],s6_root[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    challengesFRISteps[6] <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];

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
        _ <== transcriptHash_9[i]; // Unused transcript values 
    }

    signal transcriptHash_10[16] <== Poseidon(16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_9[0],transcriptHash_9[1],transcriptHash_9[2],transcriptHash_9[3]]);
    challengesFRISteps[7] <== [transcriptHash_10[0], transcriptHash_10[1], transcriptHash_10[2]];
    queriesFRI <== calculateFRIQueries0()(challengesFRISteps[7], nonce, enable);
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
    signal input evals[23][3];
    signal input publics[8];
    signal input airgroupvalues[1][3];
    signal input airvalues[3][3];
    signal input proofvalues[2][3];

    signal input Zh[3];


    signal output tmp_179[3];
    signal tmp_91[3] <== [evals[5][0] - publics[1], evals[5][1], evals[5][2]];
    signal tmp_92[3] <== CMul()(evals[3], tmp_91);
    signal tmp_93[3] <== CMul()(challengeQ, tmp_92);
    signal tmp_94[3] <== [evals[6][0] - publics[2], evals[6][1], evals[6][2]];
    signal tmp_95[3] <== CMul()(evals[3], tmp_94);
    signal tmp_96[3] <== [tmp_93[0] + tmp_95[0], tmp_93[1] + tmp_95[1], tmp_93[2] + tmp_95[2]];
    signal tmp_97[3] <== CMul()(challengeQ, tmp_96);
    signal tmp_98[3] <== [evals[6][0] - publics[3], evals[6][1], evals[6][2]];
    signal tmp_99[3] <== CMul()(evals[14], tmp_98);
    signal tmp_100[3] <== [tmp_97[0] + tmp_99[0], tmp_97[1] + tmp_99[1], tmp_97[2] + tmp_99[2]];
    signal tmp_101[3] <== CMul()(challengeQ, tmp_100);
    signal tmp_102[3] <== [evals[16][0] - evals[6][0], evals[16][1] - evals[6][1], evals[16][2] - evals[6][2]];
    signal tmp_103[3] <== [1 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_104[3] <== CMul()(tmp_102, tmp_103);
    signal tmp_105[3] <== [tmp_101[0] + tmp_104[0], tmp_101[1] + tmp_104[1], tmp_101[2] + tmp_104[2]];
    signal tmp_106[3] <== CMul()(challengeQ, tmp_105);
    signal tmp_107[3] <== [evals[19][0] - evals[17][0], evals[19][1] - evals[17][1], evals[19][2] - evals[17][2]];
    signal tmp_108[3] <== [1 - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_109[3] <== CMul()(tmp_107, tmp_108);
    signal tmp_110[3] <== [tmp_106[0] + tmp_109[0], tmp_106[1] + tmp_109[1], tmp_106[2] + tmp_109[2]];
    signal tmp_111[3] <== CMul()(challengeQ, tmp_110);
    signal tmp_112[3] <== [evals[22][0] - evals[20][0], evals[22][1] - evals[20][1], evals[22][2] - evals[20][2]];
    signal tmp_113[3] <== [1 - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_114[3] <== CMul()(tmp_112, tmp_113);
    signal tmp_115[3] <== [tmp_111[0] + tmp_114[0], tmp_111[1] + tmp_114[1], tmp_111[2] + tmp_114[2]];
    signal tmp_116[3] <== CMul()(challengeQ, tmp_115);
    signal tmp_117 <== publics[0] * proofvalues[0][0];
    signal tmp_118 <== tmp_117 - proofvalues[1][0];
    signal tmp_119[3] <== [tmp_116[0] + tmp_118, tmp_116[1], tmp_116[2]];
    signal tmp_120[3] <== CMul()(challengeQ, tmp_119);
    signal tmp_121 <== 2 * airvalues[0][0];
    signal tmp_122 <== tmp_121 - airvalues[1][0];
    signal tmp_123[3] <== [tmp_120[0] + tmp_122, tmp_120[1], tmp_120[2]];
    signal tmp_124[3] <== CMul()(challengeQ, tmp_123);
    signal tmp_125[3] <== [evals[2][0] + 1, evals[2][1], evals[2][2]];
    signal tmp_126[3] <== [evals[1][0] - tmp_125[0], evals[1][1] - tmp_125[1], evals[1][2] - tmp_125[2]];
    signal tmp_127[3] <== [tmp_124[0] + tmp_126[0], tmp_124[1] + tmp_126[1], tmp_124[2] + tmp_126[2]];
    signal tmp_128[3] <== CMul()(challengeQ, tmp_127);
    signal tmp_129[3] <== [2 * evals[5][0], 2 * evals[5][1], 2 * evals[5][2]];
    signal tmp_130[3] <== [tmp_129[0] + evals[6][0], tmp_129[1] + evals[6][1], tmp_129[2] + evals[6][2]];
    signal tmp_131[3] <== [tmp_130[0] + airvalues[0][0], tmp_130[1], tmp_130[2]];
    signal tmp_132[3] <== [54 * evals[1][0], 54 * evals[1][1], 54 * evals[1][2]];
    signal tmp_133[3] <== [tmp_131[0] + tmp_132[0], tmp_131[1] + tmp_132[1], tmp_131[2] + tmp_132[2]];
    signal tmp_134[3] <== [evals[3][0] * publics[1], evals[3][1] * publics[1], evals[3][2] * publics[1]];
    signal tmp_135[3] <== [tmp_133[0] + tmp_134[0], tmp_133[1] + tmp_134[1], tmp_133[2] + tmp_134[2]];
    signal tmp_136[3] <== [evals[7][0] - tmp_135[0], evals[7][1] - tmp_135[1], evals[7][2] - tmp_135[2]];
    signal tmp_137[3] <== [tmp_128[0] + tmp_136[0], tmp_128[1] + tmp_136[1], tmp_128[2] + tmp_136[2]];
    signal tmp_138[3] <== CMul()(challengeQ, tmp_137);
    signal tmp_139[3] <== [2 * evals[5][0], 2 * evals[5][1], 2 * evals[5][2]];
    signal tmp_140[3] <== [tmp_139[0] + evals[6][0], tmp_139[1] + evals[6][1], tmp_139[2] + evals[6][2]];
    signal tmp_141[3] <== [evals[8][0] - tmp_140[0], evals[8][1] - tmp_140[1], evals[8][2] - tmp_140[2]];
    signal tmp_142[3] <== [tmp_138[0] + tmp_141[0], tmp_138[1] + tmp_141[1], tmp_138[2] + tmp_141[2]];
    signal tmp_143[3] <== CMul()(challengeQ, tmp_142);
    signal tmp_144[3] <== [evals[8][0] + airvalues[0][0], evals[8][1], evals[8][2]];
    signal tmp_145[3] <== [54 * evals[1][0], 54 * evals[1][1], 54 * evals[1][2]];
    signal tmp_146[3] <== [tmp_144[0] + tmp_145[0], tmp_144[1] + tmp_145[1], tmp_144[2] + tmp_145[2]];
    signal tmp_147[3] <== [evals[3][0] * publics[1], evals[3][1] * publics[1], evals[3][2] * publics[1]];
    signal tmp_148[3] <== [tmp_146[0] + tmp_147[0], tmp_146[1] + tmp_147[1], tmp_146[2] + tmp_147[2]];
    signal tmp_149[3] <== [evals[9][0] - tmp_148[0], evals[9][1] - tmp_148[1], evals[9][2] - tmp_148[2]];
    signal tmp_150[3] <== [tmp_143[0] + tmp_149[0], tmp_143[1] + tmp_149[1], tmp_143[2] + tmp_149[2]];
    signal tmp_151[3] <== CMul()(challengeQ, tmp_150);
    signal tmp_152[3] <== [evals[7][0] - evals[9][0], evals[7][1] - evals[9][1], evals[7][2] - evals[9][2]];
    signal tmp_153[3] <== [tmp_151[0] + tmp_152[0], tmp_151[1] + tmp_152[1], tmp_151[2] + tmp_152[2]];
    signal tmp_154[3] <== CMul()(challengeQ, tmp_153);
    signal tmp_155[3] <== CMul()(evals[11], evals[12]);
    signal tmp_156[3] <== [1 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_157[3] <== [0 - tmp_156[0], -tmp_156[1], -tmp_156[2]];
    signal tmp_158[3] <== [tmp_155[0] - tmp_157[0], tmp_155[1] - tmp_157[1], tmp_155[2] - tmp_157[2]];
    signal tmp_159[3] <== [tmp_154[0] + tmp_158[0], tmp_154[1] + tmp_158[1], tmp_154[2] + tmp_158[2]];
    signal tmp_160[3] <== CMul()(challengeQ, tmp_159);
    signal tmp_161[3] <== [1 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_162[3] <== CMul()(evals[0], tmp_161);
    signal tmp_163[3] <== [evals[10][0] - tmp_162[0], evals[10][1] - tmp_162[1], evals[10][2] - tmp_162[2]];
    signal tmp_164[3] <== [tmp_163[0] - evals[11][0], tmp_163[1] - evals[11][1], tmp_163[2] - evals[11][2]];
    signal tmp_165[3] <== [tmp_160[0] + tmp_164[0], tmp_160[1] + tmp_164[1], tmp_160[2] + tmp_164[2]];
    signal tmp_166[3] <== CMul()(challengeQ, tmp_165);
    signal tmp_167[3] <== [airgroupvalues[0][0] - evals[10][0], airgroupvalues[0][1] - evals[10][1], airgroupvalues[0][2] - evals[10][2]];
    signal tmp_168[3] <== CMul()(evals[15], tmp_167);
    signal tmp_169[3] <== [tmp_166[0] + tmp_168[0], tmp_166[1] + tmp_168[1], tmp_166[2] + tmp_168[2]];
    signal tmp_170[3] <== CMul()(challengeQ, tmp_169);
    signal tmp_171[3] <== CMul()(evals[17], challengesStage2[0]);
    signal tmp_172[3] <== CMul()(evals[5], evals[5]);
    signal tmp_173[3] <== CMul()(evals[6], evals[6]);
    signal tmp_174[3] <== [tmp_172[0] + tmp_173[0], tmp_172[1] + tmp_173[1], tmp_172[2] + tmp_173[2]];
    signal tmp_175[3] <== [tmp_171[0] + tmp_174[0], tmp_171[1] + tmp_174[1], tmp_171[2] + tmp_174[2]];
    signal tmp_176[3] <== CMul()(tmp_175, challengesStage2[0]);
    signal tmp_177[3] <== [tmp_176[0] + 1, tmp_176[1], tmp_176[2]];
    signal tmp_87[3] <== [tmp_177[0] + challengesStage2[1][0], tmp_177[1] + challengesStage2[1][1], tmp_177[2] + challengesStage2[1][2]];
    signal tmp_178[3] <== [evals[12][0] - tmp_87[0], evals[12][1] - tmp_87[1], evals[12][2] - tmp_87[2]];
    signal tmp_89[3] <== [tmp_170[0] + tmp_178[0], tmp_170[1] + tmp_178[1], tmp_170[2] + tmp_178[2]];
    tmp_179 <== CMul()(tmp_89, Zh);
}


template parallel VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[23][3];
    signal input publics[8];
    signal input airgroupvalues[1][3];
    signal input airvalues[3][3];
    signal input proofvalues[2][3];
    signal input {binary} enable;

    // zMul stores all the powers of z (which is stored in challengeXi) up to nBits, i.e, [z, z^2, ..., z^nBits]
    signal zMul[22][3];
    for (var i=0; i< 22 ; i++) {
        if(i==0){
            zMul[i] <== CMul()(challengeXi, challengeXi);
        } else {
            zMul[i] <== CMul()(zMul[i-1], zMul[i-1]);
        }
    }

    // Store the vanishing polynomial Zh(x) = x^nBits - 1 evaluated at z
    signal Z[3] <== [zMul[21][0] - 1, zMul[21][1], zMul[21][2]];
    signal Zh[3] <== CInv()(Z);




    // Using the evaluations committed and the challenges,
    // calculate the sum of q_i, i.e, q_0(X) + challenge * q_1(X) + challenge^2 * q_2(X) +  ... + challenge^(l-1) * q_l-1(X) evaluated at z 
    signal tmp_179[3];
    (tmp_179) <== VerifyEvaluationsChunks0()(challengesStage2,challengeQ,challengeXi,evals,publics,airgroupvalues,airvalues,proofvalues,Zh);

    signal xAcc[1][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[0][3]; // Stores the evaluations of Q_i
    signal qAcc[1][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 1; i++) {
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
    enable * (tmp_179[0] - qAcc[0][0]) === 0;
    enable * (tmp_179[1] - qAcc[0][1]) === 0;
    enable * (tmp_179[2] - qAcc[0][2]) === 0;
}

template CalculateFRIPolChunks0() {
    signal input challengesFRI[2][3];
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


    signal output tmp_72[3];
    signal tmp_0[3] <== [mapValues.cm2_0[0] - evals[0][0], mapValues.cm2_0[1] - evals[0][1], mapValues.cm2_0[2] - evals[0][2]];
    signal tmp_1[3] <== CMul()(tmp_0, xDivXSubXi[0]);
    signal tmp_2[3] <== CMul()(challengesFRI[0], tmp_1);
    signal tmp_3[3] <== [mapValues.custom_rom_0_0 - evals[1][0], -evals[1][1], -evals[1][2]];
    signal tmp_4[3] <== CMul()(tmp_3, challengesFRI[1]);
    signal tmp_5[3] <== [mapValues.custom_rom_0_1 - evals[2][0], -evals[2][1], -evals[2][2]];
    signal tmp_6[3] <== [tmp_4[0] + tmp_5[0], tmp_4[1] + tmp_5[1], tmp_4[2] + tmp_5[2]];
    signal tmp_7[3] <== CMul()(tmp_6, challengesFRI[1]);
    signal tmp_8[3] <== [consts[0] - evals[3][0], -evals[3][1], -evals[3][2]];
    signal tmp_9[3] <== [tmp_7[0] + tmp_8[0], tmp_7[1] + tmp_8[1], tmp_7[2] + tmp_8[2]];
    signal tmp_10[3] <== CMul()(tmp_9, challengesFRI[1]);
    signal tmp_11[3] <== [consts[1] - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_12[3] <== [tmp_10[0] + tmp_11[0], tmp_10[1] + tmp_11[1], tmp_10[2] + tmp_11[2]];
    signal tmp_13[3] <== CMul()(tmp_12, challengesFRI[1]);
    signal tmp_14[3] <== [mapValues.cm1_0 - evals[5][0], -evals[5][1], -evals[5][2]];
    signal tmp_15[3] <== [tmp_13[0] + tmp_14[0], tmp_13[1] + tmp_14[1], tmp_13[2] + tmp_14[2]];
    signal tmp_16[3] <== CMul()(tmp_15, challengesFRI[1]);
    signal tmp_17[3] <== [mapValues.cm1_1 - evals[6][0], -evals[6][1], -evals[6][2]];
    signal tmp_18[3] <== [tmp_16[0] + tmp_17[0], tmp_16[1] + tmp_17[1], tmp_16[2] + tmp_17[2]];
    signal tmp_19[3] <== CMul()(tmp_18, challengesFRI[1]);
    signal tmp_20[3] <== [mapValues.cm1_2 - evals[7][0], -evals[7][1], -evals[7][2]];
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_21, challengesFRI[1]);
    signal tmp_23[3] <== [mapValues.cm1_3 - evals[8][0], -evals[8][1], -evals[8][2]];
    signal tmp_24[3] <== [tmp_22[0] + tmp_23[0], tmp_22[1] + tmp_23[1], tmp_22[2] + tmp_23[2]];
    signal tmp_25[3] <== CMul()(tmp_24, challengesFRI[1]);
    signal tmp_26[3] <== [mapValues.cm1_4 - evals[9][0], -evals[9][1], -evals[9][2]];
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_27, challengesFRI[1]);
    signal tmp_29[3] <== [mapValues.cm2_0[0] - evals[10][0], mapValues.cm2_0[1] - evals[10][1], mapValues.cm2_0[2] - evals[10][2]];
    signal tmp_30[3] <== [tmp_28[0] + tmp_29[0], tmp_28[1] + tmp_29[1], tmp_28[2] + tmp_29[2]];
    signal tmp_31[3] <== CMul()(tmp_30, challengesFRI[1]);
    signal tmp_32[3] <== [mapValues.cm2_1[0] - evals[11][0], mapValues.cm2_1[1] - evals[11][1], mapValues.cm2_1[2] - evals[11][2]];
    signal tmp_33[3] <== [tmp_31[0] + tmp_32[0], tmp_31[1] + tmp_32[1], tmp_31[2] + tmp_32[2]];
    signal tmp_34[3] <== CMul()(tmp_33, challengesFRI[1]);
    signal tmp_35[3] <== [mapValues.cm2_2[0] - evals[12][0], mapValues.cm2_2[1] - evals[12][1], mapValues.cm2_2[2] - evals[12][2]];
    signal tmp_36[3] <== [tmp_34[0] + tmp_35[0], tmp_34[1] + tmp_35[1], tmp_34[2] + tmp_35[2]];
    signal tmp_37[3] <== CMul()(tmp_36, challengesFRI[1]);
    signal tmp_38[3] <== [mapValues.cm3_0[0] - evals[13][0], mapValues.cm3_0[1] - evals[13][1], mapValues.cm3_0[2] - evals[13][2]];
    signal tmp_39[3] <== [tmp_37[0] + tmp_38[0], tmp_37[1] + tmp_38[1], tmp_37[2] + tmp_38[2]];
    signal tmp_40[3] <== CMul()(tmp_39, xDivXSubXi[1]);
    signal tmp_41[3] <== [tmp_2[0] + tmp_40[0], tmp_2[1] + tmp_40[1], tmp_2[2] + tmp_40[2]];
    signal tmp_42[3] <== CMul()(challengesFRI[0], tmp_41);
    signal tmp_43[3] <== [consts[0] - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_44[3] <== CMul()(tmp_43, challengesFRI[1]);
    signal tmp_45[3] <== [consts[1] - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_46[3] <== [tmp_44[0] + tmp_45[0], tmp_44[1] + tmp_45[1], tmp_44[2] + tmp_45[2]];
    signal tmp_47[3] <== CMul()(tmp_46, challengesFRI[1]);
    signal tmp_48[3] <== [mapValues.cm1_0 - evals[16][0], -evals[16][1], -evals[16][2]];
    signal tmp_49[3] <== [tmp_47[0] + tmp_48[0], tmp_47[1] + tmp_48[1], tmp_47[2] + tmp_48[2]];
    signal tmp_50[3] <== CMul()(tmp_49, challengesFRI[1]);
    signal tmp_51[3] <== [mapValues.cm1_1 - evals[17][0], -evals[17][1], -evals[17][2]];
    signal tmp_52[3] <== [tmp_50[0] + tmp_51[0], tmp_50[1] + tmp_51[1], tmp_50[2] + tmp_51[2]];
    signal tmp_53[3] <== CMul()(tmp_52, xDivXSubXi[2]);
    signal tmp_54[3] <== [tmp_42[0] + tmp_53[0], tmp_42[1] + tmp_53[1], tmp_42[2] + tmp_53[2]];
    signal tmp_55[3] <== CMul()(challengesFRI[0], tmp_54);
    signal tmp_56[3] <== [consts[0] - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_57[3] <== CMul()(tmp_56, challengesFRI[1]);
    signal tmp_58[3] <== [mapValues.cm1_0 - evals[19][0], -evals[19][1], -evals[19][2]];
    signal tmp_59[3] <== [tmp_57[0] + tmp_58[0], tmp_57[1] + tmp_58[1], tmp_57[2] + tmp_58[2]];
    signal tmp_60[3] <== CMul()(tmp_59, challengesFRI[1]);
    signal tmp_61[3] <== [mapValues.cm1_1 - evals[20][0], -evals[20][1], -evals[20][2]];
    signal tmp_62[3] <== [tmp_60[0] + tmp_61[0], tmp_60[1] + tmp_61[1], tmp_60[2] + tmp_61[2]];
    signal tmp_63[3] <== CMul()(tmp_62, xDivXSubXi[3]);
    signal tmp_64[3] <== [tmp_55[0] + tmp_63[0], tmp_55[1] + tmp_63[1], tmp_55[2] + tmp_63[2]];
    signal tmp_65[3] <== CMul()(challengesFRI[0], tmp_64);
    signal tmp_66[3] <== [consts[0] - evals[21][0], -evals[21][1], -evals[21][2]];
    signal tmp_67[3] <== CMul()(tmp_66, challengesFRI[1]);
    signal tmp_68[3] <== [mapValues.cm1_0 - evals[22][0], -evals[22][1], -evals[22][2]];
    signal tmp_69[3] <== [tmp_67[0] + tmp_68[0], tmp_67[1] + tmp_68[1], tmp_67[2] + tmp_68[2]];
    signal tmp_70[3] <== CMul()(tmp_69, xDivXSubXi[4]);
    tmp_72 <== [tmp_65[0] + tmp_70[0], tmp_65[1] + tmp_70[1], tmp_65[2] + tmp_70[2]];

}


/* 
    Verify that the initial FRI polynomial, which is the lineal combination of the committed polynomials
    during the STARK phases, is built properly
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesFRI[23];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[23][3];
    signal input cm1[5];
    signal input cm2[9];
    signal input cm3[3];
    signal input consts[2];
    signal input custom_rom_0[2];

    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;

    signal xacc[23];
    xacc[0] <== queriesFRI[0]*(7 * roots(23)-7) + 7;
    for (var i=1; i<23; i++) {
        xacc[i] <== xacc[i-1] * ( queriesFRI[i]*(roots(23 - i) - 1) +1);
    }

    signal xDivXSubXi[5][3];

    xDivXSubXi[0] <== CInv()([xacc[22] - 10420286214021487819 * challengeXi[0], - 10420286214021487819 * challengeXi[1], - 10420286214021487819 * challengeXi[2]]);
    xDivXSubXi[1] <== CInv()([xacc[22] - 1 * challengeXi[0], - 1 * challengeXi[1], - 1 * challengeXi[2]]);
    xDivXSubXi[2] <== CInv()([xacc[22] - 8124823329697072476 * challengeXi[0], - 8124823329697072476 * challengeXi[1], - 8124823329697072476 * challengeXi[2]]);
    xDivXSubXi[3] <== CInv()([xacc[22] - 6553637399136210105 * challengeXi[0], - 6553637399136210105 * challengeXi[1], - 6553637399136210105 * challengeXi[2]]);
    xDivXSubXi[4] <== CInv()([xacc[22] - 331116024603048646 * challengeXi[0], - 331116024603048646 * challengeXi[1], - 331116024603048646 * challengeXi[2]]);

    signal tmp_72[3];
    (tmp_72) <== CalculateFRIPolChunks0()(challengesFRI,evals,cm1,cm2,cm3,consts,custom_rom_0,xDivXSubXi);

    signal queryVals[3];
    queryVals[0] <== tmp_72[0];
    queryVals[1] <== tmp_72[1];
    queryVals[2] <== tmp_72[2];

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
    cm2_0 <== [vals2[0],vals2[1] , vals2[2]];
    cm2_1 <== [vals2[3],vals2[4] , vals2[5]];
    cm2_2 <== [vals2[6],vals2[7] , vals2[8]];
    cm3_0 <== [vals3[0],vals3[1] , vals3[2]];
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
    for (var k= 16; k< 32; k++) {
        for (var e=0; e<3; e++) {
            enable * lastIFFT[k][e] === 0;
        }
    }
    
    // The coefficients of lower degree can have any value
    for (var k= 0; k < 16; k++) {
        _ <== lastIFFT[k];
    }
}

template VerifySingleQuery0() {
    signal input {binary} queriesFRI[23];

    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input challengesFRISteps[8][3];
    signal input evals[23][3];
    signal input {binary} enable;

    signal input s0_vals1[5];
    signal input s0_vals1_p[5][1];
    signal input s0_siblings1[10][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[9];
    signal input s0_vals2_p[9][1];
    signal input s0_siblings2[10][12];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[3];
    signal input s0_vals3_p[3][1];
    signal input s0_siblings3[10][12];

    signal input s0_valsC[2];
    signal input s0_valsC_p[2][1];
    signal input s0_siblingsC[10][12];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];

    signal input s0_vals_rom_0[2];
    signal input s0_vals_rom_0_p[2][1];
    signal input root_rom_0[4];
    signal input s0_siblings_rom_0[10][12];
    signal input s0_last_mt_levels_rom_0[16][4];

    signal input s1_vals_p[8][3];
    signal input s1_siblings[8][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[8][3];
    signal input s2_siblings[7][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[8][3];
    signal input s3_siblings[5][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[8][3];
    signal input s4_siblings[4][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals_p[8][3];
    signal input s5_siblings[2][12];
    signal input s5_last_mt_levels[16][4];
    signal input s6_vals_p[8][3];
    signal input s6_siblings[1][12];
    signal input s6_last_mt_levels[16][4];

    signal input finalPol[32][3];

    signal {binary} queriesFRIBits[12][2];
    for(var j = 0; j < 12; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 23) {
                queriesFRIBits[j][k] <== 0;
            } else {
                queriesFRIBits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(1, 5, 4, 10, 2, 8388608)(s0_vals1_p, s0_siblings1, queriesFRIBits, s0_last_mt_levels1, enable);
    VerifyMerkleHashUntilLevel(1, 9, 4, 10, 2, 8388608)(s0_vals2_p, s0_siblings2, queriesFRIBits, s0_last_mt_levels2, enable);

    VerifyMerkleHashUntilLevel(1, 3, 4, 10, 2, 8388608)(s0_vals3_p, s0_siblings3, queriesFRIBits, s0_last_mt_levels3, enable);
    VerifyMerkleHashUntilLevel(1, 2, 4, 10, 2, 8388608)(s0_valsC_p, s0_siblingsC, queriesFRIBits, s0_last_mt_levelsC, enable);

    VerifyMerkleHashUntilLevel(1, 2, 4, 10, 2, 8388608)(s0_vals_rom_0_p, s0_siblings_rom_0, queriesFRIBits, s0_last_mt_levels_rom_0, enable);

    signal {binary} s1_keys_merkle_bits[10][2];
    for(var j = 0; j < 10; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 20) {
                s1_keys_merkle_bits[j][k] <== 0;
            } else {
                s1_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 8, 2, 1048576)(s1_vals_p, s1_siblings, s1_keys_merkle_bits, s1_last_mt_levels, enable);
    signal {binary} s2_keys_merkle_bits[9][2];
    for(var j = 0; j < 9; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 17) {
                s2_keys_merkle_bits[j][k] <== 0;
            } else {
                s2_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 7, 2, 131072)(s2_vals_p, s2_siblings, s2_keys_merkle_bits, s2_last_mt_levels, enable);
    signal {binary} s3_keys_merkle_bits[7][2];
    for(var j = 0; j < 7; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 14) {
                s3_keys_merkle_bits[j][k] <== 0;
            } else {
                s3_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 5, 2, 16384)(s3_vals_p, s3_siblings, s3_keys_merkle_bits, s3_last_mt_levels, enable);
    signal {binary} s4_keys_merkle_bits[6][2];
    for(var j = 0; j < 6; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 11) {
                s4_keys_merkle_bits[j][k] <== 0;
            } else {
                s4_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 4, 2, 2048)(s4_vals_p, s4_siblings, s4_keys_merkle_bits, s4_last_mt_levels, enable);
    signal {binary} s5_keys_merkle_bits[4][2];
    for(var j = 0; j < 4; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 8) {
                s5_keys_merkle_bits[j][k] <== 0;
            } else {
                s5_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 2, 2, 256)(s5_vals_p, s5_siblings, s5_keys_merkle_bits, s5_last_mt_levels, enable);
    signal {binary} s6_keys_merkle_bits[3][2];
    for(var j = 0; j < 3; j++) {
        for(var k = 0; k < 2; k++) {
            if (k + j * 2 >= 5) {
                s6_keys_merkle_bits[j][k] <== 0;
            } else {
                s6_keys_merkle_bits[j][k] <== queriesFRI[j*2 + k];
            }
        }
    }

    VerifyMerkleHashUntilLevel(3, 8, 4, 1, 2, 32)(s6_vals_p, s6_siblings, s6_keys_merkle_bits, s6_last_mt_levels, enable);

    VerifyQuery0(23, 20)(queriesFRI, challengeXi, challengesFRI, evals, s0_vals1, s0_vals2, s0_vals3, s0_valsC, s0_vals_rom_0, s1_vals_p, enable);

    signal {binary} s1_queriesFRI[20];
    for(var i = 0; i < 20; i++) { s1_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 23, 20, 17, 2635249152773512046)(s1_queriesFRI, challengesFRISteps[1], s1_vals_p, s2_vals_p, enable);
    signal {binary} s2_queriesFRI[17];
    for(var i = 0; i < 17; i++) { s2_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 20, 17, 14, 12421013511830570338)(s2_queriesFRI, challengesFRISteps[2], s2_vals_p, s3_vals_p, enable);
    signal {binary} s3_queriesFRI[14];
    for(var i = 0; i < 14; i++) { s3_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 17, 14, 11, 11143297345130450484)(s3_queriesFRI, challengesFRISteps[3], s3_vals_p, s4_vals_p, enable);
    signal {binary} s4_queriesFRI[11];
    for(var i = 0; i < 11; i++) { s4_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 14, 11, 8, 1138102428757299658)(s4_queriesFRI, challengesFRISteps[4], s4_vals_p, s5_vals_p, enable);
    signal {binary} s5_queriesFRI[8];
    for(var i = 0; i < 8; i++) { s5_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 11, 8, 5, 140704680260498080)(s5_queriesFRI, challengesFRISteps[5], s5_vals_p, s6_vals_p, enable);
    signal {binary} s6_queriesFRI[5];
    for(var i = 0; i < 5; i++) { s6_queriesFRI[i] <== queriesFRI[i]; }
    VerifyFRI0(23, 8, 5, 0, 10193707927880991676)(s6_queriesFRI, challengesFRISteps[6], s6_vals_p, finalPol, enable);
}

template parallel VerifyQueriesBatch0(nQueries) {
    signal input {binary} queriesFRI[nQueries][23];

    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input challengesFRISteps[8][3];
    signal input evals[23][3];
    signal input {binary} enable;

    signal input s0_vals1[nQueries][5];
    signal input s0_vals1_p[nQueries][5][1];
    signal input s0_siblings1[nQueries][10][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_vals2[nQueries][9];
    signal input s0_vals2_p[nQueries][9][1];
    signal input s0_siblings2[nQueries][10][12];
    signal input s0_last_mt_levels2[16][4];

    signal input s0_vals3[nQueries][3];
    signal input s0_vals3_p[nQueries][3][1];
    signal input s0_siblings3[nQueries][10][12];

    signal input s0_valsC[nQueries][2];
    signal input s0_valsC_p[nQueries][2][1];
    signal input s0_siblingsC[nQueries][10][12];

    signal input s0_last_mt_levels3[16][4];
    signal input s0_last_mt_levelsC[16][4];

    signal input s0_vals_rom_0[nQueries][2];
    signal input s0_vals_rom_0_p[nQueries][2][1];
    signal input root_rom_0[4];
    signal input s0_siblings_rom_0[nQueries][10][12];
    signal input s0_last_mt_levels_rom_0[16][4];

    signal input s1_vals_p[nQueries][8][3];
    signal input s1_siblings[nQueries][8][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals_p[nQueries][8][3];
    signal input s2_siblings[nQueries][7][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals_p[nQueries][8][3];
    signal input s3_siblings[nQueries][5][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals_p[nQueries][8][3];
    signal input s4_siblings[nQueries][4][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals_p[nQueries][8][3];
    signal input s5_siblings[nQueries][2][12];
    signal input s5_last_mt_levels[16][4];
    signal input s6_vals_p[nQueries][8][3];
    signal input s6_siblings[nQueries][1][12];
    signal input s6_last_mt_levels[16][4];

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
            s5_vals_p[q],
            s5_siblings[q],
            s5_last_mt_levels,
            s6_vals_p[q],
            s6_siblings[q],
            s6_last_mt_levels,
            finalPol
        );
    }
}

template StarkVerifier0() {
    signal input publics[8]; // publics polynomials
    signal input airgroupvalues[1][3]; // airgroupvalue values
    signal input airvalues[3][3]; // air values
    signal input proofvalues[2][3]; // air values
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials

    signal output rootC[4] <== [158645667716459684,15589659765854683434,15390420915795156944,8750232381855243643 ]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[23][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
    signal input s0_vals1[229][5];
    signal input s0_vals2[229][9];
    signal input s0_vals3[229][3];
    signal input s0_valsC[229][2];

    signal input s0_vals_rom_0[229][2];

    // Merkle proofs for each of the evaluations
    signal input s0_siblings1[229][10][12];
    signal input s0_last_mt_levels1[16][4];
    signal input s0_siblings2[229][10][12];
    signal input s0_last_mt_levels2[16][4];
    signal input s0_siblings3[229][10][12];
    signal input s0_last_mt_levels3[16][4];
    signal input s0_siblingsC[229][10][12];
    signal input s0_last_mt_levelsC[16][4];
    signal input s0_siblings_rom_0[229][10][12];
    signal input s0_last_mt_levels_rom_0[16][4];

    // Contains the root of the original polynomial and all the intermediate FRI polynomials except for the last step
    signal input s1_root[4];
    signal input s2_root[4];
    signal input s3_root[4];
    signal input s4_root[4];
    signal input s5_root[4];
    signal input s6_root[4];

    // For each intermediate FRI polynomial and the last one, we store at vals the values needed to check the queries.
    // Given a query r,  the verifier needs b points to check it out, being b = 2^u, where u is the difference between two consecutive step
    // and the sibling paths for each query.
    signal input s1_vals[229][24];
    signal input s1_siblings[229][8][12];
    signal input s1_last_mt_levels[16][4];
    signal input s2_vals[229][24];
    signal input s2_siblings[229][7][12];
    signal input s2_last_mt_levels[16][4];
    signal input s3_vals[229][24];
    signal input s3_siblings[229][5][12];
    signal input s3_last_mt_levels[16][4];
    signal input s4_vals[229][24];
    signal input s4_siblings[229][4][12];
    signal input s4_last_mt_levels[16][4];
    signal input s5_vals[229][24];
    signal input s5_siblings[229][2][12];
    signal input s5_last_mt_levels[16][4];
    signal input s6_vals[229][24];
    signal input s6_siblings[229][1][12];
    signal input s6_last_mt_levels[16][4];

    // Evaluations of the final FRI polynomial over a set of points of size bounded its degree
    signal input finalPol[32][3];

    signal input nonce;

    signal {binary} enabled;
    enabled <== 1;


    signal input globalChallenge[3];


    signal queryVals[229][3];


    signal challengesStage2[2][3];

    signal challengeQ[3];
    signal challengeXi[3];
    signal challengesFRI[2][3];

    // challengesFRISteps contains the random value provided by the verifier at each step of the folding so that 
    // the prover can commit the polynomial.
    // Remember that, when folding, the prover does as follows: f0 = g_0 + X*g_1 + ... + (X^b)*g_b and then the 
    // verifier provides a random X so that the prover can commit it. This value is stored here.
    signal challengesFRISteps[8][3];

    // Challenges from which we derive all the queries
    signal {binary} queriesFRI[229][23];


    ///////////
    // Calculate challenges, challengesFRISteps and queriesFRI
    ///////////

    (challengesStage2,challengeQ,challengeXi,challengesFRI,challengesFRISteps,queriesFRI) <== Transcript0()(globalChallenge,airvalues,root2,root3,evals,s1_root,s2_root,s3_root,s4_root,s5_root,s6_root,finalPol, nonce, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputsC so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
    var s0_vals1_p[229][5][1];
    var s0_vals2_p[229][9][1];
    var s0_vals3_p[229][3][1];
    var s0_valsC_p[229][2][1];
    var s0_vals_rom_0_p[229][2][1];
    var s0_vals_p[229][1][3]; 
    var s1_vals_p[229][8][3]; 
    var s2_vals_p[229][8][3]; 
    var s3_vals_p[229][8][3]; 
    var s4_vals_p[229][8][3]; 
    var s5_vals_p[229][8][3]; 
    var s6_vals_p[229][8][3]; 

    for (var q=0; q<229; q++) {
        // Preprocess vals for the initial FRI polynomial
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
            for(var c=0; c < 8; c++) {
                s6_vals_p[q][c][e] = s6_vals[q][c*3+e];
            }
        }
    }

    signal root_rom_0[4] <== [publics[4], publics[5], publics[6], publics[7]];

    ///////////
    // Verify Merkle roots and FRI constraints per query
    ///////////

    // Batch-size parameters — change QUERIES_BATCH_SIZE and recompile to test different batch sizes.
    // N_FULL_BATCHES and LAST_BATCH_SIZE are derived automatically.
    var N_QUERIES = 229;
    var QUERIES_BATCH_SIZE = 29;
    var N_FULL_BATCHES = N_QUERIES \ QUERIES_BATCH_SIZE;
    var LAST_BATCH_SIZE = N_QUERIES - N_FULL_BATCHES * QUERIES_BATCH_SIZE;
    // Used as signal dimension when LAST_BATCH_SIZE may be 0 (size-0 signals are invalid in Circom).
    var LAST_BATCH_SIZE_SAFE = LAST_BATCH_SIZE > 0 ? LAST_BATCH_SIZE : 1;

// Work buffers for full batches
    signal {binary} batch_work_queriesFRI[N_FULL_BATCHES][QUERIES_BATCH_SIZE][23];

    signal batch_work_s0_vals1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5];
    signal batch_work_s0_vals1_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5][1];
    signal batch_work_s0_siblings1[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][12];
    signal batch_work_s0_vals2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][9];
    signal batch_work_s0_vals2_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][9][1];
    signal batch_work_s0_siblings2[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][12];

    signal batch_work_s0_vals3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][3];
    signal batch_work_s0_vals3_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][3][1];
    signal batch_work_s0_siblings3[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][12];
    signal batch_work_s0_valsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2];
    signal batch_work_s0_valsC_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][1];
    signal batch_work_s0_siblingsC[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][12];

    signal batch_work_s0_vals_rom_0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2];
    signal batch_work_s0_vals_rom_0_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][1];
    signal batch_work_s0_siblings_rom_0[N_FULL_BATCHES][QUERIES_BATCH_SIZE][10][12];

    signal batch_work_s1_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s1_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][12];
    signal batch_work_s2_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s2_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][7][12];
    signal batch_work_s3_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s3_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][5][12];
    signal batch_work_s4_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s4_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][4][12];
    signal batch_work_s5_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s5_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][2][12];
    signal batch_work_s6_vals_p[N_FULL_BATCHES][QUERIES_BATCH_SIZE][8][3];
    signal batch_work_s6_siblings[N_FULL_BATCHES][QUERIES_BATCH_SIZE][1][12];

    // Process full batches with Circom loop
    for (var b = 0; b < N_FULL_BATCHES; b++) {
        var batchStart = b * QUERIES_BATCH_SIZE;

        // Fill work buffers for batch b
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 23; i++) {
                batch_work_queriesFRI[b][q][i] <== queriesFRI[batchStart + q][i];
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 5; i++) {
                batch_work_s0_vals1[b][q][i] <== s0_vals1[batchStart + q][i];
                batch_work_s0_vals1_p[b][q][i][0] <== s0_vals1_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 10; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s0_siblings1[b][q][j][k] <== s0_siblings1[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var i = 0; i < 9; i++) {
                batch_work_s0_vals2[b][q][i] <== s0_vals2[batchStart + q][i];
                batch_work_s0_vals2_p[b][q][i][0] <== s0_vals2_p[batchStart + q][i][0];
            }
            for (var j = 0; j < 10; j++) {
                for (var k = 0; k < 12; k++) {
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
            for (var j = 0; j < 10; j++) {
                for (var k = 0; k < 12; k++) {
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
            for (var j = 0; j < 10; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s0_siblings_rom_0[b][q][j][k] <== s0_siblings_rom_0[batchStart + q][j][k];
                }
            }
        }

        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s1_vals_p[b][q][c][e] <== s1_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 8; j++) {
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
            for (var j = 0; j < 7; j++) {
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
            for (var j = 0; j < 5; j++) {
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
            for (var j = 0; j < 4; j++) {
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
            for (var j = 0; j < 2; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s5_siblings[b][q][j][k] <== s5_siblings[batchStart + q][j][k];
                }
            }
        }
        for (var q = 0; q < QUERIES_BATCH_SIZE; q++) {
            for (var c = 0; c < 8; c++) {
                for (var e = 0; e < 3; e++) {
                    batch_work_s6_vals_p[b][q][c][e] <== s6_vals_p[batchStart + q][c][e];
                }
            }
            for (var j = 0; j < 1; j++) {
                for (var k = 0; k < 12; k++) {
                    batch_work_s6_siblings[b][q][j][k] <== s6_siblings[batchStart + q][j][k];
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
            batch_work_s5_vals_p[b],
            batch_work_s5_siblings[b],
            s5_last_mt_levels,
            batch_work_s6_vals_p[b],
            batch_work_s6_siblings[b],
            s6_last_mt_levels,
            finalPol
        );
    }

// Remainder batch — signal declarations always emitted; call guarded by Circom if (LAST_BATCH_SIZE > 0)
    signal {binary} remainder_queriesFRI[LAST_BATCH_SIZE_SAFE][23];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 23; i++) {
            remainder_queriesFRI[q][i] <== queriesFRI[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
        }
    }

    signal remainder_s0_vals1[LAST_BATCH_SIZE_SAFE][5];
    signal remainder_s0_vals1_p[LAST_BATCH_SIZE_SAFE][5][1];
    signal remainder_s0_siblings1[LAST_BATCH_SIZE_SAFE][10][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 5; i++) {
            remainder_s0_vals1[q][i] <== s0_vals1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals1_p[q][i][0] <== s0_vals1_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 10; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings1[q][j][k] <== s0_siblings1[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s0_vals2[LAST_BATCH_SIZE_SAFE][9];
    signal remainder_s0_vals2_p[LAST_BATCH_SIZE_SAFE][9][1];
    signal remainder_s0_siblings2[LAST_BATCH_SIZE_SAFE][10][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 9; i++) {
            remainder_s0_vals2[q][i] <== s0_vals2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals2_p[q][i][0] <== s0_vals2_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 10; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings2[q][j][k] <== s0_siblings2[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s0_vals3[LAST_BATCH_SIZE_SAFE][3];
    signal remainder_s0_vals3_p[LAST_BATCH_SIZE_SAFE][3][1];
    signal remainder_s0_siblings3[LAST_BATCH_SIZE_SAFE][10][12];
    signal remainder_s0_valsC[LAST_BATCH_SIZE_SAFE][2];
    signal remainder_s0_valsC_p[LAST_BATCH_SIZE_SAFE][2][1];
    signal remainder_s0_siblingsC[LAST_BATCH_SIZE_SAFE][10][12];

    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 3; i++) {
            remainder_s0_vals3[q][i] <== s0_vals3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals3_p[q][i][0] <== s0_vals3_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var i = 0; i < 2; i++) {
            remainder_s0_valsC[q][i] <== s0_valsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_valsC_p[q][i][0] <== s0_valsC_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 10; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings3[q][j][k] <== s0_siblings3[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
                remainder_s0_siblingsC[q][j][k] <== s0_siblingsC[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s0_vals_rom_0[LAST_BATCH_SIZE_SAFE][2];
    signal remainder_s0_vals_rom_0_p[LAST_BATCH_SIZE_SAFE][2][1];
    signal remainder_s0_siblings_rom_0[LAST_BATCH_SIZE_SAFE][10][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var i = 0; i < 2; i++) {
            remainder_s0_vals_rom_0[q][i] <== s0_vals_rom_0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i];
            remainder_s0_vals_rom_0_p[q][i][0] <== s0_vals_rom_0_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][i][0];
        }
        for (var j = 0; j < 10; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s0_siblings_rom_0[q][j][k] <== s0_siblings_rom_0[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }

    signal remainder_s1_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s1_siblings[LAST_BATCH_SIZE_SAFE][8][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s1_vals_p[q][c][e] <== s1_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 8; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s1_siblings[q][j][k] <== s1_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s2_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s2_siblings[LAST_BATCH_SIZE_SAFE][7][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s2_vals_p[q][c][e] <== s2_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 7; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s2_siblings[q][j][k] <== s2_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s3_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s3_siblings[LAST_BATCH_SIZE_SAFE][5][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s3_vals_p[q][c][e] <== s3_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 5; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s3_siblings[q][j][k] <== s3_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s4_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s4_siblings[LAST_BATCH_SIZE_SAFE][4][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s4_vals_p[q][c][e] <== s4_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 4; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s4_siblings[q][j][k] <== s4_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s5_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s5_siblings[LAST_BATCH_SIZE_SAFE][2][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s5_vals_p[q][c][e] <== s5_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s5_siblings[q][j][k] <== s5_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
            }
        }
    }
    signal remainder_s6_vals_p[LAST_BATCH_SIZE_SAFE][8][3];
    signal remainder_s6_siblings[LAST_BATCH_SIZE_SAFE][1][12];
    for (var q = 0; q < LAST_BATCH_SIZE; q++) {
        for (var c = 0; c < 8; c++) {
            for (var e = 0; e < 3; e++) {
                remainder_s6_vals_p[q][c][e] <== s6_vals_p[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][c][e];
            }
        }
        for (var j = 0; j < 1; j++) {
            for (var k = 0; k < 12; k++) {
                remainder_s6_siblings[q][j][k] <== s6_siblings[N_FULL_BATCHES * QUERIES_BATCH_SIZE + q][j][k];
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
        remainder_s5_vals_p,
        remainder_s5_siblings,
        s5_last_mt_levels,
        remainder_s6_vals_p,
        remainder_s6_siblings,
        s6_last_mt_levels,
        finalPol
    );
    }

    ///////////
    // Check constraints polynomial in the evaluation point
    ///////////

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, airgroupvalues, airvalues, proofvalues, enabled);


    VerifyMerkleRoot(2, 4, 8388608)(s0_last_mt_levels1, root1, enabled);
    VerifyMerkleRoot(2, 4, 8388608)(s0_last_mt_levels2, root2, enabled);

    VerifyMerkleRoot(2, 4, 8388608)(s0_last_mt_levels3, root3, enabled);

    VerifyMerkleRoot(2, 4, 8388608)(s0_last_mt_levelsC, rootC, enabled);

    VerifyMerkleRoot(2, 4, 8388608)(s0_last_mt_levels_rom_0, root_rom_0, enabled);

    VerifyMerkleRoot(2, 4, 1048576)(s1_last_mt_levels, s1_root, enabled);
    VerifyMerkleRoot(2, 4, 131072)(s2_last_mt_levels, s2_root, enabled);
    VerifyMerkleRoot(2, 4, 16384)(s3_last_mt_levels, s3_root, enabled);
    VerifyMerkleRoot(2, 4, 2048)(s4_last_mt_levels, s4_root, enabled);
    VerifyMerkleRoot(2, 4, 256)(s5_last_mt_levels, s5_root, enabled);
    VerifyMerkleRoot(2, 4, 32)(s6_last_mt_levels, s6_root, enabled);

    ///////////
    // Verify Merkle roots for optimized last levels (shared by all queries)
    ///////////

    VerifyFinalPol0()(finalPol, enabled);
}

