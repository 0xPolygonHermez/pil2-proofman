pragma circom 2.1.0;
pragma custom_templates;




include "cmul.circom";
include "cinv.circom";
include "poseidon2.circom";
include "bitify.circom";
include "fft.circom";
include "evalpol.circom";
include "treeselector4.circom";
include "pow.circom";
include "merklehash.circom";


/* 
    Calculate FRI Queries
*/
template calculateFRIQueries0() {
    
    signal input challengeFRIQueries[3];
    signal input nonce;
    signal input {binary} enable;
    signal output {binary} queriesFRI[229][23];

    VerifyPoW(16)(challengeFRIQueries, nonce, enable);

    

    
    signal transcriptHash_friQueries_0[16] <== Poseidon2(4, 16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],nonce,0,0,0,0,0,0,0,0], [0,0,0,0]);
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
    
    signal transcriptHash_friQueries_1[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_0[0],transcriptHash_friQueries_0[1],transcriptHash_friQueries_0[2],transcriptHash_friQueries_0[3]]);
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
    
    signal transcriptHash_friQueries_2[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_1[0],transcriptHash_friQueries_1[1],transcriptHash_friQueries_1[2],transcriptHash_friQueries_1[3]]);
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
    
    signal transcriptHash_friQueries_3[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_2[0],transcriptHash_friQueries_2[1],transcriptHash_friQueries_2[2],transcriptHash_friQueries_2[3]]);
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
    
    signal transcriptHash_friQueries_4[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_3[0],transcriptHash_friQueries_3[1],transcriptHash_friQueries_3[2],transcriptHash_friQueries_3[3]]);
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
    
    signal transcriptHash_friQueries_5[16] <== Poseidon2(4, 16)([0,0,0,0,0,0,0,0,0,0,0,0], [transcriptHash_friQueries_4[0],transcriptHash_friQueries_4[1],transcriptHash_friQueries_4[2],transcriptHash_friQueries_4[3]]);
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
    signal input evals[20][3]; 
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


    
    signal transcriptHash_0[16] <== Poseidon2(4, 16)([globalChallenge[0],globalChallenge[1],globalChallenge[2],0,0,0,0,0,0,0,0,0], [0,0,0,0]);
    challengesStage2[0] <== [transcriptHash_0[0], transcriptHash_0[1], transcriptHash_0[2]];
    challengesStage2[1] <== [transcriptHash_0[3], transcriptHash_0[4], transcriptHash_0[5]];
    for(var i = 6; i < 16; i++){
        _ <== transcriptHash_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_1[16] <== Poseidon2(4, 16)([root2[0],root2[1],root2[2],root2[3],airValues[2][0],airValues[2][1],airValues[2][2],0,0,0,0,0], [transcriptHash_0[0],transcriptHash_0[1],transcriptHash_0[2],transcriptHash_0[3]]);
    challengeQ <== [transcriptHash_1[0], transcriptHash_1[1], transcriptHash_1[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_2[16] <== Poseidon2(4, 16)([root3[0],root3[1],root3[2],root3[3],0,0,0,0,0,0,0,0], [transcriptHash_1[0],transcriptHash_1[1],transcriptHash_1[2],transcriptHash_1[3]]);
    challengeXi <== [transcriptHash_2[0], transcriptHash_2[1], transcriptHash_2[2]];
    
    signal transcriptHash_evals_0[16] <== Poseidon2(4, 16)([evals[0][0],evals[0][1],evals[0][2],evals[1][0],evals[1][1],evals[1][2],evals[2][0],evals[2][1],evals[2][2],evals[3][0],evals[3][1],evals[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_1[16] <== Poseidon2(4, 16)([evals[4][0],evals[4][1],evals[4][2],evals[5][0],evals[5][1],evals[5][2],evals[6][0],evals[6][1],evals[6][2],evals[7][0],evals[7][1],evals[7][2]], [transcriptHash_evals_0[0],transcriptHash_evals_0[1],transcriptHash_evals_0[2],transcriptHash_evals_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_2[16] <== Poseidon2(4, 16)([evals[8][0],evals[8][1],evals[8][2],evals[9][0],evals[9][1],evals[9][2],evals[10][0],evals[10][1],evals[10][2],evals[11][0],evals[11][1],evals[11][2]], [transcriptHash_evals_1[0],transcriptHash_evals_1[1],transcriptHash_evals_1[2],transcriptHash_evals_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_3[16] <== Poseidon2(4, 16)([evals[12][0],evals[12][1],evals[12][2],evals[13][0],evals[13][1],evals[13][2],evals[14][0],evals[14][1],evals[14][2],evals[15][0],evals[15][1],evals[15][2]], [transcriptHash_evals_2[0],transcriptHash_evals_2[1],transcriptHash_evals_2[2],transcriptHash_evals_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_evals_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_evals_4[16] <== Poseidon2(4, 16)([evals[16][0],evals[16][1],evals[16][2],evals[17][0],evals[17][1],evals[17][2],evals[18][0],evals[18][1],evals[18][2],evals[19][0],evals[19][1],evals[19][2]], [transcriptHash_evals_3[0],transcriptHash_evals_3[1],transcriptHash_evals_3[2],transcriptHash_evals_3[3]]);
    evalsHash <== [transcriptHash_evals_4[0], transcriptHash_evals_4[1], transcriptHash_evals_4[2], transcriptHash_evals_4[3]];
    
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_3[16] <== Poseidon2(4, 16)([evalsHash[0],evalsHash[1],evalsHash[2],evalsHash[3],0,0,0,0,0,0,0,0], [transcriptHash_2[0],transcriptHash_2[1],transcriptHash_2[2],transcriptHash_2[3]]);
    challengesFRI[0] <== [transcriptHash_3[0], transcriptHash_3[1], transcriptHash_3[2]];
    challengesFRI[1] <== [transcriptHash_3[3], transcriptHash_3[4], transcriptHash_3[5]];
    challengesFRISteps[0] <== [transcriptHash_3[6], transcriptHash_3[7], transcriptHash_3[8]];
    for(var i = 9; i < 16; i++){
        _ <== transcriptHash_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_4[16] <== Poseidon2(4, 16)([s1_root[0],s1_root[1],s1_root[2],s1_root[3],0,0,0,0,0,0,0,0], [transcriptHash_3[0],transcriptHash_3[1],transcriptHash_3[2],transcriptHash_3[3]]);
    challengesFRISteps[1] <== [transcriptHash_4[0], transcriptHash_4[1], transcriptHash_4[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_5[16] <== Poseidon2(4, 16)([s2_root[0],s2_root[1],s2_root[2],s2_root[3],0,0,0,0,0,0,0,0], [transcriptHash_4[0],transcriptHash_4[1],transcriptHash_4[2],transcriptHash_4[3]]);
    challengesFRISteps[2] <== [transcriptHash_5[0], transcriptHash_5[1], transcriptHash_5[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_6[16] <== Poseidon2(4, 16)([s3_root[0],s3_root[1],s3_root[2],s3_root[3],0,0,0,0,0,0,0,0], [transcriptHash_5[0],transcriptHash_5[1],transcriptHash_5[2],transcriptHash_5[3]]);
    challengesFRISteps[3] <== [transcriptHash_6[0], transcriptHash_6[1], transcriptHash_6[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_7[16] <== Poseidon2(4, 16)([s4_root[0],s4_root[1],s4_root[2],s4_root[3],0,0,0,0,0,0,0,0], [transcriptHash_6[0],transcriptHash_6[1],transcriptHash_6[2],transcriptHash_6[3]]);
    challengesFRISteps[4] <== [transcriptHash_7[0], transcriptHash_7[1], transcriptHash_7[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_7[i]; // Unused transcript values 
    }
    
    signal transcriptHash_8[16] <== Poseidon2(4, 16)([s5_root[0],s5_root[1],s5_root[2],s5_root[3],0,0,0,0,0,0,0,0], [transcriptHash_7[0],transcriptHash_7[1],transcriptHash_7[2],transcriptHash_7[3]]);
    challengesFRISteps[5] <== [transcriptHash_8[0], transcriptHash_8[1], transcriptHash_8[2]];
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_8[i]; // Unused transcript values 
    }
    
    signal transcriptHash_9[16] <== Poseidon2(4, 16)([s6_root[0],s6_root[1],s6_root[2],s6_root[3],0,0,0,0,0,0,0,0], [transcriptHash_8[0],transcriptHash_8[1],transcriptHash_8[2],transcriptHash_8[3]]);
    challengesFRISteps[6] <== [transcriptHash_9[0], transcriptHash_9[1], transcriptHash_9[2]];
    
    signal transcriptHash_lastPolFRI_0[16] <== Poseidon2(4, 16)([finalPol[0][0],finalPol[0][1],finalPol[0][2],finalPol[1][0],finalPol[1][1],finalPol[1][2],finalPol[2][0],finalPol[2][1],finalPol[2][2],finalPol[3][0],finalPol[3][1],finalPol[3][2]], [0,0,0,0]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_0[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_1[16] <== Poseidon2(4, 16)([finalPol[4][0],finalPol[4][1],finalPol[4][2],finalPol[5][0],finalPol[5][1],finalPol[5][2],finalPol[6][0],finalPol[6][1],finalPol[6][2],finalPol[7][0],finalPol[7][1],finalPol[7][2]], [transcriptHash_lastPolFRI_0[0],transcriptHash_lastPolFRI_0[1],transcriptHash_lastPolFRI_0[2],transcriptHash_lastPolFRI_0[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_1[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_2[16] <== Poseidon2(4, 16)([finalPol[8][0],finalPol[8][1],finalPol[8][2],finalPol[9][0],finalPol[9][1],finalPol[9][2],finalPol[10][0],finalPol[10][1],finalPol[10][2],finalPol[11][0],finalPol[11][1],finalPol[11][2]], [transcriptHash_lastPolFRI_1[0],transcriptHash_lastPolFRI_1[1],transcriptHash_lastPolFRI_1[2],transcriptHash_lastPolFRI_1[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_2[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_3[16] <== Poseidon2(4, 16)([finalPol[12][0],finalPol[12][1],finalPol[12][2],finalPol[13][0],finalPol[13][1],finalPol[13][2],finalPol[14][0],finalPol[14][1],finalPol[14][2],finalPol[15][0],finalPol[15][1],finalPol[15][2]], [transcriptHash_lastPolFRI_2[0],transcriptHash_lastPolFRI_2[1],transcriptHash_lastPolFRI_2[2],transcriptHash_lastPolFRI_2[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_3[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_4[16] <== Poseidon2(4, 16)([finalPol[16][0],finalPol[16][1],finalPol[16][2],finalPol[17][0],finalPol[17][1],finalPol[17][2],finalPol[18][0],finalPol[18][1],finalPol[18][2],finalPol[19][0],finalPol[19][1],finalPol[19][2]], [transcriptHash_lastPolFRI_3[0],transcriptHash_lastPolFRI_3[1],transcriptHash_lastPolFRI_3[2],transcriptHash_lastPolFRI_3[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_4[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_5[16] <== Poseidon2(4, 16)([finalPol[20][0],finalPol[20][1],finalPol[20][2],finalPol[21][0],finalPol[21][1],finalPol[21][2],finalPol[22][0],finalPol[22][1],finalPol[22][2],finalPol[23][0],finalPol[23][1],finalPol[23][2]], [transcriptHash_lastPolFRI_4[0],transcriptHash_lastPolFRI_4[1],transcriptHash_lastPolFRI_4[2],transcriptHash_lastPolFRI_4[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_5[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_6[16] <== Poseidon2(4, 16)([finalPol[24][0],finalPol[24][1],finalPol[24][2],finalPol[25][0],finalPol[25][1],finalPol[25][2],finalPol[26][0],finalPol[26][1],finalPol[26][2],finalPol[27][0],finalPol[27][1],finalPol[27][2]], [transcriptHash_lastPolFRI_5[0],transcriptHash_lastPolFRI_5[1],transcriptHash_lastPolFRI_5[2],transcriptHash_lastPolFRI_5[3]]);
    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_lastPolFRI_6[i]; // Unused transcript values 
    }
    
    signal transcriptHash_lastPolFRI_7[16] <== Poseidon2(4, 16)([finalPol[28][0],finalPol[28][1],finalPol[28][2],finalPol[29][0],finalPol[29][1],finalPol[29][2],finalPol[30][0],finalPol[30][1],finalPol[30][2],finalPol[31][0],finalPol[31][1],finalPol[31][2]], [transcriptHash_lastPolFRI_6[0],transcriptHash_lastPolFRI_6[1],transcriptHash_lastPolFRI_6[2],transcriptHash_lastPolFRI_6[3]]);
    lastPolFRIHash <== [transcriptHash_lastPolFRI_7[0], transcriptHash_lastPolFRI_7[1], transcriptHash_lastPolFRI_7[2], transcriptHash_lastPolFRI_7[3]];

    for(var i = 4; i < 16; i++){
        _ <== transcriptHash_9[i]; // Unused transcript values 
    }
    
    signal transcriptHash_10[16] <== Poseidon2(4, 16)([lastPolFRIHash[0],lastPolFRIHash[1],lastPolFRIHash[2],lastPolFRIHash[3],0,0,0,0,0,0,0,0], [transcriptHash_9[0],transcriptHash_9[1],transcriptHash_9[2],transcriptHash_9[3]]);
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

template VerifyEvaluations0() {
    signal input challengesStage2[2][3];
    signal input challengeQ[3];
    signal input challengeXi[3];
    signal input evals[20][3];
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
    signal tmp_65[3] <== [evals[5][0] - publics[1], evals[5][1], evals[5][2]];
    signal tmp_66[3] <== CMul()(evals[3], tmp_65);
    signal tmp_67[3] <== CMul()(challengeQ, tmp_66);
    signal tmp_68[3] <== [evals[6][0] - publics[2], evals[6][1], evals[6][2]];
    signal tmp_69[3] <== CMul()(evals[3], tmp_68);
    signal tmp_70[3] <== [tmp_67[0] + tmp_69[0], tmp_67[1] + tmp_69[1], tmp_67[2] + tmp_69[2]];
    signal tmp_71[3] <== CMul()(challengeQ, tmp_70);
    signal tmp_72[3] <== [evals[6][0] - publics[3], evals[6][1], evals[6][2]];
    signal tmp_73[3] <== CMul()(evals[11], tmp_72);
    signal tmp_74[3] <== [tmp_71[0] + tmp_73[0], tmp_71[1] + tmp_73[1], tmp_71[2] + tmp_73[2]];
    signal tmp_75[3] <== CMul()(challengeQ, tmp_74);
    signal tmp_76[3] <== [evals[13][0] - evals[6][0], evals[13][1] - evals[6][1], evals[13][2] - evals[6][2]];
    signal tmp_77[3] <== [1 - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_78[3] <== CMul()(tmp_76, tmp_77);
    signal tmp_79[3] <== [tmp_75[0] + tmp_78[0], tmp_75[1] + tmp_78[1], tmp_75[2] + tmp_78[2]];
    signal tmp_80[3] <== CMul()(challengeQ, tmp_79);
    signal tmp_81[3] <== [evals[16][0] - evals[14][0], evals[16][1] - evals[14][1], evals[16][2] - evals[14][2]];
    signal tmp_82[3] <== [1 - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_83[3] <== CMul()(tmp_81, tmp_82);
    signal tmp_84[3] <== [tmp_80[0] + tmp_83[0], tmp_80[1] + tmp_83[1], tmp_80[2] + tmp_83[2]];
    signal tmp_85[3] <== CMul()(challengeQ, tmp_84);
    signal tmp_86[3] <== [evals[19][0] - evals[17][0], evals[19][1] - evals[17][1], evals[19][2] - evals[17][2]];
    signal tmp_87[3] <== [1 - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_88[3] <== CMul()(tmp_86, tmp_87);
    signal tmp_89[3] <== [tmp_85[0] + tmp_88[0], tmp_85[1] + tmp_88[1], tmp_85[2] + tmp_88[2]];
    signal tmp_90[3] <== CMul()(challengeQ, tmp_89);
    signal tmp_91 <== publics[0] * proofvalues[0][0];
    signal tmp_92 <== tmp_91 - proofvalues[1][0];
    signal tmp_93[3] <== [tmp_90[0] + tmp_92, tmp_90[1], tmp_90[2]];
    signal tmp_94[3] <== CMul()(challengeQ, tmp_93);
    signal tmp_95 <== 2 * airvalues[0][0];
    signal tmp_96 <== tmp_95 - airvalues[1][0];
    signal tmp_97[3] <== [tmp_94[0] + tmp_96, tmp_94[1], tmp_94[2]];
    signal tmp_98[3] <== CMul()(challengeQ, tmp_97);
    signal tmp_99[3] <== [evals[2][0] + 1, evals[2][1], evals[2][2]];
    signal tmp_100[3] <== [evals[1][0] - tmp_99[0], evals[1][1] - tmp_99[1], evals[1][2] - tmp_99[2]];
    signal tmp_101[3] <== [tmp_98[0] + tmp_100[0], tmp_98[1] + tmp_100[1], tmp_98[2] + tmp_100[2]];
    signal tmp_102[3] <== CMul()(challengeQ, tmp_101);
    signal tmp_103[3] <== CMul()(evals[8], evals[9]);
    signal tmp_104[3] <== [1 - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_105[3] <== [0 - tmp_104[0], -tmp_104[1], -tmp_104[2]];
    signal tmp_106[3] <== [tmp_103[0] - tmp_105[0], tmp_103[1] - tmp_105[1], tmp_103[2] - tmp_105[2]];
    signal tmp_107[3] <== [tmp_102[0] + tmp_106[0], tmp_102[1] + tmp_106[1], tmp_102[2] + tmp_106[2]];
    signal tmp_108[3] <== CMul()(challengeQ, tmp_107);
    signal tmp_109[3] <== [1 - evals[4][0], -evals[4][1], -evals[4][2]];
    signal tmp_110[3] <== CMul()(evals[0], tmp_109);
    signal tmp_111[3] <== [evals[7][0] - tmp_110[0], evals[7][1] - tmp_110[1], evals[7][2] - tmp_110[2]];
    signal tmp_112[3] <== [tmp_111[0] - evals[8][0], tmp_111[1] - evals[8][1], tmp_111[2] - evals[8][2]];
    signal tmp_113[3] <== [tmp_108[0] + tmp_112[0], tmp_108[1] + tmp_112[1], tmp_108[2] + tmp_112[2]];
    signal tmp_114[3] <== CMul()(challengeQ, tmp_113);
    signal tmp_115[3] <== [airgroupvalues[0][0] - evals[7][0], airgroupvalues[0][1] - evals[7][1], airgroupvalues[0][2] - evals[7][2]];
    signal tmp_116[3] <== CMul()(evals[12], tmp_115);
    signal tmp_117[3] <== [tmp_114[0] + tmp_116[0], tmp_114[1] + tmp_116[1], tmp_114[2] + tmp_116[2]];
    signal tmp_118[3] <== CMul()(challengeQ, tmp_117);
    signal tmp_119[3] <== CMul()(evals[14], challengesStage2[0]);
    signal tmp_120[3] <== CMul()(evals[5], evals[5]);
    signal tmp_121[3] <== CMul()(evals[6], evals[6]);
    signal tmp_122[3] <== [tmp_120[0] + tmp_121[0], tmp_120[1] + tmp_121[1], tmp_120[2] + tmp_121[2]];
    signal tmp_123[3] <== [tmp_119[0] + tmp_122[0], tmp_119[1] + tmp_122[1], tmp_119[2] + tmp_122[2]];
    signal tmp_124[3] <== CMul()(tmp_123, challengesStage2[0]);
    signal tmp_125[3] <== [tmp_124[0] + 1, tmp_124[1], tmp_124[2]];
    signal tmp_61[3] <== [tmp_125[0] + challengesStage2[1][0], tmp_125[1] + challengesStage2[1][1], tmp_125[2] + challengesStage2[1][2]];
    signal tmp_126[3] <== [evals[9][0] - tmp_61[0], evals[9][1] - tmp_61[1], evals[9][2] - tmp_61[2]];
    signal tmp_63[3] <== [tmp_118[0] + tmp_126[0], tmp_118[1] + tmp_126[1], tmp_118[2] + tmp_126[2]];
    signal tmp_127[3] <== CMul()(tmp_63, Zh);

    signal xAcc[1][3]; //Stores, at each step, x^i evaluated at z
    signal qStep[0][3]; // Stores the evaluations of Q_i
    signal qAcc[1][3]; // Stores the accumulate sum of Q_i

    // Note: Each Qi has degree < n. qDeg determines the number of polynomials of degree < n needed to define Q
    // Calculate Q(X) = Q1(X) + X^n*Q2(X) + X^(2n)*Q3(X) + ..... X^((qDeg-1)n)*Q(X) evaluated at z 
    for (var i=0; i< 1; i++) {
        if (i==0) {
            xAcc[0] <== [1, 0, 0];
            qAcc[0] <== evals[10+i];
        } else {
            xAcc[i] <== CMul()(xAcc[i-1], zMul[21]);
            qStep[i-1] <== CMul()(xAcc[i], evals[10+i]);
            qAcc[i][0] <== qAcc[i-1][0] + qStep[i-1][0];
            qAcc[i][1] <== qAcc[i-1][1] + qStep[i-1][1];
            qAcc[i][2] <== qAcc[i-1][2] + qStep[i-1][2];
        }
    }

    // Final Verification. Check that Q(X)*Zh(X) = sum of linear combination of q_i, which is stored at tmp_127 
    enable * (tmp_127[0] - qAcc[0][0]) === 0;
    enable * (tmp_127[1] - qAcc[0][1]) === 0;
    enable * (tmp_127[2] - qAcc[0][2]) === 0;
}

/*  Calculate FRI polinomial */
template CalculateFRIPolValue0() {
    signal input {binary} queriesFRI[23];
    signal input challengeXi[3];
    signal input challengesFRI[2][3];
    signal input evals[20][3];
 
    signal input cm1[2];
 
    signal input cm2[9];
    signal input cm3[3];
    signal input consts[2];
    signal input custom_rom_0[2];
    
    signal output queryVals[3];

    // Map the s0_vals so that they are converted either into single vars (if they belong to base field) or arrays of 3 elements (if 
    // they belong to the extended field). 
    component mapValues = MapValues0();
 
    mapValues.vals1 <== cm1;
 
    mapValues.vals2 <== cm2;
    mapValues.vals3 <== cm3;
    mapValues.vals_rom_0 <== custom_rom_0;
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
    signal tmp_20[3] <== [mapValues.cm2_0[0] - evals[7][0], mapValues.cm2_0[1] - evals[7][1], mapValues.cm2_0[2] - evals[7][2]];
    signal tmp_21[3] <== [tmp_19[0] + tmp_20[0], tmp_19[1] + tmp_20[1], tmp_19[2] + tmp_20[2]];
    signal tmp_22[3] <== CMul()(tmp_21, challengesFRI[1]);
    signal tmp_23[3] <== [mapValues.cm2_1[0] - evals[8][0], mapValues.cm2_1[1] - evals[8][1], mapValues.cm2_1[2] - evals[8][2]];
    signal tmp_24[3] <== [tmp_22[0] + tmp_23[0], tmp_22[1] + tmp_23[1], tmp_22[2] + tmp_23[2]];
    signal tmp_25[3] <== CMul()(tmp_24, challengesFRI[1]);
    signal tmp_26[3] <== [mapValues.cm2_2[0] - evals[9][0], mapValues.cm2_2[1] - evals[9][1], mapValues.cm2_2[2] - evals[9][2]];
    signal tmp_27[3] <== [tmp_25[0] + tmp_26[0], tmp_25[1] + tmp_26[1], tmp_25[2] + tmp_26[2]];
    signal tmp_28[3] <== CMul()(tmp_27, challengesFRI[1]);
    signal tmp_29[3] <== [mapValues.cm3_0[0] - evals[10][0], mapValues.cm3_0[1] - evals[10][1], mapValues.cm3_0[2] - evals[10][2]];
    signal tmp_30[3] <== [tmp_28[0] + tmp_29[0], tmp_28[1] + tmp_29[1], tmp_28[2] + tmp_29[2]];
    signal tmp_31[3] <== CMul()(tmp_30, xDivXSubXi[1]);
    signal tmp_32[3] <== [tmp_2[0] + tmp_31[0], tmp_2[1] + tmp_31[1], tmp_2[2] + tmp_31[2]];
    signal tmp_33[3] <== CMul()(challengesFRI[0], tmp_32);
    signal tmp_34[3] <== [consts[0] - evals[11][0], -evals[11][1], -evals[11][2]];
    signal tmp_35[3] <== CMul()(tmp_34, challengesFRI[1]);
    signal tmp_36[3] <== [consts[1] - evals[12][0], -evals[12][1], -evals[12][2]];
    signal tmp_37[3] <== [tmp_35[0] + tmp_36[0], tmp_35[1] + tmp_36[1], tmp_35[2] + tmp_36[2]];
    signal tmp_38[3] <== CMul()(tmp_37, challengesFRI[1]);
    signal tmp_39[3] <== [mapValues.cm1_0 - evals[13][0], -evals[13][1], -evals[13][2]];
    signal tmp_40[3] <== [tmp_38[0] + tmp_39[0], tmp_38[1] + tmp_39[1], tmp_38[2] + tmp_39[2]];
    signal tmp_41[3] <== CMul()(tmp_40, challengesFRI[1]);
    signal tmp_42[3] <== [mapValues.cm1_1 - evals[14][0], -evals[14][1], -evals[14][2]];
    signal tmp_43[3] <== [tmp_41[0] + tmp_42[0], tmp_41[1] + tmp_42[1], tmp_41[2] + tmp_42[2]];
    signal tmp_44[3] <== CMul()(tmp_43, xDivXSubXi[2]);
    signal tmp_45[3] <== [tmp_33[0] + tmp_44[0], tmp_33[1] + tmp_44[1], tmp_33[2] + tmp_44[2]];
    signal tmp_46[3] <== CMul()(challengesFRI[0], tmp_45);
    signal tmp_47[3] <== [consts[0] - evals[15][0], -evals[15][1], -evals[15][2]];
    signal tmp_48[3] <== CMul()(tmp_47, challengesFRI[1]);
    signal tmp_49[3] <== [mapValues.cm1_0 - evals[16][0], -evals[16][1], -evals[16][2]];
    signal tmp_50[3] <== [tmp_48[0] + tmp_49[0], tmp_48[1] + tmp_49[1], tmp_48[2] + tmp_49[2]];
    signal tmp_51[3] <== CMul()(tmp_50, challengesFRI[1]);
    signal tmp_52[3] <== [mapValues.cm1_1 - evals[17][0], -evals[17][1], -evals[17][2]];
    signal tmp_53[3] <== [tmp_51[0] + tmp_52[0], tmp_51[1] + tmp_52[1], tmp_51[2] + tmp_52[2]];
    signal tmp_54[3] <== CMul()(tmp_53, xDivXSubXi[3]);
    signal tmp_55[3] <== [tmp_46[0] + tmp_54[0], tmp_46[1] + tmp_54[1], tmp_46[2] + tmp_54[2]];
    signal tmp_56[3] <== CMul()(challengesFRI[0], tmp_55);
    signal tmp_57[3] <== [consts[0] - evals[18][0], -evals[18][1], -evals[18][2]];
    signal tmp_58[3] <== CMul()(tmp_57, challengesFRI[1]);
    signal tmp_59[3] <== [mapValues.cm1_0 - evals[19][0], -evals[19][1], -evals[19][2]];
    signal tmp_60[3] <== [tmp_58[0] + tmp_59[0], tmp_58[1] + tmp_59[1], tmp_58[2] + tmp_59[2]];
    signal tmp_61[3] <== CMul()(tmp_60, xDivXSubXi[4]);
    signal tmp_63[3] <== [tmp_56[0] + tmp_61[0], tmp_56[1] + tmp_61[1], tmp_56[2] + tmp_61[2]];

    queryVals[0] <== tmp_63[0];
    queryVals[1] <== tmp_63[1];
    queryVals[2] <== tmp_63[2];
}

/* 
    Verify that the initial FRI polynomial, which is the lineal combination of the committed polynomials
    during the STARK phases, is built properly
*/
template VerifyQuery0(currStepBits, nextStepBits) {
    var nextStep = currStepBits - nextStepBits; 
    signal input {binary} queriesFRI[23];
    signal input queryVals[3];
    signal input s1_vals[1 << nextStep][3];
    signal input {binary} enable;
    
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
 
    signal input vals1[2];
 
    signal input vals2[9];
    signal input vals3[3];
    signal input vals_rom_0[2];
    signal output cm1_0;
    signal output cm1_1;
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

template StarkVerifier0() {
    signal input publics[8]; // publics polynomials
    signal input airgroupvalues[1][3]; // airgroupvalue values
    signal input airvalues[3][3]; // air values
    signal input proofvalues[2][3]; // air values
    signal input root1[4]; // Merkle tree root of stage 1
    signal input root2[4]; // Merkle tree root of stage 2
    signal input root3[4]; // Merkle tree root of the evaluations of the quotient Q1 and Q2 polynomials

    signal output rootC[4] <== [15507134881363252830,6492804077567420353,17699491771792986691,14472882950742956038 ]; // Merkle tree root of the evaluations of constant polynomials

    signal input evals[20][3]; // Evaluations of the set polynomials at a challenge value z and gz

    // Leaves values of the merkle tree used to check all the queries
 
    signal input s0_vals1[229][2];
 
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
    // Check constraints polynomial in the evaluation point
    ///////////

 

    VerifyEvaluations0()(challengesStage2, challengeQ, challengeXi, evals, publics, airgroupvalues, airvalues, proofvalues, enabled);

    ///////////
    // Preprocess s_i vals
    ///////////

    // Preprocess the s_i vals given as inputs so that we can use anonymous components.
    // Two different processings are done:
    // For s0_vals, the arrays are transposed so that they fit MerkleHash template
    // For (s_i)_vals, the values are passed all together in a single array of length nVals*3. We convert them to vals[nVals][3]
 
    var s0_vals1_p[229][2][1];
 
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
 
        for (var i = 0; i < 2; i++) {
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
    
    ///////////
    // Verify Merkle Roots
    ///////////

    signal {binary} queriesFRIBits[229][12][2];
    for(var i = 0; i < 229; i++) {
        for(var j = 0; j < 12; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 23) {
                    queriesFRIBits[i][j][k] <== 0;
                } else {
                    queriesFRIBits[i][j][k] <== queriesFRI[i][j*2 + k];
                }
            }
        }
    }

    //Calculate merkle root for s0 vals
 
    for (var q=0; q<229; q++) {
        VerifyMerkleHashUntilLevel(1, 2, 4, 10, 2, 8388608)(s0_vals1_p[q], s0_siblings1[q], queriesFRIBits[q], s0_last_mt_levels1, enabled);
    }
 
    for (var q=0; q<229; q++) {
        VerifyMerkleHashUntilLevel(1, 9, 4, 10, 2, 8388608)(s0_vals2_p[q], s0_siblings2[q], queriesFRIBits[q], s0_last_mt_levels2, enabled);
    }

    for (var q=0; q<229; q++) {
        VerifyMerkleHashUntilLevel(1, 3, 4, 10, 2, 8388608)(s0_vals3_p[q], s0_siblings3[q], queriesFRIBits[q], s0_last_mt_levels3, enabled);
    }

    for (var q=0; q<229; q++) {
        VerifyMerkleHashUntilLevel(1, 2, 4, 10, 2, 8388608)(s0_valsC_p[q], s0_siblingsC[q], queriesFRIBits[q], s0_last_mt_levelsC, enabled);
                                    
    }

    signal root_rom_0[4] <== [publics[4], publics[5], publics[6], publics[7]];
    for (var q=0; q<229; q++) {
        VerifyMerkleHashUntilLevel(1, 2, 4, 10, 2, 8388608)(s0_vals_rom_0_p[q], s0_siblings_rom_0[q], queriesFRIBits[q], s0_last_mt_levels_rom_0, enabled);                                    
    }

    signal {binary} s1_keys_merkle_bits[229][10][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s1 vals

        for(var j = 0; j < 10; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 20) {
                    s1_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s1_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 8, 2, 1048576)(s1_vals_p[q], s1_siblings[q], s1_keys_merkle_bits[q], s1_last_mt_levels, enabled);
    }
    signal {binary} s2_keys_merkle_bits[229][9][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s2 vals

        for(var j = 0; j < 9; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 17) {
                    s2_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s2_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 7, 2, 131072)(s2_vals_p[q], s2_siblings[q], s2_keys_merkle_bits[q], s2_last_mt_levels, enabled);
    }
    signal {binary} s3_keys_merkle_bits[229][7][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s3 vals

        for(var j = 0; j < 7; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 14) {
                    s3_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s3_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 5, 2, 16384)(s3_vals_p[q], s3_siblings[q], s3_keys_merkle_bits[q], s3_last_mt_levels, enabled);
    }
    signal {binary} s4_keys_merkle_bits[229][6][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s4 vals

        for(var j = 0; j < 6; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 11) {
                    s4_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s4_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 4, 2, 2048)(s4_vals_p[q], s4_siblings[q], s4_keys_merkle_bits[q], s4_last_mt_levels, enabled);
    }
    signal {binary} s5_keys_merkle_bits[229][4][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s5 vals

        for(var j = 0; j < 4; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 8) {
                    s5_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s5_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 2, 2, 256)(s5_vals_p[q], s5_siblings[q], s5_keys_merkle_bits[q], s5_last_mt_levels, enabled);
    }
    signal {binary} s6_keys_merkle_bits[229][3][2];
    for (var q=0; q<229; q++) {
        // Calculate merkle root for s6 vals

        for(var j = 0; j < 3; j++) {
            for(var k = 0; k < 2; k++) {
                if (k + j * 2 >= 5) {
                    s6_keys_merkle_bits[q][j][k] <== 0;
                } else {
                    s6_keys_merkle_bits[q][j][k] <== queriesFRI[q][j*2 + k];
                }
            }
        }
        VerifyMerkleHashUntilLevel(3, 8, 4, 1, 2, 32)(s6_vals_p[q], s6_siblings[q], s6_keys_merkle_bits[q], s6_last_mt_levels, enabled);
    }

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
    // Calculate FRI Polinomial
    ///////////
    
    for (var q=0; q<229; q++) {
        // Reconstruct FRI polinomial from evaluations
        queryVals[q] <== CalculateFRIPolValue0()(queriesFRI[q], challengeXi, challengesFRI, evals, s0_vals1[q], s0_vals2[q], s0_vals3[q], s0_valsC[q], s0_vals_rom_0[q]);
    }

    ///////////
    // Verify FRI Polinomial
    ///////////
    signal {binary} s1_queriesFRI[229][20];
    signal {binary} s2_queriesFRI[229][17];
    signal {binary} s3_queriesFRI[229][14];
    signal {binary} s4_queriesFRI[229][11];
    signal {binary} s5_queriesFRI[229][8];
    signal {binary} s6_queriesFRI[229][5];

    for (var q=0; q<229; q++) {
      
        // Verify that the query is properly constructed. This is done by checking that the linear combination of the set of 
        // polynomials committed during the different rounds evaluated at z matches with the commitment of the FRI polynomial
        VerifyQuery0(23, 20)(queriesFRI[q], queryVals[q], s1_vals_p[q], enabled);

        ///////////
        // Verify FRI construction
        ///////////

        // For each folding level we need to check that the polynomial is properly constructed
        // Remember that if the step between polynomials is b = 2^l, the next polynomial p_(i+1) will have degree deg(p_i) / b

        // Check S1
        for(var i = 0; i < 20; i++) { s1_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 23, 20, 17, 2635249152773512046)(s1_queriesFRI[q], challengesFRISteps[1], s1_vals_p[q], s2_vals_p[q], enabled);

        // Check S2
        for(var i = 0; i < 17; i++) { s2_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 20, 17, 14, 12421013511830570338)(s2_queriesFRI[q], challengesFRISteps[2], s2_vals_p[q], s3_vals_p[q], enabled);

        // Check S3
        for(var i = 0; i < 14; i++) { s3_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 17, 14, 11, 11143297345130450484)(s3_queriesFRI[q], challengesFRISteps[3], s3_vals_p[q], s4_vals_p[q], enabled);

        // Check S4
        for(var i = 0; i < 11; i++) { s4_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 14, 11, 8, 1138102428757299658)(s4_queriesFRI[q], challengesFRISteps[4], s4_vals_p[q], s5_vals_p[q], enabled);

        // Check S5
        for(var i = 0; i < 8; i++) { s5_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 11, 8, 5, 140704680260498080)(s5_queriesFRI[q], challengesFRISteps[5], s5_vals_p[q], s6_vals_p[q], enabled);

        // Check S6
        for(var i = 0; i < 5; i++) { s6_queriesFRI[q][i] <== queriesFRI[q][i]; }  
        VerifyFRI0(23, 8, 5, 0, 10193707927880991676)(s6_queriesFRI[q], challengesFRISteps[6], s6_vals_p[q], finalPol, enabled);
    }

    VerifyFinalPol0()(finalPol, enabled);
}
    
