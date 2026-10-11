#include <alt_bn128.hpp>
#include "pilfflonk_steps.hpp"

void PilFflonkSteps::step2prev_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_12 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_13 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_14 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_15 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_16 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_17 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_18 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_0 = E.fr.mul(params.challenges[0], tmp_15);
     AltBn128::FrElement tmp_1 = E.fr.add(tmp_0, tmp_16);
     AltBn128::FrElement tmp_2 = E.fr.mul(params.challenges[0], tmp_1);
     AltBn128::FrElement tmp_3 = E.fr.add(tmp_2, tmp_17);
     AltBn128::FrElement tmp_4 = E.fr.sub(tmp_3, params.challenges[1]);
     AltBn128::FrElement tmp_5 = E.fr.mul(tmp_4, tmp_18);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_5, params.challenges[1]);
     AltBn128::FrElement tmp_19 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_6 = E.fr.mul(tmp_12, params.challenges[0]);
     AltBn128::FrElement tmp_7 = E.fr.add(tmp_6, tmp_13);
     AltBn128::FrElement tmp_8 = E.fr.mul(tmp_7, params.challenges[0]);
     AltBn128::FrElement tmp_9 = E.fr.add(tmp_8, tmp_14);
     AltBn128::FrElement tmp_10 = E.fr.sub(tmp_9, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_11 = E.fr.mul(tmp_10, tmp_19);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_11, params.tmpExp_n[0 + i*7]);
}

void PilFflonkSteps::step2prev_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_12 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_13 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_14 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_15 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_16 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_17 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_18 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_0 = E.fr.mul(params.challenges[0], tmp_15);
     AltBn128::FrElement tmp_1 = E.fr.add(tmp_0, tmp_16);
     AltBn128::FrElement tmp_2 = E.fr.mul(params.challenges[0], tmp_1);
     AltBn128::FrElement tmp_3 = E.fr.add(tmp_2, tmp_17);
     AltBn128::FrElement tmp_4 = E.fr.sub(tmp_3, params.challenges[1]);
     AltBn128::FrElement tmp_5 = E.fr.mul(tmp_4, tmp_18);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_5, params.challenges[1]);
     AltBn128::FrElement tmp_19 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_6 = E.fr.mul(tmp_12, params.challenges[0]);
     AltBn128::FrElement tmp_7 = E.fr.add(tmp_6, tmp_13);
     AltBn128::FrElement tmp_8 = E.fr.mul(tmp_7, params.challenges[0]);
     AltBn128::FrElement tmp_9 = E.fr.add(tmp_8, tmp_14);
     AltBn128::FrElement tmp_10 = E.fr.sub(tmp_9, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_11 = E.fr.mul(tmp_10, tmp_19);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_11, params.tmpExp_n[0 + i*7]);
}

void PilFflonkSteps::step2prev_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_12 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_13 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_14 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_15 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_16 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_17 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_18 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_0 = E.fr.mul(params.challenges[0], tmp_15);
     AltBn128::FrElement tmp_1 = E.fr.add(tmp_0, tmp_16);
     AltBn128::FrElement tmp_2 = E.fr.mul(params.challenges[0], tmp_1);
     AltBn128::FrElement tmp_3 = E.fr.add(tmp_2, tmp_17);
     AltBn128::FrElement tmp_4 = E.fr.sub(tmp_3, params.challenges[1]);
     AltBn128::FrElement tmp_5 = E.fr.mul(tmp_4, tmp_18);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_5, params.challenges[1]);
     AltBn128::FrElement tmp_19 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_6 = E.fr.mul(tmp_12, params.challenges[0]);
     AltBn128::FrElement tmp_7 = E.fr.add(tmp_6, tmp_13);
     AltBn128::FrElement tmp_8 = E.fr.mul(tmp_7, params.challenges[0]);
     AltBn128::FrElement tmp_9 = E.fr.add(tmp_8, tmp_14);
     AltBn128::FrElement tmp_10 = E.fr.sub(tmp_9, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_11 = E.fr.mul(tmp_10, tmp_19);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_11, params.tmpExp_n[0 + i*7]);
}
