#include <alt_bn128.hpp>
#include "pilfflonk_steps.hpp"

void PilFflonkSteps::step3_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_86 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_87 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_88 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_89 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_90 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_91 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_92 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_74 = E.fr.mul(params.challenges[0], tmp_89);
     AltBn128::FrElement tmp_75 = E.fr.add(tmp_74, tmp_90);
     AltBn128::FrElement tmp_76 = E.fr.mul(params.challenges[0], tmp_75);
     AltBn128::FrElement tmp_77 = E.fr.add(tmp_76, tmp_91);
     AltBn128::FrElement tmp_78 = E.fr.sub(tmp_77, params.challenges[1]);
     AltBn128::FrElement tmp_79 = E.fr.mul(tmp_78, tmp_92);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_79, params.challenges[1]);
     AltBn128::FrElement tmp_93 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_80 = E.fr.mul(tmp_86, params.challenges[0]);
     AltBn128::FrElement tmp_81 = E.fr.add(tmp_80, tmp_87);
     AltBn128::FrElement tmp_82 = E.fr.mul(tmp_81, params.challenges[0]);
     AltBn128::FrElement tmp_83 = E.fr.add(tmp_82, tmp_88);
     AltBn128::FrElement tmp_84 = E.fr.sub(tmp_83, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_85 = E.fr.mul(tmp_84, tmp_93);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_85, params.tmpExp_n[0 + i*7]);
}

void PilFflonkSteps::step3_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_86 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_87 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_88 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_89 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_90 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_91 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_92 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_74 = E.fr.mul(params.challenges[0], tmp_89);
     AltBn128::FrElement tmp_75 = E.fr.add(tmp_74, tmp_90);
     AltBn128::FrElement tmp_76 = E.fr.mul(params.challenges[0], tmp_75);
     AltBn128::FrElement tmp_77 = E.fr.add(tmp_76, tmp_91);
     AltBn128::FrElement tmp_78 = E.fr.sub(tmp_77, params.challenges[1]);
     AltBn128::FrElement tmp_79 = E.fr.mul(tmp_78, tmp_92);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_79, params.challenges[1]);
     AltBn128::FrElement tmp_93 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_80 = E.fr.mul(tmp_86, params.challenges[0]);
     AltBn128::FrElement tmp_81 = E.fr.add(tmp_80, tmp_87);
     AltBn128::FrElement tmp_82 = E.fr.mul(tmp_81, params.challenges[0]);
     AltBn128::FrElement tmp_83 = E.fr.add(tmp_82, tmp_88);
     AltBn128::FrElement tmp_84 = E.fr.sub(tmp_83, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_85 = E.fr.mul(tmp_84, tmp_93);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_85, params.tmpExp_n[0 + i*7]);
}

void PilFflonkSteps::step3_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) {
     AltBn128::FrElement tmp_86 = params.cm1_n[12 + i*15];
     AltBn128::FrElement tmp_87 = params.cm1_n[13 + ((i + 1)%256)*15];
     AltBn128::FrElement tmp_88 = E.fr.mul(params.cm1_n[12 + i*15], params.cm1_n[13 + ((i + 1)%256)*15]);
     AltBn128::FrElement tmp_89 = params.const_n[7 + i * 9];
     AltBn128::FrElement tmp_90 = params.const_n[8 + i * 9];
     AltBn128::FrElement tmp_91 = params.cm1_n[14 + i*15];
     AltBn128::FrElement tmp_92 = params.const_n[6 + i * 9];
     AltBn128::FrElement tmp_74 = E.fr.mul(params.challenges[0], tmp_89);
     AltBn128::FrElement tmp_75 = E.fr.add(tmp_74, tmp_90);
     AltBn128::FrElement tmp_76 = E.fr.mul(params.challenges[0], tmp_75);
     AltBn128::FrElement tmp_77 = E.fr.add(tmp_76, tmp_91);
     AltBn128::FrElement tmp_78 = E.fr.sub(tmp_77, params.challenges[1]);
     AltBn128::FrElement tmp_79 = E.fr.mul(tmp_78, tmp_92);
     params.tmpExp_n[0 + i*7] = E.fr.add(tmp_79, params.challenges[1]);
     AltBn128::FrElement tmp_93 = params.cm1_n[11 + i*15];
     AltBn128::FrElement tmp_80 = E.fr.mul(tmp_86, params.challenges[0]);
     AltBn128::FrElement tmp_81 = E.fr.add(tmp_80, tmp_87);
     AltBn128::FrElement tmp_82 = E.fr.mul(tmp_81, params.challenges[0]);
     AltBn128::FrElement tmp_83 = E.fr.add(tmp_82, tmp_88);
     AltBn128::FrElement tmp_84 = E.fr.sub(tmp_83, params.tmpExp_n[0 + i*7]);
     AltBn128::FrElement tmp_85 = E.fr.mul(tmp_84, tmp_93);
     params.cm3_n[3 + i*4] = E.fr.add(tmp_85, params.tmpExp_n[0 + i*7]);
}
