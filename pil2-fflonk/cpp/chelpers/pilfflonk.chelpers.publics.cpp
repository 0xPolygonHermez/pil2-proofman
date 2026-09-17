#include <alt_bn128.hpp>
#include "pilfflonk_steps.hpp"

void PilFflonkSteps::publics_first(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) {
   if (pub == 0) {
     AltBn128::FrElement tmp_0 = params.cm1_n[1 + i*15];
     params.publicInputs[i] = tmp_0;
   }
}

void PilFflonkSteps::publics_i(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) {
   if (pub == 0) {
     AltBn128::FrElement tmp_0 = params.cm1_n[1 + i*15];
     params.publicInputs[i] = tmp_0;
   }
}

void PilFflonkSteps::publics_last(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) {
   if (pub == 0) {
     AltBn128::FrElement tmp_0 = params.cm1_n[1 + i*15];
     params.publicInputs[i] = tmp_0;
   }
}
