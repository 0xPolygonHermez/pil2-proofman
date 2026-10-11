#include <alt_bn128.hpp>
#include "pilfflonk_steps.hpp"

u_int64_t PilFflonkSteps::getNumConstValues() { return 4; }

void PilFflonkSteps::setConstValues(AltBn128::Engine &E, PilFflonkStepsParams &params) {
     params.constValues[0] = E.fr.set(1);
     AltBn128::FrElement v1;
     E.fr.fromString(v1, "5266228460530200451425464971825753823072228272503274930591399474110020095489");
     params.constValues[1] = v1;
     AltBn128::FrElement v2;
     E.fr.fromString(v2, "7105417016332174850410570896152637638969286250121046049601916147253836355616");
     params.constValues[2] = v2;
     params.constValues[3] = E.fr.set(0);
}