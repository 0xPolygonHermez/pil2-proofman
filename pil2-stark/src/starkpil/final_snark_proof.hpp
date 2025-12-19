#ifndef FINAL_SNARK_PROOF_HPP
#define FINAL_SNARK_PROOF_HPP
#include "timer.hpp"
#include <nlohmann/json.hpp>
#include "fflonk_prover.hpp"
#include "plonk_prover.hpp"
#include "utils.hpp"
#include "alt_bn128.hpp"

void *initFinalSnark(std::string zkeyFile, bool fflonk = true) {
   if (fflonk) {
        TimerStart(PROVER_INIT_FFLONK);

        Fflonk::FflonkProver<AltBn128::Engine>* prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine);

        prover->setZkey(zkeyFile);
        
        TimerStopAndLog(PROVER_INIT_FFLONK);
        return prover;
    } else {
        TimerStart(PROVER_INIT_PLONK);

        Plonk::PlonkProver<AltBn128::Engine>* prover = new Plonk::PlonkProver<AltBn128::Engine>(AltBn128::Engine::engine);
        prover->setZkey(zkeyFile);
        
        TimerStopAndLog(PROVER_INIT_PLONK);
        return prover;
    }
}
void genFinalSnarkProof(void *proverSnark, void *circomWitnessFinal,  std::string outputDir, bool fflonk = true) {
    TimerStart(PROVER_FINAL_SNARK_PROOF);

    AltBn128::FrElement *witnessFinal = (AltBn128::FrElement *)circomWitnessFinal;
    // Save public file
    json publicJson;
    AltBn128::FrElement aux;
    AltBn128::Fr.toMontgomery(aux, witnessFinal[1]);
    publicJson[0] = AltBn128::Fr.toString(aux);
    json2file(publicJson, outputDir + "/final_snark_publics.json");

    if(fflonk) {
        Fflonk::FflonkProver<AltBn128::Engine>* prover = (Fflonk::FflonkProver<AltBn128::Engine>*)proverSnark;
        try
        {
            TimerStart(FFLONK_PROOF);
            auto [jsonProof, publicSignalsJson] = prover->prove(witnessFinal);      
            json2file(jsonProof, outputDir + "/final_snark_proof.json");
            TimerStopAndLog(FFLONK_PROOF);
        }
        catch (std::exception &e)
        {
            zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
    } else {
        Plonk::PlonkProver<AltBn128::Engine>* prover = (Plonk::PlonkProver<AltBn128::Engine>*)proverSnark;
        try
        {
            TimerStart(PLONK_PROOF);
            auto [jsonProof, publicSignalsJson] = prover->prove(witnessFinal);        
            json2file(jsonProof, outputDir + "/final_snark_proof.json");
            TimerStopAndLog(PLONK_PROOF);
        }
        catch (std::exception &e)
        {
            zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
            exitProcess();
        }
    }
    TimerStopAndLog(PROVER_FINAL_SNARK_PROOF);
}
#endif // FINAL_SNARK_PROOF_HPP
    