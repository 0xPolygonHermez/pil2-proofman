#include "timer.hpp"
#include <nlohmann/json.hpp>
#include "fflonk_prover.hpp"
#include "utils.hpp"
#include "alt_bn128.hpp"

void genFinalSnarkProof(void *pWitnessFinal, std::string zkeyFile, std::string outputDir) {
    TimerStart(PROVER_FINAL_SNARK_PROOF);

    AltBn128::FrElement *witnessFinal = (AltBn128::FrElement *)pWitnessFinal;
    // Save public file
    json publicJson;
    AltBn128::FrElement aux;
    AltBn128::Fr.toMontgomery(aux, witnessFinal[1]);
    publicJson[0] = AltBn128::Fr.toString(aux);
    json2file(publicJson, outputDir + "/final_snark_publics.json");

    TimerStart(PROVER_INIT_FFLONK);

    Fflonk::FflonkProver<AltBn128::Engine>* prover = new Fflonk::FflonkProver<AltBn128::Engine>(AltBn128::Engine::engine);

    std::unique_ptr<BinFileUtils::BinFile> zkey = BinFileUtils::openExisting(zkeyFile, "zkey", 1);
    int protocolId = Zkey::getProtocolIdFromZkey(zkey.get());
    if(protocolId != Zkey::FFLONK_PROTOCOL_ID) {
        zklog.error("Prover::genBatchProof() zkey protocolId has to be Fflonk");
        exitProcess();
    }
    
    prover->setZkey(zkey.get());

    BinFileUtils::BinFile *pZkey = zkey.release();
    assert(zkey.get() == nullptr);
    assert(zkey == nullptr);
    delete pZkey;

    TimerStopAndLog(PROVER_INIT_FFLONK);

    TimerStart(RAPID_SNARK);
    try
    {
        auto [jsonProof, publicSignalsJson] = prover->prove(witnessFinal);        
        json2file(jsonProof, outputDir + "/final_snark_proof.json");
        TimerStopAndLog(RAPID_SNARK);
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
        exitProcess();
    }

    TimerStopAndLog(PROVER_FINAL_SNARK_PROOF);
}
    