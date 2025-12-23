#ifndef FINAL_SNARK_PROOF_HPP
#define FINAL_SNARK_PROOF_HPP
#include "timer.hpp"
#include <nlohmann/json.hpp>
#include "fflonk_prover.hpp"
#include "plonk_prover.hpp"
#include "utils.hpp"
#include "alt_bn128.hpp"
#include "zkey_utils.hpp"

struct IFinalSnarkProver {
    virtual ~IFinalSnarkProver() = default;
    
    virtual std::tuple<nlohmann::json, nlohmann::json, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = NULL) = 0;

    virtual uint32_t nPublics() const = 0;
};

class PlonkFinalProver : public IFinalSnarkProver {
    Plonk::PlonkProver<AltBn128::Engine> prover_;
    uint32_t nPublics_;
public:
    PlonkFinalProver(BinFileUtils::BinFile* fdZkey) : prover_(AltBn128::Engine::engine) {
        prover_.setZkey(fdZkey);
        auto zkeyHeader_ = Zkey::PlonkZkeyHeader::loadPlonkZkeyHeader(fdZkey);
        nPublics_ = zkeyHeader_->nPublic;
    }

    std::tuple <nlohmann::json, nlohmann::json, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = nullptr) override {
        return prover_.prove(witnessFinal, wtnsHeader);
    }

    uint32_t nPublics() const override { return nPublics_; }
};

class FflonkFinalProver : public IFinalSnarkProver {
    Fflonk::FflonkProver<AltBn128::Engine> prover_;
    uint32_t nPublics_;
public:
    FflonkFinalProver(BinFileUtils::BinFile* fdZkey) : prover_(AltBn128::Engine::engine) {
        prover_.setZkey(fdZkey);
        auto zkeyHeader_ = Zkey::FflonkZkeyHeader::loadFflonkZkeyHeader(fdZkey);
        nPublics_ = zkeyHeader_->nPublic;
    }

    std::tuple<nlohmann::json, nlohmann::json, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = nullptr) override {
        return prover_.prove(witnessFinal, wtnsHeader);
    }

    uint32_t nPublics() const override { return nPublics_; }
};

struct FinalSnark {
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    uint64_t protocolId;
    std::unique_ptr<IFinalSnarkProver> prover;
};

std::unique_ptr<IFinalSnarkProver> initFinalSnarkProver(BinFileUtils::BinFile *fdZkey) {
    int protocolId = Zkey::getProtocolIdFromZkey(fdZkey);

    if (protocolId == Zkey::FFLONK_PROTOCOL_ID) {
        TimerStart(PROVER_INIT_FFLONK);
        auto prover = std::make_unique<FflonkFinalProver>(fdZkey);
        TimerStopAndLog(PROVER_INIT_FFLONK);
        return prover;
    }

    if (protocolId == Zkey::PLONK_PROTOCOL_ID) {
        TimerStart(PROVER_INIT_PLONK);
        auto prover = std::make_unique<PlonkFinalProver>(fdZkey);
        TimerStopAndLog(PROVER_INIT_PLONK);
        return prover;
    }

    throw std::runtime_error("Unsupported protocol id");
}

void genFinalSnarkProof(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, std::string outputDir) {
    FinalSnark* finalSnarkProver = (FinalSnark*)proverSnark;

    AltBn128::FrElement *witnessFinal = (AltBn128::FrElement *)circomWitnessFinal;
    // Save public file
    json publicJson = json::array();
    for (size_t i = 0; i < finalSnarkProver->prover->nPublics(); ++i) {
        AltBn128::FrElement aux;
        AltBn128::Fr.toMontgomery(aux, witnessFinal[1 + i]);
        publicJson.push_back(AltBn128::Fr.toString(aux));
    }
    

    try
    {
        TimerStart(SNARK_PROOF);
        auto [jsonProof, publicSignalsJson, snark_proof] = finalSnarkProver->prover->prove(witnessFinal);
        memcpy(proof, snark_proof.data(), snark_proof.size());
        if (!outputDir.empty()) {
            json2file(jsonProof, outputDir + "/final_snark_proof.json");
            json2file(publicJson, outputDir + "/final_snark_publics.json");
        }
        TimerStopAndLog(SNARK_PROOF);
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
        exitProcess();
    }
}
#endif // FINAL_SNARK_PROOF_HPP
    