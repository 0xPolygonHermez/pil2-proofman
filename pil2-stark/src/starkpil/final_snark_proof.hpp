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
    
    virtual std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = NULL) = 0;

    virtual uint32_t nPublics() const = 0;
};

class PlonkFinalProver : public IFinalSnarkProver {
    Plonk::PlonkProver<AltBn128::Engine> prover_;
    uint32_t nPublics_;
public:
    PlonkFinalProver(BinFileUtils::BinFile* fdZkey) : prover_(AltBn128::Engine::engine) {
        prover_.setZkey(fdZkey);
        nPublics_ = prover_.getNPublic();
    }

    std::tuple <std::vector<uint8_t>, std::vector<uint8_t>>
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

    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>
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

int getProtocolIdFromBinFile(BinFileUtils::BinFile *fdZkey) {
    if (fdZkey->isDirectRead()) {
        uint32_t protocolId32 = 0;
        fdZkey->readSectionTo(&protocolId32, 1, 0, 4);
        return (int)protocolId32;
    } else {
        return Zkey::getProtocolIdFromZkey(fdZkey);
    }
}

std::unique_ptr<IFinalSnarkProver> initFinalSnarkProver(BinFileUtils::BinFile *fdZkey) {
    int protocolId = getProtocolIdFromBinFile(fdZkey);

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

void genFinalSnarkProof(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark) {
    FinalSnark* finalSnarkProver = (FinalSnark*)proverSnark;

    AltBn128::FrElement *witnessFinal = (AltBn128::FrElement *)circomWitnessFinal; 

    try
    {
        TimerStart(SNARK_PROOF);
        auto [snark_proof, public_bytes] = finalSnarkProver->prover->prove(witnessFinal);
        memcpy(proof, snark_proof.data(), snark_proof.size());
        memcpy(publicsSnark, public_bytes.data(), public_bytes.size());
        TimerStopAndLog(SNARK_PROOF);
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::genProof() got exception in rapid SNARK:" + string(e.what()));
        exitProcess();
    }
}

std::pair<std::string, std::string> snark_proof_to_json(
    uint8_t* proof_bytes,
    size_t proof_size,
    uint8_t* public_bytes,
    size_t public_size,
    int protocol_id
) {
    json proof_json = json::object();
    json publics_json = json::array();
    
    // Parse public inputs (always the same format)
    for (size_t i = 0; i < public_size; i += AltBn128::Fr.bytes()) {
        AltBn128::FrElement pub;
        AltBn128::Fr.fromRprBE(pub, public_bytes + i, AltBn128::Fr.bytes());
        publics_json.push_back(AltBn128::Fr.toString(pub));
    }
    
    // Parse proof based on protocol
    std::vector<std::string> orderedCommitments;
    std::vector<std::string> orderedEvaluations;
    std::string protocol_name;
    
    if (protocol_id == Zkey::PLONK_PROTOCOL_ID) {
        orderedCommitments = {"A", "B", "C", "Z", "T1", "T2", "T3", "Wxi", "Wxiw"};
        orderedEvaluations = {"eval_a", "eval_b", "eval_c", "eval_s1", "eval_s2", "eval_zw"};
        protocol_name = "plonk";
    } else if (protocol_id == Zkey::FFLONK_PROTOCOL_ID) {
        orderedCommitments = {"C1", "C2", "W1", "W2"};
        orderedEvaluations = {"ql", "qr", "qm", "qo", "qc", "s1", "s2", "s3", "a", "b", "c", "z", "zw", "t1w", "t2w", "inv"};
        protocol_name = "fflonk";
    } else {
        throw std::runtime_error("Unknown protocol ID");
    }
    
    // Validate proof size before parsing
    // Each commitment is a G1 point with 2 coordinates (x, y), each coordinate is AltBn128::Fr.bytes()
    // Each evaluation is a single Fr element
    size_t expected_size = (orderedCommitments.size() * 2 + orderedEvaluations.size()) * AltBn128::Fr.bytes();
    if (proof_size < expected_size) {
        throw std::runtime_error("Proof size (" + std::to_string(proof_size) + 
                                 " bytes) is smaller than expected (" + std::to_string(expected_size) + 
                                 " bytes) for " + protocol_name + " protocol");
    }
    
    size_t offset = 0;
    
    // Parse commitments (G1 points - each has x and y coordinates)
    for (const auto& key : orderedCommitments) {
        json point = json::array();
        
        AltBn128::FrElement x;
        AltBn128::Fr.fromRprBE(x, proof_bytes + offset, AltBn128::Fr.bytes());
        point.push_back(AltBn128::Fr.toString(x));
        offset += AltBn128::Fr.bytes();
        
        AltBn128::FrElement y;
        AltBn128::Fr.fromRprBE(y, proof_bytes + offset, AltBn128::Fr.bytes());
        point.push_back(AltBn128::Fr.toString(y));
        offset += AltBn128::Fr.bytes();
        
        point.push_back("1");
        
        proof_json[key] = point;
    }
    
    for (const auto& key : orderedEvaluations) {
        AltBn128::FrElement eval;
        AltBn128::Fr.fromRprBE(eval, proof_bytes + offset, AltBn128::Fr.bytes());
        proof_json[key] = AltBn128::Fr.toString(eval);
        offset += AltBn128::Fr.bytes();
    }
    
    proof_json["protocol"] = protocol_name;
    proof_json["curve"] = "bn128";
    
    return {proof_json.dump(), publics_json.dump()};
}

#endif // FINAL_SNARK_PROOF_HPP
    
