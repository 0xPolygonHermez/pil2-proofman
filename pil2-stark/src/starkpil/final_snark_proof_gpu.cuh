#ifndef FINAL_SNARK_PROOF_GPU_CUH
#define FINAL_SNARK_PROOF_GPU_CUH

#include "alt_bn128.hpp"
#include "timer.hpp"
#include <nlohmann/json.hpp>
#include "fflonk_prover.hpp"
#include "plonk_prover_gpu.cuh"
#include "utils.hpp"
#include "zkey_utils.hpp"

struct IFinalSnarkProverGPU {
    virtual ~IFinalSnarkProverGPU() = default;
    
    virtual std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = NULL) = 0;

    virtual uint32_t nPublics() const = 0;

    virtual void preAllocate(void * unified_buffer_gpu) {}
};

// GPU Prover classes
class PlonkFinalProverGPU : public IFinalSnarkProverGPU {
    PlonkGPU::PlonkProverGPU<AltBn128::Engine> prover_;
    uint32_t nPublics_;
public:
    PlonkFinalProverGPU(BinFileUtils::BinFile* fdZkey) : prover_(AltBn128::Engine::engine) {
        prover_.setZkey(fdZkey);
        nPublics_ = prover_.getNPublic();
    }

    std::tuple <std::vector<uint8_t>, std::vector<uint8_t>>
    prove(AltBn128::FrElement* witnessFinal, WtnsUtils::Header* wtnsHeader = nullptr) override {
        return prover_.prove(witnessFinal, wtnsHeader);
    }

    uint32_t nPublics() const override { return nPublics_; }

    void preAllocate(void * unified_buffer_gpu) override { prover_.preAllocate(unified_buffer_gpu); }
};

class FflonkFinalProverGPU : public IFinalSnarkProverGPU {
    Fflonk::FflonkProver<AltBn128::Engine> prover_;
    uint32_t nPublics_;
public:
    FflonkFinalProverGPU(BinFileUtils::BinFile* fdZkey) : prover_(AltBn128::Engine::engine) {
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

struct FinalSnarkGPU {
    std::unique_ptr<BinFileUtils::BinFile> zkey;
    uint64_t protocolId;
    std::unique_ptr<IFinalSnarkProverGPU> prover;
};

int getProtocolIdFromBinFileGPU(BinFileUtils::BinFile *fdZkey);
std::unique_ptr<IFinalSnarkProverGPU> initFinalSnarkProverGPU(BinFileUtils::BinFile *fdZkey);
void genFinalSnarkProofGPU(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark);
void preAllocateFinalSnarkProverGPU(void *snark_prover, void* unified_buffer_gpu);

#endif // FINAL_SNARK_PROOF_GPU_CUH
