// final_snark_proof_gpu.cpp - Implementation of GPU SNARK prover functions
// This file is compiled with g++ (not nvcc) and linked into the GPU library

#include "final_snark_proof_gpu.cuh"

// Implementation of functions declared in final_snark_proof_gpu.cuh

int getProtocolIdFromBinFileGPU(BinFileUtils::BinFile *fdZkey) {
    if (fdZkey->isDirectRead()) {
        uint32_t protocolId32 = 0;
        fdZkey->readSectionTo(&protocolId32, 1, 0, 4);
        return (int)protocolId32;
    } else {
        return Zkey::getProtocolIdFromZkey(fdZkey);
    }
}

std::unique_ptr<IFinalSnarkProverGPU> initFinalSnarkProverGPU(BinFileUtils::BinFile *fdZkey) {
    int protocolId = getProtocolIdFromBinFileGPU(fdZkey);

    if (protocolId == Zkey::FFLONK_PROTOCOL_ID) {
        TimerStart(PROVER_INIT_FFLONK_GPU);
        auto prover = std::make_unique<FflonkFinalProverGPU>(fdZkey);
        TimerStopAndLog(PROVER_INIT_FFLONK_GPU);
        return prover;
    }

    if (protocolId == Zkey::PLONK_PROTOCOL_ID) {
        TimerStart(PROVER_INIT_PLONK_GPU);
        auto prover = std::make_unique<PlonkFinalProverGPU>(fdZkey);
        TimerStopAndLog(PROVER_INIT_PLONK_GPU);
        return prover;
    }

    throw std::runtime_error("Unsupported protocol id");
}

void genFinalSnarkProofGPU(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark) {
    FinalSnarkGPU* finalSnarkProver = (FinalSnarkGPU*)proverSnark;

    AltBn128::FrElement *witnessFinal = (AltBn128::FrElement *)circomWitnessFinal; 

    try
    {
        TimerStart(SNARK_PROOF_GPU);
        auto [snark_proof, public_bytes] = finalSnarkProver->prover->prove(witnessFinal);
        memcpy(proof, snark_proof.data(), snark_proof.size());
        memcpy(publicsSnark, public_bytes.data(), public_bytes.size());
        TimerStopAndLog(SNARK_PROOF_GPU);
    }
    catch (std::exception &e)
    {
        zklog.error("Prover::genProofGPU() got exception in rapid SNARK:" + string(e.what()));
        exitProcess();
    }
}

// Wrapper functions callable from starks_api.cu via extern declarations

void *initFinalSnarkProverGPU(char* zkeyFile) {
    auto fdZkey = std::make_unique<BinFileUtils::BinFile>(std::string(zkeyFile), "zkey", 1, /*directRead=*/true);
    uint64_t protocolId = getProtocolIdFromBinFileGPU(fdZkey.get());

    if (protocolId == Zkey::FFLONK_PROTOCOL_ID) {
        // FFLONK protocol requires directRead=false (legacy code)
        auto zkey = BinFileUtils::openExisting(zkeyFile, "zkey", 1);
        BinFileUtils::BinFile *fdZkey = zkey.get();
        auto prover = initFinalSnarkProverGPU(fdZkey);

        FinalSnarkGPU *finalSnark = new FinalSnarkGPU{
            .zkey = std::move(zkey),
            .protocolId = protocolId,
            .prover = std::move(prover)
        };
        return finalSnark;
    }
    auto prover = initFinalSnarkProverGPU(fdZkey.get());
    FinalSnarkGPU *finalSnark = new FinalSnarkGPU{
        .zkey = std::move(fdZkey),
        .protocolId = protocolId,
        .prover = std::move(prover)
    };
    return finalSnark;
}

void freeFinalSnarkProverGPU(void *snark_prover) {
    if (snark_prover) {
        delete static_cast<FinalSnarkGPU *>(snark_prover);
    }
}

void preAllocateFinalSnarkProverGPU(void *snark_prover, void* unified_buffer_gpu) {
    FinalSnarkGPU* finalSnarkProver = (FinalSnarkGPU*)snark_prover;
    finalSnarkProver->prover->preAllocate(unified_buffer_gpu);
}

uint64_t getFinalSnarkProtocolIdGPU(void *snark_prover) {
    FinalSnarkGPU* finalSnarkProver = (FinalSnarkGPU*)snark_prover;
    return finalSnarkProver->protocolId;
}
