#ifndef PLONK_PROVER_GPU_CUH
#define PLONK_PROVER_GPU_CUH

#include <string>
#include <map>
#include <thread>
#include "snark_proof.hpp"
#include "binfile_utils.hpp"
#include <gmp.h>
#include "fft.hpp"
#include "zkey_plonk.hpp"
#include "polynomial/polynomial.hpp"
#include "polynomial/evaluations.hpp"
#include <nlohmann/json.hpp>
#include "keccak_256_transcript.hpp"
#include "wtns_utils.hpp"
#include "zkey.hpp"

using json = nlohmann::json;
using namespace std::chrono;

#define BLINDINGFACTORSLENGTH_PLONK_GPU 11
// #define PLONK_GPU_TIMING  // Uncomment or pass -DPLONK_GPU_TIMING at compile time to enable

namespace PlonkGPU {

    template<typename Engine>
    class PlonkProverGPU {
        using FrElement = typename Engine::FrElement;
        using G1Point = typename Engine::G1Point;
        using G1PointAffine = typename Engine::G1PointAffine;

        Engine &E;
        FFT<typename Engine::Fr> *fft = NULL;

        Zkey::PlonkZkeyHeader *zkey;
        u_int32_t zkeyPower;
        std::string curveName;
        size_t N;
        size_t NBytes;
        size_t NExt;
        size_t NExtBytes;

        FrElement *reservedMemoryPtr;
        uint64_t reservedMemorySize;

        FrElement *precomputedBigBuffer;
        G1PointAffine *PTau;

        u_int64_t lengthNonPrecomputedBigBuffer;
        FrElement *nonPrecomputedBigBuffer;

        u_int32_t *mapBuffersBigBuffer;

        FrElement *buffInternalWitness;
        FrElement *buffWitness;

        Zkey::Addition<Engine> *additionsBuff;
        uint8_t *additionLevels;
        uint8_t maxAdditionLevel;

        std::map<std::string, FrElement *> polPtr;

        std::map<std::string, u_int32_t *> mapBuffers;
        std::map<std::string, FrElement *> buffers;

        std::map <std::string, FrElement> challenges;

        FrElement blindingFactors[BLINDINGFACTORSLENGTH_PLONK_GPU + 1];

        Keccak256Transcript<Engine> *transcript;
        SnarkProof<Engine> *proof;

        std::thread asyncTransferSigma;       // evalConstPols=false: S1,S2,S3 — join before computeZ
        std::thread asyncTransferQ;           // evalConstPols=false: QL-QC — join before computeT
        std::thread asyncComputePI;           // PI(X) — join in computeT before compute_t_evaluations_gpu
        std::thread asyncTransferPolsBatch1;  // zkey batch 1 (QC,S1,S2,S3 coefs → slot 8)
        std::thread asyncTransferPolsBatch2;  // zkey batch 2 (QM,QL,QR,QO coefs → slot 6)
        std::thread asyncEvalNTT;             // evalConstPols=true: H2D + NTT for 9 eval polys
        size_t pinnedSize = 0;
        void* pinnedS = nullptr;
        void* pinnedQ = nullptr;
        void* pinnedPI = nullptr;

        void* d_unifiedBuffer = nullptr;  // Single GPU allocation for all buffers
        uint64_t unifiedBufferSize = 0;

        // alias for specific regions of d_unifiedBuffer
        void* d_staticEvalsBuffer = nullptr;
        void* d_piBuffer = nullptr;  
        void* d_lagBuffer = nullptr;
        void* d_evalsT = nullptr;
        void* d_evalsTz = nullptr;  
        void* d_ptau = nullptr;
        void* d_polCoefA = nullptr;
        void* d_polCoefB = nullptr;
        void* d_polCoefC = nullptr;
        void* d_polCoefZ = nullptr;
        void* d_aux = nullptr;
        void* d_gathered = nullptr;     
        void* d_ratios = nullptr;       
        void* d_zEvals = nullptr;     
        void* d_scanWork = nullptr;
        void* d_t1 = nullptr;
        void* d_t2 = nullptr;
        void* d_t3 = nullptr;
        void* d_evalsS1 = nullptr;
        void* d_evalsS2 = nullptr;
        void* d_evalsS3 = nullptr;
        void* d_evalsL1 = nullptr;
        void* d_evalsQL = nullptr;
        void* d_evalsQR = nullptr;
        void* d_evalsQM = nullptr;
        void* d_evalsQO = nullptr;
        void* d_evalsQC = nullptr;
        void* d_evalsA = nullptr;
        void* d_evalsB = nullptr;
        void* d_evalsC = nullptr;
        void* d_coefQM = nullptr;
        void* d_coefQL = nullptr;
        void* d_coefQR = nullptr;
        void* d_coefQO = nullptr;
        void* d_coefQC = nullptr;
        void* d_coefS1 = nullptr;
        void* d_coefS2 = nullptr;
        void* d_coefS3 = nullptr;
        void* d_witness = nullptr;
        void* d_intWitness = nullptr;
        void* d_mapBuffers = nullptr;
        void* d_omegaBasesN = nullptr;      // omega^(blockIdx*256) for N
        void* d_omegaTidN = nullptr;        // omega^(0..255) for N
        void* d_omegaBasesNExt = nullptr;   // omega_4x^(blockIdx*256) for NExt
        void* d_omegaTidNExt = nullptr;     // omega_4x^(0..255) for NExt
        void* d_blindings = nullptr;
        void* d_zvals = nullptr;           
        void* d_addSignalId1 = nullptr;     
        void* d_addSignalId2 = nullptr;     
        void* d_addFactor1 = nullptr;       
        void* d_addFactor2 = nullptr;       
        void* d_additionLevels = nullptr;   


        void* pTauStream = nullptr;         // Non-blocking CUDA stream to copy pTau to GPU
        void* omegasStream = nullptr;       // Non-blocking CUDA stream to generate precomputed omega tables
        void* evalNTTStream = nullptr;      // Non-blocking CUDA stream for asyncEvalNTT H2D + zero_pad
        void* pinnedD2HStaging = nullptr;   // Pinned staging buffer for async D2H
        size_t pinnedD2HStagingSize = 0;
       
        BinFileUtils::BinFile* fdZkeyPtr = nullptr;

        bool preAllocated = false;
        bool isMyDeviceBuffer = false;
        bool precomputedPinned = false;
        bool evalConstPols = true;  

    public:
        PlonkProverGPU(Engine &E);
        PlonkProverGPU(Engine &E, void* reservedMemoryPtr, uint64_t reservedMemorySize);

        ~PlonkProverGPU();

        void setZkey(BinFileUtils::BinFile *fdZkey);

        u_int32_t getNPublic() const { return zkey->nPublic; }

        void preAllocate(void* unified_buffer_gpu);

        std::tuple <std::vector<uint8_t>, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdZkey, BinFileUtils::BinFile *fdWtns);
        std::tuple <std::vector<uint8_t>, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdZkey, FrElement *wtns, WtnsUtils::Header* wtnsHeader = NULL);

        std::tuple <std::vector<uint8_t>, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdWtns);
        std::tuple <std::vector<uint8_t>, std::vector<uint8_t>> prove(FrElement *wtns, WtnsUtils::Header* wtnsHeader = NULL);

    protected:
        void initialize(void* reservedMemoryPtr, uint64_t reservedMemorySize = 0);

        void removePrecomputedData();

        void calculateAdditions();

        FrElement getWitness(u_int64_t idx);
        
        void round0();

        void round1();

        void round2();

        void round3();

        void round4();

        void round5();

        void computeWirePolynomials();

        void computeZ();

        void computeT();

        void computeR();

        void computeWxi();

        void computeWxiw();

        void applyBlindingCorrection(G1Point &commitment, FrElement *bFactors, u_int32_t nFactors);

        G1Point multiExponentiationGPU_devptr(void* dScalars, size_t npoints);
    };
}

#include "plonk_prover_gpu.c.cuh"

#endif