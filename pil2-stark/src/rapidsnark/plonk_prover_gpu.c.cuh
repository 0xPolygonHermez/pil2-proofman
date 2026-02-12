#include "plonk_prover_gpu.cuh"

#include <array>
#include "curve_utils.hpp"
#include "zkey_plonk.hpp"
#include "wtns_utils.hpp"
#include <sodium.h>
#include "thread_utils.hpp"
#include "polynomial/cpolynomial.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "dump.hpp"
#include "mul_z.hpp"

#define ELPP_NO_DEFAULT_LOG_FILE
#include "logger.hpp"
using namespace CPlusPlusLogging;

// GPU function declarations (implemented in CUDA, linked via extern "C")
extern "C" void msm_bn128_gpu(void* out, const void* points, const void* scalars, size_t npoints, bool montgomery);
extern "C" void ntt_bn128_gpu(void* data, uint32_t lg_n);
extern "C" void intt_bn128_gpu(void* data, uint32_t lg_n);
extern "C" void compute_z_ratios_gpu(
    void* ratioOut, const void* buffA, const void* buffB, const void* buffC,
    const void* dStaticEvals,
    const void* beta, const void* gamma, const void* k1, const void* k2,
    const void* omega, uint64_t domainSize);
extern "C" void compute_pi_gpu(
    void* piOut,
    const void* evalLagrange,
    const void* publicA,
    uint64_t fullN, uint32_t nPublic);
extern "C" void compute_t_evaluations_gpu(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB,
    const void* evalC, const void* evalZ,
    const void* dStaticEvals,
    const void* evalLagrange0,
    const void* piPrecomp,
    const void* blindFactors,
    const void* beta, const void* gamma,
    const void* alpha, const void* alpha2,
    const void* k1, const void* k2,
    const void* omega4x, const void* omega1,
    const void* Z1, const void* Z2, const void* Z3,
    uint64_t domainSize);
extern "C" void alloc_static_eval_buffers_gpu(void** dBuffer, uint64_t buffeSize);
extern "C" void free_static_eval_buffers_gpu(void* dBuffer);
extern "C" void alloc_pinned_buffer_gpu(void** pinnedBuffer, size_t pinnedSize);
extern "C" void free_pinned_buffer_gpu(void* pinnedBuffer);
typedef void (*FileReadFn)(void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx);
extern "C" void start_static_eval_transfer_gpu(
    FileReadFn readFn, void* readCtx,
    void* dBuffer, void* pinnedBuffer, size_t pinnedSize,
    const uint32_t* sectionIds, const uint64_t* byteOffsets, const uint64_t* byteSizes, int numArrays);

namespace PlonkGPU
{
    template <typename Engine>
    void PlonkProverGPU<Engine>::initialize(void *reservedMemoryPtr, uint64_t reservedMemorySize)
    {
        zkey = NULL;
        this->reservedMemoryPtr = (FrElement *)reservedMemoryPtr;
        this->reservedMemorySize = reservedMemorySize;

        curveName = CurveUtils::getCurveNameByEngine();
    }

    template <typename Engine>
    PlonkProverGPU<Engine>::PlonkProverGPU(Engine &_E) : E(_E)
    {
        initialize(NULL);
    }

    template <typename Engine>
    PlonkProverGPU<Engine>::PlonkProverGPU(Engine &_E, void *reservedMemoryPtr, uint64_t reservedMemorySize) : E(_E)
    {
        initialize(reservedMemoryPtr, reservedMemorySize);
    }

    template <typename Engine>
    PlonkProverGPU<Engine>::~PlonkProverGPU()
    {
        this->removePrecomputedData();

        delete transcript;
        delete proof;
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::removePrecomputedData()
    {
        // DELETE RESERVED MEMORY (if necessary)
        delete[] precomputedBigBuffer;
        delete[] mapBuffersBigBuffer;
        delete[] buffInternalWitness;
        delete[] additionsBuff;

        if (zkey != NULL)
        {
            free(zkey->k1);
            free(zkey->k2);
            free(zkey->QM);
            free(zkey->QL);
            free(zkey->QR);
            free(zkey->QO);
            free(zkey->QC);
            free(zkey->S1);
            free(zkey->S2);
            free(zkey->S3);
            free(zkey->X2);
        }

        if (NULL == reservedMemoryPtr)
        {
            delete[] inverses;
            delete[] products;
            delete[] nonPrecomputedBigBuffer;
        }

        free_static_eval_buffers_gpu(d_staticEvalsBuffer);
        free_pinned_buffer_gpu(pinnedQ);
        free_pinned_buffer_gpu(pinnedS);
        d_staticEvalsBuffer = nullptr;
        pinnedQ = nullptr;
        pinnedS = nullptr;

        delete fft;

        mapBuffers.clear();
        
        delete polynomials["QL"];
        delete polynomials["QR"];
        delete polynomials["QM"];
        delete polynomials["QO"];
        delete polynomials["QC"];
        delete polynomials["Sigma1"];
        delete polynomials["Sigma2"];
        delete polynomials["Sigma3"];

        delete evaluations["lagrange"];
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::setZkey(BinFileUtils::BinFile *fdZkey)
    {
        bool dr = fdZkey->isDirectRead();

        if (NULL != zkey)
        {
            removePrecomputedData();
        }

        LOG_TRACE("> Reading zkey file");

        if (dr) {
            zkey = Zkey::PlonkZkeyHeader::loadPlonkZkeyHeaderDirect(fdZkey);
        } else {
            zkey = Zkey::PlonkZkeyHeader::loadPlonkZkeyHeader(fdZkey);

            // Deep-copy header pointer fields so they survive after BinFile is freed
            auto deepCopy = [](void *&ptr, size_t size) {
                void *copy = malloc(size);
                memcpy(copy, ptr, size);
                ptr = copy;
            };
            deepCopy(zkey->k1, zkey->n8r);
            deepCopy(zkey->k2, zkey->n8r);
            deepCopy(zkey->QM, zkey->n8q * 2);
            deepCopy(zkey->QL, zkey->n8q * 2);
            deepCopy(zkey->QR, zkey->n8q * 2);
            deepCopy(zkey->QO, zkey->n8q * 2);
            deepCopy(zkey->QC, zkey->n8q * 2);
            deepCopy(zkey->S1, zkey->n8q * 2);
            deepCopy(zkey->S2, zkey->n8q * 2);
            deepCopy(zkey->S3, zkey->n8q * 2);
            deepCopy(zkey->X2, zkey->n8q * 4);
        }

        if (zkey->protocolId != Zkey::PLONK_PROTOCOL_ID)
        {
            throw std::invalid_argument("zkey file is not plonk");
        }

        LOG_TRACE("> Starting fft");

        fft = new FFT<typename Engine::Fr>(zkey->domainSize * 4);
        zkeyPower = fft->log2(zkey->domainSize);

        mpz_t altBbn128r;
        mpz_init(altBbn128r);
        mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

        if (mpz_cmp(zkey->rPrime, altBbn128r) != 0)
        {
            throw std::invalid_argument("zkey curve not supported");
        }

        mpz_clear(altBbn128r);

        sDomain = zkey->domainSize * sizeof(FrElement);

        ////////////////////////////////////////////////////
        // PRECOMPUTED BIG BUFFER
        ////////////////////////////////////////////////////
        // Precomputed 1 > polynomials buffer
        uint64_t lengthPrecomputedBigBuffer = 0;
        lengthPrecomputedBigBuffer += zkey->domainSize * 1 * 8; // Polynomials QL, QR, QM, QO, QC, Sigma1, Sigma2 & Sigma3
        // Precomputed 2 > evaluations buffer (Sigma & Q evals read from file directly to GPU)
        lengthPrecomputedBigBuffer += zkey->domainSize * 4 * zkey->nPublic; // Evaluations Lagrange1
        // Precomputed 3 > ptau buffer
        lengthPrecomputedBigBuffer += (zkey->domainSize + 6) * sizeof(G1PointAffine) / sizeof(FrElement); // PTau buffer

        precomputedBigBuffer = new FrElement[lengthPrecomputedBigBuffer];

        polPtr["Sigma1"] = &precomputedBigBuffer[0];
        polPtr["Sigma2"] = polPtr["Sigma1"] + zkey->domainSize;
        polPtr["Sigma3"] = polPtr["Sigma2"] + zkey->domainSize;
        polPtr["QL"] = polPtr["Sigma3"] + zkey->domainSize;
        polPtr["QR"] = polPtr["QL"] + zkey->domainSize;
        polPtr["QM"] = polPtr["QR"] + zkey->domainSize;
        polPtr["QO"] = polPtr["QM"] + zkey->domainSize;
        polPtr["QC"] = polPtr["QO"] + zkey->domainSize;

        evalPtr["lagrange"] = polPtr["QC"] + zkey->domainSize;

        PTau = (G1PointAffine *)(evalPtr["lagrange"] + zkey->domainSize * 4 * zkey->nPublic);

        // Read Q selectors polynomials and evaluations
        LOG_TRACE("... Loading QL, QR, QM, QO, & QC polynomial coefficients and evaluations");

        polynomials["QL"] = new Polynomial<Engine>(E, polPtr["QL"], zkey->domainSize);
        polynomials["QR"] = new Polynomial<Engine>(E, polPtr["QR"], zkey->domainSize);
        polynomials["QM"] = new Polynomial<Engine>(E, polPtr["QM"], zkey->domainSize);
        polynomials["QO"] = new Polynomial<Engine>(E, polPtr["QO"], zkey->domainSize);
        polynomials["QC"] = new Polynomial<Engine>(E, polPtr["QC"], zkey->domainSize);

        int nThreads = omp_get_max_threads() / 2;

        if (dr) {
            fdZkey->readSectionToParallel(polynomials["QL"]->coef, Zkey::ZKEY_PL_QL_SECTION, 0, sDomain);
            fdZkey->readSectionToParallel(polynomials["QR"]->coef, Zkey::ZKEY_PL_QR_SECTION, 0, sDomain);
            fdZkey->readSectionToParallel(polynomials["QM"]->coef, Zkey::ZKEY_PL_QM_SECTION, 0, sDomain);
            fdZkey->readSectionToParallel(polynomials["QO"]->coef, Zkey::ZKEY_PL_QO_SECTION, 0, sDomain);
            fdZkey->readSectionToParallel(polynomials["QC"]->coef, Zkey::ZKEY_PL_QC_SECTION, 0, sDomain);
        } else {
            ThreadUtils::parcpy(polynomials["QL"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QL_SECTION),
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["QR"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QR_SECTION),
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["QM"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QM_SECTION),
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["QO"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QO_SECTION),
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["QC"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QC_SECTION),
                                sDomain, nThreads);
        }

        polynomials["QL"]->fixDegree();
        polynomials["QR"]->fixDegree();
        polynomials["QM"]->fixDegree();
        polynomials["QO"]->fixDegree();
        polynomials["QC"]->fixDegree();

        // Q evaluations are read from file directly to GPU in round0 (not loaded to CPU memory)

        // Read Sigma polynomial coefficients and evaluations from zkey file
        LOG_TRACE("... Loading Sigma1, Sigma2 & Sigma3 polynomial coefficients and evaluations");

        polynomials["Sigma1"] = new Polynomial<Engine>(E, polPtr["Sigma1"], zkey->domainSize);
        polynomials["Sigma2"] = new Polynomial<Engine>(E, polPtr["Sigma2"], zkey->domainSize);
        polynomials["Sigma3"] = new Polynomial<Engine>(E, polPtr["Sigma3"], zkey->domainSize);

        if (dr) {
            fdZkey->readSectionToParallel(polynomials["Sigma1"]->coef, Zkey::ZKEY_PL_SIGMA_SECTION, 0, sDomain);
            fdZkey->readSectionToParallel(polynomials["Sigma2"]->coef, Zkey::ZKEY_PL_SIGMA_SECTION, sDomain * 5, sDomain);
            fdZkey->readSectionToParallel(polynomials["Sigma3"]->coef, Zkey::ZKEY_PL_SIGMA_SECTION, sDomain * 10, sDomain);
        } else {
            ThreadUtils::parcpy(polynomials["Sigma1"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION),
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["Sigma2"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize * 5,
                                sDomain, nThreads);
            ThreadUtils::parcpy(polynomials["Sigma3"]->coef,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize * 10,
                                sDomain, nThreads);
        }

        polynomials["Sigma1"]->fixDegree();
        polynomials["Sigma2"]->fixDegree();
        polynomials["Sigma3"]->fixDegree();

        // Sigma evals (S1, S2, S3) are read from file directly to GPU in round0 (not loaded to CPU memory)

        // Read Lagrange polynomials & evaluations from zkey file
        LOG_TRACE("... Loading Lagrange evaluations");
        evaluations["lagrange"] = new Evaluations<Engine>(E, evalPtr["lagrange"], zkey->domainSize * 4 * zkey->nPublic);
        for (uint64_t i = 0; i < zkey->nPublic; i++)
        {
            if (dr) {
                fdZkey->readSectionToParallel(evaluations["lagrange"]->eval + zkey->domainSize * 4 * i,
                                      Zkey::ZKEY_PL_LAGRANGE_SECTION,
                                      sDomain + sDomain * 5 * i,
                                      sDomain * 4);
            } else {
                ThreadUtils::parcpy(evaluations["lagrange"]->eval + zkey->domainSize * 4 * i,
                                    (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_LAGRANGE_SECTION) + zkey->domainSize + zkey->domainSize * 5 * i,
                                    sDomain * 4, nThreads);
            }
        }
        LOG_TRACE("... Loading Powers of Tau evaluations");

        if (dr) {
            fdZkey->readSectionToParallel(this->PTau, Zkey::ZKEY_PL_PTAU_SECTION, 0,
                                  (zkey->domainSize + 6) * sizeof(G1PointAffine));
        } else {
            ThreadUtils::parcpy(this->PTau,
                                (G1PointAffine *)fdZkey->getSectionData(Zkey::ZKEY_PL_PTAU_SECTION),
                                (zkey->domainSize + 6) * sizeof(G1PointAffine), nThreads);
        }

        // Load A, B & C map buffers
        LOG_TRACE("... Loading A, B & C map buffers");

        u_int64_t byteLength = sizeof(u_int32_t) * zkey->nConstraints;

        mapBuffersBigBuffer = new u_int32_t[zkey->nConstraints * 3];

        mapBuffers["A"] = mapBuffersBigBuffer;
        mapBuffers["B"] = mapBuffers["A"] + zkey->nConstraints;
        mapBuffers["C"] = mapBuffers["B"] + zkey->nConstraints;

        buffInternalWitness = new FrElement[zkey->nAdditions];

        LOG_TRACE("··· Loading additions");
        additionsBuff = new Zkey::Addition<Engine>[zkey->nAdditions];
        uint64_t additionsBytes = zkey->nAdditions * sizeof(Zkey::Addition<Engine>);

        if (dr) {
            fdZkey->readSectionToParallel(additionsBuff, Zkey::ZKEY_PL_ADDITIONS_SECTION, 0, additionsBytes);
        } else {
            auto srcAdditions = (Zkey::Addition<Engine> *)fdZkey->getSectionData(Zkey::ZKEY_PL_ADDITIONS_SECTION);
            ThreadUtils::parcpy(additionsBuff, srcAdditions, additionsBytes, nThreads);
        }

        LOG_TRACE("··· Loading map buffers");

        if (dr) {
            fdZkey->readSectionToParallel(mapBuffers["A"], Zkey::ZKEY_PL_A_MAP_SECTION, 0, byteLength);
            fdZkey->readSectionToParallel(mapBuffers["B"], Zkey::ZKEY_PL_B_MAP_SECTION, 0, byteLength);
            fdZkey->readSectionToParallel(mapBuffers["C"], Zkey::ZKEY_PL_C_MAP_SECTION, 0, byteLength);
        } else {
            ThreadUtils::parcpy(mapBuffers["A"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_A_MAP_SECTION),
                                byteLength, nThreads);
            ThreadUtils::parcpy(mapBuffers["B"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_B_MAP_SECTION),
                                byteLength, nThreads);
            ThreadUtils::parcpy(mapBuffers["C"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_C_MAP_SECTION),
                                byteLength, nThreads);
        }

        transcript = new Keccak256Transcript<Engine>(E);
        proof = new SnarkProof<Engine>(E, "plonk");

        if (NULL == this->reservedMemoryPtr)
        {
            inverses = new FrElement[zkey->domainSize];
            products = new FrElement[zkey->domainSize];
        }
        else
        {
            inverses = this->reservedMemoryPtr;
            products = inverses + zkey->domainSize;
        }

        ////////////////////////////////////////////////////
        // NON-PRECOMPUTED BIG BUFFER
        ////////////////////////////////////////////////////
        lengthNonPrecomputedBigBuffer = 0;
        lengthNonPrecomputedBigBuffer += zkey->domainSize * 4 * 4;   // Evaluations A, B, C & Z
        lengthNonPrecomputedBigBuffer += zkey->domainSize * 4 * 4;   // Buffers T, Tz & Polynomials T, Tz
        lengthNonPrecomputedBigBuffer += (zkey->domainSize + 2) * 3; // Polynomials A, B & C
        lengthNonPrecomputedBigBuffer += zkey->domainSize + 3;       // Polynomial Z
        lengthNonPrecomputedBigBuffer += (zkey->domainSize + 1) * 2; // Polynomials T1 & T2
        lengthNonPrecomputedBigBuffer += zkey->domainSize + 6;       // Polynomials T3
        lengthNonPrecomputedBigBuffer += zkey->domainSize * 3;       // Buffers A, B, C
        lengthNonPrecomputedBigBuffer += zkey->domainSize + 6;       // Buffer tmp

        // Memory reuse plan:
        // buffers.T    ← zratioArray, polynomials.R
        // buffers.Tz   ← polynomials.Wxi
        // polynomials.T ← buffers.Z

        buffersLength = zkey->domainSize * 3; // Buffers A, B, C, tmp

        if (NULL == this->reservedMemoryPtr)
        {
            nonPrecomputedBigBuffer = new FrElement[lengthNonPrecomputedBigBuffer];
        }
        else
        {
            if (lengthNonPrecomputedBigBuffer * sizeof(FrElement) > reservedMemorySize)
            {
                std::ostringstream ss;
                ss << "Not enough reserved memory to generate a prove. Increase reserved memory size at least to "
                   << lengthNonPrecomputedBigBuffer * sizeof(FrElement) << " bytes";
                throw std::runtime_error(ss.str());
            }

            nonPrecomputedBigBuffer = this->reservedMemoryPtr;
        }

        //L,A,B,C,Sigma1,Sigma2,Z
        evalPtr["A"] = &nonPrecomputedBigBuffer[0];
        evalPtr["B"] = evalPtr["A"] + zkey->domainSize * 4;
        evalPtr["C"] = evalPtr["B"] + zkey->domainSize * 4;
        evalPtr["Z"] = evalPtr["C"] + zkey->domainSize * 4;

        buffers["T"] = evalPtr["Z"] + zkey->domainSize * 4;
        buffers["Tz"] = buffers["T"] + zkey->domainSize * 4;
        polPtr["T"] = buffers["Tz"] + zkey->domainSize * 4;
        polPtr["Tz"] = polPtr["T"] + zkey->domainSize * 4;

        polPtr["A"] = polPtr["Tz"] + zkey->domainSize * 4;
        polPtr["B"] = polPtr["A"] + zkey->domainSize + 2;
        polPtr["C"] = polPtr["B"] + zkey->domainSize + 2;

        polPtr["Z"] = polPtr["C"] + zkey->domainSize + 2;

        polPtr["T1"] = polPtr["Z"] + zkey->domainSize + 3;
        polPtr["T2"] = polPtr["T1"] + zkey->domainSize + 1;

        polPtr["T3"] = polPtr["T2"] + zkey->domainSize + 1;

        buffers["A"] = polPtr["T3"] + zkey->domainSize  + 1;
        buffers["B"] = buffers["A"] + zkey->domainSize;
        buffers["C"] = buffers["B"] + zkey->domainSize;

        buffers["tmp"] = buffers["C"] + (zkey->domainSize);

        // Reuses
        buffers["zratioArray"] = buffers["T"];
        polPtr["R"] = buffers["T"];

        polPtr["Wxi"] = buffers["Tz"];

        buffers["PI"] = buffers["Tz"];

        buffers["Z"] = polPtr["T"];

        // Pre-allocate GPU + pinned buffers for async static eval transfer (two threads)
        staticEvalsBufferSize = 8 * 4 * (uint64_t)zkey->domainSize * 32;
        pinnedSize = 512ULL * 1024 * 1024;  // 512 MB per thread
        if (pinnedSize > staticEvalsBufferSize) pinnedSize = staticEvalsBufferSize;
        alloc_static_eval_buffers_gpu(&d_staticEvalsBuffer, staticEvalsBufferSize);
        alloc_pinned_buffer_gpu(&pinnedQ, pinnedSize);
        alloc_pinned_buffer_gpu(&pinnedS, pinnedSize);

        // Store file pointer for file-based GPU transfer in round0
        fdZkeyPtr = fdZkey;
    }

    template <typename Engine>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> PlonkProverGPU<Engine>::prove(BinFileUtils::BinFile *fdZkey, BinFileUtils::BinFile *fdWtns)
    {
        this->setZkey(fdZkey);
        return this->prove(fdWtns);
    }

    template <typename Engine>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> PlonkProverGPU<Engine>::prove(BinFileUtils::BinFile *fdZkey, FrElement *buffWitness, WtnsUtils::Header *wtnsHeader)
    {
        this->setZkey(fdZkey);
        return this->prove(buffWitness, wtnsHeader);
    }

    template <typename Engine>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> PlonkProverGPU<Engine>::prove(BinFileUtils::BinFile *fdWtns)
    {
        LOG_TRACE("> Reading witness file header");
        auto wtnsHeader = WtnsUtils::loadHeader(fdWtns);

        // Read witness data
        LOG_TRACE("> Reading witness file data");
        buffWitness = (FrElement *)fdWtns->getSectionData(2);

        return this->prove(buffWitness, wtnsHeader.get());
    }

    template <typename Engine>
    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> PlonkProverGPU<Engine>::prove(FrElement *buffWitness, WtnsUtils::Header *wtnsHeader)
    {
        if (NULL == zkey)
        {
            throw std::runtime_error("Zkey data not set");
        }

        LOG_TRACE("PLONK PROVER GPU STARTED");

        this->buffWitness = buffWitness;

        if (NULL != wtnsHeader)
        {
            if (mpz_cmp(zkey->rPrime, wtnsHeader->prime) != 0)
            {
                throw std::invalid_argument("Curve of the witness does not match the curve of the proving key");
            }

            if (wtnsHeader->nVars != zkey->nVars - zkey->nAdditions)
            {
                std::ostringstream ss;
                ss << "Invalid witness length. Circuit: " << zkey->nVars << ", witness: " << wtnsHeader->nVars << ", "
                   << zkey->nAdditions;
                throw std::invalid_argument(ss.str());
            }
        }

        std::ostringstream ss;
        LOG_TRACE("----------------------------");
        LOG_TRACE("  PLONK PROVE GPU SETTINGS");
        ss.str("");
        ss << "  Curve:         " << curveName;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Circuit power: " << zkeyPower;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Domain size:   " << zkey->domainSize;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Vars:          " << zkey->nVars;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Public vars:   " << zkey->nPublic;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Constraints:   " << zkey->nConstraints;
        LOG_TRACE(ss);
        ss.str("");
        ss << "  Additions:     " << zkey->nAdditions;
        LOG_TRACE(ss);
        LOG_TRACE("----------------------------");

        transcript->reset();
        proof->reset();

        // First element in plonk is not used and can be any value. (But always the same).
        // We set it to zero to go faster in the exponentiations.
        buffWitness[0] = E.fr.zero();

        // Until this point all calculations made are circuit depending and independent from the data, 
        // from here is the proof calculation

        double startTime = omp_get_wtime();

        // ROUND 0. Launch async GPU transfer of static eval arrays
        LOG_TRACE("> ROUND 0 (async static eval transfer)");
        round0();

        LOG_TRACE("> Computing Additions");

        int nThreads = omp_get_max_threads() / 2;
        // Set 0's to buffers["A"], buffers["B"], buffers["C"] & buffers["Z"]
        ThreadUtils::parset(buffers["A"], 0, buffersLength * sizeof(FrElement), nThreads);

        calculateAdditions();

        // START PLONK PROVER PROTOCOL

        // ROUND 1. Compute C1(X) polynomial
        LOG_TRACE("> ROUND 1");
        round1();

        // ROUND 2. Compute C2(X) polynomial
        LOG_TRACE("> ROUND 2");
        round2();

        // ROUND 3. Compute opening evaluations
        LOG_TRACE("> ROUND 3");
        round3();

        // ROUND 4. Compute W(X) polynomial
        LOG_TRACE("> ROUND 4");
        round4();

        // ROUND 5. Compute W'(X) polynomial
        LOG_TRACE("> ROUND 5");
        round5();

        // Prepare public inputs
        std::vector<uint8_t> publicBytes;
        FrElement montgomery;
        for (u_int32_t i = 1; i <= zkey->nPublic; i++)
        {
            E.fr.toMontgomery(montgomery, buffWitness[i]);            
            uint8_t buffer[E.fr.bytes()];
            E.fr.toRprBE(montgomery, buffer, E.fr.bytes());
            publicBytes.insert(publicBytes.end(), buffer, buffer + E.fr.bytes());
        }

        LOG_TRACE("PLONK PROVER GPU FINISHED");

        ss.str("");
        ss << "Execution time: " << omp_get_wtime() - startTime << "\n";
        LOG_TRACE(ss);

        std::vector<string> orderedCommitments = {"A", "B", "C", "Z", "T1", "T2", "T3", "Wxi", "Wxiw"};
        std::vector<string> orderedEvaluations = {"eval_a", "eval_b", "eval_c", "eval_s1", "eval_s2", "eval_zw"};
        return {proof->toBytes(orderedCommitments, orderedEvaluations), publicBytes};
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::calculateAdditions()
    {
        for (u_int32_t i = 0; i < zkey->nAdditions; i++)
        {
            // Get witness value
            FrElement witness1 = getWitness(additionsBuff[i].signalId1);
            FrElement witness2 = getWitness(additionsBuff[i].signalId2);

            // Calculate final result
            witness1 = E.fr.mul(additionsBuff[i].factor1, witness1);
            witness2 = E.fr.mul(additionsBuff[i].factor2, witness2);
            buffInternalWitness[i] = E.fr.add(witness1, witness2);
        }
    }

    template <typename Engine>
    typename Engine::FrElement PlonkProverGPU<Engine>::getWitness(u_int64_t idx)
    {
        u_int32_t diff = zkey->nVars - zkey->nAdditions;
        if (idx < diff)
        {
            return buffWitness[idx];
        }
        else if (idx < zkey->nVars)
        {
            return buffInternalWitness[idx - diff];
        }

        return E.fr.zero();
    }

    // ROUND 0 — Launch async H2D transfer of static eval arrays
    template <typename Engine>
    void PlonkProverGPU<Engine>::round0()
    {
        void* dBuf = d_staticEvalsBuffer;
        size_t pinSize = pinnedSize;
        uint64_t evalSize = sDomain * 4;  // byte size of each eval array

        // Callback that reads from the BinFile
        FileReadFn readFn = [](void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx) {
            static_cast<BinFileUtils::BinFile*>(ctx)->readSectionToParallel(dest, sectionId, offset, len, 8);
        };

        // Sigma arrays (S1,S2,S3) — eval offsets within SIGMA section
        std::array<uint32_t, 3> sSids = { Zkey::ZKEY_PL_SIGMA_SECTION, Zkey::ZKEY_PL_SIGMA_SECTION, Zkey::ZKEY_PL_SIGMA_SECTION };
        std::array<uint64_t, 3> sOffs = { sDomain * 1, sDomain * 6, sDomain * 11 };
        std::array<uint64_t, 3> sSizes = { evalSize, evalSize, evalSize };

        uint64_t sigmaBytes = 3 * evalSize;

        asyncTransferSigma = std::thread([readFn, dBuf, this, pinSize,
                                          sSids, sOffs, sSizes]() {
            start_static_eval_transfer_gpu(readFn, (void*)this->fdZkeyPtr,
                                           dBuf, this->pinnedS, pinSize,
                                           sSids.data(), sOffs.data(), sSizes.data(), 3);
        });

        // Q arrays (QL,QR,QM,QO,QC) — evals start at offset sDomain in each section
        std::array<uint32_t, 5> qSids = {
            Zkey::ZKEY_PL_QL_SECTION, Zkey::ZKEY_PL_QR_SECTION, Zkey::ZKEY_PL_QM_SECTION,
            Zkey::ZKEY_PL_QO_SECTION, Zkey::ZKEY_PL_QC_SECTION };
        std::array<uint64_t, 5> qOffs = { sDomain, sDomain, sDomain, sDomain, sDomain };
        std::array<uint64_t, 5> qSizes = { evalSize, evalSize, evalSize, evalSize, evalSize };

        void* dBufQ = (void*)((uint8_t*)dBuf + sigmaBytes);

        asyncTransferQ = std::thread([readFn, dBufQ, this, pinSize,
                                      qSids, qOffs, qSizes]() {
            start_static_eval_transfer_gpu(readFn, (void*)this->fdZkeyPtr,
                                           dBufQ, this->pinnedQ, pinSize,
                                           qSids.data(), qOffs.data(), qSizes.data(), 5);
        });
    }

    // ROUND 1
    template <typename Engine>
    void PlonkProverGPU<Engine>::round1()
    {
        // STEP 1.1 - Generate random blinding scalars (b_1, ..., b9) ∈ F

        // 0 index not used, set to zero
        for (u_int32_t i = 1; i <= BLINDINGFACTORSLENGTH_PLONK_GPU; i++)
        {
            memset((void *)&(blindingFactors[i].v[0]), 0, sizeof(FrElement));
            randombytes_buf((void *)&(blindingFactors[i].v[0]), sizeof(FrElement) - 1);
        }

        // STEP 1.2 - Compute wire polynomials a(X), b(X) and c(X)
        LOG_TRACE("> Computing A, B, C wire polynomials");
        computeWirePolynomials();

        // STEP 1.3 - Compute [a]_1, [b]_1, [c]_1
        LOG_TRACE("> Computing A, B, C polynomial commitments (GPU)");
        G1Point A = multiExponentiationGPU(polynomials["A"]);
        G1Point B = multiExponentiationGPU(polynomials["B"]);
        G1Point C = multiExponentiationGPU(polynomials["C"]);

        // First output of the prover is ([A]_1, [B]_1, [C]_1)
        proof->addPolynomialCommitment("A", A);
        proof->addPolynomialCommitment("B", B);
        proof->addPolynomialCommitment("C", C);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWirePolynomials()
    {
        // Build A, B and C evaluations buffer from zkey and witness files
        FrElement bFactorsA[2] = {blindingFactors[2], blindingFactors[1]};
        FrElement bFactorsB[2] = {blindingFactors[4], blindingFactors[3]};
        FrElement bFactorsC[2] = {blindingFactors[6], blindingFactors[5]};

        computeWirePolynomial("A", bFactorsA);

        // Launch async PI — buffers["A"] is ready
        {
            uint64_t fullN = 4 * (uint64_t)zkey->domainSize;
            auto piOut = buffers["PI"];
            auto lagrangeEval = evaluations["lagrange"]->eval;
            auto publicA = buffers["A"];
            uint32_t nPub = zkey->nPublic;
            asyncComputePI = std::thread([piOut, lagrangeEval, publicA, fullN, nPub]() {
                compute_pi_gpu((void*)piOut, (const void*)lagrangeEval,
                               (const void*)publicA, fullN, nPub);
            });
        }
        computeWirePolynomial("B", bFactorsB);
        computeWirePolynomial("C", bFactorsC);

        // Check degrees
        if (polynomials["A"]->getDegree() >= zkey->domainSize + 2)
        {
            throw std::runtime_error("A Polynomial is not well calculated");
        }
        if (polynomials["B"]->getDegree() >= zkey->domainSize + 2)
        {
            throw std::runtime_error("B Polynomial is not well calculated");
        }
        if (polynomials["C"]->getDegree() >= zkey->domainSize + 2)
        {
            throw std::runtime_error("C Polynomial is not well calculated");
        }
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWirePolynomial(std::string polName, FrElement blindingFactors[])
    {

        // Compute all witness from signal ids and set them to the polynomial buffers
        auto mapBuffersPol = mapBuffers[polName];
        auto buffersPol = buffers[polName];
#pragma omp parallel for
        for (u_int32_t i = 0; i < zkey->nConstraints; ++i)
        {
            FrElement witness = getWitness(mapBuffersPol[i]);
            E.fr.toMontgomery(buffersPol[i], witness);
        }

        // Create the polynomial
        // and compute the coefficients of the wire polynomials from evaluations using GPU IFFT
        std::ostringstream ss;
        ss << "··· Computing " << polName << " ifft (GPU)";
        LOG_TRACE(ss);
        polynomials[polName] = polynomialFromEvaluationsGPU(buffersPol, polPtr[polName], zkey->domainSize, 2, zkeyPower);

        // Compute the extended evaluations of the wire polynomials using GPU FFT
        ss.str("");
        ss << "··· Computing " << polName << " fft (GPU)";
        LOG_TRACE(ss);
        evaluations[polName] = evaluationsFromPolynomialGPU(evalPtr[polName], *polynomials[polName], zkey->domainSize * 4, zkeyPower + 2);

        polynomials[polName]->blindCoefficients(blindingFactors, 2);
    }

    // ROUND 2
    template <typename Engine>
    void PlonkProverGPU<Engine>::round2()
    {
        // STEP 2.1 - Compute permutation challenge beta and gamma ∈ F
        // Compute permutation challenge beta
        LOG_TRACE("> Computing challenges beta and gamma");
        transcript->reset();

        G1Point Commitment;
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->QM));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->QL));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->QR));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->QO));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->QC));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S1));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S2));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S3));
        transcript->addPolCommitment(Commitment);

        // Add publics to the transcript
        for (u_int32_t i = 0; i < zkey->nPublic; i++)
        {
            transcript->addScalar(buffers["A"][i]);
        }

        // Add A, B, C to the transcript
        transcript->addPolCommitment(proof->getPolynomialCommitment("A"));
        transcript->addPolCommitment(proof->getPolynomialCommitment("B"));
        transcript->addPolCommitment(proof->getPolynomialCommitment("C"));

        challenges["beta"] = transcript->getChallenge();
        std::ostringstream ss;
        ss << "··· challenges.beta: " << E.fr.toString(challenges["beta"]);
        LOG_TRACE(ss);

        // Compute permutation challenge gamma
        transcript->reset();
        transcript->addScalar(challenges["beta"]);
        challenges["gamma"] = transcript->getChallenge();
        ss.str("");
        ss << "··· challenges.gamma: " << E.fr.toString(challenges["gamma"]);
        LOG_TRACE(ss);

        // STEP 2.2 - Compute permutation polynomial z(X)
        LOG_TRACE("> Computing Z polynomial");
        computeZ();

        // The second output of the prover is ([C2]_1)
        LOG_TRACE("> Computing Z polynomial commitment (GPU)");
        G1Point Z = multiExponentiationGPU(polynomials["Z"]);
        proof->addPolynomialCommitment("Z", Z);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeZ()
    {
        FrElement *zratioArray = buffers["zratioArray"];

        auto buffersA = buffers["A"];
        auto buffersB = buffers["B"];
        auto buffersC = buffers["C"];

        // Wait for sigma transfer to complete (S1,S2,S3 needed by compute_z_ratios_gpu)
        if (asyncTransferSigma.joinable()) {
            double t0 = omp_get_wtime();
            asyncTransferSigma.join();
            std::cout<<"[computeZ] sigma join completed, waited: " << omp_get_wtime() - t0 <<endl;
        }

        LOG_TRACE("··· Computing Z evaluations (GPU)");

        // Preloaded constants
        auto beta = challenges["beta"];
        auto gamma = challenges["gamma"];
        auto k1 = *((FrElement *)zkey->k1);
        auto k2 = *((FrElement *)zkey->k2);

        // Primitive root of unity: omega = root(zkeyPower, 1)
        FrElement omega = fft->root(zkeyPower, 1);

        compute_z_ratios_gpu(
            (void*)zratioArray,
            (const void*)buffersA,
            (const void*)buffersB,
            (const void*)buffersC,
            d_staticEvalsBuffer,
            (const void*)&beta,
            (const void*)&gamma,
            (const void*)&k1,
            (const void*)&k2,
            (const void*)&omega,
            zkey->domainSize);

        // Sequential prefix product over ratios to build Z evaluations
        // Z[0] = product of all ratios (should equal 1)
        // Z[i] = product of ratios[0..i-1] for i > 0
        auto buffersZ = buffers["Z"];

        FrElement prev = zratioArray[0];
        for (u_int64_t i = 0; i < zkey->domainSize - 1; i++)
        {
            FrElement cur = zratioArray[i + 1];
            buffersZ[i + 1] = prev;
            prev = E.fr.mul(prev, cur);
        }
        buffersZ[0] = prev;

        if (!E.fr.eq(buffersZ[0], E.fr.one()))
        {
            throw std::runtime_error("Copy constraints does not match");
        }

        // Compute polynomial coefficients z(X) from buffers.Z using GPU IFFT
        LOG_TRACE("··· Computing Z ifft (GPU)");
        polynomials["Z"] = polynomialFromEvaluationsGPU(buffersZ, polPtr["Z"], zkey->domainSize, 3, zkeyPower);

        // Compute extended evaluations of z(X) polynomial using GPU FFT
        LOG_TRACE("··· Computing Z fft (GPU)");
        evaluations["Z"] = evaluationsFromPolynomialGPU(evalPtr["Z"], *polynomials["Z"], zkey->domainSize * 4, zkeyPower + 2);

        // Blind z(X) polynomial coefficients with blinding scalars b
        FrElement bFactors[3] = {blindingFactors[9], blindingFactors[8], blindingFactors[7]};
        polynomials["Z"]->blindCoefficients(bFactors, 3);

        // Check degree
        if (polynomials["Z"]->getDegree() >= zkey->domainSize + 3)
        {
            throw std::runtime_error("Z Polynomial is not well calculated");
        }
    }

    // ROUND 3
    template <typename Engine>
    void PlonkProverGPU<Engine>::round3()
    {
        LOG_TRACE("> Computing challenge alpha");
        // STEP 3.1 - Compute evaluation challenge xi ∈ S
        transcript->reset();
        transcript->addScalar(challenges["beta"]);
        transcript->addScalar(challenges["gamma"]);
        transcript->addPolCommitment(proof->getPolynomialCommitment("Z"));

        challenges["alpha"] = transcript->getChallenge();
        E.fr.square(challenges["alpha2"], challenges["alpha"]);

        std::ostringstream ss;
        ss << "··· challenges.alpha: " << E.fr.toString(challenges["alpha"]);
        LOG_TRACE(ss);

        // Compute quotient polynomial T(X)
        LOG_TRACE("> Computing T polynomial");
        computeT();

        // Compute [T1]_1, [T2]_1, [T3]_1
        LOG_TRACE("> Computing T1, T2 & T3 polynomial commitments (GPU)");
        G1Point T1 = multiExponentiationGPU(polynomials["T1"]);
        G1Point T2 = multiExponentiationGPU(polynomials["T2"]);
        G1Point T3 = multiExponentiationGPU(polynomials["T3"]);

        proof->addPolynomialCommitment("T1", T1);
        proof->addPolynomialCommitment("T2", T2);
        proof->addPolynomialCommitment("T3", T3);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeT()
    {
        LOG_TRACE("··· Computing T evaluations (GPU)");

        auto buffersA = buffers["A"];
        auto buffersT = buffers["T"];
        auto buffersTz = buffers["Tz"];

        auto evaluationsA = evaluations["A"];
        auto evaluationsB = evaluations["B"];
        auto evaluationsC = evaluations["C"];
        auto evaluationsZ = evaluations["Z"];
        auto evaluationsLagrange = evaluations["lagrange"];

        // Preloaded constants
        auto beta = challenges["beta"];
        auto gamma = challenges["gamma"];
        auto alpha = challenges["alpha"];
        auto alpha2 = challenges["alpha2"];
        auto k1 = *((FrElement *)zkey->k1);
        auto k2 = *((FrElement *)zkey->k2);
        auto omega1 = fft->root(zkeyPower, 1);

        // Primitive root of the 4x extended domain
        FrElement omega_4x = fft->root(zkeyPower + 2, 1);

        // Precompute MulZ Z1/Z2/Z3 constants (from mul_z.c.hpp)
        FrElement w2 = fft->root(2, 1);
        FrElement Z1[4], Z2[4], Z3[4];
        Z1[0] = E.fr.zero();
        Z1[1] = E.fr.add(E.fr.set(-1), w2);
        Z1[2] = E.fr.set(-2);
        Z1[3] = E.fr.sub(E.fr.set(-1), w2);
        Z2[0] = E.fr.zero();
        Z2[1] = E.fr.mul(E.fr.set(-2), w2);
        Z2[2] = E.fr.set(4);
        Z2[3] = E.fr.sub(E.fr.zero(), E.fr.mul(E.fr.set(-2), w2));
        Z3[0] = E.fr.zero();
        Z3[1] = E.fr.add(E.fr.set(2), E.fr.mul(E.fr.set(2), w2));
        Z3[2] = E.fr.set(-8);
        Z3[3] = E.fr.sub(E.fr.set(2), E.fr.mul(E.fr.set(2), w2));

        // Wait for async PI computation (launched in computeWirePolynomials)
        if (asyncComputePI.joinable()) {
            double t0 = omp_get_wtime();
            asyncComputePI.join();
            std::cout << "[computeT] PI join completed, waited: " << omp_get_wtime() - t0 << endl;
        }        
        // Wait for async Q eval transfer to complete (QL-QC needed by compute_t_evaluations_gpu)
        if (asyncTransferQ.joinable()) {
            double joinStart = omp_get_wtime();
            asyncTransferQ.join();
            std::cout <<" [computeT] Q join completed, waited: "<<omp_get_wtime() - joinStart<<std::endl;
        }
        
        double tstart = omp_get_wtime();
        compute_t_evaluations_gpu(
            (void*)buffersT,
            (void*)buffersTz,
            (const void*)evaluationsA->eval,
            (const void*)evaluationsB->eval,
            (const void*)evaluationsC->eval,
            (const void*)evaluationsZ->eval,
            d_staticEvalsBuffer,
            (const void*)evaluationsLagrange->eval,
            (const void*)buffers["PI"],
            (const void*)blindingFactors,
            (const void*)&beta,
            (const void*)&gamma,
            (const void*)&alpha,
            (const void*)&alpha2,
            (const void*)&k1,
            (const void*)&k2,
            (const void*)&omega_4x,
            (const void*)&omega1,
            (const void*)Z1,
            (const void*)Z2,
            (const void*)Z3,
            zkey->domainSize);
        std::cout<<" [computeT] computed T evaluations in "<<omp_get_wtime() - tstart<<std::endl;
        // Compute the coefficients of the polynomial T(X) from buffers.T using GPU IFFT
        LOG_TRACE("··· Computing T ifft (GPU)");
        polynomials["T"] = polynomialFromEvaluationsGPU(buffers["T"], polPtr["T"], zkey->domainSize * 4, 0, zkeyPower + 2);

        // Divide the polynomial T by Z_H(X)
        polynomials["T"]->divZh(zkey->domainSize, 4);

        // Compute the coefficients of the polynomial Tz(X) from buffers.Tz using GPU IFFT
        LOG_TRACE("··· Computing Tz ifft (GPU)");
        polynomials["Tz"] = polynomialFromEvaluationsGPU(buffers["Tz"], polPtr["Tz"], zkey->domainSize * 4, 0, zkeyPower + 2);

        // Add the polynomial Tz to T to get the final polynomial T
        polynomials["T"]->add(*polynomials["Tz"]);

        // Check degree
        if (polynomials["T"]->getDegree() >= 3 * zkey->domainSize + 6)
        {
            throw std::runtime_error("T Polynomial is not well calculated");
        }

        // t(x) has degree 3n + 5, we are going to split t(x) into three smaller polynomials:
        // T1' and T2'  with a degree < n and T3' with a degree n+5
        // such that t(x) = T1'(X) + X^n T2'(X) + X^{2n} T3'(X)
        // To randomize the parts we use blinding scalars b_10 and b_11 in a way that doesn't change t(X):
        // T1(X) = T1'(X) + b_10 X^n
        // T2(X) = T2'(X) - b_10 + b_11 X^n
        // T3(X) = T3'(X) - b_11
        // such that
        // t(X) = T1(X) + X^n T2(X) + X^2n T3(X)
        LOG_TRACE("··· Computing T1, T2, T3 polynomials");
        int nThreads = omp_get_max_threads() / 2;

        polynomials["T1"] = new Polynomial<Engine>(E, polPtr["T1"], zkey->domainSize, 1);
        ThreadUtils::parcpy(polynomials["T1"]->coef, &polynomials["T"]->coef[0], zkey->domainSize * sizeof(FrElement), nThreads);

        polynomials["T2"] = new Polynomial<Engine>(E, polPtr["T2"], zkey->domainSize, 1);
        ThreadUtils::parcpy(polynomials["T2"]->coef, &polynomials["T"]->coef[zkey->domainSize], zkey->domainSize * sizeof(FrElement), nThreads);

        polynomials["T3"] = new Polynomial<Engine>(E, polPtr["T3"], zkey->domainSize, 6);
        ThreadUtils::parcpy(polynomials["T3"]->coef, &polynomials["T"]->coef[zkey->domainSize * 2], (zkey->domainSize + 6) * sizeof(FrElement), nThreads);

        // Add blinding scalar b_10 as a new coefficient n
        polynomials["T1"]->coef[zkey->domainSize] = blindingFactors[10];

        // compute t_mid(X)
        // Subtract blinding scalar b_10 to the lowest coefficient of t_mid
        auto lowestMid = E.fr.sub(polynomials["T2"]->coef[0], blindingFactors[10]);
        polynomials["T2"]->coef[0] = lowestMid;
        polynomials["T2"]->coef[zkey->domainSize] = blindingFactors[11];

        // compute t_high(X)
        // Subtract blinding scalar b_11 to the lowest coefficient of t_high
        auto lowestHigh = E.fr.sub(polynomials["T3"]->coef[0], blindingFactors[11]);
        polynomials["T3"]->coef[0] = lowestHigh;

        polynomials["T1"]->fixDegree();
        polynomials["T2"]->fixDegree();
        polynomials["T3"]->fixDegree();
    }

    // ROUND 4
    template <typename Engine>
    void PlonkProverGPU<Engine>::round4()
    {
        LOG_TRACE("> Computing challenge xi");

        // STEP 4.1 - Compute challenge xi ∈ F
        transcript->reset();
        transcript->addScalar(challenges["alpha"]);
        transcript->addPolCommitment(proof->getPolynomialCommitment("T1"));
        transcript->addPolCommitment(proof->getPolynomialCommitment("T2"));
        transcript->addPolCommitment(proof->getPolynomialCommitment("T3"));

        challenges["xi"] = transcript->getChallenge();
        challenges["xiw"] = E.fr.mul(challenges["xi"], fft->root(zkeyPower, 1));

        std::ostringstream ss;
        ss << "··· challenges.xi: " << E.fr.toString(challenges["xi"]);
        LOG_TRACE(ss);

        // Fourth output of the prover is ( a(xi), b(xi), c(xi), s1(xi), s2(xi), z(xiw) )
        proof->addEvaluationCommitment("eval_a", polynomials["A"]->fastEvaluate(challenges["xi"]));
        proof->addEvaluationCommitment("eval_b", polynomials["B"]->fastEvaluate(challenges["xi"]));
        proof->addEvaluationCommitment("eval_c", polynomials["C"]->fastEvaluate(challenges["xi"]));
        proof->addEvaluationCommitment("eval_s1", polynomials["Sigma1"]->fastEvaluate(challenges["xi"]));
        proof->addEvaluationCommitment("eval_s2", polynomials["Sigma2"]->fastEvaluate(challenges["xi"]));
        proof->addEvaluationCommitment("eval_zw", polynomials["Z"]->fastEvaluate(challenges["xiw"]));
    }

    // ROUND 5
    template <typename Engine>
    void PlonkProverGPU<Engine>::round5()
    {
        // STEP 5.1 - Compute random evaluation point v ∈ F
        LOG_TRACE("> Computing challenge v");
        transcript->reset();
        transcript->addScalar(challenges["xi"]);
        transcript->addScalar(proof->getEvaluationCommitment("eval_a"));
        transcript->addScalar(proof->getEvaluationCommitment("eval_b"));
        transcript->addScalar(proof->getEvaluationCommitment("eval_c"));
        transcript->addScalar(proof->getEvaluationCommitment("eval_s1"));
        transcript->addScalar(proof->getEvaluationCommitment("eval_s2"));
        transcript->addScalar(proof->getEvaluationCommitment("eval_zw"));

        challenges["v1"] = transcript->getChallenge();

        std::ostringstream ss;
        ss << "··· challenges.v: " << E.fr.toString(challenges["v1"]);
        LOG_TRACE(ss);

        for (uint i = 2; i < 6; i++)
        {
            challenges["v" + std::to_string(i)] = E.fr.mul(challenges["v" + std::to_string(i - 1)], challenges["v1"]);
        }

        // STEP 5.2 Compute linearisation polynomial r(X)
        LOG_TRACE("> Computing linearisation polynomial R(X)");
        computeR();

        // STEP 5.3 Compute opening proof polynomial Wxi(X)
        LOG_TRACE("> Computing opening proof polynomial Wxi(X) polynomial");
        computeWxi();

        // STEP 5.4 Compute opening proof polynomial Wxiw(X)
        LOG_TRACE("> Computing opening proof polynomial Wxiw(X) polynomial");
        computeWxiw();

        // The fifth output of the prover is ([Wxi]_1, [Wxiw]_1)
        LOG_TRACE("> Computing Wxi, Wxiw polynomial commitments (GPU)");
        G1Point commitWxi = multiExponentiationGPU(polynomials["Wxi"]);
        G1Point commitWxiw = multiExponentiationGPU(polynomials["Wxiw"]);

        proof->addPolynomialCommitment("Wxi", commitWxi);
        proof->addPolynomialCommitment("Wxiw", commitWxiw);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeR()
    {
        challenges["xin"] = challenges["xi"];
        for (u_int32_t i = 0; i < zkeyPower; i++)
        {
            challenges["xin"] = E.fr.square(challenges["xin"]);
        }
        challenges["zh"] = E.fr.sub(challenges["xin"], E.fr.one());

        auto upper_bound = std::max(static_cast<uint32_t>(1), zkey->nPublic);
        auto L = new FrElement[upper_bound + 1];

        auto n = E.fr.set(zkey->domainSize);
        auto w = E.fr.one();
        for (u_int32_t i = 1; i <= upper_bound; i++)
        {
            E.fr.div(L[i], E.fr.mul(w, challenges["zh"]), E.fr.mul(n, E.fr.sub(challenges["xi"], w)));
            w = E.fr.mul(w, fft->root(zkeyPower, 1));
        }

        FrElement eval_l1;
        E.fr.div(eval_l1,
                 E.fr.sub(challenges["xin"], E.fr.one()),
                 E.fr.mul(n, E.fr.sub(challenges["xi"], E.fr.one())));

        LOG_TRACE("> Lagrange Evaluations:");
        std::ostringstream ss;

        for (u_int32_t i = 1; i <= upper_bound; i++)
        {
            ss.str("");
            ss << "··· L" << i << "(xi): " << E.fr.toString(L[i]);
            LOG_TRACE(ss);
        }

        auto eval_pi = E.fr.zero();
        
        auto publicSignals = new FrElement[zkey->nPublic];
        FrElement montgomery;
        for (u_int32_t i = 0; i < zkey->nPublic; i++)
        {
            E.fr.toMontgomery(montgomery, buffWitness[i + 1]);
            publicSignals[i] = montgomery;
        }

        for (u_int32_t i = 0; i < zkey->nPublic; i++)
        {
            const FrElement w = publicSignals[i];
            eval_pi = E.fr.sub(eval_pi, E.fr.mul(w, L[i + 1]));
        }

        ss.str("");
        ss << "··· PI: " << E.fr.toString(eval_pi);
        LOG_TRACE(ss);

        // Compute constant parts of R(X)
        auto coef_ab = E.fr.mul(proof->getEvaluationCommitment("eval_a"), proof->getEvaluationCommitment("eval_b"));

        auto e2a = proof->getEvaluationCommitment("eval_a");
        auto betaxi = E.fr.mul(challenges["beta"], challenges["xi"]);
        e2a = E.fr.add(e2a, betaxi);
        e2a = E.fr.add(e2a, challenges["gamma"]);

        auto e2b = proof->getEvaluationCommitment("eval_b");
        e2b = E.fr.add(e2b, E.fr.mul(betaxi, *((FrElement *)zkey->k1)));
        e2b = E.fr.add(e2b, challenges["gamma"]);

        auto e2c = proof->getEvaluationCommitment("eval_c");
        e2c = E.fr.add(e2c, E.fr.mul(betaxi, *((FrElement *)zkey->k2)));
        e2c = E.fr.add(e2c, challenges["gamma"]);

        auto e2 = E.fr.mul(E.fr.mul(E.fr.mul(e2a, e2b), e2c), challenges["alpha"]);

        auto e3a = proof->getEvaluationCommitment("eval_a");
        e3a = E.fr.add(e3a, E.fr.mul(challenges["beta"], proof->getEvaluationCommitment("eval_s1")));
        e3a = E.fr.add(e3a, challenges["gamma"]);

        auto e3b = proof->getEvaluationCommitment("eval_b");
        e3b = E.fr.add(e3b, E.fr.mul(challenges["beta"], proof->getEvaluationCommitment("eval_s2")));
        e3b = E.fr.add(e3b, challenges["gamma"]);

        auto e3 = E.fr.mul(e3a, e3b);
        e3 = E.fr.mul(e3, proof->getEvaluationCommitment("eval_zw"));
        e3 = E.fr.mul(e3, challenges["alpha"]);

        auto e4 = E.fr.mul(eval_l1, challenges["alpha2"]);

        polynomials["R"] = new Polynomial<Engine>(E, polPtr["R"], zkey->domainSize + 6);
        polynomials["R"]->addBlinding(*polynomials["QM"], coef_ab);
        auto eval_a = proof->getEvaluationCommitment("eval_a");
        polynomials["R"]->addBlinding(*polynomials["QL"], eval_a);
        auto eval_b = proof->getEvaluationCommitment("eval_b");
        polynomials["R"]->addBlinding(*polynomials["QR"], eval_b);
        auto eval_c = proof->getEvaluationCommitment("eval_c");
        polynomials["R"]->addBlinding(*polynomials["QO"], eval_c);
        polynomials["R"]->add(*polynomials["QC"]);
        polynomials["R"]->addBlinding(*polynomials["Z"], e2);
        auto val = E.fr.mul(e3, challenges["beta"]);
        polynomials["R"]->subBlinding(*polynomials["Sigma3"], val);
        polynomials["R"]->addBlinding(*polynomials["Z"], e4);

        auto tmp = polynomials["T3"];
        auto xin2 = E.fr.square(challenges["xin"]);
        tmp->mulScalar(xin2);
        tmp->addBlinding(*polynomials["T2"], challenges["xin"]);
        tmp->add(*polynomials["T1"]);
        tmp->mulScalar(challenges["zh"]);

        polynomials["R"]->sub(*tmp);

        auto r0 = E.fr.sub(eval_pi, E.fr.mul(e3, E.fr.add(proof->getEvaluationCommitment("eval_c"), challenges["gamma"])));
        r0 = E.fr.sub(r0, e4);

        ss.str("");
        ss << "··· r0: " << E.fr.toString(r0);
        LOG_TRACE(ss);

        polynomials["R"]->addScalar(r0);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWxi()
    {
        polynomials["Wxi"] = new Polynomial<Engine>(E, polPtr["Wxi"], zkey->domainSize + 6);
        polynomials["Wxi"]->add(*polynomials["R"]);
        polynomials["Wxi"]->addBlinding(*polynomials["A"], challenges["v1"]);
        polynomials["Wxi"]->addBlinding(*polynomials["B"], challenges["v2"]);
        polynomials["Wxi"]->addBlinding(*polynomials["C"], challenges["v3"]);
        polynomials["Wxi"]->addBlinding(*polynomials["Sigma1"], challenges["v4"]);
        polynomials["Wxi"]->addBlinding(*polynomials["Sigma2"], challenges["v5"]);

        auto eval_a = proof->getEvaluationCommitment("eval_a");
        auto eval_b = proof->getEvaluationCommitment("eval_b");
        auto eval_c = proof->getEvaluationCommitment("eval_c");
        auto eval_s1 = proof->getEvaluationCommitment("eval_s1");
        auto eval_s2 = proof->getEvaluationCommitment("eval_s2");

        auto val = E.fr.mul(challenges["v1"], eval_a);
        polynomials["Wxi"]->subScalar(val);
        val = E.fr.mul(challenges["v2"], eval_b);
        polynomials["Wxi"]->subScalar(val);
        val = E.fr.mul(challenges["v3"], eval_c);
        polynomials["Wxi"]->subScalar(val);
        val = E.fr.mul(challenges["v4"], eval_s1);
        polynomials["Wxi"]->subScalar(val);
        val = E.fr.mul(challenges["v5"], eval_s2);
        polynomials["Wxi"]->subScalar(val);

        polynomials["Wxi"]->divByZerofier(1, challenges["xi"]);
        polynomials["Wxi"]->fixDegree();
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWxiw()
    {
        polynomials["Wxiw"] = polynomials["Z"];
        auto val = proof->getEvaluationCommitment("eval_zw");
        polynomials["Wxiw"]->subScalar(val);

        polynomials["Wxiw"]->divByZerofier(1, challenges["xiw"]);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::batchInverse(FrElement *elements, u_int64_t length)
    {
        // Calculate products: a, ab, abc, abcd, ...
        products[0] = elements[0];
        for (u_int64_t index = 1; index < length; index++)
        {
            E.fr.mul(products[index], products[index - 1], elements[index]);
        }

        // Calculate inverses: 1/a, 1/ab, 1/abc, 1/abcd, ...
        E.fr.inv(inverses[length - 1], products[length - 1]);
        for (uint64_t index = length - 1; index > 0; index--)
        {
            E.fr.mul(inverses[index - 1], inverses[index], elements[index]);
        }

        elements[0] = inverses[0];
        for (u_int64_t index = 1; index < length; index++)
        {
            E.fr.mul(elements[index], inverses[index], products[index - 1]);
        }
    }

    template <typename Engine>
    typename Engine::FrElement *PlonkProverGPU<Engine>::polynomialFromMontgomery(Polynomial<Engine> *polynomial)
    {
        const u_int64_t length = polynomial->getLength();

        FrElement *result = buffers["tmp"];
        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parset(result, 0, length * sizeof(FrElement), nThreads);

#pragma omp parallel for
        for (u_int32_t index = 0; index < length; ++index)
        {
            E.fr.fromMontgomery(result[index], polynomial->coef[index]);
        }

        return result;
    }

    template <typename Engine>
    typename Engine::G1Point PlonkProverGPU<Engine>::multiExponentiationGPU(Polynomial<Engine> *polynomial)
    {
        G1Point value;
        FrElement *pol = this->polynomialFromMontgomery(polynomial);

        // GPU MSM returns standard Jacobian {X, Y, Z}
        // CPU expects Extended Jacobian {x, y, zz, zzz} where zz=Z², zzz=Z³
        struct JacobianPoint {
            typename Engine::F1Element X;
            typename Engine::F1Element Y;
            typename Engine::F1Element Z;
        };
        JacobianPoint gpuResult;

        msm_bn128_gpu(&gpuResult, PTau, pol, polynomial->getDegree() + 1, false);

        // Convert from standard Jacobian to Extended Jacobian
        // x = X, y = Y, zz = Z², zzz = Z³
        value.x = gpuResult.X;
        value.y = gpuResult.Y;
        E.f1.square(value.zz, gpuResult.Z);           // zz = Z²
        E.f1.mul(value.zzz, value.zz, gpuResult.Z);   // zzz = Z³

        return value;
    }

    // GPU-accelerated polynomial construction from evaluations using GPU IFFT
    template <typename Engine>
    Polynomial<Engine>* PlonkProverGPU<Engine>::polynomialFromEvaluationsGPU(
        FrElement *evaluations, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength, u_int32_t power)
    {
        Polynomial<Engine> *pol = new Polynomial<Engine>(E, reservedBuffer, length, blindLength);

        int nThreads = omp_get_max_threads() / 2;
        ThreadUtils::parcpy(pol->coef, evaluations, length * sizeof(FrElement), nThreads);

        // Use GPU IFFT instead of CPU fft->ifft
        intt_bn128_gpu(pol->coef, power);

        pol->fixDegree();

        return pol;
    }

    // GPU-accelerated evaluations construction from polynomial using GPU FFT
    template <typename Engine>
    Evaluations<Engine>* PlonkProverGPU<Engine>::evaluationsFromPolynomialGPU(
        FrElement *reservedBuffer, Polynomial<Engine> &polynomial, u_int32_t extensionLength, u_int32_t power)
    {
        Evaluations<Engine> *evals = new Evaluations<Engine>(E, reservedBuffer, extensionLength);

        u_int64_t polLength = polynomial.getLength();
        int nThreads = omp_get_max_threads() / 2;

        // Copy polynomial coefficients to evaluation buffer
        ThreadUtils::parcpy(evals->eval, polynomial.coef, polLength * sizeof(FrElement), nThreads);

        // Zero-pad the rest
        if (extensionLength > polLength) {
            ThreadUtils::parset(evals->eval + polLength, 0, (extensionLength - polLength) * sizeof(FrElement), nThreads);
        }

        // Use GPU FFT instead of CPU fft->fft
        ntt_bn128_gpu(evals->eval, power);

        return evals;
    }
}
